use crate::{ByteForgeConfig, Result};
use crate::patching::Patch;
use ndarray::{Array1, Array2, Array3, Axis, s};

#[derive(Debug, Clone)]
pub struct ByteForgeTransformer {
    config: ByteForgeConfig,
    local_encoder: LocalEncoder,
    global_transformer: GlobalTransformer,
    local_decoder: LocalDecoder,
    patch_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
}

impl ByteForgeTransformer {
    pub fn new(config: ByteForgeConfig) -> Result<Self> {
        let local_encoder = LocalEncoder::new(&config)?;
        let global_transformer = GlobalTransformer::new(&config)?;
        let local_decoder = LocalDecoder::new(&config)?;

        let patch_embeddings = Array2::zeros((config.vocab_size, config.model_dim));
        let position_embeddings = Array2::zeros((config.max_seq_len, config.model_dim));

        Ok(Self {
            config,
            local_encoder,
            global_transformer,
            local_decoder,
            patch_embeddings,
            position_embeddings,
        })
    }

    pub fn forward(&mut self, patches: &[Patch]) -> Result<Array3<f32>> {
        let encoded_patches = self.local_encoder.encode_patches(patches)?;
        let global_output = self.global_transformer.forward(&encoded_patches)?;
        let decoded_output = self.local_decoder.decode_patches(&global_output, patches)?;
        
        Ok(decoded_output)
    }

    pub fn forward_streaming(&mut self, patch: &Patch, context: &Array3<f32>) -> Result<Array2<f32>> {
        let encoded_patch = self.local_encoder.encode_single_patch(patch)?;
        let global_output = self.global_transformer.forward_streaming(&encoded_patch, context)?;
        let decoded_output = self.local_decoder.decode_single_patch(&global_output, patch)?;
        
        Ok(decoded_output)
    }
}

#[derive(Debug, Clone)]
pub struct LocalEncoder {
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
    dropout: f32,
}

impl LocalEncoder {
    pub fn new(config: &ByteForgeConfig) -> Result<Self> {
        let mut layers = Vec::new();
        let hidden_dim = config.model_dim / 4;
        for _ in 0..3 {
            layers.push(EncoderLayer::new(hidden_dim, (config.num_heads / 2).max(1))?);
        }

        Ok(Self {
            layers,
            norm: LayerNorm::new(hidden_dim),
            dropout: 0.1,
        })
    }

    pub fn encode_patches(&mut self, patches: &[Patch]) -> Result<Array3<f32>> {
        let batch_size = patches.len();
        let max_patch_len = patches.iter().map(|p| p.bytes.len()).max().unwrap_or(0);
        let hidden_dim = self.layers[0].hidden_dim;

        let mut output = Array3::zeros((batch_size, max_patch_len, hidden_dim));

        for (batch_idx, patch) in patches.iter().enumerate() {
            let patch_output = self.encode_single_patch(patch)?;
            
            let actual_len = patch_output.shape()[0].min(max_patch_len);
            output.slice_mut(s![batch_idx, ..actual_len, ..])
                .assign(&patch_output.slice(s![..actual_len, ..]));
        }

        Ok(output)
    }

    pub fn encode_single_patch(&mut self, patch: &Patch) -> Result<Array2<f32>> {
        let mut embeddings = self.create_byte_embeddings(&patch.bytes)?;
        embeddings = self.add_ngram_embeddings(embeddings, &patch.bytes)?;
        embeddings = self.add_complexity_embeddings(embeddings, patch.complexity_score)?;

        let mut hidden = embeddings;
        for layer in &mut self.layers {
            hidden = layer.forward(hidden)?;
        }

        hidden = self.norm.forward(hidden)?;
        Ok(hidden)
    }

    fn create_byte_embeddings(&self, bytes: &[u8]) -> Result<Array2<f32>> {
        let seq_len = bytes.len();
        let embed_dim = self.layers[0].hidden_dim;
        let mut embeddings = Array2::zeros((seq_len, embed_dim));

        for (i, &byte) in bytes.iter().enumerate() {
            let mut embedding = Array1::zeros(embed_dim);
            // TODO: Implement actual learned embedding lMAO 
            for j in 0..embed_dim {
                embedding[j] = ((byte as f32 * (j + 1) as f32).sin() * 0.1) as f32;
            }
            
            embeddings.slice_mut(s![i, ..]).assign(&embedding);
        }

        Ok(embeddings)
    }

    fn add_ngram_embeddings(&self, mut embeddings: Array2<f32>, bytes: &[u8]) -> Result<Array2<f32>> {
        let seq_len = bytes.len();
        
        for i in 0..seq_len {
            for n in 2..=4 {
                if i + n <= seq_len {
                    let ngram = &bytes[i..i + n];
                    let ngram_embedding = self.compute_ngram_embedding(ngram)?;
                    
                    let mut current = embeddings.slice_mut(s![i, ..]);
                    current += &ngram_embedding;
                }
            }
        }

        Ok(embeddings)
    }

    fn compute_ngram_embedding(&self, ngram: &[u8]) -> Result<Array1<f32>> {
        let embed_dim = self.layers[0].hidden_dim;
        let mut embedding = Array1::zeros(embed_dim);
        
        let hash = self.hash_ngram(ngram);
        
        for i in 0..embed_dim {
            embedding[i] = ((hash as f32 * (i + 1) as f32).sin() * 0.05) as f32;
        }

        Ok(embedding)
    }

    fn hash_ngram(&self, ngram: &[u8]) -> u64 {
        let mut hash = 0u64;
        for &byte in ngram {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    fn add_complexity_embeddings(&self, mut embeddings: Array2<f32>, complexity: f32) -> Result<Array2<f32>> {
        let complexity_factor = complexity.tanh() * 0.1;
        embeddings *= complexity_factor + 1.0;
        Ok(embeddings)
    }
}

#[derive(Debug, Clone)]
pub struct GlobalTransformer {
    layers: Vec<TransformerLayer>,
    norm: LayerNorm,
    adaptive_computation: AdaptiveComputation,
}

impl GlobalTransformer {
    pub fn new(config: &ByteForgeConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(config.model_dim, config.num_heads)?);
        }

        Ok(Self {
            layers,
            norm: LayerNorm::new(config.model_dim),
            adaptive_computation: AdaptiveComputation::new(config.model_dim),
        })
    }

    pub fn forward(&mut self, input: &Array3<f32>) -> Result<Array3<f32>> {
        let mut hidden = input.clone();
        
        for layer in &mut self.layers {
            let complexity_scores = self.adaptive_computation.compute_complexity_scores(&hidden)?;
            
            if complexity_scores.iter().any(|&s| s > 0.5) {
                hidden = layer.forward_full(hidden)?;
            } else {
                hidden = layer.forward_efficient(hidden)?;
            }
        }

        hidden = self.norm.forward_3d(hidden)?;
        Ok(hidden)
    }

    pub fn forward_streaming(&mut self, input: &Array2<f32>, context: &Array3<f32>) -> Result<Array2<f32>> {
        let mut hidden = input.clone();
        
        for layer in &mut self.layers {
            hidden = layer.forward_streaming(hidden, context)?;
        }

        hidden = self.norm.forward(hidden)?;
        Ok(hidden)
    }
}

#[derive(Debug, Clone)]
pub struct LocalDecoder {
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    output_projection: Array2<f32>,
}

impl LocalDecoder {
    pub fn new(config: &ByteForgeConfig) -> Result<Self> {
        let mut layers = Vec::new();
        let hidden_dim = config.model_dim / 4;
        for _ in 0..4 {
            layers.push(DecoderLayer::new(hidden_dim, (config.num_heads / 2).max(1))?);
        }

        let output_projection = Array2::zeros((hidden_dim, 256));

        Ok(Self {
            layers,
            norm: LayerNorm::new(hidden_dim),
            output_projection,
        })
    }

    pub fn decode_patches(&mut self, global_output: &Array3<f32>, patches: &[Patch]) -> Result<Array3<f32>> {
        let batch_size = global_output.shape()[0];
        let max_patch_len = global_output.shape()[1];
        let output_dim = 256; // WARNING: if you don't know what this is.... nvm lol

        let mut output = Array3::zeros((batch_size, max_patch_len, output_dim));

        for batch_idx in 0..batch_size {
            let patch_output = self.decode_single_patch(
                &global_output.slice(s![batch_idx, .., ..]).to_owned(),
                &patches[batch_idx]
            )?;
            
            let actual_len = patch_output.shape()[0].min(max_patch_len);
            output.slice_mut(s![batch_idx, ..actual_len, ..])
                .assign(&patch_output.slice(s![..actual_len, ..]));
        }

        Ok(output)
    }

    pub fn decode_single_patch(&mut self, global_output: &Array2<f32>, patch: &Patch) -> Result<Array2<f32>> {
        let mut hidden = self.downsample_global_output(global_output)?;
        
        for layer in &mut self.layers {
            hidden = layer.forward(hidden)?;
        }

        hidden = self.norm.forward(hidden)?;
        let output = self.project_to_bytes(hidden)?;
        
        Ok(output)
    }

    fn downsample_global_output(&self, global_output: &Array2<f32>) -> Result<Array2<f32>> {
        let seq_len = global_output.shape()[0];
        let global_dim = global_output.shape()[1];
        let local_dim = self.layers[0].hidden_dim;

        let mut downsampled = Array2::zeros((seq_len, local_dim));
        
        for i in 0..seq_len {
            for j in 0..local_dim {
                let start_idx = (j * global_dim) / local_dim;
                let end_idx = ((j + 1) * global_dim) / local_dim;
                let mut sum = 0.0;
                let mut count = 0;
                
                for k in start_idx..end_idx.min(global_dim) {
                    sum += global_output[[i, k]];
                    count += 1;
                }
                
                downsampled[[i, j]] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }

        Ok(downsampled)
    }

    fn project_to_bytes(&self, hidden: Array2<f32>) -> Result<Array2<f32>> {
        let seq_len = hidden.shape()[0];
        let output_dim = self.output_projection.shape()[1];
        let mut output = Array2::zeros((seq_len, output_dim));

        for i in 0..seq_len {
            let hidden_vec = hidden.slice(s![i, ..]);
            let output_vec = hidden_vec.dot(&self.output_projection);
            output.slice_mut(s![i, ..]).assign(&output_vec);
        }

        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct EncoderLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    hidden_dim: usize,
}

impl EncoderLayer {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new(hidden_dim, num_heads)?,
            feed_forward: FeedForward::new(hidden_dim, hidden_dim * 4)?,
            norm1: LayerNorm::new(hidden_dim),
            norm2: LayerNorm::new(hidden_dim),
            hidden_dim,
        })
    }

    pub fn forward(&mut self, input: Array2<f32>) -> Result<Array2<f32>> {
        let normed_input = self.norm1.forward(input.clone())?;
        let attention_output = self.self_attention.forward(&normed_input, &normed_input, &normed_input)?;
        let residual1 = input + attention_output;

        let normed_residual = self.norm2.forward(residual1.clone())?;
        let ff_output = self.feed_forward.forward(normed_residual)?;
        let residual2 = residual1 + ff_output;

        Ok(residual2)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    efficient_attention: EfficientAttention,
}

impl TransformerLayer {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new(hidden_dim, num_heads)?,
            feed_forward: FeedForward::new(hidden_dim, hidden_dim * 4)?,
            norm1: LayerNorm::new(hidden_dim),
            norm2: LayerNorm::new(hidden_dim),
            efficient_attention: EfficientAttention::new(hidden_dim, num_heads)?,
        })
    }

    pub fn forward_full(&mut self, input: Array3<f32>) -> Result<Array3<f32>> {
        let mut output = Array3::zeros(input.dim());
        
        for i in 0..input.shape()[0] {
            let batch_input = input.slice(s![i, .., ..]).to_owned();
            let normed_input = self.norm1.forward(batch_input.clone())?;
            let attention_output = self.self_attention.forward(&normed_input, &normed_input, &normed_input)?;
            let residual1 = batch_input + attention_output;

            let normed_residual = self.norm2.forward(residual1.clone())?;
            let ff_output = self.feed_forward.forward(normed_residual)?;
            let residual2 = residual1 + ff_output;

            output.slice_mut(s![i, .., ..]).assign(&residual2);
        }

        Ok(output)
    }

    pub fn forward_efficient(&mut self, input: Array3<f32>) -> Result<Array3<f32>> {
        let mut output = Array3::zeros(input.dim());
        
        for i in 0..input.shape()[0] {
            let batch_input = input.slice(s![i, .., ..]).to_owned();
            let normed_input = self.norm1.forward(batch_input.clone())?;
            let attention_output = self.efficient_attention.forward(&normed_input)?;
            let residual1 = batch_input + attention_output;

            let normed_residual = self.norm2.forward(residual1.clone())?;
            let ff_output = self.feed_forward.forward_efficient(normed_residual)?;
            let residual2 = residual1 + ff_output;

            output.slice_mut(s![i, .., ..]).assign(&residual2);
        }

        Ok(output)
    }

    pub fn forward_streaming(&mut self, input: Array2<f32>, context: &Array3<f32>) -> Result<Array2<f32>> {
        let normed_input = self.norm1.forward(input.clone())?;
        let attention_output = self.self_attention.forward_streaming(&normed_input, context)?;
        let residual1 = input + attention_output;

        let normed_residual = self.norm2.forward(residual1.clone())?;
        let ff_output = self.feed_forward.forward(normed_residual)?;
        let residual2 = residual1 + ff_output;

        Ok(residual2)
    }
}

#[derive(Debug, Clone)]
pub struct DecoderLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    hidden_dim: usize,
}

impl DecoderLayer {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new(hidden_dim, num_heads)?,
            feed_forward: FeedForward::new(hidden_dim, hidden_dim * 4)?,
            norm1: LayerNorm::new(hidden_dim),
            norm2: LayerNorm::new(hidden_dim),
            hidden_dim,
        })
    }

    pub fn forward(&mut self, input: Array2<f32>) -> Result<Array2<f32>> {
        let normed_input = self.norm1.forward(input.clone())?;
        let attention_output = self.self_attention.forward(&normed_input, &normed_input, &normed_input)?;
        let residual1 = input + attention_output;

        let normed_residual = self.norm2.forward(residual1.clone())?;
        let ff_output = self.feed_forward.forward(normed_residual)?;
        let residual2 = residual1 + ff_output;

        Ok(residual2)
    }
}


#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
}

impl MultiHeadAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;
        
        Ok(Self {
            num_heads,
            head_dim,
            hidden_dim,
            w_q: Array2::zeros((hidden_dim, hidden_dim)),
            w_k: Array2::zeros((hidden_dim, hidden_dim)),
            w_v: Array2::zeros((hidden_dim, hidden_dim)),
            w_o: Array2::zeros((hidden_dim, hidden_dim)),
        })
    }

    pub fn forward(&self, query: &Array2<f32>, key: &Array2<f32>, value: &Array2<f32>) -> Result<Array2<f32>> {
        let seq_len = query.shape()[0];
        let mut output = Array2::zeros((seq_len, self.hidden_dim));

        let q = query.dot(&self.w_q);
        let k = key.dot(&self.w_k);
        let v = value.dot(&self.w_v);

        let scores = q.dot(&k.t()) / (self.head_dim as f32).sqrt();
        let attention_weights = softmax_2d(&scores)?;
        let attention_output = attention_weights.dot(&v);

        output = attention_output.dot(&self.w_o);
        Ok(output)
    }

    pub fn forward_streaming(&self, query: &Array2<f32>, context: &Array3<f32>) -> Result<Array2<f32>> {
        // TODO: Implement actual streaming attention not really sure if this is correct but it works for now ngl
        self.forward(query, query, query)
    }
}

#[derive(Debug, Clone)]
pub struct EfficientAttention {
    hidden_dim: usize,
    num_heads: usize,
    linear_attention: LinearAttention,
}

impl EfficientAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            hidden_dim,
            num_heads,
            linear_attention: LinearAttention::new(hidden_dim)?,
        })
    }

    pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.linear_attention.forward(input)
    }
}

#[derive(Debug, Clone)]
pub struct LinearAttention {
    hidden_dim: usize,
    projection: Array2<f32>,
}

impl LinearAttention {
    pub fn new(hidden_dim: usize) -> Result<Self> {
        Ok(Self {
            hidden_dim,
            projection: Array2::zeros((hidden_dim, hidden_dim)),
        })
    }

    pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
      
        let output = input.dot(&self.projection);
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct FeedForward {
    w1: Array2<f32>,
    w2: Array2<f32>,
    hidden_dim: usize,
    intermediate_dim: usize,
}

impl FeedForward {
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Result<Self> {
        Ok(Self {
            w1: Array2::zeros((hidden_dim, intermediate_dim)),
            w2: Array2::zeros((intermediate_dim, hidden_dim)),
            hidden_dim,
            intermediate_dim,
        })
    }

    pub fn forward(&self, input: Array2<f32>) -> Result<Array2<f32>> {
        let intermediate = input.dot(&self.w1);
        let activated = relu_2d(&intermediate)?;
        let output = activated.dot(&self.w2);
        Ok(output)
    }

    pub fn forward_efficient(&self, input: Array2<f32>) -> Result<Array2<f32>> {
        let intermediate = input.dot(&self.w1);
        let activated = relu_2d(&intermediate)?;
        let output = activated.dot(&self.w2) * 0.5;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    hidden_dim: usize,
    eps: f32,
}

impl LayerNorm {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: Array2<f32>) -> Result<Array2<f32>> {
        let mut output = input.clone();
        
        for mut row in output.axis_iter_mut(Axis(0)) {
            let mean = row.mean().unwrap();
            let variance = row.map(|x| (x - mean).powi(2)).mean().unwrap();
            let std = (variance + self.eps).sqrt();
            
            row.mapv_inplace(|x| (x - mean) / std);
        }

        Ok(output)
    }

    pub fn forward_3d(&self, input: Array3<f32>) -> Result<Array3<f32>> {
        let mut output = input.clone();
        
        for mut batch in output.axis_iter_mut(Axis(0)) {
            for mut row in batch.axis_iter_mut(Axis(0)) {
                let mean = row.mean().unwrap();
                let variance = row.map(|x| (x - mean).powi(2)).mean().unwrap();
                let std = (variance + self.eps).sqrt();
                
                row.mapv_inplace(|x| (x - mean) / std);
            }
        }

        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveComputation {
    hidden_dim: usize,
    complexity_predictor: Array2<f32>,
}

impl AdaptiveComputation {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            complexity_predictor: Array2::zeros((hidden_dim, 1)),
        }
    }

    pub fn compute_complexity_scores(&self, input: &Array3<f32>) -> Result<Vec<f32>> {
        let batch_size = input.shape()[0];
        let mut scores = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let batch_input = input.slice(s![i, .., ..]);
            let mean_activation = batch_input.mean().unwrap();
            let complexity = (mean_activation.abs() * 2.0).tanh();
            scores.push(complexity);
        }

        Ok(scores)
    }
}


fn softmax_2d(input: &Array2<f32>) -> Result<Array2<f32>> {
    let mut output = input.clone();
    
    for mut row in output.axis_iter_mut(Axis(0)) {
        let max_val = row.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        row.mapv_inplace(|x| (x - max_val).exp());
        let sum = row.sum();
        if sum > 0.0 {
            row.mapv_inplace(|x| x / sum);
        }
    }

    Ok(output)
}

fn relu_2d(input: &Array2<f32>) -> Result<Array2<f32>> {
    Ok(input.mapv(|x| x.max(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patching::PatchType;

    #[test]
    fn test_transformer_creation() {
        let config = ByteForgeConfig::default();
        let transformer = ByteForgeTransformer::new(config).unwrap();
        assert_eq!(transformer.config.model_dim, 512);
    }

    #[test]
    fn test_local_encoder() {
        let config = ByteForgeConfig::default();
        let mut encoder = LocalEncoder::new(&config).unwrap();
        
        let patch = Patch {
            bytes: b"hello".to_vec(),
            start_pos: 0,
            end_pos: 5,
            complexity_score: 0.5,
            patch_type: PatchType::Simple,
        };

        let output = encoder.encode_single_patch(&patch).unwrap();
        assert_eq!(output.shape()[0], 5);
        assert!(output.shape()[1] > 0);
    }
} 