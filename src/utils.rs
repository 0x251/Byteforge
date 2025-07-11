use crate::Result;
use std::collections::HashMap;
use rayon::prelude::*;

pub struct PerformanceMetrics {
    pub patches_per_second: f64,
    pub bytes_per_second: f64,
    pub average_patch_size: f32,
    pub compression_ratio: f32,
    pub entropy_calculation_time: std::time::Duration,
    pub patching_time: std::time::Duration,
    pub transformer_time: std::time::Duration,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            patches_per_second: 0.0,
            bytes_per_second: 0.0,
            average_patch_size: 0.0,
            compression_ratio: 0.0,
            entropy_calculation_time: std::time::Duration::from_secs(0),
            patching_time: std::time::Duration::from_secs(0),
            transformer_time: std::time::Duration::from_secs(0),
        }
    }

    pub fn print_report(&self) {
        println!("📊 Performance Report:");
        println!("  Patches/sec: {:.2}", self.patches_per_second);
        println!("  Bytes/sec: {:.2}", self.bytes_per_second);
        println!("  Avg patch size: {:.1} bytes", self.average_patch_size);
        println!("  Compression ratio: {:.2}x", self.compression_ratio);
        println!("  Entropy calc time: {:?}", self.entropy_calculation_time);
        println!("  Patching time: {:?}", self.patching_time);
        println!("  Transformer time: {:?}", self.transformer_time);
    }
}

pub fn calculate_compression_ratio(original_size: usize, compressed_size: usize) -> f32 {
    if compressed_size == 0 {
        return 1.0;
    }
    original_size as f32 / compressed_size as f32
}

pub fn benchmark_patching_strategies() -> Result<()> {
    println!("🏃 Benchmarking different patching strategies...");
    
    let test_texts = vec![
        "Simple text with basic words.",
        "Code: fn main() { println!(\"Hello\"); }",
        "JSON: {\"complex\": true, \"nested\": {\"value\": 42}}",
        "Repetitive text with repeated patterns repeated patterns repeated patterns",
        "Mixed content: code, text, numbers 123456789, symbols !@#$%^&*()",
    ];

    for (i, text) in test_texts.iter().enumerate() {
        println!("\nText {}: {}", i + 1, text);
        
        
        let fixed_patches = create_fixed_patches(text.as_bytes(), 4);
        println!("  Fixed (4-byte): {} patches", fixed_patches.len());
        
   
        let space_patches = create_space_patches(text.as_bytes());
        println!("  Whitespace: {} patches", space_patches.len());
        
     
        println!("  Multi-signal: {} patches (estimated)", 
                 estimate_multisignal_patches(text.as_bytes()));
    }

    Ok(())
}

fn create_fixed_patches(bytes: &[u8], patch_size: usize) -> Vec<Vec<u8>> {
    bytes.chunks(patch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn create_space_patches(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut patches = Vec::new();
    let mut current_patch = Vec::new();
    
    for &byte in bytes {
        current_patch.push(byte);
        
        if byte == b' ' || byte == b'\n' || byte == b'\t' {
            if !current_patch.is_empty() {
                patches.push(current_patch.clone());
                current_patch.clear();
            }
        }
    }
    
    if !current_patch.is_empty() {
        patches.push(current_patch);
    }
    
    patches
}

fn estimate_multisignal_patches(bytes: &[u8]) -> usize {
    
    let mut patch_count = 0;
    let mut current_patch_size = 0;
    
    for i in 0..bytes.len() {
        current_patch_size += 1;
        
      
        let is_boundary = is_likely_boundary(bytes, i);
        
        if is_boundary && current_patch_size >= 2 {
            patch_count += 1;
            current_patch_size = 0;
        } else if current_patch_size >= 12 {
            patch_count += 1;
            current_patch_size = 0;
        }
    }
    
    if current_patch_size > 0 {
        patch_count += 1;
    }
    
    patch_count
}

fn is_likely_boundary(bytes: &[u8], pos: usize) -> bool {
    if pos == 0 || pos >= bytes.len() - 1 {
        return false;
    }
    
    let current = bytes[pos];
    let prev = bytes[pos - 1];
    

    let word_boundary = (prev.is_ascii_alphabetic() && !current.is_ascii_alphabetic()) ||
                       (!prev.is_ascii_alphabetic() && current.is_ascii_alphabetic());
    
 
    let structural_boundary = matches!(current, b'{' | b'}' | b'[' | b']' | b'(' | b')' | b';' | b':');

    let sentence_boundary = matches!(prev, b'.' | b'!' | b'?') && current == b' ';
    
    word_boundary || structural_boundary || sentence_boundary
}

pub fn compare_with_blt_performance(our_patches: usize, text_length: usize) -> f32 {
    let blt_average_patch_size = 4.5; // From the paper
    let blt_patches = (text_length as f32 / blt_average_patch_size).ceil() as usize;
    
    if our_patches == 0 {
        return 1.0;
    }
    
    blt_patches as f32 / our_patches as f32
}

pub fn memory_usage_estimate(
    num_patches: usize,
    avg_patch_size: f32,
    model_dim: usize,
    sequence_length: usize,
) -> usize {
    let patch_storage = (num_patches as f32 * avg_patch_size) as usize;
    let embeddings_storage = num_patches * model_dim * 4; // f32 = 4 bytes
    let attention_storage = sequence_length * sequence_length * 4; // simplified
    let intermediate_storage = num_patches * model_dim * 4 * 4; // 4x expansion in FFN
    
    patch_storage + embeddings_storage + attention_storage + intermediate_storage
}

pub fn estimate_inference_speed(
    patches_per_sequence: usize,
    model_dim: usize,
    num_layers: usize,
    hardware_flops_per_second: f64,
) -> f64 {
    // Simplified FLOP estimation
    let attention_flops = patches_per_sequence * patches_per_sequence * model_dim * 4;
    let ffn_flops = patches_per_sequence * model_dim * model_dim * 8;
    let total_flops_per_layer = attention_flops + ffn_flops;
    let total_flops = total_flops_per_layer * num_layers;
    
    hardware_flops_per_second / total_flops as f64
}

pub fn analyze_text_complexity(text: &str) -> TextComplexity {
    let bytes = text.as_bytes();
    let mut char_frequencies = HashMap::new();
    let mut entropy = 0.0;
    
    // Calculate character frequency
    for &byte in bytes {
        *char_frequencies.entry(byte).or_insert(0) += 1;
    }
    
    // Calculate entropy
    let total_chars = bytes.len() as f64;
    for &count in char_frequencies.values() {
        let probability = count as f64 / total_chars;
        entropy -= probability * probability.log2();
    }
    
    // Analyze patterns
    let repetition_score = analyze_repetition_patterns(bytes);
    let structural_score = analyze_structural_complexity(bytes);
    let semantic_score = analyze_semantic_complexity(text);
    
    TextComplexity {
        entropy,
        repetition_score,
        structural_score,
        semantic_score,
        unique_chars: char_frequencies.len(),
        total_chars: bytes.len(),
    }
}

fn analyze_repetition_patterns(bytes: &[u8]) -> f32 {
    let mut pattern_counts = HashMap::new();
    let mut total_patterns = 0;
    

    for window_size in 2..=3 {
        for i in 0..bytes.len().saturating_sub(window_size - 1) {
            let pattern = &bytes[i..i + window_size];
            *pattern_counts.entry(pattern.to_vec()).or_insert(0) += 1;
            total_patterns += 1;
        }
    }
    
    let repeated_patterns = pattern_counts.values().filter(|&&count| count > 1).count();
    if total_patterns > 0 {
        repeated_patterns as f32 / total_patterns as f32
    } else {
        0.0
    }
}

fn analyze_structural_complexity(bytes: &[u8]) -> f32 {
    let structural_chars = [b'{', b'}', b'[', b']', b'(', b')', b'<', b'>', b';', b':', b','];
    let structural_count = bytes.iter().filter(|&b| structural_chars.contains(b)).count();
    
    if bytes.len() > 0 {
        structural_count as f32 / bytes.len() as f32
    } else {
        0.0
    }
}

fn analyze_semantic_complexity(text: &str) -> f32 {
    let words: Vec<&str> = text.split_whitespace().collect();
    let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
    
    if words.len() > 0 {
        unique_words.len() as f32 / words.len() as f32
    } else {
        0.0
    }
}

#[derive(Debug, Clone)]
pub struct TextComplexity {
    pub entropy: f64,
    pub repetition_score: f32,
    pub structural_score: f32,
    pub semantic_score: f32,
    pub unique_chars: usize,
    pub total_chars: usize,
}

impl TextComplexity {
    pub fn print_analysis(&self) {
        println!("📊 Text Complexity Analysis:");
        println!("  Entropy: {:.3}", self.entropy);
        println!("  Repetition score: {:.3}", self.repetition_score);
        println!("  Structural score: {:.3}", self.structural_score);
        println!("  Semantic score: {:.3}", self.semantic_score);
        println!("  Unique chars: {} / {}", self.unique_chars, self.total_chars);
        println!("  Complexity level: {}", self.get_complexity_level());
    }
    
    pub fn get_complexity_level(&self) -> &'static str {
        let combined_score = (self.entropy / 8.0) + 
                            (self.repetition_score as f64) + 
                            (self.structural_score as f64) + 
                            (self.semantic_score as f64);
        
        if combined_score > 2.5 {
            "Very High"
        } else if combined_score > 2.0 {
            "High"
        } else if combined_score > 1.5 {
            "Medium"
        } else if combined_score > 1.0 {
            "Low"
        } else {
            "Very Low"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_ratio() {
        let ratio = calculate_compression_ratio(1000, 250);
        assert_eq!(ratio, 4.0);
    }

    #[test]
    fn test_fixed_patches() {
        let text = b"hello world";
        let patches = create_fixed_patches(text, 4);
        assert_eq!(patches.len(), 3); // "hell", "o wo", "rld"
    }

    #[test]
    fn test_space_patches() {
        let text = b"hello world test";
        let patches = create_space_patches(text);
        assert_eq!(patches.len(), 3); // "hello ", "world ", "test"
    }

    #[test]
    fn test_text_complexity() {
        let text = "Hello, world! This is a test.";
        let complexity = analyze_text_complexity(text);
        assert!(complexity.entropy > 0.0);
        assert!(complexity.unique_chars > 0);
    }
} 