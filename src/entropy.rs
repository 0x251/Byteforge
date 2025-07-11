use crate::Result;
use ahash::AHashMap;
use rayon::prelude::*;

const NGRAM_SIZE: usize = 4;
const ROLLING_HASH_BASE: u64 = 257;
const ROLLING_HASH_MOD: u64 = 1000000007;
const LOOKUP_TABLE_SIZE: usize = 1048576; // 1M entries

pub struct UltraFastEntropyCalculator {
    ngram_entropy_table: Vec<f32>,
    rolling_hash_state: RollingHashState,
    byte_frequencies: [u32; 256],
    total_bytes: u32,
    ngram_cache: AHashMap<u64, f32>,
}

#[derive(Debug, Clone)]
struct RollingHashState {
    hash: u64,
    window: [u8; NGRAM_SIZE],
    window_pos: usize,
    base_power: u64,
}

impl RollingHashState {
    fn new() -> Self {
        Self {
            hash: 0,
            window: [0; NGRAM_SIZE],
            window_pos: 0,
            base_power: ROLLING_HASH_BASE.pow(NGRAM_SIZE as u32 - 1) % ROLLING_HASH_MOD,
        }
    }

    fn update(&mut self, new_byte: u8) {
        if self.window_pos >= NGRAM_SIZE {
            let old_byte = self.window[self.window_pos % NGRAM_SIZE];
            self.hash = (self.hash + ROLLING_HASH_MOD - 
                        (old_byte as u64 * self.base_power) % ROLLING_HASH_MOD) % ROLLING_HASH_MOD;
        }

        self.window[self.window_pos % NGRAM_SIZE] = new_byte;
        self.hash = (self.hash * ROLLING_HASH_BASE + new_byte as u64) % ROLLING_HASH_MOD;
        self.window_pos += 1;
    }

    fn get_hash(&self) -> u64 {
        self.hash
    }

    fn is_ready(&self) -> bool {
        self.window_pos >= NGRAM_SIZE
    }
}

impl UltraFastEntropyCalculator {
    pub fn new() -> Self {
        Self {
            ngram_entropy_table: vec![0.0; LOOKUP_TABLE_SIZE],
            rolling_hash_state: RollingHashState::new(),
            byte_frequencies: [0; 256],
            total_bytes: 0,
            ngram_cache: AHashMap::new(),
        }
    }

    pub fn build_from_corpus<I>(&mut self, corpus_chunks: I) -> Result<()>
    where
        I: IntoIterator<Item = Vec<u8>> + Send,
        I::IntoIter: Send,
    {
        let mut ngram_counts: AHashMap<u64, u32> = AHashMap::new();
        let mut total_ngrams = 0u64;

        self.byte_frequencies = [0; 256];
        self.total_bytes = 0;

        for chunk in corpus_chunks {
            self.process_chunk_frequencies(&chunk, &mut ngram_counts, &mut total_ngrams)?;
        }

        self.build_lookup_table(&ngram_counts, total_ngrams)?;
        Ok(())
    }

    fn process_chunk_frequencies(
        &mut self,
        chunk: &[u8],
        ngram_counts: &mut AHashMap<u64, u32>,
        total_ngrams: &mut u64,
    ) -> Result<()> {
        let mut rolling_hash = RollingHashState::new();

        for &byte in chunk {
            self.byte_frequencies[byte as usize] += 1;
            self.total_bytes += 1;

            rolling_hash.update(byte);

            if rolling_hash.is_ready() {
                let hash = rolling_hash.get_hash();
                let table_index = (hash % LOOKUP_TABLE_SIZE as u64) as usize;
                *ngram_counts.entry(table_index as u64).or_insert(0) += 1;
                *total_ngrams += 1;
            }
        }

        Ok(())
    }

    fn build_lookup_table(&mut self, ngram_counts: &AHashMap<u64, u32>, total_ngrams: u64) -> Result<()> {
        self.ngram_entropy_table.par_iter_mut().enumerate().for_each(|(i, entropy)| {
            let count = ngram_counts.get(&(i as u64)).copied().unwrap_or(1);
            let probability = count as f64 / total_ngrams as f64;
            *entropy = -probability.log2() as f32;
        });

        Ok(())
    }

    pub fn calculate_entropy_fast(&mut self, bytes: &[u8], pos: usize) -> Result<f32> {
        if bytes.is_empty() {
            return Ok(0.0);
        }
        
        if pos >= bytes.len() {
            return Ok(0.0);
        }
        
        if pos < NGRAM_SIZE {
            return Ok(self.calculate_byte_entropy(bytes[pos]));
        }

        let ngram_start = pos.saturating_sub(NGRAM_SIZE);
        let ngram = &bytes[ngram_start..pos];
        let hash = self.hash_ngram(ngram);
        
        if let Some(&cached_entropy) = self.ngram_cache.get(&hash) {
            return Ok(cached_entropy);
        }

        let entropy = self.calculate_ngram_entropy_from_table(hash);
        self.ngram_cache.insert(hash, entropy);
        
        Ok(entropy)
    }

    fn calculate_byte_entropy(&self, byte: u8) -> f32 {
        if self.total_bytes == 0 {
            return 8.0; // Maximum entropy for a byte
        }

        let frequency = self.byte_frequencies[byte as usize];
        if frequency == 0 {
            return 8.0;
        }

        let probability = frequency as f64 / self.total_bytes as f64;
        -probability.log2() as f32
    }

    fn calculate_ngram_entropy_from_table(&self, hash: u64) -> f32 {
        let table_index = (hash % LOOKUP_TABLE_SIZE as u64) as usize;
        self.ngram_entropy_table[table_index]
    }

    fn hash_ngram(&self, ngram: &[u8]) -> u64 {
        let mut hash = 0u64;
        for &byte in ngram {
            hash = (hash * ROLLING_HASH_BASE + byte as u64) % ROLLING_HASH_MOD;
        }
        hash
    }

    pub fn calculate_context_entropy(&mut self, bytes: &[u8], pos: usize, context_size: usize) -> Result<f32> {
        if pos == 0 {
            return Ok(0.0);
        }

        let context_start = pos.saturating_sub(context_size);
        let context = &bytes[context_start..pos];
        
        let mut total_entropy = 0.0f32;
        let mut count = 0;

        for i in NGRAM_SIZE..context.len() {
            let entropy = self.calculate_entropy_fast(context, i)?;
            total_entropy += entropy;
            count += 1;
        }

        if count > 0 {
            Ok(total_entropy / count as f32)
        } else {
            Ok(0.0)
        }
    }

    pub fn calculate_adaptive_entropy(&mut self, bytes: &[u8], pos: usize) -> Result<f32> {
        if pos < NGRAM_SIZE {
            return Ok(self.calculate_byte_entropy(bytes[pos]));
        }

        let short_entropy = self.calculate_entropy_fast(bytes, pos)?;
        let long_entropy = self.calculate_context_entropy(bytes, pos, 16)?;
        
        let blend_factor = 0.7;
        Ok(blend_factor * short_entropy + (1.0 - blend_factor) * long_entropy)
    }

    pub fn get_entropy_statistics(&self) -> EntropyStatistics {
        let mut min_entropy = f32::MAX;
        let mut max_entropy = f32::MIN;
        let mut sum_entropy = 0.0f32;
        let mut count = 0;

        for &entropy in &self.ngram_entropy_table {
            if entropy > 0.0 {
                min_entropy = min_entropy.min(entropy);
                max_entropy = max_entropy.max(entropy);
                sum_entropy += entropy;
                count += 1;
            }
        }

        EntropyStatistics {
            min_entropy,
            max_entropy,
            mean_entropy: if count > 0 { sum_entropy / count as f32 } else { 0.0 },
            table_utilization: count as f32 / LOOKUP_TABLE_SIZE as f32,
        }
    }

    pub fn predict_next_byte_entropy(&mut self, bytes: &[u8], pos: usize) -> Result<f32> {
        if pos < NGRAM_SIZE - 1 {
            return Ok(8.0); // Maximum uncertainty
        }

        let context = &bytes[pos.saturating_sub(NGRAM_SIZE - 1)..pos];
        let mut total_entropy = 0.0f32;
        let mut predictions = 0;

        for next_byte in 0..=255u8 {
            let mut extended_context = context.to_vec();
            extended_context.push(next_byte);
            
            let hash = self.hash_ngram(&extended_context);
            let entropy = self.calculate_ngram_entropy_from_table(hash);
            
            if entropy > 0.0 {
                total_entropy += entropy;
                predictions += 1;
            }
        }

        if predictions > 0 {
            Ok(total_entropy / predictions as f32)
        } else {
            Ok(8.0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct EntropyStatistics {
    pub min_entropy: f32,
    pub max_entropy: f32,
    pub mean_entropy: f32,
    pub table_utilization: f32,
}

pub struct StreamingEntropyCalculator {
    base_calculator: UltraFastEntropyCalculator,
    running_window: Vec<u8>,
    window_size: usize,
    position: usize,
}

impl StreamingEntropyCalculator {
    pub fn new(window_size: usize) -> Self {
        Self {
            base_calculator: UltraFastEntropyCalculator::new(),
            running_window: Vec::with_capacity(window_size),
            window_size,
            position: 0,
        }
    }

    pub fn feed_byte(&mut self, byte: u8) -> Result<f32> {
        if self.running_window.len() < self.window_size {
            self.running_window.push(byte);
        } else {
            self.running_window[self.position % self.window_size] = byte;
        }

        self.position += 1;

        // Fix: ensure we don't go out of bounds
        let window_pos = if self.running_window.len() <= 1 {
            0
        } else {
            self.running_window.len() - 1
        };

        let entropy = self.base_calculator.calculate_entropy_fast(
            &self.running_window,
            window_pos
        )?;

        Ok(entropy)
    }

    pub fn get_current_entropy(&mut self) -> Result<f32> {
        if self.running_window.is_empty() {
            return Ok(0.0);
        }

        self.base_calculator.calculate_entropy_fast(
            &self.running_window,
            self.running_window.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_hash() {
        let mut hash_state = RollingHashState::new();
        
        let test_data = b"hello";
        for &byte in test_data {
            hash_state.update(byte);
        }

        assert!(hash_state.is_ready());
        assert!(hash_state.get_hash() > 0);
    }

    #[test]
    fn test_entropy_calculation() {
        let mut calculator = UltraFastEntropyCalculator::new();
        
        let corpus_chunks = vec![
            b"hello world".to_vec(),
            b"hello rust".to_vec(),
            b"world rust".to_vec(),
        ];

        calculator.build_from_corpus(corpus_chunks).unwrap();
        
        let test_data = b"hello";
        let entropy = calculator.calculate_entropy_fast(test_data, 4).unwrap();
        
        assert!(entropy > 0.0);
        assert!(entropy <= 8.0);
    }

    #[test]
    fn test_streaming_entropy() {
        let mut streaming_calc = StreamingEntropyCalculator::new(100);
        
        let test_data = b"hello world test";
        for &byte in test_data {
            let entropy = streaming_calc.feed_byte(byte).unwrap();
            assert!(entropy >= 0.0);
        }
    }
} 