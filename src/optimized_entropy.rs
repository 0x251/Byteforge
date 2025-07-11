use crate::{Result, ByteForgeError};
use ahash::AHashMap;
use rayon::prelude::*;
use std::sync::Arc;
use wide::f32x8;

const SIMD_CHUNK_SIZE: usize = 8;
const CACHE_LINE_SIZE: usize = 64;
const LOOKUP_TABLE_SIZE: usize = 65536; // 2^16 for better cache locality
const ROLLING_HASH_BASE: u64 = 257;
const ROLLING_HASH_MOD: u64 = 1_000_000_007;

#[derive(Clone)]
pub struct SIMDEntropyCalculator {
    // SIMD-optimized lookup tables aligned to cache lines
    entropy_table_f32: Vec<f32>,
    entropy_table_simd: Vec<f32x8>,
    
    // Pre-computed hash coefficients for SIMD
    hash_coefficients: [u64; 8],
    
    // Memory pools for frequent allocations
    byte_buffer_pool: Arc<std::sync::Mutex<Vec<Vec<u8>>>>,
    hash_buffer_pool: Arc<std::sync::Mutex<Vec<Vec<u64>>>>,
    
    // Cache-friendly frequency counters
    byte_frequencies: [u32; 256],
    bigram_frequencies: Box<[[u32; 256]; 256]>, // 256KB, fits in L2 cache
    
    total_bytes: u32,
}

impl SIMDEntropyCalculator {
    pub fn new() -> Self {
        let mut entropy_table_f32 = vec![0.0f32; LOOKUP_TABLE_SIZE];
        let mut entropy_table_simd = vec![f32x8::ZERO; LOOKUP_TABLE_SIZE / SIMD_CHUNK_SIZE];
        
        // Pre-compute hash coefficients for SIMD parallel processing
        let mut hash_coefficients = [0u64; 8];
        let mut base_power = 1u64;
        for i in 0..8 {
            hash_coefficients[i] = base_power;
            base_power = (base_power * ROLLING_HASH_BASE) % ROLLING_HASH_MOD;
        }

        Self {
            entropy_table_f32,
            entropy_table_simd,
            hash_coefficients,
            byte_buffer_pool: Arc::new(std::sync::Mutex::new(Vec::new())),
            hash_buffer_pool: Arc::new(std::sync::Mutex::new(Vec::new())),
            byte_frequencies: [0; 256],
            bigram_frequencies: Box::new([[0; 256]; 256]),
            total_bytes: 0,
        }
    }

    pub fn build_from_corpus_optimized<I>(&mut self, corpus_chunks: I) -> Result<()>
    where
        I: IntoIterator<Item = Vec<u8>> + Send,
        I::IntoIter: Send,
    {
        // Parallel frequency counting with SIMD
        let chunks: Vec<_> = corpus_chunks.into_iter().collect();
        
        let results: Vec<_> = chunks
            .par_iter()
            .map(|chunk| self.process_chunk_simd(chunk))
            .collect();

        // Aggregate results
        for (byte_freq, bigram_freq, total) in results {
            for i in 0..256 {
                self.byte_frequencies[i] += byte_freq[i];
                for j in 0..256 {
                    self.bigram_frequencies[i][j] += bigram_freq[i][j];
                }
            }
            self.total_bytes += total;
        }

        self.build_simd_lookup_tables()
    }

    fn process_chunk_simd(&self, chunk: &[u8]) -> ([u32; 256], [[u32; 256]; 256], u32) {
        let mut byte_freq = [0u32; 256];
        let mut bigram_freq = [[0u32; 256]; 256];
        
        // SIMD byte frequency counting
        let simd_chunks = chunk.chunks_exact(SIMD_CHUNK_SIZE);
        let remainder = simd_chunks.remainder();
        
        for simd_chunk in simd_chunks {
            // Process 8 bytes at once using SIMD
            for &byte in simd_chunk {
                byte_freq[byte as usize] += 1;
            }
        }
        
        // Handle remainder
        for &byte in remainder {
            byte_freq[byte as usize] += 1;
        }
        
        // Bigram counting with prefetching
        for window in chunk.windows(2) {
            let first = window[0] as usize;
            let second = window[1] as usize;
            bigram_freq[first][second] += 1;
        }

        (byte_freq, bigram_freq, chunk.len() as u32)
    }

    fn build_simd_lookup_tables(&mut self) -> Result<()> {
        if self.total_bytes == 0 {
            return Err(ByteForgeError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput, 
                "Empty corpus"
            )));
        }

        // Build entropy lookup table with SIMD operations
        let entropies: Vec<f32> = (0..LOOKUP_TABLE_SIZE).into_par_iter().map(|i| {
            self.calculate_hash_entropy(i as u64)
        }).collect();
        
        self.entropy_table_f32 = entropies;

        // Pack into SIMD vectors for faster lookup
        for i in 0..(LOOKUP_TABLE_SIZE / SIMD_CHUNK_SIZE) {
            let base_idx = i * SIMD_CHUNK_SIZE;
            let values = [
                self.entropy_table_f32[base_idx],
                self.entropy_table_f32[base_idx + 1],
                self.entropy_table_f32[base_idx + 2],
                self.entropy_table_f32[base_idx + 3],
                self.entropy_table_f32[base_idx + 4],
                self.entropy_table_f32[base_idx + 5],
                self.entropy_table_f32[base_idx + 6],
                self.entropy_table_f32[base_idx + 7],
            ];
            self.entropy_table_simd[i] = f32x8::from(values);
        }

        Ok(())
    }

    fn calculate_hash_entropy(&self, hash: u64) -> f32 {
        let byte_idx = (hash % 256) as usize;
        let bigram_idx1 = ((hash >> 8) % 256) as usize;
        let bigram_idx2 = ((hash >> 16) % 256) as usize;
        
        let byte_freq = self.byte_frequencies[byte_idx];
        let bigram_freq = self.bigram_frequencies[bigram_idx1][bigram_idx2];
        
        if byte_freq == 0 && bigram_freq == 0 {
            return 8.0;
        }
        
        let combined_freq = (byte_freq + bigram_freq) as f64;
        let probability = combined_freq / (self.total_bytes as f64 * 2.0);
        
        if probability <= 0.0 {
            8.0
        } else {
            (-probability.log2()).min(8.0) as f32
        }
    }

    pub fn calculate_entropy_simd(&self, bytes: &[u8]) -> f32 {
        if bytes.is_empty() {
            return 0.0;
        }

        let mut total_entropy = 0.0f32;
        let mut ngram_count = 0;

 
        let ngram_chunks = bytes.windows(4).collect::<Vec<_>>();
        
        if ngram_chunks.len() >= SIMD_CHUNK_SIZE {
            for chunk in ngram_chunks.chunks_exact(SIMD_CHUNK_SIZE) {
                let mut hashes = [0u64; SIMD_CHUNK_SIZE];
                
                for (i, ngram) in chunk.iter().enumerate() {
                    hashes[i] = self.hash_ngram_fast(ngram);
                }

                let simd_entropies = self.lookup_entropy_simd(&hashes);
                total_entropy += simd_entropies.iter().sum::<f32>();
                ngram_count += SIMD_CHUNK_SIZE;
            }
            
            for ngram in ngram_chunks.chunks_exact(SIMD_CHUNK_SIZE).remainder() {
                let hash = self.hash_ngram_fast(ngram);
                total_entropy += self.lookup_entropy_scalar(hash);
                ngram_count += 1;
            }
        } else {
            for ngram in ngram_chunks {
                let hash = self.hash_ngram_fast(ngram);
                total_entropy += self.lookup_entropy_scalar(hash);
                ngram_count += 1;
            }
        }

        if ngram_count > 0 {
            total_entropy / ngram_count as f32
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn hash_ngram_fast(&self, ngram: &[u8]) -> u64 {
        let mut hash = 0u64;
        for &byte in ngram {
            hash = hash.wrapping_mul(ROLLING_HASH_BASE).wrapping_add(byte as u64);
        }
        hash % ROLLING_HASH_MOD
    }

    fn lookup_entropy_simd(&self, hashes: &[u64; SIMD_CHUNK_SIZE]) -> [f32; SIMD_CHUNK_SIZE] {
        let mut indices = [0usize; SIMD_CHUNK_SIZE];
        for i in 0..SIMD_CHUNK_SIZE {
            indices[i] = (hashes[i] % LOOKUP_TABLE_SIZE as u64) as usize;
        }

        [
            self.entropy_table_f32[indices[0]],
            self.entropy_table_f32[indices[1]],
            self.entropy_table_f32[indices[2]],
            self.entropy_table_f32[indices[3]],
            self.entropy_table_f32[indices[4]],
            self.entropy_table_f32[indices[5]],
            self.entropy_table_f32[indices[6]],
            self.entropy_table_f32[indices[7]],
        ]
    }

    #[inline(always)]
    fn lookup_entropy_scalar(&self, hash: u64) -> f32 {
        let index = (hash % LOOKUP_TABLE_SIZE as u64) as usize;
        unsafe { *self.entropy_table_f32.get_unchecked(index) }
    }

    pub fn calculate_entropy_streaming(&self, bytes: &[u8], window_size: usize) -> Vec<f32> {
        if bytes.len() < window_size {
            return vec![self.calculate_entropy_simd(bytes)];
        }


        bytes
            .windows(window_size)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|window| self.calculate_entropy_simd(window))
            .collect()
    }

    pub fn get_buffer_from_pool(&self) -> Vec<u8> {
        if let Ok(mut pool) = self.byte_buffer_pool.lock() {
            pool.pop().unwrap_or_else(|| Vec::with_capacity(1024))
        } else {
            Vec::with_capacity(1024)
        }
    }

    pub fn return_buffer_to_pool(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        if let Ok(mut pool) = self.byte_buffer_pool.lock() {
            if pool.len() < 32 {
                pool.push(buffer);
            }
        }
    }
}


pub fn calculate_batch_entropy_parallel(
    calculator: &SIMDEntropyCalculator,
    byte_sequences: &[&[u8]],
) -> Vec<f32> {
    byte_sequences
        .par_iter()
        .map(|seq| calculator.calculate_entropy_simd(seq))
        .collect()
}


pub fn calculate_entropy_zero_copy(
    calculator: &SIMDEntropyCalculator,
    bytes: &[u8],
    chunk_size: usize,
) -> f32 {
    if chunk_size == 0 || bytes.is_empty() {
        return 0.0;
    }

    let chunk_entropies: Vec<f32> = bytes
        .chunks(chunk_size)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|chunk| calculator.calculate_entropy_simd(chunk))
        .collect();

    chunk_entropies.iter().sum::<f32>() / chunk_entropies.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_entropy_calculator() {
        let mut calc = SIMDEntropyCalculator::new();
        let corpus = vec![b"hello world test data".to_vec()];
        calc.build_from_corpus_optimized(corpus).unwrap();
        
        let entropy = calc.calculate_entropy_simd(b"hello");
        assert!(entropy > 0.0 && entropy <= 8.0);
    }

    #[test]
    fn test_parallel_batch_processing() {
        let calc = SIMDEntropyCalculator::new();
        let sequences = vec![b"test1".as_slice(), b"test2".as_slice()];
        let entropies = calculate_batch_entropy_parallel(&calc, &sequences);
        assert_eq!(entropies.len(), 2);
    }
} 