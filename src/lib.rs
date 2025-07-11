pub mod patching;
pub mod entropy;
pub mod transformer;
pub mod training;
pub mod inference;
pub mod utils;
// pub mod benchmark; // Disabled for now due to plotting library errors
pub mod simple_benchmark;
pub mod optimized_entropy;
pub mod optimized_patching;
pub mod turbo_benchmark;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ByteForgeError {
    #[error("Invalid patch configuration: {0}")]
    InvalidPatchConfig(String),
    #[error("Model error: {0}")]
    ModelError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ByteForgeError>;

#[derive(Debug, Clone)]
pub struct ByteForgeConfig {
    pub patch_size_range: (usize, usize),
    pub entropy_threshold: f32,
    pub compression_threshold: f32,
    pub semantic_weight: f32,
    pub model_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub use_quantization: bool,
    pub use_streaming: bool,
}

impl Default for ByteForgeConfig {
    fn default() -> Self {
        Self {
            patch_size_range: (1, 16),
            entropy_threshold: 0.5,
            compression_threshold: 0.3,
            semantic_weight: 0.2,
            model_dim: 512,
            num_heads: 8,
            num_layers: 6,
            vocab_size: 256,
            max_seq_len: 4096,
            use_quantization: true,
            use_streaming: false,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_end_to_end_processing() {
        let config = ByteForgeConfig::default();
        
        let repetitive_pattern = "Repetitive pattern ".repeat(100);
        let test_cases = vec![
            "Simple text for testing",
            "fn main() { println!(\"Hello, Rust!\"); }",
            r#"{"json": "data", "with": {"nested": "objects"}}"#,
            &repetitive_pattern,
            "", // Empty input
            "x", // Single character
            "Mixed content: code, text, 123, symbols!@#",
        ];

        for (i, input) in test_cases.iter().enumerate() {
            println!("Testing case {}: {}", i, input);
            
            let mut patcher = patching::MultiSignalPatcher::new(config.clone());
            let result = patcher.patch_bytes(input.as_bytes());
            assert!(result.is_ok(), "Standard patching failed for case {}", i);
            
            let patches = result.unwrap();
            
            if !input.is_empty() {
                assert!(!patches.is_empty(), "Empty patches for non-empty input case {}", i);
                
                let total_bytes: usize = patches.iter().map(|p| p.bytes.len()).sum();
                assert!(total_bytes <= input.len(), "Patches exceed input size for case {}", i);
            }
            
            if input.len() > 10 {
                let mut entropy_calc = optimized_entropy::SIMDEntropyCalculator::new();
                let corpus = vec![input.as_bytes().to_vec()];
                entropy_calc.build_from_corpus_optimized(corpus).unwrap();
                
                let mut turbo_patcher = optimized_patching::TurboMultiSignalPatcher::new(Arc::new(entropy_calc));
                let turbo_result = turbo_patcher.patch_bytes_turbo(input.as_bytes());
                assert!(turbo_result.is_ok(), "Turbo patching failed for case {}", i);
            }
        }
    }

    #[test]
    fn test_error_handling() {
        let config = ByteForgeConfig::default();
        let mut patcher = patching::MultiSignalPatcher::new(config);
        
        let huge_input = "x".repeat(1_000_000);
        let result = patcher.patch_bytes(huge_input.as_bytes());
        assert!(result.is_ok(), "Should handle large inputs gracefully");
        
        let mut entropy_calc = entropy::UltraFastEntropyCalculator::new();
        
        let empty_result = entropy_calc.build_from_corpus(vec![]);
        match empty_result {
            Ok(_) => println!("Empty corpus handled gracefully"),
            Err(_) => println!("Empty corpus rejected as expected"),
        }
        
        let tiny_corpus = vec![vec![b'a']];
        let tiny_result = entropy_calc.build_from_corpus(tiny_corpus);
        assert!(tiny_result.is_ok(), "Should handle tiny corpus");
    }

    #[test]
    fn test_memory_efficiency() {
        let config = ByteForgeConfig::default();
        let input = "Test memory efficiency with reasonable input size".repeat(100);
        
        let mut patcher = patching::MultiSignalPatcher::new(config);
        let patches = patcher.patch_bytes(input.as_bytes()).unwrap();
        
        let patch_memory: usize = patches.iter().map(|p| p.bytes.len()).sum();
        assert!(patch_memory <= input.len() * 2, "Patch memory should be reasonable");
    }

    #[test]
    fn test_performance_regression() {
        use std::time::Instant;
        
        let config = ByteForgeConfig::default();
        let test_input = "Performance regression test input with mixed content".repeat(1000);
        
        let mut patcher = patching::MultiSignalPatcher::new(config);
        
        let start = Instant::now();
        let patches = patcher.patch_bytes(test_input.as_bytes()).unwrap();
        let duration = start.elapsed();
        
        let throughput = test_input.len() as f64 / duration.as_secs_f64();
        assert!(throughput > 50_000.0, "Throughput too low: {:.0} bytes/s", throughput);
        
        let patch_ratio = patches.len() as f64 / test_input.len() as f64;
        assert!(patch_ratio < 0.5, "Too many patches created: ratio {:.2}", patch_ratio);
    }

    #[test]
    fn test_streaming_processing() {
        let mut streaming_calc = entropy::StreamingEntropyCalculator::new(128);
        
        let test_stream = "Streaming test with various content types: code, text, numbers 123";
        let mut entropies = Vec::new();
        
        for byte in test_stream.bytes() {
            let entropy = streaming_calc.feed_byte(byte).unwrap();
            entropies.push(entropy);
        }
        
        assert_eq!(entropies.len(), test_stream.len());
        assert!(entropies.iter().all(|&e| e >= 0.0 && e <= 8.0));
    }
} 