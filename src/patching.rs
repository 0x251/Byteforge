use crate::{ByteForgeConfig, Result};
use std::collections::VecDeque;
use ahash::AHashMap;

#[derive(Debug, Clone)]
pub struct Patch {
    pub bytes: Vec<u8>,
    pub start_pos: usize,
    pub end_pos: usize,
    pub complexity_score: f32,
    pub patch_type: PatchType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatchType {
    Simple,      // Low complexity, common patterns
    Complex,     // High complexity, rare patterns
    Semantic,    // Word/sentence boundaries
    Repetitive,  // Repeated patterns
    Structural,  // Code/markup structure
}

pub struct MultiSignalPatcher {
    config: ByteForgeConfig,
    entropy_cache: AHashMap<u64, f32>,
    pattern_cache: AHashMap<Vec<u8>, f32>,
    ngram_counts: AHashMap<Vec<u8>, u32>,
    rolling_hash: u64,
    window_buffer: VecDeque<u8>,
}

impl MultiSignalPatcher {
    pub fn new(config: ByteForgeConfig) -> Self {
        Self {
            config,
            entropy_cache: AHashMap::new(),
            pattern_cache: AHashMap::new(),
            ngram_counts: AHashMap::new(),
            rolling_hash: 0,
            window_buffer: VecDeque::new(),
        }
    }

    pub fn patch_bytes(&mut self, bytes: &[u8]) -> Result<Vec<Patch>> {
        let mut patches = Vec::new();
        let mut current_patch_start = 0;
        let mut i = 0;

        while i < bytes.len() {
            let signals = self.calculate_signals(&bytes, i)?;
            
            let should_split = self.should_split_patch(&signals, i - current_patch_start);
            
            if should_split || i - current_patch_start >= self.config.patch_size_range.1 {
                if i > current_patch_start {
                    let patch = self.create_patch(&bytes, current_patch_start, i, &signals)?;
                    patches.push(patch);
                    current_patch_start = i;
                }
            }
            
            i += 1;
        }

        if current_patch_start < bytes.len() {
            let signals = self.calculate_signals(&bytes, bytes.len() - 1)?;
            let patch = self.create_patch(&bytes, current_patch_start, bytes.len(), &signals)?;
            patches.push(patch);
        }

        Ok(patches)
    }

    fn calculate_signals(&mut self, bytes: &[u8], pos: usize) -> Result<PatchingSignals> {
        let entropy = self.calculate_fast_entropy(bytes, pos)?;
        let compression_ratio = self.calculate_compression_ratio(bytes, pos)?;
        let semantic_boundary = self.detect_semantic_boundary(bytes, pos)?;
        let repetition_score = self.calculate_repetition_score(bytes, pos)?;
        let structural_score = self.calculate_structural_score(bytes, pos)?;

        Ok(PatchingSignals {
            entropy,
            compression_ratio,
            semantic_boundary,
            repetition_score,
            structural_score,
        })
    }

    fn should_split_patch(&self, signals: &PatchingSignals, current_length: usize) -> bool {
        if current_length < self.config.patch_size_range.0 {
            return false;
        }

        let entropy_trigger = signals.entropy > self.config.entropy_threshold;
        let compression_trigger = signals.compression_ratio > self.config.compression_threshold;
        let semantic_trigger = signals.semantic_boundary > self.config.semantic_weight;
        let repetition_trigger = signals.repetition_score > 0.8;
        let structural_trigger = signals.structural_score > 0.7;

        let signal_count = [entropy_trigger, compression_trigger, semantic_trigger, 
                           repetition_trigger, structural_trigger]
            .iter()
            .map(|&x| x as u32)
            .sum::<u32>();

        signal_count >= 2 || (signal_count >= 1 && current_length >= self.config.patch_size_range.1 / 2)
    }

    fn calculate_fast_entropy(&mut self, bytes: &[u8], pos: usize) -> Result<f32> {
        if pos < 3 {
            return Ok(0.0);
        }

        let ngram = &bytes[pos.saturating_sub(3)..=pos];
        let hash = self.hash_bytes(ngram);
        
        if let Some(&cached_entropy) = self.entropy_cache.get(&hash) {
            return Ok(cached_entropy);
        }

        let entropy = self.compute_ngram_entropy(ngram)?;
        self.entropy_cache.insert(hash, entropy);
        
        Ok(entropy)
    }

    fn compute_ngram_entropy(&self, ngram: &[u8]) -> Result<f32> {
        let mut counts = [0u32; 256];
        let mut total = 0u32;
        
        for &byte in ngram {
            counts[byte as usize] += 1;
            total += 1;
        }

        let mut entropy = 0.0f32;
        for count in counts.iter().filter(|&&c| c > 0) {
            let p = *count as f32 / total as f32;
            entropy -= p * p.log2();
        }

        Ok(entropy)
    }

    fn calculate_compression_ratio(&self, bytes: &[u8], pos: usize) -> Result<f32> {
        if pos < 8 {
            return Ok(0.0);
        }

        let window = &bytes[pos.saturating_sub(8)..=pos];
        let original_size = window.len();
        let compressed_size = self.estimate_compression_size(window)?;
        
        Ok(1.0 - (compressed_size as f32 / original_size as f32))
    }

    fn estimate_compression_size(&self, window: &[u8]) -> Result<usize> {
        let mut unique_bytes = std::collections::HashSet::new();
        let mut repeat_count = 0;
        
        for i in 0..window.len() {
            if !unique_bytes.insert(window[i]) {
                repeat_count += 1;
            }
        }

        Ok(window.len() - repeat_count / 2)
    }

    fn detect_semantic_boundary(&self, bytes: &[u8], pos: usize) -> Result<f32> {
        if pos == 0 {
            return Ok(0.0);
        }

        let current_byte = bytes[pos];
        let prev_byte = bytes[pos - 1];

        let is_word_boundary = self.is_word_boundary(prev_byte, current_byte);
        let is_sentence_boundary = self.is_sentence_boundary(prev_byte, current_byte);
        let is_line_boundary = current_byte == b'\n';

        let score = if is_sentence_boundary {
            0.9
        } else if is_line_boundary {
            0.8
        } else if is_word_boundary {
            0.6
        } else {
            0.0
        };

        Ok(score)
    }

    fn is_word_boundary(&self, prev: u8, current: u8) -> bool {
        let prev_is_alphanum = prev.is_ascii_alphanumeric();
        let current_is_alphanum = current.is_ascii_alphanumeric();
        let is_space_transition = prev == b' ' || current == b' ';
        
        (prev_is_alphanum != current_is_alphanum) || is_space_transition
    }

    fn is_sentence_boundary(&self, prev: u8, current: u8) -> bool {
        matches!(prev, b'.' | b'!' | b'?') && (current == b' ' || current == b'\n')
    }

    fn calculate_repetition_score(&self, bytes: &[u8], pos: usize) -> Result<f32> {
        if pos < 4 {
            return Ok(0.0);
        }

        let window_size = 8.min(pos);
        let window = &bytes[pos - window_size..pos];
        
        let mut max_repeat_length = 0;
        
        for pattern_len in 1..=window_size / 2 {
            let pattern = &window[window_size - pattern_len..];
            let mut repeat_count = 0;
            
            for i in (0..window_size - pattern_len).step_by(pattern_len) {
                if &window[i..i + pattern_len] == pattern {
                    repeat_count += 1;
                } else {
                    break;
                }
            }
            
            if repeat_count > 1 {
                max_repeat_length = max_repeat_length.max(pattern_len * repeat_count);
            }
        }

        Ok(max_repeat_length as f32 / window_size as f32)
    }

    fn calculate_structural_score(&self, bytes: &[u8], pos: usize) -> Result<f32> {
        if pos == 0 {
            return Ok(0.0);
        }

        let current_byte = bytes[pos];
        
        let structural_chars = [b'{', b'}', b'[', b']', b'(', b')', b'<', b'>', b';', b':', b','];
        let is_structural = structural_chars.contains(&current_byte);
        
        if is_structural {
            Ok(0.8)
        } else {
            Ok(0.0)
        }
    }

    fn create_patch(&self, bytes: &[u8], start: usize, end: usize, signals: &PatchingSignals) -> Result<Patch> {
        let patch_bytes = bytes[start..end].to_vec();
        let complexity_score = self.calculate_complexity_score(signals);
        let patch_type = self.determine_patch_type(signals);

        Ok(Patch {
            bytes: patch_bytes,
            start_pos: start,
            end_pos: end,
            complexity_score,
            patch_type,
        })
    }

    fn calculate_complexity_score(&self, signals: &PatchingSignals) -> f32 {
        let weights = [0.3, 0.2, 0.2, 0.15, 0.15];
        let values = [
            signals.entropy,
            signals.compression_ratio,
            signals.semantic_boundary,
            signals.repetition_score,
            signals.structural_score,
        ];

        weights.iter().zip(values.iter()).map(|(w, v)| w * v).sum()
    }

    fn determine_patch_type(&self, signals: &PatchingSignals) -> PatchType {
        if signals.repetition_score > 0.7 {
            PatchType::Repetitive
        } else if signals.structural_score > 0.6 {
            PatchType::Structural
        } else if signals.semantic_boundary > 0.5 {
            PatchType::Semantic
        } else if signals.entropy > 0.7 {
            PatchType::Complex
        } else {
            PatchType::Simple
        }
    }

    fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        bytes.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Debug, Clone)]
struct PatchingSignals {
    entropy: f32,
    compression_ratio: f32,
    semantic_boundary: f32,
    repetition_score: f32,
    structural_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_patching() {
        let config = ByteForgeConfig::default();
        let mut patcher = MultiSignalPatcher::new(config);
        
        let text = b"Hello world! This is a test.";
        let patches = patcher.patch_bytes(text).unwrap();
        
        assert!(!patches.is_empty());
        assert!(patches.iter().all(|p| !p.bytes.is_empty()));
    }

    #[test]
    fn test_semantic_boundary_detection() {
        let config = ByteForgeConfig::default();
        let patcher = MultiSignalPatcher::new(config);
        
        let text = b"Hello world";
        let score = patcher.detect_semantic_boundary(text, 5).unwrap(); // space position
        assert!(score > 0.0);
    }
} 