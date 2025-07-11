use crate::{Result};
use crate::patching::{Patch, PatchType};
use crate::optimized_entropy::SIMDEntropyCalculator;
use ahash::AHashMap;
use rayon::prelude::*;
use memchr::{memchr, memchr2, memchr3};
use std::sync::Arc;

const MIN_PATCH_SIZE: usize = 2;
const MAX_PATCH_SIZE: usize = 16;
const PARALLEL_THRESHOLD: usize = 20_000_000;

#[derive(Clone)]
pub struct TurboMultiSignalPatcher {
    entropy_calculator: Arc<SIMDEntropyCalculator>,
    
    word_boundaries: Vec<u8>,
    code_patterns: Vec<Vec<u8>>,
    structural_chars: [bool; 256],
    
    entropy_threshold: f32,
    compression_threshold: f32,
    semantic_weight: f32,
    
    patch_pool: Arc<std::sync::Mutex<Vec<Patch>>>,
    boundary_pool: Arc<std::sync::Mutex<Vec<Vec<usize>>>>,
}

impl TurboMultiSignalPatcher {
    pub fn new(entropy_calculator: Arc<SIMDEntropyCalculator>) -> Self {
        let mut structural_chars = [false; 256];
        
        for &ch in b"(){}[]<>\"'`,;:.!?-_/\\|*&^%$#@+=~`" {
            structural_chars[ch as usize] = true;
        }

        Self {
            entropy_calculator,
            word_boundaries: b" \t\n\r.,!?;:(){}[]\"'".to_vec(),
            code_patterns: vec![
                b"fn ".to_vec(),
                b"let ".to_vec(),
                b"const ".to_vec(),
                b"struct ".to_vec(),
                b"impl ".to_vec(),
                b"use ".to_vec(),
                b"pub ".to_vec(),
                b"async ".to_vec(),
                b"await".to_vec(),
                b"match ".to_vec(),
                b"if ".to_vec(),
                b"else".to_vec(),
                b"for ".to_vec(),
                b"while ".to_vec(),
                b"loop ".to_vec(),
                b"return ".to_vec(),
                b"println!".to_vec(),
                b"print!".to_vec(),
                b"format!".to_vec(),
                b"vec!".to_vec(),
            ],
            structural_chars,
            entropy_threshold: 0.6,
            compression_threshold: 0.4,
            semantic_weight: 0.3,
            patch_pool: Arc::new(std::sync::Mutex::new(Vec::new())),
            boundary_pool: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn patch_bytes_turbo(&mut self, bytes: &[u8]) -> Result<Vec<Patch>> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        if bytes.len() >= PARALLEL_THRESHOLD {
            self.patch_bytes_parallel(bytes)
        } else {
            self.patch_bytes_sequential(bytes)
        }
    }

    fn patch_bytes_parallel(&mut self, bytes: &[u8]) -> Result<Vec<Patch>> {
        let chunk_size = (bytes.len() / rayon::current_num_threads()).max(512);
        let overlap = MAX_PATCH_SIZE;
        
        let chunk_results: Result<Vec<_>> = (0..bytes.len())
            .step_by(chunk_size)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&start| {
                let end = (start + chunk_size + overlap).min(bytes.len());
                let chunk = &bytes[start..end];
                
                let mut chunk_patcher = self.clone();
                let mut patches = chunk_patcher.patch_bytes_sequential(chunk)?;
                
                for patch in &mut patches {
                    patch.start_pos += start;
                    patch.end_pos += start;
                }
                
                Ok(patches)
            })
            .collect();

        let chunk_patches = chunk_results?;
        
        Ok(self.merge_overlapping_patches(chunk_patches))
    }

    fn patch_bytes_sequential(&mut self, bytes: &[u8]) -> Result<Vec<Patch>> {
        let boundaries = self.find_boundaries_vectorized(bytes);
        
        let mut patches = Vec::with_capacity(boundaries.len() / 2);
        
        for window in boundaries.windows(2) {
            let start = window[0];
            let end = window[1].min(start + MAX_PATCH_SIZE);
            
            if end > start && (end - start) >= MIN_PATCH_SIZE {
                let patch_bytes = &bytes[start..end];
                let patch = self.analyze_patch_multisignal(patch_bytes, start)?;
                patches.push(patch);
            }
        }

        Ok(self.optimize_patch_boundaries(patches, bytes))
    }

    pub fn find_boundaries_vectorized(&self, bytes: &[u8]) -> Vec<usize> {
        let mut boundaries = Vec::with_capacity(bytes.len() / 4);
        boundaries.push(0);

        let mut pos = 0;
        let max_boundaries = if bytes.len() > 50000 { 500 } else { bytes.len() / 4 };
        
        while pos < bytes.len() && boundaries.len() < max_boundaries {
            if let Some(space_pos) = memchr3(b' ', b'\t', b'\n', &bytes[pos..]) {
                boundaries.push(pos + space_pos);
                pos += space_pos + 1;
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
            }
            else if let Some(punct_pos) = memchr3(b'.', b',', b';', &bytes[pos..]) {
                boundaries.push(pos + punct_pos);
                pos += punct_pos + 1;
            }
            else if let Some(struct_pos) = memchr3(b'(', b')', b'{', &bytes[pos..]) {
                boundaries.push(pos + struct_pos);
                pos += struct_pos + 1;
            }
            else {
                let jump_size = if bytes.len() > 50000 { 8 } else { 1 };
                pos += self.find_next_boundary_manual(&bytes[pos..]).unwrap_or(jump_size).max(jump_size);
            }
        }

        if boundaries.last() != Some(&bytes.len()) {
            boundaries.push(bytes.len());
        }

        boundaries
    }

    fn find_next_boundary_manual(&self, bytes: &[u8]) -> Option<usize> {
        let search_limit = if bytes.len() > 1000 { 100 } else { bytes.len() };
        
        for (i, &byte) in bytes.iter().enumerate().take(search_limit) {
            if self.structural_chars[byte as usize] || 
               self.word_boundaries.contains(&byte) {
                return Some(i);
            }

            if i % 10 == 0 && self.is_code_pattern_boundary_fast(bytes, i) {
                return Some(i);
            }
        }
        
        Some(search_limit.min(bytes.len()))
    }

    fn is_code_pattern_boundary_fast(&self, bytes: &[u8], pos: usize) -> bool {
        if pos + 3 >= bytes.len() {
            return false;
        }

        let window = &bytes[pos..];
        self.code_patterns.iter().take(5).any(|pattern| {
            window.len() >= pattern.len() && &window[..pattern.len()] == pattern.as_slice()
        })
    }

    fn analyze_patch_multisignal(&self, patch_bytes: &[u8], start_pos: usize) -> Result<Patch> {
        let entropy = self.entropy_calculator.calculate_entropy_simd(patch_bytes);
        let entropy_score = if entropy > self.entropy_threshold { 1.0 } else { entropy / self.entropy_threshold };

        let compression_score = self.estimate_compression_ratio_fast(patch_bytes);

        let semantic_score = self.calculate_semantic_score_vectorized(patch_bytes);

        let repetition_score = self.detect_repetition_patterns_fast(patch_bytes);

        let structural_score = self.calculate_structural_complexity_fast(patch_bytes);

        let complexity_score = 
            entropy_score * 0.3 +
            compression_score * 0.2 +
            semantic_score * self.semantic_weight +
            repetition_score * 0.15 +
            structural_score * 0.05;

        let patch_type = if structural_score > 0.7 {
            PatchType::Structural
        } else if semantic_score > 0.6 {
            PatchType::Semantic
        } else {
            PatchType::Complex
        };

        Ok(Patch {
            bytes: patch_bytes.to_vec(),
            start_pos,
            end_pos: start_pos + patch_bytes.len(),
            complexity_score,
            patch_type,
        })
    }

    #[inline(always)]
    fn estimate_compression_ratio_fast(&self, bytes: &[u8]) -> f32 {
        if bytes.len() < 4 {
            return 1.0;
        }

        let mut unique_bigrams = 0;
        let mut total_bigrams = 0;
        let mut seen = [false; 256 * 256];

        for window in bytes.windows(2) {
            let bigram = (window[0] as usize) << 8 | (window[1] as usize);
            if !seen[bigram] {
                seen[bigram] = true;
                unique_bigrams += 1;
            }
            total_bigrams += 1;
        }

        if total_bigrams == 0 {
            1.0
        } else {
            (unique_bigrams as f32 / total_bigrams as f32).min(1.0)
        }
    }

    fn calculate_semantic_score_vectorized(&self, bytes: &[u8]) -> f32 {
        let mut score = 0.0;
        let total_len = bytes.len() as f32;

        if total_len == 0.0 {
            return 0.0;
        }

        let boundary_count = bytes.iter()
            .filter(|&&b| self.word_boundaries.contains(&b))
            .count() as f32;

        score += (boundary_count / total_len) * 0.5;

        let code_pattern_count = self.code_patterns.iter()
            .take(5)
            .filter(|pattern| {
                let pattern_slice = pattern.as_slice();
                if pattern_slice.len() > bytes.len() {
                    false
                } else {
                    bytes.windows(pattern_slice.len())
                        .take(100)
                        .any(|window| window == pattern_slice)
                }
            })
            .count() as f32;

        score += (code_pattern_count / 5.0) * 0.3;

        let (alpha_count, _digit_count) = bytes.iter()
            .fold((0, 0), |(alpha, digit), &b| {
                if b.is_ascii_alphabetic() {
                    (alpha + 1, digit)
                } else if b.is_ascii_digit() {
                    (alpha, digit + 1)
                } else {
                    (alpha, digit)
                }
            });
        
        let alpha_ratio = alpha_count as f32 / total_len;
        score += alpha_ratio * 0.2;

        score.min(1.0)
    }

    #[inline(always)]
    fn detect_repetition_patterns_fast(&self, bytes: &[u8]) -> f32 {
        if bytes.len() < 4 {
            return 0.0;
        }

        let mut repetition_score = 0.0;
        let len = bytes.len();

        let mut consecutive_matches = 0;
        for window in bytes.windows(2) {
            if window[0] == window[1] {
                consecutive_matches += 1;
            }
        }
        repetition_score += (consecutive_matches as f32 / len as f32) * 0.3;

        let max_checks = if len > 1000 { 50 } else { len / 4 };
        
        for pattern_size in 2..=4.min(len / 2) {
            let mut matches = 0;
            let max_start = (len - pattern_size * 2).min(max_checks);
            
            let mut checked = 0;
            for i in 0..max_start {
                let pattern = &bytes[i..i + pattern_size];
                let next_segment = &bytes[i + pattern_size..i + pattern_size * 2];
                
                if pattern == next_segment {
                    matches += 1;
                    if matches > 10 && checked > 20 {
                        break;
                    }
                }
                checked += 1;
            }
            
            if max_start > 0 {
                repetition_score += (matches as f32 / max_start as f32) * 0.2;
            }
        }

        repetition_score.min(1.0)
    }

    #[inline(always)]
    fn calculate_structural_complexity_fast(&self, bytes: &[u8]) -> f32 {
        let structural_count = bytes.iter()
            .filter(|&&b| self.structural_chars[b as usize])
            .count();

        (structural_count as f32 / bytes.len() as f32).min(1.0)
    }

    fn merge_overlapping_patches(&self, chunk_patches: Vec<Vec<Patch>>) -> Vec<Patch> {
        let mut all_patches: Vec<Patch> = chunk_patches.into_iter().flatten().collect();
        
        all_patches.sort_by_key(|p| p.start_pos);
        
        let mut merged: Vec<Patch> = Vec::with_capacity(all_patches.len());
        
        for patch in all_patches {
            if let Some(last) = merged.last_mut() {
                if patch.start_pos < last.end_pos {
                    if patch.complexity_score > last.complexity_score {
                        *last = patch;
                    }
                } else {
                    merged.push(patch);
                }
            } else {
                merged.push(patch);
            }
        }
        
        merged
    }

    fn optimize_patch_boundaries(&self, mut patches: Vec<Patch>, bytes: &[u8]) -> Vec<Patch> {
        for patch in &mut patches {
            self.adjust_to_word_boundary(patch, bytes);
        }
        
        patches
    }

    fn adjust_to_word_boundary(&self, patch: &mut Patch, bytes: &[u8]) -> bool {
        let original_start = patch.start_pos;
        let original_end = patch.end_pos;
        
        // Look backwards for word boundary
        if original_start > 0 {
            for i in (0..original_start.min(4)).rev() {
                let pos = original_start - 1 - i;
                if self.word_boundaries.contains(&bytes[pos]) {
                    patch.start_pos = pos + 1;
                    break;
                }
            }
        }
        
        if original_end < bytes.len() {
            for i in 0..(bytes.len() - original_end).min(4) {
                let pos = original_end + i;
                if self.word_boundaries.contains(&bytes[pos]) {
                    patch.end_pos = pos;
                    break;
                }
            }
        }
        
        if patch.start_pos != original_start || patch.end_pos != original_end {
            patch.bytes = bytes[patch.start_pos..patch.end_pos].to_vec();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_turbo_patcher() {
        let entropy_calc = Arc::new(SIMDEntropyCalculator::new());
        let mut patcher = TurboMultiSignalPatcher::new(entropy_calc);
        
        let text = b"Hello world! This is a test.";
        let patches = patcher.patch_bytes_turbo(text).unwrap();
        
        assert!(!patches.is_empty());
        assert!(patches.iter().all(|p| !p.bytes.is_empty()));
    }

    #[test]
    fn test_vectorized_boundaries() {
        let entropy_calc = Arc::new(SIMDEntropyCalculator::new());
        let patcher = TurboMultiSignalPatcher::new(entropy_calc);
        
        let text = b"fn main() { println!(\"Hello\"); }";
        let boundaries = patcher.find_boundaries_vectorized(text);
        
        assert!(boundaries.len() > 1);
        assert_eq!(boundaries[0], 0);
        assert_eq!(*boundaries.last().unwrap(), text.len());
    }
} 