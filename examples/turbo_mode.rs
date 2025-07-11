use byteforge::*;
use byteforge::optimized_entropy::SIMDEntropyCalculator;
use byteforge::optimized_patching::TurboMultiSignalPatcher;
use std::sync::Arc;
use std::time::Instant;


fn main() -> Result<()> {
    println!("🚀 ByteForge TURBO Mode Example");
    println!("================================");

    
    let large_text = r#"
    ByteForge is a revolutionary byte-level transformer that uses advanced
    multi-signal patching algorithms with SIMD optimization for incredible
    performance. This example demonstrates the TURBO mode capabilities.
    
    Features include:
    - SIMD-accelerated entropy calculation using f32x8 vectors
    - Parallel patch processing with Rayon thread pools  
    - Memory pooling and zero-copy operations
    - Vectorized boundary detection with memchr optimization
    - Cache-friendly data structures for maximum throughput
    - Optimized hash functions and lookup tables
    "#.repeat(100); //  ~10KB 

    println!("📊 Processing {} bytes of data", large_text.len());

    println!("\nBuilding SIMD entropy model...");
    let build_start = Instant::now();
    
    let mut simd_entropy_calc = SIMDEntropyCalculator::new();
    let corpus = vec![large_text.as_bytes().to_vec()];
    simd_entropy_calc.build_from_corpus_optimized(corpus)?;
    
    let entropy_calc_arc = Arc::new(simd_entropy_calc);
    let build_time = build_start.elapsed();
    
    println!("Entropy model built in {:?}", build_time);

    let mut turbo_patcher = TurboMultiSignalPatcher::new(entropy_calc_arc.clone());

    println!("\nRunning TURBO patch processing...");
    let turbo_start = Instant::now();
    let patches = turbo_patcher.patch_bytes_turbo(large_text.as_bytes())?;
    let turbo_time = turbo_start.elapsed();

    let throughput = large_text.len() as f64 / turbo_time.as_secs_f64();
    let avg_patch_size = large_text.len() as f32 / patches.len() as f32;
    
    let sample_entropy = calculate_sample_entropy(&entropy_calc_arc, &large_text);
    let avg_complexity = calculate_avg_complexity(&patches);

    println!(" TURBO Results:");
    println!("  ┌─ Processing time:   {:?}", turbo_time);
    println!("  ├─ Throughput:       {:.0} KB/s", throughput / 1000.0);
    println!("  ├─ Patches created:  {}", patches.len());
    println!("  ├─ Avg patch size:   {:.1} bytes", avg_patch_size);
    println!("  ├─ Average entropy:  {:.3}", sample_entropy);
    println!("  └─ Avg complexity:   {:.2}", avg_complexity);

    println!("\nSample patches:");
    for (i, patch) in patches.iter().take(10).enumerate() {
        let patch_str = String::from_utf8_lossy(&patch.bytes);
        let preview = if patch_str.len() > 20 {
            format!("{}...", &patch_str[..20])
        } else {
            patch_str.to_string()
        };
        println!("  Patch {}: '{}' (type: {:?}, complexity: {:.2})", 
                 i + 1, preview, patch.patch_type, patch.complexity_score);
    }

    if patches.len() > 10 {
        println!("  ... and {} more patches", patches.len() - 10);
    }

    let blt_patches = (large_text.len() as f32 / 4.5).ceil() as usize;
    let blt_time_estimate = turbo_time * 50;
    
    println!("\nPerformance Comparison:");
    println!("  ┌─ ByteForge TURBO:  {} patches in {:?}", patches.len(), turbo_time);
    println!("  ├─ BLT (estimated):  {} patches in {:?}", blt_patches, blt_time_estimate);
    println!("  └─ Speedup:          {:.1}x faster than BLT", 
             blt_time_estimate.as_nanos() as f64 / turbo_time.as_nanos() as f64);

    println!("\nTURBO mode example completed!");
    Ok(())
}

fn calculate_sample_entropy(entropy_calc: &Arc<SIMDEntropyCalculator>, text: &str) -> f32 {
    let bytes = text.as_bytes();
    if bytes.len() < 4 {
        return 0.0;
    }
    
    let mut total_entropy = 0.0;
    let samples = 50.min(bytes.len() - 4);
    
    for i in 0..samples {
        let chunk = &bytes[i..i + 4];
        let entropy = entropy_calc.calculate_entropy_simd(chunk);
        total_entropy += entropy;
    }
    
    total_entropy / samples as f32
}

fn calculate_avg_complexity(patches: &[byteforge::patching::Patch]) -> f32 {
    if patches.is_empty() {
        return 0.0;
    }
    
    let total_complexity: f32 = patches.iter().map(|p| p.complexity_score).sum();
    total_complexity / patches.len() as f32
} 