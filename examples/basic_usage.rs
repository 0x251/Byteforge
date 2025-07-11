use byteforge::*;
use byteforge::patching::MultiSignalPatcher;
use byteforge::entropy::UltraFastEntropyCalculator;
use std::time::Instant;


fn main() -> Result<()> {
    println!("🚀 ByteForge Basic Usage Example");
    println!("=================================");

    
    let config = ByteForgeConfig {
        patch_size_range: (2, 12),
        entropy_threshold: 0.6,
        compression_threshold: 0.4,
        semantic_weight: 0.3,
        model_dim: 256,
        num_heads: 8,
        num_layers: 4,
        vocab_size: 256,
        max_seq_len: 2048,
        use_quantization: true,
        use_streaming: false,
    };

    
    let repetitive_text = "abc123".repeat(10);
    let examples = vec![
        ("Simple Text", "Hello, world! This is a simple example."),
        ("Code Sample", r#"
fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
"#),
        ("JSON Data", r#"{"name": "ByteForge", "fast": true, "version": "0.1.0"}"#),
        ("Repetitive", &repetitive_text),
    ];

    for (name, text) in examples {
        println!("\n Processing: {}", name);
        println!("Input: {}", text);
        
        
        let mut patcher = MultiSignalPatcher::new(config.clone());
        let mut entropy_calc = UltraFastEntropyCalculator::new();
        
        
        let corpus = vec![text.as_bytes().to_vec()];
        entropy_calc.build_from_corpus(corpus)?;
        
        
        let start = Instant::now();
        let patches = patcher.patch_bytes(text.as_bytes())?;
        let duration = start.elapsed();
        
        println!(" Created {} patches in {:?}", patches.len(), duration);
        
        for (i, patch) in patches.iter().enumerate() {
            let patch_str = String::from_utf8_lossy(&patch.bytes);
            println!("  Patch {}: '{}' (type: {:?}, complexity: {:.2})", 
                     i + 1, patch_str, patch.patch_type, patch.complexity_score);
        }
        
        
        let fixed_patches = (text.len() as f32 / 4.0).ceil() as usize;
        let efficiency = fixed_patches as f32 / patches.len() as f32;
        println!("⚡ Efficiency vs 4-byte patches: {:.1}x", efficiency);
    }

    println!("\n Example completed successfully!");
    Ok(())
} 