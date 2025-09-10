// File: benches/benchmark_analysis.rs
// Utility script to analyze benchmark results

use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug)]
struct BenchmarkResult {
    function_name: String,
    iterations: u32,
    dimensions: u32,
    mean_time_ns: f64,
    std_dev_ns: f64,
    // throughput: Option<f64>,
}

pub fn analyze_benchmark_results() {
    let target_dir = "target/criterion";

    if !Path::new(target_dir).exists() {
        println!("No benchmark results found. Run benchmarks first with:");
        println!("cargo bench");
        return;
    }

    println!("Analyzing benchmark results...\n");

    // Parse criterion output files
    let results = parse_criterion_results(target_dir);

    // Generate analysis reports
    print_performance_summary(&results);
    print_scaling_analysis(&results);
    print_convergence_analysis(&results);

    println!("\nFor detailed HTML reports, check: target/criterion/reports/index.html");
}

fn parse_criterion_results(base_path: &str) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // This is a simplified parser - criterion generates complex JSON structures
    // In practice, you might want to use the criterion output files directly

    if let Ok(entries) = fs::read_dir(base_path) {
        for entry in entries.flatten() {
            if entry.file_type().unwrap().is_dir() {
                let group_path = entry.path();
                if let Some(group_name) = group_path.file_name() {
                    parse_group_results(
                        &group_path,
                        group_name.to_string_lossy().as_ref(),
                        &mut results,
                    );
                }
            }
        }
    }

    results
}

fn parse_group_results(group_path: &Path, group_name: &str, results: &mut Vec<BenchmarkResult>) {
    let estimates_path = group_path.join("base/estimates.json");

    if estimates_path.exists() {
        if let Ok(content) = fs::read_to_string(&estimates_path) {
            if let Ok(data) = serde_json::from_str::<Value>(&content) {
                if let Some(mean) = data["mean"]["point_estimate"].as_f64() {
                    if let Some(std_dev) = data["std_dev"]["point_estimate"].as_f64() {
                        // Parse group name to extract function info
                        let (function_name, iterations, dimensions) =
                            parse_benchmark_name(group_name);

                        results.push(BenchmarkResult {
                            function_name,
                            iterations,
                            dimensions,
                            mean_time_ns: mean,
                            std_dev_ns: std_dev,
                            // throughput: None, // Could be extracted from throughput.json
                        });
                    }
                }
            }
        }
    }
}

fn parse_benchmark_name(name: &str) -> (String, u32, u32) {
    // Parse names like "nelder_mead_10_iterations/optimization/sphere_2d"
    let parts: Vec<&str> = name.split('/').collect();

    let iterations = if name.contains("1_iteration") {
        1
    } else if name.contains("10_iteration") {
        10
    } else if name.contains("100_iteration") {
        100
    } else {
        0
    };

    let function_name = parts.last().unwrap_or(&name).to_string();

    let dimensions = if function_name.contains("2d") {
        2
    } else if function_name.contains("5d") {
        5
    } else if function_name.contains("10d") {
        10
    } else if function_name.contains("20d") {
        20
    } else {
        0
    };

    (function_name, iterations, dimensions)
}

fn print_performance_summary(results: &[BenchmarkResult]) {
    println!("=== PERFORMANCE SUMMARY ===");
    println!(
        "{:<20} {:<6} {:<5} {:<15} {:<15}",
        "Function", "Iter", "Dim", "Mean Time (ms)", "Std Dev (ms)"
    );
    println!("{}", "-".repeat(70));

    for result in results {
        println!(
            "{:<20} {:<6} {:<5} {:<15.3} {:<15.3}",
            result.function_name,
            result.iterations,
            result.dimensions,
            result.mean_time_ns / 1_000_000.0, // Convert to ms
            result.std_dev_ns / 1_000_000.0
        );
    }
    println!();
}

fn print_scaling_analysis(results: &[BenchmarkResult]) {
    println!("=== SCALING ANALYSIS ===");

    // Group by function name
    let mut function_groups: std::collections::HashMap<String, Vec<&BenchmarkResult>> =
        std::collections::HashMap::new();

    for result in results {
        function_groups
            .entry(result.function_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    for (function_name, mut group) in function_groups {
        group.sort_by_key(|r| r.iterations);

        println!("Function: {}", function_name);

        if group.len() >= 2 {
            let first = group[0];
            let last = group[group.len() - 1];

            if first.mean_time_ns > 0.0 && first.iterations > 0 && last.iterations > 0 {
                let time_ratio = last.mean_time_ns / first.mean_time_ns;
                let iter_ratio = last.iterations as f64 / first.iterations as f64;
                let scaling_factor = time_ratio.ln() / iter_ratio.ln();

                println!("  Time scaling: O(n^{:.2})", scaling_factor);
                println!(
                    "  {} iter: {:.2} ms -> {} iter: {:.2} ms ({}x speedup per iteration)",
                    first.iterations,
                    first.mean_time_ns / 1_000_000.0,
                    last.iterations,
                    last.mean_time_ns / 1_000_000.0,
                    time_ratio / iter_ratio
                );
            }
        }
        println!();
    }
}

fn print_convergence_analysis(results: &[BenchmarkResult]) {
    println!("=== CONVERGENCE ANALYSIS ===");

    // Group by dimension and analyze iteration scaling
    let mut dim_groups: std::collections::HashMap<u32, Vec<&BenchmarkResult>> =
        std::collections::HashMap::new();

    for result in results {
        if result.dimensions > 0 {
            dim_groups
                .entry(result.dimensions)
                .or_insert_with(Vec::new)
                .push(result);
        }
    }

    for (dimensions, mut group) in dim_groups {
        group.sort_by_key(|r| r.iterations);

        println!("Dimension: {}", dimensions);

        for result in &group {
            println!(
                "  {} iterations: {:.2} ms ± {:.2} ms",
                result.iterations,
                result.mean_time_ns / 1_000_000.0,
                result.std_dev_ns / 1_000_000.0
            );
        }

        // Calculate efficiency (time per iteration)
        if let Some(result_100) = group.iter().find(|r| r.iterations == 100) {
            let efficiency = result_100.mean_time_ns / (100.0 * 1_000_000.0); // ms per iteration
            println!("  Efficiency: {:.3} ms/iteration", efficiency);
        }

        println!();
    }
}

// Run analysis if executed directly
fn main() {
    analyze_benchmark_results();
}
