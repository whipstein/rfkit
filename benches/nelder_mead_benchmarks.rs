// File: benches/nelder_mead_benchmarks.rs

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ndarray::prelude::*;
use std::time::Duration;

// Import your crate modules - adjust the crate name as needed
use rfkit_base_ndarray::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum TestType {
    NelderMead,
    NelderMeadBounded,
}

impl TestType {
    pub fn name(&self) -> &'static str {
        match self {
            TestType::NelderMead => "NelderMead",
            TestType::NelderMeadBounded => "NelderMeadBounded",
        }
    }

    pub fn all_types() -> Vec<TestType> {
        vec![TestType::NelderMead, TestType::NelderMeadBounded]
    }

    pub fn _supports_bounds(&self) -> bool {
        matches!(self, TestType::NelderMeadBounded)
    }
}

#[derive(Clone)]
struct TestProblem {
    name: String,
    function: fn(Array1<MyFloat>) -> MyFloat,
    x0: Array1<f64>,
    scale: Array1<f64>,
    lb: Array1<f64>,
    ub: Array1<f64>,
    expected_min: f64,
    dimensions: usize,
}

// Test Functions
fn sphere_function(x: Array1<MyFloat>) -> MyFloat {
    x.iter().map(|xi| xi * xi).sum()
}

fn rosenbrock_function(x: Array1<MyFloat>) -> MyFloat {
    if x.len() < 2 {
        return MyFloat::new(0.0);
    }

    let mut sum = MyFloat::new(0.0);
    for i in 0..x.len() - 1 {
        let term1 = MyFloat::new(100.0) * (&x[i + 1] - &x[i] * &x[i]) * (&x[i + 1] - &x[i] * &x[i]);
        let term2 = (MyFloat::new(1.0) - &x[i]) * (MyFloat::new(1.0) - &x[i]);
        sum += term1 + term2;
    }
    sum
}

fn rastrigin_function(x: Array1<MyFloat>) -> MyFloat {
    let a = MyFloat::new(10.0);
    let n = MyFloat::new(x.len() as f64);
    let pi = MyFloat::new(std::f64::consts::PI);

    let mut sum = MyFloat::new(0.0);
    for xi in x.iter() {
        sum += xi * xi - &a * (2.0 * &pi * xi).cos();
    }
    &a * &n + sum
}

fn ackley_function(x: Array1<MyFloat>) -> MyFloat {
    let a = MyFloat::new(20.0);
    let b = MyFloat::new(0.2);
    let c = MyFloat::new(2.0 * std::f64::consts::PI);
    let n = MyFloat::new(x.len() as f64);
    let e = MyFloat::new(std::f64::consts::E);

    let sum1: MyFloat = x.iter().map(|xi| xi * xi).sum();
    let sum2: MyFloat = x.iter().map(|xi| (&c * xi).cos()).sum();

    -&a * (-&b * (sum1 / &n).sqrt()).exp() - (sum2 / &n).exp() + &a + e
}

fn himmelblau_function(x: Array1<MyFloat>) -> MyFloat {
    if x.len() != 2 {
        // Extend to n-dimensions by summing pairs
        let mut result = MyFloat::new(0.0);
        let mut i = 0;
        while i < x.len() - 1 {
            if i + 1 < x.len() {
                let term1 = (&x[i] * &x[i] + &x[i + 1] - 11.0) * (&x[i] * &x[i] + &x[i + 1] - 11.0);
                let term2 =
                    (&x[i] + &x[i + 1] * &x[i + 1] - 7.0) * (&x[i] + &x[i + 1] * &x[i + 1] - 7.0);
                result += term1 + term2;
            }
            i += 2;
        }
        result
    } else {
        let term1 = (&x[0] * &x[0] + &x[1] - 11.0) * (&x[0] * &x[0] + &x[1] - 11.0);
        let term2 = (&x[0] + &x[1] * &x[1] - 7.0) * (&x[0] + &x[1] * &x[1] - 7.0);
        term1 + term2
    }
}

fn setup_test_problems() -> Vec<TestProblem> {
    vec![
        // 2D Problems
        TestProblem {
            name: "sphere_2d".to_string(),
            function: sphere_function,
            x0: array![1.5, 1.5],
            scale: array![1.0, 1.0],
            lb: array![-5.0, -5.0],
            ub: array![5.0, 5.0],
            expected_min: 0.0,
            dimensions: 2,
        },
        TestProblem {
            name: "rosenbrock_2d".to_string(),
            function: rosenbrock_function,
            x0: array![-1.2, 1.0],
            scale: array![1.0, 1.0],
            lb: array![-2.0, -2.0],
            ub: array![2.0, 2.0],
            expected_min: 0.0,
            dimensions: 2,
        },
        TestProblem {
            name: "himmelblau_2d".to_string(),
            function: himmelblau_function,
            x0: array![0.0, 0.0],
            scale: array![1.0, 1.0],
            lb: array![-5.0, -5.0],
            ub: array![5.0, 5.0],
            expected_min: 0.0,
            dimensions: 2,
        },
        // 3D Problems
        TestProblem {
            name: "sphere_3d".to_string(),
            function: sphere_function,
            x0: Array1::from_elem(3, 0.5),
            scale: Array1::ones(3),
            lb: Array1::from_elem(3, -3.0),
            ub: Array1::from_elem(3, 3.0),
            expected_min: 0.0,
            dimensions: 3,
        },
        // 4D Problems
        TestProblem {
            name: "sphere_4d".to_string(),
            function: sphere_function,
            x0: Array1::from_elem(4, 0.5),
            scale: Array1::ones(4),
            lb: Array1::from_elem(4, -3.0),
            ub: Array1::from_elem(4, 3.0),
            expected_min: 0.0,
            dimensions: 4,
        },
        // 5D Problems
        TestProblem {
            name: "sphere_5d".to_string(),
            function: sphere_function,
            x0: Array1::from_elem(5, 0.5),
            scale: Array1::ones(5),
            lb: Array1::from_elem(5, -3.0),
            ub: Array1::from_elem(5, 3.0),
            expected_min: 0.0,
            dimensions: 5,
        },
        TestProblem {
            name: "rastrigin_5d".to_string(),
            function: rastrigin_function,
            x0: Array1::from_elem(5, 0.1),
            scale: Array1::ones(5),
            lb: Array1::from_elem(5, -5.12),
            ub: Array1::from_elem(5, 5.12),
            expected_min: 0.0,
            dimensions: 5,
        },
        // 6D Problems
        TestProblem {
            name: "sphere_6d".to_string(),
            function: sphere_function,
            x0: Array1::from_elem(6, 0.5),
            scale: Array1::ones(6),
            lb: Array1::from_elem(6, -3.0),
            ub: Array1::from_elem(6, 3.0),
            expected_min: 0.0,
            dimensions: 6,
        },
        // 7D Problems
        TestProblem {
            name: "sphere_7d".to_string(),
            function: sphere_function,
            x0: Array1::from_elem(7, 0.5),
            scale: Array1::ones(7),
            lb: Array1::from_elem(7, -3.0),
            ub: Array1::from_elem(7, 3.0),
            expected_min: 0.0,
            dimensions: 7,
        },
        // 8D Problems
        TestProblem {
            name: "sphere_8d".to_string(),
            function: sphere_function,
            x0: Array1::from_elem(8, 0.5),
            scale: Array1::ones(8),
            lb: Array1::from_elem(8, -3.0),
            ub: Array1::from_elem(8, 3.0),
            expected_min: 0.0,
            dimensions: 8,
        },
        // 9D Problems
        TestProblem {
            name: "sphere_9d".to_string(),
            function: sphere_function,
            x0: Array1::from_elem(9, 0.5),
            scale: Array1::ones(9),
            lb: Array1::from_elem(9, -3.0),
            ub: Array1::from_elem(9, 3.0),
            expected_min: 0.0,
            dimensions: 9,
        },
        // 10D Problems
        TestProblem {
            name: "sphere_10d".to_string(),
            function: sphere_function,
            x0: Array1::from_elem(10, 0.3),
            scale: Array1::ones(10),
            lb: Array1::from_elem(10, -2.0),
            ub: Array1::from_elem(10, 2.0),
            expected_min: 0.0,
            dimensions: 10,
        },
        TestProblem {
            name: "ackley_10d".to_string(),
            function: ackley_function,
            x0: Array1::from_elem(10, 0.1),
            scale: Array1::ones(10),
            lb: Array1::from_elem(10, -32.768),
            ub: Array1::from_elem(10, 32.768),
            expected_min: 0.0,
            dimensions: 10,
        },
    ]
}

fn run_optimization_benchmark(
    problem: &TestProblem,
    category: TestType,
    iterations: usize,
) -> (f64, f64, usize, Option<f64>) {
    // Add small random perturbation to starting point
    let mut x_start = problem.x0.clone();
    for i in 0..x_start.len() {
        x_start[i] += fastrand::f64() * 0.2 - 0.1; // Random perturbation ±0.1
        x_start[i] = x_start[i].max(problem.lb[i] + 1e-6);
        x_start[i] = x_start[i].min(problem.ub[i] - 1e-6);
    }

    // Create boxed optimizer for dynamic dispatch
    let mut solver: Box<dyn Minimizer> = match category {
        TestType::NelderMead => Box::new(NelderMead::new(
            x_start,
            problem.scale.clone(),
            problem.function,
        )),
        TestType::NelderMeadBounded => Box::new(NelderMeadBounded::new(
            x_start,
            problem.scale.clone(),
            problem.lb.clone(),
            problem.ub.clone(),
            problem.function,
        )),
    };

    solver.solve(iterations);

    // Calculate final function value
    let x_myfloat = Array1::from_shape_fn(solver.x().len(), |i| MyFloat::new(solver.x()[i]));
    let final_value = (problem.function)(x_myfloat).to_f64();
    let error = (final_value - problem.expected_min).abs();

    (final_value, error, solver.iterations(), solver.tolerance())
}

// Benchmark functions for each iteration count
fn bench_iterations_1(c: &mut Criterion) {
    let problems = setup_test_problems();

    let mut group = c.benchmark_group("nelder_mead_1_iteration");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for problem in &problems {
        for category in TestType::all_types() {
            group.throughput(Throughput::Elements(problem.dimensions as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("optimization_{}", category.name()), &problem.name),
                &(problem, category),
                |b, (prob, cat)| {
                    b.iter(|| {
                        let (final_value, error, iterations, tolerance) =
                            run_optimization_benchmark(
                                black_box(prob),
                                black_box(*cat),
                                black_box(1),
                            );
                        black_box((final_value, error, iterations, tolerance))
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_iterations_10(c: &mut Criterion) {
    let problems = setup_test_problems();

    let mut group = c.benchmark_group("nelder_mead_10_iterations");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);

    for problem in &problems {
        for category in TestType::all_types() {
            group.throughput(Throughput::Elements(problem.dimensions as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("optimization_{}", category.name()), &problem.name),
                &(problem, category),
                |b, (prob, cat)| {
                    b.iter(|| {
                        let (final_value, error, iterations, tolerance) =
                            run_optimization_benchmark(
                                black_box(prob),
                                black_box(*cat),
                                black_box(10),
                            );
                        black_box((final_value, error, iterations, tolerance))
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_iterations_100(c: &mut Criterion) {
    let problems = setup_test_problems();

    let mut group = c.benchmark_group("nelder_mead_100_iterations");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(20);

    for problem in &problems {
        for category in TestType::all_types() {
            group.throughput(Throughput::Elements(problem.dimensions as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("optimization_{}", category.name()), &problem.name),
                &(problem, category),
                |b, (prob, cat)| {
                    b.iter(|| {
                        let (final_value, error, iterations, tolerance) =
                            run_optimization_benchmark(
                                black_box(prob),
                                black_box(*cat),
                                black_box(100),
                            );
                        black_box((final_value, error, iterations, tolerance))
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_iterations_1000(c: &mut Criterion) {
    let problems = setup_test_problems();

    let mut group = c.benchmark_group("nelder_mead_1000_iterations");
    group.measurement_time(Duration::from_secs(600));
    group.sample_size(20);

    for problem in &problems {
        for category in TestType::all_types() {
            group.throughput(Throughput::Elements(problem.dimensions as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("optimization_{}", category.name()), &problem.name),
                &(problem, category),
                |b, (prob, cat)| {
                    b.iter(|| {
                        let (final_value, error, iterations, tolerance) =
                            run_optimization_benchmark(
                                black_box(prob),
                                black_box(*cat),
                                black_box(1000),
                            );
                        black_box((final_value, error, iterations, tolerance))
                    });
                },
            );
        }
    }
    group.finish();
}

// Scaling benchmark - compare different iteration counts on same problem
fn bench_scaling_analysis(c: &mut Criterion) {
    let problems = setup_test_problems();

    let mut group = c.benchmark_group("nelder_mead_scaling");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(20);

    // Test with sphere function in different dimensions
    let sphere_problems: Vec<_> = problems
        .iter()
        .filter(|p| p.name.starts_with("sphere"))
        .collect();

    for problem in sphere_problems {
        for category in TestType::all_types() {
            for &iterations in &[1, 10, 100] {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}_{}_{}_iter", problem.name, category.name(), iterations),
                        problem.dimensions,
                    ),
                    &(problem, category, iterations),
                    |b, (prob, cat, iter_count)| {
                        b.iter(|| {
                            let (final_value, error, iterations, tolerance) =
                                run_optimization_benchmark(
                                    black_box(prob),
                                    black_box(*cat),
                                    black_box(*iter_count),
                                );
                            black_box((final_value, error, iterations, tolerance))
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

// Convergence quality benchmark - measure final error vs iterations
fn bench_convergence_quality(c: &mut Criterion) {
    let problems = setup_test_problems();

    let mut group = c.benchmark_group("nelder_mead_convergence");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(20);

    // Test convergence quality for each problem type
    for problem in &problems {
        for category in TestType::all_types() {
            for &iterations in &[1, 10, 100] {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!(
                            "convergence_{}_{}_{}_iter",
                            problem.name,
                            category.name(),
                            iterations
                        ),
                        iterations,
                    ),
                    &(problem, category, iterations),
                    |b, (prob, cat, iter_count)| {
                        b.iter_custom(|iters| {
                            let start = std::time::Instant::now();
                            let mut total_error = 0.0;

                            for _ in 0..iters {
                                let (_, error, _, _) = run_optimization_benchmark(
                                    black_box(prob),
                                    black_box(*cat),
                                    black_box(*iter_count),
                                );
                                total_error += error;
                            }

                            // Store final error in a way that can be accessed later
                            let _avg_error = total_error / iters as f64;
                            black_box(_avg_error);

                            start.elapsed()
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

// Function dimension scaling benchmark
fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder_mead_dimensions");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(12);

    for category in TestType::all_types() {
        // Create sphere problems with varying dimensions
        for &dim in &[2, 5, 10, 20] {
            let problem = TestProblem {
                name: format!("sphere_{}d", dim),
                function: sphere_function,
                x0: Array1::from_elem(dim, 0.5),
                scale: Array1::ones(dim),
                lb: Array1::from_elem(dim, -3.0),
                ub: Array1::from_elem(dim, 3.0),
                expected_min: 0.0,
                dimensions: dim,
            };

            group.throughput(Throughput::Elements(dim as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("sphere_{}_10_iter", category.name()), dim),
                &(problem, category),
                |b, (prob, cat)| {
                    b.iter(|| {
                        let (final_value, error, iterations, tolerance) =
                            run_optimization_benchmark(
                                black_box(prob),
                                black_box(*cat),
                                black_box(10),
                            );
                        black_box((final_value, error, iterations, tolerance))
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    nelder_mead_benches,
    bench_iterations_1,
    bench_iterations_10,
    bench_iterations_100,
    bench_iterations_1000,
    bench_scaling_analysis,
    bench_convergence_quality,
    bench_dimension_scaling,
);
criterion_main!(nelder_mead_benches);
