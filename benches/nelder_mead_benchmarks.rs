use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::prelude::*;
use rfkit::{
    minimize::{
        Minimizer, MultiDimFn, NelderMead, NelderMeadBounded, NelderMeadBoundedOptions,
        NelderMeadOptions, Powell,
    },
    pts::{Points, Points1, Pts},
};

// ============================================================================
// Standard Test Functions for Optimization Benchmarks
// ============================================================================

/// Sphere function: f(x) = sum(x_i^2)
/// Global minimum: f(0, 0, ..., 0) = 0
fn sphere(x: &Points1<f64>) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Rosenbrock function: f(x) = sum_{i=0}^{n-2} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
/// Global minimum: f(1, 1, ..., 1) = 0
fn rosenbrock(x: &Points1<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let term1 = x[i + 1] - x[i] * x[i];
        let term2 = 1.0 - x[i];
        sum += 100.0 * term1 * term1 + term2 * term2;
    }
    sum
}

/// Rastrigin function: f(x) = 10n + sum_{i=1}^{n} [x_i^2 - 10*cos(2*pi*x_i)]
/// Global minimum: f(0, 0, ..., 0) = 0
fn rastrigin(x: &Points1<f64>) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x
        .iter()
        .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
        .sum();
    10.0 * n + sum
}

/// Ackley function (2D): A challenging function with many local minima
/// Global minimum: f(0, 0) = 0
fn ackley(x: &Points1<f64>) -> f64 {
    let n = x.len() as f64;
    let sum1: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum2: f64 = x
        .iter()
        .map(|xi| (2.0 * std::f64::consts::PI * xi).cos())
        .sum();

    -20.0 * (-0.2 * (sum1 / n).sqrt()).exp() - (sum2 / n).exp() + 20.0 + std::f64::consts::E
}

/// Himmelblau's function (2D): f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
/// Has four equal minima
fn himmelblau(x: &Points1<f64>) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    (x0 * x0 + x1 - 11.0).powi(2) + (x0 + x1 * x1 - 7.0).powi(2)
}

/// Beale's function (2D): Tests optimizer behavior in narrow curved valley
/// Global minimum: f(3, 0.5) = 0
fn beale(x: &Points1<f64>) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    (1.5 - x0 + x0 * x1).powi(2)
        + (2.25 - x0 + x0 * x1 * x1).powi(2)
        + (2.625 - x0 + x0 * x1 * x1 * x1).powi(2)
}

/// Booth's function (2D): f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
/// Global minimum: f(1, 3) = 0
fn booth(x: &Points1<f64>) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    (x0 + 2.0 * x1 - 7.0).powi(2) + (2.0 * x0 + x1 - 5.0).powi(2)
}

// ============================================================================
// Nelder-Mead Benchmarks
// ============================================================================

fn bench_nelder_mead_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder_mead/sphere");

    for &dim in &[2, 5, 10, 20] {
        let init = Points(Array1::from_elem(dim, 5.0));
        let scale = Points1::ones(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                let options = NelderMeadOptions::new(
                    &init,
                    Some(&scale),
                    Some(500),
                    Some(1e-8),
                    None,
                    None,
                    None,
                    None,
                    Some(0),
                );
                let objective = MultiDimFn::new(sphere);
                let mut minimizer = NelderMead::new(objective);
                black_box(minimizer.minimize(&options))
            })
        });
    }

    group.finish();
}

fn bench_nelder_mead_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder_mead/rosenbrock");

    for &dim in &[2, 5, 10] {
        let init = Points(Array1::from_elem(dim, 0.0));
        let scale = Points1::ones(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                let options = NelderMeadOptions::new(
                    &init,
                    Some(&scale),
                    Some(2000),
                    Some(1e-8),
                    None,
                    None,
                    None,
                    None,
                    Some(0),
                );
                let objective = MultiDimFn::new(rosenbrock);
                let mut minimizer = NelderMead::new(objective);
                black_box(minimizer.minimize(&options))
            })
        });
    }

    group.finish();
}

fn bench_nelder_mead_2d_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder_mead/2d_functions");

    let init = array![0.0, 0.0].into();
    let scale = Points1::ones(2);

    group.bench_function("himmelblau", |b| {
        b.iter(|| {
            let options = NelderMeadOptions::new(
                &init,
                Some(&scale),
                Some(500),
                Some(1e-8),
                None,
                None,
                None,
                None,
                Some(0),
            );
            let objective = MultiDimFn::new(himmelblau);
            let mut minimizer = NelderMead::new(objective);
            black_box(minimizer.minimize(&options))
        })
    });

    group.bench_function("beale", |b| {
        let init = array![0.0, 0.0].into();
        b.iter(|| {
            let options = NelderMeadOptions::new(
                &init,
                Some(&scale),
                Some(500),
                Some(1e-8),
                None,
                None,
                None,
                None,
                Some(0),
            );
            let objective = MultiDimFn::new(beale);
            let mut minimizer = NelderMead::new(objective);
            black_box(minimizer.minimize(&options))
        })
    });

    group.bench_function("booth", |b| {
        let init = array![0.0, 0.0].into();
        b.iter(|| {
            let options = NelderMeadOptions::new(
                &init,
                Some(&scale),
                Some(500),
                Some(1e-8),
                None,
                None,
                None,
                None,
                Some(0),
            );
            let objective = MultiDimFn::new(booth);
            let mut minimizer = NelderMead::new(objective);
            black_box(minimizer.minimize(&options))
        })
    });

    group.bench_function("ackley", |b| {
        let init = array![2.0, 2.0].into();
        b.iter(|| {
            let options = NelderMeadOptions::new(
                &init,
                Some(&scale),
                Some(500),
                Some(1e-8),
                None,
                None,
                None,
                None,
                Some(0),
            );
            let objective = MultiDimFn::new(ackley);
            let mut minimizer = NelderMead::new(objective);
            black_box(minimizer.minimize(&options))
        })
    });

    group.finish();
}

// ============================================================================
// Nelder-Mead Bounded Benchmarks
// ============================================================================

fn bench_nelder_mead_bounded_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder_mead_bounded/sphere");

    for &dim in &[2, 5, 10, 20] {
        let init = Points(Array1::from_elem(dim, 5.0));
        let scale = Points1::ones(dim);
        let lb = Points(Array1::from_elem(dim, -10.0));
        let ub = Points(Array1::from_elem(dim, 10.0));

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                let options = NelderMeadBoundedOptions::new(
                    &init,
                    Some(&scale),
                    Some(&lb),
                    Some(&ub),
                    Some(500),
                    Some(1e-8),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(0),
                );
                let objective = MultiDimFn::new(sphere);
                let mut minimizer = NelderMeadBounded::new(objective);
                black_box(minimizer.minimize(&options))
            })
        });
    }

    group.finish();
}

fn bench_nelder_mead_bounded_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder_mead_bounded/rosenbrock");

    for &dim in &[2, 5, 10] {
        let init = Points(Array1::from_elem(dim, 0.0));
        let scale = Points1::ones(dim);
        let lb = Points(Array1::from_elem(dim, -5.0));
        let ub = Points(Array1::from_elem(dim, 5.0));

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                let options = NelderMeadBoundedOptions::new(
                    &init,
                    Some(&scale),
                    Some(&lb),
                    Some(&ub),
                    Some(2000),
                    Some(1e-8),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(0),
                );
                let objective = MultiDimFn::new(rosenbrock);
                let mut minimizer = NelderMeadBounded::new(objective);
                black_box(minimizer.minimize(&options))
            })
        });
    }

    group.finish();
}

fn bench_nelder_mead_bounded_with_anti_stagnation(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder_mead_bounded/anti_stagnation");

    let dim = 5;
    let init = Points(Array1::from_elem(dim, 2.0));
    let scale = Points1::ones(dim);
    let lb = Points(Array1::from_elem(dim, -5.12));
    let ub = Points(Array1::from_elem(dim, 5.12));

    group.bench_function("rastrigin_basic", |b| {
        b.iter(|| {
            let options = NelderMeadBoundedOptions::new(
                &init,
                Some(&scale),
                Some(&lb),
                Some(&ub),
                Some(1000),
                Some(1e-8),
                None,
                None,
                None,
                None,
                None,
                Some(0),
            );
            let objective = MultiDimFn::new(rastrigin);
            let mut minimizer = NelderMeadBounded::new(objective);
            black_box(minimizer.minimize(&options))
        })
    });

    group.bench_function("rastrigin_anti_stagnation", |b| {
        b.iter(|| {
            let mut options = NelderMeadBoundedOptions::new(
                &init,
                Some(&scale),
                Some(&lb),
                Some(&ub),
                Some(1000),
                Some(1e-8),
                None,
                None,
                None,
                None,
                None,
                Some(0),
            );
            options.set_random_restart(true);
            options.set_adaptive_simplex(true);
            options.enable_anti_stagnation(Some(3), Some(10), Some(0.1));

            let objective = MultiDimFn::new(rastrigin);
            let mut minimizer = NelderMeadBounded::new(objective);
            black_box(minimizer.minimize(&options))
        })
    });

    group.finish();
}

// ============================================================================
// Powell Benchmarks
// ============================================================================

fn bench_powell_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("powell/sphere");

    for &dim in &[2, 5, 10, 20] {
        let init = Points(Array1::from_elem(dim, 5.0));

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                let objective = MultiDimFn::new(sphere);
                let mut minimizer = Powell::new(objective);
                black_box(minimizer.powell_method(&init, Some(1e-8), Some(500), None))
            })
        });
    }

    group.finish();
}

fn bench_powell_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("powell/rosenbrock");

    for &dim in &[2, 5, 10] {
        let init = Points(Array1::from_elem(dim, 0.0));

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| {
                let objective = MultiDimFn::new(rosenbrock);
                let mut minimizer = Powell::new(objective);
                black_box(minimizer.powell_method(&init, Some(1e-8), Some(2000), None))
            })
        });
    }

    group.finish();
}

fn bench_powell_2d_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("powell/2d_functions");

    group.bench_function("himmelblau", |b| {
        let init = array![0.0, 0.0].into();
        b.iter(|| {
            let objective = MultiDimFn::new(himmelblau);
            let mut minimizer = Powell::new(objective);
            black_box(minimizer.powell_method(&init, Some(1e-8), Some(500), None))
        })
    });

    group.bench_function("beale", |b| {
        let init = array![0.0, 0.0].into();
        b.iter(|| {
            let objective = MultiDimFn::new(beale);
            let mut minimizer = Powell::new(objective);
            black_box(minimizer.powell_method(&init, Some(1e-8), Some(500), None))
        })
    });

    group.bench_function("booth", |b| {
        let init = array![0.0, 0.0].into();
        b.iter(|| {
            let objective = MultiDimFn::new(booth);
            let mut minimizer = Powell::new(objective);
            black_box(minimizer.powell_method(&init, Some(1e-8), Some(500), None))
        })
    });

    group.bench_function("ackley", |b| {
        let init = array![2.0, 2.0].into();
        b.iter(|| {
            let objective = MultiDimFn::new(ackley);
            let mut minimizer = Powell::new(objective);
            black_box(minimizer.powell_method(&init, Some(1e-8), Some(500), None))
        })
    });

    group.finish();
}

// ============================================================================
// Comparison Benchmarks (same problem, different methods)
// ============================================================================

fn bench_method_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison/rosenbrock_5d");

    let dim = 5;
    let init = Points(Array1::from_elem(dim, 0.0));
    let scale = Points1::ones(dim);
    let lb = Points(Array1::from_elem(dim, -5.0));
    let ub = Points(Array1::from_elem(dim, 5.0));

    group.bench_function("nelder_mead", |b| {
        b.iter(|| {
            let options = NelderMeadOptions::new(
                &init,
                Some(&scale),
                Some(1000),
                Some(1e-8),
                None,
                None,
                None,
                None,
                Some(0),
            );
            let objective = MultiDimFn::new(rosenbrock);
            let mut minimizer = NelderMead::new(objective);
            black_box(minimizer.minimize(&options))
        })
    });

    group.bench_function("nelder_mead_bounded", |b| {
        b.iter(|| {
            let options = NelderMeadBoundedOptions::new(
                &init,
                Some(&scale),
                Some(&lb),
                Some(&ub),
                Some(1000),
                Some(1e-8),
                None,
                None,
                None,
                None,
                None,
                Some(0),
            );
            let objective = MultiDimFn::new(rosenbrock);
            let mut minimizer = NelderMeadBounded::new(objective);
            black_box(minimizer.minimize(&options))
        })
    });

    group.bench_function("powell", |b| {
        b.iter(|| {
            let objective = MultiDimFn::new(rosenbrock);
            let mut minimizer = Powell::new(objective);
            black_box(minimizer.powell_method(&init, Some(1e-8), Some(1000), None))
        })
    });

    group.finish();
}

fn bench_iteration_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/iterations");

    let dim = 5;
    let init = Points(Array1::from_elem(dim, 5.0));
    let scale = Points1::ones(dim);

    for &max_iters in &[10, 50, 100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("nelder_mead", max_iters),
            &max_iters,
            |b, &iters| {
                b.iter(|| {
                    let options = NelderMeadOptions::new(
                        &init,
                        Some(&scale),
                        Some(iters),
                        Some(1e-8),
                        None,
                        None,
                        None,
                        None,
                        Some(0),
                    );
                    let objective = MultiDimFn::new(sphere);
                    let mut minimizer = NelderMead::new(objective);
                    black_box(minimizer.minimize(&options))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_nelder_mead_sphere,
    bench_nelder_mead_rosenbrock,
    bench_nelder_mead_2d_functions,
    bench_nelder_mead_bounded_sphere,
    bench_nelder_mead_bounded_rosenbrock,
    bench_nelder_mead_bounded_with_anti_stagnation,
    bench_powell_sphere,
    bench_powell_rosenbrock,
    bench_powell_2d_functions,
    bench_method_comparison,
    bench_iteration_scaling,
);

criterion_main!(benches);
