// use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
// use ndarray::prelude::*;
// use num::complex::{Complex64, c64};
// use rfkit_base_ndarray::file::read_touchstone;
// use rfkit_base_ndarray::frequency::Frequency;
// use rfkit_base_ndarray::minimize::{NelderMead, NelderMeadBounded};
// use rfkit_base_ndarray::mycomplex::MyComplex;
// use rfkit_base_ndarray::myfloat::MyFloat;
// use rfkit_base_ndarray::network::{Network, NetworkBuilder};

// // Helper functions from your tests
// fn calc_feed_y(freq: Frequency, x: Array1<MyFloat>) -> Vec<Array2<MyComplex>> {
//     let mut yfeed: Vec<Array2<MyComplex>> = vec![
//         array![
//             [MyComplex::new(0.0, 0.0), MyComplex::new(0.0, 0.0)],
//             [MyComplex::new(0.0, 0.0), MyComplex::new(0.0, 0.0)]
//         ];
//         freq.npts()
//     ];

//     let w = freq.w();
//     let mut zs: Vec<MyComplex> = vec![];
//     let mut zm: Vec<MyComplex> = vec![];
//     let mut zp: Vec<MyComplex> = vec![];
//     let mut zall: Vec<MyComplex> = vec![];
//     for i in 0..freq.npts() {
//         zs.push(
//             &x[3] * MyComplex::new(0.0, 1.0) * w[i] * &x[2]
//                 / (&x[3] + MyComplex::new(0.0, 1.0) * w[i] * &x[2]),
//         );
//         zm.push(&x[1] + MyComplex::new(0.0, 1.0) * w[i] * &x[0] + &zs[i]);
//         zp.push(&x[5] - MyComplex::new(0.0, 1.0) / (w[i] * &x[4]));
//         zall.push(&zm[i] * &zp[i] / (&zm[i] + &zp[i]));
//     }

//     for i in 0..freq.npts() {
//         yfeed[i][(0, 1)] = -1.0 / &zall[i];
//         yfeed[i][(1, 0)] = -1.0 / &zall[i];
//         yfeed[i][(0, 0)] = MyComplex::new(0.0, 1.0) * w[i] * &x[6] + 1.0 / &zall[i];
//         yfeed[i][(1, 1)] = MyComplex::new(0.0, 1.0) * w[i] * &x[7] + 1.0 / &zall[i];
//     }

//     yfeed
// }

// fn calc_err(meas: &Network, model: &Network) -> MyFloat {
//     let mut err: f64 = 0.0;
//     let meas_h = meas.h();
//     let model_h = model.h();
//     let meas_y = meas.y();
//     let model_y = model.y();
//     let meas_z = meas.z();
//     let model_z = model.z();
//     for i in 0..meas.freq().npts() {
//         for port in [(0, 0), (0, 1), (1, 1)].iter() {
//             err += ((model_h[i][*port].re - meas_h[i][*port].re) / meas_h[i][*port].re).powi(2)
//                 + ((model_h[i][*port].im - meas_h[i][*port].im) / meas_h[i][*port].im).powi(2);
//             err += ((model_y[i][*port].re - meas_y[i][*port].re) / meas_y[i][*port].re).powi(2)
//                 + ((model_y[i][*port].im - meas_y[i][*port].im) / meas_y[i][*port].im).powi(2);
//             err += ((model_z[i][*port].re - meas_z[i][*port].re) / meas_z[i][*port].re).powi(2)
//                 + ((model_z[i][*port].im - meas_z[i][*port].im) / meas_z[i][*port].im).powi(2);
//         }
//     }

//     MyFloat::new(err)
// }

// fn eval_f_simplex(x: Array1<MyFloat>, meas: &Network) -> MyFloat {
//     let model_y = calc_feed_y(meas.freq().clone(), x);
//     let model_y_c64: Vec<Array2<Complex64>> = model_y
//         .iter()
//         .map(|x| {
//             Array2::from_shape_fn((x.nrows(), x.ncols()), |(i, j)| {
//                 c64(x[(i, j)].real(), x[(i, j)].imag())
//             })
//         })
//         .collect();
//     let model = NetworkBuilder::new()
//         .freq(meas.freq().clone())
//         .z0(meas.z0().clone())
//         .y(model_y_c64)
//         .build();

//     calc_err(meas, &model)
// }

// // Benchmark setup function
// fn setup_benchmark_data() -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Network) {
//     let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
//     let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
//     let lb: Array1<f64> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
//     let ub: Array1<f64> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];

//     // Note: You'll need to ensure this file path is correct for your benchmark environment
//     let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
//     let net = read_touchstone(&filename).expect("Failed to load test data file");

//     (x, scale, lb, ub, net)
// }

// // Benchmarks for unbounded Nelder-Mead
// fn bench_nelder_mead_unbounded(c: &mut Criterion) {
//     let (x, scale, _lb, _ub, net) = setup_benchmark_data();

//     let mut group = c.benchmark_group("nelder_mead_unbounded");

//     // Benchmark different iteration counts
//     for iterations in [1, 2, 5, 6, 10, 100, 1000].iter() {
//         // Reduce sample size for iterations > 100
//         if *iterations > 100 {
//             group.sample_size(10);
//             group.measurement_time(std::time::Duration::from_secs(30));
//         }

//         group.bench_with_input(
//             BenchmarkId::new("iterations", iterations),
//             iterations,
//             |b, &iterations| {
//                 b.iter(|| {
//                     let mut solver = NelderMead::new(
//                         black_box(x.clone()),
//                         black_box(scale.clone()),
//                         black_box(net.clone()),
//                         black_box(eval_f_simplex),
//                     );
//                     solver.solve(black_box(iterations));
//                     black_box(solver.get_res())
//                 })
//             },
//         );
//     }

//     group.finish();
// }

// // Benchmarks for bounded Nelder-Mead
// fn bench_nelder_mead_bounded(c: &mut Criterion) {
//     let (x, scale, lb, ub, net) = setup_benchmark_data();

//     let mut group = c.benchmark_group("nelder_mead_bounded");

//     // Benchmark different iteration counts
//     for iterations in [1, 2, 10, 100, 200, 500, 1000].iter() {
//         // Reduce sample size for iterations > 100
//         if *iterations > 100 {
//             group.sample_size(10);
//             group.measurement_time(std::time::Duration::from_secs(30));
//         }

//         group.bench_with_input(
//             BenchmarkId::new("iterations", iterations),
//             iterations,
//             |b, &iterations| {
//                 b.iter(|| {
//                     let mut solver = NelderMeadBounded::new(
//                         black_box(x.clone()),
//                         black_box(scale.clone()),
//                         black_box(lb.clone()),
//                         black_box(ub.clone()),
//                         black_box(net.clone()),
//                         black_box(eval_f_simplex),
//                     );
//                     solver.solve(black_box(iterations));
//                     black_box(solver.get_res())
//                 })
//             },
//         );
//     }

//     group.finish();
// }

// // Benchmark different mu values for bounded solver
// fn bench_nelder_mead_bounded_mu_variants(c: &mut Criterion) {
//     let (x, scale, lb, ub, net) = setup_benchmark_data();

//     let mut group = c.benchmark_group("nelder_mead_bounded_mu_variants");

//     for mu in [0.01, 0.1, 1.0, 10.0].iter() {
//         group.bench_with_input(BenchmarkId::new("mu", mu), mu, |b, &mu| {
//             b.iter(|| {
//                 let mut solver = NelderMeadBounded::new(
//                     black_box(x.clone()),
//                     black_box(scale.clone()),
//                     black_box(lb.clone()),
//                     black_box(ub.clone()),
//                     black_box(net.clone()),
//                     black_box(eval_f_simplex),
//                 );
//                 solver.set_mu(black_box(mu));
//                 solver.solve(black_box(200));
//                 black_box(solver.get_res())
//             })
//         });
//     }

//     group.finish();
// }

// // Benchmark solver initialization overhead
// fn bench_solver_initialization(c: &mut Criterion) {
//     let (x, scale, lb, ub, net) = setup_benchmark_data();

//     let mut group = c.benchmark_group("solver_initialization");

//     group.bench_function("nelder_mead_unbounded_init", |b| {
//         b.iter(|| {
//             black_box(NelderMead::new(
//                 black_box(x.clone()),
//                 black_box(scale.clone()),
//                 black_box(net.clone()),
//                 black_box(eval_f_simplex),
//             ))
//         })
//     });

//     group.bench_function("nelder_mead_bounded_init", |b| {
//         b.iter(|| {
//             black_box(NelderMeadBounded::new(
//                 black_box(x.clone()),
//                 black_box(scale.clone()),
//                 black_box(lb.clone()),
//                 black_box(ub.clone()),
//                 black_box(net.clone()),
//                 black_box(eval_f_simplex),
//             ))
//         })
//     });

//     group.finish();
// }

// // Benchmark objective function evaluation
// fn bench_objective_function(c: &mut Criterion) {
//     let (x, _scale, _lb, _ub, net) = setup_benchmark_data();
//     let x_myfloat = Array1::from_shape_fn(x.len(), |i| MyFloat::new(x[i]));

//     c.bench_function("objective_function_eval", |b| {
//         b.iter(|| {
//             black_box(eval_f_simplex(
//                 black_box(x_myfloat.clone()),
//                 black_box(&net),
//             ))
//         })
//     });
// }

// // Benchmark convergence comparison
// fn bench_convergence_comparison(c: &mut Criterion) {
//     let (x, scale, lb, ub, net) = setup_benchmark_data();

//     let mut group = c.benchmark_group("convergence_comparison");
//     group.sample_size(10); // Reduce sample size for longer-running benchmarks

//     // Compare unbounded vs bounded for same iteration count
//     group.bench_function("unbounded_100_iters", |b| {
//         b.iter(|| {
//             let mut solver = NelderMead::new(
//                 black_box(x.clone()),
//                 black_box(scale.clone()),
//                 black_box(net.clone()),
//                 black_box(eval_f_simplex),
//             );
//             solver.solve(black_box(100));
//             black_box((solver.get_res(), solver.get_tol(), solver.get_iters()))
//         })
//     });

//     group.bench_function("bounded_100_iters", |b| {
//         b.iter(|| {
//             let mut solver = NelderMeadBounded::new(
//                 black_box(x.clone()),
//                 black_box(scale.clone()),
//                 black_box(lb.clone()),
//                 black_box(ub.clone()),
//                 black_box(net.clone()),
//                 black_box(eval_f_simplex),
//             );
//             solver.solve(black_box(100));
//             black_box((solver.get_res(), solver.get_tol(), solver.get_iters()))
//         })
//     });

//     group.finish();
// }

// // Memory allocation benchmark
// fn bench_memory_usage(c: &mut Criterion) {
//     let (x, scale, lb, ub, net) = setup_benchmark_data();

//     let mut group = c.benchmark_group("memory_usage");

//     // Test memory allocation patterns for different problem sizes
//     group.bench_function("large_simplex_operations", |b| {
//         b.iter(|| {
//             let mut solver = NelderMeadBounded::new(
//                 black_box(x.clone()),
//                 black_box(scale.clone()),
//                 black_box(lb.clone()),
//                 black_box(ub.clone()),
//                 black_box(net.clone()),
//                 black_box(eval_f_simplex),
//             );
//             // Run just a few iterations to test memory allocation patterns
//             solver.solve(black_box(10));
//             black_box(solver.get_simplex())
//         })
//     });

//     group.finish();
// }

// criterion_group!(
//     benches,
//     bench_nelder_mead_unbounded,
//     bench_nelder_mead_bounded,
//     bench_nelder_mead_bounded_mu_variants,
//     bench_solver_initialization,
//     bench_objective_function,
//     bench_convergence_comparison,
//     bench_memory_usage
// );

// criterion_main!(benches);

// #[cfg(test)]
// mod benchmark_tests {
//     // use super::*;

//     /// Quick smoke test to ensure benchmark functions work
//     #[test]
//     fn test_benchmark_setup() {
//         let (x, scale, lb, ub, net) = setup_benchmark_data();

//         // Test unbounded solver
//         let mut unbounded_solver =
//             NelderMead::new(x.clone(), scale.clone(), net.clone(), eval_f_simplex);
//         unbounded_solver.solve(1);
//         assert!(unbounded_solver.get_res().is_some());

//         // Test bounded solver
//         let mut bounded_solver = NelderMeadBounded::new(
//             x.clone(),
//             scale.clone(),
//             lb,
//             ub,
//             net.clone(),
//             eval_f_simplex,
//         );
//         bounded_solver.solve(1);
//         assert!(bounded_solver.get_res().is_some());

//         // Test objective function
//         let x_myfloat = Array1::from_shape_fn(x.len(), |i| MyFloat::new(x[i]));
//         let result = eval_f_simplex(x_myfloat, &net);
//         assert!(result.to_f64() > 0.0);
//     }
// }
