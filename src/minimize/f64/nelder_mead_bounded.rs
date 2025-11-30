#![allow(unused_assignments)]
#![allow(unused_variables)]
use crate::{
    error::MinimizerError,
    minimize::{Minimizer, MinimizerOptions, MinimizerResult, f64::ObjFn},
};
use ndarray::prelude::*;

/// Result of Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadBoundedResult {
    pub xmin: Array1<f64>,
    pub fmin: f64,
    pub tolerance: f64,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_simplex_size: f64,
    pub history: Array1<f64>,
}

impl MinimizerResult<Array1<f64>, f64> for NelderMeadBoundedResult {
    fn converged(&self) -> bool {
        self.converged
    }

    fn fmin(&self) -> f64 {
        self.fmin
    }

    fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn fn_evals(&self) -> usize {
        self.fn_evals
    }

    fn iters(&self) -> usize {
        self.iters
    }

    fn xmin(&self) -> Array1<f64> {
        self.xmin.clone()
    }

    fn history(&self) -> Array1<f64> {
        self.history.clone()
    }
}

#[derive(Debug, Clone)]
pub struct NelderMeadBoundedOptions {
    initial_point: Array1<f64>,
    scale: Array1<f64>,
    max_iterations: usize,
    tolerance: f64,
    alpha: f64, // Reflection coefficient
    beta: f64,  // Contraction coefficient
    gamma: f64, // Expansion coefficient
    rho: f64,   // Scaling coefficient
    mu: f64,
    use_random_restart: bool,    // Enable random restarts when stuck
    use_adaptive_simplex: bool,  // Use adaptive initial simplex sizing
    stagnation_iters: usize,     // Counter for detecting stagnation
    stagnation_threshold: usize, // Max iterations without improvement before restart
    verbosity: usize,
}

impl NelderMeadBoundedOptions {
    pub fn new(
        init: Array1<f64>,
        scale: Option<Array1<f64>>,
        max_iters: Option<usize>,
        tol: Option<f64>,
        alpha: Option<f64>,
        beta: Option<f64>,
        gamma: Option<f64>,
        rho: Option<f64>,
        mu: Option<f64>,
        verbosity: Option<usize>,
    ) -> Self {
        let n = init.len();
        Self {
            initial_point: init,
            scale: scale.unwrap_or(Array1::ones(n)),
            max_iterations: max_iters.unwrap_or(1000),
            tolerance: tol.unwrap_or(0.0),
            alpha: alpha.unwrap_or(1.0),
            beta: beta.unwrap_or(0.5),
            gamma: gamma.unwrap_or(2.0),
            rho: rho.unwrap_or(0.5),
            mu: mu.unwrap_or(10.0),
            use_random_restart: true,
            use_adaptive_simplex: true,
            stagnation_iters: 0,
            stagnation_threshold: 50,
            verbosity: verbosity.unwrap_or(0),
        }
    }

    pub fn set_initial_point(&mut self, init: Array1<f64>) {
        self.initial_point = init;
    }

    pub fn set_scale(&mut self, scale: Array1<f64>) {
        self.scale = scale;
    }

    pub fn set_max_iterations(&mut self, iters: usize) {
        self.max_iterations = iters;
    }

    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
    }

    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
    }

    pub fn set_rho(&mut self, rho: f64) {
        self.rho = rho;
    }

    pub fn set_mu(&mut self, mu: f64) {
        self.mu = mu;
    }

    pub fn set_random_restart(&mut self, restart: bool) {
        self.use_random_restart = restart;
    }

    pub fn set_adaptive_simplex(&mut self, val: bool) {
        self.use_adaptive_simplex = val;
    }

    pub fn set_stagnation_iters(&mut self, val: usize) {
        self.stagnation_iters = val;
    }

    pub fn set_stagnation_threshold(&mut self, val: usize) {
        self.stagnation_threshold = val;
    }

    pub fn set_verbosity(&mut self, val: usize) {
        self.verbosity = val;
    }
}

impl MinimizerOptions<Array1<f64>, f64> for NelderMeadBoundedOptions {
    fn initial_point(&self) -> Array1<f64> {
        self.initial_point.clone()
    }

    fn scale(&self) -> Array1<f64> {
        self.scale.clone()
    }

    fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn verbosity(&self) -> usize {
        self.verbosity
    }
}

pub struct NelderMeadBounded {
    scale: Array1<f64>,
    x: Array1<f64>,
    x_scaled: Array1<f64>,
    lb: Array1<f64>,
    lb_vec: Vec<f64>,
    ub: Array1<f64>,
    ub_vec: Vec<f64>,
    res: Option<Array1<f64>>,
    simplex: Option<Array2<f64>>,
    fn_evals: usize,
    f: Box<dyn ObjFn<f64>>,
    n: usize,
    iters: usize,
    max_iters: usize, // Maximum iterations
    tol: Option<f64>,
    target_tol: f64, // Convergent tolerance
    alpha: f64,      // Reflection coefficient
    beta: f64,       // Contraction coefficient
    gamma: f64,      // Expansion coefficient
    rho: f64,        // Scaling coefficient
    mu: f64,
    verbosity: usize,
    use_random_restart: bool,           // Enable random restarts when stuck
    use_adaptive_simplex: bool,         // Use adaptive initial simplex sizing
    stagnation_iters: usize,            // Counter for detecting stagnation
    stagnation_threshold: usize,        // Max iterations without improvement before restart
    best_fmin_global: f64,              // Best value found across all restarts
    best_x_global: Option<Array1<f64>>, // Best solution found globally
}

impl NelderMeadBounded {
    pub fn new<F>(
        x: Array1<f64>,
        scale: Array1<f64>,
        lb: Array1<f64>,
        ub: Array1<f64>,
        f: F,
    ) -> Self
    where
        F: ObjFn<f64> + 'static,
    {
        let mut lb_vec: Vec<f64> = vec![];
        let mut ub_vec: Vec<f64> = vec![];
        for i in 0..x.len() {
            lb_vec.push(lb[i]);
            ub_vec.push(ub[i]);
        }
        NelderMeadBounded {
            x_scaled: Array1::from_shape_fn(x.len(), |i| x[i] * scale[i]),
            x: x.clone(),
            lb_vec,
            ub_vec,
            scale,
            lb,
            ub,
            res: None,
            simplex: None,
            fn_evals: 0,
            f: Box::new(f),
            n: x.len(),
            iters: 0,
            max_iters: 1,
            tol: None,
            target_tol: 0.0,
            alpha: 1.0,
            beta: 0.5,
            gamma: 2.0,
            rho: 0.5,
            mu: 10.0,
            verbosity: 0,
            use_random_restart: false, // Disabled by default for backward compatibility
            use_adaptive_simplex: false, // Disabled by default for backward compatibility
            stagnation_iters: 0,
            stagnation_threshold: 50,
            best_fmin_global: f64::INFINITY,
            best_x_global: None,
        }
    }

    fn check_bounds(&self, x: f64, lb: f64, ub: f64, scale: f64) -> f64 {
        if x < (lb * scale) {
            return (1.0 + 0.05 / (self.iters as f64 + 1.0)) * lb * scale;
        } else if x > (ub * scale) {
            return (1.0 - 0.05 / (self.iters as f64 + 1.0)) * ub * scale;
        }
        x
    }

    fn res_all(&self) -> Option<Array1<f64>> {
        self.res.clone()
    }

    fn x_scaled(&self) -> Array1<f64> {
        self.x_scaled.clone()
    }

    fn simplex(&self) -> Option<Array2<f64>> {
        self.simplex.clone()
    }

    pub fn get_mu(&self) -> f64 {
        self.mu
    }

    pub fn set_mu(&mut self, mu: f64) {
        self.mu = mu;
    }

    pub fn set_target_tolerance(&mut self, tol: f64) {
        self.target_tol = tol;
    }

    pub fn set_x(&mut self, x: Array1<f64>) {
        self.x = x;
        self.x_scaled = Array1::from_shape_fn(self.x.len(), |i| self.x[i] * self.scale[i]);
        self.res = None;
        self.simplex = None;
        self.tol = None;
    }

    pub fn set_verbosity(&mut self, verbose: usize) {
        self.verbosity = verbose;
    }

    pub fn set_random_restart(&mut self, enable: bool) {
        self.use_random_restart = enable;
    }

    pub fn set_stagnation_threshold(&mut self, threshold: usize) {
        self.stagnation_threshold = threshold;
    }

    pub fn set_adaptive_simplex(&mut self, enable: bool) {
        self.use_adaptive_simplex = enable;
    }

    /// Generate a random point within bounds using Latin Hypercube Sampling for better coverage
    fn random_point_in_bounds(&self) -> Array1<f64> {
        use std::f64::consts::PI;
        let mut rng_state = (self.iters as u64).wrapping_mul(0x9e3779b97f4a7c15);
        Array1::from_shape_fn(self.n, |i| {
            // Simple xorshift random number generator (deterministic but pseudo-random)
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let rand_val = (rng_state as f64) / (u64::MAX as f64);

            // Use a perturbed value to avoid exact duplicates
            let adjusted_rand = (rand_val + (i as f64) * 0.1 * PI.sin()).fract();
            self.lb[i] + adjusted_rand * (self.ub[i] - self.lb[i])
        })
    }

    /// Create a randomized simplex around a point for restart
    fn create_random_simplex(&self, center: &Array1<f64>) -> Array2<f64> {
        let nrows = self.n + 1;
        let mut rng_state = (self.iters as u64 + 12345).wrapping_mul(0x9e3779b97f4a7c15);

        Array2::from_shape_fn((nrows, self.n), |(i, j)| {
            if i == nrows - 1 {
                // Last point is the center
                self.check_bounds(
                    center[j] * self.scale[j],
                    self.lb[j],
                    self.ub[j],
                    self.scale[j],
                )
            } else {
                // Generate points around center with some randomness
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let rand_val = (rng_state as f64) / (u64::MAX as f64);

                // Random perturbation scaled by the range
                let range = (self.ub[j] - self.lb[j]) * 0.1; // 10% of range
                let perturbation = (rand_val - 0.5) * 2.0 * range;

                self.check_bounds(
                    (center[j] + perturbation) * self.scale[j],
                    self.lb[j],
                    self.ub[j],
                    self.scale[j],
                )
            }
        })
    }

    pub fn calc_obj(&mut self, x: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += (self.ub[i] - x[i].clone()).ln() + (x[i].clone() - self.lb[i]).ln();
        }
        let x_debug = x.clone().to_vec();
        let f_debug = self.f.call(x);
        self.f.call(x) - self.mu * sum
    }

    pub fn x(&self) -> &Array1<f64> {
        &self.x
    }

    pub fn final_value(&self) -> Option<f64> {
        self.res.as_ref().map(|x| x[0])
    }

    pub fn tolerance(&self) -> Option<f64> {
        self.tol
    }

    pub fn iterations(&self) -> usize {
        self.iters
    }

    pub fn name(&self) -> &str {
        "NelderMeadBounded"
    }

    /// Solve using Nelder-Mead algorithm
    fn minimize_opt(
        &mut self,
        opt: Box<NelderMeadBoundedOptions>,
    ) -> Result<Box<NelderMeadBoundedResult>, MinimizerError> {
        self.scale = opt.scale;
        self.target_tol = opt.tolerance;
        self.set_x(opt.initial_point);
        self.max_iters = opt.max_iterations;
        self.iters = 0;
        self.fn_evals = 0;
        self.stagnation_iters = 0;
        self.best_fmin_global = f64::INFINITY;
        self.best_x_global = None;
        let mut history: Vec<f64> = vec![];
        let mut last_best_fmin = f64::INFINITY;

        // Generate initial simplex vertices
        let c: f64 = 1.0;
        let b = c / (self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0);
        let a = c / 2f64.sqrt();
        let _ncols = self.n;
        let nrows = self.n + 1;

        let mut simplex = if self.use_adaptive_simplex {
            // Use adaptive sizing based on parameter ranges for better global exploration
            let range_factor = Array1::from_shape_fn(self.n, |i| {
                (self.ub[i] - self.lb[i]) * self.scale[i] * 0.1 // 10% of parameter range
            });

            Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
                if i == j && i < self.n {
                    let step = (a + b) * range_factor[j].max(1.0);
                    self.check_bounds(
                        self.x_scaled[j] + step,
                        self.lb[j],
                        self.ub[j],
                        self.scale[j],
                    )
                } else if i < self.n {
                    let step = b * range_factor[j].max(1.0);
                    self.check_bounds(
                        self.x_scaled[j] + step,
                        self.lb[j],
                        self.ub[j],
                        self.scale[j],
                    )
                } else {
                    self.x_scaled[j]
                }
            })
        } else {
            // Use standard simplex construction (backward compatible)
            Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
                if i == j && i < self.n {
                    self.check_bounds(
                        self.x_scaled[j] + a + b,
                        self.lb[j],
                        self.ub[j],
                        self.scale[j],
                    )
                } else if i < self.n {
                    self.check_bounds(self.x_scaled[j] + b, self.lb[j], self.ub[j], self.scale[j])
                } else {
                    self.x_scaled[j]
                }
            })
        };

        // Evaluate function at simplex vertices
        let mut res: Array1<f64> = simplex
            .rows()
            .into_iter()
            .map(|x| self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x[i] / self.scale[i])))
            .collect();
        self.fn_evals += self.n;
        let mut prev_tol = 1.0;
        history.push(res[0]);

        while self.iters < self.max_iters {
            if self.verbosity > 1 {
                println!(
                    "iteration: {}\terr: {}\ttol: {}",
                    self.iters,
                    res[0],
                    self.tol.unwrap_or(1.0)
                );
            }
            self.iters += 1;

            // Check for stagnation and perform random restart if enabled
            let current_best = res[0];
            if (last_best_fmin - current_best).abs() < 1e-10 {
                self.stagnation_iters += 1;
            } else {
                self.stagnation_iters = 0;
                last_best_fmin = current_best;
            }

            // Update global best
            if current_best < self.best_fmin_global {
                self.best_fmin_global = current_best;
                self.best_x_global = Some(Array1::from_shape_fn(self.n, |i| {
                    simplex[(0, i)] / self.scale[i]
                }));
            }

            // Perform random restart if stuck
            if self.use_random_restart && self.stagnation_iters >= self.stagnation_threshold {
                if self.verbosity > 0 {
                    println!(
                        "Stagnation detected at iteration {}. Performing random restart. Best global: {}",
                        self.iters, self.best_fmin_global
                    );
                }

                // Generate new random starting point
                let random_x = self.random_point_in_bounds();
                simplex = self.create_random_simplex(&random_x);

                // Re-evaluate simplex
                res = simplex
                    .rows()
                    .into_iter()
                    .map(|x| {
                        self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x[i] / self.scale[i]))
                    })
                    .collect();
                self.fn_evals += self.n + 1;

                self.stagnation_iters = 0;
                last_best_fmin = f64::INFINITY;

                // Continue to next iteration
                history.push(res[0]);
                continue;
            }

            // Sort points from best to worst
            let mut order: Vec<usize> = (0..res.len()).collect();
            order.sort_by(|&a, &b| {
                res[a]
                    .partial_cmp(&res[b])
                    .unwrap_or(std::cmp::Ordering::Greater)
            });
            let tmp_res = res.clone();
            let tmp_simplex = simplex.clone();
            for i in 0..order.len() {
                res[i] = tmp_res[order[i]].clone();
                for j in 0..self.n {
                    simplex[(i, j)] = tmp_simplex[(order[i], j)].clone();
                }
            }
            let shape = match simplex.shape() {
                [a, b] => (*a, *b),
                _ => panic!("Shape must be 2-dimensional"),
            };
            self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| simplex[(i, j)]));

            // Determine if convergence criteria met
            let tol = 2.0 * (res[res.len() - 1].clone() - res[0].clone()).abs()
                / (res[res.len() - 1].clone().abs() + res[0].clone().abs() + 1e-10);
            // if tol.to_f64() < self.target_tol || (&prev_tol - &tol).abs().to_f64() < 1e-15 {
            //     break;
            // }
            match opt.tolerance {
                0.0 => (),
                _ => {
                    if tol < opt.tolerance {
                        break;
                    }
                }
            }
            prev_tol = tol.clone();

            let x_b = Array1::from_shape_fn(self.n, |i| simplex[(0, i)].clone()); // Best point
            let _x_l = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 2, i)].clone()); // Lousy point
            let x_w = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 1, i)].clone()); // Worst point

            // Calculate the average of the n best points
            let mut x_avg = Array1::zeros(self.n);
            for i in 0..self.n {
                let mut sum = 0.0;
                for j in 0..self.n {
                    sum += &simplex[(j, i)];
                }
                x_avg[i] = 1.0 / self.n as f64 * sum;
            }

            // Calculate reflection point
            let x_r = Array1::from_shape_fn(self.n, |i| {
                self.check_bounds(
                    &x_avg[i] + self.alpha * (&x_avg[i] - &x_w[i]),
                    self.lb[i],
                    self.ub[i],
                    self.scale[i],
                )
            });
            let f_r = self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_r[i] / self.scale[i]));
            self.fn_evals += self.n;

            //
            // Determine simplex adjustment
            //
            if f_r <= res[0] {
                // Perform expansion
                if self.verbosity > 1 {
                    println!("performing expansion");
                }
                let x_e = Array1::from_shape_fn(self.n, |i| {
                    self.check_bounds(
                        &x_r[i] + self.gamma * (&x_r[i] - &x_avg[i]),
                        self.lb[i],
                        self.ub[i],
                        self.scale[i],
                    )
                });
                let f_e =
                    self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_e[i] / self.scale[i]));
                self.fn_evals += self.n;
                if f_e < res[0] {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_e[i].clone();
                    }
                    res[nrows - 1] = f_e;
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_r[i].clone();
                    }
                    res[nrows - 1] = f_r;
                }
            } else if f_r > res[nrows - 1] {
                // Perform inside contraction
                if self.verbosity > 1 {
                    println!("performing inside contraction");
                }
                let x_ic = Array1::from_shape_fn(self.n, |i| {
                    self.check_bounds(
                        &x_avg[i] - self.beta * (&x_avg[i] - &x_w[i]),
                        self.lb[i],
                        self.ub[i],
                        self.scale[i],
                    )
                });
                let f_ic =
                    self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_ic[i] / self.scale[i]));
                self.fn_evals += 1;
                if f_ic > res[nrows - 1] {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] = self.check_bounds(
                                &x_b[j] + self.rho * (&simplex[(i + 1, j)] - &x_b[j]),
                                self.lb[j],
                                self.ub[j],
                                self.scale[j],
                            );
                        }

                        res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
                            &simplex[(i + 1, j)] / self.scale[j]
                        }));
                        self.fn_evals += self.n;
                    }
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_ic[i].clone();
                    }
                    res[nrows - 1] = f_ic;
                }
            } else if f_r > res[nrows - 2] {
                // Perform outside contraction
                if self.verbosity > 1 {
                    println!("performing outside contraction");
                }
                let x_oc = Array1::from_shape_fn(self.n, |i| {
                    self.check_bounds(
                        &x_avg[i] + self.beta * (&x_avg[i] - &x_w[i]),
                        self.lb[i],
                        self.ub[i],
                        self.scale[i],
                    )
                });
                let f_oc =
                    self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_oc[i] / self.scale[i]));
                self.fn_evals += 1;
                if f_oc > f_r {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] = self.check_bounds(
                                &x_b[j] + self.rho * (&simplex[(i + 1, j)] - &x_b[j]),
                                self.lb[j],
                                self.ub[j],
                                self.scale[j],
                            );
                        }
                        res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
                            &simplex[(i + 1, j)] / self.scale[j]
                        }));
                        self.fn_evals += self.n;
                    }
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_oc[i].clone();
                    }
                    res[nrows - 1] = f_oc;
                }
            } else {
                for i in 0..self.n {
                    simplex[(nrows - 1, i)] = x_r[i].clone();
                }
                res[nrows - 1] = f_r;
            }

            self.tol = Some(tol);
            history.push(res[0]);
        }

        // Final check: use the best global solution if better than current
        let current_best_x = Array1::from_shape_fn(self.n, |i| simplex[(0, i)] / self.scale[i]);
        let current_best_f = self.calc_obj(&current_best_x);

        let (final_x, final_f) = if self.use_random_restart
            && self.best_x_global.is_some()
            && self.best_fmin_global < current_best_f
        {
            if self.verbosity > 0 {
                println!(
                    "Using best global solution: {} (current: {})",
                    self.best_fmin_global, current_best_f
                );
            }
            (self.best_x_global.clone().unwrap(), self.best_fmin_global)
        } else {
            (current_best_x, current_best_f)
        };

        let shape = match simplex.shape() {
            [a, b] => (*a, *b),
            _ => panic!("Shape must be 2-dimensional"),
        };
        self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| simplex[(i, j)]));
        self.x_scaled = Array1::from_shape_fn(self.n, |i| final_x[i] * self.scale[i]);
        self.x = final_x.clone();
        // self.res = Some(res);
        self.res = Some(Array1::from_shape_fn(res.len(), |i| res[i]));

        if self.verbosity > 0 {
            println!("x: {:?}", self.x());
            println!("res: {:?}", self.res_all().unwrap());
            println!("iters: {}", self.iterations());
            println!("tol: {:?}", self.tolerance().unwrap());
        }

        Ok(Box::new(NelderMeadBoundedResult {
            xmin: self.x.clone(),
            fmin: final_f,
            tolerance: match self.tol {
                Some(x) => x,
                _ => 0.0,
            },
            iters: self.iters,
            fn_evals: self.fn_evals,
            converged: true,
            final_simplex_size: 0.0,
            history: Array1::from_vec(history),
        }))
    }
}

impl Minimizer<Array1<f64>, f64> for NelderMeadBounded {
    fn minimize(
        &mut self,
        opt: Box<dyn MinimizerOptions<Array1<f64>, f64>>,
    ) -> Result<Box<dyn MinimizerResult<Array1<f64>, f64>>, MinimizerError> {
        // Extract values from the trait object
        let initial_point = opt.initial_point();
        let scale = opt.scale();
        let tolerance = opt.tolerance();
        let max_iters = opt.max_iterations();
        let verbosity = opt.verbosity();

        // Create a concrete NelderMeadBoundedOptions
        let concrete_opt = Box::new(NelderMeadBoundedOptions::new(
            initial_point,
            Some(scale),
            Some(max_iters),
            Some(tolerance),
            None,
            None,
            None,
            None,
            None,
            Some(verbosity),
        ));

        // Call minimize_opt and cast result to trait object
        let result = self.minimize_opt(concrete_opt)?;
        Ok(result as Box<dyn MinimizerResult<Array1<f64>, f64>>)
    }
    // /// Solve using Nelder-Mead algorithm
    // fn minimize(
    //     &mut self,
    //     opt: Box<dyn MinimizerOptions<Array1<f64>, f64>>,
    // ) -> Box<dyn MinimizerResult<Array1<f64>, f64>> {
    //     self.max_iters = match max_iters {
    //         Some(x) => x,
    //         _ => 1000,
    //     };
    //     self.iters = 0;
    //     self.fn_evals = 0;
    //     self.stagnation_iters = 0;
    //     self.best_fmin_global = f64::INFINITY;
    //     self.best_x_global = None;
    //     let mut history: Vec<f64> = vec![];
    //     let mut last_best_fmin = f64::INFINITY;
    //     let _x_debug = self.x.clone().to_vec();
    //     let _x_scaled_debug = self.x_scaled.clone().to_vec();
    //     let _lb_debug = self.lb.clone().to_vec();
    //     let _ub_debug = self.ub.clone().to_vec();

    //     // Generate initial simplex vertices
    //     let c: f64 = 1.0;
    //     let b = c / (self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0);
    //     let a = c / 2f64.sqrt();
    //     let _ncols = self.n;
    //     let nrows = self.n + 1;

    //     let mut simplex = if self.use_adaptive_simplex {
    //         // Use adaptive sizing based on parameter ranges for better global exploration
    //         let range_factor = Array1::from_shape_fn(self.n, |i| {
    //             (self.ub[i] - self.lb[i]) * self.scale[i] * 0.1 // 10% of parameter range
    //         });

    //         Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
    //             if i == j && i < self.n {
    //                 let step = (a + b) * range_factor[j].max(1.0);
    //                 self.check_bounds(
    //                     self.x_scaled[j] + step,
    //                     self.lb[j],
    //                     self.ub[j],
    //                     self.scale[j],
    //                 )
    //             } else if i < self.n {
    //                 let step = b * range_factor[j].max(1.0);
    //                 self.check_bounds(
    //                     self.x_scaled[j] + step,
    //                     self.lb[j],
    //                     self.ub[j],
    //                     self.scale[j],
    //                 )
    //             } else {
    //                 self.x_scaled[j]
    //             }
    //         })
    //     } else {
    //         // Use standard simplex construction (backward compatible)
    //         Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
    //             if i == j && i < self.n {
    //                 self.check_bounds(
    //                     self.x_scaled[j] + a + b,
    //                     self.lb[j],
    //                     self.ub[j],
    //                     self.scale[j],
    //                 )
    //             } else if i < self.n {
    //                 self.check_bounds(self.x_scaled[j] + b, self.lb[j], self.ub[j], self.scale[j])
    //             } else {
    //                 self.x_scaled[j]
    //             }
    //         })
    //     };

    //     // Evaluate function at simplex vertices
    //     let mut res: Array1<f64> = simplex
    //         .rows()
    //         .into_iter()
    //         .map(|x| self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x[i] / self.scale[i])))
    //         .collect();
    //     self.fn_evals += self.n;
    //     let mut prev_tol = 1.0;
    //     history.push(res[0]);
    //     let _res_debug = res.clone().to_vec();

    //     while self.iters < self.max_iters {
    //         if self.verbosity > 1 {
    //             println!(
    //                 "iteration: {}\terr: {}\ttol: {}",
    //                 self.iters,
    //                 res[0],
    //                 self.tol.unwrap_or(1.0)
    //             );
    //         }
    //         self.iters += 1;

    //         // Check for stagnation and perform random restart if enabled
    //         let current_best = res[0];
    //         if (last_best_fmin - current_best).abs() < 1e-10 {
    //             self.stagnation_iters += 1;
    //         } else {
    //             self.stagnation_iters = 0;
    //             last_best_fmin = current_best;
    //         }

    //         // Update global best
    //         if current_best < self.best_fmin_global {
    //             self.best_fmin_global = current_best;
    //             self.best_x_global = Some(Array1::from_shape_fn(self.n, |i| {
    //                 simplex[(0, i)] / self.scale[i]
    //             }));
    //         }

    //         // Perform random restart if stuck
    //         if self.use_random_restart && self.stagnation_iters >= self.stagnation_threshold {
    //             if self.verbosity > 0 {
    //                 println!(
    //                     "Stagnation detected at iteration {}. Performing random restart. Best global: {}",
    //                     self.iters, self.best_fmin_global
    //                 );
    //             }

    //             // Generate new random starting point
    //             let random_x = self.random_point_in_bounds();
    //             simplex = self.create_random_simplex(&random_x);

    //             // Re-evaluate simplex
    //             res = simplex
    //                 .rows()
    //                 .into_iter()
    //                 .map(|x| {
    //                     self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x[i] / self.scale[i]))
    //                 })
    //                 .collect();
    //             self.fn_evals += self.n + 1;

    //             self.stagnation_iters = 0;
    //             last_best_fmin = f64::INFINITY;

    //             // Continue to next iteration
    //             history.push(res[0]);
    //             continue;
    //         }

    //         let _simplex_debug = simplex.clone().into_flat().to_vec();

    //         // Sort points from best to worst
    //         let mut order: Vec<usize> = (0..res.len()).collect();
    //         order.sort_by(|&a, &b| {
    //             res[a]
    //                 .partial_cmp(&res[b])
    //                 .unwrap_or(std::cmp::Ordering::Greater)
    //         });
    //         let tmp_res = res.clone();
    //         let tmp_simplex = simplex.clone();
    //         for i in 0..order.len() {
    //             res[i] = tmp_res[order[i]].clone();
    //             for j in 0..self.n {
    //                 simplex[(i, j)] = tmp_simplex[(order[i], j)].clone();
    //             }
    //         }
    //         let shape = match simplex.shape() {
    //             [a, b] => (*a, *b),
    //             _ => panic!("Shape must be 2-dimensional"),
    //         };
    //         self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| simplex[(i, j)]));

    //         // Determine if convergence criteria met
    //         let tol = 2.0 * (res[res.len() - 1].clone() - res[0].clone()).abs()
    //             / (res[res.len() - 1].clone().abs() + res[0].clone().abs() + 1e-10);
    //         // if tol.to_f64() < self.target_tol || (&prev_tol - &tol).abs().to_f64() < 1e-15 {
    //         //     break;
    //         // }
    //         match self.target_tol {
    //             Some(target_tol) => {
    //                 if tol < target_tol {
    //                     break;
    //                 }
    //             }
    //             _ => (),
    //         }
    //         prev_tol = tol.clone();

    //         let x_b = Array1::from_shape_fn(self.n, |i| simplex[(0, i)].clone()); // Best point
    //         let _x_l = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 2, i)].clone()); // Lousy point
    //         let x_w = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 1, i)].clone()); // Worst point

    //         // Calculate the average of the n best points
    //         let mut x_avg = Array1::zeros(self.n);
    //         for i in 0..self.n {
    //             let mut sum = 0.0;
    //             for j in 0..self.n {
    //                 sum += &simplex[(j, i)];
    //             }
    //             x_avg[i] = 1.0 / self.n as f64 * sum;
    //         }

    //         // Calculate reflection point
    //         let x_r = Array1::from_shape_fn(self.n, |i| {
    //             self.check_bounds(
    //                 &x_avg[i] + self.alpha * (&x_avg[i] - &x_w[i]),
    //                 self.lb[i],
    //                 self.ub[i],
    //                 self.scale[i],
    //             )
    //         });
    //         let f_r = self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_r[i] / self.scale[i]));
    //         self.fn_evals += self.n;

    //         //
    //         // Determine simplex adjustment
    //         //
    //         if f_r <= res[0] {
    //             // Perform expansion
    //             if self.verbosity > 1 {
    //                 println!("performing expansion");
    //             }
    //             let x_e = Array1::from_shape_fn(self.n, |i| {
    //                 self.check_bounds(
    //                     &x_r[i] + self.gamma * (&x_r[i] - &x_avg[i]),
    //                     self.lb[i],
    //                     self.ub[i],
    //                     self.scale[i],
    //                 )
    //             });
    //             let f_e =
    //                 self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_e[i] / self.scale[i]));
    //             self.fn_evals += self.n;
    //             if f_e < res[0] {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_e[i].clone();
    //                 }
    //                 res[nrows - 1] = f_e;
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_r[i].clone();
    //                 }
    //                 res[nrows - 1] = f_r;
    //             }
    //         } else if f_r > res[nrows - 1] {
    //             // Perform inside contraction
    //             if self.verbosity > 1 {
    //                 println!("performing inside contraction");
    //             }
    //             let x_ic = Array1::from_shape_fn(self.n, |i| {
    //                 self.check_bounds(
    //                     &x_avg[i] - self.beta * (&x_avg[i] - &x_w[i]),
    //                     self.lb[i],
    //                     self.ub[i],
    //                     self.scale[i],
    //                 )
    //             });
    //             let f_ic =
    //                 self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_ic[i] / self.scale[i]));
    //             self.fn_evals += 1;
    //             if f_ic > res[nrows - 1] {
    //                 // Shrink simplex
    //                 if self.verbosity > 1 {
    //                     println!("shrinking simplex");
    //                 }
    //                 for i in 0..self.n {
    //                     for j in 0..self.n {
    //                         simplex[(i + 1, j)] = self.check_bounds(
    //                             &x_b[j] + self.rho * (&simplex[(i + 1, j)] - &x_b[j]),
    //                             self.lb[j],
    //                             self.ub[j],
    //                             self.scale[j],
    //                         );
    //                     }

    //                     res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
    //                         &simplex[(i + 1, j)] / self.scale[j]
    //                     }));
    //                     self.fn_evals += self.n;
    //                 }
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_ic[i].clone();
    //                 }
    //                 res[nrows - 1] = f_ic;
    //             }
    //         } else if f_r > res[nrows - 2] {
    //             // Perform outside contraction
    //             if self.verbosity > 1 {
    //                 println!("performing outside contraction");
    //             }
    //             let x_oc = Array1::from_shape_fn(self.n, |i| {
    //                 self.check_bounds(
    //                     &x_avg[i] + self.beta * (&x_avg[i] - &x_w[i]),
    //                     self.lb[i],
    //                     self.ub[i],
    //                     self.scale[i],
    //                 )
    //             });
    //             let f_oc =
    //                 self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_oc[i] / self.scale[i]));
    //             self.fn_evals += 1;
    //             if f_oc > f_r {
    //                 // Shrink simplex
    //                 if self.verbosity > 1 {
    //                     println!("shrinking simplex");
    //                 }
    //                 for i in 0..self.n {
    //                     for j in 0..self.n {
    //                         simplex[(i + 1, j)] = self.check_bounds(
    //                             &x_b[j] + self.rho * (&simplex[(i + 1, j)] - &x_b[j]),
    //                             self.lb[j],
    //                             self.ub[j],
    //                             self.scale[j],
    //                         );
    //                     }
    //                     res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
    //                         &simplex[(i + 1, j)] / self.scale[j]
    //                     }));
    //                     self.fn_evals += self.n;
    //                 }
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_oc[i].clone();
    //                 }
    //                 res[nrows - 1] = f_oc;
    //             }
    //         } else {
    //             for i in 0..self.n {
    //                 simplex[(nrows - 1, i)] = x_r[i].clone();
    //             }
    //             res[nrows - 1] = f_r;
    //         }

    //         self.tol = Some(tol);
    //         history.push(res[0]);
    //     }

    //     // Final check: use the best global solution if better than current
    //     let current_best_x = Array1::from_shape_fn(self.n, |i| simplex[(0, i)] / self.scale[i]);
    //     let current_best_f = self.calc_obj(&current_best_x);

    //     let (final_x, final_f) = if self.use_random_restart
    //         && self.best_x_global.is_some()
    //         && self.best_fmin_global < current_best_f
    //     {
    //         if self.verbosity > 0 {
    //             println!(
    //                 "Using best global solution: {} (current: {})",
    //                 self.best_fmin_global, current_best_f
    //             );
    //         }
    //         (self.best_x_global.clone().unwrap(), self.best_fmin_global)
    //     } else {
    //         (current_best_x, current_best_f)
    //     };

    //     let shape = match simplex.shape() {
    //         [a, b] => (*a, *b),
    //         _ => panic!("Shape must be 2-dimensional"),
    //     };
    //     self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| simplex[(i, j)]));
    //     self.x_scaled = Array1::from_shape_fn(self.n, |i| final_x[i] * self.scale[i]);
    //     self.x = final_x.clone();
    //     // self.res = Some(res);
    //     self.res = Some(Array1::from_shape_fn(res.len(), |i| res[i]));

    //     if self.verbosity > 0 {
    //         println!("x: {:?}", self.x());
    //         println!("res: {:?}", self.res_all().unwrap());
    //         println!("iters: {}", self.iterations());
    //         println!("tol: {:?}", self.tolerance().unwrap());
    //     }

    //     Box::new(NelderMeadBoundedResult {
    //         xmin: self.x.clone(),
    //         fmin: final_f,
    //         tolerance: match self.tol {
    //             Some(x) => x,
    //             _ => 0.0,
    //         },
    //         iters: self.iters,
    //         fn_evals: self.fn_evals,
    //         converged: true,
    //         final_simplex_size: 0.0,
    //         history: Array1::from_vec(history),
    //     })
    // }
}

#[cfg(test)]
mod minimize_f64_neldermeadbounded_tests {
    use super::*;
    use crate::file::read_touchstone;
    use crate::frequency::*;
    use crate::network::{Network, NetworkBuilder};
    use crate::points::{Points, Pts};
    use crate::util::*;
    use float_cmp::F64Margin;
    use num::complex::*;

    fn calc_feed_y(freq: Frequency, x: &Array1<f64>) -> Array3<Complex64> {
        let mut yfeed = Array3::<Complex64>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut zs = Array1::<Complex64>::zeros(freq.npts());
        let mut zm = Array1::<Complex64>::zeros(freq.npts());
        let mut zp = Array1::<Complex64>::zeros(freq.npts());
        let mut zall = Array1::<Complex64>::zeros(freq.npts());
        for i in 0..freq.npts() {
            zs[i] = x[3] * Complex::I * w[i] * x[2] / (x[3] + Complex::I * w[i] * x[2]);
            zm[i] = x[1] + Complex::I * w[i] * x[0] + zs[i];
            zp[i] = x[5] - Complex::I / (w[i] * x[4]);
            zall[i] = zm[i] * zp[i] / (zm[i] + zp[i]);
        }

        for i in 0..freq.npts() {
            yfeed[[i, 0, 1]] = -1.0 / zall[i];
            yfeed[[i, 1, 0]] = -1.0 / zall[i];
            yfeed[[i, 0, 0]] = Complex::I * w[i] * x[6] + 1.0 / zall[i];
            yfeed[[i, 1, 1]] = Complex::I * w[i] * x[7] + 1.0 / zall[i];
        }

        yfeed
    }

    fn calc_err(model: &Network, meas: &Network) -> f64 {
        let mut err: f64 = 0.0;
        let meas_h = meas.h();
        let model_h = model.h();
        let meas_y = meas.y();
        let model_y = model.y();
        let meas_z = meas.z();
        let model_z = model.z();
        for i in 0..meas.freq().npts() {
            for port in [(0, 0), (0, 1), (1, 1)].iter() {
                err += ((model_h[[i, port.0, port.1]].re - meas_h[[i, port.0, port.1]].re)
                    / meas_h[[i, port.0, port.1]].re)
                    .powi(2)
                    + ((model_h[[i, port.0, port.1]].im - meas_h[[i, port.0, port.1]].im)
                        / meas_h[[i, port.0, port.1]].im)
                        .powi(2);
                err += ((model_y[[i, port.0, port.1]].re - meas_y[[i, port.0, port.1]].re)
                    / meas_y[[i, port.0, port.1]].re)
                    .powi(2)
                    + ((model_y[[i, port.0, port.1]].im - meas_y[[i, port.0, port.1]].im)
                        / meas_y[[i, port.0, port.1]].im)
                        .powi(2);
                err += ((model_z[[i, port.0, port.1]].re - meas_z[[i, port.0, port.1]].re)
                    / meas_z[[i, port.0, port.1]].re)
                    .powi(2)
                    + ((model_z[[i, port.0, port.1]].im - meas_z[[i, port.0, port.1]].im)
                        / meas_z[[i, port.0, port.1]].im)
                        .powi(2);
            }
        }

        err
    }

    fn eval_f_simplex(x: &Array1<f64>, meas: &Network) -> f64 {
        let model_y = calc_feed_y(meas.freq().clone(), x);
        let model_y_c64 =
            Points::from_shape_fn(model_y.dim(), |(i, j, k)| model_y[[i, j, k]].clone().into());
        let model = NetworkBuilder::new()
            .freq(meas.freq().clone())
            .z0(meas.z0().clone())
            .y(model_y_c64)
            .build();

        calc_err(&model, meas)
    }

    fn calc_xfmr_z(
        freq: Frequency,
        ls: f64,
        rs: f64,
        km: f64,
        qp: f64,
        x: Array1<f64>,
    ) -> Array3<Complex64> {
        let mut z = Array3::<Complex64>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut m = Array1::<f64>::zeros(freq.npts());
        let mut n = Array1::<f64>::zeros(freq.npts());
        let mut rp = Array1::<f64>::zeros(freq.npts());
        for i in 0..freq.npts() {
            m[i] = km * (x[0] * ls).sqrt();
            n[i] = (ls / x[0]).sqrt();
            rp[i] = w[i] * x[0] / qp;
        }

        for i in 0..freq.npts() {
            z[[i, 0, 1]] = -Complex::I * freq.w_pt(i) * m[i];
            z[[i, 1, 0]] = -Complex::I * freq.w_pt(i) * m[i];
            z[[i, 0, 0]] = c64(rp[i], w[i] * x[0]);
            z[[i, 1, 1]] = c64(rs, -w[i] * ls);
        }

        z
    }

    fn calc_err_xfmr(meas: &Network, model: &Network) -> f64 {
        let mut err = 0.0;
        let meas_s = meas.s();
        let model_s = model.s();
        for i in 0..meas.freq().npts() {
            for port in [(0, 0)].iter() {
                err += ((model_s[[i, port.0, port.1]].re - meas_s[[i, port.0, port.1]].re)
                    / meas_s[[i, port.0, port.1]].re)
                    .powi(2)
                    + ((model_s[[i, port.0, port.1]].im - meas_s[[i, port.0, port.1]].im)
                        / meas_s[[i, port.0, port.1]].im)
                        .powi(2);
            }
        }

        err
    }

    fn eval_f_xfmr(x: Array1<f64>, ls: f64, rs: f64, km: f64, qp: f64, meas: &Network) -> f64 {
        let model_z = calc_xfmr_z(meas.freq().clone(), ls, rs, km, qp, x);
        let model_z_c64 =
            Points::from_shape_fn(model_z.dim(), |(i, j, k)| model_z[[i, j, k]].clone().into());
        let model = NetworkBuilder::new()
            .freq(meas.freq().clone())
            .z0(meas.z0().clone())
            .z(model_z_c64)
            .build();

        calc_err_xfmr(meas, &model)
    }

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    #[test]
    fn nelder_mead_bounded_iter1_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let lb: Array1<f64> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
        let ub: Array1<f64> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
        let exemplar_res = array![
            272695.2468393795,
            326794.52891594684,
            342207.96817168326,
            342220.223907122,
            343582.48617426824,
            344033.75696385157,
            416206.242967765,
            2403422.1818864923,
            1206556.0229802025
        ];
        let exemplar_tol = f64::NAN;
        let exemplar_x = array![
            1.0883883476483184e-11,
            1.7777669529663687e-01,
            2.7677669529663682e-13,
            1.7677769529663687e-01,
            1.1767766952966369e-15,
            1.0001767766952967e+03,
            1.1767766952966369e-15,
            1.1767766952966369e-15
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = Box::new(NelderMeadBoundedOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMeadBounded::new(x.clone(), scale, lb, ub, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_f64(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_f64(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter2_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let lb: Array1<f64> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
        let ub: Array1<f64> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
        let exemplar_res = array![
            272695.2468393795,
            326794.52891594684,
            342207.96817168326,
            342220.223907122,
            343582.48617426824,
            344033.75696385157,
            416206.242967765,
            1206556.0229802025,
            6976.855033465337
        ];
        let exemplar_tol = 1.5924016727932413;
        let exemplar_x = array![
            1.0883883476483184e-11,
            1.7777669529663687e-01,
            2.7677669529663682e-13,
            1.7677769529663687e-01,
            1.1767766952966369e-15,
            1.0001767766952967e+03,
            1.1767766952966369e-15,
            1.1767766952966369e-15
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = Box::new(NelderMeadBoundedOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(2),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMeadBounded::new(x.clone(), scale, lb, ub, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_f64(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_f64(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter10_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let lb: Array1<f64> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
        let ub: Array1<f64> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
        let exemplar_res = array![
            4829.760385868178,
            6492.298384719511,
            6874.050950543377,
            6976.855033465337,
            7477.43796137955,
            22859.706730326958,
            57891.50712044708,
            103376.08526629584,
            4891.293047218964
        ];
        let exemplar_tol = 1.9304191405840867;
        let exemplar_x = array![
            1.2361669322101397e-11,
            1.0055555555555556e-06,
            5.849383897826444e-13,
            1.0055555555555556e-06,
            1.9006085977366987e-15,
            1.0008701159555326e+03,
            1.4565740713168557e-15,
            2.5661741932665137e-15
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        // let mut test =
        //     NelderMeadBounded::new(x.clone(), scale, lb, ub, net.clone(), eval_f_simplex);
        let options = Box::new(NelderMeadBoundedOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(10),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMeadBounded::new(x.clone(), scale, lb, ub, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_f64(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_f64(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter100_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let lb: Array1<f64> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
        let ub: Array1<f64> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
        let exemplar_res = array![
            3447.7296019042656,
            3447.9423463765825,
            3448.4259409304887,
            3450.039717508152,
            3450.5038774187087,
            3451.5888673344157,
            3451.7919206666056,
            3451.8941427266477,
            3451.836766408438
        ];
        let exemplar_tol = 0.0012174654891831463;
        let exemplar_x = array![
            2.3804481112658933e-11,
            1.0059277453279175e-6,
            7.013163931317152e-13,
            0.5815528215099403,
            4.947616036977013e-15,
            1001.6662201191243,
            5.401451482867253e-16,
            1.1334435942547104e-14
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = Box::new(NelderMeadBoundedOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(100),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMeadBounded::new(x.clone(), scale, lb, ub, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_f64(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_f64(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    // #[test]
    // fn nelder_mead_bounded_xfmr() {
    //     let z0 = 50.0;
    //     let freq = FrequencyBuilder::new()
    //         .freqs_scaled(array![275.0], Scale::Giga)
    //         .build();
    //     let src = ImpedanceBuilder::new()
    //         .kind(ImpedanceType::Z)
    //         .category(ComplexNumberType::ReIm)
    //         .mode(ImpedanceMode::Se)
    //         // .re(42.4)
    //         // .im(-19.6)
    //         .re(50.0)
    //         .im(0.0)
    //         .z0(z0)
    //         .freq(freq.unitval(0))
    //         .build();
    //     let load = ImpedanceBuilder::new()
    //         .kind(ImpedanceType::Z)
    //         .category(ComplexNumberType::ReIm)
    //         .mode(ImpedanceMode::Se)
    //         // .re(212.3)
    //         // .im(-43.2)
    //         .re(100.0)
    //         .im(0.0)
    //         .z0(z0)
    //         .freq(freq.unitval(0))
    //         .build();
    //     let x: Array1<f64> = array![1e-11];
    //     let scale: Array1<f64> = array![1e12];
    //     let lb: Array1<f64> = array![1e-15];
    //     let ub: Array1<f64> = array![1e-9];
    //     let exemplar_res = array![3444.854217601127,];
    //     let exemplar_tol = 0.0014246249880836607;
    //     let exemplar_x = array![3.498285705078592e-12];
    //     let exemplar_km = 0.4;
    //     let exemplar_qp = 25.0;
    //     let exemplar_qs = 15.0;
    //     let ls = -(((exemplar_qs * load.z().im.powi(2) + exemplar_qs * load.z().re.powi(2))
    //         * src.z().im)
    //         / ((2.0 * exemplar_qs * load.z().im + 2.0 * load.z().re) * src.z().im
    //             + exemplar_qs * load.z().im.powi(2)
    //             + exemplar_qs * load.z().re.powi(2)))
    //         / freq.w()[0];

    //     let net = NetworkBuilder::new()
    //         .freq(freq)
    //         .z0(array![src.z(), load.z()])
    //         .s(points![[
    //             [c64(0.0, 0.0), c64(1.0, 0.0)],
    //             [c64(1.0, 0.0), c64(0.0, 0.0)]
    //         ]])
    //         .build();

    //     let mut test = NelderMeadBounded::new(x, scale, lb, ub, move |x| {
    //         eval_f_xfmr(x, ls, exemplar_km, exemplar_qp, exemplar_qs, &net)
    //     });
    //     test.minimize(Some(50));

    //     let margin = MARGIN;
    //     comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_f64(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_f64(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter200_() {
    //     let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<f64> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<f64> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
    //     let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
    //     let net = read_touchstone(&filename).unwrap();
    //     let mut test =
    //         NelderMeadBounded::new(x.clone(), scale, lb, ub, net.clone(), eval_f_simplex);

    //     let margin = MARGIN;
    //     let exemplar_res = array![
    //         818.7670362019819,
    //         819.7858499276429,
    //         819.8191081224865,
    //         819.8818753756948,
    //         820.072179669824,
    //         820.400571229628,
    //         820.6580990875366,
    //         820.6632007827918,
    //         819.4152758850769
    //     ];
    //     let exemplar_tol = 0.0026640279313826782;
    //     let exemplar_x = array![
    //         2.5558359727448667e-11,
    //         1.0009436575297008e-6,
    //         8.640548821790739e-13,
    //         0.04575718526325763,
    //         1.1871308027782806e-15,
    //         987.0763620759078,
    //         1.311157416982076e-16,
    //         9.087060407601609e-15
    //     ];
    //     let mu = 1.0;
    //     test.set_mu(mu);
    //     assert_eq!(mu, test.get_mu());
    //     test.minimize(Some(200));
    //     comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x, mu=1.0)");
    //     comp_row_f64(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=1.0)",
    //     );
    //     comp_f64(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol, mu=1.0)",
    //         "",
    //     );

    //     let exemplar_res = array![
    //         555.311267334037,
    //         555.3896833543715,
    //         555.5822722884807,
    //         555.9175385731802,
    //         556.5515771850778,
    //         556.820749296749,
    //         556.9013339933028,
    //         556.9337932429075,
    //         556.1600388642868
    //     ];
    //     let exemplar_tol = 0.003641404147217269;
    //     let exemplar_x = array![
    //         2.528067770783543e-11,
    //         1.0006951943403888e-6,
    //         1.0169226355037147e-12,
    //         0.023668253316077553,
    //         1.1089003073348476e-14,
    //         1013.2824818461836,
    //         9.197901642318022e-16,
    //         8.668218207170575e-15
    //     ];
    //     test.set_x(x.clone());
    //     let mu = 0.1;
    //     test.set_mu(mu);
    //     assert_eq!(mu, test.get_mu());
    //     test.minimize(Some(200));
    //     comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x, mu=0.1)");
    //     comp_row_f64(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=0.1)",
    //     );
    //     comp_f64(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol, mu=0.1)",
    //         "",
    //     );

    //     let exemplar_res = array![
    //         542.4623317818372,
    //         542.6035903708896,
    //         542.7839845275608,
    //         542.9156097461646,
    //         542.9335968602808,
    //         542.9503436944373,
    //         543.0027100742718,
    //         543.017174454764,
    //         542.5970783474999
    //     ];
    //     let exemplar_tol = 0.0011096407061036788;
    //     let exemplar_x = array![
    //         2.427259390407293e-11,
    //         1.009829574420645e-6,
    //         6.351662277864478e-13,
    //         0.026075078478467278,
    //         7.621742679101595e-15,
    //         1006.9703272139681,
    //         1.1713222130218483e-15,
    //         8.873892874525603e-15
    //     ];
    //     test.set_x(x.clone());
    //     let mu = 0.01;
    //     test.set_mu(mu);
    //     assert_eq!(mu, test.get_mu());
    //     test.minimize(Some(200));
    //     comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x, mu=0.01)");
    //     comp_row_f64(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=0.01)",
    //     );
    //     comp_f64(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol, mu=0.01)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter500_() {
    //     let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<f64> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<f64> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
    //     let exemplar_res = array![
    //         3203.2879708700866,
    //         3203.2949572112325,
    //         3203.2954429260863,
    //         3203.2955724351086,
    //         3203.299995287214,
    //         3203.300056320635,
    //         3203.3004882934383,
    //         3203.3007569470956,
    //         3203.2930641437715
    //     ];
    //     let exemplar_tol = 0.000005526963054275563;
    //     let exemplar_x = array![
    //         2.745441131485568e-11,
    //         5.802690745506808e-6,
    //         2.1735312351394178e-11,
    //         0.02148541121361614,
    //         4.0575542618309124e-13,
    //         2851.5938365163875,
    //         1.6942970452531605e-15,
    //         7.731480438940903e-15
    //     ];

    //     let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
    //     let net = read_touchstone(&filename).unwrap();
    //     let mut test =
    //         NelderMeadBounded::new(x.clone(), scale, lb, ub, net.clone(), eval_f_simplex);
    //     test.minimize(Some(500));

    //     let margin = MARGIN;
    //     comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_f64(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_f64(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter1000_() {
    //     let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<f64> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<f64> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
    //     let exemplar_res = array![
    //         3203.2829155938794,
    //         3203.2829156375756,
    //         3203.282916068842,
    //         3203.2829162901735,
    //         3203.282916290638,
    //         3203.2829162913545,
    //         3203.282916374713,
    //         3203.28291642157,
    //         3203.282916681512
    //     ];
    //     let exemplar_tol = 0.0000000003395367987351268;
    //     let exemplar_x = array![
    //         2.7406879533217748e-11,
    //         5.842620278464627e-6,
    //         2.193783307045718e-11,
    //         0.0214722393586234,
    //         4.091786190272255e-13,
    //         2866.7924256179062,
    //         1.6948944506279059e-15,
    //         7.754134539035408e-15
    //     ];

    //     let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
    //     let net = read_touchstone(&filename).unwrap();
    //     let mut test =
    //         NelderMeadBounded::new(x.clone(), scale, lb, ub, net.clone(), eval_f_simplex);
    //     test.minimize(Some(1000));

    //     let margin = MARGIN;
    //     comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_f64(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_f64(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }
}
