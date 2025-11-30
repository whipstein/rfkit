#![allow(dead_code)]
use crate::{
    error::MinimizerError,
    minimize::{Minimizer, MinimizerOptions, MinimizerResult, f64::ObjFn},
};
use ndarray::prelude::*;

/// Result of Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadResult {
    pub xmin: Array1<f64>,
    pub fmin: f64,
    pub tolerance: f64,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_simplex_size: f64,
    pub history: Array1<f64>,
}

impl MinimizerResult<Array1<f64>, f64> for NelderMeadResult {
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
pub struct NelderMeadOptions {
    initial_point: Array1<f64>,
    scale: Array1<f64>,
    max_iterations: usize,
    tolerance: f64,
    alpha: f64,                  // Reflection coefficient
    beta: f64,                   // Contraction coefficient
    gamma: f64,                  // Expansion coefficient
    rho: f64,                    // Scaling coefficient
    use_random_restart: bool,    // Enable random restarts when stuck
    use_adaptive_simplex: bool,  // Use adaptive initial simplex sizing
    stagnation_iters: usize,     // Counter for detecting stagnation
    stagnation_threshold: usize, // Max iterations without improvement before restart
    verbosity: usize,
}

impl NelderMeadOptions {
    pub fn new(
        init: Array1<f64>,
        scale: Option<Array1<f64>>,
        max_iters: Option<usize>,
        tol: Option<f64>,
        alpha: Option<f64>,
        beta: Option<f64>,
        gamma: Option<f64>,
        rho: Option<f64>,
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

impl MinimizerOptions<Array1<f64>, f64> for NelderMeadOptions {
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

pub struct NelderMead {
    scale: Array1<f64>,
    x: Array1<f64>,
    x_scaled: Array1<f64>,
    res: Option<Array1<f64>>,
    simplex: Option<Array2<f64>>,
    fn_evals: usize,
    f: Box<dyn ObjFn<f64>>,
    n: usize,
    iters: usize,
    max_iters: usize,
    tol: Option<f64>,
    target_tol: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    rho: f64,
    verbosity: u32,
}

impl NelderMead {
    pub fn new<F>(x: Array1<f64>, scale: Array1<f64>, f: F) -> Self
    where
        F: ObjFn<f64> + 'static,
    {
        NelderMead {
            x_scaled: Array1::from_shape_fn(x.len(), |i| x[i] * scale[i]),
            scale,
            x: x.clone(),
            res: None,
            simplex: None,
            fn_evals: 0,
            f: Box::new(f),
            n: x.len(),
            iters: 0,
            max_iters: 1, // Maximum iterations
            tol: None,
            target_tol: 1e-12, // Convergent tolerance
            alpha: 1.0,        // Reflection coefficient
            beta: 0.5,         // Contraction coefficient
            gamma: 2.0,        // Expansion coefficient
            rho: 0.5,          // Scaling coefficient
            verbosity: 0,
        }
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

    pub fn set_x(&mut self, x: Array1<f64>) {
        self.x = x;
        self.x_scaled = Array1::from_shape_fn(self.x.len(), |i| self.x[i] * self.scale[i]);
        self.res = None;
        self.simplex = None;
        self.tol = None;
    }

    pub fn set_verbosity(&mut self, verbose: u32) {
        self.verbosity = verbose;
    }

    pub fn set_target_tolerance(&mut self, tol: f64) {
        self.target_tol = tol;
    }

    pub fn calc_obj(&mut self, x: &Array1<f64>) -> f64 {
        let _x_debug = x.clone().to_vec();
        let _f_debug = self.f.call(x);
        self.fn_evals += 1;
        self.f.call(x)
    }

    fn x(&self) -> &Array1<f64> {
        &self.x
    }

    fn final_value(&self) -> Option<f64> {
        self.res.as_ref().map(|x| x[0])
    }

    fn tolerance(&self) -> Option<f64> {
        self.tol
    }

    fn iterations(&self) -> usize {
        self.iters
    }

    fn name(&self) -> &str {
        "NelderMead"
    }

    fn minimize_opt(
        &mut self,
        opt: Box<NelderMeadOptions>,
    ) -> Result<Box<NelderMeadResult>, MinimizerError> {
        self.scale = opt.scale;
        self.target_tol = opt.tolerance;
        self.set_x(opt.initial_point);
        self.max_iters = opt.max_iterations;
        self.iters = 0;
        self.fn_evals = 0;
        let mut history: Vec<f64> = vec![];

        // Generate initial simplex veritces
        let c: f64 = 1.0;
        let b = c / (self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0);
        let a = b + c / 2f64.sqrt();
        let _ncols = self.n;
        let nrows = self.n + 1;
        let mut simplex = Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
            if i == j && i < self.n {
                self.x_scaled[j] + a + b
            } else if i < self.n {
                self.x_scaled[j] + b
            } else {
                self.x_scaled[j]
            }
        });

        // Evaluate function at simplex vertices
        let mut res: Array1<f64> = simplex
            .rows()
            .into_iter()
            .map(|x| self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x[i] / self.scale[i])))
            .collect();
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

            // Sort points from best to worst
            let mut order: Vec<usize> = (0..res.len()).collect();
            order.sort_by(|&a, &b| res[a].partial_cmp(&res[b]).unwrap());
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
            let tol = 2.0 * (res[res.len() - 1] - res[0]).abs()
                / (res[res.len() - 1].abs() + res[0].abs() + 1e-15);
            if tol.is_finite()
                && (tol < self.target_tol
                    || ((prev_tol - tol).abs() < 1e-15 && prev_tol.is_finite()))
            {
                break;
            }
            prev_tol = tol.clone();

            let x_b = Array1::from_shape_fn(self.n, |i| simplex[(0, i)]); // Best point
            let _x_l = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 2, i)]); // Lousy point
            let x_w = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 1, i)]); // Worst point

            // Calculate the average of the n best points
            let mut x_avg = Array1::zeros(self.n);
            for i in 0..self.n {
                let mut sum = 0.0;
                for j in 0..self.n {
                    sum += simplex[(j, i)];
                }
                x_avg[i] = 1.0 / self.n as f64 * sum;
            }

            // Calculate reflection point
            let x_r =
                Array1::from_shape_fn(self.n, |i| x_avg[i] + self.alpha * (x_avg[i] - x_w[i]));
            let f_r = self.calc_obj(&Array1::from_shape_fn(self.n, |i| x_r[i] / self.scale[i]));

            //
            // Determine simplex adjustment
            //
            if f_r <= res[0] {
                // Perform expansion
                if self.verbosity > 1 {
                    println!("performing expansion");
                }
                let x_e =
                    Array1::from_shape_fn(self.n, |i| x_r[i] + self.gamma * (x_r[i] - x_avg[i]));
                let f_e = self.calc_obj(&Array1::from_shape_fn(self.n, |i| x_e[i] / self.scale[i]));
                if f_e < res[0] {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_e[i];
                    }
                    res[nrows - 1] = f_e;
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_r[i];
                    }
                    res[nrows - 1] = f_r;
                }
            } else if f_r > res[nrows - 1] {
                // Perform inside contraction
                if self.verbosity > 1 {
                    println!("performing inside contraction");
                }
                let x_ic =
                    Array1::from_shape_fn(self.n, |i| x_avg[i] - self.beta * (x_avg[i] - x_w[i]));
                let f_ic =
                    self.calc_obj(&Array1::from_shape_fn(self.n, |i| x_ic[i] / self.scale[i]));
                if f_ic > res[nrows - 1] {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] =
                                x_b[j] + self.rho * (simplex[(i + 1, j)] - x_b[j]);
                        }
                        res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
                            simplex[(i + 1, j)] / self.scale[j]
                        }));
                    }
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_ic[i];
                    }
                    res[nrows - 1] = f_ic;
                }
            } else if f_r > res[nrows - 2] {
                // Perform outside contraction
                if self.verbosity > 1 {
                    println!("performing outside contraction");
                }
                let x_oc =
                    Array1::from_shape_fn(self.n, |i| x_avg[i] + self.beta * (x_avg[i] - x_w[i]));
                let f_oc =
                    self.calc_obj(&Array1::from_shape_fn(self.n, |i| x_oc[i] / self.scale[i]));
                let mut x_oc_vec: Vec<f64> = vec![];
                for i in 0..self.n {
                    x_oc_vec.push(x_oc[i]);
                }
                if f_oc > f_r {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] =
                                x_b[j] + self.rho * (simplex[(i + 1, j)] - x_b[j]);
                        }
                        res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
                            simplex[(i + 1, j)] / self.scale[j]
                        }));
                    }
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_oc[i];
                    }
                    res[nrows - 1] = f_oc;
                }
            } else {
                for i in 0..self.n {
                    simplex[(nrows - 1, i)] = x_r[i];
                }
                res[nrows - 1] = f_r;
            }

            self.tol = Some(tol);
            history.push(res[0]);
        }

        self.simplex = Some(simplex.clone());
        self.x_scaled = Array1::from_shape_fn(simplex.ncols(), |i| simplex[(0, i)]);
        self.x = Array1::from_shape_fn(self.n, |i| self.x_scaled[i] / self.scale[i]);
        self.res = Some(res.clone());

        if self.verbosity > 0 {
            println!("x: {:?}", self.x());
            println!("res: {:?}", self.res_all().unwrap());
            println!("iters: {}", self.iterations());
            println!("tol: {:?}", self.tolerance().unwrap());
        }

        Ok(Box::new(NelderMeadResult {
            xmin: self.x.clone(),
            fmin: self.f.call(&self.x),
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

impl Minimizer<Array1<f64>, f64> for NelderMead {
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

        // Create a concrete NelderMeadOptions
        let concrete_opt = Box::new(NelderMeadOptions::new(
            initial_point,
            Some(scale),
            Some(max_iters),
            Some(tolerance),
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
    // fn minimize(&mut self, max_iters: Option<usize>) -> Box<dyn MinimizerResult<Array1<f64>, f64>> {
    //     self.max_iters = match max_iters {
    //         Some(x) => x,
    //         _ => 1000,
    //     };
    //     self.iters = 0;
    //     self.fn_evals = 0;
    //     let mut history: Vec<f64> = vec![];

    //     // Generate initial simplex veritces
    //     let c: f64 = 1.0;
    //     let b = c / (self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0);
    //     let a = b + c / 2f64.sqrt();
    //     let _ncols = self.n;
    //     let nrows = self.n + 1;
    //     let mut simplex = Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
    //         if i == j && i < self.n {
    //             self.x_scaled[j] + a + b
    //         } else if i < self.n {
    //             self.x_scaled[j] + b
    //         } else {
    //             self.x_scaled[j]
    //         }
    //     });

    //     // Evaluate function at simplex vertices
    //     let mut res: Array1<f64> = simplex
    //         .rows()
    //         .into_iter()
    //         .map(|x| self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x[i] / self.scale[i])))
    //         .collect();
    //     let mut prev_tol = 1.0;
    //     history.push(res[0]);

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
    //         let _x_debug = self.x.clone().to_vec();
    //         let _x_scaled_debug = self.x_scaled.clone().to_vec();
    //         let _simplex_debug = simplex.clone().into_flat().to_vec();

    //         // Sort points from best to worst
    //         let mut order: Vec<usize> = (0..res.len()).collect();
    //         order.sort_by(|&a, &b| res[a].partial_cmp(&res[b]).unwrap());
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
    //         let _res_debug = res.clone().to_vec();

    //         // Determine if convergence criteria met
    //         let tol = 2.0 * (res[res.len() - 1] - res[0]).abs()
    //             / (res[res.len() - 1].abs() + res[0].abs() + 1e-15);
    //         if tol.is_finite()
    //             && (tol < self.target_tol
    //                 || ((prev_tol - tol).abs() < 1e-15 && prev_tol.is_finite()))
    //         {
    //             break;
    //         }
    //         prev_tol = tol.clone();

    //         let x_b = Array1::from_shape_fn(self.n, |i| simplex[(0, i)]); // Best point
    //         let _x_l = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 2, i)]); // Lousy point
    //         let x_w = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 1, i)]); // Worst point

    //         // Calculate the average of the n best points
    //         let mut x_avg = Array1::zeros(self.n);
    //         for i in 0..self.n {
    //             let mut sum = 0.0;
    //             for j in 0..self.n {
    //                 sum += simplex[(j, i)];
    //             }
    //             x_avg[i] = 1.0 / self.n as f64 * sum;
    //         }

    //         // Calculate reflection point
    //         let x_r =
    //             Array1::from_shape_fn(self.n, |i| x_avg[i] + self.alpha * (x_avg[i] - x_w[i]));
    //         let f_r = self.calc_obj(&Array1::from_shape_fn(self.n, |i| x_r[i] / self.scale[i]));

    //         //
    //         // Determine simplex adjustment
    //         //
    //         if f_r <= res[0] {
    //             // Perform expansion
    //             if self.verbosity > 1 {
    //                 println!("performing expansion");
    //             }
    //             let x_e =
    //                 Array1::from_shape_fn(self.n, |i| x_r[i] + self.gamma * (x_r[i] - x_avg[i]));
    //             let f_e = self.calc_obj(&Array1::from_shape_fn(self.n, |i| x_e[i] / self.scale[i]));
    //             if f_e < res[0] {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_e[i];
    //                 }
    //                 res[nrows - 1] = f_e;
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_r[i];
    //                 }
    //                 res[nrows - 1] = f_r;
    //             }
    //         } else if f_r > res[nrows - 1] {
    //             // Perform inside contraction
    //             if self.verbosity > 1 {
    //                 println!("performing inside contraction");
    //             }
    //             let x_ic =
    //                 Array1::from_shape_fn(self.n, |i| x_avg[i] - self.beta * (x_avg[i] - x_w[i]));
    //             let f_ic =
    //                 self.calc_obj(&Array1::from_shape_fn(self.n, |i| x_ic[i] / self.scale[i]));
    //             if f_ic > res[nrows - 1] {
    //                 // Shrink simplex
    //                 if self.verbosity > 1 {
    //                     println!("shrinking simplex");
    //                 }
    //                 for i in 0..self.n {
    //                     for j in 0..self.n {
    //                         simplex[(i + 1, j)] =
    //                             x_b[j] + self.rho * (simplex[(i + 1, j)] - x_b[j]);
    //                     }
    //                     res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
    //                         simplex[(i + 1, j)] / self.scale[j]
    //                     }));
    //                 }
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_ic[i];
    //                 }
    //                 res[nrows - 1] = f_ic;
    //             }
    //         } else if f_r > res[nrows - 2] {
    //             // Perform outside contraction
    //             if self.verbosity > 1 {
    //                 println!("performing outside contraction");
    //             }
    //             let x_oc =
    //                 Array1::from_shape_fn(self.n, |i| x_avg[i] + self.beta * (x_avg[i] - x_w[i]));
    //             let f_oc =
    //                 self.calc_obj(&Array1::from_shape_fn(self.n, |i| x_oc[i] / self.scale[i]));
    //             let mut x_oc_vec: Vec<f64> = vec![];
    //             for i in 0..self.n {
    //                 x_oc_vec.push(x_oc[i]);
    //             }
    //             if f_oc > f_r {
    //                 // Shrink simplex
    //                 if self.verbosity > 1 {
    //                     println!("shrinking simplex");
    //                 }
    //                 for i in 0..self.n {
    //                     for j in 0..self.n {
    //                         simplex[(i + 1, j)] =
    //                             x_b[j] + self.rho * (simplex[(i + 1, j)] - x_b[j]);
    //                     }
    //                     res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
    //                         simplex[(i + 1, j)] / self.scale[j]
    //                     }));
    //                 }
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_oc[i];
    //                 }
    //                 res[nrows - 1] = f_oc;
    //             }
    //         } else {
    //             for i in 0..self.n {
    //                 simplex[(nrows - 1, i)] = x_r[i];
    //             }
    //             res[nrows - 1] = f_r;
    //         }

    //         self.tol = Some(tol);
    //         history.push(res[0]);
    //     }

    //     self.simplex = Some(simplex.clone());
    //     self.x_scaled = Array1::from_shape_fn(simplex.ncols(), |i| simplex[(0, i)]);
    //     self.x = Array1::from_shape_fn(self.n, |i| self.x_scaled[i] / self.scale[i]);
    //     // self.res = Some(res);
    //     self.res = Some(res.clone());

    //     if self.verbosity > 0 {
    //         println!("x: {:?}", self.x());
    //         println!("res: {:?}", self.res_all().unwrap());
    //         println!("iters: {}", self.iterations());
    //         println!("tol: {:?}", self.tolerance().unwrap());
    //     }

    //     Box::new(NelderMeadResult {
    //         xmin: self.x.clone(),
    //         fmin: self.f.call(&self.x),
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
mod minimize_f64_neldermead_tests {
    use super::*;
    use crate::file::read_touchstone;
    use crate::frequency::Frequency;
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
            zs[i] = &x[3] * Complex::I * w[i] * x[2] / (x[3] + Complex::I * w[i] * x[2]);
            zm[i] = &x[1] + Complex::I * w[i] * x[0] + zs[i];
            zp[i] = &x[5] - Complex::I / (w[i] * x[4]);
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
        let mut err = 0.0;
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

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    #[test]
    fn nelder_mead_iter1_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let exemplar_res = 5736.50737797145;
        let exemplar_tol = 1.9918188782290724;
        let exemplar_x = array![
            1.0e-11, 1.0e-03, 1.0e-13, 1.0e-06, 1.0e-15, 1.0e03, 1.0e-15, 1.0e-15
        ];
        let exemplar_simplex = array![
            [
                1.0000000000000000e+01,
                1.0000000000000000e-03,
                1.0000000000000001e-01,
                9.9999999999999995e-07,
                1.0000000000000000e+00,
                1.0000000000000000e+03,
                1.0000000000000000e+00,
                1.0000000000000000e+00
            ],
            [
                1.1060660171779821e+01,
                1.7777669529663687e-01,
                2.7677669529663684e-01,
                1.7677769529663687e-01,
                1.1767766952966370e+00,
                1.0001767766952967e+03,
                1.1767766952966370e+00,
                1.1767766952966370e+00
            ],
            [
                1.0176776695296637e+01,
                1.7777669529663687e-01,
                2.7677669529663684e-01,
                1.0606611717798211e+00,
                1.1767766952966370e+00,
                1.0001767766952967e+03,
                1.1767766952966370e+00,
                1.1767766952966370e+00
            ],
            [
                1.0176776695296637e+01,
                1.7777669529663687e-01,
                2.7677669529663684e-01,
                1.7677769529663687e-01,
                1.1767766952966370e+00,
                1.0001767766952967e+03,
                1.1767766952966370e+00,
                2.0606601717798214e+00
            ],
            [
                1.0176776695296637e+01,
                1.7777669529663687e-01,
                2.7677669529663684e-01,
                1.7677769529663687e-01,
                1.1767766952966370e+00,
                1.0001767766952967e+03,
                2.0606601717798214e+00,
                1.1767766952966370e+00
            ],
            [
                1.0176776695296637e+01,
                1.7777669529663687e-01,
                2.7677669529663684e-01,
                1.7677769529663687e-01,
                1.1767766952966370e+00,
                1.0010606601717799e+03,
                1.1767766952966370e+00,
                1.1767766952966370e+00
            ],
            [
                1.0176776695296637e+01,
                1.7777669529663687e-01,
                2.7677669529663684e-01,
                1.7677769529663687e-01,
                2.0606601717798214e+00,
                1.0001767766952967e+03,
                1.1767766952966370e+00,
                1.1767766952966370e+00
            ],
            [
                1.0176776695296637e+01,
                1.7777669529663687e-01,
                1.1606601717798211e+00,
                1.7677769529663687e-01,
                1.1767766952966370e+00,
                1.0001767766952967e+03,
                1.1767766952966370e+00,
                1.1767766952966370e+00
            ],
            [
                1.0309359216769113e+01,
                -2.9731067331307476e-01,
                4.0935921676911446e-01,
                3.0936021676911446e-01,
                1.3093592167691148e+00,
                1.0003093592167692e+03,
                1.3093592167691148e+00,
                1.3093592167691148e+00
            ],
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let objective = move |x: &Array1<f64>| -> f64 { eval_f_simplex(x, &net) };
        let options = Box::new(NelderMeadOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMead::new(x.clone(), scale, objective);
        _ = test.minimize(options);

        println!("{}", test.final_value().unwrap());
        let margin = MARGIN;
        comp_mat_f64(
            &exemplar_simplex,
            &test.simplex().unwrap(),
            margin,
            "minimize(simplex)",
        );
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_f64(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter2_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let exemplar_res = 5736.50737797145;
        let exemplar_tol = 1.9644567884855046;
        let exemplar_x = array![
            1.0e-11, 1.0e-03, 1.0e-13, 1.0e-06, 1.0e-15, 1.0e03, 1.0e-15, 1.0e-15
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = Box::new(NelderMeadOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(2),
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMead::new(x.clone(), scale, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_f64(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter5_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let exemplar_res = 5736.50737797145;
        let exemplar_tol = 1.9338066674460301;
        let exemplar_x = array![
            1.0e-11, 1.0e-03, 1.0e-13, 1.0e-06, 1.0e-15, 1.0e03, 1.0e-15, 1.0e-15
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = Box::new(NelderMeadOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(5),
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMead::new(x.clone(), scale, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_f64(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter6_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let exemplar_res = 3988.2427199255285;
        let exemplar_tol = 1.953538011236703;
        let exemplar_x = array![
            1.0496148654572788e-11,
            1.1616959727288240e-02,
            -2.5331914829451519e-14,
            4.9614965457278737e-01,
            9.9896419905099658e-16,
            9.9939129430896890e02,
            1.4961486545727874e-15,
            1.4961486545727874e-15
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = Box::new(NelderMeadOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(6),
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMead::new(x.clone(), scale, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_f64(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter10_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let exemplar_res = 3988.2427199255285;
        let exemplar_tol = 1.8703270580082012;
        let exemplar_x = array![
            1.0496148654572788e-11,
            1.1616959727288240e-02,
            -2.5331914829451519e-14,
            4.9614965457278737e-01,
            9.9896419905099658e-16,
            9.9939129430896890e02,
            1.4961486545727874e-15,
            1.4961486545727874e-15
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        // let mut test = NelderMead::new(x.clone(), scale, net.clone(), eval_f_simplex);
        let options = Box::new(NelderMeadOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(10),
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMead::new(x.clone(), scale, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_f64(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter100_() {
        let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
        let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
        let exemplar_res = 718.9275737194077;
        let exemplar_tol = 0.01731057861988469;
        let exemplar_x = array![
            1.6110247811984210e-11,
            7.3332764346083906e-03,
            1.2618495970192951e-12,
            3.9419573384356825,
            -2.1119233982250064e-16,
            996.5630184329216,
            4.5268760596139400e-15,
            9.1487688225672705e-15
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = Box::new(NelderMeadOptions::new(
            x.clone(),
            Some(scale.clone()),
            Some(100),
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let mut test = NelderMead::new(x.clone(), scale, move |x: &Array1<f64>| {
            eval_f_simplex(x, &net)
        });
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_f64(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_f64(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    // fn nelder_mead_iter1000_() {
    //     let x: Array1<f64> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<f64> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let exemplar_res = array![
    //         507.4959059628858,
    //         507.4967205720617,
    //         507.51821424026133,
    //         507.5235865302632,
    //         507.52680461964536,
    //         507.5316821046423,
    //         507.5417921505074,
    //         507.5461487183355,
    //         507.51123384796256
    //     ];
    //     let exemplar_tol = 0.00009982680601788009;
    //     let exemplar_x = array![
    //         0.000000000026860481752007364,
    //         0.012235450601304963,
    //         0.00000000000042331577699512585,
    //         0.06078702185828572,
    //         -0.0000000000000006373686337134062,
    //         1010.7442005951383,
    //         -0.0000000000000006234256481709852,
    //         0.000000000000010028433106081285
    //     ];

    //     let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
    //     let net = read_touchstone(&filename).unwrap();
    //     let mut test = NelderMead::new(x.clone(), scale, net.clone(), eval_f_simplex);
    //     test.minimize(1000);

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
