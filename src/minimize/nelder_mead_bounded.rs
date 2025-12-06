#![allow(unused_assignments)]
#![allow(unused_variables)]
use crate::{
    error::MinimizerError,
    float::RFFloat,
    minimize::{Minimizer, MinimizerOptions, MinimizerResult, ObjFn},
};
use ndarray::prelude::*;

/// Result of Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadBoundedResult<T> {
    pub xmin: Array1<T>,
    pub fmin: T,
    pub tolerance: T,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_simplex_size: T,
    pub history: Array1<T>,
}

impl<T> MinimizerResult<Array1<T>, T> for NelderMeadBoundedResult<T>
where
    T: RFFloat,
{
    fn converged(&self) -> bool {
        self.converged
    }

    fn fmin(&self) -> T {
        self.fmin.clone()
    }

    fn tolerance(&self) -> T {
        self.tolerance.clone()
    }

    fn fn_evals(&self) -> usize {
        self.fn_evals
    }

    fn iters(&self) -> usize {
        self.iters
    }

    fn xmin(&self) -> Array1<T> {
        self.xmin.clone()
    }

    fn history(&self) -> Array1<T> {
        self.history.clone()
    }
}

#[derive(Debug, Clone)]
pub struct NelderMeadBoundedOptions<T> {
    initial_point: Array1<T>,
    scale: Array1<T>,
    max_iterations: usize,
    tolerance: T,
    alpha: T, // Reflection coefficient
    beta: T,  // Contraction coefficient
    gamma: T, // Expansion coefficient
    rho: T,   // Scaling coefficient
    mu: T,
    use_random_restart: bool,    // Enable random restarts when stuck
    use_adaptive_simplex: bool,  // Use adaptive initial simplex sizing
    stagnation_iters: usize,     // Counter for detecting stagnation
    stagnation_threshold: usize, // Max iterations without improvement before restart
    verbosity: usize,
}

impl<T> NelderMeadBoundedOptions<T>
where
    T: RFFloat,
{
    pub fn new(
        init: Array1<T>,
        scale: Option<Array1<T>>,
        max_iters: Option<usize>,
        tol: Option<T>,
        alpha: Option<T>,
        beta: Option<T>,
        gamma: Option<T>,
        rho: Option<T>,
        mu: Option<T>,
        verbosity: Option<usize>,
    ) -> Self {
        let n = init.len();
        Self {
            initial_point: init,
            scale: scale.unwrap_or(Array1::ones(n)),
            max_iterations: max_iters.unwrap_or(1000),
            tolerance: tol.unwrap_or(T::from_f64(0.0)),
            alpha: alpha.unwrap_or(T::from_f64(1.0)),
            beta: beta.unwrap_or(T::from_f64(0.5)),
            gamma: gamma.unwrap_or(T::from_f64(2.0)),
            rho: rho.unwrap_or(T::from_f64(0.5)),
            mu: mu.unwrap_or(T::from_f64(10.0)),
            use_random_restart: true,
            use_adaptive_simplex: true,
            stagnation_iters: 0,
            stagnation_threshold: 50,
            verbosity: verbosity.unwrap_or(0),
        }
    }

    pub fn set_initial_point(&mut self, init: Array1<T>) {
        self.initial_point = init;
    }

    pub fn set_scale(&mut self, scale: Array1<T>) {
        self.scale = scale;
    }

    pub fn set_max_iterations(&mut self, iters: usize) {
        self.max_iterations = iters;
    }

    pub fn set_tolerance(&mut self, tol: T) {
        self.tolerance = tol;
    }

    pub fn set_alpha(&mut self, alpha: T) {
        self.alpha = alpha;
    }

    pub fn set_beta(&mut self, beta: T) {
        self.beta = beta;
    }

    pub fn set_gamma(&mut self, gamma: T) {
        self.gamma = gamma;
    }

    pub fn set_rho(&mut self, rho: T) {
        self.rho = rho;
    }

    pub fn set_mu(&mut self, mu: T) {
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

impl<T> MinimizerOptions<Array1<T>, T> for NelderMeadBoundedOptions<T>
where
    T: RFFloat,
{
    fn initial_point(&self) -> Array1<T> {
        self.initial_point.clone()
    }

    fn scale(&self) -> Array1<T> {
        self.scale.clone()
    }

    fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    fn tolerance(&self) -> T {
        self.tolerance.clone()
    }

    fn verbosity(&self) -> usize {
        self.verbosity
    }
}

pub struct NelderMeadBounded<T> {
    scale: Array1<T>,
    x: Array1<T>,
    x_scaled: Array1<T>,
    lb: Array1<T>,
    lb_vec: Vec<T>,
    ub: Array1<T>,
    ub_vec: Vec<T>,
    res: Option<Array1<T>>,
    simplex: Option<Array2<T>>,
    f: Box<dyn ObjFn<T>>,
    n: usize,
    iters: usize,
    max_iters: usize, // Maximum iterations
    tol: Option<T>,
    target_tol: T, // Convergent tolerance
    alpha: T,      // Reflection coefficient
    beta: T,       // Contraction coefficient
    gamma: T,      // Expansion coefficient
    rho: T,        // Scaling coefficient
    mu: T,
    verbosity: u32,
}

impl<T> NelderMeadBounded<T>
where
    T: RFFloat,
{
    pub fn new<F>(x: Array1<T>, scale: Array1<T>, lb: Array1<T>, ub: Array1<T>, f: F) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        let mut lb_vec: Vec<T> = vec![];
        let mut ub_vec: Vec<T> = vec![];
        for i in 0..x.len() {
            lb_vec.push(lb[i].clone());
            ub_vec.push(ub[i].clone());
        }
        NelderMeadBounded {
            x_scaled: Array1::from_shape_fn(x.len(), |i| x[i].clone() * scale[i].clone()),
            x: x.clone(),
            lb_vec,
            ub_vec,
            scale,
            lb,
            ub,
            res: None,
            simplex: None,
            f: Box::new(f),
            n: x.len(),
            iters: 0,
            max_iters: 1,
            tol: None,
            target_tol: T::from_f64(1e-12),
            alpha: T::from_f64(1.0),
            beta: T::from_f64(0.5),
            gamma: T::from_f64(2.0),
            rho: T::from_f64(0.5),
            mu: T::from_f64(10.0),
            verbosity: 0,
        }
    }

    fn check_bounds(&self, x: &T, lb: &T, ub: &T, scale: &T) -> T {
        if *x < (lb.clone() * scale.clone()) {
            return T::from_f64(1.0 + 0.05 / self.iters as f64) * lb.clone() * scale.clone();
        } else if *x > (ub.clone() * scale.clone()) {
            return T::from_f64(1.0 - 0.05 / self.iters as f64) * ub.clone() * scale.clone();
        }
        x.clone()
    }

    fn res_all(&self) -> Option<Array1<T>> {
        self.res.clone()
    }

    fn x_scaled(&self) -> Array1<T> {
        self.x_scaled.clone()
    }

    fn simplex(&self) -> Option<Array2<T>> {
        self.simplex.clone()
    }

    pub fn get_mu(&self) -> T {
        self.mu.clone()
    }

    pub fn set_mu(&mut self, mu: &T) {
        self.mu = mu.clone();
    }

    pub fn set_target_tolerance(&mut self, tol: &T) {
        self.target_tol = tol.clone();
    }

    pub fn set_x(&mut self, x: &Array1<T>) {
        self.x = x.clone();
        self.x_scaled =
            Array1::from_shape_fn(self.x.len(), |i| self.x[i].clone() * self.scale[i].clone());
        self.res = None;
        self.simplex = None;
        self.tol = None;
    }

    pub fn set_verbosity(&mut self, verbose: u32) {
        self.verbosity = verbose;
    }

    pub fn calc_obj(&mut self, x: &Array1<T>) -> T {
        let mut sum = T::from_f64(0.0);
        for i in 0..x.len() {
            sum +=
                (self.ub[i].clone() - x[i].clone()).ln() + (x[i].clone() - self.lb[i].clone()).ln();
        }
        self.f.call(x) - self.mu.clone() * sum
    }

    pub fn x(&self) -> &Array1<T> {
        &self.x
    }

    pub fn final_value(&self) -> Option<T> {
        self.res.as_ref().map(|x| x[0].clone())
    }

    pub fn tolerance(&self) -> Option<T> {
        self.tol.clone()
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
        opt: Box<NelderMeadBoundedOptions<T>>,
    ) -> Box<NelderMeadBoundedResult<T>> {
        self.scale = opt.scale.clone();
        self.target_tol = opt.tolerance.clone();
        self.set_x(&opt.initial_point);
        self.max_iters = opt.max_iterations;
        self.iters = 0;

        // Generate initial simplex veritces
        let c = T::from_f64(1.0);
        let b = c.clone()
            / T::from_f64((self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0));
        let a = c.clone() / T::from_f64(2f64.sqrt());
        let _ncols = self.n;
        let nrows = self.n + 1;
        let mut simplex = Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
            if i == j && i < self.n {
                self.check_bounds(
                    &(self.x_scaled[j].clone() + a.clone() + b.clone()),
                    &self.lb[j],
                    &self.ub[j],
                    &self.scale[j],
                )
            } else if i < self.n {
                self.check_bounds(
                    &(self.x_scaled[j].clone() + b.clone()),
                    &self.lb[j],
                    &self.ub[j],
                    &self.scale[j],
                )
            } else {
                self.x_scaled[j].clone()
            }
        });

        // Evaluate function at simplex vertices
        let mut res: Array1<T> = simplex
            .rows()
            .into_iter()
            .map(|x| {
                self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
                    x[i].clone() / self.scale[i].clone()
                }))
            })
            .collect();
        let mut prev_tol = T::from_f64(1.0);

        while self.iters < self.max_iters {
            if self.verbosity > 1 {
                println!(
                    "iteration: {}\terr: {}\ttol: {}",
                    self.iters,
                    res[0],
                    self.tol.clone().unwrap_or(T::from_f64(1.0))
                );
            }
            self.iters += 1;

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
            self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| {
                simplex[(i, j)].clone()
            }));

            // Determine if convergence criteria met
            let tol = T::from_f64(2.0) * (res[res.len() - 1].clone() - res[0].clone()).abs()
                / (res[res.len() - 1].abs() + res[0].abs() + T::from_f64(1e-10));
            // if tol.to_f64() < self.target_tol || (&prev_tol - &tol).abs().to_f64() < 1e-15 {
            //     break;
            // }
            match opt.tolerance.to_f64() {
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
            let mut x_avg = Array1::<T>::zeros(self.n);
            for i in 0..self.n {
                let mut sum = T::from_f64(0.0);
                for j in 0..self.n {
                    sum += simplex[(j, i)].clone();
                }
                x_avg[i] = T::from_f64(1.0 / self.n as f64) * sum;
            }

            // Calculate reflection point
            let x_r = Array1::from_shape_fn(self.n, |i| {
                self.check_bounds(
                    &(x_avg[i].clone() + self.alpha.clone() * (x_avg[i].clone() - x_w[i].clone())),
                    &self.lb[i],
                    &self.ub[i],
                    &self.scale[i],
                )
            });
            let f_r = self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
                x_r[i].clone() / self.scale[i].clone()
            }));

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
                        &(x_r[i].clone()
                            + self.gamma.clone() * (x_r[i].clone() - x_avg[i].clone())),
                        &self.lb[i],
                        &self.ub[i],
                        &self.scale[i],
                    )
                });
                let f_e = self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
                    x_e[i].clone() / self.scale[i].clone()
                }));
                if f_e < res[0] {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_e[i].clone();
                    }
                    res[nrows - 1] = f_e.clone();
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_r[i].clone();
                    }
                    res[nrows - 1] = f_r.clone();
                }
            } else if f_r > res[nrows - 1] {
                // Perform inside contraction
                if self.verbosity > 1 {
                    println!("performing inside contraction");
                }
                let x_ic = Array1::from_shape_fn(self.n, |i| {
                    self.check_bounds(
                        &(x_avg[i].clone()
                            - self.beta.clone() * (x_avg[i].clone() - x_w[i].clone())),
                        &self.lb[i],
                        &self.ub[i],
                        &self.scale[i],
                    )
                });
                let f_ic = self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
                    x_ic[i].clone() / self.scale[i].clone()
                }));
                if f_ic > res[nrows - 1] {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] = self.check_bounds(
                                &(x_b[j].clone()
                                    + self.rho.clone()
                                        * (simplex[(i + 1, j)].clone() - x_b[j].clone())),
                                &self.lb[j],
                                &self.ub[j],
                                &self.scale[j],
                            );
                        }

                        res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
                            simplex[(i + 1, j)].clone() / self.scale[j].clone()
                        }));
                    }
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_ic[i].clone();
                    }
                    res[nrows - 1] = f_ic.clone();
                }
            } else if f_r > res[nrows - 2] {
                // Perform outside contraction
                if self.verbosity > 1 {
                    println!("performing outside contraction");
                }
                let x_oc = Array1::from_shape_fn(self.n, |i| {
                    self.check_bounds(
                        &(x_avg[i].clone()
                            + self.beta.clone() * (x_avg[i].clone() - x_w[i].clone())),
                        &self.lb[i],
                        &self.ub[i],
                        &self.scale[i],
                    )
                });
                let f_oc = self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
                    x_oc[i].clone() / self.scale[i].clone()
                }));
                if f_oc > f_r {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] = self.check_bounds(
                                &(x_b[j].clone()
                                    + self.rho.clone()
                                        * (simplex[(i + 1, j)].clone() - x_b[j].clone())),
                                &self.lb[j],
                                &self.ub[j],
                                &self.scale[j],
                            );
                        }
                        res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
                            simplex[(i + 1, j)].clone() / self.scale[j].clone()
                        }));
                    }
                } else {
                    for i in 0..self.n {
                        simplex[(nrows - 1, i)] = x_oc[i].clone();
                    }
                    res[nrows - 1] = f_oc.clone();
                }
            } else {
                for i in 0..self.n {
                    simplex[(nrows - 1, i)] = x_r[i].clone();
                }
                res[nrows - 1] = f_r.clone();
            }

            self.tol = Some(tol.clone());
        }

        let shape = match simplex.shape() {
            [a, b] => (*a, *b),
            _ => panic!("Shape must be 2-dimensional"),
        };
        self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| {
            simplex[(i, j)].clone()
        }));
        self.x_scaled = Array1::from_shape_fn(simplex.ncols(), |i| simplex[(0, i)].clone());
        self.x =
            Array1::from_shape_fn(self.n, |i| self.x_scaled[i].clone() / self.scale[i].clone());
        // self.res = Some(res);
        self.res = Some(Array1::from_shape_fn(res.len(), |i| res[i].clone()));

        if self.verbosity > 0 {
            println!("x: {:?}", self.x());
            println!("res: {:?}", self.res_all().unwrap());
            println!("iters: {}", self.iterations());
            println!("tol: {:?}", self.tolerance().unwrap());
        }

        Box::new(NelderMeadBoundedResult {
            xmin: self.x.clone(),
            fmin: self.calc_obj(&self.x.clone()),
            tolerance: match self.tol.clone() {
                Some(x) => x,
                _ => T::from_f64(0.0),
            },
            iters: 0,
            fn_evals: 0,
            converged: true,
            final_simplex_size: T::from_f64(0.0),
            history: array![],
        })
    }
}

impl<T> Minimizer<Array1<T>, T> for NelderMeadBounded<T>
where
    T: RFFloat + 'static,
{
    fn minimize(
        &mut self,
        opt: Box<dyn MinimizerOptions<Array1<T>, T>>,
    ) -> Result<Box<dyn MinimizerResult<Array1<T>, T>>, MinimizerError> {
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
        let result = self.minimize_opt(concrete_opt);
        Ok(result as Box<dyn MinimizerResult<Array1<T>, T>>)
    }
    // /// Solve using Nelder-Mead algorithm
    // fn minimize(
    //     &mut self,
    //     max_iters: Option<usize>,
    // ) -> Box<dyn MinimizerResult<Array1<T>, T>> {
    //     self.max_iters = match max_iters {
    //         Some(x) => x,
    //         _ => 1000,
    //     };
    //     self.iters = 0;

    //     // Generate initial simplex veritces
    //     let c = T::new(1.0);
    //     let b = &c / (self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0);
    //     let a = &c / 2f64.sqrt();
    //     let _ncols = self.n;
    //     let nrows = self.n + 1;
    //     let mut simplex = Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
    //         if i == j && i < self.n {
    //             self.check_bounds(
    //                 &(&self.x_scaled[j] + &a + &b),
    //                 &self.lb[j],
    //                 &self.ub[j],
    //                 &self.scale[j],
    //             )
    //         } else if i < self.n {
    //             self.check_bounds(
    //                 &(&self.x_scaled[j] + &b),
    //                 &self.lb[j],
    //                 &self.ub[j],
    //                 &self.scale[j],
    //             )
    //         } else {
    //             self.x_scaled[j].clone()
    //         }
    //     });

    //     // Evaluate function at simplex vertices
    //     let mut res: Array1<T> = simplex
    //         .rows()
    //         .into_iter()
    //         .map(|x| self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x[i] / &self.scale[i])))
    //         .collect();
    //     let mut prev_tol = T::new(1.0);

    //     while self.iters < self.max_iters {
    //         if self.verbosity > 1 {
    //             println!(
    //                 "iteration: {}\terr: {}\ttol: {}",
    //                 self.iters,
    //                 res[0],
    //                 self.tol.clone().unwrap_or(T::from_f64(1.0))
    //             );
    //         }
    //         self.iters += 1;

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
    //         self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| {
    //             simplex[(i, j)].clone()
    //         }));

    //         // Determine if convergence criteria met
    //         let tol = 2.0 * (&res[res.len() - 1] - &res[0]).abs()
    //             / (res[res.len() - 1].abs() + res[0].abs() + 1e-10);
    //         // if tol.to_f64() < self.target_tol || (&prev_tol - &tol).abs().to_f64() < 1e-15 {
    //         //     break;
    //         // }
    //         if tol < self.target_tol {
    //             break;
    //         }
    //         prev_tol = tol.clone();

    //         let x_b = Array1::from_shape_fn(self.n, |i| simplex[(0, i)].clone()); // Best point
    //         let _x_l = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 2, i)].clone()); // Lousy point
    //         let x_w = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 1, i)].clone()); // Worst point

    //         // Calculate the average of the n best points
    //         let mut x_avg = Array1::<T>::zeros(self.n);
    //         for i in 0..self.n {
    //             let mut sum = T::new(0.0);
    //             for j in 0..self.n {
    //                 sum += &simplex[(j, i)].clone();
    //             }
    //             x_avg[i] = 1.0 / self.n as f64 * &sum;
    //         }

    //         // Calculate reflection point
    //         let x_r = Array1::from_shape_fn(self.n, |i| {
    //             self.check_bounds(
    //                 &(&x_avg[i] + &self.alpha * (&x_avg[i] - &x_w[i])),
    //                 &self.lb[i],
    //                 &self.ub[i],
    //                 &self.scale[i],
    //             )
    //         });
    //         let f_r = self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_r[i] / &self.scale[i]));

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
    //                     &(&x_r[i] + &self.gamma * (&x_r[i] - &x_avg[i])),
    //                     &self.lb[i],
    //                     &self.ub[i],
    //                     &self.scale[i],
    //                 )
    //             });
    //             let f_e =
    //                 self.calc_obj(&Array1::from_shape_fn(self.n, |i| &x_e[i] / &self.scale[i]));
    //             if f_e < res[0] {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_e[i].clone();
    //                 }
    //                 res[nrows - 1] = f_e.clone();
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_r[i].clone();
    //                 }
    //                 res[nrows - 1] = f_r.clone();
    //             }
    //         } else if f_r > res[nrows - 1] {
    //             // Perform inside contraction
    //             if self.verbosity > 1 {
    //                 println!("performing inside contraction");
    //             }
    //             let x_ic = Array1::from_shape_fn(self.n, |i| {
    //                 self.check_bounds(
    //                     &(&x_avg[i] - &self.beta * (&x_avg[i] - &x_w[i])),
    //                     &self.lb[i],
    //                     &self.ub[i],
    //                     &self.scale[i],
    //                 )
    //             });
    //             let f_ic = self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
    //                 &x_ic[i] / &self.scale[i]
    //             }));
    //             if f_ic > res[nrows - 1] {
    //                 // Shrink simplex
    //                 if self.verbosity > 1 {
    //                     println!("shrinking simplex");
    //                 }
    //                 for i in 0..self.n {
    //                     for j in 0..self.n {
    //                         simplex[(i + 1, j)] = self.check_bounds(
    //                             &(&x_b[j] + &self.rho * (&simplex[(i + 1, j)] - &x_b[j])),
    //                             &self.lb[j],
    //                             &self.ub[j],
    //                             &self.scale[j],
    //                         );
    //                     }

    //                     res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
    //                         &simplex[(i + 1, j)] / &self.scale[j]
    //                     }));
    //                 }
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_ic[i].clone();
    //                 }
    //                 res[nrows - 1] = f_ic.clone();
    //             }
    //         } else if f_r > res[nrows - 2] {
    //             // Perform outside contraction
    //             if self.verbosity > 1 {
    //                 println!("performing outside contraction");
    //             }
    //             let x_oc = Array1::from_shape_fn(self.n, |i| {
    //                 self.check_bounds(
    //                     &(&x_avg[i] + &self.beta * (&x_avg[i] - &x_w[i])),
    //                     &self.lb[i],
    //                     &self.ub[i],
    //                     &self.scale[i],
    //                 )
    //             });
    //             let f_oc = self.calc_obj(&Array1::from_shape_fn(self.n, |i| {
    //                 &x_oc[i] / &self.scale[i]
    //             }));
    //             if f_oc > f_r {
    //                 // Shrink simplex
    //                 if self.verbosity > 1 {
    //                     println!("shrinking simplex");
    //                 }
    //                 for i in 0..self.n {
    //                     for j in 0..self.n {
    //                         simplex[(i + 1, j)] = self.check_bounds(
    //                             &(&x_b[j] + &self.rho * (&simplex[(i + 1, j)] - &x_b[j])),
    //                             &self.lb[j],
    //                             &self.ub[j],
    //                             &self.scale[j],
    //                         );
    //                     }
    //                     res[i + 1] = self.calc_obj(&Array1::from_shape_fn(self.n, |j| {
    //                         &simplex[(i + 1, j)] / &self.scale[j]
    //                     }));
    //                 }
    //             } else {
    //                 for i in 0..self.n {
    //                     simplex[(nrows - 1, i)] = x_oc[i].clone();
    //                 }
    //                 res[nrows - 1] = f_oc.clone();
    //             }
    //         } else {
    //             for i in 0..self.n {
    //                 simplex[(nrows - 1, i)] = x_r[i].clone();
    //             }
    //             res[nrows - 1] = f_r.clone();
    //         }

    //         self.tol = Some(tol.clone());
    //     }

    //     let shape = match simplex.shape() {
    //         [a, b] => (*a, *b),
    //         _ => panic!("Shape must be 2-dimensional"),
    //     };
    //     self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| {
    //         simplex[(i, j)].clone()
    //     }));
    //     self.x_scaled = Array1::from_shape_fn(simplex.ncols(), |i| simplex[(0, i)].clone());
    //     self.x = Array1::from_shape_fn(self.n, |i| &self.x_scaled[i] / &self.scale[i]);
    //     // self.res = Some(res);
    //     self.res = Some(Array1::from_shape_fn(res.len(), |i| res[i].clone()));

    //     if self.verbosity > 0 {
    //         println!("x: {:?}", self.x());
    //         println!("res: {:?}", self.res_all().unwrap());
    //         println!("iters: {}", self.iterations());
    //         println!("tol: {:?}", self.tolerance().unwrap());
    //     }

    //     Box::new(NelderMeadBoundedResult<T> {
    //         xmin: self.x.clone(),
    //         fmin: self.calc_obj(&self.x.clone()),
    //         tolerance: match self.tol.clone() {
    //             Some(x) => x,
    //             _ => T::new(0.0),
    //         },
    //         iters: 0,
    //         fn_evals: 0,
    //         converged: true,
    //         final_simplex_size: T::from_f64(0.0),
    //         history: array![],
    //     })
    // }
}

#[cfg(test)]
mod minimize_neldermeadbounded_tests {
    use super::*;
    use crate::{
        file::read_touchstone,
        frequency::*,
        minimize::MultiDimFn,
        mycomplex::MyComplex,
        myfloat::MyFloat,
        network::{Network, NetworkBuilder},
        points::{Points, Pts},
        util::*,
    };
    use float_cmp::F64Margin;

    fn calc_feed_y(freq: Frequency, x: &Array1<MyFloat>) -> Array3<MyComplex> {
        let mut yfeed = Array3::<MyComplex>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut zs = Array1::<MyComplex>::zeros(freq.npts());
        let mut zm = Array1::<MyComplex>::zeros(freq.npts());
        let mut zp = Array1::<MyComplex>::zeros(freq.npts());
        let mut zall = Array1::<MyComplex>::zeros(freq.npts());
        for i in 0..freq.npts() {
            zs[i] = &x[3] * MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[2]
                / (&x[3] + MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[2]);
            zm[i] = &x[1] + MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[0] + &zs[i];
            zp[i] = &x[5] - MyComplex::new(0.0.into(), 1.0.into()) / (w[i] * &x[4]);
            zall[i] = &zm[i] * &zp[i] / (&zm[i] + &zp[i]);
        }

        for i in 0..freq.npts() {
            yfeed[[i, 0, 1]] = -1.0 / &zall[i];
            yfeed[[i, 1, 0]] = -1.0 / &zall[i];
            yfeed[[i, 0, 0]] =
                MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[6] + 1.0 / &zall[i];
            yfeed[[i, 1, 1]] =
                MyComplex::new(0.0.into(), 1.0.into()) * w[i] * &x[7] + 1.0 / &zall[i];
        }

        yfeed
    }

    fn calc_err(model: &Network, meas: &Network) -> MyFloat {
        let mut err = MyFloat::new(0.0);
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

    fn eval_f_simplex(x: &Array1<MyFloat>, meas: &Network) -> MyFloat {
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
        freq: &Frequency,
        ls: &MyFloat,
        rs: &MyFloat,
        km: &MyFloat,
        qp: &MyFloat,
        x: &Array1<MyFloat>,
    ) -> Array3<MyComplex> {
        let mut z = Array3::<MyComplex>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut m = Array1::<MyFloat>::zeros(freq.npts());
        let mut n = Array1::<MyFloat>::zeros(freq.npts());
        let mut rp = Array1::<MyFloat>::zeros(freq.npts());
        for i in 0..freq.npts() {
            m[i] = km * (&x[0] * ls).sqrt();
            n[i] = (ls / &x[0]).sqrt();
            rp[i] = w[i] * &x[0] / qp;
        }

        for i in 0..freq.npts() {
            z[[i, 0, 1]] = -MyComplex::new(0.0.into(), 1.0.into()) * freq.w_pt(i) * &m[i];
            z[[i, 1, 0]] = -MyComplex::new(0.0.into(), 1.0.into()) * freq.w_pt(i) * &m[i];
            z[[i, 0, 0]] = MyComplex::new(rp[i].clone(), w[i] * &x[0]);
            z[[i, 1, 1]] = MyComplex::new(rs.clone(), -w[i] * ls);
        }

        z
    }

    fn calc_err_xfmr(meas: &Network, model: &Network) -> MyFloat {
        let mut err = MyFloat::new(0.0);
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

    fn eval_f_xfmr(
        x: &Array1<MyFloat>,
        ls: &MyFloat,
        rs: &MyFloat,
        km: &MyFloat,
        qp: &MyFloat,
        meas: &Network,
    ) -> MyFloat {
        let model_z = calc_xfmr_z(&meas.freq(), ls, rs, km, qp, x);
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
        let x: Array1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ];
        let scale: Array1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ];
        let lb: Array1<MyFloat> = array![
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1.0.into(),
            1e-18.into(),
            1e-18.into()
        ];
        let ub: Array1<MyFloat> = array![
            1e-9.into(),
            1.0.into(),
            1e-9.into(),
            1.0.into(),
            1e-12.into(),
            1e6.into(),
            1e-12.into(),
            1e-12.into()
        ];
        let exemplar_res = array![
            MyFloat::new(34472.62826006022),
            38483.43015651799.into(),
            41372.522571978014.into(),
            41378.9459201732.into(),
            43347.204418114095.into(),
            43450.74945866097.into(),
            70738.75601950094.into(),
            2222749.170091806.into(),
            582197.2281772306.into()
        ];
        let exemplar_tol = f64::NAN.into();
        let exemplar_x = array![
            MyFloat::new(1.0883883476483184e-11),
            0.04519417382415922.into(),
            2.7677669529663682e-13.into(),
            0.04419517382415922.into(),
            1.1767766952966369e-15.into(),
            1000.0441941738242.into(),
            1.1767766952966369e-15.into(),
            1.1767766952966369e-15.into()
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
        let mut test = NelderMeadBounded::new(
            x.clone(),
            scale,
            lb,
            ub,
            MultiDimFn::new(move |x: &Array1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_myfloat(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter2_() {
        let x: Array1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ];
        let scale: Array1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ];
        let lb: Array1<MyFloat> = array![
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1.0.into(),
            1e-18.into(),
            1e-18.into()
        ];
        let ub: Array1<MyFloat> = array![
            1e-9.into(),
            1.0.into(),
            1e-9.into(),
            1.0.into(),
            1e-12.into(),
            1e6.into(),
            1e-12.into(),
            1e-12.into()
        ];
        let exemplar_res = array![
            34472.62826006022.into(),
            38483.43015651799.into(),
            41372.522571978014.into(),
            41378.9459201732.into(),
            43347.204418114095.into(),
            43450.74945866097.into(),
            70738.75601950094.into(),
            582197.2281772306.into(),
            6696.610974817988.into()
        ];
        let exemplar_tol = 1.938911402884323.into();
        let exemplar_x = array![
            1.0883883476483184e-11.into(),
            0.04519417382415922.into(),
            2.7677669529663682e-13.into(),
            0.04419517382415922.into(),
            1.1767766952966369e-15.into(),
            1000.0441941738242.into(),
            1.1767766952966369e-15.into(),
            1.1767766952966369e-15.into()
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
        let mut test = NelderMeadBounded::new(
            x.clone(),
            scale,
            lb,
            ub,
            MultiDimFn::new(move |x: &Array1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_myfloat(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter10_() {
        let x: Array1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ];
        let scale: Array1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ];
        let lb: Array1<MyFloat> = array![
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1.0.into(),
            1e-18.into(),
            1e-18.into()
        ];
        let ub: Array1<MyFloat> = array![
            1e-9.into(),
            1.0.into(),
            1e-9.into(),
            1.0.into(),
            1e-12.into(),
            1e6.into(),
            1e-12.into(),
            1e-12.into()
        ];
        let exemplar_res = array![
            4661.545515813834.into(),
            5505.196136761653.into(),
            6081.625223914001.into(),
            6116.822531002638.into(),
            6696.610974817988.into(),
            7577.93497452239.into(),
            8960.382934958692.into(),
            12576.764010367328.into(),
            4846.708605585247.into()
        ];
        let exemplar_tol = 1.523531985879034.into();
        let exemplar_x = array![
            1.2361669322101397e-11.into(),
            1.0055555555555556e-06.into(),
            5.849383897826444e-13.into(),
            0.9916666666666667.into(),
            1.9006085977366987e-15.into(),
            998.5403838293637.into(),
            1.4565740713168557e-15.into(),
            2.5661741932665137e-15.into()
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
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
        let mut test = NelderMeadBounded::new(
            x.clone(),
            scale,
            lb,
            ub,
            MultiDimFn::new(move |x: &Array1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_myfloat(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance().unwrap(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn nelder_mead_bounded_iter100_() {
        let x: Array1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ];
        let scale: Array1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ];
        let lb: Array1<MyFloat> = array![
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1e-6.into(),
            1e-15.into(),
            1.0.into(),
            1e-18.into(),
            1e-18.into()
        ];
        let ub: Array1<MyFloat> = array![
            1e-9.into(),
            1.0.into(),
            1e-9.into(),
            1.0.into(),
            1e-12.into(),
            1e6.into(),
            1e-12.into(),
            1e-12.into()
        ];
        let exemplar_res = array![
            3425.6196627417.into(),
            3426.9294925346976.into(),
            3427.1583551128997.into(),
            3427.2281799174566.into(),
            3427.2353930328595.into(),
            3427.6816934631543.into(),
            3428.0395457254826.into(),
            3428.2925608812147.into(),
            3426.7925753080654.into()
        ];
        let exemplar_tol = 0.0009633892025137282.into();
        let exemplar_x = array![
            2.3804481112658933e-11.into(),
            1.0059277453279175e-6.into(),
            7.013163931317152e-13.into(),
            0.5593096729128388.into(),
            4.947616036977013e-15.into(),
            977.8167045688951.into(),
            5.401451482867253e-16.into(),
            1.1334435942547104e-14.into()
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
        let mut test = NelderMeadBounded::new(
            x.clone(),
            scale,
            lb,
            ub,
            MultiDimFn::new(move |x: &Array1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
        comp_row_myfloat(
            &exemplar_res,
            &test.res_all().unwrap(),
            margin,
            "minimize(res)",
        );
        comp_myfloat(
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
    //     let x: Array1<MyFloat> = array![1e-11];
    //     let scale: Array1<MyFloat> = array![1e12];
    //     let lb: Array1<MyFloat> = array![1e-15];
    //     let ub: Array1<MyFloat> = array![1e-9];
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
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter200_() {
    //     let x: Array1<MyFloat> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<MyFloat> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<MyFloat> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<MyFloat> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
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
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x, mu=1.0)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=1.0)",
    //     );
    //     comp_myfloat(
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
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x, mu=0.1)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=0.1)",
    //     );
    //     comp_myfloat(
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
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x, mu=0.01)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res, mu=0.01)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol, mu=0.01)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter500_() {
    //     let x: Array1<MyFloat> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<MyFloat> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<MyFloat> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<MyFloat> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
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
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }

    // #[test]
    // fn nelder_mead_bounded_iter1000_() {
    //     let x: Array1<MyFloat> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<MyFloat> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
    //     let lb: Array1<MyFloat> = array![1e-15, 1e-6, 1e-15, 1e-6, 1e-15, 1.0, 1e-18, 1e-18];
    //     let ub: Array1<MyFloat> = array![1e-9, 1.0, 1e-9, 1.0, 1e-12, 1e6, 1e-12, 1e-12];
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
    //     comp_row_myfloat(&exemplar_x, &test.x(), margin, "minimize(x)");
    //     comp_row_myfloat(
    //         &exemplar_res,
    //         &test.res_all().unwrap(),
    //         margin,
    //         "minimize(res)",
    //     );
    //     comp_myfloat(
    //         &exemplar_tol,
    //         &test.tolerance().unwrap(),
    //         margin,
    //         "minimize(tol)",
    //         "",
    //     );
    // }
}
