#![allow(dead_code)]
use crate::{
    error::MinimizerError,
    float::RFFloat,
    minimize::{Minimizer, MinimizerOptions, MinimizerResult, ObjFn},
};
use ndarray::prelude::*;

/// Result of Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadResult<T> {
    pub xmin: Array1<T>,
    pub fmin: T,
    pub tolerance: T,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_simplex_size: T,
    pub history: Array1<T>,
}

impl<T> MinimizerResult<Array1<T>, T> for NelderMeadResult<T>
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
pub struct NelderMeadOptions<T> {
    initial_point: Array1<T>,
    scale: Array1<T>,
    max_iterations: usize,
    tolerance: T,
    alpha: T,                    // Reflection coefficient
    beta: T,                     // Contraction coefficient
    gamma: T,                    // Expansion coefficient
    rho: T,                      // Scaling coefficient
    use_random_restart: bool,    // Enable random restarts when stuck
    use_adaptive_simplex: bool,  // Use adaptive initial simplex sizing
    stagnation_iters: usize,     // Counter for detecting stagnation
    stagnation_threshold: usize, // Max iterations without improvement before restart
    verbosity: usize,
}

impl<T> NelderMeadOptions<T>
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

impl<T> MinimizerOptions<Array1<T>, T> for NelderMeadOptions<T>
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

pub struct NelderMead<T> {
    scale: Array1<T>,
    x: Array1<T>,
    x_scaled: Array1<T>,
    res: Option<Array1<T>>,
    simplex: Option<Array2<T>>,
    f: Box<dyn ObjFn<T>>,
    n: usize,
    iters: usize,
    max_iters: usize,
    tol: Option<T>,
    target_tol: T,
    alpha: T,
    beta: T,
    gamma: T,
    rho: T,
    verbosity: u32,
}

impl<T> NelderMead<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
    for<'a> &'a T: std::ops::Add<T, Output = T>,
    for<'a> &'a T: std::ops::Sub<T, Output = T>,
    for<'a> &'a T: std::ops::Mul<T, Output = T>,
    for<'a> &'a T: std::ops::Div<T, Output = T>,
    for<'a> &'a T: std::ops::Add<f64, Output = T>,
    for<'a> &'a T: std::ops::Sub<f64, Output = T>,
    for<'a> &'a T: std::ops::Mul<f64, Output = T>,
    for<'a> &'a T: std::ops::Div<f64, Output = T>,
    f64: std::ops::Add<T, Output = T>,
    f64: std::ops::Sub<T, Output = T>,
    f64: std::ops::Mul<T, Output = T>,
    f64: std::ops::Div<T, Output = T>,
    for<'a> f64: std::ops::Add<&'a T, Output = T>,
    for<'a> f64: std::ops::Sub<&'a T, Output = T>,
    for<'a> f64: std::ops::Mul<&'a T, Output = T>,
    for<'a> f64: std::ops::Div<&'a T, Output = T>,
{
    pub fn new<F>(x: Array1<T>, scale: Array1<T>, f: F) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        NelderMead {
            x_scaled: Array1::from_shape_fn(x.len(), |i| &x[i] * &scale[i]),
            scale,
            x: x.clone(),
            res: None,
            simplex: None,
            f: Box::new(f),
            n: x.len(),
            iters: 0,
            max_iters: 1, // Maximum iterations
            tol: None,
            target_tol: 1e-12.into(), // Convergent tolerance
            alpha: T::one(),          // Reflection coefficient
            beta: 0.5.into(),         // Contraction coefficient
            gamma: 2.0.into(),        // Expansion coefficient
            rho: 0.5.into(),          // Scaling coefficient
            verbosity: 0,
        }
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

    pub fn set_x(&mut self, x: ArrayView1<T>) {
        self.x = x.to_owned();
        self.x_scaled = Array1::from_shape_fn(self.x.len(), |i| &self.x[i] * &self.scale[i]);
        self.res = None;
        self.simplex = None;
        self.tol = None;
    }

    pub fn set_verbosity(&mut self, verbose: u32) {
        self.verbosity = verbose;
    }

    fn x(&self) -> &Array1<T> {
        &self.x
    }

    fn final_value(&self) -> Option<T> {
        self.res.as_ref().map(|x| x[0].clone())
    }

    fn tolerance(&self) -> Option<T> {
        self.tol.clone()
    }

    fn iterations(&self) -> usize {
        self.iters
    }

    fn name(&self) -> &str {
        "NelderMead"
    }

    fn minimize_opt(
        &mut self,
        opt: Box<NelderMeadOptions<T>>,
    ) -> Result<Box<NelderMeadResult<T>>, MinimizerError> {
        self.scale = opt.scale.clone();
        self.target_tol = opt.tolerance.clone();
        self.set_x(opt.initial_point.view());
        self.max_iters = opt.max_iterations;
        self.iters = 0;

        // Generate initial simplex veritces
        let c = T::one();
        let b = &c / ((self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0));
        let a = &b + &c / 2f64.sqrt();
        let _ncols = self.n;
        let nrows = self.n + 1;
        let mut simplex = Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
            if i == j && i < self.n {
                &self.x_scaled[j] + &a + &b
            } else if i < self.n {
                &self.x_scaled[j] + &b
            } else {
                self.x_scaled[j].clone()
            }
        });

        // Evaluate function at simplex vertices
        let mut res: Array1<T> = simplex
            .rows()
            .into_iter()
            .map(|x| {
                self.f
                    .call(Array1::from_shape_fn(self.n, |i| &x[i] / &self.scale[i]).view())
            })
            .collect();
        let mut prev_tol = T::one();

        while self.iters < self.max_iters {
            if self.verbosity > 1 {
                println!(
                    "iteration: {}\terr: {}\ttol: {}",
                    self.iters,
                    res[0],
                    self.tol.clone().unwrap_or(T::one())
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
            self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| {
                simplex[(i, j)].clone()
            }));

            // Determine if convergence criteria met
            let tol = 2.0 * (&res[res.len() - 1] - &res[0]).abs()
                / (res[res.len() - 1].abs() + res[0].abs() + 1e-15);
            match opt.tolerance.to_f64() {
                0.0 => (),
                _ => {
                    if tol.is_finite()
                        && (tol < self.target_tol
                            || ((&prev_tol - &tol).abs() < 1e-15.into() && prev_tol.is_finite()))
                    {
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
                let mut sum = T::from_f64(0.0);
                for j in 0..self.n {
                    sum += simplex[(j, i)].clone();
                }
                x_avg[i] = &sum / self.n as f64;
            }

            // Calculate reflection point
            let x_r =
                Array1::from_shape_fn(self.n, |i| &x_avg[i] + &self.alpha * (&x_avg[i] - &x_w[i]));
            let f_r = self
                .f
                .call(Array1::from_shape_fn(self.n, |i| &x_r[i] / &self.scale[i]).view());

            //
            // Determine simplex adjustment
            //
            if f_r <= res[0] {
                // Perform expansion
                if self.verbosity > 1 {
                    println!("performing expansion");
                }
                let x_e = Array1::from_shape_fn(self.n, |i| {
                    &x_r[i] + &self.gamma * (&x_r[i] - &x_avg[i])
                });
                let f_e = self
                    .f
                    .call(Array1::from_shape_fn(self.n, |i| &x_e[i] / &self.scale[i]).view());
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
                    &x_avg[i] - &self.beta * (&x_avg[i] - &x_w[i])
                });
                let f_ic = self
                    .f
                    .call(Array1::from_shape_fn(self.n, |i| &x_ic[i] / &self.scale[i]).view());
                if f_ic > res[nrows - 1] {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] =
                                &x_b[j] + &self.rho * (&simplex[(i + 1, j)] - &x_b[j]);
                        }
                        res[i + 1] = self.f.call(
                            Array1::from_shape_fn(self.n, |j| {
                                &simplex[(i + 1, j)] / &self.scale[j]
                            })
                            .view(),
                        );
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
                    &x_avg[i] + &self.beta * (&x_avg[i] - &x_w[i])
                });
                let f_oc = self
                    .f
                    .call(Array1::from_shape_fn(self.n, |i| &x_oc[i] / &self.scale[i]).view());
                let mut x_oc_vec: Vec<T> = vec![];
                for i in 0..self.n {
                    x_oc_vec.push(x_oc[i].clone());
                }
                if f_oc > f_r {
                    // Shrink simplex
                    if self.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.n {
                        for j in 0..self.n {
                            simplex[(i + 1, j)] =
                                &x_b[j] + &self.rho * (&simplex[(i + 1, j)] - &x_b[j]);
                        }
                        res[i + 1] = self.f.call(
                            Array1::from_shape_fn(self.n, |j| {
                                &simplex[(i + 1, j)] / &self.scale[j]
                            })
                            .view(),
                        );
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

        self.simplex = Some(simplex.clone());
        self.x_scaled = Array1::from_shape_fn(simplex.ncols(), |i| simplex[(0, i)].clone());
        self.x = Array1::from_shape_fn(self.n, |i| &self.x_scaled[i] / &self.scale[i]);
        // self.res = Some(res);
        self.res = Some(res.clone());

        if self.verbosity > 0 {
            println!("x: {:?}", self.x());
            println!("res: {:?}", self.res_all().unwrap());
            println!("iters: {}", self.iterations());
            println!("tol: {:?}", self.tolerance().unwrap());
        }

        Ok(Box::new(NelderMeadResult {
            xmin: self.x.clone(),
            fmin: self.f.call(self.x.view()),
            tolerance: match self.tol.clone() {
                Some(x) => x.clone(),
                _ => T::zero(),
            },
            iters: 0,
            fn_evals: 0,
            converged: true,
            final_simplex_size: T::zero(),
            history: array![],
        }))
    }
}

impl<T> Minimizer<Array1<T>, T> for NelderMead<T>
where
    T: RFFloat + 'static,
    for<'a, 'b> &'a T: std::ops::Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
    for<'a> &'a T: std::ops::Add<T, Output = T>,
    for<'a> &'a T: std::ops::Sub<T, Output = T>,
    for<'a> &'a T: std::ops::Mul<T, Output = T>,
    for<'a> &'a T: std::ops::Div<T, Output = T>,
    for<'a> &'a T: std::ops::Add<f64, Output = T>,
    for<'a> &'a T: std::ops::Sub<f64, Output = T>,
    for<'a> &'a T: std::ops::Mul<f64, Output = T>,
    for<'a> &'a T: std::ops::Div<f64, Output = T>,
    f64: std::ops::Add<T, Output = T>,
    f64: std::ops::Sub<T, Output = T>,
    f64: std::ops::Mul<T, Output = T>,
    f64: std::ops::Div<T, Output = T>,
    for<'a> f64: std::ops::Add<&'a T, Output = T>,
    for<'a> f64: std::ops::Sub<&'a T, Output = T>,
    for<'a> f64: std::ops::Mul<&'a T, Output = T>,
    for<'a> f64: std::ops::Div<&'a T, Output = T>,
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
        Ok(result as Box<dyn MinimizerResult<Array1<T>, T>>)
    }
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
    //     let c = T::from_f64(1.0);
    //     let b = &c / (self.n as f64 * 2f64.sqrt()) * ((self.n as f64 + 1.0).sqrt() - 1.0);
    //     let a = &b + &c / 2f64.sqrt();
    //     let _ncols = self.n;
    //     let nrows = self.n + 1;
    //     let mut simplex = Array2::from_shape_fn((self.n + 1, self.n), |(i, j)| {
    //         if i == j && i < self.n {
    //             &self.x_scaled[j] + &a + &b
    //         } else if i < self.n {
    //             &self.x_scaled[j] + &b
    //         } else {
    //             self.x_scaled[j].clone()
    //         }
    //     });

    //     // Evaluate function at simplex vertices
    //     let mut res: Array1<T> = simplex
    //         .rows()
    //         .into_iter()
    //         .map(|x| {
    //             self.f
    //                 .call(&Array1::from_shape_fn(self.n, |i| &x[i] / &self.scale[i]))
    //         })
    //         .collect();
    //     let mut prev_tol = T::from_f64(1.0);

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
    //         self.simplex = Some(Array2::from_shape_fn(shape, |(i, j)| {
    //             simplex[(i, j)].clone()
    //         }));

    //         // Determine if convergence criteria met
    //         let tol = 2.0 * (&res[res.len() - 1] - &res[0]).abs()
    //             / (res[res.len() - 1].abs() + res[0].abs() + 1e-15);
    //         if tol.is_finite()
    //             && (tol < self.target_tol
    //                 || ((&prev_tol - &tol).abs() < 1e-15 && prev_tol.is_finite()))
    //         {
    //             break;
    //         }
    //         prev_tol = tol.clone();

    //         let x_b = Array1::from_shape_fn(self.n, |i| simplex[(0, i)].clone()); // Best point
    //         let _x_l = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 2, i)].clone()); // Lousy point
    //         let x_w = Array1::from_shape_fn(self.n, |i| simplex[(nrows - 1, i)].clone()); // Worst point

    //         // Calculate the average of the n best points
    //         let mut x_avg = Array1::zeros(self.n);
    //         for i in 0..self.n {
    //             let mut sum = T::from_f64(0.0);
    //             for j in 0..self.n {
    //                 sum += &simplex[(j, i)];
    //             }
    //             x_avg[i] = 1.0 / self.n as f64 * &sum;
    //         }

    //         // Calculate reflection point
    //         let x_r =
    //             Array1::from_shape_fn(self.n, |i| &x_avg[i] + &self.alpha * (&x_avg[i] - &x_w[i]));
    //         let f_r = self
    //             .f
    //             .call(&Array1::from_shape_fn(self.n, |i| &x_r[i] / &self.scale[i]));

    //         //
    //         // Determine simplex adjustment
    //         //
    //         if f_r <= res[0] {
    //             // Perform expansion
    //             if self.verbosity > 1 {
    //                 println!("performing expansion");
    //             }
    //             let x_e = Array1::from_shape_fn(self.n, |i| {
    //                 &x_r[i] + &self.gamma * (&x_r[i] - &x_avg[i])
    //             });
    //             let f_e = self
    //                 .f
    //                 .call(&Array1::from_shape_fn(self.n, |i| &x_e[i] / &self.scale[i]));
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
    //                 &x_avg[i] - &self.beta * (&x_avg[i] - &x_w[i])
    //             });
    //             let f_ic = self.f.call(&Array1::from_shape_fn(self.n, |i| {
    //                 &x_ic[i] / &self.scale[i]
    //             }));
    //             if f_ic > res[nrows - 1] {
    //                 // Shrink simplex
    //                 if self.verbosity > 1 {
    //                     println!("shrinking simplex");
    //                 }
    //                 for i in 0..self.n {
    //                     for j in 0..self.n {
    //                         simplex[(i + 1, j)] =
    //                             &x_b[j] + &self.rho * (&simplex[(i + 1, j)] - &x_b[j]);
    //                     }
    //                     res[i + 1] = self.f.call(&Array1::from_shape_fn(self.n, |j| {
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
    //                 &x_avg[i] + &self.beta * (&x_avg[i] - &x_w[i])
    //             });
    //             let f_oc = self.f.call(&Array1::from_shape_fn(self.n, |i| {
    //                 &x_oc[i] / &self.scale[i]
    //             }));
    //             let mut x_oc_vec: Vec<T> = vec![];
    //             for i in 0..self.n {
    //                 x_oc_vec.push(x_oc[i].clone());
    //             }
    //             if f_oc > f_r {
    //                 // Shrink simplex
    //                 if self.verbosity > 1 {
    //                     println!("shrinking simplex");
    //                 }
    //                 for i in 0..self.n {
    //                     for j in 0..self.n {
    //                         simplex[(i + 1, j)] =
    //                             &x_b[j] + &self.rho * (&simplex[(i + 1, j)] - &x_b[j]);
    //                     }
    //                     res[i + 1] = self.f.call(&Array1::from_shape_fn(self.n, |j| {
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

    //     self.simplex = Some(simplex.clone());
    //     self.x_scaled = Array1::from_shape_fn(simplex.ncols(), |i| simplex[(0, i)].clone());
    //     self.x = Array1::from_shape_fn(self.n, |i| &self.x_scaled[i] / &self.scale[i]);
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
    //         tolerance: match self.tol.clone() {
    //             Some(x) => x.clone(),
    //             _ => T::from_f64(0.0),
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
mod minimize_neldermead_tests {
    use super::*;
    use crate::{
        file::read_touchstone,
        frequency::Frequency,
        minimize::MultiDimFn,
        mycomplex::MyComplex,
        myfloat::MyFloat,
        network::{Network, NetworkBuilder},
        pts::{Points, Pts},
        util::*,
    };
    use float_cmp::F64Margin;
    use num::complex::Complex64;

    fn calc_feed_y(freq: Frequency, x: ArrayView1<MyFloat>) -> Array3<MyComplex> {
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

    fn eval_f_simplex(x: ArrayView1<MyFloat>, meas: &Network) -> MyFloat {
        let model_y = calc_feed_y(meas.freq().clone(), x);
        let model_y_c64 = Points::<Complex64, Ix3>::from_shape_fn(model_y.dim(), |(i, j, k)| {
            model_y[[i, j, k]].clone().into()
        });
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
        let exemplar_res = MyFloat::new(5736.50737797145);
        let exemplar_tol = MyFloat::new(1.9902303400016697);
        let exemplar_x = array![
            MyFloat::new(1.0e-11),
            1.0e-03.into(),
            1.0e-13.into(),
            1.0e-06.into(),
            1.0e-15.into(),
            1.0e03.into(),
            1.0e-15.into(),
            1.0e-15.into()
        ];
        let exemplar_simplex = array![
            [
                MyFloat::new(1.0000000000000000e+01),
                1.0000000000000000e-03.into(),
                1.0000000000000001e-01.into(),
                9.9999999999999995e-07.into(),
                1.0000000000000000e+00.into(),
                1.0000000000000000e+03.into(),
                1.0000000000000000e+00.into(),
                1.0000000000000000e+00.into()
            ],
            [
                10.795495128834867.into(),
                4.5194173824159217e-2.into(),
                1.4419417382415922e-1.into(),
                4.4195173824159217e-2.into(),
                1.0441941738241591.into(),
                1000.0441941738242.into(),
                1.0441941738241591.into(),
                1.0441941738241591.into()
            ],
            [
                10.04419417382416.into(),
                0.04519417382415922.into(),
                0.14419417382415922.into(),
                0.795496128834866.into(),
                1.0441941738241591.into(),
                1000.0441941738242.into(),
                1.0441941738241591.into(),
                1.0441941738241591.into()
            ],
            [
                10.04419417382416.into(),
                0.04519417382415922.into(),
                0.14419417382415922.into(),
                0.04419517382415922.into(),
                1.0441941738241591.into(),
                1000.0441941738242.into(),
                1.0441941738241591.into(),
                1.7954951288348657.into()
            ],
            [
                10.04419417382416.into(),
                0.04519417382415922.into(),
                0.14419417382415922.into(),
                0.04419517382415922.into(),
                1.0441941738241591.into(),
                1000.0441941738242.into(),
                1.7954951288348657.into(),
                1.0441941738241591.into()
            ],
            [
                10.04419417382416.into(),
                0.04519417382415922.into(),
                0.14419417382415922.into(),
                0.04419517382415922.into(),
                1.0441941738241591.into(),
                1000.7954951288349.into(),
                1.0441941738241591.into(),
                1.0441941738241591.into()
            ],
            [
                10.04419417382416.into(),
                0.04519417382415922.into(),
                0.14419417382415922.into(),
                0.04419517382415922.into(),
                1.7954951288348657.into(),
                1000.0441941738242.into(),
                1.0441941738241591.into(),
                1.0441941738241591.into()
            ],
            [
                10.04419417382416.into(),
                0.04519417382415922.into(),
                0.8954951288348659.into(),
                0.04419517382415922.into(),
                1.0441941738241591.into(),
                1000.0441941738242.into(),
                1.0441941738241591.into(),
                1.0441941738241591.into()
            ],
            [
                10.176776695296638.into(),
                MyFloat::new(-0.33874271127322403),
                0.27677669529663684.into(),
                0.1767776952966369.into(),
                1.176776695296637.into(),
                1000.1767766952967.into(),
                1.176776695296637.into(),
                1.1767766952966372.into()
            ],
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let objective =
            MultiDimFn::new(move |x: ArrayView1<MyFloat>| -> MyFloat { eval_f_simplex(x, &net) });
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

        let margin = MARGIN;
        comp_mat_myfloat(
            exemplar_simplex.view(),
            test.simplex().unwrap().view(),
            margin,
            "minimize(simplex)",
        );
        comp_row_myfloat(exemplar_x.view(), test.x().view(), margin, "minimize(x)");
        comp_myfloat(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter2_() {
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
        let exemplar_res = 5736.50737797145.into();
        let exemplar_tol = 1.9726210586239914.into();
        let exemplar_x = array![
            1.0e-11.into(),
            1.0e-03.into(),
            1.0e-13.into(),
            1.0e-06.into(),
            1.0e-15.into(),
            1.0e03.into(),
            1.0e-15.into(),
            1.0e-15.into()
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
        let mut test = NelderMead::new(
            x.clone(),
            scale,
            MultiDimFn::new(move |x: ArrayView1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(exemplar_x.view(), test.x().view(), margin, "minimize(x)");
        comp_myfloat(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter5_() {
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
        let exemplar_res = 5736.50737797145.into();
        let exemplar_tol = 1.6060281828650276.into();
        let exemplar_x = array![
            1.0e-11.into(),
            1.0e-03.into(),
            1.0e-13.into(),
            1.0e-06.into(),
            1.0e-15.into(),
            1.0e03.into(),
            1.0e-15.into(),
            1.0e-15.into()
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
        let mut test = NelderMead::new(
            x.clone(),
            scale,
            MultiDimFn::new(move |x: ArrayView1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(exemplar_x.view(), test.x().view(), margin, "minimize(x)");
        comp_myfloat(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter6_() {
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
        let exemplar_res = 5736.507376515564.into();
        let exemplar_tol = 1.5066440937567893.into();
        let exemplar_x = array![
            1.0496148654572788e-11.into(),
            0.001.into(),
            MyFloat::new(-2.5331914829451519e-14),
            0.000001.into(),
            9.9896419905099658e-16.into(),
            1000.into(),
            1.4961486545727874e-15.into(),
            1.4961486545727874e-15.into()
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
        let mut test = NelderMead::new(
            x.clone(),
            scale,
            MultiDimFn::new(move |x: ArrayView1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(exemplar_x.view(), test.x().view(), margin, "minimize(x)");
        comp_myfloat(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter10_() {
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
        let exemplar_res = 3267.394011570648.into();
        let exemplar_tol = 1.6651438670237593.into();
        let exemplar_x = array![
            1.0496148654572788e-11.into(),
            0.0028955966586259665.into(),
            MyFloat::new(-2.5331914829451519e-14),
            0.36146316242264603.into(),
            9.9896419905099658e-16.into(),
            999.4223359686597.into(),
            1.4961486545727874e-15.into(),
            1.4961486545727874e-15.into()
        ];

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
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
        let mut test = NelderMead::new(
            x.clone(),
            scale,
            MultiDimFn::new(move |x: ArrayView1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(exemplar_x.view(), test.x().view(), margin, "minimize(x)");
        comp_myfloat(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    fn nelder_mead_iter100_() {
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
        let exemplar_res = 765.6358932973524.into();
        let exemplar_tol = 0.12081175542092283.into();
        let exemplar_x = array![
            1.6110247811984210e-11.into(),
            0.010202461287747289.into(),
            1.2618495970192951e-12.into(),
            6.410718597400084.into(),
            MyFloat::new(-2.1119233982250064e-16),
            991.264467681222.into(),
            4.5268760596139400e-15.into(),
            9.1487688225672705e-15.into()
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
        let mut test = NelderMead::new(
            x.clone(),
            scale,
            MultiDimFn::new(move |x: ArrayView1<MyFloat>| eval_f_simplex(x, &net)),
        );
        _ = test.minimize(options);

        let margin = MARGIN;
        comp_row_myfloat(exemplar_x.view(), test.x().view(), margin, "minimize(x)");
        comp_myfloat(
            &exemplar_res,
            &test.final_value().unwrap(),
            margin,
            "minimize(res)",
            "",
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
    // fn nelder_mead_iter1000_() {
    //     let x: Array1<MyFloat> = array![1e-11, 1e-3, 1e-13, 1e-6, 1e-15, 1000.0, 1e-15, 1e-15];
    //     let scale: Array1<MyFloat> = array![1e12, 1.0, 1e12, 1.0, 1e15, 1.0, 1e15, 1e15];
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
