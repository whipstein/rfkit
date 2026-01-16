#![allow(dead_code)]
use crate::{
    error::MinimizerError,
    minimize::{Minimizer, ObjFn},
    num::RFFloat,
    pts::{Matrix, Points, Points1, Points2, Pts},
};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Uniform};

/// Result of Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadResult<T>
where
    T: RFFloat,
{
    pub xmin: Points1<T>,
    pub fmin: T,
    pub tolerance: T,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_simplex_size: T,
    pub history: Points1<T>,
}

impl<T> NelderMeadResult<T>
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

    fn xmin(&self) -> Points1<T> {
        self.xmin.clone()
    }

    fn history(&self) -> Points1<T> {
        self.history.clone()
    }
}

#[derive(Debug, Clone, Default)]
pub struct NelderMeadOptions<T>
where
    T: RFFloat,
{
    initial_point: Points1<T>,
    scale: Points1<T>,
    n: usize,
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
    max_restarts: usize,         // Maximum number of random restarts
    perturbation_scale: T,       // Scale for random perturbation (fraction of bounds range)
    rng_seed: Option<u64>,       // Optional seed for reproducibility
    verbosity: usize,
}

impl<T> NelderMeadOptions<T>
where
    T: RFFloat,
{
    pub fn new(
        init: &Points1<T>,
        scale: Option<&Points1<T>>,
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
            initial_point: init.to_owned(),
            scale: scale.unwrap_or(&Points1::ones(n)).to_owned(),
            n: init.len(),
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
            max_restarts: 3,
            perturbation_scale: T::from_f64(0.1),
            rng_seed: None,
            verbosity: verbosity.unwrap_or(0),
        }
    }

    fn update(&mut self, opt: &NelderMeadOptions<T>) {
        self.initial_point = opt.initial_point.clone();
        self.scale = opt.scale.clone();
        self.n = opt.n;
        self.max_iterations = opt.max_iterations;
        self.tolerance = opt.tolerance.clone();
        self.alpha = opt.alpha.clone();
        self.beta = opt.beta.clone();
        self.gamma = opt.gamma.clone();
        self.rho = opt.rho.clone();
        self.use_random_restart = opt.use_random_restart;
        self.use_adaptive_simplex = opt.use_adaptive_simplex;
        self.stagnation_iters = opt.stagnation_iters;
        self.stagnation_threshold = opt.stagnation_threshold;
        self.max_restarts = opt.max_restarts;
        self.perturbation_scale = opt.perturbation_scale.clone();
        self.rng_seed = opt.rng_seed;
        self.verbosity = opt.verbosity;
    }

    pub fn set_initial_point(&mut self, init: Points1<T>) {
        self.initial_point = init;
    }

    pub fn set_scale(&mut self, scale: Points1<T>) {
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

    /// Enable anti-stagnation features to help escape local minima.
    /// This enables random restarts and adaptive simplex perturbation.
    ///
    /// # Arguments
    /// * `max_restarts` - Maximum number of random restarts (default: 3)
    /// * `stagnation_threshold` - Iterations without improvement before triggering action (default: 50)
    /// * `perturbation_scale` - Scale factor for random perturbation as fraction of bounds range (default: 0.1)
    pub fn enable_anti_stagnation(
        &mut self,
        max_restarts: Option<usize>,
        stagnation_threshold: Option<usize>,
        perturbation_scale: Option<T>,
    ) {
        self.use_random_restart = true;
        self.use_adaptive_simplex = true;
        if let Some(mr) = max_restarts {
            self.max_restarts = mr;
        }
        if let Some(st) = stagnation_threshold {
            self.stagnation_threshold = st;
        }
        if let Some(ps) = perturbation_scale {
            self.perturbation_scale = ps;
        }
    }

    fn initial_point(&self) -> &Points1<T> {
        &self.initial_point
    }

    fn scale(&self) -> &Points1<T> {
        &self.scale
    }

    fn n(&self) -> usize {
        self.n
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

pub struct NelderMead<T>
where
    T: RFFloat,
{
    res: Option<Points1<T>>,
    simplex: Option<Points2<T>>,
    f: Box<dyn ObjFn<T>>,
    iters: usize,
    tol: Option<T>,
    options: NelderMeadOptions<T>,
}

impl<T> NelderMead<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    // for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
    for<'a> &'a T: std::ops::Add<T, Output = T>,
    for<'a> &'a T: std::ops::Sub<T, Output = T>,
    for<'a> &'a T: std::ops::Mul<T, Output = T>,
    // for<'a> &'a T: std::ops::Div<T, Output = T>,
    // for<'a> &'a T: std::ops::Add<f64, Output = T>,
    // for<'a> &'a T: std::ops::Sub<f64, Output = T>,
    // for<'a> &'a T: std::ops::Mul<f64, Output = T>,
    for<'a> &'a T: std::ops::Div<f64, Output = T>,
    f64: std::ops::Add<T, Output = T>,
    f64: std::ops::Sub<T, Output = T>,
    f64: std::ops::Mul<T, Output = T>,
    // f64: std::ops::Div<T, Output = T>,
    for<'a> f64: std::ops::Add<&'a T, Output = T>,
    for<'a> f64: std::ops::Sub<&'a T, Output = T>,
    for<'a> f64: std::ops::Mul<&'a T, Output = T>,
    // for<'a> f64: std::ops::Div<&'a T, Output = T>,
    for<'a> Points1<f64>: std::ops::Mul<&'a Points1<T>, Output = Points1<T>>,
{
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        NelderMead {
            res: None,
            simplex: None,
            f: Box::new(f),
            iters: 0,
            tol: None,
            options: NelderMeadOptions::<T>::default(),
        }
    }

    pub fn calc_obj(&mut self, x: &Points1<T>) -> T {
        self.f.call(x)
    }

    fn res_all(&self) -> Option<&Points1<T>> {
        match &self.res {
            Some(x) => Some(&x),
            None => None,
        }
    }

    fn simplex(&self) -> Option<&Points2<T>> {
        match &self.simplex {
            Some(x) => Some(&x),
            None => None,
        }
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

    /// Generate a perturbed restart point from the best known point
    fn generate_restart_point(&self, best: &Points1<T>) -> Points1<T>
    where
        for<'a> Points1<f64>: std::ops::Mul<&'a Points1<T>, Output = Points1<T>>,
    {
        Points(Array1::random(
            self.options.n,
            Uniform::new(-1_f64, 1_f64).unwrap(),
        )) * &self.options.scale
            + best
    }

    /// Perturb simplex vertices (except the best one) to escape local minima
    fn perturb_simplex(
        &mut self,
        simplex: &mut Points2<T>,
        res: &mut Points1<T>,
        rng: &mut StdRng,
    ) {
        let uniform = Uniform::new(-1.0_f64, 1.0_f64).unwrap();
        let nrows = self.options.n + 1;

        if self.options.verbosity > 5 {
            println!("\nPerturbed simplex\noriginal: {:?}", simplex.row(1));
        }

        // Perturb all vertices except the best one (index 0)
        for i in 1..nrows {
            for j in 0..self.options.n {
                let perturbation = T::from_f64(uniform.sample(rng))
                    * &self.options.perturbation_scale
                    * T::from_f64(0.5);

                simplex[(i, j)] = &simplex[(i, j)] + perturbation;
            }

            // Re-evaluate the perturbed vertex
            res[i] = self.calc_obj(&Points1::from_shape_fn(self.options.n, |j| {
                &simplex[(i, j)] / &self.options.scale[j]
            }));
        }

        if self.options.verbosity > 5 {
            println!("new: {:?}\n", simplex.row(1));
        }
    }

    fn minimize_w_restart(
        &mut self,
        opt: &NelderMeadOptions<T>,
    ) -> Result<NelderMeadResult<T>, MinimizerError> {
        self.options.update(opt);
        self.iters = 0;

        // Initialize RNG for perturbation and restarts
        let mut rng = match self.options.rng_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Track global best across restarts
        let mut best_global_point: Option<Points1<T>> = None;
        let mut best_global_fmin = T::infinity();
        let mut restart_count = 0;
        let mut total_fn_evals = 0;

        // Outer loop for restarts
        'restart_loop: loop {
            // Determine starting point for this run
            let x_scaled = if restart_count == 0 {
                &self.options.initial_point * &self.options.scale
            } else {
                if self.options.verbosity > 1 {
                    println!("Perturbing x point");
                }
                // Perturb best known point for restart
                self.generate_restart_point(&best_global_point.as_ref().unwrap())
            };

            if self.options.verbosity > 0 && restart_count > 0 {
                println!("Restart {} from perturbed best point", restart_count);
            }

            // Generate initial simplex vertices with adaptive sizing
            let simplex_scale = if self.options.use_adaptive_simplex {
                // Use larger simplex initially, based on bounds range
                let range_scale = T::from_f64(0.2); // 20% of bounds range
                range_scale
            } else {
                T::one()
            };

            let c = &simplex_scale * T::one();
            let b = &c
                / ((self.options.n as f64 * 2f64.sqrt())
                    * ((self.options.n as f64 + 1.0).sqrt() - 1.0));
            let a = &c / 2f64.sqrt();
            let nrows = self.options.n + 1;

            let x_scaled_a_b = x_scaled.map(|x| &a + &b + x);
            let x_scaled_b = x_scaled.map(|x| &b + x);
            let mut simplex =
                Points2::from_shape_fn((self.options.n + 1, self.options.n), |(i, j)| {
                    if i == j && i < self.options.n {
                        x_scaled_a_b[j].clone()
                    } else if i < self.options.n {
                        x_scaled_b[j].clone()
                    } else {
                        x_scaled[j].clone()
                    }
                });

            // Evaluate function at simplex vertices
            let mut res: Points1<T> = simplex
                .0
                .rows()
                .into_iter()
                .map(|x| {
                    total_fn_evals += 1;
                    self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
                        &x[i] / &self.options.scale[i]
                    }))
                })
                .collect();

            let mut stagnation_count: usize = 0;
            let mut prev_best_fmin = res[0].clone();
            let improvement_threshold = T::from_f64(1e-10);

            // Inner optimization loop
            while self.iters < self.options.max_iterations {
                if self.options.verbosity > 1 {
                    println!(
                        "iteration: {}\tx: {}\terr: {}\ttol: {}\tstagnation: {}",
                        self.iters,
                        simplex.row(0),
                        res[0],
                        self.tol.clone().unwrap_or(T::one()),
                        stagnation_count
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
                    for j in 0..self.options.n {
                        simplex[(i, j)] = tmp_simplex[(order[i], j)].clone();
                    }
                }
                let shape = simplex.shape();
                self.simplex = Some(Points2::from_shape_fn(shape, |(i, j)| {
                    simplex[(i, j)].clone()
                }));

                // Check for improvement and update stagnation counter
                let current_fmin = res[0].clone();
                if (&prev_best_fmin - &current_fmin) > improvement_threshold {
                    stagnation_count = 0;
                    prev_best_fmin = current_fmin.clone();

                    // Update global best if this is better
                    if current_fmin < best_global_fmin {
                        best_global_fmin = current_fmin.clone();
                        best_global_point = Some(Points1::from_shape_fn(self.options.n, |i| {
                            &simplex[(0, i)] / &self.options.scale[i]
                        }));
                    }
                } else {
                    stagnation_count += 1;
                }

                // Check for stagnation and apply perturbation or restart
                if self.options.use_random_restart
                    && stagnation_count >= self.options.stagnation_threshold
                {
                    if self.options.verbosity > 0 {
                        println!(
                            "Stagnation detected after {} iterations without improvement",
                            stagnation_count
                        );
                    }

                    // Try simplex perturbation first
                    if self.options.use_adaptive_simplex {
                        self.perturb_simplex(&mut simplex, &mut res, &mut rng);
                        total_fn_evals += self.options.n; // Re-evaluated n vertices
                        stagnation_count = 0;

                        if self.options.verbosity > 0 {
                            println!("Applied simplex perturbation");
                        }
                    } else if restart_count < self.options.max_restarts {
                        // Trigger restart
                        restart_count += 1;
                        if self.options.verbosity > 0 {
                            println!("Restart {}", restart_count);
                        }
                        continue 'restart_loop;
                    }
                }

                // Determine if convergence criteria met
                let tol = T::from_f64(2.0) * (&res[res.len() - 1] - &res[0]).abs()
                    / (res[res.len() - 1].abs() + res[0].abs() + 1e-10);

                match opt.tolerance.to_f64() {
                    0.0 => (),
                    _ => {
                        if tol < opt.tolerance {
                            break;
                        }
                    }
                }

                let x_b = Points1::from_shape_fn(self.options.n, |i| simplex[(0, i)].clone());
                let x_w =
                    Points1::from_shape_fn(self.options.n, |i| simplex[(nrows - 1, i)].clone());

                // Calculate the average of the n best points
                let mut x_avg = Points1::<T>::zeros(self.options.n);
                for i in 0..self.options.n {
                    let mut sum = T::from_f64(0.0);
                    for j in 0..self.options.n {
                        sum += &simplex[(j, i)];
                    }
                    x_avg[i] = sum / self.options.n as f64;
                }

                // Calculate reflection point
                let mut x_r = Points1::zeros(self.options.n);
                azip!((index i, x_avg in x_avg.inner(), x_w in x_w.inner()) {x_r[i] = x_avg + &self.options.alpha * (x_avg - x_w);});
                // let x_r = &x_avg + &self.options.alpha * (&x_avg - &x_w);
                total_fn_evals += 1;
                let f_r = self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
                    &x_r[i] / &self.options.scale[i]
                }));

                //
                // Determine simplex adjustment
                //
                if f_r <= res[0] {
                    // Perform expansion
                    if self.options.verbosity > 1 {
                        println!("performing expansion");
                    }
                    let mut x_e = Points1::zeros(self.options.n);
                    azip!((index i, x_avg in x_avg.inner(), x_r in x_r.inner()) {x_e[i] = x_r + &self.options.gamma * (x_r - x_avg);});
                    // let x_e = &x_r + &self.options.gamma * (&x_r - &x_avg);
                    total_fn_evals += 1;
                    let f_e = self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
                        &x_e[i] / &self.options.scale[i]
                    }));
                    if f_e < res[0] {
                        for i in 0..self.options.n {
                            simplex[(nrows - 1, i)] = x_e[i].clone();
                        }
                        res[nrows - 1] = f_e.clone();
                    } else {
                        for i in 0..self.options.n {
                            simplex[(nrows - 1, i)] = x_r[i].clone();
                        }
                        res[nrows - 1] = f_r.clone();
                    }
                } else if f_r > res[nrows - 1] {
                    // Perform inside contraction
                    if self.options.verbosity > 1 {
                        println!("performing inside contraction");
                    }
                    let mut x_ic = Points1::zeros(self.options.n);
                    azip!((index i, x_avg in x_avg.inner(), x_w in x_w.inner()) {x_ic[i] = x_avg - &self.options.beta * (x_avg - x_w);});
                    // let x_ic = &x_avg - &self.options.beta * (&x_avg - &x_w);
                    total_fn_evals += 1;
                    let f_ic = self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
                        &x_ic[i] / &self.options.scale[i]
                    }));
                    if f_ic > res[nrows - 1] {
                        // Shrink simplex
                        if self.options.verbosity > 1 {
                            println!("shrinking simplex");
                        }
                        for i in 0..self.options.n {
                            for j in 0..self.options.n {
                                simplex[(i + 1, j)] =
                                    &x_b[j] + &self.options.rho * (&simplex[(i + 1, j)] - &x_b[j]);
                            }
                            total_fn_evals += 1;
                            res[i + 1] = self
                                .calc_obj(&Points1::from_shape_fn(self.options.n, |j| {
                                    &simplex[(i + 1, j)] / &self.options.scale[j]
                                }));
                        }
                    } else {
                        for i in 0..self.options.n {
                            simplex[(nrows - 1, i)] = x_ic[i].clone();
                        }
                        res[nrows - 1] = f_ic.clone();
                    }
                } else if f_r > res[nrows - 2] {
                    // Perform outside contraction
                    if self.options.verbosity > 1 {
                        println!("performing outside contraction");
                    }
                    let mut x_oc = Points1::zeros(self.options.n);
                    azip!((index i, x_avg in x_avg.inner(), x_w in x_w.inner()) {x_oc[i] = x_avg + &self.options.beta * (x_avg - x_w);});
                    // let x_oc = &x_avg + &self.options.beta * (&x_avg - &x_w);
                    total_fn_evals += 1;
                    let f_oc = self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
                        &x_oc[i] / &self.options.scale[i]
                    }));
                    if f_oc > f_r {
                        // Shrink simplex
                        if self.options.verbosity > 1 {
                            println!("shrinking simplex");
                        }
                        for i in 0..self.options.n {
                            for j in 0..self.options.n {
                                simplex[(i + 1, j)] =
                                    &x_b[j] + &self.options.rho * (&simplex[(i + 1, j)] - &x_b[j]);
                            }
                            total_fn_evals += 1;
                            res[i + 1] = self
                                .calc_obj(&Points1::from_shape_fn(self.options.n, |j| {
                                    &simplex[(i + 1, j)] / &self.options.scale[j]
                                }));
                        }
                    } else {
                        for i in 0..self.options.n {
                            simplex[(nrows - 1, i)] = x_oc[i].clone();
                        }
                        res[nrows - 1] = f_oc.clone();
                    }
                } else {
                    for i in 0..self.options.n {
                        simplex[(nrows - 1, i)] = x_r[i].clone();
                    }
                    res[nrows - 1] = f_r.clone();
                }

                self.tol = Some(tol.clone());
            }

            // Check if we should try another restart
            if restart_count < self.options.max_restarts
                && self.options.use_random_restart
                && self.iters < self.options.max_iterations
            {
                restart_count += 1;
                if self.options.verbosity > 0 {
                    println!("Restart {}", restart_count);
                }
                continue 'restart_loop;
            }

            // Final update of global best from this run
            if res[0] < best_global_fmin {
                best_global_fmin = res[0].clone();
                best_global_point = Some(Points1::from_shape_fn(self.options.n, |i| {
                    &simplex[(0, i)] / &self.options.scale[i]
                }));
            }

            let shape = simplex.shape();
            self.simplex = Some(Points2::from_shape_fn(shape, |(i, j)| {
                simplex[(i, j)].clone()
            }));

            self.res = Some(Points1::from_shape_fn(res.len(), |i| res[i].clone()));
            break;
        }

        if self.options.verbosity > 0 {
            println!("Total restarts: {}", restart_count);
            println!("Total function evaluations: {}", total_fn_evals);
            println!("Best xmin: {:?}", best_global_point);
            println!("Best fmin: {:?}", best_global_fmin);
            println!("iters: {}", self.iterations());
            println!("tol: {:?}", self.tolerance().unwrap_or(T::zero()));
        }

        let final_x = best_global_point.unwrap_or_else(|| self.options.initial_point.clone());
        Ok(NelderMeadResult {
            xmin: final_x.clone(),
            fmin: best_global_fmin,
            tolerance: match self.tol.clone() {
                Some(x) => x,
                _ => T::zero(),
            },
            iters: self.iters,
            fn_evals: total_fn_evals,
            converged: true,
            final_simplex_size: T::zero(),
            history: array![].into(),
        })
    }
}

impl<T> Minimizer<T> for NelderMead<T>
where
    T: RFFloat + 'static,
    for<'a, 'b> &'a T: std::ops::Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    // for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
    for<'a> &'a T: std::ops::Add<T, Output = T>,
    for<'a> &'a T: std::ops::Sub<T, Output = T>,
    for<'a> &'a T: std::ops::Mul<T, Output = T>,
    // for<'a> &'a T: std::ops::Div<T, Output = T>,
    // for<'a> &'a T: std::ops::Add<f64, Output = T>,
    // for<'a> &'a T: std::ops::Sub<f64, Output = T>,
    // for<'a> &'a T: std::ops::Mul<f64, Output = T>,
    for<'a> &'a T: std::ops::Div<f64, Output = T>,
    f64: std::ops::Add<T, Output = T>,
    f64: std::ops::Sub<T, Output = T>,
    f64: std::ops::Mul<T, Output = T>,
    // f64: std::ops::Div<T, Output = T>,
    for<'a> f64: std::ops::Add<&'a T, Output = T>,
    for<'a> f64: std::ops::Sub<&'a T, Output = T>,
    for<'a> f64: std::ops::Mul<&'a T, Output = T>,
    // for<'a> f64: std::ops::Div<&'a T, Output = T>,
    for<'a> Points1<f64>: std::ops::Mul<&'a Points1<T>, Output = Points1<T>>,
    // for<'a> &'a T: std::ops::Mul<Points1<T>, Output = Points1<T>>,
{
    type Options = NelderMeadOptions<T>;
    type Result = NelderMeadResult<T>;

    fn minimize(&mut self, opt: &Self::Options) -> Result<Self::Result, MinimizerError> {
        self.options.update(opt);
        self.iters = 0;
        let mut x_scaled = &self.options.initial_point * &self.options.scale;

        // Generate initial simplex veritces
        let c = T::one();
        let b = &c
            / ((self.options.n as f64 * 2f64.sqrt())
                * ((self.options.n as f64 + 1.0).sqrt() - 1.0));
        let a = &b + &c / 2f64.sqrt();
        let _ncols = self.options.n;
        let nrows = self.options.n + 1;
        let mut simplex = Points2::from_shape_fn((self.options.n + 1, self.options.n), |(i, j)| {
            if i == j && i < self.options.n {
                &x_scaled[j] + &a + &b
            } else if i < self.options.n {
                &x_scaled[j] + &b
            } else {
                x_scaled[j].clone()
            }
        });

        // Evaluate function at simplex vertices
        let mut res: Points1<T> = simplex
            .0
            .rows()
            .into_iter()
            .map(|x| {
                self.f.call(&Points1::from_shape_fn(self.options.n, |i| {
                    &x[i] / &self.options.scale[i]
                }))
            })
            .collect();
        let mut prev_tol = T::one();

        while self.iters < self.options.max_iterations {
            if self.options.verbosity > 1 {
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
                for j in 0..self.options.n {
                    simplex[(i, j)] = tmp_simplex[(order[i], j)].clone();
                }
            }
            let shape = simplex.shape();
            self.simplex = Some(Points2::from_shape_fn(shape, |(i, j)| {
                simplex[(i, j)].clone()
            }));

            // Determine if convergence criteria met
            let tol = 2.0 * (&res[res.len() - 1] - &res[0]).abs()
                / (res[res.len() - 1].abs() + res[0].abs() + 1e-15);
            match opt.tolerance.to_f64() {
                0.0 => (),
                _ => {
                    if tol.is_finite()
                        && (tol < self.options.tolerance
                            || ((&prev_tol - &tol).abs() < 1e-15.into() && prev_tol.is_finite()))
                    {
                        break;
                    }
                }
            }
            prev_tol = tol.clone();

            let x_b = Points1::from_shape_fn(self.options.n, |i| simplex[(0, i)].clone()); // Best point
            let _x_l = Points1::from_shape_fn(self.options.n, |i| simplex[(nrows - 2, i)].clone()); // Lousy point
            let x_w = Points1::from_shape_fn(self.options.n, |i| simplex[(nrows - 1, i)].clone()); // Worst point

            // Calculate the average of the n best points
            let mut x_avg = Points1::zeros(self.options.n);
            for i in 0..self.options.n {
                let mut sum = T::from_f64(0.0);
                for j in 0..self.options.n {
                    sum += simplex[(j, i)].clone();
                }
                x_avg[i] = &sum / self.options.n as f64;
            }

            // Calculate reflection point
            let x_r = Points1::from_shape_fn(self.options.n, |i| {
                &x_avg[i] + &self.options.alpha * (&x_avg[i] - &x_w[i])
            });
            let f_r = self.f.call(&Points1::from_shape_fn(self.options.n, |i| {
                &x_r[i] / &self.options.scale[i]
            }));

            //
            // Determine simplex adjustment
            //
            if f_r <= res[0] {
                // Perform expansion
                if self.options.verbosity > 1 {
                    println!("performing expansion");
                }
                let x_e = Points1::from_shape_fn(self.options.n, |i| {
                    &x_r[i] + &self.options.gamma * (&x_r[i] - &x_avg[i])
                });
                let f_e = self.f.call(&Points1::from_shape_fn(self.options.n, |i| {
                    &x_e[i] / &self.options.scale[i]
                }));
                if f_e < res[0] {
                    for i in 0..self.options.n {
                        simplex[(nrows - 1, i)] = x_e[i].clone();
                    }
                    res[nrows - 1] = f_e.clone();
                } else {
                    for i in 0..self.options.n {
                        simplex[(nrows - 1, i)] = x_r[i].clone();
                    }
                    res[nrows - 1] = f_r.clone();
                }
            } else if f_r > res[nrows - 1] {
                // Perform inside contraction
                if self.options.verbosity > 1 {
                    println!("performing inside contraction");
                }
                let x_ic = Points1::from_shape_fn(self.options.n, |i| {
                    &x_avg[i] - &self.options.beta * (&x_avg[i] - &x_w[i])
                });
                let f_ic = self.f.call(&Points1::from_shape_fn(self.options.n, |i| {
                    &x_ic[i] / &self.options.scale[i]
                }));
                if f_ic > res[nrows - 1] {
                    // Shrink simplex
                    if self.options.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.options.n {
                        for j in 0..self.options.n {
                            simplex[(i + 1, j)] =
                                &x_b[j] + &self.options.rho * (&simplex[(i + 1, j)] - &x_b[j]);
                        }
                        res[i + 1] = self.f.call(&Points1::from_shape_fn(self.options.n, |j| {
                            &simplex[(i + 1, j)] / &self.options.scale[j]
                        }));
                    }
                } else {
                    for i in 0..self.options.n {
                        simplex[(nrows - 1, i)] = x_ic[i].clone();
                    }
                    res[nrows - 1] = f_ic.clone();
                }
            } else if f_r > res[nrows - 2] {
                // Perform outside contraction
                if self.options.verbosity > 1 {
                    println!("performing outside contraction");
                }
                let x_oc = Points1::from_shape_fn(self.options.n, |i| {
                    &x_avg[i] + &self.options.beta * (&x_avg[i] - &x_w[i])
                });
                let f_oc = self.f.call(&Points1::from_shape_fn(self.options.n, |i| {
                    &x_oc[i] / &self.options.scale[i]
                }));
                let mut x_oc_vec: Vec<T> = vec![];
                for i in 0..self.options.n {
                    x_oc_vec.push(x_oc[i].clone());
                }
                if f_oc > f_r {
                    // Shrink simplex
                    if self.options.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    for i in 0..self.options.n {
                        for j in 0..self.options.n {
                            simplex[(i + 1, j)] =
                                &x_b[j] + &self.options.rho * (&simplex[(i + 1, j)] - &x_b[j]);
                        }
                        res[i + 1] = self.f.call(&Points1::from_shape_fn(self.options.n, |j| {
                            &simplex[(i + 1, j)] / &self.options.scale[j]
                        }));
                    }
                } else {
                    for i in 0..self.options.n {
                        simplex[(nrows - 1, i)] = x_oc[i].clone();
                    }
                    res[nrows - 1] = f_oc.clone();
                }
            } else {
                for i in 0..self.options.n {
                    simplex[(nrows - 1, i)] = x_r[i].clone();
                }
                res[nrows - 1] = f_r.clone();
            }

            self.tol = Some(tol.clone());
        }

        self.simplex = Some(simplex.clone());
        x_scaled = Points1::from_shape_fn(simplex.ncols(), |i| simplex[(0, i)].clone());
        // self.res = Some(res);
        self.res = Some(res.clone());

        if self.options.verbosity > 0 {
            println!("x_scaled: {:?}", x_scaled);
            println!("res: {:?}", self.res_all().unwrap());
            println!("iters: {}", self.iterations());
            println!("tol: {:?}", self.tolerance().unwrap());
        }

        Ok(NelderMeadResult {
            xmin: &x_scaled / &self.options.scale,
            fmin: self.f.call(&(&x_scaled / &self.options.scale)),
            tolerance: match self.tol.clone() {
                Some(x) => x.clone(),
                _ => T::zero(),
            },
            iters: 0,
            fn_evals: 0,
            converged: true,
            final_simplex_size: T::zero(),
            history: array![].into(),
        })
    }

    // fn minimize(&mut self, opt: &Self::Options) -> Result<Self::Result, MinimizerError> {
    //     self.options.update(opt);
    //     self.iters = 0;

    //     // Initialize RNG for perturbation and restarts
    //     let mut rng = match self.options.rng_seed {
    //         Some(seed) => StdRng::seed_from_u64(seed),
    //         None => StdRng::from_os_rng(),
    //     };

    //     // Track global best across restarts
    //     let mut best_global_point: Option<Points1<T>> = None;
    //     let mut best_global_fmin = T::infinity();
    //     let mut restart_count = 0;
    //     let mut total_fn_evals = 0;

    //     // Outer loop for restarts
    //     'restart_loop: loop {
    //         // Determine starting point for this run
    //         let x_scaled = if restart_count == 0 {
    //             &self.options.initial_point * &self.options.scale
    //         } else {
    //             if self.options.verbosity > 1 {
    //                 println!("Perturbing x point");
    //             }
    //             // Perturb best known point for restart
    //             self.generate_restart_point(&best_global_point.as_ref().unwrap())
    //         };

    //         if self.options.verbosity > 0 && restart_count > 0 {
    //             println!("Restart {} from perturbed best point", restart_count);
    //         }

    //         // Generate initial simplex vertices with adaptive sizing
    //         let simplex_scale = if self.options.use_adaptive_simplex {
    //             // Use larger simplex initially, based on bounds range
    //             let range_scale = T::from_f64(0.2); // 20% of bounds range
    //             range_scale
    //         } else {
    //             T::one()
    //         };

    //         let c = &simplex_scale * T::one();
    //         let b = &c
    //             / ((self.options.n as f64 * 2f64.sqrt())
    //                 * ((self.options.n as f64 + 1.0).sqrt() - 1.0));
    //         let a = &c / 2f64.sqrt();
    //         let nrows = self.options.n + 1;

    //         let x_scaled_a_b = x_scaled.map(|x| &a + &b + x);
    //         let x_scaled_b = x_scaled.map(|x| &b + x);
    //         let mut simplex =
    //             Points2::from_shape_fn((self.options.n + 1, self.options.n), |(i, j)| {
    //                 if i == j && i < self.options.n {
    //                     x_scaled_a_b[j].clone()
    //                 } else if i < self.options.n {
    //                     x_scaled_b[j].clone()
    //                 } else {
    //                     x_scaled[j].clone()
    //                 }
    //             });

    //         // Evaluate function at simplex vertices
    //         let mut res: Points1<T> = simplex
    //             .0
    //             .rows()
    //             .into_iter()
    //             .map(|x| {
    //                 total_fn_evals += 1;
    //                 self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
    //                     &x[i] / &self.options.scale[i]
    //                 }))
    //             })
    //             .collect();

    //         let mut stagnation_count: usize = 0;
    //         let mut prev_best_fmin = res[0].clone();
    //         let improvement_threshold = T::from_f64(1e-10);

    //         // Inner optimization loop
    //         while self.iters < self.options.max_iterations {
    //             if self.options.verbosity > 1 {
    //                 println!(
    //                     "iteration: {}\tx: {}\terr: {}\ttol: {}\tstagnation: {}",
    //                     self.iters,
    //                     simplex.row(0),
    //                     res[0],
    //                     self.tol.clone().unwrap_or(T::one()),
    //                     stagnation_count
    //                 );
    //             }
    //             self.iters += 1;

    //             // Sort points from best to worst
    //             let mut order: Vec<usize> = (0..res.len()).collect();
    //             order.sort_by(|&a, &b| {
    //                 res[a]
    //                     .partial_cmp(&res[b])
    //                     .unwrap_or(std::cmp::Ordering::Greater)
    //             });
    //             let tmp_res = res.clone();
    //             let tmp_simplex = simplex.clone();
    //             for i in 0..order.len() {
    //                 res[i] = tmp_res[order[i]].clone();
    //                 for j in 0..self.options.n {
    //                     simplex[(i, j)] = tmp_simplex[(order[i], j)].clone();
    //                 }
    //             }
    //             let shape = simplex.shape();
    //             self.simplex = Some(Points2::from_shape_fn(shape, |(i, j)| {
    //                 simplex[(i, j)].clone()
    //             }));

    //             // Check for improvement and update stagnation counter
    //             let current_fmin = res[0].clone();
    //             if (&prev_best_fmin - &current_fmin) > improvement_threshold {
    //                 stagnation_count = 0;
    //                 prev_best_fmin = current_fmin.clone();

    //                 // Update global best if this is better
    //                 if current_fmin < best_global_fmin {
    //                     best_global_fmin = current_fmin.clone();
    //                     best_global_point = Some(Points1::from_shape_fn(self.options.n, |i| {
    //                         &simplex[(0, i)] / &self.options.scale[i]
    //                     }));
    //                 }
    //             } else {
    //                 stagnation_count += 1;
    //             }

    //             // Check for stagnation and apply perturbation or restart
    //             if self.options.use_random_restart
    //                 && stagnation_count >= self.options.stagnation_threshold
    //             {
    //                 if self.options.verbosity > 0 {
    //                     println!(
    //                         "Stagnation detected after {} iterations without improvement",
    //                         stagnation_count
    //                     );
    //                 }

    //                 // Try simplex perturbation first
    //                 if self.options.use_adaptive_simplex {
    //                     self.perturb_simplex(&mut simplex, &mut res, &mut rng);
    //                     total_fn_evals += self.options.n; // Re-evaluated n vertices
    //                     stagnation_count = 0;

    //                     if self.options.verbosity > 0 {
    //                         println!("Applied simplex perturbation");
    //                     }
    //                 } else if restart_count < self.options.max_restarts {
    //                     // Trigger restart
    //                     restart_count += 1;
    //                     if self.options.verbosity > 0 {
    //                         println!("Restart {}", restart_count);
    //                     }
    //                     continue 'restart_loop;
    //                 }
    //             }

    //             // Determine if convergence criteria met
    //             let tol = T::from_f64(2.0) * (&res[res.len() - 1] - &res[0]).abs()
    //                 / (res[res.len() - 1].abs() + res[0].abs() + 1e-10);

    //             match opt.tolerance.to_f64() {
    //                 0.0 => (),
    //                 _ => {
    //                     if tol < opt.tolerance {
    //                         break;
    //                     }
    //                 }
    //             }

    //             let x_b = Points1::from_shape_fn(self.options.n, |i| simplex[(0, i)].clone());
    //             let x_w =
    //                 Points1::from_shape_fn(self.options.n, |i| simplex[(nrows - 1, i)].clone());

    //             // Calculate the average of the n best points
    //             let mut x_avg = Points1::<T>::zeros(self.options.n);
    //             for i in 0..self.options.n {
    //                 let mut sum = T::from_f64(0.0);
    //                 for j in 0..self.options.n {
    //                     sum += &simplex[(j, i)];
    //                 }
    //                 x_avg[i] = sum / self.options.n as f64;
    //             }

    //             // Calculate reflection point
    //             let mut x_r = Points1::zeros(self.options.n);
    //             azip!((index i, x_avg in x_avg.inner(), x_w in x_w.inner()) {x_r[i] = x_avg + &self.options.alpha * (x_avg - x_w);});
    //             // let x_r = &x_avg + &self.options.alpha * (&x_avg - &x_w);
    //             total_fn_evals += 1;
    //             let f_r = self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
    //                 &x_r[i] / &self.options.scale[i]
    //             }));

    //             //
    //             // Determine simplex adjustment
    //             //
    //             if f_r <= res[0] {
    //                 // Perform expansion
    //                 if self.options.verbosity > 1 {
    //                     println!("performing expansion");
    //                 }
    //                 let mut x_e = Points1::zeros(self.options.n);
    //                 azip!((index i, x_avg in x_avg.inner(), x_r in x_r.inner()) {x_e[i] = x_r + &self.options.gamma * (x_r - x_avg);});
    //                 // let x_e = &x_r + &self.options.gamma * (&x_r - &x_avg);
    //                 total_fn_evals += 1;
    //                 let f_e = self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
    //                     &x_e[i] / &self.options.scale[i]
    //                 }));
    //                 if f_e < res[0] {
    //                     for i in 0..self.options.n {
    //                         simplex[(nrows - 1, i)] = x_e[i].clone();
    //                     }
    //                     res[nrows - 1] = f_e.clone();
    //                 } else {
    //                     for i in 0..self.options.n {
    //                         simplex[(nrows - 1, i)] = x_r[i].clone();
    //                     }
    //                     res[nrows - 1] = f_r.clone();
    //                 }
    //             } else if f_r > res[nrows - 1] {
    //                 // Perform inside contraction
    //                 if self.options.verbosity > 1 {
    //                     println!("performing inside contraction");
    //                 }
    //                 let mut x_ic = Points1::zeros(self.options.n);
    //                 azip!((index i, x_avg in x_avg.inner(), x_w in x_w.inner()) {x_ic[i] = x_avg - &self.options.beta * (x_avg - x_w);});
    //                 // let x_ic = &x_avg - &self.options.beta * (&x_avg - &x_w);
    //                 total_fn_evals += 1;
    //                 let f_ic = self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
    //                     &x_ic[i] / &self.options.scale[i]
    //                 }));
    //                 if f_ic > res[nrows - 1] {
    //                     // Shrink simplex
    //                     if self.options.verbosity > 1 {
    //                         println!("shrinking simplex");
    //                     }
    //                     for i in 0..self.options.n {
    //                         for j in 0..self.options.n {
    //                             simplex[(i + 1, j)] =
    //                                 &x_b[j] + &self.options.rho * (&simplex[(i + 1, j)] - &x_b[j]);
    //                         }
    //                         total_fn_evals += 1;
    //                         res[i + 1] = self
    //                             .calc_obj(&Points1::from_shape_fn(self.options.n, |j| {
    //                                 &simplex[(i + 1, j)] / &self.options.scale[j]
    //                             }));
    //                     }
    //                 } else {
    //                     for i in 0..self.options.n {
    //                         simplex[(nrows - 1, i)] = x_ic[i].clone();
    //                     }
    //                     res[nrows - 1] = f_ic.clone();
    //                 }
    //             } else if f_r > res[nrows - 2] {
    //                 // Perform outside contraction
    //                 if self.options.verbosity > 1 {
    //                     println!("performing outside contraction");
    //                 }
    //                 let mut x_oc = Points1::zeros(self.options.n);
    //                 azip!((index i, x_avg in x_avg.inner(), x_w in x_w.inner()) {x_oc[i] = x_avg + &self.options.beta * (x_avg - x_w);});
    //                 // let x_oc = &x_avg + &self.options.beta * (&x_avg - &x_w);
    //                 total_fn_evals += 1;
    //                 let f_oc = self.calc_obj(&Points1::from_shape_fn(self.options.n, |i| {
    //                     &x_oc[i] / &self.options.scale[i]
    //                 }));
    //                 if f_oc > f_r {
    //                     // Shrink simplex
    //                     if self.options.verbosity > 1 {
    //                         println!("shrinking simplex");
    //                     }
    //                     for i in 0..self.options.n {
    //                         for j in 0..self.options.n {
    //                             simplex[(i + 1, j)] =
    //                                 &x_b[j] + &self.options.rho * (&simplex[(i + 1, j)] - &x_b[j]);
    //                         }
    //                         total_fn_evals += 1;
    //                         res[i + 1] = self
    //                             .calc_obj(&Points1::from_shape_fn(self.options.n, |j| {
    //                                 &simplex[(i + 1, j)] / &self.options.scale[j]
    //                             }));
    //                     }
    //                 } else {
    //                     for i in 0..self.options.n {
    //                         simplex[(nrows - 1, i)] = x_oc[i].clone();
    //                     }
    //                     res[nrows - 1] = f_oc.clone();
    //                 }
    //             } else {
    //                 for i in 0..self.options.n {
    //                     simplex[(nrows - 1, i)] = x_r[i].clone();
    //                 }
    //                 res[nrows - 1] = f_r.clone();
    //             }

    //             self.tol = Some(tol.clone());
    //         }

    //         // Check if we should try another restart
    //         if restart_count < self.options.max_restarts
    //             && self.options.use_random_restart
    //             && self.iters < self.options.max_iterations
    //         {
    //             restart_count += 1;
    //             if self.options.verbosity > 0 {
    //                 println!("Restart {}", restart_count);
    //             }
    //             continue 'restart_loop;
    //         }

    //         // Final update of global best from this run
    //         if res[0] < best_global_fmin {
    //             best_global_fmin = res[0].clone();
    //             best_global_point = Some(Points1::from_shape_fn(self.options.n, |i| {
    //                 &simplex[(0, i)] / &self.options.scale[i]
    //             }));
    //         }

    //         let shape = simplex.shape();
    //         self.simplex = Some(Points2::from_shape_fn(shape, |(i, j)| {
    //             simplex[(i, j)].clone()
    //         }));

    //         self.res = Some(Points1::from_shape_fn(res.len(), |i| res[i].clone()));
    //         break;
    //     }

    //     if self.options.verbosity > 0 {
    //         println!("Total restarts: {}", restart_count);
    //         println!("Total function evaluations: {}", total_fn_evals);
    //         println!("Best xmin: {:?}", best_global_point);
    //         println!("Best fmin: {:?}", best_global_fmin);
    //         println!("iters: {}", self.iterations());
    //         println!("tol: {:?}", self.tolerance().unwrap_or(T::zero()));
    //     }

    //     let final_x = best_global_point.unwrap_or_else(|| self.options.initial_point.clone());
    //     Ok(NelderMeadResult {
    //         xmin: final_x.clone(),
    //         fmin: best_global_fmin,
    //         tolerance: match self.tol.clone() {
    //             Some(x) => x,
    //             _ => T::zero(),
    //         },
    //         iters: self.iters,
    //         fn_evals: total_fn_evals,
    //         converged: true,
    //         final_simplex_size: T::zero(),
    //         history: array![].into(),
    //     })
    // }
}

#[cfg(test)]
mod minimize_neldermead_tests {
    use super::*;
    use crate::{
        file::read_touchstone,
        frequency::{FreqArray, Frequency},
        minimize::MultiDimFn,
        network::{Network, NetworkBuilder},
        num::{MyComplex, MyFloat},
        pts::{Points, Points3, Pts},
        util::*,
    };
    use float_cmp::F64Margin;
    use num::complex::Complex64;

    fn calc_feed_y(freq: Frequency, x: &Points1<MyFloat>) -> Points3<MyComplex> {
        let mut yfeed = Points3::<MyComplex>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut zs = Points1::<MyComplex>::zeros(freq.npts());
        let mut zm = Points1::<MyComplex>::zeros(freq.npts());
        let mut zp = Points1::<MyComplex>::zeros(freq.npts());
        let mut zall = Points1::<MyComplex>::zeros(freq.npts());
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

    fn eval_f_simplex(x: &Points1<MyFloat>, meas: &Network) -> MyFloat {
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
    fn neldermead_iter1_() {
        let x: Points1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ]
        .into();
        let scale: Points1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ]
        .into();
        let exemplar_res = MyFloat::new(5736.50737797145);
        let exemplar_tol = MyFloat::new(1.9902303400016697);
        let exemplar_x: Points1<MyFloat> = array![
            MyFloat::new(1.0e-11),
            1.0e-03.into(),
            1.0e-13.into(),
            1.0e-06.into(),
            1.0e-15.into(),
            1.0e03.into(),
            1.0e-15.into(),
            1.0e-15.into()
        ]
        .into();
        let exemplar_simplex: Points2<MyFloat> = array![
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
        ]
        .into();

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let objective =
            MultiDimFn::new(move |x: &Points1<MyFloat>| -> MyFloat { eval_f_simplex(x, &net) });
        let options = NelderMeadOptions::new(
            &x,
            Some(&scale),
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mut minimizer = NelderMead::new(objective);
        let test = minimizer.minimize(&options).unwrap();

        let margin = MARGIN;
        comp_pts_ix2(
            &exemplar_simplex,
            minimizer.simplex().unwrap(),
            margin,
            "minimize(simplex)",
        );
        comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
        comp_myfloat(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn neldermead_iter2_() {
        let x: Points1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ]
        .into();
        let scale: Points1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ]
        .into();
        let exemplar_res = 5736.50737797145.into();
        let exemplar_tol = 1.9726210586239914.into();
        let exemplar_x: Points1<MyFloat> = array![
            1.0e-11.into(),
            1.0e-03.into(),
            1.0e-13.into(),
            1.0e-06.into(),
            1.0e-15.into(),
            1.0e03.into(),
            1.0e-15.into(),
            1.0e-15.into()
        ]
        .into();

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = NelderMeadOptions::new(
            &x,
            Some(&scale),
            Some(2),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mut minimizer = NelderMead::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            eval_f_simplex(x, &net)
        }));
        let test = minimizer.minimize(&options).unwrap();

        let margin = MARGIN;
        comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
        comp_myfloat(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn neldermead_iter5_() {
        let x: Points1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ]
        .into();
        let scale: Points1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ]
        .into();
        let exemplar_res = 5736.50737797145.into();
        let exemplar_tol = 1.6060281828650276.into();
        let exemplar_x: Points1<MyFloat> = array![
            1.0e-11.into(),
            1.0e-03.into(),
            1.0e-13.into(),
            1.0e-06.into(),
            1.0e-15.into(),
            1.0e03.into(),
            1.0e-15.into(),
            1.0e-15.into()
        ]
        .into();

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = NelderMeadOptions::new(
            &x,
            Some(&scale),
            Some(5),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mut minimizer = NelderMead::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            eval_f_simplex(x, &net)
        }));
        let test = minimizer.minimize(&options).unwrap();

        let margin = MARGIN;
        comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
        comp_myfloat(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn neldermead_iter6_() {
        let x: Points1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ]
        .into();
        let scale: Points1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ]
        .into();
        let exemplar_res = 5736.507376515564.into();
        let exemplar_tol = 1.5066440937567893.into();
        let exemplar_x: Points1<MyFloat> = array![
            1.0496148654572788e-11.into(),
            0.001.into(),
            MyFloat::new(-2.5331914829451519e-14),
            0.000001.into(),
            9.9896419905099658e-16.into(),
            1000.into(),
            1.4961486545727874e-15.into(),
            1.4961486545727874e-15.into()
        ]
        .into();

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = NelderMeadOptions::new(
            &x,
            Some(&scale),
            Some(6),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mut minimizer = NelderMead::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            eval_f_simplex(x, &net)
        }));
        let test = minimizer.minimize(&options).unwrap();

        let margin = MARGIN;
        comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
        comp_myfloat(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn neldermead_iter10_() {
        let x: Points1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ]
        .into();
        let scale: Points1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ]
        .into();
        let exemplar_res = 3267.394011570648.into();
        let exemplar_tol = 1.6651438670237593.into();
        let exemplar_x: Points1<MyFloat> = array![
            1.0496148654572788e-11.into(),
            0.0028955966586259665.into(),
            MyFloat::new(-2.5331914829451519e-14),
            0.36146316242264603.into(),
            9.9896419905099658e-16.into(),
            999.4223359686597.into(),
            1.4961486545727874e-15.into(),
            1.4961486545727874e-15.into()
        ]
        .into();

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = NelderMeadOptions::new(
            &x,
            Some(&scale),
            Some(10),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mut minimizer = NelderMead::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            eval_f_simplex(x, &net)
        }));
        let test = minimizer.minimize(&options).unwrap();

        let margin = MARGIN;
        comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
        comp_myfloat(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance(),
            margin,
            "minimize(tol)",
            "",
        );
    }

    #[test]
    fn neldermead_iter100_() {
        let x: Points1<MyFloat> = array![
            1e-11.into(),
            1e-3.into(),
            1e-13.into(),
            1e-6.into(),
            1e-15.into(),
            1000.0.into(),
            1e-15.into(),
            1e-15.into()
        ]
        .into();
        let scale: Points1<MyFloat> = array![
            1e12.into(),
            1.0.into(),
            1e12.into(),
            1.0.into(),
            1e15.into(),
            1.0.into(),
            1e15.into(),
            1e15.into()
        ]
        .into();
        let exemplar_res = 765.6358932973524.into();
        let exemplar_tol = 0.12081175542092283.into();
        let exemplar_x: Points1<MyFloat> = array![
            1.6110247811984210e-11.into(),
            0.010202461287747289.into(),
            1.2618495970192951e-12.into(),
            6.410718597400084.into(),
            MyFloat::new(-2.1119233982250064e-16),
            991.264467681222.into(),
            4.5268760596139400e-15.into(),
            9.1487688225672705e-15.into()
        ]
        .into();

        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let options = NelderMeadOptions::new(
            &x,
            Some(&scale),
            Some(100),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mut minimizer = NelderMead::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            eval_f_simplex(x, &net)
        }));
        let test = minimizer.minimize(&options).unwrap();

        let margin = MARGIN;
        comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
        comp_myfloat(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
        comp_myfloat(
            &exemplar_tol,
            &test.tolerance(),
            margin,
            "minimize(tol)",
            "",
        );
    }
}
