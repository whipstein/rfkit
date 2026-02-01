use crate::{
    error::MinimizerError,
    minimize::{Minimizer, ObjFn},
    num::RFFloat,
    pts::{Matrix, Points, Points1, Points2, Pts},
};
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::fmt;

/// Conjugate gradient update formulas
#[derive(Debug, Clone, PartialEq)]
pub enum NelderMeadMethod<T>
where
    T: RFFloat,
{
    Initial,
    Shrink,
    Reflection,
    Expansion(Points1<T>),
    InsideContraction,
    OutsideContraction,
    Restart,
    Perturb,
}

impl<T> fmt::Display for NelderMeadMethod<T>
where
    T: RFFloat,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NelderMeadMethod::Initial => write!(f, "Initial"),
            NelderMeadMethod::Shrink => write!(f, "Shrink"),
            NelderMeadMethod::Reflection => write!(f, "Reflection"),
            NelderMeadMethod::Expansion(_) => write!(f, "Expansion"),
            NelderMeadMethod::InsideContraction => write!(f, "InsideContraction"),
            NelderMeadMethod::OutsideContraction => write!(f, "OutsideContraction"),
            NelderMeadMethod::Restart => write!(f, "Restart"),
            NelderMeadMethod::Perturb => write!(f, "Perturb"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NelderMeadOptions<T>
where
    T: RFFloat,
{
    initial_point: Points1<T>,
    lower_bound: Points1<T>,
    upper_bound: Points1<T>,
    scale: Points1<T>,
    n: usize,
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
    max_restarts: usize,         // Maximum number of random restarts
    perturbation_scale: T,       // Scale for random perturbation (fraction of bounds range)
    rng_seed: Option<u64>,       // Optional seed for reproducibility
    bounded: bool,
    verbosity: usize,
}

impl<T> NelderMeadOptions<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a> f64: std::ops::Mul<&'a T, Output = T>,
{
    pub fn new(
        init: &Points1<T>,
        scale: Option<&Points1<T>>,
        lb: Option<&Points1<T>>,
        ub: Option<&Points1<T>>,
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
            initial_point: init.to_owned(),
            lower_bound: lb.unwrap_or(&Points1::zeros(n)).to_owned(),
            upper_bound: ub.unwrap_or(&Points1::zeros(n)).to_owned(),
            scale: scale.unwrap_or(&Points1::ones(n)).to_owned(),
            n,
            max_iterations: max_iters.unwrap_or(1000),
            tolerance: tol.unwrap_or(T::from_f64(0.0)),
            alpha: alpha.unwrap_or(T::from_f64(1.0)),
            beta: beta.unwrap_or(T::from_f64(0.5)),
            gamma: gamma.unwrap_or(T::from_f64(2.0)),
            rho: rho.unwrap_or(T::from_f64(0.5)),
            mu: mu.unwrap_or(T::from_f64(10.0)),
            use_random_restart: false, // Disabled by default for backward compatibility
            use_adaptive_simplex: false, // Disabled by default for backward compatibility
            stagnation_iters: 0,
            stagnation_threshold: 50,
            max_restarts: 3,
            perturbation_scale: T::from_f64(0.1),
            rng_seed: None,
            bounded: lb.is_some() && ub.is_some(),
            verbosity: verbosity.unwrap_or(0),
        }
    }

    fn update(&mut self, opt: &NelderMeadOptions<T>) {
        self.initial_point = opt.initial_point.clone();
        self.lower_bound = opt.lower_bound.clone();
        self.upper_bound = opt.upper_bound.clone();
        self.scale = opt.scale.clone();
        self.n = opt.n;
        self.max_iterations = opt.max_iterations;
        self.tolerance = opt.tolerance.clone();
        self.alpha = opt.alpha.clone();
        self.beta = opt.beta.clone();
        self.gamma = opt.gamma.clone();
        self.rho = opt.rho.clone();
        self.mu = opt.mu.clone();
        self.use_random_restart = opt.use_random_restart;
        self.use_adaptive_simplex = opt.use_adaptive_simplex;
        self.stagnation_iters = opt.stagnation_iters;
        self.stagnation_threshold = opt.stagnation_threshold;
        self.max_restarts = opt.max_restarts;
        self.perturbation_scale = opt.perturbation_scale.clone();
        self.rng_seed = opt.rng_seed;
        self.bounded = opt.bounded;
        self.verbosity = opt.verbosity;
    }

    fn check_bounds(&self, x_scaled: &mut Points1<T>, iters: usize) {
        if self.is_bounded() {
            azip!((index i, lb in self.lower_bound.inner(), ub in self.upper_bound.inner(), scale in self.scale.inner()) {
                if x_scaled[i] < lb * scale {
                    x_scaled[i] = (1.0 + 0.05 / (iters + 1) as f64) * lb * scale;
                } else if x_scaled[i] > ub * scale {
                    x_scaled[i] = (1.0 - 0.05 / (iters + 1) as f64) * ub * scale;
                } else {
                    ()
                }
            });
        }
    }

    fn check_bounds_pt(&self, x_scaled: &mut T, iters: usize, pt: usize) {
        if self.is_bounded() {
            if *x_scaled < &self.lower_bound[pt] * &self.scale[pt] {
                *x_scaled =
                    (1.0 + 0.05 / (iters + 1) as f64) * &self.lower_bound[pt] * &self.scale[pt];
            } else if *x_scaled > &self.upper_bound[pt] * &self.scale[pt] {
                *x_scaled =
                    (1.0 - 0.05 / (iters + 1) as f64) * &self.upper_bound[pt] * &self.scale[pt];
            } else {
                ()
            }
        }
    }

    pub fn set_initial_point(&mut self, init: Points1<T>) {
        self.initial_point = init;
    }

    pub fn set_scale(&mut self, scale: Points1<T>) {
        self.scale = scale;
    }

    pub fn set_lower_bounds(&mut self, lb: Points1<T>) {
        self.lower_bound = lb;
        self.bounded = true;
    }

    pub fn set_upper_bounds(&mut self, ub: Points1<T>) {
        self.upper_bound = ub;
        self.bounded = true;
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

    pub fn set_max_restarts(&mut self, val: usize) {
        self.max_restarts = val;
    }

    pub fn set_perturbation_scale(&mut self, val: T) {
        self.perturbation_scale = val;
    }

    pub fn set_rng_seed(&mut self, seed: Option<u64>) {
        self.rng_seed = seed;
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

    pub fn initial_point(&self) -> &Points1<T> {
        &self.initial_point
    }

    pub fn scale(&self) -> &Points1<T> {
        &self.scale
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn tolerance(&self) -> T {
        self.tolerance.clone()
    }

    pub fn verbosity(&self) -> usize {
        self.verbosity
    }

    pub fn is_bounded(&self) -> bool {
        self.bounded
    }
}

/// Result of Nelder-Mead optimization
#[derive(Clone)]
pub struct NelderMeadResult<T>
where
    T: RFFloat,
{
    pub simplex: Points2<T>,
    pub res: Points1<T>,
    pub tol: T,
    f: Box<dyn ObjFn<T>>,
    pub iters: usize,
    pub fn_evals: usize,
    pub options: NelderMeadOptions<T>,
    pub converged: bool,
    pub final_simplex_size: T,
    pub history: Points1<T>,
}

impl<T> NelderMeadResult<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Add<&'b T, Output = T>
        + std::ops::Sub<&'b T, Output = T>
        + std::ops::Mul<&'b T, Output = T>,
    for<'a> &'a T: std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::ops::Div<f64, Output = T>,
    f64: std::ops::Mul<T, Output = T>,
    for<'a> f64: std::ops::Mul<&'a T, Output = T>,
{
    pub fn new(options: &NelderMeadOptions<T>, f: &Box<dyn ObjFn<T>>) -> Self
    where
        for<'a> Points1<f64>: std::ops::Mul<&'a Points1<T>, Output = Points1<T>>,
    {
        let mut out = Self {
            simplex: Points2::zeros((options.n + 1, options.n)),
            res: Points1::zeros(options.n + 1),
            tol: T::infinity(),
            f: f.clone(),
            iters: 0,
            fn_evals: 0,
            options: options.clone(),
            converged: false,
            final_simplex_size: T::infinity(),
            history: Points1::zeros(options.n),
        };
        out.generate_point(NelderMeadMethod::Initial);

        out
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn fmin(&self) -> T {
        self.res[0].clone()
    }

    pub fn tolerance(&self) -> T {
        self.tol.clone()
    }

    pub fn fn_evals(&self) -> usize {
        self.fn_evals
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn xmin(&self) -> Points1<T> {
        self.best_x()
    }

    pub fn history(&self) -> Points1<T> {
        self.history.clone()
    }

    pub fn calc_obj(&mut self, x: &Points1<T>) -> T {
        self.fn_evals += 1;
        let mut sum = T::zero();
        if self.options.is_bounded() {
            for i in 0..x.len() {
                sum += (&self.options.upper_bound[i] - &x[i]).ln()
                    + (&x[i] - &self.options.lower_bound[i]).ln();
            }
            self.f.call(x) - &self.options.mu * sum
        } else {
            self.f.call(x)
        }
    }

    pub fn calc_obj_scaled(&mut self, x_scaled: &Points1<T>) -> T {
        let x = x_scaled / &self.options.scale;
        self.calc_obj(&x)
    }

    pub fn average_x(&self) -> Points1<T> {
        let mut x_avg = Points1::zeros(self.options.n);
        for (i, pt) in self.simplex.axis_iter(Axis(1)).enumerate() {
            x_avg[i] = pt.slice(s![..-1]).iter().sum::<T>() / self.options.n as f64;
        }

        x_avg
    }

    pub fn best_f(&self) -> T {
        self.res[0].clone()
    }

    pub fn best_x(&self) -> Points1<T> {
        self.best_x_scaled() / &self.options.scale
    }

    pub fn best_x_scaled(&self) -> Points1<T> {
        self.simplex.row(0)
    }

    pub fn calc_tol(&mut self) {
        self.tol = 2.0 * (&self.res[self.res.len() - 1] - &self.res[0]).abs()
            / (self.res[self.res.len() - 1].abs() + self.res[0].abs() + 1e-15);
    }

    pub fn update_pt(&mut self, index: usize, simplex: &Points1<T>, res: &T) {
        self.simplex.set_row(index, simplex);
        self.res[index] = res.clone();
        self.calc_tol();
        self.sort_simplex();
    }

    pub fn update_calc_pt(&mut self, index: usize, simplex: &Points1<T>) {
        self.simplex.set_row(index, simplex);
        self.res[index] = self.calc_obj_scaled(simplex);
        self.calc_tol();
        self.sort_simplex();
    }

    pub fn generate_point(&mut self, method: NelderMeadMethod<T>) -> Option<(Points1<T>, T)>
    where
        for<'a> Points1<f64>: std::ops::Mul<&'a Points1<T>, Output = Points1<T>>,
    {
        match method {
            NelderMeadMethod::Initial => {
                let x0_scaled = &self.options.initial_point * &self.options.scale;
                let c = T::one();
                let b = &c
                    / ((self.options.n as f64 * 2f64.sqrt())
                        * ((self.options.n as f64 + 1.0).sqrt() - 1.0));
                let a = &b + &c / 2f64.sqrt();
                let mut x0_a_b = x0_scaled.map(|x| &a + &b + x);
                let mut x0_b = x0_scaled.map(|x| &b + x);
                self.options.check_bounds(&mut x0_a_b, 0);
                self.options.check_bounds(&mut x0_b, 0);
                self.simplex =
                    Points2::from_shape_fn((self.options.n + 1, self.options.n), |(i, j)| {
                        if i == j && i < self.options.n {
                            x0_a_b[j].clone()
                        } else if i < self.options.n {
                            x0_b[j].clone()
                        } else {
                            x0_scaled[j].clone()
                        }
                    });
                self.res = Points1::zeros(self.options.n + 1);
                for i in 0..self.res.len() {
                    self.res[i] = self.calc_obj_scaled(&self.simplex.row(i));
                }
                self.calc_tol();
                self.sort_simplex();
                None
            }
            NelderMeadMethod::Shrink => {
                let mut x_s = Points1::zeros(self.options.n);
                for i in 0..self.simplex.nrows() - 1 {
                    azip!((index j, x_b in self.simplex.row(0).inner()) {x_s[j] = x_b + &self.options.rho * (&self.simplex[(i + 1, j)] - x_b)});
                    self.options.check_bounds(&mut x_s, self.iters);
                    self.update_calc_pt(i, &x_s);
                }
                self.calc_tol();
                self.sort_simplex();
                None
            }
            NelderMeadMethod::Expansion(x_r) => {
                let mut x_e = Points1::zeros(self.options.n);
                azip!((index j, x_avg in self.average_x().inner(), x_r in x_r.inner()) {x_e[j] = x_r + &self.options.gamma * (x_r - x_avg);});
                self.options.check_bounds(&mut x_e, self.iters);
                let f_e = self.calc_obj_scaled(&x_e);
                Some((x_e, f_e))
            }
            NelderMeadMethod::Reflection => {
                let mut x_r = Points1::zeros(self.options.n);
                azip!((index j, x_avg in self.average_x().inner(), x_w in self.simplex.row(self.simplex.nrows() - 1).inner()) {x_r[j] = x_avg + &self.options.alpha * (x_avg - x_w);});
                self.options.check_bounds(&mut x_r, self.iters);
                let f_r = self.calc_obj_scaled(&x_r);
                Some((x_r, f_r))
            }
            NelderMeadMethod::InsideContraction => {
                let mut x_ic = Points1::zeros(self.options.n);
                azip!((index j, x_avg in self.average_x().inner(), x_w in self.simplex.row(self.simplex.nrows() - 1).inner()) {x_ic[j] = x_avg - &self.options.beta * (x_avg - x_w);});
                self.options.check_bounds(&mut x_ic, self.iters);
                let f_ic = self.calc_obj_scaled(&x_ic);
                Some((x_ic, f_ic))
            }
            NelderMeadMethod::OutsideContraction => {
                let mut x_oc = Points1::zeros(self.options.n);
                azip!((index j, x_avg in self.average_x().inner(), x_w in self.simplex.row(self.simplex.nrows() - 1).inner()) {x_oc[j] = x_avg + &self.options.beta * (x_avg - x_w);});
                self.options.check_bounds(&mut x_oc, self.iters);
                let f_oc = self.calc_obj_scaled(&x_oc);
                Some((x_oc, f_oc))
            }
            NelderMeadMethod::Restart => {
                let mult = if self.options.is_bounded() {
                    &(&self.options.upper_bound - &self.options.lower_bound)
                        * &self.options.scale
                        * &self.options.perturbation_scale
                } else {
                    Points1::from_elem(self.options.n, 1e3 * &self.options.perturbation_scale)
                };
                let mut out = Points(Array1::random(
                    self.options.n,
                    Uniform::new(-1_f64, 1_f64).unwrap(),
                )) * &mult
                    + self.simplex.row(0);
                self.options.check_bounds(&mut out, self.iters);
                self.simplex.set_row(0, &out);
                self.res[0] = self.calc_obj_scaled(&out);
                self.calc_tol();
                self.sort_simplex();
                None
            }
            NelderMeadMethod::Perturb => {
                if self.options.verbosity > 5 {
                    println!("\nPerturbed simplex\noriginal: {:?}", self.simplex.row(1));
                }

                // Perturb all vertices except the best one (index 0)
                for i in 1..(self.options.n + 1) {
                    let range = if self.options.is_bounded() {
                        (&self.options.upper_bound - &self.options.lower_bound)
                            * &self.options.scale
                            * &self.options.perturbation_scale
                            * T::from_f64(0.5)
                    } else {
                        Points1::from_elem(
                            self.options.n,
                            1e-3 * &self.options.perturbation_scale * T::from_f64(0.5),
                        )
                    }; // Smaller perturbation for simplex vertices
                    let mut new_val = self.simplex.row(i)
                        + Points(Array1::random(
                            self.options.n,
                            Uniform::new(-1_f64, 1_f64).unwrap(),
                        )) * &range;

                    // Clamp to bounds
                    self.options.check_bounds(&mut new_val, self.iters);
                    self.update_calc_pt(i, &new_val);
                }

                if self.options.verbosity > 5 {
                    println!("new: {:?}\n", self.simplex.row(1));
                }

                self.calc_tol();
                self.sort_simplex();
                None
            }
        }
    }

    // Sort points from best to worst
    fn sort_simplex(&mut self) {
        let mut order: Vec<usize> = (0..self.res.len()).collect();
        order.sort_by(|&a, &b| self.res[a].partial_cmp(&self.res[b]).unwrap());
        let tmp_res = self.res.clone();
        let tmp_simplex = self.simplex.clone();
        for i in 0..order.len() {
            self.res[i] = tmp_res[order[i]].clone();
            self.simplex.set_row(i, &tmp_simplex.row(order[i]));
        }
    }
}

pub struct NelderMead<T>
where
    T: RFFloat,
{
    f: Box<dyn ObjFn<T>>,
}

impl<T> NelderMead<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Add<&'b T, Output = T>
        + std::ops::Sub<&'b T, Output = T>
        + std::ops::Mul<&'b T, Output = T>
        + std::ops::Div<&'b T, Output = T>,
    for<'a> &'a T:
        std::ops::Add<T, Output = T> + std::ops::Sub<T, Output = T> + std::ops::Mul<T, Output = T>,
    for<'a> &'a T: std::ops::Div<f64, Output = T>,
    f64: std::ops::Mul<T, Output = T>,
    for<'a> f64: std::ops::Mul<&'a T, Output = T>,
    for<'a> Points1<f64>: std::ops::Mul<&'a Points1<T>, Output = Points1<T>>,
{
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        NelderMead { f: Box::new(f) }
    }

    pub fn name(&self) -> &str {
        "NelderMead"
    }

    fn minimize_wo_restart(
        &mut self,
        opt: &NelderMeadOptions<T>,
    ) -> Result<NelderMeadResult<T>, MinimizerError> {
        let mut out = NelderMeadResult::new(opt, &self.f);

        // Generate initial simplex veritces
        let nrows = out.options.n + 1;

        // Evaluate function at simplex vertices
        let mut prev_tol = T::one();

        while out.iters < out.options.max_iterations {
            if out.options.verbosity > 1 {
                println!(
                    "iteration: {}\terr: {}\ttol: {}",
                    out.iters,
                    out.res[0],
                    out.tol.clone()
                );
            }
            out.iters += 1;

            // Determine if convergence criteria met
            match opt.tolerance.to_f64() {
                0.0 => (),
                _ => {
                    if out.tol.is_finite()
                        && (out.tol < out.options.tolerance
                            || ((&prev_tol - &out.tol).abs() < 1e-15.into()
                                && prev_tol.is_finite()))
                    {
                        break;
                    }
                }
            }
            prev_tol = out.tol.clone();

            //
            // Calculate reflection point
            //
            let (x_r, f_r) = out.generate_point(NelderMeadMethod::Reflection).unwrap();

            //
            // Determine simplex adjustment
            //
            if f_r <= out.res[0] {
                //
                // Perform expansion
                //
                if out.options.verbosity > 1 {
                    println!("performing expansion");
                }
                let (x_e, f_e) = out
                    .generate_point(NelderMeadMethod::Expansion(x_r.clone()))
                    .unwrap();
                if f_e < out.res[0] {
                    out.update_pt(nrows - 1, &x_e, &f_e);
                } else {
                    out.update_pt(nrows - 1, &x_r, &f_r);
                }
            } else if f_r > out.res[nrows - 1] {
                //
                // Perform inside contraction
                //
                if out.options.verbosity > 1 {
                    println!("performing inside contraction");
                }
                let (x_ic, f_ic) = out
                    .generate_point(NelderMeadMethod::InsideContraction)
                    .unwrap();
                if f_ic > out.res[nrows - 1] {
                    //
                    // Shrink simplex
                    //
                    if out.options.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    out.generate_point(NelderMeadMethod::Shrink);
                } else {
                    out.update_pt(nrows - 1, &x_ic, &f_ic);
                }
            } else if f_r > out.res[nrows - 2] {
                //
                // Perform outside contraction
                //
                if out.options.verbosity > 1 {
                    println!("performing outside contraction");
                }
                let (x_oc, f_oc) = out
                    .generate_point(NelderMeadMethod::OutsideContraction)
                    .unwrap();
                if f_oc > f_r {
                    //
                    // Shrink simplex
                    //
                    if out.options.verbosity > 1 {
                        println!("shrinking simplex");
                    }
                    out.generate_point(NelderMeadMethod::Shrink);
                } else {
                    out.update_pt(nrows - 1, &x_oc, &f_oc);
                }
            } else {
                out.update_pt(nrows - 1, &x_r, &f_r);
            }
        }

        if out.options.verbosity > 0 {
            self.verbose_result(&out, 0);
        }

        Ok(out)
    }

    fn minimize_w_restart(
        &mut self,
        opt: &NelderMeadOptions<T>,
    ) -> Result<NelderMeadResult<T>, MinimizerError> {
        let mut out = NelderMeadResult::new(opt, &self.f);

        let nrows = out.options.n + 1;

        // Track global best across restarts
        let mut restart_count = 0;

        // Outer loop for restarts
        'restart_loop: loop {
            // Determine starting point for this run
            if restart_count != 0 {
                if out.options.verbosity > 2 {
                    println!("Perturbing x point");
                }
                // Perturb best known point for restart
                out.generate_point(NelderMeadMethod::Restart);
            };

            if out.options.verbosity > 2 && restart_count > 0 {
                println!("Restart {} from perturbed best point", restart_count);
            }

            let mut prev_tol = T::one();
            let mut stagnation_count: usize = 0;
            let mut prev_best_fmin = out.res[0].clone();
            let improvement_threshold = T::from_f64(1e-10);

            // Inner optimization loop
            while out.iters < out.options.max_iterations {
                if out.options.verbosity > 3 {
                    println!(
                        "iteration: {}\tx: {}\terr: {}\ttol: {}\tstagnation: {}",
                        out.iters,
                        out.simplex.row(0),
                        out.res[0],
                        out.tol,
                        stagnation_count
                    );
                }
                out.iters += 1;

                // Check for improvement and update stagnation counter
                let current_fmin = out.best_f();
                if (&prev_best_fmin - &current_fmin) > improvement_threshold {
                    stagnation_count = 0;
                    prev_best_fmin = current_fmin.clone();
                } else {
                    stagnation_count += 1;
                }

                // Check for stagnation and apply perturbation or restart
                if out.options.use_random_restart
                    && stagnation_count >= out.options.stagnation_threshold
                {
                    if out.options.verbosity > 1 {
                        println!(
                            "Stagnation detected after {} iterations without improvement",
                            stagnation_count
                        );
                    }

                    // Try simplex perturbation first
                    if out.options.use_adaptive_simplex {
                        out.generate_point(NelderMeadMethod::Perturb);
                        stagnation_count = 0;

                        if out.options.verbosity > 1 {
                            println!("Applied simplex perturbation");
                        }
                    } else if restart_count < out.options.max_restarts {
                        // Trigger restart
                        restart_count += 1;
                        if out.options.verbosity > 1 {
                            println!("Restart {}", restart_count);
                        }
                        continue 'restart_loop;
                    }
                }

                // Determine if convergence criteria met
                match opt.tolerance.to_f64() {
                    0.0 => (),
                    _ => {
                        if out.tol.is_finite()
                            && (out.tol < opt.tolerance
                                || ((&prev_tol - &out.tol).abs() < 1e-15.into()
                                    && prev_tol.is_finite()))
                        {
                            break;
                        }
                    }
                }
                prev_tol = out.tol.clone();

                // Calculate reflection point
                let (x_r, f_r) = out.generate_point(NelderMeadMethod::Reflection).unwrap();

                //
                // Determine simplex adjustment
                //
                if f_r <= out.res[0] {
                    //
                    // Perform expansion
                    //
                    if out.options.verbosity > 2 {
                        println!("performing expansion");
                    }
                    let (x_e, f_e) = out
                        .generate_point(NelderMeadMethod::Expansion(x_r.clone()))
                        .unwrap();
                    if f_e < out.res[0] {
                        out.update_calc_pt(nrows - 1, &x_e);
                    } else {
                        out.update_calc_pt(nrows - 1, &x_r);
                    }
                } else if f_r > out.res[nrows - 1] {
                    //
                    // Perform inside contraction
                    //
                    if out.options.verbosity > 2 {
                        println!("performing inside contraction");
                    }
                    let (x_ic, f_ic) = out
                        .generate_point(NelderMeadMethod::InsideContraction)
                        .unwrap();
                    if f_ic > out.res[nrows - 1] {
                        //
                        // Shrink simplex
                        //
                        if out.options.verbosity > 2 {
                            println!("shrinking simplex");
                        }
                        out.generate_point(NelderMeadMethod::Shrink);
                    } else {
                        out.update_calc_pt(nrows - 1, &x_ic);
                    }
                } else if f_r > out.res[nrows - 2] {
                    //
                    // Perform outside contraction
                    //
                    if out.options.verbosity > 2 {
                        println!("performing outside contraction");
                    }
                    let (x_oc, f_oc) = out
                        .generate_point(NelderMeadMethod::OutsideContraction)
                        .unwrap();
                    if f_oc > f_r {
                        //
                        // Shrink simplex
                        //
                        if out.options.verbosity > 2 {
                            println!("shrinking simplex");
                        }
                        out.generate_point(NelderMeadMethod::Shrink);
                    } else {
                        out.update_calc_pt(nrows - 1, &x_oc);
                    }
                } else {
                    out.update_calc_pt(nrows - 1, &x_r);
                }
            }

            // Check if we should try another restart
            if restart_count < out.options.max_restarts
                && out.options.use_random_restart
                && out.iters < out.options.max_iterations
            {
                restart_count += 1;
                if out.options.verbosity > 1 {
                    println!("Restart {}", restart_count);
                }
                continue 'restart_loop;
            }

            break;
        }

        if out.options.verbosity > 0 {
            self.verbose_result(&out, restart_count);
        }

        Ok(out)
    }

    fn verbose_result(&self, result: &NelderMeadResult<T>, restarts: usize) {
        println!("Total restarts: {}", restarts);
        println!("Total function evaluations: {}", result.fn_evals);
        println!("Best xmin:");
        for i in 0..result.options.n {
            println!("{:?}", result.best_x()[i]);
        }
        println!("Best fmin: {:?}", result.best_f());
        println!("iters: {}", result.iters);
        println!("tol: {:?}", result.tol);
    }
}

impl<T> Minimizer<T> for NelderMead<T>
where
    T: RFFloat + 'static,
    for<'a, 'b> &'a T: std::ops::Add<&'b T, Output = T>
        + std::ops::Sub<&'b T, Output = T>
        + std::ops::Mul<&'b T, Output = T>
        + std::ops::Div<&'b T, Output = T>,
    for<'a> &'a T: std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Mul<T, Output = T>
        + std::ops::Div<f64, Output = T>,
    f64: std::ops::Add<T, Output = T> + std::ops::Sub<T, Output = T> + std::ops::Mul<T, Output = T>,
    for<'a> f64: std::ops::Add<&'a T, Output = T>
        + std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>,
    for<'a> Points1<f64>: std::ops::Mul<&'a Points1<T>, Output = Points1<T>>,
{
    type Options = NelderMeadOptions<T>;
    type Result = NelderMeadResult<T>;

    fn minimize(&mut self, opt: &Self::Options) -> Result<Self::Result, MinimizerError> {
        match opt.use_random_restart {
            true => self.minimize_w_restart(opt),
            false => self.minimize_wo_restart(opt),
        }
    }
}

#[cfg(test)]
mod minimize_neldermead_tests {
    use super::*;
    use crate::{
        file::read_touchstone,
        frequency::*,
        minimize::MultiDimFn,
        network::{Network, NetworkBuilder},
        num::{MyComplex, MyFloat},
        pts::Points3,
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

    fn calc_xfmr_z(
        freq: &Frequency,
        ls: &MyFloat,
        rs: &MyFloat,
        km: &MyFloat,
        qp: &MyFloat,
        x: &Points1<MyFloat>,
    ) -> Points3<MyComplex> {
        let mut z = Points3::<MyComplex>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut m = Points1::<MyFloat>::zeros(freq.npts());
        let mut n = Points1::<MyFloat>::zeros(freq.npts());
        let mut rp = Points1::<MyFloat>::zeros(freq.npts());
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
        x: &Points1<MyFloat>,
        ls: &MyFloat,
        rs: &MyFloat,
        km: &MyFloat,
        qp: &MyFloat,
        meas: &Network,
    ) -> MyFloat {
        let model_z = calc_xfmr_z(&meas.freq(), ls, rs, km, qp, x);
        let model_z_c64 = Points::<Complex64, Ix3>::from_shape_fn(model_z.dim(), |(i, j, k)| {
            model_z[[i, j, k]].clone().into()
        });
        let model = NetworkBuilder::new()
            .freq(meas.freq().clone())
            .z0(meas.z0().clone())
            .z(model_z_c64)
            .build();

        calc_err_xfmr(meas, &model)
    }

    fn calc_feed_y_f64(freq: Frequency, x: &Points1<f64>) -> Points3<Complex64> {
        let mut yfeed = Points3::<Complex64>::zeros((freq.npts(), 2, 2));

        let w = freq.w();
        let mut zs = Points1::<Complex64>::zeros(freq.npts());
        let mut zm = Points1::<Complex64>::zeros(freq.npts());
        let mut zp = Points1::<Complex64>::zeros(freq.npts());
        let mut zall = Points1::<Complex64>::zeros(freq.npts());
        for i in 0..freq.npts() {
            zs[i] = &x[3] * Complex64::new(0.0.into(), 1.0.into()) * w[i] * &x[2]
                / (&x[3] + Complex64::new(0.0.into(), 1.0.into()) * w[i] * &x[2]);
            zm[i] = &x[1] + Complex64::new(0.0.into(), 1.0.into()) * w[i] * &x[0] + &zs[i];
            zp[i] = &x[5] - Complex64::new(0.0.into(), 1.0.into()) / (w[i] * &x[4]);
            zall[i] = &zm[i] * &zp[i] / (&zm[i] + &zp[i]);
        }

        for i in 0..freq.npts() {
            yfeed[[i, 0, 1]] = -1.0 / &zall[i];
            yfeed[[i, 1, 0]] = -1.0 / &zall[i];
            yfeed[[i, 0, 0]] =
                Complex64::new(0.0.into(), 1.0.into()) * w[i] * &x[6] + 1.0 / &zall[i];
            yfeed[[i, 1, 1]] =
                Complex64::new(0.0.into(), 1.0.into()) * w[i] * &x[7] + 1.0 / &zall[i];
        }

        yfeed
    }

    fn calc_err_f64(model: &Network, meas: &Network) -> f64 {
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

    fn eval_f_simplex_f64(x: &Points1<f64>, meas: &Network) -> f64 {
        let model_y = calc_feed_y_f64(meas.freq().clone(), x);
        let model_y_c64 = Points::<Complex64, Ix3>::from_shape_fn(model_y.dim(), |(i, j, k)| {
            model_y[[i, j, k]].clone().into()
        });
        let model = NetworkBuilder::new()
            .freq(meas.freq().clone())
            .z0(meas.z0().clone())
            .y(model_y_c64)
            .build();

        calc_err_f64(&model, meas)
    }

    fn get_init() -> (
        Points1<MyFloat>,
        Points1<MyFloat>,
        Points1<MyFloat>,
        Points1<MyFloat>,
    ) {
        // (x, scale, lb, ub)
        (
            array![
                1e-11.into(),
                1e-3.into(),
                1e-13.into(),
                1e-6.into(),
                1e-15.into(),
                1000.0.into(),
                1e-15.into(),
                1e-15.into()
            ]
            .into(),
            array![
                1e12.into(),
                1.0.into(),
                1e12.into(),
                1.0.into(),
                1e15.into(),
                1.0.into(),
                1e15.into(),
                1e15.into()
            ]
            .into(),
            array![
                1e-15.into(),
                1e-6.into(),
                1e-15.into(),
                1e-6.into(),
                1e-15.into(),
                1.0.into(),
                1e-18.into(),
                1e-18.into()
            ]
            .into(),
            array![
                1e-9.into(),
                1.0.into(),
                1e-9.into(),
                1.0.into(),
                1e-12.into(),
                1e6.into(),
                1e-12.into(),
                1e-12.into()
            ]
            .into(),
        )
    }

    fn get_init_f64() -> (Points1<f64>, Points1<f64>, Points1<f64>, Points1<f64>) {
        (
            array![
                1e-11.into(),
                1e-3.into(),
                1e-13.into(),
                1e-6.into(),
                1e-15.into(),
                1000.0.into(),
                1e-15.into(),
                1e-15.into()
            ]
            .into(),
            array![
                1e12.into(),
                1.0.into(),
                1e12.into(),
                1.0.into(),
                1e15.into(),
                1.0.into(),
                1e15.into(),
                1e15.into()
            ]
            .into(),
            array![
                1e-15.into(),
                1e-6.into(),
                1e-15.into(),
                1e-6.into(),
                1e-15.into(),
                1.0.into(),
                1e-18.into(),
                1e-18.into()
            ]
            .into(),
            array![
                1e-9.into(),
                1.0.into(),
                1e-9.into(),
                1.0.into(),
                1e-12.into(),
                1e6.into(),
                1e-12.into(),
                1e-12.into()
            ]
            .into(),
        )
    }

    fn get_minimizer() -> NelderMead<MyFloat> {
        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let objective = MultiDimFn::new(move |x: &Points1<MyFloat>| eval_f_simplex(x, &net));
        NelderMead::new(objective)
    }

    fn get_minimizer_f64() -> NelderMead<f64> {
        let filename = "./data/1010_6x60um/feeds/drain_short.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let objective = MultiDimFn::new(move |x: &Points1<f64>| eval_f_simplex_f64(x, &net));
        NelderMead::new(objective)
    }

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    mod bounded_tests {
        use super::*;

        fn get_options(iters: usize) -> NelderMeadOptions<MyFloat> {
            let (x, scale, lb, ub) = get_init();
            NelderMeadOptions::new(
                &x,
                Some(&scale),
                Some(&lb),
                Some(&ub),
                Some(iters),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        }

        fn get_exemplar(iters: usize) -> (Points1<MyFloat>, MyFloat, Points1<MyFloat>) {
            // (x, tol, res)
            match iters {
                1 => (
                    array![
                        1.0883883476483184e-11.into(),
                        0.04519417382415922.into(),
                        2.7677669529663682e-13.into(),
                        0.04419517382415922.into(),
                        1.1767766952966369e-15.into(),
                        1000.0441941738242.into(),
                        1.1767766952966369e-15.into(),
                        1.1767766952966369e-15.into()
                    ]
                    .into(),
                    1.7892918605298145.into(),
                    array![
                        34009.378223556902.into(),
                        38464.430515104599.into(),
                        41299.962361391248.into(),
                        41306.789077727772.into(),
                        43347.207307893790.into(),
                        43459.577263575266.into(),
                        71891.470838700130.into(),
                        611611.20974369883.into(),
                        2345668.1203726004.into()
                    ]
                    .into(),
                ),
                2 => (
                    array![
                        1.0535854357617935e-11.into(),
                        1.0166666666666665e-6.into(),
                        6.3585435761793050e-13.into(),
                        5.3585535761793079e-1.into(),
                        1.5358543576179309e-15.into(),
                        1000.5358543576186.into(),
                        1.5358543576179309e-15.into(),
                        1.5358543576179318e-15.into()
                    ]
                    .into(),
                    1.3399527561782816.into(),
                    array![
                        6720.9921814088320.into(),
                        34009.378223556902.into(),
                        38464.430515104599.into(),
                        41299.962361391248.into(),
                        41306.789077727772.into(),
                        43347.207307893790.into(),
                        43459.577263575266.into(),
                        71891.470838700130.into(),
                        611611.20974369883.into()
                    ]
                    .into(),
                ),
                10 => (
                    array![
                        1.1076758244862255e-11.into(),
                        1.0071428571428570e-6.into(),
                        4.1119222513284393e-13.into(),
                        9.9285714285714288e-1.into(),
                        1.4766645329038703e-15.into(),
                        998.44720490232578.into(),
                        2.0767582448622552e-15.into(),
                        2.0767582448622560e-15.into()
                    ]
                    .into(),
                    4.4933775515054161e-2.into(),
                    array![
                        4588.5096445137724.into(),
                        4799.4273717325113.into(),
                        5637.4051964959672.into(),
                        5995.3329098586710.into(),
                        6040.3978068111992.into(),
                        6720.9921814088320.into(),
                        7536.7021767723654.into(),
                        8836.9181331446016.into(),
                        12445.285281662791.into()
                    ]
                    .into(),
                ),
                100 => (
                    array![
                        2.7357234954316440e-11.into(),
                        1.0068366344535046e-6.into(),
                        7.2309978663274626e-13.into(),
                        6.2667145218936882e-1.into(),
                        4.5558593814565677e-15.into(),
                        961.09716936676182.into(),
                        2.6284269578099862e-15.into(),
                        6.5265092075968652e-15.into()
                    ]
                    .into(),
                    1.2074437154540389e-4.into(),
                    array![
                        3421.8399332620584.into(),
                        3421.8519432289440.into(),
                        3422.2531261196850.into(),
                        3422.4517331839165.into(),
                        3422.4559518043652.into(),
                        3422.6615067154057.into(),
                        3422.7510632163799.into(),
                        3422.9347906078015.into(),
                        3422.9487717942866.into()
                    ]
                    .into(),
                ),
                _ => (Points1::zeros(8), 0.0.into(), Points1::zeros(8)),
            }
        }

        fn get_options_f64(iters: usize) -> NelderMeadOptions<f64> {
            let (x, scale, lb, ub) = get_init_f64();
            NelderMeadOptions::new(
                &x,
                Some(&scale),
                Some(&lb),
                Some(&ub),
                Some(iters),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        }

        fn get_exemplar_f64(iters: usize) -> (Points1<f64>, f64, Points1<f64>) {
            // (x, tol, res)
            match iters {
                1 => (
                    array![
                        1.0883883476483184e-11.into(),
                        0.04519417382415922.into(),
                        2.7677669529663682e-13.into(),
                        0.04419517382415922.into(),
                        1.1767766952966369e-15.into(),
                        1000.0441941738242.into(),
                        1.1767766952966369e-15.into(),
                        1.1767766952966369e-15.into()
                    ]
                    .into(),
                    1.7892918605298145.into(),
                    array![
                        34009.378223556902.into(),
                        38464.430515104599.into(),
                        41299.962361391248.into(),
                        41306.789077727772.into(),
                        43347.207307893790.into(),
                        43459.577263575266.into(),
                        71891.470838700130.into(),
                        611611.20974369883.into(),
                        2345668.1203726004.into()
                    ]
                    .into(),
                ),
                2 => (
                    array![
                        1.0535854357617935e-11.into(),
                        1.0166666666666665e-6.into(),
                        6.3585435761793050e-13.into(),
                        5.3585535761793079e-1.into(),
                        1.5358543576179309e-15.into(),
                        1000.5358543576186.into(),
                        1.5358543576179309e-15.into(),
                        1.5358543576179318e-15.into()
                    ]
                    .into(),
                    1.3399527561782816.into(),
                    array![
                        6720.9921814088320.into(),
                        34009.378223556902.into(),
                        38464.430515104599.into(),
                        41299.962361391248.into(),
                        41306.789077727772.into(),
                        43347.207307893790.into(),
                        43459.577263575266.into(),
                        71891.470838700130.into(),
                        611611.20974369883.into()
                    ]
                    .into(),
                ),
                10 => (
                    array![
                        1.1076758244862255e-11.into(),
                        1.0071428571428570e-6.into(),
                        4.1119222513284393e-13.into(),
                        9.9285714285714288e-1.into(),
                        1.4766645329038703e-15.into(),
                        998.44720490232578.into(),
                        2.0767582448622552e-15.into(),
                        2.0767582448622560e-15.into()
                    ]
                    .into(),
                    4.4933775515054161e-2.into(),
                    array![
                        4588.5096445137724.into(),
                        4799.4273717325113.into(),
                        5637.4051964959672.into(),
                        5995.3329098586710.into(),
                        6040.3978068111992.into(),
                        6720.9921814088320.into(),
                        7536.7021767723654.into(),
                        8836.9181331446016.into(),
                        12445.285281662791.into()
                    ]
                    .into(),
                ),
                100 => (
                    array![
                        2.7357234954316440e-11.into(),
                        1.0068366344535046e-6.into(),
                        7.2309978663274626e-13.into(),
                        6.2667145218936882e-1.into(),
                        4.5558593814565677e-15.into(),
                        961.09716936676182.into(),
                        2.6284269578099862e-15.into(),
                        6.5265092075968652e-15.into()
                    ]
                    .into(),
                    1.2074437154540389e-4.into(),
                    array![
                        3421.8399332620584.into(),
                        3421.8519432289440.into(),
                        3422.2531261196850.into(),
                        3422.4517331839165.into(),
                        3422.4559518043652.into(),
                        3422.6615067154057.into(),
                        3422.7510632163799.into(),
                        3422.9347906078015.into(),
                        3422.9487717942866.into()
                    ]
                    .into(),
                ),
                _ => (Points1::zeros(8), 0.0.into(), Points1::zeros(8)),
            }
        }

        #[test]
        fn neldermead_bounded_iter1_() {
            let iters = 1;
            let (exemplar_x, exemplar_tol, exemplar_res) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_pts_ix1(&exemplar_res, &test.res, margin, "minimize(res_all)");
            comp_num(&exemplar_res[0], &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_bounded_iter2_() {
            let iters = 2;
            let (exemplar_x, exemplar_tol, exemplar_res) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_pts_ix1(&exemplar_res, &test.res, margin, "minimize(res_all)");
            comp_num(&exemplar_res[0], &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_bounded_iter10_() {
            let iters = 10;
            let (exemplar_x, exemplar_tol, exemplar_res) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_pts_ix1(&exemplar_res, &test.res, margin, "minimize(res_all)");
            comp_num(&exemplar_res[0], &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_bounded_iter100_() {
            let iters = 100;
            let (exemplar_x, exemplar_tol, exemplar_res) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_pts_ix1(&exemplar_res, &test.res, margin, "minimize(res_all)");
            comp_num(&exemplar_res[0], &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_bounded_restart_iter100() {
            let iters = 100;
            let (_, _, exemplar_res) = get_exemplar(iters);
            let mut options = get_options(iters);
            options.set_random_restart(true);
            options.set_adaptive_simplex(true);
            options.enable_anti_stagnation(Some(3), Some(10), Some(0.1.into()));
            options.set_verbosity(1);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!(
                "\n\nf(x) old:\t{}\nf(x) new:\t{}\n\n",
                exemplar_res[0],
                test.fmin(),
            );
            assert!(
                test.fmin() < exemplar_res[0],
                "optimization should be better than standard"
            );
        }
    }

    mod unbounded_tests {
        use super::*;

        fn get_options(iters: usize) -> NelderMeadOptions<MyFloat> {
            let (x, scale, _, _) = get_init();
            NelderMeadOptions::new(
                &x,
                Some(&scale),
                None,
                None,
                Some(iters),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        }

        fn get_exemplar(iters: usize) -> (Points1<MyFloat>, MyFloat, MyFloat, Points2<MyFloat>) {
            // (x, tol, res[0], simplex)
            match iters {
                1 => (
                    array![
                        1.0e-11.into(),
                        1.0e-03.into(),
                        1.0e-13.into(),
                        1.0e-06.into(),
                        1.0e-15.into(),
                        1.0e03.into(),
                        1.0e-15.into(),
                        1.0e-15.into()
                    ]
                    .into(),
                    1.9726210586239914.into(),
                    5736.50737797145.into(),
                    array![
                        [
                            1.0000000000000000e+01.into(),
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
                            (-0.33874271127322403).into(),
                            0.27677669529663684.into(),
                            0.1767776952966369.into(),
                            1.176776695296637.into(),
                            1000.1767766952967.into(),
                            1.176776695296637.into(),
                            1.1767766952966372.into()
                        ],
                    ]
                    .into(),
                ),
                2 => (
                    array![
                        1.0e-11.into(),
                        1.0e-03.into(),
                        1.0e-13.into(),
                        1.0e-06.into(),
                        1.0e-15.into(),
                        1.0e03.into(),
                        1.0e-15.into(),
                        1.0e-15.into()
                    ]
                    .into(),
                    1.9150926772195656.into(),
                    5736.50737797145.into(),
                    Points2::zeros((9, 8)),
                ),
                5 => (
                    array![
                        1.0e-11.into(),
                        1.0e-03.into(),
                        1.0e-13.into(),
                        1.0e-06.into(),
                        1.0e-15.into(),
                        1.0e03.into(),
                        1.0e-15.into(),
                        1.0e-15.into()
                    ]
                    .into(),
                    2.0864550350295435e-1.into(),
                    5736.50737797145.into(),
                    Points2::zeros((9, 8)),
                ),
                6 => (
                    array![
                        1.0298008564702946e-11.into(),
                        1.1355312091732621e-2.into(),
                        (-5.1011146690169039e-14).into(),
                        2.9800956470294870e-1.into(),
                        3.5888237093956543e-16.into(),
                        1000.2980085647031.into(),
                        1.2980085647029487e-15.into(),
                        1.2980085647029487e-15.into()
                    ]
                    .into(),
                    1.7834451311651800e-1.into(),
                    4797.1934994922385.into(),
                    Points2::zeros((9, 8)),
                ),
                10 => (
                    array![
                        1.0496148654572788e-11.into(),
                        0.0028955966586259665.into(),
                        (-2.5331914829451519e-14).into(),
                        0.36146316242264603.into(),
                        9.9896419905099658e-16.into(),
                        999.4223359686597.into(),
                        1.4961486545727874e-15.into(),
                        1.4961486545727874e-15.into()
                    ]
                    .into(),
                    1.2031353717842206.into(),
                    3267.394011570648.into(),
                    Points2::zeros((9, 8)),
                ),
                100 => (
                    array![
                        1.6110247811984210e-11.into(),
                        0.010202461287747289.into(),
                        1.2618495970192951e-12.into(),
                        6.410718597400084.into(),
                        (-2.1119233982250064e-16).into(),
                        991.264467681222.into(),
                        4.5268760596139400e-15.into(),
                        9.1487688225672705e-15.into()
                    ]
                    .into(),
                    6.9941144516836251e-2.into(),
                    765.6358932973524.into(),
                    Points2::zeros((9, 8)),
                ),
                _ => (
                    Points1::zeros(8),
                    0.0.into(),
                    0.0.into(),
                    Points2::zeros((9, 8)),
                ),
            }
        }

        fn get_options_f64(iters: usize) -> NelderMeadOptions<f64> {
            let (x, scale, _, _) = get_init_f64();
            NelderMeadOptions::new(
                &x,
                Some(&scale),
                None,
                None,
                Some(iters),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        }

        fn get_exemplar_f64(iters: usize) -> (Points1<f64>, f64, f64, Points2<f64>) {
            // (x, tol, res)
            match iters {
                1 => (
                    array![
                        1.0e-11.into(),
                        1.0e-03.into(),
                        1.0e-13.into(),
                        1.0e-06.into(),
                        1.0e-15.into(),
                        1.0e03.into(),
                        1.0e-15.into(),
                        1.0e-15.into()
                    ]
                    .into(),
                    1.9726210586239914.into(),
                    5736.50737797145.into(),
                    array![
                        [
                            1.0000000000000000e+01.into(),
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
                            (-0.33874271127322403).into(),
                            0.27677669529663684.into(),
                            0.1767776952966369.into(),
                            1.176776695296637.into(),
                            1000.1767766952967.into(),
                            1.176776695296637.into(),
                            1.1767766952966372.into()
                        ],
                    ]
                    .into(),
                ),
                2 => (
                    array![
                        1.0e-11.into(),
                        1.0e-03.into(),
                        1.0e-13.into(),
                        1.0e-06.into(),
                        1.0e-15.into(),
                        1.0e03.into(),
                        1.0e-15.into(),
                        1.0e-15.into()
                    ]
                    .into(),
                    1.9150926772195656.into(),
                    5736.50737797145.into(),
                    Points2::zeros((9, 8)),
                ),
                5 => (
                    array![
                        1.0e-11.into(),
                        1.0e-03.into(),
                        1.0e-13.into(),
                        1.0e-06.into(),
                        1.0e-15.into(),
                        1.0e03.into(),
                        1.0e-15.into(),
                        1.0e-15.into()
                    ]
                    .into(),
                    2.0864550350295435e-1.into(),
                    5736.50737797145.into(),
                    Points2::zeros((9, 8)),
                ),
                6 => (
                    array![
                        1.0298008564702946e-11.into(),
                        1.1355312091732621e-2.into(),
                        (-5.1011146690169039e-14).into(),
                        2.9800956470294870e-1.into(),
                        3.5888237093956543e-16.into(),
                        1000.2980085647031.into(),
                        1.2980085647029487e-15.into(),
                        1.2980085647029487e-15.into()
                    ]
                    .into(),
                    1.7834451311651800e-1.into(),
                    4797.1934994922385.into(),
                    Points2::zeros((9, 8)),
                ),
                10 => (
                    array![
                        1.0496148654572788e-11.into(),
                        0.0028955966586259665.into(),
                        (-2.5331914829451519e-14).into(),
                        0.36146316242264603.into(),
                        9.9896419905099658e-16.into(),
                        999.4223359686597.into(),
                        1.4961486545727874e-15.into(),
                        1.4961486545727874e-15.into()
                    ]
                    .into(),
                    1.2031353717842206.into(),
                    3267.394011570648.into(),
                    Points2::zeros((9, 8)),
                ),
                100 => (
                    array![
                        1.6110247811984210e-11.into(),
                        0.010202461287747289.into(),
                        1.2618495970192951e-12.into(),
                        6.410718597400084.into(),
                        (-2.1119233982250064e-16).into(),
                        991.264467681222.into(),
                        4.5268760596139400e-15.into(),
                        9.1487688225672705e-15.into()
                    ]
                    .into(),
                    6.9941144516836251e-2.into(),
                    765.6358932973524.into(),
                    Points2::zeros((9, 8)),
                ),
                _ => (
                    Points1::zeros(8),
                    0.0.into(),
                    0.0.into(),
                    Points2::zeros((9, 8)),
                ),
            }
        }

        #[test]
        fn neldermead_unbounded_iter1_() {
            let iters = 1;
            let (exemplar_x, exemplar_tol, exemplar_res, exemplar_simplex) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix2(
                &exemplar_simplex,
                &test.simplex,
                margin,
                "minimize(simplex)",
            );
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_num(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_unbounded_iter2_() {
            let iters = 2;
            let (exemplar_x, exemplar_tol, exemplar_res, _) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_num(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_unbounded_iter5_() {
            let iters = 5;
            let (exemplar_x, exemplar_tol, exemplar_res, _) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_num(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_unbounded_iter6_() {
            let iters = 6;
            let (exemplar_x, exemplar_tol, exemplar_res, _) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_num(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_unbounded_iter10_() {
            let iters = 10;
            let (exemplar_x, exemplar_tol, exemplar_res, _) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_num(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_unbounded_iter100_() {
            let iters = 100;
            let (exemplar_x, exemplar_tol, exemplar_res, _) = get_exemplar(iters);
            let options = get_options(iters);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!("\n\nres: {:?}\nx: {:?}\n\n", test.res, test.xmin());
            let margin = MARGIN;
            comp_pts_ix1(&exemplar_x, &test.xmin(), margin, "minimize(x)");
            comp_num(&exemplar_res, &test.fmin(), margin, "minimize(res)", "");
            comp_num(
                &exemplar_tol,
                &test.tolerance(),
                margin,
                "minimize(tol)",
                "",
            );
        }

        #[test]
        fn neldermead_unbounded_restart_iter100_() {
            let iters = 100;
            let (_, _, exemplar_res, _) = get_exemplar(iters);
            let mut options = get_options(iters);
            options.set_random_restart(true);
            options.set_adaptive_simplex(true);
            options.enable_anti_stagnation(Some(3), Some(10), Some(0.1.into()));
            options.set_verbosity(1);
            let mut minimizer = get_minimizer();
            let test = minimizer.minimize(&options).unwrap();

            println!(
                "\n\nf(x) old:\t{}\nf(x) new:\t{}\n\n",
                exemplar_res,
                test.fmin(),
            );
            assert!(
                test.fmin() < exemplar_res,
                "optimization should be better than standard"
            );
        }
    }
}
