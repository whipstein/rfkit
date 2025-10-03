#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::{
    MinimizerError,
    f64::{ObjFn, outer},
};
use ndarray::prelude::*;
use ndarray_linalg::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct CmaRandom {
    rng: ChaCha8Rng,
    stored_value: Option<f64>,
}

impl CmaRandom {
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_os_rng(),
        };

        Self {
            rng,
            stored_value: None,
        }
    }

    pub fn uniform(&mut self) -> f64 {
        self.rng.random()
    }

    pub fn gauss(&mut self) -> f64 {
        if let Some(stored) = self.stored_value.take() {
            return stored;
        }

        loop {
            let x1 = 2.0 * self.uniform() - 1.0;
            let x2 = 2.0 * self.uniform() - 1.0;
            let r_squared = x1 * x1 + x2 * x2;

            if r_squared < 1.0 && r_squared > 0.0 {
                let fac = (-2.0 * r_squared.ln() / r_squared).sqrt();
                self.stored_value = Some(fac * x1);
                return fac * x2;
            }
        }
    }
}

#[derive(Clone)]
pub struct Timings {
    start_time: Instant,
    total_time: Duration,
    tic_time: Option<Instant>,
    tictoc_time: Duration,
    last_tictoc_time: Duration,
}

impl Timings {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_time: Duration::ZERO,
            tic_time: None,
            tictoc_time: Duration::ZERO,
            last_tictoc_time: Duration::ZERO,
        }
    }

    pub fn update(&mut self) -> Duration {
        let elapsed = self.start_time.elapsed();
        let diff = elapsed - self.total_time;
        self.total_time = elapsed;
        diff
    }

    pub fn tic(&mut self) {
        self.update();
        self.tic_time = Some(Instant::now());
    }

    pub fn toc(&mut self) -> Duration {
        if let Some(tic_start) = self.tic_time.take() {
            let elapsed = tic_start.elapsed();
            self.last_tictoc_time = elapsed;
            self.tictoc_time += elapsed;
            elapsed
        } else {
            Duration::ZERO
        }
    }

    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    pub fn tictoc_time(&self) -> Duration {
        self.tictoc_time
    }
}

#[derive(Clone)]
pub struct StopCondition {
    pub enabled: bool,
    pub value: f64,
}

impl Default for StopCondition {
    fn default() -> Self {
        Self {
            enabled: false,
            value: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct Parameters {
    pub dimension: usize,
    pub lambda: usize,
    pub mu: usize,
    pub weights: Array1<f64>,
    pub mu_eff: f64,
    pub cs: f64,
    pub ccov: f64,
    pub ccumcov: f64,
    pub damps: f64,
    pub mucov: f64,
    pub diagonal_cov: f64,

    // Stop conditions
    pub stop_fitness: StopCondition,
    pub stop_tol_fun: f64,
    pub stop_tol_fun_hist: f64,
    pub stop_tol_x: f64,
    pub stop_tol_upx_factor: f64,
    pub stop_max_fun_evals: usize,
    pub stop_max_iter: usize,

    // Initial values
    pub initial_x: Array1<f64>,
    pub initial_std_devs: Array1<f64>,
    pub seed: Option<u64>,
}

impl Parameters {
    pub fn new(dimension: usize) -> Result<Self, MinimizerError> {
        if dimension == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let lambda = 4 + (3.0 * (dimension as f64).ln()) as usize;
        let mu = lambda / 2;

        let mut params = Self {
            dimension,
            lambda,
            mu,
            weights: array![],
            mu_eff: 0.0,
            cs: 0.0,
            ccov: 0.0,
            ccumcov: 0.0,
            damps: 0.0,
            mucov: 0.0,
            diagonal_cov: 0.0,
            stop_fitness: StopCondition::default(),
            stop_tol_fun: 1e-12,
            stop_tol_fun_hist: 1e-13,
            stop_tol_x: 0.0,
            stop_tol_upx_factor: 1e3,
            stop_max_fun_evals: 900 * (dimension + 3).pow(2),
            stop_max_iter: 0,
            initial_x: Array1::ones(dimension) * 0.5,
            initial_std_devs: Array1::ones(dimension) * 0.3,
            seed: None,
        };

        params.set_weights("log")?;
        params.supplement_defaults()?;

        Ok(params)
    }

    pub fn with_initial_x(mut self, x: Array1<f64>) -> Result<Self, MinimizerError> {
        if x.len() != self.dimension {
            return Err(MinimizerError::InvalidParameters(
                "Initial x dimension mismatch".to_string(),
            ));
        }
        self.initial_x = x;
        Ok(self)
    }

    pub fn with_initial_std_devs(mut self, std_devs: Array1<f64>) -> Result<Self, MinimizerError> {
        if std_devs.len() != self.dimension {
            return Err(MinimizerError::InvalidParameters(
                "Initial std devs dimension mismatch".to_string(),
            ));
        }
        self.initial_std_devs = std_devs;
        Ok(self)
    }

    pub fn with_lambda(mut self, lambda: usize) -> Self {
        self.lambda = lambda;
        self.mu = lambda / 2;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn set_tol(&mut self, tol: f64) {
        self.stop_tol_x = tol;
        self.stop_tol_fun = tol;
    }

    fn set_weights(&mut self, mode: &str) -> Result<(), MinimizerError> {
        self.weights = Array1::zeros(self.mu);

        match mode {
            "lin" => {
                for i in 0..self.mu {
                    self.weights[i] = (self.mu - i) as f64;
                }
            }
            "equal" => {
                for i in 0..self.mu {
                    self.weights[i] = 1.0;
                }
            }
            "log" => {
                for i in 0..self.mu {
                    self.weights[i] = ((self.mu + 1) as f64).ln() - ((i + 1) as f64).ln();
                }
            }
            _ => {
                return Err(MinimizerError::InvalidParameters(format!(
                    "Unknown weight mode: {}",
                    mode
                )));
            }
        }

        // Normalize weights and calculate mu_eff
        let sum: f64 = self.weights.iter().sum();
        let sum_sq: f64 = self.weights.iter().map(|w| w * w).sum();

        self.mu_eff = sum * sum / sum_sq;

        for weight in &mut self.weights {
            *weight /= sum;
        }

        Ok(())
    }

    fn supplement_defaults(&mut self) -> Result<(), MinimizerError> {
        let n = self.dimension as f64;

        // Set cs (cumulation for sigma)
        if self.cs <= 0.0 {
            self.cs = (self.mu_eff + 2.0) / (n + self.mu_eff + 3.0);
        }

        // Set ccumcov (cumulation for covariance matrix)
        if self.ccumcov <= 0.0 {
            self.ccumcov = 4.0 / (n + 4.0);
        }

        // Set mucov
        if self.mucov <= 0.0 {
            self.mucov = self.mu_eff;
        }

        // Set ccov (learning rate for covariance matrix)
        if self.ccov <= 0.0 {
            let t1 = 2.0 / ((n + 1.4142) * (n + 1.4142));
            let mut t2 = (2.0 * self.mu_eff - 1.0) / ((n + 2.0) * (n + 2.0) + self.mu_eff);
            t2 = t2.min(1.0);
            self.ccov = (1.0 / self.mucov) * t1 + (1.0 - 1.0 / self.mucov) * t2;
        }

        // Set damps (damping for sigma)
        if self.damps <= 0.0 {
            let max_eval = if self.stop_max_iter > 0 {
                self.stop_max_iter * self.lambda
            } else {
                self.stop_max_fun_evals
            };

            self.damps = 1.0 + 2.0 * (0.0_f64).max(((self.mu_eff - 1.0) / (n + 1.0)).sqrt() - 1.0);
            self.damps *= (0.3_f64).max(
                1.0 - n
                    / (1e-6
                        + (max_eval as f64 / self.lambda as f64).min(self.stop_max_iter as f64)),
            );
            self.damps += self.cs;
        }

        // Set diagonal_cov
        if self.diagonal_cov <= 0.0 {
            self.diagonal_cov = 2.0 + 100.0 * n / (self.lambda as f64).sqrt();
        }

        // Set stop_max_iter if not set
        if self.stop_max_iter == 0 {
            self.stop_max_iter =
                (self.stop_max_fun_evals as f64 / self.lambda as f64).ceil() as usize;
        }

        Ok(())
    }
}

pub struct BoundaryTransformation {
    lower_bounds: Array1<f64>,
    upper_bounds: Array1<f64>,
    al: Array1<f64>, // lower boundary pre-image add-on
    au: Array1<f64>, // upper boundary pre-image add-on
}

impl BoundaryTransformation {
    pub fn new(
        lower_bounds: Array1<f64>,
        upper_bounds: Array1<f64>,
    ) -> Result<Self, MinimizerError> {
        if lower_bounds.len() != upper_bounds.len() {
            return Err(MinimizerError::InvalidParameters(
                "Bounds vectors must have same length".to_string(),
            ));
        }

        let len = lower_bounds.len();
        let mut al = Array1::zeros(len);
        let mut au = Array1::zeros(len);

        for i in 0..len {
            if lower_bounds[i] >= upper_bounds[i] {
                return Err(MinimizerError::InvalidParameters(
                    "Lower bound must be less than upper bound".to_string(),
                ));
            }

            let range = upper_bounds[i] - lower_bounds[i];
            al[i] = (range / 2.0).min((1.0 + lower_bounds[i].abs()) / 20.0);
            au[i] = (range / 2.0).min((1.0 + upper_bounds[i].abs()) / 20.0);
        }

        Ok(Self {
            lower_bounds,
            upper_bounds,
            al,
            au,
        })
    }

    pub fn transform(&self, x: &Array1<f64>, y: &mut Array1<f64>) -> Result<(), MinimizerError> {
        if x.len() != y.len() || x.len() != self.lower_bounds.len() {
            return Err(MinimizerError::InvalidParameters(
                "Vector length mismatch".to_string(),
            ));
        }

        // First apply shift into feasible pre-image
        self.shift_into_feasible_preimage(x, y)?;

        // Then apply boundary transformation
        for i in 0..x.len() {
            let lb = self.lower_bounds[i];
            let ub = self.upper_bounds[i];
            let al = self.al[i];
            let au = self.au[i];

            if y[i] < lb + al {
                let diff = y[i] - (lb - al);
                y[i] = lb + diff * diff / (4.0 * al);
            } else if y[i] > ub - au {
                let diff = y[i] - (ub + au);
                y[i] = ub - diff * diff / (4.0 * au);
            }
        }

        Ok(())
    }

    fn shift_into_feasible_preimage(
        &self,
        x: &Array1<f64>,
        y: &mut Array1<f64>,
    ) -> Result<(), MinimizerError> {
        for i in 0..x.len() {
            let lb = self.lower_bounds[i];
            let ub = self.upper_bounds[i];
            let al = self.al[i];
            let au = self.au[i];

            let xlow = lb - 2.0 * al - (ub - lb) / 2.0;
            let xup = ub + 2.0 * au + (ub - lb) / 2.0;
            let r = 2.0 * (ub - lb + al + au); // period

            y[i] = x[i];

            // Shift into range
            if y[i] < xlow {
                y[i] += r * (1.0 + ((xlow - y[i]) / r).floor());
            }
            if y[i] > xup {
                y[i] -= r * (1.0 + ((y[i] - xup) / r).floor());
            }

            // Mirror if necessary
            if y[i] < lb - al {
                y[i] += 2.0 * (lb - al - y[i]);
            }
            if y[i] > ub + au {
                y[i] -= 2.0 * (y[i] - ub - au);
            }
        }

        Ok(())
    }

    pub fn inverse_transform(
        &self,
        y: &Array1<f64>,
        x: &mut Array1<f64>,
    ) -> Result<(), MinimizerError> {
        if x.len() != y.len() || x.len() != self.lower_bounds.len() {
            return Err(MinimizerError::InvalidParameters(
                "Vector length mismatch".to_string(),
            ));
        }

        for i in 0..y.len() {
            let lb = self.lower_bounds[i];
            let ub = self.upper_bounds[i];
            let al = self.al[i];
            let au = self.au[i];

            x[i] = y[i];

            if y[i] < lb + al {
                x[i] = (lb - al) + 2.0 * (al * (y[i] - lb)).sqrt();
            } else if y[i] > ub - au {
                x[i] = (ub + au) - 2.0 * (au * (ub - y[i])).sqrt();
            }
        }

        Ok(())
    }
}

/// Result of CMA-ES optimization
#[derive(Debug, Clone)]
pub struct CmaEsResult {
    pub xmin: Array1<f64>,
    pub fmin: f64,
    pub generations: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_sigma: f64,
    pub condition_number: f64,
    pub history: Vec<f64>,
}

#[derive(Clone)]
pub struct CmaEs {
    f: Box<dyn ObjFn>,
    params: Parameters,
    random: CmaRandom,
    timings: Timings,

    // State variables
    generation: usize,
    count_evals: usize,
    sigma: f64,

    // Mean and population
    x_mean: Array1<f64>,
    x_old: Array1<f64>,
    x_best_ever: Array1<f64>,
    f_best_ever: f64,
    population: Vec<Array1<f64>>,
    fitness_values: Vec<f64>,

    // Evolution paths
    pc: Array1<f64>, // path for C
    ps: Array1<f64>, // path for sigma

    // Covariance matrix and eigendecomposition
    c_matrix: Array2<f64>,
    b_matrix: Array2<f64>, // eigenvectors
    d_vector: Array1<f64>, // sqrt of eigenvalues
    eigen_update_generation: usize,
    eigen_is_uptodate: bool,

    // Auxiliary
    chi_n: f64,
    index: Vec<usize>,
    stop_message: Option<String>,

    // History for termination
    fitness_history: Vec<f64>,
    max_history_length: usize,

    converged: bool,
}

impl CmaEs {
    pub fn new<F>(f: F, params: Parameters) -> Result<Self, MinimizerError>
    where
        F: ObjFn + 'static,
    {
        CmaEs::new_boxed(Box::new(f), params)
    }

    pub fn new_boxed(f: Box<dyn ObjFn>, params: Parameters) -> Result<Self, MinimizerError> {
        let dimension = params.dimension;
        let lambda = params.lambda;

        let random = CmaRandom::new(params.seed);

        // Initialize mean
        let x_mean = params.initial_x.clone();

        // Calculate initial sigma
        let trace: f64 = params.initial_std_devs.iter().map(|&s| s * s).sum();
        let sigma = (trace / dimension as f64).sqrt();

        // Initialize covariance matrix (diagonal)
        let mut c_matrix = Array2::zeros((dimension, dimension));
        let mut d_vector = Array1::zeros(dimension);

        for i in 0..dimension {
            let std_normalized = params.initial_std_devs[i] * (dimension as f64 / trace).sqrt();
            c_matrix[[i, i]] = std_normalized * std_normalized;
            d_vector[i] = std_normalized;
        }

        // Initialize B as identity
        let b_matrix = Array2::eye(dimension);

        // Initialize evolution paths
        let pc = Array1::zeros(dimension);
        let ps = Array1::zeros(dimension);

        // Initialize population
        let population = vec![Array1::zeros(dimension); lambda];
        let fitness_values = vec![0.0; lambda];

        // Initialize other variables
        let chi_n = (dimension as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * dimension as f64)
                + 1.0 / (21.0 * dimension as f64 * dimension as f64));

        let index = (0..lambda).collect();

        let max_history_length =
            10 + (3.0 * 10.0 * dimension as f64 / lambda as f64).ceil() as usize;

        Ok(Self {
            f,
            params,
            random,
            timings: Timings::new(),
            generation: 0,
            count_evals: 0,
            sigma,
            x_mean: x_mean.clone(),
            x_old: x_mean.clone(),
            x_best_ever: x_mean,
            f_best_ever: f64::INFINITY,
            population,
            fitness_values,
            pc,
            ps,
            c_matrix,
            b_matrix,
            d_vector,
            eigen_update_generation: 0,
            eigen_is_uptodate: true,
            chi_n,
            index,
            stop_message: None,
            fitness_history: Vec::new(),
            max_history_length,
            converged: false,
        })
    }

    pub fn sample_population(&mut self) -> Result<&[Array1<f64>], MinimizerError> {
        self.update_eigensystem(false)?;

        for i in 0..self.params.lambda {
            // Generate random vector
            let mut z = Array1::zeros(self.params.dimension);
            for j in 0..self.params.dimension {
                z[j] = self.random.gauss();
            }

            // Scale by D
            z *= &self.d_vector;

            // Transform by B and add to mean
            let bz = self.b_matrix.dot(&z);
            self.population[i] = &self.x_mean + self.sigma * &bz;
        }

        self.generation += 1;
        Ok(&self.population)
    }

    pub fn update_distribution(&mut self, fitness_values: &[f64]) -> Result<(), MinimizerError> {
        if fitness_values.len() != self.params.lambda {
            return Err(MinimizerError::InvalidParameters(
                "Fitness values length mismatch".to_string(),
            ));
        }

        self.fitness_values.copy_from_slice(fitness_values);
        self.count_evals += self.params.lambda;

        // Sort population by fitness
        self.index
            .sort_by(|&a, &b| fitness_values[a].partial_cmp(&fitness_values[b]).unwrap());

        // Update fitness history
        self.fitness_history
            .insert(0, fitness_values[self.index[0]]);
        if self.fitness_history.len() > self.max_history_length {
            self.fitness_history.truncate(self.max_history_length);
        }

        // Update best ever
        if fitness_values[self.index[0]] < self.f_best_ever {
            self.f_best_ever = fitness_values[self.index[0]];
            self.x_best_ever = self.population[self.index[0]].clone();
        }

        // Calculate new mean
        self.x_old = self.x_mean.clone();
        self.x_mean.fill(0.0);

        for i in 0..self.params.mu {
            let idx = self.index[i];
            self.x_mean += &(self.params.weights[i] * &self.population[idx]);
        }

        // Update evolution paths and covariance matrix
        self.update_evolution_paths()?;
        self.update_covariance_matrix()?;
        self.update_step_size()?;

        Ok(())
    }

    fn update_evolution_paths(&mut self) -> Result<(), MinimizerError> {
        let bdz = (self.params.mu_eff.sqrt() / self.sigma) * (&self.x_mean - &self.x_old);

        // Calculate z = D^(-1) * B^T * bdz
        let bt_bdz = self.b_matrix.t().dot(&bdz);
        let mut z = Array1::zeros(self.params.dimension);
        for i in 0..self.params.dimension {
            z[i] = bt_bdz[i] / self.d_vector[i];
        }

        // Update ps (path for sigma)
        let b_z = self.b_matrix.dot(&z);
        self.ps = (1.0 - self.params.cs) * &self.ps
            + (self.params.cs * (2.0 - self.params.cs)).sqrt() * &b_z;

        // Calculate hsig
        let ps_norm_sq = self.ps.dot(&self.ps);
        let expected_norm = 1.0 - (1.0 - self.params.cs).powi(2 * self.generation as i32);
        let hsig = ps_norm_sq.sqrt() / expected_norm.sqrt() / self.chi_n
            < 1.4 + 2.0 / (self.params.dimension as f64 + 1.0);

        // Update pc (path for covariance)
        let hsig_factor = if hsig { 1.0 } else { 0.0 };
        self.pc = (1.0 - self.params.ccumcov) * &self.pc
            + hsig_factor * (self.params.ccumcov * (2.0 - self.params.ccumcov)).sqrt() * &bdz;

        Ok(())
    }

    fn update_covariance_matrix(&mut self) -> Result<(), MinimizerError> {
        if self.params.ccov == 0.0 {
            return Ok(());
        }

        let ccov1 = self.params.ccov * (1.0 / self.params.mucov);
        let ccovmu = self.params.ccov * (1.0 - 1.0 / self.params.mucov);

        // Rank-one update
        let pc_outer = outer(&self.pc, &self.pc);

        // Rank-mu update
        let mut rank_mu_update = Array2::zeros((self.params.dimension, self.params.dimension));
        let sigma_sq = self.sigma * self.sigma;

        for i in 0..self.params.mu {
            let idx = self.index[i];
            let diff = &self.population[idx] - &self.x_old;
            let outer = outer(&diff, &diff);
            rank_mu_update += &(self.params.weights[i] * &outer / sigma_sq);
        }

        // Update C
        self.c_matrix =
            (1.0 - ccov1 - ccovmu) * &self.c_matrix + ccov1 * &pc_outer + ccovmu * &rank_mu_update;

        self.eigen_is_uptodate = false;

        Ok(())
    }

    fn update_step_size(&mut self) -> Result<(), MinimizerError> {
        let ps_norm_sq = self.ps.dot(&self.ps);
        let factor = (ps_norm_sq.sqrt() / self.chi_n - 1.0) * self.params.cs / self.params.damps;
        self.sigma *= factor.exp();

        Ok(())
    }

    fn update_eigensystem(&mut self, force: bool) -> Result<(), MinimizerError> {
        if self.eigen_is_uptodate && !force {
            return Ok(());
        }

        self.timings.tic();

        // Eigendecomposition of C - use symmetric eigenvalue decomposition
        let (eigenvals, eigenvecs) = self.c_matrix.eig().map_err(|e| {
            MinimizerError::LinearAlgebraError(format!("Eigendecomposition failed: {:?}", e))
        })?;

        // Extract real parts (C is symmetric, so eigenvalues should be real)
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvals
            .iter()
            .zip(eigenvecs.axis_iter(Axis(1)))
            .map(|(val, vec)| (val.re, vec.iter().map(|x| x.re).collect()))
            .collect();

        // Sort eigenvalues and eigenvectors in ascending order
        eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Update D and B
        for (i, (eigenval, eigenvec)) in eigen_pairs.iter().enumerate() {
            if *eigenval <= 0.0 {
                return Err(MinimizerError::NumericalError(
                    "Non-positive eigenvalue encountered".to_string(),
                ));
            }
            self.d_vector[i] = eigenval.sqrt();
            for j in 0..self.params.dimension {
                self.b_matrix[[j, i]] = eigenvec[j];
            }
        }

        self.eigen_is_uptodate = true;
        self.eigen_update_generation = self.generation;

        self.timings.toc();

        Ok(())
    }

    pub fn test_for_termination(&mut self) -> Option<&str> {
        if let Some(ref msg) = self.stop_message {
            return Some(msg);
        }

        let mut reasons = Vec::new();

        // Fitness target reached
        if self.params.stop_fitness.enabled && self.f_best_ever <= self.params.stop_fitness.value {
            reasons.push("Target fitness reached");
        }

        // TolFun
        if self.generation > 0 && !self.fitness_values.is_empty() {
            let current_range = self
                .fitness_values
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                    (min.min(val), max.max(val))
                });
            let hist_range = if !self.fitness_history.is_empty() {
                self.fitness_history
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                        (min.min(val), max.max(val))
                    })
            } else {
                current_range
            };

            let total_range = (
                current_range.0.min(hist_range.0),
                current_range.1.max(hist_range.1),
            );
            let range = total_range.1 - total_range.0;

            if range <= self.params.stop_tol_fun {
                reasons.push("TolFun: function value differences too small");
            }
        }

        // TolFunHist
        if self.fitness_history.len() >= self.max_history_length {
            let hist_range = self
                .fitness_history
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                    (min.min(val), max.max(val))
                });
            let range = hist_range.1 - hist_range.0;

            if range <= self.params.stop_tol_fun_hist {
                reasons.push("TolFunHist: history of function value changes too small");
            }
        }

        // TolX
        let mut tolx_count = 0;
        for i in 0..self.params.dimension {
            let std_dev = self.sigma * self.c_matrix[[i, i]].sqrt();
            let pc_component = self.sigma * self.pc[i].abs();

            if std_dev < self.params.stop_tol_x && pc_component < self.params.stop_tol_x {
                tolx_count += 1;
            }
        }

        if tolx_count == self.params.dimension {
            reasons.push("TolX: object variable changes below tolerance");
        }

        // TolUpX
        for i in 0..self.params.dimension {
            let std_dev = self.sigma * self.c_matrix[[i, i]].sqrt();
            if std_dev > self.params.stop_tol_upx_factor * self.params.initial_std_devs[i] {
                reasons.push("TolUpX: standard deviation increased too much");
                break;
            }
        }

        // MaxFunEvals
        if self.count_evals >= self.params.stop_max_fun_evals {
            reasons.push("MaxFunEvals: maximum function evaluations reached");
        }

        // MaxIter
        if self.params.stop_max_iter > 0 && self.generation >= self.params.stop_max_iter {
            reasons.push("MaxIter: maximum iterations reached");
        }

        if !reasons.is_empty() {
            self.stop_message = Some(reasons.join("; "));
            return self.stop_message.as_ref().map(|s| s.as_str());
        }

        None
    }

    // Getters
    pub fn generation(&self) -> usize {
        self.generation
    }
    pub fn count_evals(&self) -> usize {
        self.count_evals
    }
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    pub fn x_mean(&self) -> &Array1<f64> {
        &self.x_mean
    }
    pub fn x_best_ever(&self) -> &Array1<f64> {
        &self.x_best_ever
    }
    pub fn f_best_ever(&self) -> f64 {
        self.f_best_ever
    }
    pub fn dimension(&self) -> usize {
        self.params.dimension
    }
    pub fn lambda(&self) -> usize {
        self.params.lambda
    }

    pub fn best_fitness(&self) -> Option<f64> {
        if self.fitness_values.is_empty() {
            None
        } else {
            Some(self.fitness_values[self.index[0]])
        }
    }

    pub fn axis_ratio(&self) -> f64 {
        if self.d_vector.len() == 0 {
            1.0
        } else {
            let min_d = self.d_vector.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_d = self
                .d_vector
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            max_d / min_d
        }
    }

    pub fn covariance_matrix(&self) -> &Array2<f64> {
        &self.c_matrix
    }

    pub fn eigenvalues(&self) -> &Array1<f64> {
        &self.d_vector
    }

    pub fn eigenvectors(&self) -> &Array2<f64> {
        &self.b_matrix
    }

    pub fn minimize(
        &mut self,
        initial_point: Array1<f64>,
        initial_sigma: Option<f64>,
        population_size: Option<usize>,
        tol: Option<f64>,
        max_generations: Option<usize>,
    ) -> Result<CmaEsResult, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        self.sigma = initial_sigma.unwrap_or(0.3);
        self.params.lambda = population_size.unwrap_or(4 + (3.0 * (n as f64).ln()) as usize);
        self.population = vec![Array1::zeros(self.params.dimension); self.params.lambda];
        self.fitness_values = vec![0.0; self.params.lambda];
        self.params.mu = self.params.lambda / 2;
        self.params.set_tol(tol.unwrap_or(1e-11));
        self.params.stop_max_iter = max_generations.unwrap_or(1000);
        _ = self.params.set_weights("log");
        self.index = (0..self.params.lambda).collect();

        if let Some(target) = tol {
            self.params.stop_fitness.enabled = true;
            self.params.stop_fitness.value = target;
        }

        while self.test_for_termination().is_none() {
            _ = self.sample_population()?;
            let mut fitness_values = vec![0.0; self.population.len()];

            for (i, individual) in self.population.iter().enumerate() {
                fitness_values[i] = self.f.call(individual);
            }

            self.update_distribution(&fitness_values)?;
        }

        self.converged = true;

        Ok(CmaEsResult {
            xmin: self.x_best_ever.clone(),
            fmin: self.f_best_ever,
            generations: self.generation,
            fn_evals: 0,
            converged: self.converged,
            final_sigma: self.sigma,
            // condition_number: self.condition_number(&eigenvalues),
            condition_number: 0.0,
            history: self.fitness_history.clone(),
        })
    }

    pub fn optimize_with_bounds(
        &mut self,
        dimension: usize,
        lower_bounds: Array1<f64>,
        upper_bounds: Array1<f64>,
        _max_generations: u64,
    ) -> Result<(Array1<f64>, f64), MinimizerError> {
        let boundary_transform = BoundaryTransformation::new(lower_bounds, upper_bounds)?;

        let mut x_transformed = Array1::zeros(dimension);

        while self.test_for_termination().is_none() {
            _ = self.sample_population()?;
            let mut fitness_values = vec![0.0; self.population.len()];

            for (i, individual) in self.population.iter().enumerate() {
                // Transform to respect bounds
                boundary_transform.transform(individual, &mut x_transformed)?;
                fitness_values[i] = self.f.call(&x_transformed);
            }

            self.update_distribution(&fitness_values)?;
        }

        // Transform final result
        boundary_transform.transform(self.x_best_ever(), &mut x_transformed)?;
        let result = x_transformed;

        Ok((result, self.f_best_ever()))
    }
}

#[cfg(test)]
mod minimize_f64_cmaes_tests {
    use super::*;
    use crate::minimize::f64::{F1dim, MultiDimFn};
    use std::f64::consts::{E, PI};

    // Example objective functions

    // Simple sphere function: f(x) = sum(x_i^2)
    fn sphere_function(x: &Array1<f64>) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    // Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    fn rosenbrock_function(x: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            let term1 = 100.0 * (x[i + 1] - x[i] * x[i]).powi(2);
            let term2 = (1.0 - x[i]).powi(2);
            sum += term1 + term2;
        }
        sum
    }

    // Cigar function: f(x) = x_0^2 + 10^6 * sum(x_i^2) for i > 0
    fn cigar_function(x: &Array1<f64>) -> f64 {
        if x.is_empty() {
            return 0.0;
        }

        let mut sum = x[0] * x[0];
        for i in 1..x.len() {
            sum += 1e6 * x[i] * x[i];
        }
        sum
    }

    // Ellipsoid function with exponentially increasing conditioning
    fn ellipsoid_function(x: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        let n = x.len();

        for (i, &xi) in x.iter().enumerate() {
            let factor = 1000_f64.powf(i as f64 / (n - 1) as f64);
            sum += factor * xi * xi;
        }

        sum
    }

    // Helper function to create Ackley
    fn ackley(bias: f64) -> F1dim {
        F1dim::new(MultiDimFn::new(move |x: &Array1<f64>| {
            let n = x.len() as f64;
            let t1 = x.iter().map(|val| val.powi(2)).sum::<f64>();
            let t2 = x.iter().map(|val| (2.0 * PI * val).cos()).sum::<f64>();
            -20.0 * (-0.2 * (t1 / n).sqrt()).exp() - (t2 / n).exp() + 20.0 + E + bias
        }))
    }

    // Helper function to create Elliptic
    fn elliptic(bias: f64) -> F1dim {
        F1dim::new(MultiDimFn::new(move |x: &Array1<f64>| {
            let sum: f64 = x
                .iter()
                .enumerate()
                .map(|(i, val)| 1e6_f64.powf((i / (x.len() - 1)) as f64) * val.powi(2))
                .sum();
            sum + bias
        }))
    }

    // Helper function to create Griewank
    fn griewank(bias: f64) -> F1dim {
        F1dim::new(MultiDimFn::new(move |x: &Array1<f64>| {
            let t1 = x.iter().map(|val| val.powi(2)).sum::<f64>() / 4000.0;
            let t2 = x
                .iter()
                .enumerate()
                .map(|(i, val)| (val / (i as f64 + 1.0).sqrt()).cos())
                .product::<f64>();
            t1 - t2 + 1.0 + bias
        }))
    }

    // Helper function to create Rosenbrock
    fn rosenbrock(shift: Array1<f64>, bias: f64) -> F1dim {
        F1dim::new(MultiDimFn::new(move |x: &Array1<f64>| {
            let x_new = x - &shift + 1.0;
            let sum: f64 = x_new
                .windows(2)
                .into_iter()
                .map(|pair| 100.0 * (pair[0].powi(2) - pair[1]).powi(2) + (pair[0] - 1.0).powi(2))
                .sum();
            sum + bias
        }))
    }

    #[test]
    fn test_cmaes_creation() {
        let params = Parameters::new(5).unwrap();
        let cmaes = CmaEs::new(elliptic(0.0), params).unwrap();

        assert_eq!(cmaes.dimension(), 5);
        assert_eq!(cmaes.generation(), 0);
    }

    #[test]
    fn test_sphere_optimization() {
        let params = Parameters::new(3)
            .unwrap()
            .with_initial_x(array![1.0, 1.0, 1.0])
            .unwrap()
            .with_initial_std_devs(array![0.5, 0.5, 0.5])
            .unwrap();

        let mut cmaes = CmaEs::new(elliptic(0.0), params).unwrap();

        for _generation in 0..100 {
            if cmaes.test_for_termination().is_some() {
                break;
            }

            let population = cmaes.sample_population().unwrap();
            let mut fitness_values = vec![0.0; population.len()];

            for (i, individual) in population.iter().enumerate() {
                fitness_values[i] = sphere_function(individual);
            }

            cmaes.update_distribution(&fitness_values).unwrap();
        }

        // Should converge to near zero for sphere function
        assert!(cmaes.f_best_ever() < 1e-8);
    }

    #[test]
    fn test_boundary_transformation() {
        let lower = array![-1.0, -2.0];
        let upper = array![1.0, 2.0];
        let transform = BoundaryTransformation::new(lower, upper).unwrap();

        let x = array![0.0, 0.0];
        let mut y = array![0.0, 0.0];

        transform.transform(&x, &mut y).unwrap();

        // Should be within bounds
        assert!(y[0] >= -1.0 && y[0] <= 1.0);
        assert!(y[1] >= -2.0 && y[1] <= 2.0);
    }

    #[test]
    fn test_outer_product() {
        let a = array![1.0, 2.0];
        let b = array![3.0, 4.0];
        let result = outer(&a, &b);

        assert_eq!(result[[0, 0]], 3.0);
        assert_eq!(result[[0, 1]], 4.0);
        assert_eq!(result[[1, 0]], 6.0);
        assert_eq!(result[[1, 1]], 8.0);
    }

    #[test]
    fn test_parameter_validation() {
        // Test invalid dimension
        assert!(Parameters::new(0).is_err());

        // Test dimension mismatch
        let params = Parameters::new(3).unwrap();
        assert!(params.with_initial_x(array![1.0, 2.0]).is_err()); // Wrong size
    }

    mod basic_function_tests {
        use super::*;

        #[test]
        fn test_2d_quadratic() {
            // f(x,y) = (x-1)² + (y-2)², minimum at (1,2)
            let n = 2;
            let func = |x: &Array1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
            let objective = MultiDimFn::new(func);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(objective, params).unwrap();

            let result = cmaes
                .minimize(
                    array![0.0, 0.0],
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    Some(1000),
                )
                .unwrap();

            assert!((result.xmin[0] - 1.0).abs() < 1e-3);
            assert!((result.xmin[1] - 2.0).abs() < 1e-3);
            assert!(result.fmin < 1e-5);

            assert!((cmaes.x_best_ever[0] - 1.0).abs() < 1e-3);
            assert!((cmaes.x_best_ever[1] - 2.0).abs() < 1e-3);
            assert!(cmaes.f_best_ever < 1e-5);
        }

        #[test]
        fn test_1d_quadratic() {
            // f(x) = (x-5)², minimum at x=5
            let n = 1;
            let func = |x: &Array1<f64>| (x[0] - 5.0).powi(2);
            let objective = MultiDimFn::new(func);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(objective, params).unwrap();

            let result = cmaes
                .minimize(
                    array![0.0],
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    Some(1000),
                )
                .unwrap();

            assert!((result.xmin[0] - 5.0).abs() < 1e-3);
            assert!(result.fmin < 1e-5);
            assert!(result.converged || result.fmin < 1e-5);
        }

        #[test]
        fn test_rosenbrock_2d() {
            // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
            let n = 2;
            let rosenbrock =
                |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
            let objective = MultiDimFn::new(rosenbrock);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(objective, params).unwrap();

            let result = cmaes
                .minimize(
                    array![-1.2, 1.0],
                    Some(0.5),
                    Some(50),
                    Some(1e-8),
                    Some(500),
                )
                .unwrap();

            println!("\nresult.xmin = {:?}\n", result.xmin);
            assert!((result.xmin[0] - 1.0).abs() < 1e-2);
            assert!((result.xmin[1] - 1.0).abs() < 1e-2);
            assert!(result.fmin < 1e-3);

            assert!((cmaes.x_best_ever[0] - 1.0).abs() < 1e-2);
            assert!((cmaes.x_best_ever[1] - 1.0).abs() < 1e-2);
            assert!(cmaes.f_best_ever < 1e-3);
        }

        #[test]
        fn test_3d_sphere() {
            // f(x,y,z) = x² + y² + z², minimum at (0,0,0)
            let n = 3;
            let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum();
            let objective = MultiDimFn::new(sphere);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(objective, params).unwrap();

            let result = cmaes
                .minimize(
                    array![1.0, 1.0, 1.0],
                    Some(0.5),
                    Some(50),
                    Some(1e-8),
                    Some(500),
                )
                .unwrap();

            for &coord in &result.xmin {
                assert!(coord.abs() < 1e-3);
            }
            assert!(result.fmin < 1e-5);

            for &coord in &cmaes.x_best_ever {
                assert!(coord.abs() < 1e-3);
            }
            assert!(cmaes.f_best_ever < 1e-5);
        }

        #[test]
        fn test_ellipsoid_function() {
            // Ellipsoid: f(x) = sum(i * x_i^2), scaled quadratic
            let n = 3;
            let ellipsoid = |x: &Array1<f64>| {
                x.iter()
                    .enumerate()
                    .map(|(i, &xi)| (i + 1) as f64 * xi.powi(2))
                    .sum::<f64>()
            };
            let objective = MultiDimFn::new(ellipsoid);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(objective, params).unwrap();

            let result = cmaes
                .minimize(
                    array![1.0, 2.0, 3.0],
                    Some(1.0),
                    None,
                    Some(1e-8),
                    Some(200),
                )
                .unwrap();

            for &coord in &result.xmin {
                assert!(coord.abs() < 1e-3);
            }
            assert!(result.fmin < 1e-5);
        }
    }

    mod cec2005_tests {
        use super::*;
        use rand::Rng;

        const TEST_30D: bool = false;
        const TEST_50D: bool = false;
        // const TEST_30D: bool = true;
        // const TEST_50D: bool = true;

        #[test]
        fn test_f1() {
            let mut rng = rand::rng();
            let shift = array![
                -3.9311900e+001,
                5.8899900e+001,
                -4.6322400e+001,
                -7.4651500e+001,
                -1.6799700e+001,
                -8.0544100e+001,
                -1.0593500e+001,
                2.4969400e+001,
                8.9838400e+001,
                9.1119000e+000,
                -1.0744300e+001,
                -2.7855800e+001,
                -1.2580600e+001,
                7.5930000e+000,
                7.4812700e+001,
                6.8495900e+001,
                -5.3429300e+001,
                7.8854400e+001,
                -6.8595700e+001,
                6.3743200e+001,
                3.1347000e+001,
                -3.7501600e+001,
                3.3892900e+001,
                -8.8804500e+001,
                -7.8771900e+001,
                -6.6494400e+001,
                4.4197200e+001,
                1.8383600e+001,
                2.6521200e+001,
                8.4472300e+001,
                3.9176900e+001,
                -6.1486300e+001,
                -2.5603800e+001,
                -8.1182900e+001,
                5.8695800e+001,
                -3.0838600e+001,
                -7.2672500e+001,
                8.9925700e+001,
                -1.5193400e+001,
                -4.3337000e+000,
                5.3430000e+000,
                1.0560300e+001,
                -7.7726800e+001,
                5.2085900e+001,
                4.0394400e+001,
                8.8332800e+001,
                -5.5830600e+001,
                1.3181000e+000,
                3.6025000e+001,
                -6.9927100e+001,
                -8.6279000e+000,
                -5.6894400e+001,
                8.5129600e+001,
                1.7673600e+001,
                6.1529000e+000,
                -1.7695700e+001,
                -5.8953700e+001,
                3.0356400e+001,
                1.5920700e+001,
                -1.8008200e+001,
                8.0641100e+001,
                -4.2391200e+001,
                7.6277600e+001,
                -5.0165200e+001,
                -7.3573600e+001,
                2.8336900e+001,
                -5.7990500e+001,
                -2.2732700e+001,
                5.2026900e+001,
                3.9259900e+001,
                1.0867900e+001,
                7.7820700e+001,
                6.6039500e+001,
                -5.0066700e+001,
                5.5706300e+001,
                7.3714100e+001,
                3.8529600e+001,
                -5.6786500e+001,
                -8.9647700e+001,
                3.7957600e+001,
                2.9472000e+001,
                -3.5464100e+001,
                -3.1786800e+001,
                7.7323500e+001,
                5.4790600e+001,
                -4.8279400e+001,
                7.4271400e+001,
                7.2610300e+001,
                6.2964000e+001,
                -1.4144600e+001,
                2.0492300e+001,
                4.6589700e+001,
                -8.3602100e+001,
                -4.6480900e+001,
                8.3737300e+001,
                -7.9661100e+001,
                2.4347900e+001,
                -1.7230300e+001,
                7.2340400e+001,
                -3.6402200e+001
            ];
            let bias = -450.0;

            // 10d
            let n = 10;
            let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
            let shift_func = shift.clone();
            let func = move |x: &Array1<f64>| {
                x.iter()
                    .enumerate()
                    .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                    .sum()
            };
            let objective = MultiDimFn::new(func);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(objective, params).unwrap();
            let result = cmaes
                .minimize(
                    x,
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    Some(1000),
                )
                .unwrap();
            for i in 0..10 {
                assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                assert!((cmaes.x_best_ever[i] - shift[i]).abs() < 1e-2);
                assert!((cmaes.f_best_ever - bias) < 1e-3)
            }

            // 30d
            if TEST_30D {
                let n = 30;
                let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
                let shift_func = shift.clone();
                let func = move |x: &Array1<f64>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let params = Parameters::new(n).unwrap();
                let mut cmaes = CmaEs::new(objective, params).unwrap();
                let result = cmaes
                    .minimize(
                        x,
                        Some(0.3),
                        Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                        Some(1e-10),
                        Some(1000),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((result.fmin - bias) < 1e-3);

                    assert!((cmaes.x_best_ever[i] - shift[i]).abs() < 1e-2);
                    assert!((cmaes.f_best_ever - bias) < 1e-3)
                }
            }

            // 50d
            if TEST_50D {
                let n = 50;
                let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
                let shift_func = shift.clone();
                let func = move |x: &Array1<f64>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let params = Parameters::new(n).unwrap();
                let mut cmaes = CmaEs::new(objective, params).unwrap();
                let result = cmaes
                    .minimize(
                        x,
                        Some(0.3),
                        Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                        Some(1e-10),
                        Some(1000),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((result.fmin - bias) < 1e-3);

                    assert!((cmaes.x_best_ever[i] - shift[i]).abs() < 1e-2);
                    assert!((cmaes.f_best_ever - bias) < 1e-3)
                }
            }
        }

        #[test]
        fn test_f2() {
            let mut rng = rand::rng();
            let shift = array![
                3.5626700e+001,
                -8.2912300e+001,
                -1.0642300e+001,
                -8.3581500e+001,
                8.3155200e+001,
                4.7048000e+001,
                -8.9435900e+001,
                -2.7421900e+001,
                7.6144800e+001,
                -3.9059500e+001,
                4.8885700e+001,
                -3.9828000e+000,
                -7.1924300e+001,
                6.4194700e+001,
                -4.7733800e+001,
                -5.9896000e+000,
                -2.6282800e+001,
                -5.9181100e+001,
                1.4602800e+001,
                -8.5478000e+001,
                -5.0490100e+001,
                9.2400000e-001,
                3.2397800e+001,
                3.0238800e+001,
                -8.5094900e+001,
                6.0119700e+001,
                -3.6218300e+001,
                -8.5883000e+000,
                -5.1971000e+000,
                8.1553100e+001,
                -2.3431600e+001,
                -2.5350500e+001,
                -4.1248500e+001,
                8.8018000e+000,
                -2.4222200e+001,
                -8.7980700e+001,
                7.8047300e+001,
                -4.8052800e+001,
                1.4017700e+001,
                -3.6640500e+001,
                1.2216800e+001,
                1.8144900e+001,
                -6.4564700e+001,
                -8.4849300e+001,
                -7.6608800e+001,
                -1.7042000e+000,
                -3.6076100e+001,
                3.7033600e+001,
                1.8443100e+001,
                -6.4359000e+001,
                -3.9369200e+001,
                -1.7714000e+001,
                3.0198500e+001,
                -1.8548300e+001,
                9.6866000e+000,
                8.2600900e+001,
                -4.5525600e+001,
                5.1443000e+000,
                7.4204000e+001,
                6.6810300e+001,
                -6.3470400e+001,
                1.3032900e+001,
                -5.6878000e+000,
                2.9527100e+001,
                -4.3530000e-001,
                -2.6165200e+001,
                -6.6847000e+000,
                -8.0229100e+001,
                -2.9581500e+001,
                8.2042200e+001,
                7.7177000e+001,
                -1.1277000e+001,
                3.2075900e+001,
                -2.6858000e+000,
                8.1509600e+001,
                6.4077000e+001,
                -2.6129400e+001,
                -8.4782000e+001,
                -6.2876800e+001,
                -3.7635500e+001,
                7.6891600e+001,
                5.3417000e+001,
                -2.5331100e+001,
                -3.8070200e+001,
                -8.4173800e+001,
                -1.1224600e+001,
                -8.3461900e+001,
                -1.7550800e+001,
                -3.6528500e+001,
                8.9552800e+001,
                2.5879400e+001,
                6.8625200e+001,
                5.5796800e+001,
                -2.9597500e+001,
                -5.8097600e+001,
                6.5741300e+001,
                -8.8703000e+000,
                -5.3281000e+000,
                7.4066100e+001,
                4.0338000e+000
            ];
            let bias = -450.0;

            // 10d
            let n = 10;
            let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
            let shift_func = shift.clone();
            let func = move |x: &Array1<f64>| {
                x.iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let mut new_val = 0.0;
                        for j in 0..=i {
                            new_val += (x[j] - shift_func[j]).powi(2);
                        }
                        new_val + bias
                    })
                    .sum()
            };
            let objective = MultiDimFn::new(func);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(objective, params).unwrap();
            let result = cmaes
                .minimize(
                    x,
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    Some(1000),
                )
                .unwrap();
            for i in 0..10 {
                assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                assert!((cmaes.x_best_ever()[i] - shift[i]).abs() < 1e-2);
                assert!((cmaes.f_best_ever - bias) < 1e-3)
            }

            // 30d
            if TEST_30D {
                let n = 30;
                let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
                let shift_func = shift.clone();
                let func = move |x: &Array1<f64>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, _)| {
                            let mut new_val = 0.0;
                            for j in 0..=i {
                                new_val += (x[j] - shift_func[j]).powi(2);
                            }
                            new_val + bias
                        })
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let params = Parameters::new(n).unwrap();
                let mut cmaes = CmaEs::new(objective, params).unwrap();
                let result = cmaes
                    .minimize(
                        x,
                        Some(0.3),
                        Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                        Some(1e-10),
                        Some(1000),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((result.fmin - bias) < 1e-3);

                    assert!((cmaes.x_best_ever[i] - shift[i]).abs() < 1e-2);
                    assert!((cmaes.f_best_ever - bias) < 1e-3)
                }
            }

            // 50d
            if TEST_50D {
                let n = 50;
                let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
                let shift_func = shift.clone();
                let func = move |x: &Array1<f64>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, _)| {
                            let mut new_val = 0.0;
                            for j in 0..=i {
                                new_val += (x[j] - shift_func[j]).powi(2);
                            }
                            new_val + bias
                        })
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let params = Parameters::new(n).unwrap();
                let mut cmaes = CmaEs::new(objective, params).unwrap();
                let result = cmaes
                    .minimize(
                        x,
                        Some(0.3),
                        Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                        Some(1e-10),
                        Some(1000),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((result.fmin - bias) < 1e-3);

                    assert!((cmaes.x_best_ever[i] - shift[i]).abs() < 1e-2);
                    assert!((cmaes.f_best_ever - bias) < 1e-3)
                }
            }
        }

        #[test]
        fn test_f3() {
            let mut rng = rand::rng();
            let shift = array![
                -3.2201300e+001,
                6.4977600e+001,
                -3.8300000e+001,
                -2.3258200e+001,
                -5.4008800e+001,
                8.6628600e+001,
                -6.3009000e+000,
                -4.9364400e+001,
                5.3499000e+000,
                5.2241800e+001
            ];
            let m = array![
                [
                    1.7830682721057345e-001,
                    5.5786330587166588e-002,
                    4.7591905576669730e-001,
                    2.4551129863391566e-001,
                    3.1998625926387086e-001,
                    3.2102001448363848e-001,
                    2.7787561319902176e-002,
                    2.6664001046775621e-001,
                    4.1568009651337917e-001,
                    -4.7771934552669726e-001
                ],
                [
                    6.3516362859468667e-001,
                    5.0091423836646241e-002,
                    2.0110601384121973e-001,
                    -6.8076882416633511e-001,
                    -4.9934546553907944e-002,
                    -4.6399423424582961e-002,
                    -1.9460194646748039e-001,
                    1.8961539926194687e-001,
                    -1.9416259626804547e-002,
                    1.0639981029473855e-001
                ],
                [
                    3.2762147366023187e-001,
                    3.6016598714114556e-001,
                    -2.3635655094044949e-001,
                    -1.8566854017444848e-002,
                    -2.4479096747593634e-001,
                    4.4818973341886903e-001,
                    5.3518635733619568e-001,
                    -3.1206925190530521e-001,
                    -1.3863719921728737e-001,
                    -2.0713981146209595e-001
                ],
                [
                    -6.4783210587984280e-002,
                    -4.9424101683695937e-001,
                    1.3101175297435969e-001,
                    3.1615171931194543e-002,
                    -1.7506107914871860e-001,
                    6.8908039344918381e-001,
                    1.0544234469094992e-002,
                    2.1948984793273507e-001,
                    -1.6468539805844565e-001,
                    3.9048550518513409e-001
                ],
                [
                    -2.7648044785371367e-001,
                    1.1383114506120220e-001,
                    -3.0818401502810994e-001,
                    -3.5959407104438740e-001,
                    2.6446258034702191e-001,
                    2.8616788379157501e-002,
                    4.7528027904995646e-001,
                    4.0993994049770172e-001,
                    4.1131043368915432e-001,
                    2.2899345188886880e-001
                ],
                [
                    1.5454249061641606e-001,
                    5.4899186274157996e-001,
                    -1.8382029941792261e-001,
                    3.3944461903909162e-001,
                    2.8596188774255699e-001,
                    1.2833167642713417e-001,
                    -2.5495080172376317e-001,
                    3.9460752302037100e-001,
                    -3.4524640270007412e-001,
                    2.9590318323368509e-001
                ],
                [
                    -5.1907977690014512e-002,
                    -1.4450757809700329e-001,
                    -4.6086919626114314e-001,
                    -5.3687964818368079e-002,
                    -3.6317793499109247e-001,
                    2.7439997038558633e-002,
                    -2.1422629652542946e-001,
                    5.0545148893084779e-001,
                    -9.8064717019089837e-002,
                    -5.6346991018564507e-001
                ],
                [
                    5.0142989354460654e-001,
                    -5.3133659048457516e-001,
                    -3.7294385871521135e-001,
                    2.3370866431381510e-001,
                    4.4327537662488531e-001,
                    -1.6972740381143742e-001,
                    2.0364148963331691e-001,
                    -2.3717523924336927e-002,
                    -7.1805455862954920e-002,
                    -7.3332178450339763e-003
                ],
                [
                    1.0441248047680891e-001,
                    4.3064226149369542e-002,
                    -4.1675972625940993e-001,
                    1.6522876074361707e-002,
                    1.7437281849141879e-003,
                    2.9594944879030760e-001,
                    -5.1197487739368741e-001,
                    -3.2679819762357892e-001,
                    5.8253106590933512e-001,
                    1.3204141339826148e-001
                ],
                [
                    -2.9645907657631693e-001,
                    -3.1303011496605505e-002,
                    -7.8009154082116602e-002,
                    -4.1548534874482024e-001,
                    5.6959403572443468e-001,
                    2.9095198400348149e-001,
                    -1.8560717510075503e-001,
                    -2.4653488847859115e-001,
                    -3.7149025085479792e-001,
                    -3.0015617693118707e-001
                ],
            ];
            let bias = -450.0;

            // 10d
            let n = 10;
            let x = (&Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0) - &shift)
                .dot(&m);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(elliptic(bias), params).unwrap();
            let result = cmaes
                .minimize(
                    x.clone(),
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    // Some(1e3 as usize * n.pow(2)),
                    Some(10000),
                )
                .unwrap();
            for i in 0..10 {
                // assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!(result.xmin[i].abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                // assert!((cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                assert!(cmaes.x_best_ever[i].abs() < 1e-2);
                assert!((cmaes.f_best_ever - bias) < 1e-3)
            }
        }

        #[test]
        fn test_f4() {
            let mut rng = rand::rng();
            let shift = array![
                3.5626700e+001,
                -8.2912300e+001,
                -1.0642300e+001,
                -8.3581500e+001,
                8.3155200e+001,
                4.7048000e+001,
                -8.9435900e+001,
                -2.7421900e+001,
                7.6144800e+001,
                -3.9059500e+001,
                4.8885700e+001,
                -3.9828000e+000,
                -7.1924300e+001,
                6.4194700e+001,
                -4.7733800e+001,
                -5.9896000e+000,
                -2.6282800e+001,
                -5.9181100e+001,
                1.4602800e+001,
                -8.5478000e+001,
                -5.0490100e+001,
                9.2400000e-001,
                3.2397800e+001,
                3.0238800e+001,
                -8.5094900e+001,
                6.0119700e+001,
                -3.6218300e+001,
                -8.5883000e+000,
                -5.1971000e+000,
                8.1553100e+001,
                -2.3431600e+001,
                -2.5350500e+001,
                -4.1248500e+001,
                8.8018000e+000,
                -2.4222200e+001,
                -8.7980700e+001,
                7.8047300e+001,
                -4.8052800e+001,
                1.4017700e+001,
                -3.6640500e+001,
                1.2216800e+001,
                1.8144900e+001,
                -6.4564700e+001,
                -8.4849300e+001,
                -7.6608800e+001,
                -1.7042000e+000,
                -3.6076100e+001,
                3.7033600e+001,
                1.8443100e+001,
                -6.4359000e+001,
                -3.9369200e+001,
                -1.7714000e+001,
                3.0198500e+001,
                -1.8548300e+001,
                9.6866000e+000,
                8.2600900e+001,
                -4.5525600e+001,
                5.1443000e+000,
                7.4204000e+001,
                6.6810300e+001,
                -6.3470400e+001,
                1.3032900e+001,
                -5.6878000e+000,
                2.9527100e+001,
                -4.3530000e-001,
                -2.6165200e+001,
                -6.6847000e+000,
                -8.0229100e+001,
                -2.9581500e+001,
                8.2042200e+001,
                7.7177000e+001,
                -1.1277000e+001,
                3.2075900e+001,
                -2.6858000e+000,
                8.1509600e+001,
                6.4077000e+001,
                -2.6129400e+001,
                -8.4782000e+001,
                -6.2876800e+001,
                -3.7635500e+001,
                7.6891600e+001,
                5.3417000e+001,
                -2.5331100e+001,
                -3.8070200e+001,
                -8.4173800e+001,
                -1.1224600e+001,
                -8.3461900e+001,
                -1.7550800e+001,
                -3.6528500e+001,
                8.9552800e+001,
                2.5879400e+001,
                6.8625200e+001,
                5.5796800e+001,
                -2.9597500e+001,
                -5.8097600e+001,
                6.5741300e+001,
                -8.8703000e+000,
                -5.3281000e+000,
                7.4066100e+001,
                4.0338000e+000
            ];
            let bias = -450.0;

            // 10d
            let n = 10;
            let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
            let shift_func = shift.clone();
            let rand_val = rng.random::<f64>();
            let func = move |x: &Array1<f64>| {
                x.iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let mut new_val = 0.0;
                        for j in 0..=i {
                            new_val += (x[j] - shift_func[j]).powi(2);
                        }
                        new_val + bias
                    })
                    .sum::<f64>()
                    * (1.0 + 0.4 * rand_val)
            };
            let objective = MultiDimFn::new(func);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(objective, params).unwrap();
            let result = cmaes
                .minimize(
                    x,
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    Some(1000),
                )
                .unwrap();
            for i in 0..10 {
                assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                assert!((cmaes.x_best_ever[i] - shift[i]).abs() < 1e-2);
                assert!((cmaes.f_best_ever - bias) < 1e-3)
            }
        }

        #[test]
        fn test_f6() {
            let mut rng = rand::rng();
            let shift = array![
                8.1023200e+001,
                -4.8395000e+001,
                1.9231600e+001,
                -2.5231000e+000,
                7.0433800e+001,
                4.7177400e+001,
                -7.8358000e+000,
                -8.6669300e+001,
                5.7853200e+001,
                -9.9533000e+000,
                2.0777800e+001,
                5.2548600e+001,
                7.5926300e+001,
                4.2877300e+001,
                -5.8272000e+001,
                -1.6972800e+001,
                7.8384500e+001,
                7.5042700e+001,
                -1.6151300e+001,
                7.0856900e+001,
                -7.9579500e+001,
                -2.6483700e+001,
                5.6369900e+001,
                -8.8224900e+001,
                -6.4999600e+001,
                -5.3502200e+001,
                -5.4230000e+001,
                1.8682600e+001,
                -4.1006100e+001,
                -5.4213400e+001,
                -8.7250600e+001,
                4.4421400e+001,
                -9.8826000e+000,
                7.7726600e+001,
                -6.1210000e+000,
                -1.4643000e+001,
                6.2319800e+001,
                4.5274000e+000,
                -5.3523400e+001,
                3.0984700e+001,
                6.0861300e+001,
                -8.6464800e+001,
                3.2629800e+001,
                -2.1693400e+001,
                5.9723200e+001,
                5.0630000e-001,
                3.7704800e+001,
                -1.2799300e+001,
                -3.5168800e+001,
                -5.5862300e+001,
                -5.5182300e+001,
                3.2800100e+001,
                -3.5502400e+001,
                7.5012000e+000,
                -6.2842800e+001,
                3.5621700e+001,
                -2.1892800e+001,
                6.4802000e+001,
                6.3657900e+001,
                1.6841300e+001,
                -6.2050000e-001,
                7.1958400e+001,
                5.7893200e+001,
                2.6083800e+001,
                5.7235300e+001,
                2.8840900e+001,
                -2.8445200e+001,
                -3.7849300e+001,
                -2.8585100e+001,
                6.1342000e+000,
                4.0880300e+001,
                -3.4327700e+001,
                6.0929200e+001,
                1.2253000e+001,
                -2.3325500e+001,
                3.6493100e+001,
                8.3828000e+000,
                -9.9215000e+000,
                3.5022100e+001,
                2.1835800e+001,
                5.3067700e+001,
                8.2231800e+001,
                4.0662000e+000,
                6.8425500e+001,
                -5.8867800e+001,
                8.6354400e+001,
                -4.1139400e+001,
                -4.4580700e+001,
                6.7633500e+001,
                4.2715000e+001,
                -6.5426600e+001,
                -8.7883700e+001,
                7.0901600e+001,
                -5.4155100e+001,
                -3.6229800e+001,
                2.9059600e+001,
                -3.8806400e+001,
                -5.5396000e+000,
                -7.8339300e+001,
                8.7900200e+001
            ];
            let bias = 390.0;

            // 10d
            let n = 10;
            let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
            let params = Parameters::new(n).unwrap();
            let mut cmaes =
                CmaEs::new(rosenbrock(shift.slice(s![0..10]).to_owned(), bias), params).unwrap();
            let result = cmaes
                .minimize(
                    x,
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    Some(1e3 as usize * n.pow(2)),
                )
                .unwrap();
            for i in 0..10 {
                assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                assert!((cmaes.x_best_ever[i] - shift[i]).abs() < 1e-2);
                assert!((cmaes.f_best_ever - bias) < 1e-3)
            }
        }

        #[test]
        fn test_f7() {
            let mut rng = rand::rng();
            let shift = array![
                -2.7626840e+002,
                -1.1911000e+001,
                -5.7878840e+002,
                -2.8764860e+002,
                -8.4385800e+001,
                -2.2867530e+002,
                -4.5815160e+002,
                -2.0221450e+002,
                -1.0586420e+002,
                -9.6489800e+001,
            ];
            let m = array![
                [
                    -7.3696625825313500e-002,
                    1.5747490444892893e+000,
                    -6.4377942207169941e-002,
                    6.3201848730939580e-001,
                    -1.2455211411481415e+000,
                    -3.5341187428098381e-001,
                    3.5031691018519090e-001,
                    6.2886479758992697e-001,
                    6.8593632355012335e-001,
                    1.3975663076173925e+000
                ],
                [
                    6.3700016123079051e-001,
                    -1.3833836770823484e+000,
                    -2.4437874951092337e-001,
                    1.6992995943357547e+000,
                    7.1757447137502850e-001,
                    -7.7753800570270454e-002,
                    4.9291080765053624e-001,
                    1.1392847178100191e-001,
                    4.8163647386641817e-001,
                    2.8150613437207017e-001
                ],
                [
                    -1.4466181982194921e+000,
                    -1.1273816086105013e+000,
                    -1.0665724848959319e+000,
                    2.1900088934332190e-001,
                    -5.8130776006865136e-002,
                    -9.9187841926086026e-002,
                    -1.2465831572524580e-001,
                    -5.0547372808368829e-001,
                    -2.1020191419640880e-001,
                    1.1509984987284301e+000
                ],
                [
                    1.0410802679063424e+000,
                    4.7577677793232626e-001,
                    9.6430154567967874e-001,
                    1.5636976117984064e-002,
                    2.0539698111678034e-001,
                    2.5839780039821658e-001,
                    -5.1710361801897031e-001,
                    -1.5449014589834349e+000,
                    -1.4560361158442292e+000,
                    9.9877904060730438e-001
                ],
                [
                    2.6260272944635960e-001,
                    9.2947540741436874e-001,
                    -1.2953100028930926e+000,
                    6.6512029642561388e-001,
                    -2.7957781701655993e-001,
                    8.4060537698112758e-001,
                    -5.2922829607729160e-001,
                    -8.6040220072910467e-001,
                    4.9503162769183251e-001,
                    -6.3765376892958103e-001
                ],
                [
                    8.1307889477954698e-002,
                    8.0062327426592494e-001,
                    7.2294618679188488e-002,
                    4.4874698427975906e-001,
                    1.7959858022699743e-001,
                    -1.3634800693969209e+000,
                    7.5257943996576704e-002,
                    -1.2486791053473751e+000,
                    6.8143526407032673e-001,
                    1.3558980136836016e-001
                ],
                [
                    8.7913516653862697e-002,
                    2.1022739728349416e-001,
                    -1.5708234535904123e-001,
                    -3.5182196550454031e-001,
                    -6.4190160213761838e-002,
                    1.5082748057903228e+000,
                    1.1168462803089814e+000,
                    -3.6773042225135699e-001,
                    2.6828021681357744e-001,
                    4.9698836189165707e-001
                ],
                [
                    -9.5406235378612747e-001,
                    3.9879009060640763e-001,
                    5.8022630243503770e-001,
                    2.4831174649263604e-001,
                    1.1781385394000925e+000,
                    5.3134809745284084e-001,
                    7.8257240450327026e-001,
                    -3.8166809840963106e-001,
                    -4.8082474351369503e-001,
                    -6.2076533636514075e-001
                ],
                [
                    2.7628599479864874e-001,
                    3.6188284466692094e-001,
                    -1.0302756351623272e+000,
                    7.2348644867809120e-001,
                    -3.7379075361566066e-001,
                    -7.9223639600376997e-002,
                    1.6221551897070494e+000,
                    -8.2880436781697358e-003,
                    -1.0881497169330046e+000,
                    -1.9204701133595675e-001
                ],
                [
                    -3.0035568486304853e-001,
                    -5.0758053487595001e-001,
                    3.1143454840627821e-001,
                    -2.5444900307396151e-001,
                    -7.7988528102301924e-001,
                    -6.8262839999436264e-001,
                    5.5932665521935510e-001,
                    -7.9579050121422423e-001,
                    4.7071685181799255e-001,
                    -8.0019268494895490e-001
                ],
            ];
            let bias = -180.0;

            // 10d
            let n = 10;
            let x = (&Array1::from_shape_fn(n, |_| rng.random::<f64>() * 600.0) - &shift).dot(&m);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(griewank(bias), params).unwrap();
            let result = cmaes
                .minimize(
                    x,
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    Some(1e3 as usize * n.pow(2)),
                )
                .unwrap();
            println!(
                "\n\nxmin = {:?}\nfmin = {:?}\n\n",
                &result.xmin,
                result.fmin - bias
            );
            for i in 0..10 {
                // assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.xmin[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                // assert!((cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((cmaes.x_best_ever[i]).abs() < 1e-2);
                assert!((cmaes.f_best_ever - bias) < 1e-3)
            }
        }

        #[test]
        fn test_f8() {
            let mut rng = rand::rng();
            let shift = array![
                -1.6823000e+001,
                1.4976900e+001,
                6.1690000e+000,
                9.5566000e+000,
                1.9541700e+001,
                -1.7190000e+001,
                -1.8824800e+001,
                8.5110000e-001,
                -1.5116200e+001,
                1.0793400e+001,
            ];
            let m = array![
                [
                    -1.8785768809450733e+001,
                    3.3616954860437176e+001,
                    2.6882113915382682e+001,
                    -1.0433064197429360e+001,
                    9.4489289408247579e-001,
                    -3.3538964332596910e+000,
                    3.5352127339076374e+000,
                    7.3942769413967824e+000,
                    7.7909087526412346e+000,
                    2.0912921099835673e+000
                ],
                [
                    -3.8080580880289006e-001,
                    1.0420967284673154e+001,
                    9.3472919131156775e+000,
                    -2.0926490513724943e+001,
                    1.1425903158890700e+001,
                    1.1056372574995397e+000,
                    3.6879282505970373e+001,
                    -1.9103397804634721e+000,
                    7.5611684932093199e+000,
                    -9.7430664357899719e+000
                ],
                [
                    -1.2341688082549489e+001,
                    6.3621993552738045e+000,
                    8.2484299397924641e+000,
                    8.0892563664178354e+000,
                    6.9234506619011427e-002,
                    2.5786241359574857e+000,
                    -4.9734021861818611e-001,
                    -2.0627220954843271e+000,
                    1.4302051477457656e+000,
                    1.5522003760944671e+001
                ],
                [
                    -1.7006542510444671e+001,
                    -1.2679306064055677e+001,
                    5.1658120519511158e+001,
                    -3.9766120780636900e+000,
                    3.9349384750136576e+000,
                    -3.0777202564613845e+001,
                    6.1465971476271157e+000,
                    -1.1404959107806402e+001,
                    1.2694206030880832e+001,
                    -9.3951432281174387e+000
                ],
                [
                    -5.4847491542826621e+000,
                    -1.3643476981518553e+001,
                    -2.0812578603542899e+001,
                    1.2480631776818850e+001,
                    8.4497820599569917e-001,
                    2.4830393330514045e+001,
                    3.3838505185702559e+001,
                    -1.7003569707093064e+001,
                    -5.2939643674048442e+000,
                    2.6065703095424336e+001
                ],
                [
                    1.1422878520470586e+001,
                    1.0221461943150496e+001,
                    -5.9994789874002628e+000,
                    -8.9358916025741415e+000,
                    3.3407916251514460e+000,
                    3.9245488542554492e+000,
                    -6.7605717857278513e+000,
                    1.4016300477765046e+001,
                    2.3533969357952147e+000,
                    -1.5957358828479556e+001
                ],
                [
                    1.4106979735005300e+001,
                    -6.8979757292229404e-001,
                    2.5928358266684882e+001,
                    -3.0138271725378775e+001,
                    1.2953067028884863e+001,
                    -1.7125782201118525e+001,
                    1.9122903237509483e+001,
                    3.8502101712160597e+000,
                    1.4449871260335931e+001,
                    -3.7768641488073015e+001
                ],
                [
                    1.8171620273851625e+000,
                    -4.5228977429981496e+000,
                    2.5960648243684310e+000,
                    -3.0779703663335480e+000,
                    3.6662383806277021e+000,
                    -3.1422719671052084e+000,
                    -1.9391037957658499e+000,
                    -1.1328460209494431e+000,
                    -1.4593971192280721e+000,
                    -4.3850653050781068e+000
                ],
                [
                    1.7059635015136770e+001,
                    -4.0887343040678509e+001,
                    -9.0413685473717607e+000,
                    9.2078166133516532e+000,
                    2.4835590816969209e+000,
                    -3.1352382866663429e+000,
                    -5.1597084344852373e-001,
                    -1.0448164970351954e+001,
                    -3.9790838641391200e+000,
                    -5.0101517638708923e+000
                ],
                [
                    -2.1004104733841622e+000,
                    4.2857434922129016e+000,
                    1.8138803730523911e+001,
                    -5.5691566223518540e+000,
                    2.0414928764167950e-002,
                    -5.5315683071808275e+000,
                    1.7507462325577925e+000,
                    2.0183823538506891e+000,
                    8.9673707865551204e+000,
                    -3.5936542419629482e+000,
                ]
            ];
            let bias = -140.0;

            // 10d
            let n = 10;
            let x =
                (&Array1::from_shape_fn(n, |_| rng.random::<f64>() * 64.0 - 32.0) - &shift).dot(&m);
            let params = Parameters::new(n).unwrap();
            let mut cmaes = CmaEs::new(ackley(bias), params).unwrap();
            let result = cmaes
                .minimize(
                    x,
                    Some(0.3),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10),
                    Some(1e3 as usize * n.pow(2)),
                )
                .unwrap();
            println!(
                "\n\nxmin = {:?}\nfmin = {:?}\n\n",
                &result.xmin - &shift,
                result.fmin - bias
            );
            for i in 0..10 {
                assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                assert!((cmaes.x_best_ever[i] - shift[i]).abs() < 1e-2);
                assert!((cmaes.f_best_ever - bias) < 1e-3)
            }
        }
    }
}
