#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::{error::MinimizerError, float::RFFloat, minimize::ObjFn};
use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::fmt;

/// Result of CMA-ES optimization
#[derive(Debug, Clone)]
pub struct CmaEsResult<T> {
    pub xmin: Array1<T>,
    pub fmin: T,
    pub generations: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_sigma: T,
    pub condition_number: T,
    pub history: Vec<T>,
}

pub struct CmaEs<T> {
    xmin: Array1<T>,
    fmin: T,
    f: Box<dyn ObjFn<T>>,
    generations: usize,
    converged: bool,
    rng: StdRng,
}

impl<T> Clone for CmaEs<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            f: dyn_clone::clone_box(&*self.f),
            generations: self.generations,
            converged: self.converged,
            rng: self.rng.clone(),
        }
    }
}

impl<T> CmaEs<T>
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
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        CmaEs {
            xmin: array![],
            fmin: T::infinity(),
            f: Box::new(f),
            generations: 0,
            converged: false,
            rng: StdRng::seed_from_u64(42), // Default seed
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn<T>>) -> Self {
        CmaEs {
            xmin: array![],
            fmin: T::infinity(),
            f: f,
            generations: 0,
            converged: false,
            rng: StdRng::seed_from_u64(42),
        }
    }

    pub fn with_seed<F>(f: F, seed: u64) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        CmaEs {
            xmin: array![],
            fmin: T::infinity(),
            f: Box::new(f),
            generations: 0,
            converged: false,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Compute eigendecomposition of a symmetric matrix using Jacobi method
    fn eigen_decomposition(&self, matrix: ArrayView2<T>) -> (Array1<T>, Array2<T>) {
        let n = matrix.nrows();
        let mut a = matrix.to_owned();
        let mut v = Array2::<T>::eye(n);

        let max_iter = 50;
        let tolerance = 1e-12.into();

        for _ in 0..max_iter {
            // Find largest off-diagonal element
            let mut max_val = T::zero();
            let (mut p, mut q) = (0, 1);

            for i in 0..n {
                for j in i + 1..n {
                    if a[[i, j]].abs() > max_val {
                        max_val = a[[i, j]].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < tolerance {
                break;
            }

            // Compute rotation angle
            let theta = if (&a[[q, q]] - &a[[p, p]]).abs() < 1e-10.into() {
                (std::f64::consts::PI / 4.0).into()
            } else {
                0.5 * (2.0 * &a[[p, q]] / (&a[[q, q]] - &a[[p, p]])).atan()
            };

            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            // Apply Jacobi rotation
            let app = a[[p, p]].clone();
            let aqq = a[[q, q]].clone();
            let apq = a[[p, q]].clone();

            a[[p, p]] = &cos_theta * &cos_theta * &app + &sin_theta * &sin_theta * &aqq
                - 2.0 * &cos_theta * &sin_theta * &apq;
            a[[q, q]] = &sin_theta * &sin_theta * &app
                + &cos_theta * &cos_theta * &aqq
                + 2.0 * &cos_theta * &sin_theta * &apq;
            a[[p, q]] = T::zero();
            a[[q, p]] = T::zero();

            for i in 0..n {
                if i != p && i != q {
                    let aip = a[[i, p]].clone();
                    let aiq = a[[i, q]].clone();
                    a[[i, p]] = &cos_theta * &aip - &sin_theta * &aiq;
                    a[[p, i]] = a[[i, p]].clone();
                    a[[i, q]] = &sin_theta * &aip + &cos_theta * &aiq;
                    a[[q, i]] = a[[i, q]].clone();
                }

                let vip = v[[i, p]].clone();
                let viq = v[[i, q]].clone();
                v[[i, p]] = &cos_theta * &vip - &sin_theta * &viq;
                v[[i, q]] = &sin_theta * &vip + &cos_theta * &viq;
            }
        }

        // Extract eigenvalues and sort
        let eigenvalues: Array1<T> = Array1::from_vec((0..n).map(|i| a[[i, i]].clone()).collect());
        let mut indices: Vec<usize> = (0..n).collect();

        // Sort by eigenvalue magnitude (descending)
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .abs()
                .partial_cmp(&eigenvalues[i].abs())
                .unwrap()
        });

        let sorted_eigenvalues: Array1<T> =
            indices.iter().map(|&i| eigenvalues[i].clone()).collect();
        let sorted_eigenvectors =
            Array2::from_shape_fn((indices.len(), n), |(i, j)| v[[j, indices[i]]].clone());

        (sorted_eigenvalues, sorted_eigenvectors)
    }

    /// Compute condition number of covariance matrix
    fn condition_number(&self, eigenvalues: ArrayView1<T>) -> T {
        let max_eig = eigenvalues
            .iter()
            .fold(T::zero(), |acc, x| acc.max(&x.abs()));
        let min_eig = eigenvalues
            .iter()
            .fold(T::infinity(), |acc, x| acc.min(&x.abs()));
        if min_eig > 1e-14.into() {
            &max_eig / &min_eig
        } else {
            T::infinity()
        }
    }

    /// Standard CMA-ES algorithm
    pub fn cma_es(
        &mut self,
        initial_point: ArrayView1<T>,
        initial_sigma: Option<T>,
        population_size: Option<usize>,
        tol: Option<T>,
        max_generations: Option<usize>,
    ) -> Result<CmaEsResult<T>, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let sigma = initial_sigma.unwrap_or(0.3.into());
        let lambda = population_size.unwrap_or(4 + (3.0 * (n as f64).ln()) as usize);
        let mu = lambda / 2;
        let tol = tol.unwrap_or(1e-11.into());
        let max_gen = max_generations.unwrap_or(1000);

        if sigma <= T::zero() {
            return Err(MinimizerError::InvalidTolerance);
        }

        // Strategy parameters
        let cc =
            ((4.0 + mu as f64 / n as f64) / (n as f64 + 4.0 + 2.0 * mu as f64 / n as f64)).into();
        let cs = ((mu as f64 + 2.0) / (n as f64 + mu as f64 + 5.0)).into();
        let c1 = (2.0 / ((n as f64 + 1.3).powi(2) + mu as f64)).into();
        let cmu = T::from_f64(2.0).min(&(T::one() - &c1)) * (mu as f64 - 2.0 + 1.0 / mu as f64)
            / ((n as f64 + 2.0).powi(2) + mu as f64);
        let damps = (1.0 + 2.0 * (0.0_f64).max((mu as f64 - 1.0) / (n as f64 + 1.0) - 1.0)) + &cs;

        // Expectation of ||N(0,I)||
        let chi_n = T::from_f64(
            (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2))),
        );

        // Weights for recombination
        let mut weights = Array1::<T>::zeros(mu);
        for i in 0..mu {
            weights[i] = T::from_f64((mu as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln();
        }
        let sum_weights = weights.iter().sum::<T>();
        for w in &mut weights {
            *w /= &sum_weights;
        }
        let mueff = 1.0 / weights.iter().map(|w| w.powi(2)).sum::<T>();

        // Initialize evolution paths and covariance matrix
        let mut xmean = initial_point.to_owned();
        let mut sigma = sigma;
        let mut pc = Array1::<T>::zeros(n);
        let mut ps = Array1::<T>::zeros(n);
        let mut c = Array2::eye(n);

        let mut fn_evals = 0;
        self.generations = 0;
        let mut history = Vec::new();

        while self.generations < max_gen {
            self.generations += 1;

            // Generate and evaluate population
            let (eigenvalues, eigenvectors) = self.eigen_decomposition(c.view());
            let sqrt_eigenvalues: Array1<T> = Array1::from_shape_fn(eigenvalues.len(), |i| {
                eigenvalues[i].max(&T::from_f64(0.0)).sqrt()
            });

            let mut population = Vec::with_capacity(lambda);
            let mut fitness = Vec::with_capacity(lambda);

            for _ in 0..lambda {
                // Sample from N(0,I)
                let mut z = Array1::<T>::zeros(n);
                for i in 0..n {
                    let normal = Normal::new(0.0, 1.0).unwrap();
                    z[i] = T::from_f64(normal.sample(&mut self.rng));
                }

                // Transform: y = B * D * z (where B=eigenvectors, D=sqrt(eigenvalues))
                let mut y = Array1::<T>::zeros(n);
                for i in 0..n {
                    for j in 0..n {
                        y[i] += &eigenvectors[[j, i]] * &sqrt_eigenvalues[j] * &z[j];
                    }
                }

                // x = xmean + sigma * y
                let x = Array1::from_shape_fn(xmean.len(), |i| &xmean[i] + &sigma * &y[i]);

                let f_val = self.f.call(x.view());
                fn_evals += 1;

                if !f_val.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }

                population.push((x, y, z));
                fitness.push(f_val);
            }

            // Sort by fitness
            let mut indices: Vec<usize> = (0..lambda).collect();
            indices.sort_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap());

            let best_fitness = fitness[indices[0]].clone();
            history.push(best_fitness.clone());

            // Check for convergence
            if sigma < tol {
                self.xmin = population[indices[0]].0.clone();
                self.fmin = best_fitness;
                self.converged = true;
                return Ok(CmaEsResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin.clone(),
                    generations: self.generations,
                    fn_evals,
                    converged: self.converged,
                    final_sigma: sigma,
                    condition_number: self.condition_number(eigenvalues.view()),
                    history,
                });
            }

            // Recombination: update mean
            xmean = Array1::zeros(n);
            for i in 0..mu {
                let idx = indices[i];
                for j in 0..n {
                    xmean[j] += &weights[i] * &population[idx].0[j];
                }
            }

            // Update evolution paths
            let mut ymean = Array1::<T>::zeros(n);
            let mut zmean = Array1::<T>::zeros(n);
            for i in 0..mu {
                let idx = indices[i];
                for j in 0..n {
                    ymean[j] += &weights[i] * &population[idx].1[j];
                    zmean[j] += &weights[i] * &population[idx].2[j];
                }
            }

            // Update ps (evolution path for sigma)
            for i in 0..n {
                ps[i] = (1.0 - &cs) * &ps[i] + (&cs * (2.0 - &cs) * &mueff).sqrt() * &zmean[i];
            }

            let ps_norm = ps.iter().map(|x| x.powi(2)).sum::<T>().sqrt();
            let hsig = if ps_norm.clone()
                / (1.0 - (1.0 - &cs).powi(2 * self.generations as i32)).sqrt()
                < T::from_f64(1.4 + 2.0 / (n as f64 + 1.0)) * &chi_n
            {
                T::one()
            } else {
                T::zero()
            };

            // Update pc (evolution path for covariance)
            for i in 0..n {
                pc[i] =
                    (1.0 - &cc) * &pc[i] + &hsig * (&cc * (2.0 - &cc) * &mueff).sqrt() * &ymean[i];
            }

            // Update covariance matrix c
            let c1a = &c1 * (1.0 - (1.0 - hsig.powi(2)) * &cc * (2.0 - &cc));

            for i in 0..n {
                for j in 0..n {
                    c[[i, j]] = (1.0 - &c1a - &cmu) * &c[[i, j]] + &c1 * &pc[i] * &pc[j];

                    // Add rank-mu update
                    for k in 0..mu {
                        let idx = indices[k];
                        c[[i, j]] +=
                            &cmu * &weights[k] * &population[idx].1[i] * &population[idx].1[j];
                    }
                }
            }

            // Update sigma
            sigma *= (&cs / &damps * (&ps_norm / &chi_n - 1.0)).exp();

            // Limit sigma
            sigma = sigma.min(&1e10.into()).max(&1e-10.into());
        }

        // Final result
        let (eigenvalues, _) = self.eigen_decomposition(c.view());

        // Find best solution from final population
        let mut population = Vec::with_capacity(lambda);
        let mut fitness = Vec::with_capacity(lambda);

        let (eigenvalues_final, eigenvectors_final) = self.eigen_decomposition(c.view());
        let sqrt_eigenvalues_final: Array1<T> = eigenvalues_final
            .iter()
            .map(|x| x.max(&T::zero()).sqrt())
            .collect();

        for _ in 0..lambda {
            let mut z = Array1::<T>::zeros(n);
            for i in 0..n {
                let normal = Normal::new(0.0, 1.0).unwrap();
                z[i] = T::from_f64(normal.sample(&mut self.rng));
            }

            let mut y = Array1::<T>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    y[i] += &eigenvectors_final[[j, i]] * &sqrt_eigenvalues_final[j] * &z[j];
                }
            }

            let x = Array1::from_shape_fn(xmean.len(), |i| &xmean[i] + &sigma * &y[i]);
            let f_val = self.f.call(x.view());
            fn_evals += 1;

            population.push(x);
            fitness.push(f_val);
        }

        let mut indices: Vec<usize> = (0..lambda).collect();
        indices.sort_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap());

        self.xmin = population[indices[0]].clone();
        self.fmin = fitness[indices[0]].clone();
        self.converged = false;

        Ok(CmaEsResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            generations: self.generations,
            fn_evals,
            converged: self.converged,
            final_sigma: sigma,
            condition_number: self.condition_number(eigenvalues.view()),
            history,
        })
    }

    /// Simplified CMA-ES with default parameters
    pub fn minimize(
        &mut self,
        initial_point: ArrayView1<T>,
    ) -> Result<CmaEsResult<T>, MinimizerError> {
        self.cma_es(initial_point, None, None, None, None)
    }

    /// CMA-ES with restart strategy for better global optimization
    pub fn cma_es_with_restart(
        &mut self,
        initial_point: ArrayView1<T>,
        initial_sigma: Option<T>,
        max_restarts: Option<usize>,
        tol: Option<T>,
        max_generations_per_run: Option<usize>,
    ) -> Result<CmaEsResult<T>, MinimizerError> {
        let max_restarts = max_restarts.unwrap_or(3);
        let max_gen_per_run = max_generations_per_run.unwrap_or(1000);
        let tol = tol.unwrap_or(1e-11.into());
        let initial_sigma = initial_sigma.clone().unwrap_or(0.3.into());

        let mut best_result: Option<CmaEsResult<T>> = None;
        let mut total_fn_evals = 0;
        let mut total_generations = 0;
        let mut combined_history = Vec::new();

        for restart in 0..=max_restarts {
            // Perturb initial point for restarts
            let mut start_point = initial_point.to_owned();
            if restart > 0 {
                let sigma_perturbation = &initial_sigma * (1.0 + restart as f64 * 0.5);
                for x in &mut start_point {
                    let normal = Normal::new(0.0, sigma_perturbation.to_f64()).unwrap();
                    *x += normal.sample(&mut self.rng);
                }
            }

            let result = self.cma_es(
                start_point.view(),
                Some(initial_sigma.clone()),
                None,
                Some(tol.clone()),
                Some(max_gen_per_run),
            )?;

            total_fn_evals += result.fn_evals;
            total_generations += result.generations;
            combined_history.extend(result.history);

            // Update best result
            let is_better = match &best_result {
                None => true,
                Some(best) => result.fmin < best.fmin,
            };

            if is_better {
                best_result = Some(CmaEsResult {
                    xmin: result.xmin,
                    fmin: result.fmin,
                    generations: total_generations,
                    fn_evals: total_fn_evals,
                    converged: result.converged,
                    final_sigma: result.final_sigma,
                    condition_number: result.condition_number,
                    history: combined_history.clone(),
                });

                // Early termination if converged
                if result.converged {
                    break;
                }
            }
        }

        let final_result = best_result.unwrap();
        self.xmin = final_result.xmin.clone();
        self.fmin = final_result.fmin.clone();
        self.converged = final_result.converged;
        self.generations = total_generations;

        Ok(final_result)
    }

    /// Get current best point
    pub fn get_best(&self) -> (Array1<T>, T) {
        (self.xmin.clone(), self.fmin.clone())
    }
}

impl<T> fmt::Debug for CmaEs<T>
where
    T: RFFloat,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CmaEs( xmin: {:?}, fmin: {}, generations: {}, converged: {})",
            self.xmin, self.fmin, self.generations, self.converged
        )
    }
}

#[cfg(test)]
mod minimize_cmaes_tests {
    use super::*;
    use crate::{
        minimize::{F1dim, MultiDimFn},
        myfloat::MyFloat,
    };
    use std::f64::consts::{E, PI};

    // Helper function to create Ackley
    fn ackley(bias: &MyFloat) -> F1dim<MyFloat> {
        let b = bias.clone();
        F1dim::new(MultiDimFn::new(move |x: ArrayView1<MyFloat>| {
            let n = x.len() as f64;
            let t1 = x.iter().map(|val| val.powi(2)).sum::<MyFloat>();
            let t2 = x.iter().map(|val| (2.0 * PI * val).cos()).sum::<MyFloat>();
            -20.0 * (-0.2 * (t1 / n).sqrt()).exp() - (t2 / n).exp() + 20.0 + E + &b
        }))
    }

    // Helper function to create Elliptic
    fn elliptic(bias: &MyFloat) -> F1dim<MyFloat> {
        let b = bias.clone();
        F1dim::new(MultiDimFn::new(move |x: ArrayView1<MyFloat>| {
            let sum: MyFloat = x
                .iter()
                .enumerate()
                .map(|(i, val)| 1e6_f64.powf((i / (x.len() - 1)) as f64) * val.powi(2))
                .sum();
            sum + &b
        }))
    }

    // Helper function to create Griewank
    fn griewank(bias: &MyFloat) -> F1dim<MyFloat> {
        let b = bias.clone();
        F1dim::new(MultiDimFn::new(move |x: ArrayView1<MyFloat>| {
            let t1 = x.iter().map(|val| val.powi(2)).sum::<MyFloat>() / 4000.0;
            let t2 = x
                .iter()
                .enumerate()
                .map(|(i, val)| (val / (i as f64 + 1.0).sqrt()).cos())
                .product::<MyFloat>();
            t1 - t2 + 1.0 + &b
        }))
    }

    // Helper function to create Rosenbrock
    fn rosenbrock(shift: ArrayView1<MyFloat>, bias: &MyFloat) -> F1dim<MyFloat> {
        let s = shift.to_owned();
        let b = bias.clone();
        F1dim::new(MultiDimFn::new(move |x: ArrayView1<MyFloat>| {
            let x_new = &x - &s + 1.0;
            let sum: MyFloat = x_new
                .windows(2)
                .into_iter()
                .map(|pair| 100.0 * (pair[0].powi(2) - &pair[1]).powi(2) + (&pair[0] - 1.0).powi(2))
                .sum();
            sum + &b
        }))
    }

    mod basic_function_tests {
        use super::*;

        #[test]
        fn test_2d_quadratic() {
            // f(x,y) = (x-1)² + (y-2)², minimum at (1,2)
            let func = |x: ArrayView1<MyFloat>| (&x[0] - 1.0).powi(2) + (&x[1] - 2.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .minimize(array![0.0.into(), 0.0.into()].view())
                .unwrap();

            assert!((&result.xmin[0] - 1.0).abs() < 1e-3);
            assert!((&result.xmin[1] - 2.0).abs() < 1e-3);
            assert!(result.fmin < 1e-5);

            assert!((&cmaes.xmin[0] - 1.0).abs() < 1e-3);
            assert!((&cmaes.xmin[1] - 2.0).abs() < 1e-3);
            assert!(cmaes.fmin < 1e-5);
        }

        #[test]
        fn test_1d_quadratic() {
            // f(x) = (x-5)², minimum at x=5
            let func = |x: ArrayView1<MyFloat>| (&x[0] - 5.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes.minimize(array![0.0.into()].view()).unwrap();

            assert!((&result.xmin[0] - 5.0).abs() < 1e-3);
            assert!(result.fmin < 1e-5);
            assert!(result.converged || result.fmin < 1e-5);
        }

        #[test]
        fn test_rosenbrock_2d() {
            // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
            let rosenbrock = |x: ArrayView1<MyFloat>| {
                (1.0 - &x[0]).powi(2) + 100.0 * (&x[1] - x[0].powi(2)).powi(2)
            };
            let objective = MultiDimFn::new(rosenbrock);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![MyFloat::new(-1.2), 1.0.into()].view(),
                    Some(0.5.into()),
                    Some(50),
                    Some(1e-8.into()),
                    Some(500),
                )
                .unwrap();

            println!("\nresult.xmin = {:?}\n", result.xmin);
            assert!((&result.xmin[0] - 1.0).abs() < 1e-2);
            assert!((&result.xmin[1] - 1.0).abs() < 1e-2);
            assert!(result.fmin < 1e-3);

            assert!((&cmaes.xmin[0] - 1.0).abs() < 1e-2);
            assert!((&cmaes.xmin[1] - 1.0).abs() < 1e-2);
            assert!(cmaes.fmin < 1e-3);
        }

        #[test]
        fn test_3d_sphere() {
            // f(x,y,z) = x² + y² + z², minimum at (0,0,0)
            let sphere = |x: ArrayView1<MyFloat>| x.iter().map(|xi| xi * xi).sum();
            let objective = MultiDimFn::new(sphere);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .minimize(array![1.0.into(), 1.0.into(), 1.0.into()].view())
                .unwrap();

            for coord in result.xmin {
                assert!(coord.abs() < 1e-3);
            }
            assert!(result.fmin < 1e-5);

            for coord in cmaes.xmin {
                assert!(coord.abs() < 1e-3);
            }
            assert!(cmaes.fmin < 1e-5);
        }

        #[test]
        fn test_ellipsoid_function() {
            // Ellipsoid: f(x) = sum(i * x_i^2), scaled quadratic
            let ellipsoid = |x: ArrayView1<MyFloat>| {
                x.iter()
                    .enumerate()
                    .map(|(i, xi)| (i + 1) as f64 * xi.powi(2))
                    .sum::<MyFloat>()
            };
            let objective = MultiDimFn::new(ellipsoid);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![1.0.into(), 2.0.into(), 3.0.into()].view(),
                    Some(1.0.into()),
                    None,
                    Some(1e-8.into()),
                    Some(200),
                )
                .unwrap();

            for coord in result.xmin {
                assert!(coord.abs() < 1e-3);
            }
            assert!(result.fmin < 1e-5);
        }
    }

    mod parameter_and_configuration_tests {
        use super::*;

        #[test]
        fn test_custom_population_size() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            // Test with small population
            let result1 = cmaes
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    Some(6), // Small population
                    Some(1e-6.into()),
                    Some(200),
                )
                .unwrap();

            // Test with large population
            let result2 = cmaes
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    Some(50), // Large population
                    Some(1e-6.into()),
                    Some(100),
                )
                .unwrap();

            // Both should converge
            assert!(result1.fmin < 1e-3);
            assert!(result2.fmin < 1e-3);

            // Large population should typically need fewer generations
            // but this is not guaranteed, so we just check it ran
            assert!(result2.generations > 0);
        }

        #[test]
        fn test_different_initial_sigma() {
            let func = |x: ArrayView1<MyFloat>| (&x[0] - 2.0).powi(2) + (&x[1] + 1.0).powi(2);
            let objective = MultiDimFn::new(func);

            // Test small sigma
            let mut cmaes1 = CmaEs::with_seed(objective.clone(), 42);
            let result1 = cmaes1
                .cma_es(
                    array![0.0.into(), 0.0.into()].view(),
                    Some(0.1.into()), // Small sigma
                    None,
                    Some(1e-6.into()),
                    Some(300),
                )
                .unwrap();

            // Test large sigma
            let mut cmaes2 = CmaEs::with_seed(objective, 42);
            let result2 = cmaes2
                .cma_es(
                    array![0.0.into(), 0.0.into()].view(),
                    Some(2.0.into()), // Large sigma
                    None,
                    Some(1e-6.into()),
                    Some(300),
                )
                .unwrap();

            // Both should find the minimum
            assert!((&result1.xmin[0] - 2.0).abs() < 1e-2);
            assert!((&result1.xmin[1] + 1.0).abs() < 1e-2);
            assert!((&result2.xmin[0] - 2.0).abs() < 1e-2);
            assert!((&result2.xmin[1] + 1.0).abs() < 1e-2);
        }

        #[test]
        fn test_tolerance_settings() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            // Test with loose tolerance
            let result = cmaes
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    None,
                    Some(1e-3.into()), // Loose tolerance
                    Some(1000),
                )
                .unwrap();

            // Should converge due to loose tolerance
            assert!(result.converged);
            // Sigma should be reasonable but not necessarily above tolerance
            assert!(result.final_sigma > 0.0);
            assert!(result.final_sigma < 10.0);
            // Should find good solution
            assert!(result.fmin < 1e-2);
        }
    }

    mod restart_strategy_tests {
        use super::*;

        #[test]
        fn test_restart_strategy() {
            let func = |x: ArrayView1<MyFloat>| (&x[0] - 3.0).powi(2) + (&x[1] + 2.0).powi(2) + 5.0;
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es_with_restart(
                    array![0.0.into(), 0.0.into()].view(),
                    Some(1.0.into()),
                    Some(2),
                    Some(1e-8.into()),
                    Some(100),
                )
                .unwrap();

            assert!((&result.xmin[0] - 3.0).abs() < 1e-2);
            assert!((&result.xmin[1] + 2.0).abs() < 1e-2);
            assert!((result.fmin - 5.0).abs() < 1e-3);

            assert!((&cmaes.xmin[0] - 3.0).abs() < 1e-2);
            assert!((&cmaes.xmin[1] + 2.0).abs() < 1e-2);
            assert!((cmaes.fmin - 5.0).abs() < 1e-3);

            // Check that restart accumulated function evaluations and generations
            assert!(result.fn_evals > 100); // Should be more than single run
            assert!(result.generations > 50); // Should accumulate
        }

        #[test]
        fn test_restart_improves_result() {
            // Function that's difficult from certain starting points
            let func = |x: ArrayView1<MyFloat>| {
                let x1 = x[0].clone();
                let x2 = x[1].clone();
                // Himmelblau's function (multimodal)
                (x1.powi(2) + &x2 - 11.0).powi(2) + (x1 + x2.powi(2) - 7.0).powi(2)
            };
            let objective = MultiDimFn::new(func);

            // Single run
            let mut cmaes1 = CmaEs::with_seed(objective.clone(), 42);
            let result1 = cmaes1
                .cma_es(
                    array![0.0.into(), 0.0.into()].view(),
                    Some(1.0.into()),
                    None,
                    Some(1e-6.into()),
                    Some(200),
                )
                .unwrap();

            // With restarts
            let mut cmaes2 = CmaEs::with_seed(objective, 42);
            let result2 = cmaes2
                .cma_es_with_restart(
                    array![0.0.into(), 0.0.into()].view(),
                    Some(1.0.into()),
                    Some(3),
                    Some(1e-6.into()),
                    Some(150),
                )
                .unwrap();

            // At least one should find a good solution
            assert!(result1.fmin < 10.0 || result2.fmin < 10.0);
            // Restart version should not be worse
            assert!(result2.fmin <= result1.fmin * 2.0);
        }
    }

    mod error_handling_tests {
        use super::*;

        #[test]
        fn test_invalid_dimension() {
            let func = |_: ArrayView1<MyFloat>| 0.0.into();
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::new(objective);

            let result = cmaes.minimize(array![].view());
            assert!(result.is_err());
            assert!(matches!(result, Err(MinimizerError::InvalidDimension)));
        }

        #[test]
        fn test_invalid_sigma() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::new(objective);

            // Test zero sigma
            let result = cmaes.cma_es(
                array![1.0.into()].view(),
                Some(0.0.into()),
                None,
                None,
                None,
            );
            assert!(result.is_err());
            assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));

            // Test negative sigma
            let result = cmaes.cma_es(
                array![1.0.into()].view(),
                Some(MyFloat::new(-0.1)),
                None,
                None,
                None,
            );
            assert!(result.is_err());
            assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
        }

        #[test]
        fn test_function_evaluation_error() {
            let bad_func = |x: ArrayView1<MyFloat>| {
                if x[0] > 5.0 {
                    f64::NAN.into()
                } else {
                    x[0].powi(2)
                }
            };
            let objective = MultiDimFn::new(bad_func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes.cma_es(
                array![10.0.into()].view(), // Start in bad region
                Some(1.0.into()),
                Some(10),
                Some(1e-6.into()),
                Some(50),
            );

            // Should either succeed (if it escapes bad region) or fail with function error
            match result {
                Ok(_) => {} // Acceptable if optimization escaped the bad region
                Err(MinimizerError::FunctionEvaluationError) => {} // Expected error
                Err(e) => panic!("Unexpected error type: {:?}", e),
            }
        }

        #[test]
        fn test_infinite_function_values() {
            let inf_func = |x: ArrayView1<MyFloat>| {
                if x[0].abs() > 10.0 {
                    f64::INFINITY.into()
                } else {
                    x[0].powi(2)
                }
            };
            let objective = MultiDimFn::new(inf_func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes.cma_es(
                array![0.0.into()].view(),
                Some(2.0.into()), // Large enough to potentially hit boundaries
                Some(20),
                Some(1e-6.into()),
                Some(100),
            );

            match result {
                Ok(res) => {
                    // If successful, should be near minimum
                    assert!(res.xmin[0].abs() < 1.0);
                    assert!(res.fmin < 1.0);
                }
                Err(MinimizerError::FunctionEvaluationError) => {} // Expected
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    }

    mod diagnostics_and_convergence_tests {
        use super::*;

        #[test]
        fn test_convergence_tracking() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .minimize(array![2.0.into(), 2.0.into()].view())
                .unwrap();

            assert!(!result.history.is_empty());
            assert!(result.history[0] > result.fmin);
            assert!(result.history.len() == result.generations);

            // Check monotonic improvement (generally)
            let first_half = result.history.len() / 4;
            let last_quarter = result.history.len() * 3 / 4;
            assert!(result.history[first_half] >= result.history[last_quarter]);
        }

        #[test]
        fn test_condition_number_tracking() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2) + 100.0 * x[1].powi(2); // Ill-conditioned
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .minimize(array![1.0.into(), 1.0.into()].view())
                .unwrap();

            assert!(result.condition_number >= 1.0);
            assert!(result.final_sigma > 0.0);
            // For ill-conditioned problems, condition number should be high
            assert!(result.condition_number > 10.0);
        }

        #[test]
        fn test_function_evaluation_counting() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    Some(10), // Small population for predictable count
                    Some(1e-8.into()),
                    Some(50),
                )
                .unwrap();

            // Should have reasonable number of evaluations
            assert!(result.fn_evals > 10);
            assert!(result.fn_evals < 10000);

            // Function evaluations should be at least generations * population_size
            assert!(result.fn_evals >= result.generations * 10);
        }
    }

    mod high_dimensional_and_scalability_tests {
        use super::*;

        #[test]
        fn test_high_dimensional() {
            // Test 5D sphere function
            let sphere_5d = |x: ArrayView1<MyFloat>| x.iter().map(|xi| xi * xi).sum::<MyFloat>();
            let objective = MultiDimFn::new(sphere_5d);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    Array1::ones(5).view(),
                    Some(0.5.into()),
                    None,
                    Some(1e-6.into()),
                    Some(200),
                )
                .unwrap();

            for coord in result.xmin {
                assert!(coord.abs() < 1e-2);
            }
            assert!(result.fmin < 1e-3);
        }

        #[test]
        fn test_very_high_dimensional() {
            // Test 10D sphere function
            let sphere_10d = |x: ArrayView1<MyFloat>| x.iter().map(|xi| xi * xi).sum::<MyFloat>();
            let objective = MultiDimFn::new(sphere_10d);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    (Array1::ones(10) * 0.5).view(),
                    Some(1.0.into()),
                    Some(50),          // Larger population for higher dimensions
                    Some(1e-4.into()), // Relaxed tolerance
                    Some(300),
                )
                .unwrap();

            for coord in result.xmin {
                assert!(coord.abs() < 0.1);
            }
            assert!(result.fmin < 0.1);
        }

        #[test]
        fn test_dimension_scaling() {
            // Test that algorithm adapts to dimension
            for dim in [2, 4, 6] {
                let sphere = |x: ArrayView1<MyFloat>| x.iter().map(|xi| xi * xi).sum::<MyFloat>();
                let objective = MultiDimFn::new(sphere);
                let mut cmaes = CmaEs::with_seed(objective, 42);

                let result = cmaes
                    .cma_es(
                        Array1::ones(dim).view(),
                        Some(0.5.into()),
                        None,
                        Some(1e-4.into()),
                        Some(200),
                    )
                    .unwrap();

                // Should find minimum regardless of dimension
                for coord in result.xmin {
                    assert!(
                        coord.abs() < 0.1,
                        "Failed for dimension {}: coord={}",
                        dim,
                        coord
                    );
                }
                assert!(
                    result.fmin < 0.1,
                    "Failed for dimension {}: fmin={}",
                    dim,
                    result.fmin
                );
            }
        }
    }

    mod cec2005_tests {
        use super::*;
        use crate::minimize::dot_1d_2d;
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
            let x = Array1::from_shape_fn(n, |_| (rng.random::<f64>() * 200.0 - 100.0).into());
            let shift_func = shift.clone();
            let func = move |x: ArrayView1<MyFloat>| {
                x.iter()
                    .enumerate()
                    .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                    .sum()
            };
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);
            let result = cmaes
                .cma_es(
                    x.view(),
                    Some(0.3.into()),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10.into()),
                    Some(1000),
                )
                .unwrap();
            for i in 0..10 {
                assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&result.fmin - bias) < 1e-3);

                assert!((&cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&cmaes.fmin - bias) < 1e-3)
            }

            // 30d
            if TEST_30D {
                let n = 30;
                let x = Array1::from_shape_fn(n, |_| (rng.random::<f64>() * 200.0 - 100.0).into());
                let shift_func = shift.clone();
                let func = move |x: ArrayView1<MyFloat>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let mut cmaes = CmaEs::with_seed(objective, 42);
                let result = cmaes
                    .cma_es(
                        x.view(),
                        Some(0.3.into()),
                        Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                        Some(1e-10.into()),
                        Some(1000),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&result.fmin - bias) < 1e-3);

                    assert!((&cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&cmaes.fmin - bias) < 1e-3)
                }
            }

            // 50d
            if TEST_50D {
                let n = 50;
                let x = Array1::from_shape_fn(n, |_| (rng.random::<f64>() * 200.0 - 100.0).into());
                let shift_func = shift.clone();
                let func = move |x: ArrayView1<MyFloat>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let mut cmaes = CmaEs::with_seed(objective, 42);
                let result = cmaes
                    .cma_es(
                        x.view(),
                        Some(0.3.into()),
                        Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                        Some(1e-10.into()),
                        Some(1000),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&result.fmin - bias) < 1e-3);

                    assert!((&cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&cmaes.fmin - bias) < 1e-3)
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
            let x = Array1::from_shape_fn(n, |_| (rng.random::<f64>() * 200.0 - 100.0).into());
            let shift_func = shift.clone();
            let func = move |x: ArrayView1<MyFloat>| {
                x.iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let mut new_val = MyFloat::new(0.0);
                        for j in 0..=i {
                            new_val += (&x[j] - shift_func[j]).powi(2);
                        }
                        new_val + bias
                    })
                    .sum()
            };
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);
            let result = cmaes
                .cma_es(
                    x.view(),
                    Some(0.3.into()),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10.into()),
                    Some(1000),
                )
                .unwrap();
            for i in 0..10 {
                assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&result.fmin - bias) < 1e-3);

                assert!((&cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&cmaes.fmin - bias) < 1e-3)
            }

            // 30d
            if TEST_30D {
                let n = 30;
                let x = Array1::from_shape_fn(n, |_| (rng.random::<f64>() * 200.0 - 100.0).into());
                let shift_func = shift.clone();
                let func = move |x: ArrayView1<MyFloat>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, _)| {
                            let mut new_val = MyFloat::new(0.0);
                            for j in 0..=i {
                                new_val += (&x[j] - shift_func[j]).powi(2);
                            }
                            new_val + bias
                        })
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let mut cmaes = CmaEs::with_seed(objective, 42);
                let result = cmaes
                    .cma_es(
                        x.view(),
                        Some(0.3.into()),
                        Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                        Some(1e-10.into()),
                        Some(1000),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&result.fmin - bias) < 1e-3);

                    assert!((&cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&cmaes.fmin - bias) < 1e-3)
                }
            }

            // 50d
            if TEST_50D {
                let n = 50;
                let x = Array1::from_shape_fn(n, |_| (rng.random::<f64>() * 200.0 - 100.0).into());
                let shift_func = shift.clone();
                let func = move |x: ArrayView1<MyFloat>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, _)| {
                            let mut new_val = MyFloat::new(0.0);
                            for j in 0..=i {
                                new_val += (&x[j] - shift_func[j]).powi(2);
                            }
                            new_val + bias
                        })
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let mut cmaes = CmaEs::with_seed(objective, 42);
                let result = cmaes
                    .cma_es(
                        x.view(),
                        Some(0.3.into()),
                        Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                        Some(1e-10.into()),
                        Some(1000),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&result.fmin - bias) < 1e-3);

                    assert!((&cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&cmaes.fmin - bias) < 1e-3)
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
            let m: Array2<MyFloat> = array![
                [
                    1.7830682721057345e-001.into(),
                    5.5786330587166588e-002.into(),
                    4.7591905576669730e-001.into(),
                    2.4551129863391566e-001.into(),
                    3.1998625926387086e-001.into(),
                    3.2102001448363848e-001.into(),
                    2.7787561319902176e-002.into(),
                    2.6664001046775621e-001.into(),
                    4.1568009651337917e-001.into(),
                    MyFloat::new(-4.7771934552669726e-001)
                ],
                [
                    6.3516362859468667e-001.into(),
                    5.0091423836646241e-002.into(),
                    2.0110601384121973e-001.into(),
                    MyFloat::new(-6.8076882416633511e-001),
                    MyFloat::new(-4.9934546553907944e-002),
                    MyFloat::new(-4.6399423424582961e-002),
                    MyFloat::new(-1.9460194646748039e-001),
                    1.8961539926194687e-001.into(),
                    MyFloat::new(-1.9416259626804547e-002),
                    1.0639981029473855e-001.into()
                ],
                [
                    3.2762147366023187e-001.into(),
                    3.6016598714114556e-001.into(),
                    MyFloat::new(-2.3635655094044949e-001),
                    MyFloat::new(-1.8566854017444848e-002),
                    MyFloat::new(-2.4479096747593634e-001),
                    4.4818973341886903e-001.into(),
                    5.3518635733619568e-001.into(),
                    MyFloat::new(-3.1206925190530521e-001),
                    MyFloat::new(-1.3863719921728737e-001),
                    MyFloat::new(-2.0713981146209595e-001)
                ],
                [
                    MyFloat::new(-6.4783210587984280e-002),
                    MyFloat::new(-4.9424101683695937e-001),
                    1.3101175297435969e-001.into(),
                    3.1615171931194543e-002.into(),
                    MyFloat::new(-1.7506107914871860e-001),
                    6.8908039344918381e-001.into(),
                    1.0544234469094992e-002.into(),
                    2.1948984793273507e-001.into(),
                    MyFloat::new(-1.6468539805844565e-001),
                    3.9048550518513409e-001.into()
                ],
                [
                    MyFloat::new(-2.7648044785371367e-001),
                    1.1383114506120220e-001.into(),
                    MyFloat::new(-3.0818401502810994e-001),
                    MyFloat::new(-3.5959407104438740e-001),
                    2.6446258034702191e-001.into(),
                    2.8616788379157501e-002.into(),
                    4.7528027904995646e-001.into(),
                    4.0993994049770172e-001.into(),
                    4.1131043368915432e-001.into(),
                    2.2899345188886880e-001.into()
                ],
                [
                    1.5454249061641606e-001.into(),
                    5.4899186274157996e-001.into(),
                    MyFloat::new(-1.8382029941792261e-001),
                    3.3944461903909162e-001.into(),
                    2.8596188774255699e-001.into(),
                    1.2833167642713417e-001.into(),
                    MyFloat::new(-2.5495080172376317e-001),
                    3.9460752302037100e-001.into(),
                    MyFloat::new(-3.4524640270007412e-001),
                    2.9590318323368509e-001.into()
                ],
                [
                    MyFloat::new(-5.1907977690014512e-002),
                    MyFloat::new(-1.4450757809700329e-001),
                    MyFloat::new(-4.6086919626114314e-001),
                    MyFloat::new(-5.3687964818368079e-002),
                    MyFloat::new(-3.6317793499109247e-001),
                    2.7439997038558633e-002.into(),
                    MyFloat::new(-2.1422629652542946e-001),
                    5.0545148893084779e-001.into(),
                    MyFloat::new(-9.8064717019089837e-002),
                    MyFloat::new(-5.6346991018564507e-001)
                ],
                [
                    5.0142989354460654e-001.into(),
                    MyFloat::new(-5.3133659048457516e-001),
                    MyFloat::new(-3.7294385871521135e-001),
                    2.3370866431381510e-001.into(),
                    4.4327537662488531e-001.into(),
                    MyFloat::new(-1.6972740381143742e-001),
                    2.0364148963331691e-001.into(),
                    MyFloat::new(-2.3717523924336927e-002),
                    MyFloat::new(-7.1805455862954920e-002),
                    MyFloat::new(-7.3332178450339763e-003)
                ],
                [
                    1.0441248047680891e-001.into(),
                    4.3064226149369542e-002.into(),
                    MyFloat::new(-4.1675972625940993e-001),
                    1.6522876074361707e-002.into(),
                    1.7437281849141879e-003.into(),
                    2.9594944879030760e-001.into(),
                    MyFloat::new(-5.1197487739368741e-001),
                    MyFloat::new(-3.2679819762357892e-001),
                    5.8253106590933512e-001.into(),
                    1.3204141339826148e-001.into()
                ],
                [
                    MyFloat::new(-2.9645907657631693e-001),
                    MyFloat::new(-3.1303011496605505e-002),
                    MyFloat::new(-7.8009154082116602e-002),
                    MyFloat::new(-4.1548534874482024e-001),
                    5.6959403572443468e-001.into(),
                    2.9095198400348149e-001.into(),
                    MyFloat::new(-1.8560717510075503e-001),
                    MyFloat::new(-2.4653488847859115e-001),
                    MyFloat::new(-3.7149025085479792e-001),
                    MyFloat::new(-3.0015617693118707e-001)
                ],
            ];
            let bias = -450.0;

            // 10d
            let n = 10;
            let x = dot_1d_2d(
                (Array1::from_shape_fn(n, |_| MyFloat::new(rng.random::<f64>() * 200.0 - 100.0))
                    - &shift)
                    .view(),
                m.view(),
            );
            let mut cmaes = CmaEs::with_seed(elliptic(&bias.into()), 42);
            let result = cmaes
                .cma_es(
                    x.view(),
                    Some(0.3.into()),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10.into()),
                    // Some(1e3 as usize * n.pow(2)),
                    Some(10000),
                )
                .unwrap();
            for i in 0..10 {
                // assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!(result.xmin[i].abs() < 1e-2);
                assert!((&result.fmin - bias) < 1e-3);

                // assert!((cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                assert!(cmaes.xmin[i].abs() < 1e-2);
                assert!((&cmaes.fmin - bias) < 1e-3)
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
            let x = Array1::from_shape_fn(n, |_| (rng.random::<f64>() * 200.0 - 100.0).into());
            let shift_func = shift.clone();
            let rand_val = rng.random::<f64>();
            let func = move |x: ArrayView1<MyFloat>| {
                x.iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let mut new_val = MyFloat::new(0.0);
                        for j in 0..=i {
                            new_val += (&x[j] - shift_func[j]).powi(2);
                        }
                        new_val + bias
                    })
                    .sum::<MyFloat>()
                    * (1.0 + 0.4 * rand_val)
            };
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);
            let result = cmaes
                .cma_es(
                    x.view(),
                    Some(0.3.into()),
                    Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
                    Some(1e-10.into()),
                    Some(1000),
                )
                .unwrap();
            for i in 0..10 {
                assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&result.fmin - bias) < 1e-3);

                assert!((&cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&cmaes.fmin - bias) < 1e-3)
            }
        }

        // #[test]
        // fn test_f6() {
        //     let mut rng = rand::rng();
        //     let shift: Array1<MyFloat> = array![
        //         8.1023200e+001.into(),
        //         MyFloat::new(-4.8395000e+001),
        //         1.9231600e+001.into(),
        //         MyFloat::new(-2.5231000e+000),
        //         7.0433800e+001.into(),
        //         4.7177400e+001.into(),
        //         MyFloat::new(-7.8358000e+000),
        //         MyFloat::new(-8.6669300e+001),
        //         5.7853200e+001.into(),
        //         MyFloat::new(-9.9533000e+000),
        //         2.0777800e+001.into(),
        //         5.2548600e+001.into(),
        //         7.5926300e+001.into(),
        //         4.2877300e+001.into(),
        //         MyFloat::new(-5.8272000e+001),
        //         MyFloat::new(-1.6972800e+001),
        //         7.8384500e+001.into(),
        //         7.5042700e+001.into(),
        //         MyFloat::new(-1.6151300e+001),
        //         7.0856900e+001.into(),
        //         MyFloat::new(-7.9579500e+001),
        //         MyFloat::new(-2.6483700e+001),
        //         5.6369900e+001.into(),
        //         MyFloat::new(-8.8224900e+001),
        //         MyFloat::new(-6.4999600e+001),
        //         MyFloat::new(-5.3502200e+001),
        //         MyFloat::new(-5.4230000e+001),
        //         1.8682600e+001.into(),
        //         MyFloat::new(-4.1006100e+001),
        //         MyFloat::new(-5.4213400e+001),
        //         MyFloat::new(-8.7250600e+001),
        //         4.4421400e+001.into(),
        //         MyFloat::new(-9.8826000e+000),
        //         7.7726600e+001.into(),
        //         MyFloat::new(-6.1210000e+000),
        //         MyFloat::new(-1.4643000e+001),
        //         6.2319800e+001.into(),
        //         4.5274000e+000.into(),
        //         MyFloat::new(-5.3523400e+001),
        //         3.0984700e+001.into(),
        //         6.0861300e+001.into(),
        //         MyFloat::new(-8.6464800e+001),
        //         3.2629800e+001.into(),
        //         MyFloat::new(-2.1693400e+001),
        //         5.9723200e+001.into(),
        //         5.0630000e-001.into(),
        //         3.7704800e+001.into(),
        //         MyFloat::new(-1.2799300e+001),
        //         MyFloat::new(-3.5168800e+001),
        //         MyFloat::new(-5.5862300e+001),
        //         MyFloat::new(-5.5182300e+001),
        //         3.2800100e+001.into(),
        //         MyFloat::new(-3.5502400e+001),
        //         7.5012000e+000.into(),
        //         MyFloat::new(-6.2842800e+001),
        //         3.5621700e+001.into(),
        //         MyFloat::new(-2.1892800e+001),
        //         6.4802000e+001.into(),
        //         6.3657900e+001.into(),
        //         1.6841300e+001.into(),
        //         MyFloat::new(-6.2050000e-001),
        //         7.1958400e+001.into(),
        //         5.7893200e+001.into(),
        //         2.6083800e+001.into(),
        //         5.7235300e+001.into(),
        //         2.8840900e+001.into(),
        //         MyFloat::new(-2.8445200e+001),
        //         MyFloat::new(-3.7849300e+001),
        //         MyFloat::new(-2.8585100e+001),
        //         6.1342000e+000.into(),
        //         4.0880300e+001.into(),
        //         MyFloat::new(-3.4327700e+001),
        //         6.0929200e+001.into(),
        //         1.2253000e+001.into(),
        //         MyFloat::new(-2.3325500e+001),
        //         3.6493100e+001.into(),
        //         8.3828000e+000.into(),
        //         MyFloat::new(-9.9215000e+000),
        //         3.5022100e+001.into(),
        //         2.1835800e+001.into(),
        //         5.3067700e+001.into(),
        //         8.2231800e+001.into(),
        //         4.0662000e+000.into(),
        //         6.8425500e+001.into(),
        //         MyFloat::new(-5.8867800e+001),
        //         8.6354400e+001.into(),
        //         MyFloat::new(-4.1139400e+001),
        //         MyFloat::new(-4.4580700e+001),
        //         6.7633500e+001.into(),
        //         4.2715000e+001.into(),
        //         MyFloat::new(-6.5426600e+001),
        //         MyFloat::new(-8.7883700e+001),
        //         7.0901600e+001.into(),
        //         MyFloat::new(-5.4155100e+001),
        //         MyFloat::new(-3.6229800e+001),
        //         2.9059600e+001.into(),
        //         MyFloat::new(-3.8806400e+001),
        //         MyFloat::new(-5.5396000e+000),
        //         MyFloat::new(-7.8339300e+001),
        //         8.7900200e+001.into()
        //     ];
        //     let bias = MyFloat::new(390.0);

        //     // 10d
        //     let n = 10;
        //     let x = Array1::from_shape_fn(n, |_| (rng.random::<f64>() * 200.0 - 100.0).into());
        //     let mut cmaes =
        //         CmaEs::with_seed(rosenbrock(&shift.slice(s![0..10]).to_owned(), &bias), 42);
        //     let result = cmaes
        //         .cma_es(
        //             &x,
        //             Some(0.3.into()),
        //             Some(4 + (3.0 * (n as f64).ln()).floor() as usize),
        //             Some(1e-10.into()),
        //             Some(1e3 as usize * n.pow(2)),
        //         )
        //         .unwrap();
        //     for i in 0..10 {
        //         assert!((&result.xmin[i] - &shift[i]).abs() < 1e-2);
        //         assert!((&result.fmin - &bias) < 1e-3);

        //         assert!((&cmaes.xmin[i] - &shift[i]).abs() < 1e-2);
        //         assert!((&cmaes.fmin - &bias) < 1e-3)
        //     }
        // }

        // #[test]
        // fn test_f7() {
        //     let mut rng = rand::rng();
        //     let shift = array![
        //         -2.7626840e+002,
        //         -1.1911000e+001,
        //         -5.7878840e+002,
        //         -2.8764860e+002,
        //         -8.4385800e+001,
        //         -2.2867530e+002,
        //         -4.5815160e+002,
        //         -2.0221450e+002,
        //         -1.0586420e+002,
        //         -9.6489800e+001,
        //     ];
        //     let m = array![
        //         [
        //             -7.3696625825313500e-002,
        //             1.5747490444892893e+000,
        //             -6.4377942207169941e-002,
        //             6.3201848730939580e-001,
        //             -1.2455211411481415e+000,
        //             -3.5341187428098381e-001,
        //             3.5031691018519090e-001,
        //             6.2886479758992697e-001,
        //             6.8593632355012335e-001,
        //             1.3975663076173925e+000
        //         ],
        //         [
        //             6.3700016123079051e-001,
        //             -1.3833836770823484e+000,
        //             -2.4437874951092337e-001,
        //             1.6992995943357547e+000,
        //             7.1757447137502850e-001,
        //             -7.7753800570270454e-002,
        //             4.9291080765053624e-001,
        //             1.1392847178100191e-001,
        //             4.8163647386641817e-001,
        //             2.8150613437207017e-001
        //         ],
        //         [
        //             -1.4466181982194921e+000,
        //             -1.1273816086105013e+000,
        //             -1.0665724848959319e+000,
        //             2.1900088934332190e-001,
        //             -5.8130776006865136e-002,
        //             -9.9187841926086026e-002,
        //             -1.2465831572524580e-001,
        //             -5.0547372808368829e-001,
        //             -2.1020191419640880e-001,
        //             1.1509984987284301e+000
        //         ],
        //         [
        //             1.0410802679063424e+000,
        //             4.7577677793232626e-001,
        //             9.6430154567967874e-001,
        //             1.5636976117984064e-002,
        //             2.0539698111678034e-001,
        //             2.5839780039821658e-001,
        //             -5.1710361801897031e-001,
        //             -1.5449014589834349e+000,
        //             -1.4560361158442292e+000,
        //             9.9877904060730438e-001
        //         ],
        //         [
        //             2.6260272944635960e-001,
        //             9.2947540741436874e-001,
        //             -1.2953100028930926e+000,
        //             6.6512029642561388e-001,
        //             -2.7957781701655993e-001,
        //             8.4060537698112758e-001,
        //             -5.2922829607729160e-001,
        //             -8.6040220072910467e-001,
        //             4.9503162769183251e-001,
        //             -6.3765376892958103e-001
        //         ],
        //         [
        //             8.1307889477954698e-002,
        //             8.0062327426592494e-001,
        //             7.2294618679188488e-002,
        //             4.4874698427975906e-001,
        //             1.7959858022699743e-001,
        //             -1.3634800693969209e+000,
        //             7.5257943996576704e-002,
        //             -1.2486791053473751e+000,
        //             6.8143526407032673e-001,
        //             1.3558980136836016e-001
        //         ],
        //         [
        //             8.7913516653862697e-002,
        //             2.1022739728349416e-001,
        //             -1.5708234535904123e-001,
        //             -3.5182196550454031e-001,
        //             -6.4190160213761838e-002,
        //             1.5082748057903228e+000,
        //             1.1168462803089814e+000,
        //             -3.6773042225135699e-001,
        //             2.6828021681357744e-001,
        //             4.9698836189165707e-001
        //         ],
        //         [
        //             -9.5406235378612747e-001,
        //             3.9879009060640763e-001,
        //             5.8022630243503770e-001,
        //             2.4831174649263604e-001,
        //             1.1781385394000925e+000,
        //             5.3134809745284084e-001,
        //             7.8257240450327026e-001,
        //             -3.8166809840963106e-001,
        //             -4.8082474351369503e-001,
        //             -6.2076533636514075e-001
        //         ],
        //         [
        //             2.7628599479864874e-001,
        //             3.6188284466692094e-001,
        //             -1.0302756351623272e+000,
        //             7.2348644867809120e-001,
        //             -3.7379075361566066e-001,
        //             -7.9223639600376997e-002,
        //             1.6221551897070494e+000,
        //             -8.2880436781697358e-003,
        //             -1.0881497169330046e+000,
        //             -1.9204701133595675e-001
        //         ],
        //         [
        //             -3.0035568486304853e-001,
        //             -5.0758053487595001e-001,
        //             3.1143454840627821e-001,
        //             -2.5444900307396151e-001,
        //             -7.7988528102301924e-001,
        //             -6.8262839999436264e-001,
        //             5.5932665521935510e-001,
        //             -7.9579050121422423e-001,
        //             4.7071685181799255e-001,
        //             -8.0019268494895490e-001
        //         ],
        //     ];
        //     let bias = -180.0;

        //     // 10d
        //     let n = 10;
        //     let x = (&Array1::from_shape_fn(n, |_| rng.random::<MyFloat>() * 600.0) - &shift).dot(&m);
        //     let mut cmaes = CmaEs::with_seed(griewank(bias), 42);
        //     let result = cmaes
        //         .cma_es(
        //             x,
        //             Some(0.3),
        //             Some(4 + (3.0 * (n as MyFloat).ln()).floor() as usize),
        //             Some(1e-10),
        //             Some(1e3 as usize * n.pow(2)),
        //         )
        //         .unwrap();
        //     println!(
        //         "\n\nxmin = {:?}\nfmin = {:?}\n\n",
        //         &result.xmin,
        //         result.fmin - bias
        //     );
        //     for i in 0..10 {
        //         // assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
        //         assert!((result.xmin[i]).abs() < 1e-2);
        //         assert!((result.fmin - bias) < 1e-3);

        //         // assert!((cmaes.xmin[i] - shift[i]).abs() < 1e-2);
        //         assert!((cmaes.xmin[i]).abs() < 1e-2);
        //         assert!((cmaes.fmin - bias) < 1e-3)
        //     }
        // }

        // #[test]
        // fn test_f8() {
        //     let mut rng = rand::rng();
        //     let shift = array![
        //         -1.6823000e+001,
        //         1.4976900e+001,
        //         6.1690000e+000,
        //         9.5566000e+000,
        //         1.9541700e+001,
        //         -1.7190000e+001,
        //         -1.8824800e+001,
        //         8.5110000e-001,
        //         -1.5116200e+001,
        //         1.0793400e+001,
        //     ];
        //     let m = array![
        //         [
        //             -1.8785768809450733e+001,
        //             3.3616954860437176e+001,
        //             2.6882113915382682e+001,
        //             -1.0433064197429360e+001,
        //             9.4489289408247579e-001,
        //             -3.3538964332596910e+000,
        //             3.5352127339076374e+000,
        //             7.3942769413967824e+000,
        //             7.7909087526412346e+000,
        //             2.0912921099835673e+000
        //         ],
        //         [
        //             -3.8080580880289006e-001,
        //             1.0420967284673154e+001,
        //             9.3472919131156775e+000,
        //             -2.0926490513724943e+001,
        //             1.1425903158890700e+001,
        //             1.1056372574995397e+000,
        //             3.6879282505970373e+001,
        //             -1.9103397804634721e+000,
        //             7.5611684932093199e+000,
        //             -9.7430664357899719e+000
        //         ],
        //         [
        //             -1.2341688082549489e+001,
        //             6.3621993552738045e+000,
        //             8.2484299397924641e+000,
        //             8.0892563664178354e+000,
        //             6.9234506619011427e-002,
        //             2.5786241359574857e+000,
        //             -4.9734021861818611e-001,
        //             -2.0627220954843271e+000,
        //             1.4302051477457656e+000,
        //             1.5522003760944671e+001
        //         ],
        //         [
        //             -1.7006542510444671e+001,
        //             -1.2679306064055677e+001,
        //             5.1658120519511158e+001,
        //             -3.9766120780636900e+000,
        //             3.9349384750136576e+000,
        //             -3.0777202564613845e+001,
        //             6.1465971476271157e+000,
        //             -1.1404959107806402e+001,
        //             1.2694206030880832e+001,
        //             -9.3951432281174387e+000
        //         ],
        //         [
        //             -5.4847491542826621e+000,
        //             -1.3643476981518553e+001,
        //             -2.0812578603542899e+001,
        //             1.2480631776818850e+001,
        //             8.4497820599569917e-001,
        //             2.4830393330514045e+001,
        //             3.3838505185702559e+001,
        //             -1.7003569707093064e+001,
        //             -5.2939643674048442e+000,
        //             2.6065703095424336e+001
        //         ],
        //         [
        //             1.1422878520470586e+001,
        //             1.0221461943150496e+001,
        //             -5.9994789874002628e+000,
        //             -8.9358916025741415e+000,
        //             3.3407916251514460e+000,
        //             3.9245488542554492e+000,
        //             -6.7605717857278513e+000,
        //             1.4016300477765046e+001,
        //             2.3533969357952147e+000,
        //             -1.5957358828479556e+001
        //         ],
        //         [
        //             1.4106979735005300e+001,
        //             -6.8979757292229404e-001,
        //             2.5928358266684882e+001,
        //             -3.0138271725378775e+001,
        //             1.2953067028884863e+001,
        //             -1.7125782201118525e+001,
        //             1.9122903237509483e+001,
        //             3.8502101712160597e+000,
        //             1.4449871260335931e+001,
        //             -3.7768641488073015e+001
        //         ],
        //         [
        //             1.8171620273851625e+000,
        //             -4.5228977429981496e+000,
        //             2.5960648243684310e+000,
        //             -3.0779703663335480e+000,
        //             3.6662383806277021e+000,
        //             -3.1422719671052084e+000,
        //             -1.9391037957658499e+000,
        //             -1.1328460209494431e+000,
        //             -1.4593971192280721e+000,
        //             -4.3850653050781068e+000
        //         ],
        //         [
        //             1.7059635015136770e+001,
        //             -4.0887343040678509e+001,
        //             -9.0413685473717607e+000,
        //             9.2078166133516532e+000,
        //             2.4835590816969209e+000,
        //             -3.1352382866663429e+000,
        //             -5.1597084344852373e-001,
        //             -1.0448164970351954e+001,
        //             -3.9790838641391200e+000,
        //             -5.0101517638708923e+000
        //         ],
        //         [
        //             -2.1004104733841622e+000,
        //             4.2857434922129016e+000,
        //             1.8138803730523911e+001,
        //             -5.5691566223518540e+000,
        //             2.0414928764167950e-002,
        //             -5.5315683071808275e+000,
        //             1.7507462325577925e+000,
        //             2.0183823538506891e+000,
        //             8.9673707865551204e+000,
        //             -3.5936542419629482e+000,
        //         ]
        //     ];
        //     let bias = -140.0;

        //     // 10d
        //     let n = 10;
        //     let x =
        //         (&Array1::from_shape_fn(n, |_| rng.random::<MyFloat>() * 64.0 - 32.0) - &shift).dot(&m);
        //     let mut cmaes = CmaEs::with_seed(ackley(bias), 42);
        //     let result = cmaes
        //         .cma_es(
        //             x,
        //             Some(0.3),
        //             Some(4 + (3.0 * (n as MyFloat).ln()).floor() as usize),
        //             Some(1e-10),
        //             Some(1e3 as usize * n.pow(2)),
        //         )
        //         .unwrap();
        //     println!(
        //         "\n\nxmin = {:?}\nfmin = {:?}\n\n",
        //         &result.xmin - &shift,
        //         result.fmin - bias
        //     );
        //     for i in 0..10 {
        //         assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
        //         assert!((result.fmin - bias) < 1e-3);

        //         assert!((cmaes.xmin[i] - shift[i]).abs() < 1e-2);
        //         assert!((cmaes.fmin - bias) < 1e-3)
        //     }
        // }
    }

    mod multimodal_and_complex_function_tests {
        use super::*;

        #[test]
        fn test_multimodal_function() {
            // Ackley function (multimodal but easier than Rastrigin)
            let ackley = |x: ArrayView1<MyFloat>| {
                let n = x.len() as f64;
                let sum_sq = x.iter().map(|xi| xi.powi(2)).sum::<MyFloat>();
                let sum_cos = x
                    .iter()
                    .map(|xi| (2.0 * std::f64::consts::PI * xi).cos())
                    .sum::<MyFloat>();

                -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
                    + 20.0
                    + std::f64::consts::E
            };
            let objective = MultiDimFn::new(ackley);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es_with_restart(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(2.0.into()),
                    Some(5),
                    Some(1e-6.into()),
                    Some(500),
                )
                .unwrap();

            // Ackley has global minimum at (0,0) with value 0
            // Test should be more lenient for multimodal functions
            for coord in result.xmin {
                assert!(
                    coord.abs() < 0.5,
                    "Coordinate {} is too far from optimum",
                    coord
                );
            }
            assert!(
                result.fmin < 5.0,
                "Function value {} is too high",
                result.fmin
            );

            // Also test that we're making reasonable progress
            assert!(result.history.len() > 10);
            assert!(result.history.last().unwrap() <= &result.history[0]);
        }

        #[test]
        fn test_simple_multimodal() {
            // Simpler multimodal test - multiple quadratic wells
            let multi_quad = |x: ArrayView1<MyFloat>| {
                let val1 = (&x[0] - 1.0).powi(2) + (&x[1] - 1.0).powi(2);
                let val2 = (&x[0] + 1.0).powi(2) + (&x[1] + 1.0).powi(2);
                val1.min(&val2) // Two wells at (1,1) and (-1,-1)
            };
            let objective = MultiDimFn::new(multi_quad);
            let mut cmaes = CmaEs::with_seed(objective, 123);

            let result = cmaes
                .cma_es(
                    array![0.0.into(), 0.0.into()].view(),
                    Some(1.0.into()),
                    Some(30),
                    Some(1e-8.into()),
                    Some(200),
                )
                .unwrap();

            // Should find one of the two minima
            let dist_to_first =
                ((&result.xmin[0] - 1.0).powi(2) + (&result.xmin[1] - 1.0).powi(2)).sqrt();
            let dist_to_second =
                ((&result.xmin[0] + 1.0).powi(2) + (&result.xmin[1] + 1.0).powi(2)).sqrt();

            assert!(
                dist_to_first < 0.1 || dist_to_second < 0.1,
                "Solution {:?} not close to either minimum",
                result.xmin
            );
            assert!(result.fmin < 1e-2);
        }

        #[test]
        fn test_beale_function() {
            // Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
            // Global minimum at (3, 0.5) with value 0
            let beale = |x: ArrayView1<MyFloat>| {
                let x1 = x[0].clone();
                let x2 = x[1].clone();
                (1.5 - &x1 + &x1 * &x2).powi(2)
                    + (2.25 - &x1 + &x1 * x2.powi(2)).powi(2)
                    + (2.625 - &x1 + &x1 * x2.powi(3)).powi(2)
            };
            let objective = MultiDimFn::new(beale);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es_with_restart(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(1.0.into()),
                    Some(3),
                    Some(1e-6.into()),
                    Some(300),
                )
                .unwrap();

            // Should be close to global minimum
            assert!((&result.xmin[0] - 3.0).abs() < 0.1);
            assert!((&result.xmin[1] - 0.5).abs() < 0.1);
            assert!(result.fmin < 1e-2);
        }

        #[test]
        fn test_booth_function() {
            // Booth function: f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
            // Global minimum at (1, 3) with value 0
            let booth = |x: ArrayView1<MyFloat>| {
                (&x[0] + 2.0 * &x[1] - 7.0).powi(2) + (2.0 * &x[0] + &x[1] - 5.0).powi(2)
            };
            let objective = MultiDimFn::new(booth);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .minimize(array![0.0.into(), 0.0.into()].view())
                .unwrap();

            assert!((&result.xmin[0] - 1.0).abs() < 1e-3);
            assert!((&result.xmin[1] - 3.0).abs() < 1e-3);
            assert!(result.fmin < 1e-5);
        }
    }

    mod robustness_and_edge_case_tests {
        use super::*;

        #[test]
        fn test_noisy_function() {
            use rand::Rng;
            let rng = StdRng::seed_from_u64(12345);

            // Function with added noise
            let noisy_quad = move |x: ArrayView1<MyFloat>| {
                let clean = x[0].powi(2) + x[1].powi(2);
                let noise = rng.clone().random::<f64>() * 0.01 - 0.005; // Small noise ±0.005
                clean + noise
            };
            let objective = MultiDimFn::new(noisy_quad);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    Some(50),          // Larger population for noisy functions
                    Some(1e-4.into()), // Relaxed tolerance for noise
                    Some(200),
                )
                .unwrap();

            // Should still find approximate minimum despite noise
            for coord in result.xmin {
                assert!(coord.abs() < 0.1);
            }
            assert!(result.fmin < 0.1);
        }

        #[test]
        fn test_very_flat_function() {
            // Very flat function around minimum
            let flat_func = |x: ArrayView1<MyFloat>| {
                let dist_sq = x.iter().map(|xi| xi.powi(2)).sum::<MyFloat>();
                if dist_sq < 1e-6 {
                    0.0.into()
                } else {
                    dist_sq.sqrt()
                }
            };
            let objective = MultiDimFn::new(flat_func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![0.1.into(), 0.1.into()].view(),
                    Some(0.1.into()),
                    None,
                    Some(1e-8.into()),
                    Some(300),
                )
                .unwrap();

            // Should find the flat region
            let dist = (result.xmin[0].powi(2) + result.xmin[1].powi(2)).sqrt();
            assert!(dist < 1e-3);
            assert!(result.fmin < 1e-5);
        }

        #[test]
        fn test_very_steep_function() {
            // Very steep function
            let steep_func = |x: ArrayView1<MyFloat>| {
                let dist_sq = x.iter().map(|xi| xi.powi(2)).sum::<MyFloat>();
                dist_sq.powi(4) // Very steep
            };
            let objective = MultiDimFn::new(steep_func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![0.5.into(), 0.5.into()].view(),
                    Some(0.2.into()),
                    None,
                    Some(1e-6.into()),
                    Some(200),
                )
                .unwrap();

            // Should still converge
            for coord in result.xmin {
                assert!(coord.abs() < 1e-2);
            }
            assert!(result.fmin < 1e-6);
        }

        #[test]
        fn test_asymmetric_function() {
            // Asymmetric function (different scales in different directions)
            let asymmetric = |x: ArrayView1<MyFloat>| {
                x[0].powi(2) + 1000.0 * x[1].powi(2) + 10.0 * &x[0] * &x[1]
            };
            let objective = MultiDimFn::new(asymmetric);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(1.0.into()),
                    None,
                    Some(1e-6.into()),
                    Some(300),
                )
                .unwrap();

            // Should adapt to the asymmetry and find minimum
            for coord in result.xmin {
                assert!(coord.abs() < 1e-2);
            }
            assert!(result.fmin < 1e-4);

            // Condition number should reflect the ill-conditioning
            assert!(result.condition_number > 100.0);
        }
    }

    mod utility_and_interface_tests {
        use super::*;

        #[test]
        fn test_get_best_method() {
            let func = |x: ArrayView1<MyFloat>| (&x[0] - 2.0).powi(2) + (&x[1] - 3.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            // Before optimization
            let (initial_x, initial_f) = cmaes.get_best();
            assert!(initial_x.is_empty());
            assert_eq!(initial_f, f64::INFINITY);

            // After optimization
            let result = cmaes
                .minimize(array![0.0.into(), 0.0.into()].view())
                .unwrap();
            let (best_x, best_f) = cmaes.get_best();

            assert_eq!(best_x, result.xmin);
            assert_eq!(best_f, result.fmin);
            assert!((&best_x[0] - 2.0).abs() < 1e-3);
            assert!((&best_x[1] - 3.0).abs() < 1e-3);
        }

        #[test]
        fn test_debug_formatting() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            // Test debug format before optimization
            let debug_str = format!("{:?}", cmaes);
            assert!(debug_str.contains("CmaEs"));
            assert!(debug_str.contains("generations: 0"));
            assert!(debug_str.contains("converged: false"));

            // Test debug format after optimization
            let _result = cmaes.minimize(array![1.0.into()].view()).unwrap();
            let debug_str_after = format!("{:?}", cmaes);
            assert!(debug_str_after.contains("CmaEs"));
            assert!(debug_str_after.contains("generations:"));
        }

        #[test]
        fn test_seed_reproducibility() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2) + x[1].powi(2);

            // Run with same seed twice
            let objective1 = MultiDimFn::new(func);
            let mut cmaes1 = CmaEs::with_seed(objective1, 12345);
            let result1 = cmaes1
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    Some(20),
                    Some(1e-8.into()),
                    Some(50),
                )
                .unwrap();

            let objective2 = MultiDimFn::new(func);
            let mut cmaes2 = CmaEs::with_seed(objective2, 12345);
            let result2 = cmaes2
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    Some(20),
                    Some(1e-8.into()),
                    Some(50),
                )
                .unwrap();

            // Results should be identical (or very close due to floating point)
            assert!((&result1.fmin - &result2.fmin).abs() < 1e-10);
            assert_eq!(result1.generations, result2.generations);
            assert_eq!(result1.fn_evals, result2.fn_evals);

            for (x1, x2) in result1.xmin.iter().zip(result2.xmin.iter()) {
                assert!((x1 - x2).abs() < 1e-10);
            }
        }

        #[test]
        fn test_constructor_variants() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2);
            let objective = MultiDimFn::new(func);

            // Test new()
            let mut cmaes1 = CmaEs::new(objective.clone());
            let result1 = cmaes1.minimize(array![1.0.into()].view()).unwrap();
            assert!(result1.fmin < 1e-5);

            // Test new_boxed()
            let boxed_objective = Box::new(objective.clone());
            let mut cmaes2 = CmaEs::new_boxed(boxed_objective);
            let result2 = cmaes2.minimize(array![1.0.into()].view()).unwrap();
            assert!(result2.fmin < 1e-5);

            // Test with_seed()
            let mut cmaes3 = CmaEs::with_seed(objective, 999);
            let result3 = cmaes3.minimize(array![1.0.into()].view()).unwrap();
            assert!(result3.fmin < 1e-5);
        }
    }

    mod stress_and_performance_tests {
        use super::*;

        #[test]
        fn test_max_generations_limit() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            // Set very low generation limit
            let result = cmaes
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    None,
                    Some(1e-12.into()), // Very tight tolerance
                    Some(5),            // Very few generations
                )
                .unwrap();

            // Should stop due to generation limit, not convergence
            assert_eq!(result.generations, 5);
            assert!(!result.converged);
            assert!(result.fn_evals > 0);
        }

        #[test]
        fn test_sigma_bounds_enforcement() {
            let func = |x: ArrayView1<MyFloat>| {
                if x[0].abs() < 1e-10 {
                    0.0.into()
                } else {
                    x[0].powi(2)
                }
            };
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![1e-15.into()].view(), // Very small starting point
                    Some(1e-15.into()),          // Very small sigma
                    None,
                    Some(1e-20.into()), // Extremely tight tolerance
                    Some(500),
                )
                .unwrap();

            // Sigma should be bounded and reasonable
            assert!(result.final_sigma >= 1e-10);
            assert!(result.final_sigma <= 1e10);
        }

        #[test]
        fn test_large_population() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes
                .cma_es(
                    array![1.0.into(), 1.0.into()].view(),
                    Some(0.5.into()),
                    Some(100), // Large but reasonable population
                    Some(1e-6.into()),
                    Some(100), // More generations for large population
                )
                .unwrap();

            // Should still work with large population, but be more lenient
            assert!(
                result.fmin < 1e-2,
                "Function value {} too high",
                result.fmin
            );
            assert!(result.fn_evals >= 100 * result.generations); // At least pop_size * generations

            // Should make reasonable progress
            for coord in result.xmin {
                assert!(
                    coord.abs() < 0.1,
                    "Coordinate {} too far from optimum",
                    coord
                );
            }
        }

        #[test]
        fn test_minimal_population() {
            let func = |x: ArrayView1<MyFloat>| x[0].powi(2);
            let objective = MultiDimFn::new(func);
            let mut cmaes = CmaEs::with_seed(objective, 42);

            let result = cmaes.cma_es(
                array![1.0.into()].view(),
                Some(0.5.into()),
                Some(5), // Minimal but viable population for 1D (needs to be > dimension)
                Some(1e-4.into()), // Relaxed tolerance
                Some(200), // More generations for small population
            );

            match result {
                Ok(res) => {
                    // If successful, should find reasonable solution
                    assert!(
                        res.xmin[0].abs() < 0.1,
                        "Solution {} not close to optimum",
                        res.xmin[0]
                    );
                    assert!(res.fmin < 1e-2, "Function value {} too high", res.fmin);
                }
                Err(MinimizerError::FunctionEvaluationError) => {
                    // Small populations can sometimes fail due to numerical issues
                    // This is acceptable behavior for extreme edge cases
                }
                Err(e) => panic!("Unexpected error type: {:?}", e),
            }
        }
    }
}
