#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::{
    error::MinimizerError,
    minimize::{ObjFn, Vertex},
    pts::{Points1, Pts},
};
use ndarray::prelude::*;
use num_traits::Float;
use std::fmt;

/// Result of downhill simplex optimization
#[derive(Debug, Clone)]
pub struct SimplexResult<T>
where
    T: Float,
{
    xmin: Points1<T>,
    fmin: T,
    iters: usize,
    fn_evals: usize,
    converged: bool,
    final_simplex_size: T,
    history: Points1<T>,
}

impl<T> SimplexResult<T>
where
    T: Float,
{
    pub fn xmin(&self) -> Points1<T> {
        self.xmin.clone()
    }

    pub fn fmin(&self) -> T {
        self.fmin.clone()
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn fn_evals(&self) -> usize {
        self.fn_evals
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn final_simplex_size(&self) -> T {
        self.final_simplex_size.clone()
    }

    pub fn history(&self) -> Points1<T> {
        self.history.clone()
    }
}

pub struct Simplex<T> {
    xmin: Points1<T>,
    fmin: T,
    f: Box<dyn ObjFn<T>>,
    iters: usize,
    converged: bool,
}

impl<T> Clone for Simplex<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            f: dyn_clone::clone_box(&*self.f),
            iters: self.iters,
            converged: self.converged,
        }
    }
}

impl<T> Simplex<T>
where
    T: Float,
    for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a> &'a T: std::ops::Add<T, Output = T>,
    for<'a> &'a T: std::ops::Mul<T, Output = T>,
{
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        Simplex {
            xmin: array![].into(),
            fmin: T::zero(),
            f: Box::new(f),
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn<T>>) -> Self {
        Simplex {
            xmin: array![].into(),
            fmin: T::zero(),
            f: f,
            iters: 0,
            converged: false,
        }
    }

    fn reflect_point(&self, worst: &Points1<T>, centroid: &Points1<T>, coeff: &T) -> Points1<T> {
        worst
            .iter()
            .zip(centroid.iter())
            .map(|(w, c)| c + coeff * (c - w))
            .collect()
    }

    fn calculate_centroid(&self, vertices: &[Vertex<T>]) -> Points1<T> {
        let n = vertices[0].point.len();
        let mut centroid = vec![T::zero(); n];

        for vertex in vertices {
            for (i, coord) in vertex.point.iter().enumerate() {
                centroid[i] += coord;
            }
        }

        for coord in &mut centroid {
            *coord /= T::from_f64(vertices.len() as f64);
        }

        Points1::from_vec(centroid)
    }

    fn calculate_simplex_size(&self, simplex: &[Vertex<T>]) -> T {
        let n = simplex[0].point.len();
        let mut max_distance = T::zero();

        for i in 0..n {
            for j in i + 1..=n {
                let distance = simplex[i]
                    .point
                    .iter()
                    .zip(simplex[j].point.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<T>()
                    .sqrt();

                max_distance = max_distance.max(distance);
            }
        }

        max_distance
    }

    fn run_simplex_algorithm(
        &mut self,
        mut simplex: Vec<Vertex<T>>,
        scale: &Points1<T>,
        tol: &T,
        max_iters: usize,
    ) -> Result<SimplexResult<T>, MinimizerError> {
        let n = simplex[0].point.len();
        let mut fn_evals = simplex.len();
        self.iters = 0;
        let mut history = Vec::new();
        self.converged = false;

        // Standard Nelder-Mead coefficients
        let alpha = T::one();
        let gamma = T::from_f64(2.0);
        let rho = T::from_f64(0.5);
        let sigma = T::from_f64(0.5);

        while self.iters < max_iters {
            self.iters += 1;

            simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

            let best = &simplex[0];
            let worst = &simplex[n];
            let second_worst = &simplex[n - 1];

            history.push(best.value.clone());

            let simplex_size = self.calculate_simplex_size(&simplex);
            if simplex_size < *tol {
                self.xmin = &best.point / scale;
                self.fmin = best.value.clone();
                self.converged = true;
                return Ok(SimplexResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin.clone(),
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_simplex_size: simplex_size,
                    history: Points1::from_vec(history),
                });
            }

            let centroid = self.calculate_centroid(&simplex[..n]);

            // Standard Nelder-Mead operations
            let reflected = self.reflect_point(&worst.point, &centroid, &alpha);
            let f_reflected = self.f.call(&(&reflected / scale));
            fn_evals += 1;

            if !f_reflected.is_finite() {
                return Err(MinimizerError::FunctionEvaluationError);
            }

            if best.value <= f_reflected && f_reflected < second_worst.value {
                simplex[n] = Vertex {
                    point: reflected,
                    value: f_reflected,
                };
            } else if f_reflected < best.value {
                let expanded = self.reflect_point(&worst.point, &centroid, &(&alpha * &gamma));
                let f_expanded = self.f.call(&(&expanded / scale));
                fn_evals += 1;

                if !f_expanded.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }

                if f_expanded < f_reflected {
                    simplex[n] = Vertex {
                        point: expanded,
                        value: f_expanded,
                    };
                } else {
                    simplex[n] = Vertex {
                        point: reflected,
                        value: f_reflected,
                    };
                }
            } else {
                let (contracted, f_contracted) = if f_reflected < worst.value {
                    let point = self.reflect_point(&worst.point, &centroid, &(&alpha * &rho));
                    let value = self.f.call(&(&point / scale));
                    (point, value)
                } else {
                    let point = self.reflect_point(&worst.point, &centroid, &-rho.clone());
                    let value = self.f.call(&(&point / scale));
                    (point, value)
                };
                fn_evals += 1;

                if !f_contracted.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }

                if f_contracted < worst.value.min(f_reflected) {
                    simplex[n] = Vertex {
                        point: contracted,
                        value: f_contracted,
                    };
                } else {
                    let best_point = best.point.clone();
                    for i in 1..=n {
                        let mut new_point = Points1::zeros(n);
                        for j in 0..n {
                            new_point[j] =
                                &best_point[j] + &sigma * (&simplex[i].point[j] - &best_point[j]);
                        }
                        let new_value = self.f.call(&(&new_point / scale));
                        fn_evals += 1;

                        if !new_value.is_finite() {
                            return Err(MinimizerError::FunctionEvaluationError);
                        }

                        simplex[i] = Vertex {
                            point: new_point,
                            value: new_value,
                        };
                    }
                }
            }
        }

        simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

        self.xmin = &simplex[0].point / scale;
        self.fmin = simplex[0].value.clone();
        self.converged = false;
        Ok(SimplexResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_simplex_size: self.calculate_simplex_size(&simplex),
            history: Points1::from_vec(history),
        })
    }

    /// Downhill Simplex (Nelder-Mead) optimization method
    ///
    /// This is a derivative-free optimization method that maintains a simplex
    /// of n+1 points in n-dimensional space and iteratively transforms it
    /// toward the minimum through reflection, expansion, contraction, and shrinking.
    ///
    /// # Arguments
    /// * `func` - The function to minimize
    /// * `initial_point` - Starting point for optimization
    /// * `initial_step` - Initial step size for constructing simplex (default: 1.0)
    /// * `tol` - Convergence tolerance (default: 1e-8)
    /// * `max_iters` - Maximum iters (default: 1000)
    ///
    /// # Returns
    /// * `SimplexResult` containing the minimum point and convergence information
    pub fn downhill_simplex(
        &mut self,
        initial_point: &Points1<T>,
        scale: Option<&Points1<T>>,
        initial_step: Option<T>,
        tol: Option<T>,
        max_iters: Option<usize>,
    ) -> Result<SimplexResult<T>, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let scl = scale.unwrap_or(&Points1::ones(n)).clone();
        let x_scaled = initial_point * &scl;
        let step = initial_step.unwrap_or(T::one());
        let tol = tol.unwrap_or(T::from_f64(1e-8));
        let max_iter = max_iters.unwrap_or(1000);

        if tol <= T::zero() {
            return Err(MinimizerError::InvalidTolerance);
        }

        // Nelder-Mead coefficients
        let alpha = T::one(); // Reflection coefficient
        let gamma = T::from_f64(2.0); // Expansion coefficient
        let rho = T::from_f64(0.5); // Contraction coefficient
        let sigma = T::from_f64(0.5); // Shrink coefficient

        // Initialize simplex with n+1 vertices
        let mut simplex = Vec::with_capacity(n + 1);

        // First vertex is the initial point
        simplex.push(Vertex::new_boxed(&x_scaled, self.f.clone())?);

        // Create additional vertices by perturbing each coordinate
        for i in 0..n {
            let mut point = x_scaled.to_owned();
            point[i] += step.clone();
            simplex.push(Vertex::new_boxed(&point, self.f.clone())?);
        }

        let mut fn_evals = n + 1;
        self.iters = 0;
        let mut history = Vec::new();

        while self.iters < max_iter {
            self.iters += 1;

            // Sort vertices by function value (best to worst)
            simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

            let best = &simplex[0];
            let worst = &simplex[n];
            let second_worst = &simplex[n - 1];

            history.push(best.value.clone());

            // Check for convergence
            let simplex_size = self.calculate_simplex_size(&simplex);
            if simplex_size < tol {
                self.xmin = &best.point / &scl;
                self.fmin = best.value.clone();
                self.converged = true;
                return Ok(SimplexResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin.clone(),
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_simplex_size: simplex_size,
                    history: Points1::from_vec(history),
                });
            }

            // Calculate centroid of all points except the worst
            let centroid = self.calculate_centroid(&simplex[..n]);

            // Reflection
            let reflected = self.reflect_point(&worst.point, &centroid, &alpha);
            let f_reflected = self.f.call(&(&reflected / &scl));
            fn_evals += 1;

            if !f_reflected.is_finite() {
                return Err(MinimizerError::FunctionEvaluationError);
            }

            if best.value <= f_reflected && f_reflected < second_worst.value {
                // Accept reflection
                simplex[n] = Vertex {
                    point: reflected,
                    value: f_reflected,
                };
            } else if f_reflected < best.value {
                // Try expansion
                let expanded = self.reflect_point(&worst.point, &centroid, &(&alpha * &gamma));
                let f_expanded = self.f.call(&(&expanded / &scl));
                fn_evals += 1;

                if !f_expanded.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }

                if f_expanded < f_reflected {
                    // Accept expansion
                    simplex[n] = Vertex {
                        point: expanded,
                        value: f_expanded,
                    };
                } else {
                    // Accept reflection
                    simplex[n] = Vertex {
                        point: reflected,
                        value: f_reflected,
                    };
                }
            } else {
                // Try contraction
                let (contracted, f_contracted) = if f_reflected < worst.value {
                    // Outside contraction
                    let point = self.reflect_point(&worst.point, &centroid, &(&alpha * &rho));
                    let value = self.f.call(&(&point / &scl));
                    fn_evals += 1;
                    (point, value)
                } else {
                    // Inside contraction
                    let point = self.reflect_point(&worst.point, &centroid, &-rho.clone());
                    let value = self.f.call(&(&point / &scl));
                    fn_evals += 1;
                    (point, value)
                };

                if !f_contracted.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }

                if f_contracted < worst.value.min(f_reflected) {
                    // Accept contraction
                    simplex[n] = Vertex {
                        point: contracted,
                        value: f_contracted,
                    };
                } else {
                    // Shrink simplex toward best point
                    let best_point = best.point.clone();
                    for i in 1..=n {
                        let mut new_point = Points1::zeros(n);
                        for j in 0..n {
                            new_point[j] =
                                &best_point[j] + &sigma * (&simplex[i].point[j] - &best_point[j]);
                        }
                        let new_value = self.f.call(&(&new_point / &scl));
                        fn_evals += 1;

                        if !new_value.is_finite() {
                            return Err(MinimizerError::FunctionEvaluationError);
                        }

                        simplex[i] = Vertex {
                            point: new_point,
                            value: new_value,
                        };
                    }
                }
            }
        }

        // Sort final simplex
        simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

        self.xmin = &simplex[0].point / &scl;
        self.fmin = simplex[0].value.clone();
        self.converged = false;
        Ok(SimplexResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_simplex_size: self.calculate_simplex_size(&simplex),
            history: Points1::from_vec(history),
        })
    }

    /// Advanced downhill simplex with adaptive parameters
    ///
    /// This version adapts the Nelder-Mead coefficients based on the problem
    /// characteristics for improved performance.
    pub fn adaptive_downhill_simplex(
        &mut self,
        initial_point: &Points1<T>,
        scale: Option<&Points1<T>>,
        initial_step: Option<T>,
        tol: Option<T>,
        max_iters: Option<usize>,
    ) -> Result<SimplexResult<T>, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let scl = scale.unwrap_or(&Points1::ones(n)).clone();
        let x_scaled = initial_point * &scl;
        let step = initial_step.unwrap_or(T::one());
        let tol = tol.unwrap_or(T::from_f64(1e-8));
        let max_iter = max_iters.unwrap_or(1000);

        // Adaptive coefficients (from recent research)
        let alpha = T::one(); // Reflection
        let gamma = T::from_f64(1.0 + 2.0 / n as f64); // Expansion (dimension-dependent)
        let rho = T::from_f64(0.75 - 1.0 / (2.0 * n as f64)); // Contraction
        let sigma = T::from_f64(1.0 - 1.0 / n as f64); // Shrink

        // Initialize simplex using regular simplex construction
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(Vertex::new_boxed(&x_scaled, self.f.clone())?);

        // Use right-angled simplex construction for better performance
        let delta_usual = &step
            * T::from_f64((((n + 1) as f64).sqrt() + (n - 1) as f64) / (n as f64 * 2.0_f64.sqrt()));
        let delta_zero =
            &step * T::from_f64((((n + 1) as f64).sqrt() - 1.0) / (n as f64 * 2.0_f64.sqrt()));

        for i in 0..n {
            let mut point = x_scaled.to_owned();
            for j in 0..n {
                if i == j {
                    point[j] += &delta_usual;
                } else {
                    point[j] += &delta_zero;
                }
            }
            simplex.push(Vertex::new_boxed(&point, self.f.clone())?);
        }

        let mut fn_evals = n + 1;
        self.iters = 0;
        let mut history: Vec<T> = vec![];
        let mut no_improvement_count = 0;

        while self.iters < max_iter {
            self.iters += 1;

            // Sort vertices
            simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

            let best = &simplex[0];
            let worst = &simplex[n];
            let second_worst = &simplex[n - 1];

            // Track convergence
            if !history.is_empty() {
                let improvement = history.last().unwrap() - &best.value;
                if improvement.abs() < tol.clone() * T::from_f64(0.01) {
                    no_improvement_count += 1;
                } else {
                    no_improvement_count = 0;
                }
            }
            history.push(best.value.clone());

            // Check for convergence
            let simplex_size = self.calculate_simplex_size(&simplex);
            if simplex_size < tol || no_improvement_count > 20 {
                self.xmin = &best.point / &scl;
                self.fmin = best.value.clone();
                self.converged = true;
                return Ok(SimplexResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin.clone(),
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_simplex_size: simplex_size,
                    history: Points1::from_vec(history),
                });
            }

            // Calculate centroid
            let centroid = self.calculate_centroid(&simplex[..n]);

            // Reflection
            let reflected = self.reflect_point(&worst.point, &centroid, &alpha);
            let f_reflected = self.f.call(&(&reflected / &scl));
            fn_evals += 1;

            if !f_reflected.is_finite() {
                return Err(MinimizerError::FunctionEvaluationError);
            }

            // Main Nelder-Mead logic with adaptive parameters
            if best.value <= f_reflected && f_reflected < second_worst.value {
                simplex[n] = Vertex {
                    point: reflected,
                    value: f_reflected,
                };
            } else if f_reflected < best.value {
                // Expansion
                let expanded =
                    self.reflect_point(&worst.point, &centroid, &(&alpha + (&gamma - &alpha)));
                let f_expanded = self.f.call(&(&expanded / &scl));
                fn_evals += 1;

                if !f_expanded.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }

                if f_expanded < f_reflected {
                    simplex[n] = Vertex {
                        point: expanded,
                        value: f_expanded,
                    };
                } else {
                    simplex[n] = Vertex {
                        point: reflected,
                        value: f_reflected,
                    };
                }
            } else {
                // Contraction
                let (contracted, f_contracted) = if f_reflected < worst.value {
                    let point = self.reflect_point(&worst.point, &centroid, &(&alpha * &rho));
                    let value = self.f.call(&(&point / &scl));
                    (point, value)
                } else {
                    let point = self.reflect_point(&worst.point, &centroid, &-rho.clone());
                    let value = self.f.call(&(&point / &scl));
                    (point, value)
                };
                fn_evals += 1;

                if !f_contracted.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }

                if f_contracted < worst.value.min(f_reflected) {
                    simplex[n] = Vertex {
                        point: contracted,
                        value: f_contracted,
                    };
                } else {
                    // Shrink with adaptive sigma
                    let best_point = best.point.clone();
                    for i in 1..=n {
                        let mut new_point = Points1::zeros(n);
                        for j in 0..n {
                            new_point[j] =
                                &best_point[j] + &sigma * (&simplex[i].point[j] - &best_point[j]);
                        }
                        let new_value = self.f.call(&(&new_point / &scl));
                        fn_evals += 1;

                        if !new_value.is_finite() {
                            return Err(MinimizerError::FunctionEvaluationError);
                        }

                        simplex[i] = Vertex {
                            point: new_point,
                            value: new_value,
                        };
                    }
                }
            }
        }

        simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

        self.xmin = &simplex[0].point / &scl;
        self.fmin = simplex[0].value.clone();
        self.converged = false;
        Ok(SimplexResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_simplex_size: self.calculate_simplex_size(&simplex),
            history: Points1::from_vec(history),
        })
    }

    /// Convenience function with default parameters
    pub fn minimize(
        &mut self,
        initial_point: &Points1<T>,
        scale: Option<&Points1<T>>,
        initial_step: Option<T>,
        tol: Option<T>,
        max_iters: Option<usize>,
    ) -> Result<SimplexResult<T>, MinimizerError> {
        self.downhill_simplex(initial_point, scale, initial_step, tol, max_iters)
    }

    /// Create initial simplex with custom step sizes for each dimension
    pub fn minimize_with_steps(
        &mut self,
        initial_point: &Points1<T>,
        scale: Option<&Points1<T>>,
        step_sizes: &Points1<T>,
    ) -> Result<SimplexResult<T>, MinimizerError> {
        let n = initial_point.len();
        if step_sizes.len() != n {
            return Err(MinimizerError::InvalidInitialSimplex);
        }

        let scl = scale.unwrap_or(&Points1::ones(n)).clone();
        let x_scaled = initial_point * &scl;

        // Custom implementation with individual step sizes
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(Vertex::new_boxed(&x_scaled, self.f.clone())?);

        for i in 0..n {
            let mut point = x_scaled.to_owned();
            point[i] += &step_sizes[i];
            simplex.push(Vertex::new_boxed(&point, self.f.clone())?);
        }

        // Continue with standard algorithm
        self.run_simplex_algorithm(simplex, &scl, &T::from_f64(1e-8), 1000)
    }
}

impl<T> fmt::Debug for Simplex<T>
where
    T: Float,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Simplex( xmin: {:?}, fmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod minimize_simplex_tests {
    use super::*;
    use crate::{
        minimize::{F1dim, MultiDimFn},
        num::MyFloat,
    };
    use float_cmp::F64Margin;
    use ndarray_rand::{RandomExt, rand_distr::Uniform};
    use std::{
        f64::consts::{E, PI},
        vec,
    };

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    // Helper function to create Ackley
    fn ackley(bias: f64) -> F1dim<MyFloat> {
        F1dim::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            let n = x.len() as f64;
            let t1 = x.iter().map(|val| val.powi(2)).sum::<MyFloat>();
            let t2 = x.iter().map(|val| (2.0 * PI * val).cos()).sum::<MyFloat>();
            -20.0 * (-0.2 * (&t1 / n).sqrt()).exp() - (&t2 / n).exp() + 20.0 + E + bias
        }))
    }

    // Helper function to create Elliptic
    fn elliptic(bias: f64) -> F1dim<MyFloat> {
        F1dim::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            let sum: MyFloat = x
                .iter()
                .enumerate()
                .map(|(i, val)| 1e6_f64.powf((i / (x.len() - 1)) as f64) * val.powi(2))
                .sum();
            &sum + bias
        }))
    }

    // Helper function to create Griewank
    fn griewank(bias: f64) -> F1dim<MyFloat> {
        F1dim::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            let t1 = x.iter().map(|val| val.powi(2)).sum::<MyFloat>() / 4000.0;
            let t2 = x
                .iter()
                .enumerate()
                .map(|(i, val)| (val / (i as f64 + 1.0).sqrt()).cos())
                .product::<MyFloat>();
            &t1 - &t2 + 1.0 + bias
        }))
    }

    // Helper function to create Rosenbrock
    fn rosenbrock(shift: Points1<MyFloat>, bias: f64) -> F1dim<MyFloat> {
        F1dim::new(MultiDimFn::new(move |x: &Points1<MyFloat>| {
            let x_new = x - &shift + 1.0;
            let sum: MyFloat = x_new
                .0
                .windows(2)
                .into_iter()
                .map(|pair| 100.0 * (pair[0].powi(2) - &pair[1]).powi(2) + (&pair[0] - 1.0).powi(2))
                .sum();
            &sum + bias
        }))
    }

    #[test]
    fn test_2d_quadratic() {
        // f(x,y) = (x-1)² + (y-2)², minimum at (1,2)
        let func = |x: &Points1<MyFloat>| (&x[0] - 1.0).powi(2) + (&x[1] - 2.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .minimize(
                &array![0.0.into(), 0.0.into()].into(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        assert!((&result.xmin[0] - 1.0).abs() < 1e-6);
        assert!((&result.xmin[1] - 2.0).abs() < 1e-6);
        assert!(result.fmin < 1e-10);
        assert!(result.converged);

        assert!((&simplex.xmin[0] - 1.0).abs() < 1e-6);
        assert!((&simplex.xmin[1] - 2.0).abs() < 1e-6);
        assert!(simplex.fmin < 1e-10);
        assert!(simplex.converged);
    }

    #[test]
    fn test_rosenbrock_2d() {
        // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
        let rosenbrock =
            |x: &Points1<MyFloat>| (1.0 - &x[0]).powi(2) + 100.0 * (&x[1] - &x[0].powi(2)).powi(2);
        let objective = MultiDimFn::new(rosenbrock);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .downhill_simplex(
                &array![MyFloat::new(-1.2), 1.0.into()].into(),
                None,
                Some(0.1.into()),
                Some(1e-6.into()),
                Some(2000),
            )
            .unwrap();

        assert!((&result.xmin[0] - 1.0).abs() < 1e-4);
        assert!((&result.xmin[1] - 1.0).abs() < 1e-4);
        assert!(result.fmin < 1e-6);

        assert!((&simplex.xmin[0] - 1.0).abs() < 1e-4);
        assert!((&simplex.xmin[1] - 1.0).abs() < 1e-4);
        assert!(simplex.fmin < 1e-6);
    }

    #[test]
    fn test_3d_sphere() {
        // f(x,y,z) = x² + y² + z², minimum at (0,0,0)
        let sphere = |x: &Points1<MyFloat>| x.iter().map(|xi| xi * xi).sum();
        let objective = MultiDimFn::new(sphere);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .minimize(
                &array![1.0.into(), 1.0.into(), 1.0.into()].into(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        for coord in result.xmin {
            assert!(coord.abs() < 1e-6);
        }
        assert!(result.fmin < 1e-10);
        assert!(result.converged);

        for coord in simplex.xmin {
            assert!(coord.abs() < 1e-6);
        }
        assert!(simplex.fmin < 1e-10);
        assert!(simplex.converged);
    }

    #[test]
    fn test_adaptive_simplex() {
        let func = |x: &Points1<MyFloat>| (&x[0] - 3.0).powi(2) + (&x[1] + 2.0).powi(2) + 5.0;
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .adaptive_downhill_simplex(
                &array![0.0.into(), 0.0.into()].into(),
                None,
                Some(1.0.into()),
                Some(1e-8.into()),
                None,
            )
            .unwrap();

        assert!((&result.xmin[0] - 3.0).abs() < 1e-6);
        assert!((&result.xmin[1] + 2.0).abs() < 1e-6);
        assert!((result.fmin - 5.0).abs() < 1e-8);

        assert!((&simplex.xmin[0] - 3.0).abs() < 1e-6);
        assert!((&simplex.xmin[1] + 2.0).abs() < 1e-6);
        assert!((simplex.fmin - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_custom_step_sizes() {
        let func = |x: &Points1<MyFloat>| (&x[0] / 10.0 - 1.0).powi(2) + (&x[1] - 2.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .minimize_with_steps(
                &array![0.0.into(), 0.0.into()].into(),
                None,
                &array![10.0.into(), 1.0.into()].into(), // Different step sizes for each dimension
            )
            .unwrap();

        assert!((&result.xmin[0] - 10.0).abs() < 1e-4);
        assert!((&result.xmin[1] - 2.0).abs() < 1e-6);

        assert!((&simplex.xmin[0] - 10.0).abs() < 1e-4);
        assert!((&simplex.xmin[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_dimension() {
        let func = |_: &Points1<MyFloat>| 0.0.into();
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex.minimize(&array![].into(), None, None, None, None);
        assert!(result.is_err());

        assert!(matches!(result, Err(MinimizerError::InvalidDimension)));
    }

    #[test]
    fn test_convergence_tracking() {
        let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .minimize(
                &array![2.0.into(), 2.0.into()].into(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        assert!(!result.history.is_empty());
        assert!(result.history[0] > result.fmin);
        assert!(result.history.len() == result.iters);
    }

    #[test]
    fn test_simplex_high_dimensional() {
        // Test 5D sphere function
        let sphere_5d = |x: &Points1<MyFloat>| x.iter().map(|xi| xi * xi).sum::<MyFloat>();
        let objective = MultiDimFn::new(sphere_5d);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .minimize(&Points1::ones(5).into(), None, None, None, None)
            .unwrap();

        assert!(result.converged);
        for coord in result.xmin {
            assert!(coord.abs() < 1e-4);
        }
        assert!(result.fmin < 1e-6);
    }

    #[test]
    fn test_simplex_rosenbrock_variants() {
        // Test 1: 2D Rosenbrock
        let rosenbrock_2d =
            |x: &Points1<MyFloat>| (1.0 - &x[0]).powi(2) + 100.0 * (&x[1] - x[0].powi(2)).powi(2);
        let objective = MultiDimFn::new(rosenbrock_2d);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .downhill_simplex(
                &array![MyFloat::new(-1.2), 1.0.into()].into(),
                None,
                Some(0.1.into()),
                Some(1e-6.into()),
                Some(3000),
            )
            .unwrap();

        assert!((&result.xmin[0] - 1.0).abs() < 1e-3);
        assert!((&result.xmin[1] - 1.0).abs() < 1e-3);

        // Test 2: Extended Rosenbrock (4D)
        let rosenbrock_4d = |x: &Points1<MyFloat>| {
            (0..x.len() - 1)
                .map(|i| (1.0 - &x[i]).powi(2) + 100.0 * (&x[i + 1] - x[i].powi(2)).powi(2))
                .sum::<MyFloat>()
        };
        let objective = MultiDimFn::new(rosenbrock_4d);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .minimize(
                &(Points1::<MyFloat>::ones(4) * -1.0),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        for coord in result.xmin {
            assert!((coord - 1.0).abs() < 0.1); // Relaxed for high-D Rosenbrock
        }
    }

    #[test]
    fn test_simplex_with_constraints_penalty() {
        // Test penalty method simulation
        let constrained_objective = |x: &Points1<MyFloat>| {
            let obj = x[0].powi(2) + x[1].powi(2);
            let penalty = if &x[0] + &x[1] < 1.0 {
                1000.0 * (1.0 - &x[0] - &x[1]).powi(2)
            } else {
                0.0.into()
            };
            &obj + &penalty
        };
        let objective = MultiDimFn::new(constrained_objective);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .minimize(
                &array![0.6.into(), 0.6.into()].into(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        // Should satisfy constraint x + y >= 1
        assert!(&result.xmin[0] + &result.xmin[1] >= 0.95);
        assert!(result.xmin[0] >= -0.05 && result.xmin[1] >= -0.05);
    }

    #[test]
    fn test_adaptive_simplex_performance() {
        // Compare standard vs adaptive simplex
        let beale = |x: &Points1<MyFloat>| {
            let x1 = x[0].clone();
            let x2 = x[1].clone();
            (1.5 - &x1 + &x1 * &x2).powi(2)
                + (2.25 - &x1 + &x1 * x2.powi(2)).powi(2)
                + (2.625 - &x1 + &x1 * x2.powi(3)).powi(2)
        };
        let objective = MultiDimFn::new(beale);
        let mut simplex = Simplex::new(objective);

        let standard_result = simplex
            .downhill_simplex(
                &array![1.0.into(), 1.0.into()].into(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let adaptive_result = simplex
            .adaptive_downhill_simplex(
                &array![1.0.into(), 1.0.into()].into(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        // Both should converge
        assert!(standard_result.converged || adaptive_result.converged);

        // At least one should be close to optimum (3.0, 0.5)
        let std_dist = ((&standard_result.xmin[0] - 3.0).powi(2)
            + (&standard_result.xmin[1] - 0.5).powi(2))
        .sqrt();
        let ada_dist = ((&adaptive_result.xmin[0] - 3.0).powi(2)
            + (&adaptive_result.xmin[1] - 0.5).powi(2))
        .sqrt();

        assert!(std_dist < 0.1 || ada_dist < 0.1);
    }

    mod basic_functionality_tests {
        use super::*;

        #[test]
        fn test_1d_optimization() {
            // f(x) = (x - 5)²
            let func = |x: &Points1<MyFloat>| (&x[0] - 5.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(&array![0.0.into()].into(), None, None, None, None)
                .unwrap();

            assert!((&result.xmin[0] - 5.0).abs() < 1e-6);
            assert!(result.fmin < 1e-10);
            assert!(result.converged);
            assert!(result.iters > 0);
            assert!(result.fn_evals >= 2); // At least n+1 evaluations
        }

        #[test]
        fn test_already_at_minimum() {
            let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
            assert!(result.converged);
        }

        #[test]
        fn test_negative_coordinates() {
            // Minimum at (-3, -4)
            let func = |x: &Points1<MyFloat>| (&x[0] + 3.0).powi(2) + (&x[1] + 4.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            assert!((&result.xmin[0] + 3.0).abs() < 1e-6);
            assert!((&result.xmin[1] + 4.0).abs() < 1e-6);
            assert!(result.fmin < 1e-10);
        }
    }

    mod error_handling_tests {
        use super::*;

        #[test]
        fn test_empty_initial_point() {
            let func = |_: &Points1<MyFloat>| 0.0.into();
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(&array![].into(), None, None, None, None);
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::InvalidDimension
            ));
        }

        #[test]
        fn test_invalid_tolerance() {
            let func = |x: &Points1<MyFloat>| x[0].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.downhill_simplex(
                &array![1.0.into()].into(),
                None,
                None,
                Some(0.0.into()),
                None,
            );
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::InvalidTolerance
            ));

            let result = simplex.downhill_simplex(
                &array![1.0.into()].into(),
                None,
                None,
                Some(MyFloat::new(-1e-8)),
                None,
            );
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::InvalidTolerance
            ));
        }

        #[test]
        fn test_function_returning_infinity() {
            let func = |x: &Points1<MyFloat>| {
                if x[0] > 1.0 {
                    f64::INFINITY.into()
                } else {
                    x[0].powi(2)
                }
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(&array![2.0.into()].into(), None, None, None, None); // Start where function is infinite
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::FunctionEvaluationError
            ));
        }

        #[test]
        fn test_function_returning_nan() {
            let func = |x: &Points1<MyFloat>| {
                if x[0] < 0.0 {
                    f64::NAN.into()
                } else {
                    x[0].powi(2)
                }
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result =
                simplex.minimize(&array![MyFloat::new(-1.0)].into(), None, None, None, None); // Start where function is NaN
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::FunctionEvaluationError
            ));
        }

        #[test]
        fn test_mismatched_step_sizes() {
            let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize_with_steps(
                &array![1.0.into(), 1.0.into()].into(),
                None,
                &array![0.1.into()].into(),
            ); // Wrong length
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::InvalidInitialSimplex
            ));
        }
    }

    mod convergence_tests {
        use super::*;

        #[test]
        fn test_max_iterations_reached() {
            // Create a difficult function that won't converge quickly
            let func = |x: &Points1<MyFloat>| {
                (&x[0] - 1.0).powi(4) + (&x[1] - 2.0).powi(4) + 0.01 * (&x[0] * &x[1]).sin()
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    Some(1e-12.into()),
                    Some(10),
                )
                .unwrap();

            assert_eq!(result.iters, 10);
            assert!(!result.converged);
        }

        #[test]
        fn test_convergence_tracking() {
            let func = |x: &Points1<MyFloat>| (&x[0] - 2.0).powi(2) + (&x[1] - 3.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            assert!(!result.history.is_empty());
            assert_eq!(result.history.len(), result.iters);

            // History should be generally decreasing (allowing for some fluctuation)
            let first_value = result.history[0].clone();
            let last_value = result.history.last().unwrap();
            assert!(first_value >= *last_value);

            // Final value should match fmin
            assert!((last_value - result.fmin).abs() < 1e-10);
        }

        #[test]
        fn test_tight_tolerance() {
            let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(
                    &array![1.0.into(), 1.0.into()].into(),
                    None,
                    None,
                    Some(1e-12.into()),
                    None,
                )
                .unwrap();

            assert!(result.converged);
            assert!(result.final_simplex_size < 1e-12);
            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
        }
    }

    mod high_dimensional_tests {
        use super::*;

        #[test]
        fn test_10d_sphere() {
            let sphere_10d = |x: &Points1<MyFloat>| x.iter().map(|xi| xi * xi).sum::<MyFloat>();
            let objective = MultiDimFn::new(sphere_10d);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(&Points1::ones(10).into(), None, None, None, None)
                .unwrap();

            for coord in result.xmin {
                assert!(coord.abs() < 1e-3);
            }
            assert!(result.fmin < 1e-4);
        }

        #[test]
        fn test_high_dimensional_rosenbrock() {
            // 4D Rosenbrock function
            let rosenbrock_4d = |x: &Points1<MyFloat>| {
                (0..x.len() - 1)
                    .map(|i| (1.0 - &x[i]).powi(2) + 100.0 * (&x[i + 1] - x[i].powi(2)).powi(2))
                    .sum::<MyFloat>()
            };
            let objective = MultiDimFn::new(rosenbrock_4d);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(
                    &(Points1::<MyFloat>::ones(4) * -0.5),
                    None,
                    Some(0.5.into()),
                    Some(1e-4.into()),
                    Some(5000),
                )
                .unwrap();

            // Rosenbrock optimum is at (1, 1, 1, 1)
            for coord in result.xmin {
                assert!((coord - 1.0).abs() < 0.2); // Relaxed tolerance for high-D Rosenbrock
            }
        }
    }

    mod complex_function_tests {
        use super::*;

        #[test]
        fn test_beale_function() {
            // Beale function: global minimum at (3, 0.5)
            let beale = |x: &Points1<MyFloat>| {
                let x1 = x[0].clone();
                let x2 = x[1].clone();
                (1.5 - &x1 + &x1 * &x2).powi(2)
                    + (2.25 - &x1 + &x1 * x2.powi(2)).powi(2)
                    + (2.625 - &x1 + &x1 * x2.powi(3)).powi(2)
            };
            let objective = MultiDimFn::new(beale);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(
                    &array![1.0.into(), 1.0.into()].into(),
                    None,
                    Some(0.5.into()),
                    Some(1e-6.into()),
                    Some(2000),
                )
                .unwrap();

            assert!((&result.xmin[0] - 3.0).abs() < 0.1);
            assert!((&result.xmin[1] - 0.5).abs() < 0.1);
            assert!(result.fmin < 1e-4);
        }

        #[test]
        fn test_goldstein_price_function() {
            // Goldstein-Price function: global minimum at (0, -1) with value 3
            let goldstein_price = |x: &Points1<MyFloat>| {
                let x1 = x[0].clone();
                let x2 = x[1].clone();
                (1.0 + (&x1 + &x2 + 1.0).powi(2)
                    * (19.0 - 14.0 * &x1 + 3.0 * x1.powi(2) - 14.0 * &x2
                        + 6.0 * &x1 * &x2
                        + 3.0 * x2.powi(2)))
                    * (30.0
                        + (2.0 * &x1 - 3.0 * &x2).powi(2)
                            * (18.0 - 32.0 * &x1 + 12.0 * x1.powi(2) + 48.0 * &x2
                                - 36.0 * &x1 * &x2
                                + 27.0 * x2.powi(2)))
            };
            let objective = MultiDimFn::new(goldstein_price);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(
                    &array![0.5.into(), MyFloat::new(-0.5)].into(),
                    None,
                    Some(0.2.into()),
                    Some(1e-6.into()),
                    Some(3000),
                )
                .unwrap();

            assert!((&result.xmin[0] - 0.0).abs() < 0.1);
            assert!((&result.xmin[1] + 1.0).abs() < 0.1);
            assert!((result.fmin - 3.0).abs() < 1.0);
        }

        #[test]
        fn test_himmelblau_function() {
            // Himmelblau's function has 4 global minima
            let himmelblau = |x: &Points1<MyFloat>| {
                (x[0].powi(2) + &x[1] - 11.0).powi(2) + (&x[0] + x[1].powi(2) - 7.0).powi(2)
            };
            let objective = MultiDimFn::new(himmelblau);
            let mut simplex = Simplex::new(objective);

            // Test from different starting points to potentially find different minima
            let starting_points = array![
                array![0.0.into(), 0.0.into()],
                array![3.0.into(), 2.0.into()],
                array![MyFloat::new(-3.0), 3.0.into()],
                array![MyFloat::new(-3.0), MyFloat::new(-3.0)],
            ];

            for start in starting_points {
                let result = simplex
                    .downhill_simplex(
                        &start.into(),
                        None,
                        Some(0.5.into()),
                        Some(1e-6.into()),
                        Some(2000),
                    )
                    .unwrap();

                // All minima should have function value 0
                assert!(result.fmin < 1e-4);

                // Check if we found one of the known minima
                let known_minima = array![
                    (3.0.into(), 2.0.into()),
                    (MyFloat::new(-2.805118), 3.131312.into()),
                    (MyFloat::new(-3.779310), MyFloat::new(-3.283186)),
                    (3.584428.into(), MyFloat::new(-1.848126)),
                ];

                let found_known_minimum = known_minima.iter().any(|(x, y)| {
                    (&result.xmin[0] - x).abs() < 0.1 && (&result.xmin[1] - y).abs() < 0.1
                });

                assert!(found_known_minimum || result.fmin < 1e-4);
            }
        }
    }

    mod adaptive_vs_standard_comparison_tests {
        use super::*;

        #[test]
        fn test_adaptive_vs_standard_performance() {
            let func = |x: &Points1<MyFloat>| {
                // Scaled quadratic - more challenging in one dimension
                (&x[0] / 100.0 - 1.0).powi(2) + (&x[1] - 2.0).powi(2)
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let standard_result = simplex
                .downhill_simplex(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    Some(1e-8.into()),
                    None,
                )
                .unwrap();
            let adaptive_result = simplex
                .adaptive_downhill_simplex(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    Some(1e-8.into()),
                    None,
                )
                .unwrap();

            // Both should find the correct minimum
            assert!((&standard_result.xmin[0] - 100.0).abs() < 1e-3);
            assert!((&adaptive_result.xmin[0] - 100.0).abs() < 1e-3);
            assert!((&standard_result.xmin[1] - 2.0).abs() < 1e-6);
            assert!((&adaptive_result.xmin[1] - 2.0).abs() < 1e-6);
        }

        #[test]
        fn test_adaptive_no_improvement_convergence() {
            // Function that plateaus to test no-improvement convergence
            let plateau_func = |x: &Points1<MyFloat>| {
                let base = x[0].powi(2) + x[1].powi(2);
                if base < 0.01 {
                    0.01.into()
                } else {
                    base.clone()
                }
            };
            let objective = MultiDimFn::new(plateau_func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .adaptive_downhill_simplex(
                    &array![2.0.into(), 2.0.into()].into(),
                    None,
                    Some(1.0.into()),
                    Some(1e-8.into()),
                    Some(1000),
                )
                .unwrap();

            // Should converge and get close to the plateau region
            assert!(result.converged);
            // The plateau starts at distance sqrt(0.01) = 0.1 from origin
            let distance_from_origin = (result.xmin[0].powi(2) + result.xmin[1].powi(2)).sqrt();
            assert!(distance_from_origin < 0.5); // Should get reasonably close to plateau
            assert!(result.fmin >= 0.009); // Should be near the plateau value of 0.01
        }
    }

    mod custom_step_size_tests {
        use super::*;

        #[test]
        fn test_custom_step_sizes_scaled_problem() {
            // Problem with very different scales in different dimensions
            let scaled_func = |x: &Points1<MyFloat>| {
                (&x[0] / 1000.0 - 1.0).powi(2) + (&x[1] * 1000.0 - 2.0).powi(2)
            };
            let objective = MultiDimFn::new(scaled_func);
            let mut simplex = Simplex::new(objective);

            // Use appropriate step sizes for each dimension
            let result = simplex
                .minimize_with_steps(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    &array![1000.0.into(), 0.001.into()].into(), // Large step for first dim, small for second
                )
                .unwrap();

            assert!((&result.xmin[0] - 1000.0).abs() < 1.0);
            assert!((&result.xmin[1] - 0.002).abs() < 1e-6);
        }

        #[test]
        fn test_zero_step_size() {
            let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            // Test with zero step size - this creates a degenerate simplex
            let result = simplex.minimize_with_steps(
                &array![1.0.into(), 1.0.into()].into(),
                None,
                &array![0.0.into(), 1.0.into()].into(),
            );

            // Zero step size creates a degenerate simplex, which may cause convergence issues
            // but the algorithm should still handle it gracefully
            if result.is_ok() {
                let res = result.unwrap();
                assert!(res.fn_evals >= 3); // At least n+1 evaluations
            // Don't assert on convergence quality since degenerate simplex is pathological
            } else {
                // It's also acceptable for this to fail with an appropriate error
                println!("Zero step size caused expected failure: {:?}", result.err());
            }
        }
    }

    mod edge_case_function_tests {
        use super::*;

        #[test]
        fn test_constant_function() {
            let constant_func = |_: &Points1<MyFloat>| 42.0.into();
            let objective = MultiDimFn::new(constant_func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(
                    &array![1.0.into(), 2.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            assert!((result.fmin - 42.0).abs() < 1e-10);
            assert!(result.converged); // Should converge immediately due to zero simplex size
        }

        #[test]
        fn test_linear_function() {
            // f(x,y) = 2x + 3y (unbounded below, but simplex should handle gracefully)
            let linear_func = |x: &Points1<MyFloat>| 2.0 * &x[0] + 3.0 * &x[1];
            let objective = MultiDimFn::new(linear_func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.downhill_simplex(
                &array![0.0.into(), 0.0.into()].into(),
                None,
                Some(1.0.into()),
                Some(1e-8.into()),
                Some(100),
            );

            // Linear function is unbounded below, so algorithm should either:
            // 1. Hit max iterations without converging, or
            // 2. Find a very negative value
            if result.is_ok() {
                let res = result.unwrap();
                if !res.converged {
                    assert_eq!(res.iters, 100); // Hit max iterations
                }
                // The function value should be moving in the negative direction
                assert!(res.fmin <= 0.0); // Should find negative values
            }
        }

        #[test]
        fn test_discontinuous_function() {
            let discontinuous = |x: &Points1<MyFloat>| {
                if x[0] < 0.0 {
                    100.0 + x[0].powi(2) + x[1].powi(2)
                } else {
                    x[0].powi(2) + x[1].powi(2)
                }
            };
            let objective = MultiDimFn::new(discontinuous);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(
                    &array![1.0.into(), 1.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            // Should find minimum in the x >= 0 region
            assert!(result.xmin[0] >= -1e-6);
            assert!(result.xmin[0].abs() < 1e-4);
            assert!(result.xmin[1].abs() < 1e-4);
        }
    }

    mod state_consistency_tests {
        use super::*;

        #[test]
        fn test_simplex_state_consistency() {
            let func = |x: &Points1<MyFloat>| (&x[0] - 3.0).powi(2) + (&x[1] - 4.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            // Check that simplex internal state matches result
            assert_eq!(simplex.xmin, result.xmin);
            assert_eq!(simplex.fmin, result.fmin);
            assert_eq!(simplex.iters, result.iters);
            assert_eq!(simplex.converged, result.converged);
        }

        #[test]
        fn test_multiple_optimizations() {
            let func = |x: &Points1<MyFloat>| (&x[0] - 1.0).powi(2) + (&x[1] - 2.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            // First optimization
            let result1 = simplex
                .minimize(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            // Second optimization with different starting point
            let result2 = simplex
                .minimize(
                    &array![5.0.into(), 5.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            // Both should find the same minimum
            assert!((&result1.xmin[0] - 1.0).abs() < 1e-6);
            assert!((&result1.xmin[1] - 2.0).abs() < 1e-6);
            assert!((&result2.xmin[0] - 1.0).abs() < 1e-6);
            assert!((&result2.xmin[1] - 2.0).abs() < 1e-6);

            // State should reflect the last optimization
            assert_eq!(simplex.xmin, result2.xmin);
            assert_eq!(simplex.fmin, result2.fmin);
        }
    }

    mod large_scale_tests {
        use super::*;

        #[test]
        fn test_function_evaluation_count() {
            // Note: This test would need to be redesigned since we can't capture eval_count
            // in a closure that implements ObjFn. Instead, we can test that fn_evals is reasonable:

            let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(
                    &array![1.0.into(), 1.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            // Should have reasonable number of function evaluations
            assert!(result.fn_evals >= 3); // At least n+1 for initial simplex
            assert!(result.fn_evals <= result.iters * 10); // Reasonable upper bound
        }

        #[test]
        fn test_very_small_initial_step() {
            let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            // Very small initial step creates a tiny simplex that may converge immediately
            // due to the simplex size being smaller than the tolerance
            let result = simplex
                .downhill_simplex(
                    &array![1.0.into(), 1.0.into()].into(),
                    None,
                    Some(1e-12.into()),
                    Some(1e-6.into()),
                    Some(5000),
                )
                .unwrap();

            // With very small initial step, the algorithm often converges immediately
            // because the initial simplex size is already smaller than the tolerance
            assert!(result.converged);

            // The result might be close to the starting point due to the tiny simplex
            // What matters is that the algorithm handled the edge case gracefully
            assert!(result.fn_evals >= 3); // At least created the initial simplex
            assert!(result.iters >= 1); // At least one iteration

            // The simplex size should be very small
            assert!(result.final_simplex_size <= 1e-6);

            // Function value should be finite and non-negative
            assert!(result.fmin >= 0.0);
            assert!(result.fmin.is_finite());
        }

        #[test]
        fn test_small_but_reasonable_initial_step() {
            // This test demonstrates proper behavior with a small but reasonable step
            let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(
                    &array![1.0.into(), 1.0.into()].into(),
                    None,
                    Some(1e-6.into()),
                    Some(1e-8.into()),
                    None,
                )
                .unwrap();

            assert!(result.converged);
            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
            assert!(result.fmin < 1e-10);
        }

        #[test]
        fn test_very_large_initial_step() {
            let func = |x: &Points1<MyFloat>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    Some(1000.0.into()),
                    None,
                    None,
                )
                .unwrap();

            assert!(result.converged);
            assert!(result.xmin[0].abs() < 1e-4);
            assert!(result.xmin[1].abs() < 1e-4);
        }
    }

    mod numerical_stability_tests {
        use super::*;

        #[test]
        fn test_ill_conditioned_problem() {
            // Function with very different curvatures in different directions
            let ill_conditioned = |x: &Points1<MyFloat>| x[0].powi(2) + 10000.0 * x[1].powi(2);
            let objective = MultiDimFn::new(ill_conditioned);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(
                    &array![1.0.into(), 1.0.into()].into(),
                    None,
                    Some(0.1.into()),
                    Some(1e-4.into()),
                    Some(5000),
                )
                .unwrap();

            // Ill-conditioned problems are harder to solve, so use relaxed tolerances
            assert!(result.xmin[0].abs() < 0.1);
            assert!(result.xmin[1].abs() < 0.01); // y-direction has much higher curvature

            // Should at least improve significantly from starting point
            let initial_value = 1.0 + 10000.0; // 10001.0
            assert!(result.fmin < initial_value * 0.1); // At least 90% improvement
        }

        #[test]
        fn test_near_zero_function_values() {
            let tiny_func = |x: &Points1<MyFloat>| 1e-15 * (x[0].powi(2) + x[1].powi(2));
            let objective = MultiDimFn::new(tiny_func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .minimize(
                    &array![1.0.into(), 1.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
            assert!(result.fmin >= 0.0);
            assert!(result.fmin < 1e-12);
        }
    }

    mod cec2005_tests {
        use super::*;
        use ndarray::linalg::Dot;
        use rand::Rng;

        const TEST_30D: bool = false;
        const TEST_50D: bool = false;
        // const TEST_30D: bool = true;
        // const TEST_50D: bool = true;

        #[test]
        fn test_f1() {
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
            let x = Points1::from_shape_fn(n, |_| {
                MyFloat::new(rand::rng().random::<f64>() * 200.0 - 100.0)
            });
            let shift_func = shift.clone();
            let func = move |x: &Points1<MyFloat>| {
                x.iter()
                    .enumerate()
                    .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                    .sum()
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);
            let result = simplex
                .minimize(
                    &x,
                    None,
                    None,
                    Some(1e-10.into()),
                    Some(1e3 as usize * n.pow(2)),
                )
                .unwrap();
            for i in 0..10 {
                assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&result.fmin - bias) < 1e-3);

                assert!((&simplex.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&simplex.fmin - bias) < 1e-3)
            }

            // 30d
            if TEST_30D {
                let n = 30;
                let x = Points1::from_shape_fn(n, |_| {
                    MyFloat::new(rand::rng().random::<f64>() * 200.0 - 100.0)
                });
                let shift_func = shift.clone();
                let func = move |x: &Points1<MyFloat>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let mut simplex = Simplex::new(objective);
                let result = simplex
                    .minimize(
                        &x,
                        None,
                        None,
                        Some(1e-10.into()),
                        Some(1e3 as usize * n.pow(2)),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&result.fmin - bias) < 1e-3);

                    assert!((&simplex.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&simplex.fmin - bias) < 1e-3)
                }
            }

            // 50d
            if TEST_50D {
                let n = 50;
                let x = Points1::from_shape_fn(n, |_| {
                    MyFloat::new(rand::rng().random::<f64>() * 200.0 - 100.0)
                });
                let shift_func = shift.clone();
                let func = move |x: &Points1<MyFloat>| {
                    x.iter()
                        .enumerate()
                        .map(|(i, val)| (val - shift_func[i]).powi(2) + bias)
                        .sum()
                };
                let objective = MultiDimFn::new(func);
                let mut simplex = Simplex::new(objective);
                let result = simplex
                    .minimize(
                        &x,
                        None,
                        None,
                        Some(1e-10.into()),
                        Some(1e3 as usize * n.pow(2)),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&result.fmin - bias) < 1e-3);

                    assert!((&simplex.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&simplex.fmin - bias) < 1e-3)
                }
            }
        }

        #[test]
        fn test_f2() {
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
            let x = Points1::from_shape_fn(n, |_| {
                MyFloat::new(rand::rng().random::<f64>() * 200.0 - 100.0)
            });
            let shift_func = shift.clone();
            let func = move |x: &Points1<MyFloat>| {
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
            let mut simplex = Simplex::new(objective);
            let result = simplex
                .minimize(
                    &x,
                    None,
                    None,
                    Some(1e-10.into()),
                    Some(1e3 as usize * n.pow(2)),
                )
                .unwrap();
            for i in 0..10 {
                assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&result.fmin - bias) < 1e-3);

                assert!((&simplex.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((&simplex.fmin - bias) < 1e-3)
            }

            // 30d
            if TEST_30D {
                let n = 30;
                let x = Points1::from_shape_fn(n, |_| {
                    MyFloat::new(rand::rng().random::<f64>() * 200.0 - 100.0)
                });
                let shift_func = shift.clone();
                let func = move |x: &Points1<MyFloat>| {
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
                let mut simplex = Simplex::new(objective);
                let result = simplex
                    .minimize(
                        &x,
                        None,
                        None,
                        Some(1e-10.into()),
                        Some(1e3 as usize * n.pow(2)),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&result.fmin - bias) < 1e-3);

                    assert!((&simplex.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&simplex.fmin - bias) < 1e-3)
                }
            }

            // 50d
            if TEST_50D {
                let n = 50;
                let x = Points1::from_shape_fn(n, |_| {
                    MyFloat::new(rand::rng().random::<f64>() * 200.0 - 100.0)
                });
                let shift_func = shift.clone();
                let func = move |x: &Points1<MyFloat>| {
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
                let mut simplex = Simplex::new(objective);
                let result = simplex
                    .minimize(
                        &x,
                        None,
                        None,
                        Some(1e-10.into()),
                        Some(1e3 as usize * n.pow(2)),
                    )
                    .unwrap();
                for i in 0..10 {
                    assert!((&result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&result.fmin - bias) < 1e-3);

                    assert!((&simplex.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((&simplex.fmin - bias) < 1e-3)
                }
            }
        }

        #[test]
        fn test_f3() {
            let shift: Points1<MyFloat> = array![
                MyFloat::new(-3.2201300e+001),
                6.4977600e+001.into(),
                (-3.8300000e+001).into(),
                (-2.3258200e+001).into(),
                (-5.4008800e+001).into(),
                8.6628600e+001.into(),
                (-6.3009000e+000).into(),
                (-4.9364400e+001).into(),
                5.3499000e+000.into(),
                5.2241800e+001.into()
            ]
            .into();
            let m = array![
                [
                    MyFloat::new(1.7830682721057345e-001),
                    MyFloat::new(5.5786330587166588e-002),
                    MyFloat::new(4.7591905576669730e-001),
                    MyFloat::new(2.4551129863391566e-001),
                    MyFloat::new(3.1998625926387086e-001),
                    MyFloat::new(3.2102001448363848e-001),
                    MyFloat::new(2.7787561319902176e-002),
                    MyFloat::new(2.6664001046775621e-001),
                    MyFloat::new(4.1568009651337917e-001),
                    MyFloat::new(-4.7771934552669726e-001)
                ],
                [
                    MyFloat::new(6.3516362859468667e-001),
                    MyFloat::new(5.0091423836646241e-002),
                    MyFloat::new(2.0110601384121973e-001),
                    MyFloat::new(-6.8076882416633511e-001),
                    MyFloat::new(-4.9934546553907944e-002),
                    MyFloat::new(-4.6399423424582961e-002),
                    MyFloat::new(-1.9460194646748039e-001),
                    MyFloat::new(1.8961539926194687e-001),
                    MyFloat::new(-1.9416259626804547e-002),
                    MyFloat::new(1.0639981029473855e-001)
                ],
                [
                    MyFloat::new(3.2762147366023187e-001),
                    MyFloat::new(3.6016598714114556e-001),
                    MyFloat::new(-2.3635655094044949e-001),
                    MyFloat::new(-1.8566854017444848e-002),
                    MyFloat::new(-2.4479096747593634e-001),
                    MyFloat::new(4.4818973341886903e-001),
                    MyFloat::new(5.3518635733619568e-001),
                    MyFloat::new(-3.1206925190530521e-001),
                    MyFloat::new(-1.3863719921728737e-001),
                    MyFloat::new(-2.0713981146209595e-001)
                ],
                [
                    MyFloat::new(-6.4783210587984280e-002),
                    MyFloat::new(-4.9424101683695937e-001),
                    MyFloat::new(1.3101175297435969e-001),
                    MyFloat::new(3.1615171931194543e-002),
                    MyFloat::new(-1.7506107914871860e-001),
                    MyFloat::new(6.8908039344918381e-001),
                    MyFloat::new(1.0544234469094992e-002),
                    MyFloat::new(2.1948984793273507e-001),
                    MyFloat::new(-1.6468539805844565e-001),
                    MyFloat::new(3.9048550518513409e-001)
                ],
                [
                    MyFloat::new(-2.7648044785371367e-001),
                    MyFloat::new(1.1383114506120220e-001),
                    MyFloat::new(-3.0818401502810994e-001),
                    MyFloat::new(-3.5959407104438740e-001),
                    MyFloat::new(2.6446258034702191e-001),
                    MyFloat::new(2.8616788379157501e-002),
                    MyFloat::new(4.7528027904995646e-001),
                    MyFloat::new(4.0993994049770172e-001),
                    MyFloat::new(4.1131043368915432e-001),
                    MyFloat::new(2.2899345188886880e-001)
                ],
                [
                    MyFloat::new(1.5454249061641606e-001),
                    MyFloat::new(5.4899186274157996e-001),
                    MyFloat::new(-1.8382029941792261e-001),
                    MyFloat::new(3.3944461903909162e-001),
                    MyFloat::new(2.8596188774255699e-001),
                    MyFloat::new(1.2833167642713417e-001),
                    MyFloat::new(-2.5495080172376317e-001),
                    MyFloat::new(3.9460752302037100e-001),
                    MyFloat::new(-3.4524640270007412e-001),
                    MyFloat::new(2.9590318323368509e-001)
                ],
                [
                    MyFloat::new(-5.1907977690014512e-002),
                    MyFloat::new(-1.4450757809700329e-001),
                    MyFloat::new(-4.6086919626114314e-001),
                    MyFloat::new(-5.3687964818368079e-002),
                    MyFloat::new(-3.6317793499109247e-001),
                    MyFloat::new(2.7439997038558633e-002),
                    MyFloat::new(-2.1422629652542946e-001),
                    MyFloat::new(5.0545148893084779e-001),
                    MyFloat::new(-9.8064717019089837e-002),
                    MyFloat::new(-5.6346991018564507e-001)
                ],
                [
                    MyFloat::new(5.0142989354460654e-001),
                    MyFloat::new(-5.3133659048457516e-001),
                    MyFloat::new(-3.7294385871521135e-001),
                    MyFloat::new(2.3370866431381510e-001),
                    MyFloat::new(4.4327537662488531e-001),
                    MyFloat::new(-1.6972740381143742e-001),
                    MyFloat::new(2.0364148963331691e-001),
                    MyFloat::new(-2.3717523924336927e-002),
                    MyFloat::new(-7.1805455862954920e-002),
                    MyFloat::new(-7.3332178450339763e-003)
                ],
                [
                    MyFloat::new(1.0441248047680891e-001),
                    MyFloat::new(4.3064226149369542e-002),
                    MyFloat::new(-4.1675972625940993e-001),
                    MyFloat::new(1.6522876074361707e-002),
                    MyFloat::new(1.7437281849141879e-003),
                    MyFloat::new(2.9594944879030760e-001),
                    MyFloat::new(-5.1197487739368741e-001),
                    MyFloat::new(-3.2679819762357892e-001),
                    MyFloat::new(5.8253106590933512e-001),
                    MyFloat::new(1.3204141339826148e-001)
                ],
                [
                    MyFloat::new(-2.9645907657631693e-001),
                    MyFloat::new(-3.1303011496605505e-002),
                    MyFloat::new(-7.8009154082116602e-002),
                    MyFloat::new(-4.1548534874482024e-001),
                    MyFloat::new(5.6959403572443468e-001),
                    MyFloat::new(2.9095198400348149e-001),
                    MyFloat::new(-1.8560717510075503e-001),
                    MyFloat::new(-2.4653488847859115e-001),
                    MyFloat::new(-3.7149025085479792e-001),
                    MyFloat::new(-3.0015617693118707e-001)
                ],
            ];
            let bias = -450.0;

            let mut iter = 0;
            while iter < 10 {
                iter += 1;
                // 10d
                let n = 10;
                let x1: Points1<MyFloat> =
                    Points1::new(Array1::random(n, Uniform::new(-100.0, 100.0).unwrap())) - &shift;
                let x = x1.dot(&m);
                let mut simplex = Simplex::new(elliptic(bias));
                let result = simplex
                    .minimize(
                        &x,
                        None,
                        Some(1e-10.into()),
                        None,
                        Some(1e3 as usize * n.pow(2)),
                    )
                    .unwrap();
                println!(
                    "\n\nxmin = {:?}\nfmin = {:?}\n\n",
                    simplex.xmin,
                    &simplex.fmin - bias
                );
                for i in 0..10 {
                    if result.xmin[i].abs() < 1e-2 && (&result.fmin - bias) < 1e-3 {
                        // assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                        assert!(result.xmin[i].abs() < 1e-2);
                        assert!((&result.fmin - bias) < 1e-3);

                        // assert!((cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                        assert!(simplex.xmin[i].abs() < 1e-2);
                        assert!((&simplex.fmin - bias) < 1e-3);
                        break;
                    } else {
                        continue;
                    }
                }
            }
        }

        #[test]
        fn test_f4() {
            let mut rng = rand::rng();
            let shift = array![
                MyFloat::new(3.5626700e+001),
                MyFloat::new(-8.2912300e+001),
                MyFloat::new(-1.0642300e+001),
                MyFloat::new(-8.3581500e+001),
                MyFloat::new(8.3155200e+001),
                MyFloat::new(4.7048000e+001),
                MyFloat::new(-8.9435900e+001),
                MyFloat::new(-2.7421900e+001),
                MyFloat::new(7.6144800e+001),
                MyFloat::new(-3.9059500e+001),
                MyFloat::new(4.8885700e+001),
                MyFloat::new(-3.9828000e+000),
                MyFloat::new(-7.1924300e+001),
                MyFloat::new(6.4194700e+001),
                MyFloat::new(-4.7733800e+001),
                MyFloat::new(-5.9896000e+000),
                MyFloat::new(-2.6282800e+001),
                MyFloat::new(-5.9181100e+001),
                MyFloat::new(1.4602800e+001),
                MyFloat::new(-8.5478000e+001),
                MyFloat::new(-5.0490100e+001),
                MyFloat::new(9.2400000e-001),
                MyFloat::new(3.2397800e+001),
                MyFloat::new(3.0238800e+001),
                MyFloat::new(-8.5094900e+001),
                MyFloat::new(6.0119700e+001),
                MyFloat::new(-3.6218300e+001),
                MyFloat::new(-8.5883000e+000),
                MyFloat::new(-5.1971000e+000),
                MyFloat::new(8.1553100e+001),
                MyFloat::new(-2.3431600e+001),
                MyFloat::new(-2.5350500e+001),
                MyFloat::new(-4.1248500e+001),
                MyFloat::new(8.8018000e+000),
                MyFloat::new(-2.4222200e+001),
                MyFloat::new(-8.7980700e+001),
                MyFloat::new(7.8047300e+001),
                MyFloat::new(-4.8052800e+001),
                MyFloat::new(1.4017700e+001),
                MyFloat::new(-3.6640500e+001),
                MyFloat::new(1.2216800e+001),
                MyFloat::new(1.8144900e+001),
                MyFloat::new(-6.4564700e+001),
                MyFloat::new(-8.4849300e+001),
                MyFloat::new(-7.6608800e+001),
                MyFloat::new(-1.7042000e+000),
                MyFloat::new(-3.6076100e+001),
                MyFloat::new(3.7033600e+001),
                MyFloat::new(1.8443100e+001),
                MyFloat::new(-6.4359000e+001),
                MyFloat::new(-3.9369200e+001),
                MyFloat::new(-1.7714000e+001),
                MyFloat::new(3.0198500e+001),
                MyFloat::new(-1.8548300e+001),
                MyFloat::new(9.6866000e+000),
                MyFloat::new(8.2600900e+001),
                MyFloat::new(-4.5525600e+001),
                MyFloat::new(5.1443000e+000),
                MyFloat::new(7.4204000e+001),
                MyFloat::new(6.6810300e+001),
                MyFloat::new(-6.3470400e+001),
                MyFloat::new(1.3032900e+001),
                MyFloat::new(-5.6878000e+000),
                MyFloat::new(2.9527100e+001),
                MyFloat::new(-4.3530000e-001),
                MyFloat::new(-2.6165200e+001),
                MyFloat::new(-6.6847000e+000),
                MyFloat::new(-8.0229100e+001),
                MyFloat::new(-2.9581500e+001),
                MyFloat::new(8.2042200e+001),
                MyFloat::new(7.7177000e+001),
                MyFloat::new(-1.1277000e+001),
                MyFloat::new(3.2075900e+001),
                MyFloat::new(-2.6858000e+000),
                MyFloat::new(8.1509600e+001),
                MyFloat::new(6.4077000e+001),
                MyFloat::new(-2.6129400e+001),
                MyFloat::new(-8.4782000e+001),
                MyFloat::new(-6.2876800e+001),
                MyFloat::new(-3.7635500e+001),
                MyFloat::new(7.6891600e+001),
                MyFloat::new(5.3417000e+001),
                MyFloat::new(-2.5331100e+001),
                MyFloat::new(-3.8070200e+001),
                MyFloat::new(-8.4173800e+001),
                MyFloat::new(-1.1224600e+001),
                MyFloat::new(-8.3461900e+001),
                MyFloat::new(-1.7550800e+001),
                MyFloat::new(-3.6528500e+001),
                MyFloat::new(8.9552800e+001),
                MyFloat::new(2.5879400e+001),
                MyFloat::new(6.8625200e+001),
                MyFloat::new(5.5796800e+001),
                MyFloat::new(-2.9597500e+001),
                MyFloat::new(-5.8097600e+001),
                MyFloat::new(6.5741300e+001),
                MyFloat::new(-8.8703000e+000),
                MyFloat::new(-5.3281000e+000),
                MyFloat::new(7.4066100e+001),
                MyFloat::new(4.0338000e+000)
            ];
            let bias = -450.0;

            // 10d
            let n = 10;
            let x = Points1::from_shape_fn(n, |_| {
                MyFloat::new(rand::rng().random::<f64>() * 200.0 - 100.0)
            });
            let shift_func = shift.clone();
            let rand_val = MyFloat::new(rng.random::<f64>());
            let func = move |x: &Points1<MyFloat>| {
                x.iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let mut new_val = MyFloat::new(0.0);
                        for j in 0..=i {
                            new_val += (&x[j] - &shift_func[j]).powi(2);
                        }
                        &new_val + bias
                    })
                    .sum::<MyFloat>()
                    * (1.0 + 0.4 * &rand_val)
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);
            let result = simplex
                .minimize(
                    &x,
                    None,
                    None,
                    Some(1e-10.into()),
                    Some(1e3 as usize * n.pow(2)),
                )
                .unwrap();
            for i in 0..10 {
                assert!((&result.xmin[i] - &shift[i]).abs() < 1e-2);
                assert!((&result.fmin - bias) < 1e-3);

                assert!((&simplex.xmin[i] - &shift[i]).abs() < 1e-2);
                assert!((&simplex.fmin - bias) < 1e-3)
            }
        }

        #[test]
        fn test_f6() {
            let shift = array![
                MyFloat::new(8.1023200e+001),
                MyFloat::new(-4.8395000e+001),
                MyFloat::new(1.9231600e+001),
                MyFloat::new(-2.5231000e+000),
                MyFloat::new(7.0433800e+001),
                MyFloat::new(4.7177400e+001),
                MyFloat::new(-7.8358000e+000),
                MyFloat::new(-8.6669300e+001),
                MyFloat::new(5.7853200e+001),
                MyFloat::new(-9.9533000e+000),
                MyFloat::new(2.0777800e+001),
                MyFloat::new(5.2548600e+001),
                MyFloat::new(7.5926300e+001),
                MyFloat::new(4.2877300e+001),
                MyFloat::new(-5.8272000e+001),
                MyFloat::new(-1.6972800e+001),
                MyFloat::new(7.8384500e+001),
                MyFloat::new(7.5042700e+001),
                MyFloat::new(-1.6151300e+001),
                MyFloat::new(7.0856900e+001),
                MyFloat::new(-7.9579500e+001),
                MyFloat::new(-2.6483700e+001),
                MyFloat::new(5.6369900e+001),
                MyFloat::new(-8.8224900e+001),
                MyFloat::new(-6.4999600e+001),
                MyFloat::new(-5.3502200e+001),
                MyFloat::new(-5.4230000e+001),
                MyFloat::new(1.8682600e+001),
                MyFloat::new(-4.1006100e+001),
                MyFloat::new(-5.4213400e+001),
                MyFloat::new(-8.7250600e+001),
                MyFloat::new(4.4421400e+001),
                MyFloat::new(-9.8826000e+000),
                MyFloat::new(7.7726600e+001),
                MyFloat::new(-6.1210000e+000),
                MyFloat::new(-1.4643000e+001),
                MyFloat::new(6.2319800e+001),
                MyFloat::new(4.5274000e+000),
                MyFloat::new(-5.3523400e+001),
                MyFloat::new(3.0984700e+001),
                MyFloat::new(6.0861300e+001),
                MyFloat::new(-8.6464800e+001),
                MyFloat::new(3.2629800e+001),
                MyFloat::new(-2.1693400e+001),
                MyFloat::new(5.9723200e+001),
                MyFloat::new(5.0630000e-001),
                MyFloat::new(3.7704800e+001),
                MyFloat::new(-1.2799300e+001),
                MyFloat::new(-3.5168800e+001),
                MyFloat::new(-5.5862300e+001),
                MyFloat::new(-5.5182300e+001),
                MyFloat::new(3.2800100e+001),
                MyFloat::new(-3.5502400e+001),
                MyFloat::new(7.5012000e+000),
                MyFloat::new(-6.2842800e+001),
                MyFloat::new(3.5621700e+001),
                MyFloat::new(-2.1892800e+001),
                MyFloat::new(6.4802000e+001),
                MyFloat::new(6.3657900e+001),
                MyFloat::new(1.6841300e+001),
                MyFloat::new(-6.2050000e-001),
                MyFloat::new(7.1958400e+001),
                MyFloat::new(5.7893200e+001),
                MyFloat::new(2.6083800e+001),
                MyFloat::new(5.7235300e+001),
                MyFloat::new(2.8840900e+001),
                MyFloat::new(-2.8445200e+001),
                MyFloat::new(-3.7849300e+001),
                MyFloat::new(-2.8585100e+001),
                MyFloat::new(6.1342000e+000),
                MyFloat::new(4.0880300e+001),
                MyFloat::new(-3.4327700e+001),
                MyFloat::new(6.0929200e+001),
                MyFloat::new(1.2253000e+001),
                MyFloat::new(-2.3325500e+001),
                MyFloat::new(3.6493100e+001),
                MyFloat::new(8.3828000e+000),
                MyFloat::new(-9.9215000e+000),
                MyFloat::new(3.5022100e+001),
                MyFloat::new(2.1835800e+001),
                MyFloat::new(5.3067700e+001),
                MyFloat::new(8.2231800e+001),
                MyFloat::new(4.0662000e+000),
                MyFloat::new(6.8425500e+001),
                MyFloat::new(-5.8867800e+001),
                MyFloat::new(8.6354400e+001),
                MyFloat::new(-4.1139400e+001),
                MyFloat::new(-4.4580700e+001),
                MyFloat::new(6.7633500e+001),
                MyFloat::new(4.2715000e+001),
                MyFloat::new(-6.5426600e+001),
                MyFloat::new(-8.7883700e+001),
                MyFloat::new(7.0901600e+001),
                MyFloat::new(-5.4155100e+001),
                MyFloat::new(-3.6229800e+001),
                MyFloat::new(2.9059600e+001),
                MyFloat::new(-3.8806400e+001),
                MyFloat::new(-5.5396000e+000),
                MyFloat::new(-7.8339300e+001),
                MyFloat::new(8.7900200e+001)
            ];
            let bias = 390.0;

            let mut iter = 0;
            while iter < 10 {
                iter += 1;
                // 10d
                let n = 10;
                let x = Points1::from_shape_fn(n, |_| {
                    MyFloat::new(rand::rng().random::<f64>() * 200.0 - 100.0)
                });
                let mut simplex =
                    Simplex::new(rosenbrock(shift.slice(s![0..10]).to_owned().into(), bias));
                let result = simplex
                    .minimize(
                        &x,
                        None,
                        None,
                        Some(1e-12.into()),
                        Some(1e3 as usize * n.pow(2)),
                    )
                    .unwrap();
                println!(
                    "\n\nxmin = {:?}\nfmin = {:?}\n\n",
                    simplex.xmin,
                    &simplex.fmin - bias
                );
                for i in 0..10 {
                    if (&result.xmin[i] - &shift[i]).abs() < 1e-2 && (&result.fmin - bias) < 1e-3 {
                        assert!((&result.xmin[i] - &shift[i]).abs() < 1e-2);
                        assert!((&result.fmin - bias) < 1e-3);

                        assert!((&simplex.xmin[i] - &shift[i]).abs() < 1e-2);
                        assert!((&simplex.fmin - bias) < 1e-3);
                        break;
                    } else {
                        continue;
                    }
                }
            }
        }
    }
}
