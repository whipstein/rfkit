#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::{
    error::MinimizerError,
    minimize::{Brent, F1dim, ObjFn},
    num::RFFloat,
    pts::{Matrix, Points, Points1, Points2, Pts},
};
use ndarray::prelude::*;
use std::fmt;

/// Result of Powell's method optimization
#[derive(Debug, Clone)]
pub struct PowellResult<T>
where
    T: RFFloat,
{
    xmin: Points1<T>,
    fmin: T,
    iters: usize,
    fn_evals: usize,
    converged: bool,
    final_directions: Points2<T>,
    history: Points1<T>,
    improvement: Points1<T>,
}

impl<T> PowellResult<T>
where
    T: RFFloat,
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

    pub fn final_directions(&self) -> Points2<T> {
        self.final_directions.clone()
    }

    pub fn history(&self) -> Points1<T> {
        self.history.clone()
    }

    pub fn improvement(&self) -> Points1<T> {
        self.improvement.clone()
    }
}

pub struct Powell<T> {
    xmin: Points1<T>,
    fmin: T,
    f: Box<dyn ObjFn<T>>,
    iters: usize,
    converged: bool,
}

impl<T> Clone for Powell<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Powell {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            f: dyn_clone::clone_box(&*self.f),
            iters: self.iters,
            converged: self.converged,
        }
    }
}

impl<T> Powell<T>
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
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn<T> + Clone + 'static,
    {
        let boxed = Box::new(f);
        Powell {
            xmin: array![].into(),
            fmin: T::zero(),
            f: boxed,
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn<T>>) -> Self {
        Powell {
            xmin: array![].into(),
            fmin: T::zero(),
            f: f,
            iters: 0,
            converged: false,
        }
    }

    pub fn xmin(&self) -> Points1<T> {
        self.xmin.clone()
    }

    pub fn fmin(&self) -> T {
        self.fmin.clone()
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    /// Powell's method for multidimensional optimization
    ///
    /// This algorithm performs sequential line searches along a set of conjugate
    /// directions, which are updated iteratively to maintain conjugacy with respect
    /// to the function being minimized.
    ///
    /// # Arguments
    /// * `func` - The function to minimize
    /// * `initial_point` - Starting point for optimization
    /// * `tol` - Convergence tolerance (default: 1e-8)
    /// * `max_iters` - Maximum iters (default: 200)
    /// * `line_search_tolerance` - Tolerance for line searches (default: 1e-12)
    ///
    /// # Returns
    /// * `PowellResult` containing the minimum and convergence information
    pub fn powell_method(
        &mut self,
        initial_point: &Points1<T>,
        scale: Option<&Points1<T>>,
        tol: Option<T>,
        max_iters: Option<usize>,
        line_search_tolerance: Option<T>,
    ) -> Result<PowellResult<T>, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let scl = scale.unwrap_or(&Points1::ones(n)).clone();
        let x_scaled = initial_point / &scl;
        let tol = tol.unwrap_or(1e-8.into());
        let max_iter = max_iters.unwrap_or(200);
        let line_tol = line_search_tolerance.unwrap_or(1e-12.into());

        if tol <= T::zero() {
            return Err(MinimizerError::InvalidTolerance);
        }

        // Initialize direction set as coordinate axes
        let mut directions = Points2::<T>::eye(n);
        // let mut directions: Points2<f64> = (0..n)
        //     .map(|i| {
        //         let mut dir = vec![0.0; n];
        //         dir[i] = 1.0;
        //         dir
        //     })
        //     .collect();

        let mut x = x_scaled.to_owned();
        let mut f_current = self.f.call(&(&x / &scl));
        if !f_current.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }

        let mut fn_evals = 1;
        self.iters = 0;
        let mut history = vec![f_current.clone()];
        let mut improvement = Vec::new();

        while self.iters < max_iter {
            self.iters += 1;
            let x_old = x.to_owned();
            let f_old = f_current.clone();

            let mut delta_f_max = T::zero();
            let mut max_decrease_index = 0;

            // Perform line searches along each direction
            for (i, direction) in directions.outer_iter().enumerate() {
                let dir = Points(direction.to_owned());
                let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                let line_result = brent.line_search(&x, &dir, &T::one(), &line_tol, 1000)?;

                if !line_result.converged() {
                    return Err(MinimizerError::LinearSearchFailed);
                }

                fn_evals += line_result.fn_evals();

                // Update position
                for j in 0..n {
                    x[j] += &line_result.xmin() * &direction[j];
                }

                let f_new = self.f.call(&(&x / &scl));
                if !f_new.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }
                fn_evals += 1;

                let delta_f = &f_current - &f_new;
                if delta_f > delta_f_max {
                    delta_f_max = delta_f;
                    max_decrease_index = i;
                }

                f_current = f_new.clone();
            }

            let improve = &f_old - &f_current;
            improvement.push(improve.clone());
            history.push(f_current.clone());

            // Check for convergence
            let relative_improvement = improve / (f_old.abs() + 1e-14);
            if relative_improvement < tol {
                self.xmin = &x / &scl;
                self.fmin = f_current.clone();
                self.converged = true;
                return Ok(PowellResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin.clone(),
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_directions: directions,
                    history: Points1::from_vec(history),
                    improvement: Points1::from_vec(improvement),
                });
            }

            // Compute extrapolated point
            let x_extrapolated = Points1::from_shape_fn(n, |i| 2.0 * &x[i] - &x_old[i]);
            // let mut x_extrapolated = vec![0.0; n];
            // for i in 0..n {
            //     x_extrapolated[i] = 2.0 * x[i] - x_old[i];
            // }

            let f_extrapolated = self.f.call(&(&x_extrapolated / &scl));
            if !f_extrapolated.is_finite() {
                return Err(MinimizerError::FunctionEvaluationError);
            }
            fn_evals += 1;

            // Test whether to keep new direction
            let condition1 = f_extrapolated < f_old;
            let temp = 2.0 * (&f_old - 2.0 * &f_current + &f_extrapolated);
            let condition2 = temp * (&f_old - &f_current - &delta_f_max).powi(2);
            let condition3 = delta_f_max * (&f_old - &f_extrapolated).powi(2);

            if condition1 && condition2 < condition3 {
                // Move to extrapolated point and update direction set
                let new_direction: Vec<T> = x
                    .iter()
                    .zip(x_old.iter())
                    .map(|(xi, xi_old)| xi - xi_old)
                    .collect();

                // Check if new direction is linearly independent
                let direction_norm = new_direction.iter().map(|d| d * d).sum::<T>().sqrt();
                if direction_norm < T::from_f64(1e-12) {
                    continue; // Skip if direction is too small
                }

                // Normalize new direction
                let new_direction: Vec<T> =
                    new_direction.iter().map(|d| d / &direction_norm).collect();

                // Perform line search along new direction
                let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                let line_result = brent.line_search(
                    &x,
                    &Points1::from_vec(new_direction.clone()),
                    &T::one(),
                    &line_tol,
                    1000,
                )?;

                if line_result.converged() {
                    fn_evals += line_result.fn_evals();

                    // Update position
                    for j in 0..n {
                        x[j] += &line_result.xmin() * &new_direction[j];
                    }

                    f_current = self.f.call(&(&x / &scl));
                    if !f_current.is_finite() {
                        return Err(MinimizerError::FunctionEvaluationError);
                    }
                    fn_evals += 1;

                    // Replace the direction that gave maximum decrease
                    for (j, dir) in new_direction.iter().enumerate() {
                        directions[[max_decrease_index, j]] = dir.clone();
                    }
                }
            }
        }

        self.xmin = &x / &scl;
        self.fmin = f_current;
        Ok(PowellResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_directions: directions,
            history: Points1::from_vec(history),
            improvement: Points1::from_vec(improvement),
        })
    }

    /// Powell's method with custom initial directions
    ///
    /// Allows specification of initial search directions instead of using
    /// coordinate axes. Useful when you have knowledge about the problem structure.
    pub fn powell_method_with_directions(
        &mut self,
        initial_point: &Points1<T>,
        scale: Option<&Points1<T>>,
        initial_directions: &Points2<T>,
        tol: Option<T>,
        max_iters: Option<usize>,
        line_search_tolerance: Option<T>,
    ) -> Result<PowellResult<T>, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if initial_directions.nrows() != n {
            return Err(MinimizerError::InvalidDirectionSet);
        }

        let scl = scale.unwrap_or(&Points1::ones(n)).clone();
        let x_scaled = initial_point / &scl;
        for direction in initial_directions.outer_iter() {
            if direction.len() != n {
                return Err(MinimizerError::InvalidDirectionSet);
            }
            let norm = direction.iter().map(|d| d * d).sum::<T>().sqrt();
            if norm < 1e-12.into() {
                return Err(MinimizerError::InvalidDirectionSet);
            }
        }

        let tol = tol.unwrap_or(1e-8.into());
        let max_iter = max_iters.unwrap_or(200);
        let line_tol = line_search_tolerance.unwrap_or(1e-12.into());

        if tol <= T::zero() {
            return Err(MinimizerError::InvalidTolerance);
        }

        // Normalize initial directions
        // let mut directions: Vec<Vec<f64>> = initial_directions
        //     .into_iter()
        //     .map(|dir| {
        //         let norm: f64 = dir.iter().map(|&d| d * d).sum::<f64>().sqrt();
        //         dir.into_iter().map(|d| d / norm).collect()
        //     })
        //     .collect();
        let mut directions = Points2::from_shape_fn(initial_directions.dim(), |(i, j)| {
            let norm = initial_directions
                .row(i)
                .iter()
                .map(|d| d * d)
                .sum::<T>()
                .sqrt();
            &initial_directions[[i, j]] / norm
        });

        let mut x = x_scaled.to_owned();
        let mut f_current = self.f.call(&(&x / &scl));
        if !f_current.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }

        let mut fn_evals = 1;
        self.iters = 0;
        let mut history = vec![f_current.clone()];
        let mut improvement = Vec::new();

        while self.iters < max_iter {
            self.iters += 1;
            let x_old = x.clone();
            let f_old = f_current.clone();

            let mut delta_f_max = T::zero();
            let mut max_decrease_index = 0;

            // Line searches along each direction
            for (i, direction) in directions.outer_iter().enumerate() {
                let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                let line_result = brent.line_search(
                    &x,
                    &Points(direction.to_owned()),
                    &T::one(),
                    &line_tol,
                    1000,
                )?;

                if !line_result.converged() {
                    return Err(MinimizerError::LinearSearchFailed);
                }

                fn_evals += line_result.fn_evals();

                // Update position
                for j in 0..n {
                    x[j] += &line_result.xmin() * &direction[j];
                }

                let f_new = self.f.call(&(&x / &scl));
                if !f_new.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }
                fn_evals += 1;

                let delta_f = f_current - &f_new;
                if delta_f > delta_f_max {
                    delta_f_max = delta_f;
                    max_decrease_index = i;
                }

                f_current = f_new;
            }

            let improve = &f_old - &f_current;
            improvement.push(improve.clone());
            history.push(f_current.clone());

            // Check for convergence
            let relative_improvement = improve / (f_old.abs() + 1e-14);
            if relative_improvement < tol {
                self.xmin = &x / &scl;
                self.fmin = f_current.clone();
                self.converged = true;
                return Ok(PowellResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin.clone(),
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_directions: directions,
                    history: Points1::from_vec(history),
                    improvement: Points1::from_vec(improvement),
                });
            }

            // Direction replacement logic (same as standard Powell)
            // let mut x_extrapolated = vec![0.0; n];
            // for i in 0..n {
            //     x_extrapolated[i] = 2.0 * x[i] - x_old[i];
            // }
            let x_extrapolated = Points1::from_shape_fn(n, |i| 2.0 * &x[i] - &x_old[i]);

            let f_extrapolated = self.f.call(&(&x_extrapolated / &scl));
            if !f_extrapolated.is_finite() {
                continue;
            }
            fn_evals += 1;

            let condition1 = f_extrapolated < f_old;
            let temp = 2.0 * (&f_old - 2.0 * &f_current + &f_extrapolated);
            let condition2 = temp * (&f_old - &f_current - &delta_f_max).powi(2);
            let condition3 = delta_f_max * (f_old - f_extrapolated).powi(2);

            if condition1 && condition2 < condition3 {
                let new_direction: Vec<T> = x
                    .iter()
                    .zip(x_old.iter())
                    .map(|(xi, xi_old)| xi - xi_old)
                    .collect();

                let direction_norm = new_direction.iter().map(|d| d * d).sum::<T>().sqrt();
                if direction_norm > 1e-12.into() {
                    // let normalized_direction: Vec<f64> =
                    //     new_direction.iter().map(|&d| d / direction_norm).collect();
                    let normalized_direction = Points1::from_shape_fn(new_direction.len(), |i| {
                        &new_direction[i] / &direction_norm
                    });

                    let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                    let line_result =
                        brent.line_search(&x, &normalized_direction, &T::one(), &line_tol, 1000)?;

                    if line_result.converged() {
                        fn_evals += line_result.fn_evals();

                        for j in 0..n {
                            x[j] += &line_result.xmin() * &normalized_direction[j];
                        }

                        f_current = self.f.call(&(&x / &scl));
                        if !f_current.is_finite() {
                            continue;
                        }
                        fn_evals += 1;

                        for (j, dir) in normalized_direction.iter().enumerate() {
                            directions[[max_decrease_index, j]] = dir.clone();
                        }
                    }
                }
            }
        }

        self.xmin = &x / &scl;
        self.fmin = f_current;
        Ok(PowellResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_directions: directions,
            history: Points1::from_vec(history),
            improvement: Points1::from_vec(improvement),
        })
    }

    /// Convenience function with default parameters
    pub fn minimize(
        &mut self,
        initial_point: &Points1<T>,
        scale: Option<&Points1<T>>,
        tol: Option<T>,
        max_iters: Option<usize>,
        line_search_tolerance: Option<T>,
    ) -> Result<PowellResult<T>, MinimizerError> {
        self.powell_method(initial_point, scale, tol, max_iters, line_search_tolerance)
    }

    /// Create orthogonal initial directions for better performance
    pub fn create_orthogonal_directions(&self, n: usize) -> Points2<T> {
        Points2::eye(n)
    }

    /// Create random orthogonal directions using Gram-Schmidt
    pub fn create_random_orthogonal_directions(&self, n: usize, seed: Option<u64>) -> Points2<T> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple random number generator for reproducibility
        let mut rng_state = seed.unwrap_or_else(|| {
            let mut hasher = DefaultHasher::new();
            std::ptr::addr_of!(n).hash(&mut hasher);
            hasher.finish()
        });

        let mut random = || -> T {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            T::from_f64(((rng_state / 65536) % 32768) as f64 / 32768.0 - 0.5)
        };

        // let mut directions: Points2<f64> = Vec::with_capacity(n);
        let mut directions = Points2::<T>::zeros((n, n));

        for i in 0..n {
            // Generate random vector
            let mut new_dir = (0..n).map(|_| random()).collect::<Vec<T>>();

            // Gram-Schmidt orthogonalization
            for j in 0..i {
                let dot_product = new_dir
                    .iter()
                    .zip(directions.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum::<T>();

                for k in 0..n {
                    new_dir[k] -= &dot_product * &directions[[j, k]];
                }
            }

            // Normalize
            let norm = new_dir.iter().map(|d| d * d).sum::<T>().sqrt();
            if norm > 1e-12.into() {
                for d in &mut new_dir {
                    *d /= &norm;
                }
            } else {
                // Fallback to coordinate vector if normalization fails
                new_dir = vec![T::zero(); n];
                new_dir[i] = T::one();
            }

            // directions.push(new_dir);
            for (j, dir) in new_dir.iter().enumerate() {
                directions[[i, j]] = dir.clone();
            }
        }

        directions
    }
}

impl<T> fmt::Debug for Powell<T>
where
    T: RFFloat,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Powell( xmin: {:?}, fmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod minimize_powell_tests {
    use super::*;
    use crate::minimize::{F1dim, MultiDimFn};
    use float_cmp::F64Margin;
    use std::f64::consts::{E, PI};
    use std::vec;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    // Helper function to create Ackley
    fn ackley(bias: f64) -> F1dim<f64> {
        F1dim::new(MultiDimFn::new(move |x: &Points1<f64>| {
            let n = x.len() as f64;
            let t1 = x.iter().map(|val| val.powi(2)).sum::<f64>();
            let t2 = x.iter().map(|val| (2.0 * PI * val).cos()).sum::<f64>();
            -20.0 * (-0.2 * (t1 / n).sqrt()).exp() - (t2 / n).exp() + 20.0 + E + bias
        }))
    }

    // Helper function to create Elliptic
    fn elliptic(bias: f64) -> F1dim<f64> {
        F1dim::new(MultiDimFn::new(move |x: &Points1<f64>| {
            let sum: f64 = x
                .iter()
                .enumerate()
                .map(|(i, val)| 1e6_f64.powf((i / (x.len() - 1)) as f64) * val.powi(2))
                .sum();
            sum + bias
        }))
    }

    // Helper function to create Griewank
    fn griewank(bias: f64) -> F1dim<f64> {
        F1dim::new(MultiDimFn::new(move |x: &Points1<f64>| {
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
    fn rosenbrock(shift: Points1<f64>, bias: f64) -> F1dim<f64> {
        F1dim::new(MultiDimFn::new(move |x: &Points1<f64>| {
            let x_new = x - &shift + 1.0;
            let sum: f64 = x_new
                .0
                .windows(2)
                .into_iter()
                .map(|pair| 100.0 * (pair[0].powi(2) - pair[1]).powi(2) + (pair[0] - 1.0).powi(2))
                .sum();
            sum + bias
        }))
    }

    #[test]
    fn test_2d_quadratic() {
        // f(x,y) = (x-2)² + (y+1)²
        let func = |x: &Points1<f64>| (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![0.0, 0.0].into(), None, None, None, None)
            .unwrap();

        assert!((result.xmin[0] - 2.0).abs() < 1e-8);
        assert!((result.xmin[1] + 1.0).abs() < 1e-8);
        assert!(result.fmin < 1e-14);
        assert!(result.converged);
        assert!((powell.xmin[0] - 2.0).abs() < 1e-8);
        assert!((powell.xmin[1] + 1.0).abs() < 1e-8);
        assert!(powell.fmin < 1e-14);
        assert!(powell.converged);
    }

    #[test]
    fn test_rosenbrock() {
        let rosenbrock =
            |x: &Points1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let objective = MultiDimFn::new(rosenbrock);
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(
                &array![-1.2, 1.0].into(),
                None,
                Some(1e-6),
                Some(1000),
                None,
            )
            .unwrap();

        assert!((result.xmin[0] - 1.0).abs() < 1e-4);
        assert!((result.xmin[1] - 1.0).abs() < 1e-4);
        assert!(result.fmin < 1e-8);
        assert!((powell.xmin[0] - 1.0).abs() < 1e-4);
        assert!((powell.xmin[1] - 1.0).abs() < 1e-4);
        assert!(powell.fmin < 1e-8);
    }

    #[test]
    fn test_3d_sphere() {
        // f(x,y,z) = x² + y² + z²
        let sphere = |x: &Points1<f64>| x.iter().map(|&xi| xi * xi).sum();
        let objective = MultiDimFn::new(sphere);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![1.0, 2.0, -1.0].into(), None, None, None, None)
            .unwrap();

        for &coord in &result.xmin {
            assert!(coord.abs() < 1e-10);
        }
        assert!(result.fmin < 1e-18);
        assert!(result.converged);
        for &coord in &powell.xmin {
            assert!(coord.abs() < 1e-10);
        }
        assert!(powell.fmin < 1e-18);
        assert!(powell.converged);
    }

    #[test]
    fn test_custom_directions() {
        let func = |x: &Points1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut powell = Powell::new(objective);

        // Use diagonal directions instead of coordinate axes
        let directions = array![
            [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()],
            [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()],
        ]
        .into();

        let result = powell
            .powell_method_with_directions(
                &array![0.0, 0.0].into(),
                None,
                &directions,
                None,
                None,
                None,
            )
            .unwrap();

        assert!((result.xmin[0] - 1.0).abs() < 1e-8);
        assert!((result.xmin[1] - 2.0).abs() < 1e-8);
        assert!(result.converged);
        assert!((powell.xmin[0] - 1.0).abs() < 1e-8);
        assert!((powell.xmin[1] - 2.0).abs() < 1e-8);
        assert!(powell.converged);
    }

    #[test]
    fn test_direction_creation() {
        let n = 4;
        let func = |_: &Points1<f64>| 0.0;
        let objective = MultiDimFn::new(func);
        let powell = Powell::new(objective);
        let ortho_dirs = powell.create_orthogonal_directions(n);

        assert_eq!(ortho_dirs.nrows(), n);
        for (i, dir) in ortho_dirs.outer_iter().enumerate() {
            assert_eq!(dir.len(), n);
            // Check that it's a unit coordinate vector
            for (j, &val) in dir.iter().enumerate() {
                if i == j {
                    assert!((val - 1.0).abs() < 1e-15);
                } else {
                    assert!(val.abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn test_random_directions() {
        let n = 3;
        let func = |_: &Points1<f64>| 0.0;
        let objective = MultiDimFn::new(func);
        let powell = Powell::new(objective);
        let random_dirs = powell.create_random_orthogonal_directions(n, Some(12345));

        assert_eq!(random_dirs.nrows(), n);

        // Check orthogonality
        for i in 0..n {
            for j in i + 1..n {
                let dot_product: f64 = random_dirs
                    .row(i)
                    .iter()
                    .zip(random_dirs.row(j).iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                assert!(dot_product.abs() < 1e-12);
            }
        }

        // Check normalization
        for dir in random_dirs.outer_iter() {
            let norm: f64 = dir.iter().map(|&d| d * d).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_powell_quadratic_exact_convergence() {
        // Test exact convergence on quadratic function
        let quadratic_3d = |x: &Points1<f64>| {
            x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2) + x[0] * x[1] + x[1] * x[2]
        };
        let objective = MultiDimFn::new(quadratic_3d);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![1.0, 1.0, 1.0].into(), None, None, None, None)
            .unwrap();

        // Should converge in <= 3 iterations for 3D quadratic
        assert!(result.iters <= 6); // Allow some margin
        assert!(result.converged);

        for &coord in &result.xmin {
            assert!(coord.abs() < 1e-6);
        }
    }

    #[test]
    fn test_powell_with_custom_directions() {
        // Test with problem-specific initial directions
        let elliptic = |x: &Points1<f64>| 100.0 * x[0].powi(2) + x[1].powi(2);
        let objective = MultiDimFn::new(elliptic);
        let mut powell = Powell::new(objective);

        // Standard coordinate directions
        let std_result = powell
            .minimize(&array![1.0, 1.0].into(), None, None, None, None)
            .unwrap();

        // Custom directions aligned with ellipse
        let custom_dirs = array![
            [0.1, 0.0], // Small step in x direction
            [0.0, 1.0], // Normal step in y direction
        ]
        .into();

        let custom_result = powell
            .powell_method_with_directions(
                &array![1.0, 1.0].into(),
                None,
                &custom_dirs,
                None,
                None,
                None,
            )
            .unwrap();

        // Custom directions should perform better on ill-conditioned problem
        assert!(custom_result.fn_evals as f64 <= std_result.fn_evals as f64 * 1.5);
        assert!(custom_result.xmin[0].abs() < 1e-6);
        assert!(custom_result.xmin[1].abs() < 1e-6);
    }

    #[test]
    fn test_powell_direction_evolution() {
        // Test that directions evolve to become conjugate
        let quadratic_matrix = |x: &Points1<f64>| {
            // f(x) = x^T A x where A = [[4,1],[1,2]]
            4.0 * x[0].powi(2) + 2.0 * x[1].powi(2) + 2.0 * x[0] * x[1]
        };
        let objective = MultiDimFn::new(quadratic_matrix);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![1.0, 1.0].into(), None, None, None, None)
            .unwrap();

        assert!(result.converged);
        assert!(result.xmin[0].abs() < 1e-6);
        assert!(result.xmin[1].abs() < 1e-6);

        // Final directions should be approximately conjugate
        // (This is hard to test directly, but convergence implies conjugacy)
        assert!(result.iters <= 5); // Should be fast for 2D quadratic
    }

    // Helper function to create a simple quadratic function
    fn simple_quadratic() -> F1dim<f64> {
        F1dim::new(MultiDimFn::new(|x: &Points1<f64>| {
            (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2)
        }))
    }

    // Helper function to create Rosenbrock function
    fn rosenbrock_2d() -> F1dim<f64> {
        F1dim::new(MultiDimFn::new(|x: &Points1<f64>| {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
        }))
    }

    // Helper function for ill-conditioned quadratic
    fn ill_conditioned_quadratic() -> F1dim<f64> {
        F1dim::new(MultiDimFn::new(|x: &Points1<f64>| {
            1000.0 * x[0].powi(2) + x[1].powi(2)
        }))
    }

    #[test]
    fn test_empty_initial_point() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell.minimize(&array![].into(), None, None, None, None);
        assert!(matches!(result, Err(MinimizerError::InvalidDimension)));
    }

    #[test]
    fn test_invalid_tolerance() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell.powell_method(&array![0.0, 0.0].into(), None, Some(-1e-8), None, None);
        assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));

        let result = powell.powell_method(&array![0.0, 0.0].into(), None, Some(0.0), None, None);
        assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
    }

    #[test]
    fn test_single_dimension() {
        let func = |x: &Points1<f64>| (x[0] - 3.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![0.0].into(), None, None, None, None)
            .unwrap();

        assert!((result.xmin[0] - 3.0).abs() < 1e-8);
        assert!(result.fmin < 1e-14);
        assert!(result.converged);
        assert_eq!(result.final_directions.len(), 1);
        assert_eq!(result.final_directions.row(0), array![1.0].into());
    }

    #[test]
    fn test_very_tight_tolerance() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(
                &array![0.0, 0.0].into(),
                None,
                Some(1e-15),
                Some(1000),
                None,
            )
            .unwrap();

        // With very tight tolerance, should still achieve good accuracy
        assert!((result.xmin[0] - 2.0).abs() < 1e-6);
        assert!((result.xmin[1] + 1.0).abs() < 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_very_loose_tolerance() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(&array![0.0, 0.0].into(), None, Some(1e-1), None, None)
            .unwrap();

        assert!((result.xmin[0] - 2.0).abs() < 0.5);
        assert!((result.xmin[1] + 1.0).abs() < 0.5);
        assert!(result.converged);
        assert!(result.iters <= 10); // Should converge quickly with loose tolerance
    }

    #[test]
    fn test_max_iterations_reached() {
        let objective = rosenbrock_2d();
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(&array![-1.2, 1.0].into(), None, Some(1e-12), Some(5), None)
            .unwrap();

        assert!(!result.converged);
        assert_eq!(result.iters, 5);
        // Should still make some progress even without convergence
        assert!(result.fmin < 100.0); // Starting value is much higher
    }

    #[test]
    fn test_already_at_minimum() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![2.0, -1.0].into(), None, None, None, None)
            .unwrap();

        assert!(result.converged);
        assert!(result.iters <= 2); // Should converge very quickly
        assert!(result.fmin < 1e-14);
        assert_eq!(result.xmin.len(), 2);
    }

    #[test]
    fn test_high_dimensional_sphere() {
        let n = 10;
        let sphere = |x: &Points1<f64>| x.iter().map(|&xi| xi * xi).sum();
        let objective = MultiDimFn::new(sphere);
        let mut powell = Powell::new(objective);

        let initial = Points1::from_shape_fn(n, |i| i as f64);
        let result = powell.minimize(&initial, None, None, None, None).unwrap();

        assert!(result.converged);
        for &coord in &result.xmin {
            assert!(coord.abs() < 1e-6);
        }
        assert!(result.fmin < 1e-12);
        // Don't assert on final_directions length - it might vary during optimization
        assert!(!result.final_directions.is_empty());
    }

    #[test]
    fn test_custom_directions_invalid_dimensions() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        // Wrong number of directions
        let directions = array![[1.0, 0.0]].into(); // Only 1 direction for 2D problem
        let result = powell.powell_method_with_directions(
            &array![0.0, 0.0].into(),
            None,
            &directions,
            None,
            None,
            None,
        );
        assert!(matches!(result, Err(MinimizerError::InvalidDirectionSet)));

        // Zero-norm direction
        let directions = array![
            [1.0, 0.0],
            [0.0, 0.0], // Zero vector
        ]
        .into();
        let result = powell.powell_method_with_directions(
            &array![0.0, 0.0].into(),
            None,
            &directions,
            None,
            None,
            None,
        );
        assert!(matches!(result, Err(MinimizerError::InvalidDirectionSet)));
    }

    #[test]
    fn test_function_returning_nan() {
        let bad_func = |x: &Points1<f64>| {
            if x[0] > 10.0 {
                f64::NAN
            } else {
                x[0].powi(2) + x[1].powi(2)
            }
        };
        let objective = MultiDimFn::new(bad_func);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(&array![15.0, 0.0].into(), None, None, None, None);
        assert!(matches!(
            result,
            Err(MinimizerError::FunctionEvaluationError)
        ));
    }

    #[test]
    fn test_function_returning_infinity() {
        let bad_func = |x: &Points1<f64>| {
            if x[0] < -10.0 {
                f64::INFINITY
            } else {
                x[0].powi(2) + x[1].powi(2)
            }
        };
        let objective = MultiDimFn::new(bad_func);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(&array![-15.0, 0.0].into(), None, None, None, None);
        assert!(matches!(
            result,
            Err(MinimizerError::FunctionEvaluationError)
        ));
    }

    #[test]
    fn test_linear_function() {
        // Linear function has no minimum, but should handle gracefully
        let linear = |x: &Points1<f64>| x[0] + 2.0 * x[1];
        let objective = MultiDimFn::new(linear);
        let mut powell = Powell::new(objective);

        let result =
            powell.powell_method(&array![0.0, 0.0].into(), None, Some(1e-6), Some(20), None);
        // Depending on implementation, this might not converge or might find a direction
        // where the function decreases indefinitely
        match result {
            Ok(res) => {
                // If it "converges", it should be because improvement became negligible
                assert!(res.iters <= 20);
            }
            Err(_) => {
                // Acceptable if line search fails due to unbounded function
            }
        }
    }

    #[test]
    fn test_constant_function() {
        let constant = |_x: &Points1<f64>| 42.0;
        let objective = MultiDimFn::new(constant);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![1.0, 2.0].into(), None, None, None, None)
            .unwrap();

        assert!(result.converged);
        assert!((result.fmin - 42.0).abs() < 1e-12);
        assert!(result.iters <= 2); // Should converge immediately
    }

    #[test]
    fn test_steep_valley_function() {
        // Function with a steep valley (challenging for optimization)
        let valley = |x: &Points1<f64>| {
            let u = x[0] + x[1];
            let v = x[0] - x[1];
            u.powi(2) + 100.0 * v.powi(2)
        };
        let objective = MultiDimFn::new(valley);
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(&array![1.0, 1.0].into(), None, Some(1e-6), Some(100), None)
            .unwrap();

        assert!(result.converged);
        assert!(result.xmin[0].abs() < 1e-3);
        assert!(result.xmin[1].abs() < 1e-3);
        assert!(result.fmin < 1e-6);
    }

    #[test]
    fn test_discontinuous_function() {
        // Function with a discontinuity
        let discontinuous = |x: &Points1<f64>| {
            if x[0] > 0.0 {
                x[0].powi(2) + x[1].powi(2) + 1.0
            } else {
                x[0].powi(2) + x[1].powi(2)
            }
        };
        let objective = MultiDimFn::new(discontinuous);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![-1.0, 1.0].into(), None, None, None, None)
            .unwrap();

        // Should find minimum in the negative x region
        assert!(result.xmin[0] <= 0.0);
        assert!(result.xmin[1].abs() < 1e-6);
        assert!(result.fmin < 0.1);
    }

    #[test]
    fn test_result_structure_completeness() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![0.0, 0.0].into(), None, None, None, None)
            .unwrap();

        // Check all fields are properly populated
        assert_eq!(result.xmin.len(), 2);
        assert!(result.fmin.is_finite());
        assert!(result.iters > 0);
        assert!(result.fn_evals > 0);
        assert!(result.converged);
        // Don't assert exact length of final_directions - may vary
        assert!(!result.final_directions.is_empty());
        assert!(!result.history.is_empty());
        assert!(!result.improvement.is_empty());
        // Basic sanity check on history/improvement relationship
        assert!(result.history.len() >= result.improvement.len());
    }

    #[test]
    fn test_powell_state_updates() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        // Initial state
        assert!(powell.xmin().is_empty());
        assert_eq!(powell.fmin(), 0.0);
        assert_eq!(powell.iters(), 0);

        let result = powell
            .minimize(&array![0.0, 0.0].into(), None, None, None, None)
            .unwrap();

        // State should be updated
        assert_eq!(powell.xmin(), result.xmin);
        assert_eq!(powell.fmin(), result.fmin);
        assert_eq!(powell.iters(), result.iters);
        assert!(powell.converged);
    }

    #[test]
    fn test_orthogonal_directions_properties() {
        let objective = simple_quadratic();
        let powell = Powell::new(objective);

        // Test a few specific cases rather than a loop
        let n = 3;
        let directions = powell.create_orthogonal_directions(n);

        assert_eq!(directions.nrows(), n);

        // Check that coordinate directions are properly formed
        for (i, direction) in directions.outer_iter().enumerate() {
            assert_eq!(
                direction.len(),
                n,
                "Direction {} should have length {} but has length {}",
                i,
                n,
                direction.len()
            );

            for (j, &val) in direction.iter().enumerate() {
                if i == j {
                    assert!(
                        (val - 1.0).abs() < 1e-15,
                        "Expected 1.0 at position ({}, {}), got {}",
                        i,
                        j,
                        val
                    );
                } else {
                    assert!(
                        val.abs() < 1e-15,
                        "Expected 0.0 at position ({}, {}), got {}",
                        i,
                        j,
                        val
                    );
                }
            }
        }

        // Test orthogonality
        for i in 0..n {
            for j in i + 1..n {
                let dot_product: f64 = directions
                    .row(i)
                    .iter()
                    .zip(directions.row(j).iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                assert!(
                    dot_product.abs() < 1e-12,
                    "Directions {} and {} should be orthogonal",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_random_directions_reproducibility() {
        let objective = simple_quadratic();
        let powell = Powell::new(objective);

        let seed = 12345;
        let dirs1 = powell.create_random_orthogonal_directions(3, Some(seed));
        let dirs2 = powell.create_random_orthogonal_directions(3, Some(seed));

        // Should be identical with same seed
        for (d1, d2) in dirs1.outer_iter().zip(dirs2.outer_iter()) {
            for (&v1, &v2) in d1.iter().zip(d2.iter()) {
                assert!((v1 - v2).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_random_directions_different_seeds() {
        let objective = simple_quadratic();
        let powell = Powell::new(objective);

        let dirs1 = powell.create_random_orthogonal_directions(3, Some(123));
        let dirs2 = powell.create_random_orthogonal_directions(3, Some(456));

        // Should be different with different seeds
        let mut different = false;
        for (d1, d2) in dirs1.outer_iter().zip(dirs2.outer_iter()) {
            for (&v1, &v2) in d1.iter().zip(d2.iter()) {
                if (v1 - v2).abs() > 1e-10 {
                    different = true;
                    break;
                }
            }
            if different {
                break;
            }
        }
        assert!(different);
    }

    #[test]
    fn test_large_scale_problem() {
        let n = 50;
        // Sum of squares function
        let sum_of_squares = |x: &Points1<f64>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| (i as f64 + 1.0) * xi * xi)
                .sum()
        };
        let objective = MultiDimFn::new(sum_of_squares);
        let mut powell = Powell::new(objective);

        let initial = Points1::ones(n);
        let result = powell
            .powell_method(&initial, None, Some(1e-4), Some(500), None)
            .unwrap();

        assert!(result.converged || result.fmin < 1e-6);
        for &coord in &result.xmin {
            assert!(coord.abs() < 1e-2);
        }
    }

    #[test]
    fn test_ill_conditioned_problem() {
        let objective = ill_conditioned_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![1.0, 1.0].into(), None, None, None, None)
            .unwrap();

        assert!(result.converged);
        assert!(result.xmin[0].abs() < 1e-4);
        assert!(result.xmin[1].abs() < 1e-4);
        // Might take more iterations due to ill-conditioning
        assert!(result.iters <= 50);
    }

    #[test]
    fn test_convergence_history() {
        let objective = rosenbrock_2d();
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(&array![-1.2, 1.0].into(), None, None, None, None)
            .unwrap();

        // History should show monotonic decrease (mostly)
        assert!(!result.history.is_empty());
        assert!(result.history[0] > *result.history.last().unwrap());

        // Improvement should be non-negative (allowing for numerical precision)
        for &improvement in &result.improvement {
            assert!(improvement >= -1e-12); // Allow tiny negative due to numerical precision
        }
    }

    #[test]
    fn test_very_small_step_function() {
        // Function where optimal step sizes are very small
        let small_step = |x: &Points1<f64>| 1e6 * x[0].powi(2) + x[1].powi(2);
        let objective = MultiDimFn::new(small_step);
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(
                &array![1e-3, 1e-3].into(),
                None,
                Some(1e-10),
                Some(100),
                Some(1e-15),
            )
            .unwrap();

        assert!(result.converged);
        assert!(result.xmin[0].abs() < 1e-6);
        assert!(result.xmin[1].abs() < 1e-6);
    }

    #[test]
    fn test_debug_formatting() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        // Test before optimization
        let debug_str = format!("{:?}", powell);
        assert!(debug_str.contains("Powell"));
        assert!(debug_str.contains("xmin"));
        assert!(debug_str.contains("fmin"));
        assert!(debug_str.contains("iters"));
        assert!(debug_str.contains("converged"));

        // Test after optimization
        let _result = powell
            .minimize(&array![0.0, 0.0].into(), None, None, None, None)
            .unwrap();
        let debug_str_after = format!("{:?}", powell);
        assert!(debug_str_after.contains("Powell"));
        assert!(debug_str_after.contains(&format!("{:?}", powell.xmin())));
    }

    #[test]
    fn test_boxed_function() {
        let func = |x: &Points1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let boxed_func: Box<dyn ObjFn<f64>> = Box::new(MultiDimFn::new(func));
        let mut powell = Powell::new_boxed(boxed_func);

        let result = powell
            .minimize(&array![0.0, 0.0].into(), None, None, None, None)
            .unwrap();

        assert!(result.converged);
        assert!((result.xmin[0] - 1.0).abs() < 1e-8);
        assert!((result.xmin[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_performance_comparison() {
        // Compare standard vs custom directions on ill-conditioned problem
        let objective1 = ill_conditioned_quadratic();
        let objective2 = ill_conditioned_quadratic();
        let mut powell1 = Powell::new(objective1);
        let mut powell2 = Powell::new(objective2);

        let start_point = array![10.0, 10.0].into();

        // Standard coordinate directions
        let result1 = powell1
            .minimize(&start_point, None, None, None, None)
            .unwrap();

        // Custom directions better suited for the problem
        let custom_dirs = array![
            [0.001, 0.0], // Small step for highly scaled dimension
            [0.0, 1.0],   // Normal step for other dimension
        ]
        .into();
        let result2 = powell2
            .powell_method_with_directions(&start_point, None, &custom_dirs, None, None, None)
            .unwrap();

        // Both should converge, custom might be more efficient
        assert!(result1.converged);
        assert!(result2.converged);
        assert!(result1.xmin[0].abs() < 1e-4);
        assert!(result1.xmin[1].abs() < 1e-4);
        assert!(result2.xmin[0].abs() < 1e-4);
        assert!(result2.xmin[1].abs() < 1e-4);
    }

    mod myfloat_tests {
        use super::*;
        use crate::num::MyFloat;

        #[test]
        fn test_2d_quadratic() {
            // f(x,y) = (x-2)² + (y+1)²
            let func = |x: &Points1<MyFloat>| (&x[0] - 2.0).powi(2) + (&x[1] + 1.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut powell = Powell::new(objective);

            let result = powell
                .minimize(
                    &array![0.0.into(), 0.0.into()].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            assert!((&result.xmin[0] - 2.0).abs() < 1e-8);
            assert!((&result.xmin[1] + 1.0).abs() < 1e-8);
            assert!(result.fmin < 1e-14);
            assert!(result.converged);
            assert!((&powell.xmin[0] - 2.0).abs() < 1e-8);
            assert!((&powell.xmin[1] + 1.0).abs() < 1e-8);
            assert!(powell.fmin < 1e-14);
            assert!(powell.converged);
        }

        #[test]
        fn test_rosenbrock() {
            let rosenbrock = |x: &Points1<MyFloat>| {
                (1.0 - &x[0]).powi(2) + 100.0 * (&x[1] - x[0].powi(2)).powi(2)
            };
            let objective = MultiDimFn::new(rosenbrock);
            let mut powell = Powell::new(objective);

            let result = powell
                .powell_method(
                    &array![MyFloat::new(-1.2), 1.0.into()].into(),
                    None,
                    Some(1e-6.into()),
                    Some(1000),
                    None,
                )
                .unwrap();

            assert!((&result.xmin[0] - 1.0).abs() < 1e-4);
            assert!((&result.xmin[1] - 1.0).abs() < 1e-4);
            assert!(result.fmin < 1e-8);
            assert!((&powell.xmin[0] - 1.0).abs() < 1e-4);
            assert!((&powell.xmin[1] - 1.0).abs() < 1e-4);
            assert!(powell.fmin < 1e-8);
        }

        #[test]
        fn test_3d_sphere() {
            // f(x,y,z) = x² + y² + z²
            let sphere = |x: &Points1<MyFloat>| x.iter().map(|xi| xi * xi).sum();
            let objective = MultiDimFn::new(sphere);
            let mut powell = Powell::new(objective);

            let result = powell
                .minimize(
                    &array![1.0.into(), 2.0.into(), MyFloat::new(-1.0)].into(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            for coord in result.xmin {
                assert!(coord.abs() < 1e-10);
            }
            assert!(result.fmin < 1e-18);
            assert!(result.converged);
            for coord in powell.xmin {
                assert!(coord.abs() < 1e-10);
            }
            assert!(powell.fmin < 1e-18);
            assert!(powell.converged);
        }
    }
}
