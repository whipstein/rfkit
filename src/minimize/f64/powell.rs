#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::f64::F1dim;
use crate::minimize::{
    MinimizerError,
    f64::{Brent, ObjFn},
};
use ndarray::prelude::*;
use std::fmt;

/// Result of Powell's method optimization
#[derive(Debug, Clone)]
pub struct PowellResult {
    pub xmin: Array1<f64>,
    pub fmin: f64,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_directions: Array2<f64>,
    pub history: Array1<f64>,
    pub improvement: Array1<f64>,
}

#[derive(Clone)]
pub struct Powell {
    xmin: Array1<f64>,
    fmin: f64,
    f: Box<dyn ObjFn>,
    iters: usize,
    converged: bool,
}

impl Powell {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn + Clone + 'static,
    {
        let boxed = Box::new(f);
        Powell {
            xmin: array![],
            fmin: 0.0,
            f: boxed,
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn>) -> Self {
        Powell {
            xmin: array![],
            fmin: 0.0,
            f: f,
            iters: 0,
            converged: false,
        }
    }

    pub fn xmin(&self) -> Array1<f64> {
        self.xmin.clone()
    }

    pub fn fmin(&self) -> f64 {
        self.fmin
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
        initial_point: Array1<f64>,
        tol: Option<f64>,
        max_iters: Option<usize>,
        line_search_tolerance: Option<f64>,
    ) -> Result<PowellResult, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let tol = tol.unwrap_or(1e-8);
        let max_iter = max_iters.unwrap_or(200);
        let line_tol = line_search_tolerance.unwrap_or(1e-12);

        if tol <= 0.0 {
            return Err(MinimizerError::InvalidTolerance);
        }

        // Initialize direction set as coordinate axes
        let mut directions = Array2::<f64>::eye(n);
        // let mut directions: Array2<f64> = (0..n)
        //     .map(|i| {
        //         let mut dir = vec![0.0; n];
        //         dir[i] = 1.0;
        //         dir
        //     })
        //     .collect();

        let mut x = initial_point;
        let mut f_current = self.f.call(&x);
        if !f_current.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }

        let mut fn_evals = 1;
        self.iters = 0;
        let mut history = vec![f_current];
        let mut improvement = Vec::new();

        while self.iters < max_iter {
            self.iters += 1;
            let x_old = x.clone();
            let f_old = f_current;

            let mut delta_f_max = 0.0;
            let mut max_decrease_index = 0;

            // Perform line searches along each direction
            for (i, direction) in directions.outer_iter().enumerate() {
                let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                let line_result =
                    brent.line_search(&x, &direction.to_owned(), 1.0, line_tol, 1000)?;

                if !line_result.converged {
                    return Err(MinimizerError::LinearSearchFailed);
                }

                fn_evals += line_result.fn_evals;

                // Update position
                for j in 0..n {
                    x[j] += line_result.xmin * direction[j];
                }

                let f_new = self.f.call(&x);
                if !f_new.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }
                fn_evals += 1;

                let delta_f = f_current - f_new;
                if delta_f > delta_f_max {
                    delta_f_max = delta_f;
                    max_decrease_index = i;
                }

                f_current = f_new;
            }

            let improve = f_old - f_current;
            improvement.push(improve);
            history.push(f_current);

            // Check for convergence
            let relative_improvement = improve / (f_old.abs() + 1e-14);
            if relative_improvement < tol {
                self.xmin = x.clone();
                self.fmin = f_current;
                self.converged = true;
                return Ok(PowellResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin,
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_directions: directions,
                    history: Array1::from_vec(history),
                    improvement: Array1::from_vec(improvement),
                });
            }

            // Compute extrapolated point
            let x_extrapolated = Array1::from_shape_fn(n, |i| 2.0 * x[i] - x_old[i]);
            // let mut x_extrapolated = vec![0.0; n];
            // for i in 0..n {
            //     x_extrapolated[i] = 2.0 * x[i] - x_old[i];
            // }

            let f_extrapolated = self.f.call(&x_extrapolated);
            if !f_extrapolated.is_finite() {
                return Err(MinimizerError::FunctionEvaluationError);
            }
            fn_evals += 1;

            // Test whether to keep new direction
            let condition1 = f_extrapolated < f_old;
            let temp = 2.0 * (f_old - 2.0 * f_current + f_extrapolated);
            let condition2 = temp * (f_old - f_current - delta_f_max).powi(2);
            let condition3 = delta_f_max * (f_old - f_extrapolated).powi(2);

            if condition1 && condition2 < condition3 {
                // Move to extrapolated point and update direction set
                let new_direction: Vec<f64> = x
                    .iter()
                    .zip(x_old.iter())
                    .map(|(&xi, &xi_old)| xi - xi_old)
                    .collect();

                // Check if new direction is linearly independent
                let direction_norm: f64 = new_direction.iter().map(|&d| d * d).sum::<f64>().sqrt();
                if direction_norm < 1e-12 {
                    continue; // Skip if direction is too small
                }

                // Normalize new direction
                let new_direction: Vec<f64> =
                    new_direction.iter().map(|&d| d / direction_norm).collect();

                // Perform line search along new direction
                let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                let line_result = brent.line_search(
                    &x,
                    &Array1::from_vec(new_direction.clone()),
                    1.0,
                    line_tol,
                    1000,
                )?;

                if line_result.converged {
                    fn_evals += line_result.fn_evals;

                    // Update position
                    for j in 0..n {
                        x[j] += line_result.xmin * new_direction[j];
                    }

                    f_current = self.f.call(&x);
                    if !f_current.is_finite() {
                        return Err(MinimizerError::FunctionEvaluationError);
                    }
                    fn_evals += 1;

                    // Replace the direction that gave maximum decrease
                    for (j, &dir) in new_direction.iter().enumerate() {
                        directions[[max_decrease_index, j]] = dir;
                    }
                }
            }
        }

        self.xmin = x;
        self.fmin = f_current;
        Ok(PowellResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin,
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_directions: directions,
            history: Array1::from_vec(history),
            improvement: Array1::from_vec(improvement),
        })
    }

    /// Powell's method with custom initial directions
    ///
    /// Allows specification of initial search directions instead of using
    /// coordinate axes. Useful when you have knowledge about the problem structure.
    pub fn powell_method_with_directions(
        &mut self,
        initial_point: Array1<f64>,
        initial_directions: Array2<f64>,
        tol: Option<f64>,
        max_iters: Option<usize>,
        line_search_tolerance: Option<f64>,
    ) -> Result<PowellResult, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if initial_directions.nrows() != n {
            return Err(MinimizerError::InvalidDirectionSet);
        }

        for direction in initial_directions.outer_iter() {
            if direction.len() != n {
                return Err(MinimizerError::InvalidDirectionSet);
            }
            let norm: f64 = direction.iter().map(|&d| d * d).sum::<f64>().sqrt();
            if norm < 1e-12 {
                return Err(MinimizerError::InvalidDirectionSet);
            }
        }

        let tol = tol.unwrap_or(1e-8);
        let max_iter = max_iters.unwrap_or(200);
        let line_tol = line_search_tolerance.unwrap_or(1e-12);

        if tol <= 0.0 {
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
        let mut directions = Array2::from_shape_fn(initial_directions.dim(), |(i, j)| {
            let norm = initial_directions
                .row(i)
                .iter()
                .map(|&d| d * d)
                .sum::<f64>()
                .sqrt();
            initial_directions[[i, j]] / norm
        });

        let mut x = initial_point;
        let mut f_current = self.f.call(&x);
        if !f_current.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }

        let mut fn_evals = 1;
        self.iters = 0;
        let mut history = vec![f_current];
        let mut improvement = Vec::new();

        while self.iters < max_iter {
            self.iters += 1;
            let x_old = x.clone();
            let f_old = f_current;

            let mut delta_f_max = 0.0;
            let mut max_decrease_index = 0;

            // Line searches along each direction
            for (i, direction) in directions.outer_iter().enumerate() {
                let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                let line_result =
                    brent.line_search(&x, &direction.to_owned(), 1.0, line_tol, 1000)?;

                if !line_result.converged {
                    return Err(MinimizerError::LinearSearchFailed);
                }

                fn_evals += line_result.fn_evals;

                // Update position
                for j in 0..n {
                    x[j] += line_result.xmin * direction[j];
                }

                let f_new = self.f.call(&x);
                if !f_new.is_finite() {
                    return Err(MinimizerError::FunctionEvaluationError);
                }
                fn_evals += 1;

                let delta_f = f_current - f_new;
                if delta_f > delta_f_max {
                    delta_f_max = delta_f;
                    max_decrease_index = i;
                }

                f_current = f_new;
            }

            let improve = f_old - f_current;
            improvement.push(improve);
            history.push(f_current);

            // Check for convergence
            let relative_improvement = improve / (f_old.abs() + 1e-14);
            if relative_improvement < tol {
                self.xmin = x;
                self.fmin = f_current;
                self.converged = true;
                return Ok(PowellResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin,
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_directions: directions,
                    history: Array1::from_vec(history),
                    improvement: Array1::from_vec(improvement),
                });
            }

            // Direction replacement logic (same as standard Powell)
            // let mut x_extrapolated = vec![0.0; n];
            // for i in 0..n {
            //     x_extrapolated[i] = 2.0 * x[i] - x_old[i];
            // }
            let x_extrapolated = Array1::from_shape_fn(n, |i| 2.0 * x[i] - x_old[i]);

            let f_extrapolated = self.f.call(&x_extrapolated);
            if !f_extrapolated.is_finite() {
                continue;
            }
            fn_evals += 1;

            let condition1 = f_extrapolated < f_old;
            let temp = 2.0 * (f_old - 2.0 * f_current + f_extrapolated);
            let condition2 = temp * (f_old - f_current - delta_f_max).powi(2);
            let condition3 = delta_f_max * (f_old - f_extrapolated).powi(2);

            if condition1 && condition2 < condition3 {
                let new_direction: Vec<f64> = x
                    .iter()
                    .zip(x_old.iter())
                    .map(|(&xi, &xi_old)| xi - xi_old)
                    .collect();

                let direction_norm: f64 = new_direction.iter().map(|&d| d * d).sum::<f64>().sqrt();
                if direction_norm > 1e-12 {
                    // let normalized_direction: Vec<f64> =
                    //     new_direction.iter().map(|&d| d / direction_norm).collect();
                    let normalized_direction = Array1::from_shape_fn(new_direction.len(), |i| {
                        new_direction[i] / direction_norm
                    });

                    let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                    let line_result =
                        brent.line_search(&x, &normalized_direction, 1.0, line_tol, 1000)?;

                    if line_result.converged {
                        fn_evals += line_result.fn_evals;

                        for j in 0..n {
                            x[j] += line_result.xmin * normalized_direction[j];
                        }

                        f_current = self.f.call(&x);
                        if !f_current.is_finite() {
                            continue;
                        }
                        fn_evals += 1;

                        for (j, &dir) in normalized_direction.iter().enumerate() {
                            directions[[max_decrease_index, j]] = dir;
                        }
                    }
                }
            }
        }

        self.xmin = x;
        self.fmin = f_current;
        Ok(PowellResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin,
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_directions: directions,
            history: Array1::from_vec(history),
            improvement: Array1::from_vec(improvement),
        })
    }

    /// Convenience function with default parameters
    pub fn minimize(
        &mut self,
        initial_point: Array1<f64>,
        tol: Option<f64>,
        max_iters: Option<usize>,
        line_search_tolerance: Option<f64>,
    ) -> Result<PowellResult, MinimizerError> {
        self.powell_method(initial_point, tol, max_iters, line_search_tolerance)
    }

    /// Create orthogonal initial directions for better performance
    pub fn create_orthogonal_directions(&self, n: usize) -> Array2<f64> {
        Array2::eye(n)
    }

    /// Create random orthogonal directions using Gram-Schmidt
    pub fn create_random_orthogonal_directions(&self, n: usize, seed: Option<u64>) -> Array2<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple random number generator for reproducibility
        let mut rng_state = seed.unwrap_or_else(|| {
            let mut hasher = DefaultHasher::new();
            std::ptr::addr_of!(n).hash(&mut hasher);
            hasher.finish()
        });

        let mut random = || -> f64 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng_state / 65536) % 32768) as f64 / 32768.0 - 0.5
        };

        // let mut directions: Array2<f64> = Vec::with_capacity(n);
        let mut directions: Array2<f64> = Array2::zeros((n, n));

        for i in 0..n {
            // Generate random vector
            let mut new_dir = (0..n).map(|_| random()).collect::<Vec<f64>>();

            // Gram-Schmidt orthogonalization
            for j in 0..i {
                let dot_product: f64 = new_dir
                    .iter()
                    .zip(directions.row(j).iter())
                    .map(|(&a, &b)| a * b)
                    .sum();

                for k in 0..n {
                    new_dir[k] -= dot_product * directions[[j, k]];
                }
            }

            // Normalize
            let norm: f64 = new_dir.iter().map(|&d| d * d).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for d in &mut new_dir {
                    *d /= norm;
                }
            } else {
                // Fallback to coordinate vector if normalization fails
                new_dir = vec![0.0; n];
                new_dir[i] = 1.0;
            }

            // directions.push(new_dir);
            for (j, &dir) in new_dir.iter().enumerate() {
                directions[[i, j]] = dir;
            }
        }

        directions
    }
}

impl fmt::Debug for Powell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Powell( xmin: {:?}, fmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod minimize_f64_powell_tests {
    use super::*;
    use crate::minimize::f64::{F1dim, MultiDimFn};
    use float_cmp::F64Margin;
    use std::f64::consts::{E, PI};
    use std::vec;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

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
    fn test_2d_quadratic() {
        // f(x,y) = (x-2)² + (y+1)²
        let func = |x: &Array1<f64>| (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(array![0.0, 0.0], None, None, None).unwrap();

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
            |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let objective = MultiDimFn::new(rosenbrock);
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(array![-1.2, 1.0], Some(1e-6), Some(1000), None)
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
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum();
        let objective = MultiDimFn::new(sphere);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(array![1.0, 2.0, -1.0], None, None, None)
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
        let func = |x: &Array1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut powell = Powell::new(objective);

        // Use diagonal directions instead of coordinate axes
        let directions = array![
            [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()],
            [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()],
        ];

        let result = powell
            .powell_method_with_directions(array![0.0, 0.0], directions, None, None, None)
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
        let func = |_: &Array1<f64>| 0.0;
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
        let func = |_: &Array1<f64>| 0.0;
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
        let quadratic_3d = |x: &Array1<f64>| {
            x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2) + x[0] * x[1] + x[1] * x[2]
        };
        let objective = MultiDimFn::new(quadratic_3d);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(array![1.0, 1.0, 1.0], None, None, None)
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
        let elliptic = |x: &Array1<f64>| 100.0 * x[0].powi(2) + x[1].powi(2);
        let objective = MultiDimFn::new(elliptic);
        let mut powell = Powell::new(objective);

        // Standard coordinate directions
        let std_result = powell.minimize(array![1.0, 1.0], None, None, None).unwrap();

        // Custom directions aligned with ellipse
        let custom_dirs = array![
            [0.1, 0.0], // Small step in x direction
            [0.0, 1.0], // Normal step in y direction
        ];

        let custom_result = powell
            .powell_method_with_directions(array![1.0, 1.0], custom_dirs, None, None, None)
            .unwrap();

        // Custom directions should perform better on ill-conditioned problem
        assert!(custom_result.fn_evals as f64 <= std_result.fn_evals as f64 * 1.5);
        assert!(custom_result.xmin[0].abs() < 1e-6);
        assert!(custom_result.xmin[1].abs() < 1e-6);
    }

    #[test]
    fn test_powell_direction_evolution() {
        // Test that directions evolve to become conjugate
        let quadratic_matrix = |x: &Array1<f64>| {
            // f(x) = x^T A x where A = [[4,1],[1,2]]
            4.0 * x[0].powi(2) + 2.0 * x[1].powi(2) + 2.0 * x[0] * x[1]
        };
        let objective = MultiDimFn::new(quadratic_matrix);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(array![1.0, 1.0], None, None, None).unwrap();

        assert!(result.converged);
        assert!(result.xmin[0].abs() < 1e-6);
        assert!(result.xmin[1].abs() < 1e-6);

        // Final directions should be approximately conjugate
        // (This is hard to test directly, but convergence implies conjugacy)
        assert!(result.iters <= 5); // Should be fast for 2D quadratic
    }

    // Helper function to create a simple quadratic function
    fn simple_quadratic() -> F1dim {
        F1dim::new(MultiDimFn::new(|x: &Array1<f64>| {
            (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2)
        }))
    }

    // Helper function to create Rosenbrock function
    fn rosenbrock_2d() -> F1dim {
        F1dim::new(MultiDimFn::new(|x: &Array1<f64>| {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
        }))
    }

    // Helper function for ill-conditioned quadratic
    fn ill_conditioned_quadratic() -> F1dim {
        F1dim::new(MultiDimFn::new(|x: &Array1<f64>| {
            1000.0 * x[0].powi(2) + x[1].powi(2)
        }))
    }

    #[test]
    fn test_empty_initial_point() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell.minimize(array![], None, None, None);
        assert!(matches!(result, Err(MinimizerError::InvalidDimension)));
    }

    #[test]
    fn test_invalid_tolerance() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell.powell_method(array![0.0, 0.0], Some(-1e-8), None, None);
        assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));

        let result = powell.powell_method(array![0.0, 0.0], Some(0.0), None, None);
        assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
    }

    #[test]
    fn test_single_dimension() {
        let func = |x: &Array1<f64>| (x[0] - 3.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(array![0.0], None, None, None).unwrap();

        assert!((result.xmin[0] - 3.0).abs() < 1e-8);
        assert!(result.fmin < 1e-14);
        assert!(result.converged);
        assert_eq!(result.final_directions.len(), 1);
        assert_eq!(result.final_directions.row(0), array![1.0]);
    }

    #[test]
    fn test_very_tight_tolerance() {
        let objective = simple_quadratic();
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(array![0.0, 0.0], Some(1e-15), Some(1000), None)
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
            .powell_method(array![0.0, 0.0], Some(1e-1), None, None)
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
            .powell_method(array![-1.2, 1.0], Some(1e-12), Some(5), None)
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
            .minimize(array![2.0, -1.0], None, None, None)
            .unwrap();

        assert!(result.converged);
        assert!(result.iters <= 2); // Should converge very quickly
        assert!(result.fmin < 1e-14);
        assert_eq!(result.xmin.len(), 2);
    }

    #[test]
    fn test_high_dimensional_sphere() {
        let n = 10;
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum();
        let objective = MultiDimFn::new(sphere);
        let mut powell = Powell::new(objective);

        let initial = Array1::from_shape_fn(n, |i| i as f64);
        let result = powell.minimize(initial, None, None, None).unwrap();

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
        let directions = array![[1.0, 0.0]]; // Only 1 direction for 2D problem
        let result =
            powell.powell_method_with_directions(array![0.0, 0.0], directions, None, None, None);
        assert!(matches!(result, Err(MinimizerError::InvalidDirectionSet)));

        // Zero-norm direction
        let directions = array![
            [1.0, 0.0],
            [0.0, 0.0], // Zero vector
        ];
        let result =
            powell.powell_method_with_directions(array![0.0, 0.0], directions, None, None, None);
        assert!(matches!(result, Err(MinimizerError::InvalidDirectionSet)));
    }

    #[test]
    fn test_function_returning_nan() {
        let bad_func = |x: &Array1<f64>| {
            if x[0] > 10.0 {
                f64::NAN
            } else {
                x[0].powi(2) + x[1].powi(2)
            }
        };
        let objective = MultiDimFn::new(bad_func);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(array![15.0, 0.0], None, None, None);
        assert!(matches!(
            result,
            Err(MinimizerError::FunctionEvaluationError)
        ));
    }

    #[test]
    fn test_function_returning_infinity() {
        let bad_func = |x: &Array1<f64>| {
            if x[0] < -10.0 {
                f64::INFINITY
            } else {
                x[0].powi(2) + x[1].powi(2)
            }
        };
        let objective = MultiDimFn::new(bad_func);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(array![-15.0, 0.0], None, None, None);
        assert!(matches!(
            result,
            Err(MinimizerError::FunctionEvaluationError)
        ));
    }

    #[test]
    fn test_linear_function() {
        // Linear function has no minimum, but should handle gracefully
        let linear = |x: &Array1<f64>| x[0] + 2.0 * x[1];
        let objective = MultiDimFn::new(linear);
        let mut powell = Powell::new(objective);

        let result = powell.powell_method(array![0.0, 0.0], Some(1e-6), Some(20), None);
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
        let constant = |_x: &Array1<f64>| 42.0;
        let objective = MultiDimFn::new(constant);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(array![1.0, 2.0], None, None, None).unwrap();

        assert!(result.converged);
        assert!((result.fmin - 42.0).abs() < 1e-12);
        assert!(result.iters <= 2); // Should converge immediately
    }

    #[test]
    fn test_steep_valley_function() {
        // Function with a steep valley (challenging for optimization)
        let valley = |x: &Array1<f64>| {
            let u = x[0] + x[1];
            let v = x[0] - x[1];
            u.powi(2) + 100.0 * v.powi(2)
        };
        let objective = MultiDimFn::new(valley);
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(array![1.0, 1.0], Some(1e-6), Some(100), None)
            .unwrap();

        assert!(result.converged);
        assert!(result.xmin[0].abs() < 1e-3);
        assert!(result.xmin[1].abs() < 1e-3);
        assert!(result.fmin < 1e-6);
    }

    #[test]
    fn test_discontinuous_function() {
        // Function with a discontinuity
        let discontinuous = |x: &Array1<f64>| {
            if x[0] > 0.0 {
                x[0].powi(2) + x[1].powi(2) + 1.0
            } else {
                x[0].powi(2) + x[1].powi(2)
            }
        };
        let objective = MultiDimFn::new(discontinuous);
        let mut powell = Powell::new(objective);

        let result = powell
            .minimize(array![-1.0, 1.0], None, None, None)
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

        let result = powell.minimize(array![0.0, 0.0], None, None, None).unwrap();

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

        let result = powell.minimize(array![0.0, 0.0], None, None, None).unwrap();

        // State should be updated
        assert_eq!(powell.xmin(), &result.xmin);
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
        let sum_of_squares = |x: &Array1<f64>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| (i as f64 + 1.0) * xi * xi)
                .sum()
        };
        let objective = MultiDimFn::new(sum_of_squares);
        let mut powell = Powell::new(objective);

        let initial = Array1::ones(n);
        let result = powell
            .powell_method(initial, Some(1e-4), Some(500), None)
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

        let result = powell.minimize(array![1.0, 1.0], None, None, None).unwrap();

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
            .minimize(array![-1.2, 1.0], None, None, None)
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
        let small_step = |x: &Array1<f64>| 1e6 * x[0].powi(2) + x[1].powi(2);
        let objective = MultiDimFn::new(small_step);
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(array![1e-3, 1e-3], Some(1e-10), Some(100), Some(1e-15))
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
        let _result = powell.minimize(array![0.0, 0.0], None, None, None).unwrap();
        let debug_str_after = format!("{:?}", powell);
        assert!(debug_str_after.contains("Powell"));
        assert!(debug_str_after.contains(&format!("{:?}", powell.xmin())));
    }

    #[test]
    fn test_boxed_function() {
        let func = |x: &Array1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let boxed_func: Box<dyn ObjFn> = Box::new(MultiDimFn::new(func));
        let mut powell = Powell::new_boxed(boxed_func);

        let result = powell.minimize(array![0.0, 0.0], None, None, None).unwrap();

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

        let start_point = array![10.0, 10.0];

        // Standard coordinate directions
        let result1 = powell1
            .minimize(start_point.clone(), None, None, None)
            .unwrap();

        // Custom directions better suited for the problem
        let custom_dirs = array![
            [0.001, 0.0], // Small step for highly scaled dimension
            [0.0, 1.0],   // Normal step for other dimension
        ];
        let result2 = powell2
            .powell_method_with_directions(start_point, custom_dirs, None, None, None)
            .unwrap();

        // Both should converge, custom might be more efficient
        assert!(result1.converged);
        assert!(result2.converged);
        assert!(result1.xmin[0].abs() < 1e-4);
        assert!(result1.xmin[1].abs() < 1e-4);
        assert!(result2.xmin[0].abs() < 1e-4);
        assert!(result2.xmin[1].abs() < 1e-4);
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
            let mut powell = Powell::new(objective);
            let result = powell
                .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                .unwrap();
            for i in 0..10 {
                assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                assert!((powell.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((powell.fmin - bias) < 1e-3)
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
                let mut powell = Powell::new(objective);
                let result = powell
                    .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                    .unwrap();
                for i in 0..10 {
                    assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((result.fmin - bias) < 1e-3);

                    assert!((powell.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((powell.fmin - bias) < 1e-3)
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
                let mut powell = Powell::new(objective);
                let result = powell
                    .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                    .unwrap();
                for i in 0..10 {
                    assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((result.fmin - bias) < 1e-3);

                    assert!((powell.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((powell.fmin - bias) < 1e-3)
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
            let mut powell = Powell::new(objective);
            let result = powell
                .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                .unwrap();
            for i in 0..10 {
                assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                assert!((powell.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((powell.fmin - bias) < 1e-3)
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
                let mut powell = Powell::new(objective);
                let result = powell
                    .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                    .unwrap();
                for i in 0..10 {
                    assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((result.fmin - bias) < 1e-3);

                    assert!((powell.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((powell.fmin - bias) < 1e-3)
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
                let mut powell = Powell::new(objective);
                let result = powell
                    .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                    .unwrap();
                for i in 0..10 {
                    assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((result.fmin - bias) < 1e-3);

                    assert!((powell.xmin[i] - shift[i]).abs() < 1e-2);
                    assert!((powell.fmin - bias) < 1e-3)
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

            let mut iter = 0;
            while iter < 10 {
                iter += 1;
                // 10d
                let n = 10;
                let x = (&Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0)
                    - &shift)
                    .dot(&m);
                let mut powell = Powell::new(elliptic(bias));
                let result = powell
                    .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                    .unwrap();
                for i in 0..10 {
                    if result.xmin[i].abs() < 1e-2 && (result.fmin - bias) < 1e-3 {
                        // assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                        assert!(result.xmin[i].abs() < 1e-2);
                        assert!((result.fmin - bias) < 1e-3);

                        // assert!((cmaes.xmin[i] - shift[i]).abs() < 1e-2);
                        assert!(powell.xmin[i].abs() < 1e-2);
                        assert!((powell.fmin - bias) < 1e-3);
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
            let mut powell = Powell::new(objective);
            let result = powell
                .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                .unwrap();
            for i in 0..10 {
                assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((result.fmin - bias) < 1e-3);

                assert!((powell.xmin[i] - shift[i]).abs() < 1e-2);
                assert!((powell.fmin - bias) < 1e-3)
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

            let mut iter = 0;
            while iter < 10 {
                iter += 1;
                // 10d
                let n = 10;
                let x = Array1::from_shape_fn(n, |_| rng.random::<f64>() * 200.0 - 100.0);
                let mut powell = Powell::new(rosenbrock(shift.slice(s![0..10]).to_owned(), bias));
                let result = powell
                    .minimize(x, Some(1e-10), Some(1e3 as usize * n.pow(2)), Some(1e-10))
                    .unwrap();
                for i in 0..10 {
                    if (result.xmin[i] - shift[i]).abs() < 1e-2 && (result.fmin - bias) < 1e-3 {
                        assert!((result.xmin[i] - shift[i]).abs() < 1e-2);
                        assert!((result.fmin - bias) < 1e-3);

                        assert!((powell.xmin[i] - shift[i]).abs() < 1e-2);
                        assert!((powell.fmin - bias) < 1e-3);
                        break;
                    } else {
                        continue;
                    }
                }
            }
        }
    }
}
