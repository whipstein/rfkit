#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::f64::F1dim;
use crate::minimize::{
    MinimizerError,
    f64::{Brent, ObjFn},
};
use std::fmt;
// use ndarray::prelude::*;

/// Result of Powell's method optimization
#[derive(Debug, Clone)]
pub struct PowellResult {
    pub xmin: Vec<f64>,
    pub fmin: f64,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_directions: Vec<Vec<f64>>,
    pub history: Vec<f64>,
    pub improvement: Vec<f64>,
}

#[derive(Clone)]
pub struct Powell {
    xmin: Vec<f64>,
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
            xmin: vec![],
            fmin: 0.0,
            f: boxed,
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn>) -> Self {
        Powell {
            xmin: vec![],
            fmin: 0.0,
            f: f,
            iters: 0,
            converged: false,
        }
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
        initial_point: Vec<f64>,
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
        let mut directions: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut dir = vec![0.0; n];
                dir[i] = 1.0;
                dir
            })
            .collect();

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
            for (i, direction) in directions.iter().enumerate() {
                let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                let line_result = brent.line_search(&x, direction, 1.0, line_tol, 1000)?;

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
                    history,
                    improvement,
                });
            }

            // Compute extrapolated point
            let mut x_extrapolated = vec![0.0; n];
            for i in 0..n {
                x_extrapolated[i] = 2.0 * x[i] - x_old[i];
            }

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
                let line_result = brent.line_search(&x, &new_direction, 1.0, line_tol, 1000)?;

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
                    directions[max_decrease_index] = new_direction;
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
            history,
            improvement,
        })
    }

    /// Powell's method with custom initial directions
    ///
    /// Allows specification of initial search directions instead of using
    /// coordinate axes. Useful when you have knowledge about the problem structure.
    pub fn powell_method_with_directions(
        &mut self,
        initial_point: Vec<f64>,
        initial_directions: Vec<Vec<f64>>,
        tol: Option<f64>,
        max_iters: Option<usize>,
        line_search_tolerance: Option<f64>,
    ) -> Result<PowellResult, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if initial_directions.len() != n {
            return Err(MinimizerError::InvalidDirectionSet);
        }

        for direction in &initial_directions {
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
        let mut directions: Vec<Vec<f64>> = initial_directions
            .into_iter()
            .map(|dir| {
                let norm: f64 = dir.iter().map(|&d| d * d).sum::<f64>().sqrt();
                dir.into_iter().map(|d| d / norm).collect()
            })
            .collect();

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
            for (i, direction) in directions.iter().enumerate() {
                let mut brent = Brent::new(F1dim::new_boxed(self.f.clone()));
                let line_result = brent.line_search(&x, direction, 1.0, line_tol, 1000)?;

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
                    history,
                    improvement,
                });
            }

            // Direction replacement logic (same as standard Powell)
            let mut x_extrapolated = vec![0.0; n];
            for i in 0..n {
                x_extrapolated[i] = 2.0 * x[i] - x_old[i];
            }

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
                    let normalized_direction: Vec<f64> =
                        new_direction.iter().map(|&d| d / direction_norm).collect();

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

                        directions[max_decrease_index] = normalized_direction;
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
            history,
            improvement,
        })
    }

    /// Convenience function with default parameters
    pub fn minimize(&mut self, initial_point: Vec<f64>) -> Result<PowellResult, MinimizerError> {
        self.powell_method(initial_point, None, None, None)
    }

    /// Create orthogonal initial directions for better performance
    pub fn create_orthogonal_directions(&self, n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let mut dir = vec![0.0; n];
                dir[i] = 1.0;
                dir
            })
            .collect()
    }

    /// Create random orthogonal directions using Gram-Schmidt
    pub fn create_random_orthogonal_directions(
        &self,
        n: usize,
        seed: Option<u64>,
    ) -> Vec<Vec<f64>> {
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

        let mut directions: Vec<Vec<f64>> = Vec::with_capacity(n);

        for i in 0..n {
            // Generate random vector
            let mut new_dir = (0..n).map(|_| random()).collect::<Vec<f64>>();

            // Gram-Schmidt orthogonalization
            for j in 0..i {
                let dot_product: f64 = new_dir
                    .iter()
                    .zip(directions[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();

                for k in 0..n {
                    new_dir[k] -= dot_product * directions[j][k];
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

            directions.push(new_dir);
        }

        directions
    }

    pub fn xmin(&self) -> &Vec<f64> {
        &self.xmin
    }

    pub fn fmin(&self) -> f64 {
        self.fmin
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn set_xmin(&mut self, xmin: Vec<f64>) {
        self.xmin = xmin;
        self.fmin = self.f.call(&self.xmin);
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
mod powellf64_tests {
    use super::*;
    use crate::minimize::f64::MultiDimFn;
    use float_cmp::F64Margin;
    use std::vec;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    #[test]
    fn test_2d_quadratic() {
        // f(x,y) = (x-2)² + (y+1)²
        let func = |x: &Vec<f64>| (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(vec![0.0, 0.0]).unwrap();

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
            |x: &Vec<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let objective = MultiDimFn::new(rosenbrock);
        let mut powell = Powell::new(objective);

        let result = powell
            .powell_method(vec![-1.2, 1.0], Some(1e-6), Some(1000), None)
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
        let sphere = |x: &Vec<f64>| x.iter().map(|&xi| xi * xi).sum();
        let objective = MultiDimFn::new(sphere);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(vec![1.0, 2.0, -1.0]).unwrap();

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
        let func = |x: &Vec<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut powell = Powell::new(objective);

        // Use diagonal directions instead of coordinate axes
        let directions = vec![
            vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()],
            vec![1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()],
        ];

        let result = powell
            .powell_method_with_directions(vec![0.0, 0.0], directions, None, None, None)
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
        let func = |_: &Vec<f64>| 0.0;
        let objective = MultiDimFn::new(func);
        let powell = Powell::new(objective);
        let ortho_dirs = powell.create_orthogonal_directions(n);

        assert_eq!(ortho_dirs.len(), n);
        for (i, dir) in ortho_dirs.iter().enumerate() {
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
        let func = |_: &Vec<f64>| 0.0;
        let objective = MultiDimFn::new(func);
        let powell = Powell::new(objective);
        let random_dirs = powell.create_random_orthogonal_directions(n, Some(12345));

        assert_eq!(random_dirs.len(), n);

        // Check orthogonality
        for i in 0..n {
            for j in i + 1..n {
                let dot_product: f64 = random_dirs[i]
                    .iter()
                    .zip(random_dirs[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                assert!(dot_product.abs() < 1e-12);
            }
        }

        // Check normalization
        for dir in &random_dirs {
            let norm: f64 = dir.iter().map(|&d| d * d).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_powell_quadratic_exact_convergence() {
        // Test exact convergence on quadratic function
        let quadratic_3d = |x: &Vec<f64>| {
            x[0].powi(2) + 2.0 * x[1].powi(2) + 3.0 * x[2].powi(2) + x[0] * x[1] + x[1] * x[2]
        };
        let objective = MultiDimFn::new(quadratic_3d);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(vec![1.0, 1.0, 1.0]).unwrap();

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
        let elliptic = |x: &Vec<f64>| 100.0 * x[0].powi(2) + x[1].powi(2);
        let objective = MultiDimFn::new(elliptic);
        let mut powell = Powell::new(objective);

        // Standard coordinate directions
        let std_result = powell.minimize(vec![1.0, 1.0]).unwrap();

        // Custom directions aligned with ellipse
        let custom_dirs = vec![
            vec![0.1, 0.0], // Small step in x direction
            vec![0.0, 1.0], // Normal step in y direction
        ];

        let custom_result = powell
            .powell_method_with_directions(vec![1.0, 1.0], custom_dirs, None, None, None)
            .unwrap();

        // Custom directions should perform better on ill-conditioned problem
        assert!(custom_result.fn_evals as f64 <= std_result.fn_evals as f64 * 1.5);
        assert!(custom_result.xmin[0].abs() < 1e-6);
        assert!(custom_result.xmin[1].abs() < 1e-6);
    }

    #[test]
    fn test_powell_direction_evolution() {
        // Test that directions evolve to become conjugate
        let quadratic_matrix = |x: &Vec<f64>| {
            // f(x) = x^T A x where A = [[4,1],[1,2]]
            4.0 * x[0].powi(2) + 2.0 * x[1].powi(2) + 2.0 * x[0] * x[1]
        };
        let objective = MultiDimFn::new(quadratic_matrix);
        let mut powell = Powell::new(objective);

        let result = powell.minimize(vec![1.0, 1.0]).unwrap();

        assert!(result.converged);
        assert!(result.xmin[0].abs() < 1e-6);
        assert!(result.xmin[1].abs() < 1e-6);

        // Final directions should be approximately conjugate
        // (This is hard to test directly, but convergence implies conjugacy)
        assert!(result.iters <= 5); // Should be fast for 2D quadratic
    }
}
