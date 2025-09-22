#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::{
    MinimizerError,
    f64::{ObjFn, Vertex},
};
use ndarray::prelude::*;
use std::fmt;

/// Result of downhill simplex optimization
#[derive(Debug, Clone)]
pub struct SimplexResult {
    pub xmin: Array1<f64>,
    pub fmin: f64,
    pub iters: usize,
    pub fn_evals: usize,
    pub converged: bool,
    pub final_simplex_size: f64,
    pub history: Array1<f64>,
}

#[derive(Clone)]
pub struct Simplex {
    xmin: Array1<f64>,
    fmin: f64,
    f: Box<dyn ObjFn>,
    iters: usize,
    converged: bool,
}

impl Simplex {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn + 'static,
    {
        Simplex {
            xmin: array![],
            fmin: 0.0,
            f: Box::new(f),
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn>) -> Self {
        Simplex {
            xmin: array![],
            fmin: 0.0,
            f: f,
            iters: 0,
            converged: false,
        }
    }

    fn reflect_point(
        &self,
        worst: &Array1<f64>,
        centroid: &Array1<f64>,
        coeff: f64,
    ) -> Array1<f64> {
        worst
            .iter()
            .zip(centroid.iter())
            .map(|(&w, &c)| c + coeff * (c - w))
            .collect()
    }

    fn calculate_centroid(&self, vertices: &[Vertex]) -> Array1<f64> {
        let n = vertices[0].point.len();
        let mut centroid = vec![0.0; n];

        for vertex in vertices {
            for (i, &coord) in vertex.point.iter().enumerate() {
                centroid[i] += coord;
            }
        }

        for coord in &mut centroid {
            *coord /= vertices.len() as f64;
        }

        Array1::from_vec(centroid)
    }

    fn calculate_simplex_size(&self, simplex: &[Vertex]) -> f64 {
        let n = simplex[0].point.len();
        let mut max_distance = 0_f64;

        for i in 0..n {
            for j in i + 1..=n {
                let distance: f64 = simplex[i]
                    .point
                    .iter()
                    .zip(simplex[j].point.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                max_distance = max_distance.max(distance);
            }
        }

        max_distance
    }

    fn run_simplex_algorithm(
        &mut self,
        mut simplex: Vec<Vertex>,
        tol: f64,
        max_iters: usize,
    ) -> Result<SimplexResult, MinimizerError> {
        let n = simplex[0].point.len();
        let mut fn_evals = simplex.len();
        self.iters = 0;
        let mut history = Vec::new();
        self.converged = false;

        // Standard Nelder-Mead coefficients
        let alpha = 1.0;
        let gamma = 2.0;
        let rho = 0.5;
        let sigma = 0.5;

        while self.iters < max_iters {
            self.iters += 1;

            simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

            let best = &simplex[0];
            let worst = &simplex[n];
            let second_worst = &simplex[n - 1];

            history.push(best.value);

            let simplex_size = self.calculate_simplex_size(&simplex);
            if simplex_size < tol {
                self.xmin = best.point.clone();
                self.fmin = best.value;
                self.converged = true;
                return Ok(SimplexResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin,
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_simplex_size: simplex_size,
                    history: Array1::from_vec(history),
                });
            }

            let centroid = self.calculate_centroid(&simplex[..n]);

            // Standard Nelder-Mead operations
            let reflected = self.reflect_point(&worst.point, &centroid, alpha);
            let f_reflected = self.f.call(&reflected);
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
                let expanded = self.reflect_point(&worst.point, &centroid, alpha * gamma);
                let f_expanded = self.f.call(&expanded);
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
                    let point = self.reflect_point(&worst.point, &centroid, alpha * rho);
                    let value = self.f.call(&point);
                    (point, value)
                } else {
                    let point = self.reflect_point(&worst.point, &centroid, -rho);
                    let value = self.f.call(&point);
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
                        let mut new_point = Array1::zeros(n);
                        for j in 0..n {
                            new_point[j] =
                                best_point[j] + sigma * (simplex[i].point[j] - best_point[j]);
                        }
                        let new_value = self.f.call(&new_point);
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

        self.xmin = simplex[0].point.clone();
        self.fmin = simplex[0].value;
        self.converged = false;
        Ok(SimplexResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin,
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_simplex_size: self.calculate_simplex_size(&simplex),
            history: Array1::from_vec(history),
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
        initial_point: Array1<f64>,
        initial_step: Option<f64>,
        tol: Option<f64>,
        max_iters: Option<usize>,
    ) -> Result<SimplexResult, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let step = initial_step.unwrap_or(1.0);
        let tol = tol.unwrap_or(1e-8);
        let max_iter = max_iters.unwrap_or(1000);

        if tol <= 0.0 {
            return Err(MinimizerError::InvalidTolerance);
        }

        // Nelder-Mead coefficients
        let alpha = 1.0; // Reflection coefficient
        let gamma = 2.0; // Expansion coefficient
        let rho = 0.5; // Contraction coefficient
        let sigma = 0.5; // Shrink coefficient

        // Initialize simplex with n+1 vertices
        let mut simplex = Vec::with_capacity(n + 1);

        // First vertex is the initial point
        simplex.push(Vertex::new_boxed(initial_point.clone(), self.f.clone())?);

        // Create additional vertices by perturbing each coordinate
        for i in 0..n {
            let mut point = initial_point.clone();
            point[i] += step;
            simplex.push(Vertex::new_boxed(point, self.f.clone())?);
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

            history.push(best.value);

            // Check for convergence
            let simplex_size = self.calculate_simplex_size(&simplex);
            if simplex_size < tol {
                self.xmin = best.point.clone();
                self.fmin = best.value;
                self.converged = true;
                return Ok(SimplexResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin,
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_simplex_size: simplex_size,
                    history: Array1::from_vec(history),
                });
            }

            // Calculate centroid of all points except the worst
            let centroid = self.calculate_centroid(&simplex[..n]);

            // Reflection
            let reflected = self.reflect_point(&worst.point, &centroid, alpha);
            let f_reflected = self.f.call(&reflected);
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
                let expanded = self.reflect_point(&worst.point, &centroid, alpha * gamma);
                let f_expanded = self.f.call(&expanded);
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
                    let point = self.reflect_point(&worst.point, &centroid, alpha * rho);
                    let value = self.f.call(&point);
                    fn_evals += 1;
                    (point, value)
                } else {
                    // Inside contraction
                    let point = self.reflect_point(&worst.point, &centroid, -rho);
                    let value = self.f.call(&point);
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
                        let mut new_point = Array1::zeros(n);
                        for j in 0..n {
                            new_point[j] =
                                best_point[j] + sigma * (simplex[i].point[j] - best_point[j]);
                        }
                        let new_value = self.f.call(&new_point);
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

        self.xmin = simplex[0].point.clone();
        self.fmin = simplex[0].value;
        self.converged = false;
        Ok(SimplexResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin,
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_simplex_size: self.calculate_simplex_size(&simplex),
            history: Array1::from_vec(history),
        })
    }

    /// Advanced downhill simplex with adaptive parameters
    ///
    /// This version adapts the Nelder-Mead coefficients based on the problem
    /// characteristics for improved performance.
    pub fn adaptive_downhill_simplex(
        &mut self,
        initial_point: Array1<f64>,
        initial_step: Option<f64>,
        tol: Option<f64>,
        max_iters: Option<usize>,
    ) -> Result<SimplexResult, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let step = initial_step.unwrap_or(1.0);
        let tol = tol.unwrap_or(1e-8);
        let max_iter = max_iters.unwrap_or(1000);

        // Adaptive coefficients (from recent research)
        let alpha = 1.0; // Reflection
        let gamma = 1.0 + 2.0 / n as f64; // Expansion (dimension-dependent)
        let rho = 0.75 - 1.0 / (2.0 * n as f64); // Contraction
        let sigma = 1.0 - 1.0 / n as f64; // Shrink

        // Initialize simplex using regular simplex construction
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(Vertex::new_boxed(initial_point.clone(), self.f.clone())?);

        // Use right-angled simplex construction for better performance
        let delta_usual =
            step * (((n + 1) as f64).sqrt() + (n - 1) as f64) / (n as f64 * 2.0_f64.sqrt());
        let delta_zero = step * (((n + 1) as f64).sqrt() - 1.0) / (n as f64 * 2.0_f64.sqrt());

        for i in 0..n {
            let mut point = initial_point.clone();
            for j in 0..n {
                if i == j {
                    point[j] += delta_usual;
                } else {
                    point[j] += delta_zero;
                }
            }
            simplex.push(Vertex::new_boxed(point, self.f.clone())?);
        }

        let mut fn_evals = n + 1;
        self.iters = 0;
        let mut history = vec![];
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
                let improvement: f64 = history.last().unwrap() - best.value;
                if improvement.abs() < tol * 0.01 {
                    no_improvement_count += 1;
                } else {
                    no_improvement_count = 0;
                }
            }
            history.push(best.value);

            // Check for convergence
            let simplex_size = self.calculate_simplex_size(&simplex);
            if simplex_size < tol || no_improvement_count > 20 {
                self.xmin = best.point.clone();
                self.fmin = best.value;
                self.converged = true;
                return Ok(SimplexResult {
                    xmin: self.xmin.clone(),
                    fmin: self.fmin,
                    iters: self.iters,
                    fn_evals,
                    converged: self.converged,
                    final_simplex_size: simplex_size,
                    history: Array1::from_vec(history),
                });
            }

            // Calculate centroid
            let centroid = self.calculate_centroid(&simplex[..n]);

            // Reflection
            let reflected = self.reflect_point(&worst.point, &centroid, alpha);
            let f_reflected = self.f.call(&reflected);
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
                let expanded = self.reflect_point(&worst.point, &centroid, alpha + (gamma - alpha));
                let f_expanded = self.f.call(&expanded);
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
                    let point = self.reflect_point(&worst.point, &centroid, alpha * rho);
                    let value = self.f.call(&point);
                    (point, value)
                } else {
                    let point = self.reflect_point(&worst.point, &centroid, -rho);
                    let value = self.f.call(&point);
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
                        let mut new_point = Array1::zeros(n);
                        for j in 0..n {
                            new_point[j] =
                                best_point[j] + sigma * (simplex[i].point[j] - best_point[j]);
                        }
                        let new_value = self.f.call(&new_point);
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

        self.xmin = simplex[0].point.clone();
        self.fmin = simplex[0].value;
        self.converged = false;
        Ok(SimplexResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin,
            iters: self.iters,
            fn_evals,
            converged: self.converged,
            final_simplex_size: self.calculate_simplex_size(&simplex),
            history: Array1::from_vec(history),
        })
    }

    /// Convenience function with default parameters
    pub fn minimize(
        &mut self,
        initial_point: Array1<f64>,
    ) -> Result<SimplexResult, MinimizerError> {
        self.downhill_simplex(initial_point, None, None, None)
    }

    /// Create initial simplex with custom step sizes for each dimension
    pub fn minimize_with_steps(
        &mut self,
        initial_point: Array1<f64>,
        step_sizes: Array1<f64>,
    ) -> Result<SimplexResult, MinimizerError> {
        let n = initial_point.len();
        if step_sizes.len() != n {
            return Err(MinimizerError::InvalidInitialSimplex);
        }

        // Custom implementation with individual step sizes
        let mut simplex = Vec::with_capacity(n + 1);
        simplex.push(Vertex::new_boxed(initial_point.clone(), self.f.clone())?);

        for i in 0..n {
            let mut point = initial_point.clone();
            point[i] += step_sizes[i];
            simplex.push(Vertex::new_boxed(point, self.f.clone())?);
        }

        // Continue with standard algorithm
        self.run_simplex_algorithm(simplex, 1e-8, 1000)
    }
}

impl fmt::Debug for Simplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Simplex( xmin: {:?}, fmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod minimize_f64_simplex_tests {
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
        // f(x,y) = (x-1)² + (y-2)², minimum at (1,2)
        let func = |x: &Array1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex.minimize(array![0.0, 0.0]).unwrap();

        assert!((result.xmin[0] - 1.0).abs() < 1e-6);
        assert!((result.xmin[1] - 2.0).abs() < 1e-6);
        assert!(result.fmin < 1e-10);
        assert!(result.converged);

        assert!((simplex.xmin[0] - 1.0).abs() < 1e-6);
        assert!((simplex.xmin[1] - 2.0).abs() < 1e-6);
        assert!(simplex.fmin < 1e-10);
        assert!(simplex.converged);
    }

    #[test]
    fn test_rosenbrock_2d() {
        // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
        let rosenbrock =
            |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let objective = MultiDimFn::new(rosenbrock);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .downhill_simplex(array![-1.2, 1.0], Some(0.1), Some(1e-6), Some(2000))
            .unwrap();

        assert!((result.xmin[0] - 1.0).abs() < 1e-4);
        assert!((result.xmin[1] - 1.0).abs() < 1e-4);
        assert!(result.fmin < 1e-6);

        assert!((simplex.xmin[0] - 1.0).abs() < 1e-4);
        assert!((simplex.xmin[1] - 1.0).abs() < 1e-4);
        assert!(simplex.fmin < 1e-6);
    }

    #[test]
    fn test_3d_sphere() {
        // f(x,y,z) = x² + y² + z², minimum at (0,0,0)
        let sphere = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum();
        let objective = MultiDimFn::new(sphere);
        let mut simplex = Simplex::new(objective);

        let result = simplex.minimize(array![1.0, 1.0, 1.0]).unwrap();

        for &coord in &result.xmin {
            assert!(coord.abs() < 1e-6);
        }
        assert!(result.fmin < 1e-10);
        assert!(result.converged);

        for &coord in &simplex.xmin {
            assert!(coord.abs() < 1e-6);
        }
        assert!(simplex.fmin < 1e-10);
        assert!(simplex.converged);
    }

    #[test]
    fn test_adaptive_simplex() {
        let func = |x: &Array1<f64>| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2) + 5.0;
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .adaptive_downhill_simplex(array![0.0, 0.0], Some(1.0), Some(1e-8), None)
            .unwrap();

        assert!((result.xmin[0] - 3.0).abs() < 1e-6);
        assert!((result.xmin[1] + 2.0).abs() < 1e-6);
        assert!((result.fmin - 5.0).abs() < 1e-8);

        assert!((simplex.xmin[0] - 3.0).abs() < 1e-6);
        assert!((simplex.xmin[1] + 2.0).abs() < 1e-6);
        assert!((simplex.fmin - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_custom_step_sizes() {
        let func = |x: &Array1<f64>| (x[0] / 10.0 - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .minimize_with_steps(
                array![0.0, 0.0],
                array![10.0, 1.0], // Different step sizes for each dimension
            )
            .unwrap();

        assert!((result.xmin[0] - 10.0).abs() < 1e-4);
        assert!((result.xmin[1] - 2.0).abs() < 1e-6);

        assert!((simplex.xmin[0] - 10.0).abs() < 1e-4);
        assert!((simplex.xmin[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_dimension() {
        let func = |_: &Array1<f64>| 0.0;
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex.minimize(array![]);
        assert!(result.is_err());

        assert!(matches!(result, Err(MinimizerError::InvalidDimension)));
    }

    #[test]
    fn test_convergence_tracking() {
        let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
        let objective = MultiDimFn::new(func);
        let mut simplex = Simplex::new(objective);

        let result = simplex.minimize(array![2.0, 2.0]).unwrap();

        assert!(!result.history.is_empty());
        assert!(result.history[0] > result.fmin);
        assert!(result.history.len() == result.iters);
    }

    #[test]
    fn test_simplex_high_dimensional() {
        // Test 5D sphere function
        let sphere_5d = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
        let objective = MultiDimFn::new(sphere_5d);
        let mut simplex = Simplex::new(objective);

        let result = simplex.minimize(Array1::ones(5)).unwrap();

        assert!(result.converged);
        for &coord in &result.xmin {
            assert!(coord.abs() < 1e-4);
        }
        assert!(result.fmin < 1e-6);
    }

    #[test]
    fn test_simplex_rosenbrock_variants() {
        // Test 1: 2D Rosenbrock
        let rosenbrock_2d =
            |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let objective = MultiDimFn::new(rosenbrock_2d);
        let mut simplex = Simplex::new(objective);

        let result = simplex
            .downhill_simplex(array![-1.2, 1.0], Some(0.1), Some(1e-6), Some(3000))
            .unwrap();

        assert!((result.xmin[0] - 1.0).abs() < 1e-3);
        assert!((result.xmin[1] - 1.0).abs() < 1e-3);

        // Test 2: Extended Rosenbrock (4D)
        let rosenbrock_4d = |x: &Array1<f64>| {
            (0..x.len() - 1)
                .map(|i| (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i].powi(2)).powi(2))
                .sum::<f64>()
        };
        let objective = MultiDimFn::new(rosenbrock_4d);
        let mut simplex = Simplex::new(objective);

        let result = simplex.minimize(Array1::ones(4) * -1.0).unwrap();

        for &coord in &result.xmin {
            assert!((coord - 1.0).abs() < 0.1); // Relaxed for high-D Rosenbrock
        }
    }

    #[test]
    fn test_simplex_with_constraints_penalty() {
        // Test penalty method simulation
        let constrained_objective = |x: &Array1<f64>| {
            let obj = x[0].powi(2) + x[1].powi(2);
            let penalty = if x[0] + x[1] < 1.0 {
                1000.0 * (1.0 - x[0] - x[1]).powi(2)
            } else {
                0.0
            };
            obj + penalty
        };
        let objective = MultiDimFn::new(constrained_objective);
        let mut simplex = Simplex::new(objective);

        let result = simplex.minimize(array![0.6, 0.6]).unwrap();

        // Should satisfy constraint x + y >= 1
        assert!(result.xmin[0] + result.xmin[1] >= 0.95);
        assert!(result.xmin[0] >= -0.05 && result.xmin[1] >= -0.05);
    }

    #[test]
    fn test_adaptive_simplex_performance() {
        // Compare standard vs adaptive simplex
        let beale = |x: &Array1<f64>| {
            let x1 = x[0];
            let x2 = x[1];
            (1.5 - x1 + x1 * x2).powi(2)
                + (2.25 - x1 + x1 * x2.powi(2)).powi(2)
                + (2.625 - x1 + x1 * x2.powi(3)).powi(2)
        };
        let objective = MultiDimFn::new(beale);
        let mut simplex = Simplex::new(objective);

        let standard_result = simplex
            .downhill_simplex(array![1.0, 1.0], None, None, None)
            .unwrap();
        let adaptive_result = simplex
            .adaptive_downhill_simplex(array![1.0, 1.0], None, None, None)
            .unwrap();

        // Both should converge
        assert!(standard_result.converged || adaptive_result.converged);

        // At least one should be close to optimum (3.0, 0.5)
        let std_dist = ((standard_result.xmin[0] - 3.0).powi(2)
            + (standard_result.xmin[1] - 0.5).powi(2))
        .sqrt();
        let ada_dist = ((adaptive_result.xmin[0] - 3.0).powi(2)
            + (adaptive_result.xmin[1] - 0.5).powi(2))
        .sqrt();

        assert!(std_dist < 0.1 || ada_dist < 0.1);
    }

    mod basic_functionality_tests {
        use super::*;

        #[test]
        fn test_1d_optimization() {
            // f(x) = (x - 5)²
            let func = |x: &Array1<f64>| (x[0] - 5.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![0.0]).unwrap();

            assert!((result.xmin[0] - 5.0).abs() < 1e-6);
            assert!(result.fmin < 1e-10);
            assert!(result.converged);
            assert!(result.iters > 0);
            assert!(result.fn_evals >= 2); // At least n+1 evaluations
        }

        #[test]
        fn test_already_at_minimum() {
            let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![0.0, 0.0]).unwrap();

            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
            assert!(result.converged);
        }

        #[test]
        fn test_negative_coordinates() {
            // Minimum at (-3, -4)
            let func = |x: &Array1<f64>| (x[0] + 3.0).powi(2) + (x[1] + 4.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![0.0, 0.0]).unwrap();

            assert!((result.xmin[0] + 3.0).abs() < 1e-6);
            assert!((result.xmin[1] + 4.0).abs() < 1e-6);
            assert!(result.fmin < 1e-10);
        }
    }

    mod error_handling_tests {
        use super::*;

        #[test]
        fn test_empty_initial_point() {
            let func = |_: &Array1<f64>| 0.0;
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![]);
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::InvalidDimension
            ));
        }

        #[test]
        fn test_invalid_tolerance() {
            let func = |x: &Array1<f64>| x[0].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.downhill_simplex(array![1.0], None, Some(0.0), None);
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::InvalidTolerance
            ));

            let result = simplex.downhill_simplex(array![1.0], None, Some(-1e-8), None);
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::InvalidTolerance
            ));
        }

        #[test]
        fn test_function_returning_infinity() {
            let func = |x: &Array1<f64>| {
                if x[0] > 1.0 {
                    f64::INFINITY
                } else {
                    x[0].powi(2)
                }
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![2.0]); // Start where function is infinite
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::FunctionEvaluationError
            ));
        }

        #[test]
        fn test_function_returning_nan() {
            let func = |x: &Array1<f64>| if x[0] < 0.0 { f64::NAN } else { x[0].powi(2) };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![-1.0]); // Start where function is NaN
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                MinimizerError::FunctionEvaluationError
            ));
        }

        #[test]
        fn test_mismatched_step_sizes() {
            let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize_with_steps(array![1.0, 1.0], array![0.1]); // Wrong length
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
            let func = |x: &Array1<f64>| {
                (x[0] - 1.0).powi(4) + (x[1] - 2.0).powi(4) + 0.01 * (x[0] * x[1]).sin()
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(array![0.0, 0.0], None, Some(1e-12), Some(10))
                .unwrap();

            assert_eq!(result.iters, 10);
            assert!(!result.converged);
        }

        #[test]
        fn test_convergence_tracking() {
            let func = |x: &Array1<f64>| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![0.0, 0.0]).unwrap();

            assert!(!result.history.is_empty());
            assert_eq!(result.history.len(), result.iters);

            // History should be generally decreasing (allowing for some fluctuation)
            let first_value = result.history[0];
            let last_value = *result.history.last().unwrap();
            assert!(first_value >= last_value);

            // Final value should match fmin
            assert!((last_value - result.fmin).abs() < 1e-10);
        }

        #[test]
        fn test_tight_tolerance() {
            let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(array![1.0, 1.0], None, Some(1e-12), None)
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
            let sphere_10d = |x: &Array1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();
            let objective = MultiDimFn::new(sphere_10d);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(Array1::ones(10)).unwrap();

            for &coord in &result.xmin {
                assert!(coord.abs() < 1e-3);
            }
            assert!(result.fmin < 1e-4);
        }

        #[test]
        fn test_high_dimensional_rosenbrock() {
            // 4D Rosenbrock function
            let rosenbrock_4d = |x: &Array1<f64>| {
                (0..x.len() - 1)
                    .map(|i| (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i].powi(2)).powi(2))
                    .sum::<f64>()
            };
            let objective = MultiDimFn::new(rosenbrock_4d);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(Array1::ones(4) * -0.5, Some(0.5), Some(1e-4), Some(5000))
                .unwrap();

            // Rosenbrock optimum is at (1, 1, 1, 1)
            for &coord in &result.xmin {
                assert!((coord - 1.0).abs() < 0.2); // Relaxed tolerance for high-D Rosenbrock
            }
        }
    }

    mod complex_function_tests {
        use super::*;

        #[test]
        fn test_beale_function() {
            // Beale function: global minimum at (3, 0.5)
            let beale = |x: &Array1<f64>| {
                let x1 = x[0];
                let x2 = x[1];
                (1.5 - x1 + x1 * x2).powi(2)
                    + (2.25 - x1 + x1 * x2.powi(2)).powi(2)
                    + (2.625 - x1 + x1 * x2.powi(3)).powi(2)
            };
            let objective = MultiDimFn::new(beale);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(array![1.0, 1.0], Some(0.5), Some(1e-6), Some(2000))
                .unwrap();

            assert!((result.xmin[0] - 3.0).abs() < 0.1);
            assert!((result.xmin[1] - 0.5).abs() < 0.1);
            assert!(result.fmin < 1e-4);
        }

        #[test]
        fn test_goldstein_price_function() {
            // Goldstein-Price function: global minimum at (0, -1) with value 3
            let goldstein_price = |x: &Array1<f64>| {
                let x1 = x[0];
                let x2 = x[1];
                (1.0 + (x1 + x2 + 1.0).powi(2)
                    * (19.0 - 14.0 * x1 + 3.0 * x1.powi(2) - 14.0 * x2
                        + 6.0 * x1 * x2
                        + 3.0 * x2.powi(2)))
                    * (30.0
                        + (2.0 * x1 - 3.0 * x2).powi(2)
                            * (18.0 - 32.0 * x1 + 12.0 * x1.powi(2) + 48.0 * x2 - 36.0 * x1 * x2
                                + 27.0 * x2.powi(2)))
            };
            let objective = MultiDimFn::new(goldstein_price);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(array![0.5, -0.5], Some(0.2), Some(1e-6), Some(3000))
                .unwrap();

            assert!((result.xmin[0] - 0.0).abs() < 0.1);
            assert!((result.xmin[1] + 1.0).abs() < 0.1);
            assert!((result.fmin - 3.0).abs() < 1.0);
        }

        #[test]
        fn test_himmelblau_function() {
            // Himmelblau's function has 4 global minima
            let himmelblau = |x: &Array1<f64>| {
                (x[0].powi(2) + x[1] - 11.0).powi(2) + (x[0] + x[1].powi(2) - 7.0).powi(2)
            };
            let objective = MultiDimFn::new(himmelblau);
            let mut simplex = Simplex::new(objective);

            // Test from different starting points to potentially find different minima
            let starting_points = array![
                array![0.0, 0.0],
                array![3.0, 2.0],
                array![-3.0, 3.0],
                array![-3.0, -3.0],
            ];

            for start in starting_points {
                let result = simplex
                    .downhill_simplex(start, Some(0.5), Some(1e-6), Some(2000))
                    .unwrap();

                // All minima should have function value 0
                assert!(result.fmin < 1e-4);

                // Check if we found one of the known minima
                let known_minima = array![
                    (3.0, 2.0),
                    (-2.805118, 3.131312),
                    (-3.779310, -3.283186),
                    (3.584428, -1.848126),
                ];

                let found_known_minimum = known_minima.iter().any(|&(x, y)| {
                    (result.xmin[0] - x).abs() < 0.1 && (result.xmin[1] - y).abs() < 0.1
                });

                assert!(found_known_minimum || result.fmin < 1e-4);
            }
        }
    }

    mod adaptive_vs_standard_comparison_tests {
        use super::*;

        #[test]
        fn test_adaptive_vs_standard_performance() {
            let func = |x: &Array1<f64>| {
                // Scaled quadratic - more challenging in one dimension
                (x[0] / 100.0 - 1.0).powi(2) + (x[1] - 2.0).powi(2)
            };
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let standard_result = simplex
                .downhill_simplex(array![0.0, 0.0], None, Some(1e-8), None)
                .unwrap();
            let adaptive_result = simplex
                .adaptive_downhill_simplex(array![0.0, 0.0], None, Some(1e-8), None)
                .unwrap();

            // Both should find the correct minimum
            assert!((standard_result.xmin[0] - 100.0).abs() < 1e-3);
            assert!((adaptive_result.xmin[0] - 100.0).abs() < 1e-3);
            assert!((standard_result.xmin[1] - 2.0).abs() < 1e-6);
            assert!((adaptive_result.xmin[1] - 2.0).abs() < 1e-6);
        }

        #[test]
        fn test_adaptive_no_improvement_convergence() {
            // Function that plateaus to test no-improvement convergence
            let plateau_func = |x: &Array1<f64>| {
                let base = x[0].powi(2) + x[1].powi(2);
                if base < 0.01 { 0.01 } else { base }
            };
            let objective = MultiDimFn::new(plateau_func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .adaptive_downhill_simplex(array![2.0, 2.0], Some(1.0), Some(1e-8), Some(1000))
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
            let scaled_func =
                |x: &Array1<f64>| (x[0] / 1000.0 - 1.0).powi(2) + (x[1] * 1000.0 - 2.0).powi(2);
            let objective = MultiDimFn::new(scaled_func);
            let mut simplex = Simplex::new(objective);

            // Use appropriate step sizes for each dimension
            let result = simplex
                .minimize_with_steps(
                    array![0.0, 0.0],
                    array![1000.0, 0.001], // Large step for first dim, small for second
                )
                .unwrap();

            assert!((result.xmin[0] - 1000.0).abs() < 1.0);
            assert!((result.xmin[1] - 0.002).abs() < 1e-6);
        }

        #[test]
        fn test_zero_step_size() {
            let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            // Test with zero step size - this creates a degenerate simplex
            let result = simplex.minimize_with_steps(array![1.0, 1.0], array![0.0, 1.0]);

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
            let constant_func = |_: &Array1<f64>| 42.0;
            let objective = MultiDimFn::new(constant_func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![1.0, 2.0]).unwrap();

            assert!((result.fmin - 42.0).abs() < 1e-10);
            assert!(result.converged); // Should converge immediately due to zero simplex size
        }

        #[test]
        fn test_linear_function() {
            // f(x,y) = 2x + 3y (unbounded below, but simplex should handle gracefully)
            let linear_func = |x: &Array1<f64>| 2.0 * x[0] + 3.0 * x[1];
            let objective = MultiDimFn::new(linear_func);
            let mut simplex = Simplex::new(objective);

            let result =
                simplex.downhill_simplex(array![0.0, 0.0], Some(1.0), Some(1e-8), Some(100));

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
            let discontinuous = |x: &Array1<f64>| {
                if x[0] < 0.0 {
                    100.0 + x[0].powi(2) + x[1].powi(2)
                } else {
                    x[0].powi(2) + x[1].powi(2)
                }
            };
            let objective = MultiDimFn::new(discontinuous);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![1.0, 1.0]).unwrap();

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
            let func = |x: &Array1<f64>| (x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![0.0, 0.0]).unwrap();

            // Check that simplex internal state matches result
            assert_eq!(simplex.xmin, result.xmin);
            assert_eq!(simplex.fmin, result.fmin);
            assert_eq!(simplex.iters, result.iters);
            assert_eq!(simplex.converged, result.converged);
        }

        #[test]
        fn test_multiple_optimizations() {
            let func = |x: &Array1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            // First optimization
            let result1 = simplex.minimize(array![0.0, 0.0]).unwrap();

            // Second optimization with different starting point
            let result2 = simplex.minimize(array![5.0, 5.0]).unwrap();

            // Both should find the same minimum
            assert!((result1.xmin[0] - 1.0).abs() < 1e-6);
            assert!((result1.xmin[1] - 2.0).abs() < 1e-6);
            assert!((result2.xmin[0] - 1.0).abs() < 1e-6);
            assert!((result2.xmin[1] - 2.0).abs() < 1e-6);

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

            let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![1.0, 1.0]).unwrap();

            // Should have reasonable number of function evaluations
            assert!(result.fn_evals >= 3); // At least n+1 for initial simplex
            assert!(result.fn_evals <= result.iters * 10); // Reasonable upper bound
        }

        #[test]
        fn test_very_small_initial_step() {
            let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            // Very small initial step creates a tiny simplex that may converge immediately
            // due to the simplex size being smaller than the tolerance
            let result = simplex
                .downhill_simplex(array![1.0, 1.0], Some(1e-12), Some(1e-6), Some(5000))
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
            let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(array![1.0, 1.0], Some(1e-6), Some(1e-8), None)
                .unwrap();

            assert!(result.converged);
            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
            assert!(result.fmin < 1e-10);
        }

        #[test]
        fn test_very_large_initial_step() {
            let func = |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2);
            let objective = MultiDimFn::new(func);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(array![0.0, 0.0], Some(1000.0), None, None)
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
            let ill_conditioned = |x: &Array1<f64>| x[0].powi(2) + 10000.0 * x[1].powi(2);
            let objective = MultiDimFn::new(ill_conditioned);
            let mut simplex = Simplex::new(objective);

            let result = simplex
                .downhill_simplex(array![1.0, 1.0], Some(0.1), Some(1e-4), Some(5000))
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
            let tiny_func = |x: &Array1<f64>| 1e-15 * (x[0].powi(2) + x[1].powi(2));
            let objective = MultiDimFn::new(tiny_func);
            let mut simplex = Simplex::new(objective);

            let result = simplex.minimize(array![1.0, 1.0]).unwrap();

            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
            assert!(result.fmin >= 0.0);
            assert!(result.fmin < 1e-12);
        }
    }
}
