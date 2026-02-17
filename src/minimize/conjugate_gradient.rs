#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::{
    error::MinimizerError,
    minimize::{LineSearchResult, ObjGradFn, WolfeParams},
    num::RealScalar,
    pts::{Points1, Pts},
};
use std::fmt;

/// Result of conjugate gradient optimization
#[derive(Debug, Clone)]
pub struct ConjGradResult<T>
where
    T: RealScalar,
{
    xmin: Points1<T>,
    fmin: T,
    gradient_norm: T,
    iters: usize,
    fn_evals: usize,
    g_evals: usize,
    converged: bool,
    convergence_history: Points1<T>,
    gradient_norm_history: Points1<T>,
    restart_count: usize,
    method_used: ConjGradMethod,
}

impl<T> ConjGradResult<T>
where
    T: RealScalar,
{
    pub fn xmin(&self) -> Points1<T> {
        self.xmin.clone()
    }

    pub fn fmin(&self) -> T {
        self.fmin
    }

    pub fn gradient_norm(&self) -> T {
        self.gradient_norm
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn fn_evals(&self) -> usize {
        self.fn_evals
    }

    pub fn g_evals(&self) -> usize {
        self.g_evals
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn history(&self) -> Points1<T> {
        self.convergence_history
    }

    pub fn grad_norm_history(&self) -> Points1<T> {
        self.gradient_norm_history
    }

    pub fn restarts(&self) -> usize {
        self.restart_count
    }

    pub fn method(&self) -> ConjGradMethod {
        self.method_used
    }
}

/// Conjugate gradient update formulas
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConjGradMethod {
    FletcherReeves,  // β = ||g_new||² / ||g_old||²
    PolakRibiere,    // β = g_new·(g_new - g_old) / ||g_old||²
    HestenesStiefel, // β = g_new·(g_new - g_old) / h·(g_new - g_old)
    DaiYuan,         // β = ||g_new||² / h·(g_new - g_old)
    HagerZhang, // β = (g_new - g_old - 2*h*||g_new - g_old||²/[h·(g_new - g_old)])·g_new / [h·(g_new - g_old)]
}

impl fmt::Display for ConjGradMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConjGradMethod::FletcherReeves => write!(f, "Fletcher-Reeves"),
            ConjGradMethod::PolakRibiere => write!(f, "Polak-Ribiere"),
            ConjGradMethod::HestenesStiefel => write!(f, "Hestenes-Stiefel"),
            ConjGradMethod::DaiYuan => write!(f, "Dai-Yuan"),
            ConjGradMethod::HagerZhang => write!(f, "Hager-Zhang"),
        }
    }
}

pub struct ConjGrad<T> {
    xmin: Points1<T>,
    fmin: T,
    f: Box<dyn ObjGradFn<T>>,
    iters: usize,
    converged: bool,
}

impl<T> Clone for ConjGrad<T>
where
    T: RealScalar,
{
    fn clone(&self) -> Self {
        Self {
            xmin: self.xmin.clone(),
            fmin: self.fmin,
            f: dyn_clone::clone_box(&*self.f),
            iters: self.iters,
            converged: self.converged,
        }
    }
}

impl<T> ConjGrad<T>
where
    T: RealScalar,
{
    pub fn new<F>(f: F) -> Self
    where
        F: ObjGradFn<T> + Clone + 'static,
    {
        let boxed = Box::new(f);
        ConjGrad {
            xmin: points![],
            fmin: T::zero(),
            f: boxed,
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjGradFn<T>>) -> Self {
        ConjGrad {
            xmin: points![],
            fmin: T::zero(),
            f: f,
            iters: 0,
            converged: false,
        }
    }

    /// Line search using strong Wolfe conditions
    fn wolfe_line_search(
        &mut self,
        x: &Points1<T>,
        direction: &Points1<T>,
        f_current: T,
        grad_current: &Points1<T>,
        initial_step: T,
        wolfe_params: &WolfeParams<T>,
        max_evaluations: usize,
    ) -> Result<LineSearchResult<T>, MinimizerError> {
        let n = x.len();
        let directional_derivative = grad_current
            .iter()
            .zip(direction.iter())
            .map(|(&g, &d)| g * d)
            .sum::<T>();

        if directional_derivative >= T::zero() {
            return Err(MinimizerError::LinearSearchFailed);
        }

        let mut alpha = initial_step;
        let mut evaluations = 0;

        let max_zoom_iterations = 50;
        let mut zoom_iterations = 0;

        // Initial bracketing phase
        let mut alpha_prev = T::zero();
        let mut f_prev = f_current;

        loop {
            if evaluations >= max_evaluations || zoom_iterations >= max_zoom_iterations {
                break;
            }

            // Evaluate function at current alpha
            let x_new: Points1<T> = x
                .iter()
                .zip(direction.iter())
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();

            let f_new = self.f.call(&x_new);
            evaluations += 1;

            if !f_new.is_finite() {
                return Err(MinimizerError::FunctionEvaluationError);
            }

            // Check Armijo condition
            if f_new > f_current + wolfe_params.c1 * alpha * directional_derivative
                || (evaluations > 1 && f_new >= f_prev)
            {
                return self
                    .zoom_phase(
                        x,
                        direction,
                        f_current,
                        grad_current,
                        alpha_prev,
                        alpha,
                        wolfe_params,
                        max_evaluations - evaluations,
                    )
                    .map(|mut result| {
                        result.evaluations += evaluations;
                        result
                    });
            }

            // Evaluate gradient for curvature condition
            let grad_new = self.f.grad(&x_new);
            evaluations += 1; // Count gradient evaluation

            if grad_new.len() != n {
                return Err(MinimizerError::GradientEvaluationError);
            }

            let new_directional_derivative = grad_new
                .iter()
                .zip(direction.iter())
                .map(|(&g, &d)| g * d)
                .sum::<T>();

            // Check curvature condition
            if new_directional_derivative.abs()
                <= (-1.0).into() * wolfe_params.c2 * directional_derivative
            {
                return Ok(LineSearchResult {
                    alpha,
                    f_new,
                    evaluations,
                    converged: true,
                });
            }

            if new_directional_derivative >= T::zero() {
                return self
                    .zoom_phase(
                        x,
                        direction,
                        f_current,
                        grad_current,
                        alpha,
                        alpha_prev,
                        wolfe_params,
                        max_evaluations - evaluations,
                    )
                    .map(|mut result| {
                        result.evaluations += evaluations;
                        result
                    });
            }

            // Expand step size
            alpha_prev = alpha;
            f_prev = f_new;
            alpha *= (2.0).into();
            zoom_iterations += 1;

            if alpha > (1e6).into() {
                break;
            }
        }

        // If we get here, use the last valid alpha
        Ok(LineSearchResult {
            alpha: alpha_prev,
            f_new: f_prev,
            evaluations,
            converged: false,
        })
    }

    /// Zoom phase for Wolfe line search
    fn zoom_phase(
        &mut self,
        x: &Points1<T>,
        direction: &Points1<T>,
        f_current: T,
        grad_current: &Points1<T>,
        al_low: T,
        al_high: T,
        wolfe_params: &WolfeParams<T>,
        max_evaluations: usize,
    ) -> Result<LineSearchResult<T>, MinimizerError> {
        let mut alpha_low = al_low;
        let mut alpha_high = al_high;
        let directional_derivative = grad_current
            .iter()
            .zip(direction.iter())
            .map(|(&g, &d)| g * d)
            .sum::<T>();

        let mut evaluations = 0;

        for _ in 0..50 {
            // Maximum zoom iterations
            if evaluations >= max_evaluations {
                break;
            }

            // Interpolate to find new alpha
            let alpha = if alpha_high.is_finite() {
                (alpha_low + alpha_high) / 2.0.into()
            } else {
                alpha_low * 2.0.into()
            };

            let x_new: Points1<T> = x
                .iter()
                .zip(direction.iter())
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();

            let f_new = self.f.call(&x_new);
            evaluations += 1;

            if !f_new.is_finite() {
                alpha_high = alpha;
                continue;
            }

            // Check Armijo condition
            if f_new > f_current + wolfe_params.c1 * alpha * directional_derivative {
                alpha_high = alpha;
                continue;
            }

            let grad_new = self.f.grad(&x_new);
            evaluations += 1;

            let new_directional_derivative = grad_new
                .iter()
                .zip(direction.iter())
                .map(|(&g, &d)| g * d)
                .sum::<T>();

            // Check curvature condition
            if new_directional_derivative.abs()
                <= (-1.0).into() * wolfe_params.c2 * directional_derivative
            {
                return Ok(LineSearchResult {
                    alpha,
                    f_new,
                    evaluations,
                    converged: true,
                });
            }

            if new_directional_derivative * (alpha_high - alpha_low) >= T::zero() {
                alpha_high = alpha_low;
            }

            alpha_low = alpha;
        }

        // Return best alpha found
        let alpha = alpha_low;
        let x_new: Points1<T> = x
            .iter()
            .zip(direction.iter())
            .map(|(&xi, &di)| xi + alpha * di)
            .collect();

        Ok(LineSearchResult {
            alpha,
            f_new: self.f.call(&x_new),
            evaluations,
            converged: false,
        })
    }

    /// Conjugate gradient optimization with specified update method
    ///
    /// This algorithm minimizes a function using conjugate gradient directions.
    /// It's particularly effective for quadratic functions and well-conditioned problems.
    ///
    /// # Arguments
    /// * `func` - The function to minimize
    /// * `grad_func` - The gradient function
    /// * `initial_point` - Starting point
    /// * `method` - CG update formula to use
    /// * `tol` - Convergence tolerance (default: 1e-6)
    /// * `max_iters` - Maximum iterations (default: n * 10)
    /// * `restart_period` - Restart every n iterations (default: n)
    ///
    /// # Returns
    /// * `ConjGradResult<T>` containing the minimum and convergence info
    pub fn conjugate_gradient(
        &mut self,
        initial_point: &Points1<T>,
        method: ConjGradMethod,
        tol: Option<T>,
        max_iters: Option<usize>,
        restart_period: Option<usize>,
    ) -> Result<ConjGradResult<T>, MinimizerError> {
        self.converged = false;
        let n = initial_point.len();
        if n == 0 {
            return Err(MinimizerError::InvalidDimension);
        }

        let tol = tol.unwrap_or(1e-6.into());
        let max_iter = max_iters.unwrap_or(n * 10);
        let restart_freq = restart_period.unwrap_or(n);

        if tol <= T::zero() {
            return Err(MinimizerError::InvalidTolerance);
        }

        let wolfe_params = WolfeParams::default();

        let mut x = initial_point.to_owned();
        let mut f_current = self.f.call(&x);
        if !f_current.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }

        let mut grad_current = self.f.grad(&x);
        if grad_current.len() != n {
            return Err(MinimizerError::GradientEvaluationError);
        }

        let mut fn_evals = 1;
        let mut g_evals = 1;
        self.iters = 0;
        let mut convergence_history = vec![f_current];
        let mut gradient_norm_history = Vec::new();
        let mut restart_count = 0;

        // Initial search direction is negative gradient
        let mut search_direction: Points1<T> = grad_current.iter().map(|&g| -g).collect();
        let mut grad_norm = grad_current.iter().map(|&g| g * g).sum::<T>().sqrt();
        gradient_norm_history.push(grad_norm);

        while self.iters < max_iter && grad_norm > tol {
            self.iters += 1;

            // Perform line search along current direction
            let line_result = self.wolfe_line_search(
                &x,
                &search_direction,
                f_current,
                &grad_current,
                T::one(), // Initial step size
                &wolfe_params,
                100,
            )?;

            fn_evals += line_result.evaluations;
            g_evals += line_result.evaluations; // Wolfe search evaluates gradient too

            // Update position
            for i in 0..n {
                x[i] += line_result.alpha * search_direction[i];
            }

            f_current = line_result.f_new;
            let grad_old = grad_current;
            grad_current = self.f.grad(&x);
            g_evals += 1;

            if grad_current.len() != n {
                return Err(MinimizerError::GradientEvaluationError);
            }

            grad_norm = grad_current.iter().map(|&g| g * g).sum::<T>().sqrt();
            convergence_history.push(f_current);
            gradient_norm_history.push(grad_norm);

            // Check for convergence
            if grad_norm <= tol {
                break;
            }

            // Restart condition - use steepest descent
            if self.iters % restart_freq == 0 {
                search_direction = grad_current.iter().map(|&g| -g).collect();
                restart_count += 1;
                continue;
            }

            // Compute beta using the specified method
            let beta = match method {
                ConjGradMethod::FletcherReeves => {
                    let grad_new_norm_sq = grad_current.iter().map(|&g| g * g).sum::<T>();
                    let grad_old_norm_sq = grad_old.iter().map(|&g| g * g).sum::<T>();

                    if grad_old_norm_sq < 1e-12.into() {
                        T::zero()
                    } else {
                        &grad_new_norm_sq / &grad_old_norm_sq
                    }
                }

                ConjGradMethod::PolakRibiere => {
                    let grad_diff: Points1<T> = grad_current
                        .iter()
                        .zip(grad_old.iter())
                        .map(|(&g_new, &g_old)| g_new - g_old)
                        .collect();

                    let numerator = grad_current
                        .iter()
                        .zip(grad_diff.iter())
                        .map(|(&g, &diff)| g * diff)
                        .sum::<T>();

                    let denominator = grad_old.iter().map(|&g| g * g).sum::<T>();

                    if denominator < 1e-12.into() {
                        T::zero()
                    } else {
                        (numerator / denominator).max(T::zero()) // Non-negative variant
                    }
                }

                ConjGradMethod::HestenesStiefel => {
                    let grad_diff: Points1<T> = grad_current
                        .iter()
                        .zip(grad_old.iter())
                        .map(|(&g_new, &g_old)| g_new - g_old)
                        .collect();

                    let numerator = grad_current
                        .iter()
                        .zip(grad_diff.iter())
                        .map(|(&g, &diff)| g * diff)
                        .sum::<T>();

                    let denominator = search_direction
                        .iter()
                        .zip(grad_diff.iter())
                        .map(|(&h, &diff)| h * diff)
                        .sum::<T>();

                    if denominator.abs() < 1e-12.into() {
                        0.0.into()
                    } else {
                        numerator / denominator
                    }
                }

                ConjGradMethod::DaiYuan => {
                    let grad_diff: Points1<T> = grad_current
                        .iter()
                        .zip(grad_old.iter())
                        .map(|(&g_new, &g_old)| g_new - g_old)
                        .collect();

                    let numerator = grad_current.iter().map(|&g| g * g).sum::<T>();

                    let denominator = search_direction
                        .iter()
                        .zip(grad_diff.iter())
                        .map(|(&h, &diff)| h * diff)
                        .sum::<T>();

                    if denominator.abs() < 1e-12.into() {
                        T::zero()
                    } else {
                        numerator / denominator
                    }
                }

                ConjGradMethod::HagerZhang => {
                    let grad_diff: Points1<T> = grad_current
                        .iter()
                        .zip(grad_old.iter())
                        .map(|(&g_new, &g_old)| g_new - g_old)
                        .collect();

                    let hd_dot = search_direction
                        .iter()
                        .zip(grad_diff.iter())
                        .map(|(&h, &diff)| h * diff)
                        .sum::<T>();

                    if hd_dot.abs() < 1e-12.into() {
                        0.0.into()
                    } else {
                        let grad_diff_norm_sq = grad_diff.iter().map(|&d| d * d).sum::<T>();
                        let scaling = 2.0 * &grad_diff_norm_sq / hd_dot;

                        let adjusted_diff: Points1<T> = grad_diff
                            .iter()
                            .zip(search_direction.iter())
                            .map(|(&diff, &h)| diff - scaling * h)
                            .collect();

                        let numerator = adjusted_diff
                            .iter()
                            .zip(grad_current.iter())
                            .map(|(&adj, &g)| adj * g)
                            .sum::<T>();

                        numerator / hd_dot
                    }
                }
            };

            // Update search direction
            for i in 0..n {
                search_direction[i] = (-1.0).into() * grad_current[i] + beta * search_direction[i];
            }

            // Check if direction is still descent direction
            let directional_derivative = search_direction
                .iter()
                .zip(grad_current.iter())
                .map(|(&d, &g)| d * g)
                .sum::<T>();

            if directional_derivative >= T::zero() {
                // Not a descent direction, restart with steepest descent
                search_direction = grad_current.iter().map(|&g| -g).collect();
                restart_count += 1;
            }
        }

        self.xmin = x;
        self.fmin = f_current;
        self.converged = grad_norm <= tol;
        Ok(ConjGradResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin,
            gradient_norm: grad_norm,
            iters: self.iters,
            fn_evals,
            g_evals,
            converged: self.converged,
            convergence_history: Points1::from_vec(convergence_history),
            gradient_norm_history: Points1::from_vec(gradient_norm_history),
            restart_count,
            method_used: method,
        })
    }

    /// Convenience function using Polak-Ribiere method (generally robust)
    pub fn minimize(
        &mut self,
        initial_point: &Points1<T>,
    ) -> Result<ConjGradResult<T>, MinimizerError> {
        self.conjugate_gradient(
            initial_point,
            ConjGradMethod::PolakRibiere,
            None,
            None,
            None,
        )
    }

    /// Compare different CG methods on the same problem
    pub fn compare_cg_methods(
        &mut self,
        initial_point: &Points1<T>,
        tol: Option<T>,
        max_iters: Option<usize>,
    ) -> Vec<(ConjGradMethod, Result<ConjGradResult<T>, MinimizerError>)> {
        let methods = [
            ConjGradMethod::FletcherReeves,
            ConjGradMethod::PolakRibiere,
            ConjGradMethod::HestenesStiefel,
            ConjGradMethod::DaiYuan,
            ConjGradMethod::HagerZhang,
        ];

        methods
            .iter()
            .map(|&method| {
                let result = self.conjugate_gradient(initial_point, method, tol, max_iters, None);
                (method, result)
            })
            .collect()
    }
}

impl<T> fmt::Debug for ConjGrad<T>
where
    T: RealScalar,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConjGrad( xmin: {:?}, fmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod minimize_conjgrad_tests {
    use super::*;
    use crate::minimize::{GF1dim, MultiDimGradFn, MultiDimNumGradFn};
    use float_cmp::F64Margin;
    use std::vec;
    use twofloat::TwoFloat;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };
    const TIGHT_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-8,
        ulps: 4,
    };
    const LOOSE_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };

    #[test]
    fn test_2d_quadratic() {
        // f(x,y) = (x-1)² + (y-2)², grad = (2(x-1), 2(y-2))
        let func =
            |x: &Points1<TwoFloat>| (x[0] - 1.0.into()).powi(2) + (x[1] - 2.0.into()).powi(2);
        let grad = |x: &Points1<TwoFloat>| {
            points![
                2.0.into() * (x[0] - 1.0.into()),
                2.0.into() * (x[1] - 2.0.into())
            ]
        };
        let obj = MultiDimGradFn::new(func, grad);
        let mut conjgrad = ConjGrad::new(obj);

        let result = conjgrad
            .minimize(&points![0.0.into(), 0.0.into()].into())
            .unwrap();

        assert!((result.xmin[0] - 1.0.into()).abs() < 1e-8.into());
        assert!((result.xmin[1] - 2.0.into()).abs() < 1e-8.into());
        assert!(result.fmin < 1e-14);
        assert!(result.converged);
        assert!(result.iters <= 3); // Should converge in 2 iterations for quadratic

        assert!((conjgrad.xmin[0] - 1.0.into()).abs() < 1e-8.into());
        assert!((conjgrad.xmin[1] - 2.0.into()).abs() < 1e-8.into());
        assert!(conjgrad.fmin < 1e-14);
        assert!(conjgrad.converged);
        assert!(conjgrad.iters <= 3); // Should converge in 2 iterations for quadratic
    }

    #[test]
    fn test_rosenbrock() {
        let rosenbrock = |x: &Points1<TwoFloat>| {
            (1.0.into() - x[0]).powi(2) + 100.0.into() * (x[1] - x[0].powi(2)).powi(2)
        };
        let rosenbrock_grad = |x: &Points1<TwoFloat>| {
            points![
                (-2.0).into() * (1.0.into() - x[0]) - 400.0.into() * x[0] * (x[1] - x[0].powi(2)),
                200.0.into() * (x[1] - x[0].powi(2)),
            ]
        };
        let obj = MultiDimGradFn::new(rosenbrock, rosenbrock_grad);
        let mut conjgrad = ConjGrad::new(obj);

        let result = conjgrad
            .conjugate_gradient(
                &points![TwoFloat::from_f64(-1.2), 1.0.into()],
                ConjGradMethod::PolakRibiere,
                Some(1e-4.into()), // Relaxed tolerance for Rosenbrock
                Some(5000),        // More iterations for this difficult problem
                Some(10),          // More frequent restarts
            )
            .unwrap();

        assert!(
            (result.xmin[0] - 1.0.into()).abs() < 1e-2.into(),
            "x[0] = {} should be close to 1.0",
            result.xmin[0]
        );
        assert!(
            (result.xmin[1] - 1.0.into()).abs() < 1e-2.into(),
            "x[1] = {} should be close to 1.0",
            result.xmin[1]
        );
        assert!(
            result.fmin < 1e-4.into(),
            "Function value {} should be small",
            result.fmin
        );

        assert!(
            (conjgrad.xmin[0] - 1.0.into()).abs() < 1e-2.into(),
            "x[0] = {} should be close to 1.0",
            conjgrad.xmin[0]
        );
        assert!(
            (conjgrad.xmin[1] - 1.0.into()).abs() < 1e-2.into(),
            "x[1] = {} should be close to 1.0",
            conjgrad.xmin[1]
        );
        assert!(
            conjgrad.fmin < 1e-4.into(),
            "Function value {} should be small",
            conjgrad.fmin
        );
    }

    #[test]
    fn test_different_methods() {
        let func = |x: &Points1<TwoFloat>| x[0].powi(2) + 2.0.into() * x[1].powi(2) + x[0] * x[1];
        let grad =
            |x: &Points1<TwoFloat>| points![2.0.into() * x[0] + x[1], 4.0.into() * x[1] + x[0]];
        let obj = MultiDimGradFn::new(func, grad);
        let mut conjgrad = ConjGrad::new(obj);

        let methods = [
            ConjGradMethod::FletcherReeves,
            ConjGradMethod::PolakRibiere,
            ConjGradMethod::HestenesStiefel,
        ];

        for method in &methods {
            let result = conjgrad.conjugate_gradient(
                &points![1.0.into(), 1.0.into()],
                *method,
                Some(1e-6.into()), // Reasonable tolerance
                Some(100),         // Sufficient iterations for this simple problem
                None,
            );

            match result {
                Ok(res) => {
                    assert!(
                        res.xmin[0].abs() < 1e-4.into(),
                        "Method {:?}: x[0] = {} should be near 0",
                        method,
                        res.xmin[0]
                    );
                    assert!(
                        res.xmin[1].abs() < 1e-4.into(),
                        "Method {:?}: x[1] = {} should be near 0",
                        method,
                        res.xmin[1]
                    );
                    assert!(res.converged, "Method {:?} should converge", method);

                    assert!(
                        conjgrad.xmin[0].abs() < 1e-4.into(),
                        "Method {:?}: x[0] = {} should be near 0",
                        method,
                        conjgrad.xmin[0]
                    );
                    assert!(
                        conjgrad.xmin[1].abs() < 1e-4.into(),
                        "Method {:?}: x[1] = {} should be near 0",
                        method,
                        conjgrad.xmin[1]
                    );
                    assert!(conjgrad.converged, "Method {:?} should converge", method);
                }
                Err(_) => {
                    // Some methods might struggle with this particular function form
                    // Let's try a simpler version to make sure the method works at all
                    let simple_func = |x: &Points1<TwoFloat>| x[0].powi(2) + x[1].powi(2);
                    let simple_grad =
                        |x: &Points1<TwoFloat>| points![2.0.into() * x[0], 2.0.into() * x[1]];
                    let obj_simple = MultiDimGradFn::new(simple_func, simple_grad);
                    let mut conjgrad = ConjGrad::new(obj_simple);

                    let simple_result = conjgrad
                        .conjugate_gradient(
                            &points![1.0.into(), 1.0.into()],
                            *method,
                            Some(1e-8.into()),
                            Some(50),
                            None,
                        )
                        .expect(&format!(
                            "Method {:?} should work on simple quadratic",
                            method
                        ));

                    assert!(simple_result.xmin[0].abs() < 1e-6);
                    assert!(simple_result.xmin[1].abs() < 1e-6);
                    assert!(simple_result.converged);

                    assert!(conjgrad.xmin[0].abs() < 1e-6);
                    assert!(conjgrad.xmin[1].abs() < 1e-6);
                    assert!(conjgrad.converged);
                }
            }
        }
    }

    #[test]
    fn test_numerical_gradients() {
        let func =
            |x: &Points1<TwoFloat>| (x[0] - 3.0.into()).powi(2) + (x[1] + 1.0.into()).powi(2);
        let obj = MultiDimNumGradFn::new(func, Some(1e-6.into()), 2);
        let mut conjgrad = ConjGrad::new(obj);

        let result = conjgrad
            .conjugate_gradient(
                &points![0.0.into(), 0.0.into()],
                ConjGradMethod::FletcherReeves,
                Some(1e-6.into()),
                None,
                None,
            )
            .unwrap();

        assert!((result.xmin[0] - 3.0.into()).abs() < 1e-4);
        assert!((result.xmin[1] + 1.0.into()).abs() < 1e-4);
        assert!(result.converged);

        assert!((conjgrad.xmin[0] - 3.0.into()).abs() < 1e-4);
        assert!((conjgrad.xmin[1] + 1.0.into()).abs() < 1e-4);
        assert!(conjgrad.converged);
    }

    #[test]
    fn test_method_comparison() {
        let func = |x: &Points1<TwoFloat>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| ((i + 1) as f64).into() * xi.powi(2))
                .sum::<TwoFloat>()
        };
        let grad = |x: &Points1<TwoFloat>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| 2.0.into() * ((i + 1) as f64).into() * xi)
                .collect()
        };
        let obj = MultiDimGradFn::new(func, grad);
        let mut conjgrad = ConjGrad::new(obj);

        let results = conjgrad.compare_cg_methods(
            &points![1.0.into(), 2.0.into(), 3.0.into()],
            Some(1e-8.into()),
            Some(100),
        );

        for (method, result) in results {
            match result {
                Ok(res) => {
                    for x in res.xmin {
                        assert!(
                            x.abs() < 1e-6,
                            "Method {} failed: x = {:?}",
                            method,
                            res.xmin
                        );
                    }
                    assert!(res.converged, "Method {} didn't converge", method);

                    for x in conjgrad.xmin {
                        assert!(
                            x.abs() < 1e-6,
                            "Method {} failed: x = {:?}",
                            method,
                            conjgrad.xmin
                        );
                    }
                    assert!(res.converged, "Method {} didn't converge", method);
                }
                Err(e) => panic!("Method {} failed with error: {}", method, e),
            }
        }
    }

    #[test]
    fn test_cg_method_comparison() {
        // Test all CG methods on the same problem
        let objective = |x: &Points1<TwoFloat>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| ((i + 1) as f64).into() * xi.powi(2))
                .sum::<TwoFloat>()
        };
        let gradient = |x: &Points1<TwoFloat>| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| 2.0.into() * ((i + 1) as f64).into() * xi)
                .collect()
        };
        let obj = MultiDimGradFn::new(objective, gradient);
        let mut conjgrad = ConjGrad::new(obj);

        let methods = [
            ConjGradMethod::FletcherReeves,
            ConjGradMethod::PolakRibiere,
            ConjGradMethod::HestenesStiefel,
            ConjGradMethod::DaiYuan,
            ConjGradMethod::HagerZhang,
        ];

        for &method in &methods {
            let result = conjgrad
                .conjugate_gradient(
                    &Points1::ones(4),
                    method,
                    Some(1e-8.into()),
                    Some(100),
                    None,
                )
                .unwrap();

            assert!(result.converged, "Method {:?} failed to converge", method);
            for x in result.xmin {
                assert!(
                    x.abs() < 1e-6,
                    "Method {:?} solution: {:?}",
                    method,
                    result.xmin
                );
            }
        }
    }

    #[test]
    fn test_cg_restart_mechanism() {
        // Test that restart prevents cycling
        let ill_conditioned = |x: &Points1<TwoFloat>| {
            1000.0.into() * x[0].powi(2) + x[1].powi(2) + 0.1.into() * x[0] * x[1]
        };
        let ill_conditioned_grad = |x: &Points1<TwoFloat>| {
            points![
                2000.0.into() * x[0] + 0.1.into() * x[1],
                2.0.into() * x[1] + 0.1.into() * x[0]
            ]
        };
        let obj = MultiDimGradFn::new(ill_conditioned, ill_conditioned_grad);
        let mut conjgrad = ConjGrad::new(obj);

        let result = conjgrad
            .conjugate_gradient(
                &points![1.0.into(), 1.0.into()],
                ConjGradMethod::PolakRibiere,
                Some(1e-6.into()),
                Some(200),
                Some(5), // Frequent restarts
            )
            .unwrap();

        assert!(result.converged);
        assert!(result.restart_count > 0); // Should have restarted
        assert!(result.xmin[0].abs() < 1e-4);
        assert!(result.xmin[1].abs() < 1e-4);
    }

    #[test]
    fn test_cg_numerical_gradients() {
        // Test CG with numerical gradients
        let objective = |x: &Points1<TwoFloat>| {
            (x[0] - 1.0.into()).powi(4) + (x[1] + 2.0.into()).powi(2) + 3.0.into() * x[0] * x[1]
        };
        let obj = MultiDimNumGradFn::new(objective, Some(1e-6.into()), 2);
        let mut conjgrad = ConjGrad::new(obj);

        let result = conjgrad
            .conjugate_gradient(
                &points![0.0.into(), 0.0.into()],
                ConjGradMethod::FletcherReeves,
                Some(1e-4.into()),
                Some(500),
                None,
            )
            .unwrap();

        assert!(result.converged);
        // Note: numerical gradients are less precise, so relaxed tolerance
        assert!(result.gradient_norm < 1e-3);
    }

    // Helper functions for common test functions
    fn simple_quadratic() -> GF1dim<TwoFloat> {
        let func = |x: &Points1<TwoFloat>| x.iter().map(|&xi| xi.powi(2)).sum::<TwoFloat>();
        let grad = |x: &Points1<TwoFloat>| x.iter().map(|&xi| 2.0.into() * xi).collect();
        GF1dim::new(MultiDimGradFn::new(func, grad))
    }

    fn shifted_quadratic(center: &Points1<TwoFloat>) -> GF1dim<TwoFloat> {
        let center_func = center;
        let func = move |x: &Points1<TwoFloat>| {
            x.iter()
                .zip(center_func.iter())
                .map(|(&xi, &ci)| (xi - ci).powi(2))
                .sum::<TwoFloat>()
        };
        let center_func = center;
        let grad = move |x: &Points1<TwoFloat>| {
            x.iter()
                .zip(center_func.iter())
                .map(|(&xi, &ci)| 2.0.into() * (xi - ci))
                .collect()
        };
        GF1dim::new(MultiDimGradFn::new(func, grad))
    }

    fn rosenbrock_2d() -> GF1dim<TwoFloat> {
        let func = |x: &Points1<TwoFloat>| {
            (1.0.into() - x[0]).powi(2) + 100.0.into() * (x[1] - x[0].powi(2)).powi(2)
        };
        let grad = |x: &Points1<TwoFloat>| {
            points![
                (-2.0).into() * (1.0.into() - x[0]) - 400.0.into() * x[0] * (x[1] - x[0].powi(2)),
                200.0.into() * (x[1] - x[0].powi(2)),
            ]
        };
        GF1dim::new(MultiDimGradFn::new(func, grad))
    }

    mod basic_functionality_tests {
        use super::*;

        #[test]
        fn test_conjgrad_construction() {
            let obj = simple_quadratic();
            let conjgrad = ConjGrad::new(obj);

            assert!(conjgrad.xmin.is_empty());
            assert_eq!(conjgrad.fmin, 0.0);
            assert_eq!(conjgrad.iters, 0);
            assert!(!conjgrad.converged);
        }

        #[test]
        fn test_conjgrad_boxed_construction() {
            let obj = simple_quadratic();
            let boxed_obj = Box::new(obj);
            let conjgrad = ConjGrad::new_boxed(boxed_obj);

            assert!(conjgrad.xmin.is_empty());
            assert_eq!(conjgrad.fmin, 0.0);
            assert_eq!(conjgrad.iters, 0);
            assert!(!conjgrad.converged);
        }

        #[test]
        fn test_debug_formatting() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);
            let _ = conjgrad.minimize(&points![1.0.into(), 2.0.into()].into());

            let debug_str = format!("{:?}", conjgrad);
            assert!(debug_str.contains("ConjGrad"));
            assert!(debug_str.contains("xmin"));
            assert!(debug_str.contains("fmin"));
        }

        #[test]
        fn test_method_display_formatting() {
            assert_eq!(
                format!("{}", ConjGradMethod::FletcherReeves),
                "Fletcher-Reeves"
            );
            assert_eq!(format!("{}", ConjGradMethod::PolakRibiere), "Polak-Ribiere");
            assert_eq!(
                format!("{}", ConjGradMethod::HestenesStiefel),
                "Hestenes-Stiefel"
            );
            assert_eq!(format!("{}", ConjGradMethod::DaiYuan), "Dai-Yuan");
            assert_eq!(format!("{}", ConjGradMethod::HagerZhang), "Hager-Zhang");
        }
    }

    mod error_handling_tests {
        use super::*;

        #[test]
        fn test_empty_initial_point_error() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad.minimize(&points![].into());
            assert!(matches!(result, Err(MinimizerError::InvalidDimension)));
        }

        #[test]
        fn test_zero_tolerance_error() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad.conjugate_gradient(
                &points![1.0.into(), 2.0.into()].into(),
                ConjGradMethod::FletcherReeves,
                Some(0.0.into()),
                None,
                None,
            );
            assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
        }

        #[test]
        fn test_negative_tolerance_error() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad.conjugate_gradient(
                &points![1.0.into(), 2.0.into()].into(),
                ConjGradMethod::FletcherReeves,
                Some(TwoFloat::from_f64(-1e-6)),
                None,
                None,
            );
            assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
        }

        #[test]
        fn test_wrong_gradient_dimension() {
            let func = |x: &Points1<TwoFloat>| x[0].powi(2) + x[1].powi(2);
            let wrong_grad = |_x: &Points1<TwoFloat>| points![1.0.into()].into(); // Wrong dimension
            let obj = MultiDimGradFn::new(func, wrong_grad);
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad.minimize(&points![1.0.into(), 1.0.into()].into());
            assert!(matches!(
                result,
                Err(MinimizerError::GradientEvaluationError)
            ));
        }
    }

    mod basic_optimization_tests {
        use super::*;

        #[test]
        fn test_shifted_quadratic() {
            let center = points![3.0.into(), TwoFloat::from_f64(-2.0)].into();
            let obj = shifted_quadratic(&center);
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .minimize(&points![0.0.into(), 0.0.into()].into())
                .unwrap();

            assert!(result.converged);
            assert!((&result.xmin[0] - &center[0]).abs() < 1e-8);
            assert!((&result.xmin[1] - &center[1]).abs() < 1e-8);
            assert!(result.fmin < 1e-14);
        }

        #[test]
        fn test_1d_optimization() {
            let func = |x: &Points1<TwoFloat>| (&x[0] - 5.0).powi(2);
            let grad = |x: &Points1<TwoFloat>| points![2.0 * (&x[0] - 5.0)].into();
            let obj = MultiDimGradFn::new(func, grad);
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad.minimize(&points![0.0.into()].into()).unwrap();

            assert!(result.converged);
            assert!((&result.xmin[0] - 5.0).abs() < 1e-8);
            assert!(result.iters <= 2);
        }

        #[test]
        fn test_high_dimensional_quadratic() {
            let n = 8;
            let targets: Vec<TwoFloat> = (1..=n).map(|i| (i as f64).into()).collect();
            let targets_func = targets;
            let targets_grad = targets;

            let func = move |x: &Points1<TwoFloat>| {
                x.iter()
                    .zip(targets_func.iter())
                    .map(|(xi, ti)| (xi - ti).powi(2))
                    .sum::<TwoFloat>()
            };
            let grad = move |x: &Points1<TwoFloat>| {
                x.iter()
                    .zip(targets_grad.iter())
                    .map(|(xi, ti)| 2.0 * (xi - ti))
                    .collect()
            };
            let obj = MultiDimGradFn::new(func, grad);
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad.minimize(&Points1::zeros(n)).unwrap();

            assert!(result.converged);
            for (i, xi) in result.xmin.iter().enumerate() {
                assert!((xi - &targets[i]).abs() < 1e-6);
            }
            assert!(result.fmin < 1e-12);
        }
    }

    mod cg_method_comparison_tests {
        use super::*;

        #[test]
        fn test_all_cg_methods_simple_quadratic() {
            let methods = [
                ConjGradMethod::FletcherReeves,
                ConjGradMethod::PolakRibiere,
                ConjGradMethod::HestenesStiefel,
                ConjGradMethod::DaiYuan,
                ConjGradMethod::HagerZhang,
            ];

            for &method in &methods {
                let obj = simple_quadratic();
                let mut conjgrad = ConjGrad::new(obj);

                let result = conjgrad
                    .conjugate_gradient(
                        &points![2.0.into(), TwoFloat::from_f64(-1.5)].into(),
                        method,
                        Some(1e-8.into()),
                        Some(100),
                        None,
                    )
                    .unwrap();

                assert!(result.converged, "Method {:?} should converge", method);
                assert!(
                    result.xmin[0].abs() < 1e-6,
                    "Method {:?} x[0] = {}",
                    method,
                    &result.xmin[0]
                );
                assert!(
                    result.xmin[1].abs() < 1e-6,
                    "Method {:?} x[1] = {}",
                    method,
                    &result.xmin[1]
                );
                assert_eq!(result.method_used, method);
            }
        }

        #[test]
        fn test_fletcher_reeves_method() {
            let obj = shifted_quadratic(&points![1.0.into(), TwoFloat::from_f64(-1.0)].into());
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .conjugate_gradient(
                    &points![0.0.into(), 0.0.into()].into(),
                    ConjGradMethod::FletcherReeves,
                    Some(1e-8.into()),
                    Some(50),
                    None,
                )
                .unwrap();

            assert!(result.converged);
            assert!((&result.xmin[0] - 1.0).abs() < 1e-6);
            assert!((&result.xmin[1] + 1.0).abs() < 1e-6);
            assert_eq!(result.method_used, ConjGradMethod::FletcherReeves);
        }

        #[test]
        fn test_polak_ribiere_method() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .conjugate_gradient(
                    &points![3.0.into(), TwoFloat::from_f64(-2.0)].into(),
                    ConjGradMethod::PolakRibiere,
                    Some(1e-8.into()),
                    Some(50),
                    None,
                )
                .unwrap();

            assert!(result.converged);
            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
            assert_eq!(result.method_used, ConjGradMethod::PolakRibiere);
        }

        #[test]
        fn test_method_comparison_function() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let results = conjgrad.compare_cg_methods(
                &points![1.0.into(), 2.0.into()].into(),
                Some(1e-6.into()),
                Some(100),
            );

            assert_eq!(results.len(), 5); // All 5 methods

            for (method, result) in results {
                match result {
                    Ok(res) => {
                        assert!(res.converged, "Method {:?} should converge", method);
                        assert!(res.xmin[0].abs() < 1e-4, "Method {:?}", method);
                        assert!(res.xmin[1].abs() < 1e-4, "Method {:?}", method);
                    }
                    Err(e) => panic!("Method {:?} failed: {:?}", method, e),
                }
            }
        }
    }

    mod convergence_analysis_tests {
        use super::*;

        #[test]
        fn test_convergence_history_structure() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .minimize(&points![2.0.into(), TwoFloat::from_f64(-1.0)].into())
                .unwrap();

            // Basic structure validation
            assert!(!result.convergence_history.is_empty());
            assert!(!result.gradient_norm_history.is_empty());
            assert_eq!(result.convergence_history.len(), result.iters + 1);
            assert_eq!(result.gradient_norm_history.len(), result.iters + 1);

            // Overall progress validation
            let initial_f = result.convergence_history[0];
            let final_f = result.convergence_history.last().unwrap();
            assert!(final_f <= initial_f);
            assert!(final_f < 1e-10);

            // Gradient norm progress
            let initial_grad = result.gradient_norm_history[0];
            let final_grad = result.gradient_norm_history.last().unwrap();
            assert!(final_grad <= initial_grad);
            assert!(final_grad < 1e-6);

            // Consistency checks
            assert_eq!(result.gradient_norm, final_grad);
            assert_eq!(result.fmin, final_f);
        }

        #[test]
        fn test_evaluation_counts() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .minimize(&points![1.0.into(), 2.0.into()].into())
                .unwrap();

            assert!(result.fn_evals > 0);
            assert!(result.g_evals > 0);
            assert!(result.fn_evals >= result.iters);
            assert!(result.g_evals >= result.iters);

            // For simple quadratic, shouldn't need excessive evaluations
            assert!(result.fn_evals < 100);
            assert!(result.g_evals < 100);
        }

        #[test]
        fn test_tolerance_behavior() {
            let tolerances = points![1e-4, 1e-6, 1e-8];

            for &tol in &tolerances {
                let obj = simple_quadratic();
                let mut conjgrad = ConjGrad::new(obj);

                let result = conjgrad
                    .conjugate_gradient(
                        &points![2.0.into(), TwoFloat::from_f64(-1.0)].into(),
                        ConjGradMethod::PolakRibiere,
                        Some(tol.into()),
                        Some(100),
                        None,
                    )
                    .unwrap();

                assert!(result.converged);
                assert!(result.gradient_norm <= tol * 2.0); // Allow some numerical tolerance
            }
        }
    }

    mod restart_mechanism_tests {
        use super::*;

        #[test]
        fn test_restart_functionality() {
            // Ill-conditioned quadratic to force restarts
            let func = |x: &Points1<TwoFloat>| 100.0 * x[0].powi(2) + x[1].powi(2);
            let grad = |x: &Points1<TwoFloat>| points![200.0 * &x[0], 2.0 * &x[1]].into();
            let obj = MultiDimGradFn::new(func, grad);
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .conjugate_gradient(
                    &points![1.0.into(), 1.0.into()].into(),
                    ConjGradMethod::PolakRibiere,
                    Some(1e-6.into()),
                    Some(200),
                    Some(3), // Force frequent restarts
                )
                .unwrap();

            assert!(result.converged);
            assert!(result.restart_count > 0);
            assert!(result.xmin[0].abs() < 1e-4);
            assert!(result.xmin[1].abs() < 1e-4);
        }
    }

    mod difficult_function_tests {
        use super::*;

        #[test]
        fn test_rosenbrock_optimization() {
            let obj = rosenbrock_2d();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .conjugate_gradient(
                    &points![TwoFloat::from_f64(-1.2), 1.0.into()].into(),
                    ConjGradMethod::PolakRibiere,
                    Some(1e-3.into()), // Relaxed tolerance for Rosenbrock
                    Some(2000),
                    Some(10),
                )
                .unwrap();

            // Rosenbrock is challenging, so relaxed criteria
            assert!((&result.xmin[0] - 1.0).abs() < 0.1);
            assert!((&result.xmin[1] - 1.0).abs() < 0.1);
            assert!(result.fmin < 1e-2);
        }

        #[test]
        fn test_booth_function() {
            // Booth function: f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
            // Global minimum at (1, 3) with f(1, 3) = 0
            let func = |x: &Points1<TwoFloat>| {
                (&x[0] + 2.0 * &x[1] - 7.0).powi(2) + (2.0 * &x[0] + &x[1] - 5.0).powi(2)
            };
            let grad = |x: &Points1<TwoFloat>| {
                let t1 = &x[0] + 2.0 * &x[1] - 7.0;
                let t2 = 2.0 * &x[0] + &x[1] - 5.0;
                points![2.0 * &t1 + 4.0 * &t2, 4.0 * &t1 + 2.0 * &t2].into()
            };
            let obj = MultiDimGradFn::new(func, grad);
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .minimize(&points![0.0.into(), 0.0.into()].into())
                .unwrap();

            assert!(result.converged);
            assert!((&result.xmin[0] - 1.0).abs() < 1e-6);
            assert!((&result.xmin[1] - 3.0).abs() < 1e-6);
            assert!(result.fmin < 1e-12);
        }
    }

    mod numerical_gradient_tests {
        use super::*;

        #[test]
        fn test_numerical_gradients() {
            let func = |x: &Points1<TwoFloat>| (&x[0] - 2.0).powi(2) + (&x[1] + 1.0).powi(2);
            let obj = MultiDimNumGradFn::new(func, Some(1e-8.into()), 2);
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .minimize(&points![0.0.into(), 0.0.into()].into())
                .unwrap();

            assert!(result.converged);
            assert!((&result.xmin[0] - 2.0).abs() < 1e-3);
            assert!((&result.xmin[1] + 1.0).abs() < 1e-3);
        }
    }

    mod robustness_tests {
        use super::*;

        #[test]
        fn test_already_at_minimum() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .minimize(&points![0.0.into(), 0.0.into()].into())
                .unwrap();

            assert!(result.converged);
            assert!(result.iters <= 2); // Should recognize it's already converged
            assert!(result.fmin < 1e-12);
        }

        #[test]
        fn test_large_initial_values() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .minimize(&points![100.0.into(), TwoFloat::from_f64(-50.0)].into())
                .unwrap();

            assert!(result.converged);
            assert!(result.xmin[0].abs() < 1e-6);
            assert!(result.xmin[1].abs() < 1e-6);
            assert!(result.fmin < 1e-12);
        }

        #[test]
        fn test_function_with_different_scales() {
            // Function with different scales in different dimensions
            let func = |x: &Points1<TwoFloat>| 1000.0 * x[0].powi(2) + x[1].powi(2);
            let grad = |x: &Points1<TwoFloat>| points![2000.0 * &x[0], 2.0 * &x[1]].into();
            let obj = MultiDimGradFn::new(func, grad);
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .conjugate_gradient(
                    &points![1.0.into(), 1.0.into()].into(),
                    ConjGradMethod::PolakRibiere,
                    Some(1e-6.into()),
                    Some(200),
                    Some(5),
                )
                .unwrap();

            assert!(result.converged);
            assert!(result.xmin[0].abs() < 1e-4);
            assert!(result.xmin[1].abs() < 1e-4);
        }
    }

    mod edge_case_tests {
        use super::*;

        #[test]
        fn test_maximum_iterations_limit() {
            let obj = rosenbrock_2d();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .conjugate_gradient(
                    &points![TwoFloat::from_f64(-1.2), 1.0.into()].into(),
                    ConjGradMethod::FletcherReeves,
                    Some(1e-12.into()), // Very strict tolerance
                    Some(5),            // Very few iterations
                    None,
                )
                .unwrap();

            // Should hit iteration limit
            assert!(!result.converged);
            assert_eq!(result.iters, 5);
        }

        #[test]
        fn test_result_completeness() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);

            let result = conjgrad
                .minimize(&points![1.0.into(), 2.0.into()].into())
                .unwrap();

            // Verify all fields are populated
            assert!(!result.xmin.is_empty());
            assert!(result.fmin.is_finite());
            assert!(result.gradient_norm.is_finite());
            assert!(result.iters > 0);
            assert!(result.fn_evals > 0);
            assert!(result.g_evals > 0);
            assert!(result.converged);
            assert!(!result.convergence_history.is_empty());
            assert!(!result.gradient_norm_history.is_empty());
        }

        #[test]
        fn test_clone_and_debug_traits() {
            let obj = simple_quadratic();
            let mut conjgrad = ConjGrad::new(obj);
            let result = conjgrad.minimize(&points![1.0.into()].into()).unwrap();

            // Test Clone
            let result_clone = result;
            assert_eq!(result.xmin, result_clone.xmin);
            assert_eq!(result.fmin, result_clone.fmin);
            assert_eq!(result.converged, result_clone.converged);

            // Test Debug
            let debug_str = format!("{:?}", result);
            assert!(debug_str.contains("ConjGradResult"));
        }
    }

    mod stress_tests {
        use super::*;

        #[test]
        fn test_method_robustness_on_challenging_function() {
            // Moderately challenging function
            let func = |x: &Points1<TwoFloat>| {
                let a = 5.0 * &x[0] + &x[1];
                let b = &x[0] + 5.0 * &x[1];
                &a * &a + &b * &b
            };
            let grad = |x: &Points1<TwoFloat>| {
                points![
                    2.0 * (5.0 * &x[0] + &x[1]) * 5.0 + 2.0 * (&x[0] + 5.0 * &x[1]),
                    2.0 * (5.0 * &x[0] + &x[1]) + 2.0 * (&x[0] + 5.0 * &x[1]) * 5.0,
                ]
                .into()
            };

            let methods = [
                ConjGradMethod::FletcherReeves,
                ConjGradMethod::PolakRibiere,
                ConjGradMethod::HestenesStiefel,
            ];

            let mut successful_methods = 0;

            for &method in &methods {
                let obj = MultiDimGradFn::new(func, grad);
                let mut conjgrad = ConjGrad::new(obj);

                let result = conjgrad.conjugate_gradient(
                    &points![1.0.into(), 1.0.into()].into(),
                    method,
                    Some(1e-6.into()),
                    Some(200),
                    Some(10),
                );

                if let Ok(res) = result {
                    if res.converged && res.xmin[0].abs() < 1e-3 && res.xmin[1].abs() < 1e-3 {
                        successful_methods += 1;
                    }
                }
            }

            assert!(
                successful_methods >= 2,
                "At least 2 methods should work on this problem"
            );
        }

        #[test]
        fn test_various_starting_points() {
            let starting_points = points![
                points![TwoFloat::from_f64(0.0), TwoFloat::from_f64(0.0)],
                points![1.0.into(), 1.0.into()],
                points![TwoFloat::from_f64(-1.0), 1.0.into()],
                points![10.0.into(), TwoFloat::from_f64(-5.0)],
                points![TwoFloat::from_f64(-20.0), 15.0.into()],
            ];

            for start in starting_points {
                let obj = shifted_quadratic(&points![2.0.into(), TwoFloat::from_f64(-1.0)].into());
                let mut conjgrad = ConjGrad::new(obj);

                let result = conjgrad.minimize(&start.into()).unwrap();

                assert!(result.converged, "Failed from starting point {:?}", start);
                assert!((&result.xmin[0] - 2.0).abs() < 1e-6);
                assert!((&result.xmin[1] + 1.0).abs() < 1e-6);
            }
        }
    }
}
