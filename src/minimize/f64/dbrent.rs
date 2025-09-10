#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::{MinimizerError, f64::ObjDerFn};
use std::fmt;

/// Result of Brent's method with derivatives
#[derive(Debug, Clone)]
pub struct DBrentResult {
    pub xmin: f64,
    pub fmin: f64,
    pub dfmin: f64,
    pub iters: usize,
    pub converged: bool,
    pub final_bracket_size: f64,
    pub method_used: Vec<String>, // Track which methods were used
}

#[derive(Clone)]
pub struct DBrent {
    xmin: f64,
    fmin: f64,
    dfmin: f64,
    f: Box<dyn ObjDerFn>,
    // pub bracket: BracketF64,
    iters: usize,
    converged: bool,
}

impl DBrent {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjDerFn + 'static,
    {
        DBrent {
            xmin: 0.0,
            fmin: 0.0,
            dfmin: 0.0,
            f: Box::new(f),
            // bracket: BracketF64::new(),
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjDerFn>) -> Self {
        DBrent {
            xmin: 0.0,
            fmin: 0.0,
            dfmin: 0.0,
            f: f,
            // bracket: BracketF64::new(),
            iters: 0,
            converged: false,
        }
    }

    /// Brent's method with derivatives for finding roots
    ///
    /// This algorithm combines Brent's robust bracketing approach with derivative
    /// information to achieve superlinear convergence. It uses:
    /// - Newton's method when safe and converging quickly
    /// - Inverse quadratic interpolation with derivatives
    /// - Standard Brent's method as fallback
    /// - Bisection for guaranteed convergence
    ///
    /// # Arguments
    /// * `f` - The function whose root we want to find
    /// * `df` - The derivative of the function
    /// * `a` - Left bracket boundary (f(a) and f(b) must have opposite signs)
    /// * `b` - Right bracket boundary
    /// * `tol` - Convergence tolerance (default: 1e-14)
    /// * `max_iters` - Maximum iterations (default: 100)
    ///
    /// # Returns
    /// * `DBrentResult` containing root, convergence info, and method tracking
    pub fn minimize(
        &mut self,
        mut a: f64,
        mut b: f64,
        tol: Option<f64>,
        max_iters: Option<usize>,
    ) -> Result<DBrentResult, MinimizerError> {
        self.converged = false;
        let tol = tol.unwrap_or(1e-14);
        let max_iter = max_iters.unwrap_or(100);

        // Validate inputs
        if a >= b {
            return Err(MinimizerError::InvalidBracket);
        }
        if tol <= 0.0 {
            return Err(MinimizerError::InvalidTolerance);
        }

        let mut eval_a = (a, self.f.call_scalar(a), self.f.df_scalar(a));
        let mut eval_b = (b, self.f.call_scalar(b), self.f.df_scalar(b));

        // Check that f(a) and f(b) have opposite signs
        if eval_a.1 * eval_b.1 > 0.0 {
            return Err(MinimizerError::SameSignError);
        }

        // Ensure |f(a)| >= |f(b)|
        if eval_a.1.abs() < eval_b.1.abs() {
            std::mem::swap(&mut eval_a, &mut eval_b);
            std::mem::swap(&mut a, &mut b);
        }

        let mut eval_c = eval_a;
        let mut mflag = true;
        let mut eval_d = (0.0, 0.0, 0.0);
        let mut method_used = Vec::new();

        self.iters = 0;

        while eval_b.1.abs() > tol && (b - a).abs() > tol && self.iters < max_iter {
            self.iters += 1;

            let mut s = b;
            let mut method = "bisection";

            // Try Newton's method first if derivative is significant
            if eval_b.2.abs() > tol && eval_b.2.abs() > 1e-8 {
                let newton_step = eval_b.0 - eval_b.1 / eval_b.2;

                // Check if Newton step is within bracket and reasonable
                let newton_in_bracket = newton_step > a && newton_step < b;
                let newton_reasonable = (newton_step - eval_b.0).abs() < (b - a).abs();

                if newton_in_bracket && newton_reasonable {
                    s = newton_step;
                    method = "newton";
                }
            }

            // If Newton wasn't used or suitable, try other interpolation methods
            if method == "bisection" {
                // Try inverse quadratic interpolation with derivatives
                if eval_a.1 != eval_c.1 && eval_b.1 != eval_c.1 && eval_a.1 != eval_b.1 {
                    // Hermite interpolation using function values and derivatives
                    let h1 = eval_b.0 - eval_a.0;
                    let h2 = eval_c.0 - eval_b.0;

                    if h1.abs() > tol && h2.abs() > tol {
                        // Use modified inverse quadratic interpolation with derivative info
                        let delta1 = (eval_b.1 - eval_a.1) / h1;
                        let delta2 = (eval_c.1 - eval_b.1) / h2;

                        if (delta1 - eval_a.2).abs() < 1.0 && (delta2 - eval_b.2).abs() < 1.0 {
                            let denom = (eval_a.1 - eval_b.1)
                                * (eval_a.1 - eval_c.1)
                                * (eval_b.1 - eval_c.1);

                            if denom.abs() > tol {
                                s = eval_a.0 * eval_b.1 * eval_c.1
                                    / ((eval_a.1 - eval_b.1) * (eval_a.1 - eval_c.1))
                                    + eval_b.0 * eval_a.1 * eval_c.1
                                        / ((eval_b.1 - eval_a.1) * (eval_b.1 - eval_c.1))
                                    + eval_c.0 * eval_a.1 * eval_b.1
                                        / ((eval_c.1 - eval_a.1) * (eval_c.1 - eval_b.1));
                                method = "inverse_quadratic_with_derivative";
                            }
                        }
                    }
                }

                // Fallback to standard inverse quadratic interpolation
                if method == "bisection" && eval_a.1 != eval_c.1 && eval_b.1 != eval_c.1 {
                    let denom =
                        (eval_a.1 - eval_b.1) * (eval_a.1 - eval_c.1) * (eval_b.1 - eval_c.1);
                    if denom.abs() > tol {
                        s = eval_a.0 * eval_b.1 * eval_c.1
                            / ((eval_a.1 - eval_b.1) * (eval_a.1 - eval_c.1))
                            + eval_b.0 * eval_a.1 * eval_c.1
                                / ((eval_b.1 - eval_a.1) * (eval_b.1 - eval_c.1))
                            + eval_c.0 * eval_a.1 * eval_b.1
                                / ((eval_c.1 - eval_a.1) * (eval_c.1 - eval_b.1));
                        method = "inverse_quadratic";
                    }
                }

                // Fallback to secant method
                if method == "bisection" && eval_a.1 != eval_b.1 {
                    s = eval_b.0 - eval_b.1 * (eval_b.0 - eval_a.0) / (eval_b.1 - eval_a.1);
                    method = "secant";
                }
            }

            // Safety checks - use bisection if interpolated point is not acceptable
            let condition1 = !(((3.0 * a + b) / 4.0)..=b).contains(&s);
            let condition2 = mflag && (s - b).abs() >= (b - eval_c.0).abs() / 2.0;
            let condition3 = !mflag && (s - b).abs() >= (eval_c.0 - eval_d.0).abs() / 2.0;
            let condition4 = mflag && (b - eval_c.0).abs() < tol;
            let condition5 = !mflag && (eval_c.0 - eval_d.0).abs() < tol;

            if condition1 || condition2 || condition3 || condition4 || condition5 {
                s = (a + b) / 2.0;
                method = "bisection";
                mflag = true;
            } else {
                mflag = false;
            }

            let eval_s = (s, self.f.call_scalar(s), self.f.df_scalar(s));
            method_used.push(method.to_string());

            // Update for next iteration
            eval_d = eval_c;
            eval_c = eval_b;

            if eval_a.1 * eval_s.1 < 0.0 {
                b = s;
                eval_b = eval_s;
            } else {
                a = s;
                eval_a = eval_s;
            }

            // Ensure |f(a)| >= |f(b)|
            if eval_a.1.abs() < eval_b.1.abs() {
                std::mem::swap(&mut eval_a, &mut eval_b);
                std::mem::swap(&mut a, &mut b);
            }
        }

        if self.iters >= max_iter {
            return Err(MinimizerError::MaxIterationsExceeded);
        }

        (self.xmin, self.fmin, self.dfmin) = eval_b;
        self.converged = true;
        Ok(DBrentResult {
            xmin: self.xmin,
            fmin: self.fmin,
            dfmin: self.dfmin,
            iters: self.iters,
            converged: self.converged,
            final_bracket_size: (b - a).abs(),
            method_used,
        })
    }

    /// Convenience function with default parameters
    pub fn find_root_with_derivative(
        &mut self,
        a: f64,
        b: f64,
    ) -> Result<DBrentResult, MinimizerError> {
        self.minimize(a, b, None, None)
    }

    /// Newton-Raphson method with safeguards (fallback to bisection)
    ///
    /// Pure Newton's method with bracket constraints for robustness
    pub fn newton_with_bracket(
        &mut self,
        mut x0: f64,
        a: f64,
        b: f64,
        tol: Option<f64>,
        max_iters: Option<usize>,
    ) -> Result<DBrentResult, MinimizerError> {
        self.converged = false;
        let tol = tol.unwrap_or(1e-14);
        let max_iter = max_iters.unwrap_or(100);

        if a >= b {
            return Err(MinimizerError::InvalidBracket);
        }
        if tol <= 0.0 {
            return Err(MinimizerError::InvalidTolerance);
        }

        // Ensure starting point is in bracket
        x0 = x0.clamp(a, b);

        let mut method_used = Vec::new();
        self.iters = 0;

        let mut bracket_a = a;
        let mut bracket_b = b;
        let mut fa = self.f.call_scalar(bracket_a);
        let mut fb = self.f.call_scalar(bracket_b);

        // Ensure proper bracket
        if fa * fb > 0.0 {
            return Err(MinimizerError::SameSignError);
        }

        let mut x = x0;

        while self.iters < max_iter {
            self.iters += 1;

            let fx = self.f.call_scalar(x);
            if fx.abs() < tol {
                method_used.push("converged".to_string());
                break;
            }

            let dfx = self.f.df_scalar(x);

            if dfx.abs() < tol {
                // Use bisection when derivative is too small
                x = (bracket_a + bracket_b) / 2.0;
                method_used.push("bisection".to_string());
            } else {
                // Try Newton step
                let x_newton = x - fx / dfx;

                if x_newton > bracket_a && x_newton < bracket_b {
                    x = x_newton;
                    method_used.push("newton".to_string());
                } else {
                    // Newton step outside bracket, use bisection
                    x = (bracket_a + bracket_b) / 2.0;
                    method_used.push("bisection".to_string());
                }
            }

            // Update bracket
            let fx_new = self.f.call_scalar(x);
            if fa * fx_new < 0.0 {
                bracket_b = x;
                fb = fx_new;
            } else {
                bracket_a = x;
                fa = fx_new;
            }
        }

        if self.iters >= max_iter {
            return Err(MinimizerError::MaxIterationsExceeded);
        }

        self.xmin = x;
        self.fmin = self.f.call_scalar(x);
        self.dfmin = self.f.df_scalar(x);
        self.converged = true;
        Ok(DBrentResult {
            xmin: self.xmin,
            fmin: self.fmin,
            dfmin: self.dfmin,
            iters: self.iters,
            converged: self.converged,
            final_bracket_size: (bracket_b - bracket_a).abs(),
            method_used,
        })
    }

    pub fn xmin(&self) -> f64 {
        self.xmin
    }

    pub fn fmin(&self) -> f64 {
        self.fmin
    }

    pub fn dfmin(&self) -> f64 {
        self.dfmin
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn set_xmin(&mut self, xmin: f64) {
        self.xmin = xmin;
        self.fmin = self.f.call_scalar(xmin);
    }
}

impl fmt::Debug for DBrent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DBrent( xmin: {}, fmin: {}, dfmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.dfmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod dbrentf64_tests {
    use super::*;
    use crate::minimize::f64::SingleDimDerFn;
    use float_cmp::F64Margin;
    use std::f64::consts::PI;

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };
    const DEFAULT_TOL: f64 = 1e-6;
    const DEFAULT_TOL_BRENT: f64 = 1e-10;
    const DEFAULT_MAX_ITER: usize = 100;

    #[test]
    fn test_quadratic_with_derivative() {
        // f(x) = x^2 - 2, f'(x) = 2x, xmin at x = √2
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let objective = SingleDimDerFn::new(f, df);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.find_root_with_derivative(1.0, 2.0).unwrap();

        assert!((result.xmin - 2_f64.sqrt()).abs() < 1e-12);
        assert!(result.fmin.abs() < 1e-12);
        assert!(result.converged);
        assert!((dbrent.xmin - 2_f64.sqrt()).abs() < 1e-12);
        assert!(dbrent.fmin.abs() < 1e-12);
        assert!(dbrent.converged);
        // Should converge faster than standard Brent's method
        assert!(result.iters <= 10);
        assert!(dbrent.iters <= 10);
    }

    #[test]
    fn test_cubic_with_derivative() {
        // f(x) = x^3 - 2x - 5, f'(x) = 3x^2 - 2
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let df = |x: f64| 3.0 * x * x - 2.0;
        let objective = SingleDimDerFn::new(f, df);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.find_root_with_derivative(2.0, 3.0).unwrap();

        assert!(result.fmin.abs() < 1e-12);
        assert!(result.converged);
        assert!(dbrent.fmin.abs() < 1e-12);
        assert!(dbrent.converged);
        // Verify the xmin
        let verification = f(result.xmin);
        assert!(verification.abs() < 1e-10);
        let verification = f(dbrent.xmin);
        assert!(verification.abs() < 1e-10);
    }

    #[test]
    fn test_transcendental_with_derivative() {
        // f(x) = e^x - 2x - 1, f'(x) = e^x - 2
        let f = |x: f64| x.exp() - 2.0 * x - 1.0;
        let df = |x: f64| x.exp() - 2.0;
        let objective = SingleDimDerFn::new(f, df);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.find_root_with_derivative(0.0, 2.0).unwrap();

        assert!(result.fmin.abs() < 1e-12);
        assert!(result.converged);
        assert!(dbrent.fmin.abs() < 1e-12);
        assert!(dbrent.converged);
    }

    #[test]
    fn test_newton_with_bracket() {
        // f(x) = x^3 - x - 1, f'(x) = 3x^2 - 1
        let f = |x: f64| x.powi(3) - x - 1.0;
        let df = |x: f64| 3.0 * x * x - 1.0;
        let objective = SingleDimDerFn::new(f, df);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent
            .newton_with_bracket(1.5, 1.0, 2.0, None, None)
            .unwrap();

        assert!(result.fmin.abs() < 1e-12);
        assert!(result.converged);
        // Newton should be primary method used
        assert!(result.method_used.iter().any(|m| m == "newton"));
        assert!(dbrent.fmin.abs() < 1e-12);
        assert!(dbrent.converged);
    }

    #[test]
    fn test_method_tracking() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 2.0;
        let df = |x: f64| 3.0 * x * x - 2.0;
        let objective = SingleDimDerFn::new(f, df);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.find_root_with_derivative(1.0, 2.0).unwrap();

        // Should have used multiple methods
        assert!(!result.method_used.is_empty());
        assert!(result.converged);
        assert!(dbrent.converged);
    }

    #[test]
    fn test_same_sign_error() {
        let f = |x: f64| x * x + 1.0; // Always positive
        let df = |x: f64| 2.0 * x;
        let objective = SingleDimDerFn::new(f, df);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.find_root_with_derivative(0.0, 1.0);

        assert!(matches!(result, Err(MinimizerError::SameSignError)));
        assert!(!dbrent.converged);
    }

    #[test]
    fn test_high_precision() {
        // f(x) = sin(x), f'(x) = cos(x), xmin at π
        let f = |x: f64| x.sin();
        let df = |x: f64| x.cos();
        let objective = SingleDimDerFn::new(f, df);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.minimize(3.0, 3.2, Some(1e-15), None).unwrap();

        assert!((result.xmin - PI).abs() < 1e-14);
        assert!(result.fmin.abs() < 1e-14);
        assert!((dbrent.xmin - PI).abs() < 1e-14);
        assert!(dbrent.fmin.abs() < 1e-14);
    }

    #[test]
    fn test_brent_method_convergence_speed() {
        // Test 1: Quadratic convergence near root
        let cubic = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let cubic_grad = |x: f64| 3.0 * x.powi(2) - 2.0;
        let objective = SingleDimDerFn::new(cubic, cubic_grad);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.minimize(2.0, 3.0, Some(1e-12), None).unwrap();

        assert!(result.converged);
        assert!(result.iters < 10); // Should converge quickly
        assert!(result.fmin.abs() < 1e-10);
    }

    #[test]
    fn test_brent_method_difficult_functions() {
        // Test 1: Function with very flat derivative
        let flat_derivative = |x: f64| x.powi(5) - x;
        let flat_derivative_grad = |x: f64| 5.0 * x.powi(4) - 1.0;
        let objective = SingleDimDerFn::new(flat_derivative, flat_derivative_grad);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.minimize(0.5, 1.5, Some(1e-8), None);
        assert!(result.is_ok());

        // Test 2: Oscillatory function
        let oscillatory = |x: f64| x * x - 2.0 + 0.1 * (10.0 * x).sin();
        let oscillatory_grad = |x: f64| 2.0 * x + (10.0 * x).cos();
        let objective = SingleDimDerFn::new(oscillatory, oscillatory_grad);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent.minimize(1.0, 2.0, Some(1e-6), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_newton_with_bracket_robustness() {
        // Test function where pure Newton might diverge
        let difficult = |x: f64| (x - 1.0).powi(3);
        let difficult_grad = |x: f64| 3.0 * (x - 1.0).powi(2);
        let objective = SingleDimDerFn::new(difficult, difficult_grad);
        let mut dbrent = DBrent::new(objective);

        let result = dbrent
            .newton_with_bracket(2.0, 0.5, 1.5, Some(1e-10), None)
            .unwrap();

        assert!((result.xmin - 1.0).abs() < 1e-3);
        assert!(
            result.method_used.contains(&String::from("newton"))
                || result.method_used.contains(&String::from("bisection"))
        );
    }
}
