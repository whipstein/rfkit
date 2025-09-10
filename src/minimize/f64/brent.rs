#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::{
    MinimizerError,
    f64::{Bracket, F1dim, ObjFn},
};
use std::fmt;

/// Result of Brent's method root finding
#[derive(Debug, Clone)]
pub struct BrentResult {
    pub xmin: f64,
    pub fmin: f64,
    pub fn_evals: usize,
    pub iters: usize,
    pub converged: bool,
    pub final_bracket_size: f64,
}

#[derive(Clone)]
pub struct Brent {
    xmin: f64,
    fmin: f64,
    f: Box<dyn ObjFn>,
    iters: usize,
    converged: bool,
}

impl Brent {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn + 'static,
    {
        Brent {
            xmin: 0.0,
            fmin: 0.0,
            f: Box::new(f),
            // bracket: Bracket::new(),
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn>) -> Self {
        Brent {
            xmin: 0.0,
            fmin: 0.0,
            f: f,
            // bracket: Bracket::new(),
            iters: 0,
            converged: false,
        }
    }

    // pub fn bracket(&mut self, a: f64, b: f64) {
    //     self.bracket.bracket_boxed(a, b, &mut self.f);
    // }

    /// Brent's method for more robust line search
    pub fn line_search(
        &mut self,
        point: &Vec<f64>,
        direction: &Vec<f64>,
        initial_step: f64,
        tol: f64,
        max_evaluations: usize,
    ) -> Result<BrentResult, MinimizerError> {
        const CGOLD: f64 = 0.3819660;
        const ZEPS: f64 = 1e-10;

        self.converged = false;

        let mut func = F1dim::new_boxed(self.f.clone());

        // Initial bracketing with more sophisticated approach
        let mut ax = 0.0;
        let mut bx = initial_step;
        let mut cx = 0.0;

        let mut fa = func.eval(point, direction, ax)?;
        let mut fb = func.eval(point, direction, bx)?;
        let mut evaluations = 2;

        // Bracket the minimum
        if fb > fa {
            std::mem::swap(&mut ax, &mut bx);
            std::mem::swap(&mut fa, &mut fb);
        }

        cx = bx + 1.618034 * (bx - ax);
        let mut fc = func.eval(point, direction, cx)?;
        evaluations += 1;

        while fb > fc && evaluations < max_evaluations / 3 {
            let r = (bx - ax) * (fb - fc);
            let q = (bx - cx) * (fb - fa);
            let u = bx
                - ((bx - cx) * q - (bx - ax) * r)
                    / (2.0
                        * if q - r > 0.0 {
                            (q - r).max(ZEPS)
                        } else {
                            (q - r).min(-ZEPS)
                        });
            let ulim = bx + 100.0 * (cx - bx);

            let (new_u, new_fu) = if (bx - u) * (u - cx) > 0.0 {
                let fu = func.eval(point, direction, u)?;
                evaluations += 1;
                if fu < fc {
                    (cx, fc)
                } else if fu > fb {
                    (u, fu)
                } else {
                    let u_new = cx + 1.618034 * (cx - bx);
                    (u_new, func.eval(point, direction, u_new)?)
                }
            } else if (cx - u) * (u - ulim) > 0.0 {
                let fu = func.eval(point, direction, u)?;
                evaluations += 1;
                if fu < fc {
                    let u_new = u + 1.618034 * (u - cx);
                    (u_new, func.eval(point, direction, u_new)?)
                } else {
                    (u, fu)
                }
            } else if (u - ulim) * (ulim - cx) >= 0.0 {
                (ulim, func.eval(point, direction, ulim)?)
            } else {
                let u_new = cx + 1.618034 * (cx - bx);
                (u_new, func.eval(point, direction, u_new)?)
            };

            evaluations += 1;
            ax = bx;
            bx = cx;
            cx = new_u;
            fa = fb;
            fb = fc;
            fc = new_fu;
        }

        // Now minimize using Brent's method
        let mut a = if ax < cx { ax } else { cx };
        let mut b = if ax > cx { ax } else { cx };

        let mut x = bx;
        let mut w = bx;
        let mut v = bx;
        let mut fx = fb;
        let mut fw = fb;
        let mut fv = fb;

        let mut e = 0_f64;
        let mut d = 0.0;

        let tol1 = tol;

        while evaluations < max_evaluations {
            let xm = 0.5 * (a + b);
            let tol1_x = tol1 * x.abs() + ZEPS;
            let tol2_x = 2.0 * tol1_x;

            if (x - xm).abs() <= tol2_x - 0.5 * (b - a) {
                break;
            }

            let mut u = 0.0;
            let mut use_golden = true;

            if e.abs() > tol1_x {
                // Try parabolic fit
                let r = (x - w) * (fx - fv);
                let q = (x - v) * (fx - fw);
                let p = (x - v) * q - (x - w) * r;
                let q_final = 2.0 * (q - r);

                if q_final != 0.0 {
                    let p_over_q = p / q_final;
                    if q_final > 0.0 && p > 0.0 && p_over_q > (a - x) && p_over_q < (b - x) {
                        e = d;
                        d = p_over_q;
                        u = x + d;
                        if u - a < tol2_x || b - u < tol2_x {
                            d = if xm - x > 0.0 { tol1_x } else { -tol1_x };
                        }
                        use_golden = false;
                    }
                }
            }

            if use_golden {
                e = if x >= xm { a - x } else { b - x };
                d = CGOLD * e;
            }

            u = x + if d.abs() >= tol1_x {
                d
            } else {
                if d > 0.0 { tol1_x } else { -tol1_x }
            };
            let fu = func.eval(point, direction, u)?;
            evaluations += 1;

            if fu <= fx {
                if u >= x {
                    a = x;
                } else {
                    b = x;
                }
                v = w;
                w = x;
                x = u;
                fv = fw;
                fw = fx;
                fx = fu;
            } else {
                if u < x {
                    a = u;
                } else {
                    b = u;
                }
                if fu <= fw || w == x {
                    v = w;
                    w = u;
                    fv = fw;
                    fw = fu;
                } else if fu <= fv || v == x || v == w {
                    v = u;
                    fv = fu;
                }
            }
        }

        self.xmin = x;
        self.fmin = fx;
        if evaluations < max_evaluations {
            self.converged = true;
        }
        Ok(BrentResult {
            xmin: self.xmin,
            fmin: self.fmin,
            fn_evals: evaluations,
            iters: self.iters,
            converged: self.converged,
            final_bracket_size: (b - a).abs(),
        })
    }

    pub fn minimize_bracket(
        &mut self,
        a: f64,
        b: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<BrentResult, MinimizerError> {
        let mut bracket = Bracket::new_boxed(self.f.clone());
        let (a, b) = match bracket.bracket(a, b) {
            Ok(result) => (result.a, result.b),
            Err(e) => return Err(e),
        };
        self.minimize(a, b, Some(tol), Some(max_iter))
    }

    /// Brent's method for finding roots of a function
    ///
    /// This algorithm combines the robustness of bisection with the speed of
    /// inverse quadratic interpolation and secant method.
    ///
    /// # Arguments
    /// * `f` - The function whose root we want to find
    /// * `a` - Left bracket boundary (f(a) and f(b) must have opposite signs)
    /// * `b` - Right bracket boundary
    /// * `tol` - Convergence tolerance (default: 1e-12)
    /// * `max_iters` - Maximum iterations (default: 100)
    ///
    /// # Returns
    /// * `BrentResult` containing the root, function value, and convergence info
    ///
    /// # Errors
    /// * `InvalidBracket` if a >= b
    /// * `InvalidTolerance` if tolerance <= 0
    /// * `SameSignError` if f(a) and f(b) have the same sign
    /// * `MaxIterationsExceeded` if convergence not reached
    pub fn minimize(
        &mut self,
        mut a: f64,
        mut b: f64,
        tol: Option<f64>,
        max_iters: Option<usize>,
    ) -> Result<BrentResult, MinimizerError> {
        self.converged = false;
        let tol = tol.unwrap_or(1e-12);
        let max_iter = max_iters.unwrap_or(100);

        // Validate inputs
        if a >= b {
            return Err(MinimizerError::InvalidBracket);
        }
        if tol <= 0.0 {
            return Err(MinimizerError::InvalidTolerance);
        }

        let mut fa = self.f.call_scalar(a);
        let mut fb = self.f.call_scalar(b);
        let mut evaluations = 2;

        // Check that f(a) and f(b) have opposite signs
        if fa * fb > 0.0 {
            return Err(MinimizerError::SameSignError);
        }

        // Ensure |f(a)| >= |f(b)|
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }

        let mut c = a;
        let mut fc = fa;
        let mut mflag = true;
        let mut d = 0.0;

        self.iters = 0;

        while fb.abs() > tol && (b - a).abs() > tol && self.iters < max_iter {
            self.iters += 1;

            let mut s = if fa != fc && fb != fc {
                // Inverse quadratic interpolation
                a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb))
            } else {
                // Secant method
                b - fb * (b - a) / (fb - fa)
            };

            // Check conditions for using bisection instead
            let condition1 = !(((3.0 * a + b) / 4.0)..=b).contains(&s);
            let condition2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
            let condition3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;
            let condition4 = mflag && (b - c).abs() < tol;
            let condition5 = !mflag && (c - d).abs() < tol;

            let use_bisection = condition1 || condition2 || condition3 || condition4 || condition5;

            if use_bisection {
                // Bisection method
                s = (a + b) / 2.0;
                mflag = true;
            } else {
                mflag = false;
            }

            let fs = self.f.call_scalar(s);
            evaluations += 1;

            // Update for next iteration
            d = c;
            c = b;
            fc = fb;

            if fa * fs < 0.0 {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }

            // Ensure |f(a)| >= |f(b)|
            if fa.abs() < fb.abs() {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }
        }

        if self.iters >= max_iter {
            return Err(MinimizerError::MaxIterationsExceeded);
        }

        self.xmin = b;
        self.fmin = fb;
        self.converged = true;
        Ok(BrentResult {
            xmin: self.xmin,
            fmin: self.fmin,
            fn_evals: evaluations,
            iters: self.iters,
            converged: self.converged,
            final_bracket_size: (b - a).abs(),
        })
    }

    /// Convenience function with default parameters
    pub fn find_root(&mut self, a: f64, b: f64) -> Result<BrentResult, MinimizerError> {
        self.minimize(a, b, None, None)
    }

    /// Find all roots in an interval by subdividing and applying Brent's method
    ///
    /// # Arguments
    /// * `f` - The function whose roots we want to find
    /// * `a` - Left boundary of search interval
    /// * `b` - Right boundary of search interval  
    /// * `subdivisions` - Number of subdivisions to try (default: 100)
    /// * `tol` - Root tolerance (default: 1e-12)
    ///
    /// # Returns
    /// * Vector of unique roots found in the interval
    pub fn find_all_roots(
        &mut self,
        a: f64,
        b: f64,
        subdivisions: Option<usize>,
        tol: Option<f64>,
    ) -> Vec<f64> {
        let n_sub = subdivisions.unwrap_or(100);
        let tol = tol.unwrap_or(1e-12);
        let mut roots = Vec::new();

        let dx = (b - a) / n_sub as f64;

        for i in 0..n_sub {
            let x1 = a + i as f64 * dx;
            let x2 = a + (i + 1) as f64 * dx;

            let f1 = self.f.call_scalar(x1);
            let f2 = self.f.call_scalar(x2);

            // Check for sign change
            if f1 * f2 < 0.0 {
                if let Ok(result) = self.minimize(x1, x2, Some(tol), None) {
                    // Check if this root is unique (not too close to existing roots)
                    let is_unique = roots.iter().all(|&existing_root| {
                        (result.xmin - existing_root as f64).abs() > tol * 10.0
                    });

                    if is_unique {
                        roots.push(result.xmin);
                    }
                }
            }

            // Check for exact zero
            if f1.abs() < tol {
                let is_unique = roots
                    .iter()
                    .all(|&existing_root| (x1 - existing_root).abs() > tol * 10.0);
                if is_unique {
                    roots.push(x1);
                }
            }
        }

        // Check the right endpoint
        if self.f.call_scalar(b).abs() < tol {
            let is_unique = roots
                .iter()
                .all(|&existing_root| (b - existing_root).abs() > tol * 10.0);
            if is_unique {
                roots.push(b);
            }
        }

        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        roots
    }

    pub fn xmin(&self) -> f64 {
        self.xmin
    }

    pub fn fmin(&self) -> f64 {
        self.fmin
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn set_xmin(&mut self, xmin: f64) {
        self.xmin = xmin;
        self.fmin = self.f.call_scalar(xmin);
    }
}

impl fmt::Debug for Brent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Brent( xmin: {}, fmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod brentf64_tests {
    use super::*;
    use crate::minimize::f64::SingleDimFn;
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
    fn test_quadratic_root() {
        // f(x) = x^2 - 2, xmin at x = √2 ≈ 1.414
        let f = |x: f64| x * x - 2.0;
        let objective = SingleDimFn::new(f);
        let mut brent = Brent::new(objective);

        let result = brent.find_root(1.0, 2.0).unwrap();

        assert!((result.xmin - 2_f64.sqrt()).abs() < 1e-10);
        assert!(result.fmin.abs() < 1e-10);
        assert!(result.converged);
        assert!((brent.xmin - 2_f64.sqrt()).abs() < 1e-10);
        assert!(brent.fmin.abs() < 1e-10);
        assert!(brent.converged);
    }

    #[test]
    fn test_cubic_root() {
        // f(x) = x^3 - x - 1, xmin ≈ 1.324717957
        let f = |x: f64| x.powi(3) - x - 1.0;
        let objective = SingleDimFn::new(f);
        let mut brent = Brent::new(objective);

        let result = brent.find_root(1.0, 2.0).unwrap();

        assert!((result.xmin - 1.324717957).abs() < 1e-8);
        assert!(result.fmin.abs() < 1e-10);
        assert!((brent.xmin - 1.324717957).abs() < 1e-8);
        assert!(brent.fmin.abs() < 1e-10);
    }

    #[test]
    fn test_transcendental() {
        // f(x) = cos(x) - x, xmin ≈ 0.739085133
        let f = |x: f64| x.cos() - x;
        let objective = SingleDimFn::new(f);
        let mut brent = Brent::new(objective);

        let result = brent.find_root(0.0, 1.0).unwrap();

        assert!((result.xmin - 0.739085133).abs() < 1e-8);
        assert!(result.fmin.abs() < 1e-10);
        assert!((brent.xmin - 0.739085133).abs() < 1e-8);
        assert!(brent.fmin.abs() < 1e-10);
    }

    #[test]
    fn test_custom_tolerance() {
        let f = |x: f64| x * x - 4.0;
        let objective = SingleDimFn::new(f);
        let mut brent = Brent::new(objective);

        let result = brent.minimize(1.0, 3.0, Some(1e-15), None).unwrap();

        assert!((result.xmin - 2.0).abs() < 1e-14);
        assert!((brent.xmin - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_same_sign_error() {
        let f = |x: f64| x * x + 1.0; // Always positive
        let objective = SingleDimFn::new(f);
        let mut brent = Brent::new(objective);

        let result = brent.find_root(0.0, 1.0);

        assert!(matches!(result, Err(MinimizerError::SameSignError)));
        assert!(!brent.converged);
    }

    #[test]
    fn test_invalid_bracket() {
        let f = |x: f64| x;
        let objective = SingleDimFn::new(f);
        let mut brent = Brent::new(objective);

        let result = brent.find_root(2.0, 1.0);

        assert!(matches!(result, Err(MinimizerError::InvalidBracket)));
        assert!(!brent.converged);
    }

    #[test]
    fn test_find_all_roots() {
        // f(x) = sin(x) has roots at 0, π, 2π, 3π in [0, 3π]
        let f = |x: f64| x.sin();
        let objective = SingleDimFn::new(f);
        let mut brent = Brent::new(objective);

        let roots = brent.find_all_roots(0.1, 3.0 * PI - 0.1, Some(200), None);

        // Should find roots near π and 2π (0 and 3π are at boundaries)
        assert!(roots.len() >= 2);
        assert!(roots.iter().any(|&r| (r - PI).abs() < 1e-10));
        assert!(roots.iter().any(|&r| (r - 2.0 * PI).abs() < 1e-10));
    }

    #[test]
    fn test_brent_method_convergence_speed() {
        // Test 1: Quadratic convergence near root
        let cubic = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let objective = SingleDimFn::new(cubic);
        let mut brent = Brent::new(objective);

        let result = brent.minimize(2.0, 3.0, Some(1e-12), None).unwrap();

        assert!(result.converged);
        assert!(result.iters <= 20); // Should converge quickly
        assert!(result.fmin.abs() < 1e-10);
    }

    #[test]
    fn test_brent_method_difficult_functions() {
        // Test 1: Function with very flat derivative
        let flat_derivative = |x: f64| x.powi(5) - x;
        let objective = SingleDimFn::new(flat_derivative);
        let mut brent = Brent::new(objective);

        let result = brent.minimize(0.5, 1.5, Some(1e-8), None);
        assert!(result.is_ok());

        // Test 2: Oscillatory function
        let oscillatory = |x: f64| x * x - 2.0 + 0.1 * (10.0 * x).sin();
        let objective = SingleDimFn::new(oscillatory);
        let mut brent = Brent::new(objective);

        let result = brent.minimize(1.0, 2.0, Some(1e-6), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_newton_with_bracket_robustness() {
        // Test function where pure Newton might diverge
        let difficult = |x: f64| (x - 1.0).powi(3);
        let objective = SingleDimFn::new(difficult);
        let mut brent = Brent::new(objective);

        let result = brent.minimize(0.5, 1.5, Some(1e-10), None).unwrap();

        assert!((result.xmin - 1.0).abs() < 1e-8);
    }
}
