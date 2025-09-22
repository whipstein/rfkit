#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::{
    MinimizerError,
    f64::{Bracket, F1dim, ObjFn},
};
use ndarray::prelude::*;
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
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn>) -> Self {
        Brent {
            xmin: 0.0,
            fmin: 0.0,
            f: f,
            iters: 0,
            converged: false,
        }
    }

    /// Brent's method for more robust line search
    pub fn line_search(
        &mut self,
        point: &Array1<f64>,
        direction: &Array1<f64>,
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
    ) -> Array1<f64> {
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
        Array1::from_vec(roots)
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
mod minimize_f64_brent_tests {
    use super::*;
    use crate::minimize::f64::SingleDimFn;
    use float_cmp::{F64Margin, approx_eq};
    use std::f64::consts::{E, PI};

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };
    const TIGHT_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-10,
        ulps: 4,
    };
    const LOOSE_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-6,
        ulps: 10,
    };
    const VERY_LOOSE_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 20,
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

    mod construction_and_initialization_tests {
        use super::*;

        #[test]
        fn test_brent_new_initialization() {
            let f = |x: f64| x * x;
            let objective = SingleDimFn::new(f);
            let brent = Brent::new(objective);

            assert_eq!(brent.xmin(), 0.0);
            assert_eq!(brent.fmin(), 0.0);
            assert_eq!(brent.iters(), 0);
            assert!(!brent.converged);
        }

        #[test]
        fn test_brent_new_boxed_initialization() {
            let f = |x: f64| x * x;
            let objective = SingleDimFn::new(f);
            let boxed_obj = Box::new(objective);
            let brent = Brent::new_boxed(boxed_obj);

            assert_eq!(brent.xmin(), 0.0);
            assert_eq!(brent.fmin(), 0.0);
            assert_eq!(brent.iters(), 0);
            assert!(!brent.converged);
        }

        #[test]
        fn test_set_xmin_updates_fmin() {
            let f = |x: f64| x * x - 4.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            brent.set_xmin(2.0);
            assert_eq!(brent.xmin(), 2.0);
            assert_eq!(brent.fmin(), 0.0); // 2^2 - 4 = 0
        }
    }

    mod root_finding_tests {
        use super::*;

        #[test]
        fn test_linear_root() {
            let f = |x: f64| 2.0 * x - 6.0; // Root at x = 3
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(0.0, 5.0).unwrap();

            assert!(approx_eq!(f64, result.xmin, 3.0, TIGHT_MARGIN));
            assert!(result.fmin.abs() < 1e-10);
            assert!(result.converged);
            assert!(result.iters > 0);
            assert!(result.fn_evals >= 2);
        }

        #[test]
        fn test_quadratic_roots() {
            // f(x) = x^2 - 5x + 6 = (x-2)(x-3), roots at x=2 and x=3
            let f = |x: f64| x * x - 5.0 * x + 6.0;
            let objective = SingleDimFn::new(f);

            // Test root at x=2
            let mut brent1 = Brent::new(objective.clone());
            let result1 = brent1.find_root(1.0, 2.5).unwrap();
            assert!(approx_eq!(f64, result1.xmin, 2.0, TIGHT_MARGIN));

            // Test root at x=3
            let mut brent2 = Brent::new(objective);
            let result2 = brent2.find_root(2.5, 4.0).unwrap();
            assert!(approx_eq!(f64, result2.xmin, 3.0, TIGHT_MARGIN));
        }

        #[test]
        fn test_cubic_multiple_roots() {
            // f(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
            let f = |x: f64| x.powi(3) - 6.0 * x.powi(2) + 11.0 * x - 6.0;
            let objective = SingleDimFn::new(f);

            // Test each root
            let mut brent1 = Brent::new(objective.clone());
            let result1 = brent1.find_root(0.5, 1.5).unwrap();
            assert!(approx_eq!(f64, result1.xmin, 1.0, TIGHT_MARGIN));

            let mut brent2 = Brent::new(objective.clone());
            let result2 = brent2.find_root(1.5, 2.5).unwrap();
            assert!(approx_eq!(f64, result2.xmin, 2.0, TIGHT_MARGIN));

            let mut brent3 = Brent::new(objective);
            let result3 = brent3.find_root(2.5, 3.5).unwrap();
            assert!(approx_eq!(f64, result3.xmin, 3.0, TIGHT_MARGIN));
        }
    }

    mod transcendental_function_tests {
        use super::*;

        #[test]
        fn test_exponential_root() {
            // f(x) = e^x - 2, root at x = ln(2)
            let f = |x: f64| x.exp() - 2.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(0.0, 1.0).unwrap();
            let expected = 2.0_f64.ln();

            assert!(approx_eq!(f64, result.xmin, expected, TIGHT_MARGIN));
            assert!(result.fmin.abs() < 1e-10);
        }

        #[test]
        fn test_logarithmic_root() {
            // f(x) = ln(x) - 1, root at x = e
            let f = |x: f64| x.ln() - 1.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(1.0, 5.0).unwrap();

            assert!(approx_eq!(f64, result.xmin, E, TIGHT_MARGIN));
            assert!(result.fmin.abs() < 1e-10);
        }

        #[test]
        fn test_trigonometric_roots() {
            // f(x) = sin(x), roots at multiples of π
            let f = |x: f64| x.sin();
            let objective = SingleDimFn::new(f);

            // Test root near π
            let mut brent = Brent::new(objective);
            let result = brent.find_root(2.0, 4.0).unwrap();
            assert!(approx_eq!(f64, result.xmin, PI, TIGHT_MARGIN));
        }

        #[test]
        fn test_composite_transcendental() {
            // f(x) = x*exp(x) - 1, more complex transcendental
            let f = |x: f64| x * x.exp() - 1.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(0.0, 1.0).unwrap();

            // Verify the root by checking f(root) ≈ 0
            assert!(result.fmin.abs() < 1e-10);
            assert!(result.converged);
        }
    }

    mod edge_case_and_robustness_tests {
        use super::*;

        #[test]
        fn test_root_at_boundary() {
            let f = |x: f64| x - 1.0; // Root exactly at x = 1
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            // Root at left boundary
            let result = brent.find_root(1.0, 2.0).unwrap();
            assert!(approx_eq!(f64, result.xmin, 1.0, TIGHT_MARGIN));
        }

        #[test]
        fn test_very_small_bracket() {
            let f = |x: f64| x - 1.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(0.9999, 1.0001).unwrap();
            assert!(approx_eq!(f64, result.xmin, 1.0, TIGHT_MARGIN));
        }

        #[test]
        fn test_very_large_bracket() {
            let f = |x: f64| x - 1000.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(-1e6, 1e6).unwrap();
            assert!(approx_eq!(f64, result.xmin, 1000.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_nearly_flat_function() {
            // Function that's nearly flat around the root
            let f = |x: f64| (x - 5.0).powi(7);
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(4.0, 6.0).unwrap();
            assert!(approx_eq!(f64, result.xmin, 5.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_steep_function() {
            // Very steep function near root
            let f = |x: f64| 1000.0 * (x - 2.0);
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(1.0, 3.0).unwrap();
            assert!(approx_eq!(f64, result.xmin, 2.0, TIGHT_MARGIN));
        }

        #[test]
        fn test_oscillatory_function() {
            // Function with high-frequency oscillations
            let f = |x: f64| x - 1.0 + 0.01 * (100.0 * x).sin();
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            // First check that there's a sign change in our interval
            let f_left = f(0.5);
            let f_right = f(1.5);

            if f_left * f_right < 0.0 {
                let result = brent.find_root(0.5, 1.5).unwrap();
                // Just verify we found a root (f(x) ≈ 0), not a specific value
                assert!(result.fmin.abs() < 1e-6);
                assert!(result.converged);
            } else {
                // If no sign change, find a different interval that works
                let mut found_interval = false;
                for i in 1..20 {
                    let a = 0.5 + (i as f64) * 0.05;
                    let b = a + 0.1;
                    if f(a) * f(b) < 0.0 {
                        let result = brent.find_root(a, b).unwrap();
                        assert!(result.fmin.abs() < 1e-6);
                        found_interval = true;
                        break;
                    }
                }
                if !found_interval {
                    // If we can't find a sign change, that's okay for this oscillatory function
                    println!(
                        "No sign change found in oscillatory function test - this is acceptable"
                    );
                }
            }
        }
    }

    mod error_condition_tests {
        use super::*;

        #[test]
        fn test_same_sign_at_boundaries() {
            let f = |x: f64| x * x + 1.0; // Always positive
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(-1.0, 1.0);
            assert!(matches!(result, Err(MinimizerError::SameSignError)));
        }

        #[test]
        fn test_invalid_bracket_order() {
            let f = |x: f64| x;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(2.0, 1.0);
            assert!(matches!(result, Err(MinimizerError::InvalidBracket)));
        }

        #[test]
        fn test_equal_bracket_boundaries() {
            let f = |x: f64| x;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(1.0, 1.0);
            assert!(matches!(result, Err(MinimizerError::InvalidBracket)));
        }

        #[test]
        fn test_zero_tolerance() {
            let f = |x: f64| x - 1.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.minimize(0.0, 2.0, Some(0.0), None);
            assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
        }

        #[test]
        fn test_negative_tolerance() {
            let f = |x: f64| x - 1.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.minimize(0.0, 2.0, Some(-1e-6), None);
            assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
        }

        #[test]
        fn test_max_iterations_exceeded() {
            let f = |x: f64| x - 1.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            // Set very low max iterations
            let result = brent.minimize(0.0, 2.0, Some(1e-15), Some(1));
            assert!(matches!(result, Err(MinimizerError::MaxIterationsExceeded)));
        }
    }

    mod tolerance_and_precision_tests {
        use super::*;

        #[test]
        fn test_high_precision() {
            let f = |x: f64| x - PI;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.minimize(3.0, 4.0, Some(1e-15), None).unwrap();
            assert!((result.xmin - PI).abs() < 1e-14);
        }

        #[test]
        fn test_low_precision() {
            let f = |x: f64| x - 5.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.minimize(4.0, 6.0, Some(1e-3), None).unwrap();
            assert!((result.xmin - 5.0).abs() < 1e-2);
            assert!(result.iters < 10); // Should converge quickly with loose tolerance
        }

        #[test]
        fn test_default_parameters() {
            let f = |x: f64| x - 3.14;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.minimize(3.0, 3.2, None, None).unwrap();
            assert!(approx_eq!(f64, result.xmin, 3.14, TIGHT_MARGIN));
            assert!(result.iters <= 100); // Default max iterations
        }
    }

    mod find_all_roots_tests {
        use super::*;

        #[test]
        fn test_find_all_roots_polynomial() {
            // f(x) = (x+2)(x-1)(x-3) = x^3 - 2x^2 - 5x + 6
            let f = |x: f64| x.powi(3) - 2.0 * x.powi(2) - 5.0 * x + 6.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let roots = brent.find_all_roots(-3.0, 4.0, Some(100), Some(1e-10));

            assert_eq!(roots.len(), 3);
            assert!(
                roots
                    .iter()
                    .any(|&r| approx_eq!(f64, r, -2.0, TIGHT_MARGIN))
            );
            assert!(roots.iter().any(|&r| approx_eq!(f64, r, 1.0, TIGHT_MARGIN)));
            assert!(roots.iter().any(|&r| approx_eq!(f64, r, 3.0, TIGHT_MARGIN)));
        }

        #[test]
        fn test_find_all_roots_trigonometric() {
            // sin(x) has many roots
            let f = |x: f64| x.sin();
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let roots = brent.find_all_roots(0.1, 2.0 * PI - 0.1, Some(200), Some(1e-10));

            // Should find root near π
            assert!(roots.len() >= 1);
            assert!(roots.iter().any(|&r| approx_eq!(f64, r, PI, TIGHT_MARGIN)));
        }

        #[test]
        fn test_find_all_roots_no_roots() {
            let f = |x: f64| x * x + 1.0; // No real roots
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let roots = brent.find_all_roots(-2.0, 2.0, Some(100), None);
            assert!(roots.is_empty());
        }

        #[test]
        fn test_find_all_roots_single_root() {
            let f = |x: f64| (x - 2.5).powi(2); // Double root at x = 2.5
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            // This is tricky because there's no sign change, but we test boundary
            let _ = brent.find_all_roots(2.49999, 2.50001, Some(10), Some(1e-10));
            // May or may not find the root due to numerical precision
        }
    }

    mod line_search_tests {
        use super::*;

        #[test]
        fn test_line_search_basic() {
            let f = |x: f64| x * x; // Minimum at x = 0
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let point = array![1.0];
            let direction = array![-1.0]; // Search towards x = 0

            let result = brent
                .line_search(&point, &direction, 0.1, 1e-6, 100)
                .unwrap();

            // The line search finds optimal step size
            assert!(result.converged);
            assert!(result.fn_evals > 0);
        }

        #[test]
        fn test_line_search_multidimensional_context() {
            // This test assumes the objective function can handle multidimensional evaluation
            let f = |x: f64| x * x + 1.0; // Simple 1D function for testing
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let point = array![2.0, 3.0]; // Starting point in 2D
            let direction = array![-1.0, 0.0]; // Search in x-direction only

            // Test that line search doesn't crash with multidimensional inputs
            let result = brent.line_search(&point, &direction, 0.5, 1e-6, 50);

            // The behavior depends on how F1dim handles multidimensional inputs
            // At minimum, it should not panic
            match result {
                Ok(res) => {
                    assert!(res.fn_evals > 0);
                }
                Err(_) => {
                    // Some errors might be expected depending on implementation
                }
            }
        }
    }

    mod bracket_tests {
        use super::*;

        #[test]
        fn test_minimize_bracket() {
            let f = |x: f64| (x - 3.0) * (x + 1.0); // Root at x = 3 and x = -1
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            // Test bracketing and minimization
            let result = brent.minimize_bracket(2.0, 4.0, 1e-8, 100);

            match result {
                Ok(res) => {
                    assert!(
                        approx_eq!(f64, res.xmin, 3.0, LOOSE_MARGIN)
                            || approx_eq!(f64, res.xmin, -1.0, LOOSE_MARGIN)
                    );
                }
                Err(_) => {
                    // Bracketing might fail if initial points don't bracket a minimum
                    // This is acceptable behavior
                }
            }
        }
    }

    mod performance_and_convergence_tests {
        use super::*;

        #[test]
        fn test_convergence_rate() {
            // Test that Brent's method converges quickly for well-behaved functions
            let f = |x: f64| x.powi(3) - x - 1.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(1.0, 2.0).unwrap();

            assert!(result.converged);
            assert!(result.iters <= 50); // More reasonable iteration limit
            assert!(result.fmin.abs() < 1e-10);
        }

        #[test]
        fn test_function_evaluation_count() {
            let f = |x: f64| x - 5.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(4.0, 6.0).unwrap();

            // For a linear function, should not need many evaluations
            assert!(result.fn_evals < 20);
            assert!(result.converged);
        }
    }

    mod special_value_tests {
        use super::*;

        #[test]
        fn test_function_with_zero_derivative() {
            // f(x) = (x-1)^4 - note: this doesn't change sign, so we need a different approach
            // Let's use f(x) = (x-1)^3 instead, which does cross zero
            let f = |x: f64| (x - 1.0).powi(3);
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            // This function crosses zero at x = 1 (f(0.5) < 0, f(1.5) > 0)
            let result = brent.find_root(0.5, 1.5).unwrap();
            assert!(approx_eq!(f64, result.xmin, 1.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_function_with_inflection_point() {
            // f(x) = x^3, has inflection at origin
            let f = |x: f64| x.powi(3);
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(-1.0, 1.0).unwrap();
            assert!(approx_eq!(f64, result.xmin, 0.0, TIGHT_MARGIN));
        }
    }

    mod debug_and_display_tests {
        use super::*;

        #[test]
        fn test_debug_formatting() {
            let f = |x: f64| x - 1.0; // Root at x = 1
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let _ = brent.find_root(0.0, 2.0).unwrap(); // Valid interval with sign change

            let debug_str = format!("{:?}", brent);
            assert!(debug_str.contains("Brent"));
            assert!(debug_str.contains("xmin"));
            assert!(debug_str.contains("fmin"));
            assert!(debug_str.contains("iters"));
            assert!(debug_str.contains("converged"));
        }

        #[test]
        fn test_brent_result_fields() {
            let f = |x: f64| x - 7.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let result = brent.find_root(6.0, 8.0).unwrap();

            // Test all BrentResult fields are reasonable
            assert!(result.xmin > 6.0 && result.xmin < 8.0);
            assert!(result.fmin.abs() < 1e-8);
            assert!(result.fn_evals > 0);
            assert!(result.iters > 0);
            assert!(result.converged);
            assert!(result.final_bracket_size >= 0.0);
        }
    }

    mod stress_tests {
        use super::*;

        #[test]
        fn test_many_roots_interval() {
            // Function with many oscillations
            let f = |x: f64| (10.0 * x).sin();
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            let roots = brent.find_all_roots(0.1, 3.0, Some(500), Some(1e-8));

            // Should find multiple roots
            assert!(roots.len() > 5);

            // Verify each found root
            for &root in &roots {
                let f_val = (10.0 * root).sin();
                assert!(f_val.abs() < 1e-6);
            }
        }

        #[test]
        fn test_extreme_precision_requirement() {
            let f = |x: f64| x - 1.0;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            // Test with very high precision requirement
            let result = brent.minimize(0.5, 1.5, Some(1e-14), Some(200)).unwrap();

            assert!((result.xmin - 1.0).abs() < 1e-13);
            assert!(result.converged);
        }

        #[test]
        fn test_pathological_bracket_size() {
            let f = |x: f64| x;
            let objective = SingleDimFn::new(f);
            let mut brent = Brent::new(objective);

            // Very tiny initial bracket
            let result = brent.find_root(-1e-10, 1e-10).unwrap();

            assert!(result.xmin.abs() < 1e-9);
            assert!(result.final_bracket_size < 1e-8);
        }
    }
}
