#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::minimize::{MinimizerError, f64::ObjFn};
use std::fmt;

/// Result of golden section search
#[derive(Debug, Clone)]
pub struct GoldenResult {
    pub xmin: f64,
    pub fmin: f64,
    pub iters: usize,
    pub converged: bool,
}

#[derive(Clone)]
pub struct Golden {
    xmin: f64,
    fmin: f64,
    f: Box<dyn ObjFn>,
    // pub bracket: BracketF64,
    iters: usize,
    converged: bool,
}

impl Golden {
    /// Golden ratio constant (φ - 1)
    const GOLDEN_RATIO: f64 = 0.618_033_988_749_895;

    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn + 'static,
    {
        Golden {
            xmin: 0.0,
            fmin: 0.0,
            f: Box::new(f),
            // bracket: BracketF64::new(),
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn>) -> Self {
        Golden {
            xmin: 0.0,
            fmin: 0.0,
            f: f,
            // bracket: BracketF64::new(),
            iters: 0,
            converged: false,
        }
    }

    // fn set_x(&mut self, x1: f64, x2: f64, cat: bool) -> f64 {
    //     const R: f64 = 0.61803399;
    //     const C: f64 = 1.0 - R;

    //     if cat {
    //         self.bracket.bx = x1;
    //         self.bracket.bx + C * (self.bracket.cx - self.bracket.bx)
    //     } else {
    //         self.bracket.bx = x2;
    //         self.bracket.bx - C * (self.bracket.bx - self.bracket.cx)
    //     }
    // }

    /// Golden section search for finding the minimum of a unimodal function
    ///
    /// # Arguments
    /// * `f` - The function to minimize
    /// * `a` - Left bracket boundary
    /// * `b` - Right bracket boundary  
    /// * `tol` - Convergence tolerance (default: 1e-6)
    /// * `max_iters` - Maximum iterations (default: 100)
    ///
    /// # Returns
    /// * `GoldenResult` containing the minimum point, function value, and convergence info
    ///
    /// # Errors
    /// * `InvalidBracket` if a >= b
    /// * `InvalidTolerance` if tolerance <= 0
    /// * `MaxIterationsExceeded` if convergence not reached
    pub fn golden_section_search(
        &mut self,
        a: f64,
        b: f64,
        tol: Option<f64>,
        max_iters: Option<usize>,
    ) -> Result<GoldenResult, MinimizerError> {
        self.converged = false;
        let tol = tol.unwrap_or(1e-6);
        let max_iter = max_iters.unwrap_or(100);

        // Validate inputs
        if a >= b {
            return Err(MinimizerError::InvalidBracket);
        }
        if tol <= 0.0 {
            return Err(MinimizerError::InvalidTolerance);
        }

        let mut x1 = a;
        let mut x4 = b;

        // Initial interior points using golden ratio
        let mut x2 = x1 + (1.0 - Golden::GOLDEN_RATIO) * (x4 - x1);
        let mut x3 = x1 + Golden::GOLDEN_RATIO * (x4 - x1);

        let mut f2 = self.f.call_scalar(x2);
        let mut f3 = self.f.call_scalar(x3);

        self.iters = 0;

        // Main iteration loop
        while (x4 - x1).abs() > tol && self.iters < max_iter {
            self.iters += 1;

            if f2 < f3 {
                // Minimum is in [x1, x3]
                x4 = x3;
                x3 = x2;
                f3 = f2;
                x2 = x1 + (1.0 - Golden::GOLDEN_RATIO) * (x4 - x1);
                f2 = self.f.call_scalar(x2);
            } else {
                // Minimum is in [x2, x4]
                x1 = x2;
                x2 = x3;
                f2 = f3;
                x3 = x1 + Golden::GOLDEN_RATIO * (x4 - x1);
                f3 = self.f.call_scalar(x3);
            }
        }

        if self.iters >= max_iter {
            return Err(MinimizerError::MaxIterationsExceeded);
        }

        // Return the point with smaller function value
        if f2 < f3 {
            self.xmin = x2;
            self.fmin = f2;
        } else {
            self.xmin = x3;
            self.fmin = f3;
        };

        self.converged = true;
        Ok(GoldenResult {
            xmin: self.xmin,
            fmin: self.fmin,
            iters: self.iters,
            converged: self.converged,
        })
    }

    /// Convenience function with default parameters
    pub fn minimize(&mut self, a: f64, b: f64) -> Result<GoldenResult, MinimizerError> {
        self.golden_section_search(a, b, None, None)
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

impl fmt::Debug for Golden {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Goldenf64( xmin: {}, fmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod minimize_f64_golden_tests {
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
    fn test_quadratic_minimum() {
        // f(x) = (x - 2)^2, minimum at x = 2
        let f = |x: f64| (x - 2.0).powi(2);
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden.minimize(0.0, 5.0).unwrap();

        assert!((result.xmin - 2.0).abs() < 1e-5);
        assert!(result.fmin < 1e-10);
        assert!(result.converged);
        assert!((golden.xmin - 2.0).abs() < 1e-5);
        assert!(golden.fmin < 1e-10);
        assert!(golden.converged);
    }

    #[test]
    fn test_sin_minimum() {
        // f(x) = sin(x), minimum at x = 3π/2 ≈ 4.712
        let f = |x: f64| x.sin();
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden.minimize(3.0, 6.0).unwrap();

        assert!((result.xmin - 1.5 * std::f64::consts::PI).abs() < 1e-4);
        assert!((result.fmin - (-1.0)).abs() < 1e-6);
        assert!((golden.xmin - 1.5 * std::f64::consts::PI).abs() < 1e-4);
        assert!((golden.fmin - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_custom_tolerance() {
        let f = |x: f64| (x - 1.0).powi(2);
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden
            .golden_section_search(0.0, 2.0, Some(1e-10), None)
            .unwrap();

        assert!((result.xmin - 1.0).abs() < 1e-9);
        assert!((golden.xmin - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_invalid_bracket() {
        let f = |x: f64| x * x;
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden.minimize(2.0, 1.0);

        assert!(matches!(result, Err(MinimizerError::InvalidBracket)));
        assert!(!golden.converged);
    }

    #[test]
    fn test_invalid_tolerance() {
        let f = |x: f64| x * x;
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden.golden_section_search(0.0, 1.0, Some(-1.0), None);

        assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
        assert!(!golden.converged);
    }

    #[test]
    fn test_golden_section_unimodal_functions() {
        // Test 1: Simple quadratic
        let quadratic = |x: f64| (x - 3.0).powi(2) + 2.0;
        let objective = SingleDimFn::new(quadratic);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(0.0, 6.0, Some(1e-8), None)
            .unwrap();
        assert!((result.xmin - 3.0).abs() < 1e-6);
        assert!((result.fmin - 2.0).abs() < 1e-6);

        // Test 2: Quartic function
        let quartic = |x: f64| (x - 1.5).powi(4) + 0.5;
        let objective = SingleDimFn::new(quartic);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(-1.0, 4.0, Some(1e-8), None)
            .unwrap();
        assert!((result.xmin - 1.5).abs() < 1e-4);
        assert!((result.fmin - 0.5).abs() < 1e-6);

        // Test 3: Transcendental function
        let transcendental = |x: f64| x * x + (2.0 * x).sin();
        let objective = SingleDimFn::new(transcendental);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(-2.0, 2.0, Some(1e-6), None)
            .unwrap();
        assert!(result.converged);
        assert!(result.fmin < 0.0); // Should find negative minimum
    }

    #[test]
    fn test_golden_section_edge_cases() {
        // Test 1: Very narrow bracket
        let func = |x: f64| (x - 0.5).powi(2);
        let objective = SingleDimFn::new(func);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(0.49, 0.51, Some(1e-10), None)
            .unwrap();
        assert!((result.xmin - 0.5).abs() < 1e-8);

        // Test 2: Flat function
        let flat = |x: f64| {
            if (x - 1.0).abs() < 0.1 {
                0.0
            } else {
                (x - 1.0).abs()
            }
        };
        let objective = SingleDimFn::new(flat);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(0.0, 2.0, Some(1e-4), None)
            .unwrap();
        assert!((result.xmin - 1.0).abs() < 0.2);

        // Test 3: Maximum iterations
        let difficult = |x: f64| (x - PI).powi(2) + 1e-10 * (100.0 * x).sin();
        let objective = SingleDimFn::new(difficult);
        let mut golden = Golden::new(objective);
        let result = golden.golden_section_search(0.0, 2.0 * PI, Some(1e-15), Some(10));
        assert!(result.is_err()); // Should not crash even with few iterations
    }
}
