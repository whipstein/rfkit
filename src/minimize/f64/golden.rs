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
    use crate::minimize::f64::{F1dim, SingleDimFn};
    use float_cmp::{F64Margin, approx_eq};
    use std::f64::consts::{E, PI};

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 10,
    };
    const DEFAULT_TOL: f64 = 1e-6;
    const DEFAULT_TOL_BRENT: f64 = 1e-10;
    const DEFAULT_MAX_ITER: usize = 100;
    const TIGHT_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-6,
        ulps: 10,
    };
    const LOOSE_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-4,
        ulps: 20,
    };
    const VERY_LOOSE_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-2,
        ulps: 50,
    };

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

    mod basic_functionality_tests {
        use super::*;

        #[test]
        fn test_construction_and_initialization() {
            let f = |x: f64| x * x;
            let objective = SingleDimFn::new(f);
            let golden = Golden::new(objective);

            assert_eq!(golden.xmin(), 0.0);
            assert_eq!(golden.fmin(), 0.0);
            assert_eq!(golden.iters(), 0);
            assert!(!golden.converged);
        }

        #[test]
        fn test_boxed_construction() {
            let f = |x: f64| x * x;
            let objective = SingleDimFn::new(f);
            let boxed_obj: Box<dyn ObjFn> = Box::new(objective);
            let golden = Golden::new_boxed(boxed_obj);

            assert_eq!(golden.xmin(), 0.0);
            assert_eq!(golden.fmin(), 0.0);
        }

        #[test]
        fn test_set_xmin_updates_fmin() {
            let f = |x: f64| x * x + 2.0;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            golden.set_xmin(3.0);
            assert_eq!(golden.xmin(), 3.0);
            assert_eq!(golden.fmin(), 11.0); // 3^2 + 2 = 11
        }
    }

    mod mathematical_function_tests {
        use super::*;

        #[test]
        fn test_simple_quadratic_variants() {
            // Test various quadratic functions with different minima
            let test_cases = vec![
                (
                    F1dim::new(SingleDimFn::new(|x: f64| (x - 0.0).powi(2))),
                    0.0,
                    0.0,
                    -2.0,
                    2.0,
                ),
                (
                    F1dim::new(SingleDimFn::new(|x: f64| (x - 1.0).powi(2))),
                    1.0,
                    0.0,
                    -1.0,
                    3.0,
                ),
                (
                    F1dim::new(SingleDimFn::new(|x: f64| (x - 5.0).powi(2) + 3.0)),
                    5.0,
                    3.0,
                    2.0,
                    8.0,
                ),
                (
                    F1dim::new(SingleDimFn::new(|x: f64| 2.0 * (x - 2.5).powi(2) + 1.5)),
                    2.5,
                    1.5,
                    0.0,
                    5.0,
                ),
                (
                    F1dim::new(SingleDimFn::new(|x: f64| (x + 3.0).powi(2))),
                    -3.0,
                    0.0,
                    -5.0,
                    -1.0,
                ),
            ];

            for (i, (objective, expected_x, expected_f, a, b)) in test_cases.iter().enumerate() {
                let mut golden = Golden::new(objective.clone());
                let result = golden.minimize(*a, *b).unwrap();

                assert!(
                    approx_eq!(f64, result.xmin, *expected_x, LOOSE_MARGIN),
                    "Test case {}: xmin expected {}, got {}",
                    i,
                    expected_x,
                    result.xmin
                );
                assert!(
                    approx_eq!(f64, result.fmin, *expected_f, LOOSE_MARGIN),
                    "Test case {}: fmin expected {}, got {}",
                    i,
                    expected_f,
                    result.fmin
                );
                assert!(result.converged, "Test case {} should converge", i);
            }
        }

        #[test]
        fn test_higher_order_polynomials() {
            // Quartic: (x - 2)^4 + 1, minimum at x = 2, f = 1
            // Fourth-order functions require more iterations for high precision
            let quartic = |x: f64| (x - 2.0).powi(4) + 1.0;
            let objective = SingleDimFn::new(quartic);
            let mut golden = Golden::new(objective);
            let result = golden
                .golden_section_search(0.0, 4.0, Some(1e-4), None)
                .unwrap();
            assert!(
                (result.xmin - 2.0).abs() < 0.05,
                "Quartic: expected 2.0, got {}",
                result.xmin
            );
            assert!(approx_eq!(f64, result.fmin, 1.0, LOOSE_MARGIN));

            // Simpler test: x^4 with minimum at x = 0
            let simple_quartic = |x: f64| x.powi(4) + 0.5;
            let objective = SingleDimFn::new(simple_quartic);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(-2.0, 2.0).unwrap();
            assert!(
                (result.xmin).abs() < 0.01,
                "Simple quartic: expected 0.0, got {}",
                result.xmin
            );
            assert!(approx_eq!(f64, result.fmin, 0.5, LOOSE_MARGIN));

            // Sixth degree: (x - 1)^6 + 0.5 - easier to minimize
            let sixth = |x: f64| (x - 1.0).powi(6) + 0.5;
            let objective = SingleDimFn::new(sixth);
            let mut golden = Golden::new(objective);
            let result = golden
                .golden_section_search(-1.0, 3.0, Some(1e-4), None)
                .unwrap();
            assert!(
                (result.xmin - 1.0).abs() < 0.05,
                "Sixth degree: expected 1.0, got {}",
                result.xmin
            );
            assert!(approx_eq!(f64, result.fmin, 0.5, LOOSE_MARGIN));
        }

        #[test]
        fn test_trigonometric_functions() {
            // Test sin(x) in [π, 2π], minimum at 3π/2
            let sin_func = |x: f64| x.sin();
            let objective = SingleDimFn::new(sin_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(PI, 2.0 * PI).unwrap();
            assert!(approx_eq!(f64, result.xmin, 1.5 * PI, LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin, -1.0, LOOSE_MARGIN));

            // Test cos(x) in [0, π], minimum at π
            let cos_func = |x: f64| x.cos();
            let objective = SingleDimFn::new(cos_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, PI).unwrap();
            assert!(approx_eq!(f64, result.xmin, PI, LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin, -1.0, LOOSE_MARGIN));

            // Test sin^2(x) + cos^2(x) = 1 (constant function)
            let trig_identity = |x: f64| x.sin().powi(2) + x.cos().powi(2);
            let objective = SingleDimFn::new(trig_identity);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 2.0 * PI).unwrap();
            assert!(approx_eq!(f64, result.fmin, 1.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_exponential_and_logarithmic_functions() {
            // e^(x-1) has minimum at negative infinity, but in [0,2] minimum at x=0
            let exp_func = |x: f64| E.powf(x - 1.0);
            let objective = SingleDimFn::new(exp_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 2.0).unwrap();
            assert!(result.xmin < 0.1); // Should be close to 0
            assert!(approx_eq!(f64, result.fmin, E.powf(-1.0), LOOSE_MARGIN));

            // Test a function that actually has an interior minimum
            // f(x) = x^2 - ln(x) has minimum at x = 1/√2 since f'(x) = 2x - 1/x = 0
            let log_func = |x: f64| x * x - x.ln(); // Note: minus ln(x)
            let objective = SingleDimFn::new(log_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.1, 2.0).unwrap();
            let expected_min = 1.0 / 2.0_f64.sqrt(); // ≈ 0.7071
            assert!(
                (result.xmin - expected_min).abs() < 0.1,
                "Expected around {}, got {}",
                expected_min,
                result.xmin
            );
            assert!(result.converged);
        }

        #[test]
        fn test_absolute_value_functions() {
            // |x - 3|, minimum at x = 3
            let abs_func = |x: f64| (x - 3.0).abs();
            let objective = SingleDimFn::new(abs_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 6.0).unwrap();
            assert!(approx_eq!(f64, result.xmin, 3.0, LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin, 0.0, LOOSE_MARGIN));

            // |x - 2| + |x - 4|, minimum in [2, 4]
            let double_abs = |x: f64| (x - 2.0).abs() + (x - 4.0).abs();
            let objective = SingleDimFn::new(double_abs);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 6.0).unwrap();
            assert!(result.xmin >= 1.9 && result.xmin <= 4.1);
            assert!(approx_eq!(f64, result.fmin, 2.0, LOOSE_MARGIN));
        }
    }

    mod error_condition_tests {
        use super::*;

        #[test]
        fn test_invalid_bracket_conditions() {
            let f = |x: f64| x * x;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            // Equal brackets
            assert!(matches!(
                golden.minimize(1.0, 1.0),
                Err(MinimizerError::InvalidBracket)
            ));

            // Reversed brackets
            assert!(matches!(
                golden.minimize(2.0, 1.0),
                Err(MinimizerError::InvalidBracket)
            ));

            // Ensure state remains unchanged after error
            assert!(!golden.converged);
            assert_eq!(golden.iters(), 0);
        }

        #[test]
        fn test_invalid_tolerance_conditions() {
            let f = |x: f64| x * x;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            // Zero tolerance
            assert!(matches!(
                golden.golden_section_search(0.0, 1.0, Some(0.0), None),
                Err(MinimizerError::InvalidTolerance)
            ));

            // Negative tolerance
            assert!(matches!(
                golden.golden_section_search(0.0, 1.0, Some(-1e-6), None),
                Err(MinimizerError::InvalidTolerance)
            ));

            // Very small negative tolerance
            assert!(matches!(
                golden.golden_section_search(0.0, 1.0, Some(-1e-15), None),
                Err(MinimizerError::InvalidTolerance)
            ));
        }

        #[test]
        fn test_max_iterations_exceeded() {
            // Use a very tight tolerance and few iterations to force failure
            let f = |x: f64| (x - 0.5).powi(2);
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            let result = golden.golden_section_search(0.0, 1.0, Some(1e-15), Some(5));
            assert!(matches!(result, Err(MinimizerError::MaxIterationsExceeded)));
            assert!(!golden.converged);
            assert_eq!(golden.iters(), 5);
        }
    }

    mod convergence_and_precision_tests {
        use super::*;

        #[test]
        fn test_various_tolerance_levels() {
            let f = |x: f64| (x - PI).powi(2) + E;

            let tolerances = vec![1e-3, 1e-6, 1e-9, 1e-12];

            for &tol in &tolerances {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);
                let result = golden
                    .golden_section_search(0.0, 2.0 * PI, Some(tol), None)
                    .unwrap();

                // Error should be roughly proportional to tolerance
                let error = (result.xmin - PI).abs();
                // Golden section has linear convergence, so allow more generous error bounds
                // For very tight tolerances, the algorithm may not achieve theoretical precision
                let tolerance_factor = match tol {
                    t if t >= 1e-6 => 100.0,
                    t if t >= 1e-9 => 10000.0,
                    _ => 100000.0, // Very tight tolerances may not be achievable
                };
                assert!(
                    error <= tol * tolerance_factor,
                    "Tolerance {}: error {} > {} (factor: {})",
                    tol,
                    error,
                    tol * tolerance_factor,
                    tolerance_factor
                );

                // Tighter tolerance should require more iterations
                if tol <= 1e-6 {
                    assert!(
                        result.iters > 10,
                        "Should require more iterations for tight tolerance"
                    );
                }
            }
        }

        #[test]
        fn test_convergence_rates() {
            let f = |x: f64| (x - 2.0).powi(4); // Fourth-order function
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            let result = golden.minimize(0.0, 4.0).unwrap();

            // Golden section should converge reasonably quickly
            assert!(
                result.iters < 50,
                "Should converge in reasonable iterations, got {}",
                result.iters
            );
            assert!(result.converged);

            // Check that we actually found the minimum
            assert!(approx_eq!(f64, result.xmin, 2.0, LOOSE_MARGIN));
            assert!(result.fmin < 1e-10);
        }

        #[test]
        fn test_near_machine_precision() {
            let f = |x: f64| (x - 1.0).powi(2);
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            // Test with tolerance near machine epsilon (but not too tight)
            let result = golden
                .golden_section_search(0.0, 2.0, Some(1e-12), Some(200))
                .unwrap();

            assert!(approx_eq!(
                f64,
                result.xmin,
                1.0,
                F64Margin {
                    epsilon: 1e-8,
                    ulps: 20
                }
            ));
            assert!(result.fmin < 1e-15);
        }
    }

    mod bracket_and_interval_tests {
        use super::*;

        #[test]
        fn test_various_bracket_sizes() {
            let f = |x: f64| (x - 5.0).powi(2) + 1.0;

            let brackets = vec![
                (4.0, 6.0),        // Small bracket around minimum
                (0.0, 10.0),       // Medium bracket
                (-100.0, 100.0),   // Large bracket
                (4.99, 5.01),      // Very small bracket
                (-1000.0, 1000.0), // Very large bracket
            ];

            for (a, b) in brackets {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);
                let result = golden.minimize(a, b).unwrap();

                assert!(
                    approx_eq!(f64, result.xmin, 5.0, LOOSE_MARGIN),
                    "Bracket [{}, {}]: expected 5.0, got {}",
                    a,
                    b,
                    result.xmin
                );
                assert!(
                    approx_eq!(f64, result.fmin, 1.0, LOOSE_MARGIN),
                    "Bracket [{}, {}]: expected 1.0, got {}",
                    a,
                    b,
                    result.fmin
                );
                assert!(result.converged);
            }
        }

        #[test]
        fn test_asymmetric_brackets() {
            let f = |x: f64| (x - 3.0).powi(2);

            let asymmetric_brackets = vec![
                (0.0, 10.0),  // Minimum closer to left
                (-5.0, 4.0),  // Minimum closer to right
                (2.9, 20.0),  // Very close to left boundary
                (-50.0, 3.1), // Very close to right boundary
            ];

            for (a, b) in asymmetric_brackets {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);
                let result = golden.minimize(a, b).unwrap();

                assert!(
                    approx_eq!(f64, result.xmin, 3.0, LOOSE_MARGIN),
                    "Asymmetric bracket [{}, {}]: expected 3.0, got {}",
                    a,
                    b,
                    result.xmin
                );
                assert!(result.converged);
            }
        }

        #[test]
        fn test_minimum_at_boundaries() {
            // Monotonically increasing function - minimum at left boundary
            let increasing = |x: f64| x;
            let objective = SingleDimFn::new(increasing);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(1.0, 5.0).unwrap();
            assert!(result.xmin <= 1.1); // Should be close to left boundary

            // Monotonically decreasing function - minimum at right boundary
            let decreasing = |x: f64| -x;
            let objective = SingleDimFn::new(decreasing);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(1.0, 5.0).unwrap();
            assert!(result.xmin >= 4.9); // Should be close to right boundary
        }
    }

    mod edge_cases_and_robustness_tests {
        use super::*;

        #[test]
        fn test_flat_regions() {
            // Function with flat region around minimum
            let flat_minimum = |x: f64| {
                if (x - 2.0).abs() < 0.1 {
                    1.0 // Flat region
                } else {
                    1.0 + (x - 2.0).abs() - 0.1
                }
            };

            let objective = SingleDimFn::new(flat_minimum);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 4.0).unwrap();

            // Should find a point in the flat region
            assert!((result.xmin - 2.0).abs() <= 0.15);
            assert!(approx_eq!(f64, result.fmin, 1.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_nearly_flat_functions() {
            // Very shallow parabola
            let shallow = |x: f64| 1e-6 * (x - 5.0).powi(2) + 100.0;
            let objective = SingleDimFn::new(shallow);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 10.0).unwrap();

            assert!(approx_eq!(f64, result.xmin, 5.0, LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin, 100.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_oscillating_functions() {
            // Function with small oscillations around main trend
            let oscillating = |x: f64| (x - 3.0).powi(2) + 0.01 * (20.0 * x).sin();
            let objective = SingleDimFn::new(oscillating);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(1.0, 5.0).unwrap();

            // Should find minimum close to 3.0 despite oscillations
            assert!((result.xmin - 3.0).abs() < 0.2);
            assert!(result.converged);
        }

        #[test]
        fn test_functions_with_discontinuous_derivatives() {
            // |x - 2| has non-differentiable point at x = 2
            let abs_func = |x: f64| (x - 2.0).abs() + 0.1;
            let objective = SingleDimFn::new(abs_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 4.0).unwrap();

            assert!(approx_eq!(f64, result.xmin, 2.0, VERY_LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin, 0.1, LOOSE_MARGIN));
        }
    }

    mod performance_and_iteration_count_tests {
        use super::*;

        #[test]
        fn test_iteration_counts() {
            let f = |x: f64| (x - 1.0).powi(2);

            // Test that iteration count increases with tighter tolerance
            let mut prev_iters = 0;
            let tolerances = vec![1e-3, 1e-6, 1e-9];

            for tol in tolerances {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);
                let result = golden
                    .golden_section_search(0.0, 2.0, Some(tol), None)
                    .unwrap();

                if prev_iters > 0 {
                    assert!(
                        result.iters >= prev_iters,
                        "Tighter tolerance should require more iterations"
                    );
                }
                prev_iters = result.iters;
            }
        }

        #[test]
        fn test_maximum_iteration_limits() {
            let f = |x: f64| (x - 0.5).powi(2);

            // Test various max iteration limits
            let max_iters = vec![1, 5, 10, 50, 100];

            for max_iter in max_iters {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);

                let result = golden.golden_section_search(0.0, 1.0, Some(1e-10), Some(max_iter));

                match result {
                    Ok(res) => {
                        assert!(res.iters <= max_iter);
                        assert!(golden.iters() <= max_iter);
                    }
                    Err(MinimizerError::MaxIterationsExceeded) => {
                        assert_eq!(golden.iters(), max_iter);
                    }
                    _ => panic!("Unexpected error type"),
                }
            }
        }
    }

    mod floating_point_precision_tests {
        use super::*;

        #[test]
        fn test_extreme_function_values() {
            // Very large function values
            let large_values = |x: f64| 1e12 * (x - 1.0).powi(2) + 1e15;
            let objective = SingleDimFn::new(large_values);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 2.0).unwrap();
            assert!(approx_eq!(f64, result.xmin, 1.0, LOOSE_MARGIN));

            // Very small function values
            let small_values = |x: f64| 1e-12 * (x - 1.0).powi(2) + 1e-15;
            let objective = SingleDimFn::new(small_values);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(0.0, 2.0).unwrap();
            assert!(approx_eq!(f64, result.xmin, 1.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_functions_near_zero() {
            // Function that approaches zero
            let near_zero = |x: f64| (x - 2.0).powi(4) + 1e-16;
            let objective = SingleDimFn::new(near_zero);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(1.0, 3.0).unwrap();

            assert!(approx_eq!(f64, result.xmin, 2.0, LOOSE_MARGIN));
            assert!(result.fmin >= 0.0); // Should not go negative due to numerical errors
        }
    }

    mod state_consistency_tests {
        use super::*;

        #[test]
        fn test_state_after_successful_optimization() {
            let f = |x: f64| (x - 3.0).powi(2) + 5.0;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            let result = golden.minimize(1.0, 5.0).unwrap();

            // Verify internal state matches result
            assert_eq!(golden.xmin(), result.xmin);
            assert_eq!(golden.fmin(), result.fmin);
            assert_eq!(golden.iters(), result.iters);
            assert!(golden.converged);
            assert!(result.converged);
        }

        #[test]
        fn test_state_after_failed_optimization() {
            let f = |x: f64| x * x;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            // This should fail due to invalid bracket
            let result = golden.minimize(2.0, 1.0);
            assert!(result.is_err());

            // State should remain unchanged
            assert!(!golden.converged);
            assert_eq!(golden.iters(), 0);
            assert_eq!(golden.xmin(), 0.0);
            assert_eq!(golden.fmin(), 0.0);
        }

        #[test]
        fn test_multiple_optimizations() {
            let mut golden = {
                let f = |x: f64| (x - 1.0).powi(2);
                let objective = SingleDimFn::new(f);
                Golden::new(objective)
            };

            // First optimization
            let result1 = golden.minimize(0.0, 2.0).unwrap();
            assert!(approx_eq!(f64, result1.xmin, 1.0, LOOSE_MARGIN));

            // Second optimization with different bracket
            let result2 = golden.minimize(-1.0, 3.0).unwrap();
            assert!(approx_eq!(f64, result2.xmin, 1.0, LOOSE_MARGIN));

            // State should reflect most recent optimization
            assert_eq!(golden.xmin(), result2.xmin);
            assert_eq!(golden.fmin(), result2.fmin);
            assert_eq!(golden.iters(), result2.iters);
        }
    }

    mod debug_format_test {
        use super::*;

        #[test]
        fn test_debug_format() {
            let f = |x: f64| x * x;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            let _ = golden.minimize(0.0, 2.0).unwrap();
            let debug_str = format!("{:?}", golden);

            // Should contain all relevant information
            assert!(debug_str.contains("Goldenf64"));
            assert!(debug_str.contains("xmin"));
            assert!(debug_str.contains("fmin"));
            assert!(debug_str.contains("iters"));
            assert!(debug_str.contains("converged"));
        }
    }
}
