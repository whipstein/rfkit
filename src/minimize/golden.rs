#![allow(dead_code)]
#![allow(unused_assignments)]
use crate::{error::MinimizerError, float::RFFloat, minimize::ObjFn};
use std::fmt;

/// Result of golden section search
#[derive(Debug, Clone)]
pub struct GoldenResult<T> {
    pub xmin: T,
    pub fmin: T,
    pub iters: usize,
    pub converged: bool,
}

pub struct Golden<T> {
    xmin: T,
    fmin: T,
    f: Box<dyn ObjFn<T>>,
    iters: usize,
    converged: bool,
}

impl<T> Clone for Golden<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            f: dyn_clone::clone_box(&*self.f),
            iters: self.iters,
            converged: self.converged,
        }
    }
}

impl<T> Golden<T>
where
    T: RFFloat,
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
    /// Golden ratio constant (φ - 1)
    const GOLDEN_RATIO: f64 = 0.618_033_988_749_895;

    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        Golden {
            xmin: T::zero(),
            fmin: T::zero(),
            f: Box::new(f),
            iters: 0,
            converged: false,
        }
    }

    pub fn new_boxed(f: Box<dyn ObjFn<T>>) -> Self {
        Golden {
            xmin: T::zero(),
            fmin: T::zero(),
            f: f,
            iters: 0,
            converged: false,
        }
    }

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
        a: &T,
        b: &T,
        tol: Option<T>,
        max_iters: Option<usize>,
    ) -> Result<GoldenResult<T>, MinimizerError> {
        self.converged = false;
        let tol = tol.unwrap_or(1e-6.into());
        let max_iter = max_iters.unwrap_or(100);

        // Validate inputs
        if a >= b {
            return Err(MinimizerError::InvalidBracket);
        }
        if tol <= T::zero() {
            return Err(MinimizerError::InvalidTolerance);
        }

        let mut x1 = a.clone();
        let mut x4 = b.clone();

        // Initial interior points using golden ratio
        let mut x2 = x1.clone() + (1.0 - Golden::<T>::GOLDEN_RATIO) * (&x4 - &x1);
        let mut x3 = x1.clone() + Golden::<T>::GOLDEN_RATIO * (&x4 - &x1);

        let mut f2 = self.f.call_scalar(&x2);
        let mut f3 = self.f.call_scalar(&x3);

        self.iters = 0;

        // Main iteration loop
        while (&x4 - &x1).abs() > tol && self.iters < max_iter {
            self.iters += 1;

            if f2 < f3 {
                // Minimum is in [x1, x3]
                x4 = x3.clone();
                x3 = x2.clone();
                f3 = f2.clone();
                x2 = x1.clone() + (1.0 - Golden::<T>::GOLDEN_RATIO) * (&x4 - &x1);
                f2 = self.f.call_scalar(&x2);
            } else {
                // Minimum is in [x2, x4]
                x1 = x2.clone();
                x2 = x3.clone();
                f2 = f3.clone();
                x3 = x1.clone() + Golden::<T>::GOLDEN_RATIO * (&x4 - &x1);
                f3 = self.f.call_scalar(&x3);
            }
        }

        if self.iters >= max_iter {
            return Err(MinimizerError::MaxIterationsExceeded);
        }

        // Return the point with smaller function value
        if f2 < f3 {
            self.xmin = x2.clone();
            self.fmin = f2.clone();
        } else {
            self.xmin = x3.clone();
            self.fmin = f3.clone();
        };

        self.converged = true;
        Ok(GoldenResult {
            xmin: self.xmin.clone(),
            fmin: self.fmin.clone(),
            iters: self.iters,
            converged: self.converged,
        })
    }

    /// Convenience function with default parameters
    pub fn minimize(&mut self, a: &T, b: &T) -> Result<GoldenResult<T>, MinimizerError> {
        self.golden_section_search(a, b, None, None)
    }

    pub fn xmin(&self) -> T {
        self.xmin.clone()
    }

    pub fn fmin(&self) -> T {
        self.fmin.clone()
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    pub fn set_xmin(&mut self, xmin: T) {
        self.xmin = xmin;
        self.fmin = self.f.call_scalar(&self.xmin);
    }
}

impl<T> fmt::Debug for Golden<T>
where
    T: RFFloat,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Goldenf64( xmin: {}, fmin: {}, iters: {}, converged: {})",
            self.xmin, self.fmin, self.iters, self.converged
        )
    }
}

#[cfg(test)]
mod minimize_golden_tests {
    use super::*;
    use crate::{
        minimize::{F1dim, SingleDimFn},
        myfloat::MyFloat,
    };
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
        let f = |x: &MyFloat| (x - 2.0).powi(2);
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden.minimize(&0.0.into(), &5.0.into()).unwrap();

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
        let f = |x: &MyFloat| x.sin();
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden.minimize(&3.0.into(), &6.0.into()).unwrap();

        assert!((result.xmin - 1.5 * PI).abs() < 1e-4);
        assert!((result.fmin - (-1.0)).abs() < 1e-6);
        assert!((golden.xmin - 1.5 * PI).abs() < 1e-4);
        assert!((golden.fmin - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_custom_tolerance() {
        let f = |x: &MyFloat| (x - 1.0).powi(2);
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden
            .golden_section_search(&0.0.into(), &2.0.into(), Some(1e-10.into()), None)
            .unwrap();

        assert!((result.xmin - 1.0).abs() < 1e-9);
        assert!((golden.xmin - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_invalid_bracket() {
        let f = |x: &MyFloat| x * x;
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result = golden.minimize(&2.0.into(), &1.0.into());

        assert!(matches!(result, Err(MinimizerError::InvalidBracket)));
        assert!(!golden.converged);
    }

    #[test]
    fn test_invalid_tolerance() {
        let f = |x: &MyFloat| x * x;
        let objective = SingleDimFn::new(f);
        let mut golden = Golden::new(objective);

        let result =
            golden.golden_section_search(&0.0.into(), &1.0.into(), Some(MyFloat::new(-1.0)), None);

        assert!(matches!(result, Err(MinimizerError::InvalidTolerance)));
        assert!(!golden.converged);
    }

    #[test]
    fn test_golden_section_unimodal_functions() {
        // Test 1: Simple quadratic
        let quadratic = |x: &MyFloat| (x - 3.0).powi(2) + 2.0;
        let objective = SingleDimFn::new(quadratic);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(&0.0.into(), &6.0.into(), Some(1e-8.into()), None)
            .unwrap();
        assert!((result.xmin - 3.0).abs() < 1e-6);
        assert!((result.fmin - 2.0).abs() < 1e-6);

        // Test 2: Quartic function
        let quartic = |x: &MyFloat| (x - 1.5).powi(4) + 0.5;
        let objective = SingleDimFn::new(quartic);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(&MyFloat::new(-1.0), &4.0.into(), Some(1e-8.into()), None)
            .unwrap();
        assert!((result.xmin - 1.5).abs() < 1e-4);
        assert!((result.fmin - 0.5).abs() < 1e-6);

        // Test 3: Transcendental function
        let transcendental = |x: &MyFloat| x * x + (2.0 * x).sin();
        let objective = SingleDimFn::new(transcendental);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(&MyFloat::new(-2.0), &2.0.into(), Some(1e-6.into()), None)
            .unwrap();
        assert!(result.converged);
        assert!(result.fmin < 0.0); // Should find negative minimum
    }

    #[test]
    fn test_golden_section_edge_cases() {
        // Test 1: Very narrow bracket
        let func = |x: &MyFloat| (x - 0.5).powi(2);
        let objective = SingleDimFn::new(func);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(&0.49.into(), &0.51.into(), Some(1e-10.into()), None)
            .unwrap();
        assert!((result.xmin - 0.5).abs() < 1e-8);

        // Test 2: Flat function
        let flat = |x: &MyFloat| {
            if (x - 1.0).abs() < 0.1 {
                0.0.into()
            } else {
                (x - 1.0).abs()
            }
        };
        let objective = SingleDimFn::new(flat);
        let mut golden = Golden::new(objective);
        let result = golden
            .golden_section_search(&0.0.into(), &2.0.into(), Some(1e-4.into()), None)
            .unwrap();
        assert!((result.xmin - 1.0).abs() < 0.2);

        // Test 3: Maximum iterations
        let difficult = |x: &MyFloat| (x - PI).powi(2) + 1e-10 * (100.0 * x).sin();
        let objective = SingleDimFn::new(difficult);
        let mut golden = Golden::new(objective);
        let result = golden.golden_section_search(
            &0.0.into(),
            &(2.0 * PI).into(),
            Some(1e-15.into()),
            Some(10),
        );
        assert!(result.is_err()); // Should not crash even with few iterations
    }

    mod basic_functionality_tests {
        use super::*;

        #[test]
        fn test_construction_and_initialization() {
            let f = |x: &MyFloat| x * x;
            let objective = SingleDimFn::new(f);
            let golden = Golden::new(objective);

            assert_eq!(golden.xmin(), 0.0);
            assert_eq!(golden.fmin(), 0.0);
            assert_eq!(golden.iters(), 0);
            assert!(!golden.converged);
        }

        #[test]
        fn test_boxed_construction() {
            let f = |x: &MyFloat| x * x;
            let objective = SingleDimFn::new(f);
            let boxed_obj: Box<dyn ObjFn<MyFloat>> = Box::new(objective);
            let golden = Golden::new_boxed(boxed_obj);

            assert_eq!(golden.xmin(), 0.0);
            assert_eq!(golden.fmin(), 0.0);
        }

        #[test]
        fn test_set_xmin_updates_fmin() {
            let f = |x: &MyFloat| x * x + 2.0;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            golden.set_xmin(3.0.into());
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
                    F1dim::new(SingleDimFn::new(|x: &MyFloat| (x - 0.0).powi(2))),
                    0.0.into(),
                    MyFloat::new(0.0),
                    MyFloat::new(-2.0),
                    2.0.into(),
                ),
                (
                    F1dim::new(SingleDimFn::new(|x: &MyFloat| (x - 1.0).powi(2))),
                    1.0.into(),
                    0.0.into(),
                    MyFloat::new(-1.0),
                    3.0.into(),
                ),
                (
                    F1dim::new(SingleDimFn::new(|x: &MyFloat| (x - 5.0).powi(2) + 3.0)),
                    5.0.into(),
                    3.0.into(),
                    2.0.into(),
                    8.0.into(),
                ),
                (
                    F1dim::new(SingleDimFn::new(|x: &MyFloat| {
                        2.0 * (x - 2.5).powi(2) + 1.5
                    })),
                    2.5.into(),
                    1.5.into(),
                    0.0.into(),
                    5.0.into(),
                ),
                (
                    F1dim::new(SingleDimFn::new(|x: &MyFloat| (x + 3.0).powi(2))),
                    MyFloat::new(-3.0),
                    0.0.into(),
                    MyFloat::new(-5.0),
                    MyFloat::new(-1.0),
                ),
            ];

            for (i, (objective, expected_x, expected_f, a, b)) in test_cases.iter().enumerate() {
                let mut golden = Golden::new(objective.clone());
                let result = golden.minimize(a, b).unwrap();

                assert!(
                    approx_eq!(f64, result.xmin.to_f64(), expected_x.to_f64(), LOOSE_MARGIN),
                    "Test case {}: xmin expected {}, got {}",
                    i,
                    expected_x,
                    result.xmin
                );
                assert!(
                    approx_eq!(f64, result.fmin.to_f64(), expected_f.to_f64(), LOOSE_MARGIN),
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
            let quartic = |x: &MyFloat| (x - 2.0).powi(4) + 1.0;
            let objective = SingleDimFn::new(quartic);
            let mut golden = Golden::new(objective);
            let result = golden
                .golden_section_search(&0.0.into(), &4.0.into(), Some(1e-4.into()), None)
                .unwrap();
            assert!(
                (&result.xmin - 2.0).abs() < 0.05,
                "Quartic: expected 2.0, got {}",
                result.xmin
            );
            assert!(approx_eq!(f64, result.fmin.into(), 1.0, LOOSE_MARGIN));

            // Simpler test: x^4 with minimum at x = 0
            let simple_quartic = |x: &MyFloat| x.powi(4) + 0.5;
            let objective = SingleDimFn::new(simple_quartic);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&MyFloat::new(-2.0), &2.0.into()).unwrap();
            assert!(
                (result.xmin).abs() < 0.01,
                "Simple quartic: expected 0.0, got {}",
                result.xmin
            );
            assert!(approx_eq!(f64, result.fmin.into(), 0.5, LOOSE_MARGIN));

            // Sixth degree: (x - 1)^6 + 0.5 - easier to minimize
            let sixth = |x: &MyFloat| (x - 1.0).powi(6) + 0.5;
            let objective = SingleDimFn::new(sixth);
            let mut golden = Golden::new(objective);
            let result = golden
                .golden_section_search(&MyFloat::new(-1.0), &3.0.into(), Some(1e-4.into()), None)
                .unwrap();
            assert!(
                (&result.xmin - 1.0).abs() < 0.05,
                "Sixth degree: expected 1.0, got {}",
                result.xmin
            );
            assert!(approx_eq!(f64, result.fmin.into(), 0.5, LOOSE_MARGIN));
        }

        #[test]
        fn test_trigonometric_functions() {
            // Test sin(x) in [π, 2π], minimum at 3π/2
            let sin_func = |x: &MyFloat| x.sin();
            let objective = SingleDimFn::new(sin_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&PI.into(), &(2.0 * PI).into()).unwrap();
            assert!(approx_eq!(f64, result.xmin.into(), 1.5 * PI, LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin.into(), -1.0, LOOSE_MARGIN));

            // Test cos(x) in [0, π], minimum at π
            let cos_func = |x: &MyFloat| x.cos();
            let objective = SingleDimFn::new(cos_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &PI.into()).unwrap();
            assert!(approx_eq!(f64, result.xmin.into(), PI, LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin.into(), -1.0, LOOSE_MARGIN));

            // Test sin^2(x) + cos^2(x) = 1 (constant function)
            let trig_identity = |x: &MyFloat| x.sin().powi(2) + x.cos().powi(2);
            let objective = SingleDimFn::new(trig_identity);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &(2.0 * PI).into()).unwrap();
            assert!(approx_eq!(f64, result.fmin.into(), 1.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_exponential_and_logarithmic_functions() {
            // e^(x-1) has minimum at negative infinity, but in [0,2] minimum at x=0
            let exp_func = |x: &MyFloat| E.powf((x - 1.0).to_f64()).into();
            let objective = SingleDimFn::new(exp_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &2.0.into()).unwrap();
            assert!(result.xmin < 0.1); // Should be close to 0
            assert!(approx_eq!(
                f64,
                result.fmin.into(),
                E.powf(-1.0),
                LOOSE_MARGIN
            ));

            // Test a function that actually has an interior minimum
            // f(x) = x^2 - ln(x) has minimum at x = 1/√2 since f'(x) = 2x - 1/x = 0
            let log_func = |x: &MyFloat| x * x - x.ln(); // Note: minus ln(x)
            let objective = SingleDimFn::new(log_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.1.into(), &2.0.into()).unwrap();
            let expected_min = 1.0 / 2.0_f64.sqrt(); // ≈ 0.7071
            assert!(
                (&result.xmin - expected_min).abs() < 0.1,
                "Expected around {}, got {}",
                expected_min,
                result.xmin
            );
            assert!(result.converged);
        }

        #[test]
        fn test_absolute_value_functions() {
            // |x - 3|, minimum at x = 3
            let abs_func = |x: &MyFloat| (x - 3.0).abs();
            let objective = SingleDimFn::new(abs_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &6.0.into()).unwrap();
            assert!(approx_eq!(f64, result.xmin.into(), 3.0, LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin.into(), 0.0, LOOSE_MARGIN));

            // |x - 2| + |x - 4|, minimum in [2, 4]
            let double_abs = |x: &MyFloat| (x - 2.0).abs() + (x - 4.0).abs();
            let objective = SingleDimFn::new(double_abs);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &6.0.into()).unwrap();
            assert!(result.xmin >= 1.9 && result.xmin <= 4.1);
            assert!(approx_eq!(f64, result.fmin.into(), 2.0, LOOSE_MARGIN));
        }
    }

    mod error_condition_tests {
        use super::*;

        #[test]
        fn test_invalid_bracket_conditions() {
            let f = |x: &MyFloat| x * x;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            // Equal brackets
            assert!(matches!(
                golden.minimize(&1.0.into(), &1.0.into()),
                Err(MinimizerError::InvalidBracket)
            ));

            // Reversed brackets
            assert!(matches!(
                golden.minimize(&2.0.into(), &1.0.into()),
                Err(MinimizerError::InvalidBracket)
            ));

            // Ensure state remains unchanged after error
            assert!(!golden.converged);
            assert_eq!(golden.iters(), 0);
        }

        #[test]
        fn test_invalid_tolerance_conditions() {
            let f = |x: &MyFloat| x * x;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            // Zero tolerance
            assert!(matches!(
                golden.golden_section_search(&0.0.into(), &1.0.into(), Some(0.0.into()), None),
                Err(MinimizerError::InvalidTolerance)
            ));

            // Negative tolerance
            assert!(matches!(
                golden.golden_section_search(
                    &0.0.into(),
                    &1.0.into(),
                    Some(MyFloat::new(-1e-6)),
                    None
                ),
                Err(MinimizerError::InvalidTolerance)
            ));

            // Very small negative tolerance
            assert!(matches!(
                golden.golden_section_search(
                    &0.0.into(),
                    &1.0.into(),
                    Some(MyFloat::new(-1e-15)),
                    None
                ),
                Err(MinimizerError::InvalidTolerance)
            ));
        }

        #[test]
        fn test_max_iterations_exceeded() {
            // Use a very tight tolerance and few iterations to force failure
            let f = |x: &MyFloat| (x - 0.5).powi(2);
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            let result =
                golden.golden_section_search(&0.0.into(), &1.0.into(), Some(1e-15.into()), Some(5));
            assert!(matches!(result, Err(MinimizerError::MaxIterationsExceeded)));
            assert!(!golden.converged);
            assert_eq!(golden.iters(), 5);
        }
    }

    mod convergence_and_precision_tests {
        use super::*;

        #[test]
        fn test_various_tolerance_levels() {
            let f = |x: &MyFloat| (x - PI).powi(2) + E;

            let tolerances = vec![1e-3, 1e-6, 1e-9, 1e-12];

            for &tol in &tolerances {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);
                let result = golden
                    .golden_section_search(&0.0.into(), &(2.0 * PI).into(), Some(tol.into()), None)
                    .unwrap();

                // Error should be roughly proportional to tolerance
                let error = (&result.xmin - PI).abs();
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
            let f = |x: &MyFloat| (x - 2.0).powi(4); // Fourth-order function
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            let result = golden.minimize(&0.0.into(), &4.0.into()).unwrap();

            // Golden section should converge reasonably quickly
            assert!(
                result.iters < 50,
                "Should converge in reasonable iterations, got {}",
                result.iters
            );
            assert!(result.converged);

            // Check that we actually found the minimum
            assert!(approx_eq!(f64, result.xmin.into(), 2.0, LOOSE_MARGIN));
            assert!(result.fmin < 1e-10);
        }

        #[test]
        fn test_near_machine_precision() {
            let f = |x: &MyFloat| (x - 1.0).powi(2);
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            // Test with tolerance near machine epsilon (but not too tight)
            let result = golden
                .golden_section_search(&0.0.into(), &2.0.into(), Some(1e-12.into()), Some(200))
                .unwrap();

            assert!(approx_eq!(
                f64,
                result.xmin.into(),
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
            let f = |x: &MyFloat| (x - 5.0).powi(2) + 1.0;

            let brackets = vec![
                (4.0.into(), MyFloat::new(6.0)),      // Small bracket around minimum
                (0.0.into(), 10.0.into()),            // Medium bracket
                (MyFloat::new(-100.0), 100.0.into()), // Large bracket
                (4.99.into(), 5.01.into()),           // Very small bracket
                (MyFloat::new(-1000.0), 1000.0.into()), // Very large bracket
            ];

            for (a, b) in brackets {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);
                let result = golden.minimize(&a, &b).unwrap();

                assert!(
                    approx_eq!(f64, result.xmin.to_f64(), 5.0, LOOSE_MARGIN),
                    "Bracket [{}, {}]: expected 5.0, got {}",
                    a,
                    b,
                    result.xmin
                );
                assert!(
                    approx_eq!(f64, result.fmin.to_f64(), 1.0, LOOSE_MARGIN),
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
            let f = |x: &MyFloat| (x - 3.0).powi(2);

            let asymmetric_brackets = vec![
                (0.0.into(), MyFloat::new(10.0)),  // Minimum closer to left
                (MyFloat::new(-5.0), 4.0.into()),  // Minimum closer to right
                (2.9.into(), 20.0.into()),         // Very close to left boundary
                (MyFloat::new(-50.0), 3.1.into()), // Very close to right boundary
            ];

            for (a, b) in asymmetric_brackets {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);
                let result = golden.minimize(&a, &b).unwrap();

                assert!(
                    approx_eq!(f64, result.xmin.to_f64(), 3.0, LOOSE_MARGIN),
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
            let increasing = |x: &MyFloat| x.clone();
            let objective = SingleDimFn::new(increasing);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&1.0.into(), &5.0.into()).unwrap();
            assert!(result.xmin <= 1.1); // Should be close to left boundary

            // Monotonically decreasing function - minimum at right boundary
            let decreasing = |x: &MyFloat| -x;
            let objective = SingleDimFn::new(decreasing);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&1.0.into(), &5.0.into()).unwrap();
            assert!(result.xmin >= 4.9); // Should be close to right boundary
        }
    }

    mod edge_cases_and_robustness_tests {
        use super::*;

        #[test]
        fn test_flat_regions() {
            // Function with flat region around minimum
            let flat_minimum = |x: &MyFloat| {
                if (x - 2.0).abs() < 0.1 {
                    1.0.into() // Flat region
                } else {
                    1.0 + (x - 2.0).abs() - 0.1
                }
            };

            let objective = SingleDimFn::new(flat_minimum);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &4.0.into()).unwrap();

            // Should find a point in the flat region
            assert!((result.xmin - 2.0).abs() <= 0.15);
            assert!(approx_eq!(f64, result.fmin.into(), 1.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_nearly_flat_functions() {
            // Very shallow parabola
            let shallow = |x: &MyFloat| 1e-6 * (x - 5.0).powi(2) + 100.0;
            let objective = SingleDimFn::new(shallow);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &10.0.into()).unwrap();

            assert!(approx_eq!(f64, result.xmin.into(), 5.0, LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin.into(), 100.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_oscillating_functions() {
            // Function with small oscillations around main trend
            let oscillating = |x: &MyFloat| (x - 3.0).powi(2) + 0.01 * (20.0 * x).sin();
            let objective = SingleDimFn::new(oscillating);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&1.0.into(), &5.0.into()).unwrap();

            // Should find minimum close to 3.0 despite oscillations
            assert!((result.xmin - 3.0).abs() < 0.2);
            assert!(result.converged);
        }

        #[test]
        fn test_functions_with_discontinuous_derivatives() {
            // |x - 2| has non-differentiable point at x = 2
            let abs_func = |x: &MyFloat| (x - 2.0).abs() + 0.1;
            let objective = SingleDimFn::new(abs_func);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &4.0.into()).unwrap();

            assert!(approx_eq!(f64, result.xmin.into(), 2.0, VERY_LOOSE_MARGIN));
            assert!(approx_eq!(f64, result.fmin.into(), 0.1, LOOSE_MARGIN));
        }
    }

    mod performance_and_iteration_count_tests {
        use super::*;

        #[test]
        fn test_iteration_counts() {
            let f = |x: &MyFloat| (x - 1.0).powi(2);

            // Test that iteration count increases with tighter tolerance
            let mut prev_iters = 0;
            let tolerances = vec![1e-3, 1e-6, 1e-9];

            for tol in tolerances {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);
                let result = golden
                    .golden_section_search(&0.0.into(), &2.0.into(), Some(tol.into()), None)
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
            let f = |x: &MyFloat| (x - 0.5).powi(2);

            // Test various max iteration limits
            let max_iters = vec![1, 5, 10, 50, 100];

            for max_iter in max_iters {
                let objective = SingleDimFn::new(f);
                let mut golden = Golden::new(objective);

                let result = golden.golden_section_search(
                    &0.0.into(),
                    &1.0.into(),
                    Some(1e-10.into()),
                    Some(max_iter),
                );

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
            let large_values = |x: &MyFloat| 1e12 * (x - 1.0).powi(2) + 1e15;
            let objective = SingleDimFn::new(large_values);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &2.0.into()).unwrap();
            assert!(approx_eq!(f64, result.xmin.into(), 1.0, LOOSE_MARGIN));

            // Very small function values
            let small_values = |x: &MyFloat| 1e-12 * (x - 1.0).powi(2) + 1e-15;
            let objective = SingleDimFn::new(small_values);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&0.0.into(), &2.0.into()).unwrap();
            assert!(approx_eq!(f64, result.xmin.into(), 1.0, LOOSE_MARGIN));
        }

        #[test]
        fn test_functions_near_zero() {
            // Function that approaches zero
            let near_zero = |x: &MyFloat| (x - 2.0).powi(4) + 1e-16;
            let objective = SingleDimFn::new(near_zero);
            let mut golden = Golden::new(objective);
            let result = golden.minimize(&1.0.into(), &3.0.into()).unwrap();

            assert!(approx_eq!(f64, result.xmin.into(), 2.0, LOOSE_MARGIN));
            assert!(result.fmin >= 0.0); // Should not go negative due to numerical errors
        }
    }

    mod state_consistency_tests {
        use super::*;

        #[test]
        fn test_state_after_successful_optimization() {
            let f = |x: &MyFloat| (x - 3.0).powi(2) + 5.0;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            let result = golden.minimize(&1.0.into(), &5.0.into()).unwrap();

            // Verify internal state matches result
            assert_eq!(golden.xmin(), result.xmin);
            assert_eq!(golden.fmin(), result.fmin);
            assert_eq!(golden.iters(), result.iters);
            assert!(golden.converged);
            assert!(result.converged);
        }

        #[test]
        fn test_state_after_failed_optimization() {
            let f = |x: &MyFloat| x * x;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            // This should fail due to invalid bracket
            let result = golden.minimize(&2.0.into(), &1.0.into());
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
                let f = |x: &MyFloat| (x - 1.0).powi(2);
                let objective = SingleDimFn::new(f);
                Golden::new(objective)
            };

            // First optimization
            let result1 = golden.minimize(&0.0.into(), &2.0.into()).unwrap();
            assert!(approx_eq!(f64, result1.xmin.to_f64(), 1.0, LOOSE_MARGIN));

            // Second optimization with different bracket
            let result2 = golden.minimize(&MyFloat::new(-1.0), &3.0.into()).unwrap();
            assert!(approx_eq!(f64, result2.xmin.to_f64(), 1.0, LOOSE_MARGIN));

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
            let f = |x: &MyFloat| x * x;
            let objective = SingleDimFn::new(f);
            let mut golden = Golden::new(objective);

            let _ = golden.minimize(&0.0.into(), &2.0.into()).unwrap();
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
