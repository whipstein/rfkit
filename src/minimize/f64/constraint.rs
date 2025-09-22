use dyn_clone::DynClone;
use ndarray::prelude::*;
use std::fmt;

/// Constraint definition
pub trait Constraint: DynClone {
    fn evaluate(&self, x: &Array1<f64>) -> f64;
    fn gradient(&self, x: &Array1<f64>) -> Array1<f64>;
    fn hessian(&self, x: &Array1<f64>) -> Array2<f64>;
}
dyn_clone::clone_trait_object!(Constraint);

/// Linear constraint: a^T x + b ≤ 0 (inequality) or = 0 (equality)
#[derive(Clone)]
pub struct LinearConstraint {
    pub a: Array1<f64>,
    pub b: f64,
    pub is_equality: bool,
}

impl LinearConstraint {
    pub fn new(a: Array1<f64>, b: f64, is_equality: bool) -> Self {
        Self { a, b, is_equality }
    }

    pub fn inequality(a: Array1<f64>, b: f64) -> Self {
        Self::new(a, b, false)
    }

    pub fn equality(a: Array1<f64>, b: f64) -> Self {
        Self::new(a, b, true)
    }
}

impl Constraint for LinearConstraint {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        self.a
            .iter()
            .zip(x.iter())
            .map(|(&ai, &xi)| ai * xi)
            .sum::<f64>()
            + self.b
    }

    fn gradient(&self, _x: &Array1<f64>) -> Array1<f64> {
        self.a.clone()
    }

    fn hessian(&self, x: &Array1<f64>) -> Array2<f64> {
        let n = x.len();
        Array2::zeros((n, n))
    }
}

impl fmt::Debug for LinearConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.is_equality {
            true => write!(
                f,
                "LinearConstraint( {:?}^T * x + {:?} = 0)",
                self.a, self.b
            ),
            false => write!(
                f,
                "LinearConstraint( {:?}^T * x + {:?} ≤ 0)",
                self.a, self.b
            ),
        }
    }
}

/// Quadratic constraint: x^T Q x + a^T x + b ≤ 0 or = 0
#[derive(Clone)]
pub struct QuadraticConstraint {
    pub q: Array2<f64>,
    pub a: Array1<f64>,
    pub b: f64,
    pub is_equality: bool,
}

impl QuadraticConstraint {
    pub fn new(q: Array2<f64>, a: Array1<f64>, b: f64, is_equality: bool) -> Self {
        Self {
            q,
            a,
            b,
            is_equality,
        }
    }
}

impl Constraint for QuadraticConstraint {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        let n = x.len();
        let mut result = self.b;

        // Linear term: a^T x
        for i in 0..n {
            result += self.a[i] * x[i];
        }

        // Quadratic term: x^T Q x
        for i in 0..n {
            for j in 0..n {
                result += x[i] * self.q[[i, j]] * x[j];
            }
        }

        result
    }

    fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut grad = self.a.clone();

        // Add 2 * Q * x (assuming Q is symmetric)
        for i in 0..n {
            for j in 0..n {
                grad[i] += 2.0 * self.q[[i, j]] * x[j];
            }
        }

        grad
    }

    fn hessian(&self, _x: &Array1<f64>) -> Array2<f64> {
        // Hessian is 2*Q for quadratic constraint
        let n = self.q.nrows();
        let mut hess = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                hess[[i, j]] = 2.0 * self.q[[i, j]];
            }
        }

        hess
    }
}

impl fmt::Debug for QuadraticConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.is_equality {
            true => write!(
                f,
                "QuadraticConstraint( x^T * {:?} * x + {:?}^T * x + {:?} = 0)",
                self.q, self.a, self.b
            ),
            false => write!(
                f,
                "QuadraticConstraint( x^T * {:?} * x + {:?}^T * x + {:?} ≤ 0)",
                self.q, self.a, self.b
            ),
        }
    }
}

/// Create box constraints: l ≤ x ≤ u
pub fn create_box_constraints(
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> Vec<Box<dyn Constraint>> {
    let mut constraints = Vec::new();

    for (i, (&l, &u)) in lower.iter().zip(upper.iter()).enumerate() {
        // x_i ≥ l becomes -x_i + l ≤ 0
        if l.is_finite() {
            let mut a = Array1::zeros(lower.len());
            a[i] = -1.0;
            constraints.push(Box::new(LinearConstraint::inequality(a, l)) as Box<dyn Constraint>);
        }

        // x_i ≤ u becomes x_i - u ≤ 0
        if u.is_finite() {
            let mut a = Array1::zeros(lower.len());
            a[i] = 1.0;
            constraints.push(Box::new(LinearConstraint::inequality(a, -u)) as Box<dyn Constraint>);
        }
    }

    constraints
}

#[cfg(test)]
mod minimize_f64_constraint_tests {
    use super::*;
    use float_cmp::*;

    #[test]
    fn test_linear_constraint() {
        let constraint = LinearConstraint::inequality(array![1.0, 1.0], -1.0);
        let x = array![0.5, 0.3];

        // Test constraint: x1 + x2 - 1 ≤ 0
        approx_eq!(f64, constraint.evaluate(&x), -0.2); // 0.5 + 0.3 - 1 = -0.2
        assert_eq!(constraint.gradient(&x), array![1.0, 1.0]);

        let hess = constraint.hessian(&x);
        assert_eq!(hess, Array2::<f64>::zeros((2, 2)));
    }

    #[test]
    fn test_quadratic_constraint() {
        let q = Array2::eye(2) * 2.0;
        let a = array![0.0, 0.0];
        let constraint = QuadraticConstraint::new(q, a, -1.0, false);
        let x = array![0.5, 0.5];

        // Test constraint: x1² + x2² - 1 ≤ 0
        assert!((constraint.evaluate(&x)).abs() < 1e-10); // 1.0 + 0.0 - 1 = 0.0
        assert_eq!(constraint.gradient(&x), array![2.0, 2.0]); // [4*0.5, 4*0.5]

        let hess = constraint.hessian(&x);
        assert_eq!(hess, Array2::<f64>::eye(2) * 4.0);
    }

    #[test]
    fn test_box_constraints() {
        let lower = array![0.0, -1.0];
        let upper = array![2.0, 1.0];
        let constraints = create_box_constraints(&lower, &upper);

        assert_eq!(constraints.len(), 4); // 2 lower + 2 upper bounds

        let x = array![1.0, 0.0];

        // Test that point satisfies all constraints
        for constraint in &constraints {
            assert!(
                constraint.evaluate(&x) <= 1e-10,
                "Constraint violated: {}",
                constraint.evaluate(&x)
            );
        }

        // Test boundary point
        let x_boundary = array![0.0, 1.0];
        for constraint in &constraints {
            assert!(
                constraint.evaluate(&x_boundary) <= 1e-10,
                "Boundary constraint violated: {}",
                constraint.evaluate(&x_boundary)
            );
        }
    }

    // Test data generators
    fn create_identity_matrix(n: usize) -> Array2<f64> {
        let mut matrix = Array2::zeros((n, n));
        for i in 0..n {
            matrix[[i, i]] = 1.0;
        }
        matrix
    }

    fn create_symmetric_matrix(values: &[(usize, usize, f64)]) -> Array2<f64> {
        let max_idx = values
            .iter()
            .map(|(i, j, _)| (*i).max(*j))
            .max()
            .unwrap_or(0);
        let n = max_idx + 1;
        let mut matrix = Array2::zeros((n, n));

        for &(i, j, val) in values {
            matrix[[i, j]] = val;
            if i != j {
                matrix[[j, i]] = val; // Ensure symmetry
            }
        }
        matrix
    }

    // Linear Constraint Tests
    mod linear_constraint_tests {
        use super::*;

        #[test]
        fn test_linear_inequality_basic() {
            let constraint = LinearConstraint::inequality(array![1.0, 1.0], -1.0);
            let x = array![0.5, 0.3];

            // Test constraint: x1 + x2 - 1 ≤ 0
            approx_eq!(f64, constraint.evaluate(&x), -0.2);
            assert_eq!(constraint.gradient(&x), array![1.0, 1.0]);
            assert_eq!(constraint.hessian(&x), array![[0.0, 0.0], [0.0, 0.0]]);
            assert!(!constraint.is_equality);
        }

        #[test]
        fn test_linear_equality_basic() {
            let constraint = LinearConstraint::equality(array![2.0, -1.0], 3.0);
            let x = array![2.0, 1.0];

            // Test constraint: 2*x1 - x2 + 3 = 0
            approx_eq!(f64, constraint.evaluate(&x), 6.0); // 2*2 - 1 + 3 = 6
            assert_eq!(constraint.gradient(&x), array![2.0, -1.0]);
            assert!(constraint.is_equality);
        }

        #[test]
        fn test_linear_constraint_zero_coefficients() {
            let constraint = LinearConstraint::inequality(array![0.0, 0.0, 1.0], -5.0);
            let x = array![10.0, 20.0, 3.0];

            // Test constraint: 0*x1 + 0*x2 + 1*x3 - 5 ≤ 0
            approx_eq!(f64, constraint.evaluate(&x), -2.0); // 0 + 0 + 3 - 5 = -2
            assert_eq!(constraint.gradient(&x), array![0.0, 0.0, 1.0]);
        }

        #[test]
        fn test_linear_constraint_negative_coefficients() {
            let constraint = LinearConstraint::inequality(array![-2.0, -3.0], 10.0);
            let x = array![1.0, 2.0];

            // Test constraint: -2*x1 - 3*x2 + 10 ≤ 0
            approx_eq!(f64, constraint.evaluate(&x), 2.0); // -2 - 6 + 10 = 2
        }

        #[test]
        fn test_linear_constraint_single_variable() {
            let constraint = LinearConstraint::inequality(array![5.0], -10.0);
            let x = array![3.0];

            approx_eq!(f64, constraint.evaluate(&x), 5.0); // 5*3 - 10 = 5
            assert_eq!(constraint.gradient(&x), array![5.0]);
            assert_eq!(constraint.hessian(&x), array![[0.0]]);
        }

        #[test]
        fn test_linear_constraint_large_dimension() {
            let n = 100;
            let a: Array1<f64> = (0..n).map(|i| i as f64).collect();
            let x: Array1<f64> = Array1::ones(n);
            let constraint = LinearConstraint::inequality(a.clone(), -50.0);

            let expected = (0..n).sum::<usize>() as f64 - 50.0; // Sum of 0..99 = 4950
            approx_eq!(f64, constraint.evaluate(&x), expected);
            assert_eq!(constraint.gradient(&x), a);
        }

        #[test]
        fn test_linear_constraint_at_boundary() {
            let constraint = LinearConstraint::inequality(array![1.0, 1.0], -2.0);
            let x = array![1.0, 1.0]; // This should make constraint exactly 0

            approx_eq!(f64, constraint.evaluate(&x), 0.0);
        }

        #[test]
        fn test_linear_constraint_debug_format() {
            let ineq = LinearConstraint::inequality(array![1.0, -2.0], 3.0);
            let eq = LinearConstraint::equality(array![0.5, 1.5], -1.0);

            let ineq_debug = format!("{:?}", ineq);
            let eq_debug = format!("{:?}", eq);

            assert!(ineq_debug.contains("≤"));
            assert!(eq_debug.contains("="));
            assert!(ineq_debug.contains("[1.0, -2.0]"));
            assert!(eq_debug.contains("[0.5, 1.5]"));
        }
    }

    // Quadratic Constraint Tests
    mod quadratic_constraint_tests {
        use super::*;

        #[test]
        fn test_quadratic_constraint_identity_matrix() {
            let q = create_identity_matrix(2);
            let a = array![0.0, 0.0];
            let constraint = QuadraticConstraint::new(q, a, -1.0, false);
            let x = array![0.5, 0.5];

            // Test constraint: x1² + x2² - 1 ≤ 0
            approx_eq!(f64, constraint.evaluate(&x), -0.5); // 0.25 + 0.25 - 1 = -0.5
            assert_eq!(constraint.gradient(&x), array![1.0, 1.0]); // [2*0.5, 2*0.5]

            let hess = constraint.hessian(&x);
            assert_eq!(hess, array![[2.0, 0.0], [0.0, 2.0]]);
        }

        #[test]
        fn test_quadratic_constraint_with_linear_term() {
            let q = array![[1.0, 0.0], [0.0, 1.0]];
            let a = array![2.0, -1.0];
            let constraint = QuadraticConstraint::new(q, a, 3.0, true);
            let x = array![1.0, 2.0];

            // Test constraint: x1² + x2² + 2*x1 - x2 + 3 = 0
            let expected = 1.0 + 4.0 + 2.0 - 2.0 + 3.0; // = 8
            approx_eq!(f64, constraint.evaluate(&x), expected);

            // Gradient: [2*x1 + 2, 2*x2 - 1] = [4, 3]
            assert_eq!(constraint.gradient(&x), array![4.0, 3.0]);
            assert!(constraint.is_equality);
        }

        #[test]
        fn test_quadratic_constraint_cross_terms() {
            let q = array![[1.0, 0.5], [0.5, 2.0]];
            let a = array![0.0, 0.0];
            let constraint = QuadraticConstraint::new(q, a, 0.0, false);
            let x = array![1.0, 1.0];

            // Test constraint: x1² + x1*x2 + 2*x2² ≤ 0
            let expected = 1.0 + 1.0 + 2.0; // = 4
            approx_eq!(f64, constraint.evaluate(&x), expected);

            // Gradient: [2*x1 + x2, x1 + 4*x2] = [3, 5]
            assert_eq!(constraint.gradient(&x), array![3.0, 5.0]);
        }

        #[test]
        fn test_quadratic_constraint_zero_quadratic_term() {
            let q = array![[0.0, 0.0], [0.0, 0.0]];
            let a = array![3.0, -2.0];
            let constraint = QuadraticConstraint::new(q, a, 1.0, false);
            let x = array![2.0, 1.0];

            // This should behave like a linear constraint: 3*x1 - 2*x2 + 1 ≤ 0
            approx_eq!(f64, constraint.evaluate(&x), 5.0); // 6 - 2 + 1 = 5
            assert_eq!(constraint.gradient(&x), array![3.0, -2.0]);

            let hess = constraint.hessian(&x);
            assert_eq!(hess, array![[0.0, 0.0], [0.0, 0.0]]);
        }

        #[test]
        fn test_quadratic_constraint_single_variable() {
            let q = array![[2.0]];
            let a = array![-4.0];
            let constraint = QuadraticConstraint::new(q, a, 1.0, false);
            let x = array![3.0];

            // Test constraint: 2*x² - 4*x + 1 ≤ 0
            let expected = 2.0 * 9.0 - 4.0 * 3.0 + 1.0; // = 18 - 12 + 1 = 7
            approx_eq!(f64, constraint.evaluate(&x), expected);

            // Gradient: [4*x - 4] = [8]
            assert_eq!(constraint.gradient(&x), array![8.0]);

            let hess = constraint.hessian(&x);
            assert_eq!(hess, array![[4.0]]);
        }

        #[test]
        fn test_quadratic_constraint_large_dimension() {
            let n = 10;
            let q = create_identity_matrix(n);
            let a = Array1::zeros(n);
            let constraint = QuadraticConstraint::new(q, a, -5.0, false);
            let x = Array1::ones(n);

            // Test constraint: sum(xi²) - 5 ≤ 0
            approx_eq!(f64, constraint.evaluate(&x), 5.0); // 10 - 5 = 5

            let grad = constraint.gradient(&x);
            assert_eq!(grad, Array1::<f64>::ones(n) * 2.0); // 2*xi for each i
        }

        #[test]
        fn test_quadratic_constraint_debug_format() {
            let q = array![[1.0, 0.0], [0.0, 2.0]];
            let a = array![1.0, -1.0];
            let ineq = QuadraticConstraint::new(q.clone(), a.clone(), 0.5, false);
            let eq = QuadraticConstraint::new(q, a, 0.5, true);

            let ineq_debug = format!("{:?}", ineq);
            let eq_debug = format!("{:?}", eq);

            assert!(ineq_debug.contains("≤"));
            assert!(eq_debug.contains("="));
            assert!(ineq_debug.contains("0.5"));
        }
    }

    // Box Constraints Tests
    mod box_constraint_tests {
        use super::*;

        #[test]
        fn test_box_constraints_basic() {
            let lower = array![0.0, -1.0];
            let upper = array![2.0, 1.0];
            let constraints = create_box_constraints(&lower, &upper);

            assert_eq!(constraints.len(), 4); // 2 lower + 2 upper bounds

            let x = array![1.0, 0.0];

            // Test that point satisfies all constraints (should be ≤ 0)
            for constraint in &constraints {
                assert!(
                    constraint.evaluate(&x) <= 1e-10,
                    "Constraint violated: {}",
                    constraint.evaluate(&x)
                );
            }
        }

        #[test]
        fn test_box_constraints_at_boundaries() {
            let lower = array![-2.0, 0.0];
            let upper = array![3.0, 5.0];
            let constraints = create_box_constraints(&lower, &upper);

            // Test lower boundary
            let x_lower = array![-2.0, 0.0];
            for constraint in &constraints {
                assert!(
                    constraint.evaluate(&x_lower) <= 1e-10,
                    "Lower boundary constraint violated: {}",
                    constraint.evaluate(&x_lower)
                );
            }

            // Test upper boundary
            let x_upper = array![3.0, 5.0];
            for constraint in &constraints {
                assert!(
                    constraint.evaluate(&x_upper) <= 1e-10,
                    "Upper boundary constraint violated: {}",
                    constraint.evaluate(&x_upper)
                );
            }
        }

        #[test]
        fn test_box_constraints_violation() {
            let lower = array![1.0, 2.0];
            let upper = array![4.0, 6.0];
            let constraints = create_box_constraints(&lower, &upper);

            // Test point below lower bound
            let x_below = array![0.5, 3.0];
            let violations_below: Array1<f64> = constraints
                .iter()
                .map(|c| c.evaluate(&x_below))
                .filter(|&val| val > 1e-10)
                .collect();
            assert!(!violations_below.is_empty(), "Should violate lower bound");

            // Test point above upper bound
            let x_above = array![2.0, 7.0];
            let violations_above: Array1<f64> = constraints
                .iter()
                .map(|c| c.evaluate(&x_above))
                .filter(|&val| val > 1e-10)
                .collect();
            assert!(!violations_above.is_empty(), "Should violate upper bound");
        }

        #[test]
        fn test_box_constraints_infinite_bounds() {
            let lower = array![f64::NEG_INFINITY, 0.0];
            let upper = array![5.0, f64::INFINITY];
            let constraints = create_box_constraints(&lower, &upper);

            // Should only create constraints for finite bounds
            assert_eq!(constraints.len(), 2); // One lower bound, one upper bound

            let x = array![-1000.0, 1000.0]; // Large values
            for constraint in &constraints {
                let eval = constraint.evaluate(&x);
                // Only finite bounds should constrain
                if eval > 1e-10 {
                    // This constraint should correspond to x[0] ≤ 5 or x[1] ≥ 0
                    assert!(x[0] > 5.0 || x[1] < 0.0);
                }
            }
        }

        #[test]
        fn test_box_constraints_single_variable() {
            let lower = array![2.0];
            let upper = array![8.0];
            let constraints = create_box_constraints(&lower, &upper);

            assert_eq!(constraints.len(), 2);

            let x_valid = array![5.0];
            for constraint in &constraints {
                assert!(constraint.evaluate(&x_valid) <= 1e-10);
            }

            let x_invalid_low = array![1.0];
            let violations_low: Array1<f64> = constraints
                .iter()
                .map(|c| c.evaluate(&x_invalid_low))
                .filter(|&val| val > 1e-10)
                .collect();
            assert_eq!(violations_low.len(), 1);

            let x_invalid_high = array![10.0];
            let violations_high: Array1<f64> = constraints
                .iter()
                .map(|c| c.evaluate(&x_invalid_high))
                .filter(|&val| val > 1e-10)
                .collect();
            assert_eq!(violations_high.len(), 1);
        }

        #[test]
        fn test_box_constraints_empty_bounds() {
            let lower: Array1<f64> = array![];
            let upper: Array1<f64> = array![];
            let constraints = create_box_constraints(&lower, &upper);

            assert_eq!(constraints.len(), 0);
        }

        #[test]
        fn test_box_constraints_equal_bounds() {
            let lower = array![5.0, -2.0];
            let upper = array![5.0, -2.0];
            let constraints = create_box_constraints(&lower, &upper);

            // Should create both lower and upper bounds even when equal
            assert_eq!(constraints.len(), 4);

            let x_exact = array![5.0, -2.0];
            for constraint in &constraints {
                assert!(
                    constraint.evaluate(&x_exact).abs() <= 1e-10,
                    "Exact boundary should satisfy constraint"
                );
            }
        }

        #[test]
        fn test_box_constraints_gradients() {
            let lower = array![0.0, -1.0];
            let upper = array![2.0, 3.0];
            let constraints = create_box_constraints(&lower, &upper);
            let x = array![1.0, 1.0];

            for constraint in &constraints {
                let grad = constraint.gradient(&x);
                // Each box constraint should have exactly one non-zero gradient component
                let non_zero_count = grad.iter().filter(|&&val| val.abs() > 1e-10).count();
                assert_eq!(non_zero_count, 1);

                // The non-zero component should be ±1
                let non_zero_val = grad.iter().find(|&&val| val.abs() > 1e-10).unwrap();
                assert!(non_zero_val.abs() - 1.0 < 1e-10);
            }
        }
    }

    // Integration and Edge Case Tests
    mod integration_tests {
        use super::*;

        #[test]
        fn test_constraint_cloning() {
            let linear = LinearConstraint::inequality(array![1.0, 2.0], -1.0);
            let cloned_linear = linear.clone();

            let x = array![0.5, 0.25];
            assert_eq!(linear.evaluate(&x), cloned_linear.evaluate(&x));
            assert_eq!(linear.gradient(&x), cloned_linear.gradient(&x));

            let q = array![[1.0, 0.5], [0.5, 2.0]];
            let quadratic = QuadraticConstraint::new(q, array![1.0, -1.0], 2.0, false);
            let cloned_quadratic = quadratic.clone();

            assert_eq!(quadratic.evaluate(&x), cloned_quadratic.evaluate(&x));
            assert_eq!(quadratic.gradient(&x), cloned_quadratic.gradient(&x));
        }

        #[test]
        fn test_mixed_constraint_system() {
            // Create a mixed system with linear, quadratic, and box constraints
            let mut constraints: Vec<Box<dyn Constraint>> = Vec::new();

            // Linear constraint: x1 + x2 ≤ 2
            constraints.push(Box::new(LinearConstraint::inequality(
                array![1.0, 1.0],
                -2.0,
            )));

            // Quadratic constraint: x1² + x2² ≤ 1
            let q = create_identity_matrix(2);
            constraints.push(Box::new(QuadraticConstraint::new(
                q,
                array![0.0, 0.0],
                -1.0,
                false,
            )));

            // Box constraints: 0 ≤ xi ≤ 1
            let mut box_constraints = create_box_constraints(&array![0.0, 0.0], &array![1.0, 1.0]);
            constraints.append(&mut box_constraints);

            // Test feasible point
            let x_feasible = array![0.5, 0.5];
            for constraint in &constraints {
                assert!(
                    constraint.evaluate(&x_feasible) <= 1e-10,
                    "Feasible point should satisfy all constraints"
                );
            }

            // Test infeasible point
            let x_infeasible = array![1.5, 1.5];
            let violations: Array1<f64> = constraints
                .iter()
                .map(|c| c.evaluate(&x_infeasible))
                .filter(|&val| val > 1e-10)
                .collect();
            assert!(
                !violations.is_empty(),
                "Infeasible point should violate constraints"
            );
        }

        #[test]
        fn test_constraint_consistency() {
            // Test that gradient is consistent with finite differences
            let constraint = QuadraticConstraint::new(
                array![[2.0, 1.0], [1.0, 3.0]],
                array![1.0, -2.0],
                0.5,
                false,
            );

            let x = array![1.0, 0.5];
            let h = 1e-8;

            let grad = constraint.gradient(&x);

            // Check gradient using finite differences
            for i in 0..x.len() {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[i] += h;
                x_minus[i] -= h;

                let finite_diff =
                    (constraint.evaluate(&x_plus) - constraint.evaluate(&x_minus)) / (2.0 * h);

                approx_eq!(f64, grad[i], finite_diff, epsilon = 1e-5);
            }
        }

        #[test]
        fn test_hessian_symmetry() {
            let constraint = QuadraticConstraint::new(
                array![[2.0, 1.0], [1.0, 3.0]],
                array![0.0, 0.0],
                0.0,
                false,
            );

            let x = array![1.0, 2.0];
            let hess = constraint.hessian(&x);

            // Hessian should be symmetric
            for i in 0..hess.nrows() {
                for j in 0..hess.row(i).len() {
                    approx_eq!(f64, hess[[i, j]], hess[[j, i]], epsilon = 1e-12);
                }
            }
        }

        #[test]
        fn test_zero_dimensional_constraint() {
            let constraint = LinearConstraint::inequality(array![], 5.0);
            let x: Array1<f64> = array![];

            // Should evaluate to just the constant term
            approx_eq!(f64, constraint.evaluate(&x), 5.0);
            assert_eq!(constraint.gradient(&x), Array1::<f64>::zeros(0));
            assert_eq!(constraint.hessian(&x), Array2::<f64>::zeros((0, 0)));
        }

        #[test]
        fn test_constraint_with_extreme_values() {
            let constraint = LinearConstraint::inequality(array![1e10, -1e10], 1e15);
            let x = array![1e-10, 1e-10];

            // Should handle extreme values without overflow
            let result = constraint.evaluate(&x);
            assert!(result.is_finite());

            let grad = constraint.gradient(&x);
            assert!(grad.iter().all(|&val| val.is_finite()));
        }
    }

    // Performance and Memory Tests
    mod performance_tests {
        use super::*;

        #[test]
        fn test_large_linear_constraint_performance() {
            let n = 10000;
            let a: Array1<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            let x: Array1<f64> = (0..n).map(|i| (i as f64).cos()).collect();
            let constraint = LinearConstraint::inequality(a, -100.0);

            // These operations should complete in reasonable time
            let _result = constraint.evaluate(&x);
            let _grad = constraint.gradient(&x);
            let _hess = constraint.hessian(&x);
        }

        #[test]
        fn test_large_quadratic_constraint_performance() {
            let n = 100; // Smaller for quadratic due to O(n²) operations
            let q = create_identity_matrix(n);
            let a = Array1::zeros(n);
            let x = Array1::ones(n);
            let constraint = QuadraticConstraint::new(q, a, 0.0, false);

            let _result = constraint.evaluate(&x);
            let _grad = constraint.gradient(&x);
            let _hess = constraint.hessian(&x);
        }
    }
}
