use dyn_clone::DynClone;
use std::fmt;

/// Constraint definition
pub trait Constraint: DynClone {
    fn evaluate(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64>;
    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>>;
}
dyn_clone::clone_trait_object!(Constraint);

/// Linear constraint: a^T x + b ≤ 0 (inequality) or = 0 (equality)
#[derive(Clone)]
pub struct LinearConstraint {
    pub a: Vec<f64>,
    pub b: f64,
    pub is_equality: bool,
}

impl LinearConstraint {
    pub fn new(a: Vec<f64>, b: f64, is_equality: bool) -> Self {
        Self { a, b, is_equality }
    }

    pub fn inequality(a: Vec<f64>, b: f64) -> Self {
        Self::new(a, b, false)
    }

    pub fn equality(a: Vec<f64>, b: f64) -> Self {
        Self::new(a, b, true)
    }
}

impl Constraint for LinearConstraint {
    fn evaluate(&self, x: &[f64]) -> f64 {
        self.a
            .iter()
            .zip(x.iter())
            .map(|(&ai, &xi)| ai * xi)
            .sum::<f64>()
            + self.b
    }

    fn gradient(&self, _x: &[f64]) -> Vec<f64> {
        self.a.clone()
    }

    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let n = x.len();
        vec![vec![0.0; n]; n]
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
    pub q: Vec<Vec<f64>>,
    pub a: Vec<f64>,
    pub b: f64,
    pub is_equality: bool,
}

impl QuadraticConstraint {
    pub fn new(q: Vec<Vec<f64>>, a: Vec<f64>, b: f64, is_equality: bool) -> Self {
        Self {
            q,
            a,
            b,
            is_equality,
        }
    }
}

impl Constraint for QuadraticConstraint {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let n = x.len();
        let mut result = self.b;

        // Linear term: a^T x
        for i in 0..n {
            result += self.a[i] * x[i];
        }

        // Quadratic term: x^T Q x
        for i in 0..n {
            for j in 0..n {
                result += x[i] * self.q[i][j] * x[j];
            }
        }

        result
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut grad = self.a.clone();

        // Add 2 * Q * x (assuming Q is symmetric)
        for i in 0..n {
            for j in 0..n {
                grad[i] += 2.0 * self.q[i][j] * x[j];
            }
        }

        grad
    }

    fn hessian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        // Hessian is 2*Q for quadratic constraint
        let n = self.q.len();
        let mut hess = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                hess[i][j] = 2.0 * self.q[i][j];
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
pub fn create_box_constraints(lower: &[f64], upper: &[f64]) -> Vec<Box<dyn Constraint>> {
    let mut constraints = Vec::new();

    for (i, (&l, &u)) in lower.iter().zip(upper.iter()).enumerate() {
        // x_i ≥ l becomes -x_i + l ≤ 0
        if l.is_finite() {
            let mut a = vec![0.0; lower.len()];
            a[i] = -1.0;
            constraints.push(Box::new(LinearConstraint::inequality(a, l)) as Box<dyn Constraint>);
        }

        // x_i ≤ u becomes x_i - u ≤ 0
        if u.is_finite() {
            let mut a = vec![0.0; lower.len()];
            a[i] = 1.0;
            constraints.push(Box::new(LinearConstraint::inequality(a, -u)) as Box<dyn Constraint>);
        }
    }

    constraints
}

#[cfg(test)]
mod constraintf64_tests {
    use super::*;
    use float_cmp::*;

    #[test]
    fn test_linear_constraint() {
        let constraint = LinearConstraint::inequality(vec![1.0, 1.0], -1.0);
        let x = vec![0.5, 0.3];

        // Test constraint: x1 + x2 - 1 ≤ 0
        approx_eq!(f64, constraint.evaluate(&x), -0.2); // 0.5 + 0.3 - 1 = -0.2
        assert_eq!(constraint.gradient(&x), vec![1.0, 1.0]);

        let hess = constraint.hessian(&x);
        assert_eq!(hess, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn test_quadratic_constraint() {
        let q = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let a = vec![0.0, 0.0];
        let constraint = QuadraticConstraint::new(q, a, -1.0, false);
        let x = vec![0.5, 0.5];

        // Test constraint: x1² + x2² - 1 ≤ 0
        assert!((constraint.evaluate(&x)).abs() < 1e-10); // 1.0 + 0.0 - 1 = 0.0
        assert_eq!(constraint.gradient(&x), vec![2.0, 2.0]); // [4*0.5, 4*0.5]

        let hess = constraint.hessian(&x);
        assert_eq!(hess, vec![vec![4.0, 0.0], vec![0.0, 4.0]]);
    }

    #[test]
    fn test_box_constraints() {
        let lower = vec![0.0, -1.0];
        let upper = vec![2.0, 1.0];
        let constraints = create_box_constraints(&lower, &upper);

        assert_eq!(constraints.len(), 4); // 2 lower + 2 upper bounds

        let x = vec![1.0, 0.0];

        // Test that point satisfies all constraints
        for constraint in &constraints {
            assert!(
                constraint.evaluate(&x) <= 1e-10,
                "Constraint violated: {}",
                constraint.evaluate(&x)
            );
        }

        // Test boundary point
        let x_boundary = vec![0.0, 1.0];
        for constraint in &constraints {
            assert!(
                constraint.evaluate(&x_boundary) <= 1e-10,
                "Boundary constraint violated: {}",
                constraint.evaluate(&x_boundary)
            );
        }
    }
}
