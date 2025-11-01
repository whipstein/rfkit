use crate::error::MinimizerError;
use crate::minimize::ObjFn;
use crate::myfloat::MyFloat;
use ndarray::prelude::*;

pub mod bracket;
pub mod brent;
pub mod cma_es;
pub mod conjugate_gradient;
pub mod constraint;
pub mod dbrent;
pub mod golden;
pub mod interior_point;
pub mod nelder_mead;
pub mod nelder_mead_bounded;
pub mod objective;
pub mod powell;
pub mod quasi_newton;
pub mod simplex;
pub mod simulated_annealing;

pub use self::bracket::Bracket;
pub use self::brent::Brent;
pub use self::cma_es::CmaEs;
pub use self::conjugate_gradient::ConjGrad;
pub use self::constraint::{
    Constraint, LinearConstraint, QuadraticConstraint, create_box_constraints,
};
pub use self::dbrent::DBrent;
pub use self::golden::Golden;
pub use self::interior_point::InteriorPoint;
pub use self::nelder_mead::{NelderMead, NelderMeadResult};
pub use self::nelder_mead_bounded::{NelderMeadBounded, NelderMeadBoundedResult};
pub use self::objective::{
    F1dim, GF1dim, HF1dim, MultiDimFn, MultiDimGradFn, MultiDimHessFn, MultiDimNumGradFn,
    SingleDimDerFn, SingleDimFn,
};
pub use self::powell::Powell;
pub use self::quasi_newton::QuasiNewton;
pub use self::simplex::Simplex;

// Helper function for dot product
fn dot_1d_1d(a: &Array1<MyFloat>, b: &Array1<MyFloat>) -> MyFloat {
    let mut result = MyFloat::new(0.0);
    for i in 0..a.len() {
        result += &a[i] * &b[i];
    }

    result
}

// Helper function for dot product
fn dot_1d_2d(a: &Array1<MyFloat>, b: &Array2<MyFloat>) -> Array1<MyFloat> {
    let mut result = Array1::zeros(b.ncols());
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            result[j] += &a[i] * &b[[i, j]];
        }
    }

    result
}

// Helper function for dot product (matrix * vector)
fn dot_2d_1d(a: &Array2<MyFloat>, b: &Array1<MyFloat>) -> Array1<MyFloat> {
    let mut result = Array1::zeros(a.nrows());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            result[i] += &a[[i, j]] * &b[j];
        }
    }

    result
}

// Helper function for dot product
fn dot_2d_2d(a: &Array2<MyFloat>, b: &Array2<MyFloat>) -> Array2<MyFloat> {
    let mut result = Array2::zeros((a.nrows(), b.ncols()));
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            for k in 0..a.ncols() {
                result[[i, j]] += &a[[i, k]] * &b[[k, j]];
            }
        }
    }

    result
}

// Helper function for L2 norm (Euclidean norm) of 1D array
fn norm_1d(a: &Array1<MyFloat>) -> MyFloat {
    let mut sum = MyFloat::new(0.0);
    for i in 0..a.len() {
        sum += &a[i] * &a[i];
    }
    sum.sqrt()
}

// Helper function for Frobenius norm of 2D array
// This is the matrix equivalent of the Euclidean norm
fn norm_2d(a: &Array2<MyFloat>) -> MyFloat {
    let mut sum = MyFloat::new(0.0);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            sum += &a[[i, j]] * &a[[i, j]];
        }
    }
    sum.sqrt()
}

// Helper function for L1 norm (Manhattan norm) of 1D array
fn norm_1d_l1(a: &Array1<MyFloat>) -> MyFloat {
    let mut sum = MyFloat::new(0.0);
    for i in 0..a.len() {
        sum += a[i].abs();
    }
    sum
}

// Helper function for L1 norm of 2D array
fn norm_2d_l1(a: &Array2<MyFloat>) -> MyFloat {
    let mut sum = MyFloat::new(0.0);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            sum += a[[i, j]].abs();
        }
    }
    sum
}

// Helper function for L-infinity norm (maximum norm) of 1D array
fn norm_1d_linf(a: &Array1<MyFloat>) -> MyFloat {
    let mut max = MyFloat::new(0.0);
    for i in 0..a.len() {
        let abs_val = a[i].abs();
        if abs_val > max {
            max = abs_val;
        }
    }
    max
}

// Helper function for L-infinity norm of 2D array
fn norm_2d_linf(a: &Array2<MyFloat>) -> MyFloat {
    let mut max = MyFloat::new(0.0);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let abs_val = a[[i, j]].abs();
            if abs_val > max {
                max = abs_val;
            }
        }
    }
    max
}

// Helper function for outer product
fn outer(a: &Array1<MyFloat>, b: &Array1<MyFloat>) -> Array2<MyFloat> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = &a[i] * &b[j];
        }
    }

    result
}

/// A vertex of the simplex
#[derive(Debug, Clone)]
pub(crate) struct Vector {}

impl Vector {
    fn dot_product(a: &Array1<MyFloat>, b: &Array1<MyFloat>) -> MyFloat {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn vector_norm(v: &Array1<MyFloat>) -> MyFloat {
        v.iter().map(|x| x * x).sum::<MyFloat>().sqrt()
    }

    fn vector_add(a: &Array1<MyFloat>, b: &Array1<MyFloat>) -> Array1<MyFloat> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    fn vector_subtract(a: &Array1<MyFloat>, b: &Array1<MyFloat>) -> Array1<MyFloat> {
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }

    fn scalar_vector_multiply(scalar: &MyFloat, vector: &Array1<MyFloat>) -> Array1<MyFloat> {
        vector.iter().map(|x| scalar * x).collect()
    }
}

/// A vertex of the simplex
#[derive(Debug, Clone)]
pub(crate) struct Vertex {
    pub(crate) point: Array1<MyFloat>,
    pub(crate) value: MyFloat,
}

impl Vertex {
    pub(crate) fn new<F>(point: Array1<MyFloat>, f: &F) -> Result<Self, MinimizerError>
    where
        F: Fn(&Array1<MyFloat>) -> MyFloat,
    {
        let value = f(&point);
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(Vertex { point, value })
    }

    pub(crate) fn new_boxed(
        point: Array1<MyFloat>,
        f: Box<dyn ObjFn<MyFloat>>,
    ) -> Result<Self, MinimizerError> {
        let value = f.call(&point);
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(Vertex { point, value })
    }
}

/// Line search result for internal use
#[derive(Debug)]
struct LineSearchResult {
    alpha: MyFloat,
    f_new: MyFloat,
    evaluations: usize,
    converged: bool,
}

/// Strong Wolfe conditions parameters
#[derive(Debug, Clone)]
pub struct WolfeParams {
    pub c1: MyFloat, // Armijo condition parameter (typically 1e-4)
    pub c2: MyFloat, // Curvature condition parameter (typically 0.9 for CG, 0.1 for quasi-Newton)
    pub max_step: MyFloat,
    pub min_step: MyFloat,
}

impl WolfeParams {
    /// Parameters optimized for BFGS method
    /// - Strong curvature condition for good Hessian approximation
    /// - Standard Armijo parameter
    pub fn for_bfgs() -> Self {
        WolfeParams {
            c1: MyFloat::new(1e-4), // Standard Armijo parameter
            c2: MyFloat::new(0.9),  // Strong curvature condition for BFGS
            max_step: MyFloat::new(10.0),
            min_step: MyFloat::new(1e-12),
        }
    }

    /// Parameters optimized for L-BFGS method
    /// - More relaxed curvature condition for efficiency
    /// - Suitable for large-scale problems
    pub fn for_lbfgs() -> Self {
        WolfeParams {
            c1: MyFloat::new(1e-4),
            c2: MyFloat::new(0.1), // More relaxed for L-BFGS efficiency
            max_step: MyFloat::new(10.0),
            min_step: MyFloat::new(1e-12),
        }
    }

    /// Parameters optimized for DFP method
    /// - Intermediate curvature condition
    /// - Balance between BFGS and L-BFGS
    pub fn for_dfp() -> Self {
        WolfeParams {
            c1: MyFloat::new(1e-4),
            c2: MyFloat::new(0.4), // Intermediate value for DFP
            max_step: MyFloat::new(10.0),
            min_step: MyFloat::new(1e-12),
        }
    }

    /// Parameters optimized for SR1 method
    /// - Very relaxed curvature condition
    /// - Often only uses Armijo condition
    pub fn for_sr1() -> Self {
        WolfeParams {
            c1: MyFloat::new(1e-4),
            c2: MyFloat::new(0.1), // Very relaxed for SR1
            max_step: MyFloat::new(10.0),
            min_step: MyFloat::new(1e-12),
        }
    }

    /// Conservative parameters for difficult problems
    /// - Stricter conditions for robustness
    pub fn conservative() -> Self {
        WolfeParams {
            c1: MyFloat::new(1e-6),      // Stricter Armijo condition
            c2: MyFloat::new(0.95),      // Very strict curvature condition
            max_step: MyFloat::new(1.0), // Smaller maximum step
            min_step: MyFloat::new(1e-15),
        }
    }

    /// Aggressive parameters for well-behaved problems
    /// - More relaxed conditions for speed
    pub fn aggressive() -> Self {
        WolfeParams {
            c1: MyFloat::new(1e-2),        // More relaxed Armijo condition
            c2: MyFloat::new(0.1),         // Relaxed curvature condition
            max_step: MyFloat::new(100.0), // Larger maximum step
            min_step: MyFloat::new(1e-10),
        }
    }

    /// Validate that the Wolfe parameters are mathematically valid
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.c1 <= 0.0 || self.c1 >= 1.0 {
            return Err("c1 must be in (0, 1)");
        }

        if self.c2 <= self.c1 || self.c2 >= 1.0 {
            return Err("c2 must be in (c1, 1)");
        }

        if self.max_step <= 0.0 {
            return Err("max_step must be positive");
        }

        if self.min_step <= 0.0 || self.min_step >= self.max_step {
            return Err("min_step must be positive and less than max_step");
        }

        Ok(())
    }
}

impl Default for WolfeParams {
    fn default() -> Self {
        Self {
            c1: MyFloat::new(1e-4),
            // c2: 0.9, // Higher value for conjugate gradient
            c2: MyFloat::new(0.1), // Lower value for quasi-Newton (vs 0.9 for CG)
            max_step: MyFloat::new(1e6),
            min_step: MyFloat::new(1e-12),
        }
    }
}

#[cfg(test)]
mod minimize_myfloat_tests {
    use super::*;
    use float_cmp::{F64Margin, approx_eq};

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-10,
        ulps: 10,
    };

    mod dot_1d_1d_tests {
        use super::*;

        #[test]
        fn test_dot_1d_1d_basic() {
            let a = array![1.0.into(), 2.0.into(), 3.0.into()];
            let b = array![4.0.into(), 5.0.into(), 6.0.into()];
            let result = dot_1d_1d(&a, &b);
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assert!(approx_eq!(f64, result.to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_zeros() {
            let a = array![0.0.into(), 0.0.into(), 0.0.into()];
            let b = array![1.0.into(), 2.0.into(), 3.0.into()];
            let result = dot_1d_1d(&a, &b);
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_negative_values() {
            let a = array![(-1.0).into(), 2.0.into(), (-3.0).into()];
            let b = array![4.0.into(), (-5.0).into(), 6.0.into()];
            let result = dot_1d_1d(&a, &b);
            // (-1)*4 + 2*(-5) + (-3)*6 = -4 - 10 - 18 = -32
            assert!(approx_eq!(f64, result.to_f64(), -32.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_orthogonal() {
            // Orthogonal vectors should have dot product of 0
            let a = array![1.0.into(), 0.0.into()];
            let b = array![0.0.into(), 1.0.into()];
            let result = dot_1d_1d(&a, &b);
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_single_element() {
            let a = array![5.0.into()];
            let b = array![3.0.into()];
            let result = dot_1d_1d(&a, &b);
            assert!(approx_eq!(f64, result.to_f64(), 15.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_commutative() {
            let a = array![1.5.into(), 2.5.into(), 3.5.into()];
            let b = array![4.5.into(), 5.5.into(), 6.5.into()];
            let result1 = dot_1d_1d(&a, &b);
            let result2 = dot_1d_1d(&b, &a);
            assert!(approx_eq!(f64, result1.to_f64(), result2.to_f64(), MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_fractional_values() {
            let a = array![0.1.into(), 0.2.into(), 0.3.into()];
            let b = array![0.4.into(), 0.5.into(), 0.6.into()];
            let result = dot_1d_1d(&a, &b);
            // 0.1*0.4 + 0.2*0.5 + 0.3*0.6 = 0.04 + 0.1 + 0.18 = 0.32
            assert!(approx_eq!(f64, result.to_f64(), 0.32, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_large_values() {
            let a = array![1000.0.into(), 2000.0.into()];
            let b = array![3000.0.into(), 4000.0.into()];
            let result = dot_1d_1d(&a, &b);
            // 1000*3000 + 2000*4000 = 3,000,000 + 8,000,000 = 11,000,000
            assert!(approx_eq!(f64, result.to_f64(), 11_000_000.0, MARGIN));
        }
    }

    mod dot_1d_2d_tests {
        use super::*;

        #[test]
        fn test_dot_1d_2d_basic() {
            let a = array![1.0.into(), 2.0.into(), 3.0.into()];
            let b = array![
                [1.0.into(), 2.0.into()],
                [3.0.into(), 4.0.into()],
                [5.0.into(), 6.0.into()]
            ];
            let result = dot_1d_2d(&a, &b);

            // Result should be [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6]
            //                = [1 + 6 + 15, 2 + 8 + 18]
            //                = [22, 28]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 22.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 28.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_identity_like() {
            let a = array![1.0.into(), 2.0.into()];
            let b = array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]];
            let result = dot_1d_2d(&a, &b);

            // Should get the original vector back
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 1.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 2.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_zeros() {
            let a = array![0.0.into(), 0.0.into(), 0.0.into()];
            let b = array![
                [1.0.into(), 2.0.into()],
                [3.0.into(), 4.0.into()],
                [5.0.into(), 6.0.into()]
            ];
            let result = dot_1d_2d(&a, &b);

            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_negative_values() {
            let a = array![(-1.0).into(), 2.0.into()];
            let b = array![[3.0.into(), (-4.0).into()], [5.0.into(), 6.0.into()]];
            let result = dot_1d_2d(&a, &b);

            // Result: [-1*3 + 2*5, -1*(-4) + 2*6]
            //       = [-3 + 10, 4 + 12]
            //       = [7, 16]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 7.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 16.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_single_column() {
            let a = array![1.0.into(), 2.0.into(), 3.0.into()];
            let b = array![[4.0.into()], [5.0.into()], [6.0.into()]];
            let result = dot_1d_2d(&a, &b);

            // Result: [1*4 + 2*5 + 3*6] = [32]
            assert_eq!(result.len(), 1);
            assert!(approx_eq!(f64, result[0].to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_rectangular() {
            let a = array![1.0.into(), 2.0.into()];
            let b = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ];
            let result = dot_1d_2d(&a, &b);

            // Result: [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6]
            //       = [9, 12, 15]
            assert_eq!(result.len(), 3);
            assert!(approx_eq!(f64, result[0].to_f64(), 9.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 12.0, MARGIN));
            assert!(approx_eq!(f64, result[2].to_f64(), 15.0, MARGIN));
        }
    }

    mod dot_2d_1d_tests {
        use super::*;

        #[test]
        fn test_dot_2d_1d_basic() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ]; // 2x3 matrix
            let b = array![1.0.into(), 2.0.into(), 3.0.into()]; // 3-element vector
            let result = dot_2d_1d(&a, &b);

            // Result: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3]
            //       = [1 + 4 + 9, 4 + 10 + 18]
            //       = [14, 32]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 14.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_square_matrix() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![5.0.into(), 6.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result: [1*5 + 2*6, 3*5 + 4*6]
            //       = [5 + 12, 15 + 24]
            //       = [17, 39]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 17.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 39.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_identity_matrix() {
            let identity = array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]];
            let b = array![3.0.into(), 7.0.into()];
            let result = dot_2d_1d(&identity, &b);

            // Identity matrix should return the original vector
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 3.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_zero_vector() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![0.0.into(), 0.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result should be zero vector
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_zero_matrix() {
            let a = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let b = array![5.0.into(), 6.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result should be zero vector
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_negative_values() {
            let a = array![[(-1.0).into(), 2.0.into()], [3.0.into(), (-4.0).into()]];
            let b = array![5.0.into(), (-6.0).into()];
            let result = dot_2d_1d(&a, &b);

            // Result: [(-1)*5 + 2*(-6), 3*5 + (-4)*(-6)]
            //       = [-5 - 12, 15 + 24]
            //       = [-17, 39]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), -17.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 39.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_single_row() {
            let a = array![[1.0.into(), 2.0.into(), 3.0.into()]];
            let b = array![4.0.into(), 5.0.into(), 6.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result: [1*4 + 2*5 + 3*6] = [4 + 10 + 18] = [32]
            assert_eq!(result.len(), 1);
            assert!(approx_eq!(f64, result[0].to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_single_column() {
            let a = array![[2.0.into()], [3.0.into()], [4.0.into()]];
            let b = array![5.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result: [2*5, 3*5, 4*5] = [10, 15, 20]
            assert_eq!(result.len(), 3);
            assert!(approx_eq!(f64, result[0].to_f64(), 10.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 15.0, MARGIN));
            assert!(approx_eq!(f64, result[2].to_f64(), 20.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_tall_matrix() {
            let a = array![
                [1.0.into(), 2.0.into()],
                [3.0.into(), 4.0.into()],
                [5.0.into(), 6.0.into()],
                [7.0.into(), 8.0.into()]
            ]; // 4x2 matrix
            let b = array![2.0.into(), 3.0.into()]; // 2-element vector
            let result = dot_2d_1d(&a, &b);

            // Result: [1*2 + 2*3, 3*2 + 4*3, 5*2 + 6*3, 7*2 + 8*3]
            //       = [2 + 6, 6 + 12, 10 + 18, 14 + 24]
            //       = [8, 18, 28, 38]
            assert_eq!(result.len(), 4);
            assert!(approx_eq!(f64, result[0].to_f64(), 8.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 18.0, MARGIN));
            assert!(approx_eq!(f64, result[2].to_f64(), 28.0, MARGIN));
            assert!(approx_eq!(f64, result[3].to_f64(), 38.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_wide_matrix() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()],
                [5.0.into(), 6.0.into(), 7.0.into(), 8.0.into()]
            ]; // 2x4 matrix
            let b = array![1.0.into(), 1.0.into(), 1.0.into(), 1.0.into()]; // 4-element vector
            let result = dot_2d_1d(&a, &b);

            // Result: [1+2+3+4, 5+6+7+8] = [10, 26]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 10.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 26.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_3x3_matrix() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()],
                [7.0.into(), 8.0.into(), 9.0.into()]
            ];
            let b = array![2.0.into(), 1.0.into(), 3.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result: [1*2 + 2*1 + 3*3, 4*2 + 5*1 + 6*3, 7*2 + 8*1 + 9*3]
            //       = [2 + 2 + 9, 8 + 5 + 18, 14 + 8 + 27]
            //       = [13, 31, 49]
            assert_eq!(result.len(), 3);
            assert!(approx_eq!(f64, result[0].to_f64(), 13.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 31.0, MARGIN));
            assert!(approx_eq!(f64, result[2].to_f64(), 49.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_fractional_values() {
            let a = array![[0.5.into(), 0.25.into()], [0.75.into(), 0.125.into()]];
            let b = array![4.0.into(), 8.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result: [0.5*4 + 0.25*8, 0.75*4 + 0.125*8]
            //       = [2.0 + 2.0, 3.0 + 1.0]
            //       = [4.0, 4.0]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 4.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 4.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_unit_vector() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ];
            let b = array![1.0.into(), 0.0.into(), 0.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Should extract the first column
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 1.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 4.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_ones_vector() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ];
            let b = array![1.0.into(), 1.0.into(), 1.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Should sum each row
            // Result: [1+2+3, 4+5+6] = [6, 15]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 6.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 15.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_single_element() {
            let a = array![[5.0.into()]];
            let b = array![3.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result: [5*3] = [15]
            assert_eq!(result.len(), 1);
            assert!(approx_eq!(f64, result[0].to_f64(), 15.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_diagonal_matrix() {
            let a = array![
                [2.0.into(), 0.0.into(), 0.0.into()],
                [0.0.into(), 3.0.into(), 0.0.into()],
                [0.0.into(), 0.0.into(), 4.0.into()]
            ];
            let b = array![5.0.into(), 6.0.into(), 7.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Diagonal matrix scales each component
            // Result: [2*5, 3*6, 4*7] = [10, 18, 28]
            assert_eq!(result.len(), 3);
            assert!(approx_eq!(f64, result[0].to_f64(), 10.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 18.0, MARGIN));
            assert!(approx_eq!(f64, result[2].to_f64(), 28.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_linearity() {
            // Test that A(c*v) = c*(A*v)
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let v = array![2.0.into(), 3.0.into()];
            let c = MyFloat::new(5.0);

            // Compute A*(c*v)
            let cv = v.mapv(|x| &x * &c);
            let result1 = dot_2d_1d(&a, &cv);

            // Compute c*(A*v)
            let av = dot_2d_1d(&a, &v);
            let result2 = av.mapv(|x| &x * &c);

            assert_eq!(result1.len(), result2.len());
            for i in 0..result1.len() {
                assert!(approx_eq!(
                    f64,
                    result1[i].to_f64(),
                    result2[i].to_f64(),
                    MARGIN
                ));
            }
        }

        #[test]
        fn test_dot_2d_1d_distributivity() {
            // Test that A(v + w) = A*v + A*w
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let v = array![2.0.into(), 3.0.into()];
            let w = array![5.0.into(), 7.0.into()];

            // Compute A*(v + w)
            let vw = array![&v[0] + &w[0], &v[1] + &w[1]];
            let result1 = dot_2d_1d(&a, &vw);

            // Compute A*v + A*w
            let av = dot_2d_1d(&a, &v);
            let aw = dot_2d_1d(&a, &w);
            let result2 = array![&av[0] + &aw[0], &av[1] + &aw[1]];

            assert_eq!(result1.len(), result2.len());
            for i in 0..result1.len() {
                assert!(approx_eq!(
                    f64,
                    result1[i].to_f64(),
                    result2[i].to_f64(),
                    MARGIN
                ));
            }
        }

        #[test]
        fn test_dot_2d_1d_large_values() {
            let a = array![
                [1000.0.into(), 2000.0.into()],
                [3000.0.into(), 4000.0.into()]
            ];
            let b = array![5.0.into(), 6.0.into()];
            let result = dot_2d_1d(&a, &b);

            // Result: [1000*5 + 2000*6, 3000*5 + 4000*6]
            //       = [5000 + 12000, 15000 + 24000]
            //       = [17000, 39000]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 17000.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 39000.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_rotation_matrix() {
            // Test with a 2D rotation matrix (90 degrees counterclockwise)
            // [0, -1]
            // [1,  0]
            let a = array![[0.0.into(), (-1.0).into()], [1.0.into(), 0.0.into()]];
            let b = array![1.0.into(), 0.0.into()]; // Unit vector along x-axis
            let result = dot_2d_1d(&a, &b);

            // Should rotate to y-axis: [0, 1]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 1.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_consistency_with_2d_2d() {
            // Verify that dot_2d_1d(A, v) gives same result as
            // treating v as a column vector and using dot_2d_2d
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let v = array![5.0.into(), 6.0.into()];

            // Using dot_2d_1d
            let result1 = dot_2d_1d(&a, &v);

            // Using dot_2d_2d with v as column vector
            let v_col = array![[5.0.into()], [6.0.into()]];
            let result2_matrix = dot_2d_2d(&a, &v_col);

            // Extract column vector
            assert_eq!(result1.len(), result2_matrix.nrows());
            for i in 0..result1.len() {
                assert!(approx_eq!(
                    f64,
                    result1[i].to_f64(),
                    result2_matrix[[i, 0]].to_f64(),
                    MARGIN
                ));
            }
        }
    }

    mod dot_2d_2d_tests {
        use super::*;

        #[test]
        fn test_dot_2d_2d_basic() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![[5.0.into(), 6.0.into()], [7.0.into(), 8.0.into()]];
            let result = dot_2d_2d(&a, &b);

            // Result:
            // [1*5 + 2*7, 1*6 + 2*8]   [19, 22]
            // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
            assert_eq!(result.shape(), &[2, 2]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 19.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 22.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 43.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), 50.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_identity() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let identity = array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]];
            let result = dot_2d_2d(&a, &identity);

            // Should get the original matrix back
            assert_eq!(result.shape(), &[2, 2]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 1.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 2.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 3.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), 4.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_zeros() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let zeros = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let result = dot_2d_2d(&a, &zeros);

            // Result should be all zeros
            assert_eq!(result.shape(), &[2, 2]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_rectangular_compatible() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ]; // 2x3
            let b = array![
                [7.0.into(), 8.0.into()],
                [9.0.into(), 10.0.into()],
                [11.0.into(), 12.0.into()]
            ]; // 3x2
            let result = dot_2d_2d(&a, &b);

            // Result should be 2x2
            // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]   [58, 64]
            // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
            assert_eq!(result.shape(), &[2, 2]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 58.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 64.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 139.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), 154.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_single_element() {
            let a = array![[5.0.into()]];
            let b = array![[3.0.into()]];
            let result = dot_2d_2d(&a, &b);

            assert_eq!(result.shape(), &[1, 1]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 15.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_negative_values() {
            let a = array![[(-1.0).into(), 2.0.into()], [3.0.into(), (-4.0).into()]];
            let b = array![[5.0.into(), (-6.0).into()], [(-7.0).into(), 8.0.into()]];
            let result = dot_2d_2d(&a, &b);

            // Result:
            // [(-1)*5 + 2*(-7), (-1)*(-6) + 2*8]   [(-19), 22]
            // [3*5 + (-4)*(-7), 3*(-6) + (-4)*8] = [43, (-50)]
            assert_eq!(result.shape(), &[2, 2]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), -19.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 22.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 43.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), -50.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_column_vector_multiplication() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![[5.0.into()], [6.0.into()]]; // Column vector
            let result = dot_2d_2d(&a, &b);

            // Result should be a 2x1 matrix
            // [1*5 + 2*6]   [17]
            // [3*5 + 4*6] = [39]
            assert_eq!(result.shape(), &[2, 1]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 17.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 39.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_row_vector_multiplication() {
            let a = array![[1.0.into(), 2.0.into(), 3.0.into()]]; // Row vector (1x3)
            let b = array![[4.0.into()], [5.0.into()], [6.0.into()]]; // Column vector (3x1)
            let result = dot_2d_2d(&a, &b);

            // Result should be a 1x1 matrix (scalar)
            // [1*4 + 2*5 + 3*6] = [32]
            assert_eq!(result.shape(), &[1, 1]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_3x3_matrix() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()],
                [7.0.into(), 8.0.into(), 9.0.into()]
            ];
            let b = array![
                [9.0.into(), 8.0.into(), 7.0.into()],
                [6.0.into(), 5.0.into(), 4.0.into()],
                [3.0.into(), 2.0.into(), 1.0.into()]
            ];
            let result = dot_2d_2d(&a, &b);

            // Manual calculation for verification
            // Row 0: [1*9+2*6+3*3, 1*8+2*5+3*2, 1*7+2*4+3*1] = [30, 24, 18]
            // Row 1: [4*9+5*6+6*3, 4*8+5*5+6*2, 4*7+5*4+6*1] = [84, 69, 54]
            // Row 2: [7*9+8*6+9*3, 7*8+8*5+9*2, 7*7+8*4+9*1] = [138, 114, 90]
            assert_eq!(result.shape(), &[3, 3]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 30.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 24.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 2]].to_f64(), 18.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 84.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), 69.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 2]].to_f64(), 54.0, MARGIN));
            assert!(approx_eq!(f64, result[[2, 0]].to_f64(), 138.0, MARGIN));
            assert!(approx_eq!(f64, result[[2, 1]].to_f64(), 114.0, MARGIN));
            assert!(approx_eq!(f64, result[[2, 2]].to_f64(), 90.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_associativity() {
            // Test that (AB)C = A(BC)
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![[5.0.into(), 6.0.into()], [7.0.into(), 8.0.into()]];
            let c = array![[9.0.into(), 10.0.into()], [11.0.into(), 12.0.into()]];

            let ab = dot_2d_2d(&a, &b);
            let abc_left = dot_2d_2d(&ab, &c);

            let bc = dot_2d_2d(&b, &c);
            let abc_right = dot_2d_2d(&a, &bc);

            assert_eq!(abc_left.shape(), abc_right.shape());
            for i in 0..abc_left.nrows() {
                for j in 0..abc_left.ncols() {
                    assert!(approx_eq!(
                        f64,
                        abc_left[[i, j]].to_f64(),
                        abc_right[[i, j]].to_f64(),
                        MARGIN
                    ));
                }
            }
        }

        #[test]
        fn test_dot_2d_2d_fractional_values() {
            let a = array![[0.5.into(), 0.25.into()], [0.75.into(), 0.125.into()]];
            let b = array![[2.0.into(), 4.0.into()], [8.0.into(), 16.0.into()]];
            let result = dot_2d_2d(&a, &b);

            // Row 0: [0.5*2 + 0.25*8, 0.5*4 + 0.25*16] = [3.0, 6.0]
            // Row 1: [0.75*2 + 0.125*8, 0.75*4 + 0.125*16] = [2.5, 5.0]
            assert_eq!(result.shape(), &[2, 2]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 3.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 6.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 2.5, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), 5.0, MARGIN));
        }
    }

    mod norm_1d_tests {
        use super::*;

        #[test]
        fn test_norm_1d_basic() {
            let a = array![3.0.into(), 4.0.into()];
            let result = norm_1d(&a);
            // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_zeros() {
            let a = array![0.0.into(), 0.0.into(), 0.0.into()];
            let result = norm_1d(&a);
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_single_element() {
            let a = array![5.0.into()];
            let result = norm_1d(&a);
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_negative_values() {
            let a = array![(-3.0).into(), (-4.0).into()];
            let result = norm_1d(&a);
            // sqrt((-3)^2 + (-4)^2) = sqrt(9 + 16) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_mixed_signs() {
            let a = array![(-3.0).into(), 4.0.into()];
            let result = norm_1d(&a);
            // sqrt((-3)^2 + 4^2) = sqrt(9 + 16) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_unit_vector() {
            let a = array![1.0.into(), 0.0.into(), 0.0.into()];
            let result = norm_1d(&a);
            assert!(approx_eq!(f64, result.to_f64(), 1.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_all_ones() {
            let a = array![1.0.into(), 1.0.into(), 1.0.into()];
            let result = norm_1d(&a);
            // sqrt(1 + 1 + 1) = sqrt(3) ≈ 1.732050808
            assert!(approx_eq!(f64, result.to_f64(), 3.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_1d_pythagorean_triple() {
            let a = array![5.0.into(), 12.0.into()];
            let result = norm_1d(&a);
            // sqrt(25 + 144) = sqrt(169) = 13
            assert!(approx_eq!(f64, result.to_f64(), 13.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_fractional() {
            let a = array![0.6.into(), 0.8.into()];
            let result = norm_1d(&a);
            // sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0
            assert!(approx_eq!(f64, result.to_f64(), 1.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_large_values() {
            let a = array![1000.0.into(), 0.0.into()];
            let result = norm_1d(&a);
            assert!(approx_eq!(f64, result.to_f64(), 1000.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_homogeneity() {
            // Test that ||c*v|| = |c| * ||v||
            let a = array![3.0.into(), 4.0.into()];
            let norm_a = norm_1d(&a);

            let scalar = MyFloat::new(2.0);
            let scaled = a.mapv(|x| &x * &scalar);
            let norm_scaled = norm_1d(&scaled);

            assert!(approx_eq!(
                f64,
                norm_scaled.to_f64(),
                2.0 * norm_a.to_f64(),
                MARGIN
            ));
        }
    }

    mod norm_2d_tests {
        use super::*;

        #[test]
        fn test_norm_2d_basic() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let result = norm_2d(&a);
            // sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477225575
            assert!(approx_eq!(f64, result.to_f64(), 30.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_zeros() {
            let a = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let result = norm_2d(&a);
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_single_element() {
            let a = array![[5.0.into()]];
            let result = norm_2d(&a);
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_identity() {
            let a = array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]];
            let result = norm_2d(&a);
            // sqrt(1 + 0 + 0 + 1) = sqrt(2) ≈ 1.414213562
            assert!(approx_eq!(f64, result.to_f64(), 2.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_negative_values() {
            let a = array![
                [(-1.0).into(), (-2.0).into()],
                [(-3.0).into(), (-4.0).into()]
            ];
            let result = norm_2d(&a);
            // sqrt(1 + 4 + 9 + 16) = sqrt(30)
            assert!(approx_eq!(f64, result.to_f64(), 30.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_mixed_signs() {
            let a = array![[1.0.into(), (-2.0).into()], [(-3.0).into(), 4.0.into()]];
            let result = norm_2d(&a);
            // sqrt(1 + 4 + 9 + 16) = sqrt(30)
            assert!(approx_eq!(f64, result.to_f64(), 30.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_rectangular() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ];
            let result = norm_2d(&a);
            // sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91) ≈ 9.539392014
            assert!(approx_eq!(f64, result.to_f64(), 91.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_column_vector() {
            let a = array![[3.0.into()], [4.0.into()]];
            let result = norm_2d(&a);
            // sqrt(9 + 16) = sqrt(25) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_row_vector() {
            let a = array![[3.0.into(), 4.0.into()]];
            let result = norm_2d(&a);
            // sqrt(9 + 16) = sqrt(25) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_3x3() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()],
                [7.0.into(), 8.0.into(), 9.0.into()]
            ];
            let result = norm_2d(&a);
            // sqrt(1+4+9+16+25+36+49+64+81) = sqrt(285) ≈ 16.881943016
            assert!(approx_eq!(f64, result.to_f64(), 285.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_homogeneity() {
            // Test that ||c*A|| = |c| * ||A||
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let norm_a = norm_2d(&a);

            let scalar = MyFloat::new(3.0);
            let scaled = a.mapv(|x| &x * &scalar);
            let norm_scaled = norm_2d(&scaled);

            assert!(approx_eq!(
                f64,
                norm_scaled.to_f64(),
                3.0 * norm_a.to_f64(),
                MARGIN
            ));
        }
    }

    mod norm_1d_l1_tests {
        use super::*;

        #[test]
        fn test_norm_1d_l1_basic() {
            let a = array![3.0.into(), 4.0.into()];
            let result = norm_1d_l1(&a);
            // |3| + |4| = 7
            assert!(approx_eq!(f64, result.to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_l1_negative() {
            let a = array![(-3.0).into(), (-4.0).into()];
            let result = norm_1d_l1(&a);
            // |-3| + |-4| = 3 + 4 = 7
            assert!(approx_eq!(f64, result.to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_l1_mixed() {
            let a = array![(-3.0).into(), 4.0.into(), (-2.0).into()];
            let result = norm_1d_l1(&a);
            // |-3| + |4| + |-2| = 3 + 4 + 2 = 9
            assert!(approx_eq!(f64, result.to_f64(), 9.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_l1_zeros() {
            let a = array![0.0.into(), 0.0.into()];
            let result = norm_1d_l1(&a);
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }
    }

    mod norm_2d_l1_tests {
        use super::*;

        #[test]
        fn test_norm_2d_l1_basic() {
            let a = array![[1.0.into(), (-2.0).into()], [3.0.into(), (-4.0).into()]];
            let result = norm_2d_l1(&a);
            // |1| + |-2| + |3| + |-4| = 1 + 2 + 3 + 4 = 10
            assert!(approx_eq!(f64, result.to_f64(), 10.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_l1_zeros() {
            let a = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let result = norm_2d_l1(&a);
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }
    }

    mod norm_1d_linf_tests {
        use super::*;

        #[test]
        fn test_norm_1d_linf_basic() {
            let a = array![3.0.into(), 4.0.into(), 2.0.into()];
            let result = norm_1d_linf(&a);
            // max(|3|, |4|, |2|) = 4
            assert!(approx_eq!(f64, result.to_f64(), 4.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_linf_negative() {
            let a = array![3.0.into(), (-7.0).into(), 2.0.into()];
            let result = norm_1d_linf(&a);
            // max(|3|, |-7|, |2|) = 7
            assert!(approx_eq!(f64, result.to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_linf_zeros() {
            let a = array![0.0.into(), 0.0.into()];
            let result = norm_1d_linf(&a);
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_linf_single() {
            let a = array![5.0.into()];
            let result = norm_1d_linf(&a);
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }
    }

    mod norm_2d_linf_tests {
        use super::*;

        #[test]
        fn test_norm_2d_linf_basic() {
            let a = array![[1.0.into(), 2.0.into()], [(-7.0).into(), 4.0.into()]];
            let result = norm_2d_linf(&a);
            // max(|1|, |2|, |-7|, |4|) = 7
            assert!(approx_eq!(f64, result.to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_linf_all_positive() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 9.0.into()]];
            let result = norm_2d_linf(&a);
            // max(1, 2, 3, 9) = 9
            assert!(approx_eq!(f64, result.to_f64(), 9.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_linf_zeros() {
            let a = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let result = norm_2d_linf(&a);
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }
    }

    mod norm_cross_validation_tests {
        use super::*;

        #[test]
        fn test_norm_inequality_1d() {
            // For the same vector: ||x||_inf <= ||x||_2 <= ||x||_1
            let a = array![1.0.into(), 2.0.into(), 3.0.into()];

            let l_inf = norm_1d_linf(&a).to_f64();
            let l_2 = norm_1d(&a).to_f64();
            let l_1 = norm_1d_l1(&a).to_f64();

            assert!(l_inf <= l_2);
            assert!(l_2 <= l_1);
        }

        #[test]
        fn test_norm_inequality_2d() {
            // Same inequality holds for matrices
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];

            let l_inf = norm_2d_linf(&a).to_f64();
            let frobenius = norm_2d(&a).to_f64();
            let l_1 = norm_2d_l1(&a).to_f64();

            assert!(l_inf <= frobenius);
            assert!(frobenius <= l_1);
        }

        #[test]
        fn test_triangle_inequality_1d() {
            // ||a + b|| <= ||a|| + ||b||
            let a = array![1.0.into(), 2.0.into()];
            let b = array![3.0.into(), 4.0.into()];
            let sum = array![4.0.into(), 6.0.into()];

            let norm_sum = norm_1d(&sum).to_f64();
            let norm_a = norm_1d(&a).to_f64();
            let norm_b = norm_1d(&b).to_f64();

            assert!(norm_sum <= norm_a + norm_b + 1e-10); // Small epsilon for floating point
        }

        #[test]
        fn test_consistency_vector_matrix() {
            // A column vector should have the same norm whether treated as 1D or 2D
            let vec_1d = array![3.0.into(), 4.0.into()];
            let vec_2d = array![[3.0.into()], [4.0.into()]];

            let norm_1 = norm_1d(&vec_1d).to_f64();
            let norm_2 = norm_2d(&vec_2d).to_f64();

            assert!(approx_eq!(f64, norm_1, norm_2, MARGIN));
        }
    }
}
