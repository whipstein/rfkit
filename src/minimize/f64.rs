use crate::error::MinimizerError;
use crate::minimize::ObjFn;
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

// Helper function for outer product
fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }

    result
}

/// A vertex of the simplex
#[derive(Debug, Clone)]
pub(crate) struct Vector {}

impl Vector {
    fn dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn vector_norm(v: &Array1<f64>) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn vector_add(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    fn vector_subtract(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
    }

    fn scalar_vector_multiply(scalar: f64, vector: &Array1<f64>) -> Array1<f64> {
        vector.iter().map(|&x| scalar * x).collect()
    }
}

/// A vertex of the simplex
#[derive(Debug, Clone)]
pub(crate) struct Vertex {
    pub(crate) point: Array1<f64>,
    pub(crate) value: f64,
}

impl Vertex {
    pub(crate) fn new<F>(point: Array1<f64>, f: &F) -> Result<Self, MinimizerError>
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let value = f(&point);
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(Vertex { point, value })
    }

    pub(crate) fn new_boxed(
        point: Array1<f64>,
        f: Box<dyn ObjFn<f64>>,
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
    alpha: f64,
    f_new: f64,
    evaluations: usize,
    converged: bool,
}

/// Strong Wolfe conditions parameters
#[derive(Debug, Clone)]
pub struct WolfeParams {
    pub c1: f64, // Armijo condition parameter (typically 1e-4)
    pub c2: f64, // Curvature condition parameter (typically 0.9 for CG, 0.1 for quasi-Newton)
    pub max_step: f64,
    pub min_step: f64,
}

impl WolfeParams {
    /// Parameters optimized for BFGS method
    /// - Strong curvature condition for good Hessian approximation
    /// - Standard Armijo parameter
    pub fn for_bfgs() -> Self {
        WolfeParams {
            c1: 1e-4, // Standard Armijo parameter
            c2: 0.9,  // Strong curvature condition for BFGS
            max_step: 10.0,
            min_step: 1e-12,
        }
    }

    /// Parameters optimized for L-BFGS method
    /// - More relaxed curvature condition for efficiency
    /// - Suitable for large-scale problems
    pub fn for_lbfgs() -> Self {
        WolfeParams {
            c1: 1e-4,
            c2: 0.1, // More relaxed for L-BFGS efficiency
            max_step: 10.0,
            min_step: 1e-12,
        }
    }

    /// Parameters optimized for DFP method
    /// - Intermediate curvature condition
    /// - Balance between BFGS and L-BFGS
    pub fn for_dfp() -> Self {
        WolfeParams {
            c1: 1e-4,
            c2: 0.4, // Intermediate value for DFP
            max_step: 10.0,
            min_step: 1e-12,
        }
    }

    /// Parameters optimized for SR1 method
    /// - Very relaxed curvature condition
    /// - Often only uses Armijo condition
    pub fn for_sr1() -> Self {
        WolfeParams {
            c1: 1e-4,
            c2: 0.1, // Very relaxed for SR1
            max_step: 10.0,
            min_step: 1e-12,
        }
    }

    /// Conservative parameters for difficult problems
    /// - Stricter conditions for robustness
    pub fn conservative() -> Self {
        WolfeParams {
            c1: 1e-6,      // Stricter Armijo condition
            c2: 0.95,      // Very strict curvature condition
            max_step: 1.0, // Smaller maximum step
            min_step: 1e-15,
        }
    }

    /// Aggressive parameters for well-behaved problems
    /// - More relaxed conditions for speed
    pub fn aggressive() -> Self {
        WolfeParams {
            c1: 1e-2,        // More relaxed Armijo condition
            c2: 0.1,         // Relaxed curvature condition
            max_step: 100.0, // Larger maximum step
            min_step: 1e-10,
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
            c1: 1e-4,
            // c2: 0.9, // Higher value for conjugate gradient
            c2: 0.1, // Lower value for quasi-Newton (vs 0.9 for CG)
            max_step: 1e6,
            min_step: 1e-12,
        }
    }
}
