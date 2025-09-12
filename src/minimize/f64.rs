use crate::minimize::MinimizerError;

pub mod bracket;
pub mod brent;
pub mod conjugate_gradient;
pub mod constraint;
pub mod dbrent;
pub mod golden;
pub mod interior_point;
pub mod objective;
pub mod powell;
pub mod quasi_newton;
pub mod simplex;
pub mod simulated_annealing;

pub use self::bracket::Bracket;
pub use self::brent::Brent;
pub use self::conjugate_gradient::ConjGrad;
pub use self::constraint::{
    Constraint, LinearConstraint, QuadraticConstraint, create_box_constraints,
};
pub use self::dbrent::DBrent;
pub use self::golden::Golden;
pub use self::interior_point::InteriorPoint;
pub use self::objective::{
    F1dim, HF1dim, MultiDimFn, MultiDimGradFn, MultiDimHessFn, MultiDimNumGradFn, ObjDerFn, ObjFn,
    ObjGradFn, ObjHessFn, SingleDimDerFn, SingleDimFn,
};
pub use self::powell::Powell;
pub use self::quasi_newton::QuasiNewton;
pub use self::simplex::Simplex;

/// A vertex of the simplex
#[derive(Debug, Clone)]
pub(crate) struct Vector {}

impl Vector {
    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn vector_norm(v: &[f64]) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn vector_add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    fn vector_subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
    }

    fn scalar_vector_multiply(scalar: f64, vector: &[f64]) -> Vec<f64> {
        vector.iter().map(|&x| scalar * x).collect()
    }
}

/// A vertex of the simplex
#[derive(Debug, Clone)]
pub(crate) struct Vertex {
    pub(crate) point: Vec<f64>,
    pub(crate) value: f64,
}

impl Vertex {
    pub(crate) fn new<F>(point: Vec<f64>, f: &F) -> Result<Self, MinimizerError>
    where
        F: Fn(&[f64]) -> f64,
    {
        let value = f(&point);
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(Vertex { point, value })
    }

    pub(crate) fn new_boxed(point: Vec<f64>, f: Box<dyn ObjFn>) -> Result<Self, MinimizerError> {
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
