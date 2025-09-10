use crate::minimize::MinimizerError;
use ndarray::prelude::*;

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

/// Matrix operations for Hessian approximations
pub struct Matrix;

impl Matrix {
    pub(crate) fn identity(n: usize) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
        }
        matrix
    }

    pub(crate) fn multiply_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        let n = matrix.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        result
    }

    pub(crate) fn outer_product(u: &[f64], v: &[f64]) -> Vec<Vec<f64>> {
        let n = u.len();
        let mut result = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                result[i][j] = u[i] * v[j];
            }
        }

        result
    }

    pub(crate) fn scalar_multiply(matrix: &mut [Vec<f64>], scalar: f64) {
        for row in matrix {
            for element in row {
                *element *= scalar;
            }
        }
    }

    pub(crate) fn add_matrix(a: &mut [Vec<f64>], b: &[Vec<f64>]) {
        for i in 0..a.len() {
            for j in 0..a[i].len() {
                a[i][j] += b[i][j];
            }
        }
    }

    pub(crate) fn subtract_matrix(a: &mut [Vec<f64>], b: &[Vec<f64>]) {
        for i in 0..a.len() {
            for j in 0..a[i].len() {
                a[i][j] -= b[i][j];
            }
        }
    }

    /// Simple Gaussian elimination with partial pivoting
    pub(crate) fn solve_linear_system(
        mut a: Vec<Vec<f64>>,
        mut b: Vec<f64>,
    ) -> Result<Vec<f64>, MinimizerError> {
        let n = a.len();
        if b.len() != n {
            return Err(MinimizerError::LinearSystemSingular);
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in k + 1..n {
                if a[i][k].abs() > a[max_row][k].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                a.swap(k, max_row);
                b.swap(k, max_row);
            }

            // Check for singularity
            if a[k][k].abs() < 1e-14 {
                return Err(MinimizerError::LinearSystemSingular);
            }

            // Eliminate below pivot
            for i in k + 1..n {
                let factor = a[i][k] / a[k][k];
                for j in k..n {
                    a[i][j] -= factor * a[k][j];
                }
                b[i] -= factor * b[k];
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = b[i];
            for j in i + 1..n {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }

        Ok(x)
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

/// Simple Gaussian elimination with partial pivoting
pub(crate) fn solve_linear_system(
    a: &Array2<f64>,
    b: &Vec<f64>,
) -> Result<Vec<f64>, MinimizerError> {
    let mut ax = a.clone();
    let mut bx = b.clone();
    let n = ax.len();
    if bx.len() != n {
        return Err(MinimizerError::LinearSystemSingular);
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        for i in k + 1..n {
            if ax[[i, k]].abs() > ax[[max_row, k]].abs() {
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            for j in 0..ax.ncols() {
                ax.swap((k, j), (max_row, j));
            }
            bx.swap(k, max_row);
        }

        // Check for singularity
        if ax[[k, k]].abs() < 1e-14 {
            return Err(MinimizerError::LinearSystemSingular);
        }

        // Eliminate below pivot
        for i in k + 1..n {
            let factor = ax[[i, k]] / ax[[k, k]];
            for j in k..n {
                ax[[i, j]] -= factor * ax[[k, j]];
            }
            bx[i] -= factor * bx[k];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = bx[i];
        for j in i + 1..n {
            x[i] -= ax[[i, j]] * x[j];
        }
        x[i] /= ax[[i, i]];
    }

    Ok(x)
}
