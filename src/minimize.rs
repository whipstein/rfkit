#![allow(dead_code)]
use crate::myfloat::MyFloat;
use dyn_clone::DynClone;
use ndarray::prelude::*;
use std::fmt;

pub mod f64;
pub mod multi;
pub mod nelder_mead;
pub mod nelder_mead_bounded;
pub mod single;

pub use self::nelder_mead::NelderMead;
pub use self::nelder_mead_bounded::NelderMeadBounded;
pub use self::single::{Bracket, Brent, DBrent, Golden};

/// Error types for optimizers
#[derive(Debug)]
pub enum MinimizerError {
    ConstraintViolation,
    FunctionEvaluationError,
    GradientEvaluationError,
    HessianEvaluationError,
    InfeasibleStartingPoint,
    InvalidBracket,
    InvalidDimension,
    InvalidDirectionSet,
    InvalidInitialPoints,
    InvalidInitialSimplex,
    InvalidStepSize,
    InvalidTolerance,
    LinearSearchFailed,
    LinearSystemSingular,
    LineSearchFailed,
    MaxIterationsExceeded,
    NoMinimumFound,
    NumericalInstability,
    NumericalOverflow,
    SameSignError,
    SingularHessianApproximation,
    ZeroDerivative,
    ZeroGradient,
}

impl fmt::Display for MinimizerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MinimizerError::ConstraintViolation => write!(f, "Constraint violation detected"),
            MinimizerError::FunctionEvaluationError => {
                write!(f, "Function evaluation returned invalid value")
            }
            MinimizerError::GradientEvaluationError => write!(f, "Gradient evaluation error"),
            MinimizerError::HessianEvaluationError => write!(f, "Hessian evaluation error"),
            MinimizerError::InfeasibleStartingPoint => {
                write!(f, "Starting point violates constraints")
            }
            MinimizerError::InvalidBracket => {
                write!(f, "Invalid bracket: ensure a < b")
            }
            MinimizerError::InvalidDimension => write!(f, "Invalid dimension or empty vector"),
            MinimizerError::InvalidDirectionSet => {
                write!(f, "Invalid or linearly dependent direction set")
            }
            MinimizerError::InvalidInitialPoints => {
                write!(f, "Invalid initial points: ensure a < b")
            }
            MinimizerError::InvalidInitialSimplex => {
                write!(f, "Invalid initial simplex configuration")
            }
            MinimizerError::InvalidStepSize => {
                write!(f, "Step size must be positive and finite")
            }
            MinimizerError::InvalidTolerance => write!(f, "Tolerance must be positive"),
            MinimizerError::LinearSearchFailed => write!(f, "Line search failed to converge"),
            MinimizerError::LinearSystemSingular => write!(f, "Linear system is singular"),
            MinimizerError::LineSearchFailed => write!(f, "Line search failed to find valid step"),
            MinimizerError::MaxIterationsExceeded => write!(f, "Maximum iterations exceeded"),
            MinimizerError::NumericalInstability => write!(f, "Numerical instability detected"),
            MinimizerError::NoMinimumFound => {
                write!(f, "No minimum bracket found within search limits")
            }
            MinimizerError::NumericalOverflow => {
                write!(f, "Numerical overflow during bracket expansion")
            }
            MinimizerError::SameSignError => {
                write!(
                    f,
                    "Function values at bracket endpoints must have opposite signs"
                )
            }
            MinimizerError::SingularHessianApproximation => {
                write!(f, "Hessian approximation became singular")
            }
            MinimizerError::ZeroDerivative => {
                write!(f, "Encountered zero derivative, cannot continue")
            }
            MinimizerError::ZeroGradient => write!(f, "Zero gradient encountered"),
        }
    }
}

impl std::error::Error for MinimizerError {}

// Define a trait for the objective function
pub trait ObjectiveFn: DynClone {
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat;
    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat;
}
dyn_clone::clone_trait_object!(ObjectiveFn);

// Define a trait for the derivative function
pub trait ObjectiveDerFn: ObjectiveFn + DynClone {
    fn df(&self, x: &Array1<MyFloat>) -> MyFloat;
    fn df_scalar(&self, x: &MyFloat) -> MyFloat;
}
dyn_clone::clone_trait_object!(ObjectiveDerFn);

// Define a trait for the gradient function
pub trait ObjectiveGradFn: ObjectiveFn + DynClone {
    fn grad(&self, x: &Array1<MyFloat>) -> Array1<MyFloat>;
    fn grad_scalar(&self, x: &MyFloat) -> Array1<MyFloat>;
}
dyn_clone::clone_trait_object!(ObjectiveGradFn);

// Define a trait for the hessian function
pub trait ObjectiveHessianFn: ObjectiveFn + DynClone {
    fn hessian(&self, x: &Array1<MyFloat>) -> Array2<MyFloat>;
    fn hessian_scalar(&self, x: &MyFloat) -> Array2<MyFloat>;
}
dyn_clone::clone_trait_object!(ObjectiveHessianFn);

// Implement for closures
impl<F> ObjectiveFn for F
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone,
{
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        self(x)
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        self(&array![x.clone()])
    }
}

impl<F, DF> ObjectiveFn for (F, DF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
{
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        self.0(x)
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        self.0(&array![x.clone()])
    }
}

impl<F, DF, GF> ObjectiveFn for (F, DF, GF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + DynClone + Clone,
{
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        self.0(x)
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        self.0(&array![x.clone()])
    }
}

impl<F, DF> ObjectiveDerFn for (F, DF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
{
    fn df(&self, x: &Array1<MyFloat>) -> MyFloat {
        self.1(x)
    }

    fn df_scalar(&self, x: &MyFloat) -> MyFloat {
        self.1(&array![x.clone()])
    }
}

impl<F, DF, GF> ObjectiveGradFn for (F, DF, GF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + DynClone + Clone,
{
    fn grad(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        self.2(x)
    }

    fn grad_scalar(&self, x: &MyFloat) -> Array1<MyFloat> {
        self.2(&array![x.clone()])
    }
}

// Wrapper for single-dimensional functions
#[derive(Clone)]
pub struct SingleDimFn<F>(pub F)
where
    F: Fn(&MyFloat) -> MyFloat + Clone;

// Convenience constructors
impl<F> SingleDimFn<F>
where
    F: Fn(&MyFloat) -> MyFloat + Clone,
{
    pub fn new(f: F) -> Self {
        SingleDimFn(f)
    }
}

// Implementation for single-dimensional functions
impl<F> ObjectiveFn for SingleDimFn<F>
where
    F: Fn(&MyFloat) -> MyFloat + Clone,
{
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        // Take the first element for single-dim functions
        (self.0)(&x[0])
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        (self.0)(x)
    }
}

// Wrapper for single-dimensional functions
#[derive(Clone)]
pub struct SingleDimDerFn<F, DF>(pub F, pub DF)
where
    F: Fn(&MyFloat) -> MyFloat + Clone,
    DF: Fn(&MyFloat) -> MyFloat + Clone;

// Convenience constructors
impl<F, DF> SingleDimDerFn<F, DF>
where
    F: Fn(&MyFloat) -> MyFloat + Clone,
    DF: Fn(&MyFloat) -> MyFloat + Clone,
{
    pub fn new(f: F, df: DF) -> Self {
        SingleDimDerFn(f, df)
    }
}

// Implementation for single-dimensional functions
impl<F, DF> ObjectiveFn for SingleDimDerFn<F, DF>
where
    F: Fn(&MyFloat) -> MyFloat + Clone,
    DF: Fn(&MyFloat) -> MyFloat + Clone,
{
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        // Take the first element for single-dim functions
        (self.0)(&x[0])
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        (self.0)(x)
    }
}

// Implementation for single-dimensional functions
impl<F, DF> ObjectiveDerFn for SingleDimDerFn<F, DF>
where
    F: Fn(&MyFloat) -> MyFloat + Clone,
    DF: Fn(&MyFloat) -> MyFloat + Clone,
{
    fn df(&self, x: &Array1<MyFloat>) -> MyFloat {
        // Take the first element for single-dim functions
        (self.1)(&x[0])
    }

    fn df_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.1)(x)
    }
}

// Wrapper for multi-dimensional functions
#[derive(Clone)]
pub struct MultiDimFn<F>(pub F)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone;

// Convenience constructors
impl<F> MultiDimFn<F>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
{
    pub fn new(f: F) -> Self {
        MultiDimFn(f)
    }
}

// Implementation for multi-dimensional functions
impl<F> ObjectiveFn for MultiDimFn<F>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
{
    fn call(&mut self, x: &Array1<MyFloat>) -> MyFloat {
        (self.0)(x)
    }

    fn call_scalar(&mut self, x: &MyFloat) -> MyFloat {
        (self.0)(&array![x.clone()])
    }
}

// Implementation for multi-dimensional functions
impl<F> ObjectiveDerFn for MultiDimFn<F>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
{
    fn df(&self, x: &Array1<MyFloat>) -> MyFloat {
        (self.0)(x)
    }

    fn df_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.0)(&array![x.clone()])
    }
}

// // Usage examples:
//
// use ndarray::prelude::*;
// use rfkit_base_ndarray::prelude::*;
//
// // Example 1: Simple function (same as before)
// fn simple_objective(x: Array1<MyFloat>) -> MyFloat {
//     // Your existing objective function
//     MyFloat::new(0.0)
// }
//
// // Example 2: Closure with captured variables
// fn example_usage() {
//     let x = array![1e-11, 1e-3, 1e-13, 1e-6];
//     let scale = array![1e12, 1.0, 1e12, 1.0];
//
//     // Additional parameters you want to pass
//     let weight_factor = 2.0;
//     let penalty_term = 0.1;
//     let reference_data = vec![1.0, 2.0, 3.0];
//
//     // Create closure that captures additional arguments
//     let objective_with_params = move |x: Array1<MyFloat>| -> MyFloat {
//         // Your objective calculation using captured variables
//         let base_error = simple_objective(x.clone());
//         let weighted_error = base_error * weight_factor;
//         let penalty = MyFloat::new(penalty_term) * reference_data.len() as f64;
//         weighted_error + penalty
//     };
//
//     // Use the closure
//     let mut optimizer = NelderMead::new(x, scale, objective_with_params);
//     optimizer.solve(100);
// }
//
// // Example 3: Struct-based approach for complex scenarios
// pub struct ComplexObjective {
//     weights: Array1<f64>,
//     reference_measurements: Array3<Complex64>,
//     regularization_param: f64,
//     measurement_type: MeasurementType,
// }
//
// pub enum MeasurementType {
//     SParameters,
//     YParameters,
//     ZParameters,
// }
//
// impl ObjectiveFn for ComplexObjective {
//     fn call(&self, x: Array1<MyFloat>) -> MyFloat {
//         // Use all the struct fields in your calculation
//         let base_error = self.calculate_base_error(x);
//         let regularization = self.apply_regularization(&x);
//         base_error + regularization
//     }
// }
//
// impl ComplexObjective {
//     fn calculate_base_error(&self, x: Array1<MyFloat>) -> MyFloat {
//         // Implementation using self.weights, self.reference_measurements, etc.
//         MyFloat::new(0.0)
//     }
//
//     fn apply_regularization(&self, x: &Array1<MyFloat>) -> MyFloat {
//         // Use self.regularization_param
//         MyFloat::new(0.0)
//     }
// }
//
// // Usage with struct:
// fn example_struct_usage() {
//     let x = array![1e-11, 1e-3, 1e-13, 1e-6];
//     let scale = array![1e12, 1.0, 1e12, 1.0];
//
//     let objective = ComplexObjective {
//         weights: array![1.0, 2.0, 1.5, 0.8],
//         reference_measurements: Array3::zeros((100, 2, 2)),
//         regularization_param: 0.01,
//         measurement_type: MeasurementType::SParameters,
//     };
//
//     let mut optimizer = NelderMead::new(x, scale, objective);
//     optimizer.solve(100);
// }
pub trait Minimizer {
    /// Calculate the objective function
    fn calc_obj(&mut self, x: &Array1<MyFloat>) -> MyFloat;

    /// Run the optimization for specified iterations
    fn solve(&mut self, max_iters: usize);

    /// Get the final solution
    fn x(&self) -> &Array1<f64>;

    /// Get the final objective value
    fn final_value(&self) -> Option<f64>;

    /// Get the tolerance/convergence metric
    fn tolerance(&self) -> Option<f64>;

    /// Get the number of iterations performed
    fn iterations(&self) -> usize;

    /// Get optimizer name for reporting
    fn name(&self) -> &str;
}
