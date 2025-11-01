#![allow(dead_code)]
use crate::myfloat::MyFloat;
use dyn_clone::DynClone;
use ndarray::prelude::*;

pub mod f64;
pub mod myfloat;

// Define a trait for the objective function
pub trait ObjFn<T>: DynClone {
    fn call(&self, x: &Array1<T>) -> T;
    fn call_scalar(&self, x: &T) -> T;
}
dyn_clone::clone_trait_object!(<T> ObjFn<T>);

// Define a trait for the derivative function
pub trait ObjDerFn<T>: ObjFn<T> + DynClone {
    fn df(&self, x: &Array1<T>) -> T;
    fn df_scalar(&self, x: &T) -> T;
}
dyn_clone::clone_trait_object!(<T> ObjDerFn<T>);

// Define a trait for the gradient function
pub trait ObjGradFn<T>: ObjFn<T> + DynClone {
    fn grad(&self, x: &Array1<T>) -> Array1<T>;
    fn grad_scalar(&self, x: &T) -> Array1<T>;
}
dyn_clone::clone_trait_object!(<T> ObjGradFn<T>);

// Define a trait for the hessian function
pub trait ObjHessFn<T>: ObjGradFn<T> + DynClone {
    fn hess(&self, x: &Array1<T>) -> Array2<T>;
    fn hess_scalar(&self, x: &T) -> Array2<T>;
}
dyn_clone::clone_trait_object!(<T> ObjHessFn<T>);

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
impl<F> ObjFn<MyFloat> for SingleDimFn<F>
where
    F: Fn(&MyFloat) -> MyFloat + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        // Take the first element for single-dim functions
        (self.0)(&x[0])
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
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
impl<F, DF> ObjFn<MyFloat> for SingleDimDerFn<F, DF>
where
    F: Fn(&MyFloat) -> MyFloat + Clone,
    DF: Fn(&MyFloat) -> MyFloat + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        // Take the first element for single-dim functions
        (self.0)(&x[0])
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.0)(x)
    }
}

// Implementation for single-dimensional functions
impl<F, DF> ObjDerFn<MyFloat> for SingleDimDerFn<F, DF>
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
impl<F> ObjFn<MyFloat> for MultiDimFn<F>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.0)(&array![x.clone()])
    }
}

// Implementation for multi-dimensional functions
impl<F> ObjDerFn<MyFloat> for MultiDimFn<F>
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
pub trait Minimizer<T> {
    /// Run the optimization for specified iterations
    fn minimize(&mut self, max_iters: Option<usize>) -> Box<dyn MinimizerResult<T>>;
}

pub trait MinimizerResult<T> {
    fn xmin(&self) -> T;
    fn fmin(&self) -> f64;
    fn fn_evals(&self) -> usize;
    fn iters(&self) -> usize;
    fn converged(&self) -> bool;
}
