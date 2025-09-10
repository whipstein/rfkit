#![allow(dead_code)]
use crate::minimize::{
    MinimizerError,
    f64::{Constraint, Matrix},
};
use dyn_clone::DynClone;
// use ndarray::prelude::*;

// Define a trait for the objective function
pub trait ObjFn: DynClone {
    fn call(&self, x: &Vec<f64>) -> f64;
    fn call_scalar(&self, x: f64) -> f64;
}
dyn_clone::clone_trait_object!(ObjFn);

// Define a trait for the derivative function
pub trait ObjDerFn: ObjFn + DynClone {
    fn df(&self, x: &Vec<f64>) -> f64;
    fn df_scalar(&self, x: f64) -> f64;
}
dyn_clone::clone_trait_object!(ObjDerFn);

// Define a trait for the gradient function
pub trait ObjGradFn: ObjFn + DynClone {
    fn grad(&self, x: &Vec<f64>) -> Vec<f64>;
    fn grad_scalar(&self, x: f64) -> Vec<f64>;
}
dyn_clone::clone_trait_object!(ObjGradFn);

// Define a trait for the hessian function
pub trait ObjHessFn: ObjGradFn + DynClone {
    fn hessian(&self, x: &Vec<f64>) -> Vec<Vec<f64>>;
    fn hessian_scalar(&self, x: f64) -> Vec<Vec<f64>>;
}
dyn_clone::clone_trait_object!(ObjHessFn);

impl<F> ObjFn for F
where
    F: Fn(&Vec<f64>) -> f64 + DynClone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        self(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        self(&vec![x])
    }
}

impl<F, DF> ObjFn for (F, DF)
where
    F: Fn(&Vec<f64>) -> f64 + DynClone + Clone,
    DF: Fn(&Vec<f64>) -> f64 + DynClone + Clone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        self.0(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        self.0(&vec![x])
    }
}

impl<F, DF, GF> ObjFn for (F, DF, GF)
where
    F: Fn(&Vec<f64>) -> f64 + DynClone + Clone,
    DF: Fn(&Vec<f64>) -> f64 + DynClone + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + DynClone + Clone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        self.0(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        self.0(&vec![x])
    }
}

impl<F, DF> ObjDerFn for (F, DF)
where
    F: Fn(&Vec<f64>) -> f64 + DynClone + Clone,
    DF: Fn(&Vec<f64>) -> f64 + DynClone + Clone,
{
    fn df(&self, x: &Vec<f64>) -> f64 {
        self.1(x)
    }

    fn df_scalar(&self, x: f64) -> f64 {
        self.1(&vec![x])
    }
}

// impl<F, DF, GF> ObjGradFn for (F, DF, GF)
// where
//     F: Fn(&Vec<f64>) -> f64 + DynClone + Clone,
//     DF: Fn(&Vec<f64>) -> f64 + DynClone + Clone,
//     GF: Fn(&Vec<f64>) -> Vec<f64> + DynClone + Clone,
// {
//     fn grad(&self, x: &Vec<f64>) -> Vec<f64> {
//         self.2(x)
//     }

//     fn grad_scalar(&self, x: f64) -> Vec<f64> {
//         self.2(&vec![x])
//     }
// }

// Wrapper for single-dimensional functions
#[derive(Clone)]
pub struct SingleDimFn<F>(pub F)
where
    F: Fn(f64) -> f64 + Clone;

// Convenience constructors
impl<F> SingleDimFn<F>
where
    F: Fn(f64) -> f64 + Clone,
{
    pub fn new(f: F) -> Self {
        SingleDimFn(f)
    }
}

// Implementation for single-dimensional functions
impl<F> ObjFn for SingleDimFn<F>
where
    F: Fn(f64) -> f64 + Clone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        // Take the first element for single-dim functions
        (self.0)(x[0])
    }

    fn call_scalar(&self, x: f64) -> f64 {
        (self.0)(x)
    }
}

// Wrapper for single-dimensional function w/derivative
#[derive(Clone)]
pub struct SingleDimDerFn<F, DF>(pub F, pub DF)
where
    F: Fn(f64) -> f64 + Clone,
    DF: Fn(f64) -> f64 + Clone;

// Convenience constructors
impl<F, DF> SingleDimDerFn<F, DF>
where
    F: Fn(f64) -> f64 + Clone,
    DF: Fn(f64) -> f64 + Clone,
{
    pub fn new(f: F, df: DF) -> Self {
        SingleDimDerFn(f, df)
    }
}

// Implementation for single-dimensional function w/derivative
impl<F, DF> ObjFn for SingleDimDerFn<F, DF>
where
    F: Fn(f64) -> f64 + Clone,
    DF: Fn(f64) -> f64 + Clone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        // Take the first element for single-dim functions
        (self.0)(x[0])
    }

    fn call_scalar(&self, x: f64) -> f64 {
        (self.0)(x)
    }
}

impl<F, DF> ObjDerFn for SingleDimDerFn<F, DF>
where
    F: Fn(f64) -> f64 + Clone,
    DF: Fn(f64) -> f64 + Clone,
{
    fn df(&self, x: &Vec<f64>) -> f64 {
        // Take the first element for single-dim functions
        (self.1)(x[0])
    }

    fn df_scalar(&self, x: f64) -> f64 {
        (self.1)(x)
    }
}

// Wrapper for multi-dimensional functions
#[derive(Clone)]
pub struct MultiDimFn<F>(pub F)
where
    F: Fn(&Vec<f64>) -> f64 + Clone;

// Convenience constructors
impl<F> MultiDimFn<F>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
{
    pub fn new(f: F) -> Self {
        MultiDimFn(f)
    }
}

// Implementation for multi-dimensional functions
impl<F> ObjFn for MultiDimFn<F>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        (self.0)(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        (self.0)(&vec![x])
    }
}

impl<F> ObjDerFn for MultiDimFn<F>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
{
    fn df(&self, x: &Vec<f64>) -> f64 {
        (self.0)(x)
    }

    fn df_scalar(&self, x: f64) -> f64 {
        (self.0)(&vec![x])
    }
}

// Wrapper for multi-dimensional function w/gradient
#[derive(Clone)]
pub struct MultiDimGradFn<F, GF>(pub F, pub GF)
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone;

// Convenience constructors
impl<F, GF> MultiDimGradFn<F, GF>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone,
{
    pub fn new(f: F, gf: GF) -> Self {
        MultiDimGradFn(f, gf)
    }
}

// Implementation for multi-dimensional function w/gradient
impl<F, GF> ObjFn for MultiDimGradFn<F, GF>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        (self.0)(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        (self.0)(&vec![x])
    }
}

impl<F, GF> ObjGradFn for MultiDimGradFn<F, GF>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone,
{
    fn grad(&self, x: &Vec<f64>) -> Vec<f64> {
        (self.1)(x)
    }

    fn grad_scalar(&self, x: f64) -> Vec<f64> {
        (self.1)(&vec![x])
    }
}

// Wrapper for multi-dimensional function w/numerical gradient
#[derive(Clone)]
pub struct MultiDimNumGradFn<F>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
{
    f: F,
    step: f64,
    n: usize,
}

// Convenience constructors
impl<F> MultiDimNumGradFn<F>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
{
    pub fn new(f: F, step: Option<f64>, n: usize) -> Self {
        Self {
            f,
            step: step.unwrap_or(1e-8),
            n,
        }
    }

    pub fn numerical_gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut grad = vec![0.0; self.n];

        for i in 0..self.n {
            let mut x_plus_h = x.to_vec();
            x_plus_h[i] += self.step;
            let f_plus_h = (self.f)(&x_plus_h);

            let mut x_minus_h = x.to_vec();
            x_minus_h[i] -= self.step;
            let f_minus_h = (self.f)(&x_minus_h);

            grad[i] = (f_plus_h - f_minus_h) / (2.0 * self.step);
        }

        grad
    }
}

// Implementation for multi-dimensional function w/gradient
impl<F> ObjFn for MultiDimNumGradFn<F>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        (self.f)(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        (self.f)(&vec![x])
    }
}

impl<F> ObjGradFn for MultiDimNumGradFn<F>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
{
    fn grad(&self, x: &Vec<f64>) -> Vec<f64> {
        self.numerical_gradient(x)
    }

    fn grad_scalar(&self, x: f64) -> Vec<f64> {
        self.numerical_gradient(&vec![x])
    }
}

// Wrapper for multi-dimensional function w/hessian
#[derive(Clone)]
pub struct MultiDimHessFn<F, GF, HF>(pub F, pub GF, pub Option<HF>)
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone,
    HF: Fn(&Vec<f64>) -> Vec<Vec<f64>> + Clone;

// Convenience constructors
impl<F, GF, HF> MultiDimHessFn<F, GF, HF>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone,
    HF: Fn(&Vec<f64>) -> Vec<Vec<f64>> + Clone,
{
    pub fn new(f: F, gf: GF, hf: Option<HF>) -> Self {
        MultiDimHessFn(f, gf, hf)
    }
}

// Implementation for multi-dimensional function w/hessian
impl<F, GF, HF> ObjFn for MultiDimHessFn<F, GF, HF>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone,
    HF: Fn(&Vec<f64>) -> Vec<Vec<f64>> + Clone,
{
    fn call(&self, x: &Vec<f64>) -> f64 {
        (self.0)(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        (self.0)(&vec![x])
    }
}

impl<F, GF, HF> ObjGradFn for MultiDimHessFn<F, GF, HF>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone,
    HF: Fn(&Vec<f64>) -> Vec<Vec<f64>> + Clone,
{
    fn grad(&self, x: &Vec<f64>) -> Vec<f64> {
        (self.1)(x)
    }

    fn grad_scalar(&self, x: f64) -> Vec<f64> {
        (self.1)(&vec![x])
    }
}

impl<F, GF, HF> ObjHessFn for MultiDimHessFn<F, GF, HF>
where
    F: Fn(&Vec<f64>) -> f64 + Clone,
    GF: Fn(&Vec<f64>) -> Vec<f64> + Clone,
    HF: Fn(&Vec<f64>) -> Vec<Vec<f64>> + Clone,
{
    fn hessian(&self, x: &Vec<f64>) -> Vec<Vec<f64>> {
        match self.2.clone() {
            Some(hf) => (hf)(x),
            _ => Matrix::identity(x.len()),
        }
    }

    fn hessian_scalar(&self, x: f64) -> Vec<Vec<f64>> {
        match self.2.clone() {
            Some(hf) => (hf)(&vec![x]),
            _ => Matrix::identity(1),
        }
    }
}

#[derive(Clone)]
pub struct F1dim {
    f: Box<dyn ObjFn>,
}

impl F1dim {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn + 'static,
    {
        F1dim { f: Box::new(f) }
    }

    pub fn new_boxed(f: Box<dyn ObjFn>) -> Self {
        F1dim { f }
    }

    pub fn eval(
        &mut self,
        point: &Vec<f64>,
        direction: &Vec<f64>,
        t: f64,
    ) -> Result<f64, MinimizerError> {
        let test_point: Vec<f64> = point
            .iter()
            .zip(direction.iter())
            .map(|(&p, &d)| p + t * d)
            .collect();
        let value = self.f.call(&test_point);
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(value)
    }
}

impl ObjFn for F1dim {
    fn call(&self, x: &Vec<f64>) -> f64 {
        self.f.call(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        self.f.call_scalar(x)
    }
}

#[derive(Clone)]
pub struct HF1dim {
    f: Box<dyn ObjHessFn>,
    ieq: Vec<Box<dyn Constraint>>,
    eq: Vec<Box<dyn Constraint>>,
    mu: f64,
}

impl HF1dim {
    pub fn new<F>(
        f: F,
        ieq: &Vec<Box<dyn Constraint>>,
        eq: &Vec<Box<dyn Constraint>>,
        mu: Option<f64>,
    ) -> Self
    where
        F: ObjHessFn + 'static,
    {
        HF1dim {
            f: Box::new(f),
            ieq: ieq.clone(),
            eq: eq.clone(),
            mu: match mu {
                Some(x) => x,
                _ => 0.0,
            },
        }
    }

    pub fn new_boxed(
        f: Box<dyn ObjHessFn>,
        ieq: &Vec<Box<dyn Constraint>>,
        eq: &Vec<Box<dyn Constraint>>,
        mu: Option<f64>,
    ) -> Self {
        HF1dim {
            f,
            ieq: ieq.clone(),
            eq: eq.clone(),
            mu: match mu {
                Some(x) => x,
                _ => 0.0,
            },
        }
    }

    pub fn objective(&self, x: &Vec<f64>) -> f64 {
        let mut result = self.f.call(x);

        // Add barrier terms for ieq
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= -1e-12 {
                return f64::INFINITY; // Outside feasible region
            }
            result -= self.mu * val.ln();
        }

        result
    }

    pub fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut grad = self.f.grad(x);
        let n = x.len();

        // Add barrier gradient terms
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= -1e-12 {
                return vec![f64::INFINITY; n]; // Outside feasible region
            }
            let constraint_grad = constraint.gradient(x);
            let factor = -self.mu / val;

            for i in 0..n {
                grad[i] += factor * constraint_grad[i];
            }
        }

        grad
    }

    pub fn hess(&self, x: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut hess = self.f.hessian(x);
        let n = x.len();

        // Add barrier Hessian terms
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= -1e-12 {
                // Return identity matrix with large diagonal (penalty)
                let mut penalty_hess = Matrix::identity(n);
                Matrix::scalar_multiply(&mut penalty_hess, 1e6);
                return penalty_hess;
            }

            let constraint_grad = constraint.gradient(x);
            let constraint_hess = constraint.hessian(x);

            let factor1 = -self.mu / val;
            let factor2 = self.mu / (val * val);

            // Add second derivative terms
            for i in 0..n {
                for j in 0..n {
                    hess[i][j] += factor1 * constraint_hess[i][j]
                        + factor2 * constraint_grad[i] * constraint_grad[j];
                }
            }
        }

        hess
    }
}

impl ObjFn for HF1dim {
    fn call(&self, x: &Vec<f64>) -> f64 {
        self.objective(x)
    }

    fn call_scalar(&self, x: f64) -> f64 {
        self.objective(&vec![x])
    }
}

impl ObjGradFn for HF1dim {
    fn grad(&self, x: &Vec<f64>) -> Vec<f64> {
        self.gradient(x)
    }

    fn grad_scalar(&self, x: f64) -> Vec<f64> {
        self.gradient(&vec![x])
    }
}

impl ObjHessFn for HF1dim {
    fn hessian(&self, x: &Vec<f64>) -> Vec<Vec<f64>> {
        self.hess(x)
    }

    fn hessian_scalar(&self, x: f64) -> Vec<Vec<f64>> {
        self.hess(&vec![x])
    }
}
