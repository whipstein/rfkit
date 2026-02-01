#![allow(dead_code)]
use crate::{
    error::MinimizerError,
    num::RFFloat,
    pts::{Matrix, Points1, Points2, Pts},
};
use dyn_clone::DynClone;
use ndarray::prelude::*;
use std::fmt;

pub mod bracket;
pub mod brent;
pub mod cma_es;
pub mod conjugate_gradient;
pub mod dbrent;
pub mod golden;
pub mod interior_point;
pub mod nelder_mead;
// pub mod nelder_mead_bounded;
pub mod powell;
pub mod quasi_newton;
pub mod simplex;

pub use self::bracket::{Bracket, BracketOptions, BracketResult};
pub use self::brent::{Brent, BrentResult};
pub use self::cma_es::{CmaEs, CmaEsResult};
pub use self::conjugate_gradient::{ConjGrad, ConjGradMethod, ConjGradResult};
pub use self::dbrent::{DBrent, DBrentMethod, DBrentResult};
pub use self::golden::{Golden, GoldenResult};
pub use self::interior_point::{
    InteriorPoint, InteriorPointMethod, InteriorPointParams, InteriorPointResult,
};
pub use self::nelder_mead::{NelderMead, NelderMeadMethod, NelderMeadOptions, NelderMeadResult};
pub use self::powell::{Powell, PowellResult};
pub use self::quasi_newton::{QuasiNewton, QuasiNewtonMethod, QuasiNewtonResult};
pub use self::simplex::{Simplex, SimplexResult};

// Define a trait for the objective function
pub trait ObjFn<T>: DynClone {
    fn call(&self, x: &Points1<T>) -> T;
    fn call_scalar(&self, x: &T) -> T;
}
dyn_clone::clone_trait_object!(<T> ObjFn<T>);

// Define a trait for the derivative function
pub trait ObjDerFn<T>: ObjFn<T> + DynClone {
    fn df(&self, x: &Points1<T>) -> T;
    fn df_scalar(&self, x: &T) -> T;
}
dyn_clone::clone_trait_object!(<T> ObjDerFn<T>);

// Define a trait for the gradient function
pub trait ObjGradFn<T>: ObjFn<T> + DynClone {
    fn grad(&self, x: &Points1<T>) -> Points1<T>;
    fn grad_scalar(&self, x: &T) -> Points1<T>;
}
dyn_clone::clone_trait_object!(<T> ObjGradFn<T>);

// Define a trait for the hessian function
pub trait ObjHessFn<T>: ObjGradFn<T> + DynClone {
    fn hess(&self, x: &Points1<T>) -> Points2<T>;
    fn hess_scalar(&self, x: &T) -> Points2<T>;
}
dyn_clone::clone_trait_object!(<T> ObjHessFn<T>);

// Wrapper for single-dimensional functions
#[derive(Clone)]
pub struct SingleDimFn<T, F>(pub F, std::marker::PhantomData<T>)
where
    F: Fn(&T) -> T + Clone,
    T: RFFloat;

// Convenience constructors
impl<T, F> SingleDimFn<T, F>
where
    F: Fn(&T) -> T + Clone,
    T: RFFloat,
{
    pub fn new(f: F) -> Self {
        SingleDimFn(f, std::marker::PhantomData)
    }
}

// Implementation for single-dimensional functions
impl<T, F> ObjFn<T> for SingleDimFn<T, F>
where
    F: Fn(&T) -> T + Clone,
    T: RFFloat,
{
    fn call(&self, x: &Points1<T>) -> T {
        // Take the first element for single-dim functions
        (self.0)(&x[0])
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.0)(x)
    }
}

// Wrapper for single-dimensional function w/derivative
#[derive(Clone)]
pub struct SingleDimDerFn<T, F, DF>(pub F, pub DF, std::marker::PhantomData<T>)
where
    F: Fn(&T) -> T + Clone,
    DF: Fn(&T) -> T + Clone,
    T: RFFloat;

// Convenience constructors
impl<T, F, DF> SingleDimDerFn<T, F, DF>
where
    F: Fn(&T) -> T + Clone,
    DF: Fn(&T) -> T + Clone,
    T: RFFloat,
{
    pub fn new(f: F, df: DF) -> Self {
        SingleDimDerFn(f, df, std::marker::PhantomData)
    }
}

// Implementation for single-dimensional function w/derivative with f64
impl<T, F, DF> ObjFn<T> for SingleDimDerFn<T, F, DF>
where
    F: Fn(&T) -> T + Clone,
    DF: Fn(&T) -> T + Clone,
    T: RFFloat,
{
    fn call(&self, x: &Points1<T>) -> T {
        // Take the first element for single-dim functions
        (self.0)(&x[0])
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.0)(x)
    }
}

impl<T, F, DF> ObjDerFn<T> for SingleDimDerFn<T, F, DF>
where
    F: Fn(&T) -> T + Clone,
    DF: Fn(&T) -> T + Clone,
    T: RFFloat,
{
    fn df(&self, x: &Points1<T>) -> T {
        // Take the first element for single-dim functions
        (self.1)(&x[0])
    }

    fn df_scalar(&self, x: &T) -> T {
        (self.1)(x)
    }
}

// Wrapper for multi-dimensional functions
#[derive(Clone)]
pub struct MultiDimFn<T, F>(pub F, std::marker::PhantomData<T>)
where
    F: Fn(&Points1<T>) -> T + Clone,
    T: RFFloat;

// Convenience constructors
impl<T, F> MultiDimFn<T, F>
where
    F: Fn(&Points1<T>) -> T + Clone,
    T: RFFloat,
{
    pub fn new(f: F) -> Self {
        MultiDimFn(f, std::marker::PhantomData)
    }
}

// Implementation for multi-dimensional functions
impl<T, F> ObjFn<T> for MultiDimFn<T, F>
where
    F: Fn(&Points1<T>) -> T + Clone,
    T: RFFloat,
{
    fn call(&self, x: &Points1<T>) -> T {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.0)(&array![x.clone()].into())
    }
}

// Implementation for multi-dimensional functions
impl<T, F> ObjDerFn<T> for MultiDimFn<T, F>
where
    F: Fn(&Points1<T>) -> T + Clone,
    T: RFFloat,
{
    fn df(&self, x: &Points1<T>) -> T {
        (self.0)(x)
    }

    fn df_scalar(&self, x: &T) -> T {
        (self.0)(&array![x.clone()].into())
    }
}

// Wrapper for multi-dimensional function w/gradient
#[derive(Clone)]
pub struct MultiDimGradFn<T, F, GF>(pub F, pub GF, std::marker::PhantomData<T>)
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    T: RFFloat;

// Convenience constructors
impl<T, F, GF> MultiDimGradFn<T, F, GF>
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    T: RFFloat,
{
    pub fn new(f: F, gf: GF) -> Self {
        MultiDimGradFn(f, gf, std::marker::PhantomData)
    }
}

// Implementation for multi-dimensional function w/gradient
impl<T, F, GF> ObjFn<T> for MultiDimGradFn<T, F, GF>
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    T: RFFloat,
{
    fn call(&self, x: &Points1<T>) -> T {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.0)(&array![x.clone()].into())
    }
}

impl<T, F, GF> ObjGradFn<T> for MultiDimGradFn<T, F, GF>
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    T: RFFloat,
{
    fn grad(&self, x: &Points1<T>) -> Points1<T> {
        (self.1)(x)
    }

    fn grad_scalar(&self, x: &T) -> Points1<T> {
        (self.1)(&array![x.clone()].into())
    }
}

// Wrapper for multi-dimensional function w/numerical gradient
#[derive(Clone)]
pub struct MultiDimNumGradFn<T, F>
where
    F: Fn(&Points1<T>) -> T + Clone,
    T: RFFloat,
{
    f: F,
    step: T,
    n: usize,
}

// Convenience constructors
impl<T, F> MultiDimNumGradFn<T, F>
where
    F: Fn(&Points1<T>) -> T + Clone,
    T: RFFloat,
    for<'a, 'b> &'a Points1<T>: std::ops::Add<&'b T, Output = Points1<T>>,
    for<'a, 'b> &'a Points1<T>: std::ops::Sub<&'b T, Output = Points1<T>>,
    for<'a> &'a T: std::ops::Mul<f64, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
{
    pub fn new(f: F, step: Option<T>, n: usize) -> Self {
        Self {
            f,
            step: step.unwrap_or(T::from_f64(1e-8)),
            n,
        }
    }

    pub fn numerical_gradient(&self, x: &Points1<T>) -> Points1<T> {
        let mut x_plus_h = x.clone();
        let mut x_minus_h = x.clone();
        let mut f_plus_h = Points1::zeros(self.n);
        let mut f_minus_h = Points1::zeros(self.n);

        for i in 0..self.n {
            x_plus_h[i] += &self.step;
            f_plus_h[i] = (self.f)(&x_plus_h);

            x_minus_h[i] -= &self.step;
            f_minus_h[i] = (self.f)(&x_minus_h);
        }

        (f_plus_h - f_minus_h) / (&self.step * 2.0)
    }
}

// Implementation for multi-dimensional function w/gradient
impl<T, F> ObjFn<T> for MultiDimNumGradFn<T, F>
where
    F: Fn(&Points1<T>) -> T + Clone,
    T: RFFloat,
{
    fn call(&self, x: &Points1<T>) -> T {
        (self.f)(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.f)(&array![x.clone()].into())
    }
}

impl<T, F> ObjGradFn<T> for MultiDimNumGradFn<T, F>
where
    F: Fn(&Points1<T>) -> T + Clone,
    T: RFFloat,
    for<'a, 'b> &'a Points1<T>: std::ops::Add<&'b T, Output = Points1<T>>,
    for<'a, 'b> &'a Points1<T>: std::ops::Sub<&'b T, Output = Points1<T>>,
    for<'a> &'a T: std::ops::Mul<f64, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
{
    fn grad(&self, x: &Points1<T>) -> Points1<T> {
        self.numerical_gradient(x)
    }

    fn grad_scalar(&self, x: &T) -> Points1<T> {
        self.numerical_gradient(&array![x.clone()].into())
    }
}

// Wrapper for multi-dimensional function w/hessian
#[derive(Clone)]
pub struct MultiDimHessFn<T, F, GF, HF>(pub F, pub GF, pub Option<HF>, std::marker::PhantomData<T>)
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    HF: Fn(&Points1<T>) -> Points2<T> + Clone,
    T: RFFloat;

// Convenience constructors
impl<T, F, GF, HF> MultiDimHessFn<T, F, GF, HF>
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    HF: Fn(&Points1<T>) -> Points2<T> + Clone,
    T: RFFloat,
{
    pub fn new(f: F, gf: GF, hf: Option<HF>) -> Self {
        MultiDimHessFn(f, gf, hf, std::marker::PhantomData)
    }
}

// Implementation for multi-dimensional function w/hessian
impl<T, F, GF, HF> ObjFn<T> for MultiDimHessFn<T, F, GF, HF>
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    HF: Fn(&Points1<T>) -> Points2<T> + Clone,
    T: RFFloat,
{
    fn call(&self, x: &Points1<T>) -> T {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.0)(&array![x.clone()].into())
    }
}

impl<T, F, GF, HF> ObjGradFn<T> for MultiDimHessFn<T, F, GF, HF>
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    HF: Fn(&Points1<T>) -> Points2<T> + Clone,
    T: RFFloat,
{
    fn grad(&self, x: &Points1<T>) -> Points1<T> {
        (self.1)(x)
    }

    fn grad_scalar(&self, x: &T) -> Points1<T> {
        (self.1)(&array![x.clone()].into())
    }
}

impl<T, F, GF, HF> ObjHessFn<T> for MultiDimHessFn<T, F, GF, HF>
where
    F: Fn(&Points1<T>) -> T + Clone,
    GF: Fn(&Points1<T>) -> Points1<T> + Clone,
    HF: Fn(&Points1<T>) -> Points2<T> + Clone,
    T: RFFloat,
{
    fn hess(&self, x: &Points1<T>) -> Points2<T> {
        match self.2.clone() {
            Some(hf) => (hf)(x),
            _ => Points2::eye(x.len()),
        }
    }

    fn hess_scalar(&self, x: &T) -> Points2<T> {
        match self.2.clone() {
            Some(hf) => (hf)(&array![x.clone()].into()),
            _ => Points2::eye(1),
        }
    }
}

// Multi-dimensional function along a direction
#[derive(Clone)]
pub struct F1dim<T>
where
    T: RFFloat,
{
    f: Box<dyn ObjFn<T>>,
}

impl<T> F1dim<T>
where
    T: RFFloat,
    for<'a> &'a T: std::ops::Add<T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn<T> + 'static,
    {
        F1dim { f: Box::new(f) }
    }

    pub fn new_boxed(f: Box<dyn ObjFn<T>>) -> Self {
        F1dim { f }
    }

    pub fn eval(
        &mut self,
        point: &Points1<T>,
        direction: &Points1<T>,
        t: &T,
    ) -> Result<T, MinimizerError> {
        let test_point: Points1<T> = point
            .iter()
            .zip(direction.iter())
            .map(|(p, d)| p + t * d)
            .collect();
        let value = self.f.call(&test_point);
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(value)
    }
}

impl<T> ObjFn<T> for F1dim<T>
where
    T: RFFloat,
{
    fn call(&self, x: &Points1<T>) -> T {
        self.f.call(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        self.f.call_scalar(x)
    }
}

// Multi-dimensional function along a direction with gradient
#[derive(Clone)]
pub struct GF1dim<T>
where
    T: RFFloat,
{
    f: Box<dyn ObjGradFn<T>>,
}

impl<T> GF1dim<T>
where
    T: RFFloat,
{
    pub fn new<F>(f: F) -> Self
    where
        F: ObjGradFn<T> + 'static,
    {
        GF1dim { f: Box::new(f) }
    }

    pub fn new_boxed(f: Box<dyn ObjGradFn<T>>) -> Self {
        GF1dim { f }
    }
}

impl<T> ObjFn<T> for GF1dim<T>
where
    T: RFFloat,
{
    fn call(&self, x: &Points1<T>) -> T {
        self.f.call(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        self.f.call_scalar(x)
    }
}

impl<T> ObjGradFn<T> for GF1dim<T>
where
    T: RFFloat,
{
    fn grad(&self, x: &Points1<T>) -> Points1<T> {
        self.f.grad(x)
    }

    fn grad_scalar(&self, x: &T) -> Points1<T> {
        self.f.grad(&array![x.clone()].into())
    }
}

// Multi-dimensional function along a direction with gradient and hessian
#[derive(Clone)]
pub struct HF1dim<T>
where
    T: RFFloat,
{
    f: Box<dyn ObjHessFn<T>>,
    ieq: Vec<Box<dyn Constraint<T>>>,
    eq: Vec<Box<dyn Constraint<T>>>,
    mu: T,
}

impl<T> HF1dim<T>
where
    T: RFFloat,
    for<'a> &'a T: std::ops::Div<T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
{
    pub fn new<F>(
        f: F,
        ieq: &Vec<Box<dyn Constraint<T>>>,
        eq: &Vec<Box<dyn Constraint<T>>>,
        mu: Option<T>,
    ) -> Self
    where
        F: ObjHessFn<T> + 'static,
    {
        HF1dim {
            f: Box::new(f),
            ieq: ieq.clone(),
            eq: eq.clone(),
            mu: match mu {
                Some(x) => x,
                _ => T::zero(),
            },
        }
    }

    pub fn new_boxed(
        f: Box<dyn ObjHessFn<T>>,
        ieq: &Vec<Box<dyn Constraint<T>>>,
        eq: &Vec<Box<dyn Constraint<T>>>,
        mu: Option<T>,
    ) -> Self {
        HF1dim {
            f,
            ieq: ieq.clone(),
            eq: eq.clone(),
            mu: match mu {
                Some(x) => x,
                _ => T::zero(),
            },
        }
    }

    pub fn objective(&self, x: &Points1<T>) -> T {
        let mut result = self.f.call(x);

        // Add barrier terms for ieq
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= T::from_f64(-1e-12) {
                return T::from_f64(f64::INFINITY); // Outside feasible region
            }
            result -= self.mu.clone() * val.ln();
        }

        result
    }

    pub fn gradient(&self, x: &Points1<T>) -> Points1<T> {
        let mut grad = self.f.grad(x);
        let n = x.len();

        // Add barrier gradient terms
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= T::from_f64(-1e-12) {
                return Points1::from_vec(vec![T::from_f64(f64::INFINITY); n]); // Outside feasible region
            }
            let constraint_grad = constraint.gradient(x);
            let factor = (-self.mu.clone()) / val;

            for i in 0..n {
                grad[i] += &factor * &constraint_grad[i];
            }
        }

        grad
    }

    pub fn hess(&self, x: &Points1<T>) -> Points2<T> {
        let mut hess = self.f.hess(x);
        let n = x.len();

        // Add barrier Hessian terms
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= T::from_f64(-1e-12) {
                // Return identity matrix with large diagonal (penalty)
                let mut penalty_hess = Points2::<T>::eye(n);
                penalty_hess *= &Points2::<T>::from_shape_fn((n, n), |(_, _)| T::from_f64(1e6));
                return penalty_hess;
            }

            let constraint_grad = constraint.gradient(x);
            let constraint_hess = constraint.hessian(x);

            let factor1 = -self.mu.clone() / &val;
            let factor2 = &self.mu / (&val * &val);

            // Add second derivative terms
            for i in 0..n {
                for j in 0..n {
                    hess[[i, j]] += &factor1 * &constraint_hess[[i, j]]
                        + &factor2 * &constraint_grad[i] * &constraint_grad[j];
                }
            }
        }

        hess
    }
}

impl<T> ObjFn<T> for HF1dim<T>
where
    T: RFFloat,
    for<'a> &'a T: std::ops::Div<T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
{
    fn call(&self, x: &Points1<T>) -> T {
        self.objective(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        self.objective(&array![x.clone()].into())
    }
}

impl<T> ObjGradFn<T> for HF1dim<T>
where
    T: RFFloat,
    for<'a> &'a T: std::ops::Div<T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
{
    fn grad(&self, x: &Points1<T>) -> Points1<T> {
        self.gradient(x)
    }

    fn grad_scalar(&self, x: &T) -> Points1<T> {
        self.gradient(&array![x.clone()].into())
    }
}

impl<T> ObjHessFn<T> for HF1dim<T>
where
    T: RFFloat,
    for<'a> &'a T: std::ops::Div<T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Div<&'b T, Output = T>,
{
    fn hess(&self, x: &Points1<T>) -> Points2<T> {
        self.hess(x)
    }

    fn hess_scalar(&self, x: &T) -> Points2<T> {
        self.hess(&array![x.clone()].into())
    }
}

/// Constraint definition
pub trait Constraint<T>: DynClone
where
    T: RFFloat,
{
    fn evaluate(&self, x: &Points1<T>) -> T;
    fn gradient(&self, x: &Points1<T>) -> Points1<T>;
    fn hessian(&self, x: &Points1<T>) -> Points2<T>;
}
dyn_clone::clone_trait_object!(<T> Constraint<T>);

/// Linear constraint: a^T x + b ≤ 0 (inequality) or = 0 (equality)
#[derive(Clone)]
pub struct LinearConstraint<T>
where
    T: RFFloat,
{
    pub a: Points1<T>,
    pub b: T,
    pub is_equality: bool,
}

impl<T> LinearConstraint<T>
where
    T: RFFloat,
{
    pub fn new(a: &Points1<T>, b: &T, is_equality: bool) -> Self {
        Self {
            a: a.to_owned(),
            b: b.clone(),
            is_equality,
        }
    }

    pub fn inequality(a: &Points1<T>, b: &T) -> Self {
        Self::new(a, b, false)
    }

    pub fn equality(a: &Points1<T>, b: &T) -> Self {
        Self::new(a, b, true)
    }
}

impl<T> Constraint<T> for LinearConstraint<T>
where
    T: RFFloat,
{
    fn evaluate(&self, x: &Points1<T>) -> T {
        self.a
            .iter()
            .zip(x.iter())
            .map(|(ai, xi)| ai.clone() * xi.clone())
            .sum::<T>()
            + &self.b
    }

    fn gradient(&self, _x: &Points1<T>) -> Points1<T> {
        self.a.clone()
    }

    fn hessian(&self, x: &Points1<T>) -> Points2<T> {
        let n = x.len();
        Points2::zeros((n, n))
    }
}

impl<T> fmt::Debug for LinearConstraint<T>
where
    T: RFFloat,
{
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
pub struct QuadraticConstraint<T>
where
    T: RFFloat,
{
    pub q: Points2<T>,
    pub a: Points1<T>,
    pub b: T,
    pub is_equality: bool,
}

impl<T> QuadraticConstraint<T>
where
    T: RFFloat,
{
    pub fn new(q: &Points2<T>, a: &Points1<T>, b: &T, is_equality: bool) -> Self {
        Self {
            q: q.clone(),
            a: a.clone(),
            b: b.clone(),
            is_equality,
        }
    }
}

impl<T> Constraint<T> for QuadraticConstraint<T>
where
    T: RFFloat,
{
    fn evaluate(&self, x: &Points1<T>) -> T {
        let n = x.len();
        let mut result = self.b.clone();

        // Linear term: a^T x
        for i in 0..n {
            result += self.a[i].clone() * x[i].clone();
        }

        // Quadratic term: x^T Q x
        for i in 0..n {
            for j in 0..n {
                result += x[i].clone() * self.q[[i, j]].clone() * x[j].clone();
            }
        }

        result
    }

    fn gradient(&self, x: &Points1<T>) -> Points1<T> {
        let n = x.len();
        let mut grad = self.a.clone();

        // Add 2 * Q * x (assuming Q is symmetric)
        for i in 0..n {
            for j in 0..n {
                grad[i] += T::from_f64(2.0) * self.q[[i, j]].clone() * x[j].clone();
            }
        }

        grad
    }

    fn hessian(&self, _x: &Points1<T>) -> Points2<T> {
        // Hessian is 2*Q for quadratic constraint
        let n = self.q.nrows();
        let mut hess = Points2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                hess[[i, j]] = T::from_f64(2.0) * self.q[[i, j]].clone();
            }
        }

        hess
    }
}

impl<T> fmt::Debug for QuadraticConstraint<T>
where
    T: RFFloat,
{
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
pub fn create_box_constraints<T>(
    lower: &Points1<T>,
    upper: &Points1<T>,
) -> Vec<Box<dyn Constraint<T>>>
where
    T: RFFloat + 'static,
{
    let mut constraints = Vec::new();

    for (i, (l, u)) in lower.iter().zip(upper.iter()).enumerate() {
        // x_i ≥ l becomes -x_i + l ≤ 0
        if l.is_finite() {
            let mut a = Points1::<T>::zeros(lower.len());
            a[i] = T::from_f64(-1.0);
            constraints
                .push(Box::new(LinearConstraint::<T>::inequality(&a, l)) as Box<dyn Constraint<T>>);
        }

        // x_i ≤ u becomes x_i - u ≤ 0
        if u.is_finite() {
            let mut a = Points1::<T>::zeros(lower.len());
            a[i] = T::from_f64(1.0);
            let neg_u = -u.clone();
            constraints
                .push(Box::new(LinearConstraint::<T>::inequality(&a, &neg_u))
                    as Box<dyn Constraint<T>>);
        }
    }

    constraints
}

// // Usage examples:
//
// use ndarray::prelude::*;
// use rfkit::prelude::*;
//
// // Example 1: Simple function (same as before)
// fn simple_objective(x: Points1<MyFloat>) -> MyFloat {
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
//     let objective_with_params = move |x: Points1<MyFloat>| -> MyFloat {
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
//     weights: Points1<f64>,
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
//     fn call(&self, x: Points1<MyFloat>) -> MyFloat {
//         // Use all the struct fields in your calculation
//         let base_error = self.calculate_base_error(x);
//         let regularization = self.apply_regularization(&x);
//         base_error + regularization
//     }
// }
//
// impl ComplexObjective {
//     fn calculate_base_error(&self, x: Points1<MyFloat>) -> MyFloat {
//         // Implementation using self.weights, self.reference_measurements, etc.
//         MyFloat::new(0.0)
//     }
//
//     fn apply_regularization(&self, x: &Points1<MyFloat>) -> MyFloat {
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
pub trait Minimizer<T>
where
    T: RFFloat,
{
    type Options;
    type Result;

    /// Run the optimization for specified iterations
    fn minimize(&mut self, opt: &Self::Options) -> Result<Self::Result, MinimizerError>;
}

// Helper function for outer product
fn outer<T>(a: &Points1<T>, b: &Points1<T>) -> Points2<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    let n = a.len();
    let m = b.len();
    let mut result = Points2::<T>::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = &a[i] * &b[j];
        }
    }

    result
}

/// A vertex of the simplex
#[derive(Debug)]
pub(crate) struct Vector<T>
where
    T: RFFloat,
{
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Vector<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    fn dot_product(a: &Points1<T>, b: &Points1<T>) -> T {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn vector_norm(v: &Points1<T>) -> T {
        v.iter().map(|x| x * x).sum::<T>().sqrt()
    }

    fn vector_add(a: &Points1<T>, b: &Points1<T>) -> Points1<T> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    fn vector_subtract(a: &Points1<T>, b: &Points1<T>) -> Points1<T> {
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }

    fn scalar_vector_multiply(scalar: &T, vector: &Points1<T>) -> Points1<T> {
        vector.iter().map(|x| scalar * x).collect()
    }
}

/// A vertex of the simplex
#[derive(Debug)]
pub(crate) struct Vertex<T>
where
    T: RFFloat,
{
    pub(crate) point: Points1<T>,
    pub(crate) value: T,
}

impl<T> Vertex<T>
where
    T: RFFloat,
{
    pub(crate) fn new<F>(point: &Points1<T>, f: &F) -> Result<Self, MinimizerError>
    where
        F: Fn(&Points1<T>) -> T,
    {
        let value = f(point);
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(Vertex {
            point: point.to_owned(),
            value,
        })
    }

    pub(crate) fn new_boxed(
        point: &Points1<T>,
        f: Box<dyn ObjFn<T>>,
    ) -> Result<Self, MinimizerError> {
        let value = f.call(point);
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(Vertex {
            point: point.to_owned(),
            value,
        })
    }
}

/// Line search result for internal use
#[derive(Debug)]
struct LineSearchResult<T>
where
    T: RFFloat,
{
    alpha: T,
    f_new: T,
    evaluations: usize,
    converged: bool,
}

/// Strong Wolfe conditions parameters
#[derive(Debug, Clone)]
pub struct WolfeParams<T> {
    pub c1: T, // Armijo condition parameter (typically 1e-4)
    pub c2: T, // Curvature condition parameter (typically 0.9 for CG, 0.1 for quasi-Newton)
    pub max_step: T,
    pub min_step: T,
}

impl<T> WolfeParams<T>
where
    T: RFFloat,
{
    /// Parameters optimized for BFGS method
    /// - Strong curvature condition for good Hessian approximation
    /// - Standard Armijo parameter
    pub fn for_bfgs() -> Self {
        WolfeParams {
            c1: T::from_f64(1e-4), // Standard Armijo parameter
            c2: T::from_f64(0.9),  // Strong curvature condition for BFGS
            max_step: T::from_f64(10.0),
            min_step: T::from_f64(1e-12),
        }
    }

    /// Parameters optimized for L-BFGS method
    /// - More relaxed curvature condition for efficiency
    /// - Suitable for large-scale problems
    pub fn for_lbfgs() -> Self {
        WolfeParams {
            c1: T::from_f64(1e-4),
            c2: T::from_f64(0.1), // More relaxed for L-BFGS efficiency
            max_step: T::from_f64(10.0),
            min_step: T::from_f64(1e-12),
        }
    }

    /// Parameters optimized for DFP method
    /// - Intermediate curvature condition
    /// - Balance between BFGS and L-BFGS
    pub fn for_dfp() -> Self {
        WolfeParams {
            c1: T::from_f64(1e-4),
            c2: T::from_f64(0.4), // Intermediate value for DFP
            max_step: T::from_f64(10.0),
            min_step: T::from_f64(1e-12),
        }
    }

    /// Parameters optimized for SR1 method
    /// - Very relaxed curvature condition
    /// - Often only uses Armijo condition
    pub fn for_sr1() -> Self {
        WolfeParams {
            c1: T::from_f64(1e-4),
            c2: T::from_f64(0.1), // Very relaxed for SR1
            max_step: T::from_f64(10.0),
            min_step: T::from_f64(1e-12),
        }
    }

    /// Conservative parameters for difficult problems
    /// - Stricter conditions for robustness
    pub fn conservative() -> Self {
        WolfeParams {
            c1: T::from_f64(1e-6),      // Stricter Armijo condition
            c2: T::from_f64(0.95),      // Very strict curvature condition
            max_step: T::from_f64(1.0), // Smaller maximum step
            min_step: T::from_f64(1e-15),
        }
    }

    /// Aggressive parameters for well-behaved problems
    /// - More relaxed conditions for speed
    pub fn aggressive() -> Self {
        WolfeParams {
            c1: T::from_f64(1e-2),        // More relaxed Armijo condition
            c2: T::from_f64(0.1),         // Relaxed curvature condition
            max_step: T::from_f64(100.0), // Larger maximum step
            min_step: T::from_f64(1e-10),
        }
    }

    /// Validate that the Wolfe parameters are mathematically valid
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.c1 <= T::zero() || self.c1 >= T::one() {
            return Err("c1 must be in (0, 1)");
        }

        if self.c2 <= self.c1 || self.c2 >= T::one() {
            return Err("c2 must be in (c1, 1)");
        }

        if self.max_step <= T::zero() {
            return Err("max_step must be positive");
        }

        if self.min_step <= T::zero() || self.min_step >= self.max_step {
            return Err("min_step must be positive and less than max_step");
        }

        Ok(())
    }
}

impl<T> Default for WolfeParams<T>
where
    T: RFFloat,
{
    fn default() -> Self {
        Self {
            c1: T::from_f64(1e-4),
            // c2: 0.9, // Higher value for conjugate gradient
            c2: T::from_f64(0.1), // Lower value for quasi-Newton (vs 0.9 for CG)
            max_step: T::from_f64(1e6),
            min_step: T::from_f64(1e-12),
        }
    }
}
