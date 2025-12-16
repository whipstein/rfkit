#![allow(dead_code)]
use crate::{error::MinimizerError, num::RFFloat};
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
pub mod nelder_mead_bounded;
pub mod powell;
pub mod quasi_newton;
pub mod simplex;

pub use self::bracket::{Bracket, BracketOptions, BracketResult};
pub use self::brent::{Brent, BrentResult};
pub use self::cma_es::{CmaEs, CmaEsResult};
pub use self::conjugate_gradient::{ConjGrad, ConjGradMethod, ConjGradResult};
pub use self::dbrent::{DBrent, DBrentResult};
pub use self::golden::{Golden, GoldenResult};
pub use self::interior_point::{
    InteriorPoint, InteriorPointMethod, InteriorPointParams, InteriorPointResult,
};
pub use self::nelder_mead::{NelderMead, NelderMeadOptions, NelderMeadResult};
pub use self::nelder_mead_bounded::{
    NelderMeadBounded, NelderMeadBoundedOptions, NelderMeadBoundedResult,
};
pub use self::powell::{Powell, PowellResult};
pub use self::quasi_newton::{QuasiNewton, QuasiNewtonMethod, QuasiNewtonResult};
pub use self::simplex::{Simplex, SimplexResult};

// Define a trait for the objective function
pub trait ObjFn<T>: DynClone {
    fn call(&self, x: ArrayView1<T>) -> T;
    fn call_scalar(&self, x: &T) -> T;
}
dyn_clone::clone_trait_object!(<T> ObjFn<T>);

// Define a trait for the derivative function
pub trait ObjDerFn<T>: ObjFn<T> + DynClone {
    fn df(&self, x: ArrayView1<T>) -> T;
    fn df_scalar(&self, x: &T) -> T;
}
dyn_clone::clone_trait_object!(<T> ObjDerFn<T>);

// Define a trait for the gradient function
pub trait ObjGradFn<T>: ObjFn<T> + DynClone {
    fn grad(&self, x: ArrayView1<T>) -> Array1<T>;
    fn grad_scalar(&self, x: &T) -> Array1<T>;
}
dyn_clone::clone_trait_object!(<T> ObjGradFn<T>);

// Define a trait for the hessian function
pub trait ObjHessFn<T>: ObjGradFn<T> + DynClone {
    fn hess(&self, x: ArrayView1<T>) -> Array2<T>;
    fn hess_scalar(&self, x: &T) -> Array2<T>;
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
    fn call(&self, x: ArrayView1<T>) -> T {
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
    fn call(&self, x: ArrayView1<T>) -> T {
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
    fn df(&self, x: ArrayView1<T>) -> T {
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
    F: Fn(ArrayView1<T>) -> T + Clone,
    T: RFFloat;

// Convenience constructors
impl<T, F> MultiDimFn<T, F>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    T: RFFloat,
{
    pub fn new(f: F) -> Self {
        MultiDimFn(f, std::marker::PhantomData)
    }
}

// Implementation for multi-dimensional functions
impl<T, F> ObjFn<T> for MultiDimFn<T, F>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    T: RFFloat,
{
    fn call(&self, x: ArrayView1<T>) -> T {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.0)(array![x.clone()].view())
    }
}

// Implementation for multi-dimensional functions
impl<T, F> ObjDerFn<T> for MultiDimFn<T, F>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    T: RFFloat,
{
    fn df(&self, x: ArrayView1<T>) -> T {
        (self.0)(x)
    }

    fn df_scalar(&self, x: &T) -> T {
        (self.0)(array![x.clone()].view())
    }
}

// Wrapper for multi-dimensional function w/gradient
#[derive(Clone)]
pub struct MultiDimGradFn<T, F, GF>(pub F, pub GF, std::marker::PhantomData<T>)
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    T: RFFloat;

// Convenience constructors
impl<T, F, GF> MultiDimGradFn<T, F, GF>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    T: RFFloat,
{
    pub fn new(f: F, gf: GF) -> Self {
        MultiDimGradFn(f, gf, std::marker::PhantomData)
    }
}

// Implementation for multi-dimensional function w/gradient
impl<T, F, GF> ObjFn<T> for MultiDimGradFn<T, F, GF>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    T: RFFloat,
{
    fn call(&self, x: ArrayView1<T>) -> T {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.0)(array![x.clone()].view())
    }
}

impl<T, F, GF> ObjGradFn<T> for MultiDimGradFn<T, F, GF>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    T: RFFloat,
{
    fn grad(&self, x: ArrayView1<T>) -> Array1<T> {
        (self.1)(x)
    }

    fn grad_scalar(&self, x: &T) -> Array1<T> {
        (self.1)(array![x.clone()].view())
    }
}

// Wrapper for multi-dimensional function w/numerical gradient
#[derive(Clone)]
pub struct MultiDimNumGradFn<T, F>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    T: RFFloat,
{
    f: F,
    step: T,
    n: usize,
}

// Convenience constructors
impl<T, F> MultiDimNumGradFn<T, F>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    T: RFFloat,
{
    pub fn new(f: F, step: Option<T>, n: usize) -> Self {
        Self {
            f,
            step: step.unwrap_or(T::from_f64(1e-8)),
            n,
        }
    }

    pub fn numerical_gradient(&self, x: ArrayView1<T>) -> Array1<T> {
        let mut grad = Array1::zeros(self.n);

        for i in 0..self.n {
            let mut x_plus_h = x.to_owned();
            x_plus_h[i] += self.step.clone();
            let f_plus_h = (self.f)(x_plus_h.view());

            let mut x_minus_h = x.to_owned();
            x_minus_h[i] -= self.step.clone();
            let f_minus_h = (self.f)(x_minus_h.view());

            grad[i] = (f_plus_h - f_minus_h) / (self.step.clone() * 2.0);
        }

        grad
    }
}

// Implementation for multi-dimensional function w/gradient
impl<T, F> ObjFn<T> for MultiDimNumGradFn<T, F>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    T: RFFloat,
{
    fn call(&self, x: ArrayView1<T>) -> T {
        (self.f)(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.f)(array![x.clone()].view())
    }
}

impl<T, F> ObjGradFn<T> for MultiDimNumGradFn<T, F>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    T: RFFloat,
{
    fn grad(&self, x: ArrayView1<T>) -> Array1<T> {
        self.numerical_gradient(x)
    }

    fn grad_scalar(&self, x: &T) -> Array1<T> {
        self.numerical_gradient(array![x.clone()].view())
    }
}

// Wrapper for multi-dimensional function w/hessian
#[derive(Clone)]
pub struct MultiDimHessFn<T, F, GF, HF>(pub F, pub GF, pub Option<HF>, std::marker::PhantomData<T>)
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    HF: Fn(ArrayView1<T>) -> Array2<T> + Clone,
    T: RFFloat;

// Convenience constructors
impl<T, F, GF, HF> MultiDimHessFn<T, F, GF, HF>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    HF: Fn(ArrayView1<T>) -> Array2<T> + Clone,
    T: RFFloat,
{
    pub fn new(f: F, gf: GF, hf: Option<HF>) -> Self {
        MultiDimHessFn(f, gf, hf, std::marker::PhantomData)
    }
}

// Implementation for multi-dimensional function w/hessian
impl<T, F, GF, HF> ObjFn<T> for MultiDimHessFn<T, F, GF, HF>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    HF: Fn(ArrayView1<T>) -> Array2<T> + Clone,
    T: RFFloat,
{
    fn call(&self, x: ArrayView1<T>) -> T {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        (self.0)(array![x.clone()].view())
    }
}

impl<T, F, GF, HF> ObjGradFn<T> for MultiDimHessFn<T, F, GF, HF>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    HF: Fn(ArrayView1<T>) -> Array2<T> + Clone,
    T: RFFloat,
{
    fn grad(&self, x: ArrayView1<T>) -> Array1<T> {
        (self.1)(x)
    }

    fn grad_scalar(&self, x: &T) -> Array1<T> {
        (self.1)(array![x.clone()].view())
    }
}

impl<T, F, GF, HF> ObjHessFn<T> for MultiDimHessFn<T, F, GF, HF>
where
    F: Fn(ArrayView1<T>) -> T + Clone,
    GF: Fn(ArrayView1<T>) -> Array1<T> + Clone,
    HF: Fn(ArrayView1<T>) -> Array2<T> + Clone,
    T: RFFloat,
{
    fn hess(&self, x: ArrayView1<T>) -> Array2<T> {
        match self.2.clone() {
            Some(hf) => (hf)(x),
            _ => Array2::eye(x.len()),
        }
    }

    fn hess_scalar(&self, x: &T) -> Array2<T> {
        match self.2.clone() {
            Some(hf) => (hf)(array![x.clone()].view()),
            _ => Array2::eye(1),
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
        point: ArrayView1<T>,
        direction: ArrayView1<T>,
        t: &T,
    ) -> Result<T, MinimizerError> {
        let test_point: Array1<T> = point
            .iter()
            .zip(direction.iter())
            .map(|(p, d)| p.clone() + t.clone() * d.clone())
            .collect();
        let value = self.f.call(test_point.view());
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
    fn call(&self, x: ArrayView1<T>) -> T {
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
    fn call(&self, x: ArrayView1<T>) -> T {
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
    fn grad(&self, x: ArrayView1<T>) -> Array1<T> {
        self.f.grad(x)
    }

    fn grad_scalar(&self, x: &T) -> Array1<T> {
        self.f.grad(array![x.clone()].view())
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

    pub fn objective(&self, x: ArrayView1<T>) -> T {
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

    pub fn gradient(&self, x: ArrayView1<T>) -> Array1<T> {
        let mut grad = self.f.grad(x);
        let n = x.len();

        // Add barrier gradient terms
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= T::from_f64(-1e-12) {
                return Array1::from_vec(vec![T::from_f64(f64::INFINITY); n]); // Outside feasible region
            }
            let constraint_grad = constraint.gradient(x);
            let factor = (-self.mu.clone()) / val;

            for i in 0..n {
                grad[i] += factor.clone() * constraint_grad[i].clone();
            }
        }

        grad
    }

    pub fn hess(&self, x: ArrayView1<T>) -> Array2<T> {
        let mut hess = self.f.hess(x);
        let n = x.len();

        // Add barrier Hessian terms
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= T::from_f64(-1e-12) {
                // Return identity matrix with large diagonal (penalty)
                let mut penalty_hess = Array2::<T>::eye(n);
                penalty_hess *= &Array2::<T>::from_shape_fn((n, n), |(_, _)| T::from_f64(1e6));
                return penalty_hess;
            }

            let constraint_grad = constraint.gradient(x);
            let constraint_hess = constraint.hessian(x);

            let factor1 = (-self.mu.clone()) / val.clone();
            let factor2 = self.mu.clone() / (val.clone() * val.clone());

            // Add second derivative terms
            for i in 0..n {
                for j in 0..n {
                    hess[[i, j]] += factor1.clone() * constraint_hess[[i, j]].clone()
                        + factor2.clone() * constraint_grad[i].clone() * constraint_grad[j].clone();
                }
            }
        }

        hess
    }
}

impl<T> ObjFn<T> for HF1dim<T>
where
    T: RFFloat,
{
    fn call(&self, x: ArrayView1<T>) -> T {
        self.objective(x)
    }

    fn call_scalar(&self, x: &T) -> T {
        self.objective(array![x.clone()].view())
    }
}

impl<T> ObjGradFn<T> for HF1dim<T>
where
    T: RFFloat,
{
    fn grad(&self, x: ArrayView1<T>) -> Array1<T> {
        self.gradient(x)
    }

    fn grad_scalar(&self, x: &T) -> Array1<T> {
        self.gradient(array![x.clone()].view())
    }
}

impl<T> ObjHessFn<T> for HF1dim<T>
where
    T: RFFloat,
{
    fn hess(&self, x: ArrayView1<T>) -> Array2<T> {
        self.hess(x)
    }

    fn hess_scalar(&self, x: &T) -> Array2<T> {
        self.hess(array![x.clone()].view())
    }
}

/// Constraint definition
pub trait Constraint<T>: DynClone
where
    T: RFFloat,
{
    fn evaluate(&self, x: ArrayView1<T>) -> T;
    fn gradient(&self, x: ArrayView1<T>) -> Array1<T>;
    fn hessian(&self, x: ArrayView1<T>) -> Array2<T>;
}
dyn_clone::clone_trait_object!(<T> Constraint<T>);

/// Linear constraint: a^T x + b ≤ 0 (inequality) or = 0 (equality)
#[derive(Clone)]
pub struct LinearConstraint<T>
where
    T: RFFloat,
{
    pub a: Array1<T>,
    pub b: T,
    pub is_equality: bool,
}

impl<T> LinearConstraint<T>
where
    T: RFFloat,
{
    pub fn new(a: ArrayView1<T>, b: &T, is_equality: bool) -> Self {
        Self {
            a: a.to_owned(),
            b: b.clone(),
            is_equality,
        }
    }

    pub fn inequality(a: ArrayView1<T>, b: &T) -> Self {
        Self::new(a, b, false)
    }

    pub fn equality(a: ArrayView1<T>, b: &T) -> Self {
        Self::new(a, b, true)
    }
}

impl<T> Constraint<T> for LinearConstraint<T>
where
    T: RFFloat,
{
    fn evaluate(&self, x: ArrayView1<T>) -> T {
        self.a
            .iter()
            .zip(x.iter())
            .map(|(ai, xi)| ai.clone() * xi.clone())
            .sum::<T>()
            + &self.b
    }

    fn gradient(&self, _x: ArrayView1<T>) -> Array1<T> {
        self.a.clone()
    }

    fn hessian(&self, x: ArrayView1<T>) -> Array2<T> {
        let n = x.len();
        Array2::zeros((n, n))
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
    pub q: Array2<T>,
    pub a: Array1<T>,
    pub b: T,
    pub is_equality: bool,
}

impl<T> QuadraticConstraint<T>
where
    T: RFFloat,
{
    pub fn new(q: &Array2<T>, a: &Array1<T>, b: &T, is_equality: bool) -> Self {
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
    fn evaluate(&self, x: ArrayView1<T>) -> T {
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

    fn gradient(&self, x: ArrayView1<T>) -> Array1<T> {
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

    fn hessian(&self, _x: ArrayView1<T>) -> Array2<T> {
        // Hessian is 2*Q for quadratic constraint
        let n = self.q.nrows();
        let mut hess = Array2::zeros((n, n));

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
    lower: ArrayView1<T>,
    upper: ArrayView1<T>,
) -> Vec<Box<dyn Constraint<T>>>
where
    T: RFFloat + 'static,
{
    let mut constraints = Vec::new();

    for (i, (l, u)) in lower.iter().zip(upper.iter()).enumerate() {
        // x_i ≥ l becomes -x_i + l ≤ 0
        if l.is_finite() {
            let mut a = Array1::<T>::zeros(lower.len());
            a[i] = T::from_f64(-1.0);
            constraints
                .push(Box::new(LinearConstraint::<T>::inequality(a.view(), l))
                    as Box<dyn Constraint<T>>);
        }

        // x_i ≤ u becomes x_i - u ≤ 0
        if u.is_finite() {
            let mut a = Array1::<T>::zeros(lower.len());
            a[i] = T::from_f64(1.0);
            let neg_u = -u.clone();
            constraints.push(
                Box::new(LinearConstraint::<T>::inequality(a.view(), &neg_u))
                    as Box<dyn Constraint<T>>,
            );
        }
    }

    constraints
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
pub trait Minimizer<T, U> {
    /// Run the optimization for specified iterations
    fn minimize(
        &mut self,
        opt: Box<dyn MinimizerOptions<T, U>>,
    ) -> Result<Box<dyn MinimizerResult<T, U>>, MinimizerError>;
}

pub trait MinimizerOptions<T, U> {
    fn initial_point(&self) -> T;
    fn scale(&self) -> T;
    fn max_iterations(&self) -> usize;
    fn tolerance(&self) -> U;
    fn verbosity(&self) -> usize;
}

pub trait MinimizerResult<T, U> {
    fn xmin(&self) -> T;
    fn fmin(&self) -> U;
    fn tolerance(&self) -> U;
    fn fn_evals(&self) -> usize;
    fn iters(&self) -> usize;
    fn converged(&self) -> bool;
    fn history(&self) -> T;
}

// Helper function for outer product
fn outer<T>(a: &Array1<T>, b: &Array1<T>) -> Array2<T>
where
    T: RFFloat,
{
    let n = a.len();
    let m = b.len();
    let mut result = Array2::<T>::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i].clone() * b[j].clone();
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
{
    fn dot_product(a: &Array1<T>, b: &Array1<T>) -> T {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.clone() * y.clone())
            .sum()
    }

    fn vector_norm(v: &Array1<T>) -> T {
        v.iter().map(|x| x.clone() * x.clone()).sum::<T>().sqrt()
    }

    fn vector_add(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.clone() + y.clone())
            .collect()
    }

    fn vector_subtract(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.clone() - y.clone())
            .collect()
    }

    fn scalar_vector_multiply(scalar: T, vector: &Array1<T>) -> Array1<T> {
        vector.iter().map(|x| scalar.clone() * x.clone()).collect()
    }
}

/// A vertex of the simplex
#[derive(Debug)]
pub(crate) struct Vertex<T>
where
    T: RFFloat,
{
    pub(crate) point: Array1<T>,
    pub(crate) value: T,
}

impl<T> Vertex<T>
where
    T: RFFloat,
{
    pub(crate) fn new<F>(point: Array1<T>, f: &F) -> Result<Self, MinimizerError>
    where
        F: Fn(ArrayView1<T>) -> T,
    {
        let value = f(point.view());
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(Vertex { point, value })
    }

    pub(crate) fn new_boxed(
        point: Array1<T>,
        f: Box<dyn ObjFn<T>>,
    ) -> Result<Self, MinimizerError> {
        let value = f.call(point.view());
        if !value.is_finite() {
            return Err(MinimizerError::FunctionEvaluationError);
        }
        Ok(Vertex { point, value })
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

// Helper function for dot product
fn dot_1d_1d<T>(a: ArrayView1<T>, b: ArrayView1<T>) -> T
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    let mut result = T::zero();
    for i in 0..a.len() {
        result += &a[i] * &b[i];
    }

    result
}

// Helper function for dot product
fn dot_1d_2d<T>(a: ArrayView1<T>, b: ArrayView2<T>) -> Array1<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    let mut result = Array1::<T>::zeros(b.ncols());
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            result[j] += &a[i] * &b[[i, j]];
        }
    }

    result
}

// Helper function for dot product (matrix * vector)
fn dot_2d_1d<T>(a: ArrayView2<T>, b: ArrayView1<T>) -> Array1<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    let mut result = Array1::<T>::zeros(a.nrows());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            result[i] += &a[[i, j]] * &b[j];
        }
    }

    result
}

// Helper function for dot product
fn dot_2d_2d<T>(a: ArrayView2<T>, b: ArrayView2<T>) -> Array2<T>
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    let mut result = Array2::<T>::zeros((a.nrows(), b.ncols()));
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
fn norm_1d<T>(a: ArrayView1<T>) -> T
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += &a[i] * &a[i];
    }
    sum.sqrt()
}

// Helper function for Frobenius norm of 2D array
// This is the matrix equivalent of the Euclidean norm
fn norm_2d<T>(a: ArrayView2<T>) -> T
where
    T: RFFloat,
    for<'a, 'b> &'a T: std::ops::Mul<&'b T, Output = T>,
{
    let mut sum = T::zero();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            sum += &a[[i, j]] * &a[[i, j]];
        }
    }
    sum.sqrt()
}

// Helper function for L1 norm (Manhattan norm) of 1D array
fn norm_1d_l1<T>(a: ArrayView1<T>) -> T
where
    T: RFFloat,
{
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a[i].abs();
    }
    sum
}

// Helper function for L1 norm of 2D array
fn norm_2d_l1<T>(a: ArrayView2<T>) -> T
where
    T: RFFloat,
{
    let mut sum = T::zero();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            sum += a[[i, j]].abs();
        }
    }
    sum
}

// Helper function for L-infinity norm (maximum norm) of 1D array
fn norm_1d_linf<T>(a: ArrayView1<T>) -> T
where
    T: RFFloat,
{
    let mut max = T::zero();
    for i in 0..a.len() {
        let abs_val = a[i].abs();
        if abs_val > max {
            max = abs_val;
        }
    }
    max
}

// Helper function for L-infinity norm of 2D array
fn norm_2d_linf<T>(a: ArrayView2<T>) -> T
where
    T: RFFloat,
{
    let mut max = T::zero();
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

#[cfg(test)]
mod minimize_tests {
    use super::*;
    use crate::myfloat::MyFloat;
    use float_cmp::{F64Margin, approx_eq};

    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-10,
        ulps: 10,
    };

    mod dot_1d_1d_tests {
        use crate::myfloat::MyFloat;

        use super::*;

        #[test]
        fn test_dot_1d_1d_basic() {
            let a = array![1.0.into(), 2.0.into(), 3.0.into()];
            let b = array![4.0.into(), 5.0.into(), 6.0.into()];
            let result: MyFloat = dot_1d_1d(a.view(), b.view());
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assert!(approx_eq!(f64, result.to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_zeros() {
            let a = array![0.0.into(), 0.0.into(), 0.0.into()];
            let b = array![1.0.into(), 2.0.into(), 3.0.into()];
            let result: MyFloat = dot_1d_1d(a.view(), b.view());
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_negative_values() {
            let a = array![(-1.0).into(), 2.0.into(), (-3.0).into()];
            let b = array![4.0.into(), (-5.0).into(), 6.0.into()];
            let result: MyFloat = dot_1d_1d(a.view(), b.view());
            // (-1)*4 + 2*(-5) + (-3)*6 = -4 - 10 - 18 = -32
            assert!(approx_eq!(f64, result.to_f64(), -32.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_orthogonal() {
            // Orthogonal vectors should have dot product of 0
            let a = array![1.0.into(), 0.0.into()];
            let b = array![0.0.into(), 1.0.into()];
            let result: MyFloat = dot_1d_1d(a.view(), b.view());
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_single_element() {
            let a = array![5.0.into()];
            let b = array![3.0.into()];
            let result: MyFloat = dot_1d_1d(a.view(), b.view());
            assert!(approx_eq!(f64, result.to_f64(), 15.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_commutative() {
            let a = array![1.5.into(), 2.5.into(), 3.5.into()];
            let b = array![4.5.into(), 5.5.into(), 6.5.into()];
            let result1: MyFloat = dot_1d_1d(a.view(), b.view());
            let result2 = dot_1d_1d(b.view(), a.view());
            assert!(approx_eq!(f64, result1.to_f64(), result2.to_f64(), MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_fractional_values() {
            let a: Array1<MyFloat> = array![0.1.into(), 0.2.into(), 0.3.into()];
            let b = array![0.4.into(), 0.5.into(), 0.6.into()];
            let result = dot_1d_1d(a.view(), b.view());
            // 0.1*0.4 + 0.2*0.5 + 0.3*0.6 = 0.04 + 0.1 + 0.18 = 0.32
            assert!(approx_eq!(f64, result.to_f64(), 0.32, MARGIN));
        }

        #[test]
        fn test_dot_1d_1d_large_values() {
            let a = array![1000.0.into(), 2000.0.into()];
            let b = array![3000.0.into(), 4000.0.into()];
            let result: MyFloat = dot_1d_1d(a.view(), b.view());
            // 1000*3000 + 2000*4000 = 3,000,000 + 8,000,000 = 11,000,000
            assert!(approx_eq!(f64, result.to_f64(), 11_000_000.0, MARGIN));
        }
    }

    mod dot_1d_2d_tests {
        use super::*;

        #[test]
        fn test_dot_1d_2d_basic() {
            let a: Array1<MyFloat> = array![1.0.into(), 2.0.into(), 3.0.into()];
            let b = array![
                [1.0.into(), 2.0.into()],
                [3.0.into(), 4.0.into()],
                [5.0.into(), 6.0.into()]
            ];
            let result = dot_1d_2d(a.view(), b.view());

            // Result should be [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6]
            //                = [1 + 6 + 15, 2 + 8 + 18]
            //                = [22, 28]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 22.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 28.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_identity_like() {
            let a: Array1<MyFloat> = array![1.0.into(), 2.0.into()];
            let b = array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]];
            let result = dot_1d_2d(a.view(), b.view());

            // Should get the original vector back
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 1.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 2.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_zeros() {
            let a: Array1<MyFloat> = array![0.0.into(), 0.0.into(), 0.0.into()];
            let b = array![
                [1.0.into(), 2.0.into()],
                [3.0.into(), 4.0.into()],
                [5.0.into(), 6.0.into()]
            ];
            let result = dot_1d_2d(a.view(), b.view());

            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_negative_values() {
            let a: Array1<MyFloat> = array![(-1.0).into(), 2.0.into()];
            let b = array![[3.0.into(), (-4.0).into()], [5.0.into(), 6.0.into()]];
            let result = dot_1d_2d(a.view(), b.view());

            // Result: [-1*3 + 2*5, -1*(-4) + 2*6]
            //       = [-3 + 10, 4 + 12]
            //       = [7, 16]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 7.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 16.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_single_column() {
            let a: Array1<MyFloat> = array![1.0.into(), 2.0.into(), 3.0.into()];
            let b = array![[4.0.into()], [5.0.into()], [6.0.into()]];
            let result = dot_1d_2d(a.view(), b.view());

            // Result: [1*4 + 2*5 + 3*6] = [32]
            assert_eq!(result.len(), 1);
            assert!(approx_eq!(f64, result[0].to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_1d_2d_rectangular() {
            let a: Array1<MyFloat> = array![1.0.into(), 2.0.into()];
            let b = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ];
            let result = dot_1d_2d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ]; // 2x3 matrix
            let b = array![1.0.into(), 2.0.into(), 3.0.into()]; // 3-element vector
            let result = dot_2d_1d(a.view(), b.view());

            // Result: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3]
            //       = [1 + 4 + 9, 4 + 10 + 18]
            //       = [14, 32]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 14.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_square_matrix() {
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![5.0.into(), 6.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Result: [1*5 + 2*6, 3*5 + 4*6]
            //       = [5 + 12, 15 + 24]
            //       = [17, 39]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 17.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 39.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_identity_matrix() {
            let identity: Array2<MyFloat> =
                array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]];
            let b = array![3.0.into(), 7.0.into()];
            let result = dot_2d_1d(identity.view(), b.view());

            // Identity matrix should return the original vector
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 3.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_zero_vector() {
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![0.0.into(), 0.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Result should be zero vector
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_zero_matrix() {
            let a: Array2<MyFloat> = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let b = array![5.0.into(), 6.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Result should be zero vector
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_negative_values() {
            let a: Array2<MyFloat> =
                array![[(-1.0).into(), 2.0.into()], [3.0.into(), (-4.0).into()]];
            let b = array![5.0.into(), (-6.0).into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Result: [(-1)*5 + 2*(-6), 3*5 + (-4)*(-6)]
            //       = [-5 - 12, 15 + 24]
            //       = [-17, 39]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), -17.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 39.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_single_row() {
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into(), 3.0.into()]];
            let b = array![4.0.into(), 5.0.into(), 6.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Result: [1*4 + 2*5 + 3*6] = [4 + 10 + 18] = [32]
            assert_eq!(result.len(), 1);
            assert!(approx_eq!(f64, result[0].to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_single_column() {
            let a: Array2<MyFloat> = array![[2.0.into()], [3.0.into()], [4.0.into()]];
            let b = array![5.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Result: [2*5, 3*5, 4*5] = [10, 15, 20]
            assert_eq!(result.len(), 3);
            assert!(approx_eq!(f64, result[0].to_f64(), 10.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 15.0, MARGIN));
            assert!(approx_eq!(f64, result[2].to_f64(), 20.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_tall_matrix() {
            let a: Array2<MyFloat> = array![
                [1.0.into(), 2.0.into()],
                [3.0.into(), 4.0.into()],
                [5.0.into(), 6.0.into()],
                [7.0.into(), 8.0.into()]
            ]; // 4x2 matrix
            let b = array![2.0.into(), 3.0.into()]; // 2-element vector
            let result = dot_2d_1d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![
                [1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()],
                [5.0.into(), 6.0.into(), 7.0.into(), 8.0.into()]
            ]; // 2x4 matrix
            let b = array![1.0.into(), 1.0.into(), 1.0.into(), 1.0.into()]; // 4-element vector
            let result = dot_2d_1d(a.view(), b.view());

            // Result: [1+2+3+4, 5+6+7+8] = [10, 26]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 10.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 26.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_3x3_matrix() {
            let a: Array2<MyFloat> = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()],
                [7.0.into(), 8.0.into(), 9.0.into()]
            ];
            let b = array![2.0.into(), 1.0.into(), 3.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![[0.5.into(), 0.25.into()], [0.75.into(), 0.125.into()]];
            let b = array![4.0.into(), 8.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Result: [0.5*4 + 0.25*8, 0.75*4 + 0.125*8]
            //       = [2.0 + 2.0, 3.0 + 1.0]
            //       = [4.0, 4.0]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 4.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 4.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_unit_vector() {
            let a: Array2<MyFloat> = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ];
            let b = array![1.0.into(), 0.0.into(), 0.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Should extract the first column
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 1.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 4.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_ones_vector() {
            let a: Array2<MyFloat> = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ];
            let b = array![1.0.into(), 1.0.into(), 1.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Should sum each row
            // Result: [1+2+3, 4+5+6] = [6, 15]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 6.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 15.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_single_element() {
            let a: Array2<MyFloat> = array![[5.0.into()]];
            let b = array![3.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

            // Result: [5*3] = [15]
            assert_eq!(result.len(), 1);
            assert!(approx_eq!(f64, result[0].to_f64(), 15.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_diagonal_matrix() {
            let a: Array2<MyFloat> = array![
                [2.0.into(), 0.0.into(), 0.0.into()],
                [0.0.into(), 3.0.into(), 0.0.into()],
                [0.0.into(), 0.0.into(), 4.0.into()]
            ];
            let b = array![5.0.into(), 6.0.into(), 7.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let v = array![2.0.into(), 3.0.into()];
            let c = MyFloat::new(5.0);

            // Compute A*(c*v)
            let cv = v.mapv(|x| &x * &c);
            let result1 = dot_2d_1d(a.view(), cv.view());

            // Compute c*(A*v)
            let av = dot_2d_1d(a.view(), v.view());
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
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let v = array![2.0.into(), 3.0.into()];
            let w = array![5.0.into(), 7.0.into()];

            // Compute A*(v + w)
            let vw = array![&v[0] + &w[0], &v[1] + &w[1]];
            let result1 = dot_2d_1d(a.view(), vw.view());

            // Compute A*v + A*w
            let av = dot_2d_1d(a.view(), v.view());
            let aw = dot_2d_1d(a.view(), w.view());
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
            let a: Array2<MyFloat> = array![
                [1000.0.into(), 2000.0.into()],
                [3000.0.into(), 4000.0.into()]
            ];
            let b = array![5.0.into(), 6.0.into()];
            let result = dot_2d_1d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![[0.0.into(), (-1.0).into()], [1.0.into(), 0.0.into()]];
            let b = array![1.0.into(), 0.0.into()]; // Unit vector along x-axis
            let result = dot_2d_1d(a.view(), b.view());

            // Should rotate to y-axis: [0, 1]
            assert_eq!(result.len(), 2);
            assert!(approx_eq!(f64, result[0].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[1].to_f64(), 1.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_1d_consistency_with_2d_2d() {
            // Verify that dot_2d_1d(A, v) gives same result as
            // treating v as a column vector and using dot_2d_2d
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let v = array![5.0.into(), 6.0.into()];

            // Using dot_2d_1d
            let result1 = dot_2d_1d(a.view(), v.view());

            // Using dot_2d_2d with v as column vector
            let v_col = array![[5.0.into()], [6.0.into()]];
            let result2_matrix = dot_2d_2d(a.view(), v_col.view());

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
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![[5.0.into(), 6.0.into()], [7.0.into(), 8.0.into()]];
            let result = dot_2d_2d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let identity = array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]];
            let result = dot_2d_2d(a.view(), identity.view());

            // Should get the original matrix back
            assert_eq!(result.shape(), &[2, 2]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 1.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 2.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 3.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), 4.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_zeros() {
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let zeros = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let result = dot_2d_2d(a.view(), zeros.view());

            // Result should be all zeros
            assert_eq!(result.shape(), &[2, 2]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[[0, 1]].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 0.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 1]].to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_rectangular_compatible() {
            let a: Array2<MyFloat> = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ]; // 2x3
            let b = array![
                [7.0.into(), 8.0.into()],
                [9.0.into(), 10.0.into()],
                [11.0.into(), 12.0.into()]
            ]; // 3x2
            let result = dot_2d_2d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![[5.0.into()]];
            let b = array![[3.0.into()]];
            let result = dot_2d_2d(a.view(), b.view());

            assert_eq!(result.shape(), &[1, 1]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 15.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_negative_values() {
            let a: Array2<MyFloat> =
                array![[(-1.0).into(), 2.0.into()], [3.0.into(), (-4.0).into()]];
            let b = array![[5.0.into(), (-6.0).into()], [(-7.0).into(), 8.0.into()]];
            let result = dot_2d_2d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![[5.0.into()], [6.0.into()]]; // Column vector
            let result = dot_2d_2d(a.view(), b.view());

            // Result should be a 2x1 matrix
            // [1*5 + 2*6]   [17]
            // [3*5 + 4*6] = [39]
            assert_eq!(result.shape(), &[2, 1]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 17.0, MARGIN));
            assert!(approx_eq!(f64, result[[1, 0]].to_f64(), 39.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_row_vector_multiplication() {
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into(), 3.0.into()]]; // Row vector (1x3)
            let b = array![[4.0.into()], [5.0.into()], [6.0.into()]]; // Column vector (3x1)
            let result = dot_2d_2d(a.view(), b.view());

            // Result should be a 1x1 matrix (scalar)
            // [1*4 + 2*5 + 3*6] = [32]
            assert_eq!(result.shape(), &[1, 1]);
            assert!(approx_eq!(f64, result[[0, 0]].to_f64(), 32.0, MARGIN));
        }

        #[test]
        fn test_dot_2d_2d_3x3_matrix() {
            let a: Array2<MyFloat> = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()],
                [7.0.into(), 8.0.into(), 9.0.into()]
            ];
            let b = array![
                [9.0.into(), 8.0.into(), 7.0.into()],
                [6.0.into(), 5.0.into(), 4.0.into()],
                [3.0.into(), 2.0.into(), 1.0.into()]
            ];
            let result = dot_2d_2d(a.view(), b.view());

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
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let b = array![[5.0.into(), 6.0.into()], [7.0.into(), 8.0.into()]];
            let c = array![[9.0.into(), 10.0.into()], [11.0.into(), 12.0.into()]];

            let ab = dot_2d_2d(a.view(), b.view());
            let abc_left = dot_2d_2d(ab.view(), c.view());

            let bc = dot_2d_2d(b.view(), c.view());
            let abc_right = dot_2d_2d(a.view(), bc.view());

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
            let a: Array2<MyFloat> = array![[0.5.into(), 0.25.into()], [0.75.into(), 0.125.into()]];
            let b = array![[2.0.into(), 4.0.into()], [8.0.into(), 16.0.into()]];
            let result = dot_2d_2d(a.view(), b.view());

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
            let result: MyFloat = norm_1d(a.view());
            // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_zeros() {
            let a = array![0.0.into(), 0.0.into(), 0.0.into()];
            let result: MyFloat = norm_1d(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_single_element() {
            let a = array![5.0.into()];
            let result: MyFloat = norm_1d(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_negative_values() {
            let a = array![(-3.0).into(), (-4.0).into()];
            let result: MyFloat = norm_1d(a.view());
            // sqrt((-3)^2 + (-4)^2) = sqrt(9 + 16) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_mixed_signs() {
            let a = array![(-3.0).into(), 4.0.into()];
            let result: MyFloat = norm_1d(a.view());
            // sqrt((-3)^2 + 4^2) = sqrt(9 + 16) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_unit_vector() {
            let a = array![1.0.into(), 0.0.into(), 0.0.into()];
            let result: MyFloat = norm_1d(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 1.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_all_ones() {
            let a = array![1.0.into(), 1.0.into(), 1.0.into()];
            let result: MyFloat = norm_1d(a.view());
            // sqrt(1 + 1 + 1) = sqrt(3) ≈ 1.732050808
            assert!(approx_eq!(f64, result.to_f64(), 3.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_1d_pythagorean_triple() {
            let a = array![5.0.into(), 12.0.into()];
            let result: MyFloat = norm_1d(a.view());
            // sqrt(25 + 144) = sqrt(169) = 13
            assert!(approx_eq!(f64, result.to_f64(), 13.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_fractional() {
            let a = array![0.6.into(), 0.8.into()];
            let result: MyFloat = norm_1d(a.view());
            // sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0
            assert!(approx_eq!(f64, result.to_f64(), 1.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_large_values() {
            let a = array![1000.0.into(), 0.0.into()];
            let result: MyFloat = norm_1d(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 1000.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_homogeneity() {
            // Test that ||c*v|| = |c| * ||v||
            let a: Array1<MyFloat> = array![3.0.into(), 4.0.into()];
            let norm_a = norm_1d(a.view());

            let scalar = MyFloat::new(2.0);
            let scaled = a.mapv(|x| &x * &scalar);
            let norm_scaled: MyFloat = norm_1d(scaled.view());

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
            let result: MyFloat = norm_2d(a.view());
            // sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477225575
            assert!(approx_eq!(f64, result.to_f64(), 30.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_zeros() {
            let a = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let result: MyFloat = norm_2d(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_single_element() {
            let a = array![[5.0.into()]];
            let result: MyFloat = norm_2d(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_identity() {
            let a = array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]];
            let result: MyFloat = norm_2d(a.view());
            // sqrt(1 + 0 + 0 + 1) = sqrt(2) ≈ 1.414213562
            assert!(approx_eq!(f64, result.to_f64(), 2.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_negative_values() {
            let a = array![
                [(-1.0).into(), (-2.0).into()],
                [(-3.0).into(), (-4.0).into()]
            ];
            let result: MyFloat = norm_2d(a.view());
            // sqrt(1 + 4 + 9 + 16) = sqrt(30)
            assert!(approx_eq!(f64, result.to_f64(), 30.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_mixed_signs() {
            let a = array![[1.0.into(), (-2.0).into()], [(-3.0).into(), 4.0.into()]];
            let result: MyFloat = norm_2d(a.view());
            // sqrt(1 + 4 + 9 + 16) = sqrt(30)
            assert!(approx_eq!(f64, result.to_f64(), 30.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_rectangular() {
            let a = array![
                [1.0.into(), 2.0.into(), 3.0.into()],
                [4.0.into(), 5.0.into(), 6.0.into()]
            ];
            let result: MyFloat = norm_2d(a.view());
            // sqrt(1 + 4 + 9 + 16 + 25 + 36) = sqrt(91) ≈ 9.539392014
            assert!(approx_eq!(f64, result.to_f64(), 91.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_column_vector() {
            let a = array![[3.0.into()], [4.0.into()]];
            let result: MyFloat = norm_2d(a.view());
            // sqrt(9 + 16) = sqrt(25) = 5
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_row_vector() {
            let a = array![[3.0.into(), 4.0.into()]];
            let result: MyFloat = norm_2d(a.view());
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
            let result: MyFloat = norm_2d(a.view());
            // sqrt(1+4+9+16+25+36+49+64+81) = sqrt(285) ≈ 16.881943016
            assert!(approx_eq!(f64, result.to_f64(), 285.0_f64.sqrt(), MARGIN));
        }

        #[test]
        fn test_norm_2d_homogeneity() {
            // Test that ||c*A|| = |c| * ||A||
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];
            let norm_a = norm_2d(a.view());

            let scalar = MyFloat::new(3.0);
            let scaled = a.mapv(|x| &x * &scalar);
            let norm_scaled: MyFloat = norm_2d(scaled.view());

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
            let result: MyFloat = norm_1d_l1(a.view());
            // |3| + |4| = 7
            assert!(approx_eq!(f64, result.to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_l1_negative() {
            let a = array![(-3.0).into(), (-4.0).into()];
            let result: MyFloat = norm_1d_l1(a.view());
            // |-3| + |-4| = 3 + 4 = 7
            assert!(approx_eq!(f64, result.to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_l1_mixed() {
            let a = array![(-3.0).into(), 4.0.into(), (-2.0).into()];
            let result: MyFloat = norm_1d_l1(a.view());
            // |-3| + |4| + |-2| = 3 + 4 + 2 = 9
            assert!(approx_eq!(f64, result.to_f64(), 9.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_l1_zeros() {
            let a = array![0.0.into(), 0.0.into()];
            let result: MyFloat = norm_1d_l1(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }
    }

    mod norm_2d_l1_tests {
        use super::*;

        #[test]
        fn test_norm_2d_l1_basic() {
            let a = array![[1.0.into(), (-2.0).into()], [3.0.into(), (-4.0).into()]];
            let result: MyFloat = norm_2d_l1(a.view());
            // |1| + |-2| + |3| + |-4| = 1 + 2 + 3 + 4 = 10
            assert!(approx_eq!(f64, result.to_f64(), 10.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_l1_zeros() {
            let a = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let result: MyFloat = norm_2d_l1(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }
    }

    mod norm_1d_linf_tests {
        use super::*;

        #[test]
        fn test_norm_1d_linf_basic() {
            let a = array![3.0.into(), 4.0.into(), 2.0.into()];
            let result: MyFloat = norm_1d_linf(a.view());
            // max(|3|, |4|, |2|) = 4
            assert!(approx_eq!(f64, result.to_f64(), 4.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_linf_negative() {
            let a = array![3.0.into(), (-7.0).into(), 2.0.into()];
            let result: MyFloat = norm_1d_linf(a.view());
            // max(|3|, |-7|, |2|) = 7
            assert!(approx_eq!(f64, result.to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_linf_zeros() {
            let a = array![0.0.into(), 0.0.into()];
            let result: MyFloat = norm_1d_linf(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }

        #[test]
        fn test_norm_1d_linf_single() {
            let a = array![5.0.into()];
            let result: MyFloat = norm_1d_linf(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 5.0, MARGIN));
        }
    }

    mod norm_2d_linf_tests {
        use super::*;

        #[test]
        fn test_norm_2d_linf_basic() {
            let a = array![[1.0.into(), 2.0.into()], [(-7.0).into(), 4.0.into()]];
            let result: MyFloat = norm_2d_linf(a.view());
            // max(|1|, |2|, |-7|, |4|) = 7
            assert!(approx_eq!(f64, result.to_f64(), 7.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_linf_all_positive() {
            let a = array![[1.0.into(), 2.0.into()], [3.0.into(), 9.0.into()]];
            let result: MyFloat = norm_2d_linf(a.view());
            // max(1, 2, 3, 9) = 9
            assert!(approx_eq!(f64, result.to_f64(), 9.0, MARGIN));
        }

        #[test]
        fn test_norm_2d_linf_zeros() {
            let a = array![[0.0.into(), 0.0.into()], [0.0.into(), 0.0.into()]];
            let result: MyFloat = norm_2d_linf(a.view());
            assert!(approx_eq!(f64, result.to_f64(), 0.0, MARGIN));
        }
    }

    mod norm_cross_validation_tests {
        use super::*;

        #[test]
        fn test_norm_inequality_1d() {
            // For the same vector: ||x||_inf <= ||x||_2 <= ||x||_1
            let a: Array1<MyFloat> = array![1.0.into(), 2.0.into(), 3.0.into()];

            let l_inf = norm_1d_linf(a.view()).to_f64();
            let l_2 = norm_1d(a.view()).to_f64();
            let l_1 = norm_1d_l1(a.view()).to_f64();

            assert!(l_inf <= l_2);
            assert!(l_2 <= l_1);
        }

        #[test]
        fn test_norm_inequality_2d() {
            // Same inequality holds for matrices
            let a: Array2<MyFloat> = array![[1.0.into(), 2.0.into()], [3.0.into(), 4.0.into()]];

            let l_inf = norm_2d_linf(a.view()).to_f64();
            let frobenius = norm_2d(a.view()).to_f64();
            let l_1 = norm_2d_l1(a.view()).to_f64();

            assert!(l_inf <= frobenius);
            assert!(frobenius <= l_1);
        }

        #[test]
        fn test_triangle_inequality_1d() {
            // ||a + b|| <= ||a|| + ||b||
            let a: Array1<MyFloat> = array![1.0.into(), 2.0.into()];
            let b: Array1<MyFloat> = array![3.0.into(), 4.0.into()];
            let sum: Array1<MyFloat> = array![4.0.into(), 6.0.into()];

            let norm_sum = norm_1d(sum.view()).to_f64();
            let norm_a = norm_1d(a.view()).to_f64();
            let norm_b = norm_1d(b.view()).to_f64();

            assert!(norm_sum <= norm_a + norm_b + 1e-10); // Small epsilon for floating point
        }

        #[test]
        fn test_consistency_vector_matrix() {
            // A column vector should have the same norm whether treated as 1D or 2D
            let vec_1d: Array1<MyFloat> = array![3.0.into(), 4.0.into()];
            let vec_2d: Array2<MyFloat> = array![[3.0.into()], [4.0.into()]];

            let norm_1 = norm_1d(vec_1d.view()).to_f64();
            let norm_2 = norm_2d(vec_2d.view()).to_f64();

            assert!(approx_eq!(f64, norm_1, norm_2, MARGIN));
        }
    }
}
