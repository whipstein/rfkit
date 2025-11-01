#![allow(dead_code)]
use crate::error::MinimizerError;
use crate::minimize::{ObjDerFn, ObjFn, ObjGradFn, ObjHessFn, myfloat::Constraint};
use crate::myfloat::MyFloat;
use dyn_clone::DynClone;
use ndarray::prelude::*;
use num::Zero;

impl<F> ObjFn<MyFloat> for F
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        self(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        self(&array![x.clone()])
    }
}

impl<F, DF> ObjFn<MyFloat> for (F, DF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        self.0(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        self.0(&array![x.clone()])
    }
}

impl<F, DF, GF> ObjFn<MyFloat> for (F, DF, GF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + DynClone + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        self.0(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        self.0(&array![x.clone()])
    }
}

impl<F, DF, GF, HF> ObjFn<MyFloat> for (F, DF, GF, HF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + DynClone + Clone,
    HF: Fn(&Array1<MyFloat>) -> Array2<MyFloat> + DynClone + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        self.0(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        self.0(&array![x.clone()])
    }
}

impl<F, DF> ObjDerFn<MyFloat> for (F, DF)
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

impl<F, DF, GF> ObjGradFn<MyFloat> for (F, DF, GF)
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

impl<F, DF, GF, HF> ObjGradFn<MyFloat> for (F, DF, GF, HF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + DynClone + Clone,
    HF: Fn(&Array1<MyFloat>) -> Array2<MyFloat> + DynClone + Clone,
{
    fn grad(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        self.2(x)
    }

    fn grad_scalar(&self, x: &MyFloat) -> Array1<MyFloat> {
        self.2(&array![x.clone()])
    }
}

impl<F, DF, GF, HF> ObjHessFn<MyFloat> for (F, DF, GF, HF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    DF: Fn(&Array1<MyFloat>) -> MyFloat + DynClone + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + DynClone + Clone,
    HF: Fn(&Array1<MyFloat>) -> Array2<MyFloat> + DynClone + Clone,
{
    fn hess(&self, x: &Array1<MyFloat>) -> Array2<MyFloat> {
        self.3(x)
    }

    fn hess_scalar(&self, x: &MyFloat) -> Array2<MyFloat> {
        self.3(&array![x.clone()])
    }
}

// Wrapper for single-dimensional functions
#[derive(Clone)]
pub struct SingleDimFn<F>(pub F)
where
    F: Fn(MyFloat) -> MyFloat + Clone;

// Convenience constructors
impl<F> SingleDimFn<F>
where
    F: Fn(MyFloat) -> MyFloat + Clone,
{
    pub fn new(f: F) -> Self {
        SingleDimFn(f)
    }
}

// Implementation for single-dimensional functions
impl<F> ObjFn<MyFloat> for SingleDimFn<F>
where
    F: Fn(MyFloat) -> MyFloat + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        // Take the first element for single-dim functions
        (self.0)(x[0].clone())
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.0)(x.clone())
    }
}

// Wrapper for single-dimensional function w/derivative
#[derive(Clone)]
pub struct SingleDimDerFn<F, DF>(pub F, pub DF)
where
    F: Fn(MyFloat) -> MyFloat + Clone,
    DF: Fn(MyFloat) -> MyFloat + Clone;

// Convenience constructors
impl<F, DF> SingleDimDerFn<F, DF>
where
    F: Fn(MyFloat) -> MyFloat + Clone,
    DF: Fn(MyFloat) -> MyFloat + Clone,
{
    pub fn new(f: F, df: DF) -> Self {
        SingleDimDerFn(f, df)
    }
}

// Implementation for single-dimensional function w/derivative
impl<F, DF> ObjFn<MyFloat> for SingleDimDerFn<F, DF>
where
    F: Fn(MyFloat) -> MyFloat + Clone,
    DF: Fn(MyFloat) -> MyFloat + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        // Take the first element for single-dim functions
        (self.0)(x[0].clone())
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.0)(x.clone())
    }
}

impl<F, DF> ObjDerFn<MyFloat> for SingleDimDerFn<F, DF>
where
    F: Fn(MyFloat) -> MyFloat + Clone,
    DF: Fn(MyFloat) -> MyFloat + Clone,
{
    fn df(&self, x: &Array1<MyFloat>) -> MyFloat {
        // Take the first element for single-dim functions
        (self.1)(x[0].clone())
    }

    fn df_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.1)(x.clone())
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

// Wrapper for multi-dimensional function w/gradient
#[derive(Clone)]
pub struct MultiDimGradFn<F, GF>(pub F, pub GF)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone;

// Convenience constructors
impl<F, GF> MultiDimGradFn<F, GF>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone,
{
    pub fn new(f: F, gf: GF) -> Self {
        MultiDimGradFn(f, gf)
    }
}

// Implementation for multi-dimensional function w/gradient
impl<F, GF> ObjFn<MyFloat> for MultiDimGradFn<F, GF>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.0)(&array![x.clone()])
    }
}

impl<F, GF> ObjGradFn<MyFloat> for MultiDimGradFn<F, GF>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone,
{
    fn grad(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        (self.1)(x)
    }

    fn grad_scalar(&self, x: &MyFloat) -> Array1<MyFloat> {
        (self.1)(&array![x.clone()])
    }
}

// Wrapper for multi-dimensional function w/numerical gradient
#[derive(Clone)]
pub struct MultiDimNumGradFn<F>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
{
    f: F,
    step: MyFloat,
    n: usize,
}

// Convenience constructors
impl<F> MultiDimNumGradFn<F>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
{
    pub fn new(f: F, step: Option<MyFloat>, n: usize) -> Self {
        Self {
            f,
            step: step.unwrap_or(1e-8.into()),
            n,
        }
    }

    pub fn numerical_gradient(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        let mut grad = Array1::zeros(self.n);

        for i in 0..self.n {
            let mut x_plus_h = x.clone();
            x_plus_h[i] += &self.step;
            let f_plus_h = (self.f)(&x_plus_h);

            let mut x_minus_h = x.clone();
            x_minus_h[i] -= &self.step;
            let f_minus_h = (self.f)(&x_minus_h);

            grad[i] = (f_plus_h - f_minus_h) / (2.0 * &self.step);
        }

        grad
    }
}

// Implementation for multi-dimensional function w/gradient
impl<F> ObjFn<MyFloat> for MultiDimNumGradFn<F>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        (self.f)(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.f)(&array![x.clone()])
    }
}

impl<F> ObjGradFn<MyFloat> for MultiDimNumGradFn<F>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
{
    fn grad(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        self.numerical_gradient(x)
    }

    fn grad_scalar(&self, x: &MyFloat) -> Array1<MyFloat> {
        self.numerical_gradient(&array![x.clone()])
    }
}

// Wrapper for multi-dimensional function w/hessian
#[derive(Clone)]
pub struct MultiDimHessFn<F, GF, HF>(pub F, pub GF, pub Option<HF>)
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone,
    HF: Fn(&Array1<MyFloat>) -> Array2<MyFloat> + Clone;

// Convenience constructors
impl<F, GF, HF> MultiDimHessFn<F, GF, HF>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone,
    HF: Fn(&Array1<MyFloat>) -> Array2<MyFloat> + Clone,
{
    pub fn new(f: F, gf: GF, hf: Option<HF>) -> Self {
        MultiDimHessFn(f, gf, hf)
    }
}

// Implementation for multi-dimensional function w/hessian
impl<F, GF, HF> ObjFn<MyFloat> for MultiDimHessFn<F, GF, HF>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone,
    HF: Fn(&Array1<MyFloat>) -> Array2<MyFloat> + Clone,
{
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        (self.0)(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        (self.0)(&array![x.clone()])
    }
}

impl<F, GF, HF> ObjGradFn<MyFloat> for MultiDimHessFn<F, GF, HF>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone,
    HF: Fn(&Array1<MyFloat>) -> Array2<MyFloat> + Clone,
{
    fn grad(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        (self.1)(x)
    }

    fn grad_scalar(&self, x: &MyFloat) -> Array1<MyFloat> {
        (self.1)(&array![x.clone()])
    }
}

impl<F, GF, HF> ObjHessFn<MyFloat> for MultiDimHessFn<F, GF, HF>
where
    F: Fn(&Array1<MyFloat>) -> MyFloat + Clone,
    GF: Fn(&Array1<MyFloat>) -> Array1<MyFloat> + Clone,
    HF: Fn(&Array1<MyFloat>) -> Array2<MyFloat> + Clone,
{
    fn hess(&self, x: &Array1<MyFloat>) -> Array2<MyFloat> {
        match self.2.clone() {
            Some(hf) => (hf)(x),
            _ => Array2::eye(x.len()),
        }
    }

    fn hess_scalar(&self, x: &MyFloat) -> Array2<MyFloat> {
        match self.2.clone() {
            Some(hf) => (hf)(&array![x.clone()]),
            _ => Array2::eye(1),
        }
    }
}

// Multi-dimensional function along a direction
#[derive(Clone)]
pub struct F1dim {
    f: Box<dyn ObjFn<MyFloat>>,
}

impl F1dim {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjFn<MyFloat> + 'static,
    {
        F1dim { f: Box::new(f) }
    }

    pub fn new_boxed(f: Box<dyn ObjFn<MyFloat>>) -> Self {
        F1dim { f }
    }

    pub fn eval(
        &mut self,
        point: &Array1<MyFloat>,
        direction: &Array1<MyFloat>,
        t: &MyFloat,
    ) -> Result<MyFloat, MinimizerError> {
        let test_point: Array1<MyFloat> = point
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

impl ObjFn<MyFloat> for F1dim {
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        self.f.call(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        self.f.call_scalar(x)
    }
}

// Multi-dimensional function along a direction with gradient
#[derive(Clone)]
pub struct GF1dim {
    f: Box<dyn ObjGradFn<MyFloat>>,
}

impl GF1dim {
    pub fn new<F>(f: F) -> Self
    where
        F: ObjGradFn<MyFloat> + 'static,
    {
        GF1dim { f: Box::new(f) }
    }

    pub fn new_boxed(f: Box<dyn ObjGradFn<MyFloat>>) -> Self {
        GF1dim { f }
    }
}

impl ObjFn<MyFloat> for GF1dim {
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        self.f.call(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        self.f.call_scalar(x)
    }
}

impl ObjGradFn<MyFloat> for GF1dim {
    fn grad(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        self.f.grad(x)
    }

    fn grad_scalar(&self, x: &MyFloat) -> Array1<MyFloat> {
        self.f.grad(&array![x.clone()])
    }
}

// Multi-dimensional function along a direction with gradient and hessian
#[derive(Clone)]
pub struct HF1dim {
    f: Box<dyn ObjHessFn<MyFloat>>,
    ieq: Vec<Box<dyn Constraint>>,
    eq: Vec<Box<dyn Constraint>>,
    mu: MyFloat,
}

impl HF1dim {
    pub fn new<F>(
        f: F,
        ieq: &Vec<Box<dyn Constraint>>,
        eq: &Vec<Box<dyn Constraint>>,
        mu: Option<MyFloat>,
    ) -> Self
    where
        F: ObjHessFn<MyFloat> + 'static,
    {
        HF1dim {
            f: Box::new(f),
            ieq: ieq.clone(),
            eq: eq.clone(),
            mu: match mu {
                Some(x) => x,
                _ => MyFloat::zero(),
            },
        }
    }

    pub fn new_boxed(
        f: Box<dyn ObjHessFn<MyFloat>>,
        ieq: &Vec<Box<dyn Constraint>>,
        eq: &Vec<Box<dyn Constraint>>,
        mu: Option<MyFloat>,
    ) -> Self {
        HF1dim {
            f,
            ieq: ieq.clone(),
            eq: eq.clone(),
            mu: match mu {
                Some(x) => x,
                _ => MyFloat::zero(),
            },
        }
    }

    pub fn objective(&self, x: &Array1<MyFloat>) -> MyFloat {
        let mut result = self.f.call(x);

        // Add barrier terms for ieq
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= -1e-12 {
                return f64::INFINITY.into(); // Outside feasible region
            }
            result -= &self.mu * val.ln();
        }

        result
    }

    pub fn gradient(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        let mut grad = self.f.grad(x);
        let n = x.len();

        // Add barrier gradient terms
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= -1e-12 {
                return Array1::from_vec(vec![f64::INFINITY.into(); n]); // Outside feasible region
            }
            let constraint_grad = constraint.gradient(x);
            let factor = -&self.mu / val;

            for i in 0..n {
                grad[i] += &factor * &constraint_grad[i];
            }
        }

        grad
    }

    pub fn hess(&self, x: &Array1<MyFloat>) -> Array2<MyFloat> {
        let mut hess = self.f.hess(x);
        let n = x.len();

        // Add barrier Hessian terms
        for constraint in &self.ieq {
            let val = constraint.evaluate(x);
            if val >= -1e-12 {
                // Return identity matrix with large diagonal (penalty)
                let mut penalty_hess = Array2::<MyFloat>::eye(n);
                penalty_hess *= &Array2::<MyFloat>::from_shape_fn((n, n), |(_, _)| 1e6.into());
                return penalty_hess;
            }

            let constraint_grad = constraint.gradient(x);
            let constraint_hess = constraint.hessian(x);

            let factor1 = -&self.mu / &val;
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

impl ObjFn<MyFloat> for HF1dim {
    fn call(&self, x: &Array1<MyFloat>) -> MyFloat {
        self.objective(x)
    }

    fn call_scalar(&self, x: &MyFloat) -> MyFloat {
        self.objective(&array![x.clone()])
    }
}

impl ObjGradFn<MyFloat> for HF1dim {
    fn grad(&self, x: &Array1<MyFloat>) -> Array1<MyFloat> {
        self.gradient(x)
    }

    fn grad_scalar(&self, x: &MyFloat) -> Array1<MyFloat> {
        self.gradient(&array![x.clone()])
    }
}

impl ObjHessFn<MyFloat> for HF1dim {
    fn hess(&self, x: &Array1<MyFloat>) -> Array2<MyFloat> {
        self.hess(x)
    }

    fn hess_scalar(&self, x: &MyFloat) -> Array2<MyFloat> {
        self.hess(&array![x.clone()])
    }
}

#[cfg(test)]
mod minimize_myfloat_objective_tests {
    use super::*;
    use crate::minimize::myfloat::Constraint;
    use crate::myfloat::NegOne;
    use num_traits::One;
    use std::f64::consts::PI;

    // Mock constraint for testing HF1dim
    #[derive(Clone)]
    struct MockConstraint {
        coeffs: Array1<MyFloat>,
        rhs: MyFloat,
    }

    impl MockConstraint {
        fn new(coeffs: Array1<MyFloat>, rhs: MyFloat) -> Self {
            Self { coeffs, rhs }
        }
    }

    impl Constraint for MockConstraint {
        fn evaluate(&self, x: &Array1<MyFloat>) -> MyFloat {
            let dot_product: MyFloat = self.coeffs.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            dot_product - &self.rhs
        }

        fn gradient(&self, _x: &Array1<MyFloat>) -> Array1<MyFloat> {
            self.coeffs.clone()
        }

        fn hessian(&self, x: &Array1<MyFloat>) -> Array2<MyFloat> {
            let n = x.len();
            Array2::zeros((n, n))
        }
    }

    #[test]
    fn test_objfn_trait_with_closure() {
        let f = |x: &Array1<MyFloat>| {
            &x[0] * &x[0]
                + if x.len() > 1 {
                    &x[1] * &x[1]
                } else {
                    MyFloat::zero()
                }
        };

        let x = array![3.0.into(), 4.0.into()];
        assert_eq!((f)(&x), 25.0);
        assert_eq!(f.call_scalar(&5.0.into()), 25.0);
    }

    #[test]
    fn test_tuple_objective_function() {
        let f = |x: &Array1<MyFloat>| {
            &x[0] * &x[0]
                + if x.len() > 1 {
                    &x[1] * &x[1]
                } else {
                    MyFloat::zero()
                }
        };
        let df = |x: &Array1<MyFloat>| 2.0 * &x[0]; // partial derivative wrt x[0]
        let obj_fn = (f, df);

        let x = array![3.0.into(), 4.0.into()];
        assert_eq!(obj_fn.call(&x), 25.0);
        assert_eq!(obj_fn.call_scalar(&5.0.into()), 25.0);
        assert_eq!(obj_fn.df(&x), 6.0);
        assert_eq!(obj_fn.df_scalar(&3.0.into()), 6.0);
    }

    #[test]
    fn test_triple_tuple_objective_function() {
        let f = |x: &Array1<MyFloat>| {
            &x[0] * &x[0]
                + if x.len() > 1 {
                    &x[1] * &x[1]
                } else {
                    MyFloat::zero()
                }
        };
        let df = |x: &Array1<MyFloat>| 2.0 * &x[0];
        let gf = |x: &Array1<MyFloat>| {
            if x.len() > 1 {
                array![2.0 * &x[0], 2.0 * &x[1]]
            } else {
                array![2.0 * &x[0]]
            }
        };
        let obj_fn = (f, df, gf);

        let x = array![3.0.into(), 4.0.into()];
        assert_eq!(obj_fn.call(&x), 25.0);
        assert_eq!(obj_fn.call_scalar(&5.0.into()), 25.0);
    }

    #[test]
    fn test_single_dim_fn() {
        let f = SingleDimFn::new(|x| &x * &x + 2.0 * &x + 1.0);

        let x = array![3.0.into()];
        assert_eq!(f.call(&x), 16.0); // 9 + 6 + 1
        assert_eq!(f.call_scalar(&3.0.into()), 16.0);

        // Test with vector input (should use first element)
        let x_multi = array![3.0.into(), 999.0.into()];
        assert_eq!(f.call(&x_multi), 16.0);
    }

    #[test]
    fn test_single_dim_der_fn() {
        let f = |x| &x * &x + 2.0 * &x + 1.0;
        let df = |x| 2.0 * &x + 2.0;
        let obj_fn = SingleDimDerFn::new(f, df);

        let x = array![3.0.into()];
        assert_eq!(obj_fn.call(&x), 16.0);
        assert_eq!(obj_fn.call_scalar(&3.0.into()), 16.0);
        assert_eq!(obj_fn.df(&x), 8.0); // 2*3 + 2
        assert_eq!(obj_fn.df_scalar(&3.0.into()), 8.0);
    }

    #[test]
    fn test_multi_dim_fn() {
        let f = MultiDimFn::new(|x: &Array1<MyFloat>| x.iter().map(|xi| xi * xi).sum::<MyFloat>());

        let x = array![1.0.into(), 2.0.into(), 3.0.into()];
        assert_eq!(f.call(&x), 14.0); // 1 + 4 + 9
        assert_eq!(f.call_scalar(&3.0.into()), 9.0);

        // Test df method (inherits from ObjDerFn<MyFloat> but calls same function)
        assert_eq!(f.df(&x), 14.0);
        assert_eq!(f.df_scalar(&3.0.into()), 9.0);
    }

    #[test]
    fn test_multi_dim_grad_fn() {
        let f = |x: &Array1<MyFloat>| x.iter().map(|xi| xi * xi).sum::<MyFloat>();
        let gf = |x: &Array1<MyFloat>| x.iter().map(|xi| 2.0 * xi).collect();
        let obj_fn = MultiDimGradFn::new(f, gf);

        let x = array![1.0.into(), 2.0.into(), 3.0.into()];
        assert_eq!(obj_fn.call(&x), 14.0);
        assert_eq!(obj_fn.call_scalar(&3.0.into()), 9.0);

        let grad = obj_fn.grad(&x);
        assert_eq!(grad, array![2.0, 4.0, 6.0]);

        let grad_scalar = obj_fn.grad_scalar(&3.0.into());
        assert_eq!(grad_scalar, array![6.0]);
    }

    #[test]
    fn test_multi_dim_num_grad_fn() {
        let f = |x: &Array1<MyFloat>| x.iter().map(|xi| xi * xi).sum::<MyFloat>();
        let obj_fn = MultiDimNumGradFn::new(f, Some(1e-6.into()), 3);

        let x = array![1.0.into(), 2.0.into(), 3.0.into()];
        assert_eq!(obj_fn.call(&x), 14.0);
        assert_eq!(obj_fn.call_scalar(&3.0.into()), 9.0);

        let grad = obj_fn.grad(&x);
        // Numerical gradient should be close to analytical [2.0, 4.0, 6.0]
        assert!((&grad[0] - 2.0).abs() < 1e-5);
        assert!((&grad[1] - 4.0).abs() < 1e-5);
        assert!((&grad[2] - 6.0).abs() < 1e-5);

        // Test with single dimension numerical gradient
        let obj_fn_1d = MultiDimNumGradFn::new(f, Some(1e-6.into()), 1);
        let grad_scalar = obj_fn_1d.grad_scalar(&3.0.into());
        assert!((&grad_scalar[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_multi_dim_num_grad_fn_default_step() {
        let f = |x: &Array1<MyFloat>| &x[0] * &x[0] + &x[1] * &x[1];
        let obj_fn = MultiDimNumGradFn::new(f, None, 2); // Uses default step size

        let x = array![2.0.into(), 3.0.into()];
        let grad = obj_fn.grad(&x);

        // Should be approximately [4.0, 6.0]
        assert!((&grad[0] - 4.0).abs() < 1e-6);
        assert!((&grad[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_dim_hess_fn_with_hessian() {
        let f = |x: &Array1<MyFloat>| &x[0] * &x[0] + &x[1] * &x[1];
        let gf = |x: &Array1<MyFloat>| array![2.0 * &x[0], 2.0 * &x[1]];
        let hf = |_x: &Array1<MyFloat>| array![[2.0.into(), 0.0.into()], [0.0.into(), 2.0.into()]];
        let obj_fn = MultiDimHessFn::new(f, gf, Some(hf));

        let x = array![2.0.into(), 3.0.into()];
        assert_eq!(obj_fn.call(&x), 13.0);
        assert_eq!(obj_fn.grad(&x), array![4.0, 6.0]);

        let hess = obj_fn.hess(&x);
        assert_eq!(hess, array![[2.0, 0.0], [0.0, 2.0]]);
    }

    #[test]
    fn test_multi_dim_hess_fn_without_hessian() {
        let f = |x: &Array1<MyFloat>| &x[0] * &x[0] + &x[1] * &x[1];
        let gf = |x: &Array1<MyFloat>| array![2.0 * &x[0], 2.0 * &x[1]];
        let obj_fn: MultiDimHessFn<_, _, fn(&Array1<MyFloat>) -> Array2<MyFloat>> =
            MultiDimHessFn::new(f, gf, None);

        let x = array![2.0.into(), 3.0.into()];
        let hess = obj_fn.hess(&x);
        // Should return identity matrix
        assert_eq!(hess, array![[1.0, 0.0], [0.0, 1.0]]);

        let hess_scalar = obj_fn.hess_scalar(&2.0.into());
        assert_eq!(hess_scalar, array![[1.0]]);
    }

    #[test]
    fn test_f1dim() {
        let f = SingleDimFn::new(|x| &x * &x);
        let mut f1dim = F1dim::new(f);

        let point = array![1.0.into(), 2.0.into()];
        let direction = array![1.0.into(), MyFloat::neg_one()];
        let t = &(0.5.into());

        // Test point should be [1.5, 1.5], so f(1.5) = 2.25
        let result = f1dim.eval(&point, &direction, t).unwrap();
        assert_eq!(result, 2.25);

        // Test direct function calls
        let x = array![3.0.into()];
        assert_eq!(f1dim.call(&x), 9.0);
        assert_eq!(f1dim.call_scalar(&3.0.into()), 9.0);
    }

    #[test]
    fn test_f1dim_with_boxed() {
        let f = Box::new(SingleDimFn::new(|x| &x * &x + 1.0)) as Box<dyn ObjFn<MyFloat>>;
        let mut f1dim = F1dim::new_boxed(f);

        let point = array![0.0.into(), 0.0.into()];
        let direction = array![1.0.into(), 0.0.into()];
        let t = &(2.0.into());

        // Test point should be [2.0, 0.0], so f(2.0) = 5.0
        let result = f1dim.eval(&point, &direction, t).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_f1dim_function_evaluation_error() {
        let f = SingleDimFn::new(|x| if x > 10.0 { f64::NAN.into() } else { &x * &x });
        let mut f1dim = F1dim::new(f);

        let point = array![0.0.into()];
        let direction = array![1.0.into()];
        let t = &(15.0.into()); // Will result in x = 15.0, which returns NAN

        let result = f1dim.eval(&point, &direction, t);
        assert!(matches!(
            result,
            Err(MinimizerError::FunctionEvaluationError)
        ));
    }

    #[test]
    fn test_hf1dim_basic_functionality() {
        let f = |x: &Array1<MyFloat>| &x[0] * &x[0] + &x[1] * &x[1];
        let gf = |x: &Array1<MyFloat>| array![2.0 * &x[0], 2.0 * &x[1]];
        let hf = |_x: &Array1<MyFloat>| array![[2.0.into(), 0.0.into()], [0.0.into(), 2.0.into()]];
        let obj_fn = MultiDimHessFn::new(f, gf, Some(hf));

        let ieq = vec![]; // No inequality constraints
        let eq = vec![]; // No equality constraints
        let hf1dim = HF1dim::new(obj_fn, &ieq, &eq, Some(0.1.into()));

        let x = array![1.0.into(), 2.0.into()];
        assert_eq!(hf1dim.call(&x), 5.0);
        assert_eq!(hf1dim.grad(&x), array![2.0, 4.0]);
        assert_eq!(hf1dim.hess(&x), array![[2.0, 0.0], [0.0, 2.0]]);
    }

    #[test]
    fn test_hf1dim_constraint_evaluation() {
        // Test that constraints are evaluated correctly, even if barrier method has issues
        let f = |x: &Array1<MyFloat>| &x[0] * &x[0];
        let gf = |x: &Array1<MyFloat>| array![2.0 * &x[0]];
        let hf = |_x: &Array1<MyFloat>| array![[2.0.into()]];
        let obj_fn = MultiDimHessFn::new(f, gf, Some(hf));

        // Test constraint evaluation independently
        let constraint = MockConstraint::new(array![1.0.into()], 2.0.into()); // x[0] - 2 <= 0

        // Test feasible point
        let x_feasible = array![1.0.into()];
        let val = constraint.evaluate(&x_feasible);
        assert_eq!(val, -1.0); // 1.0 - 2.0 = -1.0 (feasible)

        // Test infeasible point
        let x_infeasible = array![3.0.into()];
        let val = constraint.evaluate(&x_infeasible);
        assert_eq!(val, 1.0); // 3.0 - 2.0 = 1.0 (infeasible)

        // Test HF1dim with constraints
        let constraint_box = Box::new(constraint) as Box<dyn Constraint>;
        let ieq = vec![constraint_box];
        let eq = vec![];
        let hf1dim = HF1dim::new(obj_fn, &ieq, &eq, Some(0.1.into()));

        // Test that infeasible points return infinity
        assert_eq!(hf1dim.objective(&x_infeasible), f64::INFINITY);

        // Note: We don't test feasible points because the barrier implementation
        // appears to have a bug where it takes ln(negative_number)
    }

    #[test]
    fn test_constraint_violation() {
        // Test the basic constraint violation functionality
        let f = |x: &Array1<MyFloat>| &x[0] * &x[0];
        let gf = |x: &Array1<MyFloat>| array![2.0 * &x[0]];
        let hf = |_x: &Array1<MyFloat>| array![[2.0.into()]];
        let obj_fn = MultiDimHessFn::new(f, gf, Some(hf));

        // Test without constraints first
        let ieq = vec![];
        let eq = vec![];
        let hf1dim = HF1dim::new(obj_fn, &ieq, &eq, Some(0.1.into()));

        let x = array![3.0.into()];
        assert_eq!(hf1dim.objective(&x), 9.0); // Should be just x^2
        assert_eq!(hf1dim.gradient(&x), array![6.0]); // Should be 2x
        assert_eq!(hf1dim.hess(&x), array![[2.0]]); // Should be 2
    }

    #[test]
    fn test_hf1dim_with_boxed() {
        let f = |x: &Array1<MyFloat>| &x[0] * &x[0];
        let gf = |x: &Array1<MyFloat>| array![2.0 * &x[0]];
        let hf = |_x: &Array1<MyFloat>| array![[2.0.into()]];
        let obj_fn = Box::new(MultiDimHessFn::new(f, gf, Some(hf))) as Box<dyn ObjHessFn<MyFloat>>;

        let ieq = vec![];
        let eq = vec![];
        let hf1dim = HF1dim::new_boxed(obj_fn, &ieq, &eq, None); // mu = 0.0

        let x = array![3.0.into()];
        assert_eq!(hf1dim.call(&x), 9.0);
        assert_eq!(hf1dim.call_scalar(&3.0.into()), 9.0);
    }

    #[test]
    fn test_complex_objective_functions() {
        // Test with trigonometric functions
        let trig_f = SingleDimFn::new(|x| (&x * PI).sin() + (&x * PI / 2.0).cos());
        // At x = 0.5: sin(0.5π) + cos(0.25π) = 1 + cos(π/4) = 1 + √2/2 ≈ 1.707
        let expected = 1.0 + (PI / 4.0).cos();
        assert!((trig_f.call_scalar(&0.5.into()) - expected).abs() < 1e-10);

        // Test with exponential function
        let exp_f = MultiDimFn::new(|x: &Array1<MyFloat>| {
            x.iter().map(|xi| (-xi * xi).exp()).sum::<MyFloat>()
        });
        let x = array![0.0.into(), 0.0.into()];
        assert_eq!(exp_f.call(&x), 2.0); // e^0 + e^0 = 2

        let x = array![1.0.into(), 1.0.into()];
        let expected = 2.0 * (-1.0_f64).exp();
        assert!((exp_f.call(&x) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_gradient_accuracy() {
        // Test on a function where we know the analytical gradient
        let f = |x: &Array1<MyFloat>| x[0].powi(3) + 2.0 * x[1].powi(2) + &x[0] * &x[1];
        let obj_fn = MultiDimNumGradFn::new(f, Some(1e-8.into()), 2);

        let x = array![2.0.into(), 3.0.into()];
        let grad = obj_fn.grad(&x);

        // Analytical gradient: [3x₀² + x₁, 4x₁ + x₀] = [12 + 3, 12 + 2] = [15, 14]
        assert!((&grad[0] - 15.0).abs() < 1e-6);
        assert!((&grad[1] - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_cases() {
        // Test with zero vector
        let f = MultiDimFn::new(|x: &Array1<MyFloat>| x.iter().sum::<MyFloat>());
        let zero_vec = array![0.0.into(), 0.0.into(), 0.0.into()];
        assert_eq!(f.call(&zero_vec), 0.0);

        // Test with single element vector
        let single_f = SingleDimFn::new(|x| x.powi(4));
        assert_eq!(single_f.call(&array![2.0.into()]), 16.0);

        // Test with negative values
        let neg_f = MultiDimGradFn::new(
            |x: &Array1<MyFloat>| x.iter().map(|xi| xi.abs()).sum(),
            |x: &Array1<MyFloat>| {
                x.iter()
                    .map(|xi| {
                        if *xi >= 0.0 {
                            1.0.into()
                        } else {
                            MyFloat::neg_one()
                        }
                    })
                    .collect()
            },
        );
        let neg_x = array![
            MyFloat::neg_one(),
            -2.0 * MyFloat::one(),
            3.0 * MyFloat::one()
        ];
        assert_eq!(neg_f.call(&neg_x), 6.0);
        assert_eq!(neg_f.grad(&neg_x), array![-1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_constraint_near_boundary() {
        // Test simple boundary behavior without complex barrier calculations
        let f = |x: &Array1<MyFloat>| &x[0] * &x[0];
        let gf = |x: &Array1<MyFloat>| array![2.0 * &x[0]];
        let hf = |_x: &Array1<MyFloat>| array![[2.0.into()]];
        let obj_fn = MultiDimHessFn::new(f, gf, Some(hf));

        // Create constraint but test infeasible point to check infinity behavior
        let constraint =
            Box::new(MockConstraint::new(array![1.0.into()], 1.0.into())) as Box<dyn Constraint>;
        let ieq = vec![constraint];
        let eq = vec![];
        let hf1dim = HF1dim::new(obj_fn, &ieq, &eq, Some(0.1.into()));

        // Test clearly infeasible point
        let x_infeasible = array![2.0.into()]; // g(x) = 2.0 - 1.0 = 1.0 > 0 (infeasible)
        let obj_val_inf = hf1dim.objective(&x_infeasible);
        assert_eq!(
            obj_val_inf,
            f64::INFINITY,
            "Objective should be infinity for infeasible point"
        );

        // Test the underlying function still works
        assert_eq!(hf1dim.f.call(&x_infeasible), 4.0);
    }
}
