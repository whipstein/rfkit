#![allow(dead_code)]
use crate::util::{ApproxCompare, ApproxEq};
use core::f64;
use ndarray::Array1;
use num_complex::{Complex, ComplexFloat};
use num_traits::{ConstOne, ConstZero, Float, FloatConst, Num, One, Zero};
use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};
use twofloat::TwoFloat;

pub mod complex;

pub use self::complex::{ComplexNumber, ComplexNumberBuilder, ComplexNumberType};

mod sealed {
    use super::*;
    use serde::Serialize;
    use std::ops::RemAssign;

    pub trait Sealed:
        Copy
        + Clone
        + Default
        + Display
        + Debug
        + Sized
        + Serialize
        + ConstOne
        + ConstZero
        + One
        + Zero
        + Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Mul<Output = Self>
        + MulAssign
        + Div<Output = Self>
        + DivAssign
        + PartialEq
        + RemAssign
    {
    }

    impl Sealed for f64 {}
    impl Sealed for TwoFloat {}
    impl<T: Sealed + Num + PartialOrd> Sealed for Complex<T> {}
}

pub trait Scalar: sealed::Sealed {
    fn from_f64(x: f64) -> Self;
    fn from_usize(n: usize) -> Self;
}

impl Scalar for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }

    fn from_usize(n: usize) -> f64 {
        n as f64
    }
}
impl Scalar for TwoFloat {
    fn from_f64(x: f64) -> Self {
        TwoFloat::from_f64(x)
    }

    fn from_usize(n: usize) -> TwoFloat {
        TwoFloat::from_f64(n as f64)
    }
}
impl<T> Scalar for Complex<T>
where
    T: Scalar + Num + PartialOrd,
{
    fn from_f64(val: f64) -> Self {
        Self::new(T::from_f64(val), T::ZERO)
    }

    fn from_usize(val: usize) -> Self {
        Self::new(T::from_f64(val as f64), T::ZERO)
    }
}

pub trait RealScalar:
    Scalar
    + Float
    + FloatConst
    + From<f64>
    + PartialEq
    + PartialOrd
    + ApproxEq
    + ApproxCompare
    + ToComplex<COutput = Complex<Self>>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign<f64>
    + SubAssign<f64>
    + MulAssign<f64>
    + DivAssign<f64>
    + PartialEq<f64>
    + PartialOrd<f64>
    + std::iter::Sum
    + std::iter::Product
{
}

impl RealScalar for f64 {}
impl RealScalar for TwoFloat {}

pub trait ComplexScalar: Scalar + Norm + ComplexFloat + From<Self::Real> + ApproxEq
where
    <Self as ComplexFloat>::Real: RealScalar,
{
    fn new(re: Self::Real, im: Self::Real) -> Self;
}

impl<T> ComplexScalar for Complex<T>
where
    T: RealScalar + ApproxEq<Compare = T>,
{
    fn new(re: T, im: T) -> Self {
        Complex::<T>::new(re, im)
    }
}

pub trait ToReal {
    type ROutput;

    fn to_real(self) -> Self::ROutput;
}

impl<T: RealScalar> ToReal for T {
    type ROutput = T;

    fn to_real(self) -> T {
        self
    }
}
impl<T: RealScalar> ToReal for Array1<T> {
    type ROutput = Array1<T>;

    fn to_real(self) -> Array1<T> {
        self.map(|&x| x.to_real())
    }
}

pub trait ToComplex {
    type COutput;

    fn to_complex(self) -> Self::COutput;
}

impl<T> ToComplex for T
where
    T: RealScalar,
{
    type COutput = Complex<T>;

    fn to_complex(self) -> Complex<T> {
        Complex::new(self, T::ZERO)
    }
}
impl<T> ToComplex for Array1<T>
where
    T: RealScalar + ToComplex<COutput = Complex<T>>,
{
    type COutput = Array1<Complex<T>>;

    fn to_complex(self) -> Array1<Complex<T>> {
        self.map(|&x| x.to_complex())
    }
}

pub trait Norm: Scalar + ComplexFloat
where
    <Self as ComplexFloat>::Real: RealScalar,
{
    fn norm(&self) -> <Self as ComplexFloat>::Real;
    fn norm_sqr(&self) -> <Self as ComplexFloat>::Real;
}

impl Norm for f64 {
    fn norm(&self) -> f64 {
        self.abs()
    }

    fn norm_sqr(&self) -> f64 {
        self.norm().powi(2)
    }
}
impl Norm for TwoFloat {
    fn norm(&self) -> TwoFloat {
        self.abs()
    }

    fn norm_sqr(&self) -> TwoFloat {
        self.abs().powi(2)
    }
}
impl<T: RealScalar> Norm for Complex<T> {
    fn norm(&self) -> T {
        self.abs()
    }

    fn norm_sqr(&self) -> T {
        self.norm() * self.norm()
    }
}

pub trait ScalarConst: Scalar {
    const EPSILON: Self;
    const INFINITY: Self;
    const MAX: Self;
    const MIN: Self;
    const MIN_POSITIVE: Self;
    const NAN: Self;
    const NEG_INFINITY: Self;
    const ZERO: Self;
    const ONE: Self;
}

impl ScalarConst for f64 {
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
    const MAX: Self = f64::MAX;
    const MIN: Self = f64::MIN;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const NAN: Self = f64::NAN;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}
impl ScalarConst for TwoFloat {
    const EPSILON: Self = TwoFloat::EPSILON;
    const INFINITY: Self = TwoFloat::INFINITY;
    const MAX: Self = TwoFloat::MAX;
    const MIN: Self = TwoFloat::MIN;
    const MIN_POSITIVE: Self = TwoFloat::MIN_POSITIVE;
    const NAN: Self = TwoFloat::NAN;
    const NEG_INFINITY: Self = TwoFloat::NEG_INFINITY;
    const ZERO: Self = twofloat::consts::ZERO;
    const ONE: Self = twofloat::consts::ONE;
}
impl ScalarConst for Complex<f64> {
    const EPSILON: Self = Complex {
        re: f64::EPSILON,
        im: f64::EPSILON,
    };
    const INFINITY: Self = Complex {
        re: f64::INFINITY,
        im: f64::INFINITY,
    };
    const MAX: Self = Complex {
        re: f64::MAX,
        im: f64::MAX,
    };
    const MIN: Self = Complex {
        re: f64::MIN,
        im: f64::MIN,
    };
    const MIN_POSITIVE: Self = Complex {
        re: f64::MIN_POSITIVE,
        im: f64::MIN_POSITIVE,
    };
    const NAN: Self = Complex {
        re: f64::NAN,
        im: f64::NAN,
    };
    const NEG_INFINITY: Self = Complex {
        re: f64::NEG_INFINITY,
        im: f64::NEG_INFINITY,
    };
    const ZERO: Self = Complex { re: 0.0, im: 0.0 };
    const ONE: Self = Complex { re: 1.0, im: 0.0 };
}
impl ScalarConst for Complex<TwoFloat> {
    const EPSILON: Self = Complex {
        re: TwoFloat::EPSILON,
        im: TwoFloat::EPSILON,
    };
    const INFINITY: Self = Complex {
        re: TwoFloat::INFINITY,
        im: TwoFloat::INFINITY,
    };
    const MAX: Self = Complex {
        re: TwoFloat::MAX,
        im: TwoFloat::MAX,
    };
    const MIN: Self = Complex {
        re: TwoFloat::MIN,
        im: TwoFloat::MIN,
    };
    const MIN_POSITIVE: Self = Complex {
        re: TwoFloat::MIN_POSITIVE,
        im: TwoFloat::MIN_POSITIVE,
    };
    const NAN: Self = Complex {
        re: TwoFloat::NAN,
        im: TwoFloat::NAN,
    };
    const NEG_INFINITY: Self = Complex {
        re: TwoFloat::NEG_INFINITY,
        im: TwoFloat::NEG_INFINITY,
    };
    const ZERO: Self = Complex {
        re: twofloat::consts::ZERO,
        im: twofloat::consts::ZERO,
    };
    const ONE: Self = Complex {
        re: twofloat::consts::ONE,
        im: twofloat::consts::ZERO,
    };
}

pub trait ScalarArray<T: sealed::Sealed> {}

impl ScalarArray<f64> for Array1<f64> {}
impl ScalarArray<TwoFloat> for Array1<TwoFloat> {}
impl<T> ScalarArray<Complex<T>> for Array1<Complex<T>> where T: Scalar + Num + PartialOrd {}

pub trait RealArray<T: RealScalar> {}

impl RealArray<f64> for Array1<f64> {}
impl RealArray<TwoFloat> for Array1<TwoFloat> {}

pub trait ComplexArray<T: ComplexScalar>
where
    <T as ComplexFloat>::Real: RealScalar,
{
}

impl ComplexArray<Complex<f64>> for Array1<Complex<f64>> {}
impl ComplexArray<Complex<TwoFloat>> for Array1<Complex<TwoFloat>> {}
