#![allow(dead_code)]
use crate::{
    consts::MathConst,
    util::{ApproxCompare, ApproxEq},
};
use core::f64;
use ndarray::Array1;
use num_complex::{Complex, ComplexFloat};
use num_traits::{ConstOne, ConstZero, Float, FloatConst, Num, One, Zero};
use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};
use twofloat::TwoFloat;

// pub mod mycomplex;
// pub mod myfloat;
// pub mod myusize;

// pub use self::mycomplex::MyComplex;
// pub use self::myfloat::MyFloat;
// pub use self::myusize::MyUsize;

mod private {
    pub trait Sealed {}
}

mod sealed {
    use serde::Serialize;

    use super::*;
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

pub trait Scalar: sealed::Sealed {}

impl Scalar for f64 {}
impl Scalar for TwoFloat {}
impl<T> Scalar for Complex<T> where T: Scalar + Num + PartialOrd {}

pub trait RealScalar:
    Scalar + Float + FloatConst + MathConst + From<f64> + PartialOrd + ApproxEq + ApproxCompare
{
    fn from_usize(n: usize) -> Self;
}

impl RealScalar for f64 {
    fn from_usize(n: usize) -> f64 {
        n as f64
    }
}
impl RealScalar for TwoFloat {
    fn from_usize(n: usize) -> TwoFloat {
        TwoFloat::from_f64(n as f64)
    }
}

pub trait ComplexScalar:
    Scalar + Norm + ComplexFloat + MathConst + From<Self::Real> + ApproxEq
where
    <Self as ComplexFloat>::Real: RealScalar,
{
    fn new(re: Self::Real, im: Self::Real) -> Self;
    fn from_f64(val: f64) -> Self;
}

impl<T> ComplexScalar for Complex<T>
where
    T: RealScalar + ApproxEq<Compare = T>,
{
    fn new(re: T, im: T) -> Self {
        Complex::<T>::new(re, im)
    }

    fn from_f64(val: f64) -> Self {
        Self::from(<Self::Real as From<f64>>::from(val))
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

// pub trait Value<T: Scalar> {}

// impl<T: Scalar> Value<T> for T {}
// impl<T: Scalar> Value<T> for Array1<T> {}

// pub trait RealValue<T: RealScalar> {}

// impl RealValue<f64> for f64 {}
// impl RealValue<TwoFloat> for TwoFloat {}
// impl<T: RealScalar> RealValue<T> for Array1<T> {}

// pub trait ComplexValue<T: ComplexScalar>
// where
//     T::Real: RealScalar,
// {
// }

// impl<T: ComplexScalar> ComplexValue<T> for T {}
// impl<T: ComplexScalar> ComplexValue<T> for Array1<T> {}

// pub trait RFNum:
//     Sized
//     + Clone
//     + Copy
//     + Default
//     + fmt::Display
//     + fmt::Debug
//     + std::ops::Add<Output = Self>
//     + std::ops::Sub<Output = Self>
//     + std::ops::Mul<Output = Self>
//     + std::ops::Div<Output = Self>
//     + std::ops::Neg<Output = Self>
//     + std::ops::Add<Self::Real, Output = Self>
//     + std::ops::Sub<Self::Real, Output = Self>
//     + std::ops::Mul<Self::Real, Output = Self>
//     + std::ops::Div<Self::Real, Output = Self>
//     + std::ops::Add<f64, Output = Self>
//     + std::ops::Sub<f64, Output = Self>
//     + std::ops::Mul<f64, Output = Self>
//     + std::ops::Div<f64, Output = Self>
//     + std::ops::AddAssign
//     + std::ops::SubAssign
//     + std::ops::MulAssign
//     + std::ops::DivAssign
//     + std::ops::AddAssign<f64>
//     + std::ops::SubAssign<f64>
//     + std::ops::MulAssign<f64>
//     + std::ops::DivAssign<f64>
//     + for<'a> std::ops::Add<&'a Self, Output = Self>
//     + for<'a> std::ops::Sub<&'a Self, Output = Self>
//     + for<'a> std::ops::Mul<&'a Self, Output = Self>
//     + for<'a> std::ops::Div<&'a Self, Output = Self>
//     + for<'a> std::ops::Add<&'a Self::Real, Output = Self>
//     + for<'a> std::ops::Sub<&'a Self::Real, Output = Self>
//     + for<'a> std::ops::Mul<&'a Self::Real, Output = Self>
//     + for<'a> std::ops::Div<&'a Self::Real, Output = Self>
//     + for<'a> std::ops::AddAssign<&'a Self>
//     + for<'a> std::ops::SubAssign<&'a Self>
//     + for<'a> std::ops::MulAssign<&'a Self>
//     + for<'a> std::ops::DivAssign<&'a Self>
//     + num_traits::Zero
//     + num_traits::One
//     + std::iter::Sum
//     + for<'a> std::iter::Sum<&'a Self>
//     + std::convert::From<f64>
//     + std::convert::From<Self::Real>
//     + std::cmp::PartialEq<Self>
// {
//     /// The underlying real type (e.g., f64 for Complex<f64>, MyFloat for MyComplex)
//     type Real: Into<Self> + Clone;

//     // ========== Special values (constants) ==========
//     /// Returns the NaN value
//     fn nan() -> Self;

//     /// Returns positive infinity
//     fn infinity() -> Self;

//     // ========== Classification methods ==========
//     /// Returns true if this value is NaN
//     fn is_nan(self) -> bool;

//     /// Returns true if this value is positive infinity or negative infinity
//     fn is_infinite(self) -> bool;

//     /// Returns true if this number is neither infinite nor NaN
//     fn is_finite(self) -> bool;

//     /// Returns true if the number is neither zero, infinite, subnormal, or NaN
//     fn is_normal(self) -> bool;

//     // ========== Basic operations ==========
//     /// Get the absolute value
//     fn abs(self) -> Self;

//     /// Get the complex conjugate
//     fn conj(self) -> Self;

//     /// Calculate the magnitude (norm)
//     fn norm(self) -> Self::Real;

//     /// Calculate the square of the magnitude (norm squared)
//     fn norm_sqr(self) -> Self::Real;

//     /// Convert a Real value to Self (for complex types, creates a value with zero imaginary part)
//     fn from_real(real: Self::Real) -> Self;

//     // ========== Exponential and power functions ==========
//     /// Returns e^(self)
//     fn exp(self) -> Self;

//     /// Returns 2^(self)
//     fn exp2(self) -> Self;

//     /// Returns e^(self) - 1 in a way that is accurate even if the number is close to zero
//     fn exp_m1(self) -> Self;

//     /// Returns the natural logarithm
//     fn ln(self) -> Self;

//     /// Returns ln(1+n) more accurately than if the operations were performed separately
//     fn ln_1p(self) -> Self;

//     /// Returns the base 2 logarithm of the number
//     fn log2(self) -> Self;

//     /// Returns the base 10 logarithm of the number
//     fn log10(self) -> Self;

//     /// Raises self to a complex power
//     fn pow(self, exp: Self) -> Self;

//     /// Raises self to an integer power
//     fn powi(self, n: i32) -> Self;

//     /// Raises self to a floating point power
//     fn powf(self, y: Self::Real) -> Self;

//     /// Returns the cube root
//     fn cbrt(self) -> Self;

//     /// Returns the square root
//     fn sqrt(self) -> Self;

//     /// Returns the square
//     fn square(self) -> Self::Real;

//     // ========== Trigonometric functions ==========
//     /// Simultaneously computes the sine and cosine of the number
//     fn sin_cos(self) -> (Self, Self);

//     /// Computes the sine
//     fn sin(self) -> Self;

//     /// Computes the cosine
//     fn cos(self) -> Self;

//     /// Computes the tangent
//     fn tan(self) -> Self;

//     /// Computes the arcsine
//     fn asin(self) -> Self;

//     /// Computes the arccosine
//     fn acos(self) -> Self;

//     /// Computes the arctangent
//     fn atan(self) -> Self;

//     // ========== Hyperbolic functions ==========
//     /// Hyperbolic sine function
//     fn sinh(self) -> Self;

//     /// Hyperbolic cosine function
//     fn cosh(self) -> Self;

//     /// Hyperbolic tangent function
//     fn tanh(self) -> Self;

//     /// Inverse hyperbolic sine function
//     fn asinh(self) -> Self;

//     /// Inverse hyperbolic cosine function
//     fn acosh(self) -> Self;

//     /// Inverse hyperbolic tangent function
//     fn atanh(self) -> Self;
// }

// impl RFNum for f64 {
//     type Real = Self;

//     // ========== Special values (constants) ==========
//     fn nan() -> Self {
//         f64::NAN
//     }

//     fn infinity() -> Self {
//         f64::INFINITY
//     }

//     // ========== Classification methods ==========
//     fn is_nan(self) -> bool {
//         f64::is_nan(self)
//     }

//     fn is_infinite(self) -> bool {
//         f64::is_infinite(self)
//     }

//     fn is_finite(self) -> bool {
//         f64::is_finite(self)
//     }

//     fn is_normal(self) -> bool {
//         f64::is_normal(self)
//     }

//     // ========== Basic operations ==========
//     fn abs(self) -> Self {
//         f64::abs(self)
//     }

//     fn conj(self) -> Self {
//         self
//     }

//     fn norm(self) -> Self::Real {
//         f64::abs(self)
//     }

//     fn norm_sqr(self) -> Self::Real {
//         self * self
//     }

//     fn from_real(real: Self::Real) -> Self {
//         real
//     }

//     // ========== Exponential and power functions ==========
//     fn exp(self) -> Self {
//         f64::exp(self)
//     }

//     fn exp2(self) -> Self {
//         f64::exp2(self)
//     }

//     fn exp_m1(self) -> Self {
//         f64::exp_m1(self)
//     }

//     fn ln(self) -> Self {
//         f64::ln(self)
//     }

//     fn ln_1p(self) -> Self {
//         f64::ln_1p(self)
//     }

//     fn log2(self) -> Self {
//         f64::log2(self)
//     }

//     fn log10(self) -> Self {
//         f64::log10(self)
//     }

//     fn pow(self, y: Self) -> Self {
//         f64::powf(self, y)
//     }

//     fn powi(self, n: i32) -> Self {
//         f64::powi(self, n)
//     }

//     fn powf(self, y: Self) -> Self {
//         f64::powf(self, y)
//     }

//     fn cbrt(self) -> Self {
//         f64::cbrt(self)
//     }

//     fn sqrt(self) -> Self {
//         f64::sqrt(self)
//     }

//     fn square(self) -> Self::Real {
//         self * self
//     }

//     // ========== Trigonometric functions ==========
//     fn sin_cos(self) -> (Self, Self) {
//         f64::sin_cos(self)
//     }

//     fn sin(self) -> Self {
//         f64::sin(self)
//     }

//     fn cos(self) -> Self {
//         f64::cos(self)
//     }

//     fn tan(self) -> Self {
//         f64::tan(self)
//     }

//     fn asin(self) -> Self {
//         f64::asin(self)
//     }

//     fn acos(self) -> Self {
//         f64::acos(self)
//     }

//     fn atan(self) -> Self {
//         f64::atan(self)
//     }

//     // ========== Hyperbolic functions ==========
//     fn sinh(self) -> Self {
//         f64::sinh(self)
//     }

//     fn cosh(self) -> Self {
//         f64::cosh(self)
//     }

//     fn tanh(self) -> Self {
//         f64::tanh(self)
//     }

//     fn asinh(self) -> Self {
//         f64::asinh(self)
//     }

//     fn acosh(self) -> Self {
//         f64::acosh(self)
//     }

//     fn atanh(self) -> Self {
//         f64::atanh(self)
//     }
// }

// impl RFNum for MyFloat {
//     type Real = Self;

//     // ========== Special values (constants) ==========
//     fn nan() -> Self {
//         MyFloat::NAN
//     }

//     fn infinity() -> Self {
//         MyFloat::INFINITY
//     }

//     // ========== Classification methods ==========
//     fn is_nan(self) -> bool {
//         <MyFloat as Float>::is_nan(self)
//     }

//     fn is_infinite(self) -> bool {
//         <MyFloat as Float>::is_infinite(self)
//     }

//     fn is_finite(self) -> bool {
//         <MyFloat as Float>::is_finite(self)
//     }

//     fn is_normal(self) -> bool {
//         <MyFloat as Float>::is_normal(self)
//     }

//     // ========== Basic operations ==========
//     fn abs(self) -> Self {
//         <MyFloat as Float>::abs(self)
//     }

//     fn conj(self) -> Self {
//         self
//     }

//     fn norm(self) -> Self::Real {
//         <MyFloat as Float>::abs(self)
//     }

//     fn norm_sqr(self) -> Self::Real {
//         self * self
//     }

//     fn from_real(real: Self::Real) -> Self {
//         real
//     }

//     // ========== Exponential and power functions ==========
//     fn exp(self) -> Self {
//         <MyFloat as Float>::exp(self)
//     }

//     fn exp2(self) -> Self {
//         <MyFloat as Float>::exp2(self)
//     }

//     fn exp_m1(self) -> Self {
//         <MyFloat as Float>::exp_m1(self)
//     }

//     fn ln(self) -> Self {
//         <MyFloat as Float>::ln(self)
//     }

//     fn ln_1p(self) -> Self {
//         <MyFloat as Float>::ln_1p(self)
//     }

//     fn log2(self) -> Self {
//         <MyFloat as Float>::log2(self)
//     }

//     fn log10(self) -> Self {
//         <MyFloat as Float>::log10(self)
//     }

//     fn pow(self, y: Self) -> Self {
//         MyFloat::pow(&self, y)
//     }

//     fn powi(self, n: i32) -> Self {
//         <MyFloat as Float>::powi(self, n)
//     }

//     fn powf(self, y: Self) -> Self {
//         <MyFloat as Float>::powf(self, y)
//     }

//     fn cbrt(self) -> Self {
//         <MyFloat as Float>::cbrt(self)
//     }

//     fn sqrt(self) -> Self {
//         <MyFloat as Float>::sqrt(self)
//     }

//     fn square(self) -> Self::Real {
//         self * self
//     }

//     // ========== Trigonometric functions ==========
//     fn sin_cos(self) -> (Self, Self) {
//         <MyFloat as Float>::sin_cos(self)
//     }

//     fn sin(self) -> Self {
//         <MyFloat as Float>::sin(self)
//     }

//     fn cos(self) -> Self {
//         <MyFloat as Float>::cos(self)
//     }

//     fn tan(self) -> Self {
//         <MyFloat as Float>::tan(self)
//     }

//     fn asin(self) -> Self {
//         <MyFloat as Float>::asin(self)
//     }

//     fn acos(self) -> Self {
//         <MyFloat as Float>::acos(self)
//     }

//     fn atan(self) -> Self {
//         <MyFloat as Float>::atan(self)
//     }

//     // ========== Hyperbolic functions ==========
//     fn sinh(self) -> Self {
//         <MyFloat as Float>::sinh(self)
//     }

//     fn cosh(self) -> Self {
//         <MyFloat as Float>::cosh(self)
//     }

//     fn tanh(self) -> Self {
//         <MyFloat as Float>::tanh(self)
//     }

//     fn asinh(self) -> Self {
//         <MyFloat as Float>::asinh(self)
//     }

//     fn acosh(self) -> Self {
//         <MyFloat as Float>::acosh(self)
//     }

//     fn atanh(self) -> Self {
//         <MyFloat as Float>::atanh(self)
//     }
// }

// impl RFNum for Complex64 {
//     type Real = f64;

//     // ========== Special values (constants) ==========
//     fn nan() -> Self {
//         Complex::new(f64::NAN, f64::NAN)
//     }

//     fn infinity() -> Self {
//         Complex::new(f64::INFINITY, f64::INFINITY)
//     }

//     fn is_nan(self) -> bool {
//         self.re.is_nan() || self.im.is_nan()
//     }

//     fn is_infinite(self) -> bool {
//         self.re.is_infinite() || self.im.is_infinite()
//     }

//     fn is_finite(self) -> bool {
//         self.re.is_finite() && self.im.is_finite()
//     }

//     fn is_normal(self) -> bool {
//         self.re.is_normal() && self.im.is_normal()
//     }

//     // ========== Basic operations ==========
//     fn abs(self) -> Self {
//         Complex64::new(Complex::norm(self), 0.0)
//     }

//     fn conj(self) -> Self {
//         Complex::conj(&self)
//     }

//     fn norm(self) -> Self::Real {
//         Complex::norm(self)
//     }

//     fn norm_sqr(self) -> Self::Real {
//         Complex::norm_sqr(&self)
//     }

//     fn from_real(real: Self::Real) -> Self {
//         Complex::new(real, 0.0)
//     }

//     // ========== Exponential and power functions ==========
//     fn exp(self) -> Self {
//         Complex::exp(self)
//     }

//     fn exp2(self) -> Self {
//         Complex64::exp2(self)
//     }

//     fn exp_m1(self) -> Self {
//         self.exp() - 1.0
//     }

//     fn ln(self) -> Self {
//         Complex64::ln(self)
//     }

//     fn ln_1p(self) -> Self {
//         (self + 1.0).ln()
//     }

//     fn log2(self) -> Self {
//         Complex64::log2(self)
//     }

//     fn log10(self) -> Self {
//         Complex64::log10(self)
//     }

//     /// Raises self to a complex power
//     fn pow(self, y: Self) -> Self {
//         Complex::powc(self, y)
//     }

//     /// Raises self to an integer power
//     fn powi(self, n: i32) -> Self {
//         Complex::powi(&self, n)
//     }

//     fn powf(self, y: Self::Real) -> Self {
//         Complex64::powf(self, y)
//     }

//     /// Returns the cube root
//     fn cbrt(self) -> Self {
//         Complex::cbrt(self)
//     }

//     /// Returns the square root
//     fn sqrt(self) -> Self {
//         Complex::sqrt(self)
//     }

//     /// Returns the square
//     fn square(self) -> Self::Real {
//         Complex::norm(self * self)
//     }

//     // ========== Trigonometric functions ==========
//     fn sin_cos(self) -> (Self, Self) {
//         (self.sin(), self.cos())
//     }

//     fn sin(self) -> Self {
//         Complex::sin(self)
//     }

//     fn cos(self) -> Self {
//         Complex::cos(self)
//     }

//     fn tan(self) -> Self {
//         Complex::tan(self)
//     }

//     fn asin(self) -> Self {
//         Complex::asin(self)
//     }

//     fn acos(self) -> Self {
//         Complex::acos(self)
//     }

//     fn atan(self) -> Self {
//         Complex::atan(self)
//     }

//     // ========== Hyperbolic functions ==========
//     fn sinh(self) -> Self {
//         Complex::sinh(self)
//     }

//     fn cosh(self) -> Self {
//         Complex::cosh(self)
//     }

//     fn tanh(self) -> Self {
//         Complex::tanh(self)
//     }

//     fn asinh(self) -> Self {
//         Complex::asinh(self)
//     }

//     fn acosh(self) -> Self {
//         Complex::acosh(self)
//     }

//     fn atanh(self) -> Self {
//         Complex::atanh(self)
//     }
// }

// impl RFNum for MyComplex {
//     type Real = MyFloat;

//     // ========== Special values (constants) ==========
//     fn nan() -> Self {
//         MyComplex::nan()
//     }

//     fn infinity() -> Self {
//         MyComplex::infinity()
//     }

//     fn is_nan(self) -> bool {
//         MyComplex::is_nan(&self)
//     }

//     fn is_infinite(self) -> bool {
//         MyComplex::is_infinite(&self)
//     }

//     fn is_finite(self) -> bool {
//         MyComplex::is_finite(&self)
//     }

//     fn is_normal(self) -> bool {
//         MyComplex::is_normal(&self)
//     }

//     // ========== Basic operations ==========
//     fn abs(self) -> Self {
//         MyComplex::from_real(MyComplex::abs(&self))
//     }

//     fn conj(self) -> Self {
//         MyComplex::conj(&self)
//     }

//     fn norm(self) -> Self::Real {
//         MyComplex::norm(&self)
//     }

//     fn norm_sqr(self) -> Self::Real {
//         MyComplex::norm_sqr(&self)
//     }

//     fn from_real(real: Self::Real) -> Self {
//         MyComplex::from_real(real)
//     }

//     // ========== Exponential and power functions ==========
//     fn exp(self) -> Self {
//         MyComplex::exp(&self)
//     }

//     fn exp2(self) -> Self {
//         MyComplex::exp2(&self)
//     }

//     fn exp_m1(self) -> Self {
//         MyComplex::exp_m1(&self)
//     }

//     fn ln(self) -> Self {
//         MyComplex::ln(&self)
//     }

//     fn ln_1p(self) -> Self {
//         MyComplex::ln_1p(&self)
//     }

//     fn log2(self) -> Self {
//         MyComplex::log2(&self)
//     }

//     fn log10(self) -> Self {
//         MyComplex::log10(&self)
//     }

//     /// Raises self to a complex power
//     fn pow(self, y: Self) -> Self {
//         MyComplex::pow(&self, y)
//     }

//     /// Raises self to an integer power
//     fn powi(self, n: i32) -> Self {
//         // Convert to MyComplex power
//         let exp = Self::from_real(Self::Real::from_f64(n as f64));
//         self.pow(exp)
//     }

//     fn powf(self, y: Self::Real) -> Self {
//         MyComplex::from_complex(self.into_inner().powf(y.into_inner()))
//     }

//     /// Returns the cube root
//     fn cbrt(self) -> Self {
//         MyComplex::cbrt(&self)
//     }

//     /// Returns the square root
//     fn sqrt(self) -> Self {
//         MyComplex::sqrt(&self)
//     }

//     /// Returns the square
//     fn square(self) -> Self::Real {
//         MyComplex::norm(&(self * self))
//     }

//     // ========== Trigonometric functions ==========
//     fn sin_cos(self) -> (Self, Self) {
//         MyComplex::sin_cos(&self)
//     }

//     fn sin(self) -> Self {
//         MyComplex::sin(&self)
//     }

//     fn cos(self) -> Self {
//         MyComplex::cos(&self)
//     }

//     fn tan(self) -> Self {
//         // tan(z) = sin(z) / cos(z)
//         self.sin() / self.cos()
//     }

//     fn asin(self) -> Self {
//         // asin(z) = -i * ln(i*z + sqrt(1 - z^2))
//         // -Self::I * (Self::I * self + (Self::ONE - self * self).sqrt()).ln()
//         MyComplex::asin(&self)
//     }

//     fn acos(self) -> Self {
//         // acos(z) = -i * ln(z + i * sqrt(1 - z^2))
//         -Self::I * (self + Self::I * (Self::ONE - self * self).sqrt()).ln()
//     }

//     fn atan(self) -> Self {
//         // atan(z) = (i/2) * ln((i + z) / (i - z))
//         Self::I / 2.0 * ((Self::I + self) / (Self::I - self)).ln()
//     }

//     // ========== Hyperbolic functions ==========
//     fn sinh(self) -> Self {
//         // sinh(z) = (e^z - e^(-z)) / 2
//         (self.exp() - (-self).exp()) / 2.0
//     }

//     fn cosh(self) -> Self {
//         // cosh(z) = (e^z + e^(-z)) / 2
//         (self.exp() + (-self).exp()) / 2.0
//     }

//     fn tanh(self) -> Self {
//         // tanh(z) = sinh(z) / cosh(z)
//         self.sinh() / self.cosh()
//     }

//     fn asinh(self) -> Self {
//         // asinh(z) = ln(z + sqrt(z^2 + 1))
//         (self + (self * self + Self::ONE).sqrt()).ln()
//     }

//     fn acosh(self) -> Self {
//         // acosh(z) = ln(z + sqrt(z^2 - 1))
//         (self + (self * self - Self::ONE).sqrt()).ln()
//     }

//     fn atanh(self) -> Self {
//         // atanh(z) = (1/2) * ln((1 + z) / (1 - z))
//         Self::ONE / 2.0 * ((Self::ONE + self) / (Self::ONE - self)).ln()
//     }
// }

// /// Trait for numeric types that can be used in bracketing algorithms
// pub trait RFFloat:
//     RFNum
//     + Float
//     + PartialOrd
//     + for<'a> std::cmp::PartialEq<f64>
//     + std::ops::Add<MyFloat, Output = MyFloat>
//     + std::ops::Sub<MyFloat, Output = MyFloat>
//     + std::ops::Mul<MyFloat, Output = MyFloat>
//     + std::ops::Div<MyFloat, Output = MyFloat>
//     + for<'a> std::ops::Add<&'a MyFloat, Output = MyFloat>
//     + for<'a> std::ops::Sub<&'a MyFloat, Output = MyFloat>
//     + for<'a> std::ops::Mul<&'a MyFloat, Output = MyFloat>
//     + for<'a> std::ops::Div<&'a MyFloat, Output = MyFloat>
// {
//     // ========== Custom conversion methods ==========
//     /// Create from f64
//     fn from_f64(val: f64) -> Self;

//     /// Create from usize
//     fn from_usize(val: usize) -> Self;

//     /// Create to f64
//     fn to_f64(self) -> f64;

//     // ========== Special values (constants) ==========
//     /// Returns negative infinity
//     fn neg_infinity() -> Self;

//     /// Returns negative zero
//     fn neg_zero() -> Self;

//     /// Returns the smallest finite value that this type can represent
//     fn min_value() -> Self;

//     /// Returns the largest finite value that this type can represent
//     fn max_value() -> Self;

//     /// Returns the smallest positive, normalized value that this type can represent
//     fn min_positive_value() -> Self;

//     /// Returns the machine epsilon value for this type
//     fn epsilon() -> Self;

//     // ========== Classification methods ==========
//     /// Returns true if the number is subnormal
//     fn is_subnormal(self) -> bool;

//     /// Returns the floating point category of the number
//     fn classify(self) -> std::num::FpCategory;

//     /// Returns true if the sign bit is positive
//     fn is_sign_positive(self) -> bool;

//     /// Returns true if the sign bit is negative
//     fn is_sign_negative(self) -> bool;

//     // ========== Rounding methods ==========
//     /// Returns the largest integer less than or equal to a number
//     fn floor(self) -> Self;

//     /// Returns the smallest integer greater than or equal to a number
//     fn ceil(self) -> Self;

//     /// Returns the nearest integer to a number (round half-way cases away from 0.0)
//     fn round(self) -> Self;

//     /// Returns the integer part of a number
//     fn trunc(self) -> Self;

//     /// Returns the fractional part of a number
//     fn fract(self) -> Self;

//     // ========== Basic operations ==========
//     /// Returns a number that represents the sign of self
//     fn signum(self) -> Self;

//     /// Returns a number composed of the magnitude of self and the sign of sign
//     fn copysign(self, sign: Self) -> Self;

//     /// Fused multiply-add: (self * a) + b with only one rounding error
//     fn mul_add(self, a: Self, b: Self) -> Self;

//     /// Takes the reciprocal (inverse) of a number, 1/x
//     fn recip(self) -> Self;

//     /// Returns the maximum of the two numbers
//     fn max(self, other: Self) -> Self;

//     /// Returns the minimum of the two numbers
//     fn min(self, other: Self) -> Self;

//     /// The positive difference of two numbers
//     fn abs_sub(self, other: Self) -> Self;

//     /// Restrict a value to a certain interval
//     fn clamp(self, min: Self, max: Self) -> Self;

//     // ========== Exponential and power functions ==========
//     /// Returns the logarithm of the number with respect to an arbitrary base
//     fn log(self, base: Self) -> Self;

//     // ========== Trigonometric functions ==========
//     /// Computes the four quadrant arctangent of self (y) and other (x)
//     fn atan2(self, other: Self) -> Self;

//     /// Converts radians to degrees
//     fn to_degrees(self) -> Self;

//     /// Converts degrees to radians
//     fn to_radians(self) -> Self;

//     /// Computes the Euclidean distance: sqrt(self^2 + other^2)
//     fn hypot(self, other: Self) -> Self;

//     // ========== Other methods ==========
//     /// Returns the mantissa, exponent and sign as integers
//     fn integer_decode(self) -> (u64, i16, i8);
// }

// impl RFFloat for f64 {
//     // ========== Custom conversion methods ==========
//     fn from_f64(val: f64) -> Self {
//         val
//     }

//     fn from_usize(val: usize) -> Self {
//         val as f64
//     }

//     fn to_f64(self) -> f64 {
//         self
//     }

//     // ========== Special values (constants) ==========
//     fn neg_infinity() -> Self {
//         f64::NEG_INFINITY
//     }

//     fn neg_zero() -> Self {
//         -0.0
//     }

//     fn min_value() -> Self {
//         f64::MIN
//     }

//     fn max_value() -> Self {
//         f64::MAX
//     }

//     fn min_positive_value() -> Self {
//         f64::MIN_POSITIVE
//     }

//     fn epsilon() -> Self {
//         f64::EPSILON
//     }

//     // ========== Classification methods ==========
//     fn is_subnormal(self) -> bool {
//         f64::is_subnormal(self)
//     }

//     fn classify(self) -> std::num::FpCategory {
//         f64::classify(self)
//     }

//     fn is_sign_positive(self) -> bool {
//         f64::is_sign_positive(self)
//     }

//     fn is_sign_negative(self) -> bool {
//         f64::is_sign_negative(self)
//     }

//     // ========== Rounding methods ==========
//     fn floor(self) -> Self {
//         f64::floor(self)
//     }

//     fn ceil(self) -> Self {
//         f64::ceil(self)
//     }

//     fn round(self) -> Self {
//         f64::round(self)
//     }

//     fn trunc(self) -> Self {
//         f64::trunc(self)
//     }

//     fn fract(self) -> Self {
//         f64::fract(self)
//     }

//     // ========== Basic operations ==========
//     fn signum(self) -> Self {
//         f64::signum(self)
//     }

//     fn copysign(self, sign: Self) -> Self {
//         f64::copysign(self, sign)
//     }

//     fn mul_add(self, a: Self, b: Self) -> Self {
//         f64::mul_add(self, a, b)
//     }

//     fn recip(self) -> Self {
//         f64::recip(self)
//     }

//     fn max(self, other: Self) -> Self {
//         f64::max(self, other)
//     }

//     fn min(self, other: Self) -> Self {
//         f64::min(self, other)
//     }

//     fn abs_sub(self, other: Self) -> Self {
//         if self > other { self - other } else { 0.0 }
//     }

//     fn clamp(self, min: Self, max: Self) -> Self {
//         f64::clamp(self, min, max)
//     }

//     // ========== Exponential and power functions ==========
//     fn log(self, base: Self) -> Self {
//         f64::log(self, base)
//     }

//     // ========== Trigonometric functions ==========
//     fn atan2(self, other: Self) -> Self {
//         f64::atan2(self, other)
//     }

//     fn to_degrees(self) -> Self {
//         f64::to_degrees(self)
//     }

//     fn to_radians(self) -> Self {
//         f64::to_radians(self)
//     }

//     fn hypot(self, other: Self) -> Self {
//         f64::hypot(self, other)
//     }

//     // ========== Other methods ==========
//     fn integer_decode(self) -> (u64, i16, i8) {
//         num_traits::Float::integer_decode(self)
//     }
// }

// impl RFFloat for MyFloat {
//     // ========== Custom conversion methods ==========
//     fn from_f64(val: f64) -> Self {
//         MyFloat::new(val)
//     }

//     fn from_usize(val: usize) -> Self {
//         MyFloat::from_f64(val as f64)
//     }

//     fn to_f64(self) -> f64 {
//         MyFloat::to_f64(&self)
//     }

//     // ========== Special values (constants) ==========
//     fn neg_infinity() -> Self {
//         MyFloat::NEG_INFINITY
//     }

//     fn neg_zero() -> Self {
//         MyFloat::new(-0.0)
//     }

//     fn min_value() -> Self {
//         MyFloat::MIN
//     }

//     fn max_value() -> Self {
//         MyFloat::MAX
//     }

//     fn min_positive_value() -> Self {
//         MyFloat::MIN_POSITIVE
//     }

//     fn epsilon() -> Self {
//         MyFloat::EPSILON
//     }

//     // ========== Classification methods ==========
//     fn is_subnormal(self) -> bool {
//         self.to_f64().is_subnormal()
//     }

//     fn classify(self) -> std::num::FpCategory {
//         self.to_f64().classify()
//     }

//     fn is_sign_positive(self) -> bool {
//         <MyFloat as Float>::is_sign_positive(self)
//     }

//     fn is_sign_negative(self) -> bool {
//         <MyFloat as Float>::is_sign_negative(self)
//     }

//     // ========== Rounding methods ==========
//     fn floor(self) -> Self {
//         <MyFloat as Float>::floor(self)
//     }

//     fn ceil(self) -> Self {
//         <MyFloat as Float>::ceil(self)
//     }

//     fn round(self) -> Self {
//         <MyFloat as Float>::round(self)
//     }

//     fn trunc(self) -> Self {
//         <MyFloat as Float>::trunc(self)
//     }

//     fn fract(self) -> Self {
//         <MyFloat as Float>::fract(self)
//     }

//     // ========== Basic operations ==========
//     fn signum(self) -> Self {
//         <MyFloat as Float>::signum(self)
//     }

//     fn copysign(self, sign: Self) -> Self {
//         <MyFloat as Float>::copysign(self, sign)
//     }

//     fn mul_add(self, a: Self, b: Self) -> Self {
//         (self * a) + b
//     }

//     fn recip(self) -> Self {
//         <MyFloat as Float>::recip(self)
//     }

//     fn max(self, other: Self) -> Self {
//         <MyFloat as Float>::max(self, other)
//     }

//     fn min(self, other: Self) -> Self {
//         <MyFloat as Float>::min(self, other)
//     }

//     fn abs_sub(self, other: Self) -> Self {
//         if self > other {
//             self - other
//         } else {
//             Self::from_f64(0.0)
//         }
//     }

//     fn clamp(self, min: Self, max: Self) -> Self {
//         <MyFloat as Float>::clamp(self, min, max)
//     }

//     // ========== Exponential and power functions ==========
//     fn log(self, base: Self) -> Self {
//         <MyFloat as Float>::log(self, base)
//     }

//     // ========== Trigonometric functions ==========
//     fn atan2(self, other: Self) -> Self {
//         <MyFloat as Float>::atan2(self, other)
//     }

//     fn to_degrees(self) -> Self {
//         <MyFloat as Float>::to_degrees(self)
//     }

//     fn to_radians(self) -> Self {
//         <MyFloat as Float>::to_radians(self)
//     }

//     fn hypot(self, other: Self) -> Self {
//         <MyFloat as Float>::hypot(self, other)
//     }

//     // ========== Other methods ==========
//     fn integer_decode(self) -> (u64, i16, i8) {
//         <MyFloat as Float>::integer_decode(self)
//     }
// }

// /// Trait for complex numeric types that can be used in RF and optimization algorithms
// pub trait RFComplex:
//     RFNum
//     + std::ops::Add<Complex64, Output = Self>
//     + std::ops::Sub<Complex64, Output = Self>
//     + std::ops::Mul<Complex64, Output = Self>
//     + std::ops::Div<Complex64, Output = Self>
//     + std::ops::Add<MyFloat, Output = MyComplex>
//     + std::ops::Sub<MyFloat, Output = MyComplex>
//     + std::ops::Mul<MyFloat, Output = MyComplex>
//     + std::ops::Div<MyFloat, Output = MyComplex>
//     + std::ops::Add<MyComplex, Output = MyComplex>
//     + std::ops::Sub<MyComplex, Output = MyComplex>
//     + std::ops::Mul<MyComplex, Output = MyComplex>
//     + std::ops::Div<MyComplex, Output = MyComplex>
//     + std::ops::AddAssign<Complex64>
//     + std::ops::SubAssign<Complex64>
//     + std::ops::MulAssign<Complex64>
//     + std::ops::DivAssign<Complex64>
//     + for<'a> std::ops::Add<&'a MyFloat, Output = MyComplex>
//     + for<'a> std::ops::Sub<&'a MyFloat, Output = MyComplex>
//     + for<'a> std::ops::Mul<&'a MyFloat, Output = MyComplex>
//     + for<'a> std::ops::Div<&'a MyFloat, Output = MyComplex>
//     + for<'a> std::ops::Add<&'a MyComplex, Output = MyComplex>
//     + for<'a> std::ops::Sub<&'a MyComplex, Output = MyComplex>
//     + for<'a> std::ops::Mul<&'a MyComplex, Output = MyComplex>
//     + for<'a> std::ops::Div<&'a MyComplex, Output = MyComplex>
//     + std::convert::From<num::Complex<f64>>
// {
//     fn new(re: &Self::Real, im: &Self::Real) -> Self;

//     // ========== Custom conversion methods ==========
//     /// Create from real and imaginary f64 values
//     fn from_f64(real: f64, imag: f64) -> Self;

//     /// Create from real and imaginary parts using the Real type
//     fn from_real_imag(real: Self::Real, imag: Self::Real) -> Self;

//     /// Create from real part only (imaginary = 0)
//     fn from_real(real: Self::Real) -> Self;

//     /// Create from imaginary part only (real = 0)
//     fn from_imag(imag: Self::Real) -> Self;

//     /// Create from polar coordinates (magnitude, angle in radians)
//     fn from_polar(mag: &Self::Real, ang: &Self::Real) -> Self;

//     /// Get the real part
//     fn re(self) -> Self::Real;
//     fn real(self) -> Self::Real;

//     /// Get the imaginary part
//     fn im(self) -> Self::Real;
//     fn imag(self) -> Self::Real;

//     // ========== Complex-specific operations ==========
//     /// Get the argument (phase angle) of the complex number in radians
//     fn arg(self) -> Self::Real;
// }

// // Implement RFComplex for Complex<f64>
// impl RFComplex for Complex64 {
//     fn new(re: &Self::Real, im: &Self::Real) -> Self {
//         Complex::<f64>::new(*re, *im)
//     }

//     // ========== Custom conversion methods ==========
//     fn from_f64(real: f64, imag: f64) -> Self {
//         Complex::new(real, imag)
//     }

//     fn from_real_imag(real: Self::Real, imag: Self::Real) -> Self {
//         Complex::new(real, imag)
//     }

//     fn from_real(real: Self::Real) -> Self {
//         Complex::new(real, 0.0)
//     }

//     fn from_imag(imag: Self::Real) -> Self {
//         Complex::new(0.0, imag)
//     }

//     fn from_polar(mag: &Self::Real, ang: &Self::Real) -> Self {
//         Complex::from_polar(*mag, *ang)
//     }

//     fn re(self) -> Self::Real {
//         self.re
//     }

//     fn real(self) -> Self::Real {
//         self.re
//     }

//     fn im(self) -> Self::Real {
//         self.im
//     }

//     fn imag(self) -> Self::Real {
//         self.im
//     }

//     // ========== Complex-specific operations ==========
//     fn arg(self) -> Self::Real {
//         Complex::arg(self)
//     }
// }

// // Implement RFComplex for MyComplex
// impl RFComplex for MyComplex {
//     fn new(re: &Self::Real, im: &Self::Real) -> Self {
//         MyComplex::new(re.clone(), im.clone())
//     }

//     // ========== Custom conversion methods ==========
//     fn from_f64(real: f64, imag: f64) -> Self {
//         MyComplex::from_f64(real, imag)
//     }

//     fn from_real_imag(real: Self::Real, imag: Self::Real) -> Self {
//         MyComplex::new(real, imag)
//     }

//     fn from_real(real: Self::Real) -> Self {
//         MyComplex::from_real(real)
//     }

//     fn from_imag(imag: Self::Real) -> Self {
//         MyComplex::from_imag(imag)
//     }

//     fn from_polar(mag: &Self::Real, ang: &Self::Real) -> Self {
//         MyComplex::from_polar(mag, ang)
//     }

//     fn re(self) -> Self::Real {
//         MyComplex::re(&self)
//     }

//     fn real(self) -> Self::Real {
//         self.re()
//     }

//     fn im(self) -> Self::Real {
//         MyComplex::im(&self)
//     }

//     fn imag(self) -> Self::Real {
//         self.im()
//     }

//     // ========== Complex-specific operations ==========
//     fn arg(self) -> Self::Real {
//         MyComplex::arg(&self)
//     }
// }
