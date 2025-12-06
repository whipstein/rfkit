#![allow(dead_code)]
use std::fmt;

/// Trait for numeric types that can be used in bracketing algorithms
pub trait RFFloat:
    Sized
    + Clone
    + PartialOrd
    + fmt::Display
    + fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::Add<f64, Output = Self>
    + std::ops::Sub<f64, Output = Self>
    + std::ops::Mul<f64, Output = Self>
    + std::ops::Div<f64, Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::AddAssign<f64>
    + std::ops::SubAssign<f64>
    + std::ops::MulAssign<f64>
    + std::ops::DivAssign<f64>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
    + for<'a> std::ops::Mul<&'a Self, Output = Self>
    + for<'a> std::ops::Div<&'a Self, Output = Self>
    + for<'a> std::ops::Add<f64, Output = Self>
    + for<'a> std::ops::Sub<f64, Output = Self>
    + for<'a> std::ops::Mul<f64, Output = Self>
    + for<'a> std::ops::Div<f64, Output = Self>
    + for<'a> std::cmp::PartialEq<f64>
    + num_traits::Zero
    + num_traits::One
    + std::iter::Sum
    + for<'a> std::iter::Sum<&'a Self>
{
    // ========== Custom conversion methods ==========
    /// Create from f64
    fn from_f64(val: f64) -> Self;

    /// Create from usize
    fn from_usize(val: usize) -> Self;

    /// Create to f64
    fn to_f64(&self) -> f64;

    // ========== Special values (constants) ==========
    /// Returns the NaN value
    fn nan() -> Self;

    /// Returns positive infinity
    fn infinity() -> Self;

    /// Returns negative infinity
    fn neg_infinity() -> Self;

    /// Returns negative zero
    fn neg_zero() -> Self;

    /// Returns the smallest finite value that this type can represent
    fn min_value() -> Self;

    /// Returns the largest finite value that this type can represent
    fn max_value() -> Self;

    /// Returns the smallest positive, normalized value that this type can represent
    fn min_positive_value() -> Self;

    /// Returns the machine epsilon value for this type
    fn epsilon() -> Self;

    // ========== Classification methods ==========
    /// Returns true if this value is NaN
    fn is_nan(&self) -> bool;

    /// Returns true if this value is positive infinity or negative infinity
    fn is_infinite(&self) -> bool;

    /// Returns true if this number is neither infinite nor NaN
    fn is_finite(&self) -> bool;

    /// Returns true if the number is neither zero, infinite, subnormal, or NaN
    fn is_normal(&self) -> bool;

    /// Returns true if the number is subnormal
    fn is_subnormal(&self) -> bool;

    /// Returns the floating point category of the number
    fn classify(&self) -> std::num::FpCategory;

    /// Returns true if the sign bit is positive
    fn is_sign_positive(&self) -> bool;

    /// Returns true if the sign bit is negative
    fn is_sign_negative(&self) -> bool;

    // ========== Rounding methods ==========
    /// Returns the largest integer less than or equal to a number
    fn floor(&self) -> Self;

    /// Returns the smallest integer greater than or equal to a number
    fn ceil(&self) -> Self;

    /// Returns the nearest integer to a number (round half-way cases away from 0.0)
    fn round(&self) -> Self;

    /// Returns the integer part of a number
    fn trunc(&self) -> Self;

    /// Returns the fractional part of a number
    fn fract(&self) -> Self;

    // ========== Basic operations ==========
    /// Computes the absolute value of self
    fn abs(&self) -> Self;

    /// Returns a number that represents the sign of self
    fn signum(&self) -> Self;

    /// Returns a number composed of the magnitude of self and the sign of sign
    fn copysign(&self, sign: &Self) -> Self;

    /// Fused multiply-add: (self * a) + b with only one rounding error
    fn mul_add(&self, a: &Self, b: &Self) -> Self;

    /// Takes the reciprocal (inverse) of a number, 1/x
    fn recip(&self) -> Self;

    /// Returns the maximum of the two numbers
    fn max(&self, other: &Self) -> Self;

    /// Returns the minimum of the two numbers
    fn min(&self, other: &Self) -> Self;

    /// The positive difference of two numbers
    fn abs_sub(&self, other: &Self) -> Self;

    /// Restrict a value to a certain interval
    fn clamp(&self, min: &Self, max: &Self) -> Self;

    // ========== Exponential and power functions ==========
    /// Returns e^(self)
    fn exp(&self) -> Self;

    /// Returns 2^(self)
    fn exp2(&self) -> Self;

    /// Returns e^(self) - 1 in a way that is accurate even if the number is close to zero
    fn exp_m1(&self) -> Self;

    /// Returns the natural logarithm of the number
    fn ln(&self) -> Self;

    /// Returns ln(1+n) more accurately than if the operations were performed separately
    fn ln_1p(&self) -> Self;

    /// Returns the logarithm of the number with respect to an arbitrary base
    fn log(&self, base: &Self) -> Self;

    /// Returns the base 2 logarithm of the number
    fn log2(&self) -> Self;

    /// Returns the base 10 logarithm of the number
    fn log10(&self) -> Self;

    /// Raises self to the power of exp, using exponentiation by squaring
    fn powi(&self, n: i32) -> Self;

    /// Raises self to a floating point power
    fn powf(&self, n: &Self) -> Self;

    /// Returns the square root of a number
    fn sqrt(&self) -> Self;

    /// Returns the cube root of a number
    fn cbrt(&self) -> Self;

    // ========== Trigonometric functions ==========
    /// Computes the sine of a number (in radians)
    fn sin(&self) -> Self;

    /// Computes the cosine of a number (in radians)
    fn cos(&self) -> Self;

    /// Computes the tangent of a number (in radians)
    fn tan(&self) -> Self;

    /// Computes the arcsine of a number
    fn asin(&self) -> Self;

    /// Computes the arccosine of a number
    fn acos(&self) -> Self;

    /// Computes the arctangent of a number
    fn atan(&self) -> Self;

    /// Computes the four quadrant arctangent of self (y) and other (x)
    fn atan2(&self, other: &Self) -> Self;

    /// Simultaneously computes the sine and cosine of the number
    fn sin_cos(&self) -> (Self, Self);

    /// Converts radians to degrees
    fn to_degrees(&self) -> Self;

    /// Converts degrees to radians
    fn to_radians(&self) -> Self;

    /// Computes the Euclidean distance: sqrt(self^2 + other^2)
    fn hypot(&self, other: &Self) -> Self;

    // ========== Hyperbolic functions ==========
    /// Hyperbolic sine function
    fn sinh(&self) -> Self;

    /// Hyperbolic cosine function
    fn cosh(&self) -> Self;

    /// Hyperbolic tangent function
    fn tanh(&self) -> Self;

    /// Inverse hyperbolic sine function
    fn asinh(&self) -> Self;

    /// Inverse hyperbolic cosine function
    fn acosh(&self) -> Self;

    /// Inverse hyperbolic tangent function
    fn atanh(&self) -> Self;

    // ========== Other methods ==========
    /// Returns the mantissa, exponent and sign as integers
    fn integer_decode(&self) -> (u64, i16, i8);
}

// Implement RFFloat for f64
impl RFFloat for f64 {
    // ========== Custom conversion methods ==========
    fn from_f64(val: f64) -> Self {
        val
    }

    fn from_usize(val: usize) -> Self {
        val as f64
    }

    fn to_f64(&self) -> f64 {
        *self
    }

    // ========== Special values (constants) ==========
    fn nan() -> Self {
        f64::NAN
    }

    fn infinity() -> Self {
        f64::INFINITY
    }

    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }

    fn neg_zero() -> Self {
        -0.0
    }

    fn min_value() -> Self {
        f64::MIN
    }

    fn max_value() -> Self {
        f64::MAX
    }

    fn min_positive_value() -> Self {
        f64::MIN_POSITIVE
    }

    fn epsilon() -> Self {
        f64::EPSILON
    }

    // ========== Classification methods ==========
    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
    }

    fn is_infinite(&self) -> bool {
        f64::is_infinite(*self)
    }

    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }

    fn is_normal(&self) -> bool {
        f64::is_normal(*self)
    }

    fn is_subnormal(&self) -> bool {
        f64::is_subnormal(*self)
    }

    fn classify(&self) -> std::num::FpCategory {
        f64::classify(*self)
    }

    fn is_sign_positive(&self) -> bool {
        f64::is_sign_positive(*self)
    }

    fn is_sign_negative(&self) -> bool {
        f64::is_sign_negative(*self)
    }

    // ========== Rounding methods ==========
    fn floor(&self) -> Self {
        f64::floor(*self)
    }

    fn ceil(&self) -> Self {
        f64::ceil(*self)
    }

    fn round(&self) -> Self {
        f64::round(*self)
    }

    fn trunc(&self) -> Self {
        f64::trunc(*self)
    }

    fn fract(&self) -> Self {
        f64::fract(*self)
    }

    // ========== Basic operations ==========
    fn abs(&self) -> Self {
        f64::abs(*self)
    }

    fn signum(&self) -> Self {
        f64::signum(*self)
    }

    fn copysign(&self, sign: &Self) -> Self {
        f64::copysign(*self, *sign)
    }

    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        f64::mul_add(*self, *a, *b)
    }

    fn recip(&self) -> Self {
        f64::recip(*self)
    }

    fn max(&self, other: &Self) -> Self {
        f64::max(*self, *other)
    }

    fn min(&self, other: &Self) -> Self {
        f64::min(*self, *other)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        if *self > *other { self - other } else { 0.0 }
    }

    fn clamp(&self, min: &Self, max: &Self) -> Self {
        f64::clamp(*self, *min, *max)
    }

    // ========== Exponential and power functions ==========
    fn exp(&self) -> Self {
        f64::exp(*self)
    }

    fn exp2(&self) -> Self {
        f64::exp2(*self)
    }

    fn exp_m1(&self) -> Self {
        f64::exp_m1(*self)
    }

    fn ln(&self) -> Self {
        f64::ln(*self)
    }

    fn ln_1p(&self) -> Self {
        f64::ln_1p(*self)
    }

    fn log(&self, base: &Self) -> Self {
        f64::log(*self, *base)
    }

    fn log2(&self) -> Self {
        f64::log2(*self)
    }

    fn log10(&self) -> Self {
        f64::log10(*self)
    }

    fn powi(&self, n: i32) -> Self {
        f64::powi(*self, n)
    }

    fn powf(&self, n: &Self) -> Self {
        f64::powf(*self, *n)
    }

    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }

    fn cbrt(&self) -> Self {
        f64::cbrt(*self)
    }

    // ========== Trigonometric functions ==========
    fn sin(&self) -> Self {
        f64::sin(*self)
    }

    fn cos(&self) -> Self {
        f64::cos(*self)
    }

    fn tan(&self) -> Self {
        f64::tan(*self)
    }

    fn asin(&self) -> Self {
        f64::asin(*self)
    }

    fn acos(&self) -> Self {
        f64::acos(*self)
    }

    fn atan(&self) -> Self {
        f64::atan(*self)
    }

    fn atan2(&self, other: &Self) -> Self {
        f64::atan2(*self, *other)
    }

    fn sin_cos(&self) -> (Self, Self) {
        f64::sin_cos(*self)
    }

    fn to_degrees(&self) -> Self {
        f64::to_degrees(*self)
    }

    fn to_radians(&self) -> Self {
        f64::to_radians(*self)
    }

    fn hypot(&self, other: &Self) -> Self {
        f64::hypot(*self, *other)
    }

    // ========== Hyperbolic functions ==========
    fn sinh(&self) -> Self {
        f64::sinh(*self)
    }

    fn cosh(&self) -> Self {
        f64::cosh(*self)
    }

    fn tanh(&self) -> Self {
        f64::tanh(*self)
    }

    fn asinh(&self) -> Self {
        f64::asinh(*self)
    }

    fn acosh(&self) -> Self {
        f64::acosh(*self)
    }

    fn atanh(&self) -> Self {
        f64::atanh(*self)
    }

    // ========== Other methods ==========
    fn integer_decode(&self) -> (u64, i16, i8) {
        num_traits::Float::integer_decode(*self)
    }
}

// Implement RFFloat for MyFloat
impl RFFloat for crate::myfloat::MyFloat {
    // ========== Custom conversion methods ==========
    fn from_f64(val: f64) -> Self {
        crate::myfloat::MyFloat::new(val)
    }

    fn from_usize(val: usize) -> Self {
        crate::myfloat::MyFloat::from(val as f64)
    }

    fn to_f64(&self) -> f64 {
        self.to_f64()
    }

    // ========== Special values (constants) ==========
    fn nan() -> Self {
        crate::myfloat::MyFloat::nan()
    }

    fn infinity() -> Self {
        crate::myfloat::MyFloat::infinity()
    }

    fn neg_infinity() -> Self {
        crate::myfloat::MyFloat::neg_infinity()
    }

    fn neg_zero() -> Self {
        crate::myfloat::MyFloat::new(-0.0)
    }

    fn min_value() -> Self {
        crate::myfloat::MyFloat::new(f64::MIN)
    }

    fn max_value() -> Self {
        crate::myfloat::MyFloat::new(f64::MAX)
    }

    fn min_positive_value() -> Self {
        crate::myfloat::MyFloat::new(f64::MIN_POSITIVE)
    }

    fn epsilon() -> Self {
        crate::myfloat::MyFloat::new(f64::EPSILON)
    }

    // ========== Classification methods ==========
    fn is_nan(&self) -> bool {
        crate::myfloat::MyFloat::is_nan(self)
    }

    fn is_infinite(&self) -> bool {
        crate::myfloat::MyFloat::is_infinite(self)
    }

    fn is_finite(&self) -> bool {
        crate::myfloat::MyFloat::is_finite(self)
    }

    fn is_normal(&self) -> bool {
        crate::myfloat::MyFloat::is_normal(self)
    }

    fn is_subnormal(&self) -> bool {
        self.to_f64().is_subnormal()
    }

    fn classify(&self) -> std::num::FpCategory {
        self.to_f64().classify()
    }

    fn is_sign_positive(&self) -> bool {
        crate::myfloat::MyFloat::is_sign_positive(self)
    }

    fn is_sign_negative(&self) -> bool {
        crate::myfloat::MyFloat::is_sign_negative(self)
    }

    // ========== Rounding methods ==========
    fn floor(&self) -> Self {
        crate::myfloat::MyFloat::floor(self)
    }

    fn ceil(&self) -> Self {
        crate::myfloat::MyFloat::ceil(self)
    }

    fn round(&self) -> Self {
        crate::myfloat::MyFloat::round(self)
    }

    fn trunc(&self) -> Self {
        crate::myfloat::MyFloat::trunc(self)
    }

    fn fract(&self) -> Self {
        crate::myfloat::MyFloat::fract(self)
    }

    // ========== Basic operations ==========
    fn abs(&self) -> Self {
        crate::myfloat::MyFloat::abs(self)
    }

    fn signum(&self) -> Self {
        crate::myfloat::MyFloat::signum(self)
    }

    fn copysign(&self, sign: &Self) -> Self {
        crate::myfloat::MyFloat::copysign(self, sign)
    }

    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        (self * a) + b
    }

    fn recip(&self) -> Self {
        Self::from_f64(1.0) / self
    }

    fn max(&self, other: &Self) -> Self {
        crate::myfloat::MyFloat::max(self, other)
    }

    fn min(&self, other: &Self) -> Self {
        crate::myfloat::MyFloat::min(self, other)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        if self > other {
            self - other
        } else {
            Self::from_f64(0.0)
        }
    }

    fn clamp(&self, min: &Self, max: &Self) -> Self {
        crate::myfloat::MyFloat::clamp(self, min, max)
    }

    // ========== Exponential and power functions ==========
    fn exp(&self) -> Self {
        crate::myfloat::MyFloat::exp(self)
    }

    fn exp2(&self) -> Self {
        Self::from_f64(2.0).powf(self)
    }

    fn exp_m1(&self) -> Self {
        self.exp() - Self::from_f64(1.0)
    }

    fn ln(&self) -> Self {
        crate::myfloat::MyFloat::ln(self)
    }

    fn ln_1p(&self) -> Self {
        (self + Self::from_f64(1.0)).ln()
    }

    fn log(&self, base: &Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(&self) -> Self {
        crate::myfloat::MyFloat::log2(self)
    }

    fn log10(&self) -> Self {
        crate::myfloat::MyFloat::log10(self)
    }

    fn powi(&self, n: i32) -> Self {
        crate::myfloat::MyFloat::powi(self, n as isize)
    }

    fn powf(&self, n: &Self) -> Self {
        crate::myfloat::MyFloat::pow(self, n)
    }

    fn sqrt(&self) -> Self {
        crate::myfloat::MyFloat::sqrt(self)
    }

    fn cbrt(&self) -> Self {
        self.powf(&Self::from_f64(1.0 / 3.0))
    }

    // ========== Trigonometric functions ==========
    fn sin(&self) -> Self {
        crate::myfloat::MyFloat::sin(self)
    }

    fn cos(&self) -> Self {
        crate::myfloat::MyFloat::cos(self)
    }

    fn tan(&self) -> Self {
        crate::myfloat::MyFloat::tan(self)
    }

    fn asin(&self) -> Self {
        crate::myfloat::MyFloat::asin(self)
    }

    fn acos(&self) -> Self {
        crate::myfloat::MyFloat::acos(self)
    }

    fn atan(&self) -> Self {
        crate::myfloat::MyFloat::atan(self)
    }

    fn atan2(&self, other: &Self) -> Self {
        crate::myfloat::MyFloat::atan2(self, other)
    }

    fn sin_cos(&self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    fn to_degrees(&self) -> Self {
        crate::myfloat::MyFloat::to_degrees(self)
    }

    fn to_radians(&self) -> Self {
        crate::myfloat::MyFloat::to_radians(self)
    }

    fn hypot(&self, other: &Self) -> Self {
        (self.clone() * self.clone() + other.clone() * other.clone()).sqrt()
    }

    // ========== Hyperbolic functions ==========
    fn sinh(&self) -> Self {
        crate::myfloat::MyFloat::sinh(self)
    }

    fn cosh(&self) -> Self {
        crate::myfloat::MyFloat::cosh(self)
    }

    fn tanh(&self) -> Self {
        crate::myfloat::MyFloat::tanh(self)
    }

    fn asinh(&self) -> Self {
        crate::myfloat::MyFloat::asinh(self)
    }

    fn acosh(&self) -> Self {
        crate::myfloat::MyFloat::acosh(self)
    }

    fn atanh(&self) -> Self {
        crate::myfloat::MyFloat::atanh(self)
    }

    // ========== Other methods ==========
    fn integer_decode(&self) -> (u64, i16, i8) {
        self.to_f64().integer_decode()
    }
}

/// Trait for complex numeric types that can be used in RF and optimization algorithms
pub trait RFComplex:
    Sized
    + Clone
    + fmt::Display
    + fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::Add<f64, Output = Self>
    + std::ops::Sub<f64, Output = Self>
    + std::ops::Mul<f64, Output = Self>
    + std::ops::Div<f64, Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::AddAssign<f64>
    + std::ops::SubAssign<f64>
    + std::ops::MulAssign<f64>
    + std::ops::DivAssign<f64>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
    + for<'a> std::ops::Mul<&'a Self, Output = Self>
    + for<'a> std::ops::Div<&'a Self, Output = Self>
    + num_traits::Zero
    + num_traits::One
{
    /// The underlying real type (e.g., f64 for Complex<f64>, MyFloat for MyComplex)
    type Real: RFFloat;

    // ========== Custom conversion methods ==========
    /// Create from real and imaginary f64 values
    fn from_f64(real: f64, imag: f64) -> Self;

    /// Create from real and imaginary parts using the Real type
    fn from_real_imag(real: Self::Real, imag: Self::Real) -> Self;

    /// Create from real part only (imaginary = 0)
    fn from_real(real: Self::Real) -> Self;

    /// Create from imaginary part only (real = 0)
    fn from_imag(imag: Self::Real) -> Self;

    /// Create from polar coordinates (magnitude, angle in radians)
    fn from_polar(mag: &Self::Real, ang: &Self::Real) -> Self;

    /// Get the real part
    fn re(&self) -> Self::Real;

    /// Get the imaginary part
    fn im(&self) -> Self::Real;

    // ========== Special values (constants) ==========
    /// Returns a NaN complex number
    fn nan() -> Self;

    /// Returns an infinite complex number
    fn infinity() -> Self;

    /// Check if either component is NaN
    fn is_nan(&self) -> bool;

    /// Check if either component is infinite
    fn is_infinite(&self) -> bool;

    /// Check if both components are finite
    fn is_finite(&self) -> bool;

    /// Check if both components are normal (not zero, infinite, subnormal, or NaN)
    fn is_normal(&self) -> bool;

    // ========== Complex-specific operations ==========
    /// Get the magnitude (absolute value) of the complex number
    fn abs(&self) -> Self::Real;

    /// Get the argument (phase angle) of the complex number in radians
    fn arg(&self) -> Self::Real;

    /// Get the complex conjugate
    fn conj(&self) -> Self;

    /// Calculate the square of the magnitude (norm squared)
    fn norm_sqr(&self) -> Self::Real;

    // ========== Exponential and power functions ==========
    /// Returns e^(self)
    fn exp(&self) -> Self;

    /// Returns the natural logarithm
    fn ln(&self) -> Self;

    /// Raises self to a complex power
    fn pow(&self, exp: &Self) -> Self;

    /// Raises self to an integer power
    fn powi(&self, n: i32) -> Self;

    /// Returns the square root
    fn sqrt(&self) -> Self;

    // ========== Trigonometric functions ==========
    /// Computes the sine
    fn sin(&self) -> Self;

    /// Computes the cosine
    fn cos(&self) -> Self;

    /// Computes the tangent
    fn tan(&self) -> Self;

    /// Computes the arcsine
    fn asin(&self) -> Self;

    /// Computes the arccosine
    fn acos(&self) -> Self;

    /// Computes the arctangent
    fn atan(&self) -> Self;

    // ========== Hyperbolic functions ==========
    /// Hyperbolic sine function
    fn sinh(&self) -> Self;

    /// Hyperbolic cosine function
    fn cosh(&self) -> Self;

    /// Hyperbolic tangent function
    fn tanh(&self) -> Self;

    /// Inverse hyperbolic sine function
    fn asinh(&self) -> Self;

    /// Inverse hyperbolic cosine function
    fn acosh(&self) -> Self;

    /// Inverse hyperbolic tangent function
    fn atanh(&self) -> Self;
}

// Implement RFComplex for Complex<f64>
impl RFComplex for num::complex::Complex<f64> {
    type Real = f64;

    // ========== Custom conversion methods ==========
    fn from_f64(real: f64, imag: f64) -> Self {
        num::complex::Complex::new(real, imag)
    }

    fn from_real_imag(real: Self::Real, imag: Self::Real) -> Self {
        num::complex::Complex::new(real, imag)
    }

    fn from_real(real: Self::Real) -> Self {
        num::complex::Complex::new(real, 0.0)
    }

    fn from_imag(imag: Self::Real) -> Self {
        num::complex::Complex::new(0.0, imag)
    }

    fn from_polar(mag: &Self::Real, ang: &Self::Real) -> Self {
        num::complex::Complex::from_polar(*mag, *ang)
    }

    fn re(&self) -> Self::Real {
        self.re
    }

    fn im(&self) -> Self::Real {
        self.im
    }

    // ========== Special values (constants) ==========
    fn nan() -> Self {
        num::complex::Complex::new(f64::NAN, f64::NAN)
    }

    fn infinity() -> Self {
        num::complex::Complex::new(f64::INFINITY, f64::INFINITY)
    }

    fn is_nan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    fn is_infinite(&self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }

    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn is_normal(&self) -> bool {
        self.re.is_normal() && self.im.is_normal()
    }

    // ========== Complex-specific operations ==========
    fn abs(&self) -> Self::Real {
        num::complex::Complex::norm(*self)
    }

    fn arg(&self) -> Self::Real {
        num::complex::Complex::arg(*self)
    }

    fn conj(&self) -> Self {
        num::complex::Complex::conj(self)
    }

    fn norm_sqr(&self) -> Self::Real {
        num::complex::Complex::norm_sqr(self)
    }

    // ========== Exponential and power functions ==========
    fn exp(&self) -> Self {
        num::complex::Complex::exp(*self)
    }

    fn ln(&self) -> Self {
        num::complex::Complex::ln(*self)
    }

    fn pow(&self, exp: &Self) -> Self {
        num::complex::Complex::powc(*self, *exp)
    }

    fn powi(&self, n: i32) -> Self {
        num::complex::Complex::powi(self, n)
    }

    fn sqrt(&self) -> Self {
        num::complex::Complex::sqrt(*self)
    }

    // ========== Trigonometric functions ==========
    fn sin(&self) -> Self {
        num::complex::Complex::sin(*self)
    }

    fn cos(&self) -> Self {
        num::complex::Complex::cos(*self)
    }

    fn tan(&self) -> Self {
        num::complex::Complex::tan(*self)
    }

    fn asin(&self) -> Self {
        num::complex::Complex::asin(*self)
    }

    fn acos(&self) -> Self {
        num::complex::Complex::acos(*self)
    }

    fn atan(&self) -> Self {
        num::complex::Complex::atan(*self)
    }

    // ========== Hyperbolic functions ==========
    fn sinh(&self) -> Self {
        num::complex::Complex::sinh(*self)
    }

    fn cosh(&self) -> Self {
        num::complex::Complex::cosh(*self)
    }

    fn tanh(&self) -> Self {
        num::complex::Complex::tanh(*self)
    }

    fn asinh(&self) -> Self {
        num::complex::Complex::asinh(*self)
    }

    fn acosh(&self) -> Self {
        num::complex::Complex::acosh(*self)
    }

    fn atanh(&self) -> Self {
        num::complex::Complex::atanh(*self)
    }
}

// Implement RFComplex for MyComplex
impl RFComplex for crate::mycomplex::MyComplex {
    type Real = crate::myfloat::MyFloat;

    // ========== Custom conversion methods ==========
    fn from_f64(real: f64, imag: f64) -> Self {
        crate::mycomplex::MyComplex::from_f64(real, imag)
    }

    fn from_real_imag(real: Self::Real, imag: Self::Real) -> Self {
        crate::mycomplex::MyComplex::new(real, imag)
    }

    fn from_real(real: Self::Real) -> Self {
        crate::mycomplex::MyComplex::from_real(real)
    }

    fn from_imag(imag: Self::Real) -> Self {
        crate::mycomplex::MyComplex::from_imag(imag)
    }

    fn from_polar(mag: &Self::Real, ang: &Self::Real) -> Self {
        crate::mycomplex::MyComplex::from_polar(mag, ang)
    }

    fn re(&self) -> Self::Real {
        self.re()
    }

    fn im(&self) -> Self::Real {
        self.im()
    }

    // ========== Special values (constants) ==========
    fn nan() -> Self {
        crate::mycomplex::MyComplex::nan()
    }

    fn infinity() -> Self {
        crate::mycomplex::MyComplex::infinity()
    }

    fn is_nan(&self) -> bool {
        crate::mycomplex::MyComplex::is_nan(self)
    }

    fn is_infinite(&self) -> bool {
        crate::mycomplex::MyComplex::is_infinite(self)
    }

    fn is_finite(&self) -> bool {
        crate::mycomplex::MyComplex::is_finite(self)
    }

    fn is_normal(&self) -> bool {
        crate::mycomplex::MyComplex::is_normal(self)
    }

    // ========== Complex-specific operations ==========
    fn abs(&self) -> Self::Real {
        crate::mycomplex::MyComplex::abs(self)
    }

    fn arg(&self) -> Self::Real {
        crate::mycomplex::MyComplex::arg(self)
    }

    fn conj(&self) -> Self {
        crate::mycomplex::MyComplex::conj(self)
    }

    fn norm_sqr(&self) -> Self::Real {
        crate::mycomplex::MyComplex::norm_sqr(self)
    }

    // ========== Exponential and power functions ==========
    fn exp(&self) -> Self {
        crate::mycomplex::MyComplex::exp(self)
    }

    fn ln(&self) -> Self {
        crate::mycomplex::MyComplex::ln(self)
    }

    fn pow(&self, exp: &Self) -> Self {
        crate::mycomplex::MyComplex::pow(self, exp)
    }

    fn powi(&self, n: i32) -> Self {
        // Convert to MyComplex power
        let exp = Self::from_real(Self::Real::from_f64(n as f64));
        self.pow(&exp)
    }

    fn sqrt(&self) -> Self {
        crate::mycomplex::MyComplex::sqrt(self)
    }

    // ========== Trigonometric functions ==========
    fn sin(&self) -> Self {
        crate::mycomplex::MyComplex::sin(self)
    }

    fn cos(&self) -> Self {
        crate::mycomplex::MyComplex::cos(self)
    }

    fn tan(&self) -> Self {
        // tan(z) = sin(z) / cos(z)
        self.sin() / self.cos()
    }

    fn asin(&self) -> Self {
        // asin(z) = -i * ln(i*z + sqrt(1 - z^2))
        let i = Self::from_imag(Self::Real::from_f64(1.0));
        let one = Self::from_real(Self::Real::from_f64(1.0));
        let iz = &i * self;
        let sqrt_part = (one - self * self).sqrt();
        -&i * (iz + sqrt_part).ln()
    }

    fn acos(&self) -> Self {
        // acos(z) = -i * ln(z + i * sqrt(1 - z^2))
        let i = Self::from_imag(Self::Real::from_f64(1.0));
        let one = Self::from_real(Self::Real::from_f64(1.0));
        let sqrt_part = (one - self * self).sqrt();
        -&i * (self + &i * sqrt_part).ln()
    }

    fn atan(&self) -> Self {
        // atan(z) = (i/2) * ln((i + z) / (i - z))
        let i = Self::from_imag(Self::Real::from_f64(1.0));
        let two = Self::from_real(Self::Real::from_f64(2.0));
        let i_half = &i / two;
        let numerator = &i + self;
        let denominator = &i - self;
        i_half * (numerator / denominator).ln()
    }

    // ========== Hyperbolic functions ==========
    fn sinh(&self) -> Self {
        // sinh(z) = (e^z - e^(-z)) / 2
        let exp_z = self.exp();
        let exp_neg_z = (-self).exp();
        let two = Self::from_real(Self::Real::from_f64(2.0));
        (exp_z - exp_neg_z) / two
    }

    fn cosh(&self) -> Self {
        // cosh(z) = (e^z + e^(-z)) / 2
        let exp_z = self.exp();
        let exp_neg_z = (-self).exp();
        let two = Self::from_real(Self::Real::from_f64(2.0));
        (exp_z + exp_neg_z) / two
    }

    fn tanh(&self) -> Self {
        // tanh(z) = sinh(z) / cosh(z)
        self.sinh() / self.cosh()
    }

    fn asinh(&self) -> Self {
        // asinh(z) = ln(z + sqrt(z^2 + 1))
        let one = Self::from_real(Self::Real::from_f64(1.0));
        (self + (self * self + one).sqrt()).ln()
    }

    fn acosh(&self) -> Self {
        // acosh(z) = ln(z + sqrt(z^2 - 1))
        let one = Self::from_real(Self::Real::from_f64(1.0));
        (self + (self * self - one).sqrt()).ln()
    }

    fn atanh(&self) -> Self {
        // atanh(z) = (1/2) * ln((1 + z) / (1 - z))
        let one = Self::from_real(Self::Real::from_f64(1.0));
        let two = Self::from_real(Self::Real::from_f64(2.0));
        let half = one.clone() / two;
        let numerator = &one + self;
        let denominator = one - self;
        half * (numerator / denominator).ln()
    }
}
