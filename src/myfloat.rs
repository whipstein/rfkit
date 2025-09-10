use core::f64;
use num_traits::{Num, One, Zero};
use rug::Float;
use rug::ops::{Pow, PowAssign};
use std::fmt;
use std::iter::Sum;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A float wrapper with fixed precision of 53 bits
pub struct MyFloat(rug::Float);

impl MyFloat {
    /// Fixed precision for all operations
    const PRECISION: u32 = 53;

    /// Create a new float from an f64 value
    pub fn new(value: f64) -> Self {
        MyFloat(Float::with_val(Self::PRECISION, value))
    }

    /// Create a new float from an Float value
    pub fn from_float(value: Float) -> Self {
        MyFloat(value)
    }

    /// Get the value as f64
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }

    /// Get the value as f64
    pub fn to_float(&self) -> Float {
        self.0.clone()
    }

    /// Convert the value in radians to degrees
    pub fn to_degrees(&self) -> MyFloat {
        self * 180.0 / f64::consts::PI
    }

    /// Convert the value in degrees to radians
    pub fn to_radians(&self) -> MyFloat {
        self * f64::consts::PI / 180.0
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        let mut temp = self.0.clone();
        temp.abs_mut();
        MyFloat(temp)
    }

    /// Get the integer power
    pub fn powi(&self, n: isize) -> Self {
        if n == 0 {
            return MyFloat::zero();
        }

        let mut temp = self.clone();
        for _ in 1..n.abs() {
            temp *= self;
        }
        if n < 0 {
            temp = 1.0 / temp;
        }

        temp
    }

    /// Get the magnitude in dB
    pub fn db(&self) -> MyFloat {
        20.0 * self.abs().log10()
    }

    /// Get the magnitude in dB
    pub fn db10(&self) -> MyFloat {
        10.0 * self.abs().log10()
    }

    /// Get the magnitude in dB
    pub fn db2mag(&self) -> MyFloat {
        MyFloat::new(10.0).pow(self / 20.0)
    }

    /// Get the magnitude in dB
    pub fn db102mag(&self) -> MyFloat {
        MyFloat::new(10.0).pow(self / 10.0)
    }

    /// Get the square root
    pub fn sqrt(&self) -> Self {
        let mut temp = self.0.clone();
        temp.sqrt_mut();
        MyFloat(temp)
    }

    /// Calculate the exponential function
    pub fn exp(&self) -> Self {
        let mut temp = self.0.clone();
        temp.exp_mut();
        MyFloat(temp)
    }

    /// Calculate the natural logarithm
    pub fn ln(&self) -> Self {
        let mut temp = self.0.clone();
        temp.ln_mut();
        MyFloat(temp)
    }

    /// Calculate the base-10 logarithm
    pub fn log10(&self) -> Self {
        let mut temp = self.0.clone();
        temp.log10_mut();
        MyFloat(temp)
    }

    /// Calculate the base-2 logarithm
    pub fn log2(&self) -> Self {
        let mut temp = self.0.clone();
        temp.log2_mut();
        MyFloat(temp)
    }

    /// Calculate sine
    pub fn sin(&self) -> Self {
        let mut temp = self.0.clone();
        temp.sin_mut();
        MyFloat(temp)
    }

    /// Calculate cosine
    pub fn cos(&self) -> Self {
        let mut temp = self.0.clone();
        temp.cos_mut();
        MyFloat(temp)
    }

    /// Calculate tangent
    pub fn tan(&self) -> Self {
        let mut temp = self.0.clone();
        temp.tan_mut();
        MyFloat(temp)
    }

    /// Calculate arcsine
    pub fn asin(&self) -> Self {
        let mut temp = self.0.clone();
        temp.asin_mut();
        MyFloat(temp)
    }

    /// Calculate arccosine
    pub fn acos(&self) -> Self {
        let mut temp = self.0.clone();
        temp.acos_mut();
        MyFloat(temp)
    }

    /// Calculate arctangent
    pub fn atan(&self) -> Self {
        let mut temp = self.0.clone();
        temp.atan_mut();
        MyFloat(temp)
    }

    /// Calculate arctangent of y/x
    pub fn atan2(&self, x: &MyFloat) -> Self {
        let mut temp = self.0.clone();
        temp.atan2_mut(&x.0);
        MyFloat(temp)
    }

    /// Calculate hyperbolic sine
    pub fn sinh(&self) -> Self {
        let mut temp = self.0.clone();
        temp.sinh_mut();
        MyFloat(temp)
    }

    /// Calculate hyperbolic cosine
    pub fn cosh(&self) -> Self {
        let mut temp = self.0.clone();
        temp.cosh_mut();
        MyFloat(temp)
    }

    /// Calculate hyperbolic tangent
    pub fn tanh(&self) -> Self {
        let mut temp = self.0.clone();
        temp.tanh_mut();
        MyFloat(temp)
    }

    /// Calculate inverse hyperbolic sine
    pub fn asinh(&self) -> Self {
        let mut temp = self.0.clone();
        temp.asinh_mut();
        MyFloat(temp)
    }

    /// Calculate inverse hyperbolic cosine
    pub fn acosh(&self) -> Self {
        let mut temp = self.0.clone();
        temp.acosh_mut();
        MyFloat(temp)
    }

    /// Calculate inverse hyperbolic tangent
    pub fn atanh(&self) -> Self {
        let mut temp = self.0.clone();
        temp.atanh_mut();
        MyFloat(temp)
    }

    /// Raise to a power
    pub fn pow(&self, exp: &MyFloat) -> Self {
        let mut temp = self.0.clone();
        temp.pow_assign(&exp.0);
        MyFloat(temp)
    }

    /// Calculate the floor (largest integer less than or equal to self)
    pub fn floor(&self) -> Self {
        let mut temp = self.0.clone();
        temp.floor_mut();
        MyFloat(temp)
    }

    /// Calculate the ceiling (smallest integer greater than or equal to self)
    pub fn ceil(&self) -> Self {
        let mut temp = self.0.clone();
        temp.ceil_mut();
        MyFloat(temp)
    }

    /// Round to the nearest integer
    pub fn round(&self) -> Self {
        let mut temp = self.0.clone();
        temp.round_mut();
        MyFloat(temp)
    }

    /// Truncate towards zero
    pub fn trunc(&self) -> Self {
        let mut temp = self.0.clone();
        temp.trunc_mut();
        MyFloat(temp)
    }

    /// Get the fractional part
    pub fn fract(&self) -> Self {
        let mut temp = self.0.clone();
        temp.fract_mut();
        MyFloat(temp)
    }

    /// Access the inner rug::Float (for advanced operations)
    pub fn inner(&self) -> &rug::Float {
        &self.0
    }

    /// Convert to inner rug::Float (consuming self)
    pub fn into_inner(self) -> rug::Float {
        self.0
    }

    /// Create a NaN float
    pub fn nan() -> Self {
        MyFloat(Float::with_val(Self::PRECISION, f64::NAN))
    }

    /// Check if the float is NaN
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    /// Create an infinite float
    pub fn infinity() -> Self {
        MyFloat(Float::with_val(Self::PRECISION, f64::INFINITY))
    }

    /// Create a negative infinite float
    pub fn neg_infinity() -> Self {
        MyFloat(Float::with_val(Self::PRECISION, f64::NEG_INFINITY))
    }

    /// Check if the float is infinite
    pub fn is_infinite(&self) -> bool {
        self.0.is_infinite()
    }

    /// Check if the float is finite
    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    /// Check if the float is normal (not zero, infinite, or NaN)
    pub fn is_normal(&self) -> bool {
        self.0.is_normal()
    }

    /// Check if the float is positive
    pub fn is_sign_positive(&self) -> bool {
        self.0.is_sign_positive()
    }

    /// Check if the float is negative
    pub fn is_sign_negative(&self) -> bool {
        self.0.is_sign_negative()
    }

    /// Get the minimum of two floats
    pub fn min(&self, other: &MyFloat) -> Self {
        let mut temp = self.0.clone();
        temp.min_mut(&other.0);
        MyFloat(temp)
    }

    /// Get the maximum of two floats
    pub fn max(&self, other: &MyFloat) -> Self {
        let mut temp = self.0.clone();
        temp.max_mut(&other.0);
        MyFloat(temp)
    }

    /// Clamp the value between min and max
    pub fn clamp(&self, min: &MyFloat, max: &MyFloat) -> Self {
        self.max(min).min(max)
    }

    /// Get the sign of the number (-1, 0, or 1)
    pub fn signum(&self) -> Self {
        if self.0.is_nan() {
            MyFloat::nan()
        } else if self.0.is_zero() {
            MyFloat::zero()
        } else if self.0.is_sign_positive() {
            MyFloat::one()
        } else {
            -MyFloat::one()
        }
    }

    /// Copy the sign from another float
    pub fn copysign(&self, sign: &MyFloat) -> Self {
        let mut temp = self.0.clone();
        temp.copysign_mut(&sign.0);
        MyFloat(temp)
    }
}

// Implement basic arithmetic operations
impl Add for MyFloat {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        MyFloat(self.0 + other.0)
    }
}

impl Add<&MyFloat> for MyFloat {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        MyFloat(self.0 + &other.0)
    }
}

impl Add<MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn add(self, other: MyFloat) -> MyFloat {
        MyFloat(&self.0 + other.0)
    }
}

impl Add<&MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn add(self, other: &MyFloat) -> MyFloat {
        MyFloat(&self.0 + other.0.clone())
    }
}

impl Add<f64> for MyFloat {
    type Output = Self;

    fn add(self, other: f64) -> Self::Output {
        MyFloat(self.0 + other)
    }
}

impl Add<f64> for &MyFloat {
    type Output = MyFloat;

    fn add(self, other: f64) -> Self::Output {
        MyFloat(self.0.clone() + other)
    }
}

impl Add<MyFloat> for f64 {
    type Output = MyFloat;

    fn add(self, other: MyFloat) -> Self::Output {
        MyFloat(self + other.0)
    }
}

impl Add<&MyFloat> for f64 {
    type Output = MyFloat;

    fn add(self, other: &MyFloat) -> Self::Output {
        MyFloat(self + other.0.clone())
    }
}

impl Sub for MyFloat {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        MyFloat(self.0 - other.0)
    }
}

impl Sub<&MyFloat> for MyFloat {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        MyFloat(self.0 - &other.0)
    }
}

impl Sub<MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn sub(self, other: MyFloat) -> MyFloat {
        MyFloat(&self.0 - other.0)
    }
}

impl Sub<&MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn sub(self, other: &MyFloat) -> MyFloat {
        MyFloat(&self.0 - other.0.clone())
    }
}

impl Sub<f64> for MyFloat {
    type Output = Self;

    fn sub(self, other: f64) -> Self::Output {
        MyFloat(self.0 - other)
    }
}

impl Sub<f64> for &MyFloat {
    type Output = MyFloat;

    fn sub(self, other: f64) -> Self::Output {
        MyFloat(self.0.clone() - other)
    }
}

impl Sub<MyFloat> for f64 {
    type Output = MyFloat;

    fn sub(self, other: MyFloat) -> Self::Output {
        MyFloat(self - other.0)
    }
}

impl Sub<&MyFloat> for f64 {
    type Output = MyFloat;

    fn sub(self, other: &MyFloat) -> Self::Output {
        MyFloat(self - other.0.clone())
    }
}

impl Mul for MyFloat {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        MyFloat(self.0 * other.0)
    }
}

impl Mul<&MyFloat> for MyFloat {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        MyFloat(self.0 * &other.0)
    }
}

impl Mul<MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn mul(self, other: MyFloat) -> MyFloat {
        MyFloat(&self.0 * other.0)
    }
}

impl Mul<&MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn mul(self, other: &MyFloat) -> MyFloat {
        MyFloat(&self.0 * other.0.clone())
    }
}

impl Mul<f64> for MyFloat {
    type Output = Self;

    fn mul(self, other: f64) -> Self::Output {
        MyFloat(self.0 * other)
    }
}

impl Mul<f64> for &MyFloat {
    type Output = MyFloat;

    fn mul(self, other: f64) -> Self::Output {
        MyFloat(self.0.clone() * other)
    }
}

impl Mul<MyFloat> for f64 {
    type Output = MyFloat;

    fn mul(self, other: MyFloat) -> Self::Output {
        MyFloat(self * other.0)
    }
}

impl Mul<&MyFloat> for f64 {
    type Output = MyFloat;

    fn mul(self, other: &MyFloat) -> Self::Output {
        MyFloat(self * other.0.clone())
    }
}

impl Div for MyFloat {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        MyFloat(self.0 / other.0)
    }
}

impl Div<&MyFloat> for MyFloat {
    type Output = Self;

    fn div(self, other: &Self) -> Self {
        MyFloat(self.0 / &other.0)
    }
}

impl Div<MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn div(self, other: MyFloat) -> MyFloat {
        MyFloat(&self.0 / other.0)
    }
}

impl Div<&MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn div(self, other: &MyFloat) -> MyFloat {
        MyFloat(&self.0 / other.0.clone())
    }
}

impl Div<f64> for MyFloat {
    type Output = Self;

    fn div(self, other: f64) -> Self::Output {
        MyFloat(self.0 / other)
    }
}

impl Div<f64> for &MyFloat {
    type Output = MyFloat;

    fn div(self, other: f64) -> Self::Output {
        MyFloat(self.0.clone() / other)
    }
}

impl Div<MyFloat> for f64 {
    type Output = MyFloat;

    fn div(self, other: MyFloat) -> Self::Output {
        MyFloat(self / other.0)
    }
}

impl Div<&MyFloat> for f64 {
    type Output = MyFloat;

    fn div(self, other: &MyFloat) -> Self::Output {
        MyFloat(self / other.0.clone())
    }
}

impl Neg for MyFloat {
    type Output = Self;

    fn neg(self) -> Self {
        MyFloat(-self.0)
    }
}

impl Neg for &MyFloat {
    type Output = MyFloat;

    fn neg(self) -> MyFloat {
        MyFloat(-self.0.clone())
    }
}

// Implement assignment operators
impl AddAssign for MyFloat {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl AddAssign<&MyFloat> for MyFloat {
    fn add_assign(&mut self, other: &MyFloat) {
        self.0 += &other.0;
    }
}

impl AddAssign<f64> for MyFloat {
    fn add_assign(&mut self, other: f64) {
        self.0 += other;
    }
}

impl SubAssign for MyFloat {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl SubAssign<&MyFloat> for MyFloat {
    fn sub_assign(&mut self, other: &MyFloat) {
        self.0 -= &other.0;
    }
}

impl SubAssign<f64> for MyFloat {
    fn sub_assign(&mut self, other: f64) {
        self.0 -= other;
    }
}

impl MulAssign for MyFloat {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl MulAssign<&MyFloat> for MyFloat {
    fn mul_assign(&mut self, other: &MyFloat) {
        self.0 *= &other.0;
    }
}

impl MulAssign<f64> for MyFloat {
    fn mul_assign(&mut self, other: f64) {
        self.0 *= other;
    }
}

impl DivAssign for MyFloat {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl DivAssign<&MyFloat> for MyFloat {
    fn div_assign(&mut self, other: &MyFloat) {
        self.0 /= &other.0;
    }
}

impl DivAssign<f64> for MyFloat {
    fn div_assign(&mut self, other: f64) {
        self.0 /= other;
    }
}

// Implement Rem (modulo) operations
impl Rem for MyFloat {
    type Output = MyFloat;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.0.is_zero() {
            panic!("Division by zero in float remainder operation");
        }

        let precision = self.0.prec().max(rhs.0.prec());
        let mut result = Float::with_val(precision, &self.0);
        result %= &rhs.0;
        MyFloat(result)
    }
}

impl Rem<&MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn rem(self, rhs: &MyFloat) -> Self::Output {
        if rhs.0.is_zero() {
            panic!("Division by zero in float remainder operation");
        }

        let precision = self.0.prec().max(rhs.0.prec());
        let mut result = Float::with_val(precision, &self.0);
        result %= &rhs.0;
        MyFloat(result)
    }
}

impl Rem<&MyFloat> for MyFloat {
    type Output = MyFloat;

    fn rem(self, rhs: &MyFloat) -> Self::Output {
        (&self).rem(rhs)
    }
}

impl Rem<MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn rem(self, rhs: MyFloat) -> Self::Output {
        self.rem(&rhs)
    }
}

impl Rem<f64> for MyFloat {
    type Output = MyFloat;

    fn rem(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            panic!("Division by zero in float remainder operation");
        }

        let precision = self.0.prec();
        let rhs_float = Float::with_val(precision, rhs);
        let rhs_wrapper = MyFloat(rhs_float);
        self % rhs_wrapper
    }
}

impl Rem<f64> for &MyFloat {
    type Output = MyFloat;

    fn rem(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            panic!("Division by zero in float remainder operation");
        }

        let precision = self.0.prec();
        let rhs_float = Float::with_val(precision, rhs);
        let rhs_wrapper = MyFloat(rhs_float);
        self % rhs_wrapper
    }
}

impl Rem<MyFloat> for f64 {
    type Output = MyFloat;

    fn rem(self, rhs: MyFloat) -> Self::Output {
        if rhs.0.is_zero() {
            panic!("Division by zero in float remainder operation");
        }

        let precision = rhs.0.prec();
        let mut result = Float::with_val(precision, &self);
        result %= &rhs.0;
        MyFloat(result)
    }
}

impl Rem<&MyFloat> for f64 {
    type Output = MyFloat;

    fn rem(self, rhs: &MyFloat) -> Self::Output {
        if rhs.0.is_zero() {
            panic!("Division by zero in float remainder operation");
        }

        let precision = rhs.0.prec();
        let mut result = Float::with_val(precision, &self);
        result %= &rhs.0;
        MyFloat(result)
    }
}

impl RemAssign for MyFloat {
    fn rem_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) % rhs;
    }
}

impl RemAssign<&MyFloat> for MyFloat {
    fn rem_assign(&mut self, rhs: &MyFloat) {
        let result = (&*self) % rhs;
        *self = result;
    }
}

impl RemAssign<f64> for MyFloat {
    fn rem_assign(&mut self, rhs: f64) {
        *self = std::mem::take(self) % rhs;
    }
}

// Implement Pow trait
impl Pow<MyFloat> for MyFloat {
    type Output = MyFloat;

    fn pow(self, exp: MyFloat) -> MyFloat {
        let mut temp = self.0;
        temp.pow_assign(&exp.0);
        MyFloat(temp)
    }
}

impl Pow<&MyFloat> for MyFloat {
    type Output = MyFloat;

    fn pow(self, exp: &MyFloat) -> MyFloat {
        let mut temp = self.0;
        temp.pow_assign(&exp.0);
        MyFloat(temp)
    }
}

impl Pow<MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn pow(self, exp: MyFloat) -> MyFloat {
        let mut temp = self.0.clone();
        temp.pow_assign(&exp.0);
        MyFloat(temp)
    }
}

impl Pow<&MyFloat> for &MyFloat {
    type Output = MyFloat;

    fn pow(self, exp: &MyFloat) -> MyFloat {
        let mut temp = self.0.clone();
        temp.pow_assign(&exp.0);
        MyFloat(temp)
    }
}

impl Pow<f64> for MyFloat {
    type Output = MyFloat;

    fn pow(self, exp: f64) -> MyFloat {
        let exp_float = MyFloat::new(exp);
        self.pow(exp_float)
    }
}

impl Pow<f64> for &MyFloat {
    type Output = MyFloat;

    fn pow(self, exp: f64) -> MyFloat {
        let exp_float = MyFloat::new(exp);
        self.pow(&exp_float)
    }
}

impl Pow<i32> for MyFloat {
    type Output = MyFloat;

    fn pow(self, exp: i32) -> MyFloat {
        let exp_float = MyFloat::new(exp as f64);
        self.pow(exp_float)
    }
}

impl Pow<i32> for &MyFloat {
    type Output = MyFloat;

    fn pow(self, exp: i32) -> MyFloat {
        let exp_float = MyFloat::new(exp as f64);
        self.pow(&exp_float)
    }
}

// Implement PowAssign trait
impl PowAssign<MyFloat> for MyFloat {
    fn pow_assign(&mut self, exp: MyFloat) {
        self.0.pow_assign(&exp.0);
    }
}

impl PowAssign<&MyFloat> for MyFloat {
    fn pow_assign(&mut self, exp: &MyFloat) {
        self.0.pow_assign(&exp.0);
    }
}

impl PowAssign<f64> for MyFloat {
    fn pow_assign(&mut self, exp: f64) {
        let exp_float = Float::with_val(Self::PRECISION, exp);
        self.0.pow_assign(&exp_float);
    }
}

impl PowAssign<i32> for MyFloat {
    fn pow_assign(&mut self, exp: i32) {
        let exp_float = Float::with_val(Self::PRECISION, exp);
        self.0.pow_assign(&exp_float);
    }
}

// Implement Zero trait
impl Zero for MyFloat {
    fn zero() -> Self {
        MyFloat::new(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

// Implement One trait
impl One for MyFloat {
    fn one() -> Self {
        MyFloat::new(1.0)
    }

    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

// Implement Num trait
impl Num for MyFloat {
    type FromStrRadixErr = rug::float::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let parsed = Float::parse_radix(str, radix as i32)?;
        let with_precision = Float::with_val(Self::PRECISION, parsed);
        Ok(MyFloat(with_precision))
    }
}

// Implement Clone
impl Clone for MyFloat {
    fn clone(&self) -> Self {
        MyFloat(self.0.clone())
    }
}

// Default implementation - creates zero float with default precision
impl Default for MyFloat {
    fn default() -> Self {
        MyFloat(Float::new(Self::PRECISION))
    }
}

// Implement Display for pretty printing
impl fmt::Display for MyFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Implement Debug
impl fmt::Debug for MyFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MyFloat({})", self.0)
    }
}

// Implement PartialEq for comparisons
impl PartialEq for MyFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialEq<f64> for MyFloat {
    fn eq(&self, other: &f64) -> bool {
        self.0 == *other
    }
}

// Implement PartialOrd for ordering
impl PartialOrd for MyFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl PartialOrd<f64> for MyFloat {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&MyFloat::new(*other))
    }
}

// Conversion from f64
impl From<f64> for MyFloat {
    fn from(value: f64) -> Self {
        MyFloat::new(value)
    }
}

// Conversion from Float
impl From<Float> for MyFloat {
    fn from(value: Float) -> Self {
        MyFloat(value)
    }
}

// Conversion from &Float
impl From<&Float> for MyFloat {
    fn from(value: &Float) -> Self {
        MyFloat(value.clone())
    }
}

// Conversion from i32
impl From<i32> for MyFloat {
    fn from(value: i32) -> Self {
        MyFloat::new(value as f64)
    }
}

// Conversion from u32
impl From<u32> for MyFloat {
    fn from(value: u32) -> Self {
        MyFloat::new(value as f64)
    }
}

// Conversion from i64
impl From<i64> for MyFloat {
    fn from(value: i64) -> Self {
        MyFloat::new(value as f64)
    }
}

// Conversion from u64
impl From<u64> for MyFloat {
    fn from(value: u64) -> Self {
        MyFloat::new(value as f64)
    }
}

// Conversion to f64
impl From<MyFloat> for f64 {
    fn from(value: MyFloat) -> f64 {
        value.to_f64()
    }
}

// Conversion to Float
impl From<MyFloat> for Float {
    fn from(value: MyFloat) -> Float {
        value.to_float()
    }
}

// Implement Sum trait for iterating and summing MyFloat values
impl Sum for MyFloat {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(MyFloat::zero(), |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a MyFloat> for MyFloat {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(MyFloat::zero(), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;
    use num_traits::{One, Zero};
    use rug::ops::Pow;

    #[test]
    fn test_creation() {
        let f1 = MyFloat::new(3.14159);
        assert!((f1.to_f64() - 3.14159).abs() < 1e-10);

        let f2 = MyFloat::zero();
        assert_eq!(f2.to_f64(), 0.0);

        let f3 = MyFloat::one();
        assert_eq!(f3.to_f64(), 1.0);

        let f4 = MyFloat::new(f64::consts::PI).to_degrees();
        assert_eq!(f4.to_f64(), 180.0);

        let f5 = MyFloat::new(180.0).to_radians();
        assert_eq!(f5.to_f64(), f64::consts::PI);
    }

    #[test]
    fn test_arithmetic() {
        let f1 = MyFloat::new(2.5);
        let f2 = MyFloat::new(1.5);

        let sum = &f1 + &f2;
        assert!((sum.to_f64() - 4.0).abs() < 1e-10);

        let diff = &f1 - &f2;
        assert!((diff.to_f64() - 1.0).abs() < 1e-10);

        let prod = &f1 * &f2;
        assert!((prod.to_f64() - 3.75).abs() < 1e-10);

        let quot = &f1 / &f2;
        assert!((quot.to_f64() - (5.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_arithmetic_with_f64() {
        let f1 = MyFloat::new(5.0);

        let sum = &f1 + 3.0;
        assert!((sum.to_f64() - 8.0).abs() < 1e-10);

        let sum2 = 3.0 + &f1;
        assert!((sum2.to_f64() - 8.0).abs() < 1e-10);

        let diff = &f1 - 2.0;
        assert!((diff.to_f64() - 3.0).abs() < 1e-10);

        let prod = &f1 * 2.0;
        assert!((prod.to_f64() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_transcendental_functions() {
        let f = MyFloat::new(2.0);

        let exp_result = f.exp();
        assert!((exp_result.to_f64() - std::f64::consts::E.powf(2.0)).abs() < 1e-10);

        let ln_result = f.ln();
        assert!((ln_result.to_f64() - 2.0_f64.ln()).abs() < 1e-10);

        let sqrt_result = f.sqrt();
        assert!((sqrt_result.to_f64() - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_trigonometric_functions() {
        let pi_half = MyFloat::new(std::f64::consts::PI / 2.0);

        let sin_result = pi_half.sin();
        assert!((sin_result.to_f64() - 1.0).abs() < 1e-10);

        let cos_result = pi_half.cos();
        assert!(cos_result.to_f64().abs() < 1e-10);

        let zero = MyFloat::zero();
        let atan_result = zero.atan();
        assert!(atan_result.to_f64().abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_functions() {
        let f = MyFloat::new(1.0);

        let sinh_result = f.sinh();
        assert!((sinh_result.to_f64() - 1.0_f64.sinh()).abs() < 1e-10);

        let cosh_result = f.cosh();
        assert!((cosh_result.to_f64() - 1.0_f64.cosh()).abs() < 1e-10);

        let tanh_result = f.tanh();
        assert!((tanh_result.to_f64() - 1.0_f64.tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_rounding_functions() {
        let f = MyFloat::new(3.7);

        let floor_result = f.floor();
        assert_eq!(floor_result.to_f64(), 3.0);

        let ceil_result = f.ceil();
        assert_eq!(ceil_result.to_f64(), 4.0);

        let round_result = f.round();
        assert_eq!(round_result.to_f64(), 4.0);

        let trunc_result = f.trunc();
        assert_eq!(trunc_result.to_f64(), 3.0);

        let fract_result = f.fract();
        assert!((fract_result.to_f64() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_assignment_operators() {
        let mut f1 = MyFloat::new(5.0);

        // Test AddAssign
        f1 += MyFloat::new(3.0);
        assert!((f1.to_f64() - 8.0).abs() < 1e-10);

        // Test SubAssign
        f1 -= MyFloat::new(2.0);
        assert!((f1.to_f64() - 6.0).abs() < 1e-10);

        // Test MulAssign
        f1 *= MyFloat::new(2.0);
        assert!((f1.to_f64() - 12.0).abs() < 1e-10);

        // Test DivAssign
        f1 /= MyFloat::new(3.0);
        assert!((f1.to_f64() - 4.0).abs() < 1e-10);

        // Test with f64
        f1 += 1.0;
        assert!((f1.to_f64() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pow_trait() {
        let f = MyFloat::new(2.0);
        let exp = MyFloat::new(3.0);

        let result = f.pow(exp);
        assert!((result.to_f64() - 8.0).abs() < 1e-10);

        // Test with f64 exponent
        let f2 = MyFloat::new(9.0);
        let result2 = f2.pow(0.5);
        assert!((result2.to_f64() - 3.0).abs() < 1e-10);

        // Test with i32 exponent
        let f3 = MyFloat::new(3.0);
        let result3 = f3.pow(4i32);
        assert!((result3.to_f64() - 81.0).abs() < 1e-10);
    }

    #[test]
    fn test_pow_assign_trait() {
        // Test PowAssign with MyFloat exponent
        let mut f1 = MyFloat::new(2.0);
        let exp1 = MyFloat::new(4.0);
        f1.pow_assign(exp1);
        assert!((f1.to_f64() - 16.0).abs() < 1e-10);

        // Test PowAssign with borrowed MyFloat exponent
        let mut f2 = MyFloat::new(3.0);
        let exp2 = MyFloat::new(3.0);
        f2.pow_assign(&exp2);
        assert!((f2.to_f64() - 27.0).abs() < 1e-10);

        // Test PowAssign with f64 exponent
        let mut f3 = MyFloat::new(25.0);
        f3.pow_assign(0.5);
        assert!((f3.to_f64() - 5.0).abs() < 1e-10);

        // Test PowAssign with i32 exponent
        let mut f4 = MyFloat::new(2.0);
        f4.pow_assign(10i32);
        assert!((f4.to_f64() - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_one_traits() {
        // Test Zero trait
        let zero = MyFloat::zero();
        assert_eq!(zero.to_f64(), 0.0);
        assert!(zero.is_zero());

        let non_zero = MyFloat::new(1.5);
        assert!(!non_zero.is_zero());

        // Test One trait
        let one = MyFloat::one();
        assert_eq!(one.to_f64(), 1.0);
        assert!(one.is_one());

        let non_one = MyFloat::new(2.0);
        assert!(!non_one.is_one());
    }

    #[test]
    fn test_nan_and_infinity() {
        let nan = MyFloat::nan();
        assert!(nan.is_nan());
        assert!(!nan.is_finite());
        assert!(!nan.is_normal());

        let inf = MyFloat::infinity();
        assert!(inf.is_infinite());
        assert!(!inf.is_finite());
        assert!(!inf.is_normal());

        let neg_inf = MyFloat::neg_infinity();
        assert!(neg_inf.is_infinite());
        assert!(neg_inf.is_sign_negative());

        let normal = MyFloat::new(3.14);
        assert!(!normal.is_nan());
        assert!(!normal.is_infinite());
        assert!(normal.is_finite());
        assert!(normal.is_normal());

        let zero = MyFloat::zero();
        assert!(!zero.is_nan());
        assert!(!zero.is_infinite());
        assert!(zero.is_finite());
        assert!(!zero.is_normal()); // Zero is not considered "normal"
    }

    #[test]
    fn test_sign_operations() {
        let positive = MyFloat::new(5.0);
        assert!(positive.is_sign_positive());
        assert!(!positive.is_sign_negative());

        let negative = MyFloat::new(-3.0);
        assert!(!negative.is_sign_positive());
        assert!(negative.is_sign_negative());

        let signum_pos = positive.signum();
        assert_eq!(signum_pos.to_f64(), 1.0);

        let signum_neg = negative.signum();
        assert_eq!(signum_neg.to_f64(), -1.0);

        let signum_zero = MyFloat::zero().signum();
        assert_eq!(signum_zero.to_f64(), 0.0);

        // Test copysign
        let magnitude = MyFloat::new(5.0);
        let with_neg_sign = magnitude.copysign(&negative);
        assert_eq!(with_neg_sign.to_f64(), -5.0);
    }

    #[test]
    fn test_min_max_clamp() {
        let a = MyFloat::new(3.0);
        let b = MyFloat::new(7.0);

        let min_result = a.min(&b);
        assert_eq!(min_result.to_f64(), 3.0);

        let max_result = a.max(&b);
        assert_eq!(max_result.to_f64(), 7.0);

        let value = MyFloat::new(10.0);
        let min_bound = MyFloat::new(2.0);
        let max_bound = MyFloat::new(8.0);
        let clamped = value.clamp(&min_bound, &max_bound);
        assert_eq!(clamped.to_f64(), 8.0);

        let value2 = MyFloat::new(1.0);
        let clamped2 = value2.clamp(&min_bound, &max_bound);
        assert_eq!(clamped2.to_f64(), 2.0);
    }

    #[test]
    fn test_num_trait() {
        // Test parsing from string
        let parsed = MyFloat::from_str_radix("42", 10).unwrap();
        assert_eq!(parsed.to_f64(), 42.0);

        let parsed_hex = MyFloat::from_str_radix("ff", 16).unwrap();
        assert_eq!(parsed_hex.to_f64(), 255.0);

        let parsed_float = MyFloat::from_str_radix("3.14159", 10).unwrap();
        assert!((parsed_float.to_f64() - 3.14159).abs() < 1e-5);
    }

    #[test]
    fn test_remainder_operations() {
        let a = MyFloat::new(7.0);
        let b = MyFloat::new(3.0);

        let result = &a % &b;
        assert!((result.to_f64() - 1.0).abs() < 1e-10);

        // Test with f64
        let result2 = &a % 2.5;
        assert!((result2.to_f64() - 2.0).abs() < 1e-10);

        // Test RemAssign
        let mut c = MyFloat::new(10.0);
        c %= MyFloat::new(3.0);
        assert!((c.to_f64() - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_remainder_division_by_zero() {
        let a = MyFloat::new(5.0);
        let zero = MyFloat::zero();
        let _ = a % zero;
    }

    #[test]
    fn test_partial_ord() {
        let a = MyFloat::new(3.0);
        let b = MyFloat::new(7.0);
        let c = MyFloat::new(3.0);

        assert!(a < b);
        assert!(b > a);
        assert!(a <= c);
        assert!(a >= c);
        assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Less));
        assert_eq!(a.partial_cmp(&c), Some(std::cmp::Ordering::Equal));
    }

    #[test]
    fn test_conversions() {
        // Test From implementations
        let from_f64 = MyFloat::from(3.14);
        assert!((from_f64.to_f64() - 3.14).abs() < 1e-10);

        let from_i32 = MyFloat::from(42i32);
        assert_eq!(from_i32.to_f64(), 42.0);

        let from_u32 = MyFloat::from(42u32);
        assert_eq!(from_u32.to_f64(), 42.0);

        let from_i64 = MyFloat::from(42i64);
        assert_eq!(from_i64.to_f64(), 42.0);

        let from_u64 = MyFloat::from(42u64);
        assert_eq!(from_u64.to_f64(), 42.0);

        // Test Into f64
        let my_float = MyFloat::new(2.718);
        let back_to_f64: f64 = my_float.into();
        assert!((back_to_f64 - 2.718).abs() < 1e-10);
    }

    #[test]
    fn test_display_and_debug() {
        let f = MyFloat::new(3.14159);
        let display_str = format!("{}", f);
        assert!(display_str.contains("3.1415899999999999"));

        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("MyFloat"));
        assert!(debug_str.contains("3.1415899999999999"));
    }

    #[test]
    fn test_default() {
        let default_float = MyFloat::default();
        assert!(default_float.0.is_zero());
    }

    #[test]
    fn test_abs() {
        let positive = MyFloat::new(5.0);
        assert_eq!(positive.abs().to_f64(), 5.0);

        let negative = MyFloat::new(-5.0);
        assert_eq!(negative.abs().to_f64(), 5.0);

        let zero = MyFloat::zero();
        assert_eq!(zero.abs().to_f64(), 0.0);
    }

    #[test]
    fn test_logarithms() {
        let f = MyFloat::new(100.0);

        let log10_result = f.log10();
        assert!((log10_result.to_f64() - 2.0).abs() < 1e-10);

        let f2 = MyFloat::new(8.0);
        let log2_result = f2.log2();
        assert!((log2_result.to_f64() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atan2() {
        let y = MyFloat::new(1.0);
        let x = MyFloat::new(1.0);

        let atan2_result = y.atan2(&x);
        assert!((atan2_result.to_f64() - std::f64::consts::PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_hyperbolic_functions() {
        let f = MyFloat::new(1.0);

        let asinh_result = f.asinh();
        assert!((asinh_result.to_f64() - 1.0_f64.asinh()).abs() < 1e-10);

        let f2 = MyFloat::new(2.0);
        let acosh_result = f2.acosh();
        assert!((acosh_result.to_f64() - 2.0_f64.acosh()).abs() < 1e-10);

        let f3 = MyFloat::new(0.5);
        let atanh_result = f3.atanh();
        assert!((atanh_result.to_f64() - 0.5_f64.atanh()).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_trig_functions() {
        let f = MyFloat::new(0.5);

        let asin_result = f.asin();
        assert!((asin_result.to_f64() - 0.5_f64.asin()).abs() < 1e-10);

        let acos_result = f.acos();
        assert!((acos_result.to_f64() - 0.5_f64.acos()).abs() < 1e-10);
    }
}
