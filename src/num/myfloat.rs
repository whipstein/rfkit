use crate::num::{MyComplex, MyUsize};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_complex::Complex64;
use num_traits::{ConstOne, ConstZero, Float, Inv, Num, One, Pow, Signed, ToPrimitive, Zero};
use std::{
    convert::From,
    f64, fmt,
    fmt::Debug,
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign,
        Sub, SubAssign,
    },
};
use twofloat::{TwoFloat, TwoFloatError};

/// Defines a multiplicative identity element for `Self`.
///
/// # Laws
///
/// ```text
/// a * -1 = -a       ∀ a ∈ Self
/// -1 * a = -a       ∀ a ∈ Self
/// ```
pub trait NegOne: Sized + Mul<Self, Output = Self> {
    /// Returns the multiplicative identity element of `Self`, `-1`.
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    // This cannot be an associated constant, because of bignums.
    fn neg_one() -> Self;

    /// Sets `self` to the multiplicative identity element of `Self`, `1`.
    fn set_neg_one(&mut self) {
        *self = NegOne::neg_one();
    }

    /// Returns `true` if `self` is equal to the multiplicative identity.
    ///
    /// For performance reasons, it's best to implement this manually.
    /// After a semver bump, this method will be required, and the
    /// `where Self: PartialEq` bound will be removed.
    #[inline]
    fn is_neg_one(&self) -> bool
    where
        Self: PartialEq,
    {
        *self == Self::neg_one()
    }
}

/// A float wrapper with fixed precision of 53 bits
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct MyFloat(TwoFloat);

impl MyFloat {
    /// Smallest finite `TwoFloat` value.
    pub const MIN: Self = Self(TwoFloat::MIN);

    /// Smallest positive normal `TwoFloat` value.
    pub const MIN_POSITIVE: Self = Self(TwoFloat::MIN_POSITIVE);

    /// Largest finite `TwoFloat` value.
    pub const MAX: Self = Self(TwoFloat::MAX);

    /// Represents an error value equivalent to `f64::NAN`.
    pub const NAN: Self = Self(TwoFloat::NAN);

    /// Represents the difference between 1.0 and the next representable normal value.
    pub const EPSILON: Self = Self(TwoFloat::EPSILON);

    /// A positive infinite value
    pub const INFINITY: Self = Self(TwoFloat::INFINITY);

    /// A negative infinite value
    pub const NEG_INFINITY: Self = Self(TwoFloat::NEG_INFINITY);

    /// A constant 0.
    pub const ZERO: Self = Self(TwoFloat::ZERO);

    /// A constant 1.
    pub const ONE: Self = Self(TwoFloat::ONE);

    /// Create a new float from an f64 value
    pub fn new(value: f64) -> Self {
        MyFloat(TwoFloat::from_f64(value))
    }

    /// Create a new float from an Float value
    pub fn from_float(value: TwoFloat) -> Self {
        MyFloat(value)
    }

    /// Get the value as f64
    pub fn to_f64(&self) -> f64 {
        self.0.into()
    }

    /// Get the value as f64
    pub fn to_float(&self) -> TwoFloat {
        self.0
    }

    /// Get the magnitude in dB
    pub fn db(&self) -> MyFloat {
        MyFloat(20.0 * self.0.abs().log10())
    }

    /// Get the magnitude in dB
    pub fn db10(&self) -> MyFloat {
        MyFloat(10.0 * self.0.abs().log10())
    }

    /// Get the magnitude in dB
    pub fn db_to_mag(&self) -> MyFloat {
        MyFloat(TwoFloat::from_f64(10.0).pow(self.0 / 20.0))
    }

    /// Get the magnitude in dB
    pub fn db10_to_mag(&self) -> MyFloat {
        MyFloat(TwoFloat::from_f64(10.0).pow(self.0 / 10.0))
    }

    /// Get the square
    pub fn square(&self) -> Self {
        self * self
    }

    /// Raise to a power
    pub fn pow(&self, exp: MyFloat) -> Self {
        MyFloat(self.0.pow(exp.0))
    }

    /// Access the inner rug::Float (for advanced operations)
    pub fn inner(&self) -> &TwoFloat {
        &self.0
    }

    /// Convert to inner rug::Float (consuming self)
    pub fn into_inner(self) -> TwoFloat {
        self.0
    }

    pub fn to_bits(self) -> MyUsize {
        MyUsize {
            hi: self.0.hi().to_bits(),
            lo: self.0.lo().to_bits(),
        }
    }
}

impl Deref for MyFloat {
    type Target = TwoFloat;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MyFloat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl fmt::Display for MyFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.hi())
    }
}

// impl ConstZero for MyFloat {
//     const ZERO: Self = Self::ZERO;
// }

// impl ConstOne for MyFloat {
//     const ONE: Self = Self::ONE;
// }

// impl ToPrimitive for MyFloat {
//     #[inline]
//     fn to_isize(&self) -> Option<isize> {
//         (self.0).to_isize()
//     }

//     #[inline]
//     fn to_i8(&self) -> Option<i8> {
//         (self.0).to_i8()
//     }

//     #[inline]
//     fn to_i16(&self) -> Option<i16> {
//         (self.0).to_i16()
//     }

//     #[inline]
//     fn to_i32(&self) -> Option<i32> {
//         (self.0).to_i32()
//     }

//     #[inline]
//     fn to_i64(&self) -> Option<i64> {
//         (self.0).to_i64()
//     }

//     #[inline]
//     fn to_i128(&self) -> Option<i128> {
//         (self.0).to_i128()
//     }

//     #[inline]
//     fn to_usize(&self) -> Option<usize> {
//         (self.0).to_usize()
//     }

//     #[inline]
//     fn to_u8(&self) -> Option<u8> {
//         (self.0).to_u8()
//     }

//     #[inline]
//     fn to_u16(&self) -> Option<u16> {
//         (self.0).to_u16()
//     }

//     #[inline]
//     fn to_u32(&self) -> Option<u32> {
//         (self.0).to_u32()
//     }

//     #[inline]
//     fn to_u64(&self) -> Option<u64> {
//         (self.0).to_u64()
//     }

//     #[inline]
//     fn to_u128(&self) -> Option<u128> {
//         (self.0).to_u128()
//     }

//     #[inline]
//     fn to_f32(&self) -> Option<f32> {
//         (self.0).to_f32()
//     }

//     #[inline]
//     fn to_f64(&self) -> Option<f64> {
//         (self.0).to_f64()
//     }
// }

// impl NumCast for MyFloat {
//     fn from<U: ToPrimitive>(n: U) -> Option<Self> {
//         match <TwoFloat as NumCast>::from(n) {
//             Some(x) => Some(MyFloat(x)),
//             None => None,
//         }
//     }
// }

// impl Float for MyFloat {
//     #[inline]
//     fn nan() -> Self {
//         MyFloat::NAN
//     }

//     #[inline]
//     fn infinity() -> Self {
//         MyFloat::INFINITY
//     }

//     #[inline]
//     fn neg_infinity() -> Self {
//         MyFloat::NEG_INFINITY
//     }

//     #[inline]
//     fn neg_zero() -> Self {
//         MyFloat::new(-0.0)
//     }

//     #[inline]
//     fn min_value() -> Self {
//         MyFloat::MIN
//     }

//     #[inline]
//     fn min_positive_value() -> Self {
//         MyFloat::MIN_POSITIVE
//     }

//     #[inline]
//     fn epsilon() -> Self {
//         MyFloat::EPSILON
//     }

//     #[inline]
//     fn max_value() -> Self {
//         MyFloat::MAX
//     }

//     #[inline]
//     #[allow(deprecated)]
//     fn abs_sub(self, other: Self) -> Self {
//         MyFloat(self.0.abs_sub(other.0))
//     }

//     #[inline]
//     fn integer_decode(self) -> (u64, i16, i8) {
//         let bits: MyUsize = self.to_bits();
//         let sign: i8 = if bits.hi >> 63 == 0 { 1 } else { -1 };
//         let mut exponent: i16 = ((bits.hi >> 52) & 0x7ff) as i16;
//         let mantissa = if exponent == 0 {
//             (bits.hi & 0xfffffffffffff) << 1
//         } else {
//             (bits.hi & 0xfffffffffffff) | 0x10000000000000
//         };
//         // Exponent bias + mantissa shift
//         exponent -= 1023 + 52;
//         (mantissa, exponent, sign)
//     }

//     #[inline]
//     fn is_nan(self) -> bool {
//         self.0 != self.0 // NaN is the only value that doesn't equal itself
//     }

//     #[inline]
//     fn is_infinite(self) -> bool {
//         if self.0 == TwoFloat::INFINITY || self.0 == TwoFloat::NEG_INFINITY {
//             true
//         } else {
//             false
//         }
//     }

//     #[inline]
//     fn is_finite(self) -> bool {
//         !self.is_nan() && !self.is_infinite()
//     }

//     #[inline]
//     fn is_normal(self) -> bool {
//         !self.0.is_zero() && !self.is_nan() && !self.is_infinite()
//     }

//     #[inline]
//     fn is_subnormal(self) -> bool {
//         self.0.is_subnormal()
//     }

//     #[inline]
//     fn classify(self) -> std::num::FpCategory {
//         self.0.classify()
//     }

//     #[inline]
//     fn clamp(self, min: Self, max: Self) -> Self {
//         self.max(min).min(max)
//     }

//     #[inline]
//     fn floor(self) -> Self {
//         MyFloat(self.0.floor())
//     }

//     #[inline]
//     fn ceil(self) -> Self {
//         MyFloat(self.0.ceil())
//     }

//     #[inline]
//     fn round(self) -> Self {
//         MyFloat(self.0.round())
//     }

//     #[inline]
//     fn trunc(self) -> Self {
//         MyFloat(self.0.trunc())
//     }

//     #[inline]
//     fn fract(self) -> Self {
//         MyFloat(self.0.fract())
//     }

//     #[inline]
//     fn abs(self) -> Self {
//         MyFloat(self.0.abs())
//     }

//     #[inline]
//     fn signum(self) -> Self {
//         if self.0.is_zero() {
//             MyFloat::zero()
//         } else {
//             MyFloat(self.0.signum())
//         }
//     }

//     #[inline]
//     fn is_sign_positive(self) -> bool {
//         self.0.is_sign_positive()
//     }

//     #[inline]
//     fn is_sign_negative(self) -> bool {
//         self.0.is_sign_negative()
//     }

//     #[inline]
//     fn mul_add(self, a: Self, b: Self) -> Self {
//         MyFloat(self.0.mul_add(a.0, b.0))
//     }

//     #[inline]
//     fn recip(self) -> Self {
//         MyFloat(self.0.recip())
//     }

//     #[inline]
//     fn powi(self, n: i32) -> Self {
//         MyFloat(self.0.powi(n))
//     }

//     #[inline]
//     fn powf(self, n: Self) -> Self {
//         MyFloat(self.0.powf(n.0))
//     }

//     #[inline]
//     fn sqrt(self) -> Self {
//         MyFloat(self.0.sqrt())
//     }

//     #[inline]
//     fn exp(self) -> Self {
//         MyFloat(self.0.exp())
//     }

//     #[inline]
//     fn exp2(self) -> Self {
//         MyFloat(self.0.exp2())
//     }

//     #[inline]
//     fn ln(self) -> Self {
//         MyFloat(self.0.ln())
//     }

//     #[inline]
//     fn log(self, base: Self) -> Self {
//         MyFloat(self.0.log(base.0))
//     }

//     #[inline]
//     fn log2(self) -> Self {
//         MyFloat(self.0.log2())
//     }

//     #[inline]
//     fn log10(self) -> Self {
//         MyFloat(self.0.log10())
//     }

//     #[inline]
//     fn to_degrees(self) -> Self {
//         MyFloat(self.0.to_degrees())
//     }

//     #[inline]
//     fn to_radians(self) -> Self {
//         MyFloat(self.0.to_radians())
//     }

//     #[inline]
//     fn max(self, other: Self) -> Self {
//         MyFloat(self.0.max(other.0))
//     }

//     #[inline]
//     fn min(self, other: Self) -> Self {
//         MyFloat(self.0.min(other.0))
//     }

//     #[inline]
//     fn cbrt(self) -> Self {
//         MyFloat(self.0.cbrt())
//     }

//     #[inline]
//     fn hypot(self, other: Self) -> Self {
//         MyFloat(self.0.hypot(other.0))
//     }

//     #[inline]
//     fn sin(self) -> Self {
//         MyFloat(self.0.sin())
//     }

//     #[inline]
//     fn cos(self) -> Self {
//         MyFloat(self.0.cos())
//     }

//     #[inline]
//     fn tan(self) -> Self {
//         MyFloat(self.0.tan())
//     }

//     #[inline]
//     fn asin(self) -> Self {
//         MyFloat(self.0.asin())
//     }

//     #[inline]
//     fn acos(self) -> Self {
//         MyFloat(self.0.acos())
//     }

//     #[inline]
//     fn atan(self) -> Self {
//         MyFloat(self.0.atan())
//     }

//     #[inline]
//     fn atan2(self, other: Self) -> Self {
//         MyFloat(self.0.atan2(other.0))
//     }

//     #[inline]
//     fn sin_cos(self) -> (Self, Self) {
//         let out = self.0.sin_cos();
//         (MyFloat(out.0), MyFloat(out.1))
//     }

//     #[inline]
//     fn exp_m1(self) -> Self {
//         MyFloat(self.0.exp_m1())
//     }

//     #[inline]
//     fn ln_1p(self) -> Self {
//         MyFloat(self.0.ln_1p())
//     }

//     #[inline]
//     fn sinh(self) -> Self {
//         MyFloat(self.0.sinh())
//     }

//     #[inline]
//     fn cosh(self) -> Self {
//         MyFloat(self.0.cosh())
//     }

//     #[inline]
//     fn tanh(self) -> Self {
//         MyFloat(self.0.tanh())
//     }

//     #[inline]
//     fn asinh(self) -> Self {
//         MyFloat(self.0.asinh())
//     }

//     #[inline]
//     fn acosh(self) -> Self {
//         MyFloat(self.0.acosh())
//     }

//     #[inline]
//     fn atanh(self) -> Self {
//         MyFloat(self.0.atanh())
//     }

//     #[inline]
//     fn copysign(self, sign: Self) -> Self {
//         MyFloat(self.0.copysign(sign.0))
//     }
// }

// // Implement basic arithmetic operations
// macro_rules! impl_self_math_op(
//     ($trt:ident, $operator:tt, $mth:ident) => (
//         impl $trt for MyFloat {
//             type Output = Self;

//             fn $mth(self, other: Self) -> Self::Output {
//                 MyFloat(self.0 $operator other.0)
//             }
//         }

//         impl $trt<&MyFloat> for MyFloat {
//             type Output = MyFloat;

//             fn $mth(self, other: &MyFloat) -> Self::Output {
//                 MyFloat(self.0 $operator &other.0)
//             }
//         }

//         impl $trt<MyFloat> for &MyFloat {
//             type Output = MyFloat;

//             fn $mth(self, other: MyFloat) -> Self::Output {
//                 MyFloat(&self.0 $operator other.0)
//             }
//         }

//         impl $trt<&MyFloat> for &MyFloat {
//             type Output = MyFloat;

//             fn $mth(self, other: &MyFloat) -> Self::Output {
//                 MyFloat(&self.0 $operator other.0.clone())
//             }
//         }
//     );
// );

// macro_rules! impl_math_op(
//     ($trt:ident, $operator:tt, $mth:ident, Complex64) => (
//         impl $trt<Complex64> for MyFloat {
//             type Output = MyComplex;

//             fn $mth(self, other: Complex64) -> Self::Output {
//                 MyComplex::from_real(self) $operator MyComplex::from_c64(other)
//             }
//         }

//         impl $trt<Complex64> for &MyFloat {
//             type Output = MyComplex;

//             fn $mth(self, other: Complex64) -> Self::Output {
//                 MyComplex::from_real(self.clone()) $operator MyComplex::from_c64(other)
//             }
//         }

//         impl $trt<&Complex64> for MyFloat {
//             type Output = MyComplex;

//             fn $mth(self, other: &Complex64) -> Self::Output {
//                 MyComplex::from_real(self) $operator MyComplex::from_c64(*other)
//             }
//         }

//         impl $trt<&Complex64> for &MyFloat {
//             type Output = MyComplex;

//             fn $mth(self, other: &Complex64) -> Self::Output {
//                 MyComplex::from_real(self.clone()) $operator MyComplex::from_c64(*other)
//             }
//         }

//         impl $trt<MyFloat> for Complex64 {
//             type Output = MyComplex;

//             fn $mth(self, other: MyFloat) -> Self::Output {
//                 MyComplex::from_c64(self) $operator MyComplex::from_real(other)
//             }
//         }

//         impl $trt<MyFloat> for &Complex64 {
//             type Output = MyComplex;

//             fn $mth(self, other: MyFloat) -> Self::Output {
//                 MyComplex::from_c64(*self) $operator MyComplex::from_real(other)
//             }
//         }

//         impl $trt<&MyFloat> for Complex64 {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyFloat) -> Self::Output {
//                 MyComplex::from_c64(self) $operator MyComplex::from_real(other.clone())
//             }
//         }

//         impl $trt<&MyFloat> for &Complex64 {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyFloat) -> Self::Output {
//                 MyComplex::from_c64(*self) $operator MyComplex::from_real(other.clone())
//             }
//         }
//     );
//     ($trt:ident, $operator:tt, $mth:ident, $rhs:ident) => (
//         impl $trt<$rhs> for MyFloat {
//             type Output = MyFloat;

//             fn $mth(self, other: $rhs) -> Self::Output {
//                 MyFloat(self.0 $operator other)
//             }
//         }

//         impl $trt<$rhs> for &MyFloat {
//             type Output = MyFloat;

//             fn $mth(self, other: $rhs) -> Self::Output {
//                 MyFloat(self.0.clone() $operator other)
//             }
//         }

//         impl $trt<&$rhs> for MyFloat {
//             type Output = MyFloat;

//             fn $mth(self, other: &$rhs) -> Self::Output {
//                 MyFloat(self.0 $operator *other)
//             }
//         }

//         impl $trt<&$rhs> for &MyFloat {
//             type Output = MyFloat;

//             fn $mth(self, other: &$rhs) -> Self::Output {
//                 MyFloat(self.0.clone() $operator *other)
//             }
//         }

//         impl $trt<MyFloat> for $rhs {
//             type Output = MyFloat;

//             fn $mth(self, other: MyFloat) -> Self::Output {
//                 MyFloat(self $operator other.0)
//             }
//         }

//         impl $trt<MyFloat> for &$rhs {
//             type Output = MyFloat;

//             fn $mth(self, other: MyFloat) -> Self::Output {
//                 MyFloat(self $operator other.0.clone())
//             }
//         }

//         impl $trt<&MyFloat> for $rhs {
//             type Output = MyFloat;

//             fn $mth(self, other: &MyFloat) -> Self::Output {
//                 MyFloat(self $operator other.0.clone())
//             }
//         }

//         impl $trt<&MyFloat> for &$rhs {
//             type Output = MyFloat;

//             fn $mth(self, other: &MyFloat) -> Self::Output {
//                 MyFloat(*self $operator other.0.clone())
//             }
//         }
//     );
// );

// // Implement assignment operators
// macro_rules! impl_self_assign_math_op(
//     ($trt:ident, $operator:tt, $mth:ident) => (
//         impl $trt for MyFloat {
//             fn $mth(&mut self, other: Self) {
//                 self.0 $operator other.0;
//             }
//         }

//         impl $trt<&MyFloat> for MyFloat {
//             fn $mth(&mut self, other: &MyFloat) {
//                 self.0 $operator &other.0;
//             }
//         }
//     );
// );

// macro_rules! impl_assign_math_op(
//     ($trt:ident, $operator:tt, $mth:ident, $rhs:ident) => (
//         impl $trt<$rhs> for MyFloat {
//             fn $mth(&mut self, other: $rhs) {
//                 self.0 $operator other;
//             }
//         }

//         impl $trt<&$rhs> for MyFloat {
//             fn $mth(&mut self, other: &$rhs) {
//                 self.0 $operator *other;
//             }
//         }
//     );
// );

// impl_self_math_op!(Add, +, add);
// impl_self_math_op!(Sub, -, sub);
// impl_self_math_op!(Mul, *, mul);
// impl_self_math_op!(Div, /, div);
// impl_math_op!(Add, +, add, f64);
// impl_math_op!(Sub, -, sub, f64);
// impl_math_op!(Mul, *, mul, f64);
// impl_math_op!(Div, /, div, f64);
// impl_math_op!(Add, +, add, Complex64);
// impl_math_op!(Sub, -, sub, Complex64);
// impl_math_op!(Mul, *, mul, Complex64);
// impl_math_op!(Div, /, div, Complex64);
// impl_self_assign_math_op!(AddAssign, +=, add_assign);
// impl_self_assign_math_op!(SubAssign, -=, sub_assign);
// impl_self_assign_math_op!(MulAssign, *=, mul_assign);
// impl_self_assign_math_op!(DivAssign, /=, div_assign);
// impl_assign_math_op!(AddAssign, +=, add_assign, f64);
// impl_assign_math_op!(SubAssign, -=, sub_assign, f64);
// impl_assign_math_op!(MulAssign, *=, mul_assign, f64);
// impl_assign_math_op!(DivAssign, /=, div_assign, f64);

// impl Neg for MyFloat {
//     type Output = Self;

//     fn neg(self) -> Self {
//         MyFloat(-self.0)
//     }
// }

// impl Neg for &MyFloat {
//     type Output = MyFloat;

//     fn neg(self) -> MyFloat {
//         MyFloat(-self.0.clone())
//     }
// }

// impl Inv for MyFloat {
//     type Output = MyFloat;

//     fn inv(self) -> Self::Output {
//         MyFloat(self.0.inv())
//     }
// }

// impl Inv for &MyFloat {
//     type Output = MyFloat;

//     fn inv(self) -> Self::Output {
//         MyFloat(self.0.inv())
//     }
// }

// // Implement Rem (modulo) operations
// impl Rem for MyFloat {
//     type Output = MyFloat;

//     fn rem(self, rhs: Self) -> Self::Output {
//         if rhs.0.is_zero() {
//             panic!("Division by zero in float remainder operation");
//         }

//         MyFloat(self.0 % rhs.0)
//     }
// }

// impl Rem<&MyFloat> for &MyFloat {
//     type Output = MyFloat;

//     fn rem(self, rhs: &MyFloat) -> Self::Output {
//         if rhs.0.is_zero() {
//             panic!("Division by zero in float remainder operation");
//         }

//         MyFloat(self.0 % rhs.0)
//     }
// }

// impl Rem<&MyFloat> for MyFloat {
//     type Output = MyFloat;

//     fn rem(self, rhs: &MyFloat) -> Self::Output {
//         (&self).rem(rhs)
//     }
// }

// impl Rem<MyFloat> for &MyFloat {
//     type Output = MyFloat;

//     fn rem(self, rhs: MyFloat) -> Self::Output {
//         self.rem(&rhs)
//     }
// }

// impl Rem<f64> for MyFloat {
//     type Output = MyFloat;

//     fn rem(self, rhs: f64) -> Self::Output {
//         if rhs == 0.0 {
//             panic!("Division by zero in float remainder operation");
//         }

//         let rhs_float = TwoFloat::from_f64(rhs);
//         let rhs_wrapper = MyFloat(rhs_float);
//         self % rhs_wrapper
//     }
// }

// impl Rem<f64> for &MyFloat {
//     type Output = MyFloat;

//     fn rem(self, rhs: f64) -> Self::Output {
//         if rhs == 0.0 {
//             panic!("Division by zero in float remainder operation");
//         }

//         let rhs_float = TwoFloat::from_f64(rhs);
//         let rhs_wrapper = MyFloat(rhs_float);
//         self % rhs_wrapper
//     }
// }

// impl Rem<MyFloat> for f64 {
//     type Output = MyFloat;

//     fn rem(self, rhs: MyFloat) -> Self::Output {
//         if rhs.0.is_zero() {
//             panic!("Division by zero in float remainder operation");
//         }

//         MyFloat(self % rhs.0)
//     }
// }

// impl Rem<&MyFloat> for f64 {
//     type Output = MyFloat;

//     fn rem(self, rhs: &MyFloat) -> Self::Output {
//         if rhs.0.is_zero() {
//             panic!("Division by zero in float remainder operation");
//         }

//         MyFloat(self % rhs.0)
//     }
// }

// impl RemAssign for MyFloat {
//     fn rem_assign(&mut self, rhs: Self) {
//         *self = std::mem::take(self) % rhs;
//     }
// }

// impl RemAssign<&MyFloat> for MyFloat {
//     fn rem_assign(&mut self, rhs: &MyFloat) {
//         let result = (&*self) % rhs;
//         *self = result;
//     }
// }

// impl RemAssign<f64> for MyFloat {
//     fn rem_assign(&mut self, rhs: f64) {
//         *self = std::mem::take(self) % rhs;
//     }
// }

// // Implement Pow trait
// impl Pow<MyFloat> for MyFloat {
//     type Output = MyFloat;

//     fn pow(self, exp: MyFloat) -> MyFloat {
//         MyFloat(self.0.pow(exp.0))
//     }
// }

// impl Pow<&MyFloat> for MyFloat {
//     type Output = MyFloat;

//     fn pow(self, exp: &MyFloat) -> MyFloat {
//         MyFloat(self.0.pow(&exp.0))
//     }
// }

// impl Pow<MyFloat> for &MyFloat {
//     type Output = MyFloat;

//     fn pow(self, exp: MyFloat) -> MyFloat {
//         MyFloat(self.0.pow(&exp.0))
//     }
// }

// impl Pow<&MyFloat> for &MyFloat {
//     type Output = MyFloat;

//     fn pow(self, exp: &MyFloat) -> MyFloat {
//         MyFloat(self.0.pow(&exp.0))
//     }
// }

// impl Pow<f64> for MyFloat {
//     type Output = MyFloat;

//     fn pow(self, exp: f64) -> MyFloat {
//         let exp_float = MyFloat::new(exp);
//         self.pow(exp_float)
//     }
// }

// impl Pow<f64> for &MyFloat {
//     type Output = MyFloat;

//     fn pow(self, exp: f64) -> MyFloat {
//         let exp_float = MyFloat::new(exp);
//         self.pow(exp_float)
//     }
// }

// impl Pow<i32> for MyFloat {
//     type Output = MyFloat;

//     fn pow(self, exp: i32) -> MyFloat {
//         let exp_float = MyFloat::new(exp as f64);
//         self.pow(exp_float)
//     }
// }

// impl Pow<i32> for &MyFloat {
//     type Output = MyFloat;

//     fn pow(self, exp: i32) -> MyFloat {
//         let exp_float = MyFloat::new(exp as f64);
//         self.pow(exp_float)
//     }
// }

// // Implement Num trait
// impl Zero for MyFloat {
//     fn zero() -> Self {
//         MyFloat::new(0.0)
//     }

//     fn is_zero(&self) -> bool {
//         self.0.is_zero()
//     }
// }

// impl One for MyFloat {
//     fn one() -> Self {
//         MyFloat::new(1.0)
//     }

//     fn is_one(&self) -> bool {
//         *self == Self::one()
//     }
// }

// impl NegOne for MyFloat {
//     fn neg_one() -> Self {
//         MyFloat::new(-1.0)
//     }

//     fn is_neg_one(&self) -> bool {
//         *self == Self::neg_one()
//     }
// }

// impl Num for MyFloat {
//     type FromStrRadixErr = TwoFloatError;

//     fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
//         Err(TwoFloatError::ParseError)
//     }
// }

// impl Signed for MyFloat {
//     fn abs(&self) -> Self {
//         MyFloat(self.0.abs())
//     }

//     fn abs_sub(&self, other: &Self) -> Self {
//         let diff = self.0 - other.0;
//         if diff.is_sign_positive() {
//             MyFloat(diff)
//         } else {
//             MyFloat::zero()
//         }
//     }

//     fn signum(&self) -> Self {
//         if self.0.is_zero() {
//             MyFloat::zero()
//         } else {
//             MyFloat(self.0.signum())
//         }
//     }

//     fn is_positive(&self) -> bool {
//         !self.0.is_zero() && self.0.is_sign_positive()
//     }

//     fn is_negative(&self) -> bool {
//         !self.0.is_zero() && self.0.is_sign_negative()
//     }
// }

// // Implement Traits
// impl Clone for MyFloat {
//     fn clone(&self) -> Self {
//         MyFloat(self.0.clone())
//     }
// }

// impl Default for MyFloat {
//     fn default() -> Self {
//         MyFloat::ZERO
//     }
// }

// impl fmt::Display for MyFloat {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.0.hi())
//     }
// }

// // impl fmt::Debug for MyFloat {
// //     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
// //         write!(f, "MyFloat({})", self.0)
// //         // write!(f, "{}", self.to_f64())
// //     }
// // }

// // Implement PartialEq for comparisons
// impl PartialEq for MyFloat {
//     fn eq(&self, other: &Self) -> bool {
//         self.0 == other.0
//     }
// }

// impl PartialEq<f64> for MyFloat {
//     fn eq(&self, other: &f64) -> bool {
//         self.0 == *other
//     }
// }

// // Implement PartialOrd for ordering
// impl PartialOrd for MyFloat {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         self.0.partial_cmp(&other.0)
//     }
// }

// impl PartialOrd<f64> for MyFloat {
//     fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
//         self.partial_cmp(&MyFloat::new(*other))
//     }
// }

// // Implement Conversion
// impl From<f64> for MyFloat {
//     fn from(value: f64) -> Self {
//         MyFloat::new(value)
//     }
// }

// impl From<&f64> for MyFloat {
//     fn from(value: &f64) -> Self {
//         MyFloat::new(*value)
//     }
// }

// impl From<TwoFloat> for MyFloat {
//     fn from(value: TwoFloat) -> Self {
//         MyFloat(value)
//     }
// }

// impl From<&TwoFloat> for MyFloat {
//     fn from(value: &TwoFloat) -> Self {
//         MyFloat(value.clone())
//     }
// }

// impl From<MyComplex> for MyFloat {
//     fn from(value: MyComplex) -> Self {
//         value.re()
//     }
// }

// impl From<&MyComplex> for MyFloat {
//     fn from(value: &MyComplex) -> Self {
//         value.re()
//     }
// }

// impl From<i32> for MyFloat {
//     fn from(value: i32) -> Self {
//         MyFloat::new(value as f64)
//     }
// }

// impl From<u32> for MyFloat {
//     fn from(value: u32) -> Self {
//         MyFloat::new(value as f64)
//     }
// }

// impl From<i64> for MyFloat {
//     fn from(value: i64) -> Self {
//         MyFloat::new(value as f64)
//     }
// }

// impl From<u64> for MyFloat {
//     fn from(value: u64) -> Self {
//         MyFloat::new(value as f64)
//     }
// }

// impl From<MyFloat> for f64 {
//     fn from(value: MyFloat) -> f64 {
//         value.to_f64()
//     }
// }

// impl From<&MyFloat> for f64 {
//     fn from(value: &MyFloat) -> f64 {
//         value.to_f64()
//     }
// }

// impl From<MyFloat> for TwoFloat {
//     fn from(value: MyFloat) -> TwoFloat {
//         value.to_float()
//     }
// }

// impl From<MyFloat> for Complex64 {
//     fn from(value: MyFloat) -> Complex64 {
//         Complex64::new(value.to_f64(), 0.0)
//     }
// }

// impl From<&MyFloat> for Complex64 {
//     fn from(value: &MyFloat) -> Complex64 {
//         Complex64::new(value.to_f64(), 0.0)
//     }
// }

// // Implement Sum trait for iterating and summing MyFloat values
// impl Sum for MyFloat {
//     fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
//         iter.fold(MyFloat::zero(), |acc, x| acc + x)
//     }
// }

// impl<'a> Sum<&'a MyFloat> for MyFloat {
//     fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
//         iter.fold(MyFloat::zero(), |acc, x| acc + x)
//     }
// }

// // Implement Product trait for iterating and summing MyFloat values
// impl Product for MyFloat {
//     fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
//         iter.fold(MyFloat::zero(), |acc, x| acc * x)
//     }
// }

// impl<'a> Product<&'a MyFloat> for MyFloat {
//     fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
//         iter.fold(MyFloat::zero(), |acc, x| acc * x)
//     }
// }

// // Implement Not trait (logical NOT: 0.0 -> 1.0, non-zero -> 0.0)
// impl Not for MyFloat {
//     type Output = Self;

//     fn not(self) -> Self::Output {
//         if self.0.is_zero() {
//             MyFloat::one()
//         } else {
//             MyFloat::zero()
//         }
//     }
// }

// impl Not for &MyFloat {
//     type Output = MyFloat;

//     fn not(self) -> Self::Output {
//         if self.0.is_zero() {
//             MyFloat::one()
//         } else {
//             MyFloat::zero()
//         }
//     }
// }

impl AbsDiffEq for MyFloat {
    type Epsilon = MyFloat;

    fn default_epsilon() -> Self::Epsilon {
        MyFloat::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        <MyFloat as Float>::abs(self - other) <= epsilon
    }
}

impl RelativeEq for MyFloat {
    fn default_max_relative() -> Self::Epsilon {
        MyFloat::EPSILON
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        // Handle same infinities
        if self == other {
            return true;
        }

        // Handle remaining infinities
        if Self::is_infinite(*self) || Self::is_infinite(*other) {
            return false;
        }

        let abs_diff = <Self as Float>::abs(self - other);

        // For when the numbers are really close together
        if abs_diff <= epsilon {
            return true;
        }

        let abs_self = <Self as Float>::abs(*self);
        let abs_other = <Self as Float>::abs(*other);

        let largest = if abs_other > abs_self {
            abs_other
        } else {
            abs_self
        };

        // Use a relative difference comparison
        abs_diff <= largest * max_relative
    }
}

impl UlpsEq for MyFloat {
    fn default_max_ulps() -> u32 {
        4
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        // For when the numbers are really close together
        if Self::abs_diff_eq(self, other, epsilon) {
            return true;
        }

        // Trivial negative sign check
        if self.signum() != other.signum() {
            return false;
        }

        // ULPS difference comparison
        let int_self: MyUsize = MyUsize {
            hi: self.0.hi().to_bits(),
            lo: self.0.lo().to_bits(),
        };
        let int_other: MyUsize = MyUsize {
            hi: other.0.hi().to_bits(),
            lo: other.0.lo().to_bits(),
        };

        // To be replaced with `abs_sub`, if
        // https://github.com/rust-lang/rust/issues/62111 lands.
        if int_self <= int_other {
            int_other - int_self <= max_ulps.into()
        } else {
            int_self - int_other <= max_ulps.into()
        }
    }
}

#[cfg(test)]
mod myfloat_tests {
    use super::*;
    use core::f64;
    use num_traits::{One, Zero};

    #[test]
    fn test_creation() {
        let f1 = MyFloat::new(3.14159);
        assert!((f1.to_f64() - 3.14159).abs() < 1e-10);

        let f2 = MyFloat::zero();
        assert_eq!(f2.to_f64(), 0.0);

        let f3 = MyFloat::one();
        assert_eq!(f3.to_f64(), 1.0);

        let f4 = MyFloat::new(f64::consts::PI).to_degrees();
        assert!((f4.to_f64() - 180.0).abs() < 1e-10);

        let f5 = MyFloat::new(180.0).to_radians();
        assert!((f5.to_f64() - f64::consts::PI).abs() < 1e-10);
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
        let with_neg_sign = magnitude.copysign(negative);
        assert_eq!(with_neg_sign.to_f64(), -5.0);
    }

    #[test]
    fn test_min_max_clamp() {
        let a = MyFloat::new(3.0);
        let b = MyFloat::new(7.0);

        let min_result = a.min(b);
        assert_eq!(min_result.to_f64(), 3.0);

        let max_result = a.max(b);
        assert_eq!(max_result.to_f64(), 7.0);

        let value = MyFloat::new(10.0);
        let min_bound = MyFloat::new(2.0);
        let max_bound = MyFloat::new(8.0);
        let clamped = value.clamp(min_bound, max_bound);
        assert_eq!(clamped.to_f64(), 8.0);

        let value2 = MyFloat::new(1.0);
        let clamped2 = value2.clamp(min_bound, max_bound);
        assert_eq!(clamped2.to_f64(), 2.0);
    }

    #[test]
    fn test_num_trait() {
        // Test that from_str_radix returns an error (not implemented)
        let parsed = MyFloat::from_str_radix("42", 10);
        assert!(parsed.is_err());
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
        let from_f64 = <MyFloat as From<f64>>::from(3.14);
        assert!((from_f64.to_f64() - 3.14).abs() < 1e-10);

        let from_i32 = <MyFloat as From<i32>>::from(42i32);
        assert_eq!(from_i32.to_f64(), 42.0);

        let from_u32 = <MyFloat as From<u32>>::from(42u32);
        assert_eq!(from_u32.to_f64(), 42.0);

        let from_i64 = <MyFloat as From<i64>>::from(42i64);
        assert_eq!(from_i64.to_f64(), 42.0);

        let from_u64 = <MyFloat as From<u64>>::from(42u64);
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
        assert!(display_str.contains("3.14159"));

        let debug_str = format!("{:?}", f);
        assert!(debug_str.contains("MyFloat"));
        assert!(debug_str.contains("3.14159"));
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

        let atan2_result = y.atan2(x);
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

    #[test]
    fn test_not_operator() {
        let zero = MyFloat::zero();
        let non_zero = MyFloat::new(5.0);
        let one = MyFloat::one();

        // !0.0 should be 1.0
        let not_zero = !zero;
        assert_eq!(not_zero.to_f64(), 1.0);

        // !5.0 should be 0.0
        let not_non_zero = !non_zero;
        assert_eq!(not_non_zero.to_f64(), 0.0);

        // !1.0 should be 0.0
        let not_one = !one;
        assert_eq!(not_one.to_f64(), 0.0);

        // Test with reference
        let borrowed = MyFloat::new(3.14);
        let not_borrowed = !&borrowed;
        assert_eq!(not_borrowed.to_f64(), 0.0);
    }

    #[test]
    fn test_signed_trait() {
        use num_traits::Signed;

        let positive = MyFloat::new(5.0);
        let negative = MyFloat::new(-3.0);
        let zero = MyFloat::zero();

        // Test abs
        assert_eq!(positive.abs().to_f64(), 5.0);
        assert_eq!(negative.abs().to_f64(), 3.0);
        assert_eq!(zero.abs().to_f64(), 0.0);

        // Test signum
        assert_eq!(positive.signum().to_f64(), 1.0);
        assert_eq!(negative.signum().to_f64(), -1.0);
        assert_eq!(zero.signum().to_f64(), 0.0);

        // Test is_positive
        assert!(positive.is_positive());
        assert!(!negative.is_positive());
        assert!(!zero.is_positive());

        // Test is_negative
        assert!(!positive.is_negative());
        assert!(negative.is_negative());
        assert!(!zero.is_negative());

        // Test abs_sub
        let a = MyFloat::new(10.0);
        let b = MyFloat::new(3.0);
        let abs_sub_result = a.abs_sub(b);
        assert_eq!(abs_sub_result.to_f64(), 7.0);

        let c = MyFloat::new(2.0);
        let d = MyFloat::new(8.0);
        let abs_sub_result2 = c.abs_sub(d);
        assert_eq!(abs_sub_result2.to_f64(), 0.0);
    }
}
