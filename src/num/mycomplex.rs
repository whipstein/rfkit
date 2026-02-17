use crate::num::MyFloat;
use num_complex::{Complex, Complex64};
use num_traits::{ConstOne, ConstZero, Float, Num, One, Pow, Zero};
use std::{
    convert::From,
    fmt,
    fmt::Debug,
    ops::{
        Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign,
        Sub, SubAssign,
    },
};
use twofloat::{TwoFloat, TwoFloatError};

/// A complex number wrapper with fixed precision of 53 bits
#[derive(Copy, Clone, Debug)]
pub struct MyComplex(Complex<TwoFloat>);

impl MyComplex {
    /// A constant `Complex` 0.
    pub const ZERO: Self = Self(Complex::<TwoFloat>::ZERO);

    /// A constant `Complex` 1.
    pub const ONE: Self = Self(Complex::<TwoFloat>::ONE);

    /// A constant `Complex` _i_, the imaginary unit.
    pub const I: Self = Self(Complex::<TwoFloat>::I);

    /// Create a new complex number from real and imaginary parts
    pub fn new(real: MyFloat, imag: MyFloat) -> Self {
        MyComplex(Complex::new(real.into_inner(), imag.into_inner()))
    }

    /// Create a new complex number from real and imaginary parts
    pub fn from_f64(real: f64, imag: f64) -> Self {
        MyComplex(Complex::new(real.into(), imag.into()))
    }

    /// Create a new complex number from a num::complex
    pub fn from_c64(num: Complex64) -> Self {
        MyComplex(Complex::new(num.re.into(), num.im.into()))
    }

    /// Create a new complex number from a num::complex
    pub fn from_complex(num: Complex<TwoFloat>) -> Self {
        MyComplex(num)
    }

    /// Create a new complex number from a real number (imaginary part = 0)
    pub fn from_real(real: MyFloat) -> Self {
        // let real_float = Float::with_val(Self::PRECISION, real);
        MyComplex(Complex::new(real.into_inner(), 0.0.into()))
    }

    /// Create a new complex number from an imaginary number (real part = 0)
    pub fn from_imag(imag: MyFloat) -> Self {
        MyComplex(Complex::new(0.0.into(), imag.into_inner()))
    }

    /// Create a new complex number from a real number (imaginary part = 0)
    pub fn from_real_f64(real: f64) -> Self {
        MyComplex::from_real(real.into())
    }

    /// Create a new complex number from an imaginary number (real part = 0)
    pub fn from_imag_f64(imag: f64) -> Self {
        MyComplex::from_imag(imag.into())
    }

    /// Create a new complex number from a magnitude and angle in radians
    pub fn from_polar(mag: &MyFloat, ang: &MyFloat) -> Self {
        MyComplex::new(mag * ang.cos(), mag * ang.sin())
    }

    /// Create a new complex number from a magnitude and angle in radians as f64
    pub fn from_polar_f64(mag: f64, ang: f64) -> Self {
        MyComplex::from_polar(&mag.into(), &ang.into())
    }

    /// Create a new complex number from real and imaginary parts
    pub fn from_tuple(num: &(MyFloat, MyFloat)) -> Self {
        MyComplex(Complex::new(
            num.0.clone().into_inner(),
            num.1.clone().into_inner(),
        ))
    }

    /// Create a new complex number from real and imaginary parts
    pub fn from_tuple_f64(num: (f64, f64)) -> Self {
        MyComplex(Complex::new(num.0.into(), num.1.into()))
    }

    /// Get the real part as MyFloat
    pub fn re(&self) -> MyFloat {
        self.real().clone()
    }

    /// Get the real part as MyFloat
    pub fn real(&self) -> MyFloat {
        MyFloat::from_float(self.0.re)
    }

    /// Get the imaginary part as MyFloat
    pub fn im(&self) -> MyFloat {
        self.imag().clone()
    }

    /// Get the imaginary part as MyFloat
    pub fn imag(&self) -> MyFloat {
        MyFloat::from_float(self.0.im)
    }

    /// Get 1 / self
    pub fn recip(&self) -> Self {
        MyComplex(self.0.finv())
    }

    /// Get the magnitude (absolute value) of the complex number
    pub fn abs(&self) -> MyFloat {
        MyFloat::from_float(self.0.norm())
    }

    /// Get the argument (phase angle) of the complex number
    pub fn arg(&self) -> MyFloat {
        MyFloat::from_float(self.0.arg())
    }

    /// Get the complex conjugate
    pub fn conj(&self) -> Self {
        MyComplex(self.0.conj())
    }

    /// Get the magnitude in dB of the complex number
    pub fn db(&self) -> MyFloat {
        MyFloat::from_float(TwoFloat::from_f64(20.0) * self.0.norm().log10())
    }

    /// Get the magnitude in dB of the complex number
    pub fn db10(&self) -> MyFloat {
        MyFloat::from_float(TwoFloat::from_f64(10.0) * self.0.norm().log10())
    }

    /// Calculate the square of the magnitude (norm squared)
    pub fn norm(&self) -> MyFloat {
        self.abs()
    }

    /// Calculate the square of the magnitude (norm squared)
    pub fn norm_sqr(&self) -> MyFloat {
        self.norm().square()
    }

    /// Calculate the square root
    pub fn sqrt(&self) -> Self {
        MyComplex(self.0.sqrt())
    }

    /// Calculate the cube root
    pub fn cbrt(&self) -> Self {
        MyComplex(self.0.cbrt())
    }

    /// Calculate the exponential function
    pub fn exp(&self) -> Self {
        MyComplex(self.0.exp())
    }

    /// Calculate e**(self) - 1
    pub fn exp_m1(&self) -> Self {
        MyComplex(self.0.exp()) - Self::ONE
    }

    /// Calculate 2**(self)
    pub fn exp2(&self) -> Self {
        MyComplex(self.0.exp2())
    }

    /// Calculate the natural logarithm
    pub fn ln(&self) -> Self {
        MyComplex(self.0.ln())
    }

    /// Calculate ln(1 + self)
    pub fn ln_1p(&self) -> Self {
        (self + 1.0).ln()
    }

    /// Calculate arbitrary base log(self)
    pub fn log(&self, base: MyFloat) -> Self {
        MyComplex(self.0.log(base.into_inner()))
    }

    /// Calculate the base 2 logarithm
    pub fn log2(&self) -> Self {
        MyComplex(self.0.log2())
    }

    /// Calculate the base 10 logarithm
    pub fn log10(&self) -> Self {
        MyComplex(self.0.log10())
    }

    /// Calculate sine & cosine from self in radians
    pub fn sin_cos(&self) -> (Self, Self) {
        (MyComplex(self.0.sin()), MyComplex(self.0.cos()))
    }

    /// Calculate sine from self in radians
    pub fn sin(&self) -> Self {
        MyComplex(self.0.sin())
    }

    /// Calculate cosine from self in radians
    pub fn cos(&self) -> Self {
        MyComplex(self.0.cos())
    }

    /// Calculate tangent from self in radians
    pub fn tan(&self) -> Self {
        MyComplex(self.0.tan())
    }

    /// Calculate arcsine from self in radians
    pub fn asin(&self) -> Self {
        MyComplex(self.0.asin())
    }

    /// Calculate arccosine from self in radians
    pub fn acos(&self) -> Self {
        MyComplex(self.0.acos())
    }

    /// Calculate arctangent from self in radians
    pub fn atan(&self) -> Self {
        MyComplex(self.0.atan())
    }

    /// Calculate hyperbolic sine from self in radians
    pub fn sinh(&self) -> Self {
        MyComplex(self.0.sinh())
    }

    /// Calculate hyperbolic cosine from self in radians
    pub fn cosh(&self) -> Self {
        MyComplex(self.0.cosh())
    }

    /// Calculate hyperbolic tangent from self in radians
    pub fn tanh(&self) -> Self {
        MyComplex(self.0.tanh())
    }

    /// Calculate hyperbolic arcsine from self in radians
    pub fn asinh(&self) -> Self {
        MyComplex(self.0.asinh())
    }

    /// Calculate hyperbolic arccosine from self in radians
    pub fn acosh(&self) -> Self {
        MyComplex(self.0.acosh())
    }

    /// Calculate hyperbolic arctangent from self in radians
    pub fn atanh(&self) -> Self {
        MyComplex(self.0.atanh())
    }

    /// Raise to a power
    pub fn pow(&self, exp: MyComplex) -> Self {
        MyComplex(self.0.pow(exp.0))
    }

    /// Access the inner rug::Complex (for advanced operations)
    pub fn inner(&self) -> &Complex<TwoFloat> {
        &self.0
    }

    /// Convert to inner rug::Complex (consuming self)
    pub fn into_inner(self) -> Complex<TwoFloat> {
        self.0
    }

    /// Create a NaN complex number
    pub fn nan() -> Self {
        MyComplex(Complex::new(TwoFloat::NAN, TwoFloat::NAN))
    }

    /// Check if the complex number contains NaN
    pub fn is_nan(&self) -> bool {
        self.real().is_nan() || self.imag().is_nan()
    }

    /// Create an infinite complex number
    pub fn infinity() -> Self {
        MyComplex(Complex::new(TwoFloat::INFINITY, TwoFloat::INFINITY))
    }

    /// Check if the complex number is infinite
    pub fn is_infinite(&self) -> bool {
        self.real().is_infinite() || self.imag().is_infinite()
    }

    /// Check if the complex number is finite
    pub fn is_finite(&self) -> bool {
        self.real().is_finite() && self.imag().is_finite()
    }

    /// Check if the complex number is normal (not zero, infinite, or NaN)
    pub fn is_normal(&self) -> bool {
        self.real().is_normal() && self.imag().is_normal()
    }
}

// impl ConstZero for MyComplex {
//     const ZERO: Self = Self::ZERO;
// }

// impl ConstOne for MyComplex {
//     const ONE: Self = Self::ONE;
// }

impl Deref for MyComplex {
    type Target = Complex<TwoFloat>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MyComplex {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl fmt::Display for MyComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let real = self.real().to_f64();
        let imag = self.imag().to_f64();

        if imag == f64::zero() {
            write!(f, "{}", real)
        } else if real == f64::zero() {
            if imag == f64::one() {
                write!(f, "i")
            } else if imag == -f64::one() {
                write!(f, "-i")
            } else {
                write!(f, "{}i", imag)
            }
        } else {
            if imag == f64::one() {
                write!(f, "{} + i", real)
            } else if imag == -f64::one() {
                write!(f, "{} - i", real)
            } else if imag > f64::zero() {
                write!(f, "{} + {}i", real, imag)
            } else {
                write!(f, "{} - {}i", real, -imag)
            }
        }
    }
}

// // Implement basic arithmetic operations
// macro_rules! impl_self_math_op(
//     ($trt:ident, $operator:tt, $mth:ident) => (
//         impl $trt for MyComplex {
//             type Output = Self;

//             fn $mth(self, other: Self) -> Self::Output {
//                 MyComplex(self.0 $operator other.0)
//             }
//         }

//         impl $trt<&MyComplex> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex(self.0 $operator &other.0)
//             }
//         }

//         impl $trt<MyComplex> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex(&self.0 $operator other.0)
//             }
//         }

//         impl $trt<&MyComplex> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex(&self.0 $operator other.0.clone())
//             }
//         }
//     );
// );

// macro_rules! impl_math_op(
//     ($trt:ident, $operator:tt, $mth:ident, f64) => (
//         impl $trt<f64> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: f64) -> Self::Output {
//                 self $operator MyComplex::from_real_f64(other)
//             }
//         }

//         impl $trt<f64> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: f64) -> Self::Output {
//                 self $operator MyComplex::from_real_f64(other)
//             }
//         }

//         impl $trt<&f64> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &f64) -> Self::Output {
//                 self $operator MyComplex::from_real_f64(*other)
//             }
//         }

//         impl $trt<&f64> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &f64) -> Self::Output {
//                 self $operator MyComplex::from_real_f64(*other)
//             }
//         }

//         impl $trt<MyComplex> for f64 {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex::from_real_f64(self) $operator other
//             }
//         }

//         impl $trt<MyComplex> for &f64 {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex::from_real_f64(*self) $operator other
//             }
//         }

//         impl $trt<&MyComplex> for f64 {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex::from_real_f64(self) $operator other
//             }
//         }

//         impl $trt<&MyComplex> for &f64 {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex::from_real_f64(*self) $operator other
//             }
//         }
//     );
//     ($trt:ident, $operator:tt, $mth:ident, Complex64) => (
//         impl $trt<Complex64> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: Complex64) -> Self::Output {
//                 self $operator MyComplex::from_c64(other)
//             }
//         }

//         impl $trt<Complex64> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: Complex64) -> Self::Output {
//                 self $operator MyComplex::from_c64(other)
//             }
//         }

//         impl $trt<&Complex64> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &Complex64) -> Self::Output {
//                 self $operator MyComplex::from_c64(*other)
//             }
//         }

//         impl $trt<&Complex64> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &Complex64) -> Self::Output {
//                 self $operator MyComplex::from_c64(*other)
//             }
//         }

//         impl $trt<MyComplex> for Complex64 {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex::from_c64(self) $operator other
//             }
//         }

//         impl $trt<MyComplex> for &Complex64 {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex::from_c64(*self) $operator other
//             }
//         }

//         impl $trt<&MyComplex> for Complex64 {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex::from_c64(self) $operator other
//             }
//         }

//         impl $trt<&MyComplex> for &Complex64 {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex::from_c64(*self) $operator other
//             }
//         }
//     );
//     ($trt:ident, $operator:tt, $mth:ident, MyFloat) => (
//         impl $trt<MyFloat> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: MyFloat) -> Self::Output {
//                 MyComplex(self.0 $operator other.inner())
//             }
//         }

//         impl $trt<MyFloat> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: MyFloat) -> Self::Output {
//                 MyComplex(self.0 $operator other.inner())
//             }
//         }

//         impl $trt<&MyFloat> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyFloat) -> Self::Output {
//                 MyComplex(self.0 $operator other.inner())
//             }
//         }

//         impl $trt<&MyFloat> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyFloat) -> Self::Output {
//                 MyComplex(self.0 $operator other.inner())
//             }
//         }

//         impl $trt<MyComplex> for MyFloat {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex::from_real(self) $operator other
//             }
//         }

//         impl $trt<MyComplex> for &MyFloat {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex::from_real(self.clone()) $operator other
//             }
//         }

//         impl $trt<&MyComplex> for MyFloat {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex::from_real(self) $operator other
//             }
//         }

//         impl $trt<&MyComplex> for &MyFloat {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex::from_real(self.clone()) $operator other
//             }
//         }
//     );
//     ($trt:ident, $operator:tt, $mth:ident, $rhs:ident) => (
//         impl $trt<$rhs> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: $rhs) -> Self::Output {
//                 MyComplex(self.0 $operator other)
//             }
//         }

//         impl $trt<$rhs> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: $rhs) -> Self::Output {
//                 MyComplex(self.0 $operator other)
//             }
//         }

//         impl $trt<&$rhs> for MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &$rhs) -> Self::Output {
//                 MyComplex(self.0 $operator *other)
//             }
//         }

//         impl $trt<&$rhs> for &MyComplex {
//             type Output = MyComplex;

//             fn $mth(self, other: &$rhs) -> Self::Output {
//                 MyComplex(self.0 $operator *other)
//             }
//         }

//         impl $trt<MyComplex> for $rhs {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex(self $operator other.0)
//             }
//         }

//         impl $trt<MyComplex> for &$rhs {
//             type Output = MyComplex;

//             fn $mth(self, other: MyComplex) -> Self::Output {
//                 MyComplex(self $operator other.0)
//             }
//         }

//         impl $trt<&MyComplex> for $rhs {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex(self $operator other.0)
//             }
//         }

//         impl $trt<&MyComplex> for &$rhs {
//             type Output = MyComplex;

//             fn $mth(self, other: &MyComplex) -> Self::Output {
//                 MyComplex(*self $operator other.0)
//             }
//         }
//     );
// );

// // Implement assignment operators
// macro_rules! impl_self_assign_math_op(
//     ($trt:ident, $operator:tt, $mth:ident) => (
//         impl $trt for MyComplex {
//             fn $mth(&mut self, other: Self) {
//                 self.0 $operator other.0;
//             }
//         }

//         impl $trt<&MyComplex> for MyComplex {
//             fn $mth(&mut self, other: &MyComplex) {
//                 self.0 $operator &other.0;
//             }
//         }
//     );
// );

// macro_rules! impl_assign_math_op(
//     ($trt:ident, $operator:tt, $mth:ident, Complex64) => (
//         impl $trt<Complex64> for MyComplex {
//             fn $mth(&mut self, other: Complex64) {
//                 self.0 $operator MyComplex::from_c64(other).inner();
//             }
//         }

//         impl $trt<&Complex64> for MyComplex {
//             fn $mth(&mut self, other: &Complex64) {
//                 self.0 $operator MyComplex::from_c64(*other).inner();
//             }
//         }
//     );
//     ($trt:ident, $operator:tt, $mth:ident, MyFloat) => (
//         impl $trt<MyFloat> for MyComplex {
//             fn $mth(&mut self, other: MyFloat) {
//                 self.0 $operator MyComplex::from_real(other).inner();
//             }
//         }

//         impl $trt<&MyFloat> for MyComplex {
//             fn $mth(&mut self, other: &MyFloat) {
//                 self.0 $operator MyComplex::from_real(other.clone()).inner();
//             }
//         }
//     );
//     ($trt:ident, $operator:tt, $mth:ident, $rhs:ident) => (
//         impl $trt<$rhs> for MyComplex {
//             fn $mth(&mut self, other: $rhs) {
//                 self.0 $operator MyComplex::from_f64(other, 0.0).inner();
//             }
//         }

//         impl $trt<&$rhs> for MyComplex {
//             fn $mth(&mut self, other: &$rhs) {
//                 self.0 $operator MyComplex::from_f64(*other, 0.0).inner();
//             }
//         }
//     );
// );

// impl_self_math_op!(Add, +, add);
// impl_self_math_op!(Sub, -, sub);
// impl_self_math_op!(Mul, *, mul);
// impl_self_math_op!(Div, /, div);
// impl_self_assign_math_op!(AddAssign, +=, add_assign);
// impl_self_assign_math_op!(SubAssign, -=, sub_assign);
// impl_self_assign_math_op!(MulAssign, *=, mul_assign);
// impl_self_assign_math_op!(DivAssign, /=, div_assign);
// impl_math_op!(Add, +, add, f64);
// impl_math_op!(Sub, -, sub, f64);
// impl_math_op!(Mul, *, mul, f64);
// impl_math_op!(Div, /, div, f64);
// impl_math_op!(Add, +, add, Complex64);
// impl_math_op!(Sub, -, sub, Complex64);
// impl_math_op!(Mul, *, mul, Complex64);
// impl_math_op!(Div, /, div, Complex64);
// impl_math_op!(Add, +, add, MyFloat);
// impl_math_op!(Sub, -, sub, MyFloat);
// impl_math_op!(Mul, *, mul, MyFloat);
// impl_math_op!(Div, /, div, MyFloat);
// impl_assign_math_op!(AddAssign, +=, add_assign, f64);
// impl_assign_math_op!(SubAssign, -=, sub_assign, f64);
// impl_assign_math_op!(MulAssign, *=, mul_assign, f64);
// impl_assign_math_op!(DivAssign, /=, div_assign, f64);
// impl_assign_math_op!(AddAssign, +=, add_assign, Complex64);
// impl_assign_math_op!(SubAssign, -=, sub_assign, Complex64);
// impl_assign_math_op!(MulAssign, *=, mul_assign, Complex64);
// impl_assign_math_op!(DivAssign, /=, div_assign, Complex64);
// impl_assign_math_op!(AddAssign, +=, add_assign, MyFloat);
// impl_assign_math_op!(SubAssign, -=, sub_assign, MyFloat);
// impl_assign_math_op!(MulAssign, *=, mul_assign, MyFloat);
// impl_assign_math_op!(DivAssign, /=, div_assign, MyFloat);

// impl Neg for MyComplex {
//     type Output = Self;

//     fn neg(self) -> Self {
//         MyComplex(-self.0)
//     }
// }

// impl Neg for &MyComplex {
//     type Output = MyComplex;

//     fn neg(self) -> MyComplex {
//         MyComplex(-self.0.clone())
//     }
// }

// Implement Pow trait
impl Pow<MyComplex> for MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: MyComplex) -> MyComplex {
        MyComplex(self.0.pow(exp.0))
    }
}

impl Pow<&MyComplex> for MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: &MyComplex) -> MyComplex {
        MyComplex(self.0.pow(exp.0))
    }
}

impl Pow<MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: MyComplex) -> MyComplex {
        MyComplex(self.0.pow(exp.0))
    }
}

impl Pow<&MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: &MyComplex) -> MyComplex {
        MyComplex(self.0.pow(exp.0))
    }
}

// Also implement Pow for common numeric types
impl Pow<f64> for MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: f64) -> MyComplex {
        let exp_complex = MyComplex::from_real_f64(exp);
        self.pow(exp_complex)
    }
}

impl Pow<f64> for &MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: f64) -> MyComplex {
        let exp_complex = MyComplex::from_real_f64(exp);
        self.pow(exp_complex)
    }
}

impl Pow<i32> for MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: i32) -> MyComplex {
        let exp_complex = MyComplex::from_real_f64(exp as f64);
        self.pow(exp_complex)
    }
}

impl Pow<i32> for &MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: i32) -> MyComplex {
        let exp_complex = MyComplex::from_real_f64(exp as f64);
        self.pow(exp_complex)
    }
}

// // Implement Zero trait
// impl Zero for MyComplex {
//     fn zero() -> Self {
//         MyComplex::from_f64(0.0, 0.0)
//     }

//     fn is_zero(&self) -> bool {
//         self.real().is_zero() && self.imag().is_zero()
//     }
// }

// // Implement One trait
// impl One for MyComplex {
//     fn one() -> Self {
//         MyComplex::from_f64(1.0, 0.0)
//     }

//     fn is_one(&self) -> bool {
//         *self == Self::one()
//     }
// }

// impl Rem for MyComplex {
//     type Output = MyComplex;

//     fn rem(self, rhs: Self) -> Self::Output {
//         // For complex numbers, we implement: a % b = a - b * floor(a / b)
//         // where floor for complex numbers applies to both real and imaginary parts

//         // Check for zero divisor
//         if rhs.0.is_zero() {
//             panic!("Division by zero in complex remainder operation");
//         }

//         // Calculate a / b
//         let division = self.0 / rhs.0;

//         // Floor both real and imaginary parts
//         let real_floor = division.re.floor();
//         let imag_floor = division.im.floor();

//         // Create the floored quotient
//         let floored_quotient = Complex::new(real_floor, imag_floor);

//         // Calculate b * floor(a / b)
//         let product = rhs.0 * floored_quotient;

//         // Calculate a - b * floor(a / b)
//         MyComplex(self.0 - product)
//     }
// }

// // Implement Rem for references to avoid unnecessary clones
// impl Rem<&MyComplex> for &MyComplex {
//     type Output = MyComplex;

//     fn rem(self, rhs: &MyComplex) -> Self::Output {
//         if rhs.0.is_zero() {
//             panic!("Division by zero in complex remainder operation");
//         }

//         let division = self.0 / rhs.0;

//         let real_floor = division.re.floor();
//         let imag_floor = division.im.floor();

//         let floored_quotient = Complex::new(real_floor, imag_floor);

//         let product = rhs.0 * floored_quotient;

//         MyComplex(self.0 - product)
//     }
// }

// // Implement Rem with owned and borrowed combinations
// impl Rem<&MyComplex> for MyComplex {
//     type Output = MyComplex;

//     fn rem(self, rhs: &MyComplex) -> Self::Output {
//         (&self).rem(rhs)
//     }
// }

// impl Rem<MyComplex> for &MyComplex {
//     type Output = MyComplex;

//     fn rem(self, rhs: MyComplex) -> Self::Output {
//         self.rem(&rhs)
//     }
// }

// // Implement RemAssign for in-place operations
// impl RemAssign for MyComplex {
//     fn rem_assign(&mut self, rhs: Self) {
//         *self = std::mem::take(self) % rhs;
//     }
// }

// impl RemAssign<&MyComplex> for MyComplex {
//     fn rem_assign(&mut self, rhs: &MyComplex) {
//         let result = (&*self) % rhs;
//         *self = result;
//     }
// }

// // Implement Rem with real numbers (f64)
// impl Rem<f64> for MyComplex {
//     type Output = MyComplex;

//     fn rem(self, rhs: f64) -> Self::Output {
//         if rhs == 0.0 {
//             panic!("Division by zero in complex remainder operation");
//         }

//         MyComplex(self.0 % TwoFloat::from_f64(rhs))
//     }
// }

// impl RemAssign<f64> for MyComplex {
//     fn rem_assign(&mut self, rhs: f64) {
//         *self = std::mem::take(self) % rhs;
//     }
// }

// // Implement Num trait (requires Zero + One + PartialEq + Clone + arithmetic ops)
// impl Num for MyComplex {
//     type FromStrRadixErr = TwoFloatError;

//     fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
//         Err(TwoFloatError::ParseError)
//     }
// }

// // Implement Not trait (logical NOT: 0+0i -> 1+0i, non-zero -> 0+0i)
// impl Not for MyComplex {
//     type Output = Self;

//     fn not(self) -> Self::Output {
//         if self.0.is_zero() {
//             MyComplex::one()
//         } else {
//             MyComplex::zero()
//         }
//     }
// }

// impl Not for &MyComplex {
//     type Output = MyComplex;

//     fn not(self) -> Self::Output {
//         if self.0.is_zero() {
//             MyComplex::one()
//         } else {
//             MyComplex::zero()
//         }
//     }
// }

// // Implement Traits
// impl Clone for MyComplex {
//     fn clone(&self) -> Self {
//         MyComplex(self.0.clone())
//     }
// }

// impl Default for MyComplex {
//     fn default() -> Self {
//         MyComplex(Complex::new(0.0.into(), 0.0.into()))
//     }
// }

// impl fmt::Display for MyComplex {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let real = self.real().to_f64();
//         let imag = self.imag().to_f64();

//         if imag == f64::zero() {
//             write!(f, "{}", real)
//         } else if real == f64::zero() {
//             if imag == f64::one() {
//                 write!(f, "i")
//             } else if imag == -f64::one() {
//                 write!(f, "-i")
//             } else {
//                 write!(f, "{}i", imag)
//             }
//         } else {
//             if imag == f64::one() {
//                 write!(f, "{} + i", real)
//             } else if imag == -f64::one() {
//                 write!(f, "{} - i", real)
//             } else if imag > f64::zero() {
//                 write!(f, "{} + {}i", real, imag)
//             } else {
//                 write!(f, "{} - {}i", real, -imag)
//             }
//         }
//     }
// }

// // Implement PartialEq for comparisons
// impl PartialEq for MyComplex {
//     fn eq(&self, other: &Self) -> bool {
//         self.0 == other.0
//     }
// }

// // Implement Conversion
// impl From<(f64, f64)> for MyComplex {
//     fn from((real, imag): (f64, f64)) -> Self {
//         MyComplex::from_f64(real, imag)
//     }
// }

// impl From<(MyFloat, MyFloat)> for MyComplex {
//     fn from((real, imag): (MyFloat, MyFloat)) -> Self {
//         MyComplex::new(real, imag)
//     }
// }

// impl From<Complex64> for MyComplex {
//     fn from(num: Complex64) -> Self {
//         MyComplex::from_c64(num)
//     }
// }

// impl From<&Complex64> for MyComplex {
//     fn from(num: &Complex64) -> Self {
//         MyComplex::from_c64(*num)
//     }
// }

// impl From<MyComplex> for Complex64 {
//     fn from(value: MyComplex) -> Complex64 {
//         Complex64::new(value.real().to_f64(), value.imag().to_f64())
//     }
// }

// impl From<&MyComplex> for Complex64 {
//     fn from(value: &MyComplex) -> Complex64 {
//         Complex64::new(value.real().to_f64(), value.imag().to_f64())
//     }
// }

// impl From<MyFloat> for MyComplex {
//     fn from(real: MyFloat) -> MyComplex {
//         MyComplex::from_real(real)
//     }
// }

// impl From<&MyFloat> for MyComplex {
//     fn from(real: &MyFloat) -> MyComplex {
//         MyComplex::from_real(*real)
//     }
// }

// impl From<f64> for MyComplex {
//     fn from(real: f64) -> Self {
//         MyComplex::from_real_f64(real)
//     }
// }

// impl From<&f64> for MyComplex {
//     fn from(real: &f64) -> MyComplex {
//         MyComplex::from_real_f64(*real)
//     }
// }

// // Implement Sum trait for owned values
// impl std::iter::Sum for MyComplex {
//     fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
//         iter.fold(MyComplex::zero(), |acc, x| acc + x)
//     }
// }

// // Implement Sum trait for borrowed values
// impl<'a> std::iter::Sum<&'a MyComplex> for MyComplex {
//     fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
//         iter.fold(MyComplex::zero(), |acc, x| acc + x)
//     }
// }

#[cfg(test)]
mod mycomplex_tests {
    use super::*;
    use core::f64;
    use float_cmp::*;

    #[test]
    fn test_creation() {
        let z1 = MyComplex::from_f64(3.0, 4.0);
        assert_eq!(z1.real(), MyFloat::new(3.0));
        assert_eq!(z1.imag(), MyFloat::new(4.0));

        let z2 = MyComplex::from_real_f64(5.0);
        assert_eq!(z2.real(), MyFloat::new(5.0));
        assert_eq!(z2.imag(), MyFloat::new(0.0));

        let z3 = MyComplex::from_imag_f64(2.0);
        assert_eq!(z3.real(), MyFloat::new(0.0));
        assert_eq!(z3.imag(), MyFloat::new(2.0));

        let z4 = MyComplex::from_polar_f64(1.0, f64::consts::PI / 2.0);
        approx_eq!(f64, z4.real().to_f64(), 0.0, F64Margin::default());
        approx_eq!(f64, z4.imag().to_f64(), 1.0, F64Margin::default());
    }

    #[test]
    fn test_arithmetic() {
        let z1 = MyComplex::from_f64(1.0, 2.0);
        let z2 = MyComplex::from_f64(3.0, 4.0);

        let sum = &z1 + &z2;
        assert_eq!(sum.real(), MyFloat::new(4.0));
        assert_eq!(sum.imag(), MyFloat::new(6.0));

        let diff = &z2 - &z1;
        assert_eq!(diff.real(), MyFloat::new(2.0));
        assert_eq!(diff.imag(), MyFloat::new(2.0));

        let prod = &z1 * &z2;
        assert_eq!(prod.real(), MyFloat::new(-5.0)); // (1*3 - 2*4)
        assert_eq!(prod.imag(), MyFloat::new(10.0)); // (1*4 + 2*3)
    }

    #[test]
    fn test_magnitude_and_conjugate() {
        let z = MyComplex::from_f64(3.0, 4.0);
        assert_eq!(z.abs(), MyFloat::new(5.0)); // sqrt(3^2 + 4^2) = 5

        let conj = z.conj();
        assert_eq!(conj.real(), MyFloat::new(3.0));
        assert_eq!(conj.imag(), MyFloat::new(-4.0));
    }

    #[test]
    fn test_display() {
        let z1 = MyComplex::from_f64(3.0, 4.0);
        assert_eq!(format!("{}", z1), "3 + 4i");

        let z2 = MyComplex::from_f64(3.0, -4.0);
        assert_eq!(format!("{}", z2), "3 - 4i");

        let z3 = MyComplex::from_real_f64(5.0);
        assert_eq!(format!("{}", z3), "5");

        let z4 = MyComplex::from_imag_f64(2.0);
        assert_eq!(format!("{}", z4), "2i");
    }

    #[test]
    fn test_assignment_operators() {
        let mut z1 = MyComplex::from_f64(1.0, 2.0);
        let z2 = MyComplex::from_f64(3.0, 4.0);

        // Test AddAssign
        z1 += &z2;
        assert_eq!(z1.real(), MyFloat::new(4.0));
        assert_eq!(z1.imag(), MyFloat::new(6.0));

        // Test SubAssign
        z1 -= &z2;
        assert_eq!(z1.real(), MyFloat::new(1.0));
        assert_eq!(z1.imag(), MyFloat::new(2.0));

        // Test MulAssign
        z1 *= &z2;
        assert_eq!(z1.real(), MyFloat::new(-5.0)); // (1*3 - 2*4)
        assert_eq!(z1.imag(), MyFloat::new(10.0)); // (1*4 + 2*3)

        // Test DivAssign
        let mut z3 = MyComplex::from_f64(10.0, 0.0);
        let z4 = MyComplex::from_f64(2.0, 0.0);
        z3 /= &z4;
        assert_eq!(z3.real(), MyFloat::new(5.0));
        assert_eq!(z3.imag(), MyFloat::new(0.0));
    }

    #[test]
    fn test_pow_trait() {
        let z = MyComplex::from_f64(2.0, 0.0); // Real number 2
        let exp = MyComplex::from_f64(3.0, 0.0); // Real exponent 3

        let result = z.pow(exp);
        assert!((result.real() - 8.0).abs() < MyFloat::new(1e-10)); // 2^3 = 8
        assert!(result.imag().abs() < MyFloat::new(1e-10)); // Should be real

        // Test with f64 exponent
        let z2 = MyComplex::from_f64(4.0, 0.0);
        let result2 = z2.pow(0.5); // Square root
        assert!((result2.real() - 2.0).abs() < MyFloat::new(1e-10));
        assert!(result2.imag().abs() < MyFloat::new(1e-10));

        // Test with i32 exponent
        let z3 = MyComplex::from_f64(3.0, 0.0);
        let result3 = z3.pow(2i32); // 3^2
        assert!((result3.real() - 9.0).abs() < MyFloat::new(1e-10));
        assert!(result3.imag().abs() < MyFloat::new(1e-10));
    }

    #[test]
    fn test_zero_one_traits() {
        use num_traits::{One, Zero};

        // Test Zero trait
        let zero = MyComplex::zero();
        assert_eq!(zero.real(), MyFloat::new(0.0));
        assert_eq!(zero.imag(), MyFloat::new(0.0));
        assert!(zero.is_zero());

        let non_zero = MyComplex::from_f64(1.0, 2.0);
        assert!(!non_zero.is_zero());

        // Test One trait
        let one = MyComplex::one();
        assert_eq!(one.real(), MyFloat::new(1.0));
        assert_eq!(one.imag(), MyFloat::new(0.0));
        assert!(one.is_one());

        let non_one = MyComplex::from_f64(2.0, 0.0);
        assert!(!non_one.is_one());
    }

    #[test]
    fn test_nan_and_infinity() {
        let nan = MyComplex::nan();
        assert!(nan.is_nan());
        assert!(!nan.is_finite());
        assert!(!nan.is_normal());

        let inf = MyComplex::infinity();
        assert!(inf.is_infinite());
        assert!(!inf.is_finite());
        assert!(!inf.is_normal());

        let normal = MyComplex::from_f64(1.0, 2.0);
        assert!(!normal.is_nan());
        assert!(!normal.is_infinite());
        assert!(normal.is_finite());
        assert!(normal.is_normal());

        let zero = MyComplex::zero();
        assert!(!zero.is_nan());
        assert!(!zero.is_infinite());
        assert!(zero.is_finite());
        assert!(!zero.is_normal()); // Zero is not considered "normal"
    }

    #[test]
    fn test_num_trait() {
        use num_traits::Num;

        // Test that from_str_radix returns an error (not implemented)
        let parsed = MyComplex::from_str_radix("42", 10);
        assert!(parsed.is_err());
    }

    #[test]
    fn test_complex_remainder() {
        let a = MyComplex::from_f64(5.0, 3.0);
        let b = MyComplex::from_f64(2.0, 1.0);

        let result = &a % &b;
        println!("({}) % ({}) = {}", a, b, result);

        // Test with real number
        let result2 = &a % MyComplex::from(2.0);
        println!("({}) % 2.0 = {}", a, result2);
    }

    #[test]
    fn test_rem_assign() {
        let mut a = MyComplex::from_f64(7.0, 4.0);
        let b = MyComplex::from_f64(3.0, 2.0);

        println!("Before: {}", a);
        a %= &b;
        println!("After %= : {}", a);
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_division_by_zero() {
        let a = MyComplex::from_f64(5.0, 3.0);
        let zero = MyComplex::from_f64(0.0, 0.0);
        let _ = a % zero;
    }

    #[test]
    fn test_default() {
        let default_complex = MyComplex::default();
        println!("Default complex: {}", default_complex);

        // Should be zero
        assert!(default_complex.0.is_zero());

        // Can be used in operations
        // let a = MyComplex::new(5.0, 3.0);
        // let _result = &a % &default_complex; // This should panic due to division by zero
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_default_division_by_zero() {
        let a = MyComplex::from_f64(5.0, 3.0);
        let default_zero = MyComplex::default();
        let _ = a % default_zero;
    }

    #[test]
    fn test_not_operator() {
        let zero = MyComplex::zero();
        let non_zero = MyComplex::from_f64(5.0, 3.0);
        let one = MyComplex::one();
        let pure_imag = MyComplex::from_imag_f64(2.0);

        // !0+0i should be 1+0i
        let not_zero = !zero;
        assert_eq!(not_zero.real(), MyFloat::new(1.0));
        assert_eq!(not_zero.imag(), MyFloat::new(0.0));

        // !(5+3i) should be 0+0i
        let not_non_zero = !non_zero;
        assert_eq!(not_non_zero.real(), MyFloat::new(0.0));
        assert_eq!(not_non_zero.imag(), MyFloat::new(0.0));

        // !(1+0i) should be 0+0i
        let not_one = !one;
        assert_eq!(not_one.real(), MyFloat::new(0.0));
        assert_eq!(not_one.imag(), MyFloat::new(0.0));

        // !(0+2i) should be 0+0i (non-zero imaginary part)
        let not_pure_imag = !pure_imag;
        assert_eq!(not_pure_imag.real(), MyFloat::new(0.0));
        assert_eq!(not_pure_imag.imag(), MyFloat::new(0.0));

        // Test with reference
        let borrowed = MyComplex::from_f64(3.14, 2.71);
        let not_borrowed = !&borrowed;
        assert_eq!(not_borrowed.real(), MyFloat::new(0.0));
        assert_eq!(not_borrowed.imag(), MyFloat::new(0.0));

        // Test double negation
        let double_not = !!MyComplex::zero();
        assert_eq!(double_not, MyComplex::zero());
    }

    #[test]
    fn test_sum_trait() {
        // Test Sum trait with owned values
        let values = vec![
            MyComplex::from_f64(1.0, 2.0),
            MyComplex::from_f64(3.0, 4.0),
            MyComplex::from_f64(5.0, 6.0),
        ];

        let sum: MyComplex = values.into_iter().sum();
        assert_eq!(sum.real(), MyFloat::new(9.0)); // 1 + 3 + 5
        assert_eq!(sum.imag(), MyFloat::new(12.0)); // 2 + 4 + 6

        // Test Sum trait with borrowed values
        let values = vec![
            MyComplex::from_f64(2.0, 1.0),
            MyComplex::from_f64(4.0, 3.0),
            MyComplex::from_f64(6.0, 5.0),
        ];

        let sum: MyComplex = values.iter().sum();
        assert_eq!(sum.real(), MyFloat::new(12.0)); // 2 + 4 + 6
        assert_eq!(sum.imag(), MyFloat::new(9.0)); // 1 + 3 + 5

        // Test empty iterator
        let empty: Vec<MyComplex> = vec![];
        let sum: MyComplex = empty.into_iter().sum();
        assert_eq!(sum.real(), MyFloat::new(0.0));
        assert_eq!(sum.imag(), MyFloat::new(0.0));
        assert!(sum.is_zero());
    }
}
