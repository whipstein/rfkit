use crate::myfloat::MyFloat;
use num::complex::Complex64;
use num_traits::{Num, One, Zero};
use rug::ops::{Pow, PowAssign};
use rug::{Complex, Float};
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

// pub mod inverse;
// pub mod zgemm;

/// A complex number wrapper with fixed precision of 53 bits
pub struct MyComplex(rug::Complex);

impl MyComplex {
    /// Fixed precision for all operations
    const PRECISION: u32 = 53;

    /// Create a new complex number from real and imaginary parts
    pub fn new(real: MyFloat, imag: MyFloat) -> Self {
        MyComplex(Complex::with_val(
            Self::PRECISION,
            (real.into_inner(), imag.into_inner()),
        ))
    }

    /// Create a new complex number from real and imaginary parts
    pub fn from_f64(real: f64, imag: f64) -> Self {
        let real_float = Float::with_val(Self::PRECISION, real);
        let imag_float = Float::with_val(Self::PRECISION, imag);
        MyComplex(Complex::with_val(Self::PRECISION, (real_float, imag_float)))
    }

    /// Create a new complex number from a num::complex
    pub fn from_c64(num: Complex64) -> Self {
        MyComplex(Complex::with_val(Self::PRECISION, (num.re, num.im)))
    }

    /// Create a new complex number from a real number (imaginary part = 0)
    pub fn from_real(real: MyFloat) -> Self {
        // let real_float = Float::with_val(Self::PRECISION, real);
        MyComplex(Complex::with_val(Self::PRECISION, real.into_inner()))
    }

    /// Create a new complex number from an imaginary number (real part = 0)
    pub fn from_imag(imag: MyFloat) -> Self {
        MyComplex(Complex::with_val(Self::PRECISION, (0, imag.into_inner())))
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
        MyComplex(Complex::with_val(
            Self::PRECISION,
            (num.0.clone().into_inner(), num.1.clone().into_inner()),
        ))
    }

    /// Create a new complex number from real and imaginary parts
    pub fn from_tuple_f64(num: (f64, f64)) -> Self {
        MyComplex(Complex::with_val(Self::PRECISION, num))
    }

    /// Get the real part as MyFloat
    pub fn re(&self) -> MyFloat {
        MyFloat::from_float(self.0.real().clone())
    }

    /// Get the real part as MyFloat
    pub fn real(&self) -> MyFloat {
        MyFloat::from_float(self.0.real().clone())
    }

    /// Get the imaginary part as MyFloat
    pub fn im(&self) -> MyFloat {
        MyFloat::from_float(self.0.imag().clone())
    }

    /// Get the imaginary part as MyFloat
    pub fn imag(&self) -> MyFloat {
        MyFloat::from_float(self.0.imag().clone())
    }

    /// Get the magnitude (absolute value) of the complex number
    pub fn abs(&self) -> MyFloat {
        let mut temp = self.0.clone();
        temp.abs_mut();
        temp.real().into()
    }

    /// Get the argument (phase angle) of the complex number
    pub fn arg(&self) -> MyFloat {
        let mut temp = self.0.clone();
        temp.arg_mut();
        temp.real().into()
    }

    /// Get the complex conjugate
    pub fn conj(&self) -> Self {
        let mut temp = self.0.clone();
        temp.conj_mut();
        MyComplex(temp)
    }

    /// Get the magnitude in dB of the complex number
    pub fn db(&self) -> MyFloat {
        20.0 * self.abs().log10()
    }

    /// Get the magnitude in dB of the complex number
    pub fn db10(&self) -> MyFloat {
        10.0 * self.abs().log10()
    }

    /// Calculate the square of the magnitude (norm squared)
    pub fn norm(&self) -> MyFloat {
        self.abs()
    }

    /// Calculate the square of the magnitude (norm squared)
    pub fn norm_sqr(&self) -> MyFloat {
        let mut temp = self.0.clone();
        temp.norm_mut();
        temp.real().into()
    }

    /// Calculate the square root
    pub fn sqrt(&self) -> Self {
        let mut temp = self.0.clone();
        temp.sqrt_mut();
        MyComplex(temp)
    }

    /// Calculate the exponential function
    pub fn exp(&self) -> Self {
        let mut temp = self.0.clone();
        temp.exp_mut();
        MyComplex(temp)
    }

    /// Calculate the natural logarithm
    pub fn ln(&self) -> Self {
        let mut temp = self.0.clone();
        temp.ln_mut();
        MyComplex(temp)
    }

    /// Calculate sine
    pub fn sin(&self) -> Self {
        let mut temp = self.0.clone();
        temp.sin_mut();
        MyComplex(temp)
    }

    /// Calculate cosine
    pub fn cos(&self) -> Self {
        let mut temp = self.0.clone();
        temp.cos_mut();
        MyComplex(temp)
    }

    /// Raise to a power
    pub fn pow(&self, exp: &MyComplex) -> Self {
        let mut temp = self.0.clone();
        temp.pow_assign(&exp.0);
        MyComplex(temp)
    }

    /// Access the inner rug::Complex (for advanced operations)
    pub fn inner(&self) -> &rug::Complex {
        &self.0
    }

    /// Convert to inner rug::Complex (consuming self)
    pub fn into_inner(self) -> rug::Complex {
        self.0
    }

    /// Create a NaN complex number
    pub fn nan() -> Self {
        let nan_float = Float::with_val(Self::PRECISION, f64::NAN);
        MyComplex(Complex::with_val(
            Self::PRECISION,
            (nan_float.clone(), nan_float),
        ))
    }

    /// Check if the complex number contains NaN
    pub fn is_nan(&self) -> bool {
        self.0.real().is_nan() || self.0.imag().is_nan()
    }

    /// Create an infinite complex number
    pub fn infinity() -> Self {
        let inf_float = Float::with_val(Self::PRECISION, f64::INFINITY);
        MyComplex(Complex::with_val(
            Self::PRECISION,
            (inf_float.clone(), inf_float),
        ))
    }

    /// Check if the complex number is infinite
    pub fn is_infinite(&self) -> bool {
        self.0.real().is_infinite() || self.0.imag().is_infinite()
    }

    /// Check if the complex number is finite
    pub fn is_finite(&self) -> bool {
        self.0.real().is_finite() && self.0.imag().is_finite()
    }

    /// Check if the complex number is normal (not zero, infinite, or NaN)
    pub fn is_normal(&self) -> bool {
        self.0.real().is_normal() && self.0.imag().is_normal()
    }
}

// Implement basic arithmetic operations
impl Add for MyComplex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        MyComplex(self.0 + other.0)
    }
}

impl Add<Complex64> for MyComplex {
    type Output = Self;

    fn add(self, other: Complex64) -> Self::Output {
        self + MyComplex::from_c64(other)
    }
}

impl Add<&Complex64> for MyComplex {
    type Output = Self;

    fn add(self, other: &Complex64) -> Self::Output {
        self + MyComplex::from_c64(*other)
    }
}

impl Add<Complex64> for &MyComplex {
    type Output = MyComplex;

    fn add(self, other: Complex64) -> Self::Output {
        self.clone() + MyComplex::from_c64(other)
    }
}

impl Add<&Complex64> for &MyComplex {
    type Output = MyComplex;

    fn add(self, other: &Complex64) -> Self::Output {
        self.clone() + MyComplex::from_c64(*other)
    }
}

impl Add<&MyComplex> for MyComplex {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        MyComplex(self.0 + &other.0)
    }
}

impl Add<MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn add(self, other: MyComplex) -> MyComplex {
        MyComplex(&self.0 + other.0)
    }
}

impl Add<&MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn add(self, other: &MyComplex) -> MyComplex {
        MyComplex(&self.0 + other.0.clone())
    }
}

impl Add<MyFloat> for MyComplex {
    type Output = Self;

    fn add(self, other: MyFloat) -> Self::Output {
        MyComplex(self.0 + other.inner())
    }
}

impl Add<MyFloat> for &MyComplex {
    type Output = MyComplex;

    fn add(self, other: MyFloat) -> Self::Output {
        MyComplex(self.0.clone() + other.inner())
    }
}

impl Add<&MyFloat> for MyComplex {
    type Output = Self;

    fn add(self, other: &MyFloat) -> Self::Output {
        MyComplex(self.0 + other.inner())
    }
}

impl Add<&MyFloat> for &MyComplex {
    type Output = MyComplex;

    fn add(self, other: &MyFloat) -> Self::Output {
        MyComplex(self.0.clone() + other.inner())
    }
}

impl Add<f64> for MyComplex {
    type Output = Self;

    fn add(self, other: f64) -> Self::Output {
        MyComplex(self.0 + other)
    }
}

impl Add<&f64> for MyComplex {
    type Output = Self;

    fn add(self, other: &f64) -> Self::Output {
        MyComplex(self.0 + other)
    }
}

impl Add<f64> for &MyComplex {
    type Output = MyComplex;

    fn add(self, other: f64) -> Self::Output {
        MyComplex(self.0.clone() + other)
    }
}

impl Add<&f64> for &MyComplex {
    type Output = MyComplex;

    fn add(self, other: &f64) -> Self::Output {
        MyComplex(self.0.clone() + other)
    }
}

impl Add<MyComplex> for Complex64 {
    type Output = MyComplex;

    fn add(self, other: MyComplex) -> Self::Output {
        &MyComplex::from_c64(self) + &other
    }
}

impl Add<&MyComplex> for Complex64 {
    type Output = MyComplex;

    fn add(self, other: &MyComplex) -> Self::Output {
        MyComplex::from_c64(self) + other
    }
}

impl Add<MyComplex> for &Complex64 {
    type Output = MyComplex;

    fn add(self, other: MyComplex) -> Self::Output {
        &MyComplex::from_c64(*self) + &other
    }
}

impl Add<&MyComplex> for &Complex64 {
    type Output = MyComplex;

    fn add(self, other: &MyComplex) -> Self::Output {
        MyComplex::from_c64(*self) + other
    }
}

impl Add<MyComplex> for MyFloat {
    type Output = MyComplex;

    fn add(self, other: MyComplex) -> Self::Output {
        MyComplex(self.inner() + other.0.clone())
    }
}

impl Add<&MyComplex> for MyFloat {
    type Output = MyComplex;

    fn add(self, other: &MyComplex) -> Self::Output {
        MyComplex(self.inner() + other.0.clone())
    }
}

impl Add<MyComplex> for &MyFloat {
    type Output = MyComplex;

    fn add(self, other: MyComplex) -> Self::Output {
        MyComplex(self.inner() + other.0.clone())
    }
}

impl Add<&MyComplex> for &MyFloat {
    type Output = MyComplex;

    fn add(self, other: &MyComplex) -> Self::Output {
        MyComplex(self.inner() + other.0.clone())
    }
}

impl Add<MyComplex> for f64 {
    type Output = MyComplex;

    fn add(self, other: MyComplex) -> Self::Output {
        MyComplex(self + other.0.clone())
    }
}

impl Add<&MyComplex> for f64 {
    type Output = MyComplex;

    fn add(self, other: &MyComplex) -> Self::Output {
        MyComplex(self + other.0.clone())
    }
}

impl Add<MyComplex> for &f64 {
    type Output = MyComplex;

    fn add(self, other: MyComplex) -> Self::Output {
        MyComplex(self + other.0.clone())
    }
}

impl Add<&MyComplex> for &f64 {
    type Output = MyComplex;

    fn add(self, other: &MyComplex) -> Self::Output {
        MyComplex(self + other.0.clone())
    }
}

impl Sub for MyComplex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        MyComplex(self.0 - other.0)
    }
}

impl Sub<Complex64> for MyComplex {
    type Output = Self;

    fn sub(self, other: Complex64) -> Self::Output {
        self - MyComplex::from_c64(other)
    }
}

impl Sub<&Complex64> for MyComplex {
    type Output = Self;

    fn sub(self, other: &Complex64) -> Self::Output {
        self - MyComplex::from_c64(*other)
    }
}

impl Sub<Complex64> for &MyComplex {
    type Output = MyComplex;

    fn sub(self, other: Complex64) -> Self::Output {
        self.clone() - MyComplex::from_c64(other)
    }
}

impl Sub<&Complex64> for &MyComplex {
    type Output = MyComplex;

    fn sub(self, other: &Complex64) -> Self::Output {
        self.clone() - MyComplex::from_c64(*other)
    }
}

impl Sub<&MyComplex> for MyComplex {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        MyComplex(self.0 - &other.0)
    }
}

impl Sub<MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn sub(self, other: MyComplex) -> MyComplex {
        MyComplex(&self.0 - other.0)
    }
}

impl Sub<&MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn sub(self, other: &MyComplex) -> MyComplex {
        MyComplex(&self.0 - other.0.clone())
    }
}

impl Sub<MyFloat> for MyComplex {
    type Output = Self;

    fn sub(self, other: MyFloat) -> Self::Output {
        MyComplex(self.0 - other.inner())
    }
}

impl Sub<MyFloat> for &MyComplex {
    type Output = MyComplex;

    fn sub(self, other: MyFloat) -> Self::Output {
        MyComplex(self.0.clone() - other.inner())
    }
}

impl Sub<&MyFloat> for MyComplex {
    type Output = Self;

    fn sub(self, other: &MyFloat) -> Self::Output {
        MyComplex(self.0 - other.inner())
    }
}

impl Sub<&MyFloat> for &MyComplex {
    type Output = MyComplex;

    fn sub(self, other: &MyFloat) -> Self::Output {
        MyComplex(self.0.clone() - other.inner())
    }
}

impl Sub<f64> for MyComplex {
    type Output = Self;

    fn sub(self, other: f64) -> Self::Output {
        MyComplex(self.0 - other)
    }
}

impl Sub<&f64> for MyComplex {
    type Output = Self;

    fn sub(self, other: &f64) -> Self::Output {
        MyComplex(self.0 - other)
    }
}

impl Sub<f64> for &MyComplex {
    type Output = MyComplex;

    fn sub(self, other: f64) -> Self::Output {
        MyComplex(self.0.clone() - other)
    }
}

impl Sub<&f64> for &MyComplex {
    type Output = MyComplex;

    fn sub(self, other: &f64) -> Self::Output {
        MyComplex(self.0.clone() - other)
    }
}

impl Sub<MyComplex> for Complex64 {
    type Output = MyComplex;

    fn sub(self, other: MyComplex) -> Self::Output {
        &MyComplex::from_c64(self) - &other
    }
}

impl Sub<&MyComplex> for Complex64 {
    type Output = MyComplex;

    fn sub(self, other: &MyComplex) -> Self::Output {
        MyComplex::from_c64(self) - other
    }
}

impl Sub<MyComplex> for &Complex64 {
    type Output = MyComplex;

    fn sub(self, other: MyComplex) -> Self::Output {
        &MyComplex::from_c64(*self) - &other
    }
}

impl Sub<&MyComplex> for &Complex64 {
    type Output = MyComplex;

    fn sub(self, other: &MyComplex) -> Self::Output {
        MyComplex::from_c64(*self) - other
    }
}

impl Sub<MyComplex> for MyFloat {
    type Output = MyComplex;

    fn sub(self, other: MyComplex) -> Self::Output {
        MyComplex(self.inner() - other.0.clone())
    }
}

impl Sub<&MyComplex> for MyFloat {
    type Output = MyComplex;

    fn sub(self, other: &MyComplex) -> Self::Output {
        MyComplex(self.inner() - other.0.clone())
    }
}

impl Sub<MyComplex> for &MyFloat {
    type Output = MyComplex;

    fn sub(self, other: MyComplex) -> Self::Output {
        MyComplex(self.inner() - other.0.clone())
    }
}

impl Sub<&MyComplex> for &MyFloat {
    type Output = MyComplex;

    fn sub(self, other: &MyComplex) -> Self::Output {
        MyComplex(self.inner() - other.0.clone())
    }
}

impl Sub<MyComplex> for f64 {
    type Output = MyComplex;

    fn sub(self, other: MyComplex) -> Self::Output {
        MyComplex(self - other.0.clone())
    }
}

impl Sub<&MyComplex> for f64 {
    type Output = MyComplex;

    fn sub(self, other: &MyComplex) -> Self::Output {
        MyComplex(self - other.0.clone())
    }
}

impl Sub<MyComplex> for &f64 {
    type Output = MyComplex;

    fn sub(self, other: MyComplex) -> Self::Output {
        MyComplex(self - other.0.clone())
    }
}

impl Sub<&MyComplex> for &f64 {
    type Output = MyComplex;

    fn sub(self, other: &MyComplex) -> Self::Output {
        MyComplex(self - other.0.clone())
    }
}

impl Mul for MyComplex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        MyComplex(self.0 * other.0)
    }
}

impl Mul<Complex64> for MyComplex {
    type Output = Self;

    fn mul(self, other: Complex64) -> Self::Output {
        self * MyComplex::from_c64(other)
    }
}

impl Mul<&Complex64> for MyComplex {
    type Output = Self;

    fn mul(self, other: &Complex64) -> Self::Output {
        self * MyComplex::from_c64(*other)
    }
}

impl Mul<Complex64> for &MyComplex {
    type Output = MyComplex;

    fn mul(self, other: Complex64) -> Self::Output {
        self.clone() * MyComplex::from_c64(other)
    }
}

impl Mul<&Complex64> for &MyComplex {
    type Output = MyComplex;

    fn mul(self, other: &Complex64) -> Self::Output {
        self.clone() * MyComplex::from_c64(*other)
    }
}

impl Mul<&MyComplex> for MyComplex {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        MyComplex(self.0 * &other.0)
    }
}

impl Mul<MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn mul(self, other: MyComplex) -> MyComplex {
        MyComplex(&self.0 * other.0)
    }
}

impl Mul<&MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn mul(self, other: &MyComplex) -> MyComplex {
        MyComplex(&self.0 * other.0.clone())
    }
}

impl Mul<MyFloat> for MyComplex {
    type Output = Self;

    fn mul(self, other: MyFloat) -> Self::Output {
        MyComplex(self.0 * other.inner())
    }
}

impl Mul<MyFloat> for &MyComplex {
    type Output = MyComplex;

    fn mul(self, other: MyFloat) -> Self::Output {
        MyComplex(self.0.clone() * other.inner())
    }
}

impl Mul<&MyFloat> for MyComplex {
    type Output = Self;

    fn mul(self, other: &MyFloat) -> Self::Output {
        MyComplex(self.0 * other.inner())
    }
}

impl Mul<&MyFloat> for &MyComplex {
    type Output = MyComplex;

    fn mul(self, other: &MyFloat) -> Self::Output {
        MyComplex(self.0.clone() * other.inner())
    }
}

impl Mul<f64> for MyComplex {
    type Output = Self;

    fn mul(self, other: f64) -> Self::Output {
        MyComplex(self.0 * other)
    }
}

impl Mul<&f64> for MyComplex {
    type Output = Self;

    fn mul(self, other: &f64) -> Self::Output {
        MyComplex(self.0 * other)
    }
}

impl Mul<f64> for &MyComplex {
    type Output = MyComplex;

    fn mul(self, other: f64) -> Self::Output {
        MyComplex(self.0.clone() * other)
    }
}

impl Mul<&f64> for &MyComplex {
    type Output = MyComplex;

    fn mul(self, other: &f64) -> Self::Output {
        MyComplex(self.0.clone() * other)
    }
}

impl Mul<MyComplex> for Complex64 {
    type Output = MyComplex;

    fn mul(self, other: MyComplex) -> Self::Output {
        &MyComplex::from_c64(self) * &other
    }
}

impl Mul<&MyComplex> for Complex64 {
    type Output = MyComplex;

    fn mul(self, other: &MyComplex) -> Self::Output {
        MyComplex::from_c64(self) * other
    }
}

impl Mul<MyComplex> for &Complex64 {
    type Output = MyComplex;

    fn mul(self, other: MyComplex) -> Self::Output {
        &MyComplex::from_c64(*self) * &other
    }
}

impl Mul<&MyComplex> for &Complex64 {
    type Output = MyComplex;

    fn mul(self, other: &MyComplex) -> Self::Output {
        MyComplex::from_c64(*self) * other
    }
}

impl Mul<MyComplex> for MyFloat {
    type Output = MyComplex;

    fn mul(self, other: MyComplex) -> Self::Output {
        MyComplex(self.inner() * other.0.clone())
    }
}

impl Mul<&MyComplex> for MyFloat {
    type Output = MyComplex;

    fn mul(self, other: &MyComplex) -> Self::Output {
        MyComplex(self.inner() * other.0.clone())
    }
}

impl Mul<MyComplex> for &MyFloat {
    type Output = MyComplex;

    fn mul(self, other: MyComplex) -> Self::Output {
        MyComplex(self.inner() * other.0.clone())
    }
}

impl Mul<&MyComplex> for &MyFloat {
    type Output = MyComplex;

    fn mul(self, other: &MyComplex) -> Self::Output {
        MyComplex(self.inner() * other.0.clone())
    }
}

impl Mul<MyComplex> for f64 {
    type Output = MyComplex;

    fn mul(self, other: MyComplex) -> Self::Output {
        MyComplex(self * other.0.clone())
    }
}

impl Mul<&MyComplex> for f64 {
    type Output = MyComplex;

    fn mul(self, other: &MyComplex) -> Self::Output {
        MyComplex(self * other.0.clone())
    }
}

impl Mul<MyComplex> for &f64 {
    type Output = MyComplex;

    fn mul(self, other: MyComplex) -> Self::Output {
        MyComplex(self * other.0.clone())
    }
}

impl Mul<&MyComplex> for &f64 {
    type Output = MyComplex;

    fn mul(self, other: &MyComplex) -> Self::Output {
        MyComplex(self * other.0.clone())
    }
}

impl Div for MyComplex {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        MyComplex(self.0 / other.0)
    }
}

impl Div<Complex64> for MyComplex {
    type Output = Self;

    fn div(self, other: Complex64) -> Self::Output {
        self / MyComplex::from_c64(other)
    }
}

impl Div<&Complex64> for MyComplex {
    type Output = Self;

    fn div(self, other: &Complex64) -> Self::Output {
        self / MyComplex::from_c64(*other)
    }
}

impl Div<Complex64> for &MyComplex {
    type Output = MyComplex;

    fn div(self, other: Complex64) -> Self::Output {
        self.clone() / MyComplex::from_c64(other)
    }
}

impl Div<&Complex64> for &MyComplex {
    type Output = MyComplex;

    fn div(self, other: &Complex64) -> Self::Output {
        self.clone() / MyComplex::from_c64(*other)
    }
}

impl Div<&MyComplex> for MyComplex {
    type Output = Self;

    fn div(self, other: &Self) -> Self {
        MyComplex(self.0 / &other.0)
    }
}

impl Div<MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn div(self, other: MyComplex) -> MyComplex {
        MyComplex(&self.0 / other.0)
    }
}

impl Div<&MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn div(self, other: &MyComplex) -> MyComplex {
        MyComplex(&self.0 / other.0.clone())
    }
}

impl Div<MyFloat> for MyComplex {
    type Output = Self;

    fn div(self, other: MyFloat) -> Self::Output {
        MyComplex(self.0 / other.inner())
    }
}

impl Div<MyFloat> for &MyComplex {
    type Output = MyComplex;

    fn div(self, other: MyFloat) -> Self::Output {
        MyComplex(self.0.clone() / other.inner())
    }
}

impl Div<&MyFloat> for MyComplex {
    type Output = Self;

    fn div(self, other: &MyFloat) -> Self::Output {
        MyComplex(self.0 / other.inner())
    }
}

impl Div<&MyFloat> for &MyComplex {
    type Output = MyComplex;

    fn div(self, other: &MyFloat) -> Self::Output {
        MyComplex(self.0.clone() / other.inner())
    }
}

impl Div<f64> for MyComplex {
    type Output = Self;

    fn div(self, other: f64) -> Self::Output {
        MyComplex(self.0 / other)
    }
}

impl Div<&f64> for MyComplex {
    type Output = Self;

    fn div(self, other: &f64) -> Self::Output {
        MyComplex(self.0 / other)
    }
}

impl Div<f64> for &MyComplex {
    type Output = MyComplex;

    fn div(self, other: f64) -> Self::Output {
        MyComplex(self.0.clone() / other)
    }
}

impl Div<&f64> for &MyComplex {
    type Output = MyComplex;

    fn div(self, other: &f64) -> Self::Output {
        MyComplex(self.0.clone() / other)
    }
}

impl Div<MyComplex> for Complex64 {
    type Output = MyComplex;

    fn div(self, other: MyComplex) -> Self::Output {
        &MyComplex::from_c64(self) / &other
    }
}

impl Div<&MyComplex> for Complex64 {
    type Output = MyComplex;

    fn div(self, other: &MyComplex) -> Self::Output {
        MyComplex::from_c64(self) / other
    }
}

impl Div<MyComplex> for &Complex64 {
    type Output = MyComplex;

    fn div(self, other: MyComplex) -> Self::Output {
        &MyComplex::from_c64(*self) / &other
    }
}

impl Div<&MyComplex> for &Complex64 {
    type Output = MyComplex;

    fn div(self, other: &MyComplex) -> Self::Output {
        MyComplex::from_c64(*self) / other
    }
}

impl Div<MyComplex> for MyFloat {
    type Output = MyComplex;

    fn div(self, other: MyComplex) -> Self::Output {
        MyComplex(self.inner() / other.0.clone())
    }
}

impl Div<&MyComplex> for MyFloat {
    type Output = MyComplex;

    fn div(self, other: &MyComplex) -> Self::Output {
        MyComplex(self.inner() / other.0.clone())
    }
}

impl Div<MyComplex> for &MyFloat {
    type Output = MyComplex;

    fn div(self, other: MyComplex) -> Self::Output {
        MyComplex(self.inner() / other.0.clone())
    }
}

impl Div<&MyComplex> for &MyFloat {
    type Output = MyComplex;

    fn div(self, other: &MyComplex) -> Self::Output {
        MyComplex(self.inner() / other.0.clone())
    }
}

impl Div<MyComplex> for f64 {
    type Output = MyComplex;

    fn div(self, other: MyComplex) -> Self::Output {
        MyComplex(self / other.0.clone())
    }
}

impl Div<&MyComplex> for f64 {
    type Output = MyComplex;

    fn div(self, other: &MyComplex) -> Self::Output {
        MyComplex(self / other.0.clone())
    }
}

impl Div<MyComplex> for &f64 {
    type Output = MyComplex;

    fn div(self, other: MyComplex) -> Self::Output {
        MyComplex(self / other.0.clone())
    }
}

impl Div<&MyComplex> for &f64 {
    type Output = MyComplex;

    fn div(self, other: &MyComplex) -> Self::Output {
        MyComplex(self / other.0.clone())
    }
}

impl Neg for MyComplex {
    type Output = Self;

    fn neg(self) -> Self {
        MyComplex(-self.0)
    }
}

impl Neg for &MyComplex {
    type Output = MyComplex;

    fn neg(self) -> MyComplex {
        MyComplex(-self.0.clone())
    }
}

// Implement assignment operators
impl AddAssign for MyComplex {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl AddAssign<Complex64> for MyComplex {
    fn add_assign(&mut self, other: Complex64) {
        self.0 += MyComplex::from_c64(other).inner();
    }
}

impl AddAssign<&Complex64> for MyComplex {
    fn add_assign(&mut self, other: &Complex64) {
        self.0 += MyComplex::from_c64(*other).inner();
    }
}

impl AddAssign<f64> for MyComplex {
    fn add_assign(&mut self, other: f64) {
        self.0 += MyComplex::from_f64(other, 0.0).inner();
    }
}

impl AddAssign<&f64> for MyComplex {
    fn add_assign(&mut self, other: &f64) {
        self.0 += MyComplex::from_f64(*other, 0.0).inner();
    }
}

impl AddAssign<&MyComplex> for MyComplex {
    fn add_assign(&mut self, other: &MyComplex) {
        self.0 += &other.0;
    }
}

impl AddAssign<MyFloat> for MyComplex {
    fn add_assign(&mut self, other: MyFloat) {
        self.0 += MyComplex::from_real(other).inner();
    }
}

impl AddAssign<&MyFloat> for MyComplex {
    fn add_assign(&mut self, other: &MyFloat) {
        self.0 += MyComplex::from_real(other.clone()).inner();
    }
}

impl SubAssign for MyComplex {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl SubAssign<Complex64> for MyComplex {
    fn sub_assign(&mut self, other: Complex64) {
        self.0 -= MyComplex::from_c64(other).inner();
    }
}

impl SubAssign<&Complex64> for MyComplex {
    fn sub_assign(&mut self, other: &Complex64) {
        self.0 -= MyComplex::from_c64(*other).inner();
    }
}

impl SubAssign<f64> for MyComplex {
    fn sub_assign(&mut self, other: f64) {
        self.0 -= MyComplex::from_f64(other, 0.0).inner();
    }
}

impl SubAssign<&f64> for MyComplex {
    fn sub_assign(&mut self, other: &f64) {
        self.0 -= MyComplex::from_f64(*other, 0.0).inner();
    }
}

impl SubAssign<&MyComplex> for MyComplex {
    fn sub_assign(&mut self, other: &MyComplex) {
        self.0 -= &other.0;
    }
}

impl SubAssign<MyFloat> for MyComplex {
    fn sub_assign(&mut self, other: MyFloat) {
        self.0 -= MyComplex::from_real(other).inner();
    }
}

impl SubAssign<&MyFloat> for MyComplex {
    fn sub_assign(&mut self, other: &MyFloat) {
        self.0 -= MyComplex::from_real(other.clone()).inner();
    }
}

impl MulAssign for MyComplex {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl MulAssign<Complex64> for MyComplex {
    fn mul_assign(&mut self, other: Complex64) {
        self.0 *= MyComplex::from_c64(other).inner();
    }
}

impl MulAssign<&Complex64> for MyComplex {
    fn mul_assign(&mut self, other: &Complex64) {
        self.0 *= MyComplex::from_c64(*other).inner();
    }
}

impl MulAssign<f64> for MyComplex {
    fn mul_assign(&mut self, other: f64) {
        self.0 *= MyComplex::from_f64(other, 0.0).inner();
    }
}

impl MulAssign<&f64> for MyComplex {
    fn mul_assign(&mut self, other: &f64) {
        self.0 *= MyComplex::from_f64(*other, 0.0).inner();
    }
}

impl MulAssign<&MyComplex> for MyComplex {
    fn mul_assign(&mut self, other: &MyComplex) {
        self.0 *= &other.0;
    }
}

impl MulAssign<MyFloat> for MyComplex {
    fn mul_assign(&mut self, other: MyFloat) {
        self.0 *= MyComplex::from_real(other).inner();
    }
}

impl MulAssign<&MyFloat> for MyComplex {
    fn mul_assign(&mut self, other: &MyFloat) {
        self.0 *= MyComplex::from_real(other.clone()).inner();
    }
}

impl DivAssign for MyComplex {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl DivAssign<Complex64> for MyComplex {
    fn div_assign(&mut self, other: Complex64) {
        self.0 /= MyComplex::from_c64(other).inner();
    }
}

impl DivAssign<&Complex64> for MyComplex {
    fn div_assign(&mut self, other: &Complex64) {
        self.0 /= MyComplex::from_c64(*other).inner();
    }
}

impl DivAssign<f64> for MyComplex {
    fn div_assign(&mut self, other: f64) {
        self.0 /= MyComplex::from_f64(other, 0.0).inner();
    }
}

impl DivAssign<&f64> for MyComplex {
    fn div_assign(&mut self, other: &f64) {
        self.0 /= MyComplex::from_f64(*other, 0.0).inner();
    }
}

impl DivAssign<&MyComplex> for MyComplex {
    fn div_assign(&mut self, other: &MyComplex) {
        self.0 /= &other.0;
    }
}

impl DivAssign<MyFloat> for MyComplex {
    fn div_assign(&mut self, other: MyFloat) {
        self.0 /= MyComplex::from_real(other).inner();
    }
}

impl DivAssign<&MyFloat> for MyComplex {
    fn div_assign(&mut self, other: &MyFloat) {
        self.0 /= MyComplex::from_real(other.clone()).inner();
    }
}

// Implement Pow trait
impl Pow<MyComplex> for MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: MyComplex) -> MyComplex {
        let mut temp = self.0;
        temp.pow_assign(&exp.0);
        MyComplex(temp)
    }
}

impl Pow<&MyComplex> for MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: &MyComplex) -> MyComplex {
        let mut temp = self.0;
        temp.pow_assign(&exp.0);
        MyComplex(temp)
    }
}

impl Pow<MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: MyComplex) -> MyComplex {
        let mut temp = self.0.clone();
        temp.pow_assign(&exp.0);
        MyComplex(temp)
    }
}

impl Pow<&MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn pow(self, exp: &MyComplex) -> MyComplex {
        let mut temp = self.0.clone();
        temp.pow_assign(&exp.0);
        MyComplex(temp)
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
        self.pow(&exp_complex)
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
        self.pow(&exp_complex)
    }
}

// Implement PowAssign trait
impl PowAssign<MyComplex> for MyComplex {
    fn pow_assign(&mut self, exp: MyComplex) {
        self.0.pow_assign(&exp.0);
    }
}

impl PowAssign<&MyComplex> for MyComplex {
    fn pow_assign(&mut self, exp: &MyComplex) {
        self.0.pow_assign(&exp.0);
    }
}

impl PowAssign<f64> for MyComplex {
    fn pow_assign(&mut self, exp: f64) {
        let exp_complex = Complex::with_val(Self::PRECISION, exp);
        self.0.pow_assign(&exp_complex);
    }
}

impl PowAssign<i32> for MyComplex {
    fn pow_assign(&mut self, exp: i32) {
        let exp_complex = Complex::with_val(Self::PRECISION, exp);
        self.0.pow_assign(&exp_complex);
    }
}

// Implement Zero trait
impl Zero for MyComplex {
    fn zero() -> Self {
        MyComplex::from_f64(0.0, 0.0)
    }

    fn is_zero(&self) -> bool {
        self.0.real().is_zero() && self.0.imag().is_zero()
    }
}

// Implement One trait
impl One for MyComplex {
    fn one() -> Self {
        MyComplex::from_f64(1.0, 0.0)
    }

    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

impl Rem for MyComplex {
    type Output = MyComplex;

    fn rem(self, rhs: Self) -> Self::Output {
        // For complex numbers, we implement: a % b = a - b * floor(a / b)
        // where floor for complex numbers applies to both real and imaginary parts

        // Check for zero divisor
        if rhs.0.is_zero() {
            panic!("Division by zero in complex remainder operation");
        }

        // Get the precision from the operands
        let precision = self.0.prec().max(rhs.0.prec());

        // Calculate a / b
        let mut division = Complex::with_val(precision, &self.0);
        division /= &rhs.0;

        // Floor both real and imaginary parts
        let real_floor = division.real().to_f64().floor();
        let imag_floor = division.imag().to_f64().floor();

        // Create the floored quotient
        let floored_quotient = Complex::with_val(precision, (real_floor, imag_floor));

        // Calculate b * floor(a / b)
        let mut product = Complex::with_val(precision, &rhs.0);
        product *= &floored_quotient;

        // Calculate a - b * floor(a / b)
        let mut result = Complex::with_val(precision, &self.0);
        result -= &product;

        MyComplex(result)
    }
}

// Implement Rem for references to avoid unnecessary clones
impl Rem<&MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn rem(self, rhs: &MyComplex) -> Self::Output {
        if rhs.0.is_zero() {
            panic!("Division by zero in complex remainder operation");
        }

        let precision = self.0.prec().max(rhs.0.prec());

        let mut division = Complex::with_val(precision, &self.0);
        division /= &rhs.0;

        let real_floor = division.real().to_f64().floor();
        let imag_floor = division.imag().to_f64().floor();

        let floored_quotient = Complex::with_val(precision, (real_floor, imag_floor));

        let mut product = Complex::with_val(precision, &rhs.0);
        product *= &floored_quotient;

        let mut result = Complex::with_val(precision, &self.0);
        result -= &product;

        MyComplex(result)
    }
}

// Implement Rem with owned and borrowed combinations
impl Rem<&MyComplex> for MyComplex {
    type Output = MyComplex;

    fn rem(self, rhs: &MyComplex) -> Self::Output {
        (&self).rem(rhs)
    }
}

impl Rem<MyComplex> for &MyComplex {
    type Output = MyComplex;

    fn rem(self, rhs: MyComplex) -> Self::Output {
        self.rem(&rhs)
    }
}

// Implement RemAssign for in-place operations
impl RemAssign for MyComplex {
    fn rem_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) % rhs;
    }
}

impl RemAssign<&MyComplex> for MyComplex {
    fn rem_assign(&mut self, rhs: &MyComplex) {
        let result = (&*self) % rhs;
        *self = result;
    }
}

// Implement Rem with real numbers (f64)
impl Rem<f64> for MyComplex {
    type Output = MyComplex;

    fn rem(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            panic!("Division by zero in complex remainder operation");
        }

        let precision = self.0.prec();
        let rhs_complex = Complex::with_val(precision, rhs);
        let rhs_wrapper = MyComplex(rhs_complex);

        self % rhs_wrapper
    }
}

impl RemAssign<f64> for MyComplex {
    fn rem_assign(&mut self, rhs: f64) {
        *self = std::mem::take(self) % rhs;
    }
}

// Implement Num trait (requires Zero + One + PartialEq + Clone + arithmetic ops)
impl Num for MyComplex {
    type FromStrRadixErr = rug::float::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        // Simple implementation - parse as real number only
        // For full complex parsing, you'd need a more sophisticated parser
        let real = Float::parse_radix(str, radix as i32)?;
        let real_with_precision = Float::with_val(Self::PRECISION, real);
        Ok(MyComplex(Complex::with_val(
            Self::PRECISION,
            real_with_precision,
        )))
    }
}

// Implement Not trait (logical NOT: 0+0i -> 1+0i, non-zero -> 0+0i)
impl Not for MyComplex {
    type Output = Self;

    fn not(self) -> Self::Output {
        if self.0.is_zero() {
            MyComplex::one()
        } else {
            MyComplex::zero()
        }
    }
}

impl Not for &MyComplex {
    type Output = MyComplex;

    fn not(self) -> Self::Output {
        if self.0.is_zero() {
            MyComplex::one()
        } else {
            MyComplex::zero()
        }
    }
}

// Implement Clone
impl Clone for MyComplex {
    fn clone(&self) -> Self {
        MyComplex(self.0.clone())
    }
}

// Default implementation - creates zero complex number with default precision
impl Default for MyComplex {
    fn default() -> Self {
        MyComplex(Complex::new(53)) // Creates 0 + 0i with 53-bit precision (standard f64 precision)
    }
}

// Implement Display for pretty printing
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

// Implement Debug
impl fmt::Debug for MyComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MyComplex({} + {}i)", self.real(), self.imag())
    }
}

// Implement PartialEq for comparisons
impl PartialEq for MyComplex {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

// Conversion from f64 (real number)
impl From<f64> for MyComplex {
    fn from(real: f64) -> Self {
        MyComplex::from_real_f64(real)
    }
}

// Conversion from (f64, f64) tuple
impl From<(f64, f64)> for MyComplex {
    fn from((real, imag): (f64, f64)) -> Self {
        MyComplex::from_f64(real, imag)
    }
}

// Conversion from (MyFloat, MyFloat) tuple
impl From<(MyFloat, MyFloat)> for MyComplex {
    fn from((real, imag): (MyFloat, MyFloat)) -> Self {
        MyComplex::new(real, imag)
    }
}

// Conversion from Complex64
impl From<Complex64> for MyComplex {
    fn from(num: Complex64) -> Self {
        MyComplex::from_c64(num)
    }
}

// Conversion from MyComplex
impl From<MyComplex> for Complex64 {
    fn from(value: MyComplex) -> Complex64 {
        Complex64::new(value.real().to_f64(), value.imag().to_f64())
    }
}

// Conversion from MyComplex
impl From<&MyComplex> for Complex64 {
    fn from(value: &MyComplex) -> Complex64 {
        Complex64::new(value.real().to_f64(), value.imag().to_f64())
    }
}

// Conversion to MyComplex (assuming MyComplex is available)
// This creates a complex number with the float as the real part and zero imaginary part
impl From<MyFloat> for MyComplex {
    fn from(value: MyFloat) -> MyComplex {
        MyComplex::from_real(value)
    }
}

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
    fn test_pow_assign_trait() {
        // Test PowAssign with MyComplex exponent
        let mut z1 = MyComplex::from_f64(2.0, 0.0);
        let exp1 = MyComplex::from_f64(3.0, 0.0);
        z1.pow_assign(exp1);
        assert!((z1.real() - 8.0).abs() < MyFloat::new(1e-10)); // 2^3 = 8
        assert!(z1.imag().abs() < MyFloat::new(1e-10));

        // Test PowAssign with borrowed MyComplex exponent
        let mut z2 = MyComplex::from_f64(3.0, 0.0);
        let exp2 = MyComplex::from_f64(2.0, 0.0);
        z2.pow_assign(&exp2);
        assert!((z2.real() - 9.0).abs() < MyFloat::new(1e-10)); // 3^2 = 9
        assert!(z2.imag().abs() < MyFloat::new(1e-10));

        // Test PowAssign with f64 exponent
        let mut z3 = MyComplex::from_f64(16.0, 0.0);
        z3.pow_assign(0.25); // Fourth root
        assert!((z3.real() - 2.0).abs() < MyFloat::new(1e-10));
        assert!(z3.imag().abs() < MyFloat::new(1e-10));

        // Test PowAssign with i32 exponent
        let mut z4 = MyComplex::from_f64(5.0, 0.0);
        z4.pow_assign(2i32);
        assert!((z4.real() - 25.0).abs() < MyFloat::new(1e-10)); // 5^2 = 25
        assert!(z4.imag().abs() < MyFloat::new(1e-10));
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

        // Test parsing from string
        let parsed = MyComplex::from_str_radix("42", 10).unwrap();
        assert_eq!(parsed.real(), MyFloat::new(42.0));
        assert_eq!(parsed.imag(), MyFloat::new(0.0));

        let parsed_hex = MyComplex::from_str_radix("ff", 16).unwrap();
        assert_eq!(parsed_hex.real(), MyFloat::new(255.0));
        assert_eq!(parsed_hex.imag(), MyFloat::new(0.0));
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
}
