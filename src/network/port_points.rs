#![allow(unused)]
use crate::{
    consts::MathConst,
    frequency::Frequency,
    impedance::ComplexNumberType,
    math::*,
    network::{NetworkPortPoints, PortPoints},
    num::{ComplexScalar, RealScalar},
    parameter::RFParameter,
    pts::{Points, Pts},
    unit::Unit,
};
use ndarray::{OwnedRepr, prelude::*};
use ndarray_linalg::*;
use num::Float;
use num_complex::{Complex, Complex64, ComplexFloat, c64};
use num_traits::{ConstZero, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use simple_error::SimpleError;
use std::{
    error::Error,
    f64::consts::PI,
    fmt, fs,
    iter::Iterator,
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
    process,
    process::Child,
    slice::Iter,
};
use twofloat::TwoFloat;

impl<T: ComplexScalar> NetworkPortPoints<T> for PortPoints<T>
where
    <T as ComplexFloat>::Real: RealScalar + MathConst,
{
    fn db(&self) -> PortPoints<T::Real> {
        Points::from_shape_fn(self.len(), |i| T::Real::C20 * Float::log10(self[i].abs()))
    }

    fn deg(&self) -> PortPoints<T::Real> {
        Points::from_shape_fn(self.len(), |i| self[i].arg().to_degrees())
    }

    fn im(&self) -> PortPoints<T::Real> {
        Points::from_shape_fn(self.len(), |i| self[i].im())
    }

    fn mag(&self) -> PortPoints<T::Real> {
        Points::from_shape_fn(self.len(), |i| self[i].abs())
    }

    fn rad(&self) -> PortPoints<T::Real> {
        Points::from_shape_fn(self.len(), |i| self[i].arg())
    }

    fn re(&self) -> PortPoints<T::Real> {
        Points::from_shape_fn(self.len(), |i| self[i].re())
    }
}
