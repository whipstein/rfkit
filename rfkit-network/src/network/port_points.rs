#![allow(unused)]
use super::*;
use crate::parameter::RFParameter;
use ndarray::{OwnedRepr, prelude::*};
use ndarray_linalg::*;
use num_complex::{Complex, Complex64, ComplexFloat, c64};
use num_traits::{ConstZero, Float, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use rfkit_base::prelude::*;
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

impl<T: RealScalar> NetworkPortPoints<T> for PortPoints<T>
where
    <T as ComplexFloat>::Real: RealScalar,
{
    fn db(&self) -> PortPoints<T> {
        Points::from_shape_fn(self.len(), |i| Float::log10(self[i].abs()) * 20.0)
    }

    fn deg(&self) -> PortPoints<T> {
        Points::from_shape_fn(self.len(), |i| self[i].arg().to_degrees())
    }

    fn im(&self) -> PortPoints<T> {
        Points::from_shape_fn(self.len(), |i| self[i].im())
    }

    fn mag(&self) -> PortPoints<T> {
        Points::from_shape_fn(self.len(), |i| self[i].abs())
    }

    fn rad(&self) -> PortPoints<T> {
        Points::from_shape_fn(self.len(), |i| self[i].arg())
    }

    fn re(&self) -> PortPoints<T> {
        Points::from_shape_fn(self.len(), |i| self[i].re())
    }
}
