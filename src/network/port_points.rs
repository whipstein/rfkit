#![allow(unused)]
use crate::{
    frequency::Frequency,
    impedance::ComplexNumberType,
    math::*,
    network::{NetworkPortPoints, PortPoints},
    num::{MyComplex, MyFloat},
    parameter::RFParameter,
    pts::{Points, Pts},
    unit::Unit,
};
use ndarray::{OwnedRepr, prelude::*};
use ndarray_linalg::*;
use num::{
    complex::{Complex64, c64},
    zero,
};
use num_traits::{ConstZero, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use rug::{
    Complex, Float,
    az::UnwrappedAs,
    ops::{Pow, PowAssign},
};
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

impl NetworkPortPoints for PortPoints<Complex64> {
    fn db(&self) -> PortPoints<f64> {
        Array1::<f64>::from_shape_fn(self.len(), |i| 20.0 * self[i].abs().log10())
    }

    fn deg(&self) -> PortPoints<f64> {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].arg() * 180.0 / PI)
    }

    fn im(&self) -> PortPoints<f64> {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].im())
    }

    fn mag(&self) -> PortPoints<f64> {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].abs())
    }

    // fn new_like(pt: &Points) -> PortPoints {
    //     Array1::<Complex64>::zeros(pt.len())
    // }

    fn rad(&self) -> PortPoints<f64> {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].arg())
    }

    fn re(&self) -> PortPoints<f64> {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].re())
    }
}
