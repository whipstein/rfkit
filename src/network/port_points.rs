#![allow(unused)]
use crate::frequency::Frequency;
use crate::impedance::ComplexNumberType;
use crate::math::*;
use crate::mycomplex::MyComplex;
use crate::myfloat::MyFloat;
use crate::parameter::RFParameter;
use crate::point::{Point, PointComplex};
use crate::unit::Unit;
use ndarray::OwnedRepr;
use ndarray::prelude::*;
use ndarray_linalg::*;
use num::complex::{Complex64, c64};
use num::zero;
use num_traits::{ConstZero, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use rug::az::UnwrappedAs;
use rug::ops::{Pow, PowAssign};
use rug::{Complex, Float};
use simple_error::SimpleError;
use std::error::Error;
use std::f64::consts::PI;
use std::iter::Iterator;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};
use std::process::Child;
use std::slice::Iter;
use std::{fmt, fs, mem, process};

use crate::network::{NetworkPortPoints, Points, PortPoints, PortPointsf64};

impl NetworkPortPoints for PortPoints {
    fn db(&self) -> PortPointsf64 {
        Array1::<f64>::from_shape_fn(self.len(), |i| 20.0 * self[i].abs().log10())
    }

    fn deg(&self) -> PortPointsf64 {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].arg() * 180.0 / PI)
    }

    fn im(&self) -> PortPointsf64 {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].im())
    }

    fn mag(&self) -> PortPointsf64 {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].abs())
    }

    // fn new_like(pt: &Points) -> PortPoints {
    //     Array1::<Complex64>::zeros(pt.len())
    // }

    fn rad(&self) -> PortPointsf64 {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].arg())
    }

    fn re(&self) -> PortPointsf64 {
        Array1::<f64>::from_shape_fn(self.len(), |i| self[i].re())
    }
}
