#![allow(unused)]
use crate::frequency::Frequency;
use crate::impedance::ComplexNumberType;
use crate::math::*;
use crate::mycomplex::MyComplex;
use crate::myfloat::MyFloat;
use crate::network::{NetworkPoint, WaveType};
use crate::parameter::RFParameter;
use crate::point::{Point, PointComplex, PointFloat, Pointf64, Pt};
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

impl NetworkPoint<Pointf64, (f64, f64), Complex64> for Point {
    fn a_to_g(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.a_to_g() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn a_to_h(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.a_to_h() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn a_to_s(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[[i]].re, z0[[i]].im)));
        let val: PointComplex = self.into();
        match val.a_to_s(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn a_to_t(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.a_to_t(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn a_to_y(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.a_to_y() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn a_to_z(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.a_to_z() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn connect(&self, p1: usize, net: &Self, p2: usize) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.connect(p1, &net.into(), p2) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn db(&self) -> Pointf64 {
        Pointf64::from_shape_fn(self.dim(), |(j, k)| 20.0 * self[(j, k)].norm().log10())
    }

    fn deg(&self) -> Pointf64 {
        Pointf64::from_shape_fn(self.dim(), |(j, k)| self[(j, k)].arg() * 180.0 / PI)
    }

    fn from_db(data: &[(f64, f64)]) -> Self {
        let nports = (data.len() as f64).sqrt() as usize;
        Point::from_shape_fn((nports, nports), |(i, j)| {
            c64::from_polar(
                10_f64.powf(data[i * nports + j].0 / 20.0),
                f64::to_radians(data[i * nports + j].1),
            )
        })
    }

    fn from_ma(data: &[(f64, f64)]) -> Self {
        let nports = (data.len() as f64).sqrt() as usize;
        Point::from_shape_fn((nports, nports), |(i, j)| {
            c64::from_polar(
                data[i * nports + j].0,
                f64::to_radians(data[i * nports + j].1),
            )
        })
    }

    fn from_ri(data: &[(f64, f64)]) -> Self {
        let nports = (data.len() as f64).sqrt() as usize;
        Point::from_shape_fn((nports, nports), |(i, j)| {
            c64(data[i * nports + j].0, data[i * nports + j].1)
        })
    }

    fn g_to_a(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.g_to_a() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn g_to_h(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.g_to_h() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn g_to_s(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.g_to_s(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn g_to_t(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.g_to_t(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn g_to_y(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.g_to_y() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn g_to_z(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.g_to_z() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn h_to_a(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.h_to_a() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn h_to_g(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.h_to_g() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn h_to_s(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.h_to_s(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn h_to_t(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.h_to_t(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn h_to_y(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.h_to_y() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn h_to_z(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.h_to_z() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn im(&self) -> Pointf64 {
        Pointf64::from_shape_fn(self.dim(), |(j, k)| self[(j, k)].im)
    }

    fn is_reciprocal(&self) -> bool {
        !(self.nrows() != 2 || self.reciprocity().unwrap() != Pointf64::zeros((2, 2)))
    }

    fn mag(&self) -> Pointf64 {
        Pointf64::from_shape_fn(self.dim(), |(j, k)| self[(j, k)].norm())
    }

    fn new_like(pt: &Self) -> Self {
        Point::zeros((pt.nrows(), pt.ncols()))
    }

    fn rad(&self) -> Pointf64 {
        Pointf64::from_shape_fn(self.dim(), |(j, k)| self[(j, k)].arg())
    }

    fn re(&self) -> Pointf64 {
        Pointf64::from_shape_fn(self.dim(), |(j, k)| self[(j, k)].re)
    }

    fn reciprocity(&self) -> Option<Pointf64> {
        let nrows = self.nrows();
        if nrows != 2 {
            return None;
        }

        let diff = (self - self.t().to_owned()).into_inner();
        let mut out = Pointf64::zeros(self.dim());
        azip!((index (i,j), &diff in &diff) {
            out[[i,j]] = diff.abs();
        });

        Some(out)
    }

    fn s_to_a(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.s_to_a(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn s_to_g(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.s_to_g(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn s_to_h(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.s_to_h(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn s_to_s(&self, z0: &Array1<Complex64>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let val: PointComplex = self.into();
        let z0c = Array1::<MyComplex>::from_shape_fn(z0.dim(), |j| z0[[j]].into());
        match val.s_to_s(&z0c, from, to) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn s_to_t(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.s_to_t() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn s_to_y(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.s_to_y(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn s_to_z(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.s_to_z(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn t_to_a(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.t_to_a(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn t_to_g(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.t_to_g(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn t_to_h(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.t_to_h(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn t_to_s(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.t_to_s() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn t_to_y(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.t_to_y(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn t_to_z(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.t_to_z(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn y_to_a(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.y_to_a() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn y_to_g(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.y_to_g() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn y_to_h(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.y_to_h() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn y_to_s(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.y_to_s(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn y_to_t(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.y_to_t(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn y_to_z(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.y_to_z() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn z_to_a(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.z_to_a() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn z_to_g(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.z_to_g() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn z_to_h(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.z_to_h() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn z_to_s(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.z_to_s(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn z_to_t(&self, z0: &Array1<Complex64>) -> Option<Self> {
        let z0c = Array1::from_shape_fn(z0.len(), |i| MyComplex::from((z0[i].re, z0[i].im)));
        let val: PointComplex = self.into();
        match val.z_to_t(&z0c) {
            Some(x) => Some(x.into()),
            None => None,
        }
    }

    fn z_to_y(&self) -> Option<Self> {
        let val: PointComplex = self.into();
        match val.z_to_y() {
            Some(x) => Some(x.into()),
            None => None,
        }
    }
}
