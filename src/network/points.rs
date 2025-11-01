#![allow(unused)]
use crate::frequency::Frequency;
use crate::impedance::ComplexNumberType;
use crate::math::*;
use crate::mycomplex::MyComplex;
use crate::myfloat::MyFloat;
use crate::network::{NetworkPoint, WaveType};
use crate::parameter::RFParameter;
use crate::point::{Point, Pt};
use crate::points::{Points, Pts};
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

impl NetworkPoint<Points<f64>, &[(f64, f64)], Complex64> for Points<Complex64> {
    fn a_to_g(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).a_to_g() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_h(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).a_to_h() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_s(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).a_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_t(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).a_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_y(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).a_to_y() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_z(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).a_to_z() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn connect(&self, p1: usize, net: &Points<Complex64>, p2: usize) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).connect(
                p1,
                &Point::<Complex64>::new(net.slice(s![i, .., ..]).to_owned()),
                p2,
            ) {
                Some(x) => x,
                _ => return None,
            };

            pt.assign(&val.inner());
        }
        Some(out)
    }

    fn db(&self) -> Points<f64> {
        let mut out = Points::<f64>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned())
                    .db()
                    .view(),
            );
        }

        out
    }

    fn deg(&self) -> Points<f64> {
        let mut out = Points::<f64>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned())
                    .deg()
                    .view(),
            );
        }

        out
    }

    fn from_db(data: &[&[(f64, f64)]]) -> Points<Complex64> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::from_shape_fn((npoints, nports, nports), |(i, j, k)| {
            c64::from_polar(
                10_f64.powf(data[i][j * nports + k].0 / 20.0),
                f64::to_radians(data[i][j * nports + k].1),
            )
        })
    }

    fn from_ma(data: &[&[(f64, f64)]]) -> Points<Complex64> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::from_shape_fn((npoints, nports, nports), |(i, j, k)| {
            c64::from_polar(
                data[i][j * nports + k].0,
                f64::to_radians(data[i][j * nports + k].1),
            )
        })
    }

    fn from_ri(data: &[&[(f64, f64)]]) -> Points<Complex64> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::from_shape_fn((npoints, nports, nports), |(i, j, k)| {
            c64(data[i][j * nports + k].0, data[i][j * nports + k].1)
        })
    }

    // fn is_reciprocal(&self) -> bool {
    //     if self.nrows() != 2 || self.reciprocity().unwrap() != Pointf64::zeros(2, 2) {
    //         return false;
    //     } else {
    //         return true;
    //     }
    // }

    // fn from_row_iterator(data: Vec<Complex64>) -> Point {
    //     Point::<Complex64>::from_row_iterator(data.into_iter())
    // }

    // fn from_vec(data: Vec<Complex64>) -> Point {
    //     Point::<Complex64>::from_vec(data)
    // }

    fn g_to_a(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).g_to_a() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_h(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).g_to_h() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_s(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).g_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_t(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).g_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_y(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).g_to_y() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_z(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).g_to_z() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_a(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).h_to_a() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_g(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).h_to_g() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_s(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).h_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_t(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).h_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_y(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).h_to_y() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_z(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).h_to_z() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    // fn identity() -> Point {
    //     Point::<Complex64>::one()
    // }

    fn im(&self) -> Points<f64> {
        let mut out = Points::<f64>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned())
                    .im()
                    .view(),
            );
        }

        out
    }

    fn is_reciprocal(&self) -> bool {
        true
    }

    fn mag(&self) -> Points<f64> {
        let mut out = Points::<f64>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned())
                    .mag()
                    .view(),
            );
        }

        out
    }

    fn new_like(pt: &Points<Complex64>) -> Points<Complex64> {
        Points::zeros(pt.dim())
    }

    // fn ones() -> Point {
    //     let mut val = Point::<Complex64>::zeros();
    //     val.fill(c64(1.0, 0.0));
    //     val
    // }

    fn rad(&self) -> Points<f64> {
        let mut out = Points::<f64>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned())
                    .rad()
                    .view(),
            );
        }

        out
    }

    fn re(&self) -> Points<f64> {
        let mut out = Points::<f64>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned())
                    .re()
                    .view(),
            );
        }

        out
    }

    fn reciprocity(&self) -> Option<Points<f64>> {
        let mut out = Points::<f64>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).reciprocity() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }
        Some(out)
    }

    fn s_to_a(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).s_to_a(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_g(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).s_to_g(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_h(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).s_to_h(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_s(&self, z0: &Array1<c64>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_s(z0, from, to)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_t(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).s_to_t() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_y(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).s_to_y(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_z(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).s_to_z(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_a(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).t_to_a(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_g(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).t_to_g(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_h(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).t_to_h(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_s(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).t_to_s() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_y(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).t_to_y(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_z(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).t_to_z(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_a(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).y_to_a() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_g(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).y_to_g() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_h(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).y_to_h() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_s(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).y_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_t(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).y_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_z(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).y_to_z() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_a(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).z_to_a() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_g(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).z_to_g() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_h(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).z_to_h() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_s(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).z_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_t(&self, z0: &Array1<Complex64>) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).z_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_y(&self) -> Option<Points<Complex64>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<Complex64>::new(self.slice(s![i, .., ..]).to_owned()).z_to_y() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }
}

impl NetworkPoint<Points<MyFloat>, &[(MyFloat, MyFloat)], MyComplex> for Points<MyComplex> {
    fn a_to_g(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).a_to_g() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_h(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).a_to_h() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_s(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).a_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_t(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).a_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_y(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).a_to_y() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_z(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).a_to_z() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn connect(&self, p1: usize, net: &Points<MyComplex>, p2: usize) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).connect(
                p1,
                &Point::<MyComplex>::new(net.slice(s![i, .., ..]).to_owned()),
                p2,
            ) {
                Some(x) => x,
                _ => return None,
            };

            pt.assign(&val.inner());
        }
        Some(out)
    }

    fn db(&self) -> Points<MyFloat> {
        let mut out = Points::<MyFloat>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned())
                    .db()
                    .view(),
            );
        }

        out
    }

    fn deg(&self) -> Points<MyFloat> {
        let mut out = Points::<MyFloat>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned())
                    .deg()
                    .view(),
            );
        }

        out
    }

    fn from_db(data: &[&[(MyFloat, MyFloat)]]) -> Points<MyComplex> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::from_shape_fn((npoints, nports, nports), |(i, j, k)| {
            MyComplex::from_polar(
                &MyFloat::new(10_f64).pow(&data[i][j * nports + k].0 / 20.0),
                &MyFloat::to_radians(&data[i][j * nports + k].1),
            )
        })
    }

    fn from_ma(data: &[&[(MyFloat, MyFloat)]]) -> Points<MyComplex> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::from_shape_fn((npoints, nports, nports), |(i, j, k)| {
            MyComplex::from_polar(
                &data[i][j * nports + k].0,
                &MyFloat::to_radians(&data[i][j * nports + k].1),
            )
        })
    }

    fn from_ri(data: &[&[(MyFloat, MyFloat)]]) -> Points<MyComplex> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::from_shape_fn((npoints, nports, nports), |(i, j, k)| {
            MyComplex::from_tuple(&data[i][j * nports + k])
        })
    }

    // fn is_reciprocal(&self) -> bool {
    //     if self.nrows() != 2 || self.reciprocity().unwrap() != Pointf64::zeros(2, 2) {
    //         return false;
    //     } else {
    //         return true;
    //     }
    // }

    // fn from_row_iterator(data: Vec<MyComplex>) -> Point {
    //     Point::<MyComplex>::from_row_iterator(data.into_iter())
    // }

    // fn from_vec(data: Vec<MyComplex>) -> Point {
    //     Point::<MyComplex>::from_vec(data)
    // }

    fn g_to_a(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).g_to_a() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_h(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).g_to_h() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_s(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).g_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_t(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).g_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_y(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).g_to_y() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_z(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).g_to_z() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_a(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).h_to_a() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_g(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).h_to_g() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_s(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).h_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_t(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).h_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_y(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).h_to_y() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_z(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).h_to_z() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    // fn identity() -> Point {
    //     Point::<MyComplex>::one()
    // }

    fn im(&self) -> Points<MyFloat> {
        let mut out = Points::<MyFloat>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned())
                    .im()
                    .view(),
            );
        }

        out
    }

    fn is_reciprocal(&self) -> bool {
        true
    }

    fn mag(&self) -> Points<MyFloat> {
        let mut out = Points::<MyFloat>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned())
                    .mag()
                    .view(),
            );
        }

        out
    }

    fn new_like(pt: &Points<MyComplex>) -> Points<MyComplex> {
        Points::zeros(pt.dim())
    }

    // fn ones() -> Point {
    //     let mut val = Point::<MyComplex>::zeros();
    //     val.fill(c64(1.0, 0.0));
    //     val
    // }

    fn rad(&self) -> Points<MyFloat> {
        let mut out = Points::<MyFloat>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned())
                    .rad()
                    .view(),
            );
        }

        out
    }

    fn re(&self) -> Points<MyFloat> {
        let mut out = Points::<MyFloat>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned())
                    .re()
                    .view(),
            );
        }

        out
    }

    fn reciprocity(&self) -> Option<Points<MyFloat>> {
        let mut out = Points::<MyFloat>::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).reciprocity() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }
        Some(out)
    }

    fn s_to_a(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).s_to_a(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_g(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).s_to_g(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_h(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).s_to_h(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_s(&self, z0: &Array1<MyComplex>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_s(z0, from, to)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_t(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).s_to_t() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_y(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).s_to_y(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_z(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).s_to_z(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_a(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).t_to_a(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_g(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).t_to_g(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_h(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).t_to_h(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_s(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).t_to_s() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_y(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).t_to_y(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_z(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).t_to_z(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_a(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).y_to_a() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_g(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).y_to_g() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_h(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).y_to_h() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_s(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).y_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_t(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).y_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_z(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).y_to_z() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_a(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).z_to_a() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_g(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).z_to_g() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_h(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).z_to_h() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_s(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).z_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_t(&self, z0: &Array1<MyComplex>) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).z_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_y(&self) -> Option<Points<MyComplex>> {
        let mut out = Points::zeros(self.dim());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Point::<MyComplex>::new(self.slice(s![i, .., ..]).to_owned()).z_to_y() {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }
}
