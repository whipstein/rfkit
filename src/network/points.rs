#![allow(unused)]
use crate::{
    frequency::Frequency,
    impedance::ComplexNumberType,
    math::*,
    mycomplex::MyComplex,
    myfloat::MyFloat,
    network::{NetworkPoint, WaveType},
    parameter::RFParameter,
    pts::{Points, Pts},
    unit::Unit,
};
use ndarray::{Dimension, IntoDimension, OwnedRepr, prelude::*};
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

impl NetworkPoint<Complex64, Ix3> for Points<Complex64, Ix3> {
    type Tuple<'a> = &'a [(f64, f64)];

    fn a_to_g(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_h(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_s(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_t(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_y(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_z(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn connect(
        &self,
        p1: usize,
        net: &Points<Complex64, Ix3>,
        p2: usize,
    ) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .connect(
                    p1,
                    &Points::<Complex64, Ix2>::new(net.slice(s![i, .., ..]).to_owned()),
                    p2,
                ) {
                Some(x) => x,
                _ => return None,
            };

            pt.assign(&val.inner());
        }
        Some(out)
    }

    fn db(&self) -> Points<f64, Ix3> {
        let mut out = Points::<f64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .db()
                    .view(),
            );
        }

        out
    }

    fn deg(&self) -> Points<f64, Ix3> {
        let mut out = Points::<f64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .deg()
                    .view(),
            );
        }

        out
    }

    fn from_db(data: &[Self::Tuple<'_>]) -> Points<Complex64, Ix3> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::<Complex64, Ix3>::from_shape_fn(
            (npoints, nports, nports).into_dimension(),
            |(i, j, k)| {
                c64::from_polar(
                    10_f64.powf(data[i][j * nports + k].0 / 20.0),
                    f64::to_radians(data[i][j * nports + k].1),
                )
            },
        )
    }

    fn from_ma(data: &[Self::Tuple<'_>]) -> Points<Complex64, Ix3> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::<Complex64, Ix3>::from_shape_fn(
            (npoints, nports, nports).into_dimension(),
            |(i, j, k)| {
                c64::from_polar(
                    data[i][j * nports + k].0,
                    f64::to_radians(data[i][j * nports + k].1),
                )
            },
        )
    }

    fn from_ri(data: &[Self::Tuple<'_>]) -> Points<Complex64, Ix3> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::<Complex64, Ix3>::from_shape_fn(
            (npoints, nports, nports).into_dimension(),
            |(i, j, k)| c64(data[i][j * nports + k].0, data[i][j * nports + k].1),
        )
    }

    // fn is_reciprocal(&self) -> bool {
    //     if self.nrows() != 2 || self.reciprocity().unwrap() != Pointf64::zeros(2, 2) {
    //         return false;
    //     } else {
    //         return true;
    //     }
    // }

    // fn from_row_iterator(data: Vec<Complex64>) -> Point {
    //     Points::<Complex64, Ix2>::from_row_iterator(data.into_iter())
    // }

    // fn from_vec(data: Vec<Complex64>) -> Point {
    //     Points::<Complex64, Ix2>::from_vec(data)
    // }

    fn g_to_a(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_h(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_s(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_t(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_y(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_z(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_a(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_g(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_s(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_t(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_y(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_z(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    // fn identity() -> Point {
    //     Points::<Complex64, Ix2>::one()
    // }

    fn im(&self) -> Points<f64, Ix3> {
        let mut out = Points::<f64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .im()
                    .view(),
            );
        }

        out
    }

    fn is_reciprocal(&self) -> bool {
        true
    }

    fn mag(&self) -> Points<f64, Ix3> {
        let mut out = Points::<f64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .mag()
                    .view(),
            );
        }

        out
    }

    fn new_like(pt: &Points<Complex64, Ix3>) -> Points<Complex64, Ix3> {
        Points::<Complex64, Ix3>::zeros(pt.dim().into_dimension())
    }

    // fn ones() -> Point {
    //     let mut val = Points::<Complex64, Ix2>::zeros();
    //     val.fill(c64(1.0, 0.0));
    //     val
    // }

    fn rad(&self) -> Points<f64, Ix3> {
        let mut out = Points::<f64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .rad()
                    .view(),
            );
        }

        out
    }

    fn re(&self) -> Points<f64, Ix3> {
        let mut out = Points::<f64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .re()
                    .view(),
            );
        }

        out
    }

    fn reciprocity(&self) -> Option<Points<f64, Ix3>> {
        let mut out = Points::<f64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .reciprocity()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }
        Some(out)
    }

    fn s_to_a(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_a(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_g(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_g(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_h(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_h(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_s(&self, z0: ArrayView1<c64>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_s(z0, from, to)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_t(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_t()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_y(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_y(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_z(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_z(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_a(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_a(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_g(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_g(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_h(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_h(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_s(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_s()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_y(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_y(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_z(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_z(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_a(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_g(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_h(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_s(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_t(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_z(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_a(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_g(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_h(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_s(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_t(&self, z0: ArrayView1<Complex64>) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_y(&self) -> Option<Points<Complex64, Ix3>> {
        let mut out = Points::<Complex64, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<Complex64, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }
}

impl NetworkPoint<MyComplex, Ix3> for Points<MyComplex, Ix3> {
    type Tuple<'a> = &'a [(MyFloat, MyFloat)];

    fn a_to_g(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_h(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_s(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_t(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_y(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_z(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .a_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn connect(
        &self,
        p1: usize,
        net: &Points<MyComplex, Ix3>,
        p2: usize,
    ) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .connect(
                    p1,
                    &Points::<MyComplex, Ix2>::new(net.slice(s![i, .., ..]).to_owned()),
                    p2,
                ) {
                Some(x) => x,
                _ => return None,
            };

            pt.assign(&val.inner());
        }
        Some(out)
    }

    fn db(&self) -> Points<MyFloat, Ix3> {
        let mut out = Points::<MyFloat, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .db()
                    .view(),
            );
        }

        out
    }

    fn deg(&self) -> Points<MyFloat, Ix3> {
        let mut out = Points::<MyFloat, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .deg()
                    .view(),
            );
        }

        out
    }

    fn from_db(data: &[Self::Tuple<'_>]) -> Points<MyComplex, Ix3> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::<MyComplex, Ix3>::from_shape_fn(
            (npoints, nports, nports).into_dimension(),
            |(i, j, k)| {
                MyComplex::from_polar(
                    &MyFloat::new(10_f64).pow(&data[i][j * nports + k].0 / 20.0),
                    &MyFloat::to_radians(&data[i][j * nports + k].1),
                )
            },
        )
    }

    fn from_ma(data: &[Self::Tuple<'_>]) -> Points<MyComplex, Ix3> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::<MyComplex, Ix3>::from_shape_fn(
            (npoints, nports, nports).into_dimension(),
            |(i, j, k)| {
                MyComplex::from_polar(
                    &data[i][j * nports + k].0,
                    &MyFloat::to_radians(&data[i][j * nports + k].1),
                )
            },
        )
    }

    fn from_ri(data: &[Self::Tuple<'_>]) -> Points<MyComplex, Ix3> {
        let nports = (data[0].len() as f64).sqrt() as usize;
        let npoints = data.len();
        Points::<MyComplex, Ix3>::from_shape_fn(
            (npoints, nports, nports).into_dimension(),
            |(i, j, k)| MyComplex::from_tuple(&data[i][j * nports + k]),
        )
    }

    // fn is_reciprocal(&self) -> bool {
    //     if self.nrows() != 2 || self.reciprocity().unwrap() != Pointf64::zeros(2, 2) {
    //         return false;
    //     } else {
    //         return true;
    //     }
    // }

    // fn from_row_iterator(data: Vec<MyComplex>) -> Point {
    //     Points::<MyComplex, Ix2>::from_row_iterator(data.into_iter())
    // }

    // fn from_vec(data: Vec<MyComplex>) -> Point {
    //     Points::<MyComplex, Ix2>::from_vec(data)
    // }

    fn g_to_a(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_h(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_s(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_t(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_y(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_z(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .g_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_a(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_g(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_s(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_t(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_y(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_z(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .h_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    // fn identity() -> Point {
    //     Points::<MyComplex, Ix2>::one()
    // }

    fn im(&self) -> Points<MyFloat, Ix3> {
        let mut out = Points::<MyFloat, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .im()
                    .view(),
            );
        }

        out
    }

    fn is_reciprocal(&self) -> bool {
        true
    }

    fn mag(&self) -> Points<MyFloat, Ix3> {
        let mut out = Points::<MyFloat, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .mag()
                    .view(),
            );
        }

        out
    }

    fn new_like(pt: &Points<MyComplex, Ix3>) -> Points<MyComplex, Ix3> {
        Points::<MyComplex, Ix3>::zeros(pt.dim().into_dimension())
    }

    // fn ones() -> Point {
    //     let mut val = Points::<MyComplex, Ix2>::zeros();
    //     val.fill(c64(1.0, 0.0));
    //     val
    // }

    fn rad(&self) -> Points<MyFloat, Ix3> {
        let mut out = Points::<MyFloat, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .rad()
                    .view(),
            );
        }

        out
    }

    fn re(&self) -> Points<MyFloat, Ix3> {
        let mut out = Points::<MyFloat, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            pt.assign(
                &Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                    .re()
                    .view(),
            );
        }

        out
    }

    fn reciprocity(&self) -> Option<Points<MyFloat, Ix3>> {
        let mut out = Points::<MyFloat, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .reciprocity()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }
        Some(out)
    }

    fn s_to_a(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_a(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_g(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_g(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_h(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_h(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_s(&self, z0: ArrayView1<MyComplex>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_s(z0, from, to)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_t(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_t()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_y(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_y(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_z(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_z(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_a(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_a(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_g(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_g(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_h(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_h(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_s(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_s()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_y(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_y(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_z(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .t_to_z(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_a(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_g(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_h(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_s(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_t(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_z(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .y_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_a(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_g(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_h(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_s(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_s(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_t(&self, z0: ArrayView1<MyComplex>) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_t(z0)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_y(&self) -> Option<Points<MyComplex, Ix3>> {
        let mut out = Points::<MyComplex, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<MyComplex, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .z_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }
}
