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

impl NetworkPoint<PointFloat, (MyFloat, MyFloat), MyComplex> for PointComplex {
    fn a_to_g(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = &self[[0, 0]];
        let b = &self[[0, 1]];
        let c = &self[[1, 0]];
        let d = &self[[1, 1]];

        let denom = a;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => c / denom,
                (0, 1) => -(a * d - b * c) / denom,
                (1, 0) => 1.0 / denom,
                (1, 1) => b / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn a_to_h(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = &self[[0, 0]];
        let b = &self[[0, 1]];
        let c = &self[[1, 0]];
        let d = &self[[1, 1]];

        let denom = d;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => b / denom,
                (0, 1) => (a * d - b * c) / denom,
                (1, 0) => -1.0 / denom,
                (1, 1) => c / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn a_to_s(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = &self[[0, 0]];
        let b = &self[[0, 1]];
        let c = &self[[1, 0]];
        let d = &self[[1, 1]];

        let denom = &(a * &z0[1] + b + c * &z0[0] * &z0[1] + d * &z0[0]);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => (a * &z0[1] + b - c * &z0[0].conj() * &z0[1] - d * &z0[0].conj()) / denom,
                (0, 1) => (2.0 * (a * d - b * c) * &z0sqrt) / denom,
                (1, 0) => (2.0 * &z0sqrt) / denom,
                (1, 1) => {
                    (-a * &z0[1].conj() + b - c * &z0[0] * &z0[1].conj() + d * &z0[0]) / denom
                }
                _ => 0.0 / denom,
            }
        }))
    }

    fn a_to_t(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = &self[[0, 0]];
        let b = &self[[0, 1]];
        let c = &self[[1, 0]];
        let d = &self[[1, 1]];

        let denom = &MyComplex::from(2.0 * (z0[0].real() * z0[1].real()).sqrt());
        if denom.is_zero() || denom.is_nan() {
            return None;
        }
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => {
                    ((4.0 * a * d - 4.0 * b * c) * z0[0].real() * z0[1].real()
                        + (((-(c * c * z0[0].clone()) - a * c) * &z0[0].conj()
                            + a * c * &z0[0]
                            + a * a)
                            * &z0[1]
                            + (-(c * d * &z0[0]) - a * d) * &z0[0].conj()
                            + b * c * &z0[0]
                            + a * b)
                            * &z0[1].conj()
                        + ((c * d * &z0[0] + b * c) * &z0[0].conj() - a * d * &z0[0] - a * b)
                            * &z0[1]
                        + (d * d * &z0[0] + b * d) * &z0[0].conj()
                        - b * d * &z0[0]
                        - b * b)
                        / (((c * &z0[0] + a) * &z0[1] + d * &z0[0] + b) * denom)
                }
                (0, 1) => -1.0 * ((c * &z0[0].conj() - a) * &z0[1] + d * &z0[0].conj() - b) / denom,
                (1, 0) => ((c * &z0[0] + a) * &z0[1].conj() - d * &z0[0] - b) / denom,
                (1, 1) => ((c * &z0[0] + a) * &z0[1] + d * &z0[0] + b) / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn a_to_y(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = &self[[0, 0]];
        let b = &self[[0, 1]];
        let c = &self[[1, 0]];
        let d = &self[[1, 1]];

        let denom = b;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => d / denom,
                (0, 1) => (b * c - a * d) / denom,
                (1, 0) => -1.0 / denom,
                (1, 1) => a / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn a_to_z(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = &self[[0, 0]];
        let b = &self[[0, 1]];
        let c = &self[[1, 0]];
        let d = &self[[1, 1]];

        let denom = c;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => a / denom,
                (0, 1) => (a * d - b * c) / denom,
                (1, 0) => 1.0 / denom,
                (1, 1) => d / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn connect(&self, p1: usize, net: &Self, p2: usize) -> Option<Self> {
        let pa = self.nrows();
        let pb = net.nrows();
        let nports = self.nrows() * net.nrows();
        let k = p1;
        let l = p2 + self.nrows();
        let mut matrix = PointComplex::from_shape_fn((nports, nports), |(i, j)| {
            if i < pa && j < pa {
                self[(i, j)].clone()
            } else if i >= pa && j >= pa {
                net[(i - pa, j - pa)].clone()
            } else {
                (0.0).into()
            }
        });

        let mut ext_i: Vec<usize> = vec![];
        for i in 0..nports {
            if i != k && i != l {
                ext_i.push(i);
            }
        }

        let akl = &(1.0 - &matrix[(k, l)]);
        let alk = &(1.0 - &matrix[(l, k)]);
        let akk = &matrix[(k, k)];
        let all = &matrix[(l, l)];
        let denom = &(akl * alk - akk * all);

        let mut out = PointComplex::zeros((nports - 2, nports - 2));
        for i in ext_i.iter() {
            for j in ext_i.iter() {
                let mut ii = *i;
                let mut jj = *j;
                if *i >= pa {
                    ii -= pa;
                }
                if *j >= pa {
                    jj -= pa;
                }
                out[(ii, jj)] = matrix[(*i, *j)].clone();
            }
        }

        let mut ake = Array1::zeros(nports - 2);
        let mut ale = Array1::zeros(nports - 2);
        let mut aek = Array1::zeros(nports - 2);
        let mut ael = Array1::zeros(nports - 2);
        let mut tmp_a = Array1::zeros(nports - 2);
        let mut tmp_b = Array1::zeros(nports - 2);
        for i in ext_i.iter() {
            let mut ii = *i;
            if *i >= pa {
                ii -= pa;
            }
            ake[(ii)] = matrix[(k, *i)].clone();
            ale[(ii)] = matrix[(l, *i)].clone();
            aek[(ii)] = matrix[(*i, k)].clone();
            ael[(ii)] = matrix[(*i, l)].clone();
            tmp_a[(ii)] = (&ael[(ii)] * alk + &aek[(ii)] * all) / denom;
            tmp_b[(ii)] = (&ael[(ii)] * akk + &aek[(ii)] * akl) / denom;
        }

        for ((i, j), val) in out.indexed_iter_mut() {
            val.add_assign(&ake[(j)] * &tmp_a[(i)] + &ale[(j)] * &tmp_b[(i)]);
        }

        Some(out)
    }

    fn db(&self) -> PointFloat {
        let mut pt = PointFloat::zeros((self.nrows(), self.ncols()));

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt[(i, j)] = (20.0 * self[(i, j)].abs().log10()).into();
            }
        }

        pt
    }

    fn deg(&self) -> PointFloat {
        let mut pt = PointFloat::zeros((self.nrows(), self.ncols()));

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt[(i, j)] = (self[(i, j)].arg() * 180.0 / PI).into();
            }
        }

        pt
    }

    fn from_db(data: &[(MyFloat, MyFloat)]) -> Self {
        let nports = (data.len() as f64).sqrt() as usize;
        PointComplex::from_shape_fn((nports, nports), |(i, j)| {
            MyComplex::from_polar(
                data[i * nports + j].0.db2mag(),
                data[i * nports + j].1.to_radians(),
            )
        })
    }

    fn from_ma(data: &[(MyFloat, MyFloat)]) -> Self {
        let nports = (data.len() as f64).sqrt() as usize;
        PointComplex::from_shape_fn((nports, nports), |(i, j)| {
            MyComplex::from_polar(
                data[i * nports + j].0.clone(),
                data[i * nports + j].1.to_radians(),
            )
        })
    }

    fn from_ri(data: &[(MyFloat, MyFloat)]) -> Self {
        let nports = (data.len() as f64).sqrt() as usize;
        PointComplex::from_shape_fn((nports, nports), |(i, j)| {
            MyComplex::new(
                data[i * nports + j].0.clone(),
                data[i * nports + j].1.clone(),
            )
        })
    }

    fn g_to_a(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = &self[[0, 0]];
        let g12 = &self[[0, 1]];
        let g21 = &self[[1, 0]];
        let g22 = &self[[1, 1]];

        let denom = g21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => 1.0 / denom,
                (0, 1) => g22 / denom,
                (1, 0) => g11 / denom,
                (1, 1) => (g11 * g22 - g12 * g21) / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn g_to_h(&self) -> Option<Self> {
        match self.try_inv() {
            Ok(x) => Some(PointComplex::from_shape_fn(x.dim(), |(j, k)| {
                x[[j, k]].clone()
            })),
            _ => None,
        }
    }

    fn g_to_s(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = &self[[0, 0]];
        let g12 = &self[[0, 1]];
        let g21 = &self[[1, 0]];
        let g22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &((g11 * &z0[0] + 1.0) * &z0[1] + (g11 * g22 - g12 * g21) * &z0[0] + g22);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => {
                    ((g11 * &z0[0].conj() - 1.0) * &z0[1] + (g11 * g22 - g12 * g21) * &z0[0].conj()
                        - g22)
                        / -denom
                }
                (0, 1) => -2.0 * g12 * &z0sqrt / denom,
                (1, 0) => 2.0 * g21 * &z0sqrt / denom,
                (1, 1) => {
                    ((g11 * &z0[0] + 1.0) * &z0[1].conj() + (g12 * g21 - g11 * g22) * &z0[0] - g22)
                        / -denom
                }
                _ => 0.0 / denom,
            }
        }))
    }

    fn g_to_t(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = &self[[0, 0]];
        let g12 = &self[[0, 1]];
        let g21 = &self[[1, 0]];
        let g22 = &self[[1, 1]];

        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();
        let denom = &(2.0 * g21 * &z0sqrt);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => {
                    ((4.0 * g12 * g21 * z0[0].real() * z0[1].real()
                        + (((g11 * g11 * &z0[0] + g11) * &z0[0].conj() - g11 * &z0[0] - 1.0)
                            * &z0[1]
                            + ((g11 * g11 * g22 - g11 * g12 * g21) * &z0[0] + g11 * g22
                                - g12 * g21)
                                * &z0[0].conj()
                            - g11 * g22 * &z0[0]
                            - g22)
                            * &z0[1].conj()
                        + (((g11 * g12 * g21 - g11 * g11 * g22) * &z0[0] - g11 * g22)
                            * &z0[0].conj()
                            + (g11 * g22 - g12 * g21) * &z0[0]
                            + g22)
                            * &z0[1]
                        + ((-1.0 * (g11 * g11 * g22 * g22) + 2.0 * g11 * g12 * g21 * g22
                            - g12 * g12 * g21 * g21)
                            * &z0[0]
                            - g11 * g22 * g22
                            + g12 * g21 * g22)
                            * &z0[0].conj()
                        + (g11 * g22 * g22 - g12 * g21 * g22) * &z0[0]
                        + g22 * g22)
                        / ((g11 * &z0[0] + 1.0) * &z0[1] + (g11 * g22 - g12 * g21) * &z0[0] + g22))
                        / -denom
                }
                (0, 1) => {
                    ((g11 * &z0[0].conj() - 1.0) * &z0[1] + (g11 * g22 - g12 * g21) * &z0[0].conj()
                        - g22)
                        / -denom
                }
                (1, 0) => {
                    ((g11 * &z0[0] + 1.0) * &z0[1].conj() + (g12 * g21 - g11 * g22) * &z0[0] - g22)
                        / denom
                }
                (1, 1) => {
                    ((g11 * &z0[0] + 1.0) * &z0[1] + (g11 * g22 - g12 * g21) * &z0[0] + g22) / denom
                }
                _ => 0.0 / denom,
            }
        }))
    }

    fn g_to_y(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = &self[[0, 0]];
        let g12 = &self[[0, 1]];
        let g21 = &self[[1, 0]];
        let g22 = &self[[1, 1]];

        let denom = g22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => (g11 * g22 - g12 * g21) / denom,
                (0, 1) => g12 / denom,
                (1, 0) => -g21 / denom,
                (1, 1) => 1.0 / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn g_to_z(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = &self[[0, 0]];
        let g12 = &self[[0, 1]];
        let g21 = &self[[1, 0]];
        let g22 = &self[[1, 1]];

        let denom = g11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => 1.0 / denom,
                (0, 1) => -g12 / denom,
                (1, 0) => g21 / denom,
                (1, 1) => (g11 * g22 - g12 * g21) / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn h_to_a(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = &self[[0, 0]];
        let h12 = &self[[0, 1]];
        let h21 = &self[[1, 0]];
        let h22 = &self[[1, 1]];

        let denom = h21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => -(h11 * h22 - h12 * h21) / denom,
                (0, 1) => -h11 / denom,
                (1, 0) => -h22 / denom,
                (1, 1) => -1.0 / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn h_to_g(&self) -> Option<Self> {
        match self.try_inv() {
            Ok(x) => Some(PointComplex::from_shape_fn(x.dim(), |(j, k)| {
                x[[j, k]].clone()
            })),
            _ => None,
        }
    }

    fn h_to_s(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = &self[[0, 0]];
        let h12 = &self[[0, 1]];
        let h21 = &self[[1, 0]];
        let h22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &((&z0[0] + h11) * (1.0 + h22 * &z0[1]) - h12 * h21 * &z0[1]);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => {
                    ((h11 - &z0[0].conj()) * (1.0 + h22 * &z0[1]) - h12 * h21 * &z0[1]) / denom
                }
                (0, 1) => 2.0 * h12 * &z0sqrt / denom,
                (1, 0) => -2.0 * h21 * &z0sqrt / denom,
                (1, 1) => {
                    ((&z0[0] + h11) * (1.0 - h22 * &z0[1].conj()) + h12 * h21 * &z0[1].conj())
                        / denom
                }
                _ => 0.0 / denom,
            }
        }))
    }

    fn h_to_t(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = &self[[0, 0]];
        let h12 = &self[[0, 1]];
        let h21 = &self[[1, 0]];
        let h22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &(2.0 * h21 * &z0sqrt);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => {
                    (2.0 * h21
                        * (4.0 * h12 * h21 * z0[0].real() * z0[1].real()
                            + (((h22 * h22 * &z0[0] + h11 * h22 * h22 - h12 * h21 * h22)
                                * &z0[0].conj()
                                + (h12 * h21 * h22 - h11 * h22 * h22) * &z0[0]
                                - h11 * h11 * h22 * h22
                                + 2.0 * h11 * h12 * h21 * h22
                                - h12 * h12 * h21 * h21)
                                * &z0[1]
                                + (h22 * &z0[0] + h11 * h22 - h12 * h21) * &z0[0].conj()
                                - h11 * h22 * &z0[0]
                                - h11 * h11 * h22
                                + h11 * h12 * h21)
                                * &z0[1].conj()
                            + ((-(h22 * &z0[0]) - h11 * h22) * &z0[0].conj()
                                + (h11 * h22 - h12 * h21) * &z0[0]
                                + h11 * h11 * h22
                                - h11 * h12 * h21)
                                * &z0[1]
                            + (-&z0[0] - h11) * &z0[0].conj()
                            + h11 * &z0[0]
                            + h11 * h11))
                        / ((2.0 * h21 * h22 * &z0[0] + 2.0 * h11 * h21 * h22
                            - 2.0 * h12 * h21 * h21)
                            * &z0[1]
                            + 2.0 * h21 * &z0[0]
                            + 2.0 * h11 * h21)
                        / denom
                }
                (0, 1) => {
                    ((h22 * &z0[0].conj() - h11 * h22 + h12 * h21) * &z0[1] + &z0[0].conj() - h11)
                        / denom
                }
                (1, 0) => {
                    ((h22 * &z0[0] + h11 * h22 - h12 * h21) * &z0[1].conj() - &z0[0] - h11) / -denom
                }
                (1, 1) => ((h22 * &z0[0] + h11 * h22 - h12 * h21) * &z0[1] + &z0[0] + h11) / -denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn h_to_y(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = &self[[0, 0]];
        let h12 = &self[[0, 1]];
        let h21 = &self[[1, 0]];
        let h22 = &self[[1, 1]];

        let denom = h11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => 1.0 / denom,
                (0, 1) => -h12 / denom,
                (1, 0) => h21 / denom,
                (1, 1) => (h11 * h22 - h12 * h21) / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn h_to_z(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = &self[[0, 0]];
        let h12 = &self[[0, 1]];
        let h21 = &self[[1, 0]];
        let h22 = &self[[1, 1]];

        let denom = h22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            match (j, k) {
                (0, 0) => (h11 * h22 - h12 * h21) / denom,
                (0, 1) => h12 / denom,
                (1, 0) => -h21 / denom,
                (1, 1) => 1.0 / denom,
                _ => 0.0 / denom,
            }
        }))
    }

    fn im(&self) -> PointFloat {
        let mut pt = PointFloat::zeros((self.nrows(), self.ncols()));

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt[(i, j)] = (self[(i, j)].imag()).into();
            }
        }

        pt
    }

    fn is_reciprocal(&self) -> bool {
        !(self.nrows() != 2 || self.reciprocity().unwrap() != PointFloat::zeros((2, 2)))
    }

    fn mag(&self) -> PointFloat {
        let mut pt = PointFloat::zeros((self.nrows(), self.ncols()));

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt[(i, j)] = (self[(i, j)].abs()).into();
            }
        }

        pt
    }

    fn new_like(pt: &Self) -> Self {
        PointComplex::zeros((pt.nrows(), pt.ncols()))
    }

    fn rad(&self) -> PointFloat {
        let mut pt = PointFloat::zeros((self.nrows(), self.ncols()));

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt[(i, j)] = (self[(i, j)].arg()).into();
            }
        }

        pt
    }

    fn re(&self) -> PointFloat {
        let mut pt = PointFloat::zeros((self.nrows(), self.ncols()));

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt[(i, j)] = (self[(i, j)].real()).into();
            }
        }

        pt
    }

    fn reciprocity(&self) -> Option<PointFloat> {
        // let nrows = self.nrows();
        // if nrows != 2 {
        //     return None;
        // }

        // let diff = self - self.t().to_owned();
        // let mut out = PointFloat::zeros(self.dim());
        // azip!((index (i,j), &diff in &diff) {
        //     out[[i,j]] = diff.abs();
        // });

        // Some(out)
        None
    }

    fn s_to_a(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = &self[[0, 0]];
        let s12 = &self[[0, 1]];
        let s21 = &self[[1, 0]];
        let s22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &(2.0 * s21 * &z0sqrt);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((&z0[0].conj() + s11 * &z0[0]) * (1.0 - s22) + s12 * s21 * &z0[0]) / denom;
        let x12 = ((&z0[0].conj() + s11 * &z0[0]) * (&z0[1].conj() + s22 * &z0[1])
            - s12 * s21 * &z0[0] * &z0[1])
            / denom;
        let x21 = ((1.0 - s11) * (1.0 - s22) - s12 * s21) / denom;
        let x22 = ((1.0 - s11) * (&z0[1].conj() + s22 * &z0[1]) + s12 * s21 * &z0[1]) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_g(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = &self[[0, 0]];
        let s12 = &self[[0, 1]];
        let s21 = &self[[1, 0]];
        let s22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &(4.0 * s12 * s21 * z0[0].real() * z0[1].real()
            + (((s11 - 1.0) * s22 - s12 * s21 - s11 + 1.0) * &z0[0].conj()
                + ((s11 * s11 - s11) * s22 - s11 * s12 * s21 - s11 * s11 + s11) * &z0[0])
                * &z0[1].conj()
            + (((s11 - 1.0) * s22 * s22 + (-(s12 * s21) - s11 + 1.0) * s22) * &z0[0].conj()
                + ((s11 * s11 - s11) * s22 * s22
                    + ((1.0 - 2.0 * s11) * s12 * s21 - s11 * s11 + s11) * s22
                    + s12 * s12 * s21 * s21
                    + (s11 - 1.0) * s12 * s21)
                    * &z0[0])
                * &z0[1]);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (((s11 * s11 - 2.0 * s11 + 1.0) * s22 + (1.0 - s11) * s12 * s21 - s11 * s11
            + 2.0 * s11
            - 1.0)
            * &z0[1].conj()
            + ((s11 * s11 - 2.0 * s11 + 1.0) * s22 * s22
                + ((2.0 - 2.0 * s11) * s12 * s21 - s11 * s11 + 2.0 * s11 - 1.0) * s22
                + s12 * s12 * s21 * s21
                + (s11 - 1.0) * s12 * s21)
                * &z0[1])
            / -denom;
        let x12 = (((2.0 * s11 - 2.0) * s12 * &z0[1].conj()
            + ((2.0 * s11 - 2.0) * s12 * s22 - 2.0 * s12 * s12 * s21) * &z0[1])
            * &z0sqrt)
            / denom;
        let x21 = (((2.0 * s11 - 2.0) * s21 * &z0[1].conj()
            + ((2.0 * s11 - 2.0) * s21 * s22 - 2.0 * s12 * s21 * s21) * &z0[1])
            * &z0sqrt)
            / -denom;
        let x22 = (((s11 - 1.0) * &z0[0].conj() + (s11 * s11 - s11) * &z0[0])
            * &z0[1].conj()
            * &z0[1].conj()
            + (((2.0 * s11 - 2.0) * s22 - s12 * s21) * &z0[0].conj()
                + ((2.0 * s11 * s11 - 2.0 * s11) * s22 + (1.0 - 2.0 * s11) * s12 * s21) * &z0[0])
                * &z0[1]
                * &z0[1].conj()
            + (((s11 - 1.0) * s22 * s22 - s12 * s21 * s22) * &z0[0].conj()
                + ((s11 * s11 - s11) * s22 * s22
                    + (1.0 - 2.0 * s11) * s12 * s21 * s22
                    + s12 * s12 * s21 * s21)
                    * &z0[0])
                * &z0[1]
                * &z0[1])
            / -denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_h(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = &self[[0, 0]];
        let s12 = &self[[0, 1]];
        let s21 = &self[[1, 0]];
        let s22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &((1.0 - s11) * (&z0[1].conj() + s22 * &z0[1]) + s12 * s21 * &z0[1]);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((&z0[0].conj() + s11 * &z0[0]) * (&z0[1].conj() + s22 * &z0[1])
            - s12 * s21 * &z0[0] * &z0[1])
            / denom;
        let x12 = 2.0 * s12 * &z0sqrt / denom;
        let x21 = -2.0 * s21 * &z0sqrt / denom;
        let x22 = ((1.0 - s11) * (1.0 - s22) - s12 * s21) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_s(&self, z0: &Array1<MyComplex>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let f = match from {
            WaveType::Power => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (1.0 / z0[j].real().sqrt()).into()
                } else {
                    (0.0).into()
                }
            }),
            WaveType::Pseudo => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (z0[j].abs() / z0[j].real().sqrt()).into()
                } else {
                    (0.0).into()
                }
            }),
            WaveType::Traveling => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k { z0[j].sqrt() } else { (0.0).into() }
            }),
        };
        let g = match from {
            WaveType::Power => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k { z0[j].clone() } else { (0.0).into() }
            }),
            WaveType::Pseudo => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (z0[j].abs() / (&z0[j] * z0[j].real().sqrt())).into()
                } else {
                    (0.0).into()
                }
            }),
            WaveType::Traveling => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    1.0 / z0[j].sqrt()
                } else {
                    (0.0).into()
                }
            }),
        };
        let id = PointComplex::eye(self.dim().0);

        let v = match from {
            WaveType::Power => {
                let val = g.conj() + &g.dot(&self);
                f.dot(&val)
            }
            _ => f.dot(&(&id + self)),
        };
        let i = match from {
            WaveType::Power => f.dot(&(&id - self)),
            _ => g.dot(&(&id - self)),
        };

        let f = match to {
            WaveType::Power => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (1.0 / (2.0 * z0[j].real().sqrt())).into()
                } else {
                    (0.0).into()
                }
            }),
            WaveType::Pseudo => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (z0[j].real().sqrt() / (2.0 * z0[j].abs())).into()
                } else {
                    (0.0).into()
                }
            }),
            WaveType::Traveling => PointComplex::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    1.0 / z0[j].sqrt()
                } else {
                    (0.0).into()
                }
            }),
        };
        let g = PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            if j == k { z0[j].clone() } else { (0.0).into() }
        });
        let a = f.dot(&(&v + &g.dot(&i)));
        let b = match to {
            WaveType::Power => f.dot(&(&v - &g.conj().dot(&i))),
            WaveType::Pseudo | WaveType::Traveling => f.dot(&(&v - &g.conj().dot(&i))),
        };

        Some(PointComplex::from_shape_fn(self.dim(), |(j, k)| {
            &b[[j, k]] / &a[[k, k]]
        }))
        // None
    }

    fn s_to_t(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = &self[[0, 0]];
        let s12 = &self[[0, 1]];
        let s21 = &self[[1, 0]];
        let s22 = &self[[1, 1]];

        let denom = s21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (s12 * s21 - s11 * s22) / denom;
        let x12 = s11 / denom;
        let x21 = -s22 / denom;
        let x22 = 1.0 / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_y(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        let id = PointComplex::eye(self.nrows());
        let sqz0inv = PointComplex::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
            if i == j {
                1.0 / z0[[i]].sqrt()
            } else {
                (0.0).into()
            }
        });

        let diff = &id - self;
        let sum = (&id + self).try_inv();

        match sum {
            Ok(x) => Some(sqz0inv.dot(&diff).dot(&x).dot(&sqz0inv)),
            _ => None,
        }
    }

    fn s_to_z(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        let id = PointComplex::eye(self.nrows());
        let sqz0 = PointComplex::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
            if i == j { z0[[i]].sqrt() } else { (0.0).into() }
        });

        let diff = (&id - self).try_inv();
        let sum = &id + self;

        match diff {
            Ok(x) => Some(sqz0.dot(&x).dot(&sum).dot(&sqz0)),
            _ => None,
        }
    }

    fn t_to_a(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = &self[[0, 0]];
        let t12 = &self[[0, 1]];
        let t21 = &self[[1, 0]];
        let t22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &(2.0 * &z0sqrt);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 + t21) * &z0[0].conj() + (t12 + t11) * &z0[0]) / denom;
        let x12 = ((t22 * &z0[0].conj() + t12 * &z0[0]) * &z0[1].conj()
            + (-t21 * &z0[0].conj() - t11 * &z0[0]) * &z0[1])
            / denom;
        let x21 = (t22 + t21 - t12 - t11) / denom;
        let x22 = ((t22 - t12) * &z0[1].conj() + (t11 - t21) * &z0[1]) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_g(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = &self[[0, 0]];
        let t12 = &self[[0, 1]];
        let t21 = &self[[1, 0]];
        let t22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &((4.0 * t11 * t22 - 4.0 * t12 * t21) * z0[0].real() * z0[1].real()
            + ((t22 * t22 + (t21 - t12 - t11) * t22) * &z0[0].conj()
                + (t12 * t22 + t12 * t21 - t12 * t12 - t11 * t12) * &z0[0])
                * &z0[1].conj()
            + ((-(t21 * t22) - t21 * t21 + (t12 + t11) * t21) * &z0[0].conj()
                + (-(t11 * t22) - t11 * t21 + t11 * t12 + t11 * t11) * &z0[0])
                * &z0[1]);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 * t22 + (t21 - 2.0 * t12 - t11) * t22 - t12 * t21 + t12 * t12 + t11 * t12)
            * &z0[1].conj()
            + ((t11 - t21) * t22 - t21 * t21 + (t12 + 2.0 * t11) * t21 - t11 * t12 - t11 * t11)
                * &z0[1])
            / denom;
        let x12 = (((2.0 * t11 * t22 * t22
            + (-2.0 * t12 * t21 - 2.0 * t11 * t12) * t22
            + 2.0 * t12 * t12 * t21)
            * &z0[1].conj()
            + ((2.0 * t11 * t11 - 2.0 * t11 * t21) * t22 + 2.0 * t12 * t21 * t21
                - 2.0 * t11 * t12 * t21)
                * &z0[1])
            * &z0sqrt)
            / -denom;
        let x21 = (((2.0 * t22 - 2.0 * t12) * &z0[1].conj() + (2.0 * t11 - 2.0 * t21) * &z0[1])
            * &z0sqrt)
            / denom;
        let x22 = (((t22 * t22 - t12 * t22) * &z0[0].conj() + (t12 * t22 - t12 * t12) * &z0[0])
            * &z0[1].conj()
            * &z0[1].conj()
            + (((t11 - 2.0 * t21) * t22 + t12 * t21) * &z0[0].conj()
                + (-(t11 * t22) - t12 * t21 + 2.0 * t11 * t12) * &z0[0])
                * &z0[1]
                * &z0[1].conj()
            + ((t21 * t21 - t11 * t21) * &z0[0].conj() + (t11 * t21 - t11 * t11) * &z0[0])
                * &z0[1]
                * &z0[1])
            / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_h(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = &self[[0, 0]];
        let t12 = &self[[0, 1]];
        let t21 = &self[[1, 0]];
        let t22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &((t22 - t12) * &z0[1].conj() + (t11 - t21) * &z0[1]);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 * &z0[0].conj() + t12 * &z0[0]) * &z0[1].conj()
            + (-t21 * &z0[0].conj() - t11 * &z0[0]) * &z0[1])
            / denom;
        let x12 = ((2.0 * t11 * t22 - 2.0 * t12 * t21) * &z0sqrt) / denom;
        let x21 = -2.0 * &z0sqrt / denom;
        let x22 = (t22 + t21 - t12 - t11) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_s(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = &self[[0, 0]];
        let t12 = &self[[0, 1]];
        let t21 = &self[[1, 0]];
        let t22 = &self[[1, 1]];

        let denom = t22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = t12 / denom;
        let x12 = (t11 * t22 - t12 * t21) / denom;
        let x21 = 1.0 / denom;
        let x22 = -t21 / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_y(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = &self[[0, 0]];
        let t12 = &self[[0, 1]];
        let t21 = &self[[1, 0]];
        let t22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &((t22 * &z0[0].conj() + t12 * &z0[0]) * &z0[1].conj()
            + (-t21 * &z0[0].conj() - t11 * &z0[0]) * &z0[1]);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 - t12) * &z0[1].conj() + (t11 - t21) * &z0[1]) / denom;
        let x12 = ((2.0 * t11 * t22 - 2.0 * t12 * t21) * -&z0sqrt) / denom;
        let x21 = -2.0 * &z0sqrt / denom;
        let x22 = ((t22 + t21) * &z0[0].conj() + (t12 + t11) * &z0[0]) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_z(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = &self[[0, 0]];
        let t12 = &self[[0, 1]];
        let t21 = &self[[1, 0]];
        let t22 = &self[[1, 1]];

        let denom = &(t22 + t21 - t12 - t11);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 + t21 + t12 + t11) * &z0[0]) / denom;
        let x12 = ((2.0 * t11 * t22 - 2.0 * t12 * t21) * &z0[0].sqrt() * &z0[1].sqrt()) / denom;
        let x21 = (2.0 * &z0[0].sqrt() * &z0[1].sqrt()) / denom;
        let x22 = ((t22 - t21 - t12 + t11) * &z0[1]) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_a(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = &self[[0, 0]];
        let y12 = &self[[0, 1]];
        let y21 = &self[[1, 0]];
        let y22 = &self[[1, 1]];

        let denom = y21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = -y22 / denom;
        let x12 = -1.0 / denom;
        let x21 = -(y11 * y22 - y12 * y21) / denom;
        let x22 = -y11 / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_g(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = &self[[0, 0]];
        let y12 = &self[[0, 1]];
        let y21 = &self[[1, 0]];
        let y22 = &self[[1, 1]];

        let denom = y22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (y11 * y22 - y12 * y21) / denom;
        let x12 = y12 / denom;
        let x21 = -y21 / denom;
        let x22 = 1.0 / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_h(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = &self[[0, 0]];
        let y12 = &self[[0, 1]];
        let y21 = &self[[1, 0]];
        let y22 = &self[[1, 1]];

        let denom = y11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = 1.0 / denom;
        let x12 = -y12 / denom;
        let x21 = y21 / denom;
        let x22 = (y11 * y22 - y12 * y21) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_s(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        let id = PointComplex::eye(self.nrows());
        let sqz0 = PointComplex::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
            if i == j { z0[[i]].sqrt() } else { (0.0).into() }
        });

        let diff = &id - &sqz0.dot(self).dot(&sqz0);
        let sum = (&id + &sqz0.dot(self).dot(&sqz0)).try_inv();

        match sum {
            Ok(x) => Some(x.dot(&diff)),
            _ => None,
        }
    }

    fn y_to_t(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = &self[[0, 0]];
        let y12 = &self[[0, 1]];
        let y21 = &self[[1, 0]];
        let y22 = &self[[1, 1]];
        let z0sqrt = MyComplex::from((z0[0].real() * z0[1].real())).sqrt();

        let denom = &(2.0 * y21 * &z0[0].sqrt() * &z0[1].sqrt());
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (((y11 * y22 - y12 * y21) * &z0[0] - y22) * &z0[1] - y11 * &z0[0] + 1.0) / denom;
        let x12 = (((y11 * y22 - y12 * y21) * &z0[0] - y22) * &z0[1] + y11 * &z0[0] - 1.0) / denom;
        let x21 =
            (y11 * &z0[0] + 1.0 - (((y11 * y22 - y12 * y21) * &z0[0] + y22) * &z0[1])) / denom;
        let x22 =
            ((((y11 * y22 - y12 * y21) * &z0[0] + y22) * &z0[1]) + y11 * &z0[0] + 1.0) / -denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_z(&self) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        match self.try_inv() {
            Ok(val) => Some(val),
            _ => None,
        }
    }

    fn z_to_a(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = &self[[0, 0]];
        let z12 = &self[[0, 1]];
        let z21 = &self[[1, 0]];
        let z22 = &self[[1, 1]];

        let denom = z21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = z11 / denom;
        let x12 = (z11 * z22 - z12 * z21) / denom;
        let x21 = 1.0 / denom;
        let x22 = z22 / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn z_to_g(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = &self[[0, 0]];
        let z12 = &self[[0, 1]];
        let z21 = &self[[1, 0]];
        let z22 = &self[[1, 1]];

        let denom = z11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = 1.0 / denom;
        let x12 = -z12 / denom;
        let x21 = z21 / denom;
        let x22 = (z11 * z22 - z12 * z21) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn z_to_h(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = &self[[0, 0]];
        let z12 = &self[[0, 1]];
        let z21 = &self[[1, 0]];
        let z22 = &self[[1, 1]];

        let denom = z22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (z11 * z22 - z12 * z21) / denom;
        let x12 = z12 / denom;
        let x21 = -z21 / denom;
        let x22 = 1.0 / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn z_to_s(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        let id = PointComplex::eye(self.nrows());
        let sqz0inv = PointComplex::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
            if i == j {
                1.0 / z0[[i]].sqrt()
            } else {
                (0.0).into()
            }
        });

        let diff = &sqz0inv.dot(self).dot(&sqz0inv) - &id;
        let sum = (&sqz0inv.dot(self).dot(&sqz0inv) + &id).try_inv();

        match sum {
            Ok(x) => Some(x.dot(&diff)),
            _ => None,
        }
    }

    fn z_to_t(&self, z0: &Array1<MyComplex>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = &self[[0, 0]];
        let z12 = &self[[0, 1]];
        let z21 = &self[[1, 0]];
        let z22 = &self[[1, 1]];

        let denom = &(2.0 * &z0[0].sqrt() * &z0[1].sqrt() * z21);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (z12 * z21 + &z0[1] * z11 - &z0[0] * &z0[1] - ((z11 - &z0[0]) * z22)) / denom;
        let x12 = ((z11 - &z0[0]) * z22 - z12 * z21 + &z0[1] * z11 - &z0[0] * &z0[1]) / denom;
        let x21 = (z12 * z21 + &z0[1] * z11 + &z0[0] * &z0[1] - ((z11 + &z0[0]) * z22)) / denom;
        let x22 = ((z11 + &z0[0]) * z22 - z12 * z21 + &z0[1] * z11 + &z0[0] * &z0[1]) / denom;

        let out: Option<Self> = Some(PointComplex::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn z_to_y(&self) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        match self.try_inv() {
            Ok(val) => Some(val),
            _ => None,
        }
    }
}
