#![allow(unused)]
use crate::{
    consts::MathConst,
    frequency::Frequency,
    impedance::ComplexNumberType,
    math::*,
    network::{NetworkPoint, WaveType},
    num::{ComplexScalar, RealScalar, ScalarConst},
    parameter::RFParameter,
    pts::{Matrix, MatrixComplex, MatrixReal, Points, Pts, PtsComplex, PtsReal},
    unit::Unit,
};
use ndarray::{OwnedRepr, linalg::Dot, prelude::*};
use ndarray_linalg::*;
use num::FromPrimitive;
use num_complex::{Complex, Complex64, ComplexFloat, c64};
use num_traits::{ConstZero, Float, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use simple_error::SimpleError;
use std::{
    error::Error,
    f64::consts::PI,
    fmt, fs,
    iter::Iterator,
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
    process,
    process::Child,
    slice::Iter,
};
use twofloat::TwoFloat;

impl<T> NetworkPoint<T, Ix2> for Points<T, Ix2>
where
    T: ComplexScalar + From<T::Real>,
    <T as ComplexFloat>::Real: RealScalar + MathConst,
{
    type Size = (usize, usize);
    type Tuple<'a>
        = (T::Real, T::Real)
    where
        T: 'a;

    fn a_to_g(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = self[[0, 0]];
        let b = self[[0, 1]];
        let c = self[[1, 0]];
        let d = self[[1, 1]];

        let denom = a;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => c / denom,
                (0, 1) => -(a * d - b * c) / denom,
                (1, 0) => denom.recip(),
                (1, 1) => b / denom,
                _ => T::C0,
            },
        ))
    }

    fn a_to_h(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = self[[0, 0]];
        let b = self[[0, 1]];
        let c = self[[1, 0]];
        let d = self[[1, 1]];

        let denom = d;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => b / denom,
                (0, 1) => (a * d - b * c) / denom,
                (1, 0) => -denom.recip(),
                (1, 1) => c / denom,
                _ => T::C0,
            },
        ))
    }

    fn a_to_s(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = self[[0, 0]];
        let b = self[[0, 1]];
        let c = self[[1, 0]];
        let d = self[[1, 1]];

        let denom = a * z0[1] + b + c * z0[0] * z0[1] + d * z0[0];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => (a * z0[1] + b - c * z0[0].conj() * z0[1] - d * z0[0].conj()) / denom,
                (0, 1) => (T::C2 * (a * d - b * c) * z0sqrt) / denom,
                (1, 0) => (T::C2 * z0sqrt) / denom,
                (1, 1) => (-a * z0[1].conj() + b - c * z0[0] * z0[1].conj() + d * z0[0]) / denom,
                _ => T::C0,
            },
        ))
    }

    fn a_to_t(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = self[[0, 0]];
        let b = self[[0, 1]];
        let c = self[[1, 0]];
        let d = self[[1, 1]];

        let denom = T::new(
            T::Real::C2 * Float::sqrt(z0[0].re() * z0[1].re()),
            T::Real::C0,
        );
        if denom.is_zero() || denom.is_nan() {
            return None;
        }
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => {
                    ((T::C4 * a * d - T::C4 * b * c) * z0[0].re().into() * z0[1].re().into()
                        + (((-(c * c * z0[0]) - a * c) * z0[0].conj() + a * c * z0[0] + a * a)
                            * z0[1]
                            + (-(c * d * z0[0]) - a * d) * z0[0].conj()
                            + b * c * z0[0]
                            + a * b)
                            * z0[1].conj()
                        + ((c * d * z0[0] + b * c) * z0[0].conj() - a * d * z0[0] - a * b) * z0[1]
                        + (d * d * z0[0] + b * d) * z0[0].conj()
                        - b * d * z0[0]
                        - b * b)
                        / (((c * z0[0] + a) * z0[1] + d * z0[0] + b) * denom)
                }
                (0, 1) => T::CN1 * ((c * z0[0].conj() - a) * z0[1] + d * z0[0].conj() - b) / denom,
                (1, 0) => ((c * z0[0] + a) * z0[1].conj() - d * z0[0] - b) / denom,
                (1, 1) => ((c * z0[0] + a) * z0[1] + d * z0[0] + b) / denom,
                _ => T::C0,
            },
        ))
    }

    fn a_to_y(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = self[[0, 0]];
        let b = self[[0, 1]];
        let c = self[[1, 0]];
        let d = self[[1, 1]];

        let denom = b;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => d / denom,
                (0, 1) => (b * c - a * d) / denom,
                (1, 0) => -denom.recip(),
                (1, 1) => a / denom,
                _ => T::C0,
            },
        ))
    }

    fn a_to_z(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = self[[0, 0]];
        let b = self[[0, 1]];
        let c = self[[1, 0]];
        let d = self[[1, 1]];

        let denom = c;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => a / denom,
                (0, 1) => (a * d - b * c) / denom,
                (1, 0) => denom.recip(),
                (1, 1) => d / denom,
                _ => T::C0,
            },
        ))
    }

    fn check_dims<'a>(vals: &[Self::Tuple<'a>]) -> Result<(usize, usize), &'static str>
    where
        T: 'a,
    {
        let len = vals.len();
        let nports = (len as f64).sqrt() as usize;

        Ok((nports, nports))
    }

    fn connect(&self, p1: usize, net: &Self, p2: usize) -> Result<Self, String> {
        if p1 >= self.nrows() || p2 >= net.nrows() {
            return Err("connection ports are higher than exist in network".into());
        }
        let pa = self.nrows();
        let pb = net.nrows();
        let nports = self.nrows() * net.nrows();
        let k = p1;
        let l = p2 + self.nrows();
        let mut matrix = Points::<T, Ix2>::from_shape_fn((nports, nports), |(i, j)| {
            if i < pa && j < pa {
                self[(i, j)]
            } else if i >= pa && j >= pa {
                net[(i - pa, j - pa)]
            } else {
                T::C0
            }
        });

        let mut ext_i: Vec<usize> = vec![];
        for i in 0..nports {
            if i != k && i != l {
                ext_i.push(i);
            }
        }

        let akl = T::C1 - matrix[(k, l)];
        let alk = T::C1 - matrix[(l, k)];
        let akk = matrix[(k, k)];
        let all = matrix[(l, l)];
        let denom = akl * alk - akk * all;

        let mut out = Points::<T, Ix2>::zeros((nports - 2, nports - 2));
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

        let mut ake: Array1<T> = Array1::zeros(nports - 2);
        let mut ale: Array1<T> = Array1::zeros(nports - 2);
        let mut aek: Array1<T> = Array1::zeros(nports - 2);
        let mut ael: Array1<T> = Array1::zeros(nports - 2);
        let mut tmp_a: Array1<T> = Array1::zeros(nports - 2);
        let mut tmp_b: Array1<T> = Array1::zeros(nports - 2);
        for &i in ext_i.iter() {
            let mut ii = i;
            if i >= pa {
                ii -= pa;
            }
            ake[(ii)] = matrix[(k, i)];
            ale[(ii)] = matrix[(l, i)];
            aek[(ii)] = matrix[(i, k)];
            ael[(ii)] = matrix[(i, l)];
            tmp_a[(ii)] = (ael[(ii)] * alk + aek[(ii)] * all) / denom;
            tmp_b[(ii)] = (ael[(ii)] * akk + aek[(ii)] * akl) / denom;
        }

        for ((i, j), mut val) in out.indexed_iter_mut() {
            *val += ake[(j)] * tmp_a[(i)] + ale[(j)] * tmp_b[(i)];
        }

        Ok(out)
    }

    fn db(&self) -> Points<T::Real, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| {
            T::Real::C20 * Float::log10(self[idx].abs())
        })
    }

    fn deg(&self) -> Points<T::Real, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].arg().to_degrees())
    }

    fn from_db<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a,
    {
        let dim = Self::check_dims(data)?;

        Ok(Points::from_shape_fn(dim, |(i, j)| {
            let r = ComplexFloat::powf(T::Real::C10, data[i * dim.1 + j].0 / T::Real::C20);
            let theta = data[i * dim.1 + j].1.to_radians();
            T::new(
                r * <T::Real as Float>::cos(theta),
                r * <T::Real as Float>::sin(theta),
            )
        }))
    }

    fn from_magang<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a,
    {
        let dim = Self::check_dims(data)?;

        Ok(Points::from_shape_fn(dim, |(i, j)| {
            let r = data[i * dim.1 + j].0;
            let theta = data[i * dim.1 + j].1.to_radians();
            T::new(
                r * <T::Real as Float>::cos(theta),
                r * <T::Real as Float>::sin(theta),
            )
        }))
    }

    fn from_reim<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a,
    {
        let dim = Self::check_dims(data)?;

        Ok(Points::from_shape_fn(dim, |(i, j)| {
            T::new(data[i * dim.1 + j].0, data[i * dim.1 + j].1)
        }))
    }

    fn g_to_a(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = self[[0, 0]];
        let g12 = self[[0, 1]];
        let g21 = self[[1, 0]];
        let g22 = self[[1, 1]];

        let denom = g21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => denom.recip(),
                (0, 1) => g22 / denom,
                (1, 0) => g11 / denom,
                (1, 1) => (g11 * g22 - g12 * g21) / denom,
                _ => T::C0,
            },
        ))
    }

    fn g_to_h(&self) -> Option<Self> {
        match self.try_inv() {
            Ok(x) => Some(Points::<T, Ix2>::from_shape_fn(x.dim(), |(j, k)| x[[j, k]])),
            _ => None,
        }
    }

    fn g_to_s(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = self[[0, 0]];
        let g12 = self[[0, 1]];
        let g21 = self[[1, 0]];
        let g22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = (g11 * z0[0] + T::C1) * z0[1] + (g11 * g22 - g12 * g21) * z0[0] + g22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => {
                    ((g11 * z0[0].conj() - T::C1) * z0[1] + (g11 * g22 - g12 * g21) * z0[0].conj()
                        - g22)
                        / -denom
                }
                (0, 1) => T::CN2 * g12 * z0sqrt / denom,
                (1, 0) => T::C2 * g21 * z0sqrt / denom,
                (1, 1) => {
                    ((g11 * z0[0] + T::C1) * z0[1].conj() + (g12 * g21 - g11 * g22) * z0[0] - g22)
                        / -denom
                }
                _ => T::C0,
            },
        ))
    }

    fn g_to_t(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = self[[0, 0]];
        let g12 = self[[0, 1]];
        let g21 = self[[1, 0]];
        let g22 = self[[1, 1]];

        let z0sqrt = T::new((z0[0].re() * z0[1].re()), T::Real::C0).sqrt();
        let denom = T::C2 * g21 * z0sqrt;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => {
                    ((T::C4 * g12 * g21 * z0[0].re().into() * z0[1].re().into()
                        + (((g11 * g11 * z0[0] + g11) * z0[0].conj() - g11 * z0[0] - T::C1)
                            * z0[1]
                            + ((g11 * g11 * g22 - g11 * g12 * g21) * z0[0] + g11 * g22
                                - g12 * g21)
                                * z0[0].conj()
                            - g11 * g22 * z0[0]
                            - g22)
                            * z0[1].conj()
                        + (((g11 * g12 * g21 - g11 * g11 * g22) * z0[0] - g11 * g22)
                            * z0[0].conj()
                            + (g11 * g22 - g12 * g21) * z0[0]
                            + g22)
                            * z0[1]
                        + ((T::CN1 * (g11 * g11 * g22 * g22) + T::C2 * g11 * g12 * g21 * g22
                            - g12 * g12 * g21 * g21)
                            * z0[0]
                            - g11 * g22 * g22
                            + g12 * g21 * g22)
                            * z0[0].conj()
                        + (g11 * g22 * g22 - g12 * g21 * g22) * z0[0]
                        + g22 * g22)
                        / ((g11 * z0[0] + T::C1) * z0[1] + (g11 * g22 - g12 * g21) * z0[0] + g22))
                        / -denom
                }
                (0, 1) => {
                    ((g11 * z0[0].conj() - T::C1) * z0[1] + (g11 * g22 - g12 * g21) * z0[0].conj()
                        - g22)
                        / -denom
                }
                (1, 0) => {
                    ((g11 * z0[0] + T::C1) * z0[1].conj() + (g12 * g21 - g11 * g22) * z0[0] - g22)
                        / denom
                }
                (1, 1) => {
                    ((g11 * z0[0] + T::C1) * z0[1] + (g11 * g22 - g12 * g21) * z0[0] + g22) / denom
                }
                _ => T::C0,
            },
        ))
    }

    fn g_to_y(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = self[[0, 0]];
        let g12 = self[[0, 1]];
        let g21 = self[[1, 0]];
        let g22 = self[[1, 1]];

        let denom = g22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => (g11 * g22 - g12 * g21) / denom,
                (0, 1) => g12 / denom,
                (1, 0) => -g21 / denom,
                (1, 1) => denom.recip(),
                _ => T::C0,
            },
        ))
    }

    fn g_to_z(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = self[[0, 0]];
        let g12 = self[[0, 1]];
        let g21 = self[[1, 0]];
        let g22 = self[[1, 1]];

        let denom = g11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => denom.recip(),
                (0, 1) => -g12 / denom,
                (1, 0) => g21 / denom,
                (1, 1) => (g11 * g22 - g12 * g21) / denom,
                _ => T::C0,
            },
        ))
    }

    fn h_to_a(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = self[[0, 0]];
        let h12 = self[[0, 1]];
        let h21 = self[[1, 0]];
        let h22 = self[[1, 1]];

        let denom = h21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => -(h11 * h22 - h12 * h21) / denom,
                (0, 1) => -h11 / denom,
                (1, 0) => -h22 / denom,
                (1, 1) => -denom.recip(),
                _ => T::C0,
            },
        ))
    }

    fn h_to_g(&self) -> Option<Self> {
        match self.try_inv() {
            Ok(x) => Some(Points::<T, Ix2>::from_shape_fn(x.dim(), |(j, k)| x[[j, k]])),
            _ => None,
        }
    }

    fn h_to_s(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = self[[0, 0]];
        let h12 = self[[0, 1]];
        let h21 = self[[1, 0]];
        let h22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = (z0[0] + h11) * (T::C1 + h22 * z0[1]) - h12 * h21 * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => {
                    ((h11 - z0[0].conj()) * (T::C1 + h22 * z0[1]) - h12 * h21 * z0[1]) / denom
                }
                (0, 1) => T::C2 * h12 * z0sqrt / denom,
                (1, 0) => T::CN2 * h21 * z0sqrt / denom,
                (1, 1) => {
                    ((z0[0] + h11) * (T::C1 - h22 * z0[1].conj()) + h12 * h21 * z0[1].conj())
                        / denom
                }
                _ => T::C0,
            },
        ))
    }

    fn h_to_t(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = self[[0, 0]];
        let h12 = self[[0, 1]];
        let h21 = self[[1, 0]];
        let h22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = T::C2 * h21 * z0sqrt;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => {
                    (T::C2
                        * h21
                        * (T::C4 * h12 * h21 * z0[0].re().into() * z0[1].re().into()
                            + (((h22 * h22 * z0[0] + h11 * h22 * h22 - h12 * h21 * h22)
                                * z0[0].conj()
                                + (h12 * h21 * h22 - h11 * h22 * h22) * z0[0]
                                - h11 * h11 * h22 * h22
                                + T::C2 * h11 * h12 * h21 * h22
                                - h12 * h12 * h21 * h21)
                                * z0[1]
                                + (h22 * z0[0] + h11 * h22 - h12 * h21) * z0[0].conj()
                                - h11 * h22 * z0[0]
                                - h11 * h11 * h22
                                + h11 * h12 * h21)
                                * z0[1].conj()
                            + ((-(h22 * z0[0]) - h11 * h22) * z0[0].conj()
                                + (h11 * h22 - h12 * h21) * z0[0]
                                + h11 * h11 * h22
                                - h11 * h12 * h21)
                                * z0[1]
                            + (-z0[0] - h11) * z0[0].conj()
                            + h11 * z0[0]
                            + h11 * h11))
                        / ((T::C2 * h21 * h22 * z0[0] + T::C2 * h11 * h21 * h22
                            - T::C2 * h12 * h21 * h21)
                            * z0[1]
                            + T::C2 * h21 * z0[0]
                            + T::C2 * h11 * h21)
                        / denom
                }
                (0, 1) => {
                    ((h22 * z0[0].conj() - h11 * h22 + h12 * h21) * z0[1] + z0[0].conj() - h11)
                        / denom
                }
                (1, 0) => {
                    ((h22 * z0[0] + h11 * h22 - h12 * h21) * z0[1].conj() - z0[0] - h11) / -denom
                }
                (1, 1) => ((h22 * z0[0] + h11 * h22 - h12 * h21) * z0[1] + z0[0] + h11) / -denom,
                _ => T::C0,
            },
        ))
    }

    fn h_to_y(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = self[[0, 0]];
        let h12 = self[[0, 1]];
        let h21 = self[[1, 0]];
        let h22 = self[[1, 1]];

        let denom = h11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => denom.recip(),
                (0, 1) => -h12 / denom,
                (1, 0) => h21 / denom,
                (1, 1) => (h11 * h22 - h12 * h21) / denom,
                _ => T::C0,
            },
        ))
    }

    fn h_to_z(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = self[[0, 0]];
        let h12 = self[[0, 1]];
        let h21 = self[[1, 0]];
        let h22 = self[[1, 1]];

        let denom = h22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| match (j, k) {
                (0, 0) => (h11 * h22 - h12 * h21) / denom,
                (0, 1) => h12 / denom,
                (1, 0) => -h21 / denom,
                (1, 1) => denom.recip(),
                _ => T::C0,
            },
        ))
    }

    fn im(&self) -> Points<T::Real, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].im())
    }

    fn is_reciprocal(&self) -> bool {
        !(self.nrows() != 2 || self.reciprocity().unwrap() != Points::<T::Real, Ix2>::zeros((2, 2)))
    }

    fn mag(&self) -> Points<T::Real, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].abs())
    }

    fn new_like(pt: &Self) -> Self {
        Points::<T, Ix2>::zeros((pt.nrows(), pt.ncols()))
    }

    fn rad(&self) -> Points<T::Real, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].arg())
    }

    fn re(&self) -> Points<T::Real, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].re())
    }

    fn reciprocity(&self) -> Result<Points<T::Real, Ix2>, String> {
        let nrows = self.nrows();
        if nrows != 2 {
            return Err("reciprocity is only valid for 2 port networks".into());
        }

        let diff = self - self.t().to_owned();
        let mut out = Points::zeros(self.dim());
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                out[[i, j]] = diff[[i, j]].abs();
            }
        }

        Ok(out)
    }

    fn s_to_a(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = self[[0, 0]];
        let s12 = self[[0, 1]];
        let s21 = self[[1, 0]];
        let s22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = T::C2 * s21 * z0sqrt;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((z0[0].conj() + s11 * z0[0]) * (T::C1 - s22) + s12 * s21 * z0[0]) / denom;
        let x12 = ((z0[0].conj() + s11 * z0[0]) * (z0[1].conj() + s22 * z0[1])
            - s12 * s21 * z0[0] * z0[1])
            / denom;
        let x21 = ((T::C1 - s11) * (T::C1 - s22) - s12 * s21) / denom;
        let x22 = ((T::C1 - s11) * (z0[1].conj() + s22 * z0[1]) + s12 * s21 * z0[1]) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_g(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = self[[0, 0]];
        let s12 = self[[0, 1]];
        let s21 = self[[1, 0]];
        let s22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = T::C4 * s12 * s21 * z0[0].re().into() * z0[1].re().into()
            + (((s11 - T::C1) * s22 - s12 * s21 - s11 + T::C1) * z0[0].conj()
                + ((s11 * s11 - s11) * s22 - s11 * s12 * s21 - s11 * s11 + s11) * z0[0])
                * z0[1].conj()
            + (((s11 - T::C1) * s22 * s22 + (-(s12 * s21) - s11 + T::C1) * s22) * z0[0].conj()
                + ((s11 * s11 - s11) * s22 * s22
                    + ((T::C1 - T::C2 * s11) * s12 * s21 - s11 * s11 + s11) * s22
                    + s12 * s12 * s21 * s21
                    + (s11 - T::C1) * s12 * s21)
                    * z0[0])
                * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (((s11 * s11 - T::C2 * s11 + T::C1) * s22 + (T::C1 - s11) * s12 * s21
            - s11 * s11
            + T::C2 * s11
            - T::C1)
            * z0[1].conj()
            + ((s11 * s11 - T::C2 * s11 + T::C1) * s22 * s22
                + ((T::C2 - T::C2 * s11) * s12 * s21 - s11 * s11 + T::C2 * s11 - T::C1) * s22
                + s12 * s12 * s21 * s21
                + (s11 - T::C1) * s12 * s21)
                * z0[1])
            / -denom;
        let x12 = (((T::C2 * s11 - T::C2) * s12 * z0[1].conj()
            + ((T::C2 * s11 - T::C2) * s12 * s22 - T::C2 * s12 * s12 * s21) * z0[1])
            * z0sqrt)
            / denom;
        let x21 = (((T::C2 * s11 - T::C2) * s21 * z0[1].conj()
            + ((T::C2 * s11 - T::C2) * s21 * s22 - T::C2 * s12 * s21 * s21) * z0[1])
            * z0sqrt)
            / -denom;
        let x22 = (((s11 - T::C1) * z0[0].conj() + (s11 * s11 - s11) * z0[0])
            * z0[1].conj()
            * z0[1].conj()
            + (((T::C2 * s11 - T::C2) * s22 - s12 * s21) * z0[0].conj()
                + ((T::C2 * s11 * s11 - T::C2 * s11) * s22 + (T::C1 - T::C2 * s11) * s12 * s21)
                    * z0[0])
                * z0[1]
                * z0[1].conj()
            + (((s11 - T::C1) * s22 * s22 - s12 * s21 * s22) * z0[0].conj()
                + ((s11 * s11 - s11) * s22 * s22
                    + (T::C1 - T::C2 * s11) * s12 * s21 * s22
                    + s12 * s12 * s21 * s21)
                    * z0[0])
                * z0[1]
                * z0[1])
            / -denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_h(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = self[[0, 0]];
        let s12 = self[[0, 1]];
        let s21 = self[[1, 0]];
        let s22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = (T::C1 - s11) * (z0[1].conj() + s22 * z0[1]) + s12 * s21 * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((z0[0].conj() + s11 * z0[0]) * (z0[1].conj() + s22 * z0[1])
            - s12 * s21 * z0[0] * z0[1])
            / denom;
        let x12 = T::C2 * s12 * z0sqrt / denom;
        let x21 = T::CN2 * s21 * z0sqrt / denom;
        let x22 = ((T::C1 - s11) * (T::C1 - s22) - s12 * s21) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_s(&self, z0: &Vec<T>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let f_from = match from {
            WaveType::Power => Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    T::new(z0[j].re(), T::Real::C0).sqrt().recip()
                } else {
                    T::C0
                }
            }),
            WaveType::Pseudo => Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    T::new(z0[j].abs() / Float::sqrt(z0[j].re()), T::Real::C0)
                } else {
                    T::C0
                }
            }),
            WaveType::Traveling => Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
                if j == k { z0[j].sqrt() } else { T::C0 }
            }),
        };
        let g_from = match from {
            WaveType::Power => {
                Points::<T, Ix2>::from_shape_fn(
                    self.dim(),
                    |(j, k)| {
                        if j == k { z0[j] } else { T::C0 }
                    },
                )
            }
            WaveType::Pseudo => Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (T::new(z0[j].abs(), T::Real::C0)
                        / (z0[j] * T::new(z0[j].re(), T::Real::C0).sqrt()))
                } else {
                    T::C0
                }
            }),
            WaveType::Traveling => Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
                if j == k { z0[j].sqrt().recip() } else { T::C0 }
            }),
        };
        let id = Points::<T, Ix2>::eye(self.nrows());

        let v = match from {
            WaveType::Power => {
                let val = g_from.conj() + g_from.dot(self);
                f_from.dot(&val)
            }
            _ => f_from.dot(&(&id + self)),
        };
        let i = match from {
            WaveType::Power => f_from.dot(&(&id - self)),
            _ => g_from.dot(&(&id - self)),
        };

        let f_to = match to {
            WaveType::Power => Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (T::C2 * T::new(z0[j].re(), T::Real::C0).sqrt()).recip()
                } else {
                    T::C0
                }
            }),
            WaveType::Pseudo => Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (T::new(z0[j].re(), T::Real::C0).sqrt()
                        / (T::C2 * T::new(z0[j].abs(), T::Real::C0)))
                } else {
                    T::C0
                }
            }),
            WaveType::Traveling => Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
                if j == k { z0[j].sqrt().recip() } else { T::C0 }
            }),
        };
        let g_to = Points::<T, Ix2>::from_shape_fn(
            self.dim(),
            |(j, k)| if j == k { z0[j] } else { T::C0 },
        );
        let a = f_to.dot(&(&v + &g_to.dot(&i)));
        let b = match to {
            WaveType::Power => f_to.dot(&(&v - &g_to.conj().dot(&i))),
            WaveType::Pseudo | WaveType::Traveling => f_to.dot(&(&v - &g_to.conj().dot(&i))),
        };

        Some(Points::<T, Ix2>::from_shape_fn(self.dim(), |(j, k)| {
            b[[j, k]] / a[[k, k]]
        }))
        // None
    }

    fn s_to_t(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = self[[0, 0]];
        let s12 = self[[0, 1]];
        let s21 = self[[1, 0]];
        let s22 = self[[1, 1]];

        let denom = s21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (s12 * s21 - s11 * s22) / denom;
        let x12 = s11 / denom;
        let x21 = -s22 / denom;
        let x22 = denom.recip();

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_y(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        if self.nrows() == 2 {
            let det = (z0[0].conj() + self[[0, 0]] * z0[0]) * (z0[1].conj() + self[[1, 1]] * z0[1])
                - self[[0, 1]] * self[[1, 0]] * z0[0] * z0[1];
            Some(points![
                [
                    ((T::C1 - self[[0, 0]]) * (z0[1].conj() + self[[1, 1]] * z0[1])
                        + self[[0, 1]] * self[[1, 0]] * z0[1])
                        / det,
                    T::CN2 * self[[0, 1]] * T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt()
                        / det
                ],
                [
                    T::CN2 * self[[1, 0]] * T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt()
                        / det,
                    ((z0[0].conj() + self[[0, 0]] * z0[0]) * (T::C1 - self[[1, 1]])
                        + self[[0, 1]] * self[[1, 0]] * z0[0])
                        / det
                ]
            ])
        } else {
            let id = Points::<T, Ix2>::eye(self.nrows());
            let sqz0inv =
                Points::<T, Ix2>::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
                    if i == j { z0[i].sqrt().recip() } else { T::C0 }
                });

            let diff = &id - self;
            let sum = (&id + self).try_inv();

            match sum {
                Ok(x) => Some(sqz0inv.dot(&diff).dot(&x).dot(&sqz0inv)),
                _ => None,
            }
        }
    }

    fn s_to_z(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        if self.nrows() == 2 {
            let det = (T::C1 - self[[0, 0]]) * (T::C1 - self[[1, 1]]) - self[[0, 1]] * self[[1, 0]];
            Some(points![
                [
                    ((z0[0].conj() + self[[0, 0]] * z0[0]) * (T::C1 - self[[1, 1]])
                        + self[[0, 1]] * self[[1, 0]] * z0[0])
                        / det,
                    T::C2 * self[[0, 1]] * T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt()
                        / det
                ],
                [
                    T::C2 * self[[1, 0]] * T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt()
                        / det,
                    ((T::C1 - self[[0, 0]]) * (z0[1].conj() + self[[1, 1]] * z0[1])
                        + self[[0, 1]] * self[[1, 0]] * z0[1])
                        / det
                ]
            ])
        } else {
            let id = Points::<T, Ix2>::eye(self.nrows());
            let sqz0 = Points::<T, Ix2>::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
                if i == j { z0[i].sqrt() } else { T::C0 }
            });

            let diff = (&id - self).try_inv();
            let sum = &id + self;

            match diff {
                Ok(x) => Some(sqz0.dot(&x).dot(&sum).dot(&sqz0)),
                _ => None,
            }
        }
    }

    fn t_to_a(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = T::C2 * z0sqrt;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 + t21) * z0[0].conj() + (t12 + t11) * z0[0]) / denom;
        let x12 = ((t22 * z0[0].conj() + t12 * z0[0]) * z0[1].conj()
            + (-t21 * z0[0].conj() - t11 * z0[0]) * z0[1])
            / denom;
        let x21 = (t22 + t21 - t12 - t11) / denom;
        let x22 = ((t22 - t12) * z0[1].conj() + (t11 - t21) * z0[1]) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_g(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = (T::C4 * t11 * t22 - T::C4 * t12 * t21) * z0[0].re().into() * z0[1].re().into()
            + ((t22 * t22 + (t21 - t12 - t11) * t22) * z0[0].conj()
                + (t12 * t22 + t12 * t21 - t12 * t12 - t11 * t12) * z0[0])
                * z0[1].conj()
            + ((-(t21 * t22) - t21 * t21 + (t12 + t11) * t21) * z0[0].conj()
                + (-(t11 * t22) - t11 * t21 + t11 * t12 + t11 * t11) * z0[0])
                * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 * t22 + (t21 - T::C2 * t12 - t11) * t22 - t12 * t21
            + t12 * t12
            + t11 * t12)
            * z0[1].conj()
            + ((t11 - t21) * t22 - t21 * t21 + (t12 + T::C2 * t11) * t21 - t11 * t12 - t11 * t11)
                * z0[1])
            / denom;
        let x12 = (((T::C2 * t11 * t22 * t22
            + (-T::C2 * t12 * t21 - T::C2 * t11 * t12) * t22
            + T::C2 * t12 * t12 * t21)
            * z0[1].conj()
            + ((T::C2 * t11 * t11 - T::C2 * t11 * t21) * t22 + T::C2 * t12 * t21 * t21
                - T::C2 * t11 * t12 * t21)
                * z0[1])
            * z0sqrt)
            / -denom;
        let x21 = (((T::C2 * t22 - T::C2 * t12) * z0[1].conj()
            + (T::C2 * t11 - T::C2 * t21) * z0[1])
            * z0sqrt)
            / denom;
        let x22 = (((t22 * t22 - t12 * t22) * z0[0].conj() + (t12 * t22 - t12 * t12) * z0[0])
            * z0[1].conj()
            * z0[1].conj()
            + (((t11 - T::C2 * t21) * t22 + t12 * t21) * z0[0].conj()
                + (-(t11 * t22) - t12 * t21 + T::C2 * t11 * t12) * z0[0])
                * z0[1]
                * z0[1].conj()
            + ((t21 * t21 - t11 * t21) * z0[0].conj() + (t11 * t21 - t11 * t11) * z0[0])
                * z0[1]
                * z0[1])
            / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_h(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = (t22 - t12) * z0[1].conj() + (t11 - t21) * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 * z0[0].conj() + t12 * z0[0]) * z0[1].conj()
            + (-t21 * z0[0].conj() - t11 * z0[0]) * z0[1])
            / denom;
        let x12 = ((T::C2 * t11 * t22 - T::C2 * t12 * t21) * z0sqrt) / denom;
        let x21 = T::CN2 * z0sqrt / denom;
        let x22 = (t22 + t21 - t12 - t11) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_s(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];

        let denom = t22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = t12 / denom;
        let x12 = (t11 * t22 - t12 * t21) / denom;
        let x21 = denom.recip();
        let x22 = -t21 / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_y(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = (t22 * z0[0].conj() + t12 * z0[0]) * z0[1].conj()
            + (-t21 * z0[0].conj() - t11 * z0[0]) * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 - t12) * z0[1].conj() + (t11 - t21) * z0[1]) / denom;
        let x12 = ((T::C2 * t11 * t22 - T::C2 * t12 * t21) * -z0sqrt) / denom;
        let x21 = T::CN2 * z0sqrt / denom;
        let x22 = ((t22 + t21) * z0[0].conj() + (t12 + t11) * z0[0]) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_z(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];

        let denom = t22 + t21 - t12 - t11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 + t21 + t12 + t11) * z0[0]) / denom;
        let x12 = ((T::C2 * t11 * t22 - T::C2 * t12 * t21) * z0[0].sqrt() * z0[1].sqrt()) / denom;
        let x21 = (T::C2 * z0[0].sqrt() * z0[1].sqrt()) / denom;
        let x22 = ((t22 - t21 - t12 + t11) * z0[1]) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_a(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = self[[0, 0]];
        let y12 = self[[0, 1]];
        let y21 = self[[1, 0]];
        let y22 = self[[1, 1]];

        let denom = y21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = -y22 / denom;
        let x12 = T::CN1 / denom;
        let x21 = -(y11 * y22 - y12 * y21) / denom;
        let x22 = -y11 / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_g(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = self[[0, 0]];
        let y12 = self[[0, 1]];
        let y21 = self[[1, 0]];
        let y22 = self[[1, 1]];

        let denom = y22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (y11 * y22 - y12 * y21) / denom;
        let x12 = y12 / denom;
        let x21 = -y21 / denom;
        let x22 = denom.recip();

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_h(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = self[[0, 0]];
        let y12 = self[[0, 1]];
        let y21 = self[[1, 0]];
        let y22 = self[[1, 1]];

        let denom = y11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = denom.recip();
        let x12 = -y12 / denom;
        let x21 = y21 / denom;
        let x22 = (y11 * y22 - y12 * y21) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_s(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        let id = Points::<T, Ix2>::eye(self.nrows());
        let sqz0 = Points::<T, Ix2>::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
            if i == j { z0[i].sqrt() } else { T::C0 }
        });

        let diff = &id - &sqz0.dot(self).dot(&sqz0);
        let sum = (&id + &sqz0.dot(self).dot(&sqz0)).try_inv();

        match sum {
            Ok(x) => Some(x.dot(&diff)),
            _ => None,
        }
    }

    fn y_to_t(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = self[[0, 0]];
        let y12 = self[[0, 1]];
        let y21 = self[[1, 0]];
        let y22 = self[[1, 1]];
        let z0sqrt = T::new(z0[0].re() * z0[1].re(), T::Real::C0).sqrt();

        let denom = T::C2 * y21 * z0[0].sqrt() * z0[1].sqrt();
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (((y11 * y22 - y12 * y21) * z0[0] - y22) * z0[1] - y11 * z0[0] + T::C1) / denom;
        let x12 = (((y11 * y22 - y12 * y21) * z0[0] - y22) * z0[1] + y11 * z0[0] - T::C1) / denom;
        let x21 = -(((y11 * y22 - y12 * y21) * z0[0] + y22) * z0[1] - y11 * z0[0] - T::C1) / denom;
        let x22 = -(((y11 * y22 - y12 * y21) * z0[0] + y22) * z0[1] + y11 * z0[0] + T::C1) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out

        // self.y_to_s(z0).unwrap().s_to_t()
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
        let z11 = self[[0, 0]];
        let z12 = self[[0, 1]];
        let z21 = self[[1, 0]];
        let z22 = self[[1, 1]];

        let denom = z21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = z11 / denom;
        let x12 = (z11 * z22 - z12 * z21) / denom;
        let x21 = denom.recip();
        let x22 = z22 / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn z_to_g(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = self[[0, 0]];
        let z12 = self[[0, 1]];
        let z21 = self[[1, 0]];
        let z22 = self[[1, 1]];

        let denom = z11;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = denom.recip();
        let x12 = -z12 / denom;
        let x21 = z21 / denom;
        let x22 = (z11 * z22 - z12 * z21) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn z_to_h(&self) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = self[[0, 0]];
        let z12 = self[[0, 1]];
        let z21 = self[[1, 0]];
        let z22 = self[[1, 1]];

        let denom = z22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (z11 * z22 - z12 * z21) / denom;
        let x12 = z12 / denom;
        let x21 = -z21 / denom;
        let x22 = denom.recip();

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn z_to_s(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        let id = Points::<T, Ix2>::eye(self.nrows());
        let sqz0inv = Points::<T, Ix2>::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
            if i == j { z0[i].sqrt().recip() } else { T::C0 }
        });

        let diff = &sqz0inv.dot(self).dot(&sqz0inv) - &id;
        let sum = (&sqz0inv.dot(self).dot(&sqz0inv) + &id).try_inv();

        match sum {
            Ok(x) => Some(x.dot(&diff)),
            _ => None,
        }
    }

    fn z_to_t(&self, z0: &Vec<T>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = self[[0, 0]];
        let z12 = self[[0, 1]];
        let z21 = self[[1, 0]];
        let z22 = self[[1, 1]];

        let denom = T::C2 * z0[0].sqrt() * z0[1].sqrt() * z21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (z12 * z21 + z0[1] * z11 - z0[0] * z0[1] - ((z11 - z0[0]) * z22)) / denom;
        let x12 = ((z11 - z0[0]) * z22 - z12 * z21 + z0[1] * z11 - z0[0] * z0[1]) / denom;
        let x21 = (z12 * z21 + z0[1] * z11 + z0[0] * z0[1] - ((z11 + z0[0]) * z22)) / denom;
        let x22 = ((z11 + z0[0]) * z22 - z12 * z21 + z0[1] * z11 + z0[0] * z0[1]) / denom;

        let out: Option<Self> = Some(Points::<T, Ix2>::new(array![[x11, x12], [x21, x22]]));
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
