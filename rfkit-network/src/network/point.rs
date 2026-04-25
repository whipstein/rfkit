#![allow(unused)]
use super::*;
use crate::parameter::RFParameter;
use ndarray::{OwnedRepr, linalg::Dot, prelude::*};
use ndarray_linalg::*;
use num_complex::{Complex, Complex64, ComplexFloat, c64};
use num_traits::{ConstOne, ConstZero, Float, FromPrimitive, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use rfkit_base::prelude::*;
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

impl<T: RealScalar> NetworkPoint<T, Ix2> for Points<Complex<T>, Ix2>
where
    Complex<T>: From<T>,
{
    type Size = (usize, usize);
    type Tuple<'a>
        = (T, T)
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => c / denom,
            (0, 1) => -(a * d - b * c) / denom,
            (1, 0) => denom.recip(),
            (1, 1) => b / denom,
            _ => Complex::ZERO,
        }))
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => b / denom,
            (0, 1) => (a * d - b * c) / denom,
            (1, 0) => -denom.recip(),
            (1, 1) => c / denom,
            _ => Complex::ZERO,
        }))
    }

    fn a_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
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
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => (a * z0[1] + b - c * z0[0].conj() * z0[1] - d * z0[0].conj()) / denom,
            (0, 1) => (Complex::from(2.0.into()) * (a * d - b * c) * z0sqrt) / denom,
            (1, 0) => (Complex::from(2.0.into()) * z0sqrt) / denom,
            (1, 1) => (-a * z0[1].conj() + b - c * z0[0] * z0[1].conj() + d * z0[0]) / denom,
            _ => Complex::ZERO,
        }))
    }

    fn a_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = self[[0, 0]];
        let b = self[[0, 1]];
        let c = self[[1, 0]];
        let d = self[[1, 1]];

        let denom = Complex::from((z0[0].re * z0[1].re).sqrt() * 2.0);
        if denom.is_zero() || denom.is_nan() {
            return None;
        }
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => {
                ((Complex::from(4.0.into()) * a * d - Complex::from(4.0.into()) * b * c)
                    * Complex::from(z0[0].re)
                    * Complex::from(z0[1].re)
                    + (((-(c * c * z0[0]) - a * c) * z0[0].conj() + a * c * z0[0] + a * a) * z0[1]
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
            (0, 1) => -((c * z0[0].conj() - a) * z0[1] + d * z0[0].conj() - b) / denom,
            (1, 0) => ((c * z0[0] + a) * z0[1].conj() - d * z0[0] - b) / denom,
            (1, 1) => ((c * z0[0] + a) * z0[1] + d * z0[0] + b) / denom,
            _ => Complex::ZERO,
        }))
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => d / denom,
            (0, 1) => (b * c - a * d) / denom,
            (1, 0) => -denom.recip(),
            (1, 1) => a / denom,
            _ => Complex::ZERO,
        }))
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => a / denom,
            (0, 1) => (a * d - b * c) / denom,
            (1, 0) => denom.recip(),
            (1, 1) => d / denom,
            _ => Complex::ZERO,
        }))
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
        let mut matrix = Points::from_shape_fn((nports, nports), |(i, j)| {
            if i < pa && j < pa {
                self[(i, j)]
            } else if i >= pa && j >= pa {
                net[(i - pa, j - pa)]
            } else {
                Complex::ZERO
            }
        });

        let mut ext_i: Vec<usize> = vec![];
        for i in 0..nports {
            if i != k && i != l {
                ext_i.push(i);
            }
        }

        let akl = Complex::<T>::ONE - matrix[(k, l)];
        let alk = Complex::<T>::ONE - matrix[(l, k)];
        let akk = matrix[(k, k)];
        let all = matrix[(l, l)];
        let denom = akl * alk - akk * all;

        let mut out = Points::zeros((nports - 2, nports - 2));
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

        let mut ake: Array1<Complex<T>> = Array1::zeros(nports - 2);
        let mut ale: Array1<Complex<T>> = Array1::zeros(nports - 2);
        let mut aek: Array1<Complex<T>> = Array1::zeros(nports - 2);
        let mut ael: Array1<Complex<T>> = Array1::zeros(nports - 2);
        let mut tmp_a: Array1<Complex<T>> = Array1::zeros(nports - 2);
        let mut tmp_b: Array1<Complex<T>> = Array1::zeros(nports - 2);
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

    fn db(&self) -> Points<T, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| Float::log10(self[idx].abs()) * 20.0)
    }

    fn deg(&self) -> Points<T, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].arg().to_degrees())
    }

    fn from_db<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a,
    {
        let dim = Self::check_dims(data)?;

        Ok(Points::from_shape_fn(dim, |(i, j)| {
            // let r = ComplexFloat::powf(T::Real::from_f64(10.0), data[i * dim.1 + j].0 / 20.0);
            let r = T::from_f64(10.0).powf(data[i * dim.1 + j].0 / 20.0).re();
            let theta = data[i * dim.1 + j].1.to_radians();
            Complex::new(r * theta.cos(), r * theta.sin())
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
            Complex::new(r * theta.cos(), r * theta.sin())
        }))
    }

    fn from_reim<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a,
    {
        let dim = Self::check_dims(data)?;

        Ok(Points::from_shape_fn(dim, |(i, j)| {
            Complex::new(data[i * dim.1 + j].0, data[i * dim.1 + j].1)
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => denom.recip(),
            (0, 1) => g22 / denom,
            (1, 0) => g11 / denom,
            (1, 1) => (g11 * g22 - g12 * g21) / denom,
            _ => Complex::ZERO,
        }))
    }

    fn g_to_h(&self) -> Option<Self> {
        match self.try_inv() {
            Ok(x) => Some(Points::from_shape_fn(x.dim(), |(j, k)| x[[j, k]])),
            _ => None,
        }
    }

    fn g_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = self[[0, 0]];
        let g12 = self[[0, 1]];
        let g21 = self[[1, 0]];
        let g22 = self[[1, 1]];
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        let denom = (g11 * z0[0] + T::ONE) * z0[1] + (g11 * g22 - g12 * g21) * z0[0] + g22;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => {
                ((g11 * z0[0].conj() - T::ONE) * z0[1] + (g11 * g22 - g12 * g21) * z0[0].conj()
                    - g22)
                    / -denom
            }
            (0, 1) => -Complex::from(2.0.into()) * g12 * z0sqrt / denom,
            (1, 0) => Complex::from(2.0.into()) * g21 * z0sqrt / denom,
            (1, 1) => {
                ((g11 * z0[0] + T::ONE) * z0[1].conj() + (g12 * g21 - g11 * g22) * z0[0] - g22)
                    / -denom
            }
            _ => Complex::ZERO,
        }))
    }

    fn g_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let g11 = self[[0, 0]];
        let g12 = self[[0, 1]];
        let g21 = self[[1, 0]];
        let g22 = self[[1, 1]];

        let z0sqrt = Complex::new((z0[0].re() * z0[1].re()), T::ZERO).sqrt();
        let denom = Complex::new(2.0.into(), T::ZERO) * g21 * z0sqrt;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => {
                ((Complex::from(4.0.into())
                    * g12
                    * g21
                    * Complex::from(z0[0].re)
                    * Complex::from(z0[1].re)
                    + (((g11 * g11 * z0[0] + g11) * z0[0].conj() - g11 * z0[0] - T::ONE) * z0[1]
                        + ((g11 * g11 * g22 - g11 * g12 * g21) * z0[0] + g11 * g22 - g12 * g21)
                            * z0[0].conj()
                        - g11 * g22 * z0[0]
                        - g22)
                        * z0[1].conj()
                    + (((g11 * g12 * g21 - g11 * g11 * g22) * z0[0] - g11 * g22) * z0[0].conj()
                        + (g11 * g22 - g12 * g21) * z0[0]
                        + g22)
                        * z0[1]
                    + ((-(g11 * g11 * g22 * g22)
                        + Complex::from(2.0.into()) * g11 * g12 * g21 * g22
                        - g12 * g12 * g21 * g21)
                        * z0[0]
                        - g11 * g22 * g22
                        + g12 * g21 * g22)
                        * z0[0].conj()
                    + (g11 * g22 * g22 - g12 * g21 * g22) * z0[0]
                    + g22 * g22)
                    / ((g11 * z0[0] + T::ONE) * z0[1] + (g11 * g22 - g12 * g21) * z0[0] + g22))
                    / -denom
            }
            (0, 1) => {
                ((g11 * z0[0].conj() - T::ONE) * z0[1] + (g11 * g22 - g12 * g21) * z0[0].conj()
                    - g22)
                    / -denom
            }
            (1, 0) => {
                ((g11 * z0[0] + T::ONE) * z0[1].conj() + (g12 * g21 - g11 * g22) * z0[0] - g22)
                    / denom
            }
            (1, 1) => {
                ((g11 * z0[0] + T::ONE) * z0[1] + (g11 * g22 - g12 * g21) * z0[0] + g22) / denom
            }
            _ => Complex::ZERO,
        }))
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => (g11 * g22 - g12 * g21) / denom,
            (0, 1) => g12 / denom,
            (1, 0) => -g21 / denom,
            (1, 1) => denom.recip(),
            _ => Complex::ZERO,
        }))
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => denom.recip(),
            (0, 1) => -g12 / denom,
            (1, 0) => g21 / denom,
            (1, 1) => (g11 * g22 - g12 * g21) / denom,
            _ => Complex::ZERO,
        }))
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => -(h11 * h22 - h12 * h21) / denom,
            (0, 1) => -h11 / denom,
            (1, 0) => -h22 / denom,
            (1, 1) => -denom.recip(),
            _ => Complex::ZERO,
        }))
    }

    fn h_to_g(&self) -> Option<Self> {
        match self.try_inv() {
            Ok(x) => Some(Points::from_shape_fn(x.dim(), |(j, k)| x[[j, k]])),
            _ => None,
        }
    }

    fn h_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = self[[0, 0]];
        let h12 = self[[0, 1]];
        let h21 = self[[1, 0]];
        let h22 = self[[1, 1]];
        let z0sqrt = Complex::new(z0[0].re * z0[1].re, T::ZERO).sqrt();

        let denom = (z0[0] + h11) * (Complex::<T>::ONE + h22 * z0[1]) - h12 * h21 * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => {
                ((h11 - z0[0].conj()) * (Complex::<T>::ONE + h22 * z0[1]) - h12 * h21 * z0[1])
                    / denom
            }
            (0, 1) => Complex::from(2.0.into()) * h12 * z0sqrt / denom,
            (1, 0) => -Complex::from(2.0.into()) * h21 * z0sqrt / denom,
            (1, 1) => {
                ((z0[0] + h11) * (Complex::<T>::ONE - h22 * z0[1].conj())
                    + h12 * h21 * z0[1].conj())
                    / denom
            }
            _ => Complex::ZERO,
        }))
    }

    fn h_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = self[[0, 0]];
        let h12 = self[[0, 1]];
        let h21 = self[[1, 0]];
        let h22 = self[[1, 1]];
        let z0sqrt = Complex::new(z0[0].re * z0[1].re, T::ZERO).sqrt();

        let denom = Complex::new(2.0.into(), T::ZERO) * h21 * z0sqrt;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => {
                (Complex::new(2.0.into(), T::ZERO)
                    * h21
                    * (Complex::new(4.0.into(), T::ZERO)
                        * h12
                        * h21
                        * Complex::from(z0[0].re)
                        * Complex::from(z0[1].re)
                        + (((h22 * h22 * z0[0] + h11 * h22 * h22 - h12 * h21 * h22)
                            * z0[0].conj()
                            + (h12 * h21 * h22 - h11 * h22 * h22) * z0[0]
                            - h11 * h11 * h22 * h22
                            + Complex::from(2.0.into()) * h11 * h12 * h21 * h22
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
                    / (Complex::from(2.0.into())
                        * ((h21 * h22 * z0[0] + h11 * h21 * h22 - h12 * h21 * h21) * z0[1]
                            + h21 * z0[0]
                            + h11 * h21))
                    / denom
            }
            (0, 1) => {
                ((h22 * z0[0].conj() - h11 * h22 + h12 * h21) * z0[1] + z0[0].conj() - h11) / denom
            }
            (1, 0) => ((h22 * z0[0] + h11 * h22 - h12 * h21) * z0[1].conj() - z0[0] - h11) / -denom,
            (1, 1) => ((h22 * z0[0] + h11 * h22 - h12 * h21) * z0[1] + z0[0] + h11) / -denom,
            _ => Complex::ZERO,
        }))
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => denom.recip(),
            (0, 1) => -h12 / denom,
            (1, 0) => h21 / denom,
            (1, 1) => (h11 * h22 - h12 * h21) / denom,
            _ => Complex::ZERO,
        }))
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

        Some(Points::from_shape_fn(self.dim(), |(j, k)| match (j, k) {
            (0, 0) => (h11 * h22 - h12 * h21) / denom,
            (0, 1) => h12 / denom,
            (1, 0) => -h21 / denom,
            (1, 1) => denom.recip(),
            _ => Complex::ZERO,
        }))
    }

    fn im(&self) -> Points<T, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].im())
    }

    fn is_reciprocal(&self) -> bool {
        !(self.nrows() != 2 || self.reciprocity().unwrap() != Points::zeros((2, 2)))
    }

    fn mag(&self) -> Points<T, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].abs())
    }

    fn new_like(pt: &Self) -> Self {
        Points::zeros((pt.nrows(), pt.ncols()))
    }

    fn rad(&self) -> Points<T, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].arg())
    }

    fn re(&self) -> Points<T, Ix2> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].re())
    }

    fn reciprocity(&self) -> Result<Points<T, Ix2>, String> {
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

    fn s_to_a(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = self[[0, 0]];
        let s12 = self[[0, 1]];
        let s21 = self[[1, 0]];
        let s22 = self[[1, 1]];
        let z0sqrt = Complex::new(z0[0].re * z0[1].re, T::ZERO).sqrt();

        let denom = Complex::from(2.0.into()) * s21 * z0sqrt;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 =
            ((z0[0].conj() + s11 * z0[0]) * (Complex::<T>::ONE - s22) + s12 * s21 * z0[0]) / denom;
        let x12 = ((z0[0].conj() + s11 * z0[0]) * (z0[1].conj() + s22 * z0[1])
            - s12 * s21 * z0[0] * z0[1])
            / denom;
        let x21 = ((Complex::<T>::ONE - s11) * (Complex::<T>::ONE - s22) - s12 * s21) / denom;
        let x22 =
            ((Complex::<T>::ONE - s11) * (z0[1].conj() + s22 * z0[1]) + s12 * s21 * z0[1]) / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_g(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = self[[0, 0]];
        let s12 = self[[0, 1]];
        let s21 = self[[1, 0]];
        let s22 = self[[1, 1]];
        let z0sqrt = Complex::new(z0[0].re * z0[1].re, T::ZERO).sqrt();

        let denom: Complex<T> = Complex::from(4.0.into())
            * s12
            * s21
            * Complex::from(z0[0].re)
            * Complex::from(z0[1].re)
            + (((s11 - Complex::ONE) * s22 - s12 * s21 - s11 + Complex::ONE) * z0[0].conj()
                + ((s11 * s11 - s11) * s22 - s11 * s12 * s21 - s11 * s11 + s11) * z0[0])
                * z0[1].conj()
            + (((s11 - Complex::<T>::ONE) * s22 * s22 + (-(s12 * s21) - s11 + Complex::ONE) * s22)
                * z0[0].conj()
                + ((s11 * s11 - s11) * s22 * s22
                    + ((Complex::<T>::ONE - Complex::from(2.0.into()) * s11) * s12 * s21
                        - s11 * s11
                        + s11)
                        * s22
                    + s12 * s12 * s21 * s21
                    + (s11 - Complex::ONE) * s12 * s21)
                    * z0[0])
                * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11: Complex<T> = (((s11 * s11 - Complex::from(2.0.into()) * s11 + Complex::ONE)
            * s22
            + (Complex::<T>::ONE - s11) * s12 * s21
            - s11 * s11
            + Complex::from(2.0.into()) * s11
            - Complex::ONE)
            * z0[1].conj()
            + ((s11 * s11 - Complex::from(2.0.into()) * s11 + T::ONE) * s22 * s22
                + (Complex::from(2.0.into()) * (Complex::<T>::ONE - s11) * s12 * s21 - s11 * s11
                    + Complex::from(2.0.into()) * s11
                    - Complex::ONE)
                    * s22
                + s12 * s12 * s21 * s21
                + (s11 - Complex::ONE) * s12 * s21)
                * z0[1])
            / -denom;
        let x12 = (Complex::from(2.0.into())
            * ((s11 - T::ONE) * s12 * z0[1].conj()
                + ((s11 - T::ONE) * s12 * s22 - s12 * s12 * s21) * z0[1])
            * z0sqrt)
            / denom;
        let x21 = (Complex::from(2.0.into())
            * ((s11 - Complex::ONE) * s21 * z0[1].conj()
                + ((s11 - Complex::ONE) * s21 * s22 - s12 * s21 * s21) * z0[1])
            * z0sqrt)
            / -denom;
        let x22: Complex<T> = (((s11 - Complex::ONE) * z0[0].conj() + (s11 * s11 - s11) * z0[0])
            * z0[1].conj()
            * z0[1].conj()
            + (Complex::from(2.0.into())
                * ((s11 - Complex::ONE) * s22 - s12 * s21)
                * z0[0].conj()
                + ((s11 * s11 - s11) * s22
                    + (Complex::<T>::ONE - Complex::from(2.0.into()) * s11) * s12 * s21)
                    * z0[0])
                * z0[1]
                * z0[1].conj()
            + (((s11 - Complex::<T>::ONE) * s22 * s22 - s12 * s21 * s22) * z0[0].conj()
                + ((s11 * s11 - s11) * s22 * s22
                    + (Complex::<T>::ONE - Complex::from(2.0.into()) * s11) * s12 * s21 * s22
                    + s12 * s12 * s21 * s21)
                    * z0[0])
                * z0[1]
                * z0[1])
            / -denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_h(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = self[[0, 0]];
        let s12 = self[[0, 1]];
        let s21 = self[[1, 0]];
        let s22 = self[[1, 1]];
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        let denom: Complex<T> =
            (Complex::<T>::ONE - s11) * (z0[1].conj() + s22 * z0[1]) + s12 * s21 * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((z0[0].conj() + s11 * z0[0]) * (z0[1].conj() + s22 * z0[1])
            - s12 * s21 * z0[0] * z0[1])
            / denom;
        let x12 = Complex::from(2.0.into()) * s12 * z0sqrt / denom;
        let x21 = -Complex::from(2.0.into()) * s21 * z0sqrt / denom;
        let x22 = ((Complex::<T>::ONE - s11) * (Complex::<T>::ONE - s22) - s12 * s21) / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_s(&self, z0: &Vec<Complex<T>>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let f_from = match from {
            WaveType::Power => Points::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    Complex::from(z0[j].re).sqrt().recip()
                } else {
                    Complex::ZERO
                }
            }),
            WaveType::Pseudo => Points::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    Complex::from(z0[j].abs() / z0[j].re.sqrt())
                } else {
                    Complex::ZERO
                }
            }),
            WaveType::Traveling => Points::from_shape_fn(self.dim(), |(j, k)| {
                if j == k { z0[j].sqrt() } else { Complex::ZERO }
            }),
        };
        let g_from = match from {
            WaveType::Power => {
                Points::from_shape_fn(
                    self.dim(),
                    |(j, k)| if j == k { z0[j] } else { Complex::ZERO },
                )
            }
            WaveType::Pseudo => Points::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (Complex::from(z0[j].abs()) / (z0[j] * Complex::from(z0[j].re).sqrt()))
                } else {
                    Complex::ZERO
                }
            }),
            WaveType::Traveling => Points::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    z0[j].sqrt().recip()
                } else {
                    Complex::ZERO
                }
            }),
        };
        let id = Points::eye(self.nrows());

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
            WaveType::Power => Points::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    Complex::from(z0[j].re * 2.0).sqrt().recip()
                } else {
                    Complex::ZERO
                }
            }),
            WaveType::Pseudo => Points::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    (Complex::from(z0[j].re).sqrt() / Complex::from(z0[j].abs() * 2.0))
                } else {
                    Complex::ZERO
                }
            }),
            WaveType::Traveling => Points::from_shape_fn(self.dim(), |(j, k)| {
                if j == k {
                    z0[j].sqrt().recip()
                } else {
                    Complex::ZERO
                }
            }),
        };
        let g_to = Points::from_shape_fn(
            self.dim(),
            |(j, k)| if j == k { z0[j] } else { Complex::ZERO },
        );
        let a = f_to.dot(&(&v + &g_to.dot(&i)));
        let b = match to {
            WaveType::Power => f_to.dot(&(&v - &g_to.conj().dot(&i))),
            WaveType::Pseudo | WaveType::Traveling => f_to.dot(&(&v - &g_to.conj().dot(&i))),
        };

        Some(Points::from_shape_fn(self.dim(), |(j, k)| {
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

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn s_to_y(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        if self.nrows() == 2 {
            let det = (z0[0].conj() + self[[0, 0]] * z0[0]) * (z0[1].conj() + self[[1, 1]] * z0[1])
                - self[[0, 1]] * self[[1, 0]] * z0[0] * z0[1];
            Some(points![
                [
                    ((Complex::<T>::ONE - self[[0, 0]]) * (z0[1].conj() + self[[1, 1]] * z0[1])
                        + self[[0, 1]] * self[[1, 0]] * z0[1])
                        / det,
                    -Complex::from(2.0.into())
                        * self[[0, 1]]
                        * Complex::from(z0[0].re * z0[1].re).sqrt()
                        / det
                ],
                [
                    -Complex::from(2.0.into())
                        * self[[1, 0]]
                        * Complex::from(z0[0].re * z0[1].re).sqrt()
                        / det,
                    ((z0[0].conj() + self[[0, 0]] * z0[0]) * (Complex::<T>::ONE - self[[1, 1]])
                        + self[[0, 1]] * self[[1, 0]] * z0[0])
                        / det
                ]
            ])
        } else {
            let id = Points::eye(self.nrows());
            let sqz0inv = Points::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
                if i == j {
                    z0[i].sqrt().recip()
                } else {
                    Complex::ZERO
                }
            });

            let diff = &id - self;
            let sum = (&id + self).try_inv();

            match sum {
                Ok(x) => Some(sqz0inv.dot(&diff).dot(&x).dot(&sqz0inv)),
                _ => None,
            }
        }
    }

    fn s_to_z(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        if self.nrows() == 2 {
            let det = (Complex::<T>::ONE - self[[0, 0]]) * (Complex::<T>::ONE - self[[1, 1]])
                - self[[0, 1]] * self[[1, 0]];
            Some(points![
                [
                    ((z0[0].conj() + self[[0, 0]] * z0[0]) * (Complex::<T>::ONE - self[[1, 1]])
                        + self[[0, 1]] * self[[1, 0]] * z0[0])
                        / det,
                    Complex::from(2.0.into())
                        * self[[0, 1]]
                        * Complex::from(z0[0].re * z0[1].re).sqrt()
                        / det
                ],
                [
                    Complex::from(2.0.into())
                        * self[[1, 0]]
                        * Complex::from(z0[0].re * z0[1].re).sqrt()
                        / det,
                    ((Complex::<T>::ONE - self[[0, 0]]) * (z0[1].conj() + self[[1, 1]] * z0[1])
                        + self[[0, 1]] * self[[1, 0]] * z0[1])
                        / det
                ]
            ])
        } else {
            let id = Points::eye(self.nrows());
            let sqz0 = Points::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
                if i == j { z0[i].sqrt() } else { Complex::ZERO }
            });

            let diff = (&id - self).try_inv();
            let sum = &id + self;

            match diff {
                Ok(x) => Some(sqz0.dot(&x).dot(&sum).dot(&sqz0)),
                _ => None,
            }
        }
    }

    fn t_to_a(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        let denom = Complex::from(2.0.into()) * z0sqrt;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 + t21) * z0[0].conj() + (t12 + t11) * z0[0]) / denom;
        let x12 = ((t22 * z0[0].conj() + t12 * z0[0]) * z0[1].conj()
            + (-t21 * z0[0].conj() - t11 * z0[0]) * z0[1])
            / denom;
        let x21 = (t22 + t21 - t12 - t11) / denom;
        let x22 = ((t22 - t12) * z0[1].conj() + (t11 - t21) * z0[1]) / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_g(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        let denom: Complex<T> = Complex::from(4.0.into())
            * (t11 * t22 - t12 * t21)
            * Complex::from(z0[0].re)
            * Complex::from(z0[1].re)
            + ((t22 * t22 + (t21 - t12 - t11) * t22) * z0[0].conj()
                + (t12 * t22 + t12 * t21 - t12 * t12 - t11 * t12) * z0[0])
                * z0[1].conj()
            + ((-(t21 * t22) - t21 * t21 + (t12 + t11) * t21) * z0[0].conj()
                + (-(t11 * t22) - t11 * t21 + t11 * t12 + t11 * t11) * z0[0])
                * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 * t22 + (t21 - Complex::from(2.0.into()) * t12 - t11) * t22 - t12 * t21
            + t12 * t12
            + t11 * t12)
            * z0[1].conj()
            + ((t11 - t21) * t22 - t21 * t21 + (t12 + Complex::from(2.0.into()) * t11) * t21
                - t11 * t12
                - t11 * t11)
                * z0[1])
            / denom;
        let x12 = (Complex::from(2.0.into())
            * ((t11 * t22 * t22 + (-t12 * t21 - t11 * t12) * t22 + t12 * t12 * t21)
                * z0[1].conj()
                + ((t11 * t11 - t11 * t21) * t22 + t12 * t21 * t21 - t11 * t12 * t21) * z0[1])
            * z0sqrt)
            / -denom;
        let x21 = (Complex::from(2.0.into())
            * ((t22 - t12) * z0[1].conj() + (t11 - t21) * z0[1])
            * z0sqrt)
            / denom;
        let x22 = (((t22 * t22 - t12 * t22) * z0[0].conj() + (t12 * t22 - t12 * t12) * z0[0])
            * z0[1].conj()
            * z0[1].conj()
            + (((t11 - Complex::from(2.0.into()) * t21) * t22 + t12 * t21) * z0[0].conj()
                + (-(t11 * t22) - t12 * t21 + Complex::from(2.0.into()) * t11 * t12) * z0[0])
                * z0[1]
                * z0[1].conj()
            + ((t21 * t21 - t11 * t21) * z0[0].conj() + (t11 * t21 - t11 * t11) * z0[0])
                * z0[1]
                * z0[1])
            / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_h(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        let denom = (t22 - t12) * z0[1].conj() + (t11 - t21) * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 * z0[0].conj() + t12 * z0[0]) * z0[1].conj()
            + (-t21 * z0[0].conj() - t11 * z0[0]) * z0[1])
            / denom;
        let x12 = (Complex::from(2.0.into()) * (t11 * t22 - t12 * t21) * z0sqrt) / denom;
        let x21 = -Complex::from(2.0.into()) * z0sqrt / denom;
        let x22 = (t22 + t21 - t12 - t11) / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
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

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_y(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = self[[0, 0]];
        let t12 = self[[0, 1]];
        let t21 = self[[1, 0]];
        let t22 = self[[1, 1]];
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        let denom = (t22 * z0[0].conj() + t12 * z0[0]) * z0[1].conj()
            + (-t21 * z0[0].conj() - t11 * z0[0]) * z0[1];
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = ((t22 - t12) * z0[1].conj() + (t11 - t21) * z0[1]) / denom;
        let x12 = (Complex::from(2.0.into()) * (t11 * t22 - t12 * t21) * -z0sqrt) / denom;
        let x21 = -Complex::from(2.0.into()) * z0sqrt / denom;
        let x22 = ((t22 + t21) * z0[0].conj() + (t12 + t11) * z0[0]) / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn t_to_z(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
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
        let x12 =
            (Complex::from(2.0.into()) * (t11 * t22 - t12 * t21) * z0[0].sqrt() * z0[1].sqrt())
                / denom;
        let x21 = (Complex::from(2.0.into()) * z0[0].sqrt() * z0[1].sqrt()) / denom;
        let x22 = ((t22 - t21 - t12 + t11) * z0[1]) / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
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
        let x12 = -denom.recip();
        let x21 = -(y11 * y22 - y12 * y21) / denom;
        let x22 = -y11 / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
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

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
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

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn y_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        let id = Points::eye(self.nrows());
        let sqz0 = Points::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
            if i == j { z0[i].sqrt() } else { Complex::ZERO }
        });

        let diff = &id - &sqz0.dot(self).dot(&sqz0);
        let sum = (&id + &sqz0.dot(self).dot(&sqz0)).try_inv();

        match sum {
            Ok(x) => Some(x.dot(&diff)),
            _ => None,
        }
    }

    fn y_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = self[[0, 0]];
        let y12 = self[[0, 1]];
        let y21 = self[[1, 0]];
        let y22 = self[[1, 1]];
        let z0sqrt = Complex::from(z0[0].re * z0[1].re).sqrt();

        let denom = Complex::from(2.0.into()) * y21 * z0[0].sqrt() * z0[1].sqrt();
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (((y11 * y22 - y12 * y21) * z0[0] - y22) * z0[1] - y11 * z0[0] + T::ONE) / denom;
        let x12 = (((y11 * y22 - y12 * y21) * z0[0] - y22) * z0[1] + y11 * z0[0] - T::ONE) / denom;
        let x21 = -(((y11 * y22 - y12 * y21) * z0[0] + y22) * z0[1] - y11 * z0[0] - T::ONE) / denom;
        let x22 = -(((y11 * y22 - y12 * y21) * z0[0] + y22) * z0[1] + y11 * z0[0] + T::ONE) / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
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

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
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

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
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

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
        out
    }

    fn z_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() {
            return None;
        }
        let id: Points<Complex<T>, Ix2> = Points::eye(self.nrows());
        // let sqz0inv = Points::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
        //     if i == j {
        //         z0[i].sqrt().recip()
        //     } else {
        //         Complex::ZERO
        //     }
        // });
        let z0mat = Points::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| {
            if i == j { z0[i] } else { Complex::ZERO }
        });

        // let diff = &sqz0inv.dot(self).dot(&sqz0inv) - &id;
        // let sum = (&sqz0inv.dot(self).dot(&sqz0inv) + &id).try_inv();
        // match sum {
        //     Ok(x) => Some(x.dot(&diff)),
        //     _ => None,
        // }

        let diff = self - z0mat.dot(&id);
        let sum = (self + z0mat.dot(&id)).try_inv();

        match sum {
            Ok(x) => Some(diff.dot(&x)),
            _ => None,
        }
    }

    fn z_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = self[[0, 0]];
        let z12 = self[[0, 1]];
        let z21 = self[[1, 0]];
        let z22 = self[[1, 1]];

        let denom = Complex::from(2.0.into()) * z0[0].sqrt() * z0[1].sqrt() * z21;
        if denom.is_zero() || denom.is_nan() {
            return None;
        }

        let x11 = (z12 * z21 + z0[1] * z11 - z0[0] * z0[1] - ((z11 - z0[0]) * z22)) / denom;
        let x12 = ((z11 - z0[0]) * z22 - z12 * z21 + z0[1] * z11 - z0[0] * z0[1]) / denom;
        let x21 = (z12 * z21 + z0[1] * z11 + z0[0] * z0[1] - ((z11 + z0[0]) * z22)) / denom;
        let x22 = ((z11 + z0[0]) * z22 - z12 * z21 + z0[1] * z11 + z0[0] * z0[1]) / denom;

        let out: Option<Self> = Some(Points::new(array![[x11, x12], [x21, x22]]));
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
