#![allow(unused)]
use crate::{
    consts::MathConst,
    frequency::Frequency,
    impedance::ComplexNumberType,
    math::*,
    network::{NetworkPoint, WaveType},
    num::{ComplexScalar, RealScalar},
    parameter::RFParameter,
    pts::{Points, Pts},
    unit::Unit,
};
use ndarray::{Dimension, IntoDimension, OwnedRepr, prelude::*};
use ndarray_linalg::*;
use num_complex::{Complex, ComplexFloat};
use num_traits::{ConstZero, Float, Num, One, Zero};
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

impl<T> NetworkPoint<T, Ix3> for Points<T, Ix3>
where
    T: ComplexScalar + From<T::Real>,
    <T as ComplexFloat>::Real: RealScalar + MathConst,
{
    type Size = (usize, usize, usize);
    type Tuple<'a>
        = &'a [(T::Real, T::Real)]
    where
        T: 'a;

    fn a_to_g(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).a_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_h(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).a_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_s(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).a_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_t(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).a_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_y(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).a_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn a_to_z(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).a_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn check_dims<'a>(data: &[Self::Tuple<'a>]) -> Result<(usize, usize, usize), &'static str>
    where
        T: 'a,
    {
        let npts = data.len();
        let len = data.first().map_or(0, |s| s.len());
        let nports = (len as f64).sqrt() as usize;

        if len != nports * nports {
            return Err("Incorrect number of points");
        }
        for slice in data.iter() {
            if slice.len() != len {
                return Err("Inconsistent lengths");
            }
        }

        Ok((npts, nports, nports))
    }

    fn connect(
        &self,
        p1: usize,
        net: &Points<T, Ix3>,
        p2: usize,
    ) -> Result<Points<T, Ix3>, String> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).connect(
                p1,
                &Points::<T, Ix2>::new(net.slice(s![i, .., ..]).to_owned()),
                p2,
            )?;

            pt.assign(&val.inner());
        }
        Ok(out)
    }

    fn db(&self) -> Points<T::Real, Ix3> {
        Points::from_shape_fn(self.dim(), |idx| {
            T::Real::C20 * Float::log10(self[idx].abs())
        })
    }

    fn deg(&self) -> Points<T::Real, Ix3> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].arg().to_degrees())
    }

    fn from_db<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a,
    {
        let dim = Self::check_dims(data)?;

        Ok(Points::from_shape_fn(dim, |(i, j, k)| {
            let r = ComplexFloat::powf(T::Real::C10, data[i][j * dim.1 + k].0 / T::Real::C20);
            let theta = data[i][j * dim.1 + k].1.to_radians();
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

        Ok(Points::from_shape_fn(dim, |(i, j, k)| {
            let r = data[i][j * dim.1 + k].0;
            let theta = data[i][j * dim.1 + k].1.to_radians();
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

        Ok(Points::from_shape_fn(dim, |(i, j, k)| {
            T::new(data[i][j * dim.1 + k].0, data[i][j * dim.1 + k].1)
        }))
    }

    fn g_to_a(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).g_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_h(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).g_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_s(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).g_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_t(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).g_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_y(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).g_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn g_to_z(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).g_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_a(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).h_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_g(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).h_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_s(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).h_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_t(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).h_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_y(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).h_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn h_to_z(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).h_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn im(&self) -> Points<T::Real, Ix3> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].im())
    }

    fn is_reciprocal(&self) -> bool {
        for i in 0..self.npts() - 1 {
            if Points::new(self.slice(s![i, .., ..]).to_owned()).is_reciprocal() == false {
                return false;
            }
        }
        true
    }

    fn mag(&self) -> Points<T::Real, Ix3> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].abs())
    }

    fn new_like(pt: &Points<T, Ix3>) -> Points<T, Ix3> {
        Points::<T, Ix3>::zeros(pt.dim().into_dimension())
    }

    fn rad(&self) -> Points<T::Real, Ix3> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].arg())
    }

    fn re(&self) -> Points<T::Real, Ix3> {
        Points::from_shape_fn(self.dim(), |idx| self[idx].re())
    }

    fn reciprocity(&self) -> Result<Points<T::Real, Ix3>, String> {
        let mut out = Points::<T::Real, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let val = Points::new(self.slice(s![i, .., ..]).to_owned()).reciprocity()?;
            // match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).reciprocity() {
            //     Some(x) => x,
            //     _ => return None,
            // };
            pt.assign(&val.inner());
        }
        Ok(out)
    }

    fn s_to_a(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).s_to_a(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_g(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).s_to_g(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_h(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).s_to_h(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_s(&self, z0: &Vec<T>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized,
    {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned())
                .s_to_s(z0, from, to)
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_t(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).s_to_t()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_y(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).s_to_y(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn s_to_z(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).s_to_z(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_a(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).t_to_a(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_g(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).t_to_g(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_h(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).t_to_h(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_s(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).t_to_s()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_y(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).t_to_y(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn t_to_z(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).t_to_z(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_a(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).y_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_g(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).y_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_h(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).y_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_s(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).y_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_t(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).y_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn y_to_z(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).y_to_z()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_a(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).z_to_a()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_g(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).z_to_g()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_h(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).z_to_h()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_s(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).z_to_s(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_t(&self, z0: &Vec<T>) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val =
                match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).z_to_t(z0) {
                    Some(x) => x,
                    _ => return None,
                };
            pt.assign(&val.inner());
        }

        Some(out)
    }

    fn z_to_y(&self) -> Option<Points<T, Ix3>> {
        let mut out = Points::<T, Ix3>::zeros(self.dim().into_dimension());
        for (i, mut pt) in out.axis_iter_mut(Axis(0)).enumerate() {
            let mut val = match Points::<T, Ix2>::new(self.slice(s![i, .., ..]).to_owned()).z_to_y()
            {
                Some(x) => x,
                _ => return None,
            };
            pt.assign(&val.inner());
        }

        Some(out)
    }
}
