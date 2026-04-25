#![allow(unused)]
use crate::parameter::RFParameter;
use ndarray::{OwnedRepr, prelude::*};
use ndarray_linalg::*;
use num_complex::{Complex, Complex64, ComplexFloat, c64};
use num_traits::{ConstZero, Num, One, Zero};
use regex::{Regex, RegexBuilder};
use rfkit_base::prelude::*;
use std::{
    error::Error,
    f64::consts::PI,
    fmt, fs,
    iter::Iterator,
    mem,
    ops::{
        Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, Sub, SubAssign,
    },
    process,
    process::Child,
    slice::Iter,
};

pub mod builder;
pub mod file;
pub mod network;
pub mod point;
pub mod points;
pub mod port_points;

pub use self::builder::NetworkBuilder;
pub use self::file::read_touchstone;
pub use self::network::Network;

fn network_err_msg(param: RFParameter, nports: usize) -> String {
    format!("{param} parameters do not exist for network with {nports} port(s)")
}

#[derive(Default, Debug, Clone, Copy)]
pub enum WaveType {
    #[default]
    Power,
    Pseudo,
    Traveling,
}

pub type PortPoints<T> = Points1<T>;

/// Descriptor of a port for use at an iterator over all port combinations
pub type PortVal = (usize, usize);

pub trait NetworkPortPoints<T: RealScalar> {
    fn db(&self) -> PortPoints<T>;

    fn deg(&self) -> PortPoints<T>;

    fn im(&self) -> PortPoints<T>;

    fn mag(&self) -> PortPoints<T>;

    fn rad(&self) -> PortPoints<T>;

    fn re(&self) -> PortPoints<T>;
}

pub trait NetworkPoint<T: RealScalar, D: Dimension>
where
    Self: Sized,
{
    type Size;
    type Tuple<'a>
    where
        T: 'a;

    fn a_to_g(&self) -> Option<Self>
    where
        Self: Sized;

    fn a_to_h(&self) -> Option<Self>
    where
        Self: Sized;

    fn a_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn a_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn a_to_y(&self) -> Option<Self>
    where
        Self: Sized;

    fn a_to_z(&self) -> Option<Self>
    where
        Self: Sized;

    fn check_dims<'a>(data: &[Self::Tuple<'a>]) -> Result<Self::Size, &'static str>
    where
        T: 'a;

    fn connect(&self, p1: usize, net: &Self, p2: usize) -> Result<Self, String>
    where
        Self: Sized;

    fn db(&self) -> Points<T, D>;

    fn deg(&self) -> Points<T, D>;

    fn from_db<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a;

    fn from_magang<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a;

    fn from_reim<'a>(data: &[Self::Tuple<'a>]) -> Result<Self, String>
    where
        T: 'a;

    fn g_to_a(&self) -> Option<Self>
    where
        Self: Sized;

    fn g_to_h(&self) -> Option<Self>
    where
        Self: Sized;

    fn g_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn g_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn g_to_y(&self) -> Option<Self>
    where
        Self: Sized;

    fn g_to_z(&self) -> Option<Self>
    where
        Self: Sized;

    fn h_to_a(&self) -> Option<Self>
    where
        Self: Sized;

    fn h_to_g(&self) -> Option<Self>
    where
        Self: Sized;

    fn h_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn h_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn h_to_y(&self) -> Option<Self>
    where
        Self: Sized;

    fn h_to_z(&self) -> Option<Self>
    where
        Self: Sized;

    fn im(&self) -> Points<T, D>;

    fn is_reciprocal(&self) -> bool;

    fn mag(&self) -> Points<T, D>;

    fn new_like(pt: &Self) -> Self;

    fn rad(&self) -> Points<T, D>;

    fn re(&self) -> Points<T, D>;

    fn reciprocity(&self) -> Result<Points<T, D>, String>;

    fn s_to_a(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn s_to_g(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn s_to_h(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn s_to_s(&self, z0: &Vec<Complex<T>>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized;

    fn s_to_t(&self) -> Option<Self>
    where
        Self: Sized;

    fn s_to_y(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn s_to_z(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_a(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_g(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_h(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_s(&self) -> Option<Self>
    where
        Self: Sized;

    fn t_to_y(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_z(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn y_to_a(&self) -> Option<Self>
    where
        Self: Sized;

    fn y_to_g(&self) -> Option<Self>
    where
        Self: Sized;

    fn y_to_h(&self) -> Option<Self>
    where
        Self: Sized;

    fn y_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn y_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn y_to_z(&self) -> Option<Self>
    where
        Self: Sized;

    fn z_to_a(&self) -> Option<Self>
    where
        Self: Sized;

    fn z_to_g(&self) -> Option<Self>
    where
        Self: Sized;

    fn z_to_h(&self) -> Option<Self>
    where
        Self: Sized;

    fn z_to_s(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn z_to_t(&self, z0: &Vec<Complex<T>>) -> Option<Self>
    where
        Self: Sized;

    fn z_to_y(&self) -> Option<Self>
    where
        Self: Sized;
}
