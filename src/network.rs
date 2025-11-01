#![allow(unused)]
use crate::frequency::Frequency;
use crate::impedance::ComplexNumberType;
use crate::math::*;
use crate::mycomplex::MyComplex;
use crate::myfloat::MyFloat;
use crate::parameter::RFParameter;
use crate::point::{Point, Pt};
use crate::points::Points;
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

pub mod builder;
pub mod network;
pub mod point;
pub mod points;
pub mod port_points;

pub use self::builder::NetworkBuilder;
pub use self::network::Network;

macro_rules! unwrap_or_bail {
    ($opt: expr, $msg: expr) => {
        match $opt {
            Some(v) => v,
            None => {
                bail!($msg);
            }
        }
    };
}

macro_rules! unwrap_or_break {
    ($opt: expr) => {
        match $opt {
            Some(v) => v,
            None => {
                break;
            }
        }
    };
}

macro_rules! unwrap_or_continue {
    ($opt: expr) => {
        match $opt {
            Some(v) => v,
            None => {
                continue;
            }
        }
    };
}

macro_rules! unwrap_or_panic {
    ($opt: expr, $msg: expr) => {
        match $opt {
            Some(v) => v,
            None => {
                panic!($msg);
            }
        }
    };
}

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

// pub type PointComplex = Array2<MyComplex>;
// pub type PointFloat = Array2<MyFloat>;
// pub type Point = Array2<Complex64>;
// pub type Pointf64 = Array2<f64>;
pub type PortPoints = Array1<Complex64>;
pub type PortPointsf64 = Array1<f64>;
// pub type Points = Array3<Complex64>;
// pub type Pointsf64 = Array3<f64>;

pub trait NetworkPortPoints {
    fn db(&self) -> PortPointsf64;

    fn deg(&self) -> PortPointsf64;

    fn im(&self) -> PortPointsf64;

    fn mag(&self) -> PortPointsf64;

    // fn new_like(pt: &Points) -> PortPoints;

    fn rad(&self) -> PortPointsf64;

    fn re(&self) -> PortPointsf64;
}

pub trait NetworkPoint<T, U, V> {
    const PRECISION: u32 = 53;

    fn a_to_g(&self) -> Option<Self>
    where
        Self: Sized;

    fn a_to_h(&self) -> Option<Self>
    where
        Self: Sized;

    fn a_to_s(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn a_to_t(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn a_to_y(&self) -> Option<Self>
    where
        Self: Sized;

    fn a_to_z(&self) -> Option<Self>
    where
        Self: Sized;

    fn connect(&self, p1: usize, net: &Self, p2: usize) -> Option<Self>
    where
        Self: Sized;

    fn db(&self) -> T;

    fn deg(&self) -> T;

    fn from_db(data: &[U]) -> Self;

    fn from_ma(data: &[U]) -> Self;

    fn from_ri(data: &[U]) -> Self;

    fn g_to_a(&self) -> Option<Self>
    where
        Self: Sized;

    fn g_to_h(&self) -> Option<Self>
    where
        Self: Sized;

    fn g_to_s(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn g_to_t(&self, z0: &Array1<V>) -> Option<Self>
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

    fn h_to_s(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn h_to_t(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn h_to_y(&self) -> Option<Self>
    where
        Self: Sized;

    fn h_to_z(&self) -> Option<Self>
    where
        Self: Sized;

    // fn identity() -> Self;

    fn im(&self) -> T;

    fn is_reciprocal(&self) -> bool;

    fn mag(&self) -> T;

    fn new_like(pt: &Self) -> Self;

    // fn ones() -> Self;

    fn rad(&self) -> T;

    fn re(&self) -> T;

    fn reciprocity(&self) -> Option<T>;

    fn s_to_a(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn s_to_g(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn s_to_h(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn s_to_s(&self, z0: &Array1<V>, from: WaveType, to: WaveType) -> Option<Self>
    where
        Self: Sized;

    fn s_to_t(&self) -> Option<Self>
    where
        Self: Sized;

    fn s_to_y(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn s_to_z(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_a(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_g(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_h(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_s(&self) -> Option<Self>
    where
        Self: Sized;

    fn t_to_y(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn t_to_z(&self, z0: &Array1<V>) -> Option<Self>
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

    fn y_to_s(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn y_to_t(&self, z0: &Array1<V>) -> Option<Self>
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

    fn z_to_s(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn z_to_t(&self, z0: &Array1<V>) -> Option<Self>
    where
        Self: Sized;

    fn z_to_y(&self) -> Option<Self>
    where
        Self: Sized;
}

/// Descriptor of a port for use at an iterator over all port combinations
pub type PortVal = (usize, usize);

// pub enum PortVal {
//     Leaf((usize, usize)),
//     Children(Vec<PortVal>),
// }

// impl PortVal {
//     fn iter(&self) -> PortIter<'_> {
//         PortIter {
//             children: std::slice::from_ref(self),
//             parent: None,
//         }
//     }

//     fn traverse(&self, f: impl Fn(&(usize, usize))) {
//         match self {
//             PortVal::Leaf(item) => {
//                 f(item);
//             }
//             PortVal::Children(children) => {
//                 for node in children {
//                     node.traverse(&f);
//                 }
//             }
//         }
//     }
// }

// impl<'a> IntoIterator for &'a PortVal {
//     type Item = &'a (usize, usize);

//     type IntoIter = PortIter<'a>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.iter()
//     }
// }

// pub struct PortIter<'a> {
//     children: &'a [PortVal],
//     parent: Option<Box<PortIter<'a>>>,
// }

// impl<'a> Iterator for PortIter<'a> {
//     type Item = &'a (usize, usize);

//     fn next(&mut self) -> Option<Self::Item> {
//         match self.children.get(0) {
//             None => match self.parent.take() {
//                 Some(parent) => {
//                     // continue with the parent node
//                     *self = *parent;
//                     self.next()
//                 }
//                 None => None,
//             },
//             Some(PortVal::Leaf(item)) => {
//                 self.children = &self.children[1..];
//                 Some(item)
//             }
//             Some(PortVal::Children(children)) => {
//                 self.children = &self.children[1..];

//                 // start iterating the child trees
//                 *self = PortIter {
//                     children: children.as_slice(),
//                     parent: Some(Box::new(mem::take(self))),
//                 };
//                 self.next()
//             }
//         }
//     }
// }

// impl Default for PortIter<'_> {
//     fn default() -> Self {
//         PortIter {
//             children: &[],
//             parent: None,
//         }
//     }
// }
