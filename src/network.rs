#![allow(unused)]
use crate::enums::{Unit, RFDataFormat, RFParameter};
use crate::frequency::Frequency;
use crate::math::sqrt_phase_unwrap;
use std::fmt;
use faer::io::Npy;
use regex::{Regex, RegexBuilder};
use simple_error::SimpleError;
use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::process;
use faer::{mat, row, scale, ComplexField, Mat, Row};
use faer::complex_native::c64;
use faer::linalg::solvers::PartialPivLu;
use faer::solvers::SolverCore;

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

pub type Pointf64 = Mat<f64>;
pub type Point = Mat<c64>;

pub type PortPointsf64 = Vec<f64>;
pub type PortPoints = Vec<c64>;

pub type Pointsf64 = Vec<Pointf64>;
pub type Points = Vec<Point>;

trait NetworkPointf64
{
    fn db(&self) -> Pointf64;

    fn new_like(pt: &Pointf64) -> Pointf64;

    // fn from_row_iterator(data: Vec<f64>) -> Pointf64;

    // fn from_vec(data: Vec<f64>) -> Pointf64;

    // fn identity() -> Pointf64;

    // fn ones() -> Pointf64;
}

impl NetworkPointf64 for Pointf64
{
    fn db(&self) -> Pointf64 {
        let mut pt: Pointf64 = Pointf64::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt.write(i, j, 20.0 * (self.get(i, j)).log10());
            }
        }

        pt
    }

    fn new_like(pt: &Pointf64) -> Pointf64 {
        Pointf64::zeros(pt.nrows(), pt.ncols())
    }

    // fn from_row_iterator(data: Vec<f64>) -> Pointf64 {
    //     Pointf64::from_row_iterator(data.into_iter())
    // }

    // fn from_vec(data: Vec<f64>) -> Pointf64 {
    //     Pointf64::from_vec(data)
    // }

    // fn identity() -> Pointf64 {
    //     Pointf64::one()
    // }

    // fn ones() -> Pointf64 {
    //     let mut val = Pointf64::zeros();
    //     val.fill(1.0);
    //     val
    // }
}

trait NetworkPoint
{
    fn a_to_g(&self) -> Option<Point>;

    fn a_to_h(&self) -> Option<Point>;

    fn a_to_s(&self, z0: &Row<c64>) -> Option<Point>;

    fn a_to_t(&self, z0: &Row<c64>) -> Option<Point>;

    fn a_to_y(&self) -> Option<Point>;

    fn a_to_z(&self) -> Option<Point>;

    // fn connect(&self, k: usize, net2: &Point, l: usize) -> Point;

    fn db(&self) -> Pointf64;

    fn deg(&self) -> Pointf64;

    // fn from_row_iterator(data: Vec<c64>) -> Point;

    // fn from_vec(data: Vec<c64>) -> Point;

    fn g_to_a(&self) -> Option<Point>;

    fn g_to_h(&self) -> Option<Point>;

    fn g_to_s(&self, z0: &Row<c64>) -> Option<Point>;

    fn g_to_t(&self, z0: &Row<c64>) -> Option<Point>;

    fn g_to_y(&self) -> Option<Point>;

    fn g_to_z(&self) -> Option<Point>;

    fn h_to_a(&self) -> Option<Point>;

    fn h_to_g(&self) -> Option<Point>;

    fn h_to_s(&self, z0: &Row<c64>) -> Option<Point>;

    fn h_to_t(&self, z0: &Row<c64>) -> Option<Point>;

    fn h_to_y(&self) -> Option<Point>;

    fn h_to_z(&self) -> Option<Point>;

    // fn identity() -> Point;

    fn im(&self) -> Pointf64;

    fn is_square(&self) -> bool;

    fn is_reciprocal(&self) -> bool;

    fn mag(&self) -> Pointf64;

    fn new_like(pt: &Point) -> Point;

    // fn ones() -> Point;

    fn rad(&self) -> Pointf64;

    fn re(&self) -> Pointf64;

    fn reciprocity(&self) -> Option<Pointf64>;

    fn s_to_a(&self, z0: &Row<c64>) -> Option<Point>;

    fn s_to_g(&self, z0: &Row<c64>) -> Option<Point>;

    fn s_to_h(&self, z0: &Row<c64>) -> Option<Point>;

    fn s_to_t(&self) -> Option<Point>;

    fn s_to_y(&self, z0: &Row<c64>) -> Option<Point>;

    fn s_to_z(&self, z0: &Row<c64>) -> Option<Point>;

    fn t_to_a(&self, z0: &Row<c64>) -> Option<Point>;

    fn t_to_g(&self, z0: &Row<c64>) -> Option<Point>;

    fn t_to_h(&self, z0: &Row<c64>) -> Option<Point>;

    fn t_to_s(&self) -> Option<Point>;

    fn t_to_y(&self, z0: &Row<c64>) -> Option<Point>;

    fn t_to_z(&self, z0: &Row<c64>) -> Option<Point>;

    fn y_to_a(&self) -> Option<Point>;

    fn y_to_g(&self) -> Option<Point>;

    fn y_to_h(&self) -> Option<Point>;

    fn y_to_s(&self, z0: &Row<c64>) -> Option<Point>;

    fn y_to_t(&self, z0: &Row<c64>) -> Option<Point>;

    fn y_to_z(&self) -> Option<Point>;

    fn z_to_a(&self) -> Option<Point>;

    fn z_to_g(&self) -> Option<Point>;

    fn z_to_h(&self) -> Option<Point>;

    fn z_to_s(&self, z0: &Row<c64>) -> Option<Point>;

    fn z_to_t(&self, z0: &Row<c64>) -> Option<Point>;

    fn z_to_y(&self) -> Option<Point>;
}

impl NetworkPoint for Point
{
    fn a_to_g(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        Some(PartialPivLu::new(self.a_to_h().unwrap().as_ref()).inverse())
    }

    fn a_to_h(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = *self.get(0, 0);
        let b = *self.get(0, 1);
        let c = *self.get(1, 0);
        let d = *self.get(1, 1);

        let denom = d;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let h11 = b / denom;
        let h12 = (a * d - b * c) / denom;
        let h21 = -1.0 / denom;
        let h22 = c / denom;

        Some(mat![
            [h11, h12],
            [h21, h22],
        ])
    }

    fn a_to_s(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = *self.get(0, 0);
        let b = *self.get(0, 1);
        let c = *self.get(1, 0);
        let d = *self.get(1, 1);

        let denom = a * z0[1] + b + c * z0[0] * z0[1] + d * z0[0];
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let s11 = (a * z0[1] + b - c * z0[0].conj() * z0[1] - d * z0[0].conj()) / denom;
        let s12 = (2.0 * (a * d - b * c) * (z0[0].re * z0[1].re).sqrt()) / denom;
        let s21 = (2.0 * (z0[0].re * z0[1].re).sqrt()) / denom;
        let s22 = (-a * z0[1].conj() + b - c * z0[0] * z0[1].conj() + d * z0[0]) / denom;

        Some(mat![
            [s11, s12],
            [s21, s22],
        ])
    }

    fn a_to_t(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = *self.get(0, 0);
        let b = *self.get(0, 1);
        let c = *self.get(1, 0);
        let d = *self.get(1, 1);

        let denom = 2.0 * (z0[0].re * z0[1].re).sqrt();
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let t11 = (a * z0[1] + b + c * z0[0] * z0[1] + d * z0[0]) / denom;
        let t12 = (a * z0[1].conj() - b + c * z0[0] * z0[1].conj() - d * z0[0]) / denom;
        let t21 = (a * z0[1] + b - c * z0[0].conj() * z0[1] - d * z0[0].conj()) / denom;
        let t22 = (a * z0[1].conj() - b - c * z0[0].conj() * z0[1].conj() + d * z0[0].conj()) / denom;

        Some(mat![
            [t11, t12],
            [t21, t22],
        ])
    }

    fn a_to_y(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = *self.get(0, 0);
        let b = *self.get(0, 1);
        let c = *self.get(1, 0);
        let d = *self.get(1, 1);

        let denom = b;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let y11 = d / denom;
        let y12 = (b * c - a * d) / denom;
        let y21 = -1.0 / denom;
        let y22 = a / denom;

        Some(mat![
            [y11, y12],
            [y21, y22],
        ])
    }

    fn a_to_z(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let a = *self.get(0, 0);
        let b = *self.get(0, 1);
        let c = *self.get(1, 0);
        let d = *self.get(1, 1);

        let denom = c;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let z11 = a / denom;
        let z12 = (a * d - b * c) / denom;
        let z21 = 1.0 / denom;
        let z22 = d / denom;

        Some(mat![
            [z11, z12],
            [z21, z22],
        ])
    }

    // fn connect(&self, k: usize, net2: &Point, l: usize) -> Point {
    //     let mut data: Vec<c64> = Vec::new();
    //     let nrows1 = self.nrows();
    //     let nrows2 = net2.nrows();
    //     let size = nrows1 + nrows2;

    //     for i in 0..size {
    //         for j in 0..size {
    //             if i < nrows1 && j < nrows1 {
    //                 data.push(self[(j, i)]);
    //             } else if i >= nrows1 && k >= nrows1 {
    //                 data.push(net2[(j - nrows1, i - nrows1)]);
    //             } else {
    //                 data.push(c64::new(0.0, 0.0));
    //             }
    //         }
    //     }

    //     Point::from_vec(data)
    // }

    // fn is_reciprocal(&self) -> bool {
    //     if self.nrows() != 2 || self.reciprocity().unwrap() != Pointf64::zeros(2, 2) {
    //         return false;
    //     } else {
    //         return true;
    //     }
    // }

    fn db(&self) -> Pointf64 {
        let mut pt= Pointf64::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt.write(i, j, 20.0 * (self.get(i, j)).abs().log10());
            }
        }

        pt
    }

    fn deg(&self) -> Pointf64 {
        let mut pt = Pointf64::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt.write(i, j, (self.get(i, j)).arg() * 180.0 / PI);
            }
        }

        pt
    }

    // fn from_row_iterator(data: Vec<c64>) -> Point {
    //     Point::from_row_iterator(data.into_iter())
    // }

    // fn from_vec(data: Vec<c64>) -> Point {
    //     Point::from_vec(data)
    // }

    fn g_to_a(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        PartialPivLu::new(self.as_ref()).inverse().h_to_a()
    }

    fn g_to_h(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        Some(PartialPivLu::new(self.as_ref()).inverse())
    }

    fn g_to_s(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        PartialPivLu::new(self.as_ref()).inverse().h_to_s(z0)
    }

    fn g_to_t(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        PartialPivLu::new(self.as_ref()).inverse().h_to_t(z0)
    }

    fn g_to_y(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        PartialPivLu::new(self.as_ref()).inverse().h_to_y()
    }

    fn g_to_z(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        PartialPivLu::new(self.as_ref()).inverse().h_to_z()
    }

    fn h_to_a(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = *self.get(0, 0);
        let h12 = *self.get(0, 1);
        let h21 = *self.get(1, 0);
        let h22 = *self.get(1, 1);

        let denom = h21;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let a = (h12 * h21 - h11 * h22) / denom;
        let b = -h11 / denom;
        let c = -h22 / denom;
        let d = -1.0 / denom;

        Some(mat![
            [a, b],
            [c, d],
        ])
    }

    fn h_to_g(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        Some(PartialPivLu::new(self.as_ref()).inverse())
    }

    fn h_to_s(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = *self.get(0, 0);
        let h12 = *self.get(0, 1);
        let h21 = *self.get(1, 0);
        let h22 = *self.get(1, 1);

        let denom = (z0[0] + h11) * (1.0 + h22 * z0[1])
            - h12 * h21 * z0[1];
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let s11 = ((h11 - z0[0].conj()) * (1.0 + h22 * z0[1]) - h12 * h21 * z0[1]) / denom;
        let s12 = (2.0 * h12 * (z0[0].re * z0[1].re).sqrt()) / denom;
        let s21 = (-2.0 * h21 * (z0[0].re * z0[1].re).sqrt()) / denom;
        let s22 = ((z0[0] + h11) * (1.0 - h22 * z0[1].conj()) + h12 * h21 * z0[1].conj()) / denom;

        Some(mat![
            [s11, s12],
            [s21, s22],
        ])
    }

    fn h_to_t(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = *self.get(0, 0);
        let h12 = *self.get(0, 1);
        let h21 = *self.get(1, 0);
        let h22 = *self.get(1, 1);

        let denom = 2.0 * h21 * (z0[0].re * z0[1].re).sqrt();
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let t11 = ((-h11 - z0[0]) * (1.0 + h22 * z0[1]) + h12 * h21 * z0[1]) / denom;
        let t12 = ((h11 + z0[0]) * (1.0 - h22 * z0[1].conj()) + h12 * h21 * z0[1].conj()) / denom;
        let t21 = ((z0[0].conj() - h11) * (1.0 + h22 * z0[1]) + h12 * h21 * z0[1].conj()) / denom;
        let t22 = ((h11 - z0[0].conj()) * (1.0 - h22 * z0[1].conj()) + h12 * h21 * z0[1].conj()) / denom;

        Some(mat![
            [t11, t12],
            [t21, t22],
        ])
    }

    fn h_to_y(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = *self.get(0, 0);
        let h12 = *self.get(0, 1);
        let h21 = *self.get(1, 0);
        let h22 = *self.get(1, 1);

        let denom = h11;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let y11 = 1.0 / denom;
        let y12 = -h12 / denom;
        let y21 = h21 / denom;
        let y22 = (h11 * h22 - h12 * h21) / denom;

        Some(mat![
            [y11, y12],
            [y21, y22],
        ])
    }

    fn h_to_z(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let h11 = *self.get(0, 0);
        let h12 = *self.get(0, 1);
        let h21 = *self.get(1, 0);
        let h22 = *self.get(1, 1);

        let denom = h22;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let z11 = (h11 * h22 - h12 * h21) / denom;
        let z12 = h12 / denom;
        let z21 = -h21 / denom;
        let z22 = 1.0 / denom;

        Some(mat![
            [z11, z12],
            [z21, z22],
        ])
    }

    // fn identity() -> Point {
    //     Point::one()
    // }

    fn im(&self) -> Pointf64 {
        let mut pt = Pointf64::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt.write(i, j, (self.get(i, j)).im());
            }
        }

        pt
    }

    fn is_reciprocal(&self) -> bool {
        if self.nrows() != 2 || self.reciprocity().unwrap() != Pointf64::zeros(2, 2) {
            return false;
        } else {
            return true;
        }
    }

    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    fn mag(&self) -> Pointf64 {
        let mut pt= Pointf64::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt.write(i, j, (self.get(i, j)).abs());
            }
        }

        pt
    }

    fn new_like(pt: &Point) -> Point {
        Point::zeros(pt.nrows(), pt.ncols())
    }

    // fn ones() -> Point {
    //     let mut val = Point::zeros();
    //     val.fill(c64::new(1.0, 0.0));
    //     val
    // }

    fn rad(&self) -> Pointf64 {
        let mut pt = Pointf64::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt.write(i, j, (self.get(i, j)).arg());
            }
        }

        pt
    }

    fn re(&self) -> Pointf64 {
        let mut pt = Pointf64::zeros(self.nrows(), self.ncols());

        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                pt.write(i, j, (self.get(i, j)).re());
            }
        }

        pt
    }

    fn reciprocity(&self) -> Option<Pointf64> {
        let nrows = self.nrows();
        if nrows != 2 {
            return None;
        }

        let diff = self - self.transpose();
        let mut out: Pointf64 = Pointf64::zeros(self.nrows(), self.ncols());
        for j in 0..nrows {
            for k in 0..nrows {
                out.write(j, k, diff.get(j, k).abs());
            }
        }

        Some(out)
    }

    fn s_to_a(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = *self.get(0, 0);
        let s12 = *self.get(0, 1);
        let s21 = *self.get(1, 0);
        let s22 = *self.get(1, 1);

        let denom = 2.0 * s21 * (z0[0].re * z0[1].re).sqrt();
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let a = ((z0[0].conj() + s11 * z0[0]) * (1.0 - s22) + s12 * s21 * z0[0]) / denom;
        let b = ((z0[0].conj() + s11 * z0[0]) * (z0[1].conj() + s22 * z0[1]) - s12 * s21 * z0[0] * z0[1]) / denom;
        let c = ((1.0 - s11) * (1.0 - s22) - s12 * s21) / denom;
        let d = ((1.0 - s11) * (z0[1].conj() + s22 * z0[1]) + s12 * s21 * z0[1]) / denom;

        Some(mat![
            [a, b],
            [c, d],
        ])
    }

    fn s_to_g(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        Some(PartialPivLu::new(self.s_to_h(z0).unwrap().as_ref()).inverse())
    }

    fn s_to_h(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = *self.get(0, 0);
        let s12 = *self.get(0, 1);
        let s21 = *self.get(1, 0);
        let s22 = *self.get(1, 1);

        let denom = (1.0 - s11) * (z0[1].conj() + s22 * z0[1])
            + s12 * s21 * z0[1];
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let h11 = ((z0[0].conj() + s11 * z0[0]) * (z0[1].conj() + s22 * z0[1]) - s12 * s21 * z0[0] * z0[1]) / denom;
        let h12 = (2.0 * s12 * (z0[0].re * z0[1].re).sqrt()) / denom;
        let h21 = (-2.0 * s21 * (z0[0].re * z0[1].re).sqrt()) / denom;
        let h22 = ((1.0 - s11) * (1.0 - s22) - s12 * s21) / denom;

        Some(mat![
            [h11, h12],
            [h21, h22],
        ])
    }

    fn s_to_t(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let s11 = *self.get(0, 0);
        let s12 = *self.get(0, 1);
        let s21 = *self.get(1, 0);
        let s22 = *self.get(1, 1);

        let denom = s21;
        if denom.is_nan() || denom == 0.0.into() {
            panic!("Conversion undefined!");
        }

        let t11 = (s12 * s21 - s11 * s22) / denom;
        let t12 = s11 / denom;
        let t21 = -s22 / denom;
        let t22 = 1.0 / denom;

        Some(mat![
            [t11, t12],
            [t21, t22],
        ])
    }

    fn s_to_y(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() {
            return None;
        }
        let id = Mat::<c64>::identity(self.nrows(), self.ncols());
        let sqz0inv = Mat::<c64>::from_fn(self.nrows(), self.ncols(), |i, j| if i == j {1.0 / z0[i].sqrt()} else {c64::from(0.0)});

        let val = Some(PartialPivLu::new((id.as_ref() + self.as_ref()).as_ref()).inverse());
        match val {
            Some(x) => Some(sqz0inv.as_ref() * x * (id.as_ref() - self) * sqz0inv.as_ref()),
            None => None,
        }
    }

    fn s_to_z(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() {
            return None;
        }
        let id = Mat::<c64>::identity(self.nrows(), self.ncols());
        let sqz0 = Mat::<c64>::from_fn(self.nrows(), self.ncols(), |i, j| if i == j {z0[i].sqrt()} else {c64::from(0.0)});

        let val = Some(PartialPivLu::new((id.as_ref() - self.as_ref()).as_ref()).inverse());
        match val {
            Some(x) => Some(sqz0.as_ref() * x * (id.as_ref() + self) * sqz0.as_ref()),
            None => None,
        }
    }

    fn t_to_a(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = *self.get(0, 0);
        let t12 = *self.get(0, 1);
        let t21 = *self.get(1, 0);
        let t22 = *self.get(1, 1);

        let denom = 2.0 * (z0[0].re * z0[1].re).sqrt();
        if denom.is_nan() || denom == 0.0 {
            return None;
        }

        let a = (z0[0].conj() * (t11 + t12) + z0[0] * (t21 + t22)) / denom;
        let b = (z0[1].conj() * (t11 * z0[0].conj() + t21 * z0[0]) - z0[1] * (t12 * z0[0].conj() + t22 * z0[0])) / denom;
        let c = (t11 + t12 - t21 - t22) / denom;
        let d = (z0[1].conj() * (t11 - t21) - z0[1] * (t12 - t22)) / denom;

        Some(mat![
            [a, b],
            [c, d],
        ])
    }

    fn t_to_g(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        Some(PartialPivLu::new(self.t_to_h(z0).unwrap().as_ref()).inverse())
    }

    fn t_to_h(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = *self.get(0, 0);
        let t12 = *self.get(0, 1);
        let t21 = *self.get(1, 0);
        let t22 = *self.get(1, 1);

        let denom =
        z0[1].conj() * (t11 - t21) - z0[1] * (t12 + t22);
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let h11 = (z0[1].conj() * (t11 * z0[0].conj() + t21 * z0[0]) - z0[1] * (t12 * z0[0].conj() + t22 * z0[0])) / denom;
        let h12 = (2.0 * (z0[0].re * z0[1].re).sqrt() * (t11 * t22 - t12 * t21)) / denom;
        let h21 = (-2.0 * (z0[0].re * z0[1].re).sqrt()) / denom;
        let h22 = (t11 + t12 - t21 - t22) / denom;

        Some(mat![
            [h11, h12],
            [h21, h22],
        ])
    }

    fn t_to_s(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = *self.get(0, 0);
        let t12 = *self.get(0, 1);
        let t21 = *self.get(1, 0);
        let t22 = *self.get(1, 1);

        let denom = t22;
        if denom.is_nan() || denom == 0.0.into() {
            panic!("Conversion undefined!");
        }

        let s11 = t12 / denom;
        let s12 = (t11 * t22 - t12 * t21) / denom;
        let s21 = 1.0 / denom;
        let s22 = -t21 / denom;

        Some(mat![
            [s11, s12],
            [s21, s22],
        ])
    }

    fn t_to_y(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = *self.get(0, 0);
        let t12 = *self.get(0, 1);
        let t21 = *self.get(1, 0);
        let t22 = *self.get(1, 1);

        let denom = t11 * z0[0].conj() * z0[1].conj() - t12 * z0[0].conj() * z0[1] + t21 * z0[0] * z0[1].conj() - t22 * z0[0] * z0[1];
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let y11 = ((t11 - t21) * z0[1].conj() - (t12 - t22) * z0[1]) / denom;
        let y12 = (-2.0 * (z0[0].re * z0[1].re).sqrt() * (t11 * t22 - t12 * t21)) / denom;
        let y21 = (-2.0 * (z0[0].re * z0[1].re).sqrt()) / denom;
        let y22 = ((t11 + t12) * z0[0].conj() + (t21 + t22) * z0[0]) / denom;

        Some(mat![
            [y11, y12],
            [y21, y22],
        ])
    }

    fn t_to_z(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let t11 = *self.get(0, 0);
        let t12 = *self.get(0, 1);
        let t21 = *self.get(1, 0);
        let t22 = *self.get(1, 1);

        let denom = t11 + t12 - t21 - t22;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let z11 = ((t11 + t12) * z0[0].conj() + (t21 + t22) * z0[0]) / denom;
        let z12 = (2.0 * (z0[0].re * z0[1].re).sqrt() * (t11 * t22 - t12 * t21)) / denom;
        let z21 = (2.0 * (z0[0].re * z0[1].re).sqrt()) / denom;
        let z22 = (z0[1].conj() * (t11 - t21) - z0[1] * (t12 - t22)) / denom;

        Some(mat![
            [z11, z12],
            [z21, z22],
        ])
    }

    fn y_to_a(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = *self.get(0, 0);
        let y12 = *self.get(0, 1);
        let y21 = *self.get(1, 0);
        let y22 = *self.get(1, 1);

        let denom = y21;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let a = -y22 / denom;
        let b = -1.0 / denom;
        let c = (y12 * y21 - y11 * y22) / denom;
        let d = -y11 / denom;

        Some(mat![
            [a, b],
            [c, d],
        ])
    }

    fn y_to_g(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        Some(PartialPivLu::new(self.y_to_h().unwrap().as_ref()).inverse())
    }

    fn y_to_h(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = *self.get(0, 0);
        let y12 = *self.get(0, 1);
        let y21 = *self.get(1, 0);
        let y22 = *self.get(1, 1);

        let denom = y11;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let h11 = 1.0 / denom;
        let h12 = -y12 / denom;
        let h21 = y21 / denom;
        let h22 = (y11 * y22 - y12 * y21) / denom;

        Some(mat![
            [h11, h12],
            [h21, h22],
        ])
    }

    fn y_to_s(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() {
            return None;
        }
        let id = Mat::<c64>::identity(self.nrows(), self.ncols());
        let sqz0 = Mat::<c64>::from_fn(self.nrows(), self.ncols(), |i, j| if i == j {z0[i].sqrt()} else {c64::from(0.0)});

        let val = Some(PartialPivLu::new((id.as_ref() + sqz0.as_ref() * self.as_ref() * sqz0.as_ref()).as_ref()).inverse());
        match val {
            Some(x) => Some(x * (id.as_ref() - sqz0.as_ref() * self.as_ref() * sqz0.as_ref())),
            None => None,
        }
    }

    fn y_to_t(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let y11 = *self.get(0, 0);
        let y12 = *self.get(0, 1);
        let y21 = *self.get(1, 0);
        let y22 = *self.get(1, 1);

        let denom = 2.0 * y21 * (z0[0].re * z0[1].re).sqrt();
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let t11 = ((-1.0 - y11 * z0[0]) * (1.0 + y22 * z0[1]) + y12 * y21 * z0[0] * z0[1]) / denom;
        let t12 = ((1.0 + y11 * z0[0]) * (1.0 - y22 * z0[1].conj()) + y12 * y21 * z0[0] * z0[1].conj()) / denom;
        let t21 = ((y11 * z0[0].conj() - 1.0) * (1.0 + y22 * z0[1]) - y12 * y21 * z0[0].conj() * z0[1]) / denom;
        let t22 = ((1.0 - y11 * z0[0].conj()) * (1.0 - y22 * z0[1].conj()) - y12 * y21 * z0[0].conj() * z0[1].conj()) / denom;

        Some(mat![
            [t11, t12],
            [t21, t22],
        ])
    }

    fn y_to_z(&self) -> Option<Point> {
        if !self.is_square() {
            return None;
        }
        Some(PartialPivLu::new(self.as_ref()).inverse())
    }

    fn z_to_a(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = *self.get(0, 0);
        let z12 = *self.get(0, 1);
        let z21 = *self.get(1, 0);
        let z22 = *self.get(1, 1);

        let denom = z21;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let a = z11 / denom;
        let b = (z11 * z22 - z12 * z21) / denom;
        let c = 1.0 / denom;
        let d = z22 / denom;

        Some(mat![
            [a, b],
            [c, d],
        ])
    }

    fn z_to_g(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        Some(PartialPivLu::new(self.z_to_h().unwrap().as_ref()).inverse())
    }

    fn z_to_h(&self) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = *self.get(0, 0);
        let z12 = *self.get(0, 1);
        let z21 = *self.get(1, 0);
        let z22 = *self.get(1, 1);

        let denom = z22;
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let h11 = (z11 * z22 - z12 * z21) / denom;
        let h12 = z12 / denom;
        let h21 = -z21 / denom;
        let h22 = 1.0 / denom;

        Some(mat![
            [h11, h12],
            [h21, h22],
        ])
    }

    fn z_to_s(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() {
            return None;
        }
        let id = Mat::<c64>::identity(self.nrows(), self.ncols());
        let sqz0inv = Mat::<c64>::from_fn(self.nrows(), self.ncols(), |i, j| if i == j {1.0 / z0[i].sqrt()} else {c64::from(0.0)});

        let val = Some(PartialPivLu::new((sqz0inv.as_ref() * self.as_ref() * sqz0inv.as_ref() + id.as_ref()).as_ref()).inverse());
        match val {
            Some(x) => Some(x * (sqz0inv.as_ref() * self.as_ref() * sqz0inv.as_ref() - id.as_ref())),
            None => None,
        }
    }

    fn z_to_t(&self, z0: &Row<c64>) -> Option<Point> {
        if !self.is_square() || self.nrows() != 2 {
            return None;
        }
        let z11 = *self.get(0, 0);
        let z12 = *self.get(0, 1);
        let z21 = *self.get(1, 0);
        let z22 = *self.get(1, 1);

        let denom = 2.0 * z21 * (z0[0].re * z0[1].re).sqrt();
        if denom.is_nan() || denom == 0.0.into() {
            return None;
        }

        let t11 = ((z11 + z0[0]) * (z22 + z0[1]) - z12 * z21) / denom;
        let t12 = ((z11 + z0[0]) * (z0[1].conj() - z22) + z12 * z21) / denom;
        let t21 = ((z11 - z0[0].conj()) * (z22 + z0[1]) - z12 * z21) / denom;
        let t22 = ((z0[0].conj() - z11) * (z22 - z0[1].conj()) + z12 * z21) / denom;

        Some(mat![
            [t11, t12],
            [t21, t22],
        ])
    }

    fn z_to_y(&self) -> Option<Point> {
        if !self.is_square() {
            return None;
        }
        Some(PartialPivLu::new(self.as_ref()).inverse())
    }
}

trait NetworkPortPointsf64
{
    fn db(&self) -> PortPointsf64;

    fn new_like(pt: &Pointsf64) -> PortPointsf64;
}

impl NetworkPortPointsf64 for PortPointsf64
{
    fn db(&self) -> PortPointsf64 {
        let mut out: PortPointsf64 = vec![];

        for pt in self.iter() {
            out.push(20.0 * pt.log10());
        }

        out
    }

    fn new_like(pt: &Pointsf64) -> PortPointsf64 {
        vec![0.0; pt.len()]
    }
}

trait NetworkPortPoints
{
    fn db(&self) -> PortPointsf64;

    fn deg(&self) -> PortPointsf64;

    fn im(&self) -> PortPointsf64;

    fn mag(&self) -> PortPointsf64;

    fn new_like(pt: &Points) -> PortPoints;

    fn rad(&self) -> PortPointsf64;

    fn re(&self) -> PortPointsf64;
}

impl NetworkPortPoints for PortPoints
{
    fn db(&self) -> PortPointsf64 {
        let mut out: PortPointsf64 = vec![];
        for pt in self.iter() {
            out.push(20.0 * pt.abs().log10());
        }
        out
    }

    fn deg(&self) -> PortPointsf64 {
        let mut out: PortPointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.arg() * 180.0 / PI);
        }
        out
    }

    fn im(&self) -> PortPointsf64 {
        let mut out: PortPointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.im());
        }
        out
    }

    fn mag(&self) -> PortPointsf64 {
        let mut out: PortPointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.abs());
        }
        out
    }

    fn new_like(pt: &Points) -> PortPoints {
        vec![c64::from(0.0); pt.len()]
    }

    fn rad(&self) -> PortPointsf64 {
        let mut out: PortPointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.arg());
        }
        out
    }

    fn re(&self) -> PortPointsf64 {
        let mut out: PortPointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.re());
        }
        out
    }
}

trait NetworkPoints
{
    fn a_to_g(&self) -> Option<Points>;

    fn a_to_h(&self) -> Option<Points>;

    fn a_to_s(&self, z0: &Row<c64>) -> Option<Points>;

    fn a_to_t(&self, z0: &Row<c64>) -> Option<Points>;

    fn a_to_y(&self) -> Option<Points>;

    fn a_to_z(&self) -> Option<Points>;

    // fn connect(&self, k: usize, net2: &Points, l: usize) -> Points;

    fn copy_from(&self) -> Points;

    fn db(&self) -> Pointsf64;

    fn deg(&self) -> Pointsf64;

    // fn from_row_iterator(data: Vec<c64>) -> Points;

    // fn from_vec(data: Vec<c64>) -> Points;

    fn g_to_a(&self) -> Option<Points>;

    fn g_to_h(&self) -> Option<Points>;

    fn g_to_s(&self, z0: &Row<c64>) -> Option<Points>;

    fn g_to_t(&self, z0: &Row<c64>) -> Option<Points>;

    fn g_to_y(&self) -> Option<Points>;

    fn g_to_z(&self) -> Option<Points>;

    fn h_to_a(&self) -> Option<Points>;

    fn h_to_g(&self) -> Option<Points>;

    fn h_to_s(&self, z0: &Row<c64>) -> Option<Points>;

    fn h_to_t(&self, z0: &Row<c64>) -> Option<Points>;

    fn h_to_y(&self) -> Option<Points>;

    fn h_to_z(&self) -> Option<Points>;

    // fn identity() -> Points;

    fn im(&self) -> Pointsf64;

    fn is_square(&self) -> bool;

    fn is_reciprocal(&self) -> bool;

    fn mag(&self) -> Pointsf64;

    fn new_like(pt: &Points) -> Points;

    // fn ones() -> Points;

    fn rad(&self) -> Pointsf64;

    fn re(&self) -> Pointsf64;

    // fn reciprocity(&self) -> Option<Pointsf64>;

    fn s_to_a(&self, z0: &Row<c64>) -> Option<Points>;

    fn s_to_g(&self, z0: &Row<c64>) -> Option<Points>;

    fn s_to_h(&self, z0: &Row<c64>) -> Option<Points>;

    fn s_to_t(&self) -> Option<Points>;

    fn s_to_y(&self, z0: &Row<c64>) -> Option<Points>;

    fn s_to_z(&self, z0: &Row<c64>) -> Option<Points>;

    fn t_to_a(&self, z0: &Row<c64>) -> Option<Points>;

    fn t_to_g(&self, z0: &Row<c64>) -> Option<Points>;

    fn t_to_h(&self, z0: &Row<c64>) -> Option<Points>;

    fn t_to_s(&self) -> Option<Points>;

    fn t_to_y(&self, z0: &Row<c64>) -> Option<Points>;

    fn t_to_z(&self, z0: &Row<c64>) -> Option<Points>;

    fn y_to_a(&self) -> Option<Points>;

    fn y_to_g(&self) -> Option<Points>;

    fn y_to_h(&self) -> Option<Points>;

    fn y_to_s(&self, z0: &Row<c64>) -> Option<Points>;

    fn y_to_t(&self, z0: &Row<c64>) -> Option<Points>;

    fn y_to_z(&self) -> Option<Points>;

    fn z_to_a(&self) -> Option<Points>;

    fn z_to_g(&self) -> Option<Points>;

    fn z_to_h(&self) -> Option<Points>;

    fn z_to_s(&self, z0: &Row<c64>) -> Option<Points>;

    fn z_to_t(&self, z0: &Row<c64>) -> Option<Points>;

    fn z_to_y(&self) -> Option<Points>;
}

impl NetworkPoints for Points
{
    fn a_to_g(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.a_to_g().unwrap());
        }
        Some(out)
    }

    fn a_to_h(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.a_to_h().unwrap());
        }
        Some(out)
    }

    fn a_to_s(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.a_to_s(z0).unwrap());
        }
        Some(out)
    }

    fn a_to_t(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.a_to_t(z0).unwrap());
        }
        Some(out)
    }

    fn a_to_y(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.a_to_y().unwrap());
        }
        Some(out)
    }

    fn a_to_z(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.a_to_z().unwrap());
        }
        Some(out)
    }

    // fn connect(&self, k: usize, net2: &Point, l: usize) -> Point {
    //     let mut data: Vec<c64> = Vec::new();
    //     let nrows1 = self.nrows();
    //     let nrows2 = net2.nrows();
    //     let size = nrows1 + nrows2;

    //     for i in 0..size {
    //         for j in 0..size {
    //             if i < nrows1 && j < nrows1 {
    //                 data.push(self[(j, i)]);
    //             } else if i >= nrows1 && k >= nrows1 {
    //                 data.push(net2[(j - nrows1, i - nrows1)]);
    //             } else {
    //                 data.push(c64::new(0.0, 0.0));
    //             }
    //         }
    //     }

    //     Point::from_vec(data)
    // }

    // fn is_reciprocal(&self) -> bool {
    //     if self.nrows() != 2 || self.reciprocity().unwrap() != Pointf64::zeros(2, 2) {
    //         return false;
    //     } else {
    //         return true;
    //     }
    // }

    fn copy_from(&self) -> Points {
        let mut out: Points = vec![];
        for pt in self.iter() {
            let mut outpt = Point::zeros(self[0].nrows(), self[0].ncols());
            outpt.copy_from(pt.as_ref());
            out.push(outpt);
        }
        out
    }

    fn db(&self) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.db());
        }
        out
    }

    fn deg(&self) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.deg());
        }
        out
    }

    // fn from_row_iterator(data: Vec<c64>) -> Point {
    //     Point::from_row_iterator(data.into_iter())
    // }

    // fn from_vec(data: Vec<c64>) -> Point {
    //     Point::from_vec(data)
    // }

    fn g_to_a(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.g_to_a().unwrap());
        }
        Some(out)
    }

    fn g_to_h(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.g_to_h().unwrap());
        }
        Some(out)
    }

    fn g_to_s(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.g_to_s(z0).unwrap());
        }
        Some(out)
    }

    fn g_to_t(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.g_to_t(z0).unwrap());
        }
        Some(out)
    }

    fn g_to_y(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.g_to_y().unwrap());
        }
        Some(out)
    }

    fn g_to_z(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.g_to_z().unwrap());
        }
        Some(out)
    }

    fn h_to_a(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.h_to_a().unwrap());
        }
        Some(out)
    }

    fn h_to_g(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.h_to_g().unwrap());
        }
        Some(out)
    }

    fn h_to_s(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.h_to_s(z0).unwrap());
        }
        Some(out)
    }

    fn h_to_t(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.h_to_t(z0).unwrap());
        }
        Some(out)
    }

    fn h_to_y(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.h_to_y().unwrap());
        }
        Some(out)
    }

    fn h_to_z(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.h_to_z().unwrap());
        }
        Some(out)
    }

    // fn identity() -> Point {
    //     Point::one()
    // }

    fn im(&self) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.im());
        }
        out
    }

    fn is_reciprocal(&self) -> bool {
        for pt in self.iter() {
            if !pt.is_reciprocal() {
                return false;
            }
        }
        true
    }

    fn is_square(&self) -> bool {
        self[0].nrows() == self[0].ncols()
    }

    fn mag(&self) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.mag());
        }
        out
    }

    fn new_like(pt: &Points) -> Points {
        vec![Point::zeros(pt[0].nrows(), pt[0].ncols())]
    }

    // fn ones() -> Point {
    //     let mut val = Point::zeros();
    //     val.fill(c64::new(1.0, 0.0));
    //     val
    // }

    fn rad(&self) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.rad());
        }
        out
    }

    fn re(&self) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];
        for pt in self.iter() {
            out.push(pt.re());
        }
        out
    }

    // fn reciprocity(&self) -> Option<Pointf64> {
    //     let nrows = self.nrows();
    //     if nrows != 2 {
    //         return None;
    //     }

    //     let mut out_val = Pointf64::zeros();
    //     for j in 0..nrows {
    //         for k in 0..nrows {
    //             out_val[(j, k)] = (1.0 - self[(j, k)] / self[(k, j)]).abs();
    //         }
    //     }

    //     Some(out_val)
    // }

    fn s_to_a(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.s_to_a(z0).unwrap());
        }
        Some(out)
    }

    fn s_to_g(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.s_to_g(z0).unwrap());
        }
        Some(out)
    }

    fn s_to_h(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.s_to_h(z0).unwrap());
        }
        Some(out)
    }

    fn s_to_t(&self) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.s_to_t().unwrap());
        }
        Some(out)
    }

    fn s_to_y(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.s_to_y(z0).unwrap());
        }
        Some(out)
    }

    fn s_to_z(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.s_to_z(z0).unwrap());
        }
        Some(out)
    }

    fn t_to_a(&self, z0: &Row<c64>) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.t_to_a(z0).unwrap());
        }
        Some(out)
    }

    fn t_to_g(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.t_to_g(z0).unwrap());
        }
        Some(out)
    }

    fn t_to_h(&self, z0: &Row<c64>) -> Option<Points> {
        if !self.is_square() {
            return None;
        }
        if self[0].nrows() != 2 {
            return None;
        }
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.t_to_h(z0).unwrap());
        }
        Some(out)
    }

    fn t_to_s(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.t_to_s().unwrap());
        }
        Some(out)
    }

    fn t_to_y(&self, z0: &Row<c64>) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.t_to_y(z0).unwrap());
        }
        Some(out)
    }

    fn t_to_z(&self, z0: &Row<c64>) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.t_to_z(z0).unwrap());
        }
        Some(out)
    }

    fn y_to_a(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.y_to_a().unwrap());
        }
        Some(out)
    }

    fn y_to_g(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.y_to_g().unwrap());
        }
        Some(out)
    }

    fn y_to_h(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.y_to_h().unwrap());
        }
        Some(out)
    }

    fn y_to_s(&self, z0: &Row<c64>) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.y_to_s(z0).unwrap());
        }
        Some(out)
    }

    fn y_to_t(&self, z0: &Row<c64>) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.y_to_t(z0).unwrap());
        }
        Some(out)
    }

    fn y_to_z(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.y_to_z().unwrap());
        }
        Some(out)
    }

    fn z_to_a(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.z_to_a().unwrap());
        }
        Some(out)
    }

    fn z_to_g(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.z_to_g().unwrap());
        }
        Some(out)
    }

    fn z_to_h(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.z_to_h().unwrap());
        }
        Some(out)
    }

    fn z_to_s(&self, z0: &Row<c64>) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.z_to_s(z0).unwrap());
        }
        Some(out)
    }

    fn z_to_t(&self, z0: &Row<c64>) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.z_to_t(z0).unwrap());
        }
        Some(out)
    }

    fn z_to_y(&self) -> Option<Points> {
        let mut out: Points = vec![];
        for pt in self.iter() {
            out.push(pt.z_to_y().unwrap());
        }
        Some(out)
    }
}

#[derive(Clone)]
pub struct Network
{
    name: String,
    comments: String,
    nports: usize,
    port_names: Vec<String>,
    freq: Frequency,
    npts: usize,
    z0: Row<c64>,
    a: Option<Points>,
    g: Option<Points>,
    h: Option<Points>,
    s: Option<Points>,
    t: Option<Points>,
    y: Option<Points>,
    z: Option<Points>,
}

impl Network
{
    pub fn a(&self) -> &Points {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        &self.a.as_ref().unwrap()
    }

    pub fn a_db(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_db(RFParameter::A)
    }

    pub fn a_deg(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_deg(RFParameter::A)
    }

    pub fn a_im(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_im(RFParameter::A)
    }

    pub fn a_mag(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_mag(RFParameter::A)
    }

    pub fn a_rad(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_rad(RFParameter::A)
    }

    pub fn a_re(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::A, self.nports));
        }
        self.net_re(RFParameter::A)
    }

    pub fn chop_in_half(&self) -> Network {
        if self.nports() != 2 {
            panic!("Cannot chop_in_half network that is not 2 ports")
        } else if !self.is_reciprocal() {
            panic!("Cannot chop_in_half network that is not reciprocal")
        }

        let mut tmp: Vec<c64> = vec![];
        let mut out: Points = vec![];
        for pt in self.s.as_ref().unwrap().iter() {
            let b11 = *pt.get(0, 0);
            let b12 = *pt.get(0, 1);
            let b21 = *pt.get(1, 0);
            let b22 = *pt.get(1, 1);
            let a11 = b11 / (1.0 + b12);
            tmp.push(b21 * (1.0 - b11 * b22 / (1.0 + b12).powi(2)));
            let a22 = b22 / (1.0 + b12);
            out.push(mat![
                [a11, c64::from(0.0)],
                [c64::from(0.0), a22],
            ]);
        }

        tmp = sqrt_phase_unwrap(tmp);

        for i in 0..out.len() {
            out[i].write(0, 1, tmp[i]);
            out[i].write(1, 0, tmp[i]);
        }


        Network::new(self.freq.clone(), self.z0.clone(), RFParameter::S, out, self.name.clone(), self.comments.clone())
    }

    pub fn comments(&self) -> &String {
        &self.comments
    }

    pub fn freq(&self) -> &Frequency {
        &self.freq
    }

    pub fn g(&self) -> &Points {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        &self.g.as_ref().unwrap()
    }

    pub fn g_db(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_db(RFParameter::G)
    }

    pub fn g_deg(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_deg(RFParameter::G)
    }

    pub fn g_im(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_im(RFParameter::G)
    }

    pub fn g_mag(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_mag(RFParameter::G)
    }

    pub fn g_rad(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_rad(RFParameter::G)
    }

    pub fn g_re(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::G, self.nports));
        }
        self.net_re(RFParameter::G)
    }

    pub fn h(&self) -> &Points {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        &self.h.as_ref().unwrap()
    }

    pub fn h_db(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_db(RFParameter::H)
    }

    pub fn h_deg(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_deg(RFParameter::H)
    }

    pub fn h_im(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_im(RFParameter::H)
    }

    pub fn h_mag(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_mag(RFParameter::H)
    }

    pub fn h_rad(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_rad(RFParameter::H)
    }

    pub fn h_re(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::H, self.nports));
        }
        self.net_re(RFParameter::H)
    }

    pub fn is_reciprocal(&self) -> bool {
        for val in self.s().iter() {
            if !val.is_reciprocal() {
                return false;
            }
        }
        return true;
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn net(&self, param: RFParameter) -> &Points {
        match param {
            RFParameter::A => {
                if self.nports != 2 {
                    panic!("{}", format!("ABCD parameters do not exist for network with {nports} port(s)", nports = self.nports));
                }
                self.a.as_ref().unwrap()
            },
            RFParameter::G => {
                if self.nports != 2 {
                    panic!("{}", format!("Inverse Hybrid (g) parameters do not exist for network with {nports} port(s)", nports = self.nports));
                }
                self.g.as_ref().unwrap()
            },
            RFParameter::H => {
                if self.nports != 2 {
                    panic!("{}", format!("Hybrid (h) parameters do not exist for network with {nports} port(s)", nports = self.nports));
                }
                self.h.as_ref().unwrap()
            },
            RFParameter::S => self.s.as_ref().unwrap(),
            RFParameter::T => {
                self.t.as_ref().unwrap()
            },
            RFParameter::Y => self.y.as_ref().unwrap(),
            RFParameter::Z => self.z.as_ref().unwrap(),
        }
    }

    pub fn net_at_freq_idx(&self, param: RFParameter, idx: usize) -> &Point {
        &self.net(param)[idx]
    }

    pub fn net_at_port_idx(&self, param: RFParameter, j: usize, k: usize) -> PortPoints {
        let mut out: PortPoints = vec![];

        for pt in self.net(param).iter() {
            out.push(*pt.get(j, k));
        }

        out
    }

    pub fn net_at_idx(&self, param: RFParameter, idx: usize, j: usize, k: usize) -> &c64 {
        self.net_at_freq_idx(param, idx).get(j, k)
    }

    pub fn net_db(&self, param: RFParameter) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];

        for pt in self.net(param).iter() {
            out.push(pt.db());
        }

        out
    }

    pub fn net_deg(&self, param: RFParameter) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];

        for pt in self.net(param).iter() {
            out.push(pt.deg());
        }

        out
    }

    pub fn net_im(&self, param: RFParameter) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];

        for pt in self.net(param).iter() {
            out.push(pt.im());
        }

        out
    }

    pub fn net_mag(&self, param: RFParameter) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];

        for pt in self.net(param).iter() {
            out.push(pt.mag());
        }

        out
    }

    pub fn net_rad(&self, param: RFParameter) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];

        for pt in self.net(param).iter() {
            out.push(pt.rad());
        }

        out
    }

    pub fn net_re(&self, param: RFParameter) -> Pointsf64 {
        let mut out: Pointsf64 = vec![];

        for pt in self.net(param).iter() {
            out.push(pt.re());
        }

        out
    }

    pub fn new(
        freq: Frequency,
        z0: Row<c64>,
        param: RFParameter,
        net: Points,
        name: String,
        comments: String,
    ) -> Network {
        if !net[0].is_square() {
            panic!("{}", format!("Provided data is not square!\n{rows} rows\t\t{cols} cols", rows = net[0].nrows(), cols = net[0].ncols()));
        }
        if net[0].nrows() != z0.ncols() {
            panic!("{}", format!("Number of ports and number of Z0 points do not match!\n{npts} network points\t\t{zpts} port impedance points", npts = net[0].nrows(), zpts = z0.ncols()));
        }
        if freq.npts() != net.len() {
            panic!("{}", format!("Number of frequency points does not match number of data points!\n{fpts} frequency points\t\t{npts} network points", fpts = freq.npts(), npts = net.len()));
        }

        let nports = net[0].nrows();

        Network::new_w_port_names(
            freq,
            z0,
            param,
            net,
            name,
            comments,
            vec![String::from(""); nports],
        )
    }

    pub fn new_w_port_names(
        freq: Frequency,
        z0: Row<c64>,
        param: RFParameter,
        net: Points,
        name: String,
        comments: String,
        port_names: Vec<String>,
    ) -> Network {
        if !net[0].is_square() {
            panic!("{}", format!("Provided data is not square!\n{rows} rows\t\t{cols} cols", rows = net[0].nrows(), cols = net[0].ncols()));
        }
        if net[0].nrows() != z0.ncols() {
            panic!("{}", format!("Number of ports and number of Z0 points do not match!\n{npts} network points\t\t{zpts} port impedance points", npts = net[0].nrows(), zpts = z0.ncols()));
        }
        if freq.npts() != net.len() {
            panic!("{}", format!("Number of frequency points does not match number of data points!\n{fpts} frequency points\t\t{npts} network points", fpts = freq.npts(), npts = net.len()));
        }

        let mut out = Network {
            name: name,
            comments: comments,
            nports: net[0].nrows(),
            port_names: port_names,
            freq: freq,
            npts: net.len(),
            z0: z0.clone(),
            a: None,
            g: None,
            h: None,
            s: None,
            t: None,
            y: None,
            z: None,
        };

        if out.nports == 2 {
            match param {
                RFParameter::A => {
                    out.a = Some(net.copy_from());
                    out.g = net.copy_from().a_to_g();
                    out.h = net.copy_from().a_to_h();
                    out.s = net.copy_from().a_to_s(&z0);
                    out.t = net.copy_from().a_to_t(&z0);
                    out.y = net.copy_from().a_to_y();
                    out.z = net.copy_from().a_to_z();
                },
                RFParameter::G => {
                    out.a = net.copy_from().g_to_a();
                    out.g = Some(net.copy_from());
                    out.h = net.copy_from().g_to_h();
                    out.s = net.copy_from().g_to_s(&z0);
                    out.t = net.copy_from().g_to_t(&z0);
                    out.y = net.copy_from().g_to_y();
                    out.z = net.copy_from().g_to_z();
                },
                RFParameter::H => {
                    out.a = net.copy_from().h_to_a();
                    out.g = net.copy_from().h_to_g();
                    out.h = Some(net.copy_from());
                    out.s = net.copy_from().h_to_s(&z0);
                    out.t = net.copy_from().h_to_t(&z0);
                    out.y = net.copy_from().h_to_y();
                    out.z = net.copy_from().h_to_z();
                },
                RFParameter::S => {
                    out.a = net.copy_from().s_to_a(&z0);
                    out.g = net.copy_from().s_to_g(&z0);
                    out.h = net.copy_from().s_to_h(&z0);
                    out.s = Some(net.copy_from());
                    out.t = net.copy_from().s_to_t();
                    out.y = net.copy_from().s_to_y(&z0);
                    out.z = net.copy_from().s_to_z(&z0);
                },
                RFParameter::T => {
                    out.a = net.copy_from().t_to_a(&z0);
                    out.g = net.copy_from().t_to_g(&z0);
                    out.h = net.copy_from().t_to_h(&z0);
                    out.s = net.copy_from().t_to_s();
                    out.t = Some(net.copy_from());
                    out.y = net.copy_from().t_to_y(&z0);
                    out.z = net.copy_from().t_to_z(&z0);
                },
                RFParameter::Y => {
                    out.a = net.copy_from().y_to_a();
                    out.g = net.copy_from().y_to_g();
                    out.h = net.copy_from().y_to_h();
                    out.s = net.copy_from().y_to_s(&z0);
                    out.t = net.copy_from().y_to_t(&z0);
                    out.y = Some(net.copy_from());
                    out.z = net.copy_from().y_to_z();
                },
                RFParameter::Z => {
                    out.a = net.copy_from().z_to_a();
                    out.g = net.copy_from().z_to_g();
                    out.h = net.copy_from().z_to_h();
                    out.s = net.copy_from().z_to_s(&z0);
                    out.t = net.copy_from().z_to_t(&z0);
                    out.y = net.copy_from().z_to_y();
                    out.z = Some(net.copy_from());
                },
            }
        } else {
            match param {
                RFParameter::A | RFParameter::G | RFParameter::H | RFParameter::T => {
                    panic!("{}", network_err_msg(param, out.nports))
                }
                RFParameter::S => {
                    out.s = Some(net.copy_from());
                    out.y = net.copy_from().s_to_y(&z0);
                    out.z = net.copy_from().s_to_z(&z0);
                },
                RFParameter::Y => {
                    out.s = net.copy_from().y_to_s(&z0);
                    out.y = Some(net.copy_from());
                    out.z = net.copy_from().y_to_z();
                },
                RFParameter::Z => {
                    out.s = net.copy_from().z_to_s(&z0);
                    out.y = net.copy_from().z_to_y();
                    out.z = Some(net.copy_from());
                },
            }
        }

        out
    }

    pub fn nports(&self) -> usize {
        self.nports.clone()
    }

    pub fn npoints(&self) -> usize {
        self.npts.clone()
    }

    pub fn port_name(&self, n: usize) -> &String {
        &self.port_names[n]
    }

    pub fn port_names(&self) -> &Vec<String> {
        &self.port_names
    }

    pub fn reciprocity(&self) -> Vec<Mat<f64>> {
        let mut out: Vec<Mat<f64>> = Vec::new();

        for i in 0..self.npoints() {
            out.push(unwrap_or_panic!(
                self.s()[i].reciprocity(),
                "failed to calculate reciprocal {i}"
            ));
        }

        out
    }

    // //TODO cleanup the ability to import various files
    // pub fn read_touchstone(file_path: &String) -> Result<Network, Box<dyn Error>> {
    //     // Regex patterns
    //     let re_file_ext = Regex::new(r"\d+").expect("Invalid regex!");
    //     let re_file_opts = Regex::new(r"^\[(w+)\]\s+(\w+)").expect("Invalid regex!");
    //     let re_comment = Regex::new(r"^!").expect("Invalid regex!");
    //     let re_port_name = RegexBuilder::new(r"^!\s*Port\[(\d+)\]\s*=\s?(\w+)")
    //         .case_insensitive(true)
    //         .build()
    //         .expect("Invalid regex!");

    //     // file reading variables
    //     let content = fs::read_to_string(&file_path)?;
    //     let mut file_path_split = file_path.split('.').rev();
    //     let file_ext = file_path_split.next().unwrap();
    //     let filename = file_path_split.next().unwrap().split('/').last().unwrap();
    //     let mut lines = content.lines();

    //     // Network values
    //     let nports: usize = re_file_ext
    //         .find(&file_ext)
    //         .expect("Invalid file extension")
    //         .as_str()
    //         .parse()
    //         .unwrap();
    //     let mut port_names: Vec<String> = vec![String::from(""); nports];
    //     let mut freq_unit = FreqUnit::GHz;
    //     let mut parameter = RFParameter::S;
    //     let mut format = RFDataFormat::RI;
    //     let mut impedance = na::Complex { re: 50.0, im: 0.0 };
    //     let mut z0: Vec<c64> = Vec::new();
    //     let mut freq_tmp: Vec<f64> = Vec::new();
    //     let mut point_tmp = Point::zeros();
    //     let mut data: Vec<Mat<c64>> = Vec::new();
    //     let mut comments = String::new();

    //     let mut caps: Option<regex::Captures>;
    //     let mut nfreq_check: Option<usize>;
    //     let mut nfreq_noise_check: Option<usize>;

    //     loop {
    //         let line = unwrap_or_break!(lines.next());

    //         if line.starts_with("!") || line.starts_with("!%") {
    //             //
    //             // comment line
    //             //
    //             if data.len() > 0 || z0.len() > 0 {
    //                 continue;
    //             }
    //             if comments != "" {
    //                 comments += "\n";
    //             }
    //             comments += line.strip_prefix("!").unwrap();
    //         } else if line.starts_with("[") {
    //             let cap = re_file_opts.captures(line);

    //             if cap.is_none() {
    //                 continue;
    //             }

    //             let caps = cap.unwrap();

    //             match caps.get(1).unwrap().as_str() {
    //                 // TODO: Add additional Version 2.0 options handling
    //                 "[Number of Frequencies]" => {
    //                     nfreq_check = Some(caps.get(2).unwrap().as_str().parse::<usize>().unwrap());
    //                 }
    //                 "[Number of Noise Frequencies]" => {
    //                     nfreq_noise_check =
    //                         Some(caps.get(2).unwrap().as_str().parse::<usize>().unwrap());
    //                 }
    //                 "[End]" => break,
    //                 _ => continue,
    //             }
    //         } else if line.starts_with("#") {
    //             //
    //             // File format line
    //             //
    //             let mut vals = line.strip_prefix("#").unwrap().split_whitespace();

    //             freq_unit = FreqUnit::from_str(vals.next().unwrap())?;
    //             parameter = RFParameter::from_str(vals.next().unwrap())?;
    //             format = RFDataFormat::from_str(vals.next().unwrap())?;
    //             impedance = vals.last().unwrap().to_string().parse().unwrap();
    //             for i in 0..nports {
    //                 z0.push(impedance);
    //             }
    //         } else {
    //             //
    //             // data line
    //             //
    //             let mut fields = line.trim().split_whitespace();
    //             let mut val = fields.next();
    //             if val == None {
    //                 continue;
    //             }

    //             if nports <= 2 {
    //                 freq_tmp.push(val.unwrap().parse().unwrap());

    //                 for i in 0..nports * nports {
    //                     point_tmp[i] = format.parse(
    //                         fields.next().unwrap().parse().unwrap(),
    //                         fields.next().unwrap().parse().unwrap(),
    //                     );
    //                 }
    //             } else {
    //                 for j in 0..nports {
    //                     if j == 0 {
    //                         freq_tmp.push(val.unwrap().parse().unwrap());
    //                     } else {
    //                         let line = lines.next();
    //                         if line == None {
    //                             break;
    //                         }
    //                         fields = line.unwrap().trim().split_whitespace();
    //                     }
    //                     for k in 0..nports {
    //                         point_tmp[(j, k)] = format.parse(
    //                             fields.next().unwrap().parse().unwrap(),
    //                             fields.next().unwrap().parse().unwrap(),
    //                         );
    //                     }
    //                 }
    //             }

    //             data.push(point_tmp.clone());
    //         }
    //     }

    //     Ok(Network {
    //         name: String::from(filename),
    //         comments: comments,
    //         port_names: vec![String::from(""); nports],
    //         freq: Frequency::from_vec(freq_tmp, freq_unit),
    //         z0: z0,
    //         param: parameter,
    //         net: data,
    //     })
    // }

    pub fn s(&self) -> &Points {
        &self.s.as_ref().unwrap()
    }

    pub fn s_db(&self) -> Pointsf64 {
        self.net_db(RFParameter::S)
    }

    pub fn s_deg(&self) -> Pointsf64 {
        self.net_deg(RFParameter::S)
    }

    pub fn s_im(&self) -> Pointsf64 {
        self.net_im(RFParameter::S)
    }

    pub fn s_mag(&self) -> Pointsf64 {
        self.net_mag(RFParameter::S)
    }

    pub fn s_rad(&self) -> Pointsf64 {
        self.net_rad(RFParameter::S)
    }

    pub fn s_re(&self) -> Pointsf64 {
        self.net_re(RFParameter::S)
    }

    pub fn set_port_name(&mut self, n: usize, name: String) {
        self.port_names[n] = name;
    }

    pub fn set_port_names(&mut self, names: Vec<String>) {
        self.port_names = names;
    }

    pub fn t(&self) -> &Points {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        &self.t.as_ref().unwrap()
    }

    pub fn t_db(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_db(RFParameter::T)
    }

    pub fn t_deg(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_deg(RFParameter::T)
    }

    pub fn t_im(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_im(RFParameter::T)
    }

    pub fn t_mag(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_mag(RFParameter::T)
    }

    pub fn t_rad(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_rad(RFParameter::T)
    }

    pub fn t_re(&self) -> Pointsf64 {
        if self.nports != 2 {
            panic!("{}", network_err_msg(RFParameter::T, self.nports));
        }
        self.net_re(RFParameter::T)
    }

    pub fn y(&self) -> &Points {
        &self.y.as_ref().unwrap()
    }

    pub fn y_db(&self) -> Pointsf64 {
        self.net_db(RFParameter::Y)
    }

    pub fn y_deg(&self) -> Pointsf64 {
        self.net_deg(RFParameter::Y)
    }

    pub fn y_im(&self) -> Pointsf64 {
        self.net_im(RFParameter::Y)
    }

    pub fn y_mag(&self) -> Pointsf64 {
        self.net_mag(RFParameter::Y)
    }

    pub fn y_rad(&self) -> Pointsf64 {
        self.net_rad(RFParameter::Y)
    }

    pub fn y_re(&self) -> Pointsf64 {
        self.net_re(RFParameter::Y)
    }

    pub fn z(&self) -> &Points {
        &self.z.as_ref().unwrap()
    }

    pub fn z_db(&self) -> Pointsf64 {
        self.net_db(RFParameter::Z)
    }

    pub fn z_deg(&self) -> Pointsf64 {
        self.net_deg(RFParameter::Z)
    }

    pub fn z_im(&self) -> Pointsf64 {
        self.net_im(RFParameter::Z)
    }

    pub fn z_mag(&self) -> Pointsf64 {
        self.net_mag(RFParameter::Z)
    }

    pub fn z_rad(&self) -> Pointsf64 {
        self.net_rad(RFParameter::Z)
    }

    pub fn z_re(&self) -> Pointsf64 {
        self.net_re(RFParameter::Z)
    }

    pub fn z0(&self) -> &Row<c64> {
        &self.z0
    }

    pub fn z0_at_port_idx(&self, idx: usize) -> &c64 {
        &self.z0[idx]
    }

    // pub fn z0_to_vector(
    //     &self,
    // ) -> Vector<c64, R, <DefaultAllocator as Allocator<c64, R>>::Buffer> {
    //     na::Vector::<
    //         c64,
    //         R,
    //         <DefaultAllocator as na::allocator::Allocator<c64, R>>::Buffer,
    //     >::from_row_iterator(self.z0.clone().into_iter())
    // }
}

impl fmt::Display for Network
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name: {name}\nnports: {nports}\nnpoints: {npoints}\nz0: {z0:?}",
            name = self.name,
            nports = self.nports(),
            npoints = self.npoints(),
            z0 = self.z0,
            )
    }
}

impl fmt::Debug for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Network")
         .field("name", &self.name)
         .field("comments", &self.comments)
         .field("nports", &self.nports)
         .field("port_names", &self.port_names)
         .field("freq", &self.freq)
         .field("npts", &self.npts)
         .field("z0", &self.z0)
         .field("a", &self.a)
         .field("g", &self.g)
         .field("h", &self.h)
         .field("s", &self.s)
         .field("t", &self.t)
         .field("y", &self.y)
         .field("z", &self.z)
         .finish()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn network_reciprocity() {
        let calc = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            RFParameter::S,
            vec![mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(-0.846, 0.158), c64::new(0.544, -0.129)],
            ]],
            String::from(""),
            String::from(""),
        );

        let exemplar = vec![mat![
            [0.0, 0.0],
            [0.0, 0.0],
        ]];
        comp_point_f64(&exemplar, &calc.reciprocity(), "reciprocity(1)");
        assert_eq!(true, calc.is_reciprocal());

        let calc = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            RFParameter::S,
            vec![mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ]],
            String::from(""),
            String::from(""),
        );

        let exemplar = vec![mat![
            [0.0, 0.860811245279707],
            [0.860811245279707, 0.0],
        ]];

        comp_point_f64(&exemplar, &calc.reciprocity(), "reciprocity(2)");
        assert_eq!(false, calc.is_reciprocal());
    }

    #[test]
    fn network_new() {
        let name = String::from("title");
        let comments = String::from("here are some comments\nand some more");
        let fdata = row![1.0, 2.0, 3.0];
        let z0 = row![
            c64::new(50.0, 0.0),
            c64::new(50.0, 0.0),
        ];
        let ndata: Vec<Point> = vec![
            mat![[c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)]],
            mat![[c64::new(0.849, -0.496), c64::new(-6.688, 2.413)],
                [c64::new(0.014, 0.041), c64::new(0.494, -0.248)]],
            mat![[c64::new(0.700, -0.648), c64::new(-0.724, 0.396)],
                [c64::new(0.026, 0.053), c64::new(0.435, -0.311)]],
        ];
        let freq = Frequency::from_row(fdata, Unit::Giga);
        let nports: usize = 2;
        let npoints: usize = 3;
        let mut net = Network::new(
            freq.clone(),
            z0.clone(),
            RFParameter::S,
            ndata.copy_from(),
            String::from("title"),
            String::from("here are some comments\nand some more"),
        );
        println!("ndata: {:?}", ndata);
        println!("net: {:?}", net);

        assert_eq!(name, net.name);
        assert_eq!(comments, net.comments);
        assert_eq!(nports, net.nports);
        assert_eq!(npoints, net.npts);
        for i in 0..2 {
            assert_eq!(z0[i], net.z0[i]);
            assert_eq!(z0[i], net.z0_at_port_idx(i).clone());
            assert_eq!(String::from(""), *net.port_name(i));
        }
        net.set_port_name(0, String::from("1"));
        net.set_port_name(1, String::from("2"));
        assert_eq!(String::from("1"), *net.port_name(0));
        assert_eq!(String::from("2"), *net.port_name(1));
        for i in 0..npoints {
            for j in 0..nports {
                for k in 0..nports {
                    assert_eq!(*ndata[i].get(j, k), *net.s()[i].get(j, k));
                    assert_eq!(*ndata[i].get(j, k), *net.net(RFParameter::S)[i].get(j, k));
                    assert_eq!(*ndata[i].get(j, k), *net.net_at_freq_idx(RFParameter::S, i).get(j, k));
                    assert_eq!(*ndata[i].get(j, k), net.net_at_port_idx(RFParameter::S, j, k)[i]);
                    assert_eq!(*ndata[i].get(j, k), *net.net_at_idx(RFParameter::S, i, j, k));
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "ABCD parameters do not exist for network with 1 port(s)")]
    fn network_a() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::A,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );
    }

    #[test]
    fn network_a_to_2port() {
        let param = RFParameter::A;

        // 2 Port, 3 Frequncy
        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_a = exemplar;

        let exemplar_g = vec![
            mat![
                [c64::new(-0.00197987097401748661475535120209443, 0.0224209748787405016253479489314812), c64::new(-0.545867543186822244746417492697098, 0.109719035638690776866961585879091)],
                [c64::new(0.970683926872442228782866863616119, 0.266482121886693452315693978414573), c64::new(-0.863302777392183666428306467787712, -0.0720758146702967801388118109669652)],
            ],
            mat![
                [c64::new(-1.23239272013612611331260158924554, 1.11688220696344787208501608264046), c64::new(-3.45844082303184425580132018122926, 390.051138087020451915812870397712)],
                [c64::new(0.397612147352765257082445040698021, 0.191118516250815197053694616442202), c64::new(-0.71557575036885672322393975988072, 1.24635565988144034918971827686324)],
            ],
            mat![
                [c64::new(0.00634543595955172270622761200577528, 0.0063140057451812685749717449714631), c64::new(-0.488294088178389300834248085183965, 0.472132082557874186474657345958691)],
                [c64::new(0.0468662641434151878798143943652885, 0.000925064335148286034608156631495749), c64::new(-0.0666104330091677860871472719555602, 1.137135215326697908032725668404)],
            ],
        ];

        let exemplar_h = vec![
            mat![
                [c64::new(-1.5375603451309595812449991721764, -0.0741641259593635790266201771512177), c64::new(0.962518630609417802505617870932231, -0.228877015903281448399737721039303)],
                [c64::new(-1.74037117254308526694541719249688, -0.412698311136136010093402576493789), c64::new(-0.00211787815482265048244293454219161, 0.0399389590404924177255066568605529)],
            ],
            mat![
                [c64::new(-0.00823966748632509301363867097943702, -0.000600998985342353024729247495082033), c64::new(2.02919274404511669494686821758318, -0.954081100695924242940403364938951)],
                [c64::new(-0.000000791052406640098234562540536135491, -0.00253600468324170895382167494170621), c64::new(-0.0088568524393498832283495246123452, -0.00360090029988248130396266153405555)],
            ],
            mat![
                [c64::new(-38.3519419183273361430692085913092, 36.036385466449672335037553713537), c64::new(31.3783708663005304281170677626092, -0.331564861150846574339151434055448)],
                [c64::new(-1.54302893296500743169040964888106, -1.5195786452299769038265361150716), c64::new(0.0123536115788140826368800362648244, 0.413381672193117820850131528654662)],
            ],
        ];

        let exemplar_s = vec![
            mat![
                [c64::new(-0.16238837756090263761577828703046, -0.661900848276514131613203508024783), c64::new(0.392034731208082929247242021337783, -0.460599745677321109793006406622261)],
                [c64::new(1.00744844152363131541327128912339, -0.425176866186736276041113157264109), c64::new(-0.522498332282663260856386217653155, -0.350879534511038741490089496518265)],
            ],
            mat![
                [c64::new(-0.997682149997707784744410013158271, 0.0191995641210155333019697906624953), c64::new(7.57933779867065405526680284061164, -1.04151616039905085255893477501417)],
                [c64::new(-0.00257880927581329297048123420318214, 0.00825945150664751265547784469984842), c64::new(2.25603355998507993651922893905204, 1.0176876789673772195306889826027)],
            ],
            mat![
                [c64::new(0.424108853896427060926724974422561, -0.311869079656508506874062533579927), c64::new(0.529792120160193869536888845396917, -0.837816358602444228128879789718441)],
                [c64::new(0.0667826881062830259930012390340142, -0.0148373317054258923160675596377376), c64::new(-0.976888375160090214519654921230186, 0.00619280690374469268386995805360778)],
            ],
        ];

        let exemplar_t = vec![
            mat![
                [c64::new(0.842540000000000003227418332585326, 0.355579999999999960547114596920445), c64::new(0.315459999999999963632424382353751, 0.481419999999999964068742031031426)],
                [c64::new(0.0985399999999999599842315234354789, -0.615419999999999972062347808332553), c64::new(0.659459999999999998548938506814924, -0.747579999999999975646147731822574)],
            ],
            mat![
                [c64::new(-34.4443800000000009475797924096699, -110.319010000000013356213912629752), c64::new(-34.5626200000000009460165983909971, 283.937010000000013016929756304291)],
                [c64::new(36.4826200000000010970069297400184, 109.401990000000013374403806665217), c64::new(36.6103800000000010950174100798906, -284.983990000000013003145227230578)],
            ],
            mat![
                [c64::new(14.2695800000000006091394055829368, 3.17032000000000015793588659107624), c64::new(13.9594200000000005934541746910327, 3.00868000000000016758150422901966)],
                [c64::new(7.04058000000000001716848885280336, -3.10568000000000014271250847741615), c64::new(7.3884200000000000307931458110034, -3.91532000000000015349499449257561)],
            ],
        ];

        let exemplar_y = vec![
            mat![
                [c64::new(-0.648871307447210735771635349702552, 0.0312982664578495355773068347483066), c64::new(0.617387268456013432529115229015484, -0.178636893125236286192297241304206)],
                [c64::new(1.14219365988011020070735439940172, 0.213317492034346830947450793039609), c64::new(-1.15032402657017874867204306919901, 0.0960387751795647398839969297376987)],
            ],
            mat![
                [c64::new(-120.721856144250565560012387627147, 8.80541759382321709714120732722273), c64::new(236.566832025569524363018474382095, -133.046330877834887300382581525877)],
                [c64::new(0.0224260775706715878469859040644252, 0.306144227004672246418747731778927), c64::new(-0.346450107395470198406494118247633, -0.603430247596145946207288789238094)],
            ],
            mat![
                [c64::new(-0.0138480033987572982645818282323112, -0.0130119092660559525588196198622761), c64::new(0.438842078293103990727005963174703, 0.40370100330483060411392086802194)],
                [c64::new(0.00159525045371158277288323132060076, 0.0411208827144635403978830616765474), c64::new(-0.0513369885500034998651853664842484, -0.876395706975780302534409448767241)],
            ],
        ];

        let exemplar_z = vec![
            mat![
                [c64::new(-3.90800000000000040922820687683277, -44.256000000000000961952739686464), c64::new(-6.98897600000000068242217432512655, -23.7291320000000021877000211389923)],
                [c64::new(8.00000000000000105471187339389871, -44.0000000000000023314683517128287), c64::new(-1.32399999999999957067675637745181, -24.968000000000003152311744969437)],
            ],
            mat![
                [c64::new(-0.445515451895248867386967615842596, -0.403757887416051401295502349900258), c64::new(-159.027012324367930880053248462062, 172.377436286634265955744553143897)],
                [c64::new(-0.0999767471393880616935223522346543, -0.245685292759143631728102242576931), c64::new(-96.8911674659764272181024032371247, 39.3927116177412678457066285190078)],
            ],
            mat![
                [c64::new(79.1880552813425455069073765021249, -78.7958209937479389477316041012898), c64::new(1.46502418558736555686504913395304, -75.8627550180980586500426360456851)],
                [c64::new(3.78413951957880863430876556178437, -3.61961171437972986567029928876484), c64::new(0.0722277064823955826958577216008151, -2.4169134583744652150181723845976)],
            ],
        ];

        comp_point_c64(&exemplar_a, &calc.net(RFParameter::A), "net(A)");
        comp_point_c64(&exemplar_a, &calc.a(), "a()");

        comp_point_c64(&exemplar_g, &calc.net(RFParameter::G), "net(G)");
        comp_point_c64(&exemplar_g, &calc.g(), "g()");

        comp_point_c64(&exemplar_h, &calc.net(RFParameter::H), "net(H)");
        comp_point_c64(&exemplar_h, &calc.h(), "h()");

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_t, &calc.net(RFParameter::T), "net(T)");
        comp_point_c64(&exemplar_t, &calc.t(), "t()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    #[test]
    fn network_chop_in_half() {
        let base = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            RFParameter::S,
            vec![mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(-0.846, 0.158), c64::new(0.544, -0.129)],
            ]],
            String::from(""),
            String::from(""),
        );

        let exemplar = vec![mat![
            [c64::new(2.1770336894001643, -3.941372226787181), c64::new(1.2297289917865257, -2.9608419909240657)],
            [c64::new(1.2297289917865257, -2.9608419909240657), c64::new(1.3022596548890717, -2.173746918652424)],
        ]];

        let calc = base.chop_in_half();

        comp_point_c64(&exemplar, &calc.s(), "chop_in_half");
    }

    #[test]
    #[should_panic(expected = "Inverse Hybrid (g) parameters do not exist for network with 1 port(s)")]
    fn network_g() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::G,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );
    }

    #[test]
    fn network_g_to_2port() {
        let param = RFParameter::G;

        // 2 Port, 3 Frequncy
        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_a = vec![
            mat![
                [c64::new(8.00000000000000105471187339389714, -44.0000000000000023314683517128287), c64::new(-1.32399999999999957067675637745201, -24.9680000000000031523117449694402)],
                [c64::new(-3.90800000000000040922820687683316, -44.2560000000000009619527396864703), c64::new(-6.98897600000000068242217432512576, -23.7291320000000021877000211389955)],
            ],
            mat![
                [c64::new(-0.0999767471393880616935223522346666, -0.245685292759143631728102242576931), c64::new(-96.8911674659764272181024032371247, 39.3927116177412678457066285190015)],
                [c64::new(-0.445515451895248867386967615842646, -0.403757887416051401295502349900258), c64::new(-159.027012324367930880053248462087, 172.377436286634265955744553143923)],
            ],
            mat![
                [c64::new(3.78413951957880863430876556178437, -3.61961171437972986567029928876524), c64::new(0.0722277064823955826958577216008151, -2.4169134583744652150181723845976)],
                [c64::new(79.1880552813425455069073765021249, -78.7958209937479389477316041013025), c64::new(1.46502418558736555686504913395994, -75.8627550180980586500426360456851)],
            ],
        ];

        let exemplar_g = exemplar;

        let exemplar_h = vec![
            mat![
                [c64::new(0.983339062615605700552382119247199, 0.233827900172790393489913919096089), c64::new(1.49460150665477665565964447218079, 0.432452988991509451551132581692921)],
                [c64::new(0.0114214352470227992178057046009944, -0.0387783195429568806892880025629338), c64::new(1.76080827863846485310519872588951, 0.353921365518365507567170503569742)],
            ],
            mat![
                [c64::new(0.403587022436639082458721162881338, 0.189757602732871813023124553271079), c64::new(0.00321137873997047758921840539884824, 0.00180609494050279904534746654027308)],
                [c64::new(0.00289121249557212404542717523227075, 0.00313393171677059316769924431237505), c64::new(0.000022730192220551215708716431202894, 0.00256356485428899456946143285544854)],
            ],
            mat![
                [c64::new(0.0318655280808023732715779332709627, 0.000336712490225446144228707456570845), c64::new(1.23423878324089371439391106555654, -1.1354048751434688927845269807668)],
                [c64::new(-0.000254463584406321724690661721392893, -0.0131767848986768179028856968025573), c64::new(1.05842606053831599632442636862377, 1.02339330394047149228756047634429)],
            ],
        ];

        let exemplar_s = vec![
            mat![
                [c64::new(-0.961856034390476247953062073108471, 0.0102422291302198607302732199365965), c64::new(0.0335164541445717706812989018050447, 0.00269532657640297977992345171255623)],
                [c64::new(-0.0000741917945909158341997275566216092, 0.000870473988109723583974499603256693), c64::new(-0.978389238672361856979721874012602, -0.0043092166610166418335235350145952)],
            ],
            mat![
                [c64::new(-0.983968673166477883192532506579779, 0.00744054621217117751824841863921544), c64::new(0.00672246279940744016786139716972062, 0.0026946054574608550284795537021618)],
                [c64::new(-0.00643969737439854503546926246153125, -0.00536458415521835157059384392271001), c64::new(0.965524603410289908525909983238879, -0.251622659689705895382556623618327)],
            ],
            mat![
                [c64::new(-0.998133165732073810488921516995188, 0.0000422502556085994715361188907936374), c64::new(0.00305580893176563819529235773865954, -0.0449749332503353848792385851495327)],
                [c64::new(0.00024877486685203947995319096401283, 0.000252205209498941932519768008979821), c64::new(-0.980490006248725689689331643472689, -0.0185149210694082559162602806369637)],
            ],
        ];

        let exemplar_t = vec![
            mat![
                [c64::new(-97.2077280000000100402670899502288, -1140.51424600000002633992579603721), c64::new(-90.1922720000000093664313804975984, -1116.28575400000002408917953999884)],
                [c64::new(105.181248000000011103565428216557, 1096.01488600000002394541120942511), c64::new(98.2187520000000104125567890190041, 1072.78511400000002182075742318549)],
            ],
            mat![
                [c64::new(-91.670292507794645427728599835599, 76.3658554277136888080777329883474), c64::new(69.2945431658928139966866966912359, -96.7994350912754025045809527259475)],
                [c64::new(89.6324924113357288216730294186148, -75.8236864881180070828917026605444), c64::new(-67.4566965637126735140181709787246, 95.7658955661614335159387179129804)],
            ],
            mat![
                [c64::new(1982.32668616321154872409827847802, -2009.66087734452111260329675192324), c64::new(1980.86021742349453525557931219005, -1933.74978405725556464895375242973)],
                [c64::new(-1978.5411020895030921781355957619, 2005.99292736097389343332608918668), c64::new(-1977.07752245804537453292446378227, 1930.17851061204332408758381658882)],
            ],
        ];

        let exemplar_y = vec![
            mat![
                [c64::new(0.962518630609417802505617870932231, -0.228877015903281448399737721039279), c64::new(-1.5375603451309595812449991721764, -0.07416412595936357902662017715123)],
                [c64::new(0.00211787815482265048244293454219161, -0.0399389590404924177255066568605529), c64::new(1.74037117254308526694541719249708, 0.412698311136136010093402576493739)],
            ],
            mat![
                [c64::new(2.02919274404511669494686821758318, -0.954081100695924242940403364938853), c64::new(-0.00823966748632509301363867097943702, -0.000600998985342353024729247495082322)],
                [c64::new(0.0088568524393498832283495246123452, 0.00360090029988248130396266153405555), c64::new(0.000000791052406640098234562540536246928, 0.00253600468324170895382167494170621)],
            ],
            mat![
                [c64::new(31.3783708663005304281170677626092, -0.331564861150846574339151434052589), c64::new(-38.3519419183273361430692085913029, 36.0363854664496723350375537135307)],
                [c64::new(-0.0123536115788140826368800362648244, -0.413381672193117820850131528654662), c64::new(1.54302893296500743169040964888126, 1.5195786452299769038265361150716)],
            ],
        ];

        let exemplar_z = vec![
            mat![
                [c64::new(0.970683926872442228782866863616166, 0.266482121886693452315693978414556), c64::new(0.863302777392183666428306467787739, 0.0720758146702967801388118109669587)],
                [c64::new(-0.00197987097401748661475535120209434, 0.0224209748787405016253479489314802), c64::new(0.545867543186822244746417492697067, -0.109719035638690776866961585879081)],
            ],
            mat![
                [c64::new(0.397612147352765257082445040698052, 0.191118516250815197053694616442173), c64::new(0.715575750368856723223939759880684, -1.24635565988144034918971827686337)],
                [c64::new(-1.23239272013612611331260158924553, 1.11688220696344787208501608264051), c64::new(3.4584408230318442558013201812297, -390.051138087020451915812870397718)],
            ],
            mat![
                [c64::new(0.0468662641434151878798143943652929, 0.000925064335148286034608156631487209), c64::new(0.0666104330091677860871472719554069, -1.13713521532669790803272566840385)],
                [c64::new(0.00634543595955172270622761200577549, 0.00631400574518126857497174497146204), c64::new(0.488294088178389300834248085183919, -0.472132082557874186474657345958685)],
            ],
        ];

        comp_point_c64(&exemplar_a, &calc.net(RFParameter::A), "net(A)");
        comp_point_c64(&exemplar_a, &calc.a(), "a()");

        comp_point_c64(&exemplar_g, &calc.net(RFParameter::G), "net(G)");
        comp_point_c64(&exemplar_g, &calc.g(), "g()");

        comp_point_c64(&exemplar_h, &calc.net(RFParameter::H), "net(H)");
        comp_point_c64(&exemplar_h, &calc.h(), "h()");

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_t, &calc.net(RFParameter::T), "net(T)");
        comp_point_c64(&exemplar_t, &calc.t(), "t()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    #[test]
    #[should_panic(expected = "Hybrid (h) parameters do not exist for network with 1 port(s)")]
    fn network_h() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::H,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );
    }

    #[test]
    fn network_h_to_2port() {
        let param = RFParameter::H;

        // 2 Port, 3 Frequncy
        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_a = vec![
            mat![
                [c64::new(6.98897600000000068242217432512655, 23.7291320000000021877000211389923), c64::new(3.90800000000000040922820687683277, 44.256000000000000961952739686464)],
                [c64::new(1.32399999999999957067675637745181, 24.968000000000003152311744969437), c64::new(-8.00000000000000105471187339389871, 44.0000000000000023314683517128287)],
            ],
            mat![
                [c64::new(159.027012324367930880053248462062, -172.377436286634265955744553143897), c64::new(0.445515451895248867386967615842596, 0.403757887416051401295502349900258)],
                [c64::new(96.8911674659764272181024032371247, -39.3927116177412678457066285190078), c64::new(0.0999767471393880616935223522346543, 0.245685292759143631728102242576931)],
            ],
            mat![
                [c64::new(-1.46502418558736555686504913395304, 75.8627550180980586500426360456851), c64::new(-79.1880552813425455069073765021249, 78.7958209937479389477316041012898)],
                [c64::new(-0.0722277064823955826958577216008151, 2.4169134583744652150181723845976), c64::new(-3.78413951957880863430876556178437, 3.61961171437972986567029928876484)],
            ],
        ];

        let exemplar_g = vec![
            mat![
                [c64::new(0.983339062615605700552382119247199, 0.233827900172790393489913919096089), c64::new(1.49460150665477665565964447218079, 0.432452988991509451551132581692921)],
                [c64::new(0.0114214352470227992178057046009944, -0.0387783195429568806892880025629338), c64::new(1.76080827863846485310519872588951, 0.353921365518365507567170503569742)],
            ],
            mat![
                [c64::new(0.403587022436639082458721162881338, 0.189757602732871813023124553271079), c64::new(0.00321137873997047758921840539884824, 0.00180609494050279904534746654027308)],
                [c64::new(0.00289121249557212404542717523227075, 0.00313393171677059316769924431237505), c64::new(0.000022730192220551215708716431202894, 0.00256356485428899456946143285544854)],
            ],
            mat![
                [c64::new(0.0318655280808023732715779332709627, 0.000336712490225446144228707456570845), c64::new(1.23423878324089371439391106555654, -1.1354048751434688927845269807668)],
                [c64::new(-0.000254463584406321724690661721392893, -0.0131767848986768179028856968025573), c64::new(1.05842606053831599632442636862377, 1.02339330394047149228756047634429)],
            ],
        ];

        let exemplar_h = exemplar;

        let exemplar_s = vec![
            mat![
                [c64::new(-0.962182159273888011464839786393512, -0.00885792224980607944361472977648167), c64::new(-0.0583179802445271778253411399506154, -0.00261063807217240909047843892818312)],
                [c64::new(0.0000750720597874687206532835178953338, -0.0015148661137374319802809753701194), c64::new(-0.932599761547068474800828761599924, 0.0153701453866574479432593558781635)],
            ],
            mat![
                [c64::new(-0.921351788638241583217052746017558, -0.0352326057392396907097832720726561), c64::new(-0.000316203962077200445190996978207857, -0.0000288846702769489433267225441504298)],
                [c64::new(0.000337812532076932226950695872238608, 0.000144595788268543451242916602457944), c64::new(-0.999999909280214853033385187032764, 0.000101468070047106538730190102390998)],
            ],
            mat![
                [c64::new(-0.233404853905904413740936097534058, -0.00945492554204868104693772619396554), c64::new(-0.898361021083420609986110406035247, 0.88178691009347542497172249982464)],
                [c64::new(-0.000511045444755091736939568325182609, -0.00987926399114123710809431684940762), c64::new(-0.945928764825887078201525279345446, 0.0503757563685110267370147014051763)],
            ],
        ];

        let exemplar_t = vec![
            mat![
                [c64::new(32.6335679999999890848663419706872, 658.507126000000081076997338058777), c64::new(40.5554079999999901313936512270484, 613.622006000000078726289931552167)],
                [c64::new(-25.5664319999999883942596035080074, -633.892874000000078870058262126003), c64::new(-33.6445919999999894571560410394436, -590.777994000000076557828965206855)],
            ],
            mat![
                [c64::new(2501.84713633968329241210733601143, -1070.87962836159509679066098340228), c64::new(2501.73824928350599937306607430695, -1071.13338881210256145041499569189)],
                [c64::new(-2342.81121370627745655470634819716, 898.510267232709151862942340305489), c64::new(-2342.72014726817597347036056519709, 898.74787736771997446664453250099)],
            ],
            mat![
                [c64::new(-5.22215506745640211805242415292009, 100.951978035538004022788093323182), c64::new(0.145745557749257426394488938906847, 95.7564499012833153781631619523881)],
                [c64::new(2.17336977624218565104922748891092, -23.5133065975649865937908251954686), c64::new(-0.0270086377097720731213905428307556, -21.4696113030602155070751579887285)],
            ],
        ];

        let exemplar_y = vec![
            mat![
                [c64::new(0.970683926872442228782866863616166, 0.266482121886693452315693978414556), c64::new(0.863302777392183666428306467787739, 0.0720758146702967801388118109669587)],
                [c64::new(-0.00197987097401748661475535120209434, 0.0224209748787405016253479489314833), c64::new(0.545867543186822244746417492697067, -0.109719035638690776866961585879093)],
            ],
            mat![
                [c64::new(0.397612147352765257082445040698002, 0.191118516250815197053694616442173), c64::new(0.715575750368856723223939759880684, -1.24635565988144034918971827686337)],
                [c64::new(-1.23239272013612611331260158924553, 1.11688220696344787208501608264051), c64::new(3.45844082303184425580132018121085, -390.051138087020451915812870397667)],
            ],
            mat![
                [c64::new(0.0468662641434151878798143943652868, 0.00092506433514828603460815663148952), c64::new(0.0666104330091677860871472719554069, -1.13713521532669790803272566840385)],
                [c64::new(0.00634543595955172270622761200577549, 0.00631400574518126857497174497146204), c64::new(0.48829408817838930083424808518387, -0.472132082557874186474657345958685)],
            ],
        ];

        let exemplar_z = vec![
            mat![
                [c64::new(0.962518630609417802505617870932231, -0.228877015903281448399737721039303), c64::new(-1.5375603451309595812449991721764, -0.0741641259593635790266201771512177)],
                [c64::new(0.00211787815482265048244293454219161, -0.0399389590404924177255066568605529), c64::new(1.74037117254308526694541719249688, 0.412698311136136010093402576493789)],
            ],
            mat![
                [c64::new(2.02919274404511669494686821758318, -0.954081100695924242940403364938951), c64::new(-0.00823966748632509301363867097943702, -0.000600998985342353024729247495082033)],
                [c64::new(0.0088568524393498832283495246123452, 0.00360090029988248130396266153405555), c64::new(0.000000791052406640098234562540536135491, 0.00253600468324170895382167494170621)],
            ],
            mat![
                [c64::new(31.3783708663005304281170677626092, -0.331564861150846574339151434055448), c64::new(-38.3519419183273361430692085913092, 36.036385466449672335037553713537)],
                [c64::new(-0.0123536115788140826368800362648244, -0.413381672193117820850131528654662), c64::new(1.54302893296500743169040964888106, 1.5195786452299769038265361150716)],
            ],
        ];

        comp_point_c64(&exemplar_a, &calc.net(RFParameter::A), "net(A)");
        comp_point_c64(&exemplar_a, &calc.a(), "a()");

        comp_point_c64(&exemplar_g, &calc.net(RFParameter::G), "net(G)");
        comp_point_c64(&exemplar_g, &calc.g(), "g()");

        comp_point_c64(&exemplar_h, &calc.net(RFParameter::H), "net(H)");
        comp_point_c64(&exemplar_h, &calc.h(), "h()");

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_t, &calc.net(RFParameter::T), "net(T)");
        comp_point_c64(&exemplar_t, &calc.t(), "t()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    // #[test]
    // fn network_read_touchstone() {
    //     let filename = "./data/test.s1p".to_string();
    //     let net = Network::<na::Const<1>>::read_touchstone(&filename).unwrap();
    //     let exemplar: Network<na::Const<1>> = Network::<na::Const<1>>::new(
    //         Frequency::from_vec(vec![75.0, 75.175, 75.35, 75.525], FreqUnit::GHz),
    //         vec![c64::new(50.0, 0.0)],
    //         RFParameter::S,
    //         vec![
    //             Point::<na::Const<1>>::from_vec(vec![c64::new(
    //                 0.45345337996,
    //                 0.891279996524,
    //             )]),
    //             Point::<na::Const<1>>::from_vec(vec![c64::new(
    //                 0.464543921496,
    //                 0.885550080459,
    //             )]),
    //             Point::<na::Const<1>>::from_vec(vec![c64::new(
    //                 0.475521092896,
    //                 0.879704319764,
    //             )]),
    //             Point::<na::Const<1>>::from_vec(vec![c64::new(
    //                 0.486384743723,
    //                 0.873744745949,
    //             )]),
    //         ],
    //         String::from("test"),
    //         String::from("Created with mwavepy."),
    //     );
    //     assert_eq!(exemplar.name, net.name);
    //     assert_eq!(exemplar.comments, net.comments);
    //     assert_eq!(exemplar.nports, net.nports);
    //     assert_eq!(exemplar.npoints, net.npoints);
    //     for i in 0..exemplar.nports {
    //         assert_eq!(exemplar.z0[i], net.z0[i]);
    //         assert_eq!(exemplar.z0[i], net.z0_at_pidx(i).clone());
    //     }
    //     for i in 0..exemplar.npoints {
    //         assert_eq!(exemplar.freq.freq_at(i), net.freq.freq_at(i));
    //         for j in 0..exemplar.nports {
    //             for k in 0..exemplar.nports {
    //                 assert_eq!(exemplar.net[i][(j, k)], net.net(RFParameter::S)[i][(j, k)]);
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_fidx(RFParameter::S, i)[(j, k)]
    //                 );
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_pidx(RFParameter::S, j, k)[i]
    //                 );
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_idx(RFParameter::S, i, j, k)
    //                 );
    //             }
    //         }
    //     }

    //     let filename = "./data/test.s2p".to_string();
    //     let net = Network::<na::Const<2>>::read_touchstone(&filename).unwrap();
    //     let exemplar: Network<na::Const<2>> = Network::<na::Const<2>>::new(
    //         Frequency::from_vec(vec![0.5, 1.0, 1.5, 2.0], FreqUnit::GHz),
    //         vec![
    //             c64::new(50.0, 0.0),
    //             c64::new(50.0, 0.0),
    //         ],
    //         RFParameter::S,
    //         vec![
    //             Point::<na::Const<2>>::from_vec(vec![
    //                 c64::new(0.9881388526914863, -0.13442709904013195),
    //                 c64::new(-7.363542304219899, 0.6742816969789206),
    //                 c64::new(0.0010346705444205045, 0.011178864909012504),
    //                 c64::new(0.5574653418486702, -0.06665134724424635),
    //             ]),
    //             Point::<na::Const<2>>::from_vec(vec![
    //                 c64::new(0.9578079036840927, -0.2633207328693372),
    //                 c64::new(-7.130124628011368, 1.3277987152036197),
    //                 c64::new(0.0037206104781559784, 0.021909191616475577),
    //                 c64::new(0.5435045929943587, -0.12869941397967788),
    //             ]),
    //             Point::<na::Const<2>>::from_vec(vec![
    //                 c64::new(0.9133108288727866, -0.38508398385543624),
    //                 c64::new(-6.9151682810378095, 1.800750901131042),
    //                 c64::new(0.008042664765986755, 0.03190603796445517),
    //                 c64::new(0.5235871604669029, -0.18886435408156288),
    //             ]),
    //             Point::<na::Const<2>>::from_vec(vec![
    //                 c64::new(0.849070850314753, -0.49577931076259807),
    //                 c64::new(-6.688405272002992, 2.4133819411904995),
    //                 c64::new(0.01381064392153511, 0.04080882571424955),
    //                 c64::new(0.4942211124266797, -0.24774648346309974),
    //             ]),
    //         ],
    //         String::from("test"),
    //         String::from("File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\nBase file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\nDate = 10/8/2019\nTime = 7:43 AM\nTest duration(min) = 0.15\nTest = S-Parameters\nMPTS version = 8.91.5\nState File = C:\\Data\\PNA\\20191002105035\\S-Parameters_2019-10-04.sta\nCal File = \nComputerName = CMPE-LAB-6\nComments = \nZ0 = 50\nInput Power(dBm) = -13.00\nDevice ID = EG0520F_1010_6x25_2SDBCB2EV\nRow = 66\nColumn = 42\nTechnology = QPHT09BCB\nLot/Wafer = 191593406\nGate Size = 150\nMask = EG0520\nCal Standards = EG2306 GaAs\nCal Kit = EG2308\nJob Number = 20191002105035\nDevice Type = FET\nset_Vg(v) = -1.500\nset_Vd(v) = 2.500\nset_Vd(mA) = 11.250\nfile_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\nVg(V) = -0.520\nVg(mA) = -0.001\nVd(V) = 2.500\nVd(mA) = 11.076"),
    //     );
    //     assert_eq!(exemplar.name, net.name);
    //     assert_eq!(exemplar.comments, net.comments);
    //     assert_eq!(exemplar.nports, net.nports);
    //     assert_eq!(exemplar.npoints, net.npoints);
    //     for i in 0..exemplar.nports {
    //         assert_eq!(exemplar.z0[i], net.z0[i]);
    //         assert_eq!(exemplar.z0[i], net.z0_at_pidx(i).clone());
    //     }
    //     for i in 0..exemplar.npoints {
    //         assert_eq!(exemplar.freq.freq_at(i), net.freq.freq_at(i));
    //         for j in 0..exemplar.nports {
    //             for k in 0..exemplar.nports {
    //                 assert_eq!(exemplar.net[i][(j, k)], net.net(RFParameter::S)[i][(j, k)]);
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_fidx(RFParameter::S, i)[(j, k)]
    //                 );
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_pidx(RFParameter::S, j, k)[i]
    //                 );
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_idx(RFParameter::S, i, j, k)
    //                 );
    //             }
    //         }
    //     }

    //     let filename = "./data/test.s3p".to_string();
    //     let net = Network::<na::Const<3>>::read_touchstone(&filename).unwrap();
    //     let exemplar: Network<na::Const<3>> = Network::<na::Const<3>>::new(
    //         Frequency::from_vec(vec![330.0, 330.85, 331.7, 332.55], FreqUnit::GHz),
    //         vec![
    //             c64::new(50.0, 0.0),
    //             c64::new(50.0, 0.0),
    //             c64::new(50.0, 0.0),
    //         ],
    //         RFParameter::S,
    //         vec![
    //             Point::<na::Const<3>>::from_vec(vec![
    //                 c64::new(-0.333333333333, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(-0.333333333333, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(-0.333333333333, 0.0),
    //             ]),
    //             Point::<na::Const<3>>::from_vec(vec![
    //                 c64::new(-0.333333333333, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(-0.333333333333, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(-0.333333333333, 0.0),
    //             ]),
    //             Point::<na::Const<3>>::from_vec(vec![
    //                 c64::new(-0.333333333333, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(-0.333333333333, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(-0.333333333333, 0.0),
    //             ]),
    //             Point::<na::Const<3>>::from_vec(vec![
    //                 c64::new(-0.333333333333, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(-0.333333333333, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(0.666666666667, 0.0),
    //                 c64::new(-0.333333333333, 0.0),
    //             ]),
    //         ],
    //         String::from("test"),
    //         String::from("Created with skrf (http://scikit-rf.org)."),
    //     );
    //     assert_eq!(exemplar.name, net.name);
    //     assert_eq!(exemplar.comments, net.comments);
    //     assert_eq!(exemplar.nports, net.nports);
    //     assert_eq!(exemplar.npoints, net.npoints);
    //     for i in 0..exemplar.nports {
    //         assert_eq!(exemplar.z0[i], net.z0[i]);
    //         assert_eq!(exemplar.z0[i], net.z0_at_pidx(i).clone());
    //     }
    //     for i in 0..exemplar.npoints {
    //         assert_eq!(exemplar.freq.freq_at(i), net.freq.freq_at(i));
    //         for j in 0..exemplar.nports {
    //             for k in 0..exemplar.nports {
    //                 assert_eq!(exemplar.net[i][(j, k)], net.net(RFParameter::S)[i][(j, k)]);
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_fidx(RFParameter::S, i)[(j, k)]
    //                 );
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_pidx(RFParameter::S, j, k)[i]
    //                 );
    //                 assert_eq!(
    //                     exemplar.net[i][(j, k)],
    //                     net.net_at_idx(RFParameter::S, i, j, k)
    //                 );
    //             }
    //         }
    //     }
    // }

    #[test]
    fn network_s_db() {
        let param = RFParameter::S;

        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_db = vec![
            mat![
                [-0.0571232931409698157602471654702417, -1.3036938210270008034803505896139],
                [-33.0102999566398124343011222238879, -5.05042981341051474077798143321121],
            ],
            mat![
                [7.10808722678577439452626820799157, 10.2582363703478204445299539985994],
                [11.5269507556086664924691854963622, 51.9169985529863755091065898539251],
            ],
            mat![
                [26.5811015830929560104699014097416, 27.7122202597899446941970466073987],
                [-14.3808805387243224903439848474195, -6.71178171541068598914939921950404],
            ],
        ];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_f64(&exemplar_s_db, &calc.net_db(RFParameter::S), "net_db(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");
        comp_point_f64(&exemplar_s_db, &calc.s_db(), "s_db()");
    }

    #[test]
    fn network_s_deg() {
        let param = RFParameter::S;

        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_deg = vec![
            mat![
                [-15.351227009735157493813896682283, 169.42124106201960483185916007312],
                [79.6951535312339672595779107954925, -13.3402769069429650908860377064099],
            ],
            mat![
                [-25.6719967136523418134102583838897, 94.1896222142384810674002307865579],
                [112.142888514762848850576625256636, -89.9821278079241522410312200526232],
            ],
            mat![
                [-1.1307792814754272842611967996569, 92.2216280653444960010340518233173],
                [43.7269699799432878929141211369787, -44.5612966323272438891616918249177],
            ],
        ];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_f64(&exemplar_s_deg, &calc.net_deg(RFParameter::S), "net_deg(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");
        comp_point_f64(&exemplar_s_deg, &calc.s_deg(), "s_deg()");
    }

    #[test]
    fn network_s_im() {
        let param = RFParameter::S;

        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_im = vec![
            mat![
                [-0.263, 0.158],
                [0.022, -0.129],
            ],
            mat![
                [-0.982, 3.249],
                [3.492, -394.321],
            ],
            mat![
                [-0.421, 24.282],
                [0.132, -0.324],
            ],
        ];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_f64(&exemplar_s_im, &calc.net_im(RFParameter::S), "net_im(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");
        comp_point_f64(&exemplar_s_im, &calc.s_im(), "s_im()");
    }

    #[test]
    fn network_s_mag() {
        let param = RFParameter::S;

        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_mag = vec![
            mat![
                [0.993445016092989383214754191494358, 0.8606276779188547320363076992136],
                [0.0223606797749978957228246600636754, 0.559085861026729993118859201916716],
            ],
            mat![
                [2.26675384636267918218365739653485, 3.25770548085611949025013186816706],
                [3.7700537131452119493274643947394, 394.321019183608852673148079774108],
            ],
            mat![
                [21.3331545252923161462373382375707, 24.3002651837382219396266802638806],
                [0.190965965554074595232556839140833, 0.461754263651132967274505211463413],
            ],
        ];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_f64(&exemplar_s_mag, &calc.net_mag(RFParameter::S), "net_mag(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");
        comp_point_f64(&exemplar_s_mag, &calc.s_mag(), "s_mag()");
    }

    #[test]
    fn network_s_rad() {
        let param = RFParameter::S;

        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_rad = vec![
            mat![
                [-0.267929455540962111948947391268623, 2.95695847934725672395589308518909],
                [1.39094282700241833476498088656927, -0.232831755153919938547207556795139],
            ],
            mat![
                [-0.448060868214397288426676208406736, 1.64391902884805336903629339885268],
                [1.9572626372795453631888787638813, -1.57048439819862423569595683285167],
            ],
            mat![
                [-0.0197358215750819296148809534818378, 1.60957105128986980679885216395864],
                [0.763179598070729232409291077103769, -0.7777413451919714619155793725139],
            ],
        ];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_f64(&exemplar_s_rad, &calc.net_rad(RFParameter::S), "net_rad(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");
        comp_point_f64(&exemplar_s_rad, &calc.s_rad(), "s_rad()");
    }

    #[test]
    fn network_s_re() {
        let param = RFParameter::S;

        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;
        let exemplar_s_re = vec![
            mat![
                [0.958, -0.846],
                [0.004, 0.544],
            ],
            mat![
                [2.043, -0.238],
                [-1.421, 0.123],
            ],
            mat![
                [21.329, -0.942],
                [0.138, 0.329],
            ],
        ];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_f64(&exemplar_s_re, &calc.net_re(RFParameter::S), "net_re(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");
        comp_point_f64(&exemplar_s_re, &calc.s_re(), "s_re()");
    }

    #[test]
    fn network_s_to_1port() {
        let param = RFParameter::S;

        // 1 Port
        let exemplar = vec![mat![
            [c64::new(0.958, -0.263)],
        ]];

        let calc = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;

        let exemplar_y = vec![mat![
            [c64::new(0.0000669598991322683011410796846467133, 0.00269540881178334366468222619961006)],
        ]];

        let exemplar_z = vec![mat![
            [c64::new(9.2108045620515583043706421493683, -370.772419043322540157375538364691)],
        ]];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    #[test]
    fn network_s_to_2port() {
        let param = RFParameter::S;

        // 2 Port, 3 Frequncy
        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_a = vec![
            mat![
                [c64::new(6.20248800000000044929129860982146, -19.7794339999999989767046626454285), c64::new(-105.524399999999990190381604637789, -3423.82830000000021583582143769322)],
                [c64::new(0.0624302400000000035219466232305726, 0.0149486799999999959509611180408229), c64::new(8.78648800000000128784274910920301, -0.491434000000001167063667928402902)],
            ],
            mat![
                [c64::new(127.686343795654860584537580865561, -106.209795542275364417237393127699), c64::new(-6411.59179973447487568090354168199, 5278.01761810500846921068942676095)],
                [c64::new(-0.617903061536356428562573999613567, 1.33142797263549905879605324732232), c64::new(31.2406917815736822338221452442772, -66.4133260371180451702352622587931)],
            ],
            mat![
                [c64::new(40.7174714544257965008276176041704, -2.06788211582757247417054747988658), c64::new(2112.7361673247778820194262229869, -4017.3775296150048169615677955086)],
                [c64::new(-0.740111192826587668984294195279821, 0.0173036771964461564703684115143449), c64::new(-38.398356120434353423383901176354, 74.3110254195459012585428842368045)],
            ],
        ];

        let exemplar_g = vec![
            mat![
                [c64::new(0.000213048760821325277525497566121415, 0.00308951245104338508421225735365506), c64::new(1.7689903204282919030827339659499, -0.564027324268866162290548665652407)],
                [c64::new(0.0144345665047667785208622615068543, 0.0460311338771868878575501004337997), c64::new(156.07949988012558966902470611023, -54.2788850809624038925442721797019)],
            ],
            mat![
                [c64::new(-0.00798672764284041603608352988121335, 0.00378395408839344821871371849258865), c64::new(-0.0048306646665151590758959144332717, -0.00193217651512916753176252940806199)],
                [c64::new(0.00462894606886423433510796307800752, 0.00385036802633407390117571538650149), c64::new(-50.0012229357225099523176408672366, -0.255329158880014224974981224807261)],
            ],
            mat![
                [c64::new(-0.0181515130132193684814310231688105, -0.000496875449681258561838320882623967), c64::new(-1.94713818082740457786877558053612, -2.43931164492142017724970004048629)],
                [c64::new(0.0244963002461710455919712909900429, 0.00124407187808057411306782745514073), c64::new(56.752125903959264092550825328339, -95.7824905160975790023067141690706)],
            ],
        ];

        let exemplar_h = vec![
            mat![
                [c64::new(9.75411839725456366108164033524665, -389.124163668073444598477977280313), c64::new(-0.22335655547067189974029837523878, 4.36786973299443656646522211355716)],
                [c64::new(-0.113456197465358364315070598874157, -0.00634567906371590636067352478451161), c64::new(0.00698823811154352610038219280730167, 0.00209218265683721216234437228800613)],
            ],
            mat![
                [c64::new(-102.257815389840424646191773493135, -48.4388766357741503205898133242), c64::new(0.00805128434309711647136451218299201, 0.00859011772827454413118619059138823)],
                [c64::new(-0.00579958596837412911190219608858365, -0.0123291057858426182995212221077399), c64::new(-0.0199988982463547075049939313130291, 0.000103538776919894826841166036271029)],
            ],
            mat![
                [c64::new(-54.2639407481474601049289308534532, -0.391463390728709355490817959573198), c64::new(0.54934778369570785806135095984477, -1.41864283877471352258408773652923)],
                [c64::new(0.00548817472628724373582700827815879, 0.0106210768584180586829037246468311), c64::new(0.00424564322854989379452636421914336, 0.0077658122589252871027906701737798)],
            ],
        ];

        let exemplar_s = exemplar;

        let exemplar_t = vec![
            mat![
                [c64::new(6.98897600000000068242217432512576, 23.7291320000000021877000211389955), c64::new(-3.90800000000000040922820687683277, -44.256000000000000961952739686464)],
                [c64::new(1.32399999999999957067675637745181, 24.968000000000003152311744969437), c64::new(8.00000000000000105471187339389871, -44.0000000000000023314683517128287)],
            ],
            mat![
                [c64::new(159.027012324367930880053248462062, -172.377436286634265955744553143897), c64::new(-0.445515451895248867386967615842596, -0.403757887416051401295502349900258)],
                [c64::new(96.8911674659764272181024032371247, -39.3927116177412678457066285190078), c64::new(-0.0999767471393880616935223522346543, -0.245685292759143631728102242576931)],
            ],
            mat![
                [c64::new(-1.46502418558736555686504913395856, 75.8627550180980586500426360456851), c64::new(79.1880552813425455069073765021249, -78.7958209937479389477316041012898)],
                [c64::new(-0.0722277064823955826958577216008151, 2.4169134583744652150181723845976), c64::new(3.78413951957880863430876556178437, -3.61961171437972986567029928876484)],
            ],
        ];

        let exemplar_y = vec![
            mat![
                [c64::new(0.0000643781985972787214603750759937056, 0.00256826005871265781978715968723912), c64::new(0.0112322046695956277075332640097281, 0.000292442135149206206580162558876109)],
                [c64::new(0.0000089932484722337856893059177187244, -0.00029179354375069466836702311754854), c64::new(0.00571573062451292895555336877198285, 0.00198772731819330779071976235604774)],
            ],
            mat![
                [c64::new(-0.00798702886655188471806791194449876, 0.00378340476449990313837399278570267), c64::new(0.0000968057328019028924651517594093019, 0.0000381482507187907797659349596903918)],
                [c64::new(0.0000929674581156337196757704067737861, 0.00007653074262542338691876319329766), c64::new(-0.0199989893457056667779504438109378, 0.000102124004739917434057287917373531)],
            ],
            mat![
                [c64::new(-0.0184274851078359227341227256163058, 0.000132937005743773184777354082601851), c64::new(0.00993450797186876948789720686472834, -0.0262150484343355967507773472406472)],
                [c64::new(-0.000102545192193192253288542631477874, -0.000194990154122568245419145753696648), c64::new(0.00457859758838743348727802137815361, 0.00772745466537935561159261924014919)],
            ],
        ];

        let exemplar_z = vec![
            mat![
                [c64::new(22.2146157816671576180022410951097, -322.143774918102028086464461317649), c64::new(142.400451107126191171059606354767, 582.398869915347504545197497831667)],
                [c64::new(15.1493015798358301463710993904171, -3.62744179007577422858997869032419), c64::new(131.326508310944480795337693690571, -39.3173556317843730541409150221658)],
            ],
            mat![
                [c64::new(-102.254815995145484270426998669141, -48.4463157310248314009669009903435), c64::new(-0.400351893108739487767828936477392, -0.431602259649373633168360051650213)],
                [c64::new(-0.286795883438740842682522933368365, -0.617974056800465117534742611686059), c64::new(-50.0014143155036305208594139119525, -0.258868524591639320988436736567791)],
            ],
            mat![
                [c64::new(-55.0505754936271351798507586602985, 1.50694211736989815617749407510625), c64::new(-110.866978875285773870796202221365, -131.351285328202608978507976961612)],
                [c64::new(-1.35041017032651120735649719749883, -0.0315723662830793855065315856673629), c64::new(54.1997055422709154581595792225376, -99.1380375298442079306974437725614)],
            ],
        ];

        comp_point_c64(&exemplar_a, &calc.net(RFParameter::A), "net(A)");
        comp_point_c64(&exemplar_a, &calc.a(), "a()");

        comp_point_c64(&exemplar_g, &calc.net(RFParameter::G), "net(G)");
        comp_point_c64(&exemplar_g, &calc.g(), "g()");

        comp_point_c64(&exemplar_h, &calc.net(RFParameter::H), "net(H)");
        comp_point_c64(&exemplar_h, &calc.h(), "h()");

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_t, &calc.net(RFParameter::T), "net(T)");
        comp_point_c64(&exemplar_t, &calc.t(), "t()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    #[test]
    fn network_s_to_3port() {
        let param = RFParameter::S;

        let exemplar = vec![mat![
            [c64::new(0.958, -0.263), c64::new(-0.846, 0.158), c64::new(1.473, 0.230)],
            [c64::new(0.004, 0.022), c64::new(0.544, -0.129), c64::new(-30.321, 4.378)],
            [c64::new(4.234, 84.212), c64::new(0.457, 0.287), c64::new(3.489, -2.893)],
        ]];

        let calc = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = exemplar;

        let exemplar_y = vec![mat![
            [c64::new(-0.0197936353891280493729407070046144, -0.000367976873884332391896061582337862), c64::new(-0.0000139974302482355371208278962520843, -0.0000909979061403249216705913604336164), c64::new(0.000032822220696306842874533583526163, -0.000469669414847076915124881450315591)],
            [c64::new(-0.0486928218191928932898735952034563, -0.0119779781734202460600591699938779), c64::new(-0.022148388092658068519431833851198, -0.00145087268721491436830308124243175), c64::new(0.000185377466295724124104506287344493, -0.00115995985777021104467470945144128)],
            [c64::new(-0.00242156549284614570332419841185009, -0.000752322479430815488430528332548245), c64::new(-0.00139626565752539643041368973792751, -0.000266367479596423099701821357242987), c64::new(-0.0199867778076122856597582412777681, -0.0000579849458817324757675502200790038)],
        ]];

        let exemplar_z = vec![mat![
            [c64::new(-50.5306843458574909377748461776506, 0.441923515744679675977346433949763), c64::new(0.0603379440603734332030544683070073, 0.129342546861125538096134969572506), c64::new(-0.0610895095766031133270688228708623, 1.18602167695061060941382673724128)],
            [c64::new(112.514425579694536966523838112259, 19.0540719103955896862803894819252), c64::new(-45.0035003808603554875493064280616, 2.46809231310997251350980022778485), c64::new(0.358414911109330694964335324242691, 0.0210045093254157463565314862274364)],
            [c64::new(-1.47024768885943474951312984751034, -0.977859220813742090612346050886296), c64::new(3.17553247082047295210262230384999, 0.400195488821662924393599582218776), c64::new(-50.0057989787936055145108624132913, -0.00256592432461927395675931817564063)],
        ]];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
}

    #[test]
    #[should_panic(expected = "ABCD parameters do not exist for network with 1 port(s)")]
    fn network_s_to_a() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::S,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.a();
    }

    #[test]
    #[should_panic(expected = "Inverse Hybrid (g) parameters do not exist for network with 1 port(s)")]
    fn network_s_to_g() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::S,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.g();
    }

    #[test]
    #[should_panic(expected = "Hybrid (h) parameters do not exist for network with 1 port(s)")]
    fn network_s_to_h() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::S,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.h();
    }

    #[test]
    #[should_panic(expected = "T parameters do not exist for network with 1 port(s)")]
    fn network_s_to_t() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::S,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.t();
    }

    #[test]
    fn network_t_to_2port() {
        let param = RFParameter::T;

        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_a = vec![
            mat![
                [c64::new(0.330000000000000013808398868775384, -0.10600000000000000741073868937292), c64::new(31.5999999999999974485687115333121, -6.75000000000000027061686225238191)],
                [c64::new(-0.00436000000000000050709436649754056, 0.0000199999999999999483746293549302217), c64::new(1.17199999999999998796795797062487, -0.286000000000000007688294445529209)],
            ],
            mat![
                [c64::new(0.253500000000000058619775700208265, -194.281000000000013128165221587551), c64::new(18.4250000000000024868995751603507, 9839.55000000000065685235028922762)],
                [c64::new(0.0310300000000000020250467969162855, 3.93096000000000026508573114369946), c64::new(1.91250000000000008881784197001252, -201.022000000000013231193918272766)],
            ],
            mat![
                [c64::new(10.4270000000000003514966095963246, 11.8345000000000000195399252334028), c64::new(552.000000000000014266365866433262, -606.174999999999999933386618522491)],
                [c64::new(0.199200000000000006505906924303421, 0.240530000000000000470734562441072), c64::new(11.2310000000000002884359417976157, -12.5795000000000000150990331349021)],
            ],
        ];

        let exemplar_g = vec![
            mat![
                [c64::new(-0.00644715303728300899429714242856297, -0.00199847667455901395200458518859603), c64::new(-0.845219753531416361375281660681806, 0.137366420085595480807425828313056)],
                [c64::new(1.47657115776237431785568062847593, 0.465139471952812184922834778658626), c64::new(49.7993400209725070509733754629798, 4.73155199881283681167927891610054)],
            ],
            mat![
                [c64::new(0.0194489667912751356863639116440774, -0.000544117893132678735708855090394893), c64::new(3.92272242850289374414386555647724, -1.93994418108970863334946099076948)],
                [c64::new(-0.0000993569894934844964686128431861583, -0.00494842222730811740611311514750077), c64::new(48.6884172741781724216085473337871, -1.06880274550876751800447484902907)],
            ],
            mat![
                [c64::new(0.0192514668082711120069811331393428, 0.000553931921796830193884529852866028), c64::new(0.0605893608608473714621121777878414, 0.891512528328108960262454170342129)],
                [c64::new(0.0406840249946560342856332256212342, -0.0463443604928101679240869014930879), c64::new(-5.6352109246790720322134969119338, -50.2437258431668349356451923244352)],
            ],
        ];

        let exemplar_h = vec![
            mat![
                [c64::new(49.8875983514424878011893128263899, 1.72349194454852007734870555680171), c64::new(0.833650805545148090888830126287038, -0.187565005620082440583344782392765)],
                [c64::new(-1.49868864743349579425146233869264, -0.374672161858373988926163679987562), c64::new(-0.00654177594604720988334888009131546, -0.00160359685275384094319754386512035)],
            ],
            mat![
                [c64::new(50.8997839386986348571380683052792, 0.375895184963715302512431691438267), c64::new(-4.15771599610463123087850861448294, 1.90649920705730891619310057032444)],
                [c64::new(-0.000047888944904135476989140602164906, 0.00517288916514360644165953143006421), c64::new(0.020335906386573287873113008525111, 0.0000277347760659542889598702201473387)],
            ],
            mat![
                [c64::new(49.9789537952429160413329054823696, 0.581734382461893019266853787392219), c64::new(0.882022794866543961281868166053159, 0.0489775603088358834491463360017308)],
                [c64::new(-0.0405202162051619424367047093435731, -0.0455508631170759652595867094095653), c64::new(-0.00288472203748202274831842354783835, 0.0188200595367491346094341982595315)],
            ],
        ];

        let exemplar_s = vec![
            mat![
                [c64::new(-1.5375603451309595812449991721764, -0.0741641259593635790266201771512177), c64::new(0.962518630609417802505617870932231, -0.228877015903281448399737721039279)],
                [c64::new(1.74037117254308526694541719249688, 0.412698311136136010093402576493789), c64::new(0.00211787815482265048244293454219161, -0.0399389590404924177255066568605529)],
            ],
            mat![
                [c64::new(-0.00823966748632509301363867097943702, -0.000600998985342353024729247495082033), c64::new(2.02919274404511669494686821758318, -0.954081100695924242940403364938853)],
                [c64::new(0.000000791052406640098234562540536135491, 0.00253600468324170895382167494170621), c64::new(0.0088568524393498832283495246123452, 0.00360090029988248130396266153405555)],
            ],
            mat![
                [c64::new(-38.3519419183273361430692085913092, 36.036385466449672335037553713537), c64::new(31.3783708663005304281170677626092, -0.33156486115084657433915143405249)],
                [c64::new(1.54302893296500743169040964888106, 1.5195786452299769038265361150716), c64::new(-0.0123536115788140826368800362648244, -0.413381672193117820850131528654662)],
            ],
        ];

        let exemplar_t = exemplar;

        let exemplar_y = vec![
            mat![
                [c64::new(0.0373190885169125295548976758668717, -0.00107899216806457043818148336938269), c64::new(-0.0165609260407662913010840929517108, 0.00433189079825403661725483986605305)],
                [c64::new(-0.0302646480657202600443821302731326, -0.0064647586849244234172180714292533), c64::new(0.0106725982822896751626861879800399, -0.00107468232894128797203797406637048)],
            ],
            mat![
                [c64::new(-0.0204295637406422817739543754966095, -0.000232623922020959701257644385453276), c64::new(0.0814033026024068964262823668693359, -0.0380571029314908351446818992801439)],
                [c64::new(-0.000000190307322830582595824312089494934, 0.000101630307644920426214521215367182), c64::new(-0.0197447895566564491069036692903475, -0.0000627363799748357538006561206410265)],
            ],
            mat![
                [c64::new(0.0205680972842337629768768430039319, -0.000202234834655070183108092745520166), c64::new(-0.017656898533430875369548756099743, -0.000774444689316051010660577435193371)],
                [c64::new(-0.000821242654526324780550956239818892, -0.000901841967586041505307541096293727), c64::new(-0.00210975160665101943671519028858368, 0.0191225023910114457243134366972884)],
            ],
        ];

        let exemplar_z = vec![
            mat![
                [c64::new(-75.7980010520778478279769898436235, 23.9642293529721197741177398389446), c64::new(-113.581487638085207751311284209029, 56.5143051025775894191030081332685)],
                [c64::new(-229.352972119936848679673377187781, -1.05207785376117536689415608819853), c64::new(-269.102577590741680056013010915482, 64.3619147816938429683784615102921)],
            ],
            mat![
                [c64::new(-49.4197076355101861033679213873343, -0.454594686267446403163079290921221), c64::new(-204.323723875767854670421481229112, 94.0290559680421366141856055282697)],
                [c64::new(0.00200797113646257846039537644822032, -0.254374934533965112531071847687017), c64::new(-51.1311178450882535354213619120358, -0.890138436090188774142162498476499)],
            ],
            mat![
                [c64::new(50.4802838558149451462171156693598, -1.54378853332916039256399308569166), c64::new(-4.47601987035652395044168942721029, -46.1800197820439041098253725309899)],
                [c64::new(2.04233705453467166220985067154634, -2.46608098256638835255341707519551), c64::new(-8.0845782607149842908189038815315, -53.3881344927220105044399493090193)],
            ],
        ];

        comp_point_c64(&exemplar_a, &calc.net(RFParameter::A), "net(A)");
        comp_point_c64(&exemplar_a, &calc.a(), "a()");

        comp_point_c64(&exemplar_g, &calc.net(RFParameter::G), "net(G)");
        comp_point_c64(&exemplar_g, &calc.g(), "g()");

        comp_point_c64(&exemplar_h, &calc.net(RFParameter::H), "net(H)");
        comp_point_c64(&exemplar_h, &calc.h(), "h()");

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_t, &calc.net(RFParameter::T), "net(T)");
        comp_point_c64(&exemplar_t, &calc.t(), "t()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
}

    #[test]
    fn network_y_to_1port() {
        let param = RFParameter::Y;

        let exemplar = vec![mat![
            [c64::new(0.958, -0.263)],
        ]];

        let calc = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = vec![mat![
            [c64::new(-0.961858445302651090515763702814413, 0.0102568802509230715923989655676159)],
        ]];

        let exemplar_y = exemplar;

        let exemplar_z = vec![mat![
            [c64::new(0.970683926872442228782866863616166, 0.266482121886693452315693978414556)],
        ]];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    #[test]
    fn network_y_to_2port() {
        let param = RFParameter::Y;

        // 2 Port, 3 Frequncy
        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_a = vec![
            mat![
                [c64::new(1.32399999999999957067675637745181, 24.968000000000003152311744969437), c64::new(-8.00000000000000105471187339389871, 44.0000000000000023314683517128287)],
                [c64::new(6.98897600000000068242217432512655, 23.7291320000000021877000211389923), c64::new(3.90800000000000040922820687683277, 44.256000000000000961952739686464)],
            ],
            mat![
                [c64::new(96.8911674659764272181024032371247, -39.3927116177412678457066285190078), c64::new(0.0999767471393880616935223522346543, 0.245685292759143631728102242576931)],
                [c64::new(159.027012324367930880053248462062, -172.377436286634265955744553143897), c64::new(0.445515451895248867386967615842596, 0.403757887416051401295502349900258)],
            ],
            mat![
                [c64::new(-0.0722277064823955826958577216008151, 2.4169134583744652150181723845976), c64::new(-3.78413951957880863430876556178437, 3.61961171437972986567029928876484)],
                [c64::new(-1.46502418558736555686504913395304, 75.8627550180980586500426360456851), c64::new(-79.1880552813425455069073765021249, 78.7958209937479389477316041012898)],
            ],
        ];

        let exemplar_g = vec![
            mat![
                [c64::new(0.962518630609417802505617870932199, -0.228877015903281448399737721039262), c64::new(-1.53756034513095958124499917217646, -0.0741641259593635790266201771512789)],
                [c64::new(0.00211787815482265048244293454219325, -0.039938959040492417725506656860557), c64::new(1.74037117254308526694541719249696, 0.412698311136136010093402576493843)],
            ],
            mat![
                [c64::new(2.02919274404511669494686821758335, -0.954081100695924242940403364938926), c64::new(-0.00823966748632509301363867097943901, -0.000600998985342353024729247495082372)],
                [c64::new(0.0088568524393498832283495246123467, 0.00360090029988248130396266153405676), c64::new(0.000000791052406640098234562540536012665, 0.00253600468324170895382167494170649)],
            ],
            mat![
                [c64::new(31.3783708663005304281170677626096, -0.331564861150846574339151434051402), c64::new(-38.3519419183273361430692085913119, 36.0363854664496723350375537135328)],
                [c64::new(-0.0123536115788140826368800362647759, -0.413381672193117820850131528654686), c64::new(1.54302893296500743169040964888106, 1.51957864522997690382653611507186)],
            ],
        ];

        let exemplar_h = vec![
            mat![
                [c64::new(0.970683926872442228782866863616166, 0.266482121886693452315693978414556), c64::new(0.863302777392183666428306467787739, 0.0720758146702967801388118109669587)],
                [c64::new(-0.00197987097401748661475535120209434, 0.0224209748787405016253479489314833), c64::new(0.545867543186822244746417492697067, -0.109719035638690776866961585879093)],
            ],
            mat![
                [c64::new(0.397612147352765257082445040698002, 0.191118516250815197053694616442173), c64::new(0.715575750368856723223939759880684, -1.24635565988144034918971827686337)],
                [c64::new(-1.23239272013612611331260158924553, 1.11688220696344787208501608264051), c64::new(3.45844082303184425580132018121085, -390.051138087020451915812870397667)],
            ],
            mat![
                [c64::new(0.0468662641434151878798143943652868, 0.00092506433514828603460815663148952), c64::new(0.0666104330091677860871472719554069, -1.13713521532669790803272566840385)],
                [c64::new(0.00634543595955172270622761200577549, 0.00631400574518126857497174497146204), c64::new(0.48829408817838930083424808518387, -0.472132082557874186474657345958685)],
            ],
        ];

        let exemplar_s = vec![
            mat![
                [c64::new(-0.961408341407400732850900910991127, 0.00903541435054956160428211593029545), c64::new(0.056819074278657088758197131580243, 0.0157378352214202281056593458041557)],
                [c64::new(0.00041595055332214627973576894808509, -0.00147429171110075380496230612474459), c64::new(-0.93189738665390497035438290949441, 0.0132486988415427690684525353502727)],
            ],
            mat![
                [c64::new(-0.983957665141952219673821395853227, 0.00746912009070147120633151710284093), c64::new(0.000127698179473950061871908924076091, 0.0000711780149452848709114135329188339)],
                [c64::new(0.000115195318947916282099474901874512, 0.000123913898218471444843058174526802), c64::new(-0.99999908845857061209716522275275, 0.000102530381062122008813824487504103)],
            ],
            mat![
                [c64::new(-0.998713993327856757213729988245082, 0.000025705213831050957478926618341349), c64::new(0.0474045604783883165116067610186095, -0.0453969114801018787188984287548808)],
                [c64::new(-0.0000202983577609627455514200225023237, -0.000515406498986088855085079874721082), c64::new(-0.957741886625998561594678069320832, 0.0392521505280322854235816057472612)],
            ],
        ];

        let exemplar_t = vec![
            mat![
                [c64::new(177.260400000000017039959721021332, 628.280300000000056772947454319925), c64::new(173.512400000000016651825751612358, 583.144300000000055764365347599281)],
                [c64::new(-176.096400000000017490377202111749, -602.43230000000005357400634231627), c64::new(-172.028400000000017060054757767026, -559.056300000000052658682969664062)],
            ],
            mat![
                [c64::new(4024.34464933560550392469283220198, -4328.92792717809166567950211065945), c64::new(4023.89713434876746729607199413889, -4329.33659877136289995343217505444)],
                [c64::new(-3927.45148233468628894535655851778, 4289.54012926620558070643004418596), c64::new(-3927.00796641773382783920346134876, 4289.93897344776644923509098449049)],
            ],
            mat![
                [c64::new(-76.2935875287923975527709331163837, 1937.21143879565646563109749237803), c64::new(2.97015054294172412682261869685788, 1858.34322556762093208605248229111)],
                [c64::new(76.1456770319184257973889000835462, -1934.72213310299440581876591400775), c64::new(-2.96669545903254353683230110722304, -1855.99870434353406146834771589224)],
            ],
        ];

        let exemplar_y = exemplar;

        let exemplar_z = vec![
            mat![
                [c64::new(0.983339062615605700552382119247199, 0.233827900172790393489913919096089), c64::new(1.49460150665477665565964447218079, 0.432452988991509451551132581692921)],
                [c64::new(0.0114214352470227992178057046009944, -0.0387783195429568806892880025629338), c64::new(1.76080827863846485310519872588951, 0.353921365518365507567170503569742)],
            ],
            mat![
                [c64::new(0.403587022436639082458721162881338, 0.189757602732871813023124553271079), c64::new(0.00321137873997047758921840539884824, 0.00180609494050279904534746654027308)],
                [c64::new(0.00289121249557212404542717523227075, 0.00313393171677059316769924431237505), c64::new(0.000022730192220551215708716431202894, 0.00256356485428899456946143285544854)],
            ],
            mat![
                [c64::new(0.0318655280808023732715779332709627, 0.000336712490225446144228707456570845), c64::new(1.23423878324089371439391106555654, -1.1354048751434688927845269807668)],
                [c64::new(-0.000254463584406321724690661721392893, -0.0131767848986768179028856968025573), c64::new(1.05842606053831599632442636862377, 1.02339330394047149228756047634429)],
            ],
        ];

        comp_point_c64(&exemplar_a, &calc.net(RFParameter::A), "net(A)");
        comp_point_c64(&exemplar_a, &calc.a(), "a()");

        comp_point_c64(&exemplar_g, &calc.net(RFParameter::G), "net(G)");
        comp_point_c64(&exemplar_g, &calc.g(), "g()");

        comp_point_c64(&exemplar_h, &calc.net(RFParameter::H), "net(H)");
        comp_point_c64(&exemplar_h, &calc.h(), "h()");

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_t, &calc.net(RFParameter::T), "net(T)");
        comp_point_c64(&exemplar_t, &calc.t(), "t()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }
    
    #[test]
    fn network_y_to_3port() {
        let param = RFParameter::Y;

        // 3 Port
        let exemplar = vec![mat![
            [c64::new(0.958, -0.263), c64::new(-0.846, 0.158), c64::new(1.473, 0.230)],
            [c64::new(0.004, 0.022), c64::new(0.544, -0.129), c64::new(-30.321, 4.378)],
            [c64::new(4.234, 84.212), c64::new(0.457, 0.287), c64::new(3.489, -2.893)],
        ]];

        let calc = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = vec![mat![
            [c64::new(-0.999802065484307321492600345852085, -0.000253013742370538430731092373199772), c64::new(-0.0000201072420423142806560746619851886, -0.0000709326772279715233165234305758413), c64::new(0.0000273009997281201964432468119048808, -0.000472092933728165860604756741060815)],
            [c64::new(-0.0468818126822701998332861326323783, -0.00954057185948404848646051811954556), c64::new(-1.00208932792684885567187635231082, -0.00118666362397589674362414343596268), c64::new(-0.00000613564581550592748251445925124575, -0.000574357719074090799998529570913669)],
            [c64::new(-0.000896920185086412803118581363494916, -0.000107400902301002809710364251237713), c64::new(-0.0013333757382844370296874540100097, -0.000205732002691270452115239395873136), c64::new(-1.00000065301636047042441381144527, -0.0000107942650511071696207491208215757)],
        ]];

        let exemplar_y = exemplar;

        let exemplar_z = vec![mat![
            [c64::new(0.00495054822717464428091662808048489, -0.00627758686807999069198212780545863), c64::new(-0.000505139977548530981138523163189945, -0.00176348823089238474385657172040448), c64::new(0.000680541954739347557308607775482561, -0.0118034269382883182181515043828936)],
            [c64::new(-1.17110939794593614229100104159973, -0.237441626202432912001969918503773), c64::new(-0.0521941454265658254084703418944998, -0.0295514239415189840148760506221144), c64::new(-0.000233826917730008110855671633874032, -0.0140706623681673430493278923595831)],
            [c64::new(-0.0216692279986209777014548005583422, -0.00240358862780036489256153967367999), c64::new(-0.0333025210001457217302023425409882, -0.00511723009911371458159157586576231), c64::new(-0.0000185573287372341018227283282793895, -0.000255194829519354977260741165527909)],
        ]];

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

#[test]
    #[should_panic(expected = "ABCD parameters do not exist for network with 1 port(s)")]
    fn network_y_to_a() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::Y,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.a();
    }

    #[test]
    #[should_panic(expected = "Inverse Hybrid (g) parameters do not exist for network with 1 port(s)")]
    fn network_y_to_g() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::Y,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.g();
    }

    #[test]
    #[should_panic(expected = "Hybrid (h) parameters do not exist for network with 1 port(s)")]
    fn network_y_to_h() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::Y,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.h();
    }

    #[test]
    #[should_panic(expected = "T parameters do not exist for network with 1 port(s)")]
    fn network_y_to_t() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::Y,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.t();
    }

    #[test]
    fn network_z_to_1port() {
        let param = RFParameter::Z;

        // 1 Port
        let exemplar = vec![mat![
            [c64::new(0.958, -0.263)],
        ]];

        let calc = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = vec![mat![
            [c64::new(-0.962348136938965413269904413119227, -0.0101279006243366679760918680135123)],
        ]];

        let exemplar_y = vec![mat![
            [c64::new(0.970683926872442228782866863616166, 0.266482121886693452315693978414556)],
        ]];

        let exemplar_z = exemplar;

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    #[test]
    fn network_z_to_2port() {
        let param = RFParameter::Z;

        // 2 Port, 3 Frequncy
        let exemplar = vec![
            mat![
                [c64::new(0.958, -0.263), c64::new(-0.846, 0.158)],
                [c64::new(0.004, 0.022), c64::new(0.544, -0.129)],
            ],
            mat![
                [c64::new(2.043, -0.982), c64::new(-0.238, 3.249)],
                [c64::new(-1.421, 3.492), c64::new(0.123, -394.321)],
            ],
            mat![
                [c64::new(21.329, -0.421), c64::new(-0.942, 24.282)],
                [c64::new(0.138, 0.132), c64::new(0.329, -0.324)],
            ],
        ];

        let calc = Network::new(
            Frequency::new(row![1.0, 2.0, 3.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_a = vec![
            mat![
                [c64::new(-3.90800000000000040922820687683277, -44.256000000000000961952739686464), c64::new(-6.98897600000000068242217432512655, -23.7291320000000021877000211389923)],
                [c64::new(8.00000000000000105471187339389871, -44.0000000000000023314683517128287), c64::new(-1.32399999999999957067675637745181, -24.968000000000003152311744969437)],
            ],
            mat![
                [c64::new(-0.445515451895248867386967615842596, -0.403757887416051401295502349900258), c64::new(-159.027012324367930880053248462062, 172.377436286634265955744553143897)],
                [c64::new(-0.0999767471393880616935223522346543, -0.245685292759143631728102242576931), c64::new(-96.8911674659764272181024032371247, 39.3927116177412678457066285190078)],
            ],
            mat![
                [c64::new(79.1880552813425455069073765021249, -78.7958209937479389477316041012898), c64::new(1.46502418558736555686504913395304, -75.8627550180980586500426360456851)],
                [c64::new(3.78413951957880863430876556178437, -3.61961171437972986567029928876484), c64::new(0.0722277064823955826958577216008151, -2.4169134583744652150181723845976)],
            ],
        ];

        let exemplar_g = vec![
            mat![
                [c64::new(0.970683926872442228782866863616119, 0.266482121886693452315693978414573), c64::new(0.863302777392183666428306467787712, 0.0720758146702967801388118109669652)],
                [c64::new(-0.00197987097401748661475535120209443, 0.0224209748787405016253479489314812), c64::new(0.545867543186822244746417492697098, -0.109719035638690776866961585879091)],
            ],
            mat![
                [c64::new(0.397612147352765257082445040698021, 0.191118516250815197053694616442202), c64::new(0.71557575036885672322393975988072, -1.24635565988144034918971827686324)],
                [c64::new(-1.23239272013612611331260158924554, 1.11688220696344787208501608264046), c64::new(3.45844082303184425580132018122926, -390.051138087020451915812870397712)],
            ],
            mat![
                [c64::new(0.0468662641434151878798143943652885, 0.000925064335148286034608156631495748), c64::new(0.0666104330091677860871472719555602, -1.137135215326697908032725668404)],
                [c64::new(0.00634543595955172270622761200577528, 0.0063140057451812685749717449714631), c64::new(0.488294088178389300834248085183965, -0.472132082557874186474657345958691)],
            ],
        ];

        let exemplar_h = vec![
            mat![
                [c64::new(0.962518630609417802505617870932231, -0.228877015903281448399737721039303), c64::new(-1.5375603451309595812449991721764, -0.0741641259593635790266201771512177)],
                [c64::new(0.00211787815482265048244293454219161, -0.0399389590404924177255066568605529), c64::new(1.74037117254308526694541719249688, 0.412698311136136010093402576493789)],
            ],
            mat![
                [c64::new(2.02919274404511669494686821758318, -0.954081100695924242940403364938951), c64::new(-0.00823966748632509301363867097943702, -0.000600998985342353024729247495082033)],
                [c64::new(0.0088568524393498832283495246123452, 0.00360090029988248130396266153405555), c64::new(0.000000791052406640098234562540536135491, 0.00253600468324170895382167494170621)],
            ],
            mat![
                [c64::new(31.3783708663005304281170677626092, -0.331564861150846574339151434055448), c64::new(-38.3519419183273361430692085913092, 36.036385466449672335037553713537)],
                [c64::new(-0.0123536115788140826368800362648244, -0.413381672193117820850131528654662), c64::new(1.54302893296500743169040964888106, 1.5195786452299769038265361150716)],
            ],
        ];

        let exemplar_s = vec![
            mat![
                [c64::new(-0.962343087077214849966298176339521, -0.0101141357495970688032923233679108), c64::new(-0.032892112564098003084516968236267, 0.0058810257675148818188727974863375)],
                [c64::new(0.000148712514870334517894963763673215, 0.000855317908616559543957556552359901), c64::new(-0.978456185760186104057754562186095, -0.00503562728691286636905365939888605)],
            ],
            mat![
                [c64::new(-0.921216933492427444871545754629011, -0.0351803877146015293129954681160255), c64::new(-0.0157385425847964522133910839653483, 0.000551104963016969615396972129347133)],
                [c64::new(-0.0175235147010192965345147972097037, -0.00500751430695534269167462789621949), c64::new(0.968131257839102693023687512015405, -0.249601423832744079570969062074772)],
            ],
            mat![
                [c64::new(-0.400580455392576203208433086608045, -0.00950805200654293371152224044281109), c64::new(-0.0351492651514775790944114504869796, 0.675327339524792728191800355283127)],
                [c64::new(0.00379142275352205096269265968459675, 0.00372384017685358614296546985796223), c64::new(-0.98496484876803714011135674445889, -0.0145380214360561588167175811401777)],
            ],
        ];

        let exemplar_t = vec![
            mat![
                [c64::new(197.31411024000002637102013147709, -1134.84929132000006036571803536017), c64::new(198.777889760000025955345331341035, -1109.40670868000005716965228996775)],
                [c64::new(-201.3618897600000267938967818404, 1090.11870868000005936001129525086), c64::new(-202.546110240000026350925094731371, 1065.6252913200000562514535507041)],
            ],
            mat![
                [c64::new(-52.7580302606642188938832767169693, 15.0761189090503600885604525515663), c64::new(47.3136774517995669418201914893946, -27.7641414344235930762610670303145)],
                [c64::new(49.1319745622816114088952441318891, -12.032328070733726170741063838593), c64::new(-44.5786526572074571916060941360027, 23.9128348212748563558506736175387)],
            ],
            mat![
                [c64::new(134.248279725238560058089406647797, -131.855287635735429309632796822516), c64::new(134.146751535044417164256247943531, -127.921119076999002921613771717024)],
                [c64::new(-55.0309239601842672400447291629998, 51.5422115416255291889003400003187), c64::new(-54.9879967374136189684861724240779, 50.6425531836130251468830203366352)],
            ],
        ];

        let exemplar_y = vec![
            mat![
                [c64::new(0.983339062615605700552382119247199, 0.233827900172790393489913919096089), c64::new(1.49460150665477665565964447218079, 0.432452988991509451551132581692921)],
                [c64::new(0.0114214352470227992178057046009944, -0.0387783195429568806892880025629338), c64::new(1.76080827863846485310519872588951, 0.353921365518365507567170503569742)],
            ],
            mat![
                [c64::new(0.403587022436639082458721162881338, 0.189757602732871813023124553271079), c64::new(0.00321137873997047758921840539884824, 0.00180609494050279904534746654027308)],
                [c64::new(0.00289121249557212404542717523227075, 0.00313393171677059316769924431237505), c64::new(0.000022730192220551215708716431202894, 0.00256356485428899456946143285544854)],
            ],
            mat![
                [c64::new(0.0318655280808023732715779332709627, 0.000336712490225446144228707456570845), c64::new(1.23423878324089371439391106555654, -1.1354048751434688927845269807668)],
                [c64::new(-0.000254463584406321724690661721392893, -0.0131767848986768179028856968025573), c64::new(1.05842606053831599632442636862377, 1.02339330394047149228756047634429)],
            ],
        ];

        let exemplar_z = exemplar;

        comp_point_c64(&exemplar_a, &calc.net(RFParameter::A), "net(A)");
        comp_point_c64(&exemplar_a, &calc.a(), "a()");

        comp_point_c64(&exemplar_g, &calc.net(RFParameter::G), "net(G)");
        comp_point_c64(&exemplar_g, &calc.g(), "g()");

        comp_point_c64(&exemplar_h, &calc.net(RFParameter::H), "net(H)");
        comp_point_c64(&exemplar_h, &calc.h(), "h()");

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_t, &calc.net(RFParameter::T), "net(T)");
        comp_point_c64(&exemplar_t, &calc.t(), "t()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    #[test]
    fn network_z_to_3port() {
        let param = RFParameter::Z;

        // 3 Port
        let exemplar = vec![mat![
            [c64::new(0.958, -0.263), c64::new(-0.846, 0.158), c64::new(1.473, 0.230)],
            [c64::new(0.004, 0.022), c64::new(0.544, -0.129), c64::new(-30.321, 4.378)],
            [c64::new(4.234, 84.212), c64::new(0.457, 0.287), c64::new(3.489, -2.893)],
        ]];

        let calc = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![
                c64::from(50.0),
                c64::from(50.0),
                c64::from(50.0),
            ],
            param,
            exemplar.clone(),
            String::from(""),
            String::from(""),
        );

        let exemplar_s = vec![mat![
            [c64::new(-0.936358403314655353907042304777799, -0.0673580603917468450419793104777854), c64::new(-0.0328407404651124598017602337916295, 0.00449361574220472593322021906494632), c64::new(0.0338532374150907156983864533046989, 0.0172473958428260604893600449329644)],
            [c64::new(0.192900708972740175456459450016086, 1.82923582950801338922867394036326), c64::new(-0.959199262245123758759605843197135, 0.0307101827973775658142038906111744), c64::new(-1.10913363606767871186303689458132, 0.066574111941438592182109751282926)],
            [c64::new(-0.10855886998740481879751720790784, 3.03136016375700646599100408238676), c64::new(0.0231786392368656409175901291497196, 0.062851612575179139483522423554234), c64::new(-0.827225039429119618503745712887632, -0.148107823250993687901717078396325)],
        ]];

        let exemplar_y = vec![mat![
            [c64::new(0.00495054822717464428091662808048489, -0.00627758686807999069198212780545863), c64::new(-0.000505139977548530981138523163189945, -0.00176348823089238474385657172040448), c64::new(0.000680541954739347557308607775482561, -0.0118034269382883182181515043828936)],
            [c64::new(-1.17110939794593614229100104159973, -0.237441626202432912001969918503773), c64::new(-0.0521941454265658254084703418944998, -0.0295514239415189840148760506221144), c64::new(-0.000233826917730008110855671633874032, -0.0140706623681673430493278923595831)],
            [c64::new(-0.0216692279986209777014548005583422, -0.00240358862780036489256153967367999), c64::new(-0.0333025210001457217302023425409882, -0.00511723009911371458159157586576231), c64::new(-0.0000185573287372341018227283282793895, -0.000255194829519354977260741165527909)],
        ]];

        let exemplar_z = exemplar;

        comp_point_c64(&exemplar_s, &calc.net(RFParameter::S), "net(S)");
        comp_point_c64(&exemplar_s, &calc.s(), "s()");

        comp_point_c64(&exemplar_y, &calc.net(RFParameter::Y), "net(Y)");
        comp_point_c64(&exemplar_y, &calc.y(), "y()");

        comp_point_c64(&exemplar_z, &calc.net(RFParameter::Z), "net(Z)");
        comp_point_c64(&exemplar_z, &calc.z(), "z()");
    }

    #[test]
    #[should_panic(expected = "ABCD parameters do not exist for network with 1 port(s)")]
    fn network_z_to_a() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::Z,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.a();
    }

    #[test]
    #[should_panic(expected = "Inverse Hybrid (g) parameters do not exist for network with 1 port(s)")]
    fn network_z_to_g() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::Z,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.g();
    }

    #[test]
    #[should_panic(expected = "Hybrid (h) parameters do not exist for network with 1 port(s)")]
    fn network_z_to_h() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::Z,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.h();
    }

    #[test]
    #[should_panic(expected = "T parameters do not exist for network with 1 port(s)")]
    fn network_z_to_t() {
        let net = Network::new(
            Frequency::new(row![1.0], Unit::Giga),
            row![c64::from(50.0)],
            RFParameter::Z,
            vec![mat![
                [c64::new(0.958, -0.263)],
            ]],
            String::from(""),
            String::from(""),
        );

        net.t();
    }

    fn comp_point_c64(exemplar: &Vec<Point>, calc: &Vec<Point>, test: &str) {
        for i in 0..calc.len() {
            for j in 0..calc[i].nrows() {
                for k in 0..calc[i].ncols() {
                    comp_point(exemplar[i][(j, k)].re, calc[i][(j, k)].re, test, format!("re({},{},{})", i, j, k));
                    comp_point(exemplar[i][(j, k)].im, calc[i][(j, k)].im, test, format!("im({},{},{})", i, j, k));
                }
            }
        }
    }

    fn comp_point_f64(
        exemplar: &Vec<Pointf64>,
        calc: &Vec<Pointf64>,
        test: &str,
    ) {
        for i in 0..calc.len() {
            for j in 0..calc[i].nrows() {
                for k in 0..calc[i].ncols() {
                    comp_point(exemplar[i][(j, k)], calc[i][(j, k)], test, format!("({},{},{})", i, j, k));
                }
            }
        }
    }

    fn comp_point(exemplar: f64, calc: f64, test: &str, idx: String) {
        if exemplar == calc {
            return ();
        }

        // let base: f64 = 2.0;
        // let eps = base.powi(-53);
        let base: f64 = 10.0;
        let eps = base.powi(-10);

        let val = (calc - exemplar).abs() / (eps * exemplar);
        debug_assert!((calc - exemplar).abs() < (eps * exemplar).abs(), " Failed test {} at location {}\n  exemplar: {}\n      calc: {}", test, idx, exemplar, calc);
        // debug_assert!(val * eps.sqrt() < 1.0, " Failed test {} at location {}\n  left: {}\n right: {}", test, idx, exemplar, calc);
    }

    fn calc_err(a: Row<f64>, b: Row<f64>) {
        let base: f64 = 2.0;
        let eps = base.powi(-53);
    
        for i in 0..a.nrows() {
          let val = (a.get(i) - b.get(i)).abs() / (eps * a.get(i));
          assert!(val * eps.sqrt() < 1.0);
        }
    }
}
