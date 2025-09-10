use crate::elements::{Elem, ElemType, Term};
use crate::frequency::Frequency;
use crate::point;
use crate::point::Point;
use crate::points::{Points, Pts};
use ndarray::prelude::*;
use num::complex::Complex64;

#[derive(Clone, Debug, PartialEq)]
pub struct Ground {
    id: String,
    z: Complex64,
    node: [usize; 1],
    c: Point,
    z0: Complex64,
}

impl Ground {
    pub fn new() -> Ground {
        Ground::default()
    }

    fn z0(&self) -> Complex64 {
        self.z0
    }
}

impl Default for Ground {
    fn default() -> Self {
        Self {
            id: "gnd".to_string(),
            z: Complex64::ZERO,
            node: [0],
            c: point![[Complex64::ZERO]],
            z0: Complex64::ZERO,
        }
    }
}

impl Elem for Ground {
    fn c(&self, _freq: &Frequency) -> Point {
        self.c.clone()
    }

    fn c_at(&self, _freq: &Frequency, _j: usize, _k: usize) -> Complex64 {
        self.c[[0, 0]]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Ground
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points {
        Points::zeros((freq.npts(), 1, 1))
    }

    fn nodes(&self) -> Vec<usize> {
        self.node.to_vec()
    }

    fn z(&self, _freq: &Frequency) -> Complex64 {
        Complex64::ZERO
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = Frequency::new(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Term for Ground {
    fn val(&self) -> Complex64 {
        self.z
    }

    fn set_val(&mut self, val: Complex64) {
        self.z = val;
    }
}
