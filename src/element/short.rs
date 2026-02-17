use crate::{
    element::{Elem, ElemType, Lumped},
    frequency::{FreqArray, Frequency},
    pts::{Points, Pts},
    scale::Scale,
    unit::{Unit, UnitValue, Unitized},
};
use ndarray::{IntoDimension, prelude::*};
use num::complex::{Complex, Complex64, c64};

#[derive(Clone, Debug, PartialEq)]
pub struct Short {
    id: String,
    nodes: [usize; 2],
    c: Points<Complex64, Ix2>,
    z0: Complex64,
}

impl Short {
    pub fn new(id: String, nodes: [usize; 2], z0: Complex64) -> Self {
        Self {
            id: id,
            nodes: nodes,
            c: points![
                [c64(0.0, 0.0), c64(1.0, 0.0)],
                [c64(1.0, 0.0), c64(0.0, 0.0)]
            ],
            z0: z0,
        }
    }

    fn z0(&self) -> Complex64 {
        self.z0
    }
}

impl Default for Short {
    fn default() -> Self {
        Self {
            id: "S0".to_string(),
            nodes: [1, 2],
            c: points![
                [c64(0.0, 0.0), c64(1.0, 0.0)],
                [c64(1.0, 0.0), c64(0.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        }
    }
}

impl Elem for Short {
    fn c(&self, _freq: &Frequency) -> Points<Complex64, Ix2> {
        self.c.clone()
    }

    fn c_at(&self, _freq: &Frequency, j: usize, k: usize) -> Complex64 {
        self.c[[j, k]]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Short
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points<Complex64, Ix3> {
        Points::<Complex64, Ix3>::from_shape_fn(
            (freq.npts(), 2, 2).into_dimension(),
            |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.0, 0.0),
                (1, 0) | (0, 1) => c64(1.0, 0.0),
                _ => c64(0.0, 0.0),
            },
        )
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    fn z(&self, _freq: &Frequency) -> Complex<f64> {
        c64(1e-10, 0.0)
        // Complex64::ZERO
    }

    fn z_at(&self, _freq: &Frequency, _i: usize) -> Complex64 {
        self.z(_freq)
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Lumped for Short {
    fn val(&self) -> f64 {
        0.0
    }

    fn set_val(&mut self, _val: f64) {
        ();
    }
}

impl Unitized for Short {
    fn val_scaled(&self) -> f64 {
        1e-9
    }

    fn unitval(&self) -> UnitValue {
        UnitValue::default()
    }

    fn scale(&self) -> Scale {
        Scale::Base
    }

    fn unit(&self) -> Unit {
        Unit::None
    }

    fn set_val_scaled(&mut self, _val: f64) {
        ();
    }

    fn set_unitval(&mut self, _val: UnitValue) {
        ();
    }

    fn set_scale(&mut self, _scale: Scale) {
        ();
    }

    fn set_unit(&mut self, _unit: Unit) {
        ();
    }
}
