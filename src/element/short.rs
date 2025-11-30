use crate::element::{Elem, ElemType, Lumped};
use crate::frequency::Frequency;
use crate::point;
use crate::point::Point;
use crate::points::{Points, Pts};
use crate::scale::Scale;
use crate::unit::{Unit, UnitVal, Unitized};
use num::complex::{Complex, Complex64, c64};

#[derive(Clone, Debug, PartialEq)]
pub struct Short {
    id: String,
    nodes: [usize; 2],
    c: Point<Complex64>,
    z0: Complex64,
}

impl Short {
    pub fn new(id: String, nodes: [usize; 2], z0: Complex64) -> Self {
        Self {
            id: id,
            nodes: nodes,
            c: point![
                Complex64,
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
            c: point![
                Complex64,
                [c64(0.0, 0.0), c64(1.0, 0.0)],
                [c64(1.0, 0.0), c64(0.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        }
    }
}

impl Elem for Short {
    fn c(&self, _freq: &Frequency) -> Point<Complex64> {
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

    fn net(&self, freq: &Frequency) -> Points<Complex64> {
        Points::from_shape_fn((freq.npts(), 2, 2), |(_, j, k)| match (j, k) {
            (0, 0) | (1, 1) => c64(0.0, 0.0),
            (1, 0) | (0, 1) => c64(1.0, 0.0),
            _ => c64(0.0, 0.0),
        })
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

    fn unitval(&self) -> UnitVal {
        UnitVal::default()
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

    fn set_unitval(&mut self, _val: UnitVal) {
        ();
    }

    fn set_scale(&mut self, _scale: Scale) {
        ();
    }

    fn set_unit(&mut self, _unit: Unit) {
        ();
    }
}
