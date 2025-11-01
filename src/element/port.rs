use crate::element::{Elem, ElemType, Term};
use crate::frequency::Frequency;
use crate::point;
use crate::point::{Point, Pt};
use crate::points::{Points, Pts};
use ndarray::prelude::*;
use num::complex::{Complex64, c64};

#[derive(Clone, Debug, PartialEq)]
pub struct Port {
    id: String,
    z: Complex64,
    node: [usize; 1],
    c: Point<Complex64>,
}

impl Port {
    pub fn new(id: String, z: Complex64, node: [usize; 1]) -> Port {
        Port {
            id,
            z,
            node,
            c: point![Complex64, [Complex64::ZERO]],
        }
    }

    fn z0(&self) -> Complex64 {
        self.z
    }
}

impl Default for Port {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            z: Complex64::ZERO,
            node: [0],
            c: Point::zeros((1, 1)),
        }
    }
}

impl Elem for Port {
    fn c(&self, _freq: &Frequency) -> Point<Complex64> {
        self.c.clone()
    }

    fn c_at(&self, _freq: &Frequency, _j: usize, _k: usize) -> Complex64 {
        self.c[[0, 0]]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Port
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points<Complex64> {
        Points::zeros((freq.npts(), 1, 1))
    }

    fn nodes(&self) -> Vec<usize> {
        self.node.to_vec()
    }

    fn z(&self, _freq: &Frequency) -> Complex64 {
        self.z
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = Frequency::new(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Term for Port {
    fn val(&self) -> Complex64 {
        self.z
    }

    fn set_val(&mut self, val: Complex64) {
        self.z = val;
    }
}

pub struct PortBuilder {
    id: String,
    z: Complex64,
    node: [usize; 1],
    z0: Complex64,
}

impl PortBuilder {
    pub fn new() -> Self {
        PortBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn z(mut self, z: Complex64) -> Self {
        self.z = z;
        self
    }

    pub fn nodes(mut self, node: [usize; 1]) -> Self {
        self.node = node;
        self
    }

    pub fn build(self) -> Port {
        Port {
            id: self.id,
            z: self.z,
            node: self.node,
            c: Point::zeros((1, 1)),
        }
    }
}

impl Default for PortBuilder {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            z: Complex64::ZERO,
            node: [0],
            z0: c64(50.0, 0.0),
        }
    }
}

#[cfg(test)]
mod element_port_tests {
    use super::*;
    use crate::scale::Scale;
    use crate::unit::UnitValBuilder;
    use crate::util::{comp_c64, comp_point_c64};
    use float_cmp::F64Margin;

    #[test]
    fn element_port() {
        let freq_unitval = UnitValBuilder::new().val_scaled(1.0, Scale::Giga).build();
        let freq = Frequency::from_unitval(&freq_unitval);
        let val = c64(50.0, 0.0);
        let exemplar = Port {
            id: "P1".to_string(),
            z: val,
            node: [1],
            c: point![Complex64, [Complex64::ZERO]],
        };
        let exemplar_z = val;
        let calc = PortBuilder::new().z(val).nodes([1]).id("P1").build();
        let margin = F64Margin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        comp_c64(&exemplar_z.into(), &calc.z(&freq), margin, "calc.z()", "0");
    }
}
