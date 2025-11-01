use crate::element::{Elem, ElemType, Lumped};
use crate::frequency::Frequency;
use crate::point;
use crate::point::Point;
use crate::points::{Points, Pts};
use crate::scale::Scale;
use crate::unit::{Unit, UnitVal, Unitized};
use ndarray::prelude::*;
use num::complex::{Complex, Complex64, c64};
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct Inductor {
    id: String,
    ind: UnitVal,
    nodes: [usize; 2],
    c: Point<Complex64>,
    z0: Complex64,
}

impl Inductor {
    pub fn new(id: String, ind: UnitVal, nodes: [usize; 2], z0: Complex64) -> Inductor {
        Inductor {
            id: id,
            ind: ind,
            nodes: nodes,
            c: point![
                Complex64,
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: z0,
        }
    }

    fn z0(&self) -> Complex64 {
        self.z0
    }
}

impl Default for Inductor {
    fn default() -> Self {
        Self {
            id: "L0".to_string(),
            ind: UnitVal::default(),
            nodes: [0, 0],
            c: point![
                Complex64,
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        }
    }
}

impl Elem for Inductor {
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
        ElemType::Inductor
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points<Complex64> {
        Points::from_shape_fn((freq.npts(), 2, 2), |(_, j, k)| match (j, k) {
            (0, 0) | (1, 1) => c64(1.0 / 3.0, 0.0),
            (1, 0) | (0, 1) => c64(2.0 / 3.0, 0.0),
            _ => c64(0.0, 0.0),
        })
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    fn z(&self, freq: &Frequency) -> Complex<f64> {
        Complex64::I * 2.0 * PI * freq.freq(0) * self.ind.val()
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = Frequency::new(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Lumped for Inductor {
    fn val(&self) -> f64 {
        self.ind.val()
    }

    fn set_val(&mut self, val: f64) {
        self.ind.set_val(val);
    }
}

impl Unitized for Inductor {
    fn val_scaled(&self) -> f64 {
        self.ind.val_scaled()
    }

    fn unitval(&self) -> UnitVal {
        self.ind.clone()
    }

    fn scale(&self) -> Scale {
        self.ind.scale()
    }

    fn unit(&self) -> Unit {
        self.ind.unit()
    }

    fn set_val_scaled(&mut self, val: f64) {
        self.ind.set_val_scaled(val);
    }

    fn set_unitval(&mut self, val: UnitVal) {
        self.ind = val;
    }

    fn set_scale(&mut self, scale: Scale) {
        self.ind.set_scale(scale);
    }

    fn set_unit(&mut self, unit: Unit) {
        self.ind.set_unit(unit);
    }
}

pub struct InductorBuilder {
    id: String,
    ind: UnitVal,
    nodes: [usize; 2],
    z0: Complex64,
}

impl InductorBuilder {
    pub fn new() -> Self {
        InductorBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn ind(mut self, ind: UnitVal) -> Self {
        self.ind = ind;
        self
    }

    pub fn val(mut self, ind: f64) -> Self {
        self.ind.set_val(ind);
        self
    }

    pub fn val_scaled(mut self, ind: f64, scale: Scale) -> Self {
        self.ind.set_scale(scale);
        self.ind.set_val_scaled(ind);
        self
    }

    pub fn nodes(mut self, nodes: [usize; 2]) -> Self {
        self.nodes = nodes;
        self
    }

    pub fn z0(mut self, z0: Complex64) -> Self {
        self.z0 = z0;
        self
    }

    pub fn build(self) -> Inductor {
        Inductor {
            id: self.id,
            ind: self.ind,
            nodes: self.nodes,
            c: point![
                Complex64,
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: self.z0,
        }
    }
}

impl Default for InductorBuilder {
    fn default() -> Self {
        Self {
            id: "L0".to_string(),
            ind: *UnitVal::default().set_unit(Unit::Henry),
            nodes: [0, 0],
            z0: c64(50.0, 0.0),
        }
    }
}

#[cfg(test)]
mod element_inductor_tests {
    use super::*;
    use crate::unit::UnitValBuilder;
    use crate::util::{comp_c64, comp_point_c64};
    use float_cmp::F64Margin;

    #[test]
    fn element_inductor() {
        let freq_unitval = UnitValBuilder::new().val_scaled(1.0, Scale::Giga).build();
        let freq = Frequency::from_unitval(&freq_unitval);
        let val_scaled = 1.0;
        let scale = Scale::Nano;
        let unitval = UnitValBuilder::new().val_scaled(val_scaled, scale).build();
        let exemplar = Inductor {
            id: "L1".to_string(),
            ind: unitval,
            nodes: [1, 2],
            c: point![
                Complex64,
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        };
        let exemplar_z = Complex64::I * 2.0 * PI * freq.freq(0) * unitval.val();
        let calc = InductorBuilder::new()
            .val_scaled(val_scaled, scale)
            .nodes([1, 2])
            .id("L1")
            .build();
        let margin = F64Margin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.scale(), &calc.scale());
        assert_eq!(&Unit::Henry, &calc.unit());
        comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        comp_c64(&exemplar_z.into(), &calc.z(&freq), margin, "calc.z()", "0");
    }
}
