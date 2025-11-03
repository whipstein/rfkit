use crate::define_mlin_calcs;
use crate::element::{Distributed, Elem, ElemType, msub::Msub};
use crate::frequency::Frequency;
use crate::point;
use crate::point::Point;
use crate::points::{Points, Pts};
use crate::scale::Scale;
use crate::unit::{Unit, UnitVal, Unitized};
use ndarray::prelude::*;
use num::complex::{Complex64, c64};
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct Mlef {
    id: String,
    width: UnitVal,
    length: UnitVal,
    sub: Msub,
    nodes: [usize; 1],
}

impl Mlef {
    pub fn new(id: String, width: UnitVal, length: UnitVal, sub: &Msub, nodes: [usize; 1]) -> Mlef {
        Mlef {
            id,
            width,
            length,
            sub: sub.clone(),
            nodes,
        }
    }
}

define_mlin_calcs!(Mlef);

impl Default for Mlef {
    fn default() -> Self {
        Self {
            id: "ML0".to_string(),
            width: *UnitVal::default().set_unit(Unit::Meter),
            length: *UnitVal::default().set_unit(Unit::Meter),
            sub: Msub::default(),
            nodes: [1],
        }
    }
}

impl Elem for Mlef {
    // todo!("fix c value for true mlef response");
    fn c(&self, _freq: &Frequency) -> Point<Complex64> {
        // point![
        //     Complex64,
        //     [
        //         Complex64::ZERO,
        //         mlin_exp(self.length, self.gamma(freq)).exp()
        //     ],
        //     [
        //         mlin_exp(self.length, self.gamma(freq)).exp(),
        //         Complex64::ZERO
        //     ]
        // ]
        point![Complex64, [Complex64::ZERO]]
    }

    fn c_at(&self, freq: &Frequency, j: usize, k: usize) -> Complex64 {
        self.c(freq)[[j, k]]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Mlef
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points<Complex64> {
        Points::zeros((freq.npts(), 2, 2))
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    fn z(&self, freq: &Frequency) -> Complex64 {
        self.z0(&freq).into()
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = Frequency::new(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Distributed for Mlef {
    fn width(&self) -> f64 {
        self.width.val()
    }

    fn length(&self, freq: &Frequency) -> f64 {
        let w = self.width.val();
        let t = self.sub.thickness().val();
        let w_eff = self.w_eff(w, t);
        let h = self.sub.height().val();
        let er = self.sub.er();
        let er_eff = self.er(freq);
        let q1 = 0.434907 * (er_eff.powf(0.81) + 0.26) / (er_eff.powf(0.81) - 0.189)
            * ((w_eff / h).powf(0.8544) + 0.236)
            / ((w_eff / h).powf(0.8544) + 0.87);
        let q2 = 1.0 + (w_eff / h).powf(0.371) / (2.358 * er + 1.0);
        let q3 =
            1.0 + 0.5274 / er_eff.powf(0.9236) * (0.084 * (w_eff / h).powf(1.9413 / q2)).atan();
        let q4 = 1.0
            + 0.0377
                * (6.0 - 5.0 * (0.036 * (1.0 - er)).exp())
                * (0.067 * (w_eff / h).powf(1.456)).atan();
        let q5 = 1.0 - 0.218 * (-7.5 * w_eff / h).exp();

        self.length.val() + q1 * q3 * q5 / q4 * h
    }

    fn sub(&self) -> &Msub {
        &self.sub
    }

    fn val(&self) -> f64 {
        self.length.val()
    }

    fn gamma(&self, freq: &Frequency) -> Complex64 {
        c64(self.alpha(freq), self.beta(freq))
    }

    fn er(&self, freq: &Frequency) -> f64 {
        self.er(freq)
    }

    fn set_width_val(&mut self, val: f64) {
        self.width.set_val(val);
    }

    fn set_width_unit(&mut self, unit: Unit) {
        self.width.set_unit(unit);
    }

    fn set_length_val(&mut self, val: f64) {
        self.length.set_val(val);
    }

    fn set_length_unit(&mut self, unit: Unit) {
        self.length.set_unit(unit);
    }
}

impl Unitized for Mlef {
    fn val_scaled(&self) -> f64 {
        self.length.val_scaled()
    }

    fn unitval(&self) -> UnitVal {
        self.length.clone()
    }

    fn scale(&self) -> Scale {
        self.length.scale()
    }

    fn unit(&self) -> Unit {
        self.length.unit()
    }

    fn set_val_scaled(&mut self, val: f64) {
        self.length.set_val_scaled(val);
    }

    fn set_unitval(&mut self, val: UnitVal) {
        self.length = val;
    }

    fn set_scale(&mut self, scale: Scale) {
        self.length.set_scale(scale);
    }

    fn set_unit(&mut self, unit: Unit) {
        self.length.set_unit(unit);
    }
}

#[derive(Clone)]
pub struct MlefBuilder {
    id: String,
    width: UnitVal,
    length: UnitVal,
    sub: Msub,
    nodes: [usize; 1],
    z0: Option<f64>,
}

impl MlefBuilder {
    pub fn new() -> Self {
        MlefBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn width(mut self, val: UnitVal) -> Self {
        self.width = val;
        self
    }

    pub fn width_val(mut self, val: f64) -> Self {
        self.width.set_val(val);
        self
    }

    pub fn width_scaled(mut self, val: f64, scale: Scale) -> Self {
        self.width.set_scale(scale);
        self.width.set_val_scaled(val);
        self
    }

    pub fn width_scale(mut self, val: Scale) -> Self {
        self.width.set_scale(val);
        self
    }

    pub fn width_unit(mut self, val: Unit) -> Self {
        self.width.set_unit(val);
        self
    }

    pub fn length(mut self, val: UnitVal) -> Self {
        self.length = val;
        self
    }

    pub fn length_val(mut self, val: f64) -> Self {
        self.length.set_val(val);
        self
    }

    pub fn length_scaled(mut self, val: f64, scale: Scale) -> Self {
        self.length.set_scale(scale);
        self.length.set_val_scaled(val);
        self
    }

    pub fn length_scale(mut self, val: Scale) -> Self {
        self.length.set_scale(val);
        self
    }

    pub fn length_unit(mut self, val: Unit) -> Self {
        self.length.set_unit(val);
        self
    }

    pub fn sub(mut self, val: &Msub) -> Self {
        self.sub = val.clone();
        self
    }

    pub fn nodes(mut self, nodes: [usize; 1]) -> Self {
        self.nodes = nodes;
        self
    }

    pub fn z0(mut self, z0: f64) -> Self {
        self.z0 = Some(z0);
        self
    }

    pub fn build(self) -> Mlef {
        Mlef {
            id: self.id,
            width: self.width,
            length: self.length,
            sub: self.sub,
            nodes: self.nodes,
        }
    }
}

impl Default for MlefBuilder {
    fn default() -> Self {
        Self {
            id: "ML0".to_string(),
            width: *UnitVal::default().set_unit(Unit::Meter),
            length: *UnitVal::default().set_unit(Unit::Meter),
            sub: Msub::default(),
            nodes: [1],
            z0: None,
        }
    }
}

#[cfg(test)]
mod element_mlef_tests {
    use super::*;
    use crate::element::msub::MsubBuilder;
    // use crate::unit::UnitValBuilder;
    // use crate::util::{comp_c64, comp_f64, comp_point_c64};
    use float_cmp::F64Margin;

    const DEFAULT_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-10,
        ulps: 4,
    };

    const RELAXED_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-6,
        ulps: 10,
    };

    // #[test]
    // fn element_mlef() {
    //     let freq_unitval = UnitValBuilder::new()
    //         .val_scaled(1.0, Scale::Giga)
    //         .unit(Unit::Hz)
    //         .build();
    //     let freq = Frequency::from_unitval(&freq_unitval);
    //     let sub = MsubBuilder::new()
    //         .id("Msub0")
    //         .er(12.4)
    //         .tand(0.0004)
    //         .height(
    //             UnitValBuilder::new()
    //                 .val(25e-6)
    //                 .scale(Scale::Micro)
    //                 .unit(Unit::Meter)
    //                 .build(),
    //         )
    //         .thickness(
    //             UnitValBuilder::new()
    //                 .val(0.77e-6)
    //                 .scale(Scale::Micro)
    //                 .unit(Unit::Meter)
    //                 .build(),
    //         )
    //         .build();

    //     let width_val = 5.8736e-6;
    //     let length_val = 0.25;
    //     let exemplar = Mlef {
    //         id: "ML1".to_string(),
    //         width: UnitValBuilder::new()
    //             .val(width_val)
    //             .scale(Scale::Micro)
    //             .unit(Unit::Meter)
    //             .build(),
    //         length: UnitValBuilder::new()
    //             .val(length_val)
    //             .unit(Unit::Lambda)
    //             .build(),
    //         sub: sub.clone(),
    //         nodes: [1, 2],
    //     };
    //     let exemplar_z = c64(2_f64.sqrt() * 50.0, 0.0);
    //     let calc = MlefBuilder::new()
    //         .width_val(width_val)
    //         .width_scale(Scale::Micro)
    //         .width_unit(Unit::Meter)
    //         .length_val(length_val)
    //         .length_unit(Unit::Lambda)
    //         .sub(&sub)
    //         .nodes([1, 2])
    //         .id("ML1")
    //         .build();
    //     let margin = F64Margin::default();

    //     assert_eq!(&exemplar.id(), &calc.id());
    //     assert_eq!(&exemplar.scale(), &calc.scale());
    //     assert_eq!(&exemplar.unit(), &calc.unit());
    //     assert_eq!(&exemplar.gamma(&freq), &calc.gamma(&freq));
    //     assert_eq!(&exemplar.er(&freq), &calc.er(&freq));
    //     comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");

    //     let margin = F64Margin {
    //         epsilon: 1e-4,
    //         ulps: 10,
    //     };
    //     comp_c64(&exemplar_z, &calc.z(&freq), margin, "calc.z()", "0");
    //     comp_f64(&exemplar_z.re, &calc.z0(&freq), margin, "calc.z0()", "0");
    // }

    mod mlef_tests {
        use super::*;

        #[test]
        fn test_mlef_builder_default() {
            let mlef = MlefBuilder::new().build();
            assert_eq!(mlef.id(), "ML0");
            assert_eq!(mlef.nodes(), vec![1]);
        }

        #[test]
        fn test_mlef_end_effect_extends_length() {
            let freq = Frequency::new(array![1e9], Scale::Base);
            let sub = MsubBuilder::new()
                .er(12.4)
                .height_scaled(25.0, Scale::Micro)
                .thickness_scaled(0.77, Scale::Micro)
                .build();

            let mlef = MlefBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .length_scaled(1000.0, Scale::Micro)
                .sub(&sub)
                .build();

            let physical_length = mlef.val();
            let effective_length = mlef.length(&freq);

            // Effective length should be greater due to end effect
            assert!(effective_length > physical_length);
        }

        #[test]
        fn test_mlef_with_substrate() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .height_scaled(25.0, Scale::Micro)
                .build();

            let mlef = MlefBuilder::new()
                .id("ML1")
                .width_scaled(5.8736, Scale::Micro)
                .length_scaled(0.25, Scale::Base)
                .sub(&sub)
                .nodes([1])
                .build();

            assert_eq!(mlef.id(), "ML1");
            assert_eq!(mlef.nodes(), vec![1]);
        }

        #[test]
        fn test_mlef_c_matrix() {
            let freq = Frequency::new(array![1e9], Scale::Base);
            let sub = MsubBuilder::new().build();
            let mlef = MlefBuilder::new().sub(&sub).build();

            let c_matrix = mlef.c(&freq);
            assert_eq!(c_matrix[[0, 0]], Complex64::ZERO);
        }

        #[test]
        fn test_mlef_elem_type() {
            let sub = MsubBuilder::new().build();
            let mlef = MlefBuilder::new().sub(&sub).build();
            assert_eq!(mlef.elem(), ElemType::Mlef);
        }
    }
}
