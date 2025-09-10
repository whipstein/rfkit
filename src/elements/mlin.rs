use crate::define_mlin_calcs;
use crate::elements::{Distributed, Elem, ElemType, mlin_exp, msub::Msub};
use crate::frequency::Frequency;
// use crate::minimize::{Minimizer, NelderMeadBounded};
// use crate::myfloat::MyFloat;
use crate::point;
use crate::point::Point;
use crate::points::{Points, Pts};
use crate::scale::Scale;
use crate::unit::{Unit, UnitVal, Unitized};
use ndarray::prelude::*;
use num::complex::{Complex64, c64};
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct Mlin {
    id: String,
    width: UnitVal,
    length: UnitVal,
    sub: Msub,
    nodes: [usize; 2],
}

impl Mlin {
    pub fn new(id: String, width: UnitVal, length: UnitVal, sub: &Msub, nodes: [usize; 2]) -> Mlin {
        Mlin {
            id,
            width,
            length,
            sub: sub.clone(),
            nodes,
        }
    }
}

define_mlin_calcs!(Mlin);

impl Default for Mlin {
    fn default() -> Self {
        Self {
            id: "ML0".to_string(),
            width: *UnitVal::default().set_unit(Unit::Meter),
            length: *UnitVal::default().set_unit(Unit::Meter),
            sub: Msub::default(),
            nodes: [0, 0],
        }
    }
}

impl Elem for Mlin {
    fn c(&self, freq: &Frequency) -> Point {
        point![
            [
                Complex64::ZERO,
                mlin_exp(self.length, self.gamma(freq)).exp()
            ],
            [
                mlin_exp(self.length, self.gamma(freq)).exp(),
                Complex64::ZERO
            ]
        ]
    }

    fn c_at(&self, freq: &Frequency, j: usize, k: usize) -> Complex64 {
        self.c(freq)[[j, k]]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Mlin
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points {
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

impl Distributed for Mlin {
    fn width(&self) -> f64 {
        self.width.val()
    }

    fn length(&self, _freq: &Frequency) -> f64 {
        self.length.val()
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

impl Unitized for Mlin {
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

pub struct MlinBuilder {
    id: String,
    width: UnitVal,
    length: UnitVal,
    scale: Scale,
    unit: Unit,
    sub: Msub,
    nodes: [usize; 2],
    z0: Option<f64>,
    freq: Frequency,
}

impl MlinBuilder {
    pub fn new() -> Self {
        MlinBuilder::default()
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

    pub fn scale(mut self, val: Scale) -> Self {
        self.scale = val;
        self
    }

    pub fn unit(mut self, val: Unit) -> Self {
        self.unit = val;
        self
    }

    pub fn sub(mut self, val: &Msub) -> Self {
        self.sub = val.clone();
        self
    }

    pub fn nodes(mut self, nodes: [usize; 2]) -> Self {
        self.nodes = nodes;
        self
    }

    pub fn z0(mut self, z0: f64, freq: &Frequency) -> Self {
        self.z0 = Some(z0);
        self.freq = freq.clone();
        self
    }

    pub fn build(self) -> Mlin {
        // match self.z0 {
        //     Some(z0_tgt) => {
        //         let lb: Array1<f64> = array![1e-6];
        //         let ub: Array1<f64> = array![10e-6];
        //         let vals: Array1<f64> = array![(ub[0] + lb[0]) / 2.0];
        //         let scale: Array1<f64> = array![1.0 / self.scale.multiplier()];
        //         let mlin = Mlin {
        //             id: self.id.clone(),
        //             width: UnitValBuilder::new()
        //                 .val(vals[0])
        //                 .scale(self.scale)
        //                 .unit(self.unit)
        //                 .build(),
        //             length: self.length,
        //             sub: self.sub.clone(),
        //             nodes: self.nodes,
        //         };

        //         fn eval_f_mlin(
        //             vals: Array1<MyFloat>,
        //             mut mlin: Mlin,
        //             z0_tgt: MyFloat,
        //             freq: &Frequency,
        //         ) -> MyFloat {
        //             mlin.set_width_val(vals[0].to_f64());
        //             let z0 = MyFloat::new(mlin.z0(freq));
        //             println!(
        //                 "\n\nz0_tgt = {:?}\nz0 = {:?}\nerr = {:?}\nmlin = {:?}\n\n",
        //                 z0_tgt.to_f64(),
        //                 z0.to_f64(),
        //                 ((&z0 - &z0_tgt) / &z0_tgt).to_f64(),
        //                 mlin
        //             );

        //             (&z0 - &z0_tgt) / &z0_tgt
        //         }

        //         let mut test = NelderMeadBounded::new(
        //             vals.clone(),
        //             scale,
        //             lb,
        //             ub,
        //             move |x: Array1<MyFloat>| {
        //                 eval_f_mlin(x, mlin.clone(), MyFloat::new(z0_tgt), &self.freq)
        //             },
        //         );
        //         test.set_mu(5.0);
        //         test.set_target_tolerance(0.0);
        //         test.set_verbosity(10);
        //         test.solve(10000);
        //         println!(
        //             "\n\nx = {:?}\ntol = {:?}\niters = {:?}\n\n",
        //             test.x(),
        //             test.tolerance(),
        //             test.iterations()
        //         );

        //         Mlin {
        //             id: self.id,
        //             width: UnitValBuilder::new()
        //                 .val(test.x()[0])
        //                 .scale(self.scale)
        //                 .unit(self.unit)
        //                 .build(),
        //             length: self.length,
        //             sub: self.sub,
        //             nodes: self.nodes,
        //         }
        //     }
        //     None => Mlin {
        //         id: self.id,
        //         width: self.width,
        //         length: self.length,
        //         sub: self.sub,
        //         nodes: self.nodes,
        //     },
        // }
        Mlin {
            id: self.id,
            width: self.width,
            length: self.length,
            sub: self.sub,
            nodes: self.nodes,
        }
    }
}

impl Default for MlinBuilder {
    fn default() -> Self {
        Self {
            id: "ML0".to_string(),
            width: UnitVal::default(),
            length: UnitVal::default(),
            scale: Scale::Micro,
            unit: Unit::Meter,
            sub: Msub::default(),
            nodes: [0, 0],
            z0: None,
            freq: Frequency::default(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::elements::msub::MsubBuilder;
    use crate::unit::UnitValBuilder;
    use crate::util::{comp_c64, comp_f64, comp_point_c64};
    use float_cmp::F64Margin;

    #[test]
    fn element_mlin1() {
        let freq_unitval = UnitValBuilder::new()
            .val_scaled(1.0, Scale::Giga)
            .unit(Unit::Hz)
            .build();
        let freq = Frequency::from_unitval(&freq_unitval);
        let sub = MsubBuilder::new()
            .id("Msub0")
            .er(12.4)
            .tand(0.0004)
            .height(
                UnitValBuilder::new()
                    .val(25e-6)
                    .scale(Scale::Micro)
                    .unit(Unit::Meter)
                    .build(),
            )
            .thickness(
                UnitValBuilder::new()
                    .val(0.77e-6)
                    .scale(Scale::Micro)
                    .unit(Unit::Meter)
                    .build(),
            )
            .build();

        let width_val = 5.6758e-6;
        let length_val = 0.25;
        let exemplar = Mlin {
            id: "ML1".to_string(),
            width: UnitValBuilder::new()
                .val(width_val)
                .scale(Scale::Micro)
                .unit(Unit::Meter)
                .build(),
            length: UnitValBuilder::new()
                .val(length_val)
                .unit(Unit::Lambda)
                .build(),
            sub: sub.clone(),
            nodes: [1, 2],
        };
        let exemplar_z = c64(2_f64.sqrt() * 50.0, 0.0);
        let calc = MlinBuilder::new()
            .width_val(width_val)
            .width_scale(Scale::Micro)
            .width_unit(Unit::Meter)
            .length_val(length_val)
            .length_unit(Unit::Lambda)
            .sub(&sub)
            .nodes([1, 2])
            .id("ML1")
            .build();
        let margin = F64Margin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.scale(), &calc.scale());
        assert_eq!(&exemplar.unit(), &calc.unit());
        assert_eq!(&exemplar.gamma(&freq), &calc.gamma(&freq));
        assert_eq!(&exemplar.er(&freq), &calc.er(&freq));
        comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");

        let margin = F64Margin {
            epsilon: 1e-4,
            ulps: 10,
        };
        comp_c64(&exemplar_z, &calc.z(&freq), margin, "calc.z()", "0");
        comp_f64(&exemplar_z.re, &calc.z0(&freq), margin, "calc.z0()", "0");
    }

    #[test]
    fn element_mlin2() {
        let freq_unitval = UnitValBuilder::new()
            .val_scaled(1.0, Scale::Giga)
            .unit(Unit::Hz)
            .build();
        let freq = Frequency::from_unitval(&freq_unitval);
        let sub = MsubBuilder::new()
            .id("Msub0")
            .er(12.4)
            .tand(0.0004)
            .height(
                UnitValBuilder::new()
                    .val(25e-6)
                    .scale(Scale::Micro)
                    .unit(Unit::Meter)
                    .build(),
            )
            .thickness(
                UnitValBuilder::new()
                    .val(0.77e-6)
                    .scale(Scale::Micro)
                    .unit(Unit::Meter)
                    .build(),
            )
            .build();

        let z0 = 2_f64.sqrt() * 50.0;
        // let width_val = 5.6758e-6;
        let length_val = 0.25;
        // let exemplar = Mlin {
        //     id: "ML1".to_string(),
        //     width: UnitValBuilder::new()
        //         .val(width_val)
        //         .scale(Scale::Micro)
        //         .unit(Unit::Meter)
        //         .build(),
        //     length: UnitValBuilder::new()
        //         .val(length_val)
        //         .unit(Unit::Lambda)
        //         .build(),
        //     sub: sub.clone(),
        //     nodes: [1, 2],
        // };
        let exemplar_z = c64(z0, 0.0);
        let calc = MlinBuilder::new()
            .z0(z0, &freq)
            .length_val(length_val)
            .length_unit(Unit::Lambda)
            .sub(&sub)
            .nodes([1, 2])
            .id("ML1")
            .build();
        // let margin = F64Margin::default();

        // assert_eq!(&exemplar.id(), &calc.id());
        // assert_eq!(&exemplar.scale(), &calc.scale());
        // assert_eq!(&exemplar.unit(), &calc.unit());
        // assert_eq!(&exemplar.gamma(&freq), &calc.gamma(&freq));
        // assert_eq!(&exemplar.er(&freq), &calc.er(&freq));
        // comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc2.c()");

        let margin = F64Margin {
            epsilon: 1e-14,
            ulps: 10,
        };
        comp_c64(&exemplar_z, &calc.z(&freq), margin, "calc2.z()", "0");
        comp_f64(&exemplar_z.re, &calc.z0(&freq), margin, "calc2.z0()", "0");
    }
}
