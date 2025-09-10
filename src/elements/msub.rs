#![allow(unused)]
use crate::elements::{Distributed, Elem, ElemType, Lumped};
use crate::frequency::Frequency;
use crate::point;
use crate::point::{Point, Pt};
use crate::points::{Points, Pts};
use crate::scale::Scale;
use crate::unit::{Unit, UnitVal, UnitValBuilder, Unitized};
use float_cmp::{F64Margin, approx_eq};
use ndarray::prelude::*;
use num::complex::{Complex, Complex64, c64};
use num_traits::ConstZero;
use serde::Serialize;
use serde_json::error::Category;
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct Msub {
    id: String,
    er: f64,
    tand: f64,
    height: UnitVal,
    thickness: UnitVal,
    conductivity: UnitVal,
    rough: f64,
}

impl Msub {
    pub fn new(
        id: String,
        er: f64,
        tand: f64,
        height: UnitVal,
        thickness: UnitVal,
        conductivity: UnitVal,
        rough: f64,
    ) -> Msub {
        Msub {
            id,
            er,
            tand,
            height,
            thickness,
            conductivity,
            rough,
        }
    }

    pub fn er(&self) -> f64 {
        self.er
    }

    pub fn tand(&self) -> f64 {
        self.tand.tan()
    }

    pub fn height(&self) -> UnitVal {
        self.height
    }

    pub fn thickness(&self) -> UnitVal {
        self.thickness
    }

    pub fn conductivity(&self) -> UnitVal {
        self.conductivity
    }

    /// Skin depth of conductor δ
    pub fn delta(&self, freq: &Frequency) -> f64 {
        1.0 / (PI * freq.freq(0) * PI * 4e-7 * self.sigma(freq)).sqrt()
    }

    /// Quasi-static skin depth of conductor δ
    pub fn delta_qs(&self, freq: &Frequency) -> f64 {
        (2.0 / self.conductivity.val() / (2.0 * PI * freq.freq(0) * PI * 4e-7)).sqrt()
    }

    pub fn max_sh_cond(&self, freq: &Frequency) -> f64 {
        self.conductivity.val() * self.delta(freq)
    }

    pub fn min_sh_res(&self, freq: &Frequency) -> f64 {
        1.0 / self.max_sh_cond(freq)
    }

    /// Sheet conductance of the conductor
    pub fn cond_sh(&self, freq: &Frequency) -> f64 {
        self.conductivity.val() * (-self.thickness.val() / self.delta(freq))
    }

    /// Sheet resistance of the conductor
    pub fn res_sh(&self, freq: &Frequency) -> f64 {
        1.0 / self.cond_sh(freq)
    }

    /// Frequency dependent conductivity σ(f)
    pub fn sigma(&self, freq: &Frequency) -> f64 {
        self.conductivity.val() * (1.0 + 0.045 * Scale::Giga.scale(freq.freq(0))).sqrt()
    }

    /// Loss factor due to surface roughness
    pub fn roughness(&self, freq: &Frequency) -> f64 {
        1.0 + 2.0 / PI * (1.4 * (self.rough / self.delta(freq))).atan()
    }
}

impl Default for Msub {
    fn default() -> Self {
        Self {
            id: "Msub0".to_string(),
            er: 1.0,
            tand: 0.0,
            height: UnitValBuilder::new()
                .val_scaled(100.0, Scale::Micro)
                .unit(Unit::Meter)
                .build(),
            thickness: UnitValBuilder::new()
                .val_scaled(100.0, Scale::Micro)
                .unit(Unit::Meter)
                .build(),
            conductivity: UnitValBuilder::new()
                .val_scaled(35.0, Scale::Mega)
                .unit(Unit::Sieman)
                .build(),
            rough: 0.0,
        }
    }
}

impl Elem for Msub {
    fn c(&self, _freq: &Frequency) -> Point {
        Point::zeros((0, 0))
    }

    fn c_at(&self, _freq: &Frequency, j: usize, k: usize) -> Complex64 {
        Complex64::ZERO
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Msub
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points {
        Points::zeros((freq.npts(), 2, 2))
    }

    fn nodes(&self) -> Vec<usize> {
        vec![]
    }

    fn z(&self, freq: &Frequency) -> Complex64 {
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

pub struct MsubBuilder {
    id: String,
    er: f64,
    tand: f64,
    height: UnitVal,
    thickness: UnitVal,
    conductivity: UnitVal,
    rough: f64,
}

impl MsubBuilder {
    pub fn new() -> Self {
        MsubBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn er(mut self, val: f64) -> Self {
        self.er = val;
        self
    }

    pub fn tand(mut self, val: f64) -> Self {
        self.tand = val;
        self
    }

    pub fn height(mut self, val: UnitVal) -> Self {
        self.height = val;
        self
    }

    pub fn height_val(mut self, val: f64) -> Self {
        self.height.set_val(val);
        self
    }

    pub fn height_scaled(mut self, val: f64, scale: Scale) -> Self {
        self.height.set_scale(scale);
        self.height.set_val_scaled(val);
        self
    }

    pub fn thickness(mut self, val: UnitVal) -> Self {
        self.thickness = val;
        self
    }

    pub fn thickness_val(mut self, val: f64) -> Self {
        self.thickness.set_val(val);
        self
    }

    pub fn thickness_scaled(mut self, val: f64, scale: Scale) -> Self {
        self.thickness.set_scale(scale);
        self.thickness.set_val_scaled(val);
        self
    }

    pub fn conductivity(mut self, val: UnitVal) -> Self {
        self.conductivity = val;
        self
    }

    pub fn conductivity_val(mut self, val: f64) -> Self {
        self.conductivity.set_val(val);
        self
    }

    pub fn conductivity_scaled(mut self, val: f64, scale: Scale) -> Self {
        self.conductivity.set_scale(scale);
        self.conductivity.set_val_scaled(val);
        self
    }

    /// RMS surface roughness
    pub fn rough(mut self, val: f64) -> Self {
        self.rough = val;
        self
    }

    pub fn build(self) -> Msub {
        Msub {
            id: self.id,
            er: self.er,
            tand: self.tand,
            height: self.height,
            thickness: self.thickness,
            conductivity: self.conductivity,
            rough: self.rough,
        }
    }
}

impl Default for MsubBuilder {
    fn default() -> Self {
        Self {
            id: "Msub0".to_string(),
            er: 1.0,
            tand: 0.0,
            height: UnitVal::default(),
            thickness: UnitVal::default(),
            conductivity: UnitValBuilder::new()
                .val_scaled(1e50, Scale::Base)
                .unit(Unit::Sieman)
                .build(),
            rough: 0.0,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frequency::FrequencyBuilder;
    use crate::unit::UnitValBuilder;
    use crate::util::{comp_c64, comp_point_c64, comp_points_c64, comp_vec_c64};
    use float_cmp::F64Margin;

    #[test]
    fn element_msub() {
        let er: f64 = 12.4;
        let tand: f64 = 0.0004;
        let height_val = 25e-6;
        let thickness_val = 0.77e-6;
        let height = UnitValBuilder::new()
            .val(height_val)
            .unit(Unit::Meter)
            .build();
        let thickness = UnitValBuilder::new()
            .val(thickness_val)
            .unit(Unit::Meter)
            .build();
        let conductivity = UnitValBuilder::new().val(3.5e7).unit(Unit::Sieman).build();
        let rough = 0.0;
        let exemplar = Msub {
            id: "Msub1".to_string(),
            er,
            tand,
            height,
            thickness,
            conductivity,
            rough,
        };
        let calc = MsubBuilder::new()
            .id("Msub1")
            .er(er)
            .tand(tand)
            .height(height)
            .thickness(thickness)
            .conductivity(conductivity)
            .build();
        let margin = F64Margin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.er(), &calc.er());
        assert_eq!(&exemplar.tand(), &calc.tand());
        assert_eq!(&exemplar.height(), &calc.height());
        assert_eq!(&exemplar.thickness(), &calc.thickness());
        assert_eq!(&exemplar.conductivity(), &calc.conductivity());
    }
}
