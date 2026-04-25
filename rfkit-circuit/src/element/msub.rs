#![allow(unused)]
use crate::element::{Distributed, Elem, ElemType, Lumped};
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use num_traits::{ConstOne, ConstZero};
use rfkit_base::prelude::*;
use serde::Serialize;
// use serde_json::error::Category;
use std::{f64::consts::PI, fmt};

#[derive(Clone, Debug, PartialEq)]
pub struct Msub<T: RealScalar> {
    id: String,
    er: T,
    tand: T,
    height: ScalarUnitValue<T>,
    thickness: ScalarUnitValue<T>,
    conductivity: ScalarUnitValue<T>,
    rough: T,
}

impl<T: RealScalar> Msub<T> {
    pub fn new(
        id: String,
        er: T,
        tand: T,
        height: &ScalarUnitValue<T>,
        thickness: &ScalarUnitValue<T>,
        conductivity: &ScalarUnitValue<T>,
        rough: T,
    ) -> Msub<T> {
        Msub {
            id,
            er,
            tand,
            height: height.clone(),
            thickness: thickness.clone(),
            conductivity: conductivity.clone(),
            rough,
        }
    }

    fn builder() -> MsubBuilder<T> {
        MsubBuilder::new()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn er(&self) -> T {
        self.er
    }

    pub fn tand(&self) -> T {
        self.tand.tan()
    }

    pub fn height(&self) -> T {
        self.height.val()
    }

    pub fn height_uv(&self) -> ScalarUnitValue<T> {
        self.height
    }

    pub fn thickness(&self) -> T {
        self.thickness.val()
    }

    pub fn thickness_uv(&self) -> ScalarUnitValue<T> {
        self.thickness
    }

    pub fn conductivity(&self) -> T {
        self.conductivity.val()
    }

    pub fn conductivity_uv(&self) -> ScalarUnitValue<T> {
        self.conductivity
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    /// Normalized frequency
    pub fn f_norm<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| f.val() * self.height.val() / 3e8)
    }

    /// Skin depth of conductor δ
    pub fn delta<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| {
            (f.freq() * std::f64::consts::PI * std::f64::consts::PI * 4e-7 * self.sigma(f))
                .sqrt()
                .recip()
        })
    }

    /// Quasi-static skin depth of conductor δ
    pub fn delta_qs<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| {
            (T::from_f64(2.0) / self.conductivity.val() / (f.w() * std::f64::consts::PI * 4e-7))
                .sqrt()
        })
    }

    pub fn max_sh_cond<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| self.conductivity.val() * self.delta(f))
    }

    pub fn min_sh_res<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| self.max_sh_cond(f).recip())
    }

    /// Sheet conductance of the conductor
    pub fn cond_sh<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| {
            self.conductivity.val() * (self.thickness.val() / self.delta(f))
        })
    }

    /// Sheet resistance of the conductor
    pub fn res_sh<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| self.cond_sh(f).recip())
    }

    /// Frequency dependent conductivity σ(f)
    pub fn sigma<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| {
            self.conductivity.val() * (f.freq().scale(Scale::Giga) * 0.045 + 1.0).sqrt()
        })
    }

    /// Loss factor due to surface roughness
    pub fn roughness<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|f| {
            ((self.rough / self.delta(f)) * 1.4).atan() * 2.0 / std::f64::consts::PI + 1.0
        })
    }
}

impl<T: RealScalar> Default for Msub<T> {
    fn default() -> Self {
        Self {
            id: "Msub0".to_string(),
            er: T::ONE,
            tand: T::ZERO,
            height: UnitValueBuilder::new()
                .val_scaled(&T::from_f64(100.0), Scale::Micro)
                .unit(Unit::Meter)
                .build()
                .unwrap(),
            thickness: UnitValueBuilder::new()
                .val_scaled(&T::ONE, Scale::Micro)
                .unit(Unit::Meter)
                .build()
                .unwrap(),
            conductivity: UnitValueBuilder::new()
                .val_scaled(&T::from_f64(35.0), Scale::Mega)
                .unit(Unit::Sieman)
                .build()
                .unwrap(),
            rough: T::ZERO,
        }
    }
}

impl<T: RealScalar> Elem<T> for Msub<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Msub
    }

    fn nodes(&self) -> Vec<usize> {
        vec![]
    }

    fn c<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points::zeros((freq.npts(), 0, 0))
    }

    fn net<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points::zeros((freq.npts(), 2, 2).into_dimension())
    }

    fn z_scalar(&self, _freq: &ScalarUnitValue<T>) -> Complex<T> {
        Complex::ZERO
    }
}

impl<T: RealScalar> fmt::Display for Msub<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Msub\n\tid:\t{}\n\ter:\t{}\n\ttand:\t{}\n\theight:\t{}\n\tthickness:\t{}\n\tconductivity:\t{}\n\troughness:\t{}\n",
            self.id, self.er, self.tand, self.height, self.thickness, self.conductivity, self.rough
        )
    }
}

#[derive(Clone)]
pub struct MsubBuilder<T: RealScalar> {
    id: String,
    er: Option<T>,
    tand: Option<T>,
    height: Option<ScalarUnitValue<T>>,
    thickness: Option<ScalarUnitValue<T>>,
    conductivity: ScalarUnitValue<T>,
    rough: T,
}

impl<T: RealScalar> MsubBuilder<T> {
    pub fn new() -> Self {
        MsubBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn er(mut self, val: T) -> Self {
        self.er = Some(val);
        self
    }

    pub fn tand(mut self, val: T) -> Self {
        self.tand = Some(val);
        self
    }

    pub fn height(mut self, val: &ScalarUnitValue<T>) -> Self {
        self.height = Some(val.clone());
        self
    }

    pub fn height_val(mut self, val: T) -> Self {
        self.height = match self.height {
            Some(mut x) => {
                x.set_val(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&val, Scale::Base, Unit::Farad)),
        };
        self
    }

    pub fn height_val_scaled(mut self, val: T, scale: Scale) -> Self {
        self.height = match self.height {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&val, scale, Unit::Farad)),
        };
        self
    }

    pub fn thickness(mut self, val: &ScalarUnitValue<T>) -> Self {
        self.thickness = Some(val.clone());
        self
    }

    pub fn thickness_val(mut self, val: T) -> Self {
        self.thickness = match self.thickness {
            Some(mut x) => {
                x.set_val(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&val, Scale::Base, Unit::Farad)),
        };
        self
    }

    pub fn thickness_val_scaled(mut self, val: T, scale: Scale) -> Self {
        self.thickness = match self.thickness {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&val, scale, Unit::Farad)),
        };
        self
    }

    pub fn conductivity(mut self, val: &ScalarUnitValue<T>) -> Self {
        self.conductivity = val.clone();
        self
    }

    pub fn conductivity_val(mut self, val: T) -> Self {
        self.conductivity.set_val(&val);
        self
    }

    pub fn conductivity_val_scaled(mut self, val: T, scale: Scale) -> Self {
        self.conductivity.set_scale(scale);
        self.conductivity.set_val_scaled(&val);
        self
    }

    /// RMS surface roughness
    pub fn rough(mut self, val: T) -> Self {
        self.rough = val;
        self
    }

    pub fn build(self) -> Result<Msub<T>, String> {
        let er = self.er.ok_or("MsubBuilder: er is required")?;
        let tand = self.tand.ok_or("MsubBuilder: tand is required")?;
        let height = self.height.ok_or("MsubBuilder: height is required")?;
        let thickness = self.thickness.ok_or("MsubBuilder: thickness is required")?;
        Ok(Msub {
            id: self.id,
            er,
            tand,
            height,
            thickness,
            conductivity: self.conductivity,
            rough: self.rough,
        })
    }
}

impl<T: RealScalar> Default for MsubBuilder<T> {
    fn default() -> Self {
        Self {
            id: "Msub0".to_string(),
            er: None,
            tand: None,
            height: None,
            thickness: None,
            conductivity: ScalarUnitValue::new(&1e50.into(), Scale::Base, Unit::Sieman),
            rough: T::ZERO,
        }
    }
}

#[cfg(test)]
mod element_msub_tests {
    use super::*;
    use num_complex::Complex64;

    const DEFAULT_MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-10,
        relative: 1e-10,
        ulps: 4,
    };

    const RELAXED_MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-6,
        relative: 1e-6,
        ulps: 10,
    };

    #[test]
    fn element_msub() {
        let er: f64 = 12.4;
        let tand: f64 = 0.0004;
        let height_val = 25e-6;
        let thickness_val = 0.77e-6;
        let height = UnitValueBuilder::new()
            .val(&height_val)
            .unit(Unit::Meter)
            .build()
            .unwrap();
        let thickness = UnitValueBuilder::new()
            .val(&thickness_val)
            .unit(Unit::Meter)
            .build()
            .unwrap();
        let conductivity = UnitValueBuilder::new()
            .val(&3.5e7)
            .unit(Unit::Sieman)
            .build()
            .unwrap();
        let rough = 0.0;
        let exemplar: Msub<f64> = Msub {
            id: "Msub1".to_string(),
            er,
            tand,
            height,
            thickness,
            conductivity,
            rough,
        };
        let calc: Msub<f64> = MsubBuilder::new()
            .id("Msub1")
            .er(er)
            .tand(tand)
            .height(&height)
            .thickness(&thickness)
            .conductivity(&conductivity)
            .build()
            .unwrap();
        let margin = NumMargin::<f64>::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.er(), &calc.er());
        assert_eq!(&exemplar.tand(), &calc.tand());
        assert_eq!(&exemplar.height(), &calc.height());
        assert_eq!(&exemplar.thickness(), &calc.thickness());
        assert_eq!(&exemplar.conductivity(), &calc.conductivity());
    }

    mod msub_tests {
        use super::*;

        #[test]
        fn test_msub_builder_default() {
            let msub: Msub<f64> = MsubBuilder::new()
                .er(1.0)
                .tand(0.0)
                .height(&ScalarUnitValue::new(&1.0, Scale::Base, Unit::Meter))
                .thickness(&ScalarUnitValue::new(&1.0, Scale::Base, Unit::Meter))
                .build()
                .unwrap();
            assert_eq!(msub.id(), "Msub0");
            assert_eq!(msub.er(), 1.0);
            assert_eq!(msub.tand(), 0.0);
        }

        #[test]
        fn test_msub_with_parameters() {
            let height = UnitValueBuilder::new()
                .val_scaled(&25.0, Scale::Micro)
                .unit(Unit::Meter)
                .build()
                .unwrap();

            let thickness = UnitValueBuilder::new()
                .val_scaled(&0.77, Scale::Micro)
                .unit(Unit::Meter)
                .build()
                .unwrap();

            let conductivity = UnitValueBuilder::new()
                .val_scaled(&35.0, Scale::Mega)
                .unit(Unit::Sieman)
                .build()
                .unwrap();

            let msub: Msub<f64> = MsubBuilder::new()
                .id("Msub1")
                .er(12.4)
                .tand(0.0004)
                .height(&height)
                .thickness(&thickness)
                .conductivity(&conductivity)
                .build()
                .unwrap();

            assert_eq!(msub.id(), "Msub1");
            assert_eq!(msub.er(), 12.4);
            msub.tand().assert_approx_eq(
                &0.0004,
                RELAXED_MARGIN,
                "msub_with_parameters",
                "msub.tand()",
            );
        }

        #[test]
        fn test_msub_skin_depth() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let msub: Msub<f64> = MsubBuilder::new()
                .er(1.0)
                .tand(0.0)
                .height_val(1.0)
                .thickness_val(1.0)
                .conductivity_val_scaled(35.0, Scale::Mega)
                .build()
                .unwrap();

            let delta = msub.delta(&freq);
            assert!(delta[0] > 0.0);
            assert!(delta[0] < 1e-5); // Should be in micrometers range
        }

        #[test]
        fn test_msub_roughness_factor() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let msub: Msub<f64> = MsubBuilder::new()
                .er(1.0)
                .tand(0.0)
                .height_val(1.0)
                .thickness_val(1.0)
                .conductivity_val_scaled(35.0, Scale::Mega)
                .rough(1e-7)
                .build()
                .unwrap();

            let roughness = msub.roughness(&freq);
            assert!(roughness[0] >= 1.0); // Should be >= 1
        }

        #[test]
        fn test_msub_sheet_resistance() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let msub: Msub<f64> = MsubBuilder::new()
                .er(1.0)
                .tand(0.0)
                .height_val(1.0)
                .conductivity_val_scaled(35.0, Scale::Mega)
                .thickness_val_scaled(0.77, Scale::Micro)
                .build()
                .unwrap();

            let res_sh = msub.res_sh(&freq);
            assert!(res_sh[0] > 0.0);
        }

        #[test]
        fn test_msub_elem_type() {
            let msub: Msub<f64> = MsubBuilder::new()
                .er(1.0)
                .tand(0.0)
                .height_val(1.0)
                .thickness_val(1.0)
                .build()
                .unwrap();
            assert_eq!(msub.elem(), ElemType::Msub);
        }

        #[test]
        fn test_msub_zero_impedance() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let msub: Msub<f64> = MsubBuilder::new()
                .er(1.0)
                .tand(0.0)
                .height_val(1.0)
                .thickness_val(1.0)
                .build()
                .unwrap();
            msub.z(&freq).assert_approx_eq(
                &array![Complex64::ZERO],
                NumMargin::default(),
                "msub_zero_impedance",
                "",
            );
        }
    }
}
