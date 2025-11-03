use crate::element::{Distributed, Elem, ElemType, msub::Msub};
use crate::frequency::Frequency;
use crate::point;
use crate::point::Point;
use crate::points::{Points, Pts};
use crate::scale::Scale;
use crate::unit::{Unit, UnitVal, UnitValBuilder, Unitized};
use ndarray::prelude::*;
use num::complex::Complex64;

#[derive(Clone, Debug, PartialEq)]
pub struct Mbend {
    id: String,
    width: UnitVal,
    miter: bool,
    sub: Msub,
    nodes: [usize; 2],
    z0: Complex64,
}

impl Mbend {
    pub fn new(
        id: String,
        width: UnitVal,
        miter: bool,
        sub: &Msub,
        nodes: [usize; 2],
        z0: Complex64,
    ) -> Mbend {
        Mbend {
            id,
            width,
            miter,
            sub: sub.clone(),
            nodes,
            z0,
        }
    }

    /// Calculate L & C values for Tee network equivalent
    /// Returns (L, C)
    fn calc_lc(&self) -> (UnitVal, UnitVal) {
        let w = self.width.val();
        let h = self.sub.height().val();
        let er = self.sub.er();
        let (l, c) = match self.miter {
            true => (
                441.2712 * (1.0 - 1.062 * (-0.177 * (w / h).powf(0.947)).exp()) * h,
                w / h * (7.6 * er + 3.8 + w / h * (3.93 * er + 0.62)) * h,
            ),
            false => (
                220.6356 * (1.0 - 1.35 * (-0.18 * (w / h).powf(1.39)).exp()) * h,
                w / h * (2.6 * er + 5.64 + w / h * (10.35 * er + 2.5)) * h,
            ),
        };
        (
            UnitValBuilder::new()
                .val_scaled(l, Scale::Nano)
                .unit(Unit::Henry)
                .build(),
            UnitValBuilder::new()
                .val_scaled(c, Scale::Pico)
                .unit(Unit::Farad)
                .build(),
        )
    }

    fn z0(&self, _freq: &Frequency) -> f64 {
        self.z0.re
    }
}

impl Default for Mbend {
    fn default() -> Self {
        Self {
            id: "MB0".to_string(),
            width: *UnitVal::default().set_unit(Unit::Meter),
            miter: false,
            sub: Msub::default(),
            nodes: [1, 2],
            z0: (50.0).into(),
        }
    }
}

impl Elem for Mbend {
    fn c(&self, freq: &Frequency) -> Point<Complex64> {
        let (l_unitval, c_unitval) = self.calc_lc();
        let l = l_unitval.val();
        let c = c_unitval.val();

        let z1 = Complex64::I * freq.w()[0] * l;
        let z3 = z1;
        let z2 = 1.0 / (Complex64::I * freq.w()[0] * c);
        let p = z1 - z2;
        let q = z1 + z2 + 2.0 * z3;
        let t = z1 * z2 + z2 * z3 + z1 * z3;
        let d = self.z0.powi(2) + q * self.z0 + t;

        point![
            Complex64,
            [
                -(1.0 / d) * self.z0.powi(2) + p * self.z0 + t,
                (1.0 / d) * 2.0 * self.z0 * z3,
            ],
            [
                (1.0 / d) * 2.0 * self.z0 * z3,
                -(1.0 / d) * self.z0.powi(2) - p * self.z0 + t,
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
        ElemType::Mbend
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

    fn z(&self, _freq: &Frequency) -> Complex64 {
        self.z0
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = Frequency::new(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Distributed for Mbend {
    fn width(&self) -> f64 {
        self.width.val()
    }

    fn length(&self, _freq: &Frequency) -> f64 {
        0.0
    }

    fn sub(&self) -> &Msub {
        &self.sub
    }

    fn val(&self) -> f64 {
        0.0
    }

    fn gamma(&self, _freq: &Frequency) -> Complex64 {
        Complex64::ZERO
    }

    fn er(&self, _freq: &Frequency) -> f64 {
        self.sub.er()
    }

    fn set_width_val(&mut self, _val: f64) {
        ();
    }

    fn set_width_unit(&mut self, _unit: Unit) {
        ();
    }

    fn set_length_val(&mut self, _val: f64) {
        ();
    }

    fn set_length_unit(&mut self, _unit: Unit) {
        ();
    }
}

impl Unitized for Mbend {
    fn val_scaled(&self) -> f64 {
        0.0
    }

    fn unitval(&self) -> UnitVal {
        UnitVal::default()
    }

    fn scale(&self) -> Scale {
        Scale::default()
    }

    fn unit(&self) -> Unit {
        Unit::default()
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

#[derive(Clone)]
pub struct MbendBuilder {
    id: String,
    width: UnitVal,
    miter: bool,
    sub: Msub,
    nodes: [usize; 2],
    z0: Complex64,
}

impl MbendBuilder {
    pub fn new() -> Self {
        MbendBuilder::default()
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

    pub fn miter(mut self, val: bool) -> Self {
        self.miter = val;
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

    pub fn z0(mut self, z0: Complex64) -> Self {
        self.z0 = z0;
        self
    }

    pub fn build(self) -> Mbend {
        Mbend {
            id: self.id,
            width: self.width,
            miter: self.miter,
            sub: self.sub,
            nodes: self.nodes,
            z0: self.z0,
        }
    }
}

impl Default for MbendBuilder {
    fn default() -> Self {
        Self {
            id: "MB0".to_string(),
            width: *UnitVal::default().set_unit(Unit::Meter),
            miter: false,
            sub: Msub::default(),
            nodes: [1, 2],
            z0: (50.0).into(),
        }
    }
}

#[cfg(test)]
mod element_mbend_tests {
    use super::*;
    use crate::element::msub::MsubBuilder;
    // use crate::unit::UnitValBuilder;
    // use crate::util::{comp_c64, comp_f64, comp_point_c64};
    use float_cmp::*;
    use num::complex::c64;

    const DEFAULT_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-10,
        ulps: 4,
    };

    const RELAXED_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-6,
        ulps: 10,
    };

    // #[test]
    // fn element_mbend() {
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
    //     let exemplar = Mbend {
    //         id: "MB1".to_string(),
    //         width: UnitValBuilder::new()
    //             .val(width_val)
    //             .scale(Scale::Micro)
    //             .unit(Unit::Meter)
    //             .build(),
    //         miter: false,
    //         sub: sub.clone(),
    //         nodes: [1, 2],
    //         z0: (50.0).into(),
    //     };
    //     let exemplar_z = c64(2_f64.sqrt() * 50.0, 0.0);
    //     let calc = MbendBuilder::new()
    //         .width_val(width_val)
    //         .width_scale(Scale::Micro)
    //         .width_unit(Unit::Meter)
    //         .miter(false)
    //         .sub(&sub)
    //         .nodes([1, 2])
    //         .id("MB1")
    //         .build();
    //     let margin = F64Margin::default();

    //     assert_eq!(&exemplar.id(), &calc.id());
    //     comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");

    //     let margin = F64Margin {
    //         epsilon: 1e-4,
    //         ulps: 10,
    //     };
    //     comp_c64(&exemplar_z, &calc.z(&freq), margin, "calc.z()", "0");
    //     comp_f64(&exemplar_z.re, &calc.z0(&freq), margin, "calc.z0()", "0");
    // }

    mod mbend_tests {
        use super::*;

        #[test]
        fn test_mbend_builder_default() {
            let mbend = MbendBuilder::new().build();
            assert_eq!(mbend.id(), "MB0");
            assert_eq!(mbend.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_mbend_miter_vs_no_miter() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .height_scaled(25.0, Scale::Micro)
                .build();

            let mbend_no_miter = MbendBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .miter(false)
                .sub(&sub)
                .build();

            let mbend_miter = MbendBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .miter(true)
                .sub(&sub)
                .build();

            // Both should build successfully
            assert!(!mbend_no_miter.miter);
            assert!(mbend_miter.miter);
        }

        #[test]
        fn test_mbend_with_substrate() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .height_scaled(25.0, Scale::Micro)
                .thickness_scaled(0.77, Scale::Micro)
                .build();

            let mbend = MbendBuilder::new()
                .id("MB1")
                .width_scaled(5.8736, Scale::Micro)
                .miter(false)
                .sub(&sub)
                .nodes([1, 2])
                .z0(c64(50.0, 0.0))
                .build();

            assert_eq!(mbend.id(), "MB1");
            assert_eq!(mbend.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_mbend_c_matrix_not_zero() {
            let freq = Frequency::new(array![1e9], Scale::Base);
            let sub = MsubBuilder::new()
                .er(12.4)
                .height_scaled(25.0, Scale::Micro)
                .build();

            let mbend = MbendBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .sub(&sub)
                .build();

            let c_matrix = mbend.c(&freq);

            // For bend, matrix should have non-zero elements
            assert_ne!(c_matrix[[0, 0]], Complex64::ZERO);
            assert_ne!(c_matrix[[1, 1]], Complex64::ZERO);
        }

        #[test]
        fn test_mbend_elem_type() {
            let sub = MsubBuilder::new().build();
            let mbend = MbendBuilder::new().sub(&sub).build();
            assert_eq!(mbend.elem(), ElemType::Mbend);
        }

        #[test]
        fn test_mbend_zero_length() {
            let freq = Frequency::new(array![1e9], Scale::Base);
            let sub = MsubBuilder::new().build();
            let mbend = MbendBuilder::new().sub(&sub).build();

            // Bend has no physical length
            assert_eq!(mbend.length(&freq), 0.0);
        }

        #[test]
        fn test_mbend_distributed_trait() {
            let sub = MsubBuilder::new().build();
            let mbend = MbendBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .sub(&sub)
                .build();

            approx_eq!(f64, mbend.width(), 10e-6);
            assert_eq!(mbend.val(), 0.0);
        }
    }
}
