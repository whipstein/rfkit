use crate::element::{Elem, ElemType, Lumped};
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
pub struct Capacitor {
    id: String,
    cap: UnitVal,
    nodes: [usize; 2],
    c: Point<Complex64>,
    z0: Complex64,
}

impl Capacitor {
    pub fn new(id: String, cap: UnitVal, nodes: [usize; 2], z0: Complex64) -> Capacitor {
        Capacitor {
            id: id,
            cap: cap,
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

impl Default for Capacitor {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            cap: UnitVal::default(),
            nodes: [1, 2],
            c: point![
                Complex64,
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        }
    }
}

impl Elem for Capacitor {
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
        ElemType::Capacitor
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

    fn z(&self, freq: &Frequency) -> Complex64 {
        let out = 1.0 / (Complex64::I * 2.0 * PI * freq.freq(0) * self.cap.val());
        out
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = Frequency::new(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Lumped for Capacitor {
    fn val(&self) -> f64 {
        self.cap.val()
    }

    fn set_val(&mut self, val: f64) {
        self.cap.set_val(val);
    }
}

impl Unitized for Capacitor {
    fn val_scaled(&self) -> f64 {
        self.cap.val_scaled()
    }

    fn unitval(&self) -> UnitVal {
        self.cap.clone()
    }

    fn scale(&self) -> Scale {
        self.cap.scale()
    }

    fn unit(&self) -> Unit {
        self.cap.unit()
    }

    fn set_val_scaled(&mut self, val: f64) {
        self.cap.set_val_scaled(val);
    }

    fn set_unitval(&mut self, val: UnitVal) {
        self.cap = val;
    }

    fn set_scale(&mut self, scale: Scale) {
        self.cap.set_scale(scale);
    }

    fn set_unit(&mut self, unit: Unit) {
        self.cap.set_unit(unit);
    }
}

#[derive(Clone)]
pub struct CapacitorBuilder {
    id: String,
    cap: UnitVal,
    nodes: [usize; 2],
    z0: Complex64,
}

impl CapacitorBuilder {
    pub fn new() -> Self {
        CapacitorBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn cap(mut self, cap: UnitVal) -> Self {
        self.cap = cap;
        self
    }

    pub fn val(mut self, cap: f64) -> Self {
        self.cap.set_val(cap);
        self
    }

    pub fn val_scaled(mut self, cap: f64, scale: Scale) -> Self {
        self.cap.set_scale(scale);
        self.cap.set_val_scaled(cap);
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

    pub fn build(self) -> Capacitor {
        Capacitor {
            id: self.id,
            cap: self.cap,
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

impl Default for CapacitorBuilder {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            cap: *UnitVal::default().set_unit(Unit::Farad),
            nodes: [1, 2],
            z0: c64(50.0, 0.0),
        }
    }
}

#[cfg(test)]
mod element_capacitor_tests {
    use super::*;
    use crate::unit::UnitValBuilder;
    use crate::util::{comp_c64, comp_point_c64};
    use float_cmp::*;

    const DEFAULT_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-10,
        ulps: 4,
    };

    const RELAXED_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-6,
        ulps: 10,
    };

    #[test]
    fn element_capacitor() {
        let freq_unitval = UnitValBuilder::new().val_scaled(1.0, Scale::Giga).build();
        let freq = Frequency::from_unitval(&freq_unitval);
        let val_scaled = 1.0;
        let scale = Scale::Pico;
        let unitval = UnitValBuilder::new()
            .val_scaled(val_scaled, scale)
            .unit(Unit::Farad)
            .build();
        let exemplar = Capacitor {
            id: "C1".to_string(),
            cap: unitval,
            nodes: [1, 2],
            c: point![
                Complex64,
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        };
        let exemplar_z = 1.0 / (Complex64::I * 2.0 * PI * freq.freq(0) * unitval.val());
        let calc = CapacitorBuilder::new()
            .val_scaled(val_scaled, scale)
            .nodes([1, 2])
            .id("C1")
            .build();
        let margin = F64Margin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.scale(), &calc.scale());
        assert_eq!(&Unit::Farad, &calc.unit());
        comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        comp_c64(&exemplar_z, &calc.z(&freq), margin, "calc.z()", "0");
    }

    mod capacitor_tests {
        use super::*;

        #[test]
        fn test_capacitor_builder_default() {
            let cap = CapacitorBuilder::new().build();
            assert_eq!(cap.id(), "C0");
            assert_eq!(cap.nodes(), vec![1, 2]);
            assert_eq!(cap.unit(), Unit::Farad);
        }

        #[test]
        fn test_capacitor_builder_with_all_parameters() {
            let cap = CapacitorBuilder::new()
                .id("C1")
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 3])
                .z0(c64(75.0, 0.0))
                .build();

            assert_eq!(cap.id(), "C1");
            assert_eq!(cap.nodes(), vec![1, 3]);
            assert_eq!(cap.val_scaled(), 10.0);
            assert_eq!(cap.scale(), Scale::Pico);
        }

        #[test]
        fn test_capacitor_impedance_calculation() {
            let freq = Frequency::new(array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            let z = cap.z(&freq);
            let expected_z = 1.0 / (Complex64::I * 2.0 * PI * 1e9 * 1e-12);

            assert!(approx_eq!(f64, z.re, expected_z.re, epsilon = 1e-6));
            assert!(approx_eq!(f64, z.im, expected_z.im, epsilon = 1e-6));
        }

        #[test]
        fn test_capacitor_c_matrix_structure() {
            let freq = Frequency::new(array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new().build();
            let c_matrix = cap.c(&freq);

            // Check diagonal and off-diagonal elements
            assert!(approx_eq!(
                f64,
                c_matrix[[0, 0]].re,
                1.0 / 3.0,
                DEFAULT_MARGIN
            ));
            assert!(approx_eq!(
                f64,
                c_matrix[[1, 1]].re,
                1.0 / 3.0,
                DEFAULT_MARGIN
            ));
            assert!(approx_eq!(
                f64,
                c_matrix[[0, 1]].re,
                2.0 / 3.0,
                DEFAULT_MARGIN
            ));
            assert!(approx_eq!(
                f64,
                c_matrix[[1, 0]].re,
                2.0 / 3.0,
                DEFAULT_MARGIN
            ));
        }

        #[test]
        fn test_capacitor_multiple_frequencies() {
            let freqs = array![1e9, 2e9, 5e9, 10e9];
            let freq = Frequency::new(freqs.clone(), Scale::Base);
            let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            for i in 0..freqs.len() {
                let z = cap.z_at(&freq, i);
                let expected = 1.0 / (Complex64::I * 2.0 * PI * freqs[i] * 1e-12);
                assert!(approx_eq!(f64, z.re, expected.re, RELAXED_MARGIN));
                assert!(approx_eq!(f64, z.im, expected.im, RELAXED_MARGIN));
            }
        }

        #[test]
        fn test_capacitor_value_scaling() {
            let cap = CapacitorBuilder::new()
                .val_scaled(10.0, Scale::Pico)
                .build();

            assert_eq!(cap.val_scaled(), 10.0);
            assert_eq!(cap.val(), 10e-12);
        }

        #[test]
        fn test_capacitor_set_operations() {
            let mut cap = CapacitorBuilder::new().build();

            cap.set_id("C_new");
            assert_eq!(cap.id(), "C_new");

            cap.set_val(5e-12);
            assert_eq!(cap.val(), 5e-12);

            cap.set_scale(Scale::Nano);
            assert_eq!(cap.scale(), Scale::Nano);
        }

        #[test]
        fn test_capacitor_elem_type() {
            let cap = CapacitorBuilder::new().build();
            assert_eq!(cap.elem(), ElemType::Capacitor);
        }

        #[test]
        fn test_capacitor_net_matrix() {
            let freq = Frequency::new(array![1e9, 2e9], Scale::Base);
            let cap = CapacitorBuilder::new().build();
            let net = cap.net(&freq);

            assert_eq!(net.shape(), (2, 2, 2)); // [npts, 2, 2]

            // Check structure is consistent across frequencies
            for i in 0..2 {
                assert!(approx_eq!(
                    f64,
                    net[[i, 0, 0]].re,
                    1.0 / 3.0,
                    DEFAULT_MARGIN
                ));
                assert!(approx_eq!(
                    f64,
                    net[[i, 1, 1]].re,
                    1.0 / 3.0,
                    DEFAULT_MARGIN
                ));
                assert!(approx_eq!(
                    f64,
                    net[[i, 0, 1]].re,
                    2.0 / 3.0,
                    DEFAULT_MARGIN
                ));
            }
        }
    }
}
