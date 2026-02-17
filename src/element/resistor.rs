use crate::{
    element::{Elem, ElemType, Lumped},
    frequency::{FreqArray, Frequency, new_frequency},
    pts::{Points, Pts},
    scale::Scale,
    unit::{Unit, UnitValue, Unitized},
};
use ndarray::{IntoDimension, prelude::*};
use num::complex::{Complex, Complex64, c64};

#[derive(Clone, Debug, PartialEq)]
pub struct Resistor {
    id: String,
    res: UnitValue,
    nodes: [usize; 2],
    c: Points<Complex64, Ix2>,
    z0: Complex64,
}

impl Resistor {
    pub fn new(id: String, res: UnitValue, nodes: [usize; 2], z0: Complex64) -> Resistor {
        Resistor {
            id: id,
            res: res,
            nodes: nodes,
            c: points![
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

impl Default for Resistor {
    fn default() -> Self {
        Self {
            id: "R0".to_string(),
            res: UnitValue::default(),
            nodes: [1, 2],
            c: points![
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        }
    }
}

impl Elem for Resistor {
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
        ElemType::Resistor
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points<Complex64, Ix3> {
        Points::<Complex64, Ix3>::from_shape_fn(
            (freq.npts(), 2, 2).into_dimension(),
            |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(1.0 / 3.0, 0.0),
                (1, 0) | (0, 1) => c64(2.0 / 3.0, 0.0),
                _ => c64(0.0, 0.0),
            },
        )
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    fn z(&self, _freq: &Frequency) -> Complex<f64> {
        self.res.val().into()
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = new_frequency(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Lumped for Resistor {
    fn val(&self) -> f64 {
        self.res.val()
    }

    fn set_val(&mut self, val: f64) {
        self.res.set_val(val);
    }
}

impl Unitized for Resistor {
    fn val_scaled(&self) -> f64 {
        self.res.val_scaled()
    }

    fn unitval(&self) -> UnitValue {
        self.res.clone()
    }

    fn scale(&self) -> Scale {
        self.res.scale()
    }

    fn unit(&self) -> Unit {
        self.res.unit()
    }

    fn set_val_scaled(&mut self, val: f64) {
        self.res.set_val_scaled(val);
    }

    fn set_unitval(&mut self, val: UnitValue) {
        self.res = val;
    }

    fn set_scale(&mut self, scale: Scale) {
        self.res.set_scale(scale);
    }

    fn set_unit(&mut self, unit: Unit) {
        self.res.set_unit(unit);
    }
}

#[derive(Clone)]
pub struct ResistorBuilder {
    id: String,
    res: UnitValue,
    nodes: [usize; 2],
    z0: Complex64,
}

impl ResistorBuilder {
    pub fn new() -> Self {
        ResistorBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn res(mut self, res: UnitValue) -> Self {
        self.res = res;
        self
    }

    pub fn val(mut self, res: f64) -> Self {
        self.res.set_val(res);
        self
    }

    pub fn val_scaled(mut self, res: f64, scale: Scale) -> Self {
        self.res.set_scale(scale);
        self.res.set_val_scaled(res);
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

    pub fn build(self) -> Resistor {
        Resistor {
            id: self.id,
            res: self.res,
            nodes: self.nodes,
            c: points![
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: self.z0,
        }
    }
}

impl Default for ResistorBuilder {
    fn default() -> Self {
        Self {
            id: "R0".to_string(),
            res: *UnitValue::default().set_unit(Unit::Ohm),
            nodes: [1, 2],
            z0: c64(50.0, 0.0),
        }
    }
}

#[cfg(test)]
mod element_resistor_tests {
    use super::*;
    use crate::{
        unit::UnitValBuilder,
        util::{ApproxEq, NumMargin, comp_pts_ix2},
    };

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
    fn element_resistor() {
        let freq_unitval = UnitValBuilder::new().val_scaled(1.0, Scale::Giga).build();
        let freq = new_frequency(array![freq_unitval.val()], freq_unitval.scale());
        let val_scaled = 20.0;
        let scale = Scale::Base;
        let unitval = UnitValBuilder::new().val_scaled(val_scaled, scale).build();
        let exemplar = Resistor {
            id: "R1".to_string(),
            res: unitval,
            nodes: [1, 2],
            c: points![
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        };
        let exemplar_z: Complex<f64> = unitval.val().into();
        let calc = ResistorBuilder::new()
            .val_scaled(val_scaled, scale)
            .nodes([1, 2])
            .id("R1")
            .build();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.scale(), &calc.scale());
        assert_eq!(&Unit::Ohm, &calc.unit());
        comp_pts_ix2(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        exemplar_z.assert_approx_eq(calc.z(&freq), margin, "calc.z()", "0");
    }

    mod resistor_tests {
        use super::*;

        #[test]
        fn test_resistor_builder_default() {
            let res = ResistorBuilder::new().build();
            assert_eq!(res.id(), "R0");
            assert_eq!(res.nodes(), vec![1, 2]);
            assert_eq!(res.unit(), Unit::Ohm);
        }

        #[test]
        fn test_resistor_frequency_independent() {
            let freqs = array![1e6, 1e9, 10e9, 100e9];
            let freq = new_frequency(freqs.clone(), Scale::Base);
            let res = ResistorBuilder::new().val(50.0).build();

            // Resistor impedance should be constant across all frequencies
            for i in 0..freqs.len() {
                let z = res.z_at(&freq, i);
                z.assert_approx_eq(c64(50.0, 0.0), DEFAULT_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_resistor_impedance_real_only() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let values = vec![1.0, 10.0, 50.0, 100.0, 1000.0];

            for val in values {
                let res = ResistorBuilder::new().val(val).build();
                let z = res.z(&freq);
                z.assert_approx_eq(c64(val, 0.0), DEFAULT_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_resistor_c_matrix() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let res = ResistorBuilder::new().build();
            let c_matrix = res.c(&freq);

            // Same structure as capacitor
            c_matrix[[0, 0]]
                .re
                .assert_approx_eq(1.0 / 3.0, DEFAULT_MARGIN, "c_matrix", "[0,0]");
            c_matrix[[1, 1]]
                .re
                .assert_approx_eq(1.0 / 3.0, DEFAULT_MARGIN, "c_matrix", "[1,1]");
            c_matrix[[0, 1]]
                .re
                .assert_approx_eq(2.0 / 3.0, DEFAULT_MARGIN, "c_matrix", "[0,1]");
        }

        #[test]
        fn test_resistor_scaling() {
            let res = ResistorBuilder::new().val_scaled(1.0, Scale::Kilo).build();

            assert_eq!(res.val_scaled(), 1.0);
            assert_eq!(res.val(), 1000.0);
            assert_eq!(res.scale(), Scale::Kilo);
        }

        #[test]
        fn test_resistor_node_configuration() {
            let nodes = vec![[0, 1], [1, 2], [2, 5], [10, 20]];

            for node_pair in nodes {
                let res = ResistorBuilder::new().nodes(node_pair).build();
                assert_eq!(res.nodes(), vec![node_pair[0], node_pair[1]]);
            }
        }

        #[test]
        fn test_resistor_elem_type() {
            let res = ResistorBuilder::new().build();
            assert_eq!(res.elem(), ElemType::Resistor);
        }
    }
}
