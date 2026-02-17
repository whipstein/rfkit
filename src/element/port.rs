use crate::{
    element::{Elem, ElemType, Term},
    frequency::{FreqArray, Frequency, new_frequency},
    pts::Points,
};
use ndarray::{IntoDimension, prelude::*};
use num::complex::{Complex64, c64};

#[derive(Clone, Debug, PartialEq)]
pub struct Port {
    id: String,
    z: Complex64,
    node: [usize; 1],
    c: Points<Complex64, Ix2>,
}

impl Port {
    pub fn new(id: String, z: Complex64, node: [usize; 1]) -> Port {
        Port {
            id,
            z,
            node,
            c: points![[Complex64::ZERO]],
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
            node: [1],
            c: Points::zeros((1, 1)),
        }
    }
}

impl Elem for Port {
    fn c(&self, _freq: &Frequency) -> Points<Complex64, Ix2> {
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

    fn net(&self, freq: &Frequency) -> Points<Complex64, Ix3> {
        Points::<Complex64, Ix3>::zeros((freq.npts(), 1, 1).into_dimension())
    }

    fn nodes(&self) -> Vec<usize> {
        self.node.to_vec()
    }

    fn z(&self, _freq: &Frequency) -> Complex64 {
        self.z
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = new_frequency(array![freq.freq(i)], freq.scale());
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

#[derive(Clone)]
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
            c: Points::zeros((1, 1)),
        }
    }
}

impl Default for PortBuilder {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            z: Complex64::ZERO,
            node: [1],
            z0: c64(50.0, 0.0),
        }
    }
}

#[cfg(test)]
mod element_port_tests {
    use super::*;
    use crate::{
        scale::Scale,
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
    fn element_port() {
        let freq_unitval = UnitValBuilder::new().val_scaled(1.0, Scale::Giga).build();
        let freq = new_frequency(array![freq_unitval.val()], freq_unitval.scale());
        let val = c64(50.0, 0.0);
        let exemplar = Port {
            id: "P1".to_string(),
            z: val,
            node: [1],
            c: points![[Complex64::ZERO]],
        };
        let exemplar_z = val;
        let calc = PortBuilder::new().z(val).nodes([1]).id("P1").build();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        comp_pts_ix2(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        exemplar_z.assert_approx_eq(calc.z(&freq), margin, "calc.z()", "0");
    }

    mod port_tests {
        use super::*;

        #[test]
        fn test_port_builder_default() {
            let port = PortBuilder::new().build();
            assert_eq!(port.id(), "C0"); // Default ID from code
            assert_eq!(port.nodes(), vec![1]);
        }

        #[test]
        fn test_port_custom_impedance() {
            let impedances = vec![
                c64(50.0, 0.0),
                c64(75.0, 0.0),
                c64(100.0, 0.0),
                c64(50.0, 10.0), // With imaginary part
            ];

            let freq = new_frequency(array![1e9], Scale::Base);

            for z_val in impedances {
                let port = PortBuilder::new().z(z_val).build();

                let z = port.z(&freq);
                assert_eq!(z, z_val);
            }
        }

        #[test]
        fn test_port_frequency_independent() {
            let freqs = array![1e6, 1e9, 10e9, 100e9];
            let freq = new_frequency(freqs.clone(), Scale::Base);
            let port = PortBuilder::new().z(c64(50.0, 0.0)).build();

            for i in 0..freqs.len() {
                let z = port.z_at(&freq, i);
                assert_eq!(z, c64(50.0, 0.0));
            }
        }

        #[test]
        fn test_port_node_assignment() {
            let nodes = vec![0, 1, 5, 10, 100];

            for node in nodes {
                let port = PortBuilder::new().nodes([node]).build();
                assert_eq!(port.nodes(), vec![node]);
            }
        }

        #[test]
        fn test_port_c_matrix() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let port = PortBuilder::new().build();
            let c_matrix = port.c(&freq);

            assert_eq!(c_matrix[[0, 0]], Complex64::ZERO);
        }

        #[test]
        fn test_port_elem_type() {
            let port = PortBuilder::new().build();
            assert_eq!(port.elem(), ElemType::Port);
        }

        #[test]
        fn test_port_term_trait() {
            let mut port = PortBuilder::new().z(c64(50.0, 0.0)).build();

            assert_eq!(port.val(), c64(50.0, 0.0));

            port.set_val(c64(75.0, 5.0));
            assert_eq!(port.val(), c64(75.0, 5.0));
        }

        #[test]
        fn test_port_id_setting() {
            let port = PortBuilder::new().id("P1").build();
            assert_eq!(port.id(), "P1");
        }
    }
}
