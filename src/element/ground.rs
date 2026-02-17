use crate::element::{Elem, ElemType, Term};
use crate::frequency::{FreqArray, Frequency, new_frequency};
use crate::pts::Points;
use ndarray::{IntoDimension, prelude::*};
use num::complex::Complex64;

#[derive(Clone, Debug, PartialEq)]
pub struct Ground {
    id: String,
    z: Complex64,
    node: [usize; 1],
    c: Points<Complex64, Ix2>,
    z0: Complex64,
}

impl Ground {
    pub fn new() -> Ground {
        Ground::default()
    }

    fn z0(&self) -> Complex64 {
        self.z0
    }
}

impl Default for Ground {
    fn default() -> Self {
        Self {
            id: "gnd".to_string(),
            z: Complex64::ZERO,
            node: [1],
            c: points![[Complex64::ZERO]],
            z0: Complex64::ZERO,
        }
    }
}

impl Elem for Ground {
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
        ElemType::Ground
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
        Complex64::ZERO
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = new_frequency(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Term for Ground {
    fn val(&self) -> Complex64 {
        self.z
    }

    fn set_val(&mut self, val: Complex64) {
        self.z = val;
    }
}

#[cfg(test)]
mod element_ground_tests {
    use super::*;
    use crate::scale::Scale;
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

    mod ground_tests {
        use super::*;

        #[test]
        fn test_ground_default() {
            let gnd = Ground::new();
            assert_eq!(gnd.id(), "gnd");
            assert_eq!(gnd.nodes(), vec![1]);
        }

        #[test]
        fn test_ground_zero_impedance() {
            let freq = new_frequency(array![1e9, 10e9, 100e9], Scale::Base);
            let gnd = Ground::new();

            for i in 0..3 {
                let z = gnd.z_at(&freq, i);
                assert_eq!(z, Complex64::ZERO);
            }
        }

        #[test]
        fn test_ground_c_matrix() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let gnd = Ground::new();
            let c_matrix = gnd.c(&freq);

            assert_eq!(c_matrix[[0, 0]], Complex64::ZERO);
        }

        #[test]
        fn test_ground_single_node() {
            let gnd = Ground::new();
            let nodes = gnd.nodes();
            assert_eq!(nodes.len(), 1);
        }

        #[test]
        fn test_ground_elem_type() {
            let gnd = Ground::new();
            assert_eq!(gnd.elem(), ElemType::Ground);
        }

        #[test]
        fn test_ground_net_matrix() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let gnd = Ground::new();
            let net = gnd.net(&freq);

            assert_eq!(net.shape(), &[1, 1, 1]);
            assert_eq!(net[[0, 0, 0]], Complex64::ZERO);
        }

        #[test]
        fn test_ground_term_trait() {
            let mut gnd = Ground::new();
            assert_eq!(gnd.val(), Complex64::ZERO);

            gnd.set_val(c64(1.0, 2.0));
            assert_eq!(gnd.val(), c64(1.0, 2.0));
        }
    }
}
