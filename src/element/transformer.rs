use crate::{
    element::{Elem, ElemType},
    frequency::{FreqArray, Frequency, new_frequency},
    point,
    pts::{Points, Pts},
};
use ndarray::{IntoDimension, prelude::*};
use num::complex::{Complex64, c64};

#[derive(Clone, Debug, PartialEq)]
pub struct Transformer {
    id: String,
    n: f64,
    nodes: [usize; 2],
    c: Points<Complex64, Ix2>,
    z0: Complex64,
}

impl Transformer {
    pub fn new(id: String, n: f64, nodes: [usize; 2], z0: Complex64) -> Transformer {
        Transformer {
            id,
            n,
            nodes,
            c: point![
                Complex64,
                [
                    c64((1.0 - n.powi(2)) / (1.0 + n.powi(2)), 0.0),
                    c64(2.0 * n / (1.0 + n.powi(2)), 0.0)
                ],
                [
                    c64(2.0 * n / (1.0 + n.powi(2)), 0.0),
                    c64((n.powi(2) - 1.0) / (1.0 + n.powi(2)), 0.0)
                ]
            ],
            z0,
        }
    }

    pub fn z0(&self) -> Complex64 {
        self.z0
    }

    pub fn val(&self) -> f64 {
        self.n
    }

    pub fn set_val(&mut self, val: f64) {
        self.n = val;
    }
}

impl Default for Transformer {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            n: 1.0,
            nodes: [1, 2],
            c: point![
                Complex64,
                [c64(0.0, 0.0), c64(1.0, 0.0)],
                [c64(1.0, 0.0), c64(0.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        }
    }
}

impl Elem for Transformer {
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
        ElemType::Transformer
    }

    fn name(&self) -> &String {
        &self.id
    }

    fn net(&self, freq: &Frequency) -> Points<Complex64, Ix3> {
        Points::<Complex64, Ix3>::from_shape_fn(
            (freq.npts(), 2, 2).into_dimension(),
            |(_, j, k)| match (j, k) {
                (0, 0) => c64((1.0 - self.n.powi(2)) / (1.0 + self.n.powi(2)), 0.0),
                (1, 1) => c64((self.n.powi(2) - 1.0) / (1.0 + self.n.powi(2)), 0.0),
                (1, 0) | (0, 1) => c64(2.0 * self.n / (1.0 + self.n.powi(2)), 0.0),
                _ => c64(0.0, 0.0),
            },
        )
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    fn z(&self, _freq: &Frequency) -> Complex64 {
        self.n.powi(2).into()
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = new_frequency(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

#[derive(Clone)]
pub struct TransformerBuilder {
    id: String,
    n: f64,
    nodes: [usize; 2],
    z0: Complex64,
}

impl TransformerBuilder {
    pub fn new() -> Self {
        TransformerBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn n(mut self, n: f64) -> Self {
        self.n = n;
        self
    }

    pub fn val(mut self, n: f64) -> Self {
        self.n = n;
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

    pub fn build(self) -> Transformer {
        Transformer {
            id: self.id,
            n: self.n,
            nodes: self.nodes,
            c: point![
                Complex64,
                [
                    c64((1.0 - self.n.powi(2)) / (1.0 + self.n.powi(2)), 0.0),
                    c64(2.0 * self.n / (1.0 + self.n.powi(2)), 0.0)
                ],
                [
                    c64(2.0 * self.n / (1.0 + self.n.powi(2)), 0.0),
                    c64((self.n.powi(2) - 1.0) / (1.0 + self.n.powi(2)), 0.0)
                ]
            ],
            z0: self.z0,
        }
    }
}

impl Default for TransformerBuilder {
    fn default() -> Self {
        Self {
            id: "T0".to_string(),
            n: 1.0,
            nodes: [1, 2],
            z0: c64(50.0, 0.0),
        }
    }
}

#[cfg(test)]
mod element_transformer_tests {
    use super::*;
    use crate::{
        scale::Scale,
        unit::UnitValBuilder,
        util::{comp_num, comp_pts_ix2},
    };
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
    fn element_transformer() {
        let freq_unitval = UnitValBuilder::new().val_scaled(1.0, Scale::Giga).build();
        let freq = new_frequency(array![freq_unitval.val()], freq_unitval.scale());
        let val = 1.0;
        let exemplar = Transformer {
            id: "T1".to_string(),
            n: val,
            nodes: [1, 2],
            c: point![
                Complex64,
                [c64(0.0, 0.0), c64(1.0, 0.0)],
                [c64(1.0, 0.0), c64(0.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        };
        let exemplar_z = c64(val.powi(2), 0.0);
        let calc = TransformerBuilder::new()
            .val(val)
            .nodes([1, 2])
            .id("T1")
            .build();
        let margin = F64Margin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        comp_pts_ix2(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        comp_num(&exemplar_z, &calc.z(&freq), margin, "calc.z()", "0");
    }

    mod transformer_tests {
        use super::*;

        #[test]
        fn test_transformer_builder_default() {
            let n = TransformerBuilder::new().build();
            assert_eq!(n.id(), "T0");
            assert_eq!(n.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_transformer_builder_with_all_parameters() {
            let n = TransformerBuilder::new()
                .id("T1")
                .val(10.0)
                .nodes([1, 3])
                .z0(c64(75.0, 0.0))
                .build();

            assert_eq!(n.id(), "T1");
            assert_eq!(n.nodes(), vec![1, 3]);
            assert_eq!(n.val(), 10.0);
        }

        #[test]
        fn test_transformer_impedance_calculation() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let n = TransformerBuilder::new().val(1.0).build();

            let z = n.z(&freq);
            let expected_z = c64(1.0, 0.0);

            assert!(approx_eq!(f64, z.re, expected_z.re, epsilon = 1e-6));
            assert!(approx_eq!(f64, z.im, expected_z.im, epsilon = 1e-6));
        }

        #[test]
        fn test_transformer_c_matrix_structure() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let n = TransformerBuilder::new().build();
            let c_matrix = n.c(&freq);

            // Check diagonal and off-diagonal elements
            assert!(approx_eq!(f64, c_matrix[[0, 0]].re, 0.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c_matrix[[1, 1]].re, 0.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c_matrix[[0, 1]].re, 1.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c_matrix[[1, 0]].re, 1.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_multiple_frequencies() {
            let freqs = array![1e9, 2e9, 5e9, 10e9];
            let freq = new_frequency(freqs.clone(), Scale::Base);
            let n = TransformerBuilder::new().val(1.0).build();

            for i in 0..freqs.len() {
                let z = n.z_at(&freq, i);
                let expected = c64(1.0, 0.0);
                assert!(approx_eq!(f64, z.re, expected.re, RELAXED_MARGIN));
                assert!(approx_eq!(f64, z.im, expected.im, RELAXED_MARGIN));
            }
        }

        #[test]
        fn test_transformer_value_scaling() {
            let n = TransformerBuilder::new().val(10.0).build();

            assert_eq!(n.val(), 10.0);
        }

        #[test]
        fn test_transformer_set_operations() {
            let mut n = TransformerBuilder::new().build();

            n.set_id("T_new");
            assert_eq!(n.id(), "T_new");

            n.set_val(5.0);
            assert_eq!(n.val(), 5.0);
        }

        #[test]
        fn test_transformer_elem_type() {
            let n = TransformerBuilder::new().build();
            assert_eq!(n.elem(), ElemType::Transformer);
        }

        #[test]
        fn test_transformer_net_matrix() {
            let freq = new_frequency(array![1e9, 2e9], Scale::Base);
            let n = TransformerBuilder::new().build();
            let net = n.net(&freq);

            assert_eq!(net.shape(), (2, 2, 2)); // [npts, 2, 2]

            // Check structure is consistent across frequencies
            for i in 0..2 {
                assert!(approx_eq!(f64, net[[i, 0, 0]].re, 0.0, DEFAULT_MARGIN));
                assert!(approx_eq!(f64, net[[i, 1, 1]].re, 0.0, DEFAULT_MARGIN));
                assert!(approx_eq!(f64, net[[i, 0, 1]].re, 1.0, DEFAULT_MARGIN));
            }
        }
    }
}
