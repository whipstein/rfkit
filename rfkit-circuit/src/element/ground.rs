use crate::element::{Elem, ElemType, Term};
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use rfkit_base::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Ground<T: RealScalar> {
    id: String,
    z: Complex<T>,
    node: [usize; 1],
    c: Points<Complex<T>, Ix2>,
    z0: Complex<T>,
}

impl<T: RealScalar> Ground<T> {
    pub fn new() -> Ground<T> {
        Ground::default()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn z0(&self) -> Complex<T> {
        self.z0
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    fn default_c() -> Points<Complex<T>, Ix2> {
        points![[Complex::ZERO]]
    }
}

impl<T: RealScalar> Default for Ground<T> {
    fn default() -> Self {
        Self {
            id: "gnd".to_string(),
            z: Complex::ZERO,
            node: [1],
            c: Ground::default_c(),
            z0: Complex::ZERO,
        }
    }
}

impl<T: RealScalar> Elem<T> for Ground<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Ground
    }

    fn nodes(&self) -> Vec<usize> {
        self.node.to_vec()
    }

    fn c<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points(
            self.c
                .view()
                .insert_axis(Axis(0))
                .broadcast((freq.npts(), self.c.nrows(), self.c.ncols()))
                .unwrap()
                .to_owned(),
        )
    }

    fn net<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points::zeros((freq.npts(), 1, 1).into_dimension())
    }

    fn z_scalar(&self, _freq: &ScalarUnitValue<T>) -> Complex<T> {
        Complex::ZERO
    }
}

impl<T: RealScalar> Term<T> for Ground<T> {
    fn val(&self) -> Complex<T> {
        self.z
    }

    fn set_val(&mut self, val: Complex<T>) {
        self.z = val;
    }
}

#[cfg(test)]
mod element_ground_tests {
    use super::*;
    use num_complex::{Complex64, c64};

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

    mod ground_tests {
        use super::*;

        #[test]
        fn test_ground_default() {
            let gnd = Ground::<f64>::new();
            assert_eq!(gnd.id(), "gnd");
            assert_eq!(gnd.nodes(), vec![1]);
        }

        #[test]
        fn test_ground_zero_impedance() {
            let freq = ArrayUnitValue::new(&array![1e9, 10e9, 100e9], Scale::Base, Unit::Hz);
            let gnd = Ground::<f64>::new();

            for i in 0..3 {
                let z = gnd.z(&freq)[i];
                assert_eq!(z, Complex64::ZERO);
            }
        }

        #[test]
        fn test_ground_c_matrix() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let gnd = Ground::<f64>::new();
            let c_matrix = gnd.c(&freq);

            assert_eq!(c_matrix[[0, 0, 0]], Complex64::ZERO);
        }

        #[test]
        fn test_ground_single_node() {
            let gnd = Ground::<f64>::new();
            let nodes = gnd.nodes();
            assert_eq!(nodes.len(), 1);
        }

        #[test]
        fn test_ground_elem_type() {
            let gnd = Ground::<f64>::new();
            assert_eq!(gnd.elem(), ElemType::Ground);
        }

        #[test]
        fn test_ground_net_matrix() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let gnd = Ground::<f64>::new();
            let net = gnd.net(&freq);

            assert_eq!(net.shape(), (1, 1, 1));
            assert_eq!(net[[0, 0, 0]], Complex64::ZERO);
        }

        #[test]
        fn test_ground_term_trait() {
            let mut gnd = Ground::<f64>::new();
            assert_eq!(gnd.val(), Complex64::ZERO);

            gnd.set_val(c64(1.0, 2.0));
            assert_eq!(gnd.val(), c64(1.0, 2.0));
        }
    }
}
