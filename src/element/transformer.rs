use crate::{
    element::{Elem, ElemType, Q},
    frequency::{FreqArray, Frequency, new_frequency},
    network::NetworkPoint,
    point,
    pts::{Points, Pts},
    scale::Scale,
    unit::{Unit, UnitValue},
};
use ndarray::{IntoDimension, SliceInfo, prelude::*};
use num::complex::{Complex64, c64};

#[derive(Clone, Debug, PartialEq)]
pub struct Transformer {
    id: String,
    n: f64,
    km: f64,
    l1: UnitValue,
    l2: UnitValue,
    q1: Option<Q>,
    q2: Option<Q>,
    nodes: [usize; 2],
    c: Points<Complex64, Ix2>,
    z0: Complex64,
}

impl Transformer {
    pub fn new(
        id: String,
        km: f64,
        l1: UnitValue,
        l2: UnitValue,
        q1: Option<Q>,
        q2: Option<Q>,
        freq: &Frequency,
        nodes: [usize; 2],
        z0: Complex64,
    ) -> Self {
        let mut out = Self {
            id,
            n: (l1.val() / l2.val()).sqrt(),
            km,
            l1,
            l2,
            q1,
            q2,
            nodes,
            c: Points::zeros((2, 2)),
            z0,
        };

        out.c = out
            .calc_c(freq)
            .pt::<SliceInfo<[ndarray::SliceInfoElem; 3], Ix3, Ix2>>(0);

        out
    }

    fn calc_c(&self, freq: &Frequency) -> Points<Complex64, Ix3> {
        let y = Points::<Complex64, Ix3>::from_shape_fn(
            (freq.npts(), 2, 2).into_dimension(),
            |(_, j, k)| match (j, k) {
                (0, 0) => {
                    1.0 / (Complex64::I * freq.w_pt(0) * self.l1.val() * (1.0 - self.km.powi(2)))
                }
                (1, 1) => {
                    1.0 / (Complex64::I * freq.w_pt(0) * self.l2.val() * (1.0 - self.km.powi(2)))
                }
                (1, 0) | (0, 1) => {
                    self.km
                        / (Complex64::I
                            * freq.w_pt(0)
                            * (self.l1.val() * self.l2.val()).sqrt()
                            * (1.0 - self.km.powi(2)))
                }
                _ => c64(0.0, 0.0),
            },
        );

        y.y_to_s(array![self.z0, self.z0].view()).unwrap()
    }

    pub fn z0(&self) -> Complex64 {
        self.z0
    }

    pub fn n(&self) -> f64 {
        self.n
    }

    pub fn km(&self) -> f64 {
        self.km
    }

    pub fn m(&self) -> f64 {
        self.km * self.l1.val()
    }

    pub fn l1(&self) -> f64 {
        self.l1.val()
    }

    pub fn l1_scaled(&self) -> f64 {
        self.l1.val_scaled()
    }

    pub fn l2(&self) -> f64 {
        self.l2.val()
    }

    pub fn l2_scaled(&self) -> f64 {
        self.l2.val_scaled()
    }

    pub fn q1(&self) -> Option<Q> {
        self.q1.clone()
    }

    pub fn q2(&self) -> Option<Q> {
        self.q2.clone()
    }

    pub fn set_km(&mut self, val: f64) {
        self.km = val;
    }

    pub fn set_l1(&mut self, val: UnitValue) {
        self.l1 = val;
    }

    pub fn set_l1_val(&mut self, val: f64) {
        self.l1.set_val(val);
    }

    pub fn set_l1_val_scaled(&mut self, val: f64) {
        self.l1.set_val_scaled(val);
    }

    pub fn set_l2(&mut self, val: UnitValue) {
        self.l2 = val;
    }

    pub fn set_l2_val(&mut self, val: f64) {
        self.l2.set_val(val);
    }

    pub fn set_l2_val_scaled(&mut self, val: f64) {
        self.l2.set_val_scaled(val);
    }

    pub fn set_q1(&mut self, val: Q) {
        self.q1 = Some(val);
    }

    pub fn set_q2(&mut self, val: Q) {
        self.q2 = Some(val);
    }

    pub fn unset_q1(&mut self) {
        self.q1 = None;
    }

    pub fn unset_q2(&mut self) {
        self.q2 = None;
    }
}

impl Default for Transformer {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            n: 1.0,
            km: 1.0,
            l1: *UnitValue::default().set_unit(Unit::Henry),
            l2: *UnitValue::default().set_unit(Unit::Henry),
            q1: None,
            q2: None,
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
        self.calc_c(freq)
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
    n: Option<f64>,
    km: Option<f64>,
    m: Option<UnitValue>,
    l1: UnitValue,
    l2: Option<UnitValue>,
    q1: Option<Q>,
    q2: Option<Q>,
    freq: Frequency,
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

    pub fn n(mut self, n: Option<f64>) -> Self {
        self.n = n;
        self
    }

    pub fn km(mut self, km: Option<f64>) -> Self {
        self.km = km;
        self
    }

    pub fn m(mut self, ind: Option<UnitValue>) -> Self {
        self.m = ind;
        self
    }

    pub fn m_val(mut self, ind: f64) -> Self {
        let _ = *self
            .m
            .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
            .set_val(ind);
        self
    }

    pub fn m_val_scaled(mut self, ind: f64, scale: Scale) -> Self {
        let _ = *self
            .m
            .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
            .set_scale(scale);
        let _ = *self
            .m
            .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
            .set_val_scaled(ind);
        self
    }

    pub fn l1(mut self, ind: UnitValue) -> Self {
        self.l1 = ind;
        self
    }

    pub fn l1_val(mut self, ind: f64) -> Self {
        self.l1.set_val(ind);
        self
    }

    pub fn l1_val_scaled(mut self, ind: f64, scale: Scale) -> Self {
        self.l1.set_scale(scale);
        self.l1.set_val_scaled(ind);
        self
    }

    pub fn l2(mut self, ind: Option<UnitValue>) -> Self {
        self.l2 = ind;
        self
    }

    pub fn l2_val(mut self, ind: f64) -> Self {
        let _ = *self
            .l2
            .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
            .set_val(ind);
        self
    }

    pub fn l2_val_scaled(mut self, ind: f64, scale: Scale) -> Self {
        let _ = *self
            .l2
            .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
            .set_scale(scale);
        let _ = *self
            .l2
            .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
            .set_val_scaled(ind);
        self
    }

    pub fn q1(mut self, q: Option<Q>) -> Self {
        self.q1 = q;
        self
    }

    pub fn q1_val(mut self, q: f64) -> Self {
        let _ = self.q1.get_or_insert(Q::default()).set_q(q);
        if let Some(val) = self.q1.as_mut() {
            val.set_fq(&self.freq);
        }
        self
    }

    pub fn q2(mut self, q: Option<Q>) -> Self {
        self.q2 = q;
        self
    }

    pub fn q2_val(mut self, q: f64) -> Self {
        let _ = self.q2.get_or_insert(Q::default()).set_q(q);
        if let Some(val) = self.q2.as_mut() {
            val.set_fq(&self.freq);
        }
        self
    }

    pub fn freq(mut self, freq: &Frequency) -> Self {
        self.freq = freq.clone();
        if self.q1.is_some() {
            if let Some(val) = self.q1.as_mut() {
                val.set_fq(&self.freq);
            }
        }
        if self.q2.is_some() {
            if let Some(val) = self.q2.as_mut() {
                val.set_fq(&self.freq);
            }
        }
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

    pub fn build(mut self) -> Transformer {
        if self.n.is_none() && self.l2.is_none() {
            panic!("Must specify either n or l2 to completely define a Transformer");
        }
        if self.km.is_none() && self.m.is_none() {
            panic!("Must specify either km or m to completely define a Transformer");
        }

        match (self.n, self.l2) {
            (Some(n), None) => {
                self.l2 = Some(self.l1.clone());
                self.l2
                    .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
                    .set_val(self.l1.val() / n.powi(2));
            }
            _ => (),
        }
        match (self.km, self.m) {
            (None, Some(m)) => {
                self.km = Some(m.val() / self.l1.val());
            }
            _ => (),
        }
        Transformer::new(
            self.id,
            self.km.unwrap(),
            self.l1,
            self.l2.unwrap(),
            self.q1,
            self.q2,
            &self.freq,
            self.nodes,
            self.z0,
        )
    }
}

impl Default for TransformerBuilder {
    fn default() -> Self {
        Self {
            id: "T0".to_string(),
            n: None,
            km: None,
            m: None,
            l1: *UnitValue::default().set_unit(Unit::Henry),
            l2: None,
            q1: None,
            q2: None,
            freq: Frequency::default(),
            nodes: [1, 2],
            z0: c64(50.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IdealTransformer {
    id: String,
    n: f64,
    nodes: [usize; 2],
    c: Points<Complex64, Ix2>,
    z0: Complex64,
}

impl IdealTransformer {
    pub fn new(id: String, n: f64, nodes: [usize; 2], z0: Complex64) -> Self {
        Self {
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

impl Default for IdealTransformer {
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

impl Elem for IdealTransformer {
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
        ElemType::IdealTransformer
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
pub struct IdealTransformerBuilder {
    id: String,
    n: f64,
    nodes: [usize; 2],
    z0: Complex64,
}

impl IdealTransformerBuilder {
    pub fn new() -> Self {
        Self::default()
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

    pub fn build(self) -> IdealTransformer {
        IdealTransformer {
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

impl Default for IdealTransformerBuilder {
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
        let n = 1.0;
        let km = 0.5;
        let l1 = UnitValue::new(5e-12, Scale::Pico, Unit::Henry);

        let calc = TransformerBuilder::new()
            .freq(&freq)
            .n(Some(n))
            .km(Some(km))
            .l1(l1)
            .nodes([1, 2])
            .id("T1")
            .build();
        let margin = F64Margin::default();

        // Verify basic properties
        assert_eq!(&calc.id(), "T1");
        assert_eq!(calc.nodes(), vec![1, 2]);
        assert!(approx_eq!(f64, calc.n(), n, margin));
        assert!(approx_eq!(f64, calc.km(), km, margin));
        assert!(approx_eq!(f64, calc.l1(), 5e-12, margin));

        // Verify impedance ratio (z = n^2)
        let exemplar_z = c64(n.powi(2), 0.0);
        comp_num(&exemplar_z, &calc.z(&freq), margin, "calc.z()", "0");

        // Verify c matrix has valid structure
        let c = calc.c(&freq);
        assert_eq!(c.shape(), (2, 2));
        assert!(c[[0, 0]].is_finite());
        assert!(c[[1, 1]].is_finite());
        // S12 should equal S21 for reciprocal network
        assert!(approx_eq!(f64, c[[0, 1]].re, c[[1, 0]].re, margin));
        assert!(approx_eq!(f64, c[[0, 1]].im, c[[1, 0]].im, margin));
    }

    #[test]
    fn element_idealtransformer() {
        let freq_unitval = UnitValBuilder::new().val_scaled(1.0, Scale::Giga).build();
        let freq = new_frequency(array![freq_unitval.val()], freq_unitval.scale());
        let val = 1.0;
        let exemplar = IdealTransformer {
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
        let calc = IdealTransformerBuilder::new()
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

        // Helper function to create a default frequency for tests
        fn test_freq() -> Frequency {
            new_frequency(array![1e9], Scale::Base)
        }

        #[test]
        fn test_transformer_builder_default() {
            // TransformerBuilder requires n/l2 and km/m to be specified
            let freq = test_freq();
            let n = TransformerBuilder::new()
                .freq(&freq)
                .n(Some(1.0))
                .km(Some(1.0))
                .build();
            assert_eq!(n.id(), "T0");
            assert_eq!(n.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_transformer_builder_with_all_parameters() {
            let freq = test_freq();
            let mut l1 = *UnitValue::default().set_unit(Unit::Henry);
            l1.set_scale(Scale::Pico);
            l1.set_val_scaled(5.0);
            let n = TransformerBuilder::new()
                .freq(&freq)
                .id("T1")
                .n(Some(10.0))
                .km(Some(0.5))
                .l1(l1)
                .nodes([1, 3])
                .z0(c64(75.0, 0.0))
                .build();

            assert_eq!(n.id(), "T1");
            assert_eq!(n.nodes(), vec![1, 3]);
            assert_eq!(n.n(), 10.0);
            assert_eq!(n.l1(), 5e-12);
            assert_eq!(n.l2(), 5e-12 / 100.0);
            assert_eq!(n.km(), 0.5);
            assert_eq!(n.m(), 2.5e-12);
        }

        #[test]
        fn test_transformer_impedance_calculation() {
            let mut l1 = *UnitValue::default().set_unit(Unit::Henry);
            l1.set_scale(Scale::Pico);
            l1.set_val_scaled(5.0);
            let freq = new_frequency(array![1e9], Scale::Base);
            let n = TransformerBuilder::new()
                .freq(&freq)
                .n(Some(1.0))
                .km(Some(1.0))
                .l1(l1)
                .build();

            let z = n.z(&freq);
            let expected_z = c64(1.0, 0.0);

            assert!(approx_eq!(f64, z.re, expected_z.re, epsilon = 1e-6));
            assert!(approx_eq!(f64, z.im, expected_z.im, epsilon = 1e-6));
        }

        #[test]
        fn test_transformer_c_matrix_structure() {
            let freq = new_frequency(array![1e9], Scale::Base);
            // Note: km=1.0 causes division by zero in the Y-to-S conversion
            // Use km < 1.0 for valid transformer calculations
            let n = TransformerBuilder::new()
                .freq(&freq)
                .n(Some(1.0))
                .km(Some(0.9))
                .l1_val_scaled(10.0, Scale::Nano)
                .build();
            let c_matrix = n.c(&freq);

            // Verify c matrix has valid (non-NaN) values and correct shape
            assert_eq!(c_matrix.inner().dim(), (2, 2));
            assert!(c_matrix[[0, 0]].is_finite());
            assert!(c_matrix[[1, 1]].is_finite());
            assert!(c_matrix[[0, 1]].is_finite());
            assert!(c_matrix[[1, 0]].is_finite());
            // S12 should equal S21 for a reciprocal network
            assert!(approx_eq!(
                f64,
                c_matrix[[0, 1]].re,
                c_matrix[[1, 0]].re,
                DEFAULT_MARGIN
            ));
        }

        #[test]
        fn test_transformer_multiple_frequencies() {
            let mut l1 = *UnitValue::default().set_unit(Unit::Henry);
            l1.set_scale(Scale::Pico);
            l1.set_val_scaled(5.0);
            let freqs = array![1e9, 2e9, 5e9, 10e9];
            let freq = new_frequency(freqs.clone(), Scale::Base);
            let n = TransformerBuilder::new()
                .freq(&freq)
                .n(Some(1.0))
                .km(Some(1.0))
                .l1(l1)
                .build();

            for i in 0..freqs.len() {
                let z = n.z_at(&freq, i);
                let expected = c64(1.0, 0.0);
                assert!(approx_eq!(f64, z.re, expected.re, RELAXED_MARGIN));
                assert!(approx_eq!(f64, z.im, expected.im, RELAXED_MARGIN));
            }
        }

        #[test]
        fn test_transformer_value_scaling() {
            let freq = test_freq();
            let mut l1 = *UnitValue::default().set_unit(Unit::Henry);
            l1.set_scale(Scale::Pico);
            l1.set_val_scaled(5.0);
            let n = TransformerBuilder::new()
                .freq(&freq)
                .n(Some(10.0))
                .km(Some(1.0))
                .l1(l1)
                .build();

            assert_eq!(n.n(), 10.0);
        }

        #[test]
        fn test_transformer_set_operations() {
            let freq = test_freq();
            let mut n = TransformerBuilder::new()
                .freq(&freq)
                .n(Some(1.0))
                .km(Some(1.0))
                .build();

            n.set_id("T_new");
            assert_eq!(n.id(), "T_new");

            n.set_l1_val_scaled(5.0);
            assert_eq!(n.l1_scaled(), 5.0);
        }

        #[test]
        fn test_transformer_elem_type() {
            let freq = test_freq();
            let n = TransformerBuilder::new()
                .freq(&freq)
                .n(Some(1.0))
                .km(Some(1.0))
                .build();
            assert_eq!(n.elem(), ElemType::Transformer);
        }

        #[test]
        fn test_transformer_net_matrix() {
            let freq = new_frequency(array![1e9, 2e9], Scale::Base);
            // Note: km=1.0 causes division by zero in the Y-to-S conversion
            // Use km < 1.0 for valid transformer calculations
            let n = TransformerBuilder::new()
                .freq(&freq)
                .n(Some(1.0))
                .km(Some(0.9))
                .l1_val_scaled(10.0, Scale::Nano)
                .build();
            let net = n.net(&freq);

            assert_eq!(net.shape(), (2, 2, 2)); // [npts, 2, 2]

            // Verify net matrix has valid (non-NaN) values
            for i in 0..2 {
                assert!(net[[i, 0, 0]].is_finite());
                assert!(net[[i, 1, 1]].is_finite());
                assert!(net[[i, 0, 1]].is_finite());
                assert!(net[[i, 1, 0]].is_finite());
                // S12 should equal S21 for a reciprocal network
                assert!(approx_eq!(
                    f64,
                    net[[i, 0, 1]].re,
                    net[[i, 1, 0]].re,
                    DEFAULT_MARGIN
                ));
            }
        }
    }

    mod idealtransformer_tests {
        use super::*;

        #[test]
        fn test_transformer_builder_default() {
            let n = IdealTransformerBuilder::new().build();
            assert_eq!(n.id(), "T0");
            assert_eq!(n.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_transformer_builder_with_all_parameters() {
            let n = IdealTransformerBuilder::new()
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
            let n = IdealTransformerBuilder::new().val(1.0).build();

            let z = n.z(&freq);
            let expected_z = c64(1.0, 0.0);

            assert!(approx_eq!(f64, z.re, expected_z.re, epsilon = 1e-6));
            assert!(approx_eq!(f64, z.im, expected_z.im, epsilon = 1e-6));
        }

        #[test]
        fn test_transformer_c_matrix_structure() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let n = IdealTransformerBuilder::new().build();
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
            let n = IdealTransformerBuilder::new().val(1.0).build();

            for i in 0..freqs.len() {
                let z = n.z_at(&freq, i);
                let expected = c64(1.0, 0.0);
                assert!(approx_eq!(f64, z.re, expected.re, RELAXED_MARGIN));
                assert!(approx_eq!(f64, z.im, expected.im, RELAXED_MARGIN));
            }
        }

        #[test]
        fn test_transformer_value_scaling() {
            let n = IdealTransformerBuilder::new().val(10.0).build();

            assert_eq!(n.val(), 10.0);
        }

        #[test]
        fn test_transformer_set_operations() {
            let mut n = IdealTransformerBuilder::new().build();

            n.set_id("T_new");
            assert_eq!(n.id(), "T_new");

            n.set_val(5.0);
            assert_eq!(n.val(), 5.0);
        }

        #[test]
        fn test_transformer_elem_type() {
            let n = IdealTransformerBuilder::new().build();
            assert_eq!(n.elem(), ElemType::IdealTransformer);
        }

        #[test]
        fn test_transformer_net_matrix() {
            let freq = new_frequency(array![1e9, 2e9], Scale::Base);
            let n = IdealTransformerBuilder::new().build();
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

    // ============================================================
    // COMPREHENSIVE TRANSFORMER TESTS
    // ============================================================

    mod transformer_comprehensive_tests {
        use super::*;
        use crate::element::q::{QBuilder, QMode};

        // Helper function to create a default frequency for tests
        fn test_freq() -> Frequency {
            new_frequency(array![1e9], Scale::Base)
        }

        // ----------------------------------------------------------
        // Transformer struct direct tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_new_direct() {
            let freq = test_freq();
            let l1 = UnitValue::new(10e-9, Scale::Nano, Unit::Henry);
            let l2 = UnitValue::new(10e-9, Scale::Nano, Unit::Henry);

            let t = Transformer::new(
                "T_direct".to_string(),
                0.9,
                l1,
                l2,
                None,
                None,
                &freq,
                [1, 2],
                c64(50.0, 0.0),
            );

            assert_eq!(t.id(), "T_direct");
            assert!(approx_eq!(f64, t.km(), 0.9, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t.n(), 1.0, DEFAULT_MARGIN)); // sqrt(l1/l2) = 1
            assert_eq!(t.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_transformer_default() {
            let t = Transformer::default();
            assert_eq!(t.id(), "C0");
            assert!(approx_eq!(f64, t.n(), 1.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t.km(), 1.0, DEFAULT_MARGIN));
            assert_eq!(t.nodes(), vec![1, 2]);
            assert_eq!(t.z0(), c64(50.0, 0.0));
        }

        #[test]
        fn test_transformer_n_calculation() {
            // n = sqrt(l1/l2)
            let l1 = UnitValue::new(100e-9, Scale::Nano, Unit::Henry);
            let l2 = UnitValue::new(25e-9, Scale::Nano, Unit::Henry);
            let freq = test_freq();

            let t = Transformer::new(
                "T1".to_string(),
                0.9,
                l1,
                l2,
                None,
                None,
                &freq,
                [1, 2],
                c64(50.0, 0.0),
            );

            // n = sqrt(100/25) = sqrt(4) = 2
            assert!(approx_eq!(f64, t.n(), 2.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_mutual_inductance() {
            // m = km * l1
            let freq = test_freq();
            let l1 = UnitValue::new(10e-9, Scale::Nano, Unit::Henry);
            let km = 0.8;
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1(l1.clone())
                .n(Some(1.0))
                .km(Some(km))
                .build();

            let expected_m = km * l1.val();
            assert!(approx_eq!(f64, t.m(), expected_m, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_l1_accessors() {
            let freq = test_freq();
            let l1 = UnitValue::new(5e-12, Scale::Pico, Unit::Henry);
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1(l1)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            assert!(approx_eq!(f64, t.l1(), 5e-12, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t.l1_scaled(), 5.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_l2_accessors() {
            let l1 = UnitValue::new(10e-12, Scale::Pico, Unit::Henry);
            let l2 = UnitValue::new(2.5e-12, Scale::Pico, Unit::Henry);
            let freq = test_freq();

            let t = Transformer::new(
                "T1".to_string(),
                0.9,
                l1,
                l2.clone(),
                None,
                None,
                &freq,
                [1, 2],
                c64(50.0, 0.0),
            );

            assert!(approx_eq!(f64, t.l2(), l2.val(), DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t.l2_scaled(), 2.5, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_q_accessors() {
            let freq = test_freq();
            let q1 = QBuilder::new().q(100.0).mode(QMode::Constant).build();
            let q2 = QBuilder::new().q(150.0).mode(QMode::Constant).build();

            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .q1(Some(q1.clone()))
                .q2(Some(q2.clone()))
                .build();

            assert!(t.q1().is_some());
            assert!(t.q2().is_some());
            assert!(approx_eq!(f64, t.q1().unwrap().q(), 100.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t.q2().unwrap().q(), 150.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_q_none_by_default() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            assert!(t.q1().is_none());
            assert!(t.q2().is_none());
        }

        // ----------------------------------------------------------
        // Transformer setter tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_set_km() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.5))
                .build();

            t.set_km(0.95);
            assert!(approx_eq!(f64, t.km(), 0.95, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_set_l1() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            let new_l1 = UnitValue::new(20e-9, Scale::Nano, Unit::Henry);
            t.set_l1(new_l1);
            assert!(approx_eq!(f64, t.l1(), 20e-9, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_set_l1_val() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            t.set_l1_val(15e-9);
            assert!(approx_eq!(f64, t.l1(), 15e-9, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_set_l1_val_scaled() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            t.set_l1_val_scaled(25.0);
            assert!(approx_eq!(f64, t.l1_scaled(), 25.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_set_l2() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            let new_l2 = UnitValue::new(5e-9, Scale::Nano, Unit::Henry);
            t.set_l2(new_l2);
            assert!(approx_eq!(f64, t.l2(), 5e-9, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_set_l2_val() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            t.set_l2_val(8e-9);
            assert!(approx_eq!(f64, t.l2(), 8e-9, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_set_l2_val_scaled() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            t.set_l2_val_scaled(12.0);
            assert!(approx_eq!(f64, t.l2_scaled(), 12.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_set_q1() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            let q = QBuilder::new().q(200.0).build();
            t.set_q1(q);

            assert!(t.q1().is_some());
            assert!(approx_eq!(f64, t.q1().unwrap().q(), 200.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_set_q2() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            let q = QBuilder::new().q(250.0).build();
            t.set_q2(q);

            assert!(t.q2().is_some());
            assert!(approx_eq!(f64, t.q2().unwrap().q(), 250.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_unset_q1() {
            let freq = test_freq();
            let q1 = QBuilder::new().q(100.0).build();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .q1(Some(q1))
                .build();

            assert!(t.q1().is_some());
            t.unset_q1();
            assert!(t.q1().is_none());
        }

        #[test]
        fn test_transformer_unset_q2() {
            let freq = test_freq();
            let q2 = QBuilder::new().q(100.0).build();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .q2(Some(q2))
                .build();

            assert!(t.q2().is_some());
            t.unset_q2();
            assert!(t.q2().is_none());
        }

        // ----------------------------------------------------------
        // TransformerBuilder tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_builder_with_n_and_km() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .id("T_test")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(2.0))
                .km(Some(0.9))
                .nodes([1, 3])
                .z0(c64(75.0, 0.0))
                .build();

            assert_eq!(t.id(), "T_test");
            assert!(approx_eq!(f64, t.n(), 2.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t.km(), 0.9, DEFAULT_MARGIN));
            // l2 = l1 / n^2 = 10e-9 / 4 = 2.5e-9
            assert!(approx_eq!(f64, t.l2(), 2.5e-9, DEFAULT_MARGIN));
            assert_eq!(t.nodes(), vec![1, 3]);
        }

        #[test]
        fn test_transformer_builder_with_l2_explicit() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .l2_val_scaled(5.0, Scale::Nano)
                .km(Some(0.9))
                .build();

            // n = sqrt(l1/l2) = sqrt(10/5) = sqrt(2)
            let expected_n = (10.0 / 5.0_f64).sqrt();
            assert!(approx_eq!(f64, t.n(), expected_n, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t.l2(), 5e-9, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_builder_with_m_instead_of_km() {
            // m = km * l1, so km = m / l1
            let freq = test_freq();
            let l1_val = 10e-9;
            let m_val = 8e-9;
            let expected_km = m_val / l1_val;

            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val(l1_val)
                .n(Some(1.0))
                .m_val(m_val)
                .build();

            assert!(approx_eq!(f64, t.km(), expected_km, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_builder_m_val_scaled() {
            let freq = test_freq();
            let l1_val = 10e-9;
            let m_scaled = 5.0; // 5 nH
            let m_val = 5e-9;
            let expected_km = m_val / l1_val;

            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val(l1_val)
                .n(Some(1.0))
                .m_val_scaled(m_scaled, Scale::Nano)
                .build();

            assert!(approx_eq!(f64, t.km(), expected_km, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_builder_l1_val() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val(15e-9)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            assert!(approx_eq!(f64, t.l1(), 15e-9, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_builder_l2_val() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val(10e-9)
                .l2_val(5e-9)
                .km(Some(0.9))
                .build();

            assert!(approx_eq!(f64, t.l2(), 5e-9, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_builder_freq() {
            let freq = new_frequency(array![2e9], Scale::Base);
            let t = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .freq(&freq)
                .build();

            // Transformer should build successfully with custom frequency
            assert!(t.l1() > 0.0);
        }

        #[test]
        #[should_panic(expected = "Must specify either n or l2")]
        fn test_transformer_builder_missing_n_and_l2_panics() {
            let freq = test_freq();
            TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .km(Some(0.9))
                .build();
        }

        #[test]
        #[should_panic(expected = "Must specify either km or m")]
        fn test_transformer_builder_missing_km_and_m_panics() {
            let freq = test_freq();
            TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .build();
        }

        // ----------------------------------------------------------
        // Transformer Elem trait tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_elem_trait_id() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .id("T_elem")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            assert_eq!(t.id(), "T_elem");
        }

        #[test]
        fn test_transformer_elem_trait_name() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .id("T_name")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            assert_eq!(t.name(), "T_name");
        }

        #[test]
        fn test_transformer_elem_trait_elem_type() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            assert_eq!(t.elem(), ElemType::Transformer);
        }

        #[test]
        fn test_transformer_elem_trait_nodes() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .nodes([5, 10])
                .build();

            assert_eq!(t.nodes(), vec![5, 10]);
        }

        #[test]
        fn test_transformer_elem_trait_set_id() {
            let freq = test_freq();
            let mut t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            t.set_id("T_modified");
            assert_eq!(t.id(), "T_modified");
        }

        #[test]
        fn test_transformer_elem_trait_z() {
            let freq = test_freq();
            let n = 2.0;
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(n))
                .km(Some(0.9))
                .build();

            let z = t.z(&freq);
            // z = n^2
            assert!(approx_eq!(f64, z.re, n.powi(2), DEFAULT_MARGIN));
            assert!(approx_eq!(f64, z.im, 0.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_elem_trait_z_at() {
            let freq = new_frequency(array![1e9, 2e9, 5e9], Scale::Base);
            let n = 3.0;
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(n))
                .km(Some(0.9))
                .build();

            for i in 0..3 {
                let z = t.z_at(&freq, i);
                assert!(approx_eq!(f64, z.re, n.powi(2), DEFAULT_MARGIN));
            }
        }

        #[test]
        fn test_transformer_elem_trait_c() {
            let freq = test_freq();
            // Note: km=1.0 causes division by zero in the Y-to-S conversion
            // Use km < 1.0 for valid transformer calculations
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            let c = t.c(&freq);
            assert_eq!(c.shape(), (2, 2));
        }

        #[test]
        fn test_transformer_elem_trait_c_at() {
            let freq = test_freq();
            // Note: km=1.0 causes division by zero in the Y-to-S conversion
            // Use km < 1.0 for valid transformer calculations
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            let c = t.c(&freq);
            for j in 0..2 {
                for k in 0..2 {
                    let c_at = t.c_at(&freq, j, k);
                    // c_at should match corresponding element in c matrix
                    assert!(approx_eq!(f64, c_at.re, c[[j, k]].re, DEFAULT_MARGIN));
                    assert!(approx_eq!(f64, c_at.im, c[[j, k]].im, DEFAULT_MARGIN));
                }
            }
        }

        #[test]
        fn test_transformer_elem_trait_net() {
            let freq = new_frequency(array![1e9, 2e9, 3e9], Scale::Base);
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            let net = t.net(&freq);
            assert_eq!(net.shape(), (3, 2, 2));
        }

        // ----------------------------------------------------------
        // Transformer impedance transformation tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_impedance_transformation_ratio() {
            let freq = test_freq();

            // For various turns ratios, z = n^2
            let ratios = vec![0.5, 1.0, 2.0, 4.0, 10.0];

            for n in ratios {
                let t = TransformerBuilder::new()
                    .freq(&freq)
                    .l1_val_scaled(10.0, Scale::Nano)
                    .n(Some(n))
                    .km(Some(0.9))
                    .build();

                let z = t.z(&freq);
                assert!(approx_eq!(f64, z.re, n.powi(2), DEFAULT_MARGIN));
            }
        }

        #[test]
        fn test_transformer_coupling_coefficient_range() {
            // km should be between 0 and 1 for physical transformers
            let freq = test_freq();
            let km_values = vec![0.1, 0.5, 0.8, 0.95, 0.99];

            for km in km_values {
                let t = TransformerBuilder::new()
                    .freq(&freq)
                    .l1_val_scaled(10.0, Scale::Nano)
                    .n(Some(1.0))
                    .km(Some(km))
                    .build();

                assert!(approx_eq!(f64, t.km(), km, DEFAULT_MARGIN));
            }
        }

        // ----------------------------------------------------------
        // Clone, Debug, PartialEq trait tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_clone() {
            let freq = test_freq();
            let t1 = TransformerBuilder::new()
                .freq(&freq)
                .id("T_clone")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(2.0))
                .km(Some(0.9))
                .build();

            let t2 = t1.clone();

            assert_eq!(t1.id(), t2.id());
            assert!(approx_eq!(f64, t1.n(), t2.n(), DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t1.km(), t2.km(), DEFAULT_MARGIN));
            assert!(approx_eq!(f64, t1.l1(), t2.l1(), DEFAULT_MARGIN));
        }

        #[test]
        fn test_transformer_partial_eq() {
            let freq = test_freq();
            let t1 = TransformerBuilder::new()
                .freq(&freq)
                .id("T_eq")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(2.0))
                .km(Some(0.9))
                .build();

            let t2 = TransformerBuilder::new()
                .freq(&freq)
                .id("T_eq")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(2.0))
                .km(Some(0.9))
                .build();

            assert_eq!(t1, t2);
        }

        #[test]
        fn test_transformer_debug() {
            let freq = test_freq();
            let t = TransformerBuilder::new()
                .freq(&freq)
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(1.0))
                .km(Some(0.9))
                .build();

            let debug_str = format!("{:?}", t);
            assert!(debug_str.contains("Transformer"));
        }

        #[test]
        fn test_transformer_builder_clone() {
            let freq = test_freq();
            let builder1 = TransformerBuilder::new()
                .freq(&freq)
                .id("T_builder")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(Some(2.0))
                .km(Some(0.9));

            let builder2 = builder1.clone();

            let t1 = builder1.build();
            let t2 = builder2.build();

            assert_eq!(t1.id(), t2.id());
        }
    }

    // ============================================================
    // COMPREHENSIVE IDEAL TRANSFORMER TESTS
    // ============================================================

    mod ideal_transformer_comprehensive_tests {
        use super::*;

        // ----------------------------------------------------------
        // IdealTransformer struct direct tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_new_direct() {
            let t = IdealTransformer::new("IT_direct".to_string(), 2.0, [1, 2], c64(50.0, 0.0));

            assert_eq!(t.id(), "IT_direct");
            assert!(approx_eq!(f64, t.val(), 2.0, DEFAULT_MARGIN));
            assert_eq!(t.nodes(), vec![1, 2]);
            assert_eq!(t.z0(), c64(50.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_default() {
            let t = IdealTransformer::default();
            assert_eq!(t.id(), "C0");
            assert!(approx_eq!(f64, t.val(), 1.0, DEFAULT_MARGIN));
            assert_eq!(t.nodes(), vec![1, 2]);
            assert_eq!(t.z0(), c64(50.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_set_val() {
            let mut t = IdealTransformer::default();
            t.set_val(5.0);
            assert!(approx_eq!(f64, t.val(), 5.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_z0_accessor() {
            let t = IdealTransformer::new("IT".to_string(), 1.0, [1, 2], c64(75.0, 0.0));
            assert_eq!(t.z0(), c64(75.0, 0.0));
        }

        // ----------------------------------------------------------
        // IdealTransformer S-parameter matrix tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_c_matrix_n_equals_1() {
            // When n=1, S11 = S22 = 0, S12 = S21 = 1
            let freq = new_frequency(array![1e9], Scale::Base);
            let t = IdealTransformer::new("IT".to_string(), 1.0, [1, 2], c64(50.0, 0.0));

            let c = t.c(&freq);

            // S11 = (1-n^2)/(1+n^2) = 0 when n=1
            assert!(approx_eq!(f64, c[[0, 0]].re, 0.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[0, 0]].im, 0.0, DEFAULT_MARGIN));

            // S22 = (n^2-1)/(1+n^2) = 0 when n=1
            assert!(approx_eq!(f64, c[[1, 1]].re, 0.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[1, 1]].im, 0.0, DEFAULT_MARGIN));

            // S12 = S21 = 2n/(1+n^2) = 1 when n=1
            assert!(approx_eq!(f64, c[[0, 1]].re, 1.0, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[1, 0]].re, 1.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_c_matrix_n_equals_2() {
            // S11 = (1-4)/(1+4) = -3/5 = -0.6
            // S22 = (4-1)/(1+4) = 3/5 = 0.6
            // S12 = S21 = 4/(1+4) = 4/5 = 0.8
            let freq = new_frequency(array![1e9], Scale::Base);
            let t = IdealTransformer::new("IT".to_string(), 2.0, [1, 2], c64(50.0, 0.0));

            let c = t.c(&freq);

            assert!(approx_eq!(f64, c[[0, 0]].re, -0.6, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[1, 1]].re, 0.6, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[0, 1]].re, 0.8, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[1, 0]].re, 0.8, DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_c_matrix_n_equals_0_5() {
            // n = 0.5, n^2 = 0.25
            // S11 = (1-0.25)/(1+0.25) = 0.75/1.25 = 0.6
            // S22 = (0.25-1)/(1+0.25) = -0.75/1.25 = -0.6
            // S12 = S21 = 2*0.5/(1+0.25) = 1/1.25 = 0.8
            let freq = new_frequency(array![1e9], Scale::Base);
            let t = IdealTransformer::new("IT".to_string(), 0.5, [1, 2], c64(50.0, 0.0));

            let c = t.c(&freq);

            assert!(approx_eq!(f64, c[[0, 0]].re, 0.6, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[1, 1]].re, -0.6, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[0, 1]].re, 0.8, DEFAULT_MARGIN));
            assert!(approx_eq!(f64, c[[1, 0]].re, 0.8, DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_c_matrix_symmetry() {
            // S12 should always equal S21 for ideal transformer
            let freq = new_frequency(array![1e9], Scale::Base);
            let ratios = vec![0.25, 0.5, 1.0, 2.0, 4.0, 10.0];

            for n in ratios {
                let t = IdealTransformer::new("IT".to_string(), n, [1, 2], c64(50.0, 0.0));
                let c = t.c(&freq);

                assert_eq!(c[[0, 1]], c[[1, 0]], "S12 should equal S21 for n={}", n);
            }
        }

        #[test]
        fn test_ideal_transformer_c_matrix_imaginary_zero() {
            // All S-parameters should be purely real for ideal transformer
            let freq = new_frequency(array![1e9], Scale::Base);
            let t = IdealTransformer::new("IT".to_string(), 2.5, [1, 2], c64(50.0, 0.0));

            let c = t.c(&freq);

            for j in 0..2 {
                for k in 0..2 {
                    assert!(
                        approx_eq!(f64, c[[j, k]].im, 0.0, DEFAULT_MARGIN),
                        "S[{},{}] imaginary part should be 0",
                        j,
                        k
                    );
                }
            }
        }

        // ----------------------------------------------------------
        // IdealTransformerBuilder tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_builder_new() {
            let builder = IdealTransformerBuilder::new();
            let t = builder.build();

            assert_eq!(t.id(), "T0");
            assert!(approx_eq!(f64, t.val(), 1.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_builder_id() {
            let t = IdealTransformerBuilder::new().id("IT_custom").build();

            assert_eq!(t.id(), "IT_custom");
        }

        #[test]
        fn test_ideal_transformer_builder_n() {
            let t = IdealTransformerBuilder::new().n(3.0).build();

            assert!(approx_eq!(f64, t.val(), 3.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_builder_val() {
            let t = IdealTransformerBuilder::new().val(4.0).build();

            assert!(approx_eq!(f64, t.val(), 4.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_builder_nodes() {
            let t = IdealTransformerBuilder::new().nodes([5, 10]).build();

            assert_eq!(t.nodes(), vec![5, 10]);
        }

        #[test]
        fn test_ideal_transformer_builder_z0() {
            let t = IdealTransformerBuilder::new().z0(c64(75.0, 0.0)).build();

            assert_eq!(t.z0(), c64(75.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_builder_chaining() {
            let t = IdealTransformerBuilder::new()
                .id("IT_chain")
                .n(2.5)
                .nodes([3, 7])
                .z0(c64(100.0, 0.0))
                .build();

            assert_eq!(t.id(), "IT_chain");
            assert!(approx_eq!(f64, t.val(), 2.5, DEFAULT_MARGIN));
            assert_eq!(t.nodes(), vec![3, 7]);
            assert_eq!(t.z0(), c64(100.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_builder_default() {
            let builder = IdealTransformerBuilder::default();
            let t = builder.build();

            assert_eq!(t.id(), "T0");
            assert!(approx_eq!(f64, t.val(), 1.0, DEFAULT_MARGIN));
            assert_eq!(t.nodes(), vec![1, 2]);
            assert_eq!(t.z0(), c64(50.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_builder_clone() {
            let builder1 = IdealTransformerBuilder::new().id("IT_clone").n(2.0);

            let builder2 = builder1.clone();

            let t1 = builder1.build();
            let t2 = builder2.build();

            assert_eq!(t1.id(), t2.id());
            assert!(approx_eq!(f64, t1.val(), t2.val(), DEFAULT_MARGIN));
        }

        // ----------------------------------------------------------
        // IdealTransformer Elem trait tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_elem_trait_id() {
            let t = IdealTransformerBuilder::new().id("IT_elem").build();

            assert_eq!(t.id(), "IT_elem");
        }

        #[test]
        fn test_ideal_transformer_elem_trait_name() {
            let t = IdealTransformerBuilder::new().id("IT_name").build();

            assert_eq!(t.name(), "IT_name");
        }

        #[test]
        fn test_ideal_transformer_elem_trait_elem_type() {
            let t = IdealTransformerBuilder::new().build();

            assert_eq!(t.elem(), ElemType::IdealTransformer);
        }

        #[test]
        fn test_ideal_transformer_elem_trait_nodes() {
            let t = IdealTransformerBuilder::new().nodes([8, 15]).build();

            assert_eq!(t.nodes(), vec![8, 15]);
        }

        #[test]
        fn test_ideal_transformer_elem_trait_set_id() {
            let mut t = IdealTransformerBuilder::new().build();

            t.set_id("IT_modified");
            assert_eq!(t.id(), "IT_modified");
        }

        #[test]
        fn test_ideal_transformer_elem_trait_z() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let n = 3.0;
            let t = IdealTransformerBuilder::new().n(n).build();

            let z = t.z(&freq);
            // z = n^2
            assert!(approx_eq!(f64, z.re, n.powi(2), DEFAULT_MARGIN));
            assert!(approx_eq!(f64, z.im, 0.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_elem_trait_z_at() {
            let freq = new_frequency(array![1e9, 2e9, 5e9], Scale::Base);
            let n = 2.0;
            let t = IdealTransformerBuilder::new().n(n).build();

            for i in 0..3 {
                let z = t.z_at(&freq, i);
                assert!(approx_eq!(f64, z.re, n.powi(2), DEFAULT_MARGIN));
            }
        }

        #[test]
        fn test_ideal_transformer_elem_trait_c_at() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let t = IdealTransformerBuilder::new().n(2.0).build();

            let c = t.c(&freq);
            for j in 0..2 {
                for k in 0..2 {
                    let c_at = t.c_at(&freq, j, k);
                    assert_eq!(c_at, c[[j, k]]);
                }
            }
        }

        #[test]
        fn test_ideal_transformer_elem_trait_net() {
            let freq = new_frequency(array![1e9, 2e9, 3e9], Scale::Base);
            let t = IdealTransformerBuilder::new().n(2.0).build();

            let net = t.net(&freq);
            assert_eq!(net.shape(), (3, 2, 2));
        }

        #[test]
        fn test_ideal_transformer_net_frequency_independence() {
            // Ideal transformer S-parameters should be frequency independent
            let freq = new_frequency(array![1e6, 1e9, 100e9], Scale::Base);
            let n = 2.0;
            let t = IdealTransformerBuilder::new().n(n).build();

            let net = t.net(&freq);

            let expected_s11 = (1.0 - n.powi(2)) / (1.0 + n.powi(2));
            let expected_s22 = (n.powi(2) - 1.0) / (1.0 + n.powi(2));
            let expected_s12 = 2.0 * n / (1.0 + n.powi(2));

            for i in 0..3 {
                assert!(approx_eq!(
                    f64,
                    net[[i, 0, 0]].re,
                    expected_s11,
                    DEFAULT_MARGIN
                ));
                assert!(approx_eq!(
                    f64,
                    net[[i, 1, 1]].re,
                    expected_s22,
                    DEFAULT_MARGIN
                ));
                assert!(approx_eq!(
                    f64,
                    net[[i, 0, 1]].re,
                    expected_s12,
                    DEFAULT_MARGIN
                ));
                assert!(approx_eq!(
                    f64,
                    net[[i, 1, 0]].re,
                    expected_s12,
                    DEFAULT_MARGIN
                ));
            }
        }

        // ----------------------------------------------------------
        // IdealTransformer impedance transformation tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_impedance_transformation_ratio() {
            let freq = new_frequency(array![1e9], Scale::Base);

            let ratios = vec![0.5, 1.0, 2.0, 4.0, 10.0];

            for n in ratios {
                let t = IdealTransformerBuilder::new().n(n).build();
                let z = t.z(&freq);

                assert!(
                    approx_eq!(f64, z.re, n.powi(2), DEFAULT_MARGIN),
                    "z should be n^2 for n={}",
                    n
                );
            }
        }

        // ----------------------------------------------------------
        // Clone, Debug, PartialEq trait tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_clone() {
            let t1 = IdealTransformerBuilder::new()
                .id("IT_clone")
                .n(2.0)
                .nodes([3, 4])
                .z0(c64(75.0, 0.0))
                .build();

            let t2 = t1.clone();

            assert_eq!(t1.id(), t2.id());
            assert!(approx_eq!(f64, t1.val(), t2.val(), DEFAULT_MARGIN));
            assert_eq!(t1.nodes(), t2.nodes());
            assert_eq!(t1.z0(), t2.z0());
        }

        #[test]
        fn test_ideal_transformer_partial_eq() {
            let t1 = IdealTransformerBuilder::new()
                .id("IT_eq")
                .n(2.0)
                .nodes([1, 2])
                .build();

            let t2 = IdealTransformerBuilder::new()
                .id("IT_eq")
                .n(2.0)
                .nodes([1, 2])
                .build();

            assert_eq!(t1, t2);
        }

        #[test]
        fn test_ideal_transformer_not_equal() {
            let t1 = IdealTransformerBuilder::new().n(2.0).build();

            let t2 = IdealTransformerBuilder::new().n(3.0).build();

            assert_ne!(t1, t2);
        }

        #[test]
        fn test_ideal_transformer_debug() {
            let t = IdealTransformerBuilder::new().n(2.0).build();

            let debug_str = format!("{:?}", t);
            assert!(debug_str.contains("IdealTransformer"));
        }

        // ----------------------------------------------------------
        // Edge case and boundary tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_very_small_n() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let n = 0.01;
            let t = IdealTransformerBuilder::new().n(n).build();

            let z = t.z(&freq);
            assert!(z.is_finite());
            assert!(approx_eq!(f64, z.re, n.powi(2), DEFAULT_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_very_large_n() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let n = 100.0;
            let t = IdealTransformerBuilder::new().n(n).build();

            let z = t.z(&freq);
            assert!(z.is_finite());
            assert!(approx_eq!(f64, z.re, n.powi(2), RELAXED_MARGIN));
        }

        #[test]
        fn test_ideal_transformer_c_matrix_consistency() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let t = IdealTransformerBuilder::new().n(2.0).build();

            // Multiple calls should return the same result
            let c1 = t.c(&freq);
            let c2 = t.c(&freq);

            for j in 0..2 {
                for k in 0..2 {
                    assert_eq!(c1[[j, k]], c2[[j, k]]);
                }
            }
        }
    }
}
