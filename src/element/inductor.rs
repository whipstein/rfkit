use crate::{
    element::{Elem, ElemType, Lumped, Q, QMode},
    frequency::{FreqArray, Frequency, new_frequency},
    pts::{Points, Pts},
    scale::Scale,
    unit::{Unit, UnitValue, Unitized},
};
use ndarray::{IntoDimension, prelude::*};
use num::complex::{Complex, Complex64, c64};

#[derive(Clone, Debug, PartialEq)]
pub struct Inductor {
    id: String,
    ind: UnitValue,
    q: Option<Q>,
    nodes: [usize; 2],
    c: Points<Complex64, Ix2>,
    z0: Complex64,
}

impl Inductor {
    pub fn new(
        id: String,
        ind: UnitValue,
        q: Option<Q>,
        nodes: [usize; 2],
        z0: Complex64,
    ) -> Inductor {
        Inductor {
            id: id,
            ind: ind,
            q,
            nodes: nodes,
            c: points![
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: z0,
        }
    }

    pub fn z0(&self) -> Complex64 {
        self.z0
    }

    pub fn q(&self) -> Option<Q> {
        self.q.clone()
    }

    pub fn set_q(&mut self, val: Option<Q>) {
        self.q = val;
    }
}

impl Default for Inductor {
    fn default() -> Self {
        Self {
            id: "L0".to_string(),
            ind: *UnitValue::default().set_unit(Unit::Henry),
            q: None,
            nodes: [1, 2],
            c: points![
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        }
    }
}

impl Elem for Inductor {
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
        ElemType::Inductor
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

    fn z(&self, freq: &Frequency) -> Complex<f64> {
        let w = freq.w_pt(0);
        match self.q.clone() {
            Some(q) => {
                let wq = q.wq_pt(0);
                let rdc = q.rdc().val();
                let ind = self.ind.val();
                let (r, x) = match q.mode() {
                    QMode::ProportionalToFreq => {
                        // let rdcp = rdc.max(0.05 * wq * ind / q.q());
                        // let rq1 = wq * ind / q.q();
                        // let r1 = if rdcp > rq1 {
                        //     rdcp
                        // } else {
                        //     let rq2 = (rq1.powi(2) - rdcp.powi(2)).sqrt();
                        //     let qt = wq * ind / rq2;
                        //     let rac = w * ind / qt;
                        //     (rdcp.powi(2) + rac.powi(2)).sqrt()
                        // };
                        // (r1, w * ind)
                        (wq * ind / q.q() + rdc, w * ind)
                    }
                    QMode::ProportionalToSqrtFreq => {
                        let rt1 = wq * ind / q.q() - rdc;
                        let qt1 = wq * ind / rt1;
                        let rac = (w * wq).sqrt() * ind / qt1;
                        (rdc + rac, rac + w * ind * (1.0 - 1.0 / qt1))
                    }
                    QMode::Constant => {
                        let ra = 2.0 * std::f64::consts::PI / q.q() - rdc / freq.freq(0);
                        let xa = freq.freq(0) - rdc / ra;
                        (ra * xa + rdc, w * ind)
                    }
                    QMode::ProportionalToExp => {
                        let qf = q.q() * (freq.freq(0) / q.fq().freq(0)).powf(q.alpha());
                        (w * ind / qf, w * ind)
                    }
                };
                r + Complex64::I * x
            }
            _ => Complex64::I * w * self.ind.val(),
        }
    }

    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
        let freq_pt = new_frequency(array![freq.freq(i)], freq.scale());
        self.z(&freq_pt).into()
    }

    fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

impl Lumped for Inductor {
    fn val(&self) -> f64 {
        self.ind.val()
    }

    fn set_val(&mut self, val: f64) {
        self.ind.set_val(val);
    }
}

impl Unitized for Inductor {
    fn val_scaled(&self) -> f64 {
        self.ind.val_scaled()
    }

    fn unitval(&self) -> UnitValue {
        self.ind.clone()
    }

    fn scale(&self) -> Scale {
        self.ind.scale()
    }

    fn unit(&self) -> Unit {
        self.ind.unit()
    }

    fn set_val_scaled(&mut self, val: f64) {
        self.ind.set_val_scaled(val);
    }

    fn set_unitval(&mut self, val: UnitValue) {
        self.ind = val;
    }

    fn set_scale(&mut self, scale: Scale) {
        self.ind.set_scale(scale);
    }

    fn set_unit(&mut self, unit: Unit) {
        self.ind.set_unit(unit);
    }
}

#[derive(Clone)]
pub struct InductorBuilder {
    id: String,
    ind: UnitValue,
    q: Option<Q>,
    nodes: [usize; 2],
    z0: Complex64,
}

impl InductorBuilder {
    pub fn new() -> Self {
        InductorBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn ind(mut self, ind: UnitValue) -> Self {
        self.ind = ind;
        self
    }

    pub fn val(mut self, ind: f64) -> Self {
        self.ind.set_val(ind);
        self
    }

    pub fn val_scaled(mut self, ind: f64, scale: Scale) -> Self {
        self.ind.set_scale(scale);
        self.ind.set_val_scaled(ind);
        self
    }

    pub fn q(mut self, q: Option<Q>) -> Self {
        self.q = q;
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

    pub fn build(self) -> Inductor {
        Inductor {
            id: self.id,
            ind: self.ind,
            q: self.q,
            nodes: self.nodes,
            c: points![
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: self.z0,
        }
    }
}

impl Default for InductorBuilder {
    fn default() -> Self {
        Self {
            id: "L0".to_string(),
            ind: *UnitValue::default().set_unit(Unit::Henry),
            q: None,
            nodes: [1, 2],
            z0: c64(50.0, 0.0),
        }
    }
}

#[cfg(test)]
mod element_inductor_tests {
    use super::*;
    use crate::{
        unit::UnitValBuilder,
        util::{ApproxEq, NumMargin, comp_pts_ix2},
    };
    use std::f64::consts::PI;

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
    fn element_inductor() {
        let freq_unitval = UnitValBuilder::new().val_scaled(1.0, Scale::Giga).build();
        let freq = new_frequency(array![freq_unitval.val()], freq_unitval.scale());
        let val_scaled = 1.0;
        let scale = Scale::Nano;
        let unitval = UnitValBuilder::new().val_scaled(val_scaled, scale).build();
        let exemplar = Inductor {
            id: "L1".to_string(),
            ind: unitval,
            q: None,
            nodes: [1, 2],
            c: points![
                [c64(1.0 / 3.0, 0.0), c64(2.0 / 3.0, 0.0)],
                [c64(2.0 / 3.0, 0.0), c64(1.0 / 3.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        };
        let exemplar_z: Complex<f64> = Complex64::I * 2.0 * PI * freq.freq(0) * unitval.val();
        let calc = InductorBuilder::new()
            .val_scaled(val_scaled, scale)
            .nodes([1, 2])
            .id("L1")
            .build();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.scale(), &calc.scale());
        assert_eq!(&Unit::Henry, &calc.unit());
        comp_pts_ix2(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        exemplar_z.assert_approx_eq(calc.z(&freq), margin, "calc.z()", "0");
    }

    mod inductor_tests {
        use super::*;

        #[test]
        fn test_inductor_builder_default() {
            let ind = InductorBuilder::new().build();
            assert_eq!(ind.id(), "L0");
            assert_eq!(ind.nodes(), vec![1, 2]);
            assert_eq!(ind.unit(), Unit::Henry);
        }

        #[test]
        fn test_inductor_impedance_calculation() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let ind = InductorBuilder::new().val_scaled(1.0, Scale::Nano).build();

            let z = ind.z(&freq);
            let expected_z = Complex64::I * 2.0 * PI * 1e9 * 1e-9;

            z.assert_approx_eq(
                expected_z,
                NumMargin {
                    epsilon: 1e-6,
                    relative: 1e-6,
                    ulps: 4,
                },
                "z",
                "",
            );
        }

        #[test]
        fn test_inductor_frequency_proportional() {
            let freqs = array![1e9, 2e9, 5e9];
            let freq = new_frequency(freqs.clone(), Scale::Base);
            let ind = InductorBuilder::new().val_scaled(1.0, Scale::Nano).build();

            let z1 = ind.z_at(&freq, 0);
            let z2 = ind.z_at(&freq, 1);

            // At 2x frequency, impedance should be 2x
            z2.im
                .assert_approx_eq(z1.im * 2.0, RELAXED_MARGIN, "z2", "");
        }

        #[test]
        fn test_inductor_c_matrix() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let ind = InductorBuilder::new().build();
            let c_matrix = ind.c(&freq);

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
        fn test_inductor_various_values() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let test_values = vec![
                (1.0, Scale::Nano),
                (10.0, Scale::Nano),
                (1.0, Scale::Micro),
                (100.0, Scale::Pico),
            ];

            for (val, scale) in test_values {
                let ind = InductorBuilder::new().val_scaled(val, scale).build();

                let z = ind.z(&freq);
                let actual_val = val * scale.multiplier();
                let expected = Complex64::I * 2.0 * PI * 1e9 * actual_val;

                z.im.assert_approx_eq(expected.im, RELAXED_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_inductor_elem_type() {
            let ind = InductorBuilder::new().build();
            assert_eq!(ind.elem(), ElemType::Inductor);
        }

        #[test]
        fn test_inductor_net_matrix() {
            let freq = new_frequency(array![1e9, 5e9], Scale::Base);
            let ind = InductorBuilder::new().build();
            let net = ind.net(&freq);

            assert_eq!(net.shape(), (2, 2, 2));
        }
    }
}
