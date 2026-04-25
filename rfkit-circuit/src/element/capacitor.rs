use crate::element::{Elem, ElemType, Lumped, Q, QMode};
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use rfkit_base::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Capacitor<T: RealScalar> {
    id: String,
    cap: ScalarUnitValue<T>,
    q: Option<Q<T>>,
    nodes: [usize; 2],
    c: Points<Complex<T>, Ix2>,
    z0: Complex<T>,
}

impl<T: RealScalar> Capacitor<T> {
    pub fn new(
        id: &str,
        cap: ScalarUnitValue<T>,
        q: Option<Q<T>>,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Capacitor<T> {
        Capacitor {
            id: id.to_string(),
            cap,
            q,
            nodes: nodes,
            c: Capacitor::default_c(),
            z0: z0,
        }
    }

    pub fn builder() -> CapacitorBuilder<T> {
        CapacitorBuilder::new()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn z0(&self) -> Complex<T> {
        self.z0
    }

    pub fn q(&self) -> Option<Q<T>> {
        self.q.clone()
    }

    pub fn scale(&self) -> Scale {
        self.cap.scale()
    }

    pub fn unit(&self) -> Unit {
        self.cap.unit()
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    pub fn set_q(&mut self, val: Option<Q<T>>) {
        self.q = val;
    }

    fn default_c() -> Points<Complex<T>, Ix2> {
        points![
            [
                Complex::new((1.0 / 3.0).into(), T::ZERO),
                Complex::new((2.0 / 3.0).into(), T::ZERO)
            ],
            [
                Complex::new((2.0 / 3.0).into(), T::ZERO),
                Complex::new((1.0 / 3.0).into(), T::ZERO)
            ]
        ]
    }
}

impl<T: RealScalar> Default for Capacitor<T> {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            cap: *ScalarUnitValue::default().set_unit(Unit::Farad),
            q: None,
            nodes: [1, 2],
            c: Self::default_c(),
            z0: Complex::new(50.0.into(), T::ZERO),
        }
    }
}

impl<T: RealScalar> Elem<T> for Capacitor<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Capacitor
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
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
        Points::from_shape_fn((freq.npts(), 2, 2).into_dimension(), |(_, j, k)| {
            match (j, k) {
                (0, 0) | (1, 1) => Complex::new((1.0 / 3.0).into(), T::ZERO),
                (1, 0) | (0, 1) => Complex::new((2.0 / 3.0).into(), T::ZERO),
                _ => Complex::ZERO,
            }
        })
    }

    fn z_scalar(&self, freq: &ScalarUnitValue<T>) -> Complex<T> {
        let q = self.q.clone();
        let cap = self.cap.val();
        match q.as_ref() {
            Some(q) => {
                let wq = q.wq();
                let fq = q.fq().freq();
                match q.mode() {
                    QMode::ProportionalToFreq => {
                        let qt = q.q() * freq.freq() / fq;
                        Complex::new(qt / (wq * cap), (freq.w() * cap).recip())
                    }
                    QMode::ProportionalToSqrtFreq => {
                        let qt = q.q() * (freq.freq() / fq).sqrt();
                        Complex::new(qt / (wq * cap), (freq.w() * cap).recip())
                    }
                    QMode::Constant => Complex::new(
                        q.q() / (wq * cap) * fq / freq.freq(),
                        (freq.w() * cap).recip(),
                    ),
                    QMode::ProportionalToExp => {
                        let qt = q.q() * (freq.freq() / fq).powf(q.alpha());
                        Complex::new(
                            qt / (wq * cap) * (fq / freq.freq()).powf(-q.alpha() + 1.0),
                            (freq.w() * cap).recip(),
                        )
                    }
                }
            }
            None => Complex::new(T::ZERO, -(freq.w() * cap).recip()),
        }
    }
}

impl<T: RealScalar> Lumped<T> for Capacitor<T> {
    fn val(&self) -> T {
        self.cap.val()
    }

    fn val_scaled(&self) -> T {
        self.cap.val_scaled()
    }

    fn scale(&self) -> Scale {
        self.cap.scale()
    }

    fn unit(&self) -> Unit {
        self.cap.unit()
    }

    fn set_val(&mut self, val: T) {
        self.cap.set_val(&val);
    }

    fn set_val_scaled(&mut self, val: T) {
        self.cap.set_val_scaled(&val);
    }

    fn set_scale(&mut self, scale: Scale) {
        self.cap.set_scale(scale);
    }
}

#[derive(Clone)]
pub struct CapacitorBuilder<T: RealScalar> {
    id: String,
    cap: Option<ScalarUnitValue<T>>,
    q: Option<Q<T>>,
    nodes: Option<[usize; 2]>,
    z0: Complex<T>,
}

impl<T: RealScalar> CapacitorBuilder<T> {
    pub fn new() -> Self {
        CapacitorBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn cap(mut self, cap: &ScalarUnitValue<T>) -> Self {
        self.cap = Some(cap.clone());
        self
    }

    pub fn val(mut self, cap: T) -> Self {
        self.cap = match self.cap {
            Some(mut x) => {
                x.set_val(&cap);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&cap, Scale::Base, Unit::Farad)),
        };
        self
    }

    pub fn val_scaled(mut self, cap: T, scale: Scale) -> Self {
        self.cap = match self.cap {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&cap);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&cap, scale, Unit::Farad)),
        };
        self
    }

    pub fn q(mut self, q: Option<Q<T>>) -> Self {
        self.q = q;
        self
    }

    pub fn nodes(mut self, nodes: [usize; 2]) -> Self {
        self.nodes = Some(nodes);
        self
    }

    pub fn z0(mut self, z0: Complex<T>) -> Self {
        self.z0 = z0;
        self
    }

    pub fn build(self) -> Result<Capacitor<T>, String> {
        let elem = "CapacitorBuilder";
        let cap = self.cap.ok_or(format!("{elem}: cap is required"))?;
        let nodes = self.nodes.ok_or(format!("{elem}: nodes is required"))?;
        Ok(Capacitor {
            id: self.id,
            cap,
            q: self.q,
            nodes,
            c: Capacitor::default_c(),
            z0: self.z0,
        })
    }
}

impl<T: RealScalar> Default for CapacitorBuilder<T> {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            cap: None,
            q: None,
            nodes: None,
            z0: Complex::new(50.0.into(), T::ZERO),
        }
    }
}

#[cfg(test)]
mod element_capacitor_tests {
    use super::*;
    use num_complex::{Complex64, ComplexFloat, c64};
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
    fn element_capacitor() {
        let freq_unitval = ScalarUnitValue::new_scaled(&1.0, Scale::Giga, Unit::Hz);
        let freq = ArrayUnitValue::new(&array![freq_unitval.val()], freq_unitval.scale(), Unit::Hz);
        let val_scaled = 1.0;
        let scale = Scale::Pico;
        let unitval = ScalarUnitValue::new_scaled(&val_scaled, scale, Unit::Farad);
        let exemplar = Capacitor {
            id: "C1".to_string(),
            cap: unitval,
            q: None,
            nodes: [1, 2],
            c: Capacitor::default_c(),
            z0: c64(50.0, 0.0),
        };
        let exemplar_z = (Complex64::I * 2.0 * PI * freq.freq()[0] * unitval.val()).recip();
        let calc = CapacitorBuilder::new()
            .val_scaled(val_scaled, scale)
            .nodes([1, 2])
            .id("C1")
            .build()
            .unwrap();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.scale(), &calc.scale());
        assert_eq!(&Unit::Farad, &calc.unit());
        comp_pts_ix3(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        exemplar_z.assert_approx_eq(&calc.z(&freq_unitval), margin, "calc.z()", "0");
    }

    mod capacitor_tests {
        use super::*;

        #[test]
        fn test_capacitor_builder_default() {
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(cap.id(), "C0");
            assert_eq!(cap.nodes(), vec![1, 2]);
            assert_eq!(cap.unit(), Unit::Farad);
        }

        #[test]
        fn test_capacitor_builder_with_all_parameters() {
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .id("C1")
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 3])
                .z0(c64(75.0, 0.0))
                .build()
                .unwrap();

            assert_eq!(cap.id(), "C1");
            assert_eq!(cap.nodes(), vec![1, 3]);
            assert_eq!(cap.val_scaled(), 10.0);
            assert_eq!(cap.scale(), Scale::Pico);
        }

        #[test]
        fn test_capacitor_impedance_calculation() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = cap.z(&freq);
            let expected_z = array![(Complex64::I * 2.0 * PI * 1e9 * 1e-12).recip()];

            z.assert_approx_eq(
                &expected_z,
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
        fn test_capacitor_c_matrix_structure() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c_matrix = cap.c(&freq);

            // Check diagonal and off-diagonal elements
            c_matrix[[0, 0, 0]].re.assert_approx_eq(
                &(1.0 / 3.0),
                DEFAULT_MARGIN,
                "c_matrix",
                "[0,0]",
            );
            c_matrix[[0, 1, 1]].re.assert_approx_eq(
                &(1.0 / 3.0),
                DEFAULT_MARGIN,
                "c_matrix",
                "[1,1]",
            );
            c_matrix[[0, 0, 1]].re.assert_approx_eq(
                &(2.0 / 3.0),
                DEFAULT_MARGIN,
                "c_matrix",
                "[0,1]",
            );
            c_matrix[[0, 1, 0]].re.assert_approx_eq(
                &(2.0 / 3.0),
                DEFAULT_MARGIN,
                "c_matrix",
                "[1,0]",
            );
        }

        #[test]
        fn test_capacitor_multiple_frequencies() {
            let freqs = array![1e9, 2e9, 5e9, 10e9];
            let freq = ArrayUnitValue::new(&freqs, Scale::Base, Unit::Hz);
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            for i in 0..freqs.len() {
                let z = cap.z(&freq)[i];
                let expected = 1.0 / (Complex64::I * 2.0 * PI * freqs[i] * 1e-12);
                z.assert_approx_eq(&expected, RELAXED_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_capacitor_value_scaling() {
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(cap.val_scaled(), 10.0);
            assert_eq!(cap.val(), 10e-12);
        }

        #[test]
        fn test_capacitor_set_operations() {
            let mut cap: Capacitor<f64> = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            cap.set_id("C_new");
            assert_eq!(cap.id(), "C_new");

            cap.set_val(5e-12);
            assert_eq!(cap.val(), 5e-12);

            cap.set_scale(Scale::Nano);
            assert_eq!(cap.scale(), Scale::Nano);
        }

        #[test]
        fn test_capacitor_elem_type() {
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(cap.elem(), ElemType::Capacitor);
        }

        #[test]
        fn test_capacitor_net_matrix() {
            let freq = ArrayUnitValue::new(&array![1e9, 2e9], Scale::Base, Unit::Hz);
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let net = cap.net(&freq);

            assert_eq!(net.shape(), (2, 2, 2)); // [npts, 2, 2]

            // Check structure is consistent across frequencies
            for i in 0..2 {
                net[[i, 0, 0]].re.assert_approx_eq(
                    &(1.0 / 3.0),
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},0,0]").as_str(),
                );
                net[[i, 1, 1]].re.assert_approx_eq(
                    &(1.0 / 3.0),
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},1,1]").as_str(),
                );
                net[[i, 0, 1]].re.assert_approx_eq(
                    &(2.0 / 3.0),
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},0,1]").as_str(),
                );
            }
        }
    }
}
