use super::*;
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use rfkit_base::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Inductor<T: RealScalar> {
    id: String,
    ind: ScalarUnitValue<T>,
    q: Option<Q<T>>,
    nodes: [usize; 2],
    c: Points<Complex<T>, Ix2>,
    z0: Complex<T>,
}

impl<T: RealScalar> Inductor<T> {
    pub fn new(
        id: &str,
        ind: ScalarUnitValue<T>,
        q: Option<Q<T>>,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Inductor<T> {
        Inductor {
            id: id.to_string(),
            ind,
            q,
            nodes: nodes,
            c: Inductor::default_c(),
            z0: z0,
        }
    }

    pub fn builder() -> InductorBuilder<T> {
        InductorBuilder::new()
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

impl<T: RealScalar> Default for Inductor<T> {
    fn default() -> Self {
        Self {
            id: "L0".to_string(),
            ind: *ScalarUnitValue::default().set_unit(Unit::Henry),
            q: None,
            nodes: [1, 2],
            c: Inductor::default_c(),
            z0: Complex::new(50.0.into(), T::ZERO),
        }
    }
}

impl<T: RealScalar> Elem<T> for Inductor<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Inductor
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
        let ind = self.ind.val();
        match q.clone() {
            Some(q) => {
                let wq = q.wq();
                let rdc = q.rdc().val();
                let (r, x) = match q.mode() {
                    QMode::ProportionalToFreq => (wq * ind / q.q() + rdc, freq.w() * ind),
                    QMode::ProportionalToSqrtFreq => {
                        let rt1 = wq * ind / q.q() - rdc;
                        let qt1 = wq * ind / rt1;
                        let rac = (freq.w() * wq).sqrt() * ind / qt1;
                        (rdc + rac, rac + freq.w() * ind * (-qt1.recip() + 1.0))
                    }
                    QMode::Constant => {
                        let ra = q.q().recip() * 2.0 * std::f64::consts::PI - rdc / freq.freq();
                        let xa = freq.freq() - rdc / ra;
                        (ra * xa + rdc, freq.w() * ind)
                    }
                    QMode::ProportionalToExp => {
                        let qf = q.q() * (freq.freq() / q.fq().freq()).powf(q.alpha());
                        (freq.w() * ind / qf, freq.w() * ind)
                    }
                };
                Complex::new(r, x)
            }
            _ => Complex::new(T::ZERO, freq.w() * ind),
        }
    }
}

impl<T: RealScalar> Lumped<T> for Inductor<T> {
    fn val(&self) -> T {
        self.ind.val()
    }

    fn val_scaled(&self) -> T {
        self.ind.val_scaled()
    }

    fn scale(&self) -> Scale {
        self.ind.scale()
    }

    fn unit(&self) -> Unit {
        self.ind.unit()
    }

    fn set_val(&mut self, val: T) {
        self.ind.set_val(&val);
    }

    fn set_val_scaled(&mut self, val: T) {
        self.ind.set_val_scaled(&val);
    }

    fn set_scale(&mut self, scale: Scale) {
        self.ind.set_scale(scale);
    }
}

pub type InductorBuilder<T> = ElementBuilder<T, InductorSpec, ConcreteElement, 2>;
pub type InductorElementBuilder<T> = ElementBuilder<T, InductorSpec, TopLevelElement, 2>;

#[derive(Clone, Copy, Debug, Default)]
pub struct InductorSpec;

#[derive(Clone, Debug)]
pub struct InductorParams<T: RealScalar> {
    ind: Option<ScalarUnitValue<T>>,
    q: Option<Q<T>>,
}

impl<T: RealScalar> Default for InductorParams<T> {
    fn default() -> Self {
        Self { ind: None, q: None }
    }
}

impl<T: RealScalar> ElementSpec<T, 2> for InductorSpec {
    type Params = InductorParams<T>;
    type Concrete = Inductor<T>;

    const NAME: &'static str = "InductorBuilder";
    const DEFAULT_ID: &'static str = "L0";

    fn build_concrete(
        id: String,
        params: Self::Params,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Result<Self::Concrete, String> {
        let ind = params
            .ind
            .ok_or_else(|| format!("{}: ind is required", <Self as ElementSpec<T, 2>>::NAME))?;

        Ok(Inductor {
            id,
            ind,
            q: params.q,
            nodes,
            c: Inductor::default_c(),
            z0,
        })
    }
}

impl<T, M> ElementBuilder<T, InductorSpec, M, 2>
where
    T: RealScalar,
    M: ElementBuildMode<T, Inductor<T>>,
{
    pub fn ind(mut self, ind: &ScalarUnitValue<T>) -> Self {
        self.params.ind = Some(ind.clone());
        self
    }

    pub fn val(mut self, ind: T) -> Self {
        self.params.ind = match self.params.ind {
            Some(mut x) => {
                x.set_val(&ind);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&ind, Scale::Base, Unit::Henry)),
        };
        self
    }

    pub fn val_scaled(mut self, ind: T, scale: Scale) -> Self {
        self.params.ind = match self.params.ind {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&ind);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&ind, scale, Unit::Henry)),
        };
        self
    }

    pub fn q(mut self, q: Option<Q<T>>) -> Self {
        self.params.q = q;
        self
    }

    pub fn z0(mut self, z0: Complex<T>) -> Self {
        self.z0 = z0;
        self
    }
}

#[cfg(test)]
mod element_inductor_tests {
    use super::*;
    use num_complex::{Complex64, c64};
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
        let freq_unitval = ScalarUnitValue::new_scaled(&1.0, Scale::Giga, Unit::Hz);
        let freq = ArrayUnitValue::new(&array![freq_unitval.val()], freq_unitval.scale(), Unit::Hz);
        let val_scaled = 1.0;
        let scale = Scale::Nano;
        let unitval = ScalarUnitValue::new_scaled(&val_scaled, scale, Unit::Henry);
        let exemplar = Inductor {
            id: "L1".to_string(),
            ind: unitval,
            q: None,
            nodes: [1, 2],
            c: Inductor::default_c(),
            z0: c64(50.0, 0.0),
        };
        let exemplar_z: Complex64 = Complex64::I * 2.0 * PI * freq.freq()[0] * unitval.val();
        let calc = InductorBuilder::new()
            .val_scaled(val_scaled, scale)
            .nodes([1, 2])
            .id("L1")
            .build()
            .unwrap();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.scale(), &calc.scale());
        assert_eq!(&Unit::Henry, &calc.unit());
        comp_pts_ix3(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        exemplar_z.assert_approx_eq(&calc.z(&freq_unitval), margin, "calc.z()", "0");
    }

    mod inductor_tests {
        use super::*;

        #[test]
        fn test_inductor_builder_default() {
            let ind: Inductor<f64> = InductorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(ind.id(), "L0");
            assert_eq!(ind.nodes(), vec![1, 2]);
            assert_eq!(ind.unit(), Unit::Henry);
        }

        #[test]
        fn test_inductor_impedance_calculation() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let ind: Inductor<f64> = InductorBuilder::new()
                .val_scaled(1.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = ind.z(&freq);
            let expected_z = Complex64::I * 2.0 * PI * 1e9 * 1e-9;

            z.assert_approx_eq(
                &array![expected_z],
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
            let freq = ArrayUnitValue::new(&freqs, Scale::Base, Unit::Hz);
            let ind: Inductor<f64> = InductorBuilder::new()
                .val_scaled(1.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z1 = ind.z(&freq)[0];
            let z2 = ind.z(&freq)[1];

            // At 2x frequency, impedance should be 2x
            z2.im
                .assert_approx_eq(&(z1.im * 2.0), RELAXED_MARGIN, "z2", "");
        }

        #[test]
        fn test_inductor_c_matrix() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let ind: Inductor<f64> = InductorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c_matrix = ind.c(&freq);

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
        }

        #[test]
        fn test_inductor_various_values() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let test_values = vec![
                (1.0, Scale::Nano),
                (10.0, Scale::Nano),
                (1.0, Scale::Micro),
                (100.0, Scale::Pico),
            ];

            for (val, scale) in test_values {
                let ind: Inductor<f64> = InductorBuilder::new()
                    .val_scaled(val, scale)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                let z = ind.z(&freq);
                let actual_val = val.unscale(scale);
                let expected = Complex64::I * 2.0 * PI * 1e9 * actual_val;

                z[0].im
                    .assert_approx_eq(&expected.im, RELAXED_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_inductor_elem_type() {
            let ind: Inductor<f64> = InductorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(ind.elem(), ElemType::Inductor);
        }

        #[test]
        fn test_inductor_net_matrix() {
            let freq = ArrayUnitValue::new(&array![1e9, 5e9], Scale::Base, Unit::Hz);
            let ind: Inductor<f64> = InductorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let net = ind.net(&freq);

            assert_eq!(net.shape(), (2, 2, 2));
        }
    }
}
