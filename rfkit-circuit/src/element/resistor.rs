use super::*;
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use rfkit_base::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Resistor<T: RealScalar> {
    id: String,
    res: ScalarUnitValue<T>,
    nodes: [usize; 2],
    c: Points<Complex<T>, Ix2>,
    z0: Complex<T>,
}

impl<T: RealScalar> Resistor<T> {
    pub fn new(
        id: &str,
        res: ScalarUnitValue<T>,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Resistor<T> {
        Resistor {
            id: id.to_string(),
            res,
            nodes: nodes,
            c: Resistor::default_c(),
            z0: z0,
        }
    }

    pub fn builder() -> ResistorBuilder<T> {
        ResistorBuilder::new()
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

impl<T: RealScalar> Default for Resistor<T> {
    fn default() -> Self {
        Self {
            id: "R0".to_string(),
            res: *ScalarUnitValue::default().set_unit(Unit::Ohm),
            nodes: [1, 2],
            c: Resistor::default_c(),
            z0: Complex::new(50.0.into(), T::ZERO),
        }
    }
}

impl<T: RealScalar> Elem<T> for Resistor<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Resistor
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

    fn z_scalar(&self, _freq: &ScalarUnitValue<T>) -> Complex<T> {
        Complex::new(self.res.val(), T::ZERO)
    }
}

impl<T: RealScalar> Lumped<T> for Resistor<T> {
    fn val(&self) -> T {
        self.res.val()
    }

    fn val_scaled(&self) -> T {
        self.res.val_scaled()
    }

    fn scale(&self) -> Scale {
        self.res.scale()
    }

    fn unit(&self) -> Unit {
        self.res.unit()
    }

    fn set_val(&mut self, val: T) {
        self.res.set_val(&val);
    }

    fn set_val_scaled(&mut self, val: T) {
        self.res.set_val_scaled(&val);
    }

    fn set_scale(&mut self, scale: Scale) {
        self.res.set_scale(scale);
    }
}

pub type ResistorBuilder<T> = ElementBuilder<T, ResistorSpec, ConcreteElement, 2>;
pub type ResistorElementBuilder<T> = ElementBuilder<T, ResistorSpec, TopLevelElement, 2>;

#[derive(Clone, Copy, Debug, Default)]
pub struct ResistorSpec;

#[derive(Clone, Debug)]
pub struct ResistorParams<T: RealScalar> {
    res: Option<ScalarUnitValue<T>>,
}

impl<T: RealScalar> Default for ResistorParams<T> {
    fn default() -> Self {
        Self { res: None }
    }
}

impl<T: RealScalar> ElementSpec<T, 2> for ResistorSpec {
    type Params = ResistorParams<T>;
    type Concrete = Resistor<T>;

    const NAME: &'static str = "ResistorBuilder";
    const DEFAULT_ID: &'static str = "R0";

    fn build_concrete(
        id: String,
        params: Self::Params,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Result<Self::Concrete, String> {
        let res = params
            .res
            .ok_or_else(|| format!("{}: res is required", <Self as ElementSpec<T, 2>>::NAME))?;

        Ok(Resistor {
            id,
            res,
            nodes,
            c: Resistor::default_c(),
            z0,
        })
    }
}

impl<T, M> ElementBuilder<T, ResistorSpec, M, 2>
where
    T: RealScalar,
    M: ElementBuildMode<T, Resistor<T>>,
{
    pub fn res(mut self, res: &ScalarUnitValue<T>) -> Self {
        self.params.res = Some(res.clone());
        self
    }

    pub fn val(mut self, res: T) -> Self {
        self.params.res = match self.params.res {
            Some(mut x) => {
                x.set_val(&res);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&res, Scale::Base, Unit::Ohm)),
        };
        self
    }

    pub fn val_scaled(mut self, res: T, scale: Scale) -> Self {
        self.params.res = match self.params.res {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&res);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&res, scale, Unit::Ohm)),
        };
        self
    }

    pub fn z0(mut self, z0: Complex<T>) -> Self {
        self.z0 = z0;
        self
    }
}

#[cfg(test)]
mod element_resistor_tests {
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

    #[test]
    fn element_resistor() {
        let freq_unitval = ScalarUnitValue::new_scaled(&1.0, Scale::Giga, Unit::Hz);
        let freq = ArrayUnitValue::new(&array![freq_unitval.val()], freq_unitval.scale(), Unit::Hz);
        let val_scaled = 20.0;
        let scale = Scale::Base;
        let unitval = ScalarUnitValue::new_scaled(&val_scaled, scale, Unit::Ohm);
        let exemplar = Resistor {
            id: "R1".to_string(),
            res: unitval,
            nodes: [1, 2],
            c: Resistor::default_c(),
            z0: c64(50.0, 0.0),
        };
        let exemplar_z: Complex64 = unitval.val().into();
        let calc = ResistorBuilder::new()
            .val_scaled(val_scaled, scale)
            .nodes([1, 2])
            .id("R1")
            .build()
            .unwrap();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.scale(), &calc.scale());
        assert_eq!(&Unit::Ohm, &calc.unit());
        comp_pts_ix3(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        exemplar_z.assert_approx_eq(&calc.z(&freq_unitval), margin, "calc.z()", "0");
    }

    mod resistor_tests {
        use super::*;

        #[test]
        fn test_resistor_builder_default() {
            let res: Resistor<f64> = ResistorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(res.id(), "R0");
            assert_eq!(res.nodes(), vec![1, 2]);
            assert_eq!(res.unit(), Unit::Ohm);
        }

        #[test]
        fn test_resistor_frequency_independent() {
            let freqs = array![1e6, 1e9, 10e9, 100e9];
            let freq = ArrayUnitValue::new(&freqs.clone(), Scale::Base, Unit::Hz);
            let res: Resistor<f64> = ResistorBuilder::new()
                .val(50.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            // Resistor impedance should be constant across all frequencies
            for i in 0..freqs.len() {
                let z = res.z(&freq)[i];
                z.assert_approx_eq(&c64(50.0, 0.0), DEFAULT_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_resistor_impedance_real_only() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let values = vec![1.0, 10.0, 50.0, 100.0, 1000.0];

            for val in values {
                let res: Resistor<f64> = ResistorBuilder::new()
                    .val(val)
                    .nodes([1, 2])
                    .build()
                    .unwrap();
                let z = res.z(&freq);
                z.assert_approx_eq(&array![c64(val, 0.0)], DEFAULT_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_resistor_c_matrix() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let res: Resistor<f64> = ResistorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c_matrix = res.c(&freq);

            // Same structure as capacitor
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
        fn test_resistor_scaling() {
            let res: Resistor<f64> = ResistorBuilder::new()
                .val_scaled(1.0, Scale::Kilo)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(res.val_scaled(), 1.0);
            assert_eq!(res.val(), 1000.0);
            assert_eq!(res.scale(), Scale::Kilo);
        }

        #[test]
        fn test_resistor_node_configuration() {
            let nodes = vec![[0, 1], [1, 2], [2, 5], [10, 20]];

            for node_pair in nodes {
                let res: Resistor<f64> = ResistorBuilder::new()
                    .val(1.0)
                    .nodes(node_pair)
                    .build()
                    .unwrap();
                assert_eq!(res.nodes(), vec![node_pair[0], node_pair[1]]);
            }
        }

        #[test]
        fn test_resistor_elem_type() {
            let res: Resistor<f64> = ResistorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(res.elem(), ElemType::Resistor);
        }
    }
}
