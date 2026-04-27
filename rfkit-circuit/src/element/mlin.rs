use super::*;
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use rfkit_base::prelude::*;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct Mlin<T: RealScalar> {
    id: String,
    width: ScalarUnitValue<T>,
    length: ScalarUnitValue<T>,
    sub: Msub<T>,
    nodes: [usize; 2],
}

impl<T: RealScalar> Mlin<T> {
    pub fn new(
        id: String,
        width: &ScalarUnitValue<T>,
        length: &ScalarUnitValue<T>,
        sub: &Msub<T>,
        nodes: [usize; 2],
    ) -> Mlin<T> {
        Mlin {
            id,
            width: width.clone(),
            length: length.clone(),
            sub: sub.clone(),
            nodes,
        }
    }

    pub fn builder() -> MlinBuilder<T> {
        MlinBuilder::new()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn width(&self) -> T {
        self.width.val()
    }

    pub fn width_uv(&self) -> ScalarUnitValue<T> {
        self.width
    }

    pub fn length(&self) -> T {
        self.length.val()
    }

    pub fn length_uv(&self) -> ScalarUnitValue<T> {
        self.length
    }

    pub fn sub(&self) -> &Msub<T> {
        &self.sub
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }
}

crate::define_mlin_calcs!(Mlin);

impl<T: RealScalar> Default for Mlin<T> {
    fn default() -> Self {
        Self {
            id: "ML0".to_string(),
            width: *ScalarUnitValue::default().set_unit(Unit::Meter),
            length: *ScalarUnitValue::default().set_unit(Unit::Meter),
            sub: Msub::default(),
            nodes: [1, 2],
        }
    }
}

impl<T: RealScalar> Elem<T> for Mlin<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Mlin
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    fn c<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        let gamma = freq.map_scalar_to_vec(|f| Complex::new(self.alpha(f), self.beta(f)));
        Points::from_shape_fn((freq.npts(), 2, 2), |dim| match dim {
            (i, 0, 1) | (i, 1, 0) => mlin_exp(self.length, gamma[i]).exp(),
            _ => Complex::ZERO,
        })
    }

    fn net<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points::zeros((freq.npts(), 2, 2).into_dimension())
    }

    fn z_scalar(&self, freq: &ScalarUnitValue<T>) -> Complex<T> {
        Complex::new(self.z0(freq), T::ZERO)
    }
}

impl<T: RealScalar> Distributed<T> for Mlin<T> {
    fn width(&self) -> T {
        self.width.val()
    }

    fn val(&self) -> T {
        self.length.val()
    }

    fn set_width_val(&mut self, val: T) {
        self.width.set_val(&val);
    }

    fn set_width_unit(&mut self, unit: Unit) {
        self.width.set_unit(unit);
    }

    fn set_length_val(&mut self, val: T) {
        self.length.set_val(&val);
    }

    fn set_length_unit(&mut self, unit: Unit) {
        self.length.set_unit(unit);
    }

    fn length<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|_| self.length.val())
    }

    fn gamma<U: FreqValue<T>>(&self, freq: &U) -> U::COutput {
        freq.map_scalar_to_complex(|f| Complex::new(self.alpha(f), self.beta(f)))
    }

    fn er<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        self.er_eff(freq)
    }
}

impl<T: RealScalar> fmt::Display for Mlin<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mlin\n\tid:\t{}\n\twidth:\t{}\n\tlength:\t{}\n\tsub:\t{}\n\tnodes:\t{:#?}\n",
            self.id, self.width, self.length, self.sub, self.nodes
        )
    }
}

pub type MlinBuilder<T> = ElementBuilder<T, MlinSpec, ConcreteElement, 2>;
pub type MlinElementBuilder<T> = ElementBuilder<T, MlinSpec, TopLevelElement, 2>;

#[derive(Clone, Copy, Debug, Default)]
pub struct MlinSpec;

#[derive(Clone, Debug)]
pub struct MlinParams<T: RealScalar> {
    width: Option<ScalarUnitValue<T>>,
    length: Option<ScalarUnitValue<T>>,
    gamma: Option<Complex<T>>,
    sub: Option<Msub<T>>,
    z0: Option<T>,
    freq: Option<ScalarUnitValue<T>>,
}

impl<T: RealScalar> Default for MlinParams<T> {
    fn default() -> Self {
        Self {
            width: None,
            length: None,
            gamma: None,
            sub: None,
            z0: None,
            freq: None,
        }
    }
}

impl<T: RealScalar> ElementSpec<T, 2> for MlinSpec {
    type Params = MlinParams<T>;
    type Concrete = Mlin<T>;

    const NAME: &'static str = "MlinBuilder";
    const DEFAULT_ID: &'static str = "ML0";

    fn build_concrete(
        id: String,
        params: Self::Params,
        nodes: [usize; 2],
        _z0: Complex<T>,
    ) -> Result<Self::Concrete, String> {
        let elem = <Self as ElementSpec<T, 2>>::NAME;
        let length = params.length.ok_or(format!("{elem}: length is required"))?;
        let sub = params.sub.ok_or(format!("{elem}: sub is required"))?;
        let width = params.width.ok_or(format!("{elem}: width is required"))?;
        Ok(Mlin {
            id,
            width,
            length,
            sub,
            nodes,
        })
    }
}

impl<T, M> ElementBuilder<T, MlinSpec, M, 2>
where
    T: RealScalar,
    M: ElementBuildMode<T, Mlin<T>>,
{
    pub fn width(mut self, val: &ScalarUnitValue<T>) -> Self {
        self.params.width = Some(val.clone());
        self
    }

    pub fn width_val(mut self, val: T) -> Self {
        self.params.width = match self.params.width {
            Some(mut x) => {
                x.set_val(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&val, Scale::Base, Unit::Meter)),
        };
        self
    }

    pub fn width_val_scaled(mut self, val: T, scale: Scale) -> Self {
        self.params.width = match self.params.width {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&val, scale, Unit::Meter)),
        };
        self
    }

    pub fn width_scale(mut self, val: Scale) -> Self {
        self.params.width = match self.params.width {
            Some(mut x) => {
                x.set_scale(val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&T::ONE, val, Unit::Meter)),
        };
        self
    }

    pub fn width_unit(mut self, val: Unit) -> Self {
        self.params.width = match self.params.width {
            Some(mut x) => {
                x.set_unit(val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&T::ONE, Scale::Base, val)),
        };
        self
    }

    pub fn length(mut self, val: &ScalarUnitValue<T>) -> Self {
        self.params.length = Some(val.clone());
        self
    }

    pub fn length_val(mut self, val: T) -> Self {
        self.params.length = match self.params.length {
            Some(mut x) => {
                x.set_val(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&val, Scale::Base, Unit::Meter)),
        };
        self
    }

    pub fn length_val_scaled(mut self, val: T, scale: Scale) -> Self {
        self.params.length = match self.params.length {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&val, scale, Unit::Meter)),
        };
        self
    }

    pub fn length_scale(mut self, val: Scale) -> Self {
        self.params.length = match self.params.length {
            Some(mut x) => {
                x.set_scale(val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&T::ONE, val, Unit::Meter)),
        };
        self
    }

    pub fn length_unit(mut self, val: Unit) -> Self {
        self.params.length = match self.params.length {
            Some(mut x) => {
                x.set_unit(val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&T::ONE, Scale::Base, val)),
        };
        self
    }

    /// Provide gamma value in line
    pub fn gamma(mut self, val: Complex<T>) -> Self {
        self.params.gamma = Some(val);
        self
    }

    pub fn sub(mut self, val: &Msub<T>) -> Self {
        self.params.sub = Some(val.clone());
        self
    }

    pub fn z0(mut self, z0: T, freq: ScalarUnitValue<T>) -> Self {
        self.params.z0 = Some(z0);
        self.params.freq = Some(freq.clone());
        self
    }
}

#[cfg(test)]
mod element_mlin_tests {
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
    fn element_mlin1() {
        let freq_unitval: ScalarUnitValue<f64> = UnitValueBuilder::new()
            .val_scaled(&1.0, Scale::Giga)
            .unit(Unit::Hz)
            .build()
            .unwrap();
        let freq = ArrayUnitValue::new_freq(&array![freq_unitval.val()], freq_unitval.scale());
        let sub: Msub<f64> = MsubBuilder::new()
            .id("Msub0")
            .er(12.4)
            .tand(0.0004)
            .height(
                &UnitValueBuilder::new()
                    .val(&25e-6)
                    .scale(Scale::Micro)
                    .unit(Unit::Meter)
                    .build()
                    .unwrap(),
            )
            .thickness(
                &UnitValueBuilder::new()
                    .val(&0.77e-6)
                    .scale(Scale::Micro)
                    .unit(Unit::Meter)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        let width_val = 5.915e-6;
        let length_val = 0.25;
        let exemplar = Mlin {
            id: "ML1".to_string(),
            width: UnitValueBuilder::new()
                .val(&width_val)
                .scale(Scale::Micro)
                .unit(Unit::Meter)
                .build()
                .unwrap(),
            length: UnitValueBuilder::new()
                .val(&length_val)
                .unit(Unit::Lambda)
                .build()
                .unwrap(),
            sub: sub.clone(),
            nodes: [1, 2],
        };
        let exemplar_z = c64(2_f64.sqrt() * 50.0, 0.0);
        let calc = MlinBuilder::<f64>::new()
            .width_val(width_val)
            .width_scale(Scale::Micro)
            .width_unit(Unit::Meter)
            .length_val(length_val)
            .length_unit(Unit::Lambda)
            .sub(&sub)
            .nodes([1, 2])
            .id("ML1")
            .build()
            .unwrap();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        assert_eq!(&exemplar.gamma(&freq), &calc.gamma(&freq));
        assert_eq!(&exemplar.er(&freq), &calc.er(&freq));
        comp_pts_ix3(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");

        let margin = NumMargin {
            epsilon: 1e-4,
            relative: 1e-4,
            ulps: 10,
        };
        calc.z(&freq_unitval)
            .assert_approx_eq(&exemplar_z, margin, "calc.z()", "0");
        calc.z0(&freq_unitval)
            .assert_approx_eq(&exemplar_z.re, margin, "calc.z0()", "0");
    }

    // #[test]
    // fn element_mlin2() {
    //     let freq_unitval: ScalarUnitValue<f64> = UnitValueBuilder::new()
    //         .val_scaled(&1.0, Scale::Giga)
    //         .unit(Unit::Hz)
    //         .build()
    //         .unwrap();
    //     let freq = ArrayUnitValue::new_freq(&array![freq_unitval.val()], freq_unitval.scale());
    //     let sub = MsubBuilder::new()
    //         .id("Msub0")
    //         .er(12.4)
    //         .tand(0.0004)
    //         .height(
    //             &UnitValueBuilder::new()
    //                 .val(&25e-6)
    //                 .scale(Scale::Micro)
    //                 .unit(Unit::Meter)
    //                 .build()
    //                 .unwrap(),
    //         )
    //         .thickness(
    //             &UnitValueBuilder::new()
    //                 .val(&0.77e-6)
    //                 .scale(Scale::Micro)
    //                 .unit(Unit::Meter)
    //                 .build()
    //                 .unwrap(),
    //         )
    //         .build();

    //     let z0 = 2_f64.sqrt() * 50.0;
    //     // let width_val = 5.6758e-6;
    //     let length_val = 0.25;
    //     // let exemplar = Mlin {
    //     //     id: "ML1".to_string(),
    //     //     width: UnitValueBuilder::new()
    //     //         .val(width_val)
    //     //         .scale(Scale::Micro)
    //     //         .unit(Unit::Meter)
    //     //         .build(),
    //     //     length: UnitValueBuilder::new()
    //     //         .val(length_val)
    //     //         .unit(Unit::Lambda)
    //     //         .build(),
    //     //     sub: sub.clone(),
    //     //     nodes: [1, 2],
    //     // };
    //     let exemplar_z = c64(z0, 0.0);
    //     let calc = MlinBuilder::new()
    //         .z0(z0, &freq)
    //         .length_val(length_val)
    //         .length_unit(Unit::Lambda)
    //         .sub(&sub)
    //         .nodes([1, 2])
    //         .id("ML1")
    //         .build();
    //     // let margin = F64Margin::default();

    //     // assert_eq!(&exemplar.id(), &calc.id());
    //     // assert_eq!(&exemplar.scale(), &calc.scale());
    //     // assert_eq!(&exemplar.unit(), &calc.unit());
    //     // assert_eq!(&exemplar.gamma(&freq), &calc.gamma(&freq));
    //     // assert_eq!(&exemplar.er(&freq), &calc.er(&freq));
    //     // comp_pts_ix2(&exemplar.c(&freq), &calc.c(&freq), margin, "calc2.c()");

    //     let margin = NumMargin {
    //         epsilon: 1e-14,
    //         relative: 1e-14,
    //         ulps: 4,
    //     };
    //     calc.z(&freq)
    //         .assert_approx_eq(&exemplar_z, margin, "calc2.z()", "0");
    //     calc.z0(&freq)
    //         .assert_approx_eq(&exemplar_z.re, margin, "calc2.z0()", "0");
    // }

    mod mlin_tests {
        use super::*;

        #[test]
        fn test_mlin_builder_default() {
            let mlin: Mlin<f64> = MlinBuilder::<f64>::new()
                .width_val(1.0)
                .length_val(1.0)
                .nodes([1, 2])
                .sub(&Msub::default())
                .build()
                .unwrap();
            assert_eq!(mlin.id(), "ML0");
            assert_eq!(mlin.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_mlin_with_substrate() {
            let sub: Msub<f64> = MsubBuilder::new()
                .er(12.4)
                .tand(0.0004)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val_scaled(0.77, Scale::Micro)
                .build()
                .unwrap();

            let mlin = MlinBuilder::<f64>::new()
                .id("ML1")
                .width_val_scaled(5.6758, Scale::Micro)
                .length_val_scaled(0.25, Scale::Base)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(mlin.id(), "ML1");
            assert_eq!(mlin.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_mlin_c_matrix_structure() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let sub: Msub<f64> = Msub::default();
            let mlin = MlinBuilder::<f64>::new()
                .sub(&sub)
                .width_val(1.0)
                .length_val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let c_matrix = mlin.c(&freq);

            // Diagonal should be zero for transmission line
            assert_eq!(c_matrix[[0, 0, 0]], Complex64::ZERO);
            assert_eq!(c_matrix[[0, 1, 1]], Complex64::ZERO);

            // Off-diagonal should be exponential terms
            assert_ne!(c_matrix[[0, 0, 1]], Complex64::ZERO);
            assert_ne!(c_matrix[[0, 1, 0]], Complex64::ZERO);
        }

        #[test]
        fn test_mlin_distributed_trait() {
            let sub: Msub<f64> = Msub::default();
            let mlin: Mlin<f64> = MlinBuilder::<f64>::new()
                .width_val_scaled(10.0, Scale::Micro)
                .length_val_scaled(1000.0, Scale::Micro)
                .nodes([1, 2])
                .sub(&sub)
                .build()
                .unwrap();

            mlin.width()
                .assert_approx_eq(&10e-6, NumMargin::<f64>::default(), "width", "");
        }

        #[test]
        fn test_mlin_elem_type() {
            let sub: Msub<f64> = Msub::default();
            let mlin = MlinBuilder::<f64>::new()
                .sub(&sub)
                .width_val(1.0)
                .length_val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(mlin.elem(), ElemType::Mlin);
        }

        #[test]
        fn test_mlin_length_units() {
            let sub: Msub<f64> = Msub::default();
            let mlin = MlinBuilder::<f64>::new()
                .length_val_scaled(1000.0, Scale::Micro)
                .length_unit(Unit::Meter)
                .width_val(1.0)
                .nodes([1, 2])
                .sub(&sub)
                .build()
                .unwrap();

            assert_eq!(mlin.length.unit(), Unit::Meter);
        }
    }
}
