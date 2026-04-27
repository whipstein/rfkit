use super::*;
use ndarray::{IntoDimension, prelude::*};
use num_complex::{Complex, ComplexFloat};
use rfkit_base::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Mlef<T: RealScalar> {
    id: String,
    width: ScalarUnitValue<T>,
    sub: Msub<T>,
    nodes: [usize; 1],
}

impl<T: RealScalar> Mlef<T> {
    pub fn new(
        id: String,
        width: &ScalarUnitValue<T>,
        sub: &Msub<T>,
        nodes: [usize; 1],
    ) -> Mlef<T> {
        Mlef {
            id,
            width: width.clone(),
            sub: sub.clone(),
            nodes,
        }
    }

    pub fn builder() -> MlefBuilder<T> {
        MlefBuilder::new()
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

    pub fn sub(&self) -> &Msub<T> {
        &self.sub
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    fn delta_l(&self) -> T {
        let h = self.sub.height();
        let ereffdc = self.er_effdc();
        let weff = self.w_eff();
        let ueff = weff / h;

        h * 0.412 * (ereffdc + 0.3) * (ueff + 0.264) / ((ereffdc - 0.258) * (ueff + 0.8))
    }
}

crate::define_mlin_calcs!(Mlef);

impl<T: RealScalar> Default for Mlef<T> {
    fn default() -> Self {
        Self {
            id: "ML0".to_string(),
            width: *ScalarUnitValue::default().set_unit(Unit::Meter),
            sub: Msub::default(),
            nodes: [1],
        }
    }
}

impl<T: RealScalar> Elem<T> for Mlef<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Mlef
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    // todo!("fix c value for true mlef response");
    fn c<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points::from_elem((freq.npts(), 1, 1), Complex::ZERO)
    }

    fn net<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points::zeros((freq.npts(), 2, 2).into_dimension())
    }

    fn z_scalar(&self, freq: &ScalarUnitValue<T>) -> Complex<T> {
        -Complex::<T>::I
            * Complex::from(self.z0(freq))
            * (self.gamma(freq) * self.length(freq)).tan().recip()
    }
}

impl<T: RealScalar> Distributed<T> for Mlef<T> {
    fn width(&self) -> T {
        self.width.val()
    }

    fn val(&self) -> T {
        T::ZERO
    }

    fn set_width_val(&mut self, val: T) {
        self.width.set_val(&val);
    }

    fn set_width_unit(&mut self, unit: Unit) {
        self.width.set_unit(unit);
    }

    fn set_length_val(&mut self, _val: T) {
        ();
    }

    fn set_length_unit(&mut self, _unit: Unit) {
        ();
    }

    fn length<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        let l = self.delta_l();

        freq.map_scalar_to_real(|_| l)
    }

    fn gamma<U: FreqValue<T>>(&self, freq: &U) -> U::COutput {
        freq.map_scalar_to_complex(|f| Complex::new(self.alpha(f), self.beta(f)))
    }

    fn er<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput {
        self.er_eff(freq)
    }
}

pub type MlefBuilder<T> = ElementBuilder<T, MlefSpec, ConcreteElement, 1>;
pub type MlefElementBuilder<T> = ElementBuilder<T, MlefSpec, TopLevelElement, 1>;

#[derive(Clone, Copy, Debug, Default)]
pub struct MlefSpec;

#[derive(Clone, Debug)]
pub struct MlefParams<T: RealScalar> {
    width: Option<ScalarUnitValue<T>>,
    sub: Option<Msub<T>>,
}

impl<T: RealScalar> Default for MlefParams<T> {
    fn default() -> Self {
        Self {
            width: None,
            sub: None,
        }
    }
}

impl<T: RealScalar> ElementSpec<T, 1> for MlefSpec {
    type Params = MlefParams<T>;
    type Concrete = Mlef<T>;

    const NAME: &'static str = "MlefBuilder";
    const DEFAULT_ID: &'static str = "ML0";

    fn build_concrete(
        id: String,
        params: Self::Params,
        nodes: [usize; 1],
        _z0: Complex<T>,
    ) -> Result<Self::Concrete, String> {
        let width = params
            .width
            .ok_or_else(|| format!("{}: width is required", <Self as ElementSpec<T, 1>>::NAME))?;
        let sub = params
            .sub
            .ok_or_else(|| format!("{}: sub is required", <Self as ElementSpec<T, 1>>::NAME))?;

        Ok(Mlef {
            id,
            width,
            sub,
            nodes,
        })
    }
}

impl<T, M> ElementBuilder<T, MlefSpec, M, 1>
where
    T: RealScalar,
    M: ElementBuildMode<T, Mlef<T>>,
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

    pub fn sub(mut self, val: &Msub<T>) -> Self {
        self.params.sub = Some(val.clone());
        self
    }
}

#[cfg(test)]
mod element_mlef_tests {
    use super::*;

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

    // #[test]
    // fn element_mlef() {
    //     let freq_unitval = UnitValBuilder::new()
    //         .val_scaled(1.0, Scale::Giga)
    //         .unit(Unit::Hz)
    //         .build();
    //     let freq = new_frequency(array![freq_unitval.val()], freq_unitval.scale());
    //     let sub = MsubBuilder::new()
    //         .id("Msub0")
    //         .er(12.4)
    //         .tand(0.0004)
    //         .height(
    //             UnitValBuilder::new()
    //                 .val(25e-6)
    //                 .scale(Scale::Micro)
    //                 .unit(Unit::Meter)
    //                 .build(),
    //         )
    //         .thickness(
    //             UnitValBuilder::new()
    //                 .val(0.77e-6)
    //                 .scale(Scale::Micro)
    //                 .unit(Unit::Meter)
    //                 .build(),
    //         )
    //         .build();

    //     let width_val = 5.8736e-6;
    //     let length_val = 0.25;
    //     let exemplar = Mlef {
    //         id: "ML1".to_string(),
    //         width: UnitValBuilder::new()
    //             .val(width_val)
    //             .scale(Scale::Micro)
    //             .unit(Unit::Meter)
    //             .build(),
    //         length: UnitValBuilder::new()
    //             .val(length_val)
    //             .unit(Unit::Lambda)
    //             .build(),
    //         sub: sub.clone(),
    //         nodes: [1, 2],
    //     };
    //     let exemplar_z = c64(2_f64.sqrt() * 50.0, 0.0);
    //     let calc = MlefBuilder::new()
    //         .width_val(width_val)
    //         .width_scale(Scale::Micro)
    //         .width_unit(Unit::Meter)
    //         .length_val(length_val)
    //         .length_unit(Unit::Lambda)
    //         .sub(&sub)
    //         .nodes([1, 2])
    //         .id("ML1")
    //         .build();
    //     let margin = F64Margin::default();

    //     assert_eq!(&exemplar.id(), &calc.id());
    //     assert_eq!(&exemplar.scale(), &calc.scale());
    //     assert_eq!(&exemplar.unit(), &calc.unit());
    //     assert_eq!(&exemplar.gamma(&freq), &calc.gamma(&freq));
    //     assert_eq!(&exemplar.er(&freq), &calc.er(&freq));
    //     comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");

    //     let margin = F64Margin {
    //         epsilon: 1e-4,
    //         ulps: 10,
    //     };
    //     comp_c64(&exemplar_z, &calc.z(&freq), margin, "calc.z()", "0");
    //     comp_f64(&exemplar_z.re, &calc.z0(&freq), margin, "calc.z0()", "0");
    // }

    mod mlef_tests {
        use super::*;
        use num_complex::Complex64;

        #[test]
        fn test_mlef_builder_default() {
            let mlef = MlefBuilder::new()
                .width_val(1.0)
                .sub(&Msub::default())
                .nodes([1])
                .build()
                .unwrap();
            assert_eq!(mlef.id(), "ML0");
            assert_eq!(mlef.nodes(), vec![1]);
        }

        #[test]
        fn test_mlef_end_effect_extends_length() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val_scaled(0.77, Scale::Micro)
                .build()
                .unwrap();

            let mlef = MlefBuilder::new()
                .width_val_scaled(10.0, Scale::Micro)
                .sub(&sub)
                .nodes([1])
                .build()
                .unwrap();

            let physical_length = mlef.val();
            let effective_length = mlef.length(&freq)[0];

            // Effective length should be greater due to end effect
            assert!(effective_length > physical_length);
        }

        #[test]
        fn test_mlef_with_substrate() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val(0.0)
                .build()
                .unwrap();

            let mlef = MlefBuilder::new()
                .id("ML1")
                .width_val_scaled(5.8736, Scale::Micro)
                .sub(&sub)
                .nodes([1])
                .build()
                .unwrap();

            assert_eq!(mlef.id(), "ML1");
            assert_eq!(mlef.nodes(), vec![1]);
        }

        #[test]
        fn test_mlef_c_matrix() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val(0.0)
                .build()
                .unwrap();
            let mlef = MlefBuilder::new()
                .width_val_scaled(10.0, Scale::Micro)
                .sub(&sub)
                .nodes([1])
                .build()
                .unwrap();

            let c_matrix = mlef.c(&freq);
            c_matrix[[0, 0, 0]].assert_approx_eq(
                &Complex64::ZERO,
                NumMargin::default(),
                "c_matrix",
                "(0,0,0)",
            );
        }

        #[test]
        fn test_mlef_elem_type() {
            let sub = Msub::default();
            let mlef = MlefBuilder::new()
                .width_val(1.0)
                .sub(&sub)
                .nodes([1])
                .build()
                .unwrap();
            assert_eq!(mlef.elem(), ElemType::Mlef);
        }
    }
}
