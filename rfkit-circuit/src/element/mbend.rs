use super::*;
use ndarray::{IntoDimension, prelude::*};
use num_complex::{Complex, ComplexFloat};

#[derive(Clone, Debug, PartialEq)]
pub struct Mbend<T: RealScalar> {
    id: String,
    width1: ScalarUnitValue<T>,
    width2: ScalarUnitValue<T>,
    miter: T,
    sub: Msub<T>,
    nodes: [usize; 2],
}

impl<T: RealScalar> Mbend<T> {
    pub fn new(
        id: String,
        width1: &ScalarUnitValue<T>,
        width2: &ScalarUnitValue<T>,
        miter: T,
        sub: &Msub<T>,
        nodes: [usize; 2],
    ) -> Mbend<T> {
        Mbend {
            id,
            width1: width1.clone(),
            width2: width2.clone(),
            miter,
            sub: sub.clone(),
            nodes,
        }
    }

    pub fn builder() -> MsubBuilder<T> {
        MsubBuilder::new()
    }

    pub fn id(&self) -> String {
        self.id.clone()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn width1(&self) -> T {
        self.width1.val()
    }

    pub fn width1_uv(&self) -> ScalarUnitValue<T> {
        self.width1
    }

    pub fn width2(&self) -> T {
        self.width2.val()
    }

    pub fn width2_uv(&self) -> ScalarUnitValue<T> {
        self.width2
    }

    pub fn miter(&self) -> T {
        self.miter
    }

    pub fn sub(&self) -> &Msub<T> {
        &self.sub
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    /// Calculate L & C values for Tee network equivalent
    /// Returns (L, C)
    pub fn calc_lc(&self) -> (ScalarUnitValue<T>, ScalarUnitValue<T>) {
        let w = self.width.val();
        let h = self.sub.height().val();
        let er = self.sub.er();
        let (l, c) = match self.miter {
            true => (
                (-((w / h).powf(0.947.into()) * -0.177).exp() * 1.062 + 1.0) * h * 441.2712,
                w / h * (er * 7.6 + 3.8 + w / h * (er * 3.93 + 0.62)) * h,
            ),
            false => (
                (-((w / h).powf(1.39.into()) * -0.18).exp() * 1.35 + 1.0) * h * 220.6356,
                w / h * (er * 2.6 + 5.64 + w / h * (er * 10.35 + 2.5)) * h,
            ),
        };
        (
            ScalarUnitValue::builder()
                .val_scaled(&l, Scale::Nano)
                .unit(Unit::Henry)
                .build()
                .unwrap(),
            ScalarUnitValue::builder()
                .val_scaled(&c, Scale::Pico)
                .unit(Unit::Farad)
                .build()
                .unwrap(),
        )
    }
}

define_mlin_calcs!(Mbend);

impl<T: RealScalar> Default for Mbend<T> {
    fn default() -> Self {
        Self {
            id: "MB0".to_string(),
            width1: *ScalarUnitValue::default().set_unit(Unit::Meter),
            width2: *ScalarUnitValue::default().set_unit(Unit::Meter),
            miter: T::ZERO,
            sub: Msub::default(),
            nodes: [1, 2],
        }
    }
}

impl<T: RealScalar> Elem<T> for Mbend<T> {
    fn elem(&self) -> ElemType {
        ElemType::Mbend
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }
}

impl<T, U> ElemCalc<T, U> for Mbend<T>
where
    T: RealScalar,
    U: UnitValue<T> + Frequency<T> + MapScalar<T>,
{
    fn c(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        let (l_unitval, c_unitval) = self.calc_lc();
        let l = l_unitval.val();
        let c = c_unitval.val();
        let z0 = freq.map_scalar_to_vec(|f| self.z0(f));
        let w = freq.map_scalar_to_vec(|f| f.w());

        Points::from_shape_fn((freq.npts(), 2, 2), |dim| {
            let z1 = Complex::I * w[dim.0] * l;
            let z3 = z1;
            let z2 = (Complex::I * w[dim.0] * c).recip();
            let p = z1 - z2;
            let q = z1 + z2 + Complex::new(2.0.into(), T::ZERO) * z3;
            let t = z1 * z2 + z2 * z3 + z1 * z3;
            let d = q * z0[dim.0] + t + z0[dim.0].powi(2);

            match dim {
                (_, 0, 0) => -d.recip() * z0[dim.0].powi(2) + p * z0[dim.0] + t,
                (_, 0, 1) => d.recip() * Complex::new(2.0.into(), T::ZERO) * z0[dim.0] * z3,
                (_, 1, 0) => d.recip() * Complex::new(2.0.into(), T::ZERO) * z0[dim.0] * z3,
                (_, 1, 1) => -d.recip() * z0[dim.0].powi(2) - p * z0[dim.0] + t,
                _ => Complex::ZERO,
            }
        })
    }

    fn net(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points::zeros((freq.npts(), 2, 2).into_dimension())
    }

    fn z_scalar(&self, freq: &ScalarUnitValue<T>) -> Complex<T> {
        Complex::from(self.z0(freq))
    }
}

impl<T: RealScalar> Distributed<T> for Mbend<T> {
    fn width(&self) -> T {
        self.width1()
    }

    fn val(&self) -> T {
        T::ZERO
    }

    fn set_width_val(&mut self, _val: T) {
        ();
    }

    fn set_width_unit(&mut self, _unit: Unit) {
        ();
    }

    fn set_length_val(&mut self, _val: T) {
        ();
    }

    fn set_length_unit(&mut self, _unit: Unit) {
        ();
    }
}

impl<T, U> DistributedCalc<T, U> for Mbend<T>
where
    T: RealScalar,
    U: UnitValue<T> + Frequency<T> + MapScalar<T>,
{
    fn length(&self, freq: &U) -> U::ROutput {
        freq.map_scalar_to_real(|_| T::ZERO)
    }

    fn gamma(&self, freq: &U) -> U::COutput {
        freq.map_scalar_to_complex(|f| Complex::new(self.alpha(f), self.beta(f)))
    }

    fn er(&self, freq: &U) -> U::ROutput {
        self.er(freq)
    }
}

#[derive(Clone)]
pub struct MbendBuilder<T: RealScalar> {
    id: String,
    width1: Option<ScalarUnitValue<T>>,
    width2: Option<ScalarUnitValue<T>>,
    miter: T,
    sub: Option<Msub<T>>,
    nodes: Option<[usize; 2]>,
}

impl<T: RealScalar> MbendBuilder<T> {
    pub fn new() -> Self {
        MbendBuilder::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn width1_uv(mut self, val: &ScalarUnitValue<T>) -> Self {
        self.width1 = Some(val.clone());
        self
    }

    pub fn width1_val(mut self, val: T) -> Self {
        self.width1 = match self.width1 {
            Some(mut x) => {
                x.set_val(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&val, Scale::Base, Unit::Meter)),
        };
        self
    }

    pub fn width1_val_scaled(mut self, val: T, scale: Scale) -> Self {
        self.width1 = match self.width1 {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&val, scale, Unit::Meter)),
        };
        self
    }

    pub fn width1_scale(mut self, val: Scale) -> Self {
        self.width1 = match self.width1 {
            Some(mut x) => {
                x.set_scale(val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&T::ONE, val, Unit::Meter)),
        };
        self
    }

    pub fn width1_unit(mut self, val: Unit) -> Self {
        self.width1 = match self.width1 {
            Some(mut x) => {
                x.set_unit(val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&T::ONE, Scale::Base, val)),
        };
        self
    }

    pub fn width2_uv(mut self, val: &ScalarUnitValue<T>) -> Self {
        self.width2 = Some(val.clone());
        self
    }

    pub fn width2_val(mut self, val: T) -> Self {
        self.width2 = match self.width2 {
            Some(mut x) => {
                x.set_val(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&val, Scale::Base, Unit::Meter)),
        };
        self
    }

    pub fn width2_val_scaled(mut self, val: T, scale: Scale) -> Self {
        self.width2 = match self.width2 {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&val, scale, Unit::Meter)),
        };
        self
    }

    pub fn width2_scale(mut self, val: Scale) -> Self {
        self.width2 = match self.width2 {
            Some(mut x) => {
                x.set_scale(val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&T::ONE, val, Unit::Meter)),
        };
        self
    }

    pub fn width2_unit(mut self, val: Unit) -> Self {
        self.width2 = match self.width2 {
            Some(mut x) => {
                x.set_unit(val);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&T::ONE, Scale::Base, val)),
        };
        self
    }

    pub fn miter(mut self, val: T) -> Self {
        self.miter = val;
        self
    }

    pub fn sub(mut self, val: &Msub<T>) -> Self {
        self.sub = Some(val.clone());
        self
    }

    pub fn nodes(mut self, nodes: [usize; 2]) -> Self {
        self.nodes = Some(nodes);
        self
    }

    pub fn build(self) -> Result<Mbend<T>, String> {
        let elem = "MbendBuilder";
        let width1 = self.width1.ok_or(format!("{elem}: width1 is required"))?;
        let width2 = self.width2.ok_or(format!("{elem}: width2 is required"))?;
        let sub = self.sub.ok_or(format!("{elem}: sub is required"))?;
        let nodes = self.nodes.ok_or(format!("{elem}: nodes is required"))?;
        Ok(Mbend {
            id: self.id,
            width1,
            width2,
            miter: self.miter,
            sub,
            nodes,
        })
    }
}

impl<T: RealScalar> Default for MbendBuilder<T> {
    fn default() -> Self {
        Self {
            id: "MB0".to_string(),
            width1: None,
            width2: None,
            miter: T::ZERO,
            sub: None,
            nodes: None,
        }
    }
}

#[cfg(test)]
mod element_mbend_tests {
    use super::*;
    use crate::{
        element::msub::MsubBuilder,
        units::ArrayUnitValue,
        util::{ApproxEq, NumMargin},
    };
    use num_complex::Complex64;

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
    // fn element_mbend() {
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
    //     let exemplar = Mbend {
    //         id: "MB1".to_string(),
    //         width: UnitValBuilder::new()
    //             .val(width_val)
    //             .scale(Scale::Micro)
    //             .unit(Unit::Meter)
    //             .build(),
    //         miter: false,
    //         sub: sub.clone(),
    //         nodes: [1, 2],
    //         z0: (50.0).into(),
    //     };
    //     let exemplar_z = c64(2_f64.sqrt() * 50.0, 0.0);
    //     let calc = MbendBuilder::new()
    //         .width_val(width_val)
    //         .width_scale(Scale::Micro)
    //         .width_unit(Unit::Meter)
    //         .miter(false)
    //         .sub(&sub)
    //         .nodes([1, 2])
    //         .id("MB1")
    //         .build();
    //     let margin = F64Margin::default();

    //     assert_eq!(&exemplar.id(), &calc.id());
    //     comp_point_c64(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");

    //     let margin = F64Margin {
    //         epsilon: 1e-4,
    //         ulps: 10,
    //     };
    //     comp_c64(&exemplar_z, &calc.z(&freq), margin, "calc.z()", "0");
    //     comp_f64(&exemplar_z.re, &calc.z0(&freq), margin, "calc.z0()", "0");
    // }

    mod mbend_tests {
        use super::*;

        #[test]
        fn test_mbend_builder_default() {
            let mbend: Mbend<f64> = MbendBuilder::new()
                .width_val(1.0)
                .sub(&Msub::default())
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(mbend.id(), "MB0");
            assert_eq!(mbend.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_mbend_miter_vs_no_miter() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val(0.0)
                .build()
                .unwrap();

            let mbend_no_miter = MbendBuilder::new()
                .width_val_scaled(10.0, Scale::Micro)
                .miter(false)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            let mbend_miter = MbendBuilder::new()
                .width_val_scaled(10.0, Scale::Micro)
                .miter(true)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            // Both should build successfully
            assert!(!mbend_no_miter.miter);
            assert!(mbend_miter.miter);
        }

        #[test]
        fn test_mbend_with_substrate() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val_scaled(0.77, Scale::Micro)
                .build()
                .unwrap();

            let mbend = MbendBuilder::new()
                .id("MB1")
                .width_val_scaled(5.8736, Scale::Micro)
                .miter(false)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(mbend.id(), "MB1");
            assert_eq!(mbend.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_mbend_c_matrix_not_zero() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val(0.0)
                .build()
                .unwrap();

            let mbend = MbendBuilder::new()
                .width_val_scaled(10.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            let c_matrix = mbend.c(&freq);

            // For bend, matrix should have non-zero elements
            assert_ne!(c_matrix[[0, 0, 0]], Complex64::ZERO);
            assert_ne!(c_matrix[[0, 1, 1]], Complex64::ZERO);
        }

        #[test]
        fn test_mbend_elem_type() {
            let sub: Msub<f64> = Msub::default();
            let mbend = MbendBuilder::new()
                .width_val(1.0)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(mbend.elem(), ElemType::Mbend);
        }

        #[test]
        fn test_mbend_distributed_trait() {
            let sub = Msub::default();
            let mbend = MbendBuilder::new()
                .width_val_scaled(10.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            mbend
                .width()
                .assert_approx_eq(&10e-6, NumMargin::default(), "distributed_trait", "");
            assert_eq!(mbend.val(), 0.0);
        }
    }
}
