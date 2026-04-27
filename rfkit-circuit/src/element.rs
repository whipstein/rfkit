#![allow(dead_code, unused)]
use enum_dispatch::enum_dispatch;
use ndarray::prelude::*;
use num_complex::Complex;
use rfkit_base::prelude::*;
use serde::{
    Serialize,
    ser::{SerializeStruct, Serializer},
};
use std::{fmt, marker::PhantomData, str::FromStr};

pub mod capacitor;
pub mod ground;
pub mod inductor;
// pub mod mbend;
pub mod mlef;
pub mod mlin;
pub mod msub;
pub mod port;
pub mod q;
pub mod resistor;
pub mod short;
pub mod transformer;

pub use self::{
    capacitor::{Capacitor, CapacitorBuilder, CapacitorElementBuilder, CapacitorSpec},
    ground::{Ground, GroundBuilder, GroundElementBuilder, GroundSpec},
    inductor::{Inductor, InductorBuilder, InductorElementBuilder, InductorSpec},
    // mbend::{Mbend, MbendBuilder},
    mlef::{Mlef, MlefBuilder, MlefElementBuilder, MlefSpec},
    mlin::{Mlin, MlinBuilder, MlinElementBuilder, MlinSpec},
    msub::{Msub, MsubBuilder},
    port::{Port, PortBuilder, PortElementBuilder, PortSpec},
    q::{Q, QBuilder, QMode},
    resistor::{Resistor, ResistorBuilder, ResistorElementBuilder, ResistorSpec},
    short::{Short, ShortBuilder, ShortElementBuilder, ShortSpec},
    transformer::{
        IdealTransformer, IdealTransformerBuilder, IdealTransformerElementBuilder,
        IdealTransformerSpec, Transformer, TransformerBuilder, TransformerElementBuilder,
        TransformerSpec,
    },
};

pub trait ElementSpec<T: RealScalar, const N: usize> {
    type Params: Default;
    type Concrete;

    const NAME: &'static str;
    const DEFAULT_ID: &'static str;

    fn default_z0() -> Complex<T> {
        Complex::new(50.0.into(), T::ZERO)
    }

    fn build_concrete(
        id: String,
        params: Self::Params,
        nodes: [usize; N],
        z0: Complex<T>,
    ) -> Result<Self::Concrete, String>;
}

pub trait ElementBuildMode<T: RealScalar, E> {
    type Output;

    fn finish(element: E) -> Self::Output;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ConcreteElement;

#[derive(Clone, Copy, Debug, Default)]
pub struct TopLevelElement;

impl<T: RealScalar, E> ElementBuildMode<T, E> for ConcreteElement {
    type Output = E;

    fn finish(element: E) -> Self::Output {
        element
    }
}

impl<T, E> ElementBuildMode<T, E> for TopLevelElement
where
    T: RealScalar,
    Element<T>: From<E>,
{
    type Output = Element<T>;

    fn finish(element: E) -> Self::Output {
        element.into()
    }
}

#[derive(Clone)]
pub struct ElementBuilder<T, S, M, const N: usize>
where
    T: RealScalar,
    S: ElementSpec<T, N>,
{
    pub(crate) id: String,
    pub(crate) params: S::Params,
    pub(crate) nodes: Option<[usize; N]>,
    pub(crate) z0: Complex<T>,
    _spec: PhantomData<(S, M)>,
}

impl<T, S, M, const N: usize> ElementBuilder<T, S, M, N>
where
    T: RealScalar,
    S: ElementSpec<T, N>,
    M: ElementBuildMode<T, S::Concrete>,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    pub fn nodes(mut self, nodes: [usize; N]) -> Self {
        self.nodes = Some(nodes);
        self
    }

    pub fn build(self) -> Result<M::Output, String> {
        let nodes = self
            .nodes
            .ok_or_else(|| format!("{}: nodes is required", S::NAME))?;
        S::build_concrete(self.id, self.params, nodes, self.z0).map(M::finish)
    }
}

impl<T, S, M, const N: usize> Default for ElementBuilder<T, S, M, N>
where
    T: RealScalar,
    S: ElementSpec<T, N>,
{
    fn default() -> Self {
        Self {
            id: S::DEFAULT_ID.to_string(),
            params: S::Params::default(),
            nodes: None,
            z0: S::default_z0(),
            _spec: PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub enum ElemType {
    #[default]
    Capacitor,
    Ground,
    IdealTransformer,
    Inductor,
    Mbend,
    Mlef,
    Mlin,
    Msub,
    None,
    Port,
    Resistor,
    Short,
    Transformer,
}

impl ElemType {
    /// Convert ElemType to String
    pub fn to_str(&self) -> &str {
        match self {
            ElemType::Capacitor => "Capacitor",
            ElemType::Ground => "GND",
            ElemType::IdealTransformer => "Ideal Transformer",
            ElemType::Inductor => "Inductor",
            ElemType::Mbend => "MBEND",
            ElemType::Mlef => "MLEF",
            ElemType::Mlin => "MLIN",
            ElemType::Msub => "MSUB",
            ElemType::None => "None",
            ElemType::Port => "Port",
            ElemType::Resistor => "Resistor",
            ElemType::Short => "Short",
            ElemType::Transformer => "Transformer",
        }
    }
}

impl FromStr for ElemType {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "C" | "c" | "Cap" | "cap" | "Capacitor" | "capacitor" => Ok(ElemType::Capacitor),
            "GND" | "gnd" | "Ground" | "ground" => Ok(ElemType::Ground),
            "ixfmr" => Ok(ElemType::IdealTransformer),
            "L" | "l" | "Ind" | "ind" | "Inductor" | "inductor" => Ok(ElemType::Inductor),
            "mbend" | "MBEND" => Ok(ElemType::Mbend),
            "mlef" | "MLEF" => Ok(ElemType::Mlef),
            "mlin" | "MLIN" => Ok(ElemType::Mlin),
            "msub" | "MSUB" => Ok(ElemType::Msub),
            "none" | "None" => Ok(ElemType::None),
            "P" | "p" | "Port" | "port" => Ok(ElemType::Port),
            "R" | "r" | "Res" | "res" | "Resistor" | "resistor" => Ok(ElemType::Resistor),
            "SHORT" | "Short" | "short" => Ok(ElemType::Short),
            "Transformer" | "transformer" | "xfmr" => Ok(ElemType::Transformer),
            _ => Err("ElemType not recognized".to_string().into()),
        }
    }
}

impl fmt::Display for ElemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ElemType::Capacitor => write!(f, "Capacitor"),
            ElemType::Ground => write!(f, "GND"),
            ElemType::IdealTransformer => write!(f, "Ideal Transformer"),
            ElemType::Inductor => write!(f, "Inductor"),
            ElemType::Mbend => write!(f, "MBEND"),
            ElemType::Mlef => write!(f, "MLEF"),
            ElemType::Mlin => write!(f, "MLIN"),
            ElemType::Msub => write!(f, "Msub"),
            ElemType::None => write!(f, "None"),
            ElemType::Port => write!(f, "Port"),
            ElemType::Resistor => write!(f, "Resistor"),
            ElemType::Short => write!(f, "Short"),
            ElemType::Transformer => write!(f, "Transformer"),
        }
    }
}

macro_rules! define_elem_impl {
    (variants: [$($variant:ident),+ $(,)?]) => {
        impl<T: RealScalar> Elem<T> for Element<T> {
            fn id(&self) -> String {
                match self {
                    $(
                        Element::$variant(elem) => elem.id(),
                    )+
                }
            }

            fn elem(&self) -> ElemType {
                match self {
                    $(
                        Element::$variant(elem) => elem.elem(),
                    )+
                }
            }

            fn nodes(&self) -> Vec<usize> {
                match self {
                    $(
                        Element::$variant(elem) => elem.nodes(),
                    )+
                }
            }
        }

        impl<T: RealScalar, U: FreqValue<T>> ElemCalc<T, U> for Element<T> {
            fn c(&self, freq: &U) -> Points<Complex<T>, Ix3> {
                match self {
                    $(
                        Element::$variant(elem) => elem.c(freq),
                    )+
                }
            }

            fn net(&self, freq: &U) -> Points<Complex<T>, Ix3> {
                match self {
                    $(
                        Element::$variant(elem) => elem.net(freq),
                    )+
                }
            }

            fn z_scalar(&self, freq: &ScalarUnitValue<T>) -> Complex<T> {
                match self {
                    $(
                        Element::$variant(elem) => ElemCalc::<T, ScalarUnitValue<T>>::z_scalar(elem, freq),
                    )+
                }
            }
        }
    };
}

#[enum_dispatch]
pub trait Elem<T: RealScalar> {
    fn id(&self) -> String;
    fn elem(&self) -> ElemType;
    fn nodes(&self) -> Vec<usize>;
    fn c<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3>;
    fn c_at<U: FreqValue<T>>(&self, freq: &U, idx: (usize, usize, usize)) -> Complex<T> {
        self.c(freq)[idx]
    }
    fn c_at_freq<U: FreqValue<T>>(&self, freq: &U, idx: usize) -> Points<Complex<T>, Ix2> {
        Points(self.c(freq).slice(s![idx, .., ..]).to_owned())
    }
    fn net<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3>;
    fn z_scalar(&self, freq: &ScalarUnitValue<T>) -> Complex<T>;
    fn z<U: FreqValue<T>>(&self, freq: &U) -> U::COutput {
        freq.map_scalar_to_complex(|f| self.z_scalar(f))
    }
    fn z_at<U: FreqValue<T>>(&self, freq: &U, idx: usize) -> Complex<T> {
        freq.map_scalar_to_vec(|f| self.z_scalar(f))[idx]
    }
}

pub trait Lumped<T: RealScalar>: Elem<T> {
    const NODE_LEN: usize = 2;

    fn val(&self) -> T;
    fn val_scaled(&self) -> T;
    fn scale(&self) -> Scale;
    fn unit(&self) -> Unit;
    fn set_val(&mut self, val: T);
    fn set_val_scaled(&mut self, val: T);
    fn set_scale(&mut self, scale: Scale);
}

pub trait Distributed<T: RealScalar>: Elem<T> {
    const NODE_LEN: usize = 2;

    fn width(&self) -> T;
    fn val(&self) -> T;
    fn set_width_val(&mut self, val: T);
    fn set_width_unit(&mut self, unit: Unit);
    fn set_length_val(&mut self, val: T);
    fn set_length_unit(&mut self, unit: Unit);
    fn length<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput;
    fn gamma<U: FreqValue<T>>(&self, freq: &U) -> U::COutput;
    fn er<U: FreqValue<T>>(&self, freq: &U) -> U::ROutput;
}

pub trait Term<T: RealScalar>: Elem<T> {
    const NODE_LEN: usize = 1;

    fn val(&self) -> Complex<T>;
    fn set_val(&mut self, val: Complex<T>);
}

#[derive(Debug, Clone, PartialEq)]
#[enum_dispatch(Elem<T>)]
pub enum Element<T: RealScalar> {
    Capacitor(Capacitor<T>),
    Ground(Ground<T>),
    IdealTransformer(IdealTransformer<T>),
    Inductor(Inductor<T>),
    // Mbend(Mbend<T>),
    Mlef(Mlef<T>),
    Mlin(Mlin<T>),
    Port(Port<T>),
    Resistor(Resistor<T>),
    Short(Short<T>),
    Transformer(Transformer<T>),
}

impl<T: RealScalar> Element<T> {
    pub fn builder<S, const N: usize>() -> ElementBuilder<T, S, TopLevelElement, N>
    where
        S: ElementSpec<T, N>,
        TopLevelElement: ElementBuildMode<T, S::Concrete>,
    {
        ElementBuilder::new()
    }

    pub fn capacitor() -> ElementBuilder<T, CapacitorSpec, TopLevelElement, 2> {
        ElementBuilder::new()
    }

    pub fn ground() -> ElementBuilder<T, GroundSpec, TopLevelElement, 1> {
        ElementBuilder::new()
    }

    pub fn ideal_transformer() -> ElementBuilder<T, IdealTransformerSpec, TopLevelElement, 2> {
        ElementBuilder::new()
    }

    pub fn inductor() -> ElementBuilder<T, InductorSpec, TopLevelElement, 2> {
        ElementBuilder::new()
    }

    pub fn mlef() -> ElementBuilder<T, MlefSpec, TopLevelElement, 1> {
        ElementBuilder::new()
    }

    pub fn mlin() -> ElementBuilder<T, MlinSpec, TopLevelElement, 2> {
        ElementBuilder::new()
    }

    pub fn port() -> ElementBuilder<T, PortSpec, TopLevelElement, 1> {
        ElementBuilder::new()
    }

    pub fn resistor() -> ElementBuilder<T, ResistorSpec, TopLevelElement, 2> {
        ElementBuilder::new()
    }

    pub fn short() -> ElementBuilder<T, ShortSpec, TopLevelElement, 2> {
        ElementBuilder::new()
    }

    pub fn transformer() -> ElementBuilder<T, TransformerSpec, TopLevelElement, 2> {
        ElementBuilder::new()
    }

    pub fn val(&self) -> T {
        match self {
            Element::Capacitor(elem) => elem.val(),
            Element::Inductor(elem) => elem.val(),
            Element::Resistor(elem) => elem.val(),
            _ => T::ZERO,
        }
    }
}

impl<T: RealScalar> Serialize for Element<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("Element", 7)?;
        s.serialize_field("elemId", &self.id())?;
        s.serialize_field("elemType", &self.elem().to_str())?;
        s.serialize_field("elemVal", &self.val())?;
        s.serialize_field("elemNodes", &self.nodes())?;
        s.serialize_field("elemInfo", &self)?;
        s.end()
    }
}

// define_elem_impl!(
//     // variants: [Capacitor, Ground, IdealTransformer, Inductor, Mbend, Mlef, Mlin, Port, Resistor, Short, Transformer]
//     variants: [Capacitor, Ground, IdealTransformer, Inductor, Mlef, Mlin, Port, Resistor, Short, Transformer]
// );

/// Calculate exponent in radians for use in exp(-gamma * L) MLIN calculations
pub fn mlin_exp<T: RealScalar>(len: ScalarUnitValue<T>, gamma: Complex<T>) -> Complex<T> {
    match len.unit() {
        Unit::Degree => -Complex::new(0.0.into(), len.val() * std::f64::consts::PI / 180.0),
        Unit::Radian => -Complex::new(0.0.into(), len.val()),
        Unit::Lambda => -Complex::new(0.0.into(), len.val() * 2.0 * std::f64::consts::PI),
        Unit::Meter => gamma * Complex::new(len.val(), 0.0.into()),
        Unit::Inch => gamma * Complex::new(len.val(), 0.0.into()) * T::from_f64(0.0254),
        _ => panic!("not a valid unit for MLIN"),
    }
}

#[macro_export]
macro_rules! define_mlin_calcs {
    ($variant:ident) => {
        impl<T: RealScalar> $variant<T> {
            /// Effective width
            pub fn w_eff(&self) -> T {
                let w = self.width();
                if self.sub.thickness() == 0.0 {
                    return w;
                }

                let t = self.sub.thickness();
                let h = self.sub.height();
                let tau = t / h;

                w + t / std::f64::consts::PI
                    * (T::from_f64(4.0 * std::f64::consts::E) / tau + 1.0).ln()
            }

            /// Effective width for impedance
            pub fn w_eff_z(&self) -> T {
                let w = self.width();
                if self.sub.thickness() == 0.0 {
                    return w;
                }

                let t = self.sub.thickness();
                let h = self.sub.height();
                let tnorm = t / h;

                w + t / std::f64::consts::PI
                    * (T::ONE + T::from_f64(4.0 * std::f64::consts::E) / tnorm).ln()
            }

            /// Effective width for permittivity
            pub fn w_eff_e(&self) -> T {
                let w = self.width();
                if self.sub.thickness() == 0.0 {
                    return w;
                }

                let weff = self.w_eff_z();
                let dweff = weff - w;
                let er = self.sub.er();

                w + dweff * (T::ONE + er.recip()) / 2.0
            }

            /// Z0 based on modified Kirschning & Jansen model
            pub fn z0<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                let z0 = self.z0_dc();
                let ereffdc = self.er_effdc();

                freq.map_scalar_to_real(|f| {
                    let ereff = self.er_eff(f);

                    z0 * (ereffdc / ereff).sqrt()
                })
            }

            /// Z0 based on Kirschning & Jansen
            fn z0_dc(&self) -> T {
                let h = self.sub.height();
                let ueff = self.w_eff_z() / h;
                let eeff = self.er_effdc();
                if ueff <= 1.0 {
                    eeff.sqrt().recip() * 60.0 * (ueff.recip() * 8.0 + ueff * 0.25).ln()
                } else {
                    eeff.sqrt().recip() * 120.0 * std::f64::consts::PI
                        / (ueff + 1.393 + (ueff + 1.444).ln() * 0.667)
                }
            }

            /// er_eff based on modified Kirschning & Jansen model
            fn er_eff<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                let h = self.sub.height();
                let weff = self.w_eff_e();
                let ueff = weff / h;
                let er = self.sub.er();
                let ereffdc = self.er_effdc();

                freq.map_scalar_to_real(|f| {
                    let fnorm = self.sub.f_norm(f);
                    let p1 = ((fnorm * 0.0157 + 1.0).powi(-20) * 0.525 + 0.6315) * ueff + 0.27488
                        - (ueff * -8.7513).exp() * 0.065683;
                    let p2 = (T::ONE - (er * -0.03442).exp()) * 0.33622;
                    let p3 = (ueff * -4.6).exp()
                        * 0.0363
                        * (T::ONE - (-(fnorm / 38.7).powf(4.97.into())).exp());
                    let p4 = T::ONE + (T::ONE - (-(er / 15.916).powi(8)).exp()) * 2.751;
                    let p = p1 * p2 * fnorm.powf(p3 * p4);

                    er - (er - ereffdc) / (T::ONE + p)
                })
            }

            /// static eeff based on Kirschning & Jansen model
            fn er_effdc(&self) -> T {
                let h = self.sub.height();
                let er = self.sub.er();
                let weff = self.w_eff_e();
                let ueff = weff / h;
                let modifier = if ueff < 1.0 {
                    (-ueff + 1.0).powi(2) * 0.04
                } else {
                    T::ZERO
                };
                (er + 1.0) / 2.0
                    + (er - 1.0) / 2.0 * (T::from_f64(12.0) / ueff + 1.0).sqrt().recip()
                    + modifier
            }

            /// Conductor loss based on Hammerstad & Jensen model in Nepers/meter
            pub fn alpha_cdc<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| {
                    let ki = ((self.z0(f) / (120.0 * std::f64::consts::PI)).powf(0.7.into())).exp()
                        * -1.2;

                    self.sub.res_sh(f) * self.sub.roughness(f) * ki
                        / (self.z0(f) * self.width.val())
                })
            }

            /// Conductor loss based on Wheeler's model in Nepers/meter
            pub fn alpha_c<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                let r = (self.sub.conductivity() * self.width.val() * self.sub.thickness()).recip();
                let l = self.z0_dc() / 3e8;
                let _ft1 = r / (l * 2.0 * std::f64::consts::PI);
                let ft2 = T::from_f64(4.0)
                    / (self.sub.conductivity()
                        * self.sub.thickness().powi(2)
                        * std::f64::consts::PI
                        * std::f64::consts::PI
                        * 4e-7);

                freq.map_scalar_to_real(|f| {
                    if f.freq() < ft2 {
                        (self.sub.conductivity()
                            * self.width()
                            * self.sub.thickness()
                            * self.z0(f)
                            * 2.0)
                            .recip()
                    } else {
                        let wpeff = match self.width() / self.sub.height()
                            <= 1.0 / (2.0 * std::f64::consts::PI)
                        {
                            true => {
                                self.width.val()
                                    + (self.sub.thickness() - self.sub.delta(f))
                                        / std::f64::consts::PI
                                        * (((self.width.val() - self.sub.delta(f))
                                            / (self.sub.thickness() - self.sub.delta(f))
                                            * 4.0
                                            * std::f64::consts::PI)
                                            .ln()
                                            + 1.0)
                            }
                            false => {
                                self.width.val()
                                    + (self.sub.thickness() - self.sub.delta(f))
                                        / std::f64::consts::PI
                                        * (((self.sub.height() - self.sub.delta(f))
                                            / (self.sub.thickness() - self.sub.delta(f))
                                            * 2.0)
                                            .ln()
                                            + 1.0)
                            }
                        };
                        let f1 = (-((self.sub.height() + self.sub.delta(f)) * 30.666 / wpeff)
                            .powf(0.7528.into()))
                        .exp()
                            * (2.0 * std::f64::consts::PI - 6.0)
                            + 6.0;
                        let zp0 = (f1 * (self.sub.height() + self.sub.delta(f)) / wpeff
                            + (((self.sub.height() + self.sub.delta(f)) * 2.0 / wpeff).powi(2)
                                + 1.0)
                                .sqrt())
                            * 60_f64.ln();
                        let deltaz = zp0 - self.z0_dc();
                        f.wavelength(1.0.into()).recip()
                            * (self.er_eff(f)).sqrt()
                            * std::f64::consts::PI
                            * deltaz
                            / self.z0_dc()
                    }
                })
            }

            /// Dielectric loss based on Hammerstad & Jensen model in Nepers/meter
            pub fn alpha_ddc<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| {
                    (self.er_effdc() - 1.0) * std::f64::consts::PI * self.sub.er() * self.sub.tand()
                        / (f.wavelength(1.0.into())
                            * (self.sub.er() - 1.0)
                            * (self.er_effdc()).sqrt())
                })
            }

            /// Dielectric loss based on Awasthi, Singh, Sharma, Kumari & Verma in Nepers/meter
            pub fn alpha_d<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| {
                    if f.wavelength(self.sub.er())
                        < T::from_f64(30e-2.unscale(Scale::Centi)) / self.sub.er().sqrt()
                    {
                        self.sub.er() / (self.er(f)).sqrt()
                            * (self.er(f) - 1.0)
                            * std::f64::consts::PI
                            * self.sub.tand()
                            / ((self.sub.er() - 1.0) * f.wavelength(1.0.into()))
                    } else if f.wavelength(self.sub.er())
                        < T::from_f64(30e-3.unscale(Scale::Centi)) / self.sub.er().sqrt()
                    {
                        let sigma0 = self.sub.tand() * f.w() * self.sub.er() * 8.854e-12;
                        let sigma = sigma0 / (f.freq().scale(Scale::Giga) * 0.045 + 1.0).sqrt();
                        let tand = sigma / (f.w() * self.sub.er() * 8.854e-12);

                        self.sub.er() / (self.er(f)).sqrt()
                            * (self.er(f) - 1.0)
                            * std::f64::consts::PI
                            * tand
                            / ((self.sub.er() - 1.0) * f.wavelength(1.0.into()))
                    } else {
                        let f_val = T::from_f64(1e12.scale(Scale::Giga));
                        let w = T::from_f64(2.0 * std::f64::consts::PI * 1e12);
                        let sigma0 = self.sub.tand() * w * self.sub.er() * 8.854e-12;
                        let sigma = sigma0 / (f_val * 0.045 + 1.0).sqrt();
                        let tand = sigma / (w * self.sub.er() * 8.854e-12);

                        self.sub.er() / self.er(f).sqrt()
                            * (self.er(f) - 1.0)
                            * std::f64::consts::PI
                            * tand
                            / ((self.sub.er() - 1.0) * f.wavelength(1.0.into()))
                    }
                })
            }

            pub fn r<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| self.z0(f) * self.alpha_c(f) * 2.0)
            }

            pub fn l<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| self.z0(f) * self.er(f).sqrt() / 3e8)
            }

            pub fn g<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| self.alpha_d(f) / self.z0(f) * 2.0)
            }

            pub fn cap<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| self.er(f).sqrt() / 3e8 * self.z0(f))
            }

            pub fn alpha<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| {
                    self.r(f) / (self.z0(f) * 2.0) + self.g(f) * self.z0(f) / 2.0
                })
            }

            pub fn beta<U>(&self, freq: &U) -> U::ROutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_real(|f| f.w() / 3e8 * self.er_eff(f).sqrt())
            }

            pub fn gamma<U>(&self, freq: &U) -> U::COutput
            where
                U: FreqValue<T>,
            {
                freq.map_scalar_to_complex(|f| Complex::new(self.alpha(f), self.beta(f)))
            }
        }
    };
}

#[cfg(test)]
mod element_tests {
    use super::*;
    use num_complex::c64;

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

    mod integration_tests {
        use super::*;

        #[test]
        fn test_rlc_series_combination() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let r = ResistorBuilder::new()
                .val(50.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let l = InductorBuilder::new()
                .val_scaled(1.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z_r = r.z_at(&freq, 0);
            let z_l = l.z_at(&freq, 0);
            let z_c = c.z_at(&freq, 0);
            let z_total = z_r + z_l + z_c;

            // Total impedance should be sum of individual impedances
            assert!(z_total.re > 0.0); // Resistive part
            assert!(z_total.im.abs() > 0.0); // Reactive part
        }

        #[test]
        fn test_element_type_differentiation() {
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let res: Resistor<f64> = ResistorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let ind: Inductor<f64> = InductorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let gnd: Ground<f64> = Ground::new();
            let port: Port<f64> = PortBuilder::new()
                .z(c64(50.0, 0.0))
                .nodes([1])
                .build()
                .unwrap();

            assert_ne!(cap.elem(), res.elem());
            assert_ne!(res.elem(), ind.elem());
            assert_ne!(ind.elem(), gnd.elem());
            assert_ne!(gnd.elem(), port.elem());
        }

        #[test]
        fn test_lumped_elements_share_c_matrix_structure() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let cap = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let res = ResistorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let ind = InductorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let c_cap = cap.c(&freq);
            let c_res = res.c(&freq);
            let c_ind = ind.c(&freq);

            // All should have same structure (different from impedance)
            c_cap[[0, 0, 0]].re.assert_approx_eq(
                &c_res[[0, 0, 0]].re,
                DEFAULT_MARGIN,
                "lumped_elements_share_c_matrix_structure",
                "c_cap == c_res",
            );
            c_res[[0, 0, 0]].re.assert_approx_eq(
                &c_ind[[0, 0, 0]].re,
                DEFAULT_MARGIN,
                "lumped_elements_share_c_matrix_structure",
                "c_res == c_ind",
            );
        }

        #[test]
        fn test_frequency_sweep_consistency() {
            let freqs = array![1e6, 1e7, 1e8, 1e9, 10e9];
            let freq = ArrayUnitValue::new_freq(&freqs.clone(), Scale::Base);

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            // Check impedance magnitude decreases with frequency
            let z1 = cap.z_at(&freq, 0).norm();
            let z2 = cap.z_at(&freq, 1).norm();
            let z3 = cap.z_at(&freq, 2).norm();

            assert!(z1 > z2);
            assert!(z2 > z3);
        }

        #[test]
        fn test_microstrip_components_with_same_substrate() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0004)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val_scaled(0.77, Scale::Micro)
                .build()
                .unwrap();

            let mlin = MlinBuilder::new()
                .width_val_scaled(10.0, Scale::Micro)
                .length_val_scaled(1000.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            let mlef = MlefBuilder::new()
                .width_val_scaled(10.0, Scale::Micro)
                .sub(&sub)
                .nodes([2])
                .build()
                .unwrap();

            // let mbend = MbendBuilder::new()
            //     .width_scaled(10.0, Scale::Micro)
            //     .sub(&sub)
            //     .build()
            //     .unwrap();

            // All should share the same substrate properties
            assert_eq!(mlin.sub().er(), sub.er());
            assert_eq!(mlef.sub().er(), sub.er());
            // assert_eq!(mbend.sub().er(), sub.er());
        }
    }

    mod edge_case_tests {
        use super::*;

        #[test]
        fn test_very_small_component_values() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Femto)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = cap.z_at(&freq, 0);
            println!("\n\nz = {}\n\n", z.norm());
            assert!(z.norm() > 1e5); // Very large impedance for small cap
        }

        #[test]
        fn test_very_large_component_values() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Milli)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = cap.z_at(&freq, 0);
            assert!(z.norm() < 1.0); // Very small impedance for large cap
        }

        #[test]
        fn test_zero_frequency_handling() {
            let freq = ArrayUnitValue::new_freq(&array![0.0], Scale::Base);
            let res = ResistorBuilder::new()
                .val(50.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = res.z_at(&freq, 0);
            z.re.assert_approx_eq(&50.0, DEFAULT_MARGIN, "zero_frequency_handling", "z");
        }

        #[test]
        fn test_very_high_frequency() {
            let freq = ArrayUnitValue::new_freq(&array![1e12], Scale::Base); // 1 THz

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = cap.z_at(&freq, 0);
            assert!(z.norm() < 1.0); // Should be very small at THz
        }

        #[test]
        fn test_node_numbering_non_sequential() {
            let nodes_list = vec![[0, 100], [5, 50], [1, 1000]];

            for nodes in nodes_list {
                let cap: Capacitor<f64> = CapacitorBuilder::new()
                    .val(1.0)
                    .nodes(nodes)
                    .build()
                    .unwrap();
                assert_eq!(cap.nodes(), vec![nodes[0], nodes[1]]);
            }
        }

        #[test]
        fn test_same_node_both_terminals() {
            // This creates a short circuit - should still build
            let res: Resistor<f64> = ResistorBuilder::new()
                .val(1.0)
                .nodes([5, 5])
                .build()
                .unwrap();

            assert_eq!(res.nodes(), vec![5, 5]);
        }

        #[test]
        fn test_element_name_and_id_consistency() {
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .id("C_test")
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(cap.id(), "C_test");
            assert_eq!(cap.name(), "C_test");
        }
    }

    mod builder_pattern_tests {
        use super::*;

        #[test]
        fn test_builder_method_chaining() {
            let cap = CapacitorBuilder::new()
                .id("C1")
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 2])
                .z0(c64(75.0, 0.0))
                .build()
                .unwrap();

            assert_eq!(cap.id(), "C1");
            assert_eq!(cap.val_scaled(), 10.0);
            assert_eq!(cap.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_builder_partial_specification() {
            // Build with only some parameters specified
            let cap1: Capacitor<f64> = CapacitorBuilder::new()
                .id("C1")
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let cap2 = CapacitorBuilder::new()
                .val_scaled(5.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(cap1.id(), "C1");
            assert_eq!(cap2.val_scaled(), 5.0);
        }

        #[test]
        fn test_multiple_builds_from_same_pattern() {
            let base_builder = ResistorBuilder::new().val(100.0).z0(c64(50.0, 0.0));

            let res1 = base_builder.clone().id("R1").nodes([1, 2]).build().unwrap();
            let res2 = base_builder.clone().id("R2").nodes([2, 3]).build().unwrap();

            assert_eq!(res1.val(), 100.0);
            assert_eq!(res2.val(), 100.0);
            assert_ne!(res1.id(), res2.id());
        }

        #[test]
        fn test_top_level_element_builders() {
            let sub = MsubBuilder::new()
                .er(4.5)
                .tand(0.02)
                .height_val_scaled(1.6, Scale::Milli)
                .thickness_val_scaled(1.0, Scale::Micro)
                .build()
                .unwrap();

            let elements = vec![
                Element::<f64>::capacitor()
                    .val_scaled(1.0, Scale::Pico)
                    .nodes([1, 2])
                    .build()
                    .unwrap(),
                Element::<f64>::resistor()
                    .val(50.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap(),
                Element::<f64>::inductor()
                    .val_scaled(1.0, Scale::Nano)
                    .nodes([1, 2])
                    .build()
                    .unwrap(),
                Element::<f64>::port()
                    .z(c64(50.0, 0.0))
                    .nodes([1])
                    .build()
                    .unwrap(),
                Element::<f64>::ground().nodes([0]).build().unwrap(),
                Element::<f64>::short().nodes([1, 2]).build().unwrap(),
                Element::<f64>::ideal_transformer()
                    .n(2.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap(),
                Element::<f64>::transformer()
                    .n(1.0)
                    .km(0.5)
                    .l1_val(5e-12)
                    .nodes([1, 2])
                    .build()
                    .unwrap(),
                Element::<f64>::mlin()
                    .width_val_scaled(3.0, Scale::Milli)
                    .length_val_scaled(10.0, Scale::Milli)
                    .sub(&sub)
                    .nodes([1, 2])
                    .build()
                    .unwrap(),
                Element::<f64>::mlef()
                    .width_val_scaled(3.0, Scale::Milli)
                    .sub(&sub)
                    .nodes([1])
                    .build()
                    .unwrap(),
            ];

            let elem_types: Vec<ElemType> = elements.iter().map(|elem| elem.elem()).collect();
            assert_eq!(
                elem_types,
                vec![
                    ElemType::Capacitor,
                    ElemType::Resistor,
                    ElemType::Inductor,
                    ElemType::Port,
                    ElemType::Ground,
                    ElemType::Short,
                    ElemType::IdealTransformer,
                    ElemType::Transformer,
                    ElemType::Mlin,
                    ElemType::Mlef,
                ]
            );
        }
    }

    mod trait_tests {
        use super::*;

        #[test]
        fn test_lumped_trait_capacitor() {
            let mut cap = CapacitorBuilder::new()
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(cap.val_scaled(), 10.0);

            cap.set_val_scaled(20.0);
            assert_eq!(cap.val_scaled(), 20.0);
        }

        #[test]
        fn test_lumped_trait_resistor() {
            let mut res = ResistorBuilder::new()
                .val(50.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(res.val(), 50.0);

            res.set_val(75.0);
            assert_eq!(res.val(), 75.0);
        }

        #[test]
        fn test_lumped_trait_inductor() {
            let mut ind = InductorBuilder::new()
                .val_scaled(1.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(ind.val_scaled(), 1.0);

            ind.set_val_scaled(2.0);
            assert_eq!(ind.val_scaled(), 2.0);
        }

        #[test]
        fn test_unitized_trait() {
            let cap = CapacitorBuilder::new()
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(cap.scale(), Scale::Pico);
            assert_eq!(cap.unit(), Unit::Farad);
            assert_eq!(cap.val_scaled(), 10.0);
        }

        #[test]
        fn test_elem_trait_common_methods() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let elements: Vec<Element<f64>> = vec![
                CapacitorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
                ResistorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
                InductorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
            ];

            for elem in elements {
                // All should implement these methods
                let _ = elem.id();
                let _ = elem.nodes();
                let _ = elem.z(&freq);
                let _ = elem.c(&freq);
                let _ = elem.elem();
            }
        }
    }

    mod stability_tests {
        use super::*;

        #[test]
        fn test_numerical_stability_extreme_frequencies() {
            let freqs = array![1e-3, 1e0, 1e3, 1e6, 1e9, 1e12];
            let freq = ArrayUnitValue::new_freq(&freqs.clone(), Scale::Base);

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            for i in 0..freqs.len() {
                let z = cap.z_at(&freq, i);
                assert!(z.is_finite());
                assert!(!z.is_nan());
            }
        }

        #[test]
        fn test_matrix_operations_consistency() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let c1 = cap.c(&freq);
            let c2 = cap.c(&freq);

            // Multiple calls should return same result
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(c1[[0, i, j]], c2[[0, i, j]]);
                }
            }
        }

        #[test]
        fn test_c_at_equals_c_matrix_element() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let c_matrix = cap.c(&freq);

            for i in 0..2 {
                for j in 0..2 {
                    let c_at = cap.c_at(&freq, (0, i, j));
                    assert_eq!(c_at, c_matrix[[0, i, j]]);
                }
            }
        }
    }

    mod property_tests {
        use super::*;

        #[test]
        fn test_capacitor_impedance_inverse_frequency_relationship() {
            // Property: |Z(2f)| = |Z(f)| / 2 for capacitors
            let freqs = vec![1e6, 1e7, 1e8, 1e9];

            for &f in &freqs {
                let freq1 = ArrayUnitValue::new_freq(&array![f], Scale::Base);
                let freq2 = ArrayUnitValue::new_freq(&array![2.0 * f], Scale::Base);

                let cap = CapacitorBuilder::new()
                    .val_scaled(1.0, Scale::Pico)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                let z1 = cap.z_at(&freq1, 0).norm();
                let z2 = cap.z_at(&freq2, 0).norm();

                let ratio = z1 / z2;
                ratio.assert_approx_eq(
                    &2.0,
                    DEFAULT_MARGIN,
                    "capacitor_impedance_inverse_frequency_relationship",
                    "ratio",
                );
            }
        }

        #[test]
        fn test_inductor_impedance_proportional_frequency() {
            // Property: |Z(2f)| = 2 * |Z(f)| for inductors
            let freqs = vec![1e6, 1e7, 1e8, 1e9];

            for &f in &freqs {
                let freq1 = ArrayUnitValue::new_freq(&array![f], Scale::Base);
                let freq2 = ArrayUnitValue::new_freq(&array![2.0 * f], Scale::Base);

                let ind = InductorBuilder::new()
                    .val_scaled(1.0, Scale::Nano)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                let z1 = ind.z_at(&freq1, 0).norm();
                let z2 = ind.z_at(&freq2, 0).norm();

                let ratio = z2 / z1;
                ratio.assert_approx_eq(
                    &2.0,
                    DEFAULT_MARGIN,
                    "capacitor_impedance_inverse_frequency_relationship",
                    "ratio",
                );
            }
        }

        #[test]
        fn test_resistor_frequency_invariance() {
            // Property: Z(f1) = Z(f2) for all frequencies for resistors
            let freqs = array![1e3, 1e6, 1e9, 1e12];
            let freq = ArrayUnitValue::new_freq(&freqs.clone(), Scale::Base);

            let res = ResistorBuilder::new()
                .val(100.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z_reference = res.z_at(&freq, 0);

            for i in 1..freqs.len() {
                let z = res.z_at(&freq, i);
                assert_eq!(z, z_reference);
            }
        }

        #[test]
        fn test_series_impedance_additivity() {
            // Property: Z_total = Z1 + Z2 + Z3 for series connection
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let r = ResistorBuilder::new()
                .val(50.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let l = InductorBuilder::new()
                .val_scaled(10.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z_r = r.z_at(&freq, 0);
            let z_l = l.z_at(&freq, 0);
            let z_c = c.z_at(&freq, 0);
            let z_series = z_r + z_l + z_c;

            // Real part should equal resistor value
            z_series.re.assert_approx_eq(
                &50.0,
                DEFAULT_MARGIN,
                "series_impedance_additivity",
                "z_series.re",
            );

            // Imaginary part should be sum of L and C reactances
            let expected_im = z_l.im + z_c.im;
            z_series.im.assert_approx_eq(
                &expected_im,
                DEFAULT_MARGIN,
                "series_impedance_additivity",
                "z_series.im",
            );
        }

        #[test]
        fn test_capacitor_reactance_sign() {
            // Property: Capacitive reactance is always negative
            let freqs = array![1e6, 1e7, 1e8, 1e9, 1e10];
            let freq = ArrayUnitValue::new_freq(&freqs.clone(), Scale::Base);

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            for i in 0..freqs.len() {
                let z = cap.z_at(&freq, i);
                assert!(z.im < 0.0, "Capacitive reactance should be negative");
            }
        }

        #[test]
        fn test_inductor_reactance_sign() {
            // Property: Inductive reactance is always positive
            let freqs = array![1e6, 1e7, 1e8, 1e9, 1e10];
            let freq = ArrayUnitValue::new_freq(&freqs.clone(), Scale::Base);

            let ind = InductorBuilder::new()
                .val_scaled(1.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();

            for i in 0..freqs.len() {
                let z = ind.z_at(&freq, i);
                assert!(z.im > 0.0, "Inductive reactance should be positive");
            }
        }

        #[test]
        fn test_c_matrix_symmetry_lumped() {
            // Property: C matrix should be symmetric for lumped elements
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let elements: Vec<Element<f64>> = vec![
                CapacitorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
                ResistorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
                InductorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
            ];

            for elem in elements {
                let c = elem.c(&freq);
                assert_eq!(c[[0, 0, 1]], c[[0, 1, 0]], "C matrix should be symmetric");
            }
        }

        #[test]
        fn test_value_scaling_consistency() {
            // Property: val() should equal val_scaled() * scale.multiplier()
            let test_cases = vec![
                (1.0, Scale::Pico),
                (10.0, Scale::Nano),
                (1.0, Scale::Micro),
                (100.0, Scale::Milli),
            ];

            for (scaled_val, scale) in test_cases {
                let cap = CapacitorBuilder::new()
                    .val_scaled(scaled_val, scale)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                let expected = scaled_val.unscale(scale);
                cap.val().assert_approx_eq(
                    &expected,
                    DEFAULT_MARGIN,
                    "value_scaling_consistency",
                    "cap.val()",
                );
            }
        }
    }

    mod parametric_tests {
        use super::*;

        #[test]
        fn test_capacitor_values_parametric() {
            let test_values = vec![
                (1.0, Scale::Femto),
                (10.0, Scale::Femto),
                (100.0, Scale::Femto),
                (1.0, Scale::Pico),
                (10.0, Scale::Pico),
                (100.0, Scale::Pico),
                (1.0, Scale::Nano),
                (10.0, Scale::Nano),
            ];

            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            for (val, scale) in test_values {
                let cap = CapacitorBuilder::new()
                    .val_scaled(val, scale)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                let z = cap.z_at(&freq, 0);
                let actual_capacitance = val.unscale(scale);
                let expected_reactance =
                    -1.0 / (2.0 * std::f64::consts::PI * 1e9 * actual_capacitance);

                z.im.assert_approx_eq(
                    &expected_reactance,
                    RELAXED_MARGIN,
                    "capacitor_values_parametric",
                    "z.im",
                );
            }
        }

        #[test]
        fn test_resistor_values_parametric() {
            let resistances = vec![1.0, 10.0, 50.0, 75.0, 100.0, 1000.0, 10000.0];
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            for r_val in resistances {
                let res = ResistorBuilder::new()
                    .val(r_val)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                let z = res.z_at(&freq, 0);
                z.re.assert_approx_eq(&r_val, DEFAULT_MARGIN, "resistor_values_parametric", "z.re");
                z.im.assert_approx_eq(&0.0, DEFAULT_MARGIN, "resistor_values_parametric", "z.im");
            }
        }

        #[test]
        fn test_inductor_values_parametric() {
            let test_values = vec![
                (1.0, Scale::Pico),
                (10.0, Scale::Pico),
                (100.0, Scale::Pico),
                (1.0, Scale::Nano),
                (10.0, Scale::Nano),
                (100.0, Scale::Nano),
                (1.0, Scale::Micro),
            ];

            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            for (val, scale) in test_values {
                let ind = InductorBuilder::new()
                    .val_scaled(val, scale)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                let z = ind.z_at(&freq, 0);
                let actual_inductance = val.unscale(scale);
                let expected_reactance = 2.0 * std::f64::consts::PI * 1e9 * actual_inductance;

                z.im.assert_approx_eq(
                    &expected_reactance,
                    RELAXED_MARGIN,
                    "inductor_values_parametric",
                    "z.im",
                );
            }
        }

        #[test]
        fn test_frequency_sweep_parametric() {
            let frequencies = vec![
                1e3,   // 1 kHz
                1e6,   // 1 MHz
                10e6,  // 10 MHz
                100e6, // 100 MHz
                1e9,   // 1 GHz
                10e9,  // 10 GHz
                100e9, // 100 GHz
            ];

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            for &f in &frequencies {
                let freq = ArrayUnitValue::new_freq(&array![f], Scale::Base);
                let z = cap.z_at(&freq, 0);

                let expected_reactance = -1.0 / (2.0 * std::f64::consts::PI * f * 1e-12);
                z.im.assert_approx_eq(
                    &expected_reactance,
                    RELAXED_MARGIN,
                    "frequency_sweep_parametric",
                    "z.im",
                );
            }
        }

        #[test]
        fn test_port_impedances_parametric() {
            let impedances = vec![
                c64(50.0, 0.0),
                c64(75.0, 0.0),
                c64(100.0, 0.0),
                c64(50.0, 10.0),
                c64(50.0, -10.0),
                c64(25.0, 25.0),
            ];

            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            for z_val in impedances {
                let port = PortBuilder::new().z(z_val).nodes([1]).build().unwrap();

                let z = port.z_at(&freq, 0);
                assert_eq!(z, z_val);
            }
        }
    }

    mod circuit_scenarios {
        use super::*;

        #[test]
        fn test_rc_low_pass_filter() {
            // Test RC low-pass filter at cutoff frequency
            let r_val = 1000.0; // 1k ohm
            let c_val = 1.59e-7; // ~159 nF
            let fc = 1.0 / (2.0 * std::f64::consts::PI * r_val * c_val); // Cutoff frequency ~1 kHz

            let freq = ArrayUnitValue::new_freq(&array![fc], Scale::Base);
            let r = ResistorBuilder::new()
                .val(r_val)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c = CapacitorBuilder::new()
                .val(c_val)
                .nodes([1, 2])
                .build()
                .unwrap();

            let _z_r = r.z_at(&freq, 0);
            let z_c = c.z_at(&freq, 0);

            // At cutoff, |Xc| should equal R
            z_c.norm()
                .assert_approx_eq(&r_val, RELAXED_MARGIN, "rc_low_pass_filter", "z_c.norm()");
        }

        #[test]
        fn test_rl_high_pass_filter() {
            // Test RL high-pass filter
            let r_val = 100.0; // 100 ohm
            let l_val = 1.59e-3; // ~1.59 mH
            let fc = r_val / (2.0 * std::f64::consts::PI * l_val); // Cutoff frequency ~10 kHz

            let freq = ArrayUnitValue::new_freq(&array![fc], Scale::Base);
            let r = ResistorBuilder::new()
                .val(r_val)
                .nodes([1, 2])
                .build()
                .unwrap();
            let l = InductorBuilder::new()
                .val(l_val)
                .nodes([1, 2])
                .build()
                .unwrap();

            let _z_r = r.z_at(&freq, 0);
            let z_l = l.z_at(&freq, 0);

            // At cutoff, |Xl| should equal R
            z_l.norm().assert_approx_eq(
                &r_val,
                RELAXED_MARGIN,
                "rl_high_pass_filter",
                "z_l.norm()",
            );
        }

        #[test]
        fn test_series_resonance() {
            // LC series resonance: at resonance, X_L + X_C = 0
            let l_val = 1e-6; // 1 µH
            let c_val = 1e-12; // 1 pF
            let mid_val: f64 = l_val * c_val;
            let f_res = 1.0 / (2.0 * std::f64::consts::PI * mid_val.sqrt()); // ~159 MHz

            let freq = ArrayUnitValue::new_freq(&array![f_res], Scale::Base);
            let l = InductorBuilder::new()
                .val(l_val)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c = CapacitorBuilder::new()
                .val(c_val)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z_l = l.z_at(&freq, 0);
            let z_c = c.z_at(&freq, 0);
            let z_total = z_l + z_c;

            // At resonance, imaginary part should be very small
            z_total
                .im
                .assert_approx_eq(&0.0, RELAXED_MARGIN, "series_resonance", "z_total.im");
        }

        #[test]
        fn test_impedance_matching_network() {
            // Test 50 ohm to 75 ohm matching at 1 GHz
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let port_in = PortBuilder::new()
                .z(c64(50.0, 0.0))
                .nodes([1])
                .build()
                .unwrap();

            let port_out = PortBuilder::new()
                .z(c64(75.0, 0.0))
                .nodes([2])
                .build()
                .unwrap();

            assert_eq!(port_in.z_at(&freq, 0), c64(50.0, 0.0));
            assert_eq!(port_out.z_at(&freq, 0), c64(75.0, 0.0));
        }

        #[test]
        fn test_coupled_microstrip_lines() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0004)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val_scaled(0.77, Scale::Micro)
                .build()
                .unwrap();

            let mlin1 = MlinBuilder::new()
                .id("ML1")
                .width_val_scaled(10.0, Scale::Micro)
                .length_val_scaled(1000.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            let mlin2 = MlinBuilder::new()
                .id("ML2")
                .width_val_scaled(10.0, Scale::Micro)
                .length_val_scaled(1000.0, Scale::Micro)
                .sub(&sub)
                .nodes([3, 4])
                .build()
                .unwrap();

            // Both lines share the same substrate
            assert_eq!(mlin1.sub().er(), mlin2.sub().er());
            assert_ne!(mlin1.nodes(), mlin2.nodes());
        }

        // #[test]
        // fn test_microstrip_bend_corner_model() {
        //     let sub = MsubBuilder::new()
        //         .er(10.0)
        //         .height_val_scaled(50.0, Scale::Micro)
        //         .thickness_val_scaled(1.0, Scale::Micro)
        //         .build()
        //         .unwrap();

        //     let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

        //     // Test both mitered and non-mitered bends
        //     let bend_no_miter = MbendBuilder::new()
        //         .width_val_scaled(10.0, Scale::Micro)
        //         .miter(0.0)
        //         .sub(&sub)
        //         .build()
        //         .unwrap();

        //     let bend_miter = MbendBuilder::new()
        //         .width_val_scaled(10.0, Scale::Micro)
        //         .miter(0.5)
        //         .sub(&sub)
        //         .build()
        //         .unwrap();

        //     let c_no_miter = bend_no_miter.c(&freq);
        //     let c_miter = bend_miter.c(&freq);

        //     // The C matrices should be different
        //     assert_ne!(c_no_miter[[0, 0]], c_miter[[0, 0]]);
        // }

        #[test]
        fn test_transmission_line_cascade() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.00224)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val_scaled(0.77, Scale::Micro)
                .build()
                .unwrap();

            // Create three cascaded transmission line sections
            let tl1 = MlinBuilder::new()
                .id("TL1")
                .width_val_scaled(10.0, Scale::Micro)
                .length_val_scaled(500.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            let tl2 = MlinBuilder::new()
                .id("TL2")
                .width_val_scaled(10.0, Scale::Micro)
                .length_val_scaled(500.0, Scale::Micro)
                .sub(&sub)
                .nodes([2, 3])
                .build()
                .unwrap();

            let tl3 = MlinBuilder::new()
                .id("TL3")
                .width_val_scaled(10.0, Scale::Micro)
                .length_val_scaled(500.0, Scale::Micro)
                .sub(&sub)
                .nodes([3, 4])
                .build()
                .unwrap();

            // Verify cascade connectivity
            assert_eq!(tl1.nodes()[1], tl2.nodes()[0]);
            assert_eq!(tl2.nodes()[1], tl3.nodes()[0]);
        }
    }

    mod boundary_tests {
        use super::*;

        #[test]
        fn test_minimum_capacitance() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Femto) // 1 fF
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = cap.z_at(&freq, 0);
            assert!(z.is_finite());
            assert!(z.norm() > 1e5); // Very large impedance
        }

        #[test]
        fn test_maximum_capacitance() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Milli) // 1 mF
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = cap.z_at(&freq, 0);
            assert!(z.is_finite());
            assert!(z.norm() < 1.0); // Very small impedance
        }

        #[test]
        fn test_minimum_inductance() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let ind = InductorBuilder::new()
                .val_scaled(1.0, Scale::Pico) // 1 pH
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = ind.z_at(&freq, 0);
            assert!(z.is_finite());
            assert!(z.norm() < 1.0); // Very small impedance
        }

        #[test]
        fn test_maximum_inductance() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let ind = InductorBuilder::new()
                .val_scaled(1.0, Scale::Milli) // 1 mH
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = ind.z_at(&freq, 0);
            assert!(z.is_finite());
            assert!(z.norm() > 1e6); // Very large impedance
        }

        #[test]
        fn test_minimum_resistance() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let res = ResistorBuilder::new()
                .val_scaled(1.0, Scale::Milli) // 1 mOhm
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = res.z_at(&freq, 0);
            z.re.assert_approx_eq(&0.001, RELAXED_MARGIN, "minimum_resistance", "z.re");
        }

        #[test]
        fn test_maximum_resistance() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let res = ResistorBuilder::new()
                .val_scaled(1.0, Scale::Mega) // 1 MOhm
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = res.z_at(&freq, 0);
            z.re.assert_approx_eq(&1e6, RELAXED_MARGIN, "maximum_resistance", "z.re");
        }

        #[test]
        fn test_microstrip_minimum_width() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val_scaled(1.0, Scale::Micro)
                .build()
                .unwrap();

            let mlin = MlinBuilder::new()
                .width_val_scaled(0.1, Scale::Micro) // Very narrow
                .length_val_scaled(100.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert!(mlin.width() > 0.0);
            assert_eq!(mlin.width(), 0.1e-6);
        }

        #[test]
        fn test_microstrip_maximum_width() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0)
                .height_val_scaled(25.0, Scale::Micro)
                .thickness_val_scaled(1.0, Scale::Micro)
                .build()
                .unwrap();

            let mlin = MlinBuilder::new()
                .width_val_scaled(1000.0, Scale::Micro) // Very wide
                .length_val_scaled(100.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(mlin.width(), 1000e-6);
        }

        #[test]
        fn test_frequency_at_boundary_zero() {
            let freq = ArrayUnitValue::new_freq(&array![1e-12], Scale::Base); // Near zero
            let res = ResistorBuilder::new()
                .val(50.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = res.z_at(&freq, 0);
            assert!(z.is_finite());
        }

        #[test]
        fn test_frequency_at_boundary_high() {
            let freq = ArrayUnitValue::new_freq(&array![1e15], Scale::Base); // 1 PHz
            let res = ResistorBuilder::new()
                .val(50.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = res.z_at(&freq, 0);
            assert!(z.is_finite());
            z.re.assert_approx_eq(&50.0, DEFAULT_MARGIN, "frequency_at_boundary_high", "z.re");
        }
    }

    mod invariant_tests {
        use super::*;

        #[test]
        fn test_clone_consistency() {
            let cap1 = CapacitorBuilder::new()
                .id("C1")
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            let cap2 = cap1.clone();

            assert_eq!(cap1.id(), cap2.id());
            assert_eq!(cap1.val(), cap2.val());
            assert_eq!(cap1.nodes(), cap2.nodes());
        }

        #[test]
        fn test_equality_consistency() {
            let cap1 = CapacitorBuilder::new()
                .id("C1")
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            let cap2 = CapacitorBuilder::new()
                .id("C1")
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(cap1, cap2);
        }

        #[test]
        fn test_impedance_calculation_deterministic() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z1 = cap.z(&freq);
            let z2 = cap.z(&freq);
            let z3 = cap.z(&freq);

            assert_eq!(z1, z2);
            assert_eq!(z2, z3);
        }

        #[test]
        fn test_node_list_immutability_through_interface() {
            let cap: Capacitor<f64> = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let nodes1 = cap.nodes();
            let nodes2 = cap.nodes();

            assert_eq!(nodes1, nodes2);
        }

        #[test]
        fn test_c_matrix_dimensions_consistency() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let two_port_elements: Vec<Element<f64>> = vec![
                CapacitorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
                ResistorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
                InductorBuilder::new()
                    .val(1.0)
                    .nodes([1, 2])
                    .build()
                    .unwrap()
                    .into(),
            ];

            for elem in two_port_elements {
                let c = elem.c_at_freq(&freq, 0);
                assert_eq!(c.shape(), (2, 2));
            }

            let one_port_elements: Vec<Element<f64>> = vec![
                Ground::new().into(),
                PortBuilder::new()
                    .z(c64(50.0, 0.0))
                    .nodes([1])
                    .build()
                    .unwrap()
                    .into(),
            ];

            for elem in one_port_elements {
                let c = elem.c_at_freq(&freq, 0);
                assert_eq!(c.shape(), (1, 1));
            }
        }

        #[test]
        fn test_net_matrix_dimensions_consistency() {
            let freqs = array![1e9, 2e9, 5e9];
            let freq = ArrayUnitValue::new_freq(&freqs.clone(), Scale::Base);

            let cap = CapacitorBuilder::new()
                .val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let net = cap.net(&freq);

            assert_eq!(net.shape(), (3, 2, 2)); // [npts, 2, 2]
        }
    }

    mod documentation_examples {
        use super::*;

        #[test]
        fn test_basic_capacitor_usage() {
            // Example from documentation
            let cap = CapacitorBuilder::new()
                .id("C1")
                .val_scaled(10.0, Scale::Pico)
                .nodes([1, 2])
                .build()
                .unwrap();

            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let z = cap.z_at(&freq, 0);

            assert!(z.is_finite());
            assert!(z.im < 0.0); // Capacitive
        }

        #[test]
        fn test_basic_resistor_usage() {
            let res = ResistorBuilder::new()
                .id("R1")
                .val(50.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let z = res.z_at(&freq, 0);

            z.re.assert_approx_eq(&50.0, RELAXED_MARGIN, "basic_resistor_usage", "z.re");
        }

        #[test]
        fn test_basic_inductor_usage() {
            let ind = InductorBuilder::new()
                .id("L1")
                .val_scaled(1.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();

            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let z = ind.z_at(&freq, 0);

            assert!(z.is_finite());
            assert!(z.im > 0.0); // Inductive
        }

        #[test]
        fn test_basic_microstrip_usage() {
            let sub = MsubBuilder::new()
                .id("Sub1")
                .er(4.5)
                .tand(0.02)
                .height_val_scaled(1.6, Scale::Milli)
                .thickness_val_scaled(1.0, Scale::Micro)
                .build()
                .unwrap();

            let mlin = MlinBuilder::new()
                .id("TL1")
                .width_val_scaled(3.0, Scale::Milli)
                .length_val_scaled(10.0, Scale::Milli)
                .sub(&sub)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(mlin.id(), "TL1");
            assert_eq!(mlin.nodes(), vec![1, 2]);
        }
    }
}
