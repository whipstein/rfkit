use crate::{
    frequency::Frequency,
    pts::Points,
    scale::Scale,
    unit::{Unit, UnitValue, Unitized},
};
use ndarray::prelude::*;
use num::complex::{Complex64, c64};
use serde::{
    Serialize,
    ser::{SerializeStruct, Serializer},
};
use std::{fmt, str::FromStr};

pub mod capacitor;
pub mod ground;
pub mod inductor;
pub mod mbend;
pub mod mlef;
pub mod mlin;
pub mod msub;
pub mod port;
pub mod q;
pub mod resistor;
pub mod short;
pub mod transformer;

pub use self::{
    capacitor::{Capacitor, CapacitorBuilder},
    ground::Ground,
    inductor::{Inductor, InductorBuilder},
    mbend::{Mbend, MbendBuilder},
    mlef::{Mlef, MlefBuilder},
    mlin::{Mlin, MlinBuilder},
    msub::{Msub, MsubBuilder},
    port::{Port, PortBuilder},
    q::{Q, QBuilder, QMode},
    resistor::{Resistor, ResistorBuilder},
    short::Short,
    transformer::{IdealTransformer, IdealTransformerBuilder, Transformer, TransformerBuilder},
};

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
        impl Elem for Element {
            fn c(&self, freq: &Frequency) -> Points<Complex64, Ix2> {
                match self {
                    $(
                        Element::$variant(elem) => elem.c(freq),
                    )+
                }
            }

            fn c_at(&self, freq: &Frequency, j: usize, k: usize) -> Complex64 {
                match self {
                    $(
                        Element::$variant(elem) => elem.c_at(freq, j, k),
                    )+
                }
            }

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

            fn name(&self) -> &String {
                match self {
                    $(
                        Element::$variant(elem) => elem.name(),
                    )+
                }
            }

            fn net(&self, freq: &Frequency) -> Points<Complex64, Ix3> {
                match self {
                    $(
                        Element::$variant(elem) => elem.net(freq),
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

            fn z(&self, freq: &Frequency) -> Complex64 {
                match self {
                    $(
                        Element::$variant(elem) => elem.z(freq),
                    )+
                }
            }

            fn z_at(&self, freq: &Frequency, i: usize) -> Complex64 {
                match self {
                    $(
                        Element::$variant(elem) => elem.z_at(freq, i),
                    )+
                }
            }

            fn set_id(&mut self, id: &str) {
                match self {
                    $(
                        Element::$variant(elem) => elem.set_id(id),
                    )+
                }
            }
        }
    };
}

pub trait Elem {
    fn c(&self, freq: &Frequency) -> Points<Complex64, Ix2>;
    fn c_at(&self, freq: &Frequency, j: usize, k: usize) -> Complex64;
    fn id(&self) -> String;
    fn elem(&self) -> ElemType;
    fn name(&self) -> &String;
    fn net(&self, freq: &Frequency) -> Points<Complex64, Ix3>;
    fn nodes(&self) -> Vec<usize>;
    fn z(&self, freq: &Frequency) -> Complex64;
    fn z_at(&self, freq: &Frequency, i: usize) -> Complex64;
    fn set_id(&mut self, id: &str);
}

pub trait Lumped: Elem + Unitized {
    const NODE_LEN: usize = 2;

    fn val(&self) -> f64;
    fn set_val(&mut self, val: f64);
}

pub trait Distributed: Elem + Unitized {
    const NODE_LEN: usize = 2;

    fn width(&self) -> f64;
    fn length(&self, freq: &Frequency) -> f64;
    fn sub(&self) -> &Msub;
    fn val(&self) -> f64;
    fn gamma(&self, freq: &Frequency) -> Complex64;
    fn er(&self, freq: &Frequency) -> f64;
    fn set_width_val(&mut self, val: f64);
    fn set_width_unit(&mut self, unit: Unit);
    fn set_length_val(&mut self, val: f64);
    fn set_length_unit(&mut self, unit: Unit);
}

pub trait Term: Elem {
    const NODE_LEN: usize = 1;

    fn val(&self) -> Complex64;
    fn set_val(&mut self, val: Complex64);
}

#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    Capacitor(Capacitor),
    Ground(Ground),
    IdealTransformer(IdealTransformer),
    Inductor(Inductor),
    Mbend(Mbend),
    Mlef(Mlef),
    Mlin(Mlin),
    Port(Port),
    Resistor(Resistor),
    Short(Short),
    Transformer(Transformer),
}

impl Element {
    pub fn val(&self) -> f64 {
        match self {
            Element::Capacitor(elem) => elem.val(),
            Element::Inductor(elem) => elem.val(),
            Element::Resistor(elem) => elem.val(),
            _ => 0.0,
        }
    }
}

impl Serialize for Element {
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

define_elem_impl!(
    variants: [Capacitor, Ground, IdealTransformer, Inductor, Mbend, Mlef, Mlin, Port, Resistor, Short, Transformer]
);

/// Builder design pattern for Element
///
/// ## Example
/// ```
/// use ndarray::prelude::*;
/// use rfkit::prelude::*;
///
/// let elem1 = ElementBuilder::new().id("L1").unitval_val_scaled(1.0, Scale::Nano).nodes(vec![1,2]).build();
///
/// let elem2 = ElementBuilder::new().id("P1").z(50_f64.into()).nodes(vec![1]).build();
/// ```
#[derive(Default)]
pub struct ElementBuilder {
    id: String,
    unitval: UnitValue,
    val: f64,
    n: Option<f64>,
    km: Option<f64>,
    m: Option<UnitValue>,
    l1: UnitValue,
    l2: Option<UnitValue>,
    q: Option<Q>,
    q1: Option<Q>,
    q2: Option<Q>,
    width: UnitValue,
    length: UnitValue,
    gamma: Complex64,
    miter: bool,
    sub: Msub,
    er: f64,
    tand: f64,
    height: UnitValue,
    thickness: UnitValue,
    freq: Frequency,
    z: Complex64,
    nodes: Vec<usize>,
    elemtype: ElemType,
    z0: Complex64,
}

impl ElementBuilder {
    pub fn new() -> Self {
        ElementBuilder {
            id: "".to_string(),
            unitval: UnitValue::default(),
            val: 1.0,
            n: None,
            km: None,
            m: None,
            l1: *UnitValue::default().set_unit(Unit::Henry),
            l2: None,
            q: None,
            q1: None,
            q2: None,
            width: *UnitValue::default().set_unit(Unit::Meter),
            length: *UnitValue::default().set_unit(Unit::Meter),
            gamma: Complex64::ZERO,
            miter: false,
            sub: Msub::default(),
            er: 1.0,
            tand: 0.0,
            height: UnitValue::default(),
            thickness: UnitValue::default(),
            freq: Frequency::default(),
            z: Complex64::ZERO,
            nodes: vec![],
            elemtype: ElemType::None,
            z0: c64(50.0, 0.0),
        }
    }

    /// Provide element id
    pub fn id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }

    /// Provide element type
    pub fn elem(mut self, elemtype: ElemType) -> Self {
        self.elemtype = elemtype;
        self
    }

    /// Provide element value in base unit
    pub fn unitval_val(mut self, val: f64) -> Self {
        self.unitval.set_val(val);
        self
    }

    /// Provide element value in scaled unit
    pub fn unitval_val_scaled(mut self, val: f64, unit: Scale) -> Self {
        self.unitval.set_scale(unit);
        self.unitval.set_val_scaled(val);
        self
    }

    /// Provide element UnitValue
    pub fn unitval(mut self, val: UnitValue) -> Self {
        self.unitval = val;
        self
    }

    /// Provide element value
    pub fn val(mut self, val: f64) -> Self {
        self.val = val;
        if self.elemtype == ElemType::IdealTransformer {
            self.n = Some(val);
        }
        self
    }

    /// Provide element N for Transformer
    pub fn n(mut self, val: f64) -> Self {
        self.n = Some(val);
        self
    }

    /// Provide element km for Transformer
    pub fn km(mut self, km: f64) -> Self {
        self.km = Some(km);
        self
    }

    /// Provide element M for Transformer
    pub fn m(mut self, ind: UnitValue) -> Self {
        self.m = Some(ind);
        self
    }

    /// Provide element M for Transformer
    pub fn m_val(mut self, ind: f64) -> Self {
        let _ = *self
            .m
            .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
            .set_val(ind);
        self
    }

    /// Provide element M for Transformer
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

    /// Provide element L1 for Transformer
    pub fn l1(mut self, ind: UnitValue) -> Self {
        self.l1 = ind;
        self
    }

    /// Provide element L1 for Transformer
    pub fn l1_val(mut self, ind: f64) -> Self {
        self.l1.set_val(ind);
        self
    }

    /// Provide element L1 for Transformer
    pub fn l1_val_scaled(mut self, ind: f64, scale: Scale) -> Self {
        self.l1.set_scale(scale);
        self.l1.set_val_scaled(ind);
        self
    }

    /// Provide element L2 for Transformer
    pub fn l2(mut self, ind: UnitValue) -> Self {
        self.l2 = Some(ind);
        self
    }

    /// Provide element L2 for Transformer
    pub fn l2_val(mut self, ind: f64) -> Self {
        let _ = *self
            .l2
            .get_or_insert(*UnitValue::default().set_unit(Unit::Henry))
            .set_val(ind);
        self
    }

    /// Provide element L2 for Transformer
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

    /// Provide element Q or both Q1 & Q2 for Transformer
    pub fn q(mut self, val: Q) -> Self {
        self.q = Some(val.clone());
        self.q1 = Some(val.clone());
        self.q2 = Some(val);
        self
    }

    /// Provide element Q1 for Transformer
    pub fn q1(mut self, val: Q) -> Self {
        self.q1 = Some(val);
        self
    }

    /// Provide element Q1 for Transformer
    pub fn q1_val(mut self, q: f64) -> Self {
        let _ = self.q1.get_or_insert(Q::default()).set_q(q);
        if let Some(val) = self.q1.as_mut() {
            val.set_fq(&self.freq);
        }
        self
    }

    /// Provide element Q2 for Transformer
    pub fn q2(mut self, val: Q) -> Self {
        self.q2 = Some(val);
        self
    }

    /// Provide element Q2 for Transformer
    pub fn q2_val(mut self, q: f64) -> Self {
        let _ = self.q2.get_or_insert(Q::default()).set_q(q);
        if let Some(val) = self.q2.as_mut() {
            val.set_fq(&self.freq);
        }
        self
    }

    /// Provide scale for element
    pub fn scale(mut self, scale: Scale) -> Self {
        self.unitval.set_scale(scale);
        self
    }

    /// Provide unit for element
    pub fn unit(mut self, unit: Unit) -> Self {
        self.unitval.set_unit(unit);
        self
    }

    /// Provide element width in base unit
    pub fn width_val(mut self, val: f64) -> Self {
        self.width.set_val(val);
        self
    }

    /// Provide element width in scaled unit
    pub fn width_scaled(mut self, val: f64, unit: Scale) -> Self {
        self.width.set_scale(unit);
        self.width.set_val_scaled(val);
        self
    }

    /// Provide element width UnitValue
    pub fn width(mut self, val: UnitValue) -> Self {
        self.width = val;
        self
    }

    /// Provide width scale for element
    pub fn width_scale(mut self, scale: Scale) -> Self {
        self.width.set_scale(scale);
        self
    }

    /// Provide width unit for element
    pub fn width_unit(mut self, unit: Unit) -> Self {
        self.width.set_unit(unit);
        self
    }

    /// Provide element length in base unit
    pub fn length_val(mut self, val: f64) -> Self {
        self.length.set_val(val);
        self
    }

    /// Provide element length in scaled unit
    pub fn length_scaled(mut self, val: f64, unit: Scale) -> Self {
        self.length.set_scale(unit);
        self.length.set_val_scaled(val);
        self
    }

    /// Provide element length UnitValue
    pub fn length(mut self, val: UnitValue) -> Self {
        self.length = val;
        self
    }

    /// Provide length scale for element
    pub fn length_scale(mut self, scale: Scale) -> Self {
        self.length.set_scale(scale);
        self
    }

    /// Provide length unit for element
    pub fn length_unit(mut self, unit: Unit) -> Self {
        self.length.set_unit(unit);
        self
    }

    /// Provide gamma value in line
    pub fn gamma(mut self, val: Complex64) -> Self {
        self.gamma = val;
        self
    }

    /// Is bend mitered?
    pub fn miter(mut self, miter: bool) -> Self {
        self.miter = miter;
        self
    }

    /// Provide substrate element for element
    pub fn sub(mut self, val: &Msub) -> Self {
        self.sub = val.clone();
        self
    }

    /// Provide er value for substrate
    pub fn er(mut self, val: f64) -> Self {
        self.er = val;
        self
    }

    /// Provide er Frequency
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

    /// Provide element value in impedance
    pub fn z(mut self, val: Complex64) -> Self {
        self.z = val;
        self
    }

    /// Provide z0 value in impedance
    pub fn z0(mut self, val: f64) -> Self {
        self.z0 = val.into();
        self
    }

    /// Provide nodes for element
    pub fn nodes(mut self, nodes: Vec<usize>) -> Self {
        self.nodes = nodes;
        self
    }

    pub fn build(self) -> Option<Element> {
        match self.elemtype {
            ElemType::Capacitor => Some(Element::Capacitor(
                CapacitorBuilder::new()
                    .id(self.id.as_str())
                    .cap(self.unitval)
                    .nodes(self.nodes.try_into().unwrap())
                    .z0(self.z0)
                    .build(),
            )),
            ElemType::Ground => Some(Element::Ground(Ground::new())),
            ElemType::IdealTransformer => Some(Element::IdealTransformer(
                IdealTransformerBuilder::new()
                    .id(self.id.as_str())
                    .n(self.n.unwrap())
                    .nodes(self.nodes.try_into().unwrap())
                    .z0(self.z0)
                    .build(),
            )),
            ElemType::Inductor => Some(Element::Inductor(
                InductorBuilder::new()
                    .id(self.id.as_str())
                    .ind(self.unitval)
                    .q(self.q)
                    .nodes(self.nodes.try_into().unwrap())
                    .z0(self.z0)
                    .build(),
            )),
            ElemType::Mbend => Some(Element::Mbend(
                MbendBuilder::new()
                    .id(self.id.as_str())
                    .width(self.width)
                    .miter(self.miter)
                    .sub(&self.sub)
                    .nodes(self.nodes.try_into().unwrap())
                    .build(),
            )),
            ElemType::Mlef => Some(Element::Mlef(
                MlefBuilder::new()
                    .id(self.id.as_str())
                    .width(self.width)
                    .length(self.length)
                    .sub(&self.sub)
                    .nodes(self.nodes.try_into().unwrap())
                    .build(),
            )),
            ElemType::Mlin => Some(Element::Mlin(
                MlinBuilder::new()
                    .id(self.id.as_str())
                    .width(self.width)
                    .length(self.length)
                    .sub(&self.sub)
                    .nodes(self.nodes.try_into().unwrap())
                    .build(),
            )),
            ElemType::Port => Some(Element::Port(
                PortBuilder::new()
                    .id(self.id.as_str())
                    .z(self.z)
                    .nodes(self.nodes.try_into().unwrap())
                    .build(),
            )),
            ElemType::Resistor => Some(Element::Resistor(
                ResistorBuilder::new()
                    .id(self.id.as_str())
                    .res(self.unitval)
                    .nodes(self.nodes.try_into().unwrap())
                    .z0(self.z0)
                    .build(),
            )),
            ElemType::Short => Some(Element::Short(Short::new(
                self.id,
                self.nodes.try_into().unwrap(),
                self.z0,
            ))),
            ElemType::Transformer => Some(Element::Transformer(
                TransformerBuilder::new()
                    .id(self.id.as_str())
                    .freq(&self.freq)
                    .n(self.n)
                    .km(self.km)
                    .m(self.m)
                    .l1(self.l1)
                    .l2(self.l2)
                    .q1(self.q1)
                    .q2(self.q2)
                    .nodes(self.nodes.try_into().unwrap())
                    .z0(self.z0)
                    .build(),
            )),
            _ => None,
        }
    }
}

/// Calculate exponent in radians for use in exp(-gamma * L) MLIN calculations
pub fn mlin_exp(len: UnitValue, gamma: Complex64) -> Complex64 {
    match len.unit() {
        Unit::Degree => -c64(0.0, len.val() * std::f64::consts::PI / 180.0),
        Unit::Radian => -c64(0.0, len.val()),
        Unit::Lambda => -c64(0.0, len.val() * 2.0 * std::f64::consts::PI),
        Unit::Meter => gamma * len.val(),
        Unit::Inch => gamma * len.val() * 0.0254,
        _ => panic!("not a valid unit for MLIN"),
    }
}

#[macro_export]
macro_rules! define_mlin_calcs {
    ($variant:ident) => {
        impl $variant {
            /// Z0 based on modified Kirschning & Jansen model
            pub fn z0(&self, freq: &Frequency) -> f64 {
                let w = self.width.val();
                let t = self.sub.thickness().val();
                let weff = self.w_eff(w, t);
                let er = self.sub.er();
                self.z0_dc(weff, t) / (self.er_eff(w, t, er, freq)).sqrt()
            }

            /// Z0 based on modified Kirschning & Jansen model
            fn z0_f(&self, w: f64, t: f64, er: f64, freq: &Frequency) -> f64 {
                self.z0_dc(w, t) / (self.er_eff(w, t, er, freq)).sqrt()
            }

            /// Quasi-static Z0 based on modified Kirschning & Jansen model
            fn z0_qs(&self, w: f64, t: f64, er: f64) -> f64 {
                self.z0_dc(w, t) / (self.er_effqs(w, t, er)).sqrt()
            }

            /// Z0 based on Hammerstad & Jensen model
            fn z0_dc(&self, w: f64, t: f64) -> f64 {
                let f1 = 6.0
                    + (2.0 * PI - 6.0)
                        * (-(30.666 * self.sub.height().val() / self.w_eff(w, t)).powf(0.7528))
                            .exp();

                60.0 * (f1 * self.sub.height().val() / self.w_eff(w, t)
                    + (1.0 + (2.0 * self.sub.height().val() / self.w_eff(w, t)).powi(2)).sqrt())
                .ln()
            }

            /// er_eff based on modified Kirschning & Jansen model
            pub fn er(&self, freq: &Frequency) -> f64 {
                self.er_eff(
                    self.w_eff(self.width.val(), self.sub.thickness().val()),
                    self.sub.thickness().val(),
                    self.sub.er(),
                    freq,
                )
            }

            /// er_eff based on modified Kirschning & Jansen model
            fn er_eff(&self, w: f64, t: f64, er: f64, freq: &Frequency) -> f64 {
                let u = self.w_eff(w, t) / self.sub.height().val();
                let fh =
                    Scale::Giga.scale(freq.freq(0)) * Scale::Centi.scale(self.sub.height().val());
                let p1 = 0.27488
                    + u * (0.6315 + 0.525 / (1.0 + 0.157 * fh).powi(20)
                        - 0.065683 * (-8.7513 * u).exp());
                let p2 = 0.33622 * (1.0 - (-0.03442 * self.sub.er()).exp());
                let p3 = 0.0363 * (-4.6 * u).exp() * (1.0 - (-(fh / 3.87).powf(4.97)).exp());
                let p4 = 1.0 + 2.751 * (1.0 - (-(self.sub.er() / 15.916).powi(8)).exp());
                let p = p1 * p2 * ((0.1844 + p3 * p4) * 10.0 * fh).powf(1.5763);

                self.sub.er() - (self.sub.er() - self.er_effqs(w, er, t)) / (1.0 + p)
            }

            /// static eeff based on Hammerstad & Jensen model
            fn er_effdc(&self, w: f64, er: f64) -> f64 {
                let u = w / self.sub.height().val();
                let a = 1.0
                    + 1.0 / 49.0 * ((u.powi(4) + (u / 52.0).powi(2)) / ((u).powi(4) + 0.432)).ln()
                    + 1.0 / 18.7 * (1.0 + (u / 18.1).powi(3)).ln();
                let b = 0.564 * ((er - 0.9) / (er - 3.0)).powf(0.053);
                let f = (1.0 + 10.0 / u).powf(-a * b);
                let q = (f + 1.0) / 2.0;

                1.0 + q * (er - 1.0)
            }

            /// quasi-static er_eff based on Bahl & Garg model
            fn er_effqs(&self, w: f64, er: f64, t: f64) -> f64 {
                self.er_effdc(w, er)
                    - ((er - 1.0) * t / self.sub.height().val())
                        / (4.6 * (w / self.sub.height().val()).sqrt())
            }

            /// effective width based on Kirschning and Jansen model
            fn w_eff(&self, w: f64, t: f64) -> f64 {
                if w / self.sub.height().val() <= 1.0 / (2.0 * PI) {
                    w + t / PI * (1.0 + (4.0 * PI * w / t).ln())
                } else {
                    w + t / PI * (1.0 + (2.0 * self.sub.height().val() / t).ln())
                }
            }

            /// Conductor loss based on Hammerstad & Jensen model in Nepers/meter
            pub fn alpha_cdc(&self, freq: &Frequency) -> f64 {
                let ki = (-1.2 * (self.z0(freq) / (120.0 * PI)).powf(0.7)).exp();

                self.sub.res_sh(freq) * self.sub.roughness(freq) * ki
                    / (self.z0(freq) * self.width.val())
            }

            /// Conductor loss based on Wheeler's model in Nepers/meter
            pub fn alpha_c(&self, freq: &Frequency) -> f64 {
                let r = 1.0
                    / (self.sub.conductivity().val()
                        * self.width.val()
                        * self.sub.thickness().val());
                let l = self.z0_qs(
                    self.w_eff(self.width.val(), self.sub.thickness().val()),
                    self.sub.thickness().val(),
                    1.0,
                ) / 3e8;
                let _ft1 = r / (2.0 * PI * l);
                let ft2 = 4.0
                    / (PI
                        * PI
                        * 4e-7
                        * self.sub.conductivity().val()
                        * self.sub.thickness().val().powi(2));

                if freq.freq(0) < ft2 {
                    1.0 / (2.0
                        * self.sub.conductivity().val()
                        * self.width.val()
                        * self.sub.thickness().val()
                        * self.z0(freq))
                } else {
                    let wpeff = match self.width.val() / self.sub.height().val() <= 1.0 / (2.0 * PI)
                    {
                        true => {
                            self.width.val()
                                + (self.sub.thickness().val() - self.sub.delta(freq)) / PI
                                    * (1.0
                                        + (4.0 * PI * (self.width.val() - self.sub.delta(freq))
                                            / (self.sub.thickness().val() - self.sub.delta(freq)))
                                        .ln())
                        }
                        false => {
                            self.width.val()
                                + (self.sub.thickness().val() - self.sub.delta(freq)) / PI
                                    * (1.0
                                        + (2.0 * (self.sub.height().val() - self.sub.delta(freq))
                                            / (self.sub.thickness().val() - self.sub.delta(freq)))
                                        .ln())
                        }
                    };
                    let f1 = 6.0
                        + (2.0 * PI - 6.0)
                            * (-(30.666 * (self.sub.height().val() + self.sub.delta(freq))
                                / wpeff)
                                .powf(0.7528))
                            .exp();
                    let zp0 = 60.0
                        * (f1 * (self.sub.height().val() + self.sub.delta(freq)) / wpeff
                            + (1.0
                                + (2.0 * (self.sub.height().val() + self.sub.delta(freq))
                                    / wpeff)
                                    .powi(2))
                            .sqrt())
                        .ln();
                    let deltaz = zp0
                        - self.z0_dc(
                            self.w_eff(self.width.val(), self.sub.thickness().val()),
                            self.sub.thickness().val(),
                        );
                    PI / freq.wavelength(1.0, 0)
                        * (self.er_eff(
                            self.w_eff(self.width.val(), self.sub.thickness().val())
                                - self.width.val(),
                            self.sub.thickness().val(),
                            self.sub.er(),
                            freq,
                        ))
                        .sqrt()
                        * deltaz
                        / self.z0_qs(
                            self.w_eff(self.width.val(), self.sub.thickness().val()),
                            self.sub.thickness().val(),
                            1.0,
                        )
                }
            }

            /// Dielectric loss based on Hammerstad & Jensen model in Nepers/meter
            pub fn alpha_ddc(&self, freq: &Frequency) -> f64 {
                PI * (self.er_effdc(self.width.val(), self.sub.er()) - 1.0)
                    * self.sub.er()
                    * self.sub.tand()
                    / (freq.wavelength(1.0, 0)
                        * (self.sub.er() - 1.0)
                        * (self.er_effdc(self.width.val(), self.sub.er())).sqrt())
            }

            /// Dielectric loss based on Awasthi, Singh, Sharma, Kumari & Verma in Nepers/meter
            pub fn alpha_d(&self, freq: &Frequency) -> f64 {
                if freq.wavelength(self.sub.er(), 0)
                    < Scale::Centi.unscale(30e-2) / self.sub.er().sqrt()
                {
                    self.sub.er() / (self.er(freq)).sqrt()
                        * (self.er(freq) - 1.0)
                        * PI
                        * self.sub.tand()
                        / ((self.sub.er() - 1.0) * freq.wavelength(1.0, 0))
                } else if freq.wavelength(self.sub.er(), 0)
                    < Scale::Centi.unscale(30e-3) / self.sub.er().sqrt()
                {
                    let sigma0 = self.sub.tand() * freq.w()[0] * self.sub.er() * 8.854e-12;
                    let sigma = sigma0 / (1.0 + 0.045 * Scale::Giga.scale(freq.freq(0))).sqrt();
                    let tand = sigma / (freq.w()[0] * self.sub.er() * 8.854e-12);

                    self.sub.er() / (self.er(freq)).sqrt() * (self.er(freq) - 1.0) * PI * tand
                        / ((self.sub.er() - 1.0) * freq.wavelength(1.0, 0))
                } else {
                    let f = Scale::Giga.scale(1e12);
                    let w = 2.0 * PI * 1e12;
                    let sigma0 = self.sub.tand() * w * self.sub.er() * 8.854e-12;
                    let sigma = sigma0 / (1.0 + 0.045 * f).sqrt();
                    let tand = sigma / (w * self.sub.er() * 8.854e-12);

                    self.sub.er() / (self.er(freq)).sqrt() * (self.er(freq) - 1.0) * PI * tand
                        / ((self.sub.er() - 1.0) * freq.wavelength(1.0, 0))
                }
            }

            pub fn r(&self, freq: &Frequency) -> f64 {
                2.0 * self.z0(freq) * self.alpha_c(freq)
            }

            pub fn l(&self, freq: &Frequency) -> f64 {
                self.z0(freq) * self.er(freq).sqrt() / 3e8
            }

            pub fn g(&self, freq: &Frequency) -> f64 {
                2.0 * self.alpha_d(freq) / self.z0(freq)
            }

            pub fn cap(&self, freq: &Frequency) -> f64 {
                self.er(freq).sqrt() / 3e8 * self.z0(freq)
            }

            pub fn alpha(&self, freq: &Frequency) -> f64 {
                self.r(freq) / (2.0 * self.z0(freq)) + self.g(freq) * self.z0(freq) / 2.0
            }

            pub fn beta(&self, freq: &Frequency) -> f64 {
                let r = self.r(freq);
                let l = self.l(freq);
                let g = self.g(freq);
                let c = self.cap(freq);
                let w = freq.w()[0];
                w * (l * c).sqrt()
                    * (1.0 - r * g / (4.0 * w.powi(2) * l * c)
                        + g.powi(2) / (8.0 * w.powi(2) * c.powi(2))
                        + r.powi(2) / (8.0 * w.powi(2) * l.powi(2)))
            }

            pub fn gamma(&self, freq: &Frequency) -> Complex64 {
                c64(self.alpha(freq), self.beta(freq))
            }
        }
    };
}

#[cfg(test)]
mod element_tests {
    use super::*;
    use crate::frequency::new_frequency;
    use crate::pts::Pts;
    use float_cmp::*;
    use std::f64::consts::PI;

    const DEFAULT_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-10,
        ulps: 4,
    };

    const RELAXED_MARGIN: F64Margin = F64Margin {
        epsilon: 1e-6,
        ulps: 10,
    };

    mod integration_tests {
        use super::*;
        use crate::frequency::new_frequency;

        #[test]
        fn test_rlc_series_combination() {
            let freq = new_frequency(array![1e9], Scale::Base);

            let r = ResistorBuilder::new().val(50.0).build();
            let l = InductorBuilder::new().val_scaled(1.0, Scale::Nano).build();
            let c = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            let z_r = r.z(&freq);
            let z_l = l.z(&freq);
            let z_c = c.z(&freq);
            let z_total = z_r + z_l + z_c;

            // Total impedance should be sum of individual impedances
            assert!(z_total.re > 0.0); // Resistive part
            assert!(z_total.im.abs() > 0.0); // Reactive part
        }

        #[test]
        fn test_element_type_differentiation() {
            let cap = CapacitorBuilder::new().build();
            let res = ResistorBuilder::new().build();
            let ind = InductorBuilder::new().build();
            let gnd = Ground::new();
            let port = PortBuilder::new().build();

            assert_ne!(cap.elem(), res.elem());
            assert_ne!(res.elem(), ind.elem());
            assert_ne!(ind.elem(), gnd.elem());
            assert_ne!(gnd.elem(), port.elem());
        }

        #[test]
        fn test_lumped_elements_share_c_matrix_structure() {
            let freq = new_frequency(array![1e9], Scale::Base);

            let cap = CapacitorBuilder::new().build();
            let res = ResistorBuilder::new().build();
            let ind = InductorBuilder::new().build();

            let c_cap = cap.c(&freq);
            let c_res = res.c(&freq);
            let c_ind = ind.c(&freq);

            // All should have same structure (different from impedance)
            assert!(approx_eq!(
                f64,
                c_cap[[0, 0]].re,
                c_res[[0, 0]].re,
                DEFAULT_MARGIN
            ));
            assert!(approx_eq!(
                f64,
                c_res[[0, 0]].re,
                c_ind[[0, 0]].re,
                DEFAULT_MARGIN
            ));
        }

        #[test]
        fn test_frequency_sweep_consistency() {
            let freqs = array![1e6, 1e7, 1e8, 1e9, 10e9];
            let freq = new_frequency(freqs.clone(), Scale::Base);

            let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

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
                .height_scaled(25.0, Scale::Micro)
                .thickness_scaled(0.77, Scale::Micro)
                .build();

            let mlin = MlinBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .length_scaled(1000.0, Scale::Micro)
                .sub(&sub)
                .build();

            let mlef = MlefBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .length_scaled(1000.0, Scale::Micro)
                .sub(&sub)
                .build();

            let mbend = MbendBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .sub(&sub)
                .build();

            // All should share the same substrate properties
            assert_eq!(mlin.sub().er(), sub.er());
            assert_eq!(mlef.sub().er(), sub.er());
            assert_eq!(mbend.sub().er(), sub.er());
        }
    }

    mod edge_case_tests {
        use super::*;

        #[test]
        fn test_very_small_component_values() {
            let freq = new_frequency(array![1e9], Scale::Base);

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Femto)
                .build();

            let z = cap.z(&freq);
            println!("\n\nz = {}\n\n", z.norm());
            assert!(z.norm() > 1e5); // Very large impedance for small cap
        }

        #[test]
        fn test_very_large_component_values() {
            let freq = new_frequency(array![1e9], Scale::Base);

            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Milli)
                .build();

            let z = cap.z(&freq);
            assert!(z.norm() < 1.0); // Very small impedance for large cap
        }

        #[test]
        fn test_zero_frequency_handling() {
            let freq = new_frequency(array![0.0], Scale::Base);
            let res = ResistorBuilder::new().val(50.0).build();

            let z = res.z(&freq);
            assert!(approx_eq!(f64, z.re, 50.0, DEFAULT_MARGIN));
        }

        #[test]
        fn test_very_high_frequency() {
            let freq = new_frequency(array![1e12], Scale::Base); // 1 THz

            let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            let z = cap.z(&freq);
            assert!(z.norm() < 1.0); // Should be very small at THz
        }

        #[test]
        fn test_node_numbering_non_sequential() {
            let nodes_list = vec![[0, 100], [5, 50], [1, 1000]];

            for nodes in nodes_list {
                let cap = CapacitorBuilder::new().nodes(nodes).build();
                assert_eq!(cap.nodes(), vec![nodes[0], nodes[1]]);
            }
        }

        #[test]
        fn test_same_node_both_terminals() {
            // This creates a short circuit - should still build
            let res = ResistorBuilder::new().nodes([5, 5]).build();

            assert_eq!(res.nodes(), vec![5, 5]);
        }

        #[test]
        fn test_element_name_and_id_consistency() {
            let cap = CapacitorBuilder::new().id("C_test").build();

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
                .build();

            assert_eq!(cap.id(), "C1");
            assert_eq!(cap.val_scaled(), 10.0);
            assert_eq!(cap.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_builder_partial_specification() {
            // Build with only some parameters specified
            let cap1 = CapacitorBuilder::new().id("C1").build();

            let cap2 = CapacitorBuilder::new().val_scaled(5.0, Scale::Pico).build();

            assert_eq!(cap1.id(), "C1");
            assert_eq!(cap2.val_scaled(), 5.0);
        }

        #[test]
        fn test_multiple_builds_from_same_pattern() {
            let base_builder = ResistorBuilder::new().val(100.0).z0(c64(50.0, 0.0));

            let res1 = base_builder.clone().id("R1").nodes([1, 2]).build();
            let res2 = base_builder.clone().id("R2").nodes([2, 3]).build();

            assert_eq!(res1.val(), 100.0);
            assert_eq!(res2.val(), 100.0);
            assert_ne!(res1.id(), res2.id());
        }
    }

    mod trait_tests {
        use super::*;

        #[test]
        fn test_lumped_trait_capacitor() {
            let mut cap = CapacitorBuilder::new()
                .val_scaled(10.0, Scale::Pico)
                .build();

            assert_eq!(cap.val_scaled(), 10.0);

            cap.set_val_scaled(20.0);
            assert_eq!(cap.val_scaled(), 20.0);
        }

        #[test]
        fn test_lumped_trait_resistor() {
            let mut res = ResistorBuilder::new().val(50.0).build();

            assert_eq!(res.val(), 50.0);

            res.set_val(75.0);
            assert_eq!(res.val(), 75.0);
        }

        #[test]
        fn test_lumped_trait_inductor() {
            let mut ind = InductorBuilder::new().val_scaled(1.0, Scale::Nano).build();

            assert_eq!(ind.val_scaled(), 1.0);

            ind.set_val_scaled(2.0);
            assert_eq!(ind.val_scaled(), 2.0);
        }

        #[test]
        fn test_unitized_trait() {
            let cap = CapacitorBuilder::new()
                .val_scaled(10.0, Scale::Pico)
                .build();

            assert_eq!(cap.scale(), Scale::Pico);
            assert_eq!(cap.unit(), Unit::Farad);
            assert_eq!(cap.val_scaled(), 10.0);
        }

        #[test]
        fn test_elem_trait_common_methods() {
            let freq = new_frequency(array![1e9], Scale::Base);

            let elements: Vec<Box<dyn Elem>> = vec![
                Box::new(CapacitorBuilder::new().build()),
                Box::new(ResistorBuilder::new().build()),
                Box::new(InductorBuilder::new().build()),
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
            let freq = new_frequency(freqs.clone(), Scale::Base);

            let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            for i in 0..freqs.len() {
                let z = cap.z_at(&freq, i);
                assert!(z.is_finite());
                assert!(!z.is_nan());
            }
        }

        #[test]
        fn test_matrix_operations_consistency() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new().build();

            let c1 = cap.c(&freq);
            let c2 = cap.c(&freq);

            // Multiple calls should return same result
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(c1[[i, j]], c2[[i, j]]);
                }
            }
        }

        #[test]
        fn test_c_at_equals_c_matrix_element() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new().build();

            let c_matrix = cap.c(&freq);

            for i in 0..2 {
                for j in 0..2 {
                    let c_at = cap.c_at(&freq, i, j);
                    assert_eq!(c_at, c_matrix[[i, j]]);
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
                let freq1 = new_frequency(array![f], Scale::Base);
                let freq2 = new_frequency(array![2.0 * f], Scale::Base);

                let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

                let z1 = cap.z(&freq1).norm();
                let z2 = cap.z(&freq2).norm();

                let ratio = z1 / z2;
                assert!(approx_eq!(f64, ratio, 2.0, epsilon = 1e-10));
            }
        }

        #[test]
        fn test_inductor_impedance_proportional_frequency() {
            // Property: |Z(2f)| = 2 * |Z(f)| for inductors
            let freqs = vec![1e6, 1e7, 1e8, 1e9];

            for &f in &freqs {
                let freq1 = new_frequency(array![f], Scale::Base);
                let freq2 = new_frequency(array![2.0 * f], Scale::Base);

                let ind = InductorBuilder::new().val_scaled(1.0, Scale::Nano).build();

                let z1 = ind.z(&freq1).norm();
                let z2 = ind.z(&freq2).norm();

                let ratio = z2 / z1;
                assert!(approx_eq!(f64, ratio, 2.0, epsilon = 1e-10));
            }
        }

        #[test]
        fn test_resistor_frequency_invariance() {
            // Property: Z(f1) = Z(f2) for all frequencies for resistors
            let freqs = array![1e3, 1e6, 1e9, 1e12];
            let freq = new_frequency(freqs.clone(), Scale::Base);

            let res = ResistorBuilder::new().val(100.0).build();

            let z_reference = res.z_at(&freq, 0);

            for i in 1..freqs.len() {
                let z = res.z_at(&freq, i);
                assert_eq!(z, z_reference);
            }
        }

        #[test]
        fn test_series_impedance_additivity() {
            // Property: Z_total = Z1 + Z2 + Z3 for series connection
            let freq = new_frequency(array![1e9], Scale::Base);

            let r = ResistorBuilder::new().val(50.0).build();
            let l = InductorBuilder::new().val_scaled(10.0, Scale::Nano).build();
            let c = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            let z_r = r.z(&freq);
            let z_l = l.z(&freq);
            let z_c = c.z(&freq);
            let z_series = z_r + z_l + z_c;

            // Real part should equal resistor value
            assert!(approx_eq!(f64, z_series.re, 50.0, epsilon = 1e-10));

            // Imaginary part should be sum of L and C reactances
            let expected_im = z_l.im + z_c.im;
            assert!(approx_eq!(f64, z_series.im, expected_im, epsilon = 1e-6));
        }

        #[test]
        fn test_capacitor_reactance_sign() {
            // Property: Capacitive reactance is always negative
            let freqs = array![1e6, 1e7, 1e8, 1e9, 1e10];
            let freq = new_frequency(freqs.clone(), Scale::Base);

            let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            for i in 0..freqs.len() {
                let z = cap.z_at(&freq, i);
                assert!(z.im < 0.0, "Capacitive reactance should be negative");
            }
        }

        #[test]
        fn test_inductor_reactance_sign() {
            // Property: Inductive reactance is always positive
            let freqs = array![1e6, 1e7, 1e8, 1e9, 1e10];
            let freq = new_frequency(freqs.clone(), Scale::Base);

            let ind = InductorBuilder::new().val_scaled(1.0, Scale::Nano).build();

            for i in 0..freqs.len() {
                let z = ind.z_at(&freq, i);
                assert!(z.im > 0.0, "Inductive reactance should be positive");
            }
        }

        #[test]
        fn test_c_matrix_symmetry_lumped() {
            // Property: C matrix should be symmetric for lumped elements
            let freq = new_frequency(array![1e9], Scale::Base);

            let elements = vec![
                Box::new(CapacitorBuilder::new().build()) as Box<dyn Elem>,
                Box::new(ResistorBuilder::new().build()) as Box<dyn Elem>,
                Box::new(InductorBuilder::new().build()) as Box<dyn Elem>,
            ];

            for elem in elements {
                let c = elem.c(&freq);
                assert_eq!(c[[0, 1]], c[[1, 0]], "C matrix should be symmetric");
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
                    .build();

                let expected = scaled_val * scale.multiplier();
                assert!(approx_eq!(f64, cap.val(), expected, epsilon = 1e-20));
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

            let freq = new_frequency(array![1e9], Scale::Base);

            for (val, scale) in test_values {
                let cap = CapacitorBuilder::new().val_scaled(val, scale).build();

                let z = cap.z(&freq);
                let actual_capacitance = val * scale.multiplier();
                let expected_reactance = -1.0 / (2.0 * PI * 1e9 * actual_capacitance);

                assert!(approx_eq!(f64, z.im, expected_reactance, epsilon = 1e-6));
            }
        }

        #[test]
        fn test_resistor_values_parametric() {
            let resistances = vec![1.0, 10.0, 50.0, 75.0, 100.0, 1000.0, 10000.0];
            let freq = new_frequency(array![1e9], Scale::Base);

            for r_val in resistances {
                let res = ResistorBuilder::new().val(r_val).build();

                let z = res.z(&freq);
                assert!(approx_eq!(f64, z.re, r_val, epsilon = 1e-10));
                assert!(approx_eq!(f64, z.im, 0.0, epsilon = 1e-10));
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

            let freq = new_frequency(array![1e9], Scale::Base);

            for (val, scale) in test_values {
                let ind = InductorBuilder::new().val_scaled(val, scale).build();

                let z = ind.z(&freq);
                let actual_inductance = val * scale.multiplier();
                let expected_reactance = 2.0 * PI * 1e9 * actual_inductance;

                assert!(approx_eq!(f64, z.im, expected_reactance, epsilon = 1e-6));
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

            let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            for &f in &frequencies {
                let freq = new_frequency(array![f], Scale::Base);
                let z = cap.z(&freq);

                let expected_reactance = -1.0 / (2.0 * PI * f * 1e-12);
                assert!(approx_eq!(f64, z.im, expected_reactance, epsilon = 1e-6));
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

            let freq = new_frequency(array![1e9], Scale::Base);

            for z_val in impedances {
                let port = PortBuilder::new().z(z_val).build();

                let z = port.z(&freq);
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
            let fc = 1.0 / (2.0 * PI * r_val * c_val); // Cutoff frequency ~1 kHz

            let freq = new_frequency(array![fc], Scale::Base);
            let r = ResistorBuilder::new().val(r_val).build();
            let c = CapacitorBuilder::new().val(c_val).build();

            let _z_r = r.z(&freq);
            let z_c = c.z(&freq);

            // At cutoff, |Xc| should equal R
            assert!(approx_eq!(f64, z_c.norm(), r_val, epsilon = 1.0));
        }

        #[test]
        fn test_rl_high_pass_filter() {
            // Test RL high-pass filter
            let r_val = 100.0; // 100 ohm
            let l_val = 1.59e-3; // ~1.59 mH
            let fc = r_val / (2.0 * PI * l_val); // Cutoff frequency ~10 kHz

            let freq = new_frequency(array![fc], Scale::Base);
            let r = ResistorBuilder::new().val(r_val).build();
            let l = InductorBuilder::new().val(l_val).build();

            let _z_r = r.z(&freq);
            let z_l = l.z(&freq);

            // At cutoff, |Xl| should equal R
            assert!(approx_eq!(f64, z_l.norm(), r_val, epsilon = 1.0));
        }

        #[test]
        fn test_series_resonance() {
            // LC series resonance: at resonance, X_L + X_C = 0
            let l_val = 1e-6; // 1 µH
            let c_val = 1e-12; // 1 pF
            let mid_val: f64 = l_val * c_val;
            let f_res = 1.0 / (2.0 * PI * mid_val.sqrt()); // ~159 MHz

            let freq = new_frequency(array![f_res], Scale::Base);
            let l = InductorBuilder::new().val(l_val).build();
            let c = CapacitorBuilder::new().val(c_val).build();

            let z_l = l.z(&freq);
            let z_c = c.z(&freq);
            let z_total = z_l + z_c;

            // At resonance, imaginary part should be very small
            assert!(approx_eq!(f64, z_total.im, 0.0, epsilon = 100.0));
        }

        #[test]
        fn test_impedance_matching_network() {
            // Test 50 ohm to 75 ohm matching at 1 GHz
            let freq = new_frequency(array![1e9], Scale::Base);

            let port_in = PortBuilder::new().z(c64(50.0, 0.0)).nodes([1]).build();

            let port_out = PortBuilder::new().z(c64(75.0, 0.0)).nodes([2]).build();

            assert_eq!(port_in.z(&freq), c64(50.0, 0.0));
            assert_eq!(port_out.z(&freq), c64(75.0, 0.0));
        }

        #[test]
        fn test_coupled_microstrip_lines() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .tand(0.0004)
                .height_scaled(25.0, Scale::Micro)
                .thickness_scaled(0.77, Scale::Micro)
                .build();

            let mlin1 = MlinBuilder::new()
                .id("ML1")
                .width_scaled(10.0, Scale::Micro)
                .length_scaled(1000.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build();

            let mlin2 = MlinBuilder::new()
                .id("ML2")
                .width_scaled(10.0, Scale::Micro)
                .length_scaled(1000.0, Scale::Micro)
                .sub(&sub)
                .nodes([3, 4])
                .build();

            // Both lines share the same substrate
            assert_eq!(mlin1.sub().er(), mlin2.sub().er());
            assert_ne!(mlin1.nodes(), mlin2.nodes());
        }

        #[test]
        fn test_microstrip_bend_corner_model() {
            let sub = MsubBuilder::new()
                .er(10.0)
                .height_scaled(50.0, Scale::Micro)
                .thickness_scaled(1.0, Scale::Micro)
                .build();

            let freq = new_frequency(array![1e9], Scale::Base);

            // Test both mitered and non-mitered bends
            let bend_no_miter = MbendBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .miter(false)
                .sub(&sub)
                .build();

            let bend_miter = MbendBuilder::new()
                .width_scaled(10.0, Scale::Micro)
                .miter(true)
                .sub(&sub)
                .build();

            let c_no_miter = bend_no_miter.c(&freq);
            let c_miter = bend_miter.c(&freq);

            // The C matrices should be different
            assert_ne!(c_no_miter[[0, 0]], c_miter[[0, 0]]);
        }

        #[test]
        fn test_transmission_line_cascade() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .height_scaled(25.0, Scale::Micro)
                .thickness_scaled(0.77, Scale::Micro)
                .build();

            // Create three cascaded transmission line sections
            let tl1 = MlinBuilder::new()
                .id("TL1")
                .width_scaled(10.0, Scale::Micro)
                .length_scaled(500.0, Scale::Micro)
                .sub(&sub)
                .nodes([1, 2])
                .build();

            let tl2 = MlinBuilder::new()
                .id("TL2")
                .width_scaled(10.0, Scale::Micro)
                .length_scaled(500.0, Scale::Micro)
                .sub(&sub)
                .nodes([2, 3])
                .build();

            let tl3 = MlinBuilder::new()
                .id("TL3")
                .width_scaled(10.0, Scale::Micro)
                .length_scaled(500.0, Scale::Micro)
                .sub(&sub)
                .nodes([3, 4])
                .build();

            // Verify cascade connectivity
            assert_eq!(tl1.nodes()[1], tl2.nodes()[0]);
            assert_eq!(tl2.nodes()[1], tl3.nodes()[0]);
        }
    }

    mod boundary_tests {
        use super::*;

        #[test]
        fn test_minimum_capacitance() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Femto) // 1 fF
                .build();

            let z = cap.z(&freq);
            assert!(z.is_finite());
            assert!(z.norm() > 1e5); // Very large impedance
        }

        #[test]
        fn test_maximum_capacitance() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new()
                .val_scaled(1.0, Scale::Milli) // 1 mF
                .build();

            let z = cap.z(&freq);
            assert!(z.is_finite());
            assert!(z.norm() < 1.0); // Very small impedance
        }

        #[test]
        fn test_minimum_inductance() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let ind = InductorBuilder::new()
                .val_scaled(1.0, Scale::Pico) // 1 pH
                .build();

            let z = ind.z(&freq);
            assert!(z.is_finite());
            assert!(z.norm() < 1.0); // Very small impedance
        }

        #[test]
        fn test_maximum_inductance() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let ind = InductorBuilder::new()
                .val_scaled(1.0, Scale::Milli) // 1 mH
                .build();

            let z = ind.z(&freq);
            assert!(z.is_finite());
            assert!(z.norm() > 1e6); // Very large impedance
        }

        #[test]
        fn test_minimum_resistance() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let res = ResistorBuilder::new()
                .val_scaled(1.0, Scale::Milli) // 1 mOhm
                .build();

            let z = res.z(&freq);
            assert!(approx_eq!(f64, z.re, 0.001, epsilon = 1e-10));
        }

        #[test]
        fn test_maximum_resistance() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let res = ResistorBuilder::new()
                .val_scaled(1.0, Scale::Mega) // 1 MOhm
                .build();

            let z = res.z(&freq);
            assert!(approx_eq!(f64, z.re, 1e6, epsilon = 1e-4));
        }

        #[test]
        fn test_microstrip_minimum_width() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .height_scaled(25.0, Scale::Micro)
                .build();

            let mlin = MlinBuilder::new()
                .width_scaled(0.1, Scale::Micro) // Very narrow
                .length_scaled(100.0, Scale::Micro)
                .sub(&sub)
                .build();

            assert!(mlin.width() > 0.0);
            assert_eq!(mlin.width(), 0.1e-6);
        }

        #[test]
        fn test_microstrip_maximum_width() {
            let sub = MsubBuilder::new()
                .er(12.4)
                .height_scaled(25.0, Scale::Micro)
                .build();

            let mlin = MlinBuilder::new()
                .width_scaled(1000.0, Scale::Micro) // Very wide
                .length_scaled(100.0, Scale::Micro)
                .sub(&sub)
                .build();

            assert_eq!(mlin.width(), 1000e-6);
        }

        #[test]
        fn test_frequency_at_boundary_zero() {
            let freq = new_frequency(array![1e-12], Scale::Base); // Near zero
            let res = ResistorBuilder::new().val(50.0).build();

            let z = res.z(&freq);
            assert!(z.is_finite());
        }

        #[test]
        fn test_frequency_at_boundary_high() {
            let freq = new_frequency(array![1e15], Scale::Base); // 1 PHz
            let res = ResistorBuilder::new().val(50.0).build();

            let z = res.z(&freq);
            assert!(z.is_finite());
            assert!(approx_eq!(f64, z.re, 50.0, epsilon = 1e-10));
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
                .build();

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
                .build();

            let cap2 = CapacitorBuilder::new()
                .id("C1")
                .val_scaled(10.0, Scale::Pico)
                .build();

            assert_eq!(cap1, cap2);
        }

        #[test]
        fn test_impedance_calculation_deterministic() {
            let freq = new_frequency(array![1e9], Scale::Base);
            let cap = CapacitorBuilder::new().val_scaled(1.0, Scale::Pico).build();

            let z1 = cap.z(&freq);
            let z2 = cap.z(&freq);
            let z3 = cap.z(&freq);

            assert_eq!(z1, z2);
            assert_eq!(z2, z3);
        }

        #[test]
        fn test_node_list_immutability_through_interface() {
            let cap = CapacitorBuilder::new().nodes([1, 2]).build();

            let nodes1 = cap.nodes();
            let nodes2 = cap.nodes();

            assert_eq!(nodes1, nodes2);
        }

        #[test]
        fn test_c_matrix_dimensions_consistency() {
            let freq = new_frequency(array![1e9], Scale::Base);

            let two_port_elements: Vec<Box<dyn Elem>> = vec![
                Box::new(CapacitorBuilder::new().build()),
                Box::new(ResistorBuilder::new().build()),
                Box::new(InductorBuilder::new().build()),
            ];

            for elem in two_port_elements {
                let c = elem.c(&freq);
                assert_eq!(c.shape(), (2, 2));
            }

            let one_port_elements: Vec<Box<dyn Elem>> = vec![
                Box::new(Ground::new()),
                Box::new(PortBuilder::new().build()),
            ];

            for elem in one_port_elements {
                let c = elem.c(&freq);
                assert_eq!(c.shape(), (1, 1));
            }
        }

        #[test]
        fn test_net_matrix_dimensions_consistency() {
            let freqs = array![1e9, 2e9, 5e9];
            let freq = new_frequency(freqs.clone(), Scale::Base);

            let cap = CapacitorBuilder::new().build();
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
                .build();

            let freq = new_frequency(array![1e9], Scale::Base);
            let z = cap.z(&freq);

            assert!(z.is_finite());
            assert!(z.im < 0.0); // Capacitive
        }

        #[test]
        fn test_basic_resistor_usage() {
            let res = ResistorBuilder::new()
                .id("R1")
                .val(50.0)
                .nodes([1, 2])
                .build();

            let freq = new_frequency(array![1e9], Scale::Base);
            let z = res.z(&freq);

            assert!(approx_eq!(f64, z.re, 50.0, epsilon = 1e-10));
        }

        #[test]
        fn test_basic_inductor_usage() {
            let ind = InductorBuilder::new()
                .id("L1")
                .val_scaled(1.0, Scale::Nano)
                .nodes([1, 2])
                .build();

            let freq = new_frequency(array![1e9], Scale::Base);
            let z = ind.z(&freq);

            assert!(z.is_finite());
            assert!(z.im > 0.0); // Inductive
        }

        #[test]
        fn test_basic_microstrip_usage() {
            let sub = MsubBuilder::new()
                .id("Sub1")
                .er(4.5)
                .tand(0.02)
                .height_scaled(1.6, Scale::Milli)
                .build();

            let mlin = MlinBuilder::new()
                .id("TL1")
                .width_scaled(3.0, Scale::Milli)
                .length_scaled(10.0, Scale::Milli)
                .sub(&sub)
                .nodes([1, 2])
                .build();

            assert_eq!(mlin.id(), "TL1");
            assert_eq!(mlin.nodes(), vec![1, 2]);
        }
    }
}
