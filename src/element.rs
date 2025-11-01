use crate::frequency::Frequency;
use crate::point::Point;
use crate::points::Points;
use crate::scale::Scale;
use crate::unit::{Unit, UnitVal, Unitized};
use num::complex::{Complex64, c64};
use serde::Serialize;
use std::fmt;
use std::str::FromStr;

pub mod capacitor;
pub mod ground;
pub mod inductor;
pub mod mbend;
pub mod mlef;
pub mod mlin;
pub mod msub;
pub mod port;
pub mod resistor;

pub use self::capacitor::{Capacitor, CapacitorBuilder};
pub use self::ground::Ground;
pub use self::inductor::{Inductor, InductorBuilder};
pub use self::mbend::{Mbend, MbendBuilder};
pub use self::mlef::{Mlef, MlefBuilder};
pub use self::mlin::{Mlin, MlinBuilder};
pub use self::msub::Msub;
pub use self::port::{Port, PortBuilder};
pub use self::resistor::{Resistor, ResistorBuilder};

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub enum ElemType {
    #[default]
    Capacitor,
    Ground,
    Inductor,
    Mbend,
    Mlef,
    Mlin,
    Msub,
    None,
    Port,
    Resistor,
}

impl ElemType {
    /// Convert ElemType to String
    pub fn to_str(&self) -> &str {
        match self {
            ElemType::Capacitor => "Capacitor",
            ElemType::Ground => "GND",
            ElemType::Inductor => "Inductor",
            ElemType::Mbend => "MBEND",
            ElemType::Mlef => "MLEF",
            ElemType::Mlin => "MLIN",
            ElemType::Msub => "MSUB",
            ElemType::None => "None",
            ElemType::Port => "Port",
            ElemType::Resistor => "Resistor",
        }
    }
}

impl FromStr for ElemType {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "C" | "c" | "Cap" | "cap" | "Capacitor" | "capacitor" => Ok(ElemType::Capacitor),
            "GND" | "gnd" | "Ground" | "ground" => Ok(ElemType::Ground),
            "L" | "l" | "Ind" | "ind" | "Inductor" | "inductor" => Ok(ElemType::Inductor),
            "mbend" | "MBEND" => Ok(ElemType::Mbend),
            "mlef" | "MLEF" => Ok(ElemType::Mlef),
            "mlin" | "MLIN" => Ok(ElemType::Mlin),
            "msub" | "MSUB" => Ok(ElemType::Msub),
            "none" | "None" => Ok(ElemType::None),
            "P" | "p" | "Port" | "port" => Ok(ElemType::Port),
            "R" | "r" | "Res" | "res" | "Resistor" | "resistor" => Ok(ElemType::Resistor),
            _ => Err("ElemType not recognized".to_string().into()),
        }
    }
}

impl fmt::Display for ElemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ElemType::Capacitor => write!(f, "Capacitor"),
            ElemType::Ground => write!(f, "GND"),
            ElemType::Inductor => write!(f, "Inductor"),
            ElemType::Mbend => write!(f, "MBEND"),
            ElemType::Mlef => write!(f, "MLEF"),
            ElemType::Mlin => write!(f, "MLIN"),
            ElemType::Msub => write!(f, "Msub"),
            ElemType::None => write!(f, "None"),
            ElemType::Port => write!(f, "Port"),
            ElemType::Resistor => write!(f, "Resistor"),
        }
    }
}

macro_rules! define_element_impl {
    (variants: [$($variant:ident),+ $(,)?]) => {
        impl Elem for Element {
            fn c(&self, freq: &Frequency) -> Point<Complex64> {
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

            fn net(&self, freq: &Frequency) -> Points<Complex64> {
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
    fn c(&self, freq: &Frequency) -> Point<Complex64>;
    fn c_at(&self, freq: &Frequency, j: usize, k: usize) -> Complex64;
    fn id(&self) -> String;
    fn elem(&self) -> ElemType;
    fn name(&self) -> &String;
    fn net(&self, freq: &Frequency) -> Points<Complex64>;
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
    Inductor(Inductor),
    Mbend(Mbend),
    Mlef(Mlef),
    Mlin(Mlin),
    Port(Port),
    Resistor(Resistor),
}

define_element_impl!(
    variants: [Capacitor, Ground, Inductor, Mbend, Mlef, Mlin, Port, Resistor]
);
/// Builder design pattern for Element
///
/// ## Example
/// ```
/// use ndarray::prelude::*;
/// use rfkit::prelude::*;
///
/// let elem1 = ElementBuilder::new().id("L1").val_scaled(1.0, Scale::Nano).nodes(vec![1,2]).build();
///
/// let elem2 = ElementBuilder::new().id("P1").z(50_f64.into()).nodes(vec![1]).build();
/// ```
#[derive(Default)]
pub struct ElementBuilder {
    id: String,
    unitval: UnitVal,
    width: UnitVal,
    length: UnitVal,
    gamma: Complex64,
    miter: bool,
    sub: Msub,
    er: f64,
    tand: f64,
    height: UnitVal,
    thickness: UnitVal,
    z: Complex64,
    nodes: Vec<usize>,
    elemtype: ElemType,
    z0: Complex64,
}

impl ElementBuilder {
    pub fn new() -> Self {
        ElementBuilder {
            id: "".to_string(),
            unitval: UnitVal::default(),
            width: *UnitVal::default().set_unit(Unit::Meter),
            length: *UnitVal::default().set_unit(Unit::Meter),
            gamma: Complex64::ZERO,
            miter: false,
            sub: Msub::default(),
            er: 1.0,
            tand: 0.0,
            height: UnitVal::default(),
            thickness: UnitVal::default(),
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
    pub fn val(mut self, val: f64) -> Self {
        self.unitval.set_val(val);
        self
    }

    /// Provide element value in scaled unit
    pub fn val_scaled(mut self, val: f64, unit: Scale) -> Self {
        self.unitval.set_scale(unit);
        self.unitval.set_val_scaled(val);
        self
    }

    /// Provide element UnitVal
    pub fn unitval(mut self, val: UnitVal) -> Self {
        self.unitval = val;
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

    /// Provide element width UnitVal
    pub fn width(mut self, val: UnitVal) -> Self {
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

    /// Provide element length UnitVal
    pub fn length(mut self, val: UnitVal) -> Self {
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
            ElemType::Inductor => Some(Element::Inductor(
                InductorBuilder::new()
                    .id(self.id.as_str())
                    .ind(self.unitval)
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
            _ => None,
        }
    }
}

/// Calculate exponent in radians for use in exp(-gamma * L) MLIN calculations
pub fn mlin_exp(len: UnitVal, gamma: Complex64) -> Complex64 {
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
