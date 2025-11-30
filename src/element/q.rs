use ndarray::Array1;
use std::fmt;
use std::str::FromStr;

use crate::frequency::Frequency;
use crate::unit::{Unit, UnitVal};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum QMode {
    #[default]
    Constant,
    ProportionalToFreq,
    ProportionalToSqrtFreq,
    ProportionalToExp,
}

impl QMode {
    pub fn to_str(&self) -> &str {
        match self {
            QMode::Constant => "Constant",
            QMode::ProportionalToFreq => "Proportional to frequency",
            QMode::ProportionalToSqrtFreq => "Proportional to square root of frequency",
            QMode::ProportionalToExp => "Proportional to exponential scaling",
        }
    }
}

impl FromStr for QMode {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Constant" | "constant" => Ok(QMode::Constant),
            "Freq" | "freq" => Ok(QMode::ProportionalToFreq),
            "Sqrt(Freq)" | "sqrt(Freq)" | "sqrt(freq)" => Ok(QMode::ProportionalToSqrtFreq),
            "Exp" | "exp" => Ok(QMode::ProportionalToExp),
            _ => Ok(QMode::Constant),
        }
    }
}

impl fmt::Display for QMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Q {
    rdc: UnitVal,
    q: f64,
    fq: Frequency,
    alpha: f64,
    mode: QMode,
}

impl Q {
    pub fn new(rdc: UnitVal, q: f64, fq: Frequency, alpha: f64, mode: QMode) -> Q {
        Q {
            rdc,
            q,
            fq,
            alpha,
            mode,
        }
    }

    pub fn rdc(&self) -> UnitVal {
        self.rdc.clone()
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn q(&self) -> f64 {
        self.q
    }

    pub fn fq(&self) -> Frequency {
        self.fq.clone()
    }

    pub fn wq(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.fq.npts(), |i| {
            2.0 * std::f64::consts::PI * self.fq.freq(i)
        })
    }

    pub fn wq_pt(&self, pt: usize) -> f64 {
        2.0 * std::f64::consts::PI * self.fq.freq(pt)
    }

    pub fn mode(&self) -> QMode {
        self.mode
    }

    pub fn set_rdc(&mut self, val: UnitVal) {
        self.rdc = val;
    }

    pub fn set_q(&mut self, val: f64) {
        self.q = val;
    }

    pub fn set_fq(&mut self, val: Frequency) {
        self.fq = val;
    }

    pub fn set_mode(&mut self, val: QMode) {
        self.mode = val;
    }
}

impl Default for Q {
    fn default() -> Self {
        Self {
            rdc: *UnitVal::default().set_unit(Unit::Ohm),
            q: 200.0,
            fq: Frequency::default(),
            alpha: 0.0,
            mode: QMode::Constant,
        }
    }
}

#[derive(Clone)]
pub struct QBuilder {
    rdc: UnitVal,
    q: f64,
    fq: Frequency,
    alpha: f64,
    mode: QMode,
}

impl QBuilder {
    pub fn new() -> Self {
        QBuilder::default()
    }

    pub fn rdc(mut self, rdc: UnitVal) -> Self {
        self.rdc = rdc;
        self
    }

    pub fn q(mut self, q: f64) -> Self {
        self.q = q;
        self
    }

    pub fn fq(mut self, fq: Frequency) -> Self {
        self.fq = fq;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn mode(mut self, mode: QMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn build(self) -> Q {
        Q {
            rdc: self.rdc,
            q: self.q,
            fq: self.fq,
            alpha: self.alpha,
            mode: self.mode,
        }
    }
}

impl Default for QBuilder {
    fn default() -> Self {
        Self {
            rdc: *UnitVal::default().set_unit(Unit::Ohm),
            q: 200.0,
            fq: Frequency::default(),
            alpha: 0.0,
            mode: QMode::Constant,
        }
    }
}
