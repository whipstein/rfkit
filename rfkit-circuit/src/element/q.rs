use rfkit_base::prelude::*;
use std::{fmt, str::FromStr};

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
pub struct Q<T: RealScalar> {
    rdc: ScalarUnitValue<T>,
    q: T,
    fq: ScalarUnitValue<T>,
    alpha: T,
    mode: QMode,
}

impl<T: RealScalar> Q<T> {
    pub fn new(
        rdc: ScalarUnitValue<T>,
        q: T,
        fq: ScalarUnitValue<T>,
        alpha: T,
        mode: QMode,
    ) -> Q<T> {
        Q {
            rdc,
            q,
            fq,
            alpha,
            mode,
        }
    }

    pub fn builder() -> QBuilder<T> {
        QBuilder::new()
    }

    pub fn rdc(&self) -> ScalarUnitValue<T> {
        self.rdc.clone()
    }

    pub fn alpha(&self) -> T {
        self.alpha
    }

    pub fn q(&self) -> T {
        self.q
    }

    pub fn fq(&self) -> ScalarUnitValue<T> {
        self.fq
    }

    pub fn wq(&self) -> T {
        self.fq.freq() * 2.0 * std::f64::consts::PI
    }

    pub fn mode(&self) -> QMode {
        self.mode
    }

    pub fn set_rdc(&mut self, val: &ScalarUnitValue<T>) {
        self.rdc = val.clone();
    }

    pub fn set_q(&mut self, val: T) {
        self.q = val;
    }

    pub fn set_fq(&mut self, val: &ScalarUnitValue<T>) {
        self.fq = val.clone();
    }

    pub fn set_mode(&mut self, val: QMode) {
        self.mode = val;
    }
}

impl<T: RealScalar> Default for Q<T> {
    fn default() -> Self {
        Self {
            rdc: *ScalarUnitValue::default().set_unit(Unit::Ohm),
            q: T::from_f64(200.0),
            fq: *ScalarUnitValue::default().set_unit(Unit::Hz),
            alpha: T::ZERO,
            mode: QMode::Constant,
        }
    }
}

#[derive(Clone)]
pub struct QBuilder<T: RealScalar> {
    rdc: ScalarUnitValue<T>,
    q: Option<T>,
    fq: Option<ScalarUnitValue<T>>,
    alpha: T,
    mode: QMode,
}

impl<T: RealScalar> QBuilder<T> {
    pub fn new() -> Self {
        QBuilder::default()
    }

    pub fn rdc(mut self, rdc: &ScalarUnitValue<T>) -> Self {
        self.rdc = rdc.clone();
        self
    }

    pub fn q(mut self, q: T) -> Self {
        self.q = Some(q);
        self
    }

    pub fn fq(mut self, fq: &ScalarUnitValue<T>) -> Self {
        self.fq = Some(fq.clone());
        self
    }

    pub fn alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn mode(mut self, mode: QMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn build(self) -> Result<Q<T>, String> {
        let elem = "QBuilder";
        let q = self.q.ok_or(format!("{elem}: value for q is required"))?;
        let fq = self.fq.ok_or(format!("{elem}: value for fq is required"))?;
        Ok(Q {
            rdc: self.rdc,
            q,
            fq,
            alpha: self.alpha,
            mode: self.mode,
        })
    }
}

impl<T: RealScalar> Default for QBuilder<T> {
    fn default() -> Self {
        Self {
            rdc: *ScalarUnitValue::default().set_unit(Unit::Ohm),
            q: None,
            fq: None,
            alpha: T::ZERO,
            mode: QMode::Constant,
        }
    }
}
