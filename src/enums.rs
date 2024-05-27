#![allow(unused)]
use core::{fmt, num};
use faer::{mat, scale, Mat};
use faer::complex_native::c64;
use simple_error::{bail, SimpleError};
use std::f64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unit {
    Atto,
    Femto,
    Pico,
    Nano,
    Micro,
    Milli,
    Centi,
    Base,
    Kilo,
    Mega,
    Giga,
    Tera,
}

impl Unit {
    pub fn from_str(val: &str) -> Result<Unit, SimpleError> {
        match val.to_lowercase().as_str() {
            "Atto" | "atto" | "a" => Ok(Unit::Atto),
            "Femto" | "femto" | "f" => Ok(Unit::Femto),
            "Pico" | "pico" | "p" => Ok(Unit::Pico),
            "Nano" | "nano" | "n" => Ok(Unit::Nano),
            "Micro" | "micro" | "u" => Ok(Unit::Micro),
            "Milli" | "milli" | "Mil" | "mil" | "m" => Ok(Unit::Milli),
            "Centi" | "centi" | "c" => Ok(Unit::Centi),
            "Kilo" | "kilo" | "k" => Ok(Unit::Kilo),
            "Mega" | "mega" | "M" => Ok(Unit::Mega),
            "Giga" | "giga" | "G" => Ok(Unit::Giga),
            "Tera" | "tera" | "T" => Ok(Unit::Tera),
            _ => Ok(Unit::Base),
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            Unit::Atto => "a",
            Unit::Femto => "f",
            Unit::Pico => "p",
            Unit::Nano => "n",
            Unit::Micro => "u",
            Unit::Milli => "m",
            Unit::Centi => "c",
            Unit::Base => "",
            Unit::Kilo => "k",
            Unit::Mega => "M",
            Unit::Giga => "G",
            Unit::Tera => "T",
        }
    }

    pub fn scale(&self) -> f64 {
        match self {
            Unit::Atto => 1e-19,
            Unit::Femto => 1e-15,
            Unit::Pico => 1e-12,
            Unit::Nano => 1e-9,
            Unit::Micro => 1e-6,
            Unit::Milli => 1e-3,
            Unit::Centi => 1e-2,
            Unit::Base => 1.0,
            Unit::Kilo => 1e3,
            Unit::Mega => 1e6,
            Unit::Giga => 1e9,
            Unit::Tera => 1e12,
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[derive(Clone, Copy)]
pub enum RFParameter {
    A,
    G,
    H,
    S,
    T,
    Y,
    Z,
}

impl RFParameter {
    // Convert text from a string into RFParameter type
    pub fn from_str(val: &str) -> Result<RFParameter, SimpleError> {
        match val.to_lowercase().as_str() {
            "A" | "a" | "ABCD" => Ok(RFParameter::A),
            "G" | "g" => Ok(RFParameter::G),
            "H" | "h" => Ok(RFParameter::H),
            "S" | "s" => Ok(RFParameter::S),
            "T" | "t" => Ok(RFParameter::T),
            "Y" | "y" => Ok(RFParameter::Y),
            "Z" | "z" => Ok(RFParameter::Z),
            _ => bail!("string not recognized"),
        }
    }

    // Convert text from a touchstone option line string into RFParameter type
    pub fn from_option_string(val: String) -> Result<RFParameter, SimpleError> {
        match val.to_lowercase().as_str() {
            "g" => Ok(RFParameter::G),
            "h" => Ok(RFParameter::H),
            "s" => Ok(RFParameter::S),
            "y" => Ok(RFParameter::Y),
            "z" => Ok(RFParameter::Z),
            _ => bail!("string not a valid option line type"),
        }
    }

    // Convert RFParameter to string
    pub fn to_str(&self) -> &str {
        match self {
            RFParameter::A => "ABCD",
            RFParameter::G => "Inverse Hybrid (g)",
            RFParameter::H => "Hybrid (h)",
            RFParameter::S => "S",
            RFParameter::T => "T",
            RFParameter::Y => "Y",
            RFParameter::Z => "Z",
        }
    }

    // Convert RFParameter to a touchstone option line string
    pub fn to_option_string(&self) -> Result<String, SimpleError> {
        match self {
            RFParameter::G => Ok("G".to_string()),
            RFParameter::H => Ok("H".to_string()),
            RFParameter::S => Ok("S".to_string()),
            RFParameter::Y => Ok("Y".to_string()),
            RFParameter::Z => Ok("Z".to_string()),
            _ => bail!("parameter type not a valid option line type"),
        }
    }
}

impl fmt::Display for RFParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[derive(Clone, Copy)]
pub enum RFDataFormat {
    RI,
    MA,
    DB,
}

impl RFDataFormat {
    // Convert text from a option line string into RFDataFormat type
    pub fn from_str(val: &str) -> Result<RFDataFormat, SimpleError> {
        match val {
            "RI" | "ri" => Ok(RFDataFormat::RI),
            "MA" | "ma" => Ok(RFDataFormat::MA),
            "DB" | "dB" | "db" => Ok(RFDataFormat::DB),
            _ => bail!("string not recognized"),
        }
    }

    // Convert RFDataFormat to a touchstone option line string
    pub fn to_str(&self) -> &str {
        match self {
            RFDataFormat::RI => "RI",
            RFDataFormat::MA => "MA",
            RFDataFormat::DB => "DB",
        }
    }

    pub fn parse(&self, x: f64, y: f64) -> c64 {
        match self {
            RFDataFormat::RI => c64 { re: x, im: y },
            RFDataFormat::MA => c64::from_polar(x, f64::to_radians(y)),
            RFDataFormat::DB => c64::from_polar(10_f64.powf(x / 20.0), f64::to_radians(y)),
        }
    }
}

impl fmt::Display for RFDataFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}
