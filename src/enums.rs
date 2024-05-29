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
        match val {
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

    pub fn parse_tuple(&self, xy: (f64, f64)) -> c64 {
        self.parse(xy.0, xy.1)
    }
}

impl fmt::Display for RFDataFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::comp_point_c64;

    #[test]
    fn parse_unit() {
        assert_eq!(Unit::Base, Unit::from_str("").unwrap());
        assert_eq!(Unit::Base, Unit::from_str("x").unwrap());
        assert_eq!(Unit::Atto, Unit::from_str("atto").unwrap());
        assert_eq!(Unit::Atto, Unit::from_str("Atto").unwrap());
        assert_eq!(Unit::Atto, Unit::from_str("a").unwrap());
        assert_eq!(Unit::Femto, Unit::from_str("Femto").unwrap());
        assert_eq!(Unit::Femto, Unit::from_str("femto").unwrap());
        assert_eq!(Unit::Femto, Unit::from_str("f").unwrap());
        assert_eq!(Unit::Pico, Unit::from_str("Pico").unwrap());
        assert_eq!(Unit::Pico, Unit::from_str("pico").unwrap());
        assert_eq!(Unit::Pico, Unit::from_str("p").unwrap());
        assert_eq!(Unit::Nano, Unit::from_str("Nano").unwrap());
        assert_eq!(Unit::Nano, Unit::from_str("nano").unwrap());
        assert_eq!(Unit::Nano, Unit::from_str("n").unwrap());
        assert_eq!(Unit::Micro, Unit::from_str("Micro").unwrap());
        assert_eq!(Unit::Micro, Unit::from_str("micro").unwrap());
        assert_eq!(Unit::Micro, Unit::from_str("u").unwrap());
        assert_eq!(Unit::Milli, Unit::from_str("Milli").unwrap());
        assert_eq!(Unit::Milli, Unit::from_str("milli").unwrap());
        assert_eq!(Unit::Milli, Unit::from_str("Mil").unwrap());
        assert_eq!(Unit::Milli, Unit::from_str("mil").unwrap());
        assert_eq!(Unit::Milli, Unit::from_str("m").unwrap());
        assert_eq!(Unit::Centi, Unit::from_str("Centi").unwrap());
        assert_eq!(Unit::Centi, Unit::from_str("centi").unwrap());
        assert_eq!(Unit::Centi, Unit::from_str("c").unwrap());
        assert_eq!(Unit::Kilo, Unit::from_str("Kilo").unwrap());
        assert_eq!(Unit::Kilo, Unit::from_str("kilo").unwrap());
        assert_eq!(Unit::Kilo, Unit::from_str("k").unwrap());
        assert_eq!(Unit::Mega, Unit::from_str("Mega").unwrap());
        assert_eq!(Unit::Mega, Unit::from_str("mega").unwrap());
        assert_eq!(Unit::Mega, Unit::from_str("M").unwrap());
        assert_eq!(Unit::Giga, Unit::from_str("Giga").unwrap());
        assert_eq!(Unit::Giga, Unit::from_str("giga").unwrap());
        assert_eq!(Unit::Giga, Unit::from_str("G").unwrap());
        assert_eq!(Unit::Tera, Unit::from_str("Tera").unwrap());
        assert_eq!(Unit::Tera, Unit::from_str("tera").unwrap());
        assert_eq!(Unit::Tera, Unit::from_str("T").unwrap());
    }

    #[test]
    fn parse_parameters() {
        let input_ri = (0.958, -0.263);
        let input_db = (55.620, -11.574);
        let input_ma = (0.9997, -4.5700);

        let exemplar_ri = c64::new(0.958, -0.263);
        let exemplar_db = c64::new(591.668176509173182421586464390458, -121.172256857407207350291917811354);
        let exemplar_ma = c64::new(0.996521687656456974473555257676358, -0.0796530980585618737370302058833604);

        comp_point_c64(&exemplar_ri, &RFDataFormat::RI.parse(input_ri.0, input_ri.1), "RI.parse()", String::from(""));
        comp_point_c64(&exemplar_ri, &RFDataFormat::RI.parse_tuple(input_ri), "RI.parse_tuple()", String::from(""));
        comp_point_c64(&exemplar_db, &RFDataFormat::DB.parse(input_db.0, input_db.1), "DB.parse()", String::from(""));
        comp_point_c64(&exemplar_db, &RFDataFormat::DB.parse_tuple(input_db), "DB.parse_tuple()", String::from(""));
        comp_point_c64(&exemplar_ma, &RFDataFormat::MA.parse(input_ma.0, input_ma.1), "MA.parse()", String::from(""));
        comp_point_c64(&exemplar_ma, &RFDataFormat::MA.parse_tuple(input_ma), "MA.parse_tuple()", String::from(""));
    }
}