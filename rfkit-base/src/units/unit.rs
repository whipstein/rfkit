use serde::Serialize;
use std::{fmt, str::FromStr};

/// Descriptor of sweep type
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Sweep {
    #[default]
    Linear,
    Log,
}

/// Descriptor of unit
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Unit {
    #[default]
    None, // No Unit
    Hz,     // Frequency in Hz
    Degree, // Number in degree
    Radian, // Number in radians
    Lambda, // Length in wavelengths
    Second, // Length in seconds
    Meter,  // Length in meters
    Inch,   // Length in inches
    Farad,  // Capacitance in farads
    Henry,  // Inductance in henries
    Ohm,    // Resistance in ohms
    Sieman, // Conductance in siemans
    Neper,  // Legacy unit of nepers
}

impl Unit {
    pub fn to_long_string(&self) -> String {
        match self {
            Unit::None => "".to_string(),
            Unit::Hz => "hertz".to_string(),
            Unit::Degree => "degree".to_string(),
            Unit::Radian => "radian".to_string(),
            Unit::Lambda => "lambda".to_string(),
            Unit::Second => "second".to_string(),
            Unit::Meter => "meter".to_string(),
            Unit::Inch => "inch".to_string(),
            Unit::Farad => "farad".to_string(),
            Unit::Henry => "henry".to_string(),
            Unit::Ohm => "ohm".to_string(),
            Unit::Sieman => "sieman".to_string(),
            Unit::Neper => "neper".to_string(),
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            Unit::None => "",
            Unit::Hz => "Hz",
            Unit::Degree => "°",
            Unit::Radian => "rad",
            Unit::Lambda => "λ",
            Unit::Second => "s",
            Unit::Meter => "m",
            Unit::Inch => "in",
            Unit::Farad => "F",
            Unit::Henry => "H",
            Unit::Ohm => "Ω",
            Unit::Sieman => "S",
            Unit::Neper => "Np",
        }
    }
}

impl FromStr for Unit {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "HZ" | "Hz" | "hz" => Ok(Unit::Hz),
            "Degree" | "degree" | "deg" | "°" => Ok(Unit::Degree),
            "Radian" | "radian" | "rad" => Ok(Unit::Radian),
            "Lambda" | "lambda" | "λ" => Ok(Unit::Lambda),
            "Second" | "second" | "sec" | "s" => Ok(Unit::Second),
            "Meter" | "meter" | "m" => Ok(Unit::Meter),
            "Inch" | "inch" | "in" => Ok(Unit::Inch),
            "Farad" | "farad" | "F" => Ok(Unit::Farad),
            "Henry" | "henry" | "H" => Ok(Unit::Henry),
            "Ohm" | "ohm" | "Ω" => Ok(Unit::Ohm),
            "Sieman" | "sieman" | "S" => Ok(Unit::Sieman),
            "Neper" | "neper" | "Np" => Ok(Unit::Neper),
            _ => Ok(Unit::None),
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_parse_unit() {
        assert_eq!(Unit::None, Unit::from_str("").unwrap());
        assert_eq!(Unit::None, Unit::from_str("x").unwrap());
        assert_eq!(Unit::Hz, Unit::from_str("HZ").unwrap());
        assert_eq!(Unit::Hz, Unit::from_str("Hz").unwrap());
        assert_eq!(Unit::Hz, Unit::from_str("hz").unwrap());
        assert_eq!(Unit::Degree, Unit::from_str("Degree").unwrap());
        assert_eq!(Unit::Degree, Unit::from_str("degree").unwrap());
        assert_eq!(Unit::Degree, Unit::from_str("deg").unwrap());
        assert_eq!(Unit::Degree, Unit::from_str("°").unwrap());
        assert_eq!(Unit::Radian, Unit::from_str("Radian").unwrap());
        assert_eq!(Unit::Radian, Unit::from_str("radian").unwrap());
        assert_eq!(Unit::Radian, Unit::from_str("rad").unwrap());
        assert_eq!(Unit::Lambda, Unit::from_str("Lambda").unwrap());
        assert_eq!(Unit::Lambda, Unit::from_str("lambda").unwrap());
        assert_eq!(Unit::Lambda, Unit::from_str("λ").unwrap());
        assert_eq!(Unit::Second, Unit::from_str("Second").unwrap());
        assert_eq!(Unit::Second, Unit::from_str("second").unwrap());
        assert_eq!(Unit::Second, Unit::from_str("sec").unwrap());
        assert_eq!(Unit::Second, Unit::from_str("s").unwrap());
        assert_eq!(Unit::Meter, Unit::from_str("Meter").unwrap());
        assert_eq!(Unit::Meter, Unit::from_str("meter").unwrap());
        assert_eq!(Unit::Meter, Unit::from_str("m").unwrap());
        assert_eq!(Unit::Inch, Unit::from_str("Inch").unwrap());
        assert_eq!(Unit::Inch, Unit::from_str("inch").unwrap());
        assert_eq!(Unit::Inch, Unit::from_str("in").unwrap());
        assert_eq!(Unit::Farad, Unit::from_str("Farad").unwrap());
        assert_eq!(Unit::Farad, Unit::from_str("farad").unwrap());
        assert_eq!(Unit::Farad, Unit::from_str("F").unwrap());
        assert_eq!(Unit::Henry, Unit::from_str("Henry").unwrap());
        assert_eq!(Unit::Henry, Unit::from_str("henry").unwrap());
        assert_eq!(Unit::Henry, Unit::from_str("H").unwrap());
        assert_eq!(Unit::Ohm, Unit::from_str("Ohm").unwrap());
        assert_eq!(Unit::Ohm, Unit::from_str("ohm").unwrap());
        assert_eq!(Unit::Ohm, Unit::from_str("Ω").unwrap());
        assert_eq!(Unit::Sieman, Unit::from_str("Sieman").unwrap());
        assert_eq!(Unit::Sieman, Unit::from_str("sieman").unwrap());
        assert_eq!(Unit::Sieman, Unit::from_str("S").unwrap());
        assert_eq!(Unit::Neper, Unit::from_str("Neper").unwrap());
        assert_eq!(Unit::Neper, Unit::from_str("neper").unwrap());
        assert_eq!(Unit::Neper, Unit::from_str("Np").unwrap());
    }

    #[test]
    fn test_unit_from_str() {
        let hz = ["HZ", "Hz", "hz"];
        let deg = ["Degree", "degree", "deg", "°"];
        let rad = ["Radian", "radian", "rad"];
        let lambda = ["Lambda", "lambda", "λ"];
        let sec = ["Second", "second", "sec", "s"];
        let meter = ["Meter", "meter", "m"];
        let inch = ["Inch", "inch", "in"];
        let farad = ["Farad", "farad", "F"];
        let henry = ["Henry", "henry", "H"];
        let ohm = ["Ohm", "ohm", "Ω"];
        let sieman = ["Sieman", "sieman", "S"];
        let neper = ["Neper", "neper", "Np"];
        let nada = ["", "google", ".sfwe"];

        for mult in hz.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Hz);
        }

        for mult in deg.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Degree);
        }

        for mult in rad.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Radian);
        }

        for mult in lambda.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Lambda);
        }

        for mult in sec.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Second);
        }

        for mult in meter.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Meter);
        }

        for mult in inch.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Inch);
        }

        for mult in farad.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Farad);
        }

        for mult in henry.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Henry);
        }

        for mult in ohm.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Ohm);
        }

        for mult in sieman.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Sieman);
        }

        for mult in neper.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::Neper);
        }

        for mult in nada.iter() {
            assert_eq!(Unit::from_str(mult).unwrap(), Unit::None);
        }
    }
}
