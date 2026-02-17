use crate::num::RealScalar;
use ndarray::Array1;
use serde::Serialize;
use std::{fmt, str::FromStr};

/// Descriptor of scaling
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Scale {
    Atto,
    Femto,
    Pico,
    Nano,
    Micro,
    Milli,
    Centi,
    #[default]
    Base,
    Kilo,
    Mega,
    Giga,
    Tera,
}

impl Scale {
    pub fn to_long_string(&self) -> String {
        match self {
            Scale::Tera => "tera".to_string(),
            Scale::Giga => "giga".to_string(),
            Scale::Mega => "mega".to_string(),
            Scale::Kilo => "kilo".to_string(),
            Scale::Base => "".to_string(),
            Scale::Centi => "centi".to_string(),
            Scale::Milli => "milli".to_string(),
            Scale::Micro => "micro".to_string(),
            Scale::Nano => "nano".to_string(),
            Scale::Pico => "pico".to_string(),
            Scale::Femto => "femto".to_string(),
            Scale::Atto => "atto".to_string(),
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            Scale::Atto => "a",
            Scale::Femto => "f",
            Scale::Pico => "p",
            Scale::Nano => "n",
            Scale::Micro => "u",
            Scale::Milli => "m",
            Scale::Centi => "c",
            Scale::Base => "",
            Scale::Kilo => "k",
            Scale::Mega => "M",
            Scale::Giga => "G",
            Scale::Tera => "T",
        }
    }

    /// Provides multiplier for scale
    /// Scale::Pico = 1e-12
    pub fn multiplier<T>(&self) -> T
    where
        T: RealScalar,
    {
        match self {
            Scale::Atto => 1e-18.into(),
            Scale::Femto => 1e-15.into(),
            Scale::Pico => 1e-12.into(),
            Scale::Nano => 1e-9.into(),
            Scale::Micro => 1e-6.into(),
            Scale::Milli => 1e-3.into(),
            Scale::Centi => 1e-2.into(),
            Scale::Base => 1.0.into(),
            Scale::Kilo => 1e3.into(),
            Scale::Mega => 1e6.into(),
            Scale::Giga => 1e9.into(),
            Scale::Tera => 1e12.into(),
        }
    }

    pub fn scale<T: RealScalar>(&self, val: T) -> T {
        val / self.multiplier()
    }

    pub fn scale_array<T: RealScalar>(&self, val: &Array1<T>) -> Array1<T> {
        val.map(|&x| x / self.multiplier())
    }

    pub fn unscale<T: RealScalar>(&self, val: T) -> T {
        val * self.multiplier()
    }

    pub fn unscale_array<T: RealScalar>(&self, val: &Array1<T>) -> Array1<T> {
        val.map(|&x| x * self.multiplier())
    }
}

impl FromStr for Scale {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Atto" | "atto" | "a" | "aF" | "aH" => Ok(Scale::Atto),
            "Femto" | "femto" | "f" | "fF" | "fH" => Ok(Scale::Femto),
            "Pico" | "pico" | "p" | "pF" | "pH" => Ok(Scale::Pico),
            "Nano" | "nano" | "n" | "nF" | "nH" => Ok(Scale::Nano),
            "Micro" | "micro" | "u" | "uΩ" | "μΩ" | "uF" | "μF" | "uH" | "μH" => {
                Ok(Scale::Micro)
            }
            "Milli" | "milli" | "Mil" | "mil" | "m" | "mΩ" | "mF" | "mH" => Ok(Scale::Milli),
            "Centi" | "centi" | "c" | "cΩ" | "cF" | "cH" => Ok(Scale::Centi),
            "Kilo" | "kilo" | "k" | "kΩ" | "kHz" | "khz" => Ok(Scale::Kilo),
            "Mega" | "mega" | "M" | "MΩ" | "MHz" | "mhz" => Ok(Scale::Mega),
            "Giga" | "giga" | "G" | "GΩ" | "GHz" | "ghz" => Ok(Scale::Giga),
            "Tera" | "tera" | "T" | "THz" | "thz" => Ok(Scale::Tera),
            _ => Ok(Scale::Base),
        }
    }
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[cfg(test)]
mod scale_tests {
    use super::*;
    use crate::util::{ApproxEq, NumMargin};

    #[test]
    fn test_parse_scale() {
        assert_eq!(Scale::Base, Scale::from_str("").unwrap());
        assert_eq!(Scale::Base, Scale::from_str("x").unwrap());
        assert_eq!(Scale::Atto, Scale::from_str("atto").unwrap());
        assert_eq!(Scale::Atto, Scale::from_str("Atto").unwrap());
        assert_eq!(Scale::Atto, Scale::from_str("a").unwrap());
        assert_eq!(Scale::Femto, Scale::from_str("Femto").unwrap());
        assert_eq!(Scale::Femto, Scale::from_str("femto").unwrap());
        assert_eq!(Scale::Femto, Scale::from_str("f").unwrap());
        assert_eq!(Scale::Pico, Scale::from_str("Pico").unwrap());
        assert_eq!(Scale::Pico, Scale::from_str("pico").unwrap());
        assert_eq!(Scale::Pico, Scale::from_str("p").unwrap());
        assert_eq!(Scale::Nano, Scale::from_str("Nano").unwrap());
        assert_eq!(Scale::Nano, Scale::from_str("nano").unwrap());
        assert_eq!(Scale::Nano, Scale::from_str("n").unwrap());
        assert_eq!(Scale::Micro, Scale::from_str("Micro").unwrap());
        assert_eq!(Scale::Micro, Scale::from_str("micro").unwrap());
        assert_eq!(Scale::Micro, Scale::from_str("u").unwrap());
        assert_eq!(Scale::Milli, Scale::from_str("Milli").unwrap());
        assert_eq!(Scale::Milli, Scale::from_str("milli").unwrap());
        assert_eq!(Scale::Milli, Scale::from_str("Mil").unwrap());
        assert_eq!(Scale::Milli, Scale::from_str("mil").unwrap());
        assert_eq!(Scale::Milli, Scale::from_str("m").unwrap());
        assert_eq!(Scale::Centi, Scale::from_str("Centi").unwrap());
        assert_eq!(Scale::Centi, Scale::from_str("centi").unwrap());
        assert_eq!(Scale::Centi, Scale::from_str("c").unwrap());
        assert_eq!(Scale::Kilo, Scale::from_str("Kilo").unwrap());
        assert_eq!(Scale::Kilo, Scale::from_str("kilo").unwrap());
        assert_eq!(Scale::Kilo, Scale::from_str("k").unwrap());
        assert_eq!(Scale::Mega, Scale::from_str("Mega").unwrap());
        assert_eq!(Scale::Mega, Scale::from_str("mega").unwrap());
        assert_eq!(Scale::Mega, Scale::from_str("M").unwrap());
        assert_eq!(Scale::Giga, Scale::from_str("Giga").unwrap());
        assert_eq!(Scale::Giga, Scale::from_str("giga").unwrap());
        assert_eq!(Scale::Giga, Scale::from_str("G").unwrap());
        assert_eq!(Scale::Tera, Scale::from_str("Tera").unwrap());
        assert_eq!(Scale::Tera, Scale::from_str("tera").unwrap());
        assert_eq!(Scale::Tera, Scale::from_str("T").unwrap());
    }

    #[test]
    fn test_scale_from_str() {
        let tera = ["Tera", "tera", "T", "THz", "thz"];
        let giga = ["Giga", "giga", "G", "GΩ", "GHz", "ghz"];
        let mega = ["Mega", "mega", "M", "MΩ", "MHz", "mhz"];
        let kilo = ["Kilo", "kilo", "k", "kΩ", "kHz", "khz"];
        let centi = ["Centi", "centi", "c", "cΩ", "cF", "cH"];
        let milli = ["Milli", "milli", "Mil", "mil", "m", "mΩ", "mF", "mH"];
        let micro = ["Micro", "micro", "u", "uΩ", "μΩ", "uF", "μF", "uH", "μH"];
        let nano = ["Nano", "nano", "n", "nF", "nH"];
        let pico = ["Pico", "pico", "p", "pF", "pH"];
        let femto = ["Femto", "femto", "f", "fF", "fH"];
        let atto = ["Atto", "atto", "a", "aF", "aH"];
        let nada = ["", "google", ".sfwe"];

        for mult in tera.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Tera);
        }

        for mult in giga.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Giga);
        }

        for mult in mega.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Mega);
        }

        for mult in kilo.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Kilo);
        }

        for mult in centi.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Centi);
        }

        for mult in milli.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Milli);
        }

        for mult in micro.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Micro);
        }

        for mult in nano.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Nano);
        }

        for mult in pico.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Pico);
        }

        for mult in femto.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Femto);
        }

        for mult in atto.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Atto);
        }

        for mult in nada.iter() {
            assert_eq!(Scale::from_str(mult).unwrap(), Scale::Base);
        }
    }

    #[test]
    fn test_scale_unscale() {
        let val: f64 = 3.24;

        Scale::Tera.scale(val).assert_approx_eq(
            &(val * 1e-12),
            NumMargin::default(),
            "scale()",
            "Scale::Tera",
        );
        Scale::Tera.unscale(val).assert_approx_eq(
            &(val * 1e12),
            NumMargin::default(),
            "unscale()",
            "Scale::Tera",
        );

        Scale::Giga.scale(val).assert_approx_eq(
            &(val * 1e-9),
            NumMargin::default(),
            "scale()",
            "Scale::Giga",
        );
        Scale::Giga.unscale(val).assert_approx_eq(
            &(val * 1e9),
            NumMargin::default(),
            "unscale()",
            "Scale::Giga",
        );

        Scale::Mega.scale(val).assert_approx_eq(
            &(val * 1e-6),
            NumMargin::default(),
            "scale()",
            "Scale::Mega",
        );
        Scale::Mega.unscale(val).assert_approx_eq(
            &(val * 1e6),
            NumMargin::default(),
            "unscale()",
            "Scale::Mega",
        );

        Scale::Kilo.scale(val).assert_approx_eq(
            &(val * 1e-3),
            NumMargin::default(),
            "scale()",
            "Scale::Kilo",
        );
        Scale::Kilo.unscale(val).assert_approx_eq(
            &(val * 1e3),
            NumMargin::default(),
            "unscale()",
            "Scale::Kilo",
        );

        Scale::Centi.scale(val).assert_approx_eq(
            &(val * 1e2),
            NumMargin::default(),
            "scale()",
            "Scale::Centi",
        );
        Scale::Centi.unscale(val).assert_approx_eq(
            &(val * 1e-2),
            NumMargin::default(),
            "unscale()",
            "Scale::Centi",
        );

        Scale::Milli.scale(val).assert_approx_eq(
            &(val * 1e3),
            NumMargin::default(),
            "scale()",
            "Scale::Milli",
        );
        Scale::Milli.unscale(val).assert_approx_eq(
            &(val * 1e-3),
            NumMargin::default(),
            "unscale()",
            "Scale::Milli",
        );

        Scale::Micro.scale(val).assert_approx_eq(
            &(val * 1e6),
            NumMargin::default(),
            "scale()",
            "Scale::Micro",
        );
        Scale::Micro.unscale(val).assert_approx_eq(
            &(val * 1e-6),
            NumMargin::default(),
            "unscale()",
            "Scale::Micro",
        );

        Scale::Nano.scale(val).assert_approx_eq(
            &(val * 1e9),
            NumMargin::default(),
            "scale()",
            "Scale::Nano",
        );
        Scale::Nano.unscale(val).assert_approx_eq(
            &(val * 1e-9),
            NumMargin::default(),
            "unscale()",
            "Scale::Nano",
        );

        Scale::Pico.scale(val).assert_approx_eq(
            &(val * 1e12),
            NumMargin::default(),
            "scale()",
            "Scale::Pico",
        );
        Scale::Pico.unscale(val).assert_approx_eq(
            &(val * 1e-12),
            NumMargin::default(),
            "unscale()",
            "Scale::Pico",
        );

        Scale::Femto.scale(val).assert_approx_eq(
            &(val * 1e15),
            NumMargin::default(),
            "scale()",
            "Scale::Femto",
        );
        Scale::Femto.unscale(val).assert_approx_eq(
            &(val * 1e-15),
            NumMargin::default(),
            "unscale()",
            "Scale::Femto",
        );

        Scale::Atto.scale(val).assert_approx_eq(
            &(val * 1e18),
            NumMargin::default(),
            "scale()",
            "Scale::Atto",
        );
        Scale::Atto.unscale(val).assert_approx_eq(
            &(val * 1e-18),
            NumMargin::default(),
            "unscale()",
            "Scale::Atto",
        );

        Scale::Base.scale(val).assert_approx_eq(
            &(val * 1.0),
            NumMargin::default(),
            "scale()",
            "Scale::Base",
        );
        Scale::Base.unscale(val).assert_approx_eq(
            &(val * 1.0),
            NumMargin::default(),
            "unscale()",
            "Scale::Base",
        );
    }
}
