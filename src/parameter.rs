#![allow(dead_code)]
use core::fmt;
use serde::Serialize;
use simple_error::{SimpleError, bail};
use std::str::FromStr;

/// Descriptor of RF parameter for an n-port network.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Eq, Hash)]
pub enum RFParameter {
    A,
    G,
    H,
    S,
    SPower,
    SPseudo,
    STraveling,
    T,
    Y,
    Z,
}

impl RFParameter {
    // Convert text from a String into RFParameter type
    pub fn from_string(val: String) -> Result<RFParameter, SimpleError> {
        match val.to_lowercase().as_str() {
            "a" | "abcd" => Ok(RFParameter::A),
            "g" => Ok(RFParameter::G),
            "h" => Ok(RFParameter::H),
            "s" => Ok(RFParameter::S),
            "spower" | "s_power" => Ok(RFParameter::SPower),
            "spseudo" | "s_pseudo" => Ok(RFParameter::SPseudo),
            "straveling" | "s_traveling" => Ok(RFParameter::STraveling),
            "t" => Ok(RFParameter::T),
            "y" => Ok(RFParameter::Y),
            "z" => Ok(RFParameter::Z),
            _ => bail!("string not a valid option line type"),
        }
    }

    // Convert RFParameter to &str
    pub fn to_str(&self) -> &str {
        match self {
            RFParameter::A => "ABCD",
            RFParameter::G => "G",
            RFParameter::H => "H",
            RFParameter::S => "S",
            RFParameter::SPower => "SPower",
            RFParameter::SPseudo => "SPseudo",
            RFParameter::STraveling => "STraveling",
            RFParameter::T => "T",
            RFParameter::Y => "Y",
            RFParameter::Z => "Z",
        }
    }

    // // Convert RFParameter to a String
    // pub fn to_string(&self) -> String {
    //     match self {
    //         RFParameter::A => "ABCD".to_string(),
    //         RFParameter::G => "G".to_string(),
    //         RFParameter::H => "H".to_string(),
    //         RFParameter::S => "S".to_string(),
    //         RFParameter::T => "T".to_string(),
    //         RFParameter::Y => "Y".to_string(),
    //         RFParameter::Z => "Z".to_string(),
    //     }
    // }
}

impl FromStr for RFParameter {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "a" | "abcd" => Ok(RFParameter::A),
            "g" => Ok(RFParameter::G),
            "h" => Ok(RFParameter::H),
            "s" => Ok(RFParameter::S),
            "spower" | "s_power" => Ok(RFParameter::SPower),
            "spseudo" | "s_pseudo" => Ok(RFParameter::SPseudo),
            "straveling" | "s_traveling" => Ok(RFParameter::STraveling),
            "t" => Ok(RFParameter::T),
            "y" => Ok(RFParameter::Y),
            "z" => Ok(RFParameter::Z),
            _ => bail!("string not recognized"),
        }
    }
}

impl fmt::Display for RFParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = match self {
            RFParameter::A => "ABCD",
            RFParameter::G => "G",
            RFParameter::H => "H",
            RFParameter::S => "S",
            RFParameter::SPower => "SPower",
            RFParameter::SPseudo => "SPseudo",
            RFParameter::STraveling => "STraveling",
            RFParameter::T => "T",
            RFParameter::Y => "Y",
            RFParameter::Z => "Z",
        };
        write!(f, "{}", val)
        // write!(f, "{}", self.to_str())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rfparameter() {
        let a = ["A", "a", "ABCD"];
        let g = ["G", "g"];
        let h = ["H", "h"];
        let s = ["S", "s"];
        let s_power = ["SPower", "s_power"];
        let s_pseudo = ["SPseudo", "s_pseudo"];
        let s_traveling = ["STraveling", "s_traveling"];
        let t = ["T", "t"];
        let y = ["Y", "y"];
        let z = ["Z", "z"];

        for param in a.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::A);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::A
            );
        }

        for param in g.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::G);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::G
            );
        }

        for param in h.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::H);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::H
            );
        }

        for param in s.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::S);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::S
            );
        }

        for param in s_power.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::SPower);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::SPower
            );
        }

        for param in s_pseudo.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::SPseudo);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::SPseudo
            );
        }

        for param in s_traveling.iter() {
            assert_eq!(
                RFParameter::from_str(param).unwrap(),
                RFParameter::STraveling
            );
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::STraveling
            );
        }

        for param in t.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::T);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::T
            );
        }

        for param in y.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::Y);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::Y
            );
        }

        for param in z.iter() {
            assert_eq!(RFParameter::from_str(param).unwrap(), RFParameter::Z);
            assert_eq!(
                RFParameter::from_string(param.to_string()).unwrap(),
                RFParameter::Z
            );
        }
    }
}
