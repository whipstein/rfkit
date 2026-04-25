use crate::num::RealScalar;
use num_complex::Complex;
use serde::Serialize;
use std::{fmt, str::FromStr};

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize)]
pub enum ComplexNumberType {
    #[default]
    ReIm,
    MagAng,
    Db,
}

impl ComplexNumberType {
    pub fn parse<T: RealScalar>(&self, x: T, y: T) -> Complex<T> {
        match self {
            ComplexNumberType::ReIm => Complex::new(x, y),
            ComplexNumberType::MagAng => Complex::from_polar(x, y.to_radians()),
            ComplexNumberType::Db => {
                Complex::from_polar(T::from_f64(10.0).powf(x / 20.0), y.to_radians())
            }
        }
    }
}

impl FromStr for ComplexNumberType {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ri" | "RI" | "reim" => Ok(ComplexNumberType::ReIm),
            "ma" | "MA" | "magang" => Ok(ComplexNumberType::MagAng),
            "db" | "DB" | "dbang" => Ok(ComplexNumberType::Db),
            _ => Err("ComplexNumberType not recognized".to_string().into()),
        }
    }
}

impl fmt::Display for ComplexNumberType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            ComplexNumberType::ReIm => write!(f, "ri"),
            ComplexNumberType::MagAng => write!(f, "ma"),
            ComplexNumberType::Db => write!(f, "db"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct ComplexNumber<T: RealScalar> {
    kind: ComplexNumberType,
    re: T,
    im: T,
}

impl<T: RealScalar> ComplexNumber<T> {
    pub fn convert(&self, kind: ComplexNumberType) -> ComplexNumber<T> {
        match (self.kind, kind) {
            (ComplexNumberType::ReIm, ComplexNumberType::ReIm) => *self,
            (ComplexNumberType::ReIm, ComplexNumberType::MagAng) => {
                let val = Complex::new(self.re, self.im);
                ComplexNumber {
                    kind: ComplexNumberType::MagAng,
                    re: val.norm(),
                    im: val.arg().to_degrees(),
                }
            }
            (ComplexNumberType::ReIm, ComplexNumberType::Db) => {
                let val = Complex::new(self.re, self.im);
                ComplexNumber {
                    kind: ComplexNumberType::Db,
                    re: val.norm().log10() * 20.0,
                    im: val.arg().to_degrees(),
                }
            }
            (ComplexNumberType::MagAng, ComplexNumberType::ReIm) => {
                let val = Complex::from_polar(self.re, self.im.to_radians());
                ComplexNumber {
                    kind: ComplexNumberType::ReIm,
                    re: val.re,
                    im: val.im,
                }
            }
            (ComplexNumberType::MagAng, ComplexNumberType::MagAng) => *self,
            (ComplexNumberType::MagAng, ComplexNumberType::Db) => ComplexNumber {
                kind: ComplexNumberType::Db,
                re: self.re.log10() * 20.0,
                im: self.im,
            },
            (ComplexNumberType::Db, ComplexNumberType::ReIm) => {
                let val = Complex::from_polar(
                    T::from_f64(10.0).powf(self.re / 20.0),
                    self.im.to_radians(),
                );
                ComplexNumber {
                    kind: ComplexNumberType::ReIm,
                    re: val.re,
                    im: val.im,
                }
            }
            (ComplexNumberType::Db, ComplexNumberType::MagAng) => ComplexNumber {
                kind: ComplexNumberType::MagAng,
                re: T::from_f64(10.0).powf(self.re / 20.0),
                im: self.im,
            },
            (ComplexNumberType::Db, ComplexNumberType::Db) => *self,
        }
    }

    pub fn ri(&self) -> Complex<T> {
        match self.kind {
            ComplexNumberType::ReIm => Complex::new(self.re, self.im),
            ComplexNumberType::MagAng => Complex::from_polar(self.re, self.im.to_radians()),
            ComplexNumberType::Db => {
                Complex::from_polar(T::from_f64(10.0).powf(self.re / 20.0), self.im.to_radians())
            }
        }
    }

    pub fn mag(&self) -> T {
        match self.kind {
            ComplexNumberType::ReIm => Complex::new(self.re, self.im).norm(),
            ComplexNumberType::MagAng => self.re,
            ComplexNumberType::Db => T::from_f64(10.0).powf(self.re / 20.0),
        }
    }

    pub fn db(&self) -> T {
        match self.kind {
            ComplexNumberType::ReIm => Complex::new(self.re, self.im).norm().log10() * 20.0,
            ComplexNumberType::MagAng => self.re.log10() * 20.0,
            ComplexNumberType::Db => self.re,
        }
    }

    pub fn ang(&self) -> T {
        match self.kind {
            ComplexNumberType::ReIm => Complex::new(self.re, self.im).arg().to_degrees(),
            ComplexNumberType::MagAng | ComplexNumberType::Db => self.im,
        }
    }
}

#[derive(Default)]
pub struct ComplexNumberBuilder<T: RealScalar> {
    kind: Option<ComplexNumberType>,
    re: Option<T>,
    im: Option<T>,
}

impl<T: RealScalar> ComplexNumberBuilder<T> {
    pub fn new() -> Self {
        ComplexNumberBuilder::default()
    }

    pub fn kind(mut self, val: ComplexNumberType) -> Self {
        self.kind = Some(val);
        self
    }

    pub fn kind_from_str(mut self, val: &str) -> Self {
        self.kind = Some(ComplexNumberType::from_str(val).unwrap());
        self
    }

    pub fn ri(mut self, val: Complex<T>) -> Self {
        self.re = Some(val.re);
        self.im = Some(val.im);
        self
    }

    pub fn real(mut self, val: T) -> Self {
        self.re = Some(val);
        self
    }

    pub fn imag(mut self, val: T) -> Self {
        self.im = Some(val);
        self
    }

    pub fn mag(mut self, val: T) -> Self {
        self.re = Some(val);
        self
    }

    pub fn db(mut self, val: T) -> Self {
        self.re = Some(val);
        self
    }

    pub fn angle(mut self, val: T) -> Self {
        self.im = Some(val);
        self
    }

    pub fn build(self) -> Result<ComplexNumber<T>, String> {
        let kind = self.kind.ok_or("kind must be set")?;
        let re = self.re.ok_or("re must be set")?;
        let im = self.im.ok_or("im must be set")?;
        Ok(ComplexNumber { kind, re, im })
    }
}

#[cfg(test)]
mod impedance_tests {
    use super::*;
    use crate::util::{ApproxEq, NumMargin};
    use num_complex::c64;

    const MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-3,
        relative: 1e-3,
        ulps: 10,
    };
    const CP: f64 = 2.952781875545368e-14;

    #[test]
    fn test_complexnumber() {
        let val = c64(-0.37838109914277085, -0.18972099398313422);
        let mag = 0.42328041739069366;
        let db = -7.467436464485704;
        let ang = -153.3707274747144;
        let exemplar_ri = ComplexNumber {
            kind: ComplexNumberType::ReIm,
            re: val.re,
            im: val.im,
        };
        let exemplar_magang = ComplexNumber {
            kind: ComplexNumberType::MagAng,
            re: mag,
            im: ang,
        };
        let exemplar_db = ComplexNumber {
            kind: ComplexNumberType::Db,
            re: db,
            im: ang,
        };

        let test = ComplexNumberBuilder::new()
            .ri(val)
            .kind(ComplexNumberType::ReIm)
            .build()
            .unwrap();
        assert_eq!(test.convert(ComplexNumberType::ReIm), exemplar_ri);
        assert_eq!(
            test.convert(ComplexNumberType::MagAng).kind,
            exemplar_magang.kind
        );
        test.convert(ComplexNumberType::MagAng)
            .mag()
            .assert_approx_eq(
                &exemplar_magang.re,
                NumMargin::default(),
                "convert",
                "ri_to_ma",
            );
        test.convert(ComplexNumberType::MagAng)
            .ang()
            .assert_approx_eq(
                &exemplar_magang.im,
                NumMargin::default(),
                "convert",
                "ri_to_ma",
            );
        assert_eq!(test.convert(ComplexNumberType::Db).kind, exemplar_db.kind);
        test.convert(ComplexNumberType::Db).db().assert_approx_eq(
            &exemplar_db.re,
            NumMargin::default(),
            "convert",
            "ri_to_db",
        );
        test.convert(ComplexNumberType::Db).ang().assert_approx_eq(
            &exemplar_db.im,
            NumMargin::default(),
            "convert",
            "ri_to_db",
        );

        let test = ComplexNumberBuilder::new()
            .mag(mag)
            .angle(ang)
            .kind(ComplexNumberType::MagAng)
            .build()
            .unwrap();
        assert_eq!(test.convert(ComplexNumberType::ReIm).kind, exemplar_ri.kind);
        test.convert(ComplexNumberType::ReIm).ri().assert_approx_eq(
            &val,
            NumMargin::default(),
            "convert",
            "ma_to_ri",
        );
        assert_eq!(test.convert(ComplexNumberType::MagAng), exemplar_magang);
        assert_eq!(test.convert(ComplexNumberType::Db).kind, exemplar_db.kind);
        test.convert(ComplexNumberType::Db).re.assert_approx_eq(
            &exemplar_db.re,
            NumMargin::default(),
            "convert",
            "ma_to_db",
        );
        test.convert(ComplexNumberType::Db).im.assert_approx_eq(
            &exemplar_db.im,
            NumMargin::default(),
            "convert",
            "ma_to_db",
        );

        let test = ComplexNumberBuilder::new()
            .db(db)
            .angle(ang)
            .kind(ComplexNumberType::Db)
            .build()
            .unwrap();
        assert_eq!(test.convert(ComplexNumberType::ReIm).kind, exemplar_ri.kind);
        test.convert(ComplexNumberType::ReIm).re.assert_approx_eq(
            &exemplar_ri.re,
            NumMargin::default(),
            "convert",
            "db_to_ri",
        );
        test.convert(ComplexNumberType::ReIm).im.assert_approx_eq(
            &exemplar_ri.im,
            NumMargin::default(),
            "convert",
            "db_to_ri",
        );
        assert_eq!(
            test.convert(ComplexNumberType::MagAng).kind,
            exemplar_magang.kind
        );
        test.convert(ComplexNumberType::MagAng).re.assert_approx_eq(
            &exemplar_magang.re,
            NumMargin::default(),
            "convert",
            "db_to_ma",
        );
        test.convert(ComplexNumberType::MagAng).im.assert_approx_eq(
            &exemplar_magang.im,
            NumMargin::default(),
            "convert",
            "db_to_ma",
        );
        assert_eq!(test.convert(ComplexNumberType::Db), exemplar_db);
    }

    #[test]
    fn test_complexnumberbuilder() {
        let exemplar_ri = c64(-0.37838109914277085, -0.18972099398313422);
        let exemplar_mag = 0.42328041739069366;
        let exemplar_db = -7.467436464485704;
        let exemplar_ang = -153.3707274747144;

        let test = ComplexNumberBuilder::new()
            .kind(ComplexNumberType::ReIm)
            .ri(exemplar_ri)
            .build()
            .unwrap();
        assert_eq!(
            test,
            ComplexNumber {
                kind: ComplexNumberType::ReIm,
                re: exemplar_ri.re,
                im: exemplar_ri.im
            }
        );

        let test = ComplexNumberBuilder::new()
            .kind(ComplexNumberType::MagAng)
            .mag(exemplar_mag)
            .angle(exemplar_ang)
            .build()
            .unwrap();
        assert_eq!(
            test,
            ComplexNumber {
                kind: ComplexNumberType::MagAng,
                re: exemplar_mag,
                im: exemplar_ang
            }
        );

        let test = ComplexNumberBuilder::new()
            .kind(ComplexNumberType::Db)
            .mag(exemplar_db)
            .angle(exemplar_ang)
            .build()
            .unwrap();
        assert_eq!(
            test,
            ComplexNumber {
                kind: ComplexNumberType::Db,
                re: exemplar_db,
                im: exemplar_ang
            }
        );
    }

    #[test]
    fn test_complexnumbertype() {
        let ri = ["ri", "reim"];
        let ma = ["ma", "magang"];
        let db = ["db", "dbang"];
        let nada = ["", "google", ".sfwe"];

        for val in ri.iter() {
            assert_eq!(
                ComplexNumberType::from_str(val).unwrap(),
                ComplexNumberType::ReIm
            );
        }
        for val in ma.iter() {
            assert_eq!(
                ComplexNumberType::from_str(val).unwrap(),
                ComplexNumberType::MagAng
            );
        }
        for val in db.iter() {
            assert_eq!(
                ComplexNumberType::from_str(val).unwrap(),
                ComplexNumberType::Db
            );
        }
        for val in nada.iter() {
            assert_eq!(
                ComplexNumberType::from_str(val).unwrap_err().to_string(),
                "ComplexNumberType not recognized".to_string()
            )
        }
    }
}
