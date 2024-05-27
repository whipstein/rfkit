use faer::{scale, Row};
use std::f64::consts::PI;
use crate::enums::Unit;

#[derive(Clone, Debug, PartialEq)]
pub struct Frequency {
  pts: Row<f64>,
  unit: Unit,
}

impl Frequency {
  fn f_scaled(&self, unit: Unit) -> Row<f64> {
      scale(unit.scale() / self.unit.scale()) * &self.pts
  }

  pub fn freq(&self) -> &Row<f64> {
      &self.pts
  }

  pub fn freq_at(&self, pt: usize) -> &f64 {
      &self.pts.get(pt)
  }

  pub fn from_row(f: Row<f64>, unit: Unit) -> Frequency {
      Frequency{
        pts: scale(unit.scale()) * &f,
        unit: unit
      }
  }

  pub fn from_lin_range(start: f64, stop: f64, step: f64, unit: Unit) -> Frequency {
      let n = ((stop - start) / step) as usize + 1;

      Frequency {
          pts: Row::<f64>::from_fn(n, |i| (start + (i as f64)*step)*unit.scale()),
          unit: unit,
      }
  }

  pub fn get_ghz(&self, pt: usize) -> f64 {
    self.freq_at(pt) * (Unit::Giga.scale() / self.unit.scale())
  }

  pub fn get_hz(&self, pt: usize) -> f64 {
    self.freq_at(pt) * (Unit::Base.scale() / self.unit.scale())
  }

  pub fn get_khz(&self, pt: usize) -> f64 {
    (Unit::Kilo.scale() / self.unit.scale()) * self.freq_at(pt)
  }

  pub fn get_mhz(&self, pt: usize) -> f64 {
    self.freq_at(pt) * (Unit::Mega.scale() / self.unit.scale())
  }

  pub fn get_thz(&self, pt: usize) -> f64 {
    self.freq_at(pt) * (Unit::Tera.scale() / self.unit.scale())
  }

  pub fn ghz(&self) -> Row<f64> {
      self.f_scaled(Unit::Giga)
  }

  pub fn hz(&self) -> Row<f64> {
      self.f_scaled(Unit::Base)
  }

  pub fn idx_at(&self, freq: f64, unit: Unit)  -> Option<usize> {
      self.pts.as_slice().iter().position(|x| *x == freq * unit.scale())
  }

  pub fn khz(&self) -> Row<f64> {
      self.f_scaled(Unit::Kilo)
  }

  pub fn mhz(&self) -> Row<f64> {
      self.f_scaled(Unit::Mega)
  }

  pub fn new(pts: Row<f64>, unit: Unit) -> Frequency {
    Frequency {
      pts: pts,
      unit: unit,
    }
  }

  pub fn npts(&self) -> usize {
      self.pts.ncols()
  }

  pub fn thz(&self) -> Row<f64> {
      self.f_scaled(Unit::Tera)
  }

  pub fn w(&self) -> Row<f64> {
      scale(2.0*PI) * &self.pts
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use faer::row;

  #[test]
  fn frequency_equal() {
      let freq = Frequency::from_row(row![1.0, 2.0, 3.0], Unit::Giga);
      let freq_eq = Frequency::from_row(row![1.0, 2.0, 3.0], Unit::Giga);
      let freq_ne = Frequency::from_row(row![1.0, 2.0, 3.0, 4.0], Unit::Giga);
      let freq_ne2 = Frequency::from_row(row![1.0, 3.0, 4.0], Unit::Giga);

      assert_eq!(freq, freq_eq);
      assert_ne!(freq, freq_ne);
      assert_ne!(freq, freq_ne2);
  }

  #[test]
  fn frequency_from_row() {
      let data = row![1.0, 2.0, 3.0];
      let freq_hz = Frequency::from_row(data.clone(), Unit::Base);
      let freq_khz = Frequency::from_row(data.clone(), Unit::Kilo);
      let freq_mhz = Frequency::from_row(data.clone(), Unit::Mega);
      let freq_ghz = Frequency::from_row(data.clone(), Unit::Giga);
      let freq_thz = Frequency::from_row(data.clone(), Unit::Tera);

      for i in 0..data.ncols() {
          assert_eq!(data.get(i) * Unit::Base.scale(), *freq_hz.pts.get(i));
          assert_eq!(data.get(i) * Unit::Kilo.scale(), *freq_khz.pts.get(i));
          assert_eq!(data.get(i) * Unit::Mega.scale(), *freq_mhz.pts.get(i));
          assert_eq!(data.get(i) * Unit::Giga.scale(), *freq_ghz.pts.get(i));
          assert_eq!(data.get(i) * Unit::Tera.scale(), *freq_thz.pts.get(i));
      }

      assert_eq!(data.ncols(), freq_hz.npts());
      assert_eq!(data.ncols(), freq_khz.npts());
      assert_eq!(data.ncols(), freq_mhz.npts());
      assert_eq!(data.ncols(), freq_ghz.npts());
      assert_eq!(data.ncols(), freq_thz.npts());
  }

  #[test]
  fn frequency_from_lin_range() {
      let data = row![1.0, 2.0, 3.0];
      let freq_hz = Frequency::from_lin_range(1.0, 3.0, 1.0, Unit::Base);
      let freq_khz = Frequency::from_lin_range(1.0, 3.0, 1.0, Unit::Kilo);
      let freq_mhz = Frequency::from_lin_range(1.0, 3.0, 1.0, Unit::Mega);
      let freq_ghz = Frequency::from_lin_range(1.0, 3.0, 1.0, Unit::Giga);
      let freq_thz = Frequency::from_lin_range(1.0, 3.0, 1.0, Unit::Tera);

      for i in 0..data.ncols() {
          assert_eq!(data.get(i) * Unit::Base.scale(), *freq_hz.pts.get(i));
          assert_eq!(data.get(i) * Unit::Kilo.scale(), *freq_khz.pts.get(i));
          assert_eq!(data.get(i) * Unit::Mega.scale(), *freq_mhz.pts.get(i));
          assert_eq!(data.get(i) * Unit::Giga.scale(), *freq_ghz.pts.get(i));
          assert_eq!(data.get(i) * Unit::Tera.scale(), *freq_thz.pts.get(i));
      }

      assert_eq!(data.ncols(), freq_hz.npts());
      assert_eq!(data.ncols(), freq_khz.npts());
      assert_eq!(data.ncols(), freq_mhz.npts());
      assert_eq!(data.ncols(), freq_ghz.npts());
      assert_eq!(data.ncols(), freq_thz.npts());
  }

  #[test]
  fn frequency_freq_scaled() {
      let data = row![1.0, 2.0, 3.0];
      let freq_hz = Frequency::from_row(data.clone(), Unit::Base);
      let freq_khz = Frequency::from_row(data.clone(), Unit::Kilo);
      let freq_mhz = Frequency::from_row(data.clone(), Unit::Mega);
      let freq_ghz = Frequency::from_row(data.clone(), Unit::Giga);
      let freq_thz = Frequency::from_row(data.clone(), Unit::Tera);

      calc_err(scale(Unit::Base.scale()) * &data, freq_hz.hz());
      calc_err(scale(Unit::Base.scale()) * &data, freq_khz.hz());
      calc_err(scale(Unit::Base.scale()) * &data, freq_mhz.hz());
      calc_err(scale(Unit::Base.scale()) * &data, freq_ghz.hz());
      calc_err(scale(Unit::Base.scale()) * &data, freq_thz.hz());

      calc_err(scale(Unit::Kilo.scale()) * &data, freq_hz.khz());
      calc_err(scale(Unit::Kilo.scale()) * &data, freq_khz.khz());
      calc_err(scale(Unit::Kilo.scale()) * &data, freq_mhz.khz());
      calc_err(scale(Unit::Kilo.scale()) * &data, freq_ghz.khz());
      calc_err(scale(Unit::Kilo.scale()) * &data, freq_thz.khz());

      calc_err(scale(Unit::Mega.scale()) * &data, freq_hz.mhz());
      calc_err(scale(Unit::Mega.scale()) * &data, freq_khz.mhz());
      calc_err(scale(Unit::Mega.scale()) * &data, freq_mhz.mhz());
      calc_err(scale(Unit::Mega.scale()) * &data, freq_ghz.mhz());
      calc_err(scale(Unit::Mega.scale()) * &data, freq_thz.mhz());

      calc_err(scale(Unit::Giga.scale()) * &data, freq_hz.ghz());
      calc_err(scale(Unit::Giga.scale()) * &data, freq_khz.ghz());
      calc_err(scale(Unit::Giga.scale()) * &data, freq_mhz.ghz());
      calc_err(scale(Unit::Giga.scale()) * &data, freq_ghz.ghz());
      calc_err(scale(Unit::Giga.scale()) * &data, freq_thz.ghz());

      calc_err(scale(Unit::Tera.scale()) * &data, freq_hz.thz());
      calc_err(scale(Unit::Tera.scale()) * &data, freq_khz.thz());
      calc_err(scale(Unit::Tera.scale()) * &data, freq_mhz.thz());
      calc_err(scale(Unit::Tera.scale()) * &data, freq_ghz.thz());
      calc_err(scale(Unit::Tera.scale()) * &data, freq_thz.thz());
    }

  #[test]
  fn frequency_idx_at() {
      let data = row![1.0, 2.0, 3.0];
      let freq = Frequency::from_row(data.clone(), Unit::Giga);

      assert_eq!(0, freq.idx_at(1.0, Unit::Giga).unwrap());
      assert_eq!(0, freq.idx_at(1.0e9, Unit::Base).unwrap());
      assert_eq!(2, freq.idx_at(3.0, Unit::Giga).unwrap());
      assert_eq!(None, freq.idx_at(1.5, Unit::Giga));
  }

  #[test]
  fn frequency_w() {
      let data = row![1.0, 2.0, 3.0];
      let freq = Frequency::from_row(data.clone(), Unit::Giga);
      let w = freq.w();

      for i in 0..w.nrows() {
          assert_eq!(data.get(i) * 2.0 * PI * Unit::Giga.scale(), *w.get(i));
      }
  }

  fn calc_err(a: Row<f64>, b: Row<f64>) {
    let base: f64 = 2.0;
    let eps = base.powi(-53);

    for i in 0..a.nrows() {
      let val = (a.get(i) - b.get(i)).abs() / (eps * a.get(i));
      assert!(val * eps.sqrt() < 1.0);
    }
  }
}
