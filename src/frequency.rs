use faer::{scale, Row};
use std::f64::consts::PI;
use crate::enums::Unit;

// Frequency stores values in Hz
#[derive(Clone, Debug, PartialEq)]
pub struct Frequency {
    pts: Row<f64>,
}

impl Frequency {
    fn f_scaled(&self, unit: Unit) -> Row<f64> {
        &self.pts * scale(1.0 / unit.scale())
    }

    pub fn freq(&self) -> &Row<f64> {
        &self.pts
    }

    pub fn freq_at(&self, pt: usize) -> &f64 {
        &self.pts.get(pt)
    }

    pub fn freq_scaled_at(&self, pt: usize, unit: Unit) -> f64 {
        self.pts.get(pt) / unit.scale()
    }

    pub fn from_row(f: Row<f64>, unit: Unit) -> Frequency {
        Frequency{
            pts: &f * scale(unit.scale()),
        }
    }

    pub fn from_lin_range(start: f64, stop: f64, step: f64, unit: Unit) -> Frequency {
        let n = ((stop - start) / step) as usize + 1;

        Frequency {
            pts: Row::<f64>::from_fn(n, |i| (start + (i as f64)*step) * unit.scale()),
        }
    }

    pub fn from_vec(f: Vec<f64>, unit: Unit) -> Frequency {
        let mut freq: Row<f64> = Row::zeros(f.len());
        for i in 0..freq.ncols() {
            freq.write(i, f[i] * unit.scale());
        }
        Frequency{
            pts: freq,
        }
    }

    pub fn get_ghz(&self, pt: usize) -> f64 {
        self.freq_at(pt) / Unit::Giga.scale()
    }

    pub fn get_hz(&self, pt: usize) -> f64 {
        *self.freq_at(pt)
    }

    pub fn get_khz(&self, pt: usize) -> f64 {
        self.freq_at(pt) / Unit::Kilo.scale()
    }

    pub fn get_mhz(&self, pt: usize) -> f64 {
        self.freq_at(pt) / Unit::Mega.scale()
    }

    pub fn get_thz(&self, pt: usize) -> f64 {
        self.freq_at(pt) / Unit::Tera.scale()
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

    pub fn new(pts: Row<f64>) -> Frequency {
        Frequency {
            pts: pts,
        }
    }

    pub fn new_scaled(pts: Row<f64>, unit: Unit) -> Frequency {
        Frequency {
            pts: pts * scale(unit.scale()),
        }
    }

    pub fn npts(&self) -> usize {
        self.pts.ncols()
    }

    pub fn thz(&self) -> Row<f64> {
        self.f_scaled(Unit::Tera)
    }

    pub fn w(&self) -> Row<f64> {
        &self.pts * scale(2.0*PI)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use faer::row;
    use crate::math::comp_row_f64;

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
    fn frequency_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let freq_hz = Frequency::from_vec(data.clone(), Unit::Base);
        let freq_khz = Frequency::from_vec(data.clone(), Unit::Kilo);
        let freq_mhz = Frequency::from_vec(data.clone(), Unit::Mega);
        let freq_ghz = Frequency::from_vec(data.clone(), Unit::Giga);
        let freq_thz = Frequency::from_vec(data.clone(), Unit::Tera);

        for i in 0..data.len() {
            assert_eq!(data[i] * Unit::Base.scale(), *freq_hz.pts.get(i));
            assert_eq!(data[i] * Unit::Kilo.scale(), *freq_khz.pts.get(i));
            assert_eq!(data[i] * Unit::Mega.scale(), *freq_mhz.pts.get(i));
            assert_eq!(data[i] * Unit::Giga.scale(), *freq_ghz.pts.get(i));
            assert_eq!(data[i] * Unit::Tera.scale(), *freq_thz.pts.get(i));
        }

        assert_eq!(data.len(), freq_hz.npts());
        assert_eq!(data.len(), freq_khz.npts());
        assert_eq!(data.len(), freq_mhz.npts());
        assert_eq!(data.len(), freq_ghz.npts());
        assert_eq!(data.len(), freq_thz.npts());
    }

    #[test]
    fn frequency_freq_scaled() {
        let data_hz = row![1.0e9, 2.0e9, 3.0e9];
        let data_khz = row![1.0e6, 2.0e6, 3.0e6];
        let data_mhz = row![1.0e3, 2.0e3, 3.0e3];
        let data_ghz = row![1.0, 2.0, 3.0];
        let data_thz = row![1.0e-3, 2.0e-3, 3.0e-3];

        let freq_hz = Frequency::from_row(data_hz.clone(), Unit::Base);
        let freq_khz = Frequency::from_row(data_khz.clone(), Unit::Kilo);
        let freq_mhz = Frequency::from_row(data_mhz.clone(), Unit::Mega);
        let freq_ghz = Frequency::from_row(data_ghz.clone(), Unit::Giga);
        let freq_thz = Frequency::from_row(data_thz.clone(), Unit::Tera);

        //TODO Fix Comparison Logic
        comp_row_f64(&data_hz, &freq_hz.hz(), "freq_scaled(hz)->hz");
        comp_row_f64(&data_hz, &freq_khz.hz(), "freq_scaled(khz)->hz");
        comp_row_f64(&data_hz, &freq_mhz.hz(), "freq_scaled(mhz)->hz");
        comp_row_f64(&data_hz, &freq_ghz.hz(), "freq_scaled(ghz)->hz");
        comp_row_f64(&data_hz, &freq_thz.hz(), "freq_scaled(thz)->hz");

        comp_row_f64(&data_khz, &freq_hz.khz(), "freq_scaled(hz)->khz");
        comp_row_f64(&data_khz, &freq_khz.khz(), "freq_scaled(khz)->khz");
        comp_row_f64(&data_khz, &freq_mhz.khz(), "freq_scaled(mhz)->khz");
        comp_row_f64(&data_khz, &freq_ghz.khz(), "freq_scaled(ghz)->khz");
        comp_row_f64(&data_khz, &freq_thz.khz(), "freq_scaled(thz)->khz");

        comp_row_f64(&data_mhz, &freq_hz.mhz(), "freq_scaled(hz)->mhz");
        comp_row_f64(&data_mhz, &freq_khz.mhz(), "freq_scaled(khz)->mhz");
        comp_row_f64(&data_mhz, &freq_mhz.mhz(), "freq_scaled(mhz)->mhz");
        comp_row_f64(&data_mhz, &freq_ghz.mhz(), "freq_scaled(ghz)->mhz");
        comp_row_f64(&data_mhz, &freq_thz.mhz(), "freq_scaled(thz)->mhz");

        comp_row_f64(&data_ghz, &freq_hz.ghz(), "freq_scaled(hz)->ghz");
        comp_row_f64(&data_ghz, &freq_khz.ghz(), "freq_scaled(khz)->ghz");
        comp_row_f64(&data_ghz, &freq_mhz.ghz(), "freq_scaled(mhz)->ghz");
        comp_row_f64(&data_ghz, &freq_ghz.ghz(), "freq_scaled(ghz)->ghz");
        comp_row_f64(&data_ghz, &freq_thz.ghz(), "freq_scaled(thz)->ghz");

        comp_row_f64(&data_thz, &freq_hz.thz(), "freq_scaled(hz)->thz");
        comp_row_f64(&data_thz, &freq_khz.thz(), "freq_scaled(khz)->thz");
        comp_row_f64(&data_thz, &freq_mhz.thz(), "freq_scaled(mhz)->thz");
        comp_row_f64(&data_thz, &freq_ghz.thz(), "freq_scaled(ghz)->thz");
        comp_row_f64(&data_thz, &freq_thz.thz(), "freq_scaled(thz)->thz");
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
}
