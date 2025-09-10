#![allow(unused)]
use crate::frequency::FrequencyBuilder;
use crate::impedance::ComplexNumberType;
use crate::network::{Network, NetworkBuilder};
use crate::parameter::RFParameter;
use crate::point::{Point, Pt};
use crate::points::{Points, Pts};
use crate::scale::Scale;
use crate::unit::Unit;
use ndarray::prelude::*;
use num::complex::{Complex64, c64};
use regex::{Regex, RegexBuilder};
use std::error::Error;
use std::fs;
use std::str::{FromStr, SplitWhitespace};

macro_rules! unwrap_or_bail {
    ($opt: expr, $msg: expr) => {
        match $opt {
            Some(v) => v,
            None => {
                bail!($msg);
            }
        }
    };
}

macro_rules! unwrap_or_break {
    ($opt: expr) => {
        match $opt {
            Some(v) => v,
            None => {
                break;
            }
        }
    };
}

macro_rules! unwrap_or_continue {
    ($opt: expr) => {
        match $opt {
            Some(v) => v,
            None => {
                continue;
            }
        }
    };
}

macro_rules! unwrap_or_panic {
    ($opt: expr, $msg: expr) => {
        match $opt {
            Some(v) => v,
            None => {
                panic!($msg);
            }
        }
    };
}

use std::ops::{Bound, RangeBounds};

trait StringUtils {
    fn substring(&self, start: usize, len: usize) -> &str;
    fn slice(&self, range: impl RangeBounds<usize>) -> &str;
}

impl StringUtils for str {
    fn substring(&self, start: usize, len: usize) -> &str {
        let mut char_pos = 0;
        let mut byte_start = 0;
        let mut it = self.chars();
        loop {
            if char_pos == start {
                break;
            }
            if let Some(c) = it.next() {
                char_pos += 1;
                byte_start += c.len_utf8();
            } else {
                break;
            }
        }
        char_pos = 0;
        let mut byte_end = byte_start;
        loop {
            if char_pos == len {
                break;
            }
            if let Some(c) = it.next() {
                char_pos += 1;
                byte_end += c.len_utf8();
            } else {
                break;
            }
        }
        &self[byte_start..byte_end]
    }
    fn slice(&self, range: impl RangeBounds<usize>) -> &str {
        let start = match range.start_bound() {
            Bound::Included(bound) | Bound::Excluded(bound) => *bound,
            Bound::Unbounded => 0,
        };
        let len = match range.end_bound() {
            Bound::Included(bound) => *bound + 1,
            Bound::Excluded(bound) => *bound,
            Bound::Unbounded => self.len(),
        } - start;
        self.substring(start, len)
    }
}

fn parse_value(
    pt: &mut Point,
    i: usize,
    j: usize,
    fmt: ComplexNumberType,
    fields: &mut SplitWhitespace,
) {
    pt[(i, j)] = fmt.parse(
        fields.next().unwrap().parse().unwrap(),
        fields.next().unwrap().parse().unwrap(),
    );
}

//TODO cleanup the ability to import various files
pub fn read_touchstone(file_path: &String) -> Result<Network, Box<dyn Error>> {
    // Regex patterns
    let re_file_ext = Regex::new(r"\d+").expect("Invalid regex!");
    let re_file_opts = Regex::new(r"(?i)^#\s+(?<freq>\w?)hz\s+(?<param>g|h|s|y|z)\s+(?<format>db|ma|ri)\s+R\s+(?<impedance>\d+\.?\d*)").expect("Invalid regex!");
    let re_comment =
        Regex::new(r"^!%?\s*%?([\w\s=\-\+\*\/\(\)\.:_\\]+)\s*$").expect("Invalid regex!");
    let re_freq_line = Regex::new(r"(?i)^!%?\s*%?freq").expect("Invalid regex!");
    let re_port_name = RegexBuilder::new(r"^!\s*Port\[(\d+)\]\s*=\s?(\w+)")
        .case_insensitive(true)
        .build()
        .expect("Invalid regex!");

    // file reading variables
    let content = fs::read_to_string(file_path)?;
    let mut file_path_split = file_path.split('.').rev();
    let file_ext = file_path_split.next().unwrap();
    let filename = file_path_split
        .next()
        .unwrap()
        .split('/')
        .next_back()
        .unwrap();
    let mut lines = content.lines();
    let mut data_section = false;

    // Network values
    let nports: usize = re_file_ext
        .find(file_ext)
        .expect("Invalid file extension")
        .as_str()
        .parse()
        .unwrap();
    let mut port_names: Vec<String> = vec![String::from(""); nports];
    let mut freq_unit = Scale::Giga;
    let mut parameter = RFParameter::S;
    let mut format = ComplexNumberType::ReIm;
    let mut impedance = 50.0;
    let mut z0: Array1<Complex64> = Array1::zeros(nports);
    z0.fill(c64(50.0, 0.0));
    let mut freq_tmp: Vec<f64> = vec![];
    let mut point_tmp = Point::zeros((nports, nports));
    let mut data = Points::zeros((0, nports, nports));
    let mut comments = String::new();

    let mut caps: Option<regex::Captures>;
    let mut nfreq_check: Option<usize>;
    let mut nfreq_noise_check: Option<usize>;

    while let Some(line) = lines.next() {
        // let line = unwrap_or_break!(lines.next());

        match line.slice(..1) {
            "!" => {
                //
                // comment line
                //
                if data_section {
                    continue;
                }
                if re_freq_line.captures(line).is_some() {
                    if nports > 2 {
                        for i in 0..(nports - 1) {
                            lines.next();
                        }
                    }
                    continue;
                }
                let Some(val) = re_comment.captures(line) else {
                    continue;
                };
                if !comments.is_empty() {
                    comments += "\n";
                }
                comments += val[1].trim_end();
            }
            "[" => {
                let cap = re_file_opts.captures(line);

                if cap.is_none() {
                    continue;
                }

                let caps = cap.unwrap();

                match caps.get(1).unwrap().as_str() {
                    // TODO: Add additional Version 2.0 options handling
                    "[Number of Frequencies]" => {
                        nfreq_check = Some(caps.get(2).unwrap().as_str().parse::<usize>().unwrap());
                    }
                    "[Number of Noise Frequencies]" => {
                        nfreq_noise_check =
                            Some(caps.get(2).unwrap().as_str().parse::<usize>().unwrap());
                    }
                    "[End]" => break,
                    _ => continue,
                }
            }
            "#" => {
                //
                // File format line
                //
                let Some(vals) = re_file_opts.captures(line) else {
                    panic!("Touchstone option line not valid!")
                };
                freq_unit = Scale::from_str(&vals["freq"])?;
                parameter = RFParameter::from_str(&vals["param"])?;
                format = ComplexNumberType::from_str(&vals["format"]).unwrap();
                impedance = vals["impedance"].parse().unwrap();
                for i in 0..nports {
                    z0[i] = c64(impedance, 0.0);
                }
            }
            _ => {
                //
                // data line
                //
                data_section = true;
                let mut fields = line.split_whitespace();
                let mut val = fields.next();
                if val.is_none() {
                    continue;
                }

                match nports {
                    1 => {
                        freq_tmp.push(val.unwrap().parse().unwrap());

                        parse_value(&mut point_tmp, 0, 0, format, &mut fields);
                    }
                    2 => {
                        freq_tmp.push(val.unwrap().parse().unwrap());

                        parse_value(&mut point_tmp, 0, 0, format, &mut fields);
                        parse_value(&mut point_tmp, 1, 0, format, &mut fields);
                        parse_value(&mut point_tmp, 0, 1, format, &mut fields);
                        parse_value(&mut point_tmp, 1, 1, format, &mut fields);
                    }
                    _ => {
                        for j in 0..nports {
                            if j == 0 {
                                freq_tmp.push(val.unwrap().parse().unwrap());
                            } else {
                                let line = lines.next();
                                if line.is_none() {
                                    break;
                                }
                                fields = line.unwrap().split_whitespace();
                            }
                            for k in 0..nports {
                                parse_value(&mut point_tmp, j, k, format, &mut fields);
                            }
                        }
                    }
                }

                data.push(Axis(0), point_tmp.view());
            }
        }
    }

    Ok(NetworkBuilder::new()
        .freq(
            FrequencyBuilder::new()
                .freqs_scaled(Array1::from_vec(freq_tmp), freq_unit)
                .build(),
        )
        .z0(z0)
        .params(data, parameter)
        .name(filename)
        .comments(comments.to_owned().as_str())
        .build())
}

#[cfg(test)]
mod test {
    use crate::util::{comp_line, comp_points_c64};
    use float_cmp::F64Margin;

    use super::*;

    #[test]
    fn read_touchstone_s1p() {
        let filename = "./data/test.s1p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let exemplar = Network::new(
            FrequencyBuilder::new()
                .freqs_scaled(array![75.0, 75.175, 75.35, 75.525], Scale::Giga)
                .build(),
            array![c64(50.0, 0.0)],
            RFParameter::S,
            Points::new(array![
                [[c64(0.45345337996, 0.891279996524)]],
                [[c64(0.464543921496, 0.885550080459)]],
                [[c64(0.475521092896, 0.879704319764)]],
                [[c64(0.486384743723, 0.873744745949)]],
            ]),
            String::from("test"),
            String::from("Created with mwavepy."),
        );
        debug_assert!(
            exemplar.name() == net.name(),
            " Failed name\n  exemplar: {}\n       net: {}",
            exemplar.name(),
            net.name()
        );
        comp_line(exemplar.comments(), net.comments(), "comments(test.s1p)");
        debug_assert!(
            exemplar.nports() == net.nports(),
            " Failed nports\n  exemplar: {}\n       net: {}",
            exemplar.nports(),
            net.nports()
        );
        debug_assert!(
            exemplar.npts() == net.npts(),
            " Failed npts\n  exemplar: {}\n       net: {}",
            exemplar.npts(),
            net.npts()
        );
        for i in 0..exemplar.nports() {
            debug_assert_eq!(
                exemplar.z0()[i],
                net.z0()[i],
                " Failed z0 for port {}\n  exemplar: {}\n       net: {}",
                i,
                exemplar.z0()[i],
                net.z0()[i]
            );
        }
        for i in 0..exemplar.npts() {
            assert_eq!(exemplar.freq().freq(i), net.freq().freq(i));
        }
        comp_points_c64(
            &exemplar.net(RFParameter::S),
            &net.net(RFParameter::S),
            F64Margin::default(),
            "net(S)",
        );
    }

    #[test]
    fn read_touchstone_s2p() {
        let filename = "./data/test.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let exemplar = Network::new(
            FrequencyBuilder::new()
                .freqs_scaled(array![0.5, 1.0, 1.5, 2.0], Scale::Giga)
                .build(),
            array![c64(50.0, 0.0), c64(50.0, 0.0)],
            RFParameter::S,
            Points::new(array![
                [
                    [
                        c64(0.9881388526914863, -0.13442709904013195),
                        c64(0.0010346705444205045, 0.011178864909012504)
                    ],
                    [
                        c64(-7.363542304219899, 0.6742816969789206),
                        c64(0.5574653418486702, -0.06665134724424635)
                    ],
                ],
                [
                    [
                        c64(0.9578079036840927, -0.2633207328693372),
                        c64(0.0037206104781559784, 0.021909191616475577)
                    ],
                    [
                        c64(-7.130124628011368, 1.3277987152036197),
                        c64(0.5435045929943587, -0.12869941397967788)
                    ],
                ],
                [
                    [
                        c64(0.9133108288727866, -0.38508398385543624),
                        c64(0.008042664765986755, 0.03190603796445517)
                    ],
                    [
                        c64(-6.9151682810378095, 1.800750901131042),
                        c64(0.5235871604669029, -0.18886435408156288)
                    ],
                ],
                [
                    [
                        c64(0.849070850314753, -0.49577931076259807),
                        c64(0.01381064392153511, 0.04080882571424955)
                    ],
                    [
                        c64(-6.688405272002992, 2.4133819411904995),
                        c64(0.4942211124266797, -0.24774648346309974)
                    ],
                ],
            ]),
            String::from("test"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:43 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Row = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 2.500\n\
            set_Vd(mA) = 11.250\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_2p5V_11p25mA.s2p\n\
            Vg(V) = -0.520\n\
            Vg(mA) = -0.001\n\
            Vd(V) = 2.500\n\
            Vd(mA) = 11.076",
            ),
        );
        debug_assert!(
            exemplar.name() == net.name(),
            " Failed name\n  exemplar: {}\n       net: {}",
            exemplar.name(),
            net.name()
        );
        comp_line(exemplar.comments(), net.comments(), "comments(test.s2p)");
        debug_assert!(
            exemplar.nports() == net.nports(),
            " Failed nports\n  exemplar: {}\n       net: {}",
            exemplar.nports(),
            net.nports()
        );
        debug_assert!(
            exemplar.npts() == net.npts(),
            " Failed npts\n  exemplar: {}\n       net: {}",
            exemplar.npts(),
            net.npts()
        );
        for i in 0..exemplar.nports() {
            debug_assert_eq!(
                exemplar.z0()[i],
                net.z0()[i],
                " Failed z0 for port {}\n  exemplar: {}\n       net: {}",
                i,
                exemplar.z0()[i],
                net.z0()[i]
            );
        }
        for i in 0..exemplar.npts() {
            assert_eq!(exemplar.freq().freq(i), net.freq().freq(i));
        }
        comp_points_c64(
            &exemplar.net(RFParameter::S),
            &net.net(RFParameter::S),
            F64Margin::default(),
            "net(test.s2p)",
        );

        let filename = "./data/test_2.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let exemplar = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64(50.0, 0.0), c64(50.0, 0.0)],
            RFParameter::S,
            ComplexNumberType::MagAng,
            vec![
                vec![
                    (0.9997, -4.5700),
                    (0.0386, 85.6170),
                    (0.0386, 85.5540),
                    (0.9991, -4.0220),
                ],
                vec![
                    (0.9977, -9.1360),
                    (0.0768, 81.3790),
                    (0.0767, 81.4160),
                    (0.9971, -8.0560),
                ],
                vec![
                    (0.9948, -13.6600),
                    (0.1142, 77.1890),
                    (0.1142, 77.2780),
                    (0.9944, -12.0560),
                ],
                vec![
                    (0.9907, -18.1160),
                    (0.1506, 73.1160),
                    (0.1506, 73.1740),
                    (0.9904, -15.9990),
                ],
                vec![
                    (0.9856, -22.4900),
                    (0.1858, 69.0460),
                    (0.1857, 69.1280),
                    (0.9854, -19.8790),
                ],
                vec![
                    (0.9691, -25.9160),
                    (0.2108, 64.3700),
                    (0.2108, 64.3900),
                    (0.9708, -22.9060),
                ],
                vec![
                    (0.9613, -29.9500),
                    (0.2409, 60.6450),
                    (0.2411, 60.7260),
                    (0.9632, -26.4740),
                ],
                vec![
                    (0.9526, -33.8690),
                    (0.2692, 56.9910),
                    (0.2694, 57.0490),
                    (0.9553, -29.9600),
                ],
                vec![
                    (0.9434, -37.6660),
                    (0.2958, 53.4140),
                    (0.2961, 53.4520),
                    (0.9469, -33.3510),
                ],
                vec![
                    (0.9342, -41.3370),
                    (0.3204, 49.9630),
                    (0.3207, 50.0050),
                    (0.9383, -36.6350),
                ],
                vec![
                    (0.9252, -44.9140),
                    (0.3433, 46.5880),
                    (0.3436, 46.6180),
                    (0.9298, -39.8330),
                ],
                vec![
                    (0.9162, -48.3610),
                    (0.3643, 43.3230),
                    (0.3646, 43.3770),
                    (0.9214, -42.9460),
                ],
                vec![
                    (0.9074, -51.6650),
                    (0.3834, 40.1680),
                    (0.3836, 40.2370),
                    (0.9132, -45.9500),
                ],
                vec![
                    (0.8990, -54.8470),
                    (0.4007, 37.1270),
                    (0.4011, 37.2140),
                    (0.9053, -48.8560),
                ],
                vec![
                    (0.8913, -57.9620),
                    (0.4165, 34.2060),
                    (0.4169, 34.2800),
                    (0.8978, -51.6650),
                ],
                vec![
                    (0.8847, -60.7240),
                    (0.4299, 31.5950),
                    (0.4303, 31.6560),
                    (0.8908, -54.1840),
                ],
                vec![
                    (0.8770, -63.5900),
                    (0.4427, 28.8260),
                    (0.4431, 28.8990),
                    (0.8839, -56.8260),
                ],
                vec![
                    (0.8699, -66.3240),
                    (0.4542, 26.1620),
                    (0.4545, 26.2290),
                    (0.8774, -59.3850),
                ],
                vec![
                    (0.8637, -68.9550),
                    (0.4644, 23.6030),
                    (0.4648, 23.6780),
                    (0.8714, -61.8510),
                ],
                vec![
                    (0.8578, -71.5130),
                    (0.4734, 21.1220),
                    (0.4738, 21.2040),
                    (0.8658, -64.2410),
                ],
                vec![
                    (0.8525, -73.9830),
                    (0.4813, 18.6920),
                    (0.4818, 18.8060),
                    (0.8607, -66.5480),
                ],
                vec![
                    (0.8475, -76.3470),
                    (0.4883, 16.4380),
                    (0.4888, 16.5110),
                    (0.8564, -68.7960),
                ],
                vec![
                    (0.8432, -78.6160),
                    (0.4945, 14.1950),
                    (0.4949, 14.2710),
                    (0.8527, -70.9890),
                ],
                vec![
                    (0.8394, -80.8370),
                    (0.4996, 11.9940),
                    (0.5002, 12.0840),
                    (0.8494, -73.1110),
                ],
                vec![
                    (0.8362, -82.9970),
                    (0.5041, 9.9100),
                    (0.5046, 10.0150),
                    (0.8463, -75.2390),
                ],
                vec![
                    (0.8332, -85.0540),
                    (0.5079, 7.8670),
                    (0.5084, 7.9540),
                    (0.8436, -77.2960),
                ],
                vec![
                    (0.8307, -87.0630),
                    (0.5109, 5.8800),
                    (0.5116, 5.9750),
                    (0.8415, -79.2450),
                ],
                vec![
                    (0.8286, -89.0270),
                    (0.5133, 3.9670),
                    (0.5139, 4.0510),
                    (0.8399, -81.1440),
                ],
                vec![
                    (0.8269, -90.9170),
                    (0.5152, 2.0680),
                    (0.5159, 2.1620),
                    (0.8385, -83.0620),
                ],
                vec![
                    (0.8258, -92.7180),
                    (0.5165, 0.2680),
                    (0.5171, 0.3580),
                    (0.8370, -84.9180),
                ],
                vec![
                    (0.8249, -94.4520),
                    (0.5174, -1.5330),
                    (0.5180, -1.4010),
                    (0.8361, -86.7140),
                ],
                vec![
                    (0.8239, -96.1840),
                    (0.5181, -3.2500),
                    (0.5187, -3.0980),
                    (0.8356, -88.4510),
                ],
                vec![
                    (0.8230, -97.8750),
                    (0.5181, -4.8970),
                    (0.5188, -4.7920),
                    (0.8352, -90.1490),
                ],
                vec![
                    (0.8227, -99.5140),
                    (0.5181, -6.5450),
                    (0.5188, -6.4310),
                    (0.8354, -91.8120),
                ],
                vec![
                    (0.8226, -101.0650),
                    (0.5175, -8.1950),
                    (0.5181, -8.0410),
                    (0.8353, -93.5010),
                ],
                vec![
                    (0.8223, -102.5590),
                    (0.5165, -9.7690),
                    (0.5172, -9.6050),
                    (0.8355, -95.1290),
                ],
                vec![
                    (0.8225, -104.0570),
                    (0.5154, -11.2930),
                    (0.5162, -11.1180),
                    (0.8364, -96.6950),
                ],
                vec![
                    (0.8231, -105.5240),
                    (0.5137, -12.7590),
                    (0.5147, -12.6350),
                    (0.8371, -98.2240),
                ],
                vec![
                    (0.8241, -106.9600),
                    (0.5119, -14.2310),
                    (0.5128, -14.1120),
                    (0.8382, -99.7270),
                ],
                vec![
                    (0.8253, -108.3330),
                    (0.5097, -15.6750),
                    (0.5106, -15.5180),
                    (0.8389, -101.1940),
                ],
                vec![
                    (0.8258, -109.6870),
                    (0.5075, -17.0380),
                    (0.5085, -16.9260),
                    (0.8405, -102.6710),
                ],
                vec![
                    (0.8261, -110.9890),
                    (0.5055, -18.3850),
                    (0.5063, -18.2620),
                    (0.8423, -104.1270),
                ],
                vec![
                    (0.8275, -112.3000),
                    (0.5031, -19.7310),
                    (0.5040, -19.6090),
                    (0.8438, -105.5160),
                ],
                vec![
                    (0.8294, -113.5080),
                    (0.5006, -21.0480),
                    (0.5015, -20.9170),
                    (0.8455, -106.8720),
                ],
                vec![
                    (0.8317, -114.7190),
                    (0.4981, -22.3450),
                    (0.4989, -22.2220),
                    (0.8461, -108.2350),
                ],
                vec![
                    (0.8328, -115.9170),
                    (0.4953, -23.6440),
                    (0.4962, -23.5150),
                    (0.8477, -109.6420),
                ],
                vec![
                    (0.8342, -117.1200),
                    (0.4920, -24.9740),
                    (0.4929, -24.8440),
                    (0.8498, -111.0210),
                ],
                vec![
                    (0.8355, -118.2530),
                    (0.4882, -26.2080),
                    (0.4892, -26.0620),
                    (0.8512, -112.3040),
                ],
                vec![
                    (0.8388, -119.3640),
                    (0.4850, -27.3340),
                    (0.4858, -27.2030),
                    (0.8529, -113.5140),
                ],
                vec![
                    (0.8418, -120.4270),
                    (0.4816, -28.4820),
                    (0.4824, -28.3360),
                    (0.8543, -114.7350),
                ],
                vec![
                    (0.8435, -121.6080),
                    (0.4780, -29.6320),
                    (0.4789, -29.4760),
                    (0.8562, -115.9910),
                ],
                vec![
                    (0.8439, -122.7410),
                    (0.4745, -30.7760),
                    (0.4754, -30.6250),
                    (0.8585, -117.2740),
                ],
                vec![
                    (0.8454, -123.7960),
                    (0.4711, -31.8920),
                    (0.4718, -31.7520),
                    (0.8605, -118.4640),
                ],
                vec![
                    (0.8477, -124.7240),
                    (0.4676, -32.9900),
                    (0.4683, -32.8420),
                    (0.8624, -119.5750),
                ],
                vec![
                    (0.8496, -125.6220),
                    (0.4630, -34.0920),
                    (0.4637, -33.9810),
                    (0.8635, -120.6300),
                ],
                vec![
                    (0.8508, -126.5730),
                    (0.4587, -35.1450),
                    (0.4595, -35.0120),
                    (0.8659, -121.8020),
                ],
                vec![
                    (0.8546, -127.5590),
                    (0.4548, -36.0700),
                    (0.4557, -35.9420),
                    (0.8681, -123.0140),
                ],
                vec![
                    (0.8572, -128.5610),
                    (0.4511, -37.0650),
                    (0.4520, -36.9530),
                    (0.8691, -124.1420),
                ],
                vec![
                    (0.8595, -129.4770),
                    (0.4476, -38.0490),
                    (0.4485, -37.9180),
                    (0.8713, -125.1370),
                ],
                vec![
                    (0.8610, -130.3690),
                    (0.4439, -39.0260),
                    (0.4445, -38.9210),
                    (0.8727, -126.1120),
                ],
                vec![
                    (0.8619, -131.2120),
                    (0.4401, -40.0010),
                    (0.4408, -39.8460),
                    (0.8744, -127.1510),
                ],
                vec![
                    (0.8644, -132.1630),
                    (0.4360, -40.9220),
                    (0.4368, -40.8080),
                    (0.8756, -128.2190),
                ],
                vec![
                    (0.8678, -133.0570),
                    (0.4313, -41.9330),
                    (0.4321, -41.7490),
                    (0.8775, -129.2340),
                ],
                vec![
                    (0.8695, -133.9240),
                    (0.4273, -42.7520),
                    (0.4279, -42.6060),
                    (0.8795, -130.1880),
                ],
                vec![
                    (0.8698, -134.7070),
                    (0.4236, -43.5700),
                    (0.4244, -43.4770),
                    (0.8822, -131.1260),
                ],
                vec![
                    (0.8703, -135.5670),
                    (0.4200, -44.4010),
                    (0.4208, -44.2590),
                    (0.8839, -132.1120),
                ],
                vec![
                    (0.8723, -136.3690),
                    (0.4162, -45.3320),
                    (0.4168, -45.1800),
                    (0.8851, -133.0800),
                ],
                vec![
                    (0.8751, -137.1150),
                    (0.4125, -46.2390),
                    (0.4133, -46.0040),
                    (0.8868, -134.0510),
                ],
                vec![
                    (0.8780, -137.9440),
                    (0.4089, -47.1690),
                    (0.4094, -47.0010),
                    (0.8894, -134.9170),
                ],
                vec![
                    (0.8796, -138.6930),
                    (0.4042, -48.1110),
                    (0.4051, -47.8950),
                    (0.8913, -135.9100),
                ],
                vec![
                    (0.8816, -139.4280),
                    (0.3999, -48.8430),
                    (0.4008, -48.6610),
                    (0.8921, -136.7350),
                ],
                vec![
                    (0.8840, -140.2420),
                    (0.3963, -49.5950),
                    (0.3971, -49.3420),
                    (0.8942, -137.6370),
                ],
                vec![
                    (0.8874, -140.9610),
                    (0.3927, -50.3500),
                    (0.3932, -50.1470),
                    (0.8955, -138.5070),
                ],
                vec![
                    (0.8890, -141.6450),
                    (0.3892, -51.1650),
                    (0.3900, -50.9320),
                    (0.8972, -139.3080),
                ],
                vec![
                    (0.8890, -142.4170),
                    (0.3850, -51.9660),
                    (0.3861, -51.7680),
                    (0.8977, -140.1310),
                ],
                vec![
                    (0.8884, -143.1470),
                    (0.3809, -52.7790),
                    (0.3820, -52.5320),
                    (0.8990, -140.9070),
                ],
                vec![
                    (0.8900, -143.8340),
                    (0.3769, -53.6450),
                    (0.3777, -53.4100),
                    (0.9016, -141.7750),
                ],
                vec![
                    (0.8925, -144.4670),
                    (0.3726, -54.2490),
                    (0.3733, -54.0510),
                    (0.9033, -142.5540),
                ],
                vec![
                    (0.8943, -144.9700),
                    (0.3687, -54.9350),
                    (0.3697, -54.6870),
                    (0.9051, -143.4150),
                ],
                vec![
                    (0.8950, -145.7190),
                    (0.3652, -55.6040),
                    (0.3664, -55.3840),
                    (0.9075, -144.1870),
                ],
                vec![
                    (0.8963, -146.4980),
                    (0.3615, -56.2910),
                    (0.3627, -56.0420),
                    (0.9095, -144.9490),
                ],
                vec![
                    (0.8994, -147.1760),
                    (0.3586, -57.0250),
                    (0.3592, -56.7610),
                    (0.9108, -145.7820),
                ],
                vec![
                    (0.9017, -147.6680),
                    (0.3553, -57.6520),
                    (0.3562, -57.4660),
                    (0.9103, -146.4690),
                ],
                vec![
                    (0.9039, -148.2400),
                    (0.3524, -58.4910),
                    (0.3531, -58.2450),
                    (0.9129, -147.0710),
                ],
                vec![
                    (0.9033, -148.9290),
                    (0.3490, -59.1820),
                    (0.3495, -58.9680),
                    (0.9157, -147.5890),
                ],
                vec![
                    (0.9046, -149.5050),
                    (0.3444, -59.9580),
                    (0.3450, -59.7120),
                    (0.9186, -148.3540),
                ],
                vec![
                    (0.9074, -150.1160),
                    (0.3405, -60.4700),
                    (0.3410, -60.2480),
                    (0.9189, -149.2460),
                ],
                vec![
                    (0.9066, -150.6930),
                    (0.3363, -61.0710),
                    (0.3372, -60.8410),
                    (0.9202, -150.0210),
                ],
                vec![
                    (0.9054, -151.4220),
                    (0.3335, -61.6560),
                    (0.3339, -61.4920),
                    (0.9213, -150.6100),
                ],
                vec![
                    (0.9047, -151.9350),
                    (0.3303, -62.2450),
                    (0.3309, -62.0750),
                    (0.9228, -151.3150),
                ],
                vec![
                    (0.9074, -153.2860),
                    (0.3197, -63.2800),
                    (0.3203, -63.0540),
                    (0.9232, -152.4590),
                ],
                vec![
                    (0.9109, -153.8940),
                    (0.3162, -64.0060),
                    (0.3171, -63.7940),
                    (0.9243, -153.1140),
                ],
                vec![
                    (0.9131, -154.5900),
                    (0.3128, -64.5780),
                    (0.3128, -64.4250),
                    (0.9284, -153.5880),
                ],
                vec![
                    (0.9142, -155.1620),
                    (0.3097, -65.2040),
                    (0.3104, -65.0810),
                    (0.9297, -154.2260),
                ],
                vec![
                    (0.9139, -155.8070),
                    (0.3057, -65.6880),
                    (0.3054, -65.3660),
                    (0.9303, -154.7860),
                ],
                vec![
                    (0.9154, -156.3630),
                    (0.3015, -65.9650),
                    (0.3025, -65.7200),
                    (0.9289, -155.3220),
                ],
                vec![
                    (0.9162, -156.9580),
                    (0.2997, -66.6040),
                    (0.2991, -66.2980),
                    (0.9304, -155.7570),
                ],
                vec![
                    (0.9156, -157.3130),
                    (0.2968, -67.1420),
                    (0.2982, -66.8300),
                    (0.9361, -156.2090),
                ],
                vec![
                    (0.9146, -157.7990),
                    (0.2953, -67.7290),
                    (0.2957, -67.4720),
                    (0.9395, -156.7770),
                ],
                vec![
                    (0.9147, -158.3570),
                    (0.2914, -68.3310),
                    (0.2908, -67.9730),
                    (0.9402, -157.4030),
                ],
                vec![
                    (0.9167, -158.8000),
                    (0.2875, -68.8450),
                    (0.2882, -68.5100),
                    (0.9405, -158.1570),
                ],
                vec![
                    (0.9177, -159.4290),
                    (0.2836, -69.3880),
                    (0.2848, -68.9430),
                    (0.9382, -158.7780),
                ],
                vec![
                    (0.9195, -159.7820),
                    (0.2813, -69.6720),
                    (0.2826, -69.3610),
                    (0.9399, -159.3780),
                ],
                vec![
                    (0.9223, -160.1770),
                    (0.2796, -70.2190),
                    (0.2806, -69.9990),
                    (0.9410, -159.8950),
                ],
                vec![
                    (0.9219, -160.8520),
                    (0.2769, -70.8410),
                    (0.2770, -70.5840),
                    (0.9432, -160.3020),
                ],
                vec![
                    (0.9218, -161.4880),
                    (0.2742, -71.3520),
                    (0.2743, -71.1150),
                    (0.9446, -160.7170),
                ],
                vec![
                    (0.9227, -161.8020),
                    (0.2711, -71.9150),
                    (0.2718, -71.5320),
                    (0.9463, -161.2880),
                ],
                vec![
                    (0.9234, -162.1550),
                    (0.2688, -72.1900),
                    (0.2684, -72.0170),
                    (0.9481, -161.7740),
                ],
                vec![
                    (0.9230, -162.8370),
                    (0.2647, -72.8380),
                    (0.2661, -72.3310),
                    (0.9492, -162.3060),
                ],
                vec![
                    (0.9217, -163.4000),
                    (0.2632, -73.1970),
                    (0.2635, -72.9680),
                    (0.9507, -162.7410),
                ],
                vec![
                    (0.9213, -163.8520),
                    (0.2606, -73.5730),
                    (0.2617, -73.2540),
                    (0.9516, -163.1830),
                ],
                vec![
                    (0.9229, -164.3350),
                    (0.2586, -74.0500),
                    (0.2600, -73.8080),
                    (0.9528, -163.9080),
                ],
                vec![
                    (0.9234, -164.7270),
                    (0.2564, -74.9580),
                    (0.2570, -74.8070),
                    (0.9510, -164.5920),
                ],
                vec![
                    (0.9259, -165.0210),
                    (0.2541, -75.4770),
                    (0.2546, -75.1890),
                    (0.9503, -164.8220),
                ],
                vec![
                    (0.9254, -165.4450),
                    (0.2513, -75.8050),
                    (0.2521, -75.7140),
                    (0.9510, -165.3450),
                ],
                vec![
                    (0.9294, -165.7750),
                    (0.2490, -76.3940),
                    (0.2498, -75.9880),
                    (0.9522, -165.7660),
                ],
                vec![
                    (0.9308, -166.2040),
                    (0.2460, -76.7240),
                    (0.2460, -76.1090),
                    (0.9531, -166.1950),
                ],
                vec![
                    (0.9329, -166.7230),
                    (0.2443, -77.0670),
                    (0.2440, -77.0320),
                    (0.9563, -166.6510),
                ],
                vec![
                    (0.9334, -167.0540),
                    (0.2421, -77.7590),
                    (0.2434, -77.6320),
                    (0.9569, -167.0590),
                ],
                vec![
                    (0.9341, -167.3630),
                    (0.2392, -78.2740),
                    (0.2407, -77.6480),
                    (0.9546, -167.5540),
                ],
                vec![
                    (0.9398, -167.9850),
                    (0.2378, -78.8480),
                    (0.2385, -78.6180),
                    (0.9540, -167.8110),
                ],
                vec![
                    (0.9420, -168.6010),
                    (0.2359, -79.4440),
                    (0.2369, -79.1400),
                    (0.9574, -168.1360),
                ],
                vec![
                    (0.9421, -169.1660),
                    (0.2339, -79.7850),
                    (0.2333, -79.8960),
                    (0.9581, -168.5830),
                ],
                vec![
                    (0.9438, -169.5190),
                    (0.2297, -80.7600),
                    (0.2313, -80.5260),
                    (0.9591, -168.9760),
                ],
                vec![
                    (0.9409, -169.9260),
                    (0.2274, -81.2040),
                    (0.2273, -81.0850),
                    (0.9569, -169.4860),
                ],
                vec![
                    (0.9424, -170.5240),
                    (0.2237, -81.6930),
                    (0.2257, -81.2410),
                    (0.9598, -169.7220),
                ],
                vec![
                    (0.9444, -170.9300),
                    (0.2221, -82.0410),
                    (0.2225, -81.7030),
                    (0.9623, -170.2620),
                ],
                vec![
                    (0.9457, -170.8610),
                    (0.2192, -82.8000),
                    (0.2202, -82.2520),
                    (0.9635, -170.4900),
                ],
                vec![
                    (0.9502, -171.1110),
                    (0.2169, -83.0300),
                    (0.2206, -82.8730),
                    (0.9619, -170.7030),
                ],
                vec![
                    (0.9512, -171.7310),
                    (0.2157, -83.7030),
                    (0.2155, -83.1130),
                    (0.9658, -171.2780),
                ],
                vec![
                    (0.9484, -172.3730),
                    (0.2125, -84.0790),
                    (0.2137, -84.0170),
                    (0.9645, -172.0970),
                ],
                vec![
                    (0.9498, -172.8500),
                    (0.2072, -84.8270),
                    (0.2090, -84.4050),
                    (0.9612, -172.7660),
                ],
                vec![
                    (0.9468, -173.0800),
                    (0.2050, -85.2920),
                    (0.2062, -85.2690),
                    (0.9611, -172.7550),
                ],
                vec![
                    (0.9441, -173.5190),
                    (0.2027, -85.2220),
                    (0.2027, -84.9570),
                    (0.9544, -173.1480),
                ],
                vec![
                    (0.9407, -174.5910),
                    (0.1994, -85.3030),
                    (0.1999, -85.3930),
                    (0.9510, -173.4960),
                ],
                vec![
                    (0.9367, -174.7780),
                    (0.1973, -85.3940),
                    (0.1975, -85.7300),
                    (0.9454, -173.8510),
                ],
                vec![
                    (0.9331, -175.0140),
                    (0.1954, -85.6940),
                    (0.1964, -85.7570),
                    (0.9455, -174.0080),
                ],
                vec![
                    (0.9306, -175.3730),
                    (0.1930, -85.9270),
                    (0.1935, -86.1550),
                    (0.9468, -174.2250),
                ],
                vec![
                    (0.9316, -175.7510),
                    (0.1905, -86.2360),
                    (0.1907, -86.4200),
                    (0.9450, -174.4320),
                ],
                vec![
                    (0.9343, -176.0990),
                    (0.1881, -86.3190),
                    (0.1885, -86.5280),
                    (0.9437, -174.5070),
                ],
                vec![
                    (0.9373, -176.4800),
                    (0.1859, -86.3240),
                    (0.1867, -86.5370),
                    (0.9444, -174.6210),
                ],
                vec![
                    (0.9400, -176.9060),
                    (0.1837, -86.4030),
                    (0.1848, -86.5660),
                    (0.9461, -174.7450),
                ],
                vec![
                    (0.9446, -177.4360),
                    (0.1826, -86.6400),
                    (0.1833, -86.6320),
                    (0.9503, -174.9930),
                ],
                vec![
                    (0.9436, -177.9820),
                    (0.1816, -86.9360),
                    (0.1820, -86.8810),
                    (0.9481, -175.2920),
                ],
                vec![
                    (0.9397, -178.4480),
                    (0.1806, -87.3350),
                    (0.1808, -87.2890),
                    (0.9479, -175.6400),
                ],
                vec![
                    (0.9363, -178.7090),
                    (0.1792, -87.7690),
                    (0.1794, -87.8020),
                    (0.9487, -175.8500),
                ],
                vec![
                    (0.9341, -178.9730),
                    (0.1769, -88.4620),
                    (0.1772, -88.4380),
                    (0.9529, -176.1120),
                ],
                vec![
                    (0.9308, -179.3850),
                    (0.1746, -88.9080),
                    (0.1749, -88.8440),
                    (0.9504, -176.6080),
                ],
                vec![
                    (0.9244, -179.6160),
                    (0.1724, -89.3480),
                    (0.1727, -89.3040),
                    (0.9466, -176.8220),
                ],
                vec![
                    (0.9193, -179.5730),
                    (0.1693, -89.8770),
                    (0.1694, -89.7870),
                    (0.9457, -176.8610),
                ],
                vec![
                    (0.9175, -179.4200),
                    (0.1647, -90.0820),
                    (0.1649, -90.1110),
                    (0.9465, -176.9090),
                ],
                vec![
                    (0.9216, -179.2540),
                    (0.1607, -89.4400),
                    (0.1606, -89.4930),
                    (0.9490, -177.0740),
                ],
                vec![
                    (0.9291, -179.3190),
                    (0.1585, -88.4380),
                    (0.1585, -88.4800),
                    (0.9520, -177.2580),
                ],
                vec![
                    (0.9343, -179.7990),
                    (0.1582, -87.7170),
                    (0.1585, -87.9340),
                    (0.9554, -177.6340),
                ],
                vec![
                    (0.9389, 179.5780),
                    (0.1585, -87.4100),
                    (0.1587, -87.5660),
                    (0.9586, -177.9930),
                ],
                vec![
                    (0.9429, 179.0250),
                    (0.1587, -87.3120),
                    (0.1588, -87.4780),
                    (0.9592, -178.3390),
                ],
                vec![
                    (0.9448, 178.4800),
                    (0.1582, -87.5920),
                    (0.1582, -87.7560),
                    (0.9561, -178.6730),
                ],
                vec![
                    (0.9400, 177.9680),
                    (0.1571, -87.9280),
                    (0.1571, -88.0370),
                    (0.9552, -179.0320),
                ],
                vec![
                    (0.9383, 177.5370),
                    (0.1561, -87.9180),
                    (0.1561, -88.1200),
                    (0.9550, -179.1660),
                ],
                vec![
                    (0.9329, 177.4910),
                    (0.1554, -88.0180),
                    (0.1555, -88.1590),
                    (0.9507, -179.4410),
                ],
                vec![
                    (0.9356, 177.2420),
                    (0.1546, -87.9840),
                    (0.1546, -88.2130),
                    (0.9513, -179.6480),
                ],
                vec![
                    (0.9320, 177.1700),
                    (0.1537, -88.1730),
                    (0.1538, -88.3310),
                    (0.9473, -179.3940),
                ],
                vec![
                    (0.9277, 177.0530),
                    (0.1529, -88.3950),
                    (0.1530, -88.4890),
                    (0.9442, -179.5220),
                ],
                vec![
                    (0.9264, 176.7610),
                    (0.1521, -88.5050),
                    (0.1523, -88.7420),
                    (0.9422, -179.9030),
                ],
                vec![
                    (0.9342, 176.4600),
                    (0.1507, -88.4400),
                    (0.1507, -88.6850),
                    (0.9455, -179.9650),
                ],
                vec![
                    (0.9393, 175.8150),
                    (0.1502, -88.3620),
                    (0.1498, -88.6340),
                    (0.9525, 179.9990),
                ],
                vec![
                    (0.9383, 175.3280),
                    (0.1494, -88.2070),
                    (0.1494, -88.5840),
                    (0.9560, 179.8170),
                ],
                vec![
                    (0.9394, 174.8040),
                    (0.1493, -88.2970),
                    (0.1493, -88.5760),
                    (0.9559, 179.4130),
                ],
                vec![
                    (0.9403, 174.3790),
                    (0.1492, -88.3660),
                    (0.1494, -88.6880),
                    (0.9543, 179.3140),
                ],
                vec![
                    (0.9375, 173.9810),
                    (0.1488, -88.9010),
                    (0.1490, -89.2190),
                    (0.9559, 179.2830),
                ],
                vec![
                    (0.9362, 173.6830),
                    (0.1477, -89.1130),
                    (0.1477, -89.4520),
                    (0.9607, 179.0860),
                ],
                vec![
                    (0.9334, 173.5040),
                    (0.1465, -89.2320),
                    (0.1467, -89.4640),
                    (0.9621, 178.8980),
                ],
                vec![
                    (0.9335, 173.3480),
                    (0.1460, -89.3120),
                    (0.1455, -89.4010),
                    (0.9612, 178.6870),
                ],
                vec![
                    (0.9309, 173.2340),
                    (0.1449, -89.3030),
                    (0.1448, -89.5670),
                    (0.9619, 178.6200),
                ],
                vec![
                    (0.9247, 173.1050),
                    (0.1445, -89.5340),
                    (0.1442, -89.7880),
                    (0.9646, 178.4200),
                ],
                vec![
                    (0.9266, 173.1140),
                    (0.1435, -89.5620),
                    (0.1437, -89.9420),
                    (0.9629, 178.1720),
                ],
                vec![
                    (0.9335, 172.6430),
                    (0.1429, -89.7650),
                    (0.1429, -90.1230),
                    (0.9647, 178.0270),
                ],
                vec![
                    (0.9365, 171.7940),
                    (0.1423, -89.7380),
                    (0.1424, -90.1050),
                    (0.9676, 177.8760),
                ],
                vec![
                    (0.9371, 171.2170),
                    (0.1420, -89.8850),
                    (0.1422, -90.3070),
                    (0.9711, 177.6460),
                ],
                vec![
                    (0.9363, 170.6940),
                    (0.1415, -90.0590),
                    (0.1417, -90.2920),
                    (0.9733, 177.3700),
                ],
                vec![
                    (0.9402, 170.4960),
                    (0.1409, -90.3310),
                    (0.1410, -90.5340),
                    (0.9740, 177.1220),
                ],
                vec![
                    (0.9416, 170.0070),
                    (0.1403, -90.2590),
                    (0.1406, -90.4490),
                    (0.9734, 176.9480),
                ],
                vec![
                    (0.9362, 169.5160),
                    (0.1407, -90.5860),
                    (0.1406, -90.8080),
                    (0.9751, 176.7560),
                ],
                vec![
                    (0.9343, 169.5360),
                    (0.1395, -90.5550),
                    (0.1395, -90.9440),
                    (0.9772, 176.6250),
                ],
                vec![
                    (0.9391, 169.7560),
                    (0.1391, -90.7470),
                    (0.1392, -90.9590),
                    (0.9776, 176.5740),
                ],
                vec![
                    (0.9388, 169.7040),
                    (0.1373, -90.8700),
                    (0.1384, -91.2550),
                    (0.9792, 176.4570),
                ],
                vec![
                    (0.9298, 169.4700),
                    (0.1379, -91.5400),
                    (0.1380, -91.7200),
                    (0.9830, 176.3080),
                ],
                vec![
                    (0.9259, 169.7520),
                    (0.1367, -90.9830),
                    (0.1367, -91.5490),
                    (0.9838, 176.1650),
                ],
                vec![
                    (0.9263, 169.9420),
                    (0.1360, -90.8400),
                    (0.1364, -91.4820),
                    (0.9849, 176.0810),
                ],
                vec![
                    (0.9314, 169.2990),
                    (0.1365, -91.5610),
                    (0.1367, -91.8730),
                    (0.9909, 176.0010),
                ],
                vec![
                    (0.9259, 168.2630),
                    (0.1373, -91.9160),
                    (0.1371, -92.3240),
                    (0.9980, 175.5270),
                ],
                vec![
                    (0.9198, 167.9210),
                    (0.1372, -92.7210),
                    (0.1372, -93.0570),
                    (1.0018, 175.1720),
                ],
                vec![
                    (0.9197, 167.6180),
                    (0.1359, -93.2110),
                    (0.1359, -93.5170),
                    (1.0039, 174.9120),
                ],
                vec![
                    (0.9220, 167.4130),
                    (0.1352, -93.8530),
                    (0.1348, -94.0230),
                    (1.0086, 174.7220),
                ],
                vec![
                    (0.9251, 166.9210),
                    (0.1332, -94.2360),
                    (0.1334, -94.1310),
                    (1.0144, 174.4450),
                ],
                vec![
                    (0.9230, 166.5080),
                    (0.1332, -93.9740),
                    (0.1324, -94.1880),
                    (1.0204, 173.9220),
                ],
                vec![
                    (0.9194, 166.3510),
                    (0.1305, -94.1510),
                    (0.1307, -94.2430),
                    (1.0189, 173.6440),
                ],
                vec![
                    (0.9164, 166.0750),
                    (0.1305, -94.0870),
                    (0.1311, -94.4490),
                    (1.0261, 173.6460),
                ],
                vec![
                    (0.9090, 165.9330),
                    (0.1307, -94.6070),
                    (0.1305, -94.7370),
                    (1.0334, 173.3210),
                ],
                vec![
                    (0.8974, 166.1260),
                    (0.1295, -95.0190),
                    (0.1296, -95.0320),
                    (1.0382, 172.8970),
                ],
                vec![
                    (0.8970, 166.5770),
                    (0.1280, -95.1610),
                    (0.1279, -95.3310),
                    (1.0430, 172.4240),
                ],
                vec![
                    (0.8916, 166.6850),
                    (0.1261, -95.3460),
                    (0.1264, -95.4250),
                    (1.0498, 171.7710),
                ],
                vec![
                    (0.8893, 166.0540),
                    (0.1248, -95.5920),
                    (0.1246, -95.4500),
                    (1.0560, 170.9670),
                ],
                vec![
                    (0.8893, 165.9180),
                    (0.1239, -95.3760),
                    (0.1238, -95.4160),
                    (1.0567, 170.3040),
                ],
                vec![
                    (0.8923, 165.6760),
                    (0.1221, -94.9750),
                    (0.1225, -94.8970),
                    (1.0588, 169.8410),
                ],
                vec![
                    (0.8939, 164.8690),
                    (0.1224, -94.3980),
                    (0.1225, -94.5880),
                    (1.0577, 169.2970),
                ],
                vec![
                    (0.8920, 164.1570),
                    (0.1212, -94.8570),
                    (0.1218, -94.5420),
                    (1.0617, 168.7300),
                ],
                vec![
                    (0.8850, 163.6360),
                    (0.1203, -94.0440),
                    (0.1209, -94.2150),
                    (1.0628, 167.9800),
                ],
                vec![
                    (0.8789, 163.5110),
                    (0.1209, -93.8750),
                    (0.1212, -93.5170),
                    (1.0617, 167.5850),
                ],
                vec![
                    (0.8750, 163.3600),
                    (0.1212, -93.4200),
                    (0.1211, -93.2440),
                    (1.0630, 166.9760),
                ],
                vec![
                    (0.8696, 163.4150),
                    (0.1225, -92.7030),
                    (0.1229, -92.8970),
                    (1.0645, 166.3860),
                ],
                vec![
                    (0.8688, 163.9670),
                    (0.1233, -92.7470),
                    (0.1229, -92.7930),
                    (1.0621, 165.8800),
                ],
                vec![
                    (0.8685, 164.3100),
                    (0.1248, -93.0750),
                    (0.1249, -92.8810),
                    (1.0595, 165.7450),
                ],
                vec![
                    (0.8749, 164.4550),
                    (0.1257, -93.4970),
                    (0.1255, -93.2950),
                    (1.0630, 165.5100),
                ],
                vec![
                    (0.8674, 164.1900),
                    (0.1264, -93.9530),
                    (0.1261, -93.8470),
                    (1.0640, 165.3760),
                ],
                vec![
                    (0.8621, 164.3590),
                    (0.1257, -94.8300),
                    (0.1252, -94.7070),
                    (1.0732, 165.3860),
                ],
                vec![
                    (0.8645, 164.3690),
                    (0.1252, -95.6160),
                    (0.1246, -95.1250),
                    (1.0841, 165.1780),
                ],
                vec![
                    (0.8682, 164.2530),
                    (0.1237, -95.4510),
                    (0.1234, -95.4040),
                    (1.1018, 164.7300),
                ],
                vec![
                    (0.8674, 163.4420),
                    (0.1236, -95.4490),
                    (0.1237, -95.2760),
                    (1.1191, 164.2070),
                ],
                vec![
                    (0.8578, 162.9450),
                    (0.1240, -94.5630),
                    (0.1241, -94.6250),
                    (1.1450, 163.3870),
                ],
            ],
            String::from("test_2"),
            String::from(
                "1915934D06 QPHT09 360 um ID No. EG0520U_1212_6x60_2SDBCB2EV R/C 29 52 8/23/2019 1:18 PM\n\
            Vds= 0.000 V Ids= 0.002 mA Vg1s= -1.000 V Ig1s= -0.00000 mA BiasType= Fixed_Vg_Vd\n\
            JobName= EG0520 Calstds= GaAs BCB Calkit= EG2306\n\
            WEIGHT 1\n\
            File name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Base file name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952\n\
            Date = 8/23/2019\n\
            Time = 1:18 PM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Attenuation = 0\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520U_1212_6x60_2SDBCB2EV\n\
            Row = 29\n\
            Column = 52\n\
            Technology = QPHT09\n\
            Lot/Wafer = 1915934D06\n\
            Gate Size = 360\n\
            Mask = EG0520\n\
            Cal Standards = GaAs BCB\n\
            Cal Kit = EG2306\n\
            Job Number = 20190821205423\n\
            Device Type = FET\n\
            set_Vg(v) = -1.000\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520U_1212_6x60_2SDBCB2EV_1915934D06_RC2952_Vg_N1.0V_Vd_0v.s2p\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n\
            Vg(V) = -1.000\n\
            Vg(mA) = -0.000",
            ),
        );
        debug_assert!(
            exemplar.name() == net.name(),
            " Failed name\n  exemplar: {}\n       net: {}",
            exemplar.name(),
            net.name()
        );
        comp_line(exemplar.comments(), net.comments(), "comments(test_2.s2p)");
        debug_assert!(
            exemplar.nports() == net.nports(),
            " Failed nports\n  exemplar: {}\n       net: {}",
            exemplar.nports(),
            net.nports()
        );
        debug_assert!(
            exemplar.npts() == net.npts(),
            " Failed npts\n  exemplar: {}\n       net: {}",
            exemplar.npts(),
            net.npts()
        );
        for i in 0..exemplar.nports() {
            debug_assert_eq!(
                exemplar.z0()[i],
                net.z0()[i],
                " Failed z0 for port {}\n  exemplar: {}\n       net: {}",
                i,
                exemplar.z0()[i],
                net.z0()[i]
            );
        }
        for i in 0..exemplar.npts() {
            assert_eq!(exemplar.freq().freq(i), net.freq().freq(i));
        }
        comp_points_c64(
            &exemplar.net(RFParameter::S),
            &net.net(RFParameter::S),
            F64Margin::default(),
            "net(test_2.s2p)",
        );

        let filename = "./data/test_3.s2p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let exemplar = Network::new_from(
            FrequencyBuilder::new()
                .start_stop_step_scaled(0.5, 110.0, 0.5, Scale::Giga)
                .build(),
            array![c64(50.0, 0.0), c64(50.0, 0.0)],
            RFParameter::S,
            ComplexNumberType::Db,
            vec![
                vec![
                    (-0.007, -2.703),
                    (-33.308, 87.205),
                    (-33.310, 87.223),
                    (0.002, -2.522),
                ],
                vec![
                    (-0.014, -5.380),
                    (-27.320, 84.747),
                    (-27.319, 84.763),
                    (-0.006, -5.048),
                ],
                vec![
                    (-0.020, -8.054),
                    (-23.824, 82.282),
                    (-23.822, 82.309),
                    (-0.011, -7.535),
                ],
                vec![
                    (-0.030, -10.721),
                    (-21.356, 79.850),
                    (-21.355, 79.863),
                    (-0.022, -10.008),
                ],
                vec![
                    (-0.070, -12.934),
                    (-19.767, 76.759),
                    (-19.766, 76.774),
                    (-0.073, -12.097),
                ],
                vec![
                    (-0.102, -15.528),
                    (-18.235, 74.354),
                    (-18.233, 74.341),
                    (-0.086, -14.420),
                ],
                vec![
                    (-0.122, -18.012),
                    (-16.968, 71.927),
                    (-16.964, 71.952),
                    (-0.118, -16.765),
                ],
                vec![
                    (-0.156, -20.492),
                    (-15.876, 69.541),
                    (-15.872, 69.531),
                    (-0.143, -19.112),
                ],
                vec![
                    (-0.185, -23.013),
                    (-14.932, 67.149),
                    (-14.928, 67.141),
                    (-0.181, -21.370),
                ],
                vec![
                    (-0.224, -25.457),
                    (-14.103, 64.781),
                    (-14.100, 64.755),
                    (-0.214, -23.659),
                ],
                vec![
                    (-0.265, -27.878),
                    (-13.365, 62.489),
                    (-13.360, 62.451),
                    (-0.244, -25.890),
                ],
                vec![
                    (-0.302, -30.240),
                    (-12.705, 60.203),
                    (-12.703, 60.163),
                    (-0.284, -28.109),
                ],
                vec![
                    (-0.344, -32.603),
                    (-12.109, 57.908),
                    (-12.107, 57.880),
                    (-0.322, -30.298),
                ],
                vec![
                    (-0.387, -34.903),
                    (-11.574, 55.657),
                    (-11.572, 55.620),
                    (-0.364, -32.496),
                ],
                vec![
                    (-0.429, -37.197),
                    (-11.087, 53.446),
                    (-11.086, 53.458),
                    (-0.407, -34.613),
                ],
                vec![
                    (-0.463, -39.356),
                    (-10.675, 51.542),
                    (-10.671, 51.469),
                    (-0.446, -36.438),
                ],
                vec![
                    (-0.515, -41.554),
                    (-10.268, 49.344),
                    (-10.270, 49.333),
                    (-0.488, -38.514),
                ],
                vec![
                    (-0.562, -43.746),
                    (-9.897, 47.250),
                    (-9.897, 47.193),
                    (-0.536, -40.558),
                ],
                vec![
                    (-0.610, -45.867),
                    (-9.555, 45.112),
                    (-9.555, 45.063),
                    (-0.582, -42.571),
                ],
                vec![
                    (-0.660, -47.981),
                    (-9.247, 43.031),
                    (-9.245, 42.983),
                    (-0.630, -44.513),
                ],
                vec![
                    (-0.704, -50.043),
                    (-8.963, 41.024),
                    (-8.964, 40.994),
                    (-0.679, -46.423),
                ],
                vec![
                    (-0.757, -52.064),
                    (-8.703, 39.062),
                    (-8.702, 39.007),
                    (-0.720, -48.330),
                ],
                vec![
                    (-0.799, -54.087),
                    (-8.460, 37.129),
                    (-8.460, 37.075),
                    (-0.767, -50.166),
                ],
                vec![
                    (-0.851, -56.013),
                    (-8.237, 35.258),
                    (-8.238, 35.248),
                    (-0.800, -51.986),
                ],
                vec![
                    (-0.898, -57.881),
                    (-8.025, 33.392),
                    (-8.025, 33.331),
                    (-0.839, -53.849),
                ],
                vec![
                    (-0.939, -59.713),
                    (-7.832, 31.566),
                    (-7.832, 31.496),
                    (-0.882, -55.676),
                ],
                vec![
                    (-0.982, -61.565),
                    (-7.656, 29.737),
                    (-7.658, 29.701),
                    (-0.927, -57.406),
                ],
                vec![
                    (-1.021, -63.357),
                    (-7.495, 27.999),
                    (-7.495, 27.920),
                    (-0.968, -59.135),
                ],
                vec![
                    (-1.056, -65.122),
                    (-7.345, 26.233),
                    (-7.349, 26.172),
                    (-1.010, -60.814),
                ],
                vec![
                    (-1.093, -66.877),
                    (-7.211, 24.513),
                    (-7.210, 24.431),
                    (-1.045, -62.470),
                ],
                vec![
                    (-1.130, -68.611),
                    (-7.084, 22.839),
                    (-7.085, 22.760),
                    (-1.077, -64.114),
                ],
                vec![
                    (-1.166, -70.296),
                    (-6.964, 21.187),
                    (-6.963, 21.105),
                    (-1.113, -65.762),
                ],
                vec![
                    (-1.199, -71.933),
                    (-6.856, 19.554),
                    (-6.855, 19.467),
                    (-1.149, -67.361),
                ],
                vec![
                    (-1.228, -73.544),
                    (-6.761, 17.969),
                    (-6.761, 17.907),
                    (-1.184, -68.913),
                ],
                vec![
                    (-1.258, -75.146),
                    (-6.668, 16.427),
                    (-6.669, 16.307),
                    (-1.209, -70.427),
                ],
                vec![
                    (-1.293, -76.757),
                    (-6.583, 14.877),
                    (-6.587, 14.748),
                    (-1.235, -71.896),
                ],
                vec![
                    (-1.316, -78.293),
                    (-6.503, 13.359),
                    (-6.507, 13.277),
                    (-1.261, -73.391),
                ],
                vec![
                    (-1.339, -79.774),
                    (-6.431, 11.869),
                    (-6.434, 11.757),
                    (-1.286, -74.841),
                ],
                vec![
                    (-1.362, -81.225),
                    (-6.369, 10.428),
                    (-6.372, 10.325),
                    (-1.308, -76.268),
                ],
                vec![
                    (-1.387, -82.666),
                    (-6.312, 8.931),
                    (-6.317, 8.864),
                    (-1.330, -77.697),
                ],
                vec![
                    (-1.405, -84.120),
                    (-6.257, 7.509),
                    (-6.262, 7.430),
                    (-1.347, -79.106),
                ],
                vec![
                    (-1.419, -85.518),
                    (-6.208, 6.101),
                    (-6.213, 5.976),
                    (-1.365, -80.482),
                ],
                vec![
                    (-1.429, -86.867),
                    (-6.165, 4.717),
                    (-6.169, 4.653),
                    (-1.384, -81.874),
                ],
                vec![
                    (-1.447, -88.250),
                    (-6.127, 3.340),
                    (-6.132, 3.226),
                    (-1.400, -83.269),
                ],
                vec![
                    (-1.465, -89.627),
                    (-6.089, 1.959),
                    (-6.092, 1.849),
                    (-1.410, -84.648),
                ],
                vec![
                    (-1.477, -90.985),
                    (-6.063, 0.594),
                    (-6.066, 0.471),
                    (-1.424, -85.964),
                ],
                vec![
                    (-1.481, -92.320),
                    (-6.052, -0.794),
                    (-6.055, -0.900),
                    (-1.441, -87.215),
                ],
                vec![
                    (-1.488, -93.626),
                    (-6.056, -2.034),
                    (-6.060, -2.165),
                    (-1.466, -88.427),
                ],
                vec![
                    (-1.504, -94.858),
                    (-6.038, -3.236),
                    (-6.039, -3.361),
                    (-1.474, -89.665),
                ],
                vec![
                    (-1.520, -96.087),
                    (-6.023, -4.482),
                    (-6.024, -4.590),
                    (-1.486, -90.966),
                ],
                vec![
                    (-1.522, -97.307),
                    (-6.009, -5.661),
                    (-6.008, -5.786),
                    (-1.490, -92.174),
                ],
                vec![
                    (-1.518, -98.492),
                    (-6.002, -6.874),
                    (-6.003, -6.991),
                    (-1.485, -93.349),
                ],
                vec![
                    (-1.517, -99.685),
                    (-5.991, -8.055),
                    (-5.995, -8.196),
                    (-1.487, -94.556),
                ],
                vec![
                    (-1.539, -100.827),
                    (-5.988, -9.359),
                    (-5.989, -9.478),
                    (-1.508, -95.800),
                ],
                vec![
                    (-1.549, -101.915),
                    (-6.011, -10.524),
                    (-6.016, -10.644),
                    (-1.520, -96.906),
                ],
                vec![
                    (-1.535, -103.121),
                    (-6.030, -11.560),
                    (-6.035, -11.671),
                    (-1.513, -97.983),
                ],
                vec![
                    (-1.527, -104.305),
                    (-6.029, -12.498),
                    (-6.032, -12.599),
                    (-1.483, -99.050),
                ],
                vec![
                    (-1.536, -105.323),
                    (-6.025, -13.680),
                    (-6.030, -13.823),
                    (-1.482, -100.338),
                ],
                vec![
                    (-1.543, -106.386),
                    (-6.035, -14.827),
                    (-6.036, -14.953),
                    (-1.508, -101.481),
                ],
                vec![
                    (-1.532, -107.502),
                    (-6.043, -15.925),
                    (-6.049, -16.028),
                    (-1.519, -102.578),
                ],
                vec![
                    (-1.523, -108.630),
                    (-6.073, -17.002),
                    (-6.074, -17.148),
                    (-1.520, -103.680),
                ],
                vec![
                    (-1.517, -109.630),
                    (-6.085, -18.045),
                    (-6.088, -18.163),
                    (-1.507, -104.645),
                ],
                vec![
                    (-1.517, -110.526),
                    (-6.108, -19.017),
                    (-6.115, -19.129),
                    (-1.502, -105.568),
                ],
                vec![
                    (-1.523, -111.551),
                    (-6.133, -19.965),
                    (-6.132, -20.109),
                    (-1.491, -106.576),
                ],
                vec![
                    (-1.520, -112.602),
                    (-6.146, -20.916),
                    (-6.148, -21.090),
                    (-1.497, -107.712),
                ],
                vec![
                    (-1.505, -113.579),
                    (-6.160, -21.929),
                    (-6.164, -22.133),
                    (-1.493, -108.803),
                ],
                vec![
                    (-1.496, -114.428),
                    (-6.187, -22.916),
                    (-6.190, -23.068),
                    (-1.490, -109.691),
                ],
                vec![
                    (-1.492, -115.261),
                    (-6.202, -23.910),
                    (-6.201, -24.079),
                    (-1.478, -110.630),
                ],
                vec![
                    (-1.479, -116.159),
                    (-6.233, -24.888),
                    (-6.235, -25.011),
                    (-1.486, -111.506),
                ],
                vec![
                    (-1.450, -117.227),
                    (-6.269, -25.884),
                    (-6.268, -25.983),
                    (-1.472, -112.436),
                ],
                vec![
                    (-1.415, -118.310),
                    (-6.304, -26.742),
                    (-6.300, -26.933),
                    (-1.449, -113.459),
                ],
                vec![
                    (-1.406, -119.173),
                    (-6.318, -27.627),
                    (-6.319, -27.787),
                    (-1.436, -114.468),
                ],
                vec![
                    (-1.416, -120.155),
                    (-6.333, -28.537),
                    (-6.336, -28.705),
                    (-1.442, -115.341),
                ],
                vec![
                    (-1.412, -121.119),
                    (-6.354, -29.574),
                    (-6.357, -29.733),
                    (-1.429, -116.311),
                ],
                vec![
                    (-1.375, -122.107),
                    (-6.378, -30.598),
                    (-6.386, -30.749),
                    (-1.411, -117.337),
                ],
                vec![
                    (-1.348, -122.983),
                    (-6.417, -31.525),
                    (-6.420, -31.689),
                    (-1.405, -118.305),
                ],
                vec![
                    (-1.340, -123.747),
                    (-6.459, -32.495),
                    (-6.472, -32.682),
                    (-1.397, -119.061),
                ],
                vec![
                    (-1.351, -124.603),
                    (-6.527, -33.267),
                    (-6.525, -33.467),
                    (-1.388, -119.865),
                ],
                vec![
                    (-1.339, -125.547),
                    (-6.566, -34.045),
                    (-6.567, -34.205),
                    (-1.374, -120.594),
                ],
                vec![
                    (-1.313, -126.503),
                    (-6.593, -34.907),
                    (-6.594, -35.053),
                    (-1.332, -121.429),
                ],
                vec![
                    (-1.287, -127.265),
                    (-6.622, -35.784),
                    (-6.631, -36.002),
                    (-1.303, -122.280),
                ],
                vec![
                    (-1.265, -127.923),
                    (-6.640, -36.573),
                    (-6.655, -36.827),
                    (-1.275, -123.026),
                ],
                vec![
                    (-1.276, -128.868),
                    (-6.676, -37.454),
                    (-6.683, -37.685),
                    (-1.248, -124.052),
                ],
                vec![
                    (-1.276, -129.931),
                    (-6.720, -38.438),
                    (-6.734, -38.649),
                    (-1.253, -124.964),
                ],
                vec![
                    (-1.258, -130.950),
                    (-6.775, -39.360),
                    (-6.782, -39.607),
                    (-1.251, -125.870),
                ],
                vec![
                    (-1.250, -131.674),
                    (-6.823, -40.202),
                    (-6.838, -40.386),
                    (-1.227, -126.610),
                ],
                vec![
                    (-1.245, -132.330),
                    (-6.872, -40.937),
                    (-6.868, -41.097),
                    (-1.203, -127.499),
                ],
                vec![
                    (-1.264, -133.188),
                    (-6.919, -41.663),
                    (-6.921, -41.828),
                    (-1.186, -128.275),
                ],
                vec![
                    (-1.241, -133.962),
                    (-6.979, -42.437),
                    (-6.979, -42.740),
                    (-1.181, -129.251),
                ],
                vec![
                    (-1.244, -134.873),
                    (-7.035, -43.207),
                    (-7.046, -43.432),
                    (-1.178, -130.173),
                ],
                vec![
                    (-1.233, -135.444),
                    (-7.079, -43.963),
                    (-7.079, -44.149),
                    (-1.163, -130.927),
                ],
                vec![
                    (-1.204, -137.506),
                    (-7.295, -45.675),
                    (-7.287, -45.895),
                    (-1.120, -132.380),
                ],
                vec![
                    (-1.196, -138.266),
                    (-7.331, -46.540),
                    (-7.340, -46.723),
                    (-1.101, -132.999),
                ],
                vec![
                    (-1.184, -139.237),
                    (-7.406, -47.224),
                    (-7.392, -47.498),
                    (-1.081, -133.941),
                ],
                vec![
                    (-1.156, -139.806),
                    (-7.439, -47.942),
                    (-7.442, -48.261),
                    (-1.066, -134.851),
                ],
                vec![
                    (-1.141, -140.499),
                    (-7.512, -48.823),
                    (-7.507, -49.026),
                    (-1.057, -135.763),
                ],
                vec![
                    (-1.151, -141.070),
                    (-7.544, -49.468),
                    (-7.552, -49.677),
                    (-1.033, -136.450),
                ],
                vec![
                    (-1.139, -141.988),
                    (-7.619, -50.241),
                    (-7.621, -50.511),
                    (-1.060, -137.163),
                ],
                vec![
                    (-1.126, -142.624),
                    (-7.678, -51.043),
                    (-7.661, -51.286),
                    (-1.045, -138.047),
                ],
                vec![
                    (-1.090, -143.317),
                    (-7.754, -51.827),
                    (-7.754, -51.818),
                    (-1.059, -138.695),
                ],
                vec![
                    (-1.090, -143.804),
                    (-7.812, -52.472),
                    (-7.801, -52.722),
                    (-0.973, -139.574),
                ],
                vec![
                    (-1.082, -144.444),
                    (-7.886, -53.018),
                    (-7.910, -53.153),
                    (-0.968, -140.498),
                ],
                vec![
                    (-1.091, -145.084),
                    (-7.937, -53.366),
                    (-7.948, -53.552),
                    (-0.981, -140.970),
                ],
                vec![
                    (-1.073, -145.770),
                    (-7.979, -53.983),
                    (-7.999, -54.149),
                    (-0.994, -141.596),
                ],
                vec![
                    (-1.063, -146.276),
                    (-8.043, -54.559),
                    (-8.026, -54.782),
                    (-1.008, -142.399),
                ],
                vec![
                    (-1.042, -146.836),
                    (-8.078, -55.281),
                    (-8.074, -55.531),
                    (-1.001, -143.284),
                ],
                vec![
                    (-1.033, -147.673),
                    (-8.132, -56.080),
                    (-8.110, -56.310),
                    (-0.981, -143.668),
                ],
                vec![
                    (-1.042, -148.451),
                    (-8.160, -56.728),
                    (-8.174, -56.898),
                    (-0.965, -144.102),
                ],
                vec![
                    (-1.024, -148.900),
                    (-8.249, -57.285),
                    (-8.226, -57.521),
                    (-0.941, -144.796),
                ],
                vec![
                    (-0.995, -149.415),
                    (-8.286, -57.910),
                    (-8.272, -58.101),
                    (-0.938, -145.596),
                ],
                vec![
                    (-0.989, -150.029),
                    (-8.310, -58.664),
                    (-8.327, -58.674),
                    (-0.907, -146.361),
                ],
                vec![
                    (-0.991, -150.499),
                    (-8.371, -59.225),
                    (-8.362, -59.245),
                    (-0.863, -146.713),
                ],
                vec![
                    (-1.005, -150.999),
                    (-8.404, -59.719),
                    (-8.422, -59.990),
                    (-0.836, -147.464),
                ],
                vec![
                    (-0.991, -151.557),
                    (-8.484, -60.340),
                    (-8.449, -60.444),
                    (-0.834, -148.351),
                ],
                vec![
                    (-0.992, -152.156),
                    (-8.521, -61.244),
                    (-8.512, -61.261),
                    (-0.813, -149.016),
                ],
                vec![
                    (-0.994, -152.469),
                    (-8.563, -61.662),
                    (-8.583, -62.009),
                    (-0.825, -149.822),
                ],
                vec![
                    (-0.996, -152.775),
                    (-8.620, -62.219),
                    (-8.625, -62.491),
                    (-0.844, -150.308),
                ],
                vec![
                    (-0.998, -153.136),
                    (-8.656, -62.852),
                    (-8.703, -63.102),
                    (-0.866, -150.828),
                ],
                vec![
                    (-0.956, -153.739),
                    (-8.707, -63.513),
                    (-8.698, -63.648),
                    (-0.869, -151.174),
                ],
                vec![
                    (-0.964, -154.217),
                    (-8.762, -64.131),
                    (-8.749, -64.117),
                    (-0.803, -151.983),
                ],
                vec![
                    (-0.949, -154.618),
                    (-8.809, -64.854),
                    (-8.772, -65.287),
                    (-0.774, -152.579),
                ],
                vec![
                    (-0.920, -155.032),
                    (-8.834, -65.647),
                    (-8.851, -65.891),
                    (-0.753, -153.294),
                ],
                vec![
                    (-0.903, -155.591),
                    (-8.860, -66.464),
                    (-8.894, -66.700),
                    (-0.741, -154.031),
                ],
                vec![
                    (-0.907, -156.265),
                    (-8.938, -67.399),
                    (-8.959, -67.342),
                    (-0.720, -154.847),
                ],
                vec![
                    (-0.896, -156.802),
                    (-9.031, -67.658),
                    (-9.033, -68.226),
                    (-0.692, -155.857),
                ],
                vec![
                    (-0.856, -157.201),
                    (-9.084, -68.597),
                    (-9.143, -68.901),
                    (-0.706, -156.318),
                ],
                vec![
                    (-0.874, -157.546),
                    (-9.202, -69.240),
                    (-9.200, -69.721),
                    (-0.727, -156.732),
                ],
                vec![
                    (-0.922, -158.150),
                    (-9.258, -69.906),
                    (-9.279, -70.164),
                    (-0.807, -157.422),
                ],
                vec![
                    (-0.947, -158.704),
                    (-9.308, -70.926),
                    (-9.333, -70.844),
                    (-0.835, -158.104),
                ],
                vec![
                    (-0.895, -159.033),
                    (-9.413, -71.526),
                    (-9.401, -71.758),
                    (-0.804, -158.610),
                ],
                vec![
                    (-0.854, -159.318),
                    (-9.522, -72.082),
                    (-9.547, -72.352),
                    (-0.752, -158.691),
                ],
                vec![
                    (-0.838, -159.873),
                    (-9.683, -72.618),
                    (-9.692, -73.056),
                    (-0.775, -159.259),
                ],
                vec![
                    (-0.849, -160.766),
                    (-9.746, -73.258),
                    (-9.704, -73.203),
                    (-0.850, -160.019),
                ],
                vec![
                    (-0.846, -161.398),
                    (-9.808, -73.008),
                    (-9.869, -73.463),
                    (-0.825, -161.307),
                ],
                vec![
                    (-0.756, -161.361),
                    (-9.948, -73.728),
                    (-9.931, -73.058),
                    (-0.713, -161.524),
                ],
                vec![
                    (-0.761, -162.046),
                    (-10.012, -74.283),
                    (-9.980, -73.262),
                    (-0.727, -162.222),
                ],
                vec![
                    (-0.760, -162.772),
                    (-10.097, -74.685),
                    (-10.090, -73.887),
                    (-0.778, -163.039),
                ],
                vec![
                    (-0.759, -163.211),
                    (-10.197, -74.987),
                    (-10.190, -74.236),
                    (-0.827, -163.492),
                ],
                vec![
                    (-0.743, -163.534),
                    (-10.288, -75.174),
                    (-10.306, -74.376),
                    (-0.845, -163.829),
                ],
                vec![
                    (-0.718, -164.022),
                    (-10.406, -75.414),
                    (-10.419, -74.530),
                    (-0.869, -164.190),
                ],
                vec![
                    (-0.713, -164.578),
                    (-10.501, -75.436),
                    (-10.514, -74.649),
                    (-0.882, -164.538),
                ],
                vec![
                    (-0.721, -164.938),
                    (-10.563, -75.321),
                    (-10.556, -74.375),
                    (-0.872, -164.822),
                ],
                vec![
                    (-0.740, -165.270),
                    (-10.616, -75.215),
                    (-10.615, -74.492),
                    (-0.882, -164.933),
                ],
                vec![
                    (-0.727, -165.532),
                    (-10.636, -75.164),
                    (-10.632, -74.424),
                    (-0.853, -165.150),
                ],
                vec![
                    (-0.704, -166.050),
                    (-10.623, -75.065),
                    (-10.634, -74.288),
                    (-0.849, -165.342),
                ],
                vec![
                    (-0.701, -166.646),
                    (-10.619, -75.230),
                    (-10.623, -74.467),
                    (-0.840, -165.459),
                ],
                vec![
                    (-0.696, -167.209),
                    (-10.617, -75.602),
                    (-10.618, -74.787),
                    (-0.809, -165.838),
                ],
                vec![
                    (-0.716, -167.810),
                    (-10.631, -75.931),
                    (-10.637, -75.128),
                    (-0.774, -166.238),
                ],
                vec![
                    (-0.741, -168.274),
                    (-10.612, -76.177),
                    (-10.622, -75.507),
                    (-0.745, -166.740),
                ],
                vec![
                    (-0.764, -168.664),
                    (-10.584, -76.727),
                    (-10.596, -76.030),
                    (-0.731, -167.341),
                ],
                vec![
                    (-0.765, -168.885),
                    (-10.590, -77.366),
                    (-10.609, -76.537),
                    (-0.708, -167.767),
                ],
                vec![
                    (-0.772, -169.145),
                    (-10.629, -77.899),
                    (-10.640, -77.188),
                    (-0.683, -168.348),
                ],
                vec![
                    (-0.750, -169.444),
                    (-10.667, -78.526),
                    (-10.673, -77.889),
                    (-0.677, -169.053),
                ],
                vec![
                    (-0.736, -169.760),
                    (-10.715, -79.048),
                    (-10.721, -78.526),
                    (-0.659, -169.607),
                ],
                vec![
                    (-0.729, -169.988),
                    (-10.774, -79.624),
                    (-10.776, -79.055),
                    (-0.645, -170.279),
                ],
                vec![
                    (-0.726, -170.132),
                    (-10.837, -80.185),
                    (-10.837, -79.496),
                    (-0.652, -170.933),
                ],
                vec![
                    (-0.700, -170.491),
                    (-10.903, -80.593),
                    (-10.902, -79.850),
                    (-0.681, -171.309),
                ],
                vec![
                    (-0.666, -170.890),
                    (-10.965, -81.065),
                    (-10.973, -80.269),
                    (-0.694, -171.561),
                ],
                vec![
                    (-0.659, -171.335),
                    (-11.040, -81.499),
                    (-11.050, -80.629),
                    (-0.707, -171.762),
                ],
                vec![
                    (-0.626, -171.867),
                    (-11.113, -81.675),
                    (-11.129, -80.926),
                    (-0.693, -172.083),
                ],
                vec![
                    (-0.607, -172.055),
                    (-11.179, -81.751),
                    (-11.191, -80.954),
                    (-0.685, -172.412),
                ],
                vec![
                    (-0.604, -172.623),
                    (-11.229, -82.010),
                    (-11.237, -81.134),
                    (-0.692, -172.867),
                ],
                vec![
                    (-0.581, -173.185),
                    (-11.233, -82.302),
                    (-11.249, -81.467),
                    (-0.727, -173.284),
                ],
                vec![
                    (-0.555, -173.548),
                    (-11.270, -82.562),
                    (-11.284, -81.615),
                    (-0.719, -173.825),
                ],
                vec![
                    (-0.529, -174.035),
                    (-11.295, -82.750),
                    (-11.294, -81.725),
                    (-0.699, -174.420),
                ],
                vec![
                    (-0.493, -174.425),
                    (-11.329, -82.919),
                    (-11.331, -81.878),
                    (-0.714, -174.634),
                ],
                vec![
                    (-0.474, -174.909),
                    (-11.354, -83.028),
                    (-11.362, -82.069),
                    (-0.722, -174.945),
                ],
                vec![
                    (-0.463, -175.232),
                    (-11.350, -83.348),
                    (-11.367, -82.421),
                    (-0.717, -175.277),
                ],
                vec![
                    (-0.432, -175.548),
                    (-11.368, -83.709),
                    (-11.363, -82.693),
                    (-0.739, -175.601),
                ],
                vec![
                    (-0.393, -176.181),
                    (-11.396, -84.163),
                    (-11.398, -83.231),
                    (-0.752, -175.840),
                ],
                vec![
                    (-0.385, -176.735),
                    (-11.445, -84.475),
                    (-11.452, -83.539),
                    (-0.736, -176.095),
                ],
                vec![
                    (-0.378, -177.418),
                    (-11.479, -84.621),
                    (-11.486, -83.766),
                    (-0.728, -176.458),
                ],
                vec![
                    (-0.357, -178.081),
                    (-11.505, -84.919),
                    (-11.501, -84.040),
                    (-0.745, -176.797),
                ],
                vec![
                    (-0.373, -178.550),
                    (-11.532, -85.329),
                    (-11.527, -84.379),
                    (-0.737, -177.162),
                ],
                vec![
                    (-0.398, -179.219),
                    (-11.555, -85.876),
                    (-11.537, -84.887),
                    (-0.712, -177.517),
                ],
                vec![
                    (-0.406, -179.387),
                    (-11.592, -86.304),
                    (-11.596, -85.116),
                    (-0.697, -178.130),
                ],
                vec![
                    (-0.388, -179.751),
                    (-11.608, -86.455),
                    (-11.617, -85.489),
                    (-0.708, -178.407),
                ],
                vec![
                    (-0.351, 179.972),
                    (-11.662, -86.823),
                    (-11.666, -85.790),
                    (-0.737, -178.606),
                ],
                vec![
                    (-0.309, 179.552),
                    (-11.715, -87.019),
                    (-11.706, -85.940),
                    (-0.744, -178.758),
                ],
                vec![
                    (-0.265, 179.038),
                    (-11.732, -87.209),
                    (-11.729, -86.175),
                    (-0.744, -178.952),
                ],
                vec![
                    (-0.268, 178.546),
                    (-11.731, -87.406),
                    (-11.728, -86.540),
                    (-0.754, -179.276),
                ],
                vec![
                    (-0.257, 178.084),
                    (-11.753, -87.778),
                    (-11.749, -87.009),
                    (-0.785, -179.361),
                ],
                vec![
                    (-0.252, 177.567),
                    (-11.757, -88.069),
                    (-11.774, -87.224),
                    (-0.760, -179.478),
                ],
                vec![
                    (-0.264, 177.133),
                    (-11.802, -88.337),
                    (-11.792, -87.409),
                    (-0.743, -179.786),
                ],
                vec![
                    (-0.284, 176.662),
                    (-11.827, -88.625),
                    (-11.825, -87.942),
                    (-0.737, 179.938),
                ],
                vec![
                    (-0.281, 176.114),
                    (-11.838, -88.888),
                    (-11.841, -88.009),
                    (-0.737, 179.650),
                ],
                vec![
                    (-0.284, 175.439),
                    (-11.849, -89.096),
                    (-11.828, -88.048),
                    (-0.699, 179.298),
                ],
                vec![
                    (-0.315, 174.679),
                    (-11.814, -89.502),
                    (-11.828, -88.608),
                    (-0.682, 178.625),
                ],
                vec![
                    (-0.346, 174.031),
                    (-11.835, -89.972),
                    (-11.824, -88.968),
                    (-0.720, 178.373),
                ],
                vec![
                    (-0.338, 173.956),
                    (-11.827, -90.109),
                    (-11.831, -89.305),
                    (-0.713, 178.598),
                ],
                vec![
                    (-0.273, 173.583),
                    (-11.855, -90.206),
                    (-11.862, -89.369),
                    (-0.648, 178.595),
                ],
                vec![
                    (-0.266, 173.005),
                    (-11.846, -90.457),
                    (-11.845, -89.570),
                    (-0.609, 178.294),
                ],
                vec![
                    (-0.232, 172.456),
                    (-11.843, -90.902),
                    (-11.855, -90.128),
                    (-0.612, 177.855),
                ],
                vec![
                    (-0.223, 172.030),
                    (-11.847, -91.056),
                    (-11.850, -90.397),
                    (-0.575, 177.768),
                ],
                vec![
                    (-0.227, 171.491),
                    (-11.866, -91.933),
                    (-11.857, -91.056),
                    (-0.561, 177.593),
                ],
                vec![
                    (-0.211, 170.829),
                    (-11.908, -92.002),
                    (-11.879, -91.092),
                    (-0.489, 177.305),
                ],
                vec![
                    (-0.233, 170.119),
                    (-11.866, -92.554),
                    (-11.893, -91.468),
                    (-0.476, 176.579),
                ],
                vec![
                    (-0.204, 169.606),
                    (-11.915, -92.825),
                    (-11.903, -92.017),
                    (-0.448, 176.373),
                ],
                vec![
                    (-0.235, 168.777),
                    (-11.933, -93.385),
                    (-11.912, -92.605),
                    (-0.420, 175.689),
                ],
                vec![
                    (-0.271, 167.832),
                    (-11.947, -93.891),
                    (-11.939, -93.206),
                    (-0.472, 175.246),
                ],
                vec![
                    (-0.306, 167.102),
                    (-11.957, -94.149),
                    (-11.954, -93.210),
                    (-0.448, 175.324),
                ],
                vec![
                    (-0.258, 166.964),
                    (-11.928, -94.426),
                    (-11.944, -93.481),
                    (-0.419, 175.523),
                ],
                vec![
                    (-0.229, 165.980),
                    (-11.950, -94.223),
                    (-11.941, -93.532),
                    (-0.339, 175.625),
                ],
                vec![
                    (-0.213, 165.242),
                    (-11.921, -94.718),
                    (-11.931, -93.936),
                    (-0.281, 175.172),
                ],
                vec![
                    (-0.198, 164.306),
                    (-11.915, -95.109),
                    (-11.905, -94.295),
                    (-0.217, 174.767),
                ],
                vec![
                    (-0.188, 163.445),
                    (-11.865, -95.670),
                    (-11.879, -94.820),
                    (-0.205, 174.577),
                ],
                vec![
                    (-0.181, 162.394),
                    (-11.874, -96.178),
                    (-11.869, -95.263),
                    (-0.135, 174.270),
                ],
                vec![
                    (-0.234, 161.079),
                    (-11.820, -96.781),
                    (-11.834, -95.848),
                    (-0.079, 173.670),
                ],
                vec![
                    (-0.311, 159.862),
                    (-11.797, -97.531),
                    (-11.792, -96.706),
                    (-0.026, 173.252),
                ],
                vec![
                    (-0.473, 158.035),
                    (-11.767, -99.069),
                    (-11.777, -98.139),
                    (0.031, 172.884),
                ],
                vec![
                    (-0.865, 156.756),
                    (-11.916, -100.603),
                    (-11.916, -99.695),
                    (0.055, 172.980),
                ],
                vec![
                    (-1.383, 157.170),
                    (-12.183, -101.839),
                    (-12.185, -100.941),
                    (0.148, 172.980),
                ],
                vec![
                    (-1.786, 159.984),
                    (-12.561, -101.601),
                    (-12.577, -100.705),
                    (0.301, 173.255),
                ],
                vec![
                    (-1.716, 163.563),
                    (-12.816, -99.966),
                    (-12.805, -98.952),
                    (0.502, 172.921),
                ],
                vec![
                    (-1.393, 165.231),
                    (-12.859, -97.814),
                    (-12.853, -96.821),
                    (0.657, 171.744),
                ],
                vec![
                    (-1.030, 165.580),
                    (-12.700, -96.201),
                    (-12.669, -95.453),
                    (0.747, 170.716),
                ],
                vec![
                    (-0.759, 164.875),
                    (-12.514, -95.622),
                    (-12.508, -94.678),
                    (0.780, 169.512),
                ],
                vec![
                    (-0.585, 164.044),
                    (-12.361, -95.610),
                    (-12.335, -94.626),
                    (0.755, 168.895),
                ],
                vec![
                    (-0.439, 163.431),
                    (-12.203, -95.395),
                    (-12.209, -94.420),
                    (0.799, 168.418),
                ],
                vec![
                    (-0.318, 162.847),
                    (-12.067, -95.224),
                    (-12.068, -94.311),
                    (0.841, 167.973),
                ],
            ],
            String::from("test_3"),
            String::from(
                "File name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Base file name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642\n\
            Date = 10/8/2019\n\
            Time = 7:48 AM\n\
            Test duration(min) = 0.15\n\
            Test = S-Parameters\n\
            MPTS version = 8.91.5\n\
            Cal File =\n\
            ComputerName = CMPE-LAB-6\n\
            Comments =\n\
            Z0 = 50\n\
            Input Power(dBm) = -13.00\n\
            Device ID = EG0520F_1010_6x25_2SDBCB2EV\n\
            Row = 66\n\
            Column = 42\n\
            Technology = QPHT09BCB\n\
            Lot/Wafer = 191593406\n\
            Gate Size = 150\n\
            Mask = EG0520\n\
            Cal Standards = EG2306 GaAs\n\
            Cal Kit = EG2308\n\
            Job Number = 20191002105035\n\
            Device Type = FET\n\
            set_Vg(v) = -1.500\n\
            set_Vd(v) = 0.000\n\
            file_name = EG0520F_1010_6x25_2SDBCB2EV_191593406_RC6642_Vg_N1p5_Vd_0.s2p\n\
            Vg(V) = -1.500\n\
            Vg(mA) = -0.000\n\
            Vd(V) = 0.000\n\
            Vd(mA) = 0.002\n",
            ),
        );
        debug_assert!(
            exemplar.name() == net.name(),
            " Failed name\n  exemplar: {}\n       net: {}",
            exemplar.name(),
            net.name()
        );
        comp_line(exemplar.comments(), net.comments(), "comments(test_3.s2p)");
        debug_assert!(
            exemplar.nports() == net.nports(),
            " Failed nports\n  exemplar: {}\n       net: {}",
            exemplar.nports(),
            net.nports()
        );
        debug_assert!(
            exemplar.npts() == net.npts(),
            " Failed npts\n  exemplar: {}\n       net: {}",
            exemplar.npts(),
            net.npts()
        );
        for i in 0..exemplar.nports() {
            debug_assert_eq!(
                exemplar.z0()[i],
                net.z0()[i],
                " Failed z0 for port {}\n  exemplar: {}\n       net: {}",
                i,
                exemplar.z0()[i],
                net.z0()[i]
            );
        }
        for i in 0..exemplar.npts() {
            assert_eq!(exemplar.freq().freq(i), net.freq().freq(i));
        }
        comp_points_c64(
            &exemplar.net(RFParameter::S),
            &net.net(RFParameter::S),
            F64Margin::default(),
            "net(test_3.s2p)",
        );
    }

    #[test]
    fn read_touchstone_s3p() {
        let filename = "./data/test.s3p".to_string();
        let net = read_touchstone(&filename).unwrap();
        let exemplar = Network::new(
            FrequencyBuilder::new()
                .freqs_scaled(array![330.0, 330.85, 331.7, 332.55], Scale::Giga)
                .build(),
            array![c64(50.0, 0.0), c64(50.0, 0.0), c64(50.0, 0.0)],
            RFParameter::S,
            Points::new(array![
                [
                    [
                        c64(-0.333333333333, 0.0),
                        c64(0.666666666667, 0.0),
                        c64(0.666666666667, 0.0)
                    ],
                    [
                        c64(0.666666666667, 0.0),
                        c64(-0.333333333333, 0.0),
                        c64(0.666666666667, 0.0)
                    ],
                    [
                        c64(0.666666666667, 0.0),
                        c64(0.666666666667, 0.0),
                        c64(-0.333333333333, 0.0)
                    ],
                ],
                [
                    [
                        c64(-0.333333333333, 0.0),
                        c64(0.666666666667, 0.0),
                        c64(0.666666666667, 0.0)
                    ],
                    [
                        c64(0.666666666667, 0.0),
                        c64(-0.333333333333, 0.0),
                        c64(0.666666666667, 0.0)
                    ],
                    [
                        c64(0.666666666667, 0.0),
                        c64(0.666666666667, 0.0),
                        c64(-0.333333333333, 0.0)
                    ],
                ],
                [
                    [
                        c64(-0.333333333333, 0.0),
                        c64(0.666666666667, 0.0),
                        c64(0.666666666667, 0.0)
                    ],
                    [
                        c64(0.666666666667, 0.0),
                        c64(-0.333333333333, 0.0),
                        c64(0.666666666667, 0.0)
                    ],
                    [
                        c64(0.666666666667, 0.0),
                        c64(0.666666666667, 0.0),
                        c64(-0.333333333333, 0.0)
                    ],
                ],
                [
                    [
                        c64(-0.333333333333, 0.0),
                        c64(0.666666666667, 0.0),
                        c64(0.666666666667, 0.0)
                    ],
                    [
                        c64(0.666666666667, 0.0),
                        c64(-0.333333333333, 0.0),
                        c64(0.666666666667, 0.0)
                    ],
                    [
                        c64(0.666666666667, 0.0),
                        c64(0.666666666667, 0.0),
                        c64(-0.333333333333, 0.0)
                    ],
                ],
            ]),
            String::from("test"),
            String::from("Created with skrf (http://scikit-rf.org)."),
        );
        debug_assert!(
            exemplar.name() == net.name(),
            " Failed name\n  exemplar: {}\n       net: {}",
            exemplar.name(),
            net.name()
        );
        comp_line(exemplar.comments(), net.comments(), "comments(test.s3p)");
        debug_assert!(
            exemplar.nports() == net.nports(),
            " Failed nports\n  exemplar: {}\n       net: {}",
            exemplar.nports(),
            net.nports()
        );
        debug_assert!(
            exemplar.npts() == net.npts(),
            " Failed npts\n  exemplar: {}\n       net: {}",
            exemplar.npts(),
            net.npts()
        );
        for i in 0..exemplar.nports() {
            debug_assert_eq!(
                exemplar.z0()[i],
                net.z0()[i],
                " Failed z0 for port {}\n  exemplar: {}\n       net: {}",
                i,
                exemplar.z0()[i],
                net.z0()[i]
            );
        }
        for i in 0..exemplar.npts() {
            assert_eq!(exemplar.freq().freq(i), net.freq().freq(i));
        }
        comp_points_c64(
            &exemplar.net(RFParameter::S),
            &net.net(RFParameter::S),
            F64Margin::default(),
            "net(S)",
        );
    }
}
