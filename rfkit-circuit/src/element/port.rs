use super::*;
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use rfkit_base::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Port<T: RealScalar> {
    id: String,
    z: Complex<T>,
    nodes: [usize; 1],
    c: Points<Complex<T>, Ix2>,
}

impl<T: RealScalar> Port<T> {
    pub fn new(id: &str, z: Complex<T>, nodes: [usize; 1]) -> Port<T> {
        Port {
            id: id.to_string(),
            z,
            nodes,
            c: Port::default_c(),
        }
    }

    pub fn builder() -> PortBuilder<T> {
        PortBuilder::new()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn z0(&self) -> Complex<T> {
        self.z
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    fn default_c() -> Points<Complex<T>, Ix2> {
        points![[Complex::ZERO]]
    }
}

impl<T: RealScalar> Default for Port<T> {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            z: Complex::ZERO,
            nodes: [1],
            c: Port::default_c(),
        }
    }
}

impl<T: RealScalar> Elem<T> for Port<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Port
    }

    fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }

    fn c<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points(
            self.c
                .view()
                .insert_axis(Axis(0))
                .broadcast((freq.npts(), self.c.nrows(), self.c.ncols()))
                .unwrap()
                .to_owned(),
        )
    }

    fn net<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        Points::zeros((freq.npts(), 1, 1).into_dimension())
    }

    fn z_scalar(&self, _freq: &ScalarUnitValue<T>) -> Complex<T> {
        self.z
    }
}

impl<T: RealScalar> Term<T> for Port<T> {
    fn val(&self) -> Complex<T> {
        self.z
    }

    fn set_val(&mut self, val: Complex<T>) {
        self.z = val;
    }
}

pub type PortBuilder<T> = ElementBuilder<T, PortSpec, ConcreteElement, 1>;
pub type PortElementBuilder<T> = ElementBuilder<T, PortSpec, TopLevelElement, 1>;

#[derive(Clone, Copy, Debug, Default)]
pub struct PortSpec;

#[derive(Clone, Debug)]
pub struct PortParams<T: RealScalar> {
    z: Option<Complex<T>>,
}

impl<T: RealScalar> Default for PortParams<T> {
    fn default() -> Self {
        Self { z: None }
    }
}

impl<T: RealScalar> ElementSpec<T, 1> for PortSpec {
    type Params = PortParams<T>;
    type Concrete = Port<T>;

    const NAME: &'static str = "PortBuilder";
    const DEFAULT_ID: &'static str = "C0";

    fn build_concrete(
        id: String,
        params: Self::Params,
        nodes: [usize; 1],
        _z0: Complex<T>,
    ) -> Result<Self::Concrete, String> {
        let z = params
            .z
            .ok_or_else(|| format!("{}: z is required", <Self as ElementSpec<T, 1>>::NAME))?;

        Ok(Port {
            id,
            z,
            nodes,
            c: Port::default_c(),
        })
    }
}

impl<T, M> ElementBuilder<T, PortSpec, M, 1>
where
    T: RealScalar,
    M: ElementBuildMode<T, Port<T>>,
{
    pub fn z(mut self, z: Complex<T>) -> Self {
        self.params.z = Some(z);
        self
    }
}

#[cfg(test)]
mod element_port_tests {
    use super::*;
    use num_complex::{Complex64, c64};

    const DEFAULT_MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-10,
        relative: 1e-10,
        ulps: 4,
    };

    const RELAXED_MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-6,
        relative: 1e-6,
        ulps: 10,
    };

    #[test]
    fn element_port() {
        let freq_unitval = ScalarUnitValue::new_scaled(&1.0, Scale::Giga, Unit::Hz);
        let freq = ArrayUnitValue::new(&array![freq_unitval.val()], freq_unitval.scale(), Unit::Hz);
        let val = c64(50.0, 0.0);
        let exemplar = Port {
            id: "P1".to_string(),
            z: val,
            nodes: [1],
            c: Port::default_c(),
        };
        let exemplar_z = val;
        let calc = PortBuilder::new()
            .z(val)
            .nodes([1])
            .id("P1")
            .build()
            .unwrap();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        comp_pts_ix3(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        exemplar_z.assert_approx_eq(&calc.z(&freq_unitval), margin, "calc.z()", "0");
    }

    mod port_tests {
        use super::*;

        #[test]
        fn test_port_builder_default() {
            let port: Port<f64> = PortBuilder::new()
                .z(Complex64::ZERO)
                .nodes([1])
                .build()
                .unwrap();
            assert_eq!(port.id(), "C0"); // Default ID from code
            assert_eq!(port.nodes(), vec![1]);
        }

        #[test]
        fn test_port_custom_impedance() {
            let impedances = vec![
                c64(50.0, 0.0),
                c64(75.0, 0.0),
                c64(100.0, 0.0),
                c64(50.0, 10.0), // With imaginary part
            ];

            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);

            for z_val in impedances {
                let port = PortBuilder::new().z(z_val).nodes([1]).build().unwrap();

                let z = port.z(&freq);
                assert_eq!(z[0], z_val);
            }
        }

        #[test]
        fn test_port_frequency_independent() {
            let freqs = array![1e6, 1e9, 10e9, 100e9];
            let freq = ArrayUnitValue::new(&freqs, Scale::Base, Unit::Hz);
            let port = PortBuilder::new()
                .z(c64(50.0, 0.0))
                .nodes([1])
                .build()
                .unwrap();

            for i in 0..freqs.len() {
                let z = port.z(&freq)[i];
                assert_eq!(z, c64(50.0, 0.0));
            }
        }

        #[test]
        fn test_port_node_assignment() {
            let nodes = vec![0, 1, 5, 10, 100];

            for node in nodes {
                let port: Port<f64> = PortBuilder::new()
                    .z(Complex64::ZERO)
                    .nodes([node])
                    .build()
                    .unwrap();
                assert_eq!(port.nodes(), vec![node]);
            }
        }

        #[test]
        fn test_port_c_matrix() {
            let freq = ArrayUnitValue::new(&array![1e9], Scale::Base, Unit::Hz);
            let port: Port<f64> = PortBuilder::new()
                .z(Complex64::ZERO)
                .nodes([1])
                .build()
                .unwrap();
            let c_matrix = port.c(&freq);

            assert_eq!(c_matrix[[0, 0, 0]], Complex64::ZERO);
        }

        #[test]
        fn test_port_elem_type() {
            let port = PortBuilder::<f64>::new()
                .z(Complex64::ZERO)
                .nodes([1])
                .build()
                .unwrap();
            assert_eq!(port.elem(), ElemType::Port);
        }

        #[test]
        fn test_port_term_trait() {
            let mut port = PortBuilder::new()
                .z(c64(50.0, 0.0))
                .nodes([1])
                .build()
                .unwrap();

            assert_eq!(port.val(), c64(50.0, 0.0));

            port.set_val(c64(75.0, 5.0));
            assert_eq!(port.val(), c64(75.0, 5.0));
        }

        #[test]
        fn test_port_id_setting() {
            let port: Port<f64> = PortBuilder::new()
                .id("P1")
                .z(Complex64::ZERO)
                .nodes([1])
                .build()
                .unwrap();
            assert_eq!(port.id(), "P1");
        }
    }
}
