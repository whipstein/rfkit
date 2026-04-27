use super::*;
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use rfkit_base::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Short<T: RealScalar> {
    id: String,
    nodes: [usize; 2],
    c: Points<Complex<T>, Ix2>,
    z0: Complex<T>,
}

impl<T: RealScalar> Short<T> {
    pub fn new(id: &str, nodes: [usize; 2], z0: Complex<T>) -> Self {
        Self {
            id: id.to_string(),
            nodes,
            c: Short::default_c(),
            z0: z0,
        }
    }

    pub fn builder() -> ShortBuilder<T> {
        ShortBuilder::new()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn z0(&self) -> Complex<T> {
        self.z0
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    fn default_c() -> Points<Complex<T>, Ix2> {
        points![[Complex::ZERO, Complex::ONE], [Complex::ONE, Complex::ZERO]]
    }
}

impl<T: RealScalar> Default for Short<T> {
    fn default() -> Self {
        Self {
            id: "S0".to_string(),
            nodes: [1, 2],
            c: Short::default_c(),
            z0: Complex::new(50.0.into(), T::ZERO),
        }
    }
}

impl<T: RealScalar> Elem<T> for Short<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Short
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
        Points::from_shape_fn((freq.npts(), 2, 2).into_dimension(), |(_, j, k)| {
            match (j, k) {
                (0, 0) | (1, 1) => Complex::ZERO,
                (1, 0) | (0, 1) => Complex::ONE,
                _ => Complex::ZERO,
            }
        })
    }

    fn z_scalar(&self, _freq: &ScalarUnitValue<T>) -> Complex<T> {
        Complex::new(1e-10.into(), T::ZERO)
    }
}

impl<T: RealScalar> Lumped<T> for Short<T> {
    fn val(&self) -> T {
        T::ZERO
    }

    fn val_scaled(&self) -> T {
        T::ZERO
    }

    fn scale(&self) -> Scale {
        Scale::Base
    }

    fn unit(&self) -> Unit {
        Unit::Ohm
    }

    fn set_val(&mut self, _val: T) {
        ();
    }

    fn set_val_scaled(&mut self, _val: T) {
        ();
    }

    fn set_scale(&mut self, _scale: Scale) {
        ();
    }
}

pub type ShortBuilder<T> = ElementBuilder<T, ShortSpec, ConcreteElement, 2>;
pub type ShortElementBuilder<T> = ElementBuilder<T, ShortSpec, TopLevelElement, 2>;

#[derive(Clone, Copy, Debug, Default)]
pub struct ShortSpec;

impl<T: RealScalar> ElementSpec<T, 2> for ShortSpec {
    type Params = ();
    type Concrete = Short<T>;

    const NAME: &'static str = "ShortBuilder";
    const DEFAULT_ID: &'static str = "S0";

    fn build_concrete(
        id: String,
        _params: Self::Params,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Result<Self::Concrete, String> {
        Ok(Short {
            id,
            nodes,
            c: Short::default_c(),
            z0,
        })
    }
}

impl<T: RealScalar, M> ElementBuilder<T, ShortSpec, M, 2> {
    pub fn z0(mut self, z0: Complex<T>) -> Self {
        self.z0 = z0;
        self
    }
}
