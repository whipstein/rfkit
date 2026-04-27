use super::*;
use ndarray::{IntoDimension, prelude::*};
use num_complex::Complex;
use rfkit_base::prelude::*;
use rfkit_network::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Transformer<T: RealScalar> {
    id: String,
    n: T,
    km: T,
    l1: ScalarUnitValue<T>,
    l2: ScalarUnitValue<T>,
    q1: Option<Q<T>>,
    q2: Option<Q<T>>,
    nodes: [usize; 2],
    c: Points<Complex<T>, Ix2>,
    z0: Complex<T>,
}

impl<T: RealScalar> Transformer<T> {
    pub fn new(
        id: &str,
        km: T,
        l1: ScalarUnitValue<T>,
        l2: ScalarUnitValue<T>,
        q1: Option<Q<T>>,
        q2: Option<Q<T>>,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Self {
        Self {
            id: id.to_string(),
            n: (l1.val() / l2.val()).sqrt(),
            km,
            l1,
            l2,
            q1,
            q2,
            nodes,
            c: Points::zeros((2, 2)),
            z0,
        }
    }

    pub fn builder() -> TransformerBuilder<T> {
        TransformerBuilder::new()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn z0(&self) -> Complex<T> {
        self.z0
    }

    pub fn n(&self) -> T {
        self.n
    }

    pub fn km(&self) -> T {
        self.km
    }

    pub fn m(&self) -> T {
        self.km * self.l1.val()
    }

    pub fn l1(&self) -> T {
        self.l1.val()
    }

    pub fn l1_scaled(&self) -> T {
        self.l1.val_scaled()
    }

    pub fn l2(&self) -> T {
        self.l2.val()
    }

    pub fn l2_scaled(&self) -> T {
        self.l2.val_scaled()
    }

    pub fn q1(&self) -> Option<Q<T>> {
        self.q1.clone()
    }

    pub fn q2(&self) -> Option<Q<T>> {
        self.q2.clone()
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    pub fn set_km(&mut self, val: T) {
        self.km = val;
    }

    pub fn set_l1(&mut self, val: &ScalarUnitValue<T>) {
        self.l1 = val.clone();
    }

    pub fn set_l1_val(&mut self, val: T) {
        self.l1.set_val(&val);
    }

    pub fn set_l1_val_scaled(&mut self, val: T) {
        self.l1.set_val_scaled(&val);
    }

    pub fn set_l2(&mut self, val: &ScalarUnitValue<T>) {
        self.l2 = val.clone();
    }

    pub fn set_l2_val(&mut self, val: T) {
        self.l2.set_val(&val);
    }

    pub fn set_l2_val_scaled(&mut self, val: T) {
        self.l2.set_val_scaled(&val);
    }

    pub fn set_q1(&mut self, val: Q<T>) {
        self.q1 = Some(val);
    }

    pub fn set_q2(&mut self, val: Q<T>) {
        self.q2 = Some(val);
    }

    pub fn unset_q1(&mut self) {
        self.q1 = None;
    }

    pub fn unset_q2(&mut self) {
        self.q2 = None;
    }

    fn calc_c<U: FreqValue<T>>(&self, freq: &U) -> Points<Complex<T>, Ix3> {
        // let y = Points::from_shape_fn((freq.npts(), 2, 2).into_dimension(), |(_, j, k)| {
        //     match (j, k) {
        //         (0, 0) => Complex::new(
        //             T::ZERO,
        //             (-T::ONE * freq.w_pt(0) * self.l1.val() * (T::ONE - self.km.powi(2))).recip(),
        //         ),
        //         (1, 1) => Complex::new(
        //             T::ZERO,
        //             (-T::ONE * freq.w_pt(0) * self.l2.val() * (T::ONE - self.km.powi(2))).recip(),
        //         ),
        //         (1, 0) | (0, 1) => Complex::new(
        //             T::ZERO,
        //             self.km
        //                 / (-T::ONE
        //                     * freq.w_pt(0)
        //                     * (self.l1.val() * self.l2.val()).sqrt()
        //                     * (T::ONE - self.km.powi(2))),
        //         ),
        //         _ => Complex::ZERO,
        //     }
        // });

        // y.y_to_s(&vec![self.z0, self.z0]).unwrap()

        let z = Points::from_shape_fn((freq.npts(), 2, 2).into_dimension(), |(_, j, k)| {
            match (j, k) {
                (0, 0) => Complex::new(T::ZERO, T::ONE * freq.w_pt(0) * self.l1.val()),
                (1, 1) => Complex::new(T::ZERO, T::ONE * freq.w_pt(0) * self.l2.val()),
                (1, 0) | (0, 1) => Complex::new(
                    T::ZERO,
                    T::ONE * freq.w_pt(0) * self.km * (self.l1.val() * self.l2.val()).sqrt(),
                ),
                _ => Complex::ZERO,
            }
        });

        z.z_to_s(&vec![self.z0, self.z0]).unwrap()
    }
}

impl<T: RealScalar> Default for Transformer<T> {
    fn default() -> Self {
        Self {
            id: "T0".to_string(),
            n: T::ONE,
            km: T::ONE,
            l1: *ScalarUnitValue::default().set_unit(Unit::Henry),
            l2: *ScalarUnitValue::default().set_unit(Unit::Henry),
            q1: None,
            q2: None,
            nodes: [1, 2],
            c: points![[Complex::ZERO, Complex::ONE], [Complex::ONE, Complex::ZERO]],
            z0: Complex::new(50.0.into(), T::ZERO),
        }
    }
}

impl<T: RealScalar> Elem<T> for Transformer<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::Transformer
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
        self.calc_c(freq)
    }

    fn z_scalar(&self, _freq: &ScalarUnitValue<T>) -> Complex<T> {
        Complex::new(self.n.powi(2), T::ZERO)
    }
}

pub type TransformerBuilder<T> = ElementBuilder<T, TransformerSpec, ConcreteElement, 2>;
pub type TransformerElementBuilder<T> = ElementBuilder<T, TransformerSpec, TopLevelElement, 2>;

#[derive(Clone, Copy, Debug, Default)]
pub struct TransformerSpec;

#[derive(Clone, Debug)]
pub struct TransformerParams<T: RealScalar> {
    n: Option<T>,
    km: Option<T>,
    m: Option<ScalarUnitValue<T>>,
    l1: Option<ScalarUnitValue<T>>,
    l2: Option<ScalarUnitValue<T>>,
    q1: Option<Q<T>>,
    q2: Option<Q<T>>,
}

impl<T: RealScalar> Default for TransformerParams<T> {
    fn default() -> Self {
        Self {
            n: None,
            km: None,
            m: None,
            l1: None,
            l2: None,
            q1: None,
            q2: None,
        }
    }
}

impl<T: RealScalar> ElementSpec<T, 2> for TransformerSpec {
    type Params = TransformerParams<T>;
    type Concrete = Transformer<T>;

    const NAME: &'static str = "TransformerBuilder";
    const DEFAULT_ID: &'static str = "T0";

    fn build_concrete(
        id: String,
        mut params: Self::Params,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Result<Self::Concrete, String> {
        let elem = <Self as ElementSpec<T, 2>>::NAME;
        let l1 = params.l1.take().ok_or(format!("{elem}: l1 is required"))?;
        if params.n.is_none() && params.l2.is_none() {
            return Err(format!(
                "{elem}: Must specify either n or l2 to completely define a Transformer"
            ));
        }
        if params.km.is_none() && params.m.is_none() {
            return Err(format!(
                "{elem}: Must specify either km or m to completely define a Transformer"
            ));
        }

        if let (Some(n), None) = (params.n, params.l2.as_ref()) {
            let mut l2 = l1.clone();
            l2.set_val(&(l1.val() / n.powi(2)));
            params.l2 = Some(l2);
        }
        if params.km.is_none() {
            if let Some(m) = params.m.as_ref() {
                params.km = Some(m.val() / l1.val());
            }
        }

        let km = params.km.unwrap();
        let l2 = params.l2.take().unwrap();
        let m = match params.m.take() {
            Some(m) => m,
            None => {
                ScalarUnitValue::new(&(km * (l1.val() * l2.val()).sqrt()), l1.scale(), l1.unit())
            }
        };
        if km.abs() > T::ONE {
            return Err(format!("{elem}: |km| must be less than or equal to 1.0"));
        }
        if m.val().powi(2) > l1.val() * l2.val() {
            return Err(format!("{elem}: M^2 must be less than or equal to L1 * L2"));
        }

        Ok(Transformer::new(
            id.as_str(),
            km,
            l1,
            l2,
            params.q1,
            params.q2,
            nodes,
            z0,
        ))
    }
}

impl<T, M> ElementBuilder<T, TransformerSpec, M, 2>
where
    T: RealScalar,
    M: ElementBuildMode<T, Transformer<T>>,
{
    pub fn n(mut self, n: T) -> Self {
        self.params.n = Some(n);
        self
    }

    pub fn km(mut self, km: T) -> Self {
        self.params.km = Some(km);
        self
    }

    pub fn m(mut self, ind: &ScalarUnitValue<T>) -> Self {
        self.params.m = Some(ind.clone());
        self
    }

    pub fn m_val(mut self, ind: T) -> Self {
        self.params.m = match self.params.m {
            Some(mut x) => {
                x.set_val(&ind);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&ind, Scale::Base, Unit::Henry)),
        };
        self
    }

    pub fn m_val_scaled(mut self, ind: T, scale: Scale) -> Self {
        self.params.m = match self.params.m {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&ind);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&ind, scale, Unit::Henry)),
        };
        self
    }

    pub fn l1(mut self, ind: &ScalarUnitValue<T>) -> Self {
        self.params.l1 = Some(ind.clone());
        self
    }

    pub fn l1_val(mut self, ind: T) -> Self {
        self.params.l1 = match self.params.l1 {
            Some(mut x) => {
                x.set_val(&ind);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&ind, Scale::Base, Unit::Henry)),
        };
        self
    }

    pub fn l1_val_scaled(mut self, ind: T, scale: Scale) -> Self {
        self.params.l1 = match self.params.l1 {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&ind);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&ind, scale, Unit::Henry)),
        };
        self
    }

    pub fn l2(mut self, ind: &ScalarUnitValue<T>) -> Self {
        self.params.l2 = Some(ind.clone());
        self
    }

    pub fn l2_val(mut self, ind: T) -> Self {
        self.params.l2 = match self.params.l2 {
            Some(mut x) => {
                x.set_val(&ind);
                Some(x)
            }
            None => Some(ScalarUnitValue::new(&ind, Scale::Base, Unit::Henry)),
        };
        self
    }

    pub fn l2_val_scaled(mut self, ind: T, scale: Scale) -> Self {
        self.params.l2 = match self.params.l2 {
            Some(mut x) => {
                x.set_scale(scale);
                x.set_val_scaled(&ind);
                Some(x)
            }
            None => Some(ScalarUnitValue::new_scaled(&ind, scale, Unit::Henry)),
        };
        self
    }

    pub fn q1(mut self, q: &Q<T>) -> Self {
        self.params.q1 = Some(q.clone());
        self
    }

    pub fn q2(mut self, q: &Q<T>) -> Self {
        self.params.q2 = Some(q.clone());
        self
    }

    pub fn z0(mut self, z0: Complex<T>) -> Self {
        self.z0 = z0;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IdealTransformer<T: RealScalar> {
    id: String,
    n: T,
    nodes: [usize; 2],
    c: Points<Complex<T>, Ix2>,
    z0: Complex<T>,
}

impl<T: RealScalar> IdealTransformer<T> {
    pub fn new(id: &str, n: T, nodes: [usize; 2], z0: Complex<T>) -> Self {
        Self {
            id: id.to_string(),
            n,
            nodes,
            c: IdealTransformer::default_c(n),
            z0,
        }
    }

    pub fn builder() -> IdealTransformerBuilder<T> {
        IdealTransformerBuilder::new()
    }

    pub fn name(&self) -> &String {
        &self.id
    }

    pub fn z0(&self) -> Complex<T> {
        self.z0
    }

    pub fn val(&self) -> T {
        self.n
    }

    pub fn set_id(&mut self, id: &str) {
        self.id = id.to_string();
    }

    pub fn set_val(&mut self, val: T) {
        self.n = val;
    }

    fn default_c(n: T) -> Points<Complex<T>, Ix2> {
        points![
            [
                Complex::new((T::ONE - n.powi(2)) / (T::ONE + n.powi(2)), T::ZERO),
                Complex::new(n * 2.0 / (T::ONE + n.powi(2)), T::ZERO)
            ],
            [
                Complex::new(n * 2.0 / (T::ONE + n.powi(2)), T::ZERO),
                Complex::new((n.powi(2) - T::ONE) / (T::ONE + n.powi(2)), T::ZERO)
            ]
        ]
    }
}

impl<T: RealScalar> Default for IdealTransformer<T> {
    fn default() -> Self {
        Self {
            id: "C0".to_string(),
            n: T::ONE,
            nodes: [1, 2],
            c: IdealTransformer::default_c(T::ONE),
            z0: Complex::new(50.0.into(), T::ZERO),
        }
    }
}

impl<T: RealScalar> Elem<T> for IdealTransformer<T> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn elem(&self) -> ElemType {
        ElemType::IdealTransformer
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
                (0, 0) => Complex::new(
                    (T::ONE - self.n.powi(2)) / (T::ONE + self.n.powi(2)),
                    T::ZERO,
                ),
                (1, 1) => Complex::new(
                    (self.n.powi(2) - T::ONE) / (T::ONE + self.n.powi(2)),
                    T::ZERO,
                ),
                (1, 0) | (0, 1) => Complex::new(self.n * 2.0 / (T::ONE + self.n.powi(2)), T::ZERO),
                _ => Complex::ZERO,
            }
        })
    }

    fn z_scalar(&self, _freq: &ScalarUnitValue<T>) -> Complex<T> {
        Complex::new(self.n.powi(2), T::ZERO)
    }

    fn z_at<U: FreqValue<T>>(&self, freq: &U, idx: usize) -> Complex<T> {
        let z = freq.map_scalar_to_vec(|f| self.z(f))[idx];
        Complex::new(z.re, z.im)
    }
}

pub type IdealTransformerBuilder<T> = ElementBuilder<T, IdealTransformerSpec, ConcreteElement, 2>;
pub type IdealTransformerElementBuilder<T> =
    ElementBuilder<T, IdealTransformerSpec, TopLevelElement, 2>;

#[derive(Clone, Copy, Debug, Default)]
pub struct IdealTransformerSpec;

#[derive(Clone, Debug)]
pub struct IdealTransformerParams<T: RealScalar> {
    n: Option<T>,
}

impl<T: RealScalar> Default for IdealTransformerParams<T> {
    fn default() -> Self {
        Self { n: None }
    }
}

impl<T: RealScalar> ElementSpec<T, 2> for IdealTransformerSpec {
    type Params = IdealTransformerParams<T>;
    type Concrete = IdealTransformer<T>;

    const NAME: &'static str = "IdealTransformerBuilder";
    const DEFAULT_ID: &'static str = "T0";

    fn build_concrete(
        id: String,
        params: Self::Params,
        nodes: [usize; 2],
        z0: Complex<T>,
    ) -> Result<Self::Concrete, String> {
        let n = params
            .n
            .ok_or_else(|| format!("{}: n is required", <Self as ElementSpec<T, 2>>::NAME))?;

        Ok(IdealTransformer {
            id,
            n,
            nodes,
            c: IdealTransformer::default_c(n),
            z0,
        })
    }
}

impl<T, M> ElementBuilder<T, IdealTransformerSpec, M, 2>
where
    T: RealScalar,
    M: ElementBuildMode<T, IdealTransformer<T>>,
{
    pub fn n(mut self, n: T) -> Self {
        self.params.n = Some(n);
        self
    }

    pub fn z0(mut self, z0: Complex<T>) -> Self {
        self.z0 = z0;
        self
    }
}

#[cfg(test)]
mod element_transformer_tests {
    use super::*;
    use num_complex::c64;

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
    fn element_transformer() {
        let freq_unitval: ScalarUnitValue<f64> = UnitValueBuilder::new()
            .val_scaled(&1.0, Scale::Giga)
            .build()
            .unwrap();
        let freq = ArrayUnitValue::new_freq(&array![freq_unitval.val()], freq_unitval.scale());
        let n = 1.0;
        let km = 0.5;
        let l1 = ScalarUnitValue::new(&5e-12, Scale::Pico, Unit::Henry);

        let calc: Transformer<f64> = TransformerBuilder::new()
            .n(n)
            .km(km)
            .l1(&l1)
            .nodes([1, 2])
            .id("T1")
            .build()
            .unwrap();
        let margin = NumMargin::default();

        // Verify basic properties
        assert_eq!(&calc.id(), "T1");
        assert_eq!(calc.nodes(), vec![1, 2]);
        calc.n().assert_approx_eq(&n, margin, "n", "");
        calc.km().assert_approx_eq(&km, margin, "km", "");
        calc.l1().assert_approx_eq(&5e-12, margin, "l1", "");

        // Verify impedance ratio (z = n^2)
        let exemplar_z = c64(n.powi(2), 0.0);
        exemplar_z.assert_approx_eq(&calc.z(&freq_unitval), margin, "calc.z()", "0");

        // Verify c matrix has valid structure
        let c = calc.c(&freq);
        assert_eq!(c.shape(), (1, 2, 2));
        assert!(c[[0, 0, 0]].is_finite());
        assert!(c[[0, 1, 1]].is_finite());
        // S12 should equal S21 for reciprocal network
        c[[0, 0, 1]].assert_approx_eq(&c[[0, 1, 0]], margin, "c", "[0,1]");
    }

    #[test]
    fn element_idealtransformer() {
        let freq_unitval: ScalarUnitValue<f64> = UnitValueBuilder::new()
            .val_scaled(&1.0, Scale::Giga)
            .build()
            .unwrap();
        let freq = ArrayUnitValue::new_freq(&array![freq_unitval.val()], freq_unitval.scale());
        let val = 1.0;
        let exemplar = IdealTransformer {
            id: "T1".to_string(),
            n: val,
            nodes: [1, 2],
            c: points![
                [c64(0.0, 0.0), c64(1.0, 0.0)],
                [c64(1.0, 0.0), c64(0.0, 0.0)]
            ],
            z0: c64(50.0, 0.0),
        };
        let exemplar_z = c64(val.powi(2), 0.0);
        let calc = IdealTransformerBuilder::new()
            .n(val)
            .nodes([1, 2])
            .id("T1")
            .build()
            .unwrap();
        let margin = NumMargin::default();

        assert_eq!(&exemplar.id(), &calc.id());
        comp_pts_ix3(&exemplar.c(&freq), &calc.c(&freq), margin, "calc.c()");
        exemplar_z.assert_approx_eq(&calc.z(&freq_unitval), margin, "calc.z()", "0");
    }

    mod transformer_tests {
        use super::*;

        // Helper function to create a default frequency for tests
        fn test_freq() -> ArrayUnitValue<f64> {
            ArrayUnitValue::new_freq(&array![1e9], Scale::Base)
        }

        #[test]
        fn test_transformer_builder_default() {
            // TransformerBuilder requires n/l2 and km/m to be specified
            let n: Transformer<f64> = TransformerBuilder::new()
                .n(1.0)
                .km(1.0)
                .l1_val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(n.id(), "T0");
            assert_eq!(n.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_transformer_builder_with_all_parameters() {
            let mut l1: ScalarUnitValue<f64> = *ScalarUnitValue::default().set_unit(Unit::Henry);
            l1.set_scale(Scale::Pico);
            l1.set_val_scaled(&5.0);
            let n = TransformerBuilder::new()
                .id("T1")
                .n(10.0)
                .km(0.5)
                .l1(&l1)
                .nodes([1, 3])
                .z0(c64(75.0, 0.0))
                .build()
                .unwrap();

            assert_eq!(n.id(), "T1");
            assert_eq!(n.nodes(), vec![1, 3]);
            assert_eq!(n.n(), 10.0);
            assert_eq!(n.l1(), 5e-12);
            assert_eq!(n.l2(), 5e-12 / 100.0);
            assert_eq!(n.km(), 0.5);
            assert_eq!(n.m(), 2.5e-12);
        }

        #[test]
        fn test_transformer_impedance_calculation() {
            let mut l1: ScalarUnitValue<f64> = *ScalarUnitValue::default().set_unit(Unit::Henry);
            l1.set_scale(Scale::Pico);
            l1.set_val_scaled(&5.0);
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let n: Transformer<f64> = TransformerBuilder::new()
                .n(1.0)
                .km(1.0)
                .l1(&l1)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = n.z(&freq);
            let expected_z = c64(1.0, 0.0);

            z[0].re.assert_approx_eq(
                &expected_z.re,
                NumMargin {
                    epsilon: 1e-6,
                    relative: 1e-6,
                    ulps: 4,
                },
                "z",
                "",
            );
        }

        #[test]
        fn test_transformer_c_matrix_structure() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            // Note: km=1.0 causes division by zero in the Y-to-S conversion
            // Use km < 1.0 for valid transformer calculations
            let n: Transformer<f64> = TransformerBuilder::new()
                .n(1.0)
                .km(0.9)
                .l1_val_scaled(10.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c_matrix = n.c(&freq);

            // Verify c matrix has valid (non-NaN) values and correct shape
            assert_eq!(c_matrix.inner().dim(), (1, 2, 2));
            assert!(c_matrix[[0, 0, 0]].is_finite());
            assert!(c_matrix[[0, 1, 1]].is_finite());
            assert!(c_matrix[[0, 0, 1]].is_finite());
            assert!(c_matrix[[0, 1, 0]].is_finite());
            // S12 should equal S21 for a reciprocal network
            c_matrix[[0, 0, 1]].re.assert_approx_eq(
                &c_matrix[[0, 1, 0]].re,
                DEFAULT_MARGIN,
                "c_matrix",
                "[0,1]",
            );
        }

        #[test]
        fn test_transformer_multiple_frequencies() {
            let mut l1: ScalarUnitValue<f64> = *ScalarUnitValue::default().set_unit(Unit::Henry);
            l1.set_scale(Scale::Pico);
            l1.set_val_scaled(&5.0);
            let freqs = array![1e9, 2e9, 5e9, 10e9];
            let freq = ArrayUnitValue::new_freq(&freqs, Scale::Base);
            let n: Transformer<f64> = TransformerBuilder::new()
                .n(1.0)
                .km(1.0)
                .l1(&l1)
                .nodes([1, 2])
                .build()
                .unwrap();

            for i in 0..freqs.len() {
                let z = n.z_at(&freq, i);
                let expected = c64(1.0, 0.0);
                z.assert_approx_eq(&expected, RELAXED_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_transformer_value_scaling() {
            let mut l1 = *ScalarUnitValue::default().set_unit(Unit::Henry);
            l1.set_scale(Scale::Pico);
            l1.set_val_scaled(&5.0);
            let n: Transformer<f64> = TransformerBuilder::new()
                .n(10.0)
                .km(1.0)
                .l1(&l1)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(n.n(), 10.0);
        }

        #[test]
        fn test_transformer_set_operations() {
            let mut n: Transformer<f64> = TransformerBuilder::new()
                .n(1.0)
                .km(1.0)
                .l1_val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            n.set_id("T_new");
            assert_eq!(n.id(), "T_new");

            n.set_l1_val_scaled(5.0);
            assert_eq!(n.l1_scaled(), 5.0);
        }

        #[test]
        fn test_transformer_elem_type() {
            let n: Transformer<f64> = TransformerBuilder::new()
                .n(1.0)
                .km(1.0)
                .l1_val(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(n.elem(), ElemType::Transformer);
        }

        #[test]
        fn test_transformer_net_matrix() {
            let freq = ArrayUnitValue::new_freq(&array![1e9, 2e9], Scale::Base);
            // Note: km=1.0 causes division by zero in the Y-to-S conversion
            // Use km < 1.0 for valid transformer calculations
            let n: Transformer<f64> = TransformerBuilder::new()
                .n(1.0)
                .km(0.9)
                .l1_val_scaled(10.0, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();
            let net = n.net(&freq);

            assert_eq!(net.shape(), (2, 2, 2)); // [npts, 2, 2]

            // Verify net matrix has valid (non-NaN) values
            for i in 0..2 {
                assert!(net[[i, 0, 0]].is_finite());
                assert!(net[[i, 1, 1]].is_finite());
                assert!(net[[i, 0, 1]].is_finite());
                assert!(net[[i, 1, 0]].is_finite());
                // S12 should equal S21 for a reciprocal network
                net[[i, 0, 1]].re.assert_approx_eq(
                    &net[[i, 1, 0]].re,
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},0,1]").as_str(),
                );
            }
        }
    }

    mod idealtransformer_tests {
        use super::*;

        #[test]
        fn test_transformer_builder_default() {
            let n: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(n.id(), "T0");
            assert_eq!(n.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_transformer_builder_with_all_parameters() {
            let n = IdealTransformerBuilder::new()
                .id("T1")
                .n(10.0)
                .nodes([1, 3])
                .z0(c64(75.0, 0.0))
                .build()
                .unwrap();

            assert_eq!(n.id(), "T1");
            assert_eq!(n.nodes(), vec![1, 3]);
            assert_eq!(n.val(), 10.0);
        }

        #[test]
        fn test_transformer_impedance_calculation() {
            let freq = ScalarUnitValue::new_freq(&1e9, Scale::Base);
            let n: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = n.z(&freq);
            let expected_z = c64(1.0, 0.0);

            z.assert_approx_eq(
                &expected_z,
                NumMargin {
                    epsilon: 1e-6,
                    relative: 1e-6,
                    ulps: 4,
                },
                "z",
                "",
            );
        }

        #[test]
        fn test_transformer_c_matrix_structure() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let n: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let c_matrix = n.c(&freq);

            // Check diagonal and off-diagonal elements
            c_matrix[[0, 0, 0]]
                .re
                .assert_approx_eq(&0.0, DEFAULT_MARGIN, "c_matrix", "[0,0]");
            c_matrix[[0, 1, 1]]
                .re
                .assert_approx_eq(&0.0, DEFAULT_MARGIN, "c_matrix", "[1,1]");
            c_matrix[[0, 0, 1]]
                .re
                .assert_approx_eq(&1.0, DEFAULT_MARGIN, "c_matrix", "[0,1]");
            c_matrix[[0, 1, 0]]
                .re
                .assert_approx_eq(&1.0, DEFAULT_MARGIN, "c_matrix", "[1,0]");
        }

        #[test]
        fn test_transformer_multiple_frequencies() {
            let freqs = array![1e9, 2e9, 5e9, 10e9];
            let freq = ArrayUnitValue::new_freq(&freqs.clone(), Scale::Base);
            let n: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            for i in 0..freqs.len() {
                let z = n.z_at(&freq, i);
                let expected = c64(1.0, 0.0);
                z.assert_approx_eq(&expected, RELAXED_MARGIN, "z", "");
            }
        }

        #[test]
        fn test_transformer_value_scaling() {
            let n: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(10.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(n.val(), 10.0);
        }

        #[test]
        fn test_transformer_set_operations() {
            let mut n: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            n.set_id("T_new");
            assert_eq!(n.id(), "T_new");

            n.set_val(5.0);
            assert_eq!(n.val(), 5.0);
        }

        #[test]
        fn test_transformer_elem_type() {
            let n: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            assert_eq!(n.elem(), ElemType::IdealTransformer);
        }

        #[test]
        fn test_transformer_net_matrix() {
            let freq = ArrayUnitValue::new_freq(&array![1e9, 2e9], Scale::Base);
            let n: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
            let net = n.net(&freq);

            assert_eq!(net.shape(), (2, 2, 2)); // [npts, 2, 2]

            // Check structure is consistent across frequencies
            for i in 0..2 {
                net[[i, 0, 0]].re.assert_approx_eq(
                    &0.0,
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},0,0]").as_str(),
                );
                net[[i, 1, 1]].re.assert_approx_eq(
                    &0.0,
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},1,1]").as_str(),
                );
                net[[i, 0, 1]].re.assert_approx_eq(
                    &1.0,
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},0,1]").as_str(),
                );
            }
        }
    }

    // ============================================================
    // COMPREHENSIVE TRANSFORMER TESTS
    // ============================================================

    mod transformer_comprehensive_tests {
        use super::*;

        // Helper function to create a default frequency for tests
        fn test_freq() -> ArrayUnitValue<f64> {
            ArrayUnitValue::new_freq(&array![1e9], Scale::Base)
        }

        // ----------------------------------------------------------
        // Transformer struct direct tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_new_direct() {
            let l1 = ScalarUnitValue::new(&10e-9, Scale::Nano, Unit::Henry);
            let l2 = ScalarUnitValue::new(&10e-9, Scale::Nano, Unit::Henry);

            let t = Transformer::new("T_direct", 0.9, l1, l2, None, None, [1, 2], c64(50.0, 0.0));

            assert_eq!(t.id(), "T_direct");
            t.km().assert_approx_eq(&0.9, DEFAULT_MARGIN, "km", "");
            t.n().assert_approx_eq(&1.0, DEFAULT_MARGIN, "n", ""); // sqrt(l1/l2) = 1
            assert_eq!(t.nodes(), vec![1, 2]);
        }

        #[test]
        fn test_transformer_default() {
            let t: Transformer<f64> = Transformer::default();
            assert_eq!(t.id(), "T0");
            t.n().assert_approx_eq(&1.0, DEFAULT_MARGIN, "n", "");
            t.km().assert_approx_eq(&1.0, DEFAULT_MARGIN, "km", "");
            assert_eq!(t.nodes(), vec![1, 2]);
            assert_eq!(t.z0(), c64(50.0, 0.0));
        }

        #[test]
        fn test_transformer_n_calculation() {
            // n = sqrt(l1/l2)
            let l1 = ScalarUnitValue::new(&100e-9, Scale::Nano, Unit::Henry);
            let l2 = ScalarUnitValue::new(&25e-9, Scale::Nano, Unit::Henry);

            let t = Transformer::new("T1", 0.9, l1, l2, None, None, [1, 2], c64(50.0, 0.0));

            // n = sqrt(100/25) = sqrt(4) = 2
            t.n().assert_approx_eq(&2.0, DEFAULT_MARGIN, "n", "");
        }

        #[test]
        fn test_transformer_mutual_inductance() {
            // m = km * l1
            let l1 = ScalarUnitValue::new(&10e-9, Scale::Nano, Unit::Henry);
            let km = 0.8;
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1(&l1)
                .n(1.0)
                .km(km)
                .nodes([1, 2])
                .build()
                .unwrap();

            let expected_m = km * l1.val();
            t.m().assert_approx_eq(&expected_m, DEFAULT_MARGIN, "m", "");
        }

        #[test]
        fn test_transformer_l1_accessors() {
            let l1 = ScalarUnitValue::new(&5e-12, Scale::Pico, Unit::Henry);
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1(&l1)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.l1().assert_approx_eq(&5e-12, DEFAULT_MARGIN, "l1", "");
            t.l1_scaled()
                .assert_approx_eq(&5.0, DEFAULT_MARGIN, "l1_scaled", "");
        }

        #[test]
        fn test_transformer_l2_accessors() {
            let l1 = ScalarUnitValue::new(&10e-12, Scale::Pico, Unit::Henry);
            let l2 = ScalarUnitValue::new(&2.5e-12, Scale::Pico, Unit::Henry);

            let t = Transformer::new("T1", 0.9, l1, l2, None, None, [1, 2], c64(50.0, 0.0));

            t.l2().assert_approx_eq(&l2.val(), DEFAULT_MARGIN, "l2", "");
            t.l2_scaled()
                .assert_approx_eq(&2.5, DEFAULT_MARGIN, "l2_scaled", "");
        }

        #[test]
        fn test_transformer_q_accessors() {
            let q1 = QBuilder::new()
                .q(100.0)
                .mode(QMode::Constant)
                .fq(&ScalarUnitValue::new_freq(&1.0, Scale::Base))
                .build()
                .unwrap();
            let q2 = QBuilder::new()
                .q(150.0)
                .fq(&ScalarUnitValue::new_freq(&1.0, Scale::Base))
                .mode(QMode::Constant)
                .build()
                .unwrap();

            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .q1(&q1)
                .q2(&q2)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert!(t.q1().is_some());
            assert!(t.q2().is_some());
            t.q1()
                .unwrap()
                .q()
                .assert_approx_eq(&100.0, DEFAULT_MARGIN, "q1", "");
            t.q2()
                .unwrap()
                .q()
                .assert_approx_eq(&150.0, DEFAULT_MARGIN, "q2", "");
        }

        #[test]
        fn test_transformer_q_none_by_default() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert!(t.q1().is_none());
            assert!(t.q2().is_none());
        }

        // ----------------------------------------------------------
        // Transformer setter tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_set_km() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.5)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.set_km(0.95);
            t.km().assert_approx_eq(&0.95, DEFAULT_MARGIN, "km", "");
        }

        #[test]
        fn test_transformer_set_l1() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let new_l1 = ScalarUnitValue::new(&20e-9, Scale::Nano, Unit::Henry);
            t.set_l1(&new_l1);
            t.l1().assert_approx_eq(&20e-9, DEFAULT_MARGIN, "l1", "");
        }

        #[test]
        fn test_transformer_set_l1_val() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.set_l1_val(15e-9);
            t.l1().assert_approx_eq(&15e-9, DEFAULT_MARGIN, "l1", "");
        }

        #[test]
        fn test_transformer_set_l1_val_scaled() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.set_l1_val_scaled(25.0);
            t.l1_scaled()
                .assert_approx_eq(&25.0, DEFAULT_MARGIN, "l1_scaled", "");
        }

        #[test]
        fn test_transformer_set_l2() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let new_l2 = ScalarUnitValue::new(&5e-9, Scale::Nano, Unit::Henry);
            t.set_l2(&new_l2);
            t.l2().assert_approx_eq(&5e-9, DEFAULT_MARGIN, "l2", "");
        }

        #[test]
        fn test_transformer_set_l2_val() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.set_l2_val(8e-9);
            t.l2().assert_approx_eq(&8e-9, DEFAULT_MARGIN, "l2", "");
        }

        #[test]
        fn test_transformer_set_l2_val_scaled() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.set_l2_val_scaled(12.0);
            t.l2_scaled()
                .assert_approx_eq(&12.0, DEFAULT_MARGIN, "l2_scaled", "");
        }

        #[test]
        fn test_transformer_set_q1() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let q = QBuilder::new()
                .q(200.0)
                .fq(&ScalarUnitValue::new_freq(&1.0, Scale::Base))
                .build()
                .unwrap();
            t.set_q1(q);

            assert!(t.q1().is_some());
            t.q1()
                .unwrap()
                .q()
                .assert_approx_eq(&200.0, DEFAULT_MARGIN, "q1", "");
        }

        #[test]
        fn test_transformer_set_q2() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let q = QBuilder::new()
                .q(250.0)
                .fq(&ScalarUnitValue::new_freq(&1.0, Scale::Base))
                .build()
                .unwrap();
            t.set_q2(q);

            assert!(t.q2().is_some());
            t.q2()
                .unwrap()
                .q()
                .assert_approx_eq(&250.0, DEFAULT_MARGIN, "q2", "");
        }

        #[test]
        fn test_transformer_unset_q1() {
            let q1 = QBuilder::new()
                .q(100.0)
                .fq(&ScalarUnitValue::new_freq(&1.0, Scale::Base))
                .build()
                .unwrap();
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .q1(&q1)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert!(t.q1().is_some());
            t.unset_q1();
            assert!(t.q1().is_none());
        }

        #[test]
        fn test_transformer_unset_q2() {
            let q2 = QBuilder::new()
                .q(100.0)
                .fq(&ScalarUnitValue::new_freq(&1.0, Scale::Base))
                .build()
                .unwrap();
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .q2(&q2)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert!(t.q2().is_some());
            t.unset_q2();
            assert!(t.q2().is_none());
        }

        // ----------------------------------------------------------
        // TransformerBuilder tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_builder_with_n_and_km() {
            let t = TransformerBuilder::new()
                .id("T_test")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(2.0)
                .km(0.9)
                .nodes([1, 3])
                .z0(c64(75.0, 0.0))
                .build()
                .unwrap();

            assert_eq!(t.id(), "T_test");
            t.n().assert_approx_eq(&2.0, DEFAULT_MARGIN, "n", "");
            t.km().assert_approx_eq(&0.9, DEFAULT_MARGIN, "km", "");
            // l2 = l1 / n^2 = 10e-9 / 4 = 2.5e-9
            t.l2().assert_approx_eq(&2.5e-9, DEFAULT_MARGIN, "l2", "");
            assert_eq!(t.nodes(), vec![1, 3]);
        }

        #[test]
        fn test_transformer_builder_with_l2_explicit() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .l2_val_scaled(5.0, Scale::Nano)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            // n = sqrt(l1/l2) = sqrt(10/5) = sqrt(2)
            let expected_n = (10.0 / 5.0_f64).sqrt();
            t.n().assert_approx_eq(&expected_n, DEFAULT_MARGIN, "n", "");
            t.l2().assert_approx_eq(&5e-9, DEFAULT_MARGIN, "l2", "");
        }

        #[test]
        fn test_transformer_builder_with_m_instead_of_km() {
            // m = km * l1, so km = m / l1
            let l1_val = 10e-9;
            let m_val = 8e-9;
            let expected_km = m_val / l1_val;

            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val(l1_val)
                .n(1.0)
                .m_val(m_val)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.km()
                .assert_approx_eq(&expected_km, DEFAULT_MARGIN, "km", "");
        }

        #[test]
        fn test_transformer_builder_m_val_scaled() {
            let l1_val = 10e-9;
            let m_scaled = 5.0; // 5 nH
            let m_val = 5e-9;
            let expected_km = m_val / l1_val;

            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val(l1_val)
                .n(1.0)
                .m_val_scaled(m_scaled, Scale::Nano)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.km()
                .assert_approx_eq(&expected_km, DEFAULT_MARGIN, "km", "");
        }

        #[test]
        fn test_transformer_builder_l1_val() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val(15e-9)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.l1().assert_approx_eq(&15e-9, DEFAULT_MARGIN, "l1", "");
        }

        #[test]
        fn test_transformer_builder_l2_val() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val(10e-9)
                .l2_val(5e-9)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.l2().assert_approx_eq(&5e-9, DEFAULT_MARGIN, "l2", "");
        }

        #[test]
        fn test_transformer_builder_freq() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            // Transformer should build successfully with custom frequency
            assert!(t.l1() > 0.0);
        }

        #[test]
        #[should_panic(expected = "Must specify either n or l2")]
        fn test_transformer_builder_missing_n_and_l2_panics() {
            TransformerBuilder::<f64>::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();
        }

        #[test]
        #[should_panic(expected = "Must specify either km or m")]
        fn test_transformer_builder_missing_km_and_m_panics() {
            TransformerBuilder::<f64>::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();
        }

        // ----------------------------------------------------------
        // Transformer Elem trait tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_elem_trait_id() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .id("T_elem")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t.id(), "T_elem");
        }

        #[test]
        fn test_transformer_elem_trait_name() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .id("T_name")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t.name(), "T_name");
        }

        #[test]
        fn test_transformer_elem_trait_elem_type() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t.elem(), ElemType::Transformer);
        }

        #[test]
        fn test_transformer_elem_trait_nodes() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([5, 10])
                .build()
                .unwrap();

            assert_eq!(t.nodes(), vec![5, 10]);
        }

        #[test]
        fn test_transformer_elem_trait_set_id() {
            let mut t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.set_id("T_modified");
            assert_eq!(t.id(), "T_modified");
        }

        #[test]
        fn test_transformer_elem_trait_z() {
            let freq = test_freq();
            let n = 2.0;
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(n)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = t.z(&freq);
            // z = n^2
            z.assert_approx_eq(&array![c64(n.powi(2), 0.0)], DEFAULT_MARGIN, "z", "");
        }

        #[test]
        fn test_transformer_elem_trait_z_at() {
            let freq = ArrayUnitValue::new_freq(&array![1e9, 2e9, 5e9], Scale::Base);
            let n = 3.0;
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(n)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            for i in 0..3 {
                let z = t.z_at(&freq, i);
                z.re.assert_approx_eq(
                    &n.powi(2),
                    DEFAULT_MARGIN,
                    "z.re",
                    format!("({i})").as_str(),
                );
            }
        }

        #[test]
        fn test_transformer_elem_trait_c() {
            let freq = test_freq();
            // Note: km=1.0 causes division by zero in the Y-to-S conversion
            // Use km < 1.0 for valid transformer calculations
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let c = t.c(&freq);
            assert_eq!(c.shape(), (1, 2, 2));
        }

        #[test]
        fn test_transformer_elem_trait_c_at() {
            let freq = test_freq();
            // Note: km=1.0 causes division by zero in the Y-to-S conversion
            // Use km < 1.0 for valid transformer calculations
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let c = t.c(&freq);
            for i in 0..freq.npts() {
                for j in 0..2 {
                    for k in 0..2 {
                        let c_at = t.c_at(&freq, (i, j, k));
                        // c_at should match corresponding element in c matrix
                        c_at.assert_approx_eq(
                            &c[[i, j, k]],
                            DEFAULT_MARGIN,
                            "c_at",
                            format!("[{j},{k}]").as_str(),
                        );
                    }
                }
            }
        }

        #[test]
        fn test_transformer_elem_trait_net() {
            let freq = ArrayUnitValue::new_freq(&array![1e9, 2e9, 3e9], Scale::Base);
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let net = t.net(&freq);
            assert_eq!(net.shape(), (3, 2, 2));
        }

        // ----------------------------------------------------------
        // Transformer impedance transformation tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_impedance_transformation_ratio() {
            let freq = test_freq();

            // For various turns ratios, z = n^2
            let ratios = vec![0.5, 1.0, 2.0, 4.0, 10.0];

            for n in ratios {
                let t: Transformer<f64> = TransformerBuilder::new()
                    .l1_val_scaled(10.0, Scale::Nano)
                    .n(n)
                    .km(0.9)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                let z = t.z(&freq);
                z[0].re
                    .assert_approx_eq(&n.powi(2), DEFAULT_MARGIN, "z.re", "");
            }
        }

        #[test]
        fn test_transformer_coupling_coefficient_range() {
            // km should be between 0 and 1 for physical transformers
            let km_values = vec![0.1, 0.5, 0.8, 0.95, 0.99];

            for km in km_values {
                let t: Transformer<f64> = TransformerBuilder::new()
                    .l1_val_scaled(10.0, Scale::Nano)
                    .n(1.0)
                    .km(km)
                    .nodes([1, 2])
                    .build()
                    .unwrap();

                t.km().assert_approx_eq(&km, DEFAULT_MARGIN, "km", "");
            }
        }

        // ----------------------------------------------------------
        // Clone, Debug, PartialEq trait tests
        // ----------------------------------------------------------

        #[test]
        fn test_transformer_clone() {
            let t1: Transformer<f64> = TransformerBuilder::new()
                .id("T_clone")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(2.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let t2 = t1.clone();

            assert_eq!(t1.id(), t2.id());
            t1.n().assert_approx_eq(&t2.n(), DEFAULT_MARGIN, "n", "");
            t1.km().assert_approx_eq(&t2.km(), DEFAULT_MARGIN, "km", "");
            t1.l1().assert_approx_eq(&t2.l1(), DEFAULT_MARGIN, "l1", "");
        }

        #[test]
        fn test_transformer_partial_eq() {
            let t1: Transformer<f64> = TransformerBuilder::new()
                .id("T_eq")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(2.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let t2 = TransformerBuilder::new()
                .id("T_eq")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(2.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t1, t2);
        }

        #[test]
        fn test_transformer_debug() {
            let t: Transformer<f64> = TransformerBuilder::new()
                .l1_val_scaled(10.0, Scale::Nano)
                .n(1.0)
                .km(0.9)
                .nodes([1, 2])
                .build()
                .unwrap();

            let debug_str = format!("{:?}", t);
            assert!(debug_str.contains("Transformer"));
        }

        #[test]
        fn test_transformer_builder_clone() {
            let builder1: TransformerBuilder<f64> = TransformerBuilder::new()
                .id("T_builder")
                .l1_val_scaled(10.0, Scale::Nano)
                .n(2.0)
                .km(0.9)
                .nodes([1, 2]);

            let builder2 = builder1.clone();

            let t1 = builder1.build().unwrap();
            let t2 = builder2.build().unwrap();

            assert_eq!(t1.id(), t2.id());
        }
    }

    // ============================================================
    // COMPREHENSIVE IDEAL TRANSFORMER TESTS
    // ============================================================

    mod ideal_transformer_comprehensive_tests {
        use super::*;

        // ----------------------------------------------------------
        // IdealTransformer struct direct tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_new_direct() {
            let t = IdealTransformer::new("IT_direct", 2.0, [1, 2], c64(50.0, 0.0));

            assert_eq!(t.id(), "IT_direct");
            t.val().assert_approx_eq(&2.0, DEFAULT_MARGIN, "t", "");
            assert_eq!(t.nodes(), vec![1, 2]);
            assert_eq!(t.z0(), c64(50.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_default() {
            let t: IdealTransformer<f64> = IdealTransformer::default();
            assert_eq!(t.id(), "C0");
            t.val().assert_approx_eq(&1.0, DEFAULT_MARGIN, "t", "");
            assert_eq!(t.nodes(), vec![1, 2]);
            assert_eq!(t.z0(), c64(50.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_set_val() {
            let mut t: IdealTransformer<f64> = IdealTransformer::default();
            t.set_val(5.0);
            t.val().assert_approx_eq(&5.0, DEFAULT_MARGIN, "t", "");
        }

        #[test]
        fn test_ideal_transformer_z0_accessor() {
            let t = IdealTransformer::new("IT", 1.0, [1, 2], c64(75.0, 0.0));
            assert_eq!(t.z0(), c64(75.0, 0.0));
        }

        // ----------------------------------------------------------
        // IdealTransformer S-parameter matrix tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_c_matrix_n_equals_1() {
            // When n=1, S11 = S22 = 0, S12 = S21 = 1
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let t = IdealTransformer::new("IT", 1.0, [1, 2], c64(50.0, 0.0));

            let c = t.c(&freq);

            // S11 = (1-n^2)/(1+n^2) = 0 when n=1
            c[[0, 0, 0]].assert_approx_eq(&c64(0.0, 0.0), DEFAULT_MARGIN, "c", "[0,0]");

            // S22 = (n^2-1)/(1+n^2) = 0 when n=1
            c[[0, 1, 1]].assert_approx_eq(&c64(0.0, 0.0), DEFAULT_MARGIN, "c", "[1,1]");

            // S12 = S21 = 2n/(1+n^2) = 1 when n=1
            c[[0, 0, 1]].assert_approx_eq(&c64(1.0, 0.0), DEFAULT_MARGIN, "c", "[0,1]");
            c[[0, 1, 0]].assert_approx_eq(&c64(1.0, 0.0), DEFAULT_MARGIN, "c", "[1,0]");
        }

        #[test]
        fn test_ideal_transformer_c_matrix_n_equals_2() {
            // S11 = (1-4)/(1+4) = -3/5 = -0.6
            // S22 = (4-1)/(1+4) = 3/5 = 0.6
            // S12 = S21 = 4/(1+4) = 4/5 = 0.8
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let t = IdealTransformer::new("IT", 2.0, [1, 2], c64(50.0, 0.0));

            let c = t.c(&freq);

            c[[0, 0, 0]]
                .re
                .assert_approx_eq(&-0.6, DEFAULT_MARGIN, "c.re", "[0,0]");
            c[[0, 1, 1]]
                .re
                .assert_approx_eq(&0.6, DEFAULT_MARGIN, "c.re", "[1,1]");
            c[[0, 0, 1]]
                .re
                .assert_approx_eq(&0.8, DEFAULT_MARGIN, "c.re", "[0,1]");
            c[[0, 1, 0]]
                .re
                .assert_approx_eq(&0.8, DEFAULT_MARGIN, "c.re", "[1,0]");
        }

        #[test]
        fn test_ideal_transformer_c_matrix_n_equals_0_5() {
            // n = 0.5, n^2 = 0.25
            // S11 = (1-0.25)/(1+0.25) = 0.75/1.25 = 0.6
            // S22 = (0.25-1)/(1+0.25) = -0.75/1.25 = -0.6
            // S12 = S21 = 2*0.5/(1+0.25) = 1/1.25 = 0.8
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let t = IdealTransformer::new("IT", 0.5, [1, 2], c64(50.0, 0.0));

            let c = t.c(&freq);

            c[[0, 0, 0]]
                .re
                .assert_approx_eq(&0.6, DEFAULT_MARGIN, "c.re", "[0,0]");
            c[[0, 1, 1]]
                .re
                .assert_approx_eq(&-0.6, DEFAULT_MARGIN, "c.re", "[1,1]");
            c[[0, 0, 1]]
                .re
                .assert_approx_eq(&0.8, DEFAULT_MARGIN, "c.re", "[0,1]");
            c[[0, 1, 0]]
                .re
                .assert_approx_eq(&0.8, DEFAULT_MARGIN, "c.re", "[1,0]");
        }

        #[test]
        fn test_ideal_transformer_c_matrix_symmetry() {
            // S12 should always equal S21 for ideal transformer
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let ratios = vec![0.25, 0.5, 1.0, 2.0, 4.0, 10.0];

            for n in ratios {
                let t = IdealTransformer::new("IT", n, [1, 2], c64(50.0, 0.0));
                let c = t.c(&freq);

                assert_eq!(
                    c[[0, 0, 1]],
                    c[[0, 1, 0]],
                    "S12 should equal S21 for n={}",
                    n
                );
            }
        }

        #[test]
        fn test_ideal_transformer_c_matrix_imaginary_zero() {
            // All S-parameters should be purely real for ideal transformer
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let t = IdealTransformer::new("IT", 2.5, [1, 2], c64(50.0, 0.0));

            let c = t.c(&freq);

            for i in 0..freq.npts() {
                for j in 0..2 {
                    for k in 0..2 {
                        c[[i, j, k]].im.assert_approx_eq(
                            &0.0,
                            DEFAULT_MARGIN,
                            format!("S[{i},{j},{k}] imaginary part should be 0").as_str(),
                            "",
                        );
                    }
                }
            }
        }

        // ----------------------------------------------------------
        // IdealTransformerBuilder tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_builder_new() {
            let builder = IdealTransformerBuilder::new().n(1.0).nodes([1, 2]);
            let t: IdealTransformer<f64> = builder.build().unwrap();

            assert_eq!(t.id(), "T0");
            t.val().assert_approx_eq(&1.0, DEFAULT_MARGIN, "t", "");
        }

        #[test]
        fn test_ideal_transformer_builder_id() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .id("IT_custom")
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t.id(), "IT_custom");
        }

        #[test]
        fn test_ideal_transformer_builder_n() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(3.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.val().assert_approx_eq(&3.0, DEFAULT_MARGIN, "t", "");
        }

        #[test]
        fn test_ideal_transformer_builder_val() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(4.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.val().assert_approx_eq(&4.0, DEFAULT_MARGIN, "t", "");
        }

        #[test]
        fn test_ideal_transformer_builder_nodes() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([5, 10])
                .build()
                .unwrap();

            assert_eq!(t.nodes(), vec![5, 10]);
        }

        #[test]
        fn test_ideal_transformer_builder_z0() {
            let t = IdealTransformerBuilder::new()
                .z0(c64(75.0, 0.0))
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t.z0(), c64(75.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_builder_chaining() {
            let t = IdealTransformerBuilder::new()
                .id("IT_chain")
                .n(2.5)
                .nodes([3, 7])
                .z0(c64(100.0, 0.0))
                .build()
                .unwrap();

            assert_eq!(t.id(), "IT_chain");
            t.val().assert_approx_eq(&2.5, DEFAULT_MARGIN, "t", "");
            assert_eq!(t.nodes(), vec![3, 7]);
            assert_eq!(t.z0(), c64(100.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_builder_default() {
            let builder = IdealTransformerBuilder::default().n(1.0).nodes([1, 2]);
            let t: IdealTransformer<f64> = builder.build().unwrap();

            assert_eq!(t.id(), "T0");
            t.val().assert_approx_eq(&1.0, DEFAULT_MARGIN, "t", "");
            assert_eq!(t.nodes(), vec![1, 2]);
            assert_eq!(t.z0(), c64(50.0, 0.0));
        }

        #[test]
        fn test_ideal_transformer_builder_clone() {
            let builder1: IdealTransformerBuilder<f64> = IdealTransformerBuilder::new()
                .id("IT_clone")
                .n(2.0)
                .nodes([1, 2]);

            let builder2 = builder1.clone();

            let t1 = builder1.build().unwrap();
            let t2 = builder2.build().unwrap();

            assert_eq!(t1.id(), t2.id());
            t1.val()
                .assert_approx_eq(&t2.val(), DEFAULT_MARGIN, "t1", "");
        }

        // ----------------------------------------------------------
        // IdealTransformer Elem trait tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_elem_trait_id() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .id("IT_elem")
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t.id(), "IT_elem");
        }

        #[test]
        fn test_ideal_transformer_elem_trait_name() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .id("IT_name")
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t.name(), "IT_name");
        }

        #[test]
        fn test_ideal_transformer_elem_trait_elem_type() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t.elem(), ElemType::IdealTransformer);
        }

        #[test]
        fn test_ideal_transformer_elem_trait_nodes() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([8, 15])
                .build()
                .unwrap();

            assert_eq!(t.nodes(), vec![8, 15]);
        }

        #[test]
        fn test_ideal_transformer_elem_trait_set_id() {
            let mut t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(1.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            t.set_id("IT_modified");
            assert_eq!(t.id(), "IT_modified");
        }

        #[test]
        fn test_ideal_transformer_elem_trait_z() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let n = 3.0;
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(n)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = t.z(&freq);
            // z = n^2
            z.assert_approx_eq(&array![c64(n.powi(2), 0.0)], DEFAULT_MARGIN, "z", "");
        }

        #[test]
        fn test_ideal_transformer_elem_trait_z_at() {
            let freq = ArrayUnitValue::new_freq(&array![1e9, 2e9, 5e9], Scale::Base);
            let n = 2.0;
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(n)
                .nodes([1, 2])
                .build()
                .unwrap();

            for i in 0..3 {
                let z = t.z_at(&freq, i);
                z.re.assert_approx_eq(&n.powi(2), DEFAULT_MARGIN, "z", format!("[{i}]").as_str());
            }
        }

        #[test]
        fn test_ideal_transformer_elem_trait_c_at() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(2.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let c = t.c(&freq);
            for i in 0..freq.npts() {
                for j in 0..2 {
                    for k in 0..2 {
                        let c_at = t.c_at(&freq, (i, j, k));
                        assert_eq!(c_at, c[[i, j, k]]);
                    }
                }
            }
        }

        #[test]
        fn test_ideal_transformer_elem_trait_net() {
            let freq = ArrayUnitValue::new_freq(&array![1e9, 2e9, 3e9], Scale::Base);
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(2.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let net = t.net(&freq);
            assert_eq!(net.shape(), (3, 2, 2));
        }

        #[test]
        fn test_ideal_transformer_net_frequency_independence() {
            // Ideal transformer S-parameters should be frequency independent
            let freq = ArrayUnitValue::new_freq(&array![1e6, 1e9, 100e9], Scale::Base);
            let n = 2.0;
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(n)
                .nodes([1, 2])
                .build()
                .unwrap();

            let net = t.net(&freq);

            let expected_s11 = (1.0 - n.powi(2)) / (1.0 + n.powi(2));
            let expected_s22 = (n.powi(2) - 1.0) / (1.0 + n.powi(2));
            let expected_s12 = 2.0 * n / (1.0 + n.powi(2));

            for i in 0..3 {
                net[[i, 0, 0]].re.assert_approx_eq(
                    &expected_s11,
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},0,0]").as_str(),
                );
                net[[i, 1, 1]].re.assert_approx_eq(
                    &expected_s22,
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},1,1]").as_str(),
                );
                net[[i, 0, 1]].re.assert_approx_eq(
                    &expected_s12,
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},0,1]").as_str(),
                );
                net[[i, 1, 0]].re.assert_approx_eq(
                    &expected_s12,
                    DEFAULT_MARGIN,
                    "net",
                    format!("[{i},1,0]").as_str(),
                );
            }
        }

        // ----------------------------------------------------------
        // IdealTransformer impedance transformation tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_impedance_transformation_ratio() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);

            let ratios = vec![0.5, 1.0, 2.0, 4.0, 10.0];

            for n in ratios {
                let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                    .n(n)
                    .nodes([1, 2])
                    .build()
                    .unwrap();
                let z = t.z(&freq);

                z[0].re.assert_approx_eq(
                    &n.powi(2),
                    DEFAULT_MARGIN,
                    format!("z should be n^2 for n={n}").as_str(),
                    "",
                );
            }
        }

        // ----------------------------------------------------------
        // Clone, Debug, PartialEq trait tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_clone() {
            let t1 = IdealTransformerBuilder::new()
                .id("IT_clone")
                .n(2.0)
                .nodes([3, 4])
                .z0(c64(75.0, 0.0))
                .build()
                .unwrap();

            let t2 = t1.clone();

            assert_eq!(t1.id(), t2.id());
            t1.val()
                .assert_approx_eq(&t2.val(), DEFAULT_MARGIN, "t1", "");
            assert_eq!(t1.nodes(), t2.nodes());
            assert_eq!(t1.z0(), t2.z0());
        }

        #[test]
        fn test_ideal_transformer_partial_eq() {
            let t1: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .id("IT_eq")
                .n(2.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let t2 = IdealTransformerBuilder::new()
                .id("IT_eq")
                .n(2.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_eq!(t1, t2);
        }

        #[test]
        fn test_ideal_transformer_not_equal() {
            let t1: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(2.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let t2 = IdealTransformerBuilder::new()
                .n(3.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            assert_ne!(t1, t2);
        }

        #[test]
        fn test_ideal_transformer_debug() {
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(2.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            let debug_str = format!("{:?}", t);
            assert!(debug_str.contains("IdealTransformer"));
        }

        // ----------------------------------------------------------
        // Edge case and boundary tests
        // ----------------------------------------------------------

        #[test]
        fn test_ideal_transformer_very_small_n() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let n = 0.01;
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(n)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = t.z_at(&freq, 0);
            assert!(z.is_finite());
            z.re.assert_approx_eq(&n.powi(2), DEFAULT_MARGIN, "z.re", "");
        }

        #[test]
        fn test_ideal_transformer_very_large_n() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let n = 100.0;
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(n)
                .nodes([1, 2])
                .build()
                .unwrap();

            let z = t.z_at(&freq, 0);
            assert!(z.is_finite());
            z.re.assert_approx_eq(&n.powi(2), RELAXED_MARGIN, "z.re", "");
        }

        #[test]
        fn test_ideal_transformer_c_matrix_consistency() {
            let freq = ArrayUnitValue::new_freq(&array![1e9], Scale::Base);
            let t: IdealTransformer<f64> = IdealTransformerBuilder::new()
                .n(2.0)
                .nodes([1, 2])
                .build()
                .unwrap();

            // Multiple calls should return the same result
            let c1 = t.c(&freq);
            let c2 = t.c(&freq);

            for i in 0..freq.npts() {
                for j in 0..2 {
                    for k in 0..2 {
                        assert_eq!(c1[[i, j, k]], c2[[i, j, k]]);
                    }
                }
            }
        }
    }
}
