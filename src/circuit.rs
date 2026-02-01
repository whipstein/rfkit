use crate::{
    element::{Elem, ElemType, Element},
    frequency::{FreqArray, Frequency},
    network::{Network, NetworkBuilder},
    prelude::ElementBuilder,
    pts::{Matrix, Points, Pts},
};
use ndarray::{IntoDimension, linalg::Dot, prelude::*};
use num::complex::{Complex, Complex64};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::{Debug, Display, Formatter, Result},
    panic,
};

/// Representation of a node in a circuit
#[derive(Clone)]
struct Node {
    elem: Vec<Element>,        // Elements connected to the node
    port: Option<Element>,     // Port connected to the node
    x: Points<Complex64, Ix3>, // Interaction scattering matrix for the node
    y: Vec<Complex64>,         // Sum of admittances connected to the node
    ground: bool,              // Is this a ground node
}

impl Node {
    /// Add element connected to Node
    fn add_elem(&self, elem: &Element, freq: &Frequency) -> Node {
        let mut new_node = self.copy();
        if self.elem.len() != 0 {
            new_node.x = Points::<Complex64, Ix3>::from_shape_fn(
                (self.x.dim().0, self.x.dim().1 + 1, self.x.dim().2 + 1).into_dimension(),
                |(i, j, k)| {
                    if j < self.x.dim().1 && k < self.x.dim().2 {
                        self.x[[i, j, k]]
                    } else {
                        Complex64::ZERO
                    }
                },
            );
        }

        match elem.elem() {
            ElemType::Port => {
                match new_node.port {
                    Some(_) => panic!("Only one port is allowed on each node"),
                    None => new_node.port = Some(elem.clone()),
                }
                new_node.elem.insert(0, elem.clone());
            }
            _ => new_node.elem.push(elem.clone()),
        }

        new_node.calc_y(freq);
        new_node.calc_x(freq);

        new_node
    }

    /// Add element connected to Node
    fn add_elem_inplace(&mut self, elem: Element, freq: &Frequency) {
        match elem.elem() {
            ElemType::Port => {
                match self.port {
                    Some(_) => panic!("Only one port is allowed on each node"),
                    None => self.port = Some(elem.clone()),
                }
                self.elem.insert(0, elem);
            }
            _ => self.elem.push(elem),
        }

        self.calc_y(freq);
        self.calc_x(freq);
    }

    /// Calculate the interaction scattering matrix
    fn calc_x(&mut self, freq: &Frequency) {
        if self.ground {
            for (_, mut pt) in self.x.axis_iter_mut(Axis(0)).enumerate() {
                let x = Array2::<Complex64>::from_shape_fn(
                    (self.elem.len(), self.elem.len()),
                    |(j, k)| {
                        if j == k {
                            -Complex64::ONE
                        } else {
                            Complex64::ZERO
                        }
                    },
                );
                pt.assign(&x);
            }
        } else {
            for (i, mut pt) in self.x.axis_iter_mut(Axis(0)).enumerate() {
                let x = Array2::<Complex64>::from_shape_fn(
                    (self.elem.len(), self.elem.len()),
                    |(j, k)| {
                        if j == k {
                            2.0 / (self.elem[k].z_at(&freq, i) * self.y[i]) - 1.0
                        } else {
                            2.0 / ((self.elem[j].z_at(&freq, i) * self.elem[k].z_at(&freq, i))
                                .sqrt()
                                * self.y[i])
                        }
                    },
                );
                pt.assign(&x);
            }
        }
    }

    /// Calculate the sum of admittances connected to Node
    fn calc_y(&mut self, freq: &Frequency) {
        for i in 0..self.y.len() {
            self.y[i] = Complex64::ZERO;

            for elem in self.elem.iter() {
                self.y[i] += 1.0 / elem.z_at(&freq, i);
            }
        }
    }

    /// Create a deep copy of Node
    fn copy(&self) -> Node {
        Node {
            elem: self.elem.clone(),
            port: self.port.clone(),
            x: self.x.clone(),
            y: self.y.clone(),
            ground: self.ground,
        }
    }

    /// Number of elements connected to Node
    fn len(&self) -> usize {
        self.elem.len()
    }

    /// Create a new Node
    fn new(freq: &Frequency, ground: bool) -> Node {
        let elem = match ground {
            true => vec![
                ElementBuilder::new()
                    .elem(ElemType::Ground)
                    .build()
                    .unwrap(),
            ],
            false => vec![],
        };
        Node {
            elem: elem,
            port: None,
            x: Points::<Complex64, Ix3>::zeros((freq.npts(), 1, 1).into_dimension()),
            y: vec![Complex64::ZERO; freq.npts()],
            ground: ground,
        }
    }

    /// Is this a ground node
    fn ground(&self) -> bool {
        self.ground
    }

    /// Return port connected to Node
    fn port(&self) -> &Option<Element> {
        &self.port
    }

    /// Return interaction scattering matrix of Node
    fn x(&self) -> &Points<Complex64, Ix3> {
        &self.x
    }

    /// Return a point from the interaction scattering matrix of Node
    fn x_val(&self, i: usize, j: usize, k: usize) -> &Complex<f64> {
        &self.x[[i, j, k]]
    }

    /// Return sum of admittances for Node
    fn y(&self) -> &Vec<Complex64> {
        &self.y
    }

    /// Return a point from the sum of admittances for Node
    fn y_val(&self, i: usize) -> &Complex<f64> {
        &self.y[i]
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Node")
            .field("elements", &self.elem.len())
            .field("has a port", &self.port.is_some())
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "{} elements, has a port: {}, {}, {}",
            self.elem.len(),
            self.port.is_some(),
            self.x.slice(s![0, .., ..]),
            self.y[0]
        )
    }
}

impl Eq for Node {}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.elem == other.elem && self.x() == other.x() && self.y() == other.y()
    }
}

#[derive(Clone, Debug)]
struct CircuitMap {
    names: Vec<String>, // Vec of <Element Names>; Vec position is Circuit matrix position
    elem_map: Vec<(String, usize)>, // Vec of (Element ID, Element Node) for entire circuit
    map: Vec<usize>,    // Vec of <Element Node>; Vec position is Circuit matrix position
    node_map: Vec<usize>, // Vec of <Node number>; Vec position is Circuit matrix position
    port_map: Vec<usize>, // Vec of <Port number>; Vec position is Circuit matrix position
}

impl CircuitMap {
    /// Add element to port
    /// Finds port locating in vector and inserts new element node right before it
    pub fn add(&mut self, node: usize, pin: usize, id: String) {
        if self.node_map.len() > node + 1 {
            let position = self.node_map[node + 1];
            self.elem_map.insert(position, (id.clone(), pin));
            self.names.insert(position, id);
            self.map.insert(position, pin);
            for (i, pos) in self.node_map.iter_mut().enumerate() {
                if i > node {
                    *pos += 1;
                }
            }
            for (i, pos) in self.port_map.iter_mut().enumerate() {
                if i > node {
                    *pos += 1;
                }
            }
        } else {
            self.elem_map.push((id.clone(), pin));
            self.names.push(id);
            self.map.push(pin);
            if self.node_map.len() <= node {
                self.node_map.push(self.map.len() - 1);
            }
        }
    }

    /// Add port
    /// Places port at the end of the map
    /// Assumes ports are added in order
    pub fn add_port(&mut self, node: usize, id: String) {
        if self.node_map.len() > node + 1 {
            let mut node_val = 0;
            for (i, &pos) in self.node_map.iter().enumerate() {
                if i > node && pos != 0 && (node_val == 0 || pos < node_val) {
                    node_val = pos;
                }
            }
            self.node_map[node] = node_val;
            for (i, pos) in self.node_map.iter_mut().enumerate() {
                if i > node && *pos != 0 {
                    *pos += 1;
                }
            }
            let position = self.node_map[node];
            self.elem_map.insert(position, (id.clone(), 0));
            self.names.insert(position, id);
            self.map.insert(position, 0);
            self.port_map.insert(position, position);
            for (i, pos) in self.port_map.iter_mut().enumerate() {
                if i > position {
                    *pos += 1;
                }
            }
        } else {
            self.elem_map.push((id.clone(), 0));
            self.names.push(id);
            self.map.push(0);
            for _ in self.node_map.len() - 1..node - 1 {
                self.node_map.push(0);
            }
            self.node_map.push(self.map.len() - 1);
            self.port_map.push(self.map.len() - 1);
        }
    }

    pub fn clear(&mut self) {
        self.names = vec!["gnd".to_string()];
        self.elem_map = vec![("gnd".to_string(), 0)];
        self.map = vec![0];
        self.node_map = vec![0];
        self.port_map = vec![0];
    }

    pub fn elem_map(&self) -> Vec<(String, usize)> {
        self.elem_map.clone()
    }

    pub fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    pub fn map(&self) -> Vec<usize> {
        self.map.clone()
    }

    pub fn new() -> CircuitMap {
        CircuitMap {
            names: vec!["gnd".to_string()],
            elem_map: vec![("gnd".to_string(), 0)],
            map: vec![0],
            node_map: vec![0],
            port_map: vec![0],
        }
    }

    pub fn node_map(&self) -> Vec<usize> {
        self.node_map.clone()
    }

    pub fn port_map(&self) -> Vec<usize> {
        let mut out = self.port_map.clone();
        _ = out.remove(0);
        out
    }
}

/// "Method for Calculating the
/// Scattering Matrix of Arbitrary
/// Microwave Networks Giving Both
/// Internal and External
/// Scattering," P. Hallbjörner,
/// Microwave and Optical Technology Letters,
/// Vol. 38, No. 2, 2003, pgs 99-102.
#[derive(Debug, Clone)]
pub struct Circuit {
    c: Points<Complex64, Ix3>,          // Element Scattering Matrix
    freq: Frequency,                    // Frequencies
    z0: Array1<Complex64>,              // Port Impedances
    map: CircuitMap,                    // Map element node to full circuit matrix position
    nodes: BTreeMap<usize, Node>,       // Interactions
    ports_only: bool,                   // Circuit Contains Ports Only
    s: Points<Complex64, Ix3>,          // Circuit Scattering Matrix
    x: Points<Complex64, Ix3>,          // Interaction Scattering Matrix
    dim: (usize, usize, usize),         // Dimension of S
    dimx: (usize, usize, usize),        // Dimension of C & X
    elements: HashMap<String, Element>, // Map of <Element ID, Element>
}

impl Circuit {
    /// Add an element to the circuit
    pub fn add_elem(&mut self, elem: &Element, freq: &Frequency) {
        for (i, &elem_node) in elem.nodes().iter().enumerate() {
            if elem.elem() == ElemType::Port {
                self.dim = (self.dim.0, self.dim.1 + 1, self.dim.2 + 1);
                self.map.add_port(elem_node, elem.id());
                let mut z0 = self.z0.clone().to_vec();
                z0.push(elem.z(&self.freq));
                self.z0 = Array1::from_vec(z0);
            } else {
                self.ports_only = false;
                self.map.add(elem_node, i, elem.id());
            }

            if !self.nodes.contains_key(&elem_node) {
                self.nodes.insert(elem_node, Node::new(&self.freq, false));
            }

            self.nodes.insert(
                elem_node,
                self.nodes
                    .get(&elem_node)
                    .unwrap()
                    .add_elem(&elem, &self.freq),
            );

            self.dimx = (self.dimx.0, self.dimx.1 + 1, self.dimx.2 + 1);
        }

        self.elements.insert(elem.id(), elem.clone());

        self.update(freq);
    }

    /// Calculate the element scattering matrix
    fn calc_c(&mut self, freq: &Frequency) {
        self.c = Points::<Complex64, Ix3>::zeros(self.dimx.into_dimension());

        for (j, (id1, node1)) in self.map.elem_map().iter().enumerate() {
            for (k, (id2, node2)) in self.map.elem_map().iter().enumerate() {
                if id1 == id2 {
                    for i in 0..self.freq.npts() {
                        let val = match self.elements.get(id1) {
                            Some(x) => x.c_at(freq, *node1, *node2),
                            None => Complex64::ZERO,
                        };
                        self.c[[i, j, k]] = val;
                    }
                }
            }
        }
    }

    /// Calculate the circuit scattering matrix
    fn calc_s(&mut self) {
        self.s = Points::<Complex64, Ix3>::zeros(self.dim.into_dimension());

        if self.ports_only {
            return;
        }

        let id = Points::<Complex64, Ix2>::eye(self.x.dim().1);

        for i in 0..self.freq.npts() {
            let c: Points<Complex64, Ix2> = Points::from(self.c.slice(s![i, .., ..]));
            let x: Points<Complex64, Ix2> = Points::from(self.x.slice(s![i, .., ..]));

            let mut net = &id - &c.dot(&x);

            let net_inv = match net.try_inv() {
                Ok(val) => val,
                _ => panic!("{}", format!("inverse does not exist. {:?}", net)),
            };

            net = x.dot(&net_inv);

            for (j, &map_j) in self.map.port_map().iter().enumerate() {
                for (k, &map_k) in self.map.port_map().iter().enumerate() {
                    self.s[[i, j, k]] = net[[map_j, map_k]];
                }
            }
        }
    }

    /// Calculate the interaction scattering matrix\
    fn calc_x(&mut self) {
        self.x = Points::<Complex64, Ix3>::zeros(self.dimx.into_dimension());

        let mut cntr = 0;
        for node in self.nodes.clone().into_values() {
            let j = cntr + node.len();
            self.x
                .slice_mut(s![.., cntr..j, cntr..j])
                .assign(node.x().inner());
            cntr += node.len();
        }
    }

    /// Clear the circuit
    pub fn clear(&mut self) {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, Node::new(&self.freq, true));
        self.c = Points::<Complex64, Ix3>::zeros((self.dimx.0, 1, 1).into_dimension());
        self.map.clear();
        self.nodes = nodes;
        self.ports_only = true;
        self.s = Points::<Complex64, Ix3>::zeros((self.dim.0, 0, 0).into_dimension());
        self.x = Points::<Complex64, Ix3>::zeros((self.dimx.0, 1, 1).into_dimension());
        self.dim = (self.dim.0, 0, 0);
        self.dimx = (self.dimx.0, 1, 1);
        self.elements.clear();
    }

    /// Make a deep copy of the circuit
    pub fn copy(&self) -> Circuit {
        Circuit {
            c: self.c.clone(),
            freq: self.freq.clone(),
            z0: self.z0.clone(),
            map: self.map.clone(),
            nodes: self.nodes.clone(),
            ports_only: self.ports_only,
            s: self.s.clone(),
            x: self.x.clone(),
            dim: self.dim,
            dimx: self.dim,
            elements: self.elements.clone(),
        }
    }

    /// Create a new empty circuit
    pub fn new(f: &Frequency) -> Circuit {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, Node::new(f, true));
        Circuit {
            c: Points::<Complex64, Ix3>::zeros((f.npts(), 1, 1).into_dimension()),
            dim: (f.npts(), 0, 0),
            dimx: (f.npts(), 1, 1),
            freq: f.clone(),
            z0: array![],
            map: CircuitMap::new(),
            nodes: nodes,
            ports_only: true,
            s: Points::<Complex64, Ix3>::zeros((f.npts(), 0, 0).into_dimension()),
            x: Points::<Complex64, Ix3>::zeros((f.npts(), 1, 1).into_dimension()),
            elements: HashMap::new(),
        }
    }

    /// Update derived values after changing inputs
    fn update(&mut self, freq: &Frequency) {
        self.calc_x();
        self.calc_c(freq);
        self.calc_s();
    }

    pub fn c(&self) -> Points<Complex64, Ix3> {
        self.c.clone()
    }

    pub fn freq(&self) -> Frequency {
        self.freq.clone()
    }

    pub fn ports_only(&self) -> bool {
        self.ports_only
    }

    pub fn s(&self) -> Points<Complex64, Ix3> {
        self.s.clone()
    }

    pub fn x(&self) -> Points<Complex64, Ix3> {
        self.x.clone()
    }

    pub fn dim(&self) -> (usize, usize, usize) {
        self.dim
    }

    pub fn dimx(&self) -> (usize, usize, usize) {
        self.dimx
    }

    pub fn elements(&self) -> HashMap<String, Element> {
        self.elements.clone()
    }

    pub fn z0(&self) -> Array1<Complex64> {
        self.z0.clone()
    }

    pub fn net(&self) -> Network {
        NetworkBuilder::new()
            .freq(self.freq().clone())
            .z0(self.z0.clone())
            .s(self.s.clone())
            .build()
    }
}

#[cfg(test)]
mod circuit_tests {
    use super::*;
    use crate::{
        element::{ElementBuilder, QBuilder, QMode, msub::MsubBuilder},
        frequency::FrequencyBuilder,
        impedance::{ComplexNumberType, ImpedanceBuilder, ImpedanceMode, ImpedanceType},
        points,
        scale::Scale,
        unit::{Unit, UnitValBuilder},
        util::{comp_pts_ix3, comp_vec_c64},
    };
    use float_cmp::F64Margin;
    use num::complex::c64;

    // const MARGIN: F64Margin = F64Margin {
    //     epsilon: f64::EPSILON,
    //     ulps: 10,
    // };
    const MARGIN: F64Margin = F64Margin {
        epsilon: 1e-14,
        ulps: 10,
    };

    #[test]
    fn node_rlc() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let mut node = Node::new(&freq, false);
        let margin = F64Margin::default();

        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 2])
            .id("C1")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val_scaled(20.0, Scale::Base)
            .nodes(vec![1, 2])
            .id("R1")
            .build()
            .unwrap();
        let z_p1 = p1.z(&freq);
        let z_r1 = r1.z(&freq);
        let z_c1 = c1.z(&freq);
        let z_l1 = l1.z(&freq);

        let y = 1.0 / z_p1;
        let exemplar_y = vec![y];
        let exemplar_x = Points::<Complex64, Ix3>::ones((1, 1, 1));
        node = node.add_elem(&p1, &freq);
        comp_vec_c64(&exemplar_y, &node.y(), margin, "node.y(p1)");
        comp_pts_ix3(&exemplar_x, &node.x, margin, "node.x(p1)");

        let y = 1.0 / z_p1 + 1.0 / z_r1;
        let x11 = 2.0 / (z_p1 * y) - 1.0;
        let x22 = 2.0 / (z_r1 * y) - 1.0;
        let x12 = 2.0 / ((z_p1 * z_r1).sqrt() * y);
        let exemplar_y = vec![y];
        let exemplar_x = points![Complex64, [[x11, x12], [x12, x22]]];
        node = node.add_elem(&r1, &freq);
        comp_vec_c64(&exemplar_y, &node.y(), margin, "node.y(r1)");
        comp_pts_ix3(&exemplar_x, &node.x, margin, "node.x(r1)");

        let y = 1.0 / z_p1 + 1.0 / z_r1 + 1.0 / z_c1;
        let x11 = 2.0 / (z_p1 * y) - 1.0;
        let x22 = 2.0 / (z_r1 * y) - 1.0;
        let x33 = 2.0 / (z_c1 * y) - 1.0;
        let x12 = 2.0 / ((z_p1 * z_r1).sqrt() * y);
        let x13 = 2.0 / ((z_p1 * z_c1).sqrt() * y);
        let x23 = 2.0 / ((z_r1 * z_c1).sqrt() * y);
        let exemplar_y = vec![y];
        let exemplar_x = points![
            Complex64,
            [[x11, x12, x13], [x12, x22, x23], [x13, x23, x33]]
        ];
        node = node.add_elem(&c1, &freq);
        comp_vec_c64(&exemplar_y, &node.y(), margin, "node.y(c1)");
        comp_pts_ix3(&exemplar_x, &node.x, margin, "node.x(c1)");

        let y = 1.0 / z_p1 + 1.0 / z_r1 + 1.0 / z_c1 + 1.0 / z_l1;
        let x11 = 2.0 / (z_p1 * y) - 1.0;
        let x22 = 2.0 / (z_r1 * y) - 1.0;
        let x33 = 2.0 / (z_c1 * y) - 1.0;
        let x44 = 2.0 / (z_l1 * y) - 1.0;
        let x12 = 2.0 / ((z_p1 * z_r1).sqrt() * y);
        let x13 = 2.0 / ((z_p1 * z_c1).sqrt() * y);
        let x14 = 2.0 / ((z_p1 * z_l1).sqrt() * y);
        let x23 = 2.0 / ((z_r1 * z_c1).sqrt() * y);
        let x24 = 2.0 / ((z_r1 * z_l1).sqrt() * y);
        let x34 = 2.0 / ((z_c1 * z_l1).sqrt() * y);
        let exemplar_y = vec![y];
        let exemplar_x = points![
            Complex64,
            [
                [x11, x12, x13, x14],
                [x12, x22, x23, x24],
                [x13, x23, x33, x34],
                [x14, x24, x34, x44]
            ]
        ];
        node = node.add_elem(&l1, &freq);
        comp_vec_c64(&exemplar_y, &node.y(), margin, "node.y(l1)");
        comp_pts_ix3(&exemplar_x, &node.x, margin, "node.x(l1)");
    }

    #[test]
    fn circuitmap1() {
        let mut map = CircuitMap::new();
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string()],
            elem_map: vec![("gnd".to_string(), 0)],
            map: vec![0],      // [gnd]
            node_map: vec![0], // [0]
            port_map: vec![0], // [gnd]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add_port(1, "p1".to_string());
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string(), "p1".to_string()],
            elem_map: vec![("gnd".to_string(), 0), ("p1".to_string(), 0)],
            map: vec![0, 0],      // [gnd, p1]
            node_map: vec![0, 1], // [0, 1]
            port_map: vec![0, 1], // [gnd, p1]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add_port(2, "p2".to_string());
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string(), "p1".to_string(), "p2".to_string()],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("p2".to_string(), 0),
            ],
            map: vec![0, 0, 0],      // [gnd, p1, p2]
            node_map: vec![0, 1, 2], // [0, 1, 2]
            port_map: vec![0, 1, 2], // [gnd, p1, p2]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add_port(3, "p3".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("p2".to_string(), 0),
                ("p3".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0],      // [gnd, p1, p2, p3]
            node_map: vec![0, 1, 2, 3], // [0, 1, 2, 3]
            port_map: vec![0, 1, 2, 3], // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add(1, 0, "L1".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("p3".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 0],   // [gnd, p1, L1, p2, p3]
            node_map: vec![0, 1, 3, 4], // [0, 1, 2, 3]
            port_map: vec![0, 1, 3, 4], // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add(2, 1, "L1".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "L1".to_string(),
                "p3".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("L1".to_string(), 1),
                ("p3".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 1, 0], // [gnd, p1, L1, p2, L1, p3]
            node_map: vec![0, 1, 3, 5],  // [0, 1, 2, 3]
            port_map: vec![0, 1, 3, 5],  // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add(1, 0, "L2".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "p2".to_string(),
                "L1".to_string(),
                "p3".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("L2".to_string(), 0),
                ("p2".to_string(), 0),
                ("L1".to_string(), 1),
                ("p3".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 0, 1, 0], // [gnd, p1, L1, L2, p2, L1, p3]
            node_map: vec![0, 1, 4, 6],     // [0, 1, 2, 3]
            port_map: vec![0, 1, 4, 6],     // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add(3, 1, "L2".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "p2".to_string(),
                "L1".to_string(),
                "p3".to_string(),
                "L2".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("L2".to_string(), 0),
                ("p2".to_string(), 0),
                ("L1".to_string(), 1),
                ("p3".to_string(), 0),
                ("L2".to_string(), 1),
            ],
            map: vec![0, 0, 0, 0, 0, 1, 0, 1], // [gnd, p1, L1, L2, p2, L1, p3, L2]
            node_map: vec![0, 1, 4, 6],        // [0, 1, 2, 3]
            port_map: vec![0, 1, 4, 6],        // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add(2, 0, "R1".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "p2".to_string(),
                "L1".to_string(),
                "R1".to_string(),
                "p3".to_string(),
                "L2".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("L2".to_string(), 0),
                ("p2".to_string(), 0),
                ("L1".to_string(), 1),
                ("R1".to_string(), 0),
                ("p3".to_string(), 0),
                ("L2".to_string(), 1),
            ],
            map: vec![0, 0, 0, 0, 0, 1, 0, 0, 1], // [gnd, p1, L1, L2, p2, L1, R1, p3, L2]
            node_map: vec![0, 1, 4, 7],           // [0, 1, 2, 3]
            port_map: vec![0, 1, 4, 7],           // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add(3, 1, "R1".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "p2".to_string(),
                "L1".to_string(),
                "R1".to_string(),
                "p3".to_string(),
                "L2".to_string(),
                "R1".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("L2".to_string(), 0),
                ("p2".to_string(), 0),
                ("L1".to_string(), 1),
                ("R1".to_string(), 0),
                ("p3".to_string(), 0),
                ("L2".to_string(), 1),
                ("R1".to_string(), 1),
            ],
            map: vec![0, 0, 0, 0, 0, 1, 0, 0, 1, 1], // [gnd, p1, L1, L2, p2, L1, R1, p3, L2, R1]
            node_map: vec![0, 1, 4, 7],              // [0, 1, 2, 3]
            port_map: vec![0, 1, 4, 7],              // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.names(), &map.names());
        assert_eq!(&exemplar.elem_map(), &map.elem_map());
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add(1, 0, "C10".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "C10".to_string(),
                "p2".to_string(),
                "L1".to_string(),
                "R1".to_string(),
                "p3".to_string(),
                "L2".to_string(),
                "R1".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("L2".to_string(), 0),
                ("C10".to_string(), 0),
                ("p2".to_string(), 0),
                ("L1".to_string(), 1),
                ("R1".to_string(), 0),
                ("p3".to_string(), 0),
                ("L2".to_string(), 1),
                ("R1".to_string(), 1),
            ],
            map: vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1], // [gnd, p1, L1, L, C10, p2, L1, R1, p3, L2, R1]
            node_map: vec![0, 1, 5, 8],                 // [0, 1, 2, 3]
            port_map: vec![0, 1, 5, 8],                 // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.names(), &map.names());
        assert_eq!(&exemplar.elem_map(), &map.elem_map());
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add(0, 1, "C10".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "C10".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "C10".to_string(),
                "p2".to_string(),
                "L1".to_string(),
                "R1".to_string(),
                "p3".to_string(),
                "L2".to_string(),
                "R1".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("C10".to_string(), 1),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("L2".to_string(), 0),
                ("C10".to_string(), 0),
                ("p2".to_string(), 0),
                ("L1".to_string(), 1),
                ("R1".to_string(), 0),
                ("p3".to_string(), 0),
                ("L2".to_string(), 1),
                ("R1".to_string(), 1),
            ],
            map: vec![0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1], // [gnd, C10, p1, L1, L, C10, p2, L1, R1, p3, L2, R1]
            node_map: vec![0, 2, 6, 9],                    // [0, 1, 2, 3]
            port_map: vec![0, 2, 6, 9],                    // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.names(), &map.names());
        assert_eq!(&exemplar.elem_map(), &map.elem_map());
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());
    }

    #[test]
    fn circuitmap2() {
        let mut map2 = CircuitMap::new();
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string()],
            elem_map: vec![("gnd".to_string(), 0)],
            map: vec![0],      // [gnd]
            node_map: vec![0], // [0]
            port_map: vec![0], // [gnd]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add_port(1, "p1".to_string());
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string(), "p1".to_string()],
            elem_map: vec![("gnd".to_string(), 0), ("p1".to_string(), 0)],
            map: vec![0, 0],      // [gnd, p1]
            node_map: vec![0, 1], // [0, 1]
            port_map: vec![0, 1], // [gnd, p1]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add_port(2, "p2".to_string());
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string(), "p1".to_string(), "p2".to_string()],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("p2".to_string(), 0),
            ],
            map: vec![0, 0, 0],      // [gnd, p1, p2]
            node_map: vec![0, 1, 2], // [0, 1, 2]
            port_map: vec![0, 1, 2], // [gnd, p1, p2]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add_port(3, "p3".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("p2".to_string(), 0),
                ("p3".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0],      // [gnd, p1, p2, p3]
            node_map: vec![0, 1, 2, 3], // [0, 1, 2, 3]
            port_map: vec![0, 1, 2, 3], // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add(1, 0, "L1".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("p3".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 0],   // [gnd, p1, L1, p2, p3]
            node_map: vec![0, 1, 3, 4], // [0, 1, 2, 3]
            port_map: vec![0, 1, 3, 4], // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add(4, 1, "L1".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
                "L1".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("p3".to_string(), 0),
                ("L1".to_string(), 1),
            ],
            map: vec![0, 0, 0, 0, 0, 1],   // [gnd, p1, L1, p2, p3, L1]
            node_map: vec![0, 1, 3, 4, 5], // [0, 1, 2, 3, 4]
            port_map: vec![0, 1, 3, 4],    // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add(4, 0, "L2".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
                "L1".to_string(),
                "L2".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("p3".to_string(), 0),
                ("L1".to_string(), 1),
                ("L2".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 0, 1, 0], // [gnd, p1, L1, p2, p3, L1, L2]
            node_map: vec![0, 1, 3, 4, 5],  // [0, 1, 2, 3, 4]
            port_map: vec![0, 1, 3, 4],     // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add(2, 1, "L2".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "L2".to_string(),
                "p3".to_string(),
                "L1".to_string(),
                "L2".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("L2".to_string(), 1),
                ("p3".to_string(), 0),
                ("L1".to_string(), 1),
                ("L2".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 1, 0, 1, 0], // [gnd, p1, L1, p2, L2, p3, L1, L2]
            node_map: vec![0, 1, 3, 5, 6],     // [0, 1, 2, 3, 4]
            port_map: vec![0, 1, 3, 5],        // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add(4, 0, "R1".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "L2".to_string(),
                "p3".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "R1".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("L2".to_string(), 1),
                ("p3".to_string(), 0),
                ("L1".to_string(), 1),
                ("L2".to_string(), 0),
                ("R1".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 1, 0, 1, 0, 0], // [gnd, p1, L1, p2, L2, p3, L1, L2, R1]
            node_map: vec![0, 1, 3, 5, 6],        // [0, 1, 2, 3, 4]
            port_map: vec![0, 1, 3, 5],           // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add(3, 1, "R1".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "L2".to_string(),
                "p3".to_string(),
                "R1".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "R1".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("L2".to_string(), 1),
                ("p3".to_string(), 0),
                ("R1".to_string(), 1),
                ("L1".to_string(), 1),
                ("L2".to_string(), 0),
                ("R1".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 1, 0, 1, 1, 0, 0], // [gnd, p1, L1, p2, L2, p3, R1, L1, L2, R1]
            node_map: vec![0, 1, 3, 5, 7],           // [0, 1, 2, 3, 4]
            port_map: vec![0, 1, 3, 5],              // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.names(), &map2.names());
        assert_eq!(&exemplar.elem_map(), &map2.elem_map());
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add(3, 0, "C10".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "L2".to_string(),
                "p3".to_string(),
                "R1".to_string(),
                "C10".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "R1".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("L2".to_string(), 1),
                ("p3".to_string(), 0),
                ("R1".to_string(), 1),
                ("C10".to_string(), 0),
                ("L1".to_string(), 1),
                ("L2".to_string(), 0),
                ("R1".to_string(), 0),
            ],
            map: vec![0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], // [gnd, p1, L1, p2, L2, p3, R1, C10, L1, L2, R1]
            node_map: vec![0, 1, 3, 5, 8],              // [0, 1, 2, 3, 4]
            port_map: vec![0, 1, 3, 5],                 // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.names(), &map2.names());
        assert_eq!(&exemplar.elem_map(), &map2.elem_map());
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());

        map2.add(0, 1, "C10".to_string());
        let exemplar = CircuitMap {
            names: vec![
                "gnd".to_string(),
                "C10".to_string(),
                "p1".to_string(),
                "L1".to_string(),
                "p2".to_string(),
                "L2".to_string(),
                "p3".to_string(),
                "R1".to_string(),
                "C10".to_string(),
                "L1".to_string(),
                "L2".to_string(),
                "R1".to_string(),
            ],
            elem_map: vec![
                ("gnd".to_string(), 0),
                ("C10".to_string(), 1),
                ("p1".to_string(), 0),
                ("L1".to_string(), 0),
                ("p2".to_string(), 0),
                ("L2".to_string(), 1),
                ("p3".to_string(), 0),
                ("R1".to_string(), 1),
                ("C10".to_string(), 0),
                ("L1".to_string(), 1),
                ("L2".to_string(), 0),
                ("R1".to_string(), 0),
            ],
            map: vec![0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], // [gnd, C10, p1, L1, p2, L2, p3, R1, C10, L1, L2, R1]
            node_map: vec![0, 2, 4, 6, 9],                 // [0, 1, 2, 3, 4]
            port_map: vec![0, 2, 4, 6],                    // [gnd, p1, p2, p3]
        };
        assert_eq!(&exemplar.names(), &map2.names());
        assert_eq!(&exemplar.elem_map(), &map2.elem_map());
        assert_eq!(&exemplar.map(), &map2.map());
        assert_eq!(&exemplar.node_map(), &map2.node_map());
        assert_eq!(&exemplar.port_map(), &map2.port_map());
    }

    #[test]
    fn circuitmap3() {
        let mut map = CircuitMap::new();
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string()],
            elem_map: vec![("gnd".to_string(), 0)],
            map: vec![0],      // [gnd]
            node_map: vec![0], // [0]
            port_map: vec![0], // [gnd]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add_port(4, "p4".to_string());
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string(), "p4".to_string()],
            elem_map: vec![("gnd".to_string(), 0), ("p4".to_string(), 0)],
            map: vec![0, 0],               // [gnd, p4]
            node_map: vec![0, 0, 0, 0, 1], // [0, X, X, X, 4]
            port_map: vec![0, 1],          // [gnd, p4]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add_port(2, "p2".to_string());
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string(), "p2".to_string()],
            elem_map: vec![("gnd".to_string(), 0), ("p2".to_string(), 0)],
            map: vec![0, 0, 0],            // [gnd, p2, p4]
            node_map: vec![0, 0, 1, 0, 2], // [0, X, 2, X, 4]
            port_map: vec![0, 1, 2],       // [gnd, p2, p4]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());

        map.add_port(1, "p1".to_string());
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string(), "p1".to_string()],
            elem_map: vec![("gnd".to_string(), 0), ("p1".to_string(), 0)],
            map: vec![0, 0, 0, 0],         // [gnd, p1, p2, p4]
            node_map: vec![0, 1, 2, 0, 3], // [0, 1, 2, X, 3]
            port_map: vec![0, 1, 2, 3],    // [gnd, p1, p2, p4]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());
    }

    #[test]
    fn circuit_r() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val_scaled(20.0, Scale::Base)
            .nodes(vec![1, 2])
            .id("R1")
            .build()
            .unwrap();
        let margin = MARGIN;

        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        cir.add_elem(&p1, &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(1)");

        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = Points::<Complex64, Ix3>::from_shape_fn((1, 3, 3), |(_, j, k)| {
            if j == k && j != 0 {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        cir.add_elem(&p2, &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(2)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(2)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(2)");

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(0.9035079029052513, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.9035079029052513, 0.0),
                    c64(0.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(0.9035079029052513, 0.0),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.9035079029052513, 0.0),
                    c64(0.4285714285714284, 0.0),
                ]
            ]
        ];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => 0.1666666666666659.into(),
                (0, 1) | (1, 0) => 0.8333333333333326.into(),
                _ => Complex64::ZERO,
            });
        cir.add_elem(&r1, &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(3)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(3)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(3)");

        let mut cir2 = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(10.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val_scaled(20.0, Scale::Base)
            .nodes(vec![1, 2])
            .id("R1")
            .build()
            .unwrap();

        let exemplar2_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar2_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar2_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        cir2.add_elem(&p1, &freq);
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(1)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(1)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(1)");

        let exemplar2_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar2_x = Points::<Complex64, Ix3>::from_shape_fn((1, 3, 3), |(_, j, k)| {
            if j == k && j != 0 {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        let exemplar2_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        cir2.add_elem(&p2, &freq);
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(2)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(2)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(2)");

        let exemplar2_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar2_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(0.9035079029052513, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.9035079029052513, 0.0),
                    c64(0.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.33333333333333304, 0.0),
                    c64(0.9428090415820632, 0.0),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.9428090415820632, 0.0),
                    c64(-0.3333333333333335, 0.0),
                ]
            ]
        ];
        let exemplar2_s = points![
            Complex64,
            [
                [c64(-0.25, 0.0), c64(0.5590169943749473, 0.0)],
                [c64(0.5590169943749473, 0.0), c64(0.75, 0.0)]
            ]
        ];
        cir2.add_elem(&r1, &freq);
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(3)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(3)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(3)");
    }

    #[test]
    fn circuit_c() {
        let margin = MARGIN;
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();

        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 2])
            .id("C1")
            .build()
            .unwrap();

        cir.add_elem(&p1, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(1)");

        cir.add_elem(&p2, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(2)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(2)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(2)");

        cir.add_elem(&c1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.8203396752925509, -0.5718765750937107),
                    c64(0.9481135966932567, 0.4948067885071891),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.9481135966932567, 0.4948067885071891),
                    c64(-0.8203396752925507, 0.5718765750937107),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.8203396752925509, -0.5718765750937107),
                    c64(0.9481135966932567, 0.4948067885071891),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.9481135966932567, 0.4948067885071891),
                    c64(-0.8203396752925507, 0.5718765750937107),
                ]
            ]
        ];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.7169568003248977, -0.4504772433683886),
                (0, 1) | (1, 0) => c64(0.2830431996751022, 0.4504772433683886),
                _ => Complex64::ZERO,
            });
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(3)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(3)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(3)");

        let mut cir2 = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(10.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 2])
            .id("C1")
            .build()
            .unwrap();

        cir2.add_elem(&p1, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(1)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(1)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(1)");

        cir2.add_elem(&p2, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(2)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(2)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(2)");

        cir2.add_elem(&c1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.8203396752925509, -0.5718765750937107),
                    c64(0.9481135966932567, 0.4948067885071891),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.9481135966932567, 0.4948067885071891),
                    c64(-0.8203396752925507, 0.5718765750937107),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.9921353648143452, -0.1251695565411434),
                    c64(0.37528252613977364, 0.3309110736382766),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.37528252613977364, 0.3309110736382766),
                    c64(-0.992135364814345, 0.1251695565411434),
                ]
            ]
        ];
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(0.7926049557687088, -0.5501324410362041),
                    c64(0.0927498834195485, 0.2460267069569694)
                ],
                [
                    c64(0.0927498834195484, 0.2460267069569693),
                    c64(0.9585209911537415, -0.1100264882072408)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(3)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(3)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(3)");
    }

    #[test]
    fn circuit_l() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();

        cir.add_elem(&p1, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(1)");

        cir.add_elem(&p2, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(2)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(2)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(2)");

        cir.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.9689082471969974, 0.24742030739945778),
                    c64(0.5555511820823532, 0.43151303443327826),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.5555511820823532, 0.43151303443327826),
                    c64(0.9689082471969976, -0.24742030739945778),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.9689082471969974, 0.24742030739945778),
                    c64(0.5555511820823532, 0.43151303443327826)
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5555511820823532, 0.43151303443327826),
                    c64(0.9689082471969976, -0.24742030739945778)
                ]
            ]
        ];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.0039323175928275, 0.0625847782705717),
                (0, 1) | (1, 0) => c64(0.9960676824071726, -0.0625847782705717),
                _ => Complex64::ZERO,
            });
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(3)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(3)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(3)");

        let mut cir2 = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(10.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();

        cir2.add_elem(&p1, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(1)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(1)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(1)");

        cir2.add_elem(&p2, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(2)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(2)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(2)");

        cir2.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.9689082471969974, 0.24742030739945778),
                    c64(0.5555511820823532, 0.43151303443327826),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.5555511820823532, 0.43151303443327826),
                    c64(0.9689082471969976, -0.24742030739945778),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.43391360064979534, 0.9009544867367774),
                    c64(1.3086915121249574, 0.298723115218169)
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(1.3086915121249574, 0.298723115218169),
                    c64(0.43391360064979545, -0.9009544867367774)
                ]
            ]
        ];
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.6485878775864338, 0.172639718834091),
                    c64(0.7372709122330735, -0.0772068293858956),
                ],
                [
                    c64(0.7372709122330735, -0.0772068293858956),
                    c64(0.6702824244827131, 0.0345279437668182),
                ],
            ]
        ];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(3)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(3)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(3)");
    }

    #[test]
    fn circuit_shunt_l() {
        let margin = MARGIN;
        let z0 = 50.0;
        let freq = FrequencyBuilder::new()
            .freqs_scaled(array![275.0], Scale::Giga)
            .build();
        let src = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(92.8568)
            .im(0.0)
            .z0(z0)
            .freq(freq.unitval(0))
            .build();
        let load = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(18.9383)
            .im(0.0)
            .z0(z0)
            .freq(freq.unitval(0))
            .build();
        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(src.rp().val().into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(load.rp().val().into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let short = ElementBuilder::new()
            .elem(ElemType::Short)
            .nodes(vec![1, 2])
            .id("S0")
            .build()
            .unwrap();
        let ls = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(5.0, Scale::Pico)
            .nodes(vec![2, 0])
            .id("Ls")
            .build()
            .unwrap();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.9214835975613086, 0.14295557061866743),
                    c64(0.1738588301058095, 0.3165464475308452)
                ],
                [
                    c64(0.1738588301058095, 0.3165464475308452),
                    c64(-0.6150244806572353, 0.7009286382528254)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(1)");

        let src = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(9.4595)
            .im(-28.0873)
            .z0(z0)
            .freq(freq.unitval(0))
            .build();
        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(src.rp().val().into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(load.rp().val().into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let cp1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val(src.cp().val().into())
            .scale(Scale::Femto)
            .nodes(vec![1, 0])
            .id("Cp1")
            .build()
            .unwrap();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&cp1, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.8761891493750931, 0.1631491116557464),
                    c64(0.27415425099996193, 0.3612609256905665)
                ],
                [
                    c64(0.27415425099996193, 0.3612609256905665),
                    c64(-0.3929404978481734, 0.7999397306323501)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(2)");

        let load = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(16.923)
            .im(-5.84)
            .z0(z0)
            .freq(freq.unitval(0))
            .build();
        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(src.rp().val().into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(load.rp().val().into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let cp1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val(src.cp().val().into())
            .scale(Scale::Femto)
            .nodes(vec![1, 0])
            .id("Cp1")
            .build()
            .unwrap();
        let cp2 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val(load.cp().val().into())
            .scale(Scale::Femto)
            .nodes(vec![2, 0])
            .id("Cp2")
            .build()
            .unwrap();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&cp1, &freq);
        cir.add_elem(&cp2, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.8357881602107229, 0.1693191357519407),
                    c64(0.36361373893745236, 0.3749228076576333)
                ],
                [
                    c64(0.36361373893745236, 0.3749228076576333),
                    c64(-0.1948512889585977, 0.8301903448634361)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(3)");

        let mut cir = Circuit::new(&freq);
        let ls = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(5.0, Scale::Pico)
            .q(QBuilder::new()
                .q(15.0)
                .fq(freq.clone())
                .mode(QMode::ProportionalToFreq)
                .build())
            .nodes(vec![2, 0])
            .id("Ls")
            .build()
            .unwrap();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&cp1, &freq);
        cir.add_elem(&cp2, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.835105153421519, 0.15050653480106269),
                    c64(0.36512611863346456, 0.33326612699645153)
                ],
                [
                    c64(0.36512611863346456, 0.33326612699645153),
                    c64(-0.19150243155181312, 0.7379500933299726)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(4)");

        let mut cir = Circuit::new(&freq);
        let ls = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(11.4782, Scale::Pico)
            .q(QBuilder::new()
                .q(15.0)
                .fq(freq.clone())
                .mode(QMode::ProportionalToFreq)
                .build())
            .nodes(vec![2, 0])
            .id("Ls")
            .build()
            .unwrap();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&cp1, &freq);
        cir.add_elem(&cp2, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.6781331823072998, -0.00000023520389949955467),
                    c64(0.712708640079339, -0.0000005208112242055668)
                ],
                [
                    c64(0.712708640079339, -0.0000005208112242055668),
                    c64(0.5781484071113706, -0.0000011532305877395315)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(5)");
    }

    #[test]
    fn circuit_rc() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 2])
            .id("C1")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val_scaled(20.0, Scale::Base)
            .nodes(vec![1, 2])
            .id("R1")
            .build()
            .unwrap();

        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        cir.add_elem(&p1, &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(p1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(p1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(p1)");

        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = Points::<Complex64, Ix3>::from_shape_fn((1, 3, 3), |(_, j, k)| {
            if j == k && j != 0 {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        cir.add_elem(&p2, &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(p2)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(p2)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(p2)");

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(0.9035079029052513, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.9035079029052513, 0.0),
                    c64(0.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(0.9035079029052513, 0.0),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.9035079029052513, 0.0),
                    c64(0.4285714285714284, 0.0),
                ]
            ]
        ];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => 0.1666666666666659.into(),
                (0, 1) | (1, 0) => 0.8333333333333326.into(),
                _ => Complex64::ZERO,
            });
        cir.add_elem(&r1, &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(r1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(r1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(r1)");

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.4331385293595438, -0.050881366621918944),
                    c64(0.8962866825082543, -0.08045050449366568),
                    c64(0.24483170498225598, 0.20449980312782579),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.8962866825082543, -0.08045050449366568),
                    c64(0.4171536766011408, -0.12720341655479742),
                    c64(0.3871129155831617, 0.32334257946997763),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.24483170498225598, 0.20449980312782579),
                    c64(0.3871129155831617, 0.32334257946997763),
                    c64(-0.984015147241597, 0.1780847831767164),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.4331385293595438, -0.050881366621918944),
                    c64(0.8962866825082543, -0.08045050449366568),
                    c64(0.24483170498225598, 0.20449980312782579),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.8962866825082543, -0.08045050449366568),
                    c64(0.4171536766011408, -0.12720341655479742),
                    c64(0.3871129155831617, 0.32334257946997763),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.24483170498225598, 0.20449980312782579),
                    c64(0.3871129155831617, 0.32334257946997763),
                    c64(-0.984015147241597, 0.1780847831767164),
                ],
            ]
        ];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.1648587877586434, -0.0172639718834091),
                (0, 1) | (1, 0) => c64(0.8351412122413568, 0.0172639718834091),
                _ => Complex64::ZERO,
            });
        cir.add_elem(&c1, &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(c1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(c1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(c1)");

        let mut cir2 = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(10.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 2])
            .id("C1")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val_scaled(20.0, Scale::Base)
            .nodes(vec![1, 2])
            .id("R1")
            .build()
            .unwrap();

        let exemplar2_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar2_x = points![
            Complex64,
            [
                [Complex64::ZERO, Complex64::ZERO],
                [Complex64::ZERO, Complex64::ONE]
            ]
        ];
        let exemplar2_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        cir2.add_elem(&p1, &freq);
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(1)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(1)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(1)");

        let exemplar2_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar2_x = Points::<Complex64, Ix3>::from_shape_fn((1, 3, 3), |(_, j, k)| {
            if j == k && j != 0 {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        let exemplar2_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        cir2.add_elem(&p2, &freq);
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(2)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(2)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(2)");

        let exemplar2_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar2_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.4285714285714286, 0.0),
                    c64(0.9035079029052513, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.9035079029052513, 0.0),
                    c64(0.4285714285714284, 0.0),
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.33333333333333304, 0.0),
                    c64(0.9428090415820632, 0.0),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.9428090415820632, 0.0),
                    c64(-0.3333333333333335, 0.0),
                ]
            ]
        ];
        let exemplar2_s = points![
            Complex64,
            [
                [c64(-0.25, 0.0), c64(0.5590169943749473, 0.0)],
                [c64(0.5590169943749473, 0.0), c64(0.75, 0.0)]
            ]
        ];
        cir2.add_elem(&r1, &freq);
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(3)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(3)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(3)");

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.4331385293595438, -0.050881366621918944),
                    c64(0.8962866825082543, -0.08045050449366568),
                    c64(0.24483170498225598, 0.20449980312782579),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.8962866825082543, -0.08045050449366568),
                    c64(0.4171536766011408, -0.12720341655479742),
                    c64(0.3871129155831617, 0.32334257946997763),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.24483170498225598, 0.20449980312782579),
                    c64(0.3871129155831617, 0.32334257946997763),
                    c64(-0.984015147241597, 0.1780847831767164),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.3309979691707785, -0.055752712558531356),
                    c64(0.9411576897461811, -0.03942312111968192),
                    c64(0.24579515860769652, 0.22603133659313707),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.9411576897461811, -0.03942312111968192),
                    c64(-0.33450101541461075, -0.027876356279265678),
                    c64(0.17380342343432525, 0.15982829086566624),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.24579515860769652, 0.22603133659313707),
                    c64(0.17380342343432525, 0.15982829086566624),
                    c64(-0.9964969537561683, 0.08362906883779705),
                ],
            ]
        ];
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.253668515533063, -0.0389241587264249),
                    c64(0.5606576043966359, 0.0174074129758555)
                ],
                [
                    c64(0.5606576043966359, 0.0174074129758555),
                    c64(0.7492662968933874, -0.007784831745285)
                ]
            ]
        ];
        cir2.add_elem(&c1, &freq);
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(c1)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(c1)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(c1)");
    }

    #[test]
    fn circuit_rl() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val_scaled(20.0, Scale::Base)
            .nodes(vec![1, 2])
            .id("R1")
            .build()
            .unwrap();

        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&r1, &freq);
        cir.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.9073776846815567, 0.2105899903363776),
                    c64(0.14644873928229243, 0.33297201094790096),
                    c64(0.6048210433185525, 0.23531146642654904),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.14644873928229243, 0.33297201094790096),
                    c64(-0.7684442117038917, 0.526474975840944),
                    c64(0.9563060368429952, 0.3720600967310689),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.6048210433185525, 0.23531146642654904),
                    c64(0.9563060368429952, 0.3720600967310689),
                    c64(0.6758218963854483, -0.7370649661773218),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.9073776846815567, 0.2105899903363776),
                    c64(0.14644873928229243, 0.33297201094790096),
                    c64(0.6048210433185525, 0.23531146642654904),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.14644873928229243, 0.33297201094790096),
                    c64(-0.7684442117038917, 0.526474975840944),
                    c64(0.9563060368429952, 0.3720600967310689),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.6048210433185525, 0.23531146642654904),
                    c64(0.9563060368429952, 0.3720600967310689),
                    c64(0.6758218963854483, -0.7370649661773218),
                ]
            ]
        ];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.0207395044231295, 0.0550132441036204),
                (0, 1) | (1, 0) => c64(0.9792604955768716, -0.0550132441036205),
                _ => Complex64::ZERO,
            });
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(l1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(l1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(l1)");

        let mut cir2 = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(10.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val_scaled(20.0, Scale::Base)
            .nodes(vec![1, 2])
            .id("R1")
            .build()
            .unwrap();

        cir2.add_elem(&p1, &freq);
        cir2.add_elem(&p2, &freq);
        cir2.add_elem(&r1, &freq);
        cir2.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.9073776846815567, 0.2105899903363776),
                    c64(0.14644873928229243, 0.33297201094790096),
                    c64(0.6048210433185525, 0.23531146642654904),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.14644873928229243, 0.33297201094790096),
                    c64(-0.7684442117038917, 0.526474975840944),
                    c64(0.9563060368429952, 0.3720600967310689),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.6048210433185525, 0.23531146642654904),
                    c64(0.9563060368429952, 0.3720600967310689),
                    c64(0.6758218963854483, -0.7370649661773218),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.3727824712587392, 0.6654984672870303),
                    c64(0.44350976785201374, 0.4705784790879129),
                    c64(1.153182891925262, 0.03414897282423488),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.44350976785201374, 0.4705784790879129),
                    c64(-0.6863912356293695, 0.33274923364351516),
                    c64(0.8154234428286663, 0.024146970254571586),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(1.153182891925262, 0.03414897282423488),
                    c64(0.8154234428286663, 0.024146970254571586),
                    c64(0.05917370688810886, -0.9982477009305457),
                ]
            ]
        ];
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.6044712678228586, 0.1484805774534605),
                    c64(0.7175413645594365, -0.0664025329048721)
                ],
                [
                    c64(0.7175413645594366, -0.066402532904872),
                    c64(0.6791057464354284, 0.0296961154906921)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(l1)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(l1)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(l1)");
    }

    #[test]
    fn circuit_lc() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 2])
            .id("C1")
            .build()
            .unwrap();

        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&c1, &freq);
        cir.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.966343811726842, 0.25725403308255007),
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(0.5802819354986586, 0.44601317050552247),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(-1.0808187380438485, 0.010573403382678814),
                    c64(0.05321510615131428, 0.4067543409025826),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    c64(0.5802819354986586, 0.44601317050552247),
                    c64(0.05321510615131428, 0.4067543409025826),
                    c64(1.0471625497706905, -0.26782743646522883),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.966343811726842, 0.25725403308255007),
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(0.5802819354986586, 0.44601317050552247),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(-1.0808187380438485, 0.010573403382678814),
                    c64(0.05321510615131428, 0.4067543409025826),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5802819354986586, 0.44601317050552247),
                    c64(0.05321510615131428, 0.4067543409025826),
                    c64(1.0471625497706905, -0.26782743646522883),
                ]
            ]
        ];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.0042607993839932, 0.0651355891399034),
                (0, 1) | (1, 0) => c64(0.9957392006160075, -0.0651355891399034),
                _ => Complex64::ZERO,
            });
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");

        let mut cir2 = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 0])
            .id("C1")
            .build()
            .unwrap();

        cir2.add_elem(&p1, &freq);
        cir2.add_elem(&p2, &freq);
        cir2.add_elem(&c1, &freq);
        cir2.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (1, 1) | (3, 3) | (4, 4) | (6, 6) => 0.3333333333333333.into(),
                (1, 3) | (4, 6) | (3, 1) | (6, 4) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    -Complex64::ONE,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    -Complex64::ONE,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.966343811726842, 0.25725403308255007),
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(0.5802819354986586, 0.44601317050552247),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(-1.0808187380438485, 0.010573403382678814),
                    c64(0.05321510615131428, 0.4067543409025826),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5802819354986586, 0.44601317050552247),
                    c64(0.05321510615131428, 0.4067543409025826),
                    c64(1.0471625497706905, -0.26782743646522883),
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.9689082471969974, 0.24742030739945778),
                    c64(0.5555511820823532, 0.43151303443327826)
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5555511820823532, 0.43151303443327826),
                    c64(0.9689082471969976, -0.24742030739945778)
                ]
            ]
        ];
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.0013639498786209, -0.0958396298280963),
                    c64(0.9712550421795498, -0.217891138039029)
                ],
                [
                    c64(0.9712550421795499, -0.2178911380390291),
                    c64(-0.0397075620341229, -0.0872376324883029)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c()");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x()");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s()");

        let mut cir3 = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![2, 0])
            .id("C1")
            .build()
            .unwrap();

        cir3.add_elem(&p1, &freq);
        cir3.add_elem(&p2, &freq);
        cir3.add_elem(&c1, &freq);
        cir3.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (1, 1) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (1, 5) | (3, 6) | (5, 1) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    -Complex64::ONE,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    -Complex64::ONE,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.9689082471969974, 0.24742030739945778),
                    c64(0.5555511820823532, 0.43151303443327826),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5555511820823532, 0.43151303443327826),
                    c64(0.9689082471969976, -0.24742030739945778),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.966343811726842, 0.25725403308255007),
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(0.5802819354986586, 0.44601317050552247),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(-1.0808187380438485, 0.010573403382678814),
                    c64(0.05321510615131428, 0.4067543409025826),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5802819354986586, 0.44601317050552247),
                    c64(0.05321510615131428, 0.4067543409025826),
                    c64(1.0471625497706905, -0.26782743646522883),
                ]
            ]
        ];
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.0397075620341229, -0.0872376324883029),
                    c64(0.9712550421795499, -0.2178911380390291)
                ],
                [
                    c64(0.9712550421795498, -0.217891138039029),
                    c64(-0.0013639498786209, -0.0958396298280963)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_c, &cir3.c, margin, "cir3.c()");
        comp_pts_ix3(&exemplar_x, &cir3.x, margin, "cir3.x()");
        comp_pts_ix3(&exemplar_s, &cir3.s, margin, "cir3.s()");

        let mut cir4 = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![2, 0])
            .id("L1")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 2])
            .id("C1")
            .build()
            .unwrap();

        cir4.add_elem(&p1, &freq);
        cir4.add_elem(&p2, &freq);
        cir4.add_elem(&c1, &freq);
        cir4.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (1, 1) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (1, 6) | (3, 5) | (6, 1) | (5, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    -Complex64::ONE,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    -Complex64::ONE,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.8203396752925509, -0.5718765750937107),
                    c64(0.9481135966932567, 0.4948067885071891),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.9481135966932567, 0.4948067885071891),
                    c64(-0.8203396752925507, 0.5718765750937107),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.966343811726842, 0.25725403308255007),
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(0.5802819354986586, 0.44601317050552247),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.08861914420189358, 0.11529724214516848),
                    c64(-1.0808187380438485, 0.010573403382678814),
                    c64(0.05321510615131428, 0.4067543409025826),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.5802819354986586, 0.44601317050552247),
                    c64(0.05321510615131428, 0.4067543409025826),
                    c64(1.0471625497706905, -0.26782743646522883),
                ]
            ]
        ];
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(0.804537187163662, -0.5888426474354541),
                    c64(-0.0698071736897519, 0.033334809635529)
                ],
                [
                    c64(-0.0698071736897518, 0.033334809635529),
                    c64(-0.9636991790793161, 0.2555379447554992)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_c, &cir4.c, margin, "cir4.c()");
        comp_pts_ix3(&exemplar_x, &cir4.x, margin, "cir4.x()");
        comp_pts_ix3(&exemplar_s, &cir4.s, margin, "cir4.s()");
    }

    #[test]
    fn circuit_rlc() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(c64(24.4, 0.0))
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![1, 2])
            .id("C1")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 2])
            .id("L1")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val_scaled(20.0, Scale::Base)
            .nodes(vec![1, 2])
            .id("R1")
            .build()
            .unwrap();

        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&r1, &freq);
        cir.add_elem(&c1, &freq);
        cir.add_elem(&l1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 9, 9), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (4, 4) | (6, 6) | (7, 7) | (8, 8) => 0.3333333333333333.into(),
                (2, 6) | (3, 7) | (4, 8) | (6, 2) | (7, 3) | (8, 4) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![
            Complex64,
            [
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    c64(-0.9009542853970355, 0.2163041784628353),
                    c64(0.15660502531218384, 0.3420069356770492),
                    c64(-0.04647336706911738, 0.12498348394848209),
                    c64(0.6290320273200906, 0.2338967948431292),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    c64(0.15660502531218384, 0.3420069356770492),
                    c64(-0.7523857134925888, 0.5407604461570882),
                    c64(-0.07348084523773735, 0.1976162395901491),
                    c64(0.9945869637623739, 0.3698233045587068),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    c64(-0.04647336706911738, 0.12498348394848209),
                    c64(-0.07348084523773735, 0.1976162395901491),
                    c64(-1.0679539617999618, 0.031116128936622463),
                    c64(0.15660502531218382, 0.34200693567704915),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    c64(0.6290320273200906, 0.2338967948431292),
                    c64(0.9945869637623739, 0.3698233045587068),
                    c64(0.15660502531218382, 0.34200693567704915),
                    c64(0.7212939606895861, -0.788180753556546),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.7643540849057923, 0.395935120823352),
                    c64(0.2602794204440981, 0.437324634887758),
                    c64(-0.04437865404125869, 0.17486340495920796),
                    c64(0.8800737397504786, 0.2233542692154159),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.2602794204440981, 0.437324634887758),
                    c64(-0.7125119835850666, 0.48304084740448944),
                    c64(-0.04901782553425585, 0.1931429436469065),
                    c64(0.9720732177290143, 0.24670285382143028),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.04437865404125869, 0.17486340495920796),
                    c64(-0.04901782553425585, 0.1931429436469065),
                    c64(-1.0607007031035893, 0.03612680961457026),
                    c64(0.1818233863750128, 0.305501856139204),
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.8800737397504786, 0.2233542692154159),
                    c64(0.9720732177290143, 0.24670285382143028),
                    c64(0.1818233863750128, 0.305501856139204),
                    c64(0.5375667715944483, -0.9151027778424118),
                ]
            ]
        ];
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-0.3022491170600611, 0.1008132787157404),
                    c64(0.9097121238541618, -0.0704251288726103)
                ],
                [
                    c64(0.9097121238541618, -0.0704251288726103),
                    c64(0.36450243087469, 0.0491968800132813)
                ]
            ]
        ];
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
    }

    #[test]
    fn circuit_wilkinson() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let margin = F64Margin {
            epsilon: 1e-4,
            ulps: 10,
        };

        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let p3 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![3])
            .id("p3")
            .build()
            .unwrap();
        let r1 = ElementBuilder::new()
            .elem(ElemType::Resistor)
            .unitval_val(100.0)
            .nodes(vec![2, 3])
            .id("R1")
            .build()
            .unwrap();
        let sub = MsubBuilder::new()
            .id("Msub0")
            .er(12.4)
            .tand(0.0004)
            .height(
                UnitValBuilder::new()
                    .val(25e-6)
                    .scale(Scale::Micro)
                    .unit(Unit::Meter)
                    .build(),
            )
            .thickness(
                UnitValBuilder::new()
                    .val(0.77e-6)
                    .scale(Scale::Micro)
                    .unit(Unit::Meter)
                    .build(),
            )
            .build();
        let gamma = c64(
            0.0,
            2.0 * std::f64::consts::PI / (3e8 / (1e9 * 12.4_f64.sqrt())),
        );
        let width = 5.6758e-6;
        let ml1 = ElementBuilder::new()
            .elem(ElemType::Mlin)
            .width_val(width)
            .width_scale(Scale::Micro)
            .width_unit(Unit::Meter)
            .length_val(0.25)
            .length_unit(Unit::Lambda)
            .gamma(gamma)
            .sub(&sub)
            .nodes(vec![1, 2])
            .id("ML1")
            .build()
            .unwrap();
        let ml2 = ElementBuilder::new()
            .elem(ElemType::Mlin)
            .width_val(width)
            .width_scale(Scale::Micro)
            .width_unit(Unit::Meter)
            .length_val(0.25)
            .length_unit(Unit::Lambda)
            .gamma(gamma)
            .sub(&sub)
            .nodes(vec![1, 3])
            .id("ML2")
            .build()
            .unwrap();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&p3, &freq);
        cir.add_elem(&ml1, &freq);
        cir.add_elem(&ml2, &freq);
        cir.add_elem(&r1, &freq);
        let exemplar_s = points![
            Complex64,
            [
                [
                    c64(-1.3112230673462122e-07, -5.685569916314089e-17),
                    c64(-1.2060919872574568e-16, -7.071067811865412e-01),
                    c64(-1.2060919872574566e-16, -7.071067811865414e-01)
                ],
                [
                    c64(-1.2060919872574566e-16, -7.071067811865414e-01),
                    c64(-6.5561153783644244e-08, 7.455052910997203e-24),
                    c64(-6.5561153621858092e-08, 5.685573643840818e-17)
                ],
                [
                    c64(-1.2060919872574566e-16, -7.071067811865414e-01),
                    c64(-6.5561153662869508e-08, 5.685573643840819e-17),
                    c64(-6.5561153783644244e-08, 7.455052905639125e-24)
                ]
            ]
        ];
        let exemplar_x =
            Points::<Complex64, Ix3>::from_shape_fn((1, 10, 10), |(_, j, k)| match (j, k) {
                (1, 1) => (-0.17157293888502256).into(),
                (2, 2) | (3, 3) => (-0.41421353055748866).into(),
                (1, 2) | (1, 3) | (2, 1) | (3, 1) => (0.6966213916620551).into(),
                (2, 3) | (3, 2) => (0.5857864694425113).into(),
                (4, 4) | (7, 7) => (-0.09383635942272539).into(),
                (5, 5) | (8, 8) => (-0.3592454608659119).into(),
                (6, 6) | (9, 9) => (-0.5469181797113627).into(),
                (4, 5) | (5, 4) | (7, 8) | (8, 7) => (0.7619898069516147).into(),
                (4, 6) | (6, 4) | (7, 9) | (9, 7) => (0.6407544551168801).into(),
                (5, 6) | (6, 5) | (8, 9) | (9, 8) => (0.538808159690515).into(),
                _ => Complex64::ZERO,
            });
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 10, 10), |(_, j, k)| match (j, k) {
                (2, 5) | (3, 8) | (5, 2) | (8, 3) => -Complex64::I,
                (6, 6) | (9, 9) => (1.0 / 3.0).into(),
                (6, 9) | (9, 6) => (2.0 / 3.0).into(),
                _ => Complex64::ZERO,
            });

        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
    }

    #[test]
    fn circuit_tee() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = FrequencyBuilder::new()
            .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
            .build();
        let margin = F64Margin {
            epsilon: 1e-11,
            ulps: 10,
        };

        let mut cir = Circuit::new(&freq);
        let p1 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![1])
            .id("p1")
            .build()
            .unwrap();
        let p2 = ElementBuilder::new()
            .elem(ElemType::Port)
            .z(50.0.into())
            .nodes(vec![2])
            .id("p2")
            .build()
            .unwrap();
        let l1 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![1, 3])
            .id("L1")
            .build()
            .unwrap();
        let l2 = ElementBuilder::new()
            .elem(ElemType::Inductor)
            .unitval_val_scaled(1.0, Scale::Nano)
            .nodes(vec![3, 2])
            .id("L2")
            .build()
            .unwrap();
        let c1 = ElementBuilder::new()
            .elem(ElemType::Capacitor)
            .unitval_val_scaled(1.0, Scale::Pico)
            .nodes(vec![3, 0])
            .id("C1")
            .build()
            .unwrap();

        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&l1, &freq);
        cir.add_elem(&l2, &freq);
        cir.add_elem(&c1, &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 9, 9), |(_, j, k)| match (j, k) {
                (1, 1) | (3, 3) | (5, 5) | (6, 6) | (7, 7) | (8, 8) => (1.0 / 3.0).into(),
                (1, 8) | (8, 1) | (3, 6) | (6, 3) | (5, 7) | (7, 5) => (2.0 / 3.0).into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x =
            Points::<Complex64, Ix3>::from_shape_fn((1, 9, 9), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => (-1.0).into(),
                (2, 2) | (4, 4) => c64(-0.9689082471969974, 0.24742030739945778),
                (3, 3) | (5, 5) => c64(0.9689082471969976, -0.24742030739945778),
                (2, 3) | (3, 2) | (4, 5) | (5, 4) => c64(0.5555511820823531, 0.43151303443327826),
                (6, 6) | (7, 7) => (0.02013669115344152).into(),
                (8, 8) => (-1.04027338230688).into(),
                (6, 7) | (7, 6) => (1.0201366911534415).into(),
                (6, 8) | (8, 6) | (7, 8) | (8, 7) => c64(0.0, 0.20269276002882086),
                _ => Complex64::ZERO,
            });
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(-0.0094890066888926, -0.0325208858508174),
                (0, 1) | (1, 0) => c64(0.9594192405081047, -0.2799411932502752),
                _ => Complex64::ZERO,
            });
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
    }

    mod xfmr_tests {
        use super::*;

        #[test]
        fn circuit_xfmr_main() {
            let z0 = 50.0;
            let freq_points: Vec<f64> = vec![275.0];
            let freq = FrequencyBuilder::new()
                .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
                .build();
            let src = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(9.4595)
                .im(-28.0873)
                .z0(z0)
                .freq(freq.unitval(0))
                .build();
            let load = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(16.923)
                .im(-5.84)
                .z0(z0)
                .freq(freq.unitval(0))
                .build();
            let l_scale = Scale::Pico;
            let km = 0.4;
            // let lp_ads = 18.7193;
            // let ls_ads = 11.3005;
            let lp = 18.7193;
            let ls = 11.3005;
            let qp = None;
            let qs = None;
            let exemplar_sparms = Points(array![
                [c64(0.8521, -0.0492), c64(0.2451, -0.4598)],
                [c64(0.2451, -0.4598), c64(0.4342, 0.7349)]
            ]);
            let margin = MARGIN;
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                20.0 * exemplar_sparms[(0, 0)].norm().log10(),
                20.0 * exemplar_sparms[(0, 1)].norm().log10(),
                20.0 * exemplar_sparms[(1, 0)].norm().log10(),
                20.0 * exemplar_sparms[(1, 1)].norm().log10()
            );

            let mut cir = Circuit::new(&freq);
            let n2 = lp / ls;
            let mut load2 = load.clone();
            load2.set_z(n2 * load.z());
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Port)
                    .z(src.rp().val().into())
                    .nodes(vec![1])
                    .id("p1")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Port)
                    .z(load.rp().val().into())
                    .nodes(vec![2])
                    .id("p2")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Capacitor)
                    .unitval(src.cp())
                    .nodes(vec![1, 0])
                    .id("Cp1")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Capacitor)
                    .unitval(load.cp())
                    .nodes(vec![2, 0])
                    .id("Cp2")
                    .build()
                    .unwrap(),
                &freq,
            );
            match qp {
                Some(qp) => cir.add_elem(
                    &ElementBuilder::new()
                        .elem(ElemType::Inductor)
                        .unitval_val_scaled((1.0 - km) * lp, l_scale)
                        .q(QBuilder::new().q(qp).build())
                        .nodes(vec![1, 3])
                        .id("Lp")
                        .build()
                        .unwrap(),
                    &freq,
                ),
                None => cir.add_elem(
                    &ElementBuilder::new()
                        .elem(ElemType::Inductor)
                        .unitval_val_scaled((1.0 - km) * lp, l_scale)
                        .nodes(vec![1, 3])
                        .id("Lp")
                        .build()
                        .unwrap(),
                    &freq,
                ),
            }
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Inductor)
                    .unitval_val_scaled(km * lp, l_scale)
                    .nodes(vec![3, 0])
                    .id("M")
                    .build()
                    .unwrap(),
                &freq,
            );
            match qs {
                Some(qs) => cir.add_elem(
                    &ElementBuilder::new()
                        .elem(ElemType::Inductor)
                        .unitval_val_scaled(n2 * (1.0 - km) * ls, l_scale)
                        .q(QBuilder::new().q(qs).build())
                        .nodes(vec![3, 2])
                        .id("Ls")
                        .build()
                        .unwrap(),
                    &freq,
                ),
                None => cir.add_elem(
                    &ElementBuilder::new()
                        .elem(ElemType::Inductor)
                        .unitval_val_scaled(n2 * (1.0 - km) * ls, l_scale)
                        .nodes(vec![3, 2])
                        .id("Ls")
                        .build()
                        .unwrap(),
                    &freq,
                ),
            }

            let exemplar_c =
                Points::<Complex64, Ix3>::from_shape_fn((1, 13, 13), |(_, j, k)| match (j, k) {
                    (2, 2) | (3, 3) | (4, 4) | (6, 6) | (7, 7) | (8, 8) => {
                        0.3333333333333333.into()
                    }
                    (2, 6) | (3, 7) | (4, 8) | (6, 2) | (7, 3) | (8, 4) => {
                        0.6666666666666667.into()
                    }
                    _ => Complex64::ZERO,
                });
            let exemplar_x = points![
                Complex64,
                [
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                    ],
                    [
                        Complex64::ZERO,
                        c64(-0.9009542853970355, 0.2163041784628353),
                        c64(0.15660502531218384, 0.3420069356770492),
                        c64(-0.04647336706911738, 0.12498348394848209),
                        c64(0.6290320273200906, 0.2338967948431292),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(0.15660502531218384, 0.3420069356770492),
                        c64(-0.7523857134925888, 0.5407604461570882),
                        c64(-0.07348084523773735, 0.1976162395901491),
                        c64(0.9945869637623739, 0.3698233045587068),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(-0.04647336706911738, 0.12498348394848209),
                        c64(-0.07348084523773735, 0.1976162395901491),
                        c64(-1.0679539617999618, 0.031116128936622463),
                        c64(0.15660502531218382, 0.34200693567704915),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(0.6290320273200906, 0.2338967948431292),
                        c64(0.9945869637623739, 0.3698233045587068),
                        c64(0.15660502531218382, 0.34200693567704915),
                        c64(0.7212939606895861, -0.788180753556546),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(-0.7643540849057923, 0.395935120823352),
                        c64(0.2602794204440981, 0.437324634887758),
                        c64(-0.04437865404125869, 0.17486340495920796),
                        c64(0.8800737397504786, 0.2233542692154159),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(0.2602794204440981, 0.437324634887758),
                        c64(-0.7125119835850666, 0.48304084740448944),
                        c64(-0.04901782553425585, 0.1931429436469065),
                        c64(0.9720732177290143, 0.24670285382143028),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(-0.04437865404125869, 0.17486340495920796),
                        c64(-0.04901782553425585, 0.1931429436469065),
                        c64(-1.0607007031035893, 0.03612680961457026),
                        c64(0.1818233863750128, 0.305501856139204),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(0.8800737397504786, 0.2233542692154159),
                        c64(0.9720732177290143, 0.24670285382143028),
                        c64(0.1818233863750128, 0.305501856139204),
                        c64(0.5375667715944483, -0.9151027778424118),
                    ]
                ]
            ];
            let exemplar_s = points![
                Complex64,
                [
                    [
                        c64(-0.3022491170600611, 0.1008132787157404),
                        c64(0.9097121238541618, -0.0704251288726103)
                    ],
                    [
                        c64(0.9097121238541618, -0.0704251288726103),
                        c64(0.36450243087469, 0.0491968800132813)
                    ]
                ]
            ];
            let sparms = cir.net().s_db();
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                sparms[(0, 0, 0)],
                sparms[(0, 0, 1)],
                sparms[(0, 1, 0)],
                sparms[(0, 1, 1)]
            );
            comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
            comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
            comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
        }

        #[test]
        fn circuit_xfmr_alt1() {
            let z0 = 50.0;
            let freq_points: Vec<f64> = vec![275.0];
            let freq = FrequencyBuilder::new()
                .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
                .build();
            let src = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(9.4595)
                .im(-28.0873)
                .z0(z0)
                .freq(freq.unitval(0))
                .build();
            let load = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(16.923)
                .im(-5.84)
                .z0(z0)
                .freq(freq.unitval(0))
                .build();
            let l_scale = Scale::Pico;
            let km: f64 = 0.4;
            // let lp_ads = 18.7517;
            // let ls_ads = 11.4782;
            let lp: f64 = 18.7517;
            let ls: f64 = 11.5291;
            let _qp: Option<f64> = None;
            let _qs: Option<f64> = None;
            let exemplar_sparms = Points(array![
                [c64(0.4331, 0.1449), c64(0.6264, -0.6316)],
                [c64(0.6265, -0.6316), c64(0.1484, 0.4319)]
            ]);
            let margin = MARGIN;
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                20.0 * exemplar_sparms[(0, 0)].norm().log10(),
                20.0 * exemplar_sparms[(0, 1)].norm().log10(),
                20.0 * exemplar_sparms[(1, 0)].norm().log10(),
                20.0 * exemplar_sparms[(1, 1)].norm().log10()
            );

            let mut cir = Circuit::new(&freq);
            let _w = freq.w_pt(0);
            let n2 = lp / ls;
            let mut load2 = load.clone();
            load2.set_z(n2 * load.z());
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Port)
                    .z(src.rp().val().into())
                    .nodes(vec![1])
                    .id("p1")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Port)
                    .z(load.rp().val().into())
                    .nodes(vec![2])
                    .id("p2")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Capacitor)
                    .unitval(src.cp())
                    .nodes(vec![1, 0])
                    .id("Cp1")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Capacitor)
                    .unitval(load.cp())
                    .nodes(vec![2, 0])
                    .id("Cp2")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Transformer)
                    .freq(&freq)
                    .km(km)
                    .l1_val_scaled(lp, l_scale)
                    .l2_val_scaled(ls, l_scale)
                    .nodes(vec![1, 2])
                    .id("T0")
                    .build()
                    .unwrap(),
                &freq,
            );

            let exemplar_c =
                Points::<Complex64, Ix3>::from_shape_fn((1, 13, 13), |(_, j, k)| match (j, k) {
                    (2, 2) | (3, 3) | (4, 4) | (6, 6) | (7, 7) | (8, 8) => {
                        0.3333333333333333.into()
                    }
                    (2, 6) | (3, 7) | (4, 8) | (6, 2) | (7, 3) | (8, 4) => {
                        0.6666666666666667.into()
                    }
                    _ => Complex64::ZERO,
                });
            let exemplar_x = points![
                Complex64,
                [
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                    ],
                    [
                        Complex64::ZERO,
                        c64(-0.9009542853970355, 0.2163041784628353),
                        c64(0.15660502531218384, 0.3420069356770492),
                        c64(-0.04647336706911738, 0.12498348394848209),
                        c64(0.6290320273200906, 0.2338967948431292),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(0.15660502531218384, 0.3420069356770492),
                        c64(-0.7523857134925888, 0.5407604461570882),
                        c64(-0.07348084523773735, 0.1976162395901491),
                        c64(0.9945869637623739, 0.3698233045587068),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(-0.04647336706911738, 0.12498348394848209),
                        c64(-0.07348084523773735, 0.1976162395901491),
                        c64(-1.0679539617999618, 0.031116128936622463),
                        c64(0.15660502531218382, 0.34200693567704915),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(0.6290320273200906, 0.2338967948431292),
                        c64(0.9945869637623739, 0.3698233045587068),
                        c64(0.15660502531218382, 0.34200693567704915),
                        c64(0.7212939606895861, -0.788180753556546),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(-0.7643540849057923, 0.395935120823352),
                        c64(0.2602794204440981, 0.437324634887758),
                        c64(-0.04437865404125869, 0.17486340495920796),
                        c64(0.8800737397504786, 0.2233542692154159),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(0.2602794204440981, 0.437324634887758),
                        c64(-0.7125119835850666, 0.48304084740448944),
                        c64(-0.04901782553425585, 0.1931429436469065),
                        c64(0.9720732177290143, 0.24670285382143028),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(-0.04437865404125869, 0.17486340495920796),
                        c64(-0.04901782553425585, 0.1931429436469065),
                        c64(-1.0607007031035893, 0.03612680961457026),
                        c64(0.1818233863750128, 0.305501856139204),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(0.8800737397504786, 0.2233542692154159),
                        c64(0.9720732177290143, 0.24670285382143028),
                        c64(0.1818233863750128, 0.305501856139204),
                        c64(0.5375667715944483, -0.9151027778424118),
                    ]
                ]
            ];
            let exemplar_s = points![
                Complex64,
                [
                    [
                        c64(-0.3022491170600611, 0.1008132787157404),
                        c64(0.9097121238541618, -0.0704251288726103)
                    ],
                    [
                        c64(0.9097121238541618, -0.0704251288726103),
                        c64(0.36450243087469, 0.0491968800132813)
                    ]
                ]
            ];
            let sparms = cir.net().s_db();
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                sparms[(0, 0, 0)],
                sparms[(0, 0, 1)],
                sparms[(0, 1, 0)],
                sparms[(0, 1, 1)]
            );
            comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
            comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
            comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
        }

        #[test]
        fn circuit_xfmr_alt2() {
            let z0 = 50.0;
            let freq_points: Vec<f64> = vec![275.0];
            let freq = FrequencyBuilder::new()
                .freqs_scaled(Array1::from_vec(freq_points), Scale::Giga)
                .build();
            let src = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(9.4595)
                .im(-28.0873)
                .z0(z0)
                .freq(freq.unitval(0))
                .build();
            let load = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(16.923)
                .im(-5.84)
                .z0(z0)
                .freq(freq.unitval(0))
                .build();
            let l_scale = Scale::Pico;
            let km: f64 = 0.4;
            // let lp_ads = 18.7193;
            // let ls_ads = 11.3005;
            let lp: f64 = 18.7193;
            let ls: f64 = 11.3005;
            let _qp: Option<f64> = None;
            let _qs: Option<f64> = None;
            let exemplar_sparms = Points(array![
                [c64(0.4331, 0.1449), c64(0.6264, -0.6316)],
                [c64(0.6265, -0.6316), c64(0.1484, 0.4319)]
            ]);
            let margin = MARGIN;
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                20.0 * exemplar_sparms[(0, 0)].norm().log10(),
                20.0 * exemplar_sparms[(0, 1)].norm().log10(),
                20.0 * exemplar_sparms[(1, 0)].norm().log10(),
                20.0 * exemplar_sparms[(1, 1)].norm().log10()
            );

            let mut cir = Circuit::new(&freq);
            let _w = freq.w_pt(0);
            let n2 = lp / ls;
            let mut load2 = load.clone();
            load2.set_z(n2 * load.z());
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Port)
                    .z(src.rp().val().into())
                    .nodes(vec![1])
                    .id("p1")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Port)
                    .z(load.rp().val().into())
                    .nodes(vec![2])
                    .id("p2")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Capacitor)
                    .unitval(src.cp())
                    .nodes(vec![1, 0])
                    .id("Cp1")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Capacitor)
                    .unitval(load.cp())
                    .nodes(vec![2, 0])
                    .id("Cp2")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Inductor)
                    .unitval_val_scaled((1.0 - km) * lp, l_scale)
                    .nodes(vec![1, 3])
                    .id("Lp")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Inductor)
                    .unitval_val_scaled(km * lp, l_scale)
                    .nodes(vec![3, 0])
                    .id("M")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::IdealTransformer)
                    .n(n2.sqrt())
                    .nodes(vec![3, 4])
                    .id("T")
                    .build()
                    .unwrap(),
                &freq,
            );
            cir.add_elem(
                &ElementBuilder::new()
                    .elem(ElemType::Inductor)
                    .unitval_val_scaled((1.0 - km) * ls, l_scale)
                    .nodes(vec![4, 2])
                    .id("Ls")
                    .build()
                    .unwrap(),
                &freq,
            );

            let exemplar_c =
                Points::<Complex64, Ix3>::from_shape_fn((1, 13, 13), |(_, j, k)| match (j, k) {
                    (2, 2) | (3, 3) | (4, 4) | (6, 6) | (7, 7) | (8, 8) => {
                        0.3333333333333333.into()
                    }
                    (2, 6) | (3, 7) | (4, 8) | (6, 2) | (7, 3) | (8, 4) => {
                        0.6666666666666667.into()
                    }
                    _ => Complex64::ZERO,
                });
            let exemplar_x = points![
                Complex64,
                [
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                    ],
                    [
                        Complex64::ZERO,
                        c64(-0.9009542853970355, 0.2163041784628353),
                        c64(0.15660502531218384, 0.3420069356770492),
                        c64(-0.04647336706911738, 0.12498348394848209),
                        c64(0.6290320273200906, 0.2338967948431292),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(0.15660502531218384, 0.3420069356770492),
                        c64(-0.7523857134925888, 0.5407604461570882),
                        c64(-0.07348084523773735, 0.1976162395901491),
                        c64(0.9945869637623739, 0.3698233045587068),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(-0.04647336706911738, 0.12498348394848209),
                        c64(-0.07348084523773735, 0.1976162395901491),
                        c64(-1.0679539617999618, 0.031116128936622463),
                        c64(0.15660502531218382, 0.34200693567704915),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        c64(0.6290320273200906, 0.2338967948431292),
                        c64(0.9945869637623739, 0.3698233045587068),
                        c64(0.15660502531218382, 0.34200693567704915),
                        c64(0.7212939606895861, -0.788180753556546),
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(-0.7643540849057923, 0.395935120823352),
                        c64(0.2602794204440981, 0.437324634887758),
                        c64(-0.04437865404125869, 0.17486340495920796),
                        c64(0.8800737397504786, 0.2233542692154159),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(0.2602794204440981, 0.437324634887758),
                        c64(-0.7125119835850666, 0.48304084740448944),
                        c64(-0.04901782553425585, 0.1931429436469065),
                        c64(0.9720732177290143, 0.24670285382143028),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(-0.04437865404125869, 0.17486340495920796),
                        c64(-0.04901782553425585, 0.1931429436469065),
                        c64(-1.0607007031035893, 0.03612680961457026),
                        c64(0.1818233863750128, 0.305501856139204),
                    ],
                    [
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::ZERO,
                        c64(0.8800737397504786, 0.2233542692154159),
                        c64(0.9720732177290143, 0.24670285382143028),
                        c64(0.1818233863750128, 0.305501856139204),
                        c64(0.5375667715944483, -0.9151027778424118),
                    ]
                ]
            ];
            let exemplar_s = points![
                Complex64,
                [
                    [
                        c64(-0.3022491170600611, 0.1008132787157404),
                        c64(0.9097121238541618, -0.0704251288726103)
                    ],
                    [
                        c64(0.9097121238541618, -0.0704251288726103),
                        c64(0.36450243087469, 0.0491968800132813)
                    ]
                ]
            ];
            let sparms = cir.net().s_db();
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                sparms[(0, 0, 0)],
                sparms[(0, 0, 1)],
                sparms[(0, 1, 0)],
                sparms[(0, 1, 1)]
            );
            comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
            comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
            comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
        }
    }
}
