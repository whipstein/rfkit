#![allow(dead_code, unused)]
use crate::element::{Elem, ElemType, Element, Ground, IdealTransformer, Inductor, Transformer};
use ndarray::prelude::*;
use ndarray::{IntoDimension, linalg::Dot};
use num_complex::{Complex, ComplexFloat};
use rfkit_base::prelude::*;
use rfkit_network::prelude::*;
use std::f32::INFINITY;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Debug, Display, Formatter, Result},
    panic,
};

/// Representation of a node in a circuit
#[derive(Clone)]
struct Node<T: RealScalar> {
    elem: Vec<Element<T>>,      // Elements connected to the node
    port: Option<Element<T>>,   // Port connected to the node
    x: Points<Complex<T>, Ix3>, // Interaction scattering matrix for the node
    y: Vec<Complex<T>>, // Sum of admittances connected to the node (length is number of frequency points)
    ground: bool,       // Is this a ground node
}

impl<T: RealScalar + ScalarConst> Node<T> {
    /// Add element connected to Node
    fn add_elem<U: FreqValue<T>>(&self, elem: &Element<T>, freq: &U) -> Node<T> {
        let mut new_node = self.copy();
        if self.elem.len() != 0 {
            new_node.x = Points::from_shape_fn(
                (self.x.dim().0, self.x.dim().1 + 1, self.x.dim().2 + 1).into_dimension(),
                |(i, j, k)| {
                    if j < self.x.dim().1 && k < self.x.dim().2 {
                        self.x[[i, j, k]]
                    } else {
                        Complex::ZERO
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
    fn add_elem_inplace<U: FreqValue<T>>(&mut self, elem: Element<T>, freq: &U) {
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
    fn calc_x<U: FreqValue<T>>(&mut self, freq: &U) {
        match self.ground {
            true => {
                for (_, mut pt) in self.x.axis_iter_mut(Axis(0)).enumerate() {
                    let x = Array2::<Complex<T>>::from_shape_fn(
                        (self.elem.len(), self.elem.len()),
                        |(j, k)| {
                            if j == k { -Complex::ONE } else { Complex::ZERO }
                        },
                    );
                    pt.assign(&x);
                }
            }
            false => {
                for (i, mut pt) in self.x.axis_iter_mut(Axis(0)).enumerate() {
                    let x = Array2::<Complex<T>>::from_shape_fn(
                        (self.elem.len(), self.elem.len()),
                        |(j, k)| {
                            if j == k {
                                Complex::<T>::from_f64(2.0)
                                    / (self.elem[k].z_at(freq, i) * self.y[i])
                                    - Complex::ONE
                            } else {
                                // Complex::<T>::from_f64(2.0)
                                //     / ((self.elem[j].z_at(freq, i) * self.elem[k].z_at(freq, i))
                                //         .sqrt()
                                //         * self.y[i])
                                Complex::<T>::from_f64(2.0)
                                    / (self.elem[k].z_at(freq, i) * self.y[i])
                            }
                        },
                    );
                    pt.assign(&x);
                }
            }
        }
    }

    /// Calculate the sum of admittances connected to Node
    fn calc_y<U: FreqValue<T>>(&mut self, freq: &U) {
        match self.ground {
            true => {
                for i in 0..self.y.len() {
                    self.y[i] = Complex::<T>::new(T::INFINITY, T::INFINITY);
                }
            }
            false => {
                for i in 0..self.y.len() {
                    self.y[i] = Complex::ZERO;

                    for elem in self.elem.iter() {
                        self.y[i] += elem.z_at(freq, i).recip();
                    }
                }
            }
        }
    }

    /// Create a deep copy of Node
    fn copy(&self) -> Node<T> {
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
    fn new<U: FreqValue<T>>(freq: &U, ground: bool) -> Node<T> {
        let elem: Vec<Element<T>> = match ground {
            true => vec![Ground::new().into()],
            false => vec![],
        };
        Node {
            elem: elem,
            port: None,
            x: Points::zeros((freq.npts(), 1, 1).into_dimension()),
            y: vec![Complex::ZERO; freq.npts()],
            ground: ground,
        }
    }

    /// Is this a ground node
    fn ground(&self) -> bool {
        self.ground
    }

    /// Return port connected to Node
    fn port(&self) -> &Option<Element<T>> {
        &self.port
    }

    /// Return interaction scattering matrix of Node
    fn x(&self) -> &Points<Complex<T>, Ix3> {
        &self.x
    }

    /// Return a point from the interaction scattering matrix of Node
    fn x_val(&self, i: usize, j: usize, k: usize) -> &Complex<T> {
        &self.x[[i, j, k]]
    }

    /// Return sum of admittances for Node
    fn y(&self) -> &Vec<Complex<T>> {
        &self.y
    }

    /// Return a point from the sum of admittances for Node
    fn y_val(&self, i: usize) -> Complex<T> {
        self.y[i]
    }

    #[cfg(debug_assertions)]
    pub fn debug_elem(&self) -> Vec<String> {
        self.elem.iter().map(|e| format!("{:?}", e.id())).collect()
    }

    #[cfg(debug_assertions)]
    pub fn debug_x(&self) -> Vec<Vec<Vec<Complex<T>>>> {
        self.x.to_vec()
    }
}

impl<T: RealScalar> Debug for Node<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Node")
            .field("elements", &self.elem.len())
            .field("has a port", &self.port.is_some())
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl<T: RealScalar> Display for Node<T> {
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

impl<T: RealScalar + ScalarConst> Eq for Node<T> {}

impl<T: RealScalar + ScalarConst> PartialEq for Node<T> {
    fn eq(&self, other: &Node<T>) -> bool {
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
    pub fn new() -> CircuitMap {
        CircuitMap {
            names: vec!["gnd".to_string()],
            elem_map: vec![("gnd".to_string(), 0)],
            map: vec![0],
            node_map: vec![0],
            port_map: vec![],
        }
    }

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
                if *pos >= position {
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
                if i >= position {
                    *pos += 1;
                }
            }
            // for (i, pos) in self.port_map.iter_mut().enumerate() {
            //     if *pos > position {
            //         *pos += 1;
            //     }
            // }
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

    pub fn node_map(&self) -> Vec<usize> {
        self.node_map.clone()
    }

    pub fn port_map(&self) -> Vec<usize> {
        // let mut out = self.port_map.clone();
        // _ = out.remove(0);
        // out
        self.port_map.clone()
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
pub struct Circuit<T: RealScalar, U: FreqValue<T>> {
    c: Points<Complex<T>, Ix3>,            // Element Scattering Matrix
    freq: U,                               // Frequencies
    z0: Vec<Complex<T>>,                   // Port Impedances
    map: CircuitMap,                       // Map element node to full circuit matrix position
    nodes: BTreeMap<usize, Node<T>>,       // Interactions
    ports_only: bool,                      // Circuit Contains Ports Only
    s: Points<Complex<T>, Ix3>,            // Circuit Scattering Matrix
    x: Points<Complex<T>, Ix3>,            // Interaction Scattering Matrix
    dim: (usize, usize, usize),            // Dimension of S (nfreqs, nrows, ncols)
    dimx: (usize, usize, usize),           // Dimension of C & X (nfreqs, nrows, ncols)
    elements: HashMap<String, Element<T>>, // Public user-specified elements
    internal_elements: HashMap<String, Element<T>>, // Private generated implementation elements
    internal_nodes: HashSet<usize>,        // Circuit-only nodes generated while lowering elements
    public_node_aliases: HashMap<usize, usize>, // Public node labels remapped away from internals
}

impl<T: RealScalar + ScalarConst, U: FreqValue<T>> Circuit<T, U> {
    /// Create a new empty circuit
    pub fn new(f: &U) -> Circuit<T, U> {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, Node::new(f, true));
        Circuit {
            c: Points::zeros((f.npts(), 1, 1).into_dimension()),
            dim: (f.npts(), 0, 0),
            dimx: (f.npts(), 1, 1),
            freq: f.clone(),
            z0: vec![],
            map: CircuitMap::new(),
            nodes: nodes,
            ports_only: true,
            s: Points::zeros((f.npts(), 0, 0).into_dimension()),
            x: Points::zeros((f.npts(), 1, 1).into_dimension()),
            elements: HashMap::new(),
            internal_elements: HashMap::new(),
            internal_nodes: HashSet::new(),
            public_node_aliases: HashMap::new(),
        }
    }

    /// Clear the circuit
    pub fn clear(&mut self) {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, Node::new(&self.freq, true));
        self.c = Points::zeros((self.dimx.0, 1, 1).into_dimension());
        self.map.clear();
        self.nodes = nodes;
        self.ports_only = true;
        self.s = Points::zeros((self.dim.0, 0, 0).into_dimension());
        self.x = Points::zeros((self.dimx.0, 1, 1).into_dimension());
        self.dim = (self.dim.0, 0, 0);
        self.dimx = (self.dimx.0, 1, 1);
        self.elements.clear();
        self.internal_elements.clear();
        self.internal_nodes.clear();
        self.public_node_aliases.clear();
    }

    /// Make a deep copy of the circuit
    pub fn copy(&self) -> Circuit<T, U> {
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
            dimx: self.dimx,
            elements: self.elements.clone(),
            internal_elements: self.internal_elements.clone(),
            internal_nodes: self.internal_nodes.clone(),
            public_node_aliases: self.public_node_aliases.clone(),
        }
    }

    /// Update derived values after changing inputs
    fn update(&mut self, freq: &U) {
        self.calc_x();
        #[cfg(debug_assertions)]
        let debug_x = self.x.to_vec();
        self.calc_c(freq);
        self.calc_s();
    }

    /// Add an element to the circuit
    pub fn add_elem(&mut self, elem: &Element<T>, freq: &U) {
        if let Element::Transformer(transformer) = elem {
            self.add_transformer(transformer);
            self.elements.insert(elem.id(), elem.clone());
        } else {
            let nodes = elem
                .nodes()
                .into_iter()
                .map(|node| self.circuit_node_for_public(node))
                .collect::<Vec<_>>();
            self.add_elem_to_matrix(elem, &nodes);
            self.elements.insert(elem.id(), elem.clone());
        }

        self.update(freq);
    }

    fn add_elem_to_matrix(&mut self, elem: &Element<T>, nodes: &[usize]) {
        for (i, &elem_node) in nodes.iter().enumerate() {
            if elem.elem() == ElemType::Port {
                self.dim = (self.dim.0, self.dim.1 + 1, self.dim.2 + 1);
                self.map.add_port(elem_node, elem.id());
                self.z0.push(elem.z_at(&self.freq, 0));
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
    }

    fn add_transformer(&mut self, transformer: &Transformer<T>) {
        let nodes = transformer.nodes();
        let primary_external = self.circuit_node_for_public(nodes[0]);
        let secondary_external = self.circuit_node_for_public(nodes[1]);
        let primary_internal = self.allocate_internal_node();
        let secondary_internal = self.allocate_internal_node();
        let id = transformer.id();
        let z0 = transformer.z0();
        let one = <T as ScalarConst>::ONE;

        let primary_leakage: Element<T> = Inductor::new(
            &format!("{id}::lp"),
            ScalarUnitValue::new(
                &(transformer.l1() * (one - transformer.km())),
                Scale::Base,
                Unit::Henry,
            ),
            transformer.q1(),
            [primary_external, primary_internal],
            z0,
        )
        .into();
        let magnetizing: Element<T> = Inductor::new(
            &format!("{id}::lm"),
            ScalarUnitValue::new(
                &(transformer.km() * transformer.l1()),
                Scale::Base,
                Unit::Henry,
            ),
            None,
            [primary_internal, 0],
            z0,
        )
        .into();
        let secondary_leakage: Element<T> = Inductor::new(
            &format!("{id}::ls"),
            ScalarUnitValue::new(
                &(transformer.l2() * (one - transformer.km())),
                Scale::Base,
                Unit::Henry,
            ),
            transformer.q2(),
            [secondary_internal, secondary_external],
            z0,
        )
        .into();
        let ideal: Element<T> = IdealTransformer::new(
            &format!("{id}::ideal"),
            transformer.n(),
            [primary_internal, secondary_internal],
            z0,
        )
        .into();

        for elem in [primary_leakage, magnetizing, secondary_leakage, ideal] {
            let nodes = elem.nodes();
            self.add_elem_to_matrix(&elem, &nodes);
            self.internal_elements.insert(elem.id(), elem);
        }
    }

    fn allocate_internal_node(&mut self) -> usize {
        let node = self.next_available_node();
        self.internal_nodes.insert(node);
        node
    }

    fn circuit_node_for_public(&mut self, node: usize) -> usize {
        if node == 0 {
            return 0;
        }

        if let Some(mapped) = self.public_node_aliases.get(&node) {
            return *mapped;
        }

        let reserved_alias = self
            .public_node_aliases
            .values()
            .any(|&mapped_node| mapped_node == node);
        if self.internal_nodes.contains(&node) || reserved_alias {
            let mapped = self.next_available_node();
            self.public_node_aliases.insert(node, mapped);
            mapped
        } else {
            node
        }
    }

    fn next_available_node(&self) -> usize {
        let mut node = self
            .nodes
            .last_key_value()
            .map(|(&node, _)| node + 1)
            .unwrap_or(1);

        while self.nodes.contains_key(&node)
            || self.internal_nodes.contains(&node)
            || self
                .public_node_aliases
                .values()
                .any(|&mapped_node| mapped_node == node)
        {
            node += 1;
        }

        node
    }

    /// Calculate the element scattering matrix
    fn calc_c(&mut self, freq: &U) {
        self.c = Points::zeros(self.dimx.into_dimension());

        for (j, (id1, node1)) in self.map.elem_map().iter().enumerate() {
            for (k, (id2, node2)) in self.map.elem_map().iter().enumerate() {
                if id1 == id2 {
                    for i in 0..self.freq.npts() {
                        let val = match self.element_for_id(id1) {
                            Some(x) => x.c_at(freq, (i, *node1, *node2)),
                            None => Complex::ZERO,
                        };
                        self.c[[i, j, k]] = val;
                    }
                }
            }
        }
    }

    fn element_for_id(&self, id: &str) -> Option<&Element<T>> {
        self.elements
            .get(id)
            .or_else(|| self.internal_elements.get(id))
    }

    /// Calculate the circuit scattering matrix
    fn calc_s(&mut self) {
        self.s = Points::zeros(self.dim.into_dimension());

        if self.ports_only {
            return;
        }

        let id = Points::<Complex<T>, Ix2>::eye(self.x.dim().1);
        #[cfg(debug_assertions)]
        let debug_id = id.to_vec();

        for i in 0..self.freq.npts() {
            let c: Points<Complex<T>, Ix2> = Points::from(self.c.slice(s![i, .., ..]));
            let x: Points<Complex<T>, Ix2> = Points::from(self.x.slice(s![i, .., ..]));
            #[cfg(debug_assertions)]
            let debug_c = c.to_vec();
            #[cfg(debug_assertions)]
            let debug_x = x.to_vec();

            let mut net = &id - &c.dot(&x);
            #[cfg(debug_assertions)]
            let debug_net = net.to_vec();

            let net_inv = match net.try_inv() {
                Ok(val) => val,
                _ => panic!("{}", format!("inverse does not exist. {:?}", net)),
            };
            #[cfg(debug_assertions)]
            let debug_net_inv = net_inv.to_vec();

            net = x.dot(&net_inv);
            #[cfg(debug_assertions)]
            let debug_net = net.to_vec();

            for (j, &map_j) in self.map.port_map().iter().enumerate() {
                for (k, &map_k) in self.map.port_map().iter().enumerate() {
                    self.s[[i, j, k]] = net[[map_j, map_k]];
                }
            }
        }
    }

    /// Calculate the interaction scattering matrix\
    fn calc_x(&mut self) {
        self.x = Points::zeros(self.dimx.into_dimension());

        let mut cntr = 0;
        for node in self.nodes.clone().into_values() {
            #[cfg(debug_assertions)]
            let debug_node_x = node.debug_x();
            let j = cntr + node.len();
            self.x
                .slice_mut(s![.., cntr..j, cntr..j])
                .assign(node.x().inner());
            cntr += node.len();
        }
    }

    pub fn c(&self) -> &Points<Complex<T>, Ix3> {
        &self.c
    }

    pub fn freq(&self) -> &U {
        &self.freq
    }

    pub fn ports_only(&self) -> bool {
        self.ports_only
    }

    pub fn s(&self) -> &Points<Complex<T>, Ix3> {
        &self.s
    }

    pub fn x(&self) -> &Points<Complex<T>, Ix3> {
        &self.x
    }

    fn y(&self) -> Vec<Vec<Complex<T>>> {
        let mut out: Vec<Vec<Complex<T>>> = vec![];

        for (_, node) in self.nodes.iter() {
            out.push(node.y().clone());
        }

        out
    }

    pub fn dim(&self) -> (usize, usize, usize) {
        self.dim
    }

    pub fn dimx(&self) -> (usize, usize, usize) {
        self.dimx
    }

    pub fn elements(&self) -> &HashMap<String, Element<T>> {
        &self.elements
    }

    pub fn z0(&self) -> &Vec<Complex<T>> {
        &self.z0
    }

    pub fn net(&self) -> Network<T, U> {
        NetworkBuilder::new()
            .freq(&self.freq)
            .z0(&self.z0)
            .net(&self.s, RFParameter::S)
            .build()
            .unwrap()
    }

    #[cfg(debug_assertions)]
    pub fn debug_c(&self) -> Vec<Vec<Vec<Complex<T>>>> {
        self.c.to_vec()
    }

    #[cfg(debug_assertions)]
    pub fn debug_s(&self) -> Vec<Vec<Vec<Complex<T>>>> {
        self.s.to_vec()
    }

    #[cfg(debug_assertions)]
    pub fn debug_x(&self) -> Vec<Vec<Vec<Complex<T>>>> {
        self.x.to_vec()
    }

    #[cfg(debug_assertions)]
    pub fn debug_nodes(&self) -> Vec<(usize, Vec<String>)> {
        self.nodes
            .iter()
            .map(|(k, v)| (*k, v.debug_elem()))
            .collect()
    }

    #[cfg(debug_assertions)]
    pub fn debug_elements(&self) -> Vec<(String, Element<T>)> {
        self.elements
            .iter()
            .chain(self.internal_elements.iter())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[cfg(test)]
mod circuit_tests {
    use super::*;
    use crate::element::{
        Capacitor, IdealTransformer, IdealTransformerBuilder, Inductor, MlinBuilder, MsubBuilder,
        Port, Q, QBuilder, QMode, Resistor, Short, Transformer, TransformerBuilder,
    };
    use num_complex::{Complex64, c64};
    use num_traits::ConstZero;

    const MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-14,
        relative: 1e-14,
        ulps: 10,
    };
    const LOOSE_MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-10,
        relative: 1e-10,
        ulps: 10,
    };
    const VERY_LOOSE_MARGIN: NumMargin<f64> = NumMargin {
        epsilon: 1e-4,
        relative: 1e-4,
        ulps: 10,
    };

    #[test]
    fn node_rlc() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let mut node = Node::new(&freq, false);
        let margin = NumMargin::default();

        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let r1: Element<f64> = Resistor::new(
            "R1",
            ScalarUnitValue::new_scaled(&20.0, Scale::Base, Unit::Ohm),
            [1, 2],
            50.0.into(),
        )
        .into();

        freq.map_scalar_to_vec(|f| {
            let z_p1 = p1.z(f);
            let z_r1 = r1.z(f);
            let z_c1 = c1.z(f);
            let z_l1 = l1.z(f);

            let y = z_p1.recip();
            let exemplar_y = vec![y];
            let exemplar_x = Points::<Complex64, Ix3>::ones((1, 1, 1));
            node = node.add_elem(&p1, &freq);
            comp_vec_c64(&exemplar_y, &node.y(), margin, "node.y(p1)");
            comp_pts_ix3(&exemplar_x, &node.x, margin, "node.x(p1)");

            let y = 1.0 / z_p1 + 1.0 / z_r1;
            let x11 = 2.0 / (z_p1 * y) - 1.0;
            let x22 = 2.0 / (z_r1 * y) - 1.0;
            let x12 = 2.0 / (z_r1 * y);
            let x21 = 2.0 / (z_p1 * y);
            let exemplar_y = vec![y];
            let exemplar_x = points![[[x11, x12], [x21, x22]]];
            node = node.add_elem(&r1, &freq);
            comp_vec_c64(&exemplar_y, &node.y(), margin, "node.y(r1)");
            comp_pts_ix3(&exemplar_x, &node.x, margin, "node.x(r1)");

            let y = 1.0 / z_p1 + 1.0 / z_r1 + 1.0 / z_c1;
            let x11 = 2.0 / (z_p1 * y) - 1.0;
            let x22 = 2.0 / (z_r1 * y) - 1.0;
            let x33 = 2.0 / (z_c1 * y) - 1.0;
            let x12 = 2.0 / (z_r1 * y);
            let x13 = 2.0 / (z_c1 * y);
            let x21 = 2.0 / (z_p1 * y);
            let exemplar_y = vec![y];
            let exemplar_x = points![[[x11, x12, x13], [x21, x22, x13], [x21, x12, x33]]];
            node = node.add_elem(&c1, &freq);
            comp_vec_c64(&exemplar_y, &node.y(), margin, "node.y(c1)");
            comp_pts_ix3(&exemplar_x, &node.x, margin, "node.x(c1)");

            let y = 1.0 / z_p1 + 1.0 / z_r1 + 1.0 / z_c1 + 1.0 / z_l1;
            let x11 = 2.0 / (z_p1 * y) - 1.0;
            let x22 = 2.0 / (z_r1 * y) - 1.0;
            let x33 = 2.0 / (z_c1 * y) - 1.0;
            let x44 = 2.0 / (z_l1 * y) - 1.0;
            let x12 = 2.0 / (z_r1 * y);
            let x13 = 2.0 / (z_c1 * y);
            let x14 = 2.0 / (z_l1 * y);
            let x21 = 2.0 / (z_p1 * y);
            let exemplar_y = vec![y];
            let exemplar_x = points![[
                [x11, x12, x13, x14],
                [x21, x22, x13, x14],
                [x21, x12, x33, x14],
                [x21, x12, x13, x44]
            ]];
            node = node.add_elem(&l1, &freq);
            comp_vec_c64(&exemplar_y, &node.y(), margin, "node.y(l1)");
            comp_pts_ix3(&exemplar_x, &node.x, margin, "node.x(l1)");
        });
    }

    #[test]
    fn circuitmap1() {
        let mut map = CircuitMap::new();
        let exemplar = CircuitMap {
            names: vec!["gnd".to_string()],
            elem_map: vec![("gnd".to_string(), 0)],
            map: vec![0],      // [gnd]
            node_map: vec![0], // [0]
            port_map: vec![],  // []
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
            port_map: vec![1],    // [p1]
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
            port_map: vec![1, 2],    // [p1, p2]
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
            port_map: vec![1, 2, 3],    // [p1, p2, p3]
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
            port_map: vec![1, 3, 4],    // [p1, p2, p3]
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
            port_map: vec![1, 3, 5],     // [p1, p2, p3]
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
            port_map: vec![1, 4, 6],        // [p1, p2, p3]
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
            port_map: vec![1, 4, 6],           // [p1, p2, p3]
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
            port_map: vec![1, 4, 7],              // [p1, p2, p3]
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
            port_map: vec![1, 4, 7],                 // [p1, p2, p3]
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
            port_map: vec![1, 5, 8],                    // [p1, p2, p3]
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
            port_map: vec![2, 6, 9],                       // [p1, p2, p3]
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
            port_map: vec![],  // []
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
            port_map: vec![1],    // [p1]
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
            port_map: vec![1, 2],    // [p1, p2]
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
            port_map: vec![1, 2, 3],    // [p1, p2, p3]
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
            port_map: vec![1, 3, 4],    // [p1, p2, p3]
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
            port_map: vec![1, 3, 4],       // [p1, p2, p3]
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
            port_map: vec![1, 3, 4],        // [p1, p2, p3]
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
            port_map: vec![1, 3, 5],           // [p1, p2, p3]
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
            port_map: vec![1, 3, 5],              // [p1, p2, p3]
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
            port_map: vec![1, 3, 5],                 // [p1, p2, p3]
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
            port_map: vec![1, 3, 5],                    // [p1, p2, p3]
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
            port_map: vec![2, 4, 6],                       // [p1, p2, p3]
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
            port_map: vec![],  // []
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
            port_map: vec![1],             // [p4]
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
            port_map: vec![1, 2],          // [p2, p4]
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
            port_map: vec![1, 2, 3],       // [p1, p2, p4]
        };
        assert_eq!(&exemplar.map(), &map.map());
        assert_eq!(&exemplar.node_map(), &map.node_map());
        assert_eq!(&exemplar.port_map(), &map.port_map());
    }

    #[test]
    fn circuit_r() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let mut cir = Circuit::new(&freq);
        let p1 = Port::new("p1", 50.0.into(), [1]);
        let p2 = Port::new("p2", 50.0.into(), [2]);
        let r1 = Resistor::new(
            "R1",
            ScalarUnitValue::new_scaled(&20.0, Scale::Base, Unit::Ohm),
            [1, 2],
            50.0.into(),
        );
        let margin = MARGIN;

        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE]
        ]];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        cir.add_elem(&p1.into(), &freq);
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
        cir.add_elem(&p2.into(), &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(2)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(2)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(2)");

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
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
                c64(1.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.5714285714285714, 0.0),
                c64(0.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.4285714285714286, 0.0),
                c64(1.4285714285714284, 0.0),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.5714285714285714, 0.0),
                c64(0.4285714285714284, 0.0),
            ]
        ]];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => 0.1666666666666659.into(),
                (0, 1) | (1, 0) => 0.8333333333333326.into(),
                _ => Complex64::ZERO,
            });
        cir.add_elem(&r1.into(), &freq);
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(3)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(3)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(3)");

        let mut cir2 = Circuit::new(&freq);
        let p1 = Port::new("p1", 50.0.into(), [1]);
        let p2 = Port::new("p2", 10.0.into(), [2]);
        let r1 = Resistor::new(
            "R1",
            ScalarUnitValue::new(&20.0, Scale::Base, Unit::Ohm),
            [1, 2],
            50.0.into(),
        );

        let exemplar2_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar2_x = points![[
            [Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE]
        ]];
        let exemplar2_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        cir2.add_elem(&p1.into(), &freq);
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
        cir2.add_elem(&p2.into(), &freq);
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(2)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(2)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(2)");

        let exemplar2_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar2_x = points![[
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
                c64(1.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.5714285714285714, 0.0),
                c64(0.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.33333333333333304, 0.0),
                c64(0.6666666666666665, 0.0),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.333333333333333, 0.0),
                c64(-0.3333333333333335, 0.0),
            ]
        ]];
        let exemplar2_y = vec![
            vec![Complex::INFINITY],
            vec![c64(0.07, 0.0)],
            vec![c64(0.15, 0.0)],
        ];
        let exemplar2_s = points![[
            [c64(-0.2500000000000001, 0.0), c64(1.2499999999999993, 0.0)],
            [c64(0.24999999999999992, 0.0), c64(0.7499999999999993, 0.0)]
        ]];
        cir2.add_elem(&r1.into(), &freq);
        #[cfg(debug_assertions)]
        let debug_c = cir2.debug_c();
        #[cfg(debug_assertions)]
        let debug_s = cir2.debug_s();
        #[cfg(debug_assertions)]
        let debug_x = cir2.debug_x();
        #[cfg(debug_assertions)]
        let debug_y = cir2.y();
        #[cfg(debug_assertions)]
        let debug_nodes = cir2.debug_nodes();
        let calc_y = cir2.y();
        for (i, pt) in exemplar2_y.iter().enumerate() {
            comp_vec_c64(
                pt,
                &calc_y[i],
                margin,
                format!("cir.y(3)[{i}]").to_owned().as_str(),
            );
        }
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(3)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(3)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(3)");
    }

    #[test]
    fn circuit_c() {
        let margin = MARGIN;
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);

        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();

        cir.add_elem(&p1, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE]
        ]];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(1)");

        cir.add_elem(&p2, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ZERO, Complex64::ONE]
        ]];
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
        let exemplar_x = points![[
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.8203396752925505, -0.5718765750937107),
                c64(0.17966032470744933, 0.5718765750937107),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(1.8203396752925505, -0.5718765750937107),
                c64(-0.8203396752925507, 0.5718765750937107),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.8203396752925505, -0.5718765750937107),
                c64(0.17966032470744933, 0.5718765750937107),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.8203396752925505, -0.5718765750937107),
                c64(-0.8203396752925507, 0.5718765750937107),
            ]
        ]];
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
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 10.0.into(), [2]).into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();

        cir2.add_elem(&p1, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE]
        ]];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(1)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(1)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(1)");

        cir2.add_elem(&p2, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ZERO, Complex64::ONE]
        ]];
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
        let exemplar_x = points![[
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.8203396752925505, -0.5718765750937107),
                c64(0.17966032470744933, 0.5718765750937107),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(1.8203396752925505, -0.5718765750937107),
                c64(-0.8203396752925507, 0.5718765750937107),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.9921353648143452, -0.1251695565411434),
                c64(0.007864635185654967, 0.1251695565411434),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.9921353648143452, -0.1251695565411434),
                c64(-0.992135364814345, 0.1251695565411434),
            ]
        ]];
        let exemplar_s = points![[
            [
                c64(0.7926049557687088, -0.5501324410362041),
                c64(0.2073950442312911, 0.550132441036204)
            ],
            [
                c64(0.041479008846258214, 0.11002648820724084),
                c64(0.9585209911537419, -0.11002648820724081)
            ]
        ]];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(3)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(3)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(3)");
    }

    #[test]
    fn circuit_l() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();

        cir.add_elem(&p1, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE]
        ]];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(1)");

        cir.add_elem(&p2, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ZERO, Complex64::ONE]
        ]];
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
        let exemplar_x = points![[
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
                c64(1.9689082471969976, -0.24742030739945778),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.0310917528030026, 0.24742030739945778),
                c64(0.9689082471969976, -0.24742030739945778),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.9689082471969974, 0.24742030739945778),
                c64(1.9689082471969976, -0.24742030739945778)
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.0310917528030026, 0.24742030739945778),
                c64(0.9689082471969976, -0.24742030739945778)
            ]
        ]];
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
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 10.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();

        cir2.add_elem(&p1, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE]
        ]];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(1)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(1)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(1)");

        cir2.add_elem(&p2, &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ZERO, Complex64::ONE]
        ]];
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
        let exemplar_x = points![[
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
                c64(1.9689082471969976, -0.24742030739945778),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.0310917528030026, 0.24742030739945778),
                c64(0.9689082471969976, -0.24742030739945778),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.43391360064979545, 0.9009544867367774),
                c64(1.4339136006497955, -0.9009544867367774)
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.5660863993502045, 0.9009544867367774),
                c64(0.43391360064979545, -0.9009544867367774)
            ]
        ]];
        let exemplar_s = points![[
            [
                c64(-0.6485878775864337, 0.172639718834091),
                c64(1.6485878775864344, -0.1726397188340909),
            ],
            [
                c64(0.3297175755172867, -0.03452794376681821),
                c64(0.6702824244827135, 0.034527943766818114),
            ],
        ]];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(3)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(3)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(3)");
    }

    #[test]
    fn circuit_shunt_l() {
        let margin = MARGIN;
        let z0 = 50.0;
        let freq = ArrayUnitValue::new_freq_scaled(&array![275.0], Scale::Giga);
        let src = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(92.8568)
            .im(0.0)
            .z0(z0)
            .freq(freq.freq_scalar(0))
            .build()
            .unwrap();
        let load = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(18.9383)
            .im(0.0)
            .z0(z0)
            .freq(freq.freq_scalar(0))
            .build()
            .unwrap();
        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", src.rp().val().into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", load.rp().val().into(), [2]).into();
        let short: Element<f64> = Short::new("S0", [1, 2], 50.0.into()).into();
        let ls: Element<f64> = Inductor::new(
            "Ls",
            ScalarUnitValue::new_scaled(&5.0, Scale::Pico, Unit::Henry),
            None,
            [2, 0],
            50.0.into(),
        )
        .into();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (1, 1) | (5, 5) => 0.3333333333333333.into(),
                (3, 6) | (6, 3) => 1.0.into(),
                (1, 5) | (5, 1) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
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
                c64(-0.9999999999978462, 0.0),
                c64(1.9999999999978462, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                c64(2.15385410653592e-12, 0.0),
                c64(0.9999999999978462, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.9999999999894393, 1.2223805676344343e-22),
                c64(2.679568492992206e-22, -2.314980990415345e-11),
                c64(1.9999999999894391, 2.3149809904031206e-11),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.0560609980776728e-11, 1.2223805676344343e-22),
                c64(-1.0, -2.314980990415345e-11),
                c64(1.9999999999894391, 2.3149809904031206e-11),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.0560609980776728e-11, 1.2223805676344343e-22),
                c64(2.679568492992206e-22, -2.314980990415345e-11),
                c64(0.9999999999894391, 2.3149809904031206e-11),
            ]
        ]];
        let exemplar_s = points![[
            [
                c64(-0.921483597561384, 0.14295557061878594),
                c64(0.38497551934239493, 0.7009286382534062),
            ],
            [
                c64(0.078516402438616, 0.14295557061878597),
                c64(-0.6150244806576051, 0.7009286382534062),
            ]
        ]];
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(1)");
        comp_pts_ix3(&exemplar_s, &cir.s, LOOSE_MARGIN, "cir.s(1)");

        let src = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(9.4595)
            .im(-28.0873)
            .z0(z0)
            .freq(freq.freq_scalar(0))
            .build()
            .unwrap();
        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", src.rp().val().into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", load.rp().val().into(), [2]).into();
        let cp1: Element<f64> = Capacitor::new(
            "Cp1",
            ScalarUnitValue::new(&src.cp().val(), Scale::Femto, Unit::Farad),
            None,
            [1, 0],
            50.0.into(),
        )
        .into();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&cp1, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_s = points![[
            [
                c64(-0.8761891493754457, 0.1631491116570081),
                c64(0.6070595021500979, 0.7999397306385363),
            ],
            [
                c64(0.12381085062455434, 0.1631491116570081),
                c64(-0.39294049784990215, 0.7999397306385363),
            ]
        ]];
        comp_pts_ix3(&exemplar_s, &cir.s, LOOSE_MARGIN, "cir.s(2)");

        let load = ImpedanceBuilder::new()
            .kind(ImpedanceType::Z)
            .category(ComplexNumberType::ReIm)
            .mode(ImpedanceMode::Se)
            .re(16.923)
            .im(-5.84)
            .z0(z0)
            .freq(freq.freq_scalar(0))
            .build()
            .unwrap();
        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", src.rp().val().into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", load.rp().val().into(), [2]).into();
        let cp1: Element<f64> = Capacitor::new(
            "Cp1",
            ScalarUnitValue::new(&src.cp().val(), Scale::Femto, Unit::Farad),
            None,
            [1, 0],
            50.0.into(),
        )
        .into();
        let cp2: Element<f64> = Capacitor::new(
            "Cp2",
            ScalarUnitValue::new(&load.cp().val(), Scale::Femto, Unit::Farad),
            None,
            [2, 0],
            50.0.into(),
        )
        .into();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&cp1, &freq);
        cir.add_elem(&cp2, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 11, 11), |(_, j, k)| match (j, k) {
                (1, 1) | (2, 2) | (3, 3) | (5, 5) | (8, 8) | (9, 9) => 0.3333333333333333.into(),
                (6, 10) | (10, 6) => 1.0.into(),
                (1, 5) | (5, 1) | (2, 8) | (8, 2) | (3, 9) | (9, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
            [
                -Complex64::ONE,
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
                -Complex64::ONE,
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
                Complex64::ZERO,
                -Complex64::ONE,
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
                Complex64::ZERO,
                Complex64::ZERO,
                -Complex64::ONE,
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
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.9999999999978462, -6.887232040743972e-24),
                c64(2.044968047972812e-23, 6.395260820283739e-12),
                c64(1.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(2.1538549354859325e-12, -6.887232040743972e-24),
                c64(-1.0, 6.395260820283739e-12),
                c64(1.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(2.1538549354859325e-12, -6.887232040743972e-24),
                c64(2.044968047972812e-23, 6.395260820283739e-12),
                c64(0.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.9999999999894394, 1.0299440436073173e-22),
                c64(-3.5542594189367926e-23, 3.644379353424739e-12),
                c64(2.257735046740184e-22, -2.314980990415345e-11),
                c64(1.9999999999894391, 1.9505430550625713e-11),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.056058763664501e-11, 1.0299440436073173e-22),
                c64(-1.0, 3.644379353424739e-12),
                c64(2.257735046740184e-22, -2.314980990415345e-11),
                c64(1.9999999999894391, 1.9505430550625713e-11),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.056058763664501e-11, 1.0299440436073173e-22),
                c64(-3.5542594189367926e-23, 3.644379353424739e-12),
                c64(-1.0, -2.314980990415345e-11),
                c64(1.9999999999894391, 1.9505430550625713e-11),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.056058763664501e-11, 1.0299440436073173e-22),
                c64(-3.5542594189367926e-23, 3.644379353424739e-12),
                c64(2.257735046740184e-22, -2.314980990415345e-11),
                c64(0.9999999999894391, 1.9505430550625713e-11),
            ],
        ]];
        let exemplar_s = points![[
            [
                c64(-0.835788160210767, 0.16931913575337662),
                c64(0.8051487110411862, 0.8301903448704762),
            ],
            [
                c64(0.16421183978923312, 0.16931913575337662),
                c64(-0.19485128895881376, 0.830190344870476),
            ]
        ]];
        let exemplar_y = vec![
            vec![Complex::INFINITY],
            vec![c64(10000000000.010769, 0.03197630410145313)],
            vec![c64(10000000000.052803, -0.09752715275415853)],
        ];
        let calc_y = cir.y();
        for (i, pt) in exemplar_y.iter().enumerate() {
            comp_vec_c64(
                pt,
                &calc_y[i],
                margin,
                format!("cir.y(3)[{i}]").to_owned().as_str(),
            );
        }
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(3)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(3)");
        comp_pts_ix3(&exemplar_s, &cir.s, LOOSE_MARGIN, "cir.s(3)");

        let mut cir = Circuit::new(&freq);
        let ls: Element<f64> = Inductor::new(
            "Ls",
            ScalarUnitValue::new_scaled(&5.0, Scale::Pico, Unit::Henry),
            Some(Q::new(
                ScalarUnitValue::new(&0.0, Scale::Base, Unit::Ohm),
                15.0,
                freq.freq_scalar(0),
                0.0,
                QMode::ProportionalToFreq,
            )),
            [2, 0],
            50.0.into(),
        )
        .into();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&cp1, &freq);
        cir.add_elem(&cp2, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 11, 11), |(_, j, k)| match (j, k) {
                (1, 1) | (2, 2) | (3, 3) | (5, 5) | (8, 8) | (9, 9) => 0.3333333333333333.into(),
                (6, 10) | (10, 6) => 1.0.into(),
                (1, 5) | (5, 1) | (2, 8) | (8, 2) | (3, 9) | (9, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
            [
                -Complex64::ONE,
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
                -Complex64::ONE,
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
                Complex64::ZERO,
                -Complex64::ONE,
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
                Complex64::ZERO,
                Complex64::ZERO,
                -Complex64::ONE,
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
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.9999999999978462, -6.887232040743972e-24),
                c64(2.044968047972812e-23, 6.395260820283739e-12),
                c64(1.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(2.1538549354859325e-12, -6.887232040743972e-24),
                c64(-1.0, 6.395260820283739e-12),
                c64(1.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(2.1538549354859325e-12, -6.887232040743972e-24),
                c64(2.044968047972812e-23, 6.395260820283739e-12),
                c64(0.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.9999999999894394, 1.024535291475998e-22),
                c64(-3.5355942221945444e-23, 3.644379353421939e-12),
                c64(1.5364918080202088e-12, -2.3047377116934318e-11),
                c64(1.9999999999879026, 1.9402997763409922e-11),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.0560587636636896e-11, 1.024535291475998e-22),
                c64(-1.0, 3.644379353421939e-12),
                c64(1.5364918080202088e-12, -2.3047377116934318e-11),
                c64(1.9999999999879026, 1.9402997763409922e-11),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.0560587636636896e-11, 1.024535291475998e-22),
                c64(-3.5355942221945444e-23, 3.644379353421939e-12),
                c64(-0.9999999999984636, -2.3047377116934318e-11),
                c64(1.9999999999879026, 1.9402997763409922e-11),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.0560587636636896e-11, 1.024535291475998e-22),
                c64(-3.5355942221945444e-23, 3.644379353421939e-12),
                c64(1.5364918080202088e-12, -2.3047377116934318e-11),
                c64(0.9999999999879026, 1.9402997763409922e-11),
            ],
        ]];
        let exemplar_s = points![[
            [
                c64(-0.835105153421399, 0.1505065348023749),
                c64(0.8084975684487749, 0.7379500933364064),
            ],
            [
                c64(0.164894846578601, 0.15050653480237491),
                c64(-0.19150243155122515, 0.7379500933364064),
            ]
        ]];
        let exemplar_y = vec![
            vec![Complex::INFINITY],
            vec![c64(10000000000.010769, 0.03197630410145313)],
            vec![c64(10000000000.060486, -0.09701498881822324)],
        ];
        let calc_y = cir.y();
        for (i, pt) in exemplar_y.iter().enumerate() {
            comp_vec_c64(
                pt,
                &calc_y[i],
                margin,
                format!("cir.y(4)[{i}]").to_owned().as_str(),
            );
        }
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(4)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(4)");
        comp_pts_ix3(&exemplar_s, &cir.s, LOOSE_MARGIN, "cir.s(4)");

        let mut cir = Circuit::new(&freq);
        let ls: Element<f64> = Inductor::new(
            "Ls",
            ScalarUnitValue::new_scaled(&11.4782, Scale::Pico, Unit::Henry),
            Some(Q::new(
                ScalarUnitValue::new(&0.0, Scale::Base, Unit::Ohm),
                15.0,
                freq.freq_scalar(0),
                0.0,
                QMode::ProportionalToFreq,
            )),
            [2, 0],
            50.0.into(),
        )
        .into();
        cir.add_elem(&p1, &freq);
        cir.add_elem(&p2, &freq);
        cir.add_elem(&cp1, &freq);
        cir.add_elem(&cp2, &freq);
        cir.add_elem(&ls, &freq);
        cir.add_elem(&short, &freq);

        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 11, 11), |(_, j, k)| match (j, k) {
                (1, 1) | (2, 2) | (3, 3) | (5, 5) | (8, 8) | (9, 9) => 0.3333333333333333.into(),
                (6, 10) | (10, 6) => 1.0.into(),
                (1, 5) | (5, 1) | (2, 8) | (8, 2) | (3, 9) | (9, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
            [
                -Complex64::ONE,
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
                -Complex64::ONE,
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
                Complex64::ZERO,
                -Complex64::ONE,
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
                Complex64::ZERO,
                Complex64::ZERO,
                -Complex64::ONE,
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
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.9999999999978462, -6.887232040743972e-24),
                c64(2.044968047972812e-23, 6.395260820283739e-12),
                c64(1.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(2.1538549354859325e-12, -6.887232040743972e-24),
                c64(-1.0, 6.395260820283739e-12),
                c64(1.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(2.1538549354859325e-12, -6.887232040743972e-24),
                c64(2.044968047972812e-23, 6.395260820283739e-12),
                c64(0.9999999999978462, -6.395260820276852e-12),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.9999999999894394, 3.376880453450238e-23),
                c64(-1.1653360425544757e-23, 3.644379353423519e-12),
                c64(6.693086929444417e-13, -1.0039630393682938e-11),
                c64(1.9999999999887699, 6.395251040225651e-12),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.0560587636641474e-11, 3.376880453450238e-23),
                c64(-1.0, 3.644379353423519e-12),
                c64(6.693086929444417e-13, -1.0039630393682938e-11),
                c64(1.9999999999887699, 6.395251040225651e-12),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.0560587636641474e-11, 3.376880453450238e-23),
                c64(-1.1653360425544757e-23, 3.644379353423519e-12),
                c64(-0.9999999999993306, -1.0039630393682938e-11),
                c64(1.9999999999887699, 6.395251040225651e-12),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.0560587636641474e-11, 3.376880453450238e-23),
                c64(-1.1653360425544757e-23, 3.644379353423519e-12),
                c64(6.693086929444417e-13, -1.0039630393682938e-11),
                c64(0.9999999999887699, 6.395251040225651e-12),
            ],
        ]];
        let exemplar_y = vec![
            vec![Complex::INFINITY],
            vec![c64(10000000000.010769, 0.03197630410145313)],
            vec![c64(10000000000.05615, -0.03197625520148735)],
        ];
        let exemplar_s = points![[
            [
                c64(-0.6781438620736016, -2.3518829132440623e-07),
                c64(1.5780960430425985, -1.1531540591401409e-06),
            ],
            [
                c64(0.32185613792639844, -2.3518829132440626e-07),
                c64(0.5780960430425984, -1.1531540591401409e-06),
            ]
        ]];
        let calc_y = cir.y();
        for (i, pt) in exemplar_y.iter().enumerate() {
            comp_vec_c64(
                pt,
                &calc_y[i],
                margin,
                format!("cir.y(5)[{i}]").to_owned().as_str(),
            );
        }
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(5)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(5)");
        comp_pts_ix3(&exemplar_s, &cir.s, VERY_LOOSE_MARGIN, "cir.s(5)");
    }

    #[test]
    fn circuit_rc() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let r1: Element<f64> = Resistor::new(
            "R1",
            ScalarUnitValue::new(&20.0, Scale::Base, Unit::Ohm),
            [1, 2],
            50.0.into(),
        )
        .into();

        cir.add_elem(&p1.into(), &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar_x = points![[
            [Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE]
        ]];
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(p1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(p1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(p1)");

        cir.add_elem(&p2.into(), &freq);
        let exemplar_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar_x = Points::<Complex64, Ix3>::from_shape_fn((1, 3, 3), |(_, j, k)| {
            if j == k && j != 0 {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        let exemplar_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(p2)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(p2)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(p2)");

        cir.add_elem(&r1.into(), &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
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
                c64(1.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.5714285714285714, 0.0),
                c64(0.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(-0.4285714285714286, 0.0),
                c64(1.4285714285714284, 0.0),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.5714285714285714, 0.0),
                c64(0.4285714285714284, 0.0),
            ]
        ]];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => 0.1666666666666659.into(),
                (0, 1) | (1, 0) => 0.8333333333333326.into(),
                _ => Complex64::ZERO,
            });
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(r1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(r1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(r1)");

        cir.add_elem(&c1.into(), &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
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
                c64(-0.4331385293595438, -0.05088136662191896),
                c64(1.4171536766011408, -0.12720341655479742),
                c64(0.015984852758402952, 0.1780847831767164),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.5668614706404562, -0.05088136662191896),
                c64(0.4171536766011408, -0.12720341655479742),
                c64(0.015984852758402952, 0.1780847831767164),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.5668614706404562, -0.05088136662191896),
                c64(1.4171536766011408, -0.12720341655479742),
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
                c64(-0.4331385293595438, -0.05088136662191896),
                c64(1.4171536766011408, -0.12720341655479742),
                c64(0.015984852758402952, 0.1780847831767164),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.5668614706404562, -0.05088136662191896),
                c64(0.4171536766011408, -0.12720341655479742),
                c64(0.015984852758402952, 0.1780847831767164),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.5668614706404562, -0.05088136662191896),
                c64(1.4171536766011408, -0.12720341655479742),
                c64(-0.984015147241597, 0.1780847831767164),
            ],
        ]];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.16485878775864288, -0.017263971883409085),
                (0, 1) | (1, 0) => c64(0.8351412122413563, 0.017263971883409127),
                _ => Complex64::ZERO,
            });
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(c1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(c1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(c1)");

        let mut cir2 = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 10.0.into(), [2]).into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let r1: Element<f64> = Resistor::new(
            "R1",
            ScalarUnitValue::new(&20.0, Scale::Base, Unit::Ohm),
            [1, 2],
            50.0.into(),
        )
        .into();

        cir2.add_elem(&p1.into(), &freq);
        let exemplar2_c = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        let exemplar2_x = points![[
            [Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE]
        ]];
        let exemplar2_s = Points::<Complex64, Ix3>::zeros((1, 1, 1));
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(p1)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(p1)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(p1)");

        cir2.add_elem(&p2.into(), &freq);
        let exemplar2_c = Points::<Complex64, Ix3>::zeros((1, 3, 3));
        let exemplar2_x = Points::<Complex64, Ix3>::from_shape_fn((1, 3, 3), |(_, j, k)| {
            if j == k && j != 0 {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        let exemplar2_s = Points::<Complex64, Ix3>::zeros((1, 2, 2));
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(p2)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(p2)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(p2)");

        cir2.add_elem(&r1.into(), &freq);
        let exemplar2_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 5, 5), |(_, j, k)| match (j, k) {
                (2, 2) | (4, 4) => 0.3333333333333333.into(),
                (2, 4) | (4, 2) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar2_x = points![[
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
                c64(1.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.5714285714285714, 0.0),
                c64(0.4285714285714284, 0.0),
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.33333333333333304, 0.0),
                c64(0.6666666666666665, 0.0),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.333333333333333, 0.0),
                c64(-0.3333333333333335, 0.0),
            ]
        ]];
        let exemplar2_s = points![[
            [c64(-0.2500000000000001, 0.0), c64(1.2499999999999993, 0.0),],
            [c64(0.24999999999999992, 0.0), c64(0.7499999999999993, 0.0),]
        ]];
        comp_pts_ix3(&exemplar2_c, &cir2.c, margin, "cir2.c(r1)");
        comp_pts_ix3(&exemplar2_x, &cir2.x, margin, "cir2.x(r1)");
        comp_pts_ix3(&exemplar2_s, &cir2.s, margin, "cir2.s(r1)");

        cir2.add_elem(&c1.into(), &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
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
                c64(-0.4331385293595438, -0.05088136662191896),
                c64(1.4171536766011408, -0.12720341655479742),
                c64(0.015984852758402952, 0.1780847831767164),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.5668614706404562, -0.05088136662191896),
                c64(0.4171536766011408, -0.12720341655479742),
                c64(0.015984852758402952, 0.1780847831767164),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.5668614706404562, -0.05088136662191896),
                c64(1.4171536766011408, -0.12720341655479742),
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
                c64(0.3309979691707785, -0.05575271255853135),
                c64(0.6654989845853893, -0.027876356279265675),
                c64(0.0035030462438317114, 0.08362906883779705),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.3309979691707785, -0.05575271255853135),
                c64(-0.33450101541461075, -0.027876356279265675),
                c64(0.0035030462438317114, 0.08362906883779705),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.3309979691707785, -0.05575271255853135),
                c64(0.6654989845853893, -0.027876356279265675),
                c64(-0.9964969537561683, 0.08362906883779705),
            ],
        ]];
        let exemplar_s = points![[
            [
                c64(-0.2536685155330628, -0.03892415872642491),
                c64(1.2536685155330627, 0.03892415872642495),
            ],
            [
                c64(0.2507337031066125, 0.007784831745284982),
                c64(0.7492662968933869, -0.007784831745284965),
            ]
        ]];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(c1)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(c1)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(c1)");
    }

    #[test]
    fn circuit_rl() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let margin = MARGIN;
        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let r1: Element<f64> = Resistor::new(
            "R1",
            ScalarUnitValue::new(&20.0, Scale::Base, Unit::Ohm),
            [1, 2],
            50.0.into(),
        )
        .into();

        cir.add_elem(&p1.into(), &freq);
        cir.add_elem(&p2.into(), &freq);
        cir.add_elem(&r1.into(), &freq);
        cir.add_elem(&l1.into(), &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
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
                c64(0.23155578829610837, 0.526474975840944),
                c64(1.6758218963854483, -0.7370649661773218),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.09262231531844334, 0.2105899903363776),
                c64(-0.7684442117038917, 0.526474975840944),
                c64(1.6758218963854483, -0.7370649661773218),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.09262231531844334, 0.2105899903363776),
                c64(0.23155578829610837, 0.526474975840944),
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
                c64(0.23155578829610837, 0.526474975840944),
                c64(1.6758218963854483, -0.7370649661773218),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.09262231531844334, 0.2105899903363776),
                c64(-0.7684442117038917, 0.526474975840944),
                c64(1.6758218963854483, -0.7370649661773218),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.09262231531844334, 0.2105899903363776),
                c64(0.23155578829610837, 0.526474975840944),
                c64(0.6758218963854483, -0.7370649661773218),
            ]
        ]];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.020739504423127997, 0.05501324410362024),
                (0, 1) | (1, 0) => c64(0.9792604955768702, -0.05501324410362057),
                _ => Complex64::ZERO,
            });
        let exemplar_y = vec![
            vec![Complex::INFINITY],
            vec![c64(0.07, -0.15915494309189532)],
            vec![c64(0.07, -0.15915494309189532)],
        ];
        let calc_y = cir.y();
        for (i, pt) in exemplar_y.iter().enumerate() {
            comp_vec_c64(
                pt,
                &calc_y[i],
                margin,
                format!("cir.y(l1)[{i}]").to_owned().as_str(),
            );
        }
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c(l1)");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x(l1)");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s(l1)");

        let mut cir2 = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 10.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let r1: Element<f64> = Resistor::new(
            "R1",
            ScalarUnitValue::new(&20.0, Scale::Base, Unit::Ohm),
            [1, 2],
            50.0.into(),
        )
        .into();

        cir2.add_elem(&p1.into(), &freq);
        cir2.add_elem(&p2.into(), &freq);
        cir2.add_elem(&r1.into(), &freq);
        cir2.add_elem(&l1.into(), &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 7, 7), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (5, 5) | (6, 6) => 0.3333333333333333.into(),
                (2, 5) | (3, 6) | (5, 2) | (6, 3) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
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
                c64(0.23155578829610837, 0.526474975840944),
                c64(1.6758218963854483, -0.7370649661773218),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.09262231531844334, 0.2105899903363776),
                c64(-0.7684442117038917, 0.526474975840944),
                c64(1.6758218963854483, -0.7370649661773218),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.09262231531844334, 0.2105899903363776),
                c64(0.23155578829610837, 0.526474975840944),
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
                c64(-0.3727824712587393, 0.6654984672870303),
                c64(0.31360876437063034, 0.33274923364351516),
                c64(1.0591737068881089, -0.9982477009305457),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.6272175287412607, 0.6654984672870303),
                c64(-0.6863912356293697, 0.33274923364351516),
                c64(1.0591737068881089, -0.9982477009305457),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.6272175287412607, 0.6654984672870303),
                c64(0.31360876437063034, 0.33274923364351516),
                c64(0.05917370688810886, -0.9982477009305457),
            ]
        ]];
        let exemplar_s = points![[
            [
                c64(-0.6044712678228585, 0.1484805774534603),
                c64(1.604471267822858, -0.14848057745346066),
            ],
            [
                c64(0.3208942535645715, -0.029696115490692093),
                c64(0.6791057464354275, 0.029696115490691954),
            ]
        ]];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c(l1)");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x(l1)");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s(l1)");
    }

    #[test]
    fn circuit_lc() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();

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
        let exemplar_x = points![[
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
                c64(-0.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-1.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-0.0808187380438485, 0.010573403382678814),
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
                c64(-0.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-1.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-0.0808187380438485, 0.010573403382678814),
                c64(1.0471625497706905, -0.26782743646522883),
            ]
        ]];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(0.004260799383993152, 0.06513558913990422),
                (0, 1) | (1, 0) => c64(0.9957392006160074, -0.06513558913990257),
                _ => Complex64::ZERO,
            });
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");

        let mut cir2 = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 0],
            50.0.into(),
        )
        .into();

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
        let exemplar_x = points![[
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
                c64(-0.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-1.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-0.0808187380438485, 0.010573403382678814),
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
                c64(1.9689082471969976, -0.24742030739945778),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.0310917528030026, 0.24742030739945778),
                c64(0.9689082471969976, -0.24742030739945778),
            ]
        ]];
        let exemplar_s = points![[
            [
                c64(-0.0013639498786206516, -0.09583962982809613),
                c64(0.9712550421795497, -0.21789113803902885),
            ],
            [
                c64(0.97125504217955, -0.21789113803902885),
                c64(-0.039707562034122956, -0.08723763248830255),
            ]
        ]];
        comp_pts_ix3(&exemplar_c, &cir2.c, margin, "cir2.c()");
        comp_pts_ix3(&exemplar_x, &cir2.x, margin, "cir2.x()");
        comp_pts_ix3(&exemplar_s, &cir2.s, margin, "cir2.s()");

        let mut cir3 = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [2, 0],
            50.0.into(),
        )
        .into();

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
        let exemplar_x = points![[
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
                c64(1.9689082471969976, -0.24742030739945778),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.0310917528030026, 0.24742030739945778),
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
                c64(-0.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-1.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-0.0808187380438485, 0.010573403382678814),
                c64(1.0471625497706905, -0.26782743646522883),
            ]
        ]];
        let exemplar_s = points![[
            [
                c64(-0.039707562034122956, -0.08723763248830282),
                c64(0.9712550421795499, -0.21789113803902913),
            ],
            [
                c64(0.9712550421795496, -0.21789113803902901),
                c64(-0.0013639498786208737, -0.0958396298280963),
            ]
        ]];
        comp_pts_ix3(&exemplar_c, &cir3.c, margin, "cir3.c()");
        comp_pts_ix3(&exemplar_x, &cir3.x, margin, "cir3.x()");
        comp_pts_ix3(&exemplar_s, &cir3.s, margin, "cir3.s()");

        let mut cir4 = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [2, 0],
            50.0.into(),
        )
        .into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();

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
        let exemplar_x = points![[
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
                c64(0.8203396752925505, -0.5718765750937107),
                c64(0.17966032470744933, 0.5718765750937107),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                c64(1.8203396752925505, -0.5718765750937107),
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
                c64(-0.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-1.0808187380438485, 0.010573403382678814),
                c64(2.0471625497706905, -0.26782743646522883),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.033656188273158014, 0.25725403308255007),
                c64(-0.0808187380438485, 0.010573403382678814),
                c64(1.0471625497706905, -0.26782743646522883),
            ]
        ]];
        let exemplar_s = points![[
            [
                c64(0.8045371871636617, -0.5888426474354542),
                c64(-0.06980717368975178, 0.03333480963552911),
            ],
            [
                c64(-0.06980717368975181, 0.0333348096355291),
                c64(-0.9636991790793159, 0.2555379447554991),
            ]
        ]];
        comp_pts_ix3(&exemplar_c, &cir4.c, margin, "cir4.c()");
        comp_pts_ix3(&exemplar_x, &cir4.x, margin, "cir4.x()");
        comp_pts_ix3(&exemplar_s, &cir4.s, margin, "cir4.s()");
    }

    #[test]
    fn circuit_rlc() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let margin = MARGIN;

        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 24.4.into(), [2]).into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 2],
            50.0.into(),
        )
        .into();
        let r1: Element<f64> = Resistor::new(
            "R1",
            ScalarUnitValue::new(&20.0, Scale::Base, Unit::Ohm),
            [1, 2],
            50.0.into(),
        )
        .into();

        cir.add_elem(&p1.into(), &freq);
        cir.add_elem(&p2.into(), &freq);
        cir.add_elem(&r1.into(), &freq);
        cir.add_elem(&c1.into(), &freq);
        cir.add_elem(&l1.into(), &freq);
        let exemplar_c =
            Points::<Complex64, Ix3>::from_shape_fn((1, 9, 9), |(_, j, k)| match (j, k) {
                (2, 2) | (3, 3) | (4, 4) | (6, 6) | (7, 7) | (8, 8) => 0.3333333333333333.into(),
                (2, 6) | (3, 7) | (4, 8) | (6, 2) | (7, 3) | (8, 4) => 0.6666666666666667.into(),
                _ => Complex64::ZERO,
            });
        let exemplar_x = points![[
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
                c64(0.24761428650741124, 0.5407604461570882),
                c64(-0.06795396179996191, 0.031116128936622463),
                c64(1.721293960689586, -0.7881807535565462),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                c64(0.09904571460296452, 0.2163041784628353),
                c64(-0.7523857134925888, 0.5407604461570882),
                c64(-0.06795396179996191, 0.031116128936622463),
                c64(1.721293960689586, -0.7881807535565462),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                c64(0.09904571460296452, 0.2163041784628353),
                c64(0.24761428650741124, 0.5407604461570882),
                c64(-1.0679539617999618, 0.031116128936622463),
                c64(1.721293960689586, -0.7881807535565462),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO
            ],
            [
                Complex64::ZERO,
                c64(0.09904571460296452, 0.2163041784628353),
                c64(0.24761428650741124, 0.5407604461570882),
                c64(-0.06795396179996191, 0.031116128936622463),
                c64(0.7212939606895861, -0.7881807535565462),
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
                c64(0.2874880164149333, 0.48304084740448944),
                c64(-0.0607007031035893, 0.03612680961457026),
                c64(1.5375667715944483, -0.9151027778424118),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.2356459150942077, 0.395935120823352),
                c64(-0.7125119835850666, 0.48304084740448944),
                c64(-0.0607007031035893, 0.03612680961457026),
                c64(1.5375667715944483, -0.9151027778424118),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.2356459150942077, 0.395935120823352),
                c64(0.2874880164149333, 0.48304084740448944),
                c64(-1.0607007031035893, 0.03612680961457026),
                c64(1.5375667715944483, -0.9151027778424118),
            ],
            [
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                c64(0.2356459150942077, 0.395935120823352),
                c64(0.2874880164149333, 0.48304084740448944),
                c64(-0.0607007031035893, 0.03612680961457026),
                c64(0.5375667715944483, -0.9151027778424118),
            ]
        ]];
        let exemplar_s = points![[
            [
                c64(-0.30224911706006186, 0.10081327871574058),
                c64(1.3022491170600594, -0.10081327871574003),
            ],
            [
                c64(0.6354975691253093, -0.049196880013280986),
                c64(0.36450243087468903, 0.04919688001328182),
            ]
        ]];
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
    }

    #[test]
    fn circuit_wilkinson() {
        let freq_points: Vec<f64> = vec![1.0];
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let margin = NumMargin {
            epsilon: 1e-4,
            relative: 1.0,
            ulps: 10,
        };

        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let p3: Element<f64> = Port::new("p3", 50.0.into(), [3]).into();
        let r1: Element<f64> = Resistor::new(
            "R1",
            ScalarUnitValue::new(&100.0, Scale::Base, Unit::Ohm),
            [2, 3],
            50.0.into(),
        )
        .into();
        let sub = MsubBuilder::new()
            .id("Msub0")
            .er(12.4)
            .tand(0.0004)
            .height(&ScalarUnitValue::new(&25e-6, Scale::Micro, Unit::Meter))
            .thickness(&ScalarUnitValue::new(&0.77e-6, Scale::Micro, Unit::Meter))
            .build()
            .unwrap();
        let gamma = c64(
            0.0,
            2.0 * std::f64::consts::PI / (3e8 / (1e9 * 12.4_f64.sqrt())),
        );
        let width = 5.915e-6;
        let ml1: Element<f64> = MlinBuilder::new()
            .width_val(width)
            .width_scale(Scale::Micro)
            .width_unit(Unit::Meter)
            .length_val(0.25)
            .length_unit(Unit::Lambda)
            .gamma(gamma)
            .sub(&sub)
            .nodes([1, 2])
            .id("ML1")
            .build()
            .unwrap()
            .into();
        let ml2: Element<f64> = MlinBuilder::new()
            .width_val(width)
            .width_scale(Scale::Micro)
            .width_unit(Unit::Meter)
            .length_val(0.25)
            .length_unit(Unit::Lambda)
            .gamma(gamma)
            .sub(&sub)
            .nodes([1, 3])
            .id("ML2")
            .build()
            .unwrap()
            .into();
        cir.add_elem(&p1.into(), &freq);
        cir.add_elem(&p2.into(), &freq);
        cir.add_elem(&p3.into(), &freq);
        cir.add_elem(&ml1.into(), &freq);
        cir.add_elem(&ml2.into(), &freq);
        cir.add_elem(&r1.into(), &freq);
        let exemplar_s = points![[
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
        ]];
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
        let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
        let margin = NumMargin {
            epsilon: 1e-11,
            relative: 1.0,
            ulps: 10,
        };

        let mut cir = Circuit::new(&freq);
        let p1: Element<f64> = Port::new("p1", 50.0.into(), [1]).into();
        let p2: Element<f64> = Port::new("p2", 50.0.into(), [2]).into();
        let l1: Element<f64> = Inductor::new(
            "L1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [1, 3],
            50.0.into(),
        )
        .into();
        let l2: Element<f64> = Inductor::new(
            "L2",
            ScalarUnitValue::new_scaled(&1.0, Scale::Nano, Unit::Henry),
            None,
            [3, 2],
            50.0.into(),
        )
        .into();
        let c1: Element<f64> = Capacitor::new(
            "C1",
            ScalarUnitValue::new_scaled(&1.0, Scale::Pico, Unit::Farad),
            None,
            [3, 0],
            50.0.into(),
        )
        .into();

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
        let exemplar_x = points![[
            [
                -Complex::ONE,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
            ],
            [
                Complex::ZERO,
                -Complex::ONE,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
            ],
            [
                Complex::ZERO,
                Complex::ZERO,
                c64(-0.9689082471969974, 0.24742030739945778),
                c64(1.9689082471969976, -0.24742030739945778),
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
            ],
            [
                Complex::ZERO,
                Complex::ZERO,
                c64(0.0310917528030026, 0.24742030739945778),
                c64(0.9689082471969976, -0.24742030739945778),
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
            ],
            [
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                c64(-0.9689082471969974, 0.24742030739945778),
                c64(1.9689082471969976, -0.24742030739945778),
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
            ],
            [
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                c64(0.0310917528030026, 0.24742030739945778),
                c64(0.9689082471969976, -0.24742030739945778),
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
            ],
            [
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                c64(0.02013669115344152, 0.0),
                c64(1.0201366911534415, 0.0),
                c64(-0.04027338230688298, -0.0),
            ],
            [
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                c64(1.0201366911534415, 0.0),
                c64(0.02013669115344152, 0.0),
                c64(-0.04027338230688298, -0.0),
            ],
            [
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                Complex::ZERO,
                c64(1.0201366911534415, 0.0),
                c64(1.0201366911534415, 0.0),
                c64(-1.040273382306883, -0.0),
            ],
        ]];
        let exemplar_s =
            Points::<Complex64, Ix3>::from_shape_fn((1, 2, 2), |(_, j, k)| match (j, k) {
                (0, 0) | (1, 1) => c64(-0.00948900668889252, -0.03252088585081872),
                (0, 1) | (1, 0) => c64(0.9594192405081049, -0.2799411932502765),
                _ => Complex64::ZERO,
            });
        let exemplar_y = vec![
            vec![Complex::INFINITY],
            vec![c64(0.02, -0.15915494309189532)],
            vec![c64(0.02, -0.15915494309189532)],
            vec![c64(0.0, -0.31202670087661105)],
        ];
        let calc_y = cir.y();
        for (i, pt) in exemplar_y.iter().enumerate() {
            comp_vec_c64(
                pt,
                &calc_y[i],
                margin,
                format!("cir.y()[{i}]").to_owned().as_str(),
            );
        }
        comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
        comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
        comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
    }

    mod xfmr_tests {
        use super::*;

        //
        // todo: Test results are completely wrong!!!!!!
        //
        #[test]
        fn circuit_xfmr_main() {
            let z0 = 50.0;
            let freq_points: Vec<f64> = vec![275.0];
            let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
            let src = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(9.4595)
                .im(-28.0873)
                .z0(z0)
                .freq(freq.freq_scalar(0))
                .build()
                .unwrap();
            let load = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(16.923)
                .im(-5.84)
                .z0(z0)
                .freq(freq.freq_scalar(0))
                .build()
                .unwrap();
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
            cir.add_elem(&Port::new("p1", src.rp().val().into(), [1]).into(), &freq);
            cir.add_elem(&Port::new("p2", load.rp().val().into(), [2]).into(), &freq);
            cir.add_elem(
                &Capacitor::new("Cp1", src.cp(), None, [1, 0], 50.0.into()).into(),
                &freq,
            );
            cir.add_elem(
                &Capacitor::new("Cp2", load.cp(), None, [2, 0], 50.0.into()).into(),
                &freq,
            );
            match qp {
                Some(qp) => cir.add_elem(
                    &Inductor::new(
                        "Lp",
                        ScalarUnitValue::new_scaled(&((1.0 - km) * lp), l_scale, Unit::Henry),
                        Some(QBuilder::new().q(qp).build().unwrap()),
                        [1, 3],
                        50.0.into(),
                    )
                    .into(),
                    &freq,
                ),
                None => cir.add_elem(
                    &Inductor::new(
                        "Lp",
                        ScalarUnitValue::new_scaled(&((1.0 - km) * lp), l_scale, Unit::Henry),
                        None,
                        [1, 3],
                        50.0.into(),
                    )
                    .into(),
                    &freq,
                ),
            }
            cir.add_elem(
                &Inductor::new(
                    "M",
                    ScalarUnitValue::new_scaled(&(km * lp), l_scale, Unit::Henry),
                    None,
                    [3, 0],
                    50.0.into(),
                )
                .into(),
                &freq,
            );
            match qs {
                Some(qs) => cir.add_elem(
                    &Inductor::new(
                        "Ls",
                        ScalarUnitValue::new_scaled(&(n2 * (1.0 - km) * ls), l_scale, Unit::Henry),
                        Some(QBuilder::new().q(qs).build().unwrap()),
                        [3, 2],
                        50.0.into(),
                    )
                    .into(),
                    &freq,
                ),
                None => cir.add_elem(
                    &Inductor::new(
                        "Ls",
                        ScalarUnitValue::new_scaled(&(n2 * (1.0 - km) * ls), l_scale, Unit::Henry),
                        None,
                        [3, 2],
                        50.0.into(),
                    )
                    .into(),
                    &freq,
                ),
            }

            // Node order:
            //  0:  gnd
            //  1:  Cp1
            //  2:  Cp2
            //  3:  M
            //  4:  p1
            //  5:  Cp1
            //  6:  Lp
            //  7:  p2
            //  8:  Cp2
            //  9:  Ls
            //  10: Lp
            //  11: M
            //  12: Ls
            let exemplar_c =
                Points::<Complex64, Ix3>::from_shape_fn((1, 13, 13), |(_, j, k)| match (j, k) {
                    (1, 1)
                    | (2, 2)
                    | (3, 3)
                    | (5, 5)
                    | (6, 6)
                    | (8, 8)
                    | (9, 9)
                    | (10, 10)
                    | (11, 11)
                    | (12, 12) => 0.3333333333333333.into(),
                    (1, 5)
                    | (5, 1)
                    | (2, 8)
                    | (8, 2)
                    | (3, 11)
                    | (11, 3)
                    | (6, 10)
                    | (10, 6)
                    | (9, 12)
                    | (12, 9) => 0.6666666666666667.into(),
                    _ => Complex64::ZERO,
                });
            let exemplar_x = points![[
                // Node 0
                [
                    -Complex64::ONE,
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
                    -Complex64::ONE,
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
                    Complex64::ZERO,
                    -Complex64::ONE,
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
                    Complex64::ZERO,
                    Complex64::ZERO,
                    -Complex64::ONE,
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
                // Node 1
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.5344721447286099, 0.8451860898696806),
                    c64(-2.509540172524624, 1.3822528177349873),
                    c64(4.044012317253234, -2.2274389076046677),
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
                    c64(0.46552785527139007, 0.8451860898696806),
                    c64(-3.509540172524624, 1.3822528177349873),
                    c64(4.044012317253234, -2.2274389076046677),
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
                    c64(0.46552785527139007, 0.8451860898696806),
                    c64(-2.509540172524624, 1.3822528177349873),
                    c64(3.044012317253234, -2.2274389076046677),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO
                ],
                // Node 2
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.4307477631518295, 0.9024723622026857),
                    c64(-0.3114364235220519, 0.4937402905398975),
                    c64(0.8806886603702224, -1.396212652742583),
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
                    c64(1.4307477631518295, 0.9024723622026857),
                    c64(-1.311436423522052, 0.4937402905398975),
                    c64(0.8806886603702224, -1.396212652742583),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(1.4307477631518295, 0.9024723622026857),
                    c64(-0.3114364235220519, 0.4937402905398975),
                    c64(-0.11931133962977758, -1.396212652742583),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                // Node 3
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
                    c64(-0.4285714285714286, 0.0),
                    c64(0.8571428571428571, 0.0),
                    c64(0.5714285714285714, 0.0),
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
                    Complex64::ZERO,
                    c64(0.5714285714285714, 0.0),
                    c64(-0.1428571428571429, 0.0),
                    c64(0.5714285714285714, 0.0),
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
                    Complex64::ZERO,
                    c64(0.5714285714285714, 0.0),
                    c64(0.8571428571428571, 0.0),
                    c64(-0.4285714285714286, 0.0),
                ],
            ]];
            let exemplar_y = vec![
                vec![Complex::INFINITY],
                vec![c64(0.01076927467744126, -0.01955208706051092)],
                vec![c64(0.05280293818350386, -0.03330649439474413)],
                vec![c64(0.0, -0.18034936906687415)],
            ];
            let exemplar_s = points![[
                [
                    c64(0.40841857286037536, 0.3461258058035369),
                    c64(1.0239905164392369, -1.5650172257301713)
                ],
                [
                    c64(0.20884510442023488, -0.31918868454332905),
                    c64(0.4807137729520777, 0.23563334355441276)
                ]
            ]];
            let sparms = cir.net().s_db();
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                sparms[(0, 0, 0)],
                sparms[(0, 0, 1)],
                sparms[(0, 1, 0)],
                sparms[(0, 1, 1)]
            );
            #[cfg(debug_assertions)]
            let debug_c = cir.debug_c();
            #[cfg(debug_assertions)]
            let debug_s = cir.debug_s();
            #[cfg(debug_assertions)]
            let debug_x = cir.debug_x();
            #[cfg(debug_assertions)]
            let debug_nodes = cir.debug_nodes();
            #[cfg(debug_assertions)]
            let debug_elements = cir.debug_elements();
            let calc_y = cir.y();
            for (i, pt) in exemplar_y.iter().enumerate() {
                comp_vec_c64(
                    pt,
                    &calc_y[i],
                    margin,
                    format!("cir.y()[{i}]").to_owned().as_str(),
                );
            }
            comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
            comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
            comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
        }

        #[test]
        fn circuit_xfmr_alt1() {
            let z0 = 50.0;
            let freq_points: Vec<f64> = vec![275.0];
            let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
            let src = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(9.4595)
                .im(-28.0873)
                .z0(z0)
                .freq(freq.freq_scalar(0))
                .build()
                .unwrap();
            let load = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(16.923)
                .im(-5.84)
                .z0(z0)
                .freq(freq.freq_scalar(0))
                .build()
                .unwrap();
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
            cir.add_elem(&Port::new("p1", src.rp().val().into(), [1]).into(), &freq);
            cir.add_elem(&Port::new("p2", load.rp().val().into(), [2]).into(), &freq);
            cir.add_elem(
                &Capacitor::new("Cp1", src.cp(), None, [1, 0], 50.0.into()).into(),
                &freq,
            );
            cir.add_elem(
                &Capacitor::new("Cp2", load.cp(), None, [2, 0], 50.0.into()).into(),
                &freq,
            );
            cir.add_elem(
                &TransformerBuilder::new()
                    // .freq(freq.freq_scalar(0))
                    .km(km)
                    .l1_val_scaled(lp, l_scale)
                    .l2_val_scaled(ls, l_scale)
                    .nodes([1, 2])
                    .id("T0")
                    .build()
                    .unwrap()
                    .into(),
                &freq,
            );

            let mut explicit = Circuit::new(&freq);
            explicit.add_elem(&Port::new("p1", src.rp().val().into(), [1]).into(), &freq);
            explicit.add_elem(&Port::new("p2", load.rp().val().into(), [2]).into(), &freq);
            explicit.add_elem(
                &Capacitor::new("Cp1", src.cp(), None, [1, 0], 50.0.into()).into(),
                &freq,
            );
            explicit.add_elem(
                &Capacitor::new("Cp2", load.cp(), None, [2, 0], 50.0.into()).into(),
                &freq,
            );
            explicit.add_elem(
                &Inductor::new(
                    "T0::lp",
                    ScalarUnitValue::new_scaled(&((1.0 - km) * lp), l_scale, Unit::Henry),
                    None,
                    [1, 3],
                    50.0.into(),
                )
                .into(),
                &freq,
            );
            explicit.add_elem(
                &Inductor::new(
                    "T0::lm",
                    ScalarUnitValue::new_scaled(&(km * lp), l_scale, Unit::Henry),
                    None,
                    [3, 0],
                    50.0.into(),
                )
                .into(),
                &freq,
            );
            explicit.add_elem(
                &Inductor::new(
                    "T0::ls",
                    ScalarUnitValue::new_scaled(&((1.0 - km) * ls), l_scale, Unit::Henry),
                    None,
                    [4, 2],
                    50.0.into(),
                )
                .into(),
                &freq,
            );
            explicit.add_elem(
                &IdealTransformerBuilder::new()
                    .n(n2.sqrt())
                    .nodes([3, 4])
                    .id("T0::ideal")
                    .build()
                    .unwrap()
                    .into(),
                &freq,
            );

            let sparms = cir.net().s_db();
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                sparms[(0, 0, 0)],
                sparms[(0, 0, 1)],
                sparms[(0, 1, 0)],
                sparms[(0, 1, 1)]
            );
            assert!(cir.elements().contains_key("T0"));
            assert!(!cir.elements().contains_key("T0::lp"));
            assert!(!cir.elements().contains_key("T0::lm"));
            assert!(!cir.elements().contains_key("T0::ls"));
            assert!(!cir.elements().contains_key("T0::ideal"));
            comp_pts_ix3(&explicit.c, &cir.c, margin, "cir.c()");
            comp_pts_ix3(&explicit.x, &cir.x, margin, "cir.x()");
            comp_pts_ix3(&explicit.s, &cir.s, margin, "cir.s()");
        }

        #[test]
        fn circuit_xfmr_alt2() {
            let z0 = 50.0;
            let freq_points: Vec<f64> = vec![275.0];
            let freq = ArrayUnitValue::new_freq_scaled(&Array1::from_vec(freq_points), Scale::Giga);
            let src = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(9.4595)
                .im(-28.0873)
                .z0(z0)
                .freq(freq.freq_scalar(0))
                .build()
                .unwrap();
            let load = ImpedanceBuilder::new()
                .kind(ImpedanceType::Z)
                .category(ComplexNumberType::ReIm)
                .mode(ImpedanceMode::Se)
                .re(16.923)
                .im(-5.84)
                .z0(z0)
                .freq(freq.freq_scalar(0))
                .build()
                .unwrap();
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
            cir.add_elem(&Port::new("p1", src.rp().val().into(), [1]).into(), &freq);
            cir.add_elem(&Port::new("p2", load.rp().val().into(), [2]).into(), &freq);
            cir.add_elem(
                &Capacitor::new("Cp1", src.cp(), None, [1, 0], 50.0.into()).into(),
                &freq,
            );
            cir.add_elem(
                &Capacitor::new("Cp2", load.cp(), None, [2, 0], 50.0.into()).into(),
                &freq,
            );
            cir.add_elem(
                &Inductor::new(
                    "Lp",
                    ScalarUnitValue::new_scaled(&((1.0 - km) * lp), l_scale, Unit::Henry),
                    None,
                    [1, 3],
                    50.0.into(),
                )
                .into(),
                &freq,
            );
            cir.add_elem(
                &Inductor::new(
                    "M",
                    ScalarUnitValue::new_scaled(&(km * lp), l_scale, Unit::Henry),
                    None,
                    [3, 0],
                    50.0.into(),
                )
                .into(),
                &freq,
            );
            cir.add_elem(
                &Inductor::new(
                    "Ls",
                    ScalarUnitValue::new_scaled(&((1.0 - km) * ls), l_scale, Unit::Henry),
                    None,
                    [4, 2],
                    50.0.into(),
                )
                .into(),
                &freq,
            );
            cir.add_elem(
                &IdealTransformerBuilder::new()
                    .n(n2.sqrt())
                    .nodes([3, 4])
                    .id("T")
                    .build()
                    .unwrap()
                    .into(),
                &freq,
            );

            // Node order:
            //  0
            //      0:  gnd
            //      1:  Cp1
            //      2:  Cp2
            //      3:  M
            //  1
            //      4:  p1
            //      5:  Cp1
            //      6:  Lp
            //  2
            //      7:  p2
            //      8:  Cp2
            //      9:  Ls
            //  3
            //      10: Lp
            //      11: M
            //      12: T
            //  4
            //      13: Ls
            //      14: T
            let exemplar_c =
                Points::<Complex64, Ix3>::from_shape_fn((1, 15, 15), |(_, j, k)| match (j, k) {
                    (1, 1)
                    | (2, 2)
                    | (3, 3)
                    | (5, 5)
                    | (6, 6)
                    | (8, 8)
                    | (9, 9)
                    | (10, 10)
                    | (11, 11)
                    | (13, 13) => 0.3333333333333333.into(),
                    (1, 5)
                    | (5, 1)
                    | (2, 8)
                    | (8, 2)
                    | (3, 11)
                    | (11, 3)
                    | (6, 10)
                    | (10, 6)
                    | (9, 13)
                    | (13, 9) => 0.6666666666666667.into(),
                    (12, 12) => ((1.0 - n2) / (1.0 + n2)).into(),
                    (14, 14) => ((n2 - 1.0) / (1.0 + n2)).into(),
                    (12, 14) | (14, 12) => (2.0 * n2.sqrt() / (1.0 + n2)).into(),
                    _ => Complex64::ZERO,
                });
            let exemplar_x = points![[
                [
                    -Complex64::ONE,
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
                    Complex64::ZERO,
                    -Complex64::ONE,
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
                    Complex64::ZERO,
                    Complex64::ZERO,
                    -Complex64::ONE,
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
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.5344721447286099, 0.8451860898696806),
                    c64(-2.509540172524624, 1.3822528177349873),
                    c64(4.044012317253234, -2.2274389076046677),
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
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.46552785527139007, 0.8451860898696806),
                    c64(-3.509540172524624, 1.3822528177349873),
                    c64(4.044012317253234, -2.2274389076046677),
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
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.46552785527139007, 0.8451860898696806),
                    c64(-2.509540172524624, 1.3822528177349873),
                    c64(3.044012317253234, -2.2274389076046677),
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
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.23562648716304502, 0.9718436903881217),
                    c64(-0.3353759470464239, 0.2637795494278684),
                    c64(1.571002434209469, -1.2356232398159903),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.764373512836955, 0.9718436903881217),
                    c64(-1.335375947046424, 0.2637795494278684),
                    c64(1.571002434209469, -1.2356232398159903),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                ],
                [
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.764373512836955, 0.9718436903881217),
                    c64(-0.3353759470464239, 0.2637795494278684),
                    c64(0.5710024342094691, -1.2356232398159903),
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
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
                    Complex64::ZERO,
                    c64(-0.9651576071391915, -0.1632786634817296),
                    c64(0.052263589291212775, -0.2449179952225944),
                    c64(1.912894017847979, 0.408196658704324),
                    Complex64::ZERO,
                    Complex64::ZERO,
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
                    Complex64::ZERO,
                    c64(0.034842392860808515, -0.1632786634817296),
                    c64(-0.9477364107087872, -0.2449179952225944),
                    c64(1.912894017847979, 0.408196658704324),
                    Complex64::ZERO,
                    Complex64::ZERO,
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
                    Complex64::ZERO,
                    c64(0.034842392860808515, -0.1632786634817296),
                    c64(0.052263589291212775, -0.2449179952225944),
                    c64(0.912894017847979, 0.408196658704324),
                    Complex64::ZERO,
                    Complex64::ZERO,
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
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(-0.960799276439074, -0.2772449285273797),
                    c64(1.960799276439074, 0.27724492852737975),
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
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    Complex64::ZERO,
                    c64(0.03920072356092601, -0.2772449285273797),
                    c64(0.960799276439074, 0.27724492852737975),
                ],
            ]];
            let exemplar_y = vec![
                vec![Complex::INFINITY],
                vec![c64(0.01076927467744126, -0.01955208706051092)],
                vec![c64(0.05280293818350386, -0.0671349823689381)],
                vec![c64(0.6036817616043335, -0.1288209779049101)],
                vec![c64(0.6036817616043335, -0.08535687913615801)],
            ];
            let exemplar_s = points![[
                [
                    c64(0.058354436800376996, 0.4000784797436545),
                    c64(1.4007413718968187, -1.4627138685566834)
                ],
                [
                    c64(0.28568426502306693, -0.2983236911977529),
                    c64(0.4022285574546962, 0.04099046006327445)
                ]
            ]];
            let sparms = cir.net().s_db();
            println!(
                "s11={:.3}\ts12={:.3}\ns21={:.3}\ts22={:.3}\n",
                sparms[(0, 0, 0)],
                sparms[(0, 0, 1)],
                sparms[(0, 1, 0)],
                sparms[(0, 1, 1)]
            );
            #[cfg(debug_assertions)]
            let debug_c = cir.debug_c();
            #[cfg(debug_assertions)]
            let debug_s = cir.debug_s();
            #[cfg(debug_assertions)]
            let debug_x = cir.debug_x();
            #[cfg(debug_assertions)]
            let debug_y = cir.y();
            #[cfg(debug_assertions)]
            let debug_nodes = cir.debug_nodes();
            let calc_y = cir.y();
            for (i, pt) in exemplar_y.iter().enumerate() {
                comp_vec_c64(
                    pt,
                    &calc_y[i],
                    margin,
                    format!("cir.y()[{i}]").to_owned().as_str(),
                );
            }
            comp_pts_ix3(&exemplar_c, &cir.c, margin, "cir.c()");
            comp_pts_ix3(&exemplar_x, &cir.x, margin, "cir.x()");
            comp_pts_ix3(&exemplar_s, &cir.s, margin, "cir.s()");
        }
    }
}
