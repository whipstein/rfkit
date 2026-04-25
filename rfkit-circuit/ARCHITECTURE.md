# `rfkit-circuit` crate architecture

This document maps the current public shape of the `rfkit-circuit` crate.

Notes:

- The crate root exports `circuit`, `element`, and `prelude`.
- The crate builds on `rfkit-base` for units, points, and impedance helpers, and on `rfkit-network` for network representations.
- `mbend.rs` exists, but the `mbend` module is currently commented out of [`element.rs`](./src/element.rs).

## Diagram Legend

```mermaid
flowchart TB
    A["Concrete
    struct or
    data type"] --> B["Direct
    structural
    relationship"]
    C["Trait or
    interface"] -.-> D["Conceptual
    or supporting
    dependency"]

    E["Enum or
    category"]
    F["Builder
    type"]
    G["Collection
    or wrapper"]

    style A fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style B fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style C fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style D fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style E fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style F fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style G fill:#eef2f7,stroke:#6c7a89,color:#23303d
```

## rfkit-circuit Module Map

```mermaid
flowchart LR
    A["rfkit-circuit::lib
    crate root
    public exports"] --> C["circuit
    circuit assembly
    node map
    scattering solve"]
    A --> E["element
    element traits
    enum wrappers
    concrete elements"]
    A --> R["prelude
    common
    re-exports"]

    E --> E0["element.rs
    Elem traits
    Element enum
    ElementBuilder
    helper function"]
    E --> EC["capacitor.rs
    Capacitor"]
    E --> EG["ground.rs
    Ground"]
    E --> EI["inductor.rs
    Inductor"]
    E --> EMF["mlef.rs
    Mlef"]
    E --> EML["mlin.rs
    Mlin"]
    E --> EMS["msub.rs
    Msub"]
    E --> EP["port.rs
    Port"]
    E --> EQ["q.rs
    Q and QMode"]
    E --> ER["resistor.rs
    Resistor"]
    E --> ES["short.rs
    Short"]
    E --> ET["transformer.rs
    Transformer and
    IdealTransformer"]

    C --> C0["circuit.rs
    CircuitMap
    Circuit"]

    R --> R1["re-exports
    Circuit"]
    R --> R2["re-exports
    element traits
    Element enum
    ElementBuilder"]
    R --> R3["re-exports
    concrete elements
    and builders"]

    E -. uses rfkit-base
    units and
    element math .-> B["rfkit-base"]
    C -. builds circuit
    scattering networks
    from elements .-> N["rfkit-network"]
    E -. element calculations
    feed circuit assembly .-> C
```

## rfkit-circuit Core Dataflow

```mermaid
flowchart LR
    BUILD["ElementBuilder
    or concrete
    builders"] --> EL["Element enum
    wraps concrete
    elements"]
    EL --> ETRAIT["Elem and ElemCalc
    expose IDs nodes
    c matrices and z"]
    ETRAIT --> CIR["Circuit
    collects elements
    by node map"]

    CIR --> CMAP["CircuitMap
    maps node names
    element slots
    port slots"]
    CIR --> NODE["Per-node
    interaction data"]
    NODE --> X["Interaction
    scattering matrix X"]
    EL --> CMTX["Element
    scattering matrix C"]
    CMTX --> SOLVE["Circuit solve
    S = X * inv(I - C X)"]
    X --> SOLVE
    SOLVE --> NET["Circuit S matrix
    and derived
    rfkit-network
    views"]
```

## Public Module Inventory

```mermaid
mindmap
  root((rfkit-circuit
  Module Inventory))
    lib.rs
      circuit
      element
      prelude
    circuit
      CircuitMap
      Circuit
    element
      element.rs
      capacitor.rs
      ground.rs
      inductor.rs
      mlef.rs
      mlin.rs
      msub.rs
      port.rs
      q.rs
      resistor.rs
      short.rs
      transformer.rs
      Elem traits
      Element enum
      ElementBuilder
    prelude
      Circuit
      Element
      ElementBuilder
      concrete element types
      concrete builders
```

## Detailed Element Interfaces

```mermaid
classDiagram
    direction LR

    class ElemType {
      <<enumeration>>
      Capacitor
      Ground
      IdealTransformer
      Inductor
      Mbend
      Mlef
      Mlin
      Msub
      None
      Port
      Resistor
      Short
      Transformer
      +to_str()
    }

    class Elem {
      <<trait>>
      +id()
      +elem()
      +nodes()
    }

    class ElemCalc {
      <<trait>>
      +c(freq)
      +c_at(freq, idx)
      +c_at_freq(freq, idx)
      +net(freq)
      +z_scalar(freq)
      +z(freq)
      +z_at(freq, idx)
    }

    class Lumped {
      <<trait>>
      +val()
      +val_scaled()
      +scale()
      +unit()
      +set_val(val)
      +set_val_scaled(val)
      +set_scale(scale)
    }

    class Distributed {
      <<trait>>
      +width()
      +val()
      +set_width_val(val)
      +set_width_unit(unit)
      +set_length_val(val)
      +set_length_unit(unit)
    }

    class DistributedCalc {
      <<trait>>
      +length(freq)
      +gamma(freq)
      +er(freq)
    }

    class Term {
      <<trait>>
      +val()
      +set_val(val)
    }

    class Element {
      <<enumeration wrapper>>
      Capacitor
      Ground
      IdealTransformer
      Inductor
      Mlef
      Mlin
      Port
      Resistor
      Short
      Transformer
      +val()
    }

    class ElementBuilder {
      -id
      -unitval
      -val
      -n
      -km
      -m
      -l1
      -l2
      -q
      -q1
      -q2
      -width
      -length
      -gamma
      +build()
    }

    class QMode {
      <<enumeration>>
      None
      Series
      Parallel
    }

    class Q
    class QBuilder
    class Capacitor
    class CapacitorBuilder
    class Inductor
    class InductorBuilder
    class Resistor
    class ResistorBuilder
    class Port
    class PortBuilder
    class Ground
    class Short
    class Msub
    class MsubBuilder
    class Mlin
    class MlinBuilder
    class Mlef
    class MlefBuilder
    class Transformer
    class TransformerBuilder
    class IdealTransformer
    class IdealTransformerBuilder

    ElemCalc --|> Elem
    Lumped --|> Elem
    Distributed --|> Elem
    DistributedCalc --|> Elem
    Term --|> Elem

    Element ..|> Elem
    Element ..|> ElemCalc
    Element --> ElemType
    ElementBuilder --> Element

    Q --> QMode
    QBuilder --> Q
    Capacitor --> Q
    Inductor --> Q
    Transformer --> Q

    CapacitorBuilder --> Capacitor
    InductorBuilder --> Inductor
    ResistorBuilder --> Resistor
    PortBuilder --> Port
    MsubBuilder --> Msub
    MlinBuilder --> Mlin
    MlefBuilder --> Mlef
    TransformerBuilder --> Transformer
    IdealTransformerBuilder --> IdealTransformer

    style ElemType fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style QMode fill:#fff4de,stroke:#b08a4b,color:#3b2f1a

    style Elem fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style ElemCalc fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style Lumped fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style Distributed fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style DistributedCalc fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style Term fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f

    style Element fill:#eef2f7,stroke:#6c7a89,color:#23303d

    style ElementBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style QBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style CapacitorBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style InductorBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style ResistorBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style PortBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style MsubBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style MlinBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style MlefBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style TransformerBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style IdealTransformerBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340

    style Q fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Capacitor fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Inductor fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Resistor fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Port fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Ground fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Short fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Msub fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Mlin fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Mlef fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Transformer fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style IdealTransformer fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
```

## Detailed Circuit Interfaces

```mermaid
classDiagram
    direction LR

    class CircuitMap {
      -names
      -elem_map
      -map
      -node_map
      -port_map
      +new()
      +add(node, pos, id)
      +add_port(node, id)
      +clear()
      +elem_map()
      +map()
      +node_map()
      +port_map()
    }

    class Circuit {
      -c
      -freq
      -z0
      -map
      -nodes
      -ports_only
      -s
      -x
      -dim
      -dimx
      -elements
      +new(freq)
      +add_elem(elem, freq)
      +clear()
      +copy()
      +update(freq)
      +calc_c(freq)
      +calc_s()
      +calc_x()
    }

    class Element {
      <<wrapper>>
    }

    class Node {
      <<internal helper>>
    }

    Circuit --> CircuitMap
    Circuit --> Element
    Circuit --> Node
    CircuitMap ..> Element

    style Circuit fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style CircuitMap fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style Element fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style Node fill:#eef2f7,stroke:#6c7a89,color:#23303d
```
