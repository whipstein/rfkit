# `rfkit-network` crate architecture

This document maps the current public shape of the `rfkit-network` crate.

Notes:

- The crate root exports `network`, `parameter`, and `prelude`.
- The crate builds on `rfkit-base` for numeric traits, point containers, units, and comparison helpers.

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
    G["Function
    or I O"]

    style A fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style B fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style C fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style D fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style E fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style F fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style G fill:#eef2f7,stroke:#6c7a89,color:#23303d
```

- Solid line: direct structural relationship.
- Dashed line: conceptual or supporting dependency.
- Soft blue: concrete structs or data-carrying types.
- Soft green: traits or interfaces.
- Soft cream: enums and type categories.
- Soft lavender: builders.
- Soft gray-blue: functions or I/O helpers.

## rfkit-network Module Map

```mermaid
flowchart LR
    A["rfkit-network::lib
    crate root
    public exports"] --> N["network
    network types
    conversions and I O"]
    A --> P["parameter
    RF parameter
    category enum"]
    A --> R["prelude
    common
    re-exports"]

    N --> NB["builder.rs
    NetworkBuilder"]
    N --> NF["file.rs
    read_touchstone"]
    N --> NN["network.rs
    Network core"]
    N --> NPT["point.rs
    per-frequency
    network math"]
    N --> NPTS["points.rs
    batched network
    math"]
    N --> NPP["port_points.rs
    per-port
    projections"]
    N --> NMOD["network module root
    WaveType
    NetworkPoint
    NetworkPortPoints"]

    P --> PE["RFParameter
    A G H
    S T Y Z
    plus S variants"]

    R --> R1["re-exports
    RFParameter"]
    R --> R2["re-exports
    Network"]
    R --> R3["re-exports
    NetworkBuilder"]
    R --> R4["re-exports
    NetworkPoint and
    NetworkPortPoints"]
    R --> R5["re-exports
    WaveType and
    read_touchstone"]

    N -. uses rfkit-base
    points units and
    numeric traits .-> B["rfkit-base"]
    P -. selects stored
    representation
    and conversions .-> N
    NF -. parses Touchstone
    text into Network .-> NN
    NPT -. implements
    pointwise conversions .-> NMOD
    NPTS -. implements
    batched conversions .-> NMOD
    NPP -. implements
    port projections .-> NMOD
```

## rfkit-network Core Dataflow

```mermaid
flowchart LR
    F["Touchstone file
    or in-memory
    network data"] --> IO["read_touchstone
    or Network
    constructors"]
    IO --> NET["Network
    name comments
    freq z0 param
    cached matrices"]

    PARAM["RFParameter
    source representation"] --> NET
    FREQ["Frequency values
    from rfkit-base
    unit types"] --> NET
    Z0["Reference
    impedances z0"] --> NET

    NET --> STORE["Stored OnceLock
    matrices for
    a g h s y t z"]
    STORE --> CONV["Lazy conversion
    methods choose
    source and derive
    target form"]

    CONV --> PNT["point.rs
    single frequency
    conversions"]
    CONV --> PTS["points.rs
    all-frequency
    conversions"]
    CONV --> PORT["port_points.rs
    db mag re im
    per port pair"]
```

## Public Module Inventory

```mermaid
mindmap
  root((rfkit-network
  Module Inventory))
    lib.rs
      network
      parameter
      prelude
    network
      builder.rs
      file.rs
      network.rs
      point.rs
      points.rs
      port_points.rs
      Network
      NetworkBuilder
      NetworkPoint trait
      NetworkPortPoints trait
      WaveType
      read_touchstone
    parameter
      RFParameter
    prelude
      RFParameter
      Network
      NetworkBuilder
      NetworkPoint
      NetworkPortPoints
      WaveType
      read_touchstone
```

## Detailed Network Types And Interfaces

```mermaid
classDiagram
    direction LR

    class RFParameter {
      <<enumeration>>
      A
      G
      H
      S
      SPower
      SPseudo
      STraveling
      T
      Y
      Z
      +from_string(val)
      +to_str()
    }

    class WaveType {
      <<enumeration>>
      Power
      Pseudo
      Traveling
    }

    class Network {
      -name
      -comments
      -nports
      -port_names
      -ports
      -freq
      -npts
      -dim
      -z0
      -param
      -a
      -g
      -h
      -s_power
      -s_pseudo
      -s_traveling
      -y
      -t
      -z
      +new(freq, z0, param, net, name, comments)
      +new_from(freq, z0, param, format, net, name, comments)
      +new_w_port_names(...)
      +builder()
      +a()
      +g()
      +h()
      +s()
      +s_power()
      +s_pseudo()
      +s_traveling()
      +t()
      +y()
      +z()
    }

    class NetworkBuilder {
      -name
      -comments
      -nports
      -port_names
      -ports
      -freq
      -npts
      -dim
      -z0
      -param
      -net
      +new()
      +name(name)
      +comments(comments)
      +port_names(names)
      +freq(freq)
      +z0(z0)
      +net(net, param)
      +build()
    }

    class NetworkPoint {
      <<trait>>
      +a_to_g()
      +a_to_h()
      +a_to_s(z0)
      +a_to_t(z0)
      +a_to_y()
      +a_to_z()
      +check_dims(data)
      +connect(p1, net, p2)
      +db()
      +deg()
      +from_db(data)
      +from_magang(data)
      +from_reim(data)
      +g_to_a()
      +h_to_a()
      +s_to_s(z0, from, to)
      +s_to_t()
      +s_to_y(z0)
      +s_to_z(z0)
      +y_to_s(z0)
      +z_to_s(z0)
    }

    class NetworkPortPoints {
      <<trait>>
      +db()
      +deg()
      +im()
      +mag()
      +rad()
      +re()
    }

    class read_touchstone {
      <<function>>
      +read_touchstone(file_path)
    }

    NetworkBuilder --> Network
    Network --> RFParameter
    Network --> WaveType
    read_touchstone --> Network
    read_touchstone --> RFParameter
    NetworkPoint ..> RFParameter
    NetworkPoint ..> WaveType

    style Network fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style RFParameter fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style WaveType fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style NetworkBuilder fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style NetworkPoint fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style NetworkPortPoints fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style read_touchstone fill:#eef2f7,stroke:#6c7a89,color:#23303d
```
