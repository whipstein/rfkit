# `rfkit-minimize` crate architecture

This document maps the current public shape of the `rfkit-minimize` crate.

Notes:

- The crate root exports `error`, `minimize`, and `prelude`.
- The crate builds on `rfkit-base` point and numeric abstractions, and also uses `rfkit-network` in some optimization workflows.

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
    F["Builder like
    options or params"]
    G["Result
    object"]

    style A fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style B fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style C fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style D fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style E fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style F fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style G fill:#eef2f7,stroke:#6c7a89,color:#23303d
```

## rfkit-minimize Module Map

```mermaid
flowchart LR
    A["rfkit-minimize::lib
    crate root
    public exports"] --> E["error
    optimizer and
    inversion errors"]
    A --> M["minimize
    traits wrappers
    constraints and
    algorithms"]
    A --> R["prelude
    common
    re-exports"]

    M --> M0["minimize.rs
    objective traits
    wrappers
    constraints
    Minimizer trait"]
    M --> MB["bracket.rs
    bracketing"]
    M --> MBR["brent.rs
    Brent search"]
    M --> MDB["dbrent.rs
    derivative Brent"]
    M --> MG["golden.rs
    golden search"]
    M --> MC["conjugate_gradient.rs
    conjugate gradient"]
    M --> MQ["quasi_newton.rs
    quasi Newton"]
    M --> MN["nelder_mead.rs
    simplex search"]
    M --> MP["powell.rs
    Powell method"]
    M --> MI["interior_point.rs
    constrained search"]
    M --> MS["simplex.rs
    simplex geometry"]
    M --> MCA["cma_es.rs
    CMA ES"]

    E --> EE["MinimizerError
    InversionError"]

    R --> R1["re-exports
    objective traits
    and wrappers"]
    R --> R2["re-exports
    constraints and
    Minimizer trait"]
    R --> R3["re-exports
    algorithm structs
    method enums
    options params
    result types"]

    M -. uses rfkit-base
    points arrays and
    numeric traits .-> B["rfkit-base"]
    M -. some workflows
    use rfkit-network .-> N["rfkit-network"]
    E -. shared failure
    surface for
    algorithms .-> M
```

## rfkit-minimize Core Dataflow

```mermaid
flowchart LR
    OBJ["Objective traits
    ObjFn ObjDerFn
    ObjGradFn ObjHessFn"] --> WRAP["Function wrappers
    single dim
    multi dim
    numerical grad"]
    WRAP --> ALG["Algorithms
    bracket Brent
    Powell NM
    CG QN IP
    CMA ES"]

    CONS["Constraint types
    linear quadratic
    box constraints"] --> ALG
    BASE["Minimizer trait
    shared interface
    solve minimize
    iteration stats"] --> ALG
    ALG --> RES["Result objects
    solution value
    iterations
    convergence info"]
    ERR["MinimizerError"] --> ALG
```

## Public Module Inventory

```mermaid
mindmap
  root((rfkit-minimize
  Module Inventory))
    lib.rs
      error
      minimize
      prelude
    minimize
      minimize.rs
      bracket.rs
      brent.rs
      dbrent.rs
      golden.rs
      conjugate_gradient.rs
      quasi_newton.rs
      nelder_mead.rs
      powell.rs
      simplex.rs
      interior_point.rs
      cma_es.rs
    core interfaces
      ObjFn
      ObjDerFn
      ObjGradFn
      ObjHessFn
      Constraint
      Minimizer
    wrappers
      SingleDimFn
      SingleDimDerFn
      MultiDimFn
      MultiDimGradFn
      MultiDimNumGradFn
      MultiDimHessFn
      F1dim
      GF1dim
      HF1dim
    support
      WolfeParams
      LinearConstraint
      QuadraticConstraint
      create_box_constraints
```

## Detailed Objective And Constraint Interfaces

```mermaid
classDiagram
    direction LR

    class ObjFn {
      <<trait>>
      +call(x)
      +call_scalar(x)
    }

    class ObjDerFn {
      <<trait>>
      +df(x)
      +df_scalar(x)
    }

    class ObjGradFn {
      <<trait>>
      +grad(x)
      +grad_scalar(x)
    }

    class ObjHessFn {
      <<trait>>
      +hess(x)
      +hess_scalar(x)
    }

    class SingleDimFn {
      +new(f)
    }

    class SingleDimDerFn {
      +new(f, df)
    }

    class MultiDimFn {
      +new(f)
    }

    class MultiDimGradFn {
      +new(f, gf)
    }

    class MultiDimNumGradFn {
      -f
      -step
      -n
      +new(f, step, n)
    }

    class MultiDimHessFn {
      +new(f, gf, hf)
    }

    class F1dim {
      -p
      -xi
      -func
      +new(p, xi, func)
    }

    class GF1dim {
      -p
      -xi
      -func
      +new(p, xi, func)
    }

    class HF1dim {
      -p
      -xi
      -func
      +new(p, xi, func)
    }

    class Constraint {
      <<trait>>
      +value(x)
      +gradient(x)
      +is_satisfied(x)
    }

    class LinearConstraint {
      -a
      -b
      +new(a, b)
    }

    class QuadraticConstraint {
      -q
      -c
      -rhs
      +new(q, c, rhs)
    }

    class WolfeParams {
      +c1
      +c2
    }

    ObjDerFn --|> ObjFn
    ObjGradFn --|> ObjFn
    ObjHessFn --|> ObjGradFn

    SingleDimFn ..|> ObjFn
    SingleDimDerFn ..|> ObjFn
    SingleDimDerFn ..|> ObjDerFn
    MultiDimFn ..|> ObjFn
    MultiDimFn ..|> ObjDerFn
    MultiDimGradFn ..|> ObjFn
    MultiDimGradFn ..|> ObjGradFn
    MultiDimNumGradFn ..|> ObjFn
    MultiDimNumGradFn ..|> ObjGradFn
    MultiDimHessFn ..|> ObjFn
    MultiDimHessFn ..|> ObjGradFn
    MultiDimHessFn ..|> ObjHessFn

    LinearConstraint ..|> Constraint
    QuadraticConstraint ..|> Constraint

    style ObjFn fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style ObjDerFn fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style ObjGradFn fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style ObjHessFn fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style Constraint fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f

    style SingleDimFn fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style SingleDimDerFn fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style MultiDimFn fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style MultiDimGradFn fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style MultiDimNumGradFn fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style MultiDimHessFn fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style F1dim fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style GF1dim fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style HF1dim fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style LinearConstraint fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style QuadraticConstraint fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style WolfeParams fill:#f3ecff,stroke:#8a74b8,color:#2c2340
```

## Detailed Algorithm Interfaces

```mermaid
classDiagram
    direction LR

    class Minimizer {
      <<trait>>
      +minimize()
      +x()
      +fx()
      +iterations()
      +converged()
    }

    class Bracket
    class BracketOptions
    class BracketResult
    class Brent
    class BrentResult
    class Golden
    class GoldenResult
    class DBrent
    class DBrentMethod {
      <<enumeration>>
      Bisection
      Newton
      Secant
    }
    class DBrentResult
    class NelderMead
    class NelderMeadMethod {
      <<enumeration>>
    }
    class NelderMeadOptions
    class NelderMeadResult
    class Powell
    class PowellResult
    class ConjGrad
    class ConjGradMethod {
      <<enumeration>>
      FletcherReeves
      PolakRibiere
      HestenesStiefel
    }
    class ConjGradResult
    class QuasiNewton
    class QuasiNewtonMethod {
      <<enumeration>>
      BFGS
      DFP
    }
    class QuasiNewtonResult
    class InteriorPoint
    class InteriorPointMethod {
      <<enumeration>>
      Barrier
      PrimalDual
    }
    class InteriorPointParams
    class InteriorPointResult
    class Simplex
    class SimplexResult
    class CmaEs
    class CmaEsResult
    class MinimizerError {
      <<enumeration>>
      ConstraintViolation
      InvalidParameters
      MaxIterationsExceeded
      NumericalError
      LineSearchFailed
      NoMinimumFound
      and others
    }

    Bracket ..|> Minimizer
    Brent ..|> Minimizer
    Golden ..|> Minimizer
    DBrent ..|> Minimizer
    NelderMead ..|> Minimizer
    Powell ..|> Minimizer
    ConjGrad ..|> Minimizer
    QuasiNewton ..|> Minimizer
    InteriorPoint ..|> Minimizer
    CmaEs ..|> Minimizer

    Bracket --> BracketOptions
    Bracket --> BracketResult
    Brent --> BrentResult
    Golden --> GoldenResult
    DBrent --> DBrentMethod
    DBrent --> DBrentResult
    NelderMead --> NelderMeadMethod
    NelderMead --> NelderMeadOptions
    NelderMead --> NelderMeadResult
    Powell --> PowellResult
    ConjGrad --> ConjGradMethod
    ConjGrad --> ConjGradResult
    QuasiNewton --> QuasiNewtonMethod
    QuasiNewton --> QuasiNewtonResult
    InteriorPoint --> InteriorPointMethod
    InteriorPoint --> InteriorPointParams
    InteriorPoint --> InteriorPointResult
    Simplex --> SimplexResult
    CmaEs --> CmaEsResult
    MinimizerError ..> Minimizer

    style Minimizer fill:#eaf8ee,stroke:#6f9b76,color:#1f2d1f
    style DBrentMethod fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style NelderMeadMethod fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style ConjGradMethod fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style QuasiNewtonMethod fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style InteriorPointMethod fill:#fff4de,stroke:#b08a4b,color:#3b2f1a
    style MinimizerError fill:#fdecef,stroke:#b46a7a,color:#4a2230

    style Bracket fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Brent fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Golden fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style DBrent fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style NelderMead fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Powell fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style ConjGrad fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style QuasiNewton fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style InteriorPoint fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style Simplex fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d
    style CmaEs fill:#eaf4ff,stroke:#5b7fa3,color:#1f2d3d

    style BracketOptions fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style NelderMeadOptions fill:#f3ecff,stroke:#8a74b8,color:#2c2340
    style InteriorPointParams fill:#f3ecff,stroke:#8a74b8,color:#2c2340

    style BracketResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style BrentResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style GoldenResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style DBrentResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style NelderMeadResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style PowellResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style ConjGradResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style QuasiNewtonResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style InteriorPointResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style SimplexResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
    style CmaEsResult fill:#eef2f7,stroke:#6c7a89,color:#23303d
```
