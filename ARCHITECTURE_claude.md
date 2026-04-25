# rfkit Workspace Architecture Notes

The current architecture is split by responsibility:

- `rfkit-base`: numeric foundations, `Points`, units, impedance, math helpers, utilities, macros
- `rfkit-network`: `Network`, `NetworkBuilder`, RF parameter conversions, Touchstone file I/O
- `rfkit-circuit`: `Circuit`, `Element`, element builders, circuit solving
- `rfkit-minimize`: optimization traits, options, results, and algorithms
- `rfkit`: facade and compatibility layer

The old monolithic root `src/` tree is no longer part of the build. New code should land in the implementation crate that owns the behavior, with the facade exposing only stable user-facing paths.
