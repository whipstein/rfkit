# Quick Start: Avoiding Local Minima

## TL;DR - Just show me the code!

```rust
use ndarray::array;
use rfkit::minimize::f64::nelder_mead_bounded::NelderMeadBounded;
use rfkit::minimize::Minimizer;

let x0 = array![1.0, 1.0, 1.0];
let scale = array![1.0, 1.0, 1.0];
let lb = array![0.0, 0.0, 0.0];
let ub = array![10.0, 10.0, 10.0];

let mut optimizer = NelderMeadBounded::new(x0, scale, lb, ub, |x: &Array1<f64>| {
    // Your objective function
    x[0].powi(2) + x[1].powi(2) + x[2].powi(2)
});

// ⭐ ADD THESE TWO LINES TO AVOID LOCAL MINIMA ⭐
optimizer.set_random_restart(true);
optimizer.set_adaptive_simplex(true);

let result = optimizer.minimize(Some(1000));
```

That's it! Your optimizer will now:
- Automatically detect when it's stuck
- Restart from random locations
- Track the best solution globally
- Return the best result found

## What Changed?

### Before (gets stuck in local minima):
```rust
let result = optimizer.minimize(Some(1000));
```

### After (explores globally):
```rust
optimizer.set_random_restart(true);      // Auto-restart when stuck
optimizer.set_adaptive_simplex(true);    // Better initial exploration
let result = optimizer.minimize(Some(1000));
```

## How It Works

1. **Normal Nelder-Mead**: Converges to nearest local minimum
2. **With Random Restart**: When stuck, jumps to new random location
3. **Global Best Tracking**: Always returns the best solution found anywhere
4. **Adaptive Simplex**: Starts with better coverage of parameter space

## Common Settings

### Recommended (balanced):
```rust
optimizer.set_random_restart(true);
optimizer.set_adaptive_simplex(true);
optimizer.set_stagnation_threshold(50);      // Default
optimizer.minimize(Some(1000));
```

### Aggressive (find global optimum):
```rust
optimizer.set_random_restart(true);
optimizer.set_adaptive_simplex(true);
optimizer.set_stagnation_threshold(20);      // Restart sooner
optimizer.minimize(Some(2000));              // More iterations
```

### Conservative (save function evaluations):
```rust
optimizer.set_random_restart(true);
optimizer.set_adaptive_simplex(true);
optimizer.set_stagnation_threshold(100);     // Restart later
optimizer.minimize(Some(500));
```

## When to Use This

✅ **USE** random restarts when:
- Your function has multiple local minima
- You don't know the function landscape
- Your starting point might be poor
- Solution quality matters more than speed

❌ **DON'T USE** random restarts when:
- Your function is known to be unimodal (single minimum)
- You have a very good starting guess
- Function evaluations are extremely expensive
- You want exact reproduction of old behavior

## Monitoring Progress

```rust
optimizer.set_verbosity(1);  // See when restarts happen
```

You'll see output like:
```
Stagnation detected at iteration 65. Performing random restart. Best global: 2.345
Stagnation detected at iteration 142. Performing random restart. Best global: 0.891
Using best global solution: 0.891 (current: 1.234)
```

## Performance Impact

- **Function evaluations**: Typically 2-5x more
- **Solution quality**: Often 10-90% better
- **Wall time**: Depends on function cost

For your RF matching problem, if function evaluation takes 0.1s and you run 1000 iterations:
- Without restarts: ~100 seconds, may find local minimum
- With restarts: ~300 seconds, much better chance of global minimum

## Full Documentation

- [NELDER_MEAD_IMPROVEMENTS.md](NELDER_MEAD_IMPROVEMENTS.md) - Complete technical details
- [examples/nelder_mead_multimodal.md](examples/nelder_mead_multimodal.md) - Worked example

## Quick Debug

If it's not working well:

1. **Too many restarts?** Increase `stagnation_threshold`
2. **Not enough exploration?** Decrease `stagnation_threshold`
3. **Still getting bad results?** Increase `max_iters`
4. **Want to see what's happening?** Set `verbosity` to 1 or 2
