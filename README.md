# lau-vibe-field

A scalar tensor field over 2D space. What a tensor is to PyTorch, a vibe field is to PLATO.

This is the compute primitive: a flat `Vec<f64>` buffer with SIMD-friendly layout, conservation enforcement, and field operations (diffuse, advect, gradient, laplacian). GPU-ready.

## The concept in 60 seconds

A vibe field is a 2D grid of `f64` values. You **deposit** energy, it spreads via **diffusion**, moves via **advection**, and you sample it with **gradient** queries. Total energy is tracked — every deposit and withdraw is balanced.

The field is deliberately minimal. No rendering, no UI, no opinions about what the values *mean*. It's a physics-grade scalar field you can build anything on top of.

## Quick start

```rust
use lau_vibe_field::*;

// Create a 64×64 field
let mut field = VibeField::new(64, 64, 1.0);

// Deposit energy at a point
field.deposit(32, 32, 100.0);
assert_eq!(field.total_energy(), 100.0);

// Diffuse — energy spreads to neighbors
field.diffuse(0.1);

// Sample the gradient at a point
let (dx, dy) = field.gradient(33, 32);
println!("Energy flows toward ({}, {})", dx, dy);

// Withdraw energy
field.withdraw(32, 32, 10.0);

// Conservation check
assert!(field.is_conserved(90.0, 0.01));
```

## Field operations

```rust
field.deposit(x, y, amount);           // Add energy at a point
field.withdraw(x, y, amount);          // Remove energy at a point
field.diffuse(rate);                    // Spread energy to neighbors
field.advect(&vx, &vy, dt);           // Move energy along velocity fields
field.gradient(x, y);                  // → (df/dx, df/dy)
field.laplacian(x, y);                 // → ∇²f (curvature)
field.divergence(x, y);               // → ∇·F
field.normalize();                     // Scale total energy to 1.0
```

## Key types

| Type | What it does |
|------|-------------|
| `VibeField` | The core field: 2D grid + energy tracking |
| `Writer` / `Reader` | Positional agents that deposit/sample with radius |
| `VibeSnapshot` | Immutable snapshot at a given tick |
| `VibeFieldEngine` | Orchestrates multiple fields + writers/readers |
| `FieldStats` | min, max, mean, std_dev, total_energy, entropy |

## Writer/Reader pattern

```rust
let mut engine = VibeFieldEngine::new(64, 64, 1.0);

// Writers deposit energy at positions
engine.add_writer(Writer {
    id: "source".into(),
    position: (32, 32),
    radius: 4,
    strength: 50.0,
});

// Readers sample the field
engine.add_reader(Reader {
    id: "sensor".into(),
    position: (48, 48),
    radius: 2,
});

// Tick the engine
let result = engine.tick();
```

## Presets

```rust
let small  = test_10x10();       // 10×10 test field
let room   = room_64x64();       // Room-scale field
let world  = world_256x256();    // World-scale field
```

## Contributing

PRs welcome. This crate is part of the [SuperInstance](https://github.com/SuperInstance) ecosystem. The field operations are deliberately minimal — if you need a new operation, open an issue first to discuss the math.
