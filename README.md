# lau-vibe-field

**CUDA/cuDNN of PLATO — the vibe field as a first-class compute primitive.**

What a tensor is to PyTorch, a vibe field is to PLATO. A scalar field `f64` over 2D space with conservation enforcement, diffusion, advection, gradient/Laplacian computation, local extrema detection, bilinear interpolation, and resampling. Flat buffer with SIMD-friendly layout, ready for GPU offload.

---

## What This Does

`lau-vibe-field` provides the core spatial data structure for the Lau platform. A **vibe field** is a 2D grid of `f64` values representing energy, attention, emotion, or any continuous scalar quantity distributed across space. The library enforces **strict energy conservation** (deposits and withdrawals are atomic, total energy is tracked incrementally) and provides PDE-level operations: diffusion (heat equation), semi-Lagrangian advection, gradient/Laplacian computation, and divergence.

Multiple fields are managed by a `VibeFieldEngine` that ticks them forward together, and `VibeFieldPair` enables conservation-preserving energy transfers between two fields.

---

## Key Idea

A vibe field is just a flat `Vec<f64>` with width × height, but the invariant is what matters: **every operation preserves or explicitly tracks total energy**. You can deposit, withdraw, diffuse, advect, and resample — and at any point, `total_energy` is the exact sum of all cells. This makes it safe to use as a budgeting mechanism: rooms share a global energy budget, and no room can spend more than the system allows.

---

## Install

```toml
[dependencies]
lau-vibe-field = "0.1.0"
```

```bash
cargo add lau-vibe-field
```

Requires Rust 2021 edition. Only external dependency: `serde` (with `derive`).

---

## Quick Start

```rust
use lau_vibe_field::*;

// Create a 64×64 field
let mut field = VibeField::new(64, 64, 1.0);

// Deposit energy at the center
field.deposit(32, 32, 100.0);

// Compute gradient and Laplacian
let (gx, gy) = field.gradient(32, 32);
let lap = field.laplacian(32, 32);

// Diffuse for 10 steps (heat equation, rate 0.1)
for _ in 0..10 {
    field.diffuse(0.1);
}

// Energy is approximately conserved through diffusion
assert!((field.total_energy() - 100.0).abs() < 5.0);

// Find where the energy peaks are
let maxima = field.local_maxima();

// Take a snapshot for rendering
let snapshot = field.snapshot();

// Get statistics
let stats = field.field_stats();
println!("Energy: {}, Entropy: {:.3}", stats.total_energy, stats.entropy);
```

### Multi-field Engine

```rust
let mut engine = VibeFieldEngine::new(0.1);
engine.create_field("attention", 64, 64, 1.0);
engine.create_field("emotion", 64, 64, 1.0);

engine.deposit("attention", 32, 32, 50.0);
engine.deposit("emotion", 10, 10, 30.0);

engine.tick(); // diffuses all fields
engine.tick();

let stats = engine.engine_stats();
// EngineStats { field_count: 2, total_energy: 80.0, tick: 2 }
```

### Field-to-Field Transfer

```rust
let mut source = VibeField::new(10, 10, 1.0);
source.deposit(5, 5, 100.0);
let sink = VibeField::new(10, 10, 1.0);

let mut pair = VibeFieldPair::new(source, sink);
pair.transfer(5, 5, 8, 8, 30.0); // move 30 energy

assert!(pair.is_conserved(100.0, 1e-10)); // combined energy unchanged
```

---

## API Reference

### VibeField

| Method | Description |
|--------|-------------|
| `new(width, height, resolution)` | Zeroed field |
| `get(x, y)` / `set(x, y, value)` | Cell access (OOB returns 0.0 for get, no-op for set) |
| `deposit(x, y, amount)` | Add energy (fails if OOB or negative) |
| `withdraw(x, y, amount)` | Remove energy (fails if OOB, negative, or insufficient) |
| `gradient(x, y)` | Central-difference gradient → `(∂f/∂x, ∂f/∂y)` |
| `laplacian(x, y)` | Discrete Laplacian: `Σneighbors − 4·center` |
| `divergence(x, y)` | Divergence of the gradient field |
| `diffuse(rate)` | One heat equation step. Rate ∈ [0, 0.25] for stability. |
| `advect(vx, vy, dt)` | Semi-Lagrangian advection by velocity fields |
| `interpolate(x, y)` | Bilinear interpolation at continuous coordinates |
| `resample(w, h)` | Resample to new grid size via bilinear interpolation |
| `normalize()` | Scale so total energy = 1.0 |
| `neighborhood(x, y, r)` | All cells within Euclidean radius `r` |
| `local_maxima()` / `local_minima()` | Cells greater/less than all 8 neighbors |
| `snapshot()` | Immutable `VibeSnapshot` |
| `field_stats()` | `FieldStats` { min, max, mean, std_dev, total_energy, entropy } |
| `is_conserved(expected, tol)` | Check energy invariant |
| `total_energy()` | Incrementally tracked sum |

### VibeFieldPair

| Method | Description |
|--------|-------------|
| `new(source, sink)` | Two fields with conservation-preserving transfers |
| `transfer(sx, sy, dx, dy, amount)` | Move energy from source to sink (atomic with rollback) |
| `is_conserved(expected, tol)` | Check combined energy invariant |

### VibeFieldEngine

| Method | Description |
|--------|-------------|
| `new(diffusion_rate)` | Engine with global diffusion rate |
| `create_field(name, w, h, res)` | Register a named field |
| `deposit(field, x, y, amount)` | Deposit into a named field |
| `withdraw(field, x, y, amount)` | Withdraw from a named field |
| `tick()` | Diffuse all fields, advance tick counter |
| `snapshot(field)` | Snapshot of a named field (with tick) |
| `field_stats(field)` | Statistics for a named field |
| `engine_stats()` | `EngineStats` { field_count, total_energy, tick } |

### Writer / Reader

Spatial agents that deposit/sample energy at positions with a radius:

```rust
pub struct Writer { id, position, radius, strength }
pub struct Reader { id, position, radius }
```

### Pre-built Fields

| Function | Size | Description |
|----------|------|-------------|
| `test_10x10()` | 10×10 | Empty test field |
| `room_64x64()` | 64×64 | 100 energy deposited at center (32, 32) |
| `world_256x256()` | 256×256 | 25 energy at each of 4 corners |

---

## How It Works

### Storage

Flat `Vec<f64>` with row-major indexing: `data[y * width + x]`. Cache-friendly for row sweeps. Total energy is tracked incrementally — every `set`, `deposit`, and `withdraw` updates `total_energy` in O(1), so there's no need to recompute the sum.

### Conservation

- `deposit`: adds energy, increments total. Fails on OOB or negative amount.
- `withdraw`: removes energy, decrements total. Fails on OOB, negative, or insufficient funds.
- `set`: updates total by `new_value - old_value`.
- `transfer` (in `VibeFieldPair`): atomic withdraw + deposit with rollback on failure.
- `diffuse`: recomputes total from scratch after each step (diffusion can slightly violate conservation at boundaries).

### Diffusion (Heat Equation)

One explicit Euler step of the heat equation:

$$u^{n+1}_{i,j} = u^n_{i,j} + r \cdot \nabla^2 u^n_{i,j}$$

where $r$ = diffusion rate and $\nabla^2$ is the discrete Laplacian. Stability requires $r \leq 0.25$.

### Advection (Semi-Lagrangian)

For each grid cell $(x, y)$, trace backwards through the velocity field:

$$\text{src}_x = x - \Delta t \cdot v_x(x, y), \quad \text{src}_y = y - \Delta t \cdot v_y(x, y)$$

Sample the source position via bilinear interpolation. Unconditionally stable (Courant et al., 1952).

### Gradient / Laplacian

Central differences with boundary clamping (boundary gradients return 0.0):

$$\frac{\partial f}{\partial x}\bigg|_{i,j} = \frac{f_{i+1,j} - f_{i-1,j}}{2}, \quad \nabla^2 f\big|_{i,j} = f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4f_{i,j}$$

### Entropy

Shannon entropy over the probability distribution formed by normalizing cell values:

$$H = -\sum_{i} p_i \ln p_i, \quad p_i = \frac{v_i}{\sum v_j}$$

---

## The Math

### Energy Conservation Invariant

$$\sum_{(x,y)} f(x, y) = E_{\text{total}}$$

Maintained incrementally. After any sequence of deposits and withdrawals:

```rust
field.total_energy() == field.data.iter().sum()
```

### Diffusion Stability

The explicit Euler scheme for the heat equation is stable when:

$$r \leq \frac{1}{2d} = \frac{1}{4}$$

for 2D. The API recommends `rate ∈ [0, 0.25]`.

### Bilinear Interpolation

$$f(x, y) \approx (1-\alpha)(1-\beta)f_{00} + \alpha(1-\beta)f_{10} + (1-\alpha)\beta f_{01} + \alpha\beta f_{11}$$

where $\alpha = x - \lfloor x \rfloor$ and $\beta = y - \lfloor y \rfloor$.

---

## Test Coverage

**57 tests** covering:

- **Basics**: new (zeroed), set/get, set replaces, OOB returns 0
- **Deposit/Withdraw**: basic, accumulates, OOB fails, negative fails, insufficient fails
- **Gradient/Laplacian**: flat field (zero gradient), slope, boundary, peak Laplacian, boundary Laplacian
- **Conservation**: is_conserved true/false, total matches sum
- **Normalize**: basic, zero field (no panic)
- **Diffusion**: spreads energy, multiple steps
- **Neighborhood**: center, corner, radius 0
- **Extrema**: single peak, flat (all maxima), single dip
- **Interpolation**: on-grid, midpoint
- **Resample**: same size, downsize
- **Snapshot**: basic, gradient
- **FieldStats**: basic, empty
- **VibeFieldPair**: transfer, insufficient, conserved
- **VibeFieldEngine**: create, deposit/withdraw, missing field, tick, stats, snapshot tick, snapshot missing
- **Pre-built fields**: test_10x10, room_64x64, world_256x256
- **Advection**: no velocity (identity), with velocity (shifts energy)
- **Divergence**: flat field
- **Serde**: field, snapshot, engine roundtrips

```bash
cargo test
```

---

## License

MIT
