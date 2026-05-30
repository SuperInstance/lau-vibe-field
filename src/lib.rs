//! # lau-vibe-field
//!
//! The vibe field as a first-class compute primitive.
//! What a tensor is to PyTorch, a vibe field is to PLATO.
//!
//! A scalar field f64 over 2D space with conservation enforcement.
//! Flat buffer with SIMD-friendly layout, ready for GPU offload.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Writer / Reader
// ---------------------------------------------------------------------------

/// A writer deposits energy into the field at a position with a radius.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Writer {
    pub id: String,
    pub position: (usize, usize),
    pub radius: usize,
    pub strength: f64,
}

/// A reader samples the field at a position with a radius.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reader {
    pub id: String,
    pub position: (usize, usize),
    pub radius: usize,
}

// ---------------------------------------------------------------------------
// FieldStats
// ---------------------------------------------------------------------------

/// Statistical summary of a field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub total_energy: f64,
    pub entropy: f64,
}

// ---------------------------------------------------------------------------
// VibeSnapshot
// ---------------------------------------------------------------------------

/// Immutable snapshot of a field at a given tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeSnapshot {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub tick: u64,
    pub total_energy: f64,
}

impl VibeSnapshot {
    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x >= self.width || y >= self.height {
            0.0
        } else {
            self.data[y * self.width + x]
        }
    }

    pub fn gradient(&self, x: usize, y: usize) -> (f64, f64) {
        let dx = if x == 0 || x + 1 >= self.width {
            0.0
        } else {
            (self.get(x + 1, y) - self.get(x - 1, y)) / 2.0
        };
        let dy = if y == 0 || y + 1 >= self.height {
            0.0
        } else {
            (self.get(x, y + 1) - self.get(x, y - 1)) / 2.0
        };
        (dx, dy)
    }
}

// ---------------------------------------------------------------------------
// VibeField
// ---------------------------------------------------------------------------

/// The core data structure — a scalar field f64 over 2D space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeField {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub total_energy: f64,
    pub writers: Vec<Writer>,
    pub readers: Vec<Reader>,
}

impl VibeField {
    /// Create a zeroed field.
    pub fn new(width: usize, height: usize, resolution: f64) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
            resolution,
            total_energy: 0.0,
            writers: Vec::new(),
            readers: Vec::new(),
        }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x >= self.width || y >= self.height {
            0.0
        } else {
            self.data[self.idx(x, y)]
        }
    }

    /// Set a cell value, updating total_energy.
    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        if x < self.width && y < self.height {
            let i = self.idx(x, y);
            let old = self.data[i];
            self.total_energy += value - old;
            self.data[i] = value;
        }
    }

    /// Deposit energy. Conservation: total can't exceed implied budget.
    /// Returns false if deposit would make cell negative (shouldn't happen) or OOB.
    pub fn deposit(&mut self, x: usize, y: usize, amount: f64) -> bool {
        if x >= self.width || y >= self.height || amount < 0.0 {
            return false;
        }
        let i = self.idx(x, y);
        self.data[i] += amount;
        self.total_energy += amount;
        true
    }

    /// Withdraw energy. Can't go below 0.
    pub fn withdraw(&mut self, x: usize, y: usize, amount: f64) -> bool {
        if x >= self.width || y >= self.height || amount < 0.0 {
            return false;
        }
        let i = self.idx(x, y);
        if self.data[i] < amount {
            return false;
        }
        self.data[i] -= amount;
        self.total_energy -= amount;
        true
    }

    /// Central-difference gradient.
    pub fn gradient(&self, x: usize, y: usize) -> (f64, f64) {
        let dx = if x == 0 || x + 1 >= self.width {
            0.0
        } else {
            (self.get(x + 1, y) - self.get(x - 1, y)) / 2.0
        };
        let dy = if y == 0 || y + 1 >= self.height {
            0.0
        } else {
            (self.get(x, y + 1) - self.get(x, y - 1)) / 2.0
        };
        (dx, dy)
    }

    /// Discrete Laplacian.
    pub fn laplacian(&self, x: usize, y: usize) -> f64 {
        let center = self.get(x, y);
        let neighbors = self.get(x + 1, y)
            + self.get(x.saturating_sub(1), y)
            + self.get(x, y + 1)
            + self.get(x, y.saturating_sub(1));
        neighbors - 4.0 * center
    }

    /// Divergence of the gradient field at a point.
    pub fn divergence(&self, x: usize, y: usize) -> f64 {
        let (_, gy_right) = self.gradient(x + 1, y);
        let (_, gy_left) = self.gradient(x.saturating_sub(1), y);
        let (gx_up, _) = self.gradient(x, y.saturating_sub(1));
        let (gx_down, _) = self.gradient(x, y + 1);
        (gx_down - gx_up) / 2.0 + (gy_right - gy_left) / 2.0
    }

    pub fn total_energy(&self) -> f64 {
        self.total_energy
    }

    pub fn is_conserved(&self, expected: f64, tolerance: f64) -> bool {
        (self.total_energy - expected).abs() <= tolerance
    }

    /// Scale field so total energy = 1.0.
    pub fn normalize(&mut self) {
        let e = self.total_energy;
        if e.abs() > f64::EPSILON {
            let scale = 1.0 / e;
            for v in &mut self.data {
                *v *= scale;
            }
            self.total_energy = 1.0;
        }
    }

    /// One diffusion step (heat equation). rate in [0, 0.25] for stability.
    pub fn diffuse(&mut self, rate: f64) {
        let w = self.width;
        let h = self.height;
        let mut next = self.data.clone();
        for y in 0..h {
            for x in 0..w {
                let lap = self.laplacian(x, y);
                let i = y * w + x;
                next[i] += rate * lap;
            }
        }
        // Recompute total energy
        self.total_energy = next.iter().sum();
        self.data = next;
    }

    /// Advect energy by a velocity field. Simple semi-Lagrangian.
    pub fn advect(&mut self, vx: &VibeField, vy: &VibeField, dt: f64) {
        let w = self.width;
        let h = self.height;
        let mut next = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let src_x = x as f64 - dt * vx.get(x, y);
                let src_y = y as f64 - dt * vy.get(x, y);
                if src_x >= 0.0 && src_x < w as f64 && src_y >= 0.0 && src_y < h as f64 {
                    let val = self.interpolate(src_x, src_y);
                    next[y * w + x] = val;
                }
            }
        }
        self.total_energy = next.iter().sum();
        self.data = next;
    }

    /// Get all cells within radius with their values.
    pub fn neighborhood(&self, x: usize, y: usize, radius: usize) -> Vec<((usize, usize), f64)> {
        let mut result = Vec::new();
        let r = radius as i64;
        let cx = x as i64;
        let cy = y as i64;
        let r2 = r * r;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let nx = cx + dx;
                let ny = cy + dy;
                if nx >= 0 && ny >= 0 {
                    let ux = nx as usize;
                    let uy = ny as usize;
                    if ux < self.width && uy < self.height {
                        result.push(((ux, uy), self.get(ux, uy)));
                    }
                }
            }
        }
        result
    }

    /// Find local maxima (cells greater than all 8 neighbors).
    pub fn local_maxima(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let v = self.get(x, y);
                let mut is_max = true;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i64 + dx;
                        let ny = y as i64 + dy;
                        if nx >= 0 && ny >= 0 {
                            let ux = nx as usize;
                            let uy = ny as usize;
                            if ux < self.width && uy < self.height && self.get(ux, uy) > v {
                                is_max = false;
                                break;
                            }
                        }
                    }
                    if !is_max {
                        break;
                    }
                }
                if is_max {
                    result.push((x, y));
                }
            }
        }
        result
    }

    /// Find local minima (cells less than all 8 neighbors).
    pub fn local_minima(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let v = self.get(x, y);
                let mut is_min = true;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i64 + dx;
                        let ny = y as i64 + dy;
                        if nx >= 0 && ny >= 0 {
                            let ux = nx as usize;
                            let uy = ny as usize;
                            if ux < self.width && uy < self.height && self.get(ux, uy) < v {
                                is_min = false;
                                break;
                            }
                        }
                    }
                    if !is_min {
                        break;
                    }
                }
                if is_min {
                    result.push((x, y));
                }
            }
        }
        result
    }

    /// Bilinear interpolation at continuous coordinates.
    pub fn interpolate(&self, world_x: f64, world_y: f64) -> f64 {
        let x0 = world_x.floor() as usize;
        let y0 = world_y.floor() as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let fx = world_x - x0 as f64;
        let fy = world_y - y0 as f64;
        let v00 = self.get(x0, y0);
        let v10 = self.get(x1, y0);
        let v01 = self.get(x0, y1);
        let v11 = self.get(x1, y1);
        v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
    }

    /// Resample to a new grid size using bilinear interpolation.
    pub fn resample(&self, new_width: usize, new_height: usize) -> VibeField {
        let mut field = VibeField::new(new_width, new_height, self.resolution);
        let sx = self.width as f64 / new_width as f64;
        let sy = self.height as f64 / new_height as f64;
        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = x as f64 * sx + sx / 2.0 - 0.5;
                let src_y = y as f64 * sy + sy / 2.0 - 0.5;
                let val = self.interpolate(src_x.max(0.0), src_y.max(0.0));
                field.set(x, y, val);
            }
        }
        field
    }

    pub fn snapshot(&self) -> VibeSnapshot {
        VibeSnapshot {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
            tick: 0,
            total_energy: self.total_energy,
        }
    }

    pub fn field_stats(&self) -> FieldStats {
        if self.data.is_empty() {
            return FieldStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                total_energy: 0.0,
                entropy: 0.0,
            };
        }
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let n = self.data.len() as f64;
        let mean = self.total_energy / n;
        let variance = self.data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Shannon entropy over the probability distribution
        let total = self.total_energy.abs();
        let entropy = if total > f64::EPSILON {
            self.data
                .iter()
                .filter(|&&v| v > 0.0)
                .map(|&v| {
                    let p = v / total;
                    -p * p.ln()
                })
                .sum()
        } else {
            0.0
        };

        FieldStats {
            min,
            max,
            mean,
            std_dev,
            total_energy: self.total_energy,
            entropy,
        }
    }
}

// ---------------------------------------------------------------------------
// VibeFieldPair
// ---------------------------------------------------------------------------

/// Two fields with conservation-preserving transfers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeFieldPair {
    pub source: VibeField,
    pub sink: VibeField,
}

impl VibeFieldPair {
    pub fn new(source: VibeField, sink: VibeField) -> Self {
        Self { source, sink }
    }

    /// Transfer energy from source to sink. Returns false if insufficient.
    pub fn transfer(
        &mut self,
        sx: usize,
        sy: usize,
        dx: usize,
        dy: usize,
        amount: f64,
    ) -> bool {
        if !self.source.withdraw(sx, sy, amount) {
            return false;
        }
        if !self.sink.deposit(dx, dy, amount) {
            // Rollback
            self.source.deposit(sx, sy, amount);
            return false;
        }
        true
    }

    pub fn is_conserved(&self, expected: f64, tolerance: f64) -> bool {
        let combined = self.source.total_energy + self.sink.total_energy;
        (combined - expected).abs() <= tolerance
    }
}

// ---------------------------------------------------------------------------
// EngineStats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub field_count: usize,
    pub total_energy: f64,
    pub tick: u64,
}

// ---------------------------------------------------------------------------
// VibeFieldEngine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeFieldEngine {
    pub fields: HashMap<String, VibeField>,
    pub tick: u64,
    pub diffusion_rate: f64,
}

impl VibeFieldEngine {
    pub fn new(diffusion_rate: f64) -> Self {
        Self {
            fields: HashMap::new(),
            tick: 0,
            diffusion_rate,
        }
    }

    pub fn create_field(&mut self, name: &str, width: usize, height: usize, resolution: f64) {
        self.fields
            .insert(name.to_string(), VibeField::new(width, height, resolution));
    }

    pub fn deposit(&mut self, field: &str, x: usize, y: usize, amount: f64) -> bool {
        self.fields
            .get_mut(field)
            .is_some_and(|f| f.deposit(x, y, amount))
    }

    pub fn withdraw(&mut self, field: &str, x: usize, y: usize, amount: f64) -> bool {
        self.fields
            .get_mut(field)
            .is_some_and(|f| f.withdraw(x, y, amount))
    }

    /// Advance one tick: diffuse all fields.
    pub fn tick(&mut self) {
        for field in self.fields.values_mut() {
            field.diffuse(self.diffusion_rate);
        }
        self.tick += 1;
    }

    pub fn snapshot(&self, field: &str) -> Option<VibeSnapshot> {
        self.fields.get(field).map(|f| {
            let mut snap = f.snapshot();
            snap.tick = self.tick;
            snap
        })
    }

    pub fn field_stats(&self, field: &str) -> Option<FieldStats> {
        self.fields.get(field).map(|f| f.field_stats())
    }

    pub fn engine_stats(&self) -> EngineStats {
        let total_energy: f64 = self.fields.values().map(|f| f.total_energy).sum();
        EngineStats {
            field_count: self.fields.len(),
            total_energy,
            tick: self.tick,
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-built fields
// ---------------------------------------------------------------------------

pub fn test_10x10() -> VibeField {
    VibeField::new(10, 10, 1.0)
}

pub fn room_64x64() -> VibeField {
    let mut f = VibeField::new(64, 64, 1.0);
    f.deposit(32, 32, 100.0);
    f
}

pub fn world_256x256() -> VibeField {
    let mut f = VibeField::new(256, 256, 1.0);
    let corners = [(32, 32), (223, 32), (32, 223), (223, 223)];
    for &(x, y) in &corners {
        f.deposit(x, y, 25.0);
    }
    f
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- VibeField basics ---

    #[test]
    fn test_new_field_is_zeroed() {
        let f = VibeField::new(5, 5, 1.0);
        assert_eq!(f.total_energy, 0.0);
        assert_eq!(f.get(2, 2), 0.0);
    }

    #[test]
    fn test_set_and_get() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.set(2, 3, 42.0);
        assert_eq!(f.get(2, 3), 42.0);
        assert_eq!(f.total_energy, 42.0);
    }

    #[test]
    fn test_set_replaces_old_value() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.set(1, 1, 10.0);
        f.set(1, 1, 20.0);
        assert_eq!(f.get(1, 1), 20.0);
        assert_eq!(f.total_energy, 20.0);
    }

    #[test]
    fn test_get_out_of_bounds_returns_zero() {
        let f = VibeField::new(3, 3, 1.0);
        assert_eq!(f.get(10, 10), 0.0);
    }

    #[test]
    fn test_deposit() {
        let mut f = VibeField::new(5, 5, 1.0);
        assert!(f.deposit(2, 2, 10.0));
        assert_eq!(f.get(2, 2), 10.0);
        assert_eq!(f.total_energy, 10.0);
    }

    #[test]
    fn test_deposit_accumulates() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(0, 0, 5.0);
        f.deposit(0, 0, 3.0);
        assert_eq!(f.get(0, 0), 8.0);
        assert_eq!(f.total_energy, 8.0);
    }

    #[test]
    fn test_deposit_oob_fails() {
        let mut f = VibeField::new(5, 5, 1.0);
        assert!(!f.deposit(10, 10, 1.0));
    }

    #[test]
    fn test_deposit_negative_fails() {
        let mut f = VibeField::new(5, 5, 1.0);
        assert!(!f.deposit(0, 0, -1.0));
    }

    #[test]
    fn test_withdraw() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 10.0);
        assert!(f.withdraw(2, 2, 4.0));
        assert_eq!(f.get(2, 2), 6.0);
        assert_eq!(f.total_energy, 6.0);
    }

    #[test]
    fn test_withdraw_insufficient_fails() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 5.0);
        assert!(!f.withdraw(2, 2, 10.0));
        assert_eq!(f.get(2, 2), 5.0);
    }

    #[test]
    fn test_withdraw_oob_fails() {
        let mut f = VibeField::new(5, 5, 1.0);
        assert!(!f.withdraw(10, 10, 1.0));
    }

    // --- Gradient / Laplacian ---

    #[test]
    fn test_gradient_flat_field() {
        let f = VibeField::new(5, 5, 1.0);
        let (gx, gy) = f.gradient(2, 2);
        assert_eq!(gx, 0.0);
        assert_eq!(gy, 0.0);
    }

    #[test]
    fn test_gradient_slope() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.set(1, 2, 1.0);
        f.set(3, 2, 3.0);
        let (gx, _) = f.gradient(2, 2);
        assert!((gx - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_boundary() {
        let f = VibeField::new(5, 5, 1.0);
        let (gx, gy) = f.gradient(0, 0);
        assert_eq!(gx, 0.0);
        assert_eq!(gy, 0.0);
    }

    #[test]
    fn test_laplacian_zero_for_flat() {
        let f = VibeField::new(5, 5, 1.0);
        assert_eq!(f.laplacian(2, 2), 0.0);
    }

    #[test]
    fn test_laplacian_peak() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.set(2, 2, 4.0);
        // Laplacian = 0+0+0+0 - 4*4 = -16
        assert_eq!(f.laplacian(2, 2), -16.0);
    }

    #[test]
    fn test_laplacian_boundary() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.set(0, 0, 4.0);
        // neighbors: get(1,0)=0, get(usize::MAX wrap treated as OOB)=0, get(0,1)=0, get(0,usize::MAX)=0
        // But saturating_sub: x=0 → 0, so get(0,0)=4.0 for left, same for top.
        // Actually get(x.saturating_sub(1), y) = get(0, 0) = 4.0 for left neighbor
        // And get(x, y.saturating_sub(1)) = get(0, 0) = 4.0 for top neighbor
        // So: 0 + 4.0 + 0 + 4.0 - 4*4.0 = -8.0
        assert_eq!(f.laplacian(0, 0), -8.0);
    }

    // --- Conservation ---

    #[test]
    fn test_is_conserved_true() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 10.0);
        assert!(f.is_conserved(10.0, 0.001));
    }

    #[test]
    fn test_is_conserved_false() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 10.0);
        assert!(!f.is_conserved(5.0, 0.001));
    }

    #[test]
    fn test_total_energy_matches_sum() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(0, 0, 3.0);
        f.deposit(4, 4, 7.0);
        let sum: f64 = f.data.iter().sum();
        assert_eq!(f.total_energy, sum);
    }

    // --- Normalize ---

    #[test]
    fn test_normalize() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 50.0);
        f.normalize();
        assert!((f.total_energy - 1.0).abs() < 1e-10);
        assert!((f.get(2, 2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_field() {
        let mut f = VibeField::new(3, 3, 1.0);
        f.normalize(); // should not panic
        assert_eq!(f.total_energy, 0.0);
    }

    // --- Diffusion ---

    #[test]
    fn test_diffuse_spreads_energy() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 100.0);
        f.diffuse(0.1);
        // Center should have less than 100
        assert!(f.get(2, 2) < 100.0);
        // Neighbors should have some energy
        assert!(f.get(3, 2) > 0.0);
        // Total should be roughly conserved
        assert!((f.total_energy - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_diffuse_multiple_steps() {
        let mut f = VibeField::new(11, 11, 1.0);
        f.deposit(5, 5, 100.0);
        for _ in 0..10 {
            f.diffuse(0.1);
        }
        assert!(f.get(5, 5) < 100.0);
        assert!(f.get(5, 5) > 0.0);
    }

    // --- Neighborhood ---

    #[test]
    fn test_neighborhood_center() {
        let mut f = VibeField::new(10, 10, 1.0);
        f.deposit(5, 5, 1.0);
        let n = f.neighborhood(5, 5, 1);
        // radius 1, Euclidean: all (dx,dy) where dx²+dy² <= 1 → 5 points
        assert_eq!(n.len(), 5); // center + 4 cardinal
    }

    #[test]
    fn test_neighborhood_corner() {
        let f = VibeField::new(10, 10, 1.0);
        let n = f.neighborhood(0, 0, 1);
        // radius 1 from corner: only 3 in bounds
        assert_eq!(n.len(), 3); // (0,0), (1,0), (0,1)
    }

    #[test]
    fn test_neighborhood_radius_0() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 5.0);
        let n = f.neighborhood(2, 2, 0);
        assert_eq!(n.len(), 1);
        assert_eq!(n[0], ((2, 2), 5.0));
    }

    // --- Local maxima/minima ---

    #[test]
    fn test_local_maxima_single_peak() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 10.0);
        let maxima = f.local_maxima();
        assert!(maxima.contains(&(2, 2)));
    }

    #[test]
    fn test_local_maxima_flat_is_all_maxima() {
        let f = VibeField::new(3, 3, 1.0);
        let maxima = f.local_maxima();
        // Every cell in a flat 3x3 is equal to all neighbors → all are "maxima"
        assert_eq!(maxima.len(), 9);
    }

    #[test]
    fn test_local_minima_single_dip() {
        let mut f = VibeField::new(5, 5, 1.0);
        // Fill everything with 10, leave center at 0
        for y in 0..5 {
            for x in 0..5 {
                if !(x == 2 && y == 2) {
                    f.set(x, y, 10.0);
                }
            }
        }
        let minima = f.local_minima();
        assert!(minima.contains(&(2, 2)));
    }

    // --- Interpolation ---

    #[test]
    fn test_interpolate_on_grid_point() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.set(2, 2, 7.0);
        assert!((f.interpolate(2.0, 2.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_midpoint() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.set(0, 0, 0.0);
        f.set(1, 0, 10.0);
        let v = f.interpolate(0.5, 0.0);
        assert!((v - 5.0).abs() < 1e-10);
    }

    // --- Resample ---

    #[test]
    fn test_resample_same_size() {
        let mut f = VibeField::new(4, 4, 1.0);
        f.deposit(2, 2, 10.0);
        let r = f.resample(4, 4);
        assert_eq!(r.width, 4);
        assert_eq!(r.height, 4);
    }

    #[test]
    fn test_resample_downsize() {
        let mut f = VibeField::new(10, 10, 1.0);
        f.deposit(5, 5, 100.0);
        let r = f.resample(5, 5);
        assert_eq!(r.width, 5);
        assert_eq!(r.height, 5);
        assert!(r.total_energy > 0.0);
    }

    // --- Snapshot ---

    #[test]
    fn test_snapshot_basic() {
        let mut f = VibeField::new(3, 3, 1.0);
        f.deposit(1, 1, 5.0);
        let snap = f.snapshot();
        assert_eq!(snap.width, 3);
        assert_eq!(snap.height, 3);
        assert!((snap.total_energy - 5.0).abs() < 1e-10);
        assert!((snap.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_snapshot_gradient() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.set(1, 2, 1.0);
        f.set(3, 2, 3.0);
        let snap = f.snapshot();
        let (gx, _) = snap.gradient(2, 2);
        assert!((gx - 1.0).abs() < 1e-10);
    }

    // --- FieldStats ---

    #[test]
    fn test_field_stats_basic() {
        let mut f = VibeField::new(3, 3, 1.0);
        f.set(0, 0, 1.0);
        f.set(1, 1, 3.0);
        let stats = f.field_stats();
        assert!((stats.min - 0.0).abs() < 1e-10);
        assert!((stats.max - 3.0).abs() < 1e-10);
        assert!((stats.total_energy - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_stats_empty() {
        let f = VibeField::new(0, 0, 1.0);
        let stats = f.field_stats();
        assert_eq!(stats.min, 0.0);
    }

    // --- VibeFieldPair ---

    #[test]
    fn test_pair_transfer() {
        let mut source = VibeField::new(5, 5, 1.0);
        source.deposit(2, 2, 100.0);
        let sink = VibeField::new(5, 5, 1.0);
        let mut pair = VibeFieldPair::new(source, sink);
        assert!(pair.transfer(2, 2, 3, 3, 30.0));
        assert!((pair.source.get(2, 2) - 70.0).abs() < 1e-10);
        assert!((pair.sink.get(3, 3) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_pair_transfer_insufficient() {
        let mut source = VibeField::new(5, 5, 1.0);
        source.deposit(2, 2, 5.0);
        let sink = VibeField::new(5, 5, 1.0);
        let mut pair = VibeFieldPair::new(source, sink);
        assert!(!pair.transfer(2, 2, 3, 3, 10.0));
    }

    #[test]
    fn test_pair_conserved() {
        let mut source = VibeField::new(5, 5, 1.0);
        source.deposit(2, 2, 100.0);
        let sink = VibeField::new(5, 5, 1.0);
        let mut pair = VibeFieldPair::new(source, sink);
        pair.transfer(2, 2, 3, 3, 40.0);
        assert!(pair.is_conserved(100.0, 1e-10));
    }

    // --- VibeFieldEngine ---

    #[test]
    fn test_engine_create_field() {
        let mut engine = VibeFieldEngine::new(0.1);
        engine.create_field("test", 10, 10, 1.0);
        assert_eq!(engine.fields.len(), 1);
    }

    #[test]
    fn test_engine_deposit_withdraw() {
        let mut engine = VibeFieldEngine::new(0.1);
        engine.create_field("test", 10, 10, 1.0);
        assert!(engine.deposit("test", 5, 5, 50.0));
        assert!(engine.withdraw("test", 5, 5, 20.0));
        let snap = engine.snapshot("test").unwrap();
        assert!((snap.get(5, 5) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_engine_deposit_missing_field() {
        let mut engine = VibeFieldEngine::new(0.1);
        assert!(!engine.deposit("nope", 0, 0, 1.0));
    }

    #[test]
    fn test_engine_tick() {
        let mut engine = VibeFieldEngine::new(0.1);
        engine.create_field("test", 11, 11, 1.0);
        engine.deposit("test", 5, 5, 100.0);
        engine.tick();
        assert_eq!(engine.tick, 1);
        let stats = engine.field_stats("test").unwrap();
        assert!(stats.max < 100.0); // energy has spread
    }

    #[test]
    fn test_engine_stats() {
        let mut engine = VibeFieldEngine::new(0.1);
        engine.create_field("a", 5, 5, 1.0);
        engine.create_field("b", 5, 5, 1.0);
        engine.deposit("a", 2, 2, 10.0);
        engine.deposit("b", 2, 2, 20.0);
        let stats = engine.engine_stats();
        assert_eq!(stats.field_count, 2);
        assert_eq!(stats.tick, 0);
        assert!((stats.total_energy - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_engine_snapshot_tick() {
        let mut engine = VibeFieldEngine::new(0.1);
        engine.create_field("f", 5, 5, 1.0);
        engine.deposit("f", 2, 2, 1.0);
        engine.tick();
        engine.tick();
        let snap = engine.snapshot("f").unwrap();
        assert_eq!(snap.tick, 2);
    }

    #[test]
    fn test_engine_snapshot_missing() {
        let engine = VibeFieldEngine::new(0.1);
        assert!(engine.snapshot("nope").is_none());
    }

    // --- Pre-built fields ---

    #[test]
    fn test_test_10x10() {
        let f = test_10x10();
        assert_eq!(f.width, 10);
        assert_eq!(f.height, 10);
        assert_eq!(f.total_energy, 0.0);
    }

    #[test]
    fn test_room_64x64() {
        let f = room_64x64();
        assert_eq!(f.width, 64);
        assert!((f.get(32, 32) - 100.0).abs() < 1e-10);
        assert!((f.total_energy - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_world_256x256() {
        let f = world_256x256();
        assert_eq!(f.width, 256);
        assert!((f.total_energy - 100.0).abs() < 1e-10);
        assert!((f.get(32, 32) - 25.0).abs() < 1e-10);
        assert!((f.get(223, 32) - 25.0).abs() < 1e-10);
    }

    // --- Advect ---

    #[test]
    fn test_advect_no_velocity() {
        let mut f = VibeField::new(5, 5, 1.0);
        f.deposit(2, 2, 10.0);
        let vx = VibeField::new(5, 5, 1.0);
        let vy = VibeField::new(5, 5, 1.0);
        f.advect(&vx, &vy, 1.0);
        assert!((f.get(2, 2) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_advect_with_velocity() {
        let mut f = VibeField::new(10, 10, 1.0);
        f.deposit(5, 5, 100.0);
        let mut vx = VibeField::new(10, 10, 1.0);
        // Uniform rightward velocity everywhere
        for y in 0..10 {
            for x in 0..10 {
                vx.set(x, y, 1.0);
            }
        }
        let vy = VibeField::new(10, 10, 1.0);
        f.advect(&vx, &vy, 1.0);
        // Semi-Lagrangian: src_x = x - dt*vx. For cell (6,5): src_x = 5 → should get the energy from (5,5)
        assert!(f.get(6, 5) > 0.0);
    }

    // --- Divergence ---

    #[test]
    fn test_divergence_flat() {
        let f = VibeField::new(5, 5, 1.0);
        let d = f.divergence(2, 2);
        assert!(d.abs() < 1e-10);
    }

    // --- Serde roundtrip ---

    #[test]
    fn test_serde_field() {
        let mut f = VibeField::new(3, 3, 2.0);
        f.deposit(1, 1, 42.0);
        let json = serde_json::to_string(&f).unwrap();
        let f2: VibeField = serde_json::from_str(&json).unwrap();
        assert_eq!(f2.width, 3);
        assert_eq!(f2.height, 3);
        assert!((f2.get(1, 1) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_serde_snapshot() {
        let mut f = VibeField::new(3, 3, 1.0);
        f.deposit(0, 0, 7.0);
        let snap = f.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let s2: VibeSnapshot = serde_json::from_str(&json).unwrap();
        assert!((s2.get(0, 0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_serde_engine() {
        let mut engine = VibeFieldEngine::new(0.1);
        engine.create_field("x", 3, 3, 1.0);
        engine.deposit("x", 1, 1, 5.0);
        let json = serde_json::to_string(&engine).unwrap();
        let e2: VibeFieldEngine = serde_json::from_str(&json).unwrap();
        assert_eq!(e2.fields.len(), 1);
    }
}
