//! Deterministic synthetic data for tests, examples and validation.
//!
//! Generates reference scans of a *flat* (angle-independent) scatterer seen
//! through known per-channel gains, with Poisson-like counting noise. After a
//! correction process, the recovered factors should approach `1/gain`, and the
//! corrected intensity spread (TFU) should fall from the XRNU level toward the
//! Poisson floor — reproducing the central result of Kato et al. (Figs. 4–5).
//!
//! Uses a self-contained `splitmix64` generator (no external crates) so results
//! are reproducible across platforms for a given seed.

use crate::correction::MsStep;
use crate::reference::Scan;

/// A small, fast, deterministic pseudo-random generator (`splitmix64`).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    gauss_cache: Option<f64>,
}

impl Rng {
    /// Create a generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            gauss_cache: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits -> [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal sample (Box–Muller, with the second value cached).
    pub fn gaussian(&mut self) -> f64 {
        if let Some(z) = self.gauss_cache.take() {
            return z;
        }
        // u1 in (0, 1] to avoid ln(0).
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.gauss_cache = Some(r * theta.sin());
        r * theta.cos()
    }

    /// A Poisson-like count for mean `lambda`, via the normal approximation
    /// (`mean = lambda`, `std = √lambda`), clamped to be non-negative.
    ///
    /// Adequate for the large counts (`>= 10⁴`) used to study XRNU; for those the
    /// normal approximation is faithful and far cheaper than exact sampling.
    pub fn poisson_like(&mut self, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        (lambda + self.gaussian() * lambda.sqrt()).max(0.0)
    }
}

/// Draw `n_channels` per-channel gains `~ 1 + relative_spread · N(0,1)`,
/// clamped to stay positive. `relative_spread = 0.01` is the ~1 % XRNU reported
/// for trimmed MYTHEN modules.
pub fn gains(n_channels: usize, relative_spread: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n_channels)
        .map(|_| (1.0 + relative_spread * rng.gaussian()).max(1e-3))
        .collect()
}

/// Build an SS/OSS reference scan of a flat scatterer.
///
/// `block_size` is the shift per step; the scan has `n_channels / block_size`
/// steps (the SS/OSS requirement). For a flat scatterer the lab position is
/// irrelevant, so each entry is `Poisson(gain[c] · intensity)`.
///
/// # Panics
/// Panics if `block_size` is zero or does not divide `n_channels`, or if
/// `gains.len() != n_channels`.
pub fn flat_scan(block_size: usize, gains: &[f64], intensity: f64, seed: u64) -> Scan {
    let n = gains.len();
    assert!(
        block_size > 0 && n.is_multiple_of(block_size),
        "bad block size"
    );
    let n_steps = n / block_size;
    flat_scan_with_steps(block_size, n_steps, gains, intensity, seed)
}

/// Build a flat-scatterer scan with an explicit shift and step count (used for
/// the finer multi-step steps, which take only two measurements).
///
/// # Panics
/// Panics if `shift_per_step` is zero or `n_steps` is zero.
pub fn flat_scan_with_steps(
    shift_per_step: usize,
    n_steps: usize,
    gains: &[f64],
    intensity: f64,
    seed: u64,
) -> Scan {
    assert!(shift_per_step > 0 && n_steps > 0, "bad scan shape");
    let n = gains.len();
    let mut rng = Rng::new(seed);
    let measurements = (0..n_steps)
        .map(|_| {
            (0..n)
                .map(|c| rng.poisson_like(gains[c] * intensity))
                .collect()
        })
        .collect();
    Scan::new(shift_per_step, measurements).expect("valid synthetic scan")
}

/// Build the full set of multi-step steps for a flat scatterer.
///
/// `block_sizes` must be coarse-to-fine and halve each time (e.g. `[80,40,20,10,5]`
/// for a 1280-channel module, or `[4,2,1]` for the 8-channel illustration). The
/// first step gets `n_channels / block_sizes[0]` measurements (full overlap);
/// every later step gets two.
pub fn flat_ms_steps(
    block_sizes: &[usize],
    gains: &[f64],
    intensity: f64,
    seed: u64,
) -> Vec<MsStep> {
    let n = gains.len();
    let mut steps = Vec::with_capacity(block_sizes.len());
    for (k, &b) in block_sizes.iter().enumerate() {
        let scan = if k == 0 {
            flat_scan(
                b,
                gains,
                intensity,
                seed ^ (k as u64).wrapping_mul(0x1234_5678),
            )
        } else {
            flat_scan_with_steps(
                b,
                2,
                gains,
                intensity,
                seed ^ (k as u64).wrapping_mul(0x1234_5678),
            )
        };
        debug_assert_eq!(scan.n_channels(), n);
        steps.push(MsStep::new(scan));
    }
    steps
}

/// A fresh independent sample measurement of a flat scatterer (one row).
pub fn flat_sample(gains: &[f64], intensity: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    gains
        .iter()
        .map(|&g| rng.poisson_like(g * intensity))
        .collect()
}
