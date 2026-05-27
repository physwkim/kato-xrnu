//! Optimized single-step (OSS) process — Kato & Shigeta (2020) eq. (10)–(11).
//!
//! The OSS process uses the *same* acquisition as SS, but instead of the single
//! fully-overlapped reference point it exploits every overlap level. For
//! `n_blocks` blocks there are `2·n_blocks − 3` usable reference points per
//! within-block position (overlap of 2, 3, …, `n_blocks`, …, 3, 2 channels).
//! Each yields a "local" factor `c_k(i) = reference_k / observed_k(i)`; the
//! "global" factor is the inverse-variance weighted mean of the local factors
//! (eq. 11), `w_k(i) = 1/σ_k(i)²`.

use crate::error::KatoError;
use crate::figures::sample_std;
use crate::reference::{ChannelFactor, CorrectionFactors, EdgeExclusion, Scan};

/// How the per-reference-point weight `w_k(i) = 1/σ_k(i)²` is determined.
///
/// Kato & Shigeta (2020) eq. (11) define `σ_k(i)` as the **sample standard
/// deviation of the local factor `c_OSS_k(i)`** over the channels that share
/// overlap level `k`. [`SampleStd`](Self::SampleStd) implements that definition
/// literally and is the default; the other two are alternatives kept for
/// cross-checking and are documented deviations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OssWeight {
    /// **Paper-faithful (eq. 11h).** `σ_k` is the sample standard deviation of
    /// the local factors `{c_OSS_k(i)}` over the channels contributing to level
    /// `k` — one weight `1/σ_k²` shared by every channel in that level. A level
    /// whose local factors are perfectly consistent (`σ_k = 0`) carries no
    /// inverse-variance weight and is skipped. **Default.**
    #[default]
    SampleStd,
    /// Propagate Poisson counting statistics through `c = reference / observed`,
    /// per channel.
    ///
    /// With the reference a mean of `n` channels, `var(reference) = reference/n`
    /// and `var(observed) = observed` (counts), giving
    /// `σ² = reference/(n·observed²) + reference²/observed³`.
    /// A *deviation* from eq. (11): the weight is per-channel from the Poisson
    /// premise rather than the paper's per-level sample std. On a flat scatterer
    /// it tracks [`SampleStd`](Self::SampleStd) closely (see
    /// `examples/xcheck_oss`).
    PoissonPropagated,
    /// Weight purely by the number of overlapping channels (`w_k = n`).
    ///
    /// A simpler proxy that ignores intensity, useful as a cross-check; favours
    /// the full-overlap level, which minimises the subset-normalisation bias of
    /// the partial-overlap levels on a flat field.
    OverlapCount,
}

/// Optimized single-step (OSS) process — Kato & Shigeta (2020) eq. (10)–(11).
///
/// The scan is the same as for [`single_step`](crate::single_step): it shifts by
/// one block per step and has `n_channels / block_size` steps.
pub fn optimized_single_step(
    scan: &Scan,
    edges: EdgeExclusion,
    weight: OssWeight,
) -> Result<CorrectionFactors, KatoError> {
    let n = scan.n_channels();
    edges.validate(n)?;
    let block = scan.shift_per_step();
    if !n.is_multiple_of(block) {
        return Err(KatoError::ChannelsNotDivisible {
            n_channels: n,
            block_size: block,
        });
    }
    let n_blocks = n / block;
    if scan.n_steps() != n_blocks {
        return Err(KatoError::WrongMeasurementCount {
            expected: n_blocks,
            found: scan.n_steps(),
        });
    }
    if n_blocks < 2 {
        return Err(KatoError::WrongMeasurementCount {
            expected: 2,
            found: n_blocks,
        });
    }

    // Inverse-variance accumulators per channel.
    let mut wsum = vec![0.0_f64; n];
    let mut wfsum = vec![0.0_f64; n];

    let last = 2 * n_blocks - 3; // highest overlap level with >= 2 channels
    for pos in 0..block {
        for s in 1..=last {
            // Contributing blocks q at overlap level s: q in [s-(n_blocks-1), s],
            // clamped to [0, n_blocks-1]; channel pos+q·block observes at step s-q.
            let q_lo = s.saturating_sub(n_blocks - 1);
            let q_hi = s.min(n_blocks - 1);

            let mut chans = Vec::with_capacity(q_hi - q_lo + 1);
            let mut vals = Vec::with_capacity(q_hi - q_lo + 1);
            for q in q_lo..=q_hi {
                let ch = pos + q * block;
                if edges.is_excluded(ch, n) {
                    continue;
                }
                chans.push(ch);
                vals.push(scan.at(s - q, ch));
            }
            let count = chans.len();
            if count < 2 {
                continue;
            }
            let reference = vals.iter().sum::<f64>() / count as f64;

            // Paper eq. (11): `σ_k` is the sample std of the local factors over
            // this level, so the weight is computed once and shared by every
            // contributing channel. A perfectly consistent level (`σ_k = 0`) has
            // no inverse-variance weight and is skipped entirely.
            let level_w = if let OssWeight::SampleStd = weight {
                let locs: Vec<f64> = chans
                    .iter()
                    .zip(&vals)
                    .filter(|&(_, &v)| v > 0.0)
                    .map(|(_, &v)| reference / v)
                    .collect();
                if locs.len() < 2 {
                    continue;
                }
                let sd = sample_std(&locs);
                if sd <= 0.0 {
                    continue;
                }
                1.0 / (sd * sd)
            } else {
                0.0 // unused for the per-channel weights below
            };

            for (&ch, &v) in chans.iter().zip(&vals) {
                if v <= 0.0 {
                    continue;
                }
                let factor = reference / v;
                // reference > 0 (a positive `v` is included) and v > 0, so every
                // branch yields a strictly positive weight.
                let w = match weight {
                    OssWeight::SampleStd => level_w,
                    OssWeight::PoissonPropagated => {
                        let variance = reference / (count as f64 * v * v)
                            + (reference * reference) / (v * v * v);
                        1.0 / variance
                    }
                    OssWeight::OverlapCount => count as f64,
                };
                wsum[ch] += w;
                wfsum[ch] += w * factor;
            }
        }
    }

    let factors = (0..n)
        .map(|ch| {
            if edges.is_excluded(ch, n) {
                ChannelFactor::Excluded
            } else if wsum[ch] > 0.0 {
                ChannelFactor::Determined(wfsum[ch] / wsum[ch])
            } else {
                ChannelFactor::Undetermined
            }
        })
        .collect();
    Ok(CorrectionFactors::new(factors))
}
