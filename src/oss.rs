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
use crate::reference::{ChannelFactor, CorrectionFactors, EdgeExclusion, Scan};

/// How the per-reference-point weight `w_k(i) = 1/σ_k(i)²` is determined.
///
/// Kato & Shigeta (2020) define `σ_k(i)` as the sample standard deviation of the
/// local factor `c_OSS_k(i)`. Realised here as:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OssWeight {
    /// Propagate Poisson counting statistics through `c = reference / observed`.
    ///
    /// With the reference a mean of `n` channels, `var(reference) = reference/n`
    /// and `var(observed) = observed` (counts), giving
    /// `σ² = reference/(n·observed²) + reference²/observed³`.
    /// This is the paper's premise (intensity constant within Poisson noise) and
    /// favours the well-overlapped central points. **Default.**
    #[default]
    PoissonPropagated,
    /// Weight purely by the number of overlapping channels (`w_k = n`).
    ///
    /// A simpler proxy that ignores intensity, useful as a cross-check.
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
            for (&ch, &v) in chans.iter().zip(&vals) {
                if v <= 0.0 {
                    continue;
                }
                let factor = reference / v;
                let variance = match weight {
                    OssWeight::PoissonPropagated => {
                        reference / (count as f64 * v * v) + (reference * reference) / (v * v * v)
                    }
                    OssWeight::OverlapCount => 1.0 / count as f64,
                };
                if variance > 0.0 {
                    let w = 1.0 / variance;
                    wsum[ch] += w;
                    wfsum[ch] += w * factor;
                }
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
