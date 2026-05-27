//! Single-step (SS) and multi-step (MS) correction processes.
//!
//! Implements Kato et al. (2019) eq. (1)–(2) (SS) and eq. (3)–(13) (MS), in the
//! cleaner block-recursive form of Kato & Shigeta (2020) Fig. 2 and eq. (1)–(9).

use crate::error::KatoError;
use crate::reference::{ChannelFactor, CorrectionFactors, EdgeExclusion, Scan};

/// Single-step (SS) process — Kato et al. (2019) eq. (1)–(2).
///
/// The scan must shift by exactly one block per step (`shift_per_step` is taken
/// as the block size) and contain exactly `n_channels / block_size` steps, so
/// that all blocks coincide at one fully-overlapped 2θ per within-block
/// position. For within-block position `pos`, the reference is the mean over the
/// `n_blocks` channels `{pos + j·block : j}`, and the factor for each is
/// `reference / observed`.
pub fn single_step(scan: &Scan, edges: EdgeExclusion) -> Result<CorrectionFactors, KatoError> {
    let n = scan.n_channels();
    edges.validate(n)?;
    let prior = vec![1.0; n];
    let factors = full_overlap(scan, &prior, &edges)?;
    Ok(finalize(factors, &edges, n))
}

/// One step of a multi-step correction.
///
/// The block size is the scan's `shift_per_step`. Steps must be supplied
/// coarse-to-fine with block sizes that halve each time. The first step is a
/// full-overlap estimate (like SS) over all blocks; each later step compares the
/// two halves of every parent (previous-step) block.
#[derive(Debug, Clone)]
pub struct MsStep {
    /// The reference scan for this step.
    pub scan: Scan,
}

impl MsStep {
    /// Wrap a scan as a multi-step step.
    pub fn new(scan: Scan) -> Self {
        Self { scan }
    }

    fn block_size(&self) -> usize {
        self.scan.shift_per_step()
    }
}

/// Multi-step (MS) process — Kato et al. (2019) eq. (3)–(13).
///
/// `steps[0]` is the coarsest (full-overlap) step; `steps[k]` for `k >= 1` must
/// have a block size exactly half of `steps[k-1]`, and at least two
/// measurements (the two halves of each parent block are compared, with the
/// intensities first corrected by the product of the preceding steps' factors).
/// The returned factor is the product `c1·c2·…·cK` (eq. 13).
pub fn multi_step(steps: &[MsStep], edges: EdgeExclusion) -> Result<CorrectionFactors, KatoError> {
    let first = steps.first().ok_or(KatoError::NoSteps)?;
    let n = first.scan.n_channels();
    edges.validate(n)?;

    // Step 1: full overlap over all coarse blocks (eq. 4 / eq. 1).
    let mut accum = full_overlap(&first.scan, &vec![1.0; n], &edges)?;
    let mut prev_block = first.block_size();

    // Steps 2..K: split each parent block into two halves (eq. 6, 8, 10, 12).
    for (k, step) in steps.iter().enumerate().skip(1) {
        if step.scan.n_channels() != n {
            // A ragged step is treated as a wrong-count mismatch on channels.
            return Err(KatoError::WrongMeasurementCount {
                expected: n,
                found: step.scan.n_channels(),
            });
        }
        let block = step.block_size();
        if block * 2 != prev_block {
            return Err(KatoError::StepBlockNotHalving {
                step: k,
                prev: prev_block,
                cur: block,
            });
        }
        // The prior passed to this step is the running product of factors so far
        // (identity for channels not yet determined).
        let prior: Vec<f64> = accum.iter().map(|c| c.multiplier()).collect();
        let ck = within_parent_pairs(&step.scan, block, &prior, &edges)?;
        for (a, c) in accum.iter_mut().zip(&ck) {
            *a = merge_step(*a, *c);
        }
        prev_block = block;
    }

    Ok(finalize(accum, &edges, n))
}

/// Combine the running accumulated factor with this step's factor (eq. 13's
/// product `c1·c2·…`). A channel is determined overall if any step determined
/// it; undetermined steps contribute identity.
fn merge_step(acc: ChannelFactor, step: ChannelFactor) -> ChannelFactor {
    match (acc, step) {
        (ChannelFactor::Determined(a), ChannelFactor::Determined(s)) => {
            ChannelFactor::Determined(a * s)
        }
        (ChannelFactor::Determined(a), _) => ChannelFactor::Determined(a),
        (_, ChannelFactor::Determined(s)) => ChannelFactor::Determined(s),
        _ => ChannelFactor::Undetermined,
    }
}

/// Full-overlap factor estimate for one scan, given the product of prior factors.
///
/// `block = shift_per_step`, `n_blocks = n_channels / block`, and the scan must
/// have exactly `n_blocks` steps. For within-block position `pos`, channel
/// `pos + j·block` observes the common 2θ at step `n_blocks - 1 - j`. The
/// reference is the mean of the prior-corrected intensities over the included
/// blocks; the returned factor `c(i) = reference / (observed(i) · prior(i))`.
fn full_overlap(
    scan: &Scan,
    prior: &[f64],
    edges: &EdgeExclusion,
) -> Result<Vec<ChannelFactor>, KatoError> {
    let n = scan.n_channels();
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

    let mut out = vec![ChannelFactor::Undetermined; n];
    for pos in 0..block {
        // Channels at this within-block position, and their corrected readings.
        let mut chans = Vec::with_capacity(n_blocks);
        let mut vals = Vec::with_capacity(n_blocks);
        for j in 0..n_blocks {
            let ch = pos + j * block;
            if edges.is_excluded(ch, n) {
                continue;
            }
            chans.push(ch);
            vals.push(scan.at(n_blocks - 1 - j, ch) * prior[ch]);
        }
        if chans.len() < 2 {
            continue; // need at least two channels to define a reference
        }
        let reference = vals.iter().sum::<f64>() / vals.len() as f64;
        for (&ch, &v) in chans.iter().zip(&vals) {
            if v > 0.0 {
                out[ch] = ChannelFactor::Determined(reference / v);
            }
        }
    }
    Ok(out)
}

/// One refinement step: compare the two halves of each parent block.
///
/// `block` is this step's block size; the parent block size is `2·block`. Within
/// each parent and at each within-block position `r`, the two channels
/// `i0 = parent_start + r` and `i1 = i0 + block` co-observe one 2θ at every
/// consecutive step pair `(p+1, p)`: shifting by `block` brings `i0` at step
/// `p+1` onto the lab position `i1` saw at step `p`. The local factor is averaged
/// over *all* such pairs the scan provides, so no supplied frame is dropped; for
/// the canonical two-frame refinement this is exactly the single `(1, 0)` pair.
fn within_parent_pairs(
    scan: &Scan,
    block: usize,
    prior: &[f64],
    edges: &EdgeExclusion,
) -> Result<Vec<ChannelFactor>, KatoError> {
    let n = scan.n_channels();
    let parent = block * 2;
    if !n.is_multiple_of(parent) {
        return Err(KatoError::ChannelsNotDivisible {
            n_channels: n,
            block_size: parent,
        });
    }
    let n_steps = scan.n_steps();
    if n_steps < 2 {
        return Err(KatoError::WrongMeasurementCount {
            expected: 2,
            found: n_steps,
        });
    }
    let n_parents = n / parent;
    let mut out = vec![ChannelFactor::Undetermined; n];
    for g in 0..n_parents {
        for r in 0..block {
            let i0 = g * parent + r;
            let i1 = i0 + block;
            if edges.is_excluded(i0, n) || edges.is_excluded(i1, n) {
                continue; // a pair needs both members to form a reference
            }
            let (mut f0, mut f1) = (0.0, 0.0);
            let (mut n0, mut n1) = (0u32, 0u32);
            for p in 0..(n_steps - 1) {
                let v0 = scan.at(p + 1, i0) * prior[i0];
                let v1 = scan.at(p, i1) * prior[i1];
                let reference = (v0 + v1) / 2.0;
                if v0 > 0.0 {
                    f0 += reference / v0;
                    n0 += 1;
                }
                if v1 > 0.0 {
                    f1 += reference / v1;
                    n1 += 1;
                }
            }
            if n0 > 0 {
                out[i0] = ChannelFactor::Determined(f0 / f64::from(n0));
            }
            if n1 > 0 {
                out[i1] = ChannelFactor::Determined(f1 / f64::from(n1));
            }
        }
    }
    Ok(out)
}

/// Overlay edge exclusion onto a producer's per-channel result. Edge-zone
/// channels become [`ChannelFactor::Excluded`]; all others keep their
/// determined / undetermined status. This is the single owner of the `Excluded`
/// classification.
fn finalize(mut factors: Vec<ChannelFactor>, edges: &EdgeExclusion, n: usize) -> CorrectionFactors {
    for (i, f) in factors.iter_mut().enumerate() {
        if edges.is_excluded(i, n) {
            *f = ChannelFactor::Excluded;
        }
    }
    CorrectionFactors::new(factors)
}
