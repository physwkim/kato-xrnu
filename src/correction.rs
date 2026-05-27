//! Single-step (SS) and multi-step (MS) correction processes.
//!
//! Implements Kato et al. (2019) eq. (1)–(2) (SS) and eq. (3)–(13) (MS), in the
//! cleaner block-recursive form of Kato & Shigeta (2020) Fig. 2 and eq. (1)–(9).

use crate::error::KatoError;
use crate::reference::{CorrectionFactors, EdgeExclusion, Scan};

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
    Ok(CorrectionFactors::new(factors, excluded_mask(n, &edges)))
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
        let ck = within_parent_pairs(&step.scan, block, &accum, &edges)?;
        for (a, c) in accum.iter_mut().zip(&ck) {
            *a *= c;
        }
        prev_block = block;
    }

    Ok(CorrectionFactors::new(accum, excluded_mask(n, &edges)))
}

/// Full-overlap factor estimate for one scan, given the product of prior factors.
///
/// `block = shift_per_step`, `n_blocks = n_channels / block`, and the scan must
/// have exactly `n_blocks` steps. For within-block position `pos`, channel
/// `pos + j·block` observes the common 2θ at step `n_blocks - 1 - j`. The
/// reference is the mean of the prior-corrected intensities over the included
/// blocks; the returned factor `c(i) = reference / (observed(i) · prior(i))`.
fn full_overlap(scan: &Scan, prior: &[f64], edges: &EdgeExclusion) -> Result<Vec<f64>, KatoError> {
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

    let mut out = vec![1.0; n];
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
                out[ch] = reference / v;
            }
        }
    }
    Ok(out)
}

/// One refinement step: compare the two halves of each parent block.
///
/// `block` is this step's block size; the parent block size is `2·block`. Within
/// each parent and at each within-block position `r`, the two channels
/// `i0 = parent_start + r` and `i1 = i0 + block` observe the same 2θ — `i0` at
/// measurement 1 and `i1` at measurement 0 (the detector having shifted by
/// `block`). The reference is the mean of their prior-corrected readings.
fn within_parent_pairs(
    scan: &Scan,
    block: usize,
    prior: &[f64],
    edges: &EdgeExclusion,
) -> Result<Vec<f64>, KatoError> {
    let n = scan.n_channels();
    let parent = block * 2;
    if !n.is_multiple_of(parent) {
        return Err(KatoError::ChannelsNotDivisible {
            n_channels: n,
            block_size: parent,
        });
    }
    if scan.n_steps() < 2 {
        return Err(KatoError::WrongMeasurementCount {
            expected: 2,
            found: scan.n_steps(),
        });
    }
    let n_parents = n / parent;
    let mut out = vec![1.0; n];
    for g in 0..n_parents {
        for r in 0..block {
            let i0 = g * parent + r;
            let i1 = i0 + block;
            if edges.is_excluded(i0, n) || edges.is_excluded(i1, n) {
                continue; // a pair needs both members to form a reference
            }
            let v0 = scan.at(1, i0) * prior[i0];
            let v1 = scan.at(0, i1) * prior[i1];
            let reference = (v0 + v1) / 2.0;
            if v0 > 0.0 {
                out[i0] = reference / v0;
            }
            if v1 > 0.0 {
                out[i1] = reference / v1;
            }
        }
    }
    Ok(out)
}

fn excluded_mask(n: usize, edges: &EdgeExclusion) -> Vec<bool> {
    (0..n).map(|c| edges.is_excluded(c, n)).collect()
}
