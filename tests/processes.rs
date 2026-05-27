//! Integration tests for the SS / MS / OSS correction processes.
//!
//! Two kinds of checks:
//!   * **Exact (noiseless)** — the algorithm math must flatten a known-gain flat
//!     field to machine precision.
//!   * **Statistical (Poisson noise)** — the corrected spread (TFU) must fall
//!     from the ~1 % XRNU level toward the Poisson floor, reproducing the central
//!     result of Kato et al. (Figs. 4–5), with OSS no worse than SS.

use kato_xrnu::{
    CorrectionFactors, EdgeExclusion, MsStep, OssWeight, Scan, multi_step, optimized_single_step,
    single_step, synth, tfu,
};

// --- helpers -------------------------------------------------------------

/// Noiseless flat-field scan: every entry is `gain[c] · intensity`.
fn noiseless_scan(shift: usize, n_steps: usize, gains: &[f64], intensity: f64) -> Scan {
    let n = gains.len();
    let meas = (0..n_steps)
        .map(|_| (0..n).map(|c| gains[c] * intensity).collect())
        .collect();
    Scan::new(shift, meas).unwrap()
}

const GAINS8: [f64; 8] = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8, 1.02];

// --- exact (noiseless) ---------------------------------------------------

#[test]
fn ss_noiseless_flattens_exactly() {
    // 8 channels, block size 1, 8 steps (full overlap of all 8).
    let scan = noiseless_scan(1, 8, &GAINS8, 1000.0);
    let cf = single_step(&scan, EdgeExclusion::NONE).unwrap();
    let sample: Vec<f64> = GAINS8.iter().map(|g| g * 1000.0).collect();
    let corrected = cf.apply(&sample);
    assert!(
        tfu(&corrected) < 1e-9,
        "SS corrected TFU = {}",
        tfu(&corrected)
    );
}

#[test]
fn oss_noiseless_reduces_spread_far_below_xrnu() {
    // Unlike SS (one shared full-overlap reference -> exact flatten), OSS blends
    // partial-overlap reference points, each a different gain subset, and edge
    // channels participate in fewer levels. So OSS does NOT flatten a flat field
    // exactly even without noise. Its interior spread is nonetheless far below
    // the raw XRNU. (Use a realistic module size so the subsets are large.)
    let n = 1280;
    let gains = synth::gains(n, 0.01, 0x55AA);
    let interior = 16..(n - 16);
    let raw_tfu = tfu(&gains[interior.clone()]); // intensity-free gain spread, ~1 %
    let scan = noiseless_scan(8, n / 8, &gains, 1.0e6);
    let cf = optimized_single_step(
        &scan,
        EdgeExclusion::symmetric(16),
        OssWeight::PoissonPropagated,
    )
    .unwrap();
    let sample: Vec<f64> = gains.iter().map(|g| g * 1.0e6).collect();
    let corrected = cf.apply(&sample);
    let t = tfu(&corrected[interior]);
    assert!(
        t < raw_tfu * 0.2,
        "OSS noiseless interior TFU {t} not << raw {raw_tfu}"
    );
}

#[test]
fn ms_noiseless_flattens_exactly() {
    // Block sizes 4 -> 2 -> 1 reach single-channel resolution => full flatten.
    let steps = vec![
        MsStep::new(noiseless_scan(4, 2, &GAINS8, 1000.0)),
        MsStep::new(noiseless_scan(2, 2, &GAINS8, 1000.0)),
        MsStep::new(noiseless_scan(1, 2, &GAINS8, 1000.0)),
    ];
    let cf = multi_step(&steps, EdgeExclusion::NONE).unwrap();
    let sample: Vec<f64> = GAINS8.iter().map(|g| g * 1000.0).collect();
    let corrected = cf.apply(&sample);
    assert!(
        tfu(&corrected) < 1e-9,
        "MS corrected TFU = {}",
        tfu(&corrected)
    );
}

// --- statistical (Poisson noise) -----------------------------------------

struct Setup {
    raw_tfu: f64,
    floor_pct: f64,
    gains: Vec<f64>,
    sample: Vec<f64>,
    interior: std::ops::Range<usize>,
}

fn setup() -> Setup {
    let n = 1280;
    let block = 8;
    let intensity = 1.0e6_f64;
    let gains = synth::gains(n, 0.01, 0xABCD);
    let _ = block;
    let sample = synth::flat_sample(&gains, intensity, 0x2222);
    let interior = 16..(n - 16);
    let raw_tfu = tfu(&sample[interior.clone()]);
    Setup {
        raw_tfu,
        floor_pct: 100.0 / intensity.sqrt(),
        gains,
        sample,
        interior,
    }
}

fn corrected_tfu(cf: &CorrectionFactors, s: &Setup) -> f64 {
    let corrected = cf.apply(&s.sample);
    tfu(&corrected[s.interior.clone()])
}

#[test]
fn raw_data_shows_percent_level_xrnu() {
    let s = setup();
    // ~1 % gain spread dominates the ~0.1 % Poisson floor.
    assert!(s.raw_tfu > 0.7, "raw TFU = {} %", s.raw_tfu);
    assert!(s.floor_pct < 0.2);
}

#[test]
fn single_step_drives_tfu_toward_poisson_floor() {
    let s = setup();
    let calib = synth::flat_scan(8, &s.gains, 1.0e6, 0x1111);
    let cf = single_step(&calib, EdgeExclusion::symmetric(16)).unwrap();
    let t = corrected_tfu(&cf, &s);
    assert!(t < s.raw_tfu * 0.4, "SS TFU {t} not << raw {}", s.raw_tfu);
    assert!(
        t < 4.0 * s.floor_pct,
        "SS TFU {t} not near floor {}",
        s.floor_pct
    );

    // The recovered factors remove the per-channel gain: factor·gain is flat.
    let recov: Vec<f64> = s
        .interior
        .clone()
        .map(|i| cf.factors()[i] * s.gains[i])
        .collect();
    assert!(tfu(&recov) < 0.5, "SS gain recovery TFU = {}", tfu(&recov));
}

#[test]
fn optimized_single_step_is_no_worse_than_single_step() {
    let s = setup();
    let calib = synth::flat_scan(8, &s.gains, 1.0e6, 0x1111);
    let ss = single_step(&calib, EdgeExclusion::symmetric(16)).unwrap();
    let oss = optimized_single_step(
        &calib,
        EdgeExclusion::symmetric(16),
        OssWeight::PoissonPropagated,
    )
    .unwrap();
    let ss_t = corrected_tfu(&ss, &s);
    let oss_t = corrected_tfu(&oss, &s);
    assert!(
        oss_t < s.raw_tfu * 0.4,
        "OSS TFU {oss_t} not << raw {}",
        s.raw_tfu
    );
    assert!(
        oss_t <= ss_t * 1.15,
        "OSS TFU {oss_t} should be <= SS {ss_t}"
    );
}

#[test]
fn multi_step_drives_tfu_toward_poisson_floor() {
    let s = setup();
    let steps = synth::flat_ms_steps(&[80, 40, 20, 10, 5], &s.gains, 1.0e6, 0x3333);
    let cf = multi_step(&steps, EdgeExclusion::symmetric(16)).unwrap();
    let t = corrected_tfu(&cf, &s);
    assert!(t < s.raw_tfu * 0.6, "MS TFU {t} not << raw {}", s.raw_tfu);
    assert!(
        t < 6.0 * s.floor_pct,
        "MS TFU {t} not near floor {}",
        s.floor_pct
    );
}

// --- behaviour / errors --------------------------------------------------

#[test]
fn excluded_edge_channels_get_identity_factor() {
    let gains = synth::gains(64, 0.01, 7);
    let calib = synth::flat_scan(8, &gains, 1.0e5, 9);
    let cf = single_step(&calib, EdgeExclusion::symmetric(8)).unwrap();
    for i in 0..8 {
        assert!(cf.excluded()[i]);
        assert_eq!(cf.factors()[i], 1.0);
    }
    for i in 56..64 {
        assert!(cf.excluded()[i]);
        assert_eq!(cf.factors()[i], 1.0);
    }
    assert!(!cf.excluded()[32]);
}

#[test]
fn wrong_measurement_count_is_rejected() {
    // SS over 64 channels with block 8 needs 8 steps; give it 4.
    let gains = synth::gains(64, 0.0, 1);
    let scan = synth::flat_scan_with_steps(8, 4, &gains, 1.0e5, 1);
    assert!(single_step(&scan, EdgeExclusion::NONE).is_err());
}

#[test]
fn non_halving_steps_are_rejected() {
    let gains = synth::gains(64, 0.0, 1);
    let s1 = MsStep::new(synth::flat_scan(8, &gains, 1.0e5, 1)); // block 8
    let s2 = MsStep::new(synth::flat_scan_with_steps(2, 2, &gains, 1.0e5, 2)); // block 2 (not 4)
    assert!(multi_step(&[s1, s2], EdgeExclusion::NONE).is_err());
}
