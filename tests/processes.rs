//! Integration tests for the SS / MS / OSS correction processes.
//!
//! Two kinds of checks:
//!   * **Exact (noiseless)** — the algorithm math must flatten a known-gain flat
//!     field to machine precision.
//!   * **Statistical (Poisson noise)** — the corrected spread (TFU) must fall
//!     from the ~1 % XRNU level toward the Poisson floor, reproducing the central
//!     result of Kato et al. (Figs. 4–5), with OSS no worse than SS.

use kato_xrnu::{
    ChannelFactor, CorrectionFactors, EdgeExclusion, MsStep, OssWeight, Scan, multi_step,
    optimized_single_step, single_step, synth, tfu,
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
fn oss_sample_std_matches_paper_eq11_reference() {
    // The default weight is the paper-faithful eq.(11) sample-std definition.
    assert_eq!(OssWeight::default(), OssWeight::SampleStd);

    // These factors were cross-verified against an independent Wolfram Language
    // implementation of eq.(10)-(11) on this exact noiseless 8-channel case
    // (see examples/xcheck_oss), agreeing to ~2e-16.
    let scan = noiseless_scan(1, 8, &GAINS8, 1000.0);
    let cf = optimized_single_step(&scan, EdgeExclusion::NONE, OssWeight::SampleStd).unwrap();
    let wolfram = [
        1.019_590_268_391_508_3,
        0.925_985_222_505_391_8,
        1.118_115_807_949_738,
        0.959_162_871_302_379,
        1.057_813_959_519_594_3,
        0.839_098_869_525_341,
        1.238_079_866_158_646_5,
        0.969_368_643_187_109,
    ];
    for (i, &w) in wolfram.iter().enumerate() {
        let got = cf.channels()[i].value().expect("determined");
        assert!(
            (got - w).abs() < 1e-12,
            "ch{i}: SampleStd {got} vs Wolfram eq.11 {w}"
        );
    }
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
        .map(|i| cf.channels()[i].value().expect("interior determined") * s.gains[i])
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

#[test]
fn ms_refinement_uses_all_frames_not_just_first_two() {
    // F3 regression: a refinement step with more than two frames must use the
    // extra co-observations, not silently drop them. Two MS runs whose
    // refinement scan differs only by an added third frame (distinct values)
    // must yield different factors.
    let gains = synth::gains(4, 0.0, 1); // unit gains; refinement values hand-set
    let coarse = synth::flat_scan(2, &gains, 1.0e5, 1); // block 2, full overlap

    let f0 = vec![10.0, 20.0, 30.0, 40.0];
    let f1 = vec![11.0, 19.0, 29.0, 41.0];
    let f2 = vec![13.0, 17.0, 33.0, 37.0]; // distinct third frame

    let refine2 = Scan::new(1, vec![f0.clone(), f1.clone()]).unwrap();
    let refine3 = Scan::new(1, vec![f0, f1, f2]).unwrap();

    let cf2 = multi_step(
        &[MsStep::new(coarse.clone()), MsStep::new(refine2)],
        EdgeExclusion::NONE,
    )
    .unwrap();
    let cf3 = multi_step(
        &[MsStep::new(coarse), MsStep::new(refine3)],
        EdgeExclusion::NONE,
    )
    .unwrap();

    let v2 = cf2.channels()[0].value().expect("determined");
    let v3 = cf3.channels()[0].value().expect("determined");
    assert!(
        (v2 - v3).abs() > 1e-9,
        "third refinement frame was ignored: {v2} == {v3}"
    );
}

// --- behaviour / errors --------------------------------------------------

#[test]
fn excluded_edge_channels_are_flagged_excluded() {
    let gains = synth::gains(64, 0.01, 7);
    let calib = synth::flat_scan(8, &gains, 1.0e5, 9);
    let cf = single_step(&calib, EdgeExclusion::symmetric(8)).unwrap();
    for i in 0..8 {
        assert_eq!(cf.channels()[i], ChannelFactor::Excluded);
        assert_eq!(cf.channels()[i].multiplier(), 1.0); // identity on apply
    }
    for i in 56..64 {
        assert_eq!(cf.channels()[i], ChannelFactor::Excluded);
    }
    // An interior channel is determined, not excluded.
    assert!(cf.channels()[32].is_determined());
    assert!(!cf.excluded()[32]);
}

#[test]
fn channel_with_no_reference_is_undetermined_not_excluded() {
    // 4 channels, block 2 -> two within-block positions {0,2} and {1,3}.
    // Exclude only channel 3 (high=1): position {1,3} loses its partner, so
    // channel 1 can form no reference -> Undetermined, while it is NOT excluded.
    let gains = synth::gains(4, 0.0, 1);
    let scan = synth::flat_scan(2, &gains, 1.0e5, 1); // 2 steps
    let cf = single_step(&scan, EdgeExclusion { low: 0, high: 1 }).unwrap();
    assert_eq!(cf.channels()[3], ChannelFactor::Excluded);
    assert_eq!(cf.channels()[1], ChannelFactor::Undetermined);
    assert!(cf.channels()[0].is_determined());
    assert!(cf.channels()[2].is_determined());
    // The three states are distinguishable, and both non-determined apply as 1.0.
    assert_eq!(cf.channels()[1].multiplier(), 1.0);
    assert!(cf.undetermined()[1]);
    assert!(!cf.excluded()[1]);
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
