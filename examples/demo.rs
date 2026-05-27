//! Demonstrates the XRNU correction on a synthetic flat scatterer.
//!
//! Run with: `cargo run --example demo`
//!
//! Builds a 1280-channel module with ~1 % per-channel gain dispersion, acquires
//! a shifted reference scan with Poisson noise, derives correction factors with
//! the single-step (SS), optimized single-step (OSS) and multi-step (MS)
//! processes, applies them to an independent sample measurement, and prints the
//! total fractional uncertainty (TFU) and Poisson noise ratio (PNR) before and
//! after correction over the interior channels.

use kato_xrnu::{
    EdgeExclusion, OssWeight, multi_step, optimized_single_step, pnr, single_step, synth, tfu,
};

fn main() {
    let n = 1280;
    let block = 8;
    let intensity = 1.0e6_f64;
    let interior = 16..(n - 16);
    let floor = 100.0 / intensity.sqrt();

    // Detector: ~1 % gain dispersion (trimmed-MYTHEN XRNU level).
    let gains = synth::gains(n, 0.01, 0xABCD);

    // Independent sample measurement we want to correct.
    let sample = synth::flat_sample(&gains, intensity, 0x2222);

    // SS / OSS share one shifted scan; MS uses its hierarchy of scans.
    let ss_scan = synth::flat_scan(block, &gains, intensity, 0x1111);
    let ms_steps = synth::flat_ms_steps(&[80, 40, 20, 10, 5], &gains, intensity, 0x3333);
    let edges = EdgeExclusion::symmetric(16);

    let ss = single_step(&ss_scan, edges).unwrap();
    let oss = optimized_single_step(&ss_scan, edges, OssWeight::PoissonPropagated).unwrap();
    let ms = multi_step(&ms_steps, edges).unwrap();

    let report = |label: &str, data: &[f64]| {
        let slice = &data[interior.clone()];
        println!(
            "  {label:<28} TFU = {:6.3} %   PNR = {:6.2}",
            tfu(slice),
            pnr(slice)
        );
    };

    println!("Synthetic flat scatterer: {n} channels, block {block}, I = {intensity:.0}");
    println!("Poisson floor (single measurement): TFU ~= {floor:.3} %, PNR = 1\n");

    report("raw sample", &sample);
    report("single-step (SS)", &ss.apply(&sample));
    report("optimized single-step (OSS)", &oss.apply(&sample));
    report("multi-step (MS)", &ms.apply(&sample));
}
