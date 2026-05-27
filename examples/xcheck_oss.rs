//! Cross-verification dump for the OSS process.
//!
//! Prints a deterministic, noiseless 8-channel / 8-block reference scan (the
//! canonical Kato & Shigeta 2020 example geometry) and the OSS correction
//! factors the crate computes for each weight. The same input matrix is
//! re-evaluated independently in Wolfram Language to confirm the Rust code
//! matches the paper's eq.(11) `σ_k = sample std of c_OSS_k` definition
//! ([`OssWeight::SampleStd`]) and its alternatives to machine precision.
//!
//! Wolfram `SampleStd` (eq. 11) reference, this exact case (18 digits):
//!   0: 1.019590268391508276   4: 1.057813959519594330
//!   1: 0.925985222505391792   5: 0.839098869525340996
//!   2: 1.118115807949737943   6: 1.238079866158646486
//!   3: 0.959162871302379066   7: 0.969368643187109025
//!
//! Run: `cargo run --example xcheck_oss`

use kato_xrnu::{EdgeExclusion, OssWeight, Scan, optimized_single_step};

fn main() {
    let gains = [1.00, 1.10, 0.90, 1.05, 0.95, 1.20, 0.80, 1.02];
    let intensity = 1000.0;

    // Noiseless flat scatterer: every (measurement, channel) entry is gain*I.
    // 8 measurements (one block shift per step), 8 channels.
    let measurements: Vec<Vec<f64>> = (0..8)
        .map(|_| gains.iter().map(|g| g * intensity).collect())
        .collect();
    let scan = Scan::new(1, measurements).unwrap();

    println!("gains = {gains:?}");
    println!("intensity = {intensity}");
    for (name, w) in [
        ("SampleStd (paper eq.11)", OssWeight::SampleStd),
        ("PoissonPropagated", OssWeight::PoissonPropagated),
        ("OverlapCount", OssWeight::OverlapCount),
    ] {
        let cf = optimized_single_step(&scan, EdgeExclusion::NONE, w).unwrap();
        println!("# OSS {name} factors (channel: value)");
        for (i, c) in cf.channels().iter().enumerate() {
            println!("{i}: {:.18}", c.value().unwrap());
        }
    }
}
