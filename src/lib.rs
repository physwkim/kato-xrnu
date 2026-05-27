//! Statistical correction of **X-ray response non-uniformity (XRNU)** in
//! photon-counting microstrip detectors (e.g. DECTRIS MYTHEN), implementing the
//! data-processing methods of:
//!
//! - **Kato, Tanaka, Yamauchi, Ohara & Hatsui (2019)**, *J. Synchrotron Rad.*
//!   **26**, 762–773 — the single-step (`SS`) and multi-step (`MS`) processes,
//!   plus the figures of merit TFU and PNR.
//! - **Kato & Shigeta (2020)**, *J. Synchrotron Rad.* **27**, 1172–1179 — the
//!   optimized single-step (`OSS`) process.
//!
//! # The idea
//!
//! XRNU is a per-channel multiplicative gain error. The scattering intensity at
//! a fixed angle 2θ from a reference scatterer is expected to be constant within
//! Poisson noise. By shifting the detector and letting *different channels*
//! observe the *same* 2θ, the per-channel gain can be estimated without assuming
//! uniform illumination (the assumption that conventional flat-field correction
//! requires and cannot meet at the 0.1 % level).
//!
//! All three processes produce a per-channel [`CorrectionFactors`] map `c(i)`
//! such that `c(i) ≈ 1/gain(i)`; multiplying a sample measurement by `c` removes
//! the gain error.
//!
//! # Processes
//!
//! | Process | Reference points per block-position | Data use | Entry point |
//! |---------|-------------------------------------|----------|-------------|
//! | SS  | 1 (full overlap only)               | low      | [`single_step`] |
//! | MS  | hierarchical, coarse→fine           | medium   | [`multi_step`]  |
//! | OSS | all overlap levels, inverse-variance weighted | high | [`optimized_single_step`] |
//!
//! # Acquisition model ([`Scan`])
//!
//! A [`Scan`] holds the measurements recorded as the detector is shifted by a
//! fixed number of channels per step. Channel `c` at step `p` observes lab
//! position `c + shift·p` (in channel units). Channels that map to the same lab
//! position co-observe one 2θ; their spread (beyond Poisson) is the XRNU.
//!
//! # Example
//!
//! ```
//! use kato_xrnu::{optimized_single_step, single_step, synth, tfu, EdgeExclusion, OssWeight};
//!
//! // A 1280-channel module with ~1 % per-channel gain dispersion.
//! let gains = synth::gains(1280, 0.01, 0xABCD);
//! let interior = 16..1264;
//!
//! // A shifted reference scan (block size 8) and an independent sample, both
//! // of a flat scatterer with Poisson noise at 10^6 counts.
//! let scan = synth::flat_scan(8, &gains, 1.0e6, 0x1111);
//! let sample = synth::flat_sample(&gains, 1.0e6, 0x2222);
//!
//! let ss = single_step(&scan, EdgeExclusion::symmetric(16)).unwrap();
//! let oss = optimized_single_step(&scan, EdgeExclusion::symmetric(16), OssWeight::default()).unwrap();
//!
//! let raw = tfu(&sample[interior.clone()]);
//! let ss_corr = tfu(&ss.apply(&sample)[interior.clone()]);
//! let oss_corr = tfu(&oss.apply(&sample)[interior]);
//!
//! // Correction drives the spread from ~1 % toward the ~0.1 % Poisson floor,
//! // and OSS is no worse than SS.
//! assert!(ss_corr < raw * 0.4);
//! assert!(oss_corr <= ss_corr * 1.15);
//! ```

mod correction;
mod error;
mod figures;
mod oss;
mod reference;
pub mod synth;

pub use correction::{MsStep, multi_step, single_step};
pub use error::KatoError;
pub use figures::{mean, pnr, poisson_sigma, sample_std, tfu};
pub use oss::{OssWeight, optimized_single_step};
pub use reference::{ChannelFactor, CorrectionFactors, EdgeExclusion, Scan};
