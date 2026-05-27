# kato-xrnu

Statistical correction of **X-ray response non-uniformity (XRNU)** in
photon-counting microstrip detectors (e.g. DECTRIS MYTHEN), in pure Rust with
zero runtime dependencies.

It implements the data-processing methods of:

- **Kato, Tanaka, Yamauchi, Ohara & Hatsui (2019)**, *J. Synchrotron Rad.* **26**,
  762–773 — the single-step (`SS`) and multi-step (`MS`) processes and the
  figures of merit TFU and PNR.
- **Kato & Shigeta (2020)**, *J. Synchrotron Rad.* **27**, 1172–1179 — the
  optimized single-step (`OSS`) process.

## Why

XRNU is a per-channel multiplicative gain error (threshold dispersion as a
function of threshold energy and temperature). Conventional flat-field
correction needs uniform illumination, which cannot be met at the 0.1 % level.

These methods exploit a different fact: the scattered intensity at a fixed angle
2θ from a reference scatterer is constant within Poisson noise. By **shifting the
detector** so that *different channels* observe the *same* 2θ, the per-channel
gain is estimated without assuming uniform illumination. The output is a
per-channel factor `c(i) ≈ 1/gain(i)`; multiplying a measurement by `c` removes
the gain error.

## Processes

| Process | Reference points per block-position | Data efficiency | Entry point |
|---------|-------------------------------------|-----------------|-------------|
| SS  | 1 (full overlap only)                         | low    | `single_step` |
| MS  | hierarchical, coarse → fine                   | medium | `multi_step` |
| OSS | all overlap levels, inverse-variance weighted | high   | `optimized_single_step` |

- **SS** shifts by one block per step for `n_channels / block` steps, so all
  blocks coincide at one fully-overlapped 2θ per within-block position. One
  shared reference per position → with no noise it flattens a flat field
  *exactly*.
- **MS** is a hierarchy of steps with block sizes that halve each time
  (e.g. `[80, 40, 20, 10, 5]`). The first step is a full overlap; each later step
  compares the two halves of every parent block, the readings first corrected by
  the product of the preceding steps' factors. Final factor = `c1·c2·…·cK`.
- **OSS** reuses the SS acquisition but blends *every* overlap level
  (2, 3, …, n, …, 3, 2 channels) by inverse-variance weighting, recovering the
  most information from one scan. Because partial-overlap references blend
  different gain subsets, OSS does not flatten a flat field exactly even without
  noise, but under real Poisson noise it reaches the lowest spread of the three.

## Figures of merit

- **TFU** (total fractional uncertainty) `= σ_I / Ī × 100` [%] — the spread of
  channel intensities; ~1 % for raw trimmed MYTHEN, target ~0.1 %.
- **PNR** (Poisson noise ratio) `= σ_I / √Ī` — `≈ 1` means the data sit at the
  Poisson floor (XRNU removed).

## Usage

```rust
use kato_xrnu::{optimized_single_step, single_step, synth, tfu, EdgeExclusion, OssWeight};

let gains = synth::gains(1280, 0.01, 0xABCD);     // ~1 % dispersion
let interior = 16..1264;

let scan = synth::flat_scan(8, &gains, 1.0e6, 0x1111);   // shifted reference scan
let sample = synth::flat_sample(&gains, 1.0e6, 0x2222);  // measurement to correct

let oss = optimized_single_step(&scan, EdgeExclusion::symmetric(16), OssWeight::default()).unwrap();
let corrected = oss.apply(&sample);

println!("raw TFU       = {:.3} %", tfu(&sample[interior.clone()]));
println!("corrected TFU = {:.3} %", tfu(&corrected[interior]));
```

Run the worked demonstration:

```sh
cargo run --example demo
```

which reproduces the central result (1280 channels, block 8, 10⁶ counts):

```
  raw sample                   TFU =  0.976 %   PNR =   9.76
  single-step (SS)             TFU =  0.159 %   PNR =   1.59
  optimized single-step (OSS)  TFU =  0.126 %   PNR =   1.26
  multi-step (MS)              TFU =  0.156 %   PNR =   1.56
```

## Verification

The OSS weighting defaults to Kato & Shigeta (2020) eq. (11) — `σ_k` is the
sample standard deviation of the local factors over each overlap level
(`OssWeight::SampleStd`). It is cross-verified against an independent Wolfram
Language implementation of eq. (10)–(11):

```sh
cargo run --example xcheck_oss                     # Rust factors, all three weights
wolframscript -f verification/oss_eq11_check.wl    # the independent Wolfram side
```

On the deterministic 8-channel case the two agree to ~1e-15; those reference
values are pinned in `tests/processes.rs`.

## Acquisition model

A `Scan` holds the per-step intensity rows recorded as the detector is shifted by
`shift_per_step` channels. Channel `c` at step `p` observes lab position
`c + shift_per_step·p` (channel units). Supply intensities already summed over
the divided-accumulation sub-frames. Edge channels (unreliable at the module
ends) are dropped via `EdgeExclusion`; excluded channels receive an identity
factor of `1.0`.

The `synth` module generates deterministic flat-scatterer scans with known gains
and Poisson noise, used by the tests, the example, and for validation.

## Scope

This crate corrects **XRNU only** — the per-channel gain. It does **not** perform
angular-offset calibration, monitor normalisation, inter-module efficiency, or
2θ rebinning (the Wright et al. 2003 channel/module combination), nor the
axial-divergence correction (which needs 2D axial pixel positions a 1D strip
detector does not provide; use a Finger–Cox–Jephcoat peak-shape model in
Rietveld refinement instead).

## License

MIT
