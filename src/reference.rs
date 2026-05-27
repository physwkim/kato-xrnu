//! Acquisition model ([`Scan`]), edge handling, and the correction-factor map.

use crate::error::KatoError;

/// Reference-scan data: the intensities recorded as the detector is shifted.
///
/// `measurements[p][c]` is the (divided-accumulation–summed) intensity recorded
/// at shift step `p` by channel `c`. Every step shifts the detector by
/// [`shift_per_step`](Scan::shift_per_step) channels, so channel `c` at step `p`
/// observes lab position `c + shift_per_step·p` (in channel units).
#[derive(Debug, Clone)]
pub struct Scan {
    n_channels: usize,
    shift_per_step: usize,
    measurements: Vec<Vec<f64>>,
}

impl Scan {
    /// Build a scan from per-step intensity rows.
    ///
    /// Every row must have the same length (the channel count) and there must be
    /// at least one row; `shift_per_step` must be non-zero.
    pub fn new(shift_per_step: usize, measurements: Vec<Vec<f64>>) -> Result<Self, KatoError> {
        if shift_per_step == 0 {
            return Err(KatoError::ZeroShift);
        }
        let n_channels = measurements.first().ok_or(KatoError::EmptyScan)?.len();
        for (step, row) in measurements.iter().enumerate() {
            if row.len() != n_channels {
                return Err(KatoError::RaggedMeasurement {
                    step,
                    found: row.len(),
                    expected: n_channels,
                });
            }
        }
        Ok(Self {
            n_channels,
            shift_per_step,
            measurements,
        })
    }

    /// Number of detector channels.
    #[inline]
    pub fn n_channels(&self) -> usize {
        self.n_channels
    }

    /// Number of shift steps (measurement rows).
    #[inline]
    pub fn n_steps(&self) -> usize {
        self.measurements.len()
    }

    /// Channels the detector is shifted by between consecutive steps.
    #[inline]
    pub fn shift_per_step(&self) -> usize {
        self.shift_per_step
    }

    /// Intensity recorded at step `p`, channel `c`.
    #[inline]
    pub fn at(&self, step: usize, channel: usize) -> f64 {
        self.measurements[step][channel]
    }
}

/// Channels to drop from the calculation at each end of the detector.
///
/// In Kato et al. (2019), 32 channels at both ends (channels 1–16 and 1265–1280
/// of each 1280-channel module) were excluded because their intensities were
/// unreliable. Excluded channels contribute to no reference estimate and receive
/// an identity factor of `1.0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EdgeExclusion {
    /// Channels excluded at the low-index end.
    pub low: usize,
    /// Channels excluded at the high-index end.
    pub high: usize,
}

impl EdgeExclusion {
    /// No channels excluded.
    pub const NONE: EdgeExclusion = EdgeExclusion { low: 0, high: 0 };

    /// Exclude the same number of channels at both ends.
    pub fn symmetric(n: usize) -> Self {
        EdgeExclusion { low: n, high: n }
    }

    /// Whether `channel` (of `n_channels`) falls in an excluded edge zone.
    #[inline]
    pub fn is_excluded(&self, channel: usize, n_channels: usize) -> bool {
        channel < self.low || channel >= n_channels.saturating_sub(self.high)
    }

    pub(crate) fn validate(&self, n_channels: usize) -> Result<(), KatoError> {
        if self.low + self.high >= n_channels {
            return Err(KatoError::EdgeExclusionTooLarge {
                n_channels,
                low: self.low,
                high: self.high,
            });
        }
        Ok(())
    }
}

/// Per-channel multiplicative correction factors `c(i) ≈ 1/gain(i)`.
///
/// Multiply a raw sample measurement by these factors to remove the XRNU gain.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionFactors {
    factors: Vec<f64>,
    excluded: Vec<bool>,
}

impl CorrectionFactors {
    pub(crate) fn new(factors: Vec<f64>, excluded: Vec<bool>) -> Self {
        debug_assert_eq!(factors.len(), excluded.len());
        Self { factors, excluded }
    }

    /// The per-channel factors. Excluded / undetermined channels hold `1.0`.
    #[inline]
    pub fn factors(&self) -> &[f64] {
        &self.factors
    }

    /// Per-channel flag: `true` if the channel was excluded from the calculation.
    #[inline]
    pub fn excluded(&self) -> &[bool] {
        &self.excluded
    }

    /// Number of channels.
    #[inline]
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// Whether there are no channels.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// Apply the correction to a raw measurement, returning the corrected copy.
    ///
    /// # Panics
    /// Panics if `raw.len()` differs from the number of channels.
    pub fn apply(&self, raw: &[f64]) -> Vec<f64> {
        assert_eq!(
            raw.len(),
            self.factors.len(),
            "raw length does not match channel count"
        );
        raw.iter()
            .zip(&self.factors)
            .map(|(&y, &c)| y * c)
            .collect()
    }

    /// Apply the correction in place.
    ///
    /// # Panics
    /// Panics if `raw.len()` differs from the number of channels.
    pub fn apply_in_place(&self, raw: &mut [f64]) {
        assert_eq!(
            raw.len(),
            self.factors.len(),
            "raw length does not match channel count"
        );
        for (y, &c) in raw.iter_mut().zip(&self.factors) {
            *y *= c;
        }
    }
}
