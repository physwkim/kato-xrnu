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

/// The correction outcome for a single channel.
///
/// Modelling the three cases as one sum type keeps them distinguishable: a bare
/// factor of `1.0` previously meant *any* of "excluded", "undetermined" or
/// "genuine unit gain", so a consumer could not tell a corrected channel from an
/// uncorrected one. Here the illegal overlap is unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChannelFactor {
    /// A gain factor was estimated from the reference scan: `c ≈ 1/gain`.
    Determined(f64),
    /// The channel was excluded from the calculation (an edge zone).
    Excluded,
    /// No reference could be formed for this channel — e.g. fewer than two
    /// co-observing channels survived edge exclusion, or the observed intensity
    /// was non-positive. The channel was neither excluded nor corrected.
    Undetermined,
}

impl ChannelFactor {
    /// The multiplier to apply to a raw count: the estimated factor when
    /// [`Determined`](ChannelFactor::Determined), otherwise identity (`1.0`), so
    /// excluded and undetermined channels pass through unchanged.
    #[inline]
    pub fn multiplier(self) -> f64 {
        match self {
            ChannelFactor::Determined(c) => c,
            ChannelFactor::Excluded | ChannelFactor::Undetermined => 1.0,
        }
    }

    /// The estimated factor when [`Determined`](ChannelFactor::Determined), else
    /// `None`.
    #[inline]
    pub fn value(self) -> Option<f64> {
        match self {
            ChannelFactor::Determined(c) => Some(c),
            _ => None,
        }
    }

    /// Whether a factor was estimated for this channel.
    #[inline]
    pub fn is_determined(self) -> bool {
        matches!(self, ChannelFactor::Determined(_))
    }
}

/// Per-channel multiplicative correction factors `c(i) ≈ 1/gain(i)`.
///
/// Multiply a raw sample measurement by these factors to remove the XRNU gain.
/// Each channel is a [`ChannelFactor`], so determined, excluded and undetermined
/// channels are distinguishable rather than all collapsing to `1.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionFactors {
    channels: Vec<ChannelFactor>,
}

impl CorrectionFactors {
    pub(crate) fn new(channels: Vec<ChannelFactor>) -> Self {
        Self { channels }
    }

    /// The per-channel correction outcomes.
    #[inline]
    pub fn channels(&self) -> &[ChannelFactor] {
        &self.channels
    }

    /// Per-channel multipliers (identity for excluded / undetermined channels) —
    /// the values [`apply`](Self::apply) uses. Allocates; prefer
    /// [`channels`](Self::channels) to inspect the outcome.
    pub fn multipliers(&self) -> Vec<f64> {
        self.channels.iter().map(|c| c.multiplier()).collect()
    }

    /// Per-channel flag: `true` if the channel was excluded from the calculation.
    pub fn excluded(&self) -> Vec<bool> {
        self.channels
            .iter()
            .map(|c| matches!(c, ChannelFactor::Excluded))
            .collect()
    }

    /// Per-channel flag: `true` if no factor could be determined for the channel
    /// (and it was not excluded).
    pub fn undetermined(&self) -> Vec<bool> {
        self.channels
            .iter()
            .map(|c| matches!(c, ChannelFactor::Undetermined))
            .collect()
    }

    /// Number of channels.
    #[inline]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Whether there are no channels.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Apply the correction to a raw measurement, returning the corrected copy.
    /// Excluded and undetermined channels pass through unchanged.
    ///
    /// # Panics
    /// Panics if `raw.len()` differs from the number of channels.
    pub fn apply(&self, raw: &[f64]) -> Vec<f64> {
        assert_eq!(
            raw.len(),
            self.channels.len(),
            "raw length does not match channel count"
        );
        raw.iter()
            .zip(&self.channels)
            .map(|(&y, c)| y * c.multiplier())
            .collect()
    }

    /// Apply the correction in place. Excluded and undetermined channels pass
    /// through unchanged.
    ///
    /// # Panics
    /// Panics if `raw.len()` differs from the number of channels.
    pub fn apply_in_place(&self, raw: &mut [f64]) {
        assert_eq!(
            raw.len(),
            self.channels.len(),
            "raw length does not match channel count"
        );
        for (y, c) in raw.iter_mut().zip(&self.channels) {
            *y *= c.multiplier();
        }
    }
}
