//! Error type for the crate.

use std::fmt;

/// Errors returned when building a [`Scan`](crate::Scan) or running a correction
/// process with inconsistent inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KatoError {
    /// A scan was constructed with no measurements.
    EmptyScan,
    /// A measurement row has a different length than the others.
    RaggedMeasurement {
        /// Index of the offending measurement (step).
        step: usize,
        /// Length found.
        found: usize,
        /// Length expected (length of the first row).
        expected: usize,
    },
    /// `shift_per_step` was zero.
    ZeroShift,
    /// The channel count is not divisible by the block size, so the channels do
    /// not partition into whole blocks.
    ChannelsNotDivisible {
        /// Number of channels.
        n_channels: usize,
        /// Block size (channels per block).
        block_size: usize,
    },
    /// The number of measurements (steps) in the scan does not match what the
    /// process requires for the given block size.
    WrongMeasurementCount {
        /// Required number of steps.
        expected: usize,
        /// Number of steps found in the scan.
        found: usize,
    },
    /// `multi_step` was called with no steps.
    NoSteps,
    /// In `multi_step`, consecutive block sizes must halve (`prev == 2·cur`).
    StepBlockNotHalving {
        /// Step index (>= 1) whose block size is inconsistent.
        step: usize,
        /// Block size of the previous step.
        prev: usize,
        /// Block size of this step.
        cur: usize,
    },
    /// The combined edge exclusion removes every channel.
    EdgeExclusionTooLarge {
        /// Number of channels.
        n_channels: usize,
        /// Channels excluded at the low end.
        low: usize,
        /// Channels excluded at the high end.
        high: usize,
    },
}

impl fmt::Display for KatoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KatoError::EmptyScan => write!(f, "scan has no measurements"),
            KatoError::RaggedMeasurement {
                step,
                found,
                expected,
            } => write!(
                f,
                "measurement {step} has length {found}, expected {expected}"
            ),
            KatoError::ZeroShift => write!(f, "shift_per_step must be non-zero"),
            KatoError::ChannelsNotDivisible {
                n_channels,
                block_size,
            } => write!(
                f,
                "channel count {n_channels} is not divisible by block size {block_size}"
            ),
            KatoError::WrongMeasurementCount { expected, found } => write!(
                f,
                "wrong number of measurements: expected {expected}, found {found}"
            ),
            KatoError::NoSteps => write!(f, "multi_step requires at least one step"),
            KatoError::StepBlockNotHalving { step, prev, cur } => write!(
                f,
                "step {step}: block size {cur} must be half of previous {prev}"
            ),
            KatoError::EdgeExclusionTooLarge {
                n_channels,
                low,
                high,
            } => write!(
                f,
                "edge exclusion low={low} + high={high} removes all {n_channels} channels"
            ),
        }
    }
}

impl std::error::Error for KatoError {}
