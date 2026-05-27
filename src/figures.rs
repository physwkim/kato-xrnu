//! Figures of merit — Kato et al. (2019) eq. (14)–(15).

/// Arithmetic mean of `x`. Returns `0.0` for an empty slice.
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample standard deviation of `x` (divisor `N − 1`).
///
/// Returns `0.0` for fewer than two values.
pub fn sample_std(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(x);
    let ss: f64 = x.iter().map(|&v| (v - m) * (v - m)).sum();
    (ss / (n as f64 - 1.0)).sqrt()
}

/// Total fractional uncertainty (TFU), in percent — eq. (14).
///
/// `TFU = σ_I / Ī × 100`, with `σ_I` the sample standard deviation and `Ī` the
/// mean of the channel intensities. Returns `0.0` if the mean is zero.
pub fn tfu(intensities: &[f64]) -> f64 {
    let m = mean(intensities);
    if m == 0.0 {
        return 0.0;
    }
    sample_std(intensities) / m * 100.0
}

/// Poisson standard deviation for a given mean count, `σ_PN = √Ī`.
pub fn poisson_sigma(mean_counts: f64) -> f64 {
    mean_counts.max(0.0).sqrt()
}

/// Poisson noise ratio (PNR) — eq. (15).
///
/// `PNR = σ_I / σ_PN`, with `σ_PN = √Ī`. A PNR near 1 means the data are at the
/// Poisson floor (XRNU removed). Returns `0.0` if the mean is non-positive.
pub fn pnr(intensities: &[f64]) -> f64 {
    let m = mean(intensities);
    if m <= 0.0 {
        return 0.0;
    }
    sample_std(intensities) / poisson_sigma(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_array_has_zero_spread() {
        let x = [5.0; 8];
        assert_eq!(sample_std(&x), 0.0);
        assert_eq!(tfu(&x), 0.0);
        assert_eq!(pnr(&x), 0.0);
    }

    #[test]
    fn mean_and_std_match_hand_values() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&x) - 5.0).abs() < 1e-12);
        // sample std (N-1) of this classic set is sqrt(32/7).
        assert!((sample_std(&x) - (32.0_f64 / 7.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn pnr_is_one_at_poisson_floor() {
        // std == sqrt(mean) -> PNR == 1.
        let m = 100.0_f64;
        let s = m.sqrt();
        // Build a two-point set with the desired mean and sample std.
        let x = [
            m - s / std::f64::consts::SQRT_2,
            m + s / std::f64::consts::SQRT_2,
        ];
        assert!((mean(&x) - m).abs() < 1e-9);
        assert!((pnr(&x) - 1.0).abs() < 1e-9);
    }
}
