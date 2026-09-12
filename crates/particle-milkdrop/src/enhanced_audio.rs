//! Enhanced audio rows for BeatDrop-style MilkDrop shader helpers.
//!
//! This module intentionally sits *beside* OjoDrop's legacy audio rails.  It
//! consumes a host-provided magnitude spectrum and stereo PCM, but never mutates
//! either input or changes the values used by the classic waveform/EEL paths.
//! BeatDrop computes its 1024-bin FFT from a larger capture window; OjoDrop's
//! current host contract supplies a potentially mono/equalized 512-bin row. That
//! route is explicitly labelled as host magnitude input. When independent spectra
//! are unavailable, this module can instead run a bounded real FFT over available
//! stereo PCM and reports its actual capture-window resolution. Neither route
//! claims bit-identical BeatDrop analysis.

/// Width of the spectrum texture. Matches BeatDrop's actual shader allocation.
pub const ENHANCED_FFT_BINS: usize = 1024;
/// Width of the stereo waveform texture.
pub const ENHANCED_WAVE_SAMPLES: usize = 512;
/// Transform size used by the bounded PCM FFT path. Shorter host PCM is zero
/// padded for evaluation only; [`EnhancedSpectrumSource`] records its real size.
pub const ENHANCED_FFT_WINDOW_SAMPLES: usize = ENHANCED_FFT_BINS * 2;
/// Legacy fallback only when a preanalyzed/recorded frame has no out-of-band
/// sample-rate metadata. Do not add a rate field to `AudioInput`: its binary
/// layout is persisted by preanalysis and recording formats.
pub const LEGACY_ASSUMED_SAMPLE_RATE_HZ: f32 = 44_100.0;
const RMS_FLOOR: f32 = 1.0e-5;
const VISIBLE_FLOOR: f32 = 5.0e-8;

/// Per-preset controls for the enhanced FFT follower.
///
/// `Default` matches BeatDrop's state defaults. These values are runtime
/// controls; parser/persistence code should omit them when they are equal to
/// this default, preserving the legacy preset representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnhancedAudioConfig {
    /// Rise follower, normalized 0..1.
    pub fft_attack: f32,
    /// Fall follower and peak-hold duration control, normalized 0..1.
    pub fft_decay: f32,
    /// Post-AGC scale. Kept separate from the preset fields because BeatDrop
    /// treats it as a player preference rather than preset data.
    pub fft_scaling: f32,
    /// Post-AGC noise threshold. Also a player preference, not preset data.
    pub fft_noise_floor: f32,
}

impl Default for EnhancedAudioConfig {
    fn default() -> Self {
        Self {
            fft_attack: 0.5,
            fft_decay: 0.7,
            fft_scaling: 0.175,
            fft_noise_floor: 0.03,
        }
    }
}

impl EnhancedAudioConfig {
    /// Return a finite, bounded configuration suitable for a frame update.
    pub fn sanitized(self) -> Self {
        Self {
            fft_attack: finite_clamp(self.fft_attack, 0.0, 1.0, Self::default().fft_attack),
            fft_decay: finite_clamp(self.fft_decay, 0.0, 1.0, Self::default().fft_decay),
            fft_scaling: finite_clamp(self.fft_scaling, 0.0, 64.0, Self::default().fft_scaling),
            fft_noise_floor: finite_clamp(
                self.fft_noise_floor,
                0.0,
                64.0,
                Self::default().fft_noise_floor,
            ),
        }
    }
}

/// CPU rows and metadata ready for the renderer's two dynamic audio textures.
///
/// FFT rows contain BeatDrop's directly uploaded smoothed/peak values. The shader
/// helper applies `sqrt` when it samples them; do not square these values to
/// cancel that operation. Wave rows are signed PCM.
#[derive(Clone, Debug, PartialEq)]
pub struct EnhancedAudioFrame {
    /// Row 0: smoothed FFT; row 1: peak-hold FFT.
    pub fft_rows: Vec<f32>,
    /// Independent raw magnitude rows retained for BeatDrop's base mode 8.
    ///
    /// These are deliberately distinct from [`Self::fft_rows`], whose rows are
    /// the AGC-smoothed and peak-hold helper values. A mode-8 caller needs the
    /// actual left/right source magnitudes, not two temporal views of one
    /// aggregate follower. Empty means the source contract was mono.
    spectrum_left: Vec<f32>,
    spectrum_right: Vec<f32>,
    /// Row 0: left PCM; row 1: right PCM.
    pub waveform_rows: Vec<f32>,
    /// Sanitized source rate used by `get_fft_hz` / `get_fft_peak_hz`. This is
    /// either renderer metadata from the live capture or the documented legacy
    /// assumption for preanalyzed/recorded frames with no rate field.
    pub sample_rate_hz: f32,
    /// Truthful provenance for the row consumed by the enhanced follower.
    pub spectrum_source: EnhancedSpectrumSource,
}

/// How the enhanced FFT input was acquired. A fixed-width GPU row never means
/// that the host supplied the same number of independent spectral bins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnhancedSpectrumSource {
    /// Existing host magnitudes. These may be mono/equalized; their frequency
    /// analysis contract is owned by the host, not reconstructed here.
    HostMagnitude { bins: usize },
    /// A bounded in-process real FFT over `input_samples` real PCM values per
    /// channel. `transform_bins` is the zero-padded FFT output count.
    PcmFft {
        input_samples: usize,
        transform_bins: usize,
    },
}

impl EnhancedAudioFrame {
    pub fn new() -> Self {
        Self {
            fft_rows: vec![0.0; ENHANCED_FFT_BINS * 2],
            spectrum_left: Vec::with_capacity(ENHANCED_FFT_BINS),
            spectrum_right: Vec::with_capacity(ENHANCED_FFT_BINS),
            waveform_rows: vec![0.0; ENHANCED_WAVE_SAMPLES * 2],
            sample_rate_hz: LEGACY_ASSUMED_SAMPLE_RATE_HZ,
            spectrum_source: EnhancedSpectrumSource::HostMagnitude { bins: 0 },
        }
    }

    /// Nyquist frequency uploaded to the shader UBO for Hz-addressed helpers.
    pub fn nyquist_hz(&self) -> f32 {
        (self.sample_rate_hz * 0.5).max(1.0)
    }

    /// Pack the FFT rows into caller-owned storage for a filterable
    /// `Rgba16Float` texture. Reuse `destination` across frames to avoid a
    /// transient upload allocation.
    pub fn write_fft_rgba16f(&self, destination: &mut Vec<u16>) {
        pack_rows_rgba16f(&self.fft_rows, destination)
    }

    /// Pack signed waveform rows into caller-owned storage for a filterable
    /// `Rgba16Float` texture. Reuse `destination` across frames to avoid a
    /// transient upload allocation.
    pub fn write_waveform_rgba16f(&self, destination: &mut Vec<u16>) {
        pack_rows_rgba16f(&self.waveform_rows, destination)
    }

    /// Actual independent L/R source magnitudes for extended waveform mode 8.
    ///
    /// This is `None` after the mono-source convenience update, even though the
    /// enhanced helper follower can consume that source. The mode-8 renderer
    /// must never turn legacy mono/equalized data into fabricated stereo.
    pub fn spectrum_rows(&self) -> Option<(&[f32], &[f32])> {
        (!self.spectrum_left.is_empty() && !self.spectrum_right.is_empty())
            .then_some((&self.spectrum_left, &self.spectrum_right))
    }
}

impl Default for EnhancedAudioFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateful, elapsed-time-based AGC/follower/peak model.
#[derive(Clone, Debug)]
pub struct EnhancedAudioProcessor {
    smoothed: Vec<f32>,
    peaks: Vec<f32>,
    peak_hold_seconds: Vec<f32>,
    smoothed_rms: f32,
    fft_left: Vec<Complex>,
    fft_right: Vec<Complex>,
    pcm_fft_left: Vec<f32>,
    pcm_fft_right: Vec<f32>,
    frame: EnhancedAudioFrame,
}

impl Default for EnhancedAudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedAudioProcessor {
    pub fn new() -> Self {
        Self {
            smoothed: vec![0.0; ENHANCED_FFT_BINS],
            peaks: vec![0.0; ENHANCED_FFT_BINS],
            peak_hold_seconds: vec![0.0; ENHANCED_FFT_BINS],
            // BeatDrop seeds this follower at 0.1 to avoid a silence-start spike.
            smoothed_rms: 0.1,
            fft_left: vec![Complex::default(); ENHANCED_FFT_WINDOW_SAMPLES],
            fft_right: vec![Complex::default(); ENHANCED_FFT_WINDOW_SAMPLES],
            pcm_fft_left: vec![0.0; ENHANCED_FFT_BINS],
            pcm_fft_right: vec![0.0; ENHANCED_FFT_BINS],
            frame: EnhancedAudioFrame::new(),
        }
    }

    /// Clear temporal state on a renderer/preset lifecycle reset.
    pub fn reset(&mut self) {
        self.smoothed.fill(0.0);
        self.peaks.fill(0.0);
        self.peak_hold_seconds.fill(0.0);
        self.smoothed_rms = 0.1;
        self.frame.fft_rows.fill(0.0);
        self.frame.spectrum_left.clear();
        self.frame.spectrum_right.clear();
        self.frame.waveform_rows.fill(0.0);
        self.frame.spectrum_source = EnhancedSpectrumSource::HostMagnitude { bins: 0 };
    }

    /// Update textures from an existing magnitude spectrum and stereo PCM.
    ///
    /// This mono-source convenience method preserves the current OjoDrop input
    /// contract by feeding the same host magnitude row to both FFT channels. Use
    /// [`Self::update_stereo_spectrum`] when a host can provide independent,
    /// un-equalized channel spectra; that is the closest input shape to BeatDrop's
    /// own analysis path.
    pub fn update(
        &mut self,
        spectrum: &[f32],
        waveform_left: &[f32],
        waveform_right: &[f32],
        sample_rate_hz: f32,
        elapsed_seconds: f32,
        config: EnhancedAudioConfig,
    ) -> &EnhancedAudioFrame {
        self.update_stereo_spectrum(
            spectrum,
            spectrum,
            waveform_left,
            waveform_right,
            sample_rate_hz,
            elapsed_seconds,
            config,
        );
        // `update` preserves the legacy mono input contract for helper use, but
        // it is not a valid stereo source for extended waveform mode 8.
        self.frame.spectrum_left.clear();
        self.frame.spectrum_right.clear();
        &self.frame
    }

    /// Update textures from independent left/right FFT magnitudes and stereo PCM.
    ///
    /// `elapsed_seconds` is wall-clock frame delta, not a frame count. Invalid,
    /// negative, or excessively stalled values are made safe and bounded before
    /// they reach exponential followers. Empty/silent input deterministically
    /// decays state and yields finite rows.
    pub fn update_stereo_spectrum(
        &mut self,
        spectrum_left: &[f32],
        spectrum_right: &[f32],
        waveform_left: &[f32],
        waveform_right: &[f32],
        sample_rate_hz: f32,
        elapsed_seconds: f32,
        config: EnhancedAudioConfig,
    ) -> &EnhancedAudioFrame {
        self.frame.spectrum_source = EnhancedSpectrumSource::HostMagnitude {
            bins: spectrum_left.len().min(spectrum_right.len()),
        };
        copy_magnitude_row(spectrum_left, &mut self.frame.spectrum_left);
        copy_magnitude_row(spectrum_right, &mut self.frame.spectrum_right);
        self.update_from_spectrum_rows(
            spectrum_left,
            spectrum_right,
            waveform_left,
            waveform_right,
            sample_rate_hz,
            elapsed_seconds,
            config,
        )
    }

    /// Compute a real stereo FFT from the PCM the renderer already has.
    ///
    /// This is the truthful fallback when no independent L/R spectrum exists.
    /// It reads at most 2,048 samples/channel, applies a Hann window over the
    /// actual captured length, and zero pads the transform. The output texture is
    /// still 1,024 samples wide for helper compatibility, but callers can inspect
    /// `frame.spectrum_source` to see the non-invented source window length.
    pub fn update_from_pcm(
        &mut self,
        waveform_left: &[f32],
        waveform_right: &[f32],
        sample_rate_hz: f32,
        elapsed_seconds: f32,
        config: EnhancedAudioConfig,
    ) -> &EnhancedAudioFrame {
        let input_samples = waveform_left
            .len()
            .min(waveform_right.len())
            .min(ENHANCED_FFT_WINDOW_SAMPLES);
        fft_magnitudes_from_pcm(
            waveform_left,
            input_samples,
            &mut self.fft_left,
            &mut self.pcm_fft_left,
        );
        fft_magnitudes_from_pcm(
            waveform_right,
            input_samples,
            &mut self.fft_right,
            &mut self.pcm_fft_right,
        );
        copy_magnitude_row(&self.pcm_fft_left, &mut self.frame.spectrum_left);
        copy_magnitude_row(&self.pcm_fft_right, &mut self.frame.spectrum_right);
        self.frame.spectrum_source = EnhancedSpectrumSource::PcmFft {
            input_samples,
            transform_bins: ENHANCED_FFT_BINS,
        };
        // Temporarily move the reusable rows out to satisfy Rust's aliasing
        // rules while the follower mutates its state, then restore their capacity.
        let pcm_fft_left = std::mem::take(&mut self.pcm_fft_left);
        let pcm_fft_right = std::mem::take(&mut self.pcm_fft_right);
        self.update_from_spectrum_rows(
            &pcm_fft_left,
            &pcm_fft_right,
            waveform_left,
            waveform_right,
            sample_rate_hz,
            elapsed_seconds,
            config,
        );
        self.pcm_fft_left = pcm_fft_left;
        self.pcm_fft_right = pcm_fft_right;
        &self.frame
    }

    #[allow(clippy::too_many_arguments)]
    fn update_from_spectrum_rows(
        &mut self,
        spectrum_left: &[f32],
        spectrum_right: &[f32],
        waveform_left: &[f32],
        waveform_right: &[f32],
        sample_rate_hz: f32,
        elapsed_seconds: f32,
        config: EnhancedAudioConfig,
    ) -> &EnhancedAudioFrame {
        let config = config.sanitized();
        let dt = finite_clamp(elapsed_seconds, 0.0, 1.0, 0.0);
        self.frame.sample_rate_hz = finite_clamp(
            sample_rate_hz,
            2.0,
            384_000.0,
            LEGACY_ASSUMED_SAMPLE_RATE_HZ,
        );

        let mut input_rms_sq = 0.0;
        for bin in 0..ENHANCED_FFT_BINS {
            let magnitude = stereo_magnitude(spectrum_left, spectrum_right, bin);
            input_rms_sq += magnitude * magnitude;
        }
        let input_rms = (input_rms_sq / ENHANCED_FFT_BINS as f32)
            .sqrt()
            .max(RMS_FLOOR);
        let rms_tau = if input_rms > self.smoothed_rms {
            2.0
        } else {
            5.0
        };
        let rms_alpha = 1.0 - (-dt / rms_tau).exp();
        self.smoothed_rms =
            (self.smoothed_rms + (input_rms - self.smoothed_rms) * rms_alpha).max(RMS_FLOOR);

        // BeatDrop's 60fps-reference coefficients expressed directly in elapsed
        // time. Its decay first squares (1 - FFTDecay), preserving that response.
        let attack_alpha = one_minus_power(config.fft_attack, dt * 60.0);
        let decay_base = (1.0 - config.fft_decay).powi(2);
        let decay_alpha = one_minus_power(decay_base, dt * 60.0);
        let hold_duration = 0.25 + config.fft_decay * 0.75;

        for bin in 0..ENHANCED_FFT_BINS {
            let raw = stereo_magnitude(spectrum_left, spectrum_right, bin);
            let bass_reduction = if bin < 24 {
                let t = bin as f32 / 24.0;
                0.15 + 0.85 * t * t
            } else {
                1.0
            };
            let target = ((raw / self.smoothed_rms) * config.fft_scaling - config.fft_noise_floor)
                .max(0.0)
                * bass_reduction;
            let follower_alpha = if target > self.smoothed[bin] {
                attack_alpha
            } else {
                decay_alpha
            };
            let smoothed = self.smoothed[bin] + (target - self.smoothed[bin]) * follower_alpha;
            self.smoothed[bin] = if smoothed.is_finite() && smoothed >= VISIBLE_FLOOR {
                smoothed
            } else {
                0.0
            };

            if self.smoothed[bin] >= self.peaks[bin] {
                self.peaks[bin] = self.smoothed[bin];
                self.peak_hold_seconds[bin] = hold_duration;
            } else {
                self.peak_hold_seconds[bin] = (self.peak_hold_seconds[bin] - dt).max(0.0);
                if self.peak_hold_seconds[bin] == 0.0 {
                    self.peaks[bin] *= (-dt / 0.3).exp();
                }
            }
            if !self.peaks[bin].is_finite() || self.peaks[bin] < VISIBLE_FLOOR {
                self.peaks[bin] = 0.0;
            }

            // BeatDrop uploads these direct follower values; get_fft() and
            // get_fft_peak() intentionally take their square root in shader.
            self.frame.fft_rows[bin] = self.smoothed[bin];
            self.frame.fft_rows[ENHANCED_FFT_BINS + bin] = self.peaks[bin];
        }

        for sample in 0..ENHANCED_WAVE_SAMPLES {
            self.frame.waveform_rows[sample] =
                sample_resampled_signed(waveform_left, sample, ENHANCED_WAVE_SAMPLES);
            self.frame.waveform_rows[ENHANCED_WAVE_SAMPLES + sample] =
                sample_resampled_signed(waveform_right, sample, ENHANCED_WAVE_SAMPLES);
        }
        &self.frame
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f32,
    im: f32,
}

/// Fill `magnitudes` with the first half of a radix-2 FFT over actual PCM.
/// `work` is fixed at 2,048 points, so this work and memory remain bounded.
fn fft_magnitudes_from_pcm(
    pcm: &[f32],
    input_samples: usize,
    work: &mut [Complex],
    magnitudes: &mut Vec<f32>,
) {
    debug_assert_eq!(work.len(), ENHANCED_FFT_WINDOW_SAMPLES);
    let n = input_samples.min(pcm.len()).min(work.len());
    for (index, value) in work.iter_mut().enumerate() {
        let sample = if index < n {
            finite_clamp(pcm[index], -1.0, 1.0, 0.0)
        } else {
            0.0
        };
        // A Hann window over the actual capture prevents a zero-padding boundary
        // impulse. It does not manufacture samples beyond the source window.
        let window = if n > 1 && index < n {
            0.5 - 0.5 * ((2.0 * std::f32::consts::PI * index as f32) / (n - 1) as f32).cos()
        } else {
            1.0
        };
        *value = Complex {
            re: sample * window,
            im: 0.0,
        };
    }
    fft_in_place(work);
    magnitudes.clear();
    magnitudes.reserve(ENHANCED_FFT_BINS.saturating_sub(magnitudes.capacity()));
    let normalization = (n.max(1) as f32 * 0.5).max(1.0);
    for value in work.iter().take(ENHANCED_FFT_BINS) {
        let magnitude = (value.re * value.re + value.im * value.im).sqrt() / normalization;
        magnitudes.push(if magnitude.is_finite() {
            magnitude
        } else {
            0.0
        });
    }
}

fn fft_in_place(values: &mut [Complex]) {
    let n = values.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
    let mut width = 2usize;
    while width <= n {
        let angle = -2.0 * std::f32::consts::PI / width as f32;
        let (sin, cos) = angle.sin_cos();
        let half = width / 2;
        for start in (0..n).step_by(width) {
            let mut twiddle = Complex { re: 1.0, im: 0.0 };
            for offset in 0..half {
                let even = values[start + offset];
                let odd_source = values[start + offset + half];
                let odd = Complex {
                    re: twiddle.re * odd_source.re - twiddle.im * odd_source.im,
                    im: twiddle.re * odd_source.im + twiddle.im * odd_source.re,
                };
                values[start + offset] = Complex {
                    re: even.re + odd.re,
                    im: even.im + odd.im,
                };
                values[start + offset + half] = Complex {
                    re: even.re - odd.re,
                    im: even.im - odd.im,
                };
                twiddle = Complex {
                    re: twiddle.re * cos - twiddle.im * sin,
                    im: twiddle.re * sin + twiddle.im * cos,
                };
            }
        }
        width *= 2;
    }
}

fn finite_clamp(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

fn one_minus_power(base: f32, exponent: f32) -> f32 {
    let base = finite_clamp(base, 0.0, 1.0, 0.0);
    let exponent = finite_clamp(exponent, 0.0, 60.0, 0.0);
    (1.0 - (1.0 - base).powf(exponent)).clamp(0.0, 1.0)
}

fn sample_resampled_nonnegative(source: &[f32], index: usize, target_len: usize) -> f32 {
    sample_resampled(source, index, target_len).max(0.0)
}

/// Retain a bounded, finite source row without resampling it into a claim of
/// additional independent bins. `spectrum_rows` is also consumed by mode 8,
/// whose two-adjacent-bin geometry depends on the source indexing remaining
/// faithful.
fn copy_magnitude_row(source: &[f32], destination: &mut Vec<f32>) {
    destination.clear();
    destination.extend(source.iter().take(ENHANCED_FFT_BINS).map(|value| {
        if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        }
    }));
}

fn stereo_magnitude(left: &[f32], right: &[f32], index: usize) -> f32 {
    (sample_resampled_nonnegative(left, index, ENHANCED_FFT_BINS)
        + sample_resampled_nonnegative(right, index, ENHANCED_FFT_BINS))
        * 0.5
}

fn sample_resampled_signed(source: &[f32], index: usize, target_len: usize) -> f32 {
    sample_resampled(source, index, target_len).clamp(-1.0, 1.0)
}

fn sample_resampled(source: &[f32], index: usize, target_len: usize) -> f32 {
    if source.is_empty() || target_len == 0 {
        return 0.0;
    }
    if source.len() == 1 || target_len == 1 {
        return finite_clamp(source[0], -1.0e20, 1.0e20, 0.0);
    }
    let position =
        index.min(target_len - 1) as f32 * (source.len() - 1) as f32 / (target_len - 1) as f32;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(source.len() - 1);
    let fraction = position - lower as f32;
    let a = finite_clamp(source[lower], -1.0e20, 1.0e20, 0.0);
    let b = finite_clamp(source[upper], -1.0e20, 1.0e20, 0.0);
    a + (b - a) * fraction
}

fn pack_rows_rgba16f(rows: &[f32], packed: &mut Vec<u16>) {
    packed.clear();
    packed.reserve(
        rows.len()
            .saturating_mul(4)
            .saturating_sub(packed.capacity()),
    );
    for &value in rows {
        let value = finite_clamp(value, -65_504.0, 65_504.0, 0.0);
        let half = f32_to_f16_bits(value);
        packed.extend_from_slice(&[half, half, half, f32_to_f16_bits(1.0)]);
    }
}

/// IEEE-754 binary16 conversion without adding an implementation dependency to
/// the renderer crate. Values were bounded before calling this function.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        return sign | ((mantissa >> shift) as u16);
    }
    if exponent >= 31 {
        return sign | 0x7bff;
    }
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_finite_and_bounded() {
        let mut processor = EnhancedAudioProcessor::new();
        let frame = processor.update(&[], &[], &[], f32::NAN, f32::INFINITY, Default::default());
        assert_eq!(frame.sample_rate_hz, LEGACY_ASSUMED_SAMPLE_RATE_HZ);
        assert!(frame.fft_rows.iter().all(|v| v.is_finite() && *v == 0.0));
        assert!(frame
            .waveform_rows
            .iter()
            .all(|v| v.is_finite() && *v == 0.0));
        assert_eq!(frame.fft_rows.len(), ENHANCED_FFT_BINS * 2);
        assert_eq!(frame.waveform_rows.len(), ENHANCED_WAVE_SAMPLES * 2);
    }

    #[test]
    fn attack_decay_and_peak_hold_use_elapsed_time() {
        let mut processor = EnhancedAudioProcessor::new();
        let config = EnhancedAudioConfig {
            fft_attack: 1.0,
            // BeatDrop squares (1 - FFTDecay) before deriving its fall
            // coefficient, so 0.0 is the immediate-decay endpoint.
            fft_decay: 0.0,
            fft_scaling: 1.0,
            fft_noise_floor: 0.0,
        };
        let hot = vec![1.0; 512];
        let silent = vec![0.0; 512];
        let raised = processor.update(&hot, &silent, &silent, 48_000.0, 1.0 / 60.0, config);
        let peak_before = raised.fft_rows[ENHANCED_FFT_BINS + 128];
        assert!(peak_before > 0.0);
        let held = processor.update(&silent, &silent, &silent, 48_000.0, 0.2, config);
        assert_eq!(held.fft_rows[ENHANCED_FFT_BINS + 128], peak_before);
        let falling = processor.update(&silent, &silent, &silent, 48_000.0, 0.1, config);
        assert!(falling.fft_rows[ENHANCED_FFT_BINS + 128] < peak_before);
    }

    #[test]
    fn mono_source_does_not_claim_mode8_stereo_and_packs_filterable_rows() {
        let mut processor = EnhancedAudioProcessor::new();
        let frame = processor.update(
            &[0.0, 1.0],
            &[-1.0, 1.0],
            &[1.0, -1.0],
            96_000.0,
            1.0 / 60.0,
            Default::default(),
        );
        assert_eq!(frame.nyquist_hz(), 48_000.0);
        assert!(frame.spectrum_rows().is_none());
        assert!(frame.waveform_rows[0] < 0.0);
        assert!(frame.waveform_rows[ENHANCED_WAVE_SAMPLES] > 0.0);
        let mut fft_upload = Vec::new();
        let mut wave_upload = Vec::new();
        frame.write_fft_rgba16f(&mut fft_upload);
        frame.write_waveform_rgba16f(&mut wave_upload);
        assert_eq!(fft_upload.len(), ENHANCED_FFT_BINS * 2 * 4);
        assert_eq!(wave_upload.len(), ENHANCED_WAVE_SAMPLES * 2 * 4);
    }

    #[test]
    fn exposes_bounded_independent_rows_for_mode8() {
        let mut processor = EnhancedAudioProcessor::new();
        let frame = processor.update_stereo_spectrum(
            &[0.25, f32::NAN, 2.0],
            &[0.5, -1.0, 4.0],
            &[0.0; 8],
            &[0.0; 8],
            48_000.0,
            1.0 / 60.0,
            Default::default(),
        );
        assert_eq!(
            frame.spectrum_rows(),
            Some((&[0.25, 0.0, 2.0][..], &[0.5, 0.0, 4.0][..]))
        );

        let pcm = [0.0; ENHANCED_WAVE_SAMPLES];
        let frame = processor.update_from_pcm(&pcm, &pcm, 48_000.0, 1.0 / 60.0, Default::default());
        let (left, right) = frame.spectrum_rows().expect("PCM has true L/R rows");
        assert_eq!(left.len(), ENHANCED_FFT_BINS);
        assert_eq!(right.len(), ENHANCED_FFT_BINS);
        assert!(left.iter().chain(right).all(|value| value.is_finite()));
    }

    #[test]
    fn pcm_fft_is_bounded_and_reports_actual_capture_window() {
        let pcm: Vec<f32> = (0..ENHANCED_WAVE_SAMPLES)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 8.0 * i as f32 / ENHANCED_WAVE_SAMPLES as f32).sin()
            })
            .collect();
        let mut work = vec![Complex::default(); ENHANCED_FFT_WINDOW_SAMPLES];
        let mut magnitudes = Vec::new();
        fft_magnitudes_from_pcm(&pcm, pcm.len(), &mut work, &mut magnitudes);
        let peak = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        // 512 captured samples, zero-padded to 2048: source bin 8 lands at 32.
        assert!((peak as isize - 32).abs() <= 1, "peak bin {peak}");

        let mut processor = EnhancedAudioProcessor::new();
        let frame = processor.update_from_pcm(
            &pcm,
            &pcm,
            48_000.0,
            1.0 / 60.0,
            EnhancedAudioConfig {
                fft_attack: 1.0,
                fft_decay: 0.7,
                fft_scaling: 1.0,
                fft_noise_floor: 0.0,
            },
        );
        assert_eq!(
            frame.spectrum_source,
            EnhancedSpectrumSource::PcmFft {
                input_samples: ENHANCED_WAVE_SAMPLES,
                transform_bins: ENHANCED_FFT_BINS,
            }
        );
        assert!(frame.fft_rows.iter().all(|value| value.is_finite()));
    }
}
