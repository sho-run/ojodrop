//! Real-time audio capture via cpal (spec §1/§2).
//!
//! Builds a realtime cpal **input** stream, preserves the first stereo pair before
//! downmixing to mono, and pushes raw frames into an `rtrb` SPSC ring. The cpal data
//! callback does **no allocation, no FFT, and no locks**: it only converts and pushes.
//! Dropped samples on a full ring are counted as *discontinuities* (P2-AUD-008), not
//! blocked on (a full ring means the DSP worker stalled — better to drop and mark a
//! discontinuity than to block capture or splice non-adjacent audio).
//!
//! ## Device selection (spec §2 — runtime-selected with fallback)
//!
//! When [`CaptureConfig::prefer_loopback`] is set, [`start`] walks a fallback
//! chain so the engine reacts to the *system mix* (whatever music is playing),
//! not the room mic:
//!
//! 1. **Virtual loopback device by name** — an input device whose name contains a
//!    known virtual-cable marker (`BlackHole`, `Loopback`, `Soundflower`,
//!    `Aggregate`, `Multi-Output`). This is the zero-permission, most-reliable
//!    path for a DJ/VJ rig, so it is tried **first**. If it is present but fails to
//!    open (a stale/again-in-use device), the failure is recorded and the chain
//!    **continues** to the next rung rather than aborting (P2-AUD-023).
//! 2. **macOS CoreAudio process-tap loopback** (14.6+) — captures the post-volume
//!    system mix with no virtual device, but needs the TCC audio-capture grant.
//!    See [`crate::capture_macos`] (currently a structured stub). When it is not
//!    available it reports so honestly and the chain proceeds (P2-AUD-022).
//! 3. **Default input device** (mic / line-in) — the universally available
//!    fallback. When reached after preferring loopback it is tagged
//!    [`CaptureSource::MicrophoneFallback`] ("mic (loopback unavailable)") so the
//!    reported source is never mislabeled as loopback.
//!
//! Other platforms have their loopback rungs stubbed with TODOs (Windows WASAPI
//! render-loopback, Linux PipeWire/Pulse `.monitor`) but always fall through to
//! the default input device, which captures real audio today.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, Stream, StreamConfig};
use rtrb::{Consumer, Producer};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::{AudioError, CaptureFrame};

/// Substrings that mark an input device as a system-audio virtual loopback /
/// aggregate cable. Matched case-insensitively against the device name.
///
/// These are the common macOS/cross-platform virtual cables and aggregate
/// devices a DJ/VJ rig routes its master out through, so opening one as a normal
/// cpal input captures the post-volume system mix with **zero permissions**.
const LOOPBACK_NAME_MARKERS: &[&str] = &[
    "blackhole",
    "loopback",
    "soundflower",
    "aggregate",
    "multi-output",
];

/// Seconds of headroom the SPSC capture ring buffers. Ring capacity is derived
/// from this and the device's *actual* sample rate (see [`ring_capacity`]), never
/// a fixed sample count (P2-AUD-008), so the headroom stays ~1 s across
/// 44.1k/48k/96k/192k devices instead of shrinking at high rates.
const RING_SECONDS: f32 = 1.0;

/// SPSC ring capacity in frames for `seconds` of headroom at `sample_rate` Hz:
/// `ceil(seconds * rate)`, floored at 1. Pure so the sizing is testable without a
/// device.
fn ring_capacity(seconds: f32, sample_rate: u32) -> usize {
    ((seconds.max(0.0) * sample_rate as f32).ceil() as usize).max(1)
}

/// Which physical/virtual path the capture stream ended up on.
///
/// Exposed so the app can show/log whether reactivity is driven by the system
/// mix (loopback/virtual) or the room mic, and branch UI accordingly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CaptureSource {
    /// A native OS system-mix loopback (macOS CoreAudio process-tap, future
    /// WASAPI render-loopback, future PipeWire `.monitor`). Post-volume desktop
    /// audio with no virtual cable.
    SystemLoopback,
    /// A virtual loopback / aggregate device opened by name (BlackHole, Loopback,
    /// Soundflower, an Aggregate or Multi-Output device). Also the system mix, via
    /// a user-installed cable.
    VirtualDevice,
    /// The default input device (microphone / line-in), chosen because loopback
    /// was **not** requested — the room, not the mix.
    Microphone,
    /// The default input device, chosen because loopback **was** requested but no
    /// loopback path (virtual device or native tap) was available. Honestly a mic
    /// fallback — reported distinctly so callers never present it as loopback
    /// (P2-AUD-022).
    MicrophoneFallback,
}

impl CaptureSource {
    /// `true` when this source carries the post-volume *system mix* rather than
    /// the room mic (so visuals follow the music even at low room volume). Both
    /// mic variants return `false`.
    pub fn is_loopback(self) -> bool {
        matches!(
            self,
            CaptureSource::SystemLoopback | CaptureSource::VirtualDevice
        )
    }

    /// Short human-readable label for logs / UI (e.g. "system loopback").
    pub fn label(self) -> &'static str {
        match self {
            CaptureSource::SystemLoopback => "system loopback",
            CaptureSource::VirtualDevice => "virtual loopback device",
            CaptureSource::Microphone => "mic",
            CaptureSource::MicrophoneFallback => "mic (loopback unavailable)",
        }
    }
}

/// The honest mic source tag: a plain [`CaptureSource::Microphone`] when loopback
/// was never requested, or [`CaptureSource::MicrophoneFallback`] when loopback was
/// preferred but no loopback path was available — so the reported source never
/// claims to be loopback when it is really the mic (P2-AUD-022).
fn mic_source(loopback_requested: bool) -> CaptureSource {
    if loopback_requested {
        CaptureSource::MicrophoneFallback
    } else {
        CaptureSource::Microphone
    }
}

/// Whether platform-native system loopback (macOS process-tap / Windows WASAPI
/// render-loopback / Linux PipeWire monitor) is implemented and available on this
/// build.
///
/// Currently `false` on every platform: the native tap is a structured stub (see
/// [`crate::capture_macos`], BLOCKED on a TCC-granted 14.6+ device under CI-011),
/// so [`start`] reports loopback UNAVAILABLE and falls back honestly to the mic
/// rather than mislabeling the mic as loopback (P2-AUD-022).
pub fn native_loopback_available() -> bool {
    false
}

/// Capture configuration / preferences.
// `PartialEq`/`Eq` added 2026-08-15 so `reconnect::next_capture_config`'s
// device-fallback choice is assertable in a unit test. The struct is a single
// `bool`, so the derives are total and add no semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureConfig {
    /// Prefer a system-loopback source (post-volume desktop mix) over the mic
    /// when one can be found. When set, [`start`] tries, in order: a virtual
    /// loopback device by name → the platform native loopback (macOS process-tap)
    /// → the default input device. When unset, only the default input is used.
    pub prefer_loopback: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        // Default to loopback-preferred: the engine's whole purpose is to react to
        // the music playing, not the room mic. Falls back to the mic automatically
        // when no loopback/virtual source exists, so this is safe as a default.
        Self {
            prefer_loopback: true,
        }
    }
}

/// A live capture stream plus the device facts the DSP worker needs.
///
/// The returned [`Stream`] must be kept alive for capture to continue (cpal stops
/// capture when the stream is dropped), so the engine owns it. The matching ring
/// [`Consumer`] is returned separately by [`start`] for the DSP worker.
pub struct Capture {
    /// Held only for its RAII lifetime: capture runs until this is dropped. The
    /// engine never reads it again, hence `dead_code` is expected and allowed.
    #[allow(dead_code)]
    pub stream: Stream,
    /// Count of realtime frames dropped on a full ring (each a discontinuity where
    /// the DSP side received non-adjacent audio). Shared with the realtime callback
    /// (producer side) and observed by the DSP worker (P2-AUD-008).
    pub overruns: Arc<AtomicU64>,
    pub sample_rate: u32,
    /// Native channel count of the source (informational; downmix is to mono).
    #[allow(dead_code)]
    pub channels: u16,
    pub device_name: String,
    /// Which selection path produced this stream (loopback/virtual/mic).
    pub source: CaptureSource,
}

/// Start capturing, selecting the source per `cfg` (spec §2).
///
/// On success returns a [`Capture`] whose `stream` is already `play()`-ing plus the
/// ring [`Consumer`] for the DSP worker. The ring is created *inside* here, sized to
/// the selected device's real sample rate (P2-AUD-008).
pub fn start(
    cfg: CaptureConfig,
    running: Arc<AtomicBool>,
) -> Result<(Capture, Consumer<CaptureFrame>), AudioError> {
    let host = cpal::default_host();

    if !cfg.prefer_loopback {
        // prefer_loopback unset: original behavior, default input only, honestly
        // tagged as a plain mic (loopback was never requested).
        return open_default_input(&host, &running, false);
    }

    // Loopback-preferred fallback chain. Each candidate is tried in order; a
    // failure records a diagnostic and falls through to the next candidate rather
    // than aborting — a stale/again-in-use virtual device no longer blocks native
    // loopback or the default input (P2-AUD-023).
    let mut candidates: Vec<(
        String,
        Box<dyn FnMut() -> Result<(Capture, Consumer<CaptureFrame>), AudioError> + '_>,
    )> = Vec::new();

    // Rung 1: a virtual loopback / aggregate device matched by name.
    if let Some((device, name)) = find_virtual_loopback_device(&host) {
        log::info!("particle-audio: found virtual loopback device '{name}'");
        let label = format!("virtual loopback device '{name}'");
        let running = &running;
        candidates.push((
            label,
            Box::new(move || {
                open_input_device(&device, name.clone(), CaptureSource::VirtualDevice, running)
            }),
        ));
    } else {
        log::debug!(
            "particle-audio: no virtual loopback device found \
             (markers: {LOOPBACK_NAME_MARKERS:?}); trying native loopback"
        );
    }

    // Rung 2: platform native system loopback (macOS process-tap / …). The tap is
    // a structured stub today, so this reports UNAVAILABLE (recorded as a
    // diagnostic) and the chain proceeds to the mic — never mislabeling the mic as
    // loopback (P2-AUD-022).
    {
        let host = &host;
        candidates.push((
            "native system loopback".to_string(),
            Box::new(move || match try_native_loopback(host) {
                NativeLoopback::Opened(cap, consumer) => Ok((cap, consumer)),
                NativeLoopback::Unavailable(reason) => Err(AudioError::LoopbackUnavailable(reason)),
            }),
        ));
    }

    // Rung 3: default input device (mic / line-in), honestly tagged as a loopback
    // fallback so the reported source is never claimed to be loopback.
    {
        let host = &host;
        let running = &running;
        candidates.push((
            "default input".to_string(),
            Box::new(move || open_default_input(host, running, true)),
        ));
    }

    let (opened, diagnostics) = try_candidates(&mut candidates);
    match opened {
        Some(cap) => {
            if !diagnostics.is_empty() {
                log::warn!(
                    "particle-audio: capture opened after earlier fallbacks: {}",
                    diagnostics.join("; ")
                );
            }
            Ok(cap)
        }
        None => Err(AudioError::Stream(format!(
            "all capture candidates failed: {}",
            diagnostics.join("; ")
        ))),
    }
}

/// Run each capture-open candidate in order and return the first success, along
/// with every failed candidate's diagnostic.
///
/// A failing candidate does NOT abort the chain: the next candidate is still
/// attempted, and each failure's error is retained (never swallowed) so a later
/// success still surfaces the earlier failures (P2-AUD-023). Pure over the
/// injected candidate closures, so the continue-and-retain behavior is testable
/// without a live device.
fn try_candidates<'a, T>(
    candidates: &mut [(String, Box<dyn FnMut() -> Result<T, AudioError> + 'a>)],
) -> (Option<T>, Vec<String>) {
    let mut diagnostics = Vec::new();
    for (label, attempt) in candidates.iter_mut() {
        match attempt() {
            Ok(value) => return (Some(value), diagnostics),
            Err(e) => diagnostics.push(format!("{label}: {e}")),
        }
    }
    (None, diagnostics)
}

/// Outcome of the platform-native system-loopback attempt.
enum NativeLoopback {
    /// A native loopback opened; capture is running into the returned ring.
    /// Reserved for the native-tap implementation (see [`try_native_loopback`]);
    /// the current stubs never construct it.
    #[allow(dead_code)]
    Opened(Capture, Consumer<CaptureFrame>),
    /// Native loopback is not available on this platform/build; the reason is
    /// preserved so the caller can log it and fall back honestly.
    Unavailable(String),
}

/// Attempt the platform-native system-loopback path.
///
/// Returns [`NativeLoopback::Opened`] with a live capture + ring consumer, or
/// [`NativeLoopback::Unavailable`] with a reason so the caller can continue to the
/// default input while preserving the diagnostic (P2-AUD-022/023).
#[allow(unused_variables)]
fn try_native_loopback(host: &cpal::Host) -> NativeLoopback {
    #[cfg(target_os = "macos")]
    {
        use crate::capture_macos::{try_open_process_tap, TapOutcome};
        match try_open_process_tap() {
            TapOutcome::Opened(tap) => {
                // TODO(macos-tap): the stub never reaches here. When the real tap
                // lands it owns its own ring; bridge (Capture { source:
                // SystemLoopback, .. }, consumer) out and return
                // NativeLoopback::Opened. BLOCKED under CI-011 (needs a TCC-granted
                // 14.6+ device). Kept so the wiring point is explicit.
                let _ = tap;
                unreachable!("process-tap stub never opens");
            }
            TapOutcome::Declined(reason) => NativeLoopback::Unavailable(reason),
        }
    }

    #[cfg(target_os = "windows")]
    {
        // TODO(windows-loopback): WASAPI render-loopback. cpal can open a render
        // endpoint in loopback mode; alternatively use the `wasapi` crate (MIT).
        // No admin and no virtual device required. Until implemented, unavailable.
        NativeLoopback::Unavailable(
            "WASAPI render-loopback not implemented on this build".to_string(),
        )
    }

    #[cfg(target_os = "linux")]
    {
        // TODO(linux-loopback): PipeWire/PulseAudio default-sink `.monitor` source.
        // Enumerate input devices for a `.monitor` (Pulse) or the sink monitor
        // (PipeWire) and open it via `open_input_device` with
        // CaptureSource::SystemLoopback. Until implemented, unavailable.
        NativeLoopback::Unavailable(
            "PipeWire/Pulse .monitor loopback not implemented on this build".to_string(),
        )
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        NativeLoopback::Unavailable("native loopback unsupported on this platform".to_string())
    }
}

/// Find the first input device whose name matches a known virtual-loopback marker.
///
/// Case-insensitive substring match against [`LOOPBACK_NAME_MARKERS`]. Returns the
/// device and its resolved name. `None` if enumeration fails or nothing matches.
fn find_virtual_loopback_device(host: &cpal::Host) -> Option<(cpal::Device, String)> {
    let devices = match host.input_devices() {
        Ok(d) => d,
        Err(e) => {
            log::warn!("particle-audio: could not enumerate input devices: {e}");
            return None;
        }
    };

    for device in devices {
        let name = match device_name(&device) {
            Some(n) => n,
            None => continue,
        };
        if is_loopback_name(&name) {
            return Some((device, name));
        }
    }
    None
}

/// `true` if a device name matches a known virtual-loopback marker
/// (case-insensitive substring). Pure function, kept separate so the matching
/// rules are unit-testable without a live device.
fn is_loopback_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    LOOPBACK_NAME_MARKERS.iter().any(|m| lower.contains(m))
}

/// Resolve a device's human-readable name (cpal 0.18 `description().name()`).
fn device_name(device: &cpal::Device) -> Option<String> {
    device.description().ok().map(|d| d.name().to_string())
}

/// Open the default input device (mic / line-in) — the universal fallback.
///
/// `loopback_requested` selects the honest source tag: `false` → plain
/// [`CaptureSource::Microphone`]; `true` (loopback preferred but unavailable) →
/// [`CaptureSource::MicrophoneFallback`] (P2-AUD-022).
fn open_default_input(
    host: &cpal::Host,
    running: &Arc<AtomicBool>,
    loopback_requested: bool,
) -> Result<(Capture, Consumer<CaptureFrame>), AudioError> {
    let device = host
        .default_input_device()
        .ok_or(AudioError::NoInputDevice)?;
    let name = device_name(&device).unwrap_or_else(|| "<unknown input device>".to_string());

    open_input_device(&device, name, mic_source(loopback_requested), running)
}

/// Build + start a cpal input stream on `device`, tagging it with `source`, and
/// create the matching ring.
///
/// Creates the SPSC ring sized to the device's actual sample rate (P2-AUD-008),
/// moves the producer + shared overrun counter into the realtime data callback,
/// and returns the [`Capture`] plus the ring [`Consumer`]. A probe/build failure
/// returns an [`AudioError`] *without* consuming any shared engine state, so the
/// caller can continue its fallback chain (P2-AUD-023).
fn open_input_device(
    device: &cpal::Device,
    device_name: String,
    source: CaptureSource,
    running: &Arc<AtomicBool>,
) -> Result<(Capture, Consumer<CaptureFrame>), AudioError> {
    // Read the *actual* device config (spec §2): many loopback/virtual devices run
    // at 44.1k/96k and have 2+ channels — never assume 48k stereo.
    let supported = device
        .default_input_config()
        .map_err(|e| AudioError::Config(e.to_string()))?;

    let sample_format = supported.sample_format();
    // P2-AUD-016: reject genuinely non-PCM formats (e.g. DSD) explicitly, by name,
    // instead of letting an otherwise-valid device fall through silently.
    sample_format_supported(sample_format)?;

    let config: StreamConfig = supported.into();
    let sample_rate = config.sample_rate;
    let channels = config.channels;

    // P2-AUD-008: size the ring by seconds*rate, not a fixed sample count.
    let capacity = ring_capacity(RING_SECONDS, sample_rate);
    let (producer, consumer) = rtrb::RingBuffer::<CaptureFrame>::new(capacity);
    let overruns = Arc::new(AtomicU64::new(0));

    log::info!(
        "particle-audio: capturing from '{device_name}' @ {sample_rate} Hz, \
         {channels} ch, {sample_format:?} [{}] (ring {capacity} frames)",
        source.label()
    );

    let stream = build_stream(
        device,
        &config,
        sample_format,
        channels,
        producer,
        overruns.clone(),
        running.clone(),
    )?;
    stream
        .play()
        .map_err(|e| AudioError::Stream(e.to_string()))?;

    Ok((
        Capture {
            stream,
            overruns,
            sample_rate,
            channels,
            device_name,
            source,
        },
        consumer,
    ))
}

/// Whether a capture stream can be built for `fmt` by converting samples to f32.
///
/// `Ok` for every PCM integer/float width cpal exposes (I8/I16/I24/I32/I64,
/// U8/U16/U24/U32/U64, F32/F64); `Err` — naming the format — for genuinely
/// non-PCM formats such as DSD, which `from_sample` would misread as PCM
/// (P2-AUD-016). Pure so the format policy is testable without a live device.
fn sample_format_supported(fmt: SampleFormat) -> Result<(), AudioError> {
    match fmt {
        SampleFormat::I8
        | SampleFormat::I16
        | SampleFormat::I24
        | SampleFormat::I32
        | SampleFormat::I64
        | SampleFormat::U8
        | SampleFormat::U16
        | SampleFormat::U24
        | SampleFormat::U32
        | SampleFormat::U64
        | SampleFormat::F32
        | SampleFormat::F64 => Ok(()),
        other => Err(AudioError::UnsupportedSampleFormat(format!("{other:?}"))),
    }
}

/// Build the typed input stream for the device's native sample format, with a
/// downmixing data callback. Generic over the sample type so we support every PCM
/// format cpal exposes (f32 / i16 / I24 / u32 / i64 / …) without duplicating logic
/// (P2-AUD-016).
fn build_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    channels: u16,
    producer: Producer<CaptureFrame>,
    overruns: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
) -> Result<Stream, AudioError> {
    macro_rules! build {
        ($t:ty) => {
            build_typed::<$t>(device, config, channels, producer, overruns, running)
        };
    }
    match sample_format {
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I24 => build!(cpal::I24),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U24 => build!(cpal::U24),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        // Non-PCM (DSD) and any future variant: explicit, named rejection — never
        // a silent drop. `sample_format_supported` already guards this path, so
        // this is defensive belt-and-braces for `#[non_exhaustive]` growth.
        other => Err(AudioError::UnsupportedSampleFormat(format!("{other:?}"))),
    }
}

/// Convert one cpal sample to a sanitized f32 in `[-1, 1]` (non-finite → 0).
#[inline]
fn sanitize_sample<T>(s: T) -> f32
where
    f32: FromSample<T>,
{
    let value = f32::from_sample(s);
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Explicit front-L/R downmix of one interleaved multichannel frame (P2-AUD-009).
///
/// The mono feed is the **front stereo pair average** `0.5*(L+R)`, NOT the mean
/// over all N channels. Averaging over N silently attenuates the mix by
/// `20*log10(active/N)` dB whenever some channels are idle — e.g. a stereo master
/// routed to channels 1-2 of a 16-channel virtual cable loses ~18 dB under the old
/// `sum/N`. Folding only the active front pair keeps the level correct and still
/// downmixes a coherent multichannel signal at unity (L == R == A → A), never
/// clipping (the coefficients sum to 1). `left`/`right` preserve the first two
/// channels for scope-style looks; a mono source duplicates L into R.
#[inline]
fn downmix_frame<T>(frame: &[T]) -> CaptureFrame
where
    T: Copy,
    f32: FromSample<T>,
{
    let left = frame.first().copied().map(sanitize_sample).unwrap_or(0.0);
    let right = if frame.len() >= 2 {
        sanitize_sample(frame[1])
    } else {
        left
    };
    let mono = 0.5 * (left + right);
    CaptureFrame { mono, left, right }
}

/// Push a captured frame into the SPSC ring.
///
/// On a full ring the frame is dropped (better an xrun on the consumer side than
/// blocking the realtime callback) and `overruns` is bumped so the DSP side learns
/// a **discontinuity** occurred — the drained blocks on either side of this drop
/// are non-adjacent audio and must not be spliced into one false-continuous stream
/// (P2-AUD-008). Returns `true` if the frame was accepted.
#[inline]
fn push_frame(
    producer: &mut Producer<CaptureFrame>,
    frame: CaptureFrame,
    overruns: &AtomicU64,
) -> bool {
    match producer.push(frame) {
        Ok(()) => true,
        Err(_) => {
            overruns.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}

fn build_typed<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    channels: u16,
    mut producer: Producer<CaptureFrame>,
    overruns: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
) -> Result<Stream, AudioError>
where
    T: Sample + cpal::SizedSample + Send + 'static,
    f32: FromSample<T>,
{
    let channels = channels.max(1) as usize;

    let data_cb = move |data: &[T], _: &cpal::InputCallbackInfo| {
        // Downmix interleaved frames via the explicit front-L/R matrix and push.
        // No allocation, no locks. A full ring drops the frame and marks a
        // discontinuity (see `push_frame`) rather than blocking the audio thread.
        for frame in data.chunks(channels) {
            push_frame(&mut producer, downmix_frame(frame), &overruns);
        }
    };

    let err_cb = move |err: cpal::Error| {
        log::error!("particle-audio: capture stream error: {err}");
        running.store(false, Ordering::Relaxed);
    };

    device
        .build_input_stream(*config, data_cb, err_cb, None)
        .map_err(|e| AudioError::Stream(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_virtual_loopback_devices() {
        // Real device names from common DJ/VJ rigs (varying case / suffixes).
        for name in [
            "BlackHole 2ch",
            "BlackHole 16ch",
            "Loopback Audio",
            "Soundflower (2ch)",
            "My Aggregate Device",
            "Multi-Output Device",
            "blackhole",     // lowercase
            "ZOOM Loopback", // marker mid-string
        ] {
            assert!(
                is_loopback_name(name),
                "expected '{name}' to match a marker"
            );
        }
    }

    #[test]
    fn does_not_match_real_input_devices() {
        for name in [
            "MacBook Pro Microphone",
            "External Microphone",
            "USB Audio CODEC",
            "Built-in Input",
            "Scarlett 2i2 USB",
        ] {
            assert!(!is_loopback_name(name), "did not expect '{name}' to match");
        }
    }

    #[test]
    fn source_classification_and_labels() {
        assert!(CaptureSource::SystemLoopback.is_loopback());
        assert!(CaptureSource::VirtualDevice.is_loopback());
        assert!(!CaptureSource::Microphone.is_loopback());
        assert!(!CaptureSource::MicrophoneFallback.is_loopback());

        assert_eq!(CaptureSource::SystemLoopback.label(), "system loopback");
        assert_eq!(
            CaptureSource::VirtualDevice.label(),
            "virtual loopback device"
        );
        assert_eq!(CaptureSource::Microphone.label(), "mic");
        assert_eq!(
            CaptureSource::MicrophoneFallback.label(),
            "mic (loopback unavailable)"
        );
    }

    #[test]
    fn default_config_prefers_loopback() {
        assert!(CaptureConfig::default().prefer_loopback);
    }

    // --- P2-AUD-008: ring sizing + overrun discontinuity marker ---

    #[test]
    fn ring_capacity_is_ceil_seconds_times_rate() {
        assert_eq!(ring_capacity(1.0, 48_000), 48_000);
        assert_eq!(ring_capacity(1.0, 44_100), 44_100);
        assert_eq!(ring_capacity(0.5, 48_000), 24_000);
        assert_eq!(ring_capacity(2.0, 96_000), 192_000);
        // ceil, not truncate.
        assert_eq!(
            ring_capacity(0.25, 44_101),
            (0.25 * 44_101.0f32).ceil() as usize
        );
        // Floored at 1 so a degenerate rate never yields a zero-length ring.
        assert!(ring_capacity(0.0, 48_000) >= 1);
    }

    #[test]
    fn ring_overrun_raises_discontinuity_marker() {
        let overruns = AtomicU64::new(0);
        // Keep the consumer bound alive; if it drops, pushes fail as "abandoned".
        let (mut producer, _consumer) = rtrb::RingBuffer::<CaptureFrame>::new(4);

        // Fill the ring — clean pushes, no discontinuity yet.
        let mut pushed = 0usize;
        while push_frame(&mut producer, CaptureFrame::default(), &overruns) {
            pushed += 1;
            assert!(pushed <= 4, "ring should report full at capacity");
        }
        assert_eq!(
            overruns.load(Ordering::Relaxed),
            1,
            "the first rejected push (full ring) raises the discontinuity marker"
        );

        // A further full-ring push marks another discontinuity, never a silent drop.
        assert!(!push_frame(
            &mut producer,
            CaptureFrame::default(),
            &overruns
        ));
        assert_eq!(overruns.load(Ordering::Relaxed), 2);
    }

    // --- P2-AUD-009: explicit multichannel downmix matrix ---

    #[test]
    fn downmix_active_stereo_pair_no_18db_loss() {
        // 16-channel frame, signal only on the front L/R pair (ch0, ch1).
        let mut frame = [0.0f32; 16];
        frame[0] = 0.8;
        frame[1] = 0.6;
        let d = downmix_frame(&frame);
        assert!((d.left - 0.8).abs() < 1e-6);
        assert!((d.right - 0.6).abs() < 1e-6);
        // Front-pair average 0.7 — NOT sum/16 == 1.4/16 == 0.0875 (~18 dB down).
        assert!(
            (d.mono - 0.7).abs() < 1e-6,
            "front-pair downmix must not divide by silent channels, got {}",
            d.mono
        );
        let old_avg_over_n = 1.4 / 16.0;
        assert!(
            d.mono > old_avg_over_n * 4.0,
            "must be well above the ~18 dB-attenuated sum/N level"
        );
    }

    #[test]
    fn downmix_coherent_multichannel_correct_gain() {
        // A coherent signal on all channels downmixes at unity, not N× (no clip)
        // and not 1/N (no attenuation).
        let frame = [0.5f32; 8];
        let d = downmix_frame(&frame);
        assert!(
            (d.mono - 0.5).abs() < 1e-6,
            "coherent multichannel must stay at source level, got {}",
            d.mono
        );
        assert!(d.mono <= 1.0, "downmix never clips");
    }

    #[test]
    fn downmix_mono_source_duplicates_into_lr() {
        let d = downmix_frame(&[0.4f32]);
        assert!((d.left - 0.4).abs() < 1e-6);
        assert!((d.right - 0.4).abs() < 1e-6);
        assert!((d.mono - 0.4).abs() < 1e-6);
    }

    #[test]
    fn integer_formats_convert_to_normalized_f32() {
        // i16 full-scale → ~±1.0.
        let d = downmix_frame(&[i16::MAX, i16::MIN]);
        assert!(
            (d.left - 1.0).abs() < 1e-3,
            "i16 max → ~1.0, got {}",
            d.left
        );
        assert!(
            (d.right + 1.0).abs() < 1e-3,
            "i16 min → ~-1.0, got {}",
            d.right
        );
        // u16 origin (32768) → ~0.0.
        let d = downmix_frame(&[32_768u16, 32_768u16]);
        assert!(d.mono.abs() < 1e-3, "u16 origin → ~0, got {}", d.mono);
        // u32 origin (1<<31) → ~0.0 — exercises a newly-added (P2-AUD-016) format.
        let d = downmix_frame(&[1u32 << 31, 1u32 << 31]);
        assert!(d.mono.abs() < 1e-3, "u32 origin → ~0, got {}", d.mono);
    }

    // --- P2-AUD-016: support-or-reject every cpal sample format ---

    #[test]
    fn previously_dropped_pcm_formats_now_supported() {
        for fmt in [
            SampleFormat::I24,
            SampleFormat::I64,
            SampleFormat::U24,
            SampleFormat::U32,
            SampleFormat::U64,
        ] {
            assert!(
                sample_format_supported(fmt).is_ok(),
                "{fmt:?} must be handled (converted), not silently dropped"
            );
        }
    }

    #[test]
    fn non_pcm_formats_rejected_by_name() {
        for fmt in [
            SampleFormat::DsdU8,
            SampleFormat::DsdU16,
            SampleFormat::DsdU32,
        ] {
            match sample_format_supported(fmt) {
                Err(AudioError::UnsupportedSampleFormat(name)) => {
                    assert!(
                        name.contains("Dsd"),
                        "rejection must name the format, got '{name}'"
                    );
                }
                other => panic!("expected explicit named rejection for {fmt:?}, got {other:?}"),
            }
        }
    }

    // --- P2-AUD-022: honest source reporting when loopback is unavailable ---

    #[test]
    fn mic_source_reports_loopback_fallback_honestly() {
        // Loopback requested but unavailable → honest fallback tag, NOT loopback.
        let fb = mic_source(true);
        assert_eq!(fb, CaptureSource::MicrophoneFallback);
        assert!(
            !fb.is_loopback(),
            "a mic fallback must never report as loopback"
        );
        assert_eq!(fb.label(), "mic (loopback unavailable)");
        // Loopback never requested → plain mic.
        assert_eq!(mic_source(false), CaptureSource::Microphone);
    }

    #[test]
    fn native_loopback_probe_reports_unavailable() {
        // The native tap is a structured stub (BLOCKED under CI-011); the probe
        // must honestly report unavailable so the mic fallback is labeled truthfully.
        assert!(!native_loopback_available());
    }

    // --- P2-AUD-023: fallback continues past a failed candidate, retains diagnostics ---

    #[test]
    fn fallback_continues_past_failed_candidate_and_retains_diagnostics() {
        let mut candidates: Vec<(String, Box<dyn FnMut() -> Result<&'static str, AudioError>>)> = vec![
            (
                "virtual loopback device 'BlackHole 16ch'".to_string(),
                Box::new(|| Err(AudioError::Stream("device in use".to_string()))),
            ),
            (
                "native system loopback".to_string(),
                Box::new(|| Err(AudioError::LoopbackUnavailable("stub".to_string()))),
            ),
            ("default input".to_string(), Box::new(|| Ok("mic-fallback"))),
        ];
        let (opened, diagnostics) = try_candidates(&mut candidates);
        assert_eq!(
            opened,
            Some("mic-fallback"),
            "a failing candidate must not abort — the next candidate is still tried"
        );
        assert_eq!(
            diagnostics.len(),
            2,
            "earlier failures are retained, not swallowed"
        );
        assert!(diagnostics[0].contains("device in use"));
        assert!(diagnostics[1].contains("native system loopback"));
    }

    #[test]
    fn fallback_all_failed_collects_every_diagnostic() {
        let mut candidates: Vec<(String, Box<dyn FnMut() -> Result<&'static str, AudioError>>)> = vec![
            (
                "a".to_string(),
                Box::new(|| Err(AudioError::Stream("x".to_string()))),
            ),
            (
                "b".to_string(),
                Box::new(|| Err(AudioError::Config("y".to_string()))),
            ),
        ];
        let (opened, diagnostics) = try_candidates(&mut candidates);
        assert_eq!(opened, None);
        assert_eq!(diagnostics.len(), 2);
        assert!(diagnostics[0].contains('x'));
        assert!(diagnostics[1].contains('y'));
    }
}
