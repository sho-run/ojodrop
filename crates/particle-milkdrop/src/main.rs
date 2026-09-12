// particle-milkdrop — renders a single .milk preset using wgpu + winit.
//
// Usage:
//   cargo run -p particle-milkdrop -- preset.milk          # windowed (needs display)
//   cargo run -p particle-milkdrop -- preset.milk --headless 300 out.png  # offscreen

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use particle_audio::{AudioEngine, CaptureConfig};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

// The engine lives in the library crate (single source of truth). This bin is a
// thin CLI/window shell over it.
use particle_milkdrop::{
    enhanced_audio::LEGACY_ASSUMED_SAMPLE_RATE_HZ, fallback_preset, load_preset_path, MilkShaders,
    MilkdropRenderer, MilkdropResizeDebouncer,
};

/// Thin path-based wrapper over the library ingest ([`load_preset_path`]) used by
/// the headless / anim CLI paths and the windowed app. `.json` → Butterchurn
/// loader, anything else → raw `.milk` parser (native converter when built in).
fn load_preset(path: &str) -> Result<MilkShaders, String> {
    load_preset_path(Path::new(path))
}

/// Audio reactivity fed to the renderer for one frame, in MilkDrop/Butterchurn
/// convention: each value is a *volume-independent reactivity ratio* (≈1.0 at the
/// band's recent average, 0 when quiet, >1 on a hit). `*_att` are the smoother
/// attenuated envelopes that lag peaks.
struct AudioFrame {
    bass: f32,
    mid: f32,
    treb: f32,
    vol: f32,
    bass_att: f32,
    mid_att: f32,
    treb_att: f32,
    vol_att: f32,
}

/// Map particle-audio's Butterchurn-faithful reactivity ratios onto MilkDrop's
/// bass/mid/treb/vol(+ _att) convention. These ratios are already in the
/// "~1.0 = average, >1 = loud" space the EEL/shaders expect — no 0.5+1.5x remap
/// (that remap was the wrong-signal hack on top of AGC'd EQ levels).
fn map_audio(f: &particle_audio::Features) -> AudioFrame {
    AudioFrame {
        bass: f.bass_react,
        mid: f.mid_react,
        treb: f.treb_react,
        vol: f.vol_react,
        bass_att: f.bass_react_att,
        mid_att: f.mid_react_att,
        treb_att: f.treb_react_att,
        vol_att: f.vol_react_att,
    }
}


// Share the audio crate's retry policy instead of maintaining a second copy.
use particle_audio::{CaptureDecision, ReconnectState};

// ---------------------------------------------------------------------------
// Headless (offscreen) mode — no display required
// ---------------------------------------------------------------------------

/// Manufactured beat-driven audio (120 BPM) for headless rendering. The Butterchurn
/// oracle (scripts/butterchurn-oracle/render.html) drives an IDENTICAL model so both
/// renderers bloom on the same beats — a fair, audio-driven fidelity comparison
/// instead of the misleading silent-audio one (which leaves reactive presets black).
struct SynthAudio {
    bass: f32,
    mid: f32,
    treb: f32,
    vol: f32,
    bass_att: f32,
    mid_att: f32,
    treb_att: f32,
    vol_att: f32,
    waveform: Vec<f32>, // 512, [-1,1]
    spectrum: Vec<f32>, // 512, [0,1]
}

fn synth_audio(frame: u32, fps: f32) -> SynthAudio {
    use std::f32::consts::PI;
    let t = frame as f32 / fps;
    let bps = 2.0_f32; // 120 BPM
    let beat = t * bps;
    let bp = beat - beat.floor(); // 0..1 within the beat
    let env = (-bp * 5.0).exp(); // kick pulse
    let env_s = (-bp * 2.0).exp(); // smoothed (attenuated band)
    let hp = {
        let x = beat + 0.5;
        x - x.floor()
    };
    let hat = (-hp * 9.0).exp(); // off-beat hi-hat
                                 // Punchy levels (bass peaks ~3.1) — deliberately HOTTER than Butterchurn's
                                 // AGC-normalized reaction to the same beat. This vibrant, energetic look is the
                                 // preferred aesthetic for the engine's output; we do NOT tone it down to match
                                 // the reference's subtler response.
                                 // Higher sustained floors (~1.0 = "average" energy) so presets that build content
                                 // from mid/treb over the run aren't starved between beats — real music (the
                                 // MilkDrop2 references) has broadband sustain, not just a bass-heavy pulse.
                                 // Higher between-beat FLOORS (the constant terms) so feedback-buildup presets
                                 // accumulate over the 0..90 run instead of starving between beats; env/hat
                                 // coefficients trimmed so the on-beat PEAKS stay at the preferred ~3.1 bass
                                 // (do not raise the peak — that only adds washed-white blow-outs).
    let synth_profile = std::env::var("MILKDROP_SYNTH_PROFILE")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let (bass, mid, treb, bass_att, mid_att, treb_att) = match synth_profile.as_str() {
        "balanced" => (
            0.85 + 1.85 * env,
            0.95 + 0.75 * env + 0.45 * hat,
            0.85 + 1.25 * hat + 0.4 * env,
            0.90 + 1.20 * env_s,
            0.90 + 0.55 * env_s,
            0.85 + 0.85 * env_s,
        ),
        "soft" | "subtle" => (
            0.55 + 1.15 * env,
            0.65 + 0.45 * env + 0.25 * hat,
            0.55 + 0.75 * hat + 0.25 * env,
            0.65 + 0.70 * env_s,
            0.65 + 0.35 * env_s,
            0.60 + 0.50 * env_s,
        ),
        _ => (
            1.3 + 1.8 * env,
            1.3 + 0.7 * env + 0.45 * hat,
            1.15 + 1.25 * hat + 0.4 * env,
            1.3 + 1.2 * env_s,
            1.2 + 0.55 * env_s,
            1.1 + 0.85 * env_s,
        ),
    };
    let vol = (bass + mid + treb) / 3.0;
    let vol_att = (bass_att + mid_att + treb_att) / 3.0;

    let n = 512usize;
    let mut waveform = vec![0.0f32; n];
    for k in 0..n {
        let x = k as f32 / n as f32;
        // Band-correct cycle counts so an FFT (Butterchurn) banks energy into the
        // bass / mid / treb thirds respectively.
        let w = 0.45 * (2.0 * PI * x * 5.0 + t * 5.0).sin() * bass.min(2.5)
            + 0.30 * (2.0 * PI * x * 110.0 + t * 9.0).sin() * mid.min(2.5)
            + 0.18 * (2.0 * PI * x * 210.0 + t * 20.0).sin() * treb.min(2.5);
        waveform[k] = (w * 0.4).clamp(-1.0, 1.0);
    }
    let mut spectrum = vec![0.0f32; n];
    for b in 0..n {
        let f = b as f32 / n as f32;
        let bass_band = (-f * 22.0).exp() * bass;
        let mid_band = (-(f - 0.33).abs() * 10.0).exp() * mid;
        let treb_band = (-(f - 0.7).abs() * 7.0).exp() * treb;
        spectrum[b] = ((bass_band + mid_band + treb_band) * 0.22).clamp(0.0, 1.0);
    }
    SynthAudio {
        bass,
        mid,
        treb,
        vol,
        bass_att,
        mid_att,
        treb_att,
        vol_att,
        waveform,
        spectrum,
    }
}

fn run_headless(milk_path: &str, frames: u32, out_path: &str, synth: bool) {
    let (w, h) = std::env::var("MILKDROP_HEADLESS_SIZE")
        .ok()
        .and_then(|s| parse_headless_size(&s))
        .unwrap_or((1280u32, 720u32));

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .or_else(|_| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: true,
        }))
    })
    .expect("no wgpu adapter — no GPU/software renderer found");
    eprintln!("adapter: {:?}", adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("milk-headless"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: Default::default(),
    }))
    .expect("no device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Offscreen RGBA target — comp pipeline writes here instead of a surface
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless-target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    // Parse preset (.milk = HLSL bodies / .json = Butterchurn GLSL bodies).
    // Headless/CLI path: fail loudly (a batch run wants a clear non-zero exit, not a
    // silent fallback). The windowed app path uses graceful fallback instead.
    let shaders = match load_preset(milk_path) {
        Ok(shaders) => shaders,
        Err(e) => {
            eprintln!("load preset: {e}");
            std::process::exit(2);
        }
    };

    let mut rnd = match MilkdropRenderer::new(device.clone(), queue.clone(), w, h, fmt, &shaders) {
        Ok(renderer) => renderer,
        Err(e) => {
            eprintln!("renderer: {e}");
            std::process::exit(3);
        }
    };

    if synth {
        rnd.set_fixed_fps(30.0); // deterministic time so the beat model lands on frames
    }
    eprintln!("Rendering {frames} frames of {milk_path}… (synth_audio={synth})");
    // Push error scopes to catch silent GPU validation failures
    let scope_oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let scope_val = device.push_error_scope(wgpu::ErrorFilter::Validation);
    for i in 0..frames {
        if synth {
            let a = synth_audio(i, 30.0);
            rnd.set_audio(a.bass, a.mid, a.treb, a.vol);
            rnd.set_audio_att(a.bass_att, a.mid_att, a.treb_att, a.vol_att);
            rnd.set_waveform(&a.waveform, &a.waveform);
            rnd.set_freq_spectrum(&a.spectrum);
        }
        rnd.render(&target_view);
        if (i + 1) % 60 == 0 {
            eprint!("  frame {}/{frames}\r", i + 1);
        }
    }
    // Flush and check for GPU errors before reading back
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    if let Some(e) = pollster::block_on(scope_val.pop()) {
        eprintln!("GPU validation error: {e}");
    }
    if let Some(e) = pollster::block_on(scope_oom.pop()) {
        eprintln!("GPU OOM error: {e}");
    }
    eprintln!("Done. Saving {out_path}…");

    // Read back the texture
    let bytes_per_row = align256(w * 4);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let data = slice.get_mapped_range();
    // Strip padding: each row is `bytes_per_row` bytes but only `w*4` are pixel data
    let stride = bytes_per_row as usize;
    let row_bytes = (w * 4) as usize;
    let mut pixels: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        pixels.extend_from_slice(&data[row * stride..row * stride + row_bytes]);
    }
    drop(data);
    readback.unmap();

    // Save PNG
    let f = std::fs::File::create(out_path).expect("create png");
    let mut enc = png::Encoder::new(f, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(&pixels).expect("png data");
    println!("Saved {out_path}  ({w}x{h})");
}

fn parse_headless_size(value: &str) -> Option<(u32, u32)> {
    let (w, h) = value.split_once('x').or_else(|| value.split_once('X'))?;
    let w = w.trim().parse::<u32>().ok()?;
    let h = h.trim().parse::<u32>().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

fn align256(n: u32) -> u32 {
    (n + 255) & !255
}

// ---------------------------------------------------------------------------
// Animation export — render a sequence of PNG frames at a fixed timestep
// ---------------------------------------------------------------------------

fn run_anim(milk_path: &str, frames: u32, out_dir: &str) {
    let (w, h) = (640u32, 360u32); // smaller for quick GIF assembly

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .or_else(|_| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: true,
        }))
    })
    .expect("no wgpu adapter");
    eprintln!("adapter: {:?}", adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("milk-anim"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: Default::default(),
    }))
    .expect("no device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("anim-target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    let shaders = match load_preset(milk_path) {
        Ok(shaders) => shaders,
        Err(e) => {
            eprintln!("load preset: {e}");
            std::process::exit(2);
        }
    };

    let mut rnd = match MilkdropRenderer::new(device.clone(), queue.clone(), w, h, fmt, &shaders) {
        Ok(renderer) => renderer,
        Err(e) => {
            eprintln!("renderer: {e}");
            std::process::exit(3);
        }
    };
    rnd.set_fixed_fps(30.0); // deterministic time so synthetic audio animates

    std::fs::create_dir_all(out_dir).expect("create out dir");
    eprintln!("Rendering {frames} frames of {milk_path} → {out_dir}/ …");

    let bytes_per_row = align256(w * 4);
    for i in 0..frames {
        rnd.render(&target_view);

        // Read back this frame
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let data = slice.get_mapped_range();
        let stride = bytes_per_row as usize;
        let row_bytes = (w * 4) as usize;
        let mut pixels: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h as usize {
            pixels.extend_from_slice(&data[row * stride..row * stride + row_bytes]);
        }
        drop(data);
        readback.unmap();

        let path = format!("{out_dir}/frame_{i:04}.png");
        let f = std::fs::File::create(&path).expect("create png");
        let mut penc = png::Encoder::new(f, w, h);
        penc.set_color(png::ColorType::Rgba);
        penc.set_depth(png::BitDepth::Eight);
        let mut writer = penc.write_header().expect("png header");
        writer.write_image_data(&pixels).expect("png data");

        if (i + 1) % 30 == 0 {
            eprint!("  frame {}/{frames}\r", i + 1);
        }
    }
    eprintln!("\nDone. {frames} frames in {out_dir}/");
}

// ---------------------------------------------------------------------------
// Windowed mode
// ---------------------------------------------------------------------------

struct App {
    /// Preset to load on startup, or `None` to open in the idle "drop a file"
    /// empty-state. After launch, presets arrive by drag-and-drop.
    initial_path: Option<String>,
    window: Option<Arc<Window>>,
    state: Option<GpuState>,
}

const APP_NAME: &str = "OjoDrop";
const IDLE_TITLE: &str = "OjoDrop — drag a .milk or .json preset here";

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    /// Kept alive so dropped presets can rebuild the renderer in place (and so mic
    /// capture stays funded). The renderer holds its own clone.
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    renderer: MilkdropRenderer,
    active_shaders: MilkShaders,
    /// Coalesces window drag events before calling the renderer's in-place
    /// resize path, so feedback/EEL state survives and target work stays bounded.
    resize_debouncer: MilkdropResizeDebouncer,
    audio: Option<AudioEngine>,
    /// Reconnect bookkeeping. Reset to 0 only when a live engine is *observed*
    /// running — never on a merely successful construction.
    audio_reconnect: ReconnectState,
    /// Epoch the reconnect state machine's monotonic durations are measured
    /// from. Fixed at construction; the state itself owns the attempt clock.
    audio_epoch: Instant,
}

impl GpuState {
    fn new(window: Arc<Window>, initial_path: Option<&str>) -> Self {
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).expect("create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no adapter");
        log::info!("adapter: {:?}", adapter.get_info());

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("milk-device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        }))
        .expect("no device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            caps.present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::Fifo)
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Opaque),
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Windowed app path: an untrusted drag-dropped file must never crash the app.
        // No initial file → idle empty-state (passthrough). On a load error
        // (unreadable / non-UTF-8 / malformed JSON) fall back to passthrough too.
        let shaders = match initial_path {
            Some(p) => {
                let s = load_preset(p).unwrap_or_else(|e| {
                    log::error!("{e} — using passthrough fallback");
                    fallback_preset()
                });
                if s.warp.is_none() {
                    log::warn!("no warp shader found in {p}");
                }
                if s.comp.is_none() {
                    log::warn!("no comp shader found in {p}");
                }
                s
            }
            None => fallback_preset(),
        };
        let (mut renderer, _compiled) =
            Self::build_renderer(&device, &queue, w, h, format, &shaders);

        // Start live mic capture (prefer the room mic, not system loopback).
        let audio = match AudioEngine::with_config(
            CaptureConfig {
                prefer_loopback: false,
            },
            1.0,
        ) {
            Ok(eng) => {
                log::info!(
                    "audio: capturing '{}' @ {} Hz",
                    eng.device_name(),
                    eng.sample_rate()
                );
                Some(eng)
            }
            Err(e) => {
                log::warn!("audio: {e} — retrying with backoff, synthetic reactivity meanwhile");
                None
            }
        };
        renderer.set_enhanced_audio_sample_rate(
            audio
                .as_ref()
                .map(|engine| engine.sample_rate() as f32)
                .unwrap_or(LEGACY_ASSUMED_SAMPLE_RATE_HZ),
        );

        Self {
            surface,
            device,
            queue,
            config,
            renderer,
            active_shaders: shaders,
            resize_debouncer: MilkdropResizeDebouncer::default(),
            // A failed construction counts as attempt #1, so the frame loop waits
            // the base delay rather than re-enumerating devices on the very next
            // frame. A successful one starts the counter clean.
            audio_reconnect: ReconnectState::with_failures(u32::from(audio.is_none())),
            audio_epoch: Instant::now(),
            audio,
        }
    }

    /// Retry lost capture with bounded backoff, using the caller's clock.
    fn ensure_audio_capture_running(&mut self, now: Instant) {
        let live = self.audio.as_ref().is_some_and(AudioEngine::is_running);
        // Dropped before the backoff gate rather than after it — the engine owns
        // the cpal stream and joins its DSP worker on drop, so it must not hold
        // the device across the wait for the very reconnect we are waiting on.
        // (Ordering preserved from the original implementation, not changed here.)
        if !live && self.audio.take().is_some() {
            log::warn!("audio: capture stopped; attempting to reconnect");
        }

        let elapsed = now.saturating_duration_since(self.audio_epoch);
        // OjoDrop asks for the plain default input, so the crate's device-fallback
        // rule has no alternate to offer and hands this straight back — the
        // `cfg` binding is still taken from the decision rather than rebuilt, so
        // a later change to OjoDrop's request automatically gets the fallback.
        let requested = CaptureConfig {
            prefer_loopback: false,
        };
        let CaptureDecision::Reconnect(cfg) = self.audio_reconnect.poll(live, elapsed, requested)
        else {
            return;
        };

        match AudioEngine::with_config(cfg, 1.0) {
            Ok(eng) => {
                log::info!(
                    "audio: reconnected to '{}' @ {} Hz",
                    eng.device_name(),
                    eng.sample_rate()
                );
                self.renderer
                    .set_enhanced_audio_sample_rate(eng.sample_rate() as f32);
                // Deliberately NOT resetting the failure count here — see
                // [`ReconnectState::poll`]. Capture must remain live for the
                // settle window before it clears; a device that opens and
                // immediately dies must keep escalating.
                self.audio = Some(eng);
            }
            Err(e) => {
                self.audio_reconnect.note_open_error(e.to_string());
                let next = self.audio_reconnect.delay();
                log::warn!("audio: reconnect failed ({e}); next attempt in {next:?}");
            }
        }
    }

    /// Build a renderer for `shaders`, falling back to the passthrough preset on a
    /// shader-compile error so an unsupported or hostile preset never crashes the
    /// app. The passthrough is a known-good internal asset; if it *also* fails that
    /// is a genuine engine bug worth surfacing loudly.
    ///
    /// Returns `(renderer, compiled)` where `compiled` is `true` when the preset's
    /// OWN shaders built, and `false` when we fell back to passthrough — so the UI
    /// can tell the user a preset is unsupported instead of silently showing blank.
    fn build_renderer(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        shaders: &MilkShaders,
    ) -> (MilkdropRenderer, bool) {
        match MilkdropRenderer::new(device.clone(), queue.clone(), w, h, format, shaders) {
            Ok(r) => (r, true),
            Err(e) => {
                log::error!("shader compile failed ({e}) — using passthrough fallback preset");
                let fallback = fallback_preset();
                let r =
                    MilkdropRenderer::new(device.clone(), queue.clone(), w, h, format, &fallback)
                        .unwrap_or_else(|e2| panic!("fallback renderer also failed: {e2}"));
                (r, false)
            }
        }
    }

    /// Load a dropped preset, rebuilding the renderer in place. Never panics on
    /// bad input: a file/parse error keeps the current visuals; a shader-compile
    /// error falls back to passthrough. Returns the human-readable outcome for the
    /// window title bar.
    fn load_path(&mut self, path: &str) -> String {
        let name = std::path::Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string());
        let is_json = path.to_ascii_lowercase().ends_with(".json");
        if !is_json && !particle_milkdrop::native_converter_available() {
            log::warn!(
                "{name}: no native .milk converter helper is available; rendering may be \
                 degraded. Drop a .json preset, or install/configure a converter helper."
            );
        }
        match load_preset(path) {
            Ok(shaders) => {
                if shaders.warp.is_none() {
                    log::warn!("no warp shader found in {name}");
                }
                if shaders.comp.is_none() {
                    log::warn!("no comp shader found in {name}");
                }
                let (w, h) = (self.config.width, self.config.height);
                let (mut renderer, compiled) = Self::build_renderer(
                    &self.device,
                    &self.queue,
                    w,
                    h,
                    self.config.format,
                    &shaders,
                );
                renderer.set_enhanced_audio_sample_rate(
                    self.audio
                        .as_ref()
                        .map(|engine| engine.sample_rate() as f32)
                        .unwrap_or(LEGACY_ASSUMED_SAMPLE_RATE_HZ),
                );
                self.renderer = renderer;
                self.active_shaders = shaders;
                // The fresh renderer already uses the current surface size.
                self.resize_debouncer.clear();
                if compiled {
                    log::info!("loaded {name}");
                    format!("{APP_NAME} — {name}")
                } else {
                    log::warn!("{name}: preset shader did not compile — showing passthrough");
                    format!("{APP_NAME} — {name}  ·  shader unsupported")
                }
            }
            Err(e) => {
                log::error!("{e} — keeping current preset");
                format!("{APP_NAME} — could not load {name}")
            }
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        let (w, h) = (size.width.max(1), size.height.max(1));
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.resize_debouncer.request(w, h, Instant::now());
    }

    fn render(&mut self) {
        let now = Instant::now();
        if let Some((width, height)) = self.resize_debouncer.take_ready(now) {
            if let Err(error) = self.renderer.try_resize(width, height) {
                log::warn!(
                    "MilkDrop resize to {width}x{height} rejected; retaining live targets: {error}"
                );
            }
        }
        // Before the surface acquire, not after: the `Outdated`/`Lost` arm below
        // returns early, and audio recovery must not be starved by a window
        // that is busy being resized or reconfigured.
        self.ensure_audio_capture_running(now);
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            other => {
                log::warn!("surface: {other:?}");
                return;
            }
        };
        // Pull the latest mic analysis and drive reactivity.
        if let Some(eng) = &self.audio {
            let f = eng.latest();
            let a = map_audio(&f);
            self.renderer.set_audio(a.bass, a.mid, a.treb, a.vol);
            self.renderer
                .set_audio_att(a.bass_att, a.mid_att, a.treb_att, a.vol_att);
            // Feed the full-resolution 512-sample waveform (time-domain) and the
            // 512-bin freq_spectrum so `bSpectrum` custom waveforms read real FFT bins.
            self.renderer
                .set_waveform(&f.waveform_left_full, &f.waveform_right_full);
            self.renderer.set_freq_spectrum(&f.freq_spectrum);
        }

        let view = frame.texture.create_view(&Default::default());
        self.renderer.render(&view);
        frame.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        // Title reflects whether we boot into a preset or the idle empty-state.
        let title = match self.initial_path.as_deref() {
            Some(p) => {
                let name = std::path::Path::new(p)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.to_string());
                format!("{APP_NAME} — {name}")
            }
            None => IDLE_TITLE.to_string(),
        };
        let attrs = Window::default_attributes()
            .with_title(&title)
            .with_inner_size(PhysicalSize::new(1280u32, 720u32));
        let win = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let state = GpuState::new(win.clone(), self.initial_path.as_deref());
        self.window = Some(win);
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.physical_key
                    == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) =>
            {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(s) = &mut self.state {
                    s.resize(size);
                }
            }
            // Drag-and-drop: a hovered file previews intent in the title bar; the
            // actual drop loads it. Loading is crash-safe (see GpuState::load_path).
            WindowEvent::HoveredFile(path) => {
                if let Some(w) = &self.window {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    w.set_title(&format!("{APP_NAME} — release to load {name}"));
                }
            }
            WindowEvent::HoveredFileCancelled => {
                if let Some(w) = &self.window {
                    let loaded = self.state.is_some();
                    w.set_title(if loaded { APP_NAME } else { IDLE_TITLE });
                }
            }
            WindowEvent::DroppedFile(path) => {
                if let (Some(s), Some(w)) = (&mut self.state, &self.window) {
                    let title = s.load_path(&path.to_string_lossy());
                    w.set_title(&title);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(s) = &mut self.state {
                    s.render();
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

/// Credits + license summary, surfaced in-app via `--about` and a one-line
/// startup banner. OjoDrop stands on MilkDrop / Butterchurn and the open shader
/// toolchain — keep the thank-you visible.
fn print_about() {
    println!(
        "\
OjoDrop — a native Rust + wgpu MilkDrop / Butterchurn preset player.

Built on the work of:
  • Ryan Geiss        — MilkDrop, the visualizer this engine reimplements
  • Jordan 'jberg' Berg — Butterchurn + milkdrop-shader-converter (HLSL→GLSL)
  • Nullsoft / Winamp  — MilkDrop's home and preset ecosystem
  • hlsl2glslfork / glsl-optimizer / Mesa / MojoShader authors

License: MIT (see LICENSE). Bundled converter components keep their own
permissive licenses (BSD-3 / zlib / MIT) — see THIRD_PARTY_NOTICES.md.
Native .milk converter available: {}",
        if particle_milkdrop::native_converter_available() {
            "yes"
        } else {
            "no (JSON-only)"
        }
    );
}

fn main() {
    // Service the isolated, one-shot converter mode before logging, argument
    // parsing, windowing, audio, or any other application threads are started.
    // Normal OjoDrop launches then register this same executable as their
    // helper, keeping app packaging self-contained without loading untrusted
    // shader text into the long-lived UI/audio process.
    #[cfg(feature = "milk-native-converter")]
    {
        if let Some(code) = particle_milkdrop_converter_sys::current_executable_helper_mode() {
            std::process::exit(code);
        }
        let _ = particle_milkdrop_converter_sys::register_current_executable_as_helper();
    }

    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--about" || a == "--credits") {
        print_about();
        return;
    }

    // Detect --anim FRAMES OUT_DIR  (dumps a PNG sequence at fixed 30fps)
    if let Some(pos) = args.iter().position(|a| a == "--anim") {
        let frames: u32 = args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(90);
        let out_dir = args.get(pos + 2).map(|s| s.as_str()).unwrap_or("frames");
        let Some(milk_path) = args
            .iter()
            .find(|a| a.ends_with(".milk") || a.ends_with(".json"))
            .map(|s| s.as_str())
        else {
            eprintln!("--anim requires a .milk or .json preset path");
            std::process::exit(2);
        };
        run_anim(milk_path, frames, out_dir);
        return;
    }

    // Detect --headless FRAMES OUTPUT.png  [--synth-audio]
    if let Some(pos) = args.iter().position(|a| a == "--headless") {
        let frames: u32 = args
            .get(pos + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        let out = args
            .get(pos + 2)
            .map(|s| s.as_str())
            .unwrap_or("milkdrop.png");
        // Opt-in manufactured beat audio (120 BPM) so audio-reactive presets bloom,
        // matching the Butterchurn oracle's identical synth model for a fair compare.
        let synth = args.iter().any(|a| a == "--synth-audio");
        let Some(milk_path) = args
            .iter()
            .find(|a| a.ends_with(".milk") || a.ends_with(".json"))
            .map(|s| s.as_str())
        else {
            eprintln!("--headless requires a .milk or .json preset path");
            std::process::exit(2);
        };
        run_headless(milk_path, frames, out, synth);
        return;
    }

    // Windowed: an optional file arg boots straight into a preset; otherwise the
    // app opens in the idle "drop a file" empty-state. Presets then arrive by
    // drag-and-drop (WindowEvent::DroppedFile).
    let initial_path = args
        .get(1)
        .filter(|a| a.ends_with(".milk") || a.ends_with(".json"))
        .cloned();

    println!("OjoDrop — MilkDrop/Butterchurn player. Credits: run with --about.");
    match &initial_path {
        Some(p) => println!("Loading: {p}"),
        None => println!("{IDLE_TITLE}"),
    }
    if !particle_milkdrop::native_converter_available() {
        println!(
            "(note: native .milk converter helper unavailable — .json presets load fully; \
             raw .milk may render degraded)"
        );
    }
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App {
        initial_path,
        window: None,
        state: None,
    };
    event_loop.run_app(&mut app).expect("run");
}

#[cfg(test)]
mod reconnect_policy_tests {
    use super::*;
    use std::time::Duration;

    use particle_audio::reconnect::{
        audio_reconnect_delay, should_try_audio_reconnect, AUDIO_RECONNECT_BASE_DELAY,
        AUDIO_RECONNECT_MAX_DELAY, AUDIO_RECONNECT_SETTLE,
    };

    /// OjoDrop asks for the plain default input, so the crate's device-fallback
    /// rule has no alternate and every decision must come back with this config.
    const OJODROP_CFG: CaptureConfig = CaptureConfig {
        prefer_loopback: false,
    };

    /// The 2-argument `poll` these tests were written against is now 3-argument.
    /// This shim keeps the replay bodies unchanged AND asserts the extra fact the
    /// new signature buys: OjoDrop's request survives the fallback rule intact.
    fn poll(state: &mut ReconnectState, live: bool, now: Duration) -> CaptureDecision {
        let decision = state.poll(live, now, OJODROP_CFG);
        if let CaptureDecision::Reconnect(cfg) = decision {
            assert_eq!(
                cfg, OJODROP_CFG,
                "OjoDrop never asks for loopback, so no fallback may be substituted"
            );
        }
        decision
    }

    #[test]
    fn first_attempt_after_a_loss_is_immediate() {
        // A freshly noticed loss must not wait: attempt 0 fires on the same frame.
        assert!(should_try_audio_reconnect(0, Duration::ZERO));
    }

    #[test]
    fn second_attempt_waits_the_base_delay() {
        assert!(!should_try_audio_reconnect(1, Duration::from_millis(1_999)));
        assert!(should_try_audio_reconnect(1, Duration::from_secs(2)));
    }

    #[test]
    fn delay_doubles_per_consecutive_failure() {
        assert_eq!(audio_reconnect_delay(0), Duration::ZERO);
        assert_eq!(audio_reconnect_delay(1), Duration::from_secs(2));
        assert_eq!(audio_reconnect_delay(2), Duration::from_secs(4));
        assert_eq!(audio_reconnect_delay(3), Duration::from_secs(8));
        assert_eq!(audio_reconnect_delay(4), Duration::from_secs(16));
    }

    #[test]
    fn delay_is_capped_and_never_overflows() {
        // 2 s << 4 would be 32 s, past the cap; everything above must clamp,
        // and a pathological attempt count must not panic or wrap.
        assert_eq!(audio_reconnect_delay(5), AUDIO_RECONNECT_MAX_DELAY);
        assert_eq!(audio_reconnect_delay(64), AUDIO_RECONNECT_MAX_DELAY);
        assert_eq!(audio_reconnect_delay(u32::MAX), AUDIO_RECONNECT_MAX_DELAY);
    }

    #[test]
    fn the_delay_boundary_is_inclusive() {
        // `should_try` must agree with `audio_reconnect_delay` exactly at the
        // boundary, or a frame landing precisely on it stalls a whole period.
        for attempts in [0u32, 1, 2, 3, 4, 5, 99] {
            let delay = audio_reconnect_delay(attempts);
            assert!(
                should_try_audio_reconnect(attempts, delay),
                "attempt {attempts} should fire at exactly its own delay {delay:?}"
            );
        }
    }

    #[test]
    fn a_capped_wait_still_eventually_fires() {
        assert!(!should_try_audio_reconnect(9, Duration::from_secs(29)));
        assert!(should_try_audio_reconnect(9, Duration::from_secs(30)));
    }

    /// Replay `frames` frames on a synthetic monotonic clock, driving the state
    /// machine exactly as `ensure_audio_capture_running` does. `live_at` decides
    /// whether capture is *observed* live on each frame. Returns how many
    /// reconnect attempts the policy authorised.
    fn replay(
        state: &mut ReconnectState,
        frames: u32,
        frame_dt: Duration,
        live_at: impl Fn(u32) -> bool,
    ) -> u32 {
        let mut attempts = 0;
        for frame in 0..frames {
            let now = frame_dt * (frame + 1);
            if poll(state, live_at(frame), now) == CaptureDecision::Reconnect(OJODROP_CFG) {
                attempts += 1;
            }
        }
        attempts
    }

    #[test]
    fn a_flapping_device_escalates_instead_of_reopening_every_frame() {
        // The device opens cleanly every time and its worker dies before the
        // next frame, so `live` is false on every poll while every attempt
        // "succeeds". Counting construction success as recovery pinned the
        // count at 0, which pinned the delay at ZERO, which reopened the cpal
        // stream ~60x/second for the rest of the set.
        let mut state = ReconnectState::default();
        let attempts = replay(&mut state, 600, Duration::from_millis(16), |_| false);
        assert!(
            attempts <= 5,
            "flapping device authorised {attempts} reopens in ~9.6 s of frames"
        );
        assert!(
            state.delay() >= AUDIO_RECONNECT_BASE_DELAY,
            "a flapping device must escalate past the base delay, got {:?}",
            state.delay()
        );
    }

    #[test]
    fn a_one_frame_liveness_blip_does_not_reset_the_ladder() {
        // The race in `AudioEngine`: `running` is initialised true at
        // construction (particle-audio/src/lib.rs:428) and only cleared later
        // from the cpal error callback (capture.rs:627). A device erroring
        // within ~5-20 ms therefore reports live for exactly one ~16.7 ms
        // frame between deaths. A reset keyed on a single live poll treats that
        // blip as recovery and puts the ladder back at zero every other frame,
        // which is the pre-fix reopen rate all over again.
        let mut state = ReconnectState::default();
        let attempts = replay(&mut state, 600, Duration::from_millis(16), |frame| {
            frame % 2 == 1
        });
        assert!(
            attempts <= 5,
            "one-frame liveness blip authorised {attempts} reopens in ~9.6 s of frames"
        );
        assert!(
            state.delay() >= AUDIO_RECONNECT_BASE_DELAY,
            "a blipping device must escalate past the base delay, got {:?}",
            state.delay()
        );
    }

    #[test]
    fn only_sustained_liveness_resets_the_counter() {
        let mut state = ReconnectState::default();
        // Three attempts, none of them ever followed by sustained liveness.
        assert_eq!(
            poll(&mut state, false, Duration::ZERO),
            CaptureDecision::Reconnect(OJODROP_CFG)
        );
        assert_eq!(state.delay(), Duration::from_secs(2));
        assert_eq!(
            poll(&mut state, false, Duration::from_secs(2)),
            CaptureDecision::Reconnect(OJODROP_CFG)
        );
        assert_eq!(state.delay(), Duration::from_secs(4));
        assert_eq!(
            poll(&mut state, false, Duration::from_secs(6)),
            CaptureDecision::Reconnect(OJODROP_CFG)
        );
        assert_eq!(state.delay(), Duration::from_secs(8));
        // A live frame alone is NOT recovery — the ladder must hold.
        assert_eq!(
            poll(&mut state, true, Duration::from_millis(6_016)),
            CaptureDecision::Idle
        );
        assert_eq!(
            state.delay(),
            Duration::from_secs(8),
            "a single live frame must not clear the ladder"
        );
        // Liveness held for the settle window is. The run began at 6.016 s, so
        // the window closes at 6.016 s + SETTLE — not 6 s + SETTLE.
        assert_eq!(
            poll(
                &mut state,
                true,
                Duration::from_millis(6_016) + AUDIO_RECONNECT_SETTLE
            ),
            CaptureDecision::Idle
        );
        assert_eq!(state.delay(), Duration::ZERO);
    }

    #[test]
    fn frames_inside_the_backoff_window_are_waits_not_attempts() {
        let mut state = ReconnectState::default();
        assert_eq!(
            poll(&mut state, false, Duration::ZERO),
            CaptureDecision::Reconnect(OJODROP_CFG)
        );
        for ms in [16u64, 500, 1_000, 1_999] {
            assert_eq!(
                poll(&mut state, false, Duration::from_millis(ms)),
                CaptureDecision::Wait,
                "{ms} ms into a 2 s backoff must be a wait"
            );
        }
        assert_eq!(
            state.delay(),
            AUDIO_RECONNECT_BASE_DELAY,
            "waiting must not escalate the delay"
        );
    }

    #[test]
    fn a_genuine_recovery_still_returns_to_zero_backoff() {
        // The legitimate case the hardening must not regress: capture dies, one
        // reconnect is attempted, and the rebuilt engine then stays up. Once it
        // has held for the settle window the ladder clears, and a later,
        // unrelated loss is treated as fresh (immediate retry).
        let mut state = ReconnectState::default();
        assert_eq!(
            poll(&mut state, false, Duration::ZERO),
            CaptureDecision::Reconnect(OJODROP_CFG)
        );
        // Continuously-live 60 Hz frames, derived from AUDIO_RECONNECT_SETTLE
        // rather than hardcoded, so raising the base delay (which the settle
        // window is defined as) cannot silently leave this guard holding for
        // less than the window it is meant to clear. `replay` polls frame `i`
        // at `FRAME * (i + 1)`, so the settle clock starts one frame in:
        // ceil(SETTLE / FRAME) + 1 is the minimum, plus a little margin.
        const FRAME: Duration = Duration::from_millis(16);
        const MARGIN_FRAMES: u32 = 4;
        let settle_frames = AUDIO_RECONNECT_SETTLE
            .as_millis()
            .div_ceil(FRAME.as_millis()) as u32
            + 1
            + MARGIN_FRAMES;
        let held = replay(&mut state, settle_frames, FRAME, |_| true);
        assert_eq!(held, 0, "a live device must never authorise a reconnect");
        assert_eq!(
            state.delay(),
            Duration::ZERO,
            "sustained liveness must clear the ladder"
        );
        assert_eq!(
            poll(&mut state, false, Duration::from_secs(600)),
            CaptureDecision::Reconnect(OJODROP_CFG),
            "a later, unrelated loss must retry immediately again"
        );
    }
}
