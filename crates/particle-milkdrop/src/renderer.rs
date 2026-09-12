#![allow(dead_code)]
use rayon::prelude::*;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

use crate::enhanced_audio::{
    EnhancedAudioConfig, EnhancedAudioProcessor, ENHANCED_FFT_BINS, ENHANCED_WAVE_SAMPLES,
    LEGACY_ASSUMED_SAMPLE_RATE_HZ,
};
use crate::equations::{EelProgram, EelRng, EelState, Env, EnvSlot, EnvSnapshot, MegaBuf};
use crate::extended_waveforms::{build_extended_waveform, ExtendedWaveformInput};
use crate::named_textures::{
    NamedTexturePlan, NamedTextureResolver, DEFAULT_NAMED_TEXTURE_LAYER_SIZE,
};
use crate::parse_milk::{CustomWaveDef, MilkShaders, ShapeBaseVals};
use crate::preprocess::{
    custom_sampler_names, fix_glsl_vector_types, glsl_milk_body_to_naga_with_named_textures,
    glsl_milk_warp_body_to_naga_with_named_textures, hlsl_milk_body_to_naga_with_named_textures,
    hlsl_milk_warp_body_to_naga_with_named_textures, normalize_milkdrop_sampler_variants,
    uses_enhanced_audio_helpers, MILKDROP_SAMPLERS,
};

// ── Warp mesh constants ──────────────────────────────────────────────────────

const GRID_W: u32 = 48;
const GRID_H: u32 = 36;
const COMP_GRID_W: u32 = 32;
const COMP_GRID_H: u32 = 24;
const GPU_TIME_WRAP_SECONDS: f64 = 65_536.0;
const GPU_FRAME_WRAP: u64 = 1 << 24;

/// Quiet period before an interactive window resize commits a new set of
/// MilkDrop feedback/blur targets. A resize reallocates several textures, so
/// applying every drag event would turn a live window drag into a GPU-allocation
/// storm. 150 ms keeps the final image responsive while coalescing the normal
/// stream of platform resize events into one state-preserving resize.
pub const INTERACTIVE_RESIZE_DEBOUNCE: Duration = Duration::from_millis(150);

/// Selects the feedback provenance used by an individual MilkDrop renderer.
///
/// [`Self::Legacy`] preserves OjoDrop's established ordering: warp the prior
/// feedback, build blur from that warped image, then composite motion vectors
/// and overlays. [`Self::Beatdrop`] is an explicit compatibility experiment:
/// motion vectors are first written into the prior feedback page, blur is built
/// from that page, and warp subsequently samples the same page. The default is
/// deliberately legacy so loading an existing preset cannot change its output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FeedbackProvenance {
    #[default]
    Legacy,
    Beatdrop,
}

/// Exact capability result for OjoDrop's native shared-feedback transition.
///
/// The bounded implementation blends evaluated mesh UVs before a single shared
/// feedback sample. It also supports a useful subset of shaderless overlays:
/// untextured, borderless custom shapes plus built-in/custom waves are emitted
/// from both advancing states at complementary opacity. It does *not* claim
/// parity for arbitrary custom warp/comp shader pairs, textured/dynamic shapes,
/// motion vectors, darken-center, or frame borders. Callers must retain their
/// ordinary texture transition as the fallback for every other result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedFeedbackSupport {
    /// Supported bounded path: one shared feedback history and two advancing
    /// renderer states. It is restricted to feedback-only presets. Their
    /// built-in comp uniforms are interpolated before the final comp pass.
    /// Per-pixel warp equations may run in both states; per-frame equations are
    /// admitted only when they never reference a comp or overlay control that
    /// this first path does not blend.
    FeedbackOnlyInterpolatedComp,
    /// Supported shaderless path with untextured, borderless shapes and/or
    /// built-in/custom waves. The outgoing geometry is weighted by `1-progress`
    /// and the target's independently evaluated geometry by `progress` in the
    /// common post-warp overlay pass; built-in COMP uniforms are interpolated.
    UntexturedOverlaysInterpolatedComp,
    /// Both arguments identify the same renderer, so a two-state transition is
    /// meaningless and would violate the bounded-state contract.
    SameRenderer,
    /// Feedback texture copies require the same underlying wgpu device.
    /// Renderer workers may wrap `device.clone()` in separate `Arc`s; those are
    /// accepted through `wgpu::Device` equality. Separately created device
    /// instances (even on the same adapter) cannot copy each other's textures.
    DifferentDevice,
    /// The renderers do not target the same host output format.
    DifferentSurfaceFormat,
    /// The renderers have different output or internal feedback dimensions.
    DifferentDimensions,
    /// Custom warp or comp shaders are deliberately not claimed by v1.
    CustomShadersUnsupported,
    /// A textured or dynamically-programmed shape, shape border, motion vector,
    /// darken-center, or frame border is present. These require a separate
    /// overlay target or specialised ordering and must use the caller fallback.
    VisibleOverlaysUnsupported,
    /// Built-in COMP contains a non-interpolable control: an enabled hue shader,
    /// mismatched echo orientation, or mismatched post-FX flag. These branch in
    /// the COMP shader, so blending their raw f32 UBO words would create a
    /// midpoint discontinuity at promotion.
    DiscreteCompUnsupported,
    /// Per-frame equations can modify composition/overlay values after a
    /// capability check. This first path permits only per-frame programs that
    /// reference warp controls and non-visual state.
    PerFrameCompOrOverlayUnsupported,
}

impl SharedFeedbackSupport {
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            Self::FeedbackOnlyInterpolatedComp | Self::UntexturedOverlaysInterpolatedComp
        )
    }
}

/// Axis-aligned clip-space bounds of emitted custom geometry for one OjoDrop
/// frame. This is populated only when geometry diagnostics are explicitly
/// enabled on [`MilkdropRenderer`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MilkdropGeometryBounds {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

/// Compact alpha evidence for emitted vertices or draw records.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MilkdropAlphaSummary {
    pub sample_count: u32,
    pub min: f32,
    pub mean: f32,
    pub max: f32,
}

/// Compact RGB evidence for emitted vertices or draw records. `visible_fraction`
/// counts samples with at least one positive channel (the portion that survives
/// an unorm render target), while `mean_abs_energy` also exposes signed-color
/// programs whose samples are currently negative.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MilkdropRgbSummary {
    pub sample_count: u32,
    pub min: [f32; 3],
    pub mean: [f32; 3],
    pub max: [f32; 3],
    pub visible_fraction: f32,
    pub mean_abs_energy: f32,
}

/// Latest custom-shape geometry emitted by an OjoDrop frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MilkdropShapeGeometryDiagnostics {
    pub enabled_pools: u32,
    pub fill_draws: u32,
    pub border_draws: u32,
    pub fill_vertices: u32,
    pub border_vertices: u32,
    pub bounds: Option<MilkdropGeometryBounds>,
    pub fill_alpha: Option<MilkdropAlphaSummary>,
    pub border_alpha: Option<MilkdropAlphaSummary>,
    pub fill_rgb: Option<MilkdropRgbSummary>,
    pub border_rgb: Option<MilkdropRgbSummary>,
}

/// Latest custom-wave geometry emitted by an OjoDrop frame. Built-in waveform
/// geometry is deliberately excluded so a blank custom-wave pool is visible.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MilkdropWaveGeometryDiagnostics {
    pub enabled_pools: u32,
    pub draws: u32,
    pub vertices: u32,
    pub bounds: Option<MilkdropGeometryBounds>,
    pub alpha: Option<MilkdropAlphaSummary>,
    pub rgb: Option<MilkdropRgbSummary>,
}

/// Opt-in, CPU-side evidence describing the latest OjoDrop custom geometry.
/// Normal rendering does not collect or retain this snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MilkdropGeometryDiagnostics {
    pub frame_index: u64,
    pub custom_shapes: MilkdropShapeGeometryDiagnostics,
    pub custom_waves: MilkdropWaveGeometryDiagnostics,
    pub post_warp_rgb: Option<MilkdropRgbSummary>,
    pub post_overlays_rgb: Option<MilkdropRgbSummary>,
    pub post_comp_rgb: Option<MilkdropRgbSummary>,
}

/// Opt-in, tightly packed RGBA8 snapshots of the three fidelity checkpoints.
/// These are exposed only while geometry diagnostics are enabled.
#[derive(Clone, Debug, PartialEq)]
pub struct MilkdropStageImages {
    pub width: u32,
    pub height: u32,
    pub post_warp_rgba: Vec<u8>,
    pub post_overlays_rgba: Vec<u8>,
    pub post_comp_rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct PendingMilkdropResize {
    width: u32,
    height: u32,
    requested_at: Instant,
}

/// Coalesces interactive resize notifications for a [`MilkdropRenderer`].
///
/// Call [`Self::request`] for each platform resize event, then call
/// [`Self::take_ready`] from the render loop. The caller applies a returned
/// size through [`MilkdropRenderer::try_resize`], which preserves shaders, EEL
/// state, frame counters, audio, and feedback history. Duplicate dimensions do
/// not restart the quiet period.
#[derive(Debug, Default)]
pub struct MilkdropResizeDebouncer {
    pending: Option<PendingMilkdropResize>,
}

impl MilkdropResizeDebouncer {
    /// Queue the latest requested output size. Returns true when this replaces
    /// the previously pending size; duplicate events are deliberately ignored.
    pub fn request(&mut self, width: u32, height: u32, now: Instant) -> bool {
        let width = width.max(1);
        let height = height.max(1);
        if self
            .pending
            .is_some_and(|pending| pending.width == width && pending.height == height)
        {
            return false;
        }
        self.pending = Some(PendingMilkdropResize {
            width,
            height,
            requested_at: now,
        });
        true
    }

    /// Return the most recent requested size only after the resize stream has
    /// been quiet for [`INTERACTIVE_RESIZE_DEBOUNCE`].
    pub fn take_ready(&mut self, now: Instant) -> Option<(u32, u32)> {
        let pending = self.pending?;
        if now
            .checked_duration_since(pending.requested_at)
            .is_none_or(|elapsed| elapsed < INTERACTIVE_RESIZE_DEBOUNCE)
        {
            return None;
        }
        self.pending
            .take()
            .map(|pending| (pending.width, pending.height))
    }

    /// Drop a queued resize when a caller creates a fresh renderer at the
    /// current dimensions (for example, after an intentional preset change).
    pub fn clear(&mut self) {
        self.pending = None;
    }

    /// Whether a resize is still waiting for its quiet period.
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }
}

fn deterministic_time_seconds(frame_idx: u64, time_per_frame: Option<f64>) -> Option<f64> {
    time_per_frame.map(|dt| {
        if dt.is_finite() && dt > 0.0 {
            frame_idx as f64 * dt
        } else {
            0.0
        }
    })
}

fn effective_fps(time_per_frame: Option<f64>) -> f64 {
    time_per_frame
        .filter(|dt| dt.is_finite() && *dt > 0.0)
        .map(|dt| 1.0 / dt)
        .unwrap_or(60.0)
}

fn shader_time_seconds(time_seconds: f64) -> f32 {
    if time_seconds.is_finite() {
        time_seconds.rem_euclid(GPU_TIME_WRAP_SECONDS) as f32
    } else {
        0.0
    }
}

fn shader_frame_index(frame_idx: u64) -> f32 {
    (frame_idx % GPU_FRAME_WRAP) as f32
}

fn shader_progress(time_seconds: f64) -> f32 {
    if time_seconds.is_finite() {
        (time_seconds.rem_euclid(30.0) / 30.0) as f32
    } else {
        0.0
    }
}

fn finite_clamp(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

/// Public-ingress floor for a MilkDrop band level (`bass`/`mid`/`treb`/`vol`).
/// Non-finite becomes `0.0` (silence); negatives clamp up to it. See
/// [`MilkdropRenderer::set_audio`] for why this is a floor and not a policy.
fn audio_level_floor(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Public-ingress floor for a MilkDrop attenuation rail (`*_att`). These are
/// DIVISORS in preset equations, so zero is as dangerous as `NaN`: anything not
/// finite-and-strictly-positive becomes `1.0`, the documented average-energy
/// baseline. See [`MilkdropRenderer::set_audio_att`].
fn audio_att_floor(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        1.0
    }
}

fn finite_f32_from_f64(value: f64, default: f64) -> f32 {
    let narrowed = value as f32;
    if narrowed.is_finite() {
        return narrowed;
    }
    let fallback = default as f32;
    if fallback.is_finite() {
        fallback
    } else {
        0.0
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WarpVert {
    pos: [f32; 2],   // NDC screen position
    uv: [f32; 2],    // warped UV (sample coord) into the previous frame [0,1], DirectX-UV
    decay: [f32; 4], // per-vertex decay rgb (a unused = 1.0)
}
const _: () = assert!(std::mem::size_of::<WarpVert>() == 32);

/// Butterchurn's composition pass is a 32x24 mesh. Its color is the smoothly
/// interpolated `hue_shader` field exposed to authored comp shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompVert {
    pos: [f32; 2],
    color: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<CompVert>() == 24);

// Per-frame warp base values (from MilkShaders), overridable by the per-frame EEL.
#[derive(Copy, Clone)]
struct WarpBase {
    zoom: f32,
    zoomexp: f32,
    rot: f32,
    warp: f32,
    cx: f32,
    cy: f32,
    dx: f32,
    dy: f32,
    sx: f32,
    sy: f32,
    warpscale: f32,
    warpanimspeed: f32,
    decay: f32,
    wrap: bool,
}

/// Allocation-free live scalar snapshot. Updating this state preserves the
/// renderer, equation programs, q/user variables, megabufs, and feedback chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MilkBaseVals {
    pub decay: f32,
    pub gamma_adj: f32,
    pub fshader: f32,
    pub echo_zoom: f32,
    pub echo_alpha: f32,
    pub echo_orient: f32,
    pub brighten: bool,
    pub darken: bool,
    pub solarize: bool,
    pub invert: bool,
    pub warpscale: f32,
    pub warpanimspeed: f32,
    pub zoom: f32,
    pub zoomexp: f32,
    pub rot: f32,
    pub warp_amount: f32,
    pub cx: f32,
    pub cy: f32,
    pub dx: f32,
    pub dy: f32,
    pub sx: f32,
    pub sy: f32,
    pub wrap: bool,
    pub wave_mode: f32,
    pub wave_x: f32,
    pub wave_y: f32,
    pub wave_r: f32,
    pub wave_g: f32,
    pub wave_b: f32,
    pub wave_a: f32,
    pub wave_mystery: f32,
    pub wave_scale: f32,
    pub wave_smoothing: f32,
    pub wave_dots: bool,
    pub wave_thick: bool,
    pub additive_wave: bool,
    pub wave_brighten: bool,
    pub modwavealphabyvolume: bool,
    pub modwavealphastart: f32,
    pub modwavealphaend: f32,
    pub mv_on: bool,
    pub mv_x: f32,
    pub mv_y: f32,
    pub mv_dx: f32,
    pub mv_dy: f32,
    pub mv_l: f32,
    pub mv_r: f32,
    pub mv_g: f32,
    pub mv_b: f32,
    pub mv_a: f32,
    pub ob_size: f32,
    pub ob_r: f32,
    pub ob_g: f32,
    pub ob_b: f32,
    pub ob_a: f32,
    pub ib_size: f32,
    pub ib_r: f32,
    pub ib_g: f32,
    pub ib_b: f32,
    pub ib_a: f32,
    pub darken_center: bool,
}

impl Default for MilkBaseVals {
    fn default() -> Self {
        Self {
            decay: 0.98,
            gamma_adj: 2.0,
            fshader: 0.0,
            echo_zoom: 2.0,
            echo_alpha: 0.0,
            echo_orient: 0.0,
            brighten: false,
            darken: false,
            solarize: false,
            invert: false,
            warpscale: 1.0,
            warpanimspeed: 1.0,
            zoom: 1.0,
            zoomexp: 1.0,
            rot: 0.0,
            warp_amount: 1.0,
            cx: 0.5,
            cy: 0.5,
            dx: 0.0,
            dy: 0.0,
            sx: 1.0,
            sy: 1.0,
            wrap: true,
            wave_mode: 0.0,
            wave_x: 0.5,
            wave_y: 0.5,
            wave_r: 1.0,
            wave_g: 1.0,
            wave_b: 1.0,
            wave_a: 1.0,
            wave_mystery: 0.0,
            wave_scale: 1.0,
            wave_smoothing: 0.75,
            wave_dots: false,
            wave_thick: false,
            additive_wave: false,
            wave_brighten: true,
            modwavealphabyvolume: false,
            modwavealphastart: 0.75,
            modwavealphaend: 0.95,
            mv_on: true,
            mv_x: 12.0,
            mv_y: 9.0,
            mv_dx: 0.0,
            mv_dy: 0.0,
            mv_l: 0.9,
            mv_r: 1.0,
            mv_g: 1.0,
            mv_b: 1.0,
            mv_a: 1.0,
            ob_size: 0.01,
            ob_r: 0.0,
            ob_g: 0.0,
            ob_b: 0.0,
            ob_a: 0.0,
            ib_size: 0.01,
            ib_r: 0.25,
            ib_g: 0.25,
            ib_b: 0.25,
            ib_a: 0.0,
            darken_center: false,
        }
    }
}

/// Per-frame default-warp parameters consumed by the vertex shader. The final
/// vector carries a CPU-mesh flag so presets with per-pixel EEL (or enabled
/// motion vectors, which sample the CPU flow field) retain the exact legacy path.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WarpGpuParams {
    transform0: [f32; 4], // zoom, zoomexp, rot, warp
    transform1: [f32; 4], // cx, cy, dx, dy
    transform2: [f32; 4], // sx, sy, decay, warpscale
    transform3: [f32; 4], // warpanimspeed, time, aspectx, aspecty
    flags: [f32; 4],      // use_cpu_mesh, reserved...
}
const _: () = assert!(std::mem::size_of::<WarpGpuParams>() == 80);

/// Pre-interned per-pixel variable slots. `reset` is intentionally limited to
/// MilkDrop's ten authored warp controls; custom temporaries carry between mesh
/// vertices. Inputs and OjoDrop's decay extension are overwritten directly.
#[derive(Clone, Copy)]
struct WarpEnvSlots {
    reset: [EnvSlot; 10],
    x: EnvSlot,
    y: EnvSlot,
    rad: EnvSlot,
    ang: EnvSlot,
    decay: EnvSlot,
    decay_r: EnvSlot,
    decay_g: EnvSlot,
    decay_b: EnvSlot,
}

impl WarpEnvSlots {
    fn intern(env: &mut Env) -> Self {
        Self {
            reset: [
                env.intern_slot("warp"),
                env.intern_slot("zoom"),
                env.intern_slot("zoomexp"),
                env.intern_slot("cx"),
                env.intern_slot("cy"),
                env.intern_slot("sx"),
                env.intern_slot("sy"),
                env.intern_slot("dx"),
                env.intern_slot("dy"),
                env.intern_slot("rot"),
            ],
            x: env.intern_slot("x"),
            y: env.intern_slot("y"),
            rad: env.intern_slot("rad"),
            ang: env.intern_slot("ang"),
            decay: env.intern_slot("decay"),
            decay_r: env.intern_slot("decay_r"),
            decay_g: env.intern_slot("decay_g"),
            decay_b: env.intern_slot("decay_b"),
        }
    }
}

// ── Custom-shape vertex (interleaved pos/color/uv) ───────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeVert {
    pos: [f32; 2],
    color: [f32; 4],
    uv: [f32; 2],
}
const _: () = assert!(std::mem::size_of::<ShapeVert>() == 32);

// ── Border vertex (pos only; color via uniform) ──────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BorderVert {
    pos: [f32; 2],
}

// ── BorderU uniform (color + thick offset) ───────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BorderU {
    color: [f32; 4],
    offset: [f32; 4],
}

// ── Waveform vertex (pos + color) ────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveVert {
    pos: [f32; 2],
    color: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<WaveVert>() == 24);

// ── Motion-vector vertex (pos only; color via uniform) ───────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MVVert {
    pos: [f32; 2],
}
// maxX*maxY*2 verts (butterchurn caps the grid at 64x48, 2 verts per arrow).
const MV_VERT_CAP: usize = 64 * 48 * 2;

// ── MV color uniform (vec4) ──────────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MVColor {
    color: [f32; 4],
}

// ── Darken-center vertex (pos + color) ───────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DarkenVert {
    pos: [f32; 2],
    color: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<DarkenVert>() == 24);

const SIDES_MAX: usize = 100;
// Each shape instance contributes (sides+2) fill verts.
const SHAPE_FILL_VERTS_MAX: usize = SIDES_MAX + 2;
const MAX_SHAPE_INSTANCES: usize = 1024;
// Static fan index count = sides*3 for sides<=100 → 300.
const SHAPE_FAN_IDX_MAX: usize = SIDES_MAX * 3;
// Custom-shape fill geometry capacity (verts for ALL shapes×instances of a frame).
// Some cream-of-the-crop presets use 512/1024-instanced shape arrays; the old 8k
// cap skipped most of those fans and left otherwise-live presets nearly black.
const SHAPE_VERT_CAP: usize = 65536;
// Waveform vertex capacity (built-in + custom). Sixteen custom line waves can
// each emit 1023 smoothed vertices; an extended built-in waveform can append
// another 1023 vertices (or two half-size strips). 20×1024 preserves that
// legitimate BeatDrop/OjoDrop combination without stale-tail draws.
const WAVE_VERT_CAP: usize = 20 * 1024;
const MAX_AUDIO_SAMPLES: usize = 8192;
// Border vertex capacity (per-frame across all shapes).
const BORDER_VERT_CAP: usize = 65536;
const BORDER_THICK_LINE_PASSES: usize = 4;
// Dynamic uniform slots for per-border color/thickness offsets. Four slots are
// needed per thick border draw, so this comfortably covers multi-instance shapes.
const BORDER_UNIFORM_SLOTS: usize = 32768;
const WAVE_THICK_LINE_PASSES: usize = 4;
const WAVE_THICK_DOT_PASSES: usize = 9;
/// Pure per-point programs at or above this compiled cost may use the adaptive
/// 256-point quality fallback. Stateful EEL always retains its authored count.
const CUSTOM_WAVE_LOD_OP_THRESHOLD: usize = 96;
const CUSTOM_WAVE_LOD_SAMPLES: usize = 256;

/// Per-render CPU storage. Every buffer is cleared and reused rather than
/// allocated on each frame; capacities are bounded by the corresponding GPU
/// buffers and retained for the lifetime of the renderer.
#[derive(Default)]
struct RendererScratch {
    warp_verts: Vec<WarpVert>,
    comp_verts: Vec<CompVert>,
    motion_verts: Vec<MVVert>,
    darken_verts: Vec<DarkenVert>,
    shape_fill_verts: Vec<ShapeVert>,
    shape_fill_draws: Vec<ShapeFillDraw>,
    shape_border_verts: Vec<BorderVert>,
    shape_border_draws: Vec<BorderDraw>,
    wave_verts: Vec<WaveVert>,
    wave_draws: Vec<WaveDraw>,
    /// Reused weighted copies of the incoming state's compatible overlay
    /// vertices. The target retains its full-opacity CPU geometry so it can be
    /// promoted immediately after an interrupted shared-feedback morph.
    shared_shape_fill_verts: Vec<ShapeVert>,
    shared_wave_verts: Vec<WaveVert>,
    frame_border_verts: Vec<BorderVert>,
    frame_border_draws: Vec<(u32, u32)>,
    border_uniform_bytes: Vec<u8>,
    frame_border_uniform_bytes: Vec<u8>,
    custom_wave_draws: Vec<Option<WaveDraw>>,
    basic_wave: BasicWaveScratch,
}

/// Persistent storage for the built-in waveform path. These vectors used
/// to allocate on every rendered frame, even though their upper bounds are
/// stable for a renderer/audio configuration.
#[derive(Default)]
struct BasicWaveScratch {
    processed_l: Vec<f32>,
    processed_r: Vec<f32>,
    positions: Vec<[f32; 2]>,
    positions2: Vec<[f32; 2]>,
    smoothed: Vec<[f32; 2]>,
}

/// Cached slots for the frame/audio variables repeatedly seeded into shape and
/// custom-wave pools. Keeping these handles beside the pool removes a dozen
/// string hashes from every shape instance and wave frame.
#[derive(Clone, Copy)]
struct FrameEnvSlots {
    time: EnvSlot,
    frame: EnvSlot,
    fps: EnvSlot,
    bass: EnvSlot,
    bass_att: EnvSlot,
    mid: EnvSlot,
    mid_att: EnvSlot,
    treb: EnvSlot,
    treb_att: EnvSlot,
    vol: EnvSlot,
    aspectx: EnvSlot,
    aspecty: EnvSlot,
}

impl FrameEnvSlots {
    fn intern(env: &mut Env) -> Self {
        Self {
            time: env.intern_slot("time"),
            frame: env.intern_slot("frame"),
            fps: env.intern_slot("fps"),
            bass: env.intern_slot("bass"),
            bass_att: env.intern_slot("bass_att"),
            mid: env.intern_slot("mid"),
            mid_att: env.intern_slot("mid_att"),
            treb: env.intern_slot("treb"),
            treb_att: env.intern_slot("treb_att"),
            vol: env.intern_slot("vol"),
            aspectx: env.intern_slot("aspectx"),
            aspecty: env.intern_slot("aspecty"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn seed(
        self,
        env: &mut Env,
        time: f64,
        frame: u64,
        fps: f64,
        bass: f64,
        mid: f64,
        treb: f64,
        vol: f64,
        bass_att: f64,
        mid_att: f64,
        treb_att: f64,
        aspectx: f64,
        aspecty: f64,
    ) {
        env.set_slot_value(self.time, time);
        env.set_slot_value(self.frame, frame as f64);
        env.set_slot_value(self.fps, fps);
        env.set_slot_value(self.bass, bass);
        env.set_slot_value(self.bass_att, bass_att);
        env.set_slot_value(self.mid, mid);
        env.set_slot_value(self.mid_att, mid_att);
        env.set_slot_value(self.treb, treb);
        env.set_slot_value(self.treb_att, treb_att);
        env.set_slot_value(self.vol, vol);
        env.set_slot_value(self.aspectx, aspectx);
        env.set_slot_value(self.aspecty, aspecty);
    }
}

/// All authored shape inputs/outputs restored and read for each instance.
#[derive(Clone, Copy)]
struct ShapeEnvSlots {
    frame: FrameEnvSlots,
    instance: EnvSlot,
    num_inst: EnvSlot,
    sides: EnvSlot,
    rad: EnvSlot,
    ang: EnvSlot,
    x: EnvSlot,
    y: EnvSlot,
    r: EnvSlot,
    g: EnvSlot,
    b: EnvSlot,
    a: EnvSlot,
    r2: EnvSlot,
    g2: EnvSlot,
    b2: EnvSlot,
    a2: EnvSlot,
    border_r: EnvSlot,
    border_g: EnvSlot,
    border_b: EnvSlot,
    border_a: EnvSlot,
    thickoutline: EnvSlot,
    textured: EnvSlot,
    tex_ang: EnvSlot,
    tex_zoom: EnvSlot,
    additive: EnvSlot,
}

impl ShapeEnvSlots {
    fn intern(env: &mut Env) -> Self {
        Self {
            frame: FrameEnvSlots::intern(env),
            instance: env.intern_slot("instance"),
            num_inst: env.intern_slot("num_inst"),
            sides: env.intern_slot("sides"),
            rad: env.intern_slot("rad"),
            ang: env.intern_slot("ang"),
            x: env.intern_slot("x"),
            y: env.intern_slot("y"),
            r: env.intern_slot("r"),
            g: env.intern_slot("g"),
            b: env.intern_slot("b"),
            a: env.intern_slot("a"),
            r2: env.intern_slot("r2"),
            g2: env.intern_slot("g2"),
            b2: env.intern_slot("b2"),
            a2: env.intern_slot("a2"),
            border_r: env.intern_slot("border_r"),
            border_g: env.intern_slot("border_g"),
            border_b: env.intern_slot("border_b"),
            border_a: env.intern_slot("border_a"),
            thickoutline: env.intern_slot("thickoutline"),
            textured: env.intern_slot("textured"),
            tex_ang: env.intern_slot("tex_ang"),
            tex_zoom: env.intern_slot("tex_zoom"),
            additive: env.intern_slot("additive"),
        }
    }
}

/// Custom-wave control and point slots touched for every generated sample.
#[derive(Clone, Copy)]
struct WaveEnvSlots {
    frame: FrameEnvSlots,
    samples: EnvSlot,
    sep: EnvSlot,
    scaling: EnvSlot,
    smoothing: EnvSlot,
    spectrum: EnvSlot,
    sample: EnvSlot,
    value1: EnvSlot,
    value2: EnvSlot,
    x: EnvSlot,
    y: EnvSlot,
    r: EnvSlot,
    g: EnvSlot,
    b: EnvSlot,
    a: EnvSlot,
}

impl WaveEnvSlots {
    fn intern(env: &mut Env) -> Self {
        Self {
            frame: FrameEnvSlots::intern(env),
            samples: env.intern_slot("samples"),
            sep: env.intern_slot("sep"),
            scaling: env.intern_slot("scaling"),
            smoothing: env.intern_slot("smoothing"),
            spectrum: env.intern_slot("spectrum"),
            sample: env.intern_slot("sample"),
            value1: env.intern_slot("value1"),
            value2: env.intern_slot("value2"),
            x: env.intern_slot("x"),
            y: env.intern_slot("y"),
            r: env.intern_slot("r"),
            g: env.intern_slot("g"),
            b: env.intern_slot("b"),
            a: env.intern_slot("a"),
        }
    }
}

// Runtime state for one custom shape (base vals + per-frame program + var pool).
struct ShapeRT {
    base: ShapeBaseVals,
    prog: Option<EelProgram>,
    env: Env,
    /// Cached destinations for the preset-global reg00..reg99 snapshot.
    reg_slots: [EnvSlot; 100],
    q_slots: [EnvSlot; 32],
    t_slots: [EnvSlot; 8],
    /// Only globals statically referenced by this pool's steady-state bytecode.
    /// Init keeps the full slot arrays above because it must thread every reg to
    /// the next authored pool, but per-instance execution can copy selectively.
    live_reg_indices: Vec<u8>,
    live_q_indices: Vec<u8>,
    live_t_indices: Vec<u8>,
    slots: ShapeEnvSlots,
    t_init: [f64; 8],
    /// Per-pool megabuf (private) sharing the preset-wide gmegabuf.
    state: EelState,
}

// Runtime state for one custom waveform.
struct WaveRT {
    def: CustomWaveDef,
    per_frame_prog: Option<EelProgram>,
    per_point_prog: Option<EelProgram>,
    env: Env,
    /// Cached destinations for the preset-global reg00..reg99 snapshot.
    reg_slots: [EnvSlot; 100],
    q_slots: [EnvSlot; 32],
    t_slots: [EnvSlot; 8],
    /// Union of globals referenced by the per-frame and per-point programs.
    live_reg_indices: Vec<u8>,
    live_q_indices: Vec<u8>,
    live_t_indices: Vec<u8>,
    slots: WaveEnvSlots,
    t_init: [f64; 8],
    /// Per-pool megabuf (private) sharing the preset-wide gmegabuf.
    state: EelState,
    /// Persistent CPU storage reused across every frame for this wave.
    scratch: WaveScratch,
}

#[derive(Default)]
struct WaveScratch {
    source_l: Vec<f32>,
    source_r: Vec<f32>,
    points_l: Vec<f32>,
    points_r: Vec<f32>,
    positions: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    output: Vec<WaveVert>,
}

// One fill draw (a shape instance). base_vertex = vertex offset into shape_vert_buf.
struct ShapeFillDraw {
    base_vertex: i32,
    sides: u32, // index count = sides*3
    additive: bool,
    border_draw_index: Option<usize>,
}
// One border source (rim verts already appended to border_vert_buf).
struct BorderDraw {
    start_vert: u32,
    count: u32, // = sides+1
    color: [f32; 4],
    thick: bool,
}
// One waveform draw record.
#[derive(Clone, Copy)]
struct WaveDraw {
    start_vert: u32,
    count: u32,
    points: bool, // PointList vs LineStrip
    additive: bool,
    thick: bool, // 4-pass thick offset expansion
}

#[derive(Clone, Copy, Debug, Default)]
struct CustomWaveGeometryExtent {
    vertices: usize,
    draws: usize,
}

#[derive(Debug, Default)]
struct GeometryDiagnosticCollector {
    enabled: bool,
    latest: Option<MilkdropGeometryDiagnostics>,
}

#[derive(Debug)]
struct GeometryStageReadback {
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    post_warp: wgpu::Buffer,
    post_overlays: wgpu::Buffer,
    post_comp: wgpu::Buffer,
}

impl GeometryStageReadback {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let unpadded_bytes_per_row = width.saturating_mul(4);
        let padded_bytes_per_row = unpadded_bytes_per_row
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .saturating_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let size = u64::from(padded_bytes_per_row).saturating_mul(u64::from(height));
        let make_buffer = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        Self {
            width,
            height,
            padded_bytes_per_row,
            post_warp: make_buffer("milkdrop-diagnostic-post-warp"),
            post_overlays: make_buffer("milkdrop-diagnostic-post-overlays"),
            post_comp: make_buffer("milkdrop-diagnostic-post-comp"),
        }
    }

    fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }

    fn read_summaries(&self, device: &wgpu::Device) -> [Option<MilkdropRgbSummary>; 3] {
        [
            read_stage_rgb_summary(
                device,
                &self.post_warp,
                self.width,
                self.height,
                self.padded_bytes_per_row,
            ),
            read_stage_rgb_summary(
                device,
                &self.post_overlays,
                self.width,
                self.height,
                self.padded_bytes_per_row,
            ),
            read_stage_rgb_summary(
                device,
                &self.post_comp,
                self.width,
                self.height,
                self.padded_bytes_per_row,
            ),
        ]
    }

    fn read_images(&self, device: &wgpu::Device) -> Option<MilkdropStageImages> {
        Some(MilkdropStageImages {
            width: self.width,
            height: self.height,
            post_warp_rgba: read_stage_rgba8(
                device,
                &self.post_warp,
                self.width,
                self.height,
                self.padded_bytes_per_row,
            )?,
            post_overlays_rgba: read_stage_rgba8(
                device,
                &self.post_overlays,
                self.width,
                self.height,
                self.padded_bytes_per_row,
            )?,
            post_comp_rgba: read_stage_rgba8(
                device,
                &self.post_comp,
                self.width,
                self.height,
                self.padded_bytes_per_row,
            )?,
        })
    }
}

impl GeometryDiagnosticCollector {
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.latest = None;
        }
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn latest(&self) -> Option<MilkdropGeometryDiagnostics> {
        self.latest
    }

    fn capture(&mut self, build: impl FnOnce() -> MilkdropGeometryDiagnostics) {
        if self.enabled {
            self.latest = Some(build());
        }
    }
}

fn geometry_bounds<'a>(
    positions: impl Iterator<Item = &'a [f32; 2]>,
) -> Option<MilkdropGeometryBounds> {
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    let mut found = false;
    for position in positions {
        if !position[0].is_finite() || !position[1].is_finite() {
            continue;
        }
        min[0] = min[0].min(position[0]);
        min[1] = min[1].min(position[1]);
        max[0] = max[0].max(position[0]);
        max[1] = max[1].max(position[1]);
        found = true;
    }
    found.then_some(MilkdropGeometryBounds { min, max })
}

fn alpha_summary(values: impl Iterator<Item = f32>) -> Option<MilkdropAlphaSummary> {
    let mut sample_count = 0u32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for value in values {
        if !value.is_finite() {
            continue;
        }
        sample_count = sample_count.saturating_add(1);
        min = min.min(value);
        max = max.max(value);
        sum += f64::from(value);
    }
    (sample_count > 0).then_some(MilkdropAlphaSummary {
        sample_count,
        min,
        mean: (sum / f64::from(sample_count)) as f32,
        max,
    })
}

fn rgb_summary(values: impl Iterator<Item = [f32; 3]>) -> Option<MilkdropRgbSummary> {
    let mut sample_count = 0u32;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut sums = [0.0f64; 3];
    let mut visible_count = 0u32;
    let mut absolute_energy = 0.0f64;
    for value in values {
        if value.iter().any(|channel| !channel.is_finite()) {
            continue;
        }
        sample_count = sample_count.saturating_add(1);
        let mut sample_visible = false;
        let mut sample_energy = 0.0f64;
        for channel in 0..3 {
            min[channel] = min[channel].min(value[channel]);
            max[channel] = max[channel].max(value[channel]);
            sums[channel] += f64::from(value[channel]);
            sample_visible |= value[channel] > 1.0e-6;
            sample_energy += f64::from(value[channel].abs());
        }
        visible_count = visible_count.saturating_add(u32::from(sample_visible));
        absolute_energy += sample_energy / 3.0;
    }
    (sample_count > 0).then_some(MilkdropRgbSummary {
        sample_count,
        min,
        mean: std::array::from_fn(|channel| (sums[channel] / f64::from(sample_count)) as f32),
        max,
        visible_fraction: visible_count as f32 / sample_count as f32,
        mean_abs_energy: (absolute_energy / f64::from(sample_count)) as f32,
    })
}

fn rgba8_rgb_summary(
    bytes: &[u8],
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) -> Option<MilkdropRgbSummary> {
    let active_bytes_per_row = usize::try_from(width).ok()?.checked_mul(4)?;
    let padded_bytes_per_row = usize::try_from(padded_bytes_per_row).ok()?;
    if active_bytes_per_row > padded_bytes_per_row {
        return None;
    }
    let pixels = bytes
        .chunks_exact(padded_bytes_per_row)
        .take(height as usize)
        .flat_map(|row| {
            row[..active_bytes_per_row].chunks_exact(4).map(|rgba| {
                [
                    f32::from(rgba[0]) / 255.0,
                    f32::from(rgba[1]) / 255.0,
                    f32::from(rgba[2]) / 255.0,
                ]
            })
        });
    rgb_summary(pixels)
}

fn read_stage_rgb_summary(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) -> Option<MilkdropRgbSummary> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().ok()?.ok()?;
    let mapped = slice.get_mapped_range();
    let summary = rgba8_rgb_summary(&mapped, width, height, padded_bytes_per_row);
    drop(mapped);
    buffer.unmap();
    summary
}

fn read_stage_rgba8(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) -> Option<Vec<u8>> {
    let active_bytes_per_row = usize::try_from(width).ok()?.checked_mul(4)?;
    let padded_bytes_per_row = usize::try_from(padded_bytes_per_row).ok()?;
    if active_bytes_per_row > padded_bytes_per_row {
        return None;
    }
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().ok()?.ok()?;
    let mapped = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity(active_bytes_per_row.checked_mul(height as usize)?);
    for row in mapped
        .chunks_exact(padded_bytes_per_row)
        .take(height as usize)
    {
        rgba.extend_from_slice(&row[..active_bytes_per_row]);
    }
    drop(mapped);
    buffer.unmap();
    Some(rgba)
}

fn encode_stage_texture_copy(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn summarize_custom_geometry(
    frame_index: u64,
    enabled_shape_pools: usize,
    fill_verts: &[ShapeVert],
    fill_draws: &[ShapeFillDraw],
    border_verts: &[BorderVert],
    border_draws: &[BorderDraw],
    enabled_wave_pools: usize,
    wave_verts: &[WaveVert],
    wave_draws: &[WaveDraw],
) -> MilkdropGeometryDiagnostics {
    MilkdropGeometryDiagnostics {
        frame_index,
        custom_shapes: MilkdropShapeGeometryDiagnostics {
            enabled_pools: enabled_shape_pools.min(u32::MAX as usize) as u32,
            fill_draws: fill_draws.len().min(u32::MAX as usize) as u32,
            border_draws: border_draws.len().min(u32::MAX as usize) as u32,
            fill_vertices: fill_verts.len().min(u32::MAX as usize) as u32,
            border_vertices: border_verts.len().min(u32::MAX as usize) as u32,
            // Border positions duplicate the fill rim, so fill bounds cover both.
            bounds: geometry_bounds(fill_verts.iter().map(|vertex| &vertex.pos)),
            fill_alpha: alpha_summary(fill_verts.iter().map(|vertex| vertex.color[3])),
            border_alpha: alpha_summary(border_draws.iter().map(|draw| draw.color[3])),
            fill_rgb: rgb_summary(
                fill_verts
                    .iter()
                    .map(|vertex| [vertex.color[0], vertex.color[1], vertex.color[2]]),
            ),
            border_rgb: rgb_summary(
                border_draws
                    .iter()
                    .map(|draw| [draw.color[0], draw.color[1], draw.color[2]]),
            ),
        },
        custom_waves: MilkdropWaveGeometryDiagnostics {
            enabled_pools: enabled_wave_pools.min(u32::MAX as usize) as u32,
            draws: wave_draws.len().min(u32::MAX as usize) as u32,
            vertices: wave_verts.len().min(u32::MAX as usize) as u32,
            bounds: geometry_bounds(wave_verts.iter().map(|vertex| &vertex.pos)),
            alpha: alpha_summary(wave_verts.iter().map(|vertex| vertex.color[3])),
            rgb: rgb_summary(
                wave_verts
                    .iter()
                    .map(|vertex| [vertex.color[0], vertex.color[1], vertex.color[2]]),
            ),
        },
        ..MilkdropGeometryDiagnostics::default()
    }
}

fn build_warp_indices() -> Vec<u32> {
    let mut idx = Vec::with_capacity((GRID_W * GRID_H * 6) as usize);
    for j in 0..GRID_H {
        for i in 0..GRID_W {
            let a = j * (GRID_W + 1) + i;
            let b = a + 1;
            let c = a + (GRID_W + 1);
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    idx
}

fn build_static_warp_verts() -> Vec<WarpVert> {
    let mut verts = Vec::with_capacity(((GRID_W + 1) * (GRID_H + 1)) as usize);
    for j in 0..=GRID_H {
        for i in 0..=GRID_W {
            let x = (i as f32 / GRID_W as f32) * 2.0 - 1.0;
            let y = (j as f32 / GRID_H as f32) * 2.0 - 1.0;
            verts.push(WarpVert {
                pos: [x, -y],
                uv: [0.0; 2],
                decay: [0.0; 4],
            });
        }
    }
    verts
}

/// Blend two independently evaluated warp meshes before the GPU samples the
/// shared feedback page. This is deliberately mesh-space blending, not a
/// dissolve of two completed feedback images. Invalid target values retain the
/// already-sanitized outgoing value so a bad incoming EEL program cannot poison
/// the shared history.
fn blend_evaluated_warp_mesh(
    outgoing: &mut [WarpVert],
    incoming: &[WarpVert],
    progress: f32,
) -> bool {
    if outgoing.len() != incoming.len() || !progress.is_finite() {
        return false;
    }
    let weight = progress.clamp(0.0, 1.0);
    for (old, new) in outgoing.iter_mut().zip(incoming) {
        for channel in 0..2 {
            if new.uv[channel].is_finite() {
                old.uv[channel] += (new.uv[channel] - old.uv[channel]) * weight;
            }
        }
        for channel in 0..4 {
            if new.decay[channel].is_finite() {
                old.decay[channel] += (new.decay[channel] - old.decay[channel]) * weight;
            }
        }
    }
    true
}

fn build_comp_indices() -> Vec<u16> {
    let mut indices = Vec::with_capacity((COMP_GRID_W * COMP_GRID_H * 6) as usize);
    let stride = COMP_GRID_W + 1;
    for j in 0..COMP_GRID_H {
        for i in 0..COMP_GRID_W {
            let a = i + stride * j;
            let b = i + stride * (j + 1);
            let c = i + 1 + stride * (j + 1);
            let d = i + 1 + stride * j;
            indices
                .extend_from_slice(&[a as u16, b as u16, d as u16, b as u16, c as u16, d as u16]);
        }
    }
    indices
}

fn generate_comp_verts(time: f32, rand_start: [f32; 4], verts: &mut Vec<CompVert>) {
    let mut hue = [[1.0f32; 3]; 4];
    for (i, corner) in hue.iter_mut().enumerate() {
        corner[0] =
            0.6 + 0.3 * (time * 30.0 * 0.0143 + 3.0 + i as f32 * 21.0 + rand_start[3]).sin();
        corner[1] =
            0.6 + 0.3 * (time * 30.0 * 0.0107 + 1.0 + i as f32 * 13.0 + rand_start[1]).sin();
        corner[2] = 0.6 + 0.3 * (time * 30.0 * 0.0129 + 6.0 + i as f32 * 9.0 + rand_start[2]).sin();
        let max_shade = corner[0].max(corner[1]).max(corner[2]);
        for channel in corner {
            *channel = 0.5 + 0.5 * (*channel / max_shade);
        }
    }

    verts.clear();
    verts.reserve(((COMP_GRID_W + 1) * (COMP_GRID_H + 1)) as usize);
    for j in 0..=COMP_GRID_H {
        let y = j as f32 / COMP_GRID_H as f32;
        for i in 0..=COMP_GRID_W {
            let x = i as f32 / COMP_GRID_W as f32;
            let mut color = [0.0f32; 4];
            for channel in 0..3 {
                color[channel] = hue[0][channel] * x * y
                    + hue[1][channel] * (1.0 - x) * y
                    + hue[2][channel] * x * (1.0 - y)
                    + hue[3][channel] * (1.0 - x) * (1.0 - y);
            }
            color[3] = 1.0;
            verts.push(CompVert {
                pos: [x * 2.0 - 1.0, 1.0 - y * 2.0],
                color,
            });
        }
    }
}

// PerFrame uniform buffer — layout must exactly match the WGSL PerFrame struct
// emitted by naga (17 leading vec4s followed by scalar controls).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PerFrame {
    texsize: [f32; 4],       //   0 — (w, h, 1/w, 1/h)
    aspect: [f32; 4],        //  16 — (aspectx, aspecty, invAspectx, invAspecty)
    slow_roam_cos: [f32; 4], //  32
    roam_cos: [f32; 4],      //  48
    slow_roam_sin: [f32; 4], //  64
    roam_sin: [f32; 4],      //  80
    rand_frame: [f32; 4],    //  96
    rand_start: [f32; 4],    // 112 — built-in hue phase offsets
    rand_preset: [f32; 4],   // 128 — custom shader rand_preset
    _qa: [f32; 4],           // 144 — q1..q4
    _qb: [f32; 4],           // 160 — q5..q8
    _qc: [f32; 4],           // 176
    _qd: [f32; 4],           // 192
    _qe: [f32; 4],           // 208
    _qf: [f32; 4],           // 224
    _qg: [f32; 4],           // 240
    _qh: [f32; 4],           // 256
    time: f32,               // 272
    fps: f32,                // 276
    frame: f32,              // 280
    progress: f32,           // 284
    bass: f32,               // 288
    mid: f32,                // 292
    treb: f32,               // 296
    vol: f32,                // 300
    bass_att: f32,           // 304
    mid_att: f32,            // 308
    treb_att: f32,           // 312
    vol_att: f32,            // 316
    f_shader: f32,           // 320
    gamma_adj: f32,          // 324
    echo_zoom: f32,          // 328
    echo_alpha: f32,         // 332
    echo_orientation: f32,   // 336
    blur1_min: f32,          // 340
    blur1_max: f32,          // 344
    blur2_min: f32,          // 348
    blur2_max: f32,          // 352
    blur3_min: f32,          // 356
    blur3_max: f32,          // 360
    scale1: f32,             // 364
    scale2: f32,             // 368
    scale3: f32,             // 372
    bias1: f32,              // 376
    bias2: f32,              // 380
    bias3: f32,              // 384
    brighten: f32,           // 388 — comp post-FX flags
    darken: f32,             // 392
    solarize: f32,           // 396
    invert: f32,             // 400
    audio_nyquist_hz: f32,  // 404 — enhanced-audio Hz helpers
    _pad: [f32; 2],          // 408 → pad to 416
}
const _: () = assert!(std::mem::size_of::<PerFrame>() == 416);

/// Blend two evaluated built-in COMP uniform snapshots. `PerFrame` is a packed
/// POD block of f32s, so this keeps new scalar controls automatically covered
/// by the same conservative finite-value rule. It is used only after both
/// shaderless renderer states have advanced for a shared-feedback frame.
fn blend_comp_perframe(outgoing: &mut PerFrame, incoming: &PerFrame, progress: f32) -> bool {
    if !progress.is_finite() {
        return false;
    }
    let weight = progress.clamp(0.0, 1.0);
    let outgoing_words: &mut [f32] =
        bytemuck::cast_slice_mut(std::slice::from_mut(outgoing));
    let incoming_words: &[f32] = bytemuck::cast_slice(std::slice::from_ref(incoming));
    for (outgoing, incoming) in outgoing_words.iter_mut().zip(incoming_words) {
        if outgoing.is_finite() && incoming.is_finite() {
            *outgoing += (*incoming - *outgoing) * weight;
        }
    }
    true
}

/// Apply a finite complementary transition weight to overlay alpha without
/// changing the authored RGB. The existing alpha/additive blend pipelines then
/// scale each state exactly once at draw time.
fn weight_shape_overlay_vertices(vertices: &mut [ShapeVert], weight: f32) {
    let weight = weight.clamp(0.0, 1.0);
    for vertex in vertices {
        vertex.color[3] = if vertex.color[3].is_finite() {
            (vertex.color[3] * weight).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

fn weight_wave_overlay_vertices(vertices: &mut [WaveVert], weight: f32) {
    let weight = weight.clamp(0.0, 1.0);
    for vertex in vertices {
        vertex.color[3] = if vertex.color[3].is_finite() {
            (vertex.color[3] * weight).clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

// ----- texture helpers -------------------------------------------------------

fn make_tex2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w: u32,
    h: u32,
    usage: wgpu::TextureUsages,
    data: Option<&[u8]>,
) -> wgpu::Texture {
    make_tex2d_with_mips(device, queue, w, h, usage, 1, data)
}

fn make_tex2d_with_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w: u32,
    h: u32,
    usage: wgpu::TextureUsages,
    mip_level_count: u32,
    data: Option<&[u8]>,
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: &[],
    });
    if let Some(pixels) = data {
        queue.write_texture(
            tex.as_image_copy(),
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    tex
}

/// Filterable two-row half-float texture used by the enhanced-audio helpers.
/// This is intentionally renderer-owned rather than a named-texture-atlas slot:
/// helpers need stable FFT/waveform rows every frame, while named textures are
/// static preset assets with finite gutters.
fn make_rgba16f_rows_texture(
    device: &wgpu::Device,
    width: u32,
    label: &'static str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Butterchurn's canonical blur target ratios and target-size quantization.
/// Widths use its slightly unusual `(size + 3) / 16` floor and heights use
/// `(size + 3) / 4`, with a 16-pixel minimum on both axes.
fn blur_dimensions(w: u32, h: u32) -> [(u32, u32); 6] {
    let size = |ratio: f64| {
        let x = ((w as f64 * ratio).max(16.0) as u32 + 3) / 16 * 16;
        let y = ((h as f64 * ratio).max(16.0) as u32 + 3) / 4 * 4;
        (x.max(16), y.max(16))
    };
    [
        size(0.25),
        size(0.125),
        size(0.0625),
        size(0.5),
        size(0.125),
        size(0.0625),
    ]
}

fn blur_min_max_remap(mut bmin: [f32; 3], mut bmax: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let fmin_dist = 0.1f32;
    if bmax[0] - bmin[0] < fmin_dist {
        let a = (bmin[0] + bmax[0]) * 0.5;
        bmin[0] = a - fmin_dist * 0.5;
        bmax[0] = a - fmin_dist * 0.5;
    }
    bmax[1] = bmax[1].min(bmax[0]);
    bmin[1] = bmin[1].max(bmin[0]);
    if bmax[1] - bmin[1] < fmin_dist {
        let a = (bmin[1] + bmax[1]) * 0.5;
        bmin[1] = a - fmin_dist * 0.5;
        bmax[1] = a - fmin_dist * 0.5;
    }
    bmax[2] = bmax[2].min(bmax[1]);
    bmin[2] = bmin[2].max(bmin[1]);
    if bmax[2] - bmin[2] < fmin_dist {
        let a = (bmin[2] + bmax[2]) * 0.5;
        bmin[2] = a - fmin_dist * 0.5;
        bmax[2] = a - fmin_dist * 0.5;
    }
    (bmin, bmax)
}

/// Butterchurn `getScaleAndBias`: the blur shader's normalize-into-`[0,1]` transform,
/// derived from the post-[`blur_min_max_remap`] endpoints. Transcribed verbatim, which
/// means it can and does produce non-finite results — see [`guard_finite_blur_scale_bias`].
fn blur_scale_and_bias(bmin: [f32; 3], bmax: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let mut scale = [1.0f32; 3];
    let mut bias = [0.0f32; 3];
    scale[0] = 1.0 / (bmax[0] - bmin[0]);
    bias[0] = -bmin[0] * scale[0];
    let t_min1 = (bmin[1] - bmin[0]) / (bmax[0] - bmin[0]);
    let t_max1 = (bmax[1] - bmin[0]) / (bmax[0] - bmin[0]);
    scale[1] = 1.0 / (t_max1 - t_min1);
    bias[1] = -t_min1 * scale[1];
    let t_min2 = (bmin[2] - bmin[1]) / (bmax[1] - bmin[1]);
    let t_max2 = (bmax[2] - bmin[1]) / (bmax[1] - bmin[1]);
    scale[2] = 1.0 / (t_max2 - t_min2);
    bias[2] = -t_min2 * scale[2];
    (scale, bias)
}

/// Replace non-finite blur coefficients before GPU upload. Large finite values
/// remain authored values; this is a safety guard, not a magnitude clamp.
fn guard_finite_blur_scale_bias(mut scale: [f32; 3], mut bias: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    for lvl in 0..3 {
        if !scale[lvl].is_finite() || !bias[lvl].is_finite() {
            scale[lvl] = 1.0;
            bias[lvl] = 0.0;
        }
    }
    (scale, bias)
}

/// Keep composition-side blur coefficients finite as well as the blur pass.
fn guard_finite_comp_blur(
    mut bmin: [f32; 3],
    mut bmax: [f32; 3],
    mut scale: [f32; 3],
    mut bias: [f32; 3],
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    for lvl in 0..3 {
        if !bmin[lvl].is_finite()
            || !bmax[lvl].is_finite()
            || !scale[lvl].is_finite()
            || !bias[lvl].is_finite()
        {
            bmin[lvl] = 0.0;
            bmax[lvl] = 1.0;
            scale[lvl] = 1.0;
            bias[lvl] = 0.0;
        }
    }
    (bmin, bmax, scale, bias)
}

fn milkdrop_angle(x: f64, y: f64, aspect_x: f64, aspect_y: f64) -> f64 {
    (y * aspect_y)
        .atan2(x * aspect_x)
        .rem_euclid(std::f64::consts::TAU)
}

fn seed_equation_inputs(env: &mut Env, width: u32, height: u32) {
    env.insert("frame", 0.0);
    env.insert("time", 0.0);
    // Match Butterchurn's `AudioLevels` state at `loadPreset`: instantaneous
    // bands have not received a PCM/FFT row yet (zero), while the attenuated
    // history starts at one. This setup frame is observable to per-frame,
    // shape, and wave init equations, so using OjoDrop's former all-ones audio
    // seed injected a phantom beat before real playback began.
    env.insert("fps", 45.0);
    for name in ["bass", "mid", "treb", "vol"] {
        env.insert(name, 0.0);
    }
    for name in ["bass_att", "mid_att", "treb_att", "vol_att"] {
        env.insert(name, 1.0);
    }
    let (aspect_x, aspect_y) = if width >= height {
        (1.0, height as f64 / width.max(1) as f64)
    } else {
        (width as f64 / height.max(1) as f64, 1.0)
    };
    env.insert("aspectx", 1.0 / aspect_x.max(f64::EPSILON));
    env.insert("aspecty", 1.0 / aspect_y.max(f64::EPSILON));
    env.insert("meshx", GRID_W as f64);
    env.insert("meshy", GRID_H as f64);
    env.insert("pixelsx", width as f64);
    env.insert("pixelsy", height as f64);
}

fn seed_preset_base_env(env: &mut Env, shaders: &MilkShaders) {
    let values = [
        ("zoom", shaders.zoom),
        ("zoomexp", shaders.zoomexp),
        ("rot", shaders.rot),
        ("warp", shaders.warp_amount),
        ("cx", shaders.cx),
        ("cy", shaders.cy),
        ("dx", shaders.dx),
        ("dy", shaders.dy),
        ("sx", shaders.sx),
        ("sy", shaders.sy),
        ("warpscale", shaders.warpscale),
        ("warpanimspeed", shaders.warpanimspeed),
        ("decay", shaders.decay),
        ("gamma", shaders.gamma_adj),
        ("gammaadj", shaders.gamma_adj),
        ("fshader", shaders.fshader),
        ("echo_zoom", shaders.echo_zoom),
        ("echo_alpha", shaders.echo_alpha),
        ("echo_orient", shaders.echo_orient),
        ("wave_mode", shaders.wave_mode),
        ("wave_x", shaders.wave_x),
        ("wave_y", shaders.wave_y),
        ("wave_r", shaders.wave_r),
        ("wave_g", shaders.wave_g),
        ("wave_b", shaders.wave_b),
        ("wave_a", shaders.wave_a),
        ("wave_mystery", shaders.wave_mystery),
        ("wave_scale", shaders.wave_scale),
        ("wave_smoothing", shaders.wave_smoothing),
        ("modwavealphastart", shaders.modwavealphastart),
        ("modwavealphaend", shaders.modwavealphaend),
        ("mv_x", shaders.mv_x),
        ("mv_y", shaders.mv_y),
        ("mv_dx", shaders.mv_dx),
        ("mv_dy", shaders.mv_dy),
        ("mv_l", shaders.mv_l),
        ("mv_r", shaders.mv_r),
        ("mv_g", shaders.mv_g),
        ("mv_b", shaders.mv_b),
        ("mv_a", shaders.mv_a),
        ("ob_size", shaders.ob_size),
        ("ob_r", shaders.ob_r),
        ("ob_g", shaders.ob_g),
        ("ob_b", shaders.ob_b),
        ("ob_a", shaders.ob_a),
        ("ib_size", shaders.ib_size),
        ("ib_r", shaders.ib_r),
        ("ib_g", shaders.ib_g),
        ("ib_b", shaders.ib_b),
        ("ib_a", shaders.ib_a),
        ("b1n", shaders.b1n),
        ("b1x", shaders.b1x),
        ("b1ed", shaders.b1ed),
        ("b2n", shaders.b2n),
        ("b2x", shaders.b2x),
        ("b3n", shaders.b3n),
        ("b3x", shaders.b3x),
    ];
    for (name, value) in values {
        env.insert(name, value as f64);
    }
    let flags = [
        ("wrap", shaders.wrap),
        ("wave_dots", shaders.wave_dots),
        ("wave_thick", shaders.wave_thick),
        ("additivewave", shaders.additive_wave),
        ("wave_brighten", shaders.wave_brighten),
        ("modwavealphabyvolume", shaders.modwavealphabyvolume),
        ("brighten", shaders.brighten),
        ("darken", shaders.darken),
        ("solarize", shaders.solarize),
        ("invert", shaders.invert),
        ("darken_center", shaders.darken_center),
    ];
    for (name, value) in flags {
        env.insert(name, if value { 1.0 } else { 0.0 });
    }
}

fn seed_shape_base_env(env: &mut Env, base: &ShapeBaseVals) {
    let values = [
        ("enabled", base.enabled as f64),
        ("sides", base.sides as f64),
        ("additive", base.additive as f64),
        ("thickoutline", base.thick_outline as f64),
        ("textured", base.textured as f64),
        ("num_inst", base.num_inst as f64),
        ("x", base.x as f64),
        ("y", base.y as f64),
        ("rad", base.rad as f64),
        ("ang", base.ang as f64),
        ("tex_ang", base.tex_ang as f64),
        ("tex_zoom", base.tex_zoom as f64),
        ("r", base.r as f64),
        ("g", base.g as f64),
        ("b", base.b as f64),
        ("a", base.a as f64),
        ("r2", base.r2 as f64),
        ("g2", base.g2 as f64),
        ("b2", base.b2 as f64),
        ("a2", base.a2 as f64),
        ("border_r", base.border_r as f64),
        ("border_g", base.border_g as f64),
        ("border_b", base.border_b as f64),
        ("border_a", base.border_a as f64),
    ];
    for (name, value) in values {
        env.insert(name, value);
    }
}

fn seed_wave_base_env(env: &mut Env, wave: &CustomWaveDef) {
    let values = [
        ("enabled", if wave.enabled { 1.0 } else { 0.0 }),
        ("samples", wave.samples as f64),
        ("sep", wave.sep as f64),
        ("spectrum", if wave.spectrum { 1.0 } else { 0.0 }),
        ("usedots", if wave.use_dots { 1.0 } else { 0.0 }),
        ("thick", if wave.draw_thick { 1.0 } else { 0.0 }),
        ("additive", if wave.additive { 1.0 } else { 0.0 }),
        ("scaling", wave.scaling as f64),
        ("smoothing", wave.smoothing as f64),
        ("r", wave.r as f64),
        ("g", wave.g as f64),
        ("b", wave.b as f64),
        ("a", wave.a as f64),
    ];
    for (name, value) in values {
        env.insert(name, value);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionError {
    /// A zero width or height was requested.
    Zero,
    /// Width or height exceeds the device `max_texture_dimension_2d`.
    ExceedsMaxTextureDimension { width: u32, height: u32, max: u32 },
    /// The pixel-count / row-byte / total-byte arithmetic overflowed.
    ArithmeticOverflow,
    /// The total texture footprint exceeds the renderer's memory budget.
    ExceedsMemoryBudget { bytes: u64, budget: u64 },
}

impl std::fmt::Display for DimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimensionError::Zero => write!(f, "texture dimensions must be non-zero"),
            DimensionError::ExceedsMaxTextureDimension { width, height, max } => write!(
                f,
                "texture dimensions {width}x{height} exceed device max_texture_dimension_2d ({max})"
            ),
            DimensionError::ArithmeticOverflow => {
                write!(f, "texture dimension arithmetic overflowed")
            }
            DimensionError::ExceedsMemoryBudget { bytes, budget } => write!(
                f,
                "texture allocation of {bytes} bytes exceeds the {budget}-byte budget"
            ),
        }
    }
}

impl std::error::Error for DimensionError {}

/// Upper-bound multiple of the base w*h*4 RGBA8 footprint that a MilkdropRenderer
/// allocates for one target: two mipmapped feedback targets (~1.34x each), three
/// blur outputs (1/16 + 1/64 + 1/256), blur temps (1/4 + 1/64 + 1/256), one
/// optional named-image atlas, and one comp target sum below ~4.5x for the
/// canonical profile; 6x is a safe ceiling.
const TEXTURE_FOOTPRINT_MULTIPLIER: u64 = 6;
/// Hard ceiling on the total texture memory a single render target may request.
/// A full 16384x16384 target (the common device max) is ~1 GiB base * 6 ≈ 6 GiB,
/// so 8 GiB admits legitimate max-dimension targets while rejecting pathological
/// (e.g. overflow-driven) sizes.
const MAX_TEXTURE_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Validate an external (w, h) render size with CHECKED arithmetic before any
/// resize or `create_texture`. `max_dim` is the device `max_texture_dimension_2d`.
/// Passing this is the precondition for every w*h-derived allocation in the
/// renderer (feedback/blur/comp textures and their CPU seed buffers).
pub(crate) fn validate_texture_dims(max_dim: u32, w: u32, h: u32) -> Result<(), DimensionError> {
    if w == 0 || h == 0 {
        return Err(DimensionError::Zero);
    }
    if w > max_dim || h > max_dim {
        return Err(DimensionError::ExceedsMaxTextureDimension {
            width: w,
            height: h,
            max: max_dim,
        });
    }
    // u32 `w * h` (the seed-buffer length) and `w * 4` (bytes_per_row) can WRAP;
    // do the math in u64 with explicit overflow checks so an out-of-range request
    // is rejected instead of silently under-sizing a buffer or a texture copy.
    let pixels = (w as u64)
        .checked_mul(h as u64)
        .ok_or(DimensionError::ArithmeticOverflow)?;
    let _row_bytes = (w as u64)
        .checked_mul(4)
        .ok_or(DimensionError::ArithmeticOverflow)?;
    let base_bytes = pixels
        .checked_mul(4)
        .ok_or(DimensionError::ArithmeticOverflow)?;
    let total_bytes = base_bytes
        .checked_mul(TEXTURE_FOOTPRINT_MULTIPLIER)
        .ok_or(DimensionError::ArithmeticOverflow)?;
    if total_bytes > MAX_TEXTURE_MEMORY_BYTES {
        return Err(DimensionError::ExceedsMemoryBudget {
            bytes: total_bytes,
            budget: MAX_TEXTURE_MEMORY_BYTES,
        });
    }
    Ok(())
}

fn mip_level_count_2d(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h).max(1);
    u32::BITS - max_dim.leading_zeros()
}

fn mip_level_view(texture: &wgpu::Texture, level: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        base_mip_level: level,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

fn mip_chain_views(texture: &wgpu::Texture, levels: u32) -> Vec<wgpu::TextureView> {
    (0..levels)
        .map(|level| mip_level_view(texture, level))
        .collect()
}

fn generate_mip_chain(
    device: &wgpu::Device,
    blitter: &wgpu::util::TextureBlitter,
    encoder: &mut wgpu::CommandEncoder,
    views: &[wgpu::TextureView],
) {
    for level in 1..views.len() {
        blitter.copy(device, encoder, &views[level - 1], &views[level]);
    }
}

fn encode_blur_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

fn downsample_rgba_volume(source: &[u8], source_size: u32) -> Vec<u8> {
    let target_size = (source_size / 2).max(1);
    let mut target =
        vec![0u8; target_size as usize * target_size as usize * target_size as usize * 4];
    for z in 0..target_size {
        for y in 0..target_size {
            for x in 0..target_size {
                let mut sum = [0u32; 4];
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sx = (x * 2 + dx).min(source_size - 1);
                            let sy = (y * 2 + dy).min(source_size - 1);
                            let sz = (z * 2 + dz).min(source_size - 1);
                            let offset =
                                (((sz * source_size + sy) * source_size + sx) * 4) as usize;
                            for channel in 0..4 {
                                sum[channel] += source[offset + channel] as u32;
                            }
                        }
                    }
                }
                let offset = (((z * target_size + y) * target_size + x) * 4) as usize;
                for channel in 0..4 {
                    target[offset + channel] = (sum[channel] / 8) as u8;
                }
            }
        }
    }
    target
}

fn make_tex3d(device: &wgpu::Device, queue: &wgpu::Queue, s: u32, data: &[u8]) -> wgpu::Texture {
    let mut mip_data = vec![data.to_vec()];
    let mut size = s;
    while size > 1 {
        mip_data.push(downsample_rgba_volume(
            mip_data.last().expect("base volume mip exists"),
            size,
        ));
        size = (size / 2).max(1);
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: s,
            height: s,
            depth_or_array_layers: s,
        },
        mip_level_count: mip_data.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut size = s;
    for (level, pixels) in mip_data.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: size,
            },
        );
        size = (size / 2).max(1);
    }
    tex
}

/// Derive a per-preset hue seed (Butterchurn's `rand_start`, normally 4× Math.random()
/// chosen at load). We hash the preset's shader/equation text so each preset gets a
/// distinct but reproducible hue (vs the old fixed 0.5 that biased everything green).
fn preset_hash64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // The shared LCG must not start from the all-zero-looking FNV offset for an
    // empty preset; mix the length and avalanche the final hash.
    h ^= s.len() as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^ (h >> 33)
}

fn preset_hue_seed(s: &str) -> [f32; 4] {
    let mut h = preset_hash64(s);
    let mut out = [0.0f32; 4];
    for slot in out.iter_mut() {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17; // xorshift64
        *slot = ((h >> 40) as f32) / ((1u64 << 24) as f32); // → [0,1)
    }
    out
}

fn named_texture_resolver() -> &'static NamedTextureResolver {
    static RESOLVER: OnceLock<NamedTextureResolver> = OnceLock::new();
    RESOLVER.get_or_init(|| NamedTextureResolver::new(Default::default()))
}

fn noise_bytes(n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    let mut x: u32 = 0xdeadbeef;
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let r = (
            (x & 0xff) as u8,
            ((x >> 8) & 0xff) as u8,
            ((x >> 16) & 0xff) as u8,
            255u8,
        );
        v.extend_from_slice(&[r.0, r.1, r.2, r.3]);
    }
    v
}

fn noise_bytes_scaled(n: usize, max_val: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    let mut x: u32 = 0xcafebabe;
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let scale = |b: u8| ((b as u32 * max_val as u32) / 255) as u8;
        v.push(scale((x & 0xff) as u8));
        v.push(scale(((x >> 8) & 0xff) as u8));
        v.push(scale(((x >> 16) & 0xff) as u8));
        v.push(255u8);
    }
    v
}

// ----- Butterchurn-faithful value/lattice noise (noise.js) -------------------
//
// Reproduces the createNoiseTex / createNoiseVolTex algorithm:
//   * random lattice fill (texRange 256 for zoom==1, 216 for zoom>1) with the JS
//     Uint8Array `& 0xFF` wrap emulated exactly (NOT clamping),
//   * separable per-axis Catmull-Rom cubic smoothing between lattice anchors
//     spaced `zoom` texels apart, wrapping (tiling) via modulo.
// RNG is a fixed-seed xorshift32 — Butterchurn uses non-deterministic Math.random
// but no preset depends on exact noise values (only on having structured value
// noise), so a deterministic seed is correct and reproducible for testing.

/// Butterchurn's seeded `xorshift128+` stream (`utils/seededRandom.js`).
///
/// The parity harness creates Butterchurn with its default deterministic seed
/// (12345), and the renderer constructs `Noise` before any other random-consuming
/// component.  Matching both the generator and its ten-value warm-up makes the
/// LQ noise texture byte-for-byte equivalent at its base level, which matters for
/// feedback presets that sample `sampler_noise_lq` directly.
struct ButterchurnRng {
    state: [u32; 4],
}

impl ButterchurnRng {
    const DEFAULT_SEED: u32 = 12_345;

    fn new(seed: u32) -> Self {
        let mut rng = Self {
            state: [
                seed,
                seed ^ 0x9e37_79b9,
                seed ^ 0x6a09_e667,
                seed ^ 0xbb67_ae85,
            ],
        };
        for _ in 0..10 {
            rng.next_unit();
        }
        rng
    }

    fn next_unit(&mut self) -> f64 {
        let mut t = self.state[3];
        let s = self.state[0];
        self.state[3] = self.state[2];
        self.state[2] = self.state[1];
        self.state[1] = s;
        t ^= t.wrapping_shl(11);
        t ^= t >> 8;
        self.state[0] = t ^ s ^ (s >> 19);
        f64::from(self.state[0]) / 4_294_967_296.0
    }
}

/// fCubicInterpolate (noise.js 158-170): Catmull-Rom-like cubic on scalar values.
fn cubic_interp(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t * t2;
    let a0 = y3 - y2 - y0 + y1;
    let a1 = y0 - y1 - a0;
    let a2 = y2 - y0;
    let a3 = y1;
    a0 * t3 + a1 * t2 + a2 * t + a3
}

/// dwCubicInterpolate (noise.js 172-184): per-channel cubic on 4 RGBA bytes.
/// Stores `f * 255` (0..255 after clamp) — JS truncates to Uint8Array, `as u8` matches.
fn dw_cubic(y0: &[u8; 4], y1: &[u8; 4], y2: &[u8; 4], y3: &[u8; 4], t: f32) -> [u8; 4] {
    let mut o = [0u8; 4];
    for c in 0..4 {
        let f = cubic_interp(
            y0[c] as f32 / 255.0,
            y1[c] as f32 / 255.0,
            y2[c] as f32 / 255.0,
            y3[c] as f32 / 255.0,
            t,
        )
        .clamp(0.0, 1.0);
        o[c] = (f * 255.0) as u8;
    }
    o
}

/// Read an RGBA texel from a flat byte buffer at texel index `i`.
fn rd4(buf: &[u8], i: usize) -> [u8; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// createNoiseTex (noise.js 318-399): size×size RGBA8 tiling value noise.
fn create_noise_tex(size: usize, zoom: usize, rng: &mut impl FnMut() -> f64) -> Vec<u8> {
    let n = size; // noiseSize
    let mut buf = vec![0u8; n * n * 4];

    // Random lattice fill.
    let range: f64 = if zoom > 1 { 216.0 } else { 256.0 };
    let half = range * 0.5;
    for px in 0..(n * n) {
        for c in 0..4 {
            let v = (rng() * range + half).floor() as i64;
            // JS Uint8Array wrap (& 0xFF), NOT clamp.
            buf[px * 4 + c] = (v as u32 & 0xFF) as u8;
        }
    }

    if zoom > 1 {
        // Pass 1 — interpolate along X (rows that are multiples of zoom).
        let mut y = 0usize;
        while y < n {
            for x in 0..n {
                if x % zoom != 0 {
                    let base_x = (x / zoom) * zoom + n; // +n keeps (base-zoom) non-negative
                    let base_y = y * n;
                    let y0 = rd4(&buf, base_y + ((base_x - zoom) % n));
                    let y1 = rd4(&buf, base_y + (base_x % n));
                    let y2 = rd4(&buf, base_y + ((base_x + zoom) % n));
                    let y3 = rd4(&buf, base_y + ((base_x + zoom * 2) % n));
                    let t = (x % zoom) as f32 / zoom as f32;
                    let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                    let dst = (y * n + x) * 4;
                    buf[dst..dst + 4].copy_from_slice(&r);
                }
            }
            y += zoom;
        }
        // Pass 2 — interpolate along Y (all columns, all rows).
        for x in 0..n {
            for y in 0..n {
                if y % zoom != 0 {
                    let base_y = (y / zoom) * zoom + n;
                    let y0 = rd4(&buf, ((base_y - zoom) % n) * n + x);
                    let y1 = rd4(&buf, (base_y % n) * n + x);
                    let y2 = rd4(&buf, ((base_y + zoom) % n) * n + x);
                    let y3 = rd4(&buf, ((base_y + zoom * 2) % n) * n + x);
                    let t = (y % zoom) as f32 / zoom as f32;
                    let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                    let dst = (y * n + x) * 4;
                    buf[dst..dst + 4].copy_from_slice(&r);
                }
            }
        }
    }

    buf
}

/// createNoiseVolTex (noise.js 183-318): size³ RGBA8 tiling value noise.
fn create_noise_vol_tex(size: usize, zoom: usize, rng: &mut impl FnMut() -> f64) -> Vec<u8> {
    let n = size;
    let words_per_slice = n * n;
    let words_per_line = n;
    let mut buf = vec![0u8; n * n * n * 4];

    // Random lattice fill.
    let range: f64 = if zoom > 1 { 216.0 } else { 256.0 };
    let half = range * 0.5;
    for px in 0..(n * n * n) {
        for c in 0..4 {
            let v = (rng() * range + half).floor() as i64;
            buf[px * 4 + c] = (v as u32 & 0xFF) as u8;
        }
    }

    if zoom > 1 {
        // Pass X (z,y step by zoom; x over all).
        let mut z = 0usize;
        while z < n {
            let mut y = 0usize;
            while y < n {
                for x in 0..n {
                    if x % zoom != 0 {
                        let base_x = (x / zoom) * zoom + n;
                        let base = z * words_per_slice + y * words_per_line;
                        let y0 = rd4(&buf, base + ((base_x - zoom) % n));
                        let y1 = rd4(&buf, base + (base_x % n));
                        let y2 = rd4(&buf, base + ((base_x + zoom) % n));
                        let y3 = rd4(&buf, base + ((base_x + zoom * 2) % n));
                        let t = (x % zoom) as f32 / zoom as f32;
                        let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                        let dst = (z * words_per_slice + y * words_per_line + x) * 4;
                        buf[dst..dst + 4].copy_from_slice(&r);
                    }
                }
                y += zoom;
            }
            z += zoom;
        }
        // Pass Y (z steps by zoom; x,y over all).
        let mut z = 0usize;
        while z < n {
            for x in 0..n {
                for y in 0..n {
                    if y % zoom != 0 {
                        let base_y = (y / zoom) * zoom + n;
                        let base_z = z * words_per_slice;
                        // sample index = ((base_y±k)%n)*words_per_line + base_z + x
                        let y0 = rd4(&buf, ((base_y - zoom) % n) * words_per_line + base_z + x);
                        let y1 = rd4(&buf, (base_y % n) * words_per_line + base_z + x);
                        let y2 = rd4(&buf, ((base_y + zoom) % n) * words_per_line + base_z + x);
                        let y3 = rd4(
                            &buf,
                            ((base_y + zoom * 2) % n) * words_per_line + base_z + x,
                        );
                        let t = (y % zoom) as f32 / zoom as f32;
                        let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                        let dst = (z * words_per_slice + y * words_per_line + x) * 4;
                        buf[dst..dst + 4].copy_from_slice(&r);
                    }
                }
            }
            z += zoom;
        }
        // Pass Z (x,y over all; z over all). FAITHFUL QUIRK: t uses (y%zoom), not z
        // (noise.js line 305) — replicate exactly to match Butterchurn.
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    if z % zoom != 0 {
                        let base_z = (z / zoom) * zoom + n;
                        let base_y = y * words_per_line;
                        let y0 = rd4(&buf, ((base_z - zoom) % n) * words_per_slice + base_y + x);
                        let y1 = rd4(&buf, (base_z % n) * words_per_slice + base_y + x);
                        let y2 = rd4(&buf, ((base_z + zoom) % n) * words_per_slice + base_y + x);
                        let y3 = rd4(
                            &buf,
                            ((base_z + zoom * 2) % n) * words_per_slice + base_y + x,
                        );
                        let t = (y % zoom) as f32 / zoom as f32; // QUIRK: y, not z
                        let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                        let dst = (z * words_per_slice + y * words_per_line + x) * 4;
                        buf[dst..dst + 4].copy_from_slice(&r);
                    }
                }
            }
        }
    }

    buf
}

// ----- bind group layout helpers --------------------------------------------

fn sampler_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> =
        Vec::with_capacity(MILKDROP_SAMPLERS.len() * 2);
    for (i, name) in MILKDROP_SAMPLERS.iter().enumerate() {
        let tex_bind = (i * 2) as u32;
        let samp_bind = tex_bind + 1;
        let dim = if name.contains("vol") {
            wgpu::TextureViewDimension::D3
        } else {
            wgpu::TextureViewDimension::D2
        };
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: tex_bind,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: samp_bind,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("milk-samplers-bgl"),
        entries: &entries,
    })
}

fn perframe_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let ubo_binding = (MILKDROP_SAMPLERS.len() * 2) as u32;
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("perframe-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: ubo_binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn blur_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blur-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn build_sampler_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    main_view: &wgpu::TextureView,
    blur1_view: &wgpu::TextureView,
    blur2_view: &wgpu::TextureView,
    blur3_view: &wgpu::TextureView,
    _noise2d_view: &wgpu::TextureView,
    noise_lq_view: &wgpu::TextureView,
    noise_mq_view: &wgpu::TextureView,
    noise_hq_view: &wgpu::TextureView,
    noise_lite_view: &wgpu::TextureView,
    named_linear_view: &wgpu::TextureView,
    named_point_view: &wgpu::TextureView,
    enhanced_audio_helpers: bool,
    noisevol_lq_view: &wgpu::TextureView,
    noisevol_hq_view: &wgpu::TextureView,
    main_samp: &wgpu::Sampler,
    repeat_samp: &wgpu::Sampler,
    samp_clamp: &wgpu::Sampler,
    samp_point: &wgpu::Sampler,
    samp_point_clamp: &wgpu::Sampler,
) -> wgpu::BindGroup {
    use wgpu::{BindGroupEntry, BindingResource};
    let mut entries: Vec<BindGroupEntry<'_>> = Vec::with_capacity(MILKDROP_SAMPLERS.len() * 2);
    for (i, name) in MILKDROP_SAMPLERS.iter().enumerate() {
        let (view, sampler) = match *name {
            // `sampler_main` follows the live per-frame `wrap` value. The
            // force-wrap variant remains repeat regardless of that value.
            "sampler_main" => (main_view, main_samp),
            "sampler_fw_main" => (main_view, repeat_samp),
            "sampler_fc_main" => (main_view, samp_clamp),
            "sampler_pw_main" => (main_view, samp_point),
            "sampler_pc_main" => (main_view, samp_point_clamp),
            "sampler_blur1" => (blur1_view, samp_clamp),
            "sampler_blur2" => (blur2_view, samp_clamp),
            "sampler_blur3" => (blur3_view, samp_clamp),
            "sampler_noise_lq" => (noise_lq_view, repeat_samp),
            "sampler_noise_lq_lite" | "sampler_noise_hq_lite" => (noise_lite_view, repeat_samp),
            "sampler_noise_mq" => (noise_mq_view, repeat_samp),
            "sampler_noise_hq" => (noise_hq_view, repeat_samp),
            "sampler_named_linear" => (named_linear_view, samp_clamp),
            "sampler_named_point" => (
                named_point_view,
                if enhanced_audio_helpers {
                    // BeatDrop's waveform helper interpolates this row too.
                    samp_clamp
                } else {
                    samp_point_clamp
                },
            ),
            "sampler_pw_noise_lq" => (noise_lq_view, samp_point),
            "sampler_noisevol_lq" => (noisevol_lq_view, repeat_samp),
            "sampler_noisevol_hq" => (noisevol_hq_view, repeat_samp),
            _ => (_noise2d_view, repeat_samp),
        };
        let tex_bind = (i * 2) as u32;
        entries.push(BindGroupEntry {
            binding: tex_bind,
            resource: BindingResource::TextureView(view),
        });
        entries.push(BindGroupEntry {
            binding: tex_bind + 1,
            resource: BindingResource::Sampler(sampler),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bgl,
        entries: &entries,
    })
}

// ----- naga compilation ------------------------------------------------------

pub fn compile_glsl(glsl: &str) -> Result<String, String> {
    use naga::{
        back::wgsl as wgsl_out,
        front::glsl as glsl_in,
        valid::{Capabilities, ValidationFlags, Validator},
    };
    // Repair HLSL-permissive type mismatches (vec<scalar comparisons, …) that naga
    // rejects. Conservative: only confidently-typed constructs are rewritten.
    let glsl_fixed = fix_glsl_vector_types(glsl);
    if std::env::var("MILKDROP_DUMP_FIXED").is_ok() {
        eprintln!("==== type-fixed GLSL ====\n{glsl_fixed}\n==== end fixed ====");
    }
    let glsl = glsl_fixed.as_str();
    let mut parser = glsl_in::Frontend::default();
    let opts = glsl_in::Options {
        stage: naga::ShaderStage::Fragment,
        defines: Default::default(),
    };
    let module = parser.parse(&opts, glsl).map_err(|e| format!("{e:?}"))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| {
            // naga's Display for a validation error stops at "Function 'main' is
            // invalid" — the actual cause (bad expression/type) is in the error
            // source chain. Append it so triage can see WHY validation failed.
            use std::error::Error;
            let mut msg = format!("{e}");
            let mut src = e.source();
            while let Some(s) = src {
                msg.push_str(&format!("  ->  {s}"));
                src = s.source();
            }
            msg
        })?;
    let wgsl = wgsl_out::write_string(&module, &info, wgsl_out::WriterFlags::empty())
        .map_err(|e| format!("{e}"))?;
    Ok(wgsl)
}

#[derive(Clone, Debug)]
pub struct CompiledMilkdropShaderBodies {
    pub warp_wgsl: String,
    pub warp_custom_wgsl: String,
    pub comp_wgsl: String,
    pub named_texture_plan: NamedTexturePlan,
}

pub fn compile_milkdrop_shader_bodies(
    shaders: &MilkShaders,
) -> Result<CompiledMilkdropShaderBodies, String> {
    compile_milkdrop_shader_bodies_from_parts(
        shaders.shaders_glsl,
        shaders.warp.as_deref(),
        shaders.comp.as_deref(),
    )
}

pub fn compile_milkdrop_shader_bodies_from_parts(
    shaders_glsl: bool,
    warp: Option<&str>,
    comp: Option<&str>,
) -> Result<CompiledMilkdropShaderBodies, String> {
    let warp_default = "ret = GetMain(uv);";
    let comp_default = "float _eh = mod(echo_orientation, 2.0); \
             float _ex = (_eh != 0.0) ? -1.0 : 1.0; \
             float _ey = (echo_orientation >= 2.0) ? -1.0 : 1.0; \
             vec2 uv_echo = ((uv - 0.5) * (1.0 / echo_zoom) * vec2(_ex, _ey)) + 0.5; \
             ret = mix(GetMain(uv), GetMain(uv_echo), echo_alpha); \
             ret = ret * gammaAdj; \
             if (fShader >= 1.0) ret = ret * hue_shader; \
             else if (fShader > 0.001) ret = mix(ret, ret * hue_shader, fShader); \
             if (brighten != 0.0) ret = sqrt(ret); \
             if (darken   != 0.0) ret = ret * ret; \
             if (solarize != 0.0) ret = ret * (1.0 - ret) * 4.0; \
             if (invert   != 0.0) ret = 1.0 - ret;";

    let named_texture_plan = NamedTexturePlan::from_sources([warp, comp].into_iter().flatten());
    let named_bindings = named_texture_plan.shader_rewrite_bindings();
    let named_layer_size = DEFAULT_NAMED_TEXTURE_LAYER_SIZE;

    // shaders_glsl path (Butterchurn converted-JSON): the custom warp/comp
    // bodies are already GLSL, so compile them via the GLSL-body path.
    let warp_custom_glsl = match (shaders_glsl, warp) {
        (true, Some(body)) => {
            glsl_milk_warp_body_to_naga_with_named_textures(body, &named_bindings, named_layer_size)
        }
        _ => hlsl_milk_warp_body_to_naga_with_named_textures(
            warp.unwrap_or(warp_default),
            &named_bindings,
            named_layer_size,
        ),
    };
    let comp_glsl = match (shaders_glsl, comp) {
        (true, Some(body)) => {
            glsl_milk_body_to_naga_with_named_textures(body, &named_bindings, named_layer_size)
        }
        _ => hlsl_milk_body_to_naga_with_named_textures(
            comp.unwrap_or(comp_default),
            &named_bindings,
            named_layer_size,
        ),
    };

    if std::env::var("MILKDROP_DUMP_GLSL").is_ok() {
        eprintln!("==== custom warp GLSL ====\n{warp_custom_glsl}\n==== end custom warp ====");
        eprintln!("==== comp GLSL ====\n{comp_glsl}\n==== end comp ====");
    }
    let warp_custom_wgsl = compile_glsl(&warp_custom_glsl)?;
    let comp_wgsl = compile_glsl(&comp_glsl)?;
    if std::env::var("MILKDROP_DUMP_WARP_WGSL").is_ok() {
        eprintln!("==== custom warp WGSL (naga) ====\n{warp_custom_wgsl}\n==== end warp WGSL ====");
    }

    Ok(CompiledMilkdropShaderBodies {
        // Retained (empty) for the particle-core shader-cache byte-accounting ABI;
        // the legacy fullscreen warp pipeline it fed no longer exists.
        warp_wgsl: String::new(),
        warp_custom_wgsl,
        comp_wgsl,
        named_texture_plan,
    })
}

fn milkdrop_body_samples_blur(body: &str, level: u8) -> bool {
    // Lowercase first so source-case variants (e.g. `SAMPLER_PW_BLUR2`) still hit
    // the normalizer's lowercase prefix patterns; then collapse mode prefixes.
    let normalized = normalize_milkdrop_sampler_variants(&body.to_ascii_lowercase());
    let n = char::from(b'0' + level);
    normalized.contains(&format!("getblur{n}")) || normalized.contains(&format!("sampler_blur{n}"))
}

pub(crate) fn needed_blur_levels(warp: Option<&str>, comp: Option<&str>) -> u8 {
    let mut level = 0u8;
    for body in [warp, comp].into_iter().flatten() {
        for candidate in 1..=3u8 {
            if candidate > level && milkdrop_body_samples_blur(body, candidate) {
                level = candidate;
            }
        }
    }
    level
}

// compute_warp_verts is now a method on MilkdropRenderer (see impl block) — it
// runs the per_pixel EEL program per vertex and composes the butterchurn warped UV.

/// How feedback content is carried across a feedback/blur/comp target rebuild.
#[derive(Copy, Clone, PartialEq, Eq)]
enum CarryMode {
    /// Preserve the top-left `min(old, new)` region (MilkDrop's legacy
    /// output-resize behaviour). Byte-preserving when the dimensions match.
    Crop,
    /// UV-space bilinear resample of the whole page. Used on an internal-scale
    /// transition so a scale change carries the full frame across the new size
    /// with no black border or luminance collapse.
    Resample,
}

/// Internal render dimensions for a given output size and internal scale.
///
/// A `scale` of `>= 1.0` (or any non-finite value) returns the output size
/// UNCHANGED so the default path (`internal_scale == 1.0`) is byte-identical to
/// rendering directly at the output resolution. A `scale` in `(0, 1)` shrinks
/// both axes (rounded, clamped to `[1, output]`); aspect ratio is preserved up
/// to rounding.
fn scaled_internal_dims(w: u32, h: u32, scale: f32) -> (u32, u32) {
    if !scale.is_finite() || scale >= 1.0 {
        return (w, h);
    }
    let s = scale.max(f32::MIN_POSITIVE);
    let iw = ((w as f32 * s).round() as u32).clamp(1, w.max(1));
    let ih = ((h as f32 * s).round() as u32).clamp(1, h.max(1));
    (iw, ih)
}

/// Hot-switchable fidelity/performance bundles for the OjoDrop MilkDrop path.
///
/// Each variant bundles an internal render scale, a feedback mip-chain cap, and
/// whether the FXAA output pass runs. [`MilkdropPerformanceProfile::Reference`]
/// (the default) is full fidelity and byte-identical to the pre-profile
/// behaviour: native internal resolution, the full feedback mip chain, and FXAA
/// enabled. The `*60` variants trade fidelity for headroom and are meant to be
/// selected by a downstream governor; nothing selects them automatically.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum MilkdropPerformanceProfile {
    /// Full fidelity. Internal scale 1.0, full feedback mips, FXAA on. Default.
    #[default]
    Reference,
    /// Native resolution and full feedback mips, but the FXAA output pass is
    /// dropped (its own final AA is expected downstream).
    High60,
    /// Slightly reduced internal resolution with the FXAA pass dropped.
    Balanced60,
    /// Aggressive rescue: reduced internal resolution, a capped feedback mip
    /// chain, and no FXAA pass.
    Rescue60,
}

impl MilkdropPerformanceProfile {
    /// `(internal_scale, feedback_mip_cap, fxaa_enabled)` for this profile.
    /// `feedback_mip_cap == u32::MAX` means "no cap" (the full log2 chain).
    fn params(self) -> (f32, u32, bool) {
        match self {
            MilkdropPerformanceProfile::Reference => (1.0, u32::MAX, true),
            MilkdropPerformanceProfile::High60 => (1.0, u32::MAX, false),
            MilkdropPerformanceProfile::Balanced60 => (0.83, u32::MAX, false),
            MilkdropPerformanceProfile::Rescue60 => (0.67, 4, false),
        }
    }
}

// ----- main renderer struct --------------------------------------------------

pub struct MilkdropRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // which warp/comp path to use
    has_custom_warp: bool,
    has_custom_comp: bool,
    /// preset's decay value (used in warp mesh pass)
    preset_decay: f32,
    /// Persistent random vectors are distinct in Butterchurn: rand_start drives
    /// built-in hue phases while rand_preset is visible to authored shaders.
    rand_start: [f32; 4],
    rand_preset: [f32; 4],

    // ping-pong feedback textures (both RGBA8, same size as render)
    tex_a: wgpu::Texture,
    tex_b: wgpu::Texture,
    // level-0 render-attachment views for feedback writes
    view_a: wgpu::TextureView,
    view_b: wgpu::TextureView,
    // all-mip sampling views for shader feedback reads
    #[allow(dead_code)]
    view_a_sample: wgpu::TextureView,
    #[allow(dead_code)]
    view_b_sample: wgpu::TextureView,
    feedback_mips_a: Vec<wgpu::TextureView>,
    feedback_mips_b: Vec<wgpu::TextureView>,
    feedback_mip_blitter: wgpu::util::TextureBlitter,
    write_to_a: bool, // true → write_to_a, read from b
    /// Legacy by default. The BeatDrop ordering is opt-in through
    /// [`Self::set_beatdrop_feedback`].
    feedback_provenance: FeedbackProvenance,
    /// Per-preset BeatDrop FFT follower controls. The renderer keeps player
    /// gain/noise-floor defaults unless a later host-level control changes them.
    enhanced_audio_config: EnhancedAudioConfig,

    // blur textures and horizontal-pass intermediates.
    blur1: wgpu::Texture,
    blur2: wgpu::Texture,
    blur3: wgpu::Texture,
    view_blur1: wgpu::TextureView,
    view_blur2: wgpu::TextureView,
    view_blur3: wgpu::TextureView,
    // All-mip sampling views plus one-level views used to build each pyramid.
    view_blur1_sample: wgpu::TextureView,
    view_blur2_sample: wgpu::TextureView,
    view_blur3_sample: wgpu::TextureView,
    blur_mips1: Vec<wgpu::TextureView>,
    blur_mips2: Vec<wgpu::TextureView>,
    blur_mips3: Vec<wgpu::TextureView>,
    // separable-blur horizontal-pass intermediates (same res as blur1/2/3)
    btemp1: wgpu::Texture,
    btemp2: wgpu::Texture,
    btemp3: wgpu::Texture,
    view_btemp1: wgpu::TextureView,
    view_btemp2: wgpu::TextureView,
    view_btemp3: wgpu::TextureView,
    view_btemp1_sample: wgpu::TextureView,
    view_btemp2_sample: wgpu::TextureView,
    view_btemp3_sample: wgpu::TextureView,
    btemp_mips1: Vec<wgpu::TextureView>,
    btemp_mips2: Vec<wgpu::TextureView>,
    btemp_mips3: Vec<wgpu::TextureView>,

    // Per-preset custom-image atlas. Custom sampler calls are rewritten to one
    // of two reserved bindings (linear/point) that share this view.
    #[allow(dead_code)]
    named_texture_atlas: wgpu::Texture,
    view_named_texture_atlas: wgpu::TextureView,

    /// Two dynamically uploaded rows used only when compiled preset code calls
    /// an enhanced-audio helper. Keeping them separate from the named-image
    /// atlas prevents a helper shader from accidentally sampling static art.
    enhanced_audio_enabled: bool,
    enhanced_fft_texture: wgpu::Texture,
    enhanced_wave_texture: wgpu::Texture,
    view_enhanced_fft_texture: wgpu::TextureView,
    view_enhanced_wave_texture: wgpu::TextureView,
    enhanced_audio_processor: EnhancedAudioProcessor,
    enhanced_audio_fft_upload: Vec<u16>,
    enhanced_audio_wave_upload: Vec<u16>,
    enhanced_audio_sample_rate_hz: f32,
    enhanced_audio_nyquist_hz: f32,

    // noise textures (Butterchurn-faithful; kept alive — views borrowed by bind groups)
    #[allow(dead_code)]
    noise2d: wgpu::Texture, // placeholder for fw/pw/pc slots
    #[allow(dead_code)]
    noise_lq: wgpu::Texture,
    #[allow(dead_code)]
    noise_mq: wgpu::Texture,
    #[allow(dead_code)]
    noise_hq: wgpu::Texture,
    #[allow(dead_code)]
    noise_lite: wgpu::Texture,
    #[allow(dead_code)]
    noisevol_lq: wgpu::Texture,
    #[allow(dead_code)]
    noisevol_hq: wgpu::Texture,
    #[allow(dead_code)]
    view_noise2d: wgpu::TextureView,
    #[allow(dead_code)]
    view_noise_lq: wgpu::TextureView,
    #[allow(dead_code)]
    view_noise_mq: wgpu::TextureView,
    #[allow(dead_code)]
    view_noise_hq: wgpu::TextureView,
    #[allow(dead_code)]
    view_noise_lite: wgpu::TextureView,
    #[allow(dead_code)]
    view_noisevol_lq: wgpu::TextureView,
    #[allow(dead_code)]
    view_noisevol_hq: wgpu::TextureView,

    // samplers
    linear_samp: wgpu::Sampler,
    clamp_samp: wgpu::Sampler,
    point_samp: wgpu::Sampler,
    point_clamp_samp: wgpu::Sampler,

    // UBO
    perframe_buf: wgpu::Buffer,
    comp_perframe_buf: wgpu::Buffer,
    /// Most recently evaluated COMP uniform snapshot. Shared feedback blends
    /// this with the independently advanced target before final composition.
    last_comp_perframe: PerFrame,

    // blur pass uniform buffers (one per pass, holds texel size of source)
    blur1_ubo: wgpu::Buffer,
    blur2_ubo: wgpu::Buffer,
    blur3_ubo: wgpu::Buffer,

    warp_custom_pipeline: wgpu::RenderPipeline,
    comp_pipeline: wgpu::RenderPipeline,
    /// Format-correct COMP path used when FXAA is disabled. `comp_pipeline`
    /// always targets the retained Rgba8 texture; this one targets the host's
    /// requested output format (Particle uses Rgba16Float HDR).
    comp_direct_pipeline: wgpu::RenderPipeline,
    comp_vert_buf: wgpu::Buffer,
    comp_idx_buf: wgpu::Buffer,
    comp_idx_count: u32,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    // FXAA output pass: COMP → comp_view (offscreen Rgba8Unorm) → FXAA → swapchain.
    #[allow(dead_code)]
    comp_tex: wgpu::Texture, // kept alive; comp_view borrows it
    comp_view: wgpu::TextureView,
    output_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    fxaa_bgl: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    fxaa_ubo: wgpu::Buffer,
    fxaa_bg: wgpu::BindGroup,
    /// Runtime toggle for the FXAA output pass. Default `true` (COMP → offscreen
    /// comp intermediate → FXAA → swapchain). When `false`, COMP writes the
    /// swapchain directly and both the FXAA pass and the intermediate
    /// write/read round-trip are skipped — reserved for a downstream tier that
    /// provides its own final anti-aliasing.
    fxaa_enabled: bool,
    /// Internal render scale in `(0, 1]`. `1.0` (default) renders the feedback,
    /// blur pyramid, and comp target at the full output resolution — byte-
    /// identical to the pre-scale behaviour. Values `< 1.0` shrink those
    /// internal targets and the final comp/FXAA pass upscales to the output.
    internal_scale: f32,
    /// Internal render dimensions == `scaled_internal_dims(width, height,
    /// internal_scale)`. These size every internal target (feedback ping-pong,
    /// blur pyramid, comp) and drive the per-frame render-canvas uniforms. At
    /// `internal_scale == 1.0` they equal `(width, height)`.
    render_w: u32,
    render_h: u32,
    /// Cap on the number of feedback ping-pong mip levels allocated AND
    /// regenerated each frame. `u32::MAX` (default) means the full log2 chain
    /// (byte-identical). Does NOT affect the separate blur pyramid.
    feedback_mip_cap: u32,
    /// Currently selected performance profile (bundles `internal_scale`,
    /// `feedback_mip_cap`, and `fxaa_enabled`). Defaults to `Reference`.
    perf_profile: MilkdropPerformanceProfile,
    // standard warp mesh (used when no custom warp shader)
    warp_mesh_pipeline: wgpu::RenderPipeline,
    warp_mesh_bg_a: wgpu::BindGroup, // reads from tex_a, repeat
    warp_mesh_bg_b: wgpu::BindGroup, // reads from tex_b, repeat
    warp_mesh_bg_a_clamp: wgpu::BindGroup,
    warp_mesh_bg_b_clamp: wgpu::BindGroup,
    warp_mesh_bgl: wgpu::BindGroupLayout,
    warp_params_buf: wgpu::Buffer,
    warp_params_bgl: wgpu::BindGroupLayout,
    warp_params_bg: wgpu::BindGroup,
    warp_vert_buf: wgpu::Buffer, // updated per frame
    warp_idx_buf: wgpu::Buffer,  // static
    warp_idx_count: u32,
    /// A shared-feedback target must publish an evaluated CPU warp mesh even
    /// when it has no per-pixel equations or motion vectors. This remains
    /// private: it is flipped only around the target's hidden state advance.
    force_cpu_warp_mesh: bool,

    // bind group layouts
    sampler_bgl: wgpu::BindGroupLayout,
    perframe_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,

    // sampler bind groups — one per ping-pong side, for WARP reading the OTHER side
    // bg_read_a: sampler_main = view_a  (use when comp reads curr=a, or warp reads prev=a)
    // bg_read_b: sampler_main = view_b
    bg_read_a: wgpu::BindGroup,
    bg_read_b: wgpu::BindGroup,
    bg_read_a_clamp: wgpu::BindGroup,
    bg_read_b_clamp: wgpu::BindGroup,

    // perframe bind group
    perframe_bg: wgpu::BindGroup,
    comp_perframe_bg: wgpu::BindGroup,

    // Blur bind groups for the separable H/V chain. Both possible blur1 sources
    // are prebuilt so the frame loop never creates a bind group.
    blur1_h_bg_a: wgpu::BindGroup,
    blur1_h_bg_b: wgpu::BindGroup,
    blur1_v_bg: wgpu::BindGroup,
    blur2_h_bg: wgpu::BindGroup,
    blur2_v_bg: wgpu::BindGroup,
    blur3_h_bg: wgpu::BindGroup,
    blur3_v_bg: wgpu::BindGroup,
    blur_levels: u8,
    last_blur_pass_count: u32,

    // EEL2 per-frame equations
    eel_program: Option<EelProgram>,
    eel_env: Env,
    /// Per-frame megabuf pool (private) + shared preset-wide gmegabuf handle.
    eel_state: EelState,
    /// Preset-owned random stream shared by every EEL pool and shader randoms.
    eel_rng: Arc<EelRng>,
    /// Preset-wide gmegabuf shared by all pools (per-frame/per-pixel/shape/wave).
    #[allow(dead_code)]
    gmegabuf: Arc<Mutex<MegaBuf>>,
    /// q1..q32 post-init snapshot — re-applied at the top of every frame so
    /// accumulator-q presets don't drift (Butterchurn's per-frame q reset).
    q_init: [f64; 32],

    // Per-vertex warp (per_pixel) program + per-frame warp base values.
    per_pixel_prog: Option<EelProgram>,
    base_warp: WarpBase,
    /// Scratch EEL env reused across warp vertices (avoids per-vertex alloc).
    warp_env: Env,
    /// Pre-interned dense slots used by the per-pixel hot loop.
    warp_slots: WarpEnvSlots,
    eel_reg_slots: [EnvSlot; 100],
    eel_q_slots: [EnvSlot; 32],
    warp_reg_slots: [EnvSlot; 100],
    /// Dense ten-control snapshot restored before every per-pixel evaluation.
    warp_snapshot: EnvSnapshot,
    /// Per-pixel megabuf pool (private) sharing the preset-wide gmegabuf.
    warp_state: EelState,

    /// Allocation-stable geometry and uniform staging storage.
    scratch: RendererScratch,
    /// Disabled by default. When enabled, summarizes already-built CPU geometry
    /// once per frame; no geometry scan runs on the production path.
    geometry_diagnostics: GeometryDiagnosticCollector,
    /// Three full-frame readback buffers allocated only while diagnostics are
    /// enabled. Copies are encoded on the worker's extra untimed frame.
    geometry_stage_readback: Option<GeometryStageReadback>,

    // frame state
    frame_idx: u64,
    start: std::time::Instant,
    /// When Some(dt), time advances by `dt` seconds per rendered frame instead
    /// of using the wall clock. Used for deterministic offscreen animation export.
    time_per_frame: Option<f64>,
    /// Live audio reactivity. When Some([bass, mid, treb, vol]), these drive the
    /// per-frame audio uniforms instead of the synthetic sine-wave fallback.
    audio: Option<[f32; 4]>,
    /// Live attenuated (smoothed) reactivity [bass_att, mid_att, treb_att, vol_att].
    /// When None (headless/synthetic), `*_att` falls back to the non-att values so
    /// deterministic renders stay bit-identical to before this wiring existed.
    audio_att: Option<[f32; 4]>,
    /// Butterchurn-shaped 512-bin FFT magnitude array for `bSpectrum` custom
    /// waveforms. Empty when no live audio (built-in/synthetic path uses time data).
    freq_spectrum: Vec<f32>,
    /// Optional independently derived right-channel spectrum. Legacy callers
    /// provide only the mono row above; extended waveform mode 8 stays empty in
    /// that case instead of pretending mono duplication is BeatDrop stereo FFT.
    freq_spectrum_right: Vec<f32>,
    /// Optional exact Butterchurn shader random vectors for the next frame. This
    /// is chiefly useful to make an offline parity capture independent of each
    /// renderer's internal PRNG implementation.
    frame_random_override: Option<([f32; 4], [f32; 4])>,
    /// Optional Butterchurn shader clock for the next frame. Offscreen parity
    /// capture can replay Butterchurn's smoothed-FPS clock exactly instead of
    /// merely approximating it with a fixed wall-clock cadence.
    frame_time_override: Option<f64>,
    pub width: u32,
    pub height: u32,

    pub surface_format: wgpu::TextureFormat,

    // ── Custom shapes ────────────────────────────────────────────────────────
    shapes: Vec<ShapeRT>,
    shapes_fill_pipeline_alpha: wgpu::RenderPipeline,
    shapes_fill_pipeline_additive: wgpu::RenderPipeline,
    shapes_border_pipeline: wgpu::RenderPipeline,
    shape_bgl: wgpu::BindGroupLayout,
    border_bgl: wgpu::BindGroupLayout,
    shape_vert_buf: wgpu::Buffer,
    shape_idx_buf: wgpu::Buffer, // static fan triangulation, 300 u32
    border_vert_buf: wgpu::Buffer,
    // border uniforms: dyn-offset buffer (4 slots of 256B = up to 4 thick passes)
    border_uniform_buf: wgpu::Buffer,
    border_bg: wgpu::BindGroup,
    // shape fill bind groups, one per ping-pong read side (prev-frame texture)
    shape_bg_read_a: wgpu::BindGroup,
    shape_bg_read_b: wgpu::BindGroup,
    shape_bg_read_a_clamp: wgpu::BindGroup,
    shape_bg_read_b_clamp: wgpu::BindGroup,

    // ── Waveforms (built-in + custom) ────────────────────────────────────────
    waves: Vec<WaveRT>,
    wave_pipeline_lines_alpha: wgpu::RenderPipeline,
    wave_pipeline_lines_additive: wgpu::RenderPipeline,
    wave_pipeline_points_alpha: wgpu::RenderPipeline,
    wave_pipeline_points_additive: wgpu::RenderPipeline,
    wave_bgl: wgpu::BindGroupLayout,
    wave_vert_buf: wgpu::Buffer,
    wave_off_buf: wgpu::Buffer, // texel size for instance-index thick offsets
    wave_bg: wgpu::BindGroup,
    /// Guarded static LOD for expensive, side-effect-free custom per-point EEL.
    custom_wave_adaptive_lod: bool,

    // built-in waveform scalar/bool state (parsed)
    bw_mode: f32,
    bw_x: f32,
    bw_y: f32,
    bw_r: f32,
    bw_g: f32,
    bw_b: f32,
    bw_a: f32,
    bw_mystery: f32,
    bw_scale: f32,
    bw_smoothing: f32,
    bw_dots: bool,
    bw_thick: bool,
    bw_additive: bool,
    bw_brighten: bool,
    bw_modalphavol: bool,
    bw_modalphastart: f32,
    bw_modalphaend: f32,

    // comp post-FX flags (bBrighten/bDarken/bSolarize/bInvert) for the built-in comp body
    comp_gamma_adj: f32,
    comp_fshader: f32,
    echo_zoom: f32,
    echo_alpha: f32,
    echo_orient: f32,
    comp_brighten: bool,
    comp_darken: bool,
    comp_solarize: bool,
    comp_invert: bool,

    // ── Motion vectors ───────────────────────────────────────────────────────
    mv_pipeline: wgpu::RenderPipeline, // LineList, alpha blend, Rgba8Unorm
    mv_bgl: wgpu::BindGroupLayout,
    mv_vert_buf: wgpu::Buffer,
    mv_color_buf: wgpu::Buffer, // 16-byte uniform (vec4 color)
    mv_bg: wgpu::BindGroup,
    mv_on: bool,
    mv_x: f32,
    mv_y: f32,
    mv_dx: f32,
    mv_dy: f32,
    mv_l: f32,
    mv_r: f32,
    mv_g: f32,
    mv_b: f32,
    mv_a: f32,

    // ── Frame borders (outer/inner) ──────────────────────────────────────────
    // Reuses border_bgl (BorderU) + a triangle-list pipeline. 24 verts/border.
    frame_border_pipeline: wgpu::RenderPipeline,
    frame_border_vert_buf: wgpu::Buffer, // up to 2 borders * 24 verts
    frame_border_uniform_buf: wgpu::Buffer, // dyn-offset, 2 slots of 256B
    frame_border_bg: wgpu::BindGroup,
    ob_size: f32,
    ob_r: f32,
    ob_g: f32,
    ob_b: f32,
    ob_a: f32,
    ib_size: f32,
    ib_r: f32,
    ib_g: f32,
    ib_b: f32,
    ib_a: f32,

    // ── Darken center ────────────────────────────────────────────────────────
    darken_pipeline: wgpu::RenderPipeline, // TriangleList, alpha blend
    darken_vert_buf: wgpu::Buffer,         // 12 verts (4 fan tris)
    darken_center: bool,

    // previous-frame volume, used to derive the MilkDrop-style `diff` pseudo-var
    // (frame-to-frame volume delta) so presets like orb_waaa can gate mv_a on it.
    vol_prev: f64,

    // ── Blur min/max (per-level range remap base; overridable per-frame via EEL) ─
    b1n: f32,
    b1x: f32,
    b1ed: f32,
    b2n: f32,
    b2x: f32,
    b3n: f32,
    b3x: f32,

    // per-sample audio waveform (range ~[-1,1]); filled by set_waveform or synthesized.
    wave_l: Vec<f32>,
    wave_r: Vec<f32>,

    // Compatibility/debug counter. Butterchurn-parity feedback starts black, so
    // no feedback-seed noise is generated at init or resize and this stays zero.
    noise_regen_count: u32,
}

impl MilkdropRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        shaders: &MilkShaders,
    ) -> Result<Self, String> {
        Self::new_with_pipeline_cache(device, queue, width, height, surface_format, shaders, None)
    }

    pub fn new_with_pipeline_cache(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        shaders: &MilkShaders,
        pipeline_cache: Option<&wgpu::PipelineCache>,
    ) -> Result<Self, String> {
        let compiled = compile_milkdrop_shader_bodies(shaders)?;
        Self::new_with_compiled_pipeline_cache(
            device,
            queue,
            width,
            height,
            surface_format,
            shaders,
            &compiled,
            pipeline_cache,
        )
    }

    pub fn new_with_compiled_pipeline_cache(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        shaders: &MilkShaders,
        compiled: &CompiledMilkdropShaderBodies,
        pipeline_cache: Option<&wgpu::PipelineCache>,
    ) -> Result<Self, String> {
        let (w, h) = (width.max(1), height.max(1));
        validate_texture_dims(device.limits().max_texture_dimension_2d, w, h)
            .map_err(|e| e.to_string())?;

        let has_custom_warp = shaders.warp.is_some();
        let has_custom_comp = shaders.comp.is_some();
        let blur_levels = needed_blur_levels(shaders.warp.as_deref(), shaders.comp.as_deref());
        let warp_custom_wgsl = compiled.warp_custom_wgsl.as_str();
        let comp_wgsl = compiled.comp_wgsl.as_str();
        let enhanced_audio_enabled = shaders
            .warp
            .as_deref()
            .is_some_and(uses_enhanced_audio_helpers)
            || shaders
                .comp
                .as_deref()
                .is_some_and(uses_enhanced_audio_helpers);
        let has_named_texture_calls = [shaders.warp.as_deref(), shaders.comp.as_deref()]
            .into_iter()
            .flatten()
            .any(|source| !custom_sampler_names(source).is_empty());
        // Both feature families intentionally share the two fixed named sampler
        // bindings emitted by preprocessing. A mixed shader would silently bind
        // either the dynamic audio rows or the image atlas to both calls, so fail
        // construction with an actionable diagnostic rather than render lies.
        if enhanced_audio_enabled && has_named_texture_calls {
            return Err(
                "enhanced-audio helpers cannot be combined with named texture calls in one preset yet"
                    .into(),
            );
        }

        // Butterchurn starts both feedback surfaces black. Authored waves/shapes
        // seed the feedback naturally; injecting lattice noise here creates false
        // detail and can hide genuinely blank presets.
        let fb_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let feedback_mip_levels = mip_level_count_2d(w, h);

        let tex_a =
            make_tex2d_with_mips(&device, &queue, w, h, fb_usage, feedback_mip_levels, None);
        let tex_b =
            make_tex2d_with_mips(&device, &queue, w, h, fb_usage, feedback_mip_levels, None);
        let view_a = mip_level_view(&tex_a, 0);
        let view_b = mip_level_view(&tex_b, 0);
        let view_a_sample = tex_a.create_view(&Default::default());
        let view_b_sample = tex_b.create_view(&Default::default());
        let feedback_mips_a = mip_chain_views(&tex_a, feedback_mip_levels);
        let feedback_mips_b = mip_chain_views(&tex_b, feedback_mip_levels);

        // Butterchurn's blur pyramid is asymmetric at level 1: horizontal temp
        // is 1/2 resolution, while the finished level is 1/4. Levels 2 and 3
        // are 1/8 and 1/16 respectively for both passes.
        let blur_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let [(bw1, bh1), (bw2, bh2), (bw3, bh3), (btw1, bth1), (btw2, bth2), (btw3, bth3)] =
            blur_dimensions(w, h);

        let blur_levels1 = mip_level_count_2d(bw1, bh1);
        let blur_levels2 = mip_level_count_2d(bw2, bh2);
        let blur_levels3 = mip_level_count_2d(bw3, bh3);
        let blur1 = make_tex2d_with_mips(&device, &queue, bw1, bh1, blur_usage, blur_levels1, None);
        let blur2 = make_tex2d_with_mips(&device, &queue, bw2, bh2, blur_usage, blur_levels2, None);
        let blur3 = make_tex2d_with_mips(&device, &queue, bw3, bh3, blur_usage, blur_levels3, None);
        let view_blur1 = mip_level_view(&blur1, 0);
        let view_blur2 = mip_level_view(&blur2, 0);
        let view_blur3 = mip_level_view(&blur3, 0);
        let view_blur1_sample = blur1.create_view(&Default::default());
        let view_blur2_sample = blur2.create_view(&Default::default());
        let view_blur3_sample = blur3.create_view(&Default::default());
        let blur_mips1 = mip_chain_views(&blur1, blur_levels1);
        let blur_mips2 = mip_chain_views(&blur2, blur_levels2);
        let blur_mips3 = mip_chain_views(&blur3, blur_levels3);

        // Butterchurn generates mipmaps after the horizontal blur too; its
        // vertical shader relies on implicit-LOD sampling of that pyramid.
        // V1 downsamples btemp1 by 2×, so implicit derivatives can select mip1.
        // Levels 2/3 are 1:1 H→V and only ever select LOD0; allocating/blitting
        // their unused tails would add pure GPU work.
        let btemp_levels1 = mip_level_count_2d(btw1, bth1).min(2);
        let btemp_levels2 = 1;
        let btemp_levels3 = 1;
        let btemp1 =
            make_tex2d_with_mips(&device, &queue, btw1, bth1, blur_usage, btemp_levels1, None);
        let btemp2 =
            make_tex2d_with_mips(&device, &queue, btw2, bth2, blur_usage, btemp_levels2, None);
        let btemp3 =
            make_tex2d_with_mips(&device, &queue, btw3, bth3, blur_usage, btemp_levels3, None);
        let view_btemp1 = mip_level_view(&btemp1, 0);
        let view_btemp2 = mip_level_view(&btemp2, 0);
        let view_btemp3 = mip_level_view(&btemp3, 0);
        let view_btemp1_sample = btemp1.create_view(&Default::default());
        let view_btemp2_sample = btemp2.create_view(&Default::default());
        let view_btemp3_sample = btemp3.create_view(&Default::default());
        let btemp_mips1 = mip_chain_views(&btemp1, btemp_levels1);
        let btemp_mips2 = mip_chain_views(&btemp2, btemp_levels2);
        let btemp_mips3 = mip_chain_views(&btemp3, btemp_levels3);

        // Offscreen full-res comp target (Rgba8Unorm). COMP now writes here; the FXAA
        // OUTPUT pass reads it and resolves into the swapchain.
        let comp_tex = make_tex2d(&device, &queue, w, h, blur_usage, None);
        let comp_view = comp_tex.create_view(&Default::default());

        let (named_width, named_height, named_pixels) = if compiled.named_texture_plan.is_empty() {
            (1, 1, vec![0u8, 0, 0, 255])
        } else {
            let atlas = named_texture_resolver().resolve_plan_atlas(&compiled.named_texture_plan);
            (atlas.width, atlas.height, atlas.rgba8)
        };
        // The atlas uses finite gutters between unrelated images. Keep the one
        // safe minification level; deeper whole-atlas mips would bleed adjacent
        // cells together (unlike Butterchurn's isolated image textures).
        let named_texture_levels = mip_level_count_2d(named_width, named_height).min(2);
        let named_texture_atlas = make_tex2d_with_mips(
            &device,
            &queue,
            named_width,
            named_height,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            named_texture_levels,
            Some(&named_pixels),
        );
        let view_named_texture_atlas = named_texture_atlas.create_view(&Default::default());
        let enhanced_fft_texture =
            make_rgba16f_rows_texture(&device, ENHANCED_FFT_BINS as u32, "enhanced-audio-fft");
        let enhanced_wave_texture = make_rgba16f_rows_texture(
            &device,
            ENHANCED_WAVE_SAMPLES as u32,
            "enhanced-audio-waveform",
        );
        let view_enhanced_fft_texture = enhanced_fft_texture.create_view(&Default::default());
        let view_enhanced_wave_texture = enhanced_wave_texture.create_view(&Default::default());
        let enhanced_audio_fft_upload = vec![0u16; ENHANCED_FFT_BINS * 2 * 4];
        let enhanced_audio_wave_upload = vec![0u16; ENHANCED_WAVE_SAMPLES * 2 * 4];
        let upload_enhanced_rows = |texture: &wgpu::Texture, rows: &[u16], width: u32| {
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(rows),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 8),
                    rows_per_image: Some(2),
                },
                wgpu::Extent3d {
                    width,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            );
        };
        upload_enhanced_rows(
            &enhanced_fft_texture,
            &enhanced_audio_fft_upload,
            ENHANCED_FFT_BINS as u32,
        );
        upload_enhanced_rows(
            &enhanced_wave_texture,
            &enhanced_audio_wave_upload,
            ENHANCED_WAVE_SAMPLES as u32,
        );

        // Noise textures — Butterchurn-faithful value/lattice noise (noise.js).
        // LQ 256² zoom1 (random), MQ 256² zoom4 (smoothed), HQ 256² zoom8 (smoothed),
        // LQ-lite 32² zoom1, noisevol_lq 32³ zoom1, noisevol_hq 32³ zoom4 (smoothed).
        let tex_binding = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let mut noise_rng = ButterchurnRng::new(ButterchurnRng::DEFAULT_SEED);
        let mut rng = || noise_rng.next_unit();
        // Match `Noise` construction order in Butterchurn exactly. The stream is
        // shared across all six textures, so merely using the same PRNG is not
        // enough when the allocation order differs.
        let n_lq = create_noise_tex(256, 1, &mut rng);
        let n_lite = create_noise_tex(32, 1, &mut rng);
        let n_mq = create_noise_tex(256, 4, &mut rng);
        let n_hq = create_noise_tex(256, 8, &mut rng);
        let nv_lq = create_noise_vol_tex(32, 1, &mut rng);
        let nv_hq = create_noise_vol_tex(32, 4, &mut rng);

        let noise_lq_levels = mip_level_count_2d(256, 256);
        let noise_lite_levels = mip_level_count_2d(32, 32);
        let noise_lq = make_tex2d_with_mips(
            &device,
            &queue,
            256,
            256,
            tex_binding,
            noise_lq_levels,
            Some(&n_lq),
        );
        let noise_mq = make_tex2d_with_mips(
            &device,
            &queue,
            256,
            256,
            tex_binding,
            noise_lq_levels,
            Some(&n_mq),
        );
        let noise_hq = make_tex2d_with_mips(
            &device,
            &queue,
            256,
            256,
            tex_binding,
            noise_lq_levels,
            Some(&n_hq),
        );
        let noise_lite = make_tex2d_with_mips(
            &device,
            &queue,
            32,
            32,
            tex_binding,
            noise_lite_levels,
            Some(&n_lite),
        );
        let noisevol_lq = make_tex3d(&device, &queue, 32, &nv_lq);
        let noisevol_hq = make_tex3d(&device, &queue, 32, &nv_hq);

        let view_noise_lq = noise_lq.create_view(&Default::default());
        let view_noise_mq = noise_mq.create_view(&Default::default());
        let view_noise_hq = noise_hq.create_view(&Default::default());
        let view_noise_lite = noise_lite.create_view(&Default::default());
        let view_noisevol_lq = noisevol_lq.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let view_noisevol_hq = noisevol_hq.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        // Placeholder view for the unrelated fw/pw/pc sampler slots (2/6/8) — keep
        // a small 2D random texture for those, matching the old behaviour.
        let n_placeholder = noise_bytes(64 * 64);
        let noise2d_levels = mip_level_count_2d(64, 64);
        let noise2d = make_tex2d_with_mips(
            &device,
            &queue,
            64,
            64,
            tex_binding,
            noise2d_levels,
            Some(&n_placeholder),
        );
        let view_noise2d = noise2d.create_view(&Default::default());

        // Sampler
        let linear_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Clamp sampler — MilkDrop "force clamp" (sampler_fc_main/pc) + the blur passes,
        // which must not wrap opposite-edge content into the borders.
        let clamp_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let point_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let point_clamp_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let feedback_mip_blitter =
            wgpu::util::TextureBlitterBuilder::new(&device, wgpu::TextureFormat::Rgba8Unorm)
                .sample_type(wgpu::FilterMode::Linear)
                .build();
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("static-texture-mips"),
            });
            generate_mip_chain(&device, &feedback_mip_blitter, &mut enc, &feedback_mips_a);
            generate_mip_chain(&device, &feedback_mip_blitter, &mut enc, &feedback_mips_b);
            generate_mip_chain(
                &device,
                &feedback_mip_blitter,
                &mut enc,
                &mip_chain_views(&noise_lq, noise_lq_levels),
            );
            generate_mip_chain(
                &device,
                &feedback_mip_blitter,
                &mut enc,
                &mip_chain_views(&noise_mq, noise_lq_levels),
            );
            generate_mip_chain(
                &device,
                &feedback_mip_blitter,
                &mut enc,
                &mip_chain_views(&noise_hq, noise_lq_levels),
            );
            generate_mip_chain(
                &device,
                &feedback_mip_blitter,
                &mut enc,
                &mip_chain_views(&noise_lite, noise_lite_levels),
            );
            generate_mip_chain(
                &device,
                &feedback_mip_blitter,
                &mut enc,
                &mip_chain_views(&noise2d, noise2d_levels),
            );
            generate_mip_chain(
                &device,
                &feedback_mip_blitter,
                &mut enc,
                &mip_chain_views(&named_texture_atlas, named_texture_levels),
            );
            queue.submit(std::iter::once(enc.finish()));
        }

        // UBO
        let perframe_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perframe-ubo"),
            size: std::mem::size_of::<PerFrame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let comp_perframe_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("comp-perframe-ubo"),
            size: std::mem::size_of::<PerFrame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Blur uniform buffers: BlurU { texel: vec4 (1/srcW, 1/srcH, 0, 0), edge: vec4 }.
        // Offsets are in the source texture's texels for each H/V pair. Edge decay
        // (ed1=1-b1ed, ed2=b1ed, ed3=5) fades the blur toward the borders, per Butterchurn.
        let b1ed = 0.25f32; // both jelly presets set b1ed=0.25 (default until parsed)
        let edge = [1.0f32 - b1ed, b1ed, 5.0f32, 0.0f32];
        // BlurU = { texel:vec4, edge:vec4, sb:vec4 } (12 floats / 48 bytes). sb (scale,
        // bias) is rewritten per-frame (offset 32B) from the blur min/max range remap.
        let blur_ubo_contents = |src_w: u32, src_h: u32| -> [f32; 12] {
            [
                1.0 / src_w as f32,
                1.0 / src_h as f32,
                0.0,
                0.0,
                edge[0],
                edge[1],
                edge[2],
                edge[3],
                1.0,
                0.0,
                0.0,
                0.0,
            ] // sb = identity (scale 1, bias 0) until updated
        };
        let blur1_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur1-ubo"),
            contents: bytemuck::cast_slice(&blur_ubo_contents(w, bth1)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blur2_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur2-ubo"),
            contents: bytemuck::cast_slice(&blur_ubo_contents(bw1, bth2)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blur3_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur3-ubo"),
            contents: bytemuck::cast_slice(&blur_ubo_contents(bw2, bth3)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group layouts
        let sampler_bgl = sampler_bgl(&device);
        let perframe_bgl = perframe_bgl(&device);
        let blur_bgl = blur_bgl(&device);
        let warp_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warp-params-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<WarpGpuParams>() as u64
                    ),
                },
                count: None,
            }],
        });
        let warp_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warp-params-ubo"),
            size: std::mem::size_of::<WarpGpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let warp_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("warp-params-bg"),
            layout: &warp_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: warp_params_buf.as_entire_binding(),
            }],
        });

        // Comp keeps the original two groups. Custom warp additionally consumes
        // the default-warp parameter UBO in its vertex shader.
        let comp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("comp-pl"),
            bind_group_layouts: &[Some(&sampler_bgl), Some(&perframe_bgl)],
            immediate_size: 0,
        });
        let warp_custom_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("warp-custom-pl"),
            bind_group_layouts: &[
                Some(&sampler_bgl),
                Some(&perframe_bgl),
                Some(&warp_params_bgl),
            ],
            immediate_size: 0,
        });
        // Pipeline layout for blur
        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur-pl"),
            bind_group_layouts: &[Some(&blur_bgl)],
            immediate_size: 0,
        });

        // Fullscreen vertex shader shared by non-mesh passes.
        let quad_src = include_str!("shaders/quad.wgsl");
        let quad_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quad-vs"),
            source: wgpu::ShaderSource::Wgsl(quad_src.into()),
        });

        // Butterchurn comp mesh: fixed 32x24 topology with a dynamic per-vertex
        // hue field. Custom comp shaders observe it as `hue_shader`.
        let mut initial_comp_verts = Vec::new();
        // Replaced with the preset's actual rand_start before the first draw.
        generate_comp_verts(0.0, [0.0; 4], &mut initial_comp_verts);
        let comp_indices = build_comp_indices();
        let comp_vert_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("comp-mesh-verts"),
            contents: bytemuck::cast_slice(&initial_comp_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let comp_idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("comp-mesh-indices"),
            contents: bytemuck::cast_slice(&comp_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let comp_idx_count = comp_indices.len() as u32;
        let comp_mesh_src = include_str!("shaders/comp_mesh.wgsl");
        let comp_mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("comp-mesh-vs"),
            source: wgpu::ShaderSource::Wgsl(comp_mesh_src.into()),
        });
        let comp_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CompVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        // Comp pipeline
        let comp_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("comp-fs"),
            source: wgpu::ShaderSource::Wgsl(comp_wgsl.into()),
        });
        let make_comp_pipeline = |label: &'static str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&comp_pl),
                vertex: wgpu::VertexState {
                    module: &comp_mesh_mod,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[comp_vbl.clone()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &comp_mod,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: pipeline_cache,
            })
        };
        // The retained COMP texture is Rgba8Unorm. When FXAA is off, COMP
        // writes directly to `surface_format`, which may instead be HDR.
        let comp_pipeline = make_comp_pipeline("comp-pipeline", wgpu::TextureFormat::Rgba8Unorm);
        let comp_direct_pipeline = if surface_format == wgpu::TextureFormat::Rgba8Unorm {
            comp_pipeline.clone()
        } else {
            make_comp_pipeline("comp-direct-pipeline", surface_format)
        };

        // Blur pipeline
        let blur_src = include_str!("shaders/blur.wgsl");
        let blur_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur-fs"),
            source: wgpu::ShaderSource::Wgsl(blur_src.into()),
        });
        let make_blur_pipeline = |label: &str, entry: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&blur_pl),
                vertex: wgpu::VertexState {
                    module: &quad_mod,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blur_mod,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: pipeline_cache,
            })
        };
        let blur_h_pipeline = make_blur_pipeline("blur-h-pipeline", "fs_blur_h");
        let blur_v_pipeline = make_blur_pipeline("blur-v-pipeline", "fs_blur_v");

        // FXAA OUTPUT pass: reads the offscreen comp result, resolves edges → swapchain.
        // Same BGL pattern as blur (texture/sampler/UBO); own self-contained VS+FS.
        // UBO = texsize vec4 (W, H, 1/W, 1/H). Rewritten by `resize` when the
        // internal targets are recreated.
        let fxaa_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fxaa-ubo"),
            contents: bytemuck::cast_slice(&[w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let fxaa_bgl = crate::renderer::blur_bgl(&device); // identical layout: {0:tex D2}{1:sampler}{2:uniform}
        let fxaa_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fxaa-pl"),
            bind_group_layouts: &[Some(&fxaa_bgl)],
            immediate_size: 0,
        });
        let fxaa_src = include_str!("shaders/fxaa.wgsl");
        let fxaa_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fxaa"),
            source: wgpu::ShaderSource::Wgsl(fxaa_src.into()),
        });
        let output_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fxaa-output-pipeline"),
            layout: Some(&fxaa_pl),
            vertex: wgpu::VertexState {
                module: &fxaa_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fxaa_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format, // ← the swapchain format
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: pipeline_cache,
        });
        let fxaa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa-bg"),
            layout: &fxaa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&comp_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fxaa_ubo.as_entire_binding(),
                },
            ],
        });

        // Perframe bind group
        let ubo_binding = (MILKDROP_SAMPLERS.len() * 2) as u32;
        let perframe_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("perframe-bg"),
            layout: &perframe_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: ubo_binding,
                resource: perframe_buf.as_entire_binding(),
            }],
        });
        let comp_perframe_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("comp-perframe-bg"),
            layout: &perframe_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: ubo_binding,
                resource: comp_perframe_buf.as_entire_binding(),
            }],
        });

        // Sampler bind groups (two, one per ping-pong side). Enhanced helpers
        // own the two fixed named sampler slots; normal presets retain the
        // static named-image atlas in both slots.
        let (named_linear_view, named_point_view) = if enhanced_audio_enabled {
            (&view_enhanced_fft_texture, &view_enhanced_wave_texture)
        } else {
            (&view_named_texture_atlas, &view_named_texture_atlas)
        };
        let bg_read_a = build_sampler_bg(
            &device,
            &sampler_bgl,
            &view_a_sample,
            &view_blur1_sample,
            &view_blur2_sample,
            &view_blur3_sample,
            &view_noise2d,
            &view_noise_lq,
            &view_noise_mq,
            &view_noise_hq,
            &view_noise_lite,
            named_linear_view,
            named_point_view,
            enhanced_audio_enabled,
            &view_noisevol_lq,
            &view_noisevol_hq,
            &linear_samp,
            &linear_samp,
            &clamp_samp,
            &point_samp,
            &point_clamp_samp,
        );
        let bg_read_b = build_sampler_bg(
            &device,
            &sampler_bgl,
            &view_b_sample,
            &view_blur1_sample,
            &view_blur2_sample,
            &view_blur3_sample,
            &view_noise2d,
            &view_noise_lq,
            &view_noise_mq,
            &view_noise_hq,
            &view_noise_lite,
            named_linear_view,
            named_point_view,
            enhanced_audio_enabled,
            &view_noisevol_lq,
            &view_noisevol_hq,
            &linear_samp,
            &linear_samp,
            &clamp_samp,
            &point_samp,
            &point_clamp_samp,
        );
        let bg_read_a_clamp = build_sampler_bg(
            &device,
            &sampler_bgl,
            &view_a_sample,
            &view_blur1_sample,
            &view_blur2_sample,
            &view_blur3_sample,
            &view_noise2d,
            &view_noise_lq,
            &view_noise_mq,
            &view_noise_hq,
            &view_noise_lite,
            named_linear_view,
            named_point_view,
            enhanced_audio_enabled,
            &view_noisevol_lq,
            &view_noisevol_hq,
            &clamp_samp,
            &linear_samp,
            &clamp_samp,
            &point_samp,
            &point_clamp_samp,
        );
        let bg_read_b_clamp = build_sampler_bg(
            &device,
            &sampler_bgl,
            &view_b_sample,
            &view_blur1_sample,
            &view_blur2_sample,
            &view_blur3_sample,
            &view_noise2d,
            &view_noise_lq,
            &view_noise_mq,
            &view_noise_hq,
            &view_noise_lite,
            named_linear_view,
            named_point_view,
            enhanced_audio_enabled,
            &view_noisevol_lq,
            &view_noisevol_hq,
            &clamp_samp,
            &linear_samp,
            &clamp_samp,
            &point_samp,
            &point_clamp_samp,
        );

        // Blur bind groups — separable: each level does H (src→temp) then V (temp→level).
        // Clamp sampler avoids wrapping opposite-edge content into the blur near borders.
        let make_blur_bg = |src_view: &wgpu::TextureView, ubo: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&clamp_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ubo.as_entire_binding(),
                    },
                ],
            })
        };
        let blur1_h_bg_a = make_blur_bg(&view_a, &blur1_ubo);
        let blur1_h_bg_b = make_blur_bg(&view_b, &blur1_ubo);
        let blur1_v_bg = make_blur_bg(&view_btemp1_sample, &blur1_ubo);
        let blur2_h_bg = make_blur_bg(&view_blur1_sample, &blur2_ubo);
        let blur2_v_bg = make_blur_bg(&view_btemp2_sample, &blur2_ubo);
        let blur3_h_bg = make_blur_bg(&view_blur2_sample, &blur3_ubo);
        let blur3_v_bg = make_blur_bg(&view_btemp3_sample, &blur3_ubo);

        // ── Warp mesh pipeline (used when no custom warp shader) ─────────────
        // Decay is now per-vertex (vertex buffer attribute 2), so the old decay
        // UBO is gone; the mesh bind group is just {texture, sampler}.
        let warp_mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warp-mesh-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Prebuild both live wrap modes; per-frame EEL selects one without
        // allocating or rebuilding bind groups.
        let make_mesh_bg = |tv: &wgpu::TextureView, mesh_samp: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &warp_mesh_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(tv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(mesh_samp),
                    },
                ],
            })
        };
        let warp_mesh_bg_a = make_mesh_bg(&view_a_sample, &linear_samp);
        let warp_mesh_bg_b = make_mesh_bg(&view_b_sample, &linear_samp);
        let warp_mesh_bg_a_clamp = make_mesh_bg(&view_a_sample, &clamp_samp);
        let warp_mesh_bg_b_clamp = make_mesh_bg(&view_b_sample, &clamp_samp);

        let static_warp_verts = build_static_warp_verts();
        let warp_vert_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("warp-verts"),
            contents: bytemuck::cast_slice(&static_warp_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let warp_indices = build_warp_indices();
        let warp_idx_count = warp_indices.len() as u32;
        let warp_idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("warp-indices"),
            contents: bytemuck::cast_slice(&warp_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let warp_mesh_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("warp-mesh-pl"),
            bind_group_layouts: &[Some(&warp_mesh_bgl), Some(&warp_params_bgl)],
            immediate_size: 0,
        });
        let warp_mesh_src = include_str!("shaders/warp_mesh.wgsl");
        let warp_mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp-mesh"),
            source: wgpu::ShaderSource::Wgsl(warp_mesh_src.into()),
        });
        // WarpVert attributes shared by both warp pipelines: pos@0, uv@1, decay@2.
        let warp_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WarpVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };
        let warp_mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("warp-mesh-pipeline"),
            layout: Some(&warp_mesh_pl),
            vertex: wgpu::VertexState {
                module: &warp_mesh_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[warp_vbl.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &warp_mesh_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: pipeline_cache,
        });

        // ── Custom-warp pipeline: warped mesh VS + the per-preset custom warp FS.
        // Uses sampler + perframe + default-warp parameter layouts so it can
        // calculate equation-free UVs in the VS and sample the MilkDrop texture set.
        let warp_mesh_vs_src = include_str!("shaders/warp_mesh_vs.wgsl");
        let warp_mesh_vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp-mesh-vs"),
            source: wgpu::ShaderSource::Wgsl(warp_mesh_vs_src.into()),
        });
        let warp_custom_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp-custom-fs"),
            source: wgpu::ShaderSource::Wgsl(warp_custom_wgsl.into()),
        });
        let warp_custom_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("warp-custom-pipeline"),
            layout: Some(&warp_custom_pl),
            vertex: wgpu::VertexState {
                module: &warp_mesh_vs_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[warp_vbl.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &warp_custom_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: pipeline_cache,
        });

        // ── Custom-shape pipelines/buffers ───────────────────────────────────
        let shape_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shape-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let border_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("border-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<BorderU>() as u64
                    ),
                },
                count: None,
            }],
        });

        let shapes_src = include_str!("shaders/shapes.wgsl");
        let shapes_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shapes"),
            source: wgpu::ShaderSource::Wgsl(shapes_src.into()),
        });

        let shape_fill_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shape-fill-pl"),
            bind_group_layouts: &[Some(&shape_bgl)],
            immediate_size: 0,
        });
        let shape_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ShapeVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };
        let blend_alpha = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let blend_additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let make_shape_fill = |label: &str, blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shape_fill_pl),
                vertex: wgpu::VertexState {
                    module: &shapes_mod,
                    entry_point: Some("vs_shape"),
                    compilation_options: Default::default(),
                    buffers: &[shape_vbl.clone()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shapes_mod,
                    entry_point: Some("fs_shape"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: pipeline_cache,
            })
        };
        let shapes_fill_pipeline_alpha = make_shape_fill("shape-fill-alpha", blend_alpha);
        let shapes_fill_pipeline_additive = make_shape_fill("shape-fill-additive", blend_additive);

        let border_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shape-border-pl"),
            bind_group_layouts: &[Some(&border_bgl)],
            immediate_size: 0,
        });
        let border_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BorderVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };
        let shapes_border_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shape-border"),
                layout: Some(&border_pl),
                vertex: wgpu::VertexState {
                    module: &shapes_mod,
                    entry_point: Some("vs_border"),
                    compilation_options: Default::default(),
                    buffers: &[border_vbl],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shapes_mod,
                    entry_point: Some("fs_border"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(blend_alpha),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: pipeline_cache,
            });

        let shape_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape-verts"),
            size: (SHAPE_VERT_CAP * std::mem::size_of::<ShapeVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let border_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("border-verts"),
            size: (BORDER_VERT_CAP * std::mem::size_of::<BorderVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Static fan triangulation: [0, k, k+1] for k in 1..=SIDES_MAX.
        let mut fan_idx: Vec<u32> = Vec::with_capacity(SHAPE_FAN_IDX_MAX);
        for k in 1..=(SIDES_MAX as u32) {
            fan_idx.extend_from_slice(&[0, k, k + 1]);
        }
        let shape_idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shape-fan-idx"),
            contents: bytemuck::cast_slice(&fan_idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        // border dyn-offset uniform: per-border color + up-to-4 thick offsets
        let border_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("border-u"),
            size: (BORDER_UNIFORM_SLOTS * 256) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let border_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("border-bg"),
            layout: &border_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &border_uniform_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<BorderU>() as u64),
                }),
            }],
        });
        let make_shape_bg = |tv: &wgpu::TextureView, sampler: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shape-bg"),
                layout: &shape_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(tv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };
        let shape_bg_read_a = make_shape_bg(&view_a_sample, &linear_samp);
        let shape_bg_read_b = make_shape_bg(&view_b_sample, &linear_samp);
        let shape_bg_read_a_clamp = make_shape_bg(&view_a_sample, &clamp_samp);
        let shape_bg_read_b_clamp = make_shape_bg(&view_b_sample, &clamp_samp);

        // One deterministic stream owns the full preset lifecycle. Every EEL
        // pool and both shader random vectors share it, while separate renderer
        // instances remain isolated and reproducible.
        let mut seed_src = String::new();
        for source in [
            shaders.warp.as_deref(),
            shaders.comp.as_deref(),
            shaders.per_frame_init.as_deref(),
            shaders.per_frame.as_deref(),
            shaders.per_pixel.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            seed_src.push_str(source);
            seed_src.push('\n');
        }
        for shape in &shaders.shapes {
            if let Some(source) = shape.per_frame_init.as_deref() {
                seed_src.push_str(source);
            }
            if let Some(source) = shape.per_frame.as_deref() {
                seed_src.push_str(source);
            }
        }
        for wave in &shaders.waves {
            for source in [
                wave.per_frame_init.as_deref(),
                wave.per_frame.as_deref(),
                wave.per_point.as_deref(),
            ]
            .into_iter()
            .flatten()
            {
                seed_src.push_str(source);
            }
        }
        let eel_rng = EelRng::shared(preset_hash64(&seed_src));
        // Butterchurn consumes distinct persistent rand_start/rand_preset
        // vectors before any init equation runs. Keep that lifecycle on the
        // preset-owned stream so every later EEL/shader draw has the same order.
        let rand_start = std::array::from_fn(|_| eel_rng.next_unit() as f32);
        let rand_preset = std::array::from_fn(|_| eel_rng.next_unit() as f32);

        // Preset-wide gmegabuf, shared by every EEL pool (per-frame, per-pixel,
        // each shape, each wave). megabuf is per-pool (private to each EelState).
        let gmegabuf: Arc<Mutex<MegaBuf>> = Arc::new(Mutex::new(MegaBuf::default()));

        // build ShapeRT list from parsed shapes
        let mut shapes: Vec<ShapeRT> = shaders
            .shapes
            .iter()
            .map(|sc| {
                let mut env = Env::new();
                let slots = ShapeEnvSlots::intern(&mut env);
                let reg_slots = std::array::from_fn(|i| env.intern_slot(&format!("reg{i:02}")));
                let q_slots = std::array::from_fn(|i| env.intern_slot(&format!("q{}", i + 1)));
                let t_slots = std::array::from_fn(|i| env.intern_slot(&format!("t{}", i + 1)));
                let prog = sc.per_frame.as_deref().map(EelProgram::parse);
                let references = |name: &str| {
                    prog.as_ref()
                        .is_some_and(|program| program.references_symbol(name))
                };
                let live_reg_indices = (0..100)
                    .filter(|index| references(&format!("reg{index:02}")))
                    .map(|index| index as u8)
                    .collect();
                let live_q_indices = (0..32)
                    .filter(|index| references(&format!("q{}", index + 1)))
                    .map(|index| index as u8)
                    .collect();
                let live_t_indices = (0..8)
                    .filter(|index| references(&format!("t{}", index + 1)))
                    .map(|index| index as u8)
                    .collect();
                let state = EelState::with_shared(gmegabuf.clone(), eel_rng.clone());
                ShapeRT {
                    base: sc.base.clone(),
                    prog,
                    env,
                    reg_slots,
                    q_slots,
                    t_slots,
                    live_reg_indices,
                    live_q_indices,
                    live_t_indices,
                    slots,
                    t_init: [0.0; 8],
                    state,
                }
            })
            .collect();

        // ── Waveform pipelines/buffers ───────────────────────────────────────
        let wave_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wave-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(16),
                },
                count: None,
            }],
        });
        let wave_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wave-pl"),
            bind_group_layouts: &[Some(&wave_bgl)],
            immediate_size: 0,
        });
        let wave_src = include_str!("shaders/wave.wgsl");
        let wave_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wave"),
            source: wgpu::ShaderSource::Wgsl(wave_src.into()),
        });
        let wave_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WaveVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };
        let make_wave_pipeline =
            |label: &str, topo: wgpu::PrimitiveTopology, blend: wgpu::BlendState| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&wave_pl),
                    vertex: wgpu::VertexState {
                        module: &wave_mod,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[wave_vbl.clone()],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &wave_mod,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: topo,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: pipeline_cache,
                })
            };
        let wave_pipeline_lines_alpha = make_wave_pipeline(
            "wave-lines-alpha",
            wgpu::PrimitiveTopology::LineStrip,
            blend_alpha,
        );
        let wave_pipeline_lines_additive = make_wave_pipeline(
            "wave-lines-add",
            wgpu::PrimitiveTopology::LineStrip,
            blend_additive,
        );
        let wave_pipeline_points_alpha = make_wave_pipeline(
            "wave-points-alpha",
            wgpu::PrimitiveTopology::PointList,
            blend_alpha,
        );
        let wave_pipeline_points_additive = make_wave_pipeline(
            "wave-points-add",
            wgpu::PrimitiveTopology::PointList,
            blend_additive,
        );

        let wave_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave-verts"),
            size: (WAVE_VERT_CAP * std::mem::size_of::<WaveVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wave_off_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave-off"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wave_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wave-bg"),
            layout: &wave_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &wave_off_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(16),
                }),
            }],
        });

        // ── Motion-vectors pipeline (LineList, single flat color uniform) ─────
        let mv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mv-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(16),
                },
                count: None,
            }],
        });
        let mv_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mv-pl"),
            bind_group_layouts: &[Some(&mv_bgl)],
            immediate_size: 0,
        });
        let mv_src = include_str!("shaders/motion_vectors.wgsl");
        let mv_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion-vectors"),
            source: wgpu::ShaderSource::Wgsl(mv_src.into()),
        });
        let mv_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MVVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };
        let mv_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motion-vectors"),
            layout: Some(&mv_pl),
            vertex: wgpu::VertexState {
                module: &mv_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[mv_vbl],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mv_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(blend_alpha),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: pipeline_cache,
        });
        let mv_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mv-verts"),
            size: (MV_VERT_CAP * std::mem::size_of::<MVVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mv_color_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mv-color"),
            size: std::mem::size_of::<MVColor>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mv_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mv-bg"),
            layout: &mv_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: mv_color_buf.as_entire_binding(),
            }],
        });

        // ── Frame-border pipeline (TriangleList; reuses border_bgl / BorderU) ─
        let frame_border_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frame-border-pl"),
            bind_group_layouts: &[Some(&border_bgl)],
            immediate_size: 0,
        });
        let frame_border_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BorderVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };
        let frame_border_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("frame-border"),
                layout: Some(&frame_border_pl),
                vertex: wgpu::VertexState {
                    module: &shapes_mod,
                    entry_point: Some("vs_border"),
                    compilation_options: Default::default(),
                    buffers: &[frame_border_vbl],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shapes_mod,
                    entry_point: Some("fs_border"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(blend_alpha),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: pipeline_cache,
            });
        // up to 2 borders (outer + inner), 24 verts each.
        let frame_border_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-border-verts"),
            size: (2 * 24 * std::mem::size_of::<BorderVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // dyn-offset uniform: 2 slots of 256B (outer color in slot 0, inner in slot 1)
        let frame_border_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-border-u"),
            size: 2 * 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_border_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame-border-bg"),
            layout: &border_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &frame_border_uniform_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<BorderU>() as u64),
                }),
            }],
        });

        // ── Darken-center pipeline (TriangleList, per-vertex color, alpha blend) ─
        let darken_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("darken-pl"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let darken_src = include_str!("shaders/darken_center.wgsl");
        let darken_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("darken-center"),
            source: wgpu::ShaderSource::Wgsl(darken_src.into()),
        });
        let darken_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DarkenVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };
        let darken_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("darken-center"),
            layout: Some(&darken_pl),
            vertex: wgpu::VertexState {
                module: &darken_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[darken_vbl],
            },
            fragment: Some(wgpu::FragmentState {
                module: &darken_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(blend_alpha),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: pipeline_cache,
        });
        // 4 fan triangles expanded to a triangle list = 12 verts.
        let darken_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("darken-verts"),
            size: (12 * std::mem::size_of::<DarkenVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut waves: Vec<WaveRT> = shaders
            .waves
            .iter()
            .map(|wd| {
                let mut env = Env::new();
                let slots = WaveEnvSlots::intern(&mut env);
                let reg_slots = std::array::from_fn(|i| env.intern_slot(&format!("reg{i:02}")));
                let q_slots = std::array::from_fn(|i| env.intern_slot(&format!("q{}", i + 1)));
                let t_slots = std::array::from_fn(|i| env.intern_slot(&format!("t{}", i + 1)));
                let per_frame_prog = wd.per_frame.as_deref().map(EelProgram::parse);
                let per_point_prog = wd.per_point.as_deref().map(EelProgram::parse);
                let references = |name: &str| {
                    per_frame_prog
                        .as_ref()
                        .is_some_and(|program| program.references_symbol(name))
                        || per_point_prog
                            .as_ref()
                            .is_some_and(|program| program.references_symbol(name))
                };
                let live_reg_indices = (0..100)
                    .filter(|index| references(&format!("reg{index:02}")))
                    .map(|index| index as u8)
                    .collect();
                let live_q_indices = (0..32)
                    .filter(|index| references(&format!("q{}", index + 1)))
                    .map(|index| index as u8)
                    .collect();
                let live_t_indices = (0..8)
                    .filter(|index| references(&format!("t{}", index + 1)))
                    .map(|index| index as u8)
                    .collect();
                let state = EelState::with_shared(gmegabuf.clone(), eel_rng.clone());
                WaveRT {
                    def: CustomWaveDef {
                        index: wd.index,
                        enabled: wd.enabled,
                        samples: wd.samples,
                        sep: wd.sep,
                        spectrum: wd.spectrum,
                        use_dots: wd.use_dots,
                        draw_thick: wd.draw_thick,
                        additive: wd.additive,
                        scaling: wd.scaling,
                        smoothing: wd.smoothing,
                        r: wd.r,
                        g: wd.g,
                        b: wd.b,
                        a: wd.a,
                        per_frame: wd.per_frame.clone(),
                        per_frame_init: wd.per_frame_init.clone(),
                        per_point: wd.per_point.clone(),
                    },
                    per_frame_prog,
                    per_point_prog,
                    env,
                    reg_slots,
                    q_slots,
                    t_slots,
                    live_reg_indices,
                    live_q_indices,
                    live_t_indices,
                    slots,
                    t_init: [0.0; 8],
                    state,
                    scratch: WaveScratch::default(),
                }
            })
            .collect();

        // EEL2 per-frame equations
        let eel_program = shaders.per_frame.as_deref().map(EelProgram::parse);
        let mut eel_env = Env::new();
        let mut eel_state = EelState::with_shared(gmegabuf.clone(), eel_rng.clone());
        let eel_reg_slots =
            std::array::from_fn(|index| eel_env.intern_slot(&format!("reg{index:02}")));
        let eel_q_slots =
            std::array::from_fn(|index| eel_env.intern_slot(&format!("q{}", index + 1)));
        seed_preset_base_env(&mut eel_env, shaders);
        seed_equation_inputs(&mut eel_env, w, h);

        // Run per-frame INIT equations ONCE before frame 0, into the persistent
        // per-frame env/megabuf. per_frame then sees the initialized vars. We then
        // snapshot q1..q32 so we can RESET them to their post-init values at the
        // top of every frame (Butterchurn's mdVS = {...mdVS, ...mdVSQInit}).
        if let Some(init) = shaders.per_frame_init.as_deref() {
            EelProgram::parse(init).run_with(&mut eel_env, &mut eel_state);
        }
        let q_init: [f64; 32] = std::array::from_fn(|index| eel_env.slot_value(eel_q_slots[index]));

        // Butterchurn executes an initial main frame before custom-wave and
        // custom-shape init programs. It then threads reg00..reg99 through each
        // enabled wave (index order) followed by each enabled shape. This makes
        // init-time q/reg/base reads deterministic and preserves authored state.
        seed_preset_base_env(&mut eel_env, shaders);
        seed_equation_inputs(&mut eel_env, w, h);
        for (slot, value) in eel_q_slots.iter().zip(&q_init) {
            eel_env.set_slot_value(*slot, *value);
        }
        if let Some(program) = &eel_program {
            program.run_with(&mut eel_env, &mut eel_state);
        }
        let q_after_init_frame: [f64; 32] =
            std::array::from_fn(|index| eel_env.slot_value(eel_q_slots[index]));
        let mut init_regs: [f64; 100] =
            std::array::from_fn(|index| eel_env.slot_value(eel_reg_slots[index]));

        for (wave, parsed) in waves.iter_mut().zip(&shaders.waves) {
            if !parsed.enabled {
                continue;
            }
            seed_wave_base_env(&mut wave.env, parsed);
            seed_equation_inputs(&mut wave.env, w, h);
            for (slot, value) in wave.q_slots.iter().zip(&q_after_init_frame) {
                wave.env.set_slot_value(*slot, *value);
            }
            for (slot, value) in wave.reg_slots.iter().zip(&init_regs) {
                wave.env.set_slot_value(*slot, *value);
            }
            if let Some(source) = parsed.per_frame_init.as_deref() {
                EelProgram::parse(source).run_with(&mut wave.env, &mut wave.state);
                init_regs = std::array::from_fn(|index| wave.env.slot_value(wave.reg_slots[index]));
            }
            wave.t_init = std::array::from_fn(|index| wave.env.slot_value(wave.t_slots[index]));
            seed_wave_base_env(&mut wave.env, parsed);
        }
        for (shape, parsed) in shapes.iter_mut().zip(&shaders.shapes) {
            if parsed.base.enabled == 0 {
                continue;
            }
            seed_shape_base_env(&mut shape.env, &parsed.base);
            seed_equation_inputs(&mut shape.env, w, h);
            for (slot, value) in shape.q_slots.iter().zip(&q_after_init_frame) {
                shape.env.set_slot_value(*slot, *value);
            }
            for (slot, value) in shape.reg_slots.iter().zip(&init_regs) {
                shape.env.set_slot_value(*slot, *value);
            }
            if let Some(source) = parsed.per_frame_init.as_deref() {
                EelProgram::parse(source).run_with(&mut shape.env, &mut shape.state);
                init_regs =
                    std::array::from_fn(|index| shape.env.slot_value(shape.reg_slots[index]));
            }
            shape.t_init = std::array::from_fn(|index| shape.env.slot_value(shape.t_slots[index]));
            seed_shape_base_env(&mut shape.env, &parsed.base);
        }
        for (slot, value) in eel_reg_slots.iter().zip(&init_regs) {
            eel_env.set_slot_value(*slot, *value);
        }

        // Per-vertex warp (per_pixel) program + per-frame warp base values.
        let per_pixel_prog = shaders.per_pixel.as_deref().map(EelProgram::parse);
        let warp_state = EelState::with_shared(gmegabuf.clone(), eel_rng.clone());
        let base_warp = WarpBase {
            zoom: shaders.zoom,
            zoomexp: shaders.zoomexp,
            rot: shaders.rot,
            warp: shaders.warp_amount,
            cx: shaders.cx,
            cy: shaders.cy,
            dx: shaders.dx,
            dy: shaders.dy,
            sx: shaders.sx,
            sy: shaders.sy,
            warpscale: shaders.warpscale,
            warpanimspeed: shaders.warpanimspeed,
            decay: shaders.decay,
            wrap: shaders.wrap,
        };
        let mut warp_env = Env::new();
        let warp_slots = WarpEnvSlots::intern(&mut warp_env);
        let warp_reg_slots =
            std::array::from_fn(|index| warp_env.intern_slot(&format!("reg{index:02}")));
        let warp_snapshot = EnvSnapshot::default();

        Ok(Self {
            device,
            queue,
            has_custom_warp,
            has_custom_comp,
            preset_decay: shaders.decay,
            rand_start,
            rand_preset,
            tex_a,
            tex_b,
            view_a,
            view_b,
            view_a_sample,
            view_b_sample,
            feedback_mips_a,
            feedback_mips_b,
            feedback_mip_blitter,
            write_to_a: true,
            feedback_provenance: if shaders.beatdrop_feedback {
                FeedbackProvenance::Beatdrop
            } else {
                FeedbackProvenance::Legacy
            },
            enhanced_audio_config: shaders.enhanced_audio.sanitized(),
            blur1,
            blur2,
            blur3,
            view_blur1,
            view_blur2,
            view_blur3,
            view_blur1_sample,
            view_blur2_sample,
            view_blur3_sample,
            blur_mips1,
            blur_mips2,
            blur_mips3,
            btemp1,
            btemp2,
            btemp3,
            view_btemp1,
            view_btemp2,
            view_btemp3,
            view_btemp1_sample,
            view_btemp2_sample,
            view_btemp3_sample,
            btemp_mips1,
            btemp_mips2,
            btemp_mips3,
            named_texture_atlas,
            view_named_texture_atlas,
            enhanced_audio_enabled,
            enhanced_fft_texture,
            enhanced_wave_texture,
            view_enhanced_fft_texture,
            view_enhanced_wave_texture,
            enhanced_audio_processor: EnhancedAudioProcessor::new(),
            enhanced_audio_fft_upload,
            enhanced_audio_wave_upload,
            enhanced_audio_sample_rate_hz: LEGACY_ASSUMED_SAMPLE_RATE_HZ,
            enhanced_audio_nyquist_hz: LEGACY_ASSUMED_SAMPLE_RATE_HZ * 0.5,
            noise2d,
            noise_lq,
            noise_mq,
            noise_hq,
            noise_lite,
            noisevol_lq,
            noisevol_hq,
            view_noise2d,
            view_noise_lq,
            view_noise_mq,
            view_noise_hq,
            view_noise_lite,
            view_noisevol_lq,
            view_noisevol_hq,
            linear_samp,
            clamp_samp,
            point_samp,
            point_clamp_samp,
            perframe_buf,
            comp_perframe_buf,
            last_comp_perframe: bytemuck::Zeroable::zeroed(),
            blur1_ubo,
            blur2_ubo,
            blur3_ubo,
            warp_custom_pipeline,
            comp_pipeline,
            comp_direct_pipeline,
            comp_vert_buf,
            comp_idx_buf,
            comp_idx_count,
            blur_h_pipeline,
            blur_v_pipeline,
            comp_tex,
            comp_view,
            output_pipeline,
            fxaa_bgl,
            fxaa_ubo,
            fxaa_bg,
            fxaa_enabled: true,
            // Default = Reference profile: native internal resolution, full
            // feedback mip chain. Constructed targets are built at (w, h), so
            // these match the initial allocation exactly (byte-identical path).
            internal_scale: 1.0,
            render_w: w,
            render_h: h,
            feedback_mip_cap: u32::MAX,
            perf_profile: MilkdropPerformanceProfile::Reference,
            warp_mesh_pipeline,
            warp_mesh_bg_a,
            warp_mesh_bg_b,
            warp_mesh_bg_a_clamp,
            warp_mesh_bg_b_clamp,
            warp_mesh_bgl,
            warp_params_buf,
            warp_params_bgl,
            warp_params_bg,
            warp_vert_buf,
            warp_idx_buf,
            warp_idx_count,
            force_cpu_warp_mesh: false,
            sampler_bgl,
            perframe_bgl,
            blur_bgl,
            bg_read_a,
            bg_read_b,
            bg_read_a_clamp,
            bg_read_b_clamp,
            perframe_bg,
            comp_perframe_bg,
            blur1_h_bg_a,
            blur1_h_bg_b,
            blur1_v_bg,
            blur2_h_bg,
            blur2_v_bg,
            blur3_h_bg,
            blur3_v_bg,
            blur_levels,
            last_blur_pass_count: 0,
            eel_program,
            eel_env,
            eel_state,
            eel_rng,
            gmegabuf,
            q_init,
            per_pixel_prog,
            base_warp,
            warp_env,
            warp_slots,
            eel_reg_slots,
            eel_q_slots,
            warp_reg_slots,
            warp_snapshot,
            warp_state,
            scratch: RendererScratch {
                warp_verts: Vec::with_capacity(((GRID_W + 1) * (GRID_H + 1)) as usize),
                comp_verts: Vec::with_capacity(((COMP_GRID_W + 1) * (COMP_GRID_H + 1)) as usize),
                motion_verts: Vec::with_capacity(MV_VERT_CAP),
                darken_verts: Vec::with_capacity(12),
                shape_fill_verts: Vec::new(),
                shape_fill_draws: Vec::new(),
                shape_border_verts: Vec::new(),
                shape_border_draws: Vec::new(),
                wave_verts: Vec::new(),
                wave_draws: Vec::new(),
                shared_shape_fill_verts: Vec::new(),
                shared_wave_verts: Vec::new(),
                frame_border_verts: Vec::with_capacity(48),
                frame_border_draws: Vec::with_capacity(2),
                border_uniform_bytes: Vec::new(),
                frame_border_uniform_bytes: Vec::with_capacity(512),
                custom_wave_draws: Vec::new(),
                basic_wave: BasicWaveScratch::default(),
            },
            geometry_diagnostics: GeometryDiagnosticCollector::default(),
            geometry_stage_readback: None,
            frame_idx: 0,
            start: std::time::Instant::now(),
            time_per_frame: None,
            audio: None,
            audio_att: None,
            freq_spectrum: Vec::new(),
            freq_spectrum_right: Vec::new(),
            frame_random_override: None,
            frame_time_override: None,
            width: w,
            height: h,
            surface_format,

            shapes,
            shapes_fill_pipeline_alpha,
            shapes_fill_pipeline_additive,
            shapes_border_pipeline,
            shape_bgl,
            border_bgl,
            shape_vert_buf,
            shape_idx_buf,
            border_vert_buf,
            border_uniform_buf,
            border_bg,
            shape_bg_read_a,
            shape_bg_read_b,
            shape_bg_read_a_clamp,
            shape_bg_read_b_clamp,

            waves,
            wave_pipeline_lines_alpha,
            wave_pipeline_lines_additive,
            wave_pipeline_points_alpha,
            wave_pipeline_points_additive,
            wave_bgl,
            wave_vert_buf,
            wave_off_buf,
            wave_bg,
            custom_wave_adaptive_lod: true,

            bw_mode: shaders.wave_mode,
            bw_x: shaders.wave_x,
            bw_y: shaders.wave_y,
            bw_r: shaders.wave_r,
            bw_g: shaders.wave_g,
            bw_b: shaders.wave_b,
            bw_a: shaders.wave_a,
            bw_mystery: shaders.wave_mystery,
            bw_scale: shaders.wave_scale,
            bw_smoothing: shaders.wave_smoothing,
            bw_dots: shaders.wave_dots,
            bw_thick: shaders.wave_thick,
            bw_additive: shaders.additive_wave,
            bw_brighten: shaders.wave_brighten,
            bw_modalphavol: shaders.modwavealphabyvolume,
            bw_modalphastart: shaders.modwavealphastart,
            bw_modalphaend: shaders.modwavealphaend,

            comp_gamma_adj: shaders.gamma_adj,
            comp_fshader: shaders.fshader,
            echo_zoom: shaders.echo_zoom,
            echo_alpha: shaders.echo_alpha,
            echo_orient: shaders.echo_orient,
            comp_brighten: shaders.brighten,
            comp_darken: shaders.darken,
            comp_solarize: shaders.solarize,
            comp_invert: shaders.invert,

            mv_pipeline,
            mv_bgl,
            mv_vert_buf,
            mv_color_buf,
            mv_bg,
            mv_on: shaders.mv_on,
            mv_x: shaders.mv_x,
            mv_y: shaders.mv_y,
            mv_dx: shaders.mv_dx,
            mv_dy: shaders.mv_dy,
            mv_l: shaders.mv_l,
            mv_r: shaders.mv_r,
            mv_g: shaders.mv_g,
            mv_b: shaders.mv_b,
            mv_a: shaders.mv_a,

            frame_border_pipeline,
            frame_border_vert_buf,
            frame_border_uniform_buf,
            frame_border_bg,
            ob_size: shaders.ob_size,
            ob_r: shaders.ob_r,
            ob_g: shaders.ob_g,
            ob_b: shaders.ob_b,
            ob_a: shaders.ob_a,
            ib_size: shaders.ib_size,
            ib_r: shaders.ib_r,
            ib_g: shaders.ib_g,
            ib_b: shaders.ib_b,
            ib_a: shaders.ib_a,

            darken_pipeline,
            darken_vert_buf,
            darken_center: shaders.darken_center,
            vol_prev: 0.0,

            b1n: shaders.b1n,
            b1x: shaders.b1x,
            b1ed: shaders.b1ed,
            b2n: shaders.b2n,
            b2x: shaders.b2x,
            b3n: shaders.b3n,
            b3x: shaders.b3x,

            wave_l: Vec::new(),
            wave_r: Vec::new(),
            noise_regen_count: 0,
        })
    }

    /// Number of feedback-seed noise generations. This remains zero because
    /// OjoDrop now follows Butterchurn's black feedback initialization.
    pub fn noise_regen_count(&self) -> u32 {
        self.noise_regen_count
    }

    /// Current backing-target dimensions. Hosts use this to decide whether an
    /// explicitly retried resize still needs to run after a previous allocation
    /// failure; it intentionally reports the last successfully committed size.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Opt in to BeatDrop-style feedback provenance for this renderer.
    ///
    /// `false` is the default and is byte-for-byte the established OjoDrop
    /// ordering. This setter deliberately controls provenance only; native
    /// two-state morphing is requested separately through
    /// [`Self::render_shared_feedback`] by the runtime that owns both states.
    pub fn set_beatdrop_feedback(&mut self, enabled: bool) {
        self.feedback_provenance = if enabled {
            FeedbackProvenance::Beatdrop
        } else {
            FeedbackProvenance::Legacy
        };
    }

    /// Current feedback provenance selection.
    pub fn feedback_provenance(&self) -> FeedbackProvenance {
        self.feedback_provenance
    }

    /// Update the preset-scoped BeatDrop FFT follower controls without touching
    /// player-level gain/noise-floor defaults. Runtime base-value updates call
    /// this alongside [`Self::set_beatdrop_feedback`].
    pub fn set_enhanced_audio_config(&mut self, config: EnhancedAudioConfig) {
        let sanitized = config.sanitized();
        self.enhanced_audio_config.fft_attack = sanitized.fft_attack;
        self.enhanced_audio_config.fft_decay = sanitized.fft_decay;
    }

    /// Current effective enhanced-audio configuration. `fft_scaling` and
    /// `fft_noise_floor` remain renderer/player defaults unless explicitly
    /// changed by a future host control.
    pub fn enhanced_audio_config(&self) -> EnhancedAudioConfig {
        self.enhanced_audio_config
    }

    /// Whether every authored overlay can be drawn into the shared post-warp
    /// pass with a single complementary-opacity weight. Waves and static,
    /// untextured, borderless shapes meet that contract. Textured/dynamic shapes
    /// sample a renderer-local feedback page, while vectors/darken/frame-borders
    /// have ordering or uniform requirements that need a dedicated overlay
    /// target; keep those on the caller's ordinary fallback for now.
    fn shared_feedback_overlays_are_supported(&self) -> bool {
        if self.mv_on
            || self.darken_center
            || self.ob_a != 0.0
            || self.ib_a != 0.0
        {
            return false;
        }
        self.shapes.iter().all(|shape| {
            shape.base.enabled == 0
                || (shape.prog.is_none()
                    && shape.base.textured == 0
                    && shape.base.border_a <= 0.0)
        })
    }

    fn shared_feedback_has_supported_overlays(&self) -> bool {
        self.bw_a != 0.0
            || !self.waves.is_empty()
            || self.shapes.iter().any(|shape| shape.base.enabled != 0)
    }

    /// Built-in COMP controls are interpolated from their evaluated UBOs and
    /// waves are converted to weighted vertex colors after their frame equations
    /// run. Controls that branch in COMP or could emit an unsupported overlay
    /// are rejected instead of being fractionally blended.
    fn shared_feedback_per_frame_avoids_unsupported_overlays(&self) -> bool {
        const UNSUPPORTED_SYMBOLS: &[&str] = &[
            "b1n",
            "b1x",
            "b1ed",
            "b2n",
            "b2x",
            "b3n",
            "b3x",
            "mv_x",
            "mv_y",
            "mv_dx",
            "mv_dy",
            "mv_l",
            "mv_r",
            "mv_g",
            "mv_b",
            "mv_a",
            "ob_size",
            "ob_r",
            "ob_g",
            "ob_b",
            "ob_a",
            "ib_size",
            "ib_r",
            "ib_g",
            "ib_b",
            "ib_a",
            "darken_center",
            "fshader",
            "echo_orient",
            "brighten",
            "darken",
            "solarize",
            "invert",
        ];
        self.eel_program.as_ref().map_or(true, |program| {
            !UNSUPPORTED_SYMBOLS
                .iter()
                .any(|symbol| program.references_symbol(symbol))
        })
    }

    /// These controls branch in the default COMP shader. Keeping both endpoints
    /// identical avoids the false middle state produced by raw UBO interpolation;
    /// hue shader is simply outside this bounded path because its mesh colors
    /// are renderer-local `rand_start` products.
    fn shared_feedback_discrete_comp_matches(&self, other: &Self) -> bool {
        self.comp_fshader.is_finite()
            && other.comp_fshader.is_finite()
            && self.comp_fshader.abs() <= 0.001
            && other.comp_fshader.abs() <= 0.001
            && self.echo_orient.is_finite()
            && other.echo_orient.is_finite()
            && self.echo_orient == other.echo_orient
            && self.comp_brighten == other.comp_brighten
            && self.comp_darken == other.comp_darken
            && self.comp_solarize == other.comp_solarize
            && self.comp_invert == other.comp_invert
    }

    /// Report whether this exact pair can use the bounded native
    /// shared-feedback path. The caller must retain its ordinary transition for
    /// every non-supported result; this does not claim arbitrary shader-pair
    /// parity.
    pub fn shared_feedback_support(&self, other: &Self) -> SharedFeedbackSupport {
        if std::ptr::eq(self, other) {
            return SharedFeedbackSupport::SameRenderer;
        }
        if self.device.as_ref() != other.device.as_ref() {
            return SharedFeedbackSupport::DifferentDevice;
        }
        if self.surface_format != other.surface_format {
            return SharedFeedbackSupport::DifferentSurfaceFormat;
        }
        if (self.width, self.height, self.render_w, self.render_h)
            != (other.width, other.height, other.render_w, other.render_h)
        {
            return SharedFeedbackSupport::DifferentDimensions;
        }
        if self.has_custom_warp
            || self.has_custom_comp
            || other.has_custom_warp
            || other.has_custom_comp
        {
            return SharedFeedbackSupport::CustomShadersUnsupported;
        }
        if !self.shared_feedback_overlays_are_supported()
            || !other.shared_feedback_overlays_are_supported()
        {
            return SharedFeedbackSupport::VisibleOverlaysUnsupported;
        }
        if !self.shared_feedback_per_frame_avoids_unsupported_overlays()
            || !other.shared_feedback_per_frame_avoids_unsupported_overlays()
        {
            return SharedFeedbackSupport::PerFrameCompOrOverlayUnsupported;
        }
        if !self.shared_feedback_discrete_comp_matches(other) {
            return SharedFeedbackSupport::DiscreteCompUnsupported;
        }
        if self.shared_feedback_has_supported_overlays()
            || other.shared_feedback_has_supported_overlays()
        {
            SharedFeedbackSupport::UntexturedOverlaysInterpolatedComp
        } else {
            SharedFeedbackSupport::FeedbackOnlyInterpolatedComp
        }
    }

    /// Boolean shorthand for [`Self::shared_feedback_support`].
    pub fn supports_shared_feedback(&self, other: &Self) -> bool {
        self.shared_feedback_support(other).is_supported()
    }

    /// Copy both ping-pong feedback pages from `other`, preserving its current
    /// write side. This lets a newly prepared renderer enter a native morph with
    /// a single history, and lets the runtime promote the current target during
    /// an interrupted morph without a feedback reset.
    ///
    /// The operation is intentionally limited to already compatible renderer
    /// pairs. It returns `false` without encoding work when dimensions, device,
    /// or format differ.
    pub fn seed_feedback_from(&mut self, other: &Self) -> bool {
        if std::ptr::eq(self, other)
            || self.device.as_ref() != other.device.as_ref()
            || self.surface_format != other.surface_format
            || (self.width, self.height, self.render_w, self.render_h)
                != (other.width, other.height, other.render_w, other.render_h)
        {
            return false;
        }

        let extent = wgpu::Extent3d {
            width: self.render_w,
            height: self.render_h,
            depth_or_array_layers: 1,
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("milkdrop-shared-feedback-seed"),
            });
        encoder.copy_texture_to_texture(other.tex_a.as_image_copy(), self.tex_a.as_image_copy(), extent);
        encoder.copy_texture_to_texture(other.tex_b.as_image_copy(), self.tex_b.as_image_copy(), extent);
        generate_mip_chain(
            &self.device,
            &self.feedback_mip_blitter,
            &mut encoder,
            &self.feedback_mips_a,
        );
        generate_mip_chain(
            &self.device,
            &self.feedback_mip_blitter,
            &mut encoder,
            &self.feedback_mips_b,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        self.write_to_a = other.write_to_a;
        true
    }

    /// Copy frame-varying host inputs into an incoming state before its hidden
    /// equation advance. Preset state (EEL environments, RNG, shape/wave pools)
    /// remains independent; only the clock and audio source are shared.
    fn mirror_shared_feedback_inputs_from(&mut self, outgoing: &Self) {
        self.start = outgoing.start;
        self.frame_idx = outgoing.frame_idx;
        self.time_per_frame = outgoing.time_per_frame;
        self.frame_time_override = outgoing.frame_time_override;
        self.audio = outgoing.audio;
        self.audio_att = outgoing.audio_att;
        self.wave_l.clone_from(&outgoing.wave_l);
        self.wave_r.clone_from(&outgoing.wave_r);
        self.freq_spectrum.clone_from(&outgoing.freq_spectrum);
        self.freq_spectrum_right
            .clone_from(&outgoing.freq_spectrum_right);
    }

    /// Replace live numeric/bool base values without rebuilding renderer-owned
    /// programs, equation state, buffers, or feedback textures.
    pub fn apply_base_vals(&mut self, values: &MilkBaseVals) {
        self.base_warp = WarpBase {
            zoom: values.zoom,
            zoomexp: values.zoomexp,
            rot: values.rot,
            warp: values.warp_amount,
            cx: values.cx,
            cy: values.cy,
            dx: values.dx,
            dy: values.dy,
            sx: values.sx,
            sy: values.sy,
            warpscale: values.warpscale,
            warpanimspeed: values.warpanimspeed,
            decay: values.decay,
            wrap: values.wrap,
        };
        self.bw_mode = values.wave_mode;
        self.bw_x = values.wave_x;
        self.bw_y = values.wave_y;
        self.bw_r = values.wave_r;
        self.bw_g = values.wave_g;
        self.bw_b = values.wave_b;
        self.bw_a = values.wave_a;
        self.bw_mystery = values.wave_mystery;
        self.bw_scale = values.wave_scale;
        self.bw_smoothing = values.wave_smoothing;
        self.bw_dots = values.wave_dots;
        self.bw_thick = values.wave_thick;
        self.bw_additive = values.additive_wave;
        self.bw_brighten = values.wave_brighten;
        self.bw_modalphavol = values.modwavealphabyvolume;
        self.bw_modalphastart = values.modwavealphastart;
        self.bw_modalphaend = values.modwavealphaend;
        self.comp_gamma_adj = values.gamma_adj;
        self.comp_fshader = values.fshader;
        self.echo_zoom = values.echo_zoom;
        self.echo_alpha = values.echo_alpha;
        self.echo_orient = values.echo_orient;
        self.comp_brighten = values.brighten;
        self.comp_darken = values.darken;
        self.comp_solarize = values.solarize;
        self.comp_invert = values.invert;
        self.mv_on = values.mv_on;
        self.mv_x = values.mv_x;
        self.mv_y = values.mv_y;
        self.mv_dx = values.mv_dx;
        self.mv_dy = values.mv_dy;
        self.mv_l = values.mv_l;
        self.mv_r = values.mv_r;
        self.mv_g = values.mv_g;
        self.mv_b = values.mv_b;
        self.mv_a = values.mv_a;
        self.ob_size = values.ob_size;
        self.ob_r = values.ob_r;
        self.ob_g = values.ob_g;
        self.ob_b = values.ob_b;
        self.ob_a = values.ob_a;
        self.ib_size = values.ib_size;
        self.ib_r = values.ib_r;
        self.ib_g = values.ib_g;
        self.ib_b = values.ib_b;
        self.ib_a = values.ib_a;
        self.darken_center = values.darken_center;

        for (name, value) in [
            ("echo_zoom", values.echo_zoom),
            ("echo_alpha", values.echo_alpha),
            ("echo_orient", values.echo_orient),
            ("gamma", values.gamma_adj),
            ("gammaadj", values.gamma_adj),
        ] {
            self.eel_env.insert(name.into(), value as f64);
        }
    }

    /// Resize the GPU targets without rebuilding the preset runtime.
    ///
    /// Shader programs, EEL environments, audio state, and frame counters stay
    /// intact. The feedback ping-pong textures are recreated black, then the
    /// overlapping region of the previous feedback is copied into them before
    /// mip generation.
    pub fn resize(&mut self, width: u32, height: u32) {
        if let Err(e) = self.try_resize(width, height) {
            log::warn!("milkdrop resize to {width}x{height} rejected: {e}");
        }
    }

    /// Fallible resize: validates the requested dimensions with checked arithmetic
    /// against the device `max_texture_dimension_2d` + a memory budget BEFORE
    /// recreating any texture, returning a [`DimensionError`] on rejection.
    pub fn try_resize(&mut self, width: u32, height: u32) -> Result<(), DimensionError> {
        let (w, h) = (width.max(1), height.max(1));
        if self.width == w && self.height == h {
            return Ok(());
        }
        // Output-size change: crop-preserve the feedback (legacy behaviour). The
        // internal targets are (re)built at `internal_scale` inside the rebuild.
        self.rebuild_render_targets(w, h, CarryMode::Crop)
    }

    /// Rebuild every size- and mip-dependent GPU target (feedback ping-pong,
    /// blur pyramid, comp) plus the bind groups and UBOs that reference them, at
    /// the given OUTPUT size and the current `internal_scale`/`feedback_mip_cap`.
    ///
    /// The internal render dimensions are `scaled_internal_dims(out, scale)`;
    /// they are bound to `(w, h)` here so the whole build below sizes the
    /// internal targets from the scaled dimensions. `carry` chooses how the
    /// previous feedback is brought forward. At `internal_scale == 1.0`,
    /// `feedback_mip_cap == u32::MAX`, and `CarryMode::Crop`, this reproduces the
    /// pre-scale resize exactly.
    fn rebuild_render_targets(
        &mut self,
        out_w: u32,
        out_h: u32,
        carry: CarryMode,
    ) -> Result<(), DimensionError> {
        let (out_w, out_h) = (out_w.max(1), out_h.max(1));
        validate_texture_dims(self.device.limits().max_texture_dimension_2d, out_w, out_h)?;
        // Internal render size (== output at scale 1.0). All internal targets
        // below are sized from these; the final comp/FXAA pass upscales to output.
        let (w, h) = scaled_internal_dims(out_w, out_h, self.internal_scale);
        let (old_rw, old_rh) = (self.render_w, self.render_h);

        let device = self.device.clone();
        let queue = self.queue.clone();

        let fb_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        // Cap the feedback mip chain (feedback only — the blur pyramid below is
        // independent). Full chain (`u32::MAX` cap) is byte-identical.
        let feedback_mip_levels = mip_level_count_2d(w, h).min(self.feedback_mip_cap).max(1);
        let tex_a =
            make_tex2d_with_mips(&device, &queue, w, h, fb_usage, feedback_mip_levels, None);
        let tex_b =
            make_tex2d_with_mips(&device, &queue, w, h, fb_usage, feedback_mip_levels, None);
        let view_a = mip_level_view(&tex_a, 0);
        let view_b = mip_level_view(&tex_b, 0);
        let view_a_sample = tex_a.create_view(&Default::default());
        let view_b_sample = tex_b.create_view(&Default::default());
        let feedback_mips_a = mip_chain_views(&tex_a, feedback_mip_levels);
        let feedback_mips_b = mip_chain_views(&tex_b, feedback_mip_levels);

        let blur_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let [(bw1, bh1), (bw2, bh2), (bw3, bh3), (btw1, bth1), (btw2, bth2), (btw3, bth3)] =
            blur_dimensions(w, h);
        let blur_levels1 = mip_level_count_2d(bw1, bh1);
        let blur_levels2 = mip_level_count_2d(bw2, bh2);
        let blur_levels3 = mip_level_count_2d(bw3, bh3);

        let blur1 = make_tex2d_with_mips(&device, &queue, bw1, bh1, blur_usage, blur_levels1, None);
        let blur2 = make_tex2d_with_mips(&device, &queue, bw2, bh2, blur_usage, blur_levels2, None);
        let blur3 = make_tex2d_with_mips(&device, &queue, bw3, bh3, blur_usage, blur_levels3, None);
        let view_blur1 = mip_level_view(&blur1, 0);
        let view_blur2 = mip_level_view(&blur2, 0);
        let view_blur3 = mip_level_view(&blur3, 0);
        let view_blur1_sample = blur1.create_view(&Default::default());
        let view_blur2_sample = blur2.create_view(&Default::default());
        let view_blur3_sample = blur3.create_view(&Default::default());
        let blur_mips1 = mip_chain_views(&blur1, blur_levels1);
        let blur_mips2 = mip_chain_views(&blur2, blur_levels2);
        let blur_mips3 = mip_chain_views(&blur3, blur_levels3);

        let btemp_levels1 = mip_level_count_2d(btw1, bth1).min(2);
        let btemp_levels2 = 1;
        let btemp_levels3 = 1;
        let btemp1 =
            make_tex2d_with_mips(&device, &queue, btw1, bth1, blur_usage, btemp_levels1, None);
        let btemp2 =
            make_tex2d_with_mips(&device, &queue, btw2, bth2, blur_usage, btemp_levels2, None);
        let btemp3 =
            make_tex2d_with_mips(&device, &queue, btw3, bth3, blur_usage, btemp_levels3, None);
        let view_btemp1 = mip_level_view(&btemp1, 0);
        let view_btemp2 = mip_level_view(&btemp2, 0);
        let view_btemp3 = mip_level_view(&btemp3, 0);
        let view_btemp1_sample = btemp1.create_view(&Default::default());
        let view_btemp2_sample = btemp2.create_view(&Default::default());
        let view_btemp3_sample = btemp3.create_view(&Default::default());
        let btemp_mips1 = mip_chain_views(&btemp1, btemp_levels1);
        let btemp_mips2 = mip_chain_views(&btemp2, btemp_levels2);
        let btemp_mips3 = mip_chain_views(&btemp3, btemp_levels3);

        let comp_tex = make_tex2d(&device, &queue, w, h, blur_usage, None);
        let comp_view = comp_tex.create_view(&Default::default());

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("milkdrop-resize-feedback-copy"),
        });
        // When the internal dimensions are unchanged (e.g. a mip-cap change), a
        // full-region copy is byte-preserving. A `Crop` carry preserves the
        // top-left overlap (MilkDrop's legacy output-resize). A `Resample` carry
        // on an actual dimension change UV-resamples the whole page so the frame
        // survives an internal-scale transition without a black border.
        let dims_unchanged = old_rw == w && old_rh == h;
        if dims_unchanged || carry == CarryMode::Crop {
            let copy_w = old_rw.min(w);
            let copy_h = old_rh.min(h);
            if copy_w > 0 && copy_h > 0 {
                let extent = wgpu::Extent3d {
                    width: copy_w,
                    height: copy_h,
                    depth_or_array_layers: 1,
                };
                enc.copy_texture_to_texture(
                    self.tex_a.as_image_copy(),
                    tex_a.as_image_copy(),
                    extent,
                );
                enc.copy_texture_to_texture(
                    self.tex_b.as_image_copy(),
                    tex_b.as_image_copy(),
                    extent,
                );
            }
        } else {
            // UV-space bilinear resample of BOTH ping-pong pages: a fullscreen
            // blit from the old level-0 page (any size) into the new level-0
            // page samples the full 0..1 UV range, so no border is left blank.
            let old_view_a = mip_level_view(&self.tex_a, 0);
            let old_view_b = mip_level_view(&self.tex_b, 0);
            self.feedback_mip_blitter
                .copy(&device, &mut enc, &old_view_a, &view_a);
            self.feedback_mip_blitter
                .copy(&device, &mut enc, &old_view_b, &view_b);
        }
        generate_mip_chain(
            &device,
            &self.feedback_mip_blitter,
            &mut enc,
            &feedback_mips_a,
        );
        generate_mip_chain(
            &device,
            &self.feedback_mip_blitter,
            &mut enc,
            &feedback_mips_b,
        );
        queue.submit(std::iter::once(enc.finish()));

        let blur_texel = |src_w: u32, src_h: u32| -> [f32; 4] {
            [1.0 / src_w as f32, 1.0 / src_h as f32, 0.0, 0.0]
        };
        queue.write_buffer(
            &self.blur1_ubo,
            0,
            bytemuck::cast_slice(&blur_texel(w, bth1)),
        );
        queue.write_buffer(
            &self.blur2_ubo,
            0,
            bytemuck::cast_slice(&blur_texel(bw1, bth2)),
        );
        queue.write_buffer(
            &self.blur3_ubo,
            0,
            bytemuck::cast_slice(&blur_texel(bw2, bth3)),
        );
        queue.write_buffer(
            &self.fxaa_ubo,
            0,
            bytemuck::cast_slice(&[w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32]),
        );

        let (named_linear_view, named_point_view) = if self.enhanced_audio_enabled {
            (
                &self.view_enhanced_fft_texture,
                &self.view_enhanced_wave_texture,
            )
        } else {
            (
                &self.view_named_texture_atlas,
                &self.view_named_texture_atlas,
            )
        };
        let bg_read_a = build_sampler_bg(
            &device,
            &self.sampler_bgl,
            &view_a_sample,
            &view_blur1_sample,
            &view_blur2_sample,
            &view_blur3_sample,
            &self.view_noise2d,
            &self.view_noise_lq,
            &self.view_noise_mq,
            &self.view_noise_hq,
            &self.view_noise_lite,
            named_linear_view,
            named_point_view,
            self.enhanced_audio_enabled,
            &self.view_noisevol_lq,
            &self.view_noisevol_hq,
            &self.linear_samp,
            &self.linear_samp,
            &self.clamp_samp,
            &self.point_samp,
            &self.point_clamp_samp,
        );
        let bg_read_b = build_sampler_bg(
            &device,
            &self.sampler_bgl,
            &view_b_sample,
            &view_blur1_sample,
            &view_blur2_sample,
            &view_blur3_sample,
            &self.view_noise2d,
            &self.view_noise_lq,
            &self.view_noise_mq,
            &self.view_noise_hq,
            &self.view_noise_lite,
            named_linear_view,
            named_point_view,
            self.enhanced_audio_enabled,
            &self.view_noisevol_lq,
            &self.view_noisevol_hq,
            &self.linear_samp,
            &self.linear_samp,
            &self.clamp_samp,
            &self.point_samp,
            &self.point_clamp_samp,
        );
        let bg_read_a_clamp = build_sampler_bg(
            &device,
            &self.sampler_bgl,
            &view_a_sample,
            &view_blur1_sample,
            &view_blur2_sample,
            &view_blur3_sample,
            &self.view_noise2d,
            &self.view_noise_lq,
            &self.view_noise_mq,
            &self.view_noise_hq,
            &self.view_noise_lite,
            named_linear_view,
            named_point_view,
            self.enhanced_audio_enabled,
            &self.view_noisevol_lq,
            &self.view_noisevol_hq,
            &self.clamp_samp,
            &self.linear_samp,
            &self.clamp_samp,
            &self.point_samp,
            &self.point_clamp_samp,
        );
        let bg_read_b_clamp = build_sampler_bg(
            &device,
            &self.sampler_bgl,
            &view_b_sample,
            &view_blur1_sample,
            &view_blur2_sample,
            &view_blur3_sample,
            &self.view_noise2d,
            &self.view_noise_lq,
            &self.view_noise_mq,
            &self.view_noise_hq,
            &self.view_noise_lite,
            named_linear_view,
            named_point_view,
            self.enhanced_audio_enabled,
            &self.view_noisevol_lq,
            &self.view_noisevol_hq,
            &self.clamp_samp,
            &self.linear_samp,
            &self.clamp_samp,
            &self.point_samp,
            &self.point_clamp_samp,
        );

        let make_blur_bg = |src_view: &wgpu::TextureView, ubo: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.clamp_samp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ubo.as_entire_binding(),
                    },
                ],
            })
        };
        let blur1_h_bg_a = make_blur_bg(&view_a, &self.blur1_ubo);
        let blur1_h_bg_b = make_blur_bg(&view_b, &self.blur1_ubo);
        let blur1_v_bg = make_blur_bg(&view_btemp1_sample, &self.blur1_ubo);
        let blur2_h_bg = make_blur_bg(&view_blur1_sample, &self.blur2_ubo);
        let blur2_v_bg = make_blur_bg(&view_btemp2_sample, &self.blur2_ubo);
        let blur3_h_bg = make_blur_bg(&view_blur2_sample, &self.blur3_ubo);
        let blur3_v_bg = make_blur_bg(&view_btemp3_sample, &self.blur3_ubo);

        let fxaa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa-bg"),
            layout: &self.fxaa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&comp_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.fxaa_ubo.as_entire_binding(),
                },
            ],
        });

        let make_mesh_bg = |tv: &wgpu::TextureView, sampler: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.warp_mesh_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(tv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };
        let warp_mesh_bg_a = make_mesh_bg(&view_a_sample, &self.linear_samp);
        let warp_mesh_bg_b = make_mesh_bg(&view_b_sample, &self.linear_samp);
        let warp_mesh_bg_a_clamp = make_mesh_bg(&view_a_sample, &self.clamp_samp);
        let warp_mesh_bg_b_clamp = make_mesh_bg(&view_b_sample, &self.clamp_samp);

        let make_shape_bg = |tv: &wgpu::TextureView, sampler: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shape-bg"),
                layout: &self.shape_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(tv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };
        let shape_bg_read_a = make_shape_bg(&view_a_sample, &self.linear_samp);
        let shape_bg_read_b = make_shape_bg(&view_b_sample, &self.linear_samp);
        let shape_bg_read_a_clamp = make_shape_bg(&view_a_sample, &self.clamp_samp);
        let shape_bg_read_b_clamp = make_shape_bg(&view_b_sample, &self.clamp_samp);

        self.tex_a = tex_a;
        self.tex_b = tex_b;
        self.view_a = view_a;
        self.view_b = view_b;
        self.view_a_sample = view_a_sample;
        self.view_b_sample = view_b_sample;
        self.feedback_mips_a = feedback_mips_a;
        self.feedback_mips_b = feedback_mips_b;
        self.blur1 = blur1;
        self.blur2 = blur2;
        self.blur3 = blur3;
        self.view_blur1 = view_blur1;
        self.view_blur2 = view_blur2;
        self.view_blur3 = view_blur3;
        self.view_blur1_sample = view_blur1_sample;
        self.view_blur2_sample = view_blur2_sample;
        self.view_blur3_sample = view_blur3_sample;
        self.blur_mips1 = blur_mips1;
        self.blur_mips2 = blur_mips2;
        self.blur_mips3 = blur_mips3;
        self.btemp1 = btemp1;
        self.btemp2 = btemp2;
        self.btemp3 = btemp3;
        self.view_btemp1 = view_btemp1;
        self.view_btemp2 = view_btemp2;
        self.view_btemp3 = view_btemp3;
        self.view_btemp1_sample = view_btemp1_sample;
        self.view_btemp2_sample = view_btemp2_sample;
        self.view_btemp3_sample = view_btemp3_sample;
        self.btemp_mips1 = btemp_mips1;
        self.btemp_mips2 = btemp_mips2;
        self.btemp_mips3 = btemp_mips3;
        self.comp_tex = comp_tex;
        self.comp_view = comp_view;
        self.fxaa_bg = fxaa_bg;
        self.warp_mesh_bg_a = warp_mesh_bg_a;
        self.warp_mesh_bg_b = warp_mesh_bg_b;
        self.warp_mesh_bg_a_clamp = warp_mesh_bg_a_clamp;
        self.warp_mesh_bg_b_clamp = warp_mesh_bg_b_clamp;
        self.bg_read_a = bg_read_a;
        self.bg_read_b = bg_read_b;
        self.bg_read_a_clamp = bg_read_a_clamp;
        self.bg_read_b_clamp = bg_read_b_clamp;
        self.blur1_h_bg_a = blur1_h_bg_a;
        self.blur1_h_bg_b = blur1_h_bg_b;
        self.blur1_v_bg = blur1_v_bg;
        self.blur2_h_bg = blur2_h_bg;
        self.blur2_v_bg = blur2_v_bg;
        self.blur3_h_bg = blur3_h_bg;
        self.blur3_v_bg = blur3_v_bg;
        self.shape_bg_read_a = shape_bg_read_a;
        self.shape_bg_read_b = shape_bg_read_b;
        self.shape_bg_read_a_clamp = shape_bg_read_a_clamp;
        self.shape_bg_read_b_clamp = shape_bg_read_b_clamp;
        // Output size is what callers see; render size is the internal canvas.
        self.width = out_w;
        self.height = out_h;
        self.render_w = w;
        self.render_h = h;
        if self.geometry_diagnostics.enabled() {
            // Diagnostic readback copies the internal canvas → size it to render.
            self.geometry_stage_readback = Some(GeometryStageReadback::new(&self.device, w, h));
        }
        Ok(())
    }

    pub fn blur_levels(&self) -> u8 {
        self.blur_levels
    }

    /// Number of blur render passes issued on the most recent [`Self::render`]
    /// call (0, 2, 4, or 6 — two per generated level).
    pub fn last_blur_pass_count(&self) -> u32 {
        self.last_blur_pass_count
    }

    /// Switch to deterministic fixed-timestep timing (for offscreen animation
    /// export). Each rendered frame advances `time` by `1/fps` seconds.
    pub fn set_fixed_fps(&mut self, fps: f32) {
        let fps = if fps.is_finite() { fps.max(1.0) } else { 60.0 };
        self.time_per_frame = Some(1.0 / f64::from(fps));
    }

    /// Enable or disable the guarded custom-wave sample LOD. When enabled, only
    /// expensive, side-effect-free per-point programs are reduced to 256 points;
    /// dots and programs using loops, random, megabuf, or gmegabuf stay exact.
    pub fn set_custom_wave_adaptive_lod(&mut self, enabled: bool) {
        self.custom_wave_adaptive_lod = enabled;
    }

    /// Enable or disable the FXAA output pass. Enabled (the default) is the
    /// full-fidelity path: COMP renders into the offscreen comp intermediate
    /// and a 9-tap FXAA pass resolves it to the swapchain. Disabling routes COMP
    /// straight to the swapchain and skips both the FXAA pass and the
    /// intermediate write/read round-trip — intended for a downstream tier that
    /// applies its own final anti-aliasing.
    pub fn set_fxaa_enabled(&mut self, enabled: bool) {
        self.fxaa_enabled = enabled;
    }

    /// Internal render scale in `(0, 1]`. `1.0` is native (default).
    pub fn internal_scale(&self) -> f32 {
        self.internal_scale
    }

    /// Set the internal render scale. Values `>= 1.0` (or non-finite) render at
    /// native output resolution — byte-identical to the default. Values in
    /// `(0, 1)` shrink the feedback/blur/comp targets; the final comp/FXAA pass
    /// upscales to the output. A scale change that alters the internal
    /// dimensions UV-resamples the feedback so the frame carries across without
    /// a black border or luminance collapse. No-op if the scale is unchanged.
    pub fn set_internal_scale(&mut self, scale: f32) {
        let scale = if scale.is_finite() {
            scale.clamp(f32::MIN_POSITIVE, 1.0)
        } else {
            1.0
        };
        if scale == self.internal_scale {
            return;
        }
        self.internal_scale = scale;
        // Rebuild at the current output size; resample the feedback across the
        // (possibly) new internal dimensions. Dimensions are already valid.
        let _ = self.rebuild_render_targets(self.width, self.height, CarryMode::Resample);
    }

    /// Current feedback mip-chain cap. `u32::MAX` means the full log2 chain.
    pub fn feedback_mip_cap(&self) -> u32 {
        self.feedback_mip_cap
    }

    /// Cap the number of feedback ping-pong mip levels allocated AND regenerated
    /// each frame. `u32::MAX` (default) keeps the full log2 chain (byte-
    /// identical); a lower cap trims the required-LOD tail. This is independent
    /// of the separate blur pyramid. No-op if the cap is unchanged.
    pub fn set_feedback_mip_cap(&mut self, cap: u32) {
        let cap = cap.max(1);
        if cap == self.feedback_mip_cap {
            return;
        }
        self.feedback_mip_cap = cap;
        // Dimensions are unchanged, so the feedback carries over losslessly; only
        // the allocated/generated mip count changes.
        let _ = self.rebuild_render_targets(self.width, self.height, CarryMode::Resample);
    }

    /// Currently selected performance profile.
    pub fn performance_profile(&self) -> MilkdropPerformanceProfile {
        self.perf_profile
    }

    /// Hot-switch the performance profile. Bundles `internal_scale`,
    /// `feedback_mip_cap`, and `fxaa_enabled`. [`MilkdropPerformanceProfile::Reference`]
    /// (the default) is byte-identical to the pre-profile behaviour. Only the
    /// pipeline targets are rebuilt (and only when scale/cap actually change);
    /// no shaders or pipelines are recompiled.
    pub fn set_performance_profile(&mut self, profile: MilkdropPerformanceProfile) {
        let (scale, mip_cap, fxaa) = profile.params();
        self.perf_profile = profile;
        self.fxaa_enabled = fxaa;
        let rebuild = scale != self.internal_scale || mip_cap != self.feedback_mip_cap;
        self.internal_scale = scale;
        self.feedback_mip_cap = mip_cap;
        if rebuild {
            let _ = self.rebuild_render_targets(self.width, self.height, CarryMode::Resample);
        }
    }

    /// Opt in to compact CPU-side summaries of custom shape/wave geometry.
    /// Disabling collection also drops the latest retained snapshot.
    pub fn set_geometry_diagnostics_enabled(&mut self, enabled: bool) {
        self.geometry_diagnostics.set_enabled(enabled);
        if enabled {
            if !self
                .geometry_stage_readback
                .as_ref()
                .is_some_and(|readback| readback.matches(self.render_w, self.render_h))
            {
                self.geometry_stage_readback = Some(GeometryStageReadback::new(
                    &self.device,
                    self.render_w,
                    self.render_h,
                ));
            }
        } else {
            self.geometry_stage_readback = None;
        }
    }

    /// Return the most recently collected custom-geometry snapshot, or `None`
    /// when collection is disabled or no enabled frame has rendered yet.
    pub fn geometry_diagnostics(&self) -> Option<MilkdropGeometryDiagnostics> {
        let mut diagnostics = self.geometry_diagnostics.latest()?;
        if let Some(readback) = self.geometry_stage_readback.as_ref() {
            let [post_warp_rgb, post_overlays_rgb, post_comp_rgb] =
                readback.read_summaries(&self.device);
            diagnostics.post_warp_rgb = post_warp_rgb;
            diagnostics.post_overlays_rgb = post_overlays_rgb;
            diagnostics.post_comp_rgb = post_comp_rgb;
        }
        Some(diagnostics)
    }

    /// Read the latest warp, overlay, and comp checkpoints as RGBA8 images.
    /// Returns `None` unless diagnostics were enabled before rendering.
    pub fn geometry_stage_images(&self) -> Option<MilkdropStageImages> {
        self.geometry_stage_readback
            .as_ref()?
            .read_images(&self.device)
    }

    pub fn set_audio(&mut self, bass: f32, mid: f32, treb: f32, vol: f32) {
        self.audio = Some([
            audio_level_floor(bass),
            audio_level_floor(mid),
            audio_level_floor(treb),
            audio_level_floor(vol),
        ]);
    }

    pub fn set_audio_att(&mut self, bass_att: f32, mid_att: f32, treb_att: f32, vol_att: f32) {
        self.audio_att = Some([
            audio_att_floor(bass_att),
            audio_att_floor(mid_att),
            audio_att_floor(treb_att),
            audio_att_floor(vol_att),
        ]);
    }

    /// Set the metadata rate used by Hz-addressed enhanced-audio helpers.
    ///
    /// This is intentionally separate from the legacy `AudioInput` layout:
    /// recordings/preanalysis do not carry this side channel and therefore keep
    /// the documented 44.1 kHz fallback. Live hosts should call it whenever the
    /// capture device sample rate changes.
    pub fn set_enhanced_audio_sample_rate(&mut self, sample_rate_hz: f32) {
        self.enhanced_audio_sample_rate_hz = if sample_rate_hz.is_finite()
            && (2.0..=384_000.0).contains(&sample_rate_hz)
        {
            sample_rate_hz
        } else {
            LEGACY_ASSUMED_SAMPLE_RATE_HZ
        };
        self.enhanced_audio_nyquist_hz = self.enhanced_audio_sample_rate_hz * 0.5;
    }

    /// Update the renderer-owned BeatDrop-style FFT and waveform textures.
    ///
    /// Independent L/R magnitudes are accepted when the host truly has them;
    /// otherwise this runs the bounded 2,048-sample PCM FFT path. Supplying the
    /// legacy mono/equalized `freqArray` twice is intentionally not treated as
    /// a stereo spectrum. Work and GPU upload are skipped unless the compiled
    /// preset actually calls an enhanced-audio shader helper, except that a
    /// base mode-8 waveform needs this bounded calculation to obtain its real
    /// independent spectrum rows. (Dynamic per-frame changes into mode 8 should
    /// use [`Self::set_stereo_freq_spectrum`] until the next preset load.)
    #[allow(clippy::too_many_arguments)]
    pub fn set_enhanced_audio(
        &mut self,
        spectrum_left: Option<&[f32]>,
        spectrum_right: Option<&[f32]>,
        waveform_left: &[f32],
        waveform_right: &[f32],
        effective_sample_rate_hz: f32,
        elapsed_seconds: f32,
        config: EnhancedAudioConfig,
    ) {
        self.set_enhanced_audio_config(config);
        self.set_enhanced_audio_sample_rate(effective_sample_rate_hz);
        let base_mode_is_stereo_spectrum = self.bw_mode.is_finite()
            && (self.bw_mode.floor() as i32).rem_euclid(18) == 8;
        if !self.enhanced_audio_enabled && !base_mode_is_stereo_spectrum {
            return;
        }

        let config = self.enhanced_audio_config;
        let sample_rate_hz = self.enhanced_audio_sample_rate_hz;
        let nyquist_hz = {
            let (processor, fft_upload, wave_upload, freq_left, freq_right) = (
                &mut self.enhanced_audio_processor,
                &mut self.enhanced_audio_fft_upload,
                &mut self.enhanced_audio_wave_upload,
                &mut self.freq_spectrum,
                &mut self.freq_spectrum_right,
            );
            let frame = match (
                spectrum_left.filter(|row| !row.is_empty()),
                spectrum_right.filter(|row| !row.is_empty()),
            ) {
                (Some(left), Some(right)) => processor.update_stereo_spectrum(
                    left,
                    right,
                    waveform_left,
                    waveform_right,
                    sample_rate_hz,
                    elapsed_seconds,
                    config,
                ),
                _ => processor.update_from_pcm(
                    waveform_left,
                    waveform_right,
                    sample_rate_hz,
                    elapsed_seconds,
                    config,
                ),
            };
            frame.write_fft_rgba16f(fft_upload);
            frame.write_waveform_rgba16f(wave_upload);
            // These raw, finite L/R rows are intentionally separate from
            // `fft_rows`, whose rows are the helper's AGC-smoothed and
            // peak-hold values. Mode 8 uses adjacent bins from the real input
            // channels, so copying the latter would change its geometry.
            match frame.spectrum_rows() {
                Some((left, right)) => {
                    let left_len = left.len().min(MAX_AUDIO_SAMPLES);
                    let right_len = right.len().min(MAX_AUDIO_SAMPLES);
                    replace_audio_samples(freq_left, &left[..left_len], left_len, false);
                    replace_audio_samples(freq_right, &right[..right_len], right_len, false);
                }
                None => freq_right.clear(),
            }
            frame.nyquist_hz()
        };
        self.enhanced_audio_nyquist_hz = nyquist_hz;

        if !self.enhanced_audio_enabled {
            return;
        }

        let upload_rows = |texture: &wgpu::Texture, rows: &[u16], width: u32| {
            self.queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(rows),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 8),
                    rows_per_image: Some(2),
                },
                wgpu::Extent3d {
                    width,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            );
        };
        upload_rows(
            &self.enhanced_fft_texture,
            &self.enhanced_audio_fft_upload,
            ENHANCED_FFT_BINS as u32,
        );
        upload_rows(
            &self.enhanced_wave_texture,
            &self.enhanced_audio_wave_upload,
            ENHANCED_WAVE_SAMPLES as u32,
        );
    }

    /// Override this frame's warp and comp `rand_frame` uniforms with values
    /// captured from Butterchurn. The override is consumed by the next render;
    /// normal renderer-owned random generation resumes when it is not supplied.
    pub fn set_frame_randoms(&mut self, warp: [f32; 4], comp: [f32; 4]) {
        self.frame_random_override = Some((warp, comp));
    }

    /// Override the shader `time` input for the next frame. The override is
    /// consumed by the next render and leaves normal interactive timing intact.
    pub fn set_frame_time_seconds(&mut self, time_seconds: f64) {
        self.frame_time_override = time_seconds.is_finite().then_some(time_seconds);
    }

    /// Override the next shader clock and FPS inputs with a captured renderer's
    /// values. This retains the supplied FPS for the frame's other EEL paths,
    /// which derive their `fps` pseudo-variable from `time_per_frame`.
    pub fn set_frame_timing(&mut self, time_seconds: f64, fps: f64) {
        self.set_frame_time_seconds(time_seconds);
        if fps.is_finite() && fps > 0.0 {
            self.time_per_frame = Some(1.0 / fps);
        }
    }

    /// Set the shader-visible frame number before the next render. Offline
    /// Butterchurn captures start at frame 1, whereas a freshly created native
    /// renderer naturally starts at zero; frame-dependent EEL must see the
    /// captured value to preserve recursive presets.
    pub fn set_frame_index(&mut self, frame_index: u64) {
        self.frame_idx = frame_index;
    }

    /// Feed the Butterchurn-shaped 512-bin FFT magnitude array (`freqArray`) for
    /// the next frame. Used by `bSpectrum` custom waveforms. Pass an empty slice
    /// (or never call) to keep the time-domain fallback.
    pub fn set_freq_spectrum(&mut self, spectrum: &[f32]) {
        // Keep the long-input behavior identical to the original setter: cap the
        // accepted row before storing it rather than resampling an unbounded tail.
        let n = spectrum.len().min(MAX_AUDIO_SAMPLES);
        self.set_freq_spectrum_resampled(&spectrum[..n], n);
    }

    /// Feed a spectrum row and resample it directly into renderer-owned storage.
    /// This lets legacy short spectrum rows reach MilkDrop's 512-bin input without
    /// an allocation in the host bridge on every frame.
    pub fn set_freq_spectrum_resampled(&mut self, spectrum: &[f32], sample_count: usize) {
        replace_audio_samples(
            &mut self.freq_spectrum,
            spectrum,
            sample_count.min(MAX_AUDIO_SAMPLES),
            false,
        );
        // The legacy ingress has one equalized/mono magnitude row. Do not
        // silently duplicate it as a false stereo source for BeatDrop mode 8.
        self.freq_spectrum_right.clear();
    }

    /// Feed independently-derived left and right spectrum rows. This optional
    /// ingress is specifically for the extended BeatDrop mode-8 waveform;
    /// ordinary custom `bSpectrum` waves continue to read the left/legacy row.
    /// Supplying either empty row disables mode 8 for that frame rather than
    /// presenting a mono approximation as stereo-compatible output.
    pub fn set_stereo_freq_spectrum(&mut self, left: &[f32], right: &[f32]) {
        let left_len = left.len().min(MAX_AUDIO_SAMPLES);
        let right_len = right.len().min(MAX_AUDIO_SAMPLES);
        replace_audio_samples(&mut self.freq_spectrum, &left[..left_len], left_len, false);
        replace_audio_samples(
            &mut self.freq_spectrum_right,
            &right[..right_len],
            right_len,
            false,
        );
    }

    /// Feed per-sample PCM waveform for the next frame (range ~[-1,1]). Used by
    /// the built-in and custom waveforms. Length equals the audio buffer length.
    pub fn set_waveform(&mut self, left: &[f32], right: &[f32]) {
        // Preserve the original shared-length contract for direct callers. The
        // explicitly resampled setter below is used when legacy left/right rows
        // have different source lengths.
        let n = left.len().min(right.len()).min(MAX_AUDIO_SAMPLES);
        self.set_waveform_resampled(&left[..n], &right[..n], n);
    }

    /// Feed a waveform and resample it directly into renderer-owned storage. Like
    /// [`Self::set_freq_spectrum_resampled`], this keeps legacy short rows from
    /// creating transient audio vectors on the live render path.
    pub fn set_waveform_resampled(&mut self, left: &[f32], right: &[f32], sample_count: usize) {
        let target_len = sample_count.min(MAX_AUDIO_SAMPLES);
        replace_audio_samples(&mut self.wave_l, left, target_len, true);
        replace_audio_samples(&mut self.wave_r, right, target_len, true);
    }

    /// Fill reusable waveform scratch with the deterministic animated fallback
    /// used when no live PCM row has been supplied.
    fn synthesize_waveform(t: f32, left: &mut Vec<f32>, right: &mut Vec<f32>) {
        // Synthesize 512 samples in [-1,1] that animate with time so the wave moves.
        // 512 (matching real butterchurn-parity feeds) is required so built-in modes
        // 1/2/3/5 (which index wave[i+32]) and 4/6/7 (capped at ~width/3) have enough
        // samples and don't degenerate or index out of bounds.
        let n = 512usize;
        left.clear();
        right.clear();
        left.reserve(n.saturating_sub(left.capacity()));
        right.reserve(n.saturating_sub(right.capacity()));
        for i in 0..n {
            let fi = i as f32;
            let a = 0.5 * (t * 6.0 + fi * 0.49).sin()
                + 0.3 * (t * 2.1 + fi * 0.21).sin()
                + 0.18 * (t * 11.3 + fi * 0.83).sin();
            let b = 0.5 * (t * 5.3 + fi * 0.55 + 1.7).sin()
                + 0.3 * (t * 2.7 + fi * 0.19 + 0.4).sin()
                + 0.18 * (t * 9.1 + fi * 0.77 + 2.1).sin();
            left.push(a.clamp(-1.0, 1.0));
            right.push(b.clamp(-1.0, 1.0));
        }
    }

    /// Build CPU-side fill + border geometry for all enabled shapes this frame.
    /// Returns (fill_verts, fill_draws, border_verts, border_draws).
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn build_shape_geometry(
        &mut self,
        t: f64,
        bass: f64,
        mid: f64,
        treb: f64,
        vol: f64,
        bass_att: f64,
        mid_att: f64,
        treb_att: f64,
        aspectx: f32,
        aspecty: f32,
        q: &[f64; 32],
        regs: &[f64; 100],
    ) -> (
        Vec<ShapeVert>,
        Vec<ShapeFillDraw>,
        Vec<BorderVert>,
        Vec<BorderDraw>,
    ) {
        use std::f32::consts::PI;
        let mut fill_verts = std::mem::take(&mut self.scratch.shape_fill_verts);
        let mut fill_draws = std::mem::take(&mut self.scratch.shape_fill_draws);
        let mut border_verts = std::mem::take(&mut self.scratch.shape_border_verts);
        let mut border_draws = std::mem::take(&mut self.scratch.shape_border_draws);
        fill_verts.clear();
        fill_draws.clear();
        border_verts.clear();
        border_draws.clear();
        let fps = effective_fps(self.time_per_frame);

        for s in self.shapes.iter_mut() {
            if s.base.enabled == 0 {
                continue;
            }
            let num_inst = (s.base.num_inst.max(1)).min(MAX_SHAPE_INSTANCES as i32);

            for j in 0..num_inst {
                if fill_verts.len() + SHAPE_FILL_VERTS_MAX > SHAPE_VERT_CAP {
                    break;
                }
                // Resolve per-instance vals: run per-frame eqs if present, else base.
                let (
                    sides_f,
                    rad,
                    ang,
                    x,
                    y,
                    r,
                    g,
                    b,
                    a,
                    r2,
                    g2,
                    b2,
                    a2,
                    border_r,
                    border_g,
                    border_b,
                    border_a,
                    thick,
                    textured,
                    tex_ang,
                    tex_zoom,
                    additive,
                );

                if let Some(prog) = &s.prog {
                    // Reset shape vars from base each instance (butterchurn semantics).
                    let env = &mut s.env;
                    let slots = s.slots;
                    for &index in &s.live_reg_indices {
                        let index = index as usize;
                        env.set_slot_value(s.reg_slots[index], regs[index]);
                    }
                    for &index in &s.live_t_indices {
                        let index = index as usize;
                        env.set_slot_value(s.t_slots[index], s.t_init[index]);
                    }
                    slots.frame.seed(
                        env,
                        t,
                        self.frame_idx,
                        fps,
                        bass,
                        mid,
                        treb,
                        vol,
                        bass_att,
                        mid_att,
                        treb_att,
                        aspectx as f64,
                        aspecty as f64,
                    );
                    for &index in &s.live_q_indices {
                        let index = index as usize;
                        env.set_slot_value(s.q_slots[index], q[index]);
                    }
                    env.set_slot_value(slots.instance, j as f64);
                    env.set_slot_value(slots.num_inst, num_inst as f64);
                    let bv = &s.base;
                    env.set_slot_value(slots.sides, bv.sides as f64);
                    env.set_slot_value(slots.rad, bv.rad as f64);
                    env.set_slot_value(slots.ang, bv.ang as f64);
                    env.set_slot_value(slots.x, bv.x as f64);
                    env.set_slot_value(slots.y, bv.y as f64);
                    env.set_slot_value(slots.r, bv.r as f64);
                    env.set_slot_value(slots.g, bv.g as f64);
                    env.set_slot_value(slots.b, bv.b as f64);
                    env.set_slot_value(slots.a, bv.a as f64);
                    env.set_slot_value(slots.r2, bv.r2 as f64);
                    env.set_slot_value(slots.g2, bv.g2 as f64);
                    env.set_slot_value(slots.b2, bv.b2 as f64);
                    env.set_slot_value(slots.a2, bv.a2 as f64);
                    env.set_slot_value(slots.border_r, bv.border_r as f64);
                    env.set_slot_value(slots.border_g, bv.border_g as f64);
                    env.set_slot_value(slots.border_b, bv.border_b as f64);
                    env.set_slot_value(slots.border_a, bv.border_a as f64);
                    env.set_slot_value(slots.thickoutline, bv.thick_outline as f64);
                    env.set_slot_value(slots.textured, bv.textured as f64);
                    env.set_slot_value(slots.tex_ang, bv.tex_ang as f64);
                    env.set_slot_value(slots.tex_zoom, bv.tex_zoom as f64);
                    env.set_slot_value(slots.additive, bv.additive as f64);
                    prog.run_with(env, &mut s.state);
                    let rd = |slot: EnvSlot| env.slot_value(slot) as f32;
                    sides_f = rd(slots.sides);
                    rad = rd(slots.rad);
                    ang = rd(slots.ang);
                    x = rd(slots.x);
                    y = rd(slots.y);
                    r = rd(slots.r);
                    g = rd(slots.g);
                    b = rd(slots.b);
                    a = rd(slots.a);
                    r2 = rd(slots.r2);
                    g2 = rd(slots.g2);
                    b2 = rd(slots.b2);
                    a2 = rd(slots.a2);
                    border_r = rd(slots.border_r);
                    border_g = rd(slots.border_g);
                    border_b = rd(slots.border_b);
                    border_a = rd(slots.border_a);
                    thick = rd(slots.thickoutline);
                    textured = rd(slots.textured);
                    tex_ang = rd(slots.tex_ang);
                    tex_zoom = rd(slots.tex_zoom);
                    additive = rd(slots.additive);
                } else {
                    let bv = &s.base;
                    sides_f = bv.sides;
                    rad = bv.rad;
                    ang = bv.ang;
                    x = bv.x;
                    y = bv.y;
                    r = bv.r;
                    g = bv.g;
                    b = bv.b;
                    a = bv.a;
                    r2 = bv.r2;
                    g2 = bv.g2;
                    b2 = bv.b2;
                    a2 = bv.a2;
                    border_r = bv.border_r;
                    border_g = bv.border_g;
                    border_b = bv.border_b;
                    border_a = bv.border_a;
                    thick = bv.thick_outline as f32;
                    textured = bv.textured as f32;
                    tex_ang = bv.tex_ang;
                    tex_zoom = bv.tex_zoom;
                    additive = bv.additive as f32;
                }

                let blend_progress = 1.0f32;
                let sides = (sides_f.clamp(3.0, 100.0)).floor() as u32;
                let x_ndc = x * 2.0 - 1.0;
                let y_ndc = y * (-2.0) + 1.0;
                let is_additive = additive.abs() >= 1.0;
                let is_textured = textured.abs() >= 1.0;
                let is_thick = thick.abs() >= 1.0;
                let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
                let r = fin(r, 0.0).clamp(0.0, 1.0);
                let g = fin(g, 0.0).clamp(0.0, 1.0);
                let b = fin(b, 0.0).clamp(0.0, 1.0);
                let a = fin(a, 0.0).clamp(0.0, 1.0);
                let r2 = fin(r2, 0.0).clamp(0.0, 1.0);
                let g2 = fin(g2, 0.0).clamp(0.0, 1.0);
                let b2 = fin(b2, 0.0).clamp(0.0, 1.0);
                let a2 = fin(a2, 0.0).clamp(0.0, 1.0);
                let border_r = fin(border_r, 0.0).clamp(0.0, 1.0);
                let border_g = fin(border_g, 0.0).clamp(0.0, 1.0);
                let border_b = fin(border_b, 0.0).clamp(0.0, 1.0);
                let border_alpha = fin(border_a, 0.0).clamp(0.0, 1.0) * blend_progress;
                let has_border = border_alpha > 0.0
                    && border_verts.len() + (sides as usize + 1) <= BORDER_VERT_CAP;
                let quarter_pi = PI * 0.25;

                let base_vertex = fill_verts.len() as i32;

                // center vertex (uv sentinel (-1,-1) when untextured → solid color in FS)
                fill_verts.push(ShapeVert {
                    pos: [x_ndc, y_ndc],
                    color: [r, g, b, a * blend_progress],
                    uv: if is_textured {
                        [0.5, 0.5]
                    } else {
                        [-1.0, -1.0]
                    },
                });

                let border_start = border_verts.len() as u32;
                // rim vertices k = 1..=sides+1 (last duplicates first to close)
                for k in 1..=(sides + 1) {
                    let p = (k - 1) as f32 / sides as f32;
                    let p_two_pi = p * 2.0 * PI;
                    let ang_sum = p_two_pi + ang + quarter_pi;
                    let (ang_sin, ang_cos) = ang_sum.sin_cos();
                    let px = x_ndc + rad * ang_cos * aspecty;
                    let py = y_ndc + rad * ang_sin;
                    let (uu, vv) = if is_textured {
                        let tex_ang_sum = p_two_pi + tex_ang + quarter_pi;
                        let (tex_sin, tex_cos) = tex_ang_sum.sin_cos();
                        let z = if tex_zoom.abs() < 1e-6 { 1.0 } else { tex_zoom };
                        (
                            0.5 + (0.5 * tex_cos / z) * aspecty,
                            0.5 + (0.5 * tex_sin / z),
                        )
                    } else {
                        (-1.0, -1.0)
                    };
                    fill_verts.push(ShapeVert {
                        pos: [px, py],
                        color: [r2, g2, b2, a2 * blend_progress],
                        uv: [uu, vv],
                    });
                    if has_border {
                        border_verts.push(BorderVert { pos: [px, py] });
                    }
                }

                fill_draws.push(ShapeFillDraw {
                    base_vertex,
                    sides,
                    additive: is_additive,
                    border_draw_index: has_border.then_some(border_draws.len()),
                });

                if has_border {
                    border_draws.push(BorderDraw {
                        start_vert: border_start,
                        count: sides + 1,
                        color: [border_r, border_g, border_b, border_alpha],
                        thick: is_thick,
                    });
                }
            }
            // textured flag is per-shape (we honor the first instance's via uniform).
            // jelly_space is untextured, so the textured path is wired but uses uv=0.5.
        }

        (fill_verts, fill_draws, border_verts, border_draws)
    }

    /// Build CPU-side waveform geometry (built-in + custom). Returns the packed
    /// vertex list and the draw records.
    #[allow(clippy::too_many_arguments)]
    fn build_wave_geometry(
        &mut self,
        t: f64,
        bass: f64,
        mid: f64,
        treb: f64,
        vol: f64,
        bass_att: f64,
        mid_att: f64,
        treb_att: f64,
        basic_aspectx: f32,
        basic_aspecty: f32,
        inv_aspectx: f32,
        inv_aspecty: f32,
        wave_l: &[f32],
        wave_r: &[f32],
        freq: &[f32],
        regs: &[f64; 100],
    ) -> (
        Vec<WaveVert>,
        Vec<WaveDraw>,
        Option<CustomWaveGeometryExtent>,
    ) {
        let mut verts = std::mem::take(&mut self.scratch.wave_verts);
        let mut draws = std::mem::take(&mut self.scratch.wave_draws);
        verts.clear();
        draws.clear();
        let audio_len = wave_l.len();

        // ── Custom waveforms first (index order), then built-in last ─────────
        self.build_custom_waves(
            t,
            bass,
            mid,
            treb,
            vol,
            bass_att,
            mid_att,
            treb_att,
            inv_aspectx,
            inv_aspecty,
            wave_l,
            wave_r,
            freq,
            regs,
            &mut verts,
            &mut draws,
        );
        let custom_extent =
            self.geometry_diagnostics
                .enabled()
                .then_some(CustomWaveGeometryExtent {
                    vertices: verts.len(),
                    draws: draws.len(),
                });

        if audio_len > 0 {
            // Built-in waveform alpha is the post-per-frame `wave_a` (butterchurn reads
            // mdVSFrame.wave_a, the value AFTER frame_eqs). Both jelly_space and parade
            // set wave_a=0 in per-frame, which correctly gates the wave off — matching
            // butterchurn. Fall back to the parsed base fWaveAlpha when per-frame never
            // touched wave_a.
            let live_wave_a = self
                .eel_env
                .get("wave_a")
                .copied()
                .map(|v| v as f32)
                .unwrap_or(self.bw_a);
            let phase_t = shader_time_seconds(t);
            let mut basic_scratch = std::mem::take(&mut self.scratch.basic_wave);
            self.build_basic_waveform(
                phase_t,
                bass,
                mid,
                treb,
                basic_aspectx,
                basic_aspecty,
                live_wave_a,
                wave_l,
                wave_r,
                freq,
                &mut verts,
                &mut draws,
                &mut basic_scratch,
            );
            self.scratch.basic_wave = basic_scratch;
        }

        (verts, draws, custom_extent)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_basic_waveform(
        &self,
        t: f32,
        bass: f64,
        mid: f64,
        treb: f64,
        aspectx: f32,
        aspecty: f32,
        live_wave_a: f32,
        time_l: &[f32],
        time_r: &[f32],
        spectrum_left: &[f32],
        verts: &mut Vec<WaveVert>,
        draws: &mut Vec<WaveDraw>,
        scratch: &mut BasicWaveScratch,
    ) {
        use std::f32::consts::PI;
        // alpha gate: built-in reads the post-per-frame wave_a (butterchurn behavior).
        let base_alpha = live_wave_a;
        let vol = ((bass + mid + treb) / 3.0) as f32;
        if !(vol > -0.01 && base_alpha > 0.0 && !time_l.is_empty()) {
            return;
        }
        let live = |k: &str, d: f32| {
            self.eel_env
                .get(k)
                .copied()
                .map(|v| v as f32)
                .filter(|v| v.is_finite())
                .unwrap_or(d)
        };
        let live_bool = |k: &str, d: bool| live(k, if d { 1.0 } else { 0.0 }) != 0.0;
        let live_wave_mode = live("wave_mode", self.bw_mode);
        let live_wave_x = live("wave_x", self.bw_x);
        let live_wave_y = live("wave_y", self.bw_y);
        let live_wave_mystery = live("wave_mystery", self.bw_mystery);
        let live_wave_scale = live("wave_scale", self.bw_scale);
        let live_wave_smoothing = live("wave_smoothing", self.bw_smoothing);
        let live_dots = live_bool("wave_dots", self.bw_dots);
        let live_thick = live_bool("wave_thick", self.bw_thick);
        let live_additive = live_bool("additivewave", self.bw_additive);
        let live_brighten = live_bool("wave_brighten", self.bw_brighten);
        let live_modalphavol = live_bool("modwavealphabyvolume", self.bw_modalphavol);
        let live_modalphastart = live("modwavealphastart", self.bw_modalphastart);
        let live_modalphaend = live("modwavealphaend", self.bw_modalphaend);

        // processWaveform (butterchurn 4520-4533): scale = wave_scale/128 on Int8.
        // Our samples are f32 in [-1,1] (== Int8/128), so the effective scale on the
        // f32 data is simply wave_scale.
        let process = |src: &[f32], out: &mut Vec<f32>| {
            let scale = live_wave_scale;
            let smooth = live_wave_smoothing;
            let smooth2 = scale * (1.0 - smooth);
            let n = src.len();
            out.clear();
            out.resize(n, 0.0);
            if n == 0 {
                return;
            }
            out[0] = src[0] * scale;
            for i in 1..n {
                out[i] = src[i] * smooth2 + out[i - 1] * smooth;
            }
        };
        process(time_l, &mut scratch.processed_l);
        process(time_r, &mut scratch.processed_r);
        let wave_l = scratch.processed_l.as_slice();
        let wave_r = scratch.processed_r.as_slice();

        // BeatDrop extends the classic 0..=7 set through mode 17. Preserve the
        // classic match below byte-for-byte for modes 0..=7; modes 8..=17 are
        // isolated in the licensed helper and share only this renderer-owned
        // color/smoothing/upload tail.
        let new_wave_mode = (live_wave_mode.floor() as i32).rem_euclid(18);
        let wave_pos_x = live_wave_x * 2.0 - 1.0;
        let wave_pos_y = live_wave_y * 2.0 - 1.0;

        if new_wave_mode >= 8 {
            // Mode 8 remains empty unless an ingress supplied real independent
            // L/R magnitudes. `set_enhanced_audio` installs the processor's raw
            // host/PCM rows here when active; `set_stereo_freq_spectrum` is the
            // direct alternative. Legacy mono spectrum input never fills right.
            let Some(geometry) = build_extended_waveform(ExtendedWaveformInput {
                mode: new_wave_mode,
                time: t,
                wave_pos_x,
                wave_pos_y,
                wave_param: live_wave_mystery,
                aspect_x: aspectx,
                aspect_y: aspecty,
                screen_dependent: false,
                render_width: self.render_w as usize,
                left: wave_l,
                right: wave_r,
                spectrum_left,
                spectrum_right: &self.freq_spectrum_right,
                bass: bass as f32,
                mid: mid as f32,
                treble: treb as f32,
                alpha: base_alpha,
                modulate_alpha_by_volume: live_modalphavol,
                modwave_alpha_start: live_modalphastart,
                modwave_alpha_end: live_modalphaend,
                blending: live_additive,
            }) else {
                return;
            };

            let mut cr = live("wave_r", self.bw_r).clamp(0.0, 1.0);
            let mut cg = live("wave_g", self.bw_g).clamp(0.0, 1.0);
            let mut cb = live("wave_b", self.bw_b).clamp(0.0, 1.0);
            if live_brighten {
                let maxc = cr.max(cg).max(cb);
                if maxc > 0.01 {
                    cr /= maxc;
                    cg /= maxc;
                    cb /= maxc;
                }
            }
            if geometry.alpha <= 0.0 {
                return;
            }
            let color = [cr, cg, cb, geometry.alpha];
            let points = live_dots || geometry.points_recommended;
            let thick = live_thick || live_dots;
            let smoothed = &mut scratch.smoothed;
            for mut strip in geometry.strips {
                for position in &mut strip {
                    position[1] = -position[1];
                }
                smooth_wave_into(&strip, smoothed);
                if smoothed.is_empty() {
                    continue;
                }
                let start = verts.len() as u32;
                verts.extend(smoothed.iter().copied().map(|pos| WaveVert { pos, color }));
                draws.push(WaveDraw {
                    start_vert: start,
                    count: smoothed.len() as u32,
                    points,
                    additive: live_additive,
                    thick,
                });
            }
            return;
        }

        let mut param2 = live_wave_mystery;
        if (new_wave_mode == 0 || new_wave_mode == 1 || new_wave_mode == 4) && param2.abs() > 1.0 {
            param2 = param2 * 0.5 + 0.5;
            param2 -= param2.floor();
            param2 = param2.abs();
            param2 = param2 * 2.0 - 1.0;
        }

        let nlen = wave_l.len();
        let positions = &mut scratch.positions;
        positions.clear();
        // Mode 7 emits a SECOND polyline (R-channel line). Reuse its backing store.
        let positions2 = &mut scratch.positions2;
        positions2.clear();
        let mut has_positions2 = false;
        let mut alpha = base_alpha;

        // mod-wave-alpha-by-volume (every mode applies this). Guarded divide like mode 0.
        let mod_alpha = |alpha: &mut f32| {
            if live_modalphavol {
                let diff = live_modalphaend - live_modalphastart;
                if diff.abs() > 1e-9 {
                    *alpha *= (vol - live_modalphastart) / diff;
                }
            }
        };

        // texsizeX / texsizeY (butterchurn) == internal render size, as f32.
        let texsize_x = self.render_w as f32;

        match new_wave_mode {
            0 => {
                // circle
                if live_modalphavol {
                    let diff = live_modalphaend - live_modalphastart;
                    if diff.abs() > 1e-9 {
                        alpha *= (vol - live_modalphastart) / diff;
                    }
                }
                alpha = alpha.clamp(0.0, 1.0);
                let num_vert = (nlen / 2) + 1;
                if num_vert < 2 {
                    return;
                }
                let num_vert_inv = 1.0 / (num_vert - 1) as f32;
                let sample_offset = (nlen.saturating_sub(num_vert)) / 2;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert - 1 {
                    let mut rad = 0.5 + 0.4 * wave_r[(i + sample_offset).min(nlen - 1)] + param2;
                    let ang = i as f32 * num_vert_inv * 2.0 * PI + t * 0.2;
                    if (i as f32) < num_vert as f32 / 10.0 {
                        let mut mix = i as f32 / (num_vert as f32 * 0.1);
                        mix = 0.5 - 0.5 * (mix * PI).cos();
                        let idx2 = (i + num_vert + sample_offset).min(nlen - 1);
                        let rad2 = 0.5 + 0.4 * wave_r[idx2] + param2;
                        rad = (1.0 - mix) * rad2 + rad * mix;
                    }
                    let (ang_sin, ang_cos) = ang.sin_cos();
                    positions[i] = [
                        rad * ang_cos * aspecty + wave_pos_x,
                        rad * ang_sin * aspectx + wave_pos_y,
                    ];
                }
                positions[num_vert - 1] = positions[0];
            }
            1 => {
                // rotating circle, ang driven by L
                alpha *= 1.25;
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let num_vert = nlen / 2;
                if num_vert < 1 {
                    return;
                }
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    let rad = 0.53 + 0.43 * wave_r[i] + param2;
                    let ang = wave_l[(i + 32).min(nlen - 1)] * 0.5 * PI + t * 2.3;
                    let (ang_sin, ang_cos) = ang.sin_cos();
                    positions[i] = [
                        rad * ang_cos * aspecty + wave_pos_x,
                        rad * ang_sin * aspectx + wave_pos_y,
                    ];
                }
            }
            2 => {
                // X/Y scatter, faint
                alpha *= if texsize_x < 1024.0 {
                    0.09
                } else if texsize_x < 2048.0 {
                    0.11
                } else {
                    0.13
                };
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let num_vert = nlen;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    positions[i] = [
                        wave_r[i] * aspecty + wave_pos_x,
                        wave_l[(i + 32) % nlen] * aspectx + wave_pos_y,
                    ];
                }
            }
            3 => {
                // X/Y scatter, treble-gated (same geometry as mode 2)
                alpha *= if texsize_x < 1024.0 {
                    0.15
                } else if texsize_x < 2048.0 {
                    0.22
                } else {
                    0.33
                };
                alpha *= 1.3;
                alpha *= (treb * treb) as f32;
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let num_vert = nlen;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    positions[i] = [
                        wave_r[i] * aspecty + wave_pos_x,
                        wave_l[(i + 32) % nlen] * aspectx + wave_pos_y,
                    ];
                }
            }
            4 => {
                // horizontal scope with momentum
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let mut num_vert = nlen;
                if num_vert > (texsize_x / 3.0) as usize {
                    num_vert = (texsize_x / 3.0).floor() as usize;
                }
                if num_vert < 2 {
                    return;
                }
                let num_vert_inv = 1.0 / num_vert as f32; // NOT num_vert-1
                let sample_offset = nlen.saturating_sub(num_vert) / 2;
                let w1 = 0.45 + 0.5 * (param2 * 0.5 + 0.5);
                let w2 = 1.0 - w1;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    let mut x = 2.0 * (i as f32) * num_vert_inv
                        + (wave_pos_x - 1.0)
                        + wave_r[(i + 25 + sample_offset) % nlen] * 0.44;
                    let mut y = wave_l[i + sample_offset] * 0.47 + wave_pos_y;
                    if i > 1 {
                        x = x * w2 + w1 * (positions[i - 1][0] * 2.0 - positions[i - 2][0]);
                        y = y * w2 + w1 * (positions[i - 1][1] * 2.0 - positions[i - 2][1]);
                    }
                    positions[i] = [x, y];
                }
            }
            5 => {
                // Lissajous-ish rotating
                alpha *= if texsize_x < 1024.0 {
                    0.09
                } else if texsize_x < 2048.0 {
                    0.11
                } else {
                    0.13
                };
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let (sin_rot, cos_rot) = (t * 0.3).sin_cos();
                let num_vert = nlen;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    let ioff = (i + 32) % nlen;
                    let x0 = wave_r[i] * wave_l[ioff] + wave_l[i] * wave_r[ioff];
                    let y0 = wave_r[i] * wave_r[i] - wave_l[ioff] * wave_l[ioff];
                    positions[i] = [
                        (x0 * cos_rot - y0 * sin_rot) * (aspecty + wave_pos_x),
                        (x0 * sin_rot + y0 * cos_rot) * (aspectx + wave_pos_y),
                    ];
                }
            }
            6 | 7 => {
                // angled line through screen, clipped to [-1.1, 1.1] box.
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let mut num_vert = nlen / 2;
                if num_vert > (texsize_x / 3.0) as usize {
                    num_vert = (texsize_x / 3.0).floor() as usize;
                }
                if num_vert < 1 {
                    return;
                }
                let sample_offset = nlen.saturating_sub(num_vert) / 2;
                let ang = PI * 0.5 * param2;
                let (ang_sin, ang_cos) = ang.sin_cos();
                let mut dx = ang_cos;
                let mut dy = ang_sin;
                let (perp_seed_sin, perp_seed_cos) = (ang + PI * 0.5).sin_cos();
                // Both edgex AND edgey seed from wave_pos_x (butterchurn quirk — literal).
                let mut edgex = [
                    wave_pos_x * perp_seed_cos - dx * 3.0,
                    wave_pos_x * perp_seed_cos + dx * 3.0,
                ];
                let mut edgey = [
                    wave_pos_x * perp_seed_sin - dy * 3.0,
                    wave_pos_x * perp_seed_sin + dy * 3.0,
                ];
                for i in 0..2 {
                    for j in 0..4 {
                        let mut tt = 0.0f32;
                        let mut clip = false;
                        match j {
                            0 => {
                                if edgex[i] > 1.1 {
                                    tt = (1.1 - edgex[1 - i]) / (edgex[i] - edgex[1 - i]);
                                    clip = true;
                                }
                            }
                            1 => {
                                if edgex[i] < -1.1 {
                                    tt = (-1.1 - edgex[1 - i]) / (edgex[i] - edgex[1 - i]);
                                    clip = true;
                                }
                            }
                            2 => {
                                if edgey[i] > 1.1 {
                                    tt = (1.1 - edgey[1 - i]) / (edgey[i] - edgey[1 - i]);
                                    clip = true;
                                }
                            }
                            3 => {
                                if edgey[i] < -1.1 {
                                    tt = (-1.1 - edgey[1 - i]) / (edgey[i] - edgey[1 - i]);
                                    clip = true;
                                }
                            }
                            _ => {}
                        }
                        if clip {
                            let dxi = edgex[i] - edgex[1 - i];
                            let dyi = edgey[i] - edgey[1 - i];
                            edgex[i] = edgex[1 - i] + dxi * tt;
                            edgey[i] = edgey[1 - i] + dyi * tt;
                        }
                    }
                }
                dx = (edgex[1] - edgex[0]) / num_vert as f32;
                dy = (edgey[1] - edgey[0]) / num_vert as f32;
                let ang2 = dy.atan2(dx);
                let (perp_dy, perp_dx) = (ang2 + PI * 0.5).sin_cos();

                if new_wave_mode == 6 {
                    positions.resize(num_vert, [0.0, 0.0]);
                    for i in 0..num_vert {
                        let s = wave_l[i + sample_offset];
                        positions[i] = [
                            edgex[0] + dx * (i as f32) + perp_dx * 0.25 * s,
                            edgey[0] + dy * (i as f32) + perp_dy * 0.25 * s,
                        ];
                    }
                } else {
                    // MODE 7: dual line — L line + R line separated by sep.
                    let sep = (wave_pos_y * 0.5 + 0.5).powi(2);
                    positions.resize(num_vert, [0.0, 0.0]);
                    positions2.resize(num_vert, [0.0, 0.0]);
                    for i in 0..num_vert {
                        let s = wave_l[i + sample_offset];
                        positions[i] = [
                            edgex[0] + dx * (i as f32) + perp_dx * (0.25 * s + sep),
                            edgey[0] + dy * (i as f32) + perp_dy * (0.25 * s + sep),
                        ];
                    }
                    for i in 0..num_vert {
                        let s = wave_r[i + sample_offset];
                        positions2[i] = [
                            edgex[0] + dx * (i as f32) + perp_dx * (0.25 * s - sep),
                            edgey[0] + dy * (i as f32) + perp_dy * (0.25 * s - sep),
                        ];
                    }
                    has_positions2 = true;
                }
            }
            _ => {
                // Unreachable: rem_euclid(8) constrains new_wave_mode to 0..=7.
                return;
            }
        }

        // color (computed once, shared by both polylines for mode 7) from the LIVE
        // post-per-frame wave_r/g/b (butterchurn basicWaveform.js:446-448 reads
        // mdVSFrame.wave_*), mirroring live_wave_a; fall back to the parsed base when
        // per-frame never wrote them (idx 9096 colors its waveform in per-frame eqs).
        let mut cr = live("wave_r", self.bw_r).clamp(0.0, 1.0);
        let mut cg = live("wave_g", self.bw_g).clamp(0.0, 1.0);
        let mut cb = live("wave_b", self.bw_b).clamp(0.0, 1.0);
        if live_brighten {
            let maxc = cr.max(cg).max(cb);
            if maxc > 0.01 {
                cr /= maxc;
                cg /= maxc;
                cb /= maxc;
            }
        }
        let color = [cr, cg, cb, alpha];

        if alpha <= 0.0 {
            return;
        }

        // Shared tail: Y-flip (butterchurn negates pos.y before smoothing), smooth,
        // push verts, push a WaveDraw. Called once for modes 0-6, twice for mode 7.
        let dots = live_dots;
        let additive = live_additive;
        let thick = live_thick || live_dots;
        let smoothed = &mut scratch.smoothed;
        let mut emit = |pos: &mut Vec<[f32; 2]>| {
            for p in pos.iter_mut() {
                p[1] = -p[1];
            }
            smooth_wave_into(pos, smoothed);
            if smoothed.is_empty() {
                return;
            }
            let start = verts.len() as u32;
            for p in smoothed.iter() {
                verts.push(WaveVert { pos: *p, color });
            }
            draws.push(WaveDraw {
                start_vert: start,
                count: smoothed.len() as u32,
                points: dots,
                additive,
                thick,
            });
        };
        emit(positions);
        if has_positions2 {
            emit(positions2);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_custom_waves(
        &mut self,
        t: f64,
        bass: f64,
        mid: f64,
        treb: f64,
        vol: f64,
        bass_att: f64,
        mid_att: f64,
        treb_att: f64,
        inv_aspectx: f32,
        inv_aspecty: f32,
        time_l: &[f32],
        time_r: &[f32],
        freq: &[f32],
        regs: &[f64; 100],
        verts: &mut Vec<WaveVert>,
        draws: &mut Vec<WaveDraw>,
    ) {
        let max_samples = time_l.len();
        let wave_scale_base = self.bw_scale;
        let frame_idx = self.frame_idx;
        let fps = effective_fps(self.time_per_frame);
        let adaptive_lod = self.custom_wave_adaptive_lod;
        // q1..q32 from the main per-frame EEL — custom waveforms read these (ORB's
        // laser tubes are entirely q1-driven). Captured before the &mut waves loop.
        // Full q1..q32 (was capped at q8) to match MilkDrop/Butterchurn shape/wave semantics.
        let qv: [f64; 32] = std::array::from_fn(|i| self.eel_env.slot_value(self.eel_q_slots[i]));

        // Independent custom-wave pools are ideal coarse-grained parallel work:
        // the expensive Dancer presets commonly carry four 512-point programs,
        // while each program's points must remain serial because EEL locals may
        // intentionally carry from one point to the next. Only programs touching
        // preset-wide gmegabuf or the shared RNG retain authored wave ordering on
        // the render thread; loops and private megabuf state remain pool-local.
        let parallel_safe = self.waves.iter().filter(|wv| wv.def.enabled).all(|wv| {
            wv.per_frame_prog
                .as_ref()
                .is_none_or(EelProgram::custom_wave_parallel_safe)
                && wv
                    .per_point_prog
                    .as_ref()
                    .is_none_or(EelProgram::custom_wave_parallel_safe)
        });
        let build_wave = |wv: &mut WaveRT| -> Option<WaveDraw> {
            wv.scratch.output.clear();
            if !wv.def.enabled {
                return None;
            }

            // ── per-frame run ────────────────────────────────────────────────
            let env = &mut wv.env;
            let slots = wv.slots;
            for &index in &wv.live_reg_indices {
                let index = index as usize;
                env.set_slot_value(wv.reg_slots[index], regs[index]);
            }
            for &index in &wv.live_t_indices {
                let index = index as usize;
                env.set_slot_value(wv.t_slots[index], wv.t_init[index]);
            }
            slots.frame.seed(
                env,
                t,
                frame_idx,
                fps,
                bass,
                mid,
                treb,
                vol,
                bass_att,
                mid_att,
                treb_att,
                inv_aspectx as f64,
                inv_aspecty as f64,
            );
            for &index in &wv.live_q_indices {
                let index = index as usize;
                env.set_slot_value(wv.q_slots[index], qv[index]);
            }
            env.set_slot_value(slots.samples, wv.def.samples as f64);
            env.set_slot_value(slots.sep, wv.def.sep as f64);
            env.set_slot_value(slots.scaling, wv.def.scaling as f64);
            env.set_slot_value(slots.smoothing, wv.def.smoothing as f64);
            env.set_slot_value(slots.spectrum, if wv.def.spectrum { 1.0 } else { 0.0 });
            env.set_slot_value(slots.r, wv.def.r as f64);
            env.set_slot_value(slots.g, wv.def.g as f64);
            env.set_slot_value(slots.b, wv.def.b as f64);
            env.set_slot_value(slots.a, wv.def.a as f64);
            if let Some(p) = &wv.per_frame_prog {
                p.run_with(env, &mut wv.state);
            }
            // Custom-wave per-frame equations are part of the authored state
            // lifecycle even when no PCM buffer is available yet. Preserve
            // those side effects, then skip only the point-generation work.
            if max_samples == 0 {
                return None;
            }
            let pf_samples = env.slot_value(slots.samples).floor().max(0.0) as usize;
            let pf_sep = env.slot_value(slots.sep).floor() as i32;
            let pf_scaling = env.slot_value(slots.scaling) as f32;
            let pf_spectrum = env.slot_value(slots.spectrum) != 0.0;
            let pf_smoothing = env.slot_value(slots.smoothing) as f32;
            let frame_r = env.slot_value(slots.r) as f32;
            let frame_g = env.slot_value(slots.g) as f32;
            let frame_b = env.slot_value(slots.b) as f32;
            let frame_a = env.slot_value(slots.a) as f32;

            // ── sample prep (generateWaveform) ───────────────────────────────
            let mut authored_samples = pf_samples.min(max_samples);
            let sep = pf_sep.max(0) as usize;
            authored_samples = authored_samples.saturating_sub(sep);
            if !(authored_samples >= 2 || (wv.def.use_dots && authored_samples >= 1)) {
                return None;
            }
            // LOD is deliberately conservative: point stamps and any equation with
            // observable side effects retain every authored point. Pure, expensive
            // programs keep the same source span and endpoints at a lower density.
            let lod_safe = wv
                .per_point_prog
                .as_ref()
                .map(|program| {
                    program.custom_wave_lod_safe()
                        && program.operation_count() >= CUSTOM_WAVE_LOD_OP_THRESHOLD
                })
                .unwrap_or(false);
            let samples = if adaptive_lod && lod_safe && !wv.def.use_dots {
                authored_samples.min(CUSTOM_WAVE_LOD_SAMPLES)
            } else {
                authored_samples
            };

            // The *128 converts our normalized [-1,1] TIME samples to butterchurn's Int8
            // [-128,127] range that the 0.004 constant assumes. The SPECTRUM branch must
            // NOT get it: butterchurn's customWaveform.js:167 uses bare 0.15 for spectrum,
            // and our synth freq array is already ~[0,1] — *128 shoves every bin off-screen
            // (idx 8548 went black; silent-audio control renders it at luma 0.30). Keep the
            // *128 on the time branch only.
            let scale =
                (if pf_spectrum { 0.15 } else { 0.004 * 128.0 }) * pf_scaling * wave_scale_base;
            let scratch = &mut wv.scratch;
            let (src_l, src_r): (&[f32], &[f32]) = if pf_spectrum && !freq.is_empty() {
                if freq.len() == max_samples {
                    (freq, freq)
                } else {
                    resample_linear_into(freq, max_samples, &mut scratch.source_l, false);
                    (&scratch.source_l, &scratch.source_l)
                }
            } else {
                let left = if time_l.len() == max_samples {
                    time_l
                } else {
                    resample_linear_into(time_l, max_samples, &mut scratch.source_l, false);
                    &scratch.source_l
                };
                let right = if time_r.len() == max_samples {
                    time_r
                } else {
                    resample_linear_into(time_r, max_samples, &mut scratch.source_r, false);
                    &scratch.source_r
                };
                (left, right)
            };
            let (j0, j1, source_step) = if pf_spectrum {
                (
                    0usize,
                    sep.min(max_samples.saturating_sub(1)),
                    ((max_samples.saturating_sub(sep)).max(1) / authored_samples.max(1)).max(1),
                )
            } else {
                let j0 = ((max_samples as f32 - authored_samples as f32) / 2.0 - sep as f32 / 2.0)
                    .floor()
                    .max(0.0) as usize;
                let j1 = ((max_samples as f32 - authored_samples as f32) / 2.0 + sep as f32 / 2.0)
                    .floor()
                    .max(0.0) as usize;
                (j0, j1, 1usize)
            };
            let mix1 = (pf_smoothing * 0.98).max(0.0).powf(0.5);
            let mix2 = 1.0 - mix1;

            scratch.points_l.resize(authored_samples, 0.0);
            scratch.points_r.resize(authored_samples, 0.0);
            scratch.points_l[0] = *src_l.get(j0.min(max_samples - 1)).unwrap_or(&0.0);
            scratch.points_r[0] = *src_r.get(j1.min(max_samples - 1)).unwrap_or(&0.0);
            for j in 1..authored_samples {
                let il = (j * source_step + j0).min(max_samples - 1);
                let ir = (j * source_step + j1).min(max_samples - 1);
                scratch.points_l[j] = src_l[il] * mix2 + scratch.points_l[j - 1] * mix1;
                scratch.points_r[j] = src_r[ir] * mix2 + scratch.points_r[j - 1] * mix1;
            }
            for j in (0..authored_samples - 1).rev() {
                scratch.points_l[j] = scratch.points_l[j] * mix2 + scratch.points_l[j + 1] * mix1;
                scratch.points_r[j] = scratch.points_r[j] * mix2 + scratch.points_r[j + 1] * mix1;
            }
            for j in 0..authored_samples {
                scratch.points_l[j] *= scale;
                scratch.points_r[j] *= scale;
            }

            // ── per-point loop ───────────────────────────────────────────────
            // A gmegabuf point program makes the complete custom-wave schedule
            // serial. Hold its shared-buffer lock for this authored point batch
            // instead of taking the same uncontended mutex once per sample.
            let point_gmegabuf_handle = wv
                .per_point_prog
                .as_ref()
                .filter(|program| program.uses_gmegabuf())
                .map(|_| wv.state.gmegabuf.clone());
            let mut point_gmegabuf_guard = point_gmegabuf_handle.as_ref().map(|handle| {
                handle
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            });
            scratch.positions.clear();
            scratch.colors.clear();
            scratch
                .positions
                .reserve(samples.saturating_sub(scratch.positions.capacity()));
            scratch
                .colors
                .reserve(samples.saturating_sub(scratch.colors.capacity()));
            for j in 0..samples {
                let authored_j = if samples <= 1 {
                    0
                } else {
                    (j * (authored_samples - 1) + (samples - 1) / 2) / (samples - 1)
                };
                let value1 = scratch.points_l[authored_j];
                let value2 = scratch.points_r[authored_j];
                let sample_t = if samples <= 1 {
                    0.0
                } else {
                    j as f64 / (samples - 1) as f64
                };
                let (px, py, cr, cg, cb, ca) = if let Some(p) = &wv.per_point_prog {
                    let env = &mut wv.env;
                    env.set_slot_value(slots.sample, sample_t);
                    env.set_slot_value(slots.value1, value1 as f64);
                    env.set_slot_value(slots.value2, value2 as f64);
                    env.set_slot_value(slots.x, 0.5 + value1 as f64);
                    env.set_slot_value(slots.y, 0.5 + value2 as f64);
                    env.set_slot_value(slots.r, frame_r as f64);
                    env.set_slot_value(slots.g, frame_g as f64);
                    env.set_slot_value(slots.b, frame_b as f64);
                    env.set_slot_value(slots.a, frame_a as f64);
                    if let Some(gmegabuf) = point_gmegabuf_guard.as_deref_mut() {
                        p.run_with_prelocked_gmegabuf(env, &mut wv.state, gmegabuf);
                    } else {
                        p.run_with(env, &mut wv.state);
                    }
                    (
                        ((env.slot_value(slots.x) * 2.0 - 1.0) * inv_aspectx as f64) as f32,
                        ((env.slot_value(slots.y) * -2.0 + 1.0) * inv_aspecty as f64) as f32,
                        env.slot_value(slots.r) as f32,
                        env.slot_value(slots.g) as f32,
                        env.slot_value(slots.b) as f32,
                        env.slot_value(slots.a) as f32,
                    )
                } else {
                    // Equation-free fast path: no environment traffic or VM call.
                    (
                        value1 * 2.0 * inv_aspectx,
                        value2 * -2.0 * inv_aspecty,
                        frame_r,
                        frame_g,
                        frame_b,
                        frame_a,
                    )
                };
                let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
                scratch.positions.push([fin(px, 0.0), fin(py, 0.0)]);
                scratch.colors.push([
                    fin(cr, frame_r).clamp(0.0, 1.0),
                    fin(cg, frame_g).clamp(0.0, 1.0),
                    fin(cb, frame_b).clamp(0.0, 1.0),
                    fin(ca, frame_a).clamp(0.0, 1.0),
                ]);
            }
            drop(point_gmegabuf_guard);

            if wv.def.use_dots {
                scratch.output.reserve(
                    scratch
                        .positions
                        .len()
                        .saturating_sub(scratch.output.capacity()),
                );
                for (p, c) in scratch.positions.iter().zip(scratch.colors.iter()) {
                    scratch.output.push(WaveVert { pos: *p, color: *c });
                }
                let draw = WaveDraw {
                    start_vert: 0,
                    count: scratch.positions.len() as u32,
                    points: true,
                    additive: wv.def.additive,
                    thick: wv.def.draw_thick || wv.def.use_dots,
                };
                Some(draw)
            } else {
                let count = emit_smoothed_wave_and_color(
                    &scratch.positions,
                    &scratch.colors,
                    &mut scratch.output,
                );
                let draw = WaveDraw {
                    start_vert: 0,
                    count,
                    points: false,
                    additive: wv.def.additive,
                    thick: wv.def.draw_thick,
                };
                Some(draw)
            }
        };
        let mut outputs = std::mem::take(&mut self.scratch.custom_wave_draws);
        outputs.clear();
        // Only dispatch onto the Rayon pool when there is more than one wave to
        // evaluate: a 0- or 1-wave preset gains nothing from parallelism and the
        // pool hand-off would only add scheduling overhead. Order-independent
        // because the guarded case is already `parallel_safe`.
        if parallel_safe && self.waves.len() > 1 {
            self.waves
                .par_iter_mut()
                .map(&build_wave)
                .collect_into_vec(&mut outputs);
        } else {
            outputs.extend(self.waves.iter_mut().map(&build_wave));
        }
        // Merge in authored wave order so GPU draw order and alpha blending remain
        // deterministic even when CPU equation evaluation completed out of order.
        for (wave, output) in self
            .waves
            .iter()
            .zip(outputs.iter().copied())
            .filter_map(|(wave, draw)| draw.map(|draw| (wave, draw)))
        {
            if verts.len() >= WAVE_VERT_CAP {
                break;
            }
            let mut draw = output;
            let remaining = WAVE_VERT_CAP - verts.len();
            let copy_len = wave.scratch.output.len().min(remaining);
            draw.start_vert = verts.len() as u32;
            draw.count = draw.count.min(copy_len as u32);
            verts.extend_from_slice(&wave.scratch.output[..copy_len]);
            if draw.count > 0 {
                draws.push(draw);
            }
        }
        self.scratch.custom_wave_draws = outputs;
    }

    fn warp_gpu_params(
        &self,
        t: f32,
        aspectx: f32,
        aspecty: f32,
        use_cpu_mesh: bool,
    ) -> WarpGpuParams {
        let b = self.base_warp;
        let getf = |k: &str, def: f32| {
            self.eel_env
                .get(k)
                .copied()
                .map(|v| v as f32)
                .unwrap_or(def)
        };
        WarpGpuParams {
            transform0: [
                getf("zoom", b.zoom),
                getf("zoomexp", b.zoomexp),
                getf("rot", b.rot),
                getf("warp", b.warp),
            ],
            transform1: [
                getf("cx", b.cx),
                getf("cy", b.cy),
                getf("dx", b.dx),
                getf("dy", b.dy),
            ],
            transform2: [
                getf("sx", b.sx),
                getf("sy", b.sy),
                getf("decay", b.decay),
                getf("warpscale", b.warpscale).max(1e-6),
            ],
            transform3: [getf("warpanimspeed", b.warpanimspeed), t, aspectx, aspecty],
            flags: [if use_cpu_mesh { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        }
    }

    /// Compute the per-vertex warped UV + decay rgb for presets that require a
    /// CPU flow field. Equation-free presets without motion vectors use the same
    /// math in the vertex shader and skip this mesh rebuild/upload entirely.
    fn compute_warp_verts(&mut self, params: &WarpGpuParams) {
        let [fzoom, fzoomexp, frot, fwarp] = params.transform0;
        let [fcx, fcy, fdx, fdy] = params.transform1;
        let [fsx, fsy, fdecay, wscale] = params.transform2;
        let [wanim, t, aspectx, aspecty] = params.transform3;

        let warp_time_v = t * wanim;
        let warp_scale_inv = 1.0_f32 / wscale;
        let warpf0 = 11.68 + 4.0 * (warp_time_v * 1.413 + 10.0).cos();
        let warpf1 = 8.77 + 3.0 * (warp_time_v * 1.113 + 7.0).cos();
        let warpf2 = 10.54 + 3.0 * (warp_time_v * 1.233 + 3.0).cos();
        let warpf3 = 11.49 + 4.0 * (warp_time_v * 0.933 + 5.0).cos();

        let ax = aspectx as f64;
        let ay = aspecty as f64;
        // EEL `aspectx`/`aspecty` are seeded INVERTED per butterchurn presetEquationRunner.
        let inv_ax = if aspectx != 0.0 {
            1.0 / aspectx as f64
        } else {
            1.0
        };
        let inv_ay = if aspecty != 0.0 {
            1.0 / aspecty as f64
        } else {
            1.0
        };

        let has_prog = self.per_pixel_prog.is_some();

        let slots = self.warp_slots;

        // Per-pixel equations start from the post-per-frame env. This carries user
        // vars like `v`, `mx`, q9..q32, etc. MilkDrop then restores only the ten
        // authored warp controls for each vertex; user temporaries intentionally
        // carry between vertices. The cross-env copy/name lookups happen once here.
        if has_prog {
            self.warp_env.copy_present_from(&self.eel_env);
            self.warp_env.insert("time", t as f64);
            self.warp_env.insert("frame", self.frame_idx as f64);
            self.warp_env
                .insert("fps", effective_fps(self.time_per_frame));
            self.warp_env.insert(
                "bass".into(),
                self.eel_env.get("bass").copied().unwrap_or(0.0),
            );
            self.warp_env.insert(
                "mid".into(),
                self.eel_env.get("mid").copied().unwrap_or(0.0),
            );
            self.warp_env.insert(
                "treb".into(),
                self.eel_env.get("treb").copied().unwrap_or(0.0),
            );
            self.warp_env.insert(
                "vol".into(),
                self.eel_env.get("vol").copied().unwrap_or(0.0),
            );
            self.warp_env.insert(
                "bass_att".into(),
                self.eel_env.get("bass_att").copied().unwrap_or(0.0),
            );
            self.warp_env.insert(
                "mid_att".into(),
                self.eel_env.get("mid_att").copied().unwrap_or(0.0),
            );
            self.warp_env.insert(
                "treb_att".into(),
                self.eel_env.get("treb_att").copied().unwrap_or(0.0),
            );
            self.warp_env.insert(
                "vol_att".into(),
                self.eel_env.get("vol_att").copied().unwrap_or(0.0),
            );
            self.warp_env.insert("aspectx", inv_ax);
            self.warp_env.insert("aspecty", inv_ay);
            let reset_values = [fwarp, fzoom, fzoomexp, fcx, fcy, fsx, fsy, fdx, fdy, frot];
            for (&slot, value) in slots.reset.iter().zip(reset_values) {
                self.warp_env.set_slot_value(slot, value as f64);
            }
            self.warp_env
                .capture_slots_into(&slots.reset, &mut self.warp_snapshot);
        }

        let vw = GRID_W + 1;
        let vh = GRID_H + 1;
        let eval_vertex =
            |index: u32, env: &mut Env, state: &mut EelState, prog: Option<&EelProgram>| {
                let i = index % vw;
                let j = index / vw;
                let x = (i as f32 / GRID_W as f32) * 2.0 - 1.0;
                let y = (j as f32 / GRID_H as f32) * 2.0 - 1.0;
                let xf = x as f64;
                let yf = y as f64;
                let rad = (xf * xf * ax * ax + yf * yf * ay * ay).sqrt();

                // Defaults (per-frame values) in case there's no program.
                let (mut zoom, mut zoomexp, mut rot, mut warp) = (fzoom, fzoomexp, frot, fwarp);
                let (mut cx, mut cy, mut dx, mut dy, mut sx, mut sy) =
                    (fcx, fcy, fdx, fdy, fsx, fsy);
                let (mut dr, mut dg, mut db) = (fdecay, fdecay, fdecay);

                if let Some(prog) = prog {
                    let ang = milkdrop_angle(xf, yf, ax, ay);
                    env.set_slot_value(slots.x, xf * 0.5 * ax + 0.5);
                    env.set_slot_value(slots.y, yf * -0.5 * ay + 0.5);
                    env.set_slot_value(slots.rad, rad);
                    env.set_slot_value(slots.ang, ang);
                    env.restore_slots(&self.warp_snapshot);
                    // Preserve OjoDrop's existing per-vertex decay extension while
                    // keeping it outside the MilkDrop ten-control snapshot.
                    env.set_slot_value(slots.decay, fdecay as f64);
                    env.set_slot_value(slots.decay_r, fdecay as f64);
                    env.set_slot_value(slots.decay_g, fdecay as f64);
                    env.set_slot_value(slots.decay_b, fdecay as f64);
                    prog.run_with(env, state);
                    warp = env.slot_value(slots.reset[0]) as f32;
                    zoom = env.slot_value(slots.reset[1]) as f32;
                    zoomexp = env.slot_value(slots.reset[2]) as f32;
                    cx = env.slot_value(slots.reset[3]) as f32;
                    cy = env.slot_value(slots.reset[4]) as f32;
                    sx = env.slot_value(slots.reset[5]) as f32;
                    sy = env.slot_value(slots.reset[6]) as f32;
                    dx = env.slot_value(slots.reset[7]) as f32;
                    dy = env.slot_value(slots.reset[8]) as f32;
                    rot = env.slot_value(slots.reset[9]) as f32;
                    dr = env.slot_value(slots.decay_r) as f32;
                    dg = env.slot_value(slots.decay_g) as f32;
                    db = env.slot_value(slots.decay_b) as f32;
                }

                if zoom.abs() < 1e-6 {
                    zoom = 1e-6;
                }
                if sx.abs() < 1e-6 {
                    sx = 1e-6;
                }
                if sy.abs() < 1e-6 {
                    sy = 1e-6;
                }

                // ── UV composition (butterchurn renderer.js runPixelEquations) ──
                let zoom2v = zoom.powf(zoomexp.powf(rad as f32 * 2.0 - 1.0));
                let zoom2inv = 1.0_f32 / zoom2v;
                let mut u = x * 0.5 * aspectx * zoom2inv + 0.5;
                let mut v = -y * 0.5 * aspecty * zoom2inv + 0.5;
                // scale about (cx,cy)
                u = (u - cx) / sx + cx;
                v = (v - cy) / sy + cy;
                // warp octaves
                if warp.abs() > 1e-9 {
                    u += warp
                        * 0.0035
                        * (warp_time_v * 0.333 + warp_scale_inv * (x * warpf0 - y * warpf3)).sin();
                    v += warp
                        * 0.0035
                        * (warp_time_v * 0.375 - warp_scale_inv * (x * warpf2 + y * warpf1)).cos();
                    u += warp
                        * 0.0035
                        * (warp_time_v * 0.753 - warp_scale_inv * (x * warpf1 - y * warpf2)).cos();
                    v += warp
                        * 0.0035
                        * (warp_time_v * 0.825 + warp_scale_inv * (x * warpf0 + y * warpf3)).sin();
                }
                // rotate about (cx,cy)
                let u2 = u - cx;
                let v2 = v - cy;
                let (sr, cr) = rot.sin_cos();
                u = u2 * cr - v2 * sr + cx;
                v = u2 * sr + v2 * cr + cy;
                // translate
                u -= dx;
                v -= dy;
                // undo aspect
                u = (u - 0.5) / aspectx + 0.5;
                v = (v - 0.5) / aspecty + 0.5;

                let px = x;
                let py = -y; // clip-space: top row (j=0, y=-1) -> py=+1
                WarpVert {
                    pos: [px, py],
                    uv: [u, v],
                    decay: [dr, dg, db, 1.0],
                }
            };

        let program = self.per_pixel_prog.as_ref();
        let mut verts = std::mem::take(&mut self.scratch.warp_verts);
        verts.clear();
        verts.reserve((vw * vh) as usize);
        for index in 0..vw * vh {
            verts.push(eval_vertex(
                index,
                &mut self.warp_env,
                &mut self.warp_state,
                program,
            ));
        }
        self.scratch.warp_verts = verts;
        if has_prog {
            // MilkDrop publishes reg00..reg99 written by the final per-pixel
            // invocation back to the preset-wide frame environment. Both slot
            // arrays were interned once at activation, so this is 100 dense
            // copies with no per-frame string allocation or hashing.
            self.eel_env.copy_slot_values_from(
                &self.eel_reg_slots,
                &self.warp_env,
                &self.warp_reg_slots,
            );
        }
    }

    /// Encode the shared blur pyramid. `source_is_write` selects the just-warped
    /// feedback page for the legacy order, or the prior feedback page for the
    /// opt-in BeatDrop provenance order.
    fn encode_blur_chain(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_is_write: bool,
    ) -> u32 {
        if self.blur_levels == 0 {
            return 0;
        }

        let source_is_a = if source_is_write {
            self.write_to_a
        } else {
            !self.write_to_a
        };
        let blur1_h_bg = if source_is_a {
            &self.blur1_h_bg_a
        } else {
            &self.blur1_h_bg_b
        };
        encode_blur_pass(
            encoder,
            "blur1-h",
            &self.blur_h_pipeline,
            blur1_h_bg,
            &self.view_btemp1,
        );
        generate_mip_chain(
            &self.device,
            &self.feedback_mip_blitter,
            encoder,
            &self.btemp_mips1,
        );
        encode_blur_pass(
            encoder,
            "blur1-v",
            &self.blur_v_pipeline,
            &self.blur1_v_bg,
            &self.view_blur1,
        );
        generate_mip_chain(
            &self.device,
            &self.feedback_mip_blitter,
            encoder,
            &self.blur_mips1,
        );
        let mut count = 2;

        if self.blur_levels >= 2 {
            encode_blur_pass(
                encoder,
                "blur2-h",
                &self.blur_h_pipeline,
                &self.blur2_h_bg,
                &self.view_btemp2,
            );
            generate_mip_chain(
                &self.device,
                &self.feedback_mip_blitter,
                encoder,
                &self.btemp_mips2,
            );
            encode_blur_pass(
                encoder,
                "blur2-v",
                &self.blur_v_pipeline,
                &self.blur2_v_bg,
                &self.view_blur2,
            );
            generate_mip_chain(
                &self.device,
                &self.feedback_mip_blitter,
                encoder,
                &self.blur_mips2,
            );
            count += 2;

            if self.blur_levels >= 3 {
                encode_blur_pass(
                    encoder,
                    "blur3-h",
                    &self.blur_h_pipeline,
                    &self.blur3_h_bg,
                    &self.view_btemp3,
                );
                generate_mip_chain(
                    &self.device,
                    &self.feedback_mip_blitter,
                    encoder,
                    &self.btemp_mips3,
                );
                encode_blur_pass(
                    encoder,
                    "blur3-v",
                    &self.blur_v_pipeline,
                    &self.blur3_v_bg,
                    &self.view_blur3,
                );
                generate_mip_chain(
                    &self.device,
                    &self.feedback_mip_blitter,
                    encoder,
                    &self.blur_mips3,
                );
                count += 2;
            }
        }
        count
    }

    /// Draw already-uploaded motion vector geometry into one feedback page.
    /// Kept separate from the authored overlay pass so opt-in provenance can
    /// place it before warp without changing legacy ordering.
    fn encode_motion_vectors(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        label: &'static str,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.mv_pipeline);
        pass.set_bind_group(0, &self.mv_bg, &[]);
        pass.set_vertex_buffer(0, self.mv_vert_buf.slice(..));
        pass.draw(0..count, 0..1);
    }

    pub fn render(&mut self, surface_view: &wgpu::TextureView) {
        self.render_impl(Some(surface_view), None, None, None, None);
    }

    /// Advance one live MilkDrop frame and retain the post-COMP Rgba8 result
    /// for a caller-owned compositor. This skips the renderer's FXAA/output
    /// pass and its full-size HDR write; [`Self::retained_comp_view`] remains
    /// valid until a resize or performance-profile scale change.
    pub fn render_to_retained_comp(&mut self) {
        self.render_impl(None, None, None, None, None);
    }

    /// Texture produced by [`Self::render_to_retained_comp`].
    pub fn retained_comp_view(&self) -> &wgpu::TextureView {
        &self.comp_view
    }

    /// Render one frame while writing timestamps around OjoDrop's GPU command
    /// stream. The caller owns query resolution so multiple frames can be
    /// collected without a synchronization point between submissions.
    pub fn render_profiled(
        &mut self,
        surface_view: &wgpu::TextureView,
        query_set: &wgpu::QuerySet,
        boundary_marker: &wgpu::Buffer,
        start_index: u32,
        end_index: u32,
    ) {
        self.render_impl(
            Some(surface_view),
            Some((query_set, boundary_marker, start_index, end_index)),
            None,
            None,
            None,
        );
    }

    /// Profiled counterpart of [`Self::render_to_retained_comp`].
    pub fn render_to_retained_comp_profiled(
        &mut self,
        query_set: &wgpu::QuerySet,
        boundary_marker: &wgpu::Buffer,
        start_index: u32,
        end_index: u32,
    ) {
        self.render_impl(
            None,
            Some((query_set, boundary_marker, start_index, end_index)),
            None,
            None,
            None,
        );
    }

    /// Render a supported native shared-feedback transition into `surface_view`.
    ///
    /// `self` is the visible outgoing state and `target` is the incoming state
    /// owned by the runtime. Both state machines advance each call. Their
    /// evaluated warp UVs (and per-vertex decay) are blended *before* one warp
    /// mesh samples the outgoing feedback history. The target is then reseeded
    /// from that shared history, so the runtime can promote it on completion or
    /// interruption without an image reset. `false` means the caller must use
    /// its existing crossfade/fallback path.
    ///
    /// Shaderless pairs may include built-in/custom waves and static
    /// untextured, borderless shapes: their independently evaluated geometry is
    /// composited at complementary opacity in the shared post-warp pass. Custom
    /// shaders, textured/dynamic shapes, motion vectors, darken-center, and
    /// frame/shape borders remain explicit fallback cases. This is not arbitrary
    /// shader-pair or full-preset transition parity.
    pub fn render_shared_feedback(
        &mut self,
        surface_view: &wgpu::TextureView,
        progress: f32,
        target: &mut Self,
    ) -> bool {
        self.render_shared_feedback_impl(Some(surface_view), progress, target)
    }

    /// Retained-COMP counterpart of [`Self::render_shared_feedback`]. This
    /// preserves the parent renderer's retained composition/direct-HDR split.
    pub fn render_shared_feedback_to_retained_comp(
        &mut self,
        progress: f32,
        target: &mut Self,
    ) -> bool {
        self.render_shared_feedback_impl(None, progress, target)
    }

    fn render_shared_feedback_impl(
        &mut self,
        surface_view: Option<&wgpu::TextureView>,
        progress: f32,
        target: &mut Self,
    ) -> bool {
        if !progress.is_finite() || !self.supports_shared_feedback(target) {
            return false;
        }
        let progress = progress.clamp(0.0, 1.0);

        // Advance the target's independent equations and evaluate its CPU warp
        // mesh without presenting its own feedback result. This currently uses a
        // complete hidden GPU frame to keep state ordering identical. Its
        // evaluated compatible overlay geometry is then redrawn into the
        // outgoing shared post-warp pass at complementary opacity.
        target.mirror_shared_feedback_inputs_from(self);
        let was_forced = target.force_cpu_warp_mesh;
        target.force_cpu_warp_mesh = true;
        target.render_impl(None, None, None, None, None);
        target.force_cpu_warp_mesh = was_forced;

        // A compatible target was force-evaluated above, so a length mismatch is
        // an internal invariant failure rather than a visually plausible image
        // blend. Return the ordinary fallback without advancing outgoing state.
        if target.scratch.warp_verts.len() != ((GRID_W + 1) * (GRID_H + 1)) as usize {
            return false;
        }
        self.render_impl(
            surface_view,
            None,
            Some((&target.scratch.warp_verts, progress)),
            Some((&target.last_comp_perframe, progress)),
            Some((&*target, progress)),
        );

        // Keep the target promotable after *every* frame, not only at p == 1.
        // This documents and implements the interruption policy: the current
        // incoming target may become the next winner with continuous feedback.
        target.seed_feedback_from(self)
    }

    fn render_impl(
        &mut self,
        surface_view: Option<&wgpu::TextureView>,
        timestamp_writes: Option<(&wgpu::QuerySet, &wgpu::Buffer, u32, u32)>,
        shared_warp: Option<(&[WarpVert], f32)>,
        shared_comp: Option<(&PerFrame, f32)>,
        shared_overlays: Option<(&Self, f32)>,
    ) {
        let t = self
            .frame_time_override
            .take()
            .or_else(|| deterministic_time_seconds(self.frame_idx, self.time_per_frame))
            .unwrap_or_else(|| self.start.elapsed().as_secs_f64());
        let shader_t = shader_time_seconds(t);
        let shader_frame = shader_frame_index(self.frame_idx);
        let progress = shader_progress(t);
        // Internal render size drives the texsize uniform: shaders sample the
        // feedback/blur targets, which are sized at the render (scaled) resolution.
        let (w, h) = (self.render_w as f32, self.render_h as f32);
        // Butterchurn shader uniform convention: aspect.xy hold the geometry aspect
        // factors and aspect.zw hold their inverses. Geometry paths below use the
        // same values, so custom shaders and CPU geometry agree.
        let (shape_aspectx, shape_aspecty) = if self.width > self.height {
            (1.0f32, self.height as f32 / self.width as f32)
        } else {
            (self.width as f32 / self.height as f32, 1.0f32)
        };
        let inv_aspectx = if shape_aspectx != 0.0 {
            1.0 / shape_aspectx
        } else {
            1.0
        };
        let inv_aspecty = if shape_aspecty != 0.0 {
            1.0 / shape_aspecty
        } else {
            1.0
        };

        // Audio reactivity: live mic features when supplied, else synthetic sine
        // waves at different frequencies so offscreen/headless renders still animate.
        let (bass, mid, treb, vol) = match self.audio {
            Some([b, m, tr, v]) => (b as f64, m as f64, tr as f64, v as f64),
            None => {
                let bass = (1.0 + (t * 1.3).sin()) as f64;
                let mid = (1.0 + (t * 2.1).sin()) as f64;
                let treb = (1.0 + (t * 3.7).sin()) as f64;
                let vol = (bass + mid + treb) / 3.0;
                (bass, mid, treb, vol)
            }
        };
        // Attenuated (smoothed) envelopes. When no live att was supplied
        // (headless/synthetic), mirror the non-att values so deterministic renders
        // are bit-identical to before — only the live path drives distinct *_att.
        let (bass_att, mid_att, treb_att, vol_att) = match self.audio_att {
            Some([b, m, tr, v]) => (b as f64, m as f64, tr as f64, v as f64),
            None => (bass, mid, treb, vol),
        };

        // Run EEL2 per-frame equations
        if let Some(prog) = &self.eel_program {
            let env = &mut self.eel_env;
            // Reset q1..q32 to their post-init values BEFORE per-frame runs, so
            // accumulator-q presets re-seed from init each frame instead of
            // carrying the previous frame's q's (Butterchurn mdVS reset). User
            // vars and regs are NOT reset (they persist). No-op when no init ran.
            for (slot, value) in self.eel_q_slots.iter().zip(&self.q_init) {
                env.set_slot_value(*slot, *value);
            }
            // Reset built-in WARP motion vars to their header baseVals each frame,
            // matching Butterchurn (mdVSFrame = mdVS baseVals + qInit + USER keys only;
            // built-in motion vars do NOT persist — only user vars/megabuf/regs do).
            // Without this, accumulator presets (`zoom = zoom + ...`) compound every
            // frame and run away: idx 313's zoom blows up by ~f70, the feedback then
            // samples a magnified central pixel, flattens, and collapses to black.
            let bw = self.base_warp;
            env.insert("zoom".into(), bw.zoom as f64);
            env.insert("zoomexp".into(), bw.zoomexp as f64);
            env.insert("rot".into(), bw.rot as f64);
            env.insert("warp".into(), bw.warp as f64);
            env.insert("cx".into(), bw.cx as f64);
            env.insert("cy".into(), bw.cy as f64);
            env.insert("dx".into(), bw.dx as f64);
            env.insert("dy".into(), bw.dy as f64);
            env.insert("sx".into(), bw.sx as f64);
            env.insert("sy".into(), bw.sy as f64);
            env.insert("warpscale".into(), bw.warpscale as f64);
            env.insert("warpanimspeed".into(), bw.warpanimspeed as f64);
            env.insert("decay".into(), bw.decay as f64);
            env.insert("wrap".into(), if bw.wrap { 1.0 } else { 0.0 });
            // Reset built-in waveform fields from header baseVals each frame, then
            // let per-frame EEL override them. Butterchurn's basic waveform reads
            // these live mdVSFrame values, so fields like wave_x/wave_mystery should
            // not persist as ordinary user variables from the previous frame.
            env.insert("wave_mode".into(), self.bw_mode as f64);
            env.insert("wave_x".into(), self.bw_x as f64);
            env.insert("wave_y".into(), self.bw_y as f64);
            env.insert("wave_r".into(), self.bw_r as f64);
            env.insert("wave_g".into(), self.bw_g as f64);
            env.insert("wave_b".into(), self.bw_b as f64);
            env.insert("wave_a".into(), self.bw_a as f64);
            env.insert("wave_mystery".into(), self.bw_mystery as f64);
            env.insert("wave_scale".into(), self.bw_scale as f64);
            env.insert("wave_smoothing".into(), self.bw_smoothing as f64);
            env.insert("wave_dots".into(), if self.bw_dots { 1.0 } else { 0.0 });
            env.insert("wave_thick".into(), if self.bw_thick { 1.0 } else { 0.0 });
            env.insert(
                "additivewave".into(),
                if self.bw_additive { 1.0 } else { 0.0 },
            );
            env.insert(
                "wave_brighten".into(),
                if self.bw_brighten { 1.0 } else { 0.0 },
            );
            env.insert(
                "modwavealphabyvolume".into(),
                if self.bw_modalphavol { 1.0 } else { 0.0 },
            );
            env.insert("modwavealphastart".into(), self.bw_modalphastart as f64);
            env.insert("modwavealphaend".into(), self.bw_modalphaend as f64);
            env.insert("b1n".into(), self.b1n as f64);
            env.insert("b1x".into(), self.b1x as f64);
            env.insert("b1ed".into(), self.b1ed as f64);
            env.insert("b2n".into(), self.b2n as f64);
            env.insert("b2x".into(), self.b2x as f64);
            env.insert("b3n".into(), self.b3n as f64);
            env.insert("b3x".into(), self.b3x as f64);
            // Reconstruct every remaining built-in from the preset base each
            // frame. These are not user variables and must not accidentally
            // accumulate their previous per-frame output.
            for (name, value) in [
                ("gamma", self.comp_gamma_adj),
                ("gammaadj", self.comp_gamma_adj),
                ("fshader", self.comp_fshader),
                ("echo_zoom", self.echo_zoom),
                ("echo_alpha", self.echo_alpha),
                ("echo_orient", self.echo_orient),
                ("mv_x", self.mv_x),
                ("mv_y", self.mv_y),
                ("mv_dx", self.mv_dx),
                ("mv_dy", self.mv_dy),
                ("mv_l", self.mv_l),
                ("mv_r", self.mv_r),
                ("mv_g", self.mv_g),
                ("mv_b", self.mv_b),
                ("mv_a", self.mv_a),
                ("ob_size", self.ob_size),
                ("ob_r", self.ob_r),
                ("ob_g", self.ob_g),
                ("ob_b", self.ob_b),
                ("ob_a", self.ob_a),
                ("ib_size", self.ib_size),
                ("ib_r", self.ib_r),
                ("ib_g", self.ib_g),
                ("ib_b", self.ib_b),
                ("ib_a", self.ib_a),
            ] {
                env.insert(name, value as f64);
            }
            env.insert("darken_center", if self.darken_center { 1.0 } else { 0.0 });
            // Reset comp post-FX flags from baseVals each frame so per-frame EEL
            // can animate them without stale persistence. Butterchurn reads these
            // uniforms from mdVSFrame after frame equations.
            env.insert(
                "brighten".into(),
                if self.comp_brighten { 1.0 } else { 0.0 },
            );
            env.insert("darken".into(), if self.comp_darken { 1.0 } else { 0.0 });
            env.insert(
                "solarize".into(),
                if self.comp_solarize { 1.0 } else { 0.0 },
            );
            env.insert("invert".into(), if self.comp_invert { 1.0 } else { 0.0 });
            // Seed read-only inputs before each frame
            env.insert("time".into(), t as f64);
            env.insert("fps".into(), effective_fps(self.time_per_frame));
            env.insert("frame".into(), self.frame_idx as f64);
            env.insert("progress".into(), (t % 30.0) as f64 / 30.0);
            env.insert("bass".into(), bass);
            env.insert("mid".into(), mid);
            env.insert("treb".into(), treb);
            env.insert("vol".into(), vol);
            env.insert("bass_att".into(), bass_att);
            env.insert("mid_att".into(), mid_att);
            env.insert("treb_att".into(), treb_att);
            env.insert("vol_att".into(), vol_att);
            // MilkDrop pseudo-var `diff`: frame-to-frame volume delta. Not a standard
            // seeded EEL input, but presets like orb_waaa gate mv_a on above(diff,10),
            // so seed it from the volume change to honor the preset's intent.
            env.insert("diff".into(), (vol - self.vol_prev).abs());
            // Butterchurn seeds the per-frame EEL aspectx/aspecty as the INVERTED geometry
            // aspect (matching the shape/wave/warp envs) + mesh/pixel dims (presetEquation
            // Runner mdVSBase). These were never seeded into the per-frame env, so per-frame
            // eqs reading aspectx/aspecty silently got 0 (idx 2007/4005/8637).
            let (gax, gay) = if self.width >= self.height {
                (1.0, self.height as f64 / self.width.max(1) as f64)
            } else {
                (self.width as f64 / self.height.max(1) as f64, 1.0)
            };
            env.insert("aspectx".into(), if gax != 0.0 { 1.0 / gax } else { 1.0 });
            env.insert("aspecty".into(), if gay != 0.0 { 1.0 / gay } else { 1.0 });
            env.insert("meshx".into(), GRID_W as f64);
            env.insert("meshy".into(), GRID_H as f64);
            env.insert("pixelsx".into(), self.render_w as f64);
            env.insert("pixelsy".into(), self.render_h as f64);
            prog.run_with(env, &mut self.eel_state);
        }
        self.vol_prev = vol;

        // Helper: read f64 var from EEL env, default 0.
        //
        // Finiteness is tested AFTER the narrowing cast, for the same reason as
        // `eqd` below — see `finite_f32_from_f64`. This closure carried the
        // identical pre-cast bug until 2026-08-15: `1e300_f64.is_finite()` is
        // true and `1e300_f64 as f32` is `+inf`, so a per-frame EEL `q1 = 1e300`
        // reached `_qa.._qh` (q1..q32), `f_shader`, `echo_alpha` and
        // `echo_orientation` in `comp_pf` — the same `comp_perframe_buf` the blur
        // guard protects.
        let eq = |k: &str| finite_f32_from_f64(self.eel_env.get(k).copied().unwrap_or(0.0), 0.0);
        // Helper: read f64 var from EEL env, default to `def` if missing.
        // Finiteness is tested AFTER the narrowing cast — see
        // `finite_f32_from_f64`; testing the f64 lets `1e300` through as `+inf`.
        let eqd = |k: &str, def: f64| {
            finite_f32_from_f64(self.eel_env.get(k).copied().unwrap_or(def), def)
        };
        let base_gamma = self.comp_gamma_adj as f64;
        let gamma = eqd("gamma", base_gamma);
        let gammaadj = eqd("gammaadj", base_gamma);
        let live_gamma = if (gammaadj - self.comp_gamma_adj).abs() > 1.0e-6 {
            gammaadj
        } else {
            gamma
        };

        // Snapshot live motion-vector / border / darken values from the per-frame EEL
        // env NOW (while eqd's immutable borrow is valid), before any &mut self call
        // (build_wave_geometry / compute_warp_verts). Used later to build geometry.
        let live_mv_a = eqd("mv_a", self.mv_a as f64);
        let live_mv_x = eqd("mv_x", self.mv_x as f64);
        let live_mv_y = eqd("mv_y", self.mv_y as f64);
        let live_mv_dx = eqd("mv_dx", self.mv_dx as f64);
        let live_mv_dy = eqd("mv_dy", self.mv_dy as f64);
        let live_mv_l = eqd("mv_l", self.mv_l as f64);
        let live_mv_r = eqd("mv_r", self.mv_r as f64);
        let live_mv_g = eqd("mv_g", self.mv_g as f64);
        let live_mv_b = eqd("mv_b", self.mv_b as f64);
        let live_darken = eqd("darken_center", if self.darken_center { 1.0 } else { 0.0 }) != 0.0;
        let live_ob_size = eqd("ob_size", self.ob_size as f64);
        let live_ob_a = eqd("ob_a", self.ob_a as f64);
        let live_ib_size = eqd("ib_size", self.ib_size as f64);
        let live_ib_a = eqd("ib_a", self.ib_a as f64);
        let live_outer_color = [
            eqd("ob_r", self.ob_r as f64),
            eqd("ob_g", self.ob_g as f64),
            eqd("ob_b", self.ob_b as f64),
            live_ob_a,
        ];
        let live_inner_color = [
            eqd("ib_r", self.ib_r as f64),
            eqd("ib_g", self.ib_g as f64),
            eqd("ib_b", self.ib_b as f64),
            live_ib_a,
        ];
        let live_wrap = eqd("wrap", if self.base_warp.wrap { 1.0 } else { 0.0 }) != 0.0;

        // ── Blur min/max range remap (butterchurn getBlurValues + getScaleAndBias) ──
        // The blur shader normalizes each level into [0,1] (scale_n,bias_n); the
        // comp/warp GetBlurN helpers apply the inverse (scale1..3, bias1..3) to recover
        // the original range. At defaults (min 0, max 1) both halves are identity.
        let (blur_sb, comp_blur) = {
            let (bmin, bmax) = blur_min_max_remap(
                [
                    eqd("b1n", self.b1n as f64),
                    eqd("b2n", self.b2n as f64),
                    eqd("b3n", self.b3n as f64),
                ],
                [
                    eqd("b1x", self.b1x as f64),
                    eqd("b2x", self.b2x as f64),
                    eqd("b3x", self.b3x as f64),
                ],
            );
            // blur-shader scale/bias (normalize into [0,1]) — butterchurn getScaleAndBias.
            let (scale, bias) = blur_scale_and_bias(bmin, bmax);
            (
                [scale, bias],
                // comp uniforms: blur1_min/max + scale1/2/3 + bias1/2/3
                guard_finite_comp_blur(
                    bmin,
                    bmax,
                    [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]],
                    [bmin[0], bmin[1], bmin[2]],
                ),
            )
        };

        // Build the shared portion of the warp/comp uniforms now. rand_frame is
        // filled after the corresponding EEL phases so shared RNG consumption
        // matches Butterchurn's observable lifecycle.
        let roam_pair = |rate: f32| {
            let (sin, cos) = (shader_t * rate).sin_cos();
            (0.5 + 0.5 * cos, 0.5 + 0.5 * sin)
        };
        let slow_roam = [
            roam_pair(0.005),
            roam_pair(0.008),
            roam_pair(0.013),
            roam_pair(0.022),
        ];
        let roam = [
            roam_pair(0.3),
            roam_pair(1.3),
            roam_pair(5.0),
            roam_pair(20.0),
        ];
        let mut pf = PerFrame {
            texsize: [w, h, 1.0 / w, 1.0 / h],
            aspect: [shape_aspectx, shape_aspecty, inv_aspectx, inv_aspecty],
            // Time-based roam oscillators in [0,1] (butterchurn warp.js:838-861 / comp.js).
            // Were zeroed (..Zeroable) → roam-using warp/comp collapsed to grayscale/dim.
            slow_roam_cos: [
                slow_roam[0].0,
                slow_roam[1].0,
                slow_roam[2].0,
                slow_roam[3].0,
            ],
            roam_cos: [roam[0].0, roam[1].0, roam[2].0, roam[3].0],
            slow_roam_sin: [
                slow_roam[0].1,
                slow_roam[1].1,
                slow_roam[2].1,
                slow_roam[3].1,
            ],
            roam_sin: [roam[0].1, roam[1].1, roam[2].1, roam[3].1],
            rand_frame: [0.0; 4],
            rand_start: self.rand_start,
            rand_preset: self.rand_preset,
            // q1-q32 mapped to _qa.._qh (slots already reserved in the UBO; the
            // q9-q32 #defines live in preprocess.rs milk_fs_preamble).
            _qa: [eq("q1"), eq("q2"), eq("q3"), eq("q4")],
            _qb: [eq("q5"), eq("q6"), eq("q7"), eq("q8")],
            _qc: [eq("q9"), eq("q10"), eq("q11"), eq("q12")],
            _qd: [eq("q13"), eq("q14"), eq("q15"), eq("q16")],
            _qe: [eq("q17"), eq("q18"), eq("q19"), eq("q20")],
            _qf: [eq("q21"), eq("q22"), eq("q23"), eq("q24")],
            _qg: [eq("q25"), eq("q26"), eq("q27"), eq("q28")],
            _qh: [eq("q29"), eq("q30"), eq("q31"), eq("q32")],
            time: shader_t,
            fps: effective_fps(self.time_per_frame) as f32,
            frame: shader_frame,
            progress,
            bass: bass as f32,
            mid: mid as f32,
            treb: treb as f32,
            vol: vol as f32,
            bass_att: bass_att as f32,
            mid_att: mid_att as f32,
            treb_att: treb_att as f32,
            vol_att: vol_att as f32,
            // EEL/Butterchurn per-frame var names are `gamma` and `fshader` (no
            // underscore); reading `gamma_adj`/`f_shader` always missed → gamma was a
            // no-op (1.0) and fshader stuck at 0 (so hue gating couldn't work).
            gamma_adj: live_gamma,
            f_shader: eq("fshader"),
            echo_zoom: eqd("echo_zoom", 1.0),
            echo_alpha: eq("echo_alpha"),
            // EEL/Butterchurn var name is "echo_orient" (the UBO field is named
            // echo_orientation). render() previously read "echo_orientation", a var no
            // preset ever sets via EEL2, so echo orientation was always silently 0.
            echo_orientation: eq("echo_orient"),
            // comp_blur = (mins, maxs, scales, biases) from the per-level blur remap.
            blur1_min: comp_blur.0[0],
            blur1_max: comp_blur.1[0],
            blur2_min: comp_blur.0[1],
            blur2_max: comp_blur.1[1],
            blur3_min: comp_blur.0[2],
            blur3_max: comp_blur.1[2],
            scale1: comp_blur.2[0],
            scale2: comp_blur.2[1],
            scale3: comp_blur.2[2],
            bias1: comp_blur.3[0],
            bias2: comp_blur.3[1],
            bias3: comp_blur.3[2],
            brighten: eqd("brighten", if self.comp_brighten { 1.0 } else { 0.0 }),
            darken: eqd("darken", if self.comp_darken { 1.0 } else { 0.0 }),
            solarize: eqd("solarize", if self.comp_solarize { 1.0 } else { 0.0 }),
            invert: eqd("invert", if self.comp_invert { 1.0 } else { 0.0 }),
            audio_nyquist_hz: self.enhanced_audio_nyquist_hz,
            ..bytemuck::Zeroable::zeroed()
        };

        let (scale, bias) = guard_finite_blur_scale_bias(blur_sb[0], blur_sb[1]);
        let live_b1ed_raw = eqd("b1ed", self.b1ed as f64);
        let live_b1ed = if live_b1ed_raw.is_finite() {
            live_b1ed_raw.clamp(0.0, 1.0)
        } else {
            self.b1ed.clamp(0.0, 1.0)
        };
        let edges = [
            [1.0f32 - live_b1ed, live_b1ed, 5.0f32, 0.0f32],
            [1.0f32, 0.0f32, 5.0f32, 0.0f32],
            [1.0f32, 0.0f32, 5.0f32, 0.0f32],
        ];
        for (ubo, lvl) in [
            (&self.blur1_ubo, 0usize),
            (&self.blur2_ubo, 1),
            (&self.blur3_ubo, 2),
        ]
        .into_iter()
        .take(self.blur_levels as usize)
        {
            // `edges` and scale/bias occupy adjacent vec4s in BlurParams. Submit
            // one queue write per live level, and skip inactive levels entirely.
            let params = [
                edges[lvl][0],
                edges[lvl][1],
                edges[lvl][2],
                edges[lvl][3],
                scale[lvl],
                bias[lvl],
                0.0f32,
                0.0f32,
            ];
            self.queue
                .write_buffer(ubo, 16, bytemuck::cast_slice(&params));
        }

        // Evaluate the warp mesh before custom shapes/waves so reg00..reg99
        // written by the final per-pixel invocation are visible to those pools in
        // the same frame, matching MilkDrop's equation-runner lifecycle.
        let requested_mv_x = live_mv_x.floor() as i32;
        let requested_mv_y = live_mv_y.floor() as i32;
        let motion_vectors_requested =
            self.mv_on && live_mv_a > 0.001 && requested_mv_x > 0 && requested_mv_y > 0;
        // Native shared-feedback morphs need both states' evaluated meshes even
        // for equation-free presets. The ordinary renderer keeps its legacy
        // shader-side mesh fast path unless a per-pixel program or motion vectors
        // already require CPU evaluation.
        let use_cpu_mesh = self.per_pixel_prog.is_some()
            || motion_vectors_requested
            || self.force_cpu_warp_mesh
            || shared_warp.is_some();
        let warp_params =
            self.warp_gpu_params(shader_t, shape_aspectx, shape_aspecty, use_cpu_mesh);
        self.queue
            .write_buffer(&self.warp_params_buf, 0, bytemuck::bytes_of(&warp_params));
        if use_cpu_mesh {
            self.compute_warp_verts(&warp_params);
            if let Some((target_mesh, progress)) = shared_warp {
                let blended = blend_evaluated_warp_mesh(
                    &mut self.scratch.warp_verts,
                    target_mesh,
                    progress,
                );
                debug_assert!(blended, "compatible shared-feedback meshes must match");
            }
            self.queue.write_buffer(
                &self.warp_vert_buf,
                0,
                bytemuck::cast_slice(&self.scratch.warp_verts),
            );
        }
        // Parity captures may provide the two vectors Butterchurn uploaded for
        // this frame. In normal playback these continue to come from OjoDrop's
        // preset-owned EEL stream.
        let frame_random_override = self.frame_random_override.take();
        let warp_rand_frame = frame_random_override
            .as_ref()
            .map(|(warp, _)| *warp)
            .unwrap_or_else(|| std::array::from_fn(|_| self.eel_rng.next_unit() as f32));
        let regsnap = std::array::from_fn(|i| self.eel_env.slot_value(self.eel_reg_slots[i]));

        // ── Build shape + waveform geometry (BEFORE any render pass opens) ────
        // Shapes use aspecty (landscape: h/w) to keep discs round. Custom waves use
        // the inverse-aspect convention (butterchurn invAspectx/invAspecty).

        // q1..q32 snapshot for shape per-frame programs (MilkDrop/Butterchurn pass the
        // full q1..q32 from mdVSQAfterFrame to custom shapes; capping at q8 left q9..q32
        // = 0 in shape eqs, e.g. idx 7550's `a = floor(rand(floor(q30)))/5` → alpha 0).
        let qsnap = std::array::from_fn(|i| self.eel_env.slot_value(self.eel_q_slots[i]));

        let (mut fill_verts, fill_draws, border_verts, border_draws) = self.build_shape_geometry(
            t,
            bass,
            mid,
            treb,
            vol,
            bass_att,
            mid_att,
            treb_att,
            shape_aspectx,
            shape_aspecty,
            &qsnap,
            &regsnap,
        );

        // Keep the renderer-owned audio rows out of the struct while waveform
        // geometry is built. This avoids cloning live PCM + FFT vectors per frame;
        // they are restored unchanged below so their capacity is reused next frame.
        let live_waveform = !self.wave_l.is_empty() && !self.wave_r.is_empty();
        let mut wave_l = std::mem::take(&mut self.wave_l);
        let mut wave_r = std::mem::take(&mut self.wave_r);
        if !live_waveform {
            Self::synthesize_waveform(shader_t, &mut wave_l, &mut wave_r);
        }
        // freqArray for bSpectrum custom waves (mono, 512 bins). Empty in the
        // headless/synthetic path → build_custom_waves falls back to time data.
        let freq = std::mem::take(&mut self.freq_spectrum);
        let (mut wave_verts, wave_draws, custom_wave_extent) = self.build_wave_geometry(
            t,
            bass,
            mid,
            treb,
            vol,
            bass_att,
            mid_att,
            treb_att,
            shape_aspectx,
            shape_aspecty,
            inv_aspectx,
            inv_aspecty,
            &wave_l,
            &wave_r,
            &freq,
            &regsnap,
        );
        if !live_waveform {
            // Preserve the allocated scratch capacity but make the next frame
            // synthesize fresh time-varying PCM rather than treating it as live.
            wave_l.clear();
            wave_r.clear();
        }
        self.wave_l = wave_l;
        self.wave_r = wave_r;
        self.freq_spectrum = freq;

        // The collector's closure is never called unless explicitly enabled, so
        // the default renderer does not scan geometry or count enabled pools.
        // Custom-wave geometry is a prefix because build_wave_geometry appends
        // the built-in waveform only after all custom wave pools.
        let diagnostic_frame_index = self.frame_idx;
        let shapes = &self.shapes;
        let waves = &self.waves;
        let geometry_diagnostics = &mut self.geometry_diagnostics;
        geometry_diagnostics.capture(|| {
            let extent = custom_wave_extent.unwrap_or_default();
            summarize_custom_geometry(
                diagnostic_frame_index,
                shapes
                    .iter()
                    .filter(|shape| shape.base.enabled != 0)
                    .count(),
                &fill_verts,
                &fill_draws,
                &border_verts,
                &border_draws,
                waves.iter().filter(|wave| wave.def.enabled).count(),
                &wave_verts[..extent.vertices.min(wave_verts.len())],
                &wave_draws[..extent.draws.min(wave_draws.len())],
            )
        });

        let comp_rand_frame = frame_random_override
            .map(|(_, comp)| comp)
            .unwrap_or_else(|| std::array::from_fn(|_| self.eel_rng.next_unit() as f32));
        pf.rand_frame = warp_rand_frame;
        self.queue
            .write_buffer(&self.perframe_buf, 0, bytemuck::bytes_of(&pf));
        let mut comp_pf = pf;
        comp_pf.rand_frame = comp_rand_frame;
        if let Some((incoming_comp, progress)) = shared_comp {
            let _ = blend_comp_perframe(&mut comp_pf, incoming_comp, progress);
        }
        self.last_comp_perframe = comp_pf;
        self.queue
            .write_buffer(&self.comp_perframe_buf, 0, bytemuck::bytes_of(&comp_pf));
        generate_comp_verts(shader_t, self.rand_start, &mut self.scratch.comp_verts);
        self.queue.write_buffer(
            &self.comp_vert_buf,
            0,
            bytemuck::cast_slice(&self.scratch.comp_verts),
        );

        // The shared-feedback path has one post-warp feedback page. Preserve
        // both independently evaluated overlay states by weighting outgoing
        // vertices here, then placing weighted incoming copies in the target's
        // existing GPU buffers below. We intentionally retain the target's
        // unweighted CPU geometry so it is valid immediately if promoted.
        if let Some((target, transition_progress)) = shared_overlays {
            let incoming_weight = transition_progress.clamp(0.0, 1.0);
            weight_shape_overlay_vertices(&mut fill_verts, 1.0 - incoming_weight);
            weight_wave_overlay_vertices(&mut wave_verts, 1.0 - incoming_weight);

            let shared_shapes = &mut self.scratch.shared_shape_fill_verts;
            shared_shapes.clear();
            shared_shapes.extend_from_slice(&target.scratch.shape_fill_verts);
            weight_shape_overlay_vertices(shared_shapes, incoming_weight);
            if !shared_shapes.is_empty() {
                let n = shared_shapes.len().min(SHAPE_VERT_CAP);
                self.queue.write_buffer(
                    &target.shape_vert_buf,
                    0,
                    bytemuck::cast_slice(&shared_shapes[..n]),
                );
            }

            let shared_waves = &mut self.scratch.shared_wave_verts;
            shared_waves.clear();
            shared_waves.extend_from_slice(&target.scratch.wave_verts);
            weight_wave_overlay_vertices(shared_waves, incoming_weight);
            if !shared_waves.is_empty() {
                let n = shared_waves.len().min(WAVE_VERT_CAP);
                self.queue.write_buffer(
                    &target.wave_vert_buf,
                    0,
                    bytemuck::cast_slice(&shared_waves[..n]),
                );
            }
        }

        // Upload all geometry up-front (no write_buffer inside a render pass).
        if !fill_verts.is_empty() {
            let n = fill_verts.len().min(SHAPE_VERT_CAP);
            self.queue.write_buffer(
                &self.shape_vert_buf,
                0,
                bytemuck::cast_slice(&fill_verts[..n]),
            );
        }
        if !border_verts.is_empty() {
            let n = border_verts.len().min(BORDER_VERT_CAP);
            self.queue.write_buffer(
                &self.border_vert_buf,
                0,
                bytemuck::cast_slice(&border_verts[..n]),
            );
        }
        if !wave_verts.is_empty() {
            let n = wave_verts.len().min(WAVE_VERT_CAP);
            self.queue.write_buffer(
                &self.wave_vert_buf,
                0,
                bytemuck::cast_slice(&wave_verts[..n]),
            );
        }

        // One texel-size uniform; the vertex shader expands thick lines/dots from
        // `instance_index`, replacing 4/9 CPU draw calls with one instanced draw.
        let tsx = 2.0 / self.render_w as f32;
        let tsy = 2.0 / self.render_h as f32;
        self.queue.write_buffer(
            &self.wave_off_buf,
            0,
            bytemuck::cast_slice(&[tsx, tsy, 0.0, 0.0]),
        );

        let border_slots_needed = border_draws.len().saturating_mul(BORDER_THICK_LINE_PASSES);
        let border_slots = border_slots_needed.min(BORDER_UNIFORM_SLOTS);
        if border_slots > 0 {
            let border_offsets = [
                [0.0f32, 0.0, 0.0, 0.0],
                [tsx, 0.0, 0.0, 0.0],
                [0.0, tsy, 0.0, 0.0],
                [tsx, tsy, 0.0, 0.0],
            ];
            let bu_bytes = &mut self.scratch.border_uniform_bytes;
            bu_bytes.clear();
            bu_bytes.resize(border_slots * 256, 0);
            for (draw_idx, draw) in border_draws.iter().enumerate() {
                let base_slot = draw_idx * BORDER_THICK_LINE_PASSES;
                if base_slot >= border_slots {
                    break;
                }
                let slots_for_draw = (border_slots - base_slot).min(BORDER_THICK_LINE_PASSES);
                for (k, o) in border_offsets.iter().take(slots_for_draw).enumerate() {
                    let bu = BorderU {
                        color: draw.color,
                        offset: *o,
                    };
                    let slot = base_slot + k;
                    bu_bytes[slot * 256..slot * 256 + std::mem::size_of::<BorderU>()]
                        .copy_from_slice(bytemuck::bytes_of(&bu));
                }
            }
            self.queue
                .write_buffer(&self.border_uniform_buf, 0, bu_bytes);
        }

        // ── MOTION VECTORS geometry (butterchurn MotionVectors.generateMotionVectors)
        // Reuses the CPU warp mesh as the flow field. Live mv_* values come from the
        // per-frame EEL env; storage is retained in RendererScratch across frames.
        let mv_count: u32 = {
            let mv_a = live_mv_a;
            let mv_x = live_mv_x;
            let mv_y = live_mv_y;
            let mv_dx = live_mv_dx;
            let mv_dy = live_mv_dy;
            let mv_l = live_mv_l;
            let mv_r = live_mv_r;
            let mv_g = live_mv_g;
            let mv_b = live_mv_b;
            if motion_vectors_requested {
                let mut n_x = requested_mv_x;
                let mut n_y = requested_mv_y;
                let mut dx = mv_x - n_x as f32;
                let mut dy = mv_y - n_y as f32;
                if n_x > 64 {
                    n_x = 64;
                    dx = 0.0;
                }
                if n_y > 48 {
                    n_y = 48;
                    dy = 0.0;
                }
                let dx2 = mv_dx;
                let dy2 = mv_dy;
                let len_mult = mv_l;
                let min_len = 1.0 / self.render_w as f32;

                // Bilinear sample of the warp UV field; returns (fx2, 1.0-fy2) (V flip,
                // matching butterchurn getMotionDir). Mesh = GRID_W x GRID_H.
                let mw = GRID_W as f32;
                let mh = GRID_H as f32;
                let grid_x1 = (GRID_W + 1) as usize;
                let warp_verts = &self.scratch.warp_verts;
                let sample = |fx: f32, fy: f32| -> (f32, f32) {
                    let mut x0 = (fx * mw).floor() as i32;
                    let mut y0 = (fy * mh).floor() as i32;
                    let ddx = fx * mw - x0 as f32;
                    let ddy = fy * mh - y0 as f32;
                    // clamp to valid vertex indices [0, GRID]
                    let gx = GRID_W as i32;
                    let gy = GRID_H as i32;
                    if x0 < 0 {
                        x0 = 0;
                    }
                    if y0 < 0 {
                        y0 = 0;
                    }
                    let x1 = (x0 + 1).min(gx);
                    let y1 = (y0 + 1).min(gy);
                    let x0 = x0.min(gx);
                    let y0 = y0.min(gy);
                    let uv = |col: i32, row: i32| -> (f32, f32) {
                        let idx = (row as usize) * grid_x1 + (col as usize);
                        let v = warp_verts[idx].uv;
                        (v[0], v[1])
                    };
                    let (u00, v00) = uv(x0, y0);
                    let (u10, v10) = uv(x1, y0);
                    let (u01, v01) = uv(x0, y1);
                    let (u11, v11) = uv(x1, y1);
                    let fx2 = u00 * (1.0 - ddx) * (1.0 - ddy)
                        + u10 * ddx * (1.0 - ddy)
                        + u01 * (1.0 - ddx) * ddy
                        + u11 * ddx * ddy;
                    let fy2 = v00 * (1.0 - ddx) * (1.0 - ddy)
                        + v10 * ddx * (1.0 - ddy)
                        + v01 * (1.0 - ddx) * ddy
                        + v11 * ddx * ddy;
                    (fx2, 1.0 - fy2)
                };

                let mv_verts = &mut self.scratch.motion_verts;
                mv_verts.clear();
                for j in 0..n_y {
                    let mut fy = (j as f32 + 0.25) / (n_y as f32 + dy + 0.25 - 1.0);
                    fy -= dy2;
                    if fy > 0.0001 && fy < 0.9999 {
                        for i in 0..n_x {
                            let mut fx = (i as f32 + 0.25) / (n_x as f32 + dx + 0.25 - 1.0);
                            fx += dx2;
                            if fx > 0.0001 && fx < 0.9999 {
                                let (fx2s, fy2s) = sample(fx, fy);
                                let mut dxi = (fx2s - fx) * len_mult;
                                let mut dyi = (fy2s - fy) * len_mult;
                                let fdist = (dxi * dxi + dyi * dyi).sqrt();
                                if fdist < min_len && fdist > 0.00000001 {
                                    let g = min_len / fdist;
                                    dxi *= g;
                                    dyi *= g;
                                } else {
                                    // VERBATIM butterchurn bug (lines 6828-6829):
                                    // dxi = minLen twice; dyi is NOT reset (keeps its
                                    // scaled value). Replicated exactly for parity.
                                    #[allow(unused_assignments)]
                                    {
                                        dxi = min_len;
                                    }
                                    dxi = min_len;
                                }
                                let efx2 = fx + dxi;
                                let efy2 = fy + dyi;
                                // NDC: x = 2*fx-1; y = 1.0-2*fy (negated vs butterchurn to
                                // match our compute_warp_verts y-down→y-up mapping).
                                let vx1 = 2.0 * fx - 1.0;
                                let vy1 = 1.0 - 2.0 * fy;
                                let vx2 = 2.0 * efx2 - 1.0;
                                let vy2 = 1.0 - 2.0 * efy2;
                                mv_verts.push(MVVert { pos: [vx1, vy1] });
                                mv_verts.push(MVVert { pos: [vx2, vy2] });
                            }
                        }
                    }
                }
                let cnt = mv_verts.len().min(MV_VERT_CAP);
                if cnt > 0 {
                    self.queue.write_buffer(
                        &self.mv_vert_buf,
                        0,
                        bytemuck::cast_slice(&mv_verts[..cnt]),
                    );
                    let col = MVColor {
                        color: [mv_r, mv_g, mv_b, mv_a],
                    };
                    self.queue
                        .write_buffer(&self.mv_color_buf, 0, bytemuck::bytes_of(&col));
                }
                cnt as u32
            } else {
                self.scratch.motion_verts.clear();
                0
            }
        };

        // ── DARKEN-CENTER geometry (butterchurn DarkenCenter). Small triangle-fan
        // (expanded to a triangle list): center black @ alpha 3/32, perimeter @ 0.
        let darken_on = live_darken;
        if darken_on {
            let half = 0.05f32;
            let ax = shape_aspecty; // butterchurn applies aspecty to the x extents
                                    // fan verts: [center, p1, p2, p3, p4, p5] with p5 == p1 (closing).
            let center = ([0.0f32, 0.0f32], [0.0f32, 0.0, 0.0, 3.0 / 32.0]);
            let p1 = ([-half * ax, 0.0f32], [0.0f32, 0.0, 0.0, 0.0]);
            let p2 = ([0.0f32, -half], [0.0f32, 0.0, 0.0, 0.0]);
            let p3 = ([half * ax, 0.0f32], [0.0f32, 0.0, 0.0, 0.0]);
            let p4 = ([0.0f32, half], [0.0f32, 0.0, 0.0, 0.0]);
            let p5 = ([-half * ax, 0.0f32], [0.0f32, 0.0, 0.0, 0.0]);
            // TRIANGLE_FAN(6 verts) → 4 triangles, expanded to a triangle list.
            let fan = [center, p1, p2, p3, p4, p5];
            let tris = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)];
            let dv = &mut self.scratch.darken_verts;
            dv.clear();
            for (a, b, c) in tris {
                for k in [a, b, c] {
                    let (pos, color) = fan[k];
                    dv.push(DarkenVert { pos, color });
                }
            }
            self.queue
                .write_buffer(&self.darken_vert_buf, 0, bytemuck::cast_slice(dv));
        } else {
            self.scratch.darken_verts.clear();
        }

        // ── FRAME-BORDER geometry (butterchurn Border.generateBorder). Outer ring
        // (prevBorderSize 0) + inner ring (prevBorderSize = ob_size). NDC, no aspect.
        let ob_size = live_ob_size;
        let ob_a = live_ob_a;
        let ib_size = live_ib_size;
        let ib_a = live_ib_a;
        let outer_color = live_outer_color;
        let inner_color = live_inner_color;
        // Append generate_border(border_size, prev_border_size)'s 24 NDC verts
        // directly into persistent storage. Returns whether a draw was emitted.
        let append_border = |border_size: f32,
                             prev_border_size: f32,
                             alpha: f32,
                             v: &mut Vec<BorderVert>|
         -> bool {
            if !(border_size > 0.0 && alpha > 0.0) {
                return false;
            }
            let width = 2.0f32;
            let height = 2.0f32;
            let wh = width / 2.0;
            let hh = height / 2.0;
            let pbw = prev_border_size / 2.0;
            let bw = border_size / 2.0 + pbw;
            let pbww = pbw * width;
            let pbwh = pbw * height;
            let bww = bw * width;
            let bwh = bw * height;
            let mut tri = |p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]| {
                v.push(BorderVert { pos: p1 });
                v.push(BorderVert { pos: p2 });
                v.push(BorderVert { pos: p3 });
            };
            // 1st side (left)
            let a1 = [-wh + pbww, -hh + bwh];
            let a2 = [-wh + pbww, hh - bwh];
            let a3 = [-wh + bww, hh - bwh];
            let a4 = [-wh + bww, -hh + bwh];
            tri(a4, a2, a1);
            tri(a4, a3, a2);
            // 2nd side (right)
            let b1 = [wh - pbww, -hh + bwh];
            let b2 = [wh - pbww, hh - bwh];
            let b3 = [wh - bww, hh - bwh];
            let b4 = [wh - bww, -hh + bwh];
            tri(b1, b2, b4);
            tri(b2, b3, b4);
            // Top
            let c1 = [-wh + pbww, -hh + pbwh];
            let c2 = [-wh + pbww, bwh - hh];
            let c3 = [wh - pbww, bwh - hh];
            let c4 = [wh - pbww, -hh + pbwh];
            tri(c4, c2, c1);
            tri(c4, c3, c2);
            // Bottom
            let d1 = [-wh + pbww, hh - pbwh];
            let d2 = [-wh + pbww, hh - bwh];
            let d3 = [wh - pbww, hh - bwh];
            let d4 = [wh - pbww, hh - pbwh];
            tri(d1, d2, d4);
            tri(d2, d3, d4);
            true
        };
        {
            let all = &mut self.scratch.frame_border_verts;
            let draws = &mut self.scratch.frame_border_draws;
            all.clear();
            draws.clear();
            let outer_start = all.len() as u32;
            if append_border(ob_size, 0.0, ob_a, all) {
                draws.push((outer_start, 0));
            }
            let inner_start = all.len() as u32;
            if append_border(ib_size, ob_size, ib_a, all) {
                draws.push((inner_start, 1));
            }
            if !all.is_empty() {
                self.queue
                    .write_buffer(&self.frame_border_vert_buf, 0, bytemuck::cast_slice(all));
                // slot 0 = outer color, slot 1 = inner color (dyn-offset 256B each)
                let fb_bytes = &mut self.scratch.frame_border_uniform_bytes;
                fb_bytes.clear();
                fb_bytes.resize(2 * 256, 0);
                let ou = BorderU {
                    color: outer_color,
                    offset: [0.0; 4],
                };
                let iu = BorderU {
                    color: inner_color,
                    offset: [0.0; 4],
                };
                fb_bytes[0..std::mem::size_of::<BorderU>()]
                    .copy_from_slice(bytemuck::bytes_of(&ou));
                fb_bytes[256..256 + std::mem::size_of::<BorderU>()]
                    .copy_from_slice(bytemuck::bytes_of(&iu));
                self.queue
                    .write_buffer(&self.frame_border_uniform_buf, 0, fb_bytes);
            } else {
                self.scratch.frame_border_uniform_bytes.clear();
            }
        }
        let border_draws_frame = &self.scratch.frame_border_draws;

        // Ping-pong: write_to_a determines current target. Keep the prior page
        // view explicit because BeatDrop provenance writes motion vectors and
        // derives blur from it before the warp samples it.
        let (write_texture, write_view, read_view, read_bg, comp_bg) =
            match (self.write_to_a, live_wrap) {
            (true, true) => (
                &self.tex_a,
                &self.view_a,
                &self.view_b,
                &self.bg_read_b,
                &self.bg_read_a,
            ),
            (false, true) => (
                &self.tex_b,
                &self.view_b,
                &self.view_a,
                &self.bg_read_a,
                &self.bg_read_b,
            ),
            (true, false) => (
                &self.tex_a,
                &self.view_a,
                &self.view_b,
                &self.bg_read_b_clamp,
                &self.bg_read_a_clamp,
            ),
            (false, false) => (
                &self.tex_b,
                &self.view_b,
                &self.view_a,
                &self.bg_read_a_clamp,
                &self.bg_read_b_clamp,
            ),
        };

        let mut enc = self.device.create_command_encoder(&Default::default());
        if let Some((query_set, boundary_marker, start_index, _)) = timestamp_writes {
            enc.write_timestamp(query_set, start_index);
            enc.clear_buffer(boundary_marker, 0, None);
        }

        let beatdrop_feedback = self.feedback_provenance == FeedbackProvenance::Beatdrop;
        if beatdrop_feedback {
            // BeatDrop's feedback provenance: inject motion vectors into the
            // outgoing page first, then derive this frame's blur from that page.
            // This is opt-in; the default follows the legacy order below.
            self.encode_motion_vectors(
                &mut enc,
                read_view,
                "feedback-motion-vectors-before-warp",
                mv_count,
            );
            // The previous page's sampling view covers every feedback mip. The
            // vector pass just changed level 0, so rebuild its chain before the
            // blur (and later warp) sample it. Without this, derivative-selected
            // samples could observe stale pre-vector history at nonzero LOD.
            if mv_count > 0 {
                let previous_feedback_mips = if self.write_to_a {
                    &self.feedback_mips_b
                } else {
                    &self.feedback_mips_a
                };
                generate_mip_chain(
                    &self.device,
                    &self.feedback_mip_blitter,
                    &mut enc,
                    previous_feedback_mips,
                );
            }
        }
        let mut blur_pass_count = if beatdrop_feedback {
            self.encode_blur_chain(&mut enc, false)
        } else {
            0
        };

        // --- WARP pass. Legacy blur observes this surface before overlays. ---
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("feedback-warp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: write_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.has_custom_warp {
                // Custom warp FS, driven by the warped mesh VS.
                rp.set_pipeline(&self.warp_custom_pipeline);
                rp.set_bind_group(0, read_bg, &[]); // sampler set (prev frame)
                rp.set_bind_group(1, &self.perframe_bg, &[]);
                rp.set_bind_group(2, &self.warp_params_bg, &[]);
            } else {
                // Default warp mesh: sample prev at warped UV, multiply per-vertex decay.
                let mesh_bg = match (self.write_to_a, live_wrap) {
                    (true, true) => &self.warp_mesh_bg_b,
                    (false, true) => &self.warp_mesh_bg_a,
                    (true, false) => &self.warp_mesh_bg_b_clamp,
                    (false, false) => &self.warp_mesh_bg_a_clamp,
                };
                rp.set_pipeline(&self.warp_mesh_pipeline);
                rp.set_bind_group(0, mesh_bg, &[]);
                rp.set_bind_group(1, &self.warp_params_bg, &[]);
            }
            rp.set_vertex_buffer(0, self.warp_vert_buf.slice(..));
            rp.set_index_buffer(self.warp_idx_buf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.warp_idx_count, 0, 0..1);
        }

        if let Some(readback) = self.geometry_stage_readback.as_ref() {
            encode_stage_texture_copy(
                &mut enc,
                write_texture,
                &readback.post_warp,
                self.render_w,
                self.render_h,
                readback.padded_bytes_per_row,
            );
        }

        // Legacy provenance derives blur from the freshly warped page before
        // authored overlays. BeatDrop provenance already built blur from the
        // previous page before warp, above.
        if !beatdrop_feedback {
            blur_pass_count = self.encode_blur_chain(&mut enc, true);
        }
        self.last_blur_pass_count = blur_pass_count;
        // Overlay pass loads the warp result and preserves MilkDrop's authored
        // draw order. It is intentionally separate so overlays cannot contaminate
        // GetBlur1/2/3 for the same frame.
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("feedback-overlays"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: write_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Motion vectors precede authored shapes/waves in the legacy path.
            // BeatDrop provenance emitted them into the previous feedback page
            // before warp, so it must not draw a second copy here.
            if mv_count > 0 && !beatdrop_feedback {
                rp.set_pipeline(&self.mv_pipeline);
                rp.set_bind_group(0, &self.mv_bg, &[]);
                rp.set_vertex_buffer(0, self.mv_vert_buf.slice(..));
                rp.draw(0..mv_count, 0..1);
            }

            // Textured shapes read the previous feedback side.
            let shape_read_bg = match (self.write_to_a, live_wrap) {
                (true, true) => &self.shape_bg_read_b,
                (false, true) => &self.shape_bg_read_a,
                (true, false) => &self.shape_bg_read_b_clamp,
                (false, false) => &self.shape_bg_read_a_clamp,
            };

            // Butterchurn composites each shape instance as fill then border;
            // batching all fills before all borders changes overlap blending.
            if !fill_draws.is_empty() {
                rp.set_index_buffer(self.shape_idx_buf.slice(..), wgpu::IndexFormat::Uint32);
                let mut fill_state_dirty = true;
                let mut last_additive = None;
                for d in &fill_draws {
                    if d.base_vertex as u32 + d.sides + 2 > SHAPE_VERT_CAP as u32 {
                        continue;
                    }
                    if fill_state_dirty {
                        rp.set_vertex_buffer(0, self.shape_vert_buf.slice(..));
                        rp.set_bind_group(0, shape_read_bg, &[]);
                        last_additive = None;
                        fill_state_dirty = false;
                    }
                    if last_additive != Some(d.additive) {
                        let pipe = if d.additive {
                            &self.shapes_fill_pipeline_additive
                        } else {
                            &self.shapes_fill_pipeline_alpha
                        };
                        rp.set_pipeline(pipe);
                        last_additive = Some(d.additive);
                    }
                    rp.draw_indexed(0..(d.sides * 3), d.base_vertex, 0..1);

                    if let Some(draw_idx) = d.border_draw_index {
                        let border = &border_draws[draw_idx];
                        if border.start_vert >= BORDER_VERT_CAP as u32 {
                            continue;
                        }
                        let base_slot = draw_idx * BORDER_THICK_LINE_PASSES;
                        if base_slot >= BORDER_UNIFORM_SLOTS {
                            continue;
                        }
                        let end = (border.start_vert + border.count).min(BORDER_VERT_CAP as u32);
                        let passes = if border.thick {
                            BORDER_THICK_LINE_PASSES
                        } else {
                            1
                        }
                        .min(BORDER_UNIFORM_SLOTS - base_slot);
                        rp.set_pipeline(&self.shapes_border_pipeline);
                        rp.set_vertex_buffer(0, self.border_vert_buf.slice(..));
                        for k in 0..passes {
                            let offset = ((base_slot + k) * 256) as u32;
                            rp.set_bind_group(0, &self.border_bg, &[offset]);
                            rp.draw(border.start_vert..end, 0..1);
                        }
                        // The border uses a different pipeline, vertex buffer, and
                        // bind-group layout. Restore fill state lazily only when a
                        // later authored instance actually needs it.
                        fill_state_dirty = true;
                    }
                }
            }

            // Built-in and custom waveforms.
            if !wave_draws.is_empty() {
                rp.set_vertex_buffer(0, self.wave_vert_buf.slice(..));
                for d in &wave_draws {
                    let pipe = match (d.points, d.additive) {
                        (true, true) => &self.wave_pipeline_points_additive,
                        (true, false) => &self.wave_pipeline_points_alpha,
                        (false, true) => &self.wave_pipeline_lines_additive,
                        (false, false) => &self.wave_pipeline_lines_alpha,
                    };
                    rp.set_pipeline(pipe);
                    if d.start_vert >= WAVE_VERT_CAP as u32 {
                        continue;
                    }
                    let end = (d.start_vert + d.count).min(WAVE_VERT_CAP as u32);
                    let passes = if d.thick {
                        if d.points {
                            WAVE_THICK_DOT_PASSES
                        } else {
                            WAVE_THICK_LINE_PASSES
                        }
                    } else {
                        1
                    };
                    rp.set_bind_group(0, &self.wave_bg, &[]);
                    rp.draw(d.start_vert..end, 0..passes as u32);
                }
            }

            // The incoming state evaluated and uploaded weighted copies before
            // this pass opened. Draw its compatible untextured shapes/waves
            // after the outgoing state, preserving each renderer's internal
            // authored order while making both families visible before promotion.
            if let Some((target, _)) = shared_overlays {
                let target_fill_draws = &target.scratch.shape_fill_draws;
                if !target_fill_draws.is_empty() {
                    rp.set_index_buffer(target.shape_idx_buf.slice(..), wgpu::IndexFormat::Uint32);
                    rp.set_vertex_buffer(0, target.shape_vert_buf.slice(..));
                    // Compatibility rejects textured shapes, so the shader never
                    // samples this bind group; it is still required by WGSL.
                    rp.set_bind_group(0, &target.shape_bg_read_a, &[]);
                    let mut last_additive = None;
                    for draw in target_fill_draws {
                        if draw.base_vertex as u32 + draw.sides + 2 > SHAPE_VERT_CAP as u32 {
                            continue;
                        }
                        if last_additive != Some(draw.additive) {
                            rp.set_pipeline(if draw.additive {
                                &target.shapes_fill_pipeline_additive
                            } else {
                                &target.shapes_fill_pipeline_alpha
                            });
                            last_additive = Some(draw.additive);
                        }
                        rp.draw_indexed(0..(draw.sides * 3), draw.base_vertex, 0..1);
                    }
                }

                let target_wave_draws = &target.scratch.wave_draws;
                if !target_wave_draws.is_empty() {
                    rp.set_vertex_buffer(0, target.wave_vert_buf.slice(..));
                    rp.set_bind_group(0, &target.wave_bg, &[]);
                    for draw in target_wave_draws {
                        let pipe = match (draw.points, draw.additive) {
                            (true, true) => &target.wave_pipeline_points_additive,
                            (true, false) => &target.wave_pipeline_points_alpha,
                            (false, true) => &target.wave_pipeline_lines_additive,
                            (false, false) => &target.wave_pipeline_lines_alpha,
                        };
                        rp.set_pipeline(pipe);
                        if draw.start_vert >= WAVE_VERT_CAP as u32 {
                            continue;
                        }
                        let end = (draw.start_vert + draw.count).min(WAVE_VERT_CAP as u32);
                        let passes = if draw.thick {
                            if draw.points {
                                WAVE_THICK_DOT_PASSES
                            } else {
                                WAVE_THICK_LINE_PASSES
                            }
                        } else {
                            1
                        };
                        rp.draw(draw.start_vert..end, 0..passes as u32);
                    }
                }
            }

            // Darken-center and frame borders follow waves.
            if darken_on {
                rp.set_pipeline(&self.darken_pipeline);
                rp.set_vertex_buffer(0, self.darken_vert_buf.slice(..));
                rp.draw(0..12, 0..1);
            }
            if !border_draws_frame.is_empty() {
                rp.set_pipeline(&self.frame_border_pipeline);
                rp.set_vertex_buffer(0, self.frame_border_vert_buf.slice(..));
                for &(start_vert, slot) in border_draws_frame {
                    rp.set_bind_group(0, &self.frame_border_bg, &[(slot * 256) as u32]);
                    rp.draw(start_vert..(start_vert + 24), 0..1);
                }
            }
        }

        if let Some(readback) = self.geometry_stage_readback.as_ref() {
            encode_stage_texture_copy(
                &mut enc,
                write_texture,
                &readback.post_overlays,
                self.render_w,
                self.render_h,
                readback.padded_bytes_per_row,
            );
        }

        let feedback_mips = if self.write_to_a {
            &self.feedback_mips_a
        } else {
            &self.feedback_mips_b
        };
        generate_mip_chain(
            &self.device,
            &self.feedback_mip_blitter,
            &mut enc,
            feedback_mips,
        );

        // --- COMP pass: read from curr, write to the comp target ---
        // With FXAA enabled we render into the offscreen comp intermediate so the
        // FXAA output pass can read it; with FXAA disabled we skip that round-trip
        // and write the swapchain directly.
        let comp_target: &wgpu::TextureView = if self.fxaa_enabled || surface_view.is_none() {
            &self.comp_view
        } else {
            surface_view.expect("direct output requires a surface")
        };
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("comp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: comp_target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(if self.fxaa_enabled || surface_view.is_none() {
                &self.comp_pipeline
            } else {
                &self.comp_direct_pipeline
            });
            rp.set_bind_group(0, comp_bg, &[]);
            rp.set_bind_group(1, &self.comp_perframe_bg, &[]);
            rp.set_vertex_buffer(0, self.comp_vert_buf.slice(..));
            rp.set_index_buffer(self.comp_idx_buf.slice(..), wgpu::IndexFormat::Uint16);
            rp.draw_indexed(0..self.comp_idx_count, 0, 0..1);
        }

        if let Some(readback) = self.geometry_stage_readback.as_ref() {
            encode_stage_texture_copy(
                &mut enc,
                &self.comp_tex,
                &readback.post_comp,
                self.render_w,
                self.render_h,
                readback.padded_bytes_per_row,
            );
        }

        // --- OUTPUT pass: FXAA the offscreen comp result → swapchain ---
        // Fullscreen triangle covers 100% → LoadOp::Clear (no needless read).
        // Skipped entirely when FXAA is disabled: COMP already wrote the
        // swapchain directly above, so there is nothing to resolve.
        if self.fxaa_enabled && surface_view.is_some() {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fxaa-output"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view.expect("FXAA output requires a surface"),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.output_pipeline);
            rp.set_bind_group(0, &self.fxaa_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        if let Some((query_set, boundary_marker, _, end_index)) = timestamp_writes {
            enc.clear_buffer(boundary_marker, 0, None);
            enc.write_timestamp(query_set, end_index);
        }
        self.queue.submit(std::iter::once(enc.finish()));

        // Return packed geometry storage to the renderer after every CPU/GPU
        // consumer is finished. The next frame clears and refills these vectors,
        // retaining capacities reached by large shape/wave presets instead of
        // repeatedly allocating and growing the same buffers.
        self.scratch.shape_fill_verts = fill_verts;
        self.scratch.shape_fill_draws = fill_draws;
        self.scratch.shape_border_verts = border_verts;
        self.scratch.shape_border_draws = border_draws;
        self.scratch.wave_verts = wave_verts;
        self.scratch.wave_draws = wave_draws;
        self.write_to_a = !self.write_to_a;
        self.frame_idx += 1;
    }
}

pub(crate) fn resample_linear(src: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(target_len);
    resample_linear_into(src, target_len, &mut out, false);
    out
}

/// Resample into caller-owned storage. The live renderer reuses this storage
/// frame-to-frame, avoiding a short-lived audio allocation for every preview and
/// program frame. `clamp_waveform` preserves `set_waveform`'s public `[-1, 1]`
/// and finite-value contract.
fn resample_linear_into(src: &[f32], target_len: usize, out: &mut Vec<f32>, clamp_waveform: bool) {
    out.clear();
    if src.is_empty() || target_len == 0 {
        return;
    }
    out.reserve(target_len.saturating_sub(out.capacity()));
    let sample = |index: usize| {
        let value = src[index];
        if clamp_waveform {
            finite_clamp(value, -1.0, 1.0, 0.0)
        } else {
            value
        }
    };
    if src.len() == 1 {
        out.resize(target_len, sample(0));
        return;
    }
    let last = (src.len() - 1) as f32;
    let denom = (target_len - 1).max(1) as f32;
    for i in 0..target_len {
        let pos = last * (i as f32) / denom;
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(src.len() - 1);
        let frac = pos - i0 as f32;
        out.push(sample(i0) * (1.0 - frac) + sample(i1) * frac);
    }
}

/// Copy or resample an audio row directly into a renderer-owned reusable buffer.
fn replace_audio_samples(
    destination: &mut Vec<f32>,
    source: &[f32],
    target_len: usize,
    clamp_waveform: bool,
) {
    resample_linear_into(source, target_len, destination, clamp_waveform);
}

fn custom_wave_sources(
    spectrum: bool,
    time_l: &[f32],
    time_r: &[f32],
    freq: &[f32],
    target_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    if spectrum && !freq.is_empty() {
        let f = resample_linear(freq, target_len);
        (f.clone(), f)
    } else {
        (
            resample_linear(time_l, target_len),
            resample_linear(time_r, target_len),
        )
    }
}

// WaveUtils.smoothWave — positions only (used by BasicWaveform). Catmull-Rom-ish.
// `pts` is a flat list of (x,y); returns interleaved smoothed list of (n*2-1).
fn smooth_wave(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut out = Vec::new();
    smooth_wave_into(pts, &mut out);
    out
}

fn smooth_wave_into(pts: &[[f32; 2]], out: &mut Vec<[f32; 2]>) {
    let n = pts.len();
    out.clear();
    if n < 2 {
        out.extend_from_slice(pts);
        return;
    }
    let c1 = -0.15f32;
    let c2 = 1.15f32;
    let c3 = 1.15f32;
    let c4 = -0.15f32;
    let inv_sum = 1.0 / (c1 + c2 + c3 + c4); // = 0.5
    out.resize(n * 2 - 1, [0.0f32; 2]);
    let mut j = 0usize;
    let mut i_below = 0usize;
    let mut i_above2 = 1usize;
    for i in 0..n - 1 {
        let i_above = i_above2;
        i_above2 = (i + 2).min(n - 1);
        out[j] = pts[i];
        out[j + 1][0] =
            (c1 * pts[i_below][0] + c2 * pts[i][0] + c3 * pts[i_above][0] + c4 * pts[i_above2][0])
                * inv_sum;
        out[j + 1][1] =
            (c1 * pts[i_below][1] + c2 * pts[i][1] + c3 * pts[i_above][1] + c4 * pts[i_above2][1])
                * inv_sum;
        i_below = i;
        j += 2;
    }
    out[j] = pts[n - 1];
}

// WaveUtils.smoothWaveAndColor — positions + held color. Returns (positions, colors).
fn smooth_wave_and_color(pts: &[[f32; 2]], cols: &[[f32; 4]]) -> (Vec<[f32; 2]>, Vec<[f32; 4]>) {
    let n = pts.len();
    if n < 2 {
        return (pts.to_vec(), cols.to_vec());
    }
    let c1 = -0.15f32;
    let c2 = 1.15f32;
    let c3 = 1.15f32;
    let c4 = -0.15f32;
    let inv_sum = 1.0 / (c1 + c2 + c3 + c4);
    let mut out_p = vec![[0.0f32; 2]; n * 2 - 1];
    let mut out_c = vec![[0.0f32; 4]; n * 2 - 1];
    let mut j = 0usize;
    let mut i_below = 0usize;
    let mut i_above2 = 1usize;
    for i in 0..n - 1 {
        let i_above = i_above2;
        i_above2 = (i + 2).min(n - 1);
        out_p[j] = pts[i];
        out_p[j + 1][0] =
            (c1 * pts[i_below][0] + c2 * pts[i][0] + c3 * pts[i_above][0] + c4 * pts[i_above2][0])
                * inv_sum;
        out_p[j + 1][1] =
            (c1 * pts[i_below][1] + c2 * pts[i][1] + c3 * pts[i_above][1] + c4 * pts[i_above2][1])
                * inv_sum;
        out_c[j] = cols[i];
        out_c[j + 1] = cols[i];
        i_below = i;
        j += 2;
    }
    out_p[j] = pts[n - 1];
    out_c[j] = cols[n - 1];
    (out_p, out_c)
}

/// WaveUtils smoothing fused directly into the final staging vertex buffer. This
/// avoids allocating two `2*n-1` temporary vectors for every custom wave/frame.
fn emit_smoothed_wave_and_color(
    points: &[[f32; 2]],
    colors: &[[f32; 4]],
    out: &mut Vec<WaveVert>,
) -> u32 {
    let n = points.len().min(colors.len());
    if n == 0 {
        return 0;
    }
    if n == 1 {
        out.push(WaveVert {
            pos: points[0],
            color: colors[0],
        });
        return 1;
    }
    let c1 = -0.15f32;
    let c2 = 1.15f32;
    let c3 = 1.15f32;
    let c4 = -0.15f32;
    let inv_sum = 1.0 / (c1 + c2 + c3 + c4);
    let mut below = 0usize;
    let mut above2 = 1usize;
    for i in 0..n - 1 {
        let above = above2;
        above2 = (i + 2).min(n - 1);
        out.push(WaveVert {
            pos: points[i],
            color: colors[i],
        });
        out.push(WaveVert {
            pos: [
                (c1 * points[below][0]
                    + c2 * points[i][0]
                    + c3 * points[above][0]
                    + c4 * points[above2][0])
                    * inv_sum,
                (c1 * points[below][1]
                    + c2 * points[i][1]
                    + c3 * points[above][1]
                    + c4 * points[above2][1])
                    * inv_sum,
            ],
            color: colors[i],
        });
        below = i;
    }
    out.push(WaveVert {
        pos: points[n - 1],
        color: colors[n - 1],
    });
    (n * 2 - 1) as u32
}

#[cfg(test)]
mod tests {
    use crate::enhanced_audio::{EnhancedAudioConfig, ENHANCED_FFT_BINS};
    use super::{
        audio_att_floor, audio_level_floor, blend_evaluated_warp_mesh, blur_dimensions, blur_min_max_remap,
        blur_scale_and_bias, build_comp_indices, compile_milkdrop_shader_bodies_from_parts,
        deterministic_time_seconds, downsample_rgba_volume, effective_fps,
        emit_smoothed_wave_and_color, finite_f32_from_f64, generate_comp_verts,
        guard_finite_blur_scale_bias, guard_finite_comp_blur, milkdrop_angle, needed_blur_levels,
        resample_linear, resample_linear_into, rgba8_rgb_summary, seed_equation_inputs,
        shader_frame_index, shader_progress, shader_time_seconds, smooth_wave_and_color,
        summarize_custom_geometry, validate_texture_dims, BorderDraw, BorderVert, ButterchurnRng,
        DimensionError, GeometryDiagnosticCollector, MilkdropGeometryDiagnostics,
        MilkdropResizeDebouncer, ShapeFillDraw, ShapeVert, WaveDraw, WaveVert, COMP_GRID_H,
        COMP_GRID_W, GPU_FRAME_WRAP, GPU_TIME_WRAP_SECONDS, GRID_H, GRID_W,
        INTERACTIVE_RESIZE_DEBOUNCE, WarpVert,
    };
    #[cfg(feature = "app")]
    use super::{MilkdropPerformanceProfile, MilkdropRenderer};
    use std::cell::Cell;
    use std::time::{Duration, Instant};


    /// The hazard is real and upstream-faithful: pins the zero-width collapse itself, so
    /// a later "parity restoration" of `blur_min_max_remap`'s sign is caught here rather
    /// than by silently making the guard below dead code.
    #[test]
    fn blur_remap_collapses_a_narrow_range_to_exactly_zero_width() {
        let (bmin, bmax) = blur_min_max_remap([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        // Both endpoints get `avg - fmin_dist * 0.5` — the upstream minus-on-max slip.
        assert_eq!(
            bmax[0] - bmin[0],
            0.0,
            "level 1 range must collapse to zero"
        );
        assert_eq!(
            bmax[1] - bmin[1],
            0.0,
            "level 2 range must collapse to zero"
        );
        assert_eq!(
            bmax[2] - bmin[2],
            0.0,
            "level 3 range must collapse to zero"
        );
        assert!((bmin[0] - -0.05).abs() < 1e-6, "bmin[0] = {}", bmin[0]);
        assert!((bmin[1] - -0.075).abs() < 1e-6, "bmin[1] = {}", bmin[1]);
        assert!((bmin[2] - -0.0875).abs() < 1e-6, "bmin[2] = {}", bmin[2]);
    }

    /// The unguarded transform really does emit the recorded Inf/NaN. Without this the
    /// guard test could pass vacuously on inputs that were never dangerous.
    #[test]
    fn blur_scale_and_bias_is_non_finite_on_the_recorded_evidence_pattern() {
        // NaN case — all six blur bounds zero (13 of the 14 affected presets).
        let (bmin, bmax) = blur_min_max_remap([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let (scale, bias) = blur_scale_and_bias(bmin, bmax);
        assert!(scale[0].is_infinite(), "scale[0] = {}", scale[0]);
        assert!(bias[0].is_infinite(), "bias[0] = {}", bias[0]);
        assert!(scale[1].is_nan(), "scale[1] = {}", scale[1]);
        assert!(bias[1].is_nan(), "bias[1] = {}", bias[1]);
        assert!(scale[2].is_nan(), "scale[2] = {}", scale[2]);
        assert!(bias[2].is_nan(), "bias[2] = {}", bias[2]);

        // Inf-only case — level 1 is a healthy [0,1], only level 2 is narrower than
        // fmin_dist. Covers the other residual form the addendum names.
        let (bmin, bmax) = blur_min_max_remap([0.0, 0.5, 0.0], [1.0, 0.5, 1.0]);
        let (scale, bias) = blur_scale_and_bias(bmin, bmax);
        assert!(scale[0].is_finite(), "scale[0] = {}", scale[0]);
        assert!(scale[1].is_infinite(), "scale[1] = {}", scale[1]);
        assert!(bias[1].is_infinite(), "bias[1] = {}", bias[1]);
    }


    /// Mechanism (a): the pre-cast finiteness trap. `1e300_f64` IS finite, and
    /// `1e300_f64 as f32` IS `+inf` — Rust's float cast saturates. The old `eqd`
    /// tested the `f64`, so this value passed the guard and poisoned the uniforms.
    #[test]
    fn eqd_narrowing_rejects_an_f64_that_is_finite_but_overflows_f32() {
        // The exact value from the reviewer's reproduction.
        assert!(1e300_f64.is_finite(), "premise: the f64 is finite");
        assert!(
            !(1e300_f64 as f32).is_finite(),
            "premise: the cast saturates to inf"
        );
        assert_eq!(finite_f32_from_f64(1e300, 0.25), 0.25);
        assert_eq!(finite_f32_from_f64(-1e300, 0.25), 0.25);
        // Genuine non-finites still fall back.
        assert_eq!(finite_f32_from_f64(f64::NAN, 0.5), 0.5);
        assert_eq!(finite_f32_from_f64(f64::INFINITY, 0.5), 0.5);
        // In-range values are untouched, including f32-subnormal magnitudes.
        assert_eq!(finite_f32_from_f64(0.75, 0.0), 0.75);
        // -3.0e38 is inside f32's range; -3.5e38 is NOT (f32::MAX ~ 3.4028e38)
        // and correctly falls back, which is the same trap this test is about.
        assert_eq!(finite_f32_from_f64(-3.0e38, 0.0), -3.0e38_f64 as f32);
        assert_eq!(finite_f32_from_f64(-3.5e38, 0.25), 0.25);
        assert!(finite_f32_from_f64(1e-45, 0.0).is_finite());
        // A non-finite DEFAULT cannot launder one through either.
        assert_eq!(finite_f32_from_f64(f64::NAN, 1e300), 0.0);
        assert_eq!(finite_f32_from_f64(f64::NAN, f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn the_eq_closure_narrows_exactly_like_eqd_so_q_vars_cannot_carry_inf() {
        // `eq` is a closure over `self.eel_env`, so the shared narrowing helper is
        // what is testable here — and it is the whole of `eq`'s body.
        assert_eq!(
            finite_f32_from_f64(1e300, 0.0),
            0.0,
            "the reviewer's q1 probe"
        );
        assert_eq!(finite_f32_from_f64(-1e300, 0.0), 0.0);
        assert_eq!(finite_f32_from_f64(f64::INFINITY, 0.0), 0.0);
        assert_eq!(finite_f32_from_f64(f64::NAN, 0.0), 0.0);
        // Ordinary q values are untouched.
        assert_eq!(finite_f32_from_f64(0.5, 0.0), 0.5);
        assert_eq!(finite_f32_from_f64(-42.0, 0.0), -42.0);

        // The two other casts in this file are bounded before narrowing, so they
        // cannot carry this defect. Both are total over every finite input.
        for t in [0.0f64, 1e300, -1e300, 86_400.0, f64::MAX] {
            assert!(
                shader_time_seconds(t).is_finite(),
                "shader_time_seconds({t}) is not finite"
            );
            assert!(
                shader_progress(t).is_finite(),
                "shader_progress({t}) is not finite"
            );
        }
    }

    #[test]
    fn public_audio_setters_cannot_admit_non_finite_or_unsafe_attenuation() {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let v = audio_level_floor(poison);
            assert!(v.is_finite() && v >= 0.0, "level {v} from {poison}");
            let a = audio_att_floor(poison);
            // `*_att` divides in preset equations, so zero is as bad as NaN.
            assert!(a.is_finite() && a > 0.0, "att {a} from {poison}");
        }
        // Negative and zero attenuation are unsafe for the same reason.
        assert_eq!(audio_level_floor(-5.0), 0.0);
        assert_eq!(audio_level_floor(-0.0), 0.0);
        assert_eq!(audio_att_floor(0.0), 1.0);
        assert_eq!(audio_att_floor(-1.0), 1.0);

        // Non-vacuity: ordinary values pass through bit-for-bit, so this is a floor
        // and not a clamp that would change in-repo behaviour. Every in-repo caller
        // arrives via `milkdrop_audio_rails`, whose output is finite and >= 0.001.
        for v in [1.4f32, 1.1, 0.9, 1.2, 0.001, 3.0e38] {
            assert_eq!(audio_level_floor(v), v);
            assert_eq!(audio_att_floor(v), v);
        }
    }

    /// Mechanism (b) + the consequence. With `bmin[0] = +inf` the range becomes
    /// `NaN`, the blur-UBO guard catches its side, and the comp side — which was
    /// unguarded — shipped `scale1 = NaN` into `comp_perframe_buf`.
    #[test]
    fn a_non_finite_blur_endpoint_no_longer_reaches_the_comp_uniforms() {
        // What `eqd("b1n", …)` used to hand downstream for a per-frame b1n=1e300.
        let poisoned = f32::INFINITY;
        let (bmin, bmax) = blur_min_max_remap([poisoned, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Premise: the narrow-range branch does NOT fire, because `NaN < 0.1` is
        // false, so the poisoned endpoint survives the remap untouched.
        assert!(bmin[0].is_infinite(), "premise: bmin[0] stays +inf");
        assert!(
            (bmax[0] - bmin[0]).is_nan(),
            "premise: the range is NaN, not zero"
        );

        // The blur UBO was already clean — that half was never the defect.
        let (raw_scale, raw_bias) = blur_scale_and_bias(bmin, bmax);
        let (scale, bias) = guard_finite_blur_scale_bias(raw_scale, raw_bias);
        for lvl in 0..3 {
            assert!(scale[lvl].is_finite() && bias[lvl].is_finite());
        }

        // The comp side is what leaked. Unguarded first, to prove non-vacuity:
        let comp_scale_raw = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
        assert!(
            comp_scale_raw[0].is_nan(),
            "premise: the unguarded comp scale1 is NaN"
        );

        let (cmin, cmax, cscale, cbias) =
            guard_finite_comp_blur(bmin, bmax, comp_scale_raw, [bmin[0], bmin[1], bmin[2]]);
        for lvl in 0..3 {
            assert!(
                cmin[lvl].is_finite()
                    && cmax[lvl].is_finite()
                    && cscale[lvl].is_finite()
                    && cbias[lvl].is_finite(),
                "comp level {lvl} still non-finite: min {} max {} scale {} bias {}",
                cmin[lvl],
                cmax[lvl],
                cscale[lvl],
                cbias[lvl]
            );
        }
        // The poisoned level is reset to the identity range as a group.
        assert_eq!(
            (cmin[0], cmax[0], cscale[0], cbias[0]),
            (0.0, 1.0, 1.0, 0.0)
        );
    }

    /// THE case `guard_finite_comp_blur` exists for, and the one no ingress guard
    /// can close: two **finite** endpoints whose **subtraction** overflows `f32`.
    /// Both values pass `parse_finite_f32` and `finite_f32_from_f64`; the overflow
    /// is a property of the pair, not of either value. Found by review 2026-08-15
    /// when it disproved this guard's original justification.
    #[test]
    fn two_finite_endpoints_whose_difference_overflows_are_still_caught() {
        let (bmin_in, bmax_in) = ([-3.0e38f32, 0.0, 0.0], [3.0e38f32, 1.0, 1.0]);
        // Premise: both endpoints are finite and would survive every ingress guard.
        assert!(bmin_in[0].is_finite() && bmax_in[0].is_finite());
        assert_eq!(finite_f32_from_f64(-3.0e38, 0.0), bmin_in[0]);
        assert_eq!(finite_f32_from_f64(3.0e38, 0.0), bmax_in[0]);

        let (bmin, bmax) = blur_min_max_remap(bmin_in, bmax_in);
        // Premise: the narrow-range branch does NOT fire — the range is enormous,
        // not narrow — so the endpoints arrive at the comp side untouched.
        assert_eq!(bmin[0], bmin_in[0]);
        assert_eq!(bmax[0], bmax_in[0]);
        let comp_scale_raw = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
        assert!(
            comp_scale_raw[0].is_infinite(),
            "premise: the subtraction overflows, got {}",
            comp_scale_raw[0]
        );

        let (cmin, cmax, cscale, cbias) =
            guard_finite_comp_blur(bmin, bmax, comp_scale_raw, [bmin[0], bmin[1], bmin[2]]);
        assert_eq!(
            (cmin[0], cmax[0], cscale[0], cbias[0]),
            (0.0, 1.0, 1.0, 0.0),
            "only this guard resets the overflowing level"
        );
        for lvl in 0..3 {
            assert!(cscale[lvl].is_finite() && cbias[lvl].is_finite());
        }
    }

    /// Non-vacuity for the comp guard: an ordinary preset is bit-identical to
    /// pre-change, and a level is only reset when one of its four is non-finite.
    #[test]
    fn comp_blur_guard_leaves_finite_levels_exactly_as_authored() {
        let (bmin, bmax) = blur_min_max_remap([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let scale = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]];
        let bias = [bmin[0], bmin[1], bmin[2]];
        let (cmin, cmax, cscale, cbias) = guard_finite_comp_blur(bmin, bmax, scale, bias);
        assert_eq!((cmin, cmax, cscale, cbias), (bmin, bmax, scale, bias));

        // Only the offending level is touched; the other two survive verbatim.
        let (cmin, cmax, cscale, cbias) = guard_finite_comp_blur(
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            [0.8, f32::NAN, 0.4],
            [0.1, 0.2, 0.3],
        );
        assert_eq!(
            (cmin[0], cmax[0], cscale[0], cbias[0]),
            (0.1, 0.9, 0.8, 0.1)
        );
        assert_eq!(
            (cmin[1], cmax[1], cscale[1], cbias[1]),
            (0.0, 1.0, 1.0, 0.0)
        );
        assert_eq!(
            (cmin[2], cmax[2], cscale[2], cbias[2]),
            (0.3, 0.7, 0.4, 0.3)
        );
    }

    #[test]
    fn blur_guard_keeps_every_value_written_to_the_blur_ubo_finite() {
        for (bmin_in, bmax_in, label) in [
            (
                [0.0f32, 0.0, 0.0],
                [0.0f32, 0.0, 0.0],
                "all-zero (NaN form)",
            ),
            (
                [0.0f32, 0.5, 0.0],
                [1.0f32, 0.5, 1.0],
                "narrow level 2 (Inf form)",
            ),
            (
                [0.2f32, 0.2, 0.2],
                [0.25f32, 0.9, 0.9],
                "narrow level 1 only",
            ),
            (
                [-1.0f32, -1.0, -1.0],
                [-1.0f32, -1.0, -1.0],
                "negative all-equal",
            ),
        ] {
            let (bmin, bmax) = blur_min_max_remap(bmin_in, bmax_in);
            let (raw_scale, raw_bias) = blur_scale_and_bias(bmin, bmax);
            let (scale, bias) = guard_finite_blur_scale_bias(raw_scale, raw_bias);
            for lvl in 0..3 {
                assert!(
                    scale[lvl].is_finite(),
                    "{label}: scale[{lvl}] reached the UBO as {} (raw {})",
                    scale[lvl],
                    raw_scale[lvl]
                );
                assert!(
                    bias[lvl].is_finite(),
                    "{label}: bias[{lvl}] reached the UBO as {} (raw {})",
                    bias[lvl],
                    raw_bias[lvl]
                );
                // Pair semantics: a substituted level is the identity transform, so the
                // shader's `b = b * scale + bias` passes the level through unremapped.
                if !raw_scale[lvl].is_finite() || !raw_bias[lvl].is_finite() {
                    assert_eq!(scale[lvl], 1.0, "{label}: level {lvl} scale not identity");
                    assert_eq!(bias[lvl], 0.0, "{label}: level {lvl} bias not identity");
                }
            }
        }
    }

    #[test]
    fn blur_guard_preserves_finite_scale_bias_including_large_magnitudes() {
        // Default blur range: both halves are identity, nothing is substituted.
        let (bmin, bmax) = blur_min_max_remap([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let (raw_scale, raw_bias) = blur_scale_and_bias(bmin, bmax);
        let (scale, bias) = guard_finite_blur_scale_bias(raw_scale, raw_bias);
        assert_eq!(scale, raw_scale);
        assert_eq!(bias, raw_bias);
        assert_eq!(scale, [1.0, 1.0, 1.0]);
        assert_eq!(bias, [0.0, 0.0, 0.0]);

        // Magnitude is NOT policy: a huge-but-finite scale passes through untouched.
        let huge = [1.0e30f32, -1.0e30, 1.0e-30];
        let (scale, bias) = guard_finite_blur_scale_bias(huge, huge);
        assert_eq!(scale, huge);
        assert_eq!(bias, huge);

        // Pair granularity: a finite scale beside a non-finite bias resets BOTH, because
        // keeping the scale and zeroing the bias is an arbitrary third transform.
        let (scale, bias) =
            guard_finite_blur_scale_bias([4.0, 1.0, 1.0], [f32::NAN, 0.25, f32::INFINITY]);
        assert_eq!(scale, [1.0, 1.0, 1.0]);
        assert_eq!(bias, [0.0, 0.25, 0.0]);
    }

    #[test]
    fn preset_setup_audio_matches_butterchurn_audio_levels() {
        let mut env = crate::equations::Env::new();
        seed_equation_inputs(&mut env, 640, 360);
        assert_eq!(env.get("fps"), Some(&45.0));
        for key in ["bass", "mid", "treb", "vol"] {
            assert_eq!(env.get(key), Some(&0.0), "{key}");
        }
        for key in ["bass_att", "mid_att", "treb_att", "vol_att"] {
            assert_eq!(env.get(key), Some(&1.0), "{key}");
        }
    }

    #[test]
    fn butterchurn_noise_rng_matches_seeded_random_reference() {
        let mut rng = ButterchurnRng::new(12_345);
        let expected = [
            0.588_103_490_881_621_8,
            0.392_734_328_517_690_3,
            0.806_113_125_290_721_7,
            0.139_376_720_180_735,
            0.852_549_671_428_278_1,
            0.934_001_052_752_137_2,
            0.789_350_499_166_175_7,
            0.595_282_274_996_861_8,
        ];
        for value in expected {
            assert!((rng.next_unit() - value).abs() < 1.0e-15);
        }
    }

    #[test]
    fn geometry_diagnostics_are_disabled_and_lazy_by_default() {
        let mut collector = GeometryDiagnosticCollector::default();
        let called = Cell::new(false);
        collector.capture(|| {
            called.set(true);
            MilkdropGeometryDiagnostics::default()
        });
        assert!(
            !called.get(),
            "disabled collector must not run summary work"
        );
        assert_eq!(collector.latest(), None);

        collector.set_enabled(true);
        collector.capture(|| {
            called.set(true);
            MilkdropGeometryDiagnostics {
                frame_index: 42,
                ..MilkdropGeometryDiagnostics::default()
            }
        });
        assert!(called.get());
        assert_eq!(collector.latest().unwrap().frame_index, 42);

        collector.set_enabled(false);
        assert_eq!(
            collector.latest(),
            None,
            "disabling drops retained evidence"
        );
    }

    #[test]
    fn geometry_diagnostics_summarize_counts_bounds_alpha_and_rgb() {
        let fill_verts = [
            ShapeVert {
                pos: [-0.75, 0.25],
                color: [1.0, 0.0, 0.0, 0.2],
                uv: [-1.0, -1.0],
            },
            ShapeVert {
                pos: [0.5, -0.5],
                color: [0.0, 1.0, 0.0, 0.6],
                uv: [-1.0, -1.0],
            },
        ];
        let fill_draws = [ShapeFillDraw {
            base_vertex: 0,
            sides: 3,
            additive: false,
            border_draw_index: Some(0),
        }];
        let border_verts = [BorderVert { pos: [0.5, -0.5] }];
        let border_draws = [BorderDraw {
            start_vert: 0,
            count: 1,
            color: [1.0, 1.0, 1.0, 0.4],
            thick: false,
        }];
        let wave_verts = [
            WaveVert {
                pos: [-1.0, -0.25],
                color: [0.0, 0.0, 1.0, 0.1],
            },
            WaveVert {
                pos: [0.25, 0.75],
                color: [1.0, 1.0, 0.0, 0.9],
            },
        ];
        let wave_draws = [WaveDraw {
            start_vert: 0,
            count: 2,
            points: false,
            additive: false,
            thick: false,
        }];

        let summary = summarize_custom_geometry(
            7,
            2,
            &fill_verts,
            &fill_draws,
            &border_verts,
            &border_draws,
            3,
            &wave_verts,
            &wave_draws,
        );
        assert_eq!(summary.frame_index, 7);
        assert_eq!(summary.custom_shapes.enabled_pools, 2);
        assert_eq!(summary.custom_shapes.fill_draws, 1);
        assert_eq!(summary.custom_shapes.border_draws, 1);
        assert_eq!(summary.custom_shapes.fill_vertices, 2);
        assert_eq!(summary.custom_shapes.border_vertices, 1);
        assert_eq!(summary.custom_shapes.bounds.unwrap().min, [-0.75, -0.5]);
        assert_eq!(summary.custom_shapes.bounds.unwrap().max, [0.5, 0.25]);
        assert!((summary.custom_shapes.fill_alpha.unwrap().mean - 0.4).abs() < 1.0e-6);
        assert_eq!(summary.custom_shapes.border_alpha.unwrap().mean, 0.4);
        let fill_rgb = summary.custom_shapes.fill_rgb.unwrap();
        assert_eq!(fill_rgb.min, [0.0, 0.0, 0.0]);
        assert_eq!(fill_rgb.mean, [0.5, 0.5, 0.0]);
        assert_eq!(fill_rgb.max, [1.0, 1.0, 0.0]);
        assert_eq!(fill_rgb.visible_fraction, 1.0);
        assert!((fill_rgb.mean_abs_energy - (1.0 / 3.0)).abs() < 1.0e-6);
        assert_eq!(
            summary.custom_shapes.border_rgb.unwrap().mean,
            [1.0, 1.0, 1.0]
        );
        assert_eq!(summary.custom_waves.enabled_pools, 3);
        assert_eq!(summary.custom_waves.draws, 1);
        assert_eq!(summary.custom_waves.vertices, 2);
        assert_eq!(summary.custom_waves.bounds.unwrap().min, [-1.0, -0.25]);
        assert_eq!(summary.custom_waves.bounds.unwrap().max, [0.25, 0.75]);
        assert!((summary.custom_waves.alpha.unwrap().mean - 0.5).abs() < 1.0e-6);
        let wave_rgb = summary.custom_waves.rgb.unwrap();
        assert_eq!(wave_rgb.mean, [0.5, 0.5, 0.5]);
        assert_eq!(wave_rgb.visible_fraction, 1.0);
        assert!((wave_rgb.mean_abs_energy - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn stage_rgb_summary_ignores_copy_row_padding() {
        let mut bytes = vec![255u8; 512];
        bytes[0..8].copy_from_slice(&[255, 0, 0, 255, 0, 255, 0, 255]);
        bytes[256..264].copy_from_slice(&[0, 0, 255, 255, 0, 0, 0, 255]);

        let summary = rgba8_rgb_summary(&bytes, 2, 2, 256).unwrap();
        assert_eq!(summary.sample_count, 4);
        assert_eq!(summary.min, [0.0, 0.0, 0.0]);
        assert_eq!(summary.mean, [0.25, 0.25, 0.25]);
        assert_eq!(summary.max, [1.0, 1.0, 1.0]);
        assert_eq!(summary.visible_fraction, 0.75);
        assert_eq!(summary.mean_abs_energy, 0.25);
    }

    #[test]
    fn shared_feedback_blends_evaluated_opposite_translations_without_non_finite_values() {
        let mut outgoing = [
            WarpVert {
                pos: [-1.0, -1.0],
                uv: [0.25, 0.75],
                decay: [0.8, 0.6, 0.4, 1.0],
            },
            WarpVert {
                pos: [1.0, 1.0],
                uv: [0.75, 0.25],
                decay: [0.4, 0.6, 0.8, 1.0],
            },
        ];
        let incoming = [
            WarpVert {
                pos: [-1.0, -1.0],
                uv: [0.75, 0.25],
                decay: [0.2, 0.4, 0.6, 1.0],
            },
            WarpVert {
                pos: [1.0, 1.0],
                uv: [0.25, 0.75],
                decay: [0.6, 0.4, 0.2, 1.0],
            },
        ];

        assert!(blend_evaluated_warp_mesh(&mut outgoing, &incoming, 0.5));
        assert_eq!(outgoing[0].uv, [0.5, 0.5]);
        assert_eq!(outgoing[1].uv, [0.5, 0.5]);
        assert_eq!(outgoing[0].decay, [0.5, 0.5, 0.5, 1.0]);
        assert!(outgoing
            .iter()
            .flat_map(|vertex| vertex.uv.iter().chain(vertex.decay.iter()))
            .all(|value| value.is_finite()));

        let before_invalid_progress = (outgoing[0].uv, outgoing[0].decay, outgoing[1].uv, outgoing[1].decay);
        assert!(!blend_evaluated_warp_mesh(&mut outgoing, &incoming, f32::NAN));
        assert_eq!(
            (outgoing[0].uv, outgoing[0].decay, outgoing[1].uv, outgoing[1].decay),
            before_invalid_progress
        );
    }

    #[test]
    fn canonical_mesh_and_blur_geometry_match_butterchurn() {
        assert_eq!((GRID_W, GRID_H), (48, 36));
        assert_eq!((COMP_GRID_W, COMP_GRID_H), (32, 24));
        assert_eq!(build_comp_indices().len(), 32 * 24 * 6);
        assert_eq!(
            blur_dimensions(1280, 720),
            [
                (320, 180),
                (160, 92),
                (80, 48),
                (640, 360),
                (160, 92),
                (80, 48),
            ]
        );
        assert_eq!(blur_dimensions(1, 1), [(16, 16); 6]);
    }

    #[test]
    fn comp_hue_mesh_has_butterchurn_topology_and_normalized_colors() {
        let mut vertices = Vec::new();
        generate_comp_verts(1.25, [0.1, 0.2, 0.3, 0.4], &mut vertices);
        assert_eq!(vertices.len(), 33 * 25);
        assert_eq!(vertices[0].pos, [-1.0, 1.0]);
        assert_eq!(vertices[32].pos, [1.0, 1.0]);
        assert_eq!(vertices[24 * 33].pos, [-1.0, -1.0]);
        assert!(vertices.iter().all(|vertex| {
            vertex.color[3] == 1.0
                && vertex.color[..3]
                    .iter()
                    .all(|channel| (0.5..=1.0).contains(channel))
        }));
    }

    #[test]
    fn comp_mesh_adapts_webgl_framebuffer_v_to_wgpu_texture_v() {
        let shader = include_str!("shaders/comp_mesh.wgsl");
        assert!(shader.contains("(1.0 - pos.y) * 0.5"));
    }

    #[test]
    fn feedback_mesh_flips_milkdrop_v_for_wgpu_textures() {
        let default_shader = include_str!("shaders/warp_mesh.wgsl");
        let custom_shader = include_str!("shaders/warp_mesh_vs.wgsl");
        for shader in [default_shader, custom_shader] {
            assert!(shader.contains("1.0 - v.uv.y"));
            assert!(shader.contains("1.0 - warp_uv.y"));
        }
    }

    #[test]
    fn per_pixel_angle_is_normalized_to_zero_through_tau() {
        use std::f64::consts::{FRAC_PI_2, PI, TAU};
        assert_eq!(milkdrop_angle(0.0, 0.0, 1.0, 1.0), 0.0);
        assert!((milkdrop_angle(0.0, 1.0, 1.0, 1.0) - FRAC_PI_2).abs() < 1.0e-12);
        assert!((milkdrop_angle(-1.0, 0.0, 1.0, 1.0) - PI).abs() < 1.0e-12);
        let lower = milkdrop_angle(0.0, -1.0, 1.0, 1.0);
        assert!((lower - (TAU - FRAC_PI_2)).abs() < 1.0e-12);
        assert!((0.0..TAU).contains(&lower));
    }

    #[test]
    fn volume_noise_mip_averages_all_eight_source_voxels() {
        let mut source = Vec::new();
        for value in 0u8..8 {
            source.extend_from_slice(&[value, value * 2, value * 3, 255]);
        }
        assert_eq!(downsample_rgba_volume(&source, 2), [3, 7, 10, 255]);
    }

    #[test]
    fn fused_custom_wave_emission_matches_legacy_smoothing() {
        let points = [[-1.0, 0.2], [-0.5, -0.3], [0.25, 0.8], [1.0, -0.1]];
        let colors = [
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, 0.4],
            [0.0, 0.0, 1.0, 0.6],
            [1.0, 1.0, 1.0, 0.8],
        ];
        let (legacy_points, legacy_colors) = smooth_wave_and_color(&points, &colors);
        let mut fused = Vec::new();
        let count = emit_smoothed_wave_and_color(&points, &colors, &mut fused);
        assert_eq!(count as usize, legacy_points.len());
        assert_eq!(fused.len(), legacy_points.len());
        for (index, vertex) in fused.iter().enumerate() {
            assert_eq!(vertex.pos, legacy_points[index]);
            assert_eq!(vertex.color, legacy_colors[index]);
        }
    }

    #[test]
    fn deterministic_clock_preserves_sub_frame_steps_past_f32_cliff() {
        let dt = 1.0 / 60.0;
        let frame = 1_u64 << 24;
        let t0 = deterministic_time_seconds(frame, Some(dt)).unwrap();
        let t1 = deterministic_time_seconds(frame + 1, Some(dt)).unwrap();

        assert!(((t1 - t0) - dt).abs() < 1.0e-10);

        let old_f32_t0 = frame as f32 * dt as f32;
        let old_f32_t1 = (frame + 1) as f32 * dt as f32;
        assert_eq!(old_f32_t0, old_f32_t1);
    }

    #[test]
    fn fixed_timestep_drives_the_exported_fps_value() {
        assert_eq!(effective_fps(None), 60.0);
        assert!((effective_fps(Some(1.0 / 30.0)) - 30.0).abs() < 1.0e-10);
        assert_eq!(effective_fps(Some(0.0)), 60.0);
        assert_eq!(effective_fps(Some(f64::NAN)), 60.0);
    }

    #[test]
    fn shader_time_and_frame_are_bounded_for_gpu_precision() {
        let long_time = GPU_TIME_WRAP_SECONDS * 1000.0 + 12.25;

        assert_eq!(shader_time_seconds(long_time), 12.25);
        assert_eq!(shader_progress(75.0), 0.5);
        assert_eq!(shader_frame_index(GPU_FRAME_WRAP + 42), 42.0);
    }

    #[test]
    fn interactive_resize_debouncer_coalesces_the_latest_size() {
        let start = Instant::now();
        let mut debouncer = MilkdropResizeDebouncer::default();

        assert!(debouncer.request(640, 360, start));
        assert!(debouncer.is_pending());
        // Repeated platform events for the same size do not starve the resize.
        assert!(!debouncer.request(640, 360, start + Duration::from_millis(10)));
        assert_eq!(
            debouncer.take_ready(start + Duration::from_millis(149)),
            None,
            "the full quiet period is required"
        );

        // A different later event replaces the older one; only the final size
        // survives a drag stream and only one target rebuild is requested.
        let final_request = start + Duration::from_millis(40);
        assert!(debouncer.request(1280, 720, final_request));
        assert_eq!(
            debouncer
                .take_ready(final_request + INTERACTIVE_RESIZE_DEBOUNCE - Duration::from_millis(1)),
            None
        );
        assert_eq!(
            debouncer.take_ready(final_request + INTERACTIVE_RESIZE_DEBOUNCE),
            Some((1280, 720))
        );
        assert!(!debouncer.is_pending());

        // Defensive normalization keeps a platform's transient zero size from
        // reaching wgpu if a caller chooses to queue it.
        assert!(debouncer.request(0, 0, final_request));
        assert_eq!(
            debouncer.take_ready(final_request + INTERACTIVE_RESIZE_DEBOUNCE),
            Some((1, 1))
        );
    }

    #[test]
    fn legacy_warp_compile_is_gone_and_live_warp_path_survives() {
        let compiled = compile_milkdrop_shader_bodies_from_parts(
            false,
            Some("ret = GetMain(uv) * 0.99;"),
            None,
        )
        .expect("live warp/comp paths must still compile");
        assert!(
            compiled.warp_wgsl.is_empty(),
            "legacy warp WGSL must no longer be produced"
        );
        // The live warp (mesh-VS) + comp paths still compile.
        assert!(!compiled.warp_custom_wgsl.is_empty());
        assert!(!compiled.comp_wgsl.is_empty());
    }

    #[test]
    fn needed_blur_levels_tracks_highest_sampled_level() {
        assert_eq!(needed_blur_levels(None, None), 0);
        assert_eq!(needed_blur_levels(Some("ret = GetMain(uv);"), None), 0);
        assert_eq!(needed_blur_levels(None, Some("ret = GetBlur1(uv);")), 1);
        assert_eq!(needed_blur_levels(None, Some("ret = GetBlur2(uv);")), 2);
        assert_eq!(needed_blur_levels(Some("ret = GetBlur3(uv);"), None), 3);
        // Direct sampler reference + case-insensitivity are both recognized.
        assert_eq!(
            needed_blur_levels(None, Some("ret = tex2D(SAMPLER_BLUR2, uv).xyz;")),
            2
        );
        // The highest level across warp AND comp wins (progressive chain).
        assert_eq!(
            needed_blur_levels(Some("ret = GetBlur1(uv);"), Some("ret = GetBlur3(uv);")),
            3
        );
    }

    #[test]
    fn needed_blur_levels_detects_mode_prefixed_samplers() {
        // pw-prefixed blur2 in a comp body → level 2 (regression: was 0).
        assert_eq!(
            needed_blur_levels(None, Some("ret = tex2D(sampler_pw_blur2, uv).xyz;")),
            2
        );
        // The other three mode prefixes the normalizer collapses are all detected.
        assert_eq!(
            needed_blur_levels(Some("ret = tex2D(sampler_fw_blur1, uv).xyz;"), None),
            1
        );
        assert_eq!(
            needed_blur_levels(None, Some("ret = tex2D(sampler_fc_blur3, uv).xyz;")),
            3
        );
        assert_eq!(
            needed_blur_levels(Some("ret = tex2D(sampler_pc_blur2, uv).xyz;"), None),
            2
        );
        // Source-case variant of a prefixed sampler is still caught.
        assert_eq!(
            needed_blur_levels(None, Some("ret = tex2D(SAMPLER_PW_BLUR2, uv).xyz;")),
            2
        );
        // Plain `getblurN` (unaffected by sampler normalization) still resolves.
        assert_eq!(needed_blur_levels(Some("ret = GetBlur3(uv);"), None), 3);
    }

    #[test]
    fn validate_texture_dims_accepts_reasonable_and_rejects_extremes() {
        // Ordinary and exactly-at-max dimensions are accepted.
        assert!(validate_texture_dims(16384, 1920, 1080).is_ok());
        assert!(validate_texture_dims(16384, 16384, 16384).is_ok());
        // Zero is rejected.
        assert_eq!(
            validate_texture_dims(16384, 0, 720),
            Err(DimensionError::Zero)
        );
        // Over the device max_texture_dimension_2d → typed rejection, no allocation.
        assert!(matches!(
            validate_texture_dims(8192, 100_000, 100_000),
            Err(DimensionError::ExceedsMaxTextureDimension { .. })
        ));
        // Within a permissive max but the total footprint is absurd.
        assert!(matches!(
            validate_texture_dims(u32::MAX, 300_000, 300_000),
            Err(DimensionError::ExceedsMemoryBudget { .. })
        ));
        // The byte arithmetic itself overflows u64 → caught, not wrapped.
        assert!(matches!(
            validate_texture_dims(u32::MAX, u32::MAX, u32::MAX),
            Err(DimensionError::ArithmeticOverflow)
        ));
    }

    // Texture-dimension boundary: the existing test probes values far
    // outside the caps (100_000 against a max of 8192). Pin the exact
    // edges so an off-by-one in either comparison is caught, and cover
    // the single-axis degenerate cases.
    #[test]
    fn validate_texture_dims_boundaries_are_exact() {
        const MAX: u32 = 16384;

        // max - 1, max: accepted. max + 1: the first rejection, and the
        // typed error carries the offending pair plus the cap.
        assert!(validate_texture_dims(MAX, MAX - 1, MAX - 1).is_ok());
        assert!(validate_texture_dims(MAX, MAX, MAX).is_ok());
        match validate_texture_dims(MAX, MAX + 1, 16) {
            Err(DimensionError::ExceedsMaxTextureDimension { width, height, max }) => {
                assert_eq!((width, height, max), (MAX + 1, 16, MAX));
            }
            other => panic!("expected ExceedsMaxTextureDimension at max + 1, got {other:?}"),
        }
        // Either axis alone is enough to trip it.
        assert!(matches!(
            validate_texture_dims(MAX, 16, MAX + 1),
            Err(DimensionError::ExceedsMaxTextureDimension { .. })
        ));

        // One pixel is legal; zero on EITHER axis is not (the existing
        // test only covers a zero width).
        assert!(validate_texture_dims(MAX, 1, 1).is_ok());
        assert_eq!(
            validate_texture_dims(MAX, 1280, 0),
            Err(DimensionError::Zero)
        );
        assert_eq!(validate_texture_dims(MAX, 0, 0), Err(DimensionError::Zero));

        // u32::MAX on one axis is refused by the dimension cap, before any
        // of the byte arithmetic runs.
        assert!(matches!(
            validate_texture_dims(MAX, u32::MAX, 1),
            Err(DimensionError::ExceedsMaxTextureDimension { .. })
        ));

        // Memory-budget edge, isolated from the dimension cap by a
        // permissive max: total = w * h * 4 * TEXTURE_FOOTPRINT_MULTIPLIER.
        let max_pixels =
            super::MAX_TEXTURE_MEMORY_BYTES / (4 * super::TEXTURE_FOOTPRINT_MULTIPLIER);
        let at_budget = u32::try_from(max_pixels).expect("budget edge fits in u32");
        assert!(
            validate_texture_dims(u32::MAX, at_budget, 1).is_ok(),
            "a footprint landing exactly on the memory budget must be accepted"
        );
        match validate_texture_dims(u32::MAX, at_budget + 1, 1) {
            Err(DimensionError::ExceedsMemoryBudget { bytes, budget }) => {
                assert_eq!(budget, super::MAX_TEXTURE_MEMORY_BYTES);
                assert!(bytes > super::MAX_TEXTURE_MEMORY_BYTES);
            }
            other => panic!("expected ExceedsMemoryBudget one pixel past the edge, got {other:?}"),
        }
    }

    // ── GPU-backed regressions (need a real adapter; skipped if none) ───────────
    #[cfg(feature = "app")]
    fn gpu_device() -> Option<(std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("milk-test"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        }))
        .ok()?;
        Some((std::sync::Arc::new(device), std::sync::Arc::new(queue)))
    }

    #[cfg(feature = "app")]
    fn offscreen_target(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        fmt: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("test-target"),
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
            })
            .create_view(&Default::default())
    }

    #[cfg(feature = "app")]
    fn seed_asymmetric_feedback(renderer: &mut MilkdropRenderer) {
        let width = renderer.render_w;
        let height = renderer.render_h;
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let pixel = ((y * width + x) * 4) as usize;
                // The off-centre L shape makes a horizontal translation
                // distinguishable from a dissolve or a uniform black page.
                let left_bar = x > width / 8 && x < width / 4 && y > height / 5;
                let top_bar = y > height / 6 && y < height / 3 && x < width * 3 / 4;
                if left_bar || top_bar {
                    pixels[pixel] = 220;
                    pixels[pixel + 1] = if top_bar { 80 } else { 20 };
                    pixels[pixel + 2] = if left_bar { 180 } else { 30 };
                    pixels[pixel + 3] = 255;
                }
            }
        }
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        for texture in [&renderer.tex_a, &renderer.tex_b] {
            renderer.queue.write_texture(
                texture.as_image_copy(),
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                extent,
            );
        }
        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shared-feedback-asymmetric-seed"),
            });
        super::generate_mip_chain(
            &renderer.device,
            &renderer.feedback_mip_blitter,
            &mut encoder,
            &renderer.feedback_mips_a,
        );
        super::generate_mip_chain(
            &renderer.device,
            &renderer.feedback_mip_blitter,
            &mut encoder,
            &renderer.feedback_mips_b,
        );
        renderer.queue.submit(std::iter::once(encoder.finish()));
    }

    #[cfg(feature = "app")]
    #[test]
    fn shared_feedback_mid_morph_has_visible_asymmetric_pixels_and_advances_both_states() {
        let (device, queue) = gpu_device().expect(
            "shared-feedback regression requires a real wgpu adapter; do not treat a missing adapter as success",
        );
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let outgoing_preset = crate::parse_milk::parse(
            "fWaveAlpha=0\n\
             bMotionVectorsOn=0\n\
             per_frame_1=dx=0.125;reg00=reg00+1;\n",
        );
        let incoming_preset = crate::parse_milk::parse(
            "fWaveAlpha=0\n\
             bMotionVectorsOn=0\n\
             fGammaAdj=3\n\
             per_frame_1=dx=-0.125;reg00=reg00+2;\n",
        );
        let target_view = offscreen_target(&device, 64, 64, format);
        // Match the runtime build worker: each renderer receives a fresh Arc
        // around a clone of the same wgpu device/queue, rather than `Arc::clone`
        // of one wrapper. `wgpu::Device` equality must retain this pair.
        let mut outgoing = MilkdropRenderer::new(
            std::sync::Arc::new(device.as_ref().clone()),
            std::sync::Arc::new(queue.as_ref().clone()),
            64,
            64,
            format,
            &outgoing_preset,
        )
        .expect("outgoing renderer");
        let mut incoming = MilkdropRenderer::new(
            std::sync::Arc::new(device.as_ref().clone()),
            std::sync::Arc::new(queue.as_ref().clone()),
            64,
            64,
            format,
            &incoming_preset,
        )
        .expect("incoming renderer");
        assert!(outgoing.supports_shared_feedback(&incoming));
        assert_eq!(
            outgoing.shared_feedback_support(&incoming),
            super::SharedFeedbackSupport::FeedbackOnlyInterpolatedComp
        );
        let overlay_preset = crate::parse_milk::parse("fWaveAlpha=0.25\nbMotionVectorsOn=0\n");
        let overlay_renderer = MilkdropRenderer::new(
            device.clone(),
            incoming.queue.clone(),
            64,
            64,
            format,
            &overlay_preset,
        )
        .expect("overlay-bearing renderer");
        assert_eq!(
            outgoing.shared_feedback_support(&overlay_renderer),
            super::SharedFeedbackSupport::UntexturedOverlaysInterpolatedComp,
            "compatible waves must fade in the shared overlay pass rather than pop on promotion"
        );
        let hue_shader_preset = crate::parse_milk::parse(
            "fWaveAlpha=0\n\
             bMotionVectorsOn=0\n\
             fShader=1\n",
        );
        let hue_shader_renderer = MilkdropRenderer::new(
            std::sync::Arc::new(device.as_ref().clone()),
            std::sync::Arc::new(queue.as_ref().clone()),
            64,
            64,
            format,
            &hue_shader_preset,
        )
        .expect("hue-shader renderer");
        assert_eq!(
            outgoing.shared_feedback_support(&hue_shader_renderer),
            super::SharedFeedbackSupport::DiscreteCompUnsupported,
            "fShader must use fallback because outgoing rand_start hue would pop at promotion"
        );

        seed_asymmetric_feedback(&mut outgoing);
        assert!(incoming.seed_feedback_from(&outgoing));
        outgoing.set_geometry_diagnostics_enabled(true);
        let before_invalid = (outgoing.frame_idx, incoming.frame_idx);
        assert!(
            !outgoing.render_shared_feedback(&target_view, f32::NAN, &mut incoming),
            "non-finite progress must select the caller fallback without advancing either state"
        );
        assert_eq!((outgoing.frame_idx, incoming.frame_idx), before_invalid);

        assert!(
            outgoing.render_shared_feedback(&target_view, 0.5, &mut incoming),
            "the p=0.5 frame must use the evaluated UV blend before feedback sampling"
        );
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll after shared feedback render");
        let post_warp = outgoing
            .geometry_stage_images()
            .expect("diagnostics enabled before shared-feedback render")
            .post_warp_rgba;
        assert!(
            post_warp.chunks_exact(4).any(|pixel| pixel[0] > 24 || pixel[2] > 24),
            "asymmetric seeded feedback must remain visibly populated before progress reaches 1"
        );
        assert!(
            post_warp
                .chunks_exact(4)
                .any(|pixel| pixel[0] != pixel[2]),
            "the mid-morph must preserve asymmetric seeded colour evidence"
        );
        assert_eq!(outgoing.frame_idx, 1);
        assert_eq!(incoming.frame_idx, 1);
        assert!(
            (outgoing.last_comp_perframe.gamma_adj - 2.5).abs() < 1.0e-6,
            "the visible built-in comp pass must interpolate outgoing/incoming gamma"
        );
        assert_ne!(
            outgoing.eel_env.get("reg00").copied(),
            incoming.eel_env.get("reg00").copied(),
            "outgoing and incoming equation states must advance independently"
        );
    }

    #[cfg(feature = "app")]
    #[test]
    fn shared_feedback_blends_untextured_builtin_wave_overlays_before_completion() {
        let (device, queue) = gpu_device().expect(
            "shared overlay regression requires a real wgpu adapter; do not treat a missing adapter as success",
        );
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let outgoing_preset = crate::parse_milk::parse(
            "fWaveAlpha=1\n\
             wave_r=1\n\
             wave_g=0\n\
             wave_b=0\n\
             bMotionVectorsOn=0\n",
        );
        let incoming_preset = crate::parse_milk::parse(
            "fWaveAlpha=1\n\
             wave_r=0\n\
             wave_g=0\n\
             wave_b=1\n\
             bMotionVectorsOn=0\n",
        );
        let target_view = offscreen_target(&device, 64, 64, format);
        let mut outgoing = MilkdropRenderer::new(
            std::sync::Arc::new(device.as_ref().clone()),
            std::sync::Arc::new(queue.as_ref().clone()),
            64,
            64,
            format,
            &outgoing_preset,
        )
        .expect("outgoing overlay renderer");
        let mut incoming = MilkdropRenderer::new(
            std::sync::Arc::new(device.as_ref().clone()),
            std::sync::Arc::new(queue.as_ref().clone()),
            64,
            64,
            format,
            &incoming_preset,
        )
        .expect("incoming overlay renderer");
        assert_eq!(
            outgoing.shared_feedback_support(&incoming),
            super::SharedFeedbackSupport::UntexturedOverlaysInterpolatedComp
        );

        let left: Vec<f32> = (0..512)
            .map(|index| (index as f32 * 2.0 * std::f32::consts::PI * 7.0 / 512.0).sin())
            .collect();
        let right: Vec<f32> = (0..512)
            .map(|index| (index as f32 * 2.0 * std::f32::consts::PI * 11.0 / 512.0).sin())
            .collect();
        outgoing.set_waveform(&left, &right);
        outgoing.set_geometry_diagnostics_enabled(true);
        assert!(
            outgoing.render_shared_feedback(&target_view, 0.5, &mut incoming),
            "the compatible overlay pair must use the shared renderer path"
        );
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll after shared overlay render");
        let post_overlays = outgoing
            .geometry_stage_images()
            .expect("diagnostics enabled before shared overlay render")
            .post_overlays_rgba;
        assert!(
            post_overlays
                .chunks_exact(4)
                .any(|pixel| pixel[0] > 20 && pixel[2] > 20),
            "p=0.5 must show both outgoing red and incoming blue waveform overlays before promotion"
        );
        assert_eq!(outgoing.frame_idx, 1);
        assert_eq!(incoming.frame_idx, 1);
    }

    #[cfg(feature = "app")]
    #[test]
    fn shared_feedback_blends_untextured_static_shape_overlays_before_completion() {
        let (device, queue) = gpu_device().expect(
            "shared shape-overlay regression requires a real wgpu adapter; do not treat a missing adapter as success",
        );
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let outgoing_preset = crate::parse_milk::parse(
            "fWaveAlpha=0\n\
             bMotionVectorsOn=0\n\
             shapecode_0_enabled=1\n\
             shapecode_0_sides=4\n\
             shapecode_0_x=0.5\n\
             shapecode_0_y=0.5\n\
             shapecode_0_rad=0.35\n\
             shapecode_0_r=1\n\
             shapecode_0_g=0\n\
             shapecode_0_b=0\n\
             shapecode_0_a=1\n\
             shapecode_0_r2=1\n\
             shapecode_0_g2=0\n\
             shapecode_0_b2=0\n\
             shapecode_0_a2=1\n\
             shapecode_0_border_a=0\n",
        );
        let incoming_preset = crate::parse_milk::parse(
            "fWaveAlpha=0\n\
             bMotionVectorsOn=0\n\
             shapecode_0_enabled=1\n\
             shapecode_0_sides=4\n\
             shapecode_0_x=0.5\n\
             shapecode_0_y=0.5\n\
             shapecode_0_rad=0.35\n\
             shapecode_0_r=0\n\
             shapecode_0_g=0\n\
             shapecode_0_b=1\n\
             shapecode_0_a=1\n\
             shapecode_0_r2=0\n\
             shapecode_0_g2=0\n\
             shapecode_0_b2=1\n\
             shapecode_0_a2=1\n\
             shapecode_0_border_a=0\n",
        );
        let target_view = offscreen_target(&device, 64, 64, format);
        let mut outgoing = MilkdropRenderer::new(
            std::sync::Arc::new(device.as_ref().clone()),
            std::sync::Arc::new(queue.as_ref().clone()),
            64,
            64,
            format,
            &outgoing_preset,
        )
        .expect("outgoing shape renderer");
        let mut incoming = MilkdropRenderer::new(
            std::sync::Arc::new(device.as_ref().clone()),
            std::sync::Arc::new(queue.as_ref().clone()),
            64,
            64,
            format,
            &incoming_preset,
        )
        .expect("incoming shape renderer");
        assert_eq!(
            outgoing.shared_feedback_support(&incoming),
            super::SharedFeedbackSupport::UntexturedOverlaysInterpolatedComp
        );

        outgoing.set_geometry_diagnostics_enabled(true);
        assert!(outgoing.render_shared_feedback(&target_view, 0.5, &mut incoming));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll after shared shape-overlay render");
        assert!(outgoing.scratch.shape_fill_draws.iter().any(|draw| draw.sides == 4));
        let post_overlays = outgoing
            .geometry_stage_images()
            .expect("diagnostics enabled before shared shape-overlay render")
            .post_overlays_rgba;
        assert!(
            post_overlays
                .chunks_exact(4)
                .any(|pixel| pixel[0] > 20 && pixel[2] > 20),
            "p=0.5 must show both outgoing red and incoming blue untextured shapes before promotion"
        );

        let textured = crate::parse_milk::parse(
            "fWaveAlpha=0\n\
             bMotionVectorsOn=0\n\
             shapecode_0_enabled=1\n\
             shapecode_0_textured=1\n",
        );
        let textured_renderer = MilkdropRenderer::new(
            std::sync::Arc::new(device.as_ref().clone()),
            std::sync::Arc::new(queue.as_ref().clone()),
            64,
            64,
            format,
            &textured,
        )
        .expect("textured shape renderer");
        assert_eq!(
            outgoing.shared_feedback_support(&textured_renderer),
            super::SharedFeedbackSupport::VisibleOverlaysUnsupported,
            "textured shapes must retain the caller fallback instead of sampling a renderer-local page"
        );
    }

    #[cfg(feature = "app")]
    #[test]
    fn beatdrop_feedback_places_vectors_in_history_before_warp() {
        let (device, queue) = gpu_device().expect("feedback provenance requires a real GPU");
        let preset = crate::parse_milk::parse(
            "fWaveAlpha=0\nbMotionVectorsOn=1\nnMotionVectorsX=4\nnMotionVectorsY=4\nmv_a=1\ndx=0.125\n",
        );
        let target = offscreen_target(&device, 64, 64, wgpu::TextureFormat::Rgba8Unorm);
        let mut legacy = MilkdropRenderer::new(
            device.clone(), queue.clone(), 64, 64, wgpu::TextureFormat::Rgba8Unorm, &preset,
        ).unwrap();
        let mut compatible = MilkdropRenderer::new(
            device.clone(), queue, 64, 64, wgpu::TextureFormat::Rgba8Unorm, &preset,
        ).unwrap();
        assert_eq!(legacy.feedback_provenance(), super::FeedbackProvenance::Legacy);
        compatible.set_beatdrop_feedback(true);
        // Exercise the previous-history blur branch even with a default COMP.
        legacy.blur_levels = 1;
        compatible.blur_levels = 1;
        legacy.set_geometry_diagnostics_enabled(true);
        compatible.set_geometry_diagnostics_enabled(true);
        legacy.render(&target);
        compatible.render(&target);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let legacy = legacy.geometry_stage_images().unwrap();
        let compatible = compatible.geometry_stage_images().unwrap();
        let has_color = |bytes: &[u8]| bytes.chunks_exact(4).any(|pixel|
            pixel[0] > 8 || pixel[1] > 8 || pixel[2] > 8);
        assert!(!has_color(&legacy.post_warp_rgba), "legacy vectors arrive after warp");
        assert!(has_color(&legacy.post_overlays_rgba), "the legacy vector draw must be non-vacuous");
        assert!(has_color(&compatible.post_warp_rgba), "compatible warp must sample this frame's vectors");
    }

    #[cfg(feature = "app")]
    #[test]
    fn mode8_uses_bounded_pcm_stereo_spectrum_without_enhanced_shader_helpers() {
        let (device, queue) = gpu_device().expect(
            "mode-8 audio regression requires a real wgpu adapter; do not treat a missing adapter as success",
        );
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let preset = crate::parse_milk::parse(
            "nWaveMode=8\n\
             fWaveAlpha=1\n\
             wave_r=1\n\
             wave_g=0\n\
             wave_b=0\n\
             bMotionVectorsOn=0\n",
        );
        let target = offscreen_target(&device, 64, 64, format);
        let mut renderer = MilkdropRenderer::new(device, queue, 64, 64, format, &preset)
            .expect("mode-8 renderer");
        assert!(
            !renderer.enhanced_audio_enabled,
            "the preset has no helper calls: this covers the mode-8-only DSP path"
        );

        // Deliberately distinguish the channels: the bounded in-renderer FFT is
        // allowed to compute them, whereas the legacy mono `freqArray` is not.
        let left: Vec<f32> = (0..1024)
            .map(|index| (index as f32 * 2.0 * std::f32::consts::PI * 19.0 / 1024.0).sin())
            .collect();
        let right: Vec<f32> = (0..1024)
            .map(|index| (index as f32 * 2.0 * std::f32::consts::PI * 47.0 / 1024.0).sin() * 0.5)
            .collect();
        renderer.set_waveform(&left, &right);
        renderer.set_enhanced_audio(
            None,
            None,
            &left,
            &right,
            48_000.0,
            1.0 / 60.0,
            EnhancedAudioConfig::default(),
        );
        assert_eq!(renderer.freq_spectrum.len(), ENHANCED_FFT_BINS);
        assert_eq!(renderer.freq_spectrum_right.len(), ENHANCED_FFT_BINS);
        assert!(
            renderer
                .freq_spectrum
                .iter()
                .zip(&renderer.freq_spectrum_right)
                .any(|(left, right)| (left - right).abs() > 1.0e-4),
            "mode 8 must receive independent raw PCM magnitudes, not a copied mono row"
        );

        renderer.set_geometry_diagnostics_enabled(true);
        renderer.render(&target);
        renderer
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll after mode-8 render");
        assert!(
            renderer.scratch.wave_draws.iter().any(|draw| draw.count > 1),
            "mode 8 must emit a drawable built-in waveform after its PCM FFT"
        );
        assert!(renderer
            .scratch
            .wave_verts
            .iter()
            .all(|vertex| vertex.pos.iter().chain(vertex.color.iter()).all(|value| value.is_finite())));
        let overlays = renderer
            .geometry_stage_images()
            .expect("diagnostics enabled before mode-8 rendering")
            .post_overlays_rgba;
        assert!(
            overlays.chunks_exact(4).any(|pixel| pixel[0] > 16),
            "the GPU overlay checkpoint must contain the mode-8 waveform"
        );
    }

    #[cfg(feature = "app")]
    #[test]
    fn reduced_profiles_render_to_hdr_without_validation_errors() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let format = wgpu::TextureFormat::Rgba16Float;
        let preset = crate::parse_milk::parse("");
        let target = offscreen_target(&device, 64, 64, format);
        let mut renderer = MilkdropRenderer::new(device.clone(), queue, 64, 64, format, &preset)
            .expect("HDR renderer");

        for profile in [
            MilkdropPerformanceProfile::High60,
            MilkdropPerformanceProfile::Balanced60,
            MilkdropPerformanceProfile::Rescue60,
        ] {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            renderer.set_performance_profile(profile);
            renderer.render(&target);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU poll");
            let error = pollster::block_on(scope.pop());
            assert!(
                error.is_none(),
                "{profile:?} raised an HDR validation error: {error:?}"
            );
        }
    }

    #[cfg(feature = "app")]
    #[test]
    fn init_lifecycle_threads_q_regs_and_distinct_random_vectors() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let preset = crate::parse_milk::parse(
            "per_frame_init_1=q1=2;reg00=3;\n\
             per_frame_1=q1=q1+1;reg00=reg00+1;\n\
             wavecode_0_enabled=1\n\
             wave_0_per_frame_init_1=t1=q1;reg01=reg00+10;\n\
             wave_0_per_frame_1=t1=t1+1+q8*0+reg04*0;\n\
             shapecode_0_enabled=1\n\
             shape_0_per_frame_init_1=t1=q1;reg02=reg01+20;\n\
             shape_0_per_frame_1=t1=t1+1+q7*0+reg03*0;\n",
        );
        let target = offscreen_target(&device, 64, 64, wgpu::TextureFormat::Rgba8Unorm);
        let mut renderer = MilkdropRenderer::new(
            device,
            queue,
            64,
            64,
            wgpu::TextureFormat::Rgba8Unorm,
            &preset,
        )
        .expect("renderer with threaded init state");

        assert_eq!(renderer.waves[0].env.get("t1").copied(), Some(3.0));
        assert_eq!(renderer.shapes[0].env.get("t1").copied(), Some(3.0));
        assert_eq!(renderer.eel_env.get("reg00").copied(), Some(4.0));
        assert_eq!(renderer.eel_env.get("reg01").copied(), Some(14.0));
        assert_eq!(renderer.eel_env.get("reg02").copied(), Some(34.0));
        assert_eq!(renderer.waves[0].live_reg_indices.as_slice(), &[4]);
        assert_eq!(renderer.waves[0].live_q_indices.as_slice(), &[7]);
        assert_eq!(renderer.waves[0].live_t_indices.as_slice(), &[0]);
        assert_eq!(renderer.shapes[0].live_reg_indices.as_slice(), &[3]);
        assert_eq!(renderer.shapes[0].live_q_indices.as_slice(), &[6]);
        assert_eq!(renderer.shapes[0].live_t_indices.as_slice(), &[0]);
        assert_ne!(renderer.rand_start, renderer.rand_preset);
        for _ in 0..2 {
            renderer.render(&target);
            assert_eq!(renderer.eel_env.get("q1").copied(), Some(3.0));
            assert_eq!(renderer.waves[0].env.get("t1").copied(), Some(4.0));
            assert_eq!(renderer.shapes[0].env.get("t1").copied(), Some(4.0));
        }
    }

    #[cfg(feature = "app")]
    #[test]
    fn every_builtin_is_reset_to_its_preset_base_before_frame_equations() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let preset = crate::parse_milk::parse(
            "fVideoEchoAlpha=0.2\n\
             fShader=0.6\n\
             ob_size=0.05\n\
             per_frame_1=echo_alpha=echo_alpha+1;ob_size=ob_size+0.1;fshader=fshader+1;\n",
        );
        let target = offscreen_target(&device, 64, 64, wgpu::TextureFormat::Rgba8Unorm);
        let mut renderer = MilkdropRenderer::new(
            device,
            queue,
            64,
            64,
            wgpu::TextureFormat::Rgba8Unorm,
            &preset,
        )
        .expect("renderer with self-updating built-ins");

        for _ in 0..2 {
            renderer.render(&target);
            let echo = renderer.eel_env.get("echo_alpha").copied().unwrap();
            let border = renderer.eel_env.get("ob_size").copied().unwrap();
            let fshader = renderer.eel_env.get("fshader").copied().unwrap();
            assert!((echo - 1.2).abs() < 1.0e-6, "echo accumulated: {echo}");
            assert!(
                (border - 0.15).abs() < 1.0e-6,
                "border accumulated: {border}"
            );
            assert!(
                (fshader - 1.6).abs() < 1.0e-6,
                "fshader accumulated or lost its preset base: {fshader}"
            );
        }
    }

    #[cfg(feature = "app")]
    #[test]
    fn named_texture_atlas_binds_and_renders_through_the_real_pipeline() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let mut shaders = crate::parse_milk::parse("");
        shaders.comp = Some("ret = tex2D(sampler_fw_worms, uv).rgb;".to_string());
        let mut renderer = MilkdropRenderer::new(device.clone(), queue, 64, 64, fmt, &shaders)
            .expect("named-texture renderer");
        renderer.render(&offscreen_target(&device, 64, 64, fmt));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll");
    }

    #[cfg(feature = "app")]
    #[test]
    fn blur_passes_scale_with_sampled_levels() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (64u32, 64u32);
        let target = offscreen_target(&device, w, h, fmt);

        // A default preset samples no blur → zero blur draws (was 6 before the fix).
        let plain = crate::parse_milk::parse("");
        let mut r0 = MilkdropRenderer::new(device.clone(), queue.clone(), w, h, fmt, &plain)
            .expect("plain renderer");
        assert_eq!(r0.blur_levels(), 0);
        r0.render(&target);
        assert_eq!(r0.last_blur_pass_count(), 0);

        // A comp shader that samples blur2 needs blur1 + blur2 → four blur draws.
        let mut sh = crate::parse_milk::parse("");
        sh.comp = Some("ret = GetBlur2(uv);".to_string());
        let mut r2 = MilkdropRenderer::new(device.clone(), queue.clone(), w, h, fmt, &sh)
            .expect("blur2 renderer");
        assert_eq!(r2.blur_levels(), 2);
        r2.render(&target);
        assert_eq!(r2.last_blur_pass_count(), 4);
    }

    #[cfg(feature = "app")]
    #[test]
    fn shapes_render_without_shapeu_binding() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (64u32, 64u32);
        let target = offscreen_target(&device, w, h, fmt);

        let shaders = crate::parse_milk::parse(
            "shapecode_0_enabled=1\nshapecode_0_sides=4\nshapecode_0_rad=0.4\nshapecode_0_a=1\n",
        );
        assert!(!shaders.shapes.is_empty(), "preset must have a shape");

        // Any bind-group/layout inconsistency from removing the ShapeU binding
        // (binding 2) would surface here as a wgpu validation error.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut r = MilkdropRenderer::new(device.clone(), queue.clone(), w, h, fmt, &shaders)
            .expect("renderer with a shape");
        r.render(&target);
        let err = pollster::block_on(scope.pop());
        assert!(
            err.is_none(),
            "shape rendering raised a validation error: {err:?}"
        );
    }

    #[cfg(feature = "app")]
    #[test]
    fn renderer_rejects_oversized_dimensions_without_allocating() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let plain = crate::parse_milk::parse("");
        let max_dim = device.limits().max_texture_dimension_2d;
        let huge = max_dim.saturating_add(1).max(100_000);

        // Construction rejects the over-limit size with a typed error mapped to a
        // String — no panic, no giant texture/CPU-seed allocation. (MilkdropRenderer
        // isn't Debug, so unwrap the error via `.err()` rather than `expect_err`.)
        let err = MilkdropRenderer::new(device.clone(), queue.clone(), huge, huge, fmt, &plain)
            .err()
            .expect("oversized target must be rejected");
        assert!(
            err.contains("max_texture_dimension_2d"),
            "unexpected error: {err}"
        );

        // A valid renderer declines an oversized try_resize with a typed error and
        // stays usable at a subsequent valid size.
        let mut r = MilkdropRenderer::new(device.clone(), queue.clone(), 64, 64, fmt, &plain)
            .expect("small renderer");
        let e = r
            .try_resize(huge, huge)
            .expect_err("oversized resize must be rejected");
        assert!(matches!(
            e,
            DimensionError::ExceedsMaxTextureDimension { .. }
        ));
        r.try_resize(128, 128)
            .expect("valid resize must still work");
    }

    #[test]
    fn resample_linear_adapts_length_without_collapsing() {
        // Identity when lengths already match.
        assert_eq!(resample_linear(&[0.0, 1.0, 2.0], 3), vec![0.0, 1.0, 2.0]);
        // Empty / degenerate inputs stay empty.
        assert!(resample_linear(&[], 8).is_empty());
        assert!(resample_linear(&[1.0, 2.0], 0).is_empty());
        // A single-sample source broadcasts.
        assert_eq!(resample_linear(&[0.7], 4), vec![0.7, 0.7, 0.7, 0.7]);
        // A constant array stays constant at any target length (the FFT case in the
        // regression below): all-ones @512 → all-ones @480, NOT a time fallback.
        let ones = vec![1.0f32; 512];
        let rs = resample_linear(&ones, 480);
        assert_eq!(rs.len(), 480);
        assert!(rs.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // Endpoints are preserved; interior is monotone for a ramp.
        let ramp: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let up = resample_linear(&ramp, 9);
        assert_eq!(up.len(), 9);
        assert!((up[0] - 0.0).abs() < 1e-6);
        assert!((up[8] - 4.0).abs() < 1e-6);
        assert!(up.windows(2).all(|w| w[1] >= w[0] - 1e-6));
    }

    #[test]
    fn resample_linear_into_reuses_audio_scratch_and_clamps_pcm() {
        let source = [f32::NAN, -2.0, 2.0];
        let mut scratch = Vec::<f32>::with_capacity(512);
        let initial_ptr = scratch.as_ptr();
        resample_linear_into(&source, 512, &mut scratch, true);

        assert_eq!(scratch.len(), 512);
        assert_eq!(scratch.as_ptr(), initial_ptr, "must reuse audio scratch");
        assert!(scratch
            .iter()
            .all(|value| value.is_finite() && (-1.0..=1.0).contains(value)));

        // Replacing a full row must keep the same backing allocation too.
        let full = [0.25f32; 512];
        resample_linear_into(&full, 512, &mut scratch, true);
        assert_eq!(scratch.as_ptr(), initial_ptr, "must not allocate per frame");
        assert!(scratch.iter().all(|value| (*value - 0.25).abs() < 1e-6));
    }

    #[cfg(feature = "app")]
    #[test]
    fn spectrum_wave_uses_fft_independent_of_pcm_length() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        // One enabled spectrum custom wave (reads freqArray, not the PCM time data).
        let shaders = crate::parse_milk::parse(
            "wavecode_0_enabled=1\nwavecode_0_bSpectrum=1\nwavecode_0_samples=256\nwavecode_0_a=1\n",
        );
        let mut r = MilkdropRenderer::new(device, queue, 64, 64, fmt, &shaders)
            .expect("renderer with a spectrum wave");

        // Silent PCM (all 0) at ONE valid length; a hot FFT (all 1) at a DIFFERENT
        // valid length. Pre-fix, `freq.len() != max_samples` forced a time fallback,
        // collapsing every point to value1 = 0 (x == 0). The fix resamples the FFT
        // to the working length independently, so the wave reflects the FFT.
        let time_l = vec![0.0f32; 480];
        let time_r = vec![0.0f32; 480];
        let freq = vec![1.0f32; 512];
        let regs = [0.0f64; 100];

        let mut verts: Vec<super::WaveVert> = Vec::new();
        let mut draws: Vec<super::WaveDraw> = Vec::new();
        r.build_custom_waves(
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, &time_l, &time_r, &freq, &regs,
            &mut verts, &mut draws,
        );
        assert!(!draws.is_empty(), "spectrum wave must emit geometry");
        // With a hot FFT the point positions spread away from centre; a time-domain
        // fallback on silent PCM would leave every x pinned at 0.
        let max_abs_x = verts.iter().map(|v| v.pos[0].abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs_x > 1e-3,
            "spectrum wave ignored the FFT (fell back to time data): max|x| = {max_abs_x}"
        );
    }

    #[cfg(feature = "app")]
    #[test]
    fn absurd_shape_instance_count_is_bounded_to_the_vertex_cap() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        // A preset demanding a million 100-gon instances. Pre-fix this ran the full
        // clamped 1024 instances × 102 verts ≈ 104k CPU verts — past the 65_536
        // vertex-buffer cap — every frame. The bound stops before overflowing.
        let shaders = crate::parse_milk::parse(
            "shapecode_0_enabled=1\nshapecode_0_num_inst=1000000\nshapecode_0_sides=100\nshapecode_0_a=1\nshapecode_0_border_a=1\n",
        );
        let mut r = MilkdropRenderer::new(device, queue, 64, 64, fmt, &shaders)
            .expect("renderer with an over-instanced shape");
        let q = [0.0f64; 32];
        let regs = [0.0f64; 100];
        let (fill_verts, fill_draws, border_verts, _border_draws) =
            r.build_shape_geometry(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, &q, &regs);

        // CPU geometry never exceeds the fixed vertex-buffer capacities.
        assert!(
            fill_verts.len() <= super::SHAPE_VERT_CAP,
            "fill verts {} exceeded SHAPE_VERT_CAP {}",
            fill_verts.len(),
            super::SHAPE_VERT_CAP
        );
        assert!(
            border_verts.len() <= super::BORDER_VERT_CAP,
            "border verts {} exceeded BORDER_VERT_CAP {}",
            border_verts.len(),
            super::BORDER_VERT_CAP
        );
        // The cap actually engaged: fewer than the 1024-instance clamp were emitted
        // (each 100-gon instance needs 102 verts, so ≤ ~642 fit), yet a healthy
        // batch still rendered.
        assert!(
            fill_draws.len() < super::MAX_SHAPE_INSTANCES,
            "cap must drop over-capacity instances (emitted {})",
            fill_draws.len()
        );
        assert!(
            fill_draws.len() > 100,
            "expected a healthy batch of instances to fit (got {})",
            fill_draws.len()
        );
    }

    #[cfg(feature = "app")]
    #[test]
    fn packed_shape_and_wave_geometry_storage_is_reused_between_frames() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let shaders = crate::parse_milk::parse(
            "shapecode_0_enabled=1\nshapecode_0_num_inst=32\nshapecode_0_sides=20\nshapecode_0_a=1\nwavecode_0_enabled=1\nwavecode_0_samples=512\nwavecode_0_bUseDots=1\nwavecode_0_a=1\n",
        );
        let mut renderer = MilkdropRenderer::new(device, queue, 64, 64, fmt, &shaders)
            .expect("renderer with packed geometry");
        renderer.set_waveform(&[0.25; 512], &[-0.25; 512]);
        let target = offscreen_target(&renderer.device, 64, 64, fmt);

        renderer.render(&target);
        assert!(!renderer.scratch.shape_fill_verts.is_empty());
        assert!(!renderer.scratch.wave_verts.is_empty());
        assert!(!renderer.waves[0].scratch.output.is_empty());
        assert!(!renderer.scratch.basic_wave.processed_l.is_empty());
        assert!(!renderer.scratch.custom_wave_draws.is_empty());
        let shape_ptr = renderer.scratch.shape_fill_verts.as_ptr();
        let shape_capacity = renderer.scratch.shape_fill_verts.capacity();
        let wave_ptr = renderer.scratch.wave_verts.as_ptr();
        let wave_capacity = renderer.scratch.wave_verts.capacity();
        let custom_wave_ptr = renderer.waves[0].scratch.output.as_ptr();
        let custom_wave_capacity = renderer.waves[0].scratch.output.capacity();
        let basic_wave_ptr = renderer.scratch.basic_wave.processed_l.as_ptr();
        let basic_wave_capacity = renderer.scratch.basic_wave.processed_l.capacity();
        let custom_wave_draws_ptr = renderer.scratch.custom_wave_draws.as_ptr();
        let custom_wave_draws_capacity = renderer.scratch.custom_wave_draws.capacity();

        renderer.render(&target);
        assert_eq!(renderer.scratch.shape_fill_verts.as_ptr(), shape_ptr);
        assert_eq!(renderer.scratch.shape_fill_verts.capacity(), shape_capacity);
        assert_eq!(renderer.scratch.wave_verts.as_ptr(), wave_ptr);
        assert_eq!(renderer.scratch.wave_verts.capacity(), wave_capacity);
        assert_eq!(renderer.waves[0].scratch.output.as_ptr(), custom_wave_ptr);
        assert_eq!(
            renderer.waves[0].scratch.output.capacity(),
            custom_wave_capacity
        );
        assert_eq!(
            renderer.scratch.basic_wave.processed_l.as_ptr(),
            basic_wave_ptr
        );
        assert_eq!(
            renderer.scratch.basic_wave.processed_l.capacity(),
            basic_wave_capacity
        );
        assert_eq!(
            renderer.scratch.custom_wave_draws.as_ptr(),
            custom_wave_draws_ptr
        );
        assert_eq!(
            renderer.scratch.custom_wave_draws.capacity(),
            custom_wave_draws_capacity
        );
    }

    // ── Feedback starts black; resize preserves runtime without seeding noise ─────
    #[cfg(feature = "app")]
    #[test]
    fn resize_preserves_runtime_without_feedback_noise() {
        let Some((device, queue)) = gpu_device() else {
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let plain = crate::parse_milk::parse("");
        let mut r = MilkdropRenderer::new(device, queue, 64, 64, fmt, &plain).expect("renderer");
        assert_eq!(
            r.noise_regen_count(),
            0,
            "black feedback initialization must not generate seed noise"
        );

        // An in-place resize must not reset the live preset runtime. Rendering
        // advances the frame counter; resize preserves it, then the next render
        // continues at the following frame while using the new target dimensions.
        let first_target = offscreen_target(&r.device, 64, 64, fmt);
        r.render(&first_target);
        let frame_before_resize = r.frame_idx;
        assert!(frame_before_resize > 0);
        r.try_resize(96, 64).expect("valid state-preserving resize");
        assert_eq!(
            r.frame_idx, frame_before_resize,
            "resize must not restart time"
        );
        assert_eq!((r.width, r.height), (96, 64));
        let resized_target = offscreen_target(&r.device, 96, 64, fmt);
        r.render(&resized_target);
        assert_eq!(
            r.frame_idx,
            frame_before_resize + 1,
            "the same renderer must continue after its target resize"
        );

        // An interactive resize storm — grow, shrink, grow — must not inject
        // feedback noise. Newly exposed texels remain black until authored draws.
        for &(w, h) in &[
            (96, 96),
            (48, 48),
            (200, 120),
            (72, 72),
            (256, 144),
            (64, 64),
        ] {
            r.resize(w, h);
        }
        assert_eq!(
            r.noise_regen_count(),
            0,
            "resizes must preserve black feedback initialization"
        );
    }
}
