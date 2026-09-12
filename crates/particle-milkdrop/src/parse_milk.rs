// Parses raw MilkDrop .milk preset files.
//
// The warp and comp shader sections are stored as numbered lines:
//   warp_1=`shader_body
//   warp_2=`{
//   warp_3=`   float3 dx = ...
//   warp_N=`}
//
// Each line has the prefix `warp_N=`` (with literal backtick) and contributes
// one line of shader code. We reassemble them in order and strip the outer
// `shader_body { }` wrapper to get just the inner body code.

use std::collections::{BTreeMap, HashMap};

/// BeatDrop/MilkDrop addresses 16 custom waveforms and 16 custom shapes (slots
/// 0..=15). Keep the direct-file parser consistent with the importer path while
/// still rejecting sparse, attacker-controlled indices at or beyond that limit.
const MAX_CUSTOM_WAVES: u32 = 16;
const MAX_CUSTOM_SHAPES: u32 = 16;

pub struct MilkShaders {
    pub enhanced_audio: crate::enhanced_audio::EnhancedAudioConfig,
    pub beatdrop_feedback: bool,
    pub warp: Option<String>,
    pub comp: Option<String>,
    /// When true, `warp`/`comp` hold already-GLSL shader BODIES (Butterchurn
    /// converted-JSON presets), so the renderer compiles them via the GLSL-body
    /// path (glsl_milk_*_to_naga) instead of the HLSL→GLSL path. .milk presets
    /// leave this false (HLSL bodies). Additive — never set for the .milk path.
    pub shaders_glsl: bool,
    pub per_frame: Option<String>,
    /// Per-frame INIT EEL program (per_frame_init_N lines) — run once at load.
    pub per_frame_init: Option<String>,
    /// Per-vertex warp EEL program (per_pixel_N lines).
    pub per_pixel: Option<String>,
    /// 0.0–1.0; default 0.98
    pub decay: f32,
    /// brightness exponent; default 2.0
    pub gamma_adj: f32,
    /// Authored composite-shader hue weighting (`fShader`); default 0.0.
    pub fshader: f32,

    // ── Video echo (in-comp-shader feedback look) ─────────────────────────────
    pub echo_zoom: f32,   // fVideoEchoZoom,        default 2.0
    pub echo_alpha: f32,  // fVideoEchoAlpha,       default 0.0
    pub echo_orient: f32, // nVideoEchoOrientation, default 0.0

    // ── Comp post-FX flags ────────────────────────────────────────────────────
    pub brighten: bool, // bBrighten
    pub darken: bool,   // bDarken
    pub solarize: bool, // bSolarize
    pub invert: bool,   // bInvert

    // ── Per-frame warp base scalars (overridable by the per-frame EEL program) ──
    pub warpscale: f32,     // fWarpScale,     default 1.0
    pub warpanimspeed: f32, // fWarpAnimSpeed, default 1.0
    pub zoom: f32,          // default 1.0
    pub zoomexp: f32,       // default 1.0
    pub rot: f32,           // default 0.0
    pub warp_amount: f32, // `warp` scalar, default 1.0 (renamed to avoid colliding with the warp shader field)
    pub cx: f32,          // default 0.5
    pub cy: f32,          // default 0.5
    pub dx: f32,          // default 0.0
    pub dy: f32,          // default 0.0
    pub sx: f32,          // default 1.0
    pub sy: f32,          // default 1.0
    pub wrap: bool,       // bTexWrap, default true

    // ── Built-in waveform scalars/bools ──────────────────────────────────────
    pub wave_mode: f32,             // nWaveMode
    pub wave_x: f32,                // wave_x
    pub wave_y: f32,                // wave_y
    pub wave_r: f32,                // wave_r
    pub wave_g: f32,                // wave_g
    pub wave_b: f32,                // wave_b
    pub wave_a: f32,                // fWaveAlpha (base alpha)
    pub wave_mystery: f32,          // fWaveParam
    pub wave_scale: f32,            // fWaveScale
    pub wave_smoothing: f32,        // fWaveSmoothing
    pub wave_dots: bool,            // bWaveDots
    pub wave_thick: bool,           // bWaveThick
    pub additive_wave: bool,        // bAdditiveWaves
    pub wave_brighten: bool,        // bMaximizeWaveColor
    pub modwavealphabyvolume: bool, // bModWaveAlphaByVolume
    pub modwavealphastart: f32,     // fModWaveAlphaStart
    pub modwavealphaend: f32,       // fModWaveAlphaEnd

    // ── Motion vectors (butterchurn runtime defaults: ON, mv_l 0.9, mv_a 1) ───
    pub mv_on: bool, // bMotionVectorsOn (DEFAULT true — render default frame)
    pub mv_x: f32,   // nMotionVectorsX, default 12
    pub mv_y: f32,   // nMotionVectorsY, default 9
    pub mv_dx: f32,  // mv_dx, default 0
    pub mv_dy: f32,  // mv_dy, default 0
    pub mv_l: f32,   // mv_l,  default 0.9
    pub mv_r: f32,   // mv_r,  default 1
    pub mv_g: f32,   // mv_g,  default 1
    pub mv_b: f32,   // mv_b,  default 1
    pub mv_a: f32,   // mv_a,  default 1

    // ── Borders (outer/inner colored frame) ──────────────────────────────────
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

    // ── Darken center ─────────────────────────────────────────────────────────
    pub darken_center: bool, // bDarkenCenter

    // ── Blur min/max (per-level range remap; butterchurn b1n/b1x …) ───────────
    pub b1n: f32,
    pub b1x: f32, // blur1 min/max (default 0 / 1)
    pub b1ed: f32,
    pub b2n: f32,
    pub b2x: f32, // blur2 min/max
    pub b3n: f32,
    pub b3x: f32, // blur3 min/max

    // ── Custom shapes (up to 16) ─────────────────────────────────────────────
    pub shapes: Vec<ShapeCode>,
    // ── Custom waveforms (up to 16) ──────────────────────────────────────────
    pub waves: Vec<CustomWaveDef>,
}

// ── Custom shape definitions ─────────────────────────────────────────────────

#[derive(Clone)]
pub struct ShapeBaseVals {
    pub enabled: i32,
    pub sides: f32,
    pub additive: i32,
    pub thick_outline: i32,
    pub textured: i32,
    pub num_inst: i32,
    pub x: f32,
    pub y: f32,
    pub rad: f32,
    pub ang: f32,
    pub tex_ang: f32,
    pub tex_zoom: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub r2: f32,
    pub g2: f32,
    pub b2: f32,
    pub a2: f32,
    pub border_r: f32,
    pub border_g: f32,
    pub border_b: f32,
    pub border_a: f32,
}

impl Default for ShapeBaseVals {
    fn default() -> Self {
        // Mirrors butterchurn shapeBaseValsDefaults (butterchurn.js ~12062).
        ShapeBaseVals {
            enabled: 0,
            sides: 4.0,
            additive: 0,
            thick_outline: 0,
            textured: 0,
            num_inst: 1,
            x: 0.5,
            y: 0.5,
            rad: 0.1,
            ang: 0.0,
            tex_ang: 0.0,
            tex_zoom: 1.0,
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
            r2: 0.0,
            g2: 1.0,
            b2: 0.0,
            a2: 0.0,
            border_r: 1.0,
            border_g: 1.0,
            border_b: 1.0,
            border_a: 0.1,
        }
    }
}

pub struct ShapeCode {
    pub base: ShapeBaseVals,
    pub per_frame: Option<String>,
    /// shape_N_per_frame_init lines — run once into the shape env at load.
    pub per_frame_init: Option<String>,
}

// ── Custom waveform definitions ──────────────────────────────────────────────

pub struct CustomWaveDef {
    pub index: u32,
    pub enabled: bool,
    pub samples: u32,
    pub sep: i32,
    pub spectrum: bool,
    pub use_dots: bool,
    pub draw_thick: bool,
    pub additive: bool,
    pub scaling: f32,
    pub smoothing: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub per_frame: Option<String>,
    /// wave_N_per_frame_init_M lines — run once into the wave env at load.
    pub per_frame_init: Option<String>,
    pub per_point: Option<String>,
}

impl Default for CustomWaveDef {
    fn default() -> Self {
        CustomWaveDef {
            index: 0,
            enabled: false,
            samples: 512,
            sep: 0,
            spectrum: false,
            use_dots: false,
            draw_thick: false,
            additive: false,
            scaling: 1.0,
            smoothing: 0.5,
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
            per_frame: None,
            per_frame_init: None,
            per_point: None,
        }
    }
}

pub fn parse(content: &str) -> MilkShaders {
    MilkShaders {
        enhanced_audio: crate::enhanced_audio::EnhancedAudioConfig {
            fft_attack: parse_float(content, "FFTAttack", 0.5),
            fft_decay: parse_float(content, "FFTDecay", 0.7),
            ..Default::default()
        }.sanitized(),
        beatdrop_feedback: parse_bool(content, "bOjoBeatDropFeedback", false),
        warp: extract_section(content, "warp_"),
        comp: extract_section(content, "comp_"),
        shaders_glsl: false, // .milk path = HLSL bodies (unchanged behavior).
        per_frame: extract_per_frame(content),
        per_frame_init: extract_per_frame_init(content),
        per_pixel: extract_per_pixel(content),
        decay: parse_float(content, "fDecay", 0.98),
        gamma_adj: parse_float(content, "fGammaAdj", 2.0),
        fshader: parse_float(content, "fShader", 0.0),

        echo_zoom: parse_float(content, "fVideoEchoZoom", 2.0),
        echo_alpha: parse_float(content, "fVideoEchoAlpha", 0.0),
        echo_orient: parse_float(content, "nVideoEchoOrientation", 0.0),

        brighten: parse_bool(content, "bBrighten", false),
        darken: parse_bool(content, "bDarken", false),
        solarize: parse_bool(content, "bSolarize", false),
        invert: parse_bool(content, "bInvert", false),

        warpscale: parse_float(content, "fWarpScale", 1.0),
        warpanimspeed: parse_float(content, "fWarpAnimSpeed", 1.0),
        zoom: parse_float(content, "zoom", 1.0),
        zoomexp: parse_float(content, "zoomexp", 1.0),
        rot: parse_float(content, "rot", 0.0),
        warp_amount: parse_float(content, "warp", 1.0),
        cx: parse_float(content, "cx", 0.5),
        cy: parse_float(content, "cy", 0.5),
        dx: parse_float(content, "dx", 0.0),
        dy: parse_float(content, "dy", 0.0),
        sx: parse_float(content, "sx", 1.0),
        sy: parse_float(content, "sy", 1.0),
        wrap: parse_bool(content, "bTexWrap", true),

        wave_mode: parse_float(content, "nWaveMode", 0.0),
        wave_x: parse_float(content, "wave_x", 0.5),
        wave_y: parse_float(content, "wave_y", 0.5),
        wave_r: parse_float(content, "wave_r", 1.0),
        wave_g: parse_float(content, "wave_g", 1.0),
        wave_b: parse_float(content, "wave_b", 1.0),
        wave_a: parse_float(content, "fWaveAlpha", 1.0),
        wave_mystery: parse_float(content, "fWaveParam", 0.0),
        wave_scale: parse_float(content, "fWaveScale", 1.0),
        wave_smoothing: parse_float(content, "fWaveSmoothing", 0.75),
        wave_dots: parse_bool(content, "bWaveDots", false),
        wave_thick: parse_bool(content, "bWaveThick", false),
        additive_wave: parse_bool(content, "bAdditiveWaves", false),
        wave_brighten: parse_bool(content, "bMaximizeWaveColor", true),
        modwavealphabyvolume: parse_bool(content, "bModWaveAlphaByVolume", false),
        modwavealphastart: parse_float(content, "fModWaveAlphaStart", 0.75),
        modwavealphaend: parse_float(content, "fModWaveAlphaEnd", 0.95),

        // Motion vectors. butterchurn RENDER default frame (visualizer.js): ON,
        // mv_x 12, mv_y 9, mv_l 0.9, mv_a 1 (NOT the blankPreset parse-fallbacks).
        mv_on: parse_bool(content, "bMotionVectorsOn", true),
        mv_x: parse_float(content, "nMotionVectorsX", 12.0),
        mv_y: parse_float(content, "nMotionVectorsY", 9.0),
        mv_dx: parse_float(content, "mv_dx", 0.0),
        mv_dy: parse_float(content, "mv_dy", 0.0),
        mv_l: parse_float(content, "mv_l", 0.9),
        mv_r: parse_float(content, "mv_r", 1.0),
        mv_g: parse_float(content, "mv_g", 1.0),
        mv_b: parse_float(content, "mv_b", 1.0),
        mv_a: parse_float(content, "mv_a", 1.0),

        // Borders (visualizer.js defaults: ob_size/ib_size 0.01, ob 0, ib 0.25, a 0).
        ob_size: parse_float(content, "ob_size", 0.01),
        ob_r: parse_float(content, "ob_r", 0.0),
        ob_g: parse_float(content, "ob_g", 0.0),
        ob_b: parse_float(content, "ob_b", 0.0),
        ob_a: parse_float(content, "ob_a", 0.0),
        ib_size: parse_float(content, "ib_size", 0.01),
        ib_r: parse_float(content, "ib_r", 0.25),
        ib_g: parse_float(content, "ib_g", 0.25),
        ib_b: parse_float(content, "ib_b", 0.25),
        ib_a: parse_float(content, "ib_a", 0.0),

        // Darken center (default off).
        darken_center: parse_bool(content, "bDarkenCenter", false),

        // Blur min/max (butterchurn visualizer.js defaults: min 0, max 1 = identity).
        b1n: parse_float(content, "b1n", 0.0),
        b1x: parse_float(content, "b1x", 1.0),
        b1ed: parse_float(content, "b1ed", 0.25),
        b2n: parse_float(content, "b2n", 0.0),
        b2x: parse_float(content, "b2x", 1.0),
        b3n: parse_float(content, "b3n", 0.0),
        b3x: parse_float(content, "b3x", 1.0),

        shapes: parse_shapes(content),
        waves: parse_waves(content),
    }
}

fn parse_bool(content: &str, key: &str, default: bool) -> bool {
    // Bool keys are stored as 0/1 floats in .milk.
    let v = parse_float(content, key, if default { 1.0 } else { 0.0 });
    v != 0.0
}

fn parse_float(content: &str, key: &str, default: f32) -> f32 {
    for line in content.lines() {
        // Case-insensitive key match: fDecay= or fdecay= etc.
        let lc = line.to_ascii_lowercase();
        let key_lc = key.to_ascii_lowercase();
        if let Some(rest) = lc.strip_prefix(&key_lc) {
            if let Some(val) = rest.strip_prefix('=') {
                if let Some(v) = parse_finite_f32(val, key) {
                    return v;
                }
            }
        }
    }
    default
}

/// Parse a numeric field value from raw `.milk` text, rejecting non-finite results.
///
/// Rust's `f32` parser accepts `inf`/`-inf`/`nan` and silently overflows an
/// out-of-range exponent (`1e999`) to `±inf`. A raw `.milk` is untrusted input, and
/// such a poisoned value must never flow downstream into the renderer's uniforms
/// (an `inf`/`nan` in a transform or color scalar corrupts the whole frame). Returns
/// `None` for unparseable OR non-finite input; the caller then keeps its default.
/// A present-but-non-finite value is logged as a diagnostic; ordinary non-numeric
/// text is skipped silently (the scan simply moves on).
fn parse_finite_f32(raw: &str, key: &str) -> Option<f32> {
    let s = raw.trim();
    match s.parse::<f32>() {
        Ok(v) if v.is_finite() => Some(v),
        Ok(_) => {
            log::warn!("ignoring non-finite value `{s}` for `{key}` in .milk preset");
            None
        }
        Err(_) => None,
    }
}

fn extract_per_frame(content: &str) -> Option<String> {
    let prefix = "per_frame_";
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();
    for line in content.lines() {
        if !line.starts_with(prefix) {
            continue;
        }
        // Skip per_frame_init_N lines (they share the per_frame_ prefix) — they
        // are collected separately by extract_per_frame_init. Without this guard
        // the `n.parse()` below failed on "init_1" and silently aborted the whole
        // per-frame collection.
        let rest = &line[prefix.len()..];
        if rest.starts_with("init_") {
            continue;
        }
        let eq = match rest.find('=') {
            Some(e) => e,
            None => continue,
        };
        let n: u32 = match rest[..eq].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        lines.insert(n, rest[eq + 1..].to_string());
    }
    if lines.is_empty() {
        return None;
    }
    Some(lines.values().cloned().collect::<Vec<_>>().join("\n"))
}

/// Per-frame INIT equations: `per_frame_init_N=<code>`. Run ONCE at preset load
/// (before frame 0) so per-frame equations see initialized vars (q1-q32, regs…).
fn extract_per_frame_init(content: &str) -> Option<String> {
    collect_numbered_eel(content, "per_frame_init_")
}

/// Per-vertex warp equations: `per_pixel_N=<code>`. Mirror of extract_per_frame.
fn extract_per_pixel(content: &str) -> Option<String> {
    let prefix = "per_pixel_";
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();
    for line in content.lines() {
        if !line.starts_with(prefix) {
            continue;
        }
        let rest = &line[prefix.len()..];
        // Skip a single off-spec line instead of `?`-aborting the whole collection
        // (which would silently discard every already-parsed line in this section).
        let Some(eq) = rest.find('=') else { continue };
        let Ok(n) = rest[..eq].parse::<u32>() else {
            continue;
        };
        lines.insert(n, rest[eq + 1..].to_string());
    }
    if lines.is_empty() {
        return None;
    }
    Some(lines.values().cloned().collect::<Vec<_>>().join("\n"))
}

/// Collect plain-EEL numbered lines of the form `<prefix><N>=<code>` (N ascending),
/// joined by '\n'. Used for shape_N_per_frame, wave_N_per_frame, wave_N_per_point.
/// Returns None if no matching line was present.
fn collect_numbered_eel(content: &str, prefix: &str) -> Option<String> {
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();
    for line in content.lines() {
        if !line.starts_with(prefix) {
            continue;
        }
        let rest = &line[prefix.len()..];
        // Custom wave/shape equations exist in both `per_frame1` and
        // `per_frame_1` spellings across the corpus. Callers whose prefix
        // already ends in `_` are unaffected; otherwise accept the optional
        // separator before the numeric index.
        let rest = rest.strip_prefix('_').unwrap_or(rest);
        // rest is like "3=code" (the trailing index then '=')
        let eq = match rest.find('=') {
            Some(e) => e,
            None => continue,
        };
        let n: u32 = match rest[..eq].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        lines.insert(n, rest[eq + 1..].to_string());
    }
    if lines.is_empty() {
        return None;
    }
    Some(lines.values().cloned().collect::<Vec<_>>().join("\n"))
}

fn combine_eel_blocks(blocks: &[Option<String>]) -> Option<String> {
    let lines = blocks
        .iter()
        .filter_map(|b| b.as_deref())
        .filter(|b| !b.trim().is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

/// Collect `<prefix><N>=code` numbered lines but SKIP any whose remainder begins
/// with `_init` (so `shape_0_per_frame_init_*` lines aren't swept into the regular
/// `shape_0_per_frame*` collection). Mirrors collect_numbered_eel otherwise.
fn collect_per_frame_no_init(content: &str, prefix: &str) -> Option<String> {
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();
    for line in content.lines() {
        if !line.starts_with(prefix) {
            continue;
        }
        let rest = &line[prefix.len()..];
        if rest.starts_with("_init") {
            continue;
        }
        let rest = rest.strip_prefix('_').unwrap_or(rest);
        let eq = match rest.find('=') {
            Some(e) => e,
            None => continue,
        };
        let n: u32 = match rest[..eq].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        lines.insert(n, rest[eq + 1..].to_string());
    }
    if lines.is_empty() {
        return None;
    }
    Some(lines.values().cloned().collect::<Vec<_>>().join("\n"))
}

/// Single pass over `content`: bucket every `<family><slot>_<field>=<value>`
/// scalar line into `slot → (field → value)`. Case-insensitive on the whole key
/// (MilkDrop writes mixed case). Slots at or beyond `max_slots` are dropped, so a
/// sparse attacker-controlled high index costs nothing. First occurrence of a
/// field wins (matching the previous first-match `parse_key_opt`). A slot is
/// `Some` iff at least one `<family><slot>_…` line was present, reproducing the old
/// presence gate. This replaces the previous O(slots × fields × lines) brute-force
/// rescans (one full `content.lines()` scan per candidate key per slot) with a
/// single O(lines) pass.
fn collect_indexed_scalars(
    content: &str,
    family: &str,
    max_slots: u32,
) -> Vec<Option<HashMap<String, f32>>> {
    let mut slots: Vec<Option<HashMap<String, f32>>> = (0..max_slots).map(|_| None).collect();
    let family_lc = family.to_ascii_lowercase();
    for line in content.lines() {
        let lc = line.to_ascii_lowercase();
        let Some(rest) = lc.strip_prefix(&family_lc) else {
            continue;
        };
        // rest is "<slot>_<field>=<value>"; split the slot on the FIRST '_' so a
        // field that itself contains '_' (num_inst, tex_ang, border_r…) is intact.
        let Some(us) = rest.find('_') else { continue };
        let Ok(slot) = rest[..us].parse::<u32>() else {
            continue;
        };
        if slot >= max_slots {
            continue; // enforce the real MilkDrop slot limit
        }
        let map = slots[slot as usize].get_or_insert_with(HashMap::new);
        let after = &rest[us + 1..];
        let Some(eq) = after.find('=') else { continue };
        let field = &after[..eq];
        if let Some(v) = parse_finite_f32(&after[eq + 1..], field) {
            map.entry(field.to_string()).or_insert(v); // first-match wins
        }
    }
    slots
}

fn parse_shapes(content: &str) -> Vec<ShapeCode> {
    let slots = collect_indexed_scalars(content, "shapecode_", MAX_CUSTOM_SHAPES);
    let mut out = Vec::new();
    for (i, fields) in slots.into_iter().enumerate() {
        let Some(fields) = fields else { continue };
        let i = i as u32;

        let mut base = ShapeBaseVals::default();
        // Field names are lowercased in the slot map; lowercase the lookup to keep
        // the previous case-insensitive matching.
        let g = |field: &str| fields.get(&field.to_ascii_lowercase()).copied();
        if let Some(v) = g("enabled") {
            base.enabled = v as i32;
        }
        if let Some(v) = g("sides") {
            base.sides = v;
        }
        if let Some(v) = g("additive") {
            base.additive = v as i32;
        }
        if let Some(v) = g("thickOutline") {
            base.thick_outline = v as i32;
        }
        if let Some(v) = g("textured") {
            base.textured = v as i32;
        }
        if let Some(v) = g("num_inst") {
            base.num_inst = v as i32;
        }
        if let Some(v) = g("x") {
            base.x = v;
        }
        if let Some(v) = g("y") {
            base.y = v;
        }
        if let Some(v) = g("rad") {
            base.rad = v;
        }
        if let Some(v) = g("ang") {
            base.ang = v;
        }
        if let Some(v) = g("tex_ang") {
            base.tex_ang = v;
        }
        if let Some(v) = g("tex_zoom") {
            base.tex_zoom = v;
        }
        if let Some(v) = g("r") {
            base.r = v;
        }
        if let Some(v) = g("g") {
            base.g = v;
        }
        if let Some(v) = g("b") {
            base.b = v;
        }
        if let Some(v) = g("a") {
            base.a = v;
        }
        if let Some(v) = g("r2") {
            base.r2 = v;
        }
        if let Some(v) = g("g2") {
            base.g2 = v;
        }
        if let Some(v) = g("b2") {
            base.b2 = v;
        }
        if let Some(v) = g("a2") {
            base.a2 = v;
        }
        if let Some(v) = g("border_r") {
            base.border_r = v;
        }
        if let Some(v) = g("border_g") {
            base.border_g = v;
        }
        if let Some(v) = g("border_b") {
            base.border_b = v;
        }
        if let Some(v) = g("border_a") {
            base.border_a = v;
        }

        // MilkDrop presets use both `shape_N_per_frame_init_M=` and raw
        // `shape_N_initM=` spellings. Both run once into the shape env at load.
        let per_frame_init = combine_eel_blocks(&[
            collect_numbered_eel(content, &format!("shape_{i}_per_frame_init_")),
            collect_numbered_eel(content, &format!("shape_{i}_init")),
        ]);
        // Regular per-frame lines use `shape_N_per_frameM=` (no separator).
        let per_frame = collect_per_frame_no_init(content, &format!("shape_{i}_per_frame"));
        out.push(ShapeCode {
            base,
            per_frame,
            per_frame_init,
        });
    }
    out
}

fn parse_waves(content: &str) -> Vec<CustomWaveDef> {
    let slots = collect_indexed_scalars(content, "wavecode_", MAX_CUSTOM_WAVES);
    let mut out = Vec::new();
    for (n, fields) in slots.into_iter().enumerate() {
        let Some(fields) = fields else { continue };
        let n = n as u32;

        let mut w = CustomWaveDef {
            index: n,
            ..Default::default()
        };
        // Field names are lowercased in the slot map; lowercase the lookup to keep
        // the previous case-insensitive matching.
        let g = |field: &str| fields.get(&field.to_ascii_lowercase()).copied();
        if let Some(v) = g("enabled") {
            w.enabled = v != 0.0;
        }
        if let Some(v) = g("samples") {
            w.samples = v.max(0.0) as u32;
        }
        if let Some(v) = g("sep") {
            w.sep = v as i32;
        }
        if let Some(v) = g("bSpectrum") {
            w.spectrum = v != 0.0;
        }
        if let Some(v) = g("bUseDots") {
            w.use_dots = v != 0.0;
        }
        if let Some(v) = g("bDrawThick") {
            w.draw_thick = v != 0.0;
        }
        if let Some(v) = g("bAdditive") {
            w.additive = v != 0.0;
        }
        if let Some(v) = g("scaling") {
            w.scaling = v;
        }
        if let Some(v) = g("smoothing") {
            w.smoothing = v;
        }
        if let Some(v) = g("r") {
            w.r = v;
        }
        if let Some(v) = g("g") {
            w.g = v;
        }
        if let Some(v) = g("b") {
            w.b = v;
        }
        if let Some(v) = g("a") {
            w.a = v;
        }

        // MilkDrop presets use both `wave_N_per_frame_init_M=` and raw
        // `wave_N_initM=` spellings. Both run once into the wave env at load.
        w.per_frame_init = combine_eel_blocks(&[
            collect_numbered_eel(content, &format!("wave_{n}_per_frame_init_")),
            collect_numbered_eel(content, &format!("wave_{n}_init")),
        ]);
        w.per_frame = collect_per_frame_no_init(content, &format!("wave_{n}_per_frame"));
        w.per_point = collect_numbered_eel(content, &format!("wave_{n}_per_point"));
        out.push(w);
    }
    out
}

fn extract_section(content: &str, prefix: &str) -> Option<String> {
    // Collect lines: `warp_N=`` or `comp_N=``
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();

    for line in content.lines() {
        // Line format: `warp_42=`actual code here`  (backtick immediately after `=`)
        if !line.starts_with(prefix) {
            continue;
        }
        let rest = &line[prefix.len()..];
        // rest is like "42=`code"
        // Skip a single off-spec line instead of `?`-aborting the whole collection
        // (which would silently discard every already-parsed warp_/comp_ line).
        let Some(eq) = rest.find('=') else { continue };
        let Ok(n) = rest[..eq].parse::<u32>() else {
            continue;
        };
        let after_eq = &rest[eq + 1..];
        // Strip the leading backtick
        let code = after_eq.strip_prefix('`').unwrap_or(after_eq);
        lines.insert(n, code.to_string());
    }

    if lines.is_empty() {
        return None;
    }

    // Reassemble
    let raw: String = lines.values().cloned().collect::<Vec<_>>().join("\n");
    let trimmed = raw.trim();

    // Strip the `shader_body { … }` wrapper. The `shader_body` token is usually the
    // first line, but sampler/#define/global declarations can precede it. Locate the
    // token wherever it sits, brace-match its open/close to extract the inner body,
    // and keep any pre-`shader_body` globals so the downstream HLSL splitter still
    // sees them at file scope. The old prefix/suffix strip silently no-oped the
    // prefix strips when globals were present but still chopped the trailing `}`,
    // leaving `shader_body` and `{` embedded with the closing brace gone.
    //
    // Uses the shared comment-/string-/token-aware scanner (see preprocess) so a
    // `shader_body` substring inside a comment, string, or larger identifier — or a
    // brace inside a block comment — never confuses the wrapper extraction.
    match crate::preprocess::find_shader_body_keyword(trimmed) {
        Some(pos) => {
            let before = trimmed[..pos].trim_end();
            let after_kw = trimmed[pos + "shader_body".len()..].trim_start();
            let inner = match crate::preprocess::find_code_byte(after_kw, b'{') {
                Some(open) => {
                    let body_src = &after_kw[open + 1..];
                    let end = crate::preprocess::scan_to_matching_brace(body_src);
                    body_src[..end].trim().to_string()
                }
                // No `{` after the token — nothing sane to strip; keep as-is.
                None => after_kw.to_string(),
            };
            if before.is_empty() {
                Some(inner)
            } else {
                // Re-emit the canonical wrapper so the HLSL path's
                // split_shader_body_wrapper still separates globals from body.
                Some(format!("{before}\nshader_body {{\n{inner}\n}}"))
            }
        }
        None => Some(trimmed.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── P2-VIS-013: bounded, single-pass indexed-key parsing ─────────────────

    #[test]
    fn wave_slot_index_beyond_limit_is_ignored() {
        // Slot 0 and slot 15 are in range; slot 16 exceeds MAX_CUSTOM_WAVES and must be
        // dropped, so an attacker-controlled sparse high index never allocates a
        // slot. The whole preset is bucketed in a single O(lines) pass.
        let content = "\
wavecode_0_enabled=1
wavecode_0_sep=120
wavecode_15_enabled=1
wavecode_15_sep=99
wavecode_16_enabled=1
wavecode_16_sep=1
";
        let waves = parse_waves(content);
        assert_eq!(waves.len(), 2, "only the 16 in-range slots parse");
        assert_eq!(waves[0].index, 0);
        assert!(waves[0].enabled);
        assert_eq!(waves[1].index, 15);
        assert_eq!(waves[1].sep, 99);
    }

    #[test]
    fn shape_slot_index_beyond_limit_is_ignored() {
        let content = "shapecode_0_enabled=1\nshapecode_15_enabled=1\nshapecode_16_enabled=1\n";
        let shapes = parse_shapes(content);
        assert_eq!(shapes.len(), 2);
    }

    #[test]
    fn sparse_in_range_wave_slot_parses() {
        // Only slot 2 present (0/1 absent) → exactly one wave at index 2.
        let content = "wavecode_2_enabled=1\nwavecode_2_samples=64\n";
        let waves = parse_waves(content);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].index, 2);
        assert_eq!(waves[0].samples, 64);
    }

    #[test]
    fn field_name_with_underscore_survives_slot_split() {
        // The slot is split on the FIRST '_'; a field that itself contains '_'
        // (num_inst here) must stay intact.
        let content = "shapecode_0_enabled=1\nshapecode_0_num_inst=5\n";
        let shapes = parse_shapes(content);
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].base.num_inst, 5);
    }

    // ── P2-VIS-011: numeric custom-wave sep preserved through this crate ─────

    #[test]
    fn custom_wave_sep_120_survives_parse() {
        // A numeric sep of 120 must NOT collapse to a boolean 1. The milkdrop
        // crate's schema keeps `sep` as i32 and parse preserves it end-to-end
        // through this crate's own path (.milk → MilkShaders → renderer runtime).
        let content = "wavecode_0_enabled=1\nwavecode_0_sep=120\n";
        let waves = parse_waves(content);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].sep, 120, "numeric sep was coerced (expected 120)");

        // …and through the public parse() entry point.
        let shaders = parse(content);
        assert_eq!(shaders.waves[0].sep, 120);
    }

    #[test]
    fn fshader_parses_case_insensitively_and_defaults_to_zero() {
        assert_eq!(parse("").fshader, 0.0);
        assert!((parse("FsHaDeR=0.600\n").fshader - 0.6).abs() < 1.0e-6);
    }

    // ── P1-044: finite validation on raw .milk numeric ingress ───────────────

    #[test]
    fn non_finite_numeric_fields_are_rejected_to_default() {
        // `1e999` overflows f32 to +inf; `nan` / `-inf` parse directly. None of
        // these may reach the renderer's uniforms — each scalar key falls back to
        // its field default instead of flowing downstream as a non-finite value.
        let overflow = parse("zoom=1e999\n");
        assert!(overflow.zoom.is_finite(), "1e999 must not survive as inf");
        assert_eq!(
            overflow.zoom, 1.0,
            "overflow falls back to the zoom default"
        );

        let nan = parse("fDecay=nan\n");
        assert!(nan.decay.is_finite());
        assert_eq!(nan.decay, 0.98, "nan falls back to the fDecay default");

        let neg_inf = parse("rot=-inf\n");
        assert!(neg_inf.rot.is_finite());
        assert_eq!(neg_inf.rot, 0.0, "-inf falls back to the rot default");

        // Indexed-scalar ingress (shape/wave fields) rejects non-finite too: an
        // inf shape field is dropped, leaving that field at its default.
        let shape = parse("shapecode_0_enabled=1\nshapecode_0_x=1e999\n");
        assert_eq!(shape.shapes.len(), 1);
        assert!(
            shape.shapes[0].base.x.is_finite(),
            "inf shape field must be rejected"
        );
        assert_eq!(shape.shapes[0].base.x, 0.5, "rejected → shape x default");

        // A finite value in the same fields still parses (no false positives).
        let ok = parse("zoom=1.5\nshapecode_0_enabled=1\nshapecode_0_x=0.25\n");
        assert_eq!(ok.zoom, 1.5);
        assert_eq!(ok.shapes[0].base.x, 0.25);
    }

    // NaN/Inf boundary for .milk input: the existing test uses
    // `1e999`, which overflows f64 as well as f32 and would be caught by
    // almost any check. The subtler hostile input is a value that is
    // perfectly finite as an f64 but overflows the f32 the field is
    // stored in — `1e39` sits just past f32::MAX (~3.40e38). It must hit
    // the same guard, and the boundary must be finiteness rather than
    // magnitude: the largest representable f32 has to survive intact.
    #[test]
    fn f32_range_overflow_and_finite_extremes_are_distinguished() {
        // Overflows f32 (but not f64) → rejected, field keeps its default.
        let over_f32 = parse("zoom=1e39\n");
        assert!(
            over_f32.zoom.is_finite(),
            "1e39 overflows f32 and must not survive as inf"
        );
        assert_eq!(
            over_f32.zoom, 1.0,
            "f32 overflow falls back to zoom default"
        );

        // Just inside the f32 range → preserved, not rejected.
        let near_max = parse("zoom=3.0e38\n");
        assert!(near_max.zoom.is_finite());
        assert_eq!(
            near_max.zoom, 3.0e38,
            "a value inside the f32 range must be preserved verbatim"
        );

        // Subnormal edge → preserved (it parses finite).
        let subnormal = parse("zoom=1e-45\n");
        assert!(subnormal.zoom.is_finite());
        assert_eq!(subnormal.zoom, 1e-45f32, "a subnormal must be preserved");

        // Rust's float parser accepts several spellings of the poisoned
        // values, case-insensitively and with a sign. Every one of them
        // must fall back to the default rather than reach the renderer.
        for spelling in [
            "inf",
            "+inf",
            "-inf",
            "INF",
            "Inf",
            "infinity",
            "Infinity",
            "-Infinity",
            "nan",
            "NaN",
            "NAN",
            "-nan",
        ] {
            let parsed = parse(&format!("zoom={spelling}\n"));
            assert!(
                parsed.zoom.is_finite(),
                "`{spelling}` must not reach the renderer as a non-finite zoom"
            );
            assert_eq!(
                parsed.zoom, 1.0,
                "`{spelling}` must fall back to the zoom default"
            );
        }

        // Indexed-scalar ingress takes the same f32-overflow path.
        let shape = parse("shapecode_0_enabled=1\nshapecode_0_x=1e39\n");
        assert_eq!(shape.shapes.len(), 1);
        assert_eq!(
            shape.shapes[0].base.x, 0.5,
            "an f32-overflowing shape field falls back to its default"
        );
    }
}
