// Loads Butterchurn converted-JSON presets (the butterchurn-presets format) into
// our `MilkShaders` so they render alongside raw `.milk` presets.
//
// The JSON format (see src/test_data/sherwin_maxawow.json):
//   { "baseVals": { gammaadj, decay, echo_zoom, ..., warp, wave_r/g/b, ob_*/ib_*, mv_*, ... },
//     "shapes": [...], "waves": [...],
//     "init_eqs_str", "frame_eqs_str", "pixel_eqs_str",   // JS-transpiled equations
//     "warp", "comp" }                                    // Butterchurn GLSL shader BODIES
//
// baseVals keys mostly MATCH our MilkShaders fields already; we apply the SAME
// defaults as parse_milk for absent keys. The *_eqs_str are JS — we convert them
// to our EEL (strip `a.` prefixes, strip `Math.`). The warp/comp are GLSL bodies
// flagged via shaders_glsl=true so the renderer compiles them through the GLSL-body
// path (preprocess::glsl_milk_*_to_naga).

use serde::Deserialize;
use std::collections::HashMap;

use crate::parse_milk::{CustomWaveDef, MilkShaders, ShapeBaseVals, ShapeCode};

#[derive(Deserialize)]
struct JsonPreset {
    #[serde(default)]
    #[serde(rename = "baseVals")]
    base_vals: HashMap<String, f64>,
    #[serde(default)]
    shapes: Vec<JsonSubObj>,
    #[serde(default)]
    waves: Vec<JsonSubObj>,
    #[serde(default)]
    init_eqs_str: Option<String>,
    #[serde(default)]
    frame_eqs_str: Option<String>,
    #[serde(default)]
    pixel_eqs_str: Option<String>,
    // The native converter (milkdrop-preset-converter-node) ALSO emits the raw
    // MilkDrop EEL source (`loop()`, `gmegabuf`, bare intrinsics) — our evaluator
    // parses that directly, sidestepping the JS dialect (`a['k']`, `Math.`, `for`)
    // that js_to_eel cannot translate. Prefer these when present.
    #[serde(default)]
    init_eqs_eel: Option<String>,
    #[serde(default)]
    frame_eqs_eel: Option<String>,
    #[serde(default)]
    pixel_eqs_eel: Option<String>,
    #[serde(default)]
    warp: Option<String>,
    #[serde(default)]
    comp: Option<String>,
}

#[derive(Deserialize)]
struct JsonSubObj {
    #[serde(default)]
    #[serde(rename = "baseVals")]
    base_vals: HashMap<String, f64>,
    // Shapes/waves may also carry their own *_eqs_str; we convert them when present.
    #[serde(default)]
    init_eqs_str: Option<String>,
    #[serde(default)]
    frame_eqs_str: Option<String>,
    #[serde(default)]
    point_eqs_str: Option<String>,
    // Raw-EEL variants (native converter) — preferred over the JS `*_str`.
    #[serde(default)]
    init_eqs_eel: Option<String>,
    #[serde(default)]
    frame_eqs_eel: Option<String>,
    #[serde(default)]
    point_eqs_eel: Option<String>,
}

/// Parse a Butterchurn converted-JSON preset string into MilkShaders.
pub fn load(content: &str) -> Result<MilkShaders, String> {
    let p: JsonPreset =
        serde_json::from_str(content).map_err(|e| format!("JSON parse error: {e}"))?;

    let bv = &p.base_vals;
    // Lookup with the SAME defaults as parse_milk for absent keys. Butterchurn JSON
    // keys are lowercase (gammaadj, decay, …); match case-insensitively to be safe.
    let lc: HashMap<String, f64> = bv
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), *v))
        .collect();
    // Finiteness is enforced here, matching `parse_milk::parse_finite_f32`: a
    // present-but-non-finite value keeps the field's default and is logged.
    //
    // The test is deliberately AFTER the narrowing cast, and that is the only
    // placement that does anything. serde_json rejects an out-of-`f64`-range
    // literal at the parse stage ("number out of range"), so the `f64` arriving
    // here is always finite and a pre-cast check could never fire. The single
    // remaining overflow boundary is the *cast*: a
    // merely f32-out-of-range `1e300` is a perfectly finite `f64` that becomes
    // `+inf` as an `f32`. The old `.map(|v| *v as f32)` had no check at all, so
    // this loader was the unguarded twin of `parse_milk`, which has enforced
    // finiteness on every scalar all along.
    let f = |key: &str, default: f32| -> f32 {
        match lc.get(&key.to_ascii_lowercase()) {
            Some(v) => {
                let narrowed = *v as f32;
                if narrowed.is_finite() {
                    narrowed
                } else {
                    log::warn!("ignoring non-finite value `{v}` for `{key}` in JSON preset");
                    default
                }
            }
            None => default,
        }
    };
    let b = |key: &str, default: bool| -> bool {
        match lc.get(&key.to_ascii_lowercase()) {
            Some(v) => *v != 0.0,
            None => default,
        }
    };

    Ok(MilkShaders {
        enhanced_audio: crate::enhanced_audio::EnhancedAudioConfig {
            fft_attack: f("fftattack", 0.5),
            fft_decay: f("fftdecay", 0.7),
            ..Default::default()
        }.sanitized(),
        beatdrop_feedback: b("bojobeatdropfeedback", false),
        // Butterchurn GLSL shader bodies → compiled via the GLSL-body path.
        warp: p.warp.clone().filter(|s| !s.trim().is_empty()),
        comp: p.comp.clone().filter(|s| !s.trim().is_empty()),
        shaders_glsl: true,

        // Equations → our EEL (raw `*_eqs_eel` preferred, else convert `*_eqs_str`).
        // init→per_frame_init, frame→per_frame, pixel→per_pixel.
        per_frame_init: pick_eqs(&p.init_eqs_eel, &p.init_eqs_str),
        per_frame: pick_eqs(&p.frame_eqs_eel, &p.frame_eqs_str),
        per_pixel: pick_eqs(&p.pixel_eqs_eel, &p.pixel_eqs_str),

        decay: f("decay", 0.98),
        gamma_adj: f("gammaadj", 2.0),
        fshader: f("fshader", 0.0),

        echo_zoom: f("echo_zoom", 2.0),
        echo_alpha: f("echo_alpha", 0.0),
        echo_orient: f("echo_orient", 0.0),

        brighten: b("brighten", false),
        darken: b("darken", false),
        solarize: b("solarize", false),
        invert: b("invert", false),

        warpscale: f("warpscale", 1.0),
        warpanimspeed: f("warpanimspeed", 1.0),
        zoom: f("zoom", 1.0),
        zoomexp: f("zoomexp", 1.0),
        rot: f("rot", 0.0),
        warp_amount: f("warp", 1.0),
        cx: f("cx", 0.5),
        cy: f("cy", 0.5),
        dx: f("dx", 0.0),
        dy: f("dy", 0.0),
        sx: f("sx", 1.0),
        sy: f("sy", 1.0),
        wrap: b("wrap", true),

        wave_mode: f("wave_mode", 0.0),
        wave_x: f("wave_x", 0.5),
        wave_y: f("wave_y", 0.5),
        wave_r: f("wave_r", 1.0),
        wave_g: f("wave_g", 1.0),
        wave_b: f("wave_b", 1.0),
        wave_a: f("wave_a", 1.0),
        wave_mystery: f("wave_mystery", 0.0),
        wave_scale: f("wave_scale", 1.0),
        wave_smoothing: f("wave_smoothing", 0.75),
        wave_dots: b("wave_dots", false),
        wave_thick: b("wave_thick", false),
        additive_wave: b("additivewave", false),
        wave_brighten: b("wave_brighten", true),
        modwavealphabyvolume: b("modwavealphabyvolume", false),
        modwavealphastart: f("modwavealphastart", 0.75),
        modwavealphaend: f("modwavealphaend", 0.95),

        // Motion vectors — same butterchurn runtime defaults as parse_milk.
        mv_on: b("bmotionvectorson", true),
        mv_x: f("mv_x", 12.0),
        mv_y: f("mv_y", 9.0),
        mv_dx: f("mv_dx", 0.0),
        mv_dy: f("mv_dy", 0.0),
        mv_l: f("mv_l", 0.9),
        mv_r: f("mv_r", 1.0),
        mv_g: f("mv_g", 1.0),
        mv_b: f("mv_b", 1.0),
        mv_a: f("mv_a", 1.0),

        ob_size: f("ob_size", 0.01),
        ob_r: f("ob_r", 0.0),
        ob_g: f("ob_g", 0.0),
        ob_b: f("ob_b", 0.0),
        ob_a: f("ob_a", 0.0),
        ib_size: f("ib_size", 0.01),
        ib_r: f("ib_r", 0.25),
        ib_g: f("ib_g", 0.25),
        ib_b: f("ib_b", 0.25),
        ib_a: f("ib_a", 0.0),

        darken_center: b("bdarkencenter", false),

        b1n: f("b1n", 0.0),
        b1x: f("b1x", 1.0),
        b1ed: f("b1ed", 0.25),
        b2n: f("b2n", 0.0),
        b2x: f("b2x", 1.0),
        b3n: f("b3n", 0.0),
        b3x: f("b3x", 1.0),

        shapes: parse_json_shapes(&p.shapes),
        waves: parse_json_waves(&p.waves),
    })
}

fn parse_json_shapes(shapes: &[JsonSubObj]) -> Vec<ShapeCode> {
    let mut out = Vec::new();
    for s in shapes.iter().take(16) {
        let lc: HashMap<String, f64> = s
            .base_vals
            .iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), *v))
            .collect();
        let g = |key: &str| lc.get(&key.to_ascii_lowercase()).copied();
        let mut base = ShapeBaseVals::default();
        if let Some(v) = g("enabled") {
            base.enabled = v as i32;
        }
        if let Some(v) = g("sides") {
            base.sides = v as f32;
        }
        if let Some(v) = g("additive") {
            base.additive = v as i32;
        }
        if let Some(v) = g("thickoutline") {
            base.thick_outline = v as i32;
        }
        if let Some(v) = g("textured") {
            base.textured = v as i32;
        }
        if let Some(v) = g("num_inst") {
            base.num_inst = v as i32;
        }
        if let Some(v) = g("x") {
            base.x = v as f32;
        }
        if let Some(v) = g("y") {
            base.y = v as f32;
        }
        if let Some(v) = g("rad") {
            base.rad = v as f32;
        }
        if let Some(v) = g("ang") {
            base.ang = v as f32;
        }
        if let Some(v) = g("tex_ang") {
            base.tex_ang = v as f32;
        }
        if let Some(v) = g("tex_zoom") {
            base.tex_zoom = v as f32;
        }
        if let Some(v) = g("r") {
            base.r = v as f32;
        }
        if let Some(v) = g("g") {
            base.g = v as f32;
        }
        if let Some(v) = g("b") {
            base.b = v as f32;
        }
        if let Some(v) = g("a") {
            base.a = v as f32;
        }
        if let Some(v) = g("r2") {
            base.r2 = v as f32;
        }
        if let Some(v) = g("g2") {
            base.g2 = v as f32;
        }
        if let Some(v) = g("b2") {
            base.b2 = v as f32;
        }
        if let Some(v) = g("a2") {
            base.a2 = v as f32;
        }
        if let Some(v) = g("border_r") {
            base.border_r = v as f32;
        }
        if let Some(v) = g("border_g") {
            base.border_g = v as f32;
        }
        if let Some(v) = g("border_b") {
            base.border_b = v as f32;
        }
        if let Some(v) = g("border_a") {
            base.border_a = v as f32;
        }
        out.push(ShapeCode {
            base,
            per_frame: pick_eqs(&s.frame_eqs_eel, &s.frame_eqs_str),
            per_frame_init: pick_eqs(&s.init_eqs_eel, &s.init_eqs_str),
        });
    }
    out
}

fn parse_json_waves(waves: &[JsonSubObj]) -> Vec<CustomWaveDef> {
    let mut out = Vec::new();
    for (n, w) in waves.iter().take(16).enumerate() {
        let lc: HashMap<String, f64> = w
            .base_vals
            .iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), *v))
            .collect();
        let g = |key: &str| lc.get(&key.to_ascii_lowercase()).copied();
        let mut cw = CustomWaveDef {
            index: n as u32,
            ..Default::default()
        };
        if let Some(v) = g("enabled") {
            cw.enabled = v != 0.0;
        }
        if let Some(v) = g("samples") {
            cw.samples = v.max(0.0) as u32;
        }
        if let Some(v) = g("sep") {
            cw.sep = v as i32;
        }
        // The native-converter JSON stores these wave flags WITHOUT the `b` prefix
        // (spectrum/usedots/thick/additive); the b-prefixed names matched nothing, so
        // ~6,800 corpus waves silently lost `additive` — the glow that bootstraps the
        // feedback buffer. Read the un-prefixed key first, fall back to b-prefixed.
        if let Some(v) = g("spectrum").or_else(|| g("bspectrum")) {
            cw.spectrum = v != 0.0;
        }
        if let Some(v) = g("usedots").or_else(|| g("busedots")) {
            cw.use_dots = v != 0.0;
        }
        if let Some(v) = g("thick").or_else(|| g("bdrawthick")) {
            cw.draw_thick = v != 0.0;
        }
        if let Some(v) = g("additive").or_else(|| g("badditive")) {
            cw.additive = v != 0.0;
        }
        if let Some(v) = g("scaling") {
            cw.scaling = v as f32;
        }
        if let Some(v) = g("smoothing") {
            cw.smoothing = v as f32;
        }
        if let Some(v) = g("r") {
            cw.r = v as f32;
        }
        if let Some(v) = g("g") {
            cw.g = v as f32;
        }
        if let Some(v) = g("b") {
            cw.b = v as f32;
        }
        if let Some(v) = g("a") {
            cw.a = v as f32;
        }
        cw.per_frame_init = pick_eqs(&w.init_eqs_eel, &w.init_eqs_str);
        cw.per_frame = pick_eqs(&w.frame_eqs_eel, &w.frame_eqs_str);
        cw.per_point = pick_eqs(&w.point_eqs_eel, &w.point_eqs_str);
        out.push(cw);
    }
    out
}

/// Convert a Butterchurn JS-transpiled equation string to our EEL.
///
/// Butterchurn stores per-frame/pixel equations as JavaScript: variables are
/// `a.NAME`, math functions are `Math.fn(...)`. Our EEL uses bare `NAME` and bare
/// `fn(...)`. The remaining constructs are already EEL-compatible after our recent
/// additions:
///   - ternary `cond ? a : b`         (parse_ternary)
///   - `div(a,b)` / `mod(a,b)`        (eval_call)
///   - `above(..)` / `below(..)` / `sqrt(..)` / `abs(..)`  (already supported)
///
/// Transform:
///   1. strip leading `a.` on identifiers: regex-free `\ba\.` → ``
///   2. strip `Math.`: `Math.` → `` (so Math.sin → sin, etc.)
/// Example:
///   `a.x1=.00001<Math.abs(above(a.d,a.r))?0:Math.sin(a.y-a.cy1)*a.dir`
///   → `x1=.00001<abs(above(d,r))?0:sin(y-cy1)*dir`
pub fn js_to_eel(src: &str) -> String {
    // `Math.` → `` first (so `Math.abs` → `abs`, never matched by the `a.` pass).
    let s = src.replace("Math.", "");
    // Strip a leading `a.` namespace prefix from identifiers, but only at a token
    // boundary (`\ba\.`) so we never eat the `a` inside e.g. `data.foo` or `aa.x`.
    strip_a_prefix(&s)
}

/// Choose an equation block: prefer the raw MilkDrop EEL (`*_eqs_eel`, emitted by
/// the native converter) which our evaluator parses directly; otherwise translate
/// the JS-transpiled `*_eqs_str` (the curated library only carries the latter).
/// js_to_eel only handles the `a.k`/`Math.` dialect — it cannot translate the
/// native converter's `a['k']` + `for(var…)` JS, so using the raw EEL is what makes
/// native-converted presets actually animate (q-vars/zoom/warp instead of frozen).
fn pick_eqs(eel: &Option<String>, str_js: &Option<String>) -> Option<String> {
    if let Some(e) = eel {
        if !e.trim().is_empty() {
            return Some(e.clone());
        }
    }
    str_js.as_deref().map(js_to_eel).filter(|s| !s.is_empty())
}

/// Remove `a.` where the `a` is at a word boundary (not preceded by an identifier
/// char). Mirrors the regex `\ba\.` → ``.
fn strip_a_prefix(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        // Match `a.` with `a` at a word boundary and `.` not part of a number.
        if bytes[i] == b'a'
            && i + 1 < bytes.len()
            && bytes[i + 1] == b'.'
            && (i == 0 || !is_ident_char(bytes[i - 1]))
        {
            // Preceding char must not be an identifier char (word boundary).
            // The following char (after the dot) must start an identifier — this
            // guards against a stray `a.` before a number, which doesn't occur in
            // these equations but keeps the transform safe.
            let next = bytes.get(i + 2).copied().unwrap_or(0);
            if is_ident_start(next) {
                i += 2; // skip "a."
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}
fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_presets_cannot_smuggle_a_non_finite_scalar_past_the_f32_narrowing() {
        let json = r#"{"baseVals":{"b1n":1e300,"b1x":-1e300,"b2n":-1e300,"zoom":1e300}}"#;
        // `MilkShaders` is not `Debug`, so unwrap by hand rather than `.expect`.
        let Ok(s) = load(json) else {
            panic!("preset must still load, with defaults substituted");
        };
        assert_eq!(s.b1n, 0.0, "b1n must fall back to its default, not +inf");
        assert_eq!(s.b1x, 1.0, "b1x must fall back to its default, not -inf");
        assert_eq!(s.b2n, 0.0, "b2n must fall back to its default");
        assert_eq!(
            s.zoom, 1.0,
            "the guard is on the shared reader, not one field"
        );
        for (name, v) in [
            ("b1n", s.b1n),
            ("b1x", s.b1x),
            ("b2n", s.b2n),
            ("b2x", s.b2x),
            ("b3n", s.b3n),
            ("b3x", s.b3x),
            ("zoom", s.zoom),
            ("decay", s.decay),
        ] {
            assert!(v.is_finite(), "{name} reached MilkShaders as {v}");
        }
    }

    /// The boundary of this guard's responsibility, pinned so it is not widened by
    /// mistake: an out-of-`f64`-range literal never reaches the field reader at
    /// all, because serde_json fails the whole document first. A reader who
    /// assumes `1e400` arrives as an `f64` infinity — as an earlier draft of the
    /// comment above did — is wrong.
    #[test]
    fn json_parse_rejects_out_of_f64_range_before_we_see_it() {
        let Err(err) = load(r#"{"baseVals":{"b1n":1e400}}"#) else {
            panic!("an out-of-f64-range literal must not load");
        };
        assert!(
            err.contains("number out of range"),
            "unexpected parse error: {err}"
        );
    }

    /// Non-vacuity: ordinary in-range values are read exactly as before, including
    /// ones near the f32 boundary. A guard that clamped magnitude rather than
    /// testing finiteness would fail this.
    #[test]
    fn json_presets_keep_finite_values_including_large_in_range_magnitudes() {
        let json = r#"{"baseVals":{"b1n":0.25,"b1x":0.75,"zoom":1.08,"decay":3.0e38}}"#;
        let Ok(s) = load(json) else { panic!("load") };
        assert_eq!(s.b1n, 0.25);
        assert_eq!(s.b1x, 0.75);
        assert_eq!(s.zoom, 1.08);
        assert_eq!(s.decay, 3.0e38_f64 as f32);
        assert!(s.decay.is_finite());
    }

    #[test]
    fn js_to_eel_pixel_line() {
        // The exact example from the task spec.
        let input = "a.x1=.00001<Math.abs(above(a.d,a.r))?0:Math.sin(a.y-a.cy1)*a.dir";
        let want = "x1=.00001<abs(above(d,r))?0:sin(y-cy1)*dir";
        assert_eq!(js_to_eel(input), want);
    }

    #[test]
    fn js_to_eel_frame_line() {
        let input = "a.ib_r=.3*Math.sin(5*a.time)+.7;a.wave_r=1-a.ib_r;";
        let want = "ib_r=.3*sin(5*time)+.7;wave_r=1-ib_r;";
        assert_eq!(js_to_eel(input), want);
    }

    #[test]
    fn js_to_eel_keeps_div() {
        let input = "a.ib_b=.5*Math.sin(4*div(a.time,3))+.5;";
        let want = "ib_b=.5*sin(4*div(time,3))+.5;";
        assert_eq!(js_to_eel(input), want);
    }

    #[test]
    fn js_to_eel_does_not_eat_embedded_a() {
        // `a.` only stripped at a word boundary — `data.x` and `aa.y` are left alone
        // (well-behaved: these don't appear in Butterchurn eqs but the guard matters).
        assert_eq!(js_to_eel("a.x=data.y"), "x=data.y");
    }

    #[test]
    fn json_custom_wave_sep_120_survives_load() {
        let json = r#"{"waves":[{"baseVals":{"enabled":1,"sep":120}}]}"#;
        let s = load(json).expect("load json");
        assert_eq!(s.waves.len(), 1);
        assert_eq!(
            s.waves[0].sep, 120,
            "numeric sep was coerced (expected 120)"
        );
    }

    #[test]
    fn loads_sherwin_basevals() {
        let json = include_str!("test_data/sherwin_maxawow.json");
        let s = load(json).expect("load sherwin_maxawow.json");
        assert!(s.shaders_glsl);
        assert!(s.warp.is_some());
        assert!(s.comp.is_some());
        // gammaadj 1.56, decay 1, echo_zoom 0.362, warpscale 0.107, zoomexp 0.1584.
        assert!(
            (s.gamma_adj - 1.56).abs() < 1e-4,
            "gamma_adj={}",
            s.gamma_adj
        );
        assert!((s.decay - 1.0).abs() < 1e-4);
        assert!((s.echo_zoom - 0.362).abs() < 1e-4);
        assert!((s.warpscale - 0.107).abs() < 1e-4);
        assert!((s.zoomexp - 0.1584).abs() < 1e-4);
        // warp scalar 0.01, fshader 1, mv_x 64, mv_y 48, ib_a 1, ob_a 1.
        assert!((s.warp_amount - 0.01).abs() < 1e-4);
        assert!((s.fshader - 1.0).abs() < 1e-4);
        assert!((s.mv_x - 64.0).abs() < 1e-4);
        assert!((s.mv_y - 48.0).abs() < 1e-4);
        // darken is 1 → true; wave_thick 1 → true.
        assert!(s.darken);
        assert!(s.wave_thick);
        // Equations converted (no `a.` / `Math.` left).
        let pf = s.per_pixel.as_deref().unwrap();
        assert!(!pf.contains("a."), "per_pixel still has a.: {pf}");
        assert!(!pf.contains("Math."), "per_pixel still has Math.");
        assert!(pf.contains("div(bass,4)"), "div not preserved: {pf}");
    }
}
