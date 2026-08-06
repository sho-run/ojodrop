#![allow(dead_code)]
// butterchurn GLSL (#version 300 es) → naga-compatible GLSL 450
//
// Butterchurn already uses GLSL (not HLSL). The transforms needed are:
//
//   1. Version bump: `#version 300 es` → `#version 450`
//   2. Strip `precision ...` declarations (not valid in 450 core)
//   3. Add layout qualifiers to `in`/`out` variables
//   4. Convert combined `uniform sampler2D/sampler3D name` →
//        separate `texture2D/texture3D + sampler` with layout qualifiers
//   5. Rewrite `texture(name, ...)` calls to `texture(sampler2D(name, name_samp), ...)`
//   6. Pack individual scalar/vector `uniform` declarations into a UBO
//   7. Convert `float PI = ...` → `const float PI = ...`

use std::collections::HashMap;

/// Hard byte budget for a single shader source fed to [`butterchurn_to_naga`].
/// Real Butterchurn comp/warp bodies are a few KiB; a 512 KiB ceiling sits far
/// above any legitimate preset yet bounds the preprocessing work (and allocation)
/// for an adversarial blob before any pass runs.
const MAX_SHADER_BYTES: usize = 512 * 1024;

/// Hard line budget. Bounds the single classification pass independently of byte
/// count (a source of many short lines still costs per-line work). Real bodies are
/// well under a few hundred lines.
const MAX_SHADER_LINES: usize = 20_000;

/// Typed rejection from the Butterchurn→naga shader preprocessor. The infallible
/// [`butterchurn_to_naga`] maps it to an inert empty program; callers that need to
/// distinguish a rejection use [`try_butterchurn_to_naga`]. Mirrors the
/// `equations::ParseError` convention (an infallible entry point plus a fallible
/// `try_` sibling).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessError {
    /// Source exceeded [`MAX_SHADER_BYTES`]; rejected before any pass ran.
    SourceTooLarge { bytes: usize, limit: usize },
    /// Source exceeded [`MAX_SHADER_LINES`]; rejected before any pass ran.
    TooManyLines { lines: usize, limit: usize },
}

/// One source line, classified exactly once. The previous implementation parsed
/// every line twice (a collection pass, then a rewrite pass); recording the
/// classification lets the emit pass replay it, so each line is parsed once —
/// single-pass symbol/reference accounting.
enum ClassifiedLine {
    /// `#version …` → replaced with `#version 450`.
    Version,
    /// `precision …` → dropped.
    Precision,
    /// `in <ty> <name>;` → emitted once with a `layout(location)` qualifier.
    In { name: String, ty: String },
    /// `out <ty> <name>;`
    Out { name: String, ty: String },
    /// `uniform sampler2D/3D <name>;` → emitted once as separate texture + sampler.
    Sampler { name: String, ty: String },
    /// `uniform <scalar/vector> <name>;` → folded into the UBO; original dropped.
    Scalar,
    /// Any other line, carried through verbatim (subject to the `float PI` fixup and
    /// the pre-body UBO emission).
    Other(String),
}

/// Convert butterchurn GLSL (`#version 300 es`) to naga-compatible GLSL 450.
///
/// Infallible entry point: an over-budget or pathological source yields an inert
/// (empty) program rather than unbounded work. Callers that must distinguish a
/// rejection use [`try_butterchurn_to_naga`].
pub fn butterchurn_to_naga(src: &str) -> String {
    try_butterchurn_to_naga(src).unwrap_or_default()
}

/// Fallible core of [`butterchurn_to_naga`]. Enforces source budgets up front, then
/// converts the shader in a SINGLE classification pass (each line parsed once),
/// returning a typed [`PreprocessError`] for over-budget input. The final texture
/// rewrite is linear in the source length (see [`rewrite_texture_calls`]), so the
/// whole conversion is bounded — no quadratic per-symbol rescans.
pub fn try_butterchurn_to_naga(src: &str) -> Result<String, PreprocessError> {
    // Budget 1: total bytes. Every downstream pass is linear in the source length,
    // so capping it bounds the total work before any allocation/scan runs.
    if src.len() > MAX_SHADER_BYTES {
        return Err(PreprocessError::SourceTooLarge {
            bytes: src.len(),
            limit: MAX_SHADER_BYTES,
        });
    }

    // Single classification pass: parse each line ONCE, recording both the decl
    // tables (scalars for the UBO) and the binding/location maps, plus a per-line
    // plan the emit pass replays without re-parsing.
    let mut classified: Vec<ClassifiedLine> = Vec::new();
    let mut scalars: Vec<ScalarDecl> = Vec::new();
    let mut sampler_map: HashMap<String, u32> = HashMap::new(); // name → tex binding
    let mut sampler_binding = 0u32;
    let mut in_locs: HashMap<String, u32> = HashMap::new();
    let mut out_locs: HashMap<String, u32> = HashMap::new();
    let mut in_loc = 0u32;
    let mut out_loc = 0u32;
    let mut line_count = 0usize;

    for line in src.lines() {
        // Budget 2: line count, checked as we scan so a many-line blob is rejected
        // in bounded work (before the emit pass and the join allocate anything).
        line_count += 1;
        if line_count > MAX_SHADER_LINES {
            return Err(PreprocessError::TooManyLines {
                lines: line_count,
                limit: MAX_SHADER_LINES,
            });
        }

        let t = line.trim();
        let classified_line = if t.starts_with("#version") {
            ClassifiedLine::Version
        } else if t.starts_with("precision ") {
            ClassifiedLine::Precision
        } else if let Some(v) = parse_io(t, "in") {
            in_locs.entry(v.name.clone()).or_insert_with(|| {
                let l = in_loc;
                in_loc += 1;
                l
            });
            ClassifiedLine::In {
                name: v.name,
                ty: v.ty,
            }
        } else if let Some(v) = parse_io(t, "out") {
            out_locs.entry(v.name.clone()).or_insert_with(|| {
                let l = out_loc;
                out_loc += 1;
                l
            });
            ClassifiedLine::Out {
                name: v.name,
                ty: v.ty,
            }
        } else if let Some(s) = parse_sampler(t) {
            // Assign a binding pair (tex + sampler) at first occurrence, in order of
            // appearance — the same slot layout the previous two-pass code produced.
            sampler_map.entry(s.name.clone()).or_insert_with(|| {
                let b = sampler_binding;
                sampler_binding += 2;
                b
            });
            ClassifiedLine::Sampler {
                name: s.name,
                ty: s.ty,
            }
        } else if let Some(s) = parse_scalar(t) {
            scalars.push(s);
            ClassifiedLine::Scalar
        } else {
            ClassifiedLine::Other(line.to_string())
        };
        classified.push(classified_line);
    }

    let ubo_binding = sampler_binding; // UBO goes after all texture slots

    // Emit pass: replay the classification (no re-parsing of source text).
    let mut result: Vec<String> = Vec::new();
    let mut ubo_emitted = false;
    let mut in_out_done: std::collections::HashSet<String> = Default::default();
    let mut sampler_done: std::collections::HashSet<String> = Default::default();

    for c in &classified {
        match c {
            ClassifiedLine::Version => result.push("#version 450".to_string()),
            // precision declarations → strip
            ClassifiedLine::Precision => {}
            // in / out variables: add layout qualifier (once per name)
            ClassifiedLine::In { name, ty } => {
                if in_out_done.insert(format!("in:{name}")) {
                    let loc = in_locs[name];
                    result.push(format!("layout(location = {loc}) in {ty} {name};"));
                }
            }
            ClassifiedLine::Out { name, ty } => {
                if in_out_done.insert(format!("out:{name}")) {
                    let loc = out_locs[name];
                    result.push(format!("layout(location = {loc}) out {ty} {name};"));
                }
            }
            // sampler uniforms → separate texture + sampler (once per name)
            ClassifiedLine::Sampler { name, ty } => {
                if sampler_done.insert(name.clone()) {
                    let b = sampler_map[name];
                    let (tex_ty, samp_ty) = if ty == "sampler3D" {
                        ("texture3D", "sampler")
                    } else {
                        ("texture2D", "sampler")
                    };
                    result.push(format!(
                        "layout(set = 0, binding = {b}) uniform {tex_ty} {name};"
                    ));
                    result.push(format!(
                        "layout(set = 0, binding = {}) uniform {samp_ty} {name}_samp;",
                        b + 1
                    ));
                }
            }
            // scalar uniforms are folded into the UBO — drop the individual decl
            ClassifiedLine::Scalar => {}
            ClassifiedLine::Other(orig) => {
                let t = orig.trim();
                // Emit the UBO just before the first real body line.
                if !ubo_emitted && !scalars.is_empty() && is_body_line(t) {
                    emit_ubo(&scalars, ubo_binding, &mut result);
                    ubo_emitted = true;
                }
                // `float PI = ...` → `const float PI = ...` (naga rejects mutable
                // globals without a storage qualifier).
                let line_out = if t.starts_with("float PI") && t.contains('=') {
                    orig.replacen("float PI", "const float PI", 1)
                } else {
                    orig.clone()
                };
                result.push(line_out);
            }
        }
    }

    // Emit UBO if we never hit a body line (edge case).
    if !ubo_emitted && !scalars.is_empty() {
        emit_ubo(&scalars, ubo_binding, &mut result);
    }

    // Final pass: rewrite texture() calls in the full string (single linear scan).
    let joined = result.join("\n");
    Ok(rewrite_texture_calls(&joined, &sampler_map))
}

// ---------------------------------------------------------------------------
// UBO emission
// ---------------------------------------------------------------------------

fn emit_ubo(scalars: &[ScalarDecl], binding: u32, out: &mut Vec<String>) {
    out.push(format!(
        "layout(set = 1, binding = {binding}) uniform PerFrame {{"
    ));
    // Emit in std140-friendly order: vec4 first, then vec3 (aligns as vec4), then
    // vec2, then float, then int (avoids alignment padding issues)
    for ty_order in &["vec4", "vec3", "vec2", "float", "int"] {
        for s in scalars {
            if &s.ty.as_str() == ty_order {
                out.push(format!("    {} {};", s.ty, s.name));
            }
        }
    }
    out.push("};".to_string());
}

// ---------------------------------------------------------------------------
// Texture call rewriting
// ---------------------------------------------------------------------------

// Rewrites `texture(name, ...)` → `texture(sampler2D(name, name_samp), ...)`
// for every sampler named in `sampler_map`.
//
// A SINGLE linear scan (O(len)) replaces the previous per-name `String::replace`
// loop, which was O(samplers × len) — quadratic when both the sampler count and the
// body length grow with the (untrusted) source. Reading the full identifier after
// each `texture(` also removes the need for the old longest-first name sort: a name
// that is a prefix of another (`sampler_noise_lq` vs `sampler_noise_lq_lite`) can no
// longer partially match, because the whole `[A-Za-z0-9_]+` token is compared.
//
// Output is byte-for-byte identical to the old rewriter for the two whitespace
// variants it handled: `texture(name,` and `texture(name ,` (exactly one space).
fn rewrite_texture_calls(src: &str, sampler_map: &HashMap<String, u32>) -> String {
    const PREFIX: &[u8] = b"texture(";
    let b = src.as_bytes();
    let n = b.len();
    let mut out = String::with_capacity(src.len() + src.len() / 8);
    let mut i = 0;
    while i < n {
        if i + PREFIX.len() <= n && &b[i..i + PREFIX.len()] == PREFIX {
            // Read the identifier immediately after `texture(`.
            let id_start = i + PREFIX.len();
            let mut j = id_start;
            while j < n && is_ident_byte(b[j]) {
                j += 1;
            }
            // Only a `texture(name,` or `texture(name ,` call (name directly
            // followed by a comma, or a single space then a comma) is a sampler
            // read to rewrite — anything else is left untouched.
            let sep_len = if j < n && b[j] == b',' {
                Some(0)
            } else if j + 1 < n && b[j] == b' ' && b[j + 1] == b',' {
                Some(1)
            } else {
                None
            };
            if let Some(sep_len) = sep_len {
                let name = &src[id_start..j];
                if !name.is_empty() && sampler_map.contains_key(name) {
                    out.push_str("texture(sampler2D(");
                    out.push_str(name);
                    out.push_str(", ");
                    out.push_str(name);
                    out.push_str("_samp)");
                    // Re-emit the separator whitespace exactly (the space in the
                    // ` ,` variant); the comma itself is copied by the main loop.
                    for _ in 0..sep_len {
                        out.push(' ');
                    }
                    i = j + sep_len; // resume at the comma
                    continue;
                }
            }
        }
        // Default: copy one UTF-8 char through unchanged.
        let ch = src[i..].chars().next().unwrap();
        let len = ch.len_utf8();
        out.push_str(&src[i..i + len]);
        i += len;
    }
    out
}

// ---------------------------------------------------------------------------
// Line classifiers
// ---------------------------------------------------------------------------

struct SamplerDecl {
    name: String,
    ty: String,
}
struct ScalarDecl {
    name: String,
    ty: String,
}
struct IoVar {
    name: String,
    ty: String,
}

fn parse_sampler(line: &str) -> Option<SamplerDecl> {
    // `uniform sampler2D name;` or `uniform sampler3D name;`
    let l = line.trim_end_matches(';').trim();
    let rest = l.strip_prefix("uniform ")?;
    let ty = if rest.starts_with("sampler2D ") {
        "sampler2D"
    } else if rest.starts_with("sampler3D ") {
        "sampler3D"
    } else {
        return None;
    };
    let name = rest[ty.len()..].trim().to_string();
    if name.is_empty() || name.contains(' ') {
        return None;
    }
    Some(SamplerDecl {
        name,
        ty: ty.to_string(),
    })
}

fn parse_scalar(line: &str) -> Option<ScalarDecl> {
    // `uniform float/int/vec2/vec4 name;` (single-variable declarations only)
    let l = line.trim_end_matches(';').trim();
    let rest = l.strip_prefix("uniform ")?;
    let ty = ["float ", "int ", "vec2 ", "vec4 ", "vec3 "]
        .iter()
        .find(|&&p| rest.starts_with(p))?;
    let ty = ty.trim();
    let name = rest[ty.len() + 1..].trim().to_string();
    // Skip if contains array syntax or space (multiple declarations)
    if name.is_empty() || name.contains('[') || name.contains(' ') || name.contains(',') {
        return None;
    }
    Some(ScalarDecl {
        name,
        ty: ty.to_string(),
    })
}

fn parse_io(line: &str, keyword: &str) -> Option<IoVar> {
    // `in vec2 name;` or `out vec4 name;`
    let l = line.trim_end_matches(';').trim();
    let rest = l.strip_prefix(&format!("{keyword} "))?;
    // Must be type + name (two tokens)
    let mut parts = rest.splitn(2, ' ');
    let ty = parts.next()?.trim().to_string();
    let name = parts.next()?.trim().to_string();
    if name.is_empty() || name.contains(' ') || ty.is_empty() {
        return None;
    }
    Some(IoVar { name, ty })
}

// Returns true for lines that represent actual shader body / non-declaration content.
fn is_body_line(line: &str) -> bool {
    let l = line.trim();
    if l.is_empty() {
        return false;
    }
    if l.starts_with("//") {
        return false;
    }
    if l.starts_with("#version") {
        return false;
    }
    if l.starts_with("#define") {
        return false;
    }
    if l.starts_with("precision") {
        return false;
    }
    if l.starts_with("uniform") {
        return false;
    }
    if l.starts_with("in ") || l.starts_with("out ") {
        return false;
    }
    if l.starts_with("layout") {
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Path: raw HLSL .milk body → naga-compatible GLSL 450
//
// Real .milk presets store HLSL shader bodies (float2/float3, tex2D, lerp, frac,
// saturate, GetBlur1, etc.).  This function wraps a raw body in a complete
// GLSL 450 program and applies all necessary substitutions.
// ---------------------------------------------------------------------------

/// All known MilkDrop sampler names.  Order determines binding slots.
pub const MILKDROP_SAMPLERS: &[&str] = &[
    "sampler_main",
    "sampler_fw_main",
    "sampler_fc_main",
    "sampler_pw_main",
    "sampler_pc_main",
    "sampler_blur1",
    "sampler_blur2",
    "sampler_blur3",
    "sampler_noise_lq",
    "sampler_noise_lq_lite",
    "sampler_noise_mq",
    "sampler_noise_hq",
    // Two legacy-redundant slots are reserved for the named-texture atlas. The
    // normalizer maps legacy hq-lite/point-lq uses to their equivalent base noise
    // samplers before compilation, keeping the total binding count unchanged.
    "sampler_named_linear",
    "sampler_named_point",
    "sampler_noisevol_lq",
    "sampler_noisevol_hq",
];

/// Shared FS preamble: sampler/texture declarations, the PerFrame UBO block,
/// q1..q8 defines, PI const, and the helper functions (GetMain/GetBlur*/saturate…).
/// `io_decls` is the per-variant `in`/`out` declaration block (comp vs warp).
fn milk_fs_preamble(io_decls: &str) -> String {
    let mut tex_decls = String::new();
    for (i, name) in MILKDROP_SAMPLERS.iter().enumerate() {
        let tex_bind = (i * 2) as u32;
        let samp_bind = tex_bind + 1;
        // Use texture3D for volume noise samplers
        let tex_ty = if name.contains("vol") {
            "texture3D"
        } else {
            "texture2D"
        };
        tex_decls.push_str(&format!(
            "layout(set = 0, binding = {tex_bind}) uniform {tex_ty} {name};\n"
        ));
        tex_decls.push_str(&format!(
            "layout(set = 0, binding = {samp_bind}) uniform sampler {name}_samp;\n"
        ));
    }

    let ubo_bind = (MILKDROP_SAMPLERS.len() * 2) as u32;

    // Helper functions: GetBlur1/2/3, GetPixel, saturate overloads, lum
    // These mirror butterchurn's comp.js preamble, in GLSL 450 form.
    let helpers = r#"
vec3 GetMain(vec2 uv)  { return texture(sampler2D(sampler_main,  sampler_main_samp),  uv).rgb; }
// The blur textures store values normalized to [0,1] (blur shader: blur*scaleN+biasN).
// GetBlurN applies the inverse (scaleN = maxN-minN, biasN = minN) to recover the range,
// matching MilkDrop/butterchurn. At defaults (min 0, max 1) this is identity.
vec3 GetBlur1(vec2 uv) { return texture(sampler2D(sampler_blur1, sampler_blur1_samp), uv).rgb * scale1 + bias1; }
vec3 GetBlur2(vec2 uv) { return texture(sampler2D(sampler_blur2, sampler_blur2_samp), uv).rgb * scale2 + bias2; }
vec3 GetBlur3(vec2 uv) { return texture(sampler2D(sampler_blur3, sampler_blur3_samp), uv).rgb * scale3 + bias3; }
vec3 GetPixel(vec2 uv) { return texture(sampler2D(sampler_main,  sampler_main_samp),  uv).rgb; }
// MilkDrop's HLSL prefix defines lum as a dot product. HLSL accepts scalar and
// float2 arguments here through its implicit width conversions; GLSL does not.
// Keep those narrower overloads scalar so expressions such as
// `uv -= lum(float2_value) * direction` preserve the authored HLSL width.
float lum(float v) { return v; }
float lum(vec2 v) { return dot(v, vec2(0.32, 0.49)); }
vec3 lum(vec3 v) { return vec3(dot(v, vec3(0.32, 0.49, 0.29))); }
vec3 lum(vec4 v) { return vec3(dot(v.rgb, vec3(0.32, 0.49, 0.29))); }
// vec3-argument Get* overloads: presets call GetBlurN/GetMain/GetPixel with a vec3
// (e.g. `GetBlur3(ret)` where ret is vec3), relying on HLSL float3->float2 implicit
// truncation. Forward `.xy` to the canonical vec2 implementation (declared above).
vec3 GetMain(vec3 uv)  { return GetMain(uv.xy); }
vec3 GetBlur1(vec3 uv) { return GetBlur1(uv.xy); }
vec3 GetBlur2(vec3 uv) { return GetBlur2(uv.xy); }
vec3 GetBlur3(vec3 uv) { return GetBlur3(uv.xy); }
vec3 GetPixel(vec3 uv) { return GetPixel(uv.xy); }
// HLSL modf(x, out ip): returns the fractional part and writes the integer part
// (trunc toward zero) to the out param. naga's GLSL frontend has no modf builtin.
float modf(float x, out float ip) { ip = trunc(x); return x - ip; }

float saturate(float x)  { return clamp(x, 0.0, 1.0); }
vec2  saturate(vec2 x)   { return clamp(x, vec2(0.0), vec2(1.0)); }
vec3  saturate(vec3 x)   { return clamp(x, vec3(0.0), vec3(1.0)); }
vec4  saturate(vec4 x)   { return clamp(x, vec4(0.0), vec4(1.0)); }

// HLSL exposes log10 for every floating genType; GLSL 4.50/naga does not.
// Keep the overloads in the fixed preamble so authored shaders retain HLSL's
// component-wise semantics without teaching the parser a synthetic builtin.
float log10(float x) { return log(x) * 0.4342944819032518; }
vec2  log10(vec2 x)  { return log(x) * 0.4342944819032518; }
vec3  log10(vec3 x)  { return log(x) * 0.4342944819032518; }
vec4  log10(vec4 x)  { return log(x) * 0.4342944819032518; }

// Guarded reciprocal: returns 0 where the denominator is 0, emulating DX9/WebGL
// fast-math where x/0->inf and inf*0->0 (so a /0 term harmlessly vanishes) instead
// of wgpu strict-IEEE where inf*0=NaN poisons the whole pixel to black. preprocess::
// guard_divides rewrites `a / b` (dynamic b) to `a * safeRecip(b)` (same precedence).
float safeRecip(float b) { return b != 0.0 ? 1.0 / b : 0.0; }
vec2  safeRecip(vec2 b)  { return vec2(b.x != 0.0 ? 1.0/b.x : 0.0, b.y != 0.0 ? 1.0/b.y : 0.0); }
vec3  safeRecip(vec3 b)  { return vec3(b.x != 0.0 ? 1.0/b.x : 0.0, b.y != 0.0 ? 1.0/b.y : 0.0, b.z != 0.0 ? 1.0/b.z : 0.0); }
vec4  safeRecip(vec4 b)  { return vec4(b.x != 0.0 ? 1.0/b.x : 0.0, b.y != 0.0 ? 1.0/b.y : 0.0, b.z != 0.0 ? 1.0/b.z : 0.0, b.w != 0.0 ? 1.0/b.w : 0.0); }

// A few raw MilkDrop shaders reference the converter-provided `rot_d1` matrix directly
// for channel offsets. Provide a deterministic shader-space equivalent so those bodies
// compile without having to special-case the preset text.
#define rot_d1 mat3(cos(time*0.37), -sin(time*0.37), 0.0, sin(time*0.37), cos(time*0.37), 0.0, 0.0, 0.0, 1.0)

// Component-wise logical ops for the Butterchurn converter's vector-bool `&&`/`||`
// (naga rejects `LogicalAnd(vecN<bool>, _)`). preprocess::rewrite_vector_logical
// emits calls to these in place of `bvecN(X) && bvecN(Y)` / `... || ...`.
bvec2 m_and2(bvec2 a, bvec2 b) { return bvec2(a.x && b.x, a.y && b.y); }
bvec3 m_and3(bvec3 a, bvec3 b) { return bvec3(a.x && b.x, a.y && b.y, a.z && b.z); }
bvec4 m_and4(bvec4 a, bvec4 b) { return bvec4(a.x && b.x, a.y && b.y, a.z && b.z, a.w && b.w); }
bvec2 m_or2(bvec2 a, bvec2 b)  { return bvec2(a.x || b.x, a.y || b.y); }
bvec3 m_or3(bvec3 a, bvec3 b)  { return bvec3(a.x || b.x, a.y || b.y, a.z || b.z); }
bvec4 m_or4(bvec4 a, bvec4 b)  { return bvec4(a.x || b.x, a.y || b.y, a.z || b.z, a.w || b.w); }
"#;

    format!(
        r#"#version 450
{io_decls}
{tex_decls}
layout(set = 1, binding = {ubo_bind}) uniform PerFrame {{
    vec4 texsize;
    vec4 aspect;
    vec4 slow_roam_cos;
    vec4 roam_cos;
    vec4 slow_roam_sin;
    vec4 roam_sin;
    vec4 rand_frame;
    vec4 rand_start;
    vec4 rand_preset;
    vec4 _qa;
    vec4 _qb;
    vec4 _qc;
    vec4 _qd;
    vec4 _qe;
    vec4 _qf;
    vec4 _qg;
    vec4 _qh;
    float time;
    float fps;
    float frame;
    float progress;
    float bass;
    float mid;
    float treb;
    float vol;
    float bass_att;
    float mid_att;
    float treb_att;
    float vol_att;
    float fShader;
    float gammaAdj;
    float echo_zoom;
    float echo_alpha;
    float echo_orientation;
    float blur1_min;
    float blur1_max;
    float blur2_min;
    float blur2_max;
    float blur3_min;
    float blur3_max;
    float scale1;
    float scale2;
    float scale3;
    float bias1;
    float bias2;
    float bias3;
    float brighten;
    float darken;
    float solarize;
    float invert;
}};

#define q1  _qa.x
#define q2  _qa.y
#define q3  _qa.z
#define q4  _qa.w
#define q5  _qb.x
#define q6  _qb.y
#define q7  _qb.z
#define q8  _qb.w
#define q9  _qc.x
#define q10 _qc.y
#define q11 _qc.z
#define q12 _qc.w
#define q13 _qd.x
#define q14 _qd.y
#define q15 _qd.z
#define q16 _qd.w
#define q17 _qe.x
#define q18 _qe.y
#define q19 _qe.z
#define q20 _qe.w
#define q21 _qf.x
#define q22 _qf.y
#define q23 _qf.z
#define q24 _qf.w
#define q25 _qg.x
#define q26 _qg.y
#define q27 _qg.z
#define q28 _qg.w
#define q29 _qh.x
#define q30 _qh.y
#define q31 _qh.z
#define q32 _qh.w

const float PI = 3.141592653589793;
// MilkDrop HLSL math constants presets assume exist. NOTE MilkDrop's M_PI_2 is 2*PI
// (NOT PI/2) and M_INV_PI_2 is 1/(2*PI) — a documented MilkDrop convention.
const float M_PI       = 3.141592653589793;
const float M_PI_2     = 6.283185307179586;
const float M_INV_PI   = 0.3183098861837907;
const float M_INV_PI_2 = 0.15915494309189535;

// Noise-texture sizes (Butterchurn noise.js). texsize_noise_X = vec4(w, h, 1/w, 1/h).
// Butterchurn warp/comp shaders sample these to convert pixel<->uv space for the
// noise lookups. Sizes: noise lq/mq/hq = 256², lq_lite = 32², noisevol = 32³.
const vec4 texsize_noise_lq      = vec4(256.0, 256.0, 1.0/256.0, 1.0/256.0);
const vec4 texsize_noise_mq      = vec4(256.0, 256.0, 1.0/256.0, 1.0/256.0);
const vec4 texsize_noise_hq      = vec4(256.0, 256.0, 1.0/256.0, 1.0/256.0);
const vec4 texsize_noise_lq_lite = vec4(32.0, 32.0, 1.0/32.0, 1.0/32.0);
const vec4 texsize_noise_hq_lite = texsize_noise_lq_lite;
const vec4 texsize_noisevol_lq   = vec4(32.0, 32.0, 1.0/32.0, 1.0/32.0);
const vec4 texsize_noisevol_hq   = vec4(32.0, 32.0, 1.0/32.0, 1.0/32.0);
{helpers}
// __MILK_BODY__
"#
    )
}

/// Convert a raw HLSL .milk shader body to a complete naga-compatible GLSL 450
/// COMP fragment shader (fullscreen-triangle VS path: vUv@0, vColor@1).
/// Try the native C++ HLSL→GLSL converter (hlsl2glslfork + glsl-optimizer).
/// Returns `Some(glsl_body)` on success — the body is ready for glsl_milk_body_to_naga.
/// Returns `None` on failure, allowing fall-through to the pure-Rust HLSL path.
/// All sampler names declared in MILK_HLSL_PREFIX (in milk_converter_shim.cpp).
/// HLSL `sampler X;` declarations for these are stripped but NOT added as extra
/// file-scope uniforms (they're already declared at file scope in the shim prefix).
/// Used by both the native pre-strip and the ungated pure-Rust fallback, so it is
/// not feature-gated.
const MILK_STANDARD_SAMPLERS: &[&str] = &[
    "sampler_main",
    "sampler_fw_main",
    "sampler_pw_main",
    "sampler_fc_main",
    "sampler_pc_main",
    "sampler_noise_lq",
    "sampler_noise_lq_lite",
    "sampler_noise_mq",
    "sampler_noise_hq",
    "sampler_named_linear",
    "sampler_named_point",
    "sampler_noisevol_lq",
    "sampler_noisevol_hq",
    "sampler_blur1",
    "sampler_blur2",
    "sampler_blur3",
];

#[cfg(feature = "milk-native-converter")]
fn try_native_convert_hlsl(body: &str) -> Option<String> {
    // hlsl2glslfork/glslopt segfaults on `const TYPE name[N] = { scalar, … }` array
    // initialisers with flat scalar lists. The subprocess boundary now contains
    // that crash, but this known-doomed shape is still cheaper to route directly
    // to the pure-Rust fallback (<1% of the corpus).
    if contains_const_array(body) {
        log::debug!(
            "native HLSL converter skipped (const array initialiser, falling back to pure-Rust)"
        );
        return None;
    }
    // Converter-output cache (MILK-02 #6): the C++ conversion is a deterministic
    // function of `body`, so a re-import of the same source reuses the prior GLSL
    // and skips the expensive helper round-trip. Key encodes the entry point so a
    // plain result can never be confused with an `_ex` result for the same body.
    let key = ConverterInput::Plain {
        body: body.to_string(),
    };
    let digest = converter_input_digest(&key);
    if let Some(cached) = converter_cache_get(digest, &key) {
        return Some(cached);
    }
    match particle_milkdrop_converter_sys::convert_milk_shader(body) {
        Ok(glsl_body) => {
            converter_cache_insert(digest, key, glsl_body.clone());
            Some(glsl_body)
        }
        Err(e) => {
            log::warn!("native HLSL converter failed (falling back to pure-Rust): {e}");
            None
        }
    }
}

/// Like `try_native_convert_hlsl` but uses the `_ex` entry point that places
/// `file_globals` at HLSL file scope (before `shader_body{}`).  Used when
/// `before` contains helper function definitions that must not live inside
/// the body function.
#[cfg(feature = "milk-native-converter")]
fn try_native_convert_hlsl_ex(file_globals: &str, body: &str) -> Option<String> {
    if contains_const_array(file_globals) || contains_const_array(body) {
        log::debug!("native HLSL converter (_ex) skipped (const array)");
        return None;
    }
    // Converter-output cache (MILK-02 #6): keyed on the exact inputs fed to the
    // converter (`file_globals`, `body`, and the always-true `optimize` flag) so a
    // re-import of the same source reuses the prior GLSL and skips the helper.
    let key = ConverterInput::Ex {
        file_globals: file_globals.to_string(),
        body: body.to_string(),
        optimize: true,
    };
    let digest = converter_input_digest(&key);
    if let Some(cached) = converter_cache_get(digest, &key) {
        return Some(cached);
    }
    match particle_milkdrop_converter_sys::convert_milk_shader_ex(file_globals, body, true) {
        Ok(glsl_body) => {
            converter_cache_insert(digest, key, glsl_body.clone());
            Some(glsl_body)
        }
        Err(e) => {
            log::warn!("native HLSL converter (_ex) failed (falling back to pure-Rust): {e}");
            None
        }
    }
}

// ---- Converter-output cache (MILK-02 #6) ------------------------------------
//
// The native C++ HLSL→GLSL conversion (hlsl2glslfork + glsl-optimizer, run via
// the out-of-process converter-sys helper) is the dominant preset-import latency
// cost. It is a pure, deterministic function of its exact inputs, so re-importing
// the same .milk otherwise re-runs the same expensive conversion. This bounded,
// thread-safe LRU caches the converter OUTPUT keyed by those exact input bytes: a
// hit returns byte-identical GLSL and skips the C++ work; any change to the source
// yields a different key and re-converts (source-hash invalidation).
//
// Distinct from the compiled shader-BODY WGSL cache in milkdrop_runtime.rs (a
// later stage); this caches the raw converter output. The idiom mirrors that
// crate's `ByteBudgetLruCache`: a 64-bit digest fast-index plus a retained full
// key that is compared on every lookup, so a digest collision can never return the
// wrong GLSL.

/// Byte budget for the converter-output cache. Bounds retained GLSL so preset
/// churn cannot grow it without limit. Sized to match the compiled-body cache.
#[cfg(feature = "milk-native-converter")]
const CONVERTER_OUTPUT_CACHE_BUDGET_BYTES: usize = 8 * 1024 * 1024;

/// The exact inputs to one native-converter invocation. The variant encodes the
/// entry point so a `convert_milk_shader(body)` result can never be confused with
/// a `convert_milk_shader_ex(globals, body, _)` result for the same `body`.
#[cfg(feature = "milk-native-converter")]
#[derive(Clone, PartialEq, Eq, Hash)]
enum ConverterInput {
    /// `convert_milk_shader(body)`.
    Plain { body: String },
    /// `convert_milk_shader_ex(file_globals, body, optimize)`.
    Ex {
        file_globals: String,
        body: String,
        optimize: bool,
    },
}

#[cfg(feature = "milk-native-converter")]
struct ConverterCacheEntry {
    digest: u64,
    key: ConverterInput,
    glsl: String,
    bytes: usize,
}

/// A byte-bounded LRU keyed by a `(digest, full input)` pair. The digest is the
/// fast index; the full key is retained and compared on every lookup so digest
/// collisions are verified rather than trusted. Total retained bytes are kept
/// within `budget_bytes` by evicting least-recently-used entries. Mirrors
/// `ByteBudgetLruCache` in milkdrop_runtime.rs.
#[cfg(feature = "milk-native-converter")]
struct ConverterOutputCache {
    budget_bytes: usize,
    total_bytes: usize,
    /// Ordered least- to most-recently-used (front = LRU, back = MRU).
    entries: Vec<ConverterCacheEntry>,
}

#[cfg(feature = "milk-native-converter")]
impl ConverterOutputCache {
    fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            total_bytes: 0,
            entries: Vec::new(),
        }
    }

    /// Look up by digest, then CONFIRM the full key matches before returning
    /// (collision verification). On a hit the entry is promoted to
    /// most-recently-used.
    fn get(&mut self, digest: u64, key: &ConverterInput) -> Option<String> {
        let idx = self
            .entries
            .iter()
            .position(|entry| entry.digest == digest && entry.key == *key)?;
        let entry = self.entries.remove(idx);
        let glsl = entry.glsl.clone();
        self.entries.push(entry);
        Some(glsl)
    }

    /// Insert (or replace) an entry, then evict LRU entries until the retained
    /// total is within budget (always keeping at least the just-inserted entry).
    fn insert(&mut self, digest: u64, key: ConverterInput, glsl: String) {
        let bytes = glsl.len();
        if let Some(idx) = self
            .entries
            .iter()
            .position(|entry| entry.digest == digest && entry.key == key)
        {
            self.total_bytes -= self.entries[idx].bytes;
            self.entries.remove(idx);
        }
        self.entries.push(ConverterCacheEntry {
            digest,
            key,
            glsl,
            bytes,
        });
        self.total_bytes += bytes;
        self.evict_to_budget();
    }

    fn evict_to_budget(&mut self) {
        while self.total_bytes > self.budget_bytes && self.entries.len() > 1 {
            let evicted = self.entries.remove(0);
            self.total_bytes -= evicted.bytes;
        }
    }
}

/// Stable 64-bit digest of a converter input, used as the cache's fast index.
/// Digest equality is only a candidate match — the full key is always compared on
/// lookup, so a hash collision can never return the wrong GLSL.
#[cfg(feature = "milk-native-converter")]
fn converter_input_digest(key: &ConverterInput) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// The process-wide converter-output cache. The conversion helpers are free
/// functions (not methods on the `!Send` renderer) and preset import may run on
/// rayon worker threads, so the cache is guarded by its own `Mutex`.
#[cfg(feature = "milk-native-converter")]
fn converter_output_cache() -> &'static std::sync::Mutex<ConverterOutputCache> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<ConverterOutputCache>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        std::sync::Mutex::new(ConverterOutputCache::new(
            CONVERTER_OUTPUT_CACHE_BUDGET_BYTES,
        ))
    })
}

/// Return the cached GLSL for `key`, if present (promoting it to MRU). A poisoned
/// lock degrades to a miss (the caller re-runs the converter) rather than panics.
#[cfg(feature = "milk-native-converter")]
fn converter_cache_get(digest: u64, key: &ConverterInput) -> Option<String> {
    converter_output_cache()
        .lock()
        .ok()
        .and_then(|mut cache| cache.get(digest, key))
}

/// Store the fresh converter output for `key`. A poisoned lock skips the insert.
#[cfg(feature = "milk-native-converter")]
fn converter_cache_insert(digest: u64, key: ConverterInput, glsl: String) {
    if let Ok(mut cache) = converter_output_cache().lock() {
        cache.insert(digest, key, glsl);
    }
}

/// Extract HLSL function definitions from `before`, returning `(func_defs, rest)`.
/// `func_defs` contains only function definitions; `rest` contains variable
/// declarations, `static const` initialisers, etc.
fn extract_function_defs(before: &str) -> (String, String) {
    let lines: Vec<&str> = before.lines().collect();
    let mut func_defs = String::new();
    let mut rest = String::new();
    let mut i = 0;
    while i < lines.len() {
        let line_no_comments = strip_comments(lines[i]);
        let t = line_no_comments.trim();
        // A function signature line: has `(` and will be followed by (or contains) `{`
        // not preceded by `=`.
        let is_sig = t.contains('(')
            && (
                // Same-line open brace (not an initializer)
                (t.contains('{') && !t.contains("= {"))
                // Next non-empty line is `{` (multi-line function def)
                || lines[i + 1..].iter().find(|l| !l.trim().is_empty())
                    .map(|l| {
                        let lt = l.trim();
                        lt.starts_with('{') && !lt.contains("= {")
                    })
                    .unwrap_or(false)
            );
        if is_sig {
            let start = i;
            let mut depth = 0i32;
            let mut opened = false;
            while i < lines.len() {
                let count_line = strip_comments(lines[i]);
                for c in count_line.chars() {
                    if c == '{' {
                        depth += 1;
                        opened = true;
                    }
                    if c == '}' {
                        depth -= 1;
                    }
                }
                i += 1;
                if opened && depth == 0 {
                    break;
                }
            }
            for j in start..i {
                func_defs.push_str(lines[j]);
                func_defs.push('\n');
            }
        } else {
            rest.push_str(lines[i]);
            rest.push('\n');
            i += 1;
        }
    }
    (func_defs, rest)
}

/// Returns true if `before` contains HLSL helper function definitions.
fn before_has_function_defs(before: &str) -> bool {
    let (func_defs, _) = extract_function_defs(before);
    !func_defs.trim().is_empty()
}

fn comp_gamma_postlude(_body: &str) -> &'static str {
    ""
}

/// Strip HLSL-only `sampler X;` lines from `src`, returning (cleaned, custom_names).
/// Standard samplers are dropped silently; custom (user-texture) ones are collected
/// so callers can alias their references to a standard sampler.
///
/// `pub(crate)` so the renderer's blur-level detector (P2-VIS-017) can collapse a
/// body through the SAME rewrite the compile path uses, guaranteeing the detector
/// sees the identical `sampler_blurN` spelling the compiled shader samples (no
/// mode-prefixed under-detection drift).
pub(crate) fn normalize_milkdrop_sampler_variants(src: &str) -> String {
    let mut out = src.to_string();
    out = out.replace("sampler_pw_noise_lq_lite", "sampler_noise_lq_lite");
    out = out.replace("sampler_pw_noise_mq", "sampler_noise_mq");
    out = out.replace("sampler_pw_noise_hq", "sampler_noise_hq");
    out = out.replace("sampler_noise_hq_lite", "sampler_noise_lq_lite");
    out = out.replace("sampler_pw_noise_lq", "sampler_noise_lq");
    out = out.replace("sampler_pw_noisevol", "sampler_noisevol");
    for mode in ["fw_", "fc_", "pc_"] {
        out = out.replace(&format!("sampler_{mode}noise"), "sampler_noise");
    }
    for mode in ["fw_", "fc_", "pw_", "pc_"] {
        out = out.replace(&format!("sampler_{mode}blur"), "sampler_blur");
    }
    out
}

#[cfg(feature = "milk-native-converter")]
fn strip_hlsl_sampler_decls(src: &str) -> (String, Vec<String>) {
    let normalized = normalize_milkdrop_sampler_variants(src);
    let mut out = String::with_capacity(src.len());
    let mut custom_samplers: Vec<String> = Vec::new();
    for raw_line in normalized.lines() {
        let line = raw_line.trim_start();
        let after_samp = line.strip_prefix("sampler ").unwrap_or("");
        if !after_samp.is_empty()
            && !after_samp.starts_with("2D")
            && !after_samp.starts_with("3D")
            && !after_samp.starts_with("Cube")
        {
            let name = after_samp
                .split([';', ' ', '\t'])
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() && !MILK_STANDARD_SAMPLERS.contains(&name) {
                custom_samplers.push(name.to_string());
            }
            continue; // drop sampler line
        }
        out.push_str(raw_line);
        out.push('\n');
    }
    (out, custom_samplers)
}

/// Strip HLSL sampler declarations from a combined before+inner body for the
/// pure-Rust GLSL fallback path, returning (stripped_body, custom_sampler_names).
///
/// Handles:
///   - Multi-line `sampler X = sampler_state { ... };` DX9 blocks
///   - Single-line `sampler X;` SM3 style
///   - `sampler2D X;` / `uniform sampler2D X;` inside function body (not file-scope)
///
/// The returned `custom_sampler_names` should be aliased to the shared custom
/// sampler fallback by
/// the caller (via replace_word) before passing to `hlsl_to_glsl_body`.
fn strip_and_alias_hlsl_samplers(src: &str) -> (String, Vec<String>) {
    let normalized = normalize_milkdrop_sampler_variants(src);
    let lines: Vec<&str> = normalized.lines().collect();
    let mut out = String::with_capacity(src.len());
    let mut custom: Vec<String> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i];
        let t = raw.trim_start();
        // Multi-line block: `sampler X = sampler_state {`
        if t.starts_with("sampler ")
            && !t.starts_with("sampler2D")
            && !t.starts_with("sampler3D")
            && !t.starts_with("samplerCube")
            && t.contains("sampler_state")
        {
            let name = t
                .strip_prefix("sampler ")
                .unwrap_or("")
                .split(['=', ' ', '\t'])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() && !MILK_STANDARD_SAMPLERS.contains(&name.as_str()) {
                custom.push(name);
            }
            // Skip until closing `};`
            while i < lines.len() {
                let cl = lines[i].trim();
                i += 1;
                if cl == "};" || (cl.ends_with("};") && !cl.contains('=')) {
                    break;
                }
            }
            continue;
        }
        // Single-line: `sampler X;` (no = sampler_state)
        if t.starts_with("sampler ")
            && !t.starts_with("sampler2D")
            && !t.starts_with("sampler3D")
            && !t.starts_with("samplerCube")
            && t.contains(';')
            && !t.contains('=')
        {
            let after = t.strip_prefix("sampler ").unwrap_or("");
            let name = after
                .split([';', ' ', '\t'])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() && !MILK_STANDARD_SAMPLERS.contains(&name.as_str()) {
                custom.push(name);
            }
            i += 1;
            continue;
        }
        // `uniform sampler2D X;` or `sampler2D X;` — sampler uniforms inside body
        // (samplers must be file-scope uniforms in GLSL, not body-local declarations)
        let nouni = t.trim_start_matches("uniform").trim_start();
        if (nouni.starts_with("sampler2D ")
            || nouni.starts_with("sampler3D ")
            || nouni.starts_with("samplerCube "))
            && nouni.contains(';')
            && !nouni.contains('(')
        {
            // Extract the name so we can alias its usages
            let after_ty = if let Some(s) = nouni
                .strip_prefix("sampler2D ")
                .or_else(|| nouni.strip_prefix("sampler3D "))
                .or_else(|| nouni.strip_prefix("samplerCube "))
            {
                s
            } else {
                ""
            };
            let name = after_ty
                .split([';', ' ', '\t', '='])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() && !MILK_STANDARD_SAMPLERS.contains(&name.as_str()) {
                custom.push(name);
            }
            i += 1;
            continue;
        }
        // `#define MACRO sampler_name` — expand the macro to the shared custom
        // sampler fallback.
        // Handles patterns like `#define MYSAMP sampler_devboxb` where the author
        // aliases an undeclared custom sampler to a macro name.
        if t.starts_with("#define ") {
            let rest = t.strip_prefix("#define ").unwrap_or("").trim();
            let mut parts = rest.splitn(2, [' ', '\t']);
            let macro_name = parts.next().unwrap_or("").trim();
            let macro_value = parts
                .next()
                .unwrap_or("")
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("");
            // If the value looks like a sampler (starts with "sampler") and is NOT a standard
            // sampler, record both the macro name and the value for aliasing.
            if !macro_name.is_empty()
                && macro_value.starts_with("sampler")
                && !MILK_STANDARD_SAMPLERS.contains(&macro_value)
                && !MILK_STANDARD_SAMPLERS.contains(&macro_name)
            {
                custom.push(macro_name.to_string());
                if !macro_value.is_empty() {
                    custom.push(macro_value.to_string());
                }
                i += 1;
                continue; // strip the #define line
            }
        }
        out.push_str(raw);
        out.push('\n');
        i += 1;
    }
    (out, custom)
}

/// Return custom/user-image sampler identifiers without discarding their names.
///
/// This is the metadata half of native named-texture support. It understands the
/// HLSL declaration forms handled by the compile path plus undeclared samplers used
/// directly as the first argument of `tex2D`/`tex3D`/`texture` calls. Built-in
/// MilkDrop feedback, blur, and noise samplers are excluded. Order is first-seen and
/// stable so the renderer can assign deterministic texture-array layers.
pub fn custom_sampler_names(src: &str) -> Vec<String> {
    let (_, declared) = strip_and_alias_hlsl_samplers(src);
    let normalized = normalize_milkdrop_sampler_variants(src);
    let without_comments = strip_comments(&normalized);
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for name in declared
        .into_iter()
        .chain(texture_first_arg_identifiers(&without_comments))
    {
        if !name.is_empty()
            && !is_builtin_sampler_name(&name)
            && seen.insert(name.to_ascii_lowercase())
        {
            names.push(name);
        }
    }
    names
}

fn texture_first_arg_identifiers(src: &str) -> Vec<String> {
    let bytes = src.as_bytes();
    let mut names = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let function = &src[start..i];
            if !matches!(function, "tex2D" | "tex2d" | "tex3D" | "tex3d" | "texture") {
                continue;
            }
            let mut cursor = i;
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            if cursor >= bytes.len() || bytes[cursor] != b'(' {
                continue;
            }
            cursor += 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let arg_start = cursor;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
            {
                cursor += 1;
            }
            if cursor > arg_start {
                names.push(src[arg_start..cursor].to_string());
            }
        } else {
            i += 1;
        }
    }
    names
}

fn is_builtin_sampler_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if matches!(lower.as_str(), "sampler2d" | "sampler3d" | "samplercube") {
        return true;
    }
    if MILK_STANDARD_SAMPLERS
        .iter()
        .chain(MILKDROP_SAMPLERS.iter())
        .any(|known| known.eq_ignore_ascii_case(&lower))
    {
        return true;
    }
    // Sampling-mode variants are normalized by the compile path even when an exact
    // spelling is absent from the fixed binding table.
    let base = lower.strip_prefix("sampler_").unwrap_or(&lower);
    let base = ["fw_", "fc_", "pw_", "pc_"]
        .iter()
        .find_map(|prefix| base.strip_prefix(prefix))
        .unwrap_or(base);
    base == "main"
        || base.starts_with("main_")
        || base == "noise"
        || base.starts_with("noise_")
        || base.starts_with("noisevol")
        || base == "blur"
        || base.starts_with("blur1")
        || base.starts_with("blur2")
        || base.starts_with("blur3")
}

/// Rewrite custom sampler identifiers through an explicit renderer-provided map.
/// Companion `texsize_<asset>` identifiers are rewritten to the target sampler's
/// `texsize_<asset>` spelling as well. The scan is token-based and simultaneous, so
/// mappings cannot cascade through one another.
///
/// OjoDrop's legacy compile path calls this with every custom sampler mapped to
/// `sampler_noise_lq`. A named-texture renderer instead maps identities to its fixed
/// atlas/array sampler(s), while retaining [`custom_sampler_names`] as the layer
/// manifest.
pub fn rewrite_custom_sampler_identifiers(
    src: &str,
    replacements: &HashMap<String, String>,
) -> String {
    let mut tokens = replacements.clone();
    for (from, to) in replacements {
        if let (Some(from_base), Some(to_base)) =
            (from.strip_prefix("sampler_"), to.strip_prefix("sampler_"))
        {
            tokens.insert(format!("texsize_{from_base}"), format!("texsize_{to_base}"));
        }
    }

    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let token = &src[start..i];
            out.push_str(tokens.get(token).map(String::as_str).unwrap_or(token));
        } else {
            let ch = src[i..].chars().next().expect("i is in bounds");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// One custom-sampler identity and its fixed named-texture-atlas location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedTextureRewriteBinding {
    pub sampler_name: String,
    pub layer: u32,
    pub point_filter: bool,
    pub clamp: bool,
}

/// Rewrite custom `tex2D`/GLSL `texture` calls to OjoDrop's two reserved atlas
/// samplers without increasing the fixed MilkDrop binding count.
///
/// The atlas is a 4×4 grid with a two-texel replicated gutter around each image.
/// UV wrap/clamp is encoded in the rewritten coordinate, so both reserved samplers
/// may use clamp addressing; separate linear/point samplers preserve the HLSL
/// `fw/fc` versus `pw/pc` filter mode. Call this on the raw body before the normal
/// HLSL/GLSL conversion path. Remaining custom declarations are harmlessly stripped
/// by that path, while their call sites retain real layer identity.
pub fn rewrite_custom_sampler_calls_for_atlas(
    src: &str,
    bindings: &[NamedTextureRewriteBinding],
    layer_size: u32,
) -> String {
    if bindings.is_empty() {
        return src.to_string();
    }
    let layer_size = layer_size.clamp(1, 4096);
    let by_name: HashMap<String, &NamedTextureRewriteBinding> = bindings
        .iter()
        .map(|binding| (binding.sampler_name.to_ascii_lowercase(), binding))
        .collect();
    let mut out = rewrite_custom_sampler_calls_inner(src, &by_name, layer_size);
    // The atlas normalizes every layer to `layer_size`, which matches the existing
    // 256² texsize constant in the canonical profile. Rewrite custom texsize symbols
    // independently from sampler call rewriting so declarations remain parseable.
    for binding in bindings {
        if let Some(base) = binding.sampler_name.strip_prefix("sampler_") {
            out = replace_word(&out, &format!("texsize_{base}"), "texsize_noise_mq");
        }
    }
    out
}

fn rewrite_custom_sampler_calls_inner(
    src: &str,
    by_name: &HashMap<String, &NamedTextureRewriteBinding>,
    layer_size: u32,
) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 128);
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let ident_start = i;
            let mut ident_end = i + 1;
            while ident_end < bytes.len()
                && (bytes[ident_end].is_ascii_alphanumeric() || bytes[ident_end] == b'_')
            {
                ident_end += 1;
            }
            let function = &src[ident_start..ident_end];
            let supported = matches!(function, "tex2D" | "tex2d" | "texture");
            let mut open = ident_end;
            while open < bytes.len() && bytes[open].is_ascii_whitespace() {
                open += 1;
            }
            if supported && open < bytes.len() && bytes[open] == b'(' {
                if let Some((sampler_arg, coord, consumed)) = split_two_args(&src[open + 1..]) {
                    let sampler_name = combined_sampler_identifier(&sampler_arg);
                    if let Some(binding) = by_name.get(&sampler_name.to_ascii_lowercase()) {
                        let coord = rewrite_custom_sampler_calls_inner(&coord, by_name, layer_size);
                        let target = if binding.point_filter {
                            "sampler_named_point"
                        } else {
                            "sampler_named_linear"
                        };
                        let transformed = atlas_uv_expression(
                            &coord,
                            binding.layer,
                            binding.clamp,
                            layer_size,
                            function == "texture",
                        );
                        out.push_str(if function == "texture" {
                            "texture("
                        } else {
                            "tex2D("
                        });
                        out.push_str(target);
                        out.push_str(", ");
                        out.push_str(&transformed);
                        out.push(')');
                        i = open + 1 + consumed;
                        continue;
                    }
                }
            }
        }
        let ch = src[i..].chars().next().expect("i is in bounds");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn combined_sampler_identifier(argument: &str) -> &str {
    let argument = argument.trim();
    for prefix in ["sampler2D(", "sampler3D("] {
        if let Some(inner) = argument
            .strip_prefix(prefix)
            .and_then(|v| v.strip_suffix(')'))
        {
            return inner.split(',').next().unwrap_or("").trim();
        }
    }
    argument
}

fn atlas_uv_expression(
    coord: &str,
    layer: u32,
    clamp: bool,
    layer_size: u32,
    glsl: bool,
) -> String {
    const GRID: u32 = 4;
    const GUTTER: u32 = 2;
    let stride = layer_size + GUTTER * 2;
    let atlas_size = stride * GRID;
    let column = layer % GRID;
    let row = layer / GRID;
    let uv = if glsl {
        if clamp {
            format!("clamp(({coord}).xy, vec2(0.0), vec2(1.0))")
        } else {
            format!("fract(({coord}).xy)")
        }
    } else if clamp {
        format!("saturate(({coord}).xy)")
    } else {
        format!("frac(({coord}).xy)")
    };
    let vec2 = if glsl { "vec2" } else { "float2" };
    let origin_x = column * stride + GUTTER;
    let origin_y = row * stride + GUTTER;
    format!(
        "({vec2}({:.1}, {:.1}) + ({uv}) * {:.1}) / {:.1}",
        origin_x as f32 + 0.5,
        origin_y as f32 + 0.5,
        layer_size.saturating_sub(1) as f32,
        atlas_size as f32,
    )
}

pub fn hlsl_milk_body_to_naga_with_named_textures(
    body: &str,
    bindings: &[NamedTextureRewriteBinding],
    layer_size: u32,
) -> String {
    hlsl_milk_body_to_naga(&rewrite_custom_sampler_calls_for_atlas(
        body, bindings, layer_size,
    ))
}

pub fn hlsl_milk_warp_body_to_naga_with_named_textures(
    body: &str,
    bindings: &[NamedTextureRewriteBinding],
    layer_size: u32,
) -> String {
    hlsl_milk_warp_body_to_naga(&rewrite_custom_sampler_calls_for_atlas(
        body, bindings, layer_size,
    ))
}

pub fn glsl_milk_body_to_naga_with_named_textures(
    body: &str,
    bindings: &[NamedTextureRewriteBinding],
    layer_size: u32,
) -> String {
    glsl_milk_body_to_naga(&rewrite_custom_sampler_calls_for_atlas(
        body, bindings, layer_size,
    ))
}

pub fn glsl_milk_warp_body_to_naga_with_named_textures(
    body: &str,
    bindings: &[NamedTextureRewriteBinding],
    layer_size: u32,
) -> String {
    glsl_milk_warp_body_to_naga(&rewrite_custom_sampler_calls_for_atlas(
        body, bindings, layer_size,
    ))
}

/// Returns true if `src` contains a const-typed array declaration (`const TYPE name[`).
#[cfg(feature = "milk-native-converter")]
fn contains_const_array(src: &str) -> bool {
    // Fast scan: look for "const " followed eventually by "[" on the same line.
    for line in src.lines() {
        let t = line.trim_start();
        if t.starts_with("const ") {
            if t.contains('[') {
                return true;
            }
        }
    }
    false
}

// ===========================================================================
// Conservative GLSL type inference + type-mismatch repairs
//
// HLSL is permissive where GLSL/naga is strict: it allows `vec < scalar`
// (componentwise), bool-as-number, and scalar→vector promotion. naga rejects
// these ("Operation Less can't work with vec3 and scalar", …). We infer the type
// of each operand from a symbol table (preamble uniforms + scanned local/param
// declarations + known builtin return types) and rewrite the offending forms.
// The inference is CONSERVATIVE: anything it cannot type confidently is `Unknown`,
// and transforms never fire on `Unknown` — so valid shaders are left untouched.
// ===========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GTy {
    F,      // float scalar (ints folded in here — we only care about float context)
    V(u8),  // float vector, width 2..4
    B,      // bool scalar
    BV(u8), // bool vector, width 2..4
    Unknown,
}

fn gty_width(t: GTy) -> u8 {
    match t {
        GTy::F | GTy::B => 1,
        GTy::V(n) | GTy::BV(n) => n.min(4),
        GTy::Unknown => 0,
    }
}

type TypeTable = HashMap<String, GTy>;

/// Seed the symbol table with names the FS preamble always provides.
fn seed_known_types(t: &mut TypeTable) {
    let v4 = [
        "texsize",
        "aspect",
        "slow_roam_cos",
        "roam_cos",
        "slow_roam_sin",
        "roam_sin",
        "rand_frame",
        "rand_start",
        "rand_preset",
        "vColor",
        "vDecay",
        "texsize_noise_lq",
        "texsize_noise_mq",
        "texsize_noise_hq",
        "texsize_noise_lq_lite",
        "texsize_noise_hq_lite",
        "texsize_noisevol_lq",
        "texsize_noisevol_hq",
    ];
    for n in v4 {
        t.insert(n.to_string(), GTy::V(4));
    }
    for n in ["uv", "uv_orig", "vUv", "vWarpUv"] {
        t.insert(n.to_string(), GTy::V(2));
    }
    for n in ["ret", "hue_shader"] {
        t.insert(n.to_string(), GTy::V(3));
    }
    for i in 1..=32 {
        t.insert(format!("q{i}"), GTy::F);
    }
    for n in [
        "time",
        "fps",
        "frame",
        "progress",
        "bass",
        "mid",
        "treb",
        "vol",
        "bass_att",
        "mid_att",
        "treb_att",
        "vol_att",
        "fShader",
        "gammaAdj",
        "echo_zoom",
        "echo_alpha",
        "echo_orientation",
        "blur1_min",
        "blur1_max",
        "blur2_min",
        "blur2_max",
        "blur3_min",
        "blur3_max",
        "scale1",
        "scale2",
        "scale3",
        "bias1",
        "bias2",
        "bias3",
        "brighten",
        "darken",
        "solarize",
        "invert",
        "rad",
        "ang",
        "PI",
        "M_PI",
        "M_PI_2",
        "M_INV_PI",
        "M_INV_PI_2",
    ] {
        t.insert(n.to_string(), GTy::F);
    }
}

/// Map a GLSL type keyword to a GTy (vectors only — scalars/ints fold to F).
fn keyword_gty(kw: &str) -> Option<GTy> {
    Some(match kw {
        "float" | "int" | "uint" => GTy::F,
        "vec2" | "ivec2" | "uvec2" => GTy::V(2),
        "vec3" | "ivec3" | "uvec3" => GTy::V(3),
        "vec4" | "ivec4" | "uvec4" => GTy::V(4),
        "bool" => GTy::B,
        "bvec2" => GTy::BV(2),
        "bvec3" => GTy::BV(3),
        "bvec4" => GTy::BV(4),
        _ => return None,
    })
}

/// Scan the whole shader text for variable/parameter declarations and function
/// return types, adding them to the table.
/// Insert a name→type, but if it already maps to a DIFFERENT type, collapse to
/// Unknown. Overloaded functions reuse parameter names (`saturate(float x)` and
/// `saturate(vec4 x)` both bind `x`); without this the table would mistype `x` and
/// the repairs would corrupt valid preamble helpers.
fn ins_ty(t: &mut TypeTable, name: String, g: GTy) {
    match t.get(&name) {
        Some(&prev) if prev != g => {
            t.insert(name, GTy::Unknown);
        }
        Some(_) => {}
        None => {
            t.insert(name, g);
        }
    }
}

fn collect_decl_types(src: &str, t: &mut TypeTable) {
    for raw in src.lines() {
        let line = raw.trim();
        // Function definition: `<ret> <name>(<params>) {` — record return type + params.
        if let Some(paren) = line.find('(') {
            let head = line[..paren].trim();
            let mut it = head.split_whitespace();
            if let (Some(rty), Some(name)) = (it.next(), it.next()) {
                if it.next().is_none() {
                    if let Some(g) = keyword_gty(rty) {
                        // looks like a function signature (next char region is a param list)
                        let after = &line[paren + 1..];
                        if after.contains(')') || raw.trim_end().ends_with(',') {
                            let nm = name
                                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                                .to_string();
                            ins_ty(t, nm, g);
                        }
                    }
                }
                // params inside the parens
                if let Some(close) = line[paren..].find(')') {
                    let params = &line[paren + 1..paren + close];
                    for p in params.split(',') {
                        let mut pi = p.split_whitespace();
                        if let (Some(pty), Some(pn)) = (pi.next(), pi.next()) {
                            if let Some(g) = keyword_gty(pty) {
                                ins_ty(
                                    t,
                                    pn.trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                                        .to_string(),
                                    g,
                                );
                            }
                        }
                    }
                }
            }
        }
        // Plain declaration: `<type> a, b = …, c;`
        let mut words = line.split_whitespace();
        if let Some(first) = words.next() {
            let first = first.trim_start_matches("const").trim();
            if let Some(g) = keyword_gty(first) {
                let rest = line[line.find(first).unwrap_or(0) + first.len()..].trim();
                // not a function def (handled above) and not a cast/constructor call
                if !rest.starts_with('(') {
                    for item in split_top_level_commas(rest.trim_end_matches(';')) {
                        let nm: String = item
                            .trim()
                            .chars()
                            .take_while(|c| c.is_alphanumeric() || *c == '_')
                            .collect();
                        if !nm.is_empty() {
                            ins_ty(t, nm, g);
                        }
                    }
                }
            }
            collect_decl_type_segments(raw, t);
        }
    }

    fn collect_decl_type_segments(line: &str, table: &mut TypeTable) {
        let b = line.as_bytes();
        let mut depth = 0i32;
        let mut start = 0usize;
        for i in 0..b.len() {
            match b[i] {
                b'(' | b'[' => depth += 1,
                b')' | b']' => depth -= 1,
                b';' | b'{' | b'}' if depth == 0 => {
                    collect_decl_type_seg(&line[start..i], table);
                    start = i + 1;
                }
                _ => {}
            }
        }
        collect_decl_type_seg(&line[start..], table);
    }

    fn collect_decl_type_seg(seg: &str, table: &mut TypeTable) {
        let core = seg.trim();
        let core = core.rsplit('{').next().unwrap_or(core).trim();
        let mut it = core.splitn(2, char::is_whitespace);
        let (Some(kw), Some(rest)) = (it.next(), it.next()) else {
            return;
        };
        let Some(g) = keyword_gty(kw) else {
            return;
        };
        if rest.trim_start().starts_with('(') {
            return;
        }
        let names = rest.split('=').next().unwrap_or(rest);
        for item in split_top_level_commas(names) {
            let ident: String = item
                .trim()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() {
                ins_ty(table, ident, g);
            }
        }
    }
}

/// Builtin function return types relevant to inference.
fn builtin_ret(name: &str, arg0: GTy) -> Option<GTy> {
    Some(match name {
        "texture" | "texture2D" | "texture3D" | "textureLod" | "texelFetch" => GTy::V(4),
        "GetMain" | "GetPixel" | "GetBlur1" | "GetBlur2" | "GetBlur3" => GTy::V(3),
        "lum" => match arg0 {
            GTy::F | GTy::V(2) => GTy::F,
            GTy::V(3) | GTy::V(4) => GTy::V(3),
            _ => return None,
        },
        "length" | "distance" | "dot" | "determinant" | "float" => GTy::F,
        "cross" => GTy::V(3),
        "vec2" => GTy::V(2),
        "vec3" => GTy::V(3),
        "vec4" => GTy::V(4),
        // shape-preserving (return the type of the first argument). Includes the
        // OVERLOADED preamble helpers (safeRecip/saturate) — they must be resolved
        // here, not via the single-type symbol table (which would record only the
        // last overload, e.g. vec4 safeRecip, and mistype `safeRecip(float)`).
        "normalize" | "abs" | "floor" | "ceil" | "fract" | "sin" | "cos" | "tan" | "asin"
        | "acos" | "atan" | "exp" | "exp2" | "log" | "log2" | "sqrt" | "inversesqrt" | "sign"
        | "radians" | "degrees" | "saturate" | "safeRecip" | "round" | "trunc" | "mix"
        | "log10" | "clamp" | "min" | "max" | "mod" | "pow" | "reflect" | "smoothstep" | "step"
        | "neg" => {
            if arg0 == GTy::Unknown {
                return None;
            }
            arg0
        }
        _ => return None,
    })
}

/// Strip one layer of fully-enclosing parentheses, if present.
fn strip_enclosing_parens(s: &str) -> &str {
    let s = s.trim();
    if !s.starts_with('(') || !s.ends_with(')') {
        return s;
    }
    let b = s.as_bytes();
    let mut depth = 0i32;
    for i in 0..b.len() {
        match b[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return if i == b.len() - 1 {
                        strip_enclosing_parens(&s[1..b.len() - 1])
                    } else {
                        s
                    };
                }
            }
            _ => {}
        }
    }
    s
}

/// Precedence groups, lowest first.
const PREC: &[&[&str]] = &[
    &["||"],
    &["&&"],
    &["==", "!="],
    &["<=", ">=", "<", ">"],
    &["+", "-"],
    &["*", "/", "%"],
];

/// Split at the first top-level binary operator of the lowest present precedence.
/// Returns (left, op, right). Handles unary +/-/! (skips them).
fn split_binop(s: &str) -> Option<(&str, &'static str, &str)> {
    let b = s.as_bytes();
    let n = b.len();
    let is_operand_end =
        |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b')' || c == b']' || c == b'.';
    for group in PREC {
        let mut depth = 0i32;
        let mut i = 0;
        while i < n {
            let c = b[i];
            if c == b'(' || c == b'[' {
                depth += 1;
            } else if c == b')' || c == b']' {
                depth -= 1;
            } else if depth == 0 {
                for op in *group {
                    let ob = op.as_bytes();
                    if i + ob.len() <= n && &b[i..i + ob.len()] == ob {
                        let prev = (1..=i)
                            .rev()
                            .map(|k| b[k - 1])
                            .find(|c| !c.is_ascii_whitespace());
                        // +/- is binary only after an operand (else unary sign)
                        if (*op == "+" || *op == "-") && prev.map_or(true, |c| !is_operand_end(c)) {
                            continue;
                        }
                        let nextc = b.get(i + ob.len()).copied();
                        // `<`/`>` followed by `=`/`<` is `<=`/`>=`/`<<` (handled elsewhere)
                        if matches!(*op, "<" | ">") && (nextc == Some(b'=') || nextc == Some(c)) {
                            continue;
                        }
                        if prev == Some(b'=') && (*op == "<" || *op == ">") {
                            continue;
                        }
                        return Some((&s[..i], op, &s[i + ob.len()..]));
                    }
                }
            }
            i += 1;
        }
    }
    None
}

/// Infer the type of a GLSL expression. Conservative — returns Unknown when unsure.
fn infer_ty(expr: &str, t: &TypeTable) -> GTy {
    let e = strip_enclosing_parens(expr);
    if e.is_empty() {
        return GTy::Unknown;
    }
    // binary operator?
    if let Some((l, op, r)) = split_binop(e) {
        let lt = infer_ty(l, t);
        let rt = infer_ty(r, t);
        return match op {
            "<" | ">" | "<=" | ">=" | "==" | "!=" => {
                let w = gty_width(lt).max(gty_width(rt));
                if w > 1 {
                    GTy::BV(w)
                } else if lt == GTy::Unknown && rt == GTy::Unknown {
                    GTy::Unknown
                } else {
                    GTy::B
                }
            }
            "&&" | "||" => GTy::B,
            _ => {
                // arithmetic: width = max, float base
                let w = gty_width(lt).max(gty_width(rt));
                match w {
                    0 => GTy::Unknown,
                    1 => GTy::F,
                    _ => GTy::V(w),
                }
            }
        };
    }
    // leading unary
    let e2 = e.trim();
    if let Some(rest) = e2.strip_prefix('!') {
        let inner = infer_ty(rest, t);
        return match inner {
            GTy::BV(n) | GTy::V(n) => GTy::BV(n),
            GTy::Unknown => GTy::Unknown,
            _ => GTy::B,
        };
    }
    if let Some(rest) = e2.strip_prefix('-').or_else(|| e2.strip_prefix('+')) {
        return infer_ty(rest, t);
    }
    // ternary  c ? a : b
    if let Some(q) = find_top_level_char(e2, b'?') {
        if let Some(colon) = find_top_level_char(&e2[q + 1..], b':') {
            let a = &e2[q + 1..q + 1 + colon];
            let at = infer_ty(a, t);
            if at != GTy::Unknown {
                return at;
            }
            return infer_ty(&e2[q + 1 + colon + 1..], t);
        }
    }
    // function call: ident(...)
    if e2.ends_with(')') {
        if let Some(open) = e2.find('(') {
            let name = &e2[..open];
            if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
                let args = &e2[open + 1..e2.len() - 1];
                let arg0 = split_top_level_commas(args)
                    .into_iter()
                    .next()
                    .unwrap_or_default();
                let arg0ty = infer_ty(arg0.trim(), t);
                if let Some(g) = builtin_ret(name, arg0ty) {
                    return g;
                }
                if let Some(&g) = t.get(name) {
                    return g; // user function return type
                }
                return GTy::Unknown;
            }
        }
    }
    // swizzle / member access: base.SWIZ
    if let Some(dot) = e2.rfind('.') {
        let after = &e2[dot + 1..];
        if !after.is_empty()
            && after
                .chars()
                .all(|c| matches!(c, 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a'))
        {
            let base = infer_ty(&e2[..dot], t);
            let w = after.chars().count();
            if w > 4 {
                return GTy::Unknown;
            }
            let w = w as u8;
            return match base {
                GTy::V(_) | GTy::F => {
                    if w > 1 {
                        GTy::V(w)
                    } else {
                        GTy::F
                    }
                }
                GTy::BV(_) | GTy::B => {
                    if w > 1 {
                        GTy::BV(w)
                    } else {
                        GTy::B
                    }
                }
                GTy::Unknown => GTy::Unknown,
            };
        }
    }
    // numeric literal
    let c0 = e2.as_bytes()[0];
    if c0.is_ascii_digit() || (c0 == b'.' && e2.len() > 1 && e2.as_bytes()[1].is_ascii_digit()) {
        return GTy::F;
    }
    // identifier (possibly with [index] suffix → component)
    if let Some(br) = e2.find('[') {
        let base = infer_ty(&e2[..br], t);
        return match base {
            GTy::V(_) => GTy::F,
            other => other,
        };
    }
    if e2.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return t.get(e2).copied().unwrap_or(GTy::Unknown);
    }
    GTy::Unknown
}

fn find_top_level_char(s: &str, target: u8) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            c if c == target && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Collect each user function's parameter types: `vec2 f(vec2 a, float b)` →
/// f → [V(2), F]. Used to promote scalar call arguments to vector parameters.
fn collect_fn_sigs(src: &str) -> HashMap<String, Vec<GTy>> {
    let mut sigs: HashMap<String, Vec<GTy>> = HashMap::new();
    for line in src.lines() {
        let line = line.trim();
        let Some(open) = line.find('(') else { continue };
        let Some(close_rel) = line[open..].find(')') else {
            continue;
        };
        let head = line[..open].trim();
        let mut hp = head.split_whitespace();
        let (Some(rty), Some(name)) = (hp.next(), hp.next()) else {
            continue;
        };
        if hp.next().is_some() || keyword_gty(rty).is_none() {
            continue; // not a `<type> <name>(` signature
        }
        let name = name.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
        let params = &line[open + 1..open + close_rel];
        if params.trim().is_empty() {
            continue;
        }
        let mut tys = Vec::new();
        let mut ok = true;
        for p in params.split(',') {
            let mut pi = p.split_whitespace();
            match pi.next().and_then(keyword_gty) {
                Some(g) => tys.push(g),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && !tys.is_empty() {
            // conflicting overloads → drop (can't promote against an ambiguous sig)
            if sigs.get(name).map_or(false, |prev| prev != &tys) {
                sigs.remove(name);
            } else {
                sigs.insert(name.to_string(), tys);
            }
        }
    }
    sigs
}

/// PUBLIC ENTRY: build a type table from the whole shader and apply the type-mismatch
/// repairs. Applied at the compile_glsl choke point so it covers every conversion path.
pub fn fix_glsl_vector_types(glsl: &str) -> String {
    let glsl = lower_mutable_q_aliases(glsl);
    let glsl = expand_simple_helper_aliases(&glsl);
    let mut table: TypeTable = HashMap::new();
    seed_known_types(&mut table);
    collect_decl_types(&glsl, &mut table);
    // Re-assert the seeded builtin types: they are canonical MilkDrop names (uv→vec2,
    // ret→vec3, texsize→vec4, …) and must not be collapsed to Unknown by a conflicting
    // FUNCTION PARAMETER of the same name. The preamble's overloaded helpers bind `uv`
    // at two widths — `GetMain(vec2 uv)` AND `GetMain(vec3 uv)` — which made collect_decl_types
    // (via ins_ty conflict→Unknown) poison the global `uv`, silently disabling op-width /
    // assignment-width / relop repair for every uv-expression. Seeds win over params.
    seed_known_types(&mut table);
    let sigs = collect_fn_sigs(&glsl);
    // Only rewrite the PRESET BODY (after the preamble marker). The fixed preamble
    // (UBO block, q #defines, GetBlur/lum/saturate/safeRecip helpers) is correct by
    // construction and must never be touched by a repair pass — two regressions came
    // from transforms mangling it. Inference still uses the whole-shader table.
    const MARK: &str = "// __MILK_BODY__";
    let (prefix, body) = match glsl.find(MARK) {
        Some(pos) => glsl.split_at(pos),
        None => ("", glsl.as_str()),
    };
    // Preset locals may shadow overloaded helper names (`float lum = ...` is
    // common). Give only unambiguous body declarations precedence over the
    // whole-module table. Names declared at multiple widths (for example `tmp`
    // in separate helpers) stay Unknown and continue to use scoped repair.
    for (name, ty) in collect_unambiguous_local_decls(body) {
        table.insert(name, ty);
    }
    let s0 = fix_array_brace_init(body);
    let s0b = join_logical_statements(&s0);
    let s0c = drop_duplicate_bare_decls_same_scope(&s0b);
    let s0d = fix_missing_function_returns(&s0c);
    let s1 = fix_vector_relops(&s0d, &table);
    let s2 = fix_call_arg_promotion(&s1, &table, &sigs);
    let s2b = fix_pow_arg_width(&s2, &table);
    let s2c = fix_mix_calls(&s2b, &table);
    let s2d = fix_op_width_mismatches(&s2c, &table);
    let s3 = fix_assignment_width_mismatches(&s2d, &table);
    let s3b = fix_return_width_mismatches(&s3, &table);
    let s3c = fix_dot_calls(&s3b, &table);
    let s3d = fix_typed_scalar_swizzles(&s3c, &table);
    // Removing a scalar bool swizzle exposes the underlying bool expression. Run
    // assignment coercion once more so HLSL's bool-as-number assignment semantics
    // are preserved (for example `float x = (a > b).x`).
    let s3e = fix_assignment_width_mismatches(&s3d, &table);
    let s3f = fix_int_compound_assignments(&s3e);
    let s3g = fix_numeric_logical_not(&s3f, &table);
    let s3h = fix_numeric_control_conditions(&s3g, &table);
    let s4 = fix_for_constructor_cond(&s3h);
    format!("{prefix}{s4}")
}

fn collect_unambiguous_local_decls(src: &str) -> TypeTable {
    fn record_segment(segment: &str, table: &mut TypeTable) {
        let core = segment.trim();
        if core.is_empty() {
            return;
        }
        let mut words = core.splitn(2, char::is_whitespace);
        let (Some(keyword), Some(rest)) = (words.next(), words.next()) else {
            return;
        };
        let Some(ty) = keyword_gty(keyword) else {
            return;
        };
        let rest = rest.trim_start();
        if rest
            .find('(')
            .is_some_and(|open| rest.find('=').is_none_or(|assignment| open < assignment))
        {
            return;
        }
        for item in split_top_level_commas(rest) {
            let name: String = item
                .trim()
                .chars()
                .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
                .collect();
            if !name.is_empty() {
                ins_ty(table, name, ty);
            }
        }
    }

    let mut table = TypeTable::new();
    for line in src.lines() {
        let bytes = line.as_bytes();
        let mut depth = 0i32;
        let mut start = 0usize;
        for (index, byte) in bytes.iter().copied().enumerate() {
            match byte {
                b'(' | b'[' => depth += 1,
                b')' | b']' => depth -= 1,
                b';' | b'{' | b'}' if depth == 0 => {
                    record_segment(&line[start..index], &mut table);
                    start = index + 1;
                }
                _ => {}
            }
        }
        record_segment(&line[start..], &mut table);
    }
    table.retain(|_, ty| *ty != GTy::Unknown);
    table
}

/// Expand the small class of object-like helper aliases emitted by MilkDrop
/// shaders (`#define MyGet GetPixel`) before type inference. naga expands the
/// macro later, but our conservative inference otherwise sees `MyGet` as an
/// unknown function and cannot repair HLSL vector-to-scalar assignments. Only
/// aliases to fixed, side-effect-free preamble helpers are accepted; all other
/// directives are left verbatim.
fn expand_simple_helper_aliases(glsl: &str) -> String {
    const MARK: &str = "// __MILK_BODY__";
    const HELPERS: &[&str] = &[
        "GetMain", "GetPixel", "GetBlur1", "GetBlur2", "GetBlur3", "lum", "saturate",
    ];
    let Some(pos) = glsl.find(MARK) else {
        return glsl.to_string();
    };
    let (prefix, body) = glsl.split_at(pos);
    let mut aliases: Vec<(String, String)> = Vec::new();
    let mut kept = String::with_capacity(body.len());
    for line in body.lines() {
        let trimmed = line.trim();
        let parsed = trimmed.strip_prefix("#define ").and_then(|rest| {
            let mut words = rest.split_whitespace();
            let alias = words.next()?;
            let target = words.next()?;
            (words.next().is_none()
                && is_plain_ident(alias)
                && is_plain_ident(target)
                && HELPERS.contains(&target))
            .then(|| (alias.to_string(), target.to_string()))
        });
        if let Some(alias) = parsed {
            aliases.push(alias);
        } else {
            kept.push_str(line);
            kept.push('\n');
        }
    }
    for (alias, target) in aliases {
        kept = replace_word(&kept, &alias, &target);
    }
    format!("{prefix}{kept}")
}

fn is_plain_ident(value: &str) -> bool {
    let mut chars = value.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// GLSL expands the preamble's `q1..q32` macros directly onto the read-only
/// `PerFrame` UBO. MilkDrop HLSL nevertheless permits shader-local q mutation
/// (`q25 = 1;`) and expects later reads in the same fragment to observe it. Lower
/// only q registers that are actually written to private per-invocation slots,
/// initialized from the UBO at the start of `main`; mutations are discarded when
/// the fragment ends and can never write through the uniform alias.
fn lower_mutable_q_aliases(glsl: &str) -> String {
    const MARK: &str = "// __MILK_BODY__";
    let Some(mark_pos) = glsl.find(MARK) else {
        return glsl.to_string();
    };
    let body_start = mark_pos + MARK.len();
    let body = &glsl[body_start..];
    let written = written_q_registers(body);
    if written.is_empty() {
        return glsl.to_string();
    }

    let rewritten = rewrite_q_registers(body, &written);
    let Some(main_pos) = rewritten.find("void main()") else {
        // The wrapper generators always emit this spelling. If a caller supplies
        // an incomplete program, leave it untouched rather than creating globals
        // with no initialization point.
        return glsl.to_string();
    };
    let Some(open_rel) = rewritten[main_pos..].find('{') else {
        return glsl.to_string();
    };
    let main_open = main_pos + open_rel;

    let mut declarations = String::new();
    let mut initializers = String::new();
    for &q in &written {
        declarations.push_str(&format!("\nfloat particle_local_q{q};"));
        initializers.push_str(&format!(
            "\n    particle_local_q{q} = {};",
            q_uniform_component(q)
        ));
    }

    let mut out = String::with_capacity(glsl.len() + declarations.len() + initializers.len());
    out.push_str(&glsl[..body_start]);
    out.push_str(&declarations);
    out.push_str(&rewritten[..=main_open]);
    out.push_str(&initializers);
    out.push_str(&rewritten[main_open + 1..]);
    out
}

fn q_register_number(token: &str) -> Option<u8> {
    let digits = token.strip_prefix('q')?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let value: u8 = digits.parse().ok()?;
    (1..=32).contains(&value).then_some(value)
}

fn q_uniform_component(q: u8) -> String {
    let zero_based = q - 1;
    let group = char::from(b'a' + zero_based / 4);
    let component = ["x", "y", "z", "w"][(zero_based % 4) as usize];
    format!("_q{group}.{component}")
}

fn written_q_registers(src: &str) -> Vec<u8> {
    let bytes = src.as_bytes();
    let mut written = [false; 33];
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'/') {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            continue;
        }
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            i = skip_string_literal(bytes, i);
            continue;
        }
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let Some(q) = q_register_number(&src[start..i]) else {
                continue;
            };
            let mut after = i;
            while after < bytes.len() && bytes[after].is_ascii_whitespace() {
                after += 1;
            }
            let suffix = bytes.get(after..after.saturating_add(2));
            let suffix_write = matches!(
                suffix,
                Some(b"++")
                    | Some(b"--")
                    | Some(b"+=")
                    | Some(b"-=")
                    | Some(b"*=")
                    | Some(b"/=")
                    | Some(b"%=")
            ) || (bytes.get(after) == Some(&b'=')
                && bytes.get(after + 1) != Some(&b'='));
            let before = src[..start].trim_end().as_bytes();
            let prefix_write = before.ends_with(b"++") || before.ends_with(b"--");
            if suffix_write || prefix_write {
                written[q as usize] = true;
            }
            continue;
        }
        let ch = src[i..].chars().next().expect("i is in bounds");
        i += ch.len_utf8();
    }
    (1..=32).filter(|&q| written[q]).map(|q| q as u8).collect()
}

fn rewrite_q_registers(src: &str, written: &[u8]) -> String {
    let mut selected = [false; 33];
    for &q in written {
        selected[q as usize] = true;
    }
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'/') {
            let start = i;
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            out.push_str(&src[start..i]);
            continue;
        }
        if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let start = i;
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            out.push_str(&src[start..i]);
            continue;
        }
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let end = skip_string_literal(bytes, i);
            out.push_str(&src[i..end]);
            i = end;
            continue;
        }
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let token = &src[start..i];
            if let Some(q) = q_register_number(token).filter(|q| selected[*q as usize]) {
                out.push_str(&format!("particle_local_q{q}"));
            } else {
                out.push_str(token);
            }
            continue;
        }
        let ch = src[i..].chars().next().expect("i is in bounds");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn drop_duplicate_bare_decls_same_scope(src: &str) -> String {
    use std::collections::HashSet;

    let mut out = String::with_capacity(src.len());
    // Numeric brace depth is not a scope identity: two sibling helper functions
    // both have locals at depth 1. Use a real scope stack so a declaration in one
    // function can never suppress a same-named local in the next.
    let mut scopes: Vec<HashSet<String>> = vec![HashSet::new()];
    for line in src.lines() {
        let trimmed = line.trim();
        let mut replacement: Option<String> = None;
        let duplicate = parse_glsl_decl_line(trimmed).is_some_and(|(ty, names, bare)| {
            let scope = scopes.last_mut().expect("root scope is retained");
            let already_seen = names.iter().all(|name| scope.contains(name));
            if already_seen && bare {
                return true;
            }
            if already_seen && names.len() == 1 {
                // HLSL accepts a same-scope redeclaration with an initializer. GLSL
                // does not, so retain the side effect and lower it to assignment.
                let indent = &line[..line.len() - line.trim_start().len()];
                if let Some(rest) = trimmed.strip_prefix(&ty) {
                    replacement = Some(format!("{indent}{}", rest.trim_start()));
                    return false;
                }
            }
            for name in names {
                scope.insert(name);
            }
            false
        });
        if !duplicate {
            out.push_str(replacement.as_deref().unwrap_or(line));
            out.push('\n');
        }
        let brace_code = line.split_once("//").map_or(line, |(code, _)| code);
        for byte in brace_code.bytes() {
            match byte {
                b'{' => scopes.push(HashSet::new()),
                b'}' if scopes.len() > 1 => {
                    scopes.pop();
                }
                _ => {}
            }
        }
    }
    out
}

fn parse_glsl_decl_line(line: &str) -> Option<(String, Vec<String>, bool)> {
    let body = line.strip_suffix(';')?.trim();
    if body.starts_with("layout") || body.starts_with("return") {
        return None;
    }
    let mut parts = body.splitn(2, char::is_whitespace);
    let ty = parts.next()?.trim();
    keyword_gty(ty)?;
    let rest = parts.next()?.trim();
    if rest.is_empty() {
        return None;
    }
    let name_end = rest
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(rest.len());
    if rest[name_end..].trim_start().starts_with('(') {
        return None; // function prototype/definition, not a local declaration
    }
    let bare = !rest.contains('=');
    let names_part = rest.split('=').next().unwrap_or(rest);
    let mut names = Vec::new();
    for item in names_part.split(',') {
        let name: String = item
            .trim()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            names.push(name);
        }
    }
    if names.is_empty() {
        None
    } else {
        Some((ty.to_string(), names, bare))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "milk-native-converter")]
    #[test]
    fn converter_cache_hit_returns_stored_output_and_miss_on_different_source() {
        let mut cache = ConverterOutputCache::new(CONVERTER_OUTPUT_CACHE_BUDGET_BYTES);
        let key_a = ConverterInput::Plain {
            body: "ret = tex2D(sampler_main, uv).xyz;".to_string(),
        };
        let digest_a = converter_input_digest(&key_a);
        assert!(cache.get(digest_a, &key_a).is_none(), "cold lookup misses");
        cache.insert(digest_a, key_a.clone(), "GLSL_A".to_string());
        // A hit returns the stored output verbatim (byte-identical to a fresh run).
        assert_eq!(cache.get(digest_a, &key_a).as_deref(), Some("GLSL_A"));

        // A different .milk source hashes to a different key → miss → re-convert.
        let key_b = ConverterInput::Plain {
            body: "ret = tex2D(sampler_main, uv).zyx;".to_string(),
        };
        let digest_b = converter_input_digest(&key_b);
        assert!(cache.get(digest_b, &key_b).is_none());
    }

    #[cfg(feature = "milk-native-converter")]
    #[test]
    fn converter_cache_distinguishes_plain_and_ex_entry_points() {
        let mut cache = ConverterOutputCache::new(CONVERTER_OUTPUT_CACHE_BUDGET_BYTES);
        let body = "ret = tex2D(sampler_main, uv).xyz;".to_string();
        let plain = ConverterInput::Plain { body: body.clone() };
        let ex = ConverterInput::Ex {
            file_globals: String::new(),
            body,
            optimize: true,
        };
        let plain_digest = converter_input_digest(&plain);
        let ex_digest = converter_input_digest(&ex);
        cache.insert(plain_digest, plain.clone(), "PLAIN".to_string());
        cache.insert(ex_digest, ex.clone(), "EX".to_string());
        // Same body, different entry point → distinct cached outputs, never confused.
        assert_eq!(cache.get(plain_digest, &plain).as_deref(), Some("PLAIN"));
        assert_eq!(cache.get(ex_digest, &ex).as_deref(), Some("EX"));
    }

    #[cfg(feature = "milk-native-converter")]
    #[test]
    fn converter_cache_evicts_lru_to_stay_within_byte_budget() {
        // Budget holds ~2 of the 100-byte payloads; a third insert evicts the LRU.
        let mut cache = ConverterOutputCache::new(250);
        let payload = "x".repeat(100);
        let mk = |i: usize| ConverterInput::Plain {
            body: format!("body-{i}"),
        };
        for i in 0..3 {
            let k = mk(i);
            cache.insert(converter_input_digest(&k), k, payload.clone());
        }
        assert!(
            cache.total_bytes <= 250,
            "retained bytes stay within budget"
        );
        // Entry 0 (least-recently-used) was evicted; 1 and 2 remain.
        let k0 = mk(0);
        assert!(cache.get(converter_input_digest(&k0), &k0).is_none());
        let k2 = mk(2);
        assert!(cache.get(converter_input_digest(&k2), &k2).is_some());
    }

    #[test]
    fn mutable_q_writes_use_private_fragment_slots() {
        let io = "layout(location = 0) out vec4 fragColor;";
        let glsl = format!(
            "{}\n\
             float authored_q_update() {{ q25 += 1.0; return q25; }}\n\
             void main() {{\n\
                 // q1 = 9.0; comments do not request a mutable slot\n\
                 q25 = authored_q_update();\n\
                 fragColor = vec4(q25);\n\
             }}\n",
            milk_fs_preamble(io)
        );

        let fixed = fix_glsl_vector_types(&glsl);
        let body = fixed
            .split_once("// __MILK_BODY__")
            .expect("generated shader has body marker")
            .1;
        assert!(body.contains("float particle_local_q25;"), "{fixed}");
        assert!(body.contains("particle_local_q25 = _qg.x;"), "{fixed}");
        assert_eq!(count_word(body, "q25"), 0, "{fixed}");
        assert!(!body.contains("particle_local_q1"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("mutable q shader did not compile: {error}\n{fixed}"));
    }

    #[test]
    fn narrow_lum_overloads_preserve_scalar_expression_width() {
        let hlsl = r#"shader_body {
            float scalar = 0.25;
            float2 pair = float2(0.5, 0.25);
            ret.x = lerp(ret.x, lum(scalar), 0.2);
            uv += float2(lum(pair));
            ret += GetMain(uv);
        }"#;

        let glsl = hlsl_milk_warp_body_to_naga(hlsl);
        assert!(glsl.contains("float lum(float v) { return v; }"), "{glsl}");
        assert!(
            glsl.contains("float lum(vec2 v) { return dot(v, vec2(0.32, 0.49)); }"),
            "{glsl}"
        );
        let mut types = TypeTable::new();
        types.insert("scalar".to_string(), GTy::F);
        types.insert("pair".to_string(), GTy::V(2));
        types.insert("color".to_string(), GTy::V(3));
        assert_eq!(infer_ty("lum(scalar)", &types), GTy::F);
        assert_eq!(infer_ty("lum(pair)", &types), GTy::F);
        assert_eq!(infer_ty("lum(color)", &types), GTy::V(3));
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("narrow lum shader did not compile: {error}\n{glsl}"));
    }

    #[test]
    fn scalar_splats_are_repaired_inside_void_main_scope() {
        // The preamble has several helper parameters named `v` at different
        // widths, so the global type table deliberately marks `v` Unknown. The
        // generated void main() scope must override that with this local vec2.
        let hlsl = r#"shader_body {
            float2 v = 0.01;
            v = 0.02;
            ret = GetMain(uv + v * 0.0);
        }"#;

        let glsl = hlsl_milk_warp_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(fixed.contains("vec2 v = vec2(0.01);"), "{fixed}");
        assert!(fixed.contains("v = vec2(0.02);"), "{fixed}");
        crate::renderer::compile_glsl(&glsl).unwrap_or_else(|error| {
            panic!("scalar-splat shader did not compile: {error}\n{fixed}")
        });
    }

    #[test]
    fn scalarizes_vector_terms_in_float_return() {
        let mut table: TypeTable = HashMap::new();
        seed_known_types(&mut table);
        table.insert("tmp".to_string(), GTy::F);
        let src = "float GetDist(vec2 uvi) {vec2 tmp; tmp = vec2(0.1, 0.2);\n  return 1-(tmp + 1.0/255*(tmp)+ ds*.7);}\n";

        let fixed = fix_return_width_mismatches(src, &table);

        assert!(fixed.contains("return 1-((tmp).x"), "{fixed}");
        assert!(fixed.contains("1.0/255*((tmp)).x"), "{fixed}");
    }

    #[test]
    fn scalarizes_vector_terms_in_float_return_full_pipeline() {
        let glsl = "// __MILK_BODY__\nfloat GetDist(vec2 uvi) {vec2 tmp; tmp = vec2(0.1, 0.2);\n  return 1-(tmp + 1.0/255*(tmp)+ ds*.7);}\n";

        let fixed = fix_glsl_vector_types(glsl);

        assert!(fixed.contains("return 1-((tmp).x"), "{fixed}");
        assert!(fixed.contains("1.0/255*((tmp)).x"), "{fixed}");
    }

    #[test]
    fn scalarizes_real_getdist_shape() {
        let glsl = "// __MILK_BODY__\nvec2 fstep2(vec2 xy) {\n  return 1.0/res*round(res*xy);\n}\nfloat GetDist(vec2 uvi) {vec2 tmp; tmp = texture(sampler2D(sampler_pc_main, sampler_pc_main_samp),(uvi).xy).gb; \n  return 1-(tmp + 1.0/255*(tmp)+ ds*.7);}\nvec2 PutDist(float x) {float fg, fb; fg = modf((1-x)*255.0,fb);\n  return(vec2(fg,fb/255.0));}\nfloat MinDist(vec2 uvi) {\n   float tmp; vec4 nb;\n   tmp = GetDist(uvi);\n   return tmp;\n}\n";

        let fixed = fix_glsl_vector_types(glsl);

        assert!(fixed.contains("return 1-((tmp).x"), "{fixed}");
        assert!(fixed.contains("1.0/255*((tmp)).x"), "{fixed}");
    }

    #[test]
    fn scalar_swizzle_repair_leaves_function_call_prefix_intact() {
        let glsl = "// __MILK_BODY__\nvec3 noise = texture(sampler2D(sampler_noise_lq, sampler_noise_lq_samp), uv).rgb;\n";

        let fixed = fix_glsl_vector_types(glsl);

        assert!(
            fixed.contains("texture(sampler2D(sampler_noise_lq, sampler_noise_lq_samp), uv).rgb"),
            "{fixed}"
        );
        assert!(!fixed.contains("texturevec"), "{fixed}");
    }

    #[test]
    fn hlsl_converter_strips_inline_helper_comments() {
        let hlsl = "float MinDistB(float2 uvi) {float tmp; float4 nb; //##nicht ideal\n  tmp = GetDist(uvi);\n  return tmp;}\n";

        let converted = hlsl_to_glsl_body_ex(hlsl, false);

        assert!(!converted.contains("//##"), "{converted}");
        assert!(converted.contains("tmp = GetDist(uvi);"), "{converted}");
        assert!(converted.contains("return tmp;"), "{converted}");
    }

    #[test]
    fn truncates_vector_rhs_for_swizzled_scalar_lvalue() {
        let mut table: TypeTable = HashMap::new();
        seed_known_types(&mut table);
        table.insert("dz".to_string(), GTy::V(2));
        let src = "void main() {\n  dz.x = lum(GetPixel(uv-hor)) - lum(GetPixel(uv+hor));\n}\n";

        let fixed = fix_assignment_width_mismatches(src, &table);

        assert!(
            fixed.contains("dz.x = (lum(GetPixel(uv-hor)) - lum(GetPixel(uv+hor))).x;"),
            "{fixed}"
        );
    }

    #[test]
    fn overwide_swizzle_relop_is_left_unknown_not_panicking() {
        let mut table: TypeTable = HashMap::new();
        seed_known_types(&mut table);
        table.insert("v".to_string(), GTy::V(4));
        let src = "void main() {\n  float m = v.xyzwx < 0.5;\n}\n";

        assert!(matches!(infer_ty("v.xyzwx", &table), GTy::Unknown));
        assert_eq!(fix_vector_relops(src, &table), src);
    }

    #[test]
    fn extracts_multiline_helper_with_inline_comment() {
        let before = "float GetDistB(float2 uvi)  {return GetDist(uvi); } // 1-GetBlur1(uvi).b;}\nfloat MinDistB (float2 uvi) {float tmp; float4 nb; //##nicht ideal\n  tmp = GetDist(uvi);\n  tmp = min(tmp,GetDistB2(uvi)*.7) ;\n  return tmp;}\nfloat after = 1;\n";

        let (funcs, rest) = extract_function_defs(before);

        assert!(
            funcs.contains("tmp = GetDist(uvi);"),
            "funcs={funcs}\nrest={rest}"
        );
        assert!(funcs.contains("return tmp;}"), "funcs={funcs}\nrest={rest}");
        assert!(
            rest.contains("float after = 1;"),
            "funcs={funcs}\nrest={rest}"
        );
    }

    #[test]
    fn corpus_failure_multiline_dead_decl_and_reserved_sample_compile() {
        let hlsl = r#"shader_body {
            float4 dead_noise = float4(uv, 0, 1)
                + float4(0, 0, 0, 0);
            float3 sample = tex2D(sampler_main, uv).xyz;
            ret = sample * sample * sample;
        }"#;

        let glsl = hlsl_milk_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(!fixed.contains("dead_noise"), "{fixed}");
        assert!(fixed.contains("particle_sample"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("reserved/dead-decl shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_matrix_pow_and_log10_compile() {
        let hlsl = r#"shader_body {
            float2x2 rot = { q10, q11, -q11, q10 };
            float lum = (lum(ret)).x;
            ret = pow(lum, 0.4 + 1.6 * rand_preset.xyz);
            ret.xy = mul(rot, ret.xy);
            ret += log10(abs(ret) + 1.0);
        }"#;

        let glsl = hlsl_milk_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(fixed.contains("mat2 rot = mat2("), "{fixed}");
        assert!(fixed.contains("pow(vec3(lum)"), "{fixed}");
        assert!(fixed.contains("log10("), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("matrix/pow/log10 shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_comma_sequence_repairs_each_assignment() {
        let hlsl = r#"shader_body {
            ret = tex2D(sampler_main, uv).z, ret -= roam_sin.wzy * roam_cos.zxy;
            ret *= 0.5;
        }"#;

        let glsl = hlsl_milk_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(
            fixed.contains("ret = vec3(texture(sampler2D(sampler_main"),
            "{fixed}"
        );
        assert!(fixed.contains(", ret -= roam_sin.wzy"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("comma-sequence shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_helper_alias_enables_scalar_lum_repair() {
        let hlsl = r#"shader_body {
            #define MyGet GetPixel
            float4 lums = 0;
            lums.x = lum(MyGet(uv + texsize.zw));
            ret = lums.xxx;
        }"#;

        let glsl = hlsl_milk_warp_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(!fixed.contains("#define MyGet"), "{fixed}");
        assert!(fixed.contains("lum(GetPixel("), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("helper-alias shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_redeclarations_are_scope_aware() {
        let hlsl = r#"
            float helper_a(float2 domain) { float2 c = domain; return c.x; }
            float helper_b(float2 domain) { float2 c = domain; return c.y; }
            shader_body {
                float2 d, uv1;
                float3 dx, dy;
                float2 d = texsize.zw;
                float3 dx = float3(d, 0);
                float3 dy = float3(d.yx, 0);
                ret = helper_a(uv) + helper_b(uv) + dx + dy;
            }
        "#;

        let glsl = hlsl_milk_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert_eq!(
            count_word(&fixed, "c"),
            4,
            "both helper locals survive: {fixed}"
        );
        assert!(!fixed.contains("vec2 d = texsize.zw"), "{fixed}");
        assert!(!fixed.contains("vec3 dx = vec3"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("redeclaration shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_numeric_flags_compile_in_control_and_arithmetic() {
        let hlsl = r#"shader_body {
            int exist_count = 0;
            exist_count += ((float3(rand_preset.xyz >= 0.7))).x;
            int first = ((rand_preset.z > 0.5)).x;
            float mask = 0;
            if (!first) { mask = 1; }
            ret = mask * (!first) + saturate(!mask) + exist_count;
        }"#;

        let glsl = hlsl_milk_warp_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(fixed.contains("exist_count += int("), "{fixed}");
        assert!(fixed.contains("int first = int("), "{fixed}");
        assert!(fixed.contains("float(float(first) == 0.0)"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("numeric-flag shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_bool_compound_and_logical_coercions_compile() {
        let hlsl = r#"shader_body {
            float noise = tex2D(sampler_noise_lq, uv).x;
            float mask1 = noise;
            float gmask = 0;
            noise *= (noise >= 0.9);
            gmask = gmask || mask1;
            ret = noise + gmask;
        }"#;

        let glsl = hlsl_milk_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(fixed.contains("noise *=float("), "{fixed}");
        assert!(fixed.contains("(mask1) != 0.0"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("bool-coercion shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_sampling_helpers_accept_legacy_coordinate_widths() {
        let hlsl = r#"shader_body {
            float scalar = 0.5;
            float3 wide = tex2D(sampler_main, uv).xyz;
            ret = GetPixel(scalar) + GetBlur1(0) + GetBlur3(wide);
        }"#;

        let glsl = hlsl_milk_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(fixed.contains("GetPixel(vec2(scalar))"), "{fixed}");
        assert!(fixed.contains("GetBlur1(vec2(0))"), "{fixed}");
        assert!(fixed.contains("GetBlur3((wide).xy)"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("sampling-coordinate shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_fallthrough_helper_gets_deterministic_return() {
        let hlsl = r#"
            float shadow(float2 uvi) {
                int n; float dark;
                dark = 0; n = 0;
                while (!dark && (n < 4)) { n += 1; }
            }
            shader_body { ret = shadow(uv); }
        "#;

        let glsl = hlsl_milk_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(fixed.contains("return 0.0;"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("fallthrough-helper shader failed: {error}\n{fixed}"));
    }

    #[test]
    fn corpus_failure_multi_declarations_and_shadowed_lum_compile() {
        let hlsl = r#"shader_body {
            float4 lum1 = 0, lum2 = 0;
            float lum = dot(ret, float3(0.3, 0.5, 0.2));
            ret = lerp(lum, ret, 1.7) + lum1.xyz + lum2.xyz;
        }"#;

        let split = split_hlsl_multi_declarators(hlsl);
        assert!(split.contains("float4 lum1 = 0;"), "{split}");
        assert!(split.contains("float4 lum2 = 0"), "{split}");
        let glsl = hlsl_milk_body_to_naga(hlsl);
        let fixed = fix_glsl_vector_types(&glsl);
        assert!(fixed.contains("mix(vec3(lum), ret"), "{fixed}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|error| panic!("multi-decl/lum shader failed: {error}\n{fixed}"));
    }

    // ── P2-VIS-014: shader source/work budgets + one-pass preprocessing ───────

    #[test]
    fn over_budget_shader_source_is_rejected() {
        // A source larger than MAX_SHADER_BYTES is rejected by an O(1) length check
        // before any pass allocates or scans — bounded work regardless of size.
        let huge = "a = b;\n".repeat(MAX_SHADER_BYTES / 6 + 10);
        assert!(huge.len() > MAX_SHADER_BYTES);
        assert!(matches!(
            try_butterchurn_to_naga(&huge),
            Err(PreprocessError::SourceTooLarge { .. })
        ));
        // The infallible entry point degrades to an inert empty program.
        assert!(butterchurn_to_naga(&huge).is_empty());
    }

    #[test]
    fn too_many_lines_shader_source_is_rejected() {
        // Many short lines stay under the byte budget but blow the line budget; the
        // scan bails during classification (bounded work), before emit/join.
        let many_lines = "\n".repeat(MAX_SHADER_LINES + 5_000);
        assert!(
            many_lines.len() < MAX_SHADER_BYTES,
            "stays under byte budget"
        );
        assert!(matches!(
            try_butterchurn_to_naga(&many_lines),
            Err(PreprocessError::TooManyLines { .. })
        ));
    }

    #[test]
    fn normal_shader_still_preprocesses() {
        // A normal butterchurn shader converts correctly: version bump, precision
        // strip, sampler split, in/out layout qualifiers, scalar → UBO, and the
        // texture() sampler rewrite.
        let src = "#version 300 es\n\
precision highp float;\n\
uniform sampler2D sampler_main;\n\
uniform float time;\n\
in vec2 uv;\n\
out vec4 frag;\n\
void main() {\n\
    frag = texture(sampler_main, uv) * time;\n\
}\n";
        let out = butterchurn_to_naga(src);
        assert!(out.contains("#version 450"), "{out}");
        assert!(!out.contains("#version 300"), "{out}");
        assert!(!out.contains("precision highp"), "{out}");
        assert!(
            out.contains("layout(set = 0, binding = 0) uniform texture2D sampler_main;"),
            "{out}"
        );
        assert!(
            out.contains("layout(set = 0, binding = 1) uniform sampler sampler_main_samp;"),
            "{out}"
        );
        assert!(out.contains("layout(location = 0) in vec2 uv;"), "{out}");
        assert!(out.contains("layout(location = 0) out vec4 frag;"), "{out}");
        // scalar folded into the UBO block.
        assert!(out.contains("uniform PerFrame {"), "{out}");
        assert!(out.contains("float time;"), "{out}");
        // texture() rewritten to the separated sampler2D form.
        assert!(
            out.contains("texture(sampler2D(sampler_main, sampler_main_samp), uv)"),
            "{out}"
        );
    }

    #[test]
    fn rewrite_texture_calls_is_linear_and_output_identical() {
        let mut map: HashMap<String, u32> = HashMap::new();
        map.insert("sampler_main".to_string(), 0);
        map.insert("sampler_blur1".to_string(), 2);

        // `texture(name,` variant.
        assert_eq!(
            rewrite_texture_calls("x = texture(sampler_main, uv);", &map),
            "x = texture(sampler2D(sampler_main, sampler_main_samp), uv);"
        );
        // `texture(name ,` (one space before comma) variant — space preserved.
        assert_eq!(
            rewrite_texture_calls("texture(sampler_blur1 , uv)", &map),
            "texture(sampler2D(sampler_blur1, sampler_blur1_samp) , uv)"
        );
        // A longer identifier that merely starts with a known sampler name is NOT a
        // partial match (the whole token is compared), so it is left untouched.
        assert_eq!(
            rewrite_texture_calls("texture(sampler_main_extra, uv)", &map),
            "texture(sampler_main_extra, uv)"
        );
        // Unknown sampler names pass through unchanged.
        assert_eq!(
            rewrite_texture_calls("texture(unknown, uv)", &map),
            "texture(unknown, uv)"
        );
    }

    #[test]
    fn custom_sampler_metadata_preserves_names_and_ignores_builtins() {
        let src = r#"
sampler sampler_fw_worms;
uniform sampler2D sampler_rose;
ret = tex2D(sampler_rand00, uv).rgb;
ret += tex2D(sampler_noise_lq, uv).rgb;
"#;
        assert_eq!(
            custom_sampler_names(src),
            ["sampler_fw_worms", "sampler_rose", "sampler_rand00"]
        );
        assert!(
            custom_sampler_names("texture(sampler2D(sampler_main, sampler_main_samp), uv);")
                .is_empty()
        );
    }

    #[test]
    fn fixed_sampler_table_reserves_named_atlas_slots_without_growing() {
        assert_eq!(MILKDROP_SAMPLERS.len(), 16);
        assert_eq!(MILKDROP_SAMPLERS[12], "sampler_named_linear");
        assert_eq!(MILKDROP_SAMPLERS[13], "sampler_named_point");
        assert!(!MILKDROP_SAMPLERS.contains(&"sampler_noise_hq_lite"));
        assert!(!MILKDROP_SAMPLERS.contains(&"sampler_pw_noise_lq"));
        assert_eq!(
            normalize_milkdrop_sampler_variants("sampler_noise_hq_lite sampler_pw_noise_lq"),
            "sampler_noise_lq_lite sampler_noise_lq"
        );
    }

    #[test]
    fn explicit_custom_sampler_rewrite_is_simultaneous_and_updates_texsize() {
        let src = "ret = tex2D(sampler_rose, uv); float2 s = texsize_rose.xy; sampler_rosebud;";
        let replacements = HashMap::from([
            ("sampler_rose".to_string(), "sampler_named0".to_string()),
            (
                "sampler_named0".to_string(),
                "sampler_should_not_cascade".to_string(),
            ),
        ]);
        let out = rewrite_custom_sampler_identifiers(src, &replacements);
        assert_eq!(
            out,
            "ret = tex2D(sampler_named0, uv); float2 s = texsize_named0.xy; sampler_rosebud;"
        );
    }

    #[test]
    fn rewrites_named_texture_calls_to_fixed_guttered_atlas_slots() {
        let bindings = [
            NamedTextureRewriteBinding {
                sampler_name: "sampler_fw_worms".to_string(),
                layer: 5,
                point_filter: false,
                clamp: false,
            },
            NamedTextureRewriteBinding {
                sampler_name: "sampler_pc_rose".to_string(),
                layer: 2,
                point_filter: true,
                clamp: true,
            },
        ];
        let hlsl = "ret = tex2D(sampler_fw_worms, uv).rgb + tex2D(sampler_pc_rose, uv*2).rgb; float2 s=texsize_fw_worms.xy;";
        let out = rewrite_custom_sampler_calls_for_atlas(hlsl, &bindings, 256);
        assert!(out.contains("tex2D(sampler_named_linear"), "{out}");
        assert!(out.contains("frac((uv).xy)"), "{out}");
        assert!(out.contains("tex2D(sampler_named_point"), "{out}");
        assert!(out.contains("saturate((uv*2).xy)"), "{out}");
        assert!(out.contains("texsize_noise_mq.xy"), "{out}");
    }

    #[test]
    fn rewrites_combined_glsl_sampler_calls_to_atlas() {
        let bindings = [NamedTextureRewriteBinding {
            sampler_name: "sampler_rose".to_string(),
            layer: 0,
            point_filter: false,
            clamp: true,
        }];
        let glsl = "ret = texture(sampler2D(sampler_rose, sampler_rose_samp), uv).rgb;";
        let out = rewrite_custom_sampler_calls_for_atlas(glsl, &bindings, 256);
        assert!(out.contains("texture(sampler_named_linear"), "{out}");
        assert!(
            out.contains("clamp((uv).xy, vec2(0.0), vec2(1.0))"),
            "{out}"
        );
    }

    #[test]
    fn named_texture_hlsl_wrapper_produces_compilable_glsl() {
        let bindings = [NamedTextureRewriteBinding {
            sampler_name: "sampler_fw_worms".to_string(),
            layer: 3,
            point_filter: false,
            clamp: false,
        }];
        let hlsl =
            "sampler sampler_fw_worms;\nshader_body { ret = tex2D(sampler_fw_worms, uv).rgb; }";
        let glsl = hlsl_milk_body_to_naga_with_named_textures(hlsl, &bindings, 256);
        assert!(!glsl.contains("sampler_fw_worms"), "{glsl}");
        assert!(glsl.contains("sampler_named_linear"), "{glsl}");
        crate::renderer::compile_glsl(&glsl)
            .unwrap_or_else(|err| panic!("named-texture shader did not compile: {err}\n{glsl}"));
    }

    // ── P2-VIS-033: tokenized (comment/string/brace-aware) shader_body extract ─

    #[test]
    fn shader_body_extraction_ignores_braces_in_block_comment() {
        // The body contains unbalanced `}` inside a block comment AND content after
        // the real closing brace. A naive brace-count (only // aware) closes the
        // wrapper at the first `}` inside the comment and truncates the body; the
        // comment-aware scanner captures the whole body and drops the trailing junk.
        let src = "shader_body {\n\
    /* closing braces } } } inside a block comment */\n\
    ret = vec3(1.0);\n\
}\n\
this_is_after_the_wrapper;";
        let (before, inner) = split_shader_body_wrapper(src);
        assert!(before.is_empty(), "before={before}");
        assert!(
            inner.contains("ret = vec3(1.0);"),
            "body truncated: inner={inner}"
        );
        assert!(
            !inner.contains("this_is_after_the_wrapper"),
            "trailing content leaked: inner={inner}"
        );
    }

    #[test]
    fn shader_body_extraction_skips_keyword_substring_in_identifier() {
        // A global whose NAME embeds `shader_body` is declared before the real
        // wrapper. A plain substring find splits at the identifier, discarding the
        // global; the token-aware scanner splits at the real `shader_body {`.
        let src = "float shader_body_gain = 2.0;\n\
shader_body {\n\
    ret = vec3(shader_body_gain);\n\
}";
        let (before, inner) = split_shader_body_wrapper(src);
        assert_eq!(before, "float shader_body_gain = 2.0;", "before={before}");
        assert!(
            inner.contains("ret = vec3(shader_body_gain);"),
            "inner={inner}"
        );
        assert!(!inner.contains("shader_body {"), "inner={inner}");
    }
}

/// Repair a `return <expr>;` whose expr differs from the enclosing function's declared
/// return type in ways HLSL accepts but GLSL/naga rejects: vector truncation,
/// scalar-vector broadcasts, and bool/float coercions (`bool f(){return x*y;}`,
/// `float f(){return x<y;}`).
/// A forward pass tracks the active function's declared return type and a merged table
/// (global ∪ params ∪ local decls) — the global table collapses colliding local names
/// (`tmp` is vec2 in one fn, float in another) to Unknown, so per-function locals are
/// needed for the inference to fire. Conservative: only acts when both sides are
/// confidently typed; Unknown skips.
fn fix_return_width_mismatches(body: &str, global: &TypeTable) -> String {
    let mut out = String::with_capacity(body.len() + 32);
    let mut cur_ret: Option<GTy> = None;
    let mut fn_level = 0i32;
    let mut depth = 0i32;
    let mut merged: TypeTable = global.clone();
    for line in body.lines() {
        let trimmed = line.trim();
        if cur_ret.is_none() {
            if let Some((ret, params)) = parse_fn_header(trimmed) {
                cur_ret = Some(ret);
                fn_level = depth;
                merged = global.clone();
                for (nm, g) in params {
                    merged.insert(nm, g);
                }
            }
        }
        if cur_ret.is_some() {
            record_local_decls(trimmed, &mut merged);
        }
        let fixed = match cur_ret {
            Some(ret) => fix_return_line(line, &merged, ret),
            _ => line.to_string(),
        };
        out.push_str(&fixed);
        out.push('\n');
        for c in line.bytes() {
            if c == b'{' {
                depth += 1;
            } else if c == b'}' {
                depth -= 1;
                if cur_ret.is_some() && depth <= fn_level {
                    cur_ret = None;
                }
            }
        }
    }
    out
}

/// Parse a function-definition header `<typekw> <ident>(<params>) {` → (return type,
/// [(param name, type)]). Requires a `{` after the `)` so calls/prototypes don't match.
fn parse_fn_header(s: &str) -> Option<(GTy, Vec<(String, GTy)>)> {
    let open = s.find('(')?;
    let head = s[..open].trim();
    let mut hp = head.split_whitespace();
    let kw = hp.next()?;
    let ident = hp.next()?;
    if hp.next().is_some() {
        return None; // more than `<type> <name>` before `(` → not a header
    }
    // `main` and many authored helpers return void. Track those functions too so
    // their local declarations override same-named preamble/helper parameters in
    // the per-function type table. Unknown is intentional here: a void function
    // has no return expression to coerce, while its locals still need repair.
    let ret = if kw == "void" {
        GTy::Unknown
    } else {
        keyword_gty(kw)?
    };
    if ident.is_empty() || !ident.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    let close = matching_close(s, open)?;
    if !s[close + 1..].trim_start().starts_with('{') {
        return None;
    }
    let mut params = Vec::new();
    let params_str = &s[open + 1..close];
    if !params_str.trim().is_empty() {
        for p in params_str.split(',') {
            let mut pi = p.split_whitespace();
            if let (Some(pkw), Some(pname)) = (pi.next(), pi.next()) {
                if let Some(g) = keyword_gty(pkw) {
                    let nm: String = pname
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !nm.is_empty() {
                        params.push((nm, g));
                    }
                }
            }
        }
    }
    Some((ret, params))
}

/// HLSL accepts a non-void helper that reaches the end of its body. The result is
/// undefined there, but the legacy converter emits that helper verbatim and naga
/// rejects the complete module even when the helper is only used behind a zero
/// multiplier. Give helpers with no authored return a deterministic zero value.
/// Functions containing any return are left untouched; this intentionally fixes
/// only the unambiguous corpus pattern instead of attempting control-flow analysis.
fn fix_missing_function_returns(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 32);
    let mut cursor = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if !(bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
            i += 1;
            continue;
        }
        let type_start = i;
        i += 1;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let return_type = &src[type_start..i];
        let Some(return_gty) = keyword_gty(return_type) else {
            continue;
        };
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let name_start = i;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        if i == name_start {
            continue;
        }
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if bytes.get(i) != Some(&b'(') {
            continue;
        }
        let Some(params_close) = matching_close(src, i) else {
            break;
        };
        i = params_close + 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if bytes.get(i) != Some(&b'{') {
            continue;
        }
        let body_open = i;
        let Some(body_close) = matching_brace(src, body_open) else {
            break;
        };
        let body = &src[body_open + 1..body_close];
        if !contains_word(body, "return") {
            let zero = match return_gty {
                GTy::F => "0.0".to_string(),
                GTy::V(width) => format!("vec{width}(0.0)"),
                GTy::B => "false".to_string(),
                GTy::BV(width) => format!("bvec{width}(false)"),
                GTy::Unknown => {
                    i = body_close + 1;
                    continue;
                }
            };
            out.push_str(&src[cursor..body_close]);
            out.push_str(&format!("\nreturn {zero};\n"));
            cursor = body_close;
        }
        i = body_close + 1;
    }
    out.push_str(&src[cursor..]);
    out
}

fn matching_brace(src: &str, open: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    for (index, byte) in bytes.iter().copied().enumerate().skip(open) {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn contains_word(src: &str, word: &str) -> bool {
    src.match_indices(word).any(|(index, _)| {
        let before = index.checked_sub(1).and_then(|i| src.as_bytes().get(i));
        let after = src.as_bytes().get(index + word.len());
        !before.is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            && !after.is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
    })
}

/// Record leading `<typekw> name[, name…]` declarations found in `line` into `table`,
/// splitting on top-level `;`/`{`/`}` (so same-line function bodies like
/// `vec2 f(float x) {float a, b; …}` register `a`/`b`). Best-effort; non-decls ignored.
fn record_local_decls(line: &str, table: &mut TypeTable) {
    let b = line.as_bytes();
    let mut depth = 0i32;
    let mut start = 0usize;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b';' | b'{' | b'}' if depth == 0 => {
                record_decl_seg(&line[start..i], table);
                start = i + 1;
            }
            _ => {}
        }
    }
    record_decl_seg(&line[start..], table);
}

fn record_decl_seg(seg: &str, table: &mut TypeTable) {
    let core = seg.trim();
    let core = core.rsplit('{').next().unwrap_or(core).trim();
    let mut it = core.splitn(2, char::is_whitespace);
    let (Some(kw), Some(rest)) = (it.next(), it.next()) else {
        return;
    };
    let Some(g) = keyword_gty(kw) else {
        return;
    };
    let names = rest.split('=').next().unwrap_or(rest);
    for nm in names.split(',') {
        let ident: String = nm
            .trim()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !ident.is_empty() {
            table.insert(ident, g);
        }
    }
}

/// Truncate one `return <expr>;` to `ret`'s width when the expr is strictly wider. The
/// line must START with `return` (after indent) but may carry a trailing `}` (the
/// function's closing brace, which the converter emits on the same line) after the `;`.
fn fix_return_line(line: &str, merged: &TypeTable, ret: GTy) -> String {
    let (code, comment) = match line.find("//") {
        Some(c) => (&line[..c], &line[c..]),
        None => (line, ""),
    };
    let mut out = String::with_capacity(line.len() + 16);
    let b = code.as_bytes();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut saw_semi = false;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b';' if depth == 0 => {
                out.push_str(&fix_return_segment(&code[start..i], merged, ret));
                out.push(';');
                start = i + 1;
                saw_semi = true;
            }
            _ => {}
        }
    }
    if !saw_semi {
        return line.to_string();
    }
    out.push_str(&code[start..]);
    out.push_str(comment);
    out
}

fn fix_return_segment(seg: &str, merged: &TypeTable, ret: GTy) -> String {
    let trimmed = seg.trim();
    if trimmed.is_empty() {
        return seg.to_string();
    }
    let lead = &seg[..seg.len() - seg.trim_start().len()];
    let trail = &seg[seg.trim_end().len()..];
    let body = trimmed;
    if let Some(prefix_len) = repairable_statement_prefix_len(body) {
        let prefix = &body[..prefix_len];
        let rest = &body[prefix_len..];
        let fixed = fix_return_segment(rest, merged, ret);
        if fixed != rest {
            return format!("{lead}{prefix}{fixed}{trail}");
        }
    }
    if !is_return_keyword(body) {
        return seg.to_string();
    }
    let expr = body["return".len()..].trim();
    if expr.is_empty() {
        return seg.to_string();
    }
    let scalarized = if ret == GTy::F {
        scalarize_vectors_for_float_expr(expr, merged)
    } else {
        None
    };
    let expr_for_infer = scalarized.as_deref().unwrap_or(expr);
    let rt = infer_ty(expr_for_infer, merged);
    let w = gty_width(ret);
    let m = gty_width(rt);
    let fixed_expr = match (ret, rt) {
        (GTy::B, GTy::F) => Some(format!("({expr_for_infer}) != 0.0")),
        (GTy::F, GTy::B) => Some(format!("float({expr_for_infer})")),
        (GTy::F, GTy::V(_)) | (GTy::F, GTy::BV(_)) => Some(format!("({expr_for_infer}).x")),
        (GTy::V(w), GTy::F) if w > 1 => Some(format!("vec{w}({expr_for_infer})")),
        (GTy::V(w), GTy::V(m)) if m > w && w > 0 => {
            let sw = &"xyzw"[..w as usize];
            Some(format!("({expr_for_infer}).{sw}"))
        }
        _ => {
            if w != 0 && m != 0 && m > w && matches!(rt, GTy::V(_)) {
                let sw = &"xyzw"[..w as usize];
                Some(format!("({expr_for_infer}).{sw}"))
            } else {
                scalarized
            }
        }
    };
    match fixed_expr {
        Some(expr) => format!("{lead}return {expr}{trail}"),
        None => seg.to_string(),
    }
}

/// HLSL permits a vector expression to flow into a scalar context and takes the first
/// component. GLSL/naga rejects mixed scalar/vector arithmetic even if the enclosing
/// function returns `float`. Repair only inside float return expressions by scalarizing
/// vector operands that participate in arithmetic, preserving normal vector returns.
fn scalarize_vectors_for_float_expr(e: &str, t: &TypeTable) -> Option<String> {
    if let Some((l, op, r)) = split_binop_last(e) {
        let lnew = scalarize_vectors_for_float_expr(l, t);
        let rnew = scalarize_vectors_for_float_expr(r, t);
        let mut ltext = lnew.clone().unwrap_or_else(|| l.to_string());
        let mut rtext = rnew.clone().unwrap_or_else(|| r.to_string());
        let mut acted = lnew.is_some() || rnew.is_some();
        if matches!(op, "+" | "-" | "*" | "/" | "%") {
            if matches!(infer_ty(&ltext, t), GTy::V(_) | GTy::BV(_)) {
                ltext = format!("({}).x", ltext.trim());
                acted = true;
            }
            if matches!(infer_ty(&rtext, t), GTy::V(_) | GTy::BV(_)) {
                rtext = format!("({}).x", rtext.trim());
                acted = true;
            }
        }
        if acted {
            if op == "%" {
                return Some(format!("mod({ltext}, {rtext})"));
            }
            return Some(format!("{ltext}{op}{rtext}"));
        }
        return None;
    }

    let trimmed = e.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lead = &e[..e.len() - e.trim_start().len()];
    let trail = &e[e.trim_end().len()..];
    if matches!(trimmed.as_bytes()[0], b'-' | b'+' | b'!') {
        let sign = &trimmed[..1];
        let rest = &trimmed[1..];
        if let Some(rn) = scalarize_vectors_for_float_expr(rest, t) {
            return Some(format!("{lead}{sign}{rn}{trail}"));
        }
        return None;
    }
    if trimmed.starts_with('(') {
        if let Some(close) = matching_close(trimmed, 0) {
            if close == trimmed.len() - 1 {
                let inner = &trimmed[1..close];
                if let Some(inr) = scalarize_vectors_for_float_expr(inner, t) {
                    return Some(format!("{lead}({inr}){trail}"));
                }
                return None;
            }
        }
    }
    if let Some(dot) = trailing_member_dot(trimmed) {
        let base = &trimmed[..dot];
        let member = &trimmed[dot..];
        if let Some(bn) = scalarize_vectors_for_float_expr(base, t) {
            return Some(format!("{lead}{bn}{member}{trail}"));
        }
        return None;
    }
    if let Some((open, close)) = whole_call_span(trimmed) {
        let name = &trimmed[..open];
        let args = &trimmed[open + 1..close];
        let parts = split_top_level_commas(args);
        let mut changed = false;
        let mut newparts: Vec<String> = Vec::with_capacity(parts.len());
        for p in &parts {
            match scalarize_vectors_for_float_expr(p, t) {
                Some(np) => {
                    changed = true;
                    newparts.push(np);
                }
                None => newparts.push(p.clone()),
            }
        }
        if changed {
            return Some(format!("{lead}{name}({}){trail}", newparts.join(",")));
        }
        return None;
    }
    if let Some(q) = find_top_level_char(trimmed, b'?') {
        if let Some(colon_rel) = find_top_level_char(&trimmed[q + 1..], b':') {
            let cond = &trimmed[..q];
            let a = &trimmed[q + 1..q + 1 + colon_rel];
            let b = &trimmed[q + 1 + colon_rel + 1..];
            let cn = scalarize_vectors_for_float_expr(cond, t);
            let an = scalarize_vectors_for_float_expr(a, t);
            let bn = scalarize_vectors_for_float_expr(b, t);
            if cn.is_some() || an.is_some() || bn.is_some() {
                let cs = cn.unwrap_or_else(|| cond.to_string());
                let as_ = an.unwrap_or_else(|| a.to_string());
                let bs = bn.unwrap_or_else(|| b.to_string());
                return Some(format!("{lead}{cs}?{as_}:{bs}{trail}"));
            }
        }
    }
    None
}

/// naga's GLSL frontend mis-parses a `for`-loop whose CONDITION begins with a scalar/
/// vector type-constructor call (`for (init; float(n) < its; step)`): it reads `float`
/// as the start of a declaration in the condition slot, hits `(`, and reports
/// `InvalidToken(LeftParen, [Identifier])`. glsl-optimizer emits exactly this when it
/// casts an int loop counter to compare against a float bound. Wrapping the condition in
/// parens (`(float(n) < its)`) is semantically identical and parses fine. Body-only +
/// conservative: only fires on a 3-clause `for (...)` header whose condition's first
/// token is a known type constructor immediately followed by `(`.
fn fix_for_constructor_cond(src: &str) -> String {
    const CTORS: &[&str] = &[
        "float", "int", "uint", "double", "bool", "vec2", "vec3", "vec4", "ivec2", "ivec3",
        "ivec4", "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4", "mat2", "mat3", "mat4",
    ];
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 16);
    let mut i = 0usize;
    while i < src.len() {
        let is_for = src[i..].starts_with("for")
            && (i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_'))
            && {
                let after = i + 3;
                after >= src.len()
                    || !(bytes[after].is_ascii_alphanumeric() || bytes[after] == b'_')
            };
        if !is_for {
            let ch = src[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        // find the '(' that opens the for-header (only whitespace between `for` and `(`)
        let mut p = i + 3;
        while p < src.len() && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        if p >= src.len() || bytes[p] != b'(' {
            out.push_str("for");
            i += 3;
            continue;
        }
        let open = p;
        let Some(close) = matching_close(src, open) else {
            out.push_str("for");
            i += 3;
            continue;
        };
        let header = &src[open + 1..close];
        // split into init;cond;step at the two TOP-LEVEL semicolons
        let mut semis = [0usize; 2];
        let mut nsemi = 0usize;
        let mut d = 0i32;
        for (k, &c) in header.as_bytes().iter().enumerate() {
            match c {
                b'(' | b'[' => d += 1,
                b')' | b']' => d -= 1,
                b';' if d == 0 => {
                    if nsemi < 2 {
                        semis[nsemi] = k;
                    }
                    nsemi += 1;
                }
                _ => {}
            }
        }
        if nsemi != 2 {
            out.push_str(&src[i..=close]);
            i = close + 1;
            continue;
        }
        let cond = &header[semis[0] + 1..semis[1]];
        let cond_trimmed = cond.trim_start();
        let starts_with_ctor = CTORS.iter().any(|c| {
            cond_trimmed.starts_with(c)
                && cond_trimmed[c.len()..].trim_start().starts_with('(')
                && !cond_trimmed
                    .as_bytes()
                    .get(c.len())
                    .is_some_and(|&b| b.is_ascii_alphanumeric() || b == b'_')
        });
        if !starts_with_ctor {
            out.push_str(&src[i..=close]);
            i = close + 1;
            continue;
        }
        let lead = &cond[..cond.len() - cond_trimmed.len()];
        let init = &header[..semis[0]];
        let step = &header[semis[1] + 1..];
        out.push_str("for (");
        out.push_str(init);
        out.push(';');
        out.push_str(lead);
        out.push('(');
        out.push_str(cond_trimmed);
        out.push(')');
        out.push(';');
        out.push_str(step);
        out.push(')');
        i = close + 1;
    }
    out
}

/// `pow(base, expo)` where base and expo are float-vectors of DIFFERENT width is
/// rejected by naga ("Unknown function 'pow'"). HLSL applies pow component-wise and
/// truncates the wider operand. Truncate the wider arg to the narrower width. naga
/// types an arithmetic base by the LEFTMOST vector operand width (NOT the max), so we
/// mirror that with `leftmost_vec_width`. HLSL also broadcasts a scalar when the other
/// argument is a vector; GLSL requires both arguments to have the same genType.
fn fix_pow_arg_width(src: &str, t: &TypeTable) -> String {
    let mut out = String::with_capacity(src.len() + 32);
    let mut in_fn = false;
    let mut fn_level = 0i32;
    let mut depth = 0i32;
    let mut merged = t.clone();
    for line in src.lines() {
        let trimmed = line.trim();
        if !in_fn {
            if let Some((_ret, params)) = parse_fn_header(trimmed) {
                in_fn = true;
                fn_level = depth;
                merged = t.clone();
                for (name, ty) in params {
                    merged.insert(name, ty);
                }
            }
        }
        if in_fn {
            record_local_decls(trimmed, &mut merged);
        }
        out.push_str(&fix_pow_arg_width_with_table(
            line,
            if in_fn { &merged } else { t },
        ));
        out.push('\n');
        for byte in line.bytes() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if in_fn && depth <= fn_level {
                        in_fn = false;
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn fix_pow_arg_width_with_table(src: &str, t: &TypeTable) -> String {
    rewrite_calls_named(src, "pow", |args| {
        if args.len() != 2 {
            return None;
        }
        let base = args[0].trim();
        let exponent = args[1].trim();
        let tb = infer_ty(base, t);
        let te = infer_ty(exponent, t);
        let wb = leftmost_vec_width(base, t);
        let we = gty_width(te);
        if tb == GTy::F && we >= 2 {
            return Some(vec![format!("vec{we}({base})"), args[1].clone()]);
        }
        if wb >= 2 && te == GTy::F {
            return Some(vec![args[0].clone(), format!("vec{wb}({exponent})")]);
        }
        if wb < 2 || we < 2 || wb == we {
            return None;
        }
        let w = wb.min(we) as usize;
        let sw = &"xyzw"[..w];
        let mut out = vec![args[0].clone(), args[1].clone()];
        if wb > we {
            out[0] = format!("({}).{sw}", args[0].trim());
        } else {
            out[1] = format!("({}).{sw}", args[1].trim());
        }
        Some(out)
    })
}

/// naga types an arithmetic expression's width by the FIRST (leftmost) vector operand,
/// not the max — `vec4 * vec3` is rejected, but `pow(vec4first*…, …)` reports the base
/// as vec4. Return that leftmost confident vector width (0 if none / Unknown).
fn leftmost_vec_width(expr: &str, t: &TypeTable) -> u8 {
    let e = strip_enclosing_parens(expr);
    if let Some((l, op, r)) = split_binop(e) {
        if matches!(op, "+" | "-" | "*" | "/" | "%") {
            let wl = leftmost_vec_width(l, t);
            if wl >= 2 {
                return wl;
            }
            return leftmost_vec_width(r, t);
        }
        return 0; // relational/logical: not arithmetic width
    }
    gty_width(infer_ty(e, t))
}

/// `mix(a, b, s)` with genType args a,b of differing known width, or a scalar-bool
/// selector, is rejected by naga. Normalize a,b to the common (narrower) width and
/// broadcast a scalar-bool selector via `float(s)`. Conservative: both genType widths
/// must be confidently known; Unknown leaves the call untouched.
fn fix_mix_calls(src: &str, t: &TypeTable) -> String {
    rewrite_calls_named(src, "mix", |args| {
        if args.len() != 3 {
            return None;
        }
        let a = args[0].trim();
        let b = args[1].trim();
        let s = args[2].trim();
        let ta = infer_ty(a, t);
        let tb = infer_ty(b, t);
        if ta == GTy::Unknown || tb == GTy::Unknown {
            return None;
        }
        let (wa, wb) = (gty_width(ta), gty_width(tb));
        let mut na = args[0].clone();
        let mut nb = args[1].clone();
        let mut ns = args[2].clone();
        let mut changed = false;
        // Step 1: normalize genType args to the narrower width when they differ.
        if wa >= 1 && wb >= 1 && wa != wb {
            let w = wa.min(wb);
            if w >= 2 {
                let sw = &"xyzw"[..w as usize];
                if wa > w {
                    na = if ta == GTy::F {
                        format!("vec{w}({a})")
                    } else {
                        format!("({a}).{sw}")
                    };
                } else if wb > w {
                    nb = if tb == GTy::F {
                        format!("vec{w}({b})")
                    } else {
                        format!("({b}).{sw}")
                    };
                }
                changed = true;
            } else if w == 1 {
                // one side scalar, the other a vector: broadcast the scalar up.
                let target = wa.max(wb);
                if wa == 1 && ta == GTy::F {
                    na = format!("vec{target}({a})");
                    changed = true;
                } else if wb == 1 && tb == GTy::F {
                    nb = format!("vec{target}({b})");
                    changed = true;
                }
            }
        }
        // Step 2: a scalar-bool selector must become float(s) (HLSL broadcasts it).
        if infer_ty(s, t) == GTy::B {
            ns = format!("float({s})");
            changed = true;
        }
        if changed {
            Some(vec![na, nb, ns])
        } else {
            None
        }
    })
}

/// HLSL accepts `dot(float, float)` as scalar multiplication and truncates mismatched
/// vector widths (`dot(float4, float3)`). GLSL/naga require a matching vector overload.
fn fix_dot_calls(src: &str, t: &TypeTable) -> String {
    rewrite_calls_named_expr(src, "dot", |args| {
        if args.len() != 2 {
            return None;
        }
        let a = args[0].trim();
        let b = args[1].trim();
        let ta = infer_ty(a, t);
        let tb = infer_ty(b, t);
        if ta == GTy::F && tb == GTy::F {
            return Some(format!("(({a}) * ({b}))"));
        }
        if let (GTy::V(wa), GTy::V(wb)) = (ta, tb) {
            if wa >= 2 && wb >= 2 && wa != wb {
                let w = wa.min(wb) as usize;
                let sw = &"xyzw"[..w];
                let na = if wa > wb {
                    format!("({a}).{sw}")
                } else {
                    a.to_string()
                };
                let nb = if wb > wa {
                    format!("({b}).{sw}")
                } else {
                    b.to_string()
                };
                return Some(format!("dot({na}, {nb})"));
            }
        }
        None
    })
}

/// HLSL allows scalar swizzles (`q1.x`, `s.xx`); GLSL/naga do not. Convert confident
/// scalar identifier swizzles to either the bare scalar or a broadcast constructor.
fn fix_typed_scalar_swizzles(src: &str, t: &TypeTable) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < src.len() {
        let c = b[i];
        if (c.is_ascii_alphabetic() || c == b'_')
            && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_'))
        {
            let start = i;
            i += 1;
            while i < src.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let ident = &src[start..i];
            let scalar_ty = t.get(ident).copied();
            if i < src.len() && b[i] == b'.' && matches!(scalar_ty, Some(GTy::F | GTy::B)) {
                let sw_start = i + 1;
                let mut sw_end = sw_start;
                while sw_end < src.len()
                    && matches!(
                        b[sw_end],
                        b'x' | b'y' | b'z' | b'w' | b'r' | b'g' | b'b' | b'a'
                    )
                {
                    sw_end += 1;
                }
                if sw_end > sw_start {
                    let width = sw_end - sw_start;
                    if width == 1 {
                        out.push_str(ident);
                    } else if scalar_ty == Some(GTy::B) {
                        out.push_str(&format!("bvec{width}({ident})"));
                    } else {
                        out.push_str(&format!("vec{width}({ident})"));
                    }
                    i = sw_end;
                    continue;
                }
            }
            out.push_str(ident);
            continue;
        }
        // A function call's `(` belongs to its callee. Treating it as a bare
        // parenthesized scalar expression would turn
        // `texture(...).rgb` into `texturevec3((...))`, which is invalid GLSL.
        // Only repair a parenthesized expression when it is not immediately
        // attached to an identifier (the function-call form).
        if c == b'(' && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_')) {
            if let Some(close) = matching_close(src, i) {
                let after = close + 1;
                let scalar_ty = infer_ty(&src[i..=close], t);
                if after < src.len() && b[after] == b'.' && matches!(scalar_ty, GTy::F | GTy::B) {
                    let sw_start = after + 1;
                    let mut sw_end = sw_start;
                    while sw_end < src.len()
                        && matches!(
                            b[sw_end],
                            b'x' | b'y' | b'z' | b'w' | b'r' | b'g' | b'b' | b'a'
                        )
                    {
                        sw_end += 1;
                    }
                    if sw_end > sw_start {
                        let width = sw_end - sw_start;
                        let expr = &src[i..=close];
                        if width == 1 {
                            out.push_str(expr);
                        } else if scalar_ty == GTy::B {
                            out.push_str(&format!("bvec{width}({expr})"));
                        } else {
                            out.push_str(&format!("vec{width}({expr})"));
                        }
                        i = sw_end;
                        continue;
                    }
                }
            }
        }
        let ch = src[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Variant of `rewrite_calls_named` for fixes that need to replace the whole call
/// expression rather than just its argument list.
fn rewrite_calls_named_expr(
    src: &str,
    name: &str,
    f: impl Fn(&[String]) -> Option<String> + Copy,
) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(n + 64);
    let mut i = 0;
    while i < n {
        if (b[i].is_ascii_alphabetic() || b[i] == '_')
            && (i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_'))
        {
            let mut j = i;
            while j < n && (b[j].is_ascii_alphanumeric() || b[j] == '_') {
                j += 1;
            }
            let ident: String = b[i..j].iter().collect();
            if j < n && b[j] == '(' {
                let mut depth = 0i32;
                let mut k = j;
                while k < n {
                    match b[k] {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                if k < n {
                    let inner: String = b[j + 1..k].iter().collect();
                    let inner_fixed = rewrite_calls_named_expr(&inner, name, f);
                    if ident == name {
                        let args = split_top_level_commas(&inner_fixed);
                        if let Some(replacement) = f(&args) {
                            out.push_str(&replacement);
                            i = k + 1;
                            continue;
                        }
                    }
                    out.push_str(&ident);
                    out.push('(');
                    out.push_str(&inner_fixed);
                    out.push(')');
                    i = k + 1;
                    continue;
                }
            }
            out.push_str(&ident);
            i = j;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Walk `src`, find every call to a function literally named `name` (identifier
/// boundary + immediately-following `(`), recurse into its argument list FIRST (so
/// nested calls of the same name are handled inner-first), then let `f` optionally
/// rewrite the (recursively-fixed) top-level argument list. `f` receives the args
/// split at top-level commas and returns the replacement args, or None to leave the
/// call verbatim. Body-only when fed the post-sentinel body.
fn rewrite_calls_named(
    src: &str,
    name: &str,
    f: impl Fn(&[String]) -> Option<Vec<String>> + Copy,
) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(n + 64);
    let mut i = 0;
    while i < n {
        if (b[i].is_ascii_alphabetic() || b[i] == '_')
            && (i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_'))
        {
            let mut j = i;
            while j < n && (b[j].is_ascii_alphanumeric() || b[j] == '_') {
                j += 1;
            }
            let ident: String = b[i..j].iter().collect();
            if j < n && b[j] == '(' {
                // find matching close paren
                let mut depth = 0i32;
                let mut k = j;
                while k < n {
                    match b[k] {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                if k < n {
                    let inner: String = b[j + 1..k].iter().collect();
                    // recurse into the argument text first (handles nested same-name calls)
                    let inner_fixed = rewrite_calls_named(&inner, name, f);
                    if ident == name {
                        let args = split_top_level_commas(&inner_fixed);
                        if let Some(new_args) = f(&args) {
                            out.push_str(&ident);
                            out.push('(');
                            out.push_str(&new_args.join(","));
                            out.push(')');
                            i = k + 1;
                            continue;
                        }
                    }
                    out.push_str(&ident);
                    out.push('(');
                    out.push_str(&inner_fixed);
                    out.push(')');
                    i = k + 1;
                    continue;
                }
            }
            out.push_str(&ident);
            i = j;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Truncate the wider operand of a `+ - * /` between two float-vectors of DIFFERENT
/// known width (vec3+vec4 etc.), which naga rejects ("Operation Add/Multiply can't work
/// with …"). HLSL implicitly truncates the wider operand to the narrower; we mirror that.
/// Body-only, statement-guarded (skip lines with an internal `;` or `//` or no trailing
/// `;`, like fix_assign_line, to avoid the historical ;-splice regression). Rewrites the
/// RHS of a plain `=` or the whole expression-statement via a recursive rewriter that
/// reuses split_binop_last (LEFT-associative) + infer_ty and only ever splices a balanced
/// `(operand).swz` — it never reconstructs unchanged text, so float-exponent literals
/// (`1.5e-5`) and spacing are preserved verbatim.
/// Paren+bracket nesting depth of `s` (braces are NOT counted — they delimit blocks, not
/// expressions, so a statement boundary `;` can sit at brace depth > 0).
fn paren_bracket_depth(s: &str) -> i32 {
    let mut d = 0i32;
    for c in s.bytes() {
        match c {
            b'(' | b'[' => d += 1,
            b')' | b']' => d -= 1,
            _ => {}
        }
    }
    d
}

/// Join physical lines that belong to one logical statement into a single line, so the
/// line-based width passes can repair statements the converter wrapped across lines — the
/// ORB family `a += …texture(…,(ret*0.1+\n vec2(…)).xy)*0.2;` and blends like
/// `hue = a\n + b;`. A statement accumulates until a `;` at paren/bracket depth 0; control
/// headers (`for(…)`/`if(…)`/`while(…)`/`else`) and block braces (`{`/`}`) flush on their
/// own so a loop body stays a separate statement. A line bearing a `//` comment forces a
/// flush (never join across a line comment — it would swallow the continuation). Body-only
/// (runs after the sentinel); GLSL is newline-insensitive so joined output is equivalent.
fn join_logical_statements(body: &str) -> String {
    let mut out = String::with_capacity(body.len() + 32);
    let mut buf = String::new();
    let flush = |out: &mut String, buf: &mut String| {
        if !buf.is_empty() {
            out.push_str(buf);
            out.push('\n');
            buf.clear();
        }
    };
    for line in body.lines() {
        // Preprocessor directives (`#define`, `#if`, …) are newline-terminated, not
        // `;`-terminated — NEVER join them with the next line (that would swallow a
        // following declaration into the macro body: `#define sat saturate float z;`).
        if line.trim_start().starts_with('#') {
            flush(&mut out, &mut buf);
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let has_comment = line.contains("//");
        if buf.is_empty() {
            buf.push_str(line);
        } else {
            buf.push(' ');
            buf.push_str(line.trim_start());
        }
        let d = paren_bracket_depth(&buf);
        let t = buf.trim_start();
        let last = buf.trim_end().bytes().last().unwrap_or(b' ');
        let is_ctrl = t.starts_with("for")
            || t.starts_with("if")
            || t.starts_with("while")
            || t.starts_with("else")
            || t.starts_with('}')
            || t.starts_with('{')
            || t.starts_with('#');
        if has_comment {
            flush(&mut out, &mut buf); // never join across a line comment
        } else if d <= 0 && matches!(last, b';' | b'{' | b'}') {
            flush(&mut out, &mut buf); // complete statement / block boundary
        } else if d <= 0 && is_ctrl && last == b')' {
            flush(&mut out, &mut buf); // control header (for/if/while …) — keep its body separate
        }
        // else: keep accumulating (open paren/bracket, or a depth-0 operator/comma continuation)
    }
    flush(&mut out, &mut buf);
    out
}

/// Group a C-style brace initializer of a vector array into GLSL array-constructor form:
/// `const vec4 s[5] = {20 bare scalars};` → `const vec4 s[5] = vec4[5](vec4(..), …);`.
/// naga rejects the brace form ("Composing expects N components but W*N were given"). The
/// statement spans physical lines, so this runs on the whole body (not line-by-line).
/// Conservative: fires ONLY when the brace holds exactly W*N bare scalar elements (no
/// nested constructor), otherwise the matched span is emitted verbatim.
fn fix_array_brace_init(body: &str) -> String {
    let b = body.as_bytes();
    let n = b.len();
    let mut out = String::with_capacity(n + 64);
    let mut i = 0;
    while i < n {
        if let Some((end, repl)) = try_matrix_brace_init(body, i) {
            out.push_str(&repl);
            i = end;
            continue;
        }
        if let Some((end, repl)) = try_array_brace_init(body, i) {
            out.push_str(&repl);
            i = end;
            continue;
        }
        let ch = body[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Translate an HLSL matrix brace initializer to a GLSL constructor:
/// `mat2 rot = {a, b, c, d}` -> `mat2 rot = mat2(a, b, c, d)`.
/// The element count must exactly match the square matrix dimensions.
fn try_matrix_brace_init(body: &str, i: usize) -> Option<(usize, String)> {
    let b = body.as_bytes();
    let n = b.len();
    if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') {
        return None;
    }
    if i + 4 > n || &b[i..i + 3] != b"mat" || !matches!(b[i + 3], b'2' | b'3' | b'4') {
        return None;
    }
    let width = (b[i + 3] - b'0') as usize;
    let mut p = i + 4;
    if p >= n || !b[p].is_ascii_whitespace() {
        return None;
    }
    while p < n && b[p].is_ascii_whitespace() {
        p += 1;
    }
    let id0 = p;
    while p < n && (b[p].is_ascii_alphanumeric() || b[p] == b'_') {
        p += 1;
    }
    if p == id0 {
        return None;
    }
    let ident = &body[id0..p];
    while p < n && b[p].is_ascii_whitespace() {
        p += 1;
    }
    if p >= n || b[p] != b'=' || b.get(p + 1) == Some(&b'=') {
        return None;
    }
    p += 1;
    while p < n && b[p].is_ascii_whitespace() {
        p += 1;
    }
    if p >= n || b[p] != b'{' {
        return None;
    }
    let open = p;
    let mut depth = 0i32;
    let mut close = open;
    while close < n {
        match b[close] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        close += 1;
    }
    if close >= n {
        return None;
    }
    let inner = &body[open + 1..close];
    if inner.contains('{') {
        return None;
    }
    let mut pieces = split_top_level_commas(inner);
    if pieces.last().is_some_and(|piece| piece.trim().is_empty()) {
        pieces.pop();
    }
    if pieces.len() != width * width || pieces.iter().any(|piece| piece.trim().is_empty()) {
        return None;
    }
    Some((
        close + 1,
        format!(
            "mat{width} {ident} = mat{width}({})",
            pieces
                .iter()
                .map(|piece| piece.trim())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ))
}

/// Try to parse `vecW <ident>[N] = { … }` starting exactly at byte `i`. Returns
/// (index just past the closing `}`, replacement text) on a confident match, else None.
fn try_array_brace_init(body: &str, i: usize) -> Option<(usize, String)> {
    let b = body.as_bytes();
    let n = b.len();
    // word boundary before the type keyword
    if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') {
        return None;
    }
    // vec2 | vec3 | vec4
    if i + 4 > n || &b[i..i + 3] != b"vec" || !matches!(b[i + 3], b'2' | b'3' | b'4') {
        return None;
    }
    let w = (b[i + 3] - b'0') as usize;
    let mut p = i + 4;
    let ws = |p: &mut usize| {
        while *p < n && b[*p].is_ascii_whitespace() {
            *p += 1
        }
    };
    // require whitespace after the type (so `vec4 s` not `vec4(` / `vec4x`)
    if p >= n || !b[p].is_ascii_whitespace() {
        return None;
    }
    ws(&mut p);
    // identifier
    let id0 = p;
    while p < n && (b[p].is_ascii_alphanumeric() || b[p] == b'_') {
        p += 1;
    }
    if p == id0 {
        return None;
    }
    let ident = &body[id0..p];
    ws(&mut p);
    // [N]
    if p >= n || b[p] != b'[' {
        return None;
    }
    p += 1;
    ws(&mut p);
    let nd0 = p;
    while p < n && b[p].is_ascii_digit() {
        p += 1;
    }
    if p == nd0 {
        return None;
    }
    let count: usize = body[nd0..p].parse().ok()?;
    ws(&mut p);
    if p >= n || b[p] != b']' {
        return None;
    }
    p += 1;
    ws(&mut p);
    // =
    if p >= n || b[p] != b'=' || b.get(p + 1) == Some(&b'=') {
        return None;
    }
    p += 1;
    ws(&mut p);
    // {
    if p >= n || b[p] != b'{' {
        return None;
    }
    let brace_open = p;
    // matching } by brace depth
    let mut depth = 0i32;
    let mut q = brace_open;
    while q < n {
        match b[q] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        q += 1;
    }
    if q >= n {
        return None;
    }
    let inner = &body[brace_open + 1..q];
    // a nested `{` means an array-of-vectors / struct init — leave alone
    if inner.contains('{') {
        return None;
    }
    let mut pieces: Vec<String> = split_top_level_commas(inner)
        .into_iter()
        .map(|s| s.trim().to_string())
        .collect();
    if pieces.last().map_or(false, |s| s.is_empty()) {
        pieces.pop(); // tolerate a trailing comma
    }
    // exactly W*N bare scalars, none already a constructor
    if pieces.len() != w * count || pieces.iter().any(|s| s.is_empty() || s.contains('(')) {
        return None;
    }
    let ctors: Vec<String> = pieces
        .chunks(w)
        .map(|c| format!("vec{w}({})", c.join(", ")))
        .collect();
    let repl = format!(
        "vec{w} {ident}[{count}] = vec{w}[{count}]({})",
        ctors.join(", ")
    );
    Some((q + 1, repl))
}

fn fix_op_width_mismatches(src: &str, t: &TypeTable) -> String {
    let mut out = String::with_capacity(src.len() + 64);
    for line in src.lines() {
        out.push_str(&fix_op_width_line(line, t));
        out.push('\n');
    }
    out
}

fn fix_op_width_line(line: &str, t: &TypeTable) -> String {
    let (code, comment) = match line.find("//") {
        Some(c) => (&line[..c], &line[c..]),
        None => (line, ""),
    };
    let mut out = String::with_capacity(line.len() + 16);
    let b = code.as_bytes();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut saw_semi = false;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b';' if depth == 0 => {
                out.push_str(&fix_op_width_segment(&code[start..i], t));
                out.push(';');
                start = i + 1;
                saw_semi = true;
            }
            _ => {}
        }
    }
    if !saw_semi {
        return line.to_string();
    }
    out.push_str(&code[start..]);
    out.push_str(comment);
    out
}

fn fix_op_width_segment(seg: &str, t: &TypeTable) -> String {
    let trimmed = seg.trim();
    if trimmed.is_empty() {
        return seg.to_string();
    }
    let lead = &seg[..seg.len() - seg.trim_start().len()];
    let trail = &seg[seg.trim_end().len()..];
    let body = trimmed;
    if let Some(prefix_len) = repairable_statement_prefix_len(body) {
        let prefix = &body[..prefix_len];
        let rest = &body[prefix_len..];
        let fixed = fix_op_width_segment(rest, t);
        if fixed != rest {
            return format!("{lead}{prefix}{fixed}{trail}");
        }
    }
    let rebuilt = if let Some(eq) = find_plain_eq(body) {
        let lhs = &body[..=eq];
        let rhs = &body[eq + 1..];
        rewrite_expr_width(rhs, t).map(|new_rhs| format!("{lhs}{new_rhs}"))
    } else if let Some(op) = find_compound_assign(body) {
        // Compound assignment `lhs op= rhs` (op in + - * /): rewrite op-width ONLY inside
        // the RHS. This repairs `ret1 -= (roam_sin*roam_cos.wzy).xyz;` (vec4*vec3 inside),
        // AND — critically — keeps the bare-expression branch below from ever splitting at
        // the `-`/`*` of an `op=` token and truncation-wrapping the LHS into invalid GLSL
        // (`(a*=b).xyz` → naga `InvalidToken: Assign`). `lhs` keeps the whole `op=` token.
        let lhs = &body[..op + 2];
        let rhs = &body[op + 2..];
        let rewritten = rewrite_expr_width(rhs, t);
        let rhs_text = rewritten.as_deref().unwrap_or(rhs);
        if let Some(numeric) = bool_to_numeric_expr(rhs_text, infer_ty(rhs_text, t)) {
            Some(format!("{lhs}{numeric}"))
        } else {
            rewritten.map(|new_rhs| format!("{lhs}{new_rhs}"))
        }
    } else if is_return_keyword(body) {
        // `return <expr>;` — rewrite op-width mismatches inside the returned expression.
        // `expr` keeps its leading whitespace / `(`, which rewrite_expr_width preserves.
        let expr = &body["return".len()..];
        rewrite_expr_width(expr, t).map(|ne| format!("return{ne}"))
    } else {
        rewrite_expr_width(body, t)
    };
    match rebuilt {
        Some(stmt) => format!("{lead}{stmt}{trail}"),
        None => seg.to_string(),
    }
}

/// True if `body` is a `return` statement: starts with the keyword `return` followed by
/// whitespace or `(` (so `returned`/`return_x` identifiers don't match).
fn is_return_keyword(body: &str) -> bool {
    body.strip_prefix("return")
        .is_some_and(|rest| rest.starts_with(|c: char| c.is_whitespace() || c == '('))
}

/// Recursively rewrite an expression, truncating the wider side of any `+ - * /` between
/// two known float-vectors of differing width. Returns Some(new) ONLY if something
/// changed — callers keep the ORIGINAL slice verbatim on None, so unchanged text is never
/// reconstructed (literals/spacing preserved). Width decisions re-infer from the
/// POST-recursion child text so left-associative chains (`vec4a - vec3b + vec4c`) resolve
/// correctly.
fn rewrite_expr_width(e: &str, t: &TypeTable) -> Option<String> {
    // 1. top-level binary operator (LAST = left-associative)
    if let Some((l, op, r)) = split_binop_last(e) {
        let lnew = rewrite_expr_width(l, t);
        let rnew = rewrite_expr_width(r, t);
        let mut ltext = lnew.clone().unwrap_or_else(|| l.to_string());
        let mut rtext = rnew.clone().unwrap_or_else(|| r.to_string());
        let mut acted = false;
        let lt = infer_ty(&ltext, t);
        let rt = infer_ty(&rtext, t);
        if matches!(op, "+" | "-" | "*" | "/" | "%") {
            if let Some(numeric) = bool_to_numeric_expr(&ltext, lt) {
                ltext = numeric;
                acted = true;
            }
            if let Some(numeric) = bool_to_numeric_expr(&rtext, rt) {
                rtext = numeric;
                acted = true;
            }
        } else if matches!(op, "&&" | "||") {
            if let Some(boolean) = numeric_to_bool_expr(&ltext, lt) {
                ltext = boolean;
                acted = true;
            }
            if let Some(boolean) = numeric_to_bool_expr(&rtext, rt) {
                rtext = boolean;
                acted = true;
            }
        }
        if op == "%" {
            let lt = infer_ty(&ltext, t);
            let rt = infer_ty(&rtext, t);
            let lw = gty_width(lt);
            let rw = gty_width(rt);
            if lw > 0
                && rw > 0
                && matches!(lt, GTy::F | GTy::V(_))
                && matches!(rt, GTy::F | GTy::V(_))
            {
                if lw >= 2 && rw >= 2 && lw != rw {
                    let w = lw.min(rw) as usize;
                    let sw = &"xyzw"[..w];
                    if lw > rw {
                        ltext = format!("({}).{sw}", ltext.trim());
                    } else {
                        rtext = format!("({}).{sw}", rtext.trim());
                    }
                } else if lw == 1 && rw >= 2 {
                    rtext = rtext.trim().to_string();
                    ltext = format!("vec{rw}({})", ltext.trim());
                }
                return Some(format!("mod({ltext}, {rtext})"));
            }
        }
        if matches!(op, "+" | "-" | "*" | "/") {
            if let (GTy::V(a), GTy::V(b)) = (infer_ty(&ltext, t), infer_ty(&rtext, t)) {
                if a >= 2 && b >= 2 && a != b {
                    let m = a.min(b) as usize;
                    let sw = &"xyzw"[..m];
                    if a > b {
                        ltext = format!("({}).{sw}", ltext.trim());
                    } else {
                        rtext = format!("({}).{sw}", rtext.trim());
                    }
                    acted = true;
                }
            }
        }
        if acted || lnew.is_some() || rnew.is_some() {
            return Some(format!("{ltext}{op}{rtext}"));
        }
        return None;
    }
    // 2. no top-level binop — structural recursion.
    let trimmed = e.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lead = &e[..e.len() - e.trim_start().len()];
    let trail = &e[e.trim_end().len()..];
    // 2a. leading unary - + !
    if matches!(trimmed.as_bytes()[0], b'-' | b'+' | b'!') {
        let sign = &trimmed[..1];
        let rest = &trimmed[1..];
        if let Some(rn) = rewrite_expr_width(rest, t) {
            return Some(format!("{lead}{sign}{rn}{trail}"));
        }
        return None;
    }
    // 2b. fully enclosing parens
    if trimmed.starts_with('(') {
        if let Some(close) = matching_close(trimmed, 0) {
            if close == trimmed.len() - 1 {
                let inner = &trimmed[1..close];
                if let Some(inr) = rewrite_expr_width(inner, t) {
                    return Some(format!("{lead}({inr}){trail}"));
                }
                return None;
            }
        }
    }
    // 2c. trailing swizzle  base.[xyzwrgba]+
    if let Some(dot) = trailing_member_dot(trimmed) {
        let base = &trimmed[..dot];
        let member = &trimmed[dot..];
        if let Some(bn) = rewrite_expr_width(base, t) {
            return Some(format!("{lead}{bn}{member}{trail}"));
        }
        return None;
    }
    // 2d. function / constructor call  name(args)  spanning the whole expr
    if let Some((open, close)) = whole_call_span(trimmed) {
        let name = &trimmed[..open];
        let args = &trimmed[open + 1..close];
        let parts = split_top_level_commas(args);
        let mut changed = false;
        let mut newparts: Vec<String> = Vec::with_capacity(parts.len());
        for p in &parts {
            match rewrite_expr_width(p, t) {
                Some(np) => {
                    changed = true;
                    newparts.push(np);
                }
                None => newparts.push(p.clone()),
            }
        }
        if changed {
            return Some(format!("{lead}{name}({}){trail}", newparts.join(",")));
        }
        return None;
    }
    // 2e. ternary  c ? a : b
    if let Some(q) = find_top_level_char(trimmed, b'?') {
        if let Some(colon_rel) = find_top_level_char(&trimmed[q + 1..], b':') {
            let cond = &trimmed[..q];
            let a = &trimmed[q + 1..q + 1 + colon_rel];
            let b = &trimmed[q + 1 + colon_rel + 1..];
            let cn = rewrite_expr_width(cond, t);
            let an = rewrite_expr_width(a, t);
            let bn = rewrite_expr_width(b, t);
            if cn.is_some() || an.is_some() || bn.is_some() {
                let cs = cn.unwrap_or_else(|| cond.to_string());
                let as_ = an.unwrap_or_else(|| a.to_string());
                let bs = bn.unwrap_or_else(|| b.to_string());
                return Some(format!("{lead}{cs}?{as_}:{bs}{trail}"));
            }
        }
    }
    None
}

/// HLSL treats comparisons and bool variables as 0/1 when they participate in
/// arithmetic. GLSL keeps bool and numeric types disjoint, so make that legacy
/// coercion explicit before width repair handles the rest of the expression.
fn bool_to_numeric_expr(expr: &str, ty: GTy) -> Option<String> {
    match ty {
        GTy::B => Some(format!("float({})", expr.trim())),
        GTy::BV(width) => Some(format!("vec{width}({})", expr.trim())),
        _ => None,
    }
}

/// The inverse legacy coercion for logical operators. Numeric vectors follow
/// HLSL's aggregate truthiness: the expression is true when any lane is non-zero.
fn numeric_to_bool_expr(expr: &str, ty: GTy) -> Option<String> {
    match ty {
        GTy::F => Some(format!("(({}) != 0.0)", expr.trim())),
        GTy::V(width) => Some(format!("any(notEqual({}, vec{width}(0.0)))", expr.trim())),
        _ => None,
    }
}

/// LAST top-level binary operator of the lowest precedence present (left-associative —
/// `a - b + c` splits at the `+`). Mirrors split_binop's operator detection (unary +/-
/// skip, `<=`/`>=`/`<<` handling).
fn split_binop_last(s: &str) -> Option<(&str, &'static str, &str)> {
    let b = s.as_bytes();
    let n = b.len();
    let is_operand_end =
        |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b')' || c == b']' || c == b'.';
    for group in PREC {
        let mut depth = 0i32;
        let mut last: Option<(usize, &'static str)> = None;
        let mut i = 0;
        while i < n {
            let c = b[i];
            if c == b'(' || c == b'[' {
                depth += 1;
                i += 1;
            } else if c == b')' || c == b']' {
                depth -= 1;
                i += 1;
            } else if depth == 0 {
                let mut adv = 1usize;
                for op in *group {
                    let ob = op.as_bytes();
                    if i + ob.len() <= n && &b[i..i + ob.len()] == ob {
                        let prev = (1..=i)
                            .rev()
                            .map(|k| b[k - 1])
                            .find(|c| !c.is_ascii_whitespace());
                        if (*op == "+" || *op == "-") && prev.map_or(true, |c| !is_operand_end(c)) {
                            continue;
                        }
                        let nextc = b.get(i + ob.len()).copied();
                        if matches!(*op, "<" | ">") && (nextc == Some(b'=') || nextc == Some(c)) {
                            continue;
                        }
                        if prev == Some(b'=') && (*op == "<" || *op == ">") {
                            continue;
                        }
                        last = Some((i, *op));
                        adv = ob.len();
                        break;
                    }
                }
                i += adv;
            } else {
                i += 1;
            }
        }
        if let Some((pos, op)) = last {
            return Some((&s[..pos], op, &s[pos + op.len()..]));
        }
    }
    None
}

/// Index of the `)` matching the `(` at byte `open` in `s` (s[open]=='('); None if unbalanced.
fn matching_close(s: &str, open: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// If `s` ends in `.<swizzle>` (pure x/y/z/w/r/g/b/a after the dot) whose base before the
/// dot is a balanced primary, return the dot index.
fn trailing_member_dot(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let n = b.len();
    if n < 2 {
        return None;
    }
    let mut k = n;
    while k > 0
        && matches!(
            b[k - 1],
            b'x' | b'y' | b'z' | b'w' | b'r' | b'g' | b'b' | b'a'
        )
    {
        k -= 1;
    }
    if k == n || k == 0 || b[k - 1] != b'.' {
        return None;
    }
    let dot = k - 1;
    let mut depth = 0i32;
    for &c in s[..dot].as_bytes() {
        match c {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    Some(dot)
}

/// If `s` is exactly `ident(args)` (an identifier then a paren group covering the rest),
/// return (open_paren_index, close_paren_index). Used to recurse into call arguments.
fn whole_call_span(s: &str) -> Option<(usize, usize)> {
    let b = s.as_bytes();
    let n = b.len();
    if n == 0 || !(b[0].is_ascii_alphabetic() || b[0] == b'_') {
        return None;
    }
    let mut j = 0;
    while j < n && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
        j += 1;
    }
    if j >= n || b[j] != b'(' {
        return None;
    }
    let close = matching_close(s, j)?;
    if close == n - 1 {
        Some((j, close))
    } else {
        None
    }
}

/// Repair a plain `<lhs> = <rhs>;` whose RHS width differs from the LHS width: HLSL
/// truncates a wider RHS (`float3 c = tex2D(...)`) and broadcasts a scalar
/// (`float3 c = 0.0`); naga rejects the mismatch ("type … doesn't match the type
/// stored"). Conservative: only fires when BOTH widths are confidently known and the
/// LHS is a declaration (`<type> name = …`) or a plain identifier in the table.
fn fix_assignment_width_mismatches(src: &str, global: &TypeTable) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_fn = false;
    let mut fn_level = 0i32;
    let mut depth = 0i32;
    let mut merged: TypeTable = global.clone();
    for line in src.lines() {
        let trimmed = line.trim();
        if !in_fn {
            if let Some((_ret, params)) = parse_fn_header(trimmed) {
                in_fn = true;
                fn_level = depth;
                merged = global.clone();
                for (nm, g) in params {
                    merged.insert(nm, g);
                }
            }
        }
        if in_fn {
            record_local_decls(trimmed, &mut merged);
        }
        let active = if in_fn { &merged } else { global };
        out.push_str(&fix_assign_line(line, active));
        out.push('\n');
        for c in line.bytes() {
            if c == b'{' {
                depth += 1;
            } else if c == b'}' {
                depth -= 1;
                if in_fn && depth <= fn_level {
                    in_fn = false;
                }
            }
        }
    }
    out
}

fn fix_assign_line(line: &str, t: &TypeTable) -> String {
    // Split off a trailing line comment (GLSL has no strings, so the first `//` starts
    // the comment) and reattach it verbatim, so a `zv = …; //##` statement is repaired
    // instead of skipped. trim() both ends so a trailing space after `;` is tolerated.
    let (code, comment) = match line.find("//") {
        Some(c) => (&line[..c], &line[c..]),
        None => (line, ""),
    };
    let mut out = String::with_capacity(line.len() + 16);
    let b = code.as_bytes();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut saw_semi = false;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b';' if depth == 0 => {
                out.push_str(&fix_assign_segment(&code[start..i], t));
                out.push(';');
                start = i + 1;
                saw_semi = true;
            }
            _ => {}
        }
    }
    if !saw_semi {
        return line.to_string();
    }
    out.push_str(&code[start..]);
    out.push_str(comment);
    out
}

fn fix_assign_segment(seg: &str, t: &TypeTable) -> String {
    let trimmed = seg.trim();
    let lead = &seg[..seg.len() - seg.trim_start().len()];
    let trail = &seg[seg.trim_end().len()..];
    if trimmed.is_empty() {
        return seg.to_string();
    }
    let body = trimmed;
    if let Some(prefix_len) = repairable_statement_prefix_len(body) {
        let prefix = &body[..prefix_len];
        let rest = &body[prefix_len..];
        let fixed = fix_assign_segment(rest, t);
        if fixed != rest {
            return format!("{lead}{prefix}{fixed}{trail}");
        }
    }
    // The legacy MilkDrop converter sometimes emits comma-expression statement
    // sequences, for example `ret = tex2D(...).z, ret -= offset;`. HLSL applies
    // the usual scalar-to-vector broadcast to each assignment independently, but
    // treating the whole sequence as one RHS hides the first assignment's type
    // from the repair below. Recurse into top-level comma clauses while preserving
    // the comma operator and all clause whitespace verbatim. Commas inside calls,
    // constructors, array indexing, and brace initializers are excluded by the
    // balanced splitter.
    let comma_clauses = split_top_level_commas(body);
    if comma_clauses.len() > 1 {
        let fixed: Vec<String> = comma_clauses
            .iter()
            .map(|clause| fix_assign_segment(clause, t))
            .collect();
        let joined = fixed.join(",");
        if joined != body {
            return format!("{lead}{joined}{trail}");
        }
    }
    let Some(eq) = find_plain_eq(body) else {
        // No plain `=`: try a compound assignment `<lhs> op= <rhs>` (e.g. `vec3 += vec4`,
        // which naga rejects). HLSL truncates the wider rhs; do the same. Only fires when
        // the lvalue is a confident vector and the rhs a strictly-wider confident vector
        // (`vec3 += float` stays — GLSL broadcasts a scalar). Single-line only.
        if let Some(op) = find_compound_assign(body) {
            let lhs = body[..op].trim();
            let rhs = body[op + 2..].trim();
            if !lhs.is_empty() && !rhs.is_empty() {
                let lw = gty_width(infer_ty(lhs, t));
                let rt = infer_ty(rhs, t);
                if lw >= 1 && gty_width(rt) > lw && matches!(rt, GTy::V(_)) {
                    let sw = &"xyzw"[..lw as usize];
                    let optok = &body[op..op + 2];
                    return format!("{lead}{lhs} {optok} ({rhs}).{sw}{trail}");
                }
            }
        }
        return seg.to_string();
    };
    let lhs = body[..eq].trim();
    let rhs = body[eq + 1..].trim();
    if lhs.is_empty() || rhs.is_empty() {
        return seg.to_string();
    }
    // LHS type: a `<type> name` declaration, or a plain identifier/expression in
    // the table. Keep bool distinct even though it has scalar width.
    let words: Vec<&str> = lhs.split_whitespace().collect();
    let lt = if words.len() == 2 && !lhs.contains(',') && !lhs.contains('[') {
        keyword_gty(words[0])
    } else if lhs.chars().all(|c| c.is_alphanumeric() || c == '_') {
        t.get(lhs).copied()
    } else {
        let inferred = infer_ty(lhs, t);
        (gty_width(inferred) >= 1).then_some(inferred)
    };
    let Some(lt) = lt.filter(|ty| gty_width(*ty) >= 1) else {
        return seg.to_string();
    };
    let lw = gty_width(lt);
    let rt = infer_ty(rhs, t);
    let rw = gty_width(rt);
    if rw == 0 {
        return seg.to_string();
    }
    let new_rhs = if lt == GTy::F && rt == GTy::B {
        format!("float({rhs})")
    } else if matches!(lt, GTy::V(_)) && rt == GTy::B {
        format!("vec{lw}(float({rhs}))")
    } else if lt == GTy::B && rt == GTy::F {
        format!("({rhs}) != 0.0")
    } else if rw == lw {
        return seg.to_string(); // already matching
    } else if rt == GTy::F && lw > 1 {
        format!("vec{lw}({rhs})") // scalar → broadcast
    } else if rw > lw {
        let sw = &"xyzw"[..lw as usize];
        format!("({rhs}).{sw}") // wider vector → truncate
    } else {
        return seg.to_string(); // narrower vector (vec2→vec3): can't safely widen
    };
    format!("{lead}{lhs} = {new_rhs}{trail}")
}

/// HLSL permits compound assignment from floating expressions to an `int` and
/// truncates the result. GLSL/naga requires an explicit conversion. Track names
/// that are declared exclusively as `int` in this body and wrap their compound
/// RHS in `int(...)`. Casting an already-integer RHS is harmless, while avoiding
/// broad scalar-kind changes to the vector-width type system above.
fn fix_int_compound_assignments(src: &str) -> String {
    use std::collections::{HashMap, HashSet};

    let mut kinds: HashMap<String, bool> = HashMap::new();
    for line in src.lines() {
        for segment in line.split(';') {
            let trimmed = segment.trim().rsplit('{').next().unwrap_or(segment).trim();
            let mut words = trimmed.splitn(2, char::is_whitespace);
            let Some(ty) = words.next() else { continue };
            let Some(rest) = words.next() else { continue };
            if keyword_gty(ty).is_none() {
                continue;
            }
            let names = rest.split('=').next().unwrap_or(rest);
            for item in split_top_level_commas(names) {
                let name: String = item
                    .trim()
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if name.is_empty() {
                    continue;
                }
                let is_int = ty == "int";
                kinds
                    .entry(name)
                    .and_modify(|only_int| *only_int &= is_int)
                    .or_insert(is_int);
            }
        }
    }
    let ints: HashSet<String> = kinds
        .into_iter()
        .filter_map(|(name, only_int)| only_int.then_some(name))
        .collect();
    if ints.is_empty() {
        return src.to_string();
    }

    let mut out = String::with_capacity(src.len() + 16);
    for line in src.lines() {
        out.push_str(&fix_int_compound_line(line, &ints));
        out.push('\n');
    }
    out
}

fn fix_int_compound_line(line: &str, ints: &std::collections::HashSet<String>) -> String {
    let (code, comment) = match line.find("//") {
        Some(pos) => (&line[..pos], &line[pos..]),
        None => (line, ""),
    };
    let mut out = String::with_capacity(line.len() + 8);
    let mut start = 0usize;
    let mut depth = 0i32;
    for (i, byte) in code.bytes().enumerate() {
        match byte {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b';' if depth == 0 => {
                out.push_str(&fix_int_compound_segment(&code[start..i], ints));
                out.push(';');
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push_str(&code[start..]);
    out.push_str(comment);
    out
}

fn fix_int_compound_segment(segment: &str, ints: &std::collections::HashSet<String>) -> String {
    let trimmed = segment.trim();
    let lead = &segment[..segment.len() - segment.trim_start().len()];
    let trail = &segment[segment.trim_end().len()..];
    if let Some(rest) = trimmed.strip_prefix("int ") {
        let name_end = rest
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        let name = &rest[..name_end];
        let after_name = rest[name_end..].trim_start();
        if !name.is_empty()
            && ints.contains(name)
            && after_name.starts_with('=')
            && !after_name.starts_with("==")
        {
            let rhs = after_name[1..].trim();
            if !rhs.is_empty() && !rhs.starts_with("int(") {
                return format!("{lead}int {name} = int({rhs}){trail}");
            }
        }
    }
    let Some(op) = find_compound_assign(trimmed) else {
        return segment.to_string();
    };
    let lhs = trimmed[..op].trim();
    let rhs = trimmed[op + 2..].trim();
    if !ints.contains(lhs) || rhs.is_empty() || rhs.starts_with("int(") {
        return segment.to_string();
    }
    let operator = &trimmed[op..op + 2];
    format!("{lead}{lhs} {operator} int({rhs}){trail}")
}

/// HLSL accepts logical-not on numeric values. GLSL requires a bool operand, and
/// the result must be converted back to float when it flows into arithmetic.
/// Lower `!numeric_ident` to a zero comparison in control headers, or to a 0/1
/// float elsewhere.
fn fix_numeric_logical_not(src: &str, global: &TypeTable) -> String {
    let mut out = String::with_capacity(src.len() + 32);
    let mut in_fn = false;
    let mut fn_level = 0i32;
    let mut depth = 0i32;
    let mut merged = global.clone();
    for line in src.lines() {
        let trimmed = line.trim();
        if !in_fn {
            if let Some((_ret, params)) = parse_fn_header(trimmed) {
                in_fn = true;
                fn_level = depth;
                merged = global.clone();
                for (name, ty) in params {
                    merged.insert(name, ty);
                }
            }
        }
        if in_fn {
            record_local_decls(trimmed, &mut merged);
        }
        out.push_str(&fix_numeric_logical_not_line(
            line,
            if in_fn { &merged } else { global },
        ));
        out.push('\n');
        for byte in line.bytes() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if in_fn && depth <= fn_level {
                        in_fn = false;
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn fix_numeric_logical_not_line(line: &str, table: &TypeTable) -> String {
    let bytes = line.as_bytes();
    let trimmed = line.trim_start();
    let leading = line.len() - trimmed.len();
    let control_span = ["if", "while"].iter().find_map(|keyword| {
        let rest = trimmed.strip_prefix(keyword)?;
        if rest
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            return None;
        }
        let open_rel = rest.find('(')? + keyword.len();
        let open = leading + open_rel;
        Some((open, matching_close(line, open)?))
    });
    let mut out = String::with_capacity(line.len() + 16);
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'!' || bytes.get(i + 1) == Some(&b'=') {
            let ch = line[i..].chars().next().expect("i is in bounds");
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        let mut start = i + 1;
        while start < bytes.len() && bytes[start].is_ascii_whitespace() {
            start += 1;
        }
        let mut end = start;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        if end == start || table.get(&line[start..end]).copied() != Some(GTy::F) {
            out.push('!');
            i += 1;
            continue;
        }
        let ident = &line[start..end];
        let as_bool = control_span.is_some_and(|(open, close)| i > open && i < close);
        if as_bool {
            out.push_str(&format!("(float({ident}) == 0.0)"));
        } else {
            out.push_str(&format!("float(float({ident}) == 0.0)"));
        }
        i = end;
    }
    out
}

/// HLSL treats any non-zero numeric value as true in control-flow headers.
/// GLSL/naga require an actual bool. Repair complete `if`/`while` conditions
/// after type inference, including scalar flags and numeric vector masks.
fn fix_numeric_control_conditions(src: &str, table: &TypeTable) -> String {
    let mut out = String::with_capacity(src.len() + 32);
    for line in src.lines() {
        let mut fixed = line.to_string();
        for keyword in ["if", "while"] {
            let mut search_from = 0usize;
            loop {
                let Some(relative) = fixed[search_from..].find(keyword) else {
                    break;
                };
                let start = search_from + relative;
                let before = start.checked_sub(1).and_then(|i| fixed.as_bytes().get(i));
                let after_keyword = start + keyword.len();
                if before.is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                    || fixed
                        .as_bytes()
                        .get(after_keyword)
                        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                {
                    search_from = after_keyword;
                    continue;
                }
                let mut open = after_keyword;
                while fixed
                    .as_bytes()
                    .get(open)
                    .is_some_and(|byte| byte.is_ascii_whitespace())
                {
                    open += 1;
                }
                if fixed.as_bytes().get(open) != Some(&b'(') {
                    search_from = after_keyword;
                    continue;
                }
                let Some(close) = matching_close(&fixed, open) else {
                    break;
                };
                let condition = fixed[open + 1..close].trim();
                let replacement = match infer_ty(condition, table) {
                    GTy::F => Some(format!("({condition}) != 0.0")),
                    GTy::V(width) => Some(format!("any(notEqual({condition}, vec{width}(0.0)))")),
                    _ => None,
                };
                if let Some(replacement) = replacement {
                    fixed.replace_range(open + 1..close, &replacement);
                    search_from = open + 1 + replacement.len() + 1;
                } else {
                    search_from = close + 1;
                }
            }
        }
        out.push_str(&fixed);
        out.push('\n');
    }
    out
}

/// Some converter output keeps control/block prefixes on the same physical segment,
/// e.g. `if(cond)ret += vec4;` or `else {ret = vec4;}`. The assignment/op fixers need
/// to see the actual statement start (`ret ...`), so peel those prefixes first.
fn repairable_statement_prefix_len(body: &str) -> Option<usize> {
    if let Some(pos) = body.rfind('{') {
        let idx = pos + 1;
        if idx < body.len() && !body[idx..].trim().is_empty() {
            return Some(idx);
        }
    }
    let trimmed = body.trim_start();
    let leading = body.len() - trimmed.len();
    if trimmed.starts_with("if") {
        let after_if = 2usize;
        let rest = &trimmed[after_if..];
        if rest
            .chars()
            .next()
            .is_some_and(|c| c.is_whitespace() || c == '(')
        {
            let open_rel = rest.find('(')? + after_if;
            let close = matching_close(trimmed, open_rel)?;
            let idx = leading + close + 1;
            if idx < body.len() && !body[idx..].trim().is_empty() {
                return Some(idx);
            }
        }
    }
    None
}

/// First top-level compound-assignment operator `+= -= *= /=`; returns the index of the
/// op char (so the token is `s[i..i+2]`). Skips `==`/`<=`/`>=`/`!=` (prev isn't +-*/).
fn find_compound_assign(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    for i in 1..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'=' if depth == 0
                && matches!(b[i - 1], b'+' | b'-' | b'*' | b'/')
                && b.get(i + 1) != Some(&b'=') =>
            {
                return Some(i - 1);
            }
            _ => {}
        }
    }
    None
}

/// First top-level plain assignment `=` (not `==`/`<=`/`>=`/`!=` or a compound `op=`).
fn find_plain_eq(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'=' if depth == 0 => {
                let prev = if i > 0 { b[i - 1] } else { b' ' };
                let next = if i + 1 < b.len() { b[i + 1] } else { b' ' };
                if next != b'='
                    && !matches!(
                        prev,
                        b'=' | b'<' | b'>' | b'!' | b'+' | b'-' | b'*' | b'/' | b'%'
                    )
                {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Promote scalar call arguments to vector parameters: HLSL implicitly broadcasts a
/// scalar passed where a vector is expected (`f(0.5)` for `f(float2 c)`); GLSL/naga
/// rejects it ("no matching overloaded function" / "Unknown function 'mix'"). For each
/// call to a function whose parameter types we know (user functions + `mix`/`pow`),
/// wrap a scalar argument in `vecN(arg)` when the parameter is a width-N vector. Only
/// fires when both the parameter width and the argument's scalar-ness are known.
fn fix_call_arg_promotion(src: &str, t: &TypeTable, sigs: &HashMap<String, Vec<GTy>>) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(n + 64);
    let mut i = 0;
    while i < n {
        // identifier immediately followed by '('?
        if (b[i].is_ascii_alphabetic() || b[i] == '_')
            && (i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_'))
        {
            let mut j = i;
            while j < n && (b[j].is_ascii_alphanumeric() || b[j] == '_') {
                j += 1;
            }
            let name: String = b[i..j].iter().collect();
            if j < n && b[j] == '(' {
                // find the matching close paren
                let mut depth = 0i32;
                let mut k = j;
                while k < n {
                    match b[k] {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                if k < n {
                    let args_str: String = b[j + 1..k].iter().collect();
                    // recurse into the arguments first (nested calls)
                    let inner_fixed = fix_call_arg_promotion(&args_str, t, sigs);
                    let want = param_widths(&name, &inner_fixed, sigs, t);
                    let args = split_top_level_commas(&inner_fixed);
                    let mut new_args: Vec<String> = Vec::with_capacity(args.len());
                    for (ai, a) in args.iter().enumerate() {
                        let at = a.trim();
                        if let (Some(Some(w)), false) = (want.get(ai).copied(), at.is_empty()) {
                            let aty = infer_ty(at, t);
                            let aw = gty_width(aty);
                            if w == 1 {
                                // scalar param: truncate a confident vector arg to `.x`
                                if aw >= 2 {
                                    new_args.push(format!("({at}).x"));
                                    continue;
                                }
                            } else if aty == GTy::F {
                                // broadcast a confirmed scalar up to the vector param width
                                new_args.push(format!("vec{w}({at})"));
                                continue;
                            } else if aw > w {
                                // truncate a wider vector arg down to the param width
                                let sw = &"xyzw"[..w as usize];
                                new_args.push(format!("({at}).{sw}"));
                                continue;
                            }
                        }
                        new_args.push(a.clone());
                    }
                    out.push_str(&name);
                    out.push('(');
                    out.push_str(&new_args.join(","));
                    out.push(')');
                    i = k + 1;
                    continue;
                }
            }
            out.push_str(&name);
            i = j;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Per-argument target widths for a call to a known user function: Some(w) means the
/// parameter is a width-w vector, None means leave the argument alone.
fn param_widths(
    name: &str,
    args_str: &str,
    sigs: &HashMap<String, Vec<GTy>>,
    t: &TypeTable,
) -> Vec<Option<u8>> {
    // MilkDrop's sampling helpers are authored as float2-coordinate functions,
    // but HLSL accepts both scalar splats (`GetPixel(0.5)`) and wider vectors
    // (`GetBlur3(tex2D(...))`, implicitly taking `.xy`). The generated preamble
    // deliberately exposes multiple helper overloads, so signature collection
    // cannot choose this coercion on its own.
    if matches!(name, "GetPixel" | "GetBlur1" | "GetBlur2" | "GetBlur3") {
        return vec![Some(2)];
    }
    if let Some(sig) = sigs.get(name) {
        return sig
            .iter()
            .map(|g| match g {
                GTy::V(w) => Some(*w),
                // Some(1) marks a known SCALAR param: a confident vector arg passed
                // here is truncated to `.x` (HLSL vector->scalar, e.g. `lavcol((ret*2).x)`).
                GTy::F => Some(1),
                _ => None,
            })
            .collect();
    }
    // genType builtins: all arguments must share a width. HLSL broadcasts a scalar
    // (`mix(0.0, vec3(…), t)`, `pow(vec, 0.5)`, `max(scalar, vec)`) → naga "Unknown
    // function". Infer the widest argument and request that width for every argument
    // (the caller broadcasts only confirmed scalars, so vector args are left alone).
    // These builtins all accept the all-vector form, so over-requesting is safe.
    if matches!(
        name,
        "mix"
            | "pow"
            | "max"
            | "min"
            | "clamp"
            | "smoothstep"
            | "step"
            | "mod"
            | "atan"
            | "reflect"
    ) {
        // Target the MINIMUM known-vector width (HLSL truncates wider operands to the
        // narrowest), ignoring Unknown args. Scalars still broadcast up to this width
        // in the call loop; known wider vectors are truncated down. Previously this used
        // max() and only broadcast scalars, so `max(vec3, vec4)` stayed unrepaired.
        let args = split_top_level_commas(args_str);
        let min_vec = args
            .iter()
            .map(|a| gty_width(infer_ty(a.trim(), t)))
            .filter(|&w| w >= 2)
            .min();
        if let Some(w) = min_vec {
            return vec![Some(w); args.len()];
        }
    }
    Vec::new()
}

/// Rewrite `A <cmp> B` where one side is a float-vector and the other a scalar into
/// `vecN(lessThan(A, vecN(B)))` (and greaterThan/…): naga rejects `vec < scalar`, and
/// these comparisons are used as bool-as-number multipliers in MilkDrop presets, so the
/// vecN(...) wrapper makes the result usable in arithmetic. When both sides are vectors
/// of differing width, the wider operand is truncated to the narrower so the componentwise
/// builtin has a matching overload. Only fires when inference is confident at least one
/// side is a vector — scalar comparisons (valid bool) are untouched.
fn fix_vector_relops(src: &str, t: &TypeTable) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(n + 64);
    let mut i = 0;
    while i < n {
        // find a top-level relational operator inside the current "primary group":
        // we scan for `<`, `>`, `<=`, `>=` not part of `<<`,`->`,`=<` etc.
        let c = b[i];
        if (c == '<' || c == '>') && !(i + 1 < n && b[i + 1] == '<') && !(i > 0 && b[i - 1] == '=')
        {
            let two = i + 1 < n && b[i + 1] == '=';
            let op: String = if two { format!("{c}=") } else { c.to_string() };
            // expand left + right operands around the operator (balanced primaries)
            if let (Some(ls), Some(re)) = (relop_left(&b, i), relop_right(&b, i + op.len())) {
                let left: String = b[ls..i].iter().collect();
                let right: String = b[i + op.len()..re].iter().collect();
                let lt = infer_ty(left.trim(), t);
                let rt = infer_ty(right.trim(), t);
                let lv = matches!(lt, GTy::V(_));
                let rv = matches!(rt, GTy::V(_));
                // componentwise builtins (lessThan, …) require both operands the same
                // vector width. When both sides are vectors of differing width, the
                // result/wrapper width is the MIN and the wider side is truncated;
                // when one side is scalar it is broadcast up to the vector width (max).
                let w = if lv && rv {
                    gty_width(lt).min(gty_width(rt))
                } else {
                    gty_width(lt).max(gty_width(rt))
                };
                // only act when confidently a vector vs (vector|scalar) comparison
                if w > 1 && (lv || rv) && lt != GTy::Unknown && rt != GTy::Unknown {
                    let fname = match op.as_str() {
                        "<" => "lessThan",
                        ">" => "greaterThan",
                        "<=" => "lessThanEqual",
                        ">=" => "greaterThanEqual",
                        _ => unreachable!(),
                    };
                    // mirror rewrite_expr_width: truncate the wider operand to `w`.
                    let sw = &"xyzw"[..w as usize];
                    let lexpr = if lv {
                        if gty_width(lt) > w {
                            format!("({}).{sw}", left.trim())
                        } else {
                            left.trim().to_string()
                        }
                    } else {
                        format!("vec{w}({})", left.trim())
                    };
                    let rexpr = if rv {
                        if gty_width(rt) > w {
                            format!("({}).{sw}", right.trim())
                        } else {
                            right.trim().to_string()
                        }
                    } else {
                        format!("vec{w}({})", right.trim())
                    };
                    // drop whatever we had buffered for the left operand, re-emit wrapped
                    let keep_chars = out.chars().count().saturating_sub(i - ls);
                    let keep = byte_index_after_chars(&out, keep_chars);
                    out.truncate(keep);
                    out.push_str(&format!("vec{w}({fname}({lexpr}, {rexpr}))"));
                    i = re;
                    continue;
                }
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn byte_index_after_chars(s: &str, count: usize) -> usize {
    if count == 0 {
        return 0;
    }
    match s.char_indices().nth(count) {
        Some((idx, _)) => idx,
        None => s.len(),
    }
}

/// Extent of the operand to the LEFT of a relational operator at `op`: a balanced
/// primary (handles `a.b`, `f(x)`, `m[i]`, leading unary). Returns its start index.
fn relop_left(b: &[char], op: usize) -> Option<usize> {
    let mut k = op as isize - 1;
    while k >= 0 && b[k as usize].is_whitespace() {
        k -= 1;
    }
    if k < 0 {
        return None;
    }
    // consume a balanced suffix chain: ) ] then identifier/.member, repeatedly
    loop {
        if k < 0 {
            break;
        }
        let c = b[k as usize];
        if c == ')' || c == ']' {
            let (open, close) = if c == ')' { ('(', ')') } else { ('[', ']') };
            let mut depth = 0i32;
            while k >= 0 {
                let cc = b[k as usize];
                if cc == close {
                    depth += 1;
                } else if cc == open {
                    depth -= 1;
                    if depth == 0 {
                        k -= 1;
                        break;
                    }
                }
                k -= 1;
            }
        } else if c.is_alphanumeric() || c == '_' || c == '.' {
            while k >= 0
                && (b[k as usize].is_alphanumeric() || b[k as usize] == '_' || b[k as usize] == '.')
            {
                k -= 1;
            }
        } else {
            break;
        }
    }
    Some((k + 1) as usize)
}

/// Extent of the operand to the RIGHT of a relational operator starting at `r0`:
/// a balanced primary. Returns the index just past it.
fn relop_right(b: &[char], r0: usize) -> Option<usize> {
    let n = b.len();
    let mut k = r0;
    while k < n && b[k].is_whitespace() {
        k += 1;
    }
    // leading unary
    while k < n && matches!(b[k], '-' | '+' | '!') {
        k += 1;
        while k < n && b[k].is_whitespace() {
            k += 1;
        }
    }
    if k >= n {
        return None;
    }
    // identifier / number / call / index chain
    loop {
        if k >= n {
            break;
        }
        let c = b[k];
        if c.is_alphanumeric() || c == '_' || c == '.' {
            while k < n && (b[k].is_alphanumeric() || b[k] == '_' || b[k] == '.') {
                k += 1;
            }
        } else if c == '(' || c == '[' {
            let (open, close) = if c == '(' { ('(', ')') } else { ('[', ']') };
            let mut depth = 0i32;
            while k < n {
                let cc = b[k];
                if cc == open {
                    depth += 1;
                } else if cc == close {
                    depth -= 1;
                    if depth == 0 {
                        k += 1;
                        break;
                    }
                }
                k += 1;
            }
        } else {
            break;
        }
    }
    if k == r0 {
        None
    } else {
        Some(k)
    }
}

pub fn hlsl_milk_body_to_naga(body: &str) -> String {
    // Split at `shader_body { }` to get file-scope globals and the body separately.
    let (before, inner) = split_shader_body_wrapper(body);
    #[cfg(feature = "milk-native-converter")]
    {
        if !before.is_empty() && before_has_function_defs(&before) {
            let (func_defs, non_func_rest) = extract_function_defs(&before);
            let body_src = if non_func_rest.trim().is_empty() {
                inner.clone()
            } else {
                format!("{}\n{}", non_func_rest.trim(), inner)
            };
            let union = format!("{func_defs}\n{body_src}");
            let (_, customs) = strip_and_alias_hlsl_samplers(&union);
            let file_globals = hlsl_pre_native_fixups(&alias_with_customs(&func_defs, &customs));
            let (deduped_body, _) = dedup_hlsl_declarations(&hlsl_pre_native_fixups(
                &alias_with_customs(&body_src, &customs),
            ));
            if let Some(glsl_body) = try_native_convert_hlsl_ex(&file_globals, &deduped_body) {
                return glsl_milk_body_to_naga(&glsl_body);
            }
        } else {
            // Combine before + inner, strip any HLSL sampler decls that would cause
            // "opaque variable must be declared uniform" failures in glsl-optimizer,
            // then dedup and try the native converter.
            let combined = if before.is_empty() {
                inner.clone()
            } else {
                format!("{before}\n{inner}")
            };
            let (pre_stripped, custom_samplers_c) = strip_and_alias_hlsl_samplers(&combined);
            let mut pre_stripped = hlsl_pre_native_fixups(&pre_stripped);
            pre_stripped = alias_custom_sampler_refs(pre_stripped, &custom_samplers_c);
            let (deduped, _) = dedup_hlsl_declarations(&pre_stripped);
            if let Some(glsl_body) = try_native_convert_hlsl(&deduped) {
                return glsl_milk_body_to_naga(&glsl_body);
            }
        }
    }
    // Helper functions in the `before` block must be hoisted to file scope (with
    // their referenced globals) or they end up nested inside main() → naga error.
    if before_has_function_defs(&before) {
        return hlsl_comp_fallback_hoisted(&before, &inner);
    }
    let combined = if before.is_empty() {
        inner
    } else {
        format!("{before}\n{inner}")
    };
    let (stripped, custom_samplers) = strip_and_alias_hlsl_samplers(&combined);
    let mut stripped = stripped;
    stripped = alias_custom_sampler_refs(stripped, &custom_samplers);
    let converted_body = hlsl_to_glsl_body(&stripped);
    let gamma_postlude = comp_gamma_postlude(&converted_body);

    let io_decls = "\
layout(location = 0) in  vec2 vUv;
layout(location = 1) in  vec4 vColor;
layout(location = 0) out vec4 fragColor;";
    let preamble = milk_fs_preamble(io_decls);

    format!(
        r#"{preamble}
void main() {{
    vec3 ret = vec3(0.0);
    vec2 uv = vUv;
    vec2 uv_orig = vUv;
    uv.y = 1.0 - uv.y;
    uv_orig.y = 1.0 - uv_orig.y;
    float rad = length(uv - 0.5);
    float ang = atan(uv.x - 0.5, uv.y - 0.5);
    // MilkDrop/Butterchurn comp "hue_shader": four time-varying corner colors,
    // bilinearly interpolated across the screen (CompShader.generateHueBase).
    // comp_19 re-colors the grayscale luminance via pow(hue_shader, ret).
    // Previously vColor was a constant white (1,1,1) -> pow()->1 -> no color.
    float _ht = time * 30.0;
    vec4 _hr = rand_start; // rand_start phase offsets: r=.w, g=.y, b=.z
    vec3 _hc0 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+ 0.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+ 0.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+ 0.0+_hr.z));
    _hc0 /= max(_hc0.x, max(_hc0.y, _hc0.z)); _hc0 = 0.5 + 0.5*_hc0;
    vec3 _hc1 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+21.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+13.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+ 9.0+_hr.z));
    _hc1 /= max(_hc1.x, max(_hc1.y, _hc1.z)); _hc1 = 0.5 + 0.5*_hc1;
    vec3 _hc2 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+42.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+26.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+18.0+_hr.z));
    _hc2 /= max(_hc2.x, max(_hc2.y, _hc2.z)); _hc2 = 0.5 + 0.5*_hc2;
    vec3 _hc3 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+63.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+39.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+27.0+_hr.z));
    _hc3 /= max(_hc3.x, max(_hc3.y, _hc3.z)); _hc3 = 0.5 + 0.5*_hc3;
    float _hx = vUv.x;
    float _hy = vUv.y;
    vec3 hue_shader = _hc0*_hx*_hy + _hc1*(1.0-_hx)*_hy
                    + _hc2*_hx*(1.0-_hy) + _hc3*(1.0-_hx)*(1.0-_hy);

{converted_body}

{gamma_postlude}
    fragColor = vec4(ret, 1.0);
}}
"#
    )
}

/// Convert a raw HLSL .milk WARP shader body to a complete naga-compatible GLSL 450
/// fragment shader driven by the WARPED MESH vertex shader.
///
/// Varyings (declaration order fixes naga in-locations 0,1,2):
///   vUv     (loc 0) — screen position 0..1 (DirectX-UV, v=0 top) → rad/ang
///   vWarpUv (loc 1) — CPU per-vertex warped sample coord (already DirectX-UV)
///   vDecay  (loc 2) — per-vertex decay rgb (a unused)
///
/// `uv` is the warped sample coord (NO y-flip — the CPU already produced
/// DirectX-UV); `uv_orig` is the screen pos used for rad/ang. The final color is
/// multiplied by vDecay.rgb (the per-pixel decay extension).
pub fn hlsl_milk_warp_body_to_naga(body: &str) -> String {
    let (before, inner) = split_shader_body_wrapper(body);
    #[cfg(feature = "milk-native-converter")]
    {
        let native_result = if !before.is_empty() && before_has_function_defs(&before) {
            let (func_defs, non_func_rest) = extract_function_defs(&before);
            let body_src = if non_func_rest.trim().is_empty() {
                inner.clone()
            } else {
                format!("{}\n{}", non_func_rest.trim(), inner)
            };
            // Strip + alias samplers across funcs AND body together (collect custom
            // names from the union) so a `sampler2D X;` in the warp body doesn't reach
            // glslopt → "opaque variables must be declared uniform". Matches the COMP
            // pre-strip; previously only the function defs were sampler-stripped.
            let union = format!("{func_defs}\n{body_src}");
            let (_, customs) = strip_and_alias_hlsl_samplers(&union);
            let file_globals = hlsl_pre_native_fixups(&alias_with_customs(&func_defs, &customs));
            let (deduped_body, _) = dedup_hlsl_declarations(&hlsl_pre_native_fixups(
                &alias_with_customs(&body_src, &customs),
            ));
            try_native_convert_hlsl_ex(&file_globals, &deduped_body)
        } else {
            let stripped = if before.is_empty() {
                inner.clone()
            } else {
                format!("{before}\n{inner}")
            };
            let (_, customs) = strip_and_alias_hlsl_samplers(&stripped);
            let (deduped, _) = dedup_hlsl_declarations(&hlsl_pre_native_fixups(
                &alias_with_customs(&stripped, &customs),
            ));
            try_native_convert_hlsl(&deduped)
        };
        if let Some(glsl_body) = native_result {
            return glsl_milk_warp_body_to_naga(&glsl_body);
        }
    }
    if before_has_function_defs(&before) {
        return hlsl_warp_fallback_hoisted(&before, &inner);
    }
    let combined = if before.is_empty() {
        inner
    } else {
        format!("{before}\n{inner}")
    };
    let (stripped, custom_samplers) = strip_and_alias_hlsl_samplers(&combined);
    let mut stripped = stripped;
    stripped = alias_custom_sampler_refs(stripped, &custom_samplers);
    let converted_body = hlsl_to_glsl_body(&stripped);

    // Declaration order MUST be vUv(0), vWarpUv(1), vDecay(2) for naga locations.
    let io_decls = "\
layout(location = 0) in  vec2 vUv;
layout(location = 1) in  vec2 vWarpUv;
layout(location = 2) in  vec4 vDecay;
layout(location = 0) out vec4 fragColor;";
    let preamble = milk_fs_preamble(io_decls);

    format!(
        r#"{preamble}
void main() {{
    vec3 ret = vec3(0.0);
    // uv = warped sample coord from the mesh (DirectX-UV, v=0 top); NO y-flip.
    vec2 uv = vWarpUv;
    // uv_orig = screen position for rad/ang (butterchurn warp.js).
    vec2 uv_orig = vUv;
    float rad = length(uv_orig - 0.5);
    float ang = atan(uv_orig.x - 0.5, uv_orig.y - 0.5);

{converted_body}

    // Butterchurn applies decay ONLY in the default warp shader; a CUSTOM warp
    // shader self-decays (e.g. jelly_space does ret.x*=0.5, channels grow via max;
    // ORB does *(0.8+q3*0.1)). Its final composite is fragColor = ret * vColor with
    // vColor=white, i.e. NO extra decay. Multiplying by vDecay here double-decayed
    // custom-warp presets (jelly_space fDecay=0.5 → ×0.5/frame), flattening their
    // slow-growing feedback (the missing tendrils). vDecay stays for the default mesh.
    fragColor = vec4(ret, 1.0);
}}
"#
    )
}

/// Pure-Rust COMP fallback for `.milk` bodies whose `before` block defines helper
/// functions. The native converter rejects these (a function referencing a `before`
/// global → "undeclared identifier"), and the plain fallback inlines the function
/// defs into main() → naga `InvalidToken`. This path hoists the functions and the
/// `before` globals to file scope (non-const initializers become main-body
/// assignments), matching MilkDrop's original HLSL file layout.
fn hlsl_comp_fallback_hoisted(before: &str, inner: &str) -> String {
    let (func_src, glob_src) = extract_function_defs(before);
    let union = format!("{func_src}\n{glob_src}\n{inner}");
    let (_, customs) = strip_and_alias_hlsl_samplers(&union);

    // Convert the globals in DECLARATION form first (so HLSL truncation passes apply
    // to `vec3 x = <vec4 expr>;`), THEN split into file-scope decls + main-body
    // assignments. Function defs and globals keep cross-fragment referents → no
    // dead-decl dropping.
    let conv_funcs = hlsl_to_glsl_body_ex(&alias_with_customs(&func_src, &customs), false);
    let conv_globals = hlsl_to_glsl_body_ex(&alias_with_customs(&glob_src, &customs), false);
    let (conv_decls, conv_inits) = split_hlsl_globals(&conv_globals);
    let conv_inner = hlsl_to_glsl_body(&alias_with_customs(inner, &customs));
    let gamma_postlude = comp_gamma_postlude(&conv_inner);

    let io_decls = "\
layout(location = 0) in  vec2 vUv;
layout(location = 1) in  vec4 vColor;
layout(location = 0) out vec4 fragColor;";
    let preamble = milk_fs_preamble(io_decls);

    format!(
        r#"{preamble}
// Per-pixel values + `before` globals promoted to file scope so hoisted helper
// functions can reference them; main() assigns them before the body runs.
vec2  uv_orig;
float rad;
float ang;
vec3  hue_shader;
{conv_decls}
{conv_funcs}
void main() {{
    vec3 ret = vec3(0.0);
    vec2 uv = vUv;
    uv_orig = vUv;
    uv.y = 1.0 - uv.y;
    uv_orig.y = 1.0 - uv_orig.y;
    rad = length(uv - 0.5);
    ang = atan(uv.x - 0.5, uv.y - 0.5);
    float _ht = time * 30.0;
    vec4 _hr = rand_start;
    vec3 _hc0 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+ 0.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+ 0.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+ 0.0+_hr.z));
    _hc0 /= max(_hc0.x, max(_hc0.y, _hc0.z)); _hc0 = 0.5 + 0.5*_hc0;
    vec3 _hc1 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+21.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+13.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+ 9.0+_hr.z));
    _hc1 /= max(_hc1.x, max(_hc1.y, _hc1.z)); _hc1 = 0.5 + 0.5*_hc1;
    vec3 _hc2 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+42.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+26.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+18.0+_hr.z));
    _hc2 /= max(_hc2.x, max(_hc2.y, _hc2.z)); _hc2 = 0.5 + 0.5*_hc2;
    vec3 _hc3 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+63.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+39.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+27.0+_hr.z));
    _hc3 /= max(_hc3.x, max(_hc3.y, _hc3.z)); _hc3 = 0.5 + 0.5*_hc3;
    float _hx = vUv.x;
    float _hy = vUv.y;
    hue_shader = _hc0*_hx*_hy + _hc1*(1.0-_hx)*_hy
                    + _hc2*_hx*(1.0-_hy) + _hc3*(1.0-_hx)*(1.0-_hy);
{conv_inits}
{conv_inner}

{gamma_postlude}
    fragColor = vec4(ret, 1.0);
}}
"#
    )
}

/// Pure-Rust WARP fallback counterpart of `hlsl_comp_fallback_hoisted` (warped-mesh
/// varyings vUv@0/vWarpUv@1/vDecay@2; no hue_shader; no decay multiply).
fn hlsl_warp_fallback_hoisted(before: &str, inner: &str) -> String {
    let (func_src, glob_src) = extract_function_defs(before);
    let union = format!("{func_src}\n{glob_src}\n{inner}");
    let (_, customs) = strip_and_alias_hlsl_samplers(&union);

    let conv_funcs = hlsl_to_glsl_body_ex(&alias_with_customs(&func_src, &customs), false);
    let conv_globals = hlsl_to_glsl_body_ex(&alias_with_customs(&glob_src, &customs), false);
    let (conv_decls, conv_inits) = split_hlsl_globals(&conv_globals);
    let conv_inner = hlsl_to_glsl_body(&alias_with_customs(inner, &customs));

    let io_decls = "\
layout(location = 0) in  vec2 vUv;
layout(location = 1) in  vec2 vWarpUv;
layout(location = 2) in  vec4 vDecay;
layout(location = 0) out vec4 fragColor;";
    let preamble = milk_fs_preamble(io_decls);

    format!(
        r#"{preamble}
vec2  uv_orig;
float rad;
float ang;
{conv_decls}
{conv_funcs}
void main() {{
    vec3 ret = vec3(0.0);
    vec2 uv = vWarpUv;
    uv_orig = vUv;
    rad = length(uv_orig - 0.5);
    ang = atan(uv_orig.x - 0.5, uv_orig.y - 0.5);
{conv_inits}
{conv_inner}

    fragColor = vec4(ret, 1.0);
}}
"#
    )
}

// ---------------------------------------------------------------------------
// Path: Butterchurn converted-JSON GLSL shader body → naga-compatible GLSL 450
//
// Butterchurn's converted-JSON presets store the warp/comp shaders as already-GLSL
// BODIES wrapped in ` shader_body { ... } ` (the HLSL→GLSL conversion was done by
// the JS toolchain at export time). They are GLSL — NOT HLSL — so we must NOT run
// the HLSL type/function substitutions (no float3→vec3, no tex2D→texture, no mul()).
// We only:
//   1. strip the outer ` shader_body { ... } ` wrapper,
//   2. wrap the inner body in the SAME FS template as the HLSL path (same samplers,
//      PerFrame UBO, helpers, q-defines, texsize_noise_* consts),
//   3. rewrite `texture(name, ...)` → `texture(sampler2D(name, name_samp), ...)` for
//      our separated samplers (Butterchurn uses combined samplers).
// These bodies reference: texture(sampler_main,uv), texture(sampler_noise_lq,..),
// texsize, texsize_noise_lq, rand_frame, bass, treb, uv, uv_orig, ret.
// ---------------------------------------------------------------------------

/// Strip the ` shader_body { ... } ` wrapper from a Butterchurn GLSL body, returning
/// the inner statements. Tolerant of leading/trailing whitespace.
///
/// Many Butterchurn exports declare global variables BEFORE the wrapper, e.g.
/// `float sustain;\nvec3 xlat_mutableuv2;\n shader_body { ... }`. Those declarations
/// are used by the body and must be kept; only the `shader_body {` token and its
/// matching trailing `}` are removed. (The old prefix-only strip left the bare
/// `shader_body` token in the GLSL for this whole class of presets → naga
/// `UnknownVariable("shader_body")`, ~245 presets.)
fn strip_shader_body_wrapper(src: &str) -> String {
    let (before, inner) = split_shader_body_wrapper(src);
    if before.is_empty() {
        inner
    } else {
        format!("{before}\n{inner}")
    }
}

/// Split a .milk shader body at the `shader_body { }` boundary.
/// Returns `(file_globals, inner_body)` where:
///   - `file_globals` is everything before `shader_body` (may be empty)
///   - `inner_body`   is the content INSIDE the matched braces (or the whole src if no wrapper)
///
/// Uses brace-counting to find the matching `}` for the `{` after `shader_body`, so
/// trailing comments/content after the closing brace are correctly excluded.
fn split_shader_body_wrapper(src: &str) -> (String, String) {
    let t = src.trim();
    // Locate the `shader_body` keyword as a whole token in real code — NOT inside a
    // comment/string, and NOT a substring of a larger identifier (`my_shader_body`,
    // `shader_body2`). A plain `str::find("shader_body")` would match all of those.
    let Some(pos) = find_shader_body_keyword(t) else {
        return (String::new(), t.to_string());
    };
    let before = t[..pos].trim().to_string();
    let after_kw = t[pos + "shader_body".len()..].trim_start();
    // The opening `{` must also be found in code (a `{` inside `/* … */` between the
    // keyword and the real brace must not be mistaken for the wrapper open).
    let Some(open) = find_code_byte(after_kw, b'{') else {
        return (before, after_kw.to_string());
    };
    // Brace-count to the matching close, comment- and string-aware.
    let body_src = &after_kw[open + 1..];
    let end = scan_to_matching_brace(body_src);
    (before, body_src[..end].trim().to_string())
}

/// True for a byte that can appear inside a GLSL/HLSL identifier.
pub(crate) fn is_ident_byte(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphanumeric()
}

/// Byte offset of the first `shader_body` keyword that appears as a whole identifier
/// in real code: skips line comments (`//…`), block comments (`/* … */`), and
/// string/char literals, and rejects matches where an identifier char abuts either
/// side (so `my_shader_body` / `shader_body2` are not matched). Returns `None` when
/// no such keyword exists. This is the comment-/string-/token-aware replacement for
/// the previous `str::find("shader_body")` substring scan.
pub(crate) fn find_shader_body_keyword(src: &str) -> Option<usize> {
    const KW: &[u8] = b"shader_body";
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0;
    while i < n {
        match b[i] {
            // line comment
            b'/' if i + 1 < n && b[i + 1] == b'/' => {
                i += 2;
                while i < n && b[i] != b'\n' {
                    i += 1;
                }
            }
            // block comment
            b'/' if i + 1 < n && b[i + 1] == b'*' => {
                i += 2;
                while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(n);
            }
            // string / char literal
            b'"' | b'\'' => {
                i = skip_string_literal(b, i);
            }
            // keyword match in code
            c if c == KW[0] && i + KW.len() <= n && &b[i..i + KW.len()] == KW => {
                let before_ok = i == 0 || !is_ident_byte(b[i - 1]);
                let after = i + KW.len();
                let after_ok = after >= n || !is_ident_byte(b[after]);
                if before_ok && after_ok {
                    return Some(i);
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    None
}

/// Byte offset of the first occurrence of `target` that lies in real code (skipping
/// comments and string/char literals). Returns `None` if not found in code.
pub(crate) fn find_code_byte(src: &str, target: u8) -> Option<usize> {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0;
    while i < n {
        match b[i] {
            b'/' if i + 1 < n && b[i + 1] == b'/' => {
                i += 2;
                while i < n && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < n && b[i + 1] == b'*' => {
                i += 2;
                while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(n);
            }
            b'"' | b'\'' => {
                i = skip_string_literal(b, i);
            }
            c if c == target => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Given bytes positioned just AFTER an opening `{` (brace depth already 1), return
/// the byte length of the inner content up to (but not including) the `}` that
/// returns depth to 0. Comment- and string-aware, so braces inside `//…`, `/* … */`,
/// or string/char literals are not counted. Returns the full length when no matching
/// close is found (mirrors the previous best-effort behavior).
pub(crate) fn scan_to_matching_brace(src: &str) -> usize {
    let b = src.as_bytes();
    let n = b.len();
    let mut depth: i32 = 1;
    let mut i = 0;
    while i < n {
        match b[i] {
            b'/' if i + 1 < n && b[i + 1] == b'/' => {
                i += 2;
                while i < n && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < n && b[i + 1] == b'*' => {
                i += 2;
                while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(n);
            }
            b'"' | b'\'' => {
                i = skip_string_literal(b, i);
            }
            b'{' => {
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    n
}

/// Advance past a string/char literal that begins at `start` (the opening quote),
/// honoring backslash escapes. Returns the index just after the closing quote (or
/// `n` if the literal is unterminated). GLSL/HLSL shader bodies contain no string
/// literals, so on the real corpus this never fires; it is defensive so a stray
/// quote can never make a scanner miscount braces.
fn skip_string_literal(b: &[u8], start: usize) -> usize {
    let n = b.len();
    let quote = b[start];
    let mut i = start + 1;
    while i < n {
        if b[i] == b'\\' && i + 1 < n {
            i += 2;
            continue;
        }
        if b[i] == quote {
            return i + 1;
        }
        i += 1;
    }
    n
}

/// In a combined HLSL body (pre-shader_body globals + inner body):
///   1. Extracts HLSL `sampler X;` declarations (not valid in function scope) and
///      converts them to `uniform sampler2D X;` preamble lines that hlsl2glslfork can
///      handle at file scope.  glslopt then inlines the resulting split function.
///   2. Deduplicates variable declarations: turns `type name = expr;` into
///      `name = expr;` when `name` was already declared on a previous line, to
///      prevent hlsl2glslfork from rejecting the body with "redefinition" errors.
///
/// Returns `(cleaned_body, extra_hlsl_prefix)` where `extra_hlsl_prefix` is a
/// `uniform sampler2D X;\n…` string (empty if no custom samplers were found).
/// The caller should use `milk_convert_shader_ex` when the prefix is non-empty.
#[cfg(feature = "milk-native-converter")]
fn dedup_hlsl_declarations(body: &str) -> (String, String) {
    use std::collections::HashSet;

    const TYPES: &[&str] = &[
        "float4", "float3", "float2", "float", "int4", "int3", "int2", "int", "bool4", "bool3",
        "bool2", "bool", "half4", "half3", "half2", "half", "uint4", "uint3", "uint2", "uint",
        "double",
    ];

    // Pass 1 — collect every variable name that appears in a typed declaration line.
    // We collect ALL names from comma-separated lists so the second pass can strip
    // later redeclarations of any of them (e.g. `float2 dz,uv1,uv2;` → all three added).
    let mut declared: HashSet<String> = HashSet::new();
    for raw_line in body.lines() {
        let line = raw_line.trim_start();
        let line = line
            .strip_prefix("static ")
            .map(str::trim_start)
            .unwrap_or(line);
        for ty in TYPES {
            let after_ty = match line.strip_prefix(ty) {
                Some(rest) if rest.starts_with(|c: char| c.is_whitespace()) => rest.trim_start(),
                _ => continue,
            };
            let names_part = after_ty.split(['=', ';']).next().unwrap_or(after_ty);
            for name in names_part.split(',') {
                // strip array brackets e.g. `samples[5]` → `samples`
                let raw_name = name.trim();
                let n = raw_name.split('[').next().unwrap_or(raw_name).trim();
                if !n.is_empty() && n.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    declared.insert(n.to_string());
                }
            }
            break;
        }
    }

    // Pass 2 — process lines:
    //   • HLSL `sampler X;` lines: strip from body, collect X as a custom sampler.
    //   • Typed declaration lines: add ALL declared names to `seen` on first encounter;
    //     on second encounter with initializer (redeclaration), strip the type prefix.
    let mut out = String::with_capacity(body.len());
    let mut seen: HashSet<String> = HashSet::new();
    let mut custom_samplers: Vec<String> = Vec::new();

    for raw_line in body.lines() {
        let line = raw_line.trim_start();

        // Detect HLSL-only `sampler X;` lines (not sampler2D / sampler3D / samplerCube).
        {
            let after_samp = line.strip_prefix("sampler ").unwrap_or("");
            if !after_samp.is_empty()
                && !after_samp.starts_with("2D")
                && !after_samp.starts_with("3D")
                && !after_samp.starts_with("Cube")
            {
                // Extract the sampler name (everything before `;` or whitespace).
                let name = after_samp
                    .split([';', ' ', '\t'])
                    .next()
                    .unwrap_or("")
                    .trim();
                // Non-standard samplers (user textures) are added to the alias list so
                // their references in the body are replaced with a standard sampler.
                // Standard samplers are already declared in MILK_HLSL_PREFIX — just drop.
                if !name.is_empty() && !MILK_STANDARD_SAMPLERS.contains(&name) {
                    custom_samplers.push(name.to_string());
                }
                continue; // drop the `sampler X;` line from the body in both cases
            }
        }

        let after_static = line
            .strip_prefix("static ")
            .map(str::trim_start)
            .unwrap_or(line);
        let mut rewritten = false;
        for ty in TYPES {
            let after_ty = match after_static.strip_prefix(ty) {
                Some(rest) if rest.starts_with(|c: char| c.is_whitespace()) => rest.trim_start(),
                _ => continue,
            };
            let name_end = after_ty
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(after_ty.len());
            let name = &after_ty[..name_end];
            let rest = &after_ty[name_end..];
            let has_init = rest.contains('=');
            if declared.contains(name) {
                if seen.contains(name) && has_init {
                    // Redeclaration with init — strip the type prefix (turn into assignment).
                    let indent: String =
                        raw_line.chars().take_while(|c| c.is_whitespace()).collect();
                    out.push_str(&indent);
                    out.push_str(name);
                    out.push_str(rest);
                    out.push('\n');
                    rewritten = true;
                } else if seen.contains(name) && !has_init {
                    // Redundant bare redeclaration (no init) — drop the line entirely.
                    rewritten = true;
                } else if !seen.contains(name) {
                    // First declaration — mark ALL names from a comma-separated list as seen.
                    let names_part = after_ty.split(['=', ';']).next().unwrap_or(after_ty);
                    for nm in names_part.split(',') {
                        let raw_nm = nm.trim();
                        let n = raw_nm.split('[').next().unwrap_or(raw_nm).trim();
                        if !n.is_empty() {
                            seen.insert(n.to_string());
                        }
                    }
                }
            }
            break;
        }
        if !rewritten {
            out.push_str(raw_line);
            out.push('\n');
        }
    }

    // Alias custom sampler references in the body so hlsl2glslfork can resolve the
    // `tex2D(X, …)` call using a declared sampler. This gives a visually degraded
    // result (noise fallback instead of the user texture) for presets that use
    // user-loaded textures, but avoids sampling empty feedback forever.
    // Mirroring the pure-Rust GLSL path's `strip_user_texture_uniforms` logic.
    let body_out = alias_custom_sampler_refs(out, &custom_samplers);

    (body_out, String::new())
}

/// Rewrite combined `texture(name, ...)` calls in a GLSL body to our separated
/// `texture(sampler2D(name, name_samp), ...)` form, for all MilkDrop samplers.
/// Reuses the same rewriter as butterchurn_to_naga (sampler_map values are unused
/// by the rewrite — only the names matter).
/// Remove user custom-texture `uniform sampler2D/3D <name>;` declarations from a
/// Butterchurn GLSL body and alias their references to a neutral declared sampler.
/// These name user images we don't load; once inlined into main() the `uniform`
/// qualifier is illegal → naga `NotImplemented("variable qualifier")`. Aliasing keeps
/// the preset rendering (without the image — a fidelity gap, not a crash).
fn strip_user_texture_uniforms(body: &str) -> String {
    let mut names_2d: Vec<String> = Vec::new();
    let mut names_3d: Vec<String> = Vec::new();
    let mut kept: Vec<&str> = Vec::new();
    let mut found = false;
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with("uniform") && t.contains("sampler") {
            let decl = t.trim_end_matches(';').trim();
            if let Some(name) = decl.rsplit(char::is_whitespace).next() {
                if !name.is_empty() {
                    found = true;
                    if decl.contains("sampler3D") || decl.contains("samplerCube") {
                        names_3d.push(name.to_string());
                    } else {
                        names_2d.push(name.to_string());
                    }
                }
            }
            continue; // drop the declaration line
        }
        kept.push(line);
    }
    if !found {
        return body.to_string();
    }
    let mut out = kept.join("\n");
    for n in &names_2d {
        out = replace_word(&out, n, "sampler_noise_lq");
    }
    for n in &names_3d {
        out = replace_word(&out, n, "sampler_noisevol_lq");
    }
    // Custom textures also expose a `texsize_<base>` size const we don't declare;
    // alias those to a declared texsize too (companion to the sampler alias above).
    for n in names_2d.iter().chain(names_3d.iter()) {
        if let Some(base) = n.strip_prefix("sampler_") {
            out = replace_word(&out, &format!("texsize_{base}"), "texsize_noise_lq");
        }
    }
    out
}

/// Rewrite a single-component access directly after an index (`<expr>].x`) into
/// `<expr>][N]`. Butterchurn's transpiler builds matrices component-wise via
/// `m[uint(0)].x = q20;`, but naga can't assign to `.x` of an indexed matrix column
/// ("Can't lookup field on this type 'x'"). `m[i][0]` is an equivalent place
/// expression naga accepts (and an equivalent read for vectors). Only a lone
/// `.x/.y/.z/.w` immediately after `]` is converted; multi-component swizzles (`].xy`)
/// and non-indexed swizzles (`v.x`, `).rgb`) are left untouched.
fn fix_indexed_component_access(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b']' {
            let mut j = i + 1;
            while j < b.len() && (b[j] == b' ' || b[j] == b'\t') {
                j += 1;
            }
            if j + 1 < b.len() && b[j] == b'.' {
                let idx = match b[j + 1] {
                    b'x' => Some(0),
                    b'y' => Some(1),
                    b'z' => Some(2),
                    b'w' => Some(3),
                    _ => None,
                };
                if let Some(idx) = idx {
                    let after = j + 2;
                    let next_ok =
                        after >= b.len() || !(b[after].is_ascii_alphanumeric() || b[after] == b'_');
                    if next_ok {
                        out.push(']');
                        out.push_str(&src[i + 1..j]); // preserve any whitespace before `.`
                        out.push_str(&format!("[{idx}]"));
                        i = after;
                        continue;
                    }
                }
            }
        }
        let ch = src[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn rewrite_glsl_texture_calls(body: &str) -> String {
    // Drop user custom-texture `uniform sampler*;` declarations (named user images we
    // don't load) and alias them to a neutral base sampler; inlined into main() the
    // `uniform` qualifier is illegal → naga NotImplemented("variable qualifier").
    let mut normalized = strip_user_texture_uniforms(body);
    // `m[i].x = …` → `m[i][0] = …` (naga can't assign to a swizzle of an indexed
    // matrix column; this is Butterchurn's component-wise matrix-build idiom).
    normalized = fix_indexed_component_access(&normalized);
    // Alias undeclared sampling-mode variants of the noise/blur textures to their
    // declared base sampler. MilkDrop names samplers `sampler_<mode>_<tex>` with mode
    // in {fw,fc,pw,pc} (filter/point × wrap/clamp); we declare those variants only for
    // `main`, while noise/noisevol/blur each have a single base sampler. So
    // `sampler_pw_noise_lq` etc. would be UnknownVariable — strip the mode infix for
    // those textures (the wrap/filter mode is a no-op for our read-only lookups).
    normalized = normalize_milkdrop_sampler_variants(&normalized);
    // Butterchurn-exported bodies write `texture (sampler_main, …)` with a space
    // between `texture` and `(` (and similarly for other GLSL builtins). Normalize
    // those spaced calls to `name(` form so rewrite_texture_calls' `texture(name,`
    // patterns match. Do the same for the GLSL builtins used in these bodies so the
    // un-spaced output is consistent (naga accepts both, but the texture rewrite
    // requires the un-spaced `texture(` form).
    let body = normalized
        .replace("texture (", "texture(")
        .replace("mix (", "mix(")
        .replace("clamp (", "clamp(")
        .replace("cos (", "cos(")
        .replace("sin (", "sin(")
        .replace("sqrt (", "sqrt(")
        .replace("dot (", "dot(");

    let mut sampler_map: HashMap<String, u32> = HashMap::new();
    for (i, name) in MILKDROP_SAMPLERS.iter().enumerate() {
        sampler_map.insert((*name).to_string(), (i * 2) as u32);
    }
    let rewritten = rewrite_texture_calls(&body, &sampler_map);
    // 3D noise-volume textures need sampler3D, not the sampler2D that
    // rewrite_texture_calls emits for everything (sampler2D over a texture3D → naga
    // "Unknown function 'sampler2D'"). GLSL-path only — keeps raw .milk byte-identical.
    let rewritten = rewritten.replace("sampler2D(sampler_noisevol", "sampler3D(sampler_noisevol");
    // A 2D texture lookup needs a vec2 coordinate. HLSL truncates a wider coord
    // implicitly, and presets sample with a vec3-typed expression (e.g.
    // `texture(sampler2D(sampler_main,…), ret*0.1 + vec2(…))` where ret is vec3) →
    // naga reports no matching `texture(sampler2D, vec3)` overload as
    // "Unknown function 'texture'". Truncate every sampler2D coord to `.xy`
    // (a no-op for already-2D coords, so no behaviour change for valid presets).
    truncate_texture2d_coords(&rewritten)
}

/// Wrap the coordinate argument of every `texture(sampler2D(…), COORD)` call in
/// `(COORD).xy` so a vec3/vec4 coordinate is truncated to the vec2 a 2D sampler needs.
fn truncate_texture2d_coords(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let needle: Vec<char> = "texture(sampler2D(".chars().collect();
    let nl = needle.len();
    let mut out = String::with_capacity(n + 64);
    let mut i = 0;
    while i < n {
        let matches_here = i + nl <= n
            && b[i..i + nl] == needle[..]
            && (i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_'));
        if matches_here {
            // close of the inner `sampler2D(` group
            let s2d_open = i + nl - 1; // index of the '(' in "sampler2D("
            let mut depth = 0i32;
            let mut k = s2d_open;
            while k < n {
                match b[k] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            // expect `, COORD` after the sampler2D close
            let mut m = k + 1;
            while m < n && b[m].is_whitespace() {
                m += 1;
            }
            if m < n && b[m] == ',' {
                let coord_start = m + 1;
                // coord ends at a depth-0 comma (lod/bias) or the texture close `)`
                let mut d = 0i32;
                let mut p = coord_start;
                let mut coord_end = None;
                while p < n {
                    match b[p] {
                        '(' | '[' => d += 1,
                        ')' | ']' => {
                            if d == 0 {
                                coord_end = Some(p);
                                break;
                            }
                            d -= 1;
                        }
                        ',' if d == 0 => {
                            coord_end = Some(p);
                            break;
                        }
                        _ => {}
                    }
                    p += 1;
                }
                if let Some(ce) = coord_end {
                    out.extend(b[i..coord_start].iter());
                    let coord: String = b[coord_start..ce].iter().collect();
                    out.push('(');
                    out.push_str(coord.trim());
                    out.push_str(").xy");
                    i = ce; // continue from the `)` or `,`
                    continue;
                }
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Convert a Butterchurn converted-JSON GLSL COMP shader body to a complete
/// naga-compatible GLSL 450 fragment shader (same template as hlsl_milk_body_to_naga,
/// but the body is already GLSL — no HLSL conversion).
/// Hoist top-level function definitions out of a GLSL body so they sit at file scope
/// (before main()). The Butterchurn/glsl-optimizer converter prepends helper functions
/// (matrix_row0, m_scalar_swizzleN0, ...) before the shader body; inlined into main()
/// those nested defs are illegal → naga InvalidToken(LeftBrace). Returns
/// (functions, remaining_body). A def is a top-level `<sig>(...) { ... }` that is not a
/// control block. No-op when there are no top-level function defs.
fn hoist_function_defs(body: &str) -> (String, String) {
    let b: Vec<char> = body.chars().collect();
    let n = b.len();
    let mut funcs = String::new();
    let mut rest = String::new();
    let mut seg = 0usize;
    let mut i = 0usize;
    while i < n {
        match b[i] {
            ';' => {
                let s: String = b[seg..=i].iter().collect();
                rest.push_str(&s);
                rest.push('\n');
                seg = i + 1;
                i += 1;
            }
            '{' => {
                let header: String = b[seg..i].iter().collect();
                let h = header.trim_start();
                let is_fn = header.contains('(')
                    && !h.starts_with("if")
                    && !h.starts_with("for")
                    && !h.starts_with("while")
                    && !h.starts_with("else")
                    && !h.starts_with("do")
                    && !h.starts_with("switch");
                let mut d = 0i32;
                let mut j = i;
                while j < n {
                    match b[j] {
                        '{' => d += 1,
                        '}' => {
                            d -= 1;
                            if d == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                let end = j.min(n - 1);
                let block: String = b[seg..=end].iter().collect();
                if is_fn {
                    funcs.push_str(&block);
                    funcs.push('\n');
                } else {
                    rest.push_str(&block);
                    rest.push('\n');
                }
                seg = end + 1;
                i = end + 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    if seg < n {
        let tail: String = b[seg..n].iter().collect();
        if !tail.trim().is_empty() {
            rest.push_str(&tail);
        }
    }
    (funcs, rest)
}

/// If `text` (an `&&`/`||` operand) is a vector-bool expression, return its width.
/// The Butterchurn converter wraps such operands as `bvecN(...)`, possibly behind
/// extra parens: `(bvec3(x) && ...)`. Scalar operands (comparisons, `bool(..)`)
/// return None — they keep `&&`/`||`, which naga accepts for scalar bool.
fn leading_bvec_n(text: &str) -> Option<u8> {
    let t = text.trim_start().trim_start_matches('(').trim_start();
    let b = t.as_bytes();
    if b.len() >= 5 && &t[..4] == "bvec" {
        match b[4] {
            b'2' => Some(2),
            b'3' => Some(3),
            b'4' => Some(4),
            _ => None,
        }
    } else {
        None
    }
}

/// Read the balanced expression to the LEFT of `op_at` (a `&&`/`||`): a trailing
/// `(...)` group, with any leading cast identifier (`bvecN`, `m_andN`, ...) folded
/// in. Returns the operand's start index, or None if there is no `)`-group there.
fn left_operand_start(s: &[char], op_at: usize) -> Option<usize> {
    let mut k = op_at as isize - 1;
    while k >= 0 && s[k as usize].is_whitespace() {
        k -= 1;
    }
    if k < 0 || s[k as usize] != ')' {
        return None;
    }
    let mut depth = 0i32;
    while k >= 0 {
        match s[k as usize] {
            ')' => depth += 1,
            '(' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        k -= 1;
    }
    if k < 0 {
        return None;
    }
    // fold a leading cast identifier (e.g. `bvec3` in `bvec3 ( ... )`) into the operand
    let mut j = k - 1;
    while j >= 0 && s[j as usize].is_whitespace() {
        j -= 1;
    }
    let mut id_start = k as usize;
    while j >= 0 && (s[j as usize].is_alphanumeric() || s[j as usize] == '_') {
        id_start = j as usize;
        j -= 1;
    }
    Some(id_start)
}

/// Read the balanced expression to the RIGHT of an `&&`/`||` whose operator chars
/// start at `r0` (= op_at + 2): an optional leading cast identifier plus a `(...)`
/// group. Returns the index just past the operand.
fn right_operand_end(s: &[char], r0: usize) -> Option<usize> {
    let n = s.len();
    let mut i = r0;
    while i < n && s[i].is_whitespace() {
        i += 1;
    }
    // optional leading identifier
    while i < n && (s[i].is_alphanumeric() || s[i] == '_') {
        i += 1;
    }
    while i < n && s[i].is_whitespace() {
        i += 1;
    }
    if i >= n || s[i] != '(' {
        return None;
    }
    let mut depth = 0i32;
    while i < n {
        match s[i] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Rewrite vector-bool `&&`/`||` (which naga rejects: `LogicalAnd(vecN<bool>, _)`)
/// into component-wise helper calls `m_andN`/`m_orN`. The Butterchurn converter's
/// un-optimized GLSL emits `bvecN(X) && bvecN(Y)`; naga only allows `&&`/`||` on
/// scalar bool. Scalar logical ops (comparisons, `bool(..) && bool(..)`) are left
/// untouched — their operands do not begin with a `bvecN` cast. Runs to a fixpoint
/// so chains and nesting (`a && b && c`, `a && (b && c)`) fully resolve.
fn rewrite_vector_logical(src: &str) -> String {
    let mut s: Vec<char> = src.chars().collect();
    loop {
        let n = s.len();
        let mut hit: Option<(usize, usize, usize, u8, char)> = None;
        let mut i = 0;
        while i + 1 < n {
            let c = s[i];
            if (c == '&' && s[i + 1] == '&') || (c == '|' && s[i + 1] == '|') {
                if let (Some(l_start), Some(r_end)) =
                    (left_operand_start(&s, i), right_operand_end(&s, i + 2))
                {
                    let l: String = s[l_start..i].iter().collect();
                    let r: String = s[i + 2..r_end].iter().collect();
                    let nvec = leading_bvec_n(&l).or_else(|| leading_bvec_n(&r));
                    if let Some(nv) = nvec {
                        hit = Some((l_start, i, r_end, nv, c));
                        break;
                    }
                }
            }
            i += 1;
        }
        match hit {
            None => break,
            Some((l_start, op_at, r_end, nv, opch)) => {
                let l: String = s[l_start..op_at].iter().collect();
                let r: String = s[op_at + 2..r_end].iter().collect();
                let fname = if opch == '&' {
                    format!("m_and{nv}")
                } else {
                    format!("m_or{nv}")
                };
                let repl: Vec<char> = format!("{fname}({}, {})", l.trim(), r.trim())
                    .chars()
                    .collect();
                let mut out: Vec<char> = Vec::with_capacity(n);
                out.extend_from_slice(&s[..l_start]);
                out.extend_from_slice(&repl);
                out.extend_from_slice(&s[r_end..]);
                s = out;
            }
        }
    }
    s.into_iter().collect()
}

/// Rewrite `a / b` -> `a * safeRecip(b)` for DYNAMIC denominators (identifiers,
/// member/index/call chains, or parenthesised expressions); numeric-literal
/// denominators, `//`/`/*` comments and `/=` are left untouched. Only the DENOMINATOR
/// (right operand, a single primary) is captured — `/` and `*` share precedence and
/// left-associativity, so `a / b` == `a * (1/b)` needs no reparenthesising. Emulates
/// DX9/WebGL fast-math (a /0 term vanishes) under wgpu strict-IEEE, where an unguarded
/// /0 over an unset q-var yields inf*0=NaN that blacks the whole pixel.
fn guard_divides(src: &str) -> String {
    let b = src.as_bytes();
    let n = b.len();
    let mut out = String::with_capacity(n + 64);
    let mut i = 0;
    while i < n {
        let c = b[i];
        if c == b'/' && i + 1 < n && b[i + 1] == b'/' {
            // line comment
            while i < n && b[i] != b'\n' {
                out.push(b[i] as char);
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < n && b[i + 1] == b'*' {
            // block comment
            out.push_str("/*");
            i += 2;
            while i < n {
                if b[i] == b'*' && i + 1 < n && b[i + 1] == b'/' {
                    out.push_str("*/");
                    i += 2;
                    break;
                }
                out.push(b[i] as char);
                i += 1;
            }
            continue;
        }
        if c == b'/' && !(i + 1 < n && b[i + 1] == b'=') {
            // division (not /=)
            let mut j = i + 1;
            while j < n && b[j].is_ascii_whitespace() {
                j += 1;
            }
            let denom_start = j;
            while j < n && matches!(b[j], b'-' | b'+' | b'!' | b'~') {
                // leading unary ops
                j += 1;
                while j < n && b[j].is_ascii_whitespace() {
                    j += 1;
                }
            }
            let (end, is_literal) = parse_primary_fwd(b, j);
            if end > j && !is_literal {
                out.push_str("* safeRecip(");
                out.push_str(&src[denom_start..end]);
                out.push(')');
                i = end;
                continue;
            }
            out.push('/'); // literal or unparsable denominator: leave as-is
            i += 1;
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Parse one GLSL "primary" at `j` (after leading unary ops). Returns (end, is_numeric_literal).
fn parse_primary_fwd(b: &[u8], mut j: usize) -> (usize, bool) {
    let n = b.len();
    if j >= n {
        return (j, false);
    }
    let c = b[j];
    if c == b'(' {
        j = consume_balanced(b, j, b'(', b')');
        return (consume_suffixes_fwd(b, j), false);
    }
    if c.is_ascii_digit() || (c == b'.' && j + 1 < n && b[j + 1].is_ascii_digit()) {
        while j < n {
            let d = b[j];
            let prev_e = j > 0 && (b[j - 1] == b'e' || b[j - 1] == b'E');
            if d.is_ascii_digit()
                || matches!(
                    d,
                    b'.' | b'e' | b'E' | b'f' | b'F' | b'u' | b'U' | b'l' | b'L'
                )
                || ((d == b'+' || d == b'-') && prev_e)
            {
                j += 1;
            } else {
                break;
            }
        }
        return (j, true);
    }
    if c.is_ascii_alphabetic() || c == b'_' {
        let id_start = j;
        while j < n && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
            j += 1;
        }
        let ident = std::str::from_utf8(&b[id_start..j]).unwrap_or("");
        // Is the identifier immediately a call `(`? A GLSL vector/matrix *constructor*
        // call (`vec2(0.0,-1.0)`, `mat3(...)`) with all-constant args is const-foldable
        // by naga. guard_divides must NOT wrap such a denominator in `safeRecip(…)` —
        // safeRecip's `1.0/0.0` branch then const-folds to an infinite literal on a
        // zero component → naga "Function 'main' is invalid". Reporting it as a literal
        // leaves the original `/ctor(…)` division (uniform/const numerators don't fold).
        let mut k = j;
        while k < n && b[k].is_ascii_whitespace() {
            k += 1;
        }
        let is_ctor_call = k < n && b[k] == b'(' && is_glsl_constructor(ident);
        return (consume_suffixes_fwd(b, j), is_ctor_call);
    }
    (j, false)
}

/// GLSL builtin vector/matrix type-constructor names (used by guard_divides to skip
/// wrapping constant constructor denominators in safeRecip — see parse_primary_fwd).
fn is_glsl_constructor(name: &str) -> bool {
    matches!(
        name,
        "vec2"
            | "vec3"
            | "vec4"
            | "ivec2"
            | "ivec3"
            | "ivec4"
            | "uvec2"
            | "uvec3"
            | "uvec4"
            | "bvec2"
            | "bvec3"
            | "bvec4"
            | "mat2"
            | "mat3"
            | "mat4"
            | "mat2x2"
            | "mat2x3"
            | "mat2x4"
            | "mat3x2"
            | "mat3x3"
            | "mat3x4"
            | "mat4x2"
            | "mat4x3"
            | "mat4x4"
    )
}

fn consume_balanced(b: &[u8], mut j: usize, open: u8, close: u8) -> usize {
    let n = b.len();
    let mut depth = 0i32;
    while j < n {
        let c = b[j];
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            j += 1;
            if depth == 0 {
                return j;
            }
            continue;
        }
        j += 1;
    }
    j
}

/// Consume `(...)` call, `[...]` index and `.member` suffixes (whitespace allowed before each).
fn consume_suffixes_fwd(b: &[u8], mut j: usize) -> usize {
    let n = b.len();
    loop {
        let mut k = j;
        while k < n && b[k].is_ascii_whitespace() {
            k += 1;
        }
        if k >= n {
            break;
        }
        match b[k] {
            b'(' => j = consume_balanced(b, k, b'(', b')'),
            b'[' => j = consume_balanced(b, k, b'[', b']'),
            b'.' if k + 1 < n && (b[k + 1].is_ascii_alphabetic() || b[k + 1] == b'_') => {
                j = k + 1;
                while j < n && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                    j += 1;
                }
            }
            _ => break,
        }
    }
    j
}

pub fn glsl_milk_body_to_naga(body: &str) -> String {
    let inner = strip_shader_body_wrapper(body);
    let (hoisted_fns, converted_body) = hoist_function_defs(&guard_divides(
        &rewrite_vector_logical(&rewrite_glsl_texture_calls(&inner)),
    ));
    let gamma_postlude = comp_gamma_postlude(&converted_body);

    let io_decls = "\
layout(location = 0) in  vec2 vUv;
layout(location = 1) in  vec4 vColor;
layout(location = 0) out vec4 fragColor;";
    let preamble = milk_fs_preamble(io_decls);

    format!(
        r#"{preamble}
// Per-pixel values promoted to file-scope globals so hoisted helper functions
// (the Butterchurn converter's `main_shader_sentinel` + helpers) can reference
// them. `main()` ASSIGNS (not declares) them before the body runs — backward
// compatible with the inlined library path, which reads them in the same scope.
vec2  uv_orig;
float rad;
float ang;
vec3  hue_shader;
{hoisted_fns}
void main() {{
    vec3 ret = vec3(0.0);
    vec2 uv = vUv;
    uv_orig = vUv;
    uv.y = 1.0 - uv.y;
    uv_orig.y = 1.0 - uv_orig.y;
    rad = length(uv - 0.5);
    ang = atan(uv.x - 0.5, uv.y - 0.5);
    // hue_shader base (see hlsl_milk_body_to_naga for the derivation).
    float _ht = time * 30.0;
    vec4 _hr = rand_start;
    vec3 _hc0 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+ 0.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+ 0.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+ 0.0+_hr.z));
    _hc0 /= max(_hc0.x, max(_hc0.y, _hc0.z)); _hc0 = 0.5 + 0.5*_hc0;
    vec3 _hc1 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+21.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+13.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+ 9.0+_hr.z));
    _hc1 /= max(_hc1.x, max(_hc1.y, _hc1.z)); _hc1 = 0.5 + 0.5*_hc1;
    vec3 _hc2 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+42.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+26.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+18.0+_hr.z));
    _hc2 /= max(_hc2.x, max(_hc2.y, _hc2.z)); _hc2 = 0.5 + 0.5*_hc2;
    vec3 _hc3 = vec3(0.6+0.3*sin(_ht*0.0143+3.0+63.0+_hr.w),
                     0.6+0.3*sin(_ht*0.0107+1.0+39.0+_hr.y),
                     0.6+0.3*sin(_ht*0.0129+6.0+27.0+_hr.z));
    _hc3 /= max(_hc3.x, max(_hc3.y, _hc3.z)); _hc3 = 0.5 + 0.5*_hc3;
    float _hx = vUv.x;
    float _hy = vUv.y;
    hue_shader = _hc0*_hx*_hy + _hc1*(1.0-_hx)*_hy
                    + _hc2*_hx*(1.0-_hy) + _hc3*(1.0-_hx)*(1.0-_hy);

{converted_body}

{gamma_postlude}
    fragColor = vec4(ret, 1.0);
}}
"#
    )
}

/// Convert a Butterchurn converted-JSON GLSL WARP shader body to a complete
/// naga-compatible GLSL 450 fragment shader driven by the warped-mesh VS.
/// Same varyings/template as hlsl_milk_warp_body_to_naga (vUv@0, vWarpUv@1,
/// vDecay@2) and the SAME no-decay-multiply final line (custom warp self-decays;
/// fragColor = vec4(ret, 1.0) with NO vDecay multiply). Body is already GLSL.
pub fn glsl_milk_warp_body_to_naga(body: &str) -> String {
    let inner = strip_shader_body_wrapper(body);
    let (hoisted_fns, converted_body) = hoist_function_defs(&guard_divides(
        &rewrite_vector_logical(&rewrite_glsl_texture_calls(&inner)),
    ));

    let io_decls = "\
layout(location = 0) in  vec2 vUv;
layout(location = 1) in  vec2 vWarpUv;
layout(location = 2) in  vec4 vDecay;
layout(location = 0) out vec4 fragColor;";
    let preamble = milk_fs_preamble(io_decls);

    format!(
        r#"{preamble}
// Per-pixel values promoted to file-scope globals so hoisted helper functions
// (the Butterchurn converter's `main_shader_sentinel` + helpers) can reference
// them. `main()` ASSIGNS (not declares) them before the body runs.
vec2  uv_orig;
float rad;
float ang;
{hoisted_fns}
void main() {{
    vec3 ret = vec3(0.0);
    // uv = warped sample coord from the mesh (DirectX-UV, v=0 top); NO y-flip.
    vec2 uv = vWarpUv;
    // uv_orig = screen position for rad/ang (butterchurn warp.js).
    uv_orig = vUv;
    rad = length(uv_orig - 0.5);
    ang = atan(uv_orig.x - 0.5, uv_orig.y - 0.5);

{converted_body}

    // Custom warp self-decays — NO extra vDecay multiply (matches the HLSL custom-warp path).
    fragColor = vec4(ret, 1.0);
}}
"#
    )
}

/// Strip HLSL sampler declarations that have no GLSL equivalent.
///
/// Handles both:
///   - Multi-line block: `sampler X = sampler_state { ... };`
///   - Single-line: `sampler X;` / `sampler2D X;` / `uniform sampler2D X;`
///
/// These appear in `.milk` files' `before` globals and must be removed before
/// HLSL→GLSL conversion, otherwise naga sees `sampler_state`/`sampler2D` as
/// unknown variables inside `void main()`.
fn strip_hlsl_sampler_blocks(src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim_start();
        // Multi-line `sampler X = sampler_state {`
        if t.starts_with("sampler ") && t.contains("sampler_state") {
            // Skip until line with closing `};`
            while i < lines.len() {
                let cl = lines[i].trim();
                i += 1;
                if cl.ends_with("};") || cl == "};" || cl.ends_with("};") {
                    break;
                }
            }
            continue;
        }
        // Single-line `sampler X;` (bare HLSL SM3 sampler declare)
        if t.starts_with("sampler ")
            && !t.starts_with("sampler2D")
            && !t.starts_with("sampler3D")
            && !t.starts_with("samplerCube")
            && (t.contains(';') || t.ends_with(';'))
        {
            i += 1;
            continue;
        }
        // `sampler2D X;` or `uniform sampler2D X;` inside a function body
        // (GLSL doesn't allow sampler declarations inside functions)
        let stripped = t.trim_start_matches("uniform").trim_start();
        if (stripped.starts_with("sampler2D ")
            || stripped.starts_with("sampler3D ")
            || stripped.starts_with("samplerCube "))
            && stripped.contains(';')
        {
            i += 1;
            continue;
        }
        out.push_str(lines[i]);
        out.push('\n');
        i += 1;
    }
    out
}

/// Apply HLSL → GLSL body substitutions.
/// Does NOT produce a complete program — just converts the inner body text.
fn hlsl_to_glsl_body(src: &str) -> String {
    hlsl_to_glsl_body_ex(src, true)
}

/// Like `hlsl_to_glsl_body` but `drop_dead` controls the final unused-declaration
/// pass. Hoisting splits a body into separate function / declaration fragments whose
/// referents live in OTHER fragments; dropping "unused" decls there would delete
/// file-scope globals that hoisted functions need, so those fragments pass `false`.
fn hlsl_to_glsl_body_ex(src: &str, drop_dead: bool) -> String {
    // HLSL authors freely space calls (`lerp (a,b)`, `max (x,0)`, `tex2D (s,uv)`).
    // Collapse `ident (` → `ident(` up front so every downstream call rewrite
    // (lerp→mix, mul, tex2D→texture, pow/max int coercion) matches. GLSL treats
    // `f (x)` and `f(x)` identically, so this changes nothing semantically.
    let mut s = split_hlsl_multi_declarators(src);
    s = collapse_call_spaces(&s);
    s = normalize_milkdrop_sampler_variants(&s);
    s = strip_comments(&s);
    s = rename_reserved_sample_local(&s);

    // HLSL bool-as-number: cast a comparison used in arithmetic to float (naga rejects
    // `bool * float`). Same transform applied to the native pre-pass.
    s = wrap_bool_arith(&s);

    // Strip HLSL `static` storage qualifier (e.g. `static const float3 t = …`);
    // GLSL/naga has no such qualifier on locals → UnknownVariable("static").
    s = replace_word(&s, "static", "");

    // HLSL vector/matrix types → GLSL (float1 is the HLSL scalar)
    s = replace_word(&s, "float1", "float");
    s = replace_word(&s, "float2", "vec2");
    s = replace_word(&s, "float3", "vec3");
    s = replace_word(&s, "float4", "vec4");
    // DX11 double-precision types (hlsl2glslfork rejects these; treat as float)
    s = replace_word(&s, "double2", "vec2");
    s = replace_word(&s, "double3", "vec3");
    s = replace_word(&s, "double4", "vec4");
    s = replace_word(&s, "double", "float");
    s = replace_word(&s, "float2x2", "mat2");
    s = replace_word(&s, "float3x3", "mat3");
    s = replace_word(&s, "float4x4", "mat4");
    // Non-square matrix types (less common but valid HLSL)
    s = replace_word(&s, "float2x3", "mat2x3");
    s = replace_word(&s, "float2x4", "mat2x4");
    s = replace_word(&s, "float3x2", "mat3x2");
    s = replace_word(&s, "float3x4", "mat3x4");
    s = replace_word(&s, "float4x2", "mat4x2");
    s = replace_word(&s, "float4x3", "mat4x3");
    // HLSL allows float2x2(float4) (packs 4 components); GLSL/WGSL matrix
    // constructors need scalars/column-vectors, not a single vec4. Expand
    // mat2(<single-arg>) → mat2((e).x,(e).y,(e).z,(e).w). Multi-arg forms
    // (mat2(a,b,c,d) / mat2(v0,v1)) have a top-level comma and are left alone.
    s = expand_mat2_from_vec(&s);
    s = replace_word(&s, "int2", "ivec2");
    s = replace_word(&s, "int3", "ivec3");
    s = replace_word(&s, "int4", "ivec4");
    s = replace_word(&s, "bool2", "bvec2");
    s = replace_word(&s, "bool3", "bvec3");
    s = replace_word(&s, "bool4", "bvec4");
    // HLSL half-precision types → float (longest-first so `half` doesn't eat `half2`).
    s = replace_word(&s, "half2", "vec2");
    s = replace_word(&s, "half3", "vec3");
    s = replace_word(&s, "half4", "vec4");
    s = replace_word(&s, "half", "float");

    // HLSL math functions → GLSL
    s = s.replace("lerp(", "mix(");
    s = s.replace("frac(", "fract(");
    s = s.replace("atan2(", "atan(");
    s = s.replace("ddx(", "dFdx(");
    s = s.replace("ddy(", "dFdy(");
    // mul(mat, vec) → mat * vec — risky if nested, but covers the common case
    s = replace_mul(&s);

    // naga's GLSL frontend resolves builtin overloads strictly by argument type, with
    // no implicit int→float promotion for a bare integer-literal argument. HLSL presets
    // write `pow(x, 2)`, `max(y, 1)`, `mix(a, b, 1)` → "Unknown function 'pow'/'mix'".
    // Coerce any whole-integer-literal argument of these float-genType builtins to a
    // float (`1` → `1.0`).
    s = coerce_builtin_int_args(
        &s,
        &[
            "pow",
            "max",
            "min",
            "mix",
            "clamp",
            "smoothstep",
            "step",
            "mod",
        ],
    );

    // HLSL scalar swizzles (`f.xxx` replicates a float to vec3) are illegal in GLSL
    // → naga "Can't lookup field on this type 'xxx'". Rewrite replicated swizzles of
    // known scalar (`float`) locals to vector constructors (`f.xxx` → `vec3(f)`).
    s = fix_scalar_swizzles(&s);

    // tex2D / tex2d → texture() calls (sampler name → sampler2D(name, name_samp))
    s = rewrite_tex2d_calls(&s);
    // Truncate sampler2D coords to .xy (HLSL samples 2D textures with wider coords).
    s = truncate_texture2d_coords(&s);

    // HLSL implicit truncation: tex2D returns float4, but presets assign it to a
    // narrower local (`float3 c = tex2D(...)` → take .xyz). naga rejects vec3 = vec4.
    s = truncate_texture_decls(&s);
    // Same truncation for decls whose RHS is a vec4 built-in expression, e.g.
    // `float3 lay1 = uv.y*pow(...)*roam_cos;` (roam_cos is vec4). HLSL truncates.
    s = truncate_vec4_builtin_decls(&s);

    // Drop unreferenced local declarations. HLSL presets often declare dead
    // temporaries that rely on HLSL's implicit vector→scalar truncation (e.g.
    // `float corr = texsize.xy*texsize_noise_lq.zw;` — a vec2 into a float), which
    // GLSL/naga reject. Since preset expressions are pure, dropping an unused
    // local is semantically safe and avoids the type error.
    if drop_dead {
        s = drop_unused_decls(&s);
    }

    s
}

/// HLSL fixups applied to the body BEFORE the native converter (hlsl2glslfork) sees
/// it. hlsl2glslfork does not know the DX11 `double` types and aborts with
/// "'double3' : undeclared identifier" (cascading into spurious syntax errors on the
/// following tokens), forcing these presets onto the weaker pure-Rust fallback.
/// Rewriting `double*` → `float*` up front lets them go through the native path.
fn hlsl_pre_native_fixups(src: &str) -> String {
    let mut s = split_hlsl_multi_declarators(src);
    s = normalize_milkdrop_sampler_variants(&s);
    s = strip_comments(&s);
    s = replace_word(&s, "double2", "float2");
    s = replace_word(&s, "double3", "float3");
    s = replace_word(&s, "double4", "float4");
    s = replace_word(&s, "double", "float");
    // HLSL bool-as-number (`(a>b)*x`, `x-(a>b)`): hlsl2glslfork rejects a bool operand
    // of an arithmetic op, forcing the weaker fallback. Cast to float so these route
    // through the native converter (and compute correctly).
    s = wrap_bool_arith(&s);
    s
}

/// hlsl2glslfork silently loses all but the first initializer in declarations
/// such as `float4 lum1=0, lum2=0;`. Split each top-level declarator into its own
/// declaration before either converter sees it. Commas in constructors/calls and
/// semicolons in `for (...)` headers are deliberately ignored.
fn split_hlsl_multi_declarators(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 32);
    let mut start = 0usize;
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    for (index, byte) in bytes.iter().copied().enumerate() {
        match byte {
            b'(' => paren_depth += 1,
            b')' => paren_depth -= 1,
            b'[' => bracket_depth += 1,
            b']' => bracket_depth -= 1,
            b';' if paren_depth == 0 && bracket_depth == 0 => {
                let segment = &src[start..index];
                if let Some(split) = split_hlsl_multi_decl_segment(segment) {
                    out.push_str(&split);
                } else {
                    out.push_str(segment);
                }
                out.push(';');
                start = index + 1;
            }
            _ => {}
        }
    }
    out.push_str(&src[start..]);
    out
}

fn split_hlsl_multi_decl_segment(segment: &str) -> Option<String> {
    let core_start = segment
        .rmatch_indices(['{', '}'])
        .next()
        .map_or(0, |(index, ch)| index + ch.len());
    let prefix = &segment[..core_start];
    let core = &segment[core_start..];
    let trimmed = core.trim_start();
    let whitespace = &core[..core.len() - trimmed.len()];

    let mut rest = trimmed;
    let mut qualifiers = Vec::new();
    loop {
        let Some((word, tail)) = split_first_word(rest) else {
            return None;
        };
        if matches!(word, "static" | "const") {
            qualifiers.push(word);
            rest = tail.trim_start();
        } else {
            break;
        }
    }
    let (ty, declarators) = split_first_word(rest)?;
    if !is_hlsl_value_type(ty) {
        return None;
    }
    let declarators = declarators.trim_start();
    let pieces = split_top_level_commas(declarators);
    if pieces.len() < 2 {
        return None;
    }

    let indent = whitespace.rsplit('\n').next().unwrap_or(whitespace);
    let decl_prefix = if qualifiers.is_empty() {
        ty.to_string()
    } else {
        format!("{} {ty}", qualifiers.join(" "))
    };
    let mut result = String::with_capacity(segment.len() + pieces.len() * decl_prefix.len());
    result.push_str(prefix);
    result.push_str(whitespace);
    for (index, piece) in pieces.iter().enumerate() {
        if index > 0 {
            result.push_str(";\n");
            result.push_str(indent);
        }
        result.push_str(&decl_prefix);
        result.push(' ');
        result.push_str(piece.trim());
    }
    Some(result)
}

fn split_first_word(src: &str) -> Option<(&str, &str)> {
    let end = src.find(char::is_whitespace)?;
    Some((&src[..end], &src[end..]))
}

/// Does `s` contain a TOP-LEVEL relational/equality operator (`<` `>` `<=` `>=` `==`
/// `!=`), i.e. is it a comparison expression? Ignores `<<`, `>>`, and `=` assignment.
fn has_top_level_comparison(s: &str) -> bool {
    let b = s.as_bytes();
    let n = b.len();
    let mut depth = 0i32;
    let mut i = 0;
    while i < n {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'<' | b'>' if depth == 0 => {
                let nx = b.get(i + 1).copied();
                if nx != Some(b[i]) {
                    // not << or >>
                    return true;
                }
                i += 1;
            }
            b'=' | b'!' if depth == 0 => {
                if b.get(i + 1).copied() == Some(b'=') {
                    return true; // == or !=
                }
            }
            _ => {}
        }
        i += 1;
    }
    false
}

/// Wrap a parenthesized comparison used in an ARITHMETIC context in `float(...)`.
/// HLSL implicitly promotes bool→float in arithmetic (`(a>b)*x`, `x-(a>b)`,
/// `-(a<b)`), but hlsl2glslfork and naga reject a bool operand of `* / + -`. Only the
/// arithmetic-context comparisons are wrapped; boolean contexts (`if(...)`, `&&`, `||`,
/// `?:`) are left untouched.
fn wrap_bool_arith(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(n + 32);
    let mut i = 0;
    while i < n {
        if b[i] == '(' {
            let mut depth = 0i32;
            let mut k = i;
            while k < n {
                match b[k] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            if k < n {
                let inner: String = b[i + 1..k].iter().collect();
                if has_top_level_comparison(&inner) {
                    let prev = (0..i).rev().map(|j| b[j]).find(|c| !c.is_whitespace());
                    let next = (k + 1..n).map(|j| b[j]).find(|c| !c.is_whitespace());
                    let arith = |c: Option<char>| {
                        matches!(c, Some('*') | Some('/') | Some('+') | Some('-'))
                    };
                    let boolean = |c: Option<char>| {
                        matches!(c, Some('&') | Some('|') | Some('?') | Some(':'))
                    };
                    // The contiguous word immediately before `(` (no whitespace): if it is
                    // a function name, this `(…)` is its ARGUMENT list (not a groupable
                    // comparison) — leave it. `if`/`while`/`for` is a boolean condition —
                    // leave it. `return` (or no word) means a grouping paren we may wrap.
                    let mut ws = i;
                    while ws > 0 && (b[ws - 1].is_alphanumeric() || b[ws - 1] == '_') {
                        ws -= 1;
                    }
                    let word: String = b[ws..i].iter().collect();
                    let is_call = !word.is_empty()
                        && !matches!(
                            word.as_str(),
                            "return" | "if" | "while" | "for" | "else" | "do"
                        );
                    let is_cond = matches!(word.as_str(), "if" | "while" | "for");
                    if !is_call
                        && !is_cond
                        && (arith(prev) || arith(next))
                        && !boolean(prev)
                        && !boolean(next)
                    {
                        // avoid gluing `float(` onto a preceding keyword (`return`)
                        if out
                            .chars()
                            .last()
                            .map_or(false, |c| c.is_alphanumeric() || c == '_')
                        {
                            out.push(' ');
                        }
                        out.push_str("float(");
                        out.push_str(&wrap_bool_arith(&inner));
                        out.push(')');
                        i = k + 1;
                        continue;
                    }
                }
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Collapse insignificant whitespace between an identifier and a following `(`
/// (`lerp (a)` → `lerp(a)`) so call/constructor rewrites match. Whitespace between a
/// function name and its argument list is insignificant in GLSL, so this is safe.
fn collapse_call_spaces(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(n);
    let mut last = '\0';
    let mut i = 0;
    while i < n {
        if (b[i] == ' ' || b[i] == '\t') && (last.is_alphanumeric() || last == '_') {
            let mut k = i;
            while k < n && (b[k] == ' ' || b[k] == '\t') {
                k += 1;
            }
            if k < n && b[k] == '(' {
                i = k; // drop the spaces; next iteration pushes '('
                continue;
            }
        }
        out.push(b[i]);
        last = b[i];
        i += 1;
    }
    out
}

/// Strip `//` line comments and `/* … */` block comments.
fn strip_comments(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(n);
    let mut i = 0;
    while i < n {
        if b[i] == '/' && i + 1 < n && b[i + 1] == '/' {
            while i < n && b[i] != '\n' {
                i += 1;
            }
        } else if b[i] == '/' && i + 1 < n && b[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(b[i] == '*' && b[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(n);
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    out
}

/// GLSL/HLSL value-type keyword (used by split_hlsl_globals to recognise declarations).
fn is_hlsl_value_type(t: &str) -> bool {
    matches!(
        t,
        "float"
            | "float2"
            | "float3"
            | "float4"
            | "float2x2"
            | "float3x3"
            | "float4x4"
            | "float2x3"
            | "float2x4"
            | "float3x2"
            | "float3x4"
            | "float4x2"
            | "float4x3"
            | "int"
            | "int2"
            | "int3"
            | "int4"
            | "bool"
            | "bool2"
            | "bool3"
            | "bool4"
            | "half"
            | "half2"
            | "half3"
            | "half4"
            | "double"
            | "double2"
            | "double3"
            | "double4"
            | "vec2"
            | "vec3"
            | "vec4"
            | "mat2"
            | "mat3"
            | "mat4"
    )
}

/// Split a byte string by top-level commas (commas not inside `()[]{}`).
fn split_top_level_commas(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                out.push(s[start..i].to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].to_string());
    out
}

/// Find the first top-level `=` that is an assignment (not `==`/`<=`/`>=`/`!=`).
fn find_top_level_assign(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'=' if depth == 0 => {
                let prev = if i > 0 { b[i - 1] } else { b' ' };
                let next = if i + 1 < b.len() { b[i + 1] } else { b' ' };
                if next != b'=' && !matches!(prev, b'=' | b'<' | b'>' | b'!') {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split a `before`-block's global variable declarations into (file_decls, main_inits).
/// Declarations (without initializers) go to GLSL file scope so hoisted helper
/// functions can reference them; non-const initializers become assignments inside
/// main() (GLSL forbids non-constant global initializers). `static` is stripped.
/// Handles multi-variable declarations and per-variable initializers; statements that
/// are not type-prefixed declarations are passed through as main-body statements.
fn split_hlsl_globals(globals: &str) -> (String, String) {
    let mut decls = String::new();
    let mut inits = String::new();
    // Comments would corrupt statement splitting (`= x ;//note` swallows the next
    // declaration's type token, a commented-out `//float3 x = …` looks like a decl).
    let globals = strip_comments(globals);
    let globals = globals.as_str();
    let b = globals.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut depth = 0i32;
    let mut start = 0;
    while i < n {
        match b[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b';' if depth == 0 => {
                process_global_stmt(globals[start..i].trim(), &mut decls, &mut inits);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < n {
        process_global_stmt(globals[start..].trim(), &mut decls, &mut inits);
    }
    (decls, inits)
}

fn process_global_stmt(stmt: &str, decls: &mut String, inits: &mut String) {
    if stmt.is_empty() {
        return;
    }
    // Strip leading storage qualifiers (`static`/`const`, in any order) so the type
    // token is recognised. A `const float res = 255, res2 = 64;` otherwise types as
    // "const" → not a declaration → the names never reach file scope → helper
    // functions referencing them fail with naga UnknownVariable.
    let mut s = stmt.trim();
    while let Some(rest) = s
        .strip_prefix("static ")
        .or_else(|| s.strip_prefix("const "))
    {
        s = rest.trim_start();
    }
    let Some(sp) = s.find(char::is_whitespace) else {
        inits.push_str(s);
        inits.push_str(";\n");
        return;
    };
    let ty = &s[..sp];
    if !is_hlsl_value_type(ty) {
        // not a recognised declaration (a bare statement) → keep it in main
        inits.push_str(s);
        inits.push_str(";\n");
        return;
    }
    let rest = s[sp..].trim();
    let mut names: Vec<String> = Vec::new();
    for item in split_top_level_commas(rest) {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        if let Some(eq) = find_top_level_assign(item) {
            let name = item[..eq].trim();
            let expr = item[eq + 1..].trim();
            names.push(name.to_string());
            inits.push_str(&format!("{name} = {expr};\n"));
        } else {
            names.push(item.to_string());
        }
    }
    if !names.is_empty() {
        decls.push_str(&format!("{ty} {};\n", names.join(", ")));
    }
}

/// Apply sampler stripping + custom-sampler aliasing to one HLSL fragment, using a
/// custom-sampler list collected from the WHOLE shader (so a sampler declared in the
/// `before` block but used in `inner` is aliased consistently across fragments).
fn alias_with_customs(fragment: &str, customs: &[String]) -> String {
    let (stripped, _) = strip_and_alias_hlsl_samplers(fragment);
    alias_custom_sampler_refs(stripped, customs)
}

fn alias_custom_sampler_refs(mut src: String, customs: &[String]) -> String {
    let replacements: HashMap<String, String> = customs
        .iter()
        .map(|name| (name.clone(), "sampler_noise_lq".to_string()))
        .collect();
    src = rewrite_custom_sampler_identifiers(&src, &replacements);
    src
}

/// Coerce whole-integer-literal arguments of the given builtin calls to float.
/// `pow(x, 2)` → `pow(x, 2.0)`. naga's GLSL frontend resolves `pow`/`max`/`min`
/// overloads strictly by argument type, with no implicit int→float promotion for a
/// bare integer-literal argument. Recurses into each argument so nested calls
/// (`pow(pow(x, 2), 3)`) are fully coerced. Only an argument that is *entirely* an
/// integer literal (optionally signed) is rewritten; expressions like `2*t` are left
/// alone (their literal already promotes inside a float expression).
fn coerce_builtin_int_args(src: &str, fns: &[&str]) -> String {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < n {
        // Match one of `fns` at a word boundary, immediately followed by `(`.
        let mut open: Option<usize> = None;
        if b[i].is_ascii_alphabetic() || b[i] == '_' {
            let before_ok = i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_');
            if before_ok {
                let mut j = i;
                while j < n && (b[j].is_ascii_alphanumeric() || b[j] == '_') {
                    j += 1;
                }
                let ident: String = b[i..j].iter().collect();
                if fns.contains(&ident.as_str()) && j < n && b[j] == '(' {
                    // emit the name and the open paren, then process the arg list
                    out.push_str(&ident);
                    out.push('(');
                    open = Some(j + 1);
                }
                if open.is_none() {
                    out.push_str(&ident);
                    i = j;
                    continue;
                }
            }
        }
        if let Some(arg0) = open {
            // Split top-level args by commas, coercing/recursing each.
            let mut depth = 1i32;
            let mut j = arg0;
            let mut arg_start = arg0;
            let emit_arg = |s: usize, e: usize, out: &mut String| {
                let raw: String = b[s..e].iter().collect();
                let processed = coerce_builtin_int_args(&raw, fns);
                let trimmed = processed.trim();
                let core = trimmed.strip_prefix(['-', '+']).unwrap_or(trimmed);
                if !core.is_empty() && core.chars().all(|c| c.is_ascii_digit()) {
                    // pure integer literal → float, preserving original surrounding ws
                    let lead: String = processed
                        .chars()
                        .take_while(|c| c.is_whitespace())
                        .collect();
                    let trail: String = processed
                        .chars()
                        .rev()
                        .take_while(|c| c.is_whitespace())
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect();
                    out.push_str(&format!("{lead}{trimmed}.0{trail}"));
                } else {
                    out.push_str(&processed);
                }
            };
            while j < n && depth > 0 {
                match b[j] {
                    '(' | '[' => depth += 1,
                    ')' | ']' => {
                        depth -= 1;
                        if depth == 0 {
                            emit_arg(arg_start, j, &mut out);
                        }
                    }
                    ',' if depth == 1 => {
                        emit_arg(arg_start, j, &mut out);
                        out.push(',');
                        arg_start = j + 1;
                    }
                    _ => {}
                }
                j += 1;
            }
            out.push(')');
            i = j; // j is just past the matched close paren
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Rewrite replicated scalar swizzles (`f.xxx`) of known `float` locals to vector
/// constructors (`vec3(f)`). HLSL permits scalar swizzling; GLSL does not, so the
/// native converter's fallback output trips naga "Can't lookup field on this type".
/// Scoped to names declared as plain `float NAME` so genuine vector swizzles
/// (`uv.xy`, `texsize.zw`) are never touched.
fn fix_scalar_swizzles(src: &str) -> String {
    let mut scalars: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in src.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("float ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                let after = rest[name.len()..].trim_start();
                if after.starts_with('=') || after.starts_with(';') {
                    scalars.insert(name);
                }
            }
        }
    }
    if scalars.is_empty() {
        return src.to_string();
    }
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < n {
        if b[i].is_ascii_alphabetic() || b[i] == '_' {
            let start = i;
            let mut j = i;
            while j < n && (b[j].is_ascii_alphanumeric() || b[j] == '_') {
                j += 1;
            }
            let ident: String = b[start..j].iter().collect();
            if scalars.contains(&ident) && j < n && b[j] == '.' {
                let sw_start = j + 1;
                let mut k = sw_start;
                while k < n && matches!(b[k], 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a') {
                    k += 1;
                }
                let sw: String = b[sw_start..k].iter().collect();
                // only pure replication of the first component (`.x…`/`.r…`)
                let replicated = !sw.is_empty()
                    && (sw.chars().all(|c| c == 'x') || sw.chars().all(|c| c == 'r'));
                if replicated {
                    let repl = match sw.len() {
                        1 => ident.clone(),
                        2 => format!("vec2({ident})"),
                        3 => format!("vec3({ident})"),
                        4 => format!("vec4({ident})"),
                        _ => {
                            out.push_str(&ident);
                            out.push('.');
                            out.push_str(&sw);
                            i = k;
                            continue;
                        }
                    };
                    out.push_str(&repl);
                    i = k;
                    continue;
                }
            }
            out.push_str(&ident);
            i = j;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// HLSL implicit truncation for `<narrowType> <id> = texture(...);` declarations
/// (HLSL's tex2D returns float4; assigning to float/float2/float3 truncates).
/// Appends the matching swizzle so GLSL/naga accept the narrower type. Only
/// whole-RHS texture calls are handled (RHS is exactly one texture(...) call).
fn truncate_texture_decls(src: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for line in src.lines() {
        let t = line.trim_start();
        let swiz = if t.starts_with("vec3 ") {
            Some(".xyz")
        } else if t.starts_with("vec2 ") {
            Some(".xy")
        } else if t.starts_with("float ") {
            Some(".x")
        } else {
            None
        };
        if let Some(sw) = swiz {
            // require RHS to be exactly `texture(...)` ending in `);`
            if let Some(eq) = t.find('=') {
                let rhs = t[eq + 1..].trim();
                if rhs.starts_with("texture(") && rhs.ends_with(");") {
                    // confirm the close paren matches the texture( open (single call)
                    let body = &rhs[..rhs.len() - 1]; // drop trailing ';'
                    if is_single_call(body) {
                        let indent = &line[..line.len() - t.len()];
                        let lhs = &t[..eq];
                        out.push(format!("{indent}{lhs}= {}{sw};", body.trim()));
                        continue;
                    }
                }
            }
        }
        out.push(line.to_string());
    }
    out.join("\n")
}

/// HLSL truncation for `<narrowType> <id> = <RHS>;` where RHS is a vec4-typed
/// expression because it references one of MilkDrop's vec4 built-ins (the roam
/// oscillators / rand vectors). Wraps the RHS so GLSL/naga accept the narrower
/// type: `vec3 lay1 = ...roam_cos;` → `vec3 lay1 = (...roam_cos).xyz;`.
fn truncate_vec4_builtin_decls(src: &str) -> String {
    const VEC4_BUILTINS: [&str; 7] = [
        "roam_cos",
        "roam_sin",
        "slow_roam_cos",
        "slow_roam_sin",
        "rand_frame",
        "rand_start",
        "rand_preset",
    ];
    let mut out: Vec<String> = Vec::new();
    for line in src.lines() {
        let t = line.trim_start();
        let sw = if t.starts_with("vec3 ") {
            Some(".xyz")
        } else if t.starts_with("vec2 ") {
            Some(".xy")
        } else if t.starts_with("float ") {
            Some(".x")
        } else {
            None
        };
        if let Some(sw) = sw {
            if let Some(eq) = t.find('=') {
                let rhs_full = t[eq + 1..].trim();
                // only whole single-statement RHS ending in ';', not already swizzled,
                // not a bare texture() (handled elsewhere)
                if rhs_full.ends_with(';') && !rhs_full.starts_with("texture(") {
                    let rhs = rhs_full[..rhs_full.len() - 1].trim();
                    let has_v4 = VEC4_BUILTINS.iter().any(|b| count_word(rhs, b) >= 1);
                    if has_v4 {
                        let indent = &line[..line.len() - t.len()];
                        let lhs = &t[..eq];
                        out.push(format!("{indent}{lhs}= ({rhs}){sw};"));
                        continue;
                    }
                }
            }
        }
        out.push(line.to_string());
    }
    out.join("\n")
}

/// True if `s` is a single balanced `name(...)` call (close paren only at the end).
fn is_single_call(s: &str) -> bool {
    let b = s.as_bytes();
    let open = match s.find('(') {
        Some(i) => i,
        None => return false,
    };
    let mut depth = 0i32;
    for (k, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return k == b.len() - 1;
                }
            }
            _ => {}
        }
    }
    false
}

/// Remove single-line local declarations whose variable is never referenced
/// again in the body (e.g. dead `float corr = <vec2>;` HLSL-truncation cruft).
fn drop_unused_decls(src: &str) -> String {
    // A declaration can span physical lines. Dropping only its first line leaves
    // an orphaned continuation such as `+ time*vec3(...));`, which then fails to
    // parse. Normalize to semicolon-complete logical statements first.
    let logical = join_logical_statements(src);
    let mut out: Vec<&str> = Vec::new();
    for line in logical.lines() {
        if let Some(ident) = decl_ident(line.trim_start()) {
            if count_word(src, &ident) <= 1 {
                continue; // declared once, never used → drop
            }
        }
        out.push(line);
    }
    out.join("\n")
}

/// `sample` is a reserved token in naga's GLSL frontend, but a legal HLSL local
/// name. Rename it only when the shader actually declares a local with that name;
/// this avoids touching sampling syntax or comments and preserves every use.
fn rename_reserved_sample_local(src: &str) -> String {
    let declares_sample = src.lines().any(|line| {
        let trimmed = line.trim_start().trim_start_matches("static ").trim_start();
        const TYPES: &[&str] = &[
            "float", "float2", "float3", "float4", "int", "int2", "int3", "int4", "bool", "bool2",
            "bool3", "bool4", "half", "half2", "half3", "half4", "double", "double2", "double3",
            "double4",
        ];
        TYPES.iter().any(|ty| {
            let Some(rest) = trimmed.strip_prefix(ty) else {
                return false;
            };
            if !rest.starts_with(char::is_whitespace) {
                return false;
            }
            let names = rest.trim_start().split('=').next().unwrap_or(rest);
            split_top_level_commas(names).iter().any(|item| {
                item.trim()
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect::<String>()
                    == "sample"
            })
        })
    });
    if declares_sample {
        replace_word(src, "sample", "particle_sample")
    } else {
        src.to_string()
    }
}

/// If `t` is `<type> <ident> = ...` (a scalar/vector local decl, not `==`),
/// return the declared identifier.
fn decl_ident(t: &str) -> Option<String> {
    for ty in [
        "float ", "vec2 ", "vec3 ", "vec4 ", "int ", "mat2 ", "mat3 ", "mat4 ", "bool ",
    ] {
        if let Some(rest) = t.strip_prefix(ty) {
            let rest = rest.trim_start();
            let ident: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if ident.is_empty() {
                continue;
            }
            let after = rest[ident.len()..].trim_start();
            if after.starts_with('=') && !after.starts_with("==") {
                return Some(ident);
            }
        }
    }
    None
}

/// Count whole-word occurrences of `w` in `src`.
fn count_word(src: &str, w: &str) -> usize {
    let b = src.as_bytes();
    let wb = w.as_bytes();
    let mut n = 0;
    let mut i = 0;
    while let Some(p) = src[i..].find(w) {
        let abs = i + p;
        let before_ok = abs == 0 || (!b[abs - 1].is_ascii_alphanumeric() && b[abs - 1] != b'_');
        let after = abs + wb.len();
        let after_ok = after >= b.len() || (!b[after].is_ascii_alphanumeric() && b[after] != b'_');
        if before_ok && after_ok {
            n += 1;
        }
        i = abs + wb.len();
    }
    n
}

/// Expand `mat2(<single-arg>)` (from HLSL `float2x2(float4)`) into the 4-scalar
/// form `mat2((e).x, (e).y, (e).z, (e).w)`. Single-argument calls only — the
/// 4-scalar / 2-vec2 constructors have a top-level comma and pass through.
fn expand_mat2_from_vec(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let is_call = src[i..].starts_with("mat2(")
            && (i == 0 || {
                let p = bytes[i - 1];
                !p.is_ascii_alphanumeric() && p != b'_'
            });
        if is_call {
            let open = i + 4; // position of '('
            let mut depth = 0i32;
            let mut j = open;
            let mut top_comma = false;
            while j < bytes.len() {
                match bytes[j] {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    b',' if depth == 1 => top_comma = true,
                    _ => {}
                }
                j += 1;
            }
            if j < bytes.len() && !top_comma {
                let inner = src[open + 1..j].trim();
                out.push_str(&format!("mat2(({0}).x, ({0}).y, ({0}).z, ({0}).w)", inner));
                i = j + 1;
                continue;
            }
        }
        let ch = src[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Replace whole-word occurrences of `from` with `to`.
/// Avoids replacing `float2x2` when looking for `float2`, etc.
fn replace_word(src: &str, from: &str, to: &str) -> String {
    let mut result = String::with_capacity(src.len());
    let mut pos = 0;
    while pos < src.len() {
        if let Some(idx) = src[pos..].find(from) {
            let abs = pos + idx;
            // Check that the character before and after are not word characters
            let before_ok = abs == 0
                || !src.as_bytes()[abs - 1].is_ascii_alphanumeric()
                    && src.as_bytes()[abs - 1] != b'_';
            let after = abs + from.len();
            let after_ok = after >= src.len()
                || (!src.as_bytes()[after].is_ascii_alphanumeric()
                    && src.as_bytes()[after] != b'_');
            result.push_str(&src[pos..abs]);
            if before_ok && after_ok {
                result.push_str(to);
            } else {
                result.push_str(from);
            }
            pos = after;
        } else {
            result.push_str(&src[pos..]);
            break;
        }
    }
    result
}

/// Rewrite `tex2D(name, uv)` and `tex2d(name, uv)` →
/// `texture(sampler2D(name, name_samp), uv)`.
fn rewrite_tex2d_calls(src: &str) -> String {
    let mut result = src.to_string();
    // Sort longest-first to avoid partial-name matches
    let mut names = MILKDROP_SAMPLERS.to_vec();
    names.sort_by_key(|s| std::cmp::Reverse(s.len()));

    for name in &names {
        for call in &[
            format!("tex2D({name},"),
            format!("tex2d({name},"),
            format!("tex2D( {name},"),
            format!("tex2d( {name},"),
            format!("tex2D({name} ,"),
            format!("tex2d({name} ,"),
        ] {
            if result.contains(call.as_str()) {
                let replacement = format!("texture(sampler2D({name}, {name}_samp),");
                result = result.replace(call.as_str(), &replacement);
            }
        }
        // tex3D → sampler3D (3D noise-volume textures: sampler_noisevol_*)
        for call in &[
            format!("tex3D({name},"),
            format!("tex3d({name},"),
            format!("tex3D( {name},"),
            format!("tex3d( {name},"),
            format!("tex3D({name} ,"),
            format!("tex3d({name} ,"),
        ] {
            if result.contains(call.as_str()) {
                let replacement = format!("texture(sampler3D({name}, {name}_samp),");
                result = result.replace(call.as_str(), &replacement);
            }
        }
    }
    result
}

/// Replace `mul(a, b)` → `(a) * (b)` for simple non-nested cases.
fn replace_mul(src: &str) -> String {
    // Simple approach: find `mul(` and extract the two comma-separated args.
    // Only handles non-nested single-level mul() calls.
    let mut result = String::new();
    let mut rest = src;
    while let Some(idx) = rest.find("mul(") {
        // Word boundary: the char before `mul` must not be an identifier char, else
        // this is the tail of `cmul(`/`cpow…mul(`/a variable — not the HLSL `mul()`
        // builtin. (Without this, `cmul(a, b)` was mangled to `c((a) * (b))`.)
        let boundary_ok = idx == 0 || {
            let prev = rest[..idx].chars().next_back().unwrap();
            !(prev.is_alphanumeric() || prev == '_')
        };
        if !boundary_ok {
            result.push_str(&rest[..idx + 4]);
            rest = &rest[idx + 4..];
            continue;
        }
        result.push_str(&rest[..idx]);
        let after = &rest[idx + 4..]; // skip "mul("
        if let Some((a, b, end)) = split_two_args(after) {
            if let Some((c0, c1)) = mat2x3_columns(a.trim()) {
                result.push_str(&format!("vec2(dot(({c0}), ({b})), dot(({c1}), ({b})))"));
            } else {
                result.push_str(&format!("(({a}) * ({b}))"));
            }
            rest = &after[end..];
        } else {
            result.push_str("mul(");
            rest = after;
        }
    }
    result.push_str(rest);
    result
}

fn mat2x3_columns(expr: &str) -> Option<(String, String)> {
    let expr = expr.trim();
    let inner = expr.strip_prefix("mat2x3(")?.strip_suffix(')')?;
    let (a, b, end) = split_two_args(&(inner.to_string() + ")"))?;
    if end == inner.len() + 1 {
        Some((a, b))
    } else {
        None
    }
}

/// Split the content inside a two-argument function call `a, b)`.
/// Returns (arg1, arg2, bytes_consumed_including_closing_paren).
fn split_two_args(s: &str) -> Option<(String, String, usize)> {
    let mut depth = 0i32;
    let mut split = None;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    let (a_raw, b_raw) = s[..i].split_at(split?);
                    return Some((
                        a_raw.trim().to_string(),
                        b_raw[1..].trim().to_string(), // skip ','
                        i + 1,
                    ));
                }
                depth -= 1;
            }
            ',' if depth == 0 && split.is_none() => split = Some(i),
            _ => {}
        }
    }
    None
}
