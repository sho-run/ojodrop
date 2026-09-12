//! # particle-milkdrop
//!
//! A native Rust + wgpu MilkDrop / Butterchurn preset player. This crate is the
//! **single source of truth** for the render engine; the `particle-milkdrop`
//! binary and the OjoDrop app built from it consume it as a library. Keep this
//! surface reusable — do not fork the engine.
//!
//! ## Public API surface (the seam other crates build on)
//!
//! - [`MilkdropRenderer`] — the wgpu render path (feedback ring → warp → comp).
//!   Construct with [`MilkdropRenderer::new`], drive per-frame with
//!   `set_audio`/`set_waveform`/`set_freq_spectrum` then `render`.
//! - [`MilkShaders`] — the parsed preset (per-frame/per-vertex EEL + warp/comp
//!   shader bodies + base values) fed to the renderer.
//! - [`load_preset_path`] / [`load_preset_str`] — high-level ingest that routes
//!   `.milk` (raw HLSL bodies, run through the native converter when the
//!   `milk-native-converter` feature is on) and `.json` (Butterchurn-converted
//!   GLSL bodies) to the same `MilkShaders`.
//! - [`fallback_preset`] — a known-good passthrough preset, used as a graceful
//!   fallback when an untrusted file fails to load or compile.
//! - [`native_converter_available`] — whether this process has a runnable,
//!   isolated helper for raw `.milk` ingestion.
//!
//! Lower-level modules ([`parse_milk`], [`load_json`], [`preprocess`],
//! [`equations`], [`renderer`]) are public for advanced consumers but most callers
//! only need the items re-exported at the crate root.

pub mod equations;
pub mod enhanced_audio;
pub mod extended_waveforms;
pub mod load_json;
pub mod named_textures;
pub mod parse_milk;
pub mod preprocess;
pub mod renderer;

use std::path::Path;

pub use named_textures::{
    NamedSamplerBinding, NamedTextureArray, NamedTextureAtlas, NamedTextureConfig,
    NamedTexturePlan, NamedTextureResolver, NamedTextureSource, ResolvedNamedTexture,
    SamplerAddressMode, SamplerFilterMode, DEFAULT_NAMED_TEXTURE_LAYER_SIZE,
    MAX_NAMED_TEXTURE_LAYERS, NAMED_TEXTURE_ATLAS_GRID, NAMED_TEXTURE_ATLAS_GUTTER,
};
pub use parse_milk::{parse, CustomWaveDef, MilkShaders, ShapeBaseVals, ShapeCode};
pub use renderer::{
    compile_glsl, compile_milkdrop_shader_bodies, compile_milkdrop_shader_bodies_from_parts,
    CompiledMilkdropShaderBodies, DimensionError, MilkBaseVals, MilkdropAlphaSummary,
    MilkdropGeometryBounds, MilkdropGeometryDiagnostics, MilkdropRenderer, MilkdropResizeDebouncer,
    MilkdropRgbSummary, MilkdropShapeGeometryDiagnostics, MilkdropWaveGeometryDiagnostics,
    INTERACTIVE_RESIZE_DEBOUNCE,
};

/// Whether this process can ingest raw `.milk` presets through the isolated
/// HLSL→GLSL helper (hlsl2glslfork + glsl-optimizer).
///
/// When `false`, only pre-converted `.json` presets load; a dropped `.milk`
/// should report that the converter is unavailable rather than rendering a blank.
/// This runtime probe is `true` only when the `milk-native-converter` feature is
/// enabled and a helper with the C++ payload can be launched successfully.
pub fn native_converter_available() -> bool {
    #[cfg(feature = "milk-native-converter")]
    {
        particle_milkdrop_converter_sys::helper_available()
    }
    #[cfg(not(feature = "milk-native-converter"))]
    {
        false
    }
}

/// Release invariant (P2-VIS-040): the `milk-native-converter` feature is what the
/// package metadata (`Cargo.toml` `[package.metadata.bundle]`) and the crate docs
/// advertise as raw `.milk` support. Enabling that feature MUST link the converter
/// sys crate and keep its capability marker reachable — otherwise the crate would
/// advertise raw `.milk` while silently shipping JSON-only behavior.
///
/// Referencing the sys crate's compile-time marker here makes the wiring a hard
/// build requirement: drop the `particle-milkdrop-converter-sys` dependency (or its
/// wiring) while leaving the advertised feature enabled and this fails to compile,
/// instead of degrading to a silent JSON-only ship. The runtime half of the
/// invariant (the advertised capability tracks the actual converter) is asserted by
/// `converter_availability_invariant::converter_availability_matches_feature_wiring`.
#[cfg(feature = "milk-native-converter")]
const _MILK_NATIVE_CONVERTER_WIRED: bool =
    particle_milkdrop_converter_sys::NATIVE_CONVERTER_AVAILABLE;

/// Load a preset from a string, dispatching on `is_json`:
/// `true` → the Butterchurn converted-JSON loader (GLSL shader bodies);
/// `false` → the raw `.milk` parser (HLSL shader bodies, run through the native
/// converter when `milk-native-converter` is enabled). Both yield a
/// [`MilkShaders`] for the same renderer.
///
/// Returns `Err` only for malformed JSON. The `.milk` parser is infallible at the
/// parse stage — an unsupported shader is caught later by the renderer, which
/// falls back rather than erroring here.
pub fn load_preset_str(content: &str, is_json: bool) -> Result<MilkShaders, String> {
    if is_json {
        load_json::load(content).map_err(|e| format!("cannot load JSON preset: {e}"))
    } else {
        Ok(parse_milk::parse(content))
    }
}

/// Load a preset from disk, dispatching on file extension (`.json` → Butterchurn
/// loader, anything else → raw `.milk` parser). See [`load_preset_str`].
pub fn load_preset_path(path: &Path) -> Result<MilkShaders, String> {
    // Match the `.json` predicate used at the call sites (and the foundation
    // `load_preset`): a full, case-insensitive ".json" suffix, so dispatch and the
    // UI/CLI gates never disagree on a pathological name.
    let is_json = path
        .to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".json");
    let content = if is_json {
        std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?
    } else {
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        String::from_utf8_lossy(&bytes).into_owned()
    };
    load_preset_str(&content, is_json).map_err(|e| format!("{} ({})", e, path.display()))
}

/// A known-good, always-compilable passthrough preset used as a graceful fallback
/// when an untrusted file fails to load or its shader fails to compile. `parse("")`
/// yields all-default scalars with `warp`/`comp` = `None`, which the renderer
/// renders as a plain feedback passthrough rather than crashing.
pub fn fallback_preset() -> MilkShaders {
    parse_milk::parse("")
}

#[cfg(test)]
mod converter_availability_invariant {
    //! P2-VIS-040: the advertised raw-`.milk` capability must match the actual
    //! converter wiring. The crate must never silently ship JSON-only behavior while
    //! its package metadata / docs promise raw `.milk` support via the
    //! `milk-native-converter` feature.

    #[test]
    fn converter_availability_matches_feature_wiring() {
        // Without the advertised feature (a lean / JSON-only build), the runtime
        // probe MUST report the converter as unavailable, so a dropped `.milk` is
        // refused rather than silently rendered blank while claiming support.
        #[cfg(not(feature = "milk-native-converter"))]
        {
            assert!(
                !crate::native_converter_available(),
                "a JSON-only build (no milk-native-converter feature) must report \
                 native_converter_available() == false"
            );
        }

        // With the advertised feature on, the runtime capability MUST be delegated to
        // the converter sys crate (not hardcoded/faked), so the metadata's raw-`.milk`
        // promise tracks the real, linked converter and cannot silently drift to
        // JSON-only.
        #[cfg(feature = "milk-native-converter")]
        {
            assert_eq!(
                crate::native_converter_available(),
                particle_milkdrop_converter_sys::helper_available(),
                "native_converter_available() must delegate to the converter sys \
                 crate; the advertised raw-.milk feature is not actually wired"
            );
        }
    }
}
