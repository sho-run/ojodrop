# OjoDrop

![OjoDrop running a MilkDrop preset](media/ojodrop-demo.gif)

 OjoDrop is a standalone, open-source (MIT) macOS app: a window where you drag
 a `.milk` MilkDrop preset (or a pre-converted `.json`) and it renders, live,
 reacting to your microphone. It is a native Rust + [wgpu](https://wgpu.rs)
 reimplementation of the MilkDrop / [Butterchurn](https://github.com/jberg/butterchurn) engine — no browser, no
Node, no Winamp. The `.milk` → GLSL converter is **baked into the binary**.

> OjoDrop is the app layer over the `particle-milkdrop` engine crate, which is the
> single source of truth for the renderer. OjoDrop adds drag-and-drop, in-process
> `.milk` ingestion, crash-safety against arbitrary files, packaging, and the
> open-source scaffolding.

DISCLAIMER: I built this out of sheer curiosity, on whether I could build actually build something
that runs Milkdrop presets but doesn't require a webview or WebGL layer. It runs on wgpu
and Rust, with a dash of C++ if you are interested in the HLSL conversion. 

What could it be used for? Who knows, but here's some ideas I came up with while eating
pizza:

[Creating an Oculus VR game to simulate the “Magic Milk” experiment from childhood
without milk, food coloring, or childhood](https://youtube.com/shorts/GGCwdV-I-0c?si=Piz4epG5nQG6AlAf) 

[Re-create your own version of this music video without a Nintendo N64. Just add
go karts and a Casio Keyboard.](https://www.youtube.com/watch?v=FuX5_OWObA0)

[Develop your own karaoke app so you can create an open source alternative to Karafun](https://www.youtube.com/watch?v=_WwA-02ZCks)

[Monetize your own 85-year-long YouTube video and do it without a web browser engine underneath](https://www.youtube.com/watch?v=qirWins5tus)

## Why call it OjoDrop?

Because when it first rendered successfully I went "OH HO HO!" 

## Does it actually work?

Mostly. In fidelity tests with the "cream of the crop" presets, I encountered a 97% pass rate
for ingest, and a 70% pass rate for fidelity and image quality. The way all Milkdrop iterations
interpret presets is a little fuzzy, so I made some middle-of-the-road decisions when finding differences.
I may iterate more to get the fidelity rate up, but there will be diminishing returns unless
I found a couple of slam dunk levers to move the needle.

## Update — August 6, 2026 (8/6/26)

OjoDrop just received a substantial round of MilkDrop compatibility, bug-fix,
and visual-fidelity improvements. The goal is recognizable behavior relative to
Butterchurn—not byte-identical frames and not one-off fixes for individual
presets.

- Fixed feedback texture-coordinate conventions in both the warp and composite
  paths. Recursive presets now develop in the expected direction instead of
  slowly diverging because WebGL and WGPU use opposite texture origins.
- Added Butterchurn-style composite-mesh behavior, including its 32×24 mesh and
  interpolated per-vertex color field.
- Fixed several EEL equation-analysis and preset-preprocessing edge cases used
  by more demanding MilkDrop presets.
- Rechecked a balanced selection of `$$$ Royal` and three-digit presets at
  1920×1080 using a real MP3 for audio reactivity. All 50 selected presets
  converted and rendered successfully, with substantially closer feedback,
  geometry, motion, and color behavior overall.
- Kept the renderer general-purpose: these changes do not add preset-specific
  brightness, palette, or gain hacks.

## Is it faster than WebGL2? Was it worth it to move to WGPU?

Listen. I just make the stuff. I was curious on Reddit 3 days ago and boom, this popped out of Claude.
I not going to host an episode of milkdrop Top Gear. 
Feel free to benchmark and create a snarky YouTube video. Something with large fonts
where you're pointing to a 4k monitor and saying something like "THE BUTTERCHURN CHURNER?!"


## Use

- **Launch it** → an empty window: *"drag a .milk or .json preset here."*
- **Drag a `.milk`** onto the window → it's converted in-process and rendered.
- **Drag a `.json`** (Butterchurn-exported preset) → loaded directly.
- A malformed or hostile file shows an error in the title bar and the app keeps
  running — it shouldn't crash on bad input, but hey, code always loves to surprise
  us.
- `Esc` quits. `--about` prints credits and license info.

Audio comes from your default input device (the room mic); if none is available
the engine falls back to synthetic reactivity so presets still move.

## Build & run from source

OjoDrop links two upstream C++ shader libraries
([hlsl2glslfork](https://github.com/aras-p/hlsl2glslfork) +
[glsl-optimizer](https://github.com/aras-p/glsl-optimizer)) to convert `.milk`
HLSL into GLSL. They are vendored as **git submodules** and built from source.

```sh
# 1. Clone with submodules (or `git submodule update --init --recursive` after).
git clone --recurse-submodules <repo-url>
cd ojodrop

# 2. Build & run. The native converter is ON by default.
cargo run --release                 # opens the empty-state window
cargo run --release -- preset.milk  # boots straight into a preset
```

**Build dependencies:** a C/C++ toolchain and `cmake` (for the converter
sources). On macOS: `xcode-select --install` and `brew install cmake`.

### JSON-only build (no C++ toolchain)

If you don't want the converter, build without it — `.json` presets still load,
and dropping a `.milk` reports the converter is unavailable rather than failing:

```sh
cargo run --release --no-default-features
```

The converter sys-crate also **degrades gracefully**: if the submodules aren't
initialized, it compiles a no-op stub instead of failing the build, so a clean
clone always builds even before you fetch the converter sources.

## macOS .app bundle

```sh
cargo install cargo-bundle      # once
cargo bundle --release          # produces target/release/bundle/osx/OjoDrop.app
```

The bundle metadata and icon live under [`bundle/`](./bundle/). App signing and
notarization for distribution are out of scope here (note for later).

## Credits

OjoDrop exists because of **Ryan Geiss** (MilkDrop), **Jordan "jberg" Berg**
(Butterchurn + the shader converter), **Nullsoft / Winamp**, and the
hlsl2glslfork / glsl-optimizer / Mesa / MojoShader authors. See
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for full attribution and the
licensing verdict. 

OjoDrop also exists because weed and other drugs were accessible to elder
millenials and Gen-Xers during college, and milkdrop was this fun thing
everyone could just watch for hours and forget that their organic chemistry
midterm was the next day.

Lastly, Claude is to blame for this. I didn't think it was possible until I just started throwing
random platforms at the thing.

So, to the creators listed above, the consumers of milkdrop over the past 3 decades,
and drugs, thank you all.

## License

MIT — see [`LICENSE`](./LICENSE). Bundled third-party components retain their own
permissive licenses (BSD-3 / zlib / MIT), reproduced in `THIRD_PARTY_NOTICES.md`.

## Contribute

I don't know how often I'll check this, but if anyone wants to improve upon the engine I'm all for it.
Most of this was vibe coded. So feel free to vibe code along with me. If you find an em-dash, please
don't hurt me. Birds aren't real. Or they might be. Either way, always keen to hear opinions or entertain updates.

Love, peace, and chicken grease,

Kenny
