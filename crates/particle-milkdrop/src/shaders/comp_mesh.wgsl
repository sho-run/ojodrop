struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) vUv: vec2<f32>,
    @location(1) vColor: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> VOut {
    var out: VOut;
    out.clip = vec4<f32>(pos, 0.0, 1.0);
    // WebGL framebuffer textures and WebGPU textures expose opposite V origins.
    // Present the comp fragment with WebGL-equivalent screen UVs so its authored
    // MilkDrop/Butterchurn Y flip samples the same feedback texels under WGPU.
    out.vUv = vec2<f32>((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    out.vColor = color;
    return out;
}
