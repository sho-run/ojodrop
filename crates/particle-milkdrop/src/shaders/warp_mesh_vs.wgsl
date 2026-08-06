// Vertex shader that drives the CUSTOM warp fragment shader through the warped mesh.
// Emits, in this exact location order to match the naga-generated FS varyings:
//   vUv     (loc 0) — screen position 0..1 (DirectX-UV, v=0 top) → rad/ang
//   vWarpUv (loc 1) — CPU-computed warped sample coord
//   vDecay  (loc 2) — per-vertex decay rgb

struct WarpParams {
    transform0: vec4<f32>,
    transform1: vec4<f32>,
    transform2: vec4<f32>,
    transform3: vec4<f32>,
    flags:      vec4<f32>,
}
@group(2) @binding(0) var<uniform> wp: WarpParams;

struct VIn {
    @location(0) pos:   vec2<f32>,
    @location(1) uv:    vec2<f32>,
    @location(2) decay: vec4<f32>,
}
struct VOut {
    @builtin(position) clip:    vec4<f32>,
    @location(0)       vUv:     vec2<f32>,
    @location(1)       vWarpUv: vec2<f32>,
    @location(2)       vDecay:  vec4<f32>,
}

fn default_warp(pos: vec2<f32>) -> vec2<f32> {
    var zoom = wp.transform0.x;
    let zoomexp = wp.transform0.y;
    let rot = wp.transform0.z;
    let warp = wp.transform0.w;
    let cx = wp.transform1.x;
    let cy = wp.transform1.y;
    let dx = wp.transform1.z;
    let dy = wp.transform1.w;
    var sx = wp.transform2.x;
    var sy = wp.transform2.y;
    let warpscale = max(wp.transform2.w, 1e-6);
    let warpanimspeed = wp.transform3.x;
    let time = wp.transform3.y;
    let aspectx = wp.transform3.z;
    let aspecty = wp.transform3.w;

    if (abs(zoom) < 1e-6) { zoom = 1e-6; }
    if (abs(sx) < 1e-6) { sx = 1e-6; }
    if (abs(sy) < 1e-6) { sy = 1e-6; }

    let x = pos.x;
    let y = -pos.y;
    let rad = sqrt(x * x * aspectx * aspectx + y * y * aspecty * aspecty);
    let zoom2v = pow(zoom, pow(zoomexp, rad * 2.0 - 1.0));
    let zoom2inv = 1.0 / zoom2v;
    var u = x * 0.5 * aspectx * zoom2inv + 0.5;
    var v = -y * 0.5 * aspecty * zoom2inv + 0.5;
    u = (u - cx) / sx + cx;
    v = (v - cy) / sy + cy;

    if (abs(warp) > 1e-9) {
        let warp_time = time * warpanimspeed;
        let warp_scale_inv = 1.0 / warpscale;
        let warpf0 = 11.68 + 4.0 * cos(warp_time * 1.413 + 10.0);
        let warpf1 = 8.77 + 3.0 * cos(warp_time * 1.113 + 7.0);
        let warpf2 = 10.54 + 3.0 * cos(warp_time * 1.233 + 3.0);
        let warpf3 = 11.49 + 4.0 * cos(warp_time * 0.933 + 5.0);
        u += warp * 0.0035 * sin(warp_time * 0.333 + warp_scale_inv * (x * warpf0 - y * warpf3));
        v += warp * 0.0035 * cos(warp_time * 0.375 - warp_scale_inv * (x * warpf2 + y * warpf1));
        u += warp * 0.0035 * cos(warp_time * 0.753 - warp_scale_inv * (x * warpf1 - y * warpf2));
        v += warp * 0.0035 * sin(warp_time * 0.825 + warp_scale_inv * (x * warpf0 + y * warpf3));
    }

    let u2 = u - cx;
    let v2 = v - cy;
    let cr = cos(rot);
    let sr = sin(rot);
    u = u2 * cr - v2 * sr + cx - dx;
    v = u2 * sr + v2 * cr + cy - dy;
    return vec2<f32>((u - 0.5) / aspectx + 0.5, (v - 0.5) / aspecty + 0.5);
}

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    o.clip = vec4<f32>(v.pos, 0.0, 1.0);
    // Screen 0..1 DirectX-UV (matches quad.wgsl): top row (pos.y=+1) -> v=0.
    o.vUv     = vec2<f32>((v.pos.x + 1.0) * 0.5, (1.0 - v.pos.y) * 0.5);
    if (wp.flags.x >= 0.5) {
        o.vWarpUv = vec2<f32>(v.uv.x, 1.0 - v.uv.y);
        o.vDecay = v.decay;
    } else {
        let warp_uv = default_warp(v.pos);
        o.vWarpUv = vec2<f32>(warp_uv.x, 1.0 - warp_uv.y);
        o.vDecay = vec4<f32>(vec3<f32>(wp.transform2.z), 1.0);
    }
    return o;
}
