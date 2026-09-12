//! BeatDrop extended base-waveform geometry (MilkDrop modes 8 through 17).
//!
//! Adapted from `vis_milk2/milkdropfs.cpp` in BeatDrop commit
//! `945ae10ecf928d24717b64f4e1a69b2c100c4829` (inspected 2026-09-12), under
//! the BSD 3-Clause License:
//!
//! Copyright (c) 2018 Maxim Volskiy and individual contributors.
//! All rights reserved.
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//!
//! * Redistributions of source code must retain the above copyright notice,
//!   this list of conditions and the following disclaimer.
//!
//! * Redistributions in binary form must reproduce the above copyright notice,
//!   this list of conditions and the following disclaimer in the documentation
//!   and/or other materials provided with the distribution.
//!
//! * Neither the name of the copyright holder nor the names of its contributors
//!   may be used to endorse or promote products derived from this software
//!   without specific prior written permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//! ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//! LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//! CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//! SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//! INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//! CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//! ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//! POSSIBILITY OF SUCH DAMAGE.
//!
//! This intentionally has no renderer dependency. The renderer owns smoothing,
//! the NDC Y flip, colors, blending, and GPU upload; this module owns only the
//! source formulas and their bounded raw polylines.

pub const FIRST_EXTENDED_WAVE_MODE: i32 = 8;
pub const LAST_EXTENDED_WAVE_MODE: i32 = 17;
/// BeatDrop's built-in waveform source is 512 samples. Keep raw helper output
/// within that ceiling even if an upstream audio provider supplies more.
pub const MAX_EXTENDED_WAVE_POINTS: usize = 512;

#[derive(Clone, Copy, Debug)]
pub struct ExtendedWaveformInput<'a> {
    pub mode: i32,
    pub time: f32,
    /// Already converted from MilkDrop's 0..1 `wave_x` / `wave_y` to NDC.
    pub wave_pos_x: f32,
    pub wave_pos_y: f32,
    pub wave_param: f32,
    pub aspect_x: f32,
    pub aspect_y: f32,
    pub screen_dependent: bool,
    pub render_width: usize,
    pub left: &'a [f32],
    pub right: &'a [f32],
    /// BeatDrop mode 8 accesses two adjacent spectrum bins per vertex.
    pub spectrum_left: &'a [f32],
    pub spectrum_right: &'a [f32],
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
    pub alpha: f32,
    pub modulate_alpha_by_volume: bool,
    pub modwave_alpha_start: f32,
    pub modwave_alpha_end: f32,
    pub blending: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedWaveformGeometry {
    /// One strip for most modes; two disjoint strips for modes 9–11.
    pub strips: Vec<Vec<[f32; 2]>>,
    /// Apply after the normal renderer color calculation, before the alpha gate.
    pub alpha: f32,
    /// BeatDrop documents mode 17 as a point/particle waveform.
    pub points_recommended: bool,
}

impl ExtendedWaveformGeometry {
    fn empty(alpha: f32) -> Self {
        Self {
            strips: Vec::new(),
            alpha,
            points_recommended: false,
        }
    }
}

/// Builds the raw, pre-Y-flip extended waveform geometry.
///
/// Returns `None` for modes outside 8..=17. Every generated strip is bounded
/// to [`MAX_EXTENDED_WAVE_POINTS`], and malformed/non-finite input yields an
/// empty geometry rather than non-finite vertices.
pub fn build_extended_waveform(
    input: ExtendedWaveformInput<'_>,
) -> Option<ExtendedWaveformGeometry> {
    if !(FIRST_EXTENDED_WAVE_MODE..=LAST_EXTENDED_WAVE_MODE).contains(&input.mode) {
        return None;
    }
    if !input.time.is_finite()
        || !input.wave_pos_x.is_finite()
        || !input.wave_pos_y.is_finite()
        || !input.wave_param.is_finite()
        || !input.aspect_x.is_finite()
        || !input.aspect_y.is_finite()
    {
        return Some(ExtendedWaveformGeometry::empty(0.0));
    }

    let n = input
        .left
        .len()
        .min(input.right.len())
        .min(MAX_EXTENDED_WAVE_POINTS);
    if n == 0 {
        return Some(ExtendedWaveformGeometry::empty(0.0));
    }
    let mut alpha = input.alpha;
    if input.modulate_alpha_by_volume {
        let denom = input.modwave_alpha_end - input.modwave_alpha_start;
        if denom.abs() > f32::EPSILON {
            alpha *=
                ((input.bass + input.mid + input.treble) / 3.0 - input.modwave_alpha_start) / denom;
        }
    }
    if matches!(input.mode, 12 | 15) {
        alpha *= 1.25;
    }
    alpha = alpha.clamp(0.0, 1.0);

    let scale_x = if input.screen_dependent {
        1.0
    } else {
        input.aspect_y
    };
    let scale_y = if input.screen_dependent {
        1.0
    } else {
        input.aspect_x
    };
    let limit = |want: usize| want.min(MAX_EXTENDED_WAVE_POINTS);
    let sample = |data: &[f32], index: usize| data.get(index).copied().unwrap_or(0.0);
    let close = |points: &mut Vec<[f32; 2]>| {
        if !input.blending && points.len() < MAX_EXTENDED_WAVE_POINTS {
            if let Some(first) = points.first().copied() {
                points.push(first);
            }
        }
    };

    let mut geometry = ExtendedWaveformGeometry::empty(alpha);
    match input.mode {
        8 => {
            // BeatDrop's unfinished spectrum counterpart to mode 6.
            let points = limit(256)
                .min(input.spectrum_left.len() / 2)
                .min(input.spectrum_right.len() / 2);
            if points == 0 {
                return Some(geometry);
            }
            let (edge0, delta, perp) = clipped_line(input.wave_pos_x, input.wave_param, points);
            let mut strip = Vec::with_capacity(points);
            for i in 0..points {
                let sum = (sample(input.spectrum_left, i * 2)
                    + sample(input.spectrum_left, i * 2 + 1)
                    + sample(input.spectrum_right, i * 2)
                    + sample(input.spectrum_right, i * 2 + 1))
                    * 0.5;
                // log(0) was unbounded in the source. The finite floor retains
                // the spectrum response without passing -inf to wgpu.
                let offset = 0.1 * sum.max(f32::MIN_POSITIVE).ln();
                strip.push([
                    edge0[0] + delta[0] * i as f32 + perp[0] * offset,
                    edge0[1] + delta[1] * i as f32 + perp[1] * offset,
                ]);
            }
            geometry.strips.push(strip);
        }
        9 => {
            let points = line_points(n, input.render_width);
            if points == 0 {
                return Some(geometry);
            }
            let offset = (n - points) / 2;
            let (edge0, delta, perp) = clipped_line(input.wave_pos_x, input.wave_param, points);
            let mut strip = Vec::with_capacity(points);
            for i in 0..points {
                let amp = (sample(input.left, i + offset) + sample(input.right, i + offset)) * 0.5;
                strip.push([
                    edge0[0] + delta[0] * i as f32 + perp[0] * amp,
                    edge0[1] + delta[1] * i as f32 + perp[1] * amp,
                ]);
            }
            geometry.strips.push(strip);
            // `milkdropfs.cpp` sets nBreak then doubles nVerts without writing
            // the second half. Preserve that source-visible zero-initialized tail
            // as a separate strip rather than silently inventing another shape.
            geometry.strips.push(vec![[0.0, 0.0]; points]);
        }
        10 => {
            let points = line_points(n, input.render_width);
            if points == 0 {
                return Some(geometry);
            }
            let offset = (n - points) / 2;
            geometry.strips.push(x_strip(
                -0.75 + input.wave_param * 3.15,
                input.wave_pos_x,
                input.left,
                offset,
                points,
            ));
            geometry.strips.push(x_strip(
                0.75 + input.wave_param * 3.15,
                input.wave_pos_x,
                input.right,
                offset,
                points,
            ));
        }
        11 => {
            let points = line_points(n, input.render_width);
            if points == 0 {
                return Some(geometry);
            }
            let offset = (n - points) / 2;
            geometry.strips.push(vertical_strip(
                -0.45,
                input.wave_pos_x,
                input.left,
                offset,
                points,
            ));
            geometry.strips.push(vertical_strip(
                0.45,
                input.wave_pos_x,
                input.right,
                offset,
                points,
            ));
        }
        12 => {
            let points = n / 2;
            let mut strip = Vec::with_capacity(points);
            for i in 0..points {
                let rad = 0.63 + 0.23 * sample(input.right, i) + input.wave_param;
                let angle = sample(input.left, i + 32) * 0.9 + input.time * 3.3;
                strip.push([
                    rad * (angle + alpha).cos() * scale_x + input.wave_pos_x,
                    rad * angle.sin() * scale_y + input.wave_pos_y,
                ]);
            }
            geometry.strips.push(strip);
        }
        13 => {
            let points = n / 2;
            if points == 0 {
                return Some(geometry);
            }
            let mut strip = Vec::with_capacity(limit(points + 1));
            for i in 0..points {
                let mut rad = 0.7
                    + 0.4
                        * (sample(input.left, i + (n - points) / 2)
                            + sample(input.right, i + (n - points) / 2))
                        * 0.5
                    + input.wave_param;
                let angle = i as f32 / (points - 1).max(1) as f32 * 6.28 + input.time * 0.2;
                if (i as f32) < points as f32 / rad.max(f32::MIN_POSITIVE) {
                    let mix_arg = i as f32 / (points as f32 * 0.1);
                    let mix = 0.5 - 0.5 * (mix_arg * 3.1416).cos();
                    let next = i + points + (n - points) / 2;
                    let rad2 = 0.5
                        + 0.4 * (sample(input.left, next) + sample(input.right, next)) * 0.5
                        + input.wave_param;
                    rad = rad2 * (1.0 - mix) + rad * mix;
                }
                strip.push([
                    rad * angle.cos() * scale_x + input.wave_pos_x,
                    rad * angle.sin() * scale_y + input.wave_pos_y,
                ]);
            }
            close(&mut strip);
            geometry.strips.push(strip);
        }
        14 => {
            let points = n / 2;
            if points == 0 {
                return Some(geometry);
            }
            let mut strip = Vec::with_capacity(limit(points + 1));
            for i in 0..points {
                let offset = (n - points) / 2;
                let mut rad = 0.7
                    + 0.7
                        * (sample(input.left, i + offset) + sample(input.right, i + offset))
                        * 0.5
                    + input.wave_param;
                let angle = i as f32 / (points - 1).max(1) as f32 * 6.28 + input.time * 0.2;
                // The two `==` lines in the source are comparisons, not
                // assignments, and therefore deliberately have no effect.
                if (i as f32) < points as f32 / rad.max(f32::MIN_POSITIVE) {
                    let mix_arg = i as f32 / (points as f32 * 0.1);
                    let mix = 0.7 - 0.7 * (mix_arg * 3.1416).cos();
                    let next = i + points + offset;
                    let rad2 = 0.7
                        + 0.7 * (sample(input.left, next) + sample(input.right, next)) * 0.5
                        + input.wave_param;
                    rad = rad2 * (1.0 - mix) + rad * (mix * 2.0) / 8.0;
                }
                strip.push([
                    rad * (angle * 3.1416).cos() * scale_x / 1.5
                        + input.wave_pos_x * 3.1416_f32.cos(),
                    rad * (angle - input.time / 3.0).sin() * scale_y / 1.5
                        + input.wave_pos_y * 3.1416_f32.cos(),
                ]);
            }
            close(&mut strip);
            geometry.strips.push(strip);
        }
        15 => {
            let points = n / 2;
            let mut strip = Vec::with_capacity(points);
            for i in 0..points {
                let mut angle =
                    (sample(input.left, i + 32) + sample(input.right, i + 32)) * 0.5 * 1.57
                        + input.time * 2.0;
                if angle.abs() < 0.001 {
                    angle = if angle.is_sign_negative() {
                        -0.001
                    } else {
                        0.001
                    };
                }
                let tangent = (input.time / angle).tan();
                strip.push([
                    input.time.cos() / 2.0 + (angle * 2.0 + tangent).cos(),
                    input.time.sin() * 2.0 * (angle * 3.14).sin() * scale_y / 2.8
                        + input.wave_pos_y,
                ]);
            }
            geometry.strips.push(strip);
        }
        16 => {
            let points = n / 2;
            if points == 0 {
                return Some(geometry);
            }
            let mut strip = Vec::with_capacity(limit(points + 1));
            let offset = input.time * 0.2;
            for i in 0..points {
                let progress = i as f32 / points as f32;
                let phi0 = ((progress * 3.0).floor() + 0.5) / 3.0 * 6.28 + offset;
                let angle = progress * 6.28 + offset;
                let mut edge = (angle - phi0).cos();
                if edge.abs() < 0.02 {
                    edge = if edge.is_sign_negative() { -0.02 } else { 0.02 };
                }
                let radius = ((0.7
                    + edge * (sample(input.left, i) + sample(input.right, i)) * 0.5
                    + input.wave_param)
                    / (2.0 * edge))
                    .clamp(-2.0, 2.0);
                strip.push([
                    radius * angle.cos() * scale_x + input.wave_pos_x,
                    radius * angle.sin() * scale_y + input.wave_pos_y,
                ]);
            }
            close(&mut strip);
            geometry.strips.push(strip);
        }
        17 => {
            let points = limit(256);
            let frequency = 1.0 - input.wave_param + 0.001;
            if !(frequency > f32::EPSILON && frequency.is_finite()) {
                return Some(geometry);
            }
            let phase = input.time.rem_euclid(frequency) / frequency;
            let burst_number = (input.time / frequency) as i32;
            let seed = burst_number as f32 * 10.0;
            let fract = |v: f32| v - v.floor();
            let mut base_x = fract(seed * 0.1345) * 2.0 - 1.0;
            let mut base_y = fract(seed * 0.2783) * 2.0 - 1.0;
            if seed.rem_euclid(1.0) > 0.3 {
                base_x *= 0.3;
                base_y *= 0.3;
            }
            let burst_size = (phase * 4.0).min(1.0);
            let fade = 1.0 - phase.powi(3);
            let audio_boost = 1.0 + 2.0 * (input.bass + input.mid) * 0.5;
            let mut strip = Vec::with_capacity(limit(points + 1));
            for i in 0..points {
                let angle = i as f32 / points as f32 * 6.283185;
                let distance_variation = 0.7 + 0.3 * fract(seed + i as f32 * 0.1);
                let amp =
                    (sample(input.left, (i * 3) % n) + sample(input.right, (i * 3) % n)) * 0.5;
                let distance = burst_size * distance_variation * (0.5 + 0.5 * amp) * audio_boost;
                let swirl = input.time * 3.0 + angle;
                let x = base_x + angle.cos() * distance + swirl.cos() * burst_size * 0.1;
                let y = base_y + angle.sin() * distance + swirl.sin() * burst_size * 0.1;
                strip.push([
                    x * scale_x + input.wave_pos_x,
                    y * scale_y + input.wave_pos_y,
                ]);
                // This is intentionally per-vertex: BeatDrop mutates the one
                // shared alpha within its particle loop, yielding fade^256.
                alpha *= fade;
            }
            close(&mut strip);
            geometry.alpha = alpha;
            geometry.points_recommended = true;
            geometry.strips.push(strip);
        }
        _ => unreachable!("range checked above"),
    }
    Some(geometry)
}

fn line_points(samples: usize, render_width: usize) -> usize {
    (samples / 2)
        .min(render_width / 3)
        .min(MAX_EXTENDED_WAVE_POINTS / 2)
}

/// Exact BeatDrop clipping setup used by modes 6–11, including its historic
/// `wave_pos_x`-for-both-axes seed quirk.
fn clipped_line(wave_pos_x: f32, wave_param: f32, points: usize) -> ([f32; 2], [f32; 2], [f32; 2]) {
    let angle = 1.57 * wave_param;
    clipped_line_at_angle(wave_pos_x, angle, points)
}

fn clipped_line_at_angle(
    wave_pos_x: f32,
    angle: f32,
    points: usize,
) -> ([f32; 2], [f32; 2], [f32; 2]) {
    let mut edge = [
        [
            wave_pos_x * (angle + 1.57).cos() - angle.cos() * 3.0,
            wave_pos_x * (angle + 1.57).sin() - angle.sin() * 3.0,
        ],
        [
            wave_pos_x * (angle + 1.57).cos() + angle.cos() * 3.0,
            wave_pos_x * (angle + 1.57).sin() + angle.sin() * 3.0,
        ],
    ];
    for i in 0..2 {
        for bound in 0..4 {
            let value = match bound {
                0 => 1.1,
                1 => -1.1,
                2 => 1.1,
                _ => -1.1,
            };
            let coordinate = if bound < 2 { 0 } else { 1 };
            let exceeds = if bound % 2 == 0 {
                edge[i][coordinate] > value
            } else {
                edge[i][coordinate] < value
            };
            if exceeds {
                let other = 1 - i;
                let denom = edge[i][coordinate] - edge[other][coordinate];
                if denom.abs() > f32::EPSILON {
                    let t = (value - edge[other][coordinate]) / denom;
                    edge[i][0] = edge[other][0] + (edge[i][0] - edge[other][0]) * t;
                    edge[i][1] = edge[other][1] + (edge[i][1] - edge[other][1]) * t;
                }
            }
        }
    }
    let delta = [
        (edge[1][0] - edge[0][0]) / points.max(1) as f32,
        (edge[1][1] - edge[0][1]) / points.max(1) as f32,
    ];
    let perpendicular_angle = delta[1].atan2(delta[0]) + 1.57;
    (
        edge[0],
        delta,
        [perpendicular_angle.cos(), perpendicular_angle.sin()],
    )
}

fn x_strip(
    angle: f32,
    wave_pos_x: f32,
    samples: &[f32],
    offset: usize,
    points: usize,
) -> Vec<[f32; 2]> {
    let (edge, delta, perp) = clipped_line_at_angle(wave_pos_x, angle, points);
    (0..points)
        .map(|i| {
            let value = samples.get(i + offset).copied().unwrap_or(0.0);
            [
                edge[0] + delta[0] * i as f32 + perp[0] * 0.35 * value,
                edge[1] + delta[1] * i as f32 + perp[1] * 0.35 * value,
            ]
        })
        .collect()
}

fn vertical_strip(
    x_shift: f32,
    wave_pos_x: f32,
    samples: &[f32],
    offset: usize,
    points: usize,
) -> Vec<[f32; 2]> {
    let (edge, delta, perp) = clipped_line_at_angle(wave_pos_x, 1.57, points);
    (0..points)
        .map(|i| {
            let value = samples.get(i + offset).copied().unwrap_or(0.0);
            [
                edge[0] + x_shift + delta[0] * i as f32 + perp[0] * 0.35 * value,
                edge[1] + delta[1] * i as f32 + perp[1] * 0.35 * value,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(mode: i32) -> ExtendedWaveformInput<'static> {
        static LEFT: [f32; 512] = [0.25; 512];
        static RIGHT: [f32; 512] = [-0.125; 512];
        static SPECTRUM: [f32; 512] = [1.0; 512];
        ExtendedWaveformInput {
            mode,
            time: 1.0,
            wave_pos_x: 0.0,
            wave_pos_y: 0.0,
            wave_param: 0.25,
            aspect_x: 1.0,
            aspect_y: 1.0,
            screen_dependent: false,
            render_width: 1280,
            left: &LEFT,
            right: &RIGHT,
            spectrum_left: &SPECTRUM,
            spectrum_right: &SPECTRUM,
            bass: 0.5,
            mid: 0.25,
            treble: 0.125,
            alpha: 0.8,
            modulate_alpha_by_volume: false,
            modwave_alpha_start: 0.75,
            modwave_alpha_end: 0.95,
            blending: false,
        }
    }

    #[test]
    fn supports_every_beatdrop_extended_mode_with_finite_bounded_geometry() {
        for mode in FIRST_EXTENDED_WAVE_MODE..=LAST_EXTENDED_WAVE_MODE {
            let geometry = build_extended_waveform(input(mode)).expect("extended mode");
            assert!(!geometry.strips.is_empty(), "mode {mode}");
            assert!(geometry.alpha.is_finite(), "mode {mode}");
            for strip in &geometry.strips {
                assert!(strip.len() <= MAX_EXTENDED_WAVE_POINTS, "mode {mode}");
                assert!(
                    strip.iter().flatten().all(|value| value.is_finite()),
                    "mode {mode}"
                );
            }
        }
    }

    #[test]
    fn mode_8_uses_two_spectrum_bins_per_vertex() {
        let mut low = input(8);
        static LOW: [f32; 512] = [0.25; 512];
        static HIGH: [f32; 512] = [4.0; 512];
        low.spectrum_left = &LOW;
        low.spectrum_right = &LOW;
        let first_low = build_extended_waveform(low).unwrap().strips[0][0];
        low.spectrum_left = &HIGH;
        low.spectrum_right = &HIGH;
        let first_high = build_extended_waveform(low).unwrap().strips[0][0];
        assert_ne!(first_low, first_high);
    }

    #[test]
    fn mode_10_and_11_preserve_their_two_disjoint_source_strips() {
        for mode in [10, 11] {
            let geometry = build_extended_waveform(input(mode)).unwrap();
            assert_eq!(geometry.strips.len(), 2, "mode {mode}");
            assert_eq!(geometry.strips[0].len(), geometry.strips[1].len());
        }
    }

    #[test]
    fn mode_17_is_point_recommended_and_preserves_source_alpha_mutation() {
        let geometry = build_extended_waveform(input(17)).unwrap();
        assert!(geometry.points_recommended);
        assert!(
            geometry.alpha < 0.8,
            "source alpha is faded inside its particle loop"
        );
    }

    #[test]
    fn singular_and_unsupported_inputs_are_safe() {
        let mut triangle = input(16);
        triangle.time = 0.0;
        triangle.wave_param = 0.0;
        assert!(build_extended_waveform(triangle)
            .unwrap()
            .strips
            .iter()
            .flatten()
            .flatten()
            .all(|v| v.is_finite()));
        assert!(build_extended_waveform(input(7)).is_none());
        let mut malformed = input(15);
        malformed.time = f32::NAN;
        assert!(build_extended_waveform(malformed)
            .unwrap()
            .strips
            .is_empty());
    }
}
