struct Globals {
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    time_exposure_bloom_haze: vec4<f32>,
    viewport: vec4<f32>,
    environment: vec4<f32>,
    fleet_lod: vec4<f32>,
    gpu_show: vec4<f32>,
    gpu_meta: vec4<f32>,
    gpu_safety: vec4<f32>,
    gpu_image: vec4<f32>,
    safety_options: vec4<f32>,
    color_controls: vec4<f32>,
};

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> globals: Globals;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
    );
    let p = positions[index];
    var out: VertexOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
    return out;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fragment_main(input: VertexOut) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let scene = textureSample(hdr_texture, hdr_sampler, uv);
    let texel = vec2<f32>(1.0 / max(globals.viewport.x, 1.0), 1.0 / max(globals.viewport.y, 1.0));
    var blur = vec3<f32>(0.0);
    let radius = 5.0;
    blur += textureSample(hdr_texture, hdr_sampler, uv + texel * vec2<f32>( radius, 0.0)).rgb;
    blur += textureSample(hdr_texture, hdr_sampler, uv + texel * vec2<f32>(-radius, 0.0)).rgb;
    blur += textureSample(hdr_texture, hdr_sampler, uv + texel * vec2<f32>(0.0,  radius)).rgb;
    blur += textureSample(hdr_texture, hdr_sampler, uv + texel * vec2<f32>(0.0, -radius)).rgb;
    blur += textureSample(hdr_texture, hdr_sampler, uv + texel * vec2<f32>( radius,  radius)).rgb;
    blur += textureSample(hdr_texture, hdr_sampler, uv + texel * vec2<f32>(-radius,  radius)).rgb;
    blur += textureSample(hdr_texture, hdr_sampler, uv + texel * vec2<f32>( radius, -radius)).rgb;
    blur += textureSample(hdr_texture, hdr_sampler, uv + texel * vec2<f32>(-radius, -radius)).rgb;
    blur *= 0.125;
    let bloom = max(blur - vec3<f32>(0.55), vec3<f32>(0.0)) * globals.time_exposure_bloom_haze.z;

    let sky_t = clamp(uv.y, 0.0, 1.0);
    var sky = mix(vec3<f32>(0.006, 0.012, 0.026), vec3<f32>(0.0015, 0.0025, 0.008), sky_t);
    let star_cell = floor(uv * vec2<f32>(520.0, 280.0));
    let star = step(0.9965, hash(star_cell)) * smoothstep(0.25, 0.02, uv.y);
    sky += vec3<f32>(0.42, 0.62, 0.95) * star * globals.environment.w;
    let cloud = pow(max(0.0, sin(uv.x * 9.0 + globals.time_exposure_bloom_haze.x * 0.008) * sin(uv.y * 11.0)), 5.0);
    sky += vec3<f32>(0.018, 0.024, 0.04) * cloud * globals.time_exposure_bloom_haze.w * globals.environment.y;

    var color = sky * (1.0 - scene.a) + scene.rgb + bloom;
    let haze = globals.time_exposure_bloom_haze.w * smoothstep(0.74, 0.42, uv.y) * smoothstep(0.18, 0.48, uv.y);
    color = mix(color, vec3<f32>(0.025, 0.075, 0.09), haze * 0.42);
    color *= globals.time_exposure_bloom_haze.y;
    // Map highlight energy as a single scalar, then restore channel ratios.
    // Per-channel ACES drove bright RGB drone lights toward white at the
    // requested 2.20 EV exposure and 2.0 bloom settings.
    let peak = max(max(color.r, color.g), max(color.b, 0.0001));
    color = color / peak * aces(vec3<f32>(peak)).r;
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = max(
        vec3<f32>(0.0),
        vec3<f32>(luminance) + (color - vec3<f32>(luminance)) * globals.color_controls.x
    );
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let vignette = 1.0 - dot(uv - 0.5, uv - 0.5) * 0.62;
    color *= vignette;
    return vec4<f32>(color, 1.0);
}
