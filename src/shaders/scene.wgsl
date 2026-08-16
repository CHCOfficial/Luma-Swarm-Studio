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

struct Instance {
    position_brightness: vec4<f32>,
    color: vec4<f32>,
    orientation: vec4<f32>,
    misc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(2) var<storage, read> processed_instances: array<Instance>;

fn rotate_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

struct BodyIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) material: f32,
    @builtin(instance_index) instance_index: u32,
};

struct BodyOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) material: f32,
};

@vertex
fn body_vertex(input: BodyIn) -> BodyOut {
    let instance = processed_instances[input.instance_index];
    var local = input.position * globals.fleet_lod.x;
    // A tiny motion on the propeller material keeps close shots feeling alive.
    if (input.material > 0.5) {
        let rotor = instance.misc.x;
        let radius = length(local.xz);
        local.y += sin(rotor + radius * 21.0) * 0.006;
    }
    let world = rotate_quat(local, instance.orientation) + instance.position_brightness.xyz;
    var out: BodyOut;
    out.clip_position = globals.view_projection * vec4<f32>(world, 1.0);
    out.world_position = world;
    out.normal = normalize(rotate_quat(input.normal, instance.orientation));
    out.color = instance.color.rgb;
    out.material = input.material;
    return out;
}

@fragment
fn body_fragment(input: BodyOut) -> @location(0) vec4<f32> {
    let key = normalize(vec3<f32>(-0.45, 0.78, 0.32));
    let view = normalize(globals.camera_position.xyz - input.world_position);
    let diffuse = max(dot(input.normal, key), 0.0);
    let rim = pow(1.0 - max(dot(input.normal, view), 0.0), 2.5);
    let carbon = mix(vec3<f32>(0.014, 0.02, 0.028), vec3<f32>(0.065, 0.08, 0.1), diffuse);
    let reflected_light = input.color * (0.04 + rim * 0.08);
    let rotor_alpha = mix(1.0, 0.38, step(0.5, input.material));
    return vec4<f32>(carbon + reflected_light + rim * 0.035, rotor_alpha);
}

struct LightOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) intensity: f32,
};

@vertex
fn light_vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> LightOut {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let instance = processed_instances[index];
    let distance_to_camera = length(instance.position_brightness.xyz - globals.camera_position.xyz);
    let rgbw_image = globals.gpu_image.z > 0.5;
    let emitter = select(0u, vertex / 6u, rgbw_image);
    let corner = corners[vertex % 6u];
    var size = (0.38 + distance_to_camera * 0.012) * globals.viewport.z * globals.fleet_lod.y;
    var emitter_offset = vec2<f32>(0.0);
    var emitter_color = instance.color.rgb;
    if (rgbw_image) {
        let offsets = array<vec2<f32>, 4>(
            vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
            vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0)
        );
        emitter_offset = offsets[emitter] * size * 0.22;
        size *= 0.78;
        let white = min(instance.color.r, min(instance.color.g, instance.color.b));
        let residual = max(instance.color.rgb - vec3<f32>(white), vec3<f32>(0.0));
        if (emitter == 0u) { emitter_color = vec3<f32>(residual.r, 0.0, 0.0); }
        else if (emitter == 1u) { emitter_color = vec3<f32>(0.0, residual.g, 0.0); }
        else if (emitter == 2u) { emitter_color = vec3<f32>(0.0, 0.0, residual.b); }
        else { emitter_color = vec3<f32>(white); }
    }
    let world = instance.position_brightness.xyz + vec3<f32>(0.0, 0.20 * globals.fleet_lod.x, 0.0)
        + globals.camera_right.xyz * emitter_offset.x
        + globals.camera_up.xyz * emitter_offset.y
        + globals.camera_right.xyz * corner.x * size
        + globals.camera_up.xyz * corner.y * size;
    var out: LightOut;
    out.clip_position = globals.view_projection * vec4<f32>(world, 1.0);
    out.uv = corner;
    out.color = emitter_color;
    out.intensity = instance.position_brightness.w * select(1.0, 3.4, rgbw_image);
    return out;
}

@fragment
fn light_fragment(input: LightOut) -> @location(0) vec4<f32> {
    let radius = length(input.uv);
    if (radius > 1.0) { discard; }
    let core = exp(-radius * radius * 34.0);
    let halo = exp(-radius * radius * 4.8) * 0.48;
    let diffraction = pow(max(0.0, 1.0 - min(abs(input.uv.x), abs(input.uv.y)) * 13.0), 5.0)
        * pow(max(0.0, 1.0 - radius), 2.0) * 0.34;
    let energy = (core * 8.0 + halo * 2.8 + diffraction) * input.intensity * globals.fleet_lod.z;
    return vec4<f32>(input.color * energy, 0.0);
}

struct EnvironmentOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn environment_vertex(@builtin(vertex_index) vertex: u32) -> EnvironmentOut {
    let positions = array<vec3<f32>, 6>(
        vec3<f32>(-90.0, 0.0, -90.0), vec3<f32>(90.0, 0.0, -90.0), vec3<f32>(90.0, 0.0, 90.0),
        vec3<f32>(-90.0, 0.0, -90.0), vec3<f32>(90.0, 0.0, 90.0), vec3<f32>(-90.0, 0.0, 90.0)
    );
    var out: EnvironmentOut;
    out.world_position = positions[vertex];
    out.clip_position = globals.view_projection * vec4<f32>(out.world_position, 1.0);
    return out;
}

@fragment
fn environment_fragment(input: EnvironmentOut) -> @location(0) vec4<f32> {
    let p = input.world_position.xz;
    let fine_grid = min(abs(fract(p.x / 4.0 + 0.5) - 0.5), abs(fract(p.y / 4.0 + 0.5) - 0.5));
    let grid = 1.0 - smoothstep(0.015, 0.055, fine_grid);
    let ring_distance = abs(length(p) - 31.0);
    let stadium_ring = 1.0 - smoothstep(0.0, 0.32, ring_distance);
    let launch_field = 1.0 - smoothstep(24.0, 54.0, length(p));
    let base = mix(vec3<f32>(0.004, 0.008, 0.012), vec3<f32>(0.012, 0.022, 0.026), launch_field);
    let detail = vec3<f32>(0.01, 0.05, 0.065) * grid * 0.28 + vec3<f32>(0.03, 0.18, 0.21) * stadium_ring;
    return vec4<f32>(base + detail * globals.viewport.w * globals.environment.z, 1.0);
}
