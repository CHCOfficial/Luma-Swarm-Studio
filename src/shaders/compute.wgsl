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

struct FormationSample {
    position: vec3<f32>,
    color: vec3<f32>,
    brightness: f32,
};

struct SpatialNode {
    cell: vec3<i32>,
    next: i32,
};

struct SafetyCounters {
    warning_pairs: atomic<u32>,
    collision_pairs: atomic<u32>,
    ground_breaches: atomic<u32>,
    minimum_distance_bits: atomic<u32>,
    collision_a: atomic<u32>,
    collision_b: atomic<u32>,
    collision_x_bits: atomic<u32>,
    collision_y_bits: atomic<u32>,
    collision_z_bits: atomic<u32>,
    collision_distance_bits: atomic<u32>,
    ground_drone: atomic<u32>,
    ground_x_bits: atomic<u32>,
    ground_y_bits: atomic<u32>,
    ground_z_bits: atomic<u32>,
    show_time_bits: atomic<u32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> raw_instances: array<Instance>;
@group(0) @binding(2) var<storage, read_write> processed_instances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> spatial_nodes: array<SpatialNode>;
@group(0) @binding(4) var<storage, read_write> bucket_heads: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> safety_counters: SafetyCounters;
@group(0) @binding(6) var<storage, read_write> corrected_instances: array<Instance>;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const GOLDEN: f32 = 2.39996322973;

fn hash01(value: u32) -> f32 {
    var x = value;
    x = x ^ (x >> 16u);
    x = x * 2146121005u;
    x = x ^ (x >> 15u);
    x = x * 2221713035u;
    x = x ^ (x >> 16u);
    return f32(x) / 4294967295.0;
}

fn palette(index: u32) -> vec3<f32> {
    if (index % 3u == 0u) { return vec3<f32>(1.0, 0.34, 0.055); }
    if (index % 3u == 1u) { return vec3<f32>(1.0, 0.035, 0.46); }
    return vec3<f32>(0.055, 0.68, 1.0);
}

fn rotate_y(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

fn rotate_x(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

fn rotate_z(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
}

fn launch_grid(index: u32, count: u32) -> FormationSample {
    let columns = max(1.0, ceil(sqrt(f32(count))));
    let rows = ceil(f32(count) / columns);
    // A real metre-scale launch lattice; never compress it as count rises.
    let spacing = 1.0;
    let column = f32(index) % columns;
    let row = floor(f32(index) / columns);
    let p = vec3<f32>((column - (columns - 1.0) * 0.5) * spacing, 0.34,
                      (row - (rows - 1.0) * 0.5) * spacing);
    return FormationSample(p, vec3<f32>(0.04, 0.12, 0.2), 0.15);
}

fn sphere_point(index: u32, count: u32) -> vec3<f32> {
    let y = 1.0 - 2.0 * (f32(index) + 0.5) / max(f32(count), 1.0);
    let r = sqrt(max(0.0, 1.0 - y * y));
    let theta = GOLDEN * f32(index);
    return vec3<f32>(cos(theta) * r, y, sin(theta) * r);
}

const GEO_VERTICES: array<vec2<f32>, 138> = array<vec2<f32>, 138>(
    vec2<f32>(-168.0, 66.0), vec2<f32>(-154.0, 72.0), vec2<f32>(-140.0, 70.0), vec2<f32>(-130.0, 60.0),
    vec2<f32>(-124.0, 50.0), vec2<f32>(-117.0, 32.0), vec2<f32>(-105.0, 22.0), vec2<f32>(-97.0, 16.0),
    vec2<f32>(-87.0, 14.0), vec2<f32>(-82.0, 22.0), vec2<f32>(-80.0, 30.0), vec2<f32>(-74.0, 40.0),
    vec2<f32>(-60.0, 47.0), vec2<f32>(-55.0, 55.0), vec2<f32>(-65.0, 64.0), vec2<f32>(-82.0, 70.0),
    vec2<f32>(-100.0, 78.0), vec2<f32>(-130.0, 72.0),
    vec2<f32>(-81.0, 12.0), vec2<f32>(-70.0, 10.0), vec2<f32>(-60.0, 7.0), vec2<f32>(-50.0, 1.0),
    vec2<f32>(-35.0, -8.0), vec2<f32>(-40.0, -20.0), vec2<f32>(-52.0, -34.0), vec2<f32>(-60.0, -52.0),
    vec2<f32>(-70.0, -55.0), vec2<f32>(-76.0, -35.0), vec2<f32>(-80.0, -15.0), vec2<f32>(-75.0, -2.0),
    vec2<f32>(-17.0, 35.0), vec2<f32>(0.0, 37.0), vec2<f32>(15.0, 33.0), vec2<f32>(32.0, 31.0),
    vec2<f32>(42.0, 12.0), vec2<f32>(51.0, 3.0), vec2<f32>(42.0, -12.0), vec2<f32>(35.0, -25.0),
    vec2<f32>(20.0, -35.0), vec2<f32>(8.0, -34.0), vec2<f32>(-2.0, -20.0), vec2<f32>(-10.0, -5.0),
    vec2<f32>(-17.0, 12.0),
    vec2<f32>(-10.0, 36.0), vec2<f32>(-10.0, 60.0), vec2<f32>(5.0, 70.0), vec2<f32>(25.0, 72.0),
    vec2<f32>(40.0, 62.0), vec2<f32>(45.0, 50.0), vec2<f32>(30.0, 42.0), vec2<f32>(22.0, 36.0),
    vec2<f32>(12.0, 44.0), vec2<f32>(0.0, 43.0),
    vec2<f32>(26.0, 36.0), vec2<f32>(40.0, 60.0), vec2<f32>(60.0, 74.0), vec2<f32>(95.0, 78.0),
    vec2<f32>(130.0, 72.0), vec2<f32>(160.0, 65.0), vec2<f32>(178.0, 55.0), vec2<f32>(160.0, 45.0),
    vec2<f32>(140.0, 40.0), vec2<f32>(125.0, 20.0), vec2<f32>(108.0, 5.0), vec2<f32>(98.0, 20.0),
    vec2<f32>(82.0, 8.0), vec2<f32>(72.0, 20.0), vec2<f32>(60.0, 28.0), vec2<f32>(48.0, 30.0),
    vec2<f32>(40.0, 40.0), vec2<f32>(32.0, 44.0),
    vec2<f32>(35.0, 30.0), vec2<f32>(50.0, 30.0), vec2<f32>(58.0, 20.0), vec2<f32>(50.0, 12.0),
    vec2<f32>(42.0, 15.0),
    vec2<f32>(68.0, 28.0), vec2<f32>(88.0, 28.0), vec2<f32>(82.0, 8.0), vec2<f32>(76.0, 6.0),
    vec2<f32>(90.0, 28.0), vec2<f32>(110.0, 22.0), vec2<f32>(125.0, 10.0), vec2<f32>(120.0, -5.0),
    vec2<f32>(105.0, 0.0), vec2<f32>(98.0, 15.0),
    vec2<f32>(112.0, -11.0), vec2<f32>(130.0, -10.0), vec2<f32>(145.0, -17.0), vec2<f32>(154.0, -28.0),
    vec2<f32>(145.0, -40.0), vec2<f32>(120.0, -35.0), vec2<f32>(112.0, -22.0),
    vec2<f32>(-55.0, 60.0), vec2<f32>(-42.0, 59.0), vec2<f32>(-25.0, 70.0), vec2<f32>(-20.0, 82.0),
    vec2<f32>(-42.0, 84.0), vec2<f32>(-60.0, 76.0), vec2<f32>(-65.0, 68.0),
    vec2<f32>(47.0, -12.0), vec2<f32>(51.0, -16.0), vec2<f32>(48.0, -27.0), vec2<f32>(44.0, -23.0),
    vec2<f32>(130.0, 31.0), vec2<f32>(136.0, 34.0), vec2<f32>(142.0, 45.0), vec2<f32>(146.0, 44.0),
    vec2<f32>(141.0, 34.0),
    vec2<f32>(-8.0, 50.0), vec2<f32>(-2.0, 50.0), vec2<f32>(1.0, 59.0), vec2<f32>(-6.0, 58.0),
    vec2<f32>(166.0, -34.0), vec2<f32>(174.0, -40.0), vec2<f32>(178.0, -47.0), vec2<f32>(169.0, -45.0),
    vec2<f32>(95.0, 5.0), vec2<f32>(113.0, 4.0), vec2<f32>(112.0, -8.0), vec2<f32>(98.0, -7.0),
    vec2<f32>(113.0, 2.0), vec2<f32>(135.0, 0.0), vec2<f32>(141.0, -9.0), vec2<f32>(118.0, -10.0),
    vec2<f32>(-96.0, 64.0), vec2<f32>(-82.0, 65.0), vec2<f32>(-76.0, 58.0), vec2<f32>(-82.0, 51.0),
    vec2<f32>(-94.0, 54.0), vec2<f32>(-99.0, 60.0),
    vec2<f32>(-6.0, 36.0), vec2<f32>(0.0, 43.0), vec2<f32>(15.0, 45.0), vec2<f32>(30.0, 41.0),
    vec2<f32>(36.0, 34.0), vec2<f32>(25.0, 30.0), vec2<f32>(10.0, 30.0)
);

fn geo_polygon_contains(lat: f32, lon: f32, start: u32, vertex_count: u32) -> bool {
    var inside = false;
    var previous = start + vertex_count - 1u;
    for (var offset = 0u; offset < vertex_count; offset++) {
        let current = start + offset;
        let a = GEO_VERTICES[current];
        let b = GEO_VERTICES[previous];
        if ((a.y > lat) != (b.y > lat)
            && lon < (b.x - a.x) * (lat - a.y) / (b.y - a.y) + a.x) {
            inside = !inside;
        }
        previous = current;
    }
    return inside;
}

fn continent_mask(lat: f32, lon: f32) -> bool {
    let land = geo_polygon_contains(lat, lon, 0u, 18u)
        || geo_polygon_contains(lat, lon, 18u, 12u)
        || geo_polygon_contains(lat, lon, 30u, 13u)
        || geo_polygon_contains(lat, lon, 43u, 10u)
        || geo_polygon_contains(lat, lon, 53u, 18u)
        || geo_polygon_contains(lat, lon, 71u, 5u)
        || geo_polygon_contains(lat, lon, 76u, 4u)
        || geo_polygon_contains(lat, lon, 80u, 6u)
        || geo_polygon_contains(lat, lon, 86u, 7u)
        || geo_polygon_contains(lat, lon, 93u, 7u)
        || geo_polygon_contains(lat, lon, 100u, 4u)
        || geo_polygon_contains(lat, lon, 104u, 5u)
        || geo_polygon_contains(lat, lon, 109u, 4u)
        || geo_polygon_contains(lat, lon, 113u, 4u)
        || geo_polygon_contains(lat, lon, 117u, 4u)
        || geo_polygon_contains(lat, lon, 121u, 4u);
    let antarctica_coast = -70.5 - sin(lon * 0.11) * 3.2 - cos(lon * 0.037) * 1.6;
    let inland_water = geo_polygon_contains(lat, lon, 125u, 6u)
        || geo_polygon_contains(lat, lon, 131u, 7u);
    return (land || lat < antarctica_coast) && !inland_water;
}

fn stellar_chrysalis(index: u32, count: u32, time: f32) -> FormationSample {
    let ribbon_end = count * 68u / 100u;
    let core_end = count * 86u / 100u;
    if (index < ribbon_end) {
        let ribbon = index % 12u;
        let local = index / 12u;
        let per_ribbon = max(1u, (ribbon_end + 11u) / 12u);
        let cross_count = max(3u, u32(ceil(sqrt(f32(per_ribbon) / 10.0))));
        let along_count = max(1u, (per_ribbon + cross_count - 1u) / cross_count);
        let u = f32(local / cross_count) / f32(max(1u, along_count - 1u));
        let across = f32(local % cross_count) / f32(max(1u, cross_count - 1u)) - 0.5;
        let latitude = (u - 0.5) * PI;
        let petal_phase = f32(ribbon) * TAU / 12.0;
        let breathing = 1.0 + 0.12 * sin(time * 0.82 + petal_phase);
        let radius = 0.8 + pow(abs(cos(latitude)), 0.62)
            * (6.1 + 1.15 * sin(petal_phase * 2.0 - time * 0.7)) * breathing;
        let longitude = petal_phase + sin(latitude) * 0.72 + time * 0.12 + across * 0.17;
        let p = vec3<f32>(cos(longitude) * (radius + across * 0.45),
            sin(latitude) * 10.2, sin(longitude) * (radius + across * 0.45));
        let sweep = 0.5 + 0.5 * sin(u * 13.0 - time * 2.2 + petal_phase);
        var color = mix(vec3<f32>(1.0, 0.04, 0.42), vec3<f32>(0.05, 0.65, 1.0), u);
        color = mix(color, vec3<f32>(1.0, 0.38, 0.04), sweep * 0.38);
        return FormationSample(p, color, 1.0 + sweep * 0.38);
    }
    if (index < core_end) {
        let local = index - ribbon_end;
        let core_count = max(1u, core_end - ribbon_end);
        let sphere = rotate_y(sphere_point(local, core_count), time * 0.52);
        let pulse = 3.2 + 0.42 * sin(time * 1.9);
        let radial = sqrt(max(0.0, 1.0 - sphere.y * sphere.y));
        return FormationSample(sphere * pulse,
            mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(1.0, 0.92, 0.52), radial),
            1.42 + 0.3 * max(0.0, sin(atan2(sphere.z, sphere.x) * 2.0 - time)));
    }
    let local = index - core_end;
    let orbit_count = max(1u, count - core_end);
    let orbit = local % 4u;
    let orbit_local = local / 4u;
    let per_orbit = max(1u, (orbit_count + 3u) / 4u);
    let angle = TAU * f32(orbit_local) / f32(per_orbit)
        + time * (0.16 + f32(orbit) * 0.035);
    let radius = 10.6 + f32(orbit) * 1.15;
    let x_flat = cos(angle) * radius;
    let z_flat = sin(angle) * radius;
    let x_angle = 0.38 + f32(orbit) * 0.15;
    let y_tilt = -z_flat * sin(x_angle);
    let z_tilt = z_flat * cos(x_angle);
    let z_angle = (f32(orbit) - 1.5) * 0.3;
    let p = vec3<f32>(x_flat * cos(z_angle) - y_tilt * sin(z_angle),
        x_flat * sin(z_angle) + y_tilt * cos(z_angle), z_tilt);
    var color = vec3<f32>(0.05, 0.65, 1.0);
    if (orbit == 1u) { color = vec3<f32>(1.0, 0.04, 0.42); }
    if (orbit == 2u) { color = vec3<f32>(1.0, 0.38, 0.04); }
    return FormationSample(p, color, 1.04 + 0.3 * max(0.0, sin(angle * 6.0 - time * 1.4)));
}

fn heart(index: u32, count: u32, time: f32) -> FormationSample {
    let cross_count = clamp(round(sqrt(max(f32(count), 1.0) / 10.0)), 6.0, 48.0);
    let cross_index = f32(index) % cross_count;
    let along_index = floor(f32(index) / cross_count);
    let along_count = ceil(f32(count) / cross_count);
    let u = (along_index + 0.5) / along_count;
    let t = TAU * u;
    let beat = 1.0 + 0.045 * max(-0.3, sin(time * 2.6));
    let layer = (cross_index + 0.5) / cross_count;
    let radial_scale = 0.76 + layer * 0.24;
    let cusp_distance = min(min(u, 1.0 - u), abs(u - 0.5));
    var cusp_side = 0.08;
    if (u < 0.5) { cusp_side = -0.08; }
    let cusp_separation = (1.0 - smoothstep(0.0, 0.06, cusp_distance)) * cusp_side;
    let x = 16.0 * pow(sin(t), 3.0) * 0.52 * radial_scale;
    let y = (13.0 * cos(t) - 5.0 * cos(2.0 * t) - 2.0 * cos(3.0 * t) - cos(4.0 * t)) * 0.52;
    let p = vec3<f32>(x, y * radial_scale, (layer - 0.5) * 1.2 + cusp_separation) * beat;
    return FormationSample(p, mix(vec3<f32>(1.0, 0.03, 0.42), vec3<f32>(1.0, 0.28, 0.035), 0.5 + 0.5 * sin(t + time)), 1.12);
}

fn galaxy(index: u32, count: u32, time: f32) -> FormationSample {
    let arm = index % 4u;
    let local = index / 4u;
    let local_count = max(1u, (count + 3u) / 4u);
    let cross_count = max(4u, u32(ceil(sqrt(f32(local_count) / 7.5))));
    let cross_index = local % cross_count;
    let along_index = local / cross_count;
    let along_count = max(1u, (local_count + cross_count - 1u) / cross_count);
    let u = (f32(along_index) + 0.5) / f32(along_count);
    let across = (f32(cross_index) + 0.5) / f32(cross_count) - 0.5;
    let radius = 0.9 + sqrt(u) * 11.1 + across * 0.72;
    let angle = f32(arm) * TAU / 4.0 + radius * 0.68 + time * 0.12 + across * 0.08;
    let flat = vec3<f32>(cos(angle) * radius, across * 2.2, sin(angle) * radius);
    let p = vec3<f32>(flat.x, flat.y * cos(0.72) - flat.z * sin(0.72), flat.y * sin(0.72) + flat.z * cos(0.72));
    return FormationSample(p, mix(vec3<f32>(0.04, 0.7, 1.0), vec3<f32>(0.68, 0.12, 1.0), u), 1.0);
}

fn phoenix(index: u32, count: u32, time: f32) -> FormationSample {
    let region = index % 20u;
    let local = index / 20u;
    let local_count = max(1u, (count + 19u) / 20u);
    let u = (f32(local) + 0.5) / f32(local_count);
    let v = fract((f32(local) + f32(region) * 0.37) * 0.61803398875);
    var p: vec3<f32>;
    if (region < 13u) {
        var side = 1.0;
        if (index % 2u == 0u) { side = -1.0; }
        p = vec3<f32>(side * (1.0 + u * 11.5),
            1.2 + sin(u * PI) * 3.6 + sin(time * 1.72) * pow(u, 1.35) * 2.7 + (v - 0.5) * (0.9 + u * 3.4),
            (v - 0.5) * (1.0 + u * 4.0) + sin(u * 10.0 + time) * 0.28);
    } else if (region < 17u) {
        let theta = TAU * v;
        let taper = sqrt(max(0.0, 1.0 - abs(u * 2.0 - 1.0)));
        p = vec3<f32>(cos(theta) * 1.3 * taper, 1.4 + sin(theta) * 1.2 * taper, (u - 0.5) * 8.0);
    } else {
        let plume = f32(region - 17u);
        let curl = u * 4.8 + time * 0.55 + plume * 1.9;
        p = vec3<f32>((plume - 1.0) * 1.7 + sin(curl) * u * 1.4, -0.4 - u * 8.0 + cos(curl) * 0.55, 3.2 + u * 5.5);
    }
    return FormationSample(p, mix(palette(index), vec3<f32>(0.05, 0.7, 1.0), v * 0.35), 1.08);
}

fn prism_cathedral(index: u32, count: u32, time: f32) -> FormationSample {
    let arch_end = count * 72u / 100u;
    let rose_end = count * 88u / 100u;
    if (index < arch_end) {
        let arch = index % 8u;
        let local = index / 8u;
        let per_arch = max(1u, (arch_end + 7u) / 8u);
        let cross_count = max(3u, u32(ceil(sqrt(f32(per_arch) / 11.0))));
        let along_count = max(1u, (per_arch + cross_count - 1u) / cross_count);
        let u = f32(local / cross_count) / f32(max(1u, along_count - 1u));
        let across = f32(local % cross_count) / f32(max(1u, cross_count - 1u)) - 0.5;
        let arch_angle = f32(arch) * TAU / 8.0 + time * 0.055;
        let local_x = 2.3 + u * 6.1;
        let vault = pow(max(0.0, sin(PI * u)), 0.72);
        let local_point = vec3<f32>(local_x,
            -6.0 + vault * (13.5 + 0.8 * sin(time * 0.7 + arch_angle)),
            across * (1.25 + vault * 0.8));
        let p = rotate_y(local_point, arch_angle);
        let sweep = 0.5 + 0.5 * sin(u * 12.0 - time * 2.4 + arch_angle);
        var color = mix(vec3<f32>(0.05, 0.65, 1.0), vec3<f32>(1.0, 0.04, 0.42), f32(arch) / 7.0);
        color = mix(color, vec3<f32>(1.0, 0.38, 0.04), sweep * 0.45);
        return FormationSample(p, color, 1.02 + sweep * 0.34);
    }
    if (index < rose_end) {
        let local = index - arch_end;
        let rose_count = max(1u, rose_end - arch_end);
        let ring = local % 5u;
        let serial = local / 5u;
        let per_ring = max(1u, (rose_count + 4u) / 5u);
        let tube_cross_count = 3u;
        let cross = serial % tube_cross_count;
        let ring_local = serial / tube_cross_count;
        let around_count = max(1u, (per_ring + tube_cross_count - 1u) / tube_cross_count);
        let angle = TAU * f32(ring_local) / f32(around_count) + time * (0.28 - f32(ring) * 0.025);
        let radius = 1.4 + f32(ring) * 1.12
            + (f32(cross) - 1.0) * 0.34 / max(globals.gpu_meta.w, 1.0);
        let face = vec3<f32>(cos(angle) * radius, 2.4 + sin(angle) * radius, 0.0);
        return FormationSample(rotate_y(face, time * 0.11),
            mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(1.0, 0.04, 0.42), f32(ring) / 4.0),
            1.18 + 0.3 * max(0.0, sin(angle * 7.0 - time)));
    }
    let local = index - rose_end;
    let spire_count = max(1u, count - rose_end);
    let spire = local % 8u;
    let along = local / 8u;
    let per_spire = max(1u, (spire_count + 7u) / 8u);
    let u = (f32(along) + 0.5) / f32(per_spire);
    let base_angle = f32(spire) * TAU / 8.0 + time * 0.055;
    let twist = base_angle + u * 1.1 + time * 0.2;
    let radius = 9.1 - u * 1.8;
    return FormationSample(vec3<f32>(cos(twist) * radius,
        -6.0 + u * (15.0 + 2.0 * f32(spire % 2u)), sin(twist) * radius),
        mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(1.0, 0.94, 0.5), u),
        1.2 + 0.36 * max(0.0, sin(u * 15.0 - time * 3.0)));
}

fn helix(index: u32, count: u32, time: f32) -> FormationSample {
    let cross_count = max(8u, u32(ceil(sqrt(f32(max(count, 1u)) / 6.0))));
    let cross_index = index % cross_count;
    let along_index = index / cross_count;
    let along_count = max(1u, (count + cross_count - 1u) / cross_count);
    let u = (f32(along_index) + 0.5) / f32(along_count);
    let y = (u - 0.5) * 17.0;
    let theta = u * TAU * 3.1 + time * 0.64;
    let strand = cross_index % 2u;
    let tube_index = cross_index / 2u;
    let tube_count = max(1u, (cross_count + 1u) / 2u);
    let radius = 4.4 + sin(y * 0.42 + time) * 0.35;
    var across = 1.0;
    if (strand == 0u) { across = -1.0; }
    let a = vec3<f32>(cos(theta) * radius, y, sin(theta) * radius);
    let b = vec3<f32>(-cos(theta) * radius, y, -sin(theta) * radius);
    var p = mix(a, b, (across + 1.0) * 0.5);
    let tube_angle = TAU * (f32(tube_index) + 0.5 * f32(along_index % 2u)) / f32(tube_count);
    p += vec3<f32>(cos(theta), 0.0, sin(theta)) * cos(tube_angle) * 0.7
        + vec3<f32>(-sin(theta), 0.0, cos(theta)) * sin(tube_angle) * 0.7;
    var color = vec3<f32>(0.05, 0.7, 1.0);
    if (across < -0.9) { color = vec3<f32>(1.0, 0.03, 0.46); }
    return FormationSample(p, color, 1.12);
}

fn planet(index: u32, count: u32, time: f32) -> FormationSample {
    let sphere_end = count * 52u / 100u;
    let ring_end = count * 82u / 100u;
    let atmosphere_end = count * 90u / 100u;
    if (index < sphere_end) {
        let local = sphere_point(index, max(1u, sphere_end));
        let latitude_band = 0.5 + 0.5 * sin(local.y * 28.0 + time * 0.32);
        let longitude_storm = 0.5 + 0.5 * cos(atan2(local.z, local.x) * 2.0 - time * 0.7);
        let storm = smoothstep(0.72, 1.0, longitude_storm)
            * (1.0 - smoothstep(0.12, 0.48, abs(local.y + 0.28)));
        var color = mix(vec3<f32>(0.03, 0.16, 0.72), vec3<f32>(0.08, 0.72, 1.0), latitude_band);
        color = mix(color, vec3<f32>(1.0, 0.04, 0.42), storm * 0.68);
        let p = rotate_y(vec3<f32>(local.x * 6.4, local.y * 5.72, local.z * 6.4), time * 0.16);
        return FormationSample(p, color, 1.02 + storm * 0.35);
    }
    if (index < ring_end) {
        let local_index = index - sphere_end;
        let ring_count = max(1u, ring_end - sphere_end);
        let radial_layers = max(5u, u32(ceil(sqrt(f32(ring_count) * 0.055))));
        let radial_index = local_index % radial_layers;
        let angular_index = local_index / radial_layers;
        let angular_count = max(1u, (ring_count + radial_layers - 1u) / radial_layers);
        let band = (f32(radial_index) + 0.5) / f32(radial_layers);
        var radius = 7.55 + band * 4.75;
        if (band > 0.58) { radius += 0.58; }
        let angle = TAU * (f32(angular_index) + 0.5 * f32(radial_index % 2u)) / f32(angular_count)
            + time * (0.08 + band * 0.035);
        let ripple = sin(angle * 7.0 - time * 1.8 + band * 9.0) * 0.075;
        let x_flat = cos(angle) * radius;
        let z_flat = sin(angle) * radius;
        let y_tilt = ripple * cos(0.46) - z_flat * sin(0.46);
        let z_tilt = ripple * sin(0.46) + z_flat * cos(0.46);
        let p = vec3<f32>(
            x_flat * cos(0.13) - y_tilt * sin(0.13),
            x_flat * sin(0.13) + y_tilt * cos(0.13),
            z_tilt
        );
        var color = mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(1.0, 0.04, 0.42), smoothstep(0.12, 0.72, band));
        color = mix(color, vec3<f32>(0.05, 0.65, 1.0), smoothstep(0.72, 1.0, band));
        return FormationSample(p, color, 1.12 + abs(ripple) * 1.8);
    }
    if (index < atmosphere_end) {
        let local_index = index - ring_end;
        let atmosphere_count = max(1u, atmosphere_end - ring_end);
        var local = sphere_point(local_index, atmosphere_count);
        let theta = atan2(local.z, local.x) + time * 0.28;
        let pulse = 0.5 + 0.5 * sin(theta * 3.0 + local.y * 12.0 - time * 1.5);
        local = rotate_y(local, time * 0.28) * (6.7 + pulse * 0.12);
        return FormationSample(local, mix(vec3<f32>(0.05, 0.65, 1.0), vec3<f32>(1.0, 0.04, 0.42), pulse * 0.7), 0.62 + pulse * 0.5);
    }
    let local_index = index - atmosphere_end;
    let moon_count = max(1u, count - atmosphere_end);
    let moon = local_index % 2u;
    let moon_local = local_index / 2u;
    let per_moon = max(1u, (moon_count + 1u) / 2u);
    let sphere = sphere_point(moon_local, per_moon);
    var orbit = time * 0.24;
    var orbit_radius = 13.9;
    var height = 2.6;
    var moon_radius = 1.05;
    var color = vec3<f32>(0.52, 0.78, 1.0);
    if (moon == 1u) {
        orbit = time * -0.17 + 2.7;
        orbit_radius = 15.5;
        height = -2.5;
        moon_radius = 0.72;
        color = vec3<f32>(1.0, 0.38, 0.04);
    }
    let center = vec3<f32>(cos(orbit) * orbit_radius, height, sin(orbit) * orbit_radius);
    return FormationSample(center + sphere * moon_radius, color, 1.15);
}

fn infinity_portal(index: u32, count: u32, time: f32) -> FormationSample {
    let ribbon_end = count * 76u / 100u;
    let halo_end = count * 92u / 100u;
    if (index < ribbon_end) {
        let strand = index % 7u;
        let local = index / 7u;
        let along_count = max(1u, (ribbon_end + 6u) / 7u);
        let u = (f32(local) + 0.5) / f32(along_count);
        let t = TAU * u;
        let denominator = 1.0 + sin(t) * sin(t);
        var p = vec3<f32>(13.8 * cos(t) / denominator,
            10.2 * sin(t) * cos(t) / denominator,
            (f32(strand) - 3.0) * 0.42 / max(globals.gpu_meta.w, 1.0));
        let crossing = 1.0 - smoothstep(0.0, 0.24, abs(cos(t)));
        var lobe_side = 0.38;
        if (sin(t) < 0.0) { lobe_side = -0.38; }
        p.z += crossing * lobe_side;
        p = rotate_y(p, time * 0.08);
        let wave = 0.5 + 0.5 * sin(t * 4.0 - time * 2.1 + f32(strand));
        var color = mix(vec3<f32>(0.05, 0.65, 1.0), vec3<f32>(1.0, 0.04, 0.42), smoothstep(-1.0, 1.0, sin(t)));
        color = mix(color, vec3<f32>(1.0, 0.38, 0.04), wave * 0.42);
        return FormationSample(p, color, 1.02 + wave * 0.34);
    }
    if (index < halo_end) {
        let local = index - ribbon_end;
        let halo_count = max(1u, halo_end - ribbon_end);
        let ring = local % 3u;
        let ring_local = local / 3u;
        let per_ring = max(1u, (halo_count + 2u) / 3u);
        let angle = TAU * (f32(ring_local) + 0.5 * f32(ring)) / f32(per_ring)
            + time * (0.16 + f32(ring) * 0.035);
        let radius = 13.0 + f32(ring) * 1.15;
        let x_flat = cos(angle) * radius;
        let z_flat = sin(angle) * radius;
        let x_tilt = x_flat;
        let y_tilt = -z_flat * sin(0.45 + f32(ring) * 0.16);
        let z_tilt = z_flat * cos(0.45 + f32(ring) * 0.16);
        let z_angle = (f32(ring) - 1.0) * 0.38;
        let p = vec3<f32>(x_tilt * cos(z_angle) - y_tilt * sin(z_angle),
            x_tilt * sin(z_angle) + y_tilt * cos(z_angle), z_tilt);
        var color = vec3<f32>(1.0, 0.04, 0.42);
        if (ring == 1u) { color = vec3<f32>(1.0, 0.38, 0.04); }
        if (ring == 2u) { color = vec3<f32>(0.05, 0.65, 1.0); }
        return FormationSample(p, color, 0.95 + 0.28 * max(0.0, sin(angle * 5.0 - time)));
    }
    let local = index - halo_end;
    let core_count = max(1u, count - halo_end);
    let pulse = 1.55 + 0.32 * sin(time * 2.4);
    let core = rotate_y(sphere_point(local, core_count), time * 0.7) * pulse
        + vec3<f32>(0.0, 0.0, 3.0);
    return FormationSample(rotate_y(core, time * 0.08),
        vec3<f32>(1.0, 0.82, 0.27), 1.55);
}

fn prismatic_lotus(index: u32, count: u32, time: f32) -> FormationSample {
    let petal_end = count * 82u / 100u;
    if (index < petal_end) {
        let petal = index % 12u;
        let local = index / 12u;
        let per_petal = max(1u, (petal_end + 11u) / 12u);
        let cross_count = max(3u, u32(ceil(sqrt(f32(per_petal) / 7.0))));
        let along_count = max(1u, (per_petal + cross_count - 1u) / cross_count);
        let u = f32(local / cross_count) / f32(max(1u, along_count - 1u));
        let across = f32(local % cross_count) / f32(max(1u, cross_count - 1u)) - 0.5;
        let phase = f32(petal) * TAU / 12.0 + time * 0.075;
        let open = 0.82 + 0.16 * sin(time * 0.72 + f32(petal) * 0.43);
        let radius = 1.2 + u * 10.8 * open;
        // A non-zero tip cross-section prevents cross-row slot collapse.
        let width = (0.34 + sin(PI * u) * (1.0 + u * 4.2)) * across;
        let height = -3.2 + sin(PI * u) * (5.1 + 0.7 * f32(petal % 2u)) - u * 1.25
            + sin(u * 8.0 - time * 1.4) * 0.22;
        let p = vec3<f32>(cos(phase) * radius - sin(phase) * width, height,
            sin(phase) * radius + cos(phase) * width);
        let hue = f32(petal) / 12.0;
        var color = mix(vec3<f32>(1.0, 0.04, 0.42), vec3<f32>(1.0, 0.38, 0.04), smoothstep(0.0, 0.48, hue));
        color = mix(color, vec3<f32>(0.05, 0.65, 1.0), smoothstep(0.45, 1.0, hue));
        return FormationSample(p, color, 0.96 + 0.34 * max(0.0, sin(u * 9.0 - time * 2.0)));
    }
    let local = index - petal_end;
    let core_count = max(1u, count - petal_end);
    let filament = local % 9u;
    let serial = local / 9u;
    let per_filament = max(1u, (core_count + 8u) / 9u);
    let tube_cross_count = 3u;
    let cross = serial % tube_cross_count;
    let along = serial / tube_cross_count;
    let along_count = max(1u, (per_filament + tube_cross_count - 1u) / tube_cross_count);
    let u = (f32(along) + 0.5) / f32(along_count);
    let angle = f32(filament) * TAU / 9.0 + time * 0.32 + u * 2.4;
    let radius = 0.45 + u * 2.7;
    let tube_angle = f32(cross) * TAU / f32(tube_cross_count);
    let tube_radius = 0.3 / max(globals.gpu_meta.w, 1.0);
    let tangent = vec3<f32>(-sin(angle), 0.0, cos(angle));
    let tube_offset = tangent * cos(tube_angle) * tube_radius
        + vec3<f32>(0.0, sin(tube_angle) * tube_radius, 0.0);
    return FormationSample(vec3<f32>(cos(angle) * radius, -1.2 + u * 5.2, sin(angle) * radius) + tube_offset,
        mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(1.0, 0.94, 0.48), u),
        1.28 + 0.32 * max(0.0, sin(time * 2.8 + f32(filament))));
}

fn celestial_crown(index: u32, count: u32, time: f32) -> FormationSample {
    let spire_end = count * 68u / 100u;
    let halo_end = count * 89u / 100u;
    if (index < spire_end) {
        let spire = index % 12u;
        let local = index / 12u;
        let per_spire = max(1u, (spire_end + 11u) / 12u);
        let cross_count = max(3u, u32(ceil(sqrt(f32(per_spire) / 9.0))));
        let along_count = max(1u, (per_spire + cross_count - 1u) / cross_count);
        let u = f32(local / cross_count) / f32(max(1u, along_count - 1u));
        let across = f32(local % cross_count) / f32(max(1u, cross_count - 1u)) - 0.5;
        let angle = f32(spire) * TAU / 12.0 + time * 0.095;
        var high = 0.0;
        if (spire % 3u == 0u) { high = 3.0; }
        let radius = 8.6 - sin(PI * u) * 2.4 + across * 1.1;
        let y = -4.5 + u * (10.5 + high) + sin(PI * u) * 2.1;
        let twist = angle + u * 0.42;
        var color = vec3<f32>(0.05, 0.65, 1.0);
        if (spire % 3u == 0u) { color = vec3<f32>(1.0, 0.38, 0.04); }
        else if (spire % 2u == 0u) { color = vec3<f32>(1.0, 0.04, 0.42); }
        return FormationSample(vec3<f32>(cos(twist) * radius, y, sin(twist) * radius), color,
            1.0 + 0.36 * max(0.0, sin(u * 12.0 - time * 2.4)));
    }
    if (index < halo_end) {
        let local = index - spire_end;
        let halo_count = max(1u, halo_end - spire_end);
        let ring = local % 4u;
        let ring_local = local / 4u;
        let per_ring = max(1u, (halo_count + 3u) / 4u);
        let angle = TAU * f32(ring_local) / f32(per_ring) + time * (0.12 + f32(ring) * 0.035);
        let radius = 6.0 + f32(ring) * 1.85;
        let x_flat = cos(angle) * radius;
        let z_flat = sin(angle) * radius;
        let x_angle = 0.2 + f32(ring) * 0.12;
        let y_tilt = -z_flat * sin(x_angle);
        let z_tilt = z_flat * cos(x_angle);
        let z_angle = (f32(ring) - 1.5) * 0.18;
        let p = vec3<f32>(x_flat * cos(z_angle) - y_tilt * sin(z_angle),
            x_flat * sin(z_angle) + y_tilt * cos(z_angle) + 1.5 + f32(ring) * 0.9, z_tilt);
        return FormationSample(p, mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(0.05, 0.65, 1.0), f32(ring) / 3.0), 1.08);
    }
    let local = index - halo_end;
    let core_count = max(1u, count - halo_end);
    let arm = local % 6u;
    let along = local / 6u;
    let per_arm = max(1u, (core_count + 5u) / 6u);
    let u = (f32(along) + 0.5) / f32(per_arm);
    let angle = f32(arm) * TAU / 6.0 + u * 5.4 - time * 0.48;
    let radius = 0.55 + (1.0 - u) * 4.25;
    return FormationSample(vec3<f32>(cos(angle) * radius, -1.0 + u * 9.5, sin(angle) * radius),
        mix(vec3<f32>(1.0, 0.04, 0.42), vec3<f32>(1.0, 0.38, 0.04), u),
        1.18 + 0.42 * max(0.0, sin(u * 15.0 - time * 3.0)));
}

fn event_horizon(index: u32, count: u32, time: f32) -> FormationSample {
    let disk_end = count * 70u / 100u;
    let photon_end = count * 85u / 100u;
    if (index < disk_end) {
        let cross_count = max(4u, u32(ceil(sqrt(f32(max(1u, disk_end)) / 18.0))));
        let along_count = max(1u, (disk_end + cross_count - 1u) / cross_count);
        let along = index / cross_count;
        let across = index % cross_count;
        let u = (f32(along) + 0.5) / f32(along_count);
        let v = (f32(across) + 0.5) / f32(cross_count);
        let radius = 3.6 + v * 8.8;
        let angular_speed = 0.18 + (1.0 - v) * 0.42;
        let angle = u * TAU + time * angular_speed + v * 0.22;
        let wave = sin(angle * 3.0 - time * 1.7 + v * 8.0);
        let flat = vec3<f32>(
            cos(angle) * radius,
            (v - 0.5) * 0.82 + wave * (0.1 + (1.0 - v) * 0.22),
            sin(angle) * radius
        );
        let inner = vec3<f32>(1.0, 0.16, 0.015);
        let outer = mix(vec3<f32>(0.05, 0.65, 1.0), vec3<f32>(1.0, 0.04, 0.42), v * 0.7);
        return FormationSample(rotate_z(flat, 0.16), mix(inner, outer, v),
            1.08 + 0.48 * max(0.0, wave));
    }
    if (index < photon_end) {
        let local = index - disk_end;
        let sphere_count = max(1u, photon_end - disk_end);
        let sphere = rotate_y(sphere_point(local, sphere_count), time * 0.34);
        let radial = sqrt(max(0.0, 1.0 - sphere.y * sphere.y));
        return FormationSample(sphere * 2.15,
            mix(vec3<f32>(1.0, 0.04, 0.42), vec3<f32>(0.05, 0.65, 1.0), radial),
            1.4 + 0.4 * max(0.0, sin(f32(local) * 0.31 - time * 1.8)));
    }
    let local = index - photon_end;
    let jet_count = max(1u, count - photon_end);
    var side = 1.0;
    if (local % 2u == 0u) { side = -1.0; }
    let jet_local = local / 2u;
    let per_side = max(1u, (jet_count + 1u) / 2u);
    let cross_count = max(3u, u32(ceil(sqrt(f32(per_side) / 12.0))));
    let along_count = max(1u, (per_side + cross_count - 1u) / cross_count);
    let u = f32(jet_local / cross_count) / f32(max(1u, along_count - 1u));
    let across = f32(jet_local % cross_count) / f32(max(1u, cross_count - 1u)) - 0.5;
    let angle = time * 1.3 + u * 11.0 + across * PI;
    let radius = 0.62 + across * 0.5 + u * 0.38;
    return FormationSample(
        vec3<f32>(cos(angle) * radius, side * (3.5 + u * 11.5), sin(angle) * radius),
        mix(vec3<f32>(0.05, 0.65, 1.0), vec3<f32>(0.45, 0.16, 1.0), u),
        1.15 + 0.5 * max(0.0, sin(u * 18.0 - time * 4.0))
    );
}

fn spectral_mandala(index: u32, count: u32, time: f32) -> FormationSample {
    let petal_end = count * 72u / 100u;
    let halo_end = count * 90u / 100u;
    if (index < petal_end) {
        let petal = index % 18u;
        let local = index / 18u;
        let per_petal = max(1u, (petal_end + 17u) / 18u);
        let cross_count = max(3u, u32(ceil(sqrt(f32(per_petal) / 9.0))));
        let along_count = max(1u, (per_petal + cross_count - 1u) / cross_count);
        let u = f32(local / cross_count) / f32(max(1u, along_count - 1u));
        let across = f32(local % cross_count) / f32(max(1u, cross_count - 1u)) - 0.5;
        let petal_angle = f32(petal) * TAU / 18.0 + time * 0.075;
        let opening = across * (0.13 + sin(PI * u) * 0.2);
        let angle = petal_angle + opening;
        let radius = 2.85 + u * 9.6;
        let depth = sin(petal_angle * 3.0 - time * 0.42) * sin(PI * u) * 1.18;
        let pulse = sin(u * 16.0 - time * 2.6 + petal_angle);
        var color = mix(vec3<f32>(1.0, 0.04, 0.42), vec3<f32>(1.0, 0.38, 0.04), u);
        color = mix(color, vec3<f32>(0.05, 0.65, 1.0), f32(petal % 3u) * 0.22);
        return FormationSample(vec3<f32>(cos(angle) * radius, sin(angle) * radius, depth),
            color, 1.02 + 0.42 * max(0.0, pulse));
    }
    if (index < halo_end) {
        let local = index - petal_end;
        let halo_count = max(1u, halo_end - petal_end);
        let ring = local % 5u;
        let ring_local = local / 5u;
        let per_ring = max(1u, (halo_count + 4u) / 5u);
        let angle = TAU * (f32(ring_local) + 0.5) / f32(per_ring)
            - time * (0.11 + f32(ring) * 0.025);
        let radius = 3.8 + f32(ring) * 1.7;
        return FormationSample(
            vec3<f32>(cos(angle) * radius, sin(angle) * radius, 3.0 + (f32(ring) - 2.0) * 0.82),
            mix(vec3<f32>(0.05, 0.65, 1.0), vec3<f32>(1.0, 0.38, 0.04), f32(ring) / 4.0),
            1.16
        );
    }
    let local = index - halo_end;
    let core_count = max(1u, count - halo_end);
    let sphere = rotate_y(sphere_point(local, core_count), -time * 0.5);
    let radial = sqrt(max(0.0, 1.0 - sphere.y * sphere.y));
    return FormationSample(sphere * 1.55, vec3<f32>(0.72, 0.26, 1.0) + vec3<f32>(0.34),
        1.45 + 0.3 * max(0.0, sin(time * 2.2 + f32(local) * 0.17 + radial)));
}

fn chrono_gyroscope(index: u32, count: u32, time: f32) -> FormationSample {
    let ring_end = count * 86u / 100u;
    if (index < ring_end) {
        let ring = index % 8u;
        let local = index / 8u;
        let per_ring = max(1u, (ring_end + 7u) / 8u);
        let cross_count = max(4u, u32(ceil(sqrt(f32(per_ring) / 12.0))));
        let along_count = max(1u, (per_ring + cross_count - 1u) / cross_count);
        let along = local / cross_count;
        let across = local % cross_count;
        let u = TAU * (f32(along) + 0.5) / f32(along_count)
            + time * (0.1 + f32(ring) * 0.018);
        let v = TAU * (f32(across) + 0.5) / f32(cross_count);
        let major = 3.2 + f32(ring) * 1.16;
        let minor = 0.3;
        let torus = vec3<f32>((major + minor * cos(v)) * cos(u),
            (major + minor * cos(v)) * sin(u), minor * sin(v));
        let tilted = rotate_y(rotate_x(torus, 0.22 + f32(ring) * 0.19),
            time * 0.045 + f32(ring) * 0.28);
        let sweep = sin(u * 3.0 - time * 2.0 + f32(ring));
        var color = mix(vec3<f32>(0.05, 0.65, 1.0), vec3<f32>(1.0, 0.04, 0.42), f32(ring) / 7.0);
        color = mix(color, vec3<f32>(1.0, 0.38, 0.04), max(0.0, sweep) * 0.3);
        return FormationSample(tilted, color, 1.04 + 0.38 * max(0.0, sweep));
    }
    let local = index - ring_end;
    let core_count = max(1u, count - ring_end);
    let sphere = rotate_y(sphere_point(local, core_count), time * 0.8);
    let radial = sqrt(max(0.0, 1.0 - sphere.y * sphere.y));
    let pulse = 1.38 + 0.12 * sin(time * 1.7);
    return FormationSample(sphere * pulse,
        mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(0.5, 0.12, 1.0), radial), 1.42);
}

fn solar_phoenix(index: u32, count: u32, time: f32) -> FormationSample {
    let wing_count = count * 68u / 100u;
    let body_count = count * 17u / 100u;
    let body_end = wing_count + body_count;
    let flap = sin(time * 1.24);
    if (index < wing_count) {
        var side = 1.0;
        if (index % 2u == 0u) { side = -1.0; }
        let local = index / 2u;
        let per_side = max(1u, (wing_count + 1u) / 2u);
        let columns = max(1u, u32(ceil(sqrt(f32(per_side)))));
        let rows = max(1u, (per_side + columns - 1u) / columns);
        let u = f32(local % columns) / f32(max(1u, columns - 1u));
        let v = f32(local / columns) / f32(max(1u, rows - 1u));
        let ripple = sin(u * 10.0 - time * 2.2 + v * PI) * 0.32;
        let p = vec3<f32>(
            side * (1.95 + u * 11.3),
            2.4 + sin(u * PI) * 4.4 + flap * (0.2 + pow(u, 1.45)) * 2.3
                + (v - 0.5) * (2.4 + u * 5.2),
            (v - 0.5) * (1.5 + u * 4.8) + ripple
        );
        let root = vec3<f32>(1.0, 0.16, 0.015);
        var tip = vec3<f32>(1.0, 0.38, 0.04);
        if (side < 0.0) { tip = vec3<f32>(1.0, 0.04, 0.42); }
        let brightness = 1.02 + 0.2 * max(0.0, sin(time * 2.0 + u * TAU + v));
        return FormationSample(p, mix(root, tip, 0.2 + u * 0.8), brightness);
    }
    if (index < body_end) {
        let local = index - wing_count;
        let torso_count = max(1u, body_count * 70u / 100u);
        let head_count = max(1u, body_count * 20u / 100u);
        if (local < torso_count) {
            let u = (f32(local) + 0.5) / f32(torso_count);
            let sphere = sphere_point(local, torso_count);
            return FormationSample(
                vec3<f32>(sphere.x * 1.7, 0.8 + sphere.y * 3.6, sphere.z * 1.3),
                mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(1.0, 0.06, 0.18), u),
                1.14
            );
        }
        if (local < torso_count + head_count) {
            let head_local = local - torso_count;
            let u = (f32(head_local) + 0.5) / f32(head_count);
            let sphere = sphere_point(head_local, head_count);
            return FormationSample(
                vec3<f32>(sphere.x * 1.15, 6.3 + sphere.y * 1.15, sphere.z),
                vec3<f32>(1.0, 0.42, 0.025),
                1.35
            );
        }
        let neck_local = local - torso_count - head_count;
        let neck_count = max(1u, body_count - torso_count - head_count);
        let around_count = max(1u, u32(ceil(sqrt(f32(neck_count) * 1.5))));
        let height_count = max(1u, (neck_count + around_count - 1u) / around_count);
        let angle = TAU * f32(neck_local % around_count) / f32(around_count);
        let u = f32(neck_local / around_count) / f32(max(1u, height_count - 1u));
        return FormationSample(
            vec3<f32>(cos(angle) * 0.62, 4.48 + u * 0.66, sin(angle) * 0.62),
            vec3<f32>(1.0, 0.38, 0.04),
            1.24
        );
    }
    let tail_count = max(1u, count - body_end);
    let local = index - body_end;
    let plume = local % 3u;
    let ribbon_index = local / 3u;
    let per_plume = max(1u, (tail_count + 2u) / 3u);
    let cross_count = max(2u, u32(ceil(sqrt(f32(per_plume) / 5.0))));
    let along_count = max(1u, (per_plume + cross_count - 1u) / cross_count);
    let u = f32(ribbon_index / cross_count) / f32(max(1u, along_count - 1u));
    let across = f32(ribbon_index % cross_count) / f32(max(1u, cross_count - 1u)) - 0.5;
    let curl = u * (4.8 + f32(plume) * 0.55) + time * 0.72 + f32(plume) * 2.0;
    let p = vec3<f32>(
        (f32(plume) - 1.0) * (1.3 + u * 1.7) + sin(curl) * u * 0.72
            + across * (0.7 - u * 0.25),
        -3.15 - u * (4.8 + abs(f32(plume) - 1.0) * 0.7) + cos(curl) * 0.35,
        1.2 + u * 3.5 + sin(curl) * 0.5 + across * 0.7
    );
    let brightness = 1.0 + 0.24 * max(0.0, sin(time * 2.4 + u * 8.0));
    return FormationSample(p, mix(vec3<f32>(1.0, 0.38, 0.04), vec3<f32>(1.0, 0.04, 0.42), u * 0.88), brightness);
}

fn formation(kind: u32, index: u32, count: u32, time: f32) -> FormationSample {
    if (kind == 0u) { return launch_grid(index, count); }
    var result: FormationSample;
    if (kind == 1u) { result = stellar_chrysalis(index, count, time); }
    else if (kind == 2u) { result = heart(index, count, time); }
    else if (kind == 3u) { result = galaxy(index, count, time); }
    else if (kind == 4u) { result = prism_cathedral(index, count, time); }
    else if (kind == 5u) { result = helix(index, count, time); }
    else if (kind == 6u) { result = planet(index, count, time); }
    else if (kind == 7u) { result = infinity_portal(index, count, time); }
    else if (kind == 8u) { result = prismatic_lotus(index, count, time); }
    else if (kind == 9u) { result = celestial_crown(index, count, time); }
    else if (kind == 10u) { result = event_horizon(index, count, time); }
    else if (kind == 11u) { result = spectral_mandala(index, count, time); }
    else if (kind == 12u) { result = chrono_gyroscope(index, count, time); }
    else {
        let source_count = u32(globals.gpu_image.x + 0.5);
        if (source_count == 0u) { return heart(index, count, time); }
        var source_index = index % source_count;
        var replica = index / source_count;
        if (count <= source_count) {
            source_index = min(source_count - 1u, index * source_count / max(count, 1u));
            replica = 0u;
        }
        let source = raw_instances[source_index];
        let layer_side = max(1u, u32(ceil(sqrt(f32(max(1u, (count + source_count - 1u) / source_count))))));
        let layer_x = f32(replica % layer_side) - f32(layer_side - 1u) * 0.5;
        let layer_z = f32(replica / layer_side) - f32(layer_side - 1u) * 0.5;
        let inverse_scale = 1.0 / max(globals.gpu_meta.w, 0.001);
        let layer_offset = vec3<f32>(
            layer_x * globals.gpu_image.y * inverse_scale,
            0.0,
            layer_z * globals.gpu_image.y * inverse_scale
        );
        result = FormationSample(
            source.position_brightness.xyz * globals.gpu_image.w + layer_offset,
            source.color.xyz,
            source.position_brightness.w
        );
    }
    var lower_extent = 9.5;
    if (kind == 1u) { lower_extent = 11.4; }
    else if (kind == 2u) { lower_extent = 9.6; }
    else if (kind == 3u) { lower_extent = 8.5; }
    else if (kind == 4u) { lower_extent = 7.2; }
    else if (kind == 5u) { lower_extent = 8.8; }
    else if (kind == 6u) { lower_extent = 6.9; }
    else if (kind == 7u) { lower_extent = 12.0; }
    else if (kind == 8u) { lower_extent = 4.9; }
    else if (kind == 9u) { lower_extent = 5.2; }
    else if (kind == 10u) { lower_extent = 15.1; }
    else if (kind == 11u) { lower_extent = 12.8; }
    else if (kind == 12u) { lower_extent = 12.0; }
    result.position *= globals.gpu_meta.w;
    result.position.y += max(15.0, globals.gpu_meta.w * lower_extent + 3.0);
    return result;
}

fn safety_cell(position: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(position / (globals.gpu_safety.z + 0.04)));
}

fn safety_hash(cell: vec3<i32>) -> u32 {
    let mixed = u32(cell.x) * 73856093u ^ u32(cell.y) * 19349663u ^ u32(cell.z) * 83492791u;
    return mixed & (arrayLength(&bucket_heads) - 1u);
}

@compute @workgroup_size(64)
fn clear_safety_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < arrayLength(&bucket_heads)) {
        atomicStore(&bucket_heads[index], -1);
    }
    if (index == 0u) {
        atomicStore(&safety_counters.warning_pairs, 0u);
        atomicStore(&safety_counters.collision_pairs, 0u);
        atomicStore(&safety_counters.ground_breaches, 0u);
        atomicStore(&safety_counters.minimum_distance_bits, bitcast<u32>(globals.gpu_safety.z));
        atomicStore(&safety_counters.collision_a, 0xffffffffu);
        atomicStore(&safety_counters.collision_b, 0xffffffffu);
        atomicStore(&safety_counters.collision_x_bits, 0u);
        atomicStore(&safety_counters.collision_y_bits, 0u);
        atomicStore(&safety_counters.collision_z_bits, 0u);
        atomicStore(&safety_counters.collision_distance_bits, 0u);
        atomicStore(&safety_counters.ground_drone, 0xffffffffu);
        atomicStore(&safety_counters.ground_x_bits, 0u);
        atomicStore(&safety_counters.ground_y_bits, 0u);
        atomicStore(&safety_counters.ground_z_bits, 0u);
        atomicStore(&safety_counters.show_time_bits, bitcast<u32>(globals.time_exposure_bloom_haze.x));
    }
}

@compute @workgroup_size(64)
fn insert_safety_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = u32(globals.fleet_lod.w + 0.5);
    if (index >= count) { return; }
    let cell = safety_cell(processed_instances[index].position_brightness.xyz);
    let bucket = safety_hash(cell);
    spatial_nodes[index].cell = cell;
    spatial_nodes[index].next = atomicExchange(&bucket_heads[bucket], i32(index));
}

@compute @workgroup_size(64)
fn correct_safety_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = u32(globals.fleet_lod.w + 0.5);
    if (index >= count) { return; }
    var instance = processed_instances[index];
    let position = instance.position_brightness.xyz;
    let own_cell = spatial_nodes[index].cell;
    var correction = vec3<f32>(0.0);
    var correction_count = 0.0;
    for (var dx = -1; dx <= 1; dx += 1) {
        for (var dy = -1; dy <= 1; dy += 1) {
            for (var dz = -1; dz <= 1; dz += 1) {
                let neighbor_cell = own_cell + vec3<i32>(dx, dy, dz);
                var node = atomicLoad(&bucket_heads[safety_hash(neighbor_cell)]);
                loop {
                    if (node < 0) { break; }
                    let other = u32(node);
                    if (other != index && all(spatial_nodes[other].cell == neighbor_cell)) {
                        let delta = position - processed_instances[other].position_brightness.xyz;
                        let distance = length(delta);
                        let correction_distance = globals.gpu_safety.z + 0.04;
                        if (distance < correction_distance) {
                            var direction: vec3<f32>;
                            if (distance > 0.00001) {
                                direction = delta / distance;
                            } else {
                                let lower = min(index, other);
                                let upper = max(index, other);
                                let angle = TAU * hash01(lower * 1664525u + upper * 1013904223u);
                                direction = normalize(vec3<f32>(cos(angle), 0.31, sin(angle)));
                                if (index > other) { direction = -direction; }
                            }
                            correction += direction * (correction_distance - distance) * 0.52;
                            correction_count += 1.0;
                        }
                    }
                    node = spatial_nodes[other].next;
                }
            }
        }
    }
    // Summing instead of averaging preserves authority in dense clusters.
    // A tiny per-aircraft escape vector breaks perfectly symmetric overlaps.
    if (correction_count > 0.0) {
        let escape_angle = TAU * hash01(index * 2246822519u + 3266489917u);
        correction += normalize(vec3<f32>(cos(escape_angle), 0.19, sin(escape_angle))) * 0.006;
    }
    let correction_length = length(correction);
    if (correction_length > 0.08) {
        correction *= 0.08 / correction_length;
    }
    let corrected_position = position + correction;
    instance.position_brightness = vec4<f32>(
        corrected_position.x,
        max(corrected_position.y, globals.gpu_safety.w + 0.012),
        corrected_position.z,
        instance.position_brightness.w
    );
    corrected_instances[index] = instance;
}

@compute @workgroup_size(64)
fn apply_safety_corrections(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = u32(globals.fleet_lod.w + 0.5);
    if (index >= count) { return; }
    processed_instances[index] = corrected_instances[index];
}

@compute @workgroup_size(64)
fn audit_safety_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = u32(globals.fleet_lod.w + 0.5);
    if (globals.gpu_safety.x < 0.5 || index >= count) { return; }
    var instance = processed_instances[index];
    let position = instance.position_brightness.xyz;
    let own_cell = spatial_nodes[index].cell;
    var warning = false;
    var collision = false;
    let ground_clearance = position.y - globals.gpu_safety.w;
    let ground_breach = ground_clearance < 0.0;
    warning = ground_clearance < 0.12;
    if (ground_breach) {
        atomicAdd(&safety_counters.ground_breaches, 1u);
        let previous_ground = atomicMin(&safety_counters.ground_drone, index);
        if (index < previous_ground) {
            atomicStore(&safety_counters.ground_x_bits, bitcast<u32>(position.x));
            atomicStore(&safety_counters.ground_y_bits, bitcast<u32>(position.y));
            atomicStore(&safety_counters.ground_z_bits, bitcast<u32>(position.z));
        }
    }
    if (globals.safety_options.y > 0.5) {
        for (var dx = -1; dx <= 1; dx += 1) {
            for (var dy = -1; dy <= 1; dy += 1) {
                for (var dz = -1; dz <= 1; dz += 1) {
                    let neighbor_cell = own_cell + vec3<i32>(dx, dy, dz);
                    var node = atomicLoad(&bucket_heads[safety_hash(neighbor_cell)]);
                    loop {
                        if (node < 0) { break; }
                        let other = u32(node);
                        if (other != index && all(spatial_nodes[other].cell == neighbor_cell)) {
                            let distance = length(position - processed_instances[other].position_brightness.xyz);
                            if (distance < globals.gpu_safety.z) {
                                warning = true;
                                if (other > index) {
                                    atomicAdd(&safety_counters.warning_pairs, 1u);
                                    atomicMin(&safety_counters.minimum_distance_bits, bitcast<u32>(distance));
                                }
                            }
                            if (distance < globals.gpu_safety.y) {
                                collision = true;
                                if (other > index) {
                                    atomicAdd(&safety_counters.collision_pairs, 1u);
                                    let previous_collision = atomicMin(&safety_counters.collision_a, index);
                                    if (index < previous_collision) {
                                        let midpoint = (position + processed_instances[other].position_brightness.xyz) * 0.5;
                                        atomicStore(&safety_counters.collision_b, other);
                                        atomicStore(&safety_counters.collision_x_bits, bitcast<u32>(midpoint.x));
                                        atomicStore(&safety_counters.collision_y_bits, bitcast<u32>(midpoint.y));
                                        atomicStore(&safety_counters.collision_z_bits, bitcast<u32>(midpoint.z));
                                        atomicStore(&safety_counters.collision_distance_bits, bitcast<u32>(distance));
                                    }
                                }
                            }
                        }
                        node = spatial_nodes[other].next;
                    }
                }
            }
        }
    }
    if (globals.safety_options.x > 0.5) {
        if (collision || ground_breach) {
            instance.misc.w = 2.0;
            instance.color = vec4<f32>(vec3<f32>(1.0, 0.015, 0.01) * 2.8, instance.color.w);
        } else if (warning) {
            instance.misc.w = 1.0;
            instance.color = vec4<f32>(vec3<f32>(1.0, 0.34, 0.015) * 1.8, instance.color.w);
        }
    }
    processed_instances[index] = instance;
}

@compute @workgroup_size(64)
fn prepare_instances(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = u32(globals.fleet_lod.w + 0.5);
    if (index >= count) { return; }
    if (globals.gpu_show.x > 0.5) {
        let previous_kind = u32(globals.gpu_show.y + 0.5);
        let target_kind = u32(globals.gpu_show.z + 0.5);
        let destination = formation(target_kind, index, count, globals.gpu_meta.x);
        var source = destination;
        var progress = 1.0;
        if (globals.gpu_meta.z > 0.5) {
            source = formation(previous_kind, index, count, globals.gpu_meta.x);
            let raw_progress = clamp(globals.gpu_show.w, 0.0, 1.0);
            progress = raw_progress * raw_progress * (3.0 - 2.0 * raw_progress);
        }
        var position = mix(source.position, destination.position, progress);
        if (globals.gpu_meta.z > 0.5) {
            // Safety-first three-stage route: climb vertically without changing
            // formation spacing, cross at one of 128 separated flight levels,
            // then descend vertically into the destination. Only aircraft on
            // the same level can have intersecting horizontal paths.
            let raw_progress = clamp(globals.gpu_show.w, 0.0, 1.0);
            let lane_altitude = f32(index % 128u) * 0.48;
            let lift = max(5.0, globals.gpu_meta.w * 0.72) + lane_altitude;
            let source_stage = source.position + vec3<f32>(0.0, lift, 0.0);
            let destination_stage = destination.position + vec3<f32>(0.0, lift, 0.0);
            if (raw_progress < 0.22) {
                let stage = smoothstep(0.0, 0.22, raw_progress);
                position = mix(source.position, source_stage, stage);
            } else if (raw_progress < 0.78) {
                let stage = smoothstep(0.22, 0.78, raw_progress);
                position = mix(source_stage, destination_stage, stage);
            } else {
                let stage = smoothstep(0.78, 1.0, raw_progress);
                position = mix(destination_stage, destination.position, stage);
            }
        }
        var brightness = mix(source.brightness, destination.brightness, progress);
        if (previous_kind == 0u && target_kind != 0u) { brightness *= smoothstep(0.02, 0.2, progress); }
        if (target_kind == 0u) { brightness *= 1.0 - smoothstep(0.78, 1.0, progress); }
        let color = mix(source.color, destination.color, progress);
        let pulse = 0.94 + 0.06 * sin(globals.time_exposure_bloom_haze.x * 3.2 + hash01(index) * 18.0);
        processed_instances[index] = Instance(
            vec4<f32>(position, brightness),
            vec4<f32>(color * brightness * pulse, 0.05),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
            vec4<f32>(globals.time_exposure_bloom_haze.x * 36.0, hash01(index), 1.0, 0.0)
        );
        return;
    }
    var instance = raw_instances[index];
    let pulse = 0.94 + 0.06 * sin(globals.time_exposure_bloom_haze.x * 3.2 + instance.misc.y * 18.0);
    instance.color *= vec4<f32>(pulse, pulse, pulse, 1.0);
    processed_instances[index] = instance;
}
