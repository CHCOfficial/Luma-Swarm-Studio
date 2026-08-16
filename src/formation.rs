use std::f32::consts::{PI, TAU};

use glam::{Mat3, Vec3};

use crate::model::{FormationKind, FormationPoint, Rgbw};

/// Generates deterministic, normalized point clouds. Animation is sampled from
/// `time`, which makes scrubbing and replay bit-for-bit repeatable.
pub fn generate(kind: FormationKind, count: usize, time: f32) -> Vec<FormationPoint> {
    match kind {
        FormationKind::LaunchGrid => launch_grid(count),
        FormationKind::Chrysalis => stellar_chrysalis(count, time),
        FormationKind::Heart => heart(count, time),
        FormationKind::Galaxy => galaxy(count, time),
        FormationKind::Cathedral => prism_cathedral(count, time),
        FormationKind::Human => human(count, time),
        FormationKind::Planet => planet(count, time),
        FormationKind::Infinity => infinity_portal(count, time),
        FormationKind::Lotus => prismatic_lotus(count, time),
        FormationKind::Crown => celestial_crown(count, time),
        FormationKind::EventHorizon => event_horizon(count, time),
        FormationKind::Mandala => spectral_mandala(count, time),
        FormationKind::Gyroscope => chrono_gyroscope(count, time),
        // Imported imagery is supplied by FleetSimulation. This deterministic
        // placeholder keeps older timelines valid before an image is loaded.
        FormationKind::Image => heart(count, time),
    }
}

pub fn launch_grid(count: usize) -> Vec<FormationPoint> {
    let columns = (count as f32).sqrt().ceil() as usize;
    let rows = count.div_ceil(columns);
    let spacing = 1.55;
    (0..count)
        .map(|i| {
            let x = (i % columns) as f32 - (columns - 1) as f32 * 0.5;
            let z = (i / columns) as f32 - (rows - 1) as f32 * 0.5;
            point(
                Vec3::new(x * spacing, 0.34, z * spacing),
                Rgbw::new(0.04, 0.12, 0.2, 0.02),
                0.15,
                hash01(i as u32),
                0,
            )
        })
        .collect()
}

fn stellar_chrysalis(count: usize, time: f32) -> Vec<FormationPoint> {
    let ribbon_end = count * 68 / 100;
    let core_end = count * 86 / 100;
    (0..count)
        .map(|i| {
            if i < ribbon_end {
                let ribbon = i % 12;
                let local = i / 12;
                let per_ribbon = ribbon_end.div_ceil(12).max(1);
                let cross_count = ((per_ribbon as f32 / 10.0).sqrt().ceil() as usize).max(3);
                let along_count = per_ribbon.div_ceil(cross_count).max(1);
                let u = (local / cross_count) as f32 / (along_count - 1).max(1) as f32;
                let across = (local % cross_count) as f32 / (cross_count - 1).max(1) as f32 - 0.5;
                let latitude = (u - 0.5) * PI;
                let petal_phase = ribbon as f32 * TAU / 12.0;
                let breathing = 1.0 + 0.12 * (time * 0.82 + petal_phase).sin();
                let radius = 0.8
                    + latitude.cos().abs().powf(0.62)
                        * (6.1 + 1.15 * (petal_phase * 2.0 - time * 0.7).sin())
                        * breathing;
                let longitude = petal_phase + latitude.sin() * 0.72 + time * 0.12 + across * 0.17;
                let p = Vec3::new(
                    longitude.cos() * (radius + across * 0.45),
                    latitude.sin() * 10.2,
                    longitude.sin() * (radius + across * 0.45),
                );
                let sweep = 0.5 + 0.5 * (u * 13.0 - time * 2.2 + petal_phase).sin();
                point(
                    p,
                    Rgbw::MAGENTA
                        .mix(Rgbw::CYAN, u)
                        .mix(Rgbw::GOLD, sweep * 0.38),
                    1.0 + sweep * 0.38,
                    u,
                    ribbon as u16,
                )
            } else if i < core_end {
                let local = i - ribbon_end;
                let core_count = (core_end - ribbon_end).max(1);
                let y = 1.0 - 2.0 * (local as f32 + 0.5) / core_count as f32;
                let radius = (1.0 - y * y).sqrt();
                let theta = PI * (3.0 - 5.0_f32.sqrt()) * local as f32 + time * 0.52;
                let pulse = 3.2 + 0.42 * (time * 1.9).sin();
                point(
                    Vec3::new(theta.cos() * radius, y, theta.sin() * radius) * pulse,
                    Rgbw::GOLD.mix(Rgbw::new(1.0, 0.92, 0.52, 0.26), radius),
                    1.42 + 0.3 * (theta * 2.0 - time).sin().max(0.0),
                    local as f32 / core_count as f32,
                    20,
                )
            } else {
                let local = i - core_end;
                let orbit_count = (count - core_end).max(1);
                let orbit = local % 4;
                let orbit_local = local / 4;
                let per_orbit = orbit_count.div_ceil(4).max(1);
                let angle = TAU * orbit_local as f32 / per_orbit as f32
                    + time * (0.16 + orbit as f32 * 0.035);
                let radius = 10.6 + orbit as f32 * 1.15;
                let flat = Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius);
                let tilt = Mat3::from_rotation_z((orbit as f32 - 1.5) * 0.3)
                    * Mat3::from_rotation_x(0.38 + orbit as f32 * 0.15);
                point(
                    tilt * flat,
                    [Rgbw::CYAN, Rgbw::MAGENTA, Rgbw::GOLD, Rgbw::CYAN][orbit],
                    1.04 + 0.3 * (angle * 6.0 - time * 1.4).sin().max(0.0),
                    angle / TAU,
                    30 + orbit as u16,
                )
            }
        })
        .collect()
}

#[allow(dead_code)]
fn continent_mask(lat: f32, lon: f32) -> bool {
    // A low-poly geographic atlas gives the dense GPU globe stable coastlines.
    // Unlike overlapping ellipses, these outlines do not merge into new shapes
    // as sampling density increases.
    const NORTH_AMERICA: &[(f32, f32)] = &[
        (-168.0, 66.0),
        (-154.0, 72.0),
        (-140.0, 70.0),
        (-130.0, 60.0),
        (-124.0, 50.0),
        (-117.0, 32.0),
        (-105.0, 22.0),
        (-97.0, 16.0),
        (-87.0, 14.0),
        (-82.0, 22.0),
        (-80.0, 30.0),
        (-74.0, 40.0),
        (-60.0, 47.0),
        (-55.0, 55.0),
        (-65.0, 64.0),
        (-82.0, 70.0),
        (-100.0, 78.0),
        (-130.0, 72.0),
    ];
    const SOUTH_AMERICA: &[(f32, f32)] = &[
        (-81.0, 12.0),
        (-70.0, 10.0),
        (-60.0, 7.0),
        (-50.0, 1.0),
        (-35.0, -8.0),
        (-40.0, -20.0),
        (-52.0, -34.0),
        (-60.0, -52.0),
        (-70.0, -55.0),
        (-76.0, -35.0),
        (-80.0, -15.0),
        (-75.0, -2.0),
    ];
    const AFRICA: &[(f32, f32)] = &[
        (-17.0, 35.0),
        (0.0, 37.0),
        (15.0, 33.0),
        (32.0, 31.0),
        (42.0, 12.0),
        (51.0, 3.0),
        (42.0, -12.0),
        (35.0, -25.0),
        (20.0, -35.0),
        (8.0, -34.0),
        (-2.0, -20.0),
        (-10.0, -5.0),
        (-17.0, 12.0),
    ];
    const EUROPE: &[(f32, f32)] = &[
        (-10.0, 36.0),
        (-10.0, 60.0),
        (5.0, 70.0),
        (25.0, 72.0),
        (40.0, 62.0),
        (45.0, 50.0),
        (30.0, 42.0),
        (22.0, 36.0),
        (12.0, 44.0),
        (0.0, 43.0),
    ];
    const ASIA: &[(f32, f32)] = &[
        (26.0, 36.0),
        (40.0, 60.0),
        (60.0, 74.0),
        (95.0, 78.0),
        (130.0, 72.0),
        (160.0, 65.0),
        (178.0, 55.0),
        (160.0, 45.0),
        (140.0, 40.0),
        (125.0, 20.0),
        (108.0, 5.0),
        (98.0, 20.0),
        (82.0, 8.0),
        (72.0, 20.0),
        (60.0, 28.0),
        (48.0, 30.0),
        (40.0, 40.0),
        (32.0, 44.0),
    ];
    const ARABIA: &[(f32, f32)] = &[
        (35.0, 30.0),
        (50.0, 30.0),
        (58.0, 20.0),
        (50.0, 12.0),
        (42.0, 15.0),
    ];
    const INDIA: &[(f32, f32)] = &[(68.0, 28.0), (88.0, 28.0), (82.0, 8.0), (76.0, 6.0)];
    const SOUTH_EAST_ASIA: &[(f32, f32)] = &[
        (90.0, 28.0),
        (110.0, 22.0),
        (125.0, 10.0),
        (120.0, -5.0),
        (105.0, 0.0),
        (98.0, 15.0),
    ];
    const AUSTRALIA: &[(f32, f32)] = &[
        (112.0, -11.0),
        (130.0, -10.0),
        (145.0, -17.0),
        (154.0, -28.0),
        (145.0, -40.0),
        (120.0, -35.0),
        (112.0, -22.0),
    ];
    const GREENLAND: &[(f32, f32)] = &[
        (-55.0, 60.0),
        (-42.0, 59.0),
        (-25.0, 70.0),
        (-20.0, 82.0),
        (-42.0, 84.0),
        (-60.0, 76.0),
        (-65.0, 68.0),
    ];
    const MADAGASCAR: &[(f32, f32)] = &[(47.0, -12.0), (51.0, -16.0), (48.0, -27.0), (44.0, -23.0)];
    const JAPAN: &[(f32, f32)] = &[
        (130.0, 31.0),
        (136.0, 34.0),
        (142.0, 45.0),
        (146.0, 44.0),
        (141.0, 34.0),
    ];
    const BRITAIN: &[(f32, f32)] = &[(-8.0, 50.0), (-2.0, 50.0), (1.0, 59.0), (-6.0, 58.0)];
    const NEW_ZEALAND: &[(f32, f32)] = &[
        (166.0, -34.0),
        (174.0, -40.0),
        (178.0, -47.0),
        (169.0, -45.0),
    ];
    const INDONESIA_WEST: &[(f32, f32)] = &[(95.0, 5.0), (113.0, 4.0), (112.0, -8.0), (98.0, -7.0)];
    const INDONESIA_EAST: &[(f32, f32)] =
        &[(113.0, 2.0), (135.0, 0.0), (141.0, -9.0), (118.0, -10.0)];
    const HUDSON_BAY: &[(f32, f32)] = &[
        (-96.0, 64.0),
        (-82.0, 65.0),
        (-76.0, 58.0),
        (-82.0, 51.0),
        (-94.0, 54.0),
        (-99.0, 60.0),
    ];
    const MEDITERRANEAN: &[(f32, f32)] = &[
        (-6.0, 36.0),
        (0.0, 43.0),
        (15.0, 45.0),
        (30.0, 41.0),
        (36.0, 34.0),
        (25.0, 30.0),
        (10.0, 30.0),
    ];

    let land = [
        NORTH_AMERICA,
        SOUTH_AMERICA,
        AFRICA,
        EUROPE,
        ASIA,
        ARABIA,
        INDIA,
        SOUTH_EAST_ASIA,
        AUSTRALIA,
        GREENLAND,
        MADAGASCAR,
        JAPAN,
        BRITAIN,
        NEW_ZEALAND,
        INDONESIA_WEST,
        INDONESIA_EAST,
    ]
    .iter()
    .any(|polygon| geo_polygon_contains(lat, lon, polygon));
    let antarctica_coast = -70.5 - (lon * 0.11).sin() * 3.2 - (lon * 0.037).cos() * 1.6;
    let inland_water =
        geo_polygon_contains(lat, lon, HUDSON_BAY) || geo_polygon_contains(lat, lon, MEDITERRANEAN);
    (land || lat < antarctica_coast) && !inland_water
}

#[allow(dead_code)]
fn geo_polygon_contains(lat: f32, lon: f32, vertices: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut previous = vertices.len() - 1;
    for current in 0..vertices.len() {
        let (current_lon, current_lat) = vertices[current];
        let (previous_lon, previous_lat) = vertices[previous];
        if (current_lat > lat) != (previous_lat > lat)
            && lon
                < (previous_lon - current_lon) * (lat - current_lat) / (previous_lat - current_lat)
                    + current_lon
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

#[allow(dead_code)]
fn distance_to_grid(value: f32, spacing: f32) -> f32 {
    let wrapped = (value + spacing * 0.5).rem_euclid(spacing) - spacing * 0.5;
    wrapped.abs()
}

fn heart(count: usize, time: f32) -> Vec<FormationPoint> {
    let beat = 1.0 + 0.045 * (time * 2.6).sin().max(-0.3);
    let cross_count = ((count.max(1) as f32 / 10.0).sqrt().round() as usize).clamp(6, 48);
    let along_count = count.div_ceil(cross_count).max(1);
    let curve_resolution = 1_024usize;
    let mut arc_lengths = Vec::with_capacity(curve_resolution + 1);
    let mut previous = heart_xy(0.0);
    let mut length = 0.0;
    arc_lengths.push(0.0);
    for step in 1..=curve_resolution {
        let p = heart_xy(TAU * step as f32 / curve_resolution as f32);
        length += p.distance(previous);
        arc_lengths.push(length);
        previous = p;
    }
    (0..count)
        .map(|i| {
            let cross_index = i % cross_count;
            let along_index = i / cross_count;
            let u = (along_index as f32 + 0.5) / along_count as f32;
            let wanted = u.fract() * length;
            let upper = arc_lengths
                .partition_point(|&sample| sample < wanted)
                .clamp(1, curve_resolution);
            let lower_length = arc_lengths[upper - 1];
            let segment_length = (arc_lengths[upper] - lower_length).max(1e-6);
            let within = (wanted - lower_length) / segment_length;
            let t = TAU * (upper as f32 - 1.0 + within) / curve_resolution as f32;
            let xy = heart_xy(t);
            let x = xy.x;
            let y = xy.y;
            let layer = (cross_index as f32 + 0.5) / cross_count as f32;
            let radial_scale = 0.76 + layer * 0.24;
            let cusp_distance = u.min(1.0 - u).min((u - 0.5).abs());
            let cusp_separation =
                (1.0 - smoothstep(0.0, 0.06, cusp_distance)) * if u < 0.5 { -0.08 } else { 0.08 };
            let p = Vec3::new(
                x * 0.52 * radial_scale,
                y * 0.52 * radial_scale,
                (layer - 0.5) * 1.2 + (t * 3.0 + time * 0.4).sin() * 0.035 + cusp_separation,
            ) * beat;
            let hot = 0.5 + 0.5 * (t + time * 1.8).sin();
            point(
                p,
                Rgbw::MAGENTA.mix(Rgbw::new(1.0, 0.18, 0.03, 0.16), hot),
                1.1,
                u,
                if t < PI { 0 } else { 1 },
            )
        })
        .collect()
}

fn heart_xy(t: f32) -> glam::Vec2 {
    glam::Vec2::new(
        16.0 * t.sin().powi(3),
        13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos(),
    )
}

fn galaxy(count: usize, time: f32) -> Vec<FormationPoint> {
    let presentation_tilt = Mat3::from_rotation_x(0.72);
    (0..count)
        .map(|i| {
            let arm = i % 4;
            let local = i / 4;
            let local_count = count.div_ceil(4).max(1);
            let cross_count = ((local_count as f32 / 7.5).sqrt().ceil() as usize).max(4);
            let cross_index = local % cross_count;
            let along_index = local / cross_count;
            let along_count = local_count.div_ceil(cross_count).max(1);
            let u = (along_index as f32 + 0.5) / along_count as f32;
            let across = (cross_index as f32 + 0.5) / cross_count as f32 - 0.5;
            let radius = 0.9 + u.sqrt() * 11.1 + across * 0.72;
            let angle = arm as f32 * TAU / 4.0 + radius * 0.68 + time * 0.12 + across * 0.08;
            let p = presentation_tilt
                * Vec3::new(angle.cos() * radius, across * 2.2, angle.sin() * radius);
            point(
                p,
                Rgbw::CYAN.mix(Rgbw::new(0.68, 0.12, 1.0, 0.1), u),
                1.0,
                u,
                arm as u16,
            )
        })
        .collect()
}

#[allow(dead_code)]
fn bird(count: usize, time: f32) -> Vec<FormationPoint> {
    let flap = (time * 1.72).sin();
    (0..count)
        .map(|i| {
            let region = i % 20;
            let u = hash01(i as u32 * 47 + 13);
            let v = hash01(i as u32 * 79 + 31);
            let (p, group) = if region < 13 {
                let side = if i & 1 == 0 { -1.0 } else { 1.0 };
                let feather = (v * 7.0).floor();
                let x = side * (1.0 + u * 11.5);
                let arch = (u * PI).sin() * 3.6;
                let feather_drop = (v - 0.5) * (0.9 + u * 3.4);
                let y = 1.2 + arch + flap * u.powf(1.35) * 2.7 + feather_drop;
                let z =
                    (v - 0.5) * (1.0 + u * 4.0) + (u * 10.0 + time * 0.9 + feather).sin() * 0.28;
                (Vec3::new(x, y, z), if side < 0.0 { 0 } else { 1 })
            } else if region < 17 {
                let theta = TAU * v;
                let length = (u - 0.5) * 8.0;
                let taper = (1.0 - (u * 2.0 - 1.0).abs()).sqrt();
                (
                    Vec3::new(
                        theta.cos() * 1.3 * taper,
                        1.4 + theta.sin() * 1.2 * taper,
                        length,
                    ),
                    2,
                )
            } else {
                let plume = region - 17;
                let curl = u * 4.8 + time * 0.55 + plume as f32 * 1.9;
                (
                    Vec3::new(
                        (plume as f32 - 1.0) * 1.7 + curl.sin() * u * 1.4,
                        -0.4 - u * 8.0 + curl.cos() * 0.55,
                        3.2 + u * 5.5 + curl.sin() * 0.7,
                    ),
                    3 + plume as u16,
                )
            };
            let spectrum = Rgbw::GOLD
                .mix(Rgbw::MAGENTA, smoothstep(-4.0, 4.5, p.y))
                .mix(Rgbw::CYAN, v * 0.42);
            point(
                p,
                spectrum,
                0.95 + 0.32 * (time * 2.0 + u * TAU).sin().max(0.0),
                u,
                group,
            )
        })
        .collect()
}

fn prism_cathedral(count: usize, time: f32) -> Vec<FormationPoint> {
    let arch_end = count * 72 / 100;
    let rose_end = count * 88 / 100;
    (0..count)
        .map(|i| {
            if i < arch_end {
                let arch = i % 8;
                let local = i / 8;
                let per_arch = arch_end.div_ceil(8).max(1);
                let cross_count = ((per_arch as f32 / 11.0).sqrt().ceil() as usize).max(3);
                let along_count = per_arch.div_ceil(cross_count).max(1);
                let u = (local / cross_count) as f32 / (along_count - 1).max(1) as f32;
                let across = (local % cross_count) as f32 / (cross_count - 1).max(1) as f32 - 0.5;
                let arch_angle = arch as f32 * TAU / 8.0 + time * 0.055;
                // Eight one-sided radial vaults avoid the exact duplicates
                // produced by rotating a symmetric full arch through 180°.
                let local_x = 2.3 + u * 6.1;
                let vault = (PI * u).sin().max(0.0).powf(0.72);
                let local_point = Vec3::new(
                    local_x,
                    -6.0 + vault * (13.5 + 0.8 * (time * 0.7 + arch_angle).sin()),
                    across * (1.25 + vault * 0.8),
                );
                let p = Mat3::from_rotation_y(arch_angle) * local_point;
                let sweep = 0.5 + 0.5 * (u * 12.0 - time * 2.4 + arch_angle).sin();
                point(
                    p,
                    Rgbw::CYAN
                        .mix(Rgbw::MAGENTA, arch as f32 / 7.0)
                        .mix(Rgbw::GOLD, sweep * 0.45),
                    1.02 + sweep * 0.34,
                    u,
                    arch as u16,
                )
            } else if i < rose_end {
                let local = i - arch_end;
                let rose_count = (rose_end - arch_end).max(1);
                let ring = local % 5;
                let serial = local / 5;
                let per_ring = rose_count.div_ceil(5).max(1);
                let tube_cross_count = 3;
                let cross = serial % tube_cross_count;
                let ring_local = serial / tube_cross_count;
                let around_count = per_ring.div_ceil(tube_cross_count).max(1);
                let angle = TAU * ring_local as f32 / around_count as f32
                    + time * (0.28 - ring as f32 * 0.025);
                let radius =
                    1.4 + ring as f32 * 1.12 + (cross as f32 - 1.0) * 0.34 / fleet_scale(count);
                let face = Vec3::new(angle.cos() * radius, 2.4 + angle.sin() * radius, 0.0);
                point(
                    Mat3::from_rotation_y(time * 0.11) * face,
                    Rgbw::GOLD.mix(Rgbw::MAGENTA, ring as f32 / 4.0),
                    1.18 + 0.3 * (angle * 7.0 - time).sin().max(0.0),
                    angle / TAU,
                    20 + ring as u16,
                )
            } else {
                let local = i - rose_end;
                let spire_count = (count - rose_end).max(1);
                let spire = local % 8;
                let along = local / 8;
                let per_spire = spire_count.div_ceil(8).max(1);
                let u = (along as f32 + 0.5) / per_spire as f32;
                let base_angle = spire as f32 * TAU / 8.0 + time * 0.055;
                let twist = base_angle + u * 1.1 + time * 0.2;
                let radius = 9.1 - u * 1.8;
                point(
                    Vec3::new(
                        twist.cos() * radius,
                        -6.0 + u * (15.0 + 2.0 * (spire & 1) as f32),
                        twist.sin() * radius,
                    ),
                    Rgbw::GOLD.mix(Rgbw::new(1.0, 0.94, 0.5, 0.28), u),
                    1.2 + 0.36 * (u * 15.0 - time * 3.0).sin().max(0.0),
                    u,
                    30 + spire as u16,
                )
            }
        })
        .collect()
}

fn human(count: usize, time: f32) -> Vec<FormationPoint> {
    let cross_count = ((count.max(1) as f32 / 6.0).sqrt().ceil() as usize).max(8);
    let along_count = count.div_ceil(cross_count).max(1);
    let tube_count = cross_count.div_ceil(2).max(1);
    (0..count)
        .map(|i| {
            let cross_index = i % cross_count;
            let along_index = i / cross_count;
            let u = (along_index as f32 + 0.5) / along_count as f32;
            let y = (u - 0.5) * 17.0;
            let theta = u * TAU * 3.1 + time * 0.64;
            let radius = 4.4 + (y * 0.42 + time).sin() * 0.35;
            let strand = if cross_index & 1 == 0 { -1.0 } else { 1.0 };
            let tube_index = cross_index / 2;
            let ribbon = Vec3::new(theta.cos() * radius, y, theta.sin() * radius);
            let opposite = Vec3::new(-theta.cos() * radius, y, -theta.sin() * radius);
            let mut p = ribbon.lerp(opposite, (strand + 1.0) * 0.5);
            let tube_angle =
                TAU * (tube_index as f32 + 0.5 * (along_index & 1) as f32) / tube_count as f32;
            let radial = Vec3::new(theta.cos(), 0.0, theta.sin());
            let tangent_side = Vec3::new(-theta.sin(), 0.0, theta.cos());
            p += radial * tube_angle.cos() * 0.7 + tangent_side * tube_angle.sin() * 0.7;
            point(
                p,
                if strand < 0.0 {
                    Rgbw::MAGENTA
                } else {
                    Rgbw::CYAN
                },
                1.12,
                u,
                if strand < 0.0 { 0 } else { 1 },
            )
        })
        .collect()
}

fn planet(count: usize, time: f32) -> Vec<FormationPoint> {
    let sphere_end = count * 52 / 100;
    let ring_end = count * 82 / 100;
    let atmosphere_end = count * 90 / 100;
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    let world_rotation = Mat3::from_rotation_y(time * 0.16);
    let ring_tilt = Mat3::from_rotation_z(0.13) * Mat3::from_rotation_x(0.46);
    (0..count)
        .map(|i| {
            if i < sphere_end {
                let u = (i as f32 + 0.5) / sphere_end.max(1) as f32;
                let y = 1.0 - 2.0 * u;
                let r = (1.0 - y * y).sqrt();
                let theta = golden * i as f32;
                let local = Vec3::new(theta.cos() * r, y, theta.sin() * r);
                let latitude_band = 0.5 + 0.5 * (y * 28.0 + time * 0.32).sin();
                let longitude_storm = 0.5 + 0.5 * (theta * 2.0 - time * 0.7).cos();
                let storm = smoothstep(0.72, 1.0, longitude_storm)
                    * (1.0 - smoothstep(0.12, 0.48, (y + 0.28).abs()));
                let color = Rgbw::new(0.03, 0.16, 0.72, 0.05)
                    .mix(Rgbw::new(0.08, 0.72, 1.0, 0.1), latitude_band)
                    .mix(Rgbw::MAGENTA, storm * 0.68);
                point(
                    world_rotation * Vec3::new(local.x * 6.4, local.y * 5.72, local.z * 6.4),
                    color,
                    1.02 + storm * 0.35,
                    u,
                    0,
                )
            } else if i < ring_end {
                let local = i - sphere_end;
                let ring_count = (ring_end - sphere_end).max(1);
                let radial_layers = ((ring_count as f32 * 0.055).sqrt().ceil() as usize).max(5);
                let radial_index = local % radial_layers;
                let angular_index = local / radial_layers;
                let angular_count = ring_count.div_ceil(radial_layers).max(1);
                let band = (radial_index as f32 + 0.5) / radial_layers as f32;
                let radius = 7.55 + band * 4.75 + if band > 0.58 { 0.58 } else { 0.0 };
                let angle = TAU * (angular_index as f32 + 0.5 * (radial_index & 1) as f32)
                    / angular_count as f32
                    + time * (0.08 + band * 0.035);
                let ripple = (angle * 7.0 - time * 1.8 + band * 9.0).sin() * 0.075;
                let p = ring_tilt * Vec3::new(angle.cos() * radius, ripple, angle.sin() * radius);
                let color = Rgbw::GOLD
                    .mix(Rgbw::MAGENTA, smoothstep(0.12, 0.72, band))
                    .mix(Rgbw::CYAN, smoothstep(0.72, 1.0, band));
                point(
                    p,
                    color,
                    1.12 + ripple.abs() * 1.8,
                    band,
                    1 + radial_index as u16,
                )
            } else if i < atmosphere_end {
                let local = i - ring_end;
                let atmosphere_count = (atmosphere_end - ring_end).max(1);
                let u = (local as f32 + 0.5) / atmosphere_count as f32;
                let y = 1.0 - 2.0 * u;
                let r = (1.0 - y * y).sqrt();
                let theta = golden * local as f32 + time * 0.28;
                let pulse = 0.5 + 0.5 * (theta * 3.0 + y * 12.0 - time * 1.5).sin();
                let local = Vec3::new(theta.cos() * r, y, theta.sin() * r) * (6.7 + pulse * 0.12);
                point(
                    world_rotation * local,
                    Rgbw::CYAN.mix(Rgbw::MAGENTA, pulse * 0.7),
                    0.62 + pulse * 0.5,
                    u,
                    40,
                )
            } else {
                let local = i - atmosphere_end;
                let moon_count = (count - atmosphere_end).max(1);
                let moon = local & 1;
                let moon_local = local / 2;
                let per_moon = moon_count.div_ceil(2).max(1);
                let sphere = {
                    let y = 1.0 - 2.0 * (moon_local as f32 + 0.5) / per_moon as f32;
                    let r = (1.0 - y * y).max(0.0).sqrt();
                    let theta = golden * moon_local as f32;
                    Vec3::new(theta.cos() * r, y, theta.sin() * r)
                };
                let orbit = time * if moon == 0 { 0.24 } else { -0.17 } + moon as f32 * 2.7;
                let center = Vec3::new(
                    orbit.cos() * (13.9 + moon as f32 * 1.6),
                    2.6 - moon as f32 * 5.1,
                    orbit.sin() * (13.9 + moon as f32 * 1.6),
                );
                let radius = if moon == 0 { 1.05 } else { 0.72 };
                point(
                    center + sphere * radius,
                    if moon == 0 {
                        Rgbw::new(0.52, 0.78, 1.0, 0.14)
                    } else {
                        Rgbw::GOLD
                    },
                    1.15,
                    moon_local as f32 / per_moon as f32,
                    50 + moon as u16,
                )
            }
        })
        .collect()
}

fn infinity_portal(count: usize, time: f32) -> Vec<FormationPoint> {
    let ribbon_end = count * 76 / 100;
    let halo_end = count * 92 / 100;
    (0..count)
        .map(|i| {
            if i < ribbon_end {
                let strand = i % 7;
                let local = i / 7;
                let along_count = ribbon_end.div_ceil(7).max(1);
                let u = (local as f32 + 0.5) / along_count as f32;
                let t = TAU * u;
                let denominator = 1.0 + t.sin().powi(2);
                let mut p = Vec3::new(
                    13.8 * t.cos() / denominator,
                    10.2 * t.sin() * t.cos() / denominator,
                    (strand as f32 - 3.0) * 0.42 / fleet_scale(count),
                );
                let crossing = 1.0 - smoothstep(0.0, 0.24, t.cos().abs());
                p.z += crossing * if t.sin() < 0.0 { -0.38 } else { 0.38 };
                p = Mat3::from_rotation_y(time * 0.08) * p;
                let wave = 0.5 + 0.5 * (t * 4.0 - time * 2.1 + strand as f32).sin();
                point(
                    p,
                    Rgbw::CYAN
                        .mix(Rgbw::MAGENTA, smoothstep(-1.0, 1.0, t.sin()))
                        .mix(Rgbw::GOLD, wave * 0.42),
                    1.02 + wave * 0.34,
                    u,
                    strand as u16,
                )
            } else if i < halo_end {
                let local = i - ribbon_end;
                let halo_count = (halo_end - ribbon_end).max(1);
                let ring = local % 3;
                let ring_local = local / 3;
                let per_ring = halo_count.div_ceil(3).max(1);
                let angle = TAU * (ring_local as f32 + 0.5 * ring as f32) / per_ring as f32
                    + time * (0.16 + ring as f32 * 0.035);
                let flat = Vec3::new(
                    angle.cos() * (13.0 + ring as f32 * 1.15),
                    0.0,
                    angle.sin() * (13.0 + ring as f32 * 1.15),
                );
                let tilt = Mat3::from_rotation_z((ring as f32 - 1.0) * 0.38)
                    * Mat3::from_rotation_x(0.45 + ring as f32 * 0.16);
                point(
                    tilt * flat,
                    [Rgbw::MAGENTA, Rgbw::GOLD, Rgbw::CYAN][ring],
                    0.95 + 0.28 * (angle * 5.0 - time).sin().max(0.0),
                    angle / TAU,
                    10 + ring as u16,
                )
            } else {
                let local = i - halo_end;
                let core_count = (count - halo_end).max(1);
                let y = 1.0 - 2.0 * (local as f32 + 0.5) / core_count as f32;
                let r = (1.0 - y * y).sqrt();
                let theta = PI * (3.0 - 5.0_f32.sqrt()) * local as f32 + time * 0.7;
                let pulse = 1.55 + 0.32 * (time * 2.4).sin();
                point(
                    Mat3::from_rotation_y(time * 0.08)
                        * (Vec3::new(theta.cos() * r, y, theta.sin() * r) * pulse + Vec3::Z * 3.0),
                    Rgbw::GOLD.mix(Rgbw::new(1.0, 0.92, 0.38, 0.3), 0.62),
                    1.55,
                    local as f32 / core_count as f32,
                    20,
                )
            }
        })
        .collect()
}

fn prismatic_lotus(count: usize, time: f32) -> Vec<FormationPoint> {
    let petal_end = count * 82 / 100;
    (0..count)
        .map(|i| {
            if i < petal_end {
                let petal = i % 12;
                let local = i / 12;
                let per_petal = petal_end.div_ceil(12).max(1);
                let cross_count = ((per_petal as f32 / 7.0).sqrt().ceil() as usize).max(3);
                let along_count = per_petal.div_ceil(cross_count).max(1);
                let u = (local / cross_count) as f32 / (along_count - 1).max(1) as f32;
                let across = (local % cross_count) as f32 / (cross_count - 1).max(1) as f32 - 0.5;
                let phase = petal as f32 * TAU / 12.0 + time * 0.075;
                let open = 0.82 + 0.16 * (time * 0.72 + petal as f32 * 0.43).sin();
                let radius = 1.2 + u * 10.8 * open;
                // Keep a real cross-section at both petal tips. Letting width
                // fall to zero assigned every cross-row to one coordinate and
                // caused thousands of unavoidable high-count collisions.
                let width = (0.34 + (PI * u).sin() * (1.0 + u * 4.2)) * across;
                let radial = Vec3::new(phase.cos(), 0.0, phase.sin());
                let tangent = Vec3::new(-phase.sin(), 0.0, phase.cos());
                let height = -3.2 + (PI * u).sin() * (5.1 + 0.7 * (petal & 1) as f32) - u * 1.25
                    + (u * 8.0 - time * 1.4).sin() * 0.22;
                let p = radial * radius + tangent * width + Vec3::Y * height;
                let hue = petal as f32 / 12.0;
                let color = Rgbw::MAGENTA
                    .mix(Rgbw::GOLD, smoothstep(0.0, 0.48, hue))
                    .mix(Rgbw::CYAN, smoothstep(0.45, 1.0, hue));
                point(
                    p,
                    color,
                    0.96 + 0.34 * (u * 9.0 - time * 2.0).sin().max(0.0),
                    u,
                    petal as u16,
                )
            } else {
                let local = i - petal_end;
                let core_count = (count - petal_end).max(1);
                let filament = local % 9;
                let serial = local / 9;
                let per_filament = core_count.div_ceil(9).max(1);
                let tube_cross_count = 3;
                let cross = serial % tube_cross_count;
                let along = serial / tube_cross_count;
                let along_count = per_filament.div_ceil(tube_cross_count).max(1);
                let u = (along as f32 + 0.5) / along_count as f32;
                let angle = filament as f32 * TAU / 9.0 + time * 0.32 + u * 2.4;
                let radius = 0.45 + u * 2.7;
                let tube_angle = cross as f32 * TAU / tube_cross_count as f32;
                let tube_radius = 0.3 / fleet_scale(count);
                let tangent = Vec3::new(-angle.sin(), 0.0, angle.cos());
                let tube_offset = tangent * tube_angle.cos() * tube_radius
                    + Vec3::Y * tube_angle.sin() * tube_radius;
                point(
                    Vec3::new(angle.cos() * radius, -1.2 + u * 5.2, angle.sin() * radius)
                        + tube_offset,
                    Rgbw::GOLD.mix(Rgbw::new(1.0, 0.94, 0.48, 0.28), u),
                    1.28 + 0.32 * (time * 2.8 + filament as f32).sin().max(0.0),
                    u,
                    20 + filament as u16,
                )
            }
        })
        .collect()
}

fn celestial_crown(count: usize, time: f32) -> Vec<FormationPoint> {
    let spire_end = count * 68 / 100;
    let halo_end = count * 89 / 100;
    (0..count)
        .map(|i| {
            if i < spire_end {
                let spire = i % 12;
                let local = i / 12;
                let per_spire = spire_end.div_ceil(12).max(1);
                let cross_count = ((per_spire as f32 / 9.0).sqrt().ceil() as usize).max(3);
                let along_count = per_spire.div_ceil(cross_count).max(1);
                let u = (local / cross_count) as f32 / (along_count - 1).max(1) as f32;
                let across = (local % cross_count) as f32 / (cross_count - 1).max(1) as f32 - 0.5;
                let angle = spire as f32 * TAU / 12.0 + time * 0.095;
                let high = if spire % 3 == 0 { 3.0 } else { 0.0 };
                let radius = 8.6 - (PI * u).sin() * 2.4 + across * 1.1;
                let y = -4.5 + u * (10.5 + high) + (PI * u).sin() * 2.1;
                let twist = angle + u * 0.42;
                let p = Vec3::new(twist.cos() * radius, y, twist.sin() * radius);
                let color = if spire % 3 == 0 {
                    Rgbw::GOLD
                } else if spire & 1 == 0 {
                    Rgbw::MAGENTA
                } else {
                    Rgbw::CYAN
                };
                point(
                    p,
                    color,
                    1.0 + 0.36 * (u * 12.0 - time * 2.4).sin().max(0.0),
                    u,
                    spire as u16,
                )
            } else if i < halo_end {
                let local = i - spire_end;
                let halo_count = (halo_end - spire_end).max(1);
                let ring = local % 4;
                let ring_local = local / 4;
                let per_ring = halo_count.div_ceil(4).max(1);
                let angle =
                    TAU * ring_local as f32 / per_ring as f32 + time * (0.12 + ring as f32 * 0.035);
                let radius = 6.0 + ring as f32 * 1.85;
                let flat = Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius);
                let tilt = Mat3::from_rotation_z((ring as f32 - 1.5) * 0.18)
                    * Mat3::from_rotation_x(0.2 + ring as f32 * 0.12);
                point(
                    tilt * flat + Vec3::Y * (1.5 + ring as f32 * 0.9),
                    Rgbw::GOLD.mix(Rgbw::CYAN, ring as f32 / 3.0),
                    1.08,
                    angle / TAU,
                    20 + ring as u16,
                )
            } else {
                let local = i - halo_end;
                let core_count = (count - halo_end).max(1);
                let arm = local % 6;
                let along = local / 6;
                let per_arm = core_count.div_ceil(6).max(1);
                let u = (along as f32 + 0.5) / per_arm as f32;
                let angle = arm as f32 * TAU / 6.0 + u * 5.4 - time * 0.48;
                let radius = 0.55 + (1.0 - u) * 4.25;
                point(
                    Vec3::new(angle.cos() * radius, -1.0 + u * 9.5, angle.sin() * radius),
                    Rgbw::MAGENTA.mix(Rgbw::GOLD, u),
                    1.18 + 0.42 * (u * 15.0 - time * 3.0).sin().max(0.0),
                    u,
                    30 + arm as u16,
                )
            }
        })
        .collect()
}

fn event_horizon(count: usize, time: f32) -> Vec<FormationPoint> {
    let disk_end = count * 70 / 100;
    let photon_end = count * 85 / 100;
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    (0..count)
        .map(|index| {
            if index < disk_end {
                let cross_count = ((disk_end.max(1) as f32 / 18.0).sqrt().ceil() as usize).max(4);
                let along_count = disk_end.div_ceil(cross_count).max(1);
                let along = index / cross_count;
                let across = index % cross_count;
                let u = (along as f32 + 0.5) / along_count as f32;
                let v = (across as f32 + 0.5) / cross_count as f32;
                let radius = 3.6 + v * 8.8;
                let angular_speed = 0.18 + (1.0 - v) * 0.42;
                let angle = u * TAU + time * angular_speed + v * 0.22;
                let wave = (angle * 3.0 - time * 1.7 + v * 8.0).sin();
                let flat = Vec3::new(
                    angle.cos() * radius,
                    (v - 0.5) * 0.82 + wave * (0.1 + (1.0 - v) * 0.22),
                    angle.sin() * radius,
                );
                let p = Mat3::from_rotation_z(0.16) * flat;
                let inner = Rgbw::new(1.0, 0.16, 0.015, 0.16);
                let outer = Rgbw::CYAN.mix(Rgbw::MAGENTA, v * 0.7);
                point(
                    p,
                    inner.mix(outer, v),
                    1.08 + 0.48 * wave.max(0.0),
                    u,
                    (v * 12.0) as u16,
                )
            } else if index < photon_end {
                let local = index - disk_end;
                let sphere_count = (photon_end - disk_end).max(1);
                let y = 1.0 - 2.0 * (local as f32 + 0.5) / sphere_count as f32;
                let radial = (1.0 - y * y).max(0.0).sqrt();
                let angle = golden * local as f32 + time * 0.34;
                let sphere = Vec3::new(angle.cos() * radial, y, angle.sin() * radial);
                point(
                    sphere * 2.15,
                    Rgbw::MAGENTA.mix(Rgbw::CYAN, radial),
                    1.4 + 0.4 * (angle * 0.5 - time * 1.8).sin().max(0.0),
                    local as f32 / sphere_count as f32,
                    20,
                )
            } else {
                let local = index - photon_end;
                let jet_count = (count - photon_end).max(1);
                let side = if local & 1 == 0 { -1.0 } else { 1.0 };
                let jet_local = local / 2;
                let per_side = jet_count.div_ceil(2).max(1);
                let cross_count = ((per_side as f32 / 12.0).sqrt().ceil() as usize).max(3);
                let along_count = per_side.div_ceil(cross_count).max(1);
                let u = (jet_local / cross_count) as f32 / (along_count - 1).max(1) as f32;
                let across =
                    (jet_local % cross_count) as f32 / (cross_count - 1).max(1) as f32 - 0.5;
                let angle = time * 1.3 + u * 11.0 + across * PI;
                let radius = 0.62 + across * 0.5 + u * 0.38;
                point(
                    Vec3::new(
                        angle.cos() * radius,
                        side * (3.5 + u * 11.5),
                        angle.sin() * radius,
                    ),
                    Rgbw::CYAN.mix(Rgbw::new(0.45, 0.16, 1.0, 0.2), u),
                    1.15 + 0.5 * (u * 18.0 - time * 4.0).sin().max(0.0),
                    u,
                    if side < 0.0 { 30 } else { 31 },
                )
            }
        })
        .collect()
}

fn spectral_mandala(count: usize, time: f32) -> Vec<FormationPoint> {
    let petal_end = count * 72 / 100;
    let halo_end = count * 90 / 100;
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    (0..count)
        .map(|index| {
            if index < petal_end {
                let petal = index % 18;
                let local = index / 18;
                let per_petal = petal_end.div_ceil(18).max(1);
                let cross_count = ((per_petal as f32 / 9.0).sqrt().ceil() as usize).max(3);
                let along_count = per_petal.div_ceil(cross_count).max(1);
                let u = (local / cross_count) as f32 / (along_count - 1).max(1) as f32;
                let across = (local % cross_count) as f32 / (cross_count - 1).max(1) as f32 - 0.5;
                let petal_angle = petal as f32 * TAU / 18.0 + time * 0.075;
                let opening = across * (0.13 + (PI * u).sin() * 0.2);
                let angle = petal_angle + opening;
                let radius = 2.85 + u * 9.6;
                let depth = (petal_angle * 3.0 - time * 0.42).sin() * (PI * u).sin() * 1.18;
                let pulse = (u * 16.0 - time * 2.6 + petal_angle).sin();
                point(
                    Vec3::new(angle.cos() * radius, angle.sin() * radius, depth),
                    Rgbw::MAGENTA
                        .mix(Rgbw::GOLD, u)
                        .mix(Rgbw::CYAN, (petal % 3) as f32 * 0.22),
                    1.02 + 0.42 * pulse.max(0.0),
                    u,
                    petal as u16,
                )
            } else if index < halo_end {
                let local = index - petal_end;
                let halo_count = (halo_end - petal_end).max(1);
                let ring = local % 5;
                let ring_local = local / 5;
                let per_ring = halo_count.div_ceil(5).max(1);
                let angle = TAU * (ring_local as f32 + 0.5) / per_ring as f32
                    - time * (0.11 + ring as f32 * 0.025);
                let radius = 3.8 + ring as f32 * 1.7;
                point(
                    Vec3::new(
                        angle.cos() * radius,
                        angle.sin() * radius,
                        3.0 + (ring as f32 - 2.0) * 0.82,
                    ),
                    Rgbw::CYAN.mix(Rgbw::GOLD, ring as f32 / 4.0),
                    1.16,
                    angle / TAU,
                    20 + ring as u16,
                )
            } else {
                let local = index - halo_end;
                let core_count = (count - halo_end).max(1);
                let y = 1.0 - 2.0 * (local as f32 + 0.5) / core_count as f32;
                let radial = (1.0 - y * y).max(0.0).sqrt();
                let angle = golden * local as f32 - time * 0.5;
                point(
                    Vec3::new(angle.cos() * radial, y, angle.sin() * radial) * 1.55,
                    Rgbw::new(0.72, 0.26, 1.0, 0.34),
                    1.45 + 0.3 * (time * 2.2 + angle).sin().max(0.0),
                    local as f32 / core_count as f32,
                    30,
                )
            }
        })
        .collect()
}

fn chrono_gyroscope(count: usize, time: f32) -> Vec<FormationPoint> {
    let ring_end = count * 86 / 100;
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    (0..count)
        .map(|index| {
            if index < ring_end {
                let ring = index % 8;
                let local = index / 8;
                let per_ring = ring_end.div_ceil(8).max(1);
                let cross_count = ((per_ring as f32 / 12.0).sqrt().ceil() as usize).max(4);
                let along_count = per_ring.div_ceil(cross_count).max(1);
                let along = local / cross_count;
                let across = local % cross_count;
                let u = TAU * (along as f32 + 0.5) / along_count as f32
                    + time * (0.1 + ring as f32 * 0.018);
                let v = TAU * (across as f32 + 0.5) / cross_count as f32;
                let major = 3.2 + ring as f32 * 1.16;
                let minor = 0.3;
                let torus = Vec3::new(
                    (major + minor * v.cos()) * u.cos(),
                    (major + minor * v.cos()) * u.sin(),
                    minor * v.sin(),
                );
                let tilt = Mat3::from_rotation_y(time * 0.045 + ring as f32 * 0.28)
                    * Mat3::from_rotation_x(0.22 + ring as f32 * 0.19);
                let sweep = (u * 3.0 - time * 2.0 + ring as f32).sin();
                point(
                    tilt * torus,
                    Rgbw::CYAN
                        .mix(Rgbw::MAGENTA, ring as f32 / 7.0)
                        .mix(Rgbw::GOLD, sweep.max(0.0) * 0.3),
                    1.04 + 0.38 * sweep.max(0.0),
                    u / TAU,
                    ring as u16,
                )
            } else {
                let local = index - ring_end;
                let core_count = (count - ring_end).max(1);
                let y = 1.0 - 2.0 * (local as f32 + 0.5) / core_count as f32;
                let radial = (1.0 - y * y).max(0.0).sqrt();
                let angle = golden * local as f32 + time * 0.8;
                let pulse = 1.38 + 0.12 * (time * 1.7).sin();
                point(
                    Vec3::new(angle.cos() * radial, y, angle.sin() * radial) * pulse,
                    Rgbw::GOLD.mix(Rgbw::new(0.5, 0.12, 1.0, 0.28), radial),
                    1.42,
                    local as f32 / core_count as f32,
                    20,
                )
            }
        })
        .collect()
}

#[allow(dead_code)]
fn solar_phoenix(count: usize, time: f32) -> Vec<FormationPoint> {
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    let wing_count = count * 68 / 100;
    let body_count = count * 17 / 100;
    let body_end = wing_count + body_count;
    let flap = (time * 1.24).sin();
    (0..count)
        .map(|i| {
            if i < wing_count {
                let side = if i % 2 == 0 { -1.0 } else { 1.0 };
                let local = i / 2;
                let per_side = wing_count.div_ceil(2).max(1);
                let columns = (per_side as f32).sqrt().ceil() as usize;
                let rows = per_side.div_ceil(columns).max(1);
                let u = (local % columns) as f32 / (columns - 1).max(1) as f32;
                let v = (local / columns) as f32 / (rows - 1).max(1) as f32;
                let feather = (v * 7.0).floor() as u16;
                let ripple = (u * 10.0 - time * 2.2 + v * PI).sin() * 0.32;
                let p = Vec3::new(
                    side * (1.95 + u * 11.3),
                    2.4 + (u * PI).sin() * 4.4
                        + flap * (0.2 + u.powf(1.45)) * 2.3
                        + (v - 0.5) * (2.4 + u * 5.2),
                    (v - 0.5) * (1.5 + u * 4.8) + ripple,
                );
                let root = Rgbw::new(1.0, 0.16, 0.015, 0.04);
                let tip = if side < 0.0 {
                    Rgbw::MAGENTA
                } else {
                    Rgbw::GOLD
                };
                point(
                    p,
                    root.mix(tip, 0.2 + u * 0.8),
                    1.02 + 0.2 * (time * 2.0 + u * TAU + v).sin().max(0.0),
                    u,
                    feather + if side < 0.0 { 0 } else { 8 },
                )
            } else if i < body_end {
                let local = i - wing_count;
                let torso_count = (body_count * 70 / 100).max(1);
                let head_count = (body_count * 20 / 100).max(1);
                if local < torso_count {
                    let u = (local as f32 + 0.5) / torso_count as f32;
                    let y = 1.0 - 2.0 * u;
                    let r = (1.0 - y * y).sqrt();
                    let theta = golden * local as f32;
                    let sphere = Vec3::new(theta.cos() * r, y, theta.sin() * r);
                    point(
                        Vec3::new(sphere.x * 1.7, 0.8 + sphere.y * 3.6, sphere.z * 1.3),
                        Rgbw::GOLD.mix(Rgbw::new(1.0, 0.06, 0.18, 0.04), u),
                        1.14,
                        u,
                        17,
                    )
                } else if local < torso_count + head_count {
                    let head_local = local - torso_count;
                    let u = (head_local as f32 + 0.5) / head_count as f32;
                    let y = 1.0 - 2.0 * u;
                    let r = (1.0 - y * y).sqrt();
                    let theta = golden * head_local as f32;
                    let sphere = Vec3::new(theta.cos() * r, y, theta.sin() * r);
                    point(
                        Vec3::new(sphere.x * 1.15, 6.3 + sphere.y * 1.15, sphere.z),
                        Rgbw::new(1.0, 0.42, 0.025, 0.03),
                        1.35,
                        u,
                        16,
                    )
                } else {
                    let neck_local = local - torso_count - head_count;
                    let neck_count = (body_count - torso_count - head_count).max(1);
                    let around_count = (neck_count as f32 * 1.5).sqrt().ceil() as usize;
                    let height_count = neck_count.div_ceil(around_count).max(1);
                    let angle =
                        TAU * (neck_local % around_count) as f32 / around_count.max(1) as f32;
                    let u = (neck_local / around_count) as f32 / (height_count - 1).max(1) as f32;
                    point(
                        Vec3::new(angle.cos() * 0.62, 4.48 + u * 0.66, angle.sin() * 0.62),
                        Rgbw::GOLD,
                        1.24,
                        u,
                        18,
                    )
                }
            } else {
                let tail_count = (count - body_end).max(1);
                let local = i - body_end;
                let plume = local % 3;
                let ribbon_index = local / 3;
                let per_plume = tail_count.div_ceil(3).max(1);
                let cross_count = ((per_plume as f32 / 5.0).sqrt().ceil() as usize).max(2);
                let along_count = per_plume.div_ceil(cross_count).max(1);
                let u = (ribbon_index / cross_count) as f32 / (along_count - 1).max(1) as f32;
                let across =
                    (ribbon_index % cross_count) as f32 / (cross_count - 1).max(1) as f32 - 0.5;
                let curl = u * (4.8 + plume as f32 * 0.55) + time * 0.72 + plume as f32 * 2.0;
                let p = Vec3::new(
                    (plume as f32 - 1.0) * (1.3 + u * 1.7)
                        + curl.sin() * u * 0.72
                        + across * (0.7 - u * 0.25),
                    -3.15 - u * (4.8 + (plume as f32 - 1.0).abs() * 0.7) + curl.cos() * 0.35,
                    1.2 + u * 3.5 + curl.sin() * 0.5 + across * 0.7,
                );
                point(
                    p,
                    Rgbw::GOLD.mix(Rgbw::MAGENTA, u * 0.88),
                    1.0 + 0.24 * (time * 2.4 + u * 8.0).sin().max(0.0),
                    u,
                    19 + plume as u16,
                )
            }
        })
        .collect()
}

fn point(position: Vec3, color: Rgbw, brightness: f32, phase: f32, group: u16) -> FormationPoint {
    FormationPoint {
        position,
        color,
        brightness,
        phase,
        group,
    }
}

fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Enlarges surface formations with the square root of fleet size so their
/// physical area grows in proportion to the number of aircraft. This is the
/// safety-first scale used by CPU and GPU choreography alike.
pub fn fleet_scale(count: usize) -> f32 {
    // 35% operational margin keeps the 0.24 m physical envelope clear even
    // after animation, quantization and imperfect surface parameterization.
    ((count.max(1) as f32 / 384.0).sqrt() * 1.35).clamp(1.0, 80.0)
}

/// Returns the density-scaled vertical offset used to keep each normalized
/// formation clear of the ground. GPU generation cannot scan every point for
/// a minimum, so these conservative extents mirror the procedural shapes.
pub fn formation_lift(kind: FormationKind, count: usize) -> f32 {
    let lower_extent = match kind {
        FormationKind::LaunchGrid => return 0.0,
        FormationKind::Chrysalis => 11.4,
        FormationKind::Heart => 9.6,
        FormationKind::Galaxy => 8.5,
        FormationKind::Cathedral => 7.2,
        FormationKind::Human => 8.8,
        FormationKind::Planet => 6.9,
        FormationKind::Infinity => 12.0,
        FormationKind::Lotus => 4.9,
        FormationKind::Crown => 5.2,
        FormationKind::EventHorizon => 15.1,
        FormationKind::Mandala => 12.8,
        FormationKind::Gyroscope => 12.0,
        FormationKind::Image => 9.5,
    };
    (lower_extent * fleet_scale(count) + 3.0).max(15.0)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn generators_are_deterministic_and_sized() {
        for kind in FormationKind::SHOWCASE {
            let a = generate(kind, 257, 1.25);
            let b = generate(kind, 257, 1.25);
            assert_eq!(a, b);
            assert_eq!(a.len(), 257);
            assert!(a.iter().all(|p| p.position.is_finite()));
        }
    }

    #[test]
    fn high_density_generators_remain_well_distributed() {
        let galaxy = generate(FormationKind::Galaxy, 5_000, 1.25);
        let vertical_extent = galaxy.iter().map(|p| p.position.y).fold(f32::MIN, f32::max)
            - galaxy.iter().map(|p| p.position.y).fold(f32::MAX, f32::min);
        assert!(vertical_extent > 10.0);
    }

    #[test]
    fn fifty_thousand_drone_chrysalis_preserves_layer_structure() {
        let dense = generate(FormationKind::Chrysalis, 50_000, 2.0);
        for group in [0, 11, 20, 30, 33] {
            assert!(dense.iter().any(|point| point.group == group));
        }
        assert!(dense.iter().all(|point| point.position.is_finite()));
    }

    #[test]
    fn seventeen_thousand_drone_showcase_has_no_collapsed_slots() {
        type SlotCell = HashMap<(i32, i32, i32), Vec<(Vec3, usize, u16)>>;

        let count = 17_000;
        let scale = fleet_scale(count);
        let cell_size = 0.38;
        for kind in FormationKind::SHOWCASE {
            let mut cells = SlotCell::new();
            let mut minimum_distance = f32::INFINITY;
            let mut closest = None;
            for (index, point) in generate(kind, count, 94.2).into_iter().enumerate() {
                let position = point.position * scale;
                let cell = (
                    (position.x / cell_size).floor() as i32,
                    (position.y / cell_size).floor() as i32,
                    (position.z / cell_size).floor() as i32,
                );
                for x in -1..=1 {
                    for y in -1..=1 {
                        for z in -1..=1 {
                            if let Some(neighbors) =
                                cells.get(&(cell.0 + x, cell.1 + y, cell.2 + z))
                            {
                                for (neighbor, other, group) in neighbors {
                                    let distance = position.distance(*neighbor);
                                    if distance < minimum_distance {
                                        minimum_distance = distance;
                                        closest = Some((index, *other, point.group, *group));
                                    }
                                }
                            }
                        }
                    }
                }
                cells
                    .entry(cell)
                    .or_default()
                    .push((position, index, point.group));
            }
            assert!(
                minimum_distance >= 0.24,
                "{} contains a {minimum_distance:.4}m physical overlap at {closest:?}",
                kind.label()
            );
        }
    }

    #[test]
    fn upgraded_planet_and_new_showpieces_have_distinct_layers() {
        let planet = generate(FormationKind::Planet, 10_000, 3.0);
        for group in [0, 40, 50, 51] {
            assert!(planet.iter().any(|point| point.group == group));
        }
        for kind in [
            FormationKind::Infinity,
            FormationKind::Lotus,
            FormationKind::Crown,
        ] {
            let points = generate(kind, 10_000, 3.0);
            let minimum = points.iter().fold(Vec3::splat(f32::MAX), |extent, point| {
                extent.min(point.position)
            });
            let maximum = points.iter().fold(Vec3::splat(f32::MIN), |extent, point| {
                extent.max(point.position)
            });
            assert!((maximum - minimum).max_element() > 16.0);
            assert!(points.iter().all(|point| point.position.is_finite()));
        }
    }

    #[test]
    fn gpu_lift_extents_keep_every_showcase_clear_of_ground() {
        let count = 50_000;
        let scale = fleet_scale(count);
        for kind in FormationKind::SHOWCASE {
            for time in [0.0, 3.0, 7.0] {
                let minimum_y = generate(kind, count, time)
                    .iter()
                    .map(|point| point.position.y)
                    .fold(f32::INFINITY, f32::min);
                let clearance = minimum_y * scale + formation_lift(kind, count);
                assert!(
                    clearance >= 2.9,
                    "{} falls to {clearance:.3}m at t={time}",
                    kind.label()
                );
            }
        }
    }
}
