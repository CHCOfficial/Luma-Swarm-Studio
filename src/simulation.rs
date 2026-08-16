use std::{collections::HashMap, time::Instant};

use glam::{Quat, Vec3};

use crate::{
    assignment, formation,
    model::{Drone, FormationKind, FormationPoint, Rgbw, SimulationSettings},
    timeline::{CueKind, ShowTimeline},
    trajectory::Trajectory,
};

pub const DRONE_COLLISION_ENVELOPE: f32 = 0.24;
pub const DRONE_WARNING_ENVELOPE: f32 = 0.38;
pub const DRONE_GROUND_RADIUS: f32 = 0.14;

#[derive(Clone, Copy, Debug)]
pub struct SafetyStats {
    pub minimum_air_separation: f32,
    pub minimum_ground_clearance: f32,
    pub near_miss_pairs: u32,
    pub collision_pairs: u32,
    pub ground_breaches: u32,
    pub incident_events: u64,
    pub monitored_steps: u64,
    pub last_incident_time: Option<f32>,
    pub closest_pair: Option<(u32, u32)>,
    pub closest_pair_position: Option<Vec3>,
    pub lowest_drone: Option<u32>,
    pub lowest_drone_position: Option<Vec3>,
}

impl Default for SafetyStats {
    fn default() -> Self {
        Self {
            minimum_air_separation: DRONE_WARNING_ENVELOPE,
            minimum_ground_clearance: f32::INFINITY,
            near_miss_pairs: 0,
            collision_pairs: 0,
            ground_breaches: 0,
            incident_events: 0,
            monitored_steps: 0,
            last_incident_time: None,
            closest_pair: None,
            closest_pair_position: None,
            lowest_drone: None,
            lowest_drone_position: None,
        }
    }
}

impl SafetyStats {
    pub fn is_clear(self) -> bool {
        self.collision_pairs == 0 && self.ground_breaches == 0
    }
}

pub struct FleetSimulation {
    pub drones: Vec<Drone>,
    pub settings: SimulationSettings,
    accumulator: f32,
    active_cue: usize,
    trajectories: Vec<Trajectory>,
    targets: Vec<FormationPoint>,
    target_velocities: Vec<Vec3>,
    target_accelerations: Vec<Vec3>,
    target_phase_lock: Vec<f32>,
    separation_offsets: Vec<Vec3>,
    tracking_error_rms: f32,
    spatial_grid: HashMap<(i32, i32, i32), Vec<usize>>,
    safety: SafetyStats,
    safety_flags: Vec<u8>,
    previous_step_unsafe: bool,
    custom_formation: Vec<FormationPoint>,
    collision_pairs_scratch: Vec<(usize, usize)>,
    constraint_iterations_last_frame: u32,
    simulation_cpu_ms: f32,
}

impl FleetSimulation {
    pub fn new(count: usize, settings: SimulationSettings) -> Self {
        let mut simulation = Self {
            drones: Vec::new(),
            settings,
            accumulator: 0.0,
            active_cue: usize::MAX,
            trajectories: Vec::new(),
            targets: Vec::new(),
            target_velocities: Vec::new(),
            target_accelerations: Vec::new(),
            target_phase_lock: Vec::new(),
            separation_offsets: Vec::new(),
            tracking_error_rms: 0.0,
            spatial_grid: HashMap::new(),
            safety: SafetyStats::default(),
            safety_flags: Vec::new(),
            previous_step_unsafe: false,
            custom_formation: Vec::new(),
            collision_pairs_scratch: Vec::new(),
            constraint_iterations_last_frame: 0,
            simulation_cpu_ms: 0.0,
        };
        simulation.resize(count);
        simulation
    }

    pub fn resize(&mut self, count: usize) {
        let pads = formation::launch_grid(count);
        self.drones = pads
            .iter()
            .enumerate()
            .map(|(index, pad)| Drone {
                id: index as u32,
                position: pad.position,
                previous_position: pad.position,
                velocity: Vec3::ZERO,
                acceleration: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                slot: index,
                color: Rgbw::OFF,
                brightness: 0.0,
                phase: pad.phase,
                rotor_angle: 0.0,
                battery: 1.0,
            })
            .collect();
        self.targets = pads;
        self.target_velocities = vec![Vec3::ZERO; count];
        self.target_accelerations = vec![Vec3::ZERO; count];
        self.target_phase_lock = vec![0.0; count];
        self.separation_offsets = vec![Vec3::ZERO; count];
        self.tracking_error_rms = 0.0;
        self.active_cue = usize::MAX;
        self.accumulator = 0.0;
        self.safety = SafetyStats::default();
        self.safety_flags = vec![0; count];
        self.previous_step_unsafe = false;
        self.constraint_iterations_last_frame = 0;
        self.simulation_cpu_ms = 0.0;
        self.rebuild_spatial_grid();
        self.update_safety(0.0);
    }

    pub fn reset(&mut self) {
        self.resize(self.drones.len());
    }

    pub fn step_frame(
        &mut self,
        frame_dt: f32,
        show_time: f32,
        timeline: &ShowTimeline,
        wind_direction_degrees: f32,
    ) {
        let started = Instant::now();
        let fixed_dt = 1.0 / self.settings.simulation_hz.clamp(30, 240) as f32;
        // Show time is already advanced by wall time before this call. More
        // than two identical-time catch-up steps creates a feedback spiral
        // under load without improving timeline accuracy.
        self.accumulator = (self.accumulator + frame_dt.min(0.1)).min(fixed_dt * 2.0);
        self.constraint_iterations_last_frame = 0;
        while self.accumulator >= fixed_dt {
            self.constraint_iterations_last_frame +=
                self.step_fixed(fixed_dt, show_time, timeline, wind_direction_degrees) as u32;
            self.accumulator -= fixed_dt;
        }
        self.simulation_cpu_ms = started.elapsed().as_secs_f32() * 1_000.0;
    }

    pub fn seek(&mut self, show_time: f32, timeline: &ShowTimeline) {
        self.reset();
        let sample = timeline.sample(show_time);
        self.active_cue = usize::MAX;
        self.prepare_cue(show_time, timeline);
        let fixed_dt = 1.0 / self.settings.simulation_hz.clamp(30, 240) as f32;
        self.update_targets(show_time, timeline, fixed_dt);
        for (drone, target) in self.drones.iter_mut().zip(&self.targets) {
            drone.position = target.position;
            drone.previous_position = target.position;
            drone.velocity = Vec3::ZERO;
            drone.color = target.color;
            drone.brightness = target.brightness;
        }
        self.target_velocities.fill(Vec3::ZERO);
        self.target_accelerations.fill(Vec3::ZERO);
        self.target_phase_lock.fill(0.0);
        self.active_cue = sample.index;
        self.resolve_collision_constraints(fixed_dt, 64);
        self.rebuild_spatial_grid();
        self.update_safety(show_time);
        // A seek is an editor teleport, not flown telemetry. Preserve the live
        // clearance reading but start incident history from the next simulated
        // fixed step so scrubbing cannot fabricate an in-flight event.
        self.clear_safety_history();
    }

    pub fn interpolated_position(&self, drone: &Drone) -> Vec3 {
        let fixed_dt = 1.0 / self.settings.simulation_hz.max(1) as f32;
        drone.previous_position.lerp(
            drone.position,
            (self.accumulator / fixed_dt).clamp(0.0, 1.0),
        )
    }

    /// The editor's separation value is calibrated around the default fleet.
    /// Dense shows reduce the real-time correction radius so collision forces
    /// cannot tear drones away from valid high-density formation slots.
    pub fn effective_separation(&self) -> f32 {
        let density_scale = (384.0 / self.drones.len().max(1) as f32)
            .powf(0.42)
            .clamp(0.28, 1.25);
        (self.settings.minimum_separation * density_scale).max(0.08)
    }

    pub fn tracking_error_rms(&self) -> f32 {
        self.tracking_error_rms
    }

    pub fn constraint_iterations_last_frame(&self) -> u32 {
        self.constraint_iterations_last_frame
    }

    pub fn simulation_cpu_ms(&self) -> f32 {
        self.simulation_cpu_ms
    }

    pub fn safety_stats(&self) -> SafetyStats {
        self.safety
    }

    pub fn safety_level(&self, index: usize) -> u8 {
        self.safety_flags.get(index).copied().unwrap_or(0)
    }

    pub fn clear_safety_history(&mut self) {
        self.safety.incident_events = 0;
        self.safety.last_incident_time = None;
        self.previous_step_unsafe = false;
    }

    pub fn set_custom_formation(&mut self, points: Vec<FormationPoint>) {
        self.custom_formation = points;
        self.active_cue = usize::MAX;
    }

    pub fn update_custom_formation_colors(&mut self, points: &[FormationPoint]) {
        if points.is_empty() || self.custom_formation.is_empty() {
            return;
        }
        let source_len = points.len();
        let destination_len = self.custom_formation.len();
        for (index, destination) in self.custom_formation.iter_mut().enumerate() {
            let source = index * source_len / destination_len.max(1);
            let source = &points[source.min(source_len - 1)];
            destination.color = source.color;
            destination.brightness = source.brightness;
        }
    }

    fn step_fixed(
        &mut self,
        dt: f32,
        show_time: f32,
        timeline: &ShowTimeline,
        wind_direction_degrees: f32,
    ) -> usize {
        let sample = timeline.sample(show_time);
        if sample.index != self.active_cue {
            self.prepare_cue(show_time, timeline);
        }
        self.update_targets(show_time, timeline, dt);
        let wind_angle = wind_direction_degrees.to_radians();
        let wind_base =
            Vec3::new(wind_angle.cos(), 0.0, wind_angle.sin()) * self.settings.wind_strength;
        let mut squared_tracking_error = 0.0;
        for index in 0..self.drones.len() {
            let target = &self.targets[index];
            let drone = &self.drones[index];
            let error = target.position - drone.position;
            let feed_forward_velocity = self.target_velocities[index];
            let feed_forward_acceleration = self.target_accelerations[index];
            let phase_lock = self.target_phase_lock[index];
            let speed_limit = self
                .settings
                .max_speed
                .max(feed_forward_velocity.length() * 1.12);
            let desired_velocity = (feed_forward_velocity
                + error * self.settings.stabilization * 0.9)
                .clamp_length_max(speed_limit);
            let mut acceleration = feed_forward_acceleration
                + (desired_velocity - drone.velocity) * self.settings.turn_rate;
            acceleration +=
                self.avoidance_force(index) * self.settings.max_acceleration * (1.0 - phase_lock);
            let gust = Vec3::new(
                (show_time * 0.71 + drone.phase * 11.0).sin(),
                (show_time * 0.43 + drone.phase * 17.0).sin() * 0.18,
                (show_time * 0.57 + drone.phase * 7.0).cos(),
            );
            acceleration += wind_base + gust * self.settings.wind_strength * 0.32;
            let acceleration_limit = self
                .settings
                .max_acceleration
                .max(feed_forward_acceleration.length() * 1.12);
            acceleration = acceleration.clamp_length_max(acceleration_limit);

            let drone = &mut self.drones[index];
            drone.previous_position = drone.position;
            drone.acceleration = acceleration;
            drone.velocity = (drone.velocity + acceleration * dt).clamp_length_max(speed_limit);
            drone.position += drone.velocity * dt;
            if phase_lock > 0.0 {
                drone.position = drone.position.lerp(target.position, phase_lock);
                drone.velocity = drone.velocity.lerp(feed_forward_velocity, phase_lock);
            }
            squared_tracking_error += drone.position.distance_squared(target.position);

            let movement_tilt = Vec3::new(-acceleration.x * 0.045, 1.0, -acceleration.z * 0.045)
                .normalize_or_zero();
            let desired_orientation = Quat::from_rotation_arc(Vec3::Y, movement_tilt);
            drone.orientation = drone
                .orientation
                .slerp(desired_orientation, 1.0 - (-dt * 5.0).exp());
            drone.rotor_angle = (drone.rotor_angle + dt * (36.0 + drone.velocity.length() * 2.0))
                % std::f32::consts::TAU;
            drone.color = drone.color.mix(target.color, 1.0 - (-dt * 8.0).exp());
            drone.brightness += (target.brightness - drone.brightness) * (1.0 - (-dt * 7.0).exp());
            drone.battery = (drone.battery - dt / (18.0 * 60.0)).max(0.0);
        }
        self.tracking_error_rms = (squared_tracking_error / self.drones.len().max(1) as f32).sqrt();
        let constraint_iterations = self.resolve_collision_constraints(dt, 3);
        self.rebuild_spatial_grid();
        self.update_safety(show_time);
        constraint_iterations
    }

    fn prepare_cue(&mut self, show_time: f32, timeline: &ShowTimeline) {
        let sample = timeline.sample(show_time);
        let requires_assignment = self.active_cue == usize::MAX
            || matches!(
                sample.cue.kind,
                CueKind::Launch { .. } | CueKind::Transition { .. } | CueKind::Landing
            );
        if !requires_assignment {
            self.target_velocities.fill(Vec3::ZERO);
            self.target_accelerations.fill(Vec3::ZERO);
            self.target_phase_lock.fill(0.0);
            self.active_cue = sample.index;
            return;
        }
        let target_kind = match sample.cue.kind {
            CueKind::Landing => FormationKind::LaunchGrid,
            _ => sample
                .cue
                .kind
                .formation()
                .unwrap_or(sample.previous_formation),
        };
        // Formation animation is keyed to the continuous show clock. Using the
        // cue-local clock here reset animated geometry at morph/animate seams.
        let mut destination = self.generate_formation(target_kind, show_time);
        if target_kind != FormationKind::LaunchGrid {
            let scale = formation::fleet_scale(self.drones.len());
            for point in &mut destination {
                point.position *= scale;
            }
            lift_formation_above_ground(&mut destination);
        }
        let source_positions: Vec<_> = self.drones.iter().map(|drone| drone.position).collect();
        let destination_positions: Vec<_> =
            destination.iter().map(|point| point.position).collect();
        let assignment = assignment::assign(&source_positions, &destination_positions);
        let average_distance = source_positions
            .iter()
            .enumerate()
            .map(|(index, source)| source.distance(destination_positions[assignment[index]]))
            .sum::<f32>()
            / self.drones.len().max(1) as f32;
        let shared_clearance = (average_distance * 0.14).clamp(0.8, 4.5);
        let lane_coherence = (384.0 / self.drones.len().max(1) as f32)
            .powf(0.65)
            .clamp(0.04, 1.0);
        self.trajectories.clear();
        self.trajectories.reserve(self.drones.len());
        self.targets.clear();
        self.targets.reserve(self.drones.len());
        for (index, drone) in self.drones.iter_mut().enumerate() {
            let slot = assignment[index];
            drone.slot = slot;
            let mut trajectory =
                Trajectory::new(drone.position, destination[slot].position, drone.id);
            trajectory.clearance = shared_clearance;
            trajectory.lateral_lane *= lane_coherence;
            self.trajectories.push(trajectory);
            let mut initial_target = destination[slot].clone();
            initial_target.position = drone.position;
            self.targets.push(initial_target);
        }
        self.target_velocities.resize(self.drones.len(), Vec3::ZERO);
        self.target_accelerations
            .resize(self.drones.len(), Vec3::ZERO);
        self.target_phase_lock.resize(self.drones.len(), 0.0);
        self.target_velocities.fill(Vec3::ZERO);
        self.target_accelerations.fill(Vec3::ZERO);
        self.target_phase_lock.fill(0.0);
        self.separation_offsets.fill(Vec3::ZERO);
        self.active_cue = sample.index;
    }

    fn update_targets(&mut self, show_time: f32, timeline: &ShowTimeline, dt: f32) {
        let sample = timeline.sample(show_time);
        let target_kind = match sample.cue.kind {
            CueKind::Landing => FormationKind::LaunchGrid,
            _ => sample
                .cue
                .kind
                .formation()
                .unwrap_or(sample.previous_formation),
        };
        let mut animated = self.generate_formation(target_kind, show_time);
        if target_kind != FormationKind::LaunchGrid {
            let scale = formation::fleet_scale(self.drones.len());
            for point in &mut animated {
                point.position *= scale;
                let organic =
                    self.settings.organic_variation * (show_time * 0.8 + point.phase * 19.0).sin();
                point.position += Vec3::new(organic, organic * 0.35, -organic * 0.5);
            }
            lift_formation_above_ground(&mut animated);
        }

        let transitioning = matches!(
            sample.cue.kind,
            CueKind::Launch { .. } | CueKind::Transition { .. } | CueKind::Landing
        );
        let offset_retention = (-dt * 0.24).exp();
        for offset in &mut self.separation_offsets {
            *offset *= offset_retention;
        }
        for index in 0..self.drones.len() {
            let slot = self.drones[index]
                .slot
                .min(animated.len().saturating_sub(1));
            let destination = &animated[slot];
            let previous_position = self.targets[index].position;
            let previous_velocity = self.target_velocities[index];
            self.targets[index] = destination.clone();
            self.target_phase_lock[index] = 0.0;
            if transitioning {
                let mut progress = sample.progress;
                if matches!(sample.cue.kind, CueKind::Launch { .. } | CueKind::Landing) {
                    let wave = (index as f32 / self.drones.len().max(1) as f32) * 0.28;
                    progress = ((progress - wave) / (1.0 - wave)).clamp(0.0, 1.0);
                }
                let trajectory_position = self.trajectories[index].sample(progress);
                // Animated destinations converge during the final third of the morph.
                let animation_blend = smoothstep(0.6, 1.0, progress);
                self.targets[index].position =
                    trajectory_position.lerp(destination.position, animation_blend);
                self.target_phase_lock[index] = smoothstep(0.72, 0.985, progress).powi(2);
                let visibility = match sample.cue.kind {
                    CueKind::Launch { .. } => smoothstep(0.02, 0.2, progress),
                    CueKind::Landing => 1.0 - smoothstep(0.78, 1.0, progress),
                    _ => 1.0,
                };
                self.targets[index].brightness *= visibility;
            }
            if matches!(sample.cue.kind, CueKind::ColorWave { .. }) {
                let wave = ((destination.phase + sample.progress) * std::f32::consts::TAU).sin()
                    * 0.5
                    + 0.5;
                self.targets[index].color = destination.color.mix(Rgbw::GOLD, wave);
            }
            self.targets[index].position += self.separation_offsets[index];
            let target_velocity = (self.targets[index].position - previous_position) / dt.max(1e-4);
            self.target_velocities[index] = target_velocity;
            self.target_accelerations[index] = (target_velocity - previous_velocity) / dt.max(1e-4);
        }
    }

    fn rebuild_spatial_grid(&mut self) {
        for bucket in self.spatial_grid.values_mut() {
            bucket.clear();
        }
        let cell = self.effective_separation().max(DRONE_WARNING_ENVELOPE);
        for (index, drone) in self.drones.iter().enumerate() {
            let key = (
                (drone.position.x / cell).floor() as i32,
                (drone.position.y / cell).floor() as i32,
                (drone.position.z / cell).floor() as i32,
            );
            self.spatial_grid.entry(key).or_default().push(index);
        }
        self.spatial_grid.retain(|_, bucket| !bucket.is_empty());
    }

    fn avoidance_force(&self, index: usize) -> Vec3 {
        let drone = &self.drones[index];
        let avoidance_radius = self.effective_separation();
        let cell = avoidance_radius.max(DRONE_WARNING_ENVELOPE);
        let base = (
            (drone.position.x / cell).floor() as i32,
            (drone.position.y / cell).floor() as i32,
            (drone.position.z / cell).floor() as i32,
        );
        let mut force = Vec3::ZERO;
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if let Some(neighbors) =
                        self.spatial_grid.get(&(base.0 + x, base.1 + y, base.2 + z))
                    {
                        for &other in neighbors {
                            if other == index {
                                continue;
                            }
                            let delta = drone.position - self.drones[other].position;
                            let distance = delta.length();
                            if distance > 0.001 && distance < avoidance_radius {
                                force +=
                                    delta / distance * (1.0 - distance / avoidance_radius).powi(2);
                            }
                        }
                    }
                }
            }
        }
        force.clamp_length_max(1.0)
    }

    fn generate_formation(&self, kind: FormationKind, time: f32) -> Vec<FormationPoint> {
        if kind != FormationKind::Image || self.custom_formation.is_empty() {
            return formation::generate(kind, self.drones.len(), time);
        }
        let source_len = self.custom_formation.len();
        (0..self.drones.len())
            .map(|index| {
                let source = index * source_len / self.drones.len().max(1);
                self.custom_formation[source.min(source_len - 1)].clone()
            })
            .collect()
    }

    /// Audits every fixed simulation step. The spatial hash is an acceleration
    /// structure only: all 27 neighboring cells are tested, which is exhaustive
    /// for the displayed warning and collision envelopes.
    fn update_safety(&mut self, show_time: f32) {
        let cell = self.effective_separation().max(DRONE_WARNING_ENVELOPE);
        let monitor_air = self.settings.monitor_air_envelope;
        self.safety_flags.resize(self.drones.len(), 0);
        self.safety_flags.fill(0);
        let flags = &mut self.safety_flags;
        let mut minimum_air = DRONE_WARNING_ENVELOPE;
        let mut closest_pair = None;
        let mut closest_pair_position = None;
        let mut near_miss_pairs = 0u32;
        let mut collision_pairs = 0u32;
        if monitor_air {
            for index in 0..self.drones.len() {
                let drone = &self.drones[index];
                let base = (
                    (drone.position.x / cell).floor() as i32,
                    (drone.position.y / cell).floor() as i32,
                    (drone.position.z / cell).floor() as i32,
                );
                for x in -1..=1 {
                    for y in -1..=1 {
                        for z in -1..=1 {
                            if let Some(neighbors) =
                                self.spatial_grid.get(&(base.0 + x, base.1 + y, base.2 + z))
                            {
                                for &other in neighbors {
                                    if other <= index {
                                        continue;
                                    }
                                    let distance =
                                        drone.position.distance(self.drones[other].position);
                                    if distance < minimum_air {
                                        minimum_air = distance;
                                        closest_pair = Some((drone.id, self.drones[other].id));
                                        closest_pair_position = Some(
                                            (drone.position + self.drones[other].position) * 0.5,
                                        );
                                    }
                                    if distance < DRONE_WARNING_ENVELOPE {
                                        near_miss_pairs += 1;
                                        flags[index] = flags[index].max(1);
                                        flags[other] = flags[other].max(1);
                                    }
                                    if distance < DRONE_COLLISION_ENVELOPE {
                                        collision_pairs += 1;
                                        flags[index] = 2;
                                        flags[other] = 2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut minimum_ground = f32::INFINITY;
        let mut ground_breaches = 0u32;
        let mut lowest_drone = None;
        let mut lowest_drone_position = None;
        for (index, drone) in self.drones.iter().enumerate() {
            let clearance = drone.position.y - DRONE_GROUND_RADIUS;
            if clearance < minimum_ground {
                minimum_ground = clearance;
                lowest_drone = Some(drone.id);
                lowest_drone_position = Some(drone.position);
            }
            if clearance < 0.0 {
                ground_breaches += 1;
                flags[index] = 2;
            } else if clearance < 0.12 {
                flags[index] = flags[index].max(1);
            }
        }
        let unsafe_now = (monitor_air && collision_pairs > 0) || ground_breaches > 0;
        let incident_events =
            self.safety.incident_events + u64::from(unsafe_now && !self.previous_step_unsafe);
        self.safety = SafetyStats {
            minimum_air_separation: minimum_air,
            minimum_ground_clearance: minimum_ground,
            near_miss_pairs,
            collision_pairs,
            ground_breaches,
            incident_events,
            monitored_steps: self.safety.monitored_steps + 1,
            last_incident_time: if unsafe_now && !self.previous_step_unsafe {
                Some(show_time)
            } else {
                self.safety.last_incident_time
            },
            closest_pair,
            closest_pair_position,
            lowest_drone,
            lowest_drone_position,
        };
        self.previous_step_unsafe = unsafe_now;
    }

    /// Position-based final correction. Flight steering handles normal
    /// separation; this narrow solver is the hard physical backstop that keeps
    /// bodies and the ground from interpenetrating between telemetry audits.
    fn resolve_collision_constraints(&mut self, dt: f32, max_iterations: usize) -> usize {
        // Resolve to the far side of the warning envelope so integration,
        // wind and quantisation cannot immediately turn a near miss back into
        // a physical overlap on the next fixed step.
        const SOLVER_DISTANCE: f32 = DRONE_WARNING_ENVELOPE + 0.04;
        for drone in &mut self.drones {
            drone.position.y = drone.position.y.max(DRONE_GROUND_RADIUS + 0.012);
        }
        let mut iterations = 0;
        let mut pairs = std::mem::take(&mut self.collision_pairs_scratch);
        if max_iterations <= 8 {
            // Live frames reuse one conservative warning-radius candidate set
            // for several projection passes. Corrections are millimetric, so
            // rebuilding the entire hash between each pass was redundant and
            // dominated dense-fleet CPU time.
            self.rebuild_spatial_grid();
            self.collect_close_pairs(DRONE_WARNING_ENVELOPE, &mut pairs);
            for _ in 0..max_iterations {
                iterations += 1;
                if !self.project_collision_pairs(&pairs, dt, SOLVER_DISTANCE) {
                    break;
                }
            }
        } else {
            // Seeks are not real-time flight and can afford rehashing to fully
            // settle an arbitrarily teleported editor state.
            for _ in 0..max_iterations {
                iterations += 1;
                self.rebuild_spatial_grid();
                self.collect_close_pairs(SOLVER_DISTANCE, &mut pairs);
                if pairs.is_empty() || !self.project_collision_pairs(&pairs, dt, SOLVER_DISTANCE) {
                    break;
                }
            }
        }
        for drone in &mut self.drones {
            drone.position.y = drone.position.y.max(DRONE_GROUND_RADIUS + 0.012);
        }
        self.collision_pairs_scratch = pairs;
        iterations
    }

    fn collect_close_pairs(&self, radius: f32, pairs: &mut Vec<(usize, usize)>) {
        pairs.clear();
        let cell = self.effective_separation().max(DRONE_WARNING_ENVELOPE);
        for index in 0..self.drones.len() {
            let position = self.drones[index].position;
            let base = (
                (position.x / cell).floor() as i32,
                (position.y / cell).floor() as i32,
                (position.z / cell).floor() as i32,
            );
            for x in -1..=1 {
                for y in -1..=1 {
                    for z in -1..=1 {
                        if let Some(neighbors) =
                            self.spatial_grid.get(&(base.0 + x, base.1 + y, base.2 + z))
                        {
                            for &other in neighbors {
                                if other > index
                                    && position.distance_squared(self.drones[other].position)
                                        < radius * radius
                                {
                                    pairs.push((index, other));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn project_collision_pairs(
        &mut self,
        pairs: &[(usize, usize)],
        dt: f32,
        solver_distance: f32,
    ) -> bool {
        let mut corrected_any = false;
        for &(index, other) in pairs {
            let (left, right) = self.drones.split_at_mut(other);
            let first = &mut left[index];
            let second = &mut right[0];
            let delta = first.position - second.position;
            let distance = delta.length();
            if distance >= solver_distance {
                continue;
            }
            corrected_any = true;
            let direction = if distance > 1e-5 {
                delta / distance
            } else {
                let angle = index as f32 * 2.399_963_1;
                Vec3::new(angle.cos(), 0.35, angle.sin()).normalize()
            };
            let correction = direction * (solver_distance - distance) * 0.505;
            first.position += correction;
            second.position -= correction;
            first.velocity += correction / dt.max(1e-4) * 0.12;
            second.velocity -= correction / dt.max(1e-4) * 0.12;
            self.separation_offsets[index] =
                (self.separation_offsets[index] + correction).clamp_length_max(2.5);
            self.separation_offsets[other] =
                (self.separation_offsets[other] - correction).clamp_length_max(2.5);
        }
        corrected_any
    }
}

fn lift_formation_above_ground(points: &mut [FormationPoint]) {
    let minimum_y = points
        .iter()
        .map(|point| point.position.y)
        .fold(f32::INFINITY, f32::min);
    let lift = (3.0 - minimum_y).max(15.0);
    for point in points {
        point.position.y += lift;
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitored_settings() -> SimulationSettings {
        SimulationSettings {
            monitor_air_envelope: true,
            ..SimulationSettings::default()
        }
    }

    #[test]
    fn avoidance_radius_adapts_to_dense_fleets() {
        let settings = SimulationSettings::default();
        let small = FleetSimulation::new(384, settings.clone());
        let dense = FleetSimulation::new(5_000, settings);
        assert!((small.effective_separation() - 0.72).abs() < 0.01);
        assert!(dense.effective_separation() < 0.3);
        assert!(dense.effective_separation() >= 0.08);
    }

    #[test]
    fn dense_transition_arrives_in_timeline_lock() {
        let timeline = ShowTimeline::showcase();
        let mut simulation = FleetSimulation::new(2_000, monitored_settings());
        let mut time = 15.9;
        simulation.seek(time, &timeline);
        while time < 21.98 {
            time += 1.0 / 60.0;
            simulation.step_frame(1.0 / 60.0, time, &timeline, 32.0);
        }
        assert!(
            simulation.tracking_error_rms() < 0.12,
            "dense fleet ended transition with {:.3}m RMS tracking error",
            simulation.tracking_error_rms()
        );
        let safety = simulation.safety_stats();
        assert!(
            safety.is_clear(),
            "safety audit found {} air and {} ground breaches",
            safety.collision_pairs,
            safety.ground_breaches
        );
        assert_eq!(
            safety.incident_events, 0,
            "the transition entered the physical collision envelope near {:?}",
            safety.last_incident_time
        );
    }

    #[test]
    fn physical_backstop_resolves_overlap_and_ground_contact() {
        let mut simulation = FleetSimulation::new(2, monitored_settings());
        simulation.drones[0].position = Vec3::new(0.0, -1.0, 0.0);
        simulation.drones[1].position = Vec3::new(0.0, -1.0, 0.0);
        simulation.resolve_collision_constraints(1.0 / 60.0, 64);
        simulation.rebuild_spatial_grid();
        simulation.update_safety(0.0);
        assert!(simulation.safety_stats().is_clear());
        assert!(simulation.drones[0].position.y >= DRONE_GROUND_RADIUS);
        assert!(simulation.drones[1].position.y >= DRONE_GROUND_RADIUS);
    }

    #[test]
    fn dense_formation_slots_respect_collision_envelope() {
        let timeline = ShowTimeline::showcase();
        for time in [15.9, 21.98] {
            let mut simulation = FleetSimulation::new(2_000, monitored_settings());
            simulation.seek(time, &timeline);
            let safety = simulation.safety_stats();
            assert!(
                safety.is_clear(),
                "seek {time:.2}s left {} collision pairs at {:.4}m",
                safety.collision_pairs,
                safety.minimum_air_separation
            );
            assert_eq!(safety.incident_events, 0);
        }
    }

    #[test]
    fn five_thousand_drone_transition_avoids_constraint_spiral() {
        let timeline = ShowTimeline::showcase();
        let mut simulation = FleetSimulation::new(5_000, monitored_settings());
        let mut time = 16.0;
        simulation.seek(time, &timeline);
        let mut total_iterations = 0u32;
        let mut peak_iterations = 0u32;
        let mut total_cpu_ms = 0.0;
        for _ in 0..180 {
            time += 1.0 / 60.0;
            simulation.step_frame(1.0 / 60.0, time, &timeline, 32.0);
            let iterations = simulation.constraint_iterations_last_frame();
            total_iterations += iterations;
            peak_iterations = peak_iterations.max(iterations);
            total_cpu_ms += simulation.simulation_cpu_ms();
        }
        let average_iterations = total_iterations as f32 / 180.0;
        let average_cpu_ms = total_cpu_ms / 180.0;
        eprintln!(
            "5k transition: {average_iterations:.2} solver passes/frame, peak {peak_iterations}, {average_cpu_ms:.2} ms CPU/frame"
        );
        assert!(simulation.safety_stats().is_clear());
        assert_eq!(simulation.safety_stats().incident_events, 0);
        assert!(average_iterations < 4.0);
        assert!(peak_iterations <= 3);
    }

    #[test]
    fn animated_target_is_continuous_across_morph_boundary() {
        let timeline = ShowTimeline::showcase();
        let animate_index = timeline
            .cues
            .iter()
            .position(|cue| {
                matches!(
                    cue.kind,
                    CueKind::FormationAnimation {
                        formation: FormationKind::Cathedral
                    }
                )
            })
            .expect("cathedral animation cue");
        let boundary = timeline.cue_start(animate_index);
        let epsilon = 1.0 / 240.0;
        let mut simulation = FleetSimulation::new(384, monitored_settings());
        simulation.seek(boundary - epsilon, &timeline);
        let before: Vec<_> = simulation
            .targets
            .iter()
            .map(|target| target.position)
            .collect();
        simulation.prepare_cue(boundary + epsilon, &timeline);
        simulation.update_targets(boundary + epsilon, &timeline, 1.0 / 60.0);
        let rms_jump = (before
            .iter()
            .zip(&simulation.targets)
            .map(|(before, after)| before.distance_squared(after.position))
            .sum::<f32>()
            / before.len() as f32)
            .sqrt();
        assert!(
            rms_jump < 0.5,
            "morph/animate boundary introduced a {rms_jump:.3}m target jump"
        );
    }

    #[test]
    fn disabling_air_monitor_keeps_ground_monitoring_active() {
        let settings = SimulationSettings {
            monitor_air_envelope: false,
            ..SimulationSettings::default()
        };
        let mut simulation = FleetSimulation::new(2, settings);
        simulation.drones[0].position = Vec3::new(0.0, -1.0, 0.0);
        simulation.drones[1].position = Vec3::new(0.0, -1.0, 0.0);
        simulation.rebuild_spatial_grid();
        simulation.update_safety(0.0);
        let safety = simulation.safety_stats();
        assert_eq!(safety.collision_pairs, 0);
        assert_eq!(safety.ground_breaches, 2);
        assert!(!safety.is_clear());
    }
}
