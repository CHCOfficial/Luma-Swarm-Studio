use std::{
    borrow::Cow,
    mem,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc, Mutex,
    },
};

use bytemuck::{Pod, Zeroable};
use eframe::egui;
use egui_wgpu::{wgpu, CallbackResources, CallbackTrait, ScreenDescriptor};
use wgpu::util::DeviceExt;

use crate::{
    camera::CameraRig,
    model::{
        EnvironmentSettings, FleetExecutionMode, FormationKind, FormationPoint, GraphicsSettings,
    },
    simulation::FleetSimulation,
    timeline::{CueKind, ShowTimeline},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RawInstance {
    position_brightness: [f32; 4],
    color: [f32; 4],
    orientation: [f32; 4],
    misc: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GlobalUniform {
    view_projection: [f32; 16],
    camera_position: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    time_exposure_bloom_haze: [f32; 4],
    viewport: [f32; 4],
    environment: [f32; 4],
    fleet_lod: [f32; 4],
    gpu_show: [f32; 4],
    gpu_meta: [f32; 4],
    gpu_safety: [f32; 4],
    gpu_image: [f32; 4],
    safety_options: [f32; 4],
    color_controls: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSafetyCountersRaw {
    warning_pairs: u32,
    collision_pairs: u32,
    ground_breaches: u32,
    minimum_distance_bits: u32,
    collision_a: u32,
    collision_b: u32,
    collision_x_bits: u32,
    collision_y_bits: u32,
    collision_z_bits: u32,
    collision_distance_bits: u32,
    ground_drone: u32,
    ground_x_bits: u32,
    ground_y_bits: u32,
    ground_z_bits: u32,
    show_time_bits: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuSafetyTelemetry {
    pub valid: bool,
    pub warning_pairs: u32,
    pub collision_pairs: u32,
    pub ground_breaches: u32,
    pub minimum_air_separation: f32,
    pub audited_frames: u64,
    pub sample_time: f32,
    pub collision_pair: Option<(u32, u32)>,
    pub collision_position: Option<[f32; 3]>,
    pub collision_separation: Option<f32>,
    pub ground_drone: Option<u32>,
    pub ground_position: Option<[f32; 3]>,
}

pub fn gpu_image_instances(points: &[FormationPoint]) -> Arc<Vec<RawInstance>> {
    Arc::new(
        points
            .iter()
            .map(|point| {
                let color = point.color.rgb();
                RawInstance {
                    position_brightness: [
                        point.position.x,
                        point.position.y,
                        point.position.z,
                        point.brightness,
                    ],
                    color: [color.x, color.y, color.z, point.color.w],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    misc: [0.0, point.phase, 1.0, 0.0],
                }
            })
            .collect(),
    )
}

pub fn gpu_resident_supported(device: &wgpu::Device, fleet_count: usize) -> bool {
    let limits = device.limits();
    let capacity = fleet_count.max(1).next_power_of_two();
    let instance_bytes = (capacity * mem::size_of::<RawInstance>()) as u64;
    let workgroups = fleet_count.max(1).div_ceil(64) as u32;
    instance_bytes <= limits.max_buffer_size
        && instance_bytes <= limits.max_storage_buffer_binding_size as u64
        && workgroups <= limits.max_compute_workgroups_per_dimension
}

impl Default for GpuSafetyTelemetry {
    fn default() -> Self {
        Self {
            valid: false,
            warning_pairs: 0,
            collision_pairs: 0,
            ground_breaches: 0,
            minimum_air_separation: 0.38,
            audited_frames: 0,
            sample_time: 0.0,
            collision_pair: None,
            collision_position: None,
            collision_separation: None,
            ground_drone: None,
            ground_position: None,
        }
    }
}

impl GpuSafetyTelemetry {
    pub fn is_clear(self) -> bool {
        self.valid && self.collision_pairs == 0 && self.ground_breaches == 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DroneVertex {
    position: [f32; 3],
    normal: [f32; 3],
    material: f32,
}

impl DroneVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: mem::size_of::<DroneVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: 24,
                shader_location: 2,
            },
        ],
    };
}

#[allow(clippy::too_many_arguments)]
pub fn make_callback(
    rect: egui::Rect,
    simulation: &FleetSimulation,
    camera: &CameraRig,
    graphics: &GraphicsSettings,
    environment: &EnvironmentSettings,
    timeline: &ShowTimeline,
    requested_fleet_count: usize,
    execution_mode: FleetExecutionMode,
    gpu_collision_correction: bool,
    gpu_safety_audit: bool,
    visualize_safety_alerts: bool,
    monitor_air_envelope: bool,
    gpu_safety_telemetry: &Arc<Mutex<GpuSafetyTelemetry>>,
    gpu_image_instances: &Arc<Vec<RawInstance>>,
    time: f32,
) -> egui::PaintCallback {
    let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
    let eye = camera.eye();
    let forward = (camera.target - eye).normalize_or_zero();
    let right = forward.cross(glam::Vec3::Y).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let gpu_resident = execution_mode == FleetExecutionMode::GpuResident;
    let instance_count = if gpu_resident {
        requested_fleet_count.clamp(100, 1_000_000)
    } else {
        simulation.drones.len()
    };
    let fleet_lod = fleet_lod(instance_count);
    let instances = if gpu_resident {
        Arc::clone(gpu_image_instances)
    } else {
        Arc::new(
            simulation
                .drones
                .iter()
                .enumerate()
                .map(|(index, drone)| {
                    let p = simulation.interpolated_position(drone);
                    let safety_level = if visualize_safety_alerts {
                        simulation.safety_level(index)
                    } else {
                        0
                    };
                    let base_color = drone.color.rgb() * drone.brightness;
                    let color = match safety_level {
                        2 => glam::Vec3::new(1.0, 0.015, 0.01) * 2.8,
                        1 => glam::Vec3::new(1.0, 0.34, 0.015) * 1.8,
                        _ => base_color,
                    };
                    RawInstance {
                        position_brightness: [p.x, p.y, p.z, drone.brightness],
                        color: [color.x, color.y, color.z, drone.color.w],
                        orientation: drone.orientation.to_array(),
                        misc: [
                            drone.rotor_angle,
                            drone.phase,
                            drone.battery,
                            safety_level as f32,
                        ],
                    }
                })
                .collect(),
        )
    };
    let sample = timeline.sample(time);
    let previous = sample.previous_formation;
    let target = match sample.cue.kind {
        CueKind::Landing => FormationKind::LaunchGrid,
        _ => sample.cue.kind.formation().unwrap_or(previous),
    };
    let image_rgbw_emitters = previous == FormationKind::Image || target == FormationKind::Image;
    let transitioning = matches!(
        sample.cue.kind,
        CueKind::Launch { .. } | CueKind::Transition { .. } | CueKind::Landing
    );
    let globals = GlobalUniform {
        view_projection: camera.view_projection(aspect).to_cols_array(),
        camera_position: [eye.x, eye.y, eye.z, 1.0],
        camera_right: [right.x, right.y, right.z, 0.0],
        camera_up: [up.x, up.y, up.z, 0.0],
        time_exposure_bloom_haze: [time, graphics.exposure, graphics.bloom, graphics.haze],
        viewport: [
            rect.width(),
            rect.height(),
            graphics.light_size,
            graphics.ground_reflections,
        ],
        environment: [
            environment.wind_direction_degrees,
            environment.cloud_cover,
            environment.field_reflectivity,
            environment.star_brightness,
        ],
        fleet_lod,
        gpu_show: [
            if gpu_resident { 1.0 } else { 0.0 },
            formation_id(previous) as f32,
            formation_id(target) as f32,
            sample.progress,
        ],
        gpu_meta: [
            time,
            time,
            if transitioning { 1.0 } else { 0.0 },
            crate::formation::fleet_scale(instance_count),
        ],
        gpu_safety: [
            if gpu_resident && gpu_safety_audit {
                1.0
            } else {
                0.0
            },
            0.24,
            0.38,
            0.14,
        ],
        gpu_image: [
            instances.len().min(instance_count) as f32,
            0.62,
            if image_rgbw_emitters { 1.0 } else { 0.0 },
            if instances.is_empty() {
                1.0
            } else {
                crate::formation::fleet_scale(instances.len())
                    / crate::formation::fleet_scale(instance_count)
            },
        ],
        safety_options: [
            if visualize_safety_alerts { 1.0 } else { 0.0 },
            if monitor_air_envelope { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ],
        color_controls: [
            if image_rgbw_emitters {
                1.0
            } else {
                graphics.saturation
            },
            1.0,
            0.0,
            0.0,
        ],
    };
    egui_wgpu::Callback::new_paint_callback(
        rect,
        FleetPaintCallback {
            globals,
            instances,
            instance_count,
            upload_instances_once: gpu_resident,
            gpu_collision_correction: gpu_resident && gpu_collision_correction,
            gpu_safety_enabled: gpu_resident && gpu_safety_audit,
            image_rgbw_emitters,
            gpu_safety_telemetry: Arc::clone(gpu_safety_telemetry),
            viewport_points: [rect.width(), rect.height()],
            render_scale: graphics.render_scale,
        },
    )
}

fn formation_id(kind: FormationKind) -> u32 {
    match kind {
        FormationKind::LaunchGrid => 0,
        FormationKind::Chrysalis => 1,
        FormationKind::Heart => 2,
        FormationKind::Galaxy => 3,
        FormationKind::Cathedral => 4,
        FormationKind::Human => 5,
        FormationKind::Planet => 6,
        FormationKind::Infinity => 7,
        FormationKind::Lotus => 8,
        FormationKind::Crown => 9,
        FormationKind::EventHorizon => 10,
        FormationKind::Mandala => 11,
        FormationKind::Gyroscope => 12,
        FormationKind::Image => 13,
    }
}

fn fleet_lod(count: usize) -> [f32; 4] {
    let relative_density = 384.0 / count.max(1) as f32;
    let body_scale = relative_density.sqrt().clamp(0.015, 1.25);
    let light_scale = relative_density.powf(0.30).clamp(0.06, 1.15);
    let energy_scale = relative_density.powf(0.45).clamp(0.015, 1.0);
    [body_scale, light_scale, energy_scale, count as f32]
}

pub struct FleetPaintCallback {
    globals: GlobalUniform,
    instances: Arc<Vec<RawInstance>>,
    instance_count: usize,
    upload_instances_once: bool,
    gpu_collision_correction: bool,
    gpu_safety_enabled: bool,
    image_rgbw_emitters: bool,
    gpu_safety_telemetry: Arc<Mutex<GpuSafetyTelemetry>>,
    viewport_points: [f32; 2],
    render_scale: f32,
}

impl CallbackTrait for FleetPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let renderer = callback_resources
            .get_mut::<FleetRenderer>()
            .expect("FleetRenderer was registered at startup");
        let scale = screen_descriptor.pixels_per_point * self.render_scale;
        let size = [
            (self.viewport_points[0] * scale).round().max(2.0) as u32,
            (self.viewport_points[1] * scale).round().max(2.0) as u32,
        ];
        renderer.ensure_size(device, size);
        renderer.ensure_capacity(device, self.instance_count.max(self.instances.len()).max(1));
        let _ = device.poll(wgpu::Maintain::Poll);
        // A staging buffer may only be mapped after the frame that copied into it has
        // been submitted. Mapping in the same callback that records the copy makes
        // the upcoming queue submission invalid on wgpu.
        for slot in &renderer.safety_readbacks {
            if slot
                .state
                .compare_exchange(1, 2, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let readback = slot.buffer.clone();
                let mapped_buffer = readback.clone();
                let state = Arc::clone(&slot.state);
                let telemetry = Arc::clone(&self.gpu_safety_telemetry);
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        if result.is_ok() {
                            let mapped = mapped_buffer.slice(..).get_mapped_range();
                            let raw = *bytemuck::from_bytes::<GpuSafetyCountersRaw>(
                                &mapped[..mem::size_of::<GpuSafetyCountersRaw>()],
                            );
                            drop(mapped);
                            mapped_buffer.unmap();
                            if let Ok(mut telemetry) = telemetry.lock() {
                                telemetry.valid = true;
                                telemetry.warning_pairs = raw.warning_pairs;
                                telemetry.collision_pairs = raw.collision_pairs;
                                telemetry.ground_breaches = raw.ground_breaches;
                                telemetry.minimum_air_separation =
                                    f32::from_bits(raw.minimum_distance_bits).min(0.38);
                                telemetry.sample_time = f32::from_bits(raw.show_time_bits);
                                telemetry.collision_pair = (raw.collision_a != u32::MAX)
                                    .then_some((raw.collision_a, raw.collision_b));
                                telemetry.collision_position =
                                    (raw.collision_a != u32::MAX).then_some([
                                        f32::from_bits(raw.collision_x_bits),
                                        f32::from_bits(raw.collision_y_bits),
                                        f32::from_bits(raw.collision_z_bits),
                                    ]);
                                telemetry.collision_separation = (raw.collision_a != u32::MAX)
                                    .then_some(f32::from_bits(raw.collision_distance_bits));
                                telemetry.ground_drone =
                                    (raw.ground_drone != u32::MAX).then_some(raw.ground_drone);
                                telemetry.ground_position =
                                    (raw.ground_drone != u32::MAX).then_some([
                                        f32::from_bits(raw.ground_x_bits),
                                        f32::from_bits(raw.ground_y_bits),
                                        f32::from_bits(raw.ground_z_bits),
                                    ]);
                                telemetry.audited_frames += 1;
                                if telemetry.audited_frames % 120 == 0
                                    && std::env::var("LUMA_SWARM_AUDIT_LOG")
                                        .is_ok_and(|value| value == "1")
                                {
                                    eprintln!(
                                        "GPU audit: warnings={} collisions={} ground={} minimum={:.4}m",
                                        telemetry.warning_pairs,
                                        telemetry.collision_pairs,
                                        telemetry.ground_breaches,
                                        telemetry.minimum_air_separation,
                                    );
                                }
                            }
                        }
                        state.store(3, Ordering::Release);
                    });
            }
        }
        queue.write_buffer(
            &renderer.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.globals),
        );
        let upload_key = Arc::as_ptr(&self.instances) as usize;
        if !self.instances.is_empty()
            && (!self.upload_instances_once
                || renderer.last_instance_upload_key != Some(upload_key))
        {
            queue.write_buffer(
                &renderer.raw_instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
            renderer.last_instance_upload_key = Some(upload_key);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fleet instance compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&renderer.compute_pipeline);
            pass.set_bind_group(0, &renderer.compute_bind_group, &[]);
            pass.dispatch_workgroups(self.instance_count.max(1).div_ceil(64) as u32, 1, 1);
            if self.gpu_collision_correction {
                let correction_passes = if self.instance_count <= 100_000 {
                    18
                } else if self.instance_count <= 500_000 {
                    12
                } else {
                    8
                };
                for _ in 0..correction_passes {
                    pass.set_pipeline(&renderer.safety_clear_pipeline);
                    pass.dispatch_workgroups(renderer.gpu_bucket_count.div_ceil(64) as u32, 1, 1);
                    pass.set_pipeline(&renderer.safety_insert_pipeline);
                    pass.dispatch_workgroups(self.instance_count.max(1).div_ceil(64) as u32, 1, 1);
                    pass.set_pipeline(&renderer.safety_correct_pipeline);
                    pass.dispatch_workgroups(self.instance_count.max(1).div_ceil(64) as u32, 1, 1);
                    pass.set_pipeline(&renderer.safety_apply_pipeline);
                    pass.dispatch_workgroups(self.instance_count.max(1).div_ceil(64) as u32, 1, 1);
                }
            }
            if self.gpu_safety_enabled {
                pass.set_pipeline(&renderer.safety_clear_pipeline);
                pass.dispatch_workgroups(renderer.gpu_bucket_count.div_ceil(64) as u32, 1, 1);
                pass.set_pipeline(&renderer.safety_insert_pipeline);
                pass.dispatch_workgroups(self.instance_count.max(1).div_ceil(64) as u32, 1, 1);
                pass.set_pipeline(&renderer.safety_audit_pipeline);
                pass.dispatch_workgroups(self.instance_count.max(1).div_ceil(64) as u32, 1, 1);
            }
        }

        if self.gpu_safety_enabled {
            let slot_index = renderer.safety_readback_index;
            let slot = &renderer.safety_readbacks[slot_index];
            renderer.safety_readback_index =
                (renderer.safety_readback_index + 1) % renderer.safety_readbacks.len();
            if slot
                .state
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                encoder.copy_buffer_to_buffer(
                    &renderer.gpu_safety_buffer,
                    0,
                    &slot.buffer,
                    0,
                    mem::size_of::<GpuSafetyCountersRaw>() as u64,
                );
            }
        } else if let Ok(mut telemetry) = self.gpu_safety_telemetry.lock() {
            telemetry.valid = false;
        }
        for slot in &renderer.safety_readbacks {
            if slot.state.load(Ordering::Acquire) == 3 {
                // One complete prepare cycle after unmapping before the buffer is reused.
                slot.state.store(0, Ordering::Release);
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HDR drone scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &renderer.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &renderer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &renderer.scene_bind_group, &[]);
            pass.set_pipeline(&renderer.environment_pipeline);
            pass.draw(0..6, 0..1);
            if self.instance_count <= 50_000 {
                pass.set_pipeline(&renderer.body_pipeline);
                pass.set_vertex_buffer(0, renderer.drone_mesh.slice(..));
                pass.draw(
                    0..renderer.drone_vertex_count,
                    0..self.instance_count as u32,
                );
            }
            pass.set_pipeline(&renderer.light_pipeline);
            pass.draw(
                0..if self.image_rgbw_emitters { 24 } else { 6 },
                0..self.instance_count as u32,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        let renderer = callback_resources
            .get::<FleetRenderer>()
            .expect("FleetRenderer was registered at startup");
        render_pass.set_pipeline(&renderer.composite_pipeline);
        render_pass.set_bind_group(0, &renderer.composite_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

pub struct FleetRenderer {
    uniform_buffer: wgpu::Buffer,
    raw_instance_buffer: wgpu::Buffer,
    processed_instance_buffer: wgpu::Buffer,
    corrected_instance_buffer: wgpu::Buffer,
    gpu_spatial_nodes: wgpu::Buffer,
    gpu_bucket_heads: wgpu::Buffer,
    gpu_safety_buffer: wgpu::Buffer,
    gpu_bucket_count: usize,
    instance_capacity: usize,
    last_instance_upload_key: Option<usize>,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    scene_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    safety_clear_pipeline: wgpu::ComputePipeline,
    safety_insert_pipeline: wgpu::ComputePipeline,
    safety_correct_pipeline: wgpu::ComputePipeline,
    safety_apply_pipeline: wgpu::ComputePipeline,
    safety_audit_pipeline: wgpu::ComputePipeline,
    environment_pipeline: wgpu::RenderPipeline,
    body_pipeline: wgpu::RenderPipeline,
    light_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    drone_mesh: wgpu::Buffer,
    drone_vertex_count: u32,
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    safety_readbacks: Vec<SafetyReadbackSlot>,
    safety_readback_index: usize,
    size: [u32; 2],
}

struct SafetyReadbackSlot {
    buffer: wgpu::Buffer,
    // 0 = reusable, 1 = copied in previous submission, 2 = map pending,
    // 3 = just unmapped (one-prepare-cycle cooldown).
    state: Arc<AtomicU8>,
}

impl FleetRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fleet globals"),
            size: mem::size_of::<GlobalUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_capacity = 512;
        let (raw_instance_buffer, processed_instance_buffer, corrected_instance_buffer) =
            create_instance_buffers(device, instance_capacity);
        let (gpu_spatial_nodes, gpu_bucket_heads, gpu_safety_buffer, gpu_bucket_count) =
            create_safety_buffers(device, instance_capacity);

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fleet compute layout"),
                entries: &[
                    buffer_entry(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Uniform,
                        false,
                    ),
                    buffer_entry(
                        1,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                    ),
                    buffer_entry(
                        2,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                    buffer_entry(
                        3,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                    buffer_entry(
                        4,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                    buffer_entry(
                        5,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                    buffer_entry(
                        6,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferBindingType::Storage { read_only: false },
                        false,
                    ),
                ],
            });
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fleet scene layout"),
                entries: &[
                    buffer_entry(
                        0,
                        wgpu::ShaderStages::VERTEX_FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                        false,
                    ),
                    buffer_entry(
                        2,
                        wgpu::ShaderStages::VERTEX_FRAGMENT,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        false,
                    ),
                ],
            });
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fleet composite layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    buffer_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferBindingType::Uniform,
                        false,
                    ),
                ],
            });
        let compute_bind_group = create_compute_bind_group(
            device,
            &compute_bind_group_layout,
            &uniform_buffer,
            &raw_instance_buffer,
            &processed_instance_buffer,
            &gpu_spatial_nodes,
            &gpu_bucket_heads,
            &gpu_safety_buffer,
            &corrected_instance_buffer,
        );
        let scene_bind_group = create_scene_bind_group(
            device,
            &scene_bind_group_layout,
            &uniform_buffer,
            &processed_instance_buffer,
        );

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fleet compute WGSL"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/compute.wgsl"))),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fleet scene WGSL"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/scene.wgsl"))),
        });
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fleet compute pipeline layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = create_compute_pipeline(
            device,
            &compute_layout,
            &compute_shader,
            "fleet instance compute",
            "prepare_instances",
        );
        let safety_clear_pipeline = create_compute_pipeline(
            device,
            &compute_layout,
            &compute_shader,
            "GPU safety clear",
            "clear_safety_grid",
        );
        let safety_insert_pipeline = create_compute_pipeline(
            device,
            &compute_layout,
            &compute_shader,
            "GPU safety spatial insert",
            "insert_safety_grid",
        );
        let safety_audit_pipeline = create_compute_pipeline(
            device,
            &compute_layout,
            &compute_shader,
            "GPU safety audit",
            "audit_safety_grid",
        );
        let safety_correct_pipeline = create_compute_pipeline(
            device,
            &compute_layout,
            &compute_shader,
            "GPU collision correction",
            "correct_safety_grid",
        );
        let safety_apply_pipeline = create_compute_pipeline(
            device,
            &compute_layout,
            &compute_shader,
            "GPU collision correction apply",
            "apply_safety_corrections",
        );
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fleet scene pipeline layout"),
            bind_group_layouts: &[&scene_bind_group_layout],
            push_constant_ranges: &[],
        });
        let environment_pipeline = render_pipeline(
            device,
            "environment",
            &scene_layout,
            &shader,
            "environment_vertex",
            "environment_fragment",
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::BlendState::REPLACE),
            true,
            wgpu::TextureFormat::Rgba16Float,
        );
        let body_pipeline = render_pipeline(
            device,
            "drone bodies",
            &scene_layout,
            &shader,
            "body_vertex",
            "body_fragment",
            &[DroneVertex::LAYOUT],
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::BlendState::ALPHA_BLENDING),
            true,
            wgpu::TextureFormat::Rgba16Float,
        );
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let light_pipeline = render_pipeline(
            device,
            "RGBW light billboards",
            &scene_layout,
            &shader,
            "light_vertex",
            "light_fragment",
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            Some(additive),
            false,
            wgpu::TextureFormat::Rgba16Float,
        );

        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filmic composite WGSL"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/composite.wgsl"))),
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite pipeline layout"),
            bind_group_layouts: &[&composite_bind_group_layout],
            push_constant_ranges: &[],
        });
        let composite_pipeline = render_pipeline(
            device,
            "HDR bloom and tone map",
            &composite_layout,
            &post_shader,
            "vertex_main",
            "fragment_main",
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::BlendState::REPLACE),
            false,
            target_format,
        );

        let mesh = build_drone_mesh();
        let drone_vertex_count = mesh.len() as u32;
        let drone_mesh = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("detailed quadcopter mesh"),
            contents: bytemuck::cast_slice(&mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HDR linear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let (hdr_texture, hdr_view, depth_texture, depth_view) = create_targets(device, [2, 2]);
        let composite_bind_group = create_composite_bind_group(
            device,
            &composite_bind_group_layout,
            &hdr_view,
            &sampler,
            &uniform_buffer,
        );
        let safety_readbacks = (0..3)
            .map(|index| SafetyReadbackSlot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("GPU safety readback {index}")),
                    size: mem::size_of::<GpuSafetyCountersRaw>() as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(0)),
            })
            .collect();
        Self {
            uniform_buffer,
            raw_instance_buffer,
            processed_instance_buffer,
            corrected_instance_buffer,
            gpu_spatial_nodes,
            gpu_bucket_heads,
            gpu_safety_buffer,
            gpu_bucket_count,
            instance_capacity,
            last_instance_upload_key: None,
            compute_bind_group_layout,
            scene_bind_group_layout,
            composite_bind_group_layout,
            compute_bind_group,
            scene_bind_group,
            composite_bind_group,
            compute_pipeline,
            safety_clear_pipeline,
            safety_insert_pipeline,
            safety_correct_pipeline,
            safety_apply_pipeline,
            safety_audit_pipeline,
            environment_pipeline,
            body_pipeline,
            light_pipeline,
            composite_pipeline,
            drone_mesh,
            drone_vertex_count,
            hdr_texture,
            hdr_view,
            depth_texture,
            depth_view,
            sampler,
            safety_readbacks,
            safety_readback_index: 0,
            size: [2, 2],
        }
    }

    fn ensure_size(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        if self.size == size {
            return;
        }
        let (hdr, hdr_view, depth, depth_view) = create_targets(device, size);
        self.hdr_texture = hdr;
        self.hdr_view = hdr_view;
        self.depth_texture = depth;
        self.depth_view = depth_view;
        self.composite_bind_group = create_composite_bind_group(
            device,
            &self.composite_bind_group_layout,
            &self.hdr_view,
            &self.sampler,
            &self.uniform_buffer,
        );
        self.size = size;
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device, needed: usize) {
        if needed <= self.instance_capacity {
            return;
        }
        self.instance_capacity = needed.next_power_of_two();
        self.last_instance_upload_key = None;
        (
            self.raw_instance_buffer,
            self.processed_instance_buffer,
            self.corrected_instance_buffer,
        ) = create_instance_buffers(device, self.instance_capacity);
        (
            self.gpu_spatial_nodes,
            self.gpu_bucket_heads,
            self.gpu_safety_buffer,
            self.gpu_bucket_count,
        ) = create_safety_buffers(device, self.instance_capacity);
        self.compute_bind_group = create_compute_bind_group(
            device,
            &self.compute_bind_group_layout,
            &self.uniform_buffer,
            &self.raw_instance_buffer,
            &self.processed_instance_buffer,
            &self.gpu_spatial_nodes,
            &self.gpu_bucket_heads,
            &self.gpu_safety_buffer,
            &self.corrected_instance_buffer,
        );
        self.scene_bind_group = create_scene_bind_group(
            device,
            &self.scene_bind_group_layout,
            &self.uniform_buffer,
            &self.processed_instance_buffer,
        );
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
    dynamic: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_instance_buffers(
    device: &wgpu::Device,
    capacity: usize,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let size = (capacity * mem::size_of::<RawInstance>()) as u64;
    (
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw drone instances"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU prepared drone instances"),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU corrected drone instances"),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
    )
}

fn create_safety_buffers(
    device: &wgpu::Device,
    capacity: usize,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, usize) {
    let bucket_count = (capacity * 2).next_power_of_two();
    let nodes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU safety spatial nodes"),
        size: (capacity * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let heads = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU safety hash bucket heads"),
        size: (bucket_count * mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let counters = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU safety counters"),
        size: mem::size_of::<GpuSafetyCountersRaw>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    (nodes, heads, counters, bucket_count)
}

#[allow(clippy::too_many_arguments)]
fn create_compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    raw: &wgpu::Buffer,
    processed: &wgpu::Buffer,
    spatial_nodes: &wgpu::Buffer,
    bucket_heads: &wgpu::Buffer,
    safety_counters: &wgpu::Buffer,
    corrected: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fleet compute bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: raw.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: processed.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: spatial_nodes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: bucket_heads.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: safety_counters.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: corrected.as_entire_binding(),
            },
        ],
    })
}

fn create_scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    processed: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fleet scene bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: processed.as_entire_binding(),
            },
        ],
    })
}

fn create_composite_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HDR composite bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}

fn create_targets(
    device: &wgpu::Device,
    size: [u32; 2],
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    let hdr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fleet HDR color"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let hdr_view = hdr.create_view(&Default::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fleet depth"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());
    (hdr, hdr_view, depth, depth_view)
}

#[allow(clippy::too_many_arguments)]
fn render_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vertex_entry: &str,
    fragment_entry: &str,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    topology: wgpu::PrimitiveTopology,
    blend: Option<wgpu::BlendState>,
    depth_write: bool,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            cull_mode: if label == "drone bodies" {
                Some(wgpu::Face::Back)
            } else {
                None
            },
            ..Default::default()
        },
        depth_stencil: if label == "HDR bloom and tone map" {
            None
        } else {
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: depth_write,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            })
        },
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn build_drone_mesh() -> Vec<DroneVertex> {
    let mut vertices = Vec::new();
    add_box(&mut vertices, [0.0, 0.0, 0.0], [0.42, 0.16, 0.34], 0.0);
    add_box(&mut vertices, [0.0, 0.01, 0.0], [1.2, 0.06, 0.07], 0.15);
    add_box(&mut vertices, [0.0, 0.02, 0.0], [0.07, 0.06, 1.2], 0.15);
    for center in [
        [0.98, 0.09, 0.0],
        [-0.98, 0.09, 0.0],
        [0.0, 0.09, 0.98],
        [0.0, 0.09, -0.98],
    ] {
        add_rotor(&mut vertices, center);
    }
    vertices
}

fn add_box(out: &mut Vec<DroneVertex>, center: [f32; 3], half: [f32; 3], material: f32) {
    let c = glam::Vec3::from(center);
    let h = glam::Vec3::from(half);
    let faces = [
        (
            glam::Vec3::X,
            [
                glam::Vec3::new(1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, -1.0, 1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
                glam::Vec3::new(1.0, 1.0, -1.0),
            ],
        ),
        (
            -glam::Vec3::X,
            [
                glam::Vec3::new(-1.0, -1.0, 1.0),
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(-1.0, 1.0, -1.0),
                glam::Vec3::new(-1.0, 1.0, 1.0),
            ],
        ),
        (
            glam::Vec3::Y,
            [
                glam::Vec3::new(-1.0, 1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
                glam::Vec3::new(-1.0, 1.0, 1.0),
            ],
        ),
        (
            -glam::Vec3::Y,
            [
                glam::Vec3::new(-1.0, -1.0, 1.0),
                glam::Vec3::new(1.0, -1.0, 1.0),
                glam::Vec3::new(1.0, -1.0, -1.0),
                glam::Vec3::new(-1.0, -1.0, -1.0),
            ],
        ),
        (
            glam::Vec3::Z,
            [
                glam::Vec3::new(1.0, -1.0, 1.0),
                glam::Vec3::new(-1.0, -1.0, 1.0),
                glam::Vec3::new(-1.0, 1.0, 1.0),
                glam::Vec3::new(1.0, 1.0, 1.0),
            ],
        ),
        (
            -glam::Vec3::Z,
            [
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, -1.0, -1.0),
                glam::Vec3::new(1.0, 1.0, -1.0),
                glam::Vec3::new(-1.0, 1.0, -1.0),
            ],
        ),
    ];
    for (normal, corners) in faces {
        for index in [0, 1, 2, 0, 2, 3] {
            let p = c + corners[index] * h;
            out.push(DroneVertex {
                position: p.to_array(),
                normal: normal.to_array(),
                material,
            });
        }
    }
}

fn add_rotor(out: &mut Vec<DroneVertex>, center: [f32; 3]) {
    let center = glam::Vec3::from(center);
    for i in 0..12 {
        let a = i as f32 * std::f32::consts::TAU / 12.0;
        let b = (i + 1) as f32 * std::f32::consts::TAU / 12.0;
        for p in [
            center,
            center + glam::Vec3::new(a.cos() * 0.47, 0.0, a.sin() * 0.47),
            center + glam::Vec3::new(b.cos() * 0.47, 0.0, b.sin() * 0.47),
        ] {
            out.push(DroneVertex {
                position: p.to_array(),
                normal: glam::Vec3::Y.to_array(),
                material: 0.75,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_fleets_reduce_geometry_glow_and_energy() {
        let baseline = fleet_lod(384);
        let dense = fleet_lod(5_000);
        assert_eq!(baseline[0], 1.0);
        assert!(dense[0] < 0.3);
        assert!(dense[1] < 0.5);
        assert!(dense[2] < 0.4);
    }
}
