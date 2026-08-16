use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::Context as _;
use eframe::egui::{self, Color32, RichText, Stroke};

use crate::{
    camera::{CameraMode, CameraRig},
    image_formation,
    model::{FleetExecutionMode, FormationKind, GraphicsPreset},
    profiling::FrameProfiler,
    project::{ImportedImageFormation, ShowProject},
    renderer::{self, FleetRenderer, GpuSafetyTelemetry},
    safety_log::{RunCollisionLog, SafetyObservation},
    simulation::FleetSimulation,
};

const VERSION_LABEL: &str = "V1.0";
const SUPPORT_URL: &str = "https://buymeacoffee.com/CHCOfficial";
const CODE_URL: &str = "https://github.com/CHCOfficial";
const GRAPHICS_URL: &str = "https://www.deviantart.com/chcofficial";
const AUDIO_URL: &str = "https://suno.com/@artfulexpchc";

#[derive(Clone, Copy, PartialEq, Eq)]
enum InspectorTab {
    Show,
    Fleet,
    Look,
}

pub struct DroneShowApp {
    project: ShowProject,
    simulation: FleetSimulation,
    camera: CameraRig,
    show_time: f32,
    playing: bool,
    presentation_mode: bool,
    inspector_tab: InspectorTab,
    selected_cue: usize,
    last_frame: Instant,
    profiler: FrameProfiler,
    status_message: Option<(String, f32)>,
    image_path_input: String,
    image_hold_duration: f32,
    image_animation: Option<image_formation::SampledAnimation>,
    active_image_frame: usize,
    gpu_safety_telemetry: Arc<Mutex<GpuSafetyTelemetry>>,
    gpu_image_instances: Arc<Vec<renderer::RawInstance>>,
    gpu_supported: bool,
    validated_drone_count: usize,
    gpu_drone_count: usize,
    free_fly_look_active: bool,
    collision_log: RunCollisionLog,
    run_elapsed: f32,
}

impl DroneShowApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> anyhow::Result<Self> {
        configure_style(&cc.egui_ctx);
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .context("Luma Swarm Studio requires the wgpu renderer")?;
        let gpu_supported = renderer::gpu_resident_supported(&render_state.device, 20_000);
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(FleetRenderer::new(
                &render_state.device,
                render_state.target_format,
            ));

        let mut project = ShowProject::default();
        let gpu_preference = std::env::var("LUMA_SWARM_GPU").ok();
        let gpu_resident = gpu_supported && gpu_preference.as_deref() != Some("0");
        project.simulation.execution_mode = if gpu_resident {
            FleetExecutionMode::GpuResident
        } else {
            FleetExecutionMode::FlightValidated
        };
        project.drone_count = if gpu_resident { 20_000 } else { 5_000 };
        if std::env::var("LUMA_SWARM_GPU_SAFETY").is_ok_and(|value| value == "1") {
            project.simulation.gpu_safety_audit = true;
        }
        if std::env::var("LUMA_SWARM_GPU_CORRECTION").is_ok_and(|value| value == "0") {
            project.simulation.gpu_collision_correction = false;
        }
        if let Ok(value) = std::env::var("LUMA_SWARM_MONITOR_AIR") {
            project.simulation.monitor_air_envelope = value != "0";
        }
        if let Ok(count) = std::env::var("LUMA_SWARM_FLEET") {
            if let Ok(count) = count.parse::<usize>() {
                project.drone_count =
                    count.clamp(100, if gpu_resident { 1_000_000 } else { 5_000 });
            }
        }
        let start_time = std::env::var("LUMA_SWARM_TIME")
            .ok()
            .and_then(|time| time.parse::<f32>().ok())
            .map(|time| time.clamp(0.0, project.timeline.duration()));
        let start_playing = std::env::var("LUMA_SWARM_PLAY").is_ok_and(|value| value == "1");
        let simulation_count = if gpu_resident {
            project.drone_count.min(384)
        } else {
            project.drone_count
        };
        let mut simulation = FleetSimulation::new(simulation_count, project.simulation.clone());
        if let Some(time) = start_time {
            simulation.seek(time, &project.timeline);
        }
        let ready_count = project.drone_count;
        let validated_drone_count = if gpu_resident {
            5_000
        } else {
            project.drone_count
        };
        let gpu_drone_count = if gpu_resident {
            project.drone_count
        } else {
            20_000
        };
        let mut app = Self {
            project,
            simulation,
            camera: CameraRig::default(),
            show_time: start_time.unwrap_or(0.0),
            playing: start_playing,
            presentation_mode: false,
            inspector_tab: if gpu_resident {
                InspectorTab::Fleet
            } else {
                InspectorTab::Show
            },
            selected_cue: 0,
            last_frame: Instant::now(),
            profiler: FrameProfiler::default(),
            status_message: Some((format!("Show loaded · {ready_count} aircraft ready"), 3.5)),
            image_path_input: String::new(),
            image_hold_duration: 8.0,
            image_animation: None,
            active_image_frame: 0,
            gpu_safety_telemetry: Arc::new(Mutex::new(GpuSafetyTelemetry::default())),
            gpu_image_instances: Arc::new(Vec::new()),
            gpu_supported,
            validated_drone_count,
            gpu_drone_count,
            free_fly_look_active: false,
            collision_log: RunCollisionLog::default(),
            run_elapsed: 0.0,
        };
        if let Ok(image_path) = std::env::var("LUMA_SWARM_IMAGE") {
            app.import_image(Path::new(&image_path));
            if let Some(time) = start_time {
                app.set_time(time);
                app.playing = start_playing;
            }
        }
        Ok(app)
    }

    fn import_image(&mut self, path: &Path) {
        let point_count =
            if self.simulation.settings.execution_mode == FleetExecutionMode::GpuResident {
                self.project.drone_count.clamp(100, 65_536)
            } else {
                self.simulation.drones.len().clamp(100, 20_000)
            };
        match image_formation::sample_media(path, point_count) {
            Ok(media) => {
                self.clear_run_history();
                let points = media.points;
                let animation = media.animation;
                let animated = animation
                    .as_ref()
                    .is_some_and(|animation| animation.frames.len() > 1);
                if let Some(animation) = &animation {
                    self.image_hold_duration = animation.duration_seconds.clamp(1.0, 300.0);
                }
                let name = path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Imported image")
                    .to_owned();
                self.project.imported_image = Some(ImportedImageFormation {
                    name: name.clone(),
                    source_path: path.display().to_string(),
                    points: points.clone(),
                    hold_duration: self.image_hold_duration,
                    animated,
                });
                self.project.timeline = crate::timeline::ShowTimeline::imported_image_show(
                    self.image_hold_duration,
                    animated,
                );
                self.image_animation = animation;
                self.active_image_frame = 0;
                self.gpu_image_instances = renderer::gpu_image_instances(&points);
                self.simulation.set_custom_formation(points);
                self.show_time = 0.0;
                self.simulation.seek(0.0, &self.project.timeline);
                self.selected_cue = 0;
                self.playing = true;
                self.status_message = Some((
                    format!(
                        "{} show created · {name} · {} slots · launch, {}, land",
                        if animated { "GIF" } else { "Image" },
                        point_count,
                        if animated { "playback" } else { "hold" }
                    ),
                    4.5,
                ));
            }
            Err(error) => {
                self.status_message = Some((format!("Image import failed · {error}"), 6.0));
            }
        }
    }

    fn activate_imported_image(&mut self) {
        let Some(imported) = &self.project.imported_image else {
            return;
        };
        let name = imported.name.clone();
        let hold_duration = imported.hold_duration;
        let animated = imported.animated;
        self.project.timeline =
            crate::timeline::ShowTimeline::imported_image_show(hold_duration, animated);
        self.selected_cue = self
            .project
            .timeline
            .cues
            .iter()
            .position(|cue| cue.kind.formation() == Some(FormationKind::Image))
            .unwrap_or(0);
        self.set_time(self.project.timeline.cue_start(self.selected_cue) + 0.15);
        self.playing = false;
        self.status_message = Some((format!("Imported image selected · {name}"), 3.0));
    }

    fn update_image_animation(&mut self) {
        let Some(animation) = &self.image_animation else {
            return;
        };
        let Some(imported) = &self.project.imported_image else {
            return;
        };
        if animation.frames.len() <= 1 {
            return;
        }
        let sample = self.project.timeline.sample(self.show_time);
        let desired_frame = match sample.cue.kind {
            crate::timeline::CueKind::ImageAnimation { .. } => {
                let playback_time = sample.progress * animation.duration_seconds;
                animation.frame_at(playback_time)
            }
            crate::timeline::CueKind::Landing => animation.frames.len() - 1,
            _ => 0,
        };
        if desired_frame == self.active_image_frame {
            return;
        }
        let points = animation.apply_frame(&imported.points, desired_frame);
        self.gpu_image_instances = renderer::gpu_image_instances(&points);
        self.simulation.update_custom_formation_colors(&points);
        self.active_image_frame = desired_frame;
    }

    fn activate_builtin_formation(&mut self, kind: FormationKind) {
        if !self
            .project
            .timeline
            .cues
            .iter()
            .any(|cue| cue.kind.formation() == Some(kind))
        {
            self.project.timeline = crate::timeline::ShowTimeline::showcase();
        }
        if let Some(index) = self
            .project
            .timeline
            .cues
            .iter()
            .position(|cue| cue.kind.formation() == Some(kind))
        {
            self.selected_cue = index;
            self.set_time(self.project.timeline.cue_start(index) + 0.15);
            self.playing = false;
            self.status_message = Some((
                format!("Built-in formation selected · {}", kind.label()),
                3.0,
            ));
        }
    }

    fn restart(&mut self) {
        self.clear_run_history();
        self.show_time = 0.0;
        self.simulation.reset();
        self.playing = true;
    }

    fn clear_run_history(&mut self) {
        self.collision_log.clear();
        self.run_elapsed = 0.0;
        self.simulation.clear_safety_history();
        if let Ok(mut telemetry) = self.gpu_safety_telemetry.lock() {
            *telemetry = GpuSafetyTelemetry::default();
        }
    }

    fn observe_safety(&mut self) {
        if !self.playing {
            return;
        }
        match self.simulation.settings.execution_mode {
            FleetExecutionMode::FlightValidated => {
                let safety = self.simulation.safety_stats();
                self.collision_log.observe(SafetyObservation {
                    audit_sequence: safety.monitored_steps,
                    run_time: self.run_elapsed,
                    show_time: self.show_time,
                    collision_pairs: safety.collision_pairs,
                    ground_breaches: safety.ground_breaches,
                    minimum_air_separation: safety.minimum_air_separation,
                    minimum_ground_clearance: safety.minimum_ground_clearance,
                    collision_pair: (safety.collision_pairs > 0)
                        .then_some(safety.closest_pair)
                        .flatten(),
                    collision_position: (safety.collision_pairs > 0)
                        .then_some(safety.closest_pair_position)
                        .flatten(),
                    ground_drone: (safety.ground_breaches > 0)
                        .then_some(safety.lowest_drone)
                        .flatten(),
                    ground_position: (safety.ground_breaches > 0)
                        .then_some(safety.lowest_drone_position)
                        .flatten(),
                });
            }
            FleetExecutionMode::GpuResident => {
                let telemetry = self
                    .gpu_safety_telemetry
                    .lock()
                    .map(|telemetry| *telemetry)
                    .unwrap_or_default();
                if !telemetry.valid {
                    return;
                }
                let collision_position = telemetry.collision_position.map(glam::Vec3::from);
                let ground_position = telemetry.ground_position.map(glam::Vec3::from);
                self.collision_log.observe(SafetyObservation {
                    audit_sequence: telemetry.audited_frames,
                    run_time: self.run_elapsed,
                    show_time: telemetry.sample_time,
                    collision_pairs: telemetry.collision_pairs,
                    ground_breaches: telemetry.ground_breaches,
                    minimum_air_separation: telemetry
                        .collision_separation
                        .unwrap_or(telemetry.minimum_air_separation),
                    minimum_ground_clearance: ground_position
                        .map(|position| position.y - crate::simulation::DRONE_GROUND_RADIUS)
                        .unwrap_or(f32::INFINITY),
                    collision_pair: telemetry.collision_pair,
                    collision_position,
                    ground_drone: telemetry.ground_drone,
                    ground_position,
                });
            }
        }
    }

    fn set_time(&mut self, time: f32) {
        self.show_time = time.clamp(0.0, self.project.timeline.duration());
        self.simulation.seek(self.show_time, &self.project.timeline);
    }

    fn current_formation(&self) -> FormationKind {
        let sample = self.project.timeline.sample(self.show_time);
        sample
            .cue
            .kind
            .formation()
            .unwrap_or(sample.previous_formation)
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_bar")
            .exact_height(64.0)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(9, 13, 20))
                    .inner_margin(egui::Margin::symmetric(18, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new(&self.project.name)
                                .size(14.0)
                                .color(Color32::from_gray(225)),
                        );
                        ui.label(
                            RichText::new(format!(
                                "{VERSION_LABEL}  /  DEFAULT CHOREOGRAPHY  /  CONTINUOUS LOOP"
                            ))
                            .size(9.0)
                            .color(Color32::from_gray(103)),
                        );
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let presentation = ui.add(
                            egui::Button::new(RichText::new("PRESENT").size(11.0).strong())
                                .fill(Color32::from_rgb(19, 93, 108))
                                .stroke(Stroke::new(1.0, Color32::from_rgb(55, 180, 199)))
                                .corner_radius(5.0)
                                .min_size(egui::vec2(86.0, 32.0)),
                        );
                        if presentation.clicked() {
                            self.presentation_mode = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(true));
                        }
                        let support = ui.add(
                            egui::Button::new(
                                RichText::new("☕  BUY ME A COFFEE")
                                    .size(10.0)
                                    .strong()
                                    .color(Color32::from_rgb(247, 211, 112)),
                            )
                            .fill(Color32::from_rgb(37, 28, 18))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(132, 99, 42)))
                            .corner_radius(5.0)
                            .min_size(egui::vec2(140.0, 32.0)),
                        );
                        if support.clicked() {
                            ctx.open_url(egui::OpenUrl::new_tab(SUPPORT_URL));
                        }
                        support.on_hover_text(SUPPORT_URL);
                    });
                });
            });
    }

    fn left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("formation_browser")
            .default_width(218.0)
            .width_range(194.0..=286.0)
            .resizable(true)
            .frame(panel_frame())
            .show(ctx, |ui| {
                section_heading(
                    ui,
                    "FORMATION LIBRARY",
                    &format!("{:02}  BUILT-IN", FormationKind::SHOWCASE.len()),
                );
                ui.add_space(10.0);
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if let Some(imported) = &self.project.imported_image {
                        let response = ui.add(
                            egui::Button::new(
                                RichText::new(format!("▧  {}", imported.name))
                                    .size(11.0)
                                    .color(Color32::from_rgb(113, 231, 236)),
                            )
                            .fill(Color32::from_rgb(17, 42, 51))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(43, 122, 135)))
                            .min_size(egui::vec2(ui.available_width(), 38.0)),
                        );
                        if response.clicked() {
                            self.activate_imported_image();
                        }
                        ui.add_space(8.0);
                    }
                    for kind in FormationKind::CORE_SHOWCASE {
                        let current = self.current_formation() == kind;
                        let fill = if current {
                            Color32::from_rgb(18, 52, 61)
                        } else {
                            Color32::from_rgb(13, 19, 28)
                        };
                        let response = egui::Frame::new()
                            .fill(fill)
                            .stroke(Stroke::new(
                                1.0,
                                if current {
                                    Color32::from_rgb(39, 131, 147)
                                } else {
                                    Color32::from_rgb(30, 39, 51)
                                },
                            ))
                            .corner_radius(6.0)
                            .inner_margin(egui::Margin::symmetric(10, 8))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(kind.glyph()).size(24.0).color(
                                        if current {
                                            Color32::from_rgb(77, 222, 235)
                                        } else {
                                            Color32::from_rgb(90, 112, 128)
                                        },
                                    ));
                                    ui.add_space(8.0);
                                    ui.vertical(|ui| {
                                        ui.label(
                                            RichText::new(kind.label())
                                                .size(12.0)
                                                .color(Color32::from_gray(220)),
                                        );
                                        ui.label(
                                            RichText::new(match kind {
                                                FormationKind::Chrysalis
                                                | FormationKind::Cathedral
                                                | FormationKind::Infinity
                                                | FormationKind::Lotus
                                                | FormationKind::Crown => "ANIMATED  ·  3D",
                                                FormationKind::Heart => "VECTOR  ·  2.5D",
                                                _ => "PROCEDURAL  ·  3D",
                                            })
                                            .size(8.5)
                                            .color(Color32::from_gray(92)),
                                        );
                                    });
                                });
                            })
                            .response
                            .interact(egui::Sense::click());
                        if response.clicked() {
                            self.activate_builtin_formation(kind);
                        }
                        ui.add_space(7.0);
                    }
                    ui.add_space(5.0);
                    section_heading(ui, "BONUS", "WOW FACTOR");
                    ui.add_space(9.0);
                    for kind in FormationKind::BONUS {
                        let current = self.current_formation() == kind;
                        let response = egui::Frame::new()
                            .fill(if current {
                                Color32::from_rgb(39, 24, 58)
                            } else {
                                Color32::from_rgb(16, 17, 29)
                            })
                            .stroke(Stroke::new(
                                1.0,
                                if current {
                                    Color32::from_rgb(172, 91, 228)
                                } else {
                                    Color32::from_rgb(52, 40, 71)
                                },
                            ))
                            .corner_radius(6.0)
                            .inner_margin(egui::Margin::symmetric(10, 8))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(kind.glyph())
                                            .size(24.0)
                                            .color(Color32::from_rgb(215, 121, 255)),
                                    );
                                    ui.add_space(8.0);
                                    ui.vertical(|ui| {
                                        ui.label(
                                            RichText::new(kind.label())
                                                .size(12.0)
                                                .color(Color32::from_gray(230)),
                                        );
                                        ui.label(
                                            RichText::new("BONUS  ·  CINEMATIC 3D")
                                                .size(8.5)
                                                .color(Color32::from_rgb(159, 103, 190)),
                                        );
                                    });
                                });
                            })
                            .response
                            .interact(egui::Sense::click());
                        if response.clicked() {
                            self.activate_builtin_formation(kind);
                        }
                        ui.add_space(7.0);
                    }
                });
            });
    }

    fn right_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("inspector")
            .default_width(282.0)
            .width_range(270.0..=372.0)
            .resizable(true)
            .frame(panel_frame())
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    tab(ui, &mut self.inspector_tab, InspectorTab::Show, "SHOW");
                    tab(ui, &mut self.inspector_tab, InspectorTab::Fleet, "FLEET");
                    tab(ui, &mut self.inspector_tab, InspectorTab::Look, "LOOK");
                });
                ui.separator();
                ui.add_space(8.0);
                egui::ScrollArea::vertical().show(ui, |ui| match self.inspector_tab {
                    InspectorTab::Show => self.show_inspector(ui),
                    InspectorTab::Fleet => self.fleet_inspector(ui),
                    InspectorTab::Look => self.look_inspector(ui),
                });
            });
    }

    fn show_inspector(&mut self, ui: &mut egui::Ui) {
        section_heading(
            ui,
            "ACTIVE CUE",
            self.project
                .timeline
                .sample(self.show_time)
                .cue
                .kind
                .label(),
        );
        ui.add_space(10.0);
        let sample = self.project.timeline.sample(self.show_time);
        info_row(ui, "Formation", self.current_formation().label());
        info_row(ui, "Cue time", &format!("{:04.1} s", sample.local_time));
        info_row(
            ui,
            "Fleet",
            &format!("{} aircraft", self.project.drone_count),
        );
        ui.add_space(16.0);
        section_heading(ui, "IMAGE / GIF SHOW", "PNG · JPEG · GIF");
        ui.add_space(8.0);
        let drop = egui::Frame::new()
            .fill(Color32::from_rgb(12, 22, 31))
            .stroke(Stroke::new(1.0, Color32::from_rgb(36, 83, 95)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::symmetric(9, 10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(
                    RichText::new("DROP IMAGE ANYWHERE")
                        .size(10.0)
                        .strong()
                        .color(Color32::from_rgb(96, 213, 220)),
                );
                ui.label(
                    RichText::new(
                        "Full-frame colour or animated GIF becomes an RGBW drone display",
                    )
                    .size(9.0)
                    .color(Color32::from_gray(106)),
                );
            })
            .response;
        drop.on_hover_text("Drop a PNG, JPEG, or animated GIF into the application window");
        ui.add_space(5.0);
        ui.text_edit_singleline(&mut self.image_path_input);
        ui.horizontal(|ui| {
            let mut hold = self.image_hold_duration;
            ui.label(
                RichText::new(
                    if self
                        .project
                        .imported_image
                        .as_ref()
                        .is_some_and(|image| image.animated)
                    {
                        "Playback"
                    } else {
                        "Hold"
                    },
                )
                .size(10.0)
                .color(Color32::from_gray(130)),
            );
            if ui
                .add(egui::Slider::new(&mut hold, 1.0..=60.0).suffix(" s"))
                .changed()
            {
                self.image_hold_duration = hold;
                if let Some(imported) = &mut self.project.imported_image {
                    imported.hold_duration = hold;
                }
                if let Some(cue) = self.project.timeline.cues.iter_mut().find(|cue| {
                    cue.kind.formation() == Some(FormationKind::Image)
                        && matches!(
                            cue.kind,
                            crate::timeline::CueKind::Hold { .. }
                                | crate::timeline::CueKind::ImageAnimation { .. }
                        )
                }) {
                    cue.duration = hold;
                }
            }
            let import_path = self.image_path_input.clone();
            if compact_button(ui, "IMPORT").clicked() && !import_path.trim().is_empty() {
                self.import_image(Path::new(import_path.trim()));
            }
        });
        ui.add_space(16.0);
        section_heading(ui, "CAMERA", "LIVE");
        ui.add_space(8.0);
        let mut requested_camera_mode = self.camera.mode;
        egui::ComboBox::from_id_salt("camera_mode")
            .selected_text(requested_camera_mode.label())
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for mode in [CameraMode::Showcase, CameraMode::Orbit, CameraMode::FreeFly] {
                    ui.selectable_value(&mut requested_camera_mode, mode, mode.label());
                }
            });
        if requested_camera_mode != self.camera.mode {
            self.camera.set_mode(requested_camera_mode);
            if requested_camera_mode != CameraMode::FreeFly {
                self.free_fly_look_active = false;
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
            }
        }
        if self.camera.mode == CameraMode::FreeFly {
            slider_row(
                ui,
                "Move speed",
                &mut self.camera.free_speed,
                1.0..=1_500.0,
                "m/s",
            );
            ui.label(
                RichText::new("WASD move · Q/E vertical · Shift sprint · click viewport for mouse look · Esc release")
                    .size(8.5)
                    .color(Color32::from_gray(101)),
            );
        } else {
            let maximum_distance =
                if self.simulation.settings.execution_mode == FleetExecutionMode::GpuResident {
                    5_000.0
                } else {
                    400.0
                };
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Distance")
                        .size(10.0)
                        .color(Color32::from_gray(130)),
                );
                ui.add(
                    egui::Slider::new(&mut self.camera.distance, 12.0..=maximum_distance)
                        .logarithmic(true)
                        .suffix("m"),
                );
            });
            slider_row(ui, "Elevation", &mut self.camera.pitch, -0.2..=1.1, "rad");
        }
        if compact_button(ui, "FRAME ENTIRE FLEET").clicked() {
            self.camera.set_mode(CameraMode::Showcase);
            self.free_fly_look_active = false;
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
        }
    }

    fn fleet_inspector(&mut self, ui: &mut egui::Ui) {
        section_heading(
            ui,
            "EXECUTION",
            self.simulation.settings.execution_mode.label(),
        );
        ui.add_space(7.0);
        let previous_mode = self.simulation.settings.execution_mode;
        egui::ComboBox::from_id_salt("fleet_execution_mode")
            .selected_text(previous_mode.label())
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.simulation.settings.execution_mode,
                    FleetExecutionMode::FlightValidated,
                    "Flight validated · CPU telemetry",
                );
                ui.add_enabled_ui(self.gpu_supported, |ui| {
                    ui.selectable_value(
                        &mut self.simulation.settings.execution_mode,
                        FleetExecutionMode::GpuResident,
                        "GPU resident · massive choreography",
                    );
                });
            });
        if !self.gpu_supported {
            ui.label(
                RichText::new("GPU-resident mode is unavailable on this device.")
                    .size(8.5)
                    .color(Color32::from_rgb(215, 159, 88)),
            );
        }
        if self.project.imported_image.is_some() {
            ui.label(
                RichText::new("Imported image slots are available in both execution modes.")
                    .size(8.5)
                    .color(Color32::from_gray(94)),
            );
        }
        if previous_mode != self.simulation.settings.execution_mode {
            self.clear_run_history();
            match self.simulation.settings.execution_mode {
                FleetExecutionMode::FlightValidated => {
                    self.gpu_drone_count = self.project.drone_count;
                    self.project.drone_count = self.validated_drone_count;
                    self.simulation.resize(self.project.drone_count);
                    self.simulation.seek(self.show_time, &self.project.timeline);
                }
                FleetExecutionMode::GpuResident => {
                    self.validated_drone_count = self.project.drone_count.min(5_000);
                    self.project.drone_count = self.gpu_drone_count;
                    self.simulation.resize(self.project.drone_count.min(384));
                }
            }
            self.project.simulation.execution_mode = self.simulation.settings.execution_mode;
            self.status_message = Some((
                match self.simulation.settings.execution_mode {
                    FleetExecutionMode::FlightValidated => {
                        "Flight-validated mode · full per-aircraft safety telemetry".to_owned()
                    }
                    FleetExecutionMode::GpuResident => {
                        "GPU-resident mode · procedural high-count preview".to_owned()
                    }
                },
                4.0,
            ));
        }
        ui.add_space(12.0);
        section_heading(ui, "AIRCRAFT", "INDEPENDENT");
        ui.add_space(9.0);
        let mut count = self.project.drone_count as u32;
        let maximum = if self.simulation.settings.execution_mode == FleetExecutionMode::GpuResident
        {
            1_000_000
        } else {
            5_000
        };
        if ui
            .add(
                egui::Slider::new(&mut count, 100..=maximum)
                    .text("Drone count")
                    .logarithmic(true),
            )
            .changed()
        {
            self.project.drone_count = count as usize;
            match self.simulation.settings.execution_mode {
                FleetExecutionMode::FlightValidated => {
                    self.validated_drone_count = self.project.drone_count;
                }
                FleetExecutionMode::GpuResident => {
                    self.gpu_drone_count = self.project.drone_count;
                }
            }
        }
        if self.simulation.settings.execution_mode == FleetExecutionMode::FlightValidated
            && self.project.drone_count != self.simulation.drones.len()
            && ui.button("APPLY FLEET SIZE").clicked()
        {
            self.clear_run_history();
            self.simulation.resize(self.project.drone_count);
            self.show_time = 0.0;
        }
        if self.simulation.settings.execution_mode == FleetExecutionMode::GpuResident {
            ui.add_space(8.0);
            let correction_response = ui.checkbox(
                &mut self.simulation.settings.gpu_collision_correction,
                "GPU collision correction",
            );
            if correction_response.changed() {
                self.project.simulation.gpu_collision_correction =
                    self.simulation.settings.gpu_collision_correction;
            }
            correction_response.on_hover_text(
                "Runs an adaptive set of GPU-resident spatial correction passes before rendering. Enabled by default for safety-first choreography.",
            );
            let response = ui.checkbox(
                &mut self.simulation.settings.gpu_safety_audit,
                "GPU safety certification",
            );
            if response.changed() {
                self.project.simulation.gpu_safety_audit =
                    self.simulation.settings.gpu_safety_audit;
                if !self.simulation.settings.gpu_safety_audit {
                    if let Ok(mut telemetry) = self.gpu_safety_telemetry.lock() {
                        telemetry.valid = false;
                    }
                }
            }
            response.on_hover_text(
                "Independently audits every rendered aircraft after collision correction and reports the current frame.",
            );
            ui.label(
                RichText::new(
                    "Correction is safety-first; the optional audit verifies its rendered result.",
                )
                .size(8.5)
                .color(Color32::from_gray(94)),
            );
        }
        ui.add_space(8.0);
        let air_response = ui.checkbox(
            &mut self.simulation.settings.monitor_air_envelope,
            "Monitor drone separation",
        );
        if air_response.changed() {
            self.project.simulation.monitor_air_envelope =
                self.simulation.settings.monitor_air_envelope;
            self.clear_run_history();
        }
        air_response.on_hover_text(
            "Audits warning and physical drone-to-drone separation. Height limits are not monitored; collision correction remains active even when this telemetry is off.",
        );
        let alert_response = ui.checkbox(
            &mut self.simulation.settings.visualize_safety_alerts,
            "Visualise amber/red alerts",
        );
        if alert_response.changed() {
            self.project.simulation.visualize_safety_alerts =
                self.simulation.settings.visualize_safety_alerts;
        }
        alert_response.on_hover_text(
            "Shows monitored warnings on the drones. Turning this off only hides recolouring; telemetry and correction continue.",
        );
        ui.label(
            RichText::new("Ground contact stays monitored · height envelope is not monitored.")
                .size(8.5)
                .color(Color32::from_gray(94)),
        );
        ui.add_space(12.0);
        let safety = self.simulation.safety_stats();
        let cpu_telemetry =
            self.simulation.settings.execution_mode == FleetExecutionMode::FlightValidated;
        let gpu_audit_requested = self.simulation.settings.execution_mode
            == FleetExecutionMode::GpuResident
            && self.simulation.settings.gpu_safety_audit;
        let monitor_air = self.simulation.settings.monitor_air_envelope;
        let gpu_telemetry = self
            .gpu_safety_telemetry
            .lock()
            .map(|telemetry| *telemetry)
            .unwrap_or_default();
        let telemetry_active = cpu_telemetry || (gpu_audit_requested && gpu_telemetry.valid);
        let telemetry_clear = if cpu_telemetry {
            safety.is_clear()
        } else {
            gpu_telemetry.is_clear()
        };
        section_heading(
            ui,
            "COLLISION SAFETY",
            if !gpu_audit_requested && !cpu_telemetry {
                "PAUSED"
            } else if !telemetry_active {
                "STARTING"
            } else if telemetry_clear {
                "CLEAR"
            } else {
                "BREACH"
            },
        );
        ui.add_space(7.0);
        egui::Frame::new()
            .fill(if !telemetry_active {
                Color32::from_rgb(29, 27, 48)
            } else if telemetry_clear {
                Color32::from_rgb(11, 37, 33)
            } else {
                Color32::from_rgb(58, 16, 18)
            })
            .stroke(Stroke::new(
                1.0,
                if !telemetry_active {
                    Color32::from_rgb(114, 92, 196)
                } else if telemetry_clear {
                    Color32::from_rgb(34, 133, 104)
                } else {
                    Color32::from_rgb(226, 62, 61)
                },
            ))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::symmetric(9, 7))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(if !gpu_audit_requested && !cpu_telemetry {
                        "◇ GPU PREVIEW · AUDIT PAUSED"
                    } else if !telemetry_active {
                        "◇ GPU AUDIT · WARMING UP"
                    } else if telemetry_clear && monitor_air {
                        "✓ NO DRONE COLLISION / GROUND CONTACT"
                    } else if telemetry_clear {
                        "✓ GROUND CLEAR · DRONE MONITOR OFF"
                    } else if !monitor_air {
                        "⚠ GROUND CONTACT"
                    } else {
                        "⚠ DRONE / GROUND COLLISION RISK"
                    })
                    .size(9.5)
                    .strong()
                    .color(if !telemetry_active {
                        Color32::from_rgb(174, 153, 255)
                    } else if telemetry_clear {
                        Color32::from_rgb(92, 231, 180)
                    } else {
                        Color32::from_rgb(255, 108, 93)
                    }),
                );
            });
        ui.add_space(5.0);
        if cpu_telemetry {
            if monitor_air {
                let nearest_aircraft = if safety.closest_pair.is_some() {
                    format!("{:.3} m", safety.minimum_air_separation)
                } else {
                    "≥ 0.380 m".to_owned()
                };
                info_row(ui, "Nearest aircraft", &nearest_aircraft);
                info_row(ui, "Warning pairs", &safety.near_miss_pairs.to_string());
                info_row(ui, "Collision pairs", &safety.collision_pairs.to_string());
            } else {
                info_row(ui, "Drone separation", "MONITOR OFF");
            }
            info_row(
                ui,
                "Ground clearance",
                &format!("{:.3} m", safety.minimum_ground_clearance),
            );
            info_row(ui, "Ground breaches", &safety.ground_breaches.to_string());
            info_row(ui, "Incident history", &safety.incident_events.to_string());
            info_row(ui, "Audit steps", &safety.monitored_steps.to_string());
            info_row(ui, "Height envelope", "NOT MONITORED");
            ui.label(
                RichText::new(if self.simulation.settings.visualize_safety_alerts {
                    "Visual alerts on · amber warning · red breach"
                } else {
                    "Visual alerts hidden · telemetry remains active"
                })
                .size(8.5)
                .color(Color32::from_gray(92)),
            );
        } else if gpu_audit_requested {
            if monitor_air {
                let nearest_aircraft = if gpu_telemetry.valid && gpu_telemetry.warning_pairs > 0 {
                    format!("{:.3} m", gpu_telemetry.minimum_air_separation)
                } else if gpu_telemetry.valid {
                    "≥ 0.380 m".to_owned()
                } else {
                    "AUDIT STARTING".to_owned()
                };
                info_row(ui, "Nearest aircraft", &nearest_aircraft);
                info_row(
                    ui,
                    "Warning pairs",
                    &gpu_telemetry.warning_pairs.to_string(),
                );
                info_row(
                    ui,
                    "Collision pairs",
                    &gpu_telemetry.collision_pairs.to_string(),
                );
            } else {
                info_row(ui, "Drone separation", "MONITOR OFF");
            }
            info_row(
                ui,
                "Ground breaches",
                &gpu_telemetry.ground_breaches.to_string(),
            );
            info_row(
                ui,
                "Audited frames",
                &gpu_telemetry.audited_frames.to_string(),
            );
            info_row(ui, "Height envelope", "NOT MONITORED");
            ui.label(
                RichText::new(if self.simulation.settings.visualize_safety_alerts {
                    "Current rendered frame · visual alerts on"
                } else {
                    "Current rendered frame · visual alerts hidden"
                })
                .size(8.5)
                .color(Color32::from_gray(92)),
            );
        } else {
            info_row(ui, "Formation", "COMPUTE SHADER");
            info_row(ui, "Assignment", "DETERMINISTIC GPU");
            info_row(ui, "Correction", "TIMELINE PHASE LOCK");
            ui.label(
                RichText::new("Enable GPU safety certification to audit every rendered aircraft without returning positions to the CPU.")
                    .size(8.8)
                    .color(Color32::from_gray(102)),
            );
        }
        self.collision_log_ui(ui);
        ui.add_space(12.0);
        section_heading(
            ui,
            "FLIGHT MODEL",
            &format!("{} HZ", self.simulation.settings.simulation_hz),
        );
        ui.add_space(8.0);
        info_row(ui, "Trajectory sync", "TIMELINE LOCKED");
        info_row(
            ui,
            "Tracking error",
            &format!("{:.3} m RMS", self.simulation.tracking_error_rms()),
        );
        if cpu_telemetry {
            info_row(
                ui,
                "Simulation cost",
                &format!("{:.2} ms", self.simulation.simulation_cpu_ms()),
            );
            info_row(
                ui,
                "Constraint passes",
                &self
                    .simulation
                    .constraint_iterations_last_frame()
                    .to_string(),
            );
        }
        ui.add_space(6.0);
        slider_row(
            ui,
            "Max speed",
            &mut self.simulation.settings.max_speed,
            4.0..=28.0,
            "m/s",
        );
        slider_row(
            ui,
            "Acceleration",
            &mut self.simulation.settings.max_acceleration,
            3.0..=20.0,
            "m/s²",
        );
        slider_row(
            ui,
            "Stabilisation",
            &mut self.simulation.settings.stabilization,
            2.0..=14.0,
            "",
        );
        slider_row(
            ui,
            "Separation",
            &mut self.simulation.settings.minimum_separation,
            0.35..=2.0,
            "m",
        );
        slider_row(
            ui,
            "Variation",
            &mut self.simulation.settings.organic_variation,
            0.0..=0.2,
            "m",
        );
        ui.add_space(12.0);
        section_heading(ui, "WEATHER", "NIGHT");
        slider_row(
            ui,
            "Wind",
            &mut self.simulation.settings.wind_strength,
            0.0..=2.5,
            "m/s",
        );
        slider_row(
            ui,
            "Direction",
            &mut self.project.environment.wind_direction_degrees,
            0.0..=360.0,
            "°",
        );
        slider_row(
            ui,
            "Clouds",
            &mut self.project.environment.cloud_cover,
            0.0..=1.0,
            "",
        );
    }

    fn collision_log_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(14.0);
        section_heading(
            ui,
            "RUN COLLISION LOG",
            &format!("{:04} EVENTS", self.collision_log.records().len()),
        );
        ui.add_space(7.0);
        info_row(
            ui,
            "Cumulative drone collisions",
            &self.collision_log.cumulative_drone_collisions().to_string(),
        );
        info_row(
            ui,
            "Cumulative ground contacts",
            &self.collision_log.cumulative_ground_contacts().to_string(),
        );
        info_row(ui, "Run elapsed", &format_time(self.run_elapsed));
        if self.collision_log.dropped_records() > 0 {
            info_row(
                ui,
                "Older records omitted",
                &self.collision_log.dropped_records().to_string(),
            );
        }

        let record_count = self.collision_log.records().len();
        let mut seek_to = None;
        let mut replay_from = None;
        egui::CollapsingHeader::new(format!("INCIDENT DETAILS · {record_count}"))
            .default_open(record_count > 0)
            .show(ui, |ui| {
                if record_count == 0 {
                    ui.label(
                        RichText::new("No collision or ground-contact episodes in this run.")
                            .size(8.8)
                            .color(Color32::from_gray(102)),
                    );
                    return;
                }
                egui::ScrollArea::vertical().max_height(230.0).show_rows(
                    ui,
                    76.0,
                    record_count,
                    |ui, row_range| {
                        for row in row_range {
                            let record = &self.collision_log.records()[record_count - 1 - row];
                            let danger = match record.kind {
                                crate::safety_log::CollisionKind::DroneToDrone => {
                                    Color32::from_rgb(255, 104, 94)
                                }
                                crate::safety_log::CollisionKind::GroundContact => {
                                    Color32::from_rgb(255, 171, 73)
                                }
                            };
                            egui::Frame::new()
                                .fill(Color32::from_rgb(16, 21, 29))
                                .stroke(Stroke::new(1.0, danger.gamma_multiply(0.55)))
                                .corner_radius(4.0)
                                .inner_margin(egui::Margin::symmetric(7, 6))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(format!(
                                                "#{:04}  {} ×{}",
                                                record.id,
                                                record.kind.label(),
                                                record.count
                                            ))
                                            .size(8.8)
                                            .strong()
                                            .color(danger),
                                        );
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if compact_button(ui, "REPLAY").clicked() {
                                                    replay_from =
                                                        Some((record.show_time - 1.0).max(0.0));
                                                }
                                                if compact_button(ui, "GO TO").clicked() {
                                                    seek_to = Some(record.show_time);
                                                }
                                            },
                                        );
                                    });
                                    ui.label(
                                        RichText::new(format!(
                                            "show {} · run {} · clearance {:.3} m",
                                            format_time(record.show_time),
                                            format_time(record.run_time),
                                            record.measured_clearance
                                        ))
                                        .size(8.3)
                                        .monospace()
                                        .color(Color32::from_gray(145)),
                                    );
                                    let drones = match (record.drone_a, record.drone_b) {
                                        (Some(a), Some(b)) => format!("drones {a} / {b}"),
                                        (Some(a), None) => format!("drone {a}"),
                                        _ => "representative unavailable".to_owned(),
                                    };
                                    ui.label(
                                        RichText::new(format!(
                                            "{} · world x{:.2} y{:.2} z{:.2}",
                                            drones,
                                            record.position.x,
                                            record.position.y,
                                            record.position.z
                                        ))
                                        .size(8.3)
                                        .monospace()
                                        .color(Color32::from_gray(115)),
                                    );
                                });
                            ui.add_space(5.0);
                        }
                    },
                );
            });
        ui.horizontal(|ui| {
            if compact_button(ui, "CLEAR RUN LOG").clicked() {
                self.clear_run_history();
            }
            ui.label(
                RichText::new("GO TO pauses at the audited timestamp · REPLAY starts 1 s earlier")
                    .size(8.0)
                    .color(Color32::from_gray(86)),
            );
        });
        if let Some(time) = seek_to {
            self.set_time(time);
            self.selected_cue = self.project.timeline.sample(self.show_time).index;
            self.playing = false;
        } else if let Some(time) = replay_from {
            self.set_time(time);
            self.selected_cue = self.project.timeline.sample(self.show_time).index;
            self.playing = true;
        }
    }

    fn look_inspector(&mut self, ui: &mut egui::Ui) {
        section_heading(ui, "RENDERING", "WGPU · HDR");
        ui.add_space(9.0);
        egui::ComboBox::from_id_salt("graphics_preset")
            .selected_text(self.project.graphics.preset.label())
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for preset in [
                    GraphicsPreset::Low,
                    GraphicsPreset::Medium,
                    GraphicsPreset::High,
                    GraphicsPreset::Cinematic,
                ] {
                    if ui
                        .selectable_label(self.project.graphics.preset == preset, preset.label())
                        .clicked()
                    {
                        self.project.graphics.apply_preset(preset);
                    }
                }
            });
        slider_row(
            ui,
            "Exposure",
            &mut self.project.graphics.exposure,
            0.35..=2.2,
            "EV",
        );
        slider_row(ui, "Bloom", &mut self.project.graphics.bloom, 0.0..=2.5, "");
        slider_row(
            ui,
            "Light size",
            &mut self.project.graphics.light_size,
            0.45..=2.0,
            "",
        );
        slider_row(
            ui,
            "Saturation",
            &mut self.project.graphics.saturation,
            0.5..=2.0,
            "×",
        );
        slider_row(
            ui,
            "Atmosphere",
            &mut self.project.graphics.haze,
            0.0..=1.0,
            "",
        );
        slider_row(
            ui,
            "Reflections",
            &mut self.project.graphics.ground_reflections,
            0.0..=1.0,
            "",
        );
        slider_row(
            ui,
            "Render scale",
            &mut self.project.graphics.render_scale,
            0.5..=1.0,
            "×",
        );
        if self.project.graphics.preset != GraphicsPreset::Custom {
            ui.label(
                RichText::new("Adjusting a control creates a custom look.")
                    .size(9.0)
                    .color(Color32::from_gray(96)),
            );
        }
        ui.add_space(16.0);
        section_heading(ui, "PIPELINE", "ACTIVE");
        for (label, value) in [
            ("GPU instancing", "ON"),
            ("Compute preparation", "ON"),
            ("Filmic tone map", "ACES"),
            ("Distance LOD", "AUTO"),
            ("Fixed timestep", "60 HZ"),
        ] {
            info_row(ui, label, value);
        }
        ui.add_space(18.0);
        section_heading(ui, "CREATOR LINKS", VERSION_LABEL);
        ui.add_space(8.0);
        creator_link(ui, "CODE", "github.com/CHCOfficial", CODE_URL);
        creator_link(ui, "GRAPHICS", "deviantart.com/chcofficial", GRAPHICS_URL);
        creator_link(ui, "AUDIO", "suno.com/@artfulexpchc", AUDIO_URL);
        ui.add_space(6.0);
        ui.label(
            RichText::new("These creator links are part of the project licence and must remain in redistributed versions.")
                .size(8.5)
                .color(Color32::from_gray(92)),
        );
    }

    fn bottom_timeline(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("timeline")
            .default_height(154.0)
            .height_range(112.0..=220.0)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(8, 12, 18))
                    .inner_margin(egui::Margin::symmetric(16, 9)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if transport_button(ui, "↺").on_hover_text("Restart").clicked() {
                        self.restart();
                    }
                    if transport_button(ui, if self.playing { "Ⅱ" } else { "▶" }).clicked() {
                        self.playing = !self.playing;
                    }
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format_time(self.show_time))
                            .monospace()
                            .size(13.0)
                            .color(Color32::from_rgb(105, 223, 231)),
                    );
                    ui.label(
                        RichText::new(format!(
                            "/  {}",
                            format_time(self.project.timeline.duration())
                        ))
                        .monospace()
                        .size(11.0)
                        .color(Color32::from_gray(82)),
                    );
                    ui.add_space(16.0);
                    let mut time = self.show_time;
                    if ui
                        .add_sized(
                            [(ui.available_width() - 264.0).max(80.0), 18.0],
                            egui::Slider::new(&mut time, 0.0..=self.project.timeline.duration())
                                .show_value(false),
                        )
                        .dragged()
                    {
                        self.playing = false;
                        self.set_time(time);
                    }
                    ui.add_space(10.0);
                    if compact_button(ui, "DUPLICATE").clicked() {
                        self.project.timeline.duplicate_cue(self.selected_cue);
                    }
                    if compact_button(ui, "◀").clicked() {
                        self.project.timeline.move_cue(self.selected_cue, -1);
                        self.selected_cue = self.selected_cue.saturating_sub(1);
                    }
                    if compact_button(ui, "▶").clicked() {
                        self.project.timeline.move_cue(self.selected_cue, 1);
                        self.selected_cue = (self.selected_cue + 1)
                            .min(self.project.timeline.cues.len().saturating_sub(1));
                    }
                });
                ui.add_space(10.0);
                let total = self.project.timeline.duration().max(1.0);
                let mut clicked_cue = None;
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (index, cue) in self.project.timeline.cues.iter().enumerate() {
                            let width = (cue.duration / total * 1080.0).clamp(68.0, 154.0);
                            let selected = index == self.selected_cue;
                            let active =
                                self.project.timeline.sample(self.show_time).index == index;
                            let color = cue_color(cue.kind.label(), active);
                            let response = egui::Frame::new()
                                .fill(if selected {
                                    Color32::from_rgb(24, 38, 49)
                                } else {
                                    Color32::from_rgb(14, 21, 30)
                                })
                                .stroke(Stroke::new(if active { 2.0 } else { 1.0 }, color))
                                .corner_radius(5.0)
                                .inner_margin(egui::Margin::symmetric(9, 8))
                                .show(ui, |ui| {
                                    ui.set_width(width);
                                    ui.label(
                                        RichText::new(cue.kind.label())
                                            .size(8.5)
                                            .strong()
                                            .color(color),
                                    );
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(&cue.name)
                                                .size(10.5)
                                                .color(Color32::from_gray(211)),
                                        )
                                        .truncate(),
                                    );
                                    ui.label(
                                        RichText::new(format!("{:.1}s", cue.duration))
                                            .size(9.0)
                                            .monospace()
                                            .color(Color32::from_gray(88)),
                                    );
                                })
                                .response
                                .interact(egui::Sense::click());
                            if response.clicked() {
                                clicked_cue = Some(index);
                            }
                        }
                    });
                });
                if let Some(index) = clicked_cue {
                    self.selected_cue = index;
                    self.set_time(self.project.timeline.cue_start(index) + 0.02);
                    self.playing = false;
                }
            });
    }

    fn viewport(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::BLACK))
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
                if self.camera.mode == CameraMode::FreeFly {
                    if response.clicked() {
                        self.free_fly_look_active = true;
                    }
                    if self.free_fly_look_active {
                        ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(
                            egui::CursorGrab::Locked,
                        ));
                        ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(false));
                        let delta = ctx.input(|input| {
                            input
                                .pointer
                                .motion()
                                .unwrap_or_else(|| input.pointer.delta())
                        });
                        self.camera.free_look(delta.x, delta.y);
                    }
                    if response.hovered() || self.free_fly_look_active {
                        let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
                        if scroll.abs() > 0.0 {
                            self.camera.adjust_free_speed(scroll);
                        }
                    }
                } else {
                    if response.dragged() {
                        let delta = ctx.input(|input| input.pointer.delta());
                        self.camera.orbit(delta.x, delta.y);
                    }
                    if response.hovered() {
                        let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
                        if scroll.abs() > 0.0 {
                            self.camera.zoom(scroll);
                        }
                    }
                }
                ui.painter().add(renderer::make_callback(
                    rect,
                    &self.simulation,
                    &self.camera,
                    &self.project.graphics,
                    &self.project.environment,
                    &self.project.timeline,
                    self.project.drone_count,
                    self.simulation.settings.execution_mode,
                    self.simulation.settings.gpu_collision_correction,
                    self.simulation.settings.gpu_safety_audit,
                    self.simulation.settings.visualize_safety_alerts,
                    self.simulation.settings.monitor_air_envelope,
                    &self.gpu_safety_telemetry,
                    &self.gpu_image_instances,
                    self.show_time,
                ));

                if self.camera.mode == CameraMode::FreeFly {
                    egui::Area::new("free_fly_status".into())
                        .fixed_pos(rect.center_bottom() + egui::vec2(-150.0, -42.0))
                        .show(ctx, |ui| {
                            egui::Frame::new()
                                .fill(Color32::from_black_alpha(185))
                                .stroke(Stroke::new(1.0, Color32::from_rgb(44, 113, 126)))
                                .corner_radius(5.0)
                                .inner_margin(egui::Margin::symmetric(10, 7))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new(if self.free_fly_look_active {
                                            "FREE FLY · WASD MOVE · MOUSE LOOK · ESC RELEASE"
                                        } else {
                                            "FREE FLY · CLICK VIEWPORT TO CAPTURE MOUSE"
                                        })
                                        .size(9.0)
                                        .monospace()
                                        .color(Color32::from_rgb(117, 224, 231)),
                                    );
                                });
                        });
                }

                let sample = self.project.timeline.sample(self.show_time);
                let safety = self.simulation.safety_stats();
                let cpu_telemetry =
                    self.simulation.settings.execution_mode == FleetExecutionMode::FlightValidated;
                let gpu_audit_requested = self.simulation.settings.execution_mode
                    == FleetExecutionMode::GpuResident
                    && self.simulation.settings.gpu_safety_audit;
                let monitor_air = self.simulation.settings.monitor_air_envelope;
                let gpu_telemetry = self
                    .gpu_safety_telemetry
                    .lock()
                    .map(|telemetry| *telemetry)
                    .unwrap_or_default();
                let telemetry_active =
                    cpu_telemetry || (gpu_audit_requested && gpu_telemetry.valid);
                let telemetry_clear = if cpu_telemetry {
                    safety.is_clear()
                } else {
                    gpu_telemetry.is_clear()
                };
                let run_drone_collisions = self.collision_log.cumulative_drone_collisions();
                let run_ground_contacts = self.collision_log.cumulative_ground_contacts();
                let safety_status_text = if cpu_telemetry && monitor_air {
                    format!(
                        "{}  DRONES {}  ·  GROUND {:.3} M  ·  RUN D{} / G{}",
                        if safety.is_clear() {
                            "✓ SAFETY CLEAR"
                        } else {
                            "⚠ BREACH"
                        },
                        if safety.closest_pair.is_some() {
                            format!("{:.3} M", safety.minimum_air_separation)
                        } else {
                            "≥0.380 M".to_owned()
                        },
                        safety.minimum_ground_clearance,
                        run_drone_collisions,
                        run_ground_contacts,
                    )
                } else if cpu_telemetry {
                    format!(
                        "{}  DRONE MONITOR OFF  ·  GROUND {:.3} M  ·  RUN G{}",
                        if safety.is_clear() {
                            "✓ GROUND CLEAR"
                        } else {
                            "⚠ GROUND CONTACT"
                        },
                        safety.minimum_ground_clearance,
                        run_ground_contacts,
                    )
                } else if gpu_audit_requested && gpu_telemetry.valid && monitor_air {
                    format!(
                        "{}  DRONES {}  ·  NOW D{} / G{}  ·  RUN D{} / G{}",
                        if gpu_telemetry.is_clear() {
                            "✓ GPU AUDIT CLEAR"
                        } else {
                            "⚠ GPU AUDIT BREACH"
                        },
                        if gpu_telemetry.warning_pairs > 0 {
                            format!("{:.3} M", gpu_telemetry.minimum_air_separation)
                        } else {
                            "≥0.380 M".to_owned()
                        },
                        gpu_telemetry.collision_pairs,
                        gpu_telemetry.ground_breaches,
                        run_drone_collisions,
                        run_ground_contacts,
                    )
                } else if gpu_audit_requested && gpu_telemetry.valid {
                    format!(
                        "{}  DRONE MONITOR OFF  ·  GROUND NOW {}  ·  RUN {}",
                        if gpu_telemetry.ground_breaches == 0 {
                            "✓ GPU GROUND CLEAR"
                        } else {
                            "⚠ GPU GROUND CONTACT"
                        },
                        gpu_telemetry.ground_breaches,
                        run_ground_contacts,
                    )
                } else if gpu_audit_requested {
                    format!(
                        "◇ GPU RESIDENT  ·  {} DRONES  ·  SAFETY AUDIT WARMING UP",
                        self.project.drone_count
                    )
                } else {
                    format!(
                        "◇ GPU RESIDENT  ·  {} DRONES  ·  SAFETY AUDIT PAUSED",
                        self.project.drone_count
                    )
                };
                egui::Area::new("viewport_status".into())
                    .fixed_pos(rect.left_top() + egui::vec2(18.0, 17.0))
                    .show(ctx, |ui| {
                        egui::Frame::new()
                            .fill(Color32::from_black_alpha(155))
                            .corner_radius(5.0)
                            .inner_margin(egui::Margin::symmetric(10, 7))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new("●  LIVE")
                                            .size(9.0)
                                            .strong()
                                            .color(Color32::from_rgb(71, 222, 184)),
                                    );
                                    ui.separator();
                                    ui.label(
                                        RichText::new(
                                            self.current_formation().label().to_uppercase(),
                                        )
                                        .size(9.0)
                                        .color(Color32::from_gray(202)),
                                    );
                                    ui.label(
                                        RichText::new(format!(
                                            "{:03}%",
                                            (sample.progress * 100.0) as u32
                                        ))
                                        .size(9.0)
                                        .monospace()
                                        .color(Color32::from_gray(112)),
                                    );
                                });
                            });
                    });
                egui::Area::new("safety_status".into())
                    .fixed_pos(rect.left_bottom() + egui::vec2(18.0, -50.0))
                    .show(ctx, |ui| {
                        egui::Frame::new()
                            .fill(if !telemetry_active {
                                Color32::from_rgba_premultiplied(30, 23, 58, 230)
                            } else if telemetry_clear {
                                Color32::from_rgba_premultiplied(8, 45, 35, 225)
                            } else {
                                Color32::from_rgba_premultiplied(77, 12, 17, 235)
                            })
                            .stroke(Stroke::new(
                                1.0,
                                if !telemetry_active {
                                    Color32::from_rgb(132, 107, 218)
                                } else if telemetry_clear {
                                    Color32::from_rgb(41, 166, 125)
                                } else {
                                    Color32::from_rgb(255, 72, 69)
                                },
                            ))
                            .corner_radius(5.0)
                            .inner_margin(egui::Margin::symmetric(10, 7))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(&safety_status_text)
                                        .size(9.0)
                                        .monospace()
                                        .strong()
                                        .color(if !telemetry_active {
                                            Color32::from_rgb(184, 164, 255)
                                        } else if telemetry_clear {
                                            Color32::from_rgb(108, 235, 186)
                                        } else {
                                            Color32::from_rgb(255, 119, 104)
                                        }),
                                );
                            });
                    });
                egui::Area::new("performance_status".into())
                    .anchor(
                        egui::Align2::RIGHT_TOP,
                        [
                            rect.right() - ctx.screen_rect().right() - 12.0,
                            rect.top() - ctx.screen_rect().top() + 12.0,
                        ],
                    )
                    .show(ctx, |ui| {
                        ui.label(
                            RichText::new(format!(
                                "{:.0} FPS   ·   {:.1} MS MAX   ·   {}   ·   {} DRONES",
                                self.profiler.fps(),
                                self.profiler.worst_frame_ms(),
                                if cpu_telemetry {
                                    format!(
                                        "SIM {:.1} MS · SYNC {:.3} M",
                                        self.simulation.simulation_cpu_ms(),
                                        self.simulation.tracking_error_rms()
                                    )
                                } else if gpu_audit_requested {
                                    "GPU PHASE LOCK · AUDIT ON".to_owned()
                                } else {
                                    "GPU PHASE LOCK".to_owned()
                                },
                                self.project.drone_count
                            ))
                            .size(9.0)
                            .monospace()
                            .color(Color32::from_gray(121)),
                        );
                    });
                if let Some((message, _)) = &self.status_message {
                    egui::Area::new("toast".into())
                        .anchor(
                            egui::Align2::CENTER_TOP,
                            [
                                rect.center().x - ctx.screen_rect().center().x,
                                rect.top() - ctx.screen_rect().top() + 18.0,
                            ],
                        )
                        .show(ctx, |ui| {
                            egui::Frame::new()
                                .fill(Color32::from_rgba_premultiplied(12, 27, 35, 235))
                                .stroke(Stroke::new(1.0, Color32::from_rgb(35, 108, 119)))
                                .corner_radius(6.0)
                                .inner_margin(egui::Margin::symmetric(14, 8))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new(message)
                                            .size(10.5)
                                            .color(Color32::from_rgb(186, 231, 235)),
                                    );
                                });
                        });
                }
            });
    }
}

impl eframe::App for DroneShowApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.profiler.record(dt);

        let dropped_paths: Vec<_> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect()
        });
        if let Some(path) = dropped_paths.first() {
            self.image_path_input = path.display().to_string();
            self.import_image(path);
        }

        if ctx.input(|input| input.key_pressed(egui::Key::Space)) {
            self.playing = !self.playing;
        }
        let escape_pressed = ctx.input(|input| input.key_pressed(egui::Key::Escape));
        if escape_pressed && self.free_fly_look_active {
            self.free_fly_look_active = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
        } else if escape_pressed && self.presentation_mode {
            self.presentation_mode = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        }
        if ctx.input(|input| input.key_pressed(egui::Key::F11)) {
            self.presentation_mode = !self.presentation_mode;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.presentation_mode));
        }

        if self.playing {
            self.run_elapsed += dt;
            self.show_time += dt;
            if self.show_time >= self.project.timeline.duration() {
                self.show_time = self
                    .show_time
                    .rem_euclid(self.project.timeline.duration().max(0.01));
            }
        }
        self.update_image_animation();
        if self.simulation.settings.execution_mode == FleetExecutionMode::FlightValidated {
            self.simulation.step_frame(
                dt,
                self.show_time,
                &self.project.timeline,
                self.project.environment.wind_direction_degrees,
            );
        }
        self.observe_safety();
        if self.camera.mode == CameraMode::FreeFly && self.free_fly_look_active {
            let (movement, sprint) = ctx.input(|input| {
                let axis = |positive, negative| {
                    (if input.key_down(positive) { 1.0 } else { 0.0 })
                        - (if input.key_down(negative) { 1.0 } else { 0.0 })
                };
                (
                    glam::Vec3::new(
                        axis(egui::Key::D, egui::Key::A),
                        axis(egui::Key::E, egui::Key::Q),
                        axis(egui::Key::W, egui::Key::S),
                    ),
                    input.modifiers.shift,
                )
            });
            self.camera.move_free(movement, dt, sprint);
        }
        if self.free_fly_look_active && !ctx.input(|input| input.focused) {
            self.free_fly_look_active = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
        }
        if self.camera.mode != CameraMode::FreeFly && self.free_fly_look_active {
            self.free_fly_look_active = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(egui::CursorGrab::None));
            ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(true));
        }
        self.camera.update(
            dt,
            self.show_time,
            self.current_formation(),
            self.project.drone_count,
        );
        if let Some((_, remaining)) = &mut self.status_message {
            *remaining -= dt;
            if *remaining <= 0.0 {
                self.status_message = None;
            }
        }

        if self.presentation_mode {
            self.viewport(ctx);
            egui::Area::new("exit_presentation".into())
                .anchor(egui::Align2::RIGHT_BOTTOM, [-18.0, -18.0])
                .show(ctx, |ui| {
                    if ui
                        .add(egui::Button::new("EXIT  ESC").fill(Color32::from_black_alpha(145)))
                        .clicked()
                    {
                        self.presentation_mode = false;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                    }
                });
        } else {
            self.top_bar(ctx);
            self.bottom_timeline(ctx);
            self.left_panel(ctx);
            self.right_panel(ctx);
            self.viewport(ctx);
        }
        ctx.request_repaint();
    }
}

fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals.dark_mode = true;
    style.visuals.panel_fill = Color32::from_rgb(9, 14, 21);
    style.visuals.window_fill = Color32::from_rgb(11, 17, 25);
    style.visuals.extreme_bg_color = Color32::from_rgb(6, 10, 16);
    style.visuals.faint_bg_color = Color32::from_rgb(14, 22, 31);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(16, 24, 34);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(21, 48, 57);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(23, 72, 82);
    style.visuals.selection.bg_fill = Color32::from_rgb(24, 102, 116);
    style.visuals.override_text_color = Some(Color32::from_rgb(205, 216, 224));
    style.spacing.item_spacing = egui::vec2(7.0, 6.0);
    style.spacing.button_padding = egui::vec2(8.0, 5.0);
    ctx.set_style(style);
}

fn panel_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(Color32::from_rgb(9, 14, 21))
        .inner_margin(egui::Margin::symmetric(14, 14))
        .stroke(Stroke::new(1.0, Color32::from_rgb(22, 29, 39)))
}

fn section_heading(ui: &mut egui::Ui, title: &str, trailing: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(title)
                .size(9.5)
                .strong()
                .color(Color32::from_gray(151)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(trailing)
                    .size(8.5)
                    .monospace()
                    .color(Color32::from_rgb(60, 146, 159)),
            );
        });
    });
}

fn info_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .size(10.5)
                .color(Color32::from_gray(120)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(value)
                    .size(10.5)
                    .color(Color32::from_gray(210)),
            );
        });
    });
}

fn creator_link(ui: &mut egui::Ui, category: &str, label: &str, url: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(category)
                .size(9.0)
                .strong()
                .color(Color32::from_gray(116)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(
                egui::Hyperlink::from_label_and_url(
                    RichText::new(label)
                        .size(9.5)
                        .color(Color32::from_rgb(90, 211, 220)),
                    url,
                )
                .open_in_new_tab(true),
            );
        });
    });
}

fn slider_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .size(10.0)
                .color(Color32::from_gray(130)),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_sized(
                [ui.available_width().clamp(96.0, 164.0), 18.0],
                egui::Slider::new(value, range).suffix(suffix),
            );
        });
    });
}

fn tab(ui: &mut egui::Ui, active: &mut InspectorTab, value: InspectorTab, label: &str) {
    let selected = *active == value;
    if ui
        .add(
            egui::Button::new(RichText::new(label).size(9.5).strong().color(if selected {
                Color32::from_rgb(102, 222, 229)
            } else {
                Color32::from_gray(99)
            }))
            .fill(if selected {
                Color32::from_rgb(17, 45, 53)
            } else {
                Color32::TRANSPARENT
            })
            .stroke(Stroke::NONE)
            .min_size(egui::vec2(76.0, 28.0)),
        )
        .clicked()
    {
        *active = value;
    }
}

fn compact_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).size(9.0).strong())
            .fill(Color32::from_rgb(16, 24, 33))
            .stroke(Stroke::new(1.0, Color32::from_rgb(35, 46, 59)))
            .corner_radius(4.0),
    )
}

fn transport_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(
            RichText::new(label)
                .size(15.0)
                .color(Color32::from_rgb(181, 235, 238)),
        )
        .fill(Color32::from_rgb(16, 42, 49))
        .stroke(Stroke::new(1.0, Color32::from_rgb(34, 91, 102)))
        .corner_radius(5.0)
        .min_size(egui::vec2(34.0, 30.0)),
    )
}

fn cue_color(label: &str, active: bool) -> Color32 {
    let base = match label {
        "LAUNCH" | "LAND" => Color32::from_rgb(76, 183, 158),
        "MORPH" => Color32::from_rgb(119, 100, 219),
        "ANIMATE" => Color32::from_rgb(37, 151, 181),
        "COLOUR" => Color32::from_rgb(214, 83, 139),
        _ => Color32::from_rgb(123, 138, 151),
    };
    if active {
        base
    } else {
        Color32::from_rgb(base.r() / 2, base.g() / 2, base.b() / 2)
    }
}

fn format_time(seconds: f32) -> String {
    let minutes = (seconds / 60.0) as u32;
    let remainder = seconds - minutes as f32 * 60.0;
    format!("{minutes:02}:{remainder:04.1}")
}
