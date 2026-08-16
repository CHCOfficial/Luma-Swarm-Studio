use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Rgbw {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub w: f32,
}

impl Rgbw {
    pub const OFF: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    pub const CYAN: Self = Self::new(0.05, 0.65, 1.0, 0.18);
    pub const MAGENTA: Self = Self::new(1.0, 0.04, 0.42, 0.08);
    pub const GOLD: Self = Self::new(1.0, 0.38, 0.04, 0.22);

    pub const fn new(r: f32, g: f32, b: f32, w: f32) -> Self {
        Self { r, g, b, w }
    }

    pub fn rgb(self) -> Vec3 {
        Vec3::new(self.r + self.w, self.g + self.w, self.b + self.w)
    }

    pub fn mix(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self::new(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.w + (other.w - self.w) * t,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FormationPoint {
    pub position: Vec3,
    pub color: Rgbw,
    pub brightness: f32,
    pub phase: f32,
    pub group: u16,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FormationKind {
    LaunchGrid,
    Chrysalis,
    Heart,
    Galaxy,
    Cathedral,
    Human,
    Planet,
    Infinity,
    Lotus,
    Crown,
    EventHorizon,
    Mandala,
    Gyroscope,
    Image,
}

impl FormationKind {
    pub const CORE_SHOWCASE: [Self; 9] = [
        Self::Chrysalis,
        Self::Heart,
        Self::Galaxy,
        Self::Cathedral,
        Self::Human,
        Self::Planet,
        Self::Infinity,
        Self::Lotus,
        Self::Crown,
    ];
    pub const BONUS: [Self; 3] = [Self::EventHorizon, Self::Mandala, Self::Gyroscope];
    pub const SHOWCASE: [Self; 12] = [
        Self::Chrysalis,
        Self::Heart,
        Self::Galaxy,
        Self::Cathedral,
        Self::Human,
        Self::Planet,
        Self::Infinity,
        Self::Lotus,
        Self::Crown,
        Self::EventHorizon,
        Self::Mandala,
        Self::Gyroscope,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::LaunchGrid => "Launch grid",
            Self::Chrysalis => "Stellar chrysalis",
            Self::Heart => "Neon heart",
            Self::Galaxy => "Spiral galaxy",
            Self::Cathedral => "Prism cathedral",
            Self::Human => "Chromatic helix",
            Self::Planet => "Ringed planet",
            Self::Infinity => "Infinity portal",
            Self::Lotus => "Prismatic lotus",
            Self::Crown => "Celestial crown",
            Self::EventHorizon => "Event horizon",
            Self::Mandala => "Spectral mandala",
            Self::Gyroscope => "Chrono gyroscope",
            Self::Image => "Imported image",
        }
    }

    pub fn glyph(self) -> &'static str {
        match self {
            Self::LaunchGrid => "▦",
            Self::Chrysalis => "◈",
            Self::Heart => "♥",
            Self::Galaxy => "✦",
            Self::Cathedral => "⌑",
            Self::Human => "◇",
            Self::Planet => "⊚",
            Self::Infinity => "∞",
            Self::Lotus => "✺",
            Self::Crown => "♕",
            Self::EventHorizon => "◉",
            Self::Mandala => "✹",
            Self::Gyroscope => "◎",
            Self::Image => "▧",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum FleetExecutionMode {
    #[default]
    FlightValidated,
    GpuResident,
}

impl FleetExecutionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::FlightValidated => "Flight validated",
            Self::GpuResident => "GPU resident",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Drone {
    pub id: u32,
    pub position: Vec3,
    pub previous_position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub orientation: Quat,
    pub slot: usize,
    pub color: Rgbw,
    pub brightness: f32,
    pub phase: f32,
    pub rotor_angle: f32,
    pub battery: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SimulationSettings {
    #[serde(default)]
    pub execution_mode: FleetExecutionMode,
    #[serde(default = "default_gpu_safety_audit")]
    pub gpu_safety_audit: bool,
    #[serde(default = "default_gpu_collision_correction")]
    pub gpu_collision_correction: bool,
    #[serde(default = "default_visualize_safety_alerts")]
    pub visualize_safety_alerts: bool,
    #[serde(default = "default_monitor_drone_separation")]
    pub monitor_air_envelope: bool,
    pub max_speed: f32,
    pub max_acceleration: f32,
    pub turn_rate: f32,
    pub stabilization: f32,
    pub minimum_separation: f32,
    pub wind_strength: f32,
    pub organic_variation: f32,
    pub simulation_hz: u32,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            execution_mode: FleetExecutionMode::GpuResident,
            gpu_safety_audit: true,
            gpu_collision_correction: true,
            visualize_safety_alerts: true,
            monitor_air_envelope: true,
            max_speed: 16.0,
            max_acceleration: 11.0,
            turn_rate: 4.5,
            stabilization: 7.5,
            minimum_separation: 0.72,
            wind_strength: 0.0,
            organic_variation: 0.045,
            simulation_hz: 60,
        }
    }
}

fn default_gpu_collision_correction() -> bool {
    true
}

fn default_gpu_safety_audit() -> bool {
    true
}

fn default_monitor_drone_separation() -> bool {
    true
}

fn default_visualize_safety_alerts() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphicsPreset {
    Low,
    Medium,
    High,
    Cinematic,
    Custom,
}

impl GraphicsPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Cinematic => "Cinematic",
            Self::Custom => "Custom",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GraphicsSettings {
    pub preset: GraphicsPreset,
    pub exposure: f32,
    pub bloom: f32,
    pub haze: f32,
    pub ground_reflections: f32,
    pub render_scale: f32,
    pub light_size: f32,
    #[serde(default = "default_saturation")]
    pub saturation: f32,
}

impl GraphicsSettings {
    pub fn apply_preset(&mut self, preset: GraphicsPreset) {
        self.preset = preset;
        let values = match preset {
            GraphicsPreset::Low => (0.9, 0.45, 0.08, 0.2, 0.7, 0.82, 1.1),
            GraphicsPreset::Medium => (1.0, 0.75, 0.18, 0.45, 0.85, 0.92, 1.18),
            GraphicsPreset::High => (2.2, 2.0, 0.27, 0.68, 1.0, 1.36, 1.28),
            GraphicsPreset::Cinematic => (1.18, 1.35, 0.38, 0.85, 1.0, 1.12, 1.32),
            GraphicsPreset::Custom => return,
        };
        self.exposure = values.0;
        self.bloom = values.1;
        self.haze = values.2;
        self.ground_reflections = values.3;
        self.render_scale = values.4;
        self.light_size = values.5;
        self.saturation = values.6;
    }
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        let mut settings = Self {
            preset: GraphicsPreset::High,
            exposure: 2.2,
            bloom: 2.0,
            haze: 0.27,
            ground_reflections: 0.68,
            render_scale: 1.0,
            light_size: 1.36,
            saturation: default_saturation(),
        };
        settings.apply_preset(GraphicsPreset::High);
        settings
    }
}

fn default_saturation() -> f32 {
    1.28
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentSettings {
    pub wind_direction_degrees: f32,
    pub cloud_cover: f32,
    pub field_reflectivity: f32,
    pub star_brightness: f32,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            wind_direction_degrees: 32.0,
            cloud_cover: 0.3,
            field_reflectivity: 0.68,
            star_brightness: 0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_high_quality_defaults_are_stable() {
        let graphics = GraphicsSettings::default();
        let simulation = SimulationSettings::default();
        assert_eq!(graphics.exposure, 2.20);
        assert_eq!(graphics.bloom, 2.0);
        assert_eq!(graphics.light_size, 1.36);
        assert_eq!(graphics.saturation, 1.28);
        assert_eq!(graphics.preset, GraphicsPreset::High);
        assert_eq!(simulation.execution_mode, FleetExecutionMode::GpuResident);
        assert!(simulation.gpu_collision_correction);
        assert!(simulation.visualize_safety_alerts);
        assert!(simulation.gpu_safety_audit);
        assert!(simulation.monitor_air_envelope);
        assert_eq!(simulation.wind_strength, 0.0);
    }
}
