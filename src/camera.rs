use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use crate::model::FormationKind;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CameraMode {
    Showcase,
    Orbit,
    FreeFly,
}

impl CameraMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Showcase => "Showcase",
            Self::Orbit => "Orbit",
            Self::FreeFly => "Free-fly",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CameraRig {
    pub mode: CameraMode,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub free_speed: f32,
    free_position: Vec3,
    smoothed_eye: Vec3,
    smoothed_target: Vec3,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            mode: CameraMode::Showcase,
            yaw: -0.55,
            pitch: 0.18,
            distance: 42.0,
            target: Vec3::new(0.0, 13.0, 0.0),
            free_speed: 16.0,
            free_position: Vec3::new(22.0, 21.0, 36.0),
            smoothed_eye: Vec3::new(22.0, 21.0, 36.0),
            smoothed_target: Vec3::new(0.0, 13.0, 0.0),
        }
    }
}

impl CameraRig {
    pub fn update(&mut self, dt: f32, time: f32, formation: FormationKind, fleet_count: usize) {
        let (target_y, base_distance) = match formation {
            FormationKind::LaunchGrid => (3.5, 42.0),
            FormationKind::Galaxy => (13.0, 38.0),
            FormationKind::Image => (14.0, 10.0),
            FormationKind::EventHorizon => (14.0, 40.0),
            FormationKind::Mandala | FormationKind::Gyroscope => (14.0, 36.0),
            _ => (14.0, 34.0),
        };
        if self.mode == CameraMode::Showcase {
            let formation_scale = crate::formation::fleet_scale(fleet_count);
            let framing_scale = if formation == FormationKind::LaunchGrid {
                (fleet_count.max(1) as f32 / 384.0).sqrt().clamp(0.8, 60.0)
            } else {
                formation_scale
            };
            let base_pitch = if formation == FormationKind::Galaxy {
                0.24
            } else {
                0.12
            };
            self.yaw = -0.7 + time * 0.035 + (time * 0.09).sin() * 0.25;
            self.pitch = base_pitch + (time * 0.07).sin() * 0.08;
            self.distance = base_distance * framing_scale + (time * 0.11).sin() * 3.0;
            self.target = Vec3::new(
                0.0,
                if formation == FormationKind::LaunchGrid {
                    target_y
                } else {
                    crate::formation::formation_lift(formation, fleet_count)
                },
                0.0,
            );
        }

        if self.mode == CameraMode::FreeFly {
            let desired_eye = self.free_position;
            let desired_target = desired_eye + self.free_forward();
            let response = 1.0 - (-dt * 18.0).exp();
            self.smoothed_eye = self.smoothed_eye.lerp(desired_eye, response);
            self.smoothed_target = self.smoothed_target.lerp(desired_target, response);
            return;
        }

        let planar = self.pitch.cos() * self.distance;
        let desired_eye = self.target
            + Vec3::new(
                self.yaw.sin() * planar,
                self.pitch.sin() * self.distance + 5.0,
                self.yaw.cos() * planar,
            );
        let response = 1.0 - (-dt * 3.5).exp();
        self.smoothed_eye = self.smoothed_eye.lerp(desired_eye, response);
        self.smoothed_target = self.smoothed_target.lerp(self.target, response);
    }

    pub fn set_mode(&mut self, mode: CameraMode) {
        if mode == self.mode {
            return;
        }
        if mode == CameraMode::FreeFly {
            let forward = (self.smoothed_target - self.smoothed_eye).normalize_or_zero();
            self.free_position = self.smoothed_eye;
            self.yaw = forward.x.atan2(forward.z);
            self.pitch = forward.y.asin().clamp(-1.48, 1.48);
            self.free_speed = (self.distance * 0.24).clamp(8.0, 320.0);
        }
        self.mode = mode;
    }

    pub fn free_look(&mut self, delta_x: f32, delta_y: f32) {
        if self.mode != CameraMode::FreeFly {
            return;
        }
        self.yaw -= delta_x * 0.0025;
        self.pitch = (self.pitch - delta_y * 0.0025).clamp(-1.48, 1.48);
    }

    pub fn move_free(&mut self, local: Vec3, dt: f32, sprint: bool) {
        if self.mode != CameraMode::FreeFly || local.length_squared() == 0.0 {
            return;
        }
        let forward = self.free_forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let direction =
            (right * local.x + Vec3::Y * local.y + forward * local.z).normalize_or_zero();
        let speed = self.free_speed * if sprint { 3.0 } else { 1.0 };
        self.free_position += direction * speed * dt;
    }

    pub fn adjust_free_speed(&mut self, scroll_delta: f32) {
        self.free_speed = (self.free_speed * (scroll_delta * 0.0015).exp()).clamp(1.0, 1_500.0);
    }

    fn free_forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
        .normalize_or_zero()
    }

    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.set_mode(CameraMode::Orbit);
        self.yaw -= delta_x * 0.007;
        self.pitch = (self.pitch + delta_y * 0.005).clamp(-0.25, 1.15);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.set_mode(CameraMode::Orbit);
        self.distance = (self.distance * (-delta * 0.001).exp()).clamp(12.0, 5_000.0);
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.smoothed_eye, self.smoothed_target, Vec3::Y);
        let projection =
            Mat4::perspective_rh(47.0_f32.to_radians(), aspect.max(0.1), 0.1, 10_000.0);
        projection * view
    }

    pub fn eye(&self) -> Vec3 {
        self.smoothed_eye
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_fly_moves_and_mouse_look_changes_heading() {
        let mut camera = CameraRig::default();
        camera.set_mode(CameraMode::FreeFly);
        let initial_eye = camera.eye();
        let initial_view = camera.view_projection(16.0 / 9.0);
        let initial_yaw = camera.yaw;
        camera.free_look(120.0, -35.0);
        assert!(
            camera.yaw < initial_yaw,
            "positive X motion must invert yaw"
        );
        camera.move_free(Vec3::new(0.0, 0.0, 1.0), 0.5, false);
        camera.update(0.5, 0.0, FormationKind::Chrysalis, 20_000);
        assert!(camera.eye().distance(initial_eye) > 1.0);
        assert_ne!(camera.view_projection(16.0 / 9.0), initial_view);
    }

    #[test]
    fn free_fly_strafes_in_screen_direction() {
        let mut camera = CameraRig::default();
        camera.set_mode(CameraMode::FreeFly);
        // Looking down -Z, screen-right is world +X.
        camera.yaw = std::f32::consts::PI;
        camera.pitch = 0.0;
        camera.move_free(Vec3::X, 1.0, false);
        camera.update(1.0, 0.0, FormationKind::Chrysalis, 20_000);
        assert!(camera.eye().x > 22.0, "D must move toward camera-right");
    }
}
