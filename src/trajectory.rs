use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Trajectory {
    pub start: Vec3,
    pub end: Vec3,
    pub clearance: f32,
    pub lateral_lane: f32,
}

impl Trajectory {
    pub fn new(start: Vec3, end: Vec3, drone_id: u32) -> Self {
        let distance = start.distance(end);
        let clearance = (distance * 0.14).clamp(0.8, 4.5);
        let lateral_lane = ((drone_id.wrapping_mul(2_654_435_761) >> 24) as f32 / 255.0 - 0.5)
            * (distance * 0.04).min(0.9);
        Self {
            start,
            end,
            clearance,
            lateral_lane,
        }
    }

    pub fn sample(&self, progress: f32) -> Vec3 {
        let t = quintic_smoothstep(progress.clamp(0.0, 1.0));
        let direction = self.end - self.start;
        let side = direction.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        self.start.lerp(self.end, t)
            + Vec3::Y * (std::f32::consts::PI * t).sin() * self.clearance
            + side * (std::f32::consts::TAU * t).sin() * self.lateral_lane
    }
}

pub fn quintic_smoothstep(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_hits_endpoints_and_has_clearance() {
        let trajectory = Trajectory::new(Vec3::ZERO, Vec3::X * 10.0, 42);
        assert!(trajectory.sample(0.0).abs_diff_eq(Vec3::ZERO, 1e-5));
        assert!(trajectory.sample(1.0).abs_diff_eq(Vec3::X * 10.0, 1e-4));
        assert!(trajectory.sample(0.5).y > 1.0);
    }

    #[test]
    fn interpolation_has_zero_endpoint_velocity() {
        let epsilon = 0.0001;
        assert!(quintic_smoothstep(epsilon) < epsilon * 0.01);
        assert!(1.0 - quintic_smoothstep(1.0 - epsilon) < epsilon * 0.01);
    }
}
