#[derive(Clone, Debug)]
pub struct FrameProfiler {
    smoothed_frame_ms: f32,
    worst_frame_ms: f32,
    sample_time: f32,
}

impl Default for FrameProfiler {
    fn default() -> Self {
        Self {
            smoothed_frame_ms: 16.67,
            worst_frame_ms: 16.67,
            sample_time: 0.0,
        }
    }
}

impl FrameProfiler {
    pub fn record(&mut self, frame_seconds: f32) {
        let frame_ms = frame_seconds.max(0.0001) * 1000.0;
        self.smoothed_frame_ms += (frame_ms - self.smoothed_frame_ms) * 0.08;
        self.worst_frame_ms = self.worst_frame_ms.max(frame_ms);
        self.sample_time += frame_seconds;
        if self.sample_time > 1.0 {
            self.sample_time = 0.0;
            self.worst_frame_ms = self.smoothed_frame_ms;
        }
    }

    pub fn fps(&self) -> f32 {
        1000.0 / self.smoothed_frame_ms.max(0.01)
    }

    pub fn worst_frame_ms(&self) -> f32 {
        self.worst_frame_ms
    }
}
