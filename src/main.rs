use anyhow::Context;
use eframe::egui;
use luma_swarm_studio::app::DroneShowApp;

fn main() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Luma Swarm Studio")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([1100.0, 700.0])
            .with_app_id("studio.lumaswarm.desktop"),
        renderer: eframe::Renderer::Wgpu,
        multisampling: 1,
        depth_buffer: 0,
        ..Default::default()
    };

    eframe::run_native(
        "Luma Swarm Studio",
        options,
        Box::new(|cc| Ok(Box::new(DroneShowApp::new(cc)?))),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
    .context("failed to start Luma Swarm Studio")
}
