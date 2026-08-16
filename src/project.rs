use serde::{Deserialize, Serialize};

use crate::{
    model::{EnvironmentSettings, FormationPoint, GraphicsSettings, SimulationSettings},
    timeline::ShowTimeline,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShowProject {
    pub format_version: u32,
    pub name: String,
    pub author: String,
    pub drone_count: usize,
    pub simulation: SimulationSettings,
    pub graphics: GraphicsSettings,
    pub environment: EnvironmentSettings,
    pub timeline: ShowTimeline,
    #[serde(default)]
    pub imported_image: Option<ImportedImageFormation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImportedImageFormation {
    pub name: String,
    pub source_path: String,
    pub points: Vec<FormationPoint>,
    pub hold_duration: f32,
    #[serde(default)]
    pub animated: bool,
}

impl Default for ShowProject {
    fn default() -> Self {
        Self {
            format_version: 1,
            name: "Midnight Odyssey".to_owned(),
            author: "Luma Swarm Studio".to_owned(),
            drone_count: 20_000,
            simulation: SimulationSettings::default(),
            graphics: GraphicsSettings::default(),
            environment: EnvironmentSettings::default(),
            timeline: ShowTimeline::showcase(),
            imported_image: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_serialization_round_trip() {
        let source = ShowProject::default();
        assert_eq!(source.drone_count, 20_000);
        let serialized = ron::ser::to_string(&source).unwrap();
        let restored: ShowProject = ron::from_str(&serialized).unwrap();
        assert_eq!(source, restored);
    }
}
