use serde::{Deserialize, Serialize};

use crate::model::FormationKind;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CueKind {
    Launch { target: FormationKind },
    Hold { formation: FormationKind },
    Transition { target: FormationKind },
    FormationAnimation { formation: FormationKind },
    ImageAnimation { formation: FormationKind },
    ColorWave { formation: FormationKind },
    CameraCue { formation: FormationKind },
    Landing,
}

impl CueKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Launch { .. } => "LAUNCH",
            Self::Hold { .. } => "HOLD",
            Self::Transition { .. } => "MORPH",
            Self::FormationAnimation { .. } => "ANIMATE",
            Self::ImageAnimation { .. } => "PLAY GIF",
            Self::ColorWave { .. } => "COLOUR",
            Self::CameraCue { .. } => "CAMERA",
            Self::Landing => "LAND",
        }
    }

    pub fn formation(&self) -> Option<FormationKind> {
        match *self {
            Self::Launch { target }
            | Self::Transition { target }
            | Self::Hold { formation: target }
            | Self::FormationAnimation { formation: target }
            | Self::ImageAnimation { formation: target }
            | Self::ColorWave { formation: target }
            | Self::CameraCue { formation: target } => Some(target),
            Self::Landing => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TimelineCue {
    pub name: String,
    pub duration: f32,
    pub kind: CueKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShowTimeline {
    pub cues: Vec<TimelineCue>,
}

#[derive(Clone, Copy, Debug)]
pub struct CueSample<'a> {
    pub index: usize,
    pub cue: &'a TimelineCue,
    pub local_time: f32,
    pub progress: f32,
    pub previous_formation: FormationKind,
}

impl ShowTimeline {
    pub fn imported_image_show(hold_duration: f32, animated: bool) -> Self {
        Self {
            cues: vec![
                cue(
                    "Image launch",
                    10.0,
                    CueKind::Launch {
                        target: FormationKind::Image,
                    },
                ),
                cue(
                    if animated {
                        "GIF playback"
                    } else {
                        "Image hold"
                    },
                    hold_duration.clamp(1.0, 300.0),
                    if animated {
                        CueKind::ImageAnimation {
                            formation: FormationKind::Image,
                        }
                    } else {
                        CueKind::Hold {
                            formation: FormationKind::Image,
                        }
                    },
                ),
                cue("Coordinated landing", 11.0, CueKind::Landing),
            ],
        }
    }

    pub fn showcase() -> Self {
        use FormationKind::*;
        let mut cues = vec![cue(
            "Coordinated launch",
            9.0,
            CueKind::Launch { target: Chrysalis },
        )];
        for (index, formation) in FormationKind::CORE_SHOWCASE.into_iter().enumerate() {
            if index > 0 {
                cues.push(cue(
                    &format!("Morph to {}", formation.label()),
                    6.0,
                    CueKind::Transition { target: formation },
                ));
            }
            cues.push(cue(
                formation.label(),
                if matches!(formation, Chrysalis | Cathedral | Infinity | Lotus | Crown) {
                    7.0
                } else {
                    5.0
                },
                CueKind::FormationAnimation { formation },
            ));
        }
        for formation in FormationKind::BONUS {
            cues.push(cue(
                &format!("Bonus reveal · {}", formation.label()),
                7.0,
                CueKind::Transition { target: formation },
            ));
            cues.push(cue(
                &format!("BONUS · {}", formation.label()),
                9.0,
                CueKind::FormationAnimation { formation },
            ));
        }
        cues.push(cue("Final descent", 10.0, CueKind::Landing));
        Self { cues }
    }

    pub fn duration(&self) -> f32 {
        self.cues.iter().map(|cue| cue.duration.max(0.01)).sum()
    }

    pub fn sample(&self, time: f32) -> CueSample<'_> {
        assert!(
            !self.cues.is_empty(),
            "timeline must contain at least one cue"
        );
        let mut cursor = 0.0;
        let wrapped = time.clamp(0.0, self.duration());
        let mut previous = FormationKind::LaunchGrid;
        for (index, cue) in self.cues.iter().enumerate() {
            let end = cursor + cue.duration.max(0.01);
            if wrapped < end || index == self.cues.len() - 1 {
                let local = (wrapped - cursor).max(0.0);
                return CueSample {
                    index,
                    cue,
                    local_time: local,
                    progress: (local / cue.duration.max(0.01)).clamp(0.0, 1.0),
                    previous_formation: previous,
                };
            }
            if let Some(formation) = cue.kind.formation() {
                previous = formation;
            }
            cursor = end;
        }
        unreachable!()
    }

    pub fn cue_start(&self, index: usize) -> f32 {
        self.cues.iter().take(index).map(|cue| cue.duration).sum()
    }

    pub fn move_cue(&mut self, from: usize, offset: isize) {
        if self.cues.is_empty() {
            return;
        }
        let to = (from as isize + offset).clamp(0, self.cues.len() as isize - 1) as usize;
        if from < self.cues.len() && from != to {
            let cue = self.cues.remove(from);
            self.cues.insert(to, cue);
        }
    }

    pub fn duplicate_cue(&mut self, index: usize) {
        if let Some(cue) = self.cues.get(index).cloned() {
            self.cues.insert(index + 1, cue);
        }
    }
}

fn cue(name: &str, duration: f32, kind: CueKind) -> TimelineCue {
    TimelineCue {
        name: name.to_owned(),
        duration,
        kind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn showcase_contains_every_builtin_formation() {
        let timeline = ShowTimeline::showcase();
        for formation in FormationKind::SHOWCASE {
            assert!(
                timeline
                    .cues
                    .iter()
                    .any(|cue| cue.kind.formation() == Some(formation)),
                "missing {}",
                formation.label()
            );
        }
        assert!(!timeline
            .cues
            .iter()
            .any(|cue| cue.kind.formation() == Some(FormationKind::Image)));
    }

    #[test]
    fn animated_image_show_places_gif_playback_between_launch_and_landing() {
        let timeline = ShowTimeline::imported_image_show(4.0, true);
        assert!(matches!(timeline.cues[0].kind, CueKind::Launch { .. }));
        assert!(matches!(
            timeline.cues[1].kind,
            CueKind::ImageAnimation { .. }
        ));
        assert!(matches!(timeline.cues[2].kind, CueKind::Landing));
    }
}
