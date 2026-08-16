use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionKind {
    DroneToDrone,
    GroundContact,
}

impl CollisionKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::DroneToDrone => "DRONE COLLISION",
            Self::GroundContact => "GROUND CONTACT",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CollisionRecord {
    pub id: u64,
    pub kind: CollisionKind,
    pub count: u32,
    pub run_time: f32,
    pub show_time: f32,
    pub position: Vec3,
    pub measured_clearance: f32,
    pub drone_a: Option<u32>,
    pub drone_b: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct SafetyObservation {
    pub audit_sequence: u64,
    pub run_time: f32,
    pub show_time: f32,
    pub collision_pairs: u32,
    pub ground_breaches: u32,
    pub minimum_air_separation: f32,
    pub minimum_ground_clearance: f32,
    pub collision_pair: Option<(u32, u32)>,
    pub collision_position: Option<Vec3>,
    pub ground_drone: Option<u32>,
    pub ground_position: Option<Vec3>,
}

#[derive(Default)]
pub struct RunCollisionLog {
    records: Vec<CollisionRecord>,
    cumulative_drone_collisions: u64,
    cumulative_ground_contacts: u64,
    last_audit_sequence: Option<u64>,
    previous_collision_pairs: u32,
    previous_ground_breaches: u32,
    previous_collision_pair: Option<(u32, u32)>,
    previous_ground_drone: Option<u32>,
    next_id: u64,
    dropped_records: u64,
}

impl RunCollisionLog {
    const MAX_RECORDS: usize = 100_000;

    pub fn records(&self) -> &[CollisionRecord] {
        &self.records
    }

    pub fn cumulative_drone_collisions(&self) -> u64 {
        self.cumulative_drone_collisions
    }

    pub fn cumulative_ground_contacts(&self) -> u64 {
        self.cumulative_ground_contacts
    }

    pub fn dropped_records(&self) -> u64 {
        self.dropped_records
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn observe(&mut self, observation: SafetyObservation) {
        if self.last_audit_sequence == Some(observation.audit_sequence) {
            return;
        }
        if self
            .last_audit_sequence
            .is_some_and(|last| observation.audit_sequence < last)
        {
            // A timeline seek restarts the underlying audit sequence but not
            // the run log. End any active episode so a reproduced event can be
            // recorded as a new incident after playback resumes.
            self.previous_collision_pairs = 0;
            self.previous_ground_breaches = 0;
            self.previous_collision_pair = None;
            self.previous_ground_drone = None;
        }
        self.last_audit_sequence = Some(observation.audit_sequence);

        let new_collisions = rising_incident_count(
            self.previous_collision_pairs,
            observation.collision_pairs,
            self.previous_collision_pair,
            observation.collision_pair,
        );
        if new_collisions > 0 {
            self.cumulative_drone_collisions += u64::from(new_collisions);
            self.push(CollisionRecord {
                id: self.next_id,
                kind: CollisionKind::DroneToDrone,
                count: new_collisions,
                run_time: observation.run_time,
                show_time: observation.show_time,
                position: observation.collision_position.unwrap_or(Vec3::ZERO),
                measured_clearance: observation.minimum_air_separation,
                drone_a: observation.collision_pair.map(|pair| pair.0),
                drone_b: observation.collision_pair.map(|pair| pair.1),
            });
        }

        let new_ground_contacts = rising_incident_count(
            self.previous_ground_breaches,
            observation.ground_breaches,
            self.previous_ground_drone,
            observation.ground_drone,
        );
        if new_ground_contacts > 0 {
            self.cumulative_ground_contacts += u64::from(new_ground_contacts);
            self.push(CollisionRecord {
                id: self.next_id,
                kind: CollisionKind::GroundContact,
                count: new_ground_contacts,
                run_time: observation.run_time,
                show_time: observation.show_time,
                position: observation.ground_position.unwrap_or(Vec3::ZERO),
                measured_clearance: observation.minimum_ground_clearance,
                drone_a: observation.ground_drone,
                drone_b: None,
            });
        }

        self.previous_collision_pairs = observation.collision_pairs;
        self.previous_ground_breaches = observation.ground_breaches;
        self.previous_collision_pair = observation.collision_pair;
        self.previous_ground_drone = observation.ground_drone;
    }

    fn push(&mut self, mut record: CollisionRecord) {
        self.next_id += 1;
        record.id = self.next_id;
        if self.records.len() == Self::MAX_RECORDS {
            self.records.remove(0);
            self.dropped_records += 1;
        }
        self.records.push(record);
    }
}

fn rising_incident_count<T: Copy + Eq>(
    previous_count: u32,
    current_count: u32,
    previous_representative: Option<T>,
    current_representative: Option<T>,
) -> u32 {
    if current_count == 0 {
        0
    } else if previous_count == 0 {
        current_count
    } else if current_count > previous_count {
        current_count - previous_count
    } else if current_representative.is_some() && current_representative != previous_representative
    {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(sequence: u64, collisions: u32) -> SafetyObservation {
        SafetyObservation {
            audit_sequence: sequence,
            run_time: sequence as f32,
            show_time: sequence as f32 * 0.5,
            collision_pairs: collisions,
            ground_breaches: 0,
            minimum_air_separation: 0.12,
            minimum_ground_clearance: 3.0,
            collision_pair: (collisions > 0).then_some((4, 9)),
            collision_position: Some(Vec3::new(1.0, 2.0, 3.0)),
            ground_drone: None,
            ground_position: None,
        }
    }

    #[test]
    fn persistent_pair_is_counted_once_but_a_new_episode_is_retained() {
        let mut log = RunCollisionLog::default();
        log.observe(observation(1, 1));
        log.observe(observation(2, 1));
        log.observe(observation(3, 0));
        log.observe(observation(4, 1));
        assert_eq!(log.cumulative_drone_collisions(), 2);
        assert_eq!(log.records().len(), 2);
        assert_eq!(log.records()[1].show_time, 2.0);
    }

    #[test]
    fn duplicate_async_readback_is_not_logged_twice() {
        let mut log = RunCollisionLog::default();
        log.observe(observation(7, 2));
        log.observe(observation(7, 2));
        assert_eq!(log.cumulative_drone_collisions(), 2);
        assert_eq!(log.records().len(), 1);
    }
}
