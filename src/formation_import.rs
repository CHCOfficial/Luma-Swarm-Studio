use glam::Vec3;
use thiserror::Error;

use crate::model::{FormationPoint, Rgbw};

#[derive(Debug, Error)]
pub enum FormationImportError {
    #[error("line {line}: invalid vertex coordinate")]
    InvalidVertex { line: usize },
    #[error("line {line}: invalid face index")]
    InvalidFace { line: usize },
    #[error("the OBJ contains no vertices")]
    Empty,
}

/// Samples an OBJ surface into the normalized point-cloud representation used
/// by the choreography system. Polygon faces are fan-triangulated; vertex-only
/// files are treated as point clouds.
pub fn sample_obj(
    source: &str,
    count: usize,
    color: Rgbw,
) -> Result<Vec<FormationPoint>, FormationImportError> {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for (line_index, line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let mut words = line.split_whitespace();
        match words.next() {
            Some("v") => {
                let coordinates: Vec<_> = words.take(3).collect();
                if coordinates.len() != 3 {
                    return Err(FormationImportError::InvalidVertex { line: line_number });
                }
                let parse = |text: &str| text.parse::<f32>().ok();
                let (Some(x), Some(y), Some(z)) = (
                    parse(coordinates[0]),
                    parse(coordinates[1]),
                    parse(coordinates[2]),
                ) else {
                    return Err(FormationImportError::InvalidVertex { line: line_number });
                };
                vertices.push(Vec3::new(x, y, z));
            }
            Some("f") => {
                let indices: Result<Vec<_>, _> = words
                    .map(|word| {
                        word.split('/')
                            .next()
                            .and_then(|index| index.parse::<usize>().ok())
                            .and_then(|index| index.checked_sub(1))
                            .ok_or(FormationImportError::InvalidFace { line: line_number })
                    })
                    .collect();
                let indices = indices?;
                if indices.len() < 3 || indices.iter().any(|&index| index >= vertices.len()) {
                    return Err(FormationImportError::InvalidFace { line: line_number });
                }
                for i in 1..indices.len() - 1 {
                    triangles.push([indices[0], indices[i], indices[i + 1]]);
                }
            }
            _ => {}
        }
    }
    if vertices.is_empty() {
        return Err(FormationImportError::Empty);
    }

    normalize(&mut vertices);
    let mut points = Vec::with_capacity(count);
    for i in 0..count {
        let position = if triangles.is_empty() {
            vertices[i % vertices.len()]
        } else {
            let triangle = triangles[i % triangles.len()];
            let a = vertices[triangle[0]];
            let b = vertices[triangle[1]];
            let c = vertices[triangle[2]];
            let r1 = hash01(i as u32 * 2 + 1).sqrt();
            let r2 = hash01(i as u32 * 2 + 2);
            a * (1.0 - r1) + b * (r1 * (1.0 - r2)) + c * (r1 * r2)
        };
        points.push(FormationPoint {
            position,
            color,
            brightness: 1.0,
            phase: i as f32 / count.max(1) as f32,
            group: (i % triangles.len().max(1)) as u16,
        });
    }
    Ok(points)
}

fn normalize(vertices: &mut [Vec3]) {
    let center = vertices.iter().copied().sum::<Vec3>() / vertices.len() as f32;
    let radius = vertices
        .iter()
        .map(|vertex| vertex.distance(center))
        .fold(0.0_f32, f32::max)
        .max(0.0001);
    for vertex in vertices {
        *vertex = (*vertex - center) * (10.0 / radius);
    }
}

fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_and_normalizes_obj_surface() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let cloud = sample_obj(obj, 128, Rgbw::CYAN).unwrap();
        assert_eq!(cloud.len(), 128);
        assert!(cloud.iter().all(|point| point.position.is_finite()));
        assert!(
            cloud
                .iter()
                .map(|point| point.position.length())
                .fold(0.0_f32, f32::max)
                <= 10.01
        );
    }
}
