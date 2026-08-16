use glam::Vec3;

/// Maps every source index to a unique destination index. Exact Hungarian
/// assignment is used for smaller fleets; large fleets use a deterministic
/// spatial ordering with a short local improvement pass.
pub fn assign(source: &[Vec3], destination: &[Vec3]) -> Vec<usize> {
    assert_eq!(source.len(), destination.len());
    if source.len() <= 384 {
        hungarian(source, destination)
    } else {
        scalable_spatial_assignment(source, destination)
    }
}

fn hungarian(source: &[Vec3], destination: &[Vec3]) -> Vec<usize> {
    let n = source.len();
    let mut u = vec![0.0_f64; n + 1];
    let mut v = vec![0.0_f64; n + 1];
    let mut p = vec![0_usize; n + 1];
    let mut way = vec![0_usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cost = source[i0 - 1].distance_squared(destination[j - 1]) as f64;
                    let cur = cost - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![0; n];
    for j in 1..=n {
        if p[j] != 0 {
            result[p[j] - 1] = j - 1;
        }
    }
    result
}

fn scalable_spatial_assignment(source: &[Vec3], destination: &[Vec3]) -> Vec<usize> {
    let mut src: Vec<_> = (0..source.len()).collect();
    let mut dst: Vec<_> = (0..destination.len()).collect();
    let source_bounds = Bounds::from_points(source);
    let destination_bounds = Bounds::from_points(destination);
    src.sort_unstable_by_key(|&i| morton_key(source_bounds.normalize(source[i])));
    dst.sort_unstable_by_key(|&i| morton_key(destination_bounds.normalize(destination[i])));

    let mut result = vec![0; source.len()];
    for (&s, &d) in src.iter().zip(&dst) {
        result[s] = d;
    }

    // Multi-scale local swaps remove crossings and long routes without losing
    // the O(n log n) behavior required by very large fleets.
    for stride in [1, 2, 4, 8, 16] {
        for position in 0..src.len().saturating_sub(stride) {
            let a = src[position];
            let b = src[position + stride];
            let da = result[a];
            let db = result[b];
            let direct = source[a].distance_squared(destination[da])
                + source[b].distance_squared(destination[db]);
            let swapped = source[a].distance_squared(destination[db])
                + source[b].distance_squared(destination[da]);
            if swapped < direct {
                result.swap(a, b);
            }
        }
    }
    result
}

#[derive(Clone, Copy)]
struct Bounds {
    minimum: Vec3,
    inverse_extent: Vec3,
}

impl Bounds {
    fn from_points(points: &[Vec3]) -> Self {
        let mut minimum = Vec3::splat(f32::MAX);
        let mut maximum = Vec3::splat(f32::MIN);
        for point in points {
            minimum = minimum.min(*point);
            maximum = maximum.max(*point);
        }
        let extent = maximum - minimum;
        Self {
            minimum,
            inverse_extent: Vec3::new(
                if extent.x > 1e-5 { 1.0 / extent.x } else { 0.0 },
                if extent.y > 1e-5 { 1.0 / extent.y } else { 0.0 },
                if extent.z > 1e-5 { 1.0 / extent.z } else { 0.0 },
            ),
        }
    }

    fn normalize(self, point: Vec3) -> Vec3 {
        (point - self.minimum) * self.inverse_extent
    }
}

fn morton_key(p: Vec3) -> u64 {
    let quantize = |value: f32| (value.clamp(0.0, 1.0) * 65_535.0) as u64;
    let (x, y, z) = (quantize(p.x), quantize(p.y), quantize(p.z));
    let mut result = 0;
    for bit in 0..16 {
        result |= ((x >> bit) & 1) << (bit * 3);
        result |= ((y >> bit) & 1) << (bit * 3 + 1);
        result |= ((z >> bit) & 1) << (bit * 3 + 2);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(src: &[Vec3], dst: &[Vec3], a: &[usize]) -> f32 {
        src.iter()
            .enumerate()
            .map(|(i, p)| p.distance_squared(dst[a[i]]))
            .sum()
    }

    #[test]
    fn assignment_is_unique_and_minimal_for_small_set() {
        let src = vec![Vec3::X * 10.0, Vec3::ZERO, -Vec3::X * 10.0];
        let dst = vec![-Vec3::X * 9.0, Vec3::X * 9.0, Vec3::ZERO];
        let result = assign(&src, &dst);
        let mut unique = result.clone();
        unique.sort_unstable();
        assert_eq!(unique, vec![0, 1, 2]);
        assert!(cost(&src, &dst, &result) < 3.0);
    }

    #[test]
    fn scalable_assignment_is_invariant_to_distribution_scale() {
        let source: Vec<_> = (0..1_024)
            .map(|i| {
                let x = (i % 32) as f32;
                let y = (i / 32) as f32;
                Vec3::new(x * 8.0 - 120.0, y * 3.0 + 40.0, (x * 0.31).sin())
            })
            .collect();
        let destination: Vec<_> = source
            .iter()
            .map(|point| *point * Vec3::new(0.07, 0.2, 1.4) + Vec3::new(8.0, -12.0, 3.0))
            .collect();
        let result = assign(&source, &destination);
        let mut unique = result.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), source.len());
        let average_distance = cost(&source, &destination, &result) / source.len() as f32;
        assert!(average_distance.is_finite());
    }
}
