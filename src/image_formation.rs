use std::{fs::File, io::BufReader, path::Path};

use glam::Vec3;
use image::{
    codecs::gif::GifDecoder, imageops::FilterType, AnimationDecoder, DynamicImage, GenericImageView,
};
use thiserror::Error;

use crate::{
    formation,
    model::{FormationPoint, Rgbw},
};

const DISPLAY_PITCH_METRES: f32 = 0.42;
const MAX_ANIMATION_FRAMES: usize = 160;

#[derive(Debug, Error)]
pub enum ImageFormationError {
    #[error("could not open image: {0}")]
    Open(#[from] std::io::Error),
    #[error("could not decode image: {0}")]
    Decode(#[from] image::ImageError),
    #[error("the image contains no pixels or animation frames")]
    Empty,
}

#[derive(Clone, Debug)]
pub struct AnimationFrame {
    packed_rgba: Vec<u32>,
    pub duration_seconds: f32,
}

#[derive(Clone, Debug)]
pub struct SampledAnimation {
    pub frames: Vec<AnimationFrame>,
    pub duration_seconds: f32,
}

impl SampledAnimation {
    pub fn frame_at(&self, playback_time: f32) -> usize {
        if self.frames.is_empty() {
            return 0;
        }
        let mut cursor = 0.0;
        let time = playback_time.clamp(0.0, self.duration_seconds.max(0.0));
        for (index, frame) in self.frames.iter().enumerate() {
            cursor += frame.duration_seconds.max(0.001);
            if time < cursor || index == self.frames.len() - 1 {
                return index;
            }
        }
        self.frames.len() - 1
    }

    pub fn apply_frame(&self, base: &[FormationPoint], index: usize) -> Vec<FormationPoint> {
        let Some(frame) = self.frames.get(index) else {
            return base.to_vec();
        };
        base.iter()
            .zip(&frame.packed_rgba)
            .map(|(point, packed)| {
                let mut animated = point.clone();
                animated.color = rgbw_from_packed(*packed);
                animated
            })
            .collect()
    }
}

pub struct SampledMedia {
    pub points: Vec<FormationPoint>,
    pub animation: Option<SampledAnimation>,
}

/// Converts the complete uploaded raster into a deterministic drone display.
///
/// The raster dimensions never exceed the fleet size, so every downsampled
/// pixel receives a drone and no scanline holes are introduced. Any remaining
/// aircraft reinforce evenly distributed pixels on a safely separated depth
/// layer. There is no segmentation, background removal, alpha crop, or random
/// positional jitter.
pub fn sample(
    path: impl AsRef<Path>,
    count: usize,
) -> Result<Vec<FormationPoint>, ImageFormationError> {
    Ok(sample_media(path, count)?.points)
}

pub fn sample_media(
    path: impl AsRef<Path>,
    count: usize,
) -> Result<SampledMedia, ImageFormationError> {
    if count == 0 {
        return Err(ImageFormationError::Empty);
    }
    let path = path.as_ref();
    let is_gif = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gif"));
    if is_gif {
        return sample_gif(path, count);
    }
    let image = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let (points, _) = sample_dynamic_image(&image, count)?;
    Ok(SampledMedia {
        points,
        animation: None,
    })
}

fn sample_gif(path: &Path, count: usize) -> Result<SampledMedia, ImageFormationError> {
    let decoder = GifDecoder::new(BufReader::new(File::open(path)?))?;
    let mut frames = Vec::new();
    let mut points = None;
    for decoded in decoder.into_frames() {
        let decoded = decoded?;
        let delay = decoded.delay();
        let (numerator, denominator) = delay.numer_denom_ms();
        let duration_seconds =
            (numerator as f32 / denominator.max(1) as f32 / 1_000.0).max(1.0 / 60.0);
        let image = DynamicImage::ImageRgba8(decoded.into_buffer());
        let (frame_points, packed_rgba) = sample_dynamic_image(&image, count)?;
        if points.is_none() {
            points = Some(frame_points);
        }
        frames.push(AnimationFrame {
            packed_rgba,
            duration_seconds,
        });
        if frames.len() > MAX_ANIMATION_FRAMES {
            compact_animation_frames(&mut frames);
        }
    }
    let points = points.ok_or(ImageFormationError::Empty)?;
    let duration_seconds = frames.iter().map(|frame| frame.duration_seconds).sum();
    Ok(SampledMedia {
        points,
        animation: Some(SampledAnimation {
            frames,
            duration_seconds,
        }),
    })
}

fn compact_animation_frames(frames: &mut Vec<AnimationFrame>) {
    let mut compacted = Vec::with_capacity(frames.len().div_ceil(2));
    let mut source = std::mem::take(frames).into_iter();
    while let Some(first) = source.next() {
        if let Some(mut second) = source.next() {
            second.duration_seconds += first.duration_seconds;
            compacted.push(second);
        } else {
            compacted.push(first);
        }
    }
    *frames = compacted;
}

fn sample_dynamic_image(
    image: &DynamicImage,
    count: usize,
) -> Result<(Vec<FormationPoint>, Vec<u32>), ImageFormationError> {
    let (source_width, source_height) = image.dimensions();
    if source_width == 0 || source_height == 0 {
        return Err(ImageFormationError::Empty);
    }
    let (width, height) = raster_dimensions(source_width, source_height, count);
    let rgba = image
        .resize_exact(width, height, FilterType::Lanczos3)
        .to_rgba8();
    let raster_len = (width as usize).saturating_mul(height as usize);
    if raster_len == 0 {
        return Err(ImageFormationError::Empty);
    }
    let center_x = width as f32 * 0.5;
    let center_y = height as f32 * 0.5;
    // CPU and GPU formation paths apply the common fleet scale later. Choosing
    // its inverse here yields a final 0.42 m image lattice: visually dense, but
    // still beyond the 0.38 m warning envelope.
    let pitch = DISPLAY_PITCH_METRES / formation::fleet_scale(count);
    let extra_count = count.saturating_sub(raster_len);
    let mut points = Vec::with_capacity(count);
    let mut packed_rgba = Vec::with_capacity(count);
    for index in 0..count {
        let (raster_index, depth_layer) = if index < raster_len {
            (index, 0usize)
        } else {
            let extra = index - raster_len;
            let distributed = (((extra as u64 * 2 + 1) * raster_len as u64)
                / (extra_count.max(1) as u64 * 2))
                .min(raster_len.saturating_sub(1) as u64) as usize;
            (distributed, 1usize)
        };
        let x = raster_index as u32 % width;
        let y = raster_index as u32 / width;
        let pixel = rgba.get_pixel(x, y).0;
        let packed = u32::from_le_bytes(pixel);
        packed_rgba.push(packed);
        points.push(FormationPoint {
            position: Vec3::new(
                (x as f32 + 0.5 - center_x) * pitch,
                (center_y - y as f32 - 0.5) * pitch,
                depth_layer as f32 * pitch,
            ),
            color: rgbw_from_packed(packed),
            // Duplicate depth pixels are a small remainder and use half energy
            // so they reinforce coverage without creating bright scanline bands.
            brightness: if depth_layer == 0 { 0.62 } else { 0.31 },
            phase: index as f32 / count as f32,
            group: ((y * 16 / height.max(1)).min(15)) as u16,
        });
    }
    Ok((points, packed_rgba))
}

fn raster_dimensions(source_width: u32, source_height: u32, count: usize) -> (u32, u32) {
    let aspect = source_width as f64 / source_height as f64;
    let ideal_width = (count as f64 * aspect).sqrt().floor().max(1.0) as u32;
    let mut best = (1u32, 1u32);
    let mut best_score = f64::INFINITY;
    for width in ideal_width.saturating_sub(8).max(1)..=ideal_width.saturating_add(8) {
        let height = (count / width as usize).max(1) as u32;
        let used = width as usize * height as usize;
        if used > count {
            continue;
        }
        let aspect_error = ((width as f64 / height as f64) / aspect).ln().abs();
        let unused = (count - used) as f64 / count.max(1) as f64;
        let score = aspect_error * 8.0 + unused;
        if score < best_score {
            best = (width, height);
            best_score = score;
        }
    }
    best
}

fn rgbw_from_packed(packed: u32) -> Rgbw {
    let [r, g, b, a] = packed.to_le_bytes();
    let alpha = a as f32 / 255.0;
    let srgb = Vec3::new(r as f32, g as f32, b as f32) / 255.0 * alpha;
    let linear = srgb_to_linear(srgb);
    let white = linear.min_element();
    Rgbw::new(
        (linear.x - white).max(0.0),
        (linear.y - white).max(0.0),
        (linear.z - white).max(0.0),
        white,
    )
}

fn srgb_to_linear(srgb: Vec3) -> Vec3 {
    srgb.map(|channel| {
        if channel <= 0.04045 {
            channel / 12.92
        } else {
            ((channel + 0.055) / 1.055).powf(2.4)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_image(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "luma-image-formation-{name}-{}.png",
            std::process::id()
        ))
    }

    #[test]
    fn full_opaque_frame_is_preserved_without_segmentation() {
        let path = temporary_image("full-frame");
        let mut artwork = image::RgbaImage::from_pixel(80, 120, image::Rgba([210, 198, 190, 255]));
        for y in 20..100 {
            for x in 26..54 {
                artwork.put_pixel(x, y, image::Rgba([220, 80, 170, 255]));
            }
        }
        artwork.save(&path).unwrap();
        let points = sample(&path, 5_000).unwrap();
        let _ = std::fs::remove_file(path);

        let minimum = points.iter().fold(Vec3::splat(f32::MAX), |extent, point| {
            extent.min(point.position)
        });
        let maximum = points.iter().fold(Vec3::splat(f32::MIN), |extent, point| {
            extent.max(point.position)
        });
        let aspect = (maximum.x - minimum.x) / (maximum.y - minimum.y);
        assert!((aspect - 80.0 / 120.0).abs() < 0.03);
        assert_eq!(points.len(), 5_000);
        assert!(points.iter().any(|point| {
            let rgb = point.color.rgb();
            rgb.x > rgb.y * 1.8 && rgb.z > rgb.y * 1.2
        }));
        assert!(points.iter().any(|point| {
            let rgb = point.color.rgb();
            (rgb.x - rgb.y).abs() < 0.12 && (rgb.y - rgb.z).abs() < 0.12
        }));
    }

    #[test]
    fn rgbw_split_reconstructs_source_linear_colour() {
        let path = temporary_image("colour");
        image::RgbaImage::from_pixel(1, 1, image::Rgba([170, 92, 215, 255]))
            .save(&path)
            .unwrap();
        let points = sample(&path, 1).unwrap();
        let _ = std::fs::remove_file(path);
        let expected = srgb_to_linear(Vec3::new(170.0, 92.0, 215.0) / 255.0);
        assert!(points[0].color.rgb().distance(expected) < 0.0001);
    }

    #[test]
    fn downsample_grid_has_no_missing_raster_pixels() {
        let path = temporary_image("grid");
        image::RgbaImage::from_pixel(2_000, 3_000, image::Rgba([120, 150, 180, 255]))
            .save(&path)
            .unwrap();
        let count = 50_000;
        let points = sample(&path, count).unwrap();
        let _ = std::fs::remove_file(path);
        let (width, height) = raster_dimensions(2_000, 3_000, count);
        let base_count = width as usize * height as usize;
        assert_eq!(points.len(), count);
        let mut base_positions: Vec<_> = points[..base_count]
            .iter()
            .map(|point| (point.position.x.to_bits(), point.position.y.to_bits()))
            .collect();
        base_positions.sort_unstable();
        base_positions.dedup();
        assert_eq!(base_positions.len(), base_count);
        assert!(count - base_count < width.max(height) as usize);
    }

    #[test]
    fn animation_frame_lookup_reaches_the_last_frame() {
        let animation = SampledAnimation {
            frames: vec![
                AnimationFrame {
                    packed_rgba: vec![0],
                    duration_seconds: 0.1,
                },
                AnimationFrame {
                    packed_rgba: vec![0],
                    duration_seconds: 0.2,
                },
            ],
            duration_seconds: 0.3,
        };
        assert_eq!(animation.frame_at(0.05), 0);
        assert_eq!(animation.frame_at(0.29), 1);
    }

    #[test]
    fn animated_gif_is_sampled_as_a_fixed_lattice_with_changing_colour() {
        use image::codecs::gif::{GifEncoder, Repeat};

        let path = std::env::temp_dir().join(format!(
            "luma-image-formation-animation-{}.gif",
            std::process::id()
        ));
        let file = File::create(&path).unwrap();
        let mut encoder = GifEncoder::new(file);
        encoder.set_repeat(Repeat::Infinite).unwrap();
        for color in [[230, 30, 80, 255], [20, 170, 240, 255]] {
            let frame = image::Frame::from_parts(
                image::RgbaImage::from_pixel(12, 8, image::Rgba(color)),
                0,
                0,
                image::Delay::from_numer_denom_ms(100, 1),
            );
            encoder.encode_frame(frame).unwrap();
        }
        drop(encoder);

        let sampled = sample_media(&path, 500).unwrap();
        let _ = std::fs::remove_file(path);
        let animation = sampled.animation.unwrap();
        assert_eq!(animation.frames.len(), 2);
        assert!((animation.duration_seconds - 0.2).abs() < 0.01);
        let first = animation.apply_frame(&sampled.points, 0);
        let second = animation.apply_frame(&sampled.points, 1);
        assert_eq!(first[0].position, second[0].position);
        assert!(first[0].color.rgb().distance(second[0].color.rgb()) > 0.2);
    }
}
