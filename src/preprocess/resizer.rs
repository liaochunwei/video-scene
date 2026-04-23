//! 图像缩放与持久化工具
//!
//! 提供图像加载、等比缩放（以短边为基准）和 JPEG 保存功能。
//! 缩放使用 Lanczos3 插值，在缩放质量和速度之间取得较好的平衡。

use image::{imageops, DynamicImage, GenericImageView};
use std::path::Path;

use crate::error::{Result, VideoSceneError};

/// 等比缩放图像，使短边等于 target_short_edge。
///
/// 若图像短边已小于等于目标值则不缩放，避免无谓放大导致画质损失。
/// 使用 Lanczos3 滤波器，适合缩小场景，在清晰度和平滑度间表现优秀。
pub fn resize_image(image: &DynamicImage, target_short_edge: u32) -> DynamicImage {
    let (w, h) = image.dimensions();
    let short = w.min(h);
    if short == 0 || short <= target_short_edge {
        return image.clone();
    }
    let scale = target_short_edge as f32 / short as f32;
    let new_w = (w as f32 * scale) as u32;
    let new_h = (h as f32 * scale) as u32;
    image.resize_exact(new_w, new_h, imageops::FilterType::Lanczos3)
}

/// 从磁盘加载图像文件。
pub fn load_image(path: &Path) -> Result<DynamicImage> {
    image::open(path)
        .map_err(|e| VideoSceneError::StorageError(format!("Failed to load image {}: {}", path.display(), e)))
}

/// 将图像保存为 JPEG 格式。
///
/// 会自动创建父目录。使用 BufWriter 包装以减少系统调用次数，提升写入性能。
/// `_quality` 参数目前未使用（image crate 的 JPEG 编码器暂不支持质量配置），
/// 预留接口以便后续替换编码实现。
pub fn save_jpeg(image: &DynamicImage, path: &Path, _quality: u8) -> Result<()> {
    // 确保目标路径的父目录存在
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
    }
    // 使用 BufWriter 减少 I/O 系统调用次数
    let mut buf = std::io::BufWriter::new(
        std::fs::File::create(path)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?,
    );
    image
        .write_to(&mut buf, image::ImageFormat::Jpeg)
        .map_err(|e| VideoSceneError::StorageError(format!("Failed to save JPEG: {}", e)))?;
    Ok(())
}
