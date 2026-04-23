//! 视频帧提取
//!
//! 通过调用 ffmpeg 在指定时间点提取视频帧，并按目标短边进行缩放。
//! 缩放可以显著减少关键帧图片的存储体积和后续处理开销。

use std::path::{Path, PathBuf};

use crate::error::{Result, VideoSceneError};

/// 提取出的单帧信息
#[derive(Debug, Clone)]
pub struct ExtractedFrame {
    pub path: PathBuf,
    pub timestamp: f32,
    pub segment_index: usize,
}

/// 在指定时间点逐帧提取视频画面。
///
/// 每个时间戳对应一帧，输出为 JPEG 格式。提取时会根据 `target_short_edge`
/// 对画面进行等比缩放，确保短边为指定像素数，从而控制图片大小。
/// `quality` 参数控制 JPEG 压缩质量（ffmpeg 的 -q:v 参数，2-31，越小质量越高）。
pub fn extract_frames(
    video_path: &Path,
    timestamps: &[f32],
    output_dir: &Path,
    target_short_edge: u32,
    quality: u8,
) -> Result<Vec<ExtractedFrame>> {
    // 确保输出目录存在
    std::fs::create_dir_all(output_dir)
        .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;

    let mut frames = Vec::new();

    for (i, &timestamp) in timestamps.iter().enumerate() {
        let output_path = output_dir.join(format!("frame_{:04}.jpg", i));
        let (w, h) = calculate_scaled_dimensions(video_path, target_short_edge)?;

        // 使用 -ss 放在 -i 前面实现快速 seek（输入级定位），
        // 比 -i 后面的 -ss 更快但精度稍低，对关键帧提取来说足够
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-y",       // 覆盖已有文件，避免因重复运行而报错
                "-ss", &timestamp.to_string(),
                "-i", &video_path.to_string_lossy(),
                "-vframes", "1", // 只提取一帧
                "-vf", &format!("scale={}:{}:force_original_aspect_ratio=decrease", w, h),
                "-q:v", &quality.to_string(),
                &output_path.to_string_lossy(),
            ])
            .output()
            .map_err(|e| VideoSceneError::VideoDecodeError(e.to_string()))?;

        // 同时检查进程退出码和文件是否存在，防止 ffmpeg 静默失败
        if output.status.success() && output_path.exists() {
            frames.push(ExtractedFrame {
                path: output_path,
                timestamp,
                segment_index: i,
            });
        }
    }

    Ok(frames)
}

/// 根据目标短边计算缩放后的视频尺寸。
///
/// 保持宽高比不变，让较短边等于 target_short_edge。
/// 最后将宽高对齐到偶数，因为 ffmpeg 的部分编码器要求偶数尺寸。
fn calculate_scaled_dimensions(video_path: &Path, target_short_edge: u32) -> Result<(u32, u32)> {
    let info = super::video_analyzer::analyze_video(video_path)?;
    let short = info.width.min(info.height);
    if short == 0 {
        return Ok((target_short_edge, target_short_edge));
    }
    let scale = target_short_edge as f32 / short as f32;
    let w = (info.width as f32 * scale) as u32;
    let h = (info.height as f32 * scale) as u32;
    // 位运算 &!1 将最低位清零，确保宽高为偶数，满足 ffmpeg 编码要求
    Ok((w & !1, h & !1))
}
