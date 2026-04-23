//! 视频元信息分析
//!
//! 通过调用 ffprobe 获取视频的时长、分辨率和帧率等基本信息，
//! 为后续的帧提取和场景分割提供必要的参数依据。

use std::path::Path;

use crate::error::{Result, VideoSceneError};

/// 视频基本元信息
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub duration: f32,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
}

/// 分析视频文件，返回时长、分辨率和帧率。
///
/// 底层调用 ffprobe 并以 JSON 格式解析输出，比手工解析文本更可靠。
/// 如果 ffprobe 不可用或返回异常，会返回对应的错误。
pub fn analyze_video(path: &Path) -> Result<VideoInfo> {
    if !path.exists() {
        return Err(VideoSceneError::VideoNotFound(path.to_path_buf()));
    }

    let output = std::process::Command::new("ffprobe")
        .args([
            "-v", "quiet",           // 静默模式，不输出警告和错误到 stderr
            "-print_format", "json", // 以 JSON 格式输出，方便结构化解析
            "-show_format",          // 显示容器级信息（时长等）
            "-show_streams",         // 显示流级信息（分辨率、帧率等）
        ])
        .arg(path)
        .output()
        .map_err(|e| VideoSceneError::VideoDecodeError(e.to_string()))?;

    if !output.status.success() {
        return Err(VideoSceneError::VideoDecodeError(
            "ffprobe failed".into(),
        ));
    }

    let json_str = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| VideoSceneError::VideoDecodeError(e.to_string()))?;

    // 时长在 format 节点中，单位为秒
    let duration = json["format"]["duration"]
        .as_str()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);

    // 在所有流中找到第一个视频流（排除音频、字幕等流）
    let video_stream = json["streams"]
        .as_array()
        .and_then(|streams| {
            streams.iter().find(|s| s["codec_type"].as_str() == Some("video"))
        });

    let (width, height, fps) = if let Some(stream) = video_stream {
        let w = stream["width"].as_u64().unwrap_or(0) as u32;
        let h = stream["height"].as_u64().unwrap_or(0) as u32;
        // r_frame_rate 通常是 "num/den" 分数格式（如 "24000/1001"），
        // 需要手动解析做除法；若不是分数格式则尝试直接解析为浮点数
        let f = stream["r_frame_rate"]
            .as_str()
            .and_then(|s| {
                let parts: Vec<&str> = s.split('/').collect();
                if parts.len() == 2 {
                    let num: f32 = parts[0].parse().ok()?;
                    let den: f32 = parts[1].parse().ok()?;
                    if den > 0.0 { Some(num / den) } else { None }
                } else {
                    s.parse().ok()
                }
            })
            .unwrap_or(30.0); // 无法解析帧率时默认 30fps
        (w, h, f)
    } else {
        (0, 0, 30.0)
    };

    Ok(VideoInfo {
        duration,
        width,
        height,
        fps,
    })
}
