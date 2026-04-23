//! 视频场景分割处理器：封装与镜头边界检测插件的通信。
//!
//! 场景分割通过分析视频帧间的视觉差异，检测镜头切换点（场景边界）。
//! 检测算法和阈值可配置，支持 content/threshold 等不同策略。

use crate::error::Result;
use crate::plugins::ProgressMessage;
use crate::plugins::PluginType;
use crate::preprocess::scene_detector::SceneBoundary;

/// 检测视频中的场景边界（镜头切换点）。
///
/// - detector: 检测算法名称，如 "content"（内容差异）、"threshold"（阈值）
/// - threshold: 灵敏度阈值，越高则只检测越明显的切换
pub fn detect_scenes(
    video_path: &str,
    detector: &str,
    threshold: f64,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<SceneBoundary>> {
    let data = serde_json::json!({
        "video_path": video_path,
        "detector": detector,
        "threshold": threshold
    });

    let response = crate::plugins::client::call_plugin(PluginType::VideoSegmentation, "detect_scenes", &data, progress_cb)?;
    let scenes = response.result["scenes"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError("Invalid scene_detect response".into()))?;

    let boundaries: Vec<SceneBoundary> = scenes
        .iter()
        .map(|s| SceneBoundary {
            start: s["start"].as_f64().unwrap_or(0.0) as f32,
            end: s["end"].as_f64().unwrap_or(0.0) as f32,
        })
        .collect();

    Ok(boundaries)
}
