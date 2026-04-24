//! 视频理解处理器：封装与云端 VLM API 插件的通信。
//!
//! 与本地 VLM（image_text_understanding）不同，这里调用的是云端 API，
//! 能处理整个视频而不仅仅是关键帧，输出带时间戳的分段描述。
//!
//! 云端 VLM 返回的 JSON 字段名是中文（片段开始、片段结束、人、前景物等），
//! 解析时需要做中文键名的映射。

use crate::error::Result;
use crate::plugins::ProgressMessage;
use crate::plugins::PluginType;
use crate::plugins::image_text_understanding::{StructuredDescription, parse_structured_response};

/// 云端 VLM 返回的视频片段，包含起止时间和结构化描述。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VlmApiSegment {
    pub start_time: f32,
    pub end_time: f32,
    pub description: StructuredDescription,
}

/// 调用云端 VLM API 对视频进行分段描述。
///
/// 云端 API 会自动做时间分段，返回每个片段的结构化描述。
/// 相比本地 VLM 只能处理关键帧，云端能理解更完整的时间上下文。
pub fn describe_video(
    video_path: &str,
    api_base: &str,
    api_key: &str,
    model: &str,
    max_pixels: u32,  // 视频帧的最大分辨率，控制 API 调用成本
    fps: f64,         // 采样帧率，影响时间精度和成本
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<VlmApiSegment>> {
    let data = serde_json::json!({
        "video_path": video_path,
        "api_base": api_base,
        "api_key": api_key,
        "model": model,
        "max_pixels": max_pixels,
        "fps": fps
    });

    let response = crate::plugins::client::call_plugin(PluginType::VideoUnderstanding, "describe_video", &data, progress_cb)?;
    let segments_json = response.result["segments"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError(
            "Invalid VLM API response: missing segments array".into()
        ))?;

    let segments: Vec<VlmApiSegment> = segments_json
        .iter()
        .map(|seg| parse_vlm_api_segment(seg))
        .collect();

    Ok(segments)
}

/// 解析云端 VLM 返回的单个片段 JSON。
///
/// 云端返回的 JSON 使用中文键名（片段开始、片段结束、人、前景物等），
/// 这里做映射并复用 parse_structured_response 来构建 StructuredDescription。
/// "标识"数组中以"字幕-"开头的条目会被拆分到 subtitles 字段。
fn parse_vlm_api_segment(value: &serde_json::Value) -> VlmApiSegment {
    let empty_map = serde_json::Map::new();
    let obj = value.as_object().unwrap_or(&empty_map);

    let start_time = obj.get("片段开始").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let end_time = obj.get("片段结束").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

    // 将"标识"数组拆分为标识标记和字幕：
    // 以"字幕"开头（不管后面跟什么分隔符）的归入字幕，其余归入标识
    let all_marks: Vec<String> = obj.get("标识").and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let (marks, subtitles): (Vec<String>, Vec<String>) = all_marks
        .into_iter()
        .partition(|s| !s.trim().starts_with("字幕"));

    // 去掉字幕条目的"字幕"前缀和分隔符
    let subtitles: Vec<String> = subtitles.into_iter().map(|s| {
        let trimmed = s.trim();
        let text = trimmed.strip_prefix("字幕").unwrap_or(trimmed);
        text.trim_start_matches(|c: char| c == ' ' || c == '-' || c == '‑' || c == '–' || c == '—' || c == '：' || c == ':')
            .to_string()
    }).collect();

    // 构造与本地 VLM 相同格式的 JSON，复用 parse_structured_response 解析逻辑
    let description = parse_structured_response(&serde_json::Value::Object({
        let mut map = serde_json::Map::new();
        if let Some(v) = obj.get("人") { map.insert("人".into(), v.clone()); }
        if let Some(v) = obj.get("前景物") { map.insert("前景物".into(), v.clone()); }
        if let Some(v) = obj.get("背景物") { map.insert("背景物".into(), v.clone()); }
        if let Some(v) = obj.get("场") { map.insert("场".into(), v.clone()); }
        if let Some(v) = obj.get("动作") { map.insert("动作".into(), v.clone()); }
        map.insert("标识".into(), serde_json::Value::Array(marks.into_iter().map(serde_json::Value::String).collect()));
        map.insert("字幕".into(), serde_json::Value::Array(subtitles.into_iter().map(serde_json::Value::String).collect()));
        map
    }));

    VlmApiSegment {
        start_time,
        end_time,
        description,
    }
}
