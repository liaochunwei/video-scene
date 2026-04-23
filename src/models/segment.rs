//! 视频片段（Segment）模型
//!
//! 将一段视频按场景切分为多个片段，每个片段是后续 AI 检测和检索的基本单位。
//! 切分依据通常是场景边界（画面突变），而非固定时长，以保证语义完整性。
//! 每个片段持有一个关键帧截图路径和一段场景向量，用于相似场景检索。

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 视频片段，表示从 `start_time` 到 `end_time` 的一段连续画面。
///
/// - `scene_vector` 是由嵌入模型提取的场景特征向量，为空表示尚未生成；
/// - `scene_description` 是对片段内容的自然语言描述，由 AI 生成。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: Uuid,
    /// 所属视频的外键，关联 `Video::id`
    pub video_id: Uuid,
    pub start_time: f32,
    pub end_time: f32,
    /// 关键帧截图的文件路径，用于缩略图展示和快速预览
    pub keyframe_path: String,
    pub scene_vector: Vec<f32>,
    pub scene_description: String,
}

impl Segment {
    /// 创建片段时 `scene_vector` 和 `scene_description` 暂置为空，
    /// 等后续 AI 管道处理完成后再回填。
    pub fn new(video_id: Uuid, start_time: f32, end_time: f32, keyframe_path: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            video_id,
            start_time,
            end_time,
            keyframe_path,
            scene_vector: Vec::new(),
            scene_description: String::new(),
        }
    }

    /// 计算片段时长（秒）
    pub fn duration(&self) -> f32 {
        self.end_time - self.start_time
    }

    /// 将时间范围格式化为 `MM:SS.ss-MM:SS.ss` 形式，便于日志和 UI 展示。
    pub fn format_time_range(&self) -> String {
        format!(
            "{}-{}",
            format_timestamp(self.start_time),
            format_timestamp(self.end_time)
        )
    }
}

/// 将秒数转为 `MM:SS.ss` 格式，保留两位小数以精确到帧级别。
fn format_timestamp(seconds: f32) -> String {
    let mins = (seconds / 60.0) as u32;
    let secs = seconds % 60.0;
    format!("{:02}:{:05.2}", mins, secs)
}
