//! 插件类型枚举：定义系统支持的所有插件类别。
//!
//! 每种类型对应一种 AI 能力（如人脸检测、目标检测、VLM 描述等），
//! 用于插件的注册、路由和配置解析。

use serde::{Deserialize, Serialize};

/// 插件类型枚举，每种变体对应一类 AI 分析能力。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginType {
    Face,                    // 人脸检测与识别（InsightFace）
    Object,                  // 目标检测（YOLO）
    VideoUnderstanding,      // 视频理解（云端 VLM API）
    VideoSegmentation,       // 视频场景分割（镜头边界检测）
    ImageTextUnderstanding,  // 图像文本理解（本地 VLM 场景描述）
    TextVectorization,       // 文本向量化（BGE/文本嵌入）
    ImageTextVectorization,  // 图像文本向量化（CLIP 图像编码）
}

impl PluginType {
    /// 返回所有插件类型的完整列表，用于遍历注册或状态展示。
    pub fn all() -> &'static [PluginType] {
        &[
            PluginType::Face,
            PluginType::Object,
            PluginType::VideoUnderstanding,
            PluginType::VideoSegmentation,
            PluginType::ImageTextUnderstanding,
            PluginType::TextVectorization,
            PluginType::ImageTextVectorization,
        ]
    }

    /// 返回类型的字符串标识，用于 IPC 通信和配置文件中的序列化。
    pub fn as_str(&self) -> &'static str {
        match self {
            PluginType::Face => "face",
            PluginType::Object => "object",
            PluginType::VideoUnderstanding => "video_understanding",
            PluginType::VideoSegmentation => "video_segmentation",
            PluginType::ImageTextUnderstanding => "image_text_understanding",
            PluginType::TextVectorization => "text_vectorization",
            PluginType::ImageTextVectorization => "image_text_vectorization",
        }
    }
}

impl std::fmt::Display for PluginType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
