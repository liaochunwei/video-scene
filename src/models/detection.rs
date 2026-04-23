//! AI 检测结果模型
//!
//! 本模块定义了 AI 对视频片段进行目标检测后的输出结构。
//! 目前支持两类检测：人脸（Face）和通用物体（Object）。
//! 人脸检测会额外保存特征向量，用于后续与脸库比对识别身份；
//! 物体检测则只记录标签和置信度，无需特征向量。

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 检测类型枚举，区分人脸和通用物体两类检测任务。
/// 后续如需扩展（如文字、声音），可在此添加变体。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionType {
    Face,
    Object,
}

/// 目标在画面中的矩形边界框。
/// 坐标系原点为图像左上角，`(x, y)` 为框的左上角，`width`/`height` 为框的尺寸。
/// 使用 `f32` 是因为检测结果通常以归一化坐标（0~1）表示，方便不同分辨率下复用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// 一次 AI 检测的完整结果。
///
/// - `segment_id` 关联到产生此检测的片段；
/// - `confidence` 范围 [0, 1]，越高表示模型越确信；
/// - `bounding_box` 对整片段检测时可能为 `None`（无明确空间位置）；
/// - `feature_vector` 仅人脸检测使用，物体检测为空向量。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub id: Uuid,
    pub segment_id: Uuid,
    pub detection_type: DetectionType,
    /// 检测到的目标标签（如人名、物体类别）
    pub label: String,
    pub confidence: f32,
    pub bounding_box: Option<BoundingBox>,
    pub feature_vector: Vec<f32>,
}

impl Detection {
    /// 构造人脸检测结果。人脸检测必须携带特征向量，以便后续与脸库匹配识别身份。
    pub fn new_face(
        segment_id: Uuid,
        label: String,
        confidence: f32,
        bounding_box: Option<BoundingBox>,
        feature_vector: Vec<f32>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            segment_id,
            detection_type: DetectionType::Face,
            label,
            confidence,
            bounding_box,
            feature_vector,
        }
    }

    /// 构造物体检测结果。物体检测不需要特征向量，因此置为空。
    pub fn new_object(
        segment_id: Uuid,
        label: String,
        confidence: f32,
        bounding_box: Option<BoundingBox>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            segment_id,
            detection_type: DetectionType::Object,
            label,
            confidence,
            bounding_box,
            feature_vector: Vec::new(),
        }
    }
}
