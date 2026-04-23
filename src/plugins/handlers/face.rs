//! 人脸检测处理器：封装与 InsightFace 插件的通信。
//!
//! 提供单张/批量检测和人脸编码功能。
//! 人脸编码返回的特征向量可用于人脸比对和聚类。

use crate::error::Result;
use crate::models::BoundingBox;
use crate::plugins::ProgressMessage;
use crate::plugins::PluginType;

/// 单张人脸的检测结果，包含位置、置信度、特征向量和质量评分。
#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub feature: Vec<f32>,  // 人脸特征向量，用于比对和聚类
    pub quality: f32,       // 人脸质量评分，过低的人脸可能不可靠
}

/// 检测单张图片中的人脸。
pub fn detect_faces(
    image_path: &str,
    min_confidence: f64,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<FaceDetection>> {
    let data = serde_json::json!({
        "image_path": image_path,
        "min_confidence": min_confidence
    });

    let response = crate::plugins::client::call_plugin(PluginType::Face, "detect", &data, progress_cb)?;
    let faces = response.result["faces"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError("Invalid insightface response".into()))?;

    let detections: Vec<FaceDetection> = faces
        .iter()
        .filter_map(|f| {
            let bbox_arr = f["bbox"].as_array()?;
            Some(FaceDetection {
                bbox: BoundingBox {
                    x: bbox_arr.first()?.as_f64()? as f32,
                    y: bbox_arr.get(1)?.as_f64()? as f32,
                    width: bbox_arr.get(2)?.as_f64()? as f32,
                    height: bbox_arr.get(3)?.as_f64()? as f32,
                },
                confidence: f["confidence"].as_f64()? as f32,
                // 人脸特征向量，维度取决于 InsightFace 模型（通常 512 维）
                feature: f["feature"].as_array()?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|x| x as f32))
                    .collect(),
                quality: f["quality"].as_f64().unwrap_or(0.0) as f32,
            })
        })
        .collect();

    Ok(detections)
}

/// 批量检测多张图片中的人脸，避免模型反复加载的开销。
pub fn detect_faces_batch(
    image_paths: &[String],
    min_confidence: f64,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<(String, Vec<FaceDetection>)>> {
    let data = serde_json::json!({
        "image_paths": image_paths,
        "min_confidence": min_confidence
    });

    let response = crate::plugins::client::call_plugin(PluginType::Face, "detect_batch", &data, progress_cb)?;
    let results = response.result["results"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError("Invalid insightface batch response".into()))?;

    let mut all_detections = Vec::new();
    for entry in results {
        let path = entry["image_path"].as_str().unwrap_or("").to_string();
        let faces = entry["faces"].as_array()
            .map(|arr| {
                arr.iter().filter_map(|f| {
                    let bbox_arr = f["bbox"].as_array()?;
                    Some(FaceDetection {
                        bbox: BoundingBox {
                            x: bbox_arr.first()?.as_f64()? as f32,
                            y: bbox_arr.get(1)?.as_f64()? as f32,
                            width: bbox_arr.get(2)?.as_f64()? as f32,
                            height: bbox_arr.get(3)?.as_f64()? as f32,
                        },
                        confidence: f["confidence"].as_f64()? as f32,
                        feature: f["feature"].as_array()?
                            .iter()
                            .filter_map(|v| v.as_f64().map(|x| x as f32))
                            .collect(),
                        quality: f["quality"].as_f64().unwrap_or(0.0) as f32,
                    })
                }).collect()
            })
            .unwrap_or_default();
        all_detections.push((path, faces));
    }

    Ok(all_detections)
}

/// 编码单张图片中的人脸特征向量。
/// 用于人脸比对：先编码，再计算向量相似度。
pub fn encode_face(
    image_path: &str,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<f32>> {
    let data = serde_json::json!({
        "image_path": image_path
    });

    let response = crate::plugins::client::call_plugin(PluginType::Face, "encode", &data, progress_cb)?;
    let feature: Vec<f32> = response.result["feature"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError("Invalid encode response".into()))?
        .iter()
        .filter_map(|v| v.as_f64().map(|x| x as f32))
        .collect();

    Ok(feature)
}
