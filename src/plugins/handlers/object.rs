//! 目标检测处理器：封装与 YOLO 插件的通信。
//!
//! 提供单张/批量检测功能，支持按类别过滤。
//! 检测结果包含中英文标签，便于后续多语言展示和搜索。

use crate::error::Result;
use crate::models::BoundingBox;
use crate::plugins::ProgressMessage;
use crate::plugins::PluginType;

/// 单个目标的检测结果，包含标签、置信度和位置。
#[derive(Debug, Clone)]
pub struct ObjectDetection {
    pub label: String,       // 英文标签，如 "person"
    pub label_zh: String,    // 中文标签，如 "人"，便于中文场景展示
    pub confidence: f32,
    pub bbox: BoundingBox,
}

/// 检测单张图片中的目标，可按类别过滤。
pub fn detect_objects(
    image_path: &str,
    min_confidence: f64,
    classes: Option<&[String]>,  // 可选的类别白名单，为空则检测所有类别
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<ObjectDetection>> {
    let mut data = serde_json::json!({
        "image_path": image_path,
        "min_confidence": min_confidence
    });
    if let Some(c) = classes {
        data["classes"] = serde_json::json!(c);
    }

    let response = crate::plugins::client::call_plugin(PluginType::Object, "detect", &data, progress_cb)?;
    let objects = response.result["objects"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError("Invalid yolo response".into()))?;

    let detections: Vec<ObjectDetection> = objects
        .iter()
        .filter_map(|o| {
            let bbox_arr = o["bbox"].as_array()?;
            Some(ObjectDetection {
                label: o["label"].as_str()?.to_string(),
                label_zh: o["label_zh"].as_str().unwrap_or("").to_string(),
                confidence: o["confidence"].as_f64()? as f32,
                bbox: BoundingBox {
                    x: bbox_arr.first()?.as_f64()? as f32,
                    y: bbox_arr.get(1)?.as_f64()? as f32,
                    width: bbox_arr.get(2)?.as_f64()? as f32,
                    height: bbox_arr.get(3)?.as_f64()? as f32,
                },
            })
        })
        .collect();

    Ok(detections)
}

/// 批量检测多张图片中的目标，避免模型反复加载的开销。
pub fn detect_objects_batch(
    image_paths: &[String],
    min_confidence: f64,
    classes: Option<&[String]>,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<(String, Vec<ObjectDetection>)>> {
    let mut data = serde_json::json!({
        "image_paths": image_paths,
        "min_confidence": min_confidence
    });
    if let Some(c) = classes {
        data["classes"] = serde_json::json!(c);
    }

    let response = crate::plugins::client::call_plugin(PluginType::Object, "detect_batch", &data, progress_cb)?;
    let results = response.result["results"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError("Invalid yolo batch response".into()))?;

    let mut all_detections = Vec::new();
    for entry in results {
        let path = entry["image_path"].as_str().unwrap_or("").to_string();
        let objects: Vec<ObjectDetection> = entry["objects"].as_array()
            .map(|arr| {
                arr.iter().filter_map(|o| {
                    let bbox_arr = o["bbox"].as_array()?;
                    Some(ObjectDetection {
                        label: o["label"].as_str()?.to_string(),
                        label_zh: o["label_zh"].as_str().unwrap_or("").to_string(),
                        confidence: o["confidence"].as_f64()? as f32,
                        bbox: BoundingBox {
                            x: bbox_arr.first()?.as_f64()? as f32,
                            y: bbox_arr.get(1)?.as_f64()? as f32,
                            width: bbox_arr.get(2)?.as_f64()? as f32,
                            height: bbox_arr.get(3)?.as_f64()? as f32,
                        },
                    })
                }).collect()
            })
            .unwrap_or_default();
        all_detections.push((path, objects));
    }

    Ok(all_detections)
}
