//! 图像文本向量化处理器：封装与 CLIP 插件的通信。
//!
//! CLIP 将图像编码为高维向量，用于跨模态检索（以图搜图、以文搜图）。
//! 提供单张/批量编码功能。

use crate::error::Result;
use crate::plugins::ProgressMessage;
use crate::plugins::PluginType;

/// 编码单张图片为 CLIP 特征向量，用于语义检索。
pub fn encode_image(
    image_path: &str,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<f32>> {
    let data = serde_json::json!({
        "image_path": image_path
    });

    let response = crate::plugins::client::call_plugin(PluginType::ImageTextVectorization, "encode_image", &data, progress_cb)?;
    let feature: Vec<f32> = response.result["feature"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError("Invalid clip image response".into()))?
        .iter()
        .filter_map(|v| v.as_f64().map(|x| x as f32))
        .collect();

    Ok(feature)
}

/// 批量编码多张图片，避免模型反复加载的开销。
/// 返回 (图片路径, 特征向量) 的配对列表。
pub fn encode_images_batch(
    image_paths: &[String],
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<(String, Vec<f32>)>> {
    let data = serde_json::json!({
        "image_paths": image_paths
    });

    let response = crate::plugins::client::call_plugin(PluginType::ImageTextVectorization, "encode_images_batch", &data, progress_cb)?;
    let results = response.result["results"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError("Invalid clip batch response".into()))?;

    let mut all_features = Vec::new();
    for entry in results {
        let path = entry["image_path"].as_str().unwrap_or("").to_string();
        let feature: Vec<f32> = entry["feature"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect())
            .unwrap_or_default();
        all_features.push((path, feature));
    }

    Ok(all_features)
}
