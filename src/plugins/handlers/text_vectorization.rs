//! 文本向量化处理器：封装与 BGE 文本嵌入插件的通信。
//!
//! 提供查询向量和文档向量两种编码模式：
//! - 查询编码：添加 SEARCH_INSTRUCT 前缀，优化检索召回
//! - 文档编码：不加前缀，用于构建索引
//!
//! 还支持按类别编码，用于分类检索场景（如按人物/场景/动作分别匹配）。

use std::collections::HashMap;

use crate::error::{Result, VideoSceneError};
use crate::plugins::ProgressMessage;
use crate::plugins::PluginType;

/// 编码单条搜索查询，使用 SEARCH_INSTRUCT 前缀优化检索效果。
/// 返回 L2 归一化的 1024 维向量。
pub fn encode_text(
    text: &str,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<f32>> {
    let data = serde_json::json!({
        "text": text
    });

    let response = crate::plugins::client::call_plugin(PluginType::TextVectorization, "encode_text", &data, progress_cb)?;
    let vector: Vec<f32> = response.result["vector"]
        .as_array()
        .ok_or_else(|| VideoSceneError::PluginExecutionError("Invalid embedding text response".into()))?
        .iter()
        .filter_map(|v| v.as_f64().map(|x| x as f32))
        .collect();

    Ok(vector)
}

/// 批量编码多条搜索查询，减少进程间通信开销。
/// 返回 (文本, 向量) 配对列表。
pub fn encode_texts_batch(
    texts: &[String],
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<(String, Vec<f32>)>> {
    let data = serde_json::json!({
        "texts": texts
    });

    let response = crate::plugins::client::call_plugin(PluginType::TextVectorization, "encode_texts_batch", &data, progress_cb)?;
    let results = response.result["results"]
        .as_array()
        .ok_or_else(|| VideoSceneError::PluginExecutionError("Invalid embedding texts batch response".into()))?;

    let mut all_features = Vec::new();
    for entry in results {
        let text = entry["text"].as_str().unwrap_or("").to_string();
        let vector: Vec<f32> = entry["vector"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect())
            .unwrap_or_default();
        all_features.push((text, vector));
    }

    Ok(all_features)
}

/// 编码单条文档文本，不加指令前缀，用于构建索引。
/// 与 encode_text 的区别是：文档编码无需 SEARCH_INSTRUCT 前缀，
/// 这是 BGE 模型为区分查询侧和文档侧而设计的机制。
pub fn encode_document(
    text: &str,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<f32>> {
    let data = serde_json::json!({
        "text": text
    });

    let response = crate::plugins::client::call_plugin(PluginType::TextVectorization, "encode_document", &data, progress_cb)?;
    let vector: Vec<f32> = response.result["vector"]
        .as_array()
        .ok_or_else(|| VideoSceneError::PluginExecutionError("Invalid embedding document response".into()))?
        .iter()
        .filter_map(|v| v.as_f64().map(|x| x as f32))
        .collect();

    Ok(vector)
}

/// 批量编码文档文本（不加指令前缀），用于批量构建索引。
pub fn encode_documents_batch(
    texts: &[String],
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<(String, Vec<f32>)>> {
    let data = serde_json::json!({
        "texts": texts
    });

    let response = crate::plugins::client::call_plugin(PluginType::TextVectorization, "encode_documents_batch", &data, progress_cb)?;
    let results = response.result["results"]
        .as_array()
        .ok_or_else(|| VideoSceneError::PluginExecutionError("Invalid embedding documents batch response".into()))?;

    let mut all_features = Vec::new();
    for entry in results {
        let text = entry["text"].as_str().unwrap_or("").to_string();
        let vector: Vec<f32> = entry["vector"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|x| x as f32)).collect())
            .unwrap_or_default();
        all_features.push((text, vector));
    }

    Ok(all_features)
}

/// 编码搜索查询并返回查询向量和预计算的类别标签向量。
///
/// 查询使用 SEARCH_INSTRUCT 编码，类别使用 CATEGORY_INSTRUCT 编码。
/// 类别向量在模型加载时预计算并缓存，避免每次查询重复编码。
/// 用于分类检索：先算查询与各类别的相似度，再在匹配类别内做细粒度检索。
pub fn encode_text_with_categories(
    text: &str,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<(Vec<f32>, HashMap<String, Vec<f32>>)> {
    let data = serde_json::json!({
        "text": text
    });

    let response = crate::plugins::client::call_plugin(PluginType::TextVectorization, "encode_text_with_categories", &data, progress_cb)?;

    let query_vector: Vec<f32> = response.result["query_vector"]
        .as_array()
        .ok_or_else(|| VideoSceneError::PluginExecutionError(
            "Invalid query_vector in encode_text_with_categories response".into()
        ))?
        .iter()
        .filter_map(|v| v.as_f64().map(|x| x as f32))
        .collect();

    // 解析各类别的预计算向量
    let mut category_vectors = HashMap::new();
    if let Some(cats) = response.result["category_vectors"].as_object() {
        for (key, val) in cats {
            if let Some(arr) = val.as_array() {
                let vec: Vec<f32> = arr.iter()
                    .filter_map(|v| v.as_f64().map(|x| x as f32))
                    .collect();
                category_vectors.insert(key.clone(), vec);
            }
        }
    }

    Ok((query_vector, category_vectors))
}
