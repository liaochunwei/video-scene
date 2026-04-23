//! 图像文本理解处理器：封装与本地 VLM（视觉语言模型）插件的通信。
//!
//! VLM 对视频帧进行结构化描述，输出按人/前景物/背景物/场/动作/标识/字幕分类。
//! 这些分类描述是后续语义搜索和索引的基础。

use crate::error::Result;
use crate::plugins::ProgressMessage;
use crate::plugins::PluginType;

/// VLM 生成的结构化场景描述，按语义类别拆分。
///
/// 每个类别独立存储，便于：
/// - 按类别分别做向量索引和检索
/// - 不同类别赋予不同权重（如人物比背景更重要）
/// - 字幕单独处理，不参与语义搜索但参与全文检索
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct StructuredDescription {
    pub person: String,       // 场景中出现的人
    pub foreground: String,   // 前景物体
    pub background: String,   // 背景物体
    pub scene: String,        // 场景/环境描述
    pub action: String,       // 动作描述
    pub marks: Vec<String>,   // 标识/标记（非字幕类标签）
    pub subtitles: Vec<String>, // 字幕文本，不参与语义搜索但参与全文检索
}

/// 可搜索的描述类别枚举。
/// 注意：字幕（subtitles）刻意不列入此枚举，因为字幕走全文检索而非语义搜索。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescriptionCategory {
    Person,
    Foreground,
    Background,
    Scene,
    Action,
    Marks,
}

impl DescriptionCategory {
    /// 所有可搜索类别的列表，用于遍历构建索引。
    pub fn all() -> &'static [DescriptionCategory] {
        &[
            DescriptionCategory::Person,
            DescriptionCategory::Foreground,
            DescriptionCategory::Background,
            DescriptionCategory::Scene,
            DescriptionCategory::Action,
            DescriptionCategory::Marks,
        ]
    }

    /// CLIP 向量检索时的权重：人物和前景权重高，背景权重低。
    /// 这样搜索"一个人在跑步"时，人物和动作维度的匹配更优先。
    pub fn clip_weight(&self) -> f32 {
        match self {
            DescriptionCategory::Person => 0.7,
            DescriptionCategory::Foreground => 0.8,
            DescriptionCategory::Marks => 0.6,
            DescriptionCategory::Scene => 0.5,
            DescriptionCategory::Action => 0.4,
            DescriptionCategory::Background => 0.3,
        }
    }

    /// 每个类别对应一个独立的 HNSW 索引文件，按类别隔离检索。
    pub fn index_filename(&self) -> &'static str {
        match self {
            DescriptionCategory::Person => "scenes_person.hnsw",
            DescriptionCategory::Foreground => "scenes_foreground.hnsw",
            DescriptionCategory::Background => "scenes_background.hnsw",
            DescriptionCategory::Scene => "scenes_scene.hnsw",
            DescriptionCategory::Action => "scenes_action.hnsw",
            DescriptionCategory::Marks => "scenes_marks.hnsw",
        }
    }
}

impl StructuredDescription {
    /// 获取指定类别的文本内容，标识类别用逗号连接。
    pub fn get_text(&self, category: DescriptionCategory) -> String {
        match category {
            DescriptionCategory::Person => self.person.clone(),
            DescriptionCategory::Foreground => self.foreground.clone(),
            DescriptionCategory::Background => self.background.clone(),
            DescriptionCategory::Scene => self.scene.clone(),
            DescriptionCategory::Action => self.action.clone(),
            DescriptionCategory::Marks => self.marks.join("，"),
        }
    }

    /// 拼接所有字段为完整描述文本，用于全文检索（FTS）。
    /// 字幕也包含在内，因为全文检索需要匹配字幕内容。
    pub fn to_full_text(&self) -> String {
        let mut parts = vec![
            format!("人: {}", self.person),
            format!("前景物: {}", self.foreground),
            format!("背景物: {}", self.background),
            format!("场: {}", self.scene),
            format!("动作: {}", self.action),
        ];
        if !self.marks.is_empty() {
            parts.push(format!("标识: {}", self.marks.join(", ")));
        }
        if !self.subtitles.is_empty() {
            parts.push(format!("字幕: {}", self.subtitles.join(", ")));
        }
        parts.join("\n")
    }
}

/// 描述单个场景，传入一帧或多帧图像。
/// VLM 通过多帧能更好地理解动作和时间上下文。
pub fn describe_scene(
    image_paths: &[String],
    model: Option<&str>,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<StructuredDescription> {
    let data = serde_json::json!({
        "image_paths": image_paths,
        "model": model.unwrap_or("")
    });

    let response = crate::plugins::client::call_plugin(PluginType::ImageTextUnderstanding, "describe_scene", &data, progress_cb)?;
    let structured = parse_structured_response(&response.result["structured"]);

    Ok(structured)
}

/// 批量描述多个场景，每个场景可包含多帧图像。
/// 一次调用处理所有场景，减少进程间通信开销。
pub fn describe_scenes_batch(
    scenes: &[Vec<String>],
    model: Option<&str>,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<StructuredDescription>> {
    let scenes_json: Vec<serde_json::Value> = scenes
        .iter()
        .map(|paths| {
            serde_json::json!({ "image_paths": paths })
        })
        .collect();

    let data = serde_json::json!({
        "scenes": scenes_json,
        "model": model.unwrap_or("")
    });

    let response = crate::plugins::client::call_plugin(PluginType::ImageTextUnderstanding, "describe_scenes_batch", &data, progress_cb)?;
    let results = response.result["results"]
        .as_array()
        .ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError(
            "Invalid VLM batch response".into()
        ))?;

    let descriptions: Vec<StructuredDescription> = results
        .iter()
        .map(|entry| parse_structured_response(&entry["structured"]))
        .collect();

    Ok(descriptions)
}

/// 解析 VLM 插件返回的结构化 JSON 为 StructuredDescription。
/// VLM 输出的字段名是中文（人、前景物等），这里做映射转换。
/// 字幕修正逻辑已在 Python 插件端完成，此处只做解析。
pub(crate) fn parse_structured_response(value: &serde_json::Value) -> StructuredDescription {
    let empty_map = serde_json::Map::new();
    let obj = value.as_object().unwrap_or(&empty_map);
    StructuredDescription {
        person: obj.get("人").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        foreground: obj.get("前景物").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        background: obj.get("背景物").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        scene: obj.get("场").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        action: obj.get("动作").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        marks: obj.get("标识").and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        subtitles: obj.get("字幕").and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
    }
}
