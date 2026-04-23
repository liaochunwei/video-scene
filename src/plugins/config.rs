//! 插件配置解析：从 plugin.toml 加载插件元信息和运行参数。
//!
//! 每个 plugin 目录下必须有一个 plugin.toml，声明插件的名称、类型、启动命令等。
//! Rust 端根据配置决定如何拉起子进程、超时多久、支持哪些 action。

use serde::Deserialize;
use std::path::PathBuf;

use super::types::PluginType;

/// 插件完整配置，对应 plugin.toml 的顶层结构。
#[derive(Debug, Clone, Deserialize)]
pub struct PluginConfig {
    /// 插件基本信息（名称、版本、类型）
    pub plugin: PluginInfo,
    /// 运行时参数（启动命令、超时）
    pub runtime: RuntimeInfo,
    /// 插件支持的能力（可选，默认空）
    #[serde(default)]
    pub capabilities: CapabilitiesInfo,
    /// 额外类型别名（可选，用于一个插件服务多种类型）
    #[serde(default)]
    pub extra_types: ExtraTypesInfo,
}

/// 插件身份信息，对应 plugin.toml 的 [plugin] 段。
#[derive(Debug, Clone, Deserialize)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    /// 使用自定义反序列化，因为 TOML 中 type 是字符串，需映射为枚举
    #[serde(rename = "type", deserialize_with = "deserialize_plugin_type")]
    pub plugin_type: PluginType,
    #[serde(default)]
    pub description: String,
}

/// 运行时参数，对应 plugin.toml 的 [runtime] 段。
#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeInfo {
    /// 启动命令，如 `python -u server.py`；socket 路径会自动追加为最后一个参数
    pub command: String,
    /// 空闲超时（秒），超时后守护进程自动回收进程，0 表示永不超时
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u64,
    /// 启动超时（秒），子进程需在此时间内连接 socket 并完成注册
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout: u64,
}

/// 插件能力声明，对应 plugin.toml 的 [capabilities] 段。
#[derive(Debug, Clone, Deserialize, Default)]
pub struct CapabilitiesInfo {
    /// 支持的 action 列表，如 ["detect", "encode"]
    #[serde(default)]
    pub actions: Vec<String>,
    /// 单次批量调用的最大条目数，防止内存溢出
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
}

/// 额外类型映射，对应 plugin.toml 的 [extra_types] 段。
/// key 是插件类型字符串，value 是对应的 action（暂未使用）。
/// 典型场景：CLIP 插件同时服务 image_text_vectorization 和 text_vectorization。
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ExtraTypesInfo {
    #[serde(default)]
    pub extra_types: std::collections::HashMap<String, String>,
}

// 默认值：空闲 5 分钟回收，启动超时 30 秒，批量上限 16 条
fn default_idle_timeout() -> u64 { 300 }
fn default_startup_timeout() -> u64 { 30 }
fn default_max_batch_size() -> usize { 16 }

/// 自定义反序列化：将 TOML 中的字符串（如 "face"）映射为 PluginType 枚举。
/// 不直接用 serde derive 是因为 TOML 的字符串值需要特殊处理。
fn deserialize_plugin_type<'de, D>(deserializer: D) -> std::result::Result<PluginType, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "face" => Ok(PluginType::Face),
        "object" => Ok(PluginType::Object),
        "video_understanding" => Ok(PluginType::VideoUnderstanding),
        "video_segmentation" => Ok(PluginType::VideoSegmentation),
        "image_text_understanding" => Ok(PluginType::ImageTextUnderstanding),
        "text_vectorization" => Ok(PluginType::TextVectorization),
        "image_text_vectorization" => Ok(PluginType::ImageTextVectorization),
        other => Err(serde::de::Error::custom(format!("unknown plugin type: {}", other))),
    }
}

impl PluginConfig {
    /// 从 plugin.toml 文件路径加载配置。
    pub fn load(path: &PathBuf) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::error::VideoSceneError::PluginConfigError(
                format!("Failed to read {}: {}", path.display(), e)
            ))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| crate::error::VideoSceneError::PluginConfigError(
                format!("Failed to parse {}: {}", path.display(), e)
            ))?;
        Ok(config)
    }
}
