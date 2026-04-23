//! 统一错误类型
//!
//! 使用 thiserror 派生宏定义应用范围内的所有错误变体。
//! 每种错误对应不同的故障场景，便于上层精确匹配和处理，
//! 同时通过 #[error] 属性提供人类可读的错误描述。

use std::path::PathBuf;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum VideoSceneError {
    #[error("Video not found: {0}")]
    VideoNotFound(PathBuf),

    #[error("Video decode error: {0}")]
    VideoDecodeError(String),

    #[error("Invalid video format: {0}")]
    InvalidVideoFormat(String),

    #[error("Index not found: {0}")]
    IndexNotFound(PathBuf),

    #[error("Index corrupted: {0}")]
    IndexCorrupted(String),

    #[error("Video already indexed: {0}")]
    VideoAlreadyIndexed(Uuid),

    // ---- 插件相关错误 ----

    #[error("Plugin config error: {0}")]
    PluginConfigError(String),

    #[error("Plugin not found: {0}")]
    PluginNotFound(String),

    #[error("Plugin execution error: {0}")]
    PluginExecutionError(String),

    #[error("Plugin timeout: {0}")]
    PluginTimeout(String),

    // ---- 模型相关错误 ----

    #[error("Model load error: {0}")]
    ModelLoadError(String),

    // ---- 搜索相关错误 ----

    #[error("Face not found: {0}")]
    FaceNotFound(String),

    #[error("No search results: {0}")]
    NoSearchResults(String),

    #[error("Invalid search query: {0}")]
    InvalidSearchQuery(String),

    // ---- 存储与配置错误 ----

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Config not found: {0}")]
    ConfigNotFound(PathBuf),

    #[error("Config invalid: {0}")]
    ConfigInvalid(String),

    #[error("Config missing: {0}")]
    ConfigMissing(String),

    #[error("Plugin daemon not running: {0}")]
    DaemonNotRunning(String),
}

/// 统一的 Result 类型别名，减少重复书写 std::result::Result<..., VideoSceneError>
pub type Result<T> = std::result::Result<T, VideoSceneError>;
