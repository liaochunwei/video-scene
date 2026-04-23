//! 视频元数据模型
//!
//! 本模块定义了已索引视频的核心数据结构。每个视频在入库时会自动分配 UUID 作为主键，
//! 确保即使文件名相同也能唯一区分。`created_at` 与 `indexed_at` 在创建时取相同值，
//! 后续可通过业务逻辑更新 `created_at` 为文件的真实创建时间。

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 一个已索引视频的完整元数据。
///
/// 持久化时以 `id` 为主键；`path` 为绝对路径，用于定位原始文件；
/// `filename` 仅用于 UI 展示，避免在界面上暴露完整路径。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Video {
    pub id: Uuid,
    pub path: String,
    pub filename: String,
    pub duration: f32,
    pub width: u32,
    pub height: u32,
    /// 文件创建时间（入库时默认取当前时间，后续可修正为文件系统记录的真实时间）
    pub created_at: NaiveDateTime,
    /// 入库时间，标记视频被系统索引的时刻，一旦写入不再变更
    pub indexed_at: NaiveDateTime,
}

impl Video {
    /// 根据基本视频信息构造 `Video`，自动生成 UUID 并将时间戳初始化为当前 UTC 时间。
    pub fn new(path: String, filename: String, duration: f32, width: u32, height: u32) -> Self {
        let now = chrono::Local::now().naive_utc();
        Self {
            id: Uuid::new_v4(),
            path,
            filename,
            duration,
            width,
            height,
            created_at: now,
            indexed_at: now,
        }
    }

    /// 返回 (宽, 高) 元组，方便调用方解构使用。
    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
