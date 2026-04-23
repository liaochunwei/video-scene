//! 插件子系统：为视频分析提供可扩展的外部能力（人脸检测、目标检测、VLM 描述等）。
//!
//! 架构概览：
//! - 每个插件是独立进程，通过 Unix Socket + NDJSON 与主程序通信
//! - 守护进程（daemon）常驻后台管理插件生命周期，CLI 通过守护进程转发调用
//! - 配置文件 plugin.toml 声明插件元信息和运行参数
//!
//! 模块组织：
//! - `protocol`  — IPC 协议定义（NDJSON 消息格式）
//! - `process`   — 插件进程启动、通信、停止
//! - `daemon`    — 守护进程主循环
//! - `client`    — CLI 与守护进程通信的客户端
//! - `manager`   — 插件发现、注册、空闲回收等生命周期管理
//! - `config`    — plugin.toml 配置解析
//! - `types`     — PluginType 枚举
//! - `handlers`  — 各类插件的具体调用逻辑

pub mod client;
pub mod config;
pub mod daemon;
pub mod handlers;
pub mod manager;
pub mod process;
pub mod protocol;
pub mod types;

// 为保持向后兼容，将 handlers 子模块直接 re-export 到 plugins 层级
pub use handlers::face;
pub use handlers::image_text_understanding;
pub use handlers::image_text_vectorization;
pub use handlers::object;
pub use handlers::text_vectorization;
pub use handlers::video_segmentation;
pub use handlers::video_understanding;

pub use manager::{PluginManager, PluginResponse, ProgressMessage};
pub use types::PluginType;
