//! 应用配置模块
//!
//! 将应用配置拆分为两个独立关注点：
//! - `settings`：用户可编辑的 TOML 配置文件，控制视频处理、索引、搜索等行为参数
//! - `state`：运行时自动维护的 JSON 状态文件，记录跨会话的临时状态（如当前工作区）

pub mod settings;
pub mod state;

pub use settings::Settings;
pub use state::StateFile;
