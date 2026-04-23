//! Web 模块
//!
//! 提供基于 Axum 的 HTTP 服务，包括 REST API 接口和前端静态资源托管。
//! 用户可通过浏览器访问 Web UI 进行视频搜索和浏览。

pub mod api;
pub mod server;
pub mod static_files;
pub mod types;

// 导出共享的应用状态和服务器启动入口
pub use api::AppState;
pub use server::start_server;
