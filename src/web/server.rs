//! Axum HTTP 服务器启动与路由配置
//!
//! 定义所有 HTTP 路由并启动监听。路由分为三部分：
//! - 前端页面和静态资源（index.html、assets、favicon）
//! - REST API 接口（搜索、视频详情、关键帧图片）
//! - 兜底路由：未匹配的路径返回 index.html，支持前端 SPA 路由

use axum::routing::get;
use axum::Router;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::web::api;
use crate::web::static_files;

/// 启动 Web 服务器，监听指定端口。
///
/// 绑定 0.0.0.0 以接受所有网络接口的连接，方便容器化部署。
/// 使用 permissive CORS 策略，允许任意来源访问（适合内部工具场景）。
pub async fn start_server(state: Arc<crate::web::api::AppState>, port: u16) -> anyhow::Result<()> {
    let app = Router::new()
        // 前端页面路由
        .route("/", get(static_files::serve_index))
        .route("/assets/{*path}", get(static_files::serve_asset))
        .route("/favicon.svg", get(static_files::serve_favicon))
        // API 路由
        .route("/api/search", get(api::search_handler))
        .route("/api/video/{video_id}", get(api::video_handler))
        .route("/api/keyframes/{*path}", get(api::keyframe_handler))
        // SPA 兜底：所有未匹配路径返回 index.html，交由前端路由处理
        .fallback(static_files::serve_index)
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Web UI: http://localhost:{}", port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
