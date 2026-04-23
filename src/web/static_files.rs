//! 嵌入式静态资源服务
//!
//! 使用 rust_embed 在编译时将前端构建产物（web/dist/）打包进二进制文件，
//! 运行时无需额外部署静态文件服务器。通过 MIME 类型推断确保正确的 Content-Type。

use rust_embed::RustEmbed;
use axum::body::Body;
use axum::extract::Path;
use axum::response::{IntoResponse, Response};

// 编译时嵌入 web/dist/ 目录下的所有文件，生成单文件部署包
#[derive(RustEmbed)]
#[folder = "web/dist/"]
struct Assets;

/// 返回前端入口页面（index.html）
pub async fn serve_index() -> impl IntoResponse {
    match Assets::get("index.html") {
        Some(index) => {
            let mime = mime_guess::from_path("index.html").first_or_octet_stream();
            Response::builder()
                .header("content-type", mime.as_ref())
                .body(Body::from(index.data.to_vec()))
                .unwrap()
        }
        None => Response::builder()
            .status(404)
            .body(Body::from("Not found"))
            .unwrap(),
    }
}

/// 返回站点图标（favicon.svg）
pub async fn serve_favicon() -> impl IntoResponse {
    match Assets::get("favicon.svg") {
        Some(file) => {
            let mime = mime_guess::from_path("favicon.svg").first_or_octet_stream();
            Response::builder()
                .header("content-type", mime.as_ref())
                .body(Body::from(file.data.to_vec()))
                .unwrap()
        }
        None => Response::builder()
            .status(404)
            .body(Body::from("Not found"))
            .unwrap(),
    }
}

/// 返回 assets/ 目录下的静态资源（JS、CSS 等）
///
/// 路径参数为 assets/ 下的相对路径，如 "index.js" 或 "style.css"。
/// MIME 类型根据文件扩展名自动推断。
pub async fn serve_asset(Path(path): Path<String>) -> impl IntoResponse {
    let full_path = format!("assets/{path}");
    match Assets::get(&full_path) {
        Some(file) => {
            let mime = mime_guess::from_path(&full_path).first_or_octet_stream();
            Response::builder()
                .header("content-type", mime.as_ref())
                .body(Body::from(file.data.to_vec()))
                .unwrap()
        }
        None => Response::builder()
            .status(404)
            .body(Body::from("Not found"))
            .unwrap(),
    }
}
