//! REST API 处理器
//!
//! 定义搜索、视频详情和关键帧图片三个核心 API 端点的处理逻辑。
//! 每个处理器从 AppState 中获取存储层实例，执行查询后将结果转换为 JSON 响应。

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Deserialize;
use std::sync::Arc;

use crate::config::Settings;
use crate::core::searcher::{self, SearchType};
use crate::error::Result;
use crate::plugins::ProgressMessage;
use crate::storage::Storage;

use super::types::{MoreSegmentJson, SearchResponse, SearchResultJson};

/// 应用共享状态，持有 Storage 的互斥访问权。
///
/// 使用 Mutex 而非 RwLock 是因为搜索操作既需要读也需要写（如缓存更新），
/// 且 Web 场景下并发冲突概率低，Mutex 实现更简单。
pub struct AppState {
    pub storage: std::sync::Mutex<Storage>,
}

/// 搜索接口的查询参数
#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub q: Option<String>,
    pub search_type: Option<String>,
    pub top_k: Option<usize>,
    pub threshold: Option<f32>,
    pub page: Option<usize>,
    pub page_size: Option<usize>,
    pub image: Option<String>,
    pub dedup: Option<bool>,
    /// 以关键帧 URL 为输入进行以图搜图（与 q 互斥，二选一）
    pub keyframe_search: Option<String>,
}

/// GET /api/search — 搜索处理器
///
/// 支持文本搜索和以图搜图两种模式。当 keyframe_search 参数存在时，
/// 即使 q 为空也允许搜索（以图搜图场景）；否则 q 为必填参数。
pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Response {
    // 判断是否为以图搜图模式：keyframe_search 存在时允许 q 为空
    let query = match params.q {
        Some(ref q) if !q.is_empty() => q.clone(),
        _ if params.keyframe_search.is_some() => String::new(),
        _ => return (StatusCode::BAD_REQUEST, "Missing query parameter 'q'").into_response(),
    };

    let search_type = match params.search_type.as_deref() {
        Some("face") => SearchType::Face,
        Some("object") => SearchType::Object,
        Some("scene") => SearchType::Scene,
        Some("image") => SearchType::Image,
        _ => SearchType::Auto, // 默认自动推断搜索类型
    };

    let top_k = params.top_k.unwrap_or(20);
    let threshold = params.threshold.unwrap_or(0.0);
    let page = params.page.unwrap_or(1).max(1); // 页码最小为 1
    let page_size = params.page_size.unwrap_or(20).max(1);
    let dedup = params.dedup.unwrap_or(false);

    // Web 模式下无需进度回调，使用空实现
    let progress_cb = |_: ProgressMessage| {};

    let result: Result<crate::core::searcher::SearchResponse> = {
        let storage = state.storage.lock().unwrap();
        let settings = Settings::load(None).unwrap_or_default();

        // 将 keyframe URL 转换为磁盘上的实际路径：
        // 去掉 "/api/keyframes/" 前缀，拼接到 file_store 的 keyframes 目录下
        let keyframe_image_path = if let Some(ref kf_url) = params.keyframe_search {
            let relative = kf_url.strip_prefix("/api/keyframes/").unwrap_or(kf_url);
            // 防止路径穿越攻击
            if relative.contains("..") {
                return StatusCode::FORBIDDEN.into_response();
            }
            storage.file_store.base_dir().join("keyframes").join(relative).to_string_lossy().to_string()
        } else {
            params.image.as_deref().unwrap_or("").to_string()
        };

        searcher::search(
            &query,
            search_type,
            top_k,
            threshold,
            page,
            page_size,
            dedup,
            &settings,
            &storage.workspace_db,
            &storage.config_db,
            &storage.face_index,
            &storage.scene_indices,
            &storage.image_index,
            &keyframe_image_path,
            &progress_cb,
        )
    };

    match result {
        Ok(resp) => {
            // 将内部搜索结果转换为前端 JSON 格式
            let json_resp = SearchResponse {
                total: resp.total,
                page: resp.page,
                page_size: resp.page_size,
                results: resp
                    .results
                    .into_iter()
                    .map(|r| {
                        let video_id_str = r.video.id.to_string();
                        let video_id_for_keyframes = r.best_segment.video_id;
                        SearchResultJson {
                            video_id: video_id_str,
                            filename: r.video.filename,
                            start_time: r.best_segment.start_time,
                            end_time: r.best_segment.end_time,
                            // 构造关键帧的访问 URL，前端直接用此 URL 展示缩略图
                            keyframe_url: format!(
                                "/api/keyframes/{}/{}.jpg",
                                video_id_for_keyframes, r.best_segment.id
                            ),
                            confidence: r.confidence,
                            confidence_low: r.confidence_low,
                            confidence_high: r.confidence_high,
                            match_type: r.match_type,
                            match_label: r.match_label,
                            description: r.best_segment.scene_description,
                            // 同一视频中的更多匹配片段
                            more: r.more.into_iter().map(|m| MoreSegmentJson {
                                segment_id: m.segment_id.to_string(),
                                start_time: m.start_time,
                                end_time: m.end_time,
                                confidence: m.confidence,
                                keyframe_url: format!(
                                    "/api/keyframes/{}/{}.jpg",
                                    video_id_for_keyframes, m.segment_id
                                ),
                            }).collect(),
                        }
                    })
                    .collect(),
            };
            Json(json_resp).into_response()
        }
        Err(e) => {
            tracing::error!("Search failed: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Search failed: {}", e)).into_response()
        }
    }
}

/// GET /api/video/{video_id} — 视频文件下载处理器
///
/// 将整个视频文件读入内存后返回。适用于中小型视频；
/// 超大文件应考虑使用流式响应或 range 请求以节省内存。
pub async fn video_handler(
    State(state): State<Arc<AppState>>,
    Path(video_id): Path<String>,
) -> Response {
    let id = match video_id.parse::<uuid::Uuid>() {
        Ok(id) => id,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    // 从数据库查询视频路径，然后异步读取文件内容
    let video_path = {
        let storage = state.storage.lock().unwrap();
        match storage.workspace_db.get_video_by_id(&id) {
            Ok(Some(v)) => v.path.clone(),
            Ok(None) => return StatusCode::NOT_FOUND.into_response(),
            Err(_) => return StatusCode::NOT_FOUND.into_response(),
        }
    };

    match tokio::fs::read(&video_path).await {
        Ok(data) => Response::builder()
            .header("content-type", "video/mp4")
            .header(
                "content-length",
                data.len().to_string(),
            )
            // 声明支持 range 请求，虽然当前实现是全量返回，
            // 但可以让浏览器尝试 range 请求（部分播放器依赖此头）
            .header("accept-ranges", "bytes")
            .body(Body::from(data))
            .unwrap(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

/// GET /api/keyframes/{video_id}/{segment_id}.jpg — 关键帧图片处理器
///
/// 从磁盘读取关键帧 JPEG 文件并返回。包含双重路径安全检查：
/// 1. 拒绝包含 ".." 的路径（快速拦截）
/// 2. canonicalize 后验证解析路径仍在 keyframes 目录内（防止符号链接逃逸）
pub async fn keyframe_handler(
    State(state): State<Arc<AppState>>,
    Path(path): Path<String>,
) -> Response {
    // 第一层防护：直接拒绝明显的路径穿越字符
    if path.contains("..") {
        return StatusCode::FORBIDDEN.into_response();
    }

    let (full_path, keyframes_dir) = {
        let storage = state.storage.lock().unwrap();
        let base = storage.file_store.base_dir();
        (base.join("keyframes").join(&path), base.join("keyframes"))
    };

    // 第二层防护：canonicalize 解析符号链接后，验证路径仍在允许目录下
    match full_path.canonicalize() {
        Ok(canonical) => {
            if let Ok(keyframes_canonical) = keyframes_dir.canonicalize() {
                if !canonical.starts_with(&keyframes_canonical) {
                    return StatusCode::FORBIDDEN.into_response();
                }
            }
        }
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    }

    match tokio::fs::read(&full_path).await {
        Ok(data) => Response::builder()
            .header("content-type", "image/jpeg")
            // 关键帧内容不会频繁变化，缓存 1 天减少服务器负载
            .header("cache-control", "public, max-age=86400")
            .body(Body::from(data))
            .unwrap(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}
