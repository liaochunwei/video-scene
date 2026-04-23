//! API 响应类型定义
//!
//! 定义搜索 API 返回给前端的 JSON 结构体。
//! 与内部 SearchResult 分离，便于独立调整 API 格式而不影响核心逻辑。

use serde::Serialize;

/// "更多匹配片段"的简要信息（搜索结果中附带的相关片段）
#[derive(Serialize)]
pub struct MoreSegmentJson {
    pub segment_id: String,
    pub start_time: f32,
    pub end_time: f32,
    pub confidence: f32,
    pub keyframe_url: String,
}

/// 搜索响应的外层结构，包含分页信息
#[derive(Serialize)]
pub struct SearchResponse {
    pub total: usize,
    pub page: usize,
    pub page_size: usize,
    pub results: Vec<SearchResultJson>,
}

/// 单条搜索结果的详细内容
#[derive(Serialize)]
pub struct SearchResultJson {
    pub video_id: String,
    pub filename: String,
    pub start_time: f32,
    pub end_time: f32,
    pub keyframe_url: String,
    pub confidence: f32,
    pub confidence_low: f32,
    pub confidence_high: f32,
    pub match_type: String,
    pub match_label: String,
    pub description: String,
    /// 同一视频中的其他匹配片段，供前端展示"更多结果"
    pub more: Vec<MoreSegmentJson>,
}
