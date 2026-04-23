//! 统一搜索模块
//!
//! 在人脸、物体、场景（CLIP 向量 + BM25 文本）、图片等多路信号上执行搜索，
//! 然后按权重融合为统一的置信度分数。核心设计思路：
//! - 各信号独立收集（`collect_*_signals`），互不干扰
//! - 融合时按预设权重加权求和，并根据信号数量估计置信区间
//! - 支持按视频去重，将同一视频的多个匹配片段归入 `more` 列表

use std::collections::HashMap;

use crate::config::Settings;
use crate::error::{Result, VideoSceneError};
use crate::models::{DetectionType, Segment, Video};
use crate::plugins::ProgressMessage;
use crate::plugins::image_text_understanding::DescriptionCategory;
use crate::storage::{ConfigDatabase, Database, SceneIndices, VectorIndex};

/// 搜索类型枚举，决定激活哪些信号通道。
/// `Auto` 模式下同时启用人脸、物体和场景三路信号。
#[derive(Debug, Clone)]
pub enum SearchType {
    Face,
    Object,
    Scene,
    Image,
    Auto,
}

/// 单个信号对某个视频片段的原始得分贡献。
/// 每个信号来自一个通道（人脸/物体/场景），携带原始分数和标签。
#[derive(Debug, Clone)]
struct SignalScore {
    video: Video,
    segment: Segment,
    raw_score: f32,
    signal_type: String, // "face", "object", "scene", "bm25", "image"
    label: String,
}

/// 同一视频下的附加匹配片段，用于去重后展示"更多结果"。
#[derive(Debug, Clone, serde::Serialize)]
pub struct MoreSegment {
    pub segment_id: uuid::Uuid,
    pub start_time: f32,
    pub end_time: f32,
    pub confidence: f32,
    pub keyframe_path: String,
}

/// 最终的统一搜索结果，包含加权置信度和置信区间。
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub video: Video,
    pub best_segment: Segment,
    /// 加权统一置信度 [0, 1]
    pub confidence: f32,
    /// 置信区间下界
    pub confidence_low: f32,
    /// 置信区间上界
    pub confidence_high: f32,
    /// 命中的信号类型组合，如 "face+scene"
    pub match_type: String,
    /// 各信号的标签描述
    pub match_label: String,
    /// 同一视频的附加匹配片段（去重后填充）
    pub more: Vec<MoreSegment>,
}

/// 分页搜索响应。
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: usize,
    pub page: usize,
    pub page_size: usize,
}

// ---- 权重配置 ----
// 人脸信号权重最高：人脸特征向量匹配精度高，误检率低
const WEIGHT_FACE: f32 = 0.50;
// 物体信号权重最低：物体检测标签粒度粗，歧义性大
const WEIGHT_OBJECT: f32 = 0.20;
// 场景信号居中：结合 CLIP 和 BM25 互补
const WEIGHT_SCENE: f32 = 0.30;

// 场景信号内部：BM25 文本匹配权重远高于 CLIP 向量。
// 原因：BM25 命中意味着文本直接出现关键词，精确度高；CLIP 是语义近似，召回好但精度低。
const WEIGHT_SCENE_BM25: f32 = 0.75;
const WEIGHT_SCENE_CLIP: f32 = 0.25;

// 各信号类型的最低有效分数阈值，低于此值的信号直接丢弃
const MIN_SCORE_FACE: f32 = 0.3;
const MIN_SCORE_OBJECT: f32 = 0.3;
const MIN_SCORE_SCENE: f32 = 0.25;

/// 将嵌入插件返回的类别字符串键映射为 DescriptionCategory 枚举。
/// 这些键来自索引时存储的分类向量元数据。
fn category_from_key(key: &str) -> Option<DescriptionCategory> {
    match key {
        "person" => Some(DescriptionCategory::Person),
        "foreground" => Some(DescriptionCategory::Foreground),
        "background" => Some(DescriptionCategory::Background),
        "scene" => Some(DescriptionCategory::Scene),
        "action" => Some(DescriptionCategory::Action),
        "marks" => Some(DescriptionCategory::Marks),
        _ => None,
    }
}

/// 利用预计算的类别向量，判断查询向量与哪些类别语义相关。
///
/// 核心思路：计算查询向量与每个类别代表向量的余弦相似度，然后用"间隔检测"
/// 自适应地选择相关类别——找到相邻分数间最大间隔，只保留间隔以上的类别。
/// 这比固定阈值更鲁棒，因为不同查询的相似度分布差异很大。
fn route_query_to_categories(
    query_vector: &[f32],
    category_vectors: &std::collections::HashMap<String, Vec<f32>>,
) -> Vec<DescriptionCategory> {
    let mut category_scores: Vec<(DescriptionCategory, f32)> = Vec::new();

    for (key, label_vector) in category_vectors {
        if let Some(category) = category_from_key(key) {
            let sim = cosine_similarity(query_vector, label_vector);
            tracing::debug!("Category routing: {} key='{}' similarity={:.3}", category.index_filename(), key, sim);
            category_scores.push((category, sim));
        }
    }

    // 按相似度降序排列
    category_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 间隔检测：找到排序后相邻分数的最大跳跃，只保留跳跃上方的类别。
    // 这样可以自适应不同查询的分数分布，避免固定阈值一刀切。
    let selected = if category_scores.len() <= 1 {
        category_scores
    } else {
        // 找到最大间隔的位置
        let mut max_gap_idx = 0;
        let mut max_gap = 0.0f32;
        for i in 0..category_scores.len() - 1 {
            let gap = category_scores[i].1 - category_scores[i + 1].1;
            if gap > max_gap {
                max_gap = gap;
                max_gap_idx = i;
            }
        }

        // 只有间隔足够大（>= 0.03）且确实能排除某些类别时才裁剪
        if max_gap >= 0.03 && max_gap_idx < category_scores.len() - 1 {
            // 额外保护：与最高分差距在 0.08 以内的类别也保留。
            // 避免多类别查询（如"女子拿着玻璃杯"）丢失相关类别。
            let top_score = category_scores[0].1;
            let gap_cutoff = category_scores[max_gap_idx].1;
            let soft_cutoff = top_score - 0.08;

            category_scores.iter()
                .filter(|(_, score)| *score > gap_cutoff || *score >= soft_cutoff)
                .cloned()
                .collect()
        } else {
            category_scores
        }
    };

    // 如果没有类别通过筛选，则回退到搜索全部类别，避免漏召回
    if selected.is_empty() {
        tracing::info!("No category above threshold, searching all categories");
        DescriptionCategory::all().to_vec()
    } else {
        let names: Vec<&str> = selected.iter().map(|(c, _)| c.index_filename()).collect();
        tracing::info!("Query routed to categories: {:?}", names);
        selected.iter().map(|(c, _)| *c).collect()
    }
}

/// 余弦相似度：衡量两个向量的方向一致性，值域 [-1, 1]。
/// 在本模块中广泛用于比较特征向量、类别向量等的相似程度。
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
}

/// 执行统一搜索：收集各路信号 → 融合打分 → 去重 → 分页返回。
#[allow(clippy::too_many_arguments)]
pub fn search(
    query: &str,
    search_type: SearchType,
    _top_k: usize,
    threshold: f32,
    page: usize,
    page_size: usize,
    dedup: bool,
    _settings: &Settings,
    db: &Database,
    config_db: &ConfigDatabase,
    face_index: &VectorIndex,
    scene_indices: &SceneIndices,
    image_index: &VectorIndex,
    image_path: &str,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<SearchResponse> {
    tracing::info!("Unified search: type={:?}, query={}", search_type, query);

    // 将人脸库名称加载到 jieba 分词器，确保人名不被错误拆分
    // 例如"张三丰"不应被拆为"张三"+"丰"
    if let Ok(names) = config_db.get_all_face_names() {
        db.add_jieba_words(&names);
    }

    let mut all_signals: Vec<SignalScore> = Vec::new();

    // 各路信号独立收集，任何一路失败只记录警告，不影响其他路
    // 人脸信号：查询是否匹配人脸库中的已知人物
    if matches!(search_type, SearchType::Auto | SearchType::Face) {
        match collect_face_signals(query, db, config_db, face_index) {
            Ok(signals) => all_signals.extend(signals),
            Err(e) => tracing::warn!("Face search failed: {}", e),
        }
    }

    // 物体信号：通过分词+同义词扩展匹配物体检测标签
    if matches!(search_type, SearchType::Auto | SearchType::Object) {
        match collect_object_signals(query, db) {
            Ok(signals) => all_signals.extend(signals),
            Err(e) => tracing::warn!("Object search failed: {}", e),
        }
    }

    // 场景 CLIP 向量信号：语义相似度搜索，带类别路由优化
    if matches!(search_type, SearchType::Auto | SearchType::Scene) {
        match collect_scene_signals(query, db, scene_indices, progress_cb) {
            Ok(signals) => all_signals.extend(signals),
            Err(e) => tracing::warn!("Scene search failed: {}", e),
        }
    }

    // BM25 文本信号：精确关键词匹配，与 CLIP 互补
    if matches!(search_type, SearchType::Auto | SearchType::Scene) {
        match collect_bm25_signals(query, db) {
            Ok(signals) => all_signals.extend(signals),
            Err(e) => tracing::warn!("BM25 search failed: {}", e),
        }
    }

    // 图片信号：以图搜图，用 CLIP 图像向量检索相似帧
    if matches!(search_type, SearchType::Image) {
        match collect_image_signals(image_path, db, image_index, progress_cb) {
            Ok(signals) => all_signals.extend(signals),
            Err(e) => tracing::warn!("Image search failed: {}", e),
        }
    }

    // 按片段维度融合多路信号为统一分数
    let mut results = merge_signals(all_signals, threshold);

    // 去重：同一视频只保留最佳片段作为主结果，其余归入 `more`
    if dedup {
        results = dedup_results(results);
    }

    results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    // 去重后过滤掉被归入其他结果 `more` 列表的非主结果
    if dedup {
        // 统计每个视频在结果中出现的次数
        let mut video_counts: HashMap<uuid::Uuid, usize> = HashMap::new();
        for r in &results {
            *video_counts.entry(r.video.id).or_insert(0) += 1;
        }
        // 只保留主结果（有 more）或该视频的唯一结果
        results.retain(|r| !r.more.is_empty() || video_counts.get(&r.video.id) == Some(&1));
    }

    // 分页
    let total = results.len();
    let start = (page - 1) * page_size;
    let end = (start + page_size).min(total);
    let page_results = if start < total {
        results[start..end].to_vec()
    } else {
        Vec::new()
    };

    Ok(SearchResponse {
        results: page_results,
        total,
        page,
        page_size,
    })
}

/// 收集人脸信号：在人脸向量索引中检索与查询人物最相似的检测记录。
///
/// 查询词必须是人脸库中已注册的人物名称，否则返回空。
/// 一个人名可能关联多个特征向量（不同角度/表情），取每个检测的最佳匹配分数。
fn collect_face_signals(
    query: &str,
    db: &Database,
    config_db: &ConfigDatabase,
    face_index: &VectorIndex,
) -> Result<Vec<SignalScore>> {
    // 查询必须精确匹配人脸库中的人物名
    let face_entry = match config_db.get_face_by_name(query)? {
        Some(e) => e,
        None => return Ok(Vec::new()),
    };

    // 用该人物的所有特征向量分别搜索，对每个检测 ID 保留最高分
    let mut all_index_results: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for fv in &face_entry.feature_vectors {
        let results = face_index.search(fv, 100, MIN_SCORE_FACE);
        for r in results {
            let entry = all_index_results.entry(r.id.clone());
            match entry {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if r.score > *e.get() {
                        e.insert(r.score);
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(r.score);
                }
            }
        }
    }

    // 将向量索引中的 ID 关联回数据库中的检测、片段和视频记录
    let mut signals = Vec::new();
    for (id_str, score) in all_index_results {
        let detection_id: uuid::Uuid = id_str.parse()
            .map_err(|_| VideoSceneError::IndexCorrupted("Invalid detection ID in face index".into()))?;

        let detection = match db.get_detection_by_id(&detection_id)? {
            Some(d) => d,
            None => continue,
        };
        let segment = match db.get_segment_by_id(&detection.segment_id)? {
            Some(s) => s,
            None => continue,
        };
        let video = match db.get_video_by_id(&segment.video_id)? {
            Some(v) => v,
            None => continue,
        };

        signals.push(SignalScore {
            video,
            segment,
            raw_score: score,
            signal_type: "face".to_string(),
            label: query.to_string(),
        });
    }

    Ok(signals)
}

/// 收集物体信号：对查询进行中文分词和同义词扩展后，匹配物体检测标签。
///
/// 分词后每个 token 会经过同义词扩展，直接匹配使用原始置信度，
/// 关联词匹配则折半降权，以区分精确匹配与模糊关联。
fn collect_object_signals(
    query: &str,
    db: &Database,
) -> Result<Vec<SignalScore>> {
    // 用 jieba 对中文查询分词，确保复合词被正确拆分
    use jieba_rs::Jieba;
    use std::sync::LazyLock;
    static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

    let tokens: Vec<String> = JIEBA.cut(query, false)
        .into_iter()
        .map(|s| s.to_string())
        // 过滤空白、单字（单字歧义太大）和非文本 token
        .filter(|t| !t.is_empty() && t.chars().all(|c| !c.is_whitespace()) && t.chars().count() > 1)
        .collect();

    // 收集所有同义词/关联词，用于扩大搜索范围
    let mut all_labels: Vec<String> = Vec::new();
    let mut direct_labels: Vec<String> = Vec::new();  // 仅直接同义词，用于区分精确匹配

    for token in &tokens {
        let (synonyms, related) = crate::storage::database::expand_synonyms_public(token);
        all_labels.extend(synonyms.clone());
        all_labels.extend(related.clone());
        direct_labels.extend(synonyms);
    }
    all_labels.sort();
    all_labels.dedup();

    let detections = if all_labels.is_empty() {
        // 同义词扩展无结果时退化为原始查询直接搜索
        db.get_detections_by_label(DetectionType::Object, query)?
    } else {
        db.get_detections_by_labels(&all_labels)?
    };

    let mut signals = Vec::new();
    for detection in &detections {
        if detection.confidence < MIN_SCORE_OBJECT {
            continue;
        }

        let segment = match db.get_segment_by_id(&detection.segment_id)? {
            Some(s) => s,
            None => continue,
        };
        let video = match db.get_video_by_id(&segment.video_id)? {
            Some(v) => v,
            None => continue,
        };

        // 直接同义词匹配保持原始置信度；关联词匹配降权到 50%
        let is_direct = direct_labels.iter().any(|l| detection.label.contains(l.as_str()));
        let score = if is_direct {
            detection.confidence
        } else {
            detection.confidence * 0.5  // 关联词匹配，降低分数
        };

        signals.push(SignalScore {
            video,
            segment,
            raw_score: score,
            signal_type: "object".to_string(),
            label: detection.label.clone(),
        });
    }

    Ok(signals)
}

/// 收集场景 CLIP 向量信号：将查询编码为向量后，在各类别的 HNSW 索引中搜索。
///
/// 通过类别路由（category routing）只搜索与查询语义相关的类别索引，
/// 减少无关类别的噪声干扰并提升搜索速度。
fn collect_scene_signals(
    query: &str,
    db: &Database,
    scene_indices: &SceneIndices,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<SignalScore>> {
    tracing::info!("Encoding search query with Harrier: {}", query);

    // 单次 Harrier 调用同时获取查询向量和缓存的类别向量，避免多次推理
    let (query_vector, category_vectors) =
        crate::plugins::text_vectorization::encode_text_with_categories(query, progress_cb)?;

    // 根据类别向量路由查询到相关类别，无需额外 CLIP 调用
    let target_categories = route_query_to_categories(&query_vector, &category_vectors);

    struct CategoryScore {
        weighted_score: f32,
    }
    // 每个片段可能在多个类别中被命中，收集所有加权分数
    let mut segment_category_scores: HashMap<uuid::Uuid, Vec<CategoryScore>> = HashMap::new();
    let mut segment_data: HashMap<uuid::Uuid, (Segment, Video)> = HashMap::new();

    for category in &target_categories {
        let index = &scene_indices.indices[category];
        let results = index.search(&query_vector, 100, MIN_SCORE_SCENE);
        // 不同类别的 CLIP 匹配权重不同（如人物类别权重更高）
        let weight = category.clip_weight();

        for vr in results {
            let weighted_score = vr.score * weight;
            tracing::debug!("CLIP {} segment {} raw={:.3} weighted={:.3}", category.index_filename(), vr.id, vr.score, weighted_score);
            let segment_id: uuid::Uuid = match vr.id.parse() {
                Ok(id) => id,
                Err(_) => continue,
            };

            segment_category_scores.entry(segment_id)
                .or_insert_with(Vec::new)
                .push(CategoryScore { weighted_score });

            // 延迟加载片段和视频数据，每个片段只查一次数据库
            if !segment_data.contains_key(&segment_id) {
                if let Some(segment) = db.get_segment_by_id(&segment_id)? {
                    if let Some(video) = db.get_video_by_id(&segment.video_id)? {
                        segment_data.insert(segment_id, (segment, video));
                    }
                }
            }
        }
    }

    // 每个片段取各类别中的最高加权分数作为该片段的场景分数
    let mut signals = Vec::new();
    for (segment_id, scores) in segment_category_scores {
        let score = scores.iter().map(|s| s.weighted_score).fold(0.0f32, f32::max);

        if let Some((segment, video)) = segment_data.remove(&segment_id) {
            signals.push(SignalScore {
                video,
                segment,
                raw_score: score,
                signal_type: "scene".to_string(),
                label: query.to_string(),
            });
        }
    }

    Ok(signals)
}

/// 收集 BM25 文本信号：在场景描述的全文索引上进行关键词匹配。
///
/// BM25 精确匹配关键词，与 CLIP 的语义模糊匹配互补——
/// 当用户查询特定名词时 BM25 往往更准确。
fn collect_bm25_signals(
    query: &str,
    db: &Database,
) -> Result<Vec<SignalScore>> {
    let bm25_results = db.search_descriptions_bm25(query, 100)?;
    tracing::info!("BM25 search returned {} results for '{}'", bm25_results.len(), query);

    let mut signals = Vec::new();
    for (segment_id, score) in bm25_results {
        tracing::debug!("BM25 segment {} score {:.3}", segment_id, score);
        // BM25 分数低于 0.1 的结果基本无关，直接跳过
        if score < 0.1 {
            continue;
        }

        let segment = match db.get_segment_by_id(&segment_id)? {
            Some(s) => s,
            None => continue,
        };
        let video = match db.get_video_by_id(&segment.video_id)? {
            Some(v) => v,
            None => continue,
        };

        signals.push(SignalScore {
            video,
            segment,
            raw_score: score,
            signal_type: "bm25".to_string(),
            label: query.to_string(),
        });
    }

    Ok(signals)
}

/// 收集图像信号：将查询图片编码为 CLIP 向量后在图像索引中检索。
///
/// 图像索引的 ID 格式为 "{segment_id}_frame_{n}"，需要解析出片段 ID
/// 并按片段取最高分，因为同一片段的多个帧可能都被匹配到。
fn collect_image_signals(
    image_path: &str,
    db: &Database,
    image_index: &VectorIndex,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<SignalScore>> {
    tracing::info!("Image search: encoding query image {}", image_path);

    let query_vector = crate::plugins::image_text_vectorization::encode_image(image_path, progress_cb)?;

    // 图像索引按帧存储，ID 格式为 "{seg_id}_frame_{n}"
    let raw_results = image_index.search(&query_vector, 100, MIN_SCORE_SCENE);

    // 同一片段可能有多帧命中，取每片段的最佳分数
    let mut segment_best: std::collections::HashMap<uuid::Uuid, f32> = std::collections::HashMap::new();
    for vr in &raw_results {
        // 从复合 ID 中提取片段 ID：取 "_frame_" 最后一次出现之前的部分
        let seg_id_str = vr.id.rsplitn(2, "_frame_").last().unwrap_or(&vr.id);
        if let Ok(seg_id) = seg_id_str.parse::<uuid::Uuid>() {
            let entry = segment_best.entry(seg_id);
            match entry {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if vr.score > *e.get() {
                        e.insert(vr.score);
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(vr.score);
                }
            }
        }
    }

    let mut signals = Vec::new();
    for (segment_id, score) in segment_best {
        let segment = match db.get_segment_by_id(&segment_id)? {
            Some(s) => s,
            None => continue,
        };
        let video = match db.get_video_by_id(&segment.video_id)? {
            Some(v) => v,
            None => continue,
        };

        signals.push(SignalScore {
            video,
            segment,
            raw_score: score,
            signal_type: "image".to_string(),
            label: image_path.to_string(),
        });
    }

    Ok(signals)
}

/// 将所有信号按片段维度融合为统一分数。
///
/// 核心逻辑：
/// 1. 按片段 ID 聚合信号，每个信号类型只保留最高分
/// 2. 场景信号内先融合 CLIP 与 BM25（BM25 权重高，两者一致时提升分数）
/// 3. 各信号类型按预设权重加权求和，再除以实际参与信号的权重总和做归一化
/// 4. 仅 CLIP 单信号时降权 30%，因为纯语义匹配精度不足
/// 5. 根据贡献信号数量估计置信区间（信号越多越确定）
fn merge_signals(signals: Vec<SignalScore>, threshold: f32) -> Vec<SearchResult> {
    use std::collections::HashMap;

    // 按片段 ID 分组，每个信号类型保留最高分
    struct SegmentSignals {
        video: Video,
        segment: Segment,
        face_best: Option<(f32, String)>,
        object_best: Option<(f32, String)>,
        scene_clip_best: Option<(f32, String)>,
        scene_bm25_best: Option<(f32, String)>,
        image_best: Option<(f32, String)>,
    }

    let mut per_segment: HashMap<uuid::Uuid, SegmentSignals> = HashMap::new();

    for sig in signals {
        let entry = per_segment.entry(sig.segment.id).or_insert_with(|| SegmentSignals {
            video: sig.video.clone(),
            segment: sig.segment.clone(),
            face_best: None,
            object_best: None,
            scene_clip_best: None,
            scene_bm25_best: None,
            image_best: None,
        });

        let best_slot = match sig.signal_type.as_str() {
            "face" => &mut entry.face_best,
            "object" => &mut entry.object_best,
            "scene" => &mut entry.scene_clip_best,
            "bm25" => &mut entry.scene_bm25_best,
            "image" => &mut entry.image_best,
            _ => continue,
        };

        // 同一信号类型只保留最高分
        if best_slot.as_ref().map_or(true, |(s, _)| sig.raw_score > *s) {
            *best_slot = Some((sig.raw_score, sig.label));
        }
    }

    // 对每个片段计算统一置信度
    let mut results: Vec<SearchResult> = Vec::new();

    for (_, ss) in per_segment {
        let face_raw = ss.face_best.as_ref().map(|(s, _)| *s).unwrap_or(0.0);
        let object_raw = ss.object_best.as_ref().map(|(s, _)| *s).unwrap_or(0.0);

        // ---- 场景信号内融合：CLIP（高召回）与 BM25（高精度）互补 ----
        let clip_raw = ss.scene_clip_best.as_ref().map(|(s, _)| *s).unwrap_or(0.0);
        let bm25_raw = ss.scene_bm25_best.as_ref().map(|(s, _)| *s).unwrap_or(0.0);
        let has_clip = ss.scene_clip_best.is_some();
        let has_bm25 = ss.scene_bm25_best.is_some();
        // 融合策略：两者共存时以加权平均为基底，再根据 BM25 强弱决定是否额外提升
        let scene_raw = if has_clip && has_bm25 {
            let avg = WEIGHT_SCENE_CLIP * clip_raw + WEIGHT_SCENE_BM25 * bm25_raw;
            if bm25_raw >= 0.3 {
                // BM25 强命中 + CLIP 共振：信号高度一致，取加权平均与两者各自 80% 的上限
                avg.max(clip_raw * 0.8).max(bm25_raw * 0.8)
            } else {
                // BM25 弱命中：CLIP 提供召回信号，温和提升
                avg.max(bm25_raw * 0.8)
            }
        } else if has_clip {
            clip_raw
        } else if has_bm25 {
            bm25_raw * 0.8 // BM25 单独不如 CLIP 可靠，降权 20%
        } else {
            0.0
        };
        let has_scene = has_clip || has_bm25;

        // 图片信号（CLIP 图像向量检索）
        let image_raw = ss.image_best.as_ref().map(|(s, _)| *s).unwrap_or(0.0);
        let has_image = ss.image_best.is_some();

        const WEIGHT_IMAGE: f32 = 0.40;

        // 加权求和后按实际参与信号的权重归一化，避免缺失信号拉低分数
        let face_w = if ss.face_best.is_some() { WEIGHT_FACE } else { 0.0 };
        let object_w = if ss.object_best.is_some() { WEIGHT_OBJECT } else { 0.0 };
        let scene_w = if has_scene { WEIGHT_SCENE } else { 0.0 };
        let image_w = if has_image { WEIGHT_IMAGE } else { 0.0 };
        let total_w = face_w + object_w + scene_w + image_w;

        if total_w == 0.0 {
            continue;
        }

        let unified = (face_w * face_raw + object_w * object_raw + scene_w * scene_raw + image_w * image_raw) / total_w;

        // 仅靠 CLIP 向量匹配时降权 30%：纯语义近似缺乏精确性佐证，
        // 需要降权以避免大量低相关结果排在有 BM25/人脸/物体佐证的结果前面
        let only_clip = has_clip && !has_bm25 && ss.face_best.is_none() && ss.object_best.is_none() && !has_image;
        let unified = if only_clip { unified * 0.7 } else { unified };

        if unified < threshold {
            continue;
        }

        // 置信区间：贡献信号越多，估计越确定，区间越窄
        let n_signals = ss.face_best.is_some() as i32
            + ss.object_best.is_some() as i32
            + has_scene as i32
            + has_image as i32;

        let margin = match n_signals {
            1 => 0.15,  // 单信号不确定性大
            2 => 0.08,  // 双信号较可信
            _ => 0.04,  // 三信号以上高度可信
        };

        let confidence_low = (unified - margin).max(0.0);
        let confidence_high = (unified + margin).min(1.0);

        // 构建匹配类型标签，方便前端展示信号来源
        let mut signal_parts: Vec<&str> = Vec::new();
        if ss.face_best.is_some() { signal_parts.push("face"); }
        if ss.object_best.is_some() { signal_parts.push("object"); }
        if has_clip { signal_parts.push("clip"); }
        if has_bm25 { signal_parts.push("bm25"); }
        if has_image { signal_parts.push("image"); }
        let match_type = signal_parts.join("+");

        // 构建匹配标签，展示各信号的具体内容
        let mut label_parts: Vec<String> = Vec::new();
        if let Some((_, label)) = &ss.face_best { label_parts.push(format!("face:{}", label)); }
        if let Some((_, label)) = &ss.object_best { label_parts.push(format!("obj:{}", label)); }
        if let Some((_, label)) = &ss.scene_clip_best { label_parts.push(format!("clip:{}", label)); }
        if let Some((_, label)) = &ss.scene_bm25_best { label_parts.push(format!("bm25:{}", label)); }
        if let Some((_, label)) = &ss.image_best { label_parts.push(format!("image:{}", label)); }
        let match_label = label_parts.join(", ");

        results.push(SearchResult {
            video: ss.video,
            best_segment: ss.segment,
            confidence: unified,
            confidence_low,
            confidence_high,
            match_type,
            match_label,
            more: Vec::new(),
        });
    }

    results
}

/// 按视频去重：同一视频只保留最高置信度的片段作为主结果，
/// 其余片段归入主结果的 `more` 列表。
///
/// 这样用户看到的是"每个视频一个条目"，而非同一视频重复出现多次。
fn dedup_results(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    use std::collections::HashMap;

    // 找到每个视频的最佳结果索引
    let mut best_per_video: HashMap<uuid::Uuid, usize> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        let entry = best_per_video.entry(r.video.id);
        match entry {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if r.confidence > results[*e.get()].confidence {
                    e.insert(i);
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(i);
            }
        }
    }

    // 将非主结果构建为 MoreSegment，归入对应主结果的 more 列表
    let mut more_map: HashMap<usize, Vec<MoreSegment>> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        if let Some(&best_idx) = best_per_video.get(&r.video.id) {
            // 置信度低于 0.3 的次要片段不展示，避免噪声
            if best_idx != i && r.confidence >= 0.3 {
                more_map.entry(best_idx).or_default().push(MoreSegment {
                    segment_id: r.best_segment.id,
                    start_time: r.best_segment.start_time,
                    end_time: r.best_segment.end_time,
                    confidence: r.confidence,
                    keyframe_path: r.best_segment.keyframe_path.clone(),
                });
            }
        }
    }

    // 按置信度降序排列每个视频的次要片段
    for segs in more_map.values_mut() {
        segs.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    }

    // 将 more 列表挂到主结果上
    for (i, r) in results.iter_mut().enumerate() {
        if let Some(segs) = more_map.remove(&i) {
            r.more = segs;
        }
    }

    results
}
