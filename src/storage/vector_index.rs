//! 向量相似度搜索索引：基于暴力余弦相似度的向量检索。
//!
//! 为什么用暴力搜索而不是 HNSW 等近似最近邻算法？
//! - 当前数据规模（千级向量）下暴力搜索的延迟完全可接受
//! - 暴力搜索实现简单，不依赖外部库，保证结果精确无遗漏
//! - 向量数据以 JSON 序列化存储在磁盘上，加载到内存后全量扫描
//!
//! 文件扩展名 .hnsw 是历史遗留命名，实际实现并非 HNSW 算法。

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, VideoSceneError};
use crate::plugins::image_text_understanding::DescriptionCategory;

/// 暴力余弦相似度向量索引，将所有向量保存在内存中。
pub struct VectorIndex {
    entries: Vec<VectorEntry>,
    path: std::path::PathBuf,
}

/// 单条向量记录：ID + 对应的浮点向量。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f32>,
}

/// 搜索结果：匹配的向量 ID 和相似度分数。
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

impl VectorIndex {
    /// 从磁盘加载向量索引，如果文件不存在则创建空索引。
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        }

        let entries = if path.exists() {
            let data = std::fs::read(path)
                .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
            // 反序列化失败时返回空列表而非报错，容忍损坏的索引文件
            serde_json::from_slice(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(Self {
            entries,
            path: path.to_path_buf(),
        })
    }

    /// 添加或更新一条向量。若 ID 已存在则覆盖，保证同一 ID 只有一份数据。
    pub fn add(&mut self, id: String, vector: Vec<f32>) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.vector = vector;
        } else {
            self.entries.push(VectorEntry { id, vector });
        }
    }

    /// 按 ID 移除一条向量。
    pub fn remove(&mut self, id: &str) {
        self.entries.retain(|e| e.id != id);
    }

    /// 搜索与查询向量最相似的 top_k 条结果，过滤掉低于阈值的结果。
    ///
    /// 流程：计算所有向量与查询的余弦相似度 -> 过滤低分 -> 按分数降序排序 -> 截取 top_k
    pub fn search(&self, query: &[f32], top_k: usize, threshold: f32) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self
            .entries
            .iter()
            .map(|entry| {
                let score = cosine_similarity(query, &entry.vector);
                SearchResult {
                    id: entry.id.clone(),
                    score,
                }
            })
            .filter(|r| r.score >= threshold)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// 将索引数据序列化写入磁盘。
    pub fn save(&self) -> Result<()> {
        let data = serde_json::to_vec(&self.entries)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        std::fs::write(&self.path, data)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> impl Iterator<Item = &VectorEntry> {
        self.entries.iter()
    }
}

/// 计算两个向量的余弦相似度。
///
/// 结果范围 [-1, 1]，值越大表示方向越一致。
/// 维度不匹配或零向量返回 0.0 作为安全兜底。
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// 按场景描述类别分组的向量索引集合。
///
/// 场景描述分为"人/前景物/背景物/场景/动作/标识"等类别，
/// 每个类别有独立的向量索引文件。这样在搜索时可以按类别限定范围，
/// 避免不同类别的向量互相干扰（如人物向量与背景向量混在一起会降低搜索精度）。
pub struct SceneIndices {
    pub indices: HashMap<DescriptionCategory, VectorIndex>,
}

impl SceneIndices {
    /// 为每个描述类别打开对应的向量索引文件。
    pub fn open(base_dir: &Path) -> Result<Self> {
        let mut indices = HashMap::new();
        for cat in DescriptionCategory::all() {
            let path = base_dir.join(cat.index_filename());
            indices.insert(*cat, VectorIndex::open(&path)?);
        }
        Ok(Self { indices })
    }

    /// 保存所有类别的索引到磁盘。
    pub fn save(&self) -> Result<()> {
        for index in self.indices.values() {
            index.save()?;
        }
        Ok(())
    }
}
