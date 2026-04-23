//! 人脸库模型
//!
//! 人脸库用于存储已知人物的参考信息，是身份识别的基准数据。
//! 每个人可关联多张照片和对应的特征向量，以覆盖不同妆容、角度、光照等外观变化，
//! 从而提高识别召回率。新增照片时会进行去重判断，避免高度相似的图片浪费存储。

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 人脸库中的一条记录，代表一个已知人物。
///
/// `images` 与 `feature_vectors` 保持一一对应关系：第 i 张图片对应第 i 个特征向量。
/// 这样设计而非合并为单个结构体，是因为序列化/反序列化时两个 Vec 更方便批量处理。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLibraryEntry {
    pub id: Uuid,
    pub name: String,
    /// 参考图片的文件路径列表
    pub images: Vec<String>,
    /// 每张图片对应的特征向量，支持同一人的多种外观（不同妆容、角度等）
    pub feature_vectors: Vec<Vec<f32>>,
}

impl FaceLibraryEntry {
    /// 用一张初始照片创建人脸条目。
    pub fn new(name: String, image_path: String, feature_vector: Vec<f32>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            images: vec![image_path],
            feature_vectors: vec![feature_vector],
        }
    }

    /// 仅当新图片与已有向量差异足够大时才添加，避免高度相似的图片冗余入库。
    ///
    /// 判断依据：新向量与所有已有向量的余弦距离均不低于 `min_diff` 时才添加。
    /// - 返回 `true` 表示已添加；
    /// - 返回 `false` 表示过于相似，已跳过。
    ///
    /// `min_diff` 为最小余弦距离（1 - 余弦相似度），值越大要求差异越明显。
    pub fn add_image_if_different(&mut self, image_path: String, feature_vector: Vec<f32>, min_diff: f32) -> bool {
        // 逐一比较：若存在任一已有向量与新向量过于相似，则跳过
        let too_similar = self.feature_vectors.iter().any(|existing| {
            let sim = cosine_similarity(existing, &feature_vector);
            (1.0 - sim) < min_diff
        });

        if too_similar {
            false
        } else {
            self.images.push(image_path);
            self.feature_vectors.push(feature_vector);
            true
        }
    }

    /// 返回第一个特征向量，用于只需要一个向量的旧接口兼容。
    /// 若向量列表为空则返回空切片，避免 panic。
    pub fn primary_feature_vector(&self) -> &[f32] {
        self.feature_vectors.first().map_or(&[], |v| v)
    }
}

/// 计算两个向量的余弦相似度。
///
/// 结果范围 [-1, 1]，值越接近 1 表示方向越一致。
/// 维度不匹配或零向量时返回 0.0，表示无法比较。
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // 维度不同或为空向量，无意义，直接返回 0
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    // 任一向量为零向量时无法计算方向，返回 0
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
