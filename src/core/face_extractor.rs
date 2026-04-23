//! 人脸提取与聚类模块
//!
//! 从视频中提取人脸并按身份聚类。流程为：
//! 1. 以固定间隔（1秒）抽帧，确保覆盖视频中所有人脸出场时刻
//! 2. 批量检测帧中人脸，过滤低质量结果（模糊、侧脸等）
//! 3. 使用单链接层次聚类将相似人脸归为同一人物
//! 4. 每个聚类选质量最高的人脸裁剪保存，作为该人物的代表图像
//!
//! 聚类算法采用单链接（single-linkage）策略：只要新脸与聚类中任意一张脸
//! 相似度超过阈值即可加入，适合处理同一人不同角度/表情的分裂问题。

use crate::config::Settings;
use crate::error::Result;
use crate::models::{BoundingBox, FaceLibraryEntry};
use crate::plugins::ProgressMessage;
use crate::preprocess::{video_analyzer, extract_frames};
use crate::storage::FileStore;

/// 从视频中提取出的单个人物信息。
/// 每个实例对应一个聚类（即一个可能的人物身份）。
#[derive(Debug, Clone)]
pub struct ExtractedPerson {
    /// 最佳人脸裁剪图路径（相对于 FileStore 根目录）
    pub best_frame_path: String,
    /// 最佳人脸出现的时间戳
    pub best_timestamp: f32,
    /// 人脸特征向量，用于后续人脸搜索
    pub feature_vector: Vec<f32>,
    /// 该人物在视频中的出现次数（出现越多说明越可能是主角）
    pub appearance_count: usize,
    /// 最佳人脸的质量分数
    pub quality: f32,
}

/// 从视频中提取并聚类人脸。
///
/// `min_confidence`：人脸检测置信度下限，过滤误检
/// `min_quality`：人脸质量下限，过滤模糊/侧脸
/// `cluster_threshold`：聚类相似度阈值，越高则同一人越不容易被拆分，但也可能把不同人合并
pub fn extract_faces(
    video_path: &str,
    min_confidence: f64,
    min_quality: f64,
    cluster_threshold: f64,
    settings: &Settings,
    file_store: &FileStore,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<ExtractedPerson>> {
    let path = std::path::Path::new(video_path);
    tracing::info!("Extracting faces from: {}", video_path);
    let info = video_analyzer::analyze_video(path)?;
    tracing::info!("Video duration: {:.1}s", info.duration);

    // 以 1 秒间隔均匀抽帧，兼顾覆盖率和性能
    // 人脸提取需要更密集的采样（相比管线中按场景抽帧），以确保捕捉短暂露脸
    let interval = 1.0;
    let mut timestamps = Vec::new();
    let mut t = 0.0;
    while t < info.duration {
        timestamps.push(t);
        t += interval;
    }

    let temp_dir = std::env::temp_dir().join("video-scene-face-extract");
    let frames = extract_frames(
        path,
        &timestamps,
        &temp_dir,
        settings.video.preprocessing.target_short_edge,
        settings.video.preprocessing.frame_quality,
    )?;

    // 批量检测人脸，优先批量调用以减少模型加载开销
    tracing::info!("Detecting faces in {} frames...", frames.len());
    progress_cb(ProgressMessage {
        id: String::new(),
        message: "Detecting faces".to_string(),
        current: 0,
        total: frames.len(),
    });
    let frame_paths: Vec<String> = frames.iter()
        .map(|f| f.path.to_string_lossy().to_string())
        .collect();
    let frame_timestamps: Vec<f32> = frames.iter().map(|f| f.timestamp).collect();

    // 收集所有通过质量过滤的人脸：(时间戳, 特征向量, 质量, 帧路径, 边界框)
    let mut all_faces: Vec<(f32, Vec<f32>, f32, String, BoundingBox)> = Vec::new();

    match crate::plugins::face::detect_faces_batch(&frame_paths, min_confidence, progress_cb) {
        Ok(batch_results) => {
            for (i, (_path, faces)) in batch_results.iter().enumerate() {
                let timestamp = frame_timestamps.get(i).copied().unwrap_or(0.0);
                let frame_path = frame_paths.get(i).cloned().unwrap_or_default();
                tracing::debug!("Frame {}: detected {} faces", i, faces.len());
                for face in faces {
                    if face.quality >= min_quality as f32 {
                        all_faces.push((timestamp, face.feature.clone(), face.quality, frame_path.clone(), face.bbox.clone()));
                    } else {
                        tracing::debug!("Frame {}: face quality {:.3} < min_quality {:.3}, filtered", i, face.quality, min_quality);
                    }
                }
            }
        }
        Err(e) => {
            // 批量失败时退化为逐帧检测，保证流程不中断
            tracing::warn!("Batch face detection failed: {}, falling back to per-frame detection", e);
            for (i, frame) in frames.iter().enumerate() {
                let frame_path = frame.path.to_string_lossy().to_string();
                if i % 10 == 0 {
                    tracing::info!("Detecting faces: {}/{}", i + 1, frames.len());
                }
                match crate::plugins::face::detect_faces(&frame_path, min_confidence, progress_cb) {
                    Ok(faces) => {
                        tracing::debug!("Frame {}: detected {} faces (before quality filter)", frame_path, faces.len());
                        for face in &faces {
                            if face.quality >= min_quality as f32 {
                                all_faces.push((frame.timestamp, face.feature.clone(), face.quality, frame_path.clone(), face.bbox.clone()));
                            } else {
                                tracing::debug!("Frame {}: face quality {:.3} < min_quality {:.3}, filtered", frame_path, face.quality, min_quality);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Face detection failed for frame {}: {}", frame_path, e);
                    }
                }
            }
        }
    }

    // 按特征向量相似度聚类，将同一个人的多张脸归为一组
    tracing::info!("Clustering {} faces...", all_faces.len());
    let clusters = cluster_faces(&all_faces, cluster_threshold as f32);
    tracing::info!("Found {} unique persons", clusters.len());

    // 每个聚类选质量最高的人脸作为代表，裁剪保存到 FileStore
    let mut persons = Vec::new();
    for (cluster_idx, cluster) in clusters.iter().enumerate() {
        if cluster.is_empty() {
            continue;
        }

        // 在聚类中找质量最高的人脸
        let best = cluster.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        let (_, _, _, best_frame, bbox) = best;

        // 用 ffmpeg 裁剪人脸区域并保存到 FileStore
        let rel_path = crop_face_to_store(&best_frame, bbox, file_store, cluster_idx)
            .unwrap_or_default();

        persons.push(ExtractedPerson {
            best_frame_path: rel_path,
            best_timestamp: best.0,
            feature_vector: best.1.clone(),
            appearance_count: cluster.len(),
            quality: best.2,
        });
    }

    // 按出现次数降序排列，出场最多的人物排最前（通常是主角）
    persons.sort_by(|a, b| b.appearance_count.cmp(&a.appearance_count));

    Ok(persons)
}

/// 从帧图像中裁剪人脸区域并保存到 FileStore。
///
/// 使用 ffmpeg 的 crop 滤镜按边界框裁剪，先写到临时文件再通过 FileStore 持久化。
/// 返回相对于 FileStore 根目录的路径。
fn crop_face_to_store(image_path: &str, bbox: &BoundingBox, file_store: &FileStore, index: usize) -> Option<String> {
    use std::process::Command;
    // 先裁剪到临时文件，再通过 FileStore 保存
    let temp_path = std::env::temp_dir().join(format!("vs-face-crop-{}.jpg", uuid::Uuid::new_v4()));
    let output = Command::new("ffmpeg")
        .args([
            "-y", "-i", image_path,
            "-vf", &format!("crop={}:{}:{}:{}", bbox.width as i32, bbox.height as i32, bbox.x as i32, bbox.y as i32),
            "-q:v", "2",
            &temp_path.to_string_lossy(),
        ])
        .output()
        .ok()?;
    if !output.status.success() || !temp_path.exists() {
        return None;
    }
    let image_data = std::fs::read(&temp_path).ok()?;
    let _ = std::fs::remove_file(&temp_path);
    // 保存到 face_library/_unknown/ 目录，后续命名时会移到以人名命名的子目录
    let saved_path = file_store.save_face_image("_unknown", &image_data, index).ok()?;
    // 返回相对路径，便于前端访问和后续重命名
    saved_path.strip_prefix(file_store.base_dir())
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

/// 单链接层次聚类：将特征向量相似度超过阈值的人脸归为同一聚类。
///
/// 算法流程：遍历所有人脸，对每个未分配的人脸创建新聚类，
/// 然后反复扩展聚类——只要未分配人脸与聚类中任一人脸的相似度超过阈值，
/// 就将其加入聚类并继续扫描，直到聚类不再增长。
///
/// 单链接策略的优缺点：
/// - 优点：能把同一人不同角度/表情的脸串起来（传递性）
/// - 缺点：可能产生链式效应，将两个不同的人通过中间脸连在一起
/// 实践中通过合理的阈值（默认 0.6）可以在两者间取得平衡。
fn cluster_faces(faces: &[(f32, Vec<f32>, f32, String, BoundingBox)], threshold: f32) -> Vec<Vec<(f32, Vec<f32>, f32, String, BoundingBox)>> {
    let mut clusters: Vec<Vec<(f32, Vec<f32>, f32, String, BoundingBox)>> = Vec::new();
    let mut assigned = vec![false; faces.len()];

    for i in 0..faces.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster = vec![faces[i].clone()];
        assigned[i] = true;

        // 反复扩展聚类直到没有新成员加入
        let mut changed = true;
        while changed {
            changed = false;
            for j in 0..faces.len() {
                if assigned[j] {
                    continue;
                }
                // 单链接：只要与聚类中任一成员相似度超阈值即可加入
                let max_sim = cluster.iter()
                    .map(|c| cosine_similarity(&c.1, &faces[j].1))
                    .fold(0.0f32, f32::max);
                if max_sim >= threshold {
                    cluster.push(faces[j].clone());
                    assigned[j] = true;
                    changed = true;
                }
            }
        }

        clusters.push(cluster);
    }

    clusters
}

/// 计算两个向量的余弦相似度。
///
/// 值域 [-1, 1]，在本模块中用于衡量两张人脸特征向量的相似程度。
/// 值越接近 1 表示两张脸越可能是同一个人。
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// 将提取的人物保存到人脸库，供后续按人名搜索。
///
/// 保存时将人脸裁剪图从 `_unknown/` 临时目录移到以人名命名的目录，
/// 并在配置数据库中写入特征向量，使搜索时能按名字检索。
pub fn save_person_to_library(
    person: &ExtractedPerson,
    name: &str,
    config_db: &crate::storage::ConfigDatabase,
    file_store: &FileStore,
) -> Result<()> {
    // 将人脸图像从 _unknown/ 移动到 {name}/ 目录
    let image_path = if !person.best_frame_path.is_empty() {
        move_face_image_to_name(&person.best_frame_path, name, file_store)
            .unwrap_or_else(|| person.best_frame_path.clone())
    } else {
        person.best_frame_path.clone()
    };

    let entry = FaceLibraryEntry::new(
        name.to_string(),
        image_path,
        person.feature_vector.clone(),
    );
    config_db.insert_face(&entry)?;
    Ok(())
}

/// 将人脸图像从 `_unknown/` 目录移动到 `{name}/` 目录。
///
/// 实际是读-写-删操作而非文件系统 move，因为 FileStore 的 save_face_image
/// 会按目标目录和索引生成路径。返回移动后的相对路径。
pub fn move_face_image_to_name(rel_path: &str, name: &str, file_store: &FileStore) -> Option<String> {
    let src = file_store.base_dir().join(rel_path);
    if !src.exists() {
        return None;
    }
    let image_data = std::fs::read(&src).ok()?;
    // 从文件名 face_N.jpg 中提取索引号，保持编号一致性
    let index: usize = src.file_name()
        .and_then(|f| f.to_str())
        .and_then(|s| s.strip_prefix("face_"))
        .and_then(|s| s.strip_suffix(".jpg"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let dest = file_store.save_face_image(name, &image_data, index).ok()?;
    // 清理源文件
    let _ = std::fs::remove_file(&src);
    dest.strip_prefix(file_store.base_dir())
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}
