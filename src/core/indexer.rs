//! 视频索引编排模块
//!
//! 负责驱动单视频文件或整个目录的索引流程。根据是否强制重建（force）来决定
//! 跳过已有索引还是删除旧索引后重新处理。处理完成后自动更新耗时统计，
//! 为后续处理提供剩余时间估算。

use std::path::Path;
use std::time::Instant;

use crate::config::Settings;
use crate::core::timing;
use crate::error::Result;
use crate::plugins::ProgressMessage;
use crate::storage::{Database, FileStore, SceneIndices, VectorIndex};

/// 索引单个视频文件（本地管线模式）。
///
/// 流程：检查/删除已有索引 → 调用管线处理 → 更新耗时统计。
/// `force` 为 true 时先删除旧索引再重建，否则跳过已索引的视频。
#[allow(clippy::too_many_arguments)]
pub fn index_video(
    video_path: &Path,
    settings: &Settings,
    db: &Database,
    file_store: &FileStore,
    face_index: &mut VectorIndex,
    scene_indices: &mut SceneIndices,
    image_index: &mut VectorIndex,
    force: bool,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<()> {
    // 根据 force 标志决定：跳过已索引的视频，或删除旧索引以便重建
    if !force {
        if (db.get_video_by_path(&video_path.to_string_lossy())?).is_some() {
            tracing::info!("Already indexed, skipping: {}", video_path.display());
            return Ok(());
        }
    } else if let Some(existing) = db.get_video_by_path(&video_path.to_string_lossy())? {
        db.delete_video(&existing.id)?;
    }

    tracing::info!("Indexing: {}", video_path.display());
    let start = Instant::now();
    // 委托给管线模块执行实际的视频分析、场景检测、AI 推理和入库
    let result = crate::core::pipeline::process_video(
        video_path, settings, db, file_store, face_index, scene_indices, image_index, progress_cb,
    )?;
    let elapsed = start.elapsed().as_secs_f64();

    tracing::info!(
        "Done: {} ({} segments, {} detections) in {:.1}s",
        result.video.filename,
        result.segments.len(),
        result.detections.len(),
        elapsed
    );

    // 用本次处理耗时更新统计模型，供后续估算剩余时间
    let index_dir = file_store.base_dir();
    let _ = timing::update_timing(index_dir, elapsed, result.segments.len());

    Ok(())
}

/// 索引单个视频文件（云端 VLM API 模式）。
///
/// 与 `index_video` 类似，但走云端视频理解 API 进行场景切分和描述，
/// 耗时统计按"处理时间/视频时长"记录，而非按片段数。
#[allow(clippy::too_many_arguments)]
pub fn index_video_vlm_api(
    video_path: &Path,
    settings: &Settings,
    db: &Database,
    file_store: &FileStore,
    face_index: &mut VectorIndex,
    scene_indices: &mut SceneIndices,
    image_index: &mut VectorIndex,
    force: bool,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<()> {
    // 同上：force 决定跳过还是删除重建
    if !force {
        if (db.get_video_by_path(&video_path.to_string_lossy())?).is_some() {
            tracing::info!("Already indexed, skipping: {}", video_path.display());
            return Ok(());
        }
    } else if let Some(existing) = db.get_video_by_path(&video_path.to_string_lossy())? {
        db.delete_video(&existing.id)?;
    }

    tracing::info!("Indexing (video mode): {}", video_path.display());
    let start = Instant::now();
    // 委托给 VLM API 管线，由云端模型完成场景切分与描述
    let result = crate::core::pipeline::process_video_vlm_api(
        video_path, settings, db, file_store, face_index, scene_indices, image_index, progress_cb,
    )?;
    let elapsed = start.elapsed().as_secs_f64();

    tracing::info!(
        "Done: {} ({} segments, {} detections) in {:.1}s",
        result.video.filename,
        result.segments.len(),
        result.detections.len(),
        elapsed
    );

    // VLM 模式下按视频总时长统计，因为处理耗时与视频长度线性相关
    let index_dir = file_store.base_dir();
    let _ = timing::update_timing_vlm(index_dir, elapsed, result.video.duration as f64);

    Ok(())
}

/// 索引目录下所有视频文件。
///
/// 遍历目录收集视频文件，为每个文件打开独立的数据库和向量索引连接后逐个处理。
/// 目前串行处理（`_parallel` 参数预留但未启用），每个视频独占一套存储连接以保证线程安全。
#[allow(clippy::too_many_arguments)]
pub fn index_directory(
    dir_path: &Path,
    settings: &Settings,
    _db: &Database,
    file_store: &FileStore,
    recursive: bool,
    extensions: &[String],
    _parallel: usize,
    force: bool,
    video_mode: bool,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<Vec<Result<()>>> {
    let video_files = collect_video_files(dir_path, recursive, extensions);
    let total = video_files.len();

    if total == 0 {
        tracing::info!("No video files found in: {}", dir_path.display());
        return Ok(Vec::new());
    }

    tracing::info!("Found {} video files to index", total);

    // 每个视频独立打开存储连接，避免跨视频共享可变状态
    let workspace_path = file_store.base_dir();
    let results: Vec<Result<()>> = video_files
        .iter()
        .enumerate()
        .map(|(i, path)| {
            tracing::info!("Indexing [{}/{}]: {}", i + 1, total, path.display());

            let db_path = workspace_path.join("index.db");
            let db = Database::open(&db_path)?;
            let face_index_path = workspace_path.join("vectors").join("faces.hnsw");
            let vectors_dir = workspace_path.join("vectors");
            let mut face_index = VectorIndex::open(&face_index_path)?;
            let mut scene_indices = SceneIndices::open(&vectors_dir)?;
            let mut image_index = VectorIndex::open(&vectors_dir.join("images.hnsw"))?;

            // 根据 video_mode 选择本地管线或云端 VLM API 模式
            if video_mode {
                index_video_vlm_api(path, settings, &db, file_store, &mut face_index, &mut scene_indices, &mut image_index, force, progress_cb)
            } else {
                index_video(path, settings, &db, file_store, &mut face_index, &mut scene_indices, &mut image_index, force, progress_cb)
            }
        })
        .collect();

    let success_count = results.iter().filter(|r| r.is_ok()).count();
    tracing::info!("Indexed {}/{} videos successfully", success_count, total);

    Ok(results)
}

/// 递归收集目录下指定扩展名的视频文件，返回排序后的路径列表。
fn collect_video_files(dir: &Path, recursive: bool, extensions: &[String]) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && recursive {
                files.extend(collect_video_files(&path, recursive, extensions));
            } else if path.is_file() {
                if let Some(ext) = path.extension() {
                    if extensions.contains(&ext.to_string_lossy().to_string()) {
                        files.push(path);
                    }
                }
            }
        }
    }
    // 排序保证不同运行间文件处理顺序一致
    files.sort();
    files
}
