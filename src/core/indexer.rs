//! 视频索引编排模块
//!
//! 负责驱动单视频文件或整个目录的索引流程。根据是否强制重建（force）来决定
//! 跳过已有索引还是删除旧索引后重新处理。处理完成后自动更新耗时统计，
//! 为后续处理提供剩余时间估算。

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::config::Settings;
use crate::core::timing;
use crate::error::Result;
use crate::plugins::ProgressMessage;
use crate::storage::{Database, FileStore, SceneIndices, VectorIndex};

/// 单个视频的索引结果，区分跳过和成功完成。
#[derive(Debug)]
pub enum IndexOutcome {
    /// 视频已有索引，被跳过
    Skipped,
    /// 视频成功完成索引
    Indexed,
}

/// 目录索引的汇总结果。
#[derive(Debug)]
pub struct DirectorySummary {
    pub total: usize,
    pub indexed: usize,
    pub skipped: usize,
    pub failed: usize,
    pub interrupted: bool,
}

impl DirectorySummary {
    fn from_results(results: &[Result<IndexOutcome>], total: usize, interrupted: bool) -> Self {
        let mut indexed = 0;
        let mut skipped = 0;
        let mut failed = 0;
        for r in results {
            match r {
                Ok(IndexOutcome::Indexed) => indexed += 1,
                Ok(IndexOutcome::Skipped) => skipped += 1,
                Err(_) => failed += 1,
            }
        }
        Self { total, indexed, skipped, failed, interrupted }
    }
}

impl std::fmt::Display for DirectorySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.interrupted {
            write!(f, "Interrupted: indexed {}, skipped {}, failed {} (processed {}/{})",
                self.indexed, self.skipped, self.failed, self.indexed + self.skipped + self.failed, self.total)
        } else {
            write!(f, "Done: indexed {}, skipped {}, failed {} (total {})",
                self.indexed, self.skipped, self.failed, self.total)
        }
    }
}

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
) -> Result<IndexOutcome> {
    if !force {
        if (db.get_video_by_path(&video_path.to_string_lossy())?).is_some() {
            tracing::info!("Already indexed, skipping: {}", video_path.display());
            return Ok(IndexOutcome::Skipped);
        }
    } else if let Some(existing) = db.get_video_by_path(&video_path.to_string_lossy())? {
        db.delete_video(&existing.id)?;
    }

    tracing::info!("Indexing: {}", video_path.display());
    let start = Instant::now();
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

    let index_dir = file_store.base_dir();
    let _ = timing::update_timing(index_dir, elapsed, result.segments.len());

    Ok(IndexOutcome::Indexed)
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
) -> Result<IndexOutcome> {
    if !force {
        if (db.get_video_by_path(&video_path.to_string_lossy())?).is_some() {
            tracing::info!("Already indexed, skipping: {}", video_path.display());
            return Ok(IndexOutcome::Skipped);
        }
    } else if let Some(existing) = db.get_video_by_path(&video_path.to_string_lossy())? {
        db.delete_video(&existing.id)?;
    }

    tracing::info!("Indexing (video mode): {}", video_path.display());
    let start = Instant::now();
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

    let index_dir = file_store.base_dir();
    let _ = timing::update_timing_vlm(index_dir, elapsed, result.video.duration as f64);

    Ok(IndexOutcome::Indexed)
}

/// 索引目录下所有视频文件。
///
/// 遍历目录收集视频文件，复用调用者传入的数据库和向量索引连接逐个处理。
/// 串行处理，单连接复用避免了每个视频重复 open + init_schema 的开销。
/// 支持 Ctrl+C 优雅中断：收到信号后停止处理新视频，打印已完成统计。
#[allow(clippy::too_many_arguments)]
pub fn index_directory(
    dir_path: &Path,
    settings: &Settings,
    db: &Database,
    file_store: &FileStore,
    face_index: &mut VectorIndex,
    scene_indices: &mut SceneIndices,
    image_index: &mut VectorIndex,
    recursive: bool,
    extensions: &[String],
    force: bool,
    video_mode: bool,
    progress_cb: &dyn Fn(ProgressMessage),
) -> Result<DirectorySummary> {
    let video_files = collect_video_files(dir_path, recursive, extensions);
    let total = video_files.len();

    if total == 0 {
        tracing::info!("No video files found in: {}", dir_path.display());
        return Ok(DirectorySummary { total: 0, indexed: 0, skipped: 0, failed: 0, interrupted: false });
    }

    tracing::info!("Found {} video files to index", total);

    // 注册 Ctrl+C 处理：收到信号后设置中断标志，当前视频处理完后停止
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_ref = cancelled.clone();
    ctrlc::set_handler(move || {
        if !cancelled_ref.load(Ordering::Relaxed) {
            eprintln!("\nInterrupted, finishing current video...");
            cancelled_ref.store(true, Ordering::Relaxed);
        }
    }).unwrap_or_else(|e| tracing::warn!("Failed to set Ctrl+C handler: {}", e));

    let mut results: Vec<Result<IndexOutcome>> = Vec::with_capacity(total);
    for (i, path) in video_files.iter().enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }

        tracing::info!("Indexing [{}/{}]: {}", i + 1, total, path.display());
        let r = if video_mode {
            index_video_vlm_api(path, settings, db, file_store, face_index, scene_indices, image_index, force, progress_cb)
        } else {
            index_video(path, settings, db, file_store, face_index, scene_indices, image_index, force, progress_cb)
        };
        results.push(r);
    }

    let interrupted = cancelled.load(Ordering::Relaxed);
    let summary = DirectorySummary::from_results(&results, total, interrupted);
    if interrupted {
        tracing::warn!("{}", summary);
    } else {
        tracing::info!("{}", summary);
    }

    Ok(summary)
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
