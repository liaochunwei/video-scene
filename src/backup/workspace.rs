//! 工作区备份与导入。

use std::collections::HashMap;
use std::fs;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use tar::Archive;

use crate::storage::ConfigDatabase;

/// 备份清单，描述归档类型和版本。
#[derive(Serialize, Deserialize)]
struct Manifest {
    r#type: String,
    version: u32,
    name: String,
}

/// 打包指定工作区到 tar.gz 归档。
pub fn backup(config_db: &ConfigDatabase, workspace_name: &Option<String>, output: &Path) -> crate::error::Result<()> {
    let active = config_db.get_active_workspace()?;
    let name = workspace_name.as_deref().unwrap_or(&active);
    let ws_path = config_db.get_workspace_path(name)?;

    let manifest = Manifest {
        r#type: "workspace".to_string(),
        version: 1,
        name: name.to_string(),
    };

    let file = fs::File::create(output)
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    let mut tar = tar::Builder::new(&mut encoder);

    // 写入 manifest.json
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    let mut header = tar::Header::new_gnu();
    header.set_size(manifest_json.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append_data(&mut header, "manifest.json", manifest_json.as_bytes())
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;

    // 写入 index.db
    let db_path = ws_path.join("index.db");
    if db_path.exists() {
        let mut data = Vec::new();
        let mut f = fs::File::open(&db_path)
            .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
        f.read_to_end(&mut data)
            .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, "index.db", data.as_slice())
            .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    }

    // 写入 keyframes/ 目录
    let keyframes_dir = ws_path.join("keyframes");
    if keyframes_dir.exists() {
        tar.append_dir_all("keyframes", &keyframes_dir)
            .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    }

    // 写入 vectors/ 目录
    let vectors_dir = ws_path.join("vectors");
    if vectors_dir.exists() {
        tar.append_dir_all("vectors", &vectors_dir)
            .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    }

    tar.finish()
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    drop(tar);
    encoder.finish()
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;

    println!("Workspace '{}' backed up → {}", name, output.display());
    Ok(())
}

/// 从 tar.gz 归档导入数据到当前工作区。
///
/// 逐条视频导入，处理源视频路径映射：
/// - 同一源目录只问一次新目录
/// - 文件不存在时提示跳过或继续
/// - 关键帧绝对路径重写为当前工作区路径
pub fn import(config_db: &ConfigDatabase, backup_path: &Path) -> crate::error::Result<()> {
    let active = config_db.get_active_workspace()?;
    let ws_path = config_db.get_workspace_path(&active)?;

    let file = fs::File::open(backup_path)
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    let tmp_dir = tempfile::tempdir()
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;

    archive.unpack(tmp_dir.path())
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;

    // 验证 manifest
    let manifest_data = fs::read_to_string(tmp_dir.path().join("manifest.json"))
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_data)
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    if manifest.r#type != "workspace" {
        return Err(crate::error::VideoSceneError::StorageError(
            format!("Not a workspace backup (type: {})", manifest.r#type)
        ));
    }

    println!("Importing workspace '{}' into '{}'...", manifest.name, active);

    // 打开备份数据库
    let backup_db_path = tmp_dir.path().join("index.db");
    if !backup_db_path.exists() {
        return Err(crate::error::VideoSceneError::StorageError(
            "Backup has no index.db".to_string()
        ));
    }
    let backup_db = crate::storage::Database::open(&backup_db_path)?;

    // 打开当前工作区数据库
    let current_db_path = ws_path.join("index.db");
    let current_db = crate::storage::Database::open(&current_db_path)?;

    // 查询备份数据库中的所有视频
    let videos = backup_db.list_videos()?;

    // 按源视频目录分组，收集路径映射
    let mut dir_mapping: HashMap<String, String> = HashMap::new();
    let mut imported = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for video in &videos {
        let original_path = PathBuf::from(&video.path);
        let parent_dir = original_path.parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        // 获取或询问新目录映射
        let new_dir = if let Some(mapped) = dir_mapping.get(&parent_dir) {
            mapped.clone()
        } else {
            println!("\n源视频目录: {}", parent_dir);
            println!("请输入对应的新目录路径: ");
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                dir_mapping.insert(parent_dir.clone(), parent_dir.clone());
                parent_dir.clone()
            } else {
                let mapped = input.trim().to_string();
                dir_mapping.insert(parent_dir.clone(), mapped.clone());
                mapped
            }
        };

        // 构建新路径
        let filename = original_path.file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        let new_path = PathBuf::from(&new_dir).join(&filename);

        // 检查文件是否存在
        if !new_path.exists() {
            println!("文件不存在: {}", new_path.display());
            println!("跳过还是继续导入？(s=跳过, c=继续): ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).ok();
            if input.trim().to_lowercase() == "s" {
                skipped += 1;
                println!("跳过: {}", filename);
                continue;
            }
        }

        // 导入视频记录
        let new_path_str = new_path.to_string_lossy().to_string();
        let old_ws_path_str = tmp_dir.path().to_string_lossy().to_string();
        let new_ws_path_str = ws_path.to_string_lossy().to_string();

        match import_video(&backup_db, &current_db, video, &new_path_str, &old_ws_path_str, &new_ws_path_str, &ws_path, tmp_dir.path()) {
            Ok(()) => {
                imported += 1;
                println!("导入: {}", filename);
            }
            Err(e) => {
                failed += 1;
                eprintln!("导入失败 {}: {}", filename, e);
            }
        }
    }

    // 重建 FTS 索引
    if let Err(e) = current_db.rebuild_fts() {
        eprintln!("Warning: FTS rebuild failed: {}", e);
    }

    println!("\n导入完成: {} 导入, {} 跳过, {} 失败 (共 {})", imported, skipped, failed, videos.len());
    Ok(())
}

/// 导入单个视频及其关联的 segments 和 detections。
fn import_video(
    backup_db: &crate::storage::Database,
    current_db: &crate::storage::Database,
    video: &crate::models::Video,
    new_video_path: &str,
    old_ws_path: &str,
    new_ws_path: &str,
    ws_path: &Path,
    tmp_dir: &Path,
) -> crate::error::Result<()> {
    // 检查是否已存在（按路径）
    if current_db.get_video_by_path(new_video_path)?.is_some() {
        println!("  已存在，跳过: {}", new_video_path);
        return Ok(());
    }

    // 插入视频记录（替换路径）
    let mut new_video = video.clone();
    new_video.path = new_video_path.to_string();
    current_db.insert_video(&new_video)?;

    // 导入 segments
    let segments = backup_db.get_segments_by_video(&video.id)?;
    for mut seg in segments {
        // 重写关键帧绝对路径
        if seg.keyframe_path.starts_with(old_ws_path) {
            seg.keyframe_path = seg.keyframe_path.replace(old_ws_path, new_ws_path);
        }
        current_db.insert_segment(&seg)?;

        // 复制关键帧图片
        let src_keyframe = tmp_dir.join(&seg.keyframe_path.replace(new_ws_path, old_ws_path));
        let dst_keyframe = ws_path.join(
            Path::new(&seg.keyframe_path).strip_prefix(new_ws_path).unwrap_or(Path::new(&seg.keyframe_path))
        );
        if src_keyframe.exists() {
            if let Some(parent) = dst_keyframe.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::copy(&src_keyframe, &dst_keyframe).ok();
        }
    }

    // 导入 detections
    let detections = backup_db.get_detections_by_video(&video.id)?;
    for det in &detections {
        current_db.insert_detection(det)?;
    }

    Ok(())
}
