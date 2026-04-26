//! 人脸库备份与导入。

use std::fs;
use std::path::Path;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use tar::Archive;

use crate::storage::ConfigDatabase;

/// 备份清单，描述归档类型和版本。
#[derive(Serialize, Deserialize)]
struct Manifest {
    r#type: String,
    version: u32,
}

/// 人脸条目的导出格式，包含特征向量。
#[derive(Serialize, Deserialize)]
struct FaceExport {
    id: String,
    name: String,
    images: Vec<String>,
    feature_vectors: Vec<Vec<f32>>,
}

/// 打包人脸库到 tar.gz 归档。
pub fn backup(config_db: &ConfigDatabase, config_dir: &Path, output: &Path) -> crate::error::Result<()> {
    let faces = config_db.list_faces()?;

    let exports: Vec<FaceExport> = faces
        .iter()
        .map(|f| FaceExport {
            id: f.id.to_string(),
            name: f.name.clone(),
            images: f.images.clone(),
            feature_vectors: f.feature_vectors.clone(),
        })
        .collect();

    let manifest = Manifest {
        r#type: "face".to_string(),
        version: 1,
    };

    let file = fs::File::create(output)
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    let mut tar = tar::Builder::new(&mut encoder);

    let pb = ProgressBar::new(exports.len() as u64);
    pb.set_style(ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}"
    ).unwrap());

    // 写入 manifest.json
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    let mut header = tar::Header::new_gnu();
    header.set_size(manifest_json.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append_data(&mut header, "manifest.json", manifest_json.as_bytes())
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;

    // 写入 face_library.json
    let faces_json = serde_json::to_string_pretty(&exports)
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    let mut header = tar::Header::new_gnu();
    header.set_size(faces_json.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append_data(&mut header, "face_library.json", faces_json.as_bytes())
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;

    // 写入人脸图片目录
    let face_dir = config_dir.join("face_library");
    if face_dir.exists() {
        pb.set_message("packing images");
        tar.append_dir_all("face_library", &face_dir)
            .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    }

    pb.set_message("finalizing");
    tar.finish()
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    drop(tar);
    encoder.finish()
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;

    pb.finish_with_message(format!("backed up {} faces → {}", exports.len(), output.display()));
    Ok(())
}

/// 从 tar.gz 归档增量导入人脸库。
///
/// 去重逻辑：通过特征向量余弦距离判断（add_image_if_different），不看图片文件名。
/// 图片复制：已有人名时，新图片从已有最大 index+1 开始编号，绝不覆盖。
pub fn import(config_db: &mut ConfigDatabase, config_dir: &Path, backup_path: &Path) -> crate::error::Result<()> {
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
    if manifest.r#type != "face" {
        return Err(crate::error::VideoSceneError::StorageError(
            format!("Not a face backup (type: {})", manifest.r#type)
        ));
    }

    // 读取导出的人脸数据
    let faces_data = fs::read_to_string(tmp_dir.path().join("face_library.json"))
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    let exports: Vec<FaceExport> = serde_json::from_str(&faces_data)
        .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;

    let existing_faces = config_db.list_faces()?;
    let mut added = 0;
    let mut merged = 0;

    let pb = ProgressBar::new(exports.len() as u64);
    pb.set_style(ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}"
    ).unwrap());

    for export in &exports {
        pb.set_message(export.name.clone());
        let existing = existing_faces.iter().find(|f| f.name == export.name);

        if let Some(existing_entry) = existing {
            // 同名人物：用 add_image_if_different 增量追加（通过特征向量去重）
            let mut updated = existing_entry.clone();
            let mut new_count = 0;

            // 找到已有图片的最大 index，新图片从 max_index+1 开始
            let max_index = existing_entry.images.iter().filter_map(|img| {
                let fname = Path::new(img).file_stem()?.to_str()?;
                fname.strip_prefix("face_")?.parse::<usize>().ok()
            }).max().unwrap_or(0);

            for (i, vector) in export.feature_vectors.iter().enumerate() {
                let new_index = max_index + new_count + 1;
                let new_image_path = format!("face_library/{}/face_{}.jpg", export.name, new_index);

                if updated.add_image_if_different(
                    new_image_path.clone(), vector.clone(), 0.1,
                ) {
                    new_count += 1;

                    // 复制对应的人脸图片
                    let src_img = tmp_dir.path().join(&export.images.get(i).unwrap_or(&String::new()));
                    let dst_img = config_dir.join(&new_image_path);
                    if src_img.exists() {
                        if let Some(parent) = dst_img.parent() {
                            fs::create_dir_all(parent).ok();
                        }
                        fs::copy(&src_img, &dst_img).ok();
                    }
                }
            }

            if new_count > 0 {
                config_db.update_face(&updated)?;
                merged += 1;
                pb.println(format!("Merged {} new appearances into '{}'", new_count, export.name));
            }
        } else {
            // 新人物：直接插入
            let entry = crate::models::face_library::FaceLibraryEntry {
                id: export.id.parse().unwrap_or_else(|_| uuid::Uuid::new_v4()),
                name: export.name.clone(),
                images: export.images.clone(),
                feature_vectors: export.feature_vectors.clone(),
            };
            config_db.insert_face(&entry)?;

            // 复制人脸图片
            for img in &export.images {
                let src = tmp_dir.path().join(img);
                let dst = config_dir.join(img);
                if src.exists() {
                    if let Some(parent) = dst.parent() {
                        fs::create_dir_all(parent).ok();
                    }
                    fs::copy(&src, &dst).ok();
                }
            }

            added += 1;
            pb.println(format!("Added new face '{}' ({} images)", export.name, export.images.len()));
        }
        pb.inc(1);
    }

    pb.finish_with_message(format!("{} added, {} merged", added, merged));
    Ok(())
}
