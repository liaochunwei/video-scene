//! 文件存储：管理关键帧图片和人脸图片的磁盘存储。
//!
//! 采用简单的目录结构组织文件：
//! - keyframes/{video_id}/{segment_id}.jpg  — 每个视频片段的关键帧
//! - face_library/{person_name}/face_{index}.jpg  — 每个人的人脸样本照片
//!
//! 不使用对象存储或特殊格式，直接存储为 JPEG 文件，
//! 这样可以通过文件系统直接浏览和调试，也方便外部工具访问。

use std::path::{Path, PathBuf};

use crate::error::{Result, VideoSceneError};

/// 文件存储管理器，以 base_dir 为根目录存储图片文件。
pub struct FileStore {
    base_dir: PathBuf,
}

impl FileStore {
    /// 创建文件存储管理器，确保根目录存在。
    pub fn new(base_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(base_dir)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        Ok(Self {
            base_dir: base_dir.to_path_buf(),
        })
    }

    /// 保存视频片段的关键帧图片。
    ///
    /// 目录结构：keyframes/{video_id}/{segment_id}.jpg
    /// 每个视频单独一个子目录，避免大量文件平铺在同一目录下影响性能。
    pub fn save_keyframe(&self, video_id: &str, segment_id: &str, image_data: &[u8]) -> Result<PathBuf> {
        let dir = self.base_dir.join("keyframes").join(video_id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        let path = dir.join(format!("{}.jpg", segment_id));
        std::fs::write(&path, image_data)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        Ok(path)
    }

    /// 保存人脸库中的人脸图片。
    ///
    /// 目录结构：face_library/{person_name}/face_{index}.jpg
    /// 同一人的多张照片通过 index 区分，支持注册多张样本人脸以提高识别准确率。
    pub fn save_face_image(&self, person_name: &str, image_data: &[u8], index: usize) -> Result<PathBuf> {
        let dir = self.base_dir.join("face_library").join(person_name);
        std::fs::create_dir_all(&dir)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        let path = dir.join(format!("face_{}.jpg", index));
        std::fs::write(&path, image_data)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        Ok(path)
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}
