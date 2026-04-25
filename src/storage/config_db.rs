//! 全局配置数据库：管理工作区注册表和人脸库。
//!
//! 此数据库位于 config_dir 下（而非工作区目录下），因为它存储的是跨工作区共享的数据：
//! - 工作区注册表：记录所有工作区的名称和路径，用于工作区切换
//! - 人脸库：已注册的人脸及其特征向量，所有工作区的人脸识别共用
//!
//! 人脸特征向量以 JSON 序列化的 BLOB 存储，读出时需要兼容新旧两种格式：
//! - 旧格式：Vec<f32>（单张人脸的单个向量）
//! - 新格式：Vec<Vec<f32>>（同一人的多张人脸的多个向量）

use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use uuid::Uuid;

use crate::error::{Result, VideoSceneError};
use crate::models::FaceLibraryEntry;

/// 配置库表结构：工作区表、人脸库表、键值元数据表。
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_library (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    images TEXT NOT NULL,
    feature_vector BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_library_name ON face_library(name);

CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
";

/// 解析人脸特征向量 BLOB，兼容新旧两种序列化格式。
///
/// 旧版本存储的是 Vec<f32>（一个人只有一个向量），新版本改为 Vec<Vec<f32>>
/// （支持同一人多张照片的多个向量）。先尝试新格式，失败则回退旧格式并包装为
/// 单元素 Vec，确保上层代码始终拿到 Vec<Vec<f32>>。
fn parse_feature_vectors(bytes: &[u8]) -> Vec<Vec<f32>> {
    // 优先尝试新格式：Vec<Vec<f32>>
    if let Ok(vectors) = serde_json::from_slice::<Vec<Vec<f32>>>(bytes) {
        // 校验：如果第一个元素是非空向量，说明确实是新格式
        if vectors.first().map_or(false, |v| !v.is_empty()) {
            return vectors;
        }
    }
    // 回退到旧格式：Vec<f32> -> 包装成单元素 Vec
    if let Ok(single) = serde_json::from_slice::<Vec<f32>>(bytes) {
        if !single.is_empty() {
            return vec![single];
        }
    }
    Vec::new()
}

/// 工作区注册条目。
#[derive(Debug, Clone)]
pub struct WorkspaceEntry {
    pub id: Uuid,
    pub name: String,
    pub path: String,
    pub created_at: String,
}

/// 全局配置数据库，管理工作区注册表和人脸库。
pub struct ConfigDatabase {
    conn: Connection,
}

impl ConfigDatabase {
    /// 打开（或创建）配置数据库。
    ///
    /// 启用 WAL 模式以提高并发读性能，然后创建表结构。
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        }
        let conn = Connection::open(path)
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        conn.execute_batch(SCHEMA)
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(Self { conn })
    }

    /// 暴露底层连接，供迁移代码直接操作数据库。
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    // --- 工作区注册表 ---

    /// 列出所有已注册的工作区。
    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceEntry>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, path, created_at FROM workspaces")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let workspaces = stmt
            .query_map([], |row| {
                Ok(WorkspaceEntry {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    name: row.get(1)?,
                    path: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(workspaces)
    }

    /// 按名称查找工作区，用于切换工作区时验证目标是否存在。
    pub fn get_workspace_by_name(&self, name: &str) -> Result<Option<WorkspaceEntry>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, path, created_at FROM workspaces WHERE name = ?1")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let result = stmt
            .query_row(params![name], |row| {
                Ok(WorkspaceEntry {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    name: row.get(1)?,
                    path: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })
            .optional()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(result)
    }

    /// 插入一条工作区记录。
    pub fn insert_workspace(&self, entry: &WorkspaceEntry) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO workspaces (id, name, path, created_at) VALUES (?1, ?2, ?3, ?4)",
                params![entry.id.to_string(), entry.name, entry.path, entry.created_at],
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    /// 便捷方法：按名称和路径插入工作区，自动生成 UUID 和创建时间。
    pub fn insert_workspace_with_defaults(&self, name: &str, path: &str) -> Result<()> {
        let entry = WorkspaceEntry {
            id: Uuid::new_v4(),
            name: name.to_string(),
            path: path.to_string(),
            created_at: chrono::Local::now().naive_utc().to_string(),
        };
        self.insert_workspace(&entry)
    }

    // --- 人脸库 ---

    /// 注册新人脸。特征向量以 JSON BLOB 存储，images 路径列表以 JSON 字符串存储。
    pub fn insert_face(&self, entry: &FaceLibraryEntry) -> Result<()> {
        let vector_bytes = serde_json::to_vec(&entry.feature_vectors)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        let images_json = serde_json::to_string(&entry.images)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        self.conn
            .execute(
                "INSERT INTO face_library (id, name, images, feature_vector) VALUES (?1, ?2, ?3, ?4)",
                params![entry.id.to_string(), entry.name, images_json, vector_bytes],
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    /// 按名称查找人脸，读出时自动处理新旧特征向量格式。
    pub fn get_face_by_name(&self, name: &str) -> Result<Option<FaceLibraryEntry>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, images, feature_vector FROM face_library WHERE name = ?1")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let result = stmt
            .query_row(params![name], |row| {
                let images: String = row.get(2)?;
                let vector_bytes: Vec<u8> = row.get(3)?;
                Ok(FaceLibraryEntry {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    name: row.get(1)?,
                    images: serde_json::from_str(&images).unwrap_or_default(),
                    feature_vectors: parse_feature_vectors(&vector_bytes),
                })
            })
            .optional()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(result)
    }

    /// 列出所有已注册人脸。
    pub fn list_faces(&self) -> Result<Vec<FaceLibraryEntry>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, images, feature_vector FROM face_library")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let faces = stmt
            .query_map([], |row| {
                let images: String = row.get(2)?;
                let vector_bytes: Vec<u8> = row.get(3)?;
                Ok(FaceLibraryEntry {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    name: row.get(1)?,
                    images: serde_json::from_str(&images).unwrap_or_default(),
                    feature_vectors: parse_feature_vectors(&vector_bytes),
                })
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(faces)
    }

    /// 按名称删除人脸。
    pub fn delete_face(&self, name: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM face_library WHERE name = ?1", params![name])
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    /// 更新人脸的图片路径和特征向量。
    pub fn update_face(&self, entry: &FaceLibraryEntry) -> Result<()> {
        let vector_bytes = serde_json::to_vec(&entry.feature_vectors)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        let images_json = serde_json::to_string(&entry.images)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        self.conn
            .execute(
                "UPDATE face_library SET images = ?1, feature_vector = ?2 WHERE name = ?3",
                params![images_json, vector_bytes, entry.name],
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    /// 获取所有人脸名称，用于加载 jieba 自定义词（确保人脸名能被正确分词）。
    pub fn get_all_face_names(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name FROM face_library")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        let names = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        names
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))
    }

    /// 获取当前激活的工作区名称。
    pub fn get_active_workspace(&self) -> Result<String> {
        let config_dir = std::path::PathBuf::from(self.conn.path().unwrap_or(""))
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        let state = crate::config::StateFile::load(&config_dir);
        Ok(state.active_workspace)
    }

    /// 获取指定工作区的路径。
    pub fn get_workspace_path(&self, name: &str) -> Result<std::path::PathBuf> {
        let entry = self.get_workspace_by_name(name)?
            .ok_or_else(|| VideoSceneError::PluginConfigError(format!("Workspace '{}' not found", name)))?;
        Ok(std::path::PathBuf::from(&entry.path))
    }
}
