//! 数据库迁移：从旧版单数据库布局迁移到工作区多数据库布局。
//!
//! 旧版本只有一个 index.db 文件，既存储视频/片段数据又存储人脸库。
//! 新版本将全局数据（人脸库、工作区注册表）拆分到 config.db，
//! 视频数据留在各工作区的 index.db 中。
//!
//! 迁移触发条件：config_dir 下存在旧 index.db 但不存在 config.db。
//! 这意味着迁移只执行一次——一旦 config.db 创建成功，后续启动不会重复迁移。

use std::path::PathBuf;
use crate::error::{Result, VideoSceneError};

/// 检查是否需要从旧布局迁移，如需要则执行迁移。
///
/// 迁移步骤：
/// 1. 创建 config.db 并初始化表结构
/// 2. 从旧 index.db 读取 face_library 数据，写入 config.db
/// 3. 从旧 index.db 删除 face_library 表（该表现在归 config.db 管理）
/// 4. 在 config.db 注册 "default" 工作区，指向原 config_dir
/// 5. 创建 state.json 标记当前活跃工作区为 "default"
pub fn migrate_if_needed(config_dir: &PathBuf) -> Result<()> {
    let config_db_path = config_dir.join("config.db");
    let old_index_db_path = config_dir.join("index.db");

    // 已迁移过（config.db 已存在），无需操作
    if config_db_path.exists() {
        return Ok(());
    }

    // 没有旧数据可迁移
    if !old_index_db_path.exists() {
        return Ok(());
    }

    tracing::info!("Migrating from old layout to workspace layout...");

    // 步骤 1：创建 config.db
    let config_db = crate::storage::ConfigDatabase::open(&config_db_path)?;

    // 步骤 2：将旧 index.db 中的 face_library 数据复制到 config.db
    // 以只读模式打开旧库，避免误修改
    let old_conn = rusqlite::Connection::open_with_flags(
        &old_index_db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ).map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

    let face_count: i64 = old_conn
        .query_row("SELECT COUNT(*) FROM face_library", [], |row| row.get(0))
        .unwrap_or(0);

    if face_count > 0 {
        let mut stmt = old_conn
            .prepare("SELECT id, name, images, feature_vector FROM face_library")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        let entries: Vec<(String, String, String, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            }).map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .filter_map(|e| e.ok())
            .collect();

        // 使用 INSERT OR IGNORE 防止重复插入（如果迁移中途失败后重试）
        let count = entries.len();
        for (id, name, images, feature_vector) in entries {
            config_db.conn().execute(
                "INSERT OR IGNORE INTO face_library (id, name, images, feature_vector) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![id, name, images, feature_vector],
            ).map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        }
        tracing::info!("Migrated {} face library entries", count);
    }

    // 步骤 3：从旧 index.db 删除 face_library 表
    // 必须先关闭只读连接，再用读写连接执行 DDL
    drop(old_conn);
    let write_conn = rusqlite::Connection::open(&old_index_db_path)
        .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
    write_conn.execute_batch("DROP TABLE IF EXISTS face_library; DROP INDEX IF EXISTS idx_face_library_name;")
        .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
    drop(write_conn);

    // 步骤 4：注册默认工作区，路径指向原 config_dir（旧 index.db 所在位置）
    let default_path = config_dir.to_string_lossy().to_string();
    config_db.insert_workspace_with_defaults("default", &default_path)?;

    // 步骤 5：创建 state.json，标记当前活跃工作区为 "default"
    let state = crate::config::StateFile::default();
    state.save(config_dir).map_err(|e| VideoSceneError::StorageError(e.to_string()))?;

    tracing::info!("Migration complete: default workspace -> {}", default_path);
    Ok(())
}
