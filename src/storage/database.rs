//! 工作区级 SQLite 数据库：管理视频、片段、检测记录，并提供基于 BM25 的中文场景描述搜索。
//!
//! 核心设计决策：
//! - 使用 WAL 模式保证读写并发安全（索引写入时不阻塞搜索）
//! - 外键级联删除确保删除视频时自动清理关联的片段和检测
//! - FTS5 使用 trigram 分词器，因为中文没有空格分隔词语，
//!   trigram 可以支持子串匹配（如搜"口红"能命中"涂了口红"）
//! - BM25 搜索用 jieba 分词 + 同义词扩展来弥补 trigram 无法理解语义的不足

use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use uuid::Uuid;

use crate::error::{Result, VideoSceneError};

/// 全局 jieba 分词器实例，BM25 搜索和自定义词添加（如人脸名）共用。
/// 用 Mutex 包裹是因为 add_word 需要 &mut self。
static JIEBA: std::sync::LazyLock<std::sync::Mutex<jieba_rs::Jieba>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(jieba_rs::Jieba::new()));

/// 记录已添加到 jieba 的自定义词，避免重复添加。
static JIEBA_EXTRA_WORDS: std::sync::LazyLock<std::sync::Mutex<Vec<String>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

/// 尝试用多种常见日期格式解析字符串，全部失败则回退到当前时间。
/// 这是因为不同版本写入的日期格式可能不一致，需要容错处理。
fn parse_datetime(s: &str) -> chrono::NaiveDateTime {
    for fmt in &[
        "%Y-%m-%d %H:%M:%S%.6f",
        "%Y-%m-%d %H:%M:%S%.3f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.6f",
        "%Y-%m-%dT%H:%M:%S",
    ] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return dt;
        }
    }
    chrono::Local::now().naive_utc()
}
use crate::models::{Detection, DetectionType, Segment, Video};

/// 数据库表结构定义。
/// segments 通过外键级联关联 videos，detections 通过外键级联关联 segments，
/// 这样删除视频时 SQLite 会自动清理下游数据，避免孤立记录。
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    duration REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id),
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    keyframe_path TEXT NOT NULL,
    scene_vector BLOB,
    scene_description TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL REFERENCES segments(id),
    detection_type TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x REAL,
    bbox_y REAL,
    bbox_w REAL,
    bbox_h REAL,
    feature_vector BLOB,
    FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_detections_segment_id ON detections(segment_id);
CREATE INDEX IF NOT EXISTS idx_detections_type_label ON detections(detection_type, label);
";

pub struct Database {
    conn: Connection,
}

impl Database {
    /// 打开（或创建）工作区数据库。
    ///
    /// 启用 WAL 模式以支持读写并发，启用外键约束以触发级联删除，
    /// 然后初始化表结构和 FTS 索引。
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        }
        let conn = Connection::open(path)
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// 初始化表结构并处理历史版本的兼容迁移。
    ///
    /// 除了创建基础表，还做了两件事：
    /// 1. 检查 segments 表是否缺少 scene_description 列（老版本没有此列），
    ///    缺少则补加。
    /// 2. 创建基于 trigram 分词的 FTS5 虚拟表。如果 FTS 索引为空但
    ///    segments 已有描述内容，则重建索引——这处理了从旧版本升级时
    ///    FTS 表尚未创建的场景。
    fn init_schema(&self) -> Result<()> {
        self.conn
            .execute_batch(SCHEMA)
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        // 兼容老版本：scene_description 列是后加的
        let has_desc: bool = self.conn
            .prepare("SELECT scene_description FROM segments LIMIT 0")
            .is_ok();
        if !has_desc {
            self.conn
                .execute_batch("ALTER TABLE segments ADD COLUMN scene_description TEXT NOT NULL DEFAULT '';")
                .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        }
        // FTS5 虚拟表使用 trigram 分词器，对中日韩文字的子串匹配友好
        self.conn
            .execute_batch(
                "CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                    scene_description,
                    content='segments',
                    content_rowid='rowid',
                    tokenize='trigram'
                );"
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        // 如果 FTS 索引为空但已有描述数据，说明是从旧版本升级过来的，需要重建索引
        let fts_count: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM segments_fts", [], |r| r.get(0))
            .unwrap_or(0);
        let desc_count: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM segments WHERE scene_description != ''", [], |r| r.get(0))
            .unwrap_or(0);
        if fts_count == 0 && desc_count > 0 {
            self.conn
                .execute_batch("INSERT INTO segments_fts(rowid, scene_description) SELECT rowid, scene_description FROM segments WHERE scene_description != '';")
                .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        }
        Ok(())
    }

    // --- 视频 CRUD ---

    pub fn insert_video(&self, video: &Video) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO videos (id, path, filename, duration, width, height, created_at, indexed_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    video.id.to_string(),
                    video.path,
                    video.filename,
                    video.duration,
                    video.width,
                    video.height,
                    video.created_at.to_string(),
                    video.indexed_at.to_string(),
                ],
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub fn get_video_by_path(&self, path: &str) -> Result<Option<Video>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, path, filename, duration, width, height, created_at, indexed_at FROM videos WHERE path = ?1")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let result = stmt
            .query_row(params![path], |row| {
                Ok(Video {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    path: row.get(1)?,
                    filename: row.get(2)?,
                    duration: row.get(3)?,
                    width: row.get(4)?,
                    height: row.get(5)?,
                    created_at: parse_datetime(&row.get::<_, String>(6)?),
                    indexed_at: parse_datetime(&row.get::<_, String>(7)?),
                })
            })
            .optional()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(result)
    }

    pub fn list_videos(&self) -> Result<Vec<Video>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, path, filename, duration, width, height, created_at, indexed_at FROM videos")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let videos = stmt
            .query_map([], |row| {
                Ok(Video {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    path: row.get(1)?,
                    filename: row.get(2)?,
                    duration: row.get(3)?,
                    width: row.get(4)?,
                    height: row.get(5)?,
                    created_at: parse_datetime(&row.get::<_, String>(6)?),
                    indexed_at: parse_datetime(&row.get::<_, String>(7)?),
                })
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(videos)
    }

    pub fn get_video_by_id(&self, id: &Uuid) -> Result<Option<Video>> {
        let result = self
            .conn
            .query_row(
                "SELECT id, path, filename, duration, width, height, created_at, indexed_at FROM videos WHERE id = ?1",
                params![id.to_string()],
                |row| {
                    Ok(Video {
                        id: row.get::<_, String>(0)?.parse().unwrap(),
                        path: row.get(1)?,
                        filename: row.get(2)?,
                        duration: row.get(3)?,
                        width: row.get(4)?,
                        height: row.get(5)?,
                        created_at: parse_datetime(&row.get::<_, String>(6)?),
                        indexed_at: parse_datetime(&row.get::<_, String>(7)?),
                    })
                },
            )
            .ok();
        Ok(result)
    }

    pub fn delete_video(&self, id: &Uuid) -> Result<()> {
        self.conn
            .execute("DELETE FROM videos WHERE id = ?1", params![id.to_string()])
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    /// 获取视频下所有片段 ID，用于删除视频前清理向量索引。
    /// 因为向量索引存储在 SQLite 之外，无法通过级联删除自动清理，
    /// 必须先查出 ID 再逐个移除。
    pub fn get_segment_ids_by_video(&self, video_id: &Uuid) -> Result<Vec<Uuid>> {
        let mut stmt = self.conn
            .prepare("SELECT id FROM segments WHERE video_id = ?1")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        let ids: Vec<Uuid> = stmt
            .query_map(params![video_id.to_string()], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|s| s.parse().ok())
            .collect();
        Ok(ids)
    }

    pub fn video_count(&self) -> Result<u64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM videos", [], |row| row.get(0))
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))
    }

    /// 获取指定视频的所有检测记录，用于导入。
    pub fn get_detections_by_video(&self, video_id: &Uuid) -> Result<Vec<Detection>> {
        let mut stmt = self.conn
            .prepare(
                "SELECT d.id, d.segment_id, d.detection_type, d.label, d.confidence, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.feature_vector \
                 FROM detections d \
                 JOIN segments s ON d.segment_id = s.id \
                 WHERE s.video_id = ?1"
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let detections = stmt
            .query_map(params![video_id.to_string()], |row| {
                let vector_bytes: Vec<u8> = row.get(9)?;
                let feature_vector: Vec<f32> = serde_json::from_slice(&vector_bytes).unwrap_or_default();
                let bbox_x: Option<f32> = row.get(5)?;
                let bbox_y: Option<f32> = row.get(6)?;
                let bbox_w: Option<f32> = row.get(7)?;
                let bbox_h: Option<f32> = row.get(8)?;
                let bounding_box = if bbox_x.is_some() {
                    Some(crate::models::BoundingBox {
                        x: bbox_x.unwrap(),
                        y: bbox_y.unwrap(),
                        width: bbox_w.unwrap(),
                        height: bbox_h.unwrap(),
                    })
                } else {
                    None
                };
                Ok(Detection {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    segment_id: row.get::<_, String>(1)?.parse().unwrap(),
                    detection_type: if row.get::<_, String>(2)? == "face" {
                        DetectionType::Face
                    } else {
                        DetectionType::Object
                    },
                    label: row.get(3)?,
                    confidence: row.get(4)?,
                    bounding_box,
                    feature_vector,
                })
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(detections)
    }

    /// 重建 FTS 索引，用于导入数据后更新全文检索。
    pub fn rebuild_fts(&self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM segments_fts; INSERT INTO segments_fts(rowid, scene_description) SELECT rowid, scene_description FROM segments WHERE scene_description != '';")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    // --- 片段 CRUD ---

    pub fn insert_segment(&self, segment: &Segment) -> Result<()> {
        // 场景向量序列化为 JSON BLOB 存储，保持浮点精度
        let vector_bytes = serde_json::to_vec(&segment.scene_vector)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        self.conn
            .execute(
                "INSERT INTO segments (id, video_id, start_time, end_time, keyframe_path, scene_vector, scene_description)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    segment.id.to_string(),
                    segment.video_id.to_string(),
                    segment.start_time,
                    segment.end_time,
                    segment.keyframe_path,
                    vector_bytes,
                    segment.scene_description,
                ],
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        // 手动同步 FTS 索引：插入片段后若描述非空，立即写入 FTS 表
        // 这里不用触发器是因为需要控制写入时机，且 FTS 是 content 表模式
        if !segment.scene_description.is_empty() {
            let rowid: i64 = self.conn
                .query_row(
                    "SELECT rowid FROM segments WHERE id = ?1",
                    params![segment.id.to_string()],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if rowid > 0 {
                let _ = self.conn.execute(
                    "INSERT INTO segments_fts(rowid, scene_description) VALUES (?1, ?2)",
                    params![rowid, segment.scene_description],
                );
            }
        }
        Ok(())
    }

    pub fn get_segments_by_video(&self, video_id: &Uuid) -> Result<Vec<Segment>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, video_id, start_time, end_time, keyframe_path, scene_vector, scene_description FROM segments WHERE video_id = ?1")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let segments = stmt
            .query_map(params![video_id.to_string()], |row| {
                let vector_bytes: Vec<u8> = row.get(5)?;
                let scene_vector: Vec<f32> = serde_json::from_slice(&vector_bytes).unwrap_or_default();
                let scene_description: String = row.get(6)?;
                Ok(Segment {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    video_id: row.get::<_, String>(1)?.parse().unwrap(),
                    start_time: row.get(2)?,
                    end_time: row.get(3)?,
                    keyframe_path: row.get(4)?,
                    scene_vector,
                    scene_description,
                })
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(segments)
    }

    pub fn segment_count(&self) -> Result<u64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM segments", [], |row| row.get(0))
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))
    }

    // --- 检测 CRUD ---

    pub fn insert_detection(&self, detection: &Detection) -> Result<()> {
        // 特征向量同样以 JSON BLOB 存储
        let vector_bytes = serde_json::to_vec(&detection.feature_vector)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        let det_type = match detection.detection_type {
            DetectionType::Face => "face",
            DetectionType::Object => "object",
        };
        let bbox = detection.bounding_box.as_ref();
        self.conn
            .execute(
                "INSERT INTO detections (id, segment_id, detection_type, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, feature_vector)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    detection.id.to_string(),
                    detection.segment_id.to_string(),
                    det_type,
                    detection.label,
                    detection.confidence,
                    bbox.map(|b| b.x),
                    bbox.map(|b| b.y),
                    bbox.map(|b| b.width),
                    bbox.map(|b| b.height),
                    vector_bytes,
                ],
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub fn get_detection_by_id(&self, id: &Uuid) -> Result<Option<Detection>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, segment_id, detection_type, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, feature_vector FROM detections WHERE id = ?1")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let result = stmt
            .query_row(params![id.to_string()], |row| {
                let vector_bytes: Vec<u8> = row.get(9)?;
                let feature_vector: Vec<f32> = serde_json::from_slice(&vector_bytes).unwrap_or_default();
                let bbox_x: Option<f32> = row.get(5)?;
                let bbox_y: Option<f32> = row.get(6)?;
                let bbox_w: Option<f32> = row.get(7)?;
                let bbox_h: Option<f32> = row.get(8)?;
                // 四个坐标分量要么全有要么全无，以此判断是否有边界框
                let bounding_box = if bbox_x.is_some() {
                    Some(crate::models::BoundingBox {
                        x: bbox_x.unwrap(),
                        y: bbox_y.unwrap(),
                        width: bbox_w.unwrap(),
                        height: bbox_h.unwrap(),
                    })
                } else {
                    None
                };
                Ok(Detection {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    segment_id: row.get::<_, String>(1)?.parse().unwrap(),
                    detection_type: if row.get::<_, String>(2)? == "face" {
                        DetectionType::Face
                    } else {
                        DetectionType::Object
                    },
                    label: row.get(3)?,
                    confidence: row.get(4)?,
                    bounding_box,
                    feature_vector,
                })
            })
            .ok();

        Ok(result)
    }

    pub fn get_segment_by_id(&self, id: &Uuid) -> Result<Option<Segment>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, video_id, start_time, end_time, keyframe_path, scene_vector, scene_description FROM segments WHERE id = ?1")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let result = stmt
            .query_row(params![id.to_string()], |row| {
                let vector_bytes: Vec<u8> = row.get(5)?;
                let scene_vector: Vec<f32> = serde_json::from_slice(&vector_bytes).unwrap_or_default();
                let scene_description: String = row.get(6)?;
                Ok(Segment {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    video_id: row.get::<_, String>(1)?.parse().unwrap(),
                    start_time: row.get(2)?,
                    end_time: row.get(3)?,
                    keyframe_path: row.get(4)?,
                    scene_vector,
                    scene_description,
                })
            })
            .ok();

        Ok(result)
    }

    /// 按检测类型和标签模糊查询检测记录。
    /// 使用 LIKE 模糊匹配是因为同一个语义可能有多种表述（如"猫"和"猫咪"）。
    pub fn get_detections_by_label(&self, detection_type: DetectionType, label: &str) -> Result<Vec<Detection>> {
        let det_type = match detection_type {
            DetectionType::Face => "face",
            DetectionType::Object => "object",
        };
        let mut stmt = self
            .conn
            .prepare("SELECT id, segment_id, detection_type, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, feature_vector FROM detections WHERE detection_type = ?1 AND label LIKE ?2 ORDER BY confidence DESC")
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let detections = stmt
            .query_map(params![det_type, format!("%{}%", label)], |row| {
                let vector_bytes: Vec<u8> = row.get(9)?;
                let feature_vector: Vec<f32> = serde_json::from_slice(&vector_bytes).unwrap_or_default();
                let bbox_x: Option<f32> = row.get(5)?;
                let bbox_y: Option<f32> = row.get(6)?;
                let bbox_w: Option<f32> = row.get(7)?;
                let bbox_h: Option<f32> = row.get(8)?;
                let bounding_box = if bbox_x.is_some() {
                    Some(crate::models::BoundingBox {
                        x: bbox_x.unwrap(),
                        y: bbox_y.unwrap(),
                        width: bbox_w.unwrap(),
                        height: bbox_h.unwrap(),
                    })
                } else {
                    None
                };
                Ok(Detection {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    segment_id: row.get::<_, String>(1)?.parse().unwrap(),
                    detection_type: if row.get::<_, String>(2)? == "face" {
                        DetectionType::Face
                    } else {
                        DetectionType::Object
                    },
                    label: row.get(3)?,
                    confidence: row.get(4)?,
                    bounding_box,
                    feature_vector,
                })
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(detections)
    }

    /// 批量按标签查询物体检测，支持同义词扩展后的多标签搜索。
    ///
    /// 与 `get_detections_by_label` 不同，此方法接受一组标签（通常是同义词扩展后的结果），
    /// 用 OR 组合 SQL 条件，一次查询返回所有匹配结果，避免多次数据库往返。
    /// 动态拼接 SQL 是因为 rusqlite 不支持变长参数列表的参数化绑定。
    pub fn get_detections_by_labels(&self, labels: &[String]) -> Result<Vec<Detection>> {
        if labels.is_empty() {
            return Ok(Vec::new());
        }
        // 动态构建 WHERE 子句：label LIKE '%l1%' OR label LIKE '%l2%' ...
        let conditions: Vec<String> = labels.iter()
            .map(|l| format!("label LIKE '%{}%'", l.replace('\'', "''")))
            .collect();
        let where_clause = conditions.join(" OR ");
        let sql = format!(
            "SELECT id, segment_id, detection_type, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, feature_vector \
             FROM detections WHERE detection_type = 'object' AND ({}) \
             ORDER BY confidence DESC",
            where_clause
        );

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let detections = stmt
            .query_map([], |row| {
                let vector_bytes: Vec<u8> = row.get(9)?;
                let feature_vector: Vec<f32> = serde_json::from_slice(&vector_bytes).unwrap_or_default();
                let bbox_x: Option<f32> = row.get(5)?;
                let bbox_y: Option<f32> = row.get(6)?;
                let bbox_w: Option<f32> = row.get(7)?;
                let bbox_h: Option<f32> = row.get(8)?;
                let bounding_box = if bbox_x.is_some() {
                    Some(crate::models::BoundingBox {
                        x: bbox_x.unwrap(),
                        y: bbox_y.unwrap(),
                        width: bbox_w.unwrap(),
                        height: bbox_h.unwrap(),
                    })
                } else {
                    None
                };
                Ok(Detection {
                    id: row.get::<_, String>(0)?.parse().unwrap(),
                    segment_id: row.get::<_, String>(1)?.parse().unwrap(),
                    detection_type: DetectionType::Object,
                    label: row.get(3)?,
                    confidence: row.get(4)?,
                    bounding_box,
                    feature_vector,
                })
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        Ok(detections)
    }

    // --- FTS5 / BM25 搜索 ---

    /// 基于 BM25 思路的场景描述搜索：用 jieba 分词、同义词扩展、类别加权、近邻加分。
    ///
    /// 为什么不用 SQLite 原生 FTS5 BM25？
    /// 因为场景描述是结构化文本（分 "人/前景物/背景物/场/动作/标识" 等类别段落），
    /// 不同类别的匹配权重不同（人物匹配比背景匹配更重要），而原生 BM25 无法区分。
    /// 此外，同义词扩展（如"衣服"也能匹配"上衣"）也超出了 FTS5 的能力范围。
    ///
    /// 评分流程：
    /// 1. jieba 分词 + 过滤停用词
    /// 2. 逐 token 在描述中查找直接匹配 / 同义词匹配 / 相关词匹配
    /// 3. 根据匹配位置所在的类别段落赋予不同权重
    /// 4. 多 token 匹配时，若在同一类别段落内位置接近则额外加分
    /// 5. 最终分数归一化到 [0, 1]
    pub fn search_descriptions_bm25(&self, query: &str, limit: usize) -> Result<Vec<(Uuid, f32)>> {
        let jieba = JIEBA.lock().unwrap();
        // jieba 分词后过滤：去除空白、单字（信息量太低）、纯空白 token
        let tokens: Vec<String> = jieba.cut(query, false)
            .into_iter()
            .map(|s| s.to_string())
            .filter(|t| !t.is_empty() && t.chars().all(|c| !c.is_whitespace()) && t.chars().count() > 1)
            .collect();
        drop(jieba); // 尽早释放锁，避免长时间持有
        tracing::info!("BM25 jieba tokens for '{}': {:?}", query, tokens);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // 加载所有有描述的片段，在内存中逐个评分
        // 对于当前数据规模（千级片段），全量扫描比 SQL 复杂查询更可控
        let mut stmt = self.conn
            .prepare(
                "SELECT id, scene_description FROM segments WHERE scene_description != ''"
            )
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?;

        let segments: Vec<(String, String)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| VideoSceneError::DatabaseError(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        let n_original = tokens.len() as f32;
        let mut scored: Vec<(Uuid, f32)> = Vec::new();

        for (id_str, desc) in segments {
            // 场景描述格式："人：...\n前景物：...\n背景物：...\n场：...\n动作：...\n标识：..."
            // 不同类别的匹配权重不同，与 CLIP 类别权重对齐
            let category_sections = [
                ("人：", 1.0),       // 人物匹配最重要
                ("前景物：", 0.8),   // 前景次之
                ("标识：", 0.6),     // 标识（水印/logo）有一定参考价值
                ("场：", 0.5),       // 场景环境
                ("动作：", 0.4),     // 动作描述
                ("背景物：", 0.3),   // 背景最不重要
            ];
            // 将描述文本按类别段落边界切分，用于后续按位置确定类别权重
            let mut section_boundaries: Vec<(usize, f32)> = Vec::new();
            for (prefix, weight) in &category_sections {
                if let Some(pos) = desc.find(prefix) {
                    section_boundaries.push((pos, *weight));
                }
            }
            section_boundaries.sort_by_key(|(pos, _)| *pos);

            let mut direct_hits = 0.0_f32;    // 直接匹配的加权分数
            let mut synonym_hits = 0.0_f32;   // 同义词匹配的加权分数
            let mut related_hits = 0.0_f32;   // 相关词匹配的加权分数
            let mut matched_token_count = 0usize; // 有多少个不同的查询词命中了
            // 记录每个匹配 token 在描述中的字节位置，用于计算近邻加分
            let mut match_positions: Vec<usize> = Vec::new();

            // 根据字节位置判断其所属类别段落，返回对应权重
            let category_weight_at = |pos: usize| -> f32 {
                let mut w = 1.0; // 在任何类别标题之前的部分使用默认权重
                for (boundary, weight) in &section_boundaries {
                    if *boundary <= pos {
                        w = *weight;
                    } else {
                        break;
                    }
                }
                w
            };

            for token in &tokens {
                let (synonyms, related_terms) = expand_synonyms(token);
                // 优先检查直接匹配
                if let Some(pos) = desc.find(token.as_str()) {
                    let cat_w = category_weight_at(pos);
                    direct_hits += cat_w;
                    match_positions.push(pos);
                    matched_token_count += 1;
                } else {
                    // 未直接命中则检查同义词（近似可互换词）
                    // 跳过单字同义词，因为单字容易产生误匹配（如"衫"误匹配"高领衫"）
                    let mut found_syn = false;
                    for syn in &synonyms {
                        if syn.as_str() != token.as_str() && syn.chars().count() > 1 {
                            if let Some(pos) = desc.find(syn.as_str()) {
                                let cat_w = category_weight_at(pos);
                                synonym_hits += 0.3 * cat_w; // 同义词降权到 30%
                                match_positions.push(pos);
                                found_syn = true;
                                matched_token_count += 1;
                                break;
                            }
                        }
                    }
                    if !found_syn {
                        // 同义词也没命中，最后检查相关词（最低权重）
                        for rel in &related_terms {
                            if let Some(pos) = desc.find(rel.as_str()) {
                                let cat_w = category_weight_at(pos);
                                related_hits += 0.1 * cat_w; // 相关词降权到 10%
                                match_positions.push(pos);
                                matched_token_count += 1;
                                break;
                            }
                        }
                    }
                }
            }

            let total_hits = direct_hits + synonym_hits + related_hits;
            if total_hits == 0.0 {
                continue;
            }

            // 基础分 = 加权命中数 / 查询词数，反映查询的覆盖度
            let base_score = (total_hits / n_original).min(1.0);

            // 覆盖率惩罚：查询有 N 个有意义的词但只命中了 M 个，按比例打折
            // 跳过停用词是因为它们没有区分度，不应参与覆盖率计算
            let is_stopword = |t: &str| -> bool {
                matches!(t, "使用" | "用" | "在" | "有" | "的" | "了" | "是" | "和" | "与" | "把" | "被" | "让" | "给" | "从" | "到" | "对" | "为" | "这" | "那" | "一" | "个" | "不" | "也" | "都" | "就" | "还" | "又" | "很" | "最")
            };
            let n_meaningful = tokens.iter().filter(|t| t.chars().count() > 1 && !is_stopword(t)).count().max(1) as f32;
            let coverage = if n_meaningful > 0.0 {
                matched_token_count as f32 / n_meaningful
            } else {
                1.0
            };
            let coverage = coverage.min(1.0);
            let base_score = base_score * coverage;

            // 近邻加分：多个查询词在同一类别段落内位置接近时额外加分。
            // 跨类别的近邻没有意义（如背景物里的"白色"和人物里的"衣服"虽然都匹配，
            // 但不代表描述的是同一个东西），所以只在同一类别段落内计算距离。
            let proximity_bonus = if tokens.len() > 1 && match_positions.len() > 1 {
                match_positions.sort();
                let mut min_same_category_dist = usize::MAX;
                for window in match_positions.windows(2) {
                    let w0 = window[0];
                    let w1 = window[1];
                    // 只计算同一类别段落内的距离
                    if category_weight_at(w0) == category_weight_at(w1) {
                        let dist = w1.saturating_sub(w0);
                        if dist < min_same_category_dist {
                            min_same_category_dist = dist;
                        }
                    }
                }
                if min_same_category_dist <= 2 {
                    0.40 // 紧邻 token——强信号（如"白色衣服"连续出现）
                } else if min_same_category_dist <= 10 {
                    0.25 // 同一短语
                } else if min_same_category_dist <= 30 {
                    0.12 // 同一句子
                } else {
                    0.0 // 距离太远或跨类别，不加分
                }
            } else {
                0.0
            };

            let score = (base_score + proximity_bonus).min(1.0);

            if let Ok(id) = id_str.parse::<Uuid>() {
                tracing::debug!("BM25 segment {} score {:.3} (base={:.3}, proximity=+{:.2}, direct={}, synonym={}, related={})", id, score, base_score, proximity_bonus, direct_hits, synonym_hits, related_hits);
                scored.push((id, score));
            }
        }

        // 按分数降序排列，截取前 limit 个结果
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored)
    }

    /// 向 jieba 添加自定义词，确保人脸名等专有名词能被正确分词。
    /// 例如 "李佳琦" 默认会被切成 "李" + "佳" + "琦"，添加后才能作为一个整体。
    pub fn add_jieba_words(&self, words: &[String]) {
        let mut extra = JIEBA_EXTRA_WORDS.lock().unwrap();
        for word in words {
            if !extra.contains(word) {
                JIEBA.lock().unwrap().add_word(word, None, None);
                extra.push(word.clone());
            }
        }
    }
}

/// 同义词扩展的公开接口，供外部模块调用。
pub fn expand_synonyms_public(token: &str) -> (Vec<String>, Vec<String>) {
    expand_synonyms(token)
}

/// 对查询词进行同义词和相关词扩展。
///
/// 返回 (同义词列表, 相关词列表)：
/// - 同义词：近似可互换的词（如"衣服"和"上衣"），匹配时获得较高权重
/// - 相关词：同一领域但不可互换的词（如"衣服"和"连衣裙"），匹配时获得较低权重
///
/// 这些词组与 YOLO 检测标签对齐，确保用户用自然语言查询时能匹配到
/// YOLO 输出的标准化标签。
fn expand_synonyms(token: &str) -> (Vec<String>, Vec<String>) {
    let mut synonyms = vec![token.to_string()];
    let mut related = Vec::new();

    // 近义同义词组：组内词汇几乎可以互换使用
    const SYNONYM_GROUPS: &[&[&str]] = &[
        // 服饰 —— YOLO 标签：连衣裙/裙子/外套/夹克/衬衫/毛衣/牛仔裤/帽子/围巾/手套/鞋子/靴子/运动鞋/高跟鞋/凉鞋
        &["衣服", "上衣", "外套", "衬衫", "衫"],
        &["裤子", "长裤", "短裤", "西裤"],
        &["裙", "裙子", "连衣裙", "半身裙"],
        &["鞋", "鞋子"],
        &["靴", "靴子"],
        &["帽", "帽子"],
        &["围巾", "丝巾", "披肩"],
        &["手套", "手袋"],
        // 美妆 —— YOLO 标签：口红/香水/面霜/睫毛膏/眼影/粉底/腮红/指甲油/粉饼/唇釉/遮瑕膏/眼线笔/化妆刷/镜子/梳子
        &["口红", "唇釉", "唇膏", "唇彩", "唇泥", "润唇膏", "唇笔"],
        &["粉底", "底妆", "粉饼"],
        &["眼影", "眼妆"],
        &["腮红", "修容"],
        &["美甲", "指甲油"],
        &["面霜", "乳霜", "润肤霜"],
        &["遮瑕膏", "遮瑕"],
        &["眼线笔", "眼线液", "眼线"],
        &["化妆刷", "粉刷", "刷子"],
        &["香水", "香氛"],
        // 人物 —— YOLO 标签：人
        &["人", "人物", "主体"],
        &["女性", "女人", "女孩", "女士"],
        &["男性", "男人", "男生", "男士"],
        &["脸", "面部", "脸颊"],
        &["嘴", "唇", "嘴唇"],
        &["手", "手指", "手掌"],
        // 家具 —— YOLO 标签：长椅/椅子/沙发/床/餐桌/马桶/花瓶
        &["椅子", "座椅"],
        &["沙发", "长沙发"],
        &["床", "床铺"],
        &["餐桌", "饭桌", "桌子"],
        &["花瓶", "瓶"],
        // 厨房 —— YOLO 标签：瓶子/酒杯/杯子/叉子/刀/勺子/碗/微波炉/烤箱/水槽/冰箱
        &["杯子", "水杯", "茶杯", "玻璃杯"],
        &["碗", "碗碟"],
        &["冰箱", "电冰箱"],
        // 电子产品 —— YOLO 标签：电视/笔记本电脑/鼠标/遥控器/键盘/手机
        &["手机", "电话", "智能手机"],
        &["电视", "电视机"],
        &["笔记本电脑", "笔记本", "电脑"],
        // 配饰 —— YOLO 标签：背包/雨伞/手提包/领带/行李箱
        &["包", "手提包", "背包", "手袋", "挎包", "包包"],
        &["行李箱", "旅行箱", "拉杆箱"],
        // 家居装饰 —— YOLO 标签：窗帘/地毯/靠垫/蜡烛/灯/吊灯/毯子/枕头/置物架/衣柜/门/窗户/风扇/空调/植物/花/花束
        &["灯", "台灯", "灯具"],
        &["吊灯", "顶灯"],
        &["置物架", "架子", "层架"],
        &["衣柜", "柜子", "衣橱"],
        &["毯子", "毛毯", "被子"],
        &["枕头", "靠枕"],
        &["花", "花朵", "鲜花"],
        &["花束", "捧花"],
        &["植物", "绿植", "盆栽"],
        &["空调", "冷气"],
        // 摄影器材 —— YOLO 标签：相机/镜头/三脚架/麦克风/音箱/耳机/显示器/投影仪/环形灯/无人机/稳定器
        &["相机", "照相机", "摄像机"],
        &["耳机", "耳麦"],
        &["显示器", "屏幕", "荧幕"],
        // 标志 —— YOLO 标签：箭头/水印/警示牌/广告牌/海报/横幅/二维码/条形码/标志
        &["广告牌", "广告", "招牌"],
        &["海报", "宣传画"],
        &["标志", "logo", "标识"],
        // 动物 —— YOLO 标签：鸟/猫/狗/马/羊/牛/大象/熊/斑马/长颈鹿
        &["猫", "猫咪"],
        &["狗", "狗狗", "犬"],
        &["鸟", "小鸟", "飞鸟"],
        // 交通工具 —— YOLO 标签：汽车/摩托车/飞机/公交车/火车/卡车/船/自行车
        &["汽车", "车", "轿车"],
        &["公交车", "巴士", "公交"],
        &["卡车", "货车"],
        // 食物 —— YOLO 标签：米饭/面条/饺子/寿司/牛排/沙拉/汤/面包/鸡蛋/牛奶/咖啡/茶/果汁/啤酒/水/巧克力/冰淇淋/饼干/各种水果
        &["咖啡", "拿铁", "美式"],
        &["茶", "茶饮", "奶茶"],
        &["果汁", "鲜榨"],
        &["面包", "吐司", "欧包"],
        // 动作
        &["讲解", "说话", "介绍", "解说"],
        &["展示", "展示", "呈现"],
        &["涂抹", "涂", "擦", "抹", "上妆"],
    ];

    // 相关词组：同一领域但不可互换，只作为最低权重的补充匹配
    const RELATED_GROUPS: &[&[&str]] = &[
        // 服饰相关
        &["衣服", "内搭", "高领", "毛衣", "夹克", "皮衣", "皮夹克", "西装", "风衣", "大衣", "卫衣", "T恤", "背心", "连衣裙", "裙子", "外套", "衬衫", "牛仔裤", "帽子", "围巾", "手套", "鞋子", "靴子", "运动鞋", "高跟鞋", "凉鞋", "领带"],
        // 裤子相关
        &["裤子", "牛仔裤", "西裤", "运动裤", "休闲裤", "短裤"],
        // 鞋类相关
        &["鞋", "靴", "运动鞋", "高跟鞋", "凉鞋", "皮鞋", "拖鞋", "靴子", "鞋子"],
        // 美妆相关
        &["口红", "化妆品", "彩妆", "唇釉", "唇膏", "唇彩", "唇泥", "润唇膏", "粉底", "粉饼", "眼影", "眼妆", "腮红", "修容", "指甲油", "面霜", "遮瑕膏", "眼线笔", "睫毛膏", "化妆刷", "香水", "镜子", "梳子"],
        // 人物相关
        &["人", "女性", "男性", "人物", "主体", "女孩", "女人", "男生", "男人", "女士", "男士"],
        // 家具相关
        &["椅子", "沙发", "床", "餐桌", "长椅", "马桶", "花瓶", "桌子", "柜子", "置物架", "衣柜"],
        // 灯光相关
        &["灯", "台灯", "灯光", "灯具", "吊灯", "环形灯", "蜡烛"],
        // 架子/柜子相关
        &["架子", "置物架", "柜", "柜子", "书架", "衣柜", "层架"],
        // 包相关
        &["包", "手提包", "背包", "手袋", "挎包", "行李箱", "包包"],
        // 电子产品相关
        &["手机", "电视", "电脑", "笔记本电脑", "鼠标", "遥控器", "键盘", "相机", "显示器", "耳机", "音箱", "麦克风"],
        // 动作相关
        &["展示", "拿", "举", "持", "握", "拿取"],
        &["涂抹", "涂", "擦", "抹", "上妆"],
        // 动物相关
        &["猫", "狗", "鸟", "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿"],
        // 交通工具相关
        &["汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船", "自行车"],
        // 食物相关
        &["米饭", "面条", "饺子", "寿司", "牛排", "沙拉", "汤", "面包", "鸡蛋", "牛奶", "咖啡", "茶", "果汁", "啤酒", "水", "巧克力", "冰淇淋", "饼干"],
        // 水果相关
        &["苹果", "香蕉", "橙子", "葡萄", "草莓", "西瓜", "芒果", "桃子", "樱桃", "柠檬", "梨", "菠萝"],
        // 家居相关
        &["窗帘", "地毯", "靠垫", "毯子", "枕头", "植物", "花", "花束", "盆栽"],
    ];

    // 在同义词组中查找，把同组其他词作为同义词
    for group in SYNONYM_GROUPS {
        if group.contains(&token) {
            for &syn in *group {
                if syn != token {
                    synonyms.push(syn.to_string());
                }
            }
            break; // 每个 token 最多属于一个同义词组
        }
    }

    // 在相关词组中查找，把同组其他词作为相关词（排除已是同义词的）
    for group in RELATED_GROUPS {
        if group.contains(&token) {
            for &rel in *group {
                if rel != token && !synonyms.contains(&rel.to_string()) {
                    related.push(rel.to_string());
                }
            }
            break;
        }
    }

    (synonyms, related)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jieba_cut_chinese() {
        use jieba_rs::Jieba;
        let jieba = Jieba::new();
        let tokens: Vec<&str> = jieba.cut("白色的衣服", false);
        println!("jieba tokens: {:?}", tokens);
        // Should split into at least "白色" and "衣服"
        assert!(tokens.contains(&"白色") || tokens.contains(&"白色的"), "Expected '白色' or '白色的' in tokens, got: {:?}", tokens);
        assert!(tokens.contains(&"衣服"), "Expected '衣服' in tokens, got: {:?}", tokens);
    }

    #[test]
    fn test_expand_synonyms_clothing() {
        let (syns, related) = expand_synonyms("衣服");
        assert!(syns.contains(&"上衣".to_string()), "Expected '上衣' in synonyms for '衣服', got: {:?}", syns);
        assert!(syns.contains(&"外套".to_string()), "Expected '外套' in synonyms for '衣服', got: {:?}", syns);
        // "内搭" should be in related, not synonyms
        assert!(!syns.contains(&"内搭".to_string()), "'内搭' should not be in synonyms for '衣服'");
        assert!(related.contains(&"内搭".to_string()), "Expected '内搭' in related for '衣服', got: {:?}", related);
        // YOLO labels should be in related
        assert!(related.contains(&"连衣裙".to_string()), "Expected '连衣裙' in related for '衣服'");
        assert!(related.contains(&"牛仔裤".to_string()), "Expected '牛仔裤' in related for '衣服'");
    }

    #[test]
    fn test_expand_synonyms_yolo_labels() {
        // YOLO label "口红" should expand to synonyms
        let (syns, _) = expand_synonyms("口红");
        assert!(syns.contains(&"唇釉".to_string()), "Expected '唇釉' in synonyms for '口红'");
        assert!(syns.contains(&"唇膏".to_string()), "Expected '唇膏' in synonyms for '口红'");
        // YOLO label "鞋子" should expand
        let (syns2, _) = expand_synonyms("鞋子");
        assert!(syns2.contains(&"鞋".to_string()), "Expected '鞋' in synonyms for '鞋子'");
        // YOLO label "置物架" should expand
        let (syns3, _) = expand_synonyms("置物架");
        assert!(syns3.contains(&"架子".to_string()), "Expected '架子' in synonyms for '置物架'");
    }

    #[test]
    fn test_expand_synonyms_no_match() {
        let (syns, related) = expand_synonyms("白色");
        // "白色" has no synonym/related group, should return just itself
        assert_eq!(syns, vec!["白色".to_string()]);
        assert!(related.is_empty());
    }
}
