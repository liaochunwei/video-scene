//! 存储层模块：统一管理视频分析项目的所有持久化存储。
//!
//! 本模块将不同类型的存储抽象为独立组件，通过 `Storage` 聚合体统一对外提供服务：
//! - `ConfigDatabase` — 全局配置库（工作区注册、人脸库），跨工作区共享
//! - `Database` — 工作区级 SQLite 数据库（视频、片段、检测记录、全文搜索）
//! - `FileStore` — 文件存储（关键帧图片、人脸图片）
//! - `VectorIndex` — 向量相似度索引（人脸特征、场景特征、图像特征）
//! - `SceneIndices` — 按描述类别分组的场景向量索引集合

pub mod config_db;
pub mod database;
pub mod migration;
pub mod vector_index;
pub mod file_store;

pub use config_db::{ConfigDatabase, WorkspaceEntry};
pub use database::Database;
pub use vector_index::VectorIndex;
pub use vector_index::SceneIndices;
pub use file_store::FileStore;

/// 存储聚合体：将所有存储组件绑定在一起，对外提供一站式访问。
///
/// 打开流程：先读取全局配置库确定当前工作区，再打开工作区对应的
/// 数据库、文件存储和向量索引。这样设计是因为全局配置（工作区列表、
/// 人脸库）和工作区数据（视频/片段/检测）的生命周期不同——
/// 人脸库在所有工作区之间共享，而视频数据属于特定工作区。
pub struct Storage {
    pub config_db: ConfigDatabase,
    pub workspace_db: Database,
    pub file_store: FileStore,
    pub face_index: VectorIndex,
    pub scene_indices: SceneIndices,
    pub image_index: VectorIndex,
    pub config_dir: std::path::PathBuf,
    pub workspace_path: std::path::PathBuf,
}

impl Storage {
    /// 根据设置和可选的工作区名称打开完整的存储栈。
    ///
    /// 如果未指定工作区名称，则使用 state 文件中记录的当前活跃工作区。
    /// 打开过程中会依次初始化配置库、工作区数据库、文件存储和三组向量索引，
    /// 任一环节失败都会立即返回错误。
    pub fn open(
        settings: &crate::config::Settings,
        workspace_name: Option<&str>,
    ) -> anyhow::Result<Self> {
        let config_dir = settings.index.config_dir.clone();
        let state = crate::config::StateFile::load(&config_dir);

        // 全局配置库位于 config_dir 下，存放工作区注册表和人脸库
        let config_db = ConfigDatabase::open(&config_dir.join("config.db"))?;

        // 解析目标工作区名称：优先使用调用方指定的，否则回退到上次活跃的工作区
        let ws_name = workspace_name.unwrap_or(&state.active_workspace);
        let ws_entry = config_db.get_workspace_by_name(ws_name)?
            .ok_or_else(|| anyhow::anyhow!("Workspace '{}' not found. Run 'workspace init {}' first.", ws_name, ws_name))?;
        let workspace_path = std::path::PathBuf::from(&ws_entry.path);

        // 工作区数据库、文件存储和向量索引都在工作区目录下
        let workspace_db = Database::open(&workspace_path.join("index.db"))?;
        let file_store = FileStore::new(&workspace_path)?;
        let face_index = VectorIndex::open(&workspace_path.join("vectors").join("faces.hnsw"))?;
        let vectors_dir = workspace_path.join("vectors");
        let scene_indices = SceneIndices::open(&vectors_dir)?;
        let image_index = VectorIndex::open(&vectors_dir.join("images.hnsw"))?;

        Ok(Self {
            config_db,
            workspace_db,
            file_store,
            face_index,
            scene_indices,
            image_index,
            config_dir,
            workspace_path,
        })
    }
}
