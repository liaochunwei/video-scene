//! video-scene 命令行入口
//!
//! 负责解析命令行参数、加载配置、执行数据存储迁移，
//! 然后根据子命令分发到对应业务逻辑。
//!
//! 支持的子命令：
//! - `index`    ：索引视频（场景检测或云端 VLM 两种模式）
//! - `search`   ：按人脸/物体/场景/图片检索视频片段
//! - `workspace`：管理工作区（初始化、列表、切换激活）
//! - `face`     ：管理人脸库（增删查改、从视频中提取人脸）
//! - `status`   ：查看当前索引导览统计
//! - `list`     ：列出已索引的视频
//! - `remove`   ：移除指定视频或清空整个索引
//! - `clean`    ：清理源文件已不存在的无效索引条目
//! - `config`   ：输出当前完整配置（TOML 格式）
//! - `plugins`  ：管理插件守护进程（启停、状态查看）

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use video_scene::cli::output::{OutputFormat, format_search_results, format_status, format_video_list};
use video_scene::config::Settings;
use video_scene::core::searcher::SearchType;
use video_scene::core::face_extractor::cosine_similarity;
use video_scene::models::FaceLibraryEntry;
use video_scene::plugins::ProgressMessage;
use video_scene::storage::FileStore;

/// 顶层命令行参数定义
///
/// `workspace` 和 `config` 为全局选项，可跟在任何子命令之前；
/// `verbose` / `quiet` 互斥，控制日志输出级别。
#[derive(Parser)]
#[command(name = "video-scene", version, about = "Video retrieval system for face, object and scene search")]
struct Cli {
    /// 指定工作区名称，缺省使用当前激活的工作区
    #[arg(long, global = true)]
    workspace: Option<String>,

    /// 指定配置文件路径，缺省按默认路径查找
    #[arg(long, global = true)]
    config: Option<String>,

    /// 详细输出（日志级别 debug）
    #[arg(short, long, global = true)]
    verbose: bool,

    /// 安静模式（仅输出错误），与 verbose 互斥
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    quiet: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

/// 工作区管理子命令
#[derive(Subcommand, Debug)]
enum WorkspaceAction {
    /// 初始化新工作区（创建目录结构与索引数据库）
    Init {
        /// 工作区名称
        name: String,
        /// 工作区目录路径，缺省为 ~/.video-scene/{name}/
        path: Option<String>,
    },
    /// 列出所有工作区（当前激活的会标记 *）
    List,
    /// 切换激活工作区
    Activate {
        /// 要激活的工作区名称
        name: String,
    },
}

/// 人脸库管理子命令
#[derive(Subcommand, Debug)]
enum FaceAction {
    /// 添加人脸到库中，或为已有人员追加新外观
    Add {
        /// 人物名称
        name: String,
        /// 参考图片路径
        image: String,
        /// 追加模式：为已有人员添加新外观而非创建新条目
        #[arg(long)]
        add: bool,
    },
    /// 列出人脸库中所有人物
    List,
    /// 查看指定人物的详细信息（图片数、特征向量数）
    Info {
        /// 人物名称
        name: String,
    },
    /// 从人脸库中移除指定人物
    Remove {
        /// 人物名称
        name: String,
    },
    /// 重命名人物（支持交互式或 --new-name 直接指定）
    Rename {
        /// 当前名称
        name: String,
        /// 新名称（提供此参数则跳过交互式提示）
        #[arg(long)]
        new_name: Option<String>,
    },
    /// 从视频中提取人脸，聚类后可交互式或自动保存到人脸库
    Extract {
        /// 视频路径
        video: String,
        /// 人脸检测最低置信度，低于此值的检测结果丢弃
        #[arg(long, default_value_t = 0.8)]
        min_confidence: f64,
        /// 人脸质量最低阈值，过滤模糊/侧脸
        #[arg(long, default_value_t = 0.5)]
        min_quality: f64,
        /// 聚类距离阈值，值越小聚类越细（更多人），值越大聚类越粗（更少人）
        #[arg(long, default_value_t = 0.5)]
        cluster_threshold: f64,
        #[arg(long)]
        output_dir: Option<String>,
        /// 是否启用交互式命名（逐个人脸提示用户输入名字）
        #[arg(long, default_value_t = true)]
        interactive: bool,
        /// 自动保存模式：新发现的人脸自动命名为 person_N
        #[arg(long)]
        auto_save: bool,
    },
}

/// 插件管理子命令
///
/// 插件以守护进程方式运行，按需启停，空闲超时后自动退出。
#[derive(Subcommand, Debug)]
enum PluginAction {
    /// 启动插件守护进程（常驻后台，接收启停指令）
    Daemon,
    /// 列出所有已注册插件及其运行状态
    List,
    /// 通过守护进程启动指定类型的插件
    Start {
        /// 插件类型标识
        plugin_type: String,
    },
    /// 通过守护进程停止指定类型的插件
    Stop {
        /// 插件类型标识
        plugin_type: String,
    },
    /// 查看所有插件的详细状态（运行中/空闲时长等）
    Status,
}

/// 主子命令枚举
///
/// 每个变体对应一个顶层功能模块，参数通过 clap 自动从字段派生。
#[derive(Subcommand, Debug)]
enum Commands {
    /// 索引视频：提取场景/人脸/图像特征并存入数据库
    Index {
        /// 视频文件或目录路径
        path: String,
        /// 递归索引子目录
        #[arg(long)]
        recursive: bool,
        /// 要索引的视频扩展名，逗号分隔
        #[arg(long, default_value = "mp4,mov,mkv")]
        extensions: String,
        /// 强制重新索引（即使已有记录）
        #[arg(long)]
        force: bool,
        /// 最小片段时长（秒），短于此值的片段会被合并
        #[arg(long, default_value_t = 2.0)]
        min_seg: f32,
        /// 最大片段时长（秒），超过此值的片段会被拆分
        #[arg(long, default_value_t = 30.0)]
        max_seg: f32,
        /// 索引模式：scene（本地场景检测+VLM）或 video（云端 VLM API，需配置 api_key）
        #[arg(long, default_value = "scene")]
        mode: String,
    },
    /// 检索视频：按文本/图片查找匹配的视频片段
    Search {
        /// 搜索查询文本（--web 模式下可选）
        query: Option<String>,
        /// 按人脸检索
        #[arg(long)]
        face: bool,
        /// 按物体检索
        #[arg(long)]
        object: bool,
        /// 按场景检索
        #[arg(long)]
        scene: bool,
        /// 按图片检索（提供图片路径）
        #[arg(long)]
        image: Option<String>,
        /// 返回结果数量上限
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// 相似度阈值，低于此值的结果被过滤
        #[arg(long)]
        threshold: Option<f64>,
        /// 输出格式：simple / json / detailed
        #[arg(long, default_value = "simple")]
        format: String,
        /// 快捷输出 JSON（等价于 --format json）
        #[arg(long)]
        json: bool,
        /// 启动 Web UI 代替命令行输出
        #[arg(long)]
        web: bool,
        /// Web UI 端口号
        #[arg(long, default_value_t = 6066)]
        port: u16,
        /// 去重：同一视频只保留最佳片段，其余放入 'more' 字段
        #[arg(long)]
        dedup: bool,
    },
    /// 管理工作区
    Workspace {
        #[command(subcommand)]
        action: WorkspaceAction,
    },
    /// 管理人脸库
    Face {
        #[command(subcommand)]
        action: FaceAction,
    },
    /// 显示索引状态（视频数、片段数、人脸数）
    Status,
    /// 列出所有已索引的视频
    List {
        /// 只显示视频数量，不列出详细列表
        #[arg(short, long)]
        count: bool,
    },
    /// 移除指定视频的索引记录，或用 --all 清空整个索引
    Remove {
        /// 要移除的视频路径（使用 --all 时不需要）
        video: Option<String>,
        /// 清空全部索引数据（数据库、关键帧、向量文件）
        #[arg(long)]
        all: bool,
    },
    /// 清理源文件已不存在的无效索引条目
    Clean,
    /// 输出当前完整配置（TOML 格式）
    Config,
    /// 管理插件守护进程
    Plugins {
        #[command(subcommand)]
        action: PluginAction,
    },
}

/// 将秒数格式化为 MM:SS 时间戳，用于在人脸提取结果中显示最佳帧位置
fn fmt_timestamp(secs: f32) -> String {
    let total = secs.round() as u32;
    let m = total / 60;
    let s = total % 60;
    format!("{:02}:{:02}", m, s)
}

/// 解析图片路径：相对路径基于 base_dir 拼接为绝对路径，绝对路径原样返回
///
/// 这是为了让用户在命令行中可以使用相对路径引用图片，
/// 内部统一转换为绝对路径以避免工作目录不同导致找不到文件。
fn resolve_image_path(img: &str, base_dir: &std::path::Path) -> String {
    if std::path::Path::new(img).is_relative() {
        base_dir.join(img).to_string_lossy().to_string()
    } else {
        img.to_string()
    }
}

/// 重命名人脸条目：迁移图片文件、更新数据库记录
///
/// 流程：
/// 1. 将旧名称目录下的图片逐一复制到新名称目录，复制成功后删除旧文件
/// 2. 清理旧的空目录
/// 3. 构建新 FaceLibraryEntry，保留所有特征向量
/// 4. 先插入新记录再删除旧记录，保证数据安全
fn do_rename(
    old_name: &str,
    new_name: &str,
    entry: &FaceLibraryEntry,
    config_db: &video_scene::storage::ConfigDatabase,
    file_store: &FileStore,
) -> anyhow::Result<()> {
    // 逐张迁移图片文件：复制到新目录，成功后删除旧文件
    let mut new_images = Vec::new();
    for img in &entry.images {
        if img.is_empty() {
            continue;
        }
        let full_path = if std::path::Path::new(img).is_relative() {
            file_store.base_dir().join(img)
        } else {
            std::path::PathBuf::from(img)
        };
        if full_path.exists() {
            if let Ok(data) = std::fs::read(&full_path) {
                let idx = new_images.len();
                // 保存到新名称目录下，成功则记录相对路径并删除旧文件
                if let Ok(dest) = file_store.save_face_image(new_name, &data, idx) {
                    if let Ok(rel) = dest.strip_prefix(file_store.base_dir()) {
                        new_images.push(rel.to_string_lossy().to_string());
                    } else {
                        new_images.push(dest.to_string_lossy().to_string());
                    }
                    let _ = std::fs::remove_file(&full_path);
                    continue;
                }
            }
        }
        // 文件不存在或复制失败时保留原始路径记录
        new_images.push(img.clone());
    }
    // 清理旧名称目录（仅当目录为空时才能成功删除）
    let old_dir = file_store.base_dir().join("face_library").join(old_name);
    if old_dir.is_dir() {
        let _ = std::fs::remove_dir(&old_dir);
    }
    // 构建新条目，确保保留所有已有的特征向量
    let mut new_entry = FaceLibraryEntry::new(
        new_name.to_string(),
        new_images.first().cloned().unwrap_or_default(),
        entry.feature_vectors.first().cloned().unwrap_or_default(),
    );
    if new_images.len() > 1 {
        new_entry.images = new_images;
    }
    // 保留全部特征向量（同一人可能有多个外观向量）
    new_entry.feature_vectors = entry.feature_vectors.clone();
    config_db.insert_face(&new_entry)?;
    config_db.delete_face(old_name)?;
    Ok(())
}

/// 打开全局配置数据库（位于 config_dir/config.db）
///
/// 配置数据库存储工作区列表、人脸库等全局数据，与工作区索引数据库分开。
fn open_config_db(settings: &Settings) -> anyhow::Result<video_scene::storage::ConfigDatabase> {
    let config_dir = &settings.index.config_dir;
    Ok(video_scene::storage::ConfigDatabase::open(&config_dir.join("config.db"))?)
}

/// 初始化日志系统
///
/// 优先使用环境变量 RUST_LOG 的配置；若未设置则根据命令行参数决定级别：
/// - verbose → debug
/// - quiet   → error
/// - 默认    → info
fn init_env(verbose: bool, quiet: bool) {
    let level = if verbose {
        "debug"
    } else if quiet {
        "error"
    } else {
        "info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level))
        )
        .init();
}

/// 加载配置文件
///
/// 若指定了 config_path 则从该路径加载，否则使用默认路径查找规则。
fn load_settings(config_path: Option<&str>) -> anyhow::Result<Settings> {
    let config_path = config_path.map(PathBuf::from);
    Ok(Settings::load(config_path.as_ref())?)
}

/// 创建带进度条显示的回调函数，用于展示插件处理进度
///
/// 区分两种进度消息：
/// - total > 1：帧级进度，更新 indicatif 进度条（显示 spinner + 计数）
/// - total == 1：插件内部单步进度，仅输出 debug 日志避免干扰进度条
fn make_progress_cb(multi: &MultiProgress) -> Arc<dyn Fn(ProgressMessage) + Send + Sync> {
    let pb = multi.add(ProgressBar::new(0));
    pb.set_style(ProgressStyle::with_template(
        "{spinner:.green} {msg} [{pos}/{len}]"
    ).unwrap_or_else(|_| ProgressStyle::default_spinner()));

    Arc::new(move |msg: ProgressMessage| {
        if msg.total > 1 {
            // 上一步已完成时重置进度条，为当前步骤做准备
            if pb.is_finished() {
                pb.reset();
            }
            pb.set_message(msg.message.clone());
            pb.set_length(msg.total as u64);
            pb.set_position(msg.current as u64);
            if msg.current >= msg.total {
                pb.finish_with_message(format!("✓ {}", msg.message));
            }
        } else {
            tracing::debug!("Plugin: {} ({}/{})", msg.message, msg.current, msg.total);
        }
    })
}

/// 安静模式下的空回调：丢弃所有进度消息，避免任何输出
fn silent_progress_cb() -> Arc<dyn Fn(ProgressMessage) + Send + Sync> {
    Arc::new(|_: ProgressMessage| {})
}

/// 程序入口：解析参数 → 初始化环境 → 分发子命令
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_env(cli.verbose, cli.quiet);

    let settings = load_settings(cli.config.as_deref())?;
    let config_dir = settings.index.config_dir.clone();
    // 首次运行时自动执行数据库 schema 迁移，保证存储结构为最新版本
    video_scene::storage::migration::migrate_if_needed(&config_dir)?;

    let multi = MultiProgress::new();

    // 根据模式选择进度回调：安静模式用空回调，否则用带进度条的回调
    let progress_cb = if cli.quiet {
        silent_progress_cb()
    } else {
        make_progress_cb(&multi)
    };

    match cli.command {
        Some(Commands::Index { path, recursive, extensions, force, min_seg, max_seg, mode }) => {
            // 命令行参数覆盖配置文件中的片段时长设置
            let mut settings = settings;
            settings.index.scene.min_segment_duration = min_seg;
            settings.index.scene.max_segment_duration = max_seg;

            let mut storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;

            // 解析逗号分隔的扩展名列表
            let exts: Vec<String> = extensions.split(',').map(|s| s.trim().to_string()).collect();
            let video_path = PathBuf::from(&path);

            // video 模式依赖云端 VLM API，必须提供 api_key
            let is_video_mode = mode == "video";
            if is_video_mode && settings.plugins.vlm_api.api_key.is_empty() {
                anyhow::bail!("plugins.vlm_api.api_key is required for --mode video. Set it in config.toml");
            }

            if video_path.is_file() {
                // 单文件索引：根据模式选择本地或云端 VLM 流程
                eprintln!("Indexing: {}", path);
                if is_video_mode {
                    video_scene::core::indexer::index_video_vlm_api(
                        &video_path, &settings, &storage.workspace_db, &storage.file_store, &mut storage.face_index, &mut storage.scene_indices, &mut storage.image_index, force,
                        &*progress_cb,
                    )?;
                } else {
                    video_scene::core::indexer::index_video(
                        &video_path, &settings, &storage.workspace_db, &storage.file_store, &mut storage.face_index, &mut storage.scene_indices, &mut storage.image_index, force,
                        &*progress_cb,
                    )?;
                }
                eprintln!("Done.");
            } else if video_path.is_dir() {
                eprintln!("Indexing directory: {}", path);
                let summary = video_scene::core::indexer::index_directory(
                    &video_path, &settings, &storage.workspace_db, &storage.file_store,
                    &mut storage.face_index, &mut storage.scene_indices, &mut storage.image_index,
                    recursive, &exts, force, is_video_mode,
                    &*progress_cb,
                )?;
                println!("{}", summary);
            } else {
                anyhow::bail!("Path does not exist: {}", path);
            }
        }

        Some(Commands::Search { query, face, object, scene, image, top, threshold, format, json, web, port, dedup }) => {
            let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;

            if web {
                // Web 模式：启动 HTTP 服务，前端通过浏览器交互
                let state = std::sync::Arc::new(video_scene::web::AppState {
                    storage: std::sync::Mutex::new(storage),
                });
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(video_scene::web::start_server(state, port))?;
                return Ok(());
            }

            // CLI 模式下查询文本为必填项
            let query = query.ok_or_else(|| anyhow::anyhow!("Search query is required (not needed with --web)"))?;

            // 根据标志位确定检索类型；无标志时使用 Auto 让搜索引擎自动判断
            let search_type = if face {
                SearchType::Face
            } else if object {
                SearchType::Object
            } else if scene {
                SearchType::Scene
            } else if image.is_some() {
                SearchType::Image
            } else {
                SearchType::Auto
            };

            // 未显式指定阈值时，根据检索类型使用不同的默认阈值
            // 不同类型的相似度分布差异较大，因此需要不同的过滤标准
            let effective_threshold = threshold.unwrap_or_else(|| match search_type {
                SearchType::Auto => 0.2,
                SearchType::Face => 0.3,
                SearchType::Object => 0.3,
                SearchType::Scene => 0.15,
                SearchType::Image => 0.3,
            }) as f32;

            let image_path = image.as_deref().unwrap_or("");

            eprintln!("Searching: {}", query);
            let response = video_scene::core::searcher::search(
                &query, search_type, top, effective_threshold, 1, top, dedup, &settings, &storage.workspace_db, &storage.config_db, &storage.face_index, &storage.scene_indices, &storage.image_index,
                image_path, &*progress_cb,
            )?;
            let results = response.results;

            // --json 是 --format json 的快捷方式
            let output_format = if json {
                OutputFormat::Json
            } else {
                OutputFormat::from(format.as_str())
            };
            println!("{}", format_search_results(&results, output_format));
        }

        Some(Commands::Workspace { action }) => {
            match action {
                WorkspaceAction::Init { name, path } => {
                    let config_db = open_config_db(&settings)?;
                    let config_dir = &settings.index.config_dir;
                    // 未指定路径时在工作区根目录下以名称创建子目录
                    let workspace_path = match path {
                        Some(p) => std::path::PathBuf::from(&p),
                        None => config_dir.join(&name),
                    };
                    std::fs::create_dir_all(&workspace_path)?;
                    // 打开数据库即自动创建 schema
                    video_scene::storage::Database::open(&workspace_path.join("index.db"))?;
                    // 在全局 config.db 中注册此工作区
                    config_db.insert_workspace_with_defaults(&name, &workspace_path.to_string_lossy())?;
                    println!("Workspace '{}' initialized at {}", name, workspace_path.display());
                }
                WorkspaceAction::List => {
                    let config_db = open_config_db(&settings)?;
                    let state = video_scene::config::StateFile::load(&settings.index.config_dir);
                    let workspaces = config_db.list_workspaces()?;
                    if workspaces.is_empty() {
                        println!("No workspaces.");
                    } else {
                        // 当前激活的工作区前标记 * 号
                        for ws in &workspaces {
                            let marker = if ws.name == state.active_workspace { "*" } else { " " };
                            println!("{} {} {}", marker, ws.name, ws.path);
                        }
                    }
                }
                WorkspaceAction::Activate { name } => {
                    let config_db = open_config_db(&settings)?;
                    // 先验证工作区是否存在，再切换激活状态
                    config_db.get_workspace_by_name(&name)?
                        .ok_or_else(|| anyhow::anyhow!("Workspace '{}' not found", name))?;
                    let mut state = video_scene::config::StateFile::load(&settings.index.config_dir);
                    state.active_workspace = name.clone();
                    state.save(&settings.index.config_dir)?;
                    println!("Active workspace set to '{}'", name);
                }
            }
        }

        Some(Commands::Face { action }) => {
            match action {
                FaceAction::Add { name, image, add } => {
                    let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
                    // config_file_store 存储人脸图片，位于全局配置目录而非工作区目录
                    let config_file_store = FileStore::new(&storage.config_dir)?;

                    eprintln!("Encoding face: {}...", name);
                    let feature = video_scene::plugins::face::encode_face(&image, &*progress_cb)?;

                    // 将参考图片复制到人脸库目录，存储相对路径以便工作区迁移
                    let image_data = std::fs::read(&image)
                        .map_err(|e| anyhow::anyhow!("Failed to read image {}: {}", image, e))?;
                    let img_index = if add {
                        // 追加模式：接在已有图片后面，使用下一个序号
                        storage.config_db.get_face_by_name(&name)?
                            .map(|e| e.images.len())
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    let saved_path = config_file_store.save_face_image(&name, &image_data, img_index)?;
                    let rel_path = match saved_path.strip_prefix(config_file_store.base_dir()) {
                        Ok(p) => p.to_string_lossy().to_string(),
                        Err(_) => saved_path.to_string_lossy().to_string(),
                    };

                    if add {
                        // 追加新外观到已有条目：检查与已有向量的相似度，避免重复添加几乎相同的向量
                        let mut entry = storage.config_db.get_face_by_name(&name)?
                            .ok_or_else(|| anyhow::anyhow!("Face '{}' not found in library", name))?;
                        let added = entry.add_image_if_different(rel_path, feature, 0.15);
                        storage.config_db.update_face(&entry)?;
                        if added {
                            println!("Added new look to face: {} (now {} vectors)", name, entry.feature_vectors.len());
                        } else {
                            println!("Added image to face: {} (similar to existing, {} vectors)", name, entry.feature_vectors.len());
                        }
                    } else {
                        let entry = video_scene::models::FaceLibraryEntry::new(name.clone(), rel_path, feature);
                        storage.config_db.insert_face(&entry)?;
                        println!("Added face: {}", name);
                    }
                }

                FaceAction::List => {
                    let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
                    let faces = storage.config_db.list_faces()?;
                    if faces.is_empty() {
                        println!("Face library is empty.");
                    } else {
                        for face in &faces {
                            println!("{} ({} images)", face.name, face.images.len());
                        }
                    }
                }

                FaceAction::Remove { name } => {
                    let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
                    storage.config_db.delete_face(&name)?;
                    println!("Removed face: {}", name);
                }

                FaceAction::Info { name } => {
                    let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
                    let config_file_store = FileStore::new(&storage.config_dir)?;
                    let entry = storage.config_db.get_face_by_name(&name)
                        .map_err(|_| anyhow::anyhow!("Face '{}' not found in library", name))?
                        .ok_or_else(|| anyhow::anyhow!("Face '{}' not found in library", name))?;

                    println!("Name: {} ({} images, {} vectors)", name, entry.images.len(), entry.feature_vectors.len());
                    for img in &entry.images {
                        if !img.is_empty() {
                            let full_path = resolve_image_path(img, config_file_store.base_dir());
                            println!("Image: {}", full_path);
                        }
                    }
                }

                FaceAction::Rename { name, new_name } => {
                    let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
                    let config_file_store = FileStore::new(&storage.config_dir)?;
                    let entry = storage.config_db.get_face_by_name(&name)
                        .map_err(|_| anyhow::anyhow!("Face '{}' not found in library", name))?
                        .ok_or_else(|| anyhow::anyhow!("Face '{}' not found in library", name))?;

                    if let Some(new) = new_name {
                        // 通过 --new-name 参数直接重命名
                        if new == name {
                            println!("Same name, nothing to do.");
                        } else {
                            do_rename(&name, &new, &entry, &storage.config_db, &config_file_store)?;
                            println!("Renamed: {} -> {}", name, new);
                        }
                    } else {
                        // 交互模式：先展示图片，再提示用户输入新名称
                        for img in &entry.images {
                            if !img.is_empty() {
                                let full_path = resolve_image_path(img, config_file_store.base_dir());
                                println!("Image: {}", full_path);
                            }
                        }
                        println!("Current name: {}", name);
                        println!("Enter new name (or press Enter to cancel):");
                        let mut input = String::new();
                        std::io::stdin().read_line(&mut input)?;
                        let new = input.trim();
                        if !new.is_empty() && new != name {
                            do_rename(&name, new, &entry, &storage.config_db, &config_file_store)?;
                            println!("Renamed: {} -> {}", name, new);
                        } else {
                            println!("Cancelled.");
                        }
                    }
                }

                FaceAction::Extract { video, min_confidence, min_quality, cluster_threshold, output_dir: _, interactive, auto_save } => {
                    let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
                    let config_file_store = FileStore::new(&storage.config_dir)?;

                    eprintln!("Extracting faces from: {}", video);
                    // 从视频中检测人脸并按特征向量聚类成不同人物
                    let persons = video_scene::core::face_extractor::extract_faces(
                        &video, min_confidence, min_quality, cluster_threshold, &settings, &storage.file_store,
                        &*progress_cb,
                    )?;

                    // 加载现有人脸库，用于将提取结果匹配到已知人物
                    let existing_faces = storage.config_db.list_faces()?;

                    // 匹配阈值：特征向量余弦相似度达到此值才认为是同一人
                    let match_threshold = 0.5f32;

                    for (i, person) in persons.iter().enumerate() {
                        // 对每个提取出的人物，在人脸库中找最相似的已有条目
                        // 每个已有条目可能有多个特征向量，取最高相似度
                        let matched_name = existing_faces.iter()
                            .filter_map(|entry| {
                                // 对该人物的所有特征向量取最佳相似度
                                let best_sim = entry.feature_vectors.iter()
                                    .map(|fv| cosine_similarity(&person.feature_vector, fv))
                                    .fold(0.0f32, f32::max);
                                if best_sim >= match_threshold { Some((entry.name.clone(), best_sim)) } else { None }
                            })
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(name, _)| name);

                        if let Some(name) = matched_name {
                            // 匹配到已有人物：将新图片和特征向量合并到已有条目
                            if let Some(mut entry) = storage.config_db.get_face_by_name(&name)? {
                                // 将图片从临时目录移到以人物名命名的正式目录
                                let img_path = if !person.best_frame_path.is_empty() {
                                    video_scene::core::face_extractor::move_face_image_to_name(
                                        &person.best_frame_path, &name, &config_file_store,
                                    ).unwrap_or_else(|| person.best_frame_path.clone())
                                } else {
                                    person.best_frame_path.clone()
                                };
                                entry.add_image_if_different(img_path, person.feature_vector.clone(), 0.15);
                                storage.config_db.update_face(&entry)?;
                                println!("[{}] Matched existing: {} (now {} images, best at {})", i + 1, name, entry.images.len(), fmt_timestamp(person.best_timestamp));
                            }
                        } else if auto_save {
                            // 自动保存模式：新人物自动命名为 person_N，避免重名追加后缀
                            let mut name = format!("person_{}", i + 1);
                            let mut suffix = 2;
                            while storage.config_db.get_face_by_name(&name)?.is_some() {
                                name = format!("person_{}_{}", i + 1, suffix);
                                suffix += 1;
                            }
                            video_scene::core::face_extractor::save_person_to_library(person, &name, &storage.config_db, &config_file_store)?;
                            println!("[{}] New: {} ({} appearances, best at {})", i + 1, name, person.appearance_count, fmt_timestamp(person.best_timestamp));
                        } else {
                            // 新人物但未开启自动保存：显示信息，交互模式下提示用户命名
                            println!("[{}] New person ({} appearances, quality: {:.2}, best at {})", i + 1, person.appearance_count, person.quality, fmt_timestamp(person.best_timestamp));
                            if interactive {
                                println!("  Enter name (or press Enter to skip):");
                                let mut input = String::new();
                                std::io::stdin().read_line(&mut input)?;
                                let name = input.trim();
                                if !name.is_empty() {
                                    // 如果用户输入的名字已存在，则合并到已有条目
                                    if let Some(mut existing) = storage.config_db.get_face_by_name(name)? {
                                        let img_path = if !person.best_frame_path.is_empty() {
                                            video_scene::core::face_extractor::move_face_image_to_name(
                                                &person.best_frame_path, name, &config_file_store,
                                            ).unwrap_or_else(|| person.best_frame_path.clone())
                                        } else {
                                            person.best_frame_path.clone()
                                        };
                                        existing.add_image_if_different(img_path, person.feature_vector.clone(), 0.15);
                                        storage.config_db.update_face(&existing)?;
                                        println!("  Merged into existing: {} (now {} images)", name, existing.images.len());
                                    } else {
                                        video_scene::core::face_extractor::save_person_to_library(person, name, &storage.config_db, &config_file_store)?;
                                        println!("  Saved: {}", name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Some(Commands::Status) => {
            let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
            let video_count = storage.workspace_db.video_count()?;
            let segment_count = storage.workspace_db.segment_count()?;
            let face_count = storage.config_db.list_faces()?.len();
            let state = video_scene::config::StateFile::load(&settings.index.config_dir);
            let workspace_name = cli.workspace.as_deref().unwrap_or(&state.active_workspace);
            println!("{}", format_status(video_count, segment_count, face_count, &storage.workspace_path.to_string_lossy(), workspace_name));
        }

        Some(Commands::List { count }) => {
            let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
            if count {
                let n = storage.workspace_db.video_count()?;
                println!("{}", n);
            } else {
                let videos = storage.workspace_db.list_videos()?;
                if videos.is_empty() {
                    println!("No videos indexed.");
                } else {
                    print!("{}", format_video_list(&videos));
                }
            }
        }

        Some(Commands::Remove { video, all }) => {
            if all {
                // 全量清除：删除整个工作区的关键帧、向量文件和数据库
                let storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
                let workspace_path = &storage.workspace_path;

                let keyframes_dir = workspace_path.join("keyframes");
                let vectors_dir = workspace_path.join("vectors");
                let db_path = workspace_path.join("index.db");

                if keyframes_dir.is_dir() {
                    let _ = std::fs::remove_dir_all(&keyframes_dir);
                }
                if vectors_dir.is_dir() {
                    let _ = std::fs::remove_dir_all(&vectors_dir);
                }
                // 清理 SQLite 的 WAL 和 SHM 文件，确保不留残留
                if db_path.exists() {
                    let _ = std::fs::remove_file(&db_path);
                    let _ = std::fs::remove_file(workspace_path.join("index.db-wal"));
                    let _ = std::fs::remove_file(workspace_path.join("index.db-shm"));
                }
                println!("Index cleared.");
            } else if let Some(video_path) = video {
                let mut storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
                if let Some(v) = storage.workspace_db.get_video_by_path(&video_path)? {
                    // 删除前先获取所有片段 ID，以便清理关联的向量数据
                    let segment_ids = storage.workspace_db.get_segment_ids_by_video(&v.id)?;

                    // 从各向量索引中移除该视频所有片段的向量
                    for seg_id in &segment_ids {
                        storage.face_index.remove(&seg_id.to_string());
                        for index in storage.scene_indices.indices.values_mut() {
                            index.remove(&seg_id.to_string());
                        }
                        // 片段下的帧级图像向量也需清理（ID 格式为 {seg_id}_frame_N）
                        let prefix = format!("{}_frame_", seg_id);
                        let ids_to_remove: Vec<String> = storage.image_index.entries()
                            .filter(|e| e.id.starts_with(&prefix))
                            .map(|e| e.id.clone())
                            .collect();
                        for id in ids_to_remove {
                            storage.image_index.remove(&id);
                        }
                    }
                    storage.face_index.save()?;
                    storage.scene_indices.save()?;
                    storage.image_index.save()?;

                    // 删除关键帧图片目录
                    let keyframe_dir = storage.file_store.base_dir().join("keyframes").join(v.id.to_string());
                    if keyframe_dir.is_dir() {
                        let _ = std::fs::remove_dir_all(&keyframe_dir);
                    }

                    // 数据库删除操作会级联删除关联的 segments 和 detections
                    storage.workspace_db.delete_video(&v.id)?;
                    println!("Removed: {}", video_path);
                } else {
                    anyhow::bail!("Video not found in index: {}", video_path);
                }
            } else {
                anyhow::bail!("Specify a video path or use --all to clear entire index");
            }
        }

        Some(Commands::Clean) => {
            let mut storage = video_scene::storage::Storage::open(&settings, cli.workspace.as_deref())?;
            let videos = storage.workspace_db.list_videos()?;
            let mut cleaned = 0;
            for video in &videos {
                // 源视频文件已不存在时，视为无效条目需清理
                if !std::path::Path::new(&video.path).exists() {
                    let segment_ids = storage.workspace_db.get_segment_ids_by_video(&video.id)?;

                    // 清理向量索引（与 Remove 命令相同的逻辑）
                    for seg_id in &segment_ids {
                        storage.face_index.remove(&seg_id.to_string());
                        for index in storage.scene_indices.indices.values_mut() {
                            index.remove(&seg_id.to_string());
                        }
                        let prefix = format!("{}_frame_", seg_id);
                        let ids_to_remove: Vec<String> = storage.image_index.entries()
                            .filter(|e| e.id.starts_with(&prefix))
                            .map(|e| e.id.clone())
                            .collect();
                        for id in ids_to_remove {
                            storage.image_index.remove(&id);
                        }
                    }

                    // 删除关键帧图片目录
                    let keyframe_dir = storage.file_store.base_dir().join("keyframes").join(video.id.to_string());
                    if keyframe_dir.is_dir() {
                        let _ = std::fs::remove_dir_all(&keyframe_dir);
                    }

                    storage.workspace_db.delete_video(&video.id)?;
                    cleaned += 1;
                }
            }
            // 仅在有清理操作时才保存向量索引，避免无谓的磁盘写入
            if cleaned > 0 {
                storage.face_index.save()?;
                storage.scene_indices.save()?;
                storage.image_index.save()?;
            }
            println!("Cleaned {} invalid entries.", cleaned);
        }

        Some(Commands::Config) => {
            // 将当前生效的完整配置序列化为 TOML 输出，便于检查和调试
            let toml_str = toml::to_string_pretty(&settings)?;
            println!("{}", toml_str);
        }

        Some(Commands::Plugins { action }) => {
            match action {
                PluginAction::Daemon => {
                    // 启动守护进程，进入主循环监听来自客户端的启停指令
                    eprintln!("Starting plugin daemon...");
                    video_scene::plugins::daemon::run()?;
                }
                PluginAction::List => {
                    let statuses = video_scene::plugins::client::daemon_status()?;
                    for s in &statuses {
                        let state = if s.running { "running" } else { "stopped" };
                        println!("{} ({}): {} [idle: {}s / {}s]", s.name, s.plugin_type, state, s.idle_secs, s.idle_timeout);
                    }
                }
                PluginAction::Start { plugin_type } => {
                    video_scene::plugins::client::daemon_start(&plugin_type)?;
                    println!("Plugin {} started", plugin_type);
                }
                PluginAction::Stop { plugin_type } => {
                    video_scene::plugins::client::daemon_stop(&plugin_type)?;
                    println!("Plugin {} stopped", plugin_type);
                }
                PluginAction::Status => {
                    let statuses = video_scene::plugins::client::daemon_status()?;
                    if statuses.is_empty() {
                        println!("No plugins registered");
                    } else {
                        for s in &statuses {
                            let state = if s.running { format!("running (idle {}s)", s.idle_secs) } else { "stopped".to_string() };
                            println!("{:20} {:25} {}", s.name, format!("[{}]", s.plugin_type), state);
                        }
                    }
                }
            }
        }

        None => {
            // 未指定子命令时提示用户查看帮助
            println!("Use --help for usage information");
        }
    }

    Ok(())
}
