//! 应用设置（TOML 配置文件）
//!
//! 定义所有用户可配置的参数，支持从 TOML 文件加载、部分覆盖默认值、以及导出默认配置。
//! 设计原则：用户只需在配置文件中写想修改的字段，其余自动回退到默认值。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{Result, VideoSceneError};

/// 应用顶层设置，对应 TOML 配置文件的完整结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub video: VideoSettings,
    pub index: IndexSettings,
    pub search: SearchSettings,
    pub plugins: PluginSettings,
    pub logging: LoggingSettings,
}

/// 视频处理相关配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoSettings {
    /// 支持的视频文件扩展名，不在列表中的文件会被跳过
    pub supported_extensions: Vec<String>,
    /// 最大并行处理任务数，受 CPU 核心数和内存限制
    pub max_parallel_jobs: usize,
    pub preprocessing: PreprocessingSettings,
    pub scene_detection: SceneDetectionSettings,
}

/// 视频预处理参数：在送入模型前统一缩放和压缩，减少计算开销
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingSettings {
    /// 缩放后短边像素数，长边按比例缩放，保持宽高比
    pub target_short_edge: u32,
    /// JPEG 编码质量 (1-100)，越高越清晰但文件更大
    pub frame_quality: u8,
}

/// 镜头切换检测参数，决定如何将视频切分为场景片段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDetectionSettings {
    /// 检测算法名称，"content" 表示基于帧间内容差异的检测
    pub detector: String,
    /// 检测灵敏度阈值，值越低越容易触发场景切换（适合变化细微的视频）
    pub threshold: f64,
}

/// 索引存储与各类索引的参数配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSettings {
    /// 索引文件和状态文件的存储根目录
    pub config_dir: PathBuf,
    pub face: FaceIndexSettings,
    pub object: ObjectIndexSettings,
    pub scene: SceneIndexSettings,
}

/// 人脸索引参数：过滤低置信度和低质量的人脸，避免索引无效数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceIndexSettings {
    /// 人脸检测最低置信度，低于此值的人脸框被丢弃
    pub min_confidence: f64,
    /// 人脸质量评分下限，过滤模糊/侧脸等不可用人脸
    pub min_quality: f64,
    /// 人脸特征相似度阈值，高于此值视为同一人（用于人脸聚类）
    pub similarity_threshold: f64,
}

/// 物体检测索引参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectIndexSettings {
    /// 物体检测最低置信度，过滤低可靠性的检测结果
    pub min_confidence: f64,
}

/// 场景索引参数：控制场景片段的粒度，影响 VLM 描述的精度和成本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneIndexSettings {
    /// 场景特征向量相似度阈值，用于判断两个片段是否属于同一场景
    pub similarity_threshold: f64,
    /// 最短片段时长（秒）。过短的片段会与前一片段合并，
    /// 避免对极短片段浪费 VLM 调用开销
    pub min_segment_duration: f32,
    /// 最长片段时长（秒）。过长的片段会被拆分为子片段，
    /// 以确保 VLM 输入可控、描述聚焦
    pub max_segment_duration: f32,
}

/// 搜索行为默认参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSettings {
    /// 未指定 limit 时的默认返回条数
    pub default_limit: usize,
    /// 默认输出格式（如 "table"、"json" 等）
    pub default_format: String,
}

/// 插件运行环境与模型路径配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSettings {
    /// Python 解释器路径，插件运行依赖此环境
    pub python_path: String,
    /// uv 包管理器路径，用于自动安装插件依赖
    pub uv_path: String,
    /// 插件目录，存放各插件子目录
    pub plugins_dir: PathBuf,
    pub models: ModelSettings,
    pub vlm_api: VlmApiConfig,
}

/// 各类视觉模型的标识名称
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSettings {
    /// InsightFace 人脸识别模型
    pub insightface_model: String,
    /// YOLO 目标检测/分割模型
    pub yolo_model: String,
    /// CLIP 图文匹配模型
    pub clip_model: String,
}

/// VLM（视觉语言模型）API 调用配置，用于生成场景文字描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmApiConfig {
    /// API 基础地址，默认使用阿里云 DashScope 兼容接口
    pub api_base: String,
    /// API 密钥，空字符串表示未配置，需要用户自行填写
    pub api_key: String,
    /// 模型名称
    pub model: String,
    /// 单帧最大像素数，限制送入 VLM 的图片尺寸以控制 token 消耗
    pub max_pixels: u32,
    /// 抽帧帧率，从视频中按此频率截取帧送入 VLM
    pub fps: f64,
}

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// 日志级别（trace/debug/info/warn/error）
    pub level: String,
    /// 日志文件输出路径
    pub file: PathBuf,
}

/// 提供所有配置项的合理默认值，确保无配置文件时应用仍可正常运行
impl Default for Settings {
    fn default() -> Self {
        // 优先使用用户主目录，回退到 /tmp（如无 HOME 环境变量）
        let home = directories::BaseDirs::new()
            .map(|b| b.home_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        let index_dir = home.join(".video-scene");

        Self {
            video: VideoSettings {
                supported_extensions: vec![
                    "mp4".into(),
                    "mov".into(),
                    "mkv".into(),
                    "avi".into(),
                    "webm".into(),
                ],
                max_parallel_jobs: 4,
                preprocessing: PreprocessingSettings {
                    target_short_edge: 640,
                    frame_quality: 85,
                },
                scene_detection: SceneDetectionSettings {
                    detector: "content".into(),
                    threshold: 27.0,
                },
            },
            index: IndexSettings {
                config_dir: index_dir.clone(),
                face: FaceIndexSettings {
                    min_confidence: 0.8,
                    min_quality: 0.5,
                    similarity_threshold: 0.7,
                },
                object: ObjectIndexSettings {
                    min_confidence: 0.5,
                },
                scene: SceneIndexSettings {
                    similarity_threshold: 0.7,
                    min_segment_duration: 2.0,
                    max_segment_duration: 30.0,
                },
            },
            search: SearchSettings {
                default_limit: 10,
                default_format: "table".into(),
            },
            plugins: PluginSettings {
                python_path: "python3".into(),
                uv_path: "uv".into(),
                plugins_dir: PathBuf::from("./plugins"),
                models: ModelSettings {
                    insightface_model: "buffalo_l".into(),
                    yolo_model: "yoloe-26s-seg.pt".into(),
                    clip_model: "ViT-B-16".into(),
                },
                vlm_api: VlmApiConfig {
                    api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1".into(),
                    api_key: String::new(),
                    model: "qwen3.6-plus".into(),
                    max_pixels: 230400,
                    fps: 2.0,
                },
            },
            logging: LoggingSettings {
                level: "info".into(),
                file: index_dir.join("video-scene.log"),
            },
        }
    }
}

impl Settings {
    /// 从配置文件加载设置，支持三种加载路径：
    /// 1. 指定路径 → 必须存在，否则报错
    /// 2. 未指定 → 尝试默认路径 ~/.video-scene/config.toml
    /// 3. 均不存在 → 使用内置默认值
    pub fn load(config_path: Option<&PathBuf>) -> Result<Self> {
        let settings = Self::default();

        if let Some(path) = config_path {
            // 用户显式指定了配置路径，文件不存在视为错误
            if !path.exists() {
                return Err(VideoSceneError::ConfigNotFound(path.clone()));
            }
            let config_str = std::fs::read_to_string(path)
                .map_err(|e| VideoSceneError::ConfigInvalid(e.to_string()))?;
            return Self::load_from_str(&settings, &config_str);
        } else {
            // 未指定路径时，自动在用户主目录下查找默认配置
            let home = directories::BaseDirs::new()
                .map(|b| b.home_dir().to_path_buf())
                .unwrap_or_else(|| PathBuf::from("/tmp"));
            let default_config = home.join(".video-scene").join("config.toml");
            if default_config.exists() {
                let config_str = std::fs::read_to_string(&default_config)
                    .map_err(|e| VideoSceneError::ConfigInvalid(e.to_string()))?;
                return Self::load_from_str(&settings, &config_str);
            }
        }

        // 没有任何配置文件，使用纯默认值
        Ok(settings)
    }

    /// 解析 TOML 配置并与默认值合并，支持部分配置（只写想改的字段）。
    /// 策略：先尝试完整解析，失败则用 TOML Value 递归合并默认值。
    fn load_from_str(defaults: &Self, config_str: &str) -> Result<Self> {
        // 快速路径：配置文件字段完整，直接反序列化即可
        if let Ok(loaded) = toml::from_str::<Settings>(config_str) {
            return Ok(loaded);
        }
        // 慢速路径：配置文件只包含部分字段，需要与默认值合并
        let overrides: toml::Value = toml::from_str(config_str)
            .map_err(|e| VideoSceneError::ConfigInvalid(e.to_string()))?;
        let base = toml::Value::try_from(defaults)
            .map_err(|e| VideoSceneError::ConfigInvalid(e.to_string()))?;
        let merged = Self::merge_toml(base, overrides);
        // 合并后重新序列化再反序列化，确保类型校验通过
        let merged_str = toml::to_string(&merged)
            .map_err(|e| VideoSceneError::ConfigInvalid(e.to_string()))?;
        let loaded: Settings = toml::from_str(&merged_str)
            .map_err(|e| VideoSceneError::ConfigInvalid(e.to_string()))?;
        Ok(loaded)
    }

    /// 递归合并 TOML 值：将 overrides 中的键覆盖到 base 中。
    /// 对于嵌套表（table），递归合并而非整表替换，保证只改用户指定的字段。
    fn merge_toml(mut base: toml::Value, overrides: toml::Value) -> toml::Value {
        if let (Some(base_table), Some(overrides_table)) = (base.as_table_mut(), overrides.as_table()) {
            for (key, value) in overrides_table {
                if let Some(existing) = base_table.get_mut(key) {
                    // 两边都是表时递归合并，保留 base 中未被覆盖的子字段
                    if existing.is_table() && value.is_table() {
                        *existing = Self::merge_toml(existing.clone(), value.clone());
                        continue;
                    }
                }
                // 非表类型或 base 中不存在该键，直接覆盖/插入
                base_table.insert(key.clone(), value.clone());
            }
        }
        base
    }

    /// 将默认配置导出为 TOML 文件，供用户作为配置模板修改
    pub fn save_default(path: &PathBuf) -> Result<()> {
        let parent = path.parent().unwrap_or(path);
        // 自动创建配置文件所在的目录结构
        std::fs::create_dir_all(parent)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        let toml_str = toml::to_string_pretty(&Self::default())
            .map_err(|e| VideoSceneError::ConfigInvalid(e.to_string()))?;
        std::fs::write(path, toml_str)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
        Ok(())
    }
}
