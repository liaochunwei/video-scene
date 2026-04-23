//! 插件生命周期管理器：发现、注册、启停、空闲回收。
//!
//! PluginManager 维护一个 PluginType → PluginEntry 的映射表：
//! - 启动时扫描 plugins/ 目录下所有 plugin.toml，按类型注册
//! - 调用时自动拉起对应插件进程（懒启动）
//! - 定期检查空闲超时，回收不活跃的进程以释放资源（如 GPU 显存）
//! - 支持额外类型别名：一个插件可同时服务多种类型

use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{Result, VideoSceneError};

use super::config::PluginConfig;
use super::process::PluginProcess;
use super::types::PluginType;

/// 守护进程控制 socket 的文件名
const DAEMON_SOCKET_NAME: &str = "daemon.sock";

/// 返回守护进程控制 socket 的路径（位于系统临时目录下）。
pub fn daemon_socket_path() -> PathBuf {
    std::env::temp_dir().join("vs-plugins").join(DAEMON_SOCKET_NAME)
}

/// 插件上报的进度信息，用于 UI 展示和日志记录。
#[derive(Debug, Clone)]
pub struct ProgressMessage {
    pub id: String,
    pub message: String,
    pub current: usize,
    pub total: usize,
}

/// 插件调用的完整响应，包含结果和所有进度消息。
pub struct PluginResponse {
    pub result: serde_json::Value,
    pub progress: Vec<ProgressMessage>,
}

/// 已注册的插件条目，持有配置和可选的运行中进程。
/// process 为 None 表示尚未启动或已被空闲回收。
struct PluginEntry {
    config: PluginConfig,
    process: Option<PluginProcess>,
}

/// 插件管理器：统筹所有插件的发现与生命周期。
pub struct PluginManager {
    entries: HashMap<PluginType, PluginEntry>,
    socket_dir: PathBuf,
    plugins_dir: PathBuf,
}

impl PluginManager {
    /// 创建管理器并扫描插件目录，注册所有发现的插件。
    pub fn new(plugins_dir: &PathBuf) -> Result<Self> {
        let socket_dir = std::env::temp_dir().join("vs-plugins");
        std::fs::create_dir_all(&socket_dir)
            .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;

        let mut entries = HashMap::new();
        Self::scan_plugins(plugins_dir, &mut entries)?;

        Ok(Self { entries, socket_dir, plugins_dir: plugins_dir.clone() })
    }

    /// 扫描 plugins/*/plugin.toml，按插件类型注册。
    /// 同一类型只注册第一个发现的插件，后续重复的会被忽略并警告。
    fn scan_plugins(
        plugins_dir: &PathBuf,
        entries: &mut HashMap<PluginType, PluginEntry>,
    ) -> Result<()> {
        let dir = std::fs::read_dir(plugins_dir)
            .map_err(|e| VideoSceneError::PluginConfigError(
                format!("Failed to read plugins dir {}: {}", plugins_dir.display(), e)
            ))?;

        for entry in dir {
            let entry = entry.map_err(|e| VideoSceneError::PluginConfigError(e.to_string()))?;
            let toml_path = entry.path().join("plugin.toml");
            if !toml_path.exists() {
                continue; // 非 plugin 目录，跳过
            }

            let config = PluginConfig::load(&toml_path)?;
            let plugin_type = config.plugin.plugin_type;

            // 同类型只保留第一个，防止配置冲突
            if entries.contains_key(&plugin_type) {
                tracing::warn!(
                    "Multiple plugins for type {}: using {}, ignoring {}",
                    plugin_type,
                    entries[&plugin_type].config.plugin.name,
                    config.plugin.name
                );
                continue;
            }

            tracing::info!("Registered plugin {} (type: {})", config.plugin.name, plugin_type);
            entries.insert(plugin_type, PluginEntry {
                config: config.clone(),
                process: None,
            });

            // 注册额外类型别名：一个插件可能同时支持多种类型
            // 例如 CLIP 插件同时服务 ImageTextVectorization 和 TextVectorization
            for (extra_type_str, _action) in &config.extra_types.extra_types {
                if let Ok(extra_type) = Self::parse_plugin_type(extra_type_str) {
                    if !entries.contains_key(&extra_type) {
                        tracing::info!("Plugin {} also serves type {}", config.plugin.name, extra_type);
                        entries.insert(extra_type, PluginEntry {
                            config: config.clone(),
                            process: None,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// 调用插件。如果进程未运行或已崩溃，自动启动。
    /// 懒启动策略：只在真正需要时才消耗资源拉起进程。
    pub fn call(
        &mut self,
        plugin_type: PluginType,
        action: &str,
        data: &serde_json::Value,
        progress_cb: &dyn Fn(ProgressMessage),
    ) -> Result<PluginResponse> {
        let entry = self.entries.get_mut(&plugin_type)
            .ok_or_else(|| VideoSceneError::PluginNotFound(plugin_type.to_string()))?;

        // 确保进程在运行
        if entry.process.is_none() || entry.process.as_mut().map(|p| p.is_alive()) == Some(false) {
            let process = PluginProcess::start(&entry.config, &self.socket_dir, &self.plugins_dir)?;
            entry.process = Some(process);
        }

        entry.process.as_mut().unwrap().call(action, data, progress_cb)
    }

    /// 检查所有插件的空闲时间，超时的进程会被停止并释放资源。
    /// idle_timeout 为 0 表示永不超时（常驻模式）。
    pub fn check_idle(&mut self) {
        for entry in self.entries.values_mut() {
            if let Some(ref process) = entry.process {
                let idle_secs = process.last_used.elapsed().as_secs();
                if idle_secs > entry.config.runtime.idle_timeout && entry.config.runtime.idle_timeout > 0 {
                    tracing::info!("Stopping idle plugin {} ({}s idle, {}s timeout)",
                        entry.config.plugin.name, idle_secs, entry.config.runtime.idle_timeout);
                    if let Some(ref mut process) = entry.process {
                        let _ = process.stop();
                    }
                    entry.process = None;
                }
            }
        }
    }

    /// 停止所有运行中的插件进程，用于守护进程关闭时。
    pub fn shutdown_all(&mut self) {
        for entry in self.entries.values_mut() {
            if let Some(ref mut process) = entry.process {
                tracing::info!("Stopping plugin {}", entry.config.plugin.name);
                let _ = process.stop();
            }
            entry.process = None;
        }
    }

    /// 返回所有已注册插件的运行状态。
    pub fn status(&mut self) -> Vec<PluginStatus> {
        self.entries.values_mut().map(|entry| {
            let (running, idle_secs) = if let Some(ref mut process) = entry.process {
                if process.is_alive() {
                    (true, process.last_used.elapsed().as_secs())
                } else {
                    (false, 0) // 进程已崩溃但尚未清理
                }
            } else {
                (false, 0) // 从未启动
            };
            PluginStatus {
                name: entry.config.plugin.name.clone(),
                plugin_type: entry.config.plugin.plugin_type,
                running,
                idle_secs,
                idle_timeout: entry.config.runtime.idle_timeout,
            }
        }).collect()
    }

    /// 字符串到 PluginType 的解析，用于扫描配置时解析额外类型。
    fn parse_plugin_type(s: &str) -> Result<PluginType> {
        match s {
            "face" => Ok(PluginType::Face),
            "object" => Ok(PluginType::Object),
            "video_understanding" => Ok(PluginType::VideoUnderstanding),
            "video_segmentation" => Ok(PluginType::VideoSegmentation),
            "image_text_understanding" => Ok(PluginType::ImageTextUnderstanding),
            "text_vectorization" => Ok(PluginType::TextVectorization),
            "image_text_vectorization" => Ok(PluginType::ImageTextVectorization),
            other => Err(VideoSceneError::PluginConfigError(format!("Unknown plugin type: {}", other))),
        }
    }

    /// 查找插件目录。
    ///
    /// 查找优先级：
    /// 1. ~/.video-scene/plugins/（全局安装位置）
    /// 2. 从当前目录向上查找包含 plugins/pyproject.toml 的祖先目录（开发模式）
    /// 3. 兜底返回 ./plugins
    pub fn find_plugins_dir() -> Result<PathBuf> {
        let home = directories::BaseDirs::new()
            .map(|b| b.home_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        let config_plugins = home.join(".video-scene").join("plugins");
        if config_plugins.join("pyproject.toml").exists() {
            return Ok(config_plugins);
        }

        // 开发模式：从当前目录往上找
        let mut dir = std::env::current_dir()
            .map_err(|e| VideoSceneError::PluginExecutionError(e.to_string()))?;
        loop {
            let plugins = dir.join("plugins");
            if plugins.join("pyproject.toml").exists() {
                return Ok(plugins);
            }
            match dir.parent() {
                Some(parent) => dir = parent.to_path_buf(),
                None => break,
            }
        }

        Ok(PathBuf::from("./plugins"))
    }
}

/// 插件运行状态信息，用于 status 命令展示。
pub struct PluginStatus {
    pub name: String,
    pub plugin_type: PluginType,
    pub running: bool,
    pub idle_secs: u64,
    pub idle_timeout: u64,
}
