//! 运行时状态文件
//!
//! 与 `settings` 不同，此文件由应用自动读写，不期望用户手动编辑。
//! 用于持久化跨会话的运行时状态（如当前活跃工作区），以 JSON 格式存储。

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// 应用运行时状态，存储于 config_dir/state.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateFile {
    /// 当前活跃的工作区名称，下次启动时自动恢复
    pub active_workspace: String,
    /// 片段评估耗时（毫秒），用于统计和性能分析；None 表示尚未测量
    pub segment_eval_duration_ms: Option<u64>,
}

impl Default for StateFile {
    fn default() -> Self {
        Self {
            active_workspace: "default".to_string(),
            segment_eval_duration_ms: None,
        }
    }
}

impl StateFile {
    /// 从配置目录加载状态文件；文件不存在或格式错误时返回默认状态（静默降级）
    pub fn load(config_dir: &PathBuf) -> Self {
        let path = config_dir.join("state.json");
        if path.exists() {
            if let Ok(data) = std::fs::read_to_string(&path) {
                if let Ok(state) = serde_json::from_str(&data) {
                    return state;
                }
            }
        }
        // 状态文件缺失或损坏不影响启动，使用默认值即可
        Self::default()
    }

    /// 将当前状态持久化到配置目录
    pub fn save(&self, config_dir: &PathBuf) -> anyhow::Result<()> {
        let path = config_dir.join("state.json");
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, data)?;
        Ok(())
    }
}
