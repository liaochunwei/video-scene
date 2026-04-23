//! 处理耗时估算模块
//!
//! 基于历史处理耗时数据，使用指数移动平均（EMA）估算剩余处理时间。
//! 维护两种统计维度：
//! - 按片段数（本地管线模式）：处理时间与场景片段数正相关
//! - 按视频时长（VLM API 模式）：处理时间与视频总时长正相关
//!
//! EMA 的优势在于：对最新观测值给予更高权重，能快速适应硬件或模型变化，
//! 同时不完全遗忘历史，避免单次异常值剧烈影响估计。

use std::path::Path;

use crate::error::{Result, VideoSceneError};

const TIMING_FILE: &str = "timing.json";

/// 处理耗时统计数据，持久化到索引目录下的 timing.json。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimingStats {
    /// 每个片段的平均处理耗时（秒），不含模型加载时间
    pub secs_per_segment: f64,
    /// 计算上述平均值所用的样本数
    pub sample_count: u64,
    /// 每秒视频的平均处理耗时（秒），用于 VLM 模式
    pub secs_per_video_second: f64,
    /// 计算上述平均值所用的样本数
    pub video_sample_count: u64,
}

/// 从索引目录加载耗时统计数据。文件不存在或解析失败时返回 None。
pub fn load_timing(index_dir: &Path) -> Option<TimingStats> {
    let path = index_dir.join(TIMING_FILE);
    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// 将耗时统计数据持久化到索引目录。
pub fn save_timing(index_dir: &Path, stats: &TimingStats) -> Result<()> {
    let path = index_dir.join(TIMING_FILE);
    let data = serde_json::to_string_pretty(stats)
        .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
    std::fs::write(&path, data)
        .map_err(|e| VideoSceneError::StorageError(e.to_string()))?;
    Ok(())
}

/// 用新的处理观测值更新片段模式耗时统计。
///
/// 使用指数移动平均（alpha=0.3）融合新旧数据：
/// - alpha=0.3 意味着新观测占 30% 权重，历史占 70%
/// - 比简单平均更能反映近期趋势，又比纯最新值更平滑
pub fn update_timing(index_dir: &Path, elapsed_secs: f64, segment_count: usize) -> Result<()> {
    if segment_count == 0 {
        return Ok(());
    }
    let observed = elapsed_secs / segment_count as f64;

    let new_stats = match load_timing(index_dir) {
        Some(existing) => {
            let alpha = 0.3;
            // EMA 更新：新值 = alpha * 观测值 + (1-alpha) * 历史值
            let new_rate = alpha * observed + (1.0 - alpha) * existing.secs_per_segment;
            TimingStats {
                secs_per_segment: new_rate,
                sample_count: existing.sample_count + 1,
                secs_per_video_second: existing.secs_per_video_second,
                video_sample_count: existing.video_sample_count,
            }
        }
        // 首次记录，直接使用观测值
        None => TimingStats {
            secs_per_segment: observed,
            sample_count: 1,
            secs_per_video_second: 0.0,
            video_sample_count: 0,
        },
    };

    save_timing(index_dir, &new_stats)
}

/// 用新的处理观测值更新 VLM 模式耗时统计。
///
/// VLM 模式下处理耗时与视频时长线性相关（云端按视频时长计费/处理），
/// 因此统计维度是"每秒视频需要的处理秒数"。
pub fn update_timing_vlm(index_dir: &Path, elapsed_secs: f64, video_duration_secs: f64) -> Result<()> {
    if video_duration_secs <= 0.0 {
        return Ok(());
    }
    let observed = elapsed_secs / video_duration_secs;

    let new_stats = match load_timing(index_dir) {
        Some(existing) => {
            let alpha = 0.3;
            // 首次记录 VLM 耗时时直接用观测值，跳过 EMA
            // 避免用 0 作为历史值导致首次估计偏低
            let new_rate = if existing.video_sample_count == 0 {
                observed
            } else {
                alpha * observed + (1.0 - alpha) * existing.secs_per_video_second
            };
            TimingStats {
                secs_per_segment: existing.secs_per_segment,
                sample_count: existing.sample_count,
                secs_per_video_second: new_rate,
                video_sample_count: existing.video_sample_count + 1,
            }
        }
        None => TimingStats {
            secs_per_segment: 0.0,
            sample_count: 0,
            secs_per_video_second: observed,
            video_sample_count: 1,
        },
    };

    save_timing(index_dir, &new_stats)
}

/// 根据待处理片段数估算剩余时间（本地管线模式）。
///
/// 返回人类可读的格式化字符串（如 "~2m 30s"），无数据时返回空串。
pub fn estimate_remaining(index_dir: &Path, segment_count: usize) -> String {
    if segment_count == 0 {
        return String::new();
    }
    match load_timing(index_dir) {
        Some(stats) if stats.sample_count > 0 => {
            let secs = stats.secs_per_segment * segment_count as f64;
            format_duration(secs)
        }
        _ => String::new(),
    }
}

/// 根据视频时长估算剩余时间（VLM 模式）。
pub fn estimate_remaining_vlm(index_dir: &Path, video_duration_secs: f64) -> String {
    if video_duration_secs <= 0.0 {
        return String::new();
    }
    match load_timing(index_dir) {
        Some(stats) if stats.video_sample_count > 0 => {
            let secs = stats.secs_per_video_second * video_duration_secs;
            format_duration(secs)
        }
        _ => String::new(),
    }
}

/// 将秒数格式化为人类可读的时间字符串。
///
/// 规则：
/// - < 1s：不显示（估算太短无意义）
/// - < 60s：显示秒，如 "~45s"
/// - < 1h：显示分+秒，如 "~2m 30s"
/// - >= 1h：显示时+分，如 "~1h 23m"
fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        String::new()
    } else if secs < 60.0 {
        format!("~{}s", secs.round() as u32)
    } else {
        let m = (secs / 60.0).floor() as u32;
        let s = (secs % 60.0).round() as u32;
        if m < 60 {
            format!("~{}m {}s", m, s)
        } else {
            let h = m / 60;
            let rm = m % 60;
            format!("~{}h {}m", h, rm)
        }
    }
}
