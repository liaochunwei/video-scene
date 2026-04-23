//! 场景检测
//!
//! 定义场景检测的统一接口（SceneDetector trait），并提供两种实现：
//! - PySceneDetector: 调用 Python PySceneDetect 库进行专业级场景切换检测
//! - SingleSceneDetector: 降级策略，将整个视频视为单一场景，用于无 Python 环境时

use std::path::Path;

use crate::error::Result;
use crate::plugins::ProgressMessage;

/// 场景边界，表示一个片段的起止时间（秒）
#[derive(Debug, Clone)]
pub struct SceneBoundary {
    pub start: f32,
    pub end: f32,
}

/// 场景检测算法的统一接口（可插拔设计）。
///
/// 不同的检测策略实现此 trait，上层代码无需关心具体算法细节。
/// 要求 Send + Sync 以便在异步和多线程环境中安全使用。
pub trait SceneDetector: Send + Sync {
    fn detect(&self, video_path: &Path, progress_cb: &dyn Fn(ProgressMessage)) -> Result<Vec<SceneBoundary>>;
}

/// 基于 PySceneDetect 的场景检测器。
///
/// 通过 Python 插件调用 PySceneDetect 库，支持多种检测算法
/// （如 ContentDetector、ThresholdDetector），并可通过 threshold 调节灵敏度。
pub struct PySceneDetector {
    pub detector: String,
    pub threshold: f64,
}

impl PySceneDetector {
    pub fn new(detector: String, threshold: f64) -> Self {
        Self { detector, threshold }
    }
}

impl SceneDetector for PySceneDetector {
    fn detect(&self, video_path: &Path, progress_cb: &dyn Fn(ProgressMessage)) -> Result<Vec<SceneBoundary>> {
        // 委托给 Python 插件层执行实际检测
        crate::plugins::video_segmentation::detect_scenes(
            &video_path.to_string_lossy(),
            &self.detector,
            self.threshold,
            progress_cb,
        )
    }
}

/// 降级检测器：将整个视频视为单一场景。
///
/// 当 Python 环境不可用或 PySceneDetect 未安装时作为后备方案，
/// 确保流程不中断。通过分析视频获取总时长，构造一个覆盖全片的边界。
pub struct SingleSceneDetector;

impl SceneDetector for SingleSceneDetector {
    fn detect(&self, video_path: &Path, _progress_cb: &dyn Fn(ProgressMessage)) -> Result<Vec<SceneBoundary>> {
        let info = super::video_analyzer::analyze_video(video_path)?;
        Ok(vec![SceneBoundary {
            start: 0.0,
            end: info.duration,
        }])
    }
}
