//! 视频预处理模块
//!
//! 负责视频分析、帧提取、场景检测和图像缩放等前置处理流程。
//! 这些步骤将原始视频转化为可供索引和搜索的结构化数据。

pub mod video_analyzer;
pub mod frame_extractor;
pub mod scene_detector;
pub mod resizer;

// 统一导出核心类型和函数，方便上层模块直接 use preprocess::xxx
pub use video_analyzer::{VideoInfo, analyze_video};
pub use frame_extractor::{ExtractedFrame, extract_frames};
pub use scene_detector::{SceneBoundary, SceneDetector, PySceneDetector, SingleSceneDetector};
pub use resizer::{resize_image, load_image, save_jpeg};
