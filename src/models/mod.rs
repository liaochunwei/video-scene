// 领域模型模块
// 定义了视频、片段、检测结果、人脸库等核心数据结构，
// 这些结构贯穿整个应用，被存储层、核心逻辑层和插件层共同使用。

pub mod video;    // 视频元数据
pub mod segment;  // 视频片段（按场景或人脸切分）
pub mod detection; // AI 检测结果（人脸、物体、文字等）
pub mod face_library; // 人脸库（已知人物的向量集合）

pub use video::Video;
pub use segment::Segment;
pub use detection::{Detection, DetectionType, BoundingBox};
pub use face_library::FaceLibraryEntry;
