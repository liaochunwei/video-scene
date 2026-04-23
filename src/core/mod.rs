//! 视频处理核心模块
//!
//! 该模块是整个视频索引与检索系统的中枢，将视频分析的各个阶段组织为独立的子模块：
//! - `indexer`: 索引编排，驱动单文件/目录的索引流程
//! - `searcher`: 统一搜索，融合人脸/物体/场景等多路信号
//! - `pipeline`: 视频处理管线，协调分析、切场景、AI 推理、入库等步骤
//! - `face_extractor`: 人脸提取与聚类，从视频中识别并归组人物
//! - `timing`: 处理耗时估算，基于历史数据预测剩余时间

pub mod indexer;
pub mod searcher;
pub mod face_extractor;
pub mod pipeline;
pub mod timing;

// 对外暴露最常用的索引入口函数，供 CLI 或上层调用
pub use indexer::{index_video, index_video_vlm_api, index_directory};
