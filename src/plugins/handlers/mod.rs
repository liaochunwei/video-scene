//! 插件处理器：各类 AI 分析能力的具体调用逻辑。
//!
//! 每个 handler 模块封装与特定类型插件的通信协议，
//! 负责构造请求、解析响应、转换为内部数据结构。
//! 调用方无需关心插件进程的细节，直接使用 handler 函数即可。

pub mod face;
pub mod image_text_understanding;
pub mod image_text_vectorization;
pub mod object;
pub mod text_vectorization;
pub mod video_segmentation;
pub mod video_understanding;
