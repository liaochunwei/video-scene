//! 命令行界面模块
//!
//! 提供搜索结果、人脸提取、索引状态等信息的终端输出格式化。

pub mod output;

pub use output::{OutputFormat, format_search_results, format_extracted_faces, format_status};
