// VideoSplit 库根模块
// 将所有子模块统一导出，供 main.rs 和集成测试使用
pub mod error;   // 统一错误类型定义
pub mod models;  // 领域数据模型（视频、片段、检测、人脸库）
pub mod config;  // 配置文件与运行时状态
pub mod storage;  // 数据持久化（SQLite、文件存储、向量索引）
pub mod preprocess; // 视频预处理（FFprobe 分析、抽帧、场景检测）
pub mod plugins; // 插件系统（进程间通信、守护进程、各 AI 模型调用）
pub mod core;    // 核心业务逻辑（索引、搜索、人脸提取、计时）
pub mod cli;     // CLI 输出格式化
pub mod backup;  // 备份与导入
pub mod web;     // Web UI 与 REST API
