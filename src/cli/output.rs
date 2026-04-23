//! 终端输出格式化
//!
//! 将搜索结果、人脸信息和索引状态等数据格式化为多种终端友好格式。
//! 支持 Table（对齐表格）、JSON（机器可读）和 Simple（单行摘要）三种输出模式。

use crate::core::searcher::SearchResult;
use crate::core::face_extractor::ExtractedPerson;
use crate::models::Video;

/// 输出格式枚举
pub enum OutputFormat {
    Table,
    Json,
    Simple,
}

impl From<&str> for OutputFormat {
    fn from(s: &str) -> Self {
        match s {
            "json" => OutputFormat::Json,
            "simple" => OutputFormat::Simple,
            _ => OutputFormat::Table, // 默认表格格式，最易阅读
        }
    }
}

/// 根据指定格式输出搜索结果
pub fn format_search_results(results: &[SearchResult], format: OutputFormat) -> String {
    match format {
        OutputFormat::Json => format_search_json(results),
        OutputFormat::Table => format_search_table(results),
        OutputFormat::Simple => format_search_simple(results),
    }
}

/// 表格格式：显示置信度区间、匹配信号、时间范围和文件路径
fn format_search_table(results: &[SearchResult]) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{:<10} {:<12} {:<10} {}\n",
        "Score", "Signals", "Time", "Path"
    ));
    output.push_str(&"-".repeat(100));
    output.push('\n');

    for r in results {
        output.push_str(&format!(
            "{:<10} {:<12} {:<10} {}\n",
            format!("{:.2} [{:.2}-{:.2}]", r.confidence, r.confidence_low, r.confidence_high),
            r.match_type,
            r.best_segment.format_time_range(),
            r.video.path,
        ));
        // 场景描述放在下一行，便于阅读完整文本
        if !r.best_segment.scene_description.is_empty() {
            output.push_str(&r.best_segment.scene_description);
            output.push('\n');
        }
        output.push('\n');
    }

    output
}

/// JSON 格式：适合管道处理和脚本消费，使用 pretty-print 提高可读性
fn format_search_json(results: &[SearchResult]) -> String {
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            // 附加匹配片段信息
            let more: Vec<serde_json::Value> = r.more.iter().map(|m| {
                serde_json::json!({
                    "segment_id": m.segment_id.to_string(),
                    "start_time": m.start_time,
                    "end_time": m.end_time,
                    "confidence": m.confidence,
                    "keyframe": m.keyframe_path,
                })
            }).collect();
            serde_json::json!({
                "video": r.video.path,
                "filename": r.video.filename,
                "time_range": r.best_segment.format_time_range(),
                "start_time": r.best_segment.start_time,
                "end_time": r.best_segment.end_time,
                "keyframe": r.best_segment.keyframe_path,
                "match_type": r.match_type,
                "confidence": r.confidence,
                "confidence_low": r.confidence_low,
                "confidence_high": r.confidence_high,
                "description": r.best_segment.scene_description,
                "more": more,
            })
        })
        .collect();

    serde_json::to_string_pretty(&json_results).unwrap_or_default()
}

/// 单行格式：每条结果一行，适合 grep 等工具进一步处理
fn format_search_simple(results: &[SearchResult]) -> String {
    results
        .iter()
        .map(|r| format!("{}:{} {} ({:.2} [{:.2}-{:.2}])", r.video.path, r.best_segment.format_time_range(), r.match_type, r.confidence, r.confidence_low, r.confidence_high))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 格式化提取出的人脸信息，包含出现次数和质量评分。
/// 末尾提示用户输入姓名以保存到人脸库（交互式 CLI 场景使用）。
pub fn format_extracted_faces(persons: &[ExtractedPerson]) -> String {
    let mut output = String::new();
    for (i, person) in persons.iter().enumerate() {
        output.push_str(&format!(
            "\nPerson #{} (appears {} times, quality: {:.2})\n",
            i + 1,
            person.appearance_count,
            person.quality
        ));
        output.push_str(&format!("  Best frame at: {:.2}s\n", person.best_timestamp));
        output.push_str("  Save to library? Enter name (or press Enter to skip):\n");
    }
    output
}

/// 格式化索引状态摘要，显示工作区信息和已索引数据量
pub fn format_status(
    video_count: u64,
    segment_count: u64,
    face_count: usize,
    workspace_path: &str,
    workspace_name: &str,
) -> String {
    format!(
        "VideoScene Index Status\n\
         =======================\n\
         Workspace:       {}\n\
         Workspace path:  {}\n\
         Videos indexed:  {}\n\
         Segments:        {}\n\
         Faces in library: {}\n",
        workspace_name, workspace_path, video_count, segment_count, face_count
    )
}

/// 格式化视频列表，以对齐表格显示 UUID、文件名、时长、分辨率和路径
pub fn format_video_list(videos: &[Video]) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{:<38} {:<40} {:<8} {:<10} {}\n",
        "ID", "Filename", "Duration", "Resolution", "Path"
    ));
    output.push_str(&"-".repeat(140));
    output.push('\n');

    for v in videos {
        output.push_str(&format!(
            "{:<38} {:<40} {:<8} {:<10} {}\n",
            truncate(&v.id.to_string(), 36),
            truncate(&v.filename, 38),
            format!("{:.1}s", v.duration),
            format!("{}x{}", v.width, v.height),
            v.path,
        ));
    }

    output
}

/// 截断过长字符串，保留尾部内容并加 "..." 前缀。
///
/// 保留尾部而非头部，因为文件名末尾通常更具区分性（如扩展名和编号）。
fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        // 从末尾取 max_len-3 个字符，前面加 "..." 表示省略
        let suffix: String = s.chars().rev().take(max_len - 3).collect::<Vec<_>>().into_iter().rev().collect();
        format!("...{}", suffix)
    }
}
