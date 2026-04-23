//! IPC 通信协议：定义主程序与插件进程、CLI 与守护进程之间的消息格式。
//!
//! 传输层：Unix Socket
//! 序列化：NDJSON（每行一个 JSON 对象），简单可靠，方便 Python 端逐行解析。
//!
//! 两套协议各管各的通道：
//! 1. PluginRequest / PluginMessage — 主程序 ↔ 插件进程（每个插件一个 socket）
//! 2. DaemonRequest / DaemonResponse — CLI ↔ 守护进程（daemon.sock）

use serde::{Deserialize, Serialize};

// ── 主程序 → 插件进程 的请求消息 ──

/// 主程序发给插件进程的消息。
///
/// - Request：调用插件的某个 action（如 "detect"、"encode"）
/// - Ping：心跳探测
/// - Shutdown：通知插件优雅退出
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginRequest {
    Request {
        id: String,
        action: String,
        data: serde_json::Value,
    },
    Ping,
    Shutdown,
}

// ── 插件进程 → 主程序 的响应消息 ──

/// 插件进程返回给主程序的消息。
///
/// - Register：插件启动后首先发送，声明自己的类型和支持的 action
/// - Registered：主程序确认注册成功
/// - Progress：长时间任务的进度回调（如逐帧检测）
/// - Response：最终结果
/// - Error：插件内部错误
/// - Pong：心跳应答
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginMessage {
    Register {
        plugin_type: String,
        actions: Vec<String>,
    },
    Registered {
        plugin_type: String,
    },
    Progress {
        id: String,
        message: String,
        current: usize,
        total: usize,
    },
    Response {
        id: String,
        data: serde_json::Value,
    },
    Error {
        id: String,
        error: String,
    },
    Pong,
}

/// 向 Unix Socket 写入一条 NDJSON 消息。
/// 每条消息占一行（writeln!），flush 确保对方立即收到。
pub fn write_message(
    stream: &mut std::os::unix::net::UnixStream,
    msg: &impl Serialize,
) -> crate::error::Result<()> {
    let line = serde_json::to_string(msg)
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?;
    use std::io::Write;
    writeln!(stream, "{}", line)
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
            format!("Failed to write to socket: {}", e)
        ))?;
    stream.flush()
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
            format!("Failed to flush socket: {}", e)
        ))?;
    Ok(())
}

/// 从 BufReader 读取一条 NDJSON 消息。
/// 跳过空行（协议允许留空行做分隔），读到 EOF 视为连接断开。
pub fn read_message(
    reader: &mut impl std::io::BufRead,
) -> crate::error::Result<PluginMessage> {
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                format!("Failed to read from socket: {}", e)
            ))?;
        if bytes == 0 {
            return Err(crate::error::VideoSceneError::PluginExecutionError(
                "Socket closed".into()
            ));
        }
        let line = line.trim();
        if line.is_empty() {
            continue; // 跳过空行
        }
        let msg: PluginMessage = serde_json::from_str(line)
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                format!("Invalid JSON from plugin: {} (line: {})", e, line)
            ))?;
        return Ok(msg);
    }
}

// ── CLI → 守护进程 的请求消息 ──

/// CLI 发给守护进程的控制命令。
///
/// - Call：转发插件调用请求
/// - Status：查询所有插件运行状态
/// - Start / Stop：手动启停插件
/// - Shutdown：关闭守护进程
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonRequest {
    Call {
        plugin_type: String,
        action: String,
        data: serde_json::Value,
    },
    Status,
    Start {
        plugin_type: String,
    },
    Stop {
        plugin_type: String,
    },
    Shutdown,
}

// ── 守护进程 → CLI 的响应消息 ──

/// 守护进程返回给 CLI 的响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonResponse {
    Response {
        result: serde_json::Value,
        progress: Vec<ProgressEntry>,
    },
    Status {
        plugins: Vec<PluginStatusEntry>,
    },
    Ok,
    Error {
        error: String,
    },
}

/// 插件任务的进度信息，随 DaemonResponse::Response 一起返回。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEntry {
    pub id: String,
    pub message: String,
    pub current: usize,
    pub total: usize,
}

/// 单个插件的运行状态，用于 `vs plugins status` 展示。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginStatusEntry {
    pub name: String,
    pub plugin_type: String,
    pub running: bool,
    pub idle_secs: u64,
    pub idle_timeout: u64,
}

/// 向守护进程控制 socket 写入请求（NDJSON）。
pub fn write_daemon_request(
    stream: &mut std::os::unix::net::UnixStream,
    msg: &DaemonRequest,
) -> crate::error::Result<()> {
    let line = serde_json::to_string(msg)
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?;
    use std::io::Write;
    writeln!(stream, "{}", line)
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
            format!("Failed to write to daemon socket: {}", e)
        ))?;
    stream.flush()
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
            format!("Failed to flush daemon socket: {}", e)
        ))?;
    Ok(())
}

/// 从守护进程控制 socket 读取响应（NDJSON）。
pub fn read_daemon_response(
    reader: &mut impl std::io::BufRead,
) -> crate::error::Result<DaemonResponse> {
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                format!("Failed to read from daemon socket: {}", e)
            ))?;
        if bytes == 0 {
            return Err(crate::error::VideoSceneError::PluginExecutionError(
                "Daemon socket closed".into()
            ));
        }
        let line = line.trim();
        if line.is_empty() {
            continue; // 跳过空行
        }
        let msg: DaemonResponse = serde_json::from_str(line)
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                format!("Invalid JSON from daemon: {} (line: {})", e, line)
            ))?;
        return Ok(msg);
    }
}
