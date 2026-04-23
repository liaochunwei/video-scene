//! 客户端：CLI 与守护进程通信的桥梁。
//!
//! 所有函数都是短连接：连接 daemon.sock → 发一个请求 → 读响应 → 断开。
//! 如果守护进程未运行，返回 DaemonNotRunning 错误并提示用户先启动。

use std::io::BufReader;

use super::manager::daemon_socket_path;
use super::protocol::{self, DaemonRequest, DaemonResponse, PluginStatusEntry};

/// 通过守护进程调用插件。连接 daemon.sock，发送 Call 请求，读取响应。
pub fn call_plugin(
    plugin_type: super::types::PluginType,
    action: &str,
    data: &serde_json::Value,
    progress_cb: &dyn Fn(super::manager::ProgressMessage),
) -> crate::error::Result<super::manager::PluginResponse> {
    let socket_path = daemon_socket_path();
    let mut stream = std::os::unix::net::UnixStream::connect(&socket_path)
        .map_err(|e| crate::error::VideoSceneError::DaemonNotRunning(
            format!("Cannot connect to daemon at {}: {}. Run `vs plugins daemon` first.", socket_path.display(), e)
        ))?;

    let request = DaemonRequest::Call {
        plugin_type: plugin_type.to_string(),
        action: action.to_string(),
        data: data.clone(),
    };
    protocol::write_daemon_request(&mut stream, &request)?;

    let mut reader = BufReader::new(stream.try_clone()
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?);
    let response = protocol::read_daemon_response(&mut reader)?;

    match response {
        DaemonResponse::Response { result, progress } => {
            // 将协议层的进度信息转回内部类型，并触发回调
            for p in &progress {
                progress_cb(super::manager::ProgressMessage {
                    id: p.id.clone(),
                    message: p.message.clone(),
                    current: p.current,
                    total: p.total,
                });
            }
            Ok(super::manager::PluginResponse {
                result,
                progress: progress.into_iter().map(|p| super::manager::ProgressMessage {
                    id: p.id,
                    message: p.message,
                    current: p.current,
                    total: p.total,
                }).collect(),
            })
        }
        DaemonResponse::Error { error } => {
            Err(crate::error::VideoSceneError::PluginExecutionError(error))
        }
        other => Err(crate::error::VideoSceneError::PluginExecutionError(
            format!("Unexpected daemon response: {:?}", other)
        )),
    }
}

/// 查询守护进程中所有插件的运行状态。
pub fn daemon_status() -> crate::error::Result<Vec<PluginStatusEntry>> {
    let socket_path = daemon_socket_path();
    let mut stream = std::os::unix::net::UnixStream::connect(&socket_path)
        .map_err(|e| crate::error::VideoSceneError::DaemonNotRunning(
            format!("Cannot connect to daemon at {}: {}. Run `vs plugins daemon` first.", socket_path.display(), e)
        ))?;

    protocol::write_daemon_request(&mut stream, &DaemonRequest::Status)?;

    let mut reader = BufReader::new(stream);
    let response = protocol::read_daemon_response(&mut reader)?;

    match response {
        DaemonResponse::Status { plugins } => Ok(plugins),
        DaemonResponse::Error { error } => Err(crate::error::VideoSceneError::PluginExecutionError(error)),
        other => Err(crate::error::VideoSceneError::PluginExecutionError(
            format!("Unexpected daemon response: {:?}", other)
        )),
    }
}

/// 请求守护进程启动指定插件。
pub fn daemon_start(plugin_type: &str) -> crate::error::Result<()> {
    let socket_path = daemon_socket_path();
    let mut stream = std::os::unix::net::UnixStream::connect(&socket_path)
        .map_err(|e| crate::error::VideoSceneError::DaemonNotRunning(
            format!("Cannot connect to daemon at {}: {}. Run `vs plugins daemon` first.", socket_path.display(), e)
        ))?;

    protocol::write_daemon_request(&mut stream, &DaemonRequest::Start { plugin_type: plugin_type.to_string() })?;

    let mut reader = BufReader::new(stream);
    let response = protocol::read_daemon_response(&mut reader)?;

    match response {
        DaemonResponse::Ok => Ok(()),
        DaemonResponse::Error { error } => Err(crate::error::VideoSceneError::PluginExecutionError(error)),
        other => Err(crate::error::VideoSceneError::PluginExecutionError(
            format!("Unexpected daemon response: {:?}", other)
        )),
    }
}

/// 请求守护进程停止指定插件。
pub fn daemon_stop(plugin_type: &str) -> crate::error::Result<()> {
    let socket_path = daemon_socket_path();
    let mut stream = std::os::unix::net::UnixStream::connect(&socket_path)
        .map_err(|e| crate::error::VideoSceneError::DaemonNotRunning(
            format!("Cannot connect to daemon at {}: {}. Run `vs plugins daemon` first.", socket_path.display(), e)
        ))?;

    protocol::write_daemon_request(&mut stream, &DaemonRequest::Stop { plugin_type: plugin_type.to_string() })?;

    let mut reader = BufReader::new(stream);
    let response = protocol::read_daemon_response(&mut reader)?;

    match response {
        DaemonResponse::Ok => Ok(()),
        DaemonResponse::Error { error } => Err(crate::error::VideoSceneError::PluginExecutionError(error)),
        other => Err(crate::error::VideoSceneError::PluginExecutionError(
            format!("Unexpected daemon response: {:?}", other)
        )),
    }
}

/// 请求守护进程关闭自身。
pub fn daemon_shutdown() -> crate::error::Result<()> {
    let socket_path = daemon_socket_path();
    let mut stream = std::os::unix::net::UnixStream::connect(&socket_path)
        .map_err(|e| crate::error::VideoSceneError::DaemonNotRunning(
            format!("Cannot connect to daemon at {}: {}. Run `vs plugins daemon` first.", socket_path.display(), e)
        ))?;

    protocol::write_daemon_request(&mut stream, &DaemonRequest::Shutdown)?;

    let mut reader = BufReader::new(stream);
    let response = protocol::read_daemon_response(&mut reader)?;

    match response {
        DaemonResponse::Ok => Ok(()),
        DaemonResponse::Error { error } => Err(crate::error::VideoSceneError::PluginExecutionError(error)),
        other => Err(crate::error::VideoSceneError::PluginExecutionError(
            format!("Unexpected daemon response: {:?}", other)
        )),
    }
}
