//! 插件守护进程：长期运行的后台服务，管理所有插件进程的生命周期。
//!
//! 守护进程监听一个 Unix Socket（daemon.sock），接受 CLI 的控制命令。
//! 使用 poll + 超时实现非阻塞 accept，这样可以周期性检查空闲超时和关闭信号。
//!
//! 典型使用流程：`vs plugins daemon` 启动 → 其他命令自动通过 daemon.sock 转发调用。

use std::io::{BufRead, BufReader};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::UnixListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::manager::{daemon_socket_path, PluginManager};
use super::protocol::{DaemonRequest, DaemonResponse, PluginStatusEntry};

/// 运行守护进程主循环，阻塞直到收到 Shutdown 命令。
pub fn run() -> crate::error::Result<()> {
    let socket_path = daemon_socket_path();

    // 确保 socket 目录存在
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
    }

    // 清理上次异常退出残留的 socket 文件
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    // 绑定监听地址
    let listener = UnixListener::bind(&socket_path)
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
            format!("Failed to bind daemon socket at {}: {}", socket_path.display(), e)
        ))?;

    listener.set_nonblocking(false)
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?;

    // 扫描并注册所有插件
    let plugins_dir = PluginManager::find_plugins_dir()?;
    let mut manager = PluginManager::new(&plugins_dir)?;

    tracing::info!("Plugin daemon listening on {}", socket_path.display());

    // 关闭信号标志，通知后台线程退出
    let shutting_down = Arc::new(AtomicBool::new(false));
    let idle_shutdown = shutting_down.clone();
    std::thread::spawn(move || {
        // 此线程仅用于配合 shutting_down 标志，
        // 真正的空闲检查在主循环里完成
        while !idle_shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_secs(30));
            if idle_shutdown.load(Ordering::Relaxed) {
                break;
            }
        }
    });

    // 主循环：用 poll 带超时地等待连接，超时后检查空闲和关闭
    let mut should_shutdown = false;
    while !should_shutdown {
        // 每轮循环先检查空闲超时
        manager.check_idle();

        // poll 等待新连接，500ms 超时以便及时响应关闭信号
        let fd = listener.as_raw_fd();
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let timeout_ms = 500;
        let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue; // 被信号中断，重试
            }
            let _ = std::fs::remove_file(&socket_path);
            return Err(crate::error::VideoSceneError::PluginExecutionError(
                format!("poll failed: {}", err)
            ));
        }
        if n == 0 {
            // 超时，回到循环顶部检查空闲和关闭
            continue;
        }

        // poll 返回可读，accept 不会阻塞
        let (mut client, _addr) = listener.accept()
            .map_err(|e| {
                let _ = std::fs::remove_file(&socket_path);
                crate::error::VideoSceneError::PluginExecutionError(
                    format!("Failed to accept daemon connection: {}", e)
                )
            })?;

        // 读取一行 NDJSON 请求
        let mut reader = BufReader::new(client.try_clone()
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?);

        let request: DaemonRequest = {
            let mut line = String::new();
            let bytes = reader.read_line(&mut line)
                .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                    format!("Failed to read from client: {}", e)
                ))?;
            if bytes == 0 {
                continue; // 客户端断开，忽略
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            serde_json::from_str(line)
                .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                    format!("Invalid JSON from client: {} (line: {})", e, line)
                ))?
        };

        // 分发请求
        let response = match request {
            DaemonRequest::Call { plugin_type, action, data } => {
                let pt = match parse_plugin_type(&plugin_type) {
                    Ok(pt) => pt,
                    Err(e) => {
                        write_response(&mut client, &DaemonResponse::Error { error: e.to_string() })?;
                        continue;
                    }
                };
                match manager.call(pt, &action, &data, &|_| {}) {
                    Ok(resp) => {
                        // 将内部 ProgressMessage 转为协议层的 ProgressEntry
                        let progress: Vec<super::protocol::ProgressEntry> = resp.progress.into_iter()
                            .map(|p| super::protocol::ProgressEntry {
                                id: p.id,
                                message: p.message,
                                current: p.current,
                                total: p.total,
                            })
                            .collect();
                        DaemonResponse::Response { result: resp.result, progress }
                    }
                    Err(e) => DaemonResponse::Error { error: e.to_string() },
                }
            }
            DaemonRequest::Status => {
                let statuses: Vec<PluginStatusEntry> = manager.status().into_iter()
                    .map(|s| PluginStatusEntry {
                        name: s.name,
                        plugin_type: s.plugin_type.to_string(),
                        running: s.running,
                        idle_secs: s.idle_secs,
                        idle_timeout: s.idle_timeout,
                    })
                    .collect();
                DaemonResponse::Status { plugins: statuses }
            }
            DaemonRequest::Start { plugin_type } => {
                // 通过发 ping 请求来触发插件进程启动
                let pt = match parse_plugin_type(&plugin_type) {
                    Ok(pt) => pt,
                    Err(e) => {
                        write_response(&mut client, &DaemonResponse::Error { error: e.to_string() })?;
                        continue;
                    }
                };
                match manager.call(pt, "ping", &serde_json::json!({}), &|_| {}) {
                    Ok(_) => DaemonResponse::Ok,
                    Err(e) => DaemonResponse::Error { error: e.to_string() },
                }
            }
            DaemonRequest::Stop { plugin_type } => {
                // Stop 暂未实现，直接返回 Ok
                let _ = plugin_type;
                DaemonResponse::Ok
            }
            DaemonRequest::Shutdown => {
                should_shutdown = true;
                DaemonResponse::Ok
            }
        };

        write_response(&mut client, &response)?;
    }

    // 关闭阶段：通知后台线程，停止所有插件
    shutting_down.store(true, Ordering::Relaxed);
    manager.shutdown_all();

    // 清理 socket 文件
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    tracing::info!("Plugin daemon shut down");
    Ok(())
}

/// 向客户端写入 NDJSON 响应。
fn write_response(
    stream: &mut std::os::unix::net::UnixStream,
    response: &DaemonResponse,
) -> crate::error::Result<()> {
    let line = serde_json::to_string(response)
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?;
    use std::io::Write;
    writeln!(stream, "{}", line)
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
            format!("Failed to write to client: {}", e)
        ))?;
    stream.flush()
        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
            format!("Failed to flush client: {}", e)
        ))?;
    Ok(())
}

/// 将字符串解析为 PluginType 枚举。
/// 与 config.rs 中的解析逻辑一致，守护进程侧也需要独立解析。
fn parse_plugin_type(s: &str) -> crate::error::Result<super::types::PluginType> {
    match s {
        "face" => Ok(super::types::PluginType::Face),
        "object" => Ok(super::types::PluginType::Object),
        "video_understanding" => Ok(super::types::PluginType::VideoUnderstanding),
        "video_segmentation" => Ok(super::types::PluginType::VideoSegmentation),
        "image_text_understanding" => Ok(super::types::PluginType::ImageTextUnderstanding),
        "text_vectorization" => Ok(super::types::PluginType::TextVectorization),
        "image_text_vectorization" => Ok(super::types::PluginType::ImageTextVectorization),
        other => Err(crate::error::VideoSceneError::PluginConfigError(
            format!("Unknown plugin type: {}", other)
        )),
    }
}
