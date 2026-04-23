//! 插件进程管理：启动、通信、停止插件子进程。
//!
//! 每个插件以独立进程运行（通常是 Python），通过 Unix Socket 与主程序通信。
//! 启动流程：创建 socket → 拉起子进程 → 等待连接 → 握手注册 → 就绪。
//! 停止流程：先发 Shutdown 消息，再 SIGTERM，最后 SIGKILL，逐级升级确保清理干净。

use std::io::BufReader;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::process::Child;
use std::time::{Duration, Instant};

use super::config::PluginConfig;
use super::protocol::{self, PluginMessage, PluginRequest};

/// 一个正在运行的插件进程，持有 socket 连接和子进程句柄。
pub struct PluginProcess {
    config: PluginConfig,
    child: Child,
    socket: std::os::unix::net::UnixStream,
    socket_path: PathBuf,
    /// 最后一次调用的时间，用于空闲超时回收
    pub last_used: Instant,
}

impl PluginProcess {
    /// 启动插件进程并等待其连接和注册。
    ///
    /// 流程：
    /// 1. 绑定 Unix Socket 监听地址
    /// 2. 将 socket 路径作为参数传给子进程，子进程主动连回来
    /// 3. 等待子进程发送 Register 消息，验证类型匹配
    /// 4. 回复 Registered 确认，关闭 listener（只需一条连接）
    pub fn start(config: &PluginConfig, socket_dir: &PathBuf, plugins_dir: &PathBuf) -> crate::error::Result<Self> {
        let plugin_name = &config.plugin.name;
        let socket_path = socket_dir.join(format!("{}.sock", plugin_name));

        // 确保 socket 目录存在
        if let Some(parent) = socket_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| crate::error::VideoSceneError::StorageError(e.to_string()))?;
        }

        // 清理上次可能残留的 socket 文件
        if socket_path.exists() {
            let _ = std::fs::remove_file(&socket_path);
        }

        // 绑定监听，等子进程连回来
        let listener = UnixListener::bind(&socket_path)
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                format!("Failed to bind socket at {}: {}", socket_path.display(), e)
            ))?;

        // 解析命令行，把 socket 路径追加为最后一个参数
        let mut args: Vec<String> = shlex_split(&config.runtime.command);
        args.push(socket_path.display().to_string());
        let program = args.first().ok_or_else(|| crate::error::VideoSceneError::PluginExecutionError(
            format!("Empty command for plugin {}", plugin_name)
        ))?.clone();
        let child_args: Vec<&str> = args[1..].iter().map(|s| s.as_str()).collect();
        let mut child = std::process::Command::new(&program)
            .args(&child_args)
            .current_dir(plugins_dir) // 工作目录设为插件所在目录，方便 Python 导入
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped()) // 捕获 stderr，超时时输出帮助诊断
            .spawn()
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                format!("Failed to start plugin {}: {}", plugin_name, e)
            ))?;

        // 等待子进程连接到 socket
        let (socket, _) = Self::wait_for_connection(&listener, &socket_path, config.runtime.startup_timeout, &mut child)?;

        // 启动后台线程持续读取插件 stderr 并转发到 Rust 日志
        // 这样 Python 端的 print(..., file=sys.stderr) 能被 Rust 看到
        if let Some(stderr) = child.stderr.take() {
            let name = plugin_name.clone();
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(l) if !l.is_empty() => tracing::info!("[{}] {}", name, l),
                        _ => break,
                    }
                }
            });
        }

        // 读取注册消息，验证插件声明的类型与配置一致
        let mut reader = BufReader::new(
            socket.try_clone()
                .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(
                    format!("Failed to clone socket: {}", e)
                ))?
        );

        let msg = protocol::read_message(&mut reader)?;
        match msg {
            PluginMessage::Register { plugin_type, actions } => {
                if plugin_type != config.plugin.plugin_type.as_str() {
                    return Err(crate::error::VideoSceneError::PluginExecutionError(
                        format!("Plugin registered as type {} but config says {}", plugin_type, config.plugin.plugin_type)
                    ));
                }
                tracing::info!("Plugin {} registered (type: {}, actions: {:?})", plugin_name, plugin_type, actions);
            }
            _ => {
                return Err(crate::error::VideoSceneError::PluginExecutionError(
                    format!("Expected register message from plugin {}, got {:?}", plugin_name, msg)
                ));
            }
        }

        // 发送注册确认
        let registered = PluginMessage::Registered { plugin_type: config.plugin.plugin_type.as_str().to_string() };
        let mut write_socket = socket.try_clone()
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?;
        protocol::write_message(&mut write_socket, &registered)?;

        // 注册完成，关闭 listener —— 后续只需这一条连接
        drop(listener);

        Ok(Self {
            config: config.clone(),
            child,
            socket,
            socket_path,
            last_used: Instant::now(),
        })
    }

    /// 等待子进程连接到 socket，带超时。
    /// 用非阻塞 + 轮询实现，避免阻塞整个线程。
    /// 超时时读取子进程 stderr 输出帮助诊断（如 uv 依赖安装慢、Python 导入错误等）。
    fn wait_for_connection(
        listener: &UnixListener,
        socket_path: &PathBuf,
        timeout_secs: u64,
        child: &mut std::process::Child,
    ) -> crate::error::Result<(std::os::unix::net::UnixStream, std::os::unix::net::SocketAddr)> {
        let start = Instant::now();
        listener.set_nonblocking(true)
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?;

        loop {
            match listener.accept() {
                Ok((stream, addr)) => {
                    listener.set_nonblocking(false).ok();
                    stream.set_nonblocking(false)
                        .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?;
                    return Ok((stream, addr));
                }
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::WouldBlock {
                        let _ = std::fs::remove_file(socket_path);
                        return Err(crate::error::VideoSceneError::PluginExecutionError(
                            format!("Failed to accept connection: {}", e)
                        ));
                    }
                    if start.elapsed() > Duration::from_secs(timeout_secs) {
                        let _ = std::fs::remove_file(socket_path);

                        // 收集诊断信息：子进程是否还活着
                        let alive = child.try_wait().map(|s| s.is_none()).unwrap_or(false);

                        let mut detail = format!("Plugin did not connect within {} seconds", timeout_secs);
                        if alive {
                            detail.push_str(" (process is still running — likely slow startup, consider increasing startup_timeout in plugin.toml)");
                        } else {
                            detail.push_str(" (process has exited prematurely)");
                        }
                        return Err(crate::error::VideoSceneError::PluginTimeout(detail));
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }

    /// 发送请求并收集响应，支持进度回调。
    ///
    /// 插件可能先发多条 Progress 消息再发最终 Response，
    /// 这里循环读取直到拿到 Response 或 Error。
    pub fn call(
        &mut self,
        action: &str,
        data: &serde_json::Value,
        progress_cb: &dyn Fn(super::ProgressMessage),
    ) -> crate::error::Result<super::PluginResponse> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let request = PluginRequest::Request {
            id: request_id.clone(),
            action: action.to_string(),
            data: data.clone(),
        };

        // 调用开始时立即更新 last_used，防止 check_idle 误杀正在处理请求的插件
        self.last_used = Instant::now();

        protocol::write_message(&mut self.socket, &request)?;

        // 克隆 socket 用于读取（Unix Socket 支持独立读写）
        let mut reader = BufReader::new(self.socket.try_clone()
            .map_err(|e| crate::error::VideoSceneError::PluginExecutionError(e.to_string()))?);

        let mut progress_messages = Vec::new();
        let result_value;

        loop {
            let msg = protocol::read_message(&mut reader);
            match msg {
                Ok(PluginMessage::Progress { id, message, current, total }) => {
                    let pm = super::ProgressMessage {
                        id,
                        message,
                        current,
                        total,
                    };
                    progress_cb(pm.clone()); // 实时通知调用方
                    progress_messages.push(pm);
                }
                Ok(PluginMessage::Response { id, data }) => {
                    if id != request_id {
                        tracing::warn!("Response ID mismatch: expected {}, got {}", request_id, id);
                    }
                    result_value = data;
                    break;
                }
                Ok(PluginMessage::Error { id: _, error }) => {
                    return Err(crate::error::VideoSceneError::PluginExecutionError(
                        format!("Plugin {} error: {}", self.config.plugin.name, error)
                    ));
                }
                Ok(other) => {
                    tracing::warn!("Unexpected message from plugin: {:?}", other);
                }
                Err(e) => {
                    // 插件进程可能已崩溃，检查存活状态并输出诊断信息
                    let alive = self.is_alive();
                    tracing::error!(
                        "Plugin {} socket read error: {} (process alive: {})",
                        self.config.plugin.name, e, alive
                    );
                    return Err(crate::error::VideoSceneError::PluginExecutionError(
                        format!("Plugin {} connection lost: {} (process alive: {})",
                            self.config.plugin.name, e, alive)
                    ));
                }
            }
        }

        self.last_used = Instant::now(); // 更新最后使用时间，用于空闲回收

        Ok(super::PluginResponse {
            result: result_value,
            progress: progress_messages,
        })
    }

    /// 检查子进程是否还活着。
    /// try_wait 返回 None 表示进程仍在运行。
    pub fn is_alive(&mut self) -> bool {
        self.child.try_wait().map(|s| s.is_none()).unwrap_or(false)
    }

    /// 停止插件进程：优雅退出 → SIGTERM → SIGKILL，逐级升级。
    ///
    /// 优先发送 Shutdown 消息让插件自行清理资源（如释放 GPU 显存），
    /// 超时后再发信号强制终止，确保不会留下僵尸进程。
    pub fn stop(&mut self) -> crate::error::Result<()> {
        // 第一步：发 Shutdown 消息，给插件 2 秒优雅退出
        let shutdown = PluginRequest::Shutdown;
        let _ = protocol::write_message(&mut self.socket, &shutdown);

        std::thread::sleep(Duration::from_secs(2));

        match self.child.try_wait() {
            Ok(Some(_)) => {} // 已退出
            Ok(None) => {
                // 第二步：发 SIGTERM，再等 3 秒
                unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM); }
                std::thread::sleep(Duration::from_secs(3));
                match self.child.try_wait() {
                    Ok(Some(_)) => {}
                    _ => {
                        // 第三步：SIGKILL 强杀
                        unsafe { libc::kill(self.child.id() as i32, libc::SIGKILL); }
                        let _ = self.child.wait();
                    }
                }
            }
            Err(_) => {}
        }

        // 清理 socket 文件
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }

        Ok(())
    }
}

impl Drop for PluginProcess {
    fn drop(&mut self) {
        // 进程对象析构时确保子进程被停止，防止资源泄漏
        if self.is_alive() {
            let _ = self.stop();
        }
    }
}

/// 简易命令行拆分：处理双引号和单引号，但不含转义和嵌套。
/// 适用于 plugin.toml 中 command 字段如 `python -u server.py` 的场景。
fn shlex_split(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let mut quote_char = ' ';
    for ch in s.chars() {
        if in_quote {
            if ch == quote_char {
                in_quote = false;
            } else {
                current.push(ch);
            }
        } else if ch == '"' || ch == '\'' {
            in_quote = true;
            quote_char = ch;
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}
