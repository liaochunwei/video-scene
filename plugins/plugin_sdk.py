"""Plugin SDK for connecting to Rust-managed plugin socket.

Usage:
    In your plugin's main.py:
        from plugin_sdk import run_plugin

        def handler(action, data, options, send_progress):
            ...

        if __name__ == "__main__":
            run_plugin(handler, "face", ["detect", "detect_batch", "encode"])
"""
import json
import signal
import socket
import sys


def _send_message(sock, msg):
    """Send a single NDJSON message.
    If the connection is broken (EPIPE), raise ConnectionError instead of crashing.
    """
    line = json.dumps(msg, ensure_ascii=False) + "\n"
    try:
        sock.sendall(line.encode("utf-8"))
    except OSError as e:
        if e.errno == 32 or e.errno == 104:  # EPIPE (Broken pipe) or ECONNRESET
            raise ConnectionError(f"Connection broken: {e}")
        raise


class _SocketBuffer:
    """Buffered reader for Unix Socket that preserves data across recv calls.

    NDJSON messages are newline-delimited, but recv() may return data spanning
    multiple messages or partial messages. This buffer ensures we only consume
    one complete message at a time, keeping the rest for subsequent reads.
    """

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def read_message(self):
        """Read one NDJSON message, buffering any leftover data for next call."""
        while b"\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Socket closed")
            self.buf += chunk
        idx = self.buf.index(b"\n")
        line = self.buf[:idx].decode("utf-8").strip()
        self.buf = self.buf[idx + 1:]
        if not line:
            return None
        return json.loads(line)


def run_plugin(handler, plugin_type, actions):
    """Connect to Rust-managed socket and serve plugin requests.

    Args:
        handler: Function with signature handler(action, data, options, send_progress) -> dict
        plugin_type: One of: face, object, video_understanding, video_segmentation,
                     image_text_understanding, text_vectorization, image_text_vectorization
        actions: List of supported action names
    """
    if len(sys.argv) < 2:
        print("Usage: python -m plugin_module <socket_path>", file=sys.stderr)
        sys.exit(1)

    socket_path = sys.argv[1]

    # Connect to Rust-managed socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    buf = _SocketBuffer(sock)

    # Send registration
    _send_message(sock, {
        "type": "register",
        "plugin_type": plugin_type,
        "actions": actions,
    })

    # Wait for registration confirmation
    resp = buf.read_message()
    if resp is None or resp.get("type") != "registered":
        print(f"ERROR: Expected registered message, got: {resp}", file=sys.stderr)
        sys.exit(1)

    # 忽略 SIGPIPE，防止写入已关闭的 socket 时进程被信号杀死
    # （Python 3 通常会抛出 BrokenPipeError 而非 SIGPIPE，但显式忽略更安全）
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    # Install SIGTERM handler for clean shutdown
    running = True
    def _handle_sigterm(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Serve requests
    while running:
        try:
            msg = buf.read_message()
        except ConnectionError as e:
            print(f"WARNING: read ConnectionError: {e}", file=sys.stderr)
            break
        except json.JSONDecodeError as e:
            print(f"WARNING: Invalid JSON from host: {e}", file=sys.stderr)
            continue

        if msg is None:
            continue

        msg_type = msg.get("type", "")

        if msg_type == "request":
            request_id = msg.get("id", "")
            action = msg.get("action", "")
            data = msg.get("data", {})

            def send_progress(message, current=0, total=0):
                _send_message(sock, {
                    "type": "progress",
                    "id": request_id,
                    "message": message,
                    "current": current,
                    "total": total,
                })

            try:
                result = handler(action, data, {}, send_progress)
                _send_message(sock, {
                    "type": "response",
                    "id": request_id,
                    "data": result,
                })
            except ConnectionError as e:
                print(f"WARNING: handler/send ConnectionError: {e}", file=sys.stderr)
                break  # 连接已断开，退出主循环
            except Exception as e:
                print(f"WARNING: handler error: {e}", file=sys.stderr)
                try:
                    _send_message(sock, {
                        "type": "error",
                        "id": request_id,
                        "error": str(e),
                    })
                except ConnectionError:
                    break  # 连接已断开，退出主循环

        elif msg_type == "ping":
            try:
                _send_message(sock, {"type": "pong"})
            except ConnectionError:
                break

        elif msg_type == "shutdown":
            break

    print(f"Plugin {plugin_type} exiting main loop, closing socket", file=sys.stderr)
    sock.close()
