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
    """Send a single NDJSON message."""
    line = json.dumps(msg, ensure_ascii=False) + "\n"
    sock.sendall(line.encode("utf-8"))


def _read_message(sock):
    """Read a single NDJSON message (newline-delimited)."""
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
        if b"\n" in data:
            break
    line = data.decode("utf-8").strip()
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

    # Send registration
    _send_message(sock, {
        "type": "register",
        "plugin_type": plugin_type,
        "actions": actions,
    })

    # Wait for registration confirmation
    resp = _read_message(sock)
    if resp is None or resp.get("type") != "registered":
        print(f"ERROR: Expected registered message, got: {resp}", file=sys.stderr)
        sys.exit(1)

    # Install SIGTERM handler for clean shutdown
    running = True
    def _handle_sigterm(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Serve requests
    while running:
        try:
            msg = _read_message(sock)
        except ConnectionError:
            break

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
            except Exception as e:
                _send_message(sock, {
                    "type": "error",
                    "id": request_id,
                    "error": str(e),
                })

        elif msg_type == "ping":
            _send_message(sock, {"type": "pong"})

        elif msg_type == "shutdown":
            break

    sock.close()
