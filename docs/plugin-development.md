# VideoScene 插件开发指南

本文档介绍如何为 VideoScene 开发自定义插件。VideoScene 的插件系统基于 NDJSON over Unix Domain Socket 协议，由 Rust 端管理插件进程的启动、通信和生命周期。

---

## 1. 快速开始

5 分钟创建一个最小可用的插件。

### 1.1 创建插件目录

在 `plugins/` 目录下创建插件子目录，并添加 `plugin.toml` 配置文件：

```
plugins/
  my_plugin/
    plugin.toml
    main.py
```

### 1.2 编写 plugin.toml

```toml
[plugin]
name = "my_plugin"
version = "0.1.0"
type = "video_segmentation"
description = "My custom scene detection plugin"

[runtime]
command = "python3 -m my_plugin.main"
idle_timeout = 120
startup_timeout = 15

[capabilities]
actions = ["detect_scenes"]
max_batch_size = 1
```

### 1.3 编写 Python 处理函数

创建 `plugins/my_plugin/main.py`：

```python
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from plugin_sdk import run_plugin

def handler(action, data, options, send_progress):
    if action == "detect_scenes":
        return detect_scenes(data, options, send_progress)
    else:
        raise ValueError(f"Unknown action: {action}")

def detect_scenes(data, options, send_progress):
    video_path = data["video_path"]
    send_progress("Analyzing video...", 0, 1)

    # 你的检测逻辑在这里
    scenes = [{"start": 0.0, "end": 10.0}]

    send_progress("Done", 1, 1)
    return {"scenes": scenes}

if __name__ == "__main__":
    run_plugin(handler, "video_segmentation", ["detect_scenes"])
```

### 1.4 测试

```bash
# 查看插件是否注册成功
vs plugins list

# 启动插件测试
vs plugins start video_segmentation

# 查看运行状态
vs plugins status
```

---

## 2. 插件类型参考

VideoScene 支持 7 种插件类型，每种类型定义了特定的 actions 及其请求/响应格式。

### 2.1 face — 人脸检测识别

| Action | 说明 |
|--------|------|
| `detect` | 单张图片人脸检测 |
| `detect_batch` | 批量图片人脸检测 |
| `encode` | 提取人脸特征向量 |

**detect**

```json
// Request data
{
  "image_path": "/path/to/image.jpg",
  "min_confidence": 0.8
}

// Response
{
  "faces": [
    {
      "bbox": [x, y, width, height],
      "confidence": 0.95,
      "feature": [0.1, 0.2, ...],  // 512-dim float vector
      "quality": 0.95
    }
  ]
}
```

**detect_batch**

```json
// Request data
{
  "image_paths": ["/path/to/1.jpg", "/path/to/2.jpg"],
  "min_confidence": 0.8
}

// Response
{
  "results": [
    {
      "image_path": "/path/to/1.jpg",
      "faces": [
        { "bbox": [x, y, w, h], "confidence": 0.95, "feature": [...], "quality": 0.95 }
      ]
    }
  ]
}
```

**encode**

```json
// Request data
{
  "image_path": "/path/to/face.jpg"
}

// Response
{
  "feature": [0.1, 0.2, ...]  // 512-dim float vector
}
```

### 2.2 object — 物体检测识别

| Action | 说明 |
|--------|------|
| `detect` | 单张图片物体检测 |
| `detect_batch` | 批量图片物体检测 |

**detect**

```json
// Request data
{
  "image_path": "/path/to/image.jpg",
  "min_confidence": 0.5,
  "classes": ["person", "car"]  // 可选，过滤检测类别
}

// Response
{
  "objects": [
    {
      "label": "person",
      "label_zh": "人",
      "confidence": 0.92,
      "bbox": [x, y, width, height]
    }
  ]
}
```

**detect_batch**

```json
// Request data
{
  "image_paths": ["/path/to/1.jpg", "/path/to/2.jpg"],
  "min_confidence": 0.5,
  "classes": ["person"]
}

// Response
{
  "results": [
    {
      "image_path": "/path/to/1.jpg",
      "objects": [
        { "label": "person", "label_zh": "人", "confidence": 0.92, "bbox": [x, y, w, h] }
      ]
    }
  ]
}
```

### 2.3 video_understanding — 视频理解

| Action | 说明 |
|--------|------|
| `describe_video` | 对整个视频进行分段理解描述（通常调用云端 VLM API） |

**describe_video**

```json
// Request data
{
  "video_path": "/path/to/video.mp4",
  "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "api_key": "sk-xxx",
  "model": "qwen3.6-plus",
  "max_pixels": 230400,
  "fps": 2.0
}

// Response
{
  "segments": [
    {
      "片段开始": 1.5,
      "片段结束": 3.0,
      "人": "年轻女性",
      "前景物": "口红",
      "背景物": "书架",
      "场": "室内",
      "动作": "涂抹口红",
      "标识": ["水印-ABC", "字幕-Hello"]
    }
  ]
}
```

### 2.4 video_segmentation — 视频分隔

| Action | 说明 |
|--------|------|
| `detect_scenes` | 检测视频场景切换点 |

**detect_scenes**

```json
// Request data
{
  "video_path": "/path/to/video.mp4",
  "detector": "content",  // "content" 或 "adaptive"
  "threshold": 27.0
}

// Response
{
  "scenes": [
    { "start": 0.0, "end": 5.3 },
    { "start": 5.3, "end": 12.8 }
  ]
}
```

### 2.5 image_text_understanding — 图文理解

| Action | 说明 |
|--------|------|
| `describe_scene` | 描述单张/多帧场景 |
| `describe_scenes_batch` | 批量描述多个场景 |

**describe_scene**

```json
// Request data
{
  "image_paths": ["/path/to/frame1.jpg", "/path/to/frame2.jpg"],
  "model": ""  // 可选，默认使用本地 VLM
}

// Response
{
  "structured": {
    "人": "年轻女性，淡妆，长发",
    "前景物": "口红、粉底",
    "背景物": "书架、台灯",
    "场": "室内温馨环境",
    "动作": "涂抹口红",
    "标识": ["水印-ABC"],
    "字幕": ["Hello World"]
  }
}
```

**describe_scenes_batch**

```json
// Request data
{
  "scenes": [
    { "image_paths": ["/path/to/frame1.jpg"] },
    { "image_paths": ["/path/to/frame2.jpg", "/path/to/frame3.jpg"] }
  ],
  "model": ""
}

// Response
{
  "results": [
    { "structured": { "人": "...", "前景物": "...", ... } },
    { "structured": { "人": "...", "前景物": "...", ... } }
  ]
}
```

### 2.6 text_vectorization — 文本向量化

| Action | 说明 |
|--------|------|
| `encode_text` | 编码单条搜索查询（带搜索指令前缀） |
| `encode_texts_batch` | 批量编码搜索查询 |
| `encode_document` | 编码单条文档（无指令前缀，用于索引） |
| `encode_documents_batch` | 批量编码文档 |
| `encode_text_with_categories` | 编码查询并返回分类标签向量 |

**encode_text / encode_document**

```json
// Request data
{ "text": "女性涂抹口红" }

// Response
{ "vector": [0.1, 0.2, ...] }  // 1024-dim float vector
```

**encode_texts_batch / encode_documents_batch**

```json
// Request data
{ "texts": ["女性涂抹口红", "室内场景"] }

// Response
{
  "results": [
    { "text": "女性涂抹口红", "vector": [...] },
    { "text": "室内场景", "vector": [...] }
  ]
}
```

**encode_text_with_categories**

```json
// Request data
{ "text": "女性涂抹口红" }

// Response
{
  "query_vector": [0.1, 0.2, ...],
  "category_vectors": {
    "person": [...],
    "foreground": [...],
    "background": [...],
    "scene": [...],
    "action": [...],
    "marks": [...]
  }
}
```

### 2.7 image_text_vectorization — 图文向量化

| Action | 说明 |
|--------|------|
| `encode_image` | 编码单张图片 |
| `encode_images_batch` | 批量编码图片 |

**encode_image**

```json
// Request data
{ "image_path": "/path/to/image.jpg" }

// Response
{ "feature": [0.1, 0.2, ...] }  // float vector
```

**encode_images_batch**

```json
// Request data
{ "image_paths": ["/path/to/1.jpg", "/path/to/2.jpg"] }

// Response
{
  "results": [
    { "image_path": "/path/to/1.jpg", "feature": [...] },
    { "image_path": "/path/to/2.jpg", "feature": [...] }
  ]
}
```

---

## 3. plugin.toml 规范

每个插件必须在根目录包含一个 `plugin.toml` 文件，描述插件的元信息、运行方式和能力。

### 3.1 [plugin] 部分

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 插件名称，唯一标识，用于 socket 文件命名 |
| `version` | string | 是 | 语义化版本号，如 `"1.0.0"` |
| `type` | string | 是 | 插件类型，必须是以下之一：`face`、`object`、`video_understanding`、`video_segmentation`、`image_text_understanding`、`text_vectorization`、`image_text_vectorization` |
| `description` | string | 否 | 插件功能描述 |

### 3.2 [runtime] 部分

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `command` | string | 是 | — | 启动插件的命令，Rust 通过 `sh -c` 执行。socket 路径会作为额外参数追加到命令末尾 |
| `idle_timeout` | integer | 否 | `300` | 空闲超时时间（秒），超过后 Rust 端自动停止插件进程。设为 `0` 表示永不超时 |
| `startup_timeout` | integer | 否 | `30` | 启动超时时间（秒），插件必须在此时间内完成 socket 连接和注册 |

### 3.3 [capabilities] 部分

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `actions` | string[] | 否 | `[]` | 插件支持的 action 列表 |
| `max_batch_size` | integer | 否 | `16` | 单次批量请求的最大数量 |

### 3.4 [extra_types] 部分

当一个插件同时服务多种类型时使用。键为插件类型名，值为该类型对应的 action 名称。

例如 VLM 插件同时服务 `image_text_understanding` 和 `video_understanding`：

```toml
[extra_types]
video_understanding = "describe_video"
```

Rust 端注册插件时会同时为 `extra_types` 中声明的类型创建别名，使同一种插件可以通过多种类型调用。

### 3.5 完整示例

```toml
[plugin]
name = "vlm"
version = "1.0.0"
type = "image_text_understanding"
description = "Image-text understanding using Qwen VLM (local) and cloud API (video understanding)"

[runtime]
command = "python3 -m vlm_plugin.main"
idle_timeout = 600
startup_timeout = 60

[capabilities]
actions = ["describe_scene", "describe_scenes_batch", "describe_video"]
max_batch_size = 1

[extra_types]
video_understanding = "describe_video"
```

---

## 4. 协议参考

插件与 Rust 端通过 Unix Domain Socket 通信，消息格式为 NDJSON（每行一个 JSON 对象，以 `\n` 分隔）。

### 4.1 启动握手：register → registered

插件启动后，从 `argv[1]` 获取 socket 路径，连接并发送注册消息：

```
插件 ──→ Rust:  {"type":"register","plugin_type":"face","actions":["detect","detect_batch","encode"]}
Rust  ──→ 插件:  {"type":"registered","plugin_type":"face"}
```

注册消息中的 `plugin_type` 必须与 `plugin.toml` 中声明的 `type` 一致，否则 Rust 端会拒绝注册并报错。

### 4.2 请求/响应流程：request → progress* → response/error

Rust 端发送请求，插件处理后返回结果：

```
Rust  ──→ 插件:  {"type":"request","id":"uuid-xxx","action":"detect","data":{"image_path":"/path/to/img.jpg"}}
插件  ──→ Rust:  {"type":"progress","id":"uuid-xxx","message":"Loading model...","current":0,"total":1}
插件  ──→ Rust:  {"type":"progress","id":"uuid-xxx","message":"Detecting faces","current":1,"total":1}
插件  ──→ Rust:  {"type":"response","id":"uuid-xxx","data":{"faces":[...]}}
```

如果处理出错：

```
插件  ──→ Rust:  {"type":"error","id":"uuid-xxx","error":"Cannot read image: /path/to/img.jpg"}
```

**消息字段说明：**

| 消息类型 | 方向 | 字段 |
|----------|------|------|
| `request` | Rust → 插件 | `id` (string), `action` (string), `data` (object) |
| `progress` | 插件 → Rust | `id` (string), `message` (string), `current` (integer), `total` (integer) |
| `response` | 插件 → Rust | `id` (string), `data` (object) |
| `error` | 插件 → Rust | `id` (string), `error` (string) |

### 4.3 心跳检测：ping → pong

Rust 端周期性发送 ping 检查插件是否存活：

```
Rust  ──→ 插件:  {"type":"ping"}
插件  ──→ Rust:  {"type":"pong"}
```

### 4.4 关闭：shutdown

Rust 端发送 shutdown 消息通知插件优雅退出：

```
Rust  ──→ 插件:  {"type":"shutdown"}
```

插件收到后应清理资源并关闭 socket 连接。如果插件在 2 秒内未退出，Rust 端会发送 SIGTERM；5 秒后仍未退出则发送 SIGKILL。

---

## 5. Python SDK

VideoScene 提供了 `plugin_sdk.py` 简化 Python 插件开发。SDK 封装了 socket 连接、注册握手、消息读写和 SIGTERM 处理。

### 5.1 导入与使用

将 `plugin_sdk.py` 放在 `plugins/` 目录下，然后在插件的 `main.py` 中导入：

```python
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from plugin_sdk import run_plugin
```

### 5.2 Handler 函数签名

```python
def handler(action: str, data: dict, options: dict, send_progress: callable) -> dict:
    """
    处理插件请求。

    Args:
        action: 请求的 action 名称，如 "detect"、"encode_text" 等
        data: 请求数据，对应协议中 request 消息的 data 字段
        options: 保留字段，当前为空 dict
        send_progress: 进度回调函数，签名为 send_progress(message, current, total)

    Returns:
        dict: 处理结果，会作为 response 消息的 data 字段发送

    Raises:
        Exception: 任何异常会被 SDK 捕获并作为 error 消息发送回 Rust 端
    """
```

### 5.3 send_progress 回调

在长时间操作中调用 `send_progress` 向 Rust 端报告进度：

```python
def detect_scenes(data, options, send_progress):
    video_path = data["video_path"]

    send_progress("Loading video...", 0, 1)
    # ... 加载视频

    send_progress("Detecting scenes", 0, total_scenes)
    for i, scene in enumerate(scenes):
        # ... 处理每个场景
        send_progress("Detecting scenes", i + 1, total_scenes)

    return {"scenes": scenes}
```

### 5.4 run_plugin 调用

在 `main.py` 底部调用 `run_plugin`，传入 handler、插件类型和 actions 列表：

```python
if __name__ == "__main__":
    run_plugin(handler, "face", ["detect", "detect_batch", "encode"])
```

`run_plugin` 会自动完成：
1. 从 `sys.argv[1]` 读取 socket 路径
2. 连接 Unix Domain Socket
3. 发送 register 消息并等待 registered 确认
4. 注册 SIGTERM 处理器，实现优雅退出
5. 进入消息循环，分发 request/ping/shutdown 消息

### 5.5 惰性加载模型

推荐在 handler 内部惰性加载模型，避免启动时阻塞：

```python
_model = None

def get_model():
    global _model
    if _model is None:
        from insightface.app import FaceAnalysis
        _model = FaceAnalysis(name="buffalo_l")
        _model.prepare(ctx_id=0, det_size=(640, 640))
    return _model

def handler(action, data, options, send_progress):
    model = get_model()  # 首次调用时加载，后续复用
    # ...
```

---

## 6. 其他语言

插件协议基于 NDJSON over Unix Domain Socket，是语言无关的。你可以使用任何编程语言实现插件。

### 6.1 关键步骤

1. **读取 socket 路径**：从命令行参数 `argv[1]` 获取 Unix Domain Socket 路径
2. **连接 socket**：建立 Unix Domain Socket 连接（`AF_UNIX` + `SOCK_STREAM`）
3. **注册**：发送 `{"type":"register","plugin_type":"<type>","actions":[...]}` 消息
4. **等待确认**：读取 `{"type":"registered","plugin_type":"<type>"}` 消息
5. **服务请求**：进入循环，读取 NDJSON 消息并处理：
   - `request` → 处理后返回 `response` 或 `error`
   - `ping` → 返回 `pong`
   - `shutdown` → 退出循环
6. **关闭连接**：关闭 socket

### 6.2 消息格式

所有消息都是单行 JSON，以换行符 `\n` 结尾：

```
{"type":"register","plugin_type":"face","actions":["detect"]}\n
{"type":"request","id":"abc-123","action":"detect","data":{"image_path":"/tmp/test.jpg"}}\n
{"type":"response","id":"abc-123","data":{"faces":[]}}\n
```

### 6.3 Node.js 示例

```javascript
const net = require('net');
const fs = require('fs');

const socketPath = process.argv[2];
const sock = net.createConnection(socketPath);

function send(msg) {
  sock.write(JSON.stringify(msg) + '\n');
}

let buffer = '';
sock.on('data', (data) => {
  buffer += data.toString();
  const lines = buffer.split('\n');
  buffer = lines.pop(); // 保留不完整的行
  for (const line of lines) {
    if (!line.trim()) continue;
    const msg = JSON.parse(line);
    handleMessage(msg);
  }
});

function handleMessage(msg) {
  switch (msg.type) {
    case 'registered':
      console.error('Registered successfully');
      break;
    case 'request':
      try {
        const result = handleAction(msg.action, msg.data);
        send({ type: 'response', id: msg.id, data: result });
      } catch (e) {
        send({ type: 'error', id: msg.id, error: e.message });
      }
      break;
    case 'ping':
      send({ type: 'pong' });
      break;
    case 'shutdown':
      sock.end();
      process.exit(0);
      break;
  }
}

// 注册
send({ type: 'register', plugin_type: 'video_segmentation', actions: ['detect_scenes'] });

function handleAction(action, data) {
  if (action === 'detect_scenes') {
    return { scenes: [{ start: 0.0, end: 10.0 }] };
  }
  throw new Error(`Unknown action: ${action}`);
}
```

### 6.4 Go 示例

```go
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "net"
    "os"
)

func main() {
    socketPath := os.Args[1]
    conn, err := net.Dial("unix", socketPath)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Connect error: %v\n", err)
        os.Exit(1)
    }
    defer conn.Close()

    reader := bufio.NewReader(conn)

    // 注册
    send(conn, map[string]interface{}{
        "type":        "register",
        "plugin_type": "video_segmentation",
        "actions":     []string{"detect_scenes"},
    })

    // 等待 registered
    msg := readMessage(reader)
    if msg["type"] != "registered" {
        fmt.Fprintf(os.Stderr, "Expected registered, got: %v\n", msg)
        os.Exit(1)
    }

    // 服务请求
    for {
        msg := readMessage(reader)
        switch msg["type"] {
        case "request":
            id := msg["id"].(string)
            action := msg["action"].(string)
            result, err := handleAction(action, msg["data"])
            if err != nil {
                send(conn, map[string]interface{}{"type": "error", "id": id, "error": err.Error()})
            } else {
                send(conn, map[string]interface{}{"type": "response", "id": id, "data": result})
            }
        case "ping":
            send(conn, map[string]interface{}{"type": "pong"})
        case "shutdown":
            return
        }
    }
}

func send(conn net.Conn, msg map[string]interface{}) {
    data, _ := json.Marshal(msg)
    conn.Write(append(data, '\n'))
}

func readMessage(reader *bufio.Reader) map[string]interface{} {
    line, _ := reader.ReadString('\n')
    var msg map[string]interface{}
    json.Unmarshal([]byte(line), &msg)
    return msg
}

func handleAction(action string, data interface{}) (map[string]interface{}, error) {
    if action == "detect_scenes" {
        return map[string]interface{}{"scenes": []map[string]float64{{"start": 0.0, "end": 10.0}}}, nil
    }
    return nil, fmt.Errorf("unknown action: %s", action)
}
```

---

## 7. 测试

### 7.1 验证插件注册

```bash
vs plugins list
```

输出示例：

```
insightface (face): stopped [idle: 0s / 300s]
yolo (object): stopped [idle: 0s / 300s]
vlm (image_text_understanding): stopped [idle: 0s / 600s]
scene_detect (video_segmentation): stopped [idle: 0s / 120s]
embedding (text_vectorization): stopped [idle: 0s / 600s]
clip (image_text_vectorization): stopped [idle: 0s / 180s]
```

如果插件未出现在列表中，检查 `plugin.toml` 是否位于 `plugins/<name>/plugin.toml` 路径下，且格式正确。

### 7.2 测试插件启动

```bash
vs plugins start <plugin_type>
```

例如：

```bash
vs plugins start video_segmentation
```

这会触发插件进程启动（通过发送一个 ping 请求来验证）。如果启动失败，请检查 `command` 路径是否正确。

### 7.3 查看运行状态

```bash
vs plugins status
```

输出示例：

```
insightface          [face]                     running (idle 5s)
yolo                 [object]                   stopped
scene_detect         [video_segmentation]       stopped
```

---

## 8. 调试

### 8.1 插件启动失败

**症状**：`vs plugins start` 报错或超时

**排查**：
- 检查 `plugin.toml` 中的 `command` 是否正确，可以手动运行该命令验证
- 确保 Python 环境已安装所需依赖（在 `plugins/.venv` 下）
- 检查 `startup_timeout` 是否足够（模型加载较慢时需要增大，如 `60`）
- 查看 stderr 输出，Rust 端会捕获插件进程的启动错误

```bash
# 手动测试插件是否能正常启动
cd plugins
python3 -m my_plugin.main /tmp/test.sock
```

### 8.2 Socket 连接被拒绝

**症状**：插件报 `Connection refused` 或 Rust 端报 `Plugin did not connect`

**排查**：
- 确认 `argv[1]` 中的 socket 路径与 Rust 端创建的路径一致
- 检查 socket 文件权限
- 确认没有残留的旧 socket 文件（Rust 端会自动清理，但异常退出可能残留）
- 检查 `/tmp/vs-plugins/` 目录下的 `.sock` 文件

### 8.3 注册类型不匹配

**症状**：Rust 端报 `Plugin registered as type X but config says Y`

**排查**：
- 确保 `run_plugin()` 的第二个参数与 `plugin.toml` 中的 `type` 字段完全一致
- 例如 `plugin.toml` 中 `type = "face"`，则 `run_plugin(handler, "face", ...)` 也必须是 `"face"`

### 8.4 处理超时

**症状**：请求长时间无响应

**排查**：
- 检查 `idle_timeout` 是否过小，导致处理过程中被 Rust 端终止
- 检查模型推理是否卡住（GPU 内存不足等）
- 确认 handler 函数没有死循环
- 在 handler 中添加 `send_progress` 调用，观察 Rust 端是否能收到进度更新

### 8.5 进程残留

**症状**：插件进程未正确退出

**排查**：
- 确认 SDK 的 SIGTERM 处理器正常工作（Python SDK 自动处理）
- Rust 端会在 shutdown 后 2 秒发送 SIGTERM，5 秒后发送 SIGKILL
- 如果使用自定义实现，确保正确处理 `shutdown` 消息和 SIGTERM 信号
