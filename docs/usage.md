# VideoScene 使用文档

## 目录

- [安装与构建](#安装与构建)
- [工作空间](#工作空间)
- [插件守护进程](#插件守护进程)
- [索引视频](#索引视频)
- [搜索](#搜索)
- [人脸库管理](#人脸库管理)
- [Web UI](#web-ui)
- [配置参考](#配置参考)
- [插件系统详解](#插件系统详解)
- [故障排除](#故障排除)

## 安装与构建

### 前置依赖

- Rust 1.75+
- Python 3.10+（插件运行时）
- [uv](https://github.com/astral-sh/uv)（Python 包管理）
- ffmpeg / ffprobe（视频处理）
- Node.js 18+（仅 Web UI 开发）

### 构建

```bash
# 构建发布版本
cargo build --release

# 开发构建
cargo build

# 构建 Web UI（修改前端后需要）
cd web && npm install && npm run build
cd .. && cargo build  # 重新嵌入前端资源
```

## 工作空间

工作空间是索引数据的隔离单元，每个工作空间有独立的数据库和配置。

```bash
# 初始化工作空间（默认路径 ~/.video-scene/workspaces/<name>）
vs workspace init my-project

# 指定存储路径
vs workspace init my-project /data/video-index

# 列出所有工作空间
vs workspace list

# 切换活动工作空间
vs workspace activate my-project

# 所有命令可通过 --workspace 指定工作空间
vs index video.mp4 --workspace my-project
```

## 插件守护进程

### 为什么需要守护进程

AI 模型（人脸识别、物体检测等）启动耗时较长（加载模型到内存/显存）。守护进程保持插件进程常驻，避免每次命令都重新加载模型。空闲超时后自动停止插件进程释放资源。

### 启动守护进程

```bash
vs plugins daemon
```

守护进程启动后会：
1. 创建 `/tmp/vs-plugins/` 目录
2. 监听 `/tmp/vs-plugins/daemon.sock`
3. 扫描 `plugins/*/plugin.toml` 注册插件
4. 进入接受循环，等待 CLI 连接

### 管理插件

```bash
# 查看所有插件状态
vs plugins status

# 列出已注册插件
vs plugins list

# 预启动某个插件（不用等到首次调用自动启动）
vs plugins start face

# 停止某个插件进程
vs plugins stop face
```

### 关闭守护进程

```bash
# 发送关闭信号（推荐）
# Ctrl+C 或 SIGTERM

# 守护进程关闭时会：
# 1. 停止所有插件进程
# 2. 删除 socket 文件
# 3. 退出
```

## 索引视频

### 基本用法

```bash
# 索引单个视频
vs index /path/to/video.mp4

# 索引目录下所有视频
vs index /path/to/videos/

# 递归索引子目录
vs index /path/to/videos/ --recursive

# 指定视频格式
vs index /path/to/videos/ --extensions mp4,mov,avi
```

### 索引模式

**scene 模式**（默认）— 本地 AI 模型处理：

1. ffprobe 提取视频元数据
2. PySceneDetect 检测场景边界
3. ffmpeg 提取关键帧
4. 人脸检测（insightface）
5. 物体检测（YOLO）
6. VLM 场景描述（本地 Qwen 模型）
7. 文本向量化（embedding）
8. 图像向量化（CLIP）

```bash
vs index video.mp4 --mode scene
```

**video 模式** — 云端 VLM API 处理：

使用阿里云 DashScope API 进行视频理解，需要配置 API 密钥。

```bash
# 在 config.toml 中设置 api_key
vs index video.mp4 --mode video
```

### 索引选项

```bash
# 强制重新索引（即使已索引过）
vs index video.mp4 --force

# 并行索引数（默认 4）
vs index videos/ --parallel 8

# 片段时长范围
vs index video.mp4 --min-seg 3.0 --max-seg 60.0
```

### 索引进度

索引过程中守护进程会输出进度信息，包括：
- 场景检测进度
- 各 AI 模型推理进度
- 向量化进度

## 搜索

### 文本搜索

```bash
# 自然语言搜索（融合所有信号）
vs search "女性在化妆"

# 限定搜索类型
vs search "张三" --face          # 人脸搜索
vs search "汽车" --object        # 物体搜索
vs search "海边日落" --scene     # 场景搜索
```

### 图像搜索

```bash
# 用图片搜索相似片段
vs search --image /path/to/query.jpg
```

### 搜索选项

```bash
# 返回结果数
vs search "query" --top 20

# 最低置信度
vs search "query" --threshold 0.5

# 输出格式
vs search "query" --format json    # JSON 格式
vs search "query" --format table   # 表格格式
vs search "query" --json           # 等同 --format json

# 合并同视频片段
vs search "query" --dedup
```

### 搜索信号

搜索系统融合五种信号：

| 信号 | 匹配方式 | 权重 |
|------|----------|------|
| 人脸 | 人脸特征向量余弦相似度 | 高 |
| 物体 | jieba 分词 + 同义词 + 标签匹配 | 中 |
| 场景 CLIP | 文本向量 + 分类路由 + 向量检索 | 高 |
| BM25 文本 | FTS5 全文搜索 + jieba + 近邻加权 | 中 |
| 图像 | CLIP 图像向量检索 | 高 |

## 人脸库管理

### 添加人脸

```bash
# 添加新人脸（从图片）
vs face add "张三" /path/to/photo.jpg

# 添加新角度到已有的人脸
vs face add "张三" /path/to/another-angle.jpg --add
```

### 从视频提取人脸

```bash
# 自动检测视频中的人脸并添加到人脸库
vs face extract /path/to/video.mp4
```

### 管理人脸

```bash
# 列出所有人脸
vs face list

# 查看人脸详情（编码数量、来源等）
vs face info "张三"

# 重命名
vs face rename "张三" --new-name "张三丰"

# 删除
vs face remove "张三"
```

## Web UI

### 启动

```bash
# 默认端口 6066
vs search --web

# 自定义端口
vs search --web --port 8080

# 带初始搜索词
vs search "query" --web
```

### 功能

- **搜索栏** — 输入自然语言搜索
- **结果卡片** — 显示关键帧缩略图、置信度、匹配类型
- **视频播放** — 点击结果自动跳转到对应片段
- **相似片段** — 点击搜索图标查找相似片段
- **分页** — 大量结果自动分页
- **合并同视频** — 勾选"合并同视频片段"去重

## 配置参考

配置文件：`~/.video-scene/config.toml`

```toml
[index]
config_dir = "~/.video-scene"

[index.face]
min_confidence = 0.8     # 人脸最低置信度
min_quality = 0.5        # 人脸最低质量分

[index.object]
min_confidence = 0.5     # 物体检测最低置信度

[index.scene]
min_segment_duration = 2.0   # 最小片段时长（秒）
max_segment_duration = 30.0  # 最大片段时长（秒）

[plugins]
python_path = "python3"
uv_path = "uv"

[plugins.models]
insightface_model = "buffalo_l"       # 人脸模型
yolo_model = "yoloe-26s-seg.pt"       # 物体检测模型
clip_model = "ViT-B-16"               # CLIP 模型

[plugins.vlm_api]
api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = ""               # 云端 VLM API 密钥
model = "qwen3.6-plus"     # VLM 模型
max_pixels = 230400        # 最大像素数
fps = 2.0                  # 采样帧率
```

## 插件系统详解

### 守护进程架构

```
┌─────────────┐     NDJSON      ┌──────────────┐    NDJSON     ┌──────────────┐
│  CLI 命令    │ ──────────────> │ Plugin Daemon │ ───────────> │ Python 插件   │
│ index/search │  daemon.sock   │              │  plugin.sock  │ insightface  │
│ face/...     │ <────────────── │ 管理生命周期  │ <─────────── │ yolo/vlm/... │
└─────────────┘                 └──────────────┘               └──────────────┘
```

### 插件生命周期

1. **注册** — 守护进程启动时扫描 `plugins/*/plugin.toml`
2. **自动启动** — 首次 `Call` 请求时自动启动插件进程
3. **通信** — NDJSON over Unix socket
4. **空闲检查** — 守护进程每 30 秒检查插件空闲时间
5. **自动停止** — 超过空闲超时后停止插件进程
6. **手动管理** — `vs plugins start/stop` 手动控制

### 内置插件

| 插件目录 | 类型 | 说明 | 空闲超时 |
|----------|------|------|----------|
| insightface | face | 人脸检测与识别 | 300s |
| yolo | object | 物体检测与分割 | 300s |
| scene_detect | video_segmentation | 场景边界检测 | 120s |
| vlm | image_text_understanding | 场景描述（本地 VLM） | 600s |
| vlm | video_understanding | 视频理解（云端 API） | 600s |
| embedding | text_vectorization | 文本向量化 | 600s |
| clip | image_text_vectorization | 图像文本向量化 | 180s |

### 插件开发

详见 [插件开发指南](plugin-development.md)。

## 故障排除

### "Plugin daemon not running"

**原因：** 守护进程未启动或已崩溃。

**解决：**
```bash
vs plugins daemon
```

### "Cannot connect to daemon"

**原因：** socket 文件存在但守护进程已停止（异常退出未清理）。

**解决：**
```bash
rm /tmp/vs-plugins/daemon.sock
vs plugins daemon
```

### 索引速度慢

- 使用 `--parallel` 增加并行数
- 确保插件守护进程已启动（避免每次冷启动模型）
- `--mode video` 使用云端 API 可能更快（取决于网络和 GPU 可用性）

### 搜索结果不相关

- 尝试不同的搜索类型（`--face`、`--object`、`--scene`）
- 调整 `--threshold` 阈值
- 使用 `--image` 进行图像相似搜索
- 确保视频已完整索引（检查 `vs status`）

### Web UI 无法访问

- 确认端口未被占用：`lsof -i :6066`
- 使用 `--port` 指定其他端口
- 确认已构建前端：`cd web && npm run build && cd .. && cargo build`
