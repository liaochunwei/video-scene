# VideoScene

视频场景检索系统 — 基于人脸、物体和场景的智能视频搜索。

## 功能

- **视频索引** — 自动检测场景、提取关键帧、运行 AI 模型（人脸识别、物体检测、VLM 场景描述、文本/图像向量化）
- **多信号搜索** — 融合人脸、物体、场景 CLIP、BM25 文本、图像相似度五种信号
- **人脸库管理** — 添加、提取、匹配、重命名人脸
- **备份与导入** — 人脸库和工作空间数据的打包备份与增量恢复
- **Web UI** — 内置 React 前端，支持视频播放和相似片段搜索
- **插件系统** — 守护进程管理 AI 模型进程，空闲自动停止
- **工作空间** — 支持多工作空间隔离

## 快速开始

### 安装

```bash
# 构建
cargo build --release

# 二进制文件
./target/release/video-scene
# 或使用短命令
alias vs=./target/release/video-scene
```

### 1. 启动插件守护进程

```bash
vs plugins daemon
```

守护进程在后台管理所有 AI 模型进程，CLI 命令通过 Unix socket 与守护进程通信。必须先启动守护进程才能执行索引和搜索。

### 2. 索引视频

```bash
# 索引单个视频
vs index /path/to/video.mp4

# 索引目录（递归）
vs index /path/to/videos/ --recursive

# 使用云端 VLM API 索引
vs index /path/to/video.mp4 --mode video
```

### 3. 搜索

```bash
# 自然语言搜索
vs search "女性涂抹口红"

# 人脸搜索
vs search "张三" --face

# 物体搜索
vs search "汽车" --object

# 图像相似搜索
vs search --image /path/to/query.jpg

# 启动 Web UI
vs search --web
```

## 命令参考

### 全局选项

| 选项 | 短选项 | 说明 |
|------|--------|------|
| `--workspace <NAME>` | | 工作空间名称（默认：活动工作空间） |
| `--config <PATH>` | | 配置文件路径 |
| `--verbose` | `-v` | 详细输出 |
| `--quiet` | `-q` | 安静模式 |

### index — 索引视频

```
vs index <PATH> [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--recursive` | false | 递归索引子目录 |
| `--extensions` | `mp4,mov,mkv` | 视频扩展名（逗号分隔） |
| `--force` | false | 强制重新索引 |
| `--parallel` | 4 | 并行索引数 |
| `--min-seg` | 2.0 | 最小片段时长（秒） |
| `--max-seg` | 30.0 | 最大片段时长（秒） |
| `--mode` | `scene` | 索引模式：`scene`（本地检测+VLM）或 `video`（云端 VLM API） |

### search — 搜索视频

```
vs search [QUERY] [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--face` | false | 人脸搜索 |
| `--object` | false | 物体搜索 |
| `--scene` | false | 场景搜索 |
| `--image <PATH>` | | 图像相似搜索 |
| `--top` | 10 | 返回结果数 |
| `--threshold` | 自动 | 最低置信度阈值 |
| `--format` | `simple` | 输出格式：`simple`、`table`、`json` |
| `--json` | | 等同 `--format json` |
| `--web` | false | 启动 Web UI |
| `--port` | 6066 | Web UI 端口 |
| `--dedup` | false | 合并同视频片段 |

### workspace — 工作空间管理

```
vs workspace init <NAME> [PATH]    # 初始化工作空间
vs workspace list                  # 列出工作空间
vs workspace activate <NAME>       # 切换活动工作空间
vs workspace backup [NAME] <OUTPUT> # 备份工作空间到 tar.gz（默认当前活动工作空间）
vs workspace import <BACKUP>       # 从 tar.gz 导入到当前工作空间
```

### face — 人脸库管理

```
vs face add <NAME> <IMAGE>         # 添加人脸
vs face add <NAME> <IMAGE> --add   # 添加新角度到已有的人脸
vs face list                       # 列出所有人脸
vs face info <NAME>                # 查看人脸详情
vs face remove <NAME>              # 删除人脸
vs face rename <NAME> [--new-name] # 重命名
vs face extract <VIDEO>            # 从视频提取人脸
vs face backup <OUTPUT>            # 备份人脸库到 tar.gz
vs face import <BACKUP>            # 从 tar.gz 增量导入人脸库
```

### plugins — 插件管理

```
vs plugins daemon                  # 启动插件守护进程
vs plugins list                    # 列出所有插件
vs plugins status                  # 查看插件运行状态
vs plugins start <TYPE>            # 启动插件进程
vs plugins stop <TYPE>             # 停止插件进程
```

### 其他命令

```
vs status                          # 查看索引状态
vs list                            # 列出已索引视频
vs remove <VIDEO>                  # 删除视频索引
vs remove --all                    # 清空索引
vs clean                           # 清理无效索引
vs config                          # 显示当前配置
```

## 配置

配置文件位于 `~/.video-scene/config.toml`，支持部分配置（与默认值合并）。

```toml
[index]
config_dir = "~/.video-scene"

[index.face]
min_confidence = 0.8
min_quality = 0.5

[index.object]
min_confidence = 0.5

[index.scene]
min_segment_duration = 2.0
max_segment_duration = 30.0

[plugins]
python_path = "python3"
uv_path = "uv"

[plugins.models]
insightface_model = "buffalo_l"
yolo_model = "yoloe-26s-seg.pt"
clip_model = "ViT-B-16"

[plugins.vlm_api]
api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = ""          # 云端 VLM API 密钥（--mode video 必需）
model = "qwen3.6-plus"
max_pixels = 230400
fps = 2.0
```

## 插件系统

VideoScene 使用守护进程架构管理 AI 模型进程：

```
CLI (index/search/face)          Plugin Daemon
       │                              │
       │  NDJSON over daemon.sock      │  管理 PluginProcess
       ├──────────────────────────────>│  生命周期、空闲检查
       │  {type:"call", ...}           │────> Plugin Process
       │                              │<────  Progress/Response
       │  {type:"response", ...}       │
       │<──────────────────────────────│
```

### 内置插件

| 插件 | 类型 | 说明 | 空闲超时 |
|------|------|------|----------|
| insightface | face | 人脸检测识别 | 300s |
| yolo | object | 物体检测 | 300s |
| scene_detect | video_segmentation | 场景边界检测 | 120s |
| vlm | image_text_understanding | 场景描述（本地 VLM） | 600s |
| vlm | video_understanding | 视频理解（云端 API） | 600s |
| embedding | text_vectorization | 文本向量化 | 600s |
| clip | image_text_vectorization | 图像向量化 | 180s |

### 插件开发

详见 [插件开发指南](docs/plugin-development.md)。

## 架构

### 索引流水线（scene 模式）

1. **视频分析** — ffprobe 提取元数据
2. **场景检测** — PySceneDetect 检测场景边界
3. **帧提取** — ffmpeg 提取关键帧
4. **AI 模型推理** — 人脸检测 → 物体检测 → VLM 描述 → 文本嵌入 → 图像嵌入
5. **存储** — 写入 SQLite + 向量索引
6. **持久化** — 保存向量索引文件

### 搜索系统

融合五种信号：
- **人脸信号** — 人脸特征向量余弦相似度
- **物体信号** — jieba 分词 + 同义词扩展 + 标签匹配
- **场景 CLIP** — 文本向量 + 分类路由 + 向量检索
- **BM25 文本** — FTS5 全文搜索 + jieba + 近邻加权
- **图像信号** — CLIP 图像向量检索

### 存储

- **SQLite** (`index.db`) — 视频、片段、检测数据，FTS5 全文索引
- **向量索引** — 余弦相似度暴力搜索（JSON 序列化）
- **文件存储** — 关键帧图片、人脸图片

## 项目结构

```
src/
  main.rs              CLI 入口
  plugins/             插件系统
    daemon.rs          守护进程服务端
    client.rs          守护进程客户端
    manager.rs         插件管理器
    process.rs         插件进程管理
    protocol.rs        NDJSON 协议
    handlers/          插件处理器实现
  core/                核心逻辑
    indexer.rs         索引编排
    pipeline.rs        处理流水线
    searcher.rs        搜索引擎
  storage/             存储层
  web/                 Web 服务
plugins/               Python 插件
web/                   React 前端
```

## License

MIT
