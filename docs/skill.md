---
name: videoscene
description: Use when you need to index, search, or manage videos using the VideoScene CLI tool — face recognition, object detection, scene search, image similarity, and web UI
---

# VideoScene 使用技能

## 概述

VideoScene 是一个视频检索工具。你将用它来索引视频、搜索内容、管理人脸库。所有操作通过 `vs` 命令完成。

## 前提条件

**必须先启动插件守护进程，否则所有命令都会报错。**

```bash
vs plugins daemon
```

守护进程启动后保持运行，另开终端执行其他命令。

## 核心工作流

### 1. 索引视频

将视频内容提取为可搜索的索引：

```bash
# 索引单个视频
vs index /path/to/video.mp4

# 索引整个目录
vs index /path/to/videos/ --recursive

# 使用云端 VLM API（更快，需配置 api_key）
vs index /path/to/video.mp4 --mode video

# 强制重新索引
vs index /path/to/video.mp4 --force
```

索引过程会自动：检测场景边界 → 提取关键帧 → 人脸识别 → 物体检测 → 场景描述 → 向量化。耗时取决于视频长度和硬件。

### 2. 搜索视频

```bash
# 自然语言搜索（自动融合人脸、物体、场景信号）
vs search "女性在化妆"

# 按类型搜索
vs search "张三" --face           # 人脸搜索
vs search "汽车" --object          # 物体搜索
vs search "海边日落" --scene        # 场景搜索

# 用图片搜索相似片段
vs search --image /path/to/photo.jpg

# 控制结果
vs search "query" --top 20         # 更多结果
vs search "query" --threshold 0.5   # 调整阈值
vs search "query" --json            # JSON 输出
vs search "query" --dedup           # 合并同视频片段

# 启动 Web 界面
vs search --web                     # 默认端口 6066
vs search --web --port 8080         # 自定义端口
```

### 3. 管理人脸库

```bash
# 添加人脸（从照片）
vs face add "张三" /path/to/photo.jpg

# 添加新角度到已有的人脸
vs face add "张三" /path/to/another.jpg --add

# 从视频自动提取人脸
vs face extract /path/to/video.mp4
vs face extract /path/to/video.mp4 --auto-save   # 自动保存，不逐个确认

# 查看和管理
vs face list                        # 列出所有人脸
vs face info "张三"                  # 查看详情
vs face rename "张三" --new-name "张三丰"
vs face remove "张三"
```

## 管理命令

```bash
# 工作空间
vs workspace init <name>            # 创建工作空间
vs workspace list                    # 列出工作空间
vs workspace activate <name>        # 切换工作空间

# 索引管理
vs status                           # 查看索引状态
vs list                             # 列出已索引视频
vs remove <video_path>               # 删除单个视频索引
vs remove --all                     # 清空整个索引
vs clean                            # 清理源文件已删除的索引

# 插件管理
vs plugins status                    # 查看插件运行状态
vs plugins start <type>              # 预启动插件
vs plugins stop <type>               # 停止插件

# 配置
vs config                           # 显示当前配置
```

## 全局选项

所有命令都支持：

```bash
--workspace <name>    # 指定工作空间
--config <path>       # 指定配置文件
-v / --verbose        # 详细日志
-q / --quiet          # 安静模式
```

## 常见场景

### 场景：找到某个人的所有出场

```bash
vs face add "目标人" photo.jpg    # 先添加到人脸库
vs index video.mp4                 # 索引视频
vs search "目标人" --face          # 搜索
```

### 场景：找到包含特定物体的视频

```bash
vs index videos/ --recursive       # 索引目录
vs search "口红" --object           # 物体搜索
```

### 场景：用图片找相似片段

```bash
vs search --image query.jpg         # 图像相似搜索
```

### 场景：批量索引并 Web 搜索

```bash
vs plugins daemon                   # 终端1：启动守护进程
vs index videos/ --recursive        # 终端2：索引
vs search --web                     # 终端2：启动 Web UI
# 浏览器打开 http://localhost:6066
```

## 故障排除

| 问题 | 原因 | 解决 |
|------|------|------|
| "Plugin daemon not running" | 守护进程未启动 | `vs plugins daemon` |
| "Cannot connect to daemon" | 守护进程崩溃后 socket 残留 | `rm /tmp/vs-plugins/daemon.sock` 然后 `vs plugins daemon` |
| 索引很慢 | 模型冷启动 | 确保守护进程已启动，或用 `--mode video` |
| 搜索无结果 | 阈值过高或未索引 | 降低 `--threshold`，检查 `vs status` |
| Web UI 端口占用 | 端口冲突 | `vs search --web --port 8080` |

## 配置

配置文件：`~/.video-scene/config.toml`

关键配置项：

```toml
[plugins.vlm_api]
api_key = ""           # 云端 VLM API 密钥（--mode video 必需）

[index.scene]
min_segment_duration = 2.0   # 最小片段时长
max_segment_duration = 30.0   # 最大片段时长
```

用 `vs config` 查看完整配置。
