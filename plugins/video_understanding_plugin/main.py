import sys
import os
import json
import base64
import tempfile
import subprocess
import urllib.request
import urllib.error

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from plugin_sdk import run_plugin

SYSTEM_VIDEO_PROMPT = """# 你是一个视觉识别专家，将这个视频按 **画面主体动作意图** 顺序拆分成片段，然后所有片段独立做详细内容理解分析，包括主体信息、主体动作、展示的产品、场景信息及变化、还有背景、各种标识信息等。

## 步骤要求：
* 先思考，但不要过渡重复思考，不要做假设；
* 按顺序拆分片段，顺序逐个分析每一张图的画面主体和动作意图，主体和上一张图一致并且动作意图有承接上图动作时不做分隔；
* 每个片段的**所有图再次做详细的独立分析**，描述要准确有细节；
* 详细目标检测项如下，注意识别每张图中的细节，没有的特征不用分析，每片段信息要完整分析：
  - 片段开始: 单位为秒，精确到0.01秒
  - 片段结束: 单位为秒，精确到0.01秒
  - 人：什么样的人（表示视频主体，可能只是手、脚、眼、嘴，也可能是动物，还可能建筑或物品等）, 妆容、发型、穿搭、指甲、黑眼圈、表情、状态等**详细细节描述**，**不能出同一个、另一个类似表达**
  - 前景物：视频前景有些什么食品、物品、装饰等 **不包括贴图、标签**
  - 背景物：视频背景有些什么食品、物品、装饰等
  - 场：在什么样的环境，什么样的氛围 **不能出同上场景类似表达**
  - 动作：做了什么事
  - 标识：描述有什么样的贴图、标签(**不是字幕**)、水印（**不是字幕**一般是半透明的文字）、字幕（一般是文字形式在视频靠中下的位置）、指向性标识等，按类型-描述信息填写，基本为图形，注意水印、标签、字幕区分开
* 处理去掉同一位，同上，同一个、另一个类似表达，复制相应的内容进行填充，**所有分析结果中不能出现同一、同前、同上、延续等依赖其它片段的词**
* JSON格式输出，用中文输出描述信息，总结输出；
* 格式示例：
```json
[{
  "片段开始": 1.5,
  "片段结束": 3.0,
  "人": "...",
  "前景物": "...",
  "背景物": "...",
  "场": "...",
  "动作": "...",
  "标识": ["水印-A","贴图-指下箭头","贴图-白色箭头","字幕-AAA","..."]
}]
```
"""


def _strip_thinking(text):
    think_close = ""
    pos = text.find(think_close)
    if pos != -1:
        return text[pos + len(think_close):].strip()
    return text.strip()


def _parse_segments(text):
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end+1])
            return [obj]
        except json.JSONDecodeError:
            pass
    return None


def _is_valid_segments(segments):
    if not segments or not isinstance(segments, list):
        return False
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        has_fields = bool(seg.get("人") or seg.get("前景物") or seg.get("背景物") or seg.get("场") or seg.get("动作"))
        has_lists = bool(seg.get("标识") or seg.get("字幕"))
        if has_fields or has_lists:
            return True
    return False


def _preprocess_video(video_path, max_pixels, fps):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return video_path

    try:
        w, h = probe.stdout.strip().split("x")
        w, h = int(w), int(h)
    except ValueError:
        return video_path

    max_side = int(max_pixels ** 0.5)
    current_max = max(w, h)
    if current_max > max_side:
        scale = max_side / current_max
        new_w = int(w * scale) & ~1
        new_h = int(h * scale) & ~1
    else:
        new_w, new_h = w & ~1, h & ~1

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"scale={new_w}:{new_h}:force_original_aspect_ratio=decrease,fps={fps}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-an",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(tmp_path)
        return video_path

    return tmp_path


def _call_api(api_base, api_key, model, messages, max_tokens=24*1024, temperature=0.6):
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "enable_thinking": True,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers)
    full_content = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full_content += content
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error {e.code}: {body}")
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}")

    return full_content


def describe_video(data, options, send_progress):
    video_path = data["video_path"]
    api_base = data.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = data.get("api_key", "")
    model = data.get("model", "qwen3.6-plus")
    max_pixels = int(data.get("max_pixels", 230400))
    fps = float(data.get("fps", 2.0))

    send_progress("Preprocessing video...", 0, 1)
    preprocessed = _preprocess_video(video_path, max_pixels, fps)
    cleanup = preprocessed != video_path

    try:
        with open(preprocessed, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        b64_size = len(video_b64.encode("utf-8"))
        if b64_size > 10 * 1024 * 1024:
            raise RuntimeError(f"Encoded video size {b64_size / 1024 / 1024:.1f}MB exceeds 10MB limit")

        ext = os.path.splitext(video_path)[1].lstrip(".") or "mp4"
        mime = f"video/{ext}"

        messages = [
            {"role": "system", "content": SYSTEM_VIDEO_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:{mime};base64,{video_b64}"},
                        "fps": fps,
                    },
                    {"type": "text", "text": "这是一个视频文件，请分析！"},
                ],
            },
        ]

        send_progress("Calling VLM API...", 0, 1)
        max_retries = 3
        segments = None
        for attempt in range(max_retries + 1):
            temp = 0.7 + attempt * 0.1
            raw_text = _call_api(api_base, api_key, model, messages, temperature=temp)
            raw_text = _strip_thinking(raw_text)
            segments = _parse_segments(raw_text)

            if segments and _is_valid_segments(segments):
                break
            if attempt < max_retries:
                print(f"WARNING: VLM API invalid format (attempt {attempt+1}, temp={temp:.1f}), retrying...", file=sys.stderr)
            else:
                print(f"WARNING: VLM API invalid format after {max_retries+1} attempts, using fallback", file=sys.stderr)

        send_progress("VLM API done", 1, 1)
        return {"segments": segments or []}
    finally:
        if cleanup and os.path.exists(preprocessed):
            os.unlink(preprocessed)


def handler(action, data, options, send_progress):
    if action == "describe_video":
        return describe_video(data, options, send_progress)
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    run_plugin(handler, "video_understanding", ["describe_video"])
