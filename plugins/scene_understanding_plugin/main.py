import sys
import os
import json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from plugin_sdk import run_plugin

# Model storage directory: plugins/models/vlm/
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "vlm")

_model = None
_processor = None

SYSTEM_PROMPT = """你是一个视觉识别专家，按时间顺序对这个视频片段做详细的内容理解和分析，包括主体信息、主体动作、展示的产品、场景信息及变化、还有背景、标识等信息。
## 要求：
* 用中文描述，总结输出，描述准确有细节；
* 详细目标检测项如下，注意识别每张图中的细节，没有的特征不用分析，信息要完整：
  - 人：什么样的人（表示视频主体，可能只是手、脚、眼、嘴，也可能是动物，还可能建筑或物品等）, 妆容、发型、穿搭、指甲、黑眼圈、表情、状态等**详细细节描述**，**不能出同一个、另一个类似表达**
  - 前景物：视频前景有些什么食品、物品、装饰等 **不包括贴图、标签**
  - 背景物：视频背景有些什么食品、物品、装饰等
  - 场：在什么样的环境，什么样的氛围
  - 动作：做了什么事
  - 标识：描述有什么样的贴图、标签(**不是字幕**)、水印（**不是字幕**一般是半透明的文字）、指向性标识等，按类型-描述信息填写，基本为图形，注意水印、标签、字幕区分开
  - 字幕：字幕（一般是文字形式在视频靠中下的位置）
* JSON格式输出，用中文输出描述信息，总结输出；
* 格式示例：
```json
{
    "人": "...",
    "前景物": "...",
    "背景物": "...",
    "场": "...",
    "动作": "...",
    "标识": ["水印-A","贴图-指下箭头","贴图-白色箭头","..."]
    "字幕": ["字幕A","字幕B"]
}
```
"""


def get_model(model_name=None):
    global _model, _processor
    if _model is None:
        from mlx_vlm import load
        if model_name is not None and not model_name.strip():
            model_name = None
        if model_name is None:
            if os.path.isdir(_MODELS_DIR):
                for d in sorted(os.listdir(_MODELS_DIR)):
                    candidate = os.path.join(_MODELS_DIR, d)
                    if os.path.isdir(candidate):
                        model_name = candidate
                        break
            if model_name is None:
                model_name = "Qwen3.5-4B-MLX-4bit"
        elif not os.path.isabs(model_name):
            local_path = os.path.join(_MODELS_DIR, model_name)
            if os.path.isdir(local_path):
                model_name = local_path

        _model, _processor = load(model_name)
    return _model, _processor


_THINK_CLOSE = chr(60) + "/think" + chr(62)


def strip_thinking(text):
    pos = text.find(_THINK_CLOSE)
    if pos != -1:
        return text[pos + len(_THINK_CLOSE):].strip()
    return text.strip()


def fix_misplaced_subtitles(structured):
    """将标识中误放的字幕条目移到字幕字段。

    VLM 有时会把字幕内容输出到标识数组里，这里做后处理修正：
    - 以"字幕-"开头的条目视为字幕，提取"字幕-"后面的部分作为字幕文本
    - 不以"字幕-"开头但内容明显是字幕的（如纯文字描述），也移到字幕
    - 只有当字幕字段当前为空时才迁移，避免重复
    """
    marks = structured.get("标识", [])
    subtitles = structured.get("字幕", [])

    if not marks:
        return

    # 拆分：以"字幕-"开头的归入字幕，其余保留在标识
    new_marks = []
    moved_subtitles = []
    for item in marks:
        if item.startswith("字幕-"):
            # 去掉"字幕-"前缀，保留实际字幕文本
            moved_subtitles.append(item[len("字幕-"):])
        elif item.startswith("字幕"):
            # "字幕xxx" 形式也视为字幕
            moved_subtitles.append(item[len("字幕"):])
        else:
            new_marks.append(item)

    # 只有字幕字段为空时才迁移，避免与 VLM 正确输出的字幕重复
    if moved_subtitles and not subtitles:
        structured["字幕"] = moved_subtitles
    structured["标识"] = new_marks


def parse_structured(text):
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end+1])
            structured = {
                "人": obj.get("人", ""),
                "前景物": obj.get("前景物", ""),
                "背景物": obj.get("背景物", ""),
                "场": obj.get("场", ""),
                "动作": obj.get("动作", ""),
                "标识": obj.get("标识", []) if isinstance(obj.get("标识"), list) else [],
                "字幕": obj.get("字幕", []) if isinstance(obj.get("字幕"), list) else [],
            }
            fix_misplaced_subtitles(structured)
            return structured
        except json.JSONDecodeError:
            pass
    return {
        "人": text, "前景物": "", "背景物": "", "场": "", "动作": "",
        "标识": [], "字幕": []
    }


def build_prompt(processor, image_count):
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    content = []
    for _ in range(image_count):
        content.append({"type": "image"})
    content.append({"type": "text", "text": "图为视频片段顺序截图，请分析！"})
    messages.append({"role": "user", "content": content})

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt


def is_valid_structured(result):
    has_other_fields = bool(result.get("前景物") or result.get("背景物") or result.get("场") or result.get("动作"))
    has_lists = bool(result.get("标识") or result.get("字幕"))
    return has_other_fields or has_lists


def describe_scene(data, options, send_progress):
    image_paths = data["image_paths"]
    model_name = data.get("model", None)

    send_progress("Loading VLM model...", 0, 1)
    model, processor = get_model(model_name)

    send_progress("Generating description...", 0, 1)
    from mlx_vlm import generate

    valid_paths = [p for p in image_paths if os.path.isfile(p)]
    if not valid_paths:
        return {"structured": {"人": "", "前景物": "", "背景物": "", "场": "", "动作": "", "标识": [], "字幕": []}}

    prompt = build_prompt(processor, len(valid_paths))
    max_retries = 3
    for attempt in range(max_retries + 1):
        temp = 0.3 + attempt * 0.1
        output = generate(model, processor, prompt, image=valid_paths, max_tokens=2048, temperature=temp, repetition_penalty=1.1)
        description = strip_thinking(output.text)
        structured = parse_structured(description)
        if is_valid_structured(structured):
            break
        if attempt < max_retries:
            print(f"WARNING: VLM response invalid format (attempt {attempt+1}, temp={temp:.1f}), retrying...", file=sys.stderr)
        else:
            print(f"WARNING: VLM response invalid format after {max_retries+1} attempts, using fallback", file=sys.stderr)

    send_progress("Description generated", 1, 1)
    return {"structured": structured}


def describe_scenes_batch(data, options, send_progress):
    scenes = data["scenes"]
    model_name = data.get("model", None)

    send_progress("Loading VLM model...", 0, len(scenes))
    model, processor = get_model(model_name)

    from mlx_vlm import generate

    results = []
    for i, scene in enumerate(scenes):
        image_paths = scene["image_paths"]

        valid_paths = [p for p in image_paths if os.path.isfile(p)]
        if not valid_paths:
            results.append({"structured": {"人": "", "前景物": "", "背景物": "", "场": "", "动作": "", "标识": [], "字幕": []}})
            send_progress("Describing scenes", i + 1, len(scenes))
            continue

        prompt = build_prompt(processor, len(valid_paths))
        max_retries = 3
        for attempt in range(max_retries + 1):
            temp = 0.3 + attempt * 0.1
            output = generate(model, processor, prompt, image=valid_paths, max_tokens=2048, temperature=temp, repetition_penalty=1.1)
            description = strip_thinking(output.text)
            structured = parse_structured(description)
            if is_valid_structured(structured):
                break
            if attempt < max_retries:
                print(f"WARNING: VLM response invalid format (scene {i+1}, attempt {attempt+1}, temp={temp:.1f}), retrying...", file=sys.stderr)
            else:
                print(f"WARNING: VLM response invalid format for scene {i+1} after {max_retries+1} attempts, using fallback", file=sys.stderr)

        results.append({"structured": structured})
        send_progress("Describing scenes", i + 1, len(scenes))

    return {"results": results}


def handler(action, data, options, send_progress):
    if action == "describe_scenes_batch":
        return describe_scenes_batch(data, options, send_progress)
    elif action == "describe_scene":
        return describe_scene(data, options, send_progress)
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    run_plugin(handler, "image_text_understanding", ["describe_scene", "describe_scenes_batch"])
