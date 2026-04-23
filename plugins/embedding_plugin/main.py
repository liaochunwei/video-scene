import sys
import os
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from plugin_sdk import run_plugin

# Model storage directory: plugins/models/harrier/
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "harrier")

# ONNX session and tokenizer (lazy loaded)
_session = None
_tokenizer = None

# Category label cache: computed once when model is first loaded
_category_vectors = None

# Instruction prefixes for different tasks
SEARCH_INSTRUCT = "Instruct: 在视频片段描述中检索与查询相关的内容\nQuery: "
CATEGORY_INSTRUCT = "Instruct: 判断查询的描述偏向于查询哪些类别\nQuery: "

CATEGORY_LABELS = {
    "person": "人 人物 女性 男性 穿着 妆容 发型",
    "foreground": "前景物 物品 产品 口红 化妆品 食品 玻璃杯 杯子",
    "background": "背景物 家具 台灯 花瓶 柜子 置物架",
    "scene": "场 环境 氛围 室内 室外 居家",
    "action": "动作 展示 涂抹 说话 喝水 讲解",
    "marks": "标识 水印 贴图 标签 账号",
}

MODEL_ID = "onnx-community/harrier-oss-v1-0.6b-ONNX"


def get_model():
    """Load Harrier ONNX model and tokenizer. Returns (session, tokenizer)."""
    global _session, _tokenizer
    if _session is None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        os.makedirs(_MODELS_DIR, exist_ok=True)

        # Load ONNX model from local directory
        model_path = os.path.join(_MODELS_DIR, "model_q4.onnx")

        # Create ONNX Runtime session — use CPU only (CoreML partial support causes slowdown)
        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

        # Load tokenizer from bundled models directory (tokenizer.json guaranteed at install time).
        _tokenizer = AutoTokenizer.from_pretrained(_MODELS_DIR, use_fast=True, local_files_only=True)

        # Pre-compute category label vectors for search routing
        _compute_category_vectors()

    return _session, _tokenizer


def _encode_text_internal(texts, instruction_prefix=None):
    """Encode texts using Harrier model.

    Args:
        texts: List of strings to encode.
        instruction_prefix: Optional prefix to prepend to each text (e.g., SEARCH_INSTRUCT).

    Returns:
        List of 1024-dim L2-normalized float vectors.
    """
    session, tokenizer = get_model()

    # Prepend instruction prefix if provided
    if instruction_prefix:
        input_texts = [instruction_prefix + t for t in texts]
    else:
        input_texts = list(texts)

    # Tokenize
    encoded = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    # Run ONNX inference
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    # Some ONNX models also need token_type_ids
    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

    outputs = session.run(None, inputs)
    # Output shape: (batch_size, 1024) — model already does pooling and normalization
    embeddings = outputs[0]

    return embeddings.tolist()


def _compute_category_vectors():
    """Pre-compute Harrier vectors for all category labels. Called once after model load."""
    global _category_vectors
    if _session is None:
        return

    labels = list(CATEGORY_LABELS.values())
    vectors = _encode_text_internal(labels, instruction_prefix=CATEGORY_INSTRUCT)

    _category_vectors = {}
    for i, key in enumerate(CATEGORY_LABELS.keys()):
        _category_vectors[key] = vectors[i]


def reset_models():
    """Reset all model state for clean unload."""
    global _session, _tokenizer, _category_vectors
    _session = None
    _tokenizer = None
    _category_vectors = None


def handler(action, data, options, send_progress):
    if action == "encode_text":
        return encode_text(data, options, send_progress)
    elif action == "encode_texts_batch":
        return encode_texts_batch(data, options, send_progress)
    elif action == "encode_document":
        return encode_document(data, options, send_progress)
    elif action == "encode_documents_batch":
        return encode_documents_batch(data, options, send_progress)
    elif action == "encode_text_with_categories":
        return encode_text_with_categories(data, options, send_progress)
    else:
        raise ValueError(f"Unknown action: {action}")


def encode_text(data, options, send_progress):
    """Encode a single search query with SEARCH_INSTRUCT prefix."""
    text = data["text"]
    send_progress("Loading Harrier embedding model...", 0, 1)
    send_progress("Encoding text...", 0, 1)
    vectors = _encode_text_internal([text], instruction_prefix=SEARCH_INSTRUCT)
    send_progress("Text encoded", 1, 1)
    return {"vector": vectors[0]}


def encode_texts_batch(data, options, send_progress):
    """Encode multiple search queries with SEARCH_INSTRUCT prefix."""
    texts = data["texts"]
    send_progress("Loading Harrier embedding model...", 0, len(texts))
    send_progress("Encoding texts...", 0, len(texts))
    vectors = _encode_text_internal(texts, instruction_prefix=SEARCH_INSTRUCT)
    send_progress("Texts encoded", len(texts), len(texts))
    results = [{"text": t, "vector": v} for t, v in zip(texts, vectors)]
    return {"results": results}


def encode_document(data, options, send_progress):
    """Encode a single document (no instruction prefix)."""
    text = data["text"]
    send_progress("Loading Harrier embedding model...", 0, 1)
    send_progress("Encoding document...", 0, 1)
    vectors = _encode_text_internal([text])
    send_progress("Document encoded", 1, 1)
    return {"vector": vectors[0]}


def encode_documents_batch(data, options, send_progress):
    """Encode multiple documents (no instruction prefix)."""
    texts = data["texts"]
    send_progress("Loading Harrier embedding model...", 0, len(texts))
    send_progress("Encoding documents...", 0, len(texts))
    vectors = _encode_text_internal(texts)
    send_progress("Documents encoded", len(texts), len(texts))
    results = [{"text": t, "vector": v} for t, v in zip(texts, vectors)]
    return {"results": results}


def encode_text_with_categories(data, options, send_progress):
    """Encode query text with SEARCH_INSTRUCT and return query vector + cached category vectors."""
    text = data["text"]
    send_progress("Loading Harrier embedding model...", 0, 1)
    send_progress("Encoding text...", 0, 1)
    vectors = _encode_text_internal([text], instruction_prefix=SEARCH_INSTRUCT)
    send_progress("Text encoded", 1, 1)
    return {
        "query_vector": vectors[0],
        "category_vectors": _category_vectors if _category_vectors else {},
    }


if __name__ == "__main__":
    run_plugin(handler, "text_vectorization", ["encode_text", "encode_texts_batch", "encode_document", "encode_documents_batch", "encode_text_with_categories"])
