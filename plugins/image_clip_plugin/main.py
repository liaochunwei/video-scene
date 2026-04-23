import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from plugin_sdk import run_plugin

# Model storage directory: plugins/models/clip/
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "image_clip")

_model = None
_preprocess = None

def get_model(model_name="ViT-B-16"):
    global _model, _preprocess
    if _model is None:
        os.makedirs(_MODELS_DIR, exist_ok=True)
        from cn_clip.clip import load_from_name
        device = "cuda" if _is_cuda_available() else "cpu"
        # Download models from ModelScope to plugins/models/clip/
        _model, _preprocess = load_from_name(
            model_name, device=device, vision_model_name=model_name,
            download_root=_MODELS_DIR, use_modelscope=True,
        )
        _model.eval()
    return _model, _preprocess

def reset_models():
    """Reset all model state for clean unload."""
    global _model, _preprocess
    _model = None
    _preprocess = None

def _is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def handler(action, data, options, send_progress):
    if action == "encode_image":
        return encode_image(data, options, send_progress)
    elif action == "encode_images_batch":
        return encode_images_batch(data, options, send_progress)
    else:
        raise ValueError(f"Unknown action: {action}")

def encode_image(data, options, send_progress):
    image_path = data["image_path"]
    send_progress("Loading Chinese-CLIP model...", 0, 1)
    model, preprocess = get_model()

    import torch
    from PIL import Image

    send_progress("Encoding image...", 0, 1)
    device = next(model.parameters()).device
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)

    send_progress("Image encoded", 1, 1)
    return {"feature": features[0].cpu().tolist()}

def encode_images_batch(data, options, send_progress):
    """Encode multiple images in a single call (avoids model reload overhead)."""
    image_paths = data["image_paths"]
    send_progress("Loading Chinese-CLIP model...", 0, len(image_paths))
    model, preprocess = get_model()

    import torch
    from PIL import Image

    device = next(model.parameters()).device
    all_features = []
    for i, image_path in enumerate(image_paths):
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)
        all_features.append({"image_path": image_path, "feature": features[0].cpu().tolist()})
        send_progress("Encoding images", i + 1, len(image_paths))

    return {"results": all_features}


if __name__ == "__main__":
    run_plugin(handler, "image_text_vectorization", ["encode_image", "encode_images_batch"])
