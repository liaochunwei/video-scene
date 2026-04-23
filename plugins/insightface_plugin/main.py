import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from plugin_sdk import run_plugin

# Model storage directory: plugins/models/insightface/
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "insightface")

_model = None

def get_model():
    global _model
    if _model is None:
        os.makedirs(_MODELS_DIR, exist_ok=True)
        from insightface.app import FaceAnalysis
        _model = FaceAnalysis(name="buffalo_l", root=_MODELS_DIR)
        _model.prepare(ctx_id=0, det_size=(640, 640))
    return _model

def handler(action, data, options, send_progress):
    if action == "detect":
        return detect_faces(data, options, send_progress)
    elif action == "detect_batch":
        return detect_faces_batch(data, options, send_progress)
    elif action == "encode":
        return encode_face(data, options, send_progress)
    else:
        raise ValueError(f"Unknown action: {action}")

def detect_faces(data, options, send_progress):
    image_path = data["image_path"]
    min_confidence = data.get("min_confidence", 0.8)

    send_progress("Loading InsightFace model...", 0, 1)
    model = get_model()

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    send_progress("Detecting faces...", 0, 1)
    faces = model.get(img)
    result = []
    for i, face in enumerate(faces):
        if face.det_score < min_confidence:
            continue
        bbox = face.bbox.tolist()
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        result.append({
            "bbox": [bbox[0], bbox[1], w, h],
            "confidence": float(face.det_score),
            "feature": face.embedding.tolist(),
            "quality": float(face.det_score),
        })
        send_progress("Detecting faces", i + 1, len(faces))

    return {"faces": result}

def detect_faces_batch(data, options, send_progress):
    """Detect faces in multiple images in a single call (avoids reloading model)."""
    image_paths = data["image_paths"]
    min_confidence = data.get("min_confidence", 0.8)

    send_progress("Loading InsightFace model...", 0, len(image_paths))
    model = get_model()

    import cv2
    all_results = []
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            all_results.append({"image_path": image_path, "faces": []})
            continue

        faces = model.get(img)
        frame_faces = []
        for face in faces:
            if face.det_score < min_confidence:
                continue
            bbox = face.bbox.tolist()
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            frame_faces.append({
                "bbox": [bbox[0], bbox[1], w, h],
                "confidence": float(face.det_score),
                "feature": face.embedding.tolist(),
                "quality": float(face.det_score),
            })

        all_results.append({"image_path": image_path, "faces": frame_faces})
        send_progress("Detecting faces", i + 1, len(image_paths))

    return {"results": all_results}

def encode_face(data, options, send_progress):
    image_path = data["image_path"]
    send_progress("Loading InsightFace model...", 0, 1)
    model = get_model()

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    send_progress("Encoding face...", 0, 1)
    faces = model.get(img)
    if not faces:
        raise ValueError("No face detected")

    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    send_progress("Face encoded", 1, 1)
    return {"feature": best_face.embedding.tolist()}

if __name__ == "__main__":
    run_plugin(handler, "face", ["detect", "detect_batch", "encode"])
