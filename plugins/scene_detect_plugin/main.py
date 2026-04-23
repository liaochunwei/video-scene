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
    detector_type = data.get("detector", "content")
    threshold = data.get("threshold", 27.0)

    send_progress("Loading video...", 0, 1)
    from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector

    video = open_video(video_path)
    scene_manager = SceneManager()

    if detector_type == "adaptive":
        scene_manager.add_detector(AdaptiveDetector(threshold=threshold))
    else:
        scene_manager.add_detector(ContentDetector(threshold=threshold))

    send_progress("Detecting scenes...", 0, 1)
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scenes.append({"start": start_time, "end": end_time})
        send_progress("Detecting scenes", i + 1, len(scene_list) if scene_list else 1)

    if not scenes:
        duration = video.duration.get_seconds()
        scenes.append({"start": 0.0, "end": duration})

    return {"scenes": scenes}

if __name__ == "__main__":
    run_plugin(handler, "video_segmentation", ["detect_scenes"])
