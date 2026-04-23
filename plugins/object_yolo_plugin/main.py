import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from plugin_sdk import run_plugin

# Model storage directory: plugins/models/
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Point ultralytics weights_dir to our models directory so mobileclip etc. download there
from ultralytics.utils import SettingsManager
_settings = SettingsManager()
_settings.update(weights_dir=_MODELS_DIR)

_model = None
_current_classes = None

# Default open-vocabulary classes for YOLOE — covers COCO + cosmetics + common objects
DEFAULT_CLASSES = [
    # COCO core
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    # Cosmetics & Beauty
    "lipstick", "perfume", "mascara", "eyeshadow", "foundation", "blush",
    "nail polish", "lip gloss", "concealer", "eyeliner", "makeup brush",
    "compact", "cream", "mirror", "comb", "razor",
    # Food & Drink
    "rice", "noodles", "dumpling", "sushi", "steak", "salad", "soup",
    "bread", "egg", "milk", "coffee", "tea", "juice", "beer", "water",
    "chocolate", "ice cream", "cookie", "grape", "strawberry", "watermelon",
    "mango", "peach", "cherry", "lemon", "pear", "pineapple",
    # Home
    "curtain", "carpet", "cushion", "candle", "lamp", "chandelier",
    "blanket", "pillow", "shelf", "wardrobe", "door", "window",
    "fan", "air conditioner", "plant", "flower", "bouquet",
    # Photography
    "camera", "lens", "tripod", "microphone", "speaker", "headphones",
    "monitor", "projector", "ring light", "drone", "gimbal",
    # Signs
    "arrow", "watermark", "warning sign", "billboard", "poster", "banner",
    "qr code", "barcode", "logo",
    # Clothing
    "dress", "skirt", "coat", "jacket", "shirt", "sweater", "jeans",
    "hat", "scarf", "gloves", "shoes", "boots", "sneakers", "heels", "sandals",
    "watch", "ring", "necklace", "earring", "bracelet", "glasses", "sunglasses",
    # Toys & Stationery
    "toy", "doll", "puzzle", "ball", "balloon",
    "pen", "pencil", "notebook", "paper", "stapler", "tape", "ruler",
]

LABEL_ZH = {
    # People & Animals
    "person": "人", "bicycle": "自行车", "bird": "鸟", "cat": "猫",
    "dog": "狗", "horse": "马", "sheep": "羊", "cow": "牛",
    "elephant": "大象", "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿",
    # Vehicles
    "car": "汽车", "motorcycle": "摩托车", "airplane": "飞机",
    "bus": "公交车", "train": "火车", "truck": "卡车", "boat": "船",
    # Traffic & Signs
    "traffic light": "红绿灯", "fire hydrant": "消防栓",
    "stop sign": "停车标志", "parking meter": "停车计时器",
    # Furniture & Home
    "bench": "长椅", "chair": "椅子", "couch": "沙发",
    "bed": "床", "dining table": "餐桌", "toilet": "马桶",
    "potted plant": "盆栽", "vase": "花瓶",
    # Kitchen & Food
    "bottle": "瓶子", "wine glass": "酒杯", "cup": "杯子",
    "fork": "叉子", "knife": "刀", "spoon": "勺子", "bowl": "碗",
    "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
    "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈",
    "cake": "蛋糕", "microwave": "微波炉", "oven": "烤箱",
    "toaster": "烤面包机", "sink": "水槽", "refrigerator": "冰箱",
    # Electronics
    "tv": "电视", "laptop": "笔记本电脑", "mouse": "鼠标",
    "remote": "遥控器", "keyboard": "键盘", "cell phone": "手机",
    # Accessories & Bags
    "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包",
    "tie": "领带", "suitcase": "行李箱",
    # Sports
    "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "单板滑雪板",
    "sports ball": "球", "kite": "风筝", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板",
    "surfboard": "冲浪板", "tennis racket": "网球拍",
    # Household items
    "book": "书", "clock": "时钟", "scissors": "剪刀",
    "teddy bear": "泰迪熊", "hair dier": "吹风机", "toothbrush": "牙刷",
    # Cosmetics & Beauty
    "lipstick": "口红", "perfume": "香水", "cream": "面霜",
    "mascara": "睫毛膏", "eyeshadow": "眼影",
    "foundation": "粉底", "blush": "腮红", "nail polish": "指甲油",
    "compact": "粉饼", "lip gloss": "唇釉", "concealer": "遮瑕膏",
    "eyeliner": "眼线笔", "makeup brush": "化妆刷", "mirror": "镜子",
    "comb": "梳子", "razor": "剃须刀",
    # Food & Drink
    "rice": "米饭", "noodles": "面条", "dumpling": "饺子",
    "sushi": "寿司", "steak": "牛排", "salad": "沙拉", "soup": "汤",
    "bread": "面包", "egg": "鸡蛋", "milk": "牛奶", "coffee": "咖啡", "tea": "茶",
    "juice": "果汁", "beer": "啤酒", "water": "水",
    "chocolate": "巧克力", "ice cream": "冰淇淋", "cookie": "饼干",
    "grape": "葡萄", "strawberry": "草莓", "watermelon": "西瓜",
    "mango": "芒果", "peach": "桃子", "cherry": "樱桃",
    "lemon": "柠檬", "pear": "梨", "pineapple": "菠萝",
    # Home Decor
    "curtain": "窗帘", "carpet": "地毯", "cushion": "靠垫",
    "candle": "蜡烛", "lamp": "灯", "chandelier": "吊灯",
    "blanket": "毯子", "pillow": "枕头",
    "shelf": "置物架", "wardrobe": "衣柜", "door": "门", "window": "窗户",
    "fan": "风扇", "air conditioner": "空调",
    "plant": "植物", "flower": "花", "bouquet": "花束",
    # Photography
    "camera": "相机", "lens": "镜头", "tripod": "三脚架",
    "microphone": "麦克风", "speaker": "音箱", "headphones": "耳机",
    "monitor": "显示器", "projector": "投影仪",
    "ring light": "环形灯", "drone": "无人机", "gimbal": "稳定器",
    # Signs
    "arrow": "箭头", "watermark": "水印", "warning sign": "警示牌",
    "billboard": "广告牌", "poster": "海报", "banner": "横幅",
    "qr code": "二维码", "barcode": "条形码", "logo": "标志",
    # Clothing
    "dress": "连衣裙", "skirt": "裙子", "coat": "外套",
    "jacket": "夹克", "shirt": "衬衫", "sweater": "毛衣", "jeans": "牛仔裤",
    "hat": "帽子", "scarf": "围巾", "gloves": "手套", "shoes": "鞋子",
    "boots": "靴子", "sneakers": "运动鞋", "heels": "高跟鞋", "sandals": "凉鞋",
    "watch": "手表", "ring": "戒指", "necklace": "项链", "earring": "耳环",
    "bracelet": "手链", "glasses": "眼镜", "sunglasses": "太阳镜",
    # Toys & Stationery
    "toy": "玩具", "doll": "玩偶", "puzzle": "拼图", "ball": "球", "balloon": "气球",
    "pen": "笔", "pencil": "铅笔", "notebook": "笔记本", "paper": "纸",
    "stapler": "订书机", "tape": "胶带", "ruler": "尺子",
}


def get_model(model_name="yoloe-26s-seg.pt", classes=None):
    global _model, _current_classes
    if classes is None:
        classes = DEFAULT_CLASSES

    os.makedirs(_MODELS_DIR, exist_ok=True)

    need_reload = _model is None
    need_set_classes = _current_classes != classes

    if need_reload:
        from ultralytics import YOLOE
        model_path = os.path.join(_MODELS_DIR, model_name)
        if os.path.exists(model_path):
            _model = YOLOE(model_path)
        else:
            _model = YOLOE(model_name)

    if need_set_classes:
        _model.set_classes(classes)
        _current_classes = list(classes)

    return _model


def handler(action, data, options, send_progress):
    if action == "detect":
        return detect_objects(data, options, send_progress)
    elif action == "detect_batch":
        return detect_objects_batch(data, options, send_progress)
    else:
        raise ValueError(f"Unknown action: {action}")


def _parse_results(results, classes=None):
    objects = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            if classes and label not in classes:
                continue
            xyxy = box.xyxy[0].tolist()
            objects.append({
                "label": label,
                "label_zh": LABEL_ZH.get(label, label),
                "confidence": float(box.conf[0]),
                "bbox": [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
            })
    return objects


def detect_objects(data, options, send_progress):
    image_path = data["image_path"]
    min_confidence = data.get("min_confidence", 0.5)
    classes = data.get("classes", None)
    model_name = data.get("model", "yoloe-26s-seg.pt")

    send_progress("Loading YOLOE model...", 0, 1)
    model = get_model(model_name, classes)

    send_progress("Detecting objects...", 0, 1)
    results = model.predict(image_path, conf=min_confidence, verbose=False)
    objects = _parse_results(results, classes)
    send_progress("Detecting objects", 1, 1)

    return {"objects": objects}


def detect_objects_batch(data, options, send_progress):
    """Detect objects in multiple images in a single call (avoids model reload overhead)."""
    image_paths = data["image_paths"]
    min_confidence = data.get("min_confidence", 0.5)
    classes = data.get("classes", None)
    model_name = data.get("model", "yoloe-26s-seg.pt")

    send_progress("Loading YOLOE model...", 0, len(image_paths))
    model = get_model(model_name, classes)

    all_results = []
    for i, image_path in enumerate(image_paths):
        results = model.predict(image_path, conf=min_confidence, verbose=False)
        frame_objects = _parse_results(results, classes)
        all_results.append({"image_path": image_path, "objects": frame_objects})
        send_progress("Detecting objects", i + 1, len(image_paths))

    return {"results": all_results}


if __name__ == "__main__":
    run_plugin(handler, "object", ["detect", "detect_batch"])
