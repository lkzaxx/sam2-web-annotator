"""SAM2 Web Annotator — click/box driven multi-class polygon labeler outputting YOLO-seg format."""
from flask import Flask, send_from_directory, jsonify, request
import os, json, base64
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__, static_folder="static")

IMG_DIR    = os.environ.get("SAM2_IMG_DIR",    os.path.expanduser("~/sam2_annotator/images"))
LABEL_DIR  = os.environ.get("SAM2_LABEL_DIR",  os.path.expanduser("~/sam2_annotator/labels"))
CKPT       = os.environ.get("SAM2_CHECKPOINT", os.path.expanduser("~/checkpoints/sam2.1_hiera_large.pt"))
CFG        = os.environ.get("SAM2_CFG",        "configs/sam2.1/sam2.1_hiera_l.yaml")
DEVICE     = os.environ.get("SAM2_DEVICE",     "cuda")
HOST       = os.environ.get("SAM2_HOST",       "0.0.0.0")
PORT       = int(os.environ.get("SAM2_PORT",   "8081"))

DEFAULT_CLASSES = [{"id": 0, "name": "object", "color": "#00ff66"}]
classes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.json")
if os.path.exists(classes_path):
    with open(classes_path) as f:
        CLASSES = json.load(f)
else:
    CLASSES = DEFAULT_CLASSES

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

print(f"[sam2-annotator] IMG_DIR  = {IMG_DIR}")
print(f"[sam2-annotator] LABEL_DIR= {LABEL_DIR}")
print(f"[sam2-annotator] CKPT     = {CKPT}")
print(f"[sam2-annotator] Loading SAM2 ...")
sam2_model = build_sam2(CFG, CKPT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)
print(f"[sam2-annotator] SAM2 ready (device={DEVICE})")

_cache = {"name": None}

def set_image(name):
    if _cache["name"] == name:
        return _cache["img"]
    img = cv2.imread(os.path.join(IMG_DIR, name))
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    _cache["name"] = name
    _cache["img"] = img
    return img

def mask_to_polygons(mask_bool, simplify_eps=2.0, min_area=50):
    contours, _ = cv2.findContours(
        (mask_bool.astype(np.uint8)) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, simplify_eps, True)
        if len(approx) < 3:
            continue
        polygons.append(approx.reshape(-1, 2).tolist())
    return polygons

def count_instances(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/static/<path:p>")
def static_file(p):
    return send_from_directory("static", p)

@app.route("/api/classes")
def api_classes():
    return jsonify(CLASSES)

@app.route("/api/images")
def api_images():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in exts)
    result = []
    for f in files:
        base = os.path.splitext(f)[0]
        label = os.path.join(LABEL_DIR, base + ".txt")
        result.append({"name": f, "labeled": os.path.exists(label),
                       "instances": count_instances(label)})
    return jsonify(result)

@app.route("/api/image/<path:name>")
def api_image(name):
    return send_from_directory(IMG_DIR, name)

@app.route("/api/label/<path:name>")
def api_label(name):
    """Return existing YOLO-seg polygons in pixel coords (de-normalized)."""
    base = os.path.splitext(name)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    img_path = os.path.join(IMG_DIR, name)
    if not os.path.exists(img_path):
        return jsonify({"width": 0, "height": 0, "instances": []})
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    instances = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                points = [[coords[i] * w, coords[i + 1] * h] for i in range(0, len(coords), 2)]
                instances.append({"class_id": cid, "points": points})
    return jsonify({"width": w, "height": h, "instances": instances})

@app.route("/api/segment", methods=["POST"])
def api_segment():
    data = request.json
    img = set_image(data["image"])
    h, w = img.shape[:2]
    kwargs = {"multimask_output": True}
    if data.get("points"):
        kwargs["point_coords"] = np.array(data["points"], dtype=np.float32)
        kwargs["point_labels"] = np.array(data["labels"], dtype=np.int32)
    if data.get("box"):
        kwargs["box"] = np.array(data["box"], dtype=np.float32)
    masks, scores, _ = predictor.predict(**kwargs)
    out = []
    # apply the same post-processing as /api/save so preview polygon == saved polygon
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for mask, score in zip(masks, scores):
        m_bool = mask > 0.5
        # largest connected component
        n, lbls, stats, _ = cv2.connectedComponentsWithStats(m_bool.astype(np.uint8), 8)
        if n > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            m_bool = (lbls == largest)
        # morph close
        m_u8 = cv2.morphologyEx((m_bool * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
        m_bool = m_u8 > 127
        polys = mask_to_polygons(m_bool, simplify_eps=2.0)
        _, buf = cv2.imencode(".png", m_u8)
        out.append({
            "mask":     base64.b64encode(buf).decode("utf-8"),
            "score":    float(score),
            "area":     float(m_bool.sum() / (h * w)),
            "polygons": polys,
        })
    return jsonify({"masks": out, "width": w, "height": h})

@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.json
    name = data["image"]
    cid  = int(data["class_id"])
    img  = set_image(name)
    h, w = img.shape[:2]
    mask_bytes = base64.b64decode(data["mask"])
    mask_img = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    mask_bool = mask_img > 127
    if data.get("largest_only", True):
        n, lbls, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), 8)
        if n > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_bool = (lbls == largest)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx((mask_bool * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    polys = mask_to_polygons(mask_u8 > 127,
                             simplify_eps=float(data.get("simplify_eps", 2.0)),
                             min_area=int(data.get("min_area", 50)))
    if not polys:
        return jsonify({"ok": False, "error": "polygon empty (mask too small or fragmented)"})
    base = os.path.splitext(name)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    with open(label_path, "a") as f:
        for poly in polys:
            coords = []
            for x, y in poly:
                coords.append(f"{x/w:.6f}")
                coords.append(f"{y/h:.6f}")
            f.write(f"{cid} " + " ".join(coords) + "\n")
    return jsonify({"ok": True, "polygons_added": len(polys),
                    "instance_count": count_instances(label_path)})

@app.route("/api/clear/<path:name>", methods=["DELETE"])
def api_clear(name):
    base = os.path.splitext(name)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    if os.path.exists(label_path):
        os.remove(label_path)
    return jsonify({"ok": True})

@app.route("/api/undo/<path:name>", methods=["POST"])
def api_undo(name):
    base = os.path.splitext(name)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    if not os.path.exists(label_path):
        return jsonify({"ok": False, "error": "no label file"})
    with open(label_path) as f:
        lines = [l for l in f if l.strip()]
    if not lines:
        return jsonify({"ok": False, "error": "no instances"})
    with open(label_path, "w") as f:
        f.writelines(lines[:-1])
    return jsonify({"ok": True, "remaining": len(lines) - 1})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
