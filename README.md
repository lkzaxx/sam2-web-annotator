# SAM2 Web Annotator

Click-driven multi-class polygon annotator for YOLO-seg datasets, powered by **Meta SAM 2.1**.

Browser → click a few positive / negative points (or drag a box) → SAM2 returns 3 candidate masks → pick one → save as YOLO-seg polygon. Designed for thin / elongated / partially-occluded objects (ropes, cables, harness straps, hooks…) where bbox-only labeling is not enough.

## Why this exists

Label Studio + SAM2 ML backend is the textbook stack, but it's heavyweight (separate Label Studio server + ML backend + a DB) for the common case of "one annotator, ~hundreds of images, YOLO output." This is a single Flask app, ~600 lines, that does the same job for that case.

## Features

- **Multi-class** polygon annotation (configure via `classes.json`)
- **SAM 2.1** (Hiera Large by default) running on CUDA
- **3 candidate masks** per click set with quality scores — pick the best
- **+ / – click prompts** and **box prompt**, mix them
- **Pan & zoom** canvas (mouse wheel + middle-button drag)
- **YOLO-seg** output: one `.txt` per image, one polygon per line, normalized coords
- **Per-image undo** + clear, **per-class instance count** in sidebar
- Keyboard shortcuts: `W/E/B` modes · `A/D` prev/next · `1-9` pick mask · `R` reset · `Z` undo · `Enter` save

## Quickstart

```bash
# 1. install deps (in a fresh venv / conda env)
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

# 2. download SAM 2.1 checkpoint
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 3. drop images into the image dir (default: ~/sam2_annotator/images)
mkdir -p ~/sam2_annotator/images ~/sam2_annotator/labels
cp /path/to/your/*.jpg ~/sam2_annotator/images/

# 4. configure classes (optional; default = single class "object")
cat > classes.json <<EOF
[
  {"id": 0, "name": "harness", "color": "#ff3366"},
  {"id": 1, "name": "hook",    "color": "#ffaa00"},
  {"id": 2, "name": "rope",    "color": "#3399ff"}
]
EOF

# 5. run
SAM2_CHECKPOINT=$(pwd)/checkpoints/sam2.1_hiera_large.pt python app.py

# 6. open http://<host>:8081/
```

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `SAM2_IMG_DIR`    | `~/sam2_annotator/images` | Where input images live |
| `SAM2_LABEL_DIR`  | `~/sam2_annotator/labels` | Where `.txt` output goes |
| `SAM2_CHECKPOINT` | `~/checkpoints/sam2.1_hiera_large.pt` | SAM 2.1 weight file |
| `SAM2_CFG`        | `configs/sam2.1/sam2.1_hiera_l.yaml` | SAM 2.1 model config (built-in to sam2 package) |
| `SAM2_DEVICE`     | `cuda` | `cuda` / `cpu` / `mps` |
| `SAM2_HOST`       | `0.0.0.0` | bind address |
| `SAM2_PORT`       | `8081`    | bind port |

Use other SAM 2.1 sizes by swapping checkpoint + config:

| Variant | Checkpoint URL suffix | Config |
|---|---|---|
| tiny      | `sam2.1_hiera_tiny.pt`      | `configs/sam2.1/sam2.1_hiera_t.yaml` |
| small     | `sam2.1_hiera_small.pt`     | `configs/sam2.1/sam2.1_hiera_s.yaml` |
| base_plus | `sam2.1_hiera_base_plus.pt` | `configs/sam2.1/sam2.1_hiera_b+.yaml` |
| large     | `sam2.1_hiera_large.pt`     | `configs/sam2.1/sam2.1_hiera_l.yaml` |

Base URL: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/`

## Output format (YOLO-seg)

For each image `foo.jpg`, a parallel file `foo.txt` in `SAM2_LABEL_DIR`:

```
<class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ... <xN_norm> <yN_norm>
```

Coordinates are normalized to `[0, 1]` against image width / height. One polygon per line; multiple polygons per image allowed. Directly consumable by `ultralytics` YOLO-seg training.

## REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/classes`              | GET    | class config list |
| `/api/images`               | GET    | list images + labeled status |
| `/api/image/<name>`         | GET    | serve image bytes |
| `/api/label/<name>`         | GET    | existing polygons in pixel coords |
| `/api/segment`              | POST   | `{image, points, labels, box}` → 3 candidate masks (base64 PNG) |
| `/api/save`                 | POST   | `{image, class_id, mask}` → append polygon to `.txt` |
| `/api/undo/<name>`          | POST   | remove last polygon |
| `/api/clear/<name>`         | DELETE | wipe label file |

## Workflow

1. **Bulk-load** images into `SAM2_IMG_DIR`
2. Open browser, pick a class
3. Click 1-N positive points on the object → SAM2 returns 3 candidates → click to pick best
4. Add negative clicks if mask leaks → re-runs SAM2 automatically
5. **Enter** to save polygon to YOLO `.txt`
6. **A / D** to navigate; left sidebar shows progress

For thin objects (rope, cable), 3-5 clicks along the centerline usually works. For elongated objects with similar-colored background, supplement with 1-2 negatives outside the object.

## Acknowledgements

UI pattern adapted from a hand-keypoint annotator we built earlier (SAM v1 + click-to-cutout). This repo upgrades to **SAM 2.1**, adds **multi-class polygon output in YOLO-seg format**, and is configured via env vars so it can be reused on any dataset.

## License

MIT
