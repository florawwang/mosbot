# mosbot

**mosbot** is a local web app for mosquito activity experiments: browse tracked frames, audit YOLO detections, and plot LD/DD actograms.

Built for circadian pipeline.

**Live app:** [https://mosbot.streamlit.app](https://mosbot.streamlit.app)

---

## Features

| Section | What it does |
|---------|----------------|
| **Frame images** | Scrub through timelapse frames with well crops and detection overlays |
| **Detection inspector** | Check where YOLO misses mosquitoes (threshold slider, jump to misses, flag frames) |
| **Activity graphs** | Actograms, LD/DD profiles, day/night totals, and stats from an activity CSV |

---

## Quick start

**Requirements:** Python 3.10–3.12

```bash
git clone https://github.com/florawwang/mosbot.git
cd mosbot
pip install -r requirements.txt
./run_app.sh
```

Open [http://127.0.0.1:8502](http://127.0.0.1:8502) and enter the lab passcode.

Stop the app with `Ctrl+C`.

### Useful options

```bash
# Different port if 8502 is taken
PORT=8503 ./run_app.sh

# Custom passcode for this session
CLOUD_VIEWER_PASSCODE=yourcode ./run_app.sh
```

For YOLO inference and detection-cache builds (heavier deps):

```bash
pip install -r requirements-ml.txt
```

---

## What files you need

| Input | Description |
|-------|-------------|
| **Raw images** | Folder of timelapse frames (`.jpg` / `.png`) |
| **Labels CSV** | MakeSense well boxes (one row per mosquito) |
| **YOLO model** | Detector weights (e.g. `uninf_det_v0.pt`) — local inference only |

After inference you also get:

- `activity_transposed.csv` — per-frame movement per mosquito  
- `frame_manifest.json` — saved detection positions for fast browsing  

Set paths in the app sidebar (or use env vars / auto-discovery when nested in the lab data repo).

---

## Typical workflow

1. **Run inference** (laptop or VM — can take hours on CPU) → activity CSV + manifest  
2. **Open mosbot** → browse frames, inspect detections, plot graphs  

Skip step 1 if someone already shared an output folder with you.

### Inference (local)

From this repo:

```bash
export PYTHONPATH="."
pip install -r requirements-ml.txt

python -m mosquito_lab.run_inference \
  --image-folder "/path/to/raw_images" \
  --labels "/path/to/labels.csv" \
  --model "/path/to/uninf_det_v0.pt" \
  --output-dir ./mosquito-lab-output \
  --serve-viewer
```

Background run:

```bash
./run_cloud.sh local "/path/to/raw_images" "/path/to/labels.csv" "/path/to/model.pt"
```


## Deploy (Streamlit Community Cloud)

Hosted at **[https://mosbot.streamlit.app](https://mosbot.streamlit.app)**.

| Setting | Value |
|---------|--------|
| Main file | `streamlit_app.py` |
| Python | 3.10–3.12 |

Live YOLO and cache builds are meant for a laptop/VM. On Cloud, upload saved CSVs and manifests.

If the app fails to start, open **Manage app → Logs** on [share.streamlit.io](https://share.streamlit.io).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| App won't start | `pip install -r requirements.txt`; use Python 3.10–3.12 |
| Port already in use | `PORT=8503 ./run_app.sh` |
| No images / empty graphs | Check sidebar paths; load a CSV or run inference first |
| Detection inspector needs YOLO | `pip install -r requirements-ml.txt`, then run `precompute.py` |
| Wrong passcode | Ask a lab member, or set `CLOUD_VIEWER_PASSCODE` |

