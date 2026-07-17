# mosbot

Browse frames, audit detections, and plot activity in one web app.

**Ask Flora for the password** (default is `florawang`).

---

## Run locally

```bash
cd mosquito-lab
pip install -r requirements.txt   # first time only
./run_app.sh
```

Open **http://127.0.0.1:8502** and enter the passcode. Stop with `Ctrl+C`.

If port 8502 is busy: `PORT=8503 ./run_app.sh` → **http://127.0.0.1:8503**

Override passcode: `CLOUD_VIEWER_PASSCODE=yourcode ./run_app.sh`

For YOLO inference / inspector cache builds: `pip install -r requirements-ml.txt`

---

## App sections

- **Frame images** — scrub detections (`frame_manifest.json`)
- **Detection inspector** — YOLO QA (threshold slider, jump to misses, flag frames)
- **Activity graphs** — actograms / LD–DD plots from an activity CSV

Fast inspector cache (local):

```bash
PYTHONPATH=. python mosquito_lab/inspector/precompute.py --experiment "28" --stride 10
```

---

## Streamlit Cloud

**Deploy settings:**
- Main file: `streamlit_app.py` (preferred) or `mosquito_lab/lab_app.py`
- Python: **3.10–3.12** (avoid 3.14 if installs get weird)

If you see **"Error running app"**:

1. Open [share.streamlit.io](https://share.streamlit.io) → your app → **Manage app** → **Logs**
2. If logs say `No module named 'mosquito_lab'`, set Main file to `streamlit_app.py` and reboot
3. Use saved CSVs / manifests on Cloud — live YOLO & cache builds need a laptop/VM

App URL example: https://mosbot.streamlit.app

---

## What you need

| Input | Example |
|-------|---------|
| Raw images | `data/…/[29] raw_images/` |
| Labels CSV | `MosquitoMovement2/experiments/…/exp29_labels.csv` |
| YOLO model (local) | `MosquitoMovement2/models/uninf_det_v0.pt` |

After inference: `activity_transposed.csv`, `frame_manifest.json`. Set paths in the sidebar.

---

## Run inference (local / VM, slow)

```bash
export PYTHONPATH="mosquito-lab:$PYTHONPATH"
pip install -r mosquito-lab/requirements-ml.txt

python -m mosquito_lab.run_inference \
  --image-folder "path/to/raw_images" \
  --labels "path/to/labels.csv" \
  --model "path/to/uninf_det_v0.pt" \
  --output-dir ./mosquito-lab-output \
  --serve-viewer
```

Or: `./run_cloud.sh local "/path/to/raw_images" "/path/to/labels.csv" "/path/to/model.pt"`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Error running app (Cloud) | Check Manage app → Logs; slim `requirements.txt`; reboot app |
| App won't start locally | `pip install -r requirements.txt`; Python 3.10+ |
| Port in use | `PORT=8503 ./run_app.sh` |
| Inspector cache / YOLO missing | `pip install -r requirements-ml.txt` then run `precompute.py` |
| No images / empty graphs | Check sidebar paths; upload CSV / run inference first |
| Wrong passcode | Ask Flora, or set `CLOUD_VIEWER_PASSCODE` |
