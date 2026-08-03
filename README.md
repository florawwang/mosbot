# mosbot

**mosbot** is a web app for mosquito activity experiments: plot LD/DD actograms and combine experiments from activity CSVs.

Built for circadian pipeline.

**Live app:** [https://mosbot.streamlit.app](https://mosbot.streamlit.app)

---

## Features

| Section | What it does |
|---------|----------------|
| **Activity graphs** | Actograms, LD/DD profiles, day/night totals, and stats from an activity CSV |
| **Combine experiments** | Merge several experiments onto one ZT timeline — kinds line up, each keeps its true ZT start + LD/DD schedule, gaps become NaN — then plot combined Figs 3–9 + stats |

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

### Passcode (keep this out of git)

The unlock code is **not** stored in the public repo. Configure it one of these ways:

1. **Local file** (recommended for laptop):
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # edit secrets.toml and set CLOUD_VIEWER_PASSCODE
   ```
2. **Environment variable:**
   ```bash
   CLOUD_VIEWER_PASSCODE=yourcode ./run_app.sh
   ```
3. **Streamlit Cloud:** App → **Settings → Secrets** → paste:
   ```toml
   CLOUD_VIEWER_PASSCODE = "yourcode"
   ```

Ask a lab member for the current passcode.

Stop the app with `Ctrl+C`. If port 8502 is busy: `PORT=8503 ./run_app.sh`

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

Set paths in the app sidebar (or use env vars / auto-discovery when nested in the lab data repo).

---

## Typical workflow

1. **Run inference** (laptop or VM — can take hours on CPU) → activity CSV  
2. **Open mosbot** → plot activity graphs and combine experiments  

Skip step 1 if someone already shared an activity CSV with you.

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

Inference is meant for a laptop/VM. On Cloud, upload saved activity CSVs.

If the app fails to start, open **Manage app → Logs** on [share.streamlit.io](https://share.streamlit.io).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| App won't start | `pip install -r requirements.txt`; use Python 3.10–3.12 |
| Port already in use | `PORT=8503 ./run_app.sh` |
| Empty graphs | Load an activity CSV in the sidebar, or run inference first |
| Wrong passcode | Ask a lab member, or set `CLOUD_VIEWER_PASSCODE` |

