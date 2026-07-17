# Detection Quality Inspector

**A point-and-click tool for checking where YOLO misses mosquitoes.**

If you've ever thought *"the model isn't detecting my mosquitoes"* but couldn't
tell **which** ones, **when**, or **why**, this is for you. It shows you each
frame with the detections drawn on top, lets you slide the confidence threshold
up and down, and points you straight to the frames where a mosquito was missed.

---

## Quick start (3 steps)

Run these from the lab project folder (the one that contains `.venv/`).

**1. Install the two extra packages (one time only):**

```bash
./.venv/bin/python -m pip install -r mosquito-lab/requirements.txt
```

**2. Prepare an experiment for viewing.** This runs the model once and saves the
results so the app is fast. Start with `--stride 10` to check every 10th frame
(quick); use `--stride 1` later for every frame (slow).

```bash
PYTHONPATH=mosquito-lab ./.venv/bin/python mosquito_lab/inspector/precompute.py --experiment "28" --stride 10
```

> Don't know the experiment number? List what's available:
> `PYTHONPATH=mosquito-lab ./.venv/bin/python mosquito_lab/inspector/precompute.py --list`

**3. Open it inside mosbot (preferred):**

```bash
cd mosquito-lab && ./run_app.sh
```

Then choose **Detection inspector** in the sidebar.

Or run the inspector alone:

```bash
PYTHONPATH=mosquito-lab ./.venv/bin/python -m streamlit run mosquito_lab/inspector/app.py
```

Pick your experiment from the sidebar. (If you skipped step 2, the app can build
a cache for you there.)

---

## What the colors mean

Each mosquito lives in a fixed box called a **well**. The app draws a box around
every well and every detection:

| Color | Meaning |
|-------|---------|
| 🟩 **Green well** | Mosquito detected at the current threshold — good. |
| 🟥 **Red well** | Missed at the current threshold — no detection here. |
| 🟨 **Yellow box** | The model *saw something* but wasn't confident enough. Lowering the threshold would recover it. |
| 🟦 **Blue well** | The mosquito you chose to focus on. |
| ⬜ **Thin grey well** | A broken label (a 0–1 pixel box). Can never be detected — fix the labels file. |

**Rule of thumb:** a red well with a yellow box inside means *lower your
threshold*. A red well with **nothing** inside means the model genuinely found
nothing there — a real miss worth investigating.

---

## How to actually use it

- **Slide the confidence threshold** (sidebar) to see detections appear and
  disappear instantly. No waiting — the model already ran.
- **Jump to misses.** Use *Next missed* / *Prev missed* to skip straight to
  frames where something was missed, instead of scrolling one by one.
- **Focus one mosquito.** Turn on *Focus a single mosquito* to zoom in on one
  well and see its detection confidences.
- **Flag bad frames.** Found glare, motion blur, or a mosquito outside its well?
  Flag the frame with a note, then export the whole list to CSV.

### The tabs at the bottom

- **Boxes (this frame)** — every candidate detection with its confidence.
- **Per-well miss rate** — *the most useful view.* Ranks mosquitoes by how often
  they're missed. A well near the top is one the model chronically struggles
  with.
- **Detections over time** — how many mosquitoes were found per frame. Dips
  reveal bad stretches of video.
- **Flagged frames** — review everything you flagged, export to CSV, jump back.

---

## Why "prepare an experiment" first?

Running the model live on every frame is slow on a laptop CPU (~0.4 frames/sec).
So the tool runs it **once** in the background and saves every detection — even
the weak, low-confidence ones — to a small file. After that, moving the
threshold slider just re-filters what's already saved, which is instant.

Everything is saved in a `.detection_cache/` folder next to your images, so you
can re-open the app anytime without re-running the model. This matches the exact
detection steps used in `localinf2.ipynb`, so what you see here is what the
activity data was built from.

---

## Troubleshooting

**"No experiments discovered."**
Your images/labels aren't where the tool expects. Use **Custom path...** in the
sidebar (or the `--images`/`--labels`/`--model` flags) to point at them directly.

**Preparing an experiment is taking forever.**
Use a bigger stride (e.g. `--stride 20`) to sample fewer frames, or cap it with
`--max-frames 200`. Do a quick pass first, then go deeper where misses cluster.

**A well is grey and thin.**
Its label box is broken (essentially 0 pixels). Fix that mosquito's row in the
labels CSV — the model can't detect a box that has no area.

**I changed the stride and my old frames disappeared.**
Preparing an experiment overwrites the saved results. Frames outside the new
sample won't show until you prepare again with more coverage.

---

## Command reference

| Task | Command |
|------|---------|
| List experiments | `precompute.py --list` |
| Prepare, quick sample | `precompute.py --experiment "28" --stride 10` |
| Prepare, every frame | `precompute.py --experiment "28" --stride 1` |
| Cap frames | add `--max-frames 200` |
| Custom paths | `--images <dir> --labels <csv> --model <.pt>` |
| Launch app | `-m streamlit run detection_inspector/app.py` |

(Prefix each with `PYTHONPATH=mosquito-lab ./.venv/bin/python mosquito_lab/inspector/`.)

---

## What's in this folder

- `app.py` — the point-and-click app you open in the browser.
- `precompute.py` — the command that prepares an experiment (runs the model).
- `inspector_core.py` — the shared logic (loading wells, running YOLO, saving
  and reading results). You don't run this directly.
- `requirements.txt` — the two extra packages the tool needs.
