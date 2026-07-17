"""Core logic for the Detection Quality Inspector.

This module is UI-agnostic: it knows how to describe an experiment, load the
per-mosquito "well" boxes, run YOLO over well crops, cache every candidate
detection (down to a low confidence floor), and read that cache back so the GUI
can re-threshold instantly.

The detection pipeline mirrors localinf2.ipynb exactly:
  full frame  ->  Cropper.crop(frame, well_idx)  ->  YOLO on the crop.
The only difference is that we keep *every* candidate box with its confidence
(not just the top box above the default threshold), so the inspector can answer
"what would YOLO have found at threshold X?" without re-running the model.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# Confidence floor used when building a cache. Every box at or above this is
# stored, so the GUI slider can explore any threshold >= this value with no
# re-inference. Keep it low so weak/borderline detections are visible.
DEFAULT_CONF_FLOOR = 0.01

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- #
# Wells (per-mosquito crop boxes, from a MakeSense labels.csv)
# --------------------------------------------------------------------------- #
@dataclass
class Well:
    idx: int
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def degenerate(self) -> bool:
        # Some labels files contain junk 0/1-pixel boxes; a well that tiny can
        # never yield a real detection, so we surface it instead of hiding it.
        return self.w < 5 or self.h < 5


def load_wells(label_file: str) -> list[Well]:
    """Load per-mosquito crop boxes from a MakeSense-style labels CSV.

    Matches utils.image_util.Cropper: columns are
    label_name, bbox_x, bbox_y, bbox_width, bbox_height, image_name, ...
    Row order defines the mosquito index (mosquito_0, mosquito_1, ...).
    """
    with open(label_file, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        wells = []
        for i, row in enumerate(reader):
            wells.append(
                Well(idx=i, x=int(row[1]), y=int(row[2]), w=int(row[3]), h=int(row[4]))
            )
    return wells


def list_frames(image_folder: str) -> list[str]:
    """Return image filenames sorted the same way localinf2 sorts them."""
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(IMAGE_EXTS)]

    def sort_key(name: str):
        stem = os.path.splitext(name)[0]
        try:
            return (0, float(stem))
        except ValueError:
            return (1, name)

    return sorted(files, key=sort_key)


# --------------------------------------------------------------------------- #
# Experiment description
# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    name: str
    image_folder: str
    label_file: str
    model_path: str
    cache_dir: str = ""

    def __post_init__(self) -> None:
        if not self.cache_dir:
            self.cache_dir = os.path.join(self.image_folder, ".detection_cache")

    @property
    def cache_file(self) -> str:
        return os.path.join(self.cache_dir, "detections.parquet")

    @property
    def meta_file(self) -> str:
        return os.path.join(self.cache_dir, "cache_meta.json")

    @property
    def flags_file(self) -> str:
        return os.path.join(self.cache_dir, "flags.json")

    def exists_check(self) -> dict[str, bool]:
        return {
            "image_folder": os.path.isdir(self.image_folder),
            "label_file": os.path.isfile(self.label_file),
            "model_path": os.path.isfile(self.model_path),
            "cache": os.path.isfile(self.cache_file),
        }


# --------------------------------------------------------------------------- #
# YOLO detection over well crops
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, Any] = {}


def get_model(model_path: str):
    """Load (and memoize) a YOLO model on CPU, matching the notebook."""
    if model_path not in _MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Building a detection cache needs ultralytics/torch. "
                "Install with: pip install -r requirements-ml.txt"
            ) from exc

        model = YOLO(model_path)
        model.to("cpu")
        _MODEL_CACHE[model_path] = model
    return _MODEL_CACHE[model_path]


def detect_frame(
    frame: np.ndarray,
    wells: list[Well],
    model,
    conf_floor: float = DEFAULT_CONF_FLOOR,
) -> list[dict]:
    """Run YOLO on every well crop of one frame.

    Returns one dict per candidate box, with coordinates already mapped back to
    full-frame pixel space so the GUI can draw them directly.
    """
    rows: list[dict] = []
    for well in wells:
        if well.degenerate:
            continue
        crop = frame[well.y : well.y2, well.x : well.x2]
        if crop.size == 0:
            continue
        result = model(crop, conf=conf_floor, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (bx1, by1, bx2, by2), conf in zip(xyxy, confs):
            rows.append(
                {
                    "well": well.idx,
                    "conf": float(conf),
                    "x1": float(bx1) + well.x,
                    "y1": float(by1) + well.y,
                    "x2": float(bx2) + well.x,
                    "y2": float(by2) + well.y,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Cache building / reading
# --------------------------------------------------------------------------- #
def build_cache(
    exp: Experiment,
    conf_floor: float = DEFAULT_CONF_FLOOR,
    stride: int = 1,
    max_frames: int | None = None,
    progress=None,
) -> pd.DataFrame:
    """Run YOLO over the experiment and write the detection cache.

    Args:
        exp: experiment to process.
        conf_floor: keep every candidate box at or above this confidence.
        stride: process every Nth frame (1 = all frames).
        max_frames: optional cap on number of processed frames.
        progress: optional callback(done, total, frame_name) for UI updates.
    """
    import cv2

    wells = load_wells(exp.label_file)
    all_frames = list_frames(exp.image_folder)
    selected = all_frames[::stride]
    if max_frames is not None:
        selected = selected[:max_frames]

    model = get_model(exp.model_path)

    records: list[dict] = []
    frame_index_map: dict[str, int] = {name: i for i, name in enumerate(all_frames)}

    total = len(selected)
    for done, name in enumerate(selected):
        path = os.path.join(exp.image_folder, name)
        frame = cv2.imread(path)
        if frame is None:
            if progress:
                progress(done + 1, total, f"{name} (load failed)")
            continue
        dets = detect_frame(frame, wells, model, conf_floor)
        for d in dets:
            d["frame_idx"] = frame_index_map[name]
            d["image_name"] = name
            records.append(d)
        if progress:
            progress(done + 1, total, name)

    df = pd.DataFrame(
        records,
        columns=["frame_idx", "image_name", "well", "conf", "x1", "y1", "x2", "y2"],
    )

    os.makedirs(exp.cache_dir, exist_ok=True)
    df.to_parquet(exp.cache_file, index=False)

    meta = {
        "experiment": exp.name,
        "image_folder": exp.image_folder,
        "label_file": exp.label_file,
        "model_path": exp.model_path,
        "conf_floor": conf_floor,
        "stride": stride,
        "n_wells": len(wells),
        "n_frames_total": len(all_frames),
        "n_frames_processed": total,
        "processed_frame_indices": [frame_index_map[n] for n in selected],
        "frame_width": 1200,
        "frame_height": 800,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(exp.meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return df


def load_cache(exp: Experiment) -> tuple[pd.DataFrame, dict]:
    """Read the detection cache and its metadata."""
    df = pd.read_parquet(exp.cache_file)
    meta: dict = {}
    if os.path.isfile(exp.meta_file):
        with open(exp.meta_file) as f:
            meta = json.load(f)
    return df, meta


# --------------------------------------------------------------------------- #
# Thresholding / detection-quality queries
# --------------------------------------------------------------------------- #
def best_box_per_well(frame_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """For one frame's rows, keep the single highest-confidence box per well
    that clears `threshold`."""
    kept = frame_df[frame_df["conf"] >= threshold]
    if kept.empty:
        return kept
    idx = kept.groupby("well")["conf"].idxmax()
    return kept.loc[idx].sort_values("well")


def detected_wells(frame_df: pd.DataFrame, threshold: float) -> set[int]:
    if frame_df.empty:
        return set()
    kept = frame_df[frame_df["conf"] >= threshold]
    return set(int(w) for w in kept["well"].unique())


def missed_wells(frame_df: pd.DataFrame, threshold: float, all_wells: list[int]) -> list[int]:
    """Wells with no detection at or above threshold for this frame."""
    det = detected_wells(frame_df, threshold)
    return [w for w in all_wells if w not in det]


def per_frame_detection_counts(
    df: pd.DataFrame, threshold: float, processed_frames: list[int]
) -> pd.DataFrame:
    """Number of wells detected per processed frame at a given threshold."""
    kept = df[df["conf"] >= threshold]
    counts = (
        kept.groupby("frame_idx")["well"].nunique()
        if not kept.empty
        else pd.Series(dtype=int)
    )
    out = pd.DataFrame({"frame_idx": processed_frames})
    out["detected"] = out["frame_idx"].map(counts).fillna(0).astype(int)
    return out


def per_well_missed_rate(
    df: pd.DataFrame, threshold: float, processed_frames: list[int], all_wells: list[int]
) -> pd.DataFrame:
    """For each well, the fraction of processed frames with no detection above
    threshold. This is the fastest way to spot a chronically-missed mosquito."""
    n = len(processed_frames) or 1
    kept = df[df["conf"] >= threshold]
    hit_counts = (
        kept.groupby("well")["frame_idx"].nunique()
        if not kept.empty
        else pd.Series(dtype=int)
    )
    rows = []
    for w in all_wells:
        hits = int(hit_counts.get(w, 0))
        rows.append(
            {
                "well": w,
                "frames_detected": hits,
                "frames_missed": n - hits,
                "missed_rate": round((n - hits) / n, 4),
            }
        )
    return pd.DataFrame(rows)


def next_missed_frame(
    df: pd.DataFrame,
    threshold: float,
    processed_frames: list[int],
    all_wells: list[int],
    current_frame: int,
    direction: int = 1,
    well: int | None = None,
) -> int | None:
    """Find the next/previous processed frame (relative to current_frame) that
    has a missed detection.

    If `well` is given, only that well counts as "missed"; otherwise any missed
    well qualifies.
    """
    try:
        pos = processed_frames.index(current_frame)
    except ValueError:
        # current frame not in processed set; find nearest insert position
        pos = min(
            range(len(processed_frames)),
            key=lambda i: abs(processed_frames[i] - current_frame),
            default=-1,
        )
    order = (
        range(pos + 1, len(processed_frames))
        if direction > 0
        else range(pos - 1, -1, -1)
    )
    for i in order:
        fidx = processed_frames[i]
        fdf = df[df["frame_idx"] == fidx]
        det = detected_wells(fdf, threshold)
        if well is not None:
            if well not in det:
                return fidx
        else:
            if any(w not in det for w in all_wells):
                return fidx
    return None


# --------------------------------------------------------------------------- #
# Frame flags (bad-frame bookkeeping)
# --------------------------------------------------------------------------- #
def load_flags(exp: Experiment) -> dict[str, dict]:
    if not os.path.isfile(exp.flags_file):
        return {}
    try:
        with open(exp.flags_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_flags(exp: Experiment, flags: dict[str, dict]) -> None:
    os.makedirs(exp.cache_dir, exist_ok=True)
    tmp = exp.flags_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(flags, f, indent=2)
    os.replace(tmp, exp.flags_file)


def set_flag(
    exp: Experiment, frame_idx: int, flagged: bool, reason: str = ""
) -> dict[str, dict]:
    flags = load_flags(exp)
    key = str(frame_idx)
    if flagged:
        flags[key] = {
            "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        flags.pop(key, None)
    save_flags(exp, flags)
    return flags


def export_flags_csv(exp: Experiment, frames: list[str]) -> str:
    """Write flagged frames to a CSV next to the cache and return its path."""
    flags = load_flags(exp)
    path = os.path.join(exp.cache_dir, "flagged_frames.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "image_name", "reason", "updated_at"])
        for key in sorted(flags, key=lambda k: int(k)):
            fidx = int(key)
            name = frames[fidx] if 0 <= fidx < len(frames) else ""
            writer.writerow(
                [fidx, name, flags[key].get("reason", ""), flags[key].get("updated_at", "")]
            )
    return path


from mosquito_lab.paths import mosquito_project_dir, repo_root

# --------------------------------------------------------------------------- #
# Experiment autodiscovery
# --------------------------------------------------------------------------- #
BASE_DIR = str(repo_root())


def discover_experiments() -> list[Experiment]:
    """Best-effort scan of the workspace for (images + labels + model) triples.

    Pairs a raw-images folder under data/ with a labels CSV under
    MosquitoMovement2/experiments/ by matching the leading experiment number.
    """
    models_dir = str(mosquito_project_dir() / "models")
    default_model = os.path.join(models_dir, "uninf_det_v0.pt")
    exp_root = str(mosquito_project_dir() / "experiments")
    data_root = os.path.join(BASE_DIR, "data")

    # Map experiment number -> labels csv
    label_by_num: dict[str, str] = {}
    if os.path.isdir(exp_root):
        for entry in os.listdir(exp_root):
            sub = os.path.join(exp_root, entry)
            if not os.path.isdir(sub):
                continue
            num = entry.split(" ")[0].strip()
            for f in os.listdir(sub):
                if f.lower().endswith(".csv") and "label" in f.lower():
                    label_by_num[num] = os.path.join(sub, f)
                    break

    found: list[Experiment] = []
    if os.path.isdir(data_root):
        for entry in sorted(os.listdir(data_root)):
            sub = os.path.join(data_root, entry)
            if not os.path.isdir(sub):
                continue
            num = entry.split(" ")[0].strip()
            img_folder = _find_raw_images(sub)
            if not img_folder:
                continue
            label = label_by_num.get(num)
            if not label:
                continue
            found.append(
                Experiment(
                    name=entry,
                    image_folder=img_folder,
                    label_file=label,
                    model_path=default_model,
                )
            )
    return found


def _find_raw_images(root: str) -> str | None:
    """Locate a folder of raw frames under an experiment directory."""
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ".detection_cache" in dirpath:
            continue
        img_count = sum(1 for f in filenames if f.lower().endswith(IMAGE_EXTS))
        if img_count >= 5:
            candidates.append((img_count, dirpath))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]
