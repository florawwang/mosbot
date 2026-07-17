"""Core inference logic (adapted from localinf2.ipynb, standalone)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from mosquito_lab.data_sources import list_image_paths
from mosquito_lab.image_util import Cropper
from mosquito_lab.status import write_status


def _load_yolo():
    """Import ultralytics only when detection is actually needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics/torch are required for detection. "
            "Install with: pip install -r requirements-ml.txt"
        ) from exc
    return YOLO


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def safe_crop(cropper: Cropper, image: np.ndarray, idx: int) -> np.ndarray | None:
    try:
        cropped = cropper.crop(image, idx)
        if cropped is None or len(cropped.shape) < 2:
            return None
        h, w = cropped.shape[:2]
        if h == 0 or w == 0:
            return None
        return cropped
    except Exception as exc:
        print(f"Crop error mosquito {idx}: {exc}")
        return None


def detect_pos_from_crop(model: Any, cropped: np.ndarray) -> tuple[float, float] | None:
    try:
        result = model(cropped, verbose=False)
        if len(result[0].boxes) == 0:
            return None
        pos = result[0].boxes[0].xywh[0].cpu().numpy()
        return (float(pos[0] + pos[2] / 2), float(pos[1] + pos[3] / 2))
    except Exception as exc:
        print(f"Detection error: {exc}")
        return None


def draw_overlay(
    frame: np.ndarray,
    cropper: Cropper,
    mosquito_num: int,
    pos: tuple[float, float] | None,
    dist: float | None,
) -> np.ndarray:
    """Draw well box + detection point on a BGR frame."""
    out = frame.copy()
    pts = cropper.points[mosquito_num]
    x1, y1, x2, y2 = pts
    cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 2)
    if pos is not None:
        cx, cy = int(pos[0] + x1), int(pos[1] + y1)
        cv2.circle(out, (cx, cy), 6, (46, 204, 113), -1)
        if dist is not None and not math.isnan(dist):
            cv2.putText(
                out,
                f"d={dist:.1f}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (46, 204, 113),
                1,
                cv2.LINE_AA,
            )
    return out


def process_mosquito(
    *,
    image_paths: list[str],
    image_folder: str | Path,
    mosquito_num: int,
    mosquito_total: int,
    cropper: Cropper,
    model: Any,
    status_path: str,
    frame_manifest: dict[str, Any],
    save_previews: bool,
    preview_dir: Path | None,
    preview_every: int,
) -> list[float]:
    write_status(
        status_path,
        state="running",
        mosquito_num=mosquito_num,
        mosquito_total=mosquito_total,
        current_photo=0,
        total_photos=len(image_paths),
    )

    activity: list[float] = []
    stats = {
        "failed_loads": 0,
        "failed_crops": 0,
        "failed_detections": 0,
        "successful_detections": 0,
    }
    last_pos = None
    mosquito_key = f"mosquito_{mosquito_num}"

    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"[Mosquito {mosquito_num}] {i}/{len(image_paths)}")
            write_status(
                status_path,
                state="running",
                mosquito_num=mosquito_num,
                mosquito_total=mosquito_total,
                current_photo=i,
                total_photos=len(image_paths),
            )

        full_path = os.path.join(image_folder, path)
        frame = cv2.imread(full_path)
        frame_entry: dict[str, Any] = {
            "filename": path,
            "mosquito": mosquito_num,
            "load_ok": frame is not None,
            "pos": None,
            "distance": None,
            "preview": None,
        }

        if frame is None:
            stats["failed_loads"] += 1
            activity.append(float("nan"))
            frame_manifest.setdefault(path, {})[mosquito_key] = frame_entry
            continue

        cropped = safe_crop(cropper, frame, mosquito_num)
        if cropped is None:
            stats["failed_crops"] += 1
            activity.append(float("nan"))
            frame_entry["crop_ok"] = False
            frame_manifest.setdefault(path, {})[mosquito_key] = frame_entry
            continue

        pos = detect_pos_from_crop(model, cropped)
        if pos is None:
            stats["failed_detections"] += 1
            activity.append(float("nan"))
            frame_entry["crop_ok"] = True
            frame_manifest.setdefault(path, {})[mosquito_key] = frame_entry
            continue

        stats["successful_detections"] += 1
        if last_pos is None:
            dist = 0.0
        else:
            dist = math.sqrt((last_pos[0] - pos[0]) ** 2 + (last_pos[1] - pos[1]) ** 2)
        last_pos = pos
        activity.append(dist)

        frame_entry.update(
            {
                "crop_ok": True,
                "pos": {"x": pos[0], "y": pos[1]},
                "distance": dist,
            }
        )

        if save_previews and preview_dir and (i % preview_every == 0):
            rel = f"mosquito_{mosquito_num}/{Path(path).stem}.jpg"
            out_path = preview_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            overlay = draw_overlay(frame, cropper, mosquito_num, pos, dist)
            cv2.imwrite(str(out_path), overlay)
            crop_path = preview_dir / f"mosquito_{mosquito_num}/crops/{Path(path).stem}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), cropped)
            frame_entry["preview"] = rel
            frame_entry["crop_preview"] = str(crop_path.relative_to(preview_dir))

        frame_manifest.setdefault(path, {})[mosquito_key] = frame_entry

    print(f"\nMosquito {mosquito_num} summary: {stats}")
    return activity


def run_inference(
    *,
    image_folder: str | Path,
    label_file: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    output_name: str = "activity_transposed.csv",
    device: str = "auto",
    status_path: str | None = None,
    save_previews: bool = True,
    preview_every: int = 50,
) -> dict[str, Any]:
    image_folder = Path(image_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status_file = status_path or str(output_dir / "inference_status.json")
    preview_dir = output_dir / "previews" if save_previews else None
    manifest_path = output_dir / "frame_manifest.json"

    print("Loading cropper...")
    cropper = Cropper(str(label_file))

    print("Loading YOLO model...")
    resolved_device = pick_device(device)
    print(f"Using device: {resolved_device}")
    YOLO = _load_yolo()
    model = YOLO(str(model_path))
    model.to(resolved_device)
    print("YOLO loaded.\n")

    image_paths = list_image_paths(image_folder)
    print(f"Images found: {len(image_paths)}")
    mosquito_total = len(cropper.boxes)

    write_status(
        status_file,
        state="running",
        mosquito_num=0,
        mosquito_total=mosquito_total,
        current_photo=0,
        total_photos=len(image_paths),
    )

    all_data: dict[str, list[float]] = {}
    frame_manifest: dict[str, Any] = {}
    global_stats = {
        "failed_loads": 0,
        "failed_crops": 0,
        "failed_detections": 0,
        "successful_detections": 0,
    }

    for mosquito_num, _ in enumerate(cropper.boxes):
        print("\n===================================")
        print(f"PROCESSING MOSQUITO {mosquito_num}")
        print("===================================\n")

        activity = process_mosquito(
            image_paths=image_paths,
            image_folder=image_folder,
            mosquito_num=mosquito_num,
            mosquito_total=mosquito_total,
            cropper=cropper,
            model=model,
            status_path=status_file,
            frame_manifest=frame_manifest,
            save_previews=save_previews,
            preview_dir=preview_dir,
            preview_every=preview_every,
        )
        all_data[f"mosquito_{mosquito_num}"] = activity

    print("\nCreating dataframe...")
    df = pd.DataFrame(all_data)
    df.insert(0, "frame", range(len(df)))

    output_file = output_dir / output_name
    print(f"\nSaving CSV -> {output_file}")
    df.to_csv(output_file, index=False)

    manifest_payload = {
        "image_folder": str(image_folder),
        "label_file": str(label_file),
        "model_path": str(model_path),
        "output_csv": str(output_file),
        "image_paths": image_paths,
        "frames": frame_manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    write_status(
        status_file,
        state="done",
        mosquito_num=mosquito_total - 1 if mosquito_total else None,
        mosquito_total=mosquito_total,
        current_photo=len(image_paths),
        total_photos=len(image_paths),
        output_file=str(output_file),
    )

    print("\n===================================")
    print("DONE")
    print("===================================")
    print(f"CSV: {output_file}")
    print(f"Manifest: {manifest_path}")

    return {
        "output_csv": str(output_file),
        "manifest": str(manifest_path),
        "status_file": status_file,
        "image_folder": str(image_folder),
        "mosquito_total": mosquito_total,
        "image_count": len(image_paths),
    }
