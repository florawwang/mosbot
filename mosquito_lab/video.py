"""Build or locate frame-sequence videos for the Mosquito Lab web app."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2

from mosquito_lab.image_util import Cropper
from mosquito_lab.inference_core import draw_overlay

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def frame_sort_key(name: str) -> tuple[int, int | str]:
    stem = Path(name).stem
    if stem.isdigit():
        return (0, int(stem))
    try:
        return (0, int(float(stem)))
    except ValueError:
        return (1, name)


def list_frame_names(image_folder: str | Path) -> list[str]:
    folder = Path(image_folder)
    names = [
        f.name
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(names, key=frame_sort_key)


def find_existing_videos(image_folder: str | Path, output_dir: str | Path) -> list[Path]:
    """Search image folder, output dir, and parents for video files."""
    image_folder = Path(image_folder)
    output_dir = Path(output_dir)
    seen: set[Path] = set()
    candidates: list[Path] = []

    search_roots = [image_folder, output_dir]
    for root in (image_folder, output_dir):
        for parent in [root, *root.parents[:3]]:
            if parent.is_dir() and parent not in search_roots:
                search_roots.append(parent)

    for folder in search_roots:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def _video_cache_dir(output_dir: Path) -> Path:
    path = output_dir / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_cache_key(
    image_folder: str,
    frame_names: list[str],
    fps: float,
    view_mode: str,
    mosquito_num: int,
    stride: int,
    max_frames: int,
    label_frames: bool,
) -> str:
    payload = {
        "image_folder": str(Path(image_folder).resolve()),
        "n_frames": len(frame_names),
        "first": frame_names[0] if frame_names else "",
        "last": frame_names[-1] if frame_names else "",
        "fps": fps,
        "view_mode": view_mode,
        "mosquito_num": mosquito_num,
        "stride": stride,
        "max_frames": max_frames,
        "label_frames": label_frames,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return digest


def cached_video_path(
    *,
    image_folder: str,
    output_dir: str | Path,
    frame_names: list[str],
    fps: float,
    view_mode: str,
    mosquito_num: int,
    stride: int,
    max_frames: int,
    label_frames: bool,
) -> Path | None:
    output_dir = Path(output_dir)
    cache_dir = _video_cache_dir(output_dir)
    key = _build_cache_key(
        image_folder,
        frame_names,
        fps,
        view_mode,
        mosquito_num,
        stride,
        max_frames,
        label_frames,
    )
    out_path = cache_dir / f"frames_{view_mode}_m{mosquito_num}_{key}.mp4"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    return None


def _load_manifest_positions(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.is_file():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", {})


def _annotate_frame(frame, label: str) -> None:
    h, w = frame.shape[:2]
    margin = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.9, w / 1200))
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
    org = (max(margin, w - text_w - margin), min(h - margin, text_h + margin))
    cv2.putText(frame, label, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def build_video_from_frames(
    *,
    image_folder: str | Path,
    output_path: Path,
    frame_names: list[str],
    fps: float = 10.0,
    view_mode: str = "full",
    mosquito_num: int = 0,
    label_file: str | Path | None = None,
    manifest_path: str | Path | None = None,
    stride: int = 1,
    max_frames: int = 0,
    label_frames: bool = True,
    progress_callback=None,
) -> Path:
    """Stitch sorted frame images into an MP4.

    view_mode: 'full' | 'crop' | 'annotated'
    max_frames: 0 means all frames (after stride).
    """
    image_folder = Path(image_folder)
    if stride < 1:
        stride = 1

    selected = frame_names[::stride]
    if max_frames > 0:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("No frames selected for video.")

    cropper = None
    manifest_frames: dict = {}
    if view_mode in {"crop", "annotated"}:
        if not label_file or not Path(label_file).is_file():
            raise ValueError("Labels CSV required for crop / annotated video.")
        cropper = Cropper(str(label_file))
    if view_mode == "annotated" and manifest_path:
        manifest_frames = _load_manifest_positions(Path(manifest_path))

    first_path = image_folder / selected[0]
    first = cv2.imread(str(first_path))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {first_path}")

    if view_mode == "crop" and cropper is not None:
        cropped = cropper.crop(first, mosquito_num)
        if cropped is None or cropped.size == 0:
            raise RuntimeError(f"Could not crop mosquito {mosquito_num} from first frame.")
        first = cropped
    elif view_mode == "annotated" and cropper is not None:
        meta = manifest_frames.get(selected[0], {}).get(f"mosquito_{mosquito_num}", {})
        pos = None
        if meta.get("pos"):
            pos = (meta["pos"]["x"], meta["pos"]["y"])
        dist = meta.get("distance")
        first = draw_overlay(first, cropper, mosquito_num, pos, dist)

    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    total = len(selected)
    try:
        for idx, name in enumerate(selected):
            if progress_callback:
                progress_callback(idx, total, name)

            frame = cv2.imread(str(image_folder / name))
            if frame is None:
                continue

            if view_mode == "crop" and cropper is not None:
                cropped = cropper.crop(frame, mosquito_num)
                if cropped is None or cropped.size == 0:
                    continue
                frame = cropped
            elif view_mode == "annotated" and cropper is not None:
                meta = manifest_frames.get(name, {}).get(f"mosquito_{mosquito_num}", {})
                pos = None
                if meta.get("pos"):
                    pos = (meta["pos"]["x"], meta["pos"]["y"])
                dist = meta.get("distance")
                frame = draw_overlay(frame, cropper, mosquito_num, pos, dist)

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            if label_frames:
                _annotate_frame(frame, name)

            writer.write(frame)
    finally:
        writer.release()

    _transcode_h264_if_available(output_path)
    return output_path


def _transcode_h264_if_available(video_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    temp = video_path.with_name(f"{video_path.stem}.h264tmp.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        temp.replace(video_path)
    except (subprocess.CalledProcessError, OSError):
        if temp.exists():
            temp.unlink(missing_ok=True)


def get_or_build_cached_video(
    *,
    image_folder: str,
    output_dir: str | Path,
    frame_names: list[str],
    fps: float,
    view_mode: str,
    mosquito_num: int,
    label_file: str,
    manifest_path: str | Path | None,
    stride: int,
    max_frames: int,
    label_frames: bool,
    progress_callback=None,
) -> Path:
    output_dir = Path(output_dir)
    cache_dir = _video_cache_dir(output_dir)
    key = _build_cache_key(
        image_folder,
        frame_names,
        fps,
        view_mode,
        mosquito_num,
        stride,
        max_frames,
        label_frames,
    )
    out_path = cache_dir / f"frames_{view_mode}_m{mosquito_num}_{key}.mp4"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    return build_video_from_frames(
        image_folder=image_folder,
        output_path=out_path,
        frame_names=frame_names,
        fps=fps,
        view_mode=view_mode,
        mosquito_num=mosquito_num,
        label_file=label_file,
        manifest_path=manifest_path,
        stride=stride,
        max_frames=max_frames,
        label_frames=label_frames,
        progress_callback=progress_callback,
    )
