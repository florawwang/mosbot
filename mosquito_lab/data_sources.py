"""Resolve image / model / label inputs from a local folder or Google Drive link."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Literal

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

_DRIVE_FOLDER_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
_DRIVE_FILE_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_OPEN_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def is_google_drive_url(value: str) -> bool:
    return "drive.google.com" in (value or "")


def extract_drive_id(url: str) -> tuple[Literal["folder", "file"], str]:
    """Return ('folder'|'file', id) from common Google Drive share URLs."""
    url = (url or "").strip()
    m = _DRIVE_FOLDER_RE.search(url)
    if m:
        return "folder", m.group(1)
    m = _DRIVE_FILE_RE.search(url)
    if m:
        return "file", m.group(1)
    m = _DRIVE_OPEN_RE.search(url)
    if m:
        kind: Literal["folder", "file"] = "folder" if "folders" in url else "file"
        return kind, m.group(1)
    raise ValueError(f"Could not parse Google Drive URL: {url}")


def download_from_drive(url: str, dest: str | Path, *, label: str = "file") -> Path:
    """Download a Drive file or folder into dest."""
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install gdown: pip install gdown") from exc

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    kind, drive_id = extract_drive_id(url)

    if kind == "folder":
        folder_url = f"https://drive.google.com/drive/folders/{drive_id}"
        print(f"[drive] Downloading folder ({label}) -> {dest_path}")
        gdown.download_folder(
            url=folder_url,
            output=str(dest_path),
            quiet=False,
            use_cookies=False,
        )
        return dest_path

    # Prefer a sensible extension so later asset discovery works.
    ext_by_label = {
        "labels": ".csv",
        "model": ".pt",
        "cache": ".parquet",
    }
    suffix = ext_by_label.get(label, "")
    if dest_path.suffix:
        out_file = dest_path
    else:
        out_file = dest_path / f"{label}{suffix or '.bin'}"
    file_url = f"https://drive.google.com/uc?id={drive_id}"
    print(f"[drive] Downloading file ({label}) -> {out_file}")
    gdown.download(url=file_url, output=str(out_file), quiet=False, fuzzy=True)
    return out_file


def resolve_path_or_drive(
    local_path: str | None,
    drive_url: str | None,
    work_dir: str | Path,
    subdir: str,
    *,
    label: str,
) -> Path:
    """Use local_path if set, else download drive_url into work_dir/subdir."""
    if local_path:
        p = Path(local_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")
        return p

    if drive_url:
        dest = Path(work_dir) / subdir
        result = download_from_drive(drive_url, dest, label=label)
        if result.is_file():
            return result
        # Folder download — if expecting a single file, try to find it.
        if label in {"labels", "model"}:
            candidates = _find_single_asset(result, label)
            if candidates:
                return candidates[0]
        return result

    raise ValueError(f"Provide --{label.replace('_', '-')} or --{label.replace('_', '-')}-drive-url")


def _find_single_asset(folder: Path, kind: str) -> list[Path]:
    if kind == "labels":
        exts = (".csv",)
    elif kind == "model":
        exts = (".pt", ".onnx")
    elif kind == "cache":
        exts = (".parquet",)
    else:
        exts = IMAGE_EXTS

    hits = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.startswith("."):
                continue
            if name.lower().endswith(exts):
                hits.append(Path(root) / name)
    return sorted(hits)


def find_image_folder(root: str | Path, *, min_images: int = 5) -> Path | None:
    """Locate a folder under root that contains the most frame images."""
    root_path = Path(root)
    if not root_path.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip cache / hidden dirs
        dirnames[:] = [d for d in dirnames if d not in {".detection_cache", ".git"} and not d.startswith(".")]
        if ".detection_cache" in Path(dirpath).parts:
            continue
        n = sum(1 for f in filenames if f.lower().endswith(IMAGE_EXTS))
        if n >= min_images:
            candidates.append((n, Path(dirpath)))

    if not candidates:
        # Accept a folder that itself is all images even if small
        if root_path.is_dir():
            n = sum(
                1
                for f in root_path.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
            if n > 0:
                return root_path
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def resolve_local_or_drive(
    value: str,
    work_dir: str | Path,
    subdir: str,
    *,
    label: str,
    expect: Literal["file", "folder", "images"] = "file",
) -> Path:
    """Resolve a local path or Google Drive URL into a local Path.

    Downloads are cached under work_dir/subdir so Streamlit reruns don't
    re-fetch the same assets every time.
    """
    value = (value or "").strip().strip("\"'")
    if not value:
        raise ValueError(f"Missing {label}")

    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    if is_google_drive_url(value):
        dest = work / subdir
        marker = dest / ".download_ok"
        cached = marker.is_file() and marker.read_text(encoding="utf-8").strip() == value
        if cached and dest.exists():
            result = dest
        else:
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            result = download_from_drive(value, dest, label=label)
            marker.write_text(value + "\n", encoding="utf-8")
            # For file downloads, keep using the returned file path.
            if result.is_file():
                pass
            else:
                result = dest
    else:
        result = Path(value).expanduser().resolve()
        if not result.exists():
            raise FileNotFoundError(f"{label} not found: {result}")

    if expect == "images":
        if result.is_file():
            raise ValueError(f"{label}: expected an image folder, got a file: {result}")
        found = find_image_folder(result)
        if found is None:
            raise FileNotFoundError(
                f"{label}: no images found under {result}. "
                "Share a Drive folder that contains the .jpg frames."
            )
        return found

    if expect == "folder":
        if result.is_file():
            return result.parent
        return result

    # expect == "file"
    if result.is_file():
        return result
    hits = _find_single_asset(
        result,
        label if label in {"labels", "model", "cache"} else "labels",
    )
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{label}: expected a file under {result}")


def resolve_image_folder(
    local_folder: str | None,
    drive_url: str | None,
    work_dir: str | Path,
) -> Path:
    if local_folder:
        return resolve_local_or_drive(
            local_folder, work_dir, "raw_images", label="images", expect="images"
        )
    if drive_url:
        return resolve_local_or_drive(
            drive_url, work_dir, "raw_images", label="images", expect="images"
        )
    raise ValueError("Provide --image-folder or --images-drive-url")


def list_image_paths(image_folder: str | Path) -> list[str]:
    """Return image filenames sorted like localinf2.ipynb."""
    folder = Path(image_folder)
    names = [
        f.name
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(names, key=lambda x: float(Path(x).stem))


def prepare_work_dir(work_dir: str | Path, *, fresh: bool = False) -> Path:
    path = Path(work_dir).expanduser().resolve()
    if fresh and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
