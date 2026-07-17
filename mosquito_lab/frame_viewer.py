"""Per-frame image viewer module (extracted for lab_app merge)."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from mosquito_lab.image_util import Cropper
from mosquito_lab.inference_core import detect_pos_from_crop, draw_overlay, pick_device
from mosquito_lab.paths import mosquito_project_dir, repo_root

LAB_ROOT = repo_root()


def resolve_paths() -> dict:
    output_dir = Path(
        os.environ.get("CLOUD_VIEWER_OUTPUT_DIR", "./mosquito_lab_output")
    ).expanduser()
    manifest_path = output_dir / "frame_manifest.json"
    status_path = output_dir / "inference_status.json"
    csv_candidates = list(output_dir.glob("*.csv"))
    csv_path = csv_candidates[0] if csv_candidates else None

    image_folder = os.environ.get("CLOUD_VIEWER_IMAGE_FOLDER", "")
    label_file = os.environ.get("CLOUD_VIEWER_LABEL_FILE", "")
    model_path = os.environ.get(
        "CLOUD_VIEWER_MODEL_PATH",
        str(mosquito_project_dir() / "models" / "uninf_det_v0.pt"),
    )

    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        image_folder = image_folder or manifest.get("image_folder", "")
        label_file = label_file or manifest.get("label_file", "")
        model_path = model_path or manifest.get("model_path", model_path)
        if not csv_path:
            out = manifest.get("output_csv")
            if out and Path(out).is_file():
                csv_path = Path(out)

    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "status_path": status_path,
        "csv_path": csv_path,
        "image_folder": image_folder,
        "label_file": label_file,
        "model_path": model_path,
    }


@st.cache_resource
def load_model(model_path: str, device: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Live YOLO needs ultralytics/torch. "
            "On your laptop: pip install -r requirements-ml.txt. "
            "On Streamlit Cloud, use saved frame_manifest.json instead of live detection."
        ) from exc

    model = YOLO(model_path)
    model.to(device)
    return model


@st.cache_data(show_spinner=False)
def load_manifest(manifest_path: str, mtime: float) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_activity_csv(csv_path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_frame_bgr(image_folder: str, filename: str) -> np.ndarray | None:
    return cv2.imread(os.path.join(image_folder, filename))


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def render_frame_sidebar(paths: dict) -> dict:
    """Sidebar controls for the frame viewer (inputs / outputs / options)."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Inputs")
    st.sidebar.caption("Raw experiment data used for viewing frames.")
    image_folder = st.sidebar.text_input(
        "Image folder",
        value=paths["image_folder"],
        help="Folder of raw frame images (`.jpg`, `.png`, …).",
        key="fv_image_folder",
    )
    label_file = st.sidebar.text_input(
        "Labels CSV",
        value=paths["label_file"],
        help="MakeSense labels CSV with per-mosquito well boxes.",
        key="fv_label_file",
    )
    model_path = st.sidebar.text_input(
        "YOLO model (.pt)",
        value=paths["model_path"],
        help="Only needed if live re-detection is enabled.",
        key="fv_model_path",
    )

    st.sidebar.markdown("#### Outputs")
    st.sidebar.caption("Where inference results are saved.")
    output_dir = st.sidebar.text_input(
        "Output directory",
        value=str(paths["output_dir"]),
        help="Contains activity CSV, manifest, previews, and status.",
        key="fv_output_dir",
    )
    csv_hint = paths.get("csv_path")
    if csv_hint and Path(csv_hint).is_file():
        st.sidebar.text_input(
            "Activity CSV (detected)",
            value=str(csv_hint),
            disabled=True,
            help="Auto-detected from output directory. Used by Activity graphs.",
            key="fv_csv_detected",
        )

    status_path = Path(output_dir) / "inference_status.json"
    if status_path.is_file():
        import json

        with open(status_path, encoding="utf-8") as f:
            status = json.load(f)
        with st.sidebar.expander("Inference run status", expanded=False):
            st.json(status)

    st.sidebar.markdown("#### Options")
    live_detect = st.sidebar.checkbox(
        "Run live YOLO on current frame",
        value=False,
        help="Re-run detection for overlay (slower, needs model).",
        key="fv_live_detect",
    )
    device = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda", "mps"],
        index=0,
        key="fv_device",
    )
    return {
        "image_folder": image_folder,
        "label_file": label_file,
        "model_path": model_path,
        "output_dir": Path(output_dir),
        "live_detect": live_detect,
        "device": device,
    }


def sidebar_config(paths: dict) -> dict:
    """Backward-compatible alias."""
    return render_frame_sidebar(paths)


def render_frame_viewer_body(cfg: dict, paths: dict) -> str:
    """Main panel for frame browser. Returns activity CSV path for graphs tab."""
    if not cfg["image_folder"] or not Path(cfg["image_folder"]).is_dir():
        st.warning("Set a valid image folder in the sidebar.")
        return default_activity_csv(paths, cfg)
    if not cfg["label_file"] or not Path(cfg["label_file"]).is_file():
        st.warning("Set a valid labels CSV in the sidebar.")
        return default_activity_csv(paths, cfg)

    manifest = None
    manifest_path = cfg["output_dir"] / "frame_manifest.json"
    if manifest_path.is_file():
        manifest = load_manifest(str(manifest_path), manifest_path.stat().st_mtime)

    activity_df = None
    csv_path = paths["csv_path"]
    if csv_path and Path(csv_path).is_file():
        activity_df = load_activity_csv(str(csv_path), Path(csv_path).stat().st_mtime)
    elif cfg["output_dir"].is_dir():
        candidates = sorted(cfg["output_dir"].glob("*.csv"))
        if candidates:
            csv_path = candidates[0]
            activity_df = load_activity_csv(str(csv_path), csv_path.stat().st_mtime)

    cropper = Cropper(cfg["label_file"])
    frames = get_frame_list(cfg["image_folder"], manifest)
    if not frames:
        st.error("No images found.")
        return default_activity_csv(paths, cfg)

    mosquito_total = len(cropper.boxes)
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        frame_idx = st.slider("Frame index", 0, len(frames) - 1, 0, key="fv_frame_idx")
    with col_b:
        mosquito = st.selectbox(
            "Mosquito",
            list(range(mosquito_total)),
            format_func=lambda i: f"Mosquito {i}",
            key="fv_mosquito",
        )
    with col_c:
        jump = st.number_input(
            "Jump to frame #",
            min_value=0,
            max_value=len(frames) - 1,
            value=frame_idx,
            key="fv_jump",
        )
        if st.button("Go", key="fv_go") and int(jump) != frame_idx:
            st.session_state["frame_idx_override"] = int(jump)
            st.rerun()
    if "frame_idx_override" in st.session_state:
        frame_idx = st.session_state.pop("frame_idx_override")

    filename = frames[frame_idx]
    frame_bgr = load_frame_bgr(cfg["image_folder"], filename)
    if frame_bgr is None:
        st.error(f"Could not load {filename}")
        return default_activity_csv(paths, cfg)

    dist = frame_activity(activity_df, frame_idx, mosquito)
    pos = None
    frame_meta = (manifest or {}).get("frames", {}).get(filename, {}).get(
        f"mosquito_{mosquito}", {}
    )
    if frame_meta.get("pos"):
        pos = (frame_meta["pos"]["x"], frame_meta["pos"]["y"])
        if dist is None and frame_meta.get("distance") is not None:
            dist = float(frame_meta["distance"])

    cropped_bgr = cropper.crop(frame_bgr, mosquito)

    if cfg["live_detect"] and Path(cfg["model_path"]).is_file():
        try:
            model = load_model(cfg["model_path"], pick_device(cfg["device"]))
            live_pos = detect_pos_from_crop(model, cropped_bgr)
            if live_pos:
                pos = live_pos
                st.info("Live detection active — position from current YOLO run.")
        except ImportError as exc:
            st.warning(str(exc))

    overlay_bgr = draw_overlay(frame_bgr, cropper, mosquito, pos, dist)

    st.markdown(f"**File:** `{filename}` · **Frame:** {frame_idx} / {len(frames) - 1}")
    if dist is not None:
        st.markdown(f"**Activity (distance):** {dist:.2f} px")
    else:
        st.markdown("**Activity:** no detection / NaN")

    left, mid, right = st.columns(3)
    with left:
        st.markdown("**Full frame**")
        st.image(bgr_to_pil(frame_bgr), use_container_width=True)
        st.download_button(
            "Download full frame",
            data=pil_to_bytes(bgr_to_pil(frame_bgr)),
            file_name=filename,
            mime="image/jpeg",
            key="dl_full",
        )
    with mid:
        st.markdown("**Well crop**")
        st.image(bgr_to_pil(cropped_bgr), use_container_width=True)
        crop_name = f"crop_m{mosquito}_{Path(filename).stem}.jpg"
        st.download_button(
            "Download crop",
            data=pil_to_bytes(bgr_to_pil(cropped_bgr)),
            file_name=crop_name,
            mime="image/jpeg",
            key="dl_crop",
        )
    with right:
        st.markdown("**Annotated overlay**")
        st.image(bgr_to_pil(overlay_bgr), use_container_width=True)
        ann_name = f"annotated_m{mosquito}_{Path(filename).stem}.jpg"
        st.download_button(
            "Download annotated",
            data=pil_to_bytes(bgr_to_pil(overlay_bgr)),
            file_name=ann_name,
            mime="image/jpeg",
            key="dl_ann",
        )

    preview_root = cfg["output_dir"] / "previews"
    preview_file = preview_root / f"mosquito_{mosquito}" / f"{Path(filename).stem}.jpg"
    crop_preview = preview_root / f"mosquito_{mosquito}" / "crops" / f"{Path(filename).stem}.jpg"
    if preview_file.is_file() or crop_preview.is_file():
        st.divider()
        st.subheader("Saved previews from inference run")
        pcol1, pcol2 = st.columns(2)
        if preview_file.is_file():
            with pcol1:
                st.image(str(preview_file), caption="Saved annotated preview", use_container_width=True)
                st.download_button(
                    "Download saved preview",
                    data=preview_file.read_bytes(),
                    file_name=preview_file.name,
                    mime="image/jpeg",
                    key="dl_saved_preview",
                )
        if crop_preview.is_file():
            with pcol2:
                st.image(str(crop_preview), caption="Saved crop preview", use_container_width=True)
                st.download_button(
                    "Download saved crop preview",
                    data=crop_preview.read_bytes(),
                    file_name=crop_preview.name,
                    mime="image/jpeg",
                    key="dl_saved_crop",
                )

    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        if st.button("⏮ First", key="fv_first") and frame_idx != 0:
            st.session_state["frame_idx_override"] = 0
            st.rerun()
    with nav2:
        if st.button("◀ Prev", key="fv_prev") and frame_idx > 0:
            st.session_state["frame_idx_override"] = frame_idx - 1
            st.rerun()
    with nav3:
        if st.button("Next ▶", key="fv_next") and frame_idx < len(frames) - 1:
            st.session_state["frame_idx_override"] = frame_idx + 1
            st.rerun()
    with nav4:
        if st.button("Last ⏭", key="fv_last") and frame_idx != len(frames) - 1:
            st.session_state["frame_idx_override"] = len(frames) - 1
            st.rerun()

    if activity_df is not None:
        st.divider()
        st.subheader(f"Activity trace — mosquito {mosquito}")
        col_name = f"mosquito_{mosquito}"
        if col_name in activity_df.columns:
            chart_df = activity_df[["frame", col_name]].rename(columns={col_name: "distance"})
            st.line_chart(chart_df, x="frame", y="distance", height=220)
            buf = io.BytesIO()
            activity_df.to_csv(buf, index=False)
            st.download_button(
                "Download full activity CSV",
                data=buf.getvalue(),
                file_name=Path(csv_path).name if csv_path else "activity.csv",
                mime="text/csv",
                key="dl_activity_csv",
            )

    return str(csv_path) if csv_path else ""


def get_frame_list(image_folder: str, manifest: dict | None) -> list[str]:
    if manifest and manifest.get("image_paths"):
        return manifest["image_paths"]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    names = [
        f.name
        for f in Path(image_folder).iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ]
    return sorted(names, key=lambda x: float(Path(x).stem))


def frame_activity(df: pd.DataFrame | None, frame_idx: int, mosquito: int) -> float | None:
    if df is None or frame_idx >= len(df):
        return None
    col = f"mosquito_{mosquito}"
    if col not in df.columns:
        return None
    val = df.iloc[frame_idx][col]
    if pd.isna(val):
        return None
    return float(val)


def default_activity_csv(paths: dict, cfg: dict) -> str:
    csv_path = paths.get("csv_path")
    if csv_path and Path(csv_path).is_file():
        return str(csv_path)
    out_dir = cfg.get("output_dir", paths["output_dir"])
    candidates = sorted(Path(out_dir).glob("*.csv"))
    if candidates:
        return str(candidates[0])
    return ""


def render_frame_viewer() -> str:
    """Backward-compatible wrapper."""
    paths = resolve_paths()
    cfg = render_frame_sidebar(paths)
    return render_frame_viewer_body(cfg, paths)
