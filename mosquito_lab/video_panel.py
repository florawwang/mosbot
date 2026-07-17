"""Streamlit panel for playing or building frame-sequence videos."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mosquito_lab.frame_viewer import resolve_paths
from mosquito_lab.image_util import Cropper
from mosquito_lab.video import (
    cached_video_path,
    find_existing_videos,
    get_or_build_cached_video,
    list_frame_names,
)


def render_video_sidebar(paths: dict) -> dict:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Inputs")
    st.sidebar.caption("Image folder to play as a video or stitch into a new MP4.")
    image_folder = st.sidebar.text_input(
        "Image folder",
        value=paths["image_folder"],
        key="vid_image_folder",
    )
    label_file = st.sidebar.text_input(
        "Labels CSV",
        value=paths["label_file"],
        help="Required for well-crop or annotated videos.",
        key="vid_label_file",
    )
    output_dir = st.sidebar.text_input(
        "Output directory",
        value=str(paths["output_dir"]),
        key="vid_output_dir",
    )

    existing = []
    if image_folder and Path(image_folder).is_dir():
        existing = find_existing_videos(image_folder, output_dir)

    st.sidebar.markdown("#### Video source")
    source_options = ["Build from images"]
    if existing:
        source_options = [f"Use existing: {p.name}" for p in existing] + source_options

    source_choice = st.sidebar.selectbox(
        "Source",
        source_options,
        key="vid_source_choice",
    )
    use_existing = source_choice.startswith("Use existing:")
    existing_path = ""
    if use_existing:
        idx = source_options.index(source_choice)
        existing_path = str(existing[idx])

    st.sidebar.markdown("#### Build settings")
    st.sidebar.caption("Only used when building from images.")
    fps = st.sidebar.slider("FPS", min_value=1, max_value=60, value=10, key="vid_fps")
    view_mode = st.sidebar.selectbox(
        "View",
        ["full", "crop", "annotated"],
        format_func=lambda x: {
            "full": "Full frame",
            "crop": "Well crop (one mosquito)",
            "annotated": "Full frame + well overlay",
        }[x],
        key="vid_view_mode",
    )
    mosquito_num = st.sidebar.number_input(
        "Mosquito #",
        min_value=0,
        value=0,
        step=1,
        key="vid_mosquito_num",
        disabled=view_mode == "full",
    )
    stride = st.sidebar.number_input(
        "Stride (every Nth frame)",
        min_value=1,
        value=1,
        step=1,
        key="vid_stride",
        help="Use 2–10 for a quicker preview of long experiments.",
    )
    max_frames = st.sidebar.number_input(
        "Max frames (0 = all)",
        min_value=0,
        value=0,
        step=100,
        key="vid_max_frames",
    )
    label_frames = st.sidebar.checkbox(
        "Burn filename on each frame",
        value=True,
        key="vid_label_frames",
    )

    return {
        "image_folder": image_folder,
        "label_file": label_file,
        "output_dir": Path(output_dir),
        "use_existing": use_existing,
        "existing_path": existing_path,
        "fps": float(fps),
        "view_mode": view_mode,
        "mosquito_num": int(mosquito_num),
        "stride": int(stride),
        "max_frames": int(max_frames),
        "label_frames": label_frames,
    }


def _show_video_file(video_path: Path, *, download_key: str) -> None:
    st.video(str(video_path))
    st.download_button(
        "Download video",
        data=video_path.read_bytes(),
        file_name=video_path.name,
        mime="video/mp4",
        key=download_key,
    )


def render_video_body(cfg: dict, paths: dict) -> None:
    if not cfg["image_folder"] or not Path(cfg["image_folder"]).is_dir():
        st.warning("Set a valid image folder in the sidebar.")
        return

    frame_names = list_frame_names(cfg["image_folder"])
    if not frame_names and not cfg["use_existing"]:
        st.error("No images found in the image folder.")
        return

    if not cfg["use_existing"]:
        st.markdown(f"**{len(frame_names)}** frames available to stitch")

    if cfg["use_existing"]:
        video_path = Path(cfg["existing_path"])
        if not video_path.is_file():
            st.error(f"Video not found: {video_path}")
            return
        st.success(f"Playing existing video: `{video_path}`")
        _show_video_file(video_path, download_key="vid_dl_existing")
        return

    if cfg["view_mode"] in {"crop", "annotated"}:
        if not cfg["label_file"] or not Path(cfg["label_file"]).is_file():
            st.warning("Set a valid labels CSV in the sidebar for crop / annotated mode.")
            return
        try:
            n_mosq = len(Cropper(cfg["label_file"]).boxes)
            if cfg["mosquito_num"] >= n_mosq:
                st.warning(f"Mosquito # must be 0–{n_mosq - 1}.")
                return
        except Exception as exc:
            st.error(f"Could not load labels: {exc}")
            return

    manifest_path = cfg["output_dir"] / "frame_manifest.json"
    if cfg["view_mode"] == "annotated" and not manifest_path.is_file():
        st.info(
            "No inference manifest found — overlay will show well boxes only. "
            "Run inference first for detection dots and distance labels."
        )

    cached = cached_video_path(
        image_folder=cfg["image_folder"],
        output_dir=cfg["output_dir"],
        frame_names=frame_names,
        fps=cfg["fps"],
        view_mode=cfg["view_mode"],
        mosquito_num=cfg["mosquito_num"],
        stride=cfg["stride"],
        max_frames=cfg["max_frames"],
        label_frames=cfg["label_frames"],
    )

    build = st.button("Build / refresh video", type="primary", key="vid_build_btn")

    if cached and not build:
        st.success(f"Cached video: `{cached}`")
        _show_video_file(cached, download_key="vid_dl_cached")
        st.caption("Change settings and click **Build / refresh video** to regenerate.")
        return

    if not build:
        est = len(frame_names) // cfg["stride"]
        if cfg["max_frames"] > 0:
            est = min(est, cfg["max_frames"])
        st.info(
            f"Click **Build / refresh video** to stitch **{est}** frames "
            f"at **{cfg['fps']} fps** (~{est / cfg['fps']:.0f}s playback). "
            "Videos cache under `output_dir/videos/`."
        )
        return

    progress = st.progress(0, text="Preparing video…")

    def _progress(done: int, total: int, name: str) -> None:
        pct = int(100 * done / max(total, 1))
        progress.progress(pct, text=f"Writing frame {done}/{total}: {name}")

    try:
        out_path = get_or_build_cached_video(
            image_folder=cfg["image_folder"],
            output_dir=cfg["output_dir"],
            frame_names=frame_names,
            fps=cfg["fps"],
            view_mode=cfg["view_mode"],
            mosquito_num=cfg["mosquito_num"],
            label_file=cfg["label_file"],
            manifest_path=manifest_path if manifest_path.is_file() else None,
            stride=cfg["stride"],
            max_frames=cfg["max_frames"],
            label_frames=cfg["label_frames"],
            progress_callback=_progress,
        )
    except Exception as exc:
        progress.empty()
        st.error(f"Video build failed: {exc}")
        return

    progress.empty()
    st.success(f"Video ready: `{out_path}` ({out_path.stat().st_size / 1_048_576:.1f} MB)")
    _show_video_file(out_path, download_key="vid_dl_built")
    st.caption(
        f"Settings: {cfg['view_mode']} · {cfg['fps']} fps · stride {cfg['stride']}"
        + (f" · max {cfg['max_frames']} frames" if cfg["max_frames"] else "")
    )


def render_video_panel() -> None:
    paths = resolve_paths()
    cfg = render_video_sidebar(paths)
    render_video_body(cfg, paths)
