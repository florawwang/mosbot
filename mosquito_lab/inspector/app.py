"""Detection Quality Inspector — Label-Studio-style GUI for auditing YOLO detections.

Standalone:
    streamlit run mosquito_lab/inspector/app.py

Or open the **Detection inspector** section inside mosbot (`lab_app.py`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow standalone `streamlit run mosquito_lab/inspector/app.py`
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from mosquito_lab.inspector import inspector_core as core
from mosquito_lab.paths import mosquito_project_dir

# Colors (RGB).
C_WELL = (120, 120, 120)
C_DETECTED = (46, 204, 113)  # green — detected at current threshold
C_MISSED = (231, 76, 60)  # red — missed at current threshold
C_CANDIDATE = (241, 196, 15)  # yellow — below threshold but above floor
C_FOCUS = (52, 152, 219)  # blue — focused well


@st.cache_data(show_spinner=False)
def _cached_load(cache_file: str, mtime: float):
    """Load the parquet cache. mtime busts the cache when the file changes."""
    return pd.read_parquet(cache_file)


def load_experiment_data(exp: core.Experiment):
    mtime = os.path.getmtime(exp.cache_file)
    df = _cached_load(exp.cache_file, mtime)
    _, meta = core.load_cache(exp)
    return df, meta


@st.cache_data(show_spinner=False)
def _load_frame_image(image_folder: str, name: str) -> Image.Image:
    return Image.open(os.path.join(image_folder, name)).convert("RGB")


def _font(size: int = 13):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_overlay(
    img: Image.Image,
    wells: list[core.Well],
    frame_df: pd.DataFrame,
    threshold: float,
    show_wells: bool,
    show_labels: bool,
    show_candidates: bool,
    focus_well: int | None,
) -> Image.Image:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(13)

    detected = core.detected_wells(frame_df, threshold)

    if show_wells:
        for well in wells:
            if focus_well is not None and well.idx == focus_well:
                color, width = C_FOCUS, 3
            elif well.degenerate:
                color, width = C_WELL, 1
            elif well.idx in detected:
                color, width = C_DETECTED, 2
            else:
                color, width = C_MISSED, 2
            draw.rectangle([well.x, well.y, well.x2, well.y2], outline=color, width=width)
            if show_labels:
                draw.text((well.x + 2, well.y + 1), str(well.idx), fill=color, font=font)

    for _, r in frame_df.iterrows():
        above = r["conf"] >= threshold
        if not above and not show_candidates:
            continue
        color = C_DETECTED if above else C_CANDIDATE
        draw.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=color, width=2)
        if show_labels:
            label = f"{r['conf']:.2f}"
            tx, ty = r["x1"], max(0, r["y1"] - 14)
            draw.rectangle([tx, ty, tx + 30, ty + 13], fill=color)
            draw.text((tx + 1, ty), label, fill=(0, 0, 0), font=font)

    return canvas


def crop_focus(img: Image.Image, well: core.Well, pad: int = 20) -> Image.Image:
    x1 = max(0, well.x - pad)
    y1 = max(0, well.y - pad)
    x2 = min(img.width, well.x2 + pad)
    y2 = min(img.height, well.y2 + pad)
    return img.crop((x1, y1, x2, y2))


def select_experiment() -> core.Experiment | None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Experiment")
    discovered = core.discover_experiments()
    names = [e.name for e in discovered]
    options = names + ["Custom path..."]
    choice = st.sidebar.selectbox(
        "Choose experiment",
        options,
        index=0 if names else len(options) - 1,
        key="insp_experiment",
    )

    if choice == "Custom path...":
        img = st.sidebar.text_input("Raw images folder", key="insp_img")
        lbl = st.sidebar.text_input("Labels CSV", key="insp_lbl")
        default_model = str(mosquito_project_dir() / "models" / "uninf_det_v0.pt")
        mdl = st.sidebar.text_input("YOLO model (.pt)", value=default_model, key="insp_mdl")
        if not (img and lbl and mdl):
            st.sidebar.info("Enter image folder, labels CSV, and model path.")
            return None
        return core.Experiment(
            name="custom", image_folder=img, label_file=lbl, model_path=mdl
        )

    return discovered[names.index(choice)]


def ensure_cache(exp: core.Experiment) -> bool:
    """Return True if a cache is available; otherwise offer to build one."""
    checks = exp.exists_check()
    if checks["cache"]:
        return True

    st.warning(f"No detection cache found for **{exp.name}**.")
    missing = [k for k in ("image_folder", "label_file", "model_path") if not checks[k]]
    if missing:
        st.error(f"Cannot build a cache — missing paths: {missing}")
        st.info(
            "On Streamlit Cloud, upload a prebuilt `.detection_cache/` or run "
            "`precompute.py` locally (needs `pip install -r requirements-ml.txt`)."
        )
        return False

    n_frames = len(core.list_frames(exp.image_folder))
    st.write(
        f"This experiment has **{n_frames}** frames. Building the full cache runs "
        "YOLO on every well of every frame (slow on CPU). For a quick look, build "
        "a strided sample first."
    )
    col1, col2 = st.columns(2)
    stride = col1.number_input(
        "Frame stride",
        min_value=1,
        value=max(1, n_frames // 300),
        step=1,
        key="insp_stride",
    )
    max_frames = col2.number_input(
        "Max frames (0 = no cap)", min_value=0, value=200, step=50, key="insp_max_frames"
    )

    if st.button("Build cache now", type="primary", key="insp_build_cache"):
        try:
            bar = st.progress(0.0, text="Starting YOLO...")

            def progress(done, total, name):
                bar.progress(done / max(total, 1), text=f"{done}/{total}  {name}")

            core.build_cache(
                exp,
                stride=int(stride),
                max_frames=int(max_frames) or None,
                progress=progress,
            )
            st.success("Cache built. Reloading...")
            st.rerun()
        except ImportError as exc:
            st.error(str(exc))
            st.info("Install YOLO deps locally: `pip install -r requirements-ml.txt`")
    return False


def render_inspector_sidebar() -> dict | None:
    """Sidebar controls for the detection inspector. Returns config or None."""
    exp = select_experiment()
    if exp is None:
        return None

    st.sidebar.markdown("#### Confidence")
    threshold = st.sidebar.slider(
        "Detection threshold", 0.0, 1.0, 0.25, 0.01, key="insp_threshold"
    )
    show_candidates = st.sidebar.checkbox(
        "Show below-threshold candidates (yellow)",
        value=True,
        key="insp_show_candidates",
        help="Boxes YOLO found above the cache floor but below the current threshold.",
    )

    st.sidebar.markdown("#### Overlay")
    show_wells = st.sidebar.checkbox("Show well boxes", value=True, key="insp_show_wells")
    show_labels = st.sidebar.checkbox(
        "Show labels / confidences", value=True, key="insp_show_labels"
    )

    st.sidebar.markdown("#### Focus well")
    focus_enabled = st.sidebar.checkbox(
        "Focus a single mosquito", value=False, key="insp_focus_enabled"
    )
    focus_well = None
    if focus_enabled:
        try:
            wells = core.load_wells(exp.label_file)
            well_ids = [w.idx for w in wells]
            focus_well = st.sidebar.selectbox(
                "Well index", well_ids, key="insp_focus_well"
            )
        except Exception:
            focus_well = st.sidebar.number_input(
                "Well index", min_value=0, value=0, step=1, key="insp_focus_well_num"
            )

    return {
        "exp": exp,
        "threshold": threshold,
        "show_candidates": show_candidates,
        "show_wells": show_wells,
        "show_labels": show_labels,
        "focus_enabled": focus_enabled,
        "focus_well": focus_well,
    }


def render_inspector_body(cfg: dict | None) -> None:
    """Main panel for detection QA (threshold, miss jumps, flags)."""
    if cfg is None:
        st.info("Pick an experiment (or custom paths) in the sidebar to begin.")
        return

    exp: core.Experiment = cfg["exp"]
    if not ensure_cache(exp):
        return

    df, meta = load_experiment_data(exp)
    wells = core.load_wells(exp.label_file)
    all_wells = [w.idx for w in wells]
    frames = core.list_frames(exp.image_folder)
    processed = meta.get("processed_frame_indices") or sorted(
        df["frame_idx"].unique().tolist()
    )
    if not processed:
        st.error("Cache is empty. Rebuild it with precompute.py.")
        return

    focus_enabled = cfg["focus_enabled"]
    focus_well = cfg.get("focus_well")
    threshold = cfg["threshold"]
    show_candidates = cfg["show_candidates"]
    show_wells = cfg["show_wells"]
    show_labels = cfg["show_labels"]

    st.sidebar.markdown("#### Cache info")
    st.sidebar.caption(
        f"floor {meta.get('conf_floor', '?')} · stride {meta.get('stride', '?')} · "
        f"{len(processed)}/{meta.get('n_frames_total', len(frames))} frames · "
        f"{len(df)} boxes"
    )

    key = f"insp_pos::{exp.cache_file}"
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] = max(0, min(st.session_state[key], len(processed) - 1))

    st.subheader("Navigate")
    nav = st.columns([1, 1, 1, 1, 3])
    if nav[0].button("First", key="insp_nav_first"):
        st.session_state[key] = 0
    if nav[1].button("Prev", key="insp_nav_prev"):
        st.session_state[key] = max(0, st.session_state[key] - 1)
    if nav[2].button("Next", key="insp_nav_next"):
        st.session_state[key] = min(len(processed) - 1, st.session_state[key] + 1)
    if nav[3].button("Last", key="insp_nav_last"):
        st.session_state[key] = len(processed) - 1

    pos = st.session_state[key]
    current_frame = processed[pos]

    slider_pos = nav[4].slider(
        "Frame position",
        0,
        len(processed) - 1,
        pos,
        key=f"insp_slider::{key}",
        label_visibility="collapsed",
    )
    if slider_pos != pos:
        st.session_state[key] = slider_pos
        pos = slider_pos
        current_frame = processed[pos]

    st.subheader("Jump to missed detections")
    jump = st.columns([1, 1, 2, 3])
    jump_well = None
    if jump[2].checkbox(
        "Only the focused well",
        value=focus_enabled and focus_well is not None,
        key="insp_jump_focused",
    ):
        jump_well = focus_well
    if jump[0].button("Prev missed", key="insp_prev_missed"):
        nxt = core.next_missed_frame(
            df, threshold, processed, all_wells, current_frame, -1, jump_well
        )
        if nxt is not None:
            st.session_state[key] = processed.index(nxt)
            st.rerun()
        else:
            st.toast("No earlier missed frame.")
    if jump[1].button("Next missed", key="insp_next_missed"):
        nxt = core.next_missed_frame(
            df, threshold, processed, all_wells, current_frame, 1, jump_well
        )
        if nxt is not None:
            st.session_state[key] = processed.index(nxt)
            st.rerun()
        else:
            st.toast("No later missed frame.")

    frame_name = frames[current_frame]
    frame_df = df[df["frame_idx"] == current_frame]
    detected = core.detected_wells(frame_df, threshold)
    missed = core.missed_wells(
        frame_df, threshold, [w.idx for w in wells if not w.degenerate]
    )
    n_real_wells = sum(1 for w in wells if not w.degenerate)

    flags = core.load_flags(exp)
    is_flagged = str(current_frame) in flags

    m = st.columns(4)
    m[0].metric("Frame", f"{frame_name}", f"idx {current_frame}")
    m[1].metric("Detected wells", f"{len(detected)}/{n_real_wells}")
    m[2].metric("Missed wells", len(missed), delta=None)
    m[3].metric("Flagged", "Yes" if is_flagged else "No")

    img = _load_frame_image(exp.image_folder, frame_name)
    overlay = draw_overlay(
        img,
        wells,
        frame_df,
        threshold,
        show_wells,
        show_labels,
        show_candidates,
        focus_well,
    )

    left, right = (
        st.columns([3, 1])
        if focus_enabled and focus_well is not None
        else (st.container(), None)
    )
    with left:
        st.image(
            overlay,
            use_container_width=True,
            caption=(
                f"{frame_name} — green=detected, red=missed well, "
                "yellow=below-threshold candidate"
            ),
        )
    if right is not None:
        fw = next(w for w in wells if w.idx == focus_well)
        with right:
            st.markdown(f"**Well {focus_well}**")
            st.image(crop_focus(overlay, fw), use_container_width=True)
            wdf = frame_df[frame_df["well"] == focus_well].sort_values(
                "conf", ascending=False
            )
            if wdf.empty:
                st.error("No candidate boxes even at the cache floor.")
            else:
                st.caption("Candidate confidences here:")
                st.write(", ".join(f"{c:.2f}" for c in wdf["conf"]))

    if missed:
        st.warning(
            f"Missed wells at threshold {threshold:.2f}: "
            + ", ".join(map(str, missed))
        )

    st.subheader("Flag this frame")
    fcols = st.columns([2, 1, 1])
    reason = fcols[0].text_input(
        "Reason",
        value=flags.get(str(current_frame), {}).get("reason", ""),
        placeholder="e.g. glare, mosquito outside well, motion blur",
        key="insp_flag_reason",
    )
    if fcols[1].button(
        ("Update flag" if is_flagged else "Flag frame"),
        type="primary",
        key="insp_flag",
    ):
        core.set_flag(exp, current_frame, True, reason)
        st.rerun()
    if is_flagged and fcols[2].button("Unflag", key="insp_unflag"):
        core.set_flag(exp, current_frame, False)
        st.rerun()

    tab_boxes, tab_wells, tab_frames, tab_flags = st.tabs(
        ["Boxes (this frame)", "Per-well miss rate", "Detections over time", "Flagged frames"]
    )

    with tab_boxes:
        if frame_df.empty:
            st.info("No candidate boxes on this frame.")
        else:
            view = frame_df.copy()
            view["above_threshold"] = view["conf"] >= threshold
            view = view[["well", "conf", "above_threshold", "x1", "y1", "x2", "y2"]]
            view = view.sort_values(["well", "conf"], ascending=[True, False])
            st.dataframe(view, use_container_width=True, hide_index=True)

    with tab_wells:
        real = [w.idx for w in wells if not w.degenerate]
        wr = core.per_well_missed_rate(df, threshold, processed, real)
        wr = wr.sort_values("missed_rate", ascending=False)
        st.caption("Wells sorted by how often they are missed across all cached frames.")
        st.dataframe(wr, use_container_width=True, hide_index=True)
        st.bar_chart(wr.set_index("well")["missed_rate"])

    with tab_frames:
        counts = core.per_frame_detection_counts(df, threshold, processed)
        st.caption(f"Wells detected per frame at threshold {threshold:.2f} (of {n_real_wells}).")
        st.line_chart(counts.set_index("frame_idx")["detected"])

    with tab_flags:
        if not flags:
            st.info("No frames flagged yet.")
        else:
            rows = []
            for k in sorted(flags, key=lambda x: int(x)):
                fidx = int(k)
                rows.append(
                    {
                        "frame_idx": fidx,
                        "image_name": frames[fidx] if fidx < len(frames) else "",
                        "reason": flags[k].get("reason", ""),
                        "updated_at": flags[k].get("updated_at", ""),
                    }
                )
            fdf = pd.DataFrame(rows)
            st.dataframe(fdf, use_container_width=True, hide_index=True)
            if st.button("Export flagged frames to CSV", key="insp_export_flags"):
                path = core.export_flags_csv(exp, frames)
                st.success(f"Wrote {path}")
            chosen = st.selectbox(
                "Go to flagged frame", fdf["frame_idx"].tolist(), key="insp_goto_flag"
            )
            if st.button("Jump there", key="insp_jump_flag"):
                if chosen in processed:
                    st.session_state[key] = processed.index(chosen)
                    st.rerun()
                else:
                    st.warning(
                        "That frame is not in the processed set (increase stride coverage)."
                    )


def main() -> None:
    st.set_page_config(
        page_title="Detection Quality Inspector", layout="wide", page_icon="🦟"
    )
    st.title("Detection Quality Inspector")
    st.caption(
        "Audit YOLO mosquito detections — scroll, overlay, threshold, flag, and jump to misses."
    )
    cfg = render_inspector_sidebar()
    render_inspector_body(cfg)


if __name__ == "__main__":
    main()
