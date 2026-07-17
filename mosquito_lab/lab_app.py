"""
mosbot — frame viewer, detection inspector, and activity graphs.

Run:
    streamlit run mosquito_lab/lab_app.py --server.port 8502

Passcode default: florawang
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit Cloud often runs this file as the main module
# (`mosquito_lab/lab_app.py`), so the repo root must be on sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from mosquito_lab.activity_plots import render_activity_graphs_body, render_graphs_sidebar
from mosquito_lab.frame_viewer import (
    render_frame_sidebar,
    render_frame_viewer_body,
    resolve_paths,
)
from mosquito_lab.inspector.app import render_inspector_body, render_inspector_sidebar

DEFAULT_PASSCODE = os.environ.get("CLOUD_VIEWER_PASSCODE", "florawang")


def check_auth(passcode: str) -> bool:
    if st.session_state.get("authenticated"):
        return True
    st.title("mosbot")
    st.caption(
        "Frame viewer, detection inspector, and activity graphs. "
        "Enter the passcode to continue."
    )
    with st.form("passcode_form", clear_on_submit=False, border=False):
        entered = st.text_input("Passcode", type="password", key="passcode_input")
        submitted = st.form_submit_button("Unlock", type="primary")
    if submitted:
        if entered == passcode:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect passcode.")
    return False


def main() -> None:
    st.set_page_config(page_title="mosbot", layout="wide", page_icon="🦟")

    if not check_auth(DEFAULT_PASSCODE):
        return

    paths = resolve_paths()

    st.sidebar.title("mosbot")
    page = st.sidebar.radio(
        "Section",
        ["Frame images", "Detection inspector", "Activity graphs"],
        key="lab_page",
        help="Browse frames, audit YOLO detections, or plot activity.",
    )

    if page == "Frame images":
        cfg = render_frame_sidebar(paths)
        settings = None
        insp_cfg = None
    elif page == "Detection inspector":
        cfg = None
        settings = None
        insp_cfg = render_inspector_sidebar()
    else:
        cfg = None
        settings = render_graphs_sidebar()
        insp_cfg = None

    st.title("mosbot")
    if page == "Frame images":
        st.caption("Browse and download per-frame images with detection overlays.")
        csv_path = render_frame_viewer_body(cfg, paths)
        if csv_path:
            st.session_state["graphs_default_csv"] = csv_path
    elif page == "Detection inspector":
        st.caption(
            "Audit YOLO detections: overlay wells, slide confidence, jump to misses, flag frames."
        )
        render_inspector_body(insp_cfg)
    else:
        st.caption(
            "Activity analysis graphs. Hover the **?** next to a figure title for help; "
            "use the **Plot style** sidebar to change labels, size, or fonts."
        )
        render_activity_graphs_body(settings)


if __name__ == "__main__":
    main()
