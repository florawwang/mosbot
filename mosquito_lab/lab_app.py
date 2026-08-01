"""
mosbot — frame viewer, detection inspector, and activity graphs.

Run:
    streamlit run mosquito_lab/lab_app.py --server.port 8502

Passcode: set CLOUD_VIEWER_PASSCODE or .streamlit/secrets.toml (never commit secrets).
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
from mosquito_lab.combine_ui import render_combine_body, render_combine_sidebar
from mosquito_lab.frame_viewer import (
    render_frame_sidebar,
    render_frame_viewer_body,
    resolve_paths,
)
from mosquito_lab.inspector.app import render_inspector_body, render_inspector_sidebar


def get_passcode() -> str | None:
    """Read passcode from env or Streamlit secrets — never hardcode in the repo."""
    env = (os.environ.get("CLOUD_VIEWER_PASSCODE") or "").strip()
    if env:
        return env
    try:
        # secrets.toml / Streamlit Cloud secrets (not in git)
        if "CLOUD_VIEWER_PASSCODE" in st.secrets:
            return str(st.secrets["CLOUD_VIEWER_PASSCODE"]).strip()
        if "passcode" in st.secrets:
            return str(st.secrets["passcode"]).strip()
    except Exception:
        pass
    return None


def is_hosted() -> bool:
    """True on the hosted (Streamlit Community Cloud) deployment.

    Frame images + Detection inspector need the full raw-frame folder on local
    disk, which isn't practical on Cloud's ephemeral storage, so we hide them
    there and show only Activity graphs + Combine (CSV-only, cheap to load).

    Detection is automatic, but can be forced either way with MOSBOT_HOSTED
    (env var or Streamlit secret): "1"/"true" = hosted, "0"/"false" = full app.
    """
    override = (os.environ.get("MOSBOT_HOSTED") or "").strip().lower()
    if not override:
        try:
            if "MOSBOT_HOSTED" in st.secrets:
                override = str(st.secrets["MOSBOT_HOSTED"]).strip().lower()
        except Exception:
            override = ""
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False

    # Auto-detect Streamlit Community Cloud (apps run from /mount/src as appuser).
    if "/mount/src" in str(Path(__file__).resolve()) or os.path.isdir("/mount/src"):
        return True
    if (os.environ.get("HOSTNAME") or "").lower().startswith("streamlit"):
        return True
    return False


def check_auth(passcode: str | None) -> bool:
    if st.session_state.get("authenticated"):
        return True

    st.title("mosbot")
    if not passcode:
        st.error("Passcode is not configured.")
        st.markdown(
            "Set **`CLOUD_VIEWER_PASSCODE`** in the environment, or add it to "
            "**.streamlit/secrets.toml** / Streamlit Cloud → **Settings → Secrets**.\n\n"
            "Example `secrets.toml` (do not commit this file):\n\n"
            "```toml\nCLOUD_VIEWER_PASSCODE = \"your-secret\"\n```"
        )
        return False

    if is_hosted():
        st.caption("Activity graphs and combine experiments. Enter the passcode to continue.")
    else:
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

    if not check_auth(get_passcode()):
        return

    paths = resolve_paths()

    st.sidebar.title("mosbot")

    hosted = is_hosted()
    if hosted:
        # Frame images + Detection inspector need the raw-frame folder on local
        # disk, which isn't practical on Cloud — show only the CSV-based sections.
        sections = ["Activity graphs", "Combine experiments"]
        radio_help = "Plot activity or combine experiments."
    else:
        sections = [
            "Frame images",
            "Detection inspector",
            "Activity graphs",
            "Combine experiments",
        ]
        radio_help = "Browse frames, audit YOLO detections, plot activity, or combine experiments."

    # A previously-stored selection may not exist in the current section list
    # (e.g. switching between local and hosted); reset it so the radio is valid.
    if st.session_state.get("lab_page") not in sections:
        st.session_state.pop("lab_page", None)

    page = st.sidebar.radio(
        "Section",
        sections,
        key="lab_page",
        help=radio_help,
    )

    cfg = None
    settings = None
    insp_cfg = None
    combine_settings = None
    if page == "Frame images":
        cfg = render_frame_sidebar(paths)
    elif page == "Detection inspector":
        insp_cfg = render_inspector_sidebar()
    elif page == "Activity graphs":
        settings = render_graphs_sidebar()
    else:
        combine_settings = render_combine_sidebar()

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
    elif page == "Activity graphs":
        st.caption(
            "Activity analysis graphs. Hover the **?** next to a figure title for help; "
            "use the **Plot style** sidebar to change labels, size, or fonts."
        )
        render_activity_graphs_body(settings)
    else:
        st.caption(
            "Combine several experiments onto one ZT timeline: kinds line up, each "
            "experiment keeps its true ZT start and LD/DD schedule, and gaps become NaN."
        )
        render_combine_body(combine_settings)


if __name__ == "__main__":
    main()
