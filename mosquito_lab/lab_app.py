"""
mosbot — activity graphs and combine experiments.

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
from mosquito_lab.critters import render_mosquito_swarm


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

    st.caption("Activity graphs and combine experiments. Enter the passcode to continue.")
    with st.form("passcode_form", clear_on_submit=False, border=False):
        entered = st.text_input("Passcode", type="password", key="passcode_input")
        submitted = st.form_submit_button("Unlock", type="primary")
    st.toggle(
        "🦟 Mosquito mode",
        value=True,
        key="mosquitoes_on",
        help="Turn the roaming mosquitoes and the mosquito cursor on or off.",
    )
    if submitted:
        if entered == passcode:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect passcode.")
    return False


def main() -> None:
    st.set_page_config(page_title="mosbot", layout="wide", page_icon="🦟")
    # Mosquito cursor + a few roaming mosquitoes 🦟 (toggle lives on the login
    # screen and in the sidebar, both keyed "mosquitoes_on"; default on).
    render_mosquito_swarm(enabled=st.session_state.get("mosquitoes_on", True))

    if not check_auth(get_passcode()):
        return

    st.sidebar.title("mosbot")
    st.sidebar.toggle(
        "🦟 Mosquito mode",
        value=True,
        key="mosquitoes_on",
        help="Roaming mosquitoes + mosquito cursor.",
    )
    page = st.sidebar.radio(
        "Section",
        ["Activity graphs", "Combine experiments"],
        key="lab_page",
        help="Plot activity or combine experiments.",
    )

    if page == "Activity graphs":
        settings = render_graphs_sidebar()
    else:
        combine_settings = render_combine_sidebar()

    st.title("mosbot")
    if page == "Activity graphs":
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
