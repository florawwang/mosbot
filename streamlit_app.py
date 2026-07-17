"""Entry point for Streamlit Community Cloud (share.streamlit.io).

Deploy settings:
  Main file: streamlit_app.py
  Python version: 3.10+
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mosquito_lab.lab_app import main

if __name__ == "__main__":
    main()
