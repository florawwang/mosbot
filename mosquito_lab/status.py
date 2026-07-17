"""Lightweight inference status tracking for cloud runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_status() -> dict[str, Any]:
    return {
        "state": "idle",
        "mosquito_num": None,
        "mosquito_total": None,
        "current_photo": 0,
        "total_photos": 0,
        "error": None,
        "output_file": None,
        "viewer_url": None,
        "updated_at": _utc_now(),
    }


def write_status(status_path: str, **fields: Any) -> None:
    try:
        status = default_status()
        if os.path.isfile(status_path):
            try:
                with open(status_path, encoding="utf-8") as f:
                    status.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        status.update(fields)
        status["updated_at"] = _utc_now()
        os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
        tmp = f"{status_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, status_path)
    except Exception as exc:
        print(f"[status] could not write ({type(exc).__name__}: {exc})")


def read_status(status_path: str) -> dict[str, Any]:
    if not os.path.isfile(status_path):
        return default_status()
    try:
        with open(status_path, encoding="utf-8") as f:
            data = default_status()
            data.update(json.load(f))
            return data
    except (json.JSONDecodeError, OSError):
        return default_status()
