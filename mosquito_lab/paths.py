"""Shared path helpers for the Mosquito Lab product."""

from __future__ import annotations

from pathlib import Path


def product_root() -> Path:
    """mosquito-lab/ directory."""
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Parent repo root (contains MosquitoMovement2/ when nested in Rijo-Ferreira Lab)."""
    for ancestor in Path(__file__).resolve().parents:
        if (ancestor / "MosquitoMovement2").is_dir():
            return ancestor
    return product_root().parent


def mosquito_project_dir() -> Path:
    return repo_root() / "MosquitoMovement2"
