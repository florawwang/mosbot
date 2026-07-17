"""
Mosquito Activity Explorer
==========================

A self-contained Streamlit web app that reproduces the
"[NN] analyze_activity_model" notebook — every figure, all nine of them —
but lets you *plug in your own values* instead of editing notebook code:

    - which activity CSV to load
    - ZT offset (start ZT of the first bin)
    - bin size, circadian period, LD/DD split
    - the mosquito-kind layout (group name -> row range -> color)
    - optional death calls (mosquito goes NaN after death)

The figures mirror the notebook's table of contents:

    Section A — General
      Fig 1  Individual actograms, pre-death-cut
      Fig 2  Individual actograms, death-cut
      Fig 3  Group-mean death comparison (cut at death vs 24 h before)
    Section B — LD
      Fig 4  Full LD period, averaged across mosquitoes
      Fig 5  LD, 24 h-folded
      Fig 6  LD, 24 h-folded mean +/- SD line plot
    Section C — DD
      Fig 7  Full DD period, averaged across mosquitoes
      Fig 8  DD, 24 h-folded
      Fig 9  DD, 24 h-folded mean +/- SD line plot

This file does NOT modify any existing notebook or data. It only reads a CSV.

Run it with the project's virtualenv:

    /Users/florawang/Downloads/Rijo-Ferreira\\ Lab/.venv/bin/python -m streamlit run \\
        "/Users/florawang/Downloads/Rijo-Ferreira Lab/activity_explorer.py"

Then open the URL it prints (usually http://localhost:8501).
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, safe for Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths / defaults
# --------------------------------------------------------------------------- #

LAB_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = LAB_ROOT / "MosquitoMovement2" / "experiments"

# The default layout mirrors the notebook (24 mosquitoes, 4 kinds of 6 each).
DEFAULT_GROUPS = pd.DataFrame(
    [
        {"name": "Female sg (WT)", "start": 0, "end": 6, "color": "#e63946"},
        {"name": "Male sg (WT)", "start": 6, "end": 12, "color": "#457b9d"},
        {"name": "Female KO", "start": 12, "end": 18, "color": "#f4a261"},
        {"name": "Male KO", "start": 18, "end": 24, "color": "#2a9d8f"},
    ]
)

# Empty template for death calls. mosquito_num is 1-based WITHIN the group;
# death_frame is a raw frame number (converted to a bin via frame // bin_size).
DEFAULT_DEATHS = pd.DataFrame(
    {
        "group": pd.Series(dtype="str"),
        "mosquito_num": pd.Series(dtype="int"),
        "death_frame": pd.Series(dtype="int"),
    }
)

BAR_COLOR = "#ef3c26"  # the single bar color the notebook uses for Figs 1-2, 4-5, 7-8


# --------------------------------------------------------------------------- #
# Core analysis (parametrized versions of the notebook logic)
# --------------------------------------------------------------------------- #


def bin_sum(size: int, arr: np.ndarray) -> np.ndarray:
    """Sum activity into bins of `size` frames (e.g. 60 frames -> 1 hour)."""
    arr = np.asarray(arr, dtype=float)
    return np.array([np.sum(arr[i : i + size]) for i in range(0, len(arr), size)])


def load_activity(csv_source) -> pd.DataFrame:
    """
    Read an activity CSV whose columns look like frame, mosquito_0, mosquito_1...
    Returns a tidy frame with one row per mosquito and a 'frames' list column.
    """
    raw = pd.read_csv(csv_source)
    mosquito_cols = [c for c in raw.columns if c.startswith("mosquito_")]
    if not mosquito_cols:
        raise ValueError(
            "No columns starting with 'mosquito_' were found in this CSV."
        )
    return pd.DataFrame(
        {
            "Mosquito": mosquito_cols,
            "frames": [raw[c].fillna(0).to_numpy(dtype=float) for c in mosquito_cols],
        }
    )


def build_counts(data: pd.DataFrame, bin_size: int) -> list[np.ndarray]:
    """Binned activity trace per mosquito."""
    return [bin_sum(bin_size, f) for f in data["frames"]]


def groups_from_editor(groups_df: pd.DataFrame, n_mosq: int) -> dict[str, range]:
    """Turn the editable group table into an ordered {name: range} mapping."""
    groups: dict[str, range] = {}
    for _, row in groups_df.iterrows():
        name = str(row["name"]).strip()
        if not name:
            continue
        start = max(0, int(row["start"]))
        end = min(n_mosq, int(row["end"]))
        if end <= start:
            continue
        groups[name] = range(start, end)
    return groups


def colors_from_editor(groups_df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, row in groups_df.iterrows():
        name = str(row["name"]).strip()
        if name:
            out[name] = str(row["color"]).strip() or BAR_COLOR
    return out


def build_idx_to_label(groups: dict[str, range]) -> dict[int, tuple[str, int]]:
    """row index -> (group name, 1-based number within group)."""
    idx_to_label: dict[int, tuple[str, int]] = {}
    for name, indices in groups.items():
        for j, idx in enumerate(indices):
            idx_to_label[idx] = (name, j + 1)
    return idx_to_label


def death_bin_by_idx(
    deaths_df: pd.DataFrame, groups: dict[str, range], bin_size: int
) -> dict[int, int]:
    """
    Map row index -> death bin (in bin units), from the death-calls table.
    death_frame is a raw frame number; we convert to a bin via frame // bin_size.
    """
    out: dict[int, int] = {}
    for _, row in deaths_df.iterrows():
        group = str(row.get("group", "")).strip()
        if group not in groups:
            continue
        try:
            num = int(row["mosquito_num"])
            frame = int(row["death_frame"])
        except (ValueError, TypeError):
            continue
        indices = list(groups[group])
        if not (1 <= num <= len(indices)):
            continue
        out[indices[num - 1]] = frame // bin_size
    return out


def apply_death_cut(
    trace: np.ndarray,
    idx: int,
    death_bins: dict[int, int],
    start_zt: float,
    exclude_hours: float = 0.0,
) -> np.ndarray:
    """
    Set everything after death to NaN. The cut index is
    (death_bin - start_zt - exclude_hours), matching the notebook. Passing
    exclude_hours=24 pulls the cutoff 24 h earlier (the "24 h before" variant).
    """
    trace = np.asarray(trace, dtype=float).copy()
    if idx not in death_bins:
        return trace
    cut = int(np.floor(death_bins[idx] - start_zt - exclude_hours))
    cut = int(np.clip(cut, 0, len(trace)))
    trace[cut:] = np.nan
    return trace


def group_mean_trace(
    counts: list[np.ndarray],
    indices: range,
    death_bins: dict[int, int],
    start_zt: float,
    exclude_hours: float = 0.0,
) -> np.ndarray:
    """NaN-aware mean across a group's (death-cut) traces, padded to equal length."""
    traces = [
        apply_death_cut(counts[i], i, death_bins, start_zt, exclude_hours)
        for i in indices
    ]
    if not traces:
        return np.array([])
    length = max(len(t) for t in traces)
    padded = [np.concatenate([t, np.full(length - len(t), np.nan)]) for t in traces]
    return np.nanmean(padded, axis=0)


def fold_24h(
    counts: list[np.ndarray],
    indices: range,
    start_zt: float,
    period: int,
    lo: int,
    hi: int | None,
    death_bins: dict[int, int],
) -> np.ndarray:
    """
    24 h-folded, weight-normalized profile for the mosquitoes in `indices`, over
    the window [lo, hi). Returns an (n_in_group, period) array of per-ZT means.
    """
    rows = []
    for m in indices:
        acc = np.zeros(period)
        wgt = np.zeros(period)
        trace = counts[m]
        end = len(trace) if hi is None else min(hi, len(trace))
        death = death_bins.get(m, math.inf)
        for i in range(lo, end):
            if i >= death:
                continue
            val = trace[i]
            zt = (start_zt + i) % period
            low = int(np.floor(zt)) % period
            high = (low + 1) % period
            frac = zt - np.floor(zt)
            acc[low] += val * (1 - frac)
            acc[high] += val * frac
            wgt[low] += 1 - frac
            wgt[high] += frac
        rows.append(np.divide(acc, wgt, out=np.full(period, np.nan), where=wgt > 0))
    return np.array(rows)


def fold_mean_bar(
    counts: list[np.ndarray],
    indices: range,
    start_zt: float,
    period: int,
    lo: int,
    hi: int | None,
) -> np.ndarray:
    """
    Simple 24 h fold used by Figs 5 & 8: average, per ZT hour, across every
    mosquito AND every timepoint in [lo, hi) whose ZT falls in [h, h+1).
    """
    width = (len(counts[indices[0]]) if hi is None else hi) - lo
    width = max(width, 0)
    if width == 0:
        return np.zeros(period)
    traces = np.array(
        [np.asarray(counts[i], dtype=float)[lo : lo + width] for i in indices]
    )
    t = np.arange(lo, lo + traces.shape[1])
    zt_hour = (t + start_zt) % period
    out = np.zeros(period)
    for h in range(period):
        mask = (zt_hour >= h) & (zt_hour < h + 1)
        if np.any(mask):
            out[h] = np.nanmean(traces[:, mask])
    return out


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #


def shade_dark_phases(ax, ld_end: int, x_end: float, period: int) -> None:
    """Grey the dark half of each LD cycle, then everything after LD (DD)."""
    start = period // 2
    while start < ld_end:
        ax.axvspan(start, min(start + period // 2, ld_end), color="grey", alpha=0.3)
        start += period
    if x_end > ld_end:
        ax.axvspan(ld_end, x_end, color="grey", alpha=0.3)


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return buf.read()


def show_and_offer(fig, filename: str, key: str) -> None:
    """Render a figure and add a PNG download button underneath it."""
    st.pyplot(fig)
    st.download_button(
        "Download PNG",
        data=fig_to_png_bytes(fig),
        file_name=filename,
        mime="image/png",
        key=key,
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def actogram_grid(
    counts, groups, group_colors, idx_to_label, start_zt, period, ld_end,
    death_bins, apply_deaths: bool, title: str, key: str,
) -> None:
    """Figs 1 & 2: one bar-actogram subplot per mosquito, grouped."""
    order = list(groups.keys())
    x_end = start_zt + max((len(c) for c in counts), default=0)

    def trace_for(i):
        if apply_deaths:
            return apply_death_cut(counts[i], i, death_bins, start_zt)
        return np.asarray(counts[i], dtype=float)

    group_ymax = {
        g: max((np.nanmax(trace_for(i)) for i in idxs), default=1.0)
        for g, idxs in groups.items()
    }
    for g in order:
        st.markdown(f"**{g}**")
        idxs = list(groups[g])
        ncol = min(3, len(idxs)) or 1
        nrow = math.ceil(len(idxs) / ncol)
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(6 * ncol, 3.2 * nrow), squeeze=False
        )
        for k, idx in enumerate(idxs):
            ax = axes[k // ncol][k % ncol]
            y = trace_for(idx)
            x = start_zt + np.arange(len(y))
            ax.bar(x, y, color=group_colors.get(g, BAR_COLOR), width=1)
            _, sub = idx_to_label[idx]
            ax.set_title(f"{g} #{sub}")
            ax.set_ylim(0, group_ymax[g] or 1.0)
            ax.set_xlim(0, x_end)
            ax.axvline(start_zt, color="blue", linestyle="--", linewidth=1.5)
            shade_dark_phases(ax, ld_end, x_end, period)
            ax.set_xticks(np.arange(0, x_end + 1, max(period, 12)))
            ax.set_xlabel("Experimental hour")
            ax.set_ylabel("Distance moved")
        for k in range(len(idxs), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(f"{title} — {g}", fontsize=14)
        fig.tight_layout()
        show_and_offer(fig, f"{key}_{g}.png", key=f"{key}_{g}")


def death_comparison(
    counts, groups, group_colors, start_zt, period, ld_end, death_bins, key: str
) -> None:
    """Fig 3: per group, group-mean actogram cut at death vs cut 24 h before."""
    order = list(groups.keys())
    means_death = {
        g: group_mean_trace(counts, groups[g], death_bins, start_zt, 0.0)
        for g in order
    }
    means_24h = {
        g: group_mean_trace(counts, groups[g], death_bins, start_zt, 24.0)
        for g in order
    }
    gymax = max(
        (np.nanmax(m) for m in list(means_death.values()) + list(means_24h.values())
         if m.size and np.isfinite(np.nanmax(m))),
        default=1.0,
    )
    x_end = start_zt + max((len(c) for c in counts), default=0)

    fig, axes = plt.subplots(len(order), 2, figsize=(18, 3.4 * len(order)), squeeze=False)
    for row, g in enumerate(order):
        for col, (means, tag) in enumerate(
            ((means_death, "cut at death"), (means_24h, "cut 24 h before"))
        ):
            ax = axes[row][col]
            y = means[g]
            x = start_zt + np.arange(len(y))
            ax.bar(x, y, color=group_colors.get(g, BAR_COLOR), width=1)
            ax.set_title(f"{g}\n({tag})")
            ax.set_ylim(0, gymax or 1.0)
            ax.set_xlim(0, x_end)
            ax.axvline(start_zt, color="blue", linestyle="--", linewidth=1.5)
            shade_dark_phases(ax, ld_end, x_end, period)
            ax.set_xticks(np.arange(0, x_end + 1, max(period, 12)))
            ax.set_xlabel("Experimental hour")
            ax.set_ylabel("Distance moved")
    fig.suptitle("Death comparison (cut at death vs 24 h before)", fontsize=16)
    fig.tight_layout()
    show_and_offer(fig, "fig3_death_comparison.png", key=key)
    return means_death  # reused by Fig 7


def full_period_bar(
    counts, groups, group_colors, start_zt, period, lo, hi, ld_end,
    title: str, xlabel: str, key: str, means_override=None,
) -> None:
    """
    Figs 4 & 7: per group, the group-mean trace over a window, drawn as bars
    on the experimental-hour axis with LD/DD shading.
    """
    order = list(groups.keys())
    ncol = min(2, len(order)) or 1
    nrow = math.ceil(len(order) / ncol)
    x_end = start_zt + max((len(c) for c in counts), default=0)

    # y range across all groups
    def mean_for(g):
        if means_override is not None:
            return means_override[g]
        traces = np.array([np.asarray(counts[i], dtype=float) for i in groups[g]])
        return np.nanmean(traces, axis=0)

    means = {g: mean_for(g) for g in order}
    gymax = max((np.nanmax(m) for m in means.values() if m.size), default=1.0)

    fig, axes = plt.subplots(nrow, ncol, figsize=(11 * ncol, 3.8 * nrow), squeeze=False)
    for k, g in enumerate(order):
        ax = axes[k // ncol][k % ncol]
        y_full = means[g]
        x_full = start_zt + np.arange(len(y_full))
        hi_eff = len(y_full) if hi is None else hi
        mask = (np.arange(len(y_full)) >= lo) & (np.arange(len(y_full)) < hi_eff)
        ax.bar(x_full[mask], np.asarray(y_full)[mask],
               color=group_colors.get(g, BAR_COLOR), width=1)
        ax.set_title(g)
        ax.set_ylim(0, gymax or 1.0)
        left = start_zt + lo if lo > 0 else 0
        right = start_zt + hi_eff
        ax.set_xlim(left, right)
        shade_dark_phases(ax, ld_end, x_end, period)
        ax.set_xticks(np.arange(math.floor(left), right + 1, max(period, 12)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Distance moved")
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    show_and_offer(fig, f"{key}.png", key=key)


def folded_bar(
    counts, groups, group_colors, start_zt, period, lo, hi, title: str, key: str
) -> None:
    """Figs 5 & 8: 24 h-folded activity bars, one subplot per group."""
    order = list(groups.keys())
    ncol = min(2, len(order)) or 1
    nrow = math.ceil(len(order) / ncol)
    folded = {g: fold_mean_bar(counts, groups[g], start_zt, period, lo, hi) for g in order}
    gymax = max((f.max() for f in folded.values() if f.size), default=1.0)

    fig, axes = plt.subplots(nrow, ncol, figsize=(10 * ncol, 3.8 * nrow), squeeze=False)
    for k, g in enumerate(order):
        ax = axes[k // ncol][k % ncol]
        ax.bar(np.arange(period), folded[g], color=group_colors.get(g, BAR_COLOR))
        ax.set_title(f"{g} (24 h-folded)")
        ax.set_xlim(0, period)
        ax.set_ylim(0, gymax or 1.0)
        ax.set_xticks(np.arange(0, period + 1, max(period // 4, 1)))
        ax.axvspan(period // 2, period, color="grey", alpha=0.3)
        ax.set_xlabel("ZT (hours)")
        ax.set_ylabel("Activity")
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    show_and_offer(fig, f"{key}.png", key=key)


def folded_line(
    counts, groups, group_colors, start_zt, period, lo, hi, title: str, key: str
) -> None:
    """Figs 6 & 9: mean +/- SD 24 h-folded line plot, one subplot per group."""
    order = list(groups.keys())
    folded = {
        g: fold_24h(counts, groups[g], start_zt, period, lo, hi, {}) for g in order
    }
    gymax = 1.0
    for arr in folded.values():
        if arr.size:
            top = np.nanmax(np.nanmean(arr, axis=0) + np.nanstd(arr, axis=0))
            if np.isfinite(top):
                gymax = max(gymax, top)

    ncol = min(2, len(order)) or 1
    nrow = math.ceil(len(order) / ncol)
    zt = np.arange(period)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9 * ncol, 4 * nrow), squeeze=False)
    for k, g in enumerate(order):
        ax = axes[k // ncol][k % ncol]
        arr = folded[g]
        if arr.size:
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            c = group_colors.get(g, BAR_COLOR)
            ax.plot(zt, mean, color=c, linewidth=2, label="Mean")
            ax.fill_between(zt, mean - std, mean + std, color=c, alpha=0.3, label="±1 SD")
        ax.axvspan(period // 2, period, color="grey", alpha=0.3)
        ax.set_title(g)
        ax.set_xlim(0, period - 1)
        ax.set_ylim(0, gymax)
        ax.set_xticks(np.arange(0, period, max(2, period // 12)))
        ax.set_xlabel("ZT")
        ax.set_ylabel("Distance moved")
        ax.legend(loc="upper right", fontsize=8)
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    show_and_offer(fig, f"{key}.png", key=key)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #


def find_default_csv() -> str:
    """Default to an experiment-27 activity CSV if we can find one; else blank."""
    if EXPERIMENTS_DIR.exists():
        hits = sorted(EXPERIMENTS_DIR.glob("27 *box1/*activity*.csv"))
        if hits:
            return str(hits[-1])  # prefer the highest-numbered (…_activity_2.csv)
        any_hits = sorted(EXPERIMENTS_DIR.glob("**/*activity*.csv"))
        if any_hits:
            return str(any_hits[0])
    return ""


def main() -> None:
    st.set_page_config(page_title="Mosquito Activity Explorer", layout="wide")
    st.title("Mosquito Activity Explorer")
    st.caption(
        "Reproduces every figure from the analyze_activity_model notebook "
        "(Sections A/B/C, Figures 1–9) with your own CSV and parameters — no "
        "notebook editing required."
    )

    # ---- Sidebar: data source & numeric parameters ---- #
    with st.sidebar:
        st.header("1. Data source")
        uploaded = st.file_uploader("Upload activity CSV", type=["csv"])
        default_path = find_default_csv()
        csv_path = st.text_input(
            "…or path to a CSV on disk",
            value=default_path,
            help="Columns should be frame, mosquito_0, mosquito_1, …",
        )

        st.header("2. Timing parameters")
        start_zt = st.number_input(
            "ZT offset (ZT of first bin)", value=9.54, step=0.01, format="%.2f"
        )
        bin_size = st.number_input(
            "Bin size (frames per bin)", value=60, min_value=1, step=1
        )
        period = st.number_input("Circadian period (h)", value=24, min_value=1, step=1)
        ld_end = st.number_input(
            "LD → DD switch (experimental hour)", value=96, min_value=0, step=1
        )

    # ---- Load data ---- #
    source = uploaded if uploaded is not None else (csv_path or None)
    if source is None:
        st.info("Upload a CSV or enter a path in the sidebar to begin.")
        st.stop()

    try:
        data = load_activity(source)
    except Exception as exc:  # noqa: BLE001 - surface any read error to the user
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

    counts = build_counts(data, int(bin_size))
    n_mosq = len(counts)
    trace_len = max((len(c) for c in counts), default=0)
    label = source if isinstance(source, str) else getattr(source, "name", "uploaded file")
    st.success(
        f"Loaded **{n_mosq} mosquitoes · {trace_len} bins** "
        f"(≈ {trace_len} h at bin size {int(bin_size)}) from `{label}`."
    )

    period_i = int(period)
    ld_end_i = int(ld_end)

    # ---- Mosquito-kind layout ---- #
    st.subheader("Mosquito-kind layout")
    st.caption(
        "Each row = one kind/group. `start`/`end` are row indices into the CSV "
        "(end is exclusive). Edit names, ranges, and colors freely."
    )
    groups_df = st.data_editor(
        DEFAULT_GROUPS,
        num_rows="dynamic",
        use_container_width=True,
        key="groups_editor",
        column_config={
            "name": st.column_config.TextColumn("Group name"),
            "start": st.column_config.NumberColumn("Start idx", min_value=0, step=1),
            "end": st.column_config.NumberColumn("End idx (excl.)", min_value=0, step=1),
            "color": st.column_config.TextColumn("Color (hex)"),
        },
    )
    groups = groups_from_editor(groups_df, n_mosq)
    group_colors = colors_from_editor(groups_df)
    if not groups:
        st.warning("Define at least one valid group to see plots.")
        st.stop()
    idx_to_label = build_idx_to_label(groups)

    # ---- Death calls (optional) ---- #
    with st.expander("Death calls (optional)"):
        st.caption(
            "Everything after death is set to NaN. `mosquito_num` is 1-based "
            "within the group; `death_frame` is a raw frame number."
        )
        deaths_df = st.data_editor(
            DEFAULT_DEATHS,
            num_rows="dynamic",
            use_container_width=True,
            key="deaths_editor",
            column_config={
                "group": st.column_config.TextColumn("Group"),
                "mosquito_num": st.column_config.NumberColumn(
                    "Mosquito # (in group)", min_value=1, step=1
                ),
                "death_frame": st.column_config.NumberColumn(
                    "Death frame", min_value=0, step=1
                ),
            },
        )
    death_bins = death_bin_by_idx(deaths_df, groups, int(bin_size))

    # ---- Sections A / B / C ---- #
    sec_a, sec_b, sec_c = st.tabs(
        ["Section A — General", "Section B — LD", "Section C — DD"]
    )

    with sec_a:
        st.markdown("### Fig 1 — Individual actograms (pre-death-cut)")
        actogram_grid(
            counts, groups, group_colors, idx_to_label, start_zt, period_i, ld_end_i,
            death_bins, apply_deaths=False,
            title="Individual actograms (pre-death-cut)", key="fig1",
        )
        st.divider()
        st.markdown("### Fig 2 — Individual actograms (death-cut)")
        actogram_grid(
            counts, groups, group_colors, idx_to_label, start_zt, period_i, ld_end_i,
            death_bins, apply_deaths=True,
            title="Individual actograms (death-cut)", key="fig2",
        )
        st.divider()
        st.markdown("### Fig 3 — Death comparison (cut at death vs 24 h before)")
        means_death = death_comparison(
            counts, groups, group_colors, start_zt, period_i, ld_end_i,
            death_bins, key="fig3",
        )

    with sec_b:
        st.markdown("### Fig 4 — Full LD period, averaged across mosquitoes")
        full_period_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, ld_end=ld_end_i,
            title="Full LD period, averaged across mosquitoes",
            xlabel="ZT / experimental hour", key="fig4",
        )
        st.divider()
        st.markdown("### Fig 5 — LD, averaged across 24 h periods")
        folded_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, title="LD, 24 h-folded", key="fig5",
        )
        st.divider()
        st.markdown("### Fig 6 — LD mean ± SD line plot")
        folded_line(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, title="LD (24 h-folded) mean ± SD", key="fig6",
        )

    with sec_c:
        st.markdown("### Fig 7 — Full DD period, averaged across mosquitoes")
        full_period_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, ld_end=ld_end_i,
            title="Full DD period, averaged across mosquitoes",
            xlabel="Experimental hour", key="fig7", means_override=means_death,
        )
        st.divider()
        st.markdown("### Fig 8 — DD, averaged across 24 h periods")
        folded_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, title="DD, 24 h-folded", key="fig8",
        )
        st.divider()
        st.markdown("### Fig 9 — DD mean ± SD line plot")
        folded_line(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, title="DD (24 h-folded) mean ± SD", key="fig9",
        )


if __name__ == "__main__":
    main()
