"""Combine several experiments onto one shared ZT timeline.

Mirrors the ``[19, 29-32]`` / ``[27-28]`` combined notebooks:

- Every mosquito keeps its **true ZT start** — an experiment recorded from
  ``start_zt`` begins at hour ~``start_zt`` on the shared grid and the hours
  before that stay ``NaN`` (cut off), rather than being re-anchored to 0.
- Mosquitoes are regrouped by **kind** across experiments (``TYPE_ORDER``), so
  the same genotype/sex lines up regardless of its column order in each CSV.
- **Data gaps** (e.g. a camera outage) are pushed later by ``gap`` hours and
  come back as ``NaN`` so the missing window shows up as empty time.
- Each experiment carries its own **LD→DD boundary** (``ld_end`` as an absolute
  ZT hour, e.g. 72 vs 96), so mixed light schedules don't get blended.

This module is pure (numpy/pandas only) so it can be unit-tested and reused by
the Streamlit UI in ``combine_ui.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Regrouping order — same as the combined notebooks.
TYPE_ORDER = ["Female KO", "Male KO", "Female sg (WT)", "Male sg (WT)"]

DEFAULT_TYPE_COLORS = {
    "Female sg (WT)": "#e63946",
    "Male sg (WT)": "#457b9d",
    "Female KO": "#f4a261",
    "Male KO": "#2a9d8f",
}


@dataclass
class ExperimentSpec:
    """One experiment's aligned inputs.

    Attributes
    ----------
    name:
        Label shown in figures / summaries (usually the CSV file name).
    counts:
        Per-mosquito binned traces indexed from this experiment's own recording
        start (list of 1-D arrays; e.g. one value per hour).
    start_zt:
        ZT hour of the first recorded bin.
    ld_end:
        LD→DD switch as an **absolute ZT hour** on the shared grid (72, 96, ...).
    groups:
        ``kind -> [local column indices]`` for this experiment's CSV.
    deaths:
        ``local column index -> death bin`` (bin index from recording start,
        i.e. ``death_frame // bin_size``). Optional.
    pause_hour / gap:
        Recording bin index of the last hour before a data gap, and the gap
        length in hours. Bins at/after ``pause_hour`` are shifted later by
        ``gap`` so the outage becomes empty (NaN) time.
    colors:
        Optional ``kind -> hex color`` overrides.
    """

    name: str
    counts: list
    start_zt: float
    ld_end: int
    groups: dict
    deaths: dict = field(default_factory=dict)
    pause_hour: int | None = None
    gap: float = 0.0
    colors: dict = field(default_factory=dict)


@dataclass
class CombinedData:
    """Result of :func:`combine_experiments`, ready for the plot helpers."""

    counts: list            # ZT-aligned full traces (NaN before start / in gaps)
    counts_ld: list         # LD-only view (DD hours set to NaN)
    counts_dd: list         # DD-only view (LD hours set to NaN)
    groups: dict            # kind -> [combined indices], ordered by TYPE_ORDER
    group_colors: dict      # kind -> hex color
    death_bins: dict        # combined index -> death cutoff on the ZT grid
    ld_end_by_idx: dict     # combined index -> that mosquito's LD→DD ZT hour
    labels: dict            # combined index -> (kind, within-kind number)
    exp_by_idx: dict        # combined index -> experiment name
    n_mosq: int
    T: int                  # shared timeline length (hours)
    period: int
    ld_max: int             # max LD→DD boundary across experiments
    dd_min: int             # min LD→DD boundary across experiments
    per_type_counts: dict   # kind -> {experiment name: n mosquitoes}


def exp_hour(i: float, pause_hour: int | None, gap: float) -> float:
    """Recording bin index -> true experiment hour (gap-aware)."""
    if pause_hour is not None and i >= pause_hour:
        return i + gap
    return i


def align_to_zt(
    counts,
    start_zt: float,
    T: int,
    pause_hour: int | None = None,
    gap: float = 0.0,
) -> np.ndarray:
    """Re-anchor an hourly trace onto the shared ZT grid.

    Bin ``i`` sits at ZT-anchored hour ``start_zt + exp_hour(i)``; its value is
    split across the two neighbouring integer ZT hours by linear interpolation.
    Hours before ``start_zt`` and the gap window receive no weight -> ``NaN``.
    """
    counts = np.asarray(counts, dtype=float)
    activity = np.zeros(T)
    weight = np.zeros(T)
    for i, val in enumerate(counts):
        if not np.isfinite(val):
            continue
        pos = start_zt + exp_hour(i, pause_hour, gap)
        low = int(math.floor(pos))
        high = low + 1
        frac = pos - low
        if 0 <= low < T:
            activity[low] += val * (1 - frac)
            weight[low] += 1 - frac
        if 0 <= high < T:
            activity[high] += val * frac
            weight[high] += frac
    return np.divide(activity, weight, out=np.full(T, np.nan), where=weight > 0)


def combine_experiments(specs: list[ExperimentSpec], period: int = 24) -> CombinedData:
    """Combine experiments onto one ZT grid, regrouped by mosquito kind."""
    if not specs:
        raise ValueError("No experiments to combine.")

    # Length of the shared timeline: enough to hold every experiment once
    # shifted forward by its start_zt (and any gap).
    T = 0
    for s in specs:
        for row in s.counts:
            last = len(row) - 1
            if last < 0:
                continue
            pos = s.start_zt + exp_hour(last, s.pause_hour, s.gap)
            T = max(T, int(math.ceil(pos)) + 2)
    if T <= 0:
        raise ValueError("Experiments contain no data.")

    records: list[dict] = []
    for order_i, s in enumerate(specs):
        for gtype, local_idxs in s.groups.items():
            for num, local in enumerate(local_idxs, start=1):
                if local < 0 or local >= len(s.counts):
                    continue
                aligned = align_to_zt(
                    s.counts[local], s.start_zt, T, s.pause_hour, s.gap
                )
                death_grid = None
                if local in s.deaths:
                    d = s.deaths[local]
                    death_grid = s.start_zt + exp_hour(d, s.pause_hour, s.gap)
                records.append(
                    {
                        "type": gtype,
                        "exp": s.name,
                        "num": num,
                        "order": order_i,
                        "aligned": aligned,
                        "death_grid": death_grid,
                        "ld_end": int(s.ld_end),
                        "color": (
                            s.colors.get(gtype)
                            or DEFAULT_TYPE_COLORS.get(gtype, "#ef3c26")
                        ),
                    }
                )
    if not records:
        raise ValueError("No mosquitoes matched the group layout.")

    # Line the same TYPE up across experiments; keep input order within a kind.
    def sort_key(r: dict) -> tuple:
        t = r["type"]
        ti = TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER)
        return (ti, r["order"], r["num"])

    records.sort(key=sort_key)

    counts: list = []
    groups: dict = {}
    group_colors: dict = {}
    death_bins: dict = {}
    ld_end_by_idx: dict = {}
    labels: dict = {}
    exp_by_idx: dict = {}
    per_type_counts: dict = {}
    type_running: dict = {}

    for idx, r in enumerate(records):
        counts.append(r["aligned"])
        t = r["type"]
        groups.setdefault(t, []).append(idx)
        group_colors.setdefault(t, r["color"])
        n = type_running.get(t, 0) + 1
        type_running[t] = n
        labels[idx] = (t, n)
        exp_by_idx[idx] = r["exp"]
        ld_end_by_idx[idx] = r["ld_end"]
        if r["death_grid"] is not None:
            death_bins[idx] = r["death_grid"]
        per_type_counts.setdefault(t, {}).setdefault(r["exp"], 0)
        per_type_counts[t][r["exp"]] += 1

    # Order groups by TYPE_ORDER (then any extras).
    ordered: dict = {t: groups[t] for t in TYPE_ORDER if t in groups}
    for t, v in groups.items():
        ordered.setdefault(t, v)
    groups = ordered

    # LD-only / DD-only masked views (each mosquito uses its own boundary).
    counts_ld: list = []
    counts_dd: list = []
    for idx, tr in enumerate(counts):
        le = ld_end_by_idx[idx]
        ld = tr.copy()
        ld[le:] = np.nan
        dd = tr.copy()
        dd[:le] = np.nan
        counts_ld.append(ld)
        counts_dd.append(dd)

    ld_max = max(ld_end_by_idx.values())
    dd_min = min(ld_end_by_idx.values())

    return CombinedData(
        counts=counts,
        counts_ld=counts_ld,
        counts_dd=counts_dd,
        groups=groups,
        group_colors=group_colors,
        death_bins=death_bins,
        ld_end_by_idx=ld_end_by_idx,
        labels=labels,
        exp_by_idx=exp_by_idx,
        n_mosq=len(counts),
        T=T,
        period=period,
        ld_max=int(ld_max),
        dd_min=int(dd_min),
        per_type_counts=per_type_counts,
    )


