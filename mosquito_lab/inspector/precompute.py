"""Build the detection cache for an experiment.

Runs YOLO once over well crops and stores every candidate box (down to a low
confidence floor) so the inspector GUI can compare thresholds instantly.

Examples
--------
List discovered experiments:
    python mosquito_lab/inspector/precompute.py --list

Cache every 10th frame of a discovered experiment (fast, good for QA browsing):
    python mosquito_lab/inspector/precompute.py --experiment "28" --stride 10

Cache a fully-custom experiment:
    python mosquito_lab/inspector/precompute.py \
        --images "/path/raw_images" \
        --labels "/path/exp28_labels.csv" \
        --model  "/path/uninf_det_v0.pt" \
        --stride 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mosquito_lab.inspector import inspector_core as core


def _resolve_experiment(args) -> core.Experiment:
    if args.images and args.labels and args.model:
        name = args.name or "custom"
        exp = core.Experiment(
            name=name,
            image_folder=args.images,
            label_file=args.labels,
            model_path=args.model,
        )
        if args.cache_dir:
            exp.cache_dir = args.cache_dir
        return exp

    discovered = core.discover_experiments()
    if not discovered:
        sys.exit("No experiments discovered. Pass --images/--labels/--model explicitly.")

    if not args.experiment:
        sys.exit("Specify --experiment <substring> or pass --images/--labels/--model.")

    matches = [e for e in discovered if args.experiment.lower() in e.name.lower()]
    if not matches:
        sys.exit(
            f"No experiment matched {args.experiment!r}. Available: "
            + ", ".join(e.name for e in discovered)
        )
    if len(matches) > 1:
        sys.exit(
            f"{args.experiment!r} is ambiguous: " + ", ".join(e.name for e in matches)
        )
    exp = matches[0]
    if args.cache_dir:
        exp.cache_dir = args.cache_dir
    return exp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List discovered experiments and exit")
    parser.add_argument("--experiment", help="Substring of a discovered experiment name")
    parser.add_argument("--images", help="Raw images folder (custom experiment)")
    parser.add_argument("--labels", help="MakeSense labels CSV (custom experiment)")
    parser.add_argument("--model", help="YOLO .pt model (custom experiment)")
    parser.add_argument("--name", help="Name for a custom experiment")
    parser.add_argument("--cache-dir", help="Override cache directory")
    parser.add_argument("--conf-floor", type=float, default=core.DEFAULT_CONF_FLOOR,
                        help=f"Lowest confidence to store (default {core.DEFAULT_CONF_FLOOR})")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (default 1)")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap number of processed frames")
    args = parser.parse_args()

    if args.list:
        discovered = core.discover_experiments()
        if not discovered:
            print("No experiments discovered.")
            return
        print("Discovered experiments:\n")
        for e in discovered:
            checks = e.exists_check()
            n_frames = len(core.list_frames(e.image_folder)) if checks["image_folder"] else 0
            cached = "cached" if checks["cache"] else "no cache"
            print(f"  {e.name}")
            print(f"      images : {e.image_folder}  ({n_frames} frames)")
            print(f"      labels : {e.label_file}")
            print(f"      model  : {e.model_path}")
            print(f"      status : {cached}\n")
        return

    exp = _resolve_experiment(args)
    checks = exp.exists_check()
    missing = [k for k in ("image_folder", "label_file", "model_path") if not checks[k]]
    if missing:
        sys.exit(f"Missing required paths for {exp.name}: {missing}")

    wells = core.load_wells(exp.label_file)
    frames = core.list_frames(exp.image_folder)
    n_selected = len(frames[:: args.stride])
    if args.max_frames is not None:
        n_selected = min(n_selected, args.max_frames)

    print(f"Experiment : {exp.name}")
    print(f"Wells      : {len(wells)}  ({sum(1 for w in wells if w.degenerate)} degenerate)")
    print(f"Frames     : {len(frames)} total, processing {n_selected} (stride {args.stride})")
    print(f"Conf floor : {args.conf_floor}")
    print(f"Cache      : {exp.cache_file}")
    print("Running YOLO (CPU)...\n")

    start = time.time()
    last = [start]

    def progress(done: int, total: int, name: str) -> None:
        now = time.time()
        if done == total or now - last[0] >= 2.0:
            last[0] = now
            rate = done / max(now - start, 1e-6)
            eta = (total - done) / rate if rate else 0
            print(f"  {done}/{total}  ({rate:.1f} frame/s, ETA {eta:5.0f}s)  {name}", flush=True)

    df = core.build_cache(
        exp,
        conf_floor=args.conf_floor,
        stride=args.stride,
        max_frames=args.max_frames,
        progress=progress,
    )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Stored {len(df)} candidate boxes.")
    print(f"Cache written to: {exp.cache_file}")
    print("\nLaunch the inspector with:")
    print("  streamlit run mosquito_lab/inspector/app.py")


if __name__ == "__main__":
    main()
