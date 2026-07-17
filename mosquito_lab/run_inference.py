#!/usr/bin/env python3
"""Run mosquito inference on a cloud VM from a local image folder or Google Drive link.

Examples
--------
# Local folder (laptop or VM):
python -m mosquito_lab.run_inference \\
  --image-folder "/path/to/raw_images" \\
  --labels "/path/to/exp_labels.csv" \\
  --model "/path/to/uninf_det_v0.pt" \\
  --output-dir ./cloud_runs/run1 \\
  --serve-viewer

# Google Drive (share links must be accessible — "Anyone with the link"):
python -m mosquito_lab.run_inference \\
  --images-drive-url "https://drive.google.com/drive/folders/XXXX" \\
  --labels-drive-url "https://drive.google.com/file/d/YYYY/view" \\
  --model-drive-url "https://drive.google.com/file/d/ZZZZ/view" \\
  --output-dir ./cloud_runs/run1 \\
  --serve-viewer --viewer-port 8502

# Detached on a VM (close laptop safely):
nohup python -m mosquito_lab.run_inference ... --serve-viewer > inference.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from mosquito_lab.data_sources import (
    prepare_work_dir,
    resolve_image_folder,
    resolve_path_or_drive,
)
from mosquito_lab.inference_core import run_inference
from mosquito_lab.status import write_status


def _product_root() -> Path:
    from mosquito_lab.paths import product_root
    return product_root()


def _start_viewer(
    *,
    passcode: str,
    port: int,
    output_dir: Path,
    image_folder: Path,
    label_file: Path,
    model_path: Path,
    host: str,
) -> subprocess.Popen | None:
    viewer_script = Path(__file__).resolve().parent / "lab_app.py"
    env = os.environ.copy()
    env.update(
        {
            "CLOUD_VIEWER_PASSCODE": passcode,
            "CLOUD_VIEWER_OUTPUT_DIR": str(output_dir),
            "CLOUD_VIEWER_IMAGE_FOLDER": str(image_folder),
            "CLOUD_VIEWER_LABEL_FILE": str(label_file),
            "CLOUD_VIEWER_MODEL_PATH": str(model_path),
        }
    )
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(viewer_script),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"[viewer] Starting on http://{host}:{port} (passcode required)")
    return subprocess.Popen(cmd, env=env, cwd=str(_product_root()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cloud mosquito inference from local folder or Google Drive."
    )

    src = p.add_argument_group("image source (pick one)")
    src.add_argument("--image-folder", help="Local folder of frame images")
    src.add_argument(
        "--images-drive-url",
        help="Google Drive folder URL containing frame images",
    )

    assets = p.add_argument_group("labels & model")
    assets.add_argument("--labels", help="Local path to MakeSense labels CSV")
    assets.add_argument("--labels-drive-url", help="Google Drive URL for labels CSV")
    assets.add_argument("--model", help="Local path to YOLO .pt model")
    assets.add_argument("--model-drive-url", help="Google Drive URL for YOLO .pt model")

    run = p.add_argument_group("run options")
    run.add_argument(
        "--work-dir",
        default="./mosquito_lab_work",
        help="Scratch dir for Drive downloads (default: ./mosquito_lab_work)",
    )
    run.add_argument(
        "--output-dir",
        default="./mosquito_lab_output",
        help="Where CSV, manifest, and previews are written",
    )
    run.add_argument("--output-name", default="activity_transposed.csv")
    run.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device (auto picks GPU if available)",
    )
    run.add_argument(
        "--fresh-download",
        action="store_true",
        help="Delete work-dir before downloading from Drive",
    )
    run.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip saving annotated preview images",
    )
    run.add_argument(
        "--preview-every",
        type=int,
        default=50,
        help="Save a preview image every N frames per mosquito",
    )

    viewer = p.add_argument_group("password-protected image viewer")
    viewer.add_argument(
        "--serve-viewer",
        action="store_true",
        help="Start Streamlit viewer to browse/download images per frame",
    )
    viewer.add_argument("--viewer-port", type=int, default=8502)
    viewer.add_argument(
        "--viewer-host",
        default="0.0.0.0",
        help="Bind address (0.0.0.0 for cloud VM access)",
    )
    viewer.add_argument(
        "--passcode",
        default=os.environ.get("CLOUD_VIEWER_PASSCODE", "florawang"),
        help="Viewer login passcode (default: florawang)",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not (args.image_folder or args.images_drive_url):
        print("Error: provide --image-folder or --images-drive-url", file=sys.stderr)
        return 2
    if not (args.labels or args.labels_drive_url):
        print("Error: provide --labels or --labels-drive-url", file=sys.stderr)
        return 2
    if not (args.model or args.model_drive_url):
        print("Error: provide --model or --model-drive-url", file=sys.stderr)
        return 2

    work_dir = prepare_work_dir(args.work_dir, fresh=args.fresh_download)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== RESOLVING INPUTS ==========")
    image_folder = resolve_image_folder(
        args.image_folder,
        args.images_drive_url,
        work_dir,
    )
    label_file = resolve_path_or_drive(
        args.labels,
        args.labels_drive_url,
        work_dir,
        "labels",
        label="labels",
    )
    model_path = resolve_path_or_drive(
        args.model,
        args.model_drive_url,
        work_dir,
        "model",
        label="model",
    )

    print(f"Images: {image_folder}")
    print(f"Labels: {label_file}")
    print(f"Model:  {model_path}")
    print(f"Output: {output_dir}")
    print("======================================\n")

    viewer_proc: subprocess.Popen | None = None
    if args.serve_viewer:
        viewer_proc = _start_viewer(
            passcode=args.passcode,
            port=args.viewer_port,
            output_dir=output_dir,
            image_folder=image_folder,
            label_file=label_file,
            model_path=model_path,
            host=args.viewer_host,
        )
        write_status(
            str(output_dir / "inference_status.json"),
            state="starting",
            viewer_url=f"http://{args.viewer_host}:{args.viewer_port}",
        )

    try:
        result = run_inference(
            image_folder=image_folder,
            label_file=label_file,
            model_path=model_path,
            output_dir=output_dir,
            output_name=args.output_name,
            device=args.device,
            save_previews=not args.no_previews,
            preview_every=args.preview_every,
        )
        if args.serve_viewer:
            viewer_url = f"http://{args.viewer_host}:{args.viewer_port}"
            write_status(
                result["status_file"],
                viewer_url=viewer_url,
            )
            print(f"\nViewer (passcode: {args.passcode}): {viewer_url}")
            print("Leave this process running (or nohup it) so the viewer stays up.")
            if viewer_proc:
                viewer_proc.wait()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        write_status(
            str(output_dir / "inference_status.json"),
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if viewer_proc and viewer_proc.poll() is None:
            viewer_proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
