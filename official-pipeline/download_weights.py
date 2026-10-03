"""Populate the weights volume. Run once, on a pod with the volume mounted -- not in the image.

    python download_weights.py                 # base checkpoint, ~83 GB
    python download_weights.py --avatar-1.5    # and the audio-driven avatar model, ~75 GB more

The base checkpoint is 83 GB, which is why it is not baked into the image: a worker starting on a machine that
has never pulled it would spend most of an hour doing so before serving its first request, and every rebuild
would repeat the upload. On a network volume it is downloaded once and mounted by every worker in that
datacentre, and the model can be replaced without touching the image.

Uses the Python API rather than the command line on purpose: `huggingface-cli` was renamed to `hf` in
huggingface_hub 1.0, so a script calling it works until the day the image is rebuilt and then exits 127.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

TARGET = Path(os.environ.get("WEIGHTS_DIR", "/runpod-volume/weights"))

MODELS = {
    "base": ("meituan-longcat/LongCat-Video", "LongCat-Video", 83),
    "avatar-1.5": ("meituan-longcat/LongCat-Video-Avatar-1.5", "LongCat-Video-Avatar-1.5", 75),
}

# What the pipeline loads by subfolder. Checked after the download, because a snapshot that was interrupted
# leaves a directory that looks present and fails on the first request instead of here.
REQUIRED_SUBFOLDERS = ["tokenizer", "text_encoder", "vae", "scheduler", "dit"]


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def fetch(repo_id: str, name: str, expected_gb: int) -> Path:
    destination = TARGET / name
    available = free_gb(TARGET)
    if available < expected_gb * 1.1:
        sys.exit(f"{destination} needs about {expected_gb} GB and the volume has {available:.0f} GB free")

    print(f"downloading {repo_id} -> {destination} (~{expected_gb} GB)", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        # Several connections, because this is tens of gigabytes and a single stream is the slow part rather
        # than the disk. Resumes by default, so an interrupted run is re-run rather than restarted.
        max_workers=8,
    )

    missing = [s for s in REQUIRED_SUBFOLDERS if not (destination / s).is_dir()]
    if missing:
        sys.exit(f"{destination} is missing {', '.join(missing)} -- the download did not finish")

    size = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file()) / 1e9
    print(f"ok: {destination} holds {size:.1f} GB", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar-1.5", dest="avatar", action="store_true", help="also fetch the avatar model")
    parser.add_argument("--only", choices=sorted(MODELS), help="fetch just one")
    args = parser.parse_args()

    wanted = [args.only] if args.only else ["base"] + (["avatar-1.5"] if args.avatar else [])
    for name in wanted:
        repo_id, folder, gb = MODELS[name]
        fetch(repo_id, folder, gb)

    print(f"\nvolume now holds: {', '.join(sorted(p.name for p in TARGET.iterdir() if p.is_dir()))}")
    print(f"free: {free_gb(TARGET):.0f} GB")


if __name__ == "__main__":
    main()
