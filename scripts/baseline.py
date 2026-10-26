#!/usr/bin/env python3
"""Baseline step: Original Video -> Extract Frames (input for Gaussian Splat Gen 1).

Usage:
    python scripts/baseline.py inputs/my_video.mp4 --fps 2
    python scripts/baseline.py inputs/my_video.mp4 --fps 3 --width 1920 --format png

Frames are written to extracted_frames/<case>/ (default case: baseline).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "extracted_frames"


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        try:  # fall back to the binary bundled with imageio-ffmpeg (installed in the project env)
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            sys.exit("ERROR: ffmpeg not found on PATH. Install it (e.g. `winget install Gyan.FFmpeg`).")
    return path


def extract_frames(video: Path, out_dir: Path, fps: float, width: int | None,
                   fmt: str, quality: int, overwrite: bool) -> int:
    if not video.is_file():
        sys.exit(f"ERROR: input video not found: {video}")

    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            sys.exit(f"ERROR: {out_dir} is not empty. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filters = [f"fps={fps}"]
    if width:
        filters.append(f"scale={width}:-2")

    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "warning", "-stats",
           "-i", str(video), "-vf", ",".join(filters), "-vsync", "vfr"]
    if fmt == "jpg":
        cmd += ["-q:v", str(quality)]
    cmd.append(str(out_dir / f"frame_%05d.{fmt}"))

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: ffmpeg failed with code {result.returncode}")

    count = len(list(out_dir.glob(f"*.{fmt}")))
    print(f"Extracted {count} frames -> {out_dir}")
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frames from a video with FFmpeg.")
    p.add_argument("video", type=Path, help="Path to the original video")
    p.add_argument("--case", default="baseline", help="Output subfolder name (default: baseline)")
    p.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract (default: 2)")
    p.add_argument("--width", type=int, default=None, help="Optional resize width (keeps aspect ratio)")
    p.add_argument("--format", choices=["png", "jpg"], default="png", dest="fmt")
    p.add_argument("--quality", type=int, default=2, help="JPG quality, 2 (best) to 31 (worst)")
    p.add_argument("--overwrite", action="store_true", help="Replace existing output folder contents")
    args = p.parse_args()

    extract_frames(args.video, FRAMES_DIR / args.case, args.fps, args.width,
                   args.fmt, args.quality, args.overwrite)


if __name__ == "__main__":
    main()
