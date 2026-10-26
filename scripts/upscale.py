"""Frame post-processing ops applied to rendered flythrough frames.

ops: none | bicubic | realesrgan | toon      (scale applies to bicubic / realesrgan)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REALESRGAN = ROOT / "tools" / "realesrgan" / "realesrgan-ncnn-vulkan.exe"


def process_dir(src: Path, dst: Path, op: str, scale: int = 2, model: str = "realesrgan-x4plus"):
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    if op == "realesrgan":
        if not REALESRGAN.exists():
            raise SystemExit(f"Real-ESRGAN binary missing: {REALESRGAN}")
        subprocess.run([str(REALESRGAN), "-i", str(src), "-o", str(dst), "-n", model, "-s", str(scale),
                        "-f", "png"], check=True, cwd=REALESRGAN.parent)
        return
    for f in files:
        img = cv2.imread(str(f))
        if op == "none":
            out = img
        elif op == "bicubic":
            out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        elif op == "toon":
            out = toon(img)
        else:
            raise ValueError(op)
        cv2.imwrite(str(dst / f.name), out)


def toon(img: np.ndarray, levels: int = 6) -> np.ndarray:
    """Deterministic cel-shade-ish stylisation (no baked outlines: those break multi-view consistency)."""
    x = img
    for _ in range(3):
        x = cv2.bilateralFilter(x, 9, 40, 7)
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.round(hsv[..., 2] / 255 * levels) / levels * 255
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
