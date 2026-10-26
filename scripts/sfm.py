"""COLMAP SfM wrapper: frames dir -> undistorted PINHOLE model (images/ + sparse/)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLMAP = ROOT / "tools" / "colmap" / "bin" / "colmap.exe"


def _run(*args):
    print("colmap", *args[:1], flush=True)
    subprocess.run([str(COLMAP), *map(str, args)], check=True)


def run(frames_dir: Path, work: Path, matcher: str = "exhaustive"):
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    sparse = work / "sparse_raw"
    sparse.mkdir()
    try:
        _run("feature_extractor", "--database_path", db, "--image_path", frames_dir,
             "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "SIMPLE_RADIAL")
        _run(f"{matcher}_matcher", "--database_path", db)
    except subprocess.CalledProcessError:
        # e.g. the COLMAP CUDA build lacks kernels for a brand-new GPU architecture -> use CPU SIFT
        print("COLMAP GPU step failed; retrying on CPU (slower)", flush=True)
        if db.exists():
            db.unlink()
        _run("feature_extractor", "--database_path", db, "--image_path", frames_dir,
             "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "SIMPLE_RADIAL",
             "--FeatureExtraction.use_gpu", 0)
        _run(f"{matcher}_matcher", "--database_path", db, "--FeatureMatching.use_gpu", 0)
    _run("mapper", "--database_path", db, "--image_path", frames_dir, "--output_path", sparse)
    models = sorted(sparse.iterdir())
    if not models:
        raise SystemExit("COLMAP produced no model")
    # pick the model with the most files / images (largest)
    best = max(models, key=lambda m: sum(f.stat().st_size for f in m.iterdir()))
    _run("image_undistorter", "--image_path", frames_dir, "--input_path", best, "--output_path", work / "undist",
         "--output_type", "COLMAP")
    shutil.move(str(work / "undist" / "images"), str(work / "images"))
    shutil.move(str(work / "undist" / "sparse"), str(work / "sparse"))
    shutil.rmtree(work / "undist", ignore_errors=True)
