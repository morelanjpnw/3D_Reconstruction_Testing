"""Download the Tanks&Temples 'truck' scene (from the 3DGS authors' tandt_db.zip) and encode it as inputs/truck.mp4.

Research-licensed dataset: for experiments only. Replace inputs/truck.mp4 with your own capture for real work.
Usage: python scripts/prepare_data.py
"""
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import baseline

ROOT = Path(__file__).resolve().parent.parent
URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
RAW = ROOT / "inputs" / "_raw"
OUT = ROOT / "inputs" / "truck.mp4"


def main():
    if OUT.exists():
        print(f"{OUT} already exists; nothing to do")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "tandt_db.zip"
    if not zpath.exists():
        print("Downloading tandt_db.zip (~650 MB)...")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if m.startswith("tandt/truck/images/") and m.endswith(".jpg")]
        z.extractall(RAW, members)
    print(f"Extracted {len(members)} images; encoding video")
    subprocess.run([baseline.find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-framerate", "10",
                    "-i", str(RAW / "tandt" / "truck" / "images" / "%06d.jpg"),
                    "-vf", "crop=978:546:0:0", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(OUT)],
                   check=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
