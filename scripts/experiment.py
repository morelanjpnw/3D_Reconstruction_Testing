"""Experiment orchestrator for Baseline and Cases 1-6.

    python scripts/experiment.py --config configs/poc_3060ti.yaml --cases baseline case1 case2 ...

Everything is cached: a step is skipped when its output exists, so the script can be re-run / resumed.
Layout (per profile):
    extracted_frames/<profile>/baseline_sfm/   undistorted real frames + COLMAP model
    flythroughs/<profile>/<run>/<path>/        raw rendered flythrough frames + cameras.json
    extracted_frames/<profile>/<run>/<path>/   processed (upscaled / stylised) frames used for training
    splats/<profile>/<run>/                    splat.pt, splat.ply, metrics.json, renders/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline as bl  # noqa: E402
import sfm  # noqa: E402
import splatlib as sl  # noqa: E402
import upscale  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# path specs: lateral / vertical (fractions of camera-orbit radius), yaw / pitch in degrees
ORIGINAL = dict(name="orig", lateral=0.0, vertical=0.0, yaw_deg=0.0, pitch_deg=0.0)
ALT_G2 = [dict(name="A", lateral=0.08, vertical=0.0, yaw_deg=6, pitch_deg=0),
          dict(name="B", lateral=-0.08, vertical=0.0, yaw_deg=-6, pitch_deg=0),
          dict(name="C", lateral=0.0, vertical=-0.07, yaw_deg=0, pitch_deg=6),
          dict(name="D", lateral=0.0, vertical=0.07, yaw_deg=0, pitch_deg=-6)]
ALT_G3 = [dict(name="E", lateral=0.04, vertical=0.04, yaw_deg=-4, pitch_deg=3),
          dict(name="F", lateral=-0.04, vertical=-0.04, yaw_deg=4, pitch_deg=-3),
          dict(name="G", lateral=0.12, vertical=0.0, yaw_deg=-10, pitch_deg=0),
          dict(name="H", lateral=-0.12, vertical=0.0, yaw_deg=10, pitch_deg=0)]


class Ctx:
    def __init__(self, cfg):
        self.cfg = cfg
        p = cfg["profile"]
        self.frames_dir = ROOT / "extracted_frames" / p
        self.fly_dir = ROOT / "flythroughs" / p
        self.splat_dir = ROOT / "splats" / p
        self.res_dir = ROOT / "results" / p
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.frames_dir / "baseline_manifest.json"
        self.base_frames = None
        self.xyz = self.rgb = None

    def load_base(self):
        if self.base_frames is None:
            self.base_frames = sl.load_manifest(self.manifest)
            self.xyz, self.rgb = sl.load_points(self.manifest)
        return self.base_frames

    def log(self, msg):
        print(msg, flush=True)
        with open(self.res_dir / "log.txt", "a") as f:
            f.write(msg + "\n")


# ----------------------------------------------------------------------------- baseline
def run_baseline_prep(ctx: Ctx, video: Path):
    if ctx.manifest.exists():
        return
    raw = ctx.frames_dir / "baseline_raw"
    ctx.log(f"[baseline] extracting frames from {video}")
    bl.extract_frames(video, raw, fps=ctx.cfg["fps"], width=None, fmt="png", quality=2, overwrite=True)
    ctx.log("[baseline] COLMAP SfM")
    sfm.run(raw, ctx.frames_dir / "baseline_sfm")
    frames, xyz, rgb = sl.load_colmap(ctx.frames_dir / "baseline_sfm" / "sparse",
                                      ctx.frames_dir / "baseline_sfm" / "images", ctx.cfg["test_every"])
    ctx.log(f"[baseline] registered {len(frames)} frames, {len(xyz)} sparse points")
    sl.save_manifest(ctx.manifest, frames, xyz, rgb)


# ----------------------------------------------------------------------------- generic step
def train_run(ctx: Ctx, run: str, pool_dirs, paths_frames, anchor: bool, seed: int, eval_note=""):
    """Train splat `run` on synthetic frames in pool_dirs (+ real frames if anchor), evaluate on held-out real."""
    out = ctx.splat_dir / run
    if (out / "metrics.json").exists():
        ctx.log(f"[{run}] cached")
        return
    base = ctx.load_base()
    train_frames = []
    for d, pf in zip(pool_dirs, paths_frames):
        for f in pf:
            g = dict(f)
            g["file"] = str(d / f["name"])
            train_frames.append(g)
    if anchor or run.startswith("baseline"):
        for f in base:
            if f["split"] == "train":
                g = dict(f)
                g["is_real"] = True
                train_frames.append(g)
    eval_frames = [f for f in base if f["split"] == "test"]
    ctx.log(f"[{run}] training on {len(train_frames)} frames (anchor={anchor}, seed={seed})")
    stats = sl.train(train_frames + eval_frames, ctx.xyz, ctx.rgb, out, steps=ctx.cfg["steps"],
                     cap=ctx.cfg["cap"], seed=seed,
                     real_weight=ctx.cfg["real_weight"] if anchor else None, log=ctx.log)
    params = sl.load_splat(out)
    m = sl.evaluate(params, base, out / "renders")
    m.update(stats)
    m.update(run=run, seed=seed, anchor=anchor, note=eval_note)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    ctx.log(f"[{run}] PSNR {m['psnr']:.2f} SSIM {m['ssim']:.4f} LPIPS {m['lpips']:.4f} "
            f"splats {m['n_splats']} ply {m['ply_mb']:.0f}MB train {m['train_seconds']:.0f}s")


def make_generation(ctx: Ctx, src_run: str, run: str, path_specs, op: str, anchor: bool, seed: int, note=""):
    """Render flythrough paths from splat `src_run` (or the real baseline when src_run=='baseline'),
    post-process with `op`, train splat `run`."""
    if (ctx.splat_dir / run / "metrics.json").exists():
        ctx.log(f"[{run}] cached")
        return
    base = ctx.load_base()
    train_real = [f for f in base if f["split"] == "train"]
    n = int(len(train_real) * ctx.cfg["path_frame_factor"])
    params = None
    pool_dirs, pool_frames = [], []
    for spec in path_specs:
        cams = sl.make_path(train_real, n, spec["lateral"], spec["vertical"], spec["yaw_deg"], spec["pitch_deg"])
        raw = ctx.fly_dir / run / spec["name"]
        proc = ctx.frames_dir / run / spec["name"]
        if not (raw / cams[-1]["name"]).exists():
            if params is None:
                params = sl.load_splat(ctx.splat_dir / src_run)
            ctx.log(f"[{run}] rendering path {spec['name']} ({n} frames) from {src_run}")
            sl.render_frames(params, cams, raw)
            sl.save_manifest(raw / "cameras.json", cams)
        if op != "none" and not (proc / cams[-1]["name"]).exists():
            ctx.log(f"[{run}] processing path {spec['name']} with {op}")
            upscale.process_dir(raw, proc, op, ctx.cfg["upscale_scale"])
        pool_dirs.append(proc if op != "none" else raw)
        pool_frames.append(cams)
    train_run(ctx, run, pool_dirs, pool_frames, anchor, seed, note)


# ----------------------------------------------------------------------------- cases
def run_case(ctx: Ctx, case: str):
    c, s0 = ctx.cfg, ctx.cfg["seeds"][0]
    up = c["upscale_op"]
    n_alt = c["n_alt_paths"]
    if case == "baseline":
        train_run(ctx, "baseline_gen1", [], [], True, s0, "real frames only")
        if len(c["seeds"]) > 1:
            for s in c["seeds"][1:]:
                train_run(ctx, f"baseline_gen1_seed{s}", [], [], True, s, "variance floor")
    elif case == "case1":
        make_generation(ctx, "baseline_gen1", "case1_gen2", [ORIGINAL], "none", False, s0, "re-render only")
    elif case == "case2":
        make_generation(ctx, "baseline_gen1", "case2_gen2", [ORIGINAL], up, False, s0, "AI upscale")
        make_generation(ctx, "baseline_gen1", "case2b_gen2_bicubic", [ORIGINAL], "bicubic", False, s0,
                        "control: plain 2x resize")
    elif case == "case3":
        for anchor in (False, True):
            tag = "anch" if anchor else "pure"
            prev = "case2_gen2"
            if anchor:
                make_generation(ctx, "baseline_gen1", "case3a_gen2_anch", [ORIGINAL], up, True, s0, "anchored gen2")
                prev = "case3a_gen2_anch"
            for g in range(3, c["n_gens"] + 1):
                name = f"case3{'a' if anchor else ''}_gen{g}_{tag}" if anchor else f"case3_gen{g}_pure"
                make_generation(ctx, prev, name, [ORIGINAL], up, anchor, s0, f"{tag} multi-gen")
                prev = name
    elif case == "case4":
        make_generation(ctx, "baseline_gen1", "case4_gen2", ALT_G2[:n_alt], up, False, s0, "multi-path + AI")
        make_generation(ctx, "baseline_gen1", "case4_noai_gen2", ALT_G2[:n_alt], "none", False, s0,
                        "ablation: multi-path, no AI")
        make_generation(ctx, "baseline_gen1", "case4a_gen2_anch", ALT_G2[:n_alt], up, True, s0,
                        "multi-path + AI + real anchor")
    elif case == "case5":
        make_generation(ctx, "case4_gen2", "case5_gen3", ALT_G3[:n_alt], up, False, s0, "multi-path x multi-gen")
        make_generation(ctx, "case4a_gen2_anch", "case5a_gen3_anch", ALT_G3[:n_alt], up, True, s0,
                        "multi-path x multi-gen, anchored")
    elif case == "case6":
        make_generation(ctx, "baseline_gen1", "case6a_toon_gen2", [ORIGINAL], "toon", False, s0,
                        "deterministic toon stylisation (metrics vs real are NOT meaningful)")
    else:
        custom_cases = c.get("custom_cases", {})
        if case in custom_cases:
            spec = custom_cases[case]
            src_run = spec.get("src_run", "baseline_gen1")
            run_name = spec.get("run_name", f"{case}_gen2")
            op = spec.get("op", up)
            anchor = bool(spec.get("anchor", False))
            note = spec.get("note", "custom case")
            paths_cfg = spec.get("paths", [ORIGINAL])
            make_generation(ctx, src_run, run_name, paths_cfg, op, anchor, s0, note)
        else:
            raise SystemExit(f"unknown case {case}")


def summarize(ctx: Ctx):
    rows = []
    for m in sorted(ctx.splat_dir.glob("*/metrics.json")):
        rows.append(json.loads(m.read_text()))
    rows.sort(key=lambda r: r["run"])
    lines = ["| Run | PSNR | SSIM | LPIPS | Splats | PLY MB | Train frames | Train s | Note |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['n_splats']} | "
                     f"{r['ply_mb']:.0f} | {r['n_train_frames']} | {r['train_seconds']:.0f} | {r['note']} |")
    (ctx.res_dir / "summary.md").write_text("\n".join(lines))
    (ctx.res_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--video", type=Path, help="source video (baseline prep)")
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. --set profile=smoke steps=300 cap=50000")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    ctx = Ctx(cfg)
    if a.video:
        run_baseline_prep(ctx, a.video)
    for case in a.cases:
        run_case(ctx, case)
    if a.summary or a.cases:
        summarize(ctx)
