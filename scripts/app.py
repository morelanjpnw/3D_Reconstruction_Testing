"""Streamlit Web Dashboard for 3D Reconstruction & Gaussian Splat Iteration Testing.

Launch via:
    tools\\env.local.cmd streamlit run scripts\\app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
INPUTS_DIR = ROOT / "inputs"
SPLATS_DIR = ROOT / "splats"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

for d in (CONFIGS_DIR, INPUTS_DIR, SPLATS_DIR, RESULTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="SplatLab – 3DGS Iteration Pipeline", layout="wide")
st.title("SplatLab: 3D Gaussian Splatting Generative Iteration Pipeline")


# ----------------------------------------------------------------------------- helpers
def list_configs() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def list_videos() -> list[Path]:
    vids = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
        vids.extend(INPUTS_DIR.glob(ext))
    return sorted(vids)


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def save_cfg(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def get_env_wrapper() -> str:
    local = ROOT / "tools" / "env.local.cmd"
    if local.exists():
        return str(local)
    return str(ROOT / "tools" / "env.cmd")


def get_all_cases(cfg: dict) -> list[str]:
    base = ["baseline", "case1", "case2", "case3", "case4", "case5", "case6"]
    custom = list((cfg.get("custom_cases") or {}).keys())
    return base + custom


def load_summary_df(profile: str) -> pd.DataFrame:
    rows = []
    p_dir = SPLATS_DIR / profile
    if p_dir.exists():
        for m in sorted(p_dir.glob("*/metrics.json")):
            try:
                rows.append(json.loads(m.read_text()))
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["run", "psnr", "ssim", "lpips", "n_splats", "ply_mb", "n_train_frames", "train_seconds", "anchor", "seed", "note"]
    existing = [c for c in cols if c in df.columns]
    return df[existing].sort_values("run")


# ----------------------------------------------------------------------------- sidebar: active profile & video
st.sidebar.header("Active Workspace")
cfg_files = list_configs()
cfg_names = [f.name for f in cfg_files]
default_idx = cfg_names.index("rtx6000_quick.yaml") if "rtx6000_quick.yaml" in cfg_names else 0
selected_cfg_name = st.sidebar.selectbox("Configuration Profile", cfg_names, index=default_idx)
selected_cfg_path = CONFIGS_DIR / selected_cfg_name
cfg = load_cfg(selected_cfg_path)
profile = cfg.get("profile", selected_cfg_path.stem)

st.sidebar.markdown(f"**Profile ID:** `{profile}`")
st.sidebar.markdown(f"**Steps:** `{cfg.get('steps', 7000)}` | **Cap:** `{cfg.get('cap', 400000):,}`")
st.sidebar.markdown(f"**Upscaler:** `{cfg.get('upscale_op', 'realesrgan')}` ({cfg.get('upscale_scale', 2)}x)")

if st.sidebar.button("Refresh UI"):
    st.rerun()

# ----------------------------------------------------------------------------- tabs
tab_run, tab_cfg, tab_cases, tab_res, tab_gal = st.tabs([
    "🚀 Run & Monitor",
    "⚙️ Test Settings & Videos",
    "🧪 Custom Case Builder",
    "📊 Results & Leaderboard",
    "🖼️ Visual Comparison & PLY Export",
])

# === TAB 1: RUN & MONITOR ===
with tab_run:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Launch Experiment Run")
        videos = list_videos()
        vid_options = ["(Skip video prep / use existing SfM manifest)"] + [v.name for v in videos]
        default_vid = 1 if videos else 0
        chosen_vid = st.selectbox("Input Video", vid_options, index=default_vid)

        if not videos:
            st.info("No videos in `inputs/` yet. Click below to download the benchmark `truck.mp4` scene, or upload a video in the Settings tab.")
            if st.button("Download & Encode Benchmark Scene (truck.mp4)"):
                with st.spinner("Downloading tandt_db.zip and encoding truck.mp4..."):
                    subprocess.run([get_env_wrapper(), "python", "scripts/prepare_data.py"], cwd=str(ROOT), check=True)
                st.success("Prepared inputs/truck.mp4!")
                st.rerun()

        available_cases = get_all_cases(cfg)
        selected_cases = st.multiselect(
            "Cases to Execute (cached runs are skipped automatically)",
            available_cases,
            default=["baseline", "case1", "case2"],
        )

        col_btn1, col_btn2 = st.columns(2)
        run_log_path = LOGS_DIR / f"webui_{profile}.log"

        with col_btn1:
            if st.button("▶️ Start Selected Cases in Background", type="primary"):
                cmd = [get_env_wrapper(), "python", "scripts/experiment.py", "--config", str(selected_cfg_path)]
                if chosen_vid != vid_options[0]:
                    cmd += ["--video", str(INPUTS_DIR / chosen_vid)]
                if selected_cases:
                    cmd += ["--cases", *selected_cases]
                with open(run_log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n=== LAUNCHED {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)} ===\n")
                    subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
                st.success(f"Started background process! Logging to `{run_log_path.name}`.")

        with col_btn2:
            if st.button("🗑️ Clear Cached Outputs for Selected Cases"):
                for c_name in selected_cases:
                    for r_dir in (SPLATS_DIR / profile).glob(f"{c_name}*"):
                        metrics_f = r_dir / "metrics.json"
                        if metrics_f.exists():
                            metrics_f.unlink()
                st.warning("Removed cached metrics.json markers for selected cases so they will re-train.")

    with col2:
        st.subheader("Live Logs")
        log_files = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        res_log = RESULTS_DIR / profile / "log.txt"
        log_choices = ([res_log] if res_log.exists() else []) + log_files
        if log_choices:
            chosen_log = st.selectbox("Log File", log_choices, format_func=lambda p: f"{p.parent.name}/{p.name}")
            tail_n = st.slider("Lines to show", 20, 300, 60)
            lines = chosen_log.read_text(errors="replace").splitlines()
            st.code("\n".join(lines[-tail_n:]), language="text")
        else:
            st.info("No log files found yet.")

# === TAB 2: TEST SETTINGS & VIDEOS ===
with tab_cfg:
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.subheader(f"Edit Settings (`{selected_cfg_name}`)")
        new_profile = st.text_input("Profile Name (subfolder under splats/ & results/)", value=str(cfg.get("profile", profile)))
        new_steps = st.number_input("Training Steps (`steps`)", min_value=100, max_value=100000, value=int(cfg.get("steps", 7000)), step=500)
        new_cap = st.number_input("Max Splat Budget (`cap`)", min_value=10000, max_value=10000000, value=int(cfg.get("cap", 400000)), step=50000)
        ops = ["realesrgan", "bicubic", "toon", "none"]
        cur_op = cfg.get("upscale_op", "realesrgan")
        new_op = st.selectbox("Default Upscale / Stylize Operator (`upscale_op`)", ops, index=ops.index(cur_op) if cur_op in ops else 0)
        new_scale = st.selectbox("Upscale Factor (`upscale_scale`)", [1, 2, 4], index=[1, 2, 4].index(int(cfg.get("upscale_scale", 2))) if int(cfg.get("upscale_scale", 2)) in [1, 2, 4] else 1)
        new_ngens = st.slider("Multi-Gen Depth (`n_gens` for Case 3)", 2, 8, int(cfg.get("n_gens", 4)))
        new_npaths = st.slider("Alternate Paths (`n_alt_paths` for Cases 4 & 5)", 1, 4, int(cfg.get("n_alt_paths", 3)))
        new_pfactor = st.slider("Path Frame Count Factor (`path_frame_factor`)", 0.25, 3.0, float(cfg.get("path_frame_factor", 1.0)), 0.25)
        new_rweight = st.slider("Real-Frame Anchor Sampling Weight (`real_weight`)", 0.0, 1.0, float(cfg.get("real_weight", 0.5)), 0.05)
        new_test_every = st.number_input("Hold-out Test Split Stride (`test_every`)", 4, 30, int(cfg.get("test_every", 8)))
        new_fps = st.number_input("Video Extraction FPS (`fps`)", 1.0, 60.0, float(cfg.get("fps", 10.0)), 1.0)
        seeds_str = st.text_input("Seeds (comma-separated integers)", value=", ".join(map(str, cfg.get("seeds", [0, 1]))))

        save_as_name = st.text_input("Save Config Filename", value=selected_cfg_name)
        if st.button("💾 Save Configuration"):
            cfg.update(
                profile=new_profile,
                steps=int(new_steps),
                cap=int(new_cap),
                upscale_op=new_op,
                upscale_scale=int(new_scale),
                n_gens=int(new_ngens),
                n_alt_paths=int(new_npaths),
                path_frame_factor=float(new_pfactor),
                real_weight=float(new_rweight),
                test_every=int(new_test_every),
                fps=float(new_fps),
                seeds=[int(s.strip()) for s in seeds_str.split(",") if s.strip()],
            )
            out_p = CONFIGS_DIR / (save_as_name if save_as_name.endswith(".yaml") else f"{save_as_name}.yaml")
            save_cfg(out_p, cfg)
            st.success(f"Saved `{out_p.name}`!")
            st.rerun()

    with c_right:
        st.subheader("Upload or Manage Input Videos")
        uploaded = st.file_uploader("Upload a video file (`.mp4`, `.mov`) to `inputs/`", type=["mp4", "mov", "avi", "mkv"])
        if uploaded is not None:
            dest = INPUTS_DIR / uploaded.name
            dest.write_bytes(uploaded.read())
            st.success(f"Saved `{dest.name}` ({dest.stat().st_size / 2**20:.1f} MB) to `inputs/`!")

        st.markdown("#### Existing Videos in `inputs/`")
        for v in list_videos():
            st.markdown(f"- `{v.name}` ({v.stat().st_size / 2**20:.1f} MB)")

# === TAB 3: CUSTOM CASE BUILDER ===
with tab_cases:
    st.subheader("Create & Manage Custom Experimental Cases")
    st.markdown(
        "Design new single-path, multi-path, multi-generational, or stylized cases without writing code. "
        "Custom cases are saved directly into the active YAML profile and can be launched from **Run & Monitor**."
    )
    cc_col1, cc_col2 = st.columns([1, 1])
    with cc_col1:
        case_id = st.text_input("Case ID (e.g. `case7_anime_multipath`)", value="case7_custom")
        existing_runs = ["baseline_gen1"] + [d.name for d in sorted((SPLATS_DIR / profile).glob("*")) if d.is_dir()]
        src_run = st.selectbox("Source Splat to Render Flythroughs From (`src_run`)", list(dict.fromkeys(existing_runs)))
        run_name = st.text_input("Output Splat Run Name (`run_name`)", value=f"{case_id}_gen2")
        cc_op = st.selectbox("Frame Post-Processing / Upscaler (`op`)", ["realesrgan", "bicubic", "toon", "none"])
        cc_anchor = st.checkbox("Anchor with Real Training Frames (`anchor=True`, prevents generational drift)", value=True)
        cc_note = st.text_input("Experiment Note / Hypothesis", value="Custom multi-path anchored test")

    with cc_col2:
        st.markdown("#### Flythrough Camera Paths")
        num_paths = st.number_input("Number of Flythrough Paths", 1, 6, 2)
        paths_list = []
        for i in range(int(num_paths)):
            with st.expander(f"Path #{i + 1} Offsets", expanded=(i == 0)):
                p_name = st.text_input(f"Path Name #{i + 1}", value="orig" if i == 0 else f"path_{chr(65 + i)}", key=f"pn_{i}")
                c1, c2, c3, c4 = st.columns(4)
                lat = c1.number_input("Lateral", -0.5, 0.5, 0.0 if i == 0 else 0.08 * (1 if i % 2 else -1), 0.02, key=f"lat_{i}")
                vert = c2.number_input("Vertical", -0.5, 0.5, 0.0, 0.02, key=f"vert_{i}")
                yaw = c3.number_input("Yaw (deg)", -45.0, 45.0, 0.0 if i == 0 else 6.0 * (1 if i % 2 else -1), 1.0, key=f"yaw_{i}")
                pitch = c4.number_input("Pitch (deg)", -45.0, 45.0, 0.0, 1.0, key=f"pit_{i}")
                paths_list.append(dict(name=p_name, lateral=float(lat), vertical=float(vert), yaw_deg=float(yaw), pitch_deg=float(pitch)))

    if st.button("➕ Save Custom Case to Profile", type="primary"):
        cfg.setdefault("custom_cases", {})[case_id] = dict(
            src_run=src_run,
            run_name=run_name,
            op=cc_op,
            anchor=bool(cc_anchor),
            note=cc_note,
            paths=paths_list,
        )
        save_cfg(selected_cfg_path, cfg)
        st.success(f"Saved `{case_id}` to `{selected_cfg_name}`! It is now selectable in the Run & Monitor tab.")
        st.rerun()

    if cfg.get("custom_cases"):
        st.markdown("#### Custom Cases in Active Profile")
        st.json(cfg["custom_cases"])

# === TAB 4: RESULTS & LEADERBOARD ===
with tab_res:
    st.subheader(f" Quantitative Metrics Leaderboard (`{profile}`)")
    st.caption("All runs are evaluated against held-out REAL ground-truth frames (`split == 'test'`). Higher PSNR/SSIM and lower LPIPS are better.")
    df = load_summary_df(profile)
    if df.empty:
        st.info(f"No completed runs with `metrics.json` found under `splats/{profile}/` yet.")
    else:
        st.dataframe(
            df.style.format({"psnr": "{:.2f}", "ssim": "{:.4f}", "lpips": "{:.4f}", "ply_mb": "{:.1f}", "train_seconds": "{:.0f}"}),
            use_container_width=True,
        )
        c_chart1, c_chart2, c_chart3 = st.columns(3)
        with c_chart1:
            st.markdown("**PSNR (dB) ↑**")
            st.bar_chart(df.set_index("run")["psnr"])
        with c_chart2:
            st.markdown("**SSIM ↑**")
            st.bar_chart(df.set_index("run")["ssim"])
        with c_chart3:
            st.markdown("**LPIPS ↓**")
            st.bar_chart(df.set_index("run")["lpips"])

# === TAB 5: VISUAL COMPARISON & PLY EXPORT ===
with tab_gal:
    st.subheader("Side-by-Side Hold-out View Renders & `.ply` Downloads")
    p_dir = SPLATS_DIR / profile
    run_dirs = [d for d in sorted(p_dir.glob("*")) if (d / "renders").exists()] if p_dir.exists() else []
    if not run_dirs:
        st.info("No test renders available yet.")
    else:
        run_names = [d.name for d in run_dirs]
        col_a, col_b = st.columns(2)
        run_a = col_a.selectbox("Left Run", run_names, index=0)
        run_b = col_b.selectbox("Right Run", run_names, index=min(1, len(run_names) - 1))

        renders_a = sorted((p_dir / run_a / "renders").glob("*.jpg"))
        render_names = [r.name for r in renders_a]
        if render_names:
            chosen_view = st.select_slider("Held-out Test View (Prediction on Left half of image | Real GT on Right half)", options=render_names)
            img_a = p_dir / run_a / "renders" / chosen_view
            img_b = p_dir / run_b / "renders" / chosen_view
            if img_a.exists():
                col_a.image(str(img_a), caption=f"{run_a} — {chosen_view} (Left: Splat Render, Right: Real GT)", use_container_width=True)
            if img_b.exists():
                col_b.image(str(img_b), caption=f"{run_b} — {chosen_view} (Left: Splat Render, Right: Real GT)", use_container_width=True)

        st.markdown("#### Export Trained Gaussian Splats (`.ply`)")
        for d in run_dirs:
            ply = d / "splat.ply"
            if ply.exists():
                st.markdown(f"- **`{d.name}`**: `{ply}` ({ply.stat().st_size / 2**20:.1f} MB) — ready for Unity (`UnityGaussianSplatting`) or SuperSplat")
