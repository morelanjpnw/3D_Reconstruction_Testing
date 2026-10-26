"""Streamlit Web Dashboard for 3D Reconstruction & Gaussian Splat Iteration Testing.

Launch via:
    tools\\env.local.cmd streamlit run scripts\\app.py
"""
from __future__ import annotations

import json
import subprocess
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
st.markdown(
    "**What is this?** This tool turns regular 2D video walk-arounds into interactive **3D Gaussian Splats** (3D scenes you can fly through in Unity or game engines) "
    "and tests whether **AI Upscaling**, **Multiple Camera Paths**, and **Multi-Generation Loops** make the 3D scene sharper—or cause it to degrade."
)

with st.expander("📖 Quick Plain-English Glossary & Guide (Click to expand)", expanded=False):
    st.markdown(
        """
- **Gaussian Splat (`.ply`):** Millions of tiny tinted 3D clouds that together form a photorealistic 3D scene you can walk around in real time.
- **SfM / COLMAP (Camera Tracking):** The first step where the computer figures out exactly where the camera was in 3D space for every video frame.
- **Flythrough:** A virtual camera recording a new video inside your 3D Splat—either following the original camera path or shifting left/right/up/down to fill in blind spots.
- **Real-Frame Anchoring (`anchor`):** Mixing the original real video frames back in with AI-upscaled frames during training so the AI doesn't "hallucinate" and drift away from reality over multiple generations.
- **How to read the Quality Scores (evaluated on held-out real photos the AI never saw during training):**
  - **PSNR (Sharpness & Accuracy):** Higher is better (typical good range: `22 – 28+ dB`). A drop of more than `0.2 dB` means noticeable quality loss.
  - **SSIM (Structural Similarity):** Higher is better (`0.0` to `1.0`, closer to `1.0` means shapes and edges match the real scene).
  - **LPIPS (Human Perceptual Blur/Artifacts):** **Lower is better** (`0.0` to `1.0`, lower means it looks cleaner and sharper to the human eye).
"""
    )

CASE_DESCRIPTIONS = {
    "baseline": "Baseline (Gen 1) — Build the initial 3D scene directly from the real source video. This is our benchmark to beat.",
    "case1": "Case 1: No-AI Re-render Control (Gen 2) — Record a virtual flythrough of Baseline and re-train WITHOUT AI upscaling. Shows how much quality is lost just from re-copying.",
    "case2": "Case 2: Single AI Upscale (Gen 2) — Record a virtual flythrough of Baseline, sharpen the frames 2x with AI (plus a non-AI resize control), and train a new 3D scene.",
    "case3": "Case 3: Multi-Generation Loop (Gen 2 → Gen 4+) — Repeat the AI upscale loop over multiple generations to see if quality compounds or collapses (tests both Pure Synthetic vs. Real-Photo Anchored).",
    "case4": "Case 4: Multi-Path Flythroughs (Gen 2) — Record multiple virtual camera paths (shifted left, right, up, down), AI-upscale all of them, and combine them into one richer 3D scene.",
    "case5": "Case 5: Multi-Path + Multi-Gen Combined (Gen 3) — Take the Case 4 multi-path 3D scene, fly brand-new camera angles through it, AI-upscale, and train Gen 3.",
    "case6": "Case 6: Stylized 3D Scene (Cartoon / Cel-Shaded) — Convert the flythrough frames into a stylized cartoon/anime look before training the 3D scene.",
}


def case_label(c_key: str, cfg: dict) -> str:
    if c_key in CASE_DESCRIPTIONS:
        return CASE_DESCRIPTIONS[c_key]
    custom = (cfg.get("custom_cases") or {}).get(c_key, {})
    return f"{c_key} (Custom) — {custom.get('note', 'User-defined custom test case')}"


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
st.sidebar.header("1. Choose Hardware / Test Profile")
cfg_files = list_configs()
cfg_names = [f.name for f in cfg_files]
default_idx = cfg_names.index("poc_3060ti.yaml") if "poc_3060ti.yaml" in cfg_names else 0
selected_cfg_name = st.sidebar.selectbox(
    "Configuration Preset",
    cfg_names,
    index=default_idx,
    help="Pick 'poc_3060ti.yaml' for this 8GB PC, 'rtx6000_quick.yaml' for a fast high-res run on the RTX 6000, or 'rtx6000.yaml' for an overnight production run.",
)
selected_cfg_path = CONFIGS_DIR / selected_cfg_name
cfg = load_cfg(selected_cfg_path)
profile = cfg.get("profile", selected_cfg_path.stem)

st.sidebar.markdown(
    f"- **Folder Name (`profile`):** `{profile}`\n"
    f"- **Training Effort (`steps`):** `{cfg.get('steps', 7000):,}` passes\n"
    f"- **3D Detail Budget (`cap`):** `{cfg.get('cap', 400000):,}` splats\n"
    f"- **AI Enhancer:** `{cfg.get('upscale_op', 'realesrgan')}` ({cfg.get('upscale_scale', 2)}x)"
)

if st.sidebar.button("🔄 Refresh Status & Scores", use_container_width=True):
    st.rerun()

# ----------------------------------------------------------------------------- tabs
tab_run, tab_cfg, tab_cases, tab_res, tab_gal = st.tabs([
    "🚀 1. Run & Monitor Tests",
    "⚙️ 2. Adjust Settings & Upload Video",
    "🧪 3. Build a Custom Test Case",
    "📊 4. Scoreboard & Charts",
    "🖼️ 5. Visual Compare & Export 3D Files",
])

# === TAB 1: RUN & MONITOR ===
with tab_run:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Start or Resume an Experiment")
        st.caption("Pick a source video and check which test cases you want the GPU to run. Any case that already finished is automatically skipped unless you clear it.")

        videos = list_videos()
        vid_options = ["(Use already-extracted video frames & camera tracking)"] + [v.name for v in videos]
        default_vid = 1 if videos else 0
        chosen_vid = st.selectbox(
            "Source Video",
            vid_options,
            index=default_vid,
            help="The real-world video we want to turn into a 3D scene. Camera tracking (COLMAP) only needs to run once per profile.",
        )

        if not videos:
            st.info("No videos in `inputs/` yet. Click below to automatically download the standard 'Truck' test video, or upload your own video in Tab 2.")
            if st.button("📥 Download Benchmark Video (truck.mp4)"):
                with st.spinner("Downloading tandt_db.zip and encoding truck.mp4..."):
                    subprocess.run([get_env_wrapper(), "python", "scripts/prepare_data.py"], cwd=str(ROOT), check=True)
                st.success("Prepared inputs/truck.mp4!")
                st.rerun()

        available_cases = get_all_cases(cfg)
        selected_cases = st.multiselect(
            "Which Experimental Cases do you want to run?",
            available_cases,
            default=["baseline", "case1", "case2"],
            format_func=lambda k: case_label(k, cfg),
        )

        col_btn1, col_btn2 = st.columns(2)
        run_log_path = LOGS_DIR / f"webui_{profile}.log"

        with col_btn1:
            if st.button("▶️ Start Selected Cases in Background", type="primary", use_container_width=True):
                cmd = [get_env_wrapper(), "python", "scripts/experiment.py", "--config", str(selected_cfg_path)]
                if chosen_vid != vid_options[0]:
                    cmd += ["--video", str(INPUTS_DIR / chosen_vid)]
                if selected_cases:
                    cmd += ["--cases", *selected_cases]
                with open(run_log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n=== LAUNCHED {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)} ===\n")
                    subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
                st.success(f"Running in the background! Watch progress on the right (`{run_log_path.name}`).")

        with col_btn2:
            if st.button("🗑️ Force Re-run (Clear Selected Cache)", use_container_width=True, help="Deletes the 'completed' marker for the selected cases so clicking Start will train them fresh."):
                for c_name in selected_cases:
                    for r_dir in (SPLATS_DIR / profile).glob(f"{c_name}*"):
                        metrics_f = r_dir / "metrics.json"
                        if metrics_f.exists():
                            metrics_f.unlink()
                st.warning("Cleared completion markers for selected cases. Click Start to re-train them.")

    with col2:
        st.subheader("Live Progress Log")
        st.caption("Shows what the GPU is doing right now (e.g., `step 3500/7000` during training, or `PSNR 23.46` when a 3D scene finishes).")
        log_files = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        res_log = RESULTS_DIR / profile / "log.txt"
        log_choices = ([res_log] if res_log.exists() else []) + log_files
        if log_choices:
            chosen_log = st.selectbox(
                "Select Log Stream",
                log_choices,
                format_func=lambda p: f"📊 {p.parent.name}/log.txt (High-level summary)" if p.name == "log.txt" else f"⚙️ {p.name} (Detailed step-by-step output)",
            )
            tail_n = st.slider("Number of recent lines to display", 20, 300, 60)
            lines = chosen_log.read_text(errors="replace").splitlines()
            st.code("\n".join(lines[-tail_n:]), language="text")
        else:
            st.info("No log files yet—start a run on the left!")

# === TAB 2: TEST SETTINGS & VIDEOS ===
with tab_cfg:
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.subheader(f"Customize Settings (`{selected_cfg_name}`)")
        st.caption("Hover over the `?` icon next to any setting for a plain-English explanation.")

        new_profile = st.text_input(
            "Profile Folder Name (`profile`)",
            value=str(cfg.get("profile", profile)),
            help="All extracted frames, 3D splats, and scoreboards for this preset are saved in a folder with this name.",
        )
        new_steps = st.number_input(
            "Training Duration — Iterations (`steps`)",
            min_value=100,
            max_value=100000,
            value=int(cfg.get("steps", 7000)),
            step=500,
            help="How many optimization passes the trainer makes per 3D scene. 7,000 is fast (~3-5 min on 3060 Ti); 15,000–30,000 gives maximum sharpness on RTX 6000.",
        )
        new_cap = st.number_input(
            "3D Detail Budget — Max Gaussians (`cap`)",
            min_value=10000,
            max_value=10000000,
            value=int(cfg.get("cap", 400000)),
            step=50000,
            help="Maximum number of 3D particles used to represent the scene. More splats = finer detail but larger .ply file size and more VRAM. (400k = ~95 MB .ply; 1M = ~240 MB .ply).",
        )

        op_labels = {
            "realesrgan": "Real-ESRGAN (AI Super-Resolution — sharpens textures & edges)",
            "bicubic": "Bicubic / Lanczos (Standard non-AI resize — control test)",
            "toon": "Toon / Cel-Shader (Stylized cartoon look)",
            "none": "None (Keep raw rendered frames as-is)",
        }
        ops = list(op_labels.keys())
        cur_op = cfg.get("upscale_op", "realesrgan")
        new_op = st.selectbox(
            "Default Frame Enhancer (`upscale_op`)",
            ops,
            index=ops.index(cur_op) if cur_op in ops else 0,
            format_func=lambda k: op_labels[k],
            help="How synthetic flythrough frames are enhanced before training the next generation.",
        )
        new_scale = st.selectbox(
            "Resolution Multiplier (`upscale_scale`)",
            [1, 2, 4],
            index=[1, 2, 4].index(int(cfg.get("upscale_scale", 2))) if int(cfg.get("upscale_scale", 2)) in [1, 2, 4] else 1,
            help="2x doubles the width and height (4x total pixels). Higher resolution takes more VRAM and training time.",
        )
        new_ngens = st.slider(
            "Number of Generations in Case 3 (`n_gens`)",
            2,
            8,
            int(cfg.get("n_gens", 4)),
            help="How many times Case 3 loops (Gen 1 → Gen 2 → Gen 3 → Gen 4) to test whether quality keeps improving or starts breaking down.",
        )
        new_npaths = st.slider(
            "Number of Alternate Camera Paths in Cases 4 & 5 (`n_alt_paths`)",
            1,
            4,
            int(cfg.get("n_alt_paths", 3)),
            help="How many shifted virtual camera flythroughs (Left, Right, Up, Down) are combined to cover blind spots.",
        )
        new_pfactor = st.slider(
            "Frames per Virtual Flythrough (`path_frame_factor`)",
            0.25,
            3.0,
            float(cfg.get("path_frame_factor", 1.0)),
            0.25,
            help="1.0 means each virtual flythrough renders the same number of frames as your real training set (e.g., 219 frames). 2.0 renders twice as many.",
        )
        new_rweight = st.slider(
            "Real-Photo Anchor Mix (`real_weight`)",
            0.0,
            1.0,
            float(cfg.get("real_weight", 0.5)),
            0.05,
            help="In 'Anchored' runs, this is how often the trainer looks at a real original photo vs. an AI-upscaled frame. 0.5 (50/50 mix) stops the AI from drifting away from reality.",
        )
        new_test_every = st.number_input(
            "Hold-out Test Photo Interval (`test_every`)",
            4,
            30,
            int(cfg.get("test_every", 8)),
            help="Every Nth real frame (e.g. every 8th photo) is locked away and NEVER used for training, so we have an honest way to grade the final 3D scene.",
        )
        new_fps = st.number_input(
            "Video Frame Extraction Rate — FPS (`fps`)",
            1.0,
            60.0,
            float(cfg.get("fps", 10.0)),
            1.0,
            help="How many frames per second to pull from the source video. Aim for ~150–300 total frames across your video.",
        )
        seeds_str = st.text_input(
            "Random Seeds (`seeds`)",
            value=", ".join(map(str, cfg.get("seeds", [0, 1]))),
            help="Running Baseline with two different random numbers (0 and 1) tells us the natural random variation (~0.04 dB) so we know if a score change is real.",
        )

        save_as_name = st.text_input("Save Preset As Filename", value=selected_cfg_name)
        if st.button("💾 Save Settings Preset", type="primary"):
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
        st.subheader("Upload Your Own Walkthrough Video")
        st.markdown(
            "**Tips for shooting a great video for 3D Splatting:**\n"
            "- Walk slowly and smoothly around the environment or object.\n"
            "- Lock camera exposure/focus if possible and avoid motion blur.\n"
            "- Avoid moving people, pets, or swaying trees (the 3D solver expects a static scene)."
        )
        uploaded = st.file_uploader("Drop a video file (`.mp4`, `.mov`) here to save it into `inputs/`", type=["mp4", "mov", "avi", "mkv"])
        if uploaded is not None:
            dest = INPUTS_DIR / uploaded.name
            dest.write_bytes(uploaded.read())
            st.success(f"Saved `{dest.name}` ({dest.stat().st_size / 2**20:.1f} MB) to `inputs/`! You can now pick it in Tab 1.")

        st.markdown("#### Videos Currently Available in `inputs/`")
        for v in list_videos():
            st.markdown(f"- 🎬 **`{v.name}`** ({v.stat().st_size / 2**20:.1f} MB)")

# === TAB 3: CUSTOM CASE BUILDER ===
with tab_cases:
    st.subheader("Design Your Own Experimental Case (No Coding Required)")
    st.markdown(
        "Want to test a new idea—like taking a 3D scene, flying 3 wide-angle paths through it, applying the cartoon filter, and anchoring with real photos? "
        "Build it here and click **Save**. It will immediately show up in **Tab 1** ready to run."
    )
    cc_col1, cc_col2 = st.columns([1, 1])
    with cc_col1:
        case_id = st.text_input("Short ID for your test (no spaces)", value="case7_custom")
        existing_runs = ["baseline_gen1"] + [d.name for d in sorted((SPLATS_DIR / profile).glob("*")) if d.is_dir()]
        src_run = st.selectbox(
            "Starting 3D Scene to Fly Through (`src_run`)",
            list(dict.fromkeys(existing_runs)),
            help="Which existing 3D splat should the virtual camera fly through? Pick 'baseline_gen1' for a Gen 2 test, or pick a Gen 2 splat to make a Gen 3 test.",
        )
        run_name = st.text_input("Output Folder Name for New 3D Splat (`run_name`)", value=f"{case_id}_gen2")
        cc_op = st.selectbox(
            "Frame Enhancer / Style Filter (`op`)",
            ["realesrgan", "bicubic", "toon", "none"],
            format_func=lambda k: {
                "realesrgan": "Real-ESRGAN (AI 2x Sharpener)",
                "bicubic": "Bicubic (Standard non-AI 2x resize)",
                "toon": "Toon (Cartoon / Cel-shade filter)",
                "none": "None (Raw virtual renders)",
            }[k],
        )
        cc_anchor = st.checkbox(
            "✅ Mix in Original Real Photos during training (`anchor` — strongly recommended!)",
            value=True,
            help="Keeps the 3D scene grounded in reality so AI upscaling adds sharpness without ruining geometry.",
        )
        cc_note = st.text_input("Plain-English Description (shown on the Scoreboard)", value="Custom multi-path anchored test")

    with cc_col2:
        st.markdown("#### Configure Virtual Camera Flythrough Paths")
        st.caption("Path #1 defaults to the original camera path (`0` offset). Add extra paths shifted slightly left/right/up/down to capture new angles.")
        num_paths = st.number_input("How many virtual camera paths should we record?", 1, 6, 2)
        paths_list = []
        for i in range(int(num_paths)):
            with st.expander(f"🎥 Virtual Camera Path #{i + 1}", expanded=(i == 0)):
                p_name = st.text_input(f"Path Label #{i + 1}", value="orig" if i == 0 else f"path_{chr(65 + i)}", key=f"pn_{i}")
                c1, c2, c3, c4 = st.columns(4)
                lat = c1.number_input("Shift Left/Right", -0.5, 0.5, 0.0 if i == 0 else 0.08 * (1 if i % 2 else -1), 0.02, key=f"lat_{i}", help="Fraction of scene radius (- is left, + is right)")
                vert = c2.number_input("Shift Up/Down", -0.5, 0.5, 0.0, 0.02, key=f"vert_{i}", help="Fraction of scene radius (- is up, + is down)")
                yaw = c3.number_input("Turn Left/Right (°)", -45.0, 45.0, 0.0 if i == 0 else 6.0 * (1 if i % 2 else -1), 1.0, key=f"yaw_{i}", help="Camera pan angle in degrees")
                pitch = c4.number_input("Tilt Up/Down (°)", -45.0, 45.0, 0.0, 1.0, key=f"pit_{i}", help="Camera tilt angle in degrees")
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
        st.success(f"Saved `{case_id}`! Go to **Tab 1 (Run & Monitor)** to select and run it.")
        st.rerun()

    if cfg.get("custom_cases"):
        st.markdown("#### Saved Custom Cases in This Profile")
        st.json(cfg["custom_cases"])

# === TAB 4: RESULTS & LEADERBOARD ===
with tab_res:
    st.subheader(f"📊 Quality Scoreboard (`{profile}`)")
    st.markdown(
        "**How to read this table:** Every 3D scene is graded by comparing its 3D renders against **held-out real camera photos** that were hidden during training.\n"
        "- **PSNR (dB) ↑:** Overall image accuracy & sharpness (**Higher is better**). Compare every case against `baseline_gen1`.\n"
        "- **SSIM ↑:** Structural accuracy from 0 to 1 (**Higher is better**).\n"
        "- **LPIPS ↓:** Perceptual artifacts/blur (**Lower is better** — closer to 0 means cleaner to the human eye)."
    )
    df = load_summary_df(profile)
    if df.empty:
        st.info(f"No finished runs found in `splats/{profile}/` yet. Check Tab 1 to see if a run is still in progress.")
    else:
        friendly_df = df.rename(columns={
            "run": "Run Name",
            "psnr": "PSNR (dB) ↑ [Higher=Better]",
            "ssim": "SSIM ↑ [Higher=Better]",
            "lpips": "LPIPS ↓ [Lower=Better]",
            "n_splats": "3D Splats Count",
            "ply_mb": "File Size (MB)",
            "n_train_frames": "Training Frames Used",
            "train_seconds": "Train Time (sec)",
            "anchor": "Anchored w/ Real Photos?",
            "seed": "Random Seed",
            "note": "What This Test Does",
        })
        st.dataframe(
            friendly_df.style.format({
                "PSNR (dB) ↑ [Higher=Better]": "{:.2f}",
                "SSIM ↑ [Higher=Better]": "{:.4f}",
                "LPIPS ↓ [Lower=Better]": "{:.4f}",
                "File Size (MB)": "{:.1f}",
                "Train Time (sec)": "{:.0f}",
            }),
            use_container_width=True,
        )
        c_chart1, c_chart2, c_chart3 = st.columns(3)
        with c_chart1:
            st.markdown("**Sharpness & Accuracy — PSNR (dB) ↑ (Higher is Better)**")
            st.bar_chart(df.set_index("run")["psnr"])
        with c_chart2:
            st.markdown("**Structure Match — SSIM ↑ (Higher is Better)**")
            st.bar_chart(df.set_index("run")["ssim"])
        with c_chart3:
            st.markdown("**Visual Artifacts/Blur — LPIPS ↓ (Lower is Better)**")
            st.bar_chart(df.set_index("run")["lpips"])

# === TAB 5: VISUAL COMPARISON & PLY EXPORT ===
with tab_gal:
    st.subheader("🖼️ Visual Side-by-Side Comparison & 3D File Exports")
    st.markdown(
        "Pick any two experimental runs below to compare how sharp they look from test camera angles the AI never saw during training. "
        "*(Inside each image: the **Left half** is the 3D Splat render, and the **Right half** is the real Ground-Truth photograph.)*"
    )
    p_dir = SPLATS_DIR / profile
    run_dirs = [d for d in sorted(p_dir.glob("*")) if (d / "renders").exists()] if p_dir.exists() else []
    if not run_dirs:
        st.info("No test renders available yet.")
    else:
        run_names = [d.name for d in run_dirs]
        col_a, col_b = st.columns(2)
        run_a = col_a.selectbox("Compare Run A (Left Column)", run_names, index=0)
        run_b = col_b.selectbox("Compare Run B (Right Column)", run_names, index=min(1, len(run_names) - 1))

        renders_a = sorted((p_dir / run_a / "renders").glob("*.jpg"))
        render_names = [r.name for r in renders_a]
        if render_names:
            chosen_view = st.select_slider("Slide to switch between held-out test camera angles", options=render_names)
            img_a = p_dir / run_a / "renders" / chosen_view
            img_b = p_dir / run_b / "renders" / chosen_view
            if img_a.exists():
                col_a.image(str(img_a), caption=f"{run_a} — {chosen_view} (Left half: 3D Splat | Right half: Real Photo)", use_container_width=True)
            if img_b.exists():
                col_b.image(str(img_b), caption=f"{run_b} — {chosen_view} (Left half: 3D Splat | Right half: Real Photo)", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📦 Export Your 3D Gaussian Splat Files (`.ply`)")
        st.caption("You can drag-and-drop these `.ply` files straight into the free web viewer **SuperSplat** (`playcanvas.com/supersplat/editor`) or import them into **Unity** using `UnityGaussianSplatting`.")
        for d in run_dirs:
            ply = d / "splat.ply"
            if ply.exists():
                st.markdown(f"- **`{d.name}`**: `{ply}` ({ply.stat().st_size / 2**20:.1f} MB)")
