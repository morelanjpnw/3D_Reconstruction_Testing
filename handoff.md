# Handoff – 3D_Reconstruction_Testing (SplatLab)

**Last Updated:** 2026-10-27 ~02:50 CDT
**Project Root:** `D:\GeminiSandbox\3D_Reconstruction_Testing`
**GitHub Repository:** `https://github.com/morelanjpnw/3D_Reconstruction_Testing.git` (Branch: `main`, fully synced)

---

## 1. Project Overview & Goal
We built **SplatLab**, an end-to-end experimental pipeline and **Streamlit Web Dashboard** (with built-in **WebGL 3D Splat Explorer** and **Live CUDA Camera Explorer**) to test improving **3D Gaussian Splats (3DGS)** for interactive gaming environments by iterating virtual camera flythroughs through AI upscaling, multi-path camera trajectories, multi-generational loops, and stylization.

- **Local PoC Machine:** Windows 11, NVIDIA GeForce RTX 3060 Ti (8 GB VRAM), 32 GB RAM, CUDA 11.8, VS 2019/2022 BuildTools.
- **Work Target Machine:** Windows 10/11, NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM).
- **Notifications:** High-priority alerts/results are sent to the user's iPhone via `https://ntfy.sh/prodiga_agy_updates2026` (`curl.exe -H "Title: ..." -d "..." https://ntfy.sh/prodiga_agy_updates2026`).

---

## 2. What Is 100% Complete

### A. Full Proof-of-Concept Matrix on RTX 3060 Ti (`poc_3060ti` — 16/16 Runs Finished)
All 16 runs across **Baseline** and **Cases 1–6** completed on the Tanks & Temples `truck` scene (`inputs/truck.mp4`, 251 frames at 978×546; 219 train frames + 32 held-out test frames at `test_every: 8`, MCMC `cap: 400,000`, `steps: 7,000`).
Full metrics are saved in `results/poc_3060ti/summary.md`, `results/poc_3060ti/summary.json`, and documented in `README.md`:

| Run | Description | Gen | Anchored? | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|:---:|---:|---:|---:|
| `baseline_gen1` | Real frames only (Seed 0) | 1 | Yes (100%) | **23.46** | **0.8102** | **0.2375** |
| `baseline_gen1_seed1` | Real frames only (Seed 1 — variance floor `±0.04 dB`) | 1 | Yes (100%) | **23.42** | **0.8109** | **0.2385** |
| `case1_gen2` | No-AI re-render control (`-1.35 dB` generation loss) | 2 | No | 22.11 | 0.7772 | 0.2965 |
| `case2_gen2` | Single-path Real-ESRGAN 2x (`1954x1090`), unanchored | 2 | No | 20.99 | 0.7316 | 0.3767 |
| `case2b_gen2_bicubic` | Single-path 2x Lanczos resize control | 2 | No | 21.91 | 0.7672 | 0.3234 |
| `case3_gen3_pure` | Unanchored multi-gen Gen 3 | 3 | No | 19.82 | 0.6922 | 0.4254 |
| `case3_gen4_pure` | Unanchored multi-gen Gen 4 (severe collapse: `-4.63 dB`) | 4 | No | 18.83 | 0.6589 | 0.4520 |
| `case3a_gen2_anch` | Single-path AI upscale + 50% Real-Frame Anchor | 2 | **Yes (50%)** | **22.66** | **0.7826** | **0.3018** |
| `case3a_gen3_anch` | Anchored multi-gen Gen 3 (`+2.55 dB` vs pure Gen 3) | 3 | **Yes (50%)** | **22.37** | **0.7736** | **0.3229** |
| `case3a_gen4_anch` | Anchored multi-gen Gen 4 (`+3.20 dB` vs pure Gen 4) | 4 | **Yes (50%)** | **22.03** | **0.7653** | **0.3343** |
| `case4_gen2` | Multi-Path (A, B, C) + AI upscale, unanchored | 2 | No | 21.41 | 0.7375 | 0.3811 |
| `case4_noai_gen2` | Multi-Path (A, B, C) no-AI ablation (`+0.30 dB` vs Case 1) | 2 | No | 22.41 | 0.7746 | 0.3167 |
| **`case4a_gen2_anch`** | **Multi-Path (A, B, C) + AI + Anchored (Best iterative run)** | 2 | **Yes (50%)** | **23.04** | **0.7895** | **0.2967** |
| `case5_gen3` | Multi-Path × Multi-Gen (Paths E, F, G), unanchored | 3 | No | 20.23 | 0.6988 | 0.4245 |
| **`case5a_gen3_anch`** | **Multi-Path × Multi-Gen (Paths E, F, G), Anchored** | 3 | **Yes (50%)** | **22.68** | **0.7809** | **0.3084** |
| `case6a_toon_gen2` | Deterministic bilateral toon / cel-shade stylization | 2 | No | 20.64 | 0.6918 | 0.3929 |

### B. Codebase & Web Interface
- **`scripts/app.py` & `launch_ui.cmd`:** Full 6-tab **Streamlit Web Dashboard** with layman explanations, expandable glossary, hover tooltips, background job launcher, live log viewer, settings editor, video uploader, **No-Code Custom Case Builder** (`custom_cases` in YAML), scoreboard charts, side-by-side hold-out render comparison, and **Tab 6: Interactive 3D Splat Explorer**:
  - **Mode A (WebGL 3D Viewer):** Uses `gsplat@1.2.4` (`dist/index.js`) with `LoadFromArrayBuffer` on compact `.splat` files (`12.8 MB` vs `95 MB` `.ply`). Includes demand-driven rendering + settle window (eliminating background WASM depth-sort race flickering) and explicit iframe/canvas focus management so `WASD` / Arrow / `Q/E` / `R/F` keys work immediately on click.
  - **Mode B (Live GPU Camera Explorer):** Real-time 6-DoF novel-view rendering on the server's GPU via `gsplat`'s CUDA rasterizer with full Degree-3 Spherical Harmonics.
- **`scripts/splatlib.py`:** COLMAP loader, spline camera path generator (`make_path`), MCMC trainer (`train`), `.ply` and compact `.splat` exporter (`export_ply`, `export_splat`), batch renderer, single-view interactive renderer (`render_single`), and held-out evaluator (`evaluate`).
- **`scripts/upscale.py`:** Supports `none | bicubic | realesrgan | toon`. Fixed a critical `realesrgan-ncnn-vulkan` bug where passing `-s 2` to `realesrgan-x4plus` corrupted tile stitching; it now runs native `-s 4` and downsamples cleanly with `cv2.INTER_AREA` to the target scale.
- **`scripts/experiment.py`:** Resumable, cached orchestrator supporting `baseline`, `case1`..`case6`, dynamic `--set KEY=VALUE` overrides, and user-defined `custom_cases` from YAML.
- **`setup_windows.ps1`:** One-command Windows installer with `-CheckOnly` mode, Blackwell (`sm_120`, CUDA >= 12.8, PyTorch `cu128`) vs. Ada/Ampere auto-detection, COLMAP & Real-ESRGAN download, `patch_gsplat_windows.py`, `tools\env.local.cmd` generation, and `selftest.py`.

---

## 3. Environment & Important Technical Gotchas
1. **Always run Python via the wrapper (`tools\env.cmd` locally, or `tools\env.local.cmd` on machines set up via `setup_windows.ps1`):**
   - Loads Visual Studio C++ `vcvars64.bat`, sets `TORCH_CUDA_ARCH_LIST`, adds the `splatlab` conda env to `PATH`, and sets `CL=/DWIN32_LEAN_AND_MEAN /DNOMINMAX` (prevents `windows.h`'s `#define small char` macro from breaking CUDA CUB headers).
2. **Never launch `streamlit` directly attached to the Antigravity CLI console:**
   - Running `vcvars64.bat` + `streamlit` inside the CLI's console tree alters Windows Console Input Mode flags. Always start Streamlit detached (`Start-Process -WindowStyle Hidden ...`) or from a separate terminal window via `launch_ui.cmd`.
3. **Installed `gsplat` Windows Patch (`scripts/patch_gsplat_windows.py`):**
   - `gsplat 1.5.3` passes `-Wno-attributes` to `extra_cflags`, which MSVC `cl.exe` rejects with `D8021`. `patch_gsplat_windows.py` patches `gsplat/cuda/_backend.py` so `-Wno-attributes` is only passed on non-Windows.

---

## 4. Next Steps for the New Session
1. **RTX PRO 6000 Runs (`configs/rtx6000_quick.yaml` & `configs/rtx6000.yaml`):**
   - Assist the user if any setup or driver quirk arises on the RTX PRO 6000 machine.
   - Compare `15,000–30,000` step / `1M–3M` splat runs on the RTX 6000 against the 3060 Ti PoC baseline, specifically testing whether higher step counts allow 2x upscaled multi-path anchored runs (`case4a_gen2_anch`) to surpass the 1x `baseline_gen1` PSNR (`23.46 dB`).
2. **ComfyUI Generative Video Upscaling & Stylization Integration (Next Feature Phase):**
   - Add a `comfyui` operator in `scripts/upscale.py` that posts frames/videos to a local ComfyUI HTTP API workflow:
     - **Track A (Fidelity):** Temporally consistent video upscalers (**SeedVR2**, **STAR**, or **Upscale-A-Video**) vs. per-frame Real-ESRGAN.
     - **Track B (Stylization — Case 6b/6c):** Diffusion stylization (Anime, Claymation) using **Wan 2.1 VACE** or **SDXL/Flux + Depth/Lineart ControlNet + IP-Adapter**, plus an **Instruct-GS2GS** iterative dataset update loop.
3. **Unity Integration & Real-Time FPS Benchmarking:**
   - Import the exported `.ply` files into Unity (`UnityGaussianSplatting`) to log in-engine frame rate (FPS), VRAM usage, and visual quality across cases.
