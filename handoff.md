# Handoff – 3D_Reconstruction_Testing (SplatLab)

**Last Updated:** 2026-10-27 ~03:20 CDT
**GitHub Repository:** `https://github.com/morelanjpnw/3D_Reconstruction_Testing.git` (branch `main`)
**Active machine:** office workstation, **NVIDIA RTX 4500 Ada Generation (24 GB VRAM, sm_89)**
**Previous machine:** home PoC box (RTX 3060 Ti 8 GB) at `D:\GeminiSandbox\3D_Reconstruction_Testing`. Retired for this project.

> Agent: read this file, `AGENTS.md`, and `docs/test_plan.md` before doing anything.
> Notifications for long jobs follow the user's global agent rules (the ntfy topic is intentionally not stored in this repo).

---

## 1. Getting started on the office machine (new context)

### 1a. First-time setup (user or agent, PowerShell)
```powershell
cd <work folder, e.g. D:\splat_work>
git clone https://github.com/morelanjpnw/3D_Reconstruction_Testing.git
cd 3D_Reconstruction_Testing
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -CheckOnly   # report only
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1              # install (5-15 min)
```
- Ada (sm_89) takes the **non-Blackwell** branch: PyTorch 2.4.1 `cu121`, `TORCH_CUDA_ARCH_LIST=8.9`. A CUDA Toolkit 12.x `nvcc` plus VS 2019/2022 C++ Build Tools must be installed. `-CheckOnly` reports anything missing.
- Setup generates `tools\env.local.cmd`. On this machine, **always** use `tools\env.local.cmd <cmd>`, never `tools\env.cmd` (that one is home-machine specific).
- Setup runs `scripts/patch_gsplat_windows.py` and `scripts/selftest.py`. The first gsplat CUDA JIT build takes a few minutes.

### 1b. Get the data (git-ignored, so it isn't in the clone)
```powershell
tools\env.local.cmd python scripts\prepare_data.py     # downloads tandt_db.zip (~650 MB) -> inputs\truck.mp4
```
`inputs\_raw\tandt_db.zip` also contains the authors' COLMAP model (`tandt/truck/sparse/0`), which test-plan case B1 needs.

### 1c. Smoke-test the new GPU, then measure throughput
```powershell
tools\env.local.cmd python scripts\experiment.py --config configs\rtx4500.yaml --video inputs\truck.mp4 --cases baseline --set profile=smoke steps=500 cap=100000 seeds=[0]
tools\env.local.cmd python scripts\experiment.py --config configs\rtx4500.yaml --video inputs\truck.mp4 --cases baseline
```
The second command is a 30k-step, 2M-splat baseline. Its result is the first data point for test plan H1, which predicts a large gain over the PoC's 23.46 dB. Record `train_seconds` and `peak_vram_gb` from `splats\rtx4500\baseline_gen1\metrics.json` and update the time estimates in `docs/test_plan.md`.

### 1d. Suggested first prompt for the new session
> Read handoff.md, AGENTS.md and docs/test_plan.md in D:\<path>\3D_Reconstruction_Testing. Verify setup (`setup_windows.ps1 -CheckOnly`), run the smoke test and the rtx4500 baseline, then start Phase 0 of the test plan.

---

## 2. Project overview
**SplatLab** is an end-to-end pipeline plus a **Streamlit dashboard** (`launch_ui.cmd`). It tests whether 3D Gaussian Splats for game environments can be improved by iterating virtual-camera flythroughs through AI upscaling, multi-path trajectories, multi-generation loops and stylization.

### Code map
- `scripts/splatlib.py`: COLMAP loader, spline flythrough paths (`make_path`), MCMC trainer (`train`), `.ply`/`.splat` export, renderer, held-out evaluator (`evaluate`).
- `scripts/experiment.py`: resumable, cached orchestrator (`baseline`, `case1`..`case6`, `custom_cases` from YAML, `--set KEY=VALUE` overrides).
- `scripts/upscale.py`: `none | bicubic | realesrgan | toon`. Real-ESRGAN always runs at native `-s 4`, then downsamples.
- `scripts/app.py` + `launch_ui.cmd`: 6-tab dashboard, including a WebGL splat viewer and a live CUDA camera explorer.
- `setup_windows.ps1`: one-command installer with GPU-architecture detection.
- `configs/`: `poc_3060ti.yaml` (8 GB), **`rtx4500.yaml` (current)**, `rtx6000_quick.yaml` / `rtx6000.yaml` (96 GB, if ever available).

---

## 3. Results so far (PoC on RTX 3060 Ti, 7k steps, 400k splats, truck 979×546)
Full table: `results/poc_3060ti/summary.md` and `README.md`. Headlines:

| Run | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| `baseline_gen1` (real only) | **23.46** | **0.810** | **0.238** |
| `case1_gen2` (re-render, no AI) | 22.11 | 0.777 | 0.297 |
| `case2_gen2` (Real-ESRGAN 2x) | 20.99 | 0.732 | 0.377 |
| `case3_gen4_pure` (unanchored gen 4) | 18.83 | 0.659 | 0.452 |
| `case4a_gen2_anch` (multi-path + AI + anchored, best iterative) | 23.04 | 0.790 | 0.297 |

**Analysis (details in `docs/test_plan.md`):** no iterative case beats the baseline, and the baseline itself is badly undertrained (5 min, 0.69 GB VRAM). Visible problems: streaky background, flat untextured ground, unreadable text. Likely causes: undertraining, sparse init, and 979 px input. Real-ESRGAN performs worse than plain Lanczos. Anchoring is the biggest lever. The PoC matrix has compute/resolution confounders, and its test set only measures on-path interpolation.

---

## 4. Environment gotchas
1. Run Python only through `tools\env.local.cmd` (vcvars64, `TORCH_CUDA_ARCH_LIST`, conda env on PATH, `CL=/DWIN32_LEAN_AND_MEAN /DNOMINMAX`).
2. Never start `streamlit` attached to the agent console. Use `Start-Process -WindowStyle Hidden ...` or have the user run `launch_ui.cmd`.
3. gsplat 1.5.3 passes `-Wno-attributes`, which MSVC rejects (D8021). Re-run `scripts/patch_gsplat_windows.py` after any gsplat reinstall.
4. Real-ESRGAN `x4plus`: never use `-s 2` (it corrupts tiles).

---

## 5. Next steps (see `docs/test_plan.md`)
1. Setup and smoke test on the RTX 4500 (section 1), then run the 30k `rtx4500` baseline.
2. **Phase 0:** extrapolation split, OCR text-legibility metric, off-path no-reference / FID metric, equal-compute protocol, region metrics.
3. **Phase 1:** B1–B7 baseline fixes, then the B* combination.
4. Phases 2–4: full-res data and own capture, Difix3D+ / input-side video SR, Unity benchmarking.
5. Open questions for the user: approval for Phase 0/1, OK to download full-res T&T truck, whether they will shoot their own scene.
