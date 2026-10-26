# Handoff – 3D_Reconstruction_Testing

Written 2026-10-26 ~09:15 CDT. Project root: `D:\GeminiSandbox\3D_Reconstruction_Testing`

## Goal
Experimental pipeline to improve Gaussian Splats by iterating video flythroughs through AI upscaling, with an eventual stylized-reconstruction branch (anime / claymation). Target use: interactive game environments. The PoC runs on this machine's GPU. A later iteration will run on an **RTX 6000**.

Experimental cases (full definitions in `README.md`): Baseline, Case 1 (re-render only), Case 2 (AI upscale), Case 3 (multi-generation), Case 4 (multi-path), Case 5 (multi-path x multi-gen). Case 6 (stylization) is an addition, not in the original spec.

## Machine
- RTX 3060 Ti **8 GB** (about 2.6 GB used by desktop apps), 32 GB RAM, D: has about 115 GB free.
- CUDA toolkit 11.8 (`nvcc`). Driver 610.47. FFmpeg is on PATH.
- VS 2019 BuildTools is used for compiling gsplat (CUDA 11.8 compatible).

## Done
- Directory tree: `inputs/ extracted_frames/ splats/ flythroughs/ scripts/ configs/ tools/ logs/`.
- `scripts/baseline.py`: FFmpeg frame extraction. Exposes `extract_frames(...)`, which `experiment.py` imports.
- `README.md`: case workflows plus a tracking table. **Not yet updated** with the profiles, run layout, evaluation protocol, or results.
- **Source data:** the Tanks&Temples "truck" scene (251 images, 979x546) from the 3DGS authors' `tandt_db.zip`, chosen with the user's approval as a stand-in for a real capture. It was re-encoded as `inputs/truck.mp4` (978x546, 10 fps, 251 frames, crf 10, cropped to even width). Raw files are in `inputs/_raw/` and can be deleted (about 650 MB zip).
  - Licensing note: this is a research dataset, so don't use it for commercial work.
- **Tools installed:**
  - COLMAP 4.2.1 (CUDA) in `tools/colmap/bin/colmap.exe`.
  - Real-ESRGAN ncnn-vulkan in `tools/realesrgan/` (model `realesrgan-x4plus`, run with `-s 2`).
  - Conda env **`splatlab`** (Python 3.10): torch 2.1.2+cu118, numpy<2, opencv, scipy, scikit-image, lpips, pycolmap 4.2.1, plyfile, torchmetrics, ninja, **gsplat 1.5.3** (pip install finished).
- **Code written, not yet run or tested at all:**
  - `scripts/sfm.py`: COLMAP wrapper (feature extraction, exhaustive matching, mapper, undistort). Flags were checked against the COLMAP 4.2.1 `-h` output.
  - `scripts/splatlib.py`: COLMAP reader, flythrough path generator with offsets, gsplat MCMC trainer, PLY export, renderer, PSNR/SSIM/LPIPS evaluation.
  - `scripts/upscale.py`: ops `none | bicubic | realesrgan | toon`.
  - `scripts/experiment.py`: cached, resumable orchestrator for all cases, plus `summarize()`.
  - `configs/poc_3060ti.yaml` and `configs/rtx6000.yaml`. The RTX 6000 `upscale_op: comfyui` is a placeholder with no code behind it yet.
  - `tools/env.cmd`: wrapper that loads the VS2019 vcvars, sets `TORCH_CUDA_ARCH_LIST=8.6`, and puts the `splatlab` env on PATH.

## Next steps (in order)
1. **Verify gsplat JIT-compiles.** Run: `cd D:\GeminiSandbox\3D_Reconstruction_Testing; cmd /c "tools\env.cmd python -c \"import torch,gsplat;print(gsplat.__version__)\""`.
   - The CUDA extension compiles on first rasterization call and can take several minutes. Trigger it with a tiny test render.
   - If it fails, check the CUDA 11.8 / VS 2019 / `torch 2.1.2` combination first.
2. **Run the baseline prep:** `cmd /c "tools\env.cmd python scripts\experiment.py --config configs\poc_3060ti.yaml --video inputs\truck.mp4"`.
   - Check that COLMAP registers nearly all 251 frames.
   - Check that `baseline_sfm/sparse` contains the `.bin` files directly, so `pycolmap.Reconstruction` can read it. Fix `splatlib.load_colmap` if the layout differs. The `cam_from_world` property/method handling is untested.
3. **Run `--cases baseline` first** and look at the speed and the numbers.
   - Expected quality for truck at this resolution is roughly PSNR 24-26 dB. Much lower means something is wrong.
   - Measure it/s and VRAM. Tune `steps` and `cap` in the PoC config. Training at 2x resolution (upscaled frames, about 1958x1092) is the expensive part.
4. Run the cases in order: `case1 case2 case4 case3 case5 case6`. Re-running is safe because everything is cached.
5. Fill in `README.md` with the results table (`results/poc_3060ti/summary.md`), the protocol, and the findings. Add renders and a comparison grid from `splats/<profile>/<run>/renders/`.

## Design decisions (keep consistent)
- **Evaluation:** every 8th real frame is held out and never used for training. Every run (including later generations) is scored against these real frames at native resolution.
- **Seeds and controls:** baseline is re-trained with a second seed to measure the noise floor. Case 2b is a plain 2x Lanczos resize, a control for "just more pixels". Case 4 also has a no-AI ablation.
- **Anchoring:** variants with "anchored" or `a` in the name mix the real training frames into the pool (50% sampling weight). Pure variants use synthetic frames only, per the original spec. Expect pure multi-gen to degrade.
- **Flythroughs:** paths are built from the training poses only (held-out poses are excluded), smoothed and spline-interpolated. Alternate paths apply offsets in camera space (lateral / vertical as a fraction of the camera orbit radius, plus yaw / pitch). Poses are never re-estimated on synthetic frames.
- **Training:** fixed splat budget through gsplat's MCMC strategy for comparability. SH degree 3. Loss is 0.8 L1 + 0.2 (1 - SSIM) plus opacity and scale regularization.
- **Shortcut:** flythroughs are rendered straight to PNG, with no lossy video encode/decode round trip. Mention this in the README.
- **Upscaler:** Real-ESRGAN is the PoC upscaler because 8 GB VRAM can't run SeedVR2 or STAR. Generative video upscalers via ComfyUI are for the RTX 6000 run.
- **Case 6a (toon)** is deterministic and has no baked outlines. Its PSNR against real frames is not meaningful. The only valid judgment is visual and consistency checks.

## RTX 6000 plan
- Reuse the same code with `configs/rtx6000.yaml`: 30k steps, 3M splat cap, 3 seeds, 5 generations, 4 alt paths.
- Still to build:
  - A ComfyUI API wrapper as a new `upscale.py` op, with the workflow JSON saved per run.
  - SeedVR2 / STAR as the video-consistent upscalers, with Real-ESRGAN as the control arm.
  - A denoise-strength sweep.
  - Stylization: Wan VACE or SDXL/Flux with depth and edge ControlNet, an iterative dataset update (Instruct-GS2GS style), and anime / clay style tests.
  - Metrics for stylized output: CLIP style similarity and multi-view warp error.
  - Multiple scenes, plus engine tests (Unity: FPS, size, import via aras-p's UnityGaussianSplatting).
  - Optional: a Case 4b dense single-path control, 2DGS/PGSR for surface quality, and splat compression.
- A real capture should be made for production: fixed exposure, 4K, a slow walk with lots of parallax.

## Gotchas
- **Windows paths:** all tools run from `D:\GeminiSandbox\3D_Reconstruction_Testing`. Run Python only through `tools\env.cmd`. The conda base Python 3.12 has no cv2.
- **Running background jobs:** `Start-Process -WindowStyle Hidden` silently failed earlier. Use daemon commands, or `cmd /c` through the run tool.
- **Slow downloads:** PowerShell's `Invoke-WebRequest` is slow. Use `curl.exe`.
- **GPU sharing:** the desktop shares the GPU (Teams, Unity, Chrome, Wallpaper Engine). Close apps if VRAM gets tight.

## Notifications
The user wants only notifications that need their attention, sent to `https://ntfy.sh/prodiga_agy_updates2026` with `curl.exe -H "Title: ..." -d "..." <url>`. A test notification was sent and accepted. The user also asked for a clickable remote-control session link. Remote control is enabled for this session (see `cli.log`), but no URL was found. The user must supply the link. Send it to ntfy once they do. Don't invent one.

## Open questions for the user
- The remote-control link (see above).
- Which real capture to use for the production run, when it exists.
- Which engine to target for game integration: Unity (installed) or another.
