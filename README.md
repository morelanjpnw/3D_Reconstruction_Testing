# 3D Reconstruction Testing

Experimental pipeline to improve Gaussian Splats by iterating video flythroughs through generative AI upscaling. We compare simple iterations against multi-generational and multi-path approaches.

## Quickstart on a new Windows machine (e.g. the RTX PRO 6000)

```powershell
mkdir D:\splat_work; cd D:\splat_work                      # any folder you like
git clone https://github.com/morelanjpnw/3D_Reconstruction_Testing.git
cd 3D_Reconstruction_Testing
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -CheckOnly   # reports anything missing (installs nothing)
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1              # installs tools + env, GPU self-test (5-15 min first time)
tools\env.local.cmd python scripts\prepare_data.py                        # downloads the truck test scene -> inputs\truck.mp4
tools\env.local.cmd python scripts\experiment.py --config configs\rtx6000_quick.yaml --video inputs\truck.mp4 --cases baseline case1 case2 case4 case3 case5 case6
```

Everything is cached: if a run is interrupted, re-run the same command and it resumes. Results land in
`results\<profile>\summary.md`; splats in `splats\<profile>\<run>\splat.ply`.
Use `--set` for quick smoke tests, e.g. `--set profile=smoke steps=300 cap=50000 seeds=[0]`.
Profiles: `configs\poc_3060ti.yaml` (8 GB GPU), `configs\rtx6000_quick.yaml`, `configs\rtx6000.yaml` (full, long).

## Directory Layout

| Folder | Purpose |
|---|---|
| `inputs/` | Original source videos |
| `extracted_frames/` | Frame sets, one subfolder per case/generation (e.g. `baseline/`, `case2_gen2/`) |
| `splats/` | Trained Gaussian Splats per case/generation |
| `flythroughs/` | Rendered flythrough videos (and camera path files) per case/generation/path |
| `scripts/` | Orchestration scripts (`baseline.py`, ...) |

Suggested naming: `<case>_gen<N>[_path<X>]`, e.g. `case4_gen2_pathB`.

## Workflows

### Baseline
Original Video → Extract Frames → Gaussian Splat (Gen 1)
- Run: `python scripts/baseline.py inputs/video.mp4 --fps 2`
- Output: `extracted_frames/baseline/`, then `splats/baseline_gen1`.

### Case 1 – Single Iteration (control, no AI)
Baseline → Render Flythrough (Original Path) → Extract Frames → Gaussian Splat (Gen 2)
- Isolates the effect of re-training on the splat's own renders.

### Case 2 – AI Upscaled Iteration
Baseline → Render Flythrough (Original Path) → ComfyUI/GenAI Upscale → Extract Frames → Gaussian Splat (Gen 2)
- Compare against Case 1 to measure the upscaler's contribution.

### Case 3 – Multi-Generational Iteration
Case 2 Gen 2 Splat → Render Flythrough → ComfyUI/GenAI Upscale → Extract Frames → Gaussian Splat (Gen 3) → repeat for N generations.
- Tests degradation vs. enhancement over generations (watch for drift, hallucinated detail, and loss of multi-view consistency).

### Case 4 – Multi-Path Variation
Baseline → Render N alternate flythrough paths (A, B, C…) → ComfyUI/GenAI Upscale → Extract frames from all paths → Combine frame pools → Gaussian Splat (Gen 2)
- Tests whether added viewpoint diversity improves coverage and quality.

### Case 5 – Combined Multi-Path & Multi-Gen
Case 4 Gen 2 Splat → Map completely new alternate paths through Gen 2 Splat → Upscale → Extract → Combine → Gaussian Splat (Gen 3)
- Can be repeated for further generations.

## Experiment Tracking

Keep the same frame count, resolution, trainer settings, and seed across cases for fair comparison. Log each run below.

| Run | Case | Gen | Paths | Upscaler / Settings | Frames | Splat | PSNR | SSIM | LPIPS | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | 1 | – | – | | | | | | |
