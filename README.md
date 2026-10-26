# 3D Reconstruction Testing

Experimental pipeline to improve Gaussian Splats by iterating video flythroughs through generative AI upscaling. We compare simple iterations against multi-generational and multi-path approaches.

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
