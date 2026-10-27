# 3D Reconstruction Testing (SplatLab)

Experimental pipeline and interactive **Streamlit Web UI** (with built-in **WebGL 3D Splat Explorer** and **Live CUDA Camera Explorer**) to test improving **3D Gaussian Splats** by iterating virtual flythroughs through AI upscaling, multi-path camera trajectories, multi-generational loops, and stylization.

## Quickstart on a Windows Machine (e.g. RTX PRO 6000 or Local GPU)

```powershell
mkdir D:\splat_work; cd D:\splat_work                      # any folder you like
git clone https://github.com/morelanjpnw/3D_Reconstruction_Testing.git
cd 3D_Reconstruction_Testing
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -CheckOnly   # reports anything missing (installs nothing)
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1              # installs tools + env, GPU self-test (5-15 min first time)

# Launch the interactive Web UI (recommended) - or run via CLI below
.\launch_ui.cmd

# --- OR run via CLI ---
tools\env.local.cmd python scripts\prepare_data.py                        # downloads the truck test scene -> inputs\truck.mp4
tools\env.local.cmd python scripts\experiment.py --config configs\rtx6000_quick.yaml --video inputs\truck.mp4 --cases baseline case1 case2 case4 case3 case5 case6
```

Everything is cached: if a run is interrupted, re-run the same command and it resumes. Results land in `results\<profile>\summary.md`; splats in `splats\<profile>\<run>\splat.ply` (and compact `.splat` for instant web viewing).
Profiles: `configs\poc_3060ti.yaml` (8 GB GPU), `configs\rtx4500.yaml` (24 GB RTX 4500 Ada, current office machine), `configs\rtx6000_quick.yaml`, `configs\rtx6000.yaml` (96 GB, full production).
Current analysis and next test plan: [`docs/test_plan.md`](docs/test_plan.md).

## Directory Layout

| Folder | Purpose |
|---|---|
| `inputs/` | Source walkthrough videos (e.g., `truck.mp4` or custom uploads) |
| `extracted_frames/` | Extracted & processed frame pools per profile/case/path (`<profile>/<run>/<path>/`) |
| `flythroughs/` | Raw rendered virtual flythrough frames + `cameras.json` poses (`<profile>/<run>/<path>/`) |
| `splats/` | Trained Gaussian Splats (`splat.pt`, `splat.ply`, `splat.splat`, `metrics.json`, `renders/`) |
| `results/` | Aggregated leaderboard tables (`summary.md`, `summary.json`, `log.txt`) |
| `configs/` | Hardware & experiment YAML presets (`poc_3060ti.yaml`, `rtx6000_quick.yaml`, `rtx6000.yaml`) |
| `scripts/` | Core library (`splatlib.py`), orchestrator (`experiment.py`), Web UI (`app.py`), SfM (`sfm.py`), upscaler (`upscale.py`) |

---

## Proof-of-Concept Experimental Results (`poc_3060ti` — Tanks & Temples `truck`)

All 16 experimental runs were evaluated against **32 held-out real camera views** (`test_every: 8`, never seen during training) at fixed MCMC budget (`400,000` splats, `7,000` steps per run).

| Run | Case / Variant | Gen | Anchored w/ Real? | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Splats | PLY MB | Train Frames | Train (s) | Notes |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_gen1` | **Baseline (Seed 0)** | 1 | Yes (100%) | **23.46** | **0.8102** | **0.2375** | 400k | 95 | 219 | 297 | Real frames only |
| `baseline_gen1_seed1` | **Baseline (Seed 1)** | 1 | Yes (100%) | **23.42** | **0.8109** | **0.2385** | 400k | 95 | 219 | 204 | Seed variance floor (`±0.04 dB`) |
| `case1_gen2` | **Case 1 (No-AI Control)** | 2 | No | 22.11 | 0.7772 | 0.2965 | 400k | 95 | 219 | 198 | Re-render only (`-1.35 dB` generation loss) |
| `case2_gen2` | **Case 2 (AI Upscale)** | 2 | No | 20.99 | 0.7316 | 0.3767 | 400k | 95 | 219 | 534 | Real-ESRGAN 2x (`1954x1090`), unanchored |
| `case2b_gen2_bicubic` | **Case 2b (2x Resize Control)** | 2 | No | 21.91 | 0.7672 | 0.3234 | 400k | 95 | 219 | 532 | Plain 2x Lanczos resize control |
| `case3_gen3_pure` | **Case 3 (Pure Multi-Gen)** | 3 | No | 19.82 | 0.6922 | 0.4254 | 400k | 95 | 219 | 505 | Unanchored Gen 3 |
| `case3_gen4_pure` | **Case 3 (Pure Multi-Gen)** | 4 | No | 18.83 | 0.6589 | 0.4520 | 400k | 95 | 219 | 493 | Unanchored Gen 4 (severe drift: `-4.63 dB`) |
| `case3a_gen2_anch` | **Case 3a (Anchored Multi-Gen)** | 2 | **Yes (50%)** | **22.66** | **0.7826** | **0.3018** | 400k | 95 | 438 | 353 | `+1.67 dB` vs unanchored `case2_gen2` |
| `case3a_gen3_anch` | **Case 3a (Anchored Multi-Gen)** | 3 | **Yes (50%)** | **22.37** | **0.7736** | **0.3229** | 400k | 95 | 438 | 355 | `+2.55 dB` vs `case3_gen3_pure` |
| `case3a_gen4_anch` | **Case 3a (Anchored Multi-Gen)** | 4 | **Yes (50%)** | **22.03** | **0.7653** | **0.3343** | 400k | 95 | 438 | 360 | `+3.20 dB` vs `case3_gen4_pure` |
| `case4_gen2` | **Case 4 (Multi-Path + AI)** | 2 | No | 21.41 | 0.7375 | 0.3811 | 400k | 95 | 657 | 531 | Paths A, B, C + Real-ESRGAN (`+0.42 dB` vs single-path Case 2) |
| `case4_noai_gen2` | **Case 4 Ablation (Multi-Path, No AI)** | 2 | No | 22.41 | 0.7746 | 0.3167 | 400k | 95 | 657 | 199 | Paths A, B, C raw (`+0.30 dB` vs single-path Case 1) |
| `case4a_gen2_anch` | **Case 4a (Multi-Path + AI + Anchored)** | 2 | **Yes (50%)** | **23.04** | **0.7895** | **0.2967** | 400k | 95 | 876 | 356 | **Best synthetic iteration overall** |
| `case5_gen3` | **Case 5 (Multi-Path × Multi-Gen)** | 3 | No | 20.23 | 0.6988 | 0.4245 | 400k | 95 | 657 | 513 | New paths E, F, G from `case4_gen2` |
| `case5a_gen3_anch` | **Case 5a (Multi-Path × Multi-Gen, Anchored)** | 3 | **Yes (50%)** | **22.68** | **0.7809** | **0.3084** | 400k | 95 | 876 | 362 | `+2.45 dB` over unanchored `case5_gen3` |
| `case6a_toon_gen2` | **Case 6a (Stylized Toon / Cel-Shade)** | 2 | No | 20.64 | 0.6918 | 0.3929 | 400k | 95 | 219 | 192 | Deterministic bilateral toon filter (evaluate visually) |

---

## Key Findings & Takeaways

1. **Re-training purely on synthetic renders degrades fidelity (Case 1):**
   - Even with exact camera poses and zero AI hallucination, training Gen 2 solely on Gen 1's rendered flythrough (`case1_gen2`) drops PSNR by **`-1.35 dB`** (`23.46 → 22.11 dB`) compared to the `±0.04 dB` random seed noise floor.
2. **Per-frame AI super-resolution introduces multi-view inconsistencies unless anchored (Case 2 vs. Case 3a):**
   - Because Real-ESRGAN sharpens edges independently per frame without 3D cross-view awareness, training exclusively on 2x AI-upscaled frames (`case2_gen2`, `20.99 dB`) scores lower on ground-truth PSNR/SSIM than plain 2x Lanczos resizing (`case2b_gen2_bicubic`, `21.91 dB`).
   - However, **Real-Frame Anchoring (`real_weight: 0.5`)** fixes this: `case3a_gen2_anch` jumps to **`22.66 dB`** (`+1.67 dB` over unanchored Case 2, and `+0.55 dB` over Case 1).
3. **Unanchored Multi-Generational loops suffer compounding model collapse, while Anchored loops stay stable (Case 3):**
   - Pure multi-gen (`case2_gen2 → case3_gen3_pure → case3_gen4_pure`) collapses steadily: **`20.99 → 19.82 → 18.83 dB`** (`-4.63 dB` below Baseline).
   - Anchored multi-gen (`case3a_gen2_anch → gen3 → gen4`) holds structure across 4 generations: **`22.66 → 22.37 → 22.03 dB`** (outperforming pure Gen 4 by **`+3.20 dB`**).
4. **Multi-Path viewpoint diversity consistently outperforms Single-Path across every regime (Case 4 & Case 5):**
   - Adding off-axis flythrough paths (`A, B, C` shifted laterally/vertically with yaw/pitch offsets) improves every single counterpart:
     - **No-AI:** `case4_noai_gen2` (`22.41 dB`) beats single-path `case1_gen2` (`22.11 dB`) by **`+0.30 dB`**.
     - **Pure AI Gen 2:** `case4_gen2` (`21.41 dB`) beats single-path `case2_gen2` (`20.99 dB`) by **`+0.42 dB`**.
     - **Pure AI Gen 3:** `case5_gen3` (`20.23 dB`) beats single-path `case3_gen3_pure` (`19.82 dB`) by **`+0.41 dB`**.
     - **Anchored AI Gen 2 (`case4a_gen2_anch`):** Achieves **`23.04 dB`** (`0.7895` SSIM, `0.2967` LPIPS) — the **highest-scoring iterative reconstruction** in the entire study.
     - **Anchored AI Gen 3 (`case5a_gen3_anch`):** Achieves **`22.68 dB`**, beating even Gen 2 single-path runs.
5. **Implications for the RTX PRO 6000 Iteration:**
   - **Always enable Real-Frame Anchoring (`anchor: True`)** when iterating with generative enhancers.
   - **Combine Multi-Path trajectories (`case4a` / `case5a`) with higher splat budgets (`cap: 1M–3M`, `steps: 15k–30k`)**: at `7,000` steps on the 3060 Ti, training on 2x resolution (`1954x1090`) with 4x the pixel count is under-converged relative to 1x resolution; `15,000–30,000` steps on the RTX 6000 will allow the high-frequency detail in the 2x upscaled multi-path pool to fully converge.
   - **Stylization (`case6a_toon_gen2`):** Produces a clean, view-consistent cel-shaded 3D splat (inspectable in Tab 6 of the Web UI) because bilateral color quantization avoids baked black outlines that would flicker across views.
