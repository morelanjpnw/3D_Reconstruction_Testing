# SplatLab (3D_Reconstruction_Testing) – Agent Rules

## Environment
- ALWAYS run Python/pip/torch commands through the env wrapper:
  - `tools\env.cmd <cmd>` on the original PoC machine (RTX 3060 Ti).
  - `tools\env.local.cmd <cmd>` on machines provisioned via `setup_windows.ps1`.
  - The wrapper loads `vcvars64.bat`, sets `TORCH_CUDA_ARCH_LIST`, puts the `splatlab`
    conda env on PATH, and sets `CL=/DWIN32_LEAN_AND_MEAN /DNOMINMAX` (prevents the
    `windows.h` `#define small char` macro from breaking CUDA CUB headers).
- NEVER launch `streamlit` attached to the agent's console (vcvars + streamlit corrupts
  Windows Console Input Mode). Start it detached, e.g.
  `Start-Process -WindowStyle Hidden ...`, or tell the user to run `launch_ui.cmd`.

## Known gotchas
- gsplat 1.5.3 passes `-Wno-attributes`, which MSVC rejects (`D8021`). After any gsplat
  (re)install, run `scripts/patch_gsplat_windows.py`.
- `realesrgan-ncnn-vulkan` with `realesrgan-x4plus`: NEVER pass `-s 2` (corrupts tile
  stitching). Run native `-s 4`, then downsample with `cv2.INTER_AREA` to the target scale.
- Blackwell (RTX PRO 6000, `sm_120`) needs CUDA >= 12.8 and PyTorch `cu128`.
- Active machine: RTX 4500 Ada (24 GB, `sm_89`), profile `configs/rtx4500.yaml`. Run heavy
  generative models (Difix3D+, SeedVR2) as separate stages, never alongside training.

## Conventions
- The current analysis and test plan live in `docs/test_plan.md`. Follow its phases and update it with results.
- `handoff.md` is the source of truth for project state. Read it at session start and
  update it before ending a substantial session.
- Comparison metrics: PoC baseline `baseline_gen1` = 23.46 dB PSNR / 0.8102 SSIM /
  0.2375 LPIPS (`results/poc_3060ti/summary.md`). Seed variance floor is about ±0.04 dB.
