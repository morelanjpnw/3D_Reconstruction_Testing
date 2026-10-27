| Run | PSNR | SSIM | LPIPS | Splats | PLY MB | Train frames | Train s | Note |
|---|---|---|---|---|---|---|---|---|
| baseline_gen1 | 23.46 | 0.8102 | 0.2375 | 400000 | 95 | 219 | 297 | real frames only |
| baseline_gen1_seed1 | 23.42 | 0.8109 | 0.2385 | 400000 | 95 | 219 | 204 | variance floor |
| case1_gen2 | 22.11 | 0.7772 | 0.2965 | 400000 | 95 | 219 | 198 | re-render only |
| case2_gen2 | 20.99 | 0.7316 | 0.3767 | 400000 | 95 | 219 | 534 | AI upscale |
| case2b_gen2_bicubic | 21.91 | 0.7672 | 0.3234 | 400000 | 95 | 219 | 532 | control: plain 2x resize |
| case3_gen3_pure | 19.82 | 0.6922 | 0.4254 | 400000 | 95 | 219 | 505 | pure multi-gen |
| case3_gen4_pure | 18.83 | 0.6589 | 0.4520 | 400000 | 95 | 219 | 493 | pure multi-gen |
| case3a_gen2_anch | 22.66 | 0.7826 | 0.3018 | 400000 | 95 | 438 | 353 | anchored gen2 |
| case3a_gen3_anch | 22.37 | 0.7736 | 0.3229 | 400000 | 95 | 438 | 355 | anch multi-gen |
| case3a_gen4_anch | 22.03 | 0.7653 | 0.3343 | 400000 | 95 | 438 | 360 | anch multi-gen |
| case4_gen2 | 21.41 | 0.7375 | 0.3811 | 400000 | 95 | 657 | 531 | multi-path + AI |
| case4_noai_gen2 | 22.41 | 0.7746 | 0.3167 | 400000 | 95 | 657 | 199 | ablation: multi-path, no AI |
| case4a_gen2_anch | 23.04 | 0.7895 | 0.2967 | 400000 | 95 | 876 | 356 | multi-path + AI + real anchor |
| case5_gen3 | 20.23 | 0.6988 | 0.4245 | 400000 | 95 | 657 | 513 | multi-path x multi-gen |
| case5a_gen3_anch | 22.68 | 0.7809 | 0.3084 | 400000 | 95 | 876 | 362 | multi-path x multi-gen, anchored |
| case6a_toon_gen2 | 20.64 | 0.6918 | 0.3929 | 400000 | 95 | 219 | 192 | deterministic toon stylisation (metrics vs real are NOT meaningful) |