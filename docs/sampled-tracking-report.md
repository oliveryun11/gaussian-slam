# Sampled tracking on Replica room0

## Implementation

Tracking can use a uniform sample of image pixels without replacement. The sample is drawn once per frame and reused across pose candidates. The RGB and depth losses retain their existing validity masks, and the sampled sums are scaled by the inverse sampling fraction. CUDA backward skips pixels with zero upstream gradients and returns early for tiles with no gradient work. Sampled forward rendering blends only selected pixels and returns full-size tensors with zeros outside the sample. Depth outlier rejection still renders the full depth image for its median calculation. Mapping uses the full image.

Set `config["tracking"]["pixel_sample_ratio"] = 0.1` in a local Replica config for 10% sampling. The default is `1.0`. Rebuild the CUDA extension after switching to this implementation using the [setup guide](setup.md#5-build-the-rasterizer).

The rebuilt extension matched the original within floating-point tolerance for all six gradient outputs in each of three synthetic cases with dense, sparse, and all-zero upstream gradients. A three-frame Replica run also completed at 10% coverage.

## Results

Each pair ran the first 100 frames of Replica `room0` on an RTX 4080 Laptop GPU under WSL2 using the same seed and iteration counts. Nsight Systems 2026.6.1 captured frame indices 80–88. The tracking backward kernel was identified by matching CUDA launches to the tracking NVTX range. There were 720 such launches in each capture.

| Comparison | Backward kernel mean, full → 10% | Tracking per frame, full → 10% | SLAM time, full → 10% | ATE RMSE | PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original baseline followed by sampled tracking | 10.81 → 2.58 ms (4.2×) | 3.67 → 2.74 s | 16:08 → 14:06 | 0.14 → 0.13 cm | 40.26 → 40.37 dB |
| Same extension, sampled tracking run first | 18.97 → 9.47 ms (2.0×) | 11.96 → 10.18 s | 47:47 → 41:08 | 0.13 → 0.12 cm | 39.61 → 40.28 dB |

The same-extension repeat reversed the run order and reduced total SLAM time by 14%. Median memory clocks matched at 6,000 MHz, and the mapping backward kernel differed by about 2%. Median GPU core clocks still differed at 825 MHz for full coverage and 908 MHz for 10% sampling. In the first capture, median memory clocks differed at 6,000 and 9,000 MHz, respectively.

A separate unprofiled pair had a 63% slower sampled run overall, with mapping also slowing by 59%. The laptop runs show a consistent reduction in measured tracking backward kernel time in the two traces, with variable end-to-end timings. Full-sequence accuracy remains unmeasured.

### Adding sampled forward rendering

The restarted Nov 22 run used 10% tracking coverage with both sampled passes. The comparison below uses the earlier 10% backward-only run with the same 100-frame sequence and capture window. Each trace recorded 720 tracking launches per rendering kernel.

| Metric | Backward only | Forward + backward |
| --- | ---: | ---: |
| Mean tracking forward kernel time | 2.27 ms | 1.82 ms |
| Mean tracking backward kernel time | 2.58 ms | 2.68 ms |
| Average tracking time per frame | 2.74 s | 2.26 s |
| Total SLAM time | 14:06 | 13:28 |
| ATE RMSE | 0.13 cm | 0.15 cm |
| PSNR | 40.37 dB | 39.94 dB |

Mean tracking forward kernel time fell by 20%, and total SLAM time fell by 4.5% in this pair. Both captures had a median GPU core clock of 2,115 MHz and a median temperature of 76 °C. Median memory clocks were 9,000 MHz for backward only and 8,500 MHz for forward plus backward. The new trace is saved locally at `experiments/Replica/room0_nsys_forward_backward/forward-backward.nsys-rep`.

### Coverage sweep

The same extension ran the first 100 frames of `room0` at four tracking coverage levels. Nsight Systems captured frames 40–89, with 4,000 launches of each tracking render kernel per setting. The kernel columns report mean GPU time per launch.

| Tracking coverage | Forward kernel | Backward kernel | Total SLAM | ATE RMSE | PSNR | Forward predicated-on lanes per warp instruction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 4.35 ms | 14.99 ms | 17:38 | 0.14 cm | 40.00 dB | 16.9 |
| 50% | 7.06 ms | 12.39 ms | 34:48 | 0.13 cm | 40.26 dB | 9.0 |
| 25% | 7.03 ms | 11.13 ms | 34:57 | 0.13 cm | 40.35 dB | 5.0 |
| 10% | 6.56 ms | 9.91 ms | 34:21 | 0.14 cm | 40.57 dB | 2.6 |

At 10% coverage, tracking backward kernel time fell by 34%. Accuracy was similar across the four settings. The sampled runs took longer overall, and mapping also slowed from 5.86 s/frame to 10.11–10.31 s/frame despite using full-image rendering in every run. This timing variation prevents attributing the end-to-end difference to tracking sampling alone.

The separate Nsight Compute pass captured four forward launches per setting at frame 40. Lane participation is the ratio of predicated-on thread instructions to executed warp instructions. At 10% coverage, the forward kernel still executed about 71% as many warp instructions as at full coverage, while lane participation fell from 16.9 to 2.6. These are whole-kernel averages. Backward lane participation remains unmeasured. The timing CSV and counter reports are saved locally under `experiments/Replica/room0_coverage_sweep/`.

## Next measurement

Repeat the same-extension coverage sweep on one exclusively allocated Delta GPU using frames 40–89. Collect backward-specific lane counters and use the speed and accuracy results to choose a ratio for full-sequence runs.

### Reproduce the coverage sweep

With the current extension built and Replica available, run from the repository root:

```bash
python scripts/profile_tracking.py experiments/Replica/room0_coverage_sweep
```

The runner captures frames 40–89 for each of 100%, 50%, 25%, and 10% coverage. It runs the first 100 frames sequentially and writes one row per coverage to `summary.csv`. Each row contains tracking render launch counts and mean GPU kernel times, tracking time per frame, total SLAM time, ATE, and PSNR. For the earlier nine-frame capture window, pass `--start 80 --end 89`. The end frame is exclusive.

On a machine with Nsight Compute and access to GPU performance counters, add `--ncu /path/to/ncu`. The runner collects Source Counters for the first four matching tracking render launches at each coverage in a separate pass. Nsight Compute 2026.4 writes `active-lanes.ncu-repz`. Build with `NVCC_APPEND_FLAGS=-lineinfo` to compare lane participation by source line. Shared batch loading and blending use different lane participation.
