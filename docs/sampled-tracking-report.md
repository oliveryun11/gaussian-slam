# Sampled tracking on Replica room0

## Implementation

Tracking can use a uniform sample of image pixels without replacement. The sample is drawn once per frame and reused across pose candidates. The RGB and depth losses retain their existing validity masks, and the sampled sums are scaled by the inverse sampling fraction. CUDA backward skips pixels with zero upstream gradients and returns early for tiles with no gradient work. Forward rendering and the mapping loss still use the full image.

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

## Next measurement

Repeat the same-extension 100% versus 10% comparison on one exclusively allocated Delta GPU. Use the same frame window for a direct comparison, then collect a longer trace for the baseline report. Evaluate 25% and 50% coverage before selecting a ratio for full-sequence runs.
