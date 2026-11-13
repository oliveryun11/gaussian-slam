# Implementation Plan

The implementation work for the project, broken into tasks.

Dates come from the proposal timeline. Milestone report: **Nov 23**. Poster/demo: **Dec 11**.

---

## What the code does today

Notes from reading `scripts/splatam.py`, `utils/slam_helpers.py` and `rasterizer/baseline/`:

- **Two render passes per iteration.** `get_loss()` (`scripts/splatam.py:214`) calls the rasterizer
  twice: once for RGB, and once for depth + silhouette + depth² (packed into the 3 "color"
  channels via `colors_precomp`). Each pass has its own forward and backward pass.
- **The rasterizer always renders the full image.** Its output is `[C, H, W]`, with H×W = 680×1200 on
  Replica. Tiles are 16×16 (`cuda_rasterizer/config.h`). The kernels are `renderCUDA` in
  `forward.cu:263` and `backward.cu:401`. Sorting and binning are in `rasterizer_impl.cu`.
- **The camera-pose gradient is *not* computed in CUDA.** `transform_to_frame()`
  (`utils/slam_helpers.py:252`) moves Gaussians into the camera frame in PyTorch. The rasterizer
  returns per-Gaussian gradients (means3D, rotations, ...), and autograd reduces them into the pose
  gradient. So in tracking, the atomic contention is on *per-Gaussian* gradients, and the
  per-Gaussian → pose reduction happens in PyTorch (see D3).
- **The mapping loss uses SSIM** (`scripts/splatam.py:290`), which needs pixel neighborhoods
  (an 11×11 window). This doesn't work with randomly scattered sampled pixels (see D2). The tracking
  loss is a plain masked L1 sum, so sampling is easy there.
- **Replica config** (`configs/replica/splatam.py`): isotropic Gaussians, 40 tracking iterations,
  60 mapping iterations, a mapping window of 24 keyframes, densification off, pruning on, and
  `use_wandb=True` (turn it off for development).
- **Existing timers** (`time.time()` around tracking/mapping) don't call `torch.cuda.synchronize()`,
  so they measure kernel *launch* time, not GPU time. Don't trust them.
- **`add_new_gaussians()`** renders a full-resolution silhouette once per frame without gradients.
  It stays full-resolution; it isn't part of the sampled path.

---

## Design decisions to make together

These affect several tasks, so we should decide them together before the dependent tasks start.

- **D1: Sparse rasterizer interface.** Proposal: add an optional `pixel_idx` argument (int32 tensor
  `[N]`, flat `y*W+x`) to the rasterizer call. Output becomes `[C, N]` instead of `[C, H, W]`, and the
  loss compares it against `gt.view(C, -1)[:, pixel_idx]`. Needed before S2/S3.
- **D2: Mapping loss under sampling.** Options: (a) drop SSIM and use L1 only on sampled pixels;
  (b) sample small patches (e.g. 8×8) so SSIM still works on them; (c) compute SSIM only on full-image
  iterations. Measure the effect on accuracy (S4).
- **D3: Pose gradient.** Keep the per-Gaussian → pose reduction in PyTorch (simple), or fuse it into
  the backward kernel with warp-level reduction (closer to what the proposal describes). Decide once B1
  shows how expensive the PyTorch reduction is.
- **D4: Data layout for the pixel-based kernels.** How each sampled pixel finds its Gaussians (reuse
  the tile lists and filter per pixel, or build per-pixel lists), and what the forward pass saves for
  the backward pass (final transmittance, number of contributing Gaussians, etc.). Needed before P1/P2.

---

## Phase 0: Environment & baseline (Nov 12–15)

- [ ] **E1: Dev environment.** Write the steps up in `docs/setup.md`. Python 3.10 (required by
  `open3d==0.16.0`), PyTorch 2.x with CUDA 12.x, and `TORCH_CUDA_ARCH_LIST` including `8.0` (A100)
  plus the dev GPU's architecture.
  *Done when:* `pip install ./rasterizer/baseline` builds and `import diff_gaussian_rasterization` works.
  Commit any build fixes, e.g. `#include <cstdint>` in `cuda_rasterizer/rasterizer_impl.h` for gcc 13.
- [ ] **E2: PyTorch 2.x compatibility fixes.** Fix whatever breaks in SplaTAM's Python code under
  PyTorch 2.x. *Done when:* `python scripts/splatam.py configs/replica/splatam.py` runs on room0
  with `num_frames` limited (e.g. 100).
- [ ] **E3: Datasets.** Download Replica (`bash_scripts/download_replica.sh`) and TUM
  (`bash_scripts/download_tum.sh`) to local storage, and symlink `data/` to them.
- [ ] **E4: Delta environment.** Same software versions as E1, plus Slurm scripts in `slurm/`
  (interactive build job and batch run job). Blocked on ACCESS allocation approval.
- [ ] **E5: Baseline accuracy.** Full Replica room0 (and ideally all 8 scenes) on Delta. Record
  ATE, PSNR, SSIM, LPIPS and FPS, and compare with the SplaTAM paper.

## Measurement infrastructure (start early, used by everything)

- [ ] **B1: Accurate per-phase timing.** Replace the `time.time()` timers with CUDA-event or
  synchronized timers around: tracking iteration, mapping iteration, RGB render fwd/bwd, depth render
  fwd/bwd, loss, optimizer step, `add_new_gaussians`. Write per-frame results to CSV in the run
  directory. Add a flag so timing can be turned off when not needed.
- [ ] **B2: Nsight profile of the baseline.** Profile ~50 frames of room0 with `nsys`. Report the share
  of time in rasterizer kernels vs. everything else (Splatonic says 94.7%), plus kernel launch counts
  and gaps. Save a summary in `docs/results/` (top-level `results/` is git-ignored, and the raw
  `.nsys-rep` files stay out of git).
- [ ] **B3: Rasterizer correctness test harness.** `tests/test_rasterizer.py`: generate random
  Gaussians and a camera, run two rasterizer implementations, and compare forward outputs and every
  gradient within a tolerance. Also test the sampled mode against the full-image output read at the
  same pixels. Needs no dataset.
- [ ] **B4: Benchmark runner.** One script that runs a config matrix (scene × variant × sample count)
  and collects ATE, PSNR and per-phase timings into a single CSV, plus a plotting script.

## Phase 1: Sparse pixel sampling in the tile-based rasterizer (Nov 16–22)

- [ ] **S1: Sampling strategies (Python).** `utils/sampling.py`:
  - tracking: random K pixels per 16×16 tile (respecting the depth > 0 and silhouette masks)
  - mapping: pixels from unmapped (low-silhouette) regions plus textured (high-gradient) regions
  - a config option for the sample count / ratio.
- [ ] **S2: Sampled forward pass in the tile rasterizer (CUDA).** Implement D1. Only threads whose
  pixel is in the sample set do work, and output goes to `[C, N]`. Python binding in
  `diff_gaussian_rasterization/__init__.py` and `rasterize_points.cu`. *Done when:* B3's sampled-mode
  test passes.
- [ ] **S3: Sampled backward pass in the tile rasterizer (CUDA).** Same as S2 for `backward.cu`.
  *Done when:* gradients match B3's full-image gradients with unsampled pixels' gradients set to
  zero.
- [ ] **S4: Sampled losses in `get_loss()`.** Index ground truth by `pixel_idx`, carry over the masks,
  and implement D2 for mapping. Behind a config flag; with the flag off, behavior is identical to the
  baseline.
- [ ] **S5: Sampling results.** Speed and accuracy vs. sample count on Replica. Measure warp
  efficiency / active threads with Nsight Compute to show the tile kernels are mostly idle under
  sampling, which is what motivates the pixel-based kernels.

### Milestone report (Nov 23)
- [ ] **M1:** Baseline profile (B2, E5), sampling results (S5), pixel-kernel design (D4) and progress.

## Phase 2: Pixel-based kernels (Nov 24 – Dec 6)

New extension in `rasterizer/pixel/` with its own Python package name (e.g.
`pixel_gaussian_rasterization`), so it can be installed next to the baseline. Preprocessing and
sorting can be copied from the baseline at first.

- [ ] **P0: Scaffold `rasterizer/pixel/`.** Copy the baseline, rename the package/module, and make the
  SLAM loop choose a rasterizer with a config option (`rasterizer: baseline | pixel`). *Done when:*
  the unmodified copy passes B3 against the baseline.
- [ ] **P1: Pixel-based forward kernel.** One warp per sampled pixel; the warp's lanes split up that
  pixel's Gaussians. Blending is front-to-back, so combine lane results with a warp-level scan of
  transmittance (or process in chunks of 32 with a shuffle-based prefix product). Early transparency
  checks and early termination for the whole warp once the pixel is opaque. Implements D4.
  *Done when:* matches the baseline to within tolerance in B3.
- [ ] **P2: Pixel-based backward kernel.** One warp per sampled pixel. Recompute or load the
  forward pass's saved state, and compute per-Gaussian gradients. Combine within the warp (shuffles)
  before `atomicAdd` to global gradients. *Done when:* gradients match in B3.
- [ ] **P3: Integrate into tracking and mapping.** Use P1/P2 for both render passes, and run full
  SLAM on room0. *Done when:* ATE/PSNR are within the proposal's tolerances (ATE within 10%, PSNR within
  1 dB) of the sampled tile-based version.
- [ ] **P4: Kernel-level profiling.** Nsight Compute on P1/P2 vs. the tile kernels: occupancy, warp
  divergence, memory throughput, atomic stalls. This is the evidence for the "which factor limits
  performance" analysis.
- [ ] **P5 (if D3 = fuse): Fused pose gradient.** Compute the pose gradient in the backward kernel
  with a warp-level reduction, then a block-level reduction, then one atomic per block.

**Fallback:** if P2 isn't working by about Dec 3, use P1 with the S3 backward pass (as stated in the proposal).

## Phase 3: Evaluation & deliverables (Dec 7–11)

- [ ] **R1: Final benchmark matrix on Delta** (exclusive node). Variants: baseline, +sampling,
  +pixel forward, +pixel backward (+ stretch goals). Datasets: Replica (all 8 scenes) and TUM (fr1/desk,
  fr2/xyz, fr3/office). Uses B4.
- [ ] **R2: Figures.** Speedup per optimization, per-phase time breakdown, accuracy vs. sample count,
  and tile vs. pixel kernel metrics.
- [ ] **R3: Demo.** Baseline and optimized versions reconstructing a Replica scene side by side, with
  FPS shown.
- [ ] **R4: Final report and poster.**

## Stretch goals

- [ ] **X1: Merged render passes.** Render RGB and depth/silhouette/depth² in one pass (6 channels),
  so preprocessing, sorting and the Gaussian loads are shared.
- [ ] **X2: Concurrent tracking and mapping** on separate CUDA streams.
- [ ] **X3: iPhone lidar scene** for the demo (`scripts/iphone_demo.py` already exists).

---

## Dependencies

```
E1 → E2 → E3 → E5          B3 (no deps) ─────────┐
          E4 ──→ E5, B2    B1 → B2, B4           │
D1 → S2 → S3 → S4 → S5     S1 → S4               ├→ P1, P2 checked against baseline
D4 → P0 → P1 → P2 → P3 → P4 → R1 → R2/R3/R4      │
                     └──────────────────────────────┘
```

Tasks that can start right now in parallel: **E1/E2/E3**, **B3** (needs only a build), and
reading the rasterizer for **D1/D4**.

## Conventions

- Branch per task, PR into `main`, and the other person skims it before merging.
- `main` must always build and pass `tests/test_rasterizer.py`.
- Machine-specific paths go in `configs/local/` (git-ignored) or environment variables, never in code.
- Timing numbers in the report come only from Delta. Other machines are for correctness only.
