# Environment Setup (federer)

How to set up a development environment on `federer.s3d.cmu.edu`. The same steps should work on
Delta later, with different paths and with `module load` instead of `/usr/local/cuda-12.4`.

## About the machine

| | |
|---|---|
| GPUs | 4× A100 80GB (GPUs 0–3) and 4× A800 40GB (GPUs 4–7). All are Ampere `sm_80`, the same architecture as Delta's A100s |
| Driver | 550.144, which supports CUDA **≤ 12.4** |
| CUDA toolkit | `/usr/local/cuda-12.4`. **`/usr/bin/nvcc` is an older 11.5. Don't use it** |
| OS / compiler | Ubuntu 22.04, gcc 11.4 |
| Storage | `/` (including `/tmp`) is 98% full and `/home` is 92% full. **Everything goes under `/work8/$USER`**: code, conda env, datasets, outputs, caches and temp files. The only things written to home are a few KB (an SSH key and conda's `~/.conda/environments.txt`) |
| Scheduler | none. The GPUs are shared with other users, and you choose one yourself (see below) |

You need a federer account and a `/work8/$USER` directory. Ask the machine's admin if you don't
have them.

## 1. Workspace and environment variables

Create an env file that you `source` in every session:

```bash
export WORK=/work8/$USER/gslam
mkdir -p $WORK
cat > $WORK/env.sh <<'EOF'
export WORK=/work8/$USER/gslam          # same path as above
# CUDA 12.4 toolkit, ahead of the old /usr/bin/nvcc (11.5)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Build the rasterizer for Ampere only (A100 / A800, and Delta's A100s)
export TORCH_CUDA_ARCH_LIST="8.0"
# Keep temp files and caches on /work8 (/ has ~1 GB free and /home is nearly full)
export TMPDIR=$WORK/tmp                  # pip/conda unpack multi-GB packages here
mkdir -p $TMPDIR
export XDG_CACHE_HOME=$WORK/.cache       # default ~/.cache (torch extensions, etc.)
export MPLCONFIGDIR=$WORK/.cache/matplotlib
export PIP_CACHE_DIR=$WORK/.cache/pip
export TORCH_HOME=$WORK/.cache/torch     # LPIPS downloads model weights here
export CUDA_CACHE_PATH=$WORK/.cache/nv
# SplaTAM configs have use_wandb=True; this turns wandb off without editing them
export WANDB_MODE=disabled
EOF
source $WORK/env.sh
```

Check it: `which nvcc` should print `/usr/local/cuda-12.4/bin/nvcc`.

## 2. Conda

If you already have conda installed under `/work8` (`which conda` prints a `/work8/...` path), use
it, and the `splatam` environment will live there too. Otherwise, install Miniforge into the work
directory:

```bash
cd $WORK
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $WORK/miniforge3
echo 'source $WORK/miniforge3/etc/profile.d/conda.sh' >> $WORK/env.sh
source $WORK/env.sh
```

Either way, check that conda puts environments and its package cache on `/work8`:

```bash
conda config --show envs_dirs pkgs_dirs     # the FIRST entry of each must be under /work8
```

## 3. Python environment

```bash
conda create -y -n splatam python=3.10       # 3.10 is required by open3d==0.16.0
conda activate splatam
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: 2.5.1+cu124 12.4 True
```

We use PyTorch 2.5.1 + CUDA 12.4 because it matches both the driver and the installed toolkit.
SplaTAM was benchmarked with PyTorch 1.12.1 + CUDA 11.6, but its README says it has also been
tested with PyTorch 2.x.

## 4. Code and dependencies

```bash
cd $WORK
git clone git@github.com:oliveryun11/gaussian-slam.git
cd gaussian-slam
conda activate splatam       # your prompt must say (splatam), not (base)
pip install -r requirements.txt "numpy<2"
```

`numpy<2` goes in the same command because `open3d==0.16.0` is built against NumPy 1.x. Installing
it together lets pip choose versions of the other packages (e.g. OpenCV) that also work with NumPy 1.x.
If pip says there's no `open3d==0.16.0`, you're in the wrong environment: that version only exists
for Python ≤ 3.10.

You need an SSH key on federer that's added to your GitHub account (`ssh-keygen -t ed25519`,
then add `~/.ssh/id_ed25519.pub` under GitHub → Settings → SSH keys).

## 5. Build the rasterizer

```bash
pip install --no-build-isolation ./rasterizer/baseline
python -c "import torch, diff_gaussian_rasterization; print('rasterizer ok')"
```

`--no-build-isolation` is required. `setup.py` imports `torch`, and without this flag pip builds in
a clean environment that doesn't have torch.

**Rebuild after every CUDA change:** editing a `.cu` file does nothing until you run the
`pip install` command again. For a full rebuild, delete `rasterizer/baseline/build/` first.

## 6. Datasets

The download scripts write into `./data/`, which is git-ignored. Run them from the repo root:

```bash
bash bash_scripts/download_replica.sh     # Replica (large; takes a while)
bash bash_scripts/download_tum.sh         # TUM-RGBD (5 sequences)
```

After unzipping, you can delete the `.zip`/`.tgz` files to save space.

## 7. Choosing a GPU

federer has no scheduler, so check which GPUs are free before every run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=6       # an idle GPU; your process will see it as cuda:0
```

Timings are only meaningful on a GPU no one else is using. Note in the results which GPU (A100 or
A800) a timing came from.

## 8. Smoke test

Make a short local config (`configs/local/` is git-ignored):

```bash
mkdir -p configs/local
cp configs/replica/splatam.py configs/local/replica_dev.py
# in configs/local/replica_dev.py, set  num_frames=100  inside the data=dict(...) block
python scripts/splatam.py configs/local/replica_dev.py
```

It should track and map 100 frames of room0, then print ATE and PSNR. Results go to
`experiments/Replica/room0_0/`.

For long runs, use `tmux` so the run survives an SSH disconnect: `tmux new -s run`, start the
command, detach with `Ctrl-b d`, and reattach with `tmux attach -t run`.

## Every session

```bash
source /work8/$USER/gslam/env.sh        # your work directory
conda activate splatam
cd $WORK/gaussian-slam
export CUDA_VISIBLE_DEVICES=<free GPU>
```

## Cleanup (end of semester)

```bash
conda env remove -n splatam
rm -rf /work8/$USER/gslam
```

Also remove the federer SSH key from your GitHub account if you added one just for this project.
