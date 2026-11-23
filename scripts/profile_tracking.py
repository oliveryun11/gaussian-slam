"""Profile Replica room0 at several tracking pixel coverage levels."""

import argparse
import csv
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
KERNEL_QUERY = """
SELECT CASE WHEN s.value LIKE '%int, int, const float *, const float2 *,%'
            THEN 'backward' ELSE 'forward' END,
       COUNT(*), AVG(k.end - k.start) / 1e6
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id = k.demangledName
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
WHERE s.value LIKE '%renderCUDA%'
  AND EXISTS (
    SELECT 1 FROM NVTX_EVENTS n
    WHERE n.text = 'tracking_iter' AND r.start BETWEEN n.start AND n.end
  )
GROUP BY 1
"""
LOG_METRICS = {
    "slam_seconds": "Total SLAM time",
    "tracking_seconds_per_frame": "Average Tracking/Frame Time",
    "ate_cm": "Final Average ATE RMSE",
    "psnr_db": "Average PSNR",
}


def summarize(database, log, coverage):
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
        kernels = {name: (count, mean) for name, count, mean in connection.execute(KERNEL_QUERY)}
    if kernels.keys() != {"forward", "backward"}:
        raise ValueError(f"Missing tracking render kernels in {database}")
    result = {"coverage": coverage}
    for name in ("forward", "backward"):
        result[f"{name}_launches"], result[f"{name}_mean_ms"] = kernels[name]
    output = log.read_text(errors="replace")
    for field, label in LOG_METRICS.items():
        match = re.search(rf"^{re.escape(label)}: ([0-9.eE+-]+)", output, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing {label} in {log}")
        result[field] = float(match.group(1))
    return result


def run(command, log, env):
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for configs, traces and CSV")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/replica/splatam.py")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--end", type=int, default=90)
    parser.add_argument("--coverage", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.1])
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--ncu", help="Optional Nsight Compute executable for a separate counter pass")
    args = parser.parse_args()
    if not 1 <= args.start < args.end <= args.frames:
        parser.error("require 1 <= start < end <= frames")
    if len(set(args.coverage)) != len(args.coverage) or any(not 0 < x <= 1 for x in args.coverage):
        parser.error("coverage values must be unique and in (0, 1]")

    output = args.output.resolve()
    output.mkdir(parents=True)
    env = dict(os.environ, PROFILE_START=str(args.start), PROFILE_END=str(args.end),
               WANDB_MODE="disabled", MPLBACKEND="Agg")
    profiles = []
    with (output / "summary.csv").open("w", newline="") as stream:
        for coverage in args.coverage:
            name = f"coverage-{coverage!r}"
            directory = output / name
            directory.mkdir()
            config = directory / "config.py"
            config.write_text(
                "from runpy import run_path\n"
                f"config = run_path({str(args.config.resolve())!r})['config']\n"
                f"config.update(workdir={str(directory)!r}, run_name='systems', "
                "use_wandb=False, load_checkpoint=False)\n"
                f"config['data']['num_frames'] = {args.frames}\n"
                f"config['tracking']['pixel_sample_ratio'] = {coverage!r}\n"
            )
            target = [sys.executable, "-u", str(ROOT / "scripts/splatam.py"), str(config)]
            trace = directory / "tracking.nsys-rep"
            run([args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                 "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                 "--output", str(trace.with_suffix("")), *target], directory / "systems.log", env)
            database = trace.with_suffix(".sqlite")
            run([args.nsys, "export", "--type=sqlite", "--output", str(database), str(trace)],
                directory / "export.log", env)
            result = summarize(database, directory / "systems.log", coverage)
            result.update(capture_start=args.start, capture_end_exclusive=args.end, frames=args.frames)
            if not profiles:
                writer = csv.DictWriter(stream, fieldnames=result)
                writer.writeheader()
            writer.writerow(result)
            stream.flush()
            profiles.append((directory, config, target))

    if args.ncu:
        for directory, config, target in profiles:
            # Replay distorts timing, so collect counters in separate processes.
            counter_config = config.with_name("compute.py")
            counter_config.write_text(config.read_text() + "config['run_name'] = 'compute'\n")
            counter_report = directory / "active-lanes"
            run([args.ncu, "--profile-from-start", "off", "--nvtx", "--nvtx-include",
                 "tracking_iter/", "--kernel-name-base", "function", "--kernel-name",
                 "regex:renderCUDA", "--launch-count", "4", "--section", "SourceCounters",
                 "--export", str(counter_report), *target[:-1], str(counter_config)],
                directory / "compute.log", env)


if __name__ == "__main__":
    main()
