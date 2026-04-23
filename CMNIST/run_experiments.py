#!/usr/bin/env python3
"""
Colab-friendly runner for all CMNIST experiments (binary N-domain + MC-CMNIST).

Experiment groups
-----------------
Binary CMNIST (train.py):
  reproduce    — original 2-domain, ERM/EQRM/IRM/VREx, 3 seeds
  scaling      — N=2,3,4 in [0.1,0.2], EQRM alpha sweep + ERM
  spread       — N=3 fixed, narrow/medium/wide coverage

MC-CMNIST (train_mc.py, 10-class 10-color):
  mc_reproduce — original 2-domain (p=0.1,0.2), same algorithms
  mc_scaling   — N=2,3,4 in [0.1,0.2], EQRM alpha sweep + ERM
  mc_spread    — N=3 fixed, narrow/medium/wide coverage

Workflow
--------
Step 1  Generate ALL command files:
    python run_experiments.py generate \\
        --data_dir /content/data \\
        --output_dir /content/results

Step 2  Run individual groups (can split across Colab sessions):
    python run_experiments.py run --group reproduce
    python run_experiments.py run --group scaling
    python run_experiments.py run --group spread
    python run_experiments.py run --group mc_reproduce
    python run_experiments.py run --group mc_scaling
    python run_experiments.py run --group mc_spread

    # Or run everything in one call (~6-7 hours on T4):
    python run_experiments.py run --group all

Step 3  Generate figures:
    python plot_results.py \\
        --output_dir /content/results \\
        --figures_dir /content/results/figures

Step 4  (optional) Print results tables:
    python run_experiments.py collect \\
        --output_dir /content/results --group reproduce
"""

import argparse
import subprocess
import sys
import os
import time


# ---------------------------------------------------------------------------
# Group registry
# ---------------------------------------------------------------------------

# Binary CMNIST groups
BINARY_GROUPS = ["reproduce", "burnin", "spurious"]

# MC-CMNIST groups (10-class, 10-color)
MC_GROUPS = ["mc_reproduce"]

# BIM groups (Binary Interleaved M2M, non-linear color shortcut)
BIM_GROUPS = ["bim_reproduce"]

# CFashionMNIST groups (binary, Red/Green, harder base dataset)
CFMNIST_GROUPS = ["cfmnist_reproduce"]

# RMNIST groups (Binary Rotated MNIST, continuous geometric shortcut)
RMNIST_GROUPS = ["rmnist_reproduce", "rmnist_grid"]

ALL_GROUPS = BINARY_GROUPS + MC_GROUPS + BIM_GROUPS + CFMNIST_GROUPS + RMNIST_GROUPS

GROUP_FILE = {
    # binary CMNIST
    "reproduce":          "job_scripts/ndomain_reproduce.txt",
    "burnin":             "job_scripts/burnin.txt",
    "spurious":           "job_scripts/spurious.txt",
    # MC-CMNIST
    "mc_reproduce":       "job_scripts/mc_reproduce.txt",
    # BIM
    "bim_reproduce":      "job_scripts/bim_reproduce.txt",
    # CFashionMNIST
    "cfmnist_reproduce":  "job_scripts/cfmnist_reproduce.txt",
    # RMNIST
    "rmnist_reproduce":   "job_scripts/rmnist_reproduce.txt",
    "rmnist_grid":        "job_scripts/rmnist_grid.txt",
}

GROUP_EXP_NAME = {
    "reproduce":          "reproduce",
    "burnin":             "burnin",
    "spurious":           "spurious",
    "mc_reproduce":       "mc_reproduce",
    "bim_reproduce":      "bim_reproduce",
    "cfmnist_reproduce":  "cfmnist_reproduce",
    "rmnist_reproduce":   "rmnist_reproduce",
    "rmnist_grid":        "rmnist_grid",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(cmd, verbose=True):
    if verbose:
        print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARNING] exit code {result.returncode}: {cmd}", file=sys.stderr)
    return result.returncode


def load_commands(filepath):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def run_group(group):
    txt_file = GROUP_FILE[group]
    if not os.path.exists(txt_file):
        print(f"Command file not found: {txt_file}")
        print("Run  `python run_experiments.py generate ...`  first.")
        sys.exit(1)

    cmds = load_commands(txt_file)
    print(f"\n{'='*60}")
    print(f"Running group '{group}': {len(cmds)} commands")
    print(f"{'='*60}")

    failed = []
    t0 = time.time()
    for i, cmd in enumerate(cmds):
        elapsed = time.time() - t0
        print(f"\n[{i+1}/{len(cmds)}]  elapsed={elapsed:.0f}s")
        rc = run_command(cmd)
        if rc != 0:
            failed.append((i + 1, cmd))
        rate = (time.time() - t0) / (i + 1)
        eta  = rate * (len(cmds) - i - 1)
        print(f"  done  |  ETA ~{eta/60:.1f} min")

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Group '{group}' finished in {total_min:.1f} min")
    if failed:
        print(f"  {len(failed)} FAILED:")
        for idx, c in failed:
            print(f"    [{idx}] {c}")
    else:
        print("  All commands succeeded.")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_generate(args):
    """Generate command files for binary, MC-CMNIST, and/or BIM experiments."""
    # Binary CMNIST
    if args.dataset in ("binary", "all"):
        cmd = (
            f"python -m job_scripts.gen_exps_ndomain "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--n_seeds {args.n_seeds}"
        )
        run_command(cmd)

    # Burn-in ablation
    if args.dataset in ("burnin", "all"):
        cmd = (
            f"python -m job_scripts.gen_exps_burnin "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--n_seeds {args.n_seeds}"
        )
        run_command(cmd)

    # Spurious strength ablation
    if args.dataset in ("spurious", "all"):
        cmd = (
            f"python -m job_scripts.gen_exps_spurious "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--n_seeds {args.n_seeds}"
        )
        run_command(cmd)

    # MC-CMNIST
    if args.dataset in ("mc", "all"):
        cmd = (
            f"python -m job_scripts.gen_exps_mc "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--n_seeds {args.n_seeds}"
        )
        run_command(cmd)

    # BIM
    if args.dataset in ("bim", "all"):
        cmd = (
            f"python -m job_scripts.gen_exps_bim "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--n_seeds {args.n_seeds}"
        )
        run_command(cmd)

    # CFashionMNIST
    if args.dataset in ("cfmnist", "all"):
        cmd = (
            f"python -m job_scripts.gen_exps_cfmnist "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--n_seeds {args.n_seeds}"
        )
        run_command(cmd)

    # RMNIST
    if args.dataset in ("rmnist", "all"):
        cmd = (
            f"python -m job_scripts.gen_exps_rmnist "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--n_seeds {args.n_seeds}"
        )
        run_command(cmd)

    # RMNIST grid (N × Spread)
    if args.dataset in ("rmnist_grid", "all"):
        cmd = (
            f"python -m job_scripts.gen_exps_rmnist_grid "
            f"--data_dir {args.data_dir} "
            f"--output_dir {args.output_dir} "
            f"--n_seeds {args.n_seeds}"
        )
        run_command(cmd)


def cmd_run(args):
    groups = ALL_GROUPS if args.group == "all" else [args.group]
    for group in groups:
        run_group(group)


def cmd_collect(args):
    group    = args.group
    exp_name = GROUP_EXP_NAME.get(group, group)
    results_dir = os.path.join(args.output_dir, "results", exp_name)

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    cmd = (
        f"python collect_results.py {results_dir} "
        f"--model_selection_type best"
    )
    if "reproduce" in group:
        cmd += " --test_envs_print 0.1,0.9"
    run_command(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runner for binary + MC-CMNIST + BIM experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- generate ----
    p_gen = sub.add_parser("generate", help="Generate command files.")
    p_gen.add_argument("--data_dir",   required=True)
    p_gen.add_argument("--output_dir", required=True)
    p_gen.add_argument("--n_seeds",    type=int, default=3)
    p_gen.add_argument("--dataset",    default="all",
                       choices=["binary", "mc", "bim", "cfmnist", "rmnist", "rmnist_grid", "burnin", "spurious", "all"],
                       help="Which dataset's commands to generate.")

    # ---- run ----
    p_run = sub.add_parser("run", help="Run an experiment group.")
    p_run.add_argument(
        "--group", default="all",
        choices=ALL_GROUPS + ["all"],
        help=(
            "Group to run. "
            "Binary: reproduce / scaling / spread. "
            "MC-CMNIST: mc_reproduce / mc_scaling / mc_spread. "
            "BIM: bim_reproduce / bim_scaling / bim_spread. "
            "CFashionMNIST: cfmnist_reproduce / cfmnist_scaling / cfmnist_spread. "
            "RMNIST: rmnist_reproduce / rmnist_scaling / rmnist_spread. "
            "Or 'all' for everything."
        ),
    )

    # ---- collect ----
    p_col = sub.add_parser("collect", help="Print results tables.")
    p_col.add_argument("--output_dir", required=True)
    p_col.add_argument("--group", default="reproduce",
                       choices=ALL_GROUPS)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "collect":
        cmd_collect(args)
