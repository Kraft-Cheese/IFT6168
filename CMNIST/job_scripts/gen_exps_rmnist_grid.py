#!/usr/bin/env python3
"""
Generate sweep commands for the RMNIST N × Spread experiment grid.

Experimental design
-------------------
Systematically varies two orthogonal axes:
  - N     : number of training domains {2, 4}
  - Spread: angular range of the spurious correlation (UltraNarrow / Narrow / Wide)

EXPERIMENT_GRID
---------------
  N2_UltraNarrow : [40.0, 50.0]                          range=10°
  N2_Narrow      : [30.0, 60.0]                          range=30°
  N2_Wide        : [15.0, 75.0]                          range=60°
  N4_UltraNarrow : [40.0, 43.3, 46.7, 50.0]             range=10°
  N4_Narrow      : [30.0, 40.0, 50.0, 60.0]             range=30°
  N4_Wide        : [15.0, 35.0, 55.0, 75.0]             range=60°

OOD test environment: delta=-60 (anti-correlated, mu_0=75°, mu_1=15°).

Algorithms
----------
  ERM   : plain empirical risk minimisation
  Oracle: train at delta=(0,0) — rotation uninformative, must use shape
  EQRM  : 1-alpha sweep {-10000}

Oracle is shared across grid entries (same train_envs='oracle' for all N),
so it is generated once per seed to avoid redundant runs.

ERM checkpoints are automatically reused across EQRM alpha values for runs
with the same (train_envs string, seed, exp_name).

Output
------
  job_scripts/rmnist_grid.txt

Usage (from CMNIST/ directory):
    python -m job_scripts.gen_exps_rmnist_grid \\
        --data_dir /content/data \\
        --output_dir /content/results
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Experiment grid definition  (matches user spec exactly)
# ---------------------------------------------------------------------------

EXPERIMENT_GRID = {
    "N2_UltraNarrow": [40.0, 50.0],
    "N2_Narrow":      [30.0, 60.0],
    "N2_Wide":        [15.0, 75.0],
    "N4_UltraNarrow": [40.0, 43.3, 46.7, 50.0],
    "N4_Narrow":      [30.0, 40.0, 50.0, 60.0],
    "N4_Wide":        [15.0, 35.0, 55.0, 75.0],
}

ALPHAS = [-10000]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def deltas_to_str(deltas):
    return ",".join(str(d) for d in deltas)


def write_cmds(cmds, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(cmds))
    print(f"  Wrote {len(cmds):4d} commands -> {filepath}")
    return len(cmds)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def gen_rmnist_grid(base_call, seeds, output_file):
    """
    Generate all commands for the N × Spread grid.

    Command count per seed:
      Oracle         :  1  (shared across all N and spread types)
      ERM × 6 configs:  6
      EQRM × 1α × 6 :  6
      ─────────────────
      Total / seed   : 13      Grand total (2 seeds): 26
    """
    cmds = []

    for seed in seeds:
        seed_base = f"{base_call} --seed {seed} --exp_name rmnist_grid"

        # ---- Oracle (once per seed — same regardless of grid config) ----
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs oracle "
            f"--steps 400 --erm_pretrain_iters 0"
        )

        # ---- Grid configs ----
        for config_name, deltas in EXPERIMENT_GRID.items():
            deltas_str = deltas_to_str(deltas)
            cfg_base   = f"{seed_base} --train_envs {deltas_str}"

            # ERM
            cmds.append(
                f"{cfg_base} --algorithm erm "
                f"--steps 400 --erm_pretrain_iters 0"
            )

            # EQRM — full alpha sweep; ERM checkpoint reused across alphas
            for alpha in ALPHAS:
                cmds.append(
                    f"{cfg_base} --algorithm eqrm "
                    f"--steps 600 --erm_pretrain_iters 400 --lr_cos_sched "
                    f"--alpha {alpha} --save_ckpts"
                )

    return write_cmds(cmds, output_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate commands for the RMNIST N×Spread grid experiment."
    )
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_seeds",    type=int, default=3)
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))

    base_call = (
        f"python train_rmnist.py "
        f"--data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--lr 1e-4 "
        f"--batch_size 25000 "
        f"--dropout_p 0.2"
    )

    print(f"\nGenerating RMNIST grid commands (seeds={seeds}) ...")
    total = gen_rmnist_grid(base_call, seeds, "job_scripts/rmnist_grid.txt")
    print(f"\nTotal commands: {total}")

    print("\nExperiment grid summary:")
    print(f"  {'Config':<12}  {'N':>2}  {'Spread':>6}  {'Range':>7}  Deltas")
    print(f"  {'-'*12}  {'-'*2}  {'-'*6}  {'-'*7}  {'-'*40}")
    for name, deltas in EXPERIMENT_GRID.items():
        n      = len(deltas)
        spread = "_".join(name.split("_")[1:])   # e.g. "UltraNarrow", "Narrow", "Wide"
        rng    = f"{max(deltas)-min(deltas):.1f}°"
        print(f"  {name:<16}  {n:>2}  {spread:<12}  {rng:>7}  {deltas}")

    print(f"\nOOD test: delta=-60  (mu_0=75°, mu_1=15°, fully inverted)")
    print(f"Oracle  : delta=0,0  (rotation uninformative)")
    cmds_per_seed = 1 + len(EXPERIMENT_GRID) * (1 + len(ALPHAS))
    print(f"Commands per seed: 1 oracle + {len(EXPERIMENT_GRID)} × (1 ERM + {len(ALPHAS)} EQRM) = {cmds_per_seed}")
    print(f"Grand total ({len(seeds)} seeds): {cmds_per_seed * len(seeds)}")
