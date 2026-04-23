#!/usr/bin/env python3
"""
Generate commands for the EQRM burn-in ablation on Binary CMNIST.

Experimental design
-------------------
Sweeps erm_pretrain_iters (ERM burn-in steps) to isolate how much
ERM initialisation EQRM needs to perform well.

  Algorithm : EQRM  (alpha = -10000)
  Dataset   : Binary CMNIST, train_envs = default (p=0.1, 0.2), OOD p=0.9
  Steps     : 600 total per run
  Seeds     : 3

Burn-in sweep (erm_pretrain_iters → fraction of total steps):
  0   →  0%   Pure EQRM (no ERM warm-up)
  150 → 25%
  300 → 50%
  450 → 75%
  600 → 100%  Pure ERM baseline (EQRM phase never runs)

All runs use --save_ckpts so checkpoints are reused across seeds that
share the same (train_envs, erm_pretrain_iters, exp_name, lr, ...).

Output
------
  job_scripts/burnin.txt

Usage (from CMNIST/ directory):
    python -m job_scripts.gen_exps_burnin \\
        --data_dir /content/data \\
        --output_dir /content/results
"""

import argparse
import os


PRETRAIN_VALUES = [0, 150, 300, 450, 600]
TOTAL_STEPS     = 600
ALPHA           = -10000


def write_cmds(cmds, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(cmds))
    print(f"  Wrote {len(cmds):4d} commands -> {filepath}")
    return len(cmds)


def gen_burnin(base_call, seeds, output_file):
    """
    Command count per seed : 5 burn-in values
    Grand total (3 seeds)  : 15
    """
    cmds = []
    for seed in seeds:
        seed_base = f"{base_call} --seed {seed} --exp_name burnin"

        for pretrain in PRETRAIN_VALUES:
            cmd = (
                f"{seed_base} --algorithm eqrm --train_envs default "
                f"--steps {TOTAL_STEPS} --erm_pretrain_iters {pretrain} "
                f"--alpha {ALPHA} --save_ckpts"
            )
            # Use cosine LR schedule for the EQRM phase whenever there is one
            if 0 < pretrain < TOTAL_STEPS:
                cmd += " --lr_cos_sched"
            cmds.append(cmd)

    return write_cmds(cmds, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate commands for the EQRM burn-in ablation."
    )
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_seeds",    type=int, default=3)
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))

    base_call = (
        f"python train.py "
        f"--data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--lr 1e-4 "
        f"--batch_size 25000 "
        f"--dropout_p 0.2"
    )

    print(f"\nGenerating burn-in ablation commands (seeds={seeds}) ...")
    total = gen_burnin(base_call, seeds, "job_scripts/burnin.txt")
    print(f"\nTotal commands: {total}")

    print("\nBurn-in sweep summary:")
    print(f"  {'pretrain_iters':>14}  {'fraction':>8}  {'label'}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*20}")
    for p in PRETRAIN_VALUES:
        frac = p / TOTAL_STEPS
        label = ("Pure EQRM (no warm-up)" if p == 0
                 else "Pure ERM baseline" if p == TOTAL_STEPS
                 else f"{frac*100:.0f}% ERM warm-up")
        print(f"  {p:>14}  {frac:>7.0%}  {label}")
    print(f"\nAlgorithm: EQRM  alpha={ALPHA}  steps={TOTAL_STEPS}")
    print(f"Dataset:   Binary CMNIST  train_envs=default (p=0.1, 0.2)  OOD p=0.9")
