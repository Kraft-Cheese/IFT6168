#!/usr/bin/env python3
"""
Generate commands for the spurious strength ablation on Binary CMNIST.

Experimental design
-------------------
Sweeps the *average* spurious correlation across training environments from
strong (0.90) down to random-noise baseline (0.50) to identify the phase
transition where simplicity bias defeats invariant learning.

The causal ceiling is 75% (25% label-flip noise → shape is 75% predictive).
Phase transition is therefore expected near mean_e ≈ 0.75.

Phase 1 — Spurious > Causal  (mean_e > 0.75):
  mean=0.90  e1=0.05, e2=0.15
  mean=0.85  e1=0.10, e2=0.20   ← standard reproduce config
  mean=0.80  e1=0.15, e2=0.25

Phase 2 — Spurious == Causal  (mean_e = 0.75):
  mean=0.75  e1=0.20, e2=0.30

Phase 3 — Spurious < Causal  (mean_e < 0.75):
  mean=0.70  e1=0.25, e2=0.35
  mean=0.60  e1=0.35, e2=0.45
  mean=0.50  e1=0.45, e2=0.55   ← color is pure noise

Algorithms
----------
  ERM   : 400 steps, no pretrain
  EQRM  : 600 steps, 400-step ERM warm-up, alpha=-10000, cosine LR
  Oracle: train_envs=gray (p=0.5,0.5), ERM — color uninformative

Command count per seed:
  Oracle       :  1
  ERM × 7      :  7
  EQRM × 7     :  7
  ─────────────
  Total / seed : 15      Grand total (3 seeds): 45

Output
------
  job_scripts/spurious.txt

Usage (from CMNIST/ directory):
    python -m job_scripts.gen_exps_spurious \\
        --data_dir /content/data \\
        --output_dir /content/results
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

# (mean_spurious, (e1, e2))  — ordered from strongest to weakest shortcut
ENV_PAIRS = [
    (0.90, (0.05, 0.15)),
    (0.85, (0.10, 0.20)),
    (0.80, (0.15, 0.25)),
    (0.75, (0.20, 0.30)),
    (0.70, (0.25, 0.35)),
    (0.60, (0.35, 0.45)),
    (0.50, (0.45, 0.55)),
]

CAUSAL_CEILING = 0.75   # 1 - label_noise (25% flip)
ALPHA          = -10000
EQRM_STEPS     = 600
ERM_PRETRAIN   = 400
ERM_STEPS      = 400


def write_cmds(cmds, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(cmds))
    print(f"  Wrote {len(cmds):4d} commands -> {filepath}")
    return len(cmds)


def gen_spurious(base_call, seeds, output_file):
    cmds = []
    for seed in seeds:
        seed_base = f"{base_call} --seed {seed} --exp_name spurious"

        # Oracle (color uninformative — upper bound on shape learning)
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs gray "
            f"--steps {ERM_STEPS} --erm_pretrain_iters 0"
        )

        for mean_e, (e1, e2) in ENV_PAIRS:
            envs_str = f"{e1},{e2}"
            cfg_base = f"{seed_base} --train_envs {envs_str}"

            # ERM
            cmds.append(
                f"{cfg_base} --algorithm erm "
                f"--steps {ERM_STEPS} --erm_pretrain_iters 0"
            )

            # EQRM
            cmds.append(
                f"{cfg_base} --algorithm eqrm "
                f"--steps {EQRM_STEPS} --erm_pretrain_iters {ERM_PRETRAIN} "
                f"--lr_cos_sched --alpha {ALPHA} --save_ckpts"
            )

    return write_cmds(cmds, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate commands for the spurious strength ablation."
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

    print(f"\nGenerating spurious strength ablation commands (seeds={seeds}) ...")
    total = gen_spurious(base_call, seeds, "job_scripts/spurious.txt")
    print(f"\nTotal commands: {total}")

    print("\nEnvironment sweep summary:")
    print(f"  {'mean_e':>6}  {'(e1, e2)':>14}  {'phase'}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*30}")
    for mean_e, (e1, e2) in ENV_PAIRS:
        if mean_e > CAUSAL_CEILING:
            phase = "Phase 1 — Spurious > Causal"
        elif mean_e == CAUSAL_CEILING:
            phase = "Phase 2 — Spurious == Causal"
        else:
            phase = "Phase 3 — Spurious < Causal"
        tag = "  ← standard reproduce" if (e1, e2) == (0.10, 0.20) else ""
        print(f"  {mean_e:>6.2f}  ({e1:.2f}, {e2:.2f})      {phase}{tag}")

    cmds_per_seed = 1 + len(ENV_PAIRS) * 2
    print(f"\nCommands per seed: 1 oracle + {len(ENV_PAIRS)} × (ERM + EQRM) = {cmds_per_seed}")
    print(f"Grand total ({len(seeds)} seeds): {cmds_per_seed * len(seeds)}")
