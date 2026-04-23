#!/usr/bin/env python3
"""
Generate sweep commands for N-domain CMNIST experiments.

Experiments
-----------
1. reproduce      — Validate setup against the paper's 2-domain results.
                    Algorithms: ERM, Oracle, EQRM (5 alphas), IRM, VREx (2 penalties each).
                    3 seeds.

2. ndomain_scaling — Fixed domain range [0.1, 0.2], vary number of environments N ∈ {2,3,4,5,6}.
                     Algorithms: ERM + EQRM (all 5 alphas).
                     3 seeds.  Same exp_name so ERM checkpoints are shared across alpha values.

3. ndomain_spread  — Fix N=3, vary spread: narrow / medium / wide.
                     Algorithms: ERM + EQRM (alpha=-1000).
                     3 seeds.

Usage (from the CMNIST/ directory):
    python -m job_scripts.gen_exps_ndomain \\
        --data_dir /content/data \\
        --output_dir /content/results

This writes:
    job_scripts/ndomain_reproduce.txt
    job_scripts/ndomain_scaling.txt
    job_scripts/ndomain_spread.txt
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evenly_spaced_envs(n, lo=0.1, hi=0.2):
    """Return n evenly-spaced flip probabilities in [lo, hi], rounded to 3 dp."""
    if n == 1:
        return [round(lo, 3)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 3) for i in range(n)]


def envs_to_str(envs):
    """Convert a list of floats to the comma-separated string expected by train.py."""
    return ",".join(str(p) for p in envs)


def write_cmds(cmds, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(cmds))
    print(f"  Wrote {len(cmds):4d} commands -> {filepath}")
    return len(cmds)


# ---------------------------------------------------------------------------
# Experiment generators
# ---------------------------------------------------------------------------

def gen_reproduce(base_call, seeds, output_file):
    """Exp 1: reproduce original 2-domain results.

    Mirrors the paper's setup (train_envs=0.1,0.2) for key algorithms so we
    can confirm our implementation matches before running new experiments.
    """
    alphas        = [-100, -500, -1000, -5000, -10000]
    irm_penalties = [1000, 10000]
    vrex_penalties = [1000, 10000]

    cmds = []
    for seed in seeds:
        seed_base = f"{base_call} --seed {seed} --exp_name reproduce"

        # ERM (standard 2-domain training)
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs default "
            f"--steps 400 --erm_pretrain_iters 0"
        )

        # Oracle: grayscale (color uninformative ⇒ must use shape)
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs gray "
            f"--steps 400 --erm_pretrain_iters 0"
        )

        # EQRM — full alpha sweep
        for alpha in alphas:
            cmds.append(
                f"{seed_base} --algorithm eqrm --train_envs default "
                f"--steps 600 --erm_pretrain_iters 400 --lr_cos_sched "
                f"--alpha {alpha} --save_ckpts"
            )

        # IRM
        for pen in irm_penalties:
            cmds.append(
                f"{seed_base} --algorithm irm --train_envs default "
                f"--steps 600 --erm_pretrain_iters 400 --lr_cos_sched "
                f"--penalty_weight {pen} --save_ckpts"
            )

        # VREx
        for pen in vrex_penalties:
            cmds.append(
                f"{seed_base} --algorithm vrex --train_envs default "
                f"--steps 600 --erm_pretrain_iters 400 --lr_cos_sched "
                f"--penalty_weight {pen} --save_ckpts"
            )

    return write_cmds(cmds, output_file)


def gen_scaling(base_call, seeds, output_file):
    """Exp 2+3: N-domain scaling (fixed range [0.1, 0.2]) × alpha sweep.

    Addresses Theorem 4.1: do more training domains improve EQRM's quantile
    estimates and thus its OOD accuracy?

    All runs share exp_name='ndomain_scaling' so that ERM checkpoints are
    reused across EQRM alpha values for the same (N, seed).

    Domain configurations (evenly spaced in [0.1, 0.2]):
        N=2 → [0.1, 0.2]
        N=3 → [0.1, 0.15, 0.2]
        N=4 → [0.1, 0.133, 0.167, 0.2]
    """
    n_values = [2, 3, 4]
    alphas   = [-100, -500, -1000, -5000, -10000]

    cmds = []
    for seed in seeds:
        for n in n_values:
            envs_str  = envs_to_str(evenly_spaced_envs(n))
            seed_base = (
                f"{base_call} --seed {seed} --exp_name ndomain_scaling "
                f"--train_envs {envs_str}"
            )

            # ERM baseline — plain interleaved training, no algorithm-specific objective
            cmds.append(
                f"{seed_base} --algorithm erm "
                f"--steps 400 --erm_pretrain_iters 0"
            )

            # EQRM — full alpha sweep; ERM checkpoint is shared across alphas
            for alpha in alphas:
                cmds.append(
                    f"{seed_base} --algorithm eqrm "
                    f"--steps 600 --erm_pretrain_iters 400 --lr_cos_sched "
                    f"--alpha {alpha} --save_ckpts"
                )

    return write_cmds(cmds, output_file)


def gen_spread(base_call, seeds, output_file):
    """Exp 4: domain spread effect (N=3 fixed, vary spatial coverage).

    Tests whether the *diversity* of training environments matters for EQRM
    beyond the number of environments.

    Configurations (all N=3):
        narrow  → [0.1, 0.15, 0.2]  same range as original 2-domain + midpoint
        medium  → [0.1, 0.2,  0.3]  modest extension
        wide    → [0.1, 0.25, 0.4]  full range (= N=3 from scaling exp)
    """
    spread_configs = {
        "narrow": "0.1,0.15,0.2",
        "medium": "0.1,0.2,0.3",
        "wide":   "0.1,0.25,0.4",
    }
    alpha_eqrm = -1000  # single canonical alpha to isolate the spread effect

    cmds = []
    for seed in seeds:
        for spread_name, envs_str in spread_configs.items():
            seed_base = (
                f"{base_call} --seed {seed} --exp_name ndomain_spread "
                f"--train_envs {envs_str}"
            )

            # ERM
            cmds.append(
                f"{seed_base} --algorithm erm "
                f"--steps 400 --erm_pretrain_iters 0"
            )

            # EQRM
            cmds.append(
                f"{seed_base} --algorithm eqrm "
                f"--steps 600 --erm_pretrain_iters 400 --lr_cos_sched "
                f"--alpha {alpha_eqrm} --save_ckpts"
            )

    return write_cmds(cmds, output_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate commands for N-domain CMNIST experiments."
    )
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Absolute path to data directory.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Absolute path to output directory (logs, results, ckpts).")
    parser.add_argument("--exp_group",  type=str, default="all",
                        choices=["reproduce", "scaling", "spread", "all"],
                        help="Which experiment group to generate (default: all).")
    parser.add_argument("--n_seeds",    type=int, default=3,
                        help="Number of seeds (default: 3).")
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))

    # Shared training hyperparameters — kept identical to the original paper
    base_call = (
        f"python train.py "
        f"--data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--lr 1e-4 "
        f"--batch_size 25000 "
        f"--dropout_p 0.2"
    )

    print(f"\nGenerating commands (seeds={seeds}) ...")
    total = 0

    if args.exp_group in ("reproduce", "all"):
        total += gen_reproduce(base_call, seeds,
                               "job_scripts/ndomain_reproduce.txt")

    if args.exp_group in ("scaling", "all"):
        total += gen_scaling(base_call, seeds,
                             "job_scripts/ndomain_scaling.txt")

    if args.exp_group in ("spread", "all"):
        total += gen_spread(base_call, seeds,
                            "job_scripts/ndomain_spread.txt")

    print(f"\nTotal commands: {total}")
    print(
        "\nDomain configurations for scaling experiment (evenly spaced in [0.1, 0.2]):"
    )
    for n in [2, 3, 4, 5]:
        envs = evenly_spaced_envs(n)
        samples_per_env = 50000 // n
        print(f"  N={n}: {envs}  ({samples_per_env} images/env)")
