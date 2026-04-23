#!/usr/bin/env python3
"""
Generate sweep commands for Colored Fashion-MNIST (CFashionMNIST) experiments.

Experiment groups
-----------------
1. cfmnist_reproduce  — Validate setup on the original 2-domain configuration
                        (train: p=0.1, 0.2; test OOD: p=0.9).
                        Algorithms: ERM, Oracle, EQRM (5 alphas), IRM, VREx.
                        3 seeds.

2. cfmnist_scaling    — Fixed flip-prob range [0.1, 0.2], vary N in {2, 3, 4}.
                        Algorithms: ERM + EQRM (all 5 alphas).
                        3 seeds.  Same exp_name so ERM checkpoints are shared.

3. cfmnist_spread     — Fix N=3, vary spread of training flip probabilities:
                        narrow [0.1, 0.15, 0.2], medium [0.1, 0.2, 0.3],
                        wide   [0.1, 0.25, 0.4].
                        Algorithms: ERM + EQRM (alpha=-1000).
                        3 seeds.

Convention
----------
All settings are identical to binary CMNIST (gen_exps_ndomain.py) — the only
differences are the training script (train_cfmnist.py), the exp_name prefix
(cfmnist_*), and the oracle keyword ('oracle' maps to p=0.5,0.5 for Red/Green
uninformative rather than the gray=0.5,0.5 of binary CMNIST).

  Binary CMNIST : p_e = flip prob, uninformative at p=0.5 ('gray')
  CFashionMNIST : p_e = flip prob, uninformative at p=0.5 ('oracle')

Usage (from CMNIST/ directory):
    python -m job_scripts.gen_exps_cfmnist \\
        --data_dir /content/data \\
        --output_dir /content/results
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Helpers  (identical to gen_exps_ndomain.py)
# ---------------------------------------------------------------------------

def evenly_spaced_envs(n, lo=0.1, hi=0.2):
    """Return n evenly-spaced flip probabilities in [lo, hi]."""
    if n == 1:
        return [round(lo, 3)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 3) for i in range(n)]


def envs_to_str(envs):
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

def gen_cfmnist_reproduce(base_call, seeds, output_file):
    """
    Exp 1 (cfmnist_reproduce): validate CFashionMNIST on 2-domain config.

    Oracle trains at p=0.5,0.5 where Red and Green are equally likely
    regardless of label -> model must rely on clothing shape.
    """
    alphas         = [-100, -500, -1000, -5000, -10000]
    irm_penalties  = [1000, 10000]
    vrex_penalties = [1000, 10000]

    cmds = []
    for seed in seeds:
        seed_base = f"{base_call} --seed {seed} --exp_name cfmnist_reproduce"

        # ERM (default: train at p=0.1, 0.2 — color is a strong shortcut)
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs default "
            f"--steps 400 --erm_pretrain_iters 0"
        )

        # Oracle (train at p=0.5,0.5 — color uninformative -> must use shape)
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs oracle "
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


def gen_cfmnist_scaling(base_call, seeds, output_file):
    """
    Exp 2 (cfmnist_scaling): N-domain scaling within [0.1, 0.2] x alpha sweep.

    Tests whether EQRM's benefit from more training domains (Theorem 4.1)
    holds on Fashion-MNIST, a harder base dataset than MNIST.

    Domain configurations (evenly spaced in [0.1, 0.2]):
        N=2 -> [0.1, 0.2]
        N=3 -> [0.1, 0.15, 0.2]
        N=4 -> [0.1, 0.133, 0.167, 0.2]
    """
    n_values = [2, 3, 4]
    alphas   = [-100, -500, -1000, -5000, -10000]

    cmds = []
    for seed in seeds:
        for n in n_values:
            envs_str  = envs_to_str(evenly_spaced_envs(n))
            seed_base = (
                f"{base_call} --seed {seed} --exp_name cfmnist_scaling "
                f"--train_envs {envs_str}"
            )

            # ERM baseline
            cmds.append(
                f"{seed_base} --algorithm erm "
                f"--steps 400 --erm_pretrain_iters 0"
            )

            # EQRM — full alpha sweep; ERM checkpoint shared across alphas
            for alpha in alphas:
                cmds.append(
                    f"{seed_base} --algorithm eqrm "
                    f"--steps 600 --erm_pretrain_iters 400 --lr_cos_sched "
                    f"--alpha {alpha} --save_ckpts"
                )

    return write_cmds(cmds, output_file)


def gen_cfmnist_spread(base_call, seeds, output_file):
    """
    Exp 3 (cfmnist_spread): domain spread effect with N=3 fixed.

    Configurations:
        narrow  -> [0.1, 0.15, 0.2]   tight cluster
        medium  -> [0.1, 0.2,  0.3]   modest extension
        wide    -> [0.1, 0.25, 0.4]   wider coverage
    """
    spread_configs = {
        "narrow": "0.1,0.15,0.2",
        "medium": "0.1,0.2,0.3",
        "wide":   "0.1,0.25,0.4",
    }
    alpha_eqrm = -1000

    cmds = []
    for seed in seeds:
        for spread_name, envs_str in spread_configs.items():
            seed_base = (
                f"{base_call} --seed {seed} --exp_name cfmnist_spread "
                f"--train_envs {envs_str}"
            )

            cmds.append(
                f"{seed_base} --algorithm erm "
                f"--steps 400 --erm_pretrain_iters 0"
            )

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
        description="Generate commands for CFashionMNIST experiments."
    )
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_group",  type=str, default="all",
                        choices=["reproduce", "scaling", "spread", "all"])
    parser.add_argument("--n_seeds",    type=int, default=3)
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))

    base_call = (
        f"python train_cfmnist.py "
        f"--data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--lr 1e-4 "
        f"--batch_size 25000 "
        f"--dropout_p 0.2"
    )

    print(f"\nGenerating CFashionMNIST commands (seeds={seeds}) ...")
    total = 0

    if args.exp_group in ("reproduce", "all"):
        total += gen_cfmnist_reproduce(base_call, seeds,
                                       "job_scripts/cfmnist_reproduce.txt")

    if args.exp_group in ("scaling", "all"):
        total += gen_cfmnist_scaling(base_call, seeds,
                                     "job_scripts/cfmnist_scaling.txt")

    if args.exp_group in ("spread", "all"):
        total += gen_cfmnist_spread(base_call, seeds,
                                    "job_scripts/cfmnist_spread.txt")

    print(f"\nTotal CFashionMNIST commands: {total}")
    print(
        "\nDomain configurations for CFashionMNIST scaling experiment "
        "(evenly spaced in [0.1, 0.2]):"
    )
    for n in [2, 3, 4]:
        envs = evenly_spaced_envs(n)
        # Fashion-MNIST has 60k training images; first 50k used, then interleaved
        spe = 50000 // n
        print(f"  N={n}: {envs}  ({spe} images/env)")

    print("\nAnalogy with binary CMNIST:")
    print("  Binary CMNIST : p_e=[0.1,0.2], OOD p=0.9, oracle='gray' (p=0.5,0.5)")
    print("  CFashionMNIST : p_e=[0.1,0.2], OOD p=0.9, oracle='oracle'(p=0.5,0.5)")
    print("  (Same parameterization; harder base dataset, explicit Red/Green colors)")
