#!/usr/bin/env python3
"""
Generate sweep commands for BIM (Binary Interleaved M2M) CMNIST experiments.

Experiment groups
-----------------
1. bim_reproduce  — Validate the BIM setup on the original 2-domain configuration
                    (train: p=0.1, 0.2; test OOD: p=0.9).
                    Algorithms: ERM, Oracle, EQRM (5 alphas), IRM, VREx.
                    3 seeds.

2. bim_scaling    — Fixed flip-prob range [0.1, 0.2], vary N in {2, 3, 4}.
                    Algorithms: ERM + EQRM (all 5 alphas).
                    3 seeds.  Same exp_name so ERM checkpoints are shared.

3. bim_spread     — Fix N=3, vary spread of training flip probabilities:
                    narrow [0.1, 0.15, 0.2], medium [0.1, 0.2, 0.3],
                    wide   [0.1, 0.25, 0.4].
                    Algorithms: ERM + EQRM (alpha=-1000).
                    3 seeds.

BIM vs binary CMNIST
--------------------
  binary CMNIST : 2-channel color (R vs G), linear decision boundary in color space
  BIM           : 10-color interleaved pools, non-linear (XOR) boundary in RGB space

Both share the same flip-probability convention (p_e = P(wrong pool/color)),
the same training env range [0.1, 0.2], and the same OOD test env p=0.9.
The oracle trains at p=0.5 (both pools equally likely -> color uninformative).

Usage (from CMNIST/ directory):
    python -m job_scripts.gen_exps_bim \\
        --data_dir /content/data \\
        --output_dir /content/results
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Helpers  (identical to gen_exps_mc.py)
# ---------------------------------------------------------------------------

def evenly_spaced_envs_bim(n, lo=0.1, hi=0.2):
    """
    Return n evenly-spaced flip probabilities in [lo, hi].

    Low values (close to lo=0.1) -> color pool is a very reliable shortcut.
    More environments add intermediate points, matching binary / MC setup.
    """
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

def gen_bim_reproduce(base_call, seeds, output_file):
    """
    Exp 1 (bim_reproduce): validate BIM on 2-domain config (p=0.1, 0.2).

    Mirrors binary 'reproduce' and mc_reproduce so results are directly
    comparable.  Oracle trains at p=0.5,0.5 where color pools are 50/50
    -> model must rely on digit shape.
    """
    alphas         = [-100, -500, -1000, -5000, -10000]
    irm_penalties  = [1000, 10000]
    vrex_penalties = [1000, 10000]

    cmds = []
    for seed in seeds:
        seed_base = f"{base_call} --seed {seed} --exp_name bim_reproduce"

        # ERM  (standard 2-domain, exploits interleaved-color shortcut)
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs default "
            f"--steps 400 --erm_pretrain_iters 0"
        )

        # Oracle  (train on p=0.5,0.5 -> color pools uninformative -> must use shape)
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


def gen_bim_scaling(base_call, seeds, output_file):
    """
    Exp 2 (bim_scaling): N-domain scaling within [0.1, 0.2] x alpha sweep.

    Tests whether more training environments help EQRM identify the invariant
    feature when the shortcut has a non-linear (interleaved) decision boundary.

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
            envs_str  = envs_to_str(evenly_spaced_envs_bim(n))
            seed_base = (
                f"{base_call} --seed {seed} --exp_name bim_scaling "
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


def gen_bim_spread(base_call, seeds, output_file):
    """
    Exp 3 (bim_spread): domain spread effect with N=3 fixed.

    Tests whether the *range* of flip probabilities seen during training
    matters for recovering the invariant feature on the non-linear shortcut.

    Configurations:
        narrow  -> [0.1, 0.15, 0.2]   same as N=3 scaling, tight cluster
        medium  -> [0.1, 0.2,  0.3]   extends to 0.3
        wide    -> [0.1, 0.25, 0.4]   extends to 0.4
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
                f"{base_call} --seed {seed} --exp_name bim_spread "
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
        description="Generate commands for BIM (Binary Interleaved M2M) CMNIST experiments."
    )
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_group",  type=str, default="all",
                        choices=["reproduce", "scaling", "spread", "all"])
    parser.add_argument("--n_seeds",    type=int, default=3)
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))

    base_call = (
        f"python train_bim.py "
        f"--data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--lr 1e-4 "
        f"--batch_size 25000 "
        f"--dropout_p 0.2"
    )

    print(f"\nGenerating BIM commands (seeds={seeds}) ...")
    total = 0

    if args.exp_group in ("reproduce", "all"):
        total += gen_bim_reproduce(base_call, seeds,
                                   "job_scripts/bim_reproduce.txt")

    if args.exp_group in ("scaling", "all"):
        total += gen_bim_scaling(base_call, seeds,
                                 "job_scripts/bim_scaling.txt")

    if args.exp_group in ("spread", "all"):
        total += gen_bim_spread(base_call, seeds,
                                "job_scripts/bim_spread.txt")

    print(f"\nTotal BIM commands: {total}")
    print(
        "\nDomain configurations for BIM scaling experiment "
        "(evenly spaced in [0.1, 0.2]):"
    )
    for n in [2, 3, 4]:
        envs = evenly_spaced_envs_bim(n)
        spe  = 50000 // n
        print(f"  N={n}: {envs}  ({spe} images/env)")

    print("\nAnalogy with binary / MC CMNIST:")
    print("  Binary  train p_e=[0.1,0.2], test OOD p_e=0.9  (linear color boundary)")
    print("  MC      train p_e=[0.1,0.2], test OOD p_e=0.9  (10-class, linear per-class)")
    print("  BIM     train p_e=[0.1,0.2], test OOD p_e=0.9  (binary, non-linear interleaved pools)")
