#!/usr/bin/env python3
"""
Generate sweep commands for MC-CMNIST (multi-class ColoredMNIST) experiments.

Experiment groups
-----------------
1. mc_reproduce   — Validate the MC-CMNIST setup on the original 2-domain configuration
                    (train: p=0.1, 0.2; test OOD: p=0.9).
                    Algorithms: ERM, Oracle, EQRM (5 alphas), IRM, VREx.
                    3 seeds.

2. mc_scaling     — Fixed flip-prob range [0.1, 0.2], vary N ∈ {2, 3, 4}.
                    Algorithms: ERM + EQRM (all 5 alphas).
                    3 seeds.  Same exp_name so ERM checkpoints are shared.

3. mc_spread      — Fix N=3, vary spread of training flip probabilities:
                    narrow [0.1, 0.15, 0.2], medium [0.1, 0.2, 0.3],
                    wide   [0.1, 0.25, 0.4].
                    Algorithms: ERM + EQRM (alpha=-1000).
                    3 seeds.

Domain semantics (MC-CMNIST vs binary CMNIST)
---------------------------------------------
  binary CMNIST : p = *flip* probability (low p → color correlates; high p → anti-corr)
                  train at low p (0.1-0.2), test OOD at high p (0.9)
  MC-CMNIST     : p = *flip* probability (low p → color correlates; high p → uninformative)
                  train at low p (0.1-0.2), test OOD at high p (0.9)

Both tasks now share the same environment parameterization. For MC-CMNIST, the truly
uninformative boundary is p=0.9 (at which point all 10 colors are equally likely
for any label).

Usage (from CMNIST/ directory):
    python -m job_scripts.gen_exps_mc \\
        --data_dir /content/data \\
        --output_dir /content/results
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Helpers (mirrored from gen_exps_ndomain.py for the MC-CMNIST setting)
# ---------------------------------------------------------------------------

def evenly_spaced_envs_mc(n, lo=0.1, hi=0.2):
    """
    Return n evenly-spaced flip probabilities in [lo, hi].

    Low values (close to lo=0.1) → color is a very reliable shortcut.
    More environments are added as intermediate points, matching the binary setup.
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

def gen_mc_reproduce(base_call, seeds, output_file):
    """
    Exp 1 (mc_reproduce): validate MC-CMNIST on 2-domain config (p=0.1, 0.2).

    Mirrors the binary 'reproduce' experiment so results are directly comparable.
    Oracle trains both environments with p=0.9 (color is uniformly random
    → model must learn digit shape).
    """
    alphas        = [-100, -500, -1000, -5000, -10000]
    irm_penalties = [1000, 10000]
    vrex_penalties = [1000, 10000]

    cmds = []
    for seed in seeds:
        seed_base = f"{base_call} --seed {seed} --exp_name mc_reproduce"

        # ERM  (standard 2-domain, exploits color shortcut)
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs default "
            f"--steps 400 --erm_pretrain_iters 0"
        )

        # Oracle  (train on p=0.9,0.9  → color is uniformly random → must use shape)
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


def gen_mc_scaling(base_call, seeds, output_file):
    """
    Exp 2+3 (mc_scaling): N-domain scaling within [0.1, 0.2] × alpha sweep.

    Directly mirrors the binary ndomain_scaling experiment. Adding intermediate
    environments (N=3,4) tests whether EQRM's quantile estimates improve on the
    harder 10-class problem.

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
            envs_str  = envs_to_str(evenly_spaced_envs_mc(n))
            seed_base = (
                f"{base_call} --seed {seed} --exp_name mc_scaling "
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


def gen_mc_spread(base_call, seeds, output_file):
    """
    Exp 4 (mc_spread): domain spread effect with N=3 fixed.

    Tests whether the *range* of flip probabilities seen during training
    matters beyond the count of environments.

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
                f"{base_call} --seed {seed} --exp_name mc_spread "
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
        description="Generate commands for MC-CMNIST experiments."
    )
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_group",  type=str, default="all",
                        choices=["reproduce", "scaling", "spread", "all"])
    parser.add_argument("--n_seeds",    type=int, default=3)
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))

    # Base command uses train_mc.py — same hparams as binary CMNIST
    base_call = (
        f"python train_mc.py "
        f"--data_dir {args.data_dir} "
        f"--output_dir {args.output_dir} "
        f"--lr 1e-4 "
        f"--batch_size 25000 "
        f"--dropout_p 0.2"
    )

    print(f"\nGenerating MC-CMNIST commands (seeds={seeds}) ...")
    total = 0

    if args.exp_group in ("reproduce", "all"):
        total += gen_mc_reproduce(base_call, seeds,
                                  "job_scripts/mc_reproduce.txt")

    if args.exp_group in ("scaling", "all"):
        total += gen_mc_scaling(base_call, seeds,
                                "job_scripts/mc_scaling.txt")

    if args.exp_group in ("spread", "all"):
        total += gen_mc_spread(base_call, seeds,
                               "job_scripts/mc_spread.txt")

    print(f"\nTotal MC-CMNIST commands: {total}")
    print(
        "\nDomain configurations for MC-CMNIST scaling experiment "
        "(evenly spaced in [0.1, 0.2]):"
    )
    for n in [2, 3, 4]:
        envs = evenly_spaced_envs_mc(n)
        spe  = 50000 // n
        print(f"  N={n}: {envs}  ({spe} images/env)")

    print("\nAnalogy with binary CMNIST:")
    print("  Binary  train p_e=[0.1,0.2] (flip prob),  test OOD p_e=0.9")
    print("  MC      train p_e=[0.1,0.2] (flip prob),  test OOD p_e=0.9")
    print("  (Same parameterization, same train/test interpretation)")
