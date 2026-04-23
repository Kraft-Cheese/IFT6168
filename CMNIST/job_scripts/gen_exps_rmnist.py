#!/usr/bin/env python3
"""
Generate sweep commands for Rotated MNIST (RMNIST) experiments.

Experiment groups
-----------------
1. rmnist_reproduce  — Validate setup on the original 2-domain configuration.
                       Train: delta=(60, 30), OOD test: delta=-60.
                       Algorithms: ERM, Oracle, EQRM (5 alphas), IRM, VREx.
                       3 seeds.

2. rmnist_scaling    — Fixed delta range [30, 60], vary N in {2, 3, 4}.
                       Algorithms: ERM + EQRM (all 5 alphas).
                       3 seeds.

3. rmnist_spread     — Fix N=3, vary spread of training deltas:
                       narrow [40, 50, 60], medium [30, 45, 60],
                       wide   [20, 40, 60].
                       Algorithms: ERM + EQRM (alpha=-1000).
                       3 seeds.

Convention
----------
All settings are identical to binary CMNIST (gen_exps_ndomain.py) — the
only differences are the training script (train_rmnist.py), the exp_name
prefix (rmnist_*), and the environment parameterisation (delta in degrees
instead of flip probability p_e).

  Binary CMNIST : train_envs p_e ∈ [0.1, 0.2], OOD p_e = 0.9
  RMNIST        : train_envs delta ∈ [30°, 60°], OOD delta = -60°

Usage (from CMNIST/ directory):
    python -m job_scripts.gen_exps_rmnist \\
        --data_dir /content/data \\
        --output_dir /content/results
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evenly_spaced_deltas(n, lo=30.0, hi=60.0):
    """Return n evenly-spaced delta values in [lo, hi] degrees."""
    if n == 1:
        return [round(lo, 1)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 1) for i in range(n)]


def deltas_to_str(deltas):
    return ",".join(str(d) for d in deltas)


def write_cmds(cmds, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(cmds))
    print(f"  Wrote {len(cmds):4d} commands -> {filepath}")
    return len(cmds)


# ---------------------------------------------------------------------------
# Experiment generators
# ---------------------------------------------------------------------------

def gen_rmnist_reproduce(base_call, seeds, output_file):
    """
    Exp 1 (rmnist_reproduce): validate RMNIST on the 2-domain config.

    Training envs: delta = (60, 30) — strong and moderate rotation shortcuts.
    Oracle: delta = (0, 0)  — rotation uninformative, model must use digit shape.
    OOD test: delta = -60  — rotation anti-correlated with label.
    """
    alphas         = [-100, -500, -1000, -5000, -10000]
    irm_penalties  = [1000, 10000]
    vrex_penalties = [1000, 10000]

    cmds = []
    for seed in seeds:
        seed_base = f"{base_call} --seed {seed} --exp_name rmnist_reproduce"

        # ERM (train at delta=60,30 — rotation is a strong shortcut)
        cmds.append(
            f"{seed_base} --algorithm erm --train_envs default "
            f"--steps 400 --erm_pretrain_iters 0"
        )

        # Oracle (delta=0,0 — rotation uninformative -> must use shape)
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


def gen_rmnist_scaling(base_call, seeds, output_file):
    """
    Exp 2 (rmnist_scaling): N-domain scaling within delta ∈ [30°, 60°].

    Domain configurations (evenly spaced in [30, 60]):
        N=2 -> [30.0, 60.0]
        N=3 -> [30.0, 45.0, 60.0]
        N=4 -> [30.0, 40.0, 50.0, 60.0]  (approx)
    """
    n_values = [2, 3, 4]
    alphas   = [-100, -500, -1000, -5000, -10000]

    cmds = []
    for seed in seeds:
        for n in n_values:
            deltas_str = deltas_to_str(evenly_spaced_deltas(n))
            seed_base  = (
                f"{base_call} --seed {seed} --exp_name rmnist_scaling "
                f"--train_envs {deltas_str}"
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


def gen_rmnist_spread(base_call, seeds, output_file):
    """
    Exp 3 (rmnist_spread): domain spread effect with N=3 fixed.

    Configurations:
        narrow -> [40.0, 50.0, 60.0]  tight cluster (all strong shortcuts)
        medium -> [30.0, 45.0, 60.0]  moderate spread (same as N=3 scaling)
        wide   -> [20.0, 40.0, 60.0]  wide coverage (includes weaker shortcuts)
    """
    spread_configs = {
        "narrow": "40.0,50.0,60.0",
        "medium": "30.0,45.0,60.0",
        "wide":   "20.0,40.0,60.0",
    }
    alpha_eqrm = -1000

    cmds = []
    for seed in seeds:
        for spread_name, deltas_str in spread_configs.items():
            seed_base = (
                f"{base_call} --seed {seed} --exp_name rmnist_spread "
                f"--train_envs {deltas_str}"
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
        description="Generate commands for RMNIST experiments."
    )
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_group",  type=str, default="all",
                        choices=["reproduce", "scaling", "spread", "all"])
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

    print(f"\nGenerating RMNIST commands (seeds={seeds}) ...")
    total = 0

    if args.exp_group in ("reproduce", "all"):
        total += gen_rmnist_reproduce(base_call, seeds,
                                      "job_scripts/rmnist_reproduce.txt")

    if args.exp_group in ("scaling", "all"):
        total += gen_rmnist_scaling(base_call, seeds,
                                    "job_scripts/rmnist_scaling.txt")

    if args.exp_group in ("spread", "all"):
        total += gen_rmnist_spread(base_call, seeds,
                                   "job_scripts/rmnist_spread.txt")

    print(f"\nTotal RMNIST commands: {total}")
    print(
        "\nDomain configurations for RMNIST scaling experiment "
        "(evenly spaced in [30°, 60°]):"
    )
    for n in [2, 3, 4]:
        deltas = evenly_spaced_deltas(n)
        spe = 50000 // n
        mu_pairs = [(45.0 - d / 2, 45.0 + d / 2) for d in deltas]
        print(f"  N={n}: deltas={deltas}  "
              f"(mu_0,mu_1)={([(round(a,1),round(b,1)) for a,b in mu_pairs])}  "
              f"({spe} images/env)")

    print("\nOOD test environment: delta=-60 -> mu_0=75°, mu_1=15° (inverted)")
    print("Oracle:               delta=0   -> mu_0=45°, mu_1=45° (uninformative)")
    print("\nAnalogy with binary CMNIST:")
    print("  Binary CMNIST : p_e=[0.1,0.2], OOD p=0.9 (anti-correlated color)")
    print("  RMNIST        : delta=[60,30], OOD delta=-60 (inverted rotation)")
