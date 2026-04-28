"""
Plot results from WILDS experiment logs using the standard WILDS/DomainBed metrics.

Reads val_eval.csv and test_eval.csv produced by run_expt.py and generates:
  1. results_table.csv     — avg / worst-group metrics at best epoch (WILDS table format)
  2. pareto.png            — (avg, worst-group) scatter per algorithm (DomainBed style)
  3. learning_curves.png   — val avg + worst-group vs epoch for each algorithm
  4. group_accuracy.png    — per-identity-group accuracy at best epoch (CivilComments only)
  5. per_country.png       — per-country r at best epoch (PovertyMap only)

Usage:
  python plot_results.py --dataset civilcomments
  python plot_results.py --dataset poverty
  python plot_results.py --dataset civilcomments --split test
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Config
HERE    = os.path.dirname(os.path.abspath(__file__))
LOG_BASE = os.path.join(HERE, "wilds_logs")

# Official WILDS primary/secondary metrics per dataset
DATASET_METRICS = {
    "civilcomments": {"primary": "acc_wg",  "secondary": "acc_avg",  "scale": 100, "unit": "%"},
    "poverty":       {"primary": "r_wg",    "secondary": "r_all",    "scale": 1,   "unit": "r"},
}

# Readable algorithm names
ALGO_LABELS = {
    "ERM":      "ERM",
    "groupDRO": "GroupDRO",
    "VREx":     "V-REx",
    "EQRM":     "EQRM",
    "IRM":      "IRM",
    "deepCORAL":"DeepCORAL",
}

ALGO_COLORS = {
    "ERM":      "#555555",
    "groupDRO": "#1f77b4",
    "VREx":     "#ff7f0e",
    "EQRM":     "#d62728",
}

ALGO_MARKERS = {
    "ERM": "o", "groupDRO": "s", "VREx": "^", "EQRM": "D",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_run_dirs(dataset, log_base, quick=False):
    """
    Return {algo: path} for completed runs of this dataset.
    Skips runs with empty val_eval.csv. When both a quick and full run exist
    for the same algo, prefers the full run unless --quick is passed.
    """
    pattern = os.path.join(log_base, dataset, f"{dataset}_*_seed*")
    candidates = {}  # algo -> list of (is_quick, path)

    for d in sorted(glob.glob(pattern)):
        name    = os.path.basename(d)
        is_quick = name.endswith("_quick")
        # Strip dataset prefix and trailing seed/quick tokens to get algo name
        parts      = name.split("_")
        strip_tail = 3 if is_quick else 2   # drop "quick" + "seed0" or just "seed0"
        algo_parts = parts[1:-strip_tail]
        algo       = "_".join(algo_parts) if algo_parts else parts[1]

        val_csv = os.path.join(d, "val_eval.csv")
        if not os.path.exists(val_csv):
            continue
        # Skip empty files (run failed before completing an epoch)
        if os.path.getsize(val_csv) == 0 or sum(1 for _ in open(val_csv)) < 2:
            continue

        candidates.setdefault(algo, []).append((is_quick, d))

    dirs = {}
    for algo, runs in candidates.items():
        # Sort: full runs first (is_quick=False sorts before True)
        runs.sort(key=lambda x: x[0])
        preferred = [r for r in runs if r[0] == quick]
        chosen    = preferred[0] if preferred else runs[0]
        dirs[algo] = chosen[1]

    return dirs


def load_val(run_dir, primary, secondary):
    """Load val_eval.csv and return DataFrame."""
    path = os.path.join(run_dir, "val_eval.csv")
    df = pd.read_csv(path)
    # Rename poverty columns if needed (r_wg may appear as Pearson_r_wg etc.)
    df.columns = [c.strip() for c in df.columns]
    return df


def best_epoch_row(df, primary, val_metric_decreasing=False):
    """Return the row at the best validation epoch."""
    if val_metric_decreasing:
        return df.loc[df[primary].idxmin()]
    return df.loc[df[primary].idxmax()]


def load_test(run_dir, primary, secondary):
    """Load test_eval.csv at the best epoch (saved as epoch:best_pred)."""
    path = os.path.join(run_dir, "test_eval.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# Results table
def make_results_table(run_dirs, cfg, split, out_dir):
    rows = []
    for algo, d in run_dirs.items():
        val_df = load_val(d, cfg["primary"], cfg["secondary"])
        best   = best_epoch_row(val_df, cfg["primary"])
        row    = {"Algorithm": ALGO_LABELS.get(algo, algo)}

        s = cfg["scale"]
        row[f"Val {cfg['primary']}"]    = round(best[cfg["primary"]]    * s, 2)
        row[f"Val {cfg['secondary']}"]  = round(best[cfg["secondary"]]  * s, 2)
        row["Best epoch"] = int(best["epoch"])

        if split == "test":
            test_df = load_test(d, cfg["primary"], cfg["secondary"])
            if test_df is not None and not test_df.empty:
                tb = test_df.iloc[-1]
                if cfg["primary"] in tb:
                    row[f"Test {cfg['primary']}"]   = round(tb[cfg["primary"]]   * s, 2)
                    row[f"Test {cfg['secondary']}"] = round(tb[cfg["secondary"]] * s, 2)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(f"Val {cfg['primary']}", ascending=False)
    out_path = os.path.join(out_dir, "results_table.csv")
    table.to_csv(out_path, index=False)
    print("\n=== Results Table ===")
    print(table.to_string(index=False))
    print(f"\nSaved {out_path}")
    return table


# Pareto scatter (avg vs worst-group)
def plot_pareto(run_dirs, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    s = cfg["scale"]

    for algo, d in run_dirs.items():
        val_df = load_val(d, cfg["primary"], cfg["secondary"])
        best   = best_epoch_row(val_df, cfg["primary"])
        x = best[cfg["secondary"]] * s   # avg
        y = best[cfg["primary"]]   * s   # worst-group

        label  = ALGO_LABELS.get(algo, algo)
        color  = ALGO_COLORS.get(algo, "gray")
        marker = ALGO_MARKERS.get(algo, "o")
        ax.scatter(x, y, color=color, marker=marker, s=120, zorder=3, label=label)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color=color)

    unit = cfg["unit"]
    ax.set_xlabel(f"Average accuracy ({unit})")
    ax.set_ylabel(f"Worst-group accuracy ({unit})")
    ax.set_title(f"{args.dataset}  Average vs Worst-group (val, best epoch)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "pareto.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

# Learning curves
def plot_learning_curves(run_dirs, cfg, out_dir):
    primary   = cfg["primary"]
    secondary = cfg["secondary"]
    s         = cfg["scale"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for algo, d in run_dirs.items():
        val_df = load_val(d, primary, secondary)
        label  = ALGO_LABELS.get(algo, algo)
        color  = ALGO_COLORS.get(algo, "gray")
        epochs = val_df["epoch"]

        if primary in val_df.columns:
            axes[0].plot(epochs, val_df[primary]   * s, label=label, color=color, lw=2)
        if secondary in val_df.columns:
            axes[1].plot(epochs, val_df[secondary] * s, label=label, color=color, lw=2)

    unit = cfg["unit"]
    axes[0].set_title(f"Worst-group ({primary}) val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(f"{primary} ({unit})")
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"Average ({secondary}) val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(f"{secondary} ({unit})")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{args.dataset} Validation learning curves", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "learning_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

# Per-identity-group accuracy bar chart

CIVIL_IDENTITY_GROUPS = [
    ("black",           "Black"),
    ("white",           "White"),
    ("muslim",          "Muslim"),
    ("christian",       "Christian"),
    ("LGBTQ",           "LGBTQ+"),
    ("female",          "Female"),
    ("male",            "Male"),
    ("other_religions", "Other religion"),
]

def plot_group_accuracy_civil(run_dirs, out_dir):
    """Bar chart: per-identity worst-group acc (toxic subgroup) for each algorithm."""
    records = {}
    for algo, d in run_dirs.items():
        val_df = load_val(d, "acc_wg", "acc_avg")
        best   = best_epoch_row(val_df, "acc_wg")
        records[algo] = {}
        for col, label in CIVIL_IDENTITY_GROUPS:
            # toxic subgroup column: acc_y:1_{col}:1
            key = f"acc_y:1_{col}:1"
            if key in best:
                records[algo][label] = best[key] * 100

    if not records:
        return

    identity_labels = [label for _, label in CIVIL_IDENTITY_GROUPS
                       if label in list(records.values())[0]]
    algos = list(records.keys())
    x     = np.arange(len(identity_labels))
    width = 0.8 / len(algos)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, algo in enumerate(algos):
        vals = [records[algo].get(lbl, np.nan) for lbl in identity_labels]
        offset = (i - len(algos) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=ALGO_LABELS.get(algo, algo),
               color=ALGO_COLORS.get(algo, "gray"),
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(identity_labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy on toxic subgroup (%)")
    ax.set_title("Per-identity toxic-subgroup accuracy (val, best epoch)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "group_accuracy.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# PovertyMap per-country r bar chart
def plot_per_country_poverty(run_dirs, out_dir):
    """Bar chart of per-country Pearson r at best epoch, one panel per algorithm."""
    country_cols = {}
    for algo, d in run_dirs.items():
        val_df = load_val(d, "r_wg", "r_all")
        best   = best_epoch_row(val_df, "r_wg")
        # Pearson r per country appears as r_country:<name> in some WILDS versions
        r_cols = {c: best[c] for c in best.index if c.startswith("r_country:") or c.startswith("Pearson")}
        if r_cols:
            country_cols[algo] = r_cols

    if not country_cols:
        print("No per-country columns found in val_eval.csv")
        return

    algos    = list(country_cols.keys())
    countries = sorted(list(country_cols[algos[0]].keys()))
    x        = np.arange(len(countries))
    width    = 0.8 / len(algos)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, algo in enumerate(algos):
        vals   = [country_cols[algo].get(c, np.nan) for c in countries]
        offset = (i - len(algos) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=ALGO_LABELS.get(algo, algo),
               color=ALGO_COLORS.get(algo, "gray"),
               alpha=0.85)

    labels = [c.replace("r_country:", "").replace("Pearson_r_", "") for c in countries]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("PovertyMap per-country Pearson r (val, best epoch)")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "per_country.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["civilcomments", "poverty"])
    parser.add_argument("--split",   default="val", choices=["val", "test"])
    parser.add_argument("--quick",   action="store_true", help="Plot quick sanity-check runs")
    args = parser.parse_args()

    cfg      = DATASET_METRICS[args.dataset]
    out_dir  = os.path.join(LOG_BASE, args.dataset, "plots")
    os.makedirs(out_dir, exist_ok=True)

    run_dirs = find_run_dirs(args.dataset, LOG_BASE, quick=args.quick)
    if not run_dirs:
        print(f"No completed runs found in {LOG_BASE}/{args.dataset}/")
        print("Run experiments first, then re-run this script.")
        exit(0)

    print(f"Found runs: {list(run_dirs.keys())}")

    make_results_table(run_dirs, cfg, args.split, out_dir)
    plot_pareto(run_dirs, cfg, out_dir)
    plot_learning_curves(run_dirs, cfg, out_dir)

    if args.dataset == "civilcomments":
        plot_group_accuracy_civil(run_dirs, out_dir)
    elif args.dataset == "poverty":
        plot_per_country_poverty(run_dirs, out_dir)

    print(f"\nAll plots saved to {out_dir}")
