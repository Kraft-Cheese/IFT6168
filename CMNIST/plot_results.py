#!/usr/bin/env python3
"""
Publication-quality figures for the 6-experiment test plan.

Experiment → Figure mapping
---------------------------
fig1_reproduce          Binary CMNIST   — risk profiles, 2-domain (ERM/EQRM/IRM/VREx)
fig6_mc_reproduce       MC-CMNIST       — risk profiles, 2-domain (all algorithms)
fig9_bim_reproduce      BIM             — risk profiles, 2-domain (all algorithms)
fig13_cfmnist_reproduce CFashionMNIST   — risk profiles, 2-domain (all algorithms)
fig16_rmnist_reproduce  RMNIST          — risk profiles, δ sweep −90°→+90° (all algorithms)
fig19_rmnist_grid       RMNIST N×Spread — OOD acc vs N and spread advantage

Usage
-----
    python plot_results.py \\
        --output_dir /content/results \\
        --figures_dir /content/results/figures

Results are auto-discovered under:
    {output_dir}/results/reproduce/
    {output_dir}/results/mc_reproduce/
    {output_dir}/results/bim_reproduce/
    {output_dir}/results/cfmnist_reproduce/
    {output_dir}/results/rmnist_reproduce/
    {output_dir}/results/rmnist_grid/
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

STYLE = {
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "legend.fontsize":    9,
    "legend.framealpha":  0.8,
    "lines.linewidth":    2.0,
    "lines.markersize":   6,
    "errorbar.capsize":   3,
}
plt.rcParams.update(STYLE)

ALGO_COLORS = {
    "erm":      "#555555",
    "oracle":   "#2ca02c",
    "eqrm":     "#1f77b4",
    "irm":      "#d62728",
    "vrex":     "#ff7f0e",
    "groupdro": "#9467bd",
}

ALL_PS     = [round(i / 10, 1) for i in range(11)]   # 0.0 … 1.0
ALL_DELTAS = [float(i * 15)    for i in range(-6, 7)] # −90 … +90
RMNIST_OOD_DELTA = -60.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl_dir(dirpath):
    records = []
    for fname in glob.glob(os.path.join(dirpath, "*.jsonl")):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


BURNIN_VALUES  = [0, 150, 300, 450, 600]
BURNIN_TOTAL   = 600


# Spurious ablation — env pairs ordered from strongest to weakest shortcut
SPURIOUS_ENV_PAIRS = [
    (0.90, (0.05, 0.15)),
    (0.85, (0.10, 0.20)),
    (0.80, (0.15, 0.25)),
    (0.75, (0.20, 0.30)),
    (0.70, (0.25, 0.35)),
    (0.60, (0.35, 0.45)),
    (0.50, (0.45, 0.55)),
]
CAUSAL_CEILING = 0.75   # 1 - label_noise (25% flip)


def load_results(output_dir):
    data = {}
    for exp_name in ["reproduce", "mc_reproduce", "bim_reproduce",
                     "cfmnist_reproduce", "rmnist_reproduce", "rmnist_grid",
                     "burnin", "spurious"]:
        dirpath = os.path.join(output_dir, "results", exp_name)
        if os.path.isdir(dirpath):
            recs = load_jsonl_dir(dirpath)
            data[exp_name] = recs
            print(f"  Loaded {len(recs):4d} records from '{exp_name}'")
        else:
            print(f"  [skip] '{exp_name}' not found at {dirpath}")
            data[exp_name] = []
    return data


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def get_acc(rec, p, ms="best"):
    return rec.get(f"{p}_acc_{ms}", None)


def group_mean_se(records, p_vals, ms="best"):
    accs = {p: [] for p in p_vals}
    for rec in records:
        for p in p_vals:
            v = get_acc(rec, p, ms)
            if v is not None:
                accs[p].append(v)
    means, ses, n = [], [], 0
    for p in p_vals:
        vals = accs[p]
        if vals:
            n = max(n, len(vals))
            means.append(np.mean(vals))
            ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.)
        else:
            means.append(np.nan)
            ses.append(np.nan)
    return np.array(means), np.array(ses), n


def best_alpha_records(records, alg="eqrm", ood_p=0.9, ms="best"):
    alpha_groups = {}
    for rec in records:
        if alg is not None and rec.get("algorithm") != alg:
            continue
        alpha = rec.get("args", {}).get("alpha", None)
        if alpha is None:
            continue
        alpha_groups.setdefault(alpha, []).append(rec)
    if not alpha_groups:
        return [], None
    best_alpha = max(alpha_groups, key=lambda a: np.mean(
        [get_acc(r, ood_p, ms) for r in alpha_groups[a]
         if get_acc(r, ood_p, ms) is not None] or [0]
    ))
    return alpha_groups[best_alpha], best_alpha


# ---------------------------------------------------------------------------
# Figure 1: Binary CMNIST Reproduce
# ---------------------------------------------------------------------------

def plot_figure1_reproduce(records, figures_dir, ms="best"):
    if not records:
        print("  [fig1] No reproduce records found, skipping.")
        return
    print("  Plotting Figure 1: Binary CMNIST Reproduce ...")
    fig, ax = plt.subplots(figsize=(6, 4))
    p_vals = ALL_PS

    def _plot(recs, label, color, ls="-", marker="o"):
        if not recs:
            return
        means, ses, n = group_mean_se(recs, p_vals, ms)
        ax.plot(p_vals, means, ls, color=color,
                label=f"{label} (n={n})", marker=marker, markevery=2)
        ax.fill_between(p_vals, means - ses, means + ses, alpha=0.15, color=color)

    erm_recs = [r for r in records if r.get("algorithm") == "erm"
                and r.get("args", {}).get("train_envs", "") == "default"]
    _plot(erm_recs, "ERM", ALGO_COLORS["erm"], marker="s")

    oracle_recs = [r for r in records if r.get("algorithm") == "oracle"]
    _plot(oracle_recs, "Oracle (gray)", ALGO_COLORS["oracle"], ls="--", marker="^")

    eqrm_recs = [r for r in records if r.get("algorithm") == "eqrm"]
    if eqrm_recs:
        best_recs, best_a = best_alpha_records(eqrm_recs, ood_p=0.9, ms=ms)
        _plot(best_recs, f"EQRM (α={best_a})", ALGO_COLORS["eqrm"])

    irm_recs = [r for r in records if r.get("algorithm") == "irm"]
    if irm_recs:
        best_recs_irm, _ = best_alpha_records(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in irm_recs], alg="irm", ood_p=0.9, ms=ms)
        if best_recs_irm:
            pen = best_recs_irm[0]["args"]["alpha"]
            _plot([r for r in irm_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"IRM (λ={pen})", ALGO_COLORS["irm"], ls="-.", marker="D")

    vrex_recs = [r for r in records if r.get("algorithm") == "vrex"]
    if vrex_recs:
        best_recs_vrex, _ = best_alpha_records(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in vrex_recs], alg="vrex", ood_p=0.9, ms=ms)
        if best_recs_vrex:
            pen = best_recs_vrex[0]["args"]["alpha"]
            _plot([r for r in vrex_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"VREx (λ={pen})", ALGO_COLORS["vrex"], ls=":", marker="v")

    for p in [0.1, 0.2]:
        ax.axvline(p, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(0.15, 0.02, "train\nenvs", transform=ax.get_xaxis_transform(),
            fontsize=8, color="gray", ha="center", va="bottom")
    ax.set_xlabel("Test flip probability  p")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig. 1 — Binary CMNIST Reproduce (train: p=0.1, 0.2)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.legend(loc="upper right")
    _save(fig, figures_dir, "fig1_reproduce")


# ---------------------------------------------------------------------------
# Figure 6: MC-CMNIST Reproduce
# ---------------------------------------------------------------------------

def plot_figure6_mc_reproduce(records, figures_dir, ms="best"):
    if not records:
        print("  [fig6] No mc_reproduce records found, skipping.")
        return
    print("  Plotting Figure 6: MC-CMNIST Reproduce ...")
    fig, ax = plt.subplots(figsize=(6, 4))
    p_vals = ALL_PS

    def _plot(recs, label, color, ls="-", marker="o"):
        if not recs:
            return
        means, ses, n = group_mean_se(recs, p_vals, ms)
        ax.plot(p_vals, means, ls, color=color,
                label=f"{label} (n={n})", marker=marker, markevery=2)
        ax.fill_between(p_vals, means - ses, means + ses, alpha=0.12, color=color)

    erm_recs = [r for r in records if r.get("algorithm") == "erm"
                and r.get("args", {}).get("train_envs", "") == "default"]
    _plot(erm_recs, "ERM", ALGO_COLORS["erm"], marker="s")

    oracle_recs = [r for r in records if r.get("algorithm") == "oracle"]
    _plot(oracle_recs, "Oracle (p=0.9,0.9)", ALGO_COLORS["oracle"], ls="--", marker="^")

    eqrm_recs = [r for r in records if r.get("algorithm") == "eqrm"]
    if eqrm_recs:
        best_recs, best_a = best_alpha_records(eqrm_recs, ood_p=0.9, ms=ms)
        _plot(best_recs, f"EQRM (α={best_a})", ALGO_COLORS["eqrm"])

    irm_recs = [r for r in records if r.get("algorithm") == "irm"]
    if irm_recs:
        best_recs_irm, _ = best_alpha_records(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in irm_recs], alg="irm", ood_p=0.9, ms=ms)
        if best_recs_irm:
            pen = best_recs_irm[0]["args"]["alpha"]
            _plot([r for r in irm_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"IRM (λ={pen})", ALGO_COLORS["irm"], ls="-.", marker="D")

    vrex_recs = [r for r in records if r.get("algorithm") == "vrex"]
    if vrex_recs:
        best_recs_vrex, _ = best_alpha_records(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in vrex_recs], alg="vrex", ood_p=0.9, ms=ms)
        if best_recs_vrex:
            pen = best_recs_vrex[0]["args"]["alpha"]
            _plot([r for r in vrex_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"VREx (λ={pen})", ALGO_COLORS["vrex"], ls=":", marker="v")

    for p, lbl in [(0.1, "train"), (0.2, "train"), (0.9, "test OOD")]:
        ax.axvline(p, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(p, 0.02, lbl, transform=ax.get_xaxis_transform(),
                fontsize=7, color="gray", ha="center", va="bottom", rotation=90)

    ax.plot(p_vals, [1 - p for p in p_vals], color="black", linestyle=":",
            linewidth=1, alpha=0.4, label="Color-only (1−p)")
    ax.axhline(0.1, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Test flip probability  p")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig. 6 — MC-CMNIST Reproduce (train p=0.1, 0.2; OOD p=0.9)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, figures_dir, "fig6_mc_reproduce")


# ---------------------------------------------------------------------------
# Figure 9: BIM Reproduce
# ---------------------------------------------------------------------------

def plot_figure9_bim_reproduce(records, figures_dir, ms="best"):
    if not records:
        print("  [fig9] No bim_reproduce records found, skipping.")
        return
    print("  Plotting Figure 9: BIM Reproduce ...")
    fig, ax = plt.subplots(figsize=(6, 4))
    p_vals = ALL_PS

    def _plot(recs, label, color, ls="-", marker="o"):
        if not recs:
            return
        means, ses, n = group_mean_se(recs, p_vals, ms)
        ax.plot(p_vals, means, ls, color=color,
                label=f"{label} (n={n})", marker=marker, markevery=2)
        ax.fill_between(p_vals, means - ses, means + ses, alpha=0.12, color=color)

    erm_recs = [r for r in records if r.get("algorithm") == "erm"
                and r.get("args", {}).get("train_envs", "") == "default"]
    _plot(erm_recs, "ERM", ALGO_COLORS["erm"], marker="s")

    oracle_recs = [r for r in records if r.get("algorithm") == "oracle"]
    _plot(oracle_recs, "Oracle (p=0.5,0.5)", ALGO_COLORS["oracle"], ls="--", marker="^")

    eqrm_recs = [r for r in records if r.get("algorithm") == "eqrm"]
    if eqrm_recs:
        best_recs, best_a = best_alpha_records(eqrm_recs, ood_p=0.9, ms=ms)
        _plot(best_recs, f"EQRM (α={best_a})", ALGO_COLORS["eqrm"])

    irm_recs = [r for r in records if r.get("algorithm") == "irm"]
    if irm_recs:
        best_recs_irm, _ = best_alpha_records(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in irm_recs], alg="irm", ood_p=0.9, ms=ms)
        if best_recs_irm:
            pen = best_recs_irm[0]["args"]["alpha"]
            _plot([r for r in irm_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"IRM (λ={pen})", ALGO_COLORS["irm"], ls="-.", marker="D")

    vrex_recs = [r for r in records if r.get("algorithm") == "vrex"]
    if vrex_recs:
        best_recs_vrex, _ = best_alpha_records(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in vrex_recs], alg="vrex", ood_p=0.9, ms=ms)
        if best_recs_vrex:
            pen = best_recs_vrex[0]["args"]["alpha"]
            _plot([r for r in vrex_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"VREx (λ={pen})", ALGO_COLORS["vrex"], ls=":", marker="v")

    ax.plot(p_vals, [1 - p for p in p_vals], color="black", linestyle=":",
            linewidth=1, alpha=0.4, label="Color-only (1−p)")
    for p, lbl in [(0.1, "train"), (0.2, "train"), (0.5, "oracle"), (0.9, "OOD")]:
        ax.axvline(p, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(p, 0.02, lbl, transform=ax.get_xaxis_transform(),
                fontsize=7, color="gray", ha="center", va="bottom", rotation=90)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Test flip probability  p")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig. 9 — BIM Reproduce (train p=0.1, 0.2; oracle p=0.5; OOD p=0.9)\n"
                 "Non-linear interleaved 10-color pools")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, figures_dir, "fig9_bim_reproduce")


# ---------------------------------------------------------------------------
# Figure 13: CFashionMNIST Reproduce
# ---------------------------------------------------------------------------

def plot_figure13_cfmnist_reproduce(records, figures_dir, ms="best"):
    if not records:
        print("  [fig13] No cfmnist_reproduce records found, skipping.")
        return
    print("  Plotting Figure 13: CFashionMNIST Reproduce ...")
    fig, ax = plt.subplots(figsize=(6, 4))
    p_vals = ALL_PS

    def _plot(recs, label, color, ls="-", marker="o"):
        if not recs:
            return
        means, ses, n = group_mean_se(recs, p_vals, ms)
        ax.plot(p_vals, means, ls, color=color,
                label=f"{label} (n={n})", marker=marker, markevery=2)
        ax.fill_between(p_vals, means - ses, means + ses, alpha=0.12, color=color)

    erm_recs = [r for r in records if r.get("algorithm") == "erm"
                and r.get("args", {}).get("train_envs", "") == "default"]
    _plot(erm_recs, "ERM", ALGO_COLORS["erm"], marker="s")

    oracle_recs = [r for r in records if r.get("algorithm") == "oracle"]
    _plot(oracle_recs, "Oracle (p=0.5,0.5)", ALGO_COLORS["oracle"], ls="--", marker="^")

    eqrm_recs = [r for r in records if r.get("algorithm") == "eqrm"]
    if eqrm_recs:
        best_recs, best_a = best_alpha_records(eqrm_recs, ood_p=0.9, ms=ms)
        _plot(best_recs, f"EQRM (α={best_a})", ALGO_COLORS["eqrm"])

    irm_recs = [r for r in records if r.get("algorithm") == "irm"]
    if irm_recs:
        best_recs_irm, _ = best_alpha_records(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in irm_recs], alg="irm", ood_p=0.9, ms=ms)
        if best_recs_irm:
            pen = best_recs_irm[0]["args"]["alpha"]
            _plot([r for r in irm_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"IRM (λ={pen})", ALGO_COLORS["irm"], ls="-.", marker="D")

    vrex_recs = [r for r in records if r.get("algorithm") == "vrex"]
    if vrex_recs:
        best_recs_vrex, _ = best_alpha_records(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in vrex_recs], alg="vrex", ood_p=0.9, ms=ms)
        if best_recs_vrex:
            pen = best_recs_vrex[0]["args"]["alpha"]
            _plot([r for r in vrex_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"VREx (λ={pen})", ALGO_COLORS["vrex"], ls=":", marker="v")

    ax.plot(p_vals, [1 - p for p in p_vals], color="black", linestyle=":",
            linewidth=1, alpha=0.4, label="Color-only (1−p)")
    for p, lbl in [(0.1, "train"), (0.2, "train"), (0.5, "oracle"), (0.9, "OOD")]:
        ax.axvline(p, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(p, 0.02, lbl, transform=ax.get_xaxis_transform(),
                fontsize=7, color="gray", ha="center", va="bottom", rotation=90)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Test flip probability  p")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig. 13 — CFashionMNIST Reproduce (train p=0.1, 0.2; OOD p=0.9)\n"
                 "Red (Apparel) vs Green (Accessories/Footwear)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, figures_dir, "fig13_cfmnist_reproduce")


# ---------------------------------------------------------------------------
# RMNIST helpers
# ---------------------------------------------------------------------------

def get_acc_delta(rec, delta, ms="best"):
    return rec.get(f"{delta}_acc_{ms}", None)


def group_mean_se_delta(records, deltas, ms="best"):
    accs = {d: [] for d in deltas}
    for rec in records:
        for d in deltas:
            v = get_acc_delta(rec, d, ms)
            if v is not None:
                accs[d].append(v)
    means, ses, n = [], [], 0
    for d in deltas:
        vals = accs[d]
        if vals:
            n = max(n, len(vals))
            means.append(np.mean(vals))
            ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.)
        else:
            means.append(np.nan)
            ses.append(np.nan)
    return np.array(means), np.array(ses), n


def get_env_deltas(rec):
    deltas = rec.get("args", {}).get("train_env_deltas", None)
    if deltas is not None:
        return tuple(sorted(deltas))
    return None


def best_alpha_records_delta(records, alg="eqrm", ood_delta=RMNIST_OOD_DELTA, ms="best"):
    alpha_groups = {}
    for rec in records:
        if alg is not None and rec.get("algorithm") != alg:
            continue
        alpha = rec.get("args", {}).get("alpha", None)
        if alpha is None:
            continue
        alpha_groups.setdefault(alpha, []).append(rec)
    if not alpha_groups:
        return [], None
    best_alpha = max(alpha_groups, key=lambda a: np.mean(
        [get_acc_delta(r, ood_delta, ms) for r in alpha_groups[a]
         if get_acc_delta(r, ood_delta, ms) is not None] or [0]
    ))
    return alpha_groups[best_alpha], best_alpha


# ---------------------------------------------------------------------------
# Figure 16: RMNIST Reproduce
# ---------------------------------------------------------------------------

def plot_figure16_rmnist_reproduce(records, figures_dir, ms="best"):
    if not records:
        print("  [fig16] No rmnist_reproduce records found, skipping.")
        return
    print("  Plotting Figure 16: RMNIST Reproduce ...")
    fig, ax = plt.subplots(figsize=(7, 4))
    deltas = ALL_DELTAS

    def _plot(recs, label, color, ls="-", marker="o"):
        if not recs:
            return
        means, ses, n = group_mean_se_delta(recs, deltas, ms)
        ax.plot(deltas, means, ls, color=color,
                label=f"{label} (n={n})", marker=marker, markevery=2)
        ax.fill_between(deltas, means - ses, means + ses, alpha=0.12, color=color)

    erm_recs = [r for r in records if r.get("algorithm") == "erm"
                and r.get("args", {}).get("train_envs", "") == "default"]
    _plot(erm_recs, "ERM", ALGO_COLORS["erm"], marker="s")

    oracle_recs = [r for r in records if r.get("algorithm") == "oracle"]
    _plot(oracle_recs, "Oracle (δ=0,0)", ALGO_COLORS["oracle"], ls="--", marker="^")

    eqrm_recs = [r for r in records if r.get("algorithm") == "eqrm"]
    if eqrm_recs:
        best_recs, best_a = best_alpha_records_delta(eqrm_recs, ms=ms)
        _plot(best_recs, f"EQRM (α={best_a})", ALGO_COLORS["eqrm"])

    irm_recs = [r for r in records if r.get("algorithm") == "irm"]
    if irm_recs:
        best_recs_irm, _ = best_alpha_records_delta(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in irm_recs], alg="irm", ms=ms)
        if best_recs_irm:
            pen = best_recs_irm[0]["args"]["alpha"]
            _plot([r for r in irm_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"IRM (λ={pen})", ALGO_COLORS["irm"], ls="-.", marker="D")

    vrex_recs = [r for r in records if r.get("algorithm") == "vrex"]
    if vrex_recs:
        best_recs_vrex, _ = best_alpha_records_delta(
            [{**r, "args": {**r.get("args", {}),
                            "alpha": r.get("args", {}).get("penalty_weight")}}
             for r in vrex_recs], alg="vrex", ms=ms)
        if best_recs_vrex:
            pen = best_recs_vrex[0]["args"]["alpha"]
            _plot([r for r in vrex_recs if r.get("args", {}).get("penalty_weight") == pen],
                  f"VREx (λ={pen})", ALGO_COLORS["vrex"], ls=":", marker="v")

    for delta, lbl in [(60, "train"), (30, "train"), (0, "oracle"), (-60, "OOD")]:
        ax.axvline(delta, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(delta, 0.02, lbl, transform=ax.get_xaxis_transform(),
                fontsize=7, color="gray", ha="center", va="bottom", rotation=90)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("Rotation delta  δ = μ₁ − μ₀  (degrees)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig. 16 — RMNIST Reproduce (train δ=60°,30°; OOD δ=−60°)\n"
                 "Spurious shortcut: rotation angle θ ~ N(μ_class, 5°²)")
    ax.set_xlim(-95, 95)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, figures_dir, "fig16_rmnist_reproduce")


# ---------------------------------------------------------------------------
# Figure 19: RMNIST N × Spread grid
# ---------------------------------------------------------------------------

_GRID_CONFIGS = {
    "N2_UltraNarrow": {"n": 2, "spread": "UltraNarrow", "deltas": (40.0, 50.0)},
    "N2_Narrow":      {"n": 2, "spread": "Narrow",      "deltas": (30.0, 60.0)},
    "N2_Wide":        {"n": 2, "spread": "Wide",        "deltas": (15.0, 75.0)},
    "N4_UltraNarrow": {"n": 4, "spread": "UltraNarrow", "deltas": (40.0, 43.3, 46.7, 50.0)},
    "N4_Narrow":      {"n": 4, "spread": "Narrow",      "deltas": (30.0, 40.0, 50.0, 60.0)},
    "N4_Wide":        {"n": 4, "spread": "Wide",        "deltas": (15.0, 35.0, 55.0, 75.0)},
}
_N_VALUES = [2, 4]
_SPREAD_STYLES  = {"UltraNarrow": {"linestyle": ":",  "marker": "^"},
                   "Narrow":      {"linestyle": "--", "marker": "o"},
                   "Wide":        {"linestyle": "-",  "marker": "s"}}
_SPREAD_COLORS  = {"erm":  {"UltraNarrow": "#cccccc", "Narrow": "#aaaaaa", "Wide": "#555555"},
                   "eqrm": {"UltraNarrow": "#c6dbef", "Narrow": "#9ecae1", "Wide": "#1f77b4"}}


def _grid_records(records, config_name, alg, alpha=None):
    target = _GRID_CONFIGS[config_name]["deltas"]
    out = []
    for r in records:
        if r.get("algorithm") != alg:
            continue
        if get_env_deltas(r) != target:
            continue
        if alpha is not None and r.get("args", {}).get("alpha") != alpha:
            continue
        out.append(r)
    return out


def plot_figure19_rmnist_grid(records, figures_dir, ms="best"):
    """
    Left  — OOD accuracy (δ=−60) vs N for ERM-Narrow, ERM-Wide,
             EQRM-Narrow, EQRM-Wide, and Oracle horizontal baseline.
    Right — Spread advantage (Wide − Narrow) vs N for ERM and EQRM.
    """
    if not records:
        print("  [fig19] No rmnist_grid records found, skipping.")
        return
    print("  Plotting Figure 19: RMNIST N × Spread grid ...")
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))
    ood_delta = RMNIST_OOD_DELTA

    # Oracle baseline
    oracle_recs = [r for r in records if r.get("algorithm") == "oracle"]
    if oracle_recs:
        m, s, n = group_mean_se_delta(oracle_recs, [ood_delta], ms)
        ax_l.axhline(m[0], color=ALGO_COLORS["oracle"], linestyle=":",
                     linewidth=1.8, label=f"Oracle (n={n})")
        ax_l.fill_between(_N_VALUES, [m[0]-s[0]]*len(_N_VALUES), [m[0]+s[0]]*len(_N_VALUES),
                           alpha=0.08, color=ALGO_COLORS["oracle"])

    spread_acc = {"erm": {}, "eqrm": {}}

    for spread in ("UltraNarrow", "Narrow", "Wide"):
        erm_m, erm_s   = [], []
        eqrm_m, eqrm_s = [], []

        for n in _N_VALUES:
            cfg = f"N{n}_{spread}"

            recs_erm = _grid_records(records, cfg, "erm")
            if recs_erm:
                m, s, _ = group_mean_se_delta(recs_erm, [ood_delta], ms)
                erm_m.append(m[0]); erm_s.append(s[0])
            else:
                erm_m.append(np.nan); erm_s.append(0.)

            recs_eq = _grid_records(records, cfg, "eqrm")
            if recs_eq:
                best_recs, _ = best_alpha_records_delta(recs_eq, ms=ms)
                m, s, _ = group_mean_se_delta(best_recs, [ood_delta], ms)
                eqrm_m.append(m[0]); eqrm_s.append(s[0])
            else:
                eqrm_m.append(np.nan); eqrm_s.append(0.)

        spread_acc["erm"][spread]  = (erm_m,  erm_s)
        spread_acc["eqrm"][spread] = (eqrm_m, eqrm_s)

        ls = _SPREAD_STYLES[spread]["linestyle"]
        mk = _SPREAD_STYLES[spread]["marker"]
        ax_l.errorbar(_N_VALUES, erm_m,  yerr=erm_s,
                      color=_SPREAD_COLORS["erm"][spread],
                      linestyle=ls, marker=mk, label=f"ERM {spread}")
        ax_l.errorbar(_N_VALUES, eqrm_m, yerr=eqrm_s,
                      color=_SPREAD_COLORS["eqrm"][spread],
                      linestyle=ls, marker=mk, label=f"EQRM {spread} (best α)")

    ax_l.set_xlabel("Number of training domains  N")
    ax_l.set_ylabel(f"Accuracy at δ={ood_delta}°  (OOD test)")
    ax_l.set_title("OOD Accuracy: N × Spread\n(solid=Wide, dashed=Narrow)")
    ax_l.set_xticks(_N_VALUES)
    ax_l.legend(fontsize=8)

    # Spread advantage — three pairwise comparisons per algorithm
    pair_styles = {
        ("Wide", "UltraNarrow"): {"linestyle": "-",  "marker": "o"},
        ("Wide", "Narrow"):      {"linestyle": "--", "marker": "s"},
        ("Narrow", "UltraNarrow"): {"linestyle": ":", "marker": "^"},
    }
    for alg, base_color, alg_lbl in [("erm",  ALGO_COLORS["erm"],  "ERM"),
                                       ("eqrm", ALGO_COLORS["eqrm"], "EQRM")]:
        for (hi, lo), style in pair_styles.items():
            hi_m, hi_s = spread_acc[alg][hi]
            lo_m, lo_s = spread_acc[alg][lo]
            adv    = [h - l if not (np.isnan(h) or np.isnan(l)) else np.nan
                      for h, l in zip(hi_m, lo_m)]
            adv_se = [np.sqrt(hs**2 + ls**2) for hs, ls in zip(hi_s, lo_s)]
            ax_r.errorbar(_N_VALUES, adv, yerr=adv_se,
                          color=base_color, label=f"{alg_lbl}: {hi}−{lo}",
                          **style)

    ax_r.axhline(0, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
    ax_r.set_xlabel("Number of training domains  N")
    ax_r.set_ylabel("Δ OOD accuracy  (higher spread − lower spread)")
    ax_r.set_title("Pairwise Spread Advantage vs N\n(positive = wider spread helps)")
    ax_r.set_xticks(_N_VALUES)
    ax_r.legend(fontsize=8)

    fig.suptitle(
        "Fig. 19 — RMNIST N × Spread Grid  "
        "(OOD δ=−60°,  UltraNarrow=[40°,50°],  Narrow=[30°,60°],  Wide=[15°,75°])",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig19_rmnist_grid")


# ---------------------------------------------------------------------------
# Figure: EQRM Burn-in Ablation
# ---------------------------------------------------------------------------

def plot_figure_burnin(records, figures_dir, ms="best"):
    """
    Two-panel figure for the EQRM burn-in ablation.

    Left  — OOD accuracy (p=0.9) vs erm_pretrain_iters with error bars.
    Right — Full risk profiles (p=0→1) for each burn-in level.
    """
    if not records:
        print("  [fig_burnin] No burnin records found, skipping.")
        return
    print("  Plotting Figure: EQRM Burn-in Ablation ...")

    # Colormap: blue (pure EQRM) → red (pure ERM)
    cmap   = plt.cm.coolwarm
    colors = {p: cmap(i / (len(BURNIN_VALUES) - 1))
              for i, p in enumerate(BURNIN_VALUES)}
    labels = {
        0:   "0  (0% — Pure EQRM)",
        150: "150 (25%)",
        300: "300 (50%)",
        450: "450 (75%)",
        600: f"600 (100% — Pure ERM)",
    }

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

    ood_means, ood_ses = [], []

    for pretrain in BURNIN_VALUES:
        recs = [r for r in records
                if r.get("algorithm") == "eqrm"
                and r.get("args", {}).get("erm_pretrain_iters") == pretrain]

        color = colors[pretrain]
        label = labels[pretrain]

        # --- Left panel: OOD accuracy point ---
        vals = [get_acc(r, 0.9, ms) for r in recs if get_acc(r, 0.9, ms) is not None]
        if vals:
            m  = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.
        else:
            m, se = np.nan, 0.
        ood_means.append(m)
        ood_ses.append(se)

        # --- Right panel: full risk profile ---
        means, ses, n = group_mean_se(recs, ALL_PS, ms)
        ax_r.plot(ALL_PS, means, color=color, linewidth=2,
                  label=f"pretrain={label} (n={n})")
        ax_r.fill_between(ALL_PS, means - ses, means + ses,
                          alpha=0.10, color=color)

    # Left panel: OOD accuracy vs burn-in
    fracs = [p / BURNIN_TOTAL for p in BURNIN_VALUES]
    ax_l.errorbar(fracs, ood_means, yerr=ood_ses,
                  color="black", marker="o", linewidth=2, capsize=4,
                  zorder=3)
    for frac, m, color in zip(fracs, ood_means, colors.values()):
        ax_l.scatter([frac], [m], color=color, s=80, zorder=4)
    ax_l.set_xlabel("ERM burn-in fraction  (erm_pretrain_iters / steps)")
    ax_l.set_ylabel("OOD Accuracy  (p = 0.9)")
    ax_l.set_title("OOD Accuracy vs Burn-in Fraction")
    ax_l.set_xlim(-0.05, 1.05)
    ax_l.set_ylim(0.0, 1.05)
    ax_l.set_xticks(fracs)
    ax_l.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_l.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.3)

    # Right panel: risk profiles
    ax_r.axvline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_r.text(0.9, 0.02, "OOD", transform=ax_r.get_xaxis_transform(),
              fontsize=7, color="gray", ha="center", va="bottom", rotation=90)
    for p in [0.1, 0.2]:
        ax_r.axvline(p, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_r.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    ax_r.set_xlabel("Test flip probability  p")
    ax_r.set_ylabel("Accuracy")
    ax_r.set_title("Risk Profiles by Burn-in Level")
    ax_r.set_xlim(-0.02, 1.02)
    ax_r.set_ylim(0.0, 1.05)
    ax_r.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "EQRM Burn-in Ablation — Binary CMNIST  "
        "(train p=0.1, 0.2 ;  OOD p=0.9 ;  α=−10000 ;  600 total steps)",
        fontsize=11
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig_burnin_ablation")


# ---------------------------------------------------------------------------
# Figure: Spurious Strength Ablation
# ---------------------------------------------------------------------------

def plot_figure_spurious(records, figures_dir, ms="best"):
    """
    Two-panel figure for the spurious strength ablation.

    Left  — OOD accuracy (p=0.9) vs mean spurious correlation for ERM,
            EQRM, and Oracle.  Phase regions shaded.
    Right — Full risk profiles at three key configurations:
            strong (mean=0.90), transition (mean=0.75), weak (mean=0.50).
    """
    if not records:
        print("  [fig_spurious] No spurious records found, skipping.")
        return
    print("  Plotting Figure: Spurious Strength Ablation ...")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    ood_p = 0.9

    # ---- Collect OOD accuracy per (alg, mean_e) ----
    oracle_recs = [r for r in records if r.get("algorithm") == "oracle"]
    if oracle_recs:
        oracle_vals = [get_acc(r, ood_p, ms) for r in oracle_recs
                       if get_acc(r, ood_p, ms) is not None]
        oracle_m  = np.mean(oracle_vals) if oracle_vals else np.nan
        oracle_se = (np.std(oracle_vals, ddof=1) / np.sqrt(len(oracle_vals))
                     if len(oracle_vals) > 1 else 0.)
    else:
        oracle_m, oracle_se = np.nan, 0.

    mean_xs, erm_ms, erm_ses, eqrm_ms, eqrm_ses = [], [], [], [], []

    for mean_e, (e1, e2) in SPURIOUS_ENV_PAIRS:
        def _recs(alg):
            return [r for r in records
                    if r.get("algorithm") == alg
                    and r.get("args", {}).get("train_env_ps") == [e1, e2]]

        for alg, ms_list, se_list in [("erm",  erm_ms,  erm_ses),
                                       ("eqrm", eqrm_ms, eqrm_ses)]:
            vals = [get_acc(r, ood_p, ms) for r in _recs(alg)
                    if get_acc(r, ood_p, ms) is not None]
            if vals:
                ms_list.append(np.mean(vals))
                se_list.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                                if len(vals) > 1 else 0.)
            else:
                ms_list.append(np.nan)
                se_list.append(0.)

        mean_xs.append(mean_e)

    mean_xs  = np.array(mean_xs)
    erm_ms   = np.array(erm_ms);   erm_ses  = np.array(erm_ses)
    eqrm_ms  = np.array(eqrm_ms);  eqrm_ses = np.array(eqrm_ses)

    # ---- Left panel: shaded phases ----
    ax_l.axvspan(CAUSAL_CEILING, 1.0,  alpha=0.06, color="#d62728",
                 label="_Phase 1 (Spur > Causal)")
    ax_l.axvspan(0.5, CAUSAL_CEILING,  alpha=0.06, color="#1f77b4",
                 label="_Phase 3 (Spur < Causal)")
    ax_l.axvline(CAUSAL_CEILING, color="gray", linestyle="--",
                 linewidth=1.0, alpha=0.7)
    ax_l.text(CAUSAL_CEILING + 0.005, 0.97, "Spurious = Causal\n(0.75)",
              fontsize=7.5, color="gray", va="top")

    # Oracle horizontal band
    if not np.isnan(oracle_m):
        ax_l.axhline(oracle_m, color=ALGO_COLORS["oracle"], linestyle=":",
                     linewidth=1.8, label=f"Oracle (n={len(oracle_recs)})")
        ax_l.fill_between([0.48, 0.92],
                          [oracle_m - oracle_se] * 2,
                          [oracle_m + oracle_se] * 2,
                          alpha=0.10, color=ALGO_COLORS["oracle"])

    n_erm  = sum(1 for r in records if r.get("algorithm") == "erm"
                 and r.get("args", {}).get("train_env_ps") == list(SPURIOUS_ENV_PAIRS[0][1]))
    n_eqrm = sum(1 for r in records if r.get("algorithm") == "eqrm"
                 and r.get("args", {}).get("train_env_ps") == list(SPURIOUS_ENV_PAIRS[0][1]))

    ax_l.errorbar(mean_xs, erm_ms,  yerr=erm_ses,
                  color=ALGO_COLORS["erm"],  marker="s", linewidth=2,
                  capsize=4, label=f"ERM (n={n_erm})")
    ax_l.errorbar(mean_xs, eqrm_ms, yerr=eqrm_ses,
                  color=ALGO_COLORS["eqrm"], marker="o", linewidth=2,
                  capsize=4, label=f"EQRM α={-10000} (n={n_eqrm})")

    ax_l.set_xlabel("Mean spurious correlation  (mean of e₁, e₂)")
    ax_l.set_ylabel(f"OOD Accuracy  (p = {ood_p})")
    ax_l.set_title("OOD Accuracy vs Spurious Strength\n"
                   "(causal ceiling = 0.75,  label noise = 25%)")
    ax_l.set_xlim(0.48, 0.92)
    ax_l.set_ylim(0.0, 1.05)
    ax_l.invert_xaxis()   # stronger shortcut on the left
    ax_l.set_xticks(mean_xs)
    ax_l.set_xticklabels([f"{x:.2f}" for x in mean_xs], rotation=45, fontsize=8)
    ax_l.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    # Phase labels
    ax_l.text(0.83, 0.05, "Phase 1\nSpur > Causal",
              transform=ax_l.transData, fontsize=7.5, color="#d62728",
              ha="center", va="bottom", alpha=0.8)
    ax_l.text(0.62, 0.05, "Phase 3\nSpur < Causal",
              transform=ax_l.transData, fontsize=7.5, color="#1f77b4",
              ha="center", va="bottom", alpha=0.8)
    ax_l.legend(loc="upper right", fontsize=9)

    # ---- Right panel: risk profiles at 3 key configs ----
    highlight = [(0.90, "mean=0.90  (Phase 1)", "#d62728"),
                 (0.75, "mean=0.75  (transition)", "#7f7f7f"),
                 (0.50, "mean=0.50  (Phase 3)", "#1f77b4")]

    for mean_e, label, color in highlight:
        pair = next(p for m, p in SPURIOUS_ENV_PAIRS if m == mean_e)
        for alg, ls, mk in [("erm",  "--", "s"), ("eqrm", "-", "o")]:
            recs = [r for r in records
                    if r.get("algorithm") == alg
                    and r.get("args", {}).get("train_env_ps") == list(pair)]
            if not recs:
                continue
            means, ses, n = group_mean_se(recs, ALL_PS, ms)
            ax_r.plot(ALL_PS, means, ls, color=color, linewidth=1.8,
                      marker=mk, markevery=3,
                      label=f"{alg.upper()}  {label} (n={n})")
            ax_r.fill_between(ALL_PS, means - ses, means + ses,
                              alpha=0.08, color=color)

    ax_r.axvline(ood_p, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_r.text(ood_p, 0.02, "OOD", transform=ax_r.get_xaxis_transform(),
              fontsize=7, color="gray", ha="center", va="bottom", rotation=90)
    ax_r.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    ax_r.set_xlabel("Test flip probability  p")
    ax_r.set_ylabel("Accuracy")
    ax_r.set_title("Risk Profiles — Phase 1 vs Transition vs Phase 3\n"
                   "(solid=EQRM, dashed=ERM)")
    ax_r.set_xlim(-0.02, 1.02)
    ax_r.set_ylim(0.0, 1.05)
    ax_r.legend(fontsize=7, loc="upper left", ncol=1)

    fig.suptitle(
        "Spurious Strength Ablation — Binary CMNIST  "
        "(OOD p=0.9 ;  label noise=25% ;  EQRM α=−10000)",
        fontsize=11
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig_spurious_ablation")


# ---------------------------------------------------------------------------
# Figure 20: Cross-dataset OOD summary
# ---------------------------------------------------------------------------

def plot_figure20_cross_dataset(all_data, figures_dir, ms="best"):
    """
    Grouped bar chart: OOD accuracy per dataset per algorithm.
    One cluster per dataset, bars per algorithm.

    BIM and RMNIST always use ms='final': early-stopping on OOD for these
    datasets selects a pre-shortcut checkpoint that is optimistically biased.
    """
    # (label, exp_name, key_type, ood_val, ms_override)
    datasets = [
        ("Binary\nCMNIST",   "reproduce",         "p",     0.9,             ms),
        ("MC-\nCMNIST",      "mc_reproduce",      "p",     0.9,             ms),
        ("BIM",              "bim_reproduce",     "p",     0.9,             "final"),
        ("CFashion\nMNIST",  "cfmnist_reproduce", "p",     0.9,             ms),
        ("RMNIST",           "rmnist_reproduce",  "delta", RMNIST_OOD_DELTA, "final"),
    ]
    algorithms = ["oracle", "erm", "eqrm", "irm", "vrex"]
    alg_labels = {"oracle": "Oracle", "erm": "ERM", "eqrm": "EQRM",
                  "irm": "IRM", "vrex": "VREx"}

    # Collect (mean, se) per (dataset_label, algorithm)
    res = {}
    for ds_label, exp_name, key_type, ood_val, ds_ms in datasets:
        records = all_data.get(exp_name, [])
        if not records:
            continue
        for alg in algorithms:
            alg_recs = [r for r in records if r.get("algorithm") == alg]
            if alg in ("eqrm",):
                if key_type == "p":
                    alg_recs, _ = best_alpha_records(alg_recs, ood_p=ood_val, ms=ds_ms)
                else:
                    alg_recs, _ = best_alpha_records_delta(alg_recs, ood_delta=ood_val, ms=ds_ms)
            elif alg in ("irm", "vrex"):
                remapped = [{**r, "args": {**r.get("args", {}),
                             "alpha": r.get("args", {}).get("penalty_weight")}}
                            for r in alg_recs]
                if key_type == "p":
                    best_recs, _ = best_alpha_records(remapped, alg=alg, ood_p=ood_val, ms=ds_ms)
                else:
                    best_recs, _ = best_alpha_records_delta(remapped, alg=alg, ood_delta=ood_val, ms=ds_ms)
                if best_recs:
                    pen = best_recs[0]["args"]["alpha"]
                    alg_recs = [r for r in alg_recs
                                if r.get("args", {}).get("penalty_weight") == pen]
                else:
                    alg_recs = []
            if not alg_recs:
                res[(ds_label, alg)] = (np.nan, 0.)
                continue
            if key_type == "p":
                vals = [get_acc(r, ood_val, ds_ms) for r in alg_recs
                        if get_acc(r, ood_val, ds_ms) is not None]
            else:
                vals = [get_acc_delta(r, ood_val, ds_ms) for r in alg_recs
                        if get_acc_delta(r, ood_val, ds_ms) is not None]
            if vals:
                res[(ds_label, alg)] = (
                    np.mean(vals),
                    np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.
                )
            else:
                res[(ds_label, alg)] = (np.nan, 0.)

    ds_labels = [d[0] for d in datasets]
    n_alg = len(algorithms)
    width = 0.14
    offsets = np.linspace(-(n_alg - 1) / 2, (n_alg - 1) / 2, n_alg) * width
    x = np.arange(len(ds_labels))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, alg in enumerate(algorithms):
        means = [res.get((ds, alg), (np.nan, 0.))[0] for ds in ds_labels]
        ses   = [res.get((ds, alg), (np.nan, 0.))[1] for ds in ds_labels]
        ax.bar(x + offsets[i], means, width,
               color=ALGO_COLORS.get(alg, "#888888"),
               label=alg_labels[alg],
               yerr=ses, capsize=3, alpha=0.85, error_kw={"linewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=10)
    ax.set_ylabel("OOD Accuracy")
    ax.set_title("Fig. 20 — Cross-Dataset OOD Accuracy\n"
                 "(p=0.9 for CMNIST variants;  δ=−60° for RMNIST)")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, figures_dir, "fig20_cross_dataset_ood")


# ---------------------------------------------------------------------------
# Figure 21: RMNIST grid — full risk profiles
# ---------------------------------------------------------------------------

def plot_figure21_rmnist_grid_profiles(records, figures_dir, ms="best"):
    """
    2×3 grid of δ-sweep risk profiles, one panel per grid config.
    Rows: N=2, N=4.  Cols: UltraNarrow, Narrow, Wide.
    Each panel: Oracle (dashed), ERM, EQRM (α=−10000).
    """
    if not records:
        print("  [fig21] No rmnist_grid records found, skipping.")
        return
    print("  Plotting Figure 21: RMNIST grid risk profiles ...")

    spreads = ["UltraNarrow", "Narrow", "Wide"]
    ns      = [2, 4]
    deltas  = ALL_DELTAS

    fig, axes = plt.subplots(len(ns), len(spreads),
                             figsize=(14, 7), sharey=True, sharex=True)

    # Oracle is shared across all configs
    oracle_recs = [r for r in records if r.get("algorithm") == "oracle"]
    if oracle_recs:
        oracle_m, oracle_s, _ = group_mean_se_delta(oracle_recs, deltas, ms)

    for row, n in enumerate(ns):
        for col, spread in enumerate(spreads):
            ax  = axes[row, col]
            cfg = f"N{n}_{spread}"

            if cfg not in _GRID_CONFIGS:
                ax.set_visible(False)
                continue

            train_deltas = list(_GRID_CONFIGS[cfg]["deltas"])
            delta_range  = max(train_deltas) - min(train_deltas)

            # Oracle
            if oracle_recs:
                ax.plot(deltas, oracle_m, "--", color=ALGO_COLORS["oracle"],
                        linewidth=1.5, label="Oracle", alpha=0.8)
                ax.fill_between(deltas,
                                oracle_m - oracle_s, oracle_m + oracle_s,
                                alpha=0.07, color=ALGO_COLORS["oracle"])

            # ERM
            erm_recs = _grid_records(records, cfg, "erm")
            if erm_recs:
                m, s, n_runs = group_mean_se_delta(erm_recs, deltas, ms)
                ax.plot(deltas, m, "-", color=ALGO_COLORS["erm"],
                        linewidth=2, label=f"ERM (n={n_runs})")
                ax.fill_between(deltas, m - s, m + s,
                                alpha=0.12, color=ALGO_COLORS["erm"])

            # EQRM (single alpha = -10000)
            eqrm_recs = _grid_records(records, cfg, "eqrm", alpha=-10000)
            if eqrm_recs:
                m, s, n_runs = group_mean_se_delta(eqrm_recs, deltas, ms)
                ax.plot(deltas, m, "-", color=ALGO_COLORS["eqrm"],
                        linewidth=2, label=f"EQRM α=−10k (n={n_runs})")
                ax.fill_between(deltas, m - s, m + s,
                                alpha=0.12, color=ALGO_COLORS["eqrm"])

            # Mark train envs and OOD delta
            for d in train_deltas:
                ax.axvline(d, color="gray", linestyle="--",
                           linewidth=0.7, alpha=0.45)
            ax.axvline(RMNIST_OOD_DELTA, color="#d62728", linestyle="--",
                       linewidth=1.0, alpha=0.7)
            ax.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.3)

            ax.set_title(f"N={n},  {spread}  (range={delta_range:.0f}°)", fontsize=10)
            ax.set_xlim(-95, 95)
            ax.set_ylim(0.0, 1.05)

            if row == len(ns) - 1:
                ax.set_xlabel("δ (°)", fontsize=9)
            if col == 0:
                ax.set_ylabel("Accuracy", fontsize=9)
            if row == 0 and col == len(spreads) - 1:
                ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(
        "Fig. 21 — RMNIST Grid: Full Risk Profiles\n"
        "(red dashed = OOD δ=−60°,  gray dashed = train envs)",
        fontsize=11
    )
    fig.tight_layout()
    _save(fig, figures_dir, "fig21_rmnist_grid_profiles")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, figures_dir, name):
    os.makedirs(figures_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(figures_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved → {figures_dir}/{name}.pdf / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Figures for the 6-experiment CMNIST test plan."
    )
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--ms",          type=str, default="best",
                        choices=["best", "final"])
    parser.add_argument("--dataset",     type=str, default="all",
                        choices=["binary", "mc", "bim", "cfmnist",
                                 "rmnist", "rmnist_grid", "cross",
                                 "burnin", "spurious", "all"])
    args = parser.parse_args()

    if args.figures_dir is None:
        args.figures_dir = os.path.join(args.output_dir, "results", "figures")

    print(f"\nLoading results from {args.output_dir} ...")
    data = load_results(args.output_dir)

    reproduce_recs         = data.get("reproduce", [])
    mc_reproduce_recs      = data.get("mc_reproduce", [])
    bim_reproduce_recs     = data.get("bim_reproduce", [])
    cfmnist_reproduce_recs = data.get("cfmnist_reproduce", [])
    rmnist_reproduce_recs  = data.get("rmnist_reproduce", [])
    rmnist_grid_recs       = data.get("rmnist_grid", [])
    burnin_recs            = data.get("burnin", [])
    spurious_recs          = data.get("spurious", [])

    print(f"\nGenerating figures -> {args.figures_dir}")

    if args.dataset in ("binary", "all"):
        plot_figure1_reproduce(reproduce_recs, args.figures_dir, ms=args.ms)

    if args.dataset in ("mc", "all"):
        plot_figure6_mc_reproduce(mc_reproduce_recs, args.figures_dir, ms=args.ms)

    if args.dataset in ("bim", "all"):
        # BIM/RMNIST: use final checkpoint — early stopping on OOD selects a
        # pre-shortcut model that is optimistically biased; final is the honest
        # measure of algorithm convergence.
        plot_figure9_bim_reproduce(bim_reproduce_recs, args.figures_dir, ms="final")

    if args.dataset in ("cfmnist", "all"):
        plot_figure13_cfmnist_reproduce(cfmnist_reproduce_recs, args.figures_dir, ms=args.ms)

    if args.dataset in ("rmnist", "all"):
        plot_figure16_rmnist_reproduce(rmnist_reproduce_recs, args.figures_dir, ms="final")

    if args.dataset in ("rmnist_grid", "all"):
        plot_figure19_rmnist_grid(rmnist_grid_recs, args.figures_dir, ms="final")
        plot_figure21_rmnist_grid_profiles(rmnist_grid_recs, args.figures_dir, ms="final")

    if args.dataset in ("cross", "all"):
        # Pass per-dataset ms overrides via the data dict directly
        plot_figure20_cross_dataset(data, args.figures_dir, ms=args.ms)

    if args.dataset in ("burnin", "all"):
        plot_figure_burnin(burnin_recs, args.figures_dir, ms=args.ms)

    if args.dataset in ("spurious", "all"):
        plot_figure_spurious(spurious_recs, args.figures_dir, ms=args.ms)

    print("\nDone.")
