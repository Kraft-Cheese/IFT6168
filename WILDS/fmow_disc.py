#!/usr/bin/env python3
"""
FMoW EDA: Three pre-training analyses connecting to the paper's narrative.

  Class prior shift by year       - z_max mean-shift
  Environment size by year        - n_envs
  Temporal autocorrelation (JSD)  -IID violation (rho)

Usage:
    python fmow_eda.py
    python fmow_eda.py --root_dir /data/rech/hamelcas/wilds_data
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import jensenshannon


def load_metadata(root_dir):
    csv = Path(root_dir) / 'fmow_v1.1' / 'rgb_metadata.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df = df[df['split'] != 'seq'].copy()
    return df

# Class prior shift by year

def analysis_class_prior(df, out_dir):
    counts = df.groupby(['year', 'category']).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)

    top_cats = props.std(axis=0).nlargest(15).index

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Heatmap: all 62 categories
    im = axes[0].imshow(props.T.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xticks(range(len(props.index)))
    axes[0].set_xticklabels(props.index, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(len(props.columns)))
    axes[0].set_yticklabels(props.columns, fontsize=6)
    axes[0].set_xlabel('Year')
    axes[0].set_title('Class Prior by Year — all 62 categories')
    plt.colorbar(im, ax=axes[0], label='Proportion')

    # Line plot: top 15 most dynamic categories
    for cat in top_cats:
        axes[1].plot(props.index, props[cat], marker='o', markersize=3, linewidth=1.2, label=cat)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Proportion of images')
    axes[1].set_title('Top 15 Most Temporally Dynamic Categories')
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'class_prior_shift.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Top 10 categories by temporal variation (std across years):")
    print(f"  {'Category':<42} {'min':>6}  {'max':>6}  {'std':>6}")
    for cat in props.std(axis=0).nlargest(10).index:
        s = props[cat]
        print(f"  {cat:<42} {s.min():6.4f}  {s.max():6.4f}  {s.std():6.4f}")


# Environment size by year

def analysis_env_size(df, out_dir):
    train_df = df[df['split'] == 'train']
    year_counts = train_df.groupby('year').size()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(year_counts.index, year_counts.values,
                  color='steelblue', edgecolor='navy', alpha=0.85)

    for bar, count in zip(bars, year_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f'{count:,}', ha='center', va='bottom', fontsize=7, rotation=45)

    ratio = year_counts.max() / year_counts.min()
    ax.text(0.02, 0.97, f'Max/Min ratio: {ratio:.1f}×',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Year')
    ax.set_ylabel('Training images')
    ax.set_title('Environment Size by Year (train split)')
    ax.set_xticks(year_counts.index)
    ax.set_xticklabels(year_counts.index, rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'env_size_by_year.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Training image counts per year:")
    for yr, cnt in year_counts.items():
        bar = '█' * (cnt // max(year_counts.values // 30, 1))
        print(f"  {yr}: {cnt:6,}  {bar}")
    print(f"Max/Min ratio: {ratio:.1f}×  "
          f"(min={year_counts.min():,} in {year_counts.idxmin()}, "
          f"max={year_counts.max():,} in {year_counts.idxmax()})")

# Temporal autocorrelation via Jensen-Shannon divergence
def analysis_temporal_autocorr(df, out_dir):
    counts = df.groupby(['year', 'category']).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    years = props.index.tolist()
    n = len(years)

    # Pairwise JSD matrix
    jsd = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p = props.iloc[i].values + 1e-12
            q = props.iloc[j].values + 1e-12
            jsd[i, j] = jensenshannon(p / p.sum(), q / q.sum())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    im = axes[0].imshow(jsd, cmap='viridis', vmin=0)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(years, rotation=45, fontsize=8)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(years, fontsize=8)
    axes[0].set_title('Jensen-Shannon Divergence Between Year Pairs\n(class distribution as proxy for feature distribution)')
    plt.colorbar(im, ax=axes[0], label='JSD')

    # Consecutive (lag=1) vs distant (lag>=5) JSD
    consec = [jsd[i, i + 1] for i in range(n - 1)]
    consec_years = [f"{years[i]}→{years[i+1]}" for i in range(n - 1)]
    distant_vals = [jsd[i, j] for i in range(n) for j in range(n) if abs(i - j) >= 5]

    axes[1].plot(range(len(consec)), consec, marker='o', color='steelblue',
                 linewidth=1.5, label='Consecutive years (lag=1)')
    axes[1].axhline(np.mean(consec), color='steelblue', linestyle=':',
                    alpha=0.6, label=f'Mean consec: {np.mean(consec):.4f}')
    axes[1].axhline(np.mean(distant_vals), color='firebrick', linestyle='--',
                    label=f'Mean distant (lag≥5): {np.mean(distant_vals):.4f}')
    axes[1].set_xticks(range(len(consec)))
    axes[1].set_xticklabels(consec_years, rotation=45, ha='right', fontsize=7)
    axes[1].set_ylabel('Jensen-Shannon Divergence')
    axes[1].set_title('Consecutive vs. Distant Year Divergence\n(high ratio = strong temporal autocorrelation = IID violation)')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'temporal_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close()

    ratio = np.mean(distant_vals) / np.mean(consec)
    print(f"Mean consecutive JSD (lag=1):  {np.mean(consec):.4f} ± {np.std(consec):.4f}")
    print(f"Mean distant JSD    (lag≥5):   {np.mean(distant_vals):.4f} ± {np.std(distant_vals):.4f}")
    print(f"Distant/consecutive ratio:      {ratio:.2f}×")
    print(f"(ratio > 1 confirms temporal autocorrelation; ratio >> 1 = strong IID violation)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/data/rech/hamelcas/wilds_data',
                        help='WILDS root directory (parent of fmow_v1.1/)')
    args = parser.parse_args()

    out_dir = Path(__file__).parent / 'fmow_eda'
    out_dir.mkdir(exist_ok=True)

    print(f"Loading FMoW metadata from {args.root_dir} ...")
    df = load_metadata(args.root_dir)
    print(f"Loaded {len(df):,} images | "
          f"{df['year'].nunique()} years ({df['year'].min()}–{df['year'].max()}) | "
          f"{df['split'].value_counts().to_dict()}")

    print("Class Prior Shift by Year ===")
    analysis_class_prior(df, out_dir)

    print("Environment Size by Year ===")
    analysis_env_size(df, out_dir)

    print("Temporal Autocorrelation ===")
    analysis_temporal_autocorr(df, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")
