"""
PovertyMap dataset composition analysis.

Generates figures saved to wilds_logs/poverty/disc/:
  country_wealth.png wealth distribution per country (box), grouped by split
  urban_rural_wealth.png urban vs rural wealth per country (train only)
  ood_coverage.png test/val wealth ranges vs training envelope (EQRM assumption)
  nightlights_wealth.png nightlight intensity vs wealth index (feature quality)
  group_sizes.png sample count per country per split
  geo_map.png lat/lon scatter coloured by wealth index
  year_distribution.png survey year distribution per split
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

# Paths
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Volumes/WILDS_DATA/wilds_data"
OUT      = os.path.join(HERE, "wilds_logs", "poverty", "disc")
os.makedirs(OUT, exist_ok=True)

# Load data + split labels from WILDS
import sys
sys.path.insert(0, os.path.join(HERE, "examples"))
import wilds

dataset = wilds.get_dataset("poverty", root_dir=DATA_DIR)

df = pd.read_csv(os.path.join(DATA_DIR, "poverty_v1.1", "dhs_metadata.csv"))
df["split"]      = dataset.split_array
split_names      = {v: k for k, v in dataset.split_dict.items()}
df["split_name"] = df["split"].map(split_names)

# Collapse id_val/id_test into train for plotting purposes
df["split_group"] = df["split_name"].replace({"id_val": "train", "id_test": "train"})

SPLIT_COLORS = {
    "train": "#1f77b4",
    "val":   "#ff7f0e",
    "test":  "#d62728",
}

COUNTRY_LABELS = {c: c.replace("_", " ").title() for c in df["country"].unique()}

# Wealth distribution per country, grouped by split
fig, ax = plt.subplots(figsize=(15, 6))

# Sort countries: train first, then val, then test
country_order = (
    sorted(df[df["split_group"] == "train"]["country"].unique()) +
    sorted(df[df["split_group"] == "val" ]["country"].unique()) +
    sorted(df[df["split_group"] == "test"]["country"].unique())
)

box_data  = [df[df["country"] == c]["wealthpooled"].values for c in country_order]
positions = range(len(country_order))
bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                widths=0.6, showfliers=False,
                medianprops=dict(color="black", lw=1.5))

for i, (patch, country) in enumerate(zip(bp["boxes"], country_order)):
    sg = df[df["country"] == country]["split_group"].iloc[0]
    patch.set_facecolor(SPLIT_COLORS[sg])
    patch.set_alpha(0.7)

ax.set_xticks(list(positions))
ax.set_xticklabels([COUNTRY_LABELS[c] for c in country_order],
                   rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Wealth index (wealthpooled)")
ax.set_title("Wealth distribution per country (IQR)")
ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.6)

# Separator between train / val / test
n_train = len(df[df["split_group"] == "train"]["country"].unique())
n_val   = len(df[df["split_group"] == "val" ]["country"].unique())
ax.axvline(n_train - 0.5,         color="black", lw=1.2, linestyle=":")
ax.axvline(n_train + n_val - 0.5, color="black", lw=1.2, linestyle=":")
ax.text(n_train / 2 - 0.5,           ax.get_ylim()[1] * 0.97, "TRAIN", ha="center", fontsize=9, color=SPLIT_COLORS["train"])
ax.text(n_train + n_val / 2 - 0.5,   ax.get_ylim()[1] * 0.97, "VAL (OOD)", ha="center", fontsize=9, color=SPLIT_COLORS["val"])
ax.text(n_train + n_val + len(df[df["split_group"]=="test"]["country"].unique()) / 2 - 0.5,
        ax.get_ylim()[1] * 0.97, "TEST (OOD)", ha="center", fontsize=9, color=SPLIT_COLORS["test"])

handles = [mpatches.Patch(color=c, alpha=0.7, label=l.upper())
           for l, c in SPLIT_COLORS.items()]
ax.legend(handles=handles, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "country_wealth.png"), dpi=150)
plt.close(fig)
print("Saved country_wealth.png")

# Urban vs rural wealth per country (train only)
train = df[df["split_group"] == "train"].copy()
train["urban_label"] = train["urban"].map({True: "Urban", False: "Rural"})

fig, ax = plt.subplots(figsize=(13, 5))
countries_train = sorted(train["country"].unique())
x = np.arange(len(countries_train))
width = 0.35

urban_means = train[train["urban"]]["groupby_country_mean"] if False else \
    [train[(train["country"] == c) & (train["urban"] == True)]["wealthpooled"].mean()
     for c in countries_train]
rural_means = [train[(train["country"] == c) & (train["urban"] == False)]["wealthpooled"].mean()
               for c in countries_train]
urban_stds  = [train[(train["country"] == c) & (train["urban"] == True)]["wealthpooled"].std()
               for c in countries_train]
rural_stds  = [train[(train["country"] == c) & (train["urban"] == False)]["wealthpooled"].std()
               for c in countries_train]

ax.bar(x - width/2, urban_means, width, yerr=urban_stds, label="Urban",
       color="#e377c2", alpha=0.8, capsize=3)
ax.bar(x + width/2, rural_means, width, yerr=rural_stds, label="Rural",
       color="#8c564b", alpha=0.8, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels([COUNTRY_LABELS[c] for c in countries_train], rotation=35, ha="right")
ax.set_ylabel("Mean wealth index (+/-1 std)")
ax.set_title("Urban vs rural wealth by country (train split)")
ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "urban_rural_wealth.png"), dpi=150)
plt.close(fig)
print("Saved urban_rural_wealth.png")

# OOD coverage for EQRM's assumption: test/val wealth ranges vs training envelope
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, ood_split, title in [
    (axes[0], "val",  "OOD Val countries vs Training envelope"),
    (axes[1], "test", "OOD Test countries vs Training envelope"),
]:
    # Training envelope: 5th–95th percentile band per wealth value
    train_q05 = train["wealthpooled"].quantile(0.05)
    train_q95 = train["wealthpooled"].quantile(0.95)
    train_med = train["wealthpooled"].median()

    ood = df[df["split_group"] == ood_split]
    ood_countries = sorted(ood["country"].unique())

    # KDE of training wealth
    from scipy.stats import gaussian_kde
    kde_x = np.linspace(-2, 3.5, 300)
    kde_train = gaussian_kde(train["wealthpooled"].dropna())(kde_x)
    ax.fill_between(kde_x, kde_train, alpha=0.15, color=SPLIT_COLORS["train"], label="Train KDE")
    ax.plot(kde_x, kde_train, color=SPLIT_COLORS["train"], lw=1.5)

    # Per OOD country KDE
    color = SPLIT_COLORS[ood_split]
    for i, c in enumerate(ood_countries):
        vals = ood[ood["country"] == c]["wealthpooled"].dropna()
        if len(vals) < 10:
            continue
        kde_c = gaussian_kde(vals)(kde_x)
        alpha = 0.5 + 0.4 * (i / max(len(ood_countries) - 1, 1))
        ax.plot(kde_x, kde_c, lw=1.5, linestyle="--",
                color=color, alpha=alpha, label=COUNTRY_LABELS[c])

    ax.set_xlabel("Wealth index")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")

fig.suptitle("OOD country distributions vs training envelope\n"
             "(EQRM assumes test risk is covered by training environment mixture)",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ood_coverage.png"), dpi=150)
plt.close(fig)
print("Saved ood_coverage.png")

# Nightlights vs wealth (feature quality / spurious correlation risk)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, split_g, color, label in [
    (axes[0], "train", SPLIT_COLORS["train"], "Train countries"),
    (axes[1], "test",  SPLIT_COLORS["test"],  "Test countries"),
]:
    sub = df[df["split_group"] == split_g].sample(min(3000, len(df[df["split_group"] == split_g])), random_state=0)
    ax.scatter(sub["nl_mean"], sub["wealthpooled"],
               alpha=0.25, s=8, color=color)
    # Correlation line
    m, b = np.polyfit(sub["nl_mean"], sub["wealthpooled"], 1)
    xl = np.array([sub["nl_mean"].min(), sub["nl_mean"].max()])
    ax.plot(xl, m * xl + b, color="black", lw=1.5)
    r = sub[["nl_mean", "wealthpooled"]].corr().iloc[0, 1]
    ax.set_title(f"{label}\nNightlights vs Wealth  (r={r:.2f})")
    ax.set_xlabel("Mean nightlight intensity (nl_mean)")
    ax.set_ylabel("Wealth index")

fig.suptitle("Nightlight intensity vs wealth index")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "nightlights_wealth.png"), dpi=150)
plt.close(fig)
print("Saved nightlights_wealth.png")

# Group sizes per country per split
counts = df.groupby(["country", "split_group"]).size().reset_index(name="n")
counts["country_label"] = counts["country"].map(COUNTRY_LABELS)

fig, ax = plt.subplots(figsize=(14, 5))
pivot = counts.pivot(index="country_label", columns="split_group", values="n").fillna(0)
pivot = pivot.reindex(sorted(pivot.index))
x     = np.arange(len(pivot))
width = 0.25
for i, (sg, color) in enumerate(SPLIT_COLORS.items()):
    if sg in pivot.columns:
        ax.bar(x + (i - 1) * width, pivot[sg], width,
               label=sg.upper(), color=color, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Number of survey clusters")
ax.set_title("Sample count per country and split")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "group_sizes.png"), dpi=150)
plt.close(fig)
print("Saved group_sizes.png")

# Geographic scatter coloured by wealth
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(df["lon"], df["lat"],
                c=df["wealthpooled"], cmap="RdYlGn",
                s=8, alpha=0.5, vmin=-1.5, vmax=2.5)
plt.colorbar(sc, ax=ax, label="Wealth index")

# Label split regions with text
for sg, color in SPLIT_COLORS.items():
    sub = df[df["split_group"] == sg]
    for country in sub["country"].unique():
        csub = sub[sub["country"] == country]
        ax.text(csub["lon"].mean(), csub["lat"].mean(),
                COUNTRY_LABELS[country], fontsize=6,
                ha="center", color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, lw=0))

handles = [mpatches.Patch(color=c, label=l.upper()) for l, c in SPLIT_COLORS.items()]
ax.legend(handles=handles, fontsize=8, loc="lower left")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic distribution coloured by wealth index")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "geo_map.png"), dpi=150)
plt.close(fig)
print("Saved geo_map.png")

# Survey year distribution per split
fig, ax = plt.subplots(figsize=(10, 4))
for sg, color in SPLIT_COLORS.items():
    sub = df[df["split_group"] == sg]
    counts_yr = sub["year"].value_counts().sort_index()
    ax.bar(counts_yr.index + list(SPLIT_COLORS.keys()).index(sg) * 0.25,
           counts_yr.values, 0.25, color=color, alpha=0.8, label=sg.upper())

ax.set_xlabel("Survey year")
ax.set_ylabel("Number of clusters")
ax.set_title("Survey year distribution by split")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "year_distribution.png"), dpi=150)
plt.close(fig)
print("Saved year_distribution.png")

print(f"\nAll figures saved to {OUT}")
