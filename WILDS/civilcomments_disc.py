"""
CivilComments dataset composition analysis.

Generates four figures saved to wilds_logs/civilcomments/disc/:
  identity_prevalence.png  % of comments mentioning each identity group
  toxicity_by_identity.png toxic vs non-toxic rate per identity (binarized)
  group_imbalance.png sample count for every (identity * toxicity) groupby combo
  identity_correlation.png pairwise co-occurrence of identity mentions
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "wilds_data", "civilcomments_v1.0", "all_data_with_identities.csv")
OUT  = os.path.join(HERE, "wilds_logs", "civilcomments", "disc")
os.makedirs(OUT, exist_ok=True)

# Load & binarize
df = pd.read_csv(CSV)

# WILDS binarization threshold for both toxicity and identity fields
THRESH = 0.5
df["toxic"] = (df["toxicity"] >= THRESH).astype(int)

# Identities used in this study (grouped / readable names)
IDENTITY_COLS = {
    "black":              "Black",
    "white":              "White",
    "muslim":             "Muslim",
    "christian":          "Christian",
    "LGBTQ":              "LGBTQ+",
    "female":             "Female",
    "male":               "Male",
    "asian_latino_etc":   "Asian / Latino",
    "disability_any":     "Disability",
    "other_religions":    "Other religion",
}

# Binarize identity columns
for col in IDENTITY_COLS:
    df[col] = (df[col] >= THRESH).astype(int)

train = df[df["split"] == "train"].copy()

# Identity prevalence in the training set (for each identity, what % of comments mention it?)
prev = {label: train[col].mean() * 100
        for col, label in IDENTITY_COLS.items()}
prev = dict(sorted(prev.items(), key=lambda x: -x[1]))

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(list(prev.keys()), list(prev.values()), color="steelblue", edgecolor="white")
ax.set_xlabel("% of training comments mentioning identity")
ax.set_title("Prevalence of Identity Groups (train split)")
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_xlim(0, max(prev.values()) * 1.25)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "identity_prevalence.png"), dpi=150)
plt.close(fig)
print("Saved identity_prevalence.png")

# Toxicity rate by identity (toxic vs non-toxic split)
tox_rates = {}
for col, label in IDENTITY_COLS.items():
    mentioned = train[train[col] == 1]
    not_mentioned = train[train[col] == 0]
    tox_rates[label] = {
        "mentioned":     mentioned["toxic"].mean() * 100,
        "not_mentioned": not_mentioned["toxic"].mean() * 100,
        "n_mentioned":   len(mentioned),
    }

labels   = sorted(tox_rates, key=lambda l: -tox_rates[l]["mentioned"])
x        = np.arange(len(labels))
width    = 0.38

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, [tox_rates[l]["mentioned"]     for l in labels],
               width, label="Mentioned", color="#d65627", alpha=0.85)
bars2 = ax.bar(x + width/2, [tox_rates[l]["not_mentioned"] for l in labels],
               width, label="Not mentioned", color="#3fa6f1", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Rate of Toxic Comments per Identity (%)")
ax.set_title("Mentioned vs not-mentioned identity (train)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend()

# Annotate n for each identity
for i, label in enumerate(labels):
    n = tox_rates[label]["n_mentioned"]
    ax.text(i - width/2, tox_rates[label]["mentioned"] + 0.5,
            f"n={n:,}", ha="center", va="bottom", fontsize=7, color="#d65627")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "toxicity_by_identity.png"), dpi=150)
plt.close(fig)
print("Saved toxicity_by_identity.png")

# Group imbalance ie: sample counts for all (identity * toxic) combos
# WILDS groupby logic for any binary identity field
group_data = []
for col, label in IDENTITY_COLS.items():
    for id_val in [0, 1]:
        for tox_val in [0, 1]:
            n = len(train[(train[col] == id_val) & (train["toxic"] == tox_val)])
            group_data.append({
                "identity":   label,
                "id_mention": "mentioned" if id_val == 1 else "not mentioned",
                "toxic":      "toxic" if tox_val == 1 else "non-toxic",
                "n":          n,
            })

gdf = pd.DataFrame(group_data)
pivot = gdf.pivot_table(
    index=["identity", "id_mention"],
    columns="toxic",
    values="n",
    aggfunc="sum",
).reset_index()
pivot["total"] = pivot["non-toxic"] + pivot["toxic"]
pivot = pivot.sort_values("total", ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
y_labels = [f"{r['identity']}\n({r['id_mention']})" for _, r in pivot.iterrows()]
y        = np.arange(len(y_labels))
w        = 0.38

ax.barh(y - w/2, pivot["non-toxic"], w, label="non-toxic", color="#3fa6f1", alpha=0.85)
ax.barh(y + w/2, pivot["toxic"],     w, label="toxic",     color="#d65627", alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(y_labels, fontsize=8)
ax.set_xlabel("Number of training samples (log)")
ax.set_xscale("log")
ax.set_title("Group sizes for all (identity & toxicity) pairs")
ax.legend()
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "group_imbalance.png"), dpi=150)
plt.close(fig)
print("Saved group_imbalance.png")

# Identity co-occurrence matrix (among mentioned comments only)
id_cols  = list(IDENTITY_COLS.keys())
id_names = list(IDENTITY_COLS.values())
mentioned = train[train["identity_any"] >= 0.5][id_cols]

# Jaccard similarity between identity columns
n = len(id_cols)
jaccard = np.zeros((n, n))
for i, c1 in enumerate(id_cols):
    for j, c2 in enumerate(id_cols):
        inter = (mentioned[c1] & mentioned[c2]).sum()
        union = (mentioned[c1] | mentioned[c2]).sum()
        jaccard[i, j] = inter / union if union > 0 else 0

fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(
    jaccard, xticklabels=id_names, yticklabels=id_names,
    annot=True, fmt=".2f", cmap="Blues", linewidths=0.5,
    vmin=0, vmax=1, ax=ax,
)
ax.set_title("Identity co-occurrence (among comments mentioning any identity)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "identity_correlation.png"), dpi=150)
plt.close(fig)
print("Saved identity_correlation.png")

print(f"\nAll figures saved to {OUT}")
