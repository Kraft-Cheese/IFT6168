import json, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import defaultdict

# Load results
records = []
for f in glob.glob("results/alpha_sweep/*.jsonl"):
    with open(f) as fp:
        for line in fp:
            records.append(json.loads(line.strip()))

print(f"Loaded {len(records)} results\n")

# Group by method
grouped = defaultdict(list)
for r in records:
    algo = r.get("algorithm", "").upper()
    if algo == "EQRM":
        alpha = r["args"]["alpha"]
        label = f"EQRM (α={'1-e^{'+str(int(alpha))+'}' if alpha < 0 else alpha})"
    else:
        label = algo
    grouped[label].append(r)

# Table
rows = []
for label in sorted(grouped.keys()):
    runs = grouped[label]
    ood = [r["OOD_test_acc_best"] for r in runs if "OOD_test_acc_best" in r]
    idd = [r["ID_test_acc_best"] for r in runs if "ID_test_acc_best" in r]
    rows.append({
        "Method": label,
        "OOD Acc": f"{np.mean(ood):.4f} ± {np.std(ood):.4f}" if ood else "-",
        "ID Acc": f"{np.mean(idd):.4f} ± {np.std(idd):.4f}" if idd else "-",
    })
print(pd.DataFrame(rows).to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
labels = sorted(grouped.keys())
ood_means = [np.mean([r["OOD_test_acc_best"] for r in grouped[l] if "OOD_test_acc_best" in r]) for l in labels]
ood_stds = [np.std([r["OOD_test_acc_best"] for r in grouped[l] if "OOD_test_acc_best" in r]) for l in labels]
colors = ["#888888" if "ERM" == l else "#e41a1c" for l in labels]

ax.bar(range(len(labels)), ood_means, yerr=ood_stds, color=colors, capsize=4)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("OOD Test Accuracy", fontsize=12)
ax.set_title("ERM vs EQRM (diabetes_readmission)", fontsize=14)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("diabetes_alpha_sweep.png", dpi=150)
plt.show()