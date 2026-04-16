# EQRM on TableShift

Extends EQRM to **tabular data** under distribution shift using the [TableShift](https://github.com/mlfoundations/tableshift) benchmark (NeurIPS 2023).

## Key Idea

EQRM was designed and tested on image and graph data. TableShift provides 10 tabular binary classification tasks with real-world domain splits (e.g., by race, geographic region, insurance type). Each domain becomes an environment for quantile risk minimization. This tests whether EQRM's benefits transfer to an entirely new modality.

## Setup

```bash
# Recommended: use TableShift's Docker image
docker pull ghcr.io/jpgard/tableshift:latest

# Or install directly (may need extra dependencies)
pip install tableshift
pip install -r requirements.txt
```

## Quick Start

```bash
# EQRM on diabetes readmission (public dataset, auto-downloads)
python train.py --task diabetes_readmission --algorithm eqrm \
    --alpha 0.75 --steps 2000 --seed 0

# ERM baseline
python train.py --task diabetes_readmission --algorithm erm --steps 2000 --seed 0
```

## Available Tasks (with Domain Generalization support)

| Task | Shift Variable | Domain |
|---|---|---|
| `diabetes_readmission` | Admission source | Healthcare |
| `anes` | Region | Civic participation |
| `food_stamps` | Geographic region | Public policy |
| `brfss_diabetes` | Race | Healthcare |
| `brfss_blood_pressure` | Race | Healthcare |
| `heloc` | Risk level | Finance |
| `hospital_readmission` | Insurance type | Healthcare |
| `icu_mortality` | Insurance type | Healthcare |
| `assistments` | School ID | Education |
| `college_scorecard` | Institution type | Education |

## Quick Sweep

```bash
for task in diabetes_readmission anes food_stamps; do
    for algo in erm eqrm vrex groupdro; do
        for seed in 0 1 2; do
            python train.py --task $task --algorithm $algo \
                --alpha 0.75 --steps 2000 --seed $seed
        done
    done
done
```

## Project Structure

```
TableShift/
├── train.py          # Main training script
├── algorithms.py     # EQRM + ERM/VREx/IRM/GroupDRO/IGA/SD
├── datasets.py       # TableShift API → per-domain environments
├── networks.py       # TabularMLP
├── lib/
│   ├── misc.py       # KDE, Nonparametric (EQRM core math)
│   └── fast_data_loader.py
├── requirements.txt
└── README.md
```
