# EQRM on Wild-Time

Extends EQRM to the [Wild-Time](https://github.com/huaxiuyao/Wild-Time) benchmark for **temporal distribution shift**. Training timestamps are treated as environments for quantile risk minimization.

## Key Idea

Wild-Time datasets have data indexed by time (e.g., yearbook photos from 1930–2013). We partition training timestamps into **time-window environments** and apply EQRM's quantile risk minimization over these environments. This tests whether EQRM's probabilistic framework — designed for i.i.d. domains — can handle the structured, non-i.i.d. case of temporal shift.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# EQRM on Yearbook (8 time-window environments, alpha close to 1)
python train.py --dataset yearbook --algorithm eqrm \
    --num_train_envs 8 --alpha -500 \
    --erm_pretrain_iters 500 --steps 2000 --seed 0

# ERM baseline
python train.py --dataset yearbook --algorithm erm --steps 2000 --seed 0

# VREx baseline
python train.py --dataset yearbook --algorithm vrex \
    --num_train_envs 8 --penalty_weight 1000 \
    --erm_pretrain_iters 500 --steps 2000 --seed 0

# GroupDRO baseline
python train.py --dataset yearbook --algorithm groupdro \
    --num_train_envs 8 \
    --erm_pretrain_iters 500 --steps 2000 --seed 0
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `yearbook` | Wild-Time dataset name |
| `--algorithm` | `eqrm` | Algorithm: `erm`, `eqrm`, `vrex`, `irm`, `groupdro`, `iga`, `sd` |
| `--num_train_envs` | `8` | Number of time-window training environments |
| `--alpha` | `-500` | EQRM quantile. Negative = `log(1-α)` for values near 1 |
| `--erm_pretrain_iters` | `500` | Burn-in steps with ERM before switching to DG objective |
| `--steps` | `2000` | Total training steps |
| `--lr` | `1e-3` | Learning rate |
| `--lr_factor_reduction` | `10` | LR reduction factor after ERM pretrain |
| `--batch_size` | `64` | Per-environment batch size |
| `--split_time` | dataset default | Timestamp separating train/test |
| `--eval_freq` | `100` | Steps between evaluations |
| `--seed` | `0` | Random seed |

### Understanding the `--alpha` parameter

The `alpha` parameter controls EQRM's conservativeness:
- `alpha = 0.5` → median risk (similar to ERM)
- `alpha = 0.9` → 90th percentile risk
- `alpha → 1` → worst-case (approaches DRO)

For values very close to 1, use negative values which represent `log(1 - alpha)`:
- `alpha = -10` → `alpha ≈ 1 - e^{-10} ≈ 0.99995`
- `alpha = -500` → `alpha ≈ 1 - e^{-500} ≈ 1`

### Understanding `--num_train_envs`

This controls how training timestamps are grouped into environments:
- `--num_train_envs 4` → 4 roughly-equal time windows
- `--num_train_envs 41` → one environment per year (for Yearbook 1930–1970)
- More environments = richer risk distribution for KDE, but smaller per-env batches

## Evaluation Protocol

Following the EQRM paper (Section 6), we report both standard and **quantile-focused** metrics:

- **Average OOD accuracy**: Mean accuracy across all test timestamps
- **Worst OOD accuracy**: Minimum accuracy across test timestamps  
- **Quantile accuracies**: 10th, 25th, 50th, 75th, 90th percentiles of per-timestamp accuracy

This quantile evaluation reveals tail performance that average metrics hide.

## Reproducing Results

```bash
# Full sweep across algorithms and seeds
for algo in erm eqrm vrex groupdro irm; do
    for seed in 0 1 2; do
        python train.py --dataset yearbook --algorithm $algo \
            --num_train_envs 8 --erm_pretrain_iters 500 \
            --steps 2000 --seed $seed --save_ckpts
    done
done
```

## Project Structure

```
WildTime/
├── train.py          # Main training script
├── algorithms.py     # EQRM, ERM, VREx, IRM, GroupDRO, IGA, SD
├── datasets.py       # Wild-Time dataset loaders → environment format
├── networks.py       # Model architectures (CNN for Yearbook)
├── lib/
│   ├── misc.py       # KDE, Nonparametric distribution, utilities
│   └── fast_data_loader.py  # Infinite/Fast data loaders
├── requirements.txt
└── README.md
```
