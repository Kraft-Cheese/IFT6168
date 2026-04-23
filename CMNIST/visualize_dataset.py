#!/usr/bin/env python3
"""
Save grid visualizations for every dataset family in CMNIST.

Each figure shows:
  - one row per environment
  - training environments first, then test environments
  - a small sample grid from that environment

Supported datasets
------------------
  cmnist   : binary ColoredMNIST
  mc       : MC-CMNIST (10-class, 10-color)
  bim      : binary interleaved multi-color shortcut
  cfmnist  : Colored Fashion-MNIST
  rmnist   : Rotated MNIST

Usage
-----
    python visualize_dataset.py --dataset all
    python visualize_dataset.py --dataset mc --samples_per_env 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import get_cmnist_datasets
from datasets_bim import get_bim_datasets
from datasets_cfmnist import get_cfmnist_datasets
from datasets_mc import get_mc_datasets
from datasets_rmnist import get_rmnist_datasets


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    getter: Callable
    train_envs: tuple[float, ...]
    test_envs: tuple[float, ...]
    env_symbol: str
    env_unit: str = ""
    getter_kwargs: dict = field(default_factory=dict)


DATASET_SPECS = {
    "cmnist": DatasetSpec(
        name="cmnist",
        title="Binary CMNIST",
        getter=get_cmnist_datasets,
        train_envs=(0.1, 0.2),
        test_envs=(0.9,),
        env_symbol="p",
        getter_kwargs={"int_target": True},
    ),
    "mc": DatasetSpec(
        name="mc",
        title="MC-CMNIST",
        getter=get_mc_datasets,
        train_envs=(0.1, 0.2),
        test_envs=(0.9,),
        env_symbol="p",
    ),
    "bim": DatasetSpec(
        name="bim",
        title="BIM-CMNIST",
        getter=get_bim_datasets,
        train_envs=(0.1, 0.2),
        test_envs=(0.9,),
        env_symbol="p",
    ),
    "cfmnist": DatasetSpec(
        name="cfmnist",
        title="CFMNIST",
        getter=get_cfmnist_datasets,
        train_envs=(0.1, 0.2),
        test_envs=(0.9,),
        env_symbol="p",
    ),
    "rmnist": DatasetSpec(
        name="rmnist",
        title="RMNIST",
        getter=get_rmnist_datasets,
        train_envs=(60.0, 30.0),
        test_envs=(-60.0,),
        env_symbol="delta",
        env_unit="deg",
    ),
}


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _resolve_data_dir(data_dir_arg: str) -> str:
    candidate = Path(data_dir_arg)
    if candidate.exists():
        return str(candidate)

    script_dir = Path(__file__).resolve().parent
    repo_candidate = (script_dir.parent / data_dir_arg).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    return str(candidate)


def _format_env_value(spec: DatasetSpec, value: float) -> str:
    if spec.env_unit == "deg":
        value_int = int(round(value))
        return f"{value_int}\N{DEGREE SIGN}"
    return f"{value:.1f}"


def _row_label(split: str, spec: DatasetSpec, value: float, n_samples: int) -> str:
    symbol = "p" if spec.env_symbol == "p" else "\N{GREEK SMALL LETTER DELTA}"
    return f"{split}\n{symbol}={_format_env_value(spec, value)}\nn={n_samples}"


def _target_to_text(target: torch.Tensor) -> str:
    if isinstance(target, torch.Tensor):
        if target.numel() == 1:
            target = target.item()
        else:
            target = target.reshape(-1)[0].item()
    if isinstance(target, float) and target.is_integer():
        target = int(target)
    return str(target)


def _image_to_numpy(image: torch.Tensor) -> tuple[np.ndarray, str | None]:
    image = image.detach().cpu().float()
    if image.ndim == 2:
        return image.numpy(), "gray"

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

    channels, height, width = image.shape

    if channels == 1:
        return image[0].numpy(), "gray"

    if channels == 2:
        rgb = torch.zeros(3, height, width, dtype=image.dtype)
        rgb[0] = image[0]
        rgb[1] = image[1]
        return rgb.permute(1, 2, 0).clamp(0.0, 1.0).numpy(), None

    return image[:3].permute(1, 2, 0).clamp(0.0, 1.0).numpy(), None


def _sample_indices(n_available: int, n_requested: int, generator: torch.Generator) -> list[int]:
    n_take = min(n_available, n_requested)
    return torch.randperm(n_available, generator=generator)[:n_take].tolist()


def _make_figure(spec: DatasetSpec,
                 datasets: list,
                 samples_per_env: int,
                 output_dir: Path,
                 seed: int) -> None:
    env_values = list(spec.train_envs) + list(spec.test_envs)
    split_names = ["Train"] * len(spec.train_envs) + ["Test"] * len(spec.test_envs)
    n_rows = len(datasets)
    n_cols = samples_per_env

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.45 * n_cols + 1.2, 1.6 * n_rows + 1.2),
        squeeze=False,
    )

    generator = torch.Generator().manual_seed(seed)

    for row, (dataset, split_name, env_value) in enumerate(zip(datasets, split_names, env_values)):
        images, targets = dataset.tensors
        sample_ids = _sample_indices(len(images), n_cols, generator)

        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")

            if col >= len(sample_ids):
                continue

            idx = sample_ids[col]
            image_np, cmap = _image_to_numpy(images[idx])
            if cmap is None:
                ax.imshow(image_np)
            else:
                ax.imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_title(f"y={_target_to_text(targets[idx])}", fontsize=8, pad=2)

        axes[row, 0].set_ylabel(
            _row_label(split_name, spec, env_value, len(images)),
            rotation=0,
            labelpad=52,
            va="center",
            ha="right",
            fontsize=9,
        )

    for col in range(n_cols):
        axes[0, col].set_xlabel(f"Sample {col + 1}", fontsize=8, labelpad=6)
        axes[0, col].xaxis.set_label_position("top")

    fig.suptitle(
        f"{spec.title} Environment Grid\n"
        "Rows: train environments first, then test environments",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0.06, 0.02, 1.0, 0.95))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{spec.name}_env_grid"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {stem.with_suffix('.png')}")


def _load_default_datasets(spec: DatasetSpec,
                           data_dir: str,
                           subsample: bool,
                           use_test_set: bool) -> list:
    kwargs = dict(spec.getter_kwargs)
    kwargs.update({
        "train_envs": spec.train_envs,
        "test_envs": spec.test_envs,
        "subsample": subsample,
        "cuda": False,
        "use_test_set": use_test_set,
    })
    return spec.getter(data_dir, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize train/test environments for CMNIST-family datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["cmnist", "mc", "bim", "cfmnist", "rmnist", "all"],
        help="Which dataset family to visualize.",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figures/dataset_grids",
        help="Directory where the grid images will be saved.",
    )
    parser.add_argument("--samples_per_env", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--full_resolution",
        action="store_true",
        help="Use 28x28 images instead of the default 14x14 subsampled view.",
    )
    parser.add_argument(
        "--use_test_set",
        action="store_true",
        help="Use the official test split for test environments instead of the held-out train split.",
    )
    args = parser.parse_args()

    _seed_everything(args.seed)
    data_dir = _resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    dataset_names = list(DATASET_SPECS) if args.dataset == "all" else [args.dataset]

    for dataset_name in dataset_names:
        spec = DATASET_SPECS[dataset_name]
        print(f"Rendering {spec.title} ...")
        datasets = _load_default_datasets(
            spec,
            data_dir=data_dir,
            subsample=not args.full_resolution,
            use_test_set=args.use_test_set,
        )
        _make_figure(
            spec,
            datasets,
            samples_per_env=args.samples_per_env,
            output_dir=output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
