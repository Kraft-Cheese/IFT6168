"""
Wild-Time dataset wrappers for EQRM experiments.

Loads Wild-Time datasets and partitions training data into temporal environments
(time windows), producing a list of PyTorch Datasets compatible with the QRM
repo's `update(minibatches)` interface where `minibatches` is a list of (x, y)
tuples — one per environment.

Supported datasets (no credentialed access needed):
  - yearbook: gender classification from yearbook photos (1930–2013)
  - huffpost: news category classification (2012–2018)
  - arxiv: paper topic classification (2007–2023)
  - fmow: satellite land-use classification (2002–2017)
  - drug: drug-target binding affinity regression (2013–2020)

MIMIC requires PhysioNet credentials and is excluded by default.
"""

import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset

# ---------------------------------------------------------------------------
# Yearbook — simplest dataset, good for debugging
# ---------------------------------------------------------------------------

YEARBOOK_URL = "https://drive.google.com/uc?export=download&id=1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb"


def _download_yearbook(data_dir):
    """Download yearbook.pkl if not present."""
    fpath = os.path.join(data_dir, "yearbook.pkl")
    if os.path.exists(fpath):
        return fpath

    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading yearbook.pkl to {data_dir}...")

    try:
        import gdown
        gdown.download(YEARBOOK_URL, fpath, quiet=False)
    except ImportError:
        raise RuntimeError(
            "Please install gdown: pip install gdown\n"
            "Or manually download yearbook.pkl from the Wild-Time repo and place it in "
            f"{data_dir}"
        )

    if not os.path.exists(fpath):
        raise RuntimeError(f"Download failed. Please manually place yearbook.pkl in {data_dir}")

    return fpath


def get_yearbook_datasets(data_dir, split_time=1970, num_train_envs=None):
    """
    Load Yearbook and split into temporal environments.

    Args:
        data_dir: path containing yearbook.pkl
        split_time: year that separates training (<=) from test (>)
        num_train_envs: if None, each year is an environment. If an int,
                        pool years into this many roughly-equal windows.

    Returns:
        train_envs: list of TensorDatasets, one per training environment
        test_envs: list of TensorDatasets, one per test year
        env_names: dict with 'train' and 'test' lists of names
    """
    fpath = _download_yearbook(data_dir)
    dataset = pickle.load(open(fpath, "rb"))

    years = sorted(dataset.keys())
    train_years = [y for y in years if y <= split_time]
    test_years = [y for y in years if y > split_time]

    def _make_tensordataset(year_list, mode=0):
        """Combine data from given years into a single TensorDataset."""
        images_list, labels_list = [], []
        for y in year_list:
            imgs = dataset[y][mode]["images"]  # (N, 32, 32, 3), float in [0,1]
            lbls = dataset[y][mode]["labels"]  # (N,), int
            images_list.append(torch.FloatTensor(imgs).permute(0, 3, 1, 2))
            labels_list.append(torch.LongTensor(lbls))
        x = torch.cat(images_list, dim=0)
        y = torch.cat(labels_list, dim=0)
        return TensorDataset(x, y)

    # Build training environments
    if num_train_envs is None or num_train_envs >= len(train_years):
        # Each year is its own environment
        train_envs = [_make_tensordataset([y], mode=0) for y in train_years]
        train_env_names = [str(y) for y in train_years]
    else:
        # Pool years into windows
        chunks = np.array_split(train_years, num_train_envs)
        train_envs = [_make_tensordataset(list(chunk), mode=0) for chunk in chunks]
        train_env_names = [f"{chunk[0]}-{chunk[-1]}" for chunk in chunks]

    # Build test environments (one per year, using held-out test split mode=1)
    test_envs = [_make_tensordataset([y], mode=1) for y in test_years]
    test_env_names = [str(y) for y in test_years]

    env_names = {"train": train_env_names, "test": test_env_names}

    return train_envs, test_envs, env_names


# ---------------------------------------------------------------------------
# HuffPost — text classification, needs wildtime package
# ---------------------------------------------------------------------------

def get_huffpost_datasets(data_dir, split_time=2014, num_train_envs=None):
    """
    Load HuffPost news classification dataset via wildtime package.

    Requires: pip install wildtime==1.1.3

    Returns same format as get_yearbook_datasets.
    """
    try:
        from wildtime.data.huffpost import HuffPost as HuffPostData
    except ImportError:
        raise ImportError(
            "HuffPost requires the wildtime package. Install with:\n"
            "  pip install wildtime==1.1.3"
        )

    import argparse
    args = argparse.Namespace(
        data_dir=data_dir,
        random_seed=0,
        mini_batch_size=64,
        method="erm",
        reduced_train_prop=None,
    )

    dataset_obj = HuffPostData(args)
    all_years = dataset_obj.ENV

    train_years = [y for y in all_years if y <= split_time]
    test_years = [y for y in all_years if y > split_time]

    def _collect(year, mode):
        dataset_obj.mode = mode
        dataset_obj.update_current_timestamp(year)
        xs, ys = [], []
        for i in range(len(dataset_obj)):
            x, y = dataset_obj[i]
            xs.append(x)
            ys.append(y)
        return TensorDataset(torch.stack(xs), torch.cat(ys))

    # Environments
    if num_train_envs is None or num_train_envs >= len(train_years):
        train_envs = [_collect(y, 0) for y in train_years]
        train_env_names = [str(y) for y in train_years]
    else:
        chunks = np.array_split(train_years, num_train_envs)
        train_envs = []
        train_env_names = []
        for chunk in chunks:
            xs, ys = [], []
            for y in chunk:
                dataset_obj.mode = 0
                dataset_obj.update_current_timestamp(y)
                for i in range(len(dataset_obj)):
                    x, lab = dataset_obj[i]
                    xs.append(x)
                    ys.append(lab)
            train_envs.append(TensorDataset(torch.stack(xs), torch.cat(ys)))
            train_env_names.append(f"{chunk[0]}-{chunk[-1]}")

    test_envs = [_collect(y, 2) for y in test_years]
    test_env_names = [str(y) for y in test_years]

    return train_envs, test_envs, {"train": train_env_names, "test": test_env_names}


# ---------------------------------------------------------------------------
# FMoW — satellite land-use classification with BOTH temporal and geographic splits
# ---------------------------------------------------------------------------

# Region mapping used by WILDS FMoW
FMOW_REGIONS = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']


class FMoWImageDataset(Dataset):
    """
    Lazy-loading FMoW dataset that reads images from disk on demand.
    Wraps the WILDS FMoW dataset and filters by given indices.
    """

    def __init__(self, wilds_dataset, indices, transform=None):
        self.wilds_dataset = wilds_dataset
        self.indices = indices
        self.transform = transform or _fmow_default_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        wilds_idx = self.indices[idx]
        x, y, _ = self.wilds_dataset[wilds_idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def _fmow_default_transform():
    """Standard ImageNet normalization for DenseNet."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def _load_fmow_wilds(data_dir):
    """Load the WILDS FMoW dataset (downloads if needed)."""
    try:
        from wilds import get_dataset
    except ImportError:
        raise ImportError(
            "FMoW requires the wilds package. Install with:\n"
            "  pip install wilds"
        )
    dataset = get_dataset(dataset="fmow", root_dir=data_dir, download=True)
    return dataset


def get_fmow_temporal_datasets(data_dir, split_time=11, num_train_envs=None,
                               poison_mode=None, poison_fraction=0.1, poison_seed=None,
                               gap_year=None, noise_std=25.0, tint_strength=40, **kwargs):
    """
    FMoW with TEMPORAL environments (years as domains).

    The WILDS FMoW dataset spans years 2002-2017, encoded as indices 0-15.
    Training: years 0 to split_time-1. Test: years split_time to 15.

    Args:
        data_dir        : root directory for WILDS data
        split_time      : year index for train/test split (default 11 = year 2013)
        num_train_envs  : pool years into this many windows (None = one per year)
        poison_mode     : None | 'label' | 'group' | 'temporal_gap' | 'year_tint'
                          See FMoWDataset.poison() for full documentation.
                          'group' corrupts year env labels before partitioning, so
                          environments will be wrongly partitioned — testing IID violation.
                          'year_tint' adds train-only per-year RGB tints, absent at test.
                          'temporal_gap' degrades one mid-training year with Gaussian noise.
        poison_fraction : fraction of train samples to corrupt (label/group modes)
        poison_seed     : RNG seed for reproducibility
        gap_year        : calendar year to degrade, e.g. 2010 (temporal_gap mode)
        noise_std       : Gaussian noise std in pixel units (temporal_gap mode)
        tint_strength   : per-channel pixel offset magnitude (year_tint mode)

    Returns:
        train_envs, test_envs, env_names
    """
    dataset = _load_fmow_wilds(data_dir)
    transform = _fmow_default_transform()

    # 'group' mode must be applied before partitioning (corrupts year labels in metadata_array)
    # Environment split will read the corrupted values
    if poison_mode == 'group':
        dataset.poison(fraction=poison_fraction, mode='group', seed=poison_seed)

    # Get metadata
    split_array = dataset.split_array
    years = dataset.metadata_array[:, 1].numpy()  # year index

    # Use train + id_val splits as training data
    train_mask = (split_array == dataset.split_dict['train']) | \
                 (split_array == dataset.split_dict['id_val'])
    test_mask = (split_array == dataset.split_dict['test']) | \
                (split_array == dataset.split_dict['val'])

    # Training environments: group by year
    train_indices = np.where(train_mask)[0]
    train_years = years[train_indices]
    unique_train_years = sorted(set(train_years[train_years < split_time]))

    if num_train_envs is None or num_train_envs >= len(unique_train_years):
        # One env per year
        train_envs = []
        train_env_names = []
        for y in unique_train_years:
            idx = train_indices[train_years == y]
            train_envs.append(FMoWImageDataset(dataset, idx, transform))
            train_env_names.append(f"yr{int(y)+2002}")
    else:
        # Pool years into windows
        chunks = np.array_split(unique_train_years, num_train_envs)
        train_envs = []
        train_env_names = []
        for chunk in chunks:
            idx = train_indices[np.isin(train_years, chunk)]
            train_envs.append(FMoWImageDataset(dataset, idx, transform))
            train_env_names.append(f"yr{int(chunk[0])+2002}-{int(chunk[-1])+2002}")

    # Test environments: one per OOD year
    test_indices = np.where(test_mask)[0]
    test_years_arr = years[test_indices]
    unique_test_years = sorted(set(test_years_arr))

    test_envs = []
    test_env_names = []
    for y in unique_test_years:
        idx = test_indices[test_years_arr == y]
        test_envs.append(FMoWImageDataset(dataset, idx, transform))
        test_env_names.append(f"yr{int(y)+2002}")

    # Apply remaining poison after partitioning
    # 'label' modifies _y_array; 'temporal_gap' and 'year_tint' hook into get_input.
    if poison_mode is not None and poison_mode != 'group':
        dataset.poison(
            fraction=poison_fraction,
            mode=poison_mode,
            seed=poison_seed,
            gap_year=gap_year,
            noise_std=noise_std,
            tint_strength=tint_strength,
        )

    env_names = {"train": train_env_names, "test": test_env_names}
    return train_envs, test_envs, env_names


def get_fmow_geo_datasets(data_dir, split_time=11, num_train_envs=None, **kwargs):
    """
    FMoW with GEOGRAPHIC environments (regions as domains).

    Same data as temporal, but environments = the 5 WILDS regions
    (Africa, Americas, Asia, Europe, Oceania). Test split is still temporal.

    This is the key comparison: same dataset, same model, but different
    environment definitions does EQRM help more with geographic envs?

    Args:
        data_dir: root directory for WILDS data
        split_time: year index for train/test split (default 11 = year 2013)
        num_train_envs: ignored (always 5 regions)

    Returns:
        train_envs, test_envs, env_names
    """
    dataset = _load_fmow_wilds(data_dir)
    transform = _fmow_default_transform()

    # Get metadata
    split_array = dataset.split_array
    years = dataset.metadata_array[:, 1].numpy()
    regions = dataset.metadata_array[:, 0].numpy()  # region index 0-4

    # Training: pre-split_time data, grouped by REGION
    train_mask = ((split_array == dataset.split_dict['train']) |
                  (split_array == dataset.split_dict['id_val']))
    train_indices = np.where(train_mask)[0]
    train_years = years[train_indices]
    train_regions = regions[train_indices]

    # Filter to training years
    time_mask = train_years < split_time
    train_indices_filtered = train_indices[time_mask]
    train_regions_filtered = train_regions[time_mask]

    train_envs = []
    train_env_names = []
    for region_idx, region_name in enumerate(FMOW_REGIONS):
        idx = train_indices_filtered[train_regions_filtered == region_idx]
        if len(idx) > 0:
            train_envs.append(FMoWImageDataset(dataset, idx, transform))
            train_env_names.append(region_name)

    # Test: OOD years, one env per region
    test_mask = ((split_array == dataset.split_dict['test']) |
                 (split_array == dataset.split_dict['val']))
    test_indices = np.where(test_mask)[0]
    test_regions = regions[test_indices]

    test_envs = []
    test_env_names = []
    for region_idx, region_name in enumerate(FMOW_REGIONS):
        idx = test_indices[test_regions == region_idx]
        if len(idx) > 0:
            test_envs.append(FMoWImageDataset(dataset, idx, transform))
            test_env_names.append(region_name)

    env_names = {"train": train_env_names, "test": test_env_names}
    return train_envs, test_envs, env_names


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "yearbook": {
        "loader": get_yearbook_datasets,
        "default_split_time": 1970,
        "num_classes": 2,
        "input_type": "image",
        "network": "yearbook_cnn",
        "description": "Yearbook portrait gender classification (1930-2013)",
    },
    "fmow_temporal": {
        "loader": get_fmow_temporal_datasets,
        "default_split_time": 11,
        "num_classes": 62,
        "input_type": "image",
        "network": "densenet121",
        "description": "FMoW satellite classification — temporal environments (years)",
        "poison_modes": ["label", "group", "temporal_gap", "year_tint"],
    },
    "fmow_geo": {
        "loader": get_fmow_geo_datasets,
        "default_split_time": 11,
        "num_classes": 62,
        "input_type": "image",
        "network": "densenet121",
        "description": "FMoW satellite classification — geographic environments (regions)",
    },
}


def get_datasets(dataset_name, data_dir, split_time=None, num_train_envs=None, **kwargs):
    """
    Unified interface for loading Wild-Time datasets.

    Args:
        dataset_name: one of the keys in DATASET_REGISTRY
        data_dir: root directory for data
        split_time: train/test split timestamp (uses default if None)
        num_train_envs: number of training environments (None = one per timestamp)

    Returns:
        train_envs, test_envs, env_names
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    info = DATASET_REGISTRY[dataset_name]
    if split_time is None:
        split_time = info["default_split_time"]

    return info["loader"](data_dir, split_time=split_time, num_train_envs=num_train_envs, **kwargs)
