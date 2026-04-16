"""
TableShift dataset wrappers for EQRM experiments.

Loads TableShift datasets and splits training data by domain labels into
environments compatible with the QRM repo's `update(minibatches)` interface.

Only the 10 tasks with multiple training subdomains (supporting domain
generalization) are included.

Requires: pip install tableshift
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset


# The 10 TableShift tasks that support domain generalization
# (have multiple training subdomains, |D_train| >= 2)
DG_TASKS = {
    "diabetes_readmission": {
        "description": "Diabetes patient readmission prediction",
        "shift": "Admission source",
        "num_features": None,  # determined at load time
    },
    "anes": {
        "description": "Voted in U.S. presidential election",
        "shift": "Region",
    },
    "assistments": {
        "description": "Student answer correctness prediction",
        "shift": "School ID",
    },
    "college_scorecard": {
        "description": "Low income student share prediction",
        "shift": "Institution type",
    },
    "brfss_diabetes": {
        "description": "Diabetes diagnosis prediction",
        "shift": "Race",
    },
    "brfss_blood_pressure": {
        "description": "High blood pressure diagnosis",
        "shift": "Race",
    },
    "acsfoodstamps": {
        "description": "Food stamp recipiency prediction",
        "shift": "Geographic region",
    },
    "heloc": {
        "description": "Home equity line of credit risk",
        "shift": "Est. third-party risk level",
    },
    "hospital_readmission": {
        "description": "Hospital readmission prediction",
        "shift": "Insurance type",
    },
    "icu_mortality": {
        "description": "ICU patient mortality prediction",
        "shift": "Insurance type",
    },
}

# A small subset for quick experiments
QUICK_TASKS = ["diabetes_readmission", "anes", "acsfoodstamps"]


def get_tableshift_datasets(task_name, cache_dir=None):
    """
    Load a TableShift task and split training data into per-domain environments.

    Args:
        task_name: one of the keys in DG_TASKS
        cache_dir: optional directory for cached data

    Returns:
        train_envs: list of TensorDatasets, one per training domain
        test_envs: list with a single TensorDataset (the OOD test set)
        env_names: dict with 'train' and 'test' lists of names
        input_dim: number of features
    """
    try:
        from tableshift import get_dataset
    except ImportError:
        raise ImportError(
            "TableShift is required. Install with:\n"
            "  pip install tableshift\n"
            "Or use Docker: docker pull ghcr.io/jpgard/tableshift:latest"
        )

    if task_name not in DG_TASKS:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available DG tasks: {list(DG_TASKS.keys())}"
        )

    # Load dataset
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    dset = get_dataset(task_name, **kwargs)

    # Get training data with domain labels
    # TableShift returns (X, y, domain, _) tuples
    X_train, y_train, d_train, _ = dset.get_pandas("train")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.LongTensor(y_train.values)
    # Domain may have multiple columns — use the first one as environment
    if hasattr(d_train, 'iloc'):
        d_train_np = d_train.iloc[:, 0].values
    else:
        d_train_np = d_train.ravel()

    input_dim = X_train_t.shape[1]

    # Split into per-domain environments
    unique_domains = sorted(set(d_train_np))
    train_envs = []
    train_env_names = []
    for domain in unique_domains:
        mask = d_train_np == domain
        if mask.sum() < 10:  # skip tiny domains
            continue
        x_d = X_train_t[mask]
        y_d = y_train_t[mask]
        train_envs.append(TensorDataset(x_d, y_d))
        train_env_names.append(str(domain))

    # OOD test set
    X_test, y_test, _, _ = dset.get_pandas("ood_test")
    X_test_t = torch.FloatTensor(X_test.values)
    y_test_t = torch.LongTensor(y_test.values)
    test_envs = [TensorDataset(X_test_t, y_test_t)]
    test_env_names = ["OOD_test"]

    # Also get ID test for comparison
    try:
        X_val, y_val, _, _ = dset.get_pandas("id_test")
        X_val_t = torch.FloatTensor(X_val.values)
        y_val_t = torch.LongTensor(y_val.values)
        test_envs.append(TensorDataset(X_val_t, y_val_t))
        test_env_names.append("ID_val")
    except Exception:
        pass  # some tasks may not have a separate validation split

    env_names = {"train": train_env_names, "test": test_env_names}

    print(f"Loaded {task_name}: {input_dim} features, "
          f"{len(train_envs)} training domains, "
          f"{sum(len(e) for e in train_envs)} train samples")
    for name, env in zip(train_env_names, train_envs):
        print(f"  Domain '{name}': {len(env)} samples")

    return train_envs, test_envs, env_names, input_dim
