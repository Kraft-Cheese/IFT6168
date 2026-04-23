"""
Colored Fashion-MNIST (CFashionMNIST) — Binary spurious-correlation benchmark.

Overview
--------
Transforms standard Fashion-MNIST into a binary colored spurious-correlation
benchmark that mirrors binary CMNIST exactly, but with a harder visual base
dataset (clothing/accessories vs. footwear) and an explicit 3-channel RGB
colorization designed for easy extension to a 10-class MC variant.

Binary super-classes
--------------------
  Group 0 (Upper/Full Body):        T-shirt/top (0), Pullover (2), Dress (3),
                                     Coat (4), Shirt (6)                    [5 classes]
  Group 1 (Lower Body/Accessories): Trouser (1), Sandal (5), Sneaker (7),
                                     Bag (8), Ankle boot (9)                [5 classes]

This 5v5 split creates a logically coherent and geometrically meaningful
division: Group 0 items cover the upper or full body (compact silhouettes
with sleeves/collars); Group 1 items cover the lower body and extremities
(long rectangular trousers, small footwear shapes, boxy bags).  The balanced
split also removes class-imbalance as a confound.

Label noise (The Trap)
----------------------
A 25% symmetric flip is applied to the binary target y → noisy target ỹ.
This caps the shape-only oracle at exactly ~75%, while the color shortcut
achieves ~90% accuracy at env_p=0.1.  Color is assigned based on ỹ (not y),
matching the binary CMNIST convention.

Spurious color feature
----------------------
Two maximally separable RGB colors are assigned one-to-one to the noisy label:
  Color A: Red   (1, 0, 0) — correlated with ỹ=0 (Apparel)
  Color B: Green (0, 1, 0) — correlated with ỹ=1 (Accessories/Footwear)

For environment with flip probability p_e (same convention as binary CMNIST):
  - With prob (1 - p_e): assign the correct color  (spurious shortcut)
  - With prob  p_e:      assign the wrong color     (flip)

  p_e=0.1 → color correct 90% of the time (strong spurious signal for ERM)
  p_e=0.5 → each color equally likely, uninformative  (oracle training point)
  p_e=0.9 → color anti-correlated with label  (hard OOD test environment)

MC extensibility
----------------
The make_colors() function and 3-channel RGB output are designed so this
file can be extended to a 10-class variant (one color per Fashion-MNIST class)
by simply changing n_classes and the super-class mapping, without touching
the colorization or environment logic.

All operations are fully vectorized — no Python loops over the batch.
"""

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import FashionMNIST


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

#: Fashion-MNIST class indices that belong to Group 1 (Lower Body/Accessories).
#: Everything else (indices 0,2,3,4,6) belongs to Group 0 (Upper/Full Body).
GROUP1_CLASSES = torch.tensor([1, 5, 7, 8, 9], dtype=torch.long)

# Human-readable label for documentation / sanity checks
CLASS_NAMES = {
    0: "T-shirt/top",   1: "Trouser",  2: "Pullover",   3: "Dress",
    4: "Coat",          5: "Sandal",   6: "Shirt",       7: "Sneaker",
    8: "Bag",           9: "Ankle boot",
}
GROUP_NAMES = {
    0: "Upper/Full Body (0,2,3,4,6)",
    1: "Lower Body/Accessories (1,5,7,8,9)",
}


# ---------------------------------------------------------------------------
# Color table
# ---------------------------------------------------------------------------

def make_colors(n_classes: int = 2) -> torch.Tensor:
    """
    Return a (n_classes, 3) float32 RGB tensor.

    For binary (n_classes=2): maximally separable Red and Green.
    For MC extension (n_classes>2): evenly-spaced HSV hues, same as
    datasets_mc.py, so the two color tables are consistent.

    Parameters
    ----------
    n_classes : int — 2 for binary CFashionMNIST; 10 for a future MC variant.

    Returns
    -------
    Tensor of shape (n_classes, 3) in [0, 1] float32.
    """
    if n_classes == 2:
        return torch.tensor([
            [1.0, 0.0, 0.0],   # Color A: Red   -> class 0 (Upper/Full Body)
            [0.0, 1.0, 0.0],   # Color B: Green -> class 1 (Lower Body/Accessories)
        ], dtype=torch.float32)
    else:
        import colorsys
        rgb_list = [colorsys.hsv_to_rgb(i / n_classes, 1.0, 1.0)
                    for i in range(n_classes)]
        return torch.tensor(rgb_list, dtype=torch.float32)


COLORS_RGB = make_colors(2)   # (2, 3) — module-level constant, CPU


# ---------------------------------------------------------------------------
# Single-environment dataset builder
# ---------------------------------------------------------------------------

def color_dataset_cfmnist(images, labels, env_p: float,
                           label_noise_rate: float = 0.25,
                           subsample: bool = True,
                           cuda: bool = True) -> TensorDataset:
    """
    Build a TensorDataset for one CFashionMNIST environment.

    All operations are fully vectorized — no Python loops over the batch.

    Parameters
    ----------
    images           : (N, 28, 28) uint8 Tensor  —  raw Fashion-MNIST images
    labels           : (N,)        int   Tensor  —  original class labels 0-9
    env_p            : float  —  color flip probability (same convention as
                                 binary CMNIST; p_e = P(color is wrong)).
                                 env_p=0.1 -> color correct 90% (strong shortcut).
                                 env_p=0.5 -> color uninformative (oracle).
                                 env_p=0.9 -> color anti-correlated (OOD test).
    label_noise_rate : float  —  symmetric binary label flip rate (default 0.25).
                                 Caps shape-only oracle at ~75%.
    subsample        : bool   —  if True, downsample 28x28 -> 14x14 (2x)
    cuda             : bool   —  if True, move output tensors to GPU

    Returns
    -------
    TensorDataset with
        x : (N, 3, H, W) float32 in [0, 1]   (H=W=14 if subsample else 28)
        y : (N,)          int64   —  noisy binary label in {0, 1}
    """
    if subsample:
        images = images.reshape(-1, 28, 28)[:, ::2, ::2]   # (N, 14, 14)

    N = len(labels)

    # ------------------------------------------------------------------
    # Step 1: Binarize  (vectorized membership test, no Python loop)
    # Group 1 if class in {5, 7, 8, 9}, else Group 0
    # ------------------------------------------------------------------
    binary_labels = torch.isin(labels, GROUP1_CLASSES).long()   # (N,)

    # ------------------------------------------------------------------
    # Step 2: Label noise  (symmetric binary flip, same as binary CMNIST)
    # ------------------------------------------------------------------
    if label_noise_rate > 0.0:
        noise_mask   = torch.rand(N) < label_noise_rate
        noisy_labels = torch.where(noise_mask, 1 - binary_labels, binary_labels)
    else:
        noisy_labels = binary_labels

    # ------------------------------------------------------------------
    # Step 3: Color assignment
    # Color A (Red) -> ỹ=0,  Color B (Green) -> ỹ=1
    # With prob env_p: assign the wrong color (flip)
    # ------------------------------------------------------------------
    flip_mask         = torch.rand(N) < env_p                              # (N,) bool
    color_assignments = torch.where(flip_mask, 1 - noisy_labels, noisy_labels)  # (N,) in {0,1}

    # ------------------------------------------------------------------
    # Step 4: Colorize  (pixel-wise multiply grayscale by RGB vector)
    # ------------------------------------------------------------------
    images_float = images.float().div_(255.0)                              # (N, H, W) in [0,1]
    color_vecs   = COLORS_RGB[color_assignments]                           # (N, 3)
    x = color_vecs[:, :, None, None] * images_float[:, None, :, :]        # (N, 3, H, W)

    y = noisy_labels                                                        # (N,) long

    if cuda:
        x, y = x.cuda(), y.cuda()

    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Multi-environment loader
# ---------------------------------------------------------------------------

def get_cfmnist_datasets(root,
                          train_envs=(0.1, 0.2),
                          test_envs=(0.9,),
                          label_noise_rate: float = 0.25,
                          dataset_transform=color_dataset_cfmnist,
                          subsample: bool = True,
                          cuda: bool = True,
                          use_test_set: bool = False):
    """
    Build CFashionMNIST TensorDatasets for all training and test environments.

    Mirrors ``get_cmnist_datasets`` in datasets.py exactly.  The same
    label_noise_rate is applied to both training and test environments —
    label noise is part of the data generating process, not a training artefact.

    Parameters
    ----------
    root             : str   — path to Fashion-MNIST data root (downloaded if needed)
    train_envs       : tuple of float  — flip probs for training envs
    test_envs        : tuple of float  — flip probs for test envs
    label_noise_rate : float — symmetric label flip rate for all envs (default 0.25)
    subsample        : bool  — downsample images 2x (28->14)
    cuda             : bool  — move tensors to GPU
    use_test_set     : bool  — if True, use Fashion-MNIST's own test split for
                               test envs (otherwise carve out last 10k of train)

    Returns
    -------
    list of TensorDataset, length len(train_envs) + len(test_envs)
    """
    if root is None:
        raise ValueError("Data directory not specified!")

    orig_data_tr  = FashionMNIST(root, train=True,  download=True)
    perm_inds_tr  = torch.randperm(len(orig_data_tr.data))

    if use_test_set:
        orig_data_tst = FashionMNIST(root, train=False, download=True)
        perm_inds_tst = torch.randperm(len(orig_data_tst.data))
        train_images  = orig_data_tr.data[perm_inds_tr]
        train_labels  = orig_data_tr.targets[perm_inds_tr]
        test_images   = orig_data_tst.data[perm_inds_tst]
        test_labels   = orig_data_tst.targets[perm_inds_tst]
    else:
        # Use last 10k of training set as validation test split (same as CMNIST)
        train_images  = orig_data_tr.data[perm_inds_tr][:50000]
        train_labels  = orig_data_tr.targets[perm_inds_tr][:50000]
        test_images   = orig_data_tr.data[perm_inds_tr][50000:]
        test_labels   = orig_data_tr.targets[perm_inds_tr][50000:]

    datasets = []

    for i, p in enumerate(train_envs):
        # Interleave: env i gets every len(train_envs)-th sample
        imgs = train_images[i::len(train_envs)]
        lbls = train_labels[i::len(train_envs)]
        datasets.append(dataset_transform(imgs, lbls, p,
                                          label_noise_rate=label_noise_rate,
                                          subsample=subsample, cuda=cuda))

    for p in test_envs:
        datasets.append(dataset_transform(test_images, test_labels, p,
                                          label_noise_rate=label_noise_rate,
                                          subsample=subsample, cuda=cuda))

    return datasets
