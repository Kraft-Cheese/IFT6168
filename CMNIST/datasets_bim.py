"""
Binary Interleaved M2M (BIM) ColoredMNIST dataset.

Overview
--------
A hybrid of binary CMNIST and the 10-color MC colorization scheme.
The prediction target is binary (digit shape class), but the spurious
shortcut uses all 10 HSV colors split into two *interleaved* pools,
creating a non-linear XOR-style decision boundary in RGB space.

Binary target
-------------
  Class 0 : Digits {0, 1, 2, 3, 4}
  Class 1 : Digits {5, 6, 7, 8, 9}

Label noise (The Trap)
----------------------
A 25% symmetric label flip caps the shape-only oracle at ~75%, making the
color shortcut (90% accurate at env_p=0.1) genuinely competitive.
Color is assigned based on the *noisy* label, identical to binary CMNIST.

Interleaved color pools
-----------------------
The 10 HSV colors are divided by index parity into two pools:

  Pool A (even indices) : colors {0, 2, 4, 6, 8}  ->  correlated with noisy label 0
  Pool B (odd  indices) : colors {1, 3, 5, 7, 9}  ->  correlated with noisy label 1

For a sample with noisy label ỹ and environment flip probability p_e:
  - With prob (1 - p_e) : sample a color uniformly from the correct pool
  - With prob  p_e      : sample a color uniformly from the wrong pool

Why is the decision boundary non-linear?
-----------------------------------------
Pool A and Pool B alternate around the HSV wheel. No linear classifier in
RGB space can separate them: the six Pool-A colors and five Pool-B colors
are interleaved in hue, so the correct color boundary requires a complex,
non-linear mapping. This is strictly harder than standard binary CMNIST,
where the shortcut is a trivial linear rule (R channel vs G channel).

Uninformative threshold
-----------------------
At env_p = 0.5, both pools are equally likely regardless of label.
The oracle trains at this threshold: 'oracle' -> train_envs = (0.5, 0.5).

Label noise is part of the data generating process
--------------------------------------------------
The same label_noise_rate (default 0.25) is applied to both training and
test environments, matching the binary CMNIST convention.  The noisy label
ỹ is the actual prediction target throughout.
"""

import colorsys

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST


# ---------------------------------------------------------------------------
# Color table  (same 10-color HSV wheel as datasets_mc.py)
# ---------------------------------------------------------------------------

def make_colors(n_classes: int = 10) -> torch.Tensor:
    """Return (n_classes, 3) float32 RGB tensor with evenly-spaced HSV hues."""
    rgb_list = [colorsys.hsv_to_rgb(i / n_classes, 1.0, 1.0) for i in range(n_classes)]
    return torch.tensor(rgb_list, dtype=torch.float32)


COLORS_RGB = make_colors(10)   # (10, 3)  — module-level constant, CPU

# Interleaved pool indices (even -> Pool A, odd -> Pool B)
POOL_A = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)   # 5 colors -> noisy label 0
POOL_B = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)   # 5 colors -> noisy label 1


# ---------------------------------------------------------------------------
# Single-environment dataset builder
# ---------------------------------------------------------------------------

def color_dataset_bim(images, labels, env_p: float,
                       label_noise_rate: float = 0.25,
                       subsample: bool = True,
                       cuda: bool = True) -> TensorDataset:
    """
    Build a TensorDataset for one BIM (Binary Interleaved M2M) environment.

    All operations are fully vectorized — no Python loops over the batch.

    Parameters
    ----------
    images           : (N, 28, 28) uint8 Tensor  —  raw MNIST images
    labels           : (N,)        int   Tensor  —  digit labels 0-9
    env_p            : float  —  flip probability (same convention as binary CMNIST).
                                 P(color from correct pool) = 1 - env_p.
                                 env_p=0.1 -> 90% chance correct pool (strong shortcut).
                                 env_p=0.5 -> random pool (uninformative color).
                                 env_p=0.9 -> 90% chance wrong pool (anti-correlation).
    label_noise_rate : float  —  probability of flipping the binary label to the
                                 other class.  Set to 0.0 for clean-label evaluation.
    subsample        : bool   —  if True, downsample 28x28 -> 14x14 (2x)
    cuda             : bool   —  if True, move output tensors to GPU

    Returns
    -------
    TensorDataset with
        x : (N, 3, H, W) float32 in [0, 1]   (H=W=14 if subsample else 28)
        y : (N,)          int64   —  (noisy) binary label in {0, 1}
    """
    if subsample:
        images = images.reshape(-1, 28, 28)[:, ::2, ::2]   # (N, 14, 14)

    N = len(labels)

    # ------------------------------------------------------------------
    # Step 1: Binarize
    # Class 0 = digits {0,1,2,3,4},  Class 1 = digits {5,6,7,8,9}
    # ------------------------------------------------------------------
    binary_labels = (labels >= 5).long()   # (N,)

    # ------------------------------------------------------------------
    # Step 2: Label noise  (symmetric flip, same as binary CMNIST)
    # ------------------------------------------------------------------
    if label_noise_rate > 0.0:
        noise_mask   = torch.rand(N) < label_noise_rate   # True -> flip
        noisy_labels = torch.where(noise_mask, 1 - binary_labels, binary_labels)
    else:
        noisy_labels = binary_labels                       # clean labels

    # ------------------------------------------------------------------
    # Step 3: Color pool assignment  (fully vectorized)
    #
    # correct pool for noisy_label=0 -> Pool A (evens)
    # correct pool for noisy_label=1 -> Pool B (odds)
    #
    # flip_mask: True with prob env_p -> assign the *wrong* pool
    # use_pool_B: XOR of (correct pool is B) and (flip occurred)
    # ------------------------------------------------------------------
    flip_mask      = torch.rand(N) < env_p                          # (N,) bool
    correct_is_B   = (noisy_labels == 1)                            # (N,) bool
    use_pool_B     = correct_is_B ^ flip_mask                       # XOR  (N,) bool

    # Uniform sample within the chosen pool (each pool has 5 colors)
    pool_idx       = torch.randint(0, 5, (N,))                      # (N,) in [0, 4]
    color_from_A   = POOL_A[pool_idx]                               # (N,) indices from Pool A
    color_from_B   = POOL_B[pool_idx]                               # (N,) indices from Pool B
    color_assignments = torch.where(use_pool_B, color_from_B, color_from_A)  # (N,)

    # ------------------------------------------------------------------
    # Step 4: Colorize
    # pixel-wise product of (H, W) grayscale and (3,) RGB color vector
    # ------------------------------------------------------------------
    images_float = images.float().div_(255.0)                       # (N, H, W) in [0, 1]
    color_vecs   = COLORS_RGB[color_assignments]                    # (N, 3)
    x = color_vecs[:, :, None, None] * images_float[:, None, :, :] # (N, 3, H, W)

    y = noisy_labels                                                 # (N,) long

    if cuda:
        x, y = x.cuda(), y.cuda()

    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Multi-environment loader
# ---------------------------------------------------------------------------

def get_bim_datasets(root,
                      train_envs=(0.1, 0.2),
                      test_envs=(0.9,),
                      label_noise_rate: float = 0.25,
                      dataset_transform=color_dataset_bim,
                      subsample: bool = True,
                      cuda: bool = True,
                      use_test_set: bool = False):
    """
    Build BIM TensorDatasets for all training and test environments.

    Mirrors ``get_cmnist_datasets`` in structure.  The same label_noise_rate
    is applied to both training and test environments — label noise is part
    of the data generating process, not a training-only artefact.

    Parameters
    ----------
    root             : str   — path to MNIST data root (downloaded if needed)
    train_envs       : tuple of float  — flip probabilities for training envs
    test_envs        : tuple of float  — flip probabilities for test envs
    label_noise_rate : float — label flip rate for all envs (default 0.25,
                               same as binary CMNIST)
    subsample        : bool  — downsample images 2x
    cuda             : bool  — move tensors to GPU
    use_test_set     : bool  — if True, use MNIST's own test set for test envs
                               (otherwise carve out last 10k of training set)

    Returns
    -------
    list of TensorDataset, length len(train_envs) + len(test_envs)
    """
    if root is None:
        raise ValueError("Data directory not specified!")

    orig_data_tr  = MNIST(root, train=True,  download=True)
    perm_inds_tr  = torch.randperm(len(orig_data_tr.data))

    if use_test_set:
        orig_data_tst = MNIST(root, train=False, download=True)
        perm_inds_tst = torch.randperm(len(orig_data_tst.data))
        train_images  = orig_data_tr.data[perm_inds_tr]
        train_labels  = orig_data_tr.targets[perm_inds_tr]
        test_images   = orig_data_tst.data[perm_inds_tst]
        test_labels   = orig_data_tst.targets[perm_inds_tst]
    else:
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
