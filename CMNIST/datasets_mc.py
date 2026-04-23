"""
Multi-Class (MC) ColoredMNIST dataset.

Overview
--------
This is a 10-class, 10-color analogue of the binary ColoredMNIST used in
Eastwood et al. (2022).  Instead of a binary digit-color assignment, every digit
class (0-9) is associated with one of 10 distinct RGB colors.

Spurious correlation structure
-------------------------------
Each *environment* is defined by a flip probability p_e  (same convention as
the binary ColoredMNIST — p_e is the probability the color is WRONG):
  - With probability 1-p_e : color c = y  (color matches the true digit label)
  - With probability p_e   : color c ~ Uniform({0,...,9} \\ {y})  (random wrong color)

Key statistics
  - Shape-only oracle accuracy: ~99% (standard MNIST digit recognition)
  - Color-only accuracy:        1 - p_e  (predicts the color class)
  - Chance baseline:            10%  (uniform 10-class)
  - "Uninformative" color:      p_e = 0.9  (at this value every color is equally
                                            likely: P(c=k) = 1/10 for all k)

Training vs test
----------------
  Train: low  p_e (e.g. 0.1-0.2) — color correct ~80-90%, strong shortcut
  Test:  high p_e (e.g. 0.9)     — color uniformly random, no net information
  Oracle: p_e = 0.9 for all train envs — color uninformative at training time,
          model is forced to rely on digit shape.

Colors
------
Ten maximally-distinct colors are produced by spacing hues evenly around the
HSV wheel (saturation=1, value=1) and converting to RGB.  Color k ∈ {0,...,9}
is associated with digit class k.
"""

import colorsys

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

# ---------------------------------------------------------------------------
# Color table
# ---------------------------------------------------------------------------

def make_colors(n_classes: int = 10) -> torch.Tensor:
    """
    Create a (n_classes, 3) float32 RGB tensor with evenly-spaced HSV hues.

    Digit k is mapped to hue = k/n_classes on the HSV wheel
    (saturation=1, value=1), giving maximally-distinguishable pure colors.

    E.g. with n_classes=10:
        0 → red       (H=0°)
        1 → orange    (H=36°)
        2 → yellow    (H=72°)
        3 → chartreuse(H=108°)
        4 → green     (H=144°)
        5 → spring    (H=180°)
        6 → cyan      (H=216°)
        7 → azure     (H=252°)
        8 → violet    (H=288°)
        9 → magenta   (H=324°)
    """
    rgb_list = [colorsys.hsv_to_rgb(i / n_classes, 1.0, 1.0)
                for i in range(n_classes)]
    return torch.tensor(rgb_list, dtype=torch.float32)   # (n_classes, 3)


# Pre-compute once at module load so all callers share the same tensor.
COLORS_RGB = make_colors(10)   # (10, 3)


# ---------------------------------------------------------------------------
# Single-environment dataset builder
# ---------------------------------------------------------------------------

def color_dataset_mc(images, labels, env_p: float,
                      n_classes: int = 10,
                      label_noise_rate: float = 0.25,
                      subsample: bool = True,
                      cuda: bool = True) -> TensorDataset:
    """
    Build a TensorDataset for one MC-CMNIST environment.

    Parameters
    ----------
    images           : (N, 28, 28) uint8 Tensor  —  raw MNIST images
    labels           : (N,)        int   Tensor  —  digit labels 0-9
    env_p            : float  —  probability that color is WRONG (flip probability,
                                 same convention as binary CMNIST).
                                 P(color = label) = 1 - env_p.
                                 env_p=0.1 → color correct 90% of the time (spurious shortcut).
                                 env_p=0.9 → each of the 10 colors equally likely (uninformative).
    n_classes        : int    —  number of digit/color classes (default 10)
    label_noise_rate : float  —  probability of flipping the label to a uniformly random
                                 OTHER class.  Mirrors the binary CMNIST label-noise
                                 (XOR-flip at rate 0.25).  With label_noise_rate=0.25 the
                                 shape-only oracle accuracy is capped at ~75%, making the
                                 color shortcut (90% at env_p=0.1) genuinely competitive.
                                 Color is assigned based on the *noisy* label, identical to
                                 the binary case.
    subsample        : bool   —  if True, downsample 28x28 -> 14x14 (2x sub-sampling)
    cuda             : bool   —  if True, move output tensors to GPU

    Returns
    -------
    TensorDataset with
        x : (N, 3, H, W) float32 in [0, 1]   (H=W=14 if subsample else 28)
        y : (N,)          int64               (noisy digit class, 10-class target)
    """
    if subsample:
        images = images.reshape(-1, 28, 28)[:, ::2, ::2]   # (N, 14, 14)

    N = len(labels)
    labels_long = labels.long()                             # (N,)

    # ------------------------------------------------------------------
    # Label noise  (multiclass analogue of binary XOR-flip)
    # ------------------------------------------------------------------
    if label_noise_rate > 0:
        noise_mask  = torch.rand(N) < label_noise_rate          # (N,) bool
        noise_idx   = torch.randint(0, n_classes - 1, (N,))     # uniform in [0, n_classes-2]
        # Shift past the true label so we always flip to a *different* class
        noise_class = noise_idx + (noise_idx >= labels_long).long()
        labels_long = torch.where(noise_mask, noise_class, labels_long)

    # ------------------------------------------------------------------
    # Color assignments  (based on the noisy label, matching binary CMNIST)
    # ------------------------------------------------------------------
    colors_rgb = make_colors(n_classes)                     # (n_classes, 3)

    # Decide which samples get their true-label color vs a random wrong color.
    # env_p is a FLIP probability: True = correct color, False = wrong color.
    match_mask = torch.rand(N) >= env_p                     # (N,) bool

    # Sample a wrong class index, uniform over {0, ..., n_classes-2}
    wrong_idx = torch.randint(0, n_classes - 1, (N,))       # (N,) in [0, n_classes-2]
    # Shift past the true label so we never accidentally assign the correct color
    wrong_class = wrong_idx + (wrong_idx >= labels_long).long()   # (N,) in [0, n_classes-1]\{y}

    color_assignments = torch.where(match_mask, labels_long, wrong_class)  # (N,)

    # ------------------------------------------------------------------
    # Colorize: pixel-wise product of (H, W) grayscale and (3,) color
    # ------------------------------------------------------------------
    images_float = images.float().div_(255.0)               # (N, H, W) in [0, 1]
    color_vecs   = colors_rgb[color_assignments]            # (N, 3)
    x = color_vecs[:, :, None, None] * images_float[:, None, :, :]  # (N, 3, H, W)

    y = labels_long                                         # (N,) long

    if cuda:
        x, y = x.cuda(), y.cuda()

    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Multi-environment loader
# ---------------------------------------------------------------------------

def get_mc_datasets(root,
                     train_envs=(0.1, 0.2),
                     test_envs=(0.9,),
                     label_noise_rate: float = 0.25,
                     dataset_transform=color_dataset_mc,
                     subsample: bool = True,
                     cuda: bool = True,
                     use_test_set: bool = False):
    """
    Build MC-CMNIST TensorDatasets for all training and test environments.

    This mirrors ``get_cmnist_datasets`` in datasets.py exactly, with the only
    difference being the colorization function (10-color RGB vs 2-channel RG).

    Parameters
    ----------
    root             : str   — path to MNIST data root (downloaded if needed)
    train_envs       : tuple of float  — flip probabilities for training envs
    test_envs        : tuple of float  — flip probabilities for test envs
    label_noise_rate : float — probability of flipping each label to a random
                               other class (default 0.25, same as binary CMNIST)
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
        # Each test env uses the full test split (different colorization)
        datasets.append(dataset_transform(test_images, test_labels, p,
                                          label_noise_rate=label_noise_rate,
                                          subsample=subsample, cuda=cuda))

    return datasets
