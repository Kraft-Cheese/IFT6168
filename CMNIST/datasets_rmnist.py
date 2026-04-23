"""
Rotated MNIST (RMNIST) — Binary spurious-correlation benchmark.

Overview
--------
Transforms standard MNIST into a binary classification task where the rotation
angle θ acts as the spurious correlate.  Unlike CMNIST, the spurious feature
is geometric (continuous rotation), not colorimetric.  This tests whether
invariant-learning algorithms can disentangle digit identity from pose.

Binary super-classes
--------------------
  Group 0: Digits {0, 1, 2, 3, 4}   (lower half of MNIST alphabet)
  Group 1: Digits {5, 6, 7, 8, 9}   (upper half of MNIST alphabet)

No label noise
--------------
Unlike CMNIST, RMNIST uses clean binary labels.  The shape-only oracle
approaches ~100% accuracy; rotation is the sole spurious signal.

Environment parameterisation — single delta parameter
------------------------------------------------------
Each environment is described by a single float delta (degrees):
    mu_0 = 45 - delta/2   (mean rotation for Group 0)
    mu_1 = 45 + delta/2   (mean rotation for Group 1)
    sigma = 5° fixed noise

  delta > 0  → Group 1 rotated more than Group 0 (spurious correlation)
  delta = 0  → both groups share mu=45° (rotation uninformative, oracle)
  delta < 0  → inverted (anti-correlated, OOD test env)

Default training environments
------------------------------
  Env 1: delta=60  → mu_0=15°, mu_1=75°  (strong shortcut,  60° separation)
  Env 2: delta=30  → mu_0=30°, mu_1=60°  (moderate shortcut, 30° separation)
  OOD:   delta=-60 → mu_0=75°, mu_1=15°  (inverted, anti-correlated)
  Oracle: delta=0  → mu_0=45°, mu_1=45°  (uninformative)

Rotation is applied as a vectorized affine transform (affine_grid + grid_sample)
so no Python loops are used and the full batch is rotated in one call.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: MNIST class indices that belong to Group 1 (upper digits 5-9).
GROUP1_DIGITS = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)

GROUP_NAMES = {
    0: "Lower digits (0,1,2,3,4)",
    1: "Upper digits (5,6,7,8,9)",
}

#: Fixed rotation noise (std dev in degrees).  Matches user spec.
SIGMA_DEG = 5.0


# ---------------------------------------------------------------------------
# Vectorized rotation
# ---------------------------------------------------------------------------

def rotate_images_batch(images_float: torch.Tensor,
                        angles_deg: torch.Tensor) -> torch.Tensor:
    """
    Rotate a batch of images by per-sample angles (fully vectorized, CPU).

    Uses pytorch's affine_grid + grid_sample (bilinear, zero-padding outside
    bounds).  Rotation is about the image centre in normalised [-1, 1] coords.

    Parameters
    ----------
    images_float : (N, C, H, W) float32 in [0, 1]
    angles_deg   : (N,)         float32 — rotation angles in degrees (CCW)

    Returns
    -------
    (N, C, H, W) float32 — rotated images, zero-padded outside the canvas
    """
    angles_rad = angles_deg.float() * (torch.pi / 180.0)   # (N,)
    cos_a = torch.cos(angles_rad)   # (N,)
    sin_a = torch.sin(angles_rad)   # (N,)

    N = images_float.size(0)
    # Rotation matrix (N, 2, 3) in pytorch affine_grid convention
    theta = torch.zeros(N, 2, 3, dtype=torch.float32)
    theta[:, 0, 0] =  cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] =  sin_a
    theta[:, 1, 1] =  cos_a
    # theta[:, :, 2] = 0  (no translation — rotation around centre)

    grid    = F.affine_grid(theta, images_float.size(), align_corners=False)
    rotated = F.grid_sample(images_float, grid,
                            mode='bilinear', padding_mode='zeros',
                            align_corners=False)
    return rotated


# ---------------------------------------------------------------------------
# Single-environment dataset builder
# ---------------------------------------------------------------------------

def rotate_dataset_rmnist(images, labels, env_delta: float,
                           sigma_deg: float = SIGMA_DEG,
                           subsample: bool = True,
                           cuda: bool = True) -> TensorDataset:
    """
    Build a TensorDataset for one RMNIST environment.

    All operations are fully vectorized — no Python loops over the batch.

    Parameters
    ----------
    images    : (N, 28, 28) uint8 Tensor  — raw MNIST images
    labels    : (N,) int Tensor           — original class labels 0-9
    env_delta : float — angle separation (degrees): mu_1 - mu_0.
                        Positive: Group 1 rotated more than Group 0.
                        Negative: inverted (OOD test).
                        Zero:     uninformative (oracle).
    sigma_deg : float — std dev of rotation Gaussian (default 5°)
    subsample : bool  — downsample 28x28 → 14x14 before rotating (default True)
    cuda      : bool  — move output tensors to GPU

    Returns
    -------
    TensorDataset with
        x : (N, 1, H, W) float32 in [0, 1]   (H=W=14 if subsample else 28)
        y : (N,) int64 — clean binary label in {0, 1}
    """
    # ------------------------------------------------------------------
    # Step 1: Binarize labels (vectorized membership test)
    # ------------------------------------------------------------------
    binary_labels = torch.isin(labels, GROUP1_DIGITS).long()   # (N,)

    # ------------------------------------------------------------------
    # Step 2: Sample rotation angles from N(mu_class, sigma^2)
    #   mu_0 = 45 - delta/2,  mu_1 = 45 + delta/2
    # ------------------------------------------------------------------
    mu_0 = 45.0 - env_delta / 2.0
    mu_1 = 45.0 + env_delta / 2.0

    mean_angles = torch.where(
        binary_labels == 0,
        torch.full_like(binary_labels, mu_0, dtype=torch.float32),
        torch.full_like(binary_labels, mu_1, dtype=torch.float32),
    )                                                           # (N,) float
    angles_deg = torch.normal(mean=mean_angles, std=sigma_deg) # (N,) float

    # ------------------------------------------------------------------
    # Step 3: Prepare images as float32 in [0, 1]
    # Subsample before rotating to keep affine_grid small.
    # ------------------------------------------------------------------
    images_float = images.float().div_(255.0)   # (N, H, W) — in-place on copy
    if subsample:
        images_float = images_float[:, ::2, ::2]   # (N, 14, 14)
    images_4d = images_float.unsqueeze(1)           # (N, 1, H, W)

    # ------------------------------------------------------------------
    # Step 4: Rotate batch (vectorized affine transform, CPU)
    # ------------------------------------------------------------------
    x = rotate_images_batch(images_4d, angles_deg)   # (N, 1, H, W)
    y = binary_labels                                  # (N,) long

    if cuda:
        x, y = x.cuda(), y.cuda()

    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Multi-environment loader
# ---------------------------------------------------------------------------

def get_rmnist_datasets(root,
                        train_envs=(60.0, 30.0),
                        test_envs=(-60.0,),
                        sigma_deg: float = SIGMA_DEG,
                        dataset_transform=rotate_dataset_rmnist,
                        subsample: bool = True,
                        cuda: bool = True,
                        use_test_set: bool = False):
    """
    Build RMNIST TensorDatasets for all training and test environments.

    Mirrors ``get_cmnist_datasets`` in datasets.py exactly, with delta values
    replacing flip probabilities.  No label noise is applied (clean labels).

    Parameters
    ----------
    root       : str   — path to MNIST data root (downloaded if needed)
    train_envs : tuple of float — delta values for training envs (degrees)
    test_envs  : tuple of float — delta values for test envs (degrees)
    sigma_deg  : float — rotation noise std dev in degrees (default 5°)
    subsample  : bool  — downsample images 2x (28→14)
    cuda       : bool  — move tensors to GPU
    use_test_set : bool — if True, use MNIST's own test split for test envs
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
        # Carve out last 10k of training set as held-out validation split
        train_images = orig_data_tr.data[perm_inds_tr][:50000]
        train_labels = orig_data_tr.targets[perm_inds_tr][:50000]
        test_images  = orig_data_tr.data[perm_inds_tr][50000:]
        test_labels  = orig_data_tr.targets[perm_inds_tr][50000:]

    datasets = []

    for i, delta in enumerate(train_envs):
        # Interleave: env i gets every len(train_envs)-th sample
        imgs = train_images[i::len(train_envs)]
        lbls = train_labels[i::len(train_envs)]
        datasets.append(dataset_transform(imgs, lbls, delta,
                                          sigma_deg=sigma_deg,
                                          subsample=subsample, cuda=cuda))

    for delta in test_envs:
        datasets.append(dataset_transform(test_images, test_labels, delta,
                                          sigma_deg=sigma_deg,
                                          subsample=subsample, cuda=cuda))

    return datasets
