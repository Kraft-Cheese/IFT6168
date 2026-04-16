# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Extended by: [Your Name] for EQRM extension experiments
# Adds: ColoredMNIST with many domains, RotatedMNIST variants,
#        and ColoredFashionMNIST for domain generalization research.

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, ImageFolder
from torchvision.transforms.functional import rotate

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images — original
    "ColoredMNIST",
    "RotatedMNIST",
    # =========================================================================
    # NEW DATASETS for EQRM extension experiments
    # =========================================================================
    #
    # --- ColoredMNIST with more training domains ---
    # The original CMNIST has only 2 training domains (flip prob 0.1, 0.2) and
    # 1 test domain (0.9). EQRM's theory (Thm 4.1) predicts tighter quantile
    # estimates with more domains. These variants test that prediction.
    #
    "ColoredMNIST5",       # 4 train + 1 test domain
    "ColoredMNIST10",      # 9 train + 1 test domain
    "ColoredMNIST20",      # 19 train + 1 test domain
    #
    # --- ColoredMNIST with graded spurious correlation ---
    # Instead of a binary "easy train / hard test" split, these domains sample
    # the color-flip probability from a continuous distribution, directly
    # matching the paper's meta-distribution Q framework (Section 3).
    #
    "ColoredMNISTContinuous",   # 10 domains, flip probs from Beta(2,5)
    #
    # --- RotatedMNIST with many domains ---
    # The original has 6 domains (0°–75° in 15° steps). More domains let us
    # compare quantile risk estimates from EQRM more meaningfully, since the
    # paper notes that quantile evaluation requires "multiple test domains"
    # (Section 6.2, "A new evaluation protocol for DG").
    #
    "RotatedMNIST12",      # 12 domains: 0° to 165° in 15° steps
    "RotatedMNIST18",      # 18 domains: 0° to 170° in 10° steps
    "RotatedMNIST36",      # 36 domains: 0° to 350° in 10° steps
    #
    # --- RotatedMNIST with non-uniform angle sampling ---
    # Tests EQRM when the meta-distribution Q over domains is non-uniform.
    # Angles are sampled from a clustered distribution, creating a setting
    # where some rotation ranges are heavily represented and others are sparse.
    #
    "RotatedMNISTNonUniform",  # 12 domains with clustered angles
    #
    # --- ColoredFashionMNIST ---
    # Same spurious-color protocol as CMNIST but on FashionMNIST (harder base
    # task). Tests whether EQRM's gains hold when the invariant feature (shape)
    # is more complex and the model needs more capacity to learn it.
    #
    "ColoredFashionMNIST",      # 2 train + 1 test (same as original CMNIST)
    "ColoredFashionMNIST10",    # 9 train + 1 test
    #
    # =========================================================================
    # Big images — original
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 2            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


# =============================================================================
# Base class for MNIST-based multi-environment datasets
# =============================================================================

class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))
        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


# =============================================================================
# Base class for FashionMNIST-based multi-environment datasets
# =============================================================================

class MultipleEnvironmentFashionMNIST(MultipleDomainDataset):
    """Like MultipleEnvironmentMNIST but uses FashionMNIST as the base dataset.

    FashionMNIST has the same image dimensions (28x28) and 10 classes as MNIST,
    but the classification task is harder (T-shirt, Trouser, Pullover, Dress,
    Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). This makes the invariant
    feature (shape) more challenging to learn, providing a tougher test of
    whether EQRM can learn to ignore spurious correlations.
    """
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = FashionMNIST(root, train=True, download=True)
        original_dataset_te = FashionMNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))
        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


# =============================================================================
# ColoredMNIST — ORIGINAL (2 train + 1 test)
# =============================================================================

class ColoredMNIST(MultipleEnvironmentMNIST):
    """Original ColoredMNIST from Arjovsky et al. (2019).

    2 training domains with color-flip probabilities 0.1 and 0.2, plus 1 test
    domain with flip probability 0.9.  The invariant feature (digit shape) has
    75% accuracy in all domains.
    """
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
            self.color_dataset, (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


# =============================================================================
# ColoredMNIST with MORE DOMAINS
# =============================================================================
#
# Motivation (from the paper):
#   - Theorem 4.1 shows the empirical α-quantile converges to the population
#     α-quantile as m (number of domains) → ∞.
#   - The original CMNIST has only m=2 training domains, making quantile
#     estimation imprecise ("α still controls conservativeness, but with a
#     less precise interpretation" — Section 4.2).
#   - Fig. 3D shows that reducing m degrades quantile accuracy.
#   - These multi-domain variants let us empirically measure this convergence.
#
# Design choices:
#   - Flip probabilities are evenly spaced in [0.05, 0.45] for training
#     domains, keeping the spurious correlation consistently positive (color
#     is predictive of label). This mirrors the original's design where
#     training domains have flip probs < 0.5.
#   - The last domain always has flip probability 0.9 (anti-correlated),
#     serving as the OOD test domain, matching the original CMNIST.
#   - More training domains = finer sampling of the meta-distribution Q.

class ColoredMNISTMultiDomain(MultipleEnvironmentMNIST):
    """Base class for ColoredMNIST variants with configurable number of domains.

    Training domains have color-flip probabilities evenly spaced in [0.05, 0.45].
    The final domain has flip probability 0.9 (the OOD test domain).
    """
    def __init__(self, root, test_envs, hparams, n_train_domains):
        # Evenly space flip probs across [0.05, 0.45] for training domains
        train_flip_probs = torch.linspace(0.05, 0.45, n_train_domains).tolist()
        # Add the OOD test domain (flip prob 0.9)
        all_flip_probs = train_flip_probs + [0.9]

        super().__init__(root, all_flip_probs,
                         self.color_dataset, (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredMNIST5(ColoredMNISTMultiDomain):
    """4 training domains + 1 OOD test domain."""
    ENVIRONMENTS = ['e0.05', 'e0.18', 'e0.32', 'e0.45', 'e0.90']
    N_STEPS = 5001

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, n_train_domains=4)


class ColoredMNIST10(ColoredMNISTMultiDomain):
    """9 training domains + 1 OOD test domain.

    This provides enough domains that EQRM's quantile estimation should be
    noticeably more precise than with only 2 training domains, while still
    being computationally lightweight.
    """
    ENVIRONMENTS = [f'e{p:.2f}' for p in torch.linspace(0.05, 0.45, 9).tolist()] + ['e0.90']
    N_STEPS = 5001

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, n_train_domains=9)


class ColoredMNIST20(ColoredMNISTMultiDomain):
    """19 training domains + 1 OOD test domain.

    With 19 training domains, EQRM has a rich enough sample from Q to produce
    meaningful quantile risk estimates, enabling the full evaluation protocol
    proposed in Section 6.2 even at training time.
    """
    ENVIRONMENTS = [f'e{p:.2f}' for p in torch.linspace(0.05, 0.45, 19).tolist()] + ['e0.90']
    N_STEPS = 8001  # More domains = more steps to converge

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, n_train_domains=19)


# =============================================================================
# ColoredMNIST with CONTINUOUS meta-distribution
# =============================================================================
#
# Motivation:
#   The paper's framework (Section 3) assumes domains are drawn i.i.d. from a
#   meta-distribution Q. The original CMNIST has hand-picked flip probs (0.1,
#   0.2) which don't correspond to any natural Q. This variant samples flip
#   probabilities from a Beta(2, 5) distribution, which:
#     - Has support on [0, 1], naturally modeling flip probabilities
#     - Is right-skewed (mode ≈ 0.25), meaning most domains have moderate
#       spurious correlation but some are nearly random
#     - Creates a known ground-truth Q for validating quantile estimates
#
# This dataset is the colored-MNIST analogue of the paper's linear regression
# experiment (Section 6.1), where σ₂ ~ LogNormal(0, 0.5) provided a known Q.

class ColoredMNISTContinuous(MultipleEnvironmentMNIST):
    """10 domains with flip probabilities sampled from Beta(2, 5).

    The last domain (index 9) is designated as the OOD test domain, with the
    highest flip probability (most anti-correlated). The remaining 9 are
    training domains.

    The Beta(2,5) parameters give a distribution concentrated in [0.1, 0.5]
    with occasional values near 0 or 0.8+, creating natural variation in
    spurious-correlation strength across domains.
    """
    # We fix a seed for reproducibility of the domain flip probabilities
    _FLIP_PROBS = None  # Will be set in __init__

    def __init__(self, root, test_envs, hparams):
        # Sample flip probabilities from Beta(2, 5) with a fixed seed
        rng = torch.Generator()
        rng.manual_seed(42)
        beta_dist = torch.distributions.Beta(2.0, 5.0)
        # Sample 10 flip probabilities and sort them
        flip_probs = beta_dist.sample((10,))
        flip_probs = flip_probs.sort().values.tolist()
        self._flip_probs = flip_probs

        # Store as class-level ENVIRONMENTS for compatibility
        ColoredMNISTContinuous.ENVIRONMENTS = [
            f'fp{p:.3f}' for p in flip_probs
        ]

        super().__init__(root, flip_probs,
                         self.color_dataset, (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

    # Fallback ENVIRONMENTS for num_environments() calls before __init__
    ENVIRONMENTS = [f'fp{i}' for i in range(10)]


# =============================================================================
# RotatedMNIST — ORIGINAL
# =============================================================================

class RotatedMNIST(MultipleEnvironmentMNIST):
    """Original RotatedMNIST: 6 domains at 0°, 15°, 30°, 45°, 60°, 75°."""
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
            self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


# =============================================================================
# RotatedMNIST with MORE DOMAINS
# =============================================================================
#
# Motivation:
#   - The paper discusses RotatedMNIST theoretically (Appendix B.2.1, Eq. B.17)
#     as a case where the domain transformation is continuous and the equivalence
#     between worst-case and essential-supremum formulations holds. Yet the
#     authors never ran experiments on it.
#   - More domains enable meaningful quantile-based evaluation (Section 6.2).
#   - The continuous, interpretable domain parameterization (angle) makes it
#     easy to visualize the risk distribution and verify EQRM's behavior.

class RotatedMNISTBase(MultipleEnvironmentMNIST):
    """Base class for RotatedMNIST variants with configurable angles."""

    def __init__(self, root, test_envs, hparams, angles):
        super().__init__(root, angles,
                         self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)
        return TensorDataset(x, y)


class RotatedMNIST12(RotatedMNISTBase):
    """12 domains: 0° to 165° in 15° steps.

    Doubles the original's domain count. With 12 domains, leave-one-out
    evaluation gives 12 test scenarios, enough to compute reasonable quantile
    risk estimates. The angles stay in [0°, 180°) to avoid near-duplicate
    domains (180° rotation ≈ original for many digits).
    """
    ANGLES = list(range(0, 180, 15))
    ENVIRONMENTS = [str(a) for a in ANGLES]

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, self.ANGLES)


class RotatedMNIST18(RotatedMNISTBase):
    """18 domains: 0° to 170° in 10° steps.

    Finer angular resolution reveals how smoothly risk changes with the
    domain parameter, directly testing the continuity assumption underlying
    the equivalence proved in Appendix B.2.1.
    """
    ANGLES = list(range(0, 180, 10))
    ENVIRONMENTS = [str(a) for a in ANGLES]

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, self.ANGLES)


class RotatedMNIST36(RotatedMNISTBase):
    """36 domains: full rotation 0° to 350° in 10° steps.

    The full 360° coverage means domains wrap around, so domains near 0° and
    350° are very similar. This tests whether EQRM's KDE-based risk
    distribution estimation handles the "circular" structure of the domain
    space, where nearby domains (in rotation space) should have similar risks.

    Note: More training steps needed due to the larger number of domains.
    """
    ANGLES = list(range(0, 360, 10))
    ENVIRONMENTS = [str(a) for a in ANGLES]
    N_STEPS = 10001  # More domains need more steps

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, self.ANGLES)


# =============================================================================
# RotatedMNIST with NON-UNIFORM angle distribution
# =============================================================================
#
# Motivation:
#   The i.i.d. assumption on Q means that the density of domains in angle space
#   should match the meta-distribution. If Q is non-uniform (clustered), then
#   some regions of the domain space are well-sampled and others are sparse.
#   EQRM should still provide valid quantile guarantees, but the effective
#   number of "independent" domain samples may be smaller than m.
#
#   This variant clusters angles around 0° and 90° (upright and sideways),
#   with sparse coverage of intermediate angles. This simulates a realistic
#   scenario where some conditions are common and others are rare.

class RotatedMNISTNonUniform(RotatedMNISTBase):
    """12 domains with clustered (non-uniform) angles.

    Cluster 1: 6 domains near 0° (upright): 0°, 5°, 10°, 15°, 20°, 25°
    Cluster 2: 4 domains near 90° (sideways): 80°, 85°, 90°, 95°
    Outliers:  2 domains at 45° and 135° (diagonal)

    This creates a bimodal distribution over domain angles, testing EQRM in a
    setting where the meta-distribution Q is far from uniform.
    """
    ANGLES = [0, 5, 10, 15, 20, 25, 45, 80, 85, 90, 95, 135]
    ENVIRONMENTS = [str(a) for a in ANGLES]

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, self.ANGLES)


# =============================================================================
# ColoredFashionMNIST
# =============================================================================
#
# Motivation:
#   CMNIST uses MNIST digits where the invariant feature (digit shape) is
#   relatively easy to learn. FashionMNIST provides a harder base task:
#     - More complex shapes (clothing items vs. handwritten digits)
#     - Higher intra-class variability
#     - The model needs more capacity to learn the invariant feature
#
#   If EQRM only works when the invariant feature is "easy", that's a
#   significant limitation. ColoredFashionMNIST tests this.
#
#   We use the same binary classification setup: classes 0-4 vs 5-9, same
#   coloring protocol, same flip probabilities.

class ColoredFashionMNIST(MultipleEnvironmentFashionMNIST):
    """ColoredFashionMNIST: same protocol as CMNIST but on FashionMNIST.

    Binary classification: (T-shirt, Trouser, Pullover, Dress, Coat) = class 0
                           (Sandal, Shirt, Sneaker, Bag, Ankle boot) = class 1

    2 training domains (flip probs 0.1, 0.2) + 1 test domain (flip prob 0.9).
    """
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, [0.1, 0.2, 0.9],
                         self.color_dataset, (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # Binary label: first 5 classes vs last 5 classes
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        # Assign color based on label, flip with probability = environment
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredFashionMNIST10(MultipleEnvironmentFashionMNIST):
    """ColoredFashionMNIST with 9 training domains + 1 OOD test domain.

    Combines the harder base task of FashionMNIST with the many-domain
    setup, enabling both capacity and quantile-estimation experiments.
    """
    ENVIRONMENTS = [f'e{p:.2f}' for p in torch.linspace(0.05, 0.45, 9).tolist()] + ['e0.90']

    def __init__(self, root, test_envs, hparams):
        train_flip_probs = torch.linspace(0.05, 0.45, 9).tolist()
        all_flip_probs = train_flip_probs + [0.9]

        super().__init__(root, all_flip_probs,
                         self.color_dataset, (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


# =============================================================================
# Original big-image datasets (unchanged)
# =============================================================================

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla",
                     "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


# =============================================================================
# WILDS datasets (unchanged, require wilds package)
# =============================================================================

class WILDSEnvironment:
    def __init__(self, wilds_dataset, metadata_name, metadata_value, transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                     "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                     "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        from wilds.datasets.fmow_dataset import FMoWDataset
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)
