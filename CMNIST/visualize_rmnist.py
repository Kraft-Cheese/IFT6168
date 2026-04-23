import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.datasets import MNIST
from datasets_rmnist import rotate_dataset_rmnist, GROUP1_DIGITS

# ---------------------------------------------------------------------------
# Panel 1: Canonical rotation angles — one image per binary group × delta
# ---------------------------------------------------------------------------

def plot_canonical_rotations(root_dir='./data'):
    """
    Shows the same digit rotated at the exact mean angles for each environment.

    Rows : delta values (strong → moderate → OOD inverted → oracle)
    Cols : one representative from Group 0 (digit 3) and Group 1 (digit 6)
    """
    mnist = MNIST(root_dir, train=False, download=True)

    # Pick one clean example per group
    group_digits = {0: None, 1: None}
    for img, label in mnist:
        g = 1 if label in GROUP1_DIGITS.tolist() else 0
        if group_digits[g] is None:
            group_digits[g] = (torch.tensor(np.array(img), dtype=torch.uint8), label)
        if all(v is not None for v in group_digits.values()):
            break

    configs = [
        ("Train Env 1\n(δ=60°)",   60.0),
        ("Train Env 2\n(δ=30°)",   30.0),
        ("OOD Test\n(δ=−60°)",    -60.0),
        ("Oracle\n(δ=0°)",          0.0),
    ]

    fig, axes = plt.subplots(len(configs), 2, figsize=(4, len(configs) * 2))

    for row, (env_label, delta) in enumerate(configs):
        mu_0 = 45.0 - delta / 2.0
        mu_1 = 45.0 + delta / 2.0

        for col, (g, (raw_img, digit_label)) in enumerate(group_digits.items()):
            mu = mu_0 if g == 0 else mu_1
            # Rotate by exactly the mean angle (no noise)
            img_float = raw_img.float().div(255.0)   # (28, 28)
            img_4d    = img_float[None, None, :, :]  # (1, 1, 28, 28)
            angle_t   = torch.tensor([mu])

            from datasets_rmnist import rotate_images_batch
            rotated = rotate_images_batch(img_4d, angle_t)[0, 0]  # (28, 28)

            ax = axes[row, col]
            ax.imshow(rotated.numpy(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(
                f"{'Group 0' if g == 0 else 'Group 1'} (digit {digit_label})\n"
                f"μ={mu:.0f}°",
                fontsize=8
            )
            ax.axis('off')

        axes[row, 0].set_ylabel(env_label, fontsize=8, rotation=0,
                                labelpad=55, va='center')

    fig.suptitle("RMNIST — Canonical Rotations per Environment\n"
                 "(no noise, rotated at exact mean angle μ)", fontsize=11)
    plt.tight_layout()
    plt.savefig("rmnist_canonical_rotations.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → rmnist_canonical_rotations.png")


# ---------------------------------------------------------------------------
# Panel 2: Sample grid — natural variation from the Gaussian noise (σ=5°)
# ---------------------------------------------------------------------------

def plot_sample_grid(root_dir='./data', n_per_group=8):
    """
    Two rows of real RMNIST samples from Train Env 1 (delta=60).

    Row 0: Group 0 digits (0–4) rotated around μ=15°  (σ=5°)
    Row 1: Group 1 digits (5–9) rotated around μ=75°  (σ=5°)

    This visualises the actual Gaussian spread around the mean angle.
    """
    mnist = MNIST(root_dir, train=False, download=True)

    g0_imgs, g0_lbls, g1_imgs, g1_lbls = [], [], [], []
    for img, label in mnist:
        t = torch.tensor(np.array(img), dtype=torch.uint8)
        if label not in GROUP1_DIGITS.tolist() and len(g0_imgs) < n_per_group:
            g0_imgs.append(t); g0_lbls.append(label)
        elif label in GROUP1_DIGITS.tolist() and len(g1_imgs) < n_per_group:
            g1_imgs.append(t); g1_lbls.append(label)
        if len(g0_imgs) == n_per_group and len(g1_imgs) == n_per_group:
            break

    images = torch.stack(g0_imgs + g1_imgs)           # (2N, 28, 28)
    labels = torch.tensor(g0_lbls + g1_lbls)

    # Build dataset with Train Env 1 (delta=60, strong shortcut), no cuda
    dataset = rotate_dataset_rmnist(
        images, labels, env_delta=60.0, sigma_deg=5.0,
        subsample=False, cuda=False
    )

    fig, axes = plt.subplots(2, n_per_group, figsize=(n_per_group * 1.5, 3.5))
    group_names = ["Group 0  (digits 0–4,  μ=15°)", "Group 1  (digits 5–9,  μ=75°)"]

    for row in range(2):
        for col in range(n_per_group):
            idx = row * n_per_group + col
            img_t, lbl = dataset[idx]
            ax = axes[row, col]
            ax.imshow(img_t[0].numpy(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"{lbl.item()}", fontsize=8)
            ax.axis('off')
        axes[row, 0].set_ylabel(group_names[row], fontsize=8,
                                rotation=0, labelpad=120, va='center')

    fig.suptitle(
        "RMNIST — Train Env 1 (δ=60°,  σ=5°)\n"
        "Gaussian rotation noise visible within each row",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("rmnist_sample_grid.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → rmnist_sample_grid.png")


# ---------------------------------------------------------------------------
# Panel 3: Angle distribution — histogram of sampled angles per group/env
# ---------------------------------------------------------------------------

def plot_angle_distributions(root_dir='./data', n_samples=500):
    """
    Histogram of rotation angles drawn for Group 0 and Group 1 across
    the four key environments, confirming the N(μ, 5²) parameterisation.
    """
    mnist    = MNIST(root_dir, train=False, download=True)
    all_imgs = mnist.data[:n_samples]
    all_lbls = mnist.targets[:n_samples]

    configs = [
        ("Train Env 1 (δ=60°)",  60.0),
        ("Train Env 2 (δ=30°)",  30.0),
        ("OOD Test   (δ=−60°)", -60.0),
        ("Oracle     (δ=0°)",    0.0),
    ]

    fig, axes = plt.subplots(1, len(configs), figsize=(14, 3.2), sharey=False)
    colors = {0: "#4477aa", 1: "#cc3311"}

    for ax, (env_label, delta) in zip(axes, configs):
        binary_labels = torch.isin(all_lbls, GROUP1_DIGITS).long()
        mu_0 = 45.0 - delta / 2.0
        mu_1 = 45.0 + delta / 2.0

        for g, (mu, color) in enumerate([(mu_0, colors[0]), (mu_1, colors[1])]):
            mask   = binary_labels == g
            n_g    = mask.sum().item()
            mean_t = torch.full((n_g,), mu)
            angles = torch.normal(mean=mean_t, std=5.0).numpy()
            ax.hist(angles, bins=25, alpha=0.65, color=color,
                    label=f"Group {g}  μ={mu:.0f}°")
            ax.axvline(mu, color=color, linestyle='--', linewidth=1.5)

        ax.set_title(env_label, fontsize=9)
        ax.set_xlabel("Rotation angle θ (°)", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlim(-15, 105)

    axes[0].set_ylabel("Count", fontsize=8)
    fig.suptitle(
        f"RMNIST — Sampled Angle Distributions  (σ=5°, n={n_samples} images)\n"
        "Dashed lines mark the group means μ₀ and μ₁",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("rmnist_angle_distributions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → rmnist_angle_distributions.png")


# ---------------------------------------------------------------------------
# Run all panels
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = './data'
    plot_canonical_rotations(DATA_DIR)
    plot_sample_grid(DATA_DIR, n_per_group=8)
    plot_angle_distributions(DATA_DIR, n_samples=500)
