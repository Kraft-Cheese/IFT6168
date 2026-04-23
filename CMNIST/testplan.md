Test Plan
Phase 1: Establishing the Baseline & The "Trap"
Before testing the theoretical bounds, you must prove that the baseline models fail reliably and that the task is structurally sound.

Experiment 1.1: The Standard Reproduction (O2O CMNIST)
Goal: Replicate the original paper's 2-domain ColoredMNIST setup to ensure your ERM and EQRM implementations are mathematically correct and the burn-in phase is functioning.

Experiment 1.2: The Multi-Color Stress Test (MC-CMNIST)
Goal: Implement the 10-class, 10-color variant. This proves that you can prevent the "dilution" of the spurious correlation and guarantees that ERM will catastrophically fail, setting a robust stage for all scaling experiments to follow.

Phase 2: Probing Theorem 4.1 (Environment Scaling)
With the MC-CMNIST baseline established, you can now directly attack the core theoretical claim: that the empirical quantile risk converges to the true population risk as environments scale.

Experiment 2.1: Domain Scaling (Idea 1)
Setup: MC-CMNIST with 2, 5, 10, and 20 training domains. Use the "Narrow Trap" spacing to keep the pooled spurious correlation consistently deceptive.

What it proves: Measures whether EQRM's test accuracy monotonically improves as the number of domains increases, validating Theorem 4.1 on high-dimensional vision data rather than just linear SCMs.

Experiment 2.2: Continuous Meta-Distributions (Idea 2)
Setup: MC-CMNIST where the spurious flip probabilities are sampled from a right-skewed Beta(2, 5) distribution, mapped strictly to your highly spurious range.

What it proves: Tests if EQRM survives when domains are not perfectly spaced but naturally clustered, with extreme tail domains acting as statistical outliers. This bridges the gap between theoretical i.i.d. assumptions and messy, real-world sampling.

Phase 3: Breaking the Mechanisms (Topology & Capacity)
Once scaling is proven, the final phase shifts to finding exactly where the algorithm collapses by introducing non-Euclidean domain shifts and high-variance causal features.

Experiment 3.1: Non-Euclidean Domain Spaces (Idea 3)
Setup: RotatedMNIST with 12, 18, and 36 domains. Include a uniform sampling track and a bimodal track (clustering angles near 0 and 90 degrees).

What it proves: Evaluates the robustness of EQRM's Kernel Density Estimation (KDE). By testing a "circular" domain space (where 0 and 350 degrees are visually identical), you stress-test the smoothing function and determine if the penalty holds up under non-linear, non-Euclidean domain transformations.

Experiment 3.2: Invariant Feature Complexity (Idea 4)
Setup: ColoredFashionMNIST using the same domain splits as Phase 2.

What it proves: Isolates the capacity of the model to learn the true causal mechanism. Because FashionMNIST has complex textures and high intra-class variance, this tests whether EQRM’s gradient penalty collapses when the invariant feature itself is difficult for the optimizer to extract, offering a brilliant critique of why the original authors relied so heavily on ERM pre-training.

Bonus Experiments

1. The Burn-In Ablation (Does EQRM actually learn, or just fine-tune?)
   The Setup: Take your MC-CMNIST baseline and sweep the erm_pretrain_iters parameter (0%, 25%, 50%, 75%, and 100% of total training steps).

What it proves: This tests whether EQRM is capable of escaping the spurious trap on its own (with 0 burn-in steps) or if it absolutely requires ERM to initialize the weights in a specific basin of attraction before the penalty becomes mathematically useful.

2. Domain Imbalance (The "Rare Environment" Problem)
   The Setup: Use your MC-CMNIST setup with N=5 domains. Instead of giving each domain 10,000 images, heavily skew the sample sizes (e.g., Domain 1 gets 80% of the data, Domain 5 gets 1%).

What it proves: Because EQRM relies on Kernel Density Estimation (KDE) to smooth the risk distribution, extreme sample-size imbalances could cause the KDE to wildly overestimate the variance of the smallest domains. This proves whether the algorithm can survive real-world data collection imbalances or if it over-indexes on noisy, low-sample domains.
