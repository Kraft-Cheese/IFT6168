import torch

from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups


class VREx(SingleModelAlgorithm):
    """
    V-REx (Variance Risk Extrapolation).

    Reference:
        Krueger et al., Out-of-Distribution Generalization via Risk Extrapolation
        https://arxiv.org/abs/2003.00688
    """

    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        """
        Algorithm-specific arguments (in config):
            - vrex_lambda
            - vrex_penalty_anneal_iters
        """
        # VREx expects group-wise minibatches, same as IRM in this codebase.
        assert config.train_loader == 'group'
        assert config.uniform_over_groups
        assert config.distinct_groups

        model = initialize_model(config, d_out).to(config.device)
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

        self.vrex_lambda = config.vrex_lambda
        self.vrex_penalty_anneal_iters = config.vrex_penalty_anneal_iters
        self.update_count = 0
        self.logged_fields.append('penalty')

    def objective(self, results):
        unique_groups, group_indices, _ = split_into_groups(results['g'])
        n_groups_per_batch = unique_groups.numel()

        group_losses = []
        for i_group in group_indices:
            losses, _ = self.loss.compute_flattened(
                results['y_pred'][i_group],
                results['y_true'][i_group],
                return_dict=False,
            )
            if losses.numel() > 0:
                group_losses.append(losses.mean())

        if len(group_losses) == 0:
            # Defensive fallback; should not happen with valid group loaders.
            avg_loss = torch.tensor(0.0, device=self.device)
            penalty = torch.tensor(0.0, device=self.device)
        else:
            group_losses = torch.stack(group_losses)
            avg_loss = group_losses.mean()
            # V-REx penalty is variance of group risks.
            penalty = ((group_losses - avg_loss) ** 2).mean()

        if self.update_count >= self.vrex_penalty_anneal_iters:
            penalty_weight = self.vrex_lambda
        else:
            penalty_weight = 1.0

        self.save_metric_for_logging(results, 'penalty', penalty)
        return avg_loss + penalty_weight * penalty

    def _update(self, results, should_step=True):
        super()._update(results, should_step=should_step)
        self.update_count += 1
