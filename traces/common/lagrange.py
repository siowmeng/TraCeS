import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.common.lagrange import Lagrange

class LagrangeH(Lagrange):

    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        use_max_grad_norm: bool,
        max_grad_norm: float | None = None,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        super().__init__(cost_limit, lagrangian_multiplier_init, lambda_lr, lambda_optimizer, lagrangian_upper_bound)
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=lambda_lr,
            # betas=(0.85, 0.98),
        )
        self.use_max_grad_norm = use_max_grad_norm
        self.max_grad_norm = max_grad_norm

    def compute_lambda_loss(self, mean_ep_cost: float, norm_cost_limit: float = None) -> torch.Tensor:
        return -self.lagrangian_multiplier * (mean_ep_cost - norm_cost_limit)

    def update_lagrange_multiplier(self, norm_Jc: float, norm_cost_limit: float = None) -> None:
        self.lambda_optimizer.zero_grad()
        # lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss = self.compute_lambda_loss(norm_Jc, norm_cost_limit)
        lambda_loss.backward()
        if self.use_max_grad_norm:
            clip_grad_norm_(
                [self.lagrangian_multiplier],
                self.max_grad_norm,
            )
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0.0,
            self.lagrangian_upper_bound,
        )  # enforce: lambda in [0, inf]
