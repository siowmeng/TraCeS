from __future__ import annotations

import torch

from omnisafe.typing import DEVICE_CPU, AdvatageEstimator, OmnisafeSpace
from omnisafe.utils import distributed
from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer


class OnPolicyBufferH(OnPolicyBuffer):  # pylint: disable=too-many-instance-attributes
    """

    Compared to the on-policy buffer, the on-policy buffer (learned H) stores extra data:

    +--------+---------------------------+---------------+-----------------------------------+
    | Name   | Shape                     | Dtype         | Description                       |
    +========+===========================+===============+===================================+
    | h_obs  | (size, \*hidden_size)     | torch.float32 | The hidden observation.       |
    +--------+---------------------------+---------------+-----------------------------------+
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        hidden_obs_size: int,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:

        super().__init__(obs_space, act_space, size, gamma, lam, lam_c, advantage_estimator,
                         penalty_coefficient, standardized_adv_r, standardized_adv_c, device)

        self.data['hidden_obs'] = torch.zeros((size, hidden_obs_size), dtype=torch.float32, device=device)

    def get(self) -> dict[str, torch.Tensor]:

        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': self.data['obs'],
            'hidden_obs': self.data['hidden_obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
        }

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        # cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        cadv_mean, cadv_std, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean
            # data['adv_c'] = (data['adv_c'] - cadv_mean) / (cadv_std + 1e-8)

        return data
