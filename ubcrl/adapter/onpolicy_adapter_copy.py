from typing import Any, List, Union

import numpy as np
import pandas as pd
import torch as th
from rich.progress import track

import gymnasium
import bullet_safety_gym

from omnisafe.envs.core import CMDP, make, support_envs
from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common import Normalizer
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device

from ubcrl.classify.classifier import CostBudgetEstMLP, PtEstGRU, DistributionGRU
from ubcrl.common.normalizer import MeanNormalizer, NormalizerH
from ubcrl.common.buffer import VectorOnPolicyBufferH
from ubcrl.models.actor_critic.constraint_actor_critic import ConstraintActorCriticH

class OnPolicyLearnedBCAdapter(OnPolicyAdapter):

    _ep_learned_cost: th.Tensor
    _ep_learned_budget: float

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._cfgs: Config = cfgs
        self._device: th.device = get_device(cfgs.train_cfgs.device)
        self._env_id: str = env_id

        env_cfgs = {}

        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()

        self._env: CMDP = make(env_id, num_envs=num_envs, device=self._device, **env_cfgs)
        if 'asynchronous' in env_cfgs:
            env_cfgs.pop('asynchronous')
        self._eval_env: CMDP = make(env_id, num_envs=1, device=self._device, **env_cfgs)

        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
        )

        self._env.set_seed(seed)
        self._reset_log()

        if cfgs.model_cfgs.critic.cost_normalize:
            self._cost_normalizer = Normalizer((), clip=5).to(self._device)
        else:
            self._cost_normalizer = None

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
        classifier: CostBudgetEstMLP = None,
        collect_trajs: bool = False
    ) -> tuple[th.Tensor, List[pd.DataFrame]]:

        self._reset_log()

        obs, info = self.reset()
        learned_budget = None

        lst_dataframes = []
        traj_obs, traj_act, traj_cost = [], [], []
        for _ in range(self._env.num_envs):
            traj_obs.append([])
            traj_act.append([])
            traj_cost.append([])

        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            orig_obs = info.get('original_obs', obs.clone())
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            # Learned cost and budget from classifier model
            obs_action = th.concat((orig_obs, act), dim=-1)
            learned_cost, learned_budget = classifier(obs_action)
            learned_cost = learned_cost.squeeze(axis=-1)

            # print("Reward Shape: ", reward.shape)
            # print("Learned Cost Shape: ", learned_cost.shape)

            self._log_value(reward=reward, cost=cost, info=info, learned_cost=learned_cost, learned_budget=learned_budget)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            if self._cost_normalizer is None:
                buffer_cost = learned_cost
            else:
                buffer_cost = self._cost_normalizer.normalize(learned_cost)

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                # cost=cost,
                # cost=learned_cost,
                cost=buffer_cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):

                if collect_trajs:
                    traj_obs[idx].append(orig_obs[idx].cpu().detach().numpy())
                    traj_act[idx].append(act[idx].cpu().detach().numpy())
                    traj_cost[idx].append(cost[idx].cpu().detach().numpy())

                if epoch_end or done or time_out:
                    last_value_r = th.zeros(1)
                    last_value_c = th.zeros(1)
                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.',
                            )
                        # Not applicable to our experiments
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        self._ep_learned_cost[idx] = 0.0
                        self._ep_learned_budget = 0.0

                    if collect_trajs:
                        obs_np = np.vstack(traj_obs[idx])
                        obs_header = ['s' + str(i) for i in range(obs_np.shape[1])]
                        act_np = np.vstack(traj_act[idx])
                        act_header = ['a' + str(i) for i in range(act_np.shape[1])]

                        traj_df = pd.DataFrame()
                        traj_df[obs_header] = obs_np
                        traj_df[act_header] = act_np
                        traj_df['c'] = traj_cost[idx]

                        lst_dataframes.append(traj_df)
                        traj_obs[idx], traj_act[idx], traj_cost[idx] = [], [], []

                    buffer.finish_path(last_value_r, last_value_c, idx)

        logger.store({'Metrics/EpLearnedBudget': learned_budget.cpu().item()})

        return learned_budget.detach().cpu().item(), lst_dataframes

    def _log_value(
        self,
        reward: th.Tensor,
        cost: th.Tensor,
        info: dict[str, Any],
        learned_cost: th.Tensor = None,
        learned_budget: th.Tensor = None
    ) -> None:

        super()._log_value(reward, cost, info)

        self._ep_learned_cost += learned_cost.cpu()
        self._ep_learned_budget = learned_budget.cpu().item()

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        super()._log_metrics(logger, idx)
        logger.store(
            {
                'Metrics/EpLearnedCost': self._ep_learned_cost[idx],
                # 'Metrics/EpLearnedBudget': self._ep_learned_budget,
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        super()._reset_log(idx)
        if idx is None:
            self._ep_learned_cost = th.zeros(self._env.num_envs)
        else:
            self._ep_learned_cost[idx] = 0.0
        self._ep_learned_budget = 0.0


class OnPolicyLearnedHAdapter(OnPolicyAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_neglogscore: th.Tensor
    _ep_ret: th.Tensor
    # _ep_cost: th.Tensor
    # _ep_len: th.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._cfgs: Config = cfgs
        self._device: th.device = get_device(cfgs.train_cfgs.device)
        self._env_id: str = env_id

        env_cfgs = {}

        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()

        self._env: CMDP = make(env_id, num_envs=num_envs, device=self._device, **env_cfgs)
        if 'asynchronous' in env_cfgs:
            env_cfgs.pop('asynchronous')
        self._eval_env: CMDP = make(env_id, num_envs=1, device=self._device, **env_cfgs)

        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
        )

        self._env.set_seed(seed)
        self._reset_log()

        # self._logscore_normalizer = Normalizer((), clip=5).to(self._device)
        self._ep_neglogscore_normalizer = NormalizerH((), clip=5).to(self._device)

        if cfgs.model_cfgs.cost_critic.cost_normalize:
            self._cost_normalizer = Normalizer((), clip=5).to(self._device)
        else:
            self._cost_normalizer = None

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCriticH,
        buffer: VectorOnPolicyBufferH,
        logger: Logger,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
        collect_trajs: bool = False
    ) -> tuple[List[float], List[pd.DataFrame]]:

        self._reset_log()

        obs, info = self.reset()

        lst_dataframes = []
        traj_obs, traj_act, traj_cost, traj_logscore_mean, traj_logscore_var = [], [], [], [], []
        for _ in range(self._env.num_envs):
            traj_obs.append([])
            traj_act.append([])
            traj_cost.append([])
            traj_logscore_mean.append([])
            traj_logscore_var.append([])

        hidden_obs = th.zeros((obs.shape[0], self._cfgs.model_cfgs.classifier.hidden_dim)).to(self._device)
        full_hidden_obs = th.zeros((self._cfgs.model_cfgs.classifier.stack_layer, obs.shape[0], self._cfgs.model_cfgs.classifier.hidden_dim)).to(self._device)
        for step in track(
                range(steps_per_epoch),
                description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs, hidden_obs)

            # Learned cost and budget from classifier model
            orig_obs = info.get('original_obs', obs.clone())
            obs_action = th.concat((orig_obs, act), dim=-1)

            prob_feasible, dict_logscores_t, hidden_obs_t, full_hidden_obs = classifier(
                obs_action.unsqueeze(dim=1),
                th.FloatTensor([1] * obs_action.shape[0]).to(self._device),
                init_h=full_hidden_obs
            )

            logscores_t, mean_logscores_t, var_logscores_t = (
                dict_logscores_t['log_scores'], dict_logscores_t['mean'], dict_logscores_t['var']
            )

            # Get logC and H at the curr timestep h
            logscores, next_hidden_obs, mean_logscores, var_logscores = (
                logscores_t[-1], hidden_obs_t[-1], mean_logscores_t[-1], var_logscores_t[-1]
            )
            # Classifier log score output is clipped at classifier.py MIN_LOGSCORE = -7
            logscores = logscores.squeeze(axis=-1)
            mean_logscores, var_logscores = mean_logscores.squeeze(axis=-1), var_logscores.squeeze(axis=-1)

            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info, neg_logscore=-logscores)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            if self._cost_normalizer is None:
                buffer_cost = -logscores
            else:
                buffer_cost = self._cost_normalizer.normalize(-logscores)

            buffer.store(
                obs=obs,
                hidden_obs=hidden_obs,
                act=act,
                reward=reward,
                # cost=cost,
                # cost=-logscores,
                cost=buffer_cost,
                # cost=-normalized_logscores,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            hidden_obs = next_hidden_obs
            epoch_end = step >= steps_per_epoch - 1

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):

                if collect_trajs:
                    traj_obs[idx].append(orig_obs[idx].cpu().detach().numpy())
                    traj_act[idx].append(act[idx].cpu().detach().numpy())
                    traj_cost[idx].append(cost[idx].cpu().detach().numpy())
                    traj_logscore_mean[idx].append(mean_logscores[idx].cpu().detach().numpy())
                    traj_logscore_var[idx].append(var_logscores[idx].cpu().detach().numpy())

                if epoch_end or done or time_out:
                    last_value_r = th.zeros(1)
                    last_value_c = th.zeros(1)
                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.',
                            )
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx], hidden_obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx], hidden_obs[idx]
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        self._ep_neglogscore[idx] = 0.0

                        hidden_obs[idx] = th.zeros(self._cfgs.model_cfgs.classifier.hidden_dim)
                        full_hidden_obs[:, idx] = th.zeros((self._cfgs.model_cfgs.classifier.stack_layer,
                                                            self._cfgs.model_cfgs.classifier.hidden_dim))

                    if collect_trajs:
                        obs_np = np.vstack(traj_obs[idx])
                        obs_header = ['s' + str(i) for i in range(obs_np.shape[1])]
                        act_np = np.vstack(traj_act[idx])
                        act_header = ['a' + str(i) for i in range(act_np.shape[1])]

                        traj_df = pd.DataFrame()
                        traj_df[obs_header] = obs_np
                        traj_df[act_header] = act_np
                        traj_df['c'] = traj_cost[idx]
                        traj_df['logscore_mean'] = traj_logscore_mean[idx]
                        traj_df['logscore_var'] = traj_logscore_var[idx]
                        traj_df['logscore_cv'] = traj_df['logscore_var'].pow(0.5) / traj_df['logscore_mean']
                        # print(traj_df)

                        lst_dataframes.append(traj_df)
                        traj_obs[idx], traj_act[idx], traj_cost[idx], traj_logscore_mean[idx], traj_logscore_var[idx]  = [], [], [], [], []

                    buffer.finish_path(last_value_r, last_value_c, idx)

        # Note: Compute the final mean and variance for the sum of RVs to get the final CV of the sum of RVs
        # Note: CV of the sum is not the sum of CVs
        return [np.float64(abs(df['logscore_var'].sum()**0.5 / df['logscore_mean'].sum())) for df in lst_dataframes], lst_dataframes

    def _log_value(
            self,
            reward: th.Tensor,
            cost: th.Tensor,
            info: dict[str, Any],
            neg_logscore: th.Tensor = None
    ) -> None:

        super()._log_value(reward, cost, info)
        self._ep_neglogscore += neg_logscore.cpu()

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        super()._log_metrics(logger, idx)
        normalized_ep_neglogscores = self._ep_neglogscore_normalizer.normalize(self._ep_neglogscore[idx])
        logger.store(
            {
                'Metrics/EpNegLogScore': self._ep_neglogscore[idx],
                'Metrics/EpNormNegLogScore': normalized_ep_neglogscores,
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        super()._reset_log(idx)
        if idx is None:
            self._ep_neglogscore = th.zeros(self._env.num_envs)
        else:
            self._ep_neglogscore[idx] = 0.0
