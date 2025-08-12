from typing import Any, List, Union

import numpy as np
import pandas as pd
import torch as th
from rich.progress import track

import gymnasium
import bullet_safety_gym

from omnisafe.envs.core import CMDP, make, support_envs
from omnisafe.envs.wrapper import (
    ActionScale,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common import Normalizer
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device

from ubcrl.classify.classifier import CostBudgetEstMLP, PtEstGRU, DistributionGRU, RLSFMLP
from ubcrl.common.normalizer import NormalizerH, MeanNormalizer
from ubcrl.common.buffer import VectorOnPolicyBufferH
from ubcrl.models.actor_critic.constraint_actor_critic import ConstraintActorCriticH
from ubcrl.envs.wrapper import (
    HiddenObsNormalize,
    UBCRLRewardNormalize,
    UBCRLActionScale,
    UBCRLUnsqueeze,
    get_wrapper_by_type
)

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

        self._env: Union[CMDP, HiddenObsNormalize] = make(env_id, num_envs=num_envs, device=self._device, **env_cfgs)
        if 'asynchronous' in env_cfgs:
            env_cfgs.pop('asynchronous')
        self._eval_env: Union[CMDP, HiddenObsNormalize] = make(env_id, num_envs=1, device=self._device, **env_cfgs)

        if hasattr(self._env, 'max_episode_steps'):
            setattr(self, 'max_episode_steps', self._env.max_episode_steps)

        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
            hidden_obs_normalize=cfgs.algo_cfgs.hidden_obs_normalize,
        )

        self._env.set_seed(seed)
        self._reset_log()

        # self._logscore_normalizer = Normalizer((), clip=5).to(self._device)
        self._ep_neglogscore_normalizer = NormalizerH((), clip=5).to(self._device)

        if cfgs.model_cfgs.cost_critic.cost_normalize:
            self._cost_normalizer = Normalizer((), clip=5).to(self._device)
        else:
            self._cost_normalizer = None

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
        hidden_obs_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.

        .. hint::
            OmniSafe supports the following wrappers:

        +-----------------+--------------------------------------------------------+
        | Wrapper         | Description                                            |
        +=================+========================================================+
        | TimeLimit       | Limit the time steps of the environment.               |
        +-----------------+--------------------------------------------------------+
        | AutoReset       | Reset the environment when the episode is done.        |
        +-----------------+--------------------------------------------------------+
        | ObsNormalize    | Normalize the observation.                             |
        +-----------------+--------------------------------------------------------+
        | RewardNormalize | Normalize the reward.                                  |
        +-----------------+--------------------------------------------------------+
        | CostNormalize   | Normalize the cost.                                    |
        +-----------------+--------------------------------------------------------+
        | ActionScale     | Scale the action.                                      |
        +-----------------+--------------------------------------------------------+
        | Unsqueeze       | Unsqueeze the step result for single environment case. |
        +-----------------+--------------------------------------------------------+


        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to True.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        if self._env.need_time_limit_wrapper:
            assert (
                self._env.max_episode_steps and self._eval_env.max_episode_steps
            ), 'You must define max_episode_steps as an integer\
                or cancel the use of the time_limit wrapper.'
            self._env = TimeLimit(
                self._env,
                time_limit=self._env.max_episode_steps,
                device=self._device,
            )
            self._eval_env = TimeLimit(
                self._eval_env,
                time_limit=self._eval_env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
            self._eval_env = AutoReset(self._eval_env, device=self._device)
        if obs_normalize:
            self._env = ObsNormalize(self._env, device=self._device)
            self._eval_env = ObsNormalize(self._eval_env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
            # self._env = UBCRLRewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        self._env = HiddenObsNormalize(self._env, device=self._device,
                                       hidden_dim=self._cfgs.model_cfgs.classifier.hidden_dim,
                                       normalize=hidden_obs_normalize)
        self._eval_env = HiddenObsNormalize(self._eval_env, device=self._device,
                                            hidden_dim=self._cfgs.model_cfgs.classifier.hidden_dim,
                                            normalize=hidden_obs_normalize)
        self._env = UBCRLActionScale(self._env, low=-1.0, high=1.0, device=self._device)
        self._eval_env = UBCRLActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
        if self._env.num_envs == 1:
            self._env = UBCRLUnsqueeze(self._env, device=self._device)
        self._eval_env = UBCRLUnsqueeze(self._eval_env, device=self._device)

    def step(
        self,
        action: th.Tensor,
        orig_obs: th.Tensor = None,
        full_hidden_obs: th.Tensor = None,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
    ) -> tuple[
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.
            orig_obs (torch.Tensor): Original Obs for classifier output.
            full_hidden_obs (torch.Tensor): Full Hidden Obs for classifier output.
            classifier (torch.Tensor): Classifier object.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        assert orig_obs is not None, "Original Obs must not be None"
        assert full_hidden_obs is not None, "Full Hidden Obs must not be None"
        assert classifier is not None, "Classifier must not be None"
        return self._env.step(action, orig_obs, full_hidden_obs, classifier)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
    ) -> tuple[th.Tensor, dict[str, Any]]:
        return self._env.reset(seed=seed, options=options, classifier=classifier)

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

        # obs, info = self.reset()
        obs, info = self.reset(classifier=classifier)
        hidden_obs, full_hidden_obs = info['hidden_obs'], info['full_hidden_obs']

        lst_dataframes = []
        traj_obs, traj_act, traj_cost, traj_logscore_mean, traj_logscore_var = [], [], [], [], []
        for _ in range(self._env.num_envs):
            traj_obs.append([])
            traj_act.append([])
            traj_cost.append([])
            traj_logscore_mean.append([])
            traj_logscore_var.append([])

        # hidden_obs = th.zeros((obs.shape[0], self._cfgs.model_cfgs.classifier.hidden_dim)).to(self._device)
        # full_hidden_obs = th.zeros((self._cfgs.model_cfgs.classifier.stack_layer, obs.shape[0], self._cfgs.model_cfgs.classifier.hidden_dim)).to(self._device)
        for step in track(
                range(steps_per_epoch),
                description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            orig_obs = info.get('original_obs', obs.clone())
            act, value_r, value_c, logp = agent.step(obs, hidden_obs)

            next_obs, reward, cost, terminated, truncated, info = self.step(act, orig_obs, full_hidden_obs, classifier)

            # next_hidden_obs, full_hidden_obs = info['next_hidden_obs'], info['next_full_hidden_obs']
            logscores, mean_logscores, var_logscores = info['logscores'], info['mean_logscores'], info['var_logscores']

            # self._log_value(reward=reward, cost=cost, info=info, neg_logscore=-logscores)
            self._log_value(reward=reward, cost=cost, info=info, neg_logscore=-mean_logscores)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            if self._cost_normalizer is None:
                # buffer_cost = -logscores
                buffer_cost = -mean_logscores
            else:
                # buffer_cost = self._cost_normalizer.normalize(-logscores)
                buffer_cost = self._cost_normalizer.normalize(-mean_logscores)

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
            # hidden_obs = next_hidden_obs
            hidden_obs, full_hidden_obs = info['next_hidden_obs'], info['next_full_hidden_obs']
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
                                # info['final_observation'][idx], hidden_obs[idx]
                                info['final_observation'][idx], info['final_hidden_obs'][idx]
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

                        # hidden_obs[idx] = th.zeros(self._cfgs.model_cfgs.classifier.hidden_dim)
                        # full_hidden_obs[:, idx] = th.zeros((self._cfgs.model_cfgs.classifier.stack_layer,
                        #                                     self._cfgs.model_cfgs.classifier.hidden_dim))

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
        # print("Self Ep Neg Log Score", self._ep_neglogscore[idx])
        # print("Norm Ep Neg Log Score", normalized_ep_neglogscores)
        logger.store(
            {
                'Metrics/EpNegLogScore': self._ep_neglogscore[idx],
                'Metrics/EpProbSafe': th.exp(-self._ep_neglogscore[idx]),
                'Metrics/EpNormNegLogScore': normalized_ep_neglogscores,
            },
        )
        # print("Logged EpNegLogScore", self._ep_neglogscore[idx])
        # print("Logged EpProbSafe", th.exp(-self._ep_neglogscore[idx]))

    def _reset_log(self, idx: int | None = None) -> None:
        super()._reset_log(idx)
        if idx is None:
            self._ep_neglogscore = th.zeros(self._env.num_envs)
        else:
            self._ep_neglogscore[idx] = 0.0


class OnPolicyLearnedHumanAdapter(OnPolicyLearnedHAdapter):

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCriticH,
        buffer: VectorOnPolicyBufferH,
        logger: Logger,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
        collect_trajs: bool = False,
        warmup: bool = False,
    ) -> tuple[List[float], List[pd.DataFrame], List, pd.DataFrame]:

        self._reset_log()

        # obs, info = self.reset()
        obs, info = self.reset(classifier=classifier)
        hidden_obs, full_hidden_obs = info['hidden_obs'], info['full_hidden_obs']
        if self._env.num_envs == 1:
            full_hidden_obs = full_hidden_obs.squeeze(0)

        lst_dataframes = []
        traj_obs, traj_act, traj_cost, traj_logscore_mean, traj_logscore_var = [], [], [], [], []
        for _ in range(self._env.num_envs):
            traj_obs.append([])
            traj_act.append([])
            traj_cost.append([])
            traj_logscore_mean.append([])
            traj_logscore_var.append([])

        # hidden_obs = th.zeros((obs.shape[0], self._cfgs.model_cfgs.classifier.hidden_dim)).to(self._device)
        # full_hidden_obs = th.zeros((self._cfgs.model_cfgs.classifier.stack_layer, obs.shape[0], self._cfgs.model_cfgs.classifier.hidden_dim)).to(self._device)
        for step in track(
                range(steps_per_epoch),
                description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            orig_obs = info.get('original_obs', obs.clone())
            # print("Obs:", obs.shape)
            # print("Orig Obs:", orig_obs.shape)
            # print("Hidden Obs:", hidden_obs.shape)
            act, value_r, value_c, logp = agent.step(obs, hidden_obs)
            # print("Act:", act.shape)
            # print("Value R:", value_r)
            # print("Value C:", value_c)
            # print("LogP:", logp)

            if self._env.num_envs == 1:
                next_obs, reward, cost, terminated, truncated, info = self.step(act.unsqueeze(0), orig_obs, full_hidden_obs, classifier)
            else:
                next_obs, reward, cost, terminated, truncated, info = self.step(act, orig_obs, full_hidden_obs, classifier)
            # print("Render", self._env.render())

            # next_hidden_obs, full_hidden_obs = info['next_hidden_obs'], info['next_full_hidden_obs']
            logscores, mean_logscores, var_logscores = info['logscores'], info['mean_logscores'], info['var_logscores']
            if self._env.num_envs == 1:
                mean_logscores = mean_logscores.squeeze(0)

            # print("Reward:", reward)
            # print("Cost:", cost)
            # print("Info:", info)
            # print("Neg Log Score:", -mean_logscores)

            # self._log_value(reward=reward, cost=cost, info=info, neg_logscore=-logscores)
            self._log_value(reward=reward, cost=cost, info=info, neg_logscore=-mean_logscores)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            if self._cost_normalizer is None:
                # buffer_cost = -logscores
                buffer_cost = -mean_logscores
            else:
                # buffer_cost = self._cost_normalizer.normalize(-logscores)
                buffer_cost = self._cost_normalizer.normalize(-mean_logscores)

            buffer.store(
                obs=obs,
                hidden_obs=hidden_obs,
                act=act,
                reward=reward,
                # reward=th.zeros_like(reward),
                # cost=cost,
                # cost=-logscores,
                cost=buffer_cost,
                # cost=-normalized_logscores,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            # hidden_obs = next_hidden_obs
            hidden_obs, full_hidden_obs = info['next_hidden_obs'], info['next_full_hidden_obs']
            if self._env.num_envs == 1:
                full_hidden_obs = full_hidden_obs.squeeze(0)
                hidden_obs = hidden_obs.squeeze(0)
            epoch_end = step >= steps_per_epoch - 1

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):

                if collect_trajs:
                    traj_obs[idx].append(orig_obs[idx].cpu().detach().numpy())
                    traj_act[idx].append(act[idx].cpu().detach().numpy())
                    traj_cost[idx].append(cost[idx].cpu().detach().numpy())
                    traj_logscore_mean[idx].append(mean_logscores[idx].cpu().detach().numpy())
                    traj_logscore_var[idx].append(var_logscores[idx].cpu().detach().numpy())

                if epoch_end or done or time_out:
                    # time_out = True  # hardcoded for horizon cut-short

                    last_value_r = th.zeros(1)
                    last_value_c = th.zeros(1)
                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.',
                            )
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx], hidden_obs[idx])
                        if time_out:
                            final_hidden_obs = info['final_hidden_obs'][idx]
                            if self._env.num_envs == 1:
                                final_hidden_obs = final_hidden_obs.squeeze(0)
                            _, last_value_r, last_value_c, _ = agent.step(
                                # info['final_observation'][idx], hidden_obs[idx]
                                info['final_observation'][idx], final_hidden_obs
                            )
                            # last_value_r, last_value_c = value_r[idx], value_c[idx]  # Use previous value to approximate
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        self._ep_neglogscore[idx] = 0.0

                        # hidden_obs[idx] = th.zeros(self._cfgs.model_cfgs.classifier.hidden_dim)
                        # full_hidden_obs[:, idx] = th.zeros((self._cfgs.model_cfgs.classifier.stack_layer,
                        #                                     self._cfgs.model_cfgs.classifier.hidden_dim))

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

        if warmup:
            eval_frames, eval_traj_df = None, None
        else:
            eval_frames, eval_traj_df = self.eval_rollout(self.max_episode_steps, agent, classifier)

        # Note: Compute the final mean and variance for the sum of RVs to get the final CV of the sum of RVs
        # Note: CV of the sum is not the sum of CVs
        return ([np.float64(abs(df['logscore_var'].sum()**0.5 / df['logscore_mean'].sum())) for df in lst_dataframes],
                lst_dataframes, eval_frames, eval_traj_df)

    def eval_rollout(
        self,
        steps_per_episode: int,
        agent: ConstraintActorCriticH,
        classifier: Union[PtEstGRU, DistributionGRU] = None
    ) -> tuple[List, pd.DataFrame]:

        assert self._eval_env.num_envs == 1

        if self._cfgs.algo_cfgs.obs_normalize:
            train_obs_norm = get_wrapper_by_type(self._env, ObsNormalize)
            eval_obs_norm = get_wrapper_by_type(self._eval_env, ObsNormalize)
            eval_obs_norm._obs_normalizer.load_state_dict(train_obs_norm._obs_normalizer.state_dict())

        if self._cfgs.algo_cfgs.hidden_obs_normalize:
            train_hidden_obs_norm = get_wrapper_by_type(self._env, HiddenObsNormalize)
            eval_hidden_obs_norm = get_wrapper_by_type(self._eval_env, HiddenObsNormalize)
            eval_hidden_obs_norm._hidden_obs_normalizer.load_state_dict(train_hidden_obs_norm._hidden_obs_normalizer.state_dict())

        obs, info = self._eval_env.reset(classifier=classifier)
        # obs, info = self.reset(classifier=classifier)
        hidden_obs, full_hidden_obs = info['hidden_obs'], info['full_hidden_obs']
        if self._eval_env.num_envs == 1:
            full_hidden_obs = full_hidden_obs.squeeze(0)

        frames = []
        traj_obs, traj_act, traj_r, traj_c = [], [], [], []
        for step in range(steps_per_episode):
            with th.no_grad():
                orig_obs = info.get('original_obs', obs.clone())
                act = agent.actor.predict(obs, hidden_obs, deterministic=True)
                # act = agent.actor.predict(obs, hidden_obs, deterministic=False)
                # act = th.tensor([[0., 0.]])

            # print("Obs", obs.shape)
            # print("hidden_obs", hidden_obs.shape)
            # print("full_hidden_obs", full_hidden_obs.shape)
            # print("act", act.shape)
            # print("orig_obs", orig_obs.shape)

            # if self._eval_env.num_envs == 1:
            #     obs, rew, cost, terminated, truncated, info = self._eval_env.step(act.unsqueeze(0), orig_obs, full_hidden_obs, classifier)
            # else:
            #     obs, rew, cost, terminated, truncated, info = self._eval_env.step(act, orig_obs, full_hidden_obs, classifier)
            # print("Eval Action:", act)
            obs, rew, cost, terminated, truncated, info = self._eval_env.step(act, orig_obs, full_hidden_obs, classifier)

            orig_rew = info.get('original_reward', rew.clone()).item()
            frames.append(self._eval_env.render())
            traj_obs.append(orig_obs.cpu().detach().numpy())
            traj_act.append(act.cpu().detach().numpy())
            traj_r.append(orig_rew)
            traj_c.append(0)

            hidden_obs, full_hidden_obs = info['next_hidden_obs'], info['next_full_hidden_obs']
            if self._eval_env.num_envs == 1:
                full_hidden_obs = full_hidden_obs.squeeze(0)
                hidden_obs = hidden_obs.squeeze(0)

            if terminated or truncated:
                break

        obs_np = np.vstack(traj_obs)
        obs_header = ['s' + str(i) for i in range(obs_np.shape[1])]
        act_np = np.vstack(traj_act)
        act_header = ['a' + str(i) for i in range(act_np.shape[1])]

        traj_df = pd.DataFrame()
        traj_df[obs_header] = obs_np
        traj_df[act_header] = act_np
        traj_df['r'] = traj_r
        traj_df['c'] = traj_c

        return frames, traj_df

    # def eval_rollout(
    #     self,
    #     steps_per_episode: int,
    #     agent: ConstraintActorCriticH,
    #     classifier: Union[PtEstGRU, DistributionGRU] = None
    # ) -> List:
    #
    #     obs, info = self.reset(classifier=classifier)
    #     hidden_obs, full_hidden_obs = info['hidden_obs'], info['full_hidden_obs']
    #     if self._env.num_envs == 1:
    #         full_hidden_obs = full_hidden_obs.squeeze(0)
    #
    #     frames = []
    #     for step in range(steps_per_episode):
    #         with th.no_grad():
    #             orig_obs = info.get('original_obs', obs.clone())
    #             # act = agent.actor.predict(obs, hidden_obs, deterministic=True)
    #             act = agent.actor.predict(obs, hidden_obs, deterministic=False)
    #
    #         if self._env.num_envs == 1:
    #             obs, rew, cost, terminated, truncated, info = self._env.step(act.unsqueeze(0), orig_obs, full_hidden_obs, classifier)
    #         else:
    #             obs, rew, cost, terminated, truncated, info = self._env.step(act, orig_obs, full_hidden_obs, classifier)
    #         frames.append(self._env.render())
    #
    #         hidden_obs, full_hidden_obs = info['next_hidden_obs'], info['next_full_hidden_obs']
    #         if self._env.num_envs == 1:
    #             full_hidden_obs = full_hidden_obs.squeeze(0)
    #             hidden_obs = hidden_obs.squeeze(0)
    #
    #     return frames

class OnPolicyLearnedBCHumanAdapter(OnPolicyLearnedBCAdapter):

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
        classifier: Union[CostBudgetEstMLP, RLSFMLP] = None,
        collect_trajs: bool = False
    ) -> tuple[th.Tensor, List[pd.DataFrame], List]:

        self._reset_log()

        obs, info = self.reset()
        learned_budget = None

        lst_dataframes = []
        traj_obs, traj_act, traj_cost = [], [], []
        for _ in range(self._env.num_envs):
            traj_obs.append([])
            traj_act.append([])
            traj_cost.append([])

        frames = []
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            orig_obs = info.get('original_obs', obs.clone())
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            frames.append(self._env.render())

            # Learned cost and budget from classifier model
            obs_action = th.concat((orig_obs, act), dim=-1)
            if isinstance(classifier, RLSFMLP):
                learned_logits = classifier(obs_action)
                learned_cost = -th.nn.Sigmoid()(learned_logits)
                learned_budget = th.zeros(1)
            else:
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
                reward=th.zeros_like(reward),
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

        return learned_budget.detach().cpu().item(), lst_dataframes, frames

    def eval_rollout(
        self,
        steps_per_episode: int,
        agent: ConstraintActorCritic,
    ) -> List:

        obs, info = self.reset()

        frames = []
        for step in range(steps_per_episode):
            with th.no_grad():
                # act = agent.actor.predict(obs, deterministic=True)
                act = agent.actor.predict(obs, deterministic=False)

            obs, rew, cost, terminated, truncated, info = self._env.step(act)
            frames.append(self._env.render())

        return frames
