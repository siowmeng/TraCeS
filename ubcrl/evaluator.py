import json
import os
from typing import Any, Union

import numpy as np
import torch

from gymnasium.spaces import Box

from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.common import Normalizer
from omnisafe.envs.core import make, CMDP
from omnisafe.envs.wrapper import ActionRepeat, ActionScale, ObsNormalize, TimeLimit
from omnisafe.models import ConstraintActorCritic
from omnisafe.models.actor import ActorBuilder
from omnisafe.models.base import Actor
from omnisafe.evaluator import Evaluator
from omnisafe.utils.config import Config

from omnisafe.envs import support_envs
from ubcrl.envs.bullet_safety_env import BulletSafetyEnv

from ubcrl.classify.classifier import mujoco_safety_gymnasium_dict, DistributionGRU
from ubcrl.envs.wrapper import HiddenObsNormalize, UBCRLActionScale, UBCRLUnsqueeze
from ubcrl.models.actor import  ActorBuilderH
from ubcrl.models.base import ActorH
from ubcrl.models import ConstraintActorCriticH

class UBCRLEvaluator(Evaluator):
    _actor: Actor | ActorH | None = None,
    _actor_critic: ConstraintActorCritic | ConstraintActorCriticH | None

    def __init__(self):
        super().__init__()

    def __set_render_mode(self, render_mode: str) -> None:
        """Set the render mode.

        Args:
            render_mode (str, optional): The render mode. Defaults to 'rgb_array'.

        Raises:
            NotImplementedError: If the render mode is not implemented.
        """
        # set the render mode
        if render_mode in ['human', 'rgb_array', 'rgb_array_list']:
            self._render_mode: str = render_mode
        else:
            raise NotImplementedError('The render mode is not implemented.')

    def __load_cfgs(self, save_dir: str) -> None:
        """Load the config from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.

        Raises:
            FileNotFoundError: If the config file is not found.
        """
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'The config file is not found in the save directory{save_dir}.',
            ) from error
        self._dict_cfgs = kwargs
        self._cfgs = Config.dict2config(kwargs)

    def __load_model_and_env(
        self,
        save_dir: str,
        model_name: str,
        env_kwargs: dict[str, Any],
        classifier_model: str = None,
    ) -> None:
        """Load the model from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.
            model_name (str): Name of the model.
            env_kwargs (dict[str, Any]): Keyword arguments for the environment.

        Raises:
            FileNotFoundError: If the model is not found.
        """
        # load the saved model
        model_path = os.path.join(save_dir, 'torch_save', model_name)
        try:
            model_params = torch.load(model_path)
        except FileNotFoundError as error:
            raise FileNotFoundError('The model is not found in the save directory.') from error

        # load the environment
        if env_kwargs['env_id'] in ['SafetyAntRun-v0', 'SafetyBallRun-v0', 'SafetyCarRun-v0', 'SafetyDroneRun-v0']:
            env_kwargs.pop('render_mode')
            env_kwargs.pop('camera_id')
            env_kwargs.pop('camera_name')
            env_kwargs.pop('width')
            env_kwargs.pop('height')
        env_kwargs.pop('asynchronous', None)
        self._env: Union[CMDP, HiddenObsNormalize] = make(**env_kwargs)

        observation_space = self._env.observation_space
        action_space = self._env.action_space
        if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
            self._safety_budget = (
                self._cfgs.algo_cfgs.safety_budget
                * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
                / (1 - self._cfgs.algo_cfgs.saute_gamma)
                / self._cfgs.algo_cfgs.max_ep_len
                * torch.ones(1)
            )
        assert isinstance(observation_space, Box), 'The observation space must be Box.'
        assert isinstance(action_space, Box), 'The action space must be Box.'

        if self._env.need_time_limit_wrapper:
            if env_kwargs['env_id'] in ['SafetyAntRun-v0', 'SafetyCarRun-v0', 'SafetyDroneRun-v0']:
                time_limit = 200
            elif env_kwargs['env_id'] == 'SafetyBallRun-v0':
                time_limit = 100
            elif 'Circle' in env_kwargs['env_id']:
                time_limit = 500
            else:
                time_limit = 1000
            # self._env = TimeLimit(self._env, device=torch.device('cpu'), time_limit=1000)
            self._env = TimeLimit(self._env, device=torch.device('cpu'), time_limit=time_limit)
        if self._cfgs['algo_cfgs']['obs_normalize']:
            obs_normalizer = Normalizer(shape=observation_space.shape, clip=5)
            obs_normalizer.load_state_dict(model_params['obs_normalizer'])
            self._env = ObsNormalize(self._env, device=torch.device('cpu'), norm=obs_normalizer)
        if ('hidden_obs_normalize' in self._cfgs['algo_cfgs']) and self._cfgs['algo_cfgs']['hidden_obs_normalize']:
            hidden_obs_normalizer = Normalizer(shape=(self._cfgs['model_cfgs']['classifier']['hidden_dim'], ), clip=5)
            hidden_obs_normalizer.load_state_dict(model_params['hidden_obs_normalizer'])
            self._env = HiddenObsNormalize(self._env, device=torch.device('cpu'),
                                           hidden_dim=self._cfgs['model_cfgs']['classifier']['hidden_dim'], norm=hidden_obs_normalizer)
        self._env = UBCRLActionScale(self._env, device=torch.device('cpu'), low=-1.0, high=1.0)

        if hasattr(self._cfgs['algo_cfgs'], 'action_repeat'):
            self._env = ActionRepeat(
                self._env,
                device=torch.device('cpu'),
                times=self._cfgs['algo_cfgs']['action_repeat'],
            )
        if hasattr(self._cfgs, 'algo') and self._cfgs['algo'] in [
            'LOOP',
            'SafeLOOP',
            'PETS',
            'CAPPETS',
            'RCEPETS',
            'CCEPETS',
        ]:
            dynamics_state_space = (
                self._env.coordinate_observation_space
                if self._env.coordinate_observation_space is not None
                else self._env.observation_space
            )
            assert self._env.action_space is not None and isinstance(
                self._env.action_space.shape,
                tuple,
            )
            if isinstance(self._env.action_space, Box):
                action_space = self._env.action_space
            else:
                raise NotImplementedError
            # if self._cfgs['algo'] in ['LOOP', 'SafeLOOP']:
            #     self._actor_critic = ConstraintActorQCritic(
            #         obs_space=dynamics_state_space,
            #         act_space=action_space,
            #         model_cfgs=self._cfgs.model_cfgs,
            #         epochs=1,
            #     )
            if self._actor_critic is not None:
                self._actor_critic.load_state_dict(model_params['actor_critic'])
                self._actor_critic.to('cpu')
            self._dynamics = EnsembleDynamicsModel(
                model_cfgs=self._cfgs.dynamics_cfgs,
                device=torch.device('cpu'),
                state_shape=dynamics_state_space.shape,
                action_shape=action_space.shape,
                actor_critic=self._actor_critic,
                rew_func=None,
                cost_func=self._env.get_cost_from_obs_tensor,
                terminal_func=None,
            )
            self._dynamics.ensemble_model.load_state_dict(model_params['dynamics'])
            self._dynamics.ensemble_model.to('cpu')
            if self._cfgs['algo'] in ['CCEPETS', 'RCEPETS', 'SafeLOOP']:
                algo_to_planner = {
                    'CCEPETS': (
                        'CCEPlanner',
                        {'cost_limit': self._cfgs['algo_cfgs']['cost_limit']},
                    ),
                    'RCEPETS': (
                        'RCEPlanner',
                        {'cost_limit': self._cfgs['algo_cfgs']['cost_limit']},
                    ),
                    'SafeLOOP': (
                        'SafeARCPlanner',
                        {
                            'cost_limit': self._cfgs['algo_cfgs']['cost_limit'],
                            'actor_critic': self._actor_critic,
                        },
                    ),
                }
            elif self._cfgs['algo'] in ['PETS', 'LOOP']:
                algo_to_planner = {
                    'PETS': ('CEMPlanner', {}),
                    'LOOP': ('ARCPlanner', {'actor_critic': self._actor_critic}),
                }
            elif self._cfgs['algo'] in ['CAPPETS']:
                lagrange: torch.nn.Parameter = torch.nn.Parameter(
                    model_params['lagrangian_multiplier'].to('cpu'),
                    requires_grad=False,
                )
                algo_to_planner = {
                    'CAPPETS': (
                        'CAPPlanner',
                        {
                            'cost_limit': self._cfgs['lagrange_cfgs']['cost_limit'],
                            'lagrange': lagrange,
                        },
                    ),
                }
            planner_name = algo_to_planner[self._cfgs['algo']][0]
            planner_special_cfgs = algo_to_planner[self._cfgs['algo']][1]
            planner_cls = globals()[f'{planner_name}']
            self._planner = planner_cls(
                dynamics=self._dynamics,
                planner_cfgs=self._cfgs.planner_cfgs,
                gamma=float(self._cfgs.algo_cfgs.gamma),
                cost_gamma=float(self._cfgs.algo_cfgs.cost_gamma),
                dynamics_state_shape=dynamics_state_space.shape,
                action_shape=action_space.shape,
                action_max=1.0,
                action_min=-1.0,
                device='cpu',
                **planner_special_cfgs,
            )

        else:
            if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                observation_space = Box(
                    low=np.hstack((observation_space.low, -np.inf)),
                    high=np.hstack((observation_space.high, np.inf)),
                    shape=(observation_space.shape[0] + 1,),
                )
            actor_type = self._cfgs['model_cfgs']['actor_type']
            pi_cfg = self._cfgs['model_cfgs']['actor']
            weight_initialization_mode = self._cfgs['model_cfgs']['weight_initialization_mode']
            # print(pi_cfg)
            if classifier_model is None:
                actor_builder = ActorBuilder(
                    obs_space=observation_space,
                    act_space=action_space,
                    hidden_sizes=pi_cfg['hidden_sizes'],
                    activation=pi_cfg['activation'],
                    weight_initialization_mode=weight_initialization_mode,
                )
            else:
                # print(pi_cfg)
                actor_builder = ActorBuilderH(
                    obs_space=observation_space,
                    # hidden_obs_size=4,
                    hidden_obs_size=self._cfgs['model_cfgs']['classifier']['hidden_dim'],
                    act_space=action_space,
                    obs_encoder_sizes=pi_cfg['obs_encoder'],
                    hidden_sizes=pi_cfg['hidden_sizes'],
                    h_encoder_sizes=pi_cfg['h_encoder'],
                    activation=pi_cfg['activation'],
                    obs_encoder_activation=pi_cfg['obs_encoder_activation'],
                    h_encoder_activation=pi_cfg['h_encoder_activation'],
                    weight_initialization_mode=weight_initialization_mode,
                    layer_norm=True,
                )
            self._actor = actor_builder.build_actor(actor_type)
            self._actor.load_state_dict(model_params['pi'])

        # load the saved classifier
        if classifier_model is None:
            self._classifier = None
        else:
            classifier_path = os.path.join(save_dir, 'torch_save', classifier_model)
            try:
                classifier_kwargs = {
                    'feature_dim': mujoco_safety_gymnasium_dict[env_kwargs['env_id']]['state_dim'] +
                                   mujoco_safety_gymnasium_dict[env_kwargs['env_id']]['action_dim'],
                    # 'nb_gru_units': 4,
                    'nb_gru_units': self._cfgs['model_cfgs']['classifier']['hidden_dim'],
                    'batch_size': self._cfgs['model_cfgs']['classifier']['batchsize'],
                    'gru_layers': self._cfgs['model_cfgs']['classifier']['stack_layer'],
                    'dropout': self._cfgs['model_cfgs']['classifier']['dropout'],
                    'mlp_arch': mujoco_safety_gymnasium_dict[env_kwargs['env_id']]['decoder_arch']}
                # if isinstance(classifier_nw_class[pt_model_type], DistributionGRU):
                #     classifier_kwargs['loc_offset'] = self._cfgs.model_cfgs.classifier.loc_offset
                #     classifier_kwargs['log_std_offset'] = self._cfgs.model_cfgs.classifier.log_std_offset

                self._classifier = DistributionGRU(**classifier_kwargs)
                self._classifier.load_state_dict(torch.load(classifier_path, weights_only=False))
            except FileNotFoundError as error:
                raise FileNotFoundError('The classifier is not found in the save directory.') from error

    def load_saved(
        self,
        save_dir: str,
        model_name: str,
        classifier_model: str = None,
        render_mode: str = 'rgb_array',
        camera_name: str | None = None,
        camera_id: int | None = None,
        width: int = 256,
        height: int = 256,
    ) -> None:
        # load the config
        self._save_dir = save_dir
        self._model_name = model_name
        self._classifier_model = classifier_model

        self.__load_cfgs(save_dir)

        self.__set_render_mode(render_mode)

        env_kwargs = {
            'env_id': self._cfgs['env_id'],
            'num_envs': 1,
            'render_mode': self._render_mode,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'width': width,
            'height': height,
        }
        if self._dict_cfgs.get('env_cfgs') is not None:
            env_kwargs.update(self._dict_cfgs['env_cfgs'])

        self.__load_model_and_env(save_dir, model_name, env_kwargs, classifier_model)

    def evaluate(
        self,
        num_episodes: int = 10,
        cost_criteria: float = 1.0,
    ) -> tuple[list[float], list[float]]:
        """Evaluate the agent for num_episodes episodes.

        Args:
            num_episodes (int, optional): The number of episodes to evaluate. Defaults to 10.
            cost_criteria (float, optional): The cost criteria. Defaults to 1.0.

        Returns:
            (episode_rewards, episode_costs): The episode rewards and costs.

        Raises:
            ValueError: If the environment and the policy are not provided or created.
        """
        if self._env is None or (self._actor is None and self._planner is None):
            raise ValueError(
                'The environment and the policy must be provided or created before evaluating the agent.',
            )

        episode_rewards: list[float] = []
        episode_costs: list[float] = []
        episode_lengths: list[float] = []

        for episode in range(num_episodes):
            obs, info = self._env.reset(classifier=self._classifier)
            if 'PPOLagLearnedH' in self._cfgs['algo']:
                hidden_obs, full_hidden_obs = info['hidden_obs'], info['full_hidden_obs']
                hidden_obs = hidden_obs.unsqueeze(dim=0)
            self._safety_obs = torch.ones(1)
            ep_ret, ep_cost, length = 0.0, 0.0, 0.0

            # hidden_obs = torch.zeros(4)
            # full_hidden_obs = torch.zeros((2, 1, 4))

            done = False
            while not done:
                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    obs = torch.cat([obs, self._safety_obs], dim=-1)
                with torch.no_grad():
                    if 'PPOLagLearnedH' in self._cfgs['algo']:
                        obs = obs.unsqueeze(dim=0)
                        orig_obs = info.get('original_obs', obs.clone()).unsqueeze(dim=0)
                        # print("Obs", obs)
                        # print("Original Obs", orig_obs)
                        # print("Hidden Obs", hidden_obs)
                        # print("Full Hidden Obs", full_hidden_obs)
                        if self._actor is not None:
                            act = self._actor.predict(
                                obs, hidden_obs,
                                deterministic=True,
                            )
                        else:
                            raise ValueError(
                                'The policy must be provided or created before evaluating the agent.',
                            )
                        # act = act.squeeze(dim=0)
                    else:
                        if self._actor is not None:
                            act = self._actor.predict(
                                obs,
                                deterministic=True,
                            )
                        elif self._planner is not None:
                            act = self._planner.output_action(
                                obs.unsqueeze(0).to('cpu'),
                            )[
                                0
                            ].squeeze(0)
                        else:
                            raise ValueError(
                                'The policy must be provided or created before evaluating the agent.',
                            )

                if 'PPOLagLearnedH' in self._cfgs['algo']:
                    # print("Obs", obs)
                    # print("Original Obs", orig_obs)
                    # print("Act", act)
                    # print("Hidden Obs", hidden_obs)
                    # print("Full Hidden Obs", full_hidden_obs)
                    obs, rew, cost, terminated, truncated, info = self._env.step(act, orig_obs, full_hidden_obs, self._classifier)
                    hidden_obs, full_hidden_obs = info['next_hidden_obs'], info['next_full_hidden_obs']
                else:
                    obs, rew, cost, terminated, truncated, info = self._env.step(act)
                #     orig_obs = info.get('original_obs', obs.clone())
                #     obs_action = torch.concat((orig_obs, act), dim=-1)
                #
                #     prob_feasible, dict_logscores_t, hidden_obs_t, full_hidden_obs = self._classifier(
                #         obs_action.unsqueeze(dim=0).unsqueeze(dim=0),
                #         torch.FloatTensor([1]),
                #         init_h=full_hidden_obs
                #     )
                #
                #     logscores_t, mean_logscores_t, var_logscores_t = (
                #         dict_logscores_t['log_scores'], dict_logscores_t['mean'], dict_logscores_t['var']
                #     )
                #
                #     # Get logC and H at the curr timestep h
                #     logscores, next_hidden_obs, mean_logscores, var_logscores = (
                #         logscores_t[-1], hidden_obs_t[-1], mean_logscores_t[-1], var_logscores_t[-1]
                #     )
                #
                #     # obs = next_obs
                #     hidden_obs = next_hidden_obs.squeeze(dim=0)

                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    self._safety_obs -= cost.unsqueeze(-1) / self._safety_budget
                    self._safety_obs /= self._cfgs.algo_cfgs.saute_gamma

                ep_ret += rew.item()
                ep_cost += (cost_criteria**length) * cost.item()
                if (
                    'EarlyTerminated' in self._cfgs['algo']
                    and ep_cost >= self._cfgs.algo_cfgs.cost_limit
                ):
                    terminated = torch.as_tensor(True)
                length += 1

                done = bool(terminated or truncated)

            episode_rewards.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(length)

            print(f'Episode {episode+1} results:')
            print(f'Episode reward: {ep_ret}')
            print(f'Episode cost: {ep_cost}')
            print(f'Episode length: {length}')

        print(self._dividing_line)
        print('Evaluation results:')
        print(f'Average episode reward: {np.mean(a=episode_rewards)}')
        print(f'Average episode cost: {np.mean(a=episode_costs)}')
        print(f'Average episode length: {np.mean(a=episode_lengths)}')

        self._env.close()
        return (
            episode_rewards,
            episode_costs,
        )
