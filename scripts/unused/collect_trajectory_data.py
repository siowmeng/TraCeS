import os
import sys
from typing import List

import numpy as np
import torch
from gymnasium.spaces import Box
from tqdm import tqdm

from omnisafe.common.offline.data_collector import OfflineAgent, OfflineDataCollector
from omnisafe.envs.core import make
from omnisafe.envs.wrapper import ActionScale


class MuJoCoOfflineDataCollector(OfflineDataCollector):

    def __init__(self, size: int, env_name: str, horizon: int) -> None:
        """Initialize the data collector.

        Args:
            size (int): The total number of data to collect.
            env_name (str): The name of the environment.
        """
        self._size = size
        self._env_name = env_name
        self._horizon = horizon

        # make a env, get the observation space and action space
        if (self._env_name != 'SafetyHalfCheetahVelocity-v1') and (self._env_name != 'SafetySwimmerVelocity-v1'):
            self._env = make(env_name, terminate_when_unhealthy=False)
        else:
            self._env = make(env_name)
        self._obs_space = self._env.observation_space
        self._action_space = self._env.action_space

        self._env = ActionScale(self._env, device=torch.device('cpu'), high=1.0, low=-1.0)

        if not isinstance(self._obs_space, Box):
            raise NotImplementedError('Only support Box observation space for now.')
        if not isinstance(self._action_space, Box):
            raise NotImplementedError('Only support Box action space for now.')

        # create a buffer to store the data
        # self._obs = np.zeros((size, *self._obs_space.shape), dtype=np.float32)
        # self._action = np.zeros((size, *self._action_space.shape), dtype=np.float32)
        # self._reward = np.zeros((size, 1), dtype=np.float32)
        # self._cost = np.zeros((size, 1), dtype=np.float32)
        # self._next_obs = np.zeros((size, *self._obs_space.shape), dtype=np.float32)
        # self._done = np.zeros((size, 1), dtype=np.float32)
        self._init_np_data()

        self.agents: List[OfflineAgent] = []

    def _init_np_data(self):
        self._obs = np.zeros((self._horizon, *self._obs_space.shape), dtype=np.float32)
        self._action = np.zeros((self._horizon, *self._action_space.shape), dtype=np.float32)
        self._reward = np.zeros((self._horizon, 1), dtype=np.float32)
        self._cost = np.zeros((self._horizon, 1), dtype=np.float32)
        self._next_obs = np.zeros((self._horizon, *self._obs_space.shape), dtype=np.float32)
        self._done = np.zeros((self._horizon, 1), dtype=np.float32)

    def collect(self, save_dir: str) -> None:
        """Collect data from the registered agents.

        Args:
            save_dir (str): The directory to save the collected data.
        """
        # check each agent's size
        total_size = 0
        for agent in self.agents:
            assert agent.size <= self._size, f'Agent {agent} size is larger than collector size.'
            total_size += agent.size
        assert total_size == self._size, 'Sum of agent size is not equal to collector size.'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # collect data
        traj_num = 0
        progress_bar = tqdm(total=self._size, desc='Collecting data...')

        for agent in self.agents:
            ep_ret, ep_cost, ep_len, single_ep_len, episode_num = 0.0, 0.0, 0.0, 0.0, 0.0
            ptx = 0
            self._init_np_data()

            obs, _ = self._env.reset()
            for _ in range(agent.size):
                action = agent.agent_step(obs)
                next_obs, reward, cost, terminate, truncated, _ = self._env.step(action)
                done = terminate or truncated

                self._obs[ptx] = obs.detach().numpy()
                self._action[ptx] = action.detach().numpy()
                self._reward[ptx] = reward.detach().numpy()
                self._cost[ptx] = cost.detach().numpy()
                self._next_obs[ptx] = next_obs.detach().numpy()
                self._done[ptx] = done.detach().numpy()

                ep_ret += reward.item()
                ep_cost += cost.item()
                ep_len += 1
                single_ep_len += 1

                ptx += 1
                obs = next_obs
                if done:
                    obs, _ = self._env.reset()
                    episode_num += 1
                    progress_bar.update(single_ep_len)
                    single_ep_len = 0

                    # save data
                    save_path = os.path.join(save_dir, f'{self._env_name}_data_{traj_num}.npz')
                    if os.path.exists(save_path):
                        print(f'Warning: {save_path} already exists.')
                        print(f'Warning: {save_path} will be overwritten.')
                    np.savez(
                        # os.path.join(save_dir, f'{self._env_name}_data.npz'),
                        save_path,
                        obs=self._obs,
                        action=self._action,
                        reward=self._reward,
                        cost=self._cost,
                        next_obs=self._next_obs,
                        done=self._done,
                    )

                    traj_num += 1
                    ptx = 0
                    self._init_np_data()

            print(f'Agent {agent} collected {agent.size} data points.')
            print(f'Average return: {ep_ret / episode_num}')
            print(f'Average cost: {ep_cost / episode_num}')
            print(f'Average episode length: {ep_len / episode_num}')
            print()


if __name__ == '__main__':
    result_path = sys.argv[1]
    # total_steps_per_agent = int(sys.argv[2])
    steps_per_episode = int(sys.argv[2])
    total_episodes_per_agent = int(sys.argv[3])
    save_path = sys.argv[4]

    total_steps_per_agent = total_episodes_per_agent * steps_per_episode

    env_name = None
    for envname in ['SafetyHopperVelocity-v1', 'SafetyHalfCheetahVelocity-v1', 'SafetyWalker2dVelocity-v1']:
        if envname in result_path:
            env_name = envname

    assert env_name is not None, "Environment name not found"

    agents = []
    for agent_folder in os.listdir(result_path):
        agent_path = os.path.join(result_path, agent_folder)
        for model_name in os.listdir(os.path.join(agent_path, "torch_save")):
            agents.append((agent_path, model_name, total_steps_per_agent))

    size = len(agents) * total_steps_per_agent

    col = MuJoCoOfflineDataCollector(size, env_name, steps_per_episode)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_path)
