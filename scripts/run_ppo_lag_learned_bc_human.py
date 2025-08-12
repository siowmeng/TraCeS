import sys
import warnings

import torch

import ubcrl
from ubcrl.common.experiment_grid import UBCRLExperimentGrid
from ubcrl.utils.exp_grid_tools import train


# if __name__ == '__main__':
#     env_id = sys.argv[1]
#     seed_start, seed_end = int(sys.argv[2]), int(sys.argv[3])
#     seeds = list(range(seed_start, seed_end + 1))
#
#     eg = UBCRLExperimentGrid(exp_name='PPOLagLearnedHuman_' + str(seed_start) + '-' + str(seed_end) + '_' + env_id)
#
#     eg.add('env_id', [env_id])
#
#     avaliable_gpus = list(range(torch.cuda.device_count()))
#     gpu_id = avaliable_gpus
#     # if you want to use CPU, please set gpu_id = None
#     # gpu_id = None
#
#     if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
#         warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
#         gpu_id = None
#
#     eg.add('algo', ['PPOLagLearnedHuman'])
#     # if env_id in ['SafetyHopperVelocity-v1', 'SafetyWalker2dVelocity-v1', 'SafetyAntVelocity-v1']:
#     #     eg.add('env_cfgs:terminate_when_unhealthy', False)
#     eg.add('seed', seeds)
#     # eg.add('train_cfgs:vector_env_nums', [20])
#     eg.add('train_cfgs:torch_threads', [1])
#     eg.add('env_cfgs:max_episode_steps', 200)
#     eg.add('env_cfgs:render_mode', 'rgb_array')
#
#     # total experiment num must be divided by num_pool
#     # meanwhile, users should decide this value according to their machine
#     eg.run(train, num_pool=len(seeds), gpu_id=gpu_id)

import os
os.environ["MUJOCO_GL"] = "egl"

if __name__ == '__main__':
    env_id = 'SafetyHalfCheetahVelocity-v1'

    agent = ubcrl.Agent('PPOLagLearnedBCHuman', env_id)
    agent.learn()

    # agent.plot(smooth=1)
    # agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    # agent.evaluate(num_episodes=1)

