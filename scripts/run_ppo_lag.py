import sys
import warnings

import torch

# from omnisafe.common.experiment_grid import ExperimentGrid
# from omnisafe.utils.exp_grid_tools import train
from ubcrl.common.experiment_grid import UBCRLExperimentGrid
from ubcrl.utils.exp_grid_tools import train

if __name__ == '__main__':
    env_id = sys.argv[1]
    seed_start, seed_end = int(sys.argv[2]), int(sys.argv[3])
    seeds = list(range(seed_start, seed_end + 1))

    eg = UBCRLExperimentGrid(exp_name='PPOLag_' + str(seed_start) + '-' + str(seed_end) + '_' + env_id)

    eg.add('env_id', [env_id])

    avaliable_gpus = list(range(torch.cuda.device_count()))  # [::-1]
    gpu_id = avaliable_gpus
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    eg.add('algo', ['PPOLag'])
    # if env_id in ['SafetyHopperVelocity-v1', 'SafetyWalker2dVelocity-v1', 'SafetyAntVelocity-v1']:
    #     eg.add('env_cfgs:terminate_when_unhealthy', False)
    eg.add('seed', seeds)
    # eg.add('train_cfgs:vector_env_nums', [20])
    eg.add('train_cfgs:torch_threads', [1])

    # total experiment num must be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=len(seeds), gpu_id=gpu_id)

    # just fill in the name of the parameter of which value you want to compare.
    # then you can specify the value of the parameter you want to compare,
    # or you can just specify how many values you want to compare in single graph at most,
    # and the function will automatically generate all possible combinations of the graph.
    # but the two mode can not be used at the same time.
    # eg.analyze(parameter='env_id', values=None, compare_num=6, cost_limit=25)
    eg.analyze(parameter='seed', values=None, compare_num=len(seeds), cost_limit=25, show_image=False)
    # eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    # eg.evaluate(num_episodes=100)
