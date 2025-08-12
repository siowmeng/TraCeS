import os
import sys

import numpy as np

from ubcrl.evaluator import UBCRLEvaluator

import ubcrl.common.utils as utils
utils.set_device('cpu')

# env_id = 'SafetyAntVelocity-v1'
# env_id = 'SafetyDroneRun-v0'
# env_id = 'SafetyPointCircle1-v0'

# expgrid_dir = '/SSD2/siowmeng/icml25_results/bulletgym/ppol/exp-x'
# expgrid_dir = '/SSD2/siowmeng/icml25_results/bulletgym/ppoc/exp-x'
# expgrid_dir = '/SSD2/siowmeng/icml25_results/bulletgym/ppoh/exp-x'

# expgrid_dir = '/SSD2/siowmeng/icml25_results/ppol_baseline/exp-x'
# expgrid_dir = '/SSD2/siowmeng/icml25_results/ppoc_baseline/exp-x'
# expgrid_dir = '/SSD2/siowmeng/icml25_results/ppoh_all_pick/exp-x'
# expgrid_dir = '/SSD2/siowmeng/neurips25_results/ppoh_mujoco_active_retrain/exp-x'
# expgrid_dir = '/SSD2/siowmeng/neurips25_results/ppoh_small_data'
# expgrid_dir = '/SSD2/siowmeng/neurips25_results/ppoh_selecttraj_0.1_lstall/exp-x'
# expgrid_dir = '/SSD2/siowmeng/neurips25_results/ppoh_random_0.1/exp-x'
# expgrid_dir = '/SSD2/siowmeng/neurips25_results/ppoh_selecttraj_0.1_lstall_extra/exp-x'
expgrid_dir = '/SSD2/siowmeng/neurips25_results/ablation_results/ablation_d/ppoh_d_0.8/exp-x'
# expgrid_dir = '/SSD2/siowmeng/neurips25_results/ablation_results/ablation_noise/ppoh_noise0.2/exp-x'

# ppoh_model = True
# last_epoch = 1000
# last_epoch = 500

if __name__ == '__main__':
    # env_id = 'SafetyPointCircle1-v0'
    env_id = sys.argv[1]
    # exp_dir = sys.argv[2]

    ppoh_model = 'ppoh' in expgrid_dir
    last_epoch = 1000 if env_id in ['SafetyAntRun-v0', 'SafetyBallRun-v0', 'SafetyCarRun-v0', 'SafetyDroneRun-v0'] else 500

    evaluator = UBCRLEvaluator()

    lst_mean_rewards, lst_mean_costs = [], []
    exp_dir = os.scandir(expgrid_dir)
    for set_of_exps in exp_dir:
        if set_of_exps.is_dir() and env_id in set_of_exps.name:
            param_dir = os.scandir(set_of_exps)
            for set_of_params in param_dir:
                if set_of_params.is_dir():
                    exp_dir = os.scandir(set_of_params)
                    for single_exp in exp_dir:
                        if single_exp.is_dir():
                            seed_dir = os.scandir(single_exp)
                            for single_seed in seed_dir:
                                if single_seed.is_dir():
                                    # print(os.path.join(single_seed.path, 'torch_save', 'epoch-' + str(last_epoch) + '.pt'))
                                    if ppoh_model:
                                        evaluator.load_saved(
                                            save_dir=single_seed.path,
                                            model_name='epoch-' + str(last_epoch) + '.pt',
                                            classifier_model='classifier-' + str(last_epoch) + '.pt',
                                        )
                                        # Classifier loading, different evaluate
                                    else:
                                        evaluator.load_saved(
                                            save_dir=single_seed.path,
                                            model_name='epoch-' + str(last_epoch) + '.pt',
                                        )
                                    episode_rewards, episode_costs = evaluator.evaluate(
                                        num_episodes=100,
                                    )
                                    lst_mean_rewards.append(np.mean(episode_rewards))
                                    lst_mean_costs.append(np.mean(episode_costs))
                            seed_dir.close()
                    exp_dir.close()
            param_dir.close()
    exp_dir.close()
    print("Path:", expgrid_dir)
    print("Env:", env_id)
    print("Average Rewards (Mean):", np.mean(lst_mean_rewards), "Average Rewards (Stdev):", np.std(lst_mean_rewards))
    print("Average Costs (Mean):", np.mean(lst_mean_costs), "Average Costs (Stdev):", np.std(lst_mean_costs))
