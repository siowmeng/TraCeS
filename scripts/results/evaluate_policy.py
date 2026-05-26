import argparse
import os

import numpy as np

from traces.evaluator import TraCeSEvaluator

import traces.common.utils as utils
utils.set_device('cpu')

# env_id = 'SafetyAntVelocity-v1'
# env_id = 'SafetyDroneRun-v0'
# env_id = 'SafetyPointCircle1-v0'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id', help='Environment name to evaluate.')
    parser.add_argument('expgrid_dir', help='Experiment grid directory to evaluate.')
    parser.add_argument('--traces', action='store_true', help='Load the TraCeS classifier checkpoint with each policy checkpoint.')
    args = parser.parse_args()

    env_id = args.env_id
    expgrid_dir = args.expgrid_dir
    ppoh_model = args.traces
    last_epoch = 1000 if env_id in ['SafetyAntRun-v0', 'SafetyBallRun-v0', 'SafetyCarRun-v0', 'SafetyDroneRun-v0'] else 500

    evaluator = TraCeSEvaluator()

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
