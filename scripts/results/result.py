import glob
import os
import numpy as np
import pandas as pd

list_dir = [('PPO-L', '/SSD2/siowmeng/icml25_results/ppol_baseline/exp-x/'),
            ('PPO-C', '/SSD2/siowmeng/icml25_results/ppoc_baseline/exp-x/'),
            ('PPO-H', '/SSD2/siowmeng/icml25_results/ppoh/old_largeH_policy_cost_critic/exp-x/'),
            ]

# ppol_dir = '/SSD2/siowmeng/ppol_baseline/exp-x/'
# ppoh_dir = '/SSD2/siowmeng/ppoh_results/exp-x/'
envs = ['SafetyAntVelocity', 'SafetyHalfCheetahVelocity', 'SafetyHopperVelocity', 'SafetySwimmerVelocity', 'SafetyWalker2dVelocity',
        # 'SafetyPointCircle1']
        'SafetyPointCircle1', 'SafetyPointCircle2', 'SafetyCarCircle1', 'SafetyCarCircle2']


for alg, directory in list_dir:
    print("Algo", alg)
    for env in envs:
        print("Env:", env)
        env_rewards, env_costs, env_retrain_trajs = [], [], []

        for basepath in glob.glob(os.path.join(directory, '*' + env + '*')):
            result_file = os.path.join(basepath, 'exp-x-results.txt')
            rewards = [float(line.split('reward:')[1].split(',')[0]) for line in open(result_file)]
            costs = [float(line.split('cost:')[1].split(',')[0]) for line in open(result_file)]
            env_rewards += rewards
            env_costs += costs

        if alg != 'PPO-L':
            for progressfile in glob.glob(os.path.join(directory, '*' + env + '*', '*', '*', 'seed-*', 'progress.csv')):
                progress_df = pd.read_csv(progressfile)
                env_retrain_trajs.append(progress_df['Classifier/NumRetrainTrajs'].tolist()[-1])

        print("Reward Mean:", np.mean(env_rewards), ", Reward Stdev:", np.std(env_rewards))
        print("Cost Mean:", np.mean(env_costs), ", Cost Stdev:", np.std(env_costs))
        if len(env_retrain_trajs) > 0:
            print("Traj Mean:", np.mean(env_retrain_trajs), ", Traj Stdev:", np.std(env_retrain_trajs))
        print()

