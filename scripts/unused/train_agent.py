# classifier clamp logscore output to -20 or regularize loss
# Next phase: ICRL? Active Learning?

import argparse
import datetime
import os
import sys
from typing import Any

from gymnasium import Env
from gymnasium.envs.registration import EnvSpec

import safety_gymnasium

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from ucrl.classify.classifier import mujoco_safety_gymnasium_dict
from ucrl.common.callbacks import EvalCostCallback, RetrainClassifierCallBack
from ucrl.common.evaluation import eval_policy_cost_traj
import ucrl.common.utils as utils
from ucrl.ppo.ppoh import PPOH, PPOC, PPOL

def make_safety_gymnasium_env(
        id: str | EnvSpec,  # pylint: disable=invalid-name,redefined-builtin
        max_episode_steps: int | None = None,
        autoreset: bool | None = None,
        apply_api_compatibility: bool | None = None,
        disable_env_checker: bool | None = None,
        **kwargs: Any,) -> Env:

    safe_env = safety_gymnasium.make(id, max_episode_steps, autoreset, apply_api_compatibility, disable_env_checker, **kwargs)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safe_env)
    return env


def main(env, algo, ptfile, trainset, testset, safety_threshold, init_lambda, learn_reward_starts, seed, logdir,
         save_eval_trajs, save_retrain_trajs, deviceno):
    # np.seterr(all='raise')
    if env not in mujoco_safety_gymnasium_dict:
        print("Given env not recognized")
        sys.exit(1)

    assert algo in ['PPO-H', 'PPO-C', 'PPO-L'], "Unknown algorithm specified"

    utils.set_device(deviceno)
    if deviceno is None:
        device_used = 'auto'
    else:
        device_used = 'cuda:' + str(deviceno)

    n_envs = mujoco_safety_gymnasium_dict[env]['n_envs']
    # train_env = make_vec_env(safety_gymnasium.make, n_envs=n_envs, env_kwargs=mujoco_safety_gymnasium_dict[env]['env_kwargs'])
    env_kwargs = {'id': env}
    if mujoco_safety_gymnasium_dict[env]['env_kwargs'] is not None:
        env_kwargs = {**env_kwargs, **mujoco_safety_gymnasium_dict[env]['env_kwargs']}
    train_env = make_vec_env(make_safety_gymnasium_env, n_envs=n_envs, env_kwargs=env_kwargs)
    train_env = VecNormalize(train_env, norm_obs=mujoco_safety_gymnasium_dict[env]['norm_obs'],
                             norm_reward=mujoco_safety_gymnasium_dict[env]['norm_reward'], clip_obs=10., clip_reward=10.,
                             gamma=mujoco_safety_gymnasium_dict[env]['gamma'])

    log_datetime = datetime.datetime.now()
    log_path = os.path.join(logdir, env, log_datetime.strftime('%Y-%m-%d'), str(seed))

    if algo == 'PPO-H':

        if ptfile is None:
            print("Classifier model file must be specified when using PPO-H")
            sys.exit(1)

        agent = PPOH("MlpPolicy", train_env, ptfile, learning_rate=mujoco_safety_gymnasium_dict[env]['learning_rate'],
                     lambda_learning_rate=mujoco_safety_gymnasium_dict[env]['lambda_learning_rate'],
                     n_steps=mujoco_safety_gymnasium_dict[env]['n_steps'], batch_size=mujoco_safety_gymnasium_dict[env]['batch_size'],
                     lambda_batch_size=mujoco_safety_gymnasium_dict[env]['lambda_batch_size'], n_epochs=mujoco_safety_gymnasium_dict[env]['n_epochs'],
                     gamma=mujoco_safety_gymnasium_dict[env]['gamma'], gae_lambda=mujoco_safety_gymnasium_dict[env]['gae_lambda'],
                     clip_range=mujoco_safety_gymnasium_dict[env]['clip_range'], ent_coef=mujoco_safety_gymnasium_dict[env]['ent_coef'],
                     vf_coef=mujoco_safety_gymnasium_dict[env]['vf_coef'], log_score_vf_coef=mujoco_safety_gymnasium_dict[env]['log_score_vf_coef'],
                     lambda_coef=init_lambda, learn_reward_starts=learn_reward_starts, max_grad_norm=mujoco_safety_gymnasium_dict[env]['max_grad_norm'],
                     safety_threshold=safety_threshold, normalize_log_score=mujoco_safety_gymnasium_dict[env]['normalize_log_score'],
                     target_kl=mujoco_safety_gymnasium_dict[env]['target_kl'], seed=seed,
                     policy_kwargs=mujoco_safety_gymnasium_dict[env]['policy_kwargs'], verbose=1, device=device_used,
                     tensorboard_log=str(log_path))  # , policy_kwargs=dict(net_arch=[400, 300]))

    elif algo == 'PPO-C':

        if ptfile is None:
            print("Classifier model file must be specified when using PPO-C")
            sys.exit(1)

        agent = PPOC("MlpPolicy", train_env, ptfile, learning_rate=mujoco_safety_gymnasium_dict[env]['learning_rate'],
                     lambda_learning_rate=mujoco_safety_gymnasium_dict[env]['lambda_learning_rate'],
                     n_steps=mujoco_safety_gymnasium_dict[env]['n_steps'], batch_size=mujoco_safety_gymnasium_dict[env]['batch_size'],
                     lambda_batch_size=mujoco_safety_gymnasium_dict[env]['lambda_batch_size'], n_epochs=mujoco_safety_gymnasium_dict[env]['n_epochs'],
                     gamma=mujoco_safety_gymnasium_dict[env]['gamma'], gae_lambda=mujoco_safety_gymnasium_dict[env]['gae_lambda'],
                     clip_range=mujoco_safety_gymnasium_dict[env]['clip_range'], ent_coef=mujoco_safety_gymnasium_dict[env]['ent_coef'],
                     vf_coef=mujoco_safety_gymnasium_dict[env]['vf_coef'], neg_cost_vf_coef=mujoco_safety_gymnasium_dict[env]['log_score_vf_coef'],
                     lambda_coef=init_lambda, learn_reward_starts=learn_reward_starts, max_grad_norm=mujoco_safety_gymnasium_dict[env]['max_grad_norm'],
                     safety_threshold=safety_threshold, normalize_neg_cost=mujoco_safety_gymnasium_dict[env]['normalize_log_score'],
                     target_kl=mujoco_safety_gymnasium_dict[env]['target_kl'], seed=seed,
                     policy_kwargs=mujoco_safety_gymnasium_dict[env]['policy_kwargs'], verbose=1, device=device_used,
                     tensorboard_log=str(log_path))  # , policy_kwargs=dict(net_arch=[400, 300]))

    else:

        agent = PPOL("MlpPolicy", train_env, 25.0, learning_rate=mujoco_safety_gymnasium_dict[env]['learning_rate'],
                     lambda_learning_rate=mujoco_safety_gymnasium_dict[env]['lambda_learning_rate'],
                     n_steps=mujoco_safety_gymnasium_dict[env]['n_steps'], batch_size=mujoco_safety_gymnasium_dict[env]['batch_size'],
                     lambda_batch_size=mujoco_safety_gymnasium_dict[env]['lambda_batch_size'], n_epochs=mujoco_safety_gymnasium_dict[env]['n_epochs'],
                     gamma=mujoco_safety_gymnasium_dict[env]['gamma'], gae_lambda=mujoco_safety_gymnasium_dict[env]['gae_lambda'],
                     clip_range=mujoco_safety_gymnasium_dict[env]['clip_range'], ent_coef=mujoco_safety_gymnasium_dict[env]['ent_coef'],
                     vf_coef=mujoco_safety_gymnasium_dict[env]['vf_coef'], neg_cost_vf_coef=mujoco_safety_gymnasium_dict[env]['log_score_vf_coef'],
                     lambda_coef=init_lambda, learn_reward_starts=learn_reward_starts, max_grad_norm=mujoco_safety_gymnasium_dict[env]['max_grad_norm'],
                     safety_threshold=safety_threshold, normalize_neg_cost=mujoco_safety_gymnasium_dict[env]['normalize_log_score'],
                     target_kl=mujoco_safety_gymnasium_dict[env]['target_kl'], seed=seed,
                     policy_kwargs=mujoco_safety_gymnasium_dict[env]['policy_kwargs'], verbose=1, device=device_used,
                     tensorboard_log=str(log_path))  # , policy_kwargs=dict(net_arch=[400,

    n_eval_envs = 100
    eval_env = make_vec_env(make_safety_gymnasium_env, n_envs=n_eval_envs, env_kwargs=env_kwargs)
    eval_env = VecNormalize(eval_env, norm_obs=mujoco_safety_gymnasium_dict[env]['norm_obs'], norm_reward=False,
                            clip_obs=10., gamma=mujoco_safety_gymnasium_dict[env]['gamma'])

    ec_callback_kwargs = {'traj_path': os.path.join(str(log_path), 'eval_trajs') if save_eval_trajs else None,
                          'n_eval_episodes': 100,
                          'eval_freq': mujoco_safety_gymnasium_dict[env]['eval_freq'],
                          # 'eval_freq': 50_000,
                          # 'eval_freq': 20000,
                          'log_path': os.path.join(str(log_path), 'results'),
                          'deterministic': False,
                          'render': False}
    eval_callback = EvalCostCallback(eval_env, **ec_callback_kwargs)

    if algo == 'PPO-L':
        callbacks = eval_callback
    else:
        retrain_callback_kwargs = {'traj_path': os.path.join(str(log_path), 'retrain_trajs') if save_retrain_trajs else None,
                                   'retrain_pt_path': os.path.join(str(log_path), 'retrain_pts') if save_retrain_trajs else None,
                                   # 'n_eval_episodes': 500,
                                   'n_eval_episodes': mujoco_safety_gymnasium_dict[env]['retrain_eval_episodes'],
                                   # 'eval_freq': 250_000,
                                   'eval_freq': mujoco_safety_gymnasium_dict[env]['retrain_eval_freq'],
                                   'log_path': os.path.join(str(log_path), 'retrain'),
                                   'deterministic': False,
                                   'render': False}
        retrain_callback = RetrainClassifierCallBack(eval_env, trainset, testset, **retrain_callback_kwargs)
        callbacks = [eval_callback, retrain_callback]

    # learn_steps = bullet_dict[env]['learn_steps']
    agent.learn(total_timesteps=10_000_000, callback=callbacks)
    # agent.learn(total_timesteps=5_000_000, callback=[eval_callback, retrain_callback])

    train_env.training = False
    eval_env.training = False
    # reward normalization is not needed at test time
    train_env.norm_reward = False
    eval_env.training = False

    (mean_reward, std_reward, mean_cost, std_cost, mean_safe_prop, std_safe_prop, mean_step_safe_prop,
     std_step_safe_prop, mean_log_score, std_log_score, _ ) = eval_policy_cost_traj(agent,
                                                                                    eval_env,
                                                                                    n_eval_episodes=100,
                                                                                    deterministic=False)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean cost = {mean_cost:.2f} +/- {std_cost:.2f}")
    print(f"Mean safety proportion = {mean_safe_prop:.2f} +/- {std_safe_prop:.2f}")
    print(f"Mean step safety proportion = {mean_step_safe_prop:.2f} +/- {std_step_safe_prop:.2f}")
    print(f"Mean logC = {mean_log_score:.2f} +/- {std_log_score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [SafetyHalfCheetahVelocity-v1, SafetyHopperVelocity-v1, SafetyWalker2dVelocity-v1]')
    parser.add_argument('-algo', type=str, required=True,
                        help='Algorithm for agent training: select from [PPO-H, PPO-C, PPO-L]')
    parser.add_argument('-ptfile', type=str, default=None,
                        help='Classifier PyTorch Model File (required for PPO-H & PPO-C)')
    parser.add_argument('-trainset', type=str, default=None,
                        help='Train dataset file for classifier training')
    parser.add_argument('-testset', type=str, default=None,
                        help='Test dataset file for classifier training')
    parser.add_argument('-safe', '--safety_threshold', type=float, default=0.95,
                        help='Trajectory safety threshold for SafeSAC-H (default: 0.95), NA for PPO-C'
                        )
    parser.add_argument('-init_lambda', type=str, default='auto',
                        help='Initial coefficient of log P(C = 1 | s, h, a) in Q(s, h, a) - required for SafeSAC-H')
    parser.add_argument('-learn_reward_starts', type=int, default=0,
                        help='Timestep to start learning reward (default: 0)'
                        )
    parser.add_argument('-seed', type=int, default=1,
                        help='Random seed to be used (default: 1)'
                        )
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)'
                        )
    parser.add_argument('-save_eval_trajs', action='store_true',
                        help='Save evaluation trajectories in logging directory'
                        )
    parser.add_argument('-save_retrain_trajs', action='store_true',
                        help='Save retraining trajectories in logging directory'
                        )
    parser.add_argument('-deviceno', type=int, default=None,
                        help='GPU device number to use (default: auto)'
                        )
    args = parser.parse_args()

    main(args.env, args.algo, args.ptfile, args.trainset, args.testset, args.safety_threshold, args.init_lambda,
         args.learn_reward_starts, args.seed, args.logdir, args.save_eval_trajs, args.save_retrain_trajs, args.deviceno)

    print("Training completed")
    sys.exit(0)
