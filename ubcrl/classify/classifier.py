#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Callable, List, Tuple, Union
import h5py
import os

import pandas as pd
from tqdm import tqdm

import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import LogNormal  # , Normal
from torch.utils.data import Dataset
from datetime import datetime

import ubcrl.common.utils as utils
from ubcrl.safety.label import LabeledNPData, mujoco_markov_safety_dict

LOC_MAX = 3
LOC_MIN = -20
LOG_STD_MAX = 2
LOG_STD_MIN = -20
MIN_LOGSCORE = -7

mujoco_safety_gymnasium_dict = {
    'SafetyPointButton1-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 76, 'action_dim': 2,
                              'horizon': 1000, 'gamma': 0.99,
                              'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                              'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                              'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
                              'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                              'env_kwargs': None,
                              'loc_offset': 4.0, 'log_std_offset': 0.0,
                              'policy_kwargs': {'log_std_init': -2,
                                                'ortho_init': False,
                                                'activation_fn': nn.ReLU,
                                                'net_arch': {'pi': [128, 128],
                                                             'vf': [128, 128],
                                                             'log_score_vf': [128, 128]
                                                             },
                                                # 'features_extractor_class': HiddenObsExtractor,
                                                'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                              # 'hidden_obs_encoder_arch': [],
                                                                              # 'hidden_obs_encoder_arch': [32, 16],
                                                                              },
                                                # 'net_arch': [400, 300]
                                                },
                              'decoder_arch': [128, 128],
                              'eval_freq': 50_000,
                              'retrain_eval_episodes': 500,
                              'retrain_eval_freq': 250_000,
                              },
    'SafetyPointButton2-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 76, 'action_dim': 2,
                              'horizon': 1000, 'gamma': 0.99,
                              'n_steps': 10000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
                              'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                              'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
                              'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                              'env_kwargs': None,
                              'loc_offset': 4.0, 'log_std_offset': 0.0,
                              'policy_kwargs': {'log_std_init': -2,
                                                'ortho_init': False,
                                                'activation_fn': nn.ReLU,
                                                'net_arch': {'pi': [128, 128],
                                                             'vf': [128, 128],
                                                             'log_score_vf': [128, 128]
                                                             },
                                                # 'features_extractor_class': HiddenObsExtractor,
                                                'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                              # 'hidden_obs_encoder_arch': [],
                                                                              # 'hidden_obs_encoder_arch': [32, 16],
                                                                              },
                                                # 'net_arch': [400, 300]
                                                },
                              'decoder_arch': [128, 128],
                              'eval_freq': 50_000,
                              'retrain_eval_episodes': 500,
                              'retrain_eval_freq': 250_000,
                              },
    'SafetyPointCircle1-v0': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 28, 'action_dim': 2,
                              'horizon': 500, 'gamma': 0.99,
                              'n_steps': 1000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
                              'clip_range': 0.2, 'n_epochs': 40, 'gae_lambda': 0.95, 'max_grad_norm': 40.0,
                              'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,
                              'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                              'env_kwargs': None,
                              'loc_offset': 0.0, 'log_std_offset': 0.0,
                              'policy_kwargs': {'log_std_init': -2,
                                                'ortho_init': False,
                                                'activation_fn': nn.Tanh,
                                                'net_arch': {'pi': [64, 64],
                                                             'vf': [64, 64],
                                                             'log_score_vf': [64, 64]
                                                             },
                                                # 'features_extractor_class': HiddenObsExtractor,
                                                'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                              # 'hidden_obs_encoder_arch': [],
                                                                              # 'hidden_obs_encoder_arch': [32, 16],
                                                                              },
                                                # 'net_arch': [400, 300]
                                                },
                              'decoder_arch': [64, 64],
                              'eval_freq': 2_500,
                              'retrain_eval_episodes': 1000,
                              'retrain_eval_freq': 25_000,
                              },
    'SafetyPointCircle2-v0': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 28, 'action_dim': 2,
                              'horizon': 500, 'gamma': 0.99,
                              'n_steps': 1000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
                              'clip_range': 0.2, 'n_epochs': 40, 'gae_lambda': 0.95, 'max_grad_norm': 40.0,
                              'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,
                              'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                              'env_kwargs': None,
                              'loc_offset': 0.0, 'log_std_offset': 0.0,
                              'policy_kwargs': {'log_std_init': -2,
                                                'ortho_init': False,
                                                'activation_fn': nn.Tanh,
                                                'net_arch': {'pi': [64, 64],
                                                             'vf': [64, 64],
                                                             'log_score_vf': [64, 64]
                                                             },
                                                # 'features_extractor_class': HiddenObsExtractor,
                                                'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                              # 'hidden_obs_encoder_arch': [],
                                                                              # 'hidden_obs_encoder_arch': [32, 16],
                                                                              },
                                                # 'net_arch': [400, 300]
                                                },
                              'decoder_arch': [64, 64],
                              'eval_freq': 2_500,
                              'retrain_eval_episodes': 1000,
                              'retrain_eval_freq': 25_000,
                              },
    'SafetyPointGoal1-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 60, 'action_dim': 2,
                            'horizon': 1000, 'gamma': 0.99,
                            'n_steps': 1000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                            'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                            'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.02,
                            'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                            'env_kwargs': None,
                            'loc_offset': 4.0, 'log_std_offset': 0.0,
                            'policy_kwargs': {'log_std_init': -2,
                                              'ortho_init': False,
                                              'activation_fn': nn.ReLU,
                                              'net_arch': {'pi': [128, 128],
                                                           'vf': [128, 128],
                                                           'log_score_vf': [128, 128]
                                                           },
                                              # 'features_extractor_class': HiddenObsExtractor,
                                              'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [32, 16],
                                                                            },
                                              # 'net_arch': [400, 300]
                                              },
                            'decoder_arch': [128, 128],
                            'eval_freq': 5_000,
                            'retrain_eval_episodes': 500,
                            'retrain_eval_freq': 25_000,
                            },
    'SafetyPointGoal2-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 60, 'action_dim': 2,
                            'horizon': 1000, 'gamma': 0.99,
                            'n_steps': 1000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                            'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                            'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.02,
                            'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                            'env_kwargs': None,
                            'loc_offset': 4.0, 'log_std_offset': 0.0,
                            'policy_kwargs': {'log_std_init': -2,
                                              'ortho_init': False,
                                              'activation_fn': nn.ReLU,
                                              'net_arch': {'pi': [128, 128],
                                                           'vf': [128, 128],
                                                           'log_score_vf': [128, 128]
                                                           },
                                              # 'features_extractor_class': HiddenObsExtractor,
                                              'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [32, 16],
                                                                            },
                                              # 'net_arch': [400, 300]
                                              },
                            'decoder_arch': [128, 128],
                            'eval_freq': 5_000,
                            'retrain_eval_episodes': 500,
                            'retrain_eval_freq': 25_000,
                            },
    'SafetyPointPush1-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 76, 'action_dim': 2,
                            'horizon': 1000, 'gamma': 0.99,
                            'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                            'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                            'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
                            'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                            'env_kwargs': None,
                            'loc_offset': 4.0, 'log_std_offset': 0.0,
                            'policy_kwargs': {'log_std_init': -2,
                                              'ortho_init': False,
                                              'activation_fn': nn.ReLU,
                                              'net_arch': {'pi': [128, 128],
                                                           'vf': [128, 128],
                                                           'log_score_vf': [128, 128]
                                                           },
                                              # 'features_extractor_class': HiddenObsExtractor,
                                              'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [32, 16],
                                                                            },
                                              # 'net_arch': [400, 300]
                                              },
                            'decoder_arch': [128, 128],
                            'eval_freq': 50_000,
                            'retrain_eval_episodes': 500,
                            'retrain_eval_freq': 250_000,
                            },
    'SafetyPointPush2-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 76, 'action_dim': 2,
                            'horizon': 1000, 'gamma': 0.99,
                            'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                            'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                            'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
                            'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                            'env_kwargs': None,
                            'loc_offset': 4.0, 'log_std_offset': 0.0,
                            'policy_kwargs': {'log_std_init': -2,
                                              'ortho_init': False,
                                              'activation_fn': nn.ReLU,
                                              'net_arch': {'pi': [128, 128],
                                                           'vf': [128, 128],
                                                           'log_score_vf': [128, 128]
                                                           },
                                              # 'features_extractor_class': HiddenObsExtractor,
                                              'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [32, 16],
                                                                            },
                                              # 'net_arch': [400, 300]
                                              },
                            'decoder_arch': [128, 128],
                            'eval_freq': 50_000,
                            'retrain_eval_episodes': 500,
                            'retrain_eval_freq': 250_000,
                            },
    'SafetyCarButton1-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 88, 'action_dim': 2,
                            'horizon': 1000, 'gamma': 0.99,
                            'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                            'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                            'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
                            'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                            'env_kwargs': None,
                            'loc_offset': 4.0, 'log_std_offset': 0.0,
                            'policy_kwargs': {'log_std_init': -2,
                                              'ortho_init': False,
                                              'activation_fn': nn.ReLU,
                                              'net_arch': {'pi': [128, 128],
                                                           'vf': [128, 128],
                                                           'log_score_vf': [128, 128]
                                                           },
                                              # 'features_extractor_class': HiddenObsExtractor,
                                              'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [32, 16],
                                                                            },
                                              # 'net_arch': [400, 300]
                                              },
                            'decoder_arch': [128, 128],
                            'eval_freq': 50_000,
                            'retrain_eval_episodes': 500,
                            'retrain_eval_freq': 250_000,
                            },
    'SafetyCarButton2-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 88, 'action_dim': 2,
                            'horizon': 1000, 'gamma': 0.99,
                            'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                            'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                            'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
                            'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                            'env_kwargs': None,
                            'loc_offset': 4.0, 'log_std_offset': 0.0,
                            'policy_kwargs': {'log_std_init': -2,
                                              'ortho_init': False,
                                              'activation_fn': nn.ReLU,
                                              'net_arch': {'pi': [128, 128],
                                                           'vf': [128, 128],
                                                           'log_score_vf': [128, 128]
                                                           },
                                              # 'features_extractor_class': HiddenObsExtractor,
                                              'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [32, 16],
                                                                            },
                                              # 'net_arch': [400, 300]
                                              },
                            'decoder_arch': [128, 128],
                            'eval_freq': 50_000,
                            'retrain_eval_episodes': 500,
                            'retrain_eval_freq': 250_000,
                            },
    'SafetyCarCircle1-v0': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 40, 'action_dim': 2,
                            'horizon': 500, 'gamma': 0.99,
                            'n_steps': 1000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
                            'clip_range': 0.2, 'n_epochs': 40, 'gae_lambda': 0.95, 'max_grad_norm': 40.0,
                            'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,
                            'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                            'env_kwargs': None,
                            'loc_offset': 0.0, 'log_std_offset': 0.0,
                            'policy_kwargs': {'log_std_init': -2,
                                              'ortho_init': False,
                                              'activation_fn': nn.Tanh,
                                              'net_arch': {'pi': [64, 64],
                                                           'vf': [64, 64],
                                                           'log_score_vf': [64, 64]
                                                           },
                                              # 'features_extractor_class': HiddenObsExtractor,
                                              'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [32, 16],
                                                                            },
                                              # 'net_arch': [400, 300]
                                              },
                            'decoder_arch': [64, 64],
                            'eval_freq': 2_500,
                            'retrain_eval_episodes': 1000,
                            'retrain_eval_freq': 25_000,
                            },
    'SafetyCarCircle2-v0': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 40, 'action_dim': 2,
                            'horizon': 500, 'gamma': 0.99,
                            'n_steps': 1000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
                            'clip_range': 0.2, 'n_epochs': 40, 'gae_lambda': 0.95, 'max_grad_norm': 40.0,
                            'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,
                            'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                            'env_kwargs': None,
                            'loc_offset': 0.0, 'log_std_offset': 0.0,
                            'policy_kwargs': {'log_std_init': -2,
                                              'ortho_init': False,
                                              'activation_fn': nn.Tanh,
                                              'net_arch': {'pi': [64, 64],
                                                           'vf': [64, 64],
                                                           'log_score_vf': [64, 64]
                                                           },
                                              # 'features_extractor_class': HiddenObsExtractor,
                                              'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [],
                                                                            # 'hidden_obs_encoder_arch': [32, 16],
                                                                            },
                                              # 'net_arch': [400, 300]
                                              },
                            'decoder_arch': [64, 64],
                            'eval_freq': 2_500,
                            'retrain_eval_episodes': 1000,
                            'retrain_eval_freq': 25_000,
                            },
    'SafetyCarGoal1-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 72, 'action_dim': 2,
                          'horizon': 1000, 'gamma': 0.99,
                          'n_steps': 1000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                          'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                          'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.02,
                          'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                          'env_kwargs': None,
                          'loc_offset': 4.0, 'log_std_offset': 0.0,
                          'policy_kwargs': {'log_std_init': -2,
                                            'ortho_init': False,
                                            'activation_fn': nn.ReLU,
                                            'net_arch': {'pi': [128, 128],
                                                         'vf': [128, 128],
                                                         'log_score_vf': [128, 128]
                                                         },
                                            # 'features_extractor_class': HiddenObsExtractor,
                                            'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                          # 'hidden_obs_encoder_arch': [],
                                                                          # 'hidden_obs_encoder_arch': [32, 16],
                                                                          },
                                            # 'net_arch': [400, 300]
                                            },
                          'decoder_arch': [128, 128],
                          'eval_freq': 5_000,
                          'retrain_eval_episodes': 500,
                          'retrain_eval_freq': 25_000,
                          },
    'SafetyCarGoal2-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 72, 'action_dim': 2,
                          'horizon': 1000, 'gamma': 0.99,
                          'n_steps': 1000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                          'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                          'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.02,
                          'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                          'env_kwargs': None,
                          'loc_offset': 4.0, 'log_std_offset': 0.0,
                          'policy_kwargs': {'log_std_init': -2,
                                            'ortho_init': False,
                                            'activation_fn': nn.ReLU,
                                            'net_arch': {'pi': [128, 128],
                                                         'vf': [128, 128],
                                                         'log_score_vf': [128, 128]
                                                         },
                                            # 'features_extractor_class': HiddenObsExtractor,
                                            'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                          # 'hidden_obs_encoder_arch': [],
                                                                          # 'hidden_obs_encoder_arch': [32, 16],
                                                                          },
                                            # 'net_arch': [400, 300]
                                            },
                          'decoder_arch': [128, 128],
                          'eval_freq': 5_000,
                          'retrain_eval_episodes': 500,
                          'retrain_eval_freq': 25_000,
                          },
    'SafetyCarPush1-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 88, 'action_dim': 2,
                          'horizon': 1000, 'gamma': 0.99,
                          'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                          'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                          'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
                          'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                          'env_kwargs': None,
                          'loc_offset': 4.0, 'log_std_offset': 0.0,
                          'policy_kwargs': {'log_std_init': -2,
                                            'ortho_init': False,
                                            'activation_fn': nn.ReLU,
                                            'net_arch': {'pi': [128, 128],
                                                         'vf': [128, 128],
                                                         'log_score_vf': [128, 128]
                                                         },
                                            # 'features_extractor_class': HiddenObsExtractor,
                                            'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                          # 'hidden_obs_encoder_arch': [],
                                                                          # 'hidden_obs_encoder_arch': [32, 16],
                                                                          },
                                            # 'net_arch': [400, 300]
                                            },
                          'decoder_arch': [128, 128],
                          'eval_freq': 50_000,
                          'retrain_eval_episodes': 500,
                          'retrain_eval_freq': 250_000,
                          },
    'SafetyCarPush2-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 88, 'action_dim': 2,
                          'horizon': 1000, 'gamma': 0.99,
                          'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
                          'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                          'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
                          'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                          'env_kwargs': None,
                          'loc_offset': 4.0, 'log_std_offset': 0.0,
                          'policy_kwargs': {'log_std_init': -2,
                                            'ortho_init': False,
                                            'activation_fn': nn.ReLU,
                                            'net_arch': {'pi': [128, 128],
                                                         'vf': [128, 128],
                                                         'log_score_vf': [128, 128]
                                                         },
                                            # 'features_extractor_class': HiddenObsExtractor,
                                            'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                          # 'hidden_obs_encoder_arch': [],
                                                                          # 'hidden_obs_encoder_arch': [32, 16],
                                                                          },
                                            # 'net_arch': [400, 300]
                                            },
                          'decoder_arch': [128, 128],
                          'eval_freq': 50_000,
                          'retrain_eval_episodes': 500,
                          'retrain_eval_freq': 250_000,
                          },
    'SafetyAntVelocity-v1': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 27, 'action_dim': 8,
                             'horizon': 1000, 'gamma': 0.99,
                             'n_steps': 1000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
                             'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5,
                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,  # 0.004,
                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': None,
                             'env_kwargs': {'terminate_when_unhealthy': False},
                             'loc_offset': 0.0, 'log_std_offset': 0.0,
                             'policy_kwargs': {'log_std_init': -2,
                                               'ortho_init': False,
                                               'activation_fn': nn.ReLU,
                                               'net_arch': {'pi': [64, 64],
                                                            'vf': [64, 64],
                                                            'log_score_vf': [64, 64]
                                                            },
                                               # 'features_extractor_class': HiddenObsExtractor,
                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                             # 'hidden_obs_encoder_arch': [],
                                                                             # 'hidden_obs_encoder_arch': [32, 16],
                                                                             },
                                               # 'net_arch': [400, 300]
                                               },
                             'decoder_arch': [64, 64],
                             'eval_freq': 2_500,
                             'retrain_eval_episodes': 1000,
                             'retrain_eval_freq': 25_000,
                             },
    'SafetySwimmerVelocity-v1': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 8, 'action_dim': 2,
                                 'horizon': 1000, 'gamma': 0.9999,
                                 'n_steps': 1000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
                                 'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.98, 'max_grad_norm': 0.5,
                                 'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,
                                 'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
                                 'env_kwargs': None,
                                 'loc_offset': 0.0, 'log_std_offset': 0.0,
                                 'policy_kwargs': {'log_std_init': -2,
                                                   'ortho_init': False,
                                                   'activation_fn': nn.ReLU,
                                                   'net_arch': {'pi': [64, 64],
                                                                'vf': [64, 64],
                                                                'log_score_vf': [64, 64]
                                                                },
                                                   # 'features_extractor_class': HiddenObsExtractor,
                                                   'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                                 # 'hidden_obs_encoder_arch': [],
                                                                                 # 'hidden_obs_encoder_arch': [32, 16],
                                                                                 },
                                                   # 'net_arch': [400, 300]
                                                   },
                                 'decoder_arch': [64, 64],
                                 'eval_freq': 2_500,
                                 'retrain_eval_episodes': 1000,
                                 'retrain_eval_freq': 25_000,
                                 },
    'SafetyHalfCheetahVelocity-v1': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 17, 'action_dim': 6,
                                     'horizon': 1000, 'gamma': 0.98,
                                     'n_steps': 1000, 'batch_size': 100, 'learning_rate': 2.0633e-05, 'ent_coef': 0.000401762,
                                     'clip_range': 0.1, 'n_epochs': 20, 'gae_lambda': 0.92, 'max_grad_norm': 0.8,
                                     'vf_coef': 0.58096, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,
                                     'log_score_vf_coef': 0.58096, 'normalize_log_score': False, 'target_kl': None,
                                     'env_kwargs': None,
                                     'loc_offset': 0.0, 'log_std_offset': 0.0,
                                     'policy_kwargs': {'log_std_init': -2,
                                                       'ortho_init': False,
                                                       'activation_fn': nn.ReLU,
                                                       'net_arch': {'pi': [64, 64],
                                                                    'vf': [64, 64],
                                                                    'log_score_vf': [64, 64]
                                                                    },
                                                       # 'features_extractor_class': HiddenObsExtractor,
                                                       'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                                     # 'hidden_obs_encoder_arch': [],
                                                                                     # 'hidden_obs_encoder_arch': [32, 16],
                                                                                     },
                                                       # 'net_arch': [400, 300]
                                                       },
                                     'decoder_arch': [64, 64],
                                     'eval_freq': 2_500,
                                     'retrain_eval_episodes': 1000,
                                     'retrain_eval_freq': 25_000,
                                     },
    'SafetyHopperVelocity-v1': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 11, 'action_dim': 3,
                                'horizon': 1000, 'gamma': 0.999,
                                'n_steps': 1000, 'batch_size': 100, 'learning_rate': 9.80828e-05, 'ent_coef': 0.00229519,
                                'clip_range': 0.2, 'n_epochs': 5, 'gae_lambda': 0.99, 'max_grad_norm': 0.7,
                                'vf_coef': 0.835671, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,  # 0.004,
                                'log_score_vf_coef': 0.835671, 'normalize_log_score': False, 'target_kl': None,
                                'env_kwargs': {'terminate_when_unhealthy': False},
                                'loc_offset': 0.0, 'log_std_offset': 0.0,
                                'policy_kwargs': {'log_std_init': -2,
                                                  'ortho_init': False,
                                                  'activation_fn': nn.ReLU,
                                                  'net_arch': {'pi': [64, 64],
                                                               'vf': [64, 64],
                                                               'log_score_vf': [64, 64]
                                                               },
                                                  # 'features_extractor_class': HiddenObsExtractor,
                                                  'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                                # 'hidden_obs_encoder_arch': [],
                                                                                # 'hidden_obs_encoder_arch': [32, 8],
                                                                                },
                                                  # 'net_arch': [400, 300]
                                                  },
                                'decoder_arch': [64, 64],
                                'eval_freq': 2_500,
                                'retrain_eval_episodes': 1000,
                                'retrain_eval_freq': 25_000,
                                },
    'SafetyWalker2dVelocity-v1': {'n_envs': 20, 'norm_obs': True, 'norm_reward': True, 'state_dim': 17, 'action_dim': 6,
                                  'horizon': 1000, 'gamma': 0.99,
                                  'n_steps': 1000, 'batch_size': 100, 'learning_rate': 5.05041e-05, 'ent_coef': 0.000585045,
                                  'clip_range': 0.1, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
                                  'vf_coef': 0.871923, 'lambda_batch_size': None, 'lambda_learning_rate': 0.007,
                                  'log_score_vf_coef': 0.871923, 'normalize_log_score': False, 'target_kl': None,
                                  'loc_offset': 0.0, 'log_std_offset': 0.0,
                                  'env_kwargs': {'terminate_when_unhealthy': False},
                                  'policy_kwargs': {'log_std_init': -2,
                                                    'ortho_init': False,
                                                    'activation_fn': nn.ReLU,
                                                    'net_arch': {'pi': [64, 64],
                                                                 'vf': [64, 64],
                                                                 'log_score_vf': [64, 64]
                                                                 },
                                                    # 'features_extractor_class': HiddenObsExtractor,
                                                    'features_extractor_kwargs': {# 'obs_encoder_arch': [],
                                                                                  # 'hidden_obs_encoder_arch': [],
                                                                                  # 'hidden_obs_encoder_arch': [32, 16],
                                                                                  },
                                                    # 'net_arch': [400, 300]
                                                    },
                                  'decoder_arch': [64, 64],
                                  'eval_freq': 2_500,
                                  'retrain_eval_episodes': 1000,
                                  'retrain_eval_freq': 25_000,
                                  },
    # 'SafetyAntCircle-v0': {'state_dim': 34, 'action_dim': 8, 'horizon': 1000, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    # 'SafetyBallCircle-v0': {'state_dim': 8, 'action_dim': 2, 'horizon': 250, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    # 'SafetyCarCircle-v0': {'state_dim': 8, 'action_dim': 2, 'horizon': 500, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    # 'SafetyDroneCircle-v0': {'state_dim': 18, 'action_dim': 4, 'horizon': 500, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    # 'SafetyAntRun-v0': {'state_dim': 33, 'action_dim': 8, 'horizon': 1000, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    # 'SafetyBallRun-v0': {'state_dim': 7, 'action_dim': 2, 'horizon': 250, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    # 'SafetyCarRun-v0': {'state_dim': 7, 'action_dim': 2, 'horizon': 500, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    # 'SafetyDroneRun-v0': {'state_dim': 17, 'action_dim': 4, 'horizon': 500, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'SafetyAntCircle-v0': {'state_dim': 34, 'action_dim': 8, 'horizon': 500, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'SafetyBallCircle-v0': {'state_dim': 8, 'action_dim': 2, 'horizon': 200, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'SafetyCarCircle-v0': {'state_dim': 8, 'action_dim': 2, 'horizon': 300, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'SafetyDroneCircle-v0': {'state_dim': 18, 'action_dim': 4, 'horizon': 300, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'SafetyAntRun-v0': {'state_dim': 33, 'action_dim': 8, 'horizon': 200, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'SafetyBallRun-v0': {'state_dim': 7, 'action_dim': 2, 'horizon': 100, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'SafetyCarRun-v0': {'state_dim': 7, 'action_dim': 2, 'horizon': 200, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'SafetyDroneRun-v0': {'state_dim': 17, 'action_dim': 4, 'horizon': 200, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'highway-v0': {'state_dim': 25, 'action_dim': 2, 'horizon': 40, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'highway-fast-v0': {'state_dim': 25, 'action_dim': 2, 'horizon': 40, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'parking-v0': {'state_dim': 18, 'action_dim': 2, 'horizon': 100, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
    'racetrack-v0':  {'state_dim': 25, 'action_dim': 2, 'horizon': 40, 'loc_offset': 0.0, 'log_std_offset': 0.0, 'decoder_arch': [64, 64],},
}

# mujoco_safety_gymnasium_dict = {
#     'SafetyPointButton1-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 76, 'action_dim': 2,
#                               'horizon': 1000, 'gamma': 0.99,
#                               'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                               'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                               'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
#                               'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                               'env_kwargs': None,
#                               'loc_offset': 4.0, 'log_std_offset': 0.0,
#                               'policy_kwargs': {'log_std_init': -2,
#                                                 'ortho_init': False,
#                                                 'activation_fn': nn.ReLU,
#                                                 'net_arch': {'pi': [128, 128],
#                                                              'vf': [128, 128],
#                                                              'log_score_vf': [128, 128]
#                                                              },
#                                                 # 'features_extractor_class': HiddenObsExtractor,
#                                                 'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                               # 'hidden_obs_encoder_arch': [],
#                                                                               # 'hidden_obs_encoder_arch': [32, 16],
#                                                                               },
#                                                 # 'net_arch': [400, 300]
#                                                 },
#                               'decoder_arch': [128, 128],
#                               'eval_freq': 50_000,
#                               'retrain_eval_episodes': 500,
#                               'retrain_eval_freq': 250_000,
#                               },
#     'SafetyPointButton2-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 76, 'action_dim': 2,
#                               'horizon': 1000, 'gamma': 0.99,
#                               'n_steps': 10000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
#                               'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                               'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
#                               'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                               'env_kwargs': None,
#                               'loc_offset': 4.0, 'log_std_offset': 0.0,
#                               'policy_kwargs': {'log_std_init': -2,
#                                                 'ortho_init': False,
#                                                 'activation_fn': nn.ReLU,
#                                                 'net_arch': {'pi': [128, 128],
#                                                              'vf': [128, 128],
#                                                              'log_score_vf': [128, 128]
#                                                              },
#                                                 # 'features_extractor_class': HiddenObsExtractor,
#                                                 'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                               # 'hidden_obs_encoder_arch': [],
#                                                                               # 'hidden_obs_encoder_arch': [32, 16],
#                                                                               },
#                                                 # 'net_arch': [400, 300]
#                                                 },
#                               'decoder_arch': [128, 128],
#                               'eval_freq': 50_000,
#                               'retrain_eval_episodes': 500,
#                               'retrain_eval_freq': 250_000,
#                               },
#     'SafetyPointCircle1-v0': {'n_envs': 10, 'norm_obs': False, 'norm_reward': True, 'state_dim': 28, 'action_dim': 2,
#                               'horizon': 500, 'gamma': 0.99,
#                               'n_steps': 500, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
#                               'clip_range': 0.2, 'n_epochs': 40, 'gae_lambda': 0.95, 'max_grad_norm': 40.0,
#                               'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,
#                               'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                               'env_kwargs': None,
#                               'loc_offset': 3.3, 'log_std_offset': 0.05,
#                               'policy_kwargs': {'log_std_init': -2,
#                                                 'ortho_init': False,
#                                                 'activation_fn': nn.Tanh,
#                                                 'net_arch': {'pi': [128, 128],
#                                                              'vf': [128, 128],
#                                                              'log_score_vf': [128, 128]
#                                                              },
#                                                 # 'features_extractor_class': HiddenObsExtractor,
#                                                 'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                               # 'hidden_obs_encoder_arch': [],
#                                                                               # 'hidden_obs_encoder_arch': [32, 16],
#                                                                               },
#                                                 # 'net_arch': [400, 300]
#                                                 },
#                               'decoder_arch': [64, 64],
#                               'eval_freq': 5_000,
#                               'retrain_eval_episodes': 500,
#                               'retrain_eval_freq': 25_000,
#                               },
#     'SafetyPointCircle2-v0': {'n_envs': 10, 'norm_obs': False, 'norm_reward': True, 'state_dim': 28, 'action_dim': 2,
#                               'horizon': 500, 'gamma': 0.99,
#                               'n_steps': 500, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
#                               'clip_range': 0.2, 'n_epochs': 40, 'gae_lambda': 0.95, 'max_grad_norm': 40.0,
#                               'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,
#                               'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                               'env_kwargs': None,
#                               'loc_offset': 3.3, 'log_std_offset': 0.05,
#                               'policy_kwargs': {'log_std_init': -2,
#                                                 'ortho_init': False,
#                                                 'activation_fn': nn.Tanh,
#                                                 'net_arch': {'pi': [128, 128],
#                                                              'vf': [128, 128],
#                                                              'log_score_vf': [128, 128]
#                                                              },
#                                                 # 'features_extractor_class': HiddenObsExtractor,
#                                                 'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                               # 'hidden_obs_encoder_arch': [],
#                                                                               # 'hidden_obs_encoder_arch': [32, 16],
#                                                                               },
#                                                 # 'net_arch': [400, 300]
#                                                 },
#                               'decoder_arch': [64, 64],
#                               'eval_freq': 5_000,
#                               'retrain_eval_episodes': 500,
#                               'retrain_eval_freq': 25_000,
#                               },
#     'SafetyPointGoal1-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 60, 'action_dim': 2,
#                             'horizon': 1000, 'gamma': 0.99,
#                             'n_steps': 1000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                             'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.02,
#                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                             'env_kwargs': None,
#                             'loc_offset': 4.0, 'log_std_offset': 0.0,
#                             'policy_kwargs': {'log_std_init': -2,
#                                               'ortho_init': False,
#                                               'activation_fn': nn.ReLU,
#                                               'net_arch': {'pi': [128, 128],
#                                                            'vf': [128, 128],
#                                                            'log_score_vf': [128, 128]
#                                                            },
#                                               # 'features_extractor_class': HiddenObsExtractor,
#                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [32, 16],
#                                                                             },
#                                               # 'net_arch': [400, 300]
#                                               },
#                             'decoder_arch': [128, 128],
#                             'eval_freq': 5_000,
#                             'retrain_eval_episodes': 500,
#                             'retrain_eval_freq': 25_000,
#                             },
#     'SafetyPointGoal2-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 60, 'action_dim': 2,
#                             'horizon': 1000, 'gamma': 0.99,
#                             'n_steps': 1000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                             'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.02,
#                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                             'env_kwargs': None,
#                             'loc_offset': 4.0, 'log_std_offset': 0.0,
#                             'policy_kwargs': {'log_std_init': -2,
#                                               'ortho_init': False,
#                                               'activation_fn': nn.ReLU,
#                                               'net_arch': {'pi': [128, 128],
#                                                            'vf': [128, 128],
#                                                            'log_score_vf': [128, 128]
#                                                            },
#                                               # 'features_extractor_class': HiddenObsExtractor,
#                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [32, 16],
#                                                                             },
#                                               # 'net_arch': [400, 300]
#                                               },
#                             'decoder_arch': [128, 128],
#                             'eval_freq': 5_000,
#                             'retrain_eval_episodes': 500,
#                             'retrain_eval_freq': 25_000,
#                             },
#     'SafetyPointPush1-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 76, 'action_dim': 2,
#                             'horizon': 1000, 'gamma': 0.99,
#                             'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                             'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
#                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                             'env_kwargs': None,
#                             'loc_offset': 4.0, 'log_std_offset': 0.0,
#                             'policy_kwargs': {'log_std_init': -2,
#                                               'ortho_init': False,
#                                               'activation_fn': nn.ReLU,
#                                               'net_arch': {'pi': [128, 128],
#                                                            'vf': [128, 128],
#                                                            'log_score_vf': [128, 128]
#                                                            },
#                                               # 'features_extractor_class': HiddenObsExtractor,
#                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [32, 16],
#                                                                             },
#                                               # 'net_arch': [400, 300]
#                                               },
#                             'decoder_arch': [128, 128],
#                             'eval_freq': 50_000,
#                             'retrain_eval_episodes': 500,
#                             'retrain_eval_freq': 250_000,
#                             },
#     'SafetyPointPush2-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 76, 'action_dim': 2,
#                             'horizon': 1000, 'gamma': 0.99,
#                             'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                             'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
#                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                             'env_kwargs': None,
#                             'loc_offset': 4.0, 'log_std_offset': 0.0,
#                             'policy_kwargs': {'log_std_init': -2,
#                                               'ortho_init': False,
#                                               'activation_fn': nn.ReLU,
#                                               'net_arch': {'pi': [128, 128],
#                                                            'vf': [128, 128],
#                                                            'log_score_vf': [128, 128]
#                                                            },
#                                               # 'features_extractor_class': HiddenObsExtractor,
#                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [32, 16],
#                                                                             },
#                                               # 'net_arch': [400, 300]
#                                               },
#                             'decoder_arch': [128, 128],
#                             'eval_freq': 50_000,
#                             'retrain_eval_episodes': 500,
#                             'retrain_eval_freq': 250_000,
#                             },
#     'SafetyCarButton1-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 88, 'action_dim': 2,
#                             'horizon': 1000, 'gamma': 0.99,
#                             'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                             'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
#                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                             'env_kwargs': None,
#                             'loc_offset': 4.0, 'log_std_offset': 0.0,
#                             'policy_kwargs': {'log_std_init': -2,
#                                               'ortho_init': False,
#                                               'activation_fn': nn.ReLU,
#                                               'net_arch': {'pi': [128, 128],
#                                                            'vf': [128, 128],
#                                                            'log_score_vf': [128, 128]
#                                                            },
#                                               # 'features_extractor_class': HiddenObsExtractor,
#                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [32, 16],
#                                                                             },
#                                               # 'net_arch': [400, 300]
#                                               },
#                             'decoder_arch': [128, 128],
#                             'eval_freq': 50_000,
#                             'retrain_eval_episodes': 500,
#                             'retrain_eval_freq': 250_000,
#                             },
#     'SafetyCarButton2-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 88, 'action_dim': 2,
#                             'horizon': 1000, 'gamma': 0.99,
#                             'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                             'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
#                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                             'env_kwargs': None,
#                             'loc_offset': 4.0, 'log_std_offset': 0.0,
#                             'policy_kwargs': {'log_std_init': -2,
#                                               'ortho_init': False,
#                                               'activation_fn': nn.ReLU,
#                                               'net_arch': {'pi': [128, 128],
#                                                            'vf': [128, 128],
#                                                            'log_score_vf': [128, 128]
#                                                            },
#                                               # 'features_extractor_class': HiddenObsExtractor,
#                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [32, 16],
#                                                                             },
#                                               # 'net_arch': [400, 300]
#                                               },
#                             'decoder_arch': [128, 128],
#                             'eval_freq': 50_000,
#                             'retrain_eval_episodes': 500,
#                             'retrain_eval_freq': 250_000,
#                             },
#     'SafetyCarCircle1-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 40, 'action_dim': 2,
#                             'horizon': 500, 'gamma': 0.99,
#                             'n_steps': 500, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
#                             'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,
#                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                             'env_kwargs': None,
#                             'loc_offset': 3.3, 'log_std_offset': 0.05,
#                             'policy_kwargs': {'log_std_init': -2,
#                                               'ortho_init': False,
#                                               'activation_fn': nn.ReLU,
#                                               'net_arch': {'pi': [128, 128],
#                                                            'vf': [128, 128],
#                                                            'log_score_vf': [128, 128]
#                                                            },
#                                               # 'features_extractor_class': HiddenObsExtractor,
#                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [32, 16],
#                                                                             },
#                                               # 'net_arch': [400, 300]
#                                               },
#                             'decoder_arch': [64, 64],
#                             'eval_freq': 5_000,
#                             'retrain_eval_episodes': 500,
#                             'retrain_eval_freq': 25_000,
#                             },
#     'SafetyCarCircle2-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 40, 'action_dim': 2,
#                             'horizon': 500, 'gamma': 0.99,
#                             'n_steps': 500, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
#                             'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                             'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,
#                             'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                             'env_kwargs': None,
#                             'loc_offset': 3.3, 'log_std_offset': 0.05,
#                             'policy_kwargs': {'log_std_init': -2,
#                                               'ortho_init': False,
#                                               'activation_fn': nn.ReLU,
#                                               'net_arch': {'pi': [128, 128],
#                                                            'vf': [128, 128],
#                                                            'log_score_vf': [128, 128]
#                                                            },
#                                               # 'features_extractor_class': HiddenObsExtractor,
#                                               'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [],
#                                                                             # 'hidden_obs_encoder_arch': [32, 16],
#                                                                             },
#                                               # 'net_arch': [400, 300]
#                                               },
#                             'decoder_arch': [64, 64],
#                             'eval_freq': 5_000,
#                             'retrain_eval_episodes': 500,
#                             'retrain_eval_freq': 25_000,
#                             },
#     'SafetyCarGoal1-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 72, 'action_dim': 2,
#                           'horizon': 1000, 'gamma': 0.99,
#                           'n_steps': 1000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                           'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                           'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.02,
#                           'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                           'env_kwargs': None,
#                           'loc_offset': 4.0, 'log_std_offset': 0.0,
#                           'policy_kwargs': {'log_std_init': -2,
#                                             'ortho_init': False,
#                                             'activation_fn': nn.ReLU,
#                                             'net_arch': {'pi': [128, 128],
#                                                          'vf': [128, 128],
#                                                          'log_score_vf': [128, 128]
#                                                          },
#                                             # 'features_extractor_class': HiddenObsExtractor,
#                                             'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                           # 'hidden_obs_encoder_arch': [],
#                                                                           # 'hidden_obs_encoder_arch': [32, 16],
#                                                                           },
#                                             # 'net_arch': [400, 300]
#                                             },
#                           'decoder_arch': [128, 128],
#                           'eval_freq': 5_000,
#                           'retrain_eval_episodes': 500,
#                           'retrain_eval_freq': 25_000,
#                           },
#     'SafetyCarGoal2-v0': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 72, 'action_dim': 2,
#                           'horizon': 1000, 'gamma': 0.99,
#                           'n_steps': 1000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                           'clip_range': 0.2, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                           'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.02,
#                           'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                           'env_kwargs': None,
#                           'loc_offset': 4.0, 'log_std_offset': 0.0,
#                           'policy_kwargs': {'log_std_init': -2,
#                                             'ortho_init': False,
#                                             'activation_fn': nn.ReLU,
#                                             'net_arch': {'pi': [128, 128],
#                                                          'vf': [128, 128],
#                                                          'log_score_vf': [128, 128]
#                                                          },
#                                             # 'features_extractor_class': HiddenObsExtractor,
#                                             'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                           # 'hidden_obs_encoder_arch': [],
#                                                                           # 'hidden_obs_encoder_arch': [32, 16],
#                                                                           },
#                                             # 'net_arch': [400, 300]
#                                             },
#                           'decoder_arch': [128, 128],
#                           'eval_freq': 5_000,
#                           'retrain_eval_episodes': 500,
#                           'retrain_eval_freq': 25_000,
#                           },
#     'SafetyCarPush1-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 88, 'action_dim': 2,
#                           'horizon': 1000, 'gamma': 0.99,
#                           'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                           'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                           'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
#                           'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                           'env_kwargs': None,
#                           'loc_offset': 4.0, 'log_std_offset': 0.0,
#                           'policy_kwargs': {'log_std_init': -2,
#                                             'ortho_init': False,
#                                             'activation_fn': nn.ReLU,
#                                             'net_arch': {'pi': [128, 128],
#                                                          'vf': [128, 128],
#                                                          'log_score_vf': [128, 128]
#                                                          },
#                                             # 'features_extractor_class': HiddenObsExtractor,
#                                             'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                           # 'hidden_obs_encoder_arch': [],
#                                                                           # 'hidden_obs_encoder_arch': [32, 16],
#                                                                           },
#                                             # 'net_arch': [400, 300]
#                                             },
#                           'decoder_arch': [128, 128],
#                           'eval_freq': 50_000,
#                           'retrain_eval_episodes': 500,
#                           'retrain_eval_freq': 250_000,
#                           },
#     'SafetyCarPush2-v0': {'n_envs': 1, 'norm_obs': True, 'norm_reward': True, 'state_dim': 88, 'action_dim': 2,
#                           'horizon': 1000, 'gamma': 0.99,
#                           'n_steps': 10000, 'batch_size': 100, 'learning_rate': 1e-4, 'ent_coef': 0.0,
#                           'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                           'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.004,
#                           'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                           'env_kwargs': None,
#                           'loc_offset': 4.0, 'log_std_offset': 0.0,
#                           'policy_kwargs': {'log_std_init': -2,
#                                             'ortho_init': False,
#                                             'activation_fn': nn.ReLU,
#                                             'net_arch': {'pi': [128, 128],
#                                                          'vf': [128, 128],
#                                                          'log_score_vf': [128, 128]
#                                                          },
#                                             # 'features_extractor_class': HiddenObsExtractor,
#                                             'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                           # 'hidden_obs_encoder_arch': [],
#                                                                           # 'hidden_obs_encoder_arch': [32, 16],
#                                                                           },
#                                             # 'net_arch': [400, 300]
#                                             },
#                           'decoder_arch': [128, 128],
#                           'eval_freq': 50_000,
#                           'retrain_eval_episodes': 500,
#                           'retrain_eval_freq': 250_000,
#                           },
#     'SafetyAntVelocity-v1': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 27, 'action_dim': 8,
#                              'horizon': 1000, 'gamma': 0.99,
#                              'n_steps': 1000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
#                              'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5,
#                              'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,  # 0.004,
#                              'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                              'env_kwargs': {'terminate_when_unhealthy': False},
#                              'loc_offset': 3.3, 'log_std_offset': 0.05,
#                              'policy_kwargs': {'log_std_init': -2,
#                                                'ortho_init': False,
#                                                'activation_fn': nn.ReLU,
#                                                'net_arch': {'pi': [64, 64],
#                                                             'vf': [64, 64],
#                                                             'log_score_vf': [64, 64]
#                                                             },
#                                                # 'features_extractor_class': HiddenObsExtractor,
#                                                'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                              # 'hidden_obs_encoder_arch': [],
#                                                                              # 'hidden_obs_encoder_arch': [32, 16],
#                                                                              },
#                                                # 'net_arch': [400, 300]
#                                                },
#                              'decoder_arch': [64, 64],
#                              'eval_freq': 5_000,
#                              'retrain_eval_episodes': 500,
#                              'retrain_eval_freq': 25_000,
#                              },
#     'SafetySwimmerVelocity-v1': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 8, 'action_dim': 2,
#                                  'horizon': 1000, 'gamma': 0.9999,
#                                  'n_steps': 1000, 'batch_size': 100, 'learning_rate': 3e-4, 'ent_coef': 0.0,
#                                  'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.98, 'max_grad_norm': 0.5,
#                                  'vf_coef': 0.5, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,
#                                  'log_score_vf_coef': 0.5, 'normalize_log_score': False, 'target_kl': 0.02,
#                                  'env_kwargs': None,
#                                  'loc_offset': 3.0, 'log_std_offset': 0.0,
#                                  'policy_kwargs': {'log_std_init': -2,
#                                                    'ortho_init': False,
#                                                    'activation_fn': nn.ReLU,
#                                                    'net_arch': {'pi': [64, 64],
#                                                                 'vf': [64, 64],
#                                                                 'log_score_vf': [64, 64]
#                                                                 },
#                                                    # 'features_extractor_class': HiddenObsExtractor,
#                                                    'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                                  # 'hidden_obs_encoder_arch': [],
#                                                                                  # 'hidden_obs_encoder_arch': [32, 16],
#                                                                                  },
#                                                    # 'net_arch': [400, 300]
#                                                    },
#                                  'decoder_arch': [64, 64],
#                                  'eval_freq': 5_000,
#                                  'retrain_eval_episodes': 500,
#                                  'retrain_eval_freq': 25_000,
#                                  },
#     'SafetyHalfCheetahVelocity-v1': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 17, 'action_dim': 6,
#                                      'horizon': 1000, 'gamma': 0.98,
#                                      'n_steps': 1000, 'batch_size': 100, 'learning_rate': 2.0633e-05, 'ent_coef': 0.000401762,
#                                      'clip_range': 0.1, 'n_epochs': 20, 'gae_lambda': 0.92, 'max_grad_norm': 0.8,
#                                      'vf_coef': 0.58096, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,
#                                      'log_score_vf_coef': 0.58096, 'normalize_log_score': False, 'target_kl': 0.02,
#                                      'env_kwargs': None,
#                                      'loc_offset': 2.9, 'log_std_offset': 0.05,
#                                      'policy_kwargs': {'log_std_init': -2,
#                                                        'ortho_init': False,
#                                                        'activation_fn': nn.ReLU,
#                                                        'net_arch': {'pi': [64, 64],
#                                                                     'vf': [64, 64],
#                                                                     'log_score_vf': [64, 64]
#                                                                     },
#                                                        # 'features_extractor_class': HiddenObsExtractor,
#                                                        'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                                      # 'hidden_obs_encoder_arch': [],
#                                                                                      # 'hidden_obs_encoder_arch': [32, 16],
#                                                                                      },
#                                                        # 'net_arch': [400, 300]
#                                                        },
#                                      'decoder_arch': [64, 64],
#                                      'eval_freq': 5_000,
#                                      'retrain_eval_episodes': 500,
#                                      'retrain_eval_freq': 25_000,
#                                      },
#     'SafetyHopperVelocity-v1': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 11, 'action_dim': 3,
#                                 'horizon': 1000, 'gamma': 0.999,
#                                 'n_steps': 1000, 'batch_size': 100, 'learning_rate': 9.80828e-05, 'ent_coef': 0.00229519,
#                                 'clip_range': 0.2, 'n_epochs': 5, 'gae_lambda': 0.99, 'max_grad_norm': 0.7,
#                                 'vf_coef': 0.835671, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,  # 0.004,
#                                 'log_score_vf_coef': 0.835671, 'normalize_log_score': False, 'target_kl': 0.02,
#                                 'env_kwargs': {'terminate_when_unhealthy': False},
#                                 'loc_offset': 2.9, 'log_std_offset': 0.05,
#                                 'policy_kwargs': {'log_std_init': -2,
#                                                   'ortho_init': False,
#                                                   'activation_fn': nn.ReLU,
#                                                   'net_arch': {'pi': [64, 64],
#                                                                'vf': [64, 64],
#                                                                'log_score_vf': [64, 64]
#                                                                },
#                                                   # 'features_extractor_class': HiddenObsExtractor,
#                                                   'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                                 # 'hidden_obs_encoder_arch': [],
#                                                                                 # 'hidden_obs_encoder_arch': [32, 8],
#                                                                                 },
#                                                   # 'net_arch': [400, 300]
#                                                   },
#                                 'decoder_arch': [64, 64],
#                                 'eval_freq': 5_000,
#                                 'retrain_eval_episodes': 500,
#                                 'retrain_eval_freq': 25_000,
#                                 },
#     'SafetyWalker2dVelocity-v1': {'n_envs': 10, 'norm_obs': True, 'norm_reward': True, 'state_dim': 17, 'action_dim': 6,
#                                   'horizon': 1000, 'gamma': 0.99,
#                                   'n_steps': 1000, 'batch_size': 100, 'learning_rate': 5.05041e-05, 'ent_coef': 0.000585045,
#                                   'clip_range': 0.1, 'n_epochs': 20, 'gae_lambda': 0.95, 'max_grad_norm': 1.0,
#                                   'vf_coef': 0.871923, 'lambda_batch_size': None, 'lambda_learning_rate': 0.01,
#                                   'log_score_vf_coef': 0.871923, 'normalize_log_score': False, 'target_kl': 0.02,
#                                   'loc_offset': 2.9, 'log_std_offset': 0.05,
#                                   'env_kwargs': {'terminate_when_unhealthy': False},
#                                   'policy_kwargs': {'log_std_init': -2,
#                                                     'ortho_init': False,
#                                                     'activation_fn': nn.ReLU,
#                                                     'net_arch': {'pi': [64, 64],
#                                                                  'vf': [64, 64],
#                                                                  'log_score_vf': [64, 64]
#                                                                  },
#                                                     # 'features_extractor_class': HiddenObsExtractor,
#                                                     'features_extractor_kwargs': {# 'obs_encoder_arch': [],
#                                                                                   # 'hidden_obs_encoder_arch': [],
#                                                                                   # 'hidden_obs_encoder_arch': [32, 16],
#                                                                                   },
#                                                     # 'net_arch': [400, 300]
#                                                     },
#                                   'decoder_arch': [64, 64],
#                                   'eval_freq': 5_000,
#                                   'retrain_eval_episodes': 500,
#                                   'retrain_eval_freq': 25_000,
#                                   },
# }


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def read_hdf5(h5path: str):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in [
        'observations', 'next_observations', 'actions', 'rewards', 'costs',
        'terminals', 'timeouts'
    ]:
        assert key in data_dict, 'Dataset is missing key %s' % key
    n_samples = data_dict['observations'].shape[0]
    if data_dict['rewards'].shape == (n_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (
        n_samples,
    ), 'Reward has wrong shape: %s' % (str(data_dict['rewards'].shape))
    if data_dict['costs'].shape == (n_samples, 1):
        data_dict['costs'] = data_dict['costs'][:, 0]
    assert data_dict['costs'].shape == (
        n_samples,
    ), 'Costs has wrong shape: %s' % (str(data_dict['costs'].shape))
    if data_dict['terminals'].shape == (n_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (
        n_samples,
    ), 'Terminals has wrong shape: %s' % (str(data_dict['rewards'].shape))
    data_dict["observations"] = data_dict["observations"].astype("float32")
    data_dict["actions"] = data_dict["actions"].astype("float32")
    data_dict["next_observations"] = data_dict["next_observations"].astype("float32")
    data_dict["rewards"] = data_dict["rewards"].astype("float32")
    data_dict["costs"] = data_dict["costs"].astype("float32")
    return data_dict


def collate(batch):
    batch_size = len(batch)
    batch_feat_dim = batch[0][0].shape[1]
    batch_lengths = [item[0].shape[0] for item in batch]
    batch_max_length = max(batch_lengths)

    x_tensor = Variable(th.zeros((batch_size, batch_max_length, batch_feat_dim)).to(utils.device)).float()

    y_tensor = []

    for idx, (item, target) in enumerate(batch):
        x_tensor[idx, :item.shape[0], :] = item
        y_tensor.append(target)

    y_tensor = th.LongTensor(y_tensor).to(utils.device)
    batch_lengths = th.tensor(batch_lengths).to(utils.device)

    return x_tensor, y_tensor, batch_lengths

def collate_maxlength(horizon=1000):

    def custom_collate(batch):
        batch_size = len(batch)
        batch_feat_dim = batch[0][0].shape[1]
        batch_lengths = [item[0].shape[0] for item in batch]
        batch_max_length = horizon

        x_tensor = Variable(th.zeros((batch_size, batch_max_length, batch_feat_dim)).to(utils.device)).float()

        y_tensor = []

        for idx, (item, target) in enumerate(batch):
            x_tensor[idx, :item.shape[0], :] = item
            y_tensor.append(target)

        y_tensor = th.LongTensor(y_tensor).to(utils.device)
        batch_lengths = th.tensor(batch_lengths).to(utils.device)

        return x_tensor, y_tensor, batch_lengths

    return custom_collate


class TrajHDF5Data:

    def __init__(self, hdf5_file: str, horizon: int, obs_dim: int = None, act_dim: int = None):

        # obs, action, reward, cost, next_obs, done
        self.horizon = horizon
        self.list_traj_sa = []
        self.list_traj_cost = []

        dataset = read_hdf5(hdf5_file)
        if obs_dim is not None:
            assert dataset['observations'].shape[1] == obs_dim, \
                'Observation shape does not match env: %s vs %s' % (
                    str(dataset['observations'].shape[1]), str(obs_dim))
        if act_dim is not None:
            assert dataset['actions'].shape[1] == act_dim, \
                'Action shape does not match env: %s vs %s' % (
                    str(dataset['actions'].shape[1]), str(act_dim))

        dataset_size = dataset['observations'].shape[0]
        dones = dataset['terminals'] + dataset['timeouts']
        num_trajectories = dones.sum()
        print(dataset_size)
        print(num_trajectories)

        # assert dataset_size % horizon == 0, "Number of steps in dataset does not match horizon"
        assert dataset_size <= num_trajectories * horizon, "Number of steps in dataset does not match horizon"

        # for i in range(0, dataset_size, horizon):
        #     self.list_traj_sa.append(np.column_stack((dataset['observations'][i:(i + horizon)],
        #                                               dataset['actions'][i:(i + horizon)])))
        #     self.list_traj_cost.append(dataset['costs'][i:(i + horizon)])

        begin_idx = 0
        for idx in np.where(dones)[0]:
            end_idx = idx + 1
            self.list_traj_sa.append(np.column_stack((dataset['observations'][begin_idx:end_idx],
                                                      dataset['actions'][begin_idx:end_idx])))
            self.list_traj_cost.append(dataset['costs'][begin_idx:end_idx])
            begin_idx = end_idx

    def get(self, traj_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.list_traj_sa[traj_idx], self.list_traj_cost[traj_idx]

    def get_num_traj(self) -> int:
        return len(self.list_traj_cost)


class TrajNPData:

    def __init__(self, list_npz_filepaths: List, horizon: int):

        # obs, action, reward, cost, next_obs, done
        self.horizon = horizon
        self.list_traj_sa = []
        self.list_traj_cost = []

        npz_files = []
        for npz_filepath in list_npz_filepaths:
            npz_files += [os.path.join(npz_filepath, name) for name in os.listdir(npz_filepath) if
                          os.path.isfile(os.path.join(npz_filepath, name)) and name.endswith('.npz')]

        for idx, npz_file in enumerate(npz_files):

            # print(datetime.now().strftime("%d-%m-%Y_%H%M%S") + " " + str(idx) + " trajectories read",
            #       flush=True)

            npz_data = np.load(npz_file)
            self.list_traj_sa.append(np.column_stack((npz_data['obs'],
                                                      npz_data['action'])))
            self.list_traj_cost.append(npz_data['cost'])

            assert self.list_traj_cost[-1].shape[0] == self.horizon, \
                "Number of steps in trajectory does not match horizon"

    def get(self, traj_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.list_traj_sa[traj_idx], self.list_traj_cost[traj_idx]

    def get_num_traj(self) -> int:
        return len(self.list_traj_cost)


class TrajDFData:

    def __init__(self, list_dataframes: List[pd.DataFrame], domain: str):

        # obs, action, reward, cost, next_obs, done
        self.domain = domain
        self.horizon = mujoco_safety_gymnasium_dict[self.domain]['horizon']
        self.list_traj_sa = []
        self.list_traj_cost = []

        for idx, df in enumerate(list_dataframes):

            # print(datetime.now().strftime("%d-%m-%Y_%H%M%S") + " " + str(idx) + " trajectories read",
            #       flush=True)

            self.list_traj_sa.append(np.column_stack(
                (df[['s' + str(i) for i in range(mujoco_safety_gymnasium_dict[self.domain]['state_dim'])]].to_numpy(),
                 df[['a' + str(i) for i in range(mujoco_safety_gymnasium_dict[self.domain]['action_dim'])]].to_numpy())
            ))
            self.list_traj_cost.append(df['c'].to_numpy())

            # print("Traj shape", self.list_traj_cost[-1].shape[0])
            # print("Horizon", self.horizon)
            # print("Traj (Last):", self.list_traj_cost[-1])

            assert self.list_traj_cost[-1].shape[0] <= self.horizon, \
                "Number of steps in trajectory is longer than horizon"

    def get(self, traj_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.list_traj_sa[traj_idx], self.list_traj_cost[traj_idx]

    def get_num_traj(self) -> int:
        return len(self.list_traj_cost)


class MujocoNPDataset(Dataset):
    def __init__(self,
                 mujoco_domain: str,
                 np_data: Union[TrajNPData, TrajDFData, TrajHDF5Data] = None,
                 indices: np.ndarray = None,
                 transform: Callable = None,
                 target_transform: Callable = None,
                 noise: float = 0.0):

        self.target_map = {'safe': 1, 'unsafe': 0}
        self.x, self.y = [], []
        self.domain = mujoco_domain
        self.n_queries = 0  # Record number of label queries
        if np_data is not None and indices is not None:
            self.add_augment_data(np_data, indices, noise)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def merge_dataset(self, dataset):
        self.x += dataset.x
        self.y += dataset.y

    def add_augment_data(self, np_data: Union[TrajNPData, TrajDFData, TrajHDF5Data], indices: Union[List, np.ndarray],
                         noise: float = 0.0) -> int:

        # Can delete for new dataset loaded from torch load, only for transition for using old dataset
        if not hasattr(self, 'n_queries'):
            self.n_queries = 0

        augment_slice = mujoco_markov_safety_dict[self.domain]['augment_slice']
        # horizon = mujoco_safety_gymnasium_dict[self.domain]['horizon']
        total_datapoints, safe_datapoints = 0, 0
        initial_n_queries = self.n_queries

        for idx, np_idx in enumerate(indices):

            if idx % 100 == 0:
                print(datetime.now().strftime("%d-%m-%Y_%H%M%S") + " " + str(idx) + " trajectories read",
                      flush=True)

            traj_sa_data, traj_cost_data = np_data.get(np_idx)

            labeled_traj = LabeledNPData(traj_sa_data, traj_cost_data, self.domain, traj_cost_data.shape[0],
                                         markov_cost=True)

            safe_bool = labeled_traj.safe
            y_label = 'safe' if (safe_bool and traj_cost_data.shape[0] == np_data.horizon) else 'unsafe'

            target = self.target_map[y_label]

            total_datapoints += 1
            if target == self.target_map['safe']:
                safe_datapoints += 1

            self.n_queries += 1

            if random.random() < noise:
                # Flip the label: assumes 'safe' and 'unsafe' are distinct ints or bools
                target = (
                    self.target_map['unsafe'] if target == self.target_map['safe']
                    else self.target_map['safe']
                )

            self.x.append(th.FloatTensor(traj_sa_data))  # .to(utils.device))
            self.y.append(target)

            for rows in range(augment_slice, np_data.horizon, augment_slice):

                # print(idx, " ", rows, flush=True)
                if rows >= traj_cost_data.shape[0]:
                    break

                sub_sa_data = traj_sa_data[:rows]

                # Overall trajectory is safe, so all subtrajectories must be safe
                if target == self.target_map['safe']:
                    sub_target = True
                else:
                    sub_target, _ = labeled_traj.label_data(end_idx=rows)

                if sub_target:
                    sub_target = self.target_map['safe']
                    self.n_queries += 1
                else:
                    sub_target = self.target_map['unsafe']

                # sub_target = self.target_map['safe'] if sub_target else self.target_map['unsafe']
                if random.random() < noise:
                    # Flip the label: assumes 'safe' and 'unsafe' are distinct ints or bools
                    sub_target = (
                        self.target_map['unsafe'] if sub_target == self.target_map['safe']
                        else self.target_map['safe']
                    )

                self.x.append(th.FloatTensor(sub_sa_data))  # .to(utils.device))
                self.y.append(sub_target)

                total_datapoints += 1
                if sub_target == 1:
                    safe_datapoints += 1

        print(str(safe_datapoints) + " safe data points")
        print(str(total_datapoints) + " total data points")
        print("Safe Proportions: " + str(safe_datapoints / total_datapoints if total_datapoints > 0 else 0))
        print(str(len(indices)) + " trajectories processed")
        print(str(self.n_queries - initial_n_queries) + " label queries made")

        return self.n_queries - initial_n_queries


class HumanNPDataset(Dataset):
    def __init__(self,
                 mujoco_domain: str,
                 np_data: Union[TrajNPData, TrajDFData, TrajHDF5Data] = None,
                 labels: np.ndarray = None,
                 indices: np.ndarray = None,
                 transform: Callable = None,
                 target_transform: Callable = None):

        self.target_map = {'safe': 1, 'unsafe': 0}
        self.x, self.y = [], []
        self.domain = mujoco_domain
        self.n_queries = 0  # Record number of label queries
        if np_data is not None and labels is not None and indices is not None:
            self.add_data(np_data, labels, indices)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def merge_dataset(self, dataset):
        self.x += dataset.x
        self.y += dataset.y

    def add_data(self, np_data: Union[TrajNPData, TrajDFData, TrajHDF5Data], labels: np.ndarray,
                 indices: Union[List, np.ndarray]) -> int:

        # Can delete for new dataset loaded from torch load, only for transition for using old dataset
        if not hasattr(self, 'n_queries'):
            self.n_queries = 0

        total_datapoints, safe_datapoints = 0, 0
        initial_n_queries = self.n_queries

        for idx, np_idx in enumerate(indices):

            traj_sa_data, traj_cost_data = np_data.get(np_idx)

            safe_bool = labels[np_idx]
            y_label = 'safe' if safe_bool else 'unsafe'

            target = self.target_map[y_label]

            self.x.append(th.FloatTensor(traj_sa_data))  # .to(utils.device))
            self.y.append(target)

            total_datapoints += 1
            if target == self.target_map['safe']:
                safe_datapoints += 1

            self.n_queries += 1

        print(str(safe_datapoints) + " safe data points")
        print(str(total_datapoints) + " total data points")
        print("Safe Proportions: " + str(safe_datapoints / total_datapoints if total_datapoints > 0 else 0))
        print(str(len(indices)) + " trajectories processed")
        print(str(self.n_queries - initial_n_queries) + " label queries made")

        return self.n_queries - initial_n_queries


class PtEstGRU(nn.Module):
    def __init__(self, feature_dim=11, nb_gru_units=16, batch_size=256, gru_layers=2, mlp_arch=None, dropout=0.0):
        super().__init__()
        if mlp_arch is None:
            mlp_arch = [64, 64]
        self.hidden = None
        self.feature_dim = feature_dim
        self.nb_gru_units = nb_gru_units
        self.gru_layers = gru_layers
        self.batch_size = batch_size
        self.mlp_arch = mlp_arch
        self.dropout = dropout

        # build actual NN
        self.__build_model()


    def __build_model(self):
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.nb_gru_units,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=self.dropout
        )

        # Decoder Module
        decoder = []
        if self.dropout > 0:
            decoder.append(nn.Dropout(self.dropout * 2))
        prev_in_features = self.nb_gru_units * 2
        for i, out_features in enumerate(self.mlp_arch):
            decoder.append(nn.Linear(prev_in_features, out_features))
            decoder.append(nn.ReLU())
            # decoder.append(nn.LayerNorm(out_features))
            if (i < len(self.mlp_arch) - 1) and (self.dropout > 0):
                decoder.append(nn.Dropout(self.dropout * 2))
            prev_in_features = out_features
        self.decoder = nn.Sequential(*decoder)

        self.decoder_output = nn.Linear(self.mlp_arch[-1], 1)
        nn.init.normal_(self.decoder_output.weight, mean=-0.5, std=0.1)
        nn.init.constant_(self.decoder_output.bias, -5.0)

    def init_hidden(self, init_h=None):

        if init_h is None:
            # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
            hidden_h = th.zeros(self.gru_layers, self.batch_size, self.nb_gru_units).to(utils.device)
        else:
            hidden_h = init_h

        hidden_h = Variable(hidden_h)

        return hidden_h

    def forward(self, x, x_lengths, init_h=None):
        # reset the hidden state. Must be done before you run a new batch
        self.hidden = self.init_hidden(init_h)
        # print(self.hidden)

        batch_size, seq_len, feature_dim = x.size()
        x_clone = x.clone().swapaxes(0, 1)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the GRU
        x = th.nn.utils.rnn.pack_padded_sequence(x, x_lengths.cpu(), enforce_sorted=False, batch_first=True)

        # now run through GRU
        x, self.hidden = self.gru(x, self.hidden)
        x_unpack = th.nn.utils.rnn.pad_packed_sequence(x, batch_first=False, padding_value=0.0)
        # print(torch.exp((-nn.ReLU()(self.out(X_unpack[0]))).sum(dim=0)))
        # print(X_unpack[0][:-1].shape)

        if init_h is None:
            h0 = th.zeros(1, batch_size, self.nb_gru_units).to(utils.device)  # self.batch, Not batch_size
        else:
            h0 = init_h[-1:]

        # combinedH = th.cat((h0, x_unpack[0][:-1]), 0)
        h_t_vector = th.cat((h0, x_unpack[0][:-1]), 0)
        h_tplusone_vector = x_unpack[0]

        # sa_embed = self.sa_embedding(x_clone)
        # combinedSAH = th.cat((sa_embed, combinedH), -1)
        combined_two_h = th.cat((h_t_vector, h_tplusone_vector), -1)

        # log_scores, log_scores_mean, log_scores_variance = self._calc_logscores(combinedSAH)
        log_scores, log_scores_mean, log_scores_variance = self._calc_logscores(combined_two_h)

        y_hat = th.exp(log_scores.sum(dim=0)).to(utils.device)
        # meanH = X_unpack[0].sum(dim=0) / X_unpack[1][:, None].to(device)
        # y_hat = self.class_output(meanH)

        # predicted probability, log C with shape [T, B, 1] (mean and variance)
        return y_hat, {'log_scores': log_scores, 'mean': log_scores_mean, 'var': log_scores_variance}, x_unpack[0], self.hidden

    def forward_loss_metrics(self, x, y, x_lengths, classweight=1.):

        # forward + backward + optimize
        outputs, dict_log_c_out, h_out, _ = self(x, x_lengths)
        loss = self.loss(outputs, y.float(), classweight)

        y = y.bool()
        correct = ((outputs > 0.5) == y).sum().item()
        tp = ((outputs > 0.5) & y).sum().item()
        fp = ((outputs > 0.5) & y.logical_not()).sum().item()
        tn = ((outputs <= 0.5) & y.logical_not()).sum().item()
        fn = ((outputs <= 0.5) & y).sum().item()

        return loss, correct, tp, fp, tn, fn

    def _calc_logscores(self, concat_two_h):
        log_scores = -nn.ReLU()(self.decoder_output(self.decoder(concat_two_h)))
        # return log_scores, mean, variance
        return th.clamp(log_scores, MIN_LOGSCORE, 0), None, None

    @staticmethod
    def loss(y_hat, y, classweight=1.):

        # loss = nn.BCELoss()
        # loss = loss(y_hat, y)
        #
        # return loss

        w0 = classweight / (classweight + 1.0)
        w1 = 1.0 - w0
        weights = y * w1 + (1.0 - y) * w0
        weights = weights / weights.mean()
        return F.binary_cross_entropy(y_hat, y, weight=weights, reduction='mean')

        # bce_loss = nn.BCELoss(reduction='none')
        # interim_loss = bce_loss(y_hat, y)
        # # weights = torch.ones_like(y) + (y == 0).float()
        # class_zero_weight = classweight / (classweight + 1)
        # class_one_weight = 1 - class_zero_weight
        # weights = th.zeros_like(y) + class_zero_weight * (y == 0) + class_one_weight * (y == 1)

        # print("y")
        # print(y)
        # print("weights")
        # print(weights)

        return th.mean(2 * weights * interim_loss)


class DistributionGRU(PtEstGRU):
    def __init__(self, feature_dim=11, nb_gru_units=16, batch_size=256, gru_layers=2, mlp_arch=None, dropout=0.0,
                 loc_offset=0.0, log_std_offset=0.0):
        super().__init__(feature_dim, nb_gru_units, batch_size, gru_layers, mlp_arch, dropout)
        # self.decoder_output = nn.Linear(self.mlp_arch[-1], 2)
        self.decoder_output_logstd = nn.Linear(self.mlp_arch[-1], 1)
        self.loc_offset = loc_offset
        self.log_std_offset = log_std_offset

    # def __build_model(self):
    #     print("Dist GRU build model")
    #     super().__build_model()
    #     self.decoder_output = nn.Linear(256, 2)
    #     print("Completed dist GRU build model")

    def _calc_logscores(self, concat_two_h):
        # [T, B, 2]
        loc_params = self.decoder_output(self.decoder(concat_two_h)) - self.loc_offset
        loc_params = th.clamp(loc_params, LOC_MIN, LOC_MAX)

        log_std_params = self.decoder_output_logstd(self.decoder(concat_two_h)) - self.log_std_offset
        log_std_params = th.clamp(log_std_params, LOG_STD_MIN, LOG_STD_MAX)
        score_std = th.ones_like(loc_params) * log_std_params.exp()
        distributions = LogNormal(loc_params, score_std)
        log_scores = -distributions.rsample()  # [T, B]

        # return th.clamp(log_scores.unsqueeze(-1), MIN_LOGSCORE, 0), -distributions.mean, distributions.variance
        return th.clamp(log_scores, MIN_LOGSCORE, 0), -distributions.mean, distributions.variance


class CostBudgetEstMLP(nn.Module):
    def __init__(self, feature_dim=11, mlp_arch=None, dropout=0.0):
        super().__init__()
        if mlp_arch is None:
            mlp_arch = [32, 32]
            # mlp_arch = [16, 16]
        self.feature_dim = feature_dim
        self.mlp_arch = mlp_arch
        self.dropout = dropout

        # build actual NN
        self.__build_model()

    def __build_model(self):

        self.log_budget = nn.Parameter(th.ones(1) * 1, requires_grad=True)

        mlp_log_cost = []
        prev_in_features = self.feature_dim
        for out_features in self.mlp_arch:
            mlp_log_cost.append(nn.Linear(prev_in_features, out_features))
            mlp_log_cost.append(nn.ReLU())
            mlp_log_cost.append(nn.LayerNorm(out_features))
            mlp_log_cost.append(nn.Dropout(self.dropout))
            prev_in_features = out_features
        self.mlp_log_cost = nn.Sequential(*mlp_log_cost)
        self.mlp_log_cost_output = nn.Linear(self.mlp_arch[-1], 1)

    def forward(self, x) -> Tuple[th.Tensor, th.Tensor]:

        # [B, T, SA]
        for i, l in enumerate(self.mlp_log_cost):
            x = l(x)

        x = self.mlp_log_cost_output(x)

        return th.exp(x), th.exp(self.log_budget)

    def forward_loss_metrics(self, x, y, x_lengths=None, classweight=1.0):

        # forward + backward + optimize
        costs_bt, budget = self(x)
        batch_costs = costs_bt.sum(dim=1)
        loss = self.loss(budget - batch_costs, y.float())

        y = y.bool()
        correct = ((th.nn.Sigmoid()(budget - batch_costs) > 0.5) == y).sum().item()
        tp = ((th.nn.Sigmoid()(budget - batch_costs) > 0.5) & y).sum().item()
        fp = ((th.nn.Sigmoid()(budget - batch_costs) > 0.5) & y.logical_not()).sum().item()
        tn = ((th.nn.Sigmoid()(budget - batch_costs) <= 0.5) & y.logical_not()).sum().item()
        fn = ((th.nn.Sigmoid()(budget - batch_costs) <= 0.5) & y).sum().item()

        return loss, correct, tp, fp, tn, fn

    @staticmethod
    def loss(y_logits, y):

        loss = nn.BCEWithLogitsLoss()
        loss = loss(y_logits, y)

        return loss


class RLSFMLP(nn.Module):
    def __init__(self, feature_dim=11, mlp_arch=None, dropout=0.0):
        super().__init__()
        if mlp_arch is None:
            mlp_arch = [32, 32]
            # mlp_arch = [16, 16]
        self.feature_dim = feature_dim
        self.mlp_arch = mlp_arch
        self.dropout = dropout

        # build actual NN
        self.__build_model()

    def __build_model(self):

        mlp_logit_safe = []
        prev_in_features = self.feature_dim
        for out_features in self.mlp_arch:
            mlp_logit_safe.append(nn.Linear(prev_in_features, out_features))
            mlp_logit_safe.append(nn.ReLU())
            mlp_logit_safe.append(nn.LayerNorm(out_features))
            mlp_logit_safe.append(nn.Dropout(self.dropout))
            prev_in_features = out_features
        self.mlp_logit_safe = nn.Sequential(*mlp_logit_safe)
        self.mlp_logit_safe_output = nn.Linear(self.mlp_arch[-1], 1)

    def forward(self, x) -> th.Tensor:

        # [B, T, SA]
        for i, l in enumerate(self.mlp_logit_safe):
            x = l(x)

        x = self.mlp_logit_safe_output(x)  # [B, T, 1]

        return x

    def forward_loss_metrics(self, x, y, x_lengths=None, classweight=1.0):

        # forward + backward + optimize
        safety_logits_bt = self(x)  # [B, T, 1]
        safety_logits = safety_logits_bt.reshape(-1, safety_logits_bt.size(2))  # [B * T, 1]
        y = y.repeat(1, safety_logits_bt.size(1))  # [B, T]
        y = y.reshape(-1, 1)  # Shape: [B * T, 1]
        loss = self.loss(safety_logits, y.float())

        y = y.bool()
        correct = ((th.nn.Sigmoid()(safety_logits) > 0.5) == y).sum().item()
        tp = ((th.nn.Sigmoid()(safety_logits) > 0.5) & y).sum().item()
        fp = ((th.nn.Sigmoid()(safety_logits) > 0.5) & y.logical_not()).sum().item()
        tn = ((th.nn.Sigmoid()(safety_logits) <= 0.5) & y.logical_not()).sum().item()
        fn = ((th.nn.Sigmoid()(safety_logits) <= 0.5) & y).sum().item()

        return loss, correct, tp, fp, tn, fn

    @staticmethod
    def loss(y_logits, y):

        loss = nn.BCEWithLogitsLoss()
        loss = loss(y_logits, y)

        return loss


class CostBudgetHMLP(CostBudgetEstMLP):
    def __init__(self, feature_dim=11, mlp_arch=None, decoder_arch=None, dropout=0.0):
        super().__init__(feature_dim=feature_dim, mlp_arch=mlp_arch, dropout=dropout)
        if decoder_arch is None:
            decoder_arch = [32, 32]
            # decoder_arch = [16, 16]
        self.decoder_arch = decoder_arch

        # Double underscore method does not get overriden
        self.__build_model()

    def __build_model(self):

        self.log_temperature = nn.Parameter(th.ones(1) * 1, requires_grad=True)
        # # self.sa_embedding = nn.Identity()
        # decoder = []
        # # prev_in_features = self.feature_dim + 1
        # prev_in_features = 1 + 1
        # for out_features in self.decoder_arch:
        #     decoder.append(nn.Linear(prev_in_features, out_features))
        #     decoder.append(nn.ReLU())
        #     decoder.append(nn.LayerNorm(out_features))
        #     prev_in_features = out_features
        # self.decoder = nn.Sequential(*decoder)
        #
        # self.decoder_output = nn.Linear(self.decoder_arch[-1], 1)


    def forward(self, sa):

        budget = th.exp(self.log_budget)
        temperature = th.exp(self.log_temperature)

        # [B, T, SA]
        x = sa.clone()
        # [B, T, 1]
        costs_t = th.exp(self.mlp_log_cost_output(self.mlp_log_cost(x)))
        # [B, T, 1]
        costs_cumsum_t = th.cumsum(costs_t, dim=1)
        # [B, T, 1]
        budget_remaining_t = (budget - costs_cumsum_t) / temperature

        budget_t = th.cat((th.ones_like(budget_remaining_t[:, :1]) * float('Inf'),  # budget,
                           budget_remaining_t.clone()[:, :-1]), dim=1)

        # # sa_embed = self.sa_embedding(sa)
        # # concat_sah = th.cat((sa_embed, budget_remaining_t), -1)
        concat_two_h = th.cat((budget_t, budget_remaining_t), -1)

        # [B, T, SAH]
        # # log_scores, log_scores_mean, log_scores_variance = self._calc_logscores(concat_sah)
        # [B, T, 2]
        log_scores, log_scores_mean, log_scores_variance = self._calc_logscores(concat_two_h)
        # log_scores, log_scores_mean, log_scores_variance = self._calc_logscores(concat_two_h / temperature)

        # [B, T, 1] which means Time is at dim=1
        y_hat = th.exp(log_scores.sum(dim=1)).to(utils.device)

        # predicted probability, log C with shape [B, T, 1] (mean and variance), remaining budget [B, T, 1]
        return y_hat, {'log_scores': log_scores, 'mean': log_scores_mean, 'var': log_scores_variance}, budget_remaining_t

    def _calc_logscores(self, concat_two_h):
        # log_scores = -nn.ReLU()(self.decoder_output(self.decoder(concat_sah)))

        # print(concat_two_h)
        # print(nn.Sigmoid()(concat_two_h))
        # [B, T, 2]
        log_sigmoid_two_h = th.log(th.clamp(nn.Sigmoid()(concat_two_h), min=utils.epsilon))
        # print(log_sigmoid_two_h)
        log_scores = log_sigmoid_two_h[:, :, 1:] - log_sigmoid_two_h[:, :, :1]
        # print(log_scores)
        # return log_scores, mean, variance
        return th.clamp(log_scores, MIN_LOGSCORE, 0), None, None

    @staticmethod
    def loss(y_hat, y, classweight=1.):

        loss = nn.BCELoss()
        loss = loss(y_hat, y)

        return loss

class DistributionCostBudgetHMLP(CostBudgetHMLP):
    def __init__(self, feature_dim=11, mlp_arch=None, decoder_arch=None, dropout=0.0, loc_offset=0.0, log_std_offset=0.0):
        super().__init__(feature_dim=feature_dim, mlp_arch=mlp_arch, decoder_arch=decoder_arch, dropout=dropout)
        self.loc_offset = loc_offset
        self.log_std_offset = log_std_offset

        self.__build_model()

    def __build_model(self):

        # self.decoder_output = nn.Linear(self.decoder_arch[-1], 2)
        decoder = []
        # prev_in_features = self.feature_dim + 1
        prev_in_features = 1 + 1
        for out_features in self.decoder_arch:
            decoder.append(nn.Linear(prev_in_features, out_features))
            decoder.append(nn.ReLU())
            decoder.append(nn.LayerNorm(out_features))
            prev_in_features = out_features
        self.decoder = nn.Sequential(*decoder)

        self.decoder_output = nn.Linear(self.decoder_arch[-1], 2)

    def _calc_logscores(self, concat_two_h):

        log_sigmoid_two_h = th.log(th.clamp(nn.Sigmoid()(concat_two_h), min=utils.epsilon))

        # [B, T, 2]
        # distri_params = self.decoder_output(self.decoder(concat_two_h))
        distri_params = self.decoder_output(self.decoder(log_sigmoid_two_h))
        loc_params = distri_params[:, :, 0] - self.loc_offset
        # loc_params = distri_params[:, :, 0] - 3.0
        loc_params = th.clamp(loc_params, LOC_MIN, LOC_MAX)
        log_std_params = distri_params[:, :, 1] - self.log_std_offset
        # log_std_params = distri_params[:, :, 1] - 0.0
        log_std_params = th.clamp(log_std_params, LOG_STD_MIN, LOG_STD_MAX)
        # [B, T]
        score_std = th.ones_like(loc_params) * log_std_params.exp()
        distributions = LogNormal(loc_params, score_std)
        # Negative lognormal, sample * -1.0
        log_scores = -distributions.rsample()  # [B, T]

        return th.clamp(log_scores.unsqueeze(-1), MIN_LOGSCORE, 0), -distributions.mean, distributions.variance


classifier_nw_class = {'DistributionGRU': DistributionGRU,
                       'GRU': PtEstGRU,
                       'CostBudgetMLP': CostBudgetEstMLP,
                       'DistributionBudgetH': DistributionCostBudgetHMLP,
                       'BudgetH': CostBudgetHMLP,
                       'RLSF': RLSFMLP}
