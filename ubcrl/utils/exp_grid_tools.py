from __future__ import annotations

import os
import sys
from typing import Any

import ubcrl
from omnisafe.typing import Tuple


def train(
    exp_id: str,
    algo: str,
    env_id: str,
    custom_cfgs: dict[str, Any],
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (Config): Custom configurations.
    """
    terminal_log_name = 'terminal.log'
    error_log_name = 'error.log'
    if 'seed' in custom_cfgs:
        terminal_log_name = f'seed{custom_cfgs["seed"]}_{terminal_log_name}'
        error_log_name = f'seed{custom_cfgs["seed"]}_{error_log_name}'
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    if not os.path.exists(custom_cfgs['logger_cfgs']['log_dir']):
        os.makedirs(custom_cfgs['logger_cfgs']['log_dir'], exist_ok=True)
    with open(
        os.path.join(
            f'{custom_cfgs["logger_cfgs"]["log_dir"]}',
            terminal_log_name,
        ),
        'w',
        encoding='utf-8',
    ) as f_out:
        sys.stdout = f_out
        with open(
            os.path.join(
                f'{custom_cfgs["logger_cfgs"]["log_dir"]}',
                error_log_name,
            ),
            'w',
            encoding='utf-8',
        ) as f_error:
            sys.stderr = f_error
            agent = ubcrl.Agent(algo, env_id, custom_cfgs=custom_cfgs)
            reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len
