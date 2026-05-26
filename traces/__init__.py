import sys

import safety_gymnasium

# Backward compatibility for torch-pickled datasets/checkpoints saved before the
# public package rename.  Old files may reference classes under ``ubcrl.*``.
sys.modules.setdefault('ubcrl', sys.modules[__name__])

from traces import algorithms
from traces.algorithms import ALGORITHMS
from traces.algorithms.algo_wrapper import TraCeSAlgoWrapper as Agent
from traces.evaluator import TraCeSEvaluator

safety_gymnasium.__register_helper(
    env_id='SafetyHopperVelocityWindowAvg-v0',
    entry_point='traces.tasks.window_velocity.safety_hopper_velocity_window_avg_v0:SafetyHopperVelocityWindowAvgEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
