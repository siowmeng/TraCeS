import safety_gymnasium
from ubcrl import algorithms
from ubcrl.algorithms import ALGORITHMS
from ubcrl.algorithms.algo_wrapper import UBCRLAlgoWrapper as Agent
from ubcrl.evaluator import UBCRLEvaluator

safety_gymnasium.__register_helper(
    env_id='SafetyHopperVelocityWindowAvg-v0',
    entry_point='ucrl.tasks.window_velocity.safety_hopper_velocity_window_avg_v0:SafetyHopperVelocityWindowAvgEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
