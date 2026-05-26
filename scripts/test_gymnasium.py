import numpy as np
import safety_gymnasium

from typing import Any

from gymnasium import Env
from gymnasium.envs.registration import EnvSpec

# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecNormalize

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

env_id = 'SafetyHalfCheetahVelocity-v1'
env_kwargs = {'id': env_id}
env = safety_gymnasium.vector.make(env_id, max_episode_steps=200, num_envs=2)#, render_mode='human')
env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

# import gymnasium
# import bullet_safety_gym
# env = gymnasium.vector.make(env_id, num_envs=2)
print(env.action_space)
# env = make_vec_env(make_safety_gymnasium_env, n_envs=2, env_kwargs=env_kwargs)

# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.,
#                    gamma=0.99)

obs, info = env.reset()
i = 0
while True:
    act = env.action_space.sample()
    # act1 = env.action_space.sample()
    # act2 = env.action_space.sample()
    # act = np.vstack((act1, act2))
    # obs, reward, cost, terminated, truncated, info = env.step(act)
    obs, reward, terminated, truncated, info = env.step(act)
    i += 1
    # obs, reward, done, info = env.step(act)
    # print(obs)
    # print(act)
    # print(reward)
    print(info['cost'])
    print("Terminated", terminated)
    print("Truncated", truncated)
    if terminated.sum() > 1 or truncated.sum() > 1:
        print(obs)
        print(info)
        break
    # print(done)
    # if done.all():
    #     break
    # env.render()
print(terminated)
print(truncated)
print(i)
