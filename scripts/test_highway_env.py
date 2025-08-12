# import numpy as np
# import safety_gymnasium
#
# from typing import Any
#
# from gymnasium import Env
# from gymnasium.envs.registration import EnvSpec
#
# def make_safety_gymnasium_env(
#         id: str | EnvSpec,  # pylint: disable=invalid-name,redefined-builtin
#         max_episode_steps: int | None = None,
#         autoreset: bool | None = None,
#         apply_api_compatibility: bool | None = None,
#         disable_env_checker: bool | None = None,
#         **kwargs: Any,) -> Env:
#
#     safe_env = safety_gymnasium.make(id, max_episode_steps, autoreset, apply_api_compatibility, disable_env_checker, **kwargs)
#     env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safe_env)
#     return env
#
# env_id = 'SafetyHalfCheetahVelocity-v1'
# env_kwargs = {'id': env_id}
# env = safety_gymnasium.vector.make(env_id, max_episode_steps=200, num_envs=2)#, render_mode='human')
# env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
#
# print(env.action_space)
#
# obs, info = env.reset()
# i = 0
# while True:
#     act = env.action_space.sample()
#     # act1 = env.action_space.sample()
#     # act2 = env.action_space.sample()
#     # act = np.vstack((act1, act2))
#     # obs, reward, cost, terminated, truncated, info = env.step(act)
#     obs, reward, terminated, truncated, info = env.step(act)
#     i += 1
#     # obs, reward, done, info = env.step(act)
#     # print(obs)
#     # print(act)
#     # print(reward)
#     print(info['cost'])
#     print("Terminated", terminated)
#     print("Truncated", truncated)
#     if terminated.sum() > 1 or truncated.sum() > 1:
#         print(obs)
#         print(info)
#         break
#     # print(done)
#     # if done.all():
#     #     break
#     # env.render()
# print(terminated)
# print(truncated)
# print(i)

import gymnasium
import highway_env
from gymnasium import ObservationWrapper, spaces
import torch as th
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
# os.environ["SDL_VIDEODRIVER"] = "dummy"


class DictObsToBoxWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Get sizes of all dict parts
        keys = ['achieved_goal', 'desired_goal', 'observation']
        self._obs_keys = keys
        size = sum(self.env.observation_space[k].shape[0] for k in keys)
        # New flat Box space
        low = np.concatenate([self.env.observation_space[k].low for k in keys])
        high = np.concatenate([self.env.observation_space[k].high for k in keys])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        # Concatenate the three parts in fixed order
        flat = np.concatenate([obs[k] for k in self._obs_keys]).astype(np.float32)
        return flat


class FlattenObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.prod(original_shape),),
            dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


env = gymnasium.make('parking-v0', render_mode='rgb_array', config={
    "action": {
        "type": "ContinuousAction"
    }
})
# env = gymnasium.make('highway-v0', autoreset=True, render_mode='rgb_array')
# env = FlattenObservation(env)
env = DictObsToBoxWrapper(env)
env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
frames = []
i = 0
while True:
    # action = env.unwrapped.action_type.actions_indexes["IDLE"]
    action = env.action_space.sample()
    # action = th.tensor([0., 0.])
    obs, reward, done, truncated, info = env.step(action)
    # print("Obs", obs)
    # print("Action", action)
    # print("Reward", reward)
    # print("Done", done)
    # print("Truncated", truncated)
    # print("Info", info)
    frames.append(env.render())
    i += 1
    if done or truncated:
        break

print(i)
print("Info", info)

height, width, _ = frames[0].shape
# os.makedirs(f"videos/epoch{epoch:03d}", exist_ok=True)
os.makedirs(f"videos", exist_ok=True)
print("Writing to path:", os.path.abspath(f"videos/trajectory_{i:03d}.mp4"))
out = cv2.VideoWriter(f"videos/trajectory_{i:03d}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

for frame in frames:
    # Convert RGB (from MuJoCo) to BGR (for OpenCV)
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
print(f"Saved trajectory_{i}.mp4")


# plt.imshow(env.render())
# plt.show()
