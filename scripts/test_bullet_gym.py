import gymnasium as gym
import bullet_safety_gym
from omnisafe.envs.core import CMDP, make
from ubcrl.envs import support_envs

env_id = 'SafetyAntRun-v0'
# env_kwargs = {'id': env_id}
env = gym.vector.make(env_id, num_envs=2, asynchronous=False)#, autoreset=True)#, render_mode='human')
# env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

# print("Reset", env.reset())
obs, info = env.reset()
act = env.action_space.sample()
print(env.step(act))
i = 0
while True:
    act = env.action_space.sample()
    # act1 = env.action_space.sample()
    # act2 = env.action_space.sample()
    # act = np.vstack((act1, act2))
    obs, reward, terminated, truncated, info = env.step(act)
    print(obs)
    i += 1
    # obs, reward, done, info = env.step(act)
    # print(obs)
    # print(act)
    # print(reward)
    print(type(info['cost']))
    print("Terminated", terminated)
    print("Truncated", truncated)
    if terminated or truncated:
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
obs, reward, terminated, truncated, info = env.step(act)
print(obs)
