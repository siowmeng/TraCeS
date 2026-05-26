import os
os.environ["MUJOCO_GL"] = "egl"
import safety_gymnasium
import cv2
import numpy as np

env = safety_gymnasium.make("SafetyHalfCheetahVelocity-v1", max_episode_steps=200, render_mode="rgb_array")
env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

episodes = 5
fps = 30  # You control this now

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    frames = []

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is None:
            print("Render returned None")
        frames.append(frame)
        done = terminated or truncated

    if not frames:
        print(f"Warning: No frames for episode {ep}")
        continue

    height, width, _ = frames[0].shape
    os.makedirs("videos", exist_ok=True)
    out = cv2.VideoWriter(f"videos/trajectory_{ep}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    print("Writing to path:", os.path.abspath(f"videos/trajectory_{ep}.mp4"))

    for frame in frames:
        # Convert RGB (from MuJoCo) to BGR (for OpenCV)
        # out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.write(frame)  # Temporarily skip COLOR_RGB2BGR

    out.release()
    print(f"Saved trajectory_{ep}.mp4")

# import os
# os.environ["MUJOCO_GL"] = "egl"
#
# import safety_gymnasium
# import cv2
# import numpy as np
#
# os.makedirs("videos", exist_ok=True)
#
# env = safety_gymnasium.make("SafetyHalfCheetahVelocity-v1", max_episode_steps=200, render_mode="rgb_array")
# env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
#
# obs, _ = env.reset()
# done = False
# frames = []
#
# while not done:
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     frame = env.render()
#     if frame is None:
#         print("Frame is None!")
#     else:
#         print("Captured frame:", frame.shape)
#         frames.append(frame)
#     done = terminated or truncated
#
# if frames:
#     h, w, _ = frames[0].shape
#     path = "videos/debug_out.avi"
#     out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
#     print("VideoWriter opened:", out.isOpened())
#     print("Writing to path:", os.path.abspath(path))
#     for f in frames:
#         out.write(f)
#     out.release()
#     print("Wrote video to:", path)
# else:
#     print("No frames captured.")