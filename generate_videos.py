import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import highway_env

from DT_car.driver_envs import register_custom_envs


def generate_video(env_id: str, model_path: str, video_dir: str, name_prefix: str):
    """Run one episode using a PPO policy and save the video."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_dir, name_prefix=name_prefix,
                      episode_trigger=lambda e: True)
    model = PPO.load(model_path)

    obs, info = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, _, done, truncated, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    register_custom_envs()
    os.makedirs("videos", exist_ok=True)

    configs = [
        ("highway-aggressive-v0", "models/ppo_aggressive.zip", "aggressive"),
        ("highway-v0", "models/ppo_normal.zip", "normal"),
        ("highway-cautious-v0", "models/ppo_cautious.zip", "cautious"),
    ]

    for env_id, model_path, prefix in configs:
        print(f"Generating video for {prefix} style...")
        generate_video(env_id, model_path, "videos", prefix)
    print("Videos saved to ./videos")
