#%%
# trainer_and_collector.py

import os
import gymnasium as gym
import numpy as np
import torch
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_attention import CustomExtractor, attention_network_kwargs  # 需引入你原來的注意力模型
from driver_envs import register_custom_envs

register_custom_envs()  # 保證在 subprocess 建立前執行


def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"])
    env.configure(kwargs["config"])
    env.reset()
    return env


def train_driver(env_id, model_path):
    env_kwargs = {
        'id': env_id,
        'config': {
            "lanes_count": 3,
            "vehicles_count": 15,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": [
                    "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"
                ],
                "absolute": False
            },
            "policy_frequency": 2,
            "duration": 120,
        }
    }

    policy_kwargs = dict(
        features_extractor_class=CustomExtractor,
        features_extractor_kwargs=attention_network_kwargs,
    )

    env = make_vec_env(make_configure_env, n_envs=8, seed=0,
                       vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    model = PPO("MlpPolicy", env,
                n_steps=512 // 8,
                batch_size=64,
                learning_rate=2e-3,
                policy_kwargs=policy_kwargs,
                verbose=2, device="cuda")

    model.learn(total_timesteps=10000)
    model.save(model_path)


def convert_to_trajectories(raw_data, style_name):
    trajectories = []
    for ep in raw_data:
        obs_seq, act_seq, rew_seq = [], [], []
        for obs, act, rew in ep:
            obs_seq.append(obs)
            act_seq.append(act)
            rew_seq.append(rew)

        rtg = []
        total = 0
        for r in reversed(rew_seq):
            total += r
            rtg.insert(0, total)

        traj = {
            "observations": np.array(obs_seq),
            "actions": np.array(act_seq),
            "rewards": np.array(rew_seq),
            "returns_to_go": np.array(rtg),
            "timesteps": np.arange(len(obs_seq)),
            "style": style_name
        }
        trajectories.append(traj)
    return trajectories


def generate_dataset(model_path, env_id, style_name, save_path):
    model = PPO.load(model_path)
    env_config = {
        "lanes_count": 3,
        "vehicles_count": 15,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": [
                "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"
            ],
            "absolute": False
        },
        "policy_frequency": 2,
        "duration": 120,
    }
    env = make_configure_env(id=env_id, config=env_config)

    all_data = []
    for ep in range(5):
        obs, info = env.reset()
        done = False
        ep_data = []
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            ep_data.append([obs, action, reward])
        all_data.append(ep_data)

    trajs = convert_to_trajectories(all_data, style_name)
    with open(save_path, "wb") as f:
        pickle.dump(trajs, f)


if __name__ == "__main__":
    # register_custom_envs()

    os.makedirs("models", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    configs = [
        ("highway-aggressive-v0", "aggressive"),
        ("highway-v0", "normal"),
        ("highway-cautious-v0", "cautious"),
    ]

    for env_id, style in configs:
        print(f"Training {style} driver...")
        train_driver(env_id, f"models/ppo_{style}.zip")
        print(f"Generating dataset for {style}...")
        generate_dataset(f"models/ppo_{style}.zip", env_id, style, f"datasets/{style}_traj.pkl")

# %%
