import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

STYLE2ID = {
    "aggressive": 0,
    "normal": 1,
    "cautious": 2
}

class HighwayDTDataset(Dataset):
    def __init__(self, paths, context_len=20):
        self.data = []
        for p in paths:
            with open(p, 'rb') as f:
                self.data += pickle.load(f)
        self.context_len = context_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        obs = traj['observations']       # shape: [L, state_dim]
        acts = traj['actions']           # shape: [L]
        rtg = traj['returns_to_go']      # shape: [L]
        ts = traj['timesteps']           # shape: [L]
        style = STYLE2ID[traj['style']]
        L = len(obs)

        si = np.random.randint(0, max(1, L))
        s = slice(max(0, si - self.context_len + 1), si + 1)

        # 預防 obs 太短或為空，強制 reshape
        obs_s = obs[s]
        if obs_s.ndim == 1:
            obs_s = obs_s.reshape(1, -1)
        if len(obs_s) == 0:
            obs_s = np.zeros((0, self.data[0]['observations'].shape[-1]))

        act_s = acts[s] if len(acts[s]) > 0 else np.zeros((0,), dtype=np.int64)
        rtg_s = rtg[s] if len(rtg[s]) > 0 else np.zeros((0,))
        ts_s = ts[s] if len(ts[s]) > 0 else np.zeros((0,))

        pad_len = self.context_len - len(obs_s)

        if obs_s.ndim == 1:
            obs_s = obs_s.reshape(1, -1)  # (1, D)
        elif obs_s.ndim == 3:
            obs_s = obs_s.reshape(obs_s.shape[0], -1)  # 展平成 2D

        obs_s = np.pad(obs_s, ((pad_len, 0), (0, 0)), mode='constant')        
        act_s = np.pad(act_s, (pad_len, 0), mode='constant')
        rtg_s = np.pad(rtg_s, (pad_len, 0), mode='constant')
        ts_s = np.pad(ts_s, (pad_len, 0), mode='constant')

        return {
            'obs': torch.tensor(obs_s, dtype=torch.float),
            'actions': torch.tensor(act_s, dtype=torch.long),
            'returns_to_go': torch.tensor(rtg_s, dtype=torch.float),
            'timesteps': torch.tensor(ts_s, dtype=torch.long),
            'style': torch.tensor(style, dtype=torch.long)
        }
