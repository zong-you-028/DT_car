import torch
from dt_dataset import HighwayDTDataset
from decision_transformer import DecisionTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 載入資料集 ===
dataset = HighwayDTDataset(
    paths=r"C:\Users\ai\Desktop\DT\datasets\datasets\aggressive_traj.pkl",  # 注意：這是相對於 eval_DT.py 的路徑
    context_len=20
)


# === 初始化模型 ===
sample_obs = dataset[0]['obs']
state_dim = sample_obs.shape[-1]
act_dim = 5  # 離散動作數量

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    hidden_size=128,
    max_length=20,
    action_tanh=False,
    num_styles=3
).to(device)

# === 載入訓練模型 ===
model.load_state_dict(torch.load('models/dt_model_style.pth'))
model.eval()

# Step 2: 載入測試資料
dataset = HighwayDTDataset(...)  # 使用不同的 style or test set
sample = dataset[0]  # or 隨機抽樣

states = sample['obs'].unsqueeze(0).cuda()          # (1, T, state_dim)
actions = sample['action'].unsqueeze(0).cuda()      # (1, T, act_dim)
returns = sample['return'].unsqueeze(0).cuda()      # (1, T, 1)
timesteps = torch.arange(states.shape[1], device='cuda').unsqueeze(0)

# Step 3: 推論
with torch.no_grad():
    pred_actions = model(states, actions, returns, timesteps)

# Step 4: 視覺化
import matplotlib.pyplot as plt
plt.plot(actions[0, :, 0].cpu(), label='GT steer')
plt.plot(pred_actions[0, :, 0].cpu(), label='Pred steer')
plt.legend()
plt.show()
