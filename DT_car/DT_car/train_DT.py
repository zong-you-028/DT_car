#%%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from decision_transformer import DecisionTransformer
from dt_dataset import HighwayDTDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 載入資料集 ===
dataset_paths = [
    r"C:\Users\ai\Desktop\DT\datasets\aggressive_traj.pkl",
    r"C:\Users\ai\Desktop\DT\datasets\normal_traj.pkl",
    r"C:\Users\ai\Desktop\DT\datasets\cautious_traj.pkl"
]
dataset = HighwayDTDataset(paths=dataset_paths, context_len=20)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#%%
# === 初始化模型 ===
sample_obs = dataset[0]['obs']
state_dim = sample_obs.shape[-1]
act_dim = 5  # highway-env 預設為 5 個離散動作
model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    hidden_size=128,
    max_length=20,
    action_tanh=False,
    num_styles=3
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === 訓練 ===
epochs = 10
for epoch in range(epochs):
    total_loss = 0                             
    for batch in dataloader:
        obs = batch['obs'].to(device)                         # [B, T, state_dim]
        actions = batch['actions'].to(device)                 # [B, T]
        returns_to_go = batch['returns_to_go'].unsqueeze(-1).to(device)  # [B, T, 1]
        timesteps = batch['timesteps'].to(device)             # [B, T]
        style_ids = batch['style'].to(device)                 # [B]

        # one-hot encode actions for input embedding
        actions_onehot = F.one_hot(actions, num_classes=act_dim).float()

        action_preds = model(
            obs, actions_onehot, None, returns_to_go, timesteps, style_ids=style_ids
        )

        # CrossEntropy loss: target is action index
        action_preds_flat = action_preds.reshape(-1, act_dim)
        actions_flat = actions.reshape(-1)
        loss = F.cross_entropy(action_preds_flat, actions_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# === 儲存模型 ===
torch.save(model.state_dict(), "models/dt_model_style.pth")
print("✅ 模型儲存為 models/dt_model_style.pth")
