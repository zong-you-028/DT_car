import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size=128,
        max_length=20,
        action_tanh=True,
        num_styles=3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=3,
            n_head=1,
            n_positions=3 * max_length
        )
        self.transformer = GPT2Model(config)

        # self.embed_timestep = nn.Embedding(max_length, hidden_size)
        self.embed_timestep = nn.Embedding(2907, hidden_size)

        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)  # 若為離散 action，請 one-hot

        self.embed_style = nn.Embedding(num_styles, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() if action_tanh else nn.Identity()
        )

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, style_ids=None):

        # print("timesteps shape:", timesteps.shape)
        # print("timesteps max:", timesteps.max().item())
        # print("embed_timestep size:", self.embed_timestep.num_embeddings)

        # states: [B, T, state_dim]
        # actions: [B, T, act_dim]
        # returns_to_go: [B, T, 1]
        # timesteps: [B, T]
        # style_ids: [B]

        B, T = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long).to(states.device)
        # print("timesteps:", timesteps)
        # print("max timestep embedding size:", self.embed_timestep.num_embeddings)
        time_emb = self.embed_timestep(timesteps)
        style_emb = self.embed_style(style_ids).unsqueeze(1)  # [B,1,H]

        state_embeddings = self.embed_state(states) + time_emb + style_emb
        action_embeddings = self.embed_action(actions) + time_emb + style_emb
        return_embeddings = self.embed_return(returns_to_go) + time_emb + style_emb

        stacked_inputs = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_mask = attention_mask.repeat_interleave(3, dim=1)

        outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_mask
        )
        x = outputs.last_hidden_state  # [B, 3T, H]
        x = x.view(B, T, 3, self.hidden_size).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:, 1])  # [B, T, act_dim]
        return action_preds
