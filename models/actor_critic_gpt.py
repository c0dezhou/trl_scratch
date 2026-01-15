from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from core.gpt_decoder_core import GPTDecoderCore

class ActorCriticGPT(nn.Module):
    """
    PPO 用的 Transformer policy（decoder-only backbone）

    训练脚本给它的 obs 是 flatten 后的：
      obs_flat: [T*D] 或 [B, T*D]
    model 内部 reshape -> [B,T,D] 再送进 core。
    """
    def __init__(
        self,
        obs_dim: int,        # 单步 obs 维度 D（POMDP 下 D=2）
        act_dim: int,
        history_len: int,    # T
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.history_len = history_len

        self.obs_emb = nn.Linear(obs_dim, d_model)
        
        self.core = GPTDecoderCore(
            max_len=history_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.actor = nn.Linear(d_model, act_dim)
        self.critic = nn.Linear(d_model, 1)

        self._init()
    
    def _init(self):
        """
        正交初始化 (Orthogonal Initialization)：这是强化学习的关键技巧。它让权重的每一行彼此垂直。
            作用：在训练刚开始时，这种初始化能保持信号的强度（方差）不变，防止梯度消失或爆炸。
            对 RL 的意义：它让模型在最开始的“瞎试”阶段表现得更稳定，更容易收敛。
        偏置归零：让模型初期处于中立状态。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0) # 正交初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0) # 偏置归零
    
    # 把“拍扁的历史数据”还原成序列
    # 环境吐出来的是 T*D（时间 X 观测维度），但在 Buffer 里通常存成了一个长向量。
    def _flat_to_seq(self, obs_flat: torch.Tensor):
        T, D = self.history_len, self.obs_dim
        flat = T * D

        single = False
        if obs_flat.dim() == 1:
            obs_flat = obs_flat.unsqueeze(0)
            single = True

        if obs_flat.dim() != 2 or obs_flat.shape[1] != flat:
            raise ValueError(f"Expected obs shape [B,{flat}] or [{flat}], got {tuple(obs_flat.shape)}")

        obs_seq = obs_flat.view(obs_flat.shape[0], T, D)  # [B,T,D]
        return obs_seq, single
    
    # Transformer 的特征提取
    def forward(self, obs_flat: torch.Tensor):
        obs_seq, single = self._flat_to_seq(obs_flat)     # [B,T,D]
        x_emb = self.obs_emb(obs_seq)                     # 映射到高维空间 [B,T,C]
        h = self.core(x_emb)                              # 喂给 GPT Core（进行 Attention 计算）[B,T,C]

        last = h[:, -1, :]                                # 关键：只取最后一帧的信息 [B,C]
        # 因为 Transformer 的每一层都有 Causal Mask（因果掩码）。
        # 最后一帧（Index -1）的隐藏状态已经通过注意力机制融合了之前所有帧的信息。
        # 它现在代表了“包含了历史背景的当前状态”。
        logits = self.actor(last)                         # 输出动作概率分布（Logits）。[B,act_dim]
        value = self.critic(last).squeeze(-1)             # 输出对当前局势的评分（Value）.[B]

        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value

    # 实战采样
    def act(self, obs_flat: torch.Tensor):
        logits, value = self(obs_flat)
        #  Logits 转 Distribution
        dist = Categorical(logits=logits)
        # 采样 (Sample)：根据概率“掷骰子”选出一个动作。
        action = dist.sample()
        # Log Prob：记录下选这个动作的概率（用于 PPO 以后计算pi_new/pi_old 的ratio）
        logp = dist.log_prob(action)
        return action, logp, value