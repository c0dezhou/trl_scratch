# DT 模型本体
# token 顺序固定为：(rtg_t, s_t, a_t) 反复拼起来
# 预测动作用 state token 的 hidden（也就是序列里第 2 个 token 位置）
# attention mask = causal mask + padding key mask（防止模型看见 padding）

from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.transformer_min import Block, LayerNorm

@dataclass
class DTBatch:
    # 前面加了个batch维
    states: torch.Tensor     # [B, K, state_dim] float
    actions: torch.Tensor    # [B, K] long
    rtg: torch.Tensor        # [B, K] float (已 scale)
    timesteps: torch.Tensor  # [B, K] long
    valid: torch.Tensor      # [B, K] float (0/1)

class DT(nn.Module):
    """最小可懂版本DT(离散动作)
    输入（长度 K 的窗口）：
      - states:    [B, K, state_dim]
      - actions:   [B, K]
      - rtg:       [B, K]
      - timesteps: [B, K]
      - valid:     [B, K]  (padding mask)

    序列拼接（每个 timestep 变 3 个 token）：
      [rtg_0, s_0, a_0,  rtg_1, s_1, a_1,  ...]  -> 总长度 L=3K

    输出：
      - action_logits: [B, K, act_dim]  （用每个 s_t 的 hidden 去预测 a_t）
    """
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        context_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_timestep: int = 512,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.K = int(context_len)
        self.d_model = int(d_model)
        self.max_timestep = int(max_timestep)

        #---不同类型token的embedding----
        # 连续信号用 Linear，离散idx用 Embedding
        # 把这几个量投影到高维空间提取特征
        self.state_emb = nn.Linear(self.state_dim,self.d_model)
        self.rtg_emb = nn.Linear(1,self.d_model)

        self.act_emb = nn.Embedding(self.act_dim,self.d_model)
        #timestep embedding: 让模型知道这是第几步（否则只position不稳）
        self.ts_emb = nn.Embedding(self.max_timestep, self.d_model)
        # pe:序列长度固定为3k
        self.pos_emb = nn.Embedding(3*self.K, self.d_model)

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            # 列表推导式，复制粘贴 N 层完全独立的 Block，叠在一起。
            Block(n_embd=self.d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        # Pre-LayerNorm的架构，block的输出未归一化，所以在最后给它归一化
        # 把这堆“堆叠”后的数据强行拉回标准分布，然后再喂给最后的预测头 (action_head)
        self.ln_f = LayerNorm(self.d_model)

        # 用state token 的 hidden预测action 也就是用 state输出的特征去预测动作action
        self.action_head = nn.Linear(self.d_model, self.act_dim)

        # 预先做一个causal mask（下三角），形状[1,1,L,L]
        L = 3*self.K
        causal = torch.tril(torch.ones(L,L,dtype=torch.bool))
        self.register_buffer("_causal_mask", causal.view(1,1,L,L),persistent=False)

    def _build_attn_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """valid: [B, K] 0/1
        变成 token_valid: [B, 3K]，每个 timestep 扩 3 倍
        然后做 mask:
          attn_mask[b, 0, i, j] = causal[i,j] AND token_valid[b,j]
        解释：
          - causal：保证“只能看过去”
          - token_valid(b,j)：保证“不能把 padding 当成可用信息”
        """
        B, K = valid.shape
        assert K == self.K

        
    