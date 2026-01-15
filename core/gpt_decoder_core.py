# 把之前实现的GPT-style的decoder的核心抽出来方便复用
from __future__ import annotations

import torch
import torch.nn as nn

from core.transformer_min import Block, LayerNorm

"""这个模块的作用是把已经初步转换过的向量序列 x_emb，
通过多层自注意力（Self-Attention）和前馈网络（FFN），
转化成高度浓缩了上下文信息的隐藏状态 h"""
class GPTDecoderCore(nn.Module):
    """
    只负责最核心的任务：从序列信息中提取复杂的时空特征
    不负责处理最初的输入（感知），也不负责最后的决策（动作）
    Decoder-only Transformer backbone (GPT-style)
    输入: x_emb [B, T, d_model]  （已经是 embedding 之后的序列）
    输出: h     [B, T, d_model]  （最后一层的 hidden states）

    注意：causal_mask 采用 True=允许看（与 MultiHeadSelfAttention 的 masked_fill(~mask) 一致）
    """
    def __init__(
        self,
        max_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        # 位置编码
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        # 堆叠的blocks：（包含 Multi-Head Attention 和 MLP）。
        # 重复迭代：通过多层堆叠，模型可以学习到更深层的逻辑。
        # 例如第一层可能只是对比两帧之间的位移，第二层可能就能理解加速度或物体的运动意图
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.ln_f = LayerNorm(d_model)

        # 因果掩码-生成下三角矩阵，严禁模型偷看未来
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_len, max_len), persistent=False)

        self._init()

    def _init(self):
        # rl中通常需要手动干预权重的初始分布
        # 位置编码初始化，模仿gpt论文中的做法，给位置编码一个非常小的随机值
        # 线性层正交初始化，原理：让矩阵每一行（列）都互相垂直且模长为1
        # 为了保证矩阵相乘后的信号强度不发生剧烈改变
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
        """
        x_emb: [B, T, d_model]
        """
        B, T, C = x_emb.shape
        if T > self.max_len:
            raise ValueError(f"T={T} > max_len={self.max_len}")

        pos = torch.arange(T, device=x_emb.device).unsqueeze(0)  # [1,T]
        x = x_emb + self.pos_emb(pos)                            # [B,T,C]
        x = self.drop(x)

        attn_mask = self.causal_mask[:, :, :T, :T]               # [1,1,T,T] True=allow
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        return self.ln_f(x)

"""
输入：一串 Embedding 向量。

加法：加上位置坐标。

循环：在 Block 1 算注意力 -> 在 Block 2 算注意力...

归一化：通过最后的 LayerNorm 保证输出数值稳定。

返回：得到一串同样长度、但内涵丰富的特征向量。
"""