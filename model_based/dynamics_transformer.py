from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from core.transformer_min import Block, LayerNorm

@dataclass
class DynOut:
    delta: torch.Tensor    # (B, K, state_dim)
    done_logits: torch.Tensor # (B, K, state_dim)

class DynamicsTransformer(nn.module):
    """最小可懂的world model: Transformer dymanics
    利用 Transformer 学习因果律：如果我在状态 s 做了动作 a，世界会变成什么样？
    Token 排列（长度 2K）：
      [s0, a0, s1, a1, ..., s_{K-1}, a_{K-1}]
    输出：
      在 action token 位（1,3,5,...）预测该步 transition 的:
        delta_s (到 s_{t+1}) 以及 done_t
    """
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        context_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_timestep: int,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.K = int(context_len)
        self.d_model = int(d_model)
        self.max_timestep = int(max_timestep)

        # embeddings
        # 将高维（或低维）物理向量映射到隐藏层空间
        self.state_embed = nn.Linear(self.state_dim, self.d_model)
        # 将离散动作 ID 映射成向量
        self.action_embed = nn.Embedding(self.act_dim,self.d_model)
        # 它告诉模型这一帧发生在游戏的什么时候
        self.max_timestep_embed = nn.Embedding(self.max_timestep, self.d_model)

        # 位置编码： 2K 长度，这一步在step中的位置
        # 力学预测（Dynamics）最核心的就是时间因果。模型必须知道 a_0 是作用在 s_0 之后的，才能预测出 s_1。
        # 没有位置编码，它就会把未来的动作和过去的状态混为一谈，物理规律就会彻底崩塌
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * self.K, self.d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(self.d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = LayerNorm(self.d_model)

        # 在动作执行后看后果
        # 将提取出的动作位置的输出变成 物理位移 和 死亡信号
        # 专门在 action token 的输出位置挂上这两个“头”来提取预测结果
        """
        预测头,Loss 函数,学习目标
        delta_head,MSE (均方误差),学习物理规律（位置、速度的变化）。
        done_head,BCE (二元交叉熵),学习环境边界（撞墙、掉落、超时）。
        """
        self.delta_head = nn.Linear(self.d_model, self.state_dim)
        self.done_head = nn.Linear(self.d_model, 1)

    def forward(
        self,
        states: torch.Tensor,     # (B, K, D) normalized
        actions: torch.Tensor,    # (B, K)    int64
        timesteps: torch.Tensor,  # (B, K)    int64
        valid: torch.Tensor,      # (B, K)    float/bool
    ) -> DynOut:
        B, K, D = states.shape
        assert K == self.K, f"expected K ={self.K}, got {K}"

        # clamp 时间戳 到 emb的范围
        ts = timesteps.clamp(min=0, max=self.max_timestep - 1)

        # embed
        s_tok = self.state_embed(states) + self.timestep_embed(ts)     # (B,K,H)
        a_tok = self.action_embed(actions) + self.timestep_embed(ts)   # (B,K,H)

        # interleave 交织,为了满足 Transformer 自回归 (Autoregressive) 的预测逻辑
        # 1. 先造一个空容器，长度是 2K (因为要装 s 和 a)
        x = torch.empty((B, 2*K, self.d_model),device=states.device, dtype=s_tok.dtype)
        # 2. 从第 0 位开始，每隔 2 个位置填一个 state
        x[:, 0::2, :] = s_tok
        # 3. 从第 1 位开始，每隔 2 个位置填一个 action
        x[:, 1::2, :] = a_tok

        # 位置编码注入,给每一个要处理的信号盖上一个时间戳戳记
        # 偶数位会被加上“状态位”的编码，奇数位会被加上“动作位的编码
        x = x+self.pos_embed[:, :2*K, :]
        x = self.drop(x)

        """key padding mask -> 针对key （被看的对象，矩阵的列）是纵向限制（竖着挡住最右边的几列），防止关注到无效填充的垃圾数据，
        causal mask是横向限制，防止时间维度上的穿越"""

        # 构建 key padding mask : token_valid(B, 2K)
        # 这一排 Token 里，哪些是真实存在的，哪些是后面为了凑长度补的零（Padding）
        if valid.dtype != torch.bool:
            v = valid > 0 # 强转T/F
        else:
            v = valid
        token_valid = torch.empty((B, 2 * K), device=states.device, dtype=torch.bool)
        token_valid[:, 0::2] = v  # 对应所有 s_tok 的位置
        token_valid[:, 1::2] = v  # 对应所有 a_tok 的位置

        # 1) 因果掩码causal mask (2K, 2K): 只能look back，不能look ahead
        causal = torch.trill(torch.ones(2*K, 2*K), device=states.device, dtype=torch.bool).view(1, 1, K, K)
        # 2) key padding mask: [B,1,1,T]
        key_mask = token_valid.to(torch.bool).view(B, 1, 1, K)
        # 3) 合并: [B,1,K,K]
        attn_mask = causal & key_mask


        for blk in self.blocks:
            x = blk(x, attn_mask =attn_mask)

        x = self.ln_f(x)

        # 从 Transformer 输出的一堆向量中，只把属于‘动作’（Action）位置的那些隐藏层状态（向量）给挑出来
        ha = x[:, 1::2, :]  # (B,K,H)

        delta = self.delta_head(ha)              # (B,K,D)
        done_logits = self.done_head(ha).squeeze(-1)  # (B,K)

        return DynOut(delta=delta, done_logits=done_logits)
    """
    输入：你给了模型一段历史 [s, a]。
    融合：模型把状态、动作、时间、位置全部揉在一起变成 Token。
    计算：Transformer 内部的 blocks 进行多层自注意力计算，理解物理因果。
    提取：模型盯着每一个动作 Token。
    翻译：通过两个 Head，把动作 Token 
    翻译成：“下一秒坐标变了多少”和“这一步会不会死”。
    """

        

