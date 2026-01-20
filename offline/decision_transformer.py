# DecisionTransformer 模型本体
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
class DecisionTransformerBatch:
    # 前面加了个batch维
    states: torch.Tensor     # [B, K, state_dim] float
    actions: torch.Tensor    # [B, K] long
    rtg: torch.Tensor        # [B, K] float (已 scale)
    timesteps: torch.Tensor  # [B, K] long
    valid: torch.Tensor      # [B, K] float (0/1)

class DecisionTransformer(nn.Module):
    """最小可懂版本DecisionTransformer(离散动作)
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
        *,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_timestep: int = 512,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.K = int(context_len)
        self.d_model = int(d_model)
        self.max_timestep = int(max_timestep)
        self.d_ff = int(d_ff) if d_ff is not None else int(4 * d_model)

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
            Block(self.d_model, n_heads, self.d_ff, dropout)
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
        # 通过 register_buffer 注册的遮罩。它会确保这个形状为 [1, 1, 3K, 3K] 的固定遮罩始终留在 GPU 上，且不会因为梯度更新而改变
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

        # valid = [1,1,0]
        # 执行 repeat_interleave(3) 后，它会把每个元素原地重复 3 次： [1, 1, 1, 1, 1, 1, 0, 0, 0]
        # dim=1 是指“顺着 第 1 维(K)的方向去做操作”
        token_valid = valid.repeat_interleave(3, dim=1) # [B, 3K]
        token_valid = token_valid.to(torch.bool)

        # key mask shape [B,1,1,3K] ，广播到query维度
        # Query (Q) 决定“谁在看”，而 Key (K) 决定“被看的是谁”（窗口内所有,len=K）
        """
        Query (Q)：代表当前正在处理的 Token（比如第 5 个 Token）。
        Key (K)：代表序列中所有的 Token（比如第 1 到 90 个 Token）。
        Mask 的作用：我们要告诉第 5 个 Token，虽然第 80 到 90 个 Token 也在序列里，但它们是 Padding（无效数据），你的 Query 不准去跟它们的 Key 进行“匹配”。
        所以，这个 Mask 是加在 Key 上的，用来屏蔽掉那些无效的“被观察者”。
        Transformer 内部的 Attention 矩阵形状通常是： [Batch, Heads, Query_Len, Key_Len],为了广播做升维
        """
        key_mask = token_valid.view(B,1,1,3*self.K) 

        # causal mask shape [1,1,3K,3K]
        """
        causal_mask禁止看未来
        key_mask禁止看padding
        不匹配的维度会自动广播，比如倒数第二维，
        key mask会在这一维复制3k次，这一行的屏蔽规则（padding）会应用到每一个query身上
        """
        attn_mask = self._causal_mask & key_mask # [B,1,3K,3K]
        return attn_mask
    
    # 训练接口
    def forward(
        self,
        states: torch.Tensor,     # [B,K,state_dim]
        actions: torch.Tensor,    # [B,K]
        rtg: torch.Tensor,        # [B,K]
        timesteps: torch.Tensor,  # [B,K]
        valid: torch.Tensor,      # [B,K]
    ) -> torch.Tensor:
        B, K, D = states.shape
        if K != self.K or D != self.state_dim:
            raise ValueError(f"states shape {states.shape} not match (B,{self.K},{self.state_dim})")

        # ts emb [B,K,d](大局观：让 R_t, S_t, A_t 共享同一个时间信息)
        t_emb = self.ts_emb(torch.clamp(timesteps,0,self.max_timestep-1))
        # 三类 token embedding: 都加上 timestep emb
        """
        为什么要“三类 Token 都加 t_emb”？
        在拼接之前，s_tok、a_tok 和 r_tok 分别通过各自的 Linear/Embedding 变成了d维向量。
        问题来了：如果不加时间信息，第 5 步的 State 和第 500 步的 State 经过 state_emb 出来的结果是一模一样的。
        解决办法：把代表“第几秒”的向量 t_emb 直接加到它们身上。
        物理意义：这相当于给数据染色。原始 State 是它的“内容”。t_emb 是它的“时间属性”。
        相加的结果：模型既知道“我现在在哪（状态）”，也知道“现在是几点（时间）”
        """
        s_tok = self.state_emb(states) + t_emb                 # [B,K,d]
        a_tok = self.act_emb(torch.clamp(actions, 0, self.act_dim - 1)) + t_emb
        # 原始 rtg 的形状是 [B, K]（每个批次、每个时间步一个分数）,
        # self.rtg_emb 是一个 nn.Linear(1, d)。它要求输入必须有一个“特征维度”
        # 这个 1 就是告诉 Linear 层：“这就是我要处理的那个唯一的实数值特征。
        r_tok = self.rtg_emb(rtg.unsqueeze(-1)) + t_emb        # [B,K,1] -> [B,K,d]

        # interleave: [B,K,3,d] -> [B,3K,d] 把k个token展平在一起
        # 它把原本在第 2 维的 3 个 Token 摊开到第 1 维（时间轴）上
        """
        交织的三个核心目的：维持因果律 (Causality)：正如之前讨论的，Transformer 的每一位只能看左边。在交织序列中，当模型处理St时，它的左边正好是Rt.这意味着模型可以根据“当前目标”来理解“当前状态”。
        局部相关性：对于 Transformer 来说，距离越近的 Token 越容易建立强的注意力联系。把同一时间步的 R, S, A 放在相邻位置，能让模型更容易学到它们之间的瞬时因果关系。
        模拟文本逻辑：这就像把 RL 变成了一种语言：“为了拿到 500分，看到 木杆右倾，我执行了 向右按键。”这三个词必须挨着说，逻辑才通顺。
        """
        x = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(B, 3 * K, self.d_model)

        # pos emb (0...3K-1)(微观秩序：让 R_t, S_t, A_t 拥有不同的位置信息)
        pos = torch.arange(3*K, device=x.device).unsqueeze(0) #[1,3K]
        x = x + self.pos_emb(pos)

        x = self.drop(x)

        # attention mask
        attn_mask = self._build_attn_mask(valid)  # [B,1,3K,3K]

        for blk in self.blocks:
            # 防止信息污染每一层都要mask
            x = blk(x, attn_mask = attn_mask)

        x = self.ln_f(x)

        # 取 state token的位置：1,4,7...也就是1::3：从索引 1 开始，每隔 3 个取一个
        # x[:, 1::3, :]：精准“抠出”state特征
        # DecisionTransformer认为：在 t 时刻，融合了当前目标 R_t 信息的状态特征 S_t，是预测动作 A_t 的最佳依据
        h_state = x[:, 1::3, :]         # [B,K,d] 
        logits = self.action_head(h_state)  # [B,K,act_dim] 它把每个时间步的 d 维抽象向量，投影到动作空间上
        return logits
        # 训练时，我们拿这个 logits 去跟数据集里的真实动作 A 做对比
        # S_0 位置的输出 -> 预测 A_hat_0 -> 对应真实 A_0

        """
        总结：DecisionTransformer 的“生产线”全貌
        输入：[R, S, A] 三股绳子。
        加工：stack + reshape 编织成一根绳子，加上时间和位置补丁。
        消化：blocks 让信息在绳子上横向流动（但只能向左看）。
        收割：1::3 像收割机一样，只在 S 的位置剪断，拿走精华。
        翻译：action_head 把精华变成动作指令。
        """
    
    # 推理接口 它的核心目标是决策
    @torch.no_grad()
    def act(
        self,
        *,
        states: torch.Tensor,     # [1,K,D]
        actions: torch.Tensor,    # [1,K]
        rtg: torch.Tensor,        # [1,K]
        timesteps: torch.Tensor,  # [1,K]
        valid: torch.Tensor,      # [1,K]
        valid_len: int,
        sample: bool = True,
    ) -> int:
        """
        从当前窗口里选“最后一个有效 timestep”的动作。

        注意：
          - actions 序列里，最后一个 timestep 的 action 可以是占位符 0。
            不会泄漏，因为我们用的是 state token 的 hidden，causal mask 保证看不到它后面的 action token。
        """
        logits = self.forward(states, actions, rtg, timesteps, valid)  # [1,K,act_dim]
        last = max(int(valid_len) - 1, 0)

        l = logits[0, last]  # [act_dim]
        if sample: # sample 参数：保守 vs 冒险
            # 概率采样，增加探索性exploration：先过 softmax 变成概率（比如：左 80%，右 20%），然后按概率“抽奖” (torch.multinomial)
            probs = F.softmax(l, dim=-1)
            a = torch.multinomial(probs, num_samples=1).item()
        else:
            # 贪婪决策：直接选分最高的，增加exploitation
            a = torch.argmax(l, dim=-1).item()
        return int(a)
    
    """
    这就像是一个滑动窗口：
    收集最近的 K 步历史。
    喂给模型。
    模型盯着最后那个有效的 State Token 看。
    action_head 吐出动作概率。
    抽签或者直接选最高分，返回给环境执行。"""
