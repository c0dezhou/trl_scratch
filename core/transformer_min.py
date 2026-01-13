# step0: 手搓GPT-style transfomer (decoder only)

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1.手搓layernorm
class LayerNorm(nn.Module):
    """手搓layernorm,对最后一维做归一化"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 输入的tensor shape是[B, T, C]（batch，time/Sequence len(context windows的长度，单位是帧), channels/embading dim(特征维度))）,对最后一维C做归一化（mean/var)
        # 然后用scale/shift做逐维度缩放平移
        """
        可以把这个 [B, T, C] 想象成 B 条面包：
        B (Batch)：桌上并排摆着 16 条面包。
        T (Time)：每条面包被切成了 30 片（切片代表时间步）。
        C (Channel)：每一片面包里的 营养成分（碳水、蛋白质、脂肪... 共有 64 种成分）。
        
        B = Batch Size 
        T = Time / Sequence Length (时间步/句子长度)
        C = Channels = Features = Embedding Dimension = d_model
        
        Transformer 在这里做什么？
        Attention (注意力)：是在 T (Time) 维度上搞事情。
        它在算：“对于第 30 片面包（现在），第 15 片面包（过去）的某个成分对它有多重要？”
        MLP / Linear：是在 C (Channel) 维度上搞事情。
        它在算：“把这片面包的 64 种成分混合加工一下，提取出新的特征。”
        """
        self.scale = nn.Parameter(torch.ones(dim)) 
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True) # [B,T,1]
        # keepdim=True 为了广播。
        # unbiased=False 为了让分母是 N，符合 LayerNorm 的官方数学定义。
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased即有偏，归一化全部对象（这不是在抽样），除以n
        x_hat = (x - mean) / torch.sqrt(var + self.eps) # [B,T,C]

        return x_hat * self.scale + self.shift  # scale,shift: [C] 会自动 broadcast 成 [1,1,C]
    

# 2.多头自注意力 (混合时间，对token之间的关系进行提取， Mix time)
class MultiHeadSelfAttention(nn.Module):
    """
    input: x:[B,T,C]
    output: y:[B,T,C]

    QKV都来自于x的线性投影
    多头机制：对C进行拆分
    """
    def __init__(self, d_model: int, n_heads: int, dropout:float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除" # C维
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 一次线性映射出 QKV（更快也更常见）
        # x: [B,T,C] -> qkv: [B,T,3C]
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        # 输出投影，把拼回来的多头结果再映射回 C
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout) # 砍掉的是 Attention Weights（关注度）,防止过度依赖特定的 Token 关系【token之间
        self.resid_drop = nn.Dropout(dropout) # 砍掉的是 Feature Vector（特征向量） 里的数值,标准神经网络正则化【token内部

    def forward(self, x:torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        attn_mask: [1,1,T,T]或[B,1,T,T]
        True表示允许注意(keep)，False表示禁止注意（mask out)
        """
        # PRE-LN 结构 （attention原文是Post-LN结构）

        # attention(q,k,v) = softmax(Q@K^T/sqrt(Hd)+mask)*V
        """
        对应到实现顺序就是：
        1.scores = QK^T / sqrt(dk)
        2.scores[masked] = -inf（或加一个很大的负数）
        3.attn = softmax(scores)
        4.attn = dropout(attn)
        5.y = attn @ V
        """
        B,T,C = x.shape

        # 1) 线性映射得到qkv
        qkv = self.qkv(x) # [B, T, 3C]
        q,k,v = qkv.chunk(3,dim=-1) # 每个都是 [B, T, C]

        # 2) 拆多头 [B, T, C] -> [B, H, T, Hd] 把C拆成[H, Hd]然后把head维度移到前面方便矩阵乘法
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2) # [B,H,T,Hd]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2) # [B,H,T,Hd]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2) # [B,H,T,Hd]

        # 3) 注意力分数 scores = (Q·K^T)/sqrt(Hd) 
        # q: [B,H,T,Hd]
        # k^T: [B,H,Hd,T]
        # scores: [B,H,T,T]，每个位置 t 对所有位置 j 的打分
        scores = (q@k.transpose(-2,-1)) / math.sqrt(self.head_dim) # [B,H,T,T] 注意力map

        # 4）mask 
        if attn_mask is not None:
            # attn_mask 里 1 (True) 代表“能看”, 把所有不能看(~attn_mask)的地方变成-inf
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        # 5) softmax得到注意力权重
        attn = F.softmax(scores, dim=-1) #[B,H,T,T]
        attn = self.attn_drop(attn)

        # 6) 加权求和输出 y = attn @ V
        y = attn @ v

        # 7) 拼回去 [B,H,T,Hd] -> [B,T,C]
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # 8)投影输出+dropout
        y = self.resid_drop(self.out(y)) # [B,T,C]

        return y
    
# 3.FFN (混合特征 Mix Channels 只对当token的特征进行加工)
class FFN(nn.Module):
    """
    transformer的mlp部分
    Linear(C -> 4C) -> GELU -> Dropout -> Linear(4C -> C)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

# 4.transformer block(pre-ln)（更稳）
class Block(nn.Module):
    """
    Pre-LN 结构：
      x = x + Attn(LN(x))
      x = x + FFN(LN(x))
    好处：训练更稳定，GPT 系常用

    attention原文：post-LN
    x = LN(x + Attn(x))
    x = LN(x + FFN(x))
    """
    def __init__(self, d_model:int, n_heads: int, d_ff: int, dropout: float=0.0):
        super.__init__()
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), attn_mask=attn_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

# 5. 最小gpt style （decoder only)
"""
我们要的核心能力是“按时间只看过去”
后面可以做的是：
1.PPO 里用历史观测做决策（POMDP） ：PPO + Transformer 记忆：用 causal 处理历史窗口
2.Decision Transformer / Trajectory 模型学轨迹 ：轨迹当 token，自回归预测动作
3.世界模型预测未来 : 自回归生成未来轨迹

它们有一个共同的物理规则：
t 时刻只能用 0..t 的信息，不能偷看 t+1.. 未来。
decoder-only + causal mask 就是专门为这件事长出来的。

核心逻辑链：
输入：一堆整数 ID (idx)。
理解：查表得到词向量，加上位置向量，形成初始认知 (x)。
思考：拿着这一摞向量，穿过 N 层 Block。每一层都在做 Attention（看上下文）和 MLP(FFN)（脑内消化）。
同时，attn_mask 全程伴随，像眼罩一样挡住未来的信息。
表达：最后把抽象的向量 (d_model 维) 映射回具体的词表概率 (vocab_size 维)。
总结这个 MiniGPT 类就是一个标准的 Decoder-only Transformer。
输入：[B, T] (你给它的上文)
输出：[B, T, V] (它对每一个位置的下一个词的预测分数)
你拿到这个 logits 后，通常只取最后一个时间步 logits[:, -1, :]，
过一个 Softmax，就能依概率采样出下一个字了。这就是 ChatGPT 生成文本的最底层原理。
"""
class MiniGPT(nn.Module):
    """
    Step0 用的最小 GPT：
    - token embedding + position embedding
    - N 层 Block
    - 最后线性头输出 logits

    输入 idx: [B,T]（整数 token id）
    输出 logits: [B,T,V]
    """
    def __init__(
            self,
            vocab_size: int,
            max_len: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 2,
            d_ff: int = 256,
            dropout: float = 0.0,
    ):
        super.__init__()
        
        self.vocab_size = vocab_size
        self.max_len = max_len

        # 1. 嵌入层 (The Eyes) 
        # 把单词 ID (比如 502) 变成向量 (d_model维)
        self.tok_emb = nn.Embedding(vocab_size,d_model)
        # 把位置 ID (比如 第0个, 第1个) 变成向量
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        # 2. 堆叠 Blocks (The Body/Brain)
        # 用 nn.ModuleList 把之前写的 TransformerBlock 装进一个列表里
        # 这就是深度学习的“深度”所在，n_layers 决定了有多少层
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # 3. 输出层 (The Mouth)
        # 最后的归一化，保证输出分布稳定
        self.ln_f = LayerNorm(d_model)
        # 最后的线性投影：把 hidden_state (d_model) 映射回 词表大小 (vocab_size)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # bias=False 通常在最后的输出层，我们不需要偏置项（bias），因为 Softmax 也就是看相对大小，去掉 bias 可以稍微减少参数量，且不影响效果

        # 4. 防泄漏机制：mask buffer 这个 Mask 确保了第 t个 token 只能看到 0 到 t 的信息，看不到 t+1（未来）。
        # 预先做一个最大长度的 causal mask(因果掩码) 存在 buffer 里（不会被训练）
        # 形状： [1,1,max_len,max_len] 创建一个下三角矩阵 (全是1)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        # register_buffer 的作用：
        # 1. 它不是参数 (Parameter)，不会被梯度下降更新。
        # 2. 但它是模型状态的一部分，会随着 model.to(device) 自动移动到 GPU/CPU。
        # 3. view(1, 1, ...) 是为了方便后面做广播 (Broadcasting)
        self.register_buffer("causal_mask", mask.view(1,1,max_len,max_len),persistent=False)
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor: # 模型真正的思考过程
        B, T = idx.shape # T:当前序列长度

        #1.生成位置索引
        # idx.device 确保这个新张量和输入数据在同一个显卡上
        # pos 变成: [0, 1, 2, ..., T-1]
        pos = torch.arange(T, device=idx.device).unsqueeze(0) # shape [1,T]

        #2. 融合信息
        # 两者直接相加（不是拼接），让向量同时携带语义和位置信息。
        # tok_emb(idx): [B, T, C]
        # pos_emb(pos): [1, T, C] (这里触发了广播，自动把 1 复制成 B)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        #3. 切出当前的mask
        # 我们之前的 buffer 是 max_len * max_len 的，现在只取 T * T
        attn_mask = self.causal_mask[:,:, :T, :T]

        #4. 层层传递
        # 每一层 Block 处理完，x 的形状依然是 [B, T, C]，但特征更抽象了
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        #5. 输出
        x = self.ln_f(x)
        logits = self.head(x) # 变成 [B, T, vocab_size]
        return logits