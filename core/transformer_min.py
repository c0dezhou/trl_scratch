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
    

# 2.多头自注意力
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