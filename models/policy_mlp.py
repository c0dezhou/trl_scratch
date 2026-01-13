# 策略梯度算法 （PG) 
# reinforce的大脑
# model-free 真实环境采样（on-policy rollouts）的动作采样。
# reinforce (Policy Gradient), 直接学出π_θ(a|s)
# On-Policy 致命的弱点：数据不能重复使用（Sample Inefficiency）
# 一旦用这一局的数据更新了模型，模型就变了。因为是 On-Policy，新模型不能再用旧模型产生的数据来更新（数学上会导致偏差）, 每一轮更新都要重新采样

import torch
import torch.nn as nn
# 在 NLP 里，我们拿到 logits 后直接算 CrossEntropy。但在 RL 里，我们需要“采样”
# Categorical(logits=logits)：它把模型输出的分数（logits）变成了像骰子一样的概率分布。
from torch.distributions import Categorical

# RL's Policy，决定做什么 和 记录为什么要这么做，（用 MLP 替代 Transformer）
# 原因：CartPole 环境的输入（obs）只有 4 个数字（车位置、速度、杆角度、角速度）。
# 对于这么简单的输入，不需要动用 Transformer 昂贵的注意力机制，简单的 MLP 就能完美解决。
class CategoricalPolicyMLP(nn.Module):
    """
    离散动作策略：给定 state -> 输出 action logits -> 采样动作
    CartPole 动作空间是 Discrete(2)，非常适合
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hidden), # 特征提取
            nn.Tanh(), # 激活更平滑 [-1,1]
            nn.Linear(hidden,hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim), # 决策层，把高位的抽象特征压缩回动作空间大小
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        只负责算分：logits
        obs: [B, obs_dim] 或 [obs_dim]
        return logits: [B, act_dim] 或 [act_dim]
        """
        # Logits 不是一种“格式”，而是一种“数学阶段”
        # Logits 指的是经过 Softmax 激活函数之前的原始输出值
        return self.net(obs)
    
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        负责做决定（action）和记录（log probability）
        在训练循环里，我们会不断调用 act 来跟环境互动
        采样动作 + 返回 log_prob
        训练 REINFORCE 需要 log_prob
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits) # 把网络输出变成“动作概率分布”
        action = dist.sample() # 按概率随机选动作
        log_prob = dist.log_prob(action) # 记录了 “刚才选中的那个动作，在当时的概率是多少”，并取了对数 log(p)
        # 它是反向传播的“钩子”。
        # 如果这个动作最后拿到了高奖励，我们就通过这个 log_prob 增大它的概率。
        # 如果这个动作导致了失败，我们就通过它减小它的概率。
        return action, log_prob

"""
 REINFORCE 的核心更新公式:
 ▽θ ≈ reward * ▽logπ(a|s)
 代码里的对应关系：
 ▽logπ(a|s) 就是这里的 log_prob。
 ▽（梯度）就是稍后我们执行 loss.backward() 时产生的东西。
 我们要做的事情就是：loss = -log_prob * reward。
"""