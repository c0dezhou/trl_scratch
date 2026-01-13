import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticMLP(nn.Module):
    """
    actor-critic网络，共用一个主干网络(backbone)
    长出两个输出头：
    self.pi (actor) -- 输出logits，决定怎么掷骰子（就是之前的policy网络，负责算出π(a|s)
    self.v (critic) -- 负责评价，输出一个标量v(s)，代表模型认为当前状态s的价值 （也就是实时baseline
    优势计算：在训练时，我们会用Gt-V(st)代替原来的Gt ，这可以进一步减小梯度的方差

    """
    # 为什么需要 self.v？
    # 在 REINFORCE 中，我们用一局游戏的平均奖励作为 Baseline。但那太迟钝了，因为那是“事后诸葛亮”。
    def __init__(self, obs_dim: int, act_dim: int, hidden: int=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden,hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim) # logits
        self.v = nn.Linear(hidden, 1) # scalar value

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1) # [B] self.v(h) 的输出维度通常是 [Batch, 1]。
        # squeeze(-1) 把最后一维去掉，变成 [Batch]。
        # 目的：让 Value 的形状和 log_prob 保持一致，方便后面直接做减法算 Loss。
        return logits, value
    
    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        l,v = self.forward(obs)
        dist = Categorical(logits=l)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, v # 比reinforce多一个v
        # 这意味着我们在玩游戏的过程中，不仅记住了自己做了什么（action）和当时的信心（logp），还记住了自己当时对局势的判断（value）
    
