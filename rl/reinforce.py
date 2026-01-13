# reinforce agent

import torch

def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """
    输入一条episode的reward序列 [r0,r1,...]
    输出每个时刻的回报G_t
    """
    G = 0.0
    returns = []
    # 递推公式：Gt = rt + gamma*G(t+1）,从后往前算降低复杂度
    for r in reversed(rewards):
        G = r + gamma*G
        returns.append(G)
    returns.reverse()
    # 一个episode的最后才给总奖励

    return torch.tensor(returns, dtype=torch.float32)

# 奖励加权 把“玩游戏”变成“算梯度”的关键一步
def reinforce_loss(log_probs: list[torch.Tensor], returns: torch.Tensor) ->torch.Tensor:
    """
    REINFORCE 目标：最大化 sum_t (G_t * log pi(a_t|s_t))
    所以 loss = -sum_t (G_t * log_prob_t)
    """
    log_probs_t = torch.stack(log_probs) # 把之前在 act() 函数里记录的一堆 log_prob（它们分散在 list 里）堆叠成一个整齐的张量，shape:[T]  
    # T 是这局游戏持续的总步数
    # returns 即 Gt，如果某一时刻t的回报即 returns很大且为正，那么 log_prob 的系数就很大, （强化它）
    # 由于adamW等优化器默认最小化loss，所以加负号 （最大化奖励=最小化负的奖励）
    return -(log_probs_t * returns).sum() # 把这一局所有步数的“带权概率”加起来，求出这一局的总损失
