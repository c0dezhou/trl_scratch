"""
1.加一个 Critic(V)：学V(s)当 baseline，显著降方差
2.用 GAE(λ) 计算 advantage：更稳、更常用
3.实现 PPO clipped objective：工业最常见策略优化
4.训练循环变成：rollout batch + update epochs + minibatch

特性,作用
Ratio + Clip,限制新旧策略差异，防止由于概率剧变导致的训练崩溃。
Value Loss,训练一个更准的“教练”，为 Actor 提供更稳的 Baseline。
Entropy,给模型“打鸡血”，让它别太快变保守，多去尝试新可能。
Mini-batch,将大批采样数据打乱更新，消除时间相关性，提高训练效率。
"""
"""
PPO Total Loss Formula:
L_total = L_pi + c1 * L_vf - c2 * L_s

1. Policy Objective (Clipped):
   ratio = exp(new_logp - old_logp) = pi_theta / pi_theta_old
   L_pi = -E[ min( ratio * Adv, clip(ratio, 1-eps, 1+eps) * Adv ) ]
   (eps = clip_coef, Adv = GAE/Advantage)

2. Value Function Loss (Mean Squared Error):
   L_vf = E[ (V(s) - Return)^2 ]
   (Return = G_t or TD-target)

3. Entropy Bonus (Encourages Exploration):
   L_s = E[ -sum( p * log(p) ) ]
   (Higher entropy means more random actions)

Combined Loss for Gradient Descent:
loss = policy_loss + (vf_coef * value_loss) - (ent_coef * entropy)
       ^(actor)       ^(critic)                ^(exploration)
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def ppo_update(
        model,
        optimizer,
        buffer,
        update_epochs: int,
        minibatch_size: int,
        clip_coef: float, # 即epsilon=0.2
        vf_coef: float, # 价值损失权重 ，评价重要性 旋钮
        ent_coef: float, # 熵系数，探索精神旋钮
        max_grad_norm: float, # 梯度裁剪阈值，参数保护锁
        target_kl: Optional[float] = None, # KL 早停阈值
        clip_vloss: bool = False, # 是否启用 value loss clipping
):
    """参数,控制对象,核心目的,默认值参考
clip_coef,策略更新幅度,保命：防止模型突然学废,0.2
vf_coef,价值网络学习,评价：让教练看局势看更准,0.5
ent_coef,策略探索度,探索：防止模型过早自满,0.01
max_grad_norm,梯度更新强度,稳定：防止参数更新步长暴走,0.5"""
    for _ in range(update_epochs):
        for obs, act, old_logp, adv, ret, old_v in buffer.get_minibatches(minibatch_size):
            logits, v = model(obs)
            dist = Categorical(logits=logits) # 概率分布对象, 像是一个装了不同大小扇形的转盘
            # dist.sample()	随机转动转盘得到的结果	0 或 1
            # dist.log_prob() 算出刚才那个结果在转盘上占多大面积（取对数）	-0.22
            new_logp = dist.log_prob(act)
            entropy = dist.entropy().mean() # 算出当前这个分布的“混乱程度”，也就是 ent_coef 要用到的那个值

            # 核心公式 1：概率变化率 ratio
            # 如果 ratio > 1，说明新模型比旧模型更倾向于选这个动作；反之则更排斥
            ratio = torch.exp(new_logp - old_logp) #[B] logA-logB = log(A/B) 求exp之后就是A/B
            
            # 核心公式 2：Clipped Objective（剪切目标）
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0-clip_coef, 1.0+clip_coef)*adv
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            # 核心公式 3：三合一 Loss
            if clip_vloss:
                v_clipped = old_v + (v - old_v).clamp(-clip_coef, clip_coef)
                v_loss_unclipped = (v - ret).pow(2)
                v_loss_clipped = (v_clipped - ret).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
            else:
                value_loss = 0.5 * F.mse_loss(v, ret)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 把梯度的模长限制在一个范围内（比如 0.5），确保更新步长永远是安全的
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            approx_kl = (old_logp - new_logp).mean().item()
            if target_kl is not None and approx_kl > 1.5 * target_kl:
                return




    
