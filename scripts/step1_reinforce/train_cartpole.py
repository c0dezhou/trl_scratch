import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym

from core.nn_utils import seed_everything, get_device
from models.policy_mlp import CategoricalPolicyMLP
from rl.reinforce import compute_returns, reinforce_loss
from configs.step1_reinforce_cartpole import ReinforceCartPoleConfig

"""
训练逻辑：
1.用 CategoricalPolicyMLP 去玩一局，存下 log_probs 和 rewards。
2.用 compute_returns 算回报。
3.用 reinforce_loss 算损失并 backward()。
4.optimizer.step()。
"""
def main():
    cfg = ReinforceCartPoleConfig()
    seed_everything(cfg.seed)
    device = get_device("cpu")

    # 1. 创建cartpole环境
    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = CategoricalPolicyMLP(obs_dim, act_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    ep_returns = []
    t0 = time.time()

    print(f"device is {device}, seed is {cfg.seed}")
    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + ep) # 每个episode不同seed
        done = False

        log_probs = []
        rewards = []

        # 采样（采集数据）
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            action_t, log_prob_t = policy.act(obs_t)
            action = int(action_t.item())

            # env.step(action): 这一步是物理模拟的核心，返回下一秒车和杆子的状态
            # terminated：杆子倒了 truncated：时间到了
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 记录 log_probs 和 rewards
            # log_prob_t（当时选动作的概率）和原始的 reward。
            # 此时模型还没有开始学习，它只是在“写日记”，把所有瞬间的感受先存起来
            log_probs.append(log_prob_t)
            rewards.append(float(reward))
            obs = next_obs
        
        # 对齐阶段：回报计算与标准化，计算Gt
        # 当一局游戏结束（done=True），我们进入“总结阶段”。
        returns = compute_returns(rewards, cfg.gamma).to(device)

        # 可选：标准化Gt 减少方差，更稳定
        if cfg.normalize_returns:
            # 我们不看绝对的奖励分数，而是看这一步的表现是否优于这一局的平均水平
            # 标准化后，表现好的步数 returns 为正（拉高概率），表现差的步数 returns 为负（降低概率）。这极大地稳定了训练，模型学得飞快
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # reinforce loss
        loss = reinforce_loss(log_probs, returns)

        opt.zero_grad(set_to_none=True)
        loss.backward() # 计算梯度
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip) # 保险丝
        opt.step()

        R = sum(rewards)
        ep_returns.append(R)

        if ep % cfg.print_every == 0:
            avg = float(np.mean(ep_returns[-cfg.print_every:]))
            print(f"ep {ep:4d} | avg_return {avg:7.1f} | last_return {R:6.1f} | time {time.time()-t0:.1f}s")

    env.close()
    print("done.")

"""
这就是 RL 的本质：
训练时：我们要“乱试”（Sampling），这样才能发现更好的路径。
测试时：我们才用 argmax，展示已经学稳的技巧。
"""
if __name__ == "__main__":
    main()

"""

策略梯度要优化的是期望回报：让“按当前策略跑很多次”的平均分更高。
这个期望算不出来，只能蒙特卡洛采样：用当前策略跑轨迹，用采样平均近似。
“似然/概率”指的是：在轨迹里每一步你选到动作的概率 π(a_t|s_t)（环境转移不由参数控制）。
用 log_prob 是因为动作是采样出来的不可导，我们改为对“选中该动作的概率”求导，才能更新网络。
出现 log π 不是“用 log 代替 π”，而是：
把连乘的轨迹概率变成连加（更稳定）
梯度形式变成“回报加权的推高/压低概率”
REINFORCE 的直觉：回报高就提高这次选到动作的概率；回报低就降低它（用 R * log_prob 做权重）。
baseline（如 V(s)）把回报变成相对分数 advantage = return - baseline，不改方向，只把更新力度变小更稳，所以降方差。

"""
