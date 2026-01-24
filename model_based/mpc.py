# 实现了 MPC（模型预测控制）中最经典的一种策略：Random Shooting（随机射击法）
# 这里只是纯mpc 还没接入其他逻辑
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from model_based.dynamics_transformer import DynamicsTransformer

@dataclass
class MPCConfig:
    horizon: int = 25
    num_samples: int = 1024
    done_threshold: float = 0.5
    gamma: float = 1.0
    seed: int = 0

# 1. 补齐缺失的记忆 左补齐
# offline用右对齐， online用左对齐
def _pad_left(arr: np.ndarray, K: int, pad_value: float = 0.0):
    """为什么要左补齐？ Transformer 的 K 是固定的（窗口大小）。
    如果游戏刚开始，历史数据只有 3 步，我们要把这 3 步放在窗口的最右侧，左边补 0。
    逻辑：保持“现在”永远在索引 K-1 的位置，这样模型处理起来逻辑最统一。"""
    L = arr.shape[0]
    if L >= K:
        return arr[-K:]
    # K-L 是我们需要补齐的长度
    # shape[0]是序列长度L,从state_dim开始处理，
    # 也就是提取出除了“时间长度”以外，数据本身的“特征维度”
    # 拼接结果：(15,) + (4,) 变成了 (15, 4)
    pad_shape = (K-L,)+arr.shape[1:] # 构造出一个完整的“填充块”的形状，
    # 这个块的宽度（特征数）和原来一模一样，但高度（长度）是欠缺的那部分
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([pad, arr], axis = 0)

@torch.no_grad()
def mpc_action(
    model: DynamicsTransformer,
    obs_hist_raw: np.ndarray,     # (L, state_dim)  最后一行是当前 obs
    act_hist: np.ndarray,         # (L,)            与 obs_hist 对齐，最后一位是“当前待选动作”的占位符
    t_hist: np.ndarray,           # (L,)            episode 时间步，与 obs_hist 对齐
    state_mean: np.ndarray,       # (state_dim,)
    state_std: np.ndarray,        # (state_dim,)
    cfg: MPCConfig,
    device: torch.device,
) -> int:
    """Random Shooting MPC.

    评分：用 model rollout H 步，累加期望 reward。
    对 CartPole：每步 reward ~ 1 - p(done)，越不“死”越好。
    """
    K = model.K
    state_dim = model.state_dim
    act_dim = model.act_dim

    # 归一化历史状态 zscore
    s_norm = (obs_hist_raw.astype(np.float32) - state_mean.astype(np.float32)) / state_std.astype(np.float32)

    # 左补齐到固定长度 K
    s_pad = _pad_left(s_norm, K, pad_value=0.0)                 # (K,D)
    a_pad = _pad_left(act_hist.astype(np.int64), K, pad_value=0) # (K,)
    t_pad = _pad_left(t_hist.astype(np.int64), K, pad_value=0)   # (K,)

    valid_len = min(obs_hist_raw.shape[0], K) # 计算实际有效的帧数
    valid = np.zeros((K,), dtype=np.float32)
    valid[K - valid_len :] = 1.0 # 要把最右侧的 valid_len 个位置设为 1.0

    N = int(cfg.num_samples) # 平行宇宙的数量，通常是 1024
    H = int(cfg.horizon) # 脑内向未来模拟的步数，通常是 25
    
    """
    我们把这份历史记忆复印了 1024 份。
    接下来的循环里，我们会给这 1024 个样本分别喂入不同的随机动作序列。
    这样，Transformer 就能一次性并行计算出 1024 个不同的未来，而不是一个一个去算。
    """
    states = torch.from_numpy(s_pad).to(device).unsqueeze(0).repeat(N, 1, 1)     # (N,K,D)
    actions = torch.from_numpy(a_pad).to(device).unsqueeze(0).repeat(N, 1)       # (N,K)
    timesteps = torch.from_numpy(t_pad).to(device).unsqueeze(0).repeat(N, 1)     # (N,K)
    valid_t = torch.from_numpy(valid).to(device).unsqueeze(0).repeat(N, 1)       # (N,K)

    # 生成动作候选
    # sample action sequences
    g = torch.Generator(device=device)
    g.manual_seed(int(cfg.seed))
    # 在 0 到 act_dim-1 之间随机生成整数
    cand = torch.randint(low=0, high=act_dim, size=(N, H), generator=g, device=device)  # (N,H)

    # 初始化计分板：
    # scores(得分)：初始为 0 
    # 与 alive，(存活状态)：初始为 1（全员存活）
    scores = torch.zeros((N,), device=device, dtype=torch.float32)
    alive = torch.ones((N,), device=device, dtype=torch.float32)

    # 定位“当下”：last_idx = K - 1
    # 因为在输入前做了 _pad_left，所以**当前的观测（Current Observation）**永远被推到了数组的最右端4
    # 在接下来的循环里，我们要把 cand 里的第 1 个预测动作填进去。填在哪？就填在 last_idx 这个位置，也就是紧跟着当前观测的地方
    last_idx = K - 1

    for h in range(H):
        # set action at current state (last token)
        actions[:, last_idx] = cand[:, h]

        next_state, done_prob = model.predict_next_from_last(states, actions, timesteps, valid_t)

        # expected step reward for CartPole: 1 - p(done)
        step_r = (1.0 - done_prob).float()
        scores += alive * (cfg.gamma ** h) * step_r

        # update alive
        dead = (done_prob >= cfg.done_threshold).float()
        alive = alive * (1.0 - dead)

        # append predicted next state: shift window left, put next at end
        states = torch.roll(states, shifts=-1, dims=1)
        actions = torch.roll(actions, shifts=-1, dims=1)
        timesteps = torch.roll(timesteps, shifts=-1, dims=1)
        valid_t = torch.roll(valid_t, shifts=-1, dims=1)

        states[:, -1, :] = next_state
        actions[:, -1] = 0  # placeholder, next step will overwrite
        timesteps[:, -1] = (timesteps[:, -2] + 1).clamp(max=model.max_timestep - 1)
        valid_t[:, -1] = 1.0

    best = int(torch.argmax(scores).item())
    return int(cand[best, 0].item())