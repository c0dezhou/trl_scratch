# 离线轨迹 -> 训练样本切片
# 把一条完整 episode 轨迹 (s0,a0,r0, s1,a1,r1, ...) 切成很多段长度为 K 的小窗口，算好 RTG，然后喂给 DecisionTransformer

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class Episode:
    states: np.ndarray    # [L, state_dim]
    actions: np.ndarray   # [L] (int)
    rewards: np.ndarray   # [L] (float)
    rtg: np.ndarray       # [L] (float) return-to-go (为了方便未折扣) TODO: 以后可以实现折扣
    timesteps: np.ndarray # [L] (int) 

def compute_rtg(rewards: np.ndarray) -> np.ndarray:
    """
    rtg就是还有多少步能走，这就是 DecisionTransformer 的“Prompt”
    Return-to-Go: RTG[t] = r[t] + r[t+1] + ... + r[T-1]
    CartPole 每步 reward=1，所以 RTG 很直观：还剩多少步能活。
    -训练：告诉模型“在这种状态下，想要拿到这个剩余总分，你应该做这个动作”。
    -推理：你可以喂给模型 target_return=500，骗它“只要你好好干，我就给你 500 分”，以此诱导它输出最优策略。
    """
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))): #dp的思路
        running += float(rewards[t])
        rtg[t] = running
    return rtg

# 还原 Episodes (从扁平化到结构化)
class DecisionTransformerDataset(Dataset):
    """
    由于 Offline RL 数据通常存成巨大的扁平数组（total_steps 级别,比如 obs 是 shape [100000, 4]），
    我们需要用 episode_ends 把它们切回成一局一局的游戏,然后把每个 episode 切成很多长度 K 的训练样本。

    最终喂给模型的是：
      states:    [K, state_dim]
      actions:   [K]
      rtg:       [K]（会做 scale）
      timesteps: [K]
      valid:     [K]  (1 表示真实数据，0 表示 padding)
    """
    def __init__(self, npz_path: str, context_len: int, rtg_scale: float = 1.0):
        super().__init__()
        self.context_len = int(context_len)
        self.rtg_scale = float(rtg_scale)

        data = np.load(npz_path, allow_pickle=False) # 防注入
        # N是 total_steps = epi * length_per_epi
        # 不按epi存是因为每局epi length不同，如果按[epi_count, max_step]存，会浪费大量空间去padding(0) （或者低效python list 存储
        obs = data["obs"].astype(np.float32)              # [N, state_dim]
        actions = data["actions"].astype(np.int64)        # [N] 分类任务中，CrossEntropyLoss 期望的 target 是一维的标签索引
        rewards = data["rewards"].astype(np.float32)      # [N] 做加法累加（计算 RTG）时，一维向量操作最快
        episode_ends = data["episode_ends"].astype(np.int64)  # [E] 每个 episode 的“结束位置(不含)，也就是一局结束的step的idx”
        # Tensor(在深度学习内部逻辑（训练评估推理）用torch，外部（构造数据集等）用numpy): 0维没有shape=标量（无方括号） | 一维(N,)=向量=数组（一层方括号） | 二维(1,N)=一行 ，(N,1)=一列 =矩阵（两层方括号）

        # 第一维（Index 0）是“样本数量”（这里是step数），第二维（Index 1）及以后才是“特征维度”
        self.state_dim = int(obs.shape[1])
        self.act_dim = int(data["act_dim"]) # 保存时写入的动作空间大小

        #----还原epi---
        episodes: list[Episode] = []
        start = 0
        for end in episode_ends:
            ep_obs = obs[start:end]
            ep_act = actions[start:end]
            ep_rew = rewards[start:end]

            # 计算未折扣rtg并归一化rtg：（压到[0,1]之间）:
            # 这么大的数值进入线性层（Linear Layer）后，会导致激活值非常大，反向传播时梯度容易爆炸，模型很难收敛
            ep_rtg = compute_rtg(ep_rew) / self.rtg_scale
            ep_ts = np.arange(len(ep_rew), dtype=np.int64) # 生成时间步索引（timesteps)
            """
            Transformer 本质上是位置无关的（它只看 Attention，不看顺序）。
            在 NLP 里，我们用 Positional Encoding 告诉模型单词的顺序。
            在 RL 里，我们用 Timesteps 告诉模型：“这是游戏开始后的第几秒”

            由于在 __getitem__ 里会随机切片（比如切出第 150 到 170 步），
            如果不传 timesteps，模型会以为这 20 步是从 0 开始的。
            有了 ep_ts，即使切片了，模型也知道这 20 步发生在游戏的“后期”，
            从而做出符合时序逻辑的决策
            """

            episodes.append(Episode(
                states=ep_obs,
                actions=ep_act,
                rewards=ep_rew,
                rtg=ep_rtg,
                timesteps=ep_ts,
            ))
            start = int(end)

        self.episodes = episodes

        #----预计算：构建采样索引---
        # 在不复制数据、不占用额外内存的前提下，瞬间定位到任何一个训练样本
        # 从epi_index（第几局）和这一局的第几秒（start time）开始切片
        """index[500] = (2, 30)。 当 DataLoader 请求第 500 个样本时，代码立刻知道：
        “去第 2 号 episode，从第 30 步开始，往后切 context_len 长度的数据。”
        按需切片，只存tuple，只有当真正训练时，才去内存里对应episode抠出那一段（不要在init里就切好所有数据）"""
        self.index: List[Tuple[int, int]] = [] # 理解成一个定位的二维数组吧
        for epi, ep in enumerate(self.episodes):
            """
            为什么用action做长度基准呢？理论上来说，len(actions)=len(rewards)=len(states)，
            但是在DecisionTransformer的序列拼接中国，我们的基本单元是三元组(rtg_t,s_t,a_t)，
            有时st会比at多出一个，但在这个状态下已经无法做出动作了
            每一个a必定对应一个s和r，所以用action来对齐
            """
            L = len(ep.actions)
            # 每个起点都做一个窗口，“滑动窗口”的逻辑，数据利用率 100%。
            # 每一个 transition 都会作为上下文的一部分被训练多次
            for s in range(L):
                self.index.append((epi, s))
        
        #----state归一化 减少不同量纲之间的的冲突 压到[0,1]之间（Z-score）---
        # 把每个epi的state抽出来竖着拼接在一起（第一维就是total steps）
        all_states = np.concatenate([ep.states for ep in self.episodes], axis=0)
        # 计算每一维特征的平均值
        self.state_mean = all_states.mean(axis=0).astype(np.float32) # np默认float64，但是pytorch默认float32，为了避免报dpble和float不匹配的错
        # 计算每一维特征的标准差（数据波动的剧烈程度）
        self.state_std = (all_states.std(axis=0)+1e-6).astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.index)
    
    # 取数据与 Padding 策略 (__getitem__)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        epi, start = self.index[idx]
        ep = self.episodes[epi]
        K = self.context_len
        L = len(ep.actions)

        # 1. 确定切片范围
        end = min(start + K, L) 
        # 比如快切到尾部的时候，
        # epi_len=5, start=3, k=3, 那么取[3,4]然后最后一个位置用0来padding
        n = end - start  # 实际取到的有效长度,真实长度（<=K）

        states = ep.states[start:end] # [n,state_dim]
        actions = ep.actions[start:end] # [n]
        rtg = ep.rtg[start:end] # [n]
        timesteps = ep.timesteps[start:end] # [n]

        # 右侧padding到K（关键：右padding能避免“前面全是padding导致attention全masked”）
        # 右侧padding符合不看未来（对应causal mask）
        # valid变量 前n个是1，后面是0（计算loss时用到）
        # 右侧 Padding (Valid Masking)，模型预测 a1 时只看 s1，预测 a2 时看 s1, s2,
        # 模型虽然也给后面的 Padding 部分输出了预测，但我们直接用 valid 把那一块的 Loss 乘 0 抹掉了
        valid = np.zeros((K,), dtype=np.float32)
        valid[:n] = 1.0

        states_pad = np.zeros((K, self.state_dim), dtype=np.float32)
        actions_pad = np.zeros((K,), dtype=np.int64) # padding 的 action 随便填个合法值即可
        rtg_pad = np.zeros((K,), dtype=np.float32)
        ts_pad = np.zeros((K,), dtype=np.int64)

        states_pad[:n] = states
        actions_pad[:n] = actions
        rtg_pad[:n] = rtg
        ts_pad[:n] = timesteps

        # ---状态归一化（只归一化 states，rtg 已经 scale 了----
        states_pad = (states_pad - self.state_mean) / self.state_std

        return {
            "states": torch.from_numpy(states_pad), # [K, state_dim]
            "actions": torch.from_numpy(actions_pad), # [K]
            "rtg": torch.from_numpy(rtg_pad),  # [K]
            "timesteps": torch.from_numpy(ts_pad),  # [K]
            "valid": torch.from_numpy(valid),  # [K]
        }

