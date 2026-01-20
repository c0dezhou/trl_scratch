# 把一堆杂乱的离线轨迹（Offline Trajectories），切成一个个整齐划一的“时间窗口”喂给 Transformer
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class Episode:
    # 在切分时间窗口时，绝对不能跨 Episode 切分
    states: np.ndarray   # (L, state_dim)
    actions: np.ndarray  # (L,)
    dones: np.ndarray    # (L,)

class DynamicsSequenceDataset(Dataset):
    """把离线轨迹切成长度 K 的窗口，用于学习 dynamics。

    输出（单个样本）:
      states:       (K, state_dim)  float32, normalized
      actions:      (K,)            int64
      timesteps:    (K,)            int64
      valid:        (K,)            float32  (1=真实，0=padding)
      delta_targets:(K, state_dim)  float32, normalized-space delta
      done_targets: (K,)            float32  (0/1)
      trans_valid:  (K,)            float32  (1=有 next_state，0=无)
    """
    def __init__(
            self,
            npz_path: str,
            context_len: int,
            split: str="train",
            val_frac_episodes: float=0.1,
            seed: int=42,
            eps:float=1e-6,
    ):
        super().__init__()
        # 只能划分为val和train集
        assert split in ("train", "val"), f"切分只能是 train/val, got {split}"
        self.npz_path = npz_path
        self.K = int(context_len)
        self.split = split
        self.val_frac_episodes = float(val_frac_episodes)
        self.seed = int(seed)
        self.eps = float(eps)

        # 1.数据加载与解包
        data = np.load(npz_path)
        obs = data["obs"].astype(np.float32)          # (N, state_dim)
        actions = data["actions"].astype(np.int64)    # (N,)
        dones = data["dones"].astype(np.float32)      # (N,)
        ends = data["episode_ends"].astype(np.int64)  # (E,)

        # 一个状态包含多少个特征
        # obs : [N, D], -1取D
        self.state_dim = int(obs.shape[-1]) # cartpole 的state dim是4
        # 侦察动作维度
        # .reshape(())：这是 NumPy 的一个小技巧，强制把它变成 0 维（标量），防止形状不对，for .item()
        # data file是npz标准数据集（有act_dim这个kv），如果没有，推断（猜测），扫描整个数据集的actions数组 ,找到最大的数字
        # 隐患：默认任务是离散动作 for actions.max()+1  TODO:如果要做mojuco，要改代码
        self.act_dim = int(data["act_dim"].reshape(()).item()) if "act_dim" in data.files else int(actions.max()+1)

        # 2.还原episode(切)
        episodes: List[Episode] = []
        st = 0
        for e in ends:
            e = int(e)
            ep_states = obs[st:e]
            ep_actions = actions[st:e]
            ep_dones = dones[st:e]
            episodes.append(Episode(ep_states,ep_actions,ep_dones))
            st = e
        self.episodes = episodes

        # 统计 mean/std（全数据，不分 split：保持一致，推理也好用）
        flat_states = np.concatenate([ep.states for ep in episodes], axis=0)
        self.state_mean = flat_states.mean(axis=0).astype(np.float32)
        self.state_std = (flat_states.std(axis=0) + self.eps).astype(np.float32)

        # 划分训练集和验证集 episode-level split 避免窗口切分导致泄露
        # 创建一个专属的随机数生成器，并指定种子,
        rng = np.random.RandomState(self.seed)
        idx = np.arange(len(self.episodes)) # 生成索引数组然后打乱
        rng.shuffle(idx) # 局部,而不是全聚德np random,不受别人的全局种子影响
        n_val = max(1, int(len(idx)* self.val_frac_episodes)) # 验证集的数量
        val_set = set(idx[:n_val].tolist()) # 验证集，取shuffle后的前nval个epi
        if split == "val":
            use_eps = [i for i in range(len(self.episodes)) if i in val_set]
        else: # tarin
            use_eps = [i for i in range(len(self.episodes)) if i not in val_set]
        self.use_episode_ids = use_eps

        # 构建窗口索引 （episode_id，start_t)
        self.index: List[Tuple[int, int]] = []
        for eid in self.use_episode_ids:
            L = self.episodes[eid].states.shape[0] # 每一局的长度，states 数组的行数就代表了这局游戏总共经历了多少个时间步
            # 每个 step 都可以作为窗口起点（最后几步会 padding）
            for s in range(L):
                self.index.append((eid,s))

    def __len__(self) -> int:
        return len(self.index)
        
    def _norm_states(self, s:np.ndarray)->np.ndarray:
        return (s-self.state_mean)/self.std
        
    def __getitem__(self, idx:int) -> Dict[str,torch.Tensor]:
        eid, start = self.index[idx]
        ep = self.episodes[eid]
        states = ep.states
        actions = ep.actions
        dones = ep.dones

        L = states.shape[0]
        K = self.K

        # 如果 start 位置靠近 Episode 的结尾，那么 start + K 就会超过总长度 L
        end = min(L, start+K)
        n = end-start

        s_win = states[start:end]           # (n, D)
        a_win = actions[start:end]          # (n,)
        d_win = dones[start:end]            # (n,)

        # next states 对齐 to each (s_t, a_t)
        # 对于最后一步 (t=L-1)，没有 next_state
        # 初始化 ns (next states) 容器
        ns = np.zeros((n, self.state_dim), dtype=np.float32)
        trans_valid = np.zeros((n,), dtype=np.float32)
        # 遍历当前窗口内的每一个 step
        for i in range(n):
            t = start + i
            if (t + 1) < L:
                ns[i] = states[t + 1] # 拿到真正的下一帧
                trans_valid[i] = 1.0 # 标记：这一步是有“未来”的
        
        # normalize
        # 将当前状态和下一时刻状态都映射到均值为 0、方差为 1 的空间
        s_norm = self._norm_states(s_win)                 # (n, D)
        ns_norm = self._norm_states(ns)                   # (n, D)（对 trans_valid=0 的行无所谓）
        """为什么不直接预测 ns_norm？ 在物理环境中，相邻两帧的变化往往很小。如果模型直接预测下一帧，
        它可能会偷懒：直接把当前帧 s_t 当作结果输出，Loss 也会很低，但模型啥也没学到。
        预测“变化量” (\Delta)：强迫模型去关注动作 a_t 到底对环境产生了什么物理影响（加速度、角速度变化等）。
        这对于学习动力学（Dynamics）至关重要。"""
        delta_norm = (ns_norm - s_norm).astype(np.float32)

        # padding 对齐补齐
        # pad容器初始化
        states_pad = np.zeros((K, self.state_dim), dtype=np.float32)
        actions_pad = np.zeros((K,), dtype=np.int64)
        dones_pad = np.zeros((K,), dtype=np.float32)
        delta_pad = np.zeros((K, self.state_dim), dtype=np.float32)
        valid_pad = np.zeros((K,), dtype=np.float32)
        trans_valid_pad = np.zeros((K,), dtype=np.float32)
        timesteps_pad = np.zeros((K,), dtype=np.int64)

        states_pad[:n] = s_norm
        actions_pad[:n] = a_win
        dones_pad[:n] = d_win
        delta_pad[:n] = delta_norm
        valid_pad[:n] = 1.0 # 前 n 位是 1.0：表示“这是真实数据，请关注”。
        trans_valid_pad[:n] = trans_valid # 后 K-n 位是 0.0：表示“这是补零的垃圾信息，请忽略”
        """
        标志位,含义,作用
        valid_pad,窗口内哪几帧是真实的。,用于 Transformer 的 Self-Attention Mask，防止模型关注到补零的区域。
        trans_valid,哪几帧有对应的“下一帧”。,用于 Loss Mask。Episode 最后一步虽然是真实数据，但它没未来，不能用来算预测 Loss。
        """
        # episode内时间戳（从 start 开始递增）
        timesteps_pad[:n] = np.arange(start, start + n, dtype=np.int64)

        # 返回tensor
        return {
            "states": torch.from_numpy(states_pad),
            "actions": torch.from_numpy(actions_pad),
            "timesteps": torch.from_numpy(timesteps_pad),
            "valid": torch.from_numpy(valid_pad),
            "delta_targets": torch.from_numpy(delta_pad),
            "done_targets": torch.from_numpy(dones_pad),
            "trans_valid": torch.from_numpy(trans_valid_pad),
        }

        



