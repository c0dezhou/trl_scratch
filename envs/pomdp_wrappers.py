from __future__ import annotations

from collections import deque
from typing import Sequence, Tuple

import gymnasium as gym
import  numpy as np
from gymnasium.spaces import Box

# 放两个 wrapper：一个遮观测，一个堆叠历史。

# 制造“盲区”
# 这个类的作用是人为地删减观测维度
class MaskObsWrapper(gym.ObservationWrapper):
    """
    把原始 obs 只保留某些维度，比如 CartPole: keep_idx=(0,2) => [x, theta]
    """
    def __init__(self, env: gym.Env, keep_idx: Sequence[int]):
        super().__init__(env)
        self.keep_idx = np.asarray(keep_idx, dtype=int)

        assert isinstance(env.observation_space, Box)
        low = env.observation_space.low[self.keep_idx]
        high = env.observation_space.high[self.keep_idx]
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[self.keep_idx].astype(np.float32)

# 赋予模型“记忆”
# 既然信息被屏蔽了（比如没了速度），我们就给模型看最近T步的快照，让它自己通过对比快照来发现规律。
class HistoryStackWrapper(gym.Wrapper):
    """
    返回最近 T 步 obs，shape: (T, obs_dim)
    reset 时前 T-1 步用 0 padding
    """
    def __init__(self, env: gym.Env, history_len: int):
        super().__init__(env)
        assert history_len >= 1
        self.history_len = history_len
        # 核心数据结构deque双端队列，当队列满了再加入新数据时，最旧的数据会自动弹出（滑动窗口）
        self._buf = deque(maxlen = history_len)

        assert isinstance(env.observation_space, Box)
        obs_space: Box = env.observation_space

        # obs_space.low[None, :]即在第0维增加一维（插入一个新轴,在这个轴上好做repeat） (2,)->(1,2)-repeat->(T,2)
        low = np.repeat(obs_space.low[None, :], history_len, axis=0)
        high = np.repeat(obs_space.high[None, :], history_len, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs.astype(np.float32)

        # 1.清空旧的记忆(上一episode的记忆不能带到下一局)
        self._buf.clear()

        #2.制作一个“空白”的快照
        zero = np.zeros_like(obs, dtype=np.float32)

        #3.填充历史
        # reset应当返回当前局的第一帧观测（即起始状态），所以填充history_len - 1个0
        for _ in range(self.history_len-1):
            self._buf.append(zero)
        
        #4.最后塞入当前真实的初始观测（起始状态
        self._buf.append(obs)

        #5.堆叠返回
        # 结果就是：[0, 0, 0, 真实obs_0]
        return np.stack(self._buf, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs.astype(np.float32)
        self._buf.append(obs)
        return np.stack(self._buf, axis=0), reward, terminated, truncated, info



