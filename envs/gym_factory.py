from __future__ import annotations 
# 允许在定义一个类的时候，在该类内部的方法中直接使用这个类本身作为返回值的类型注解。

from dataclasses import dataclass
from typing import Optional, Tuple, Any
from envs.pomdp_wrappers import MaskObsWrapper, HistoryStackWrapper

import gymnasium as gym
import numpy as np


@dataclass
class EnvSpec:
    env_id: str
    seed: int = 42
    render_mode: Optional[str] = None  # "human" / "rgb_array" / None


# def make_env(spec: EnvSpec) -> gym.Env:
#     """
#     创建环境，并做一次 reset(seed) 来固定初始随机性。
#     """
#     env = gym.make(spec.env_id, render_mode=spec.render_mode)
#     env.reset(seed=spec.seed)
#     return env
"""
当调用这个工厂函数产生的环境时，数据是这样流动的：
    内核：原始环境产生 [x, v, theta, omega]。
    Mask 层：看到内核的数据，切掉速度，只剩 [x, theta]。
    History 层：看到 Mask 层的数据，把它塞进队列，返回 [[过去3帧...], [当前帧]]。
    模型：最终拿到的是一个时空矩阵。
"""
def make_env(env_id:str, seed: int, *, pomdp_keep_idx=None, history_len: int = 1, **kwargs):
# *是指示它之后的参数必须以kv方式传入，**是打包传入剩下所有参数
    def thunk(): # 闭包返回“说明书”，真正的环境不会在make_env时创建
        env = gym.make(env_id, **kwargs)
        env.reset(seed=seed)

        if pomdp_keep_idx is not None:
            env = MaskObsWrapper(env, keep_idx=pomdp_keep_idx)
        
        if history_len and history_len > 1:
            env = HistoryStackWrapper(env, history_len=history_len)
        
        return env
    return thunk


def reset_env(env: gym.Env, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    统一 reset：返回 (obs, info)
    """
    if seed is None:
        return env.reset()
    return env.reset(seed=seed)


def step_env(env: gym.Env, action: Any):
    """
    统一 step（兼容 gymnasium 的 terminated/truncated）
    返回：next_obs, reward, terminated, truncated, info
    """
    next_obs, reward, terminated, truncated, info = env.step(action)
    return next_obs, float(reward), bool(terminated), bool(truncated), info


def get_obs_act_dims(env: gym.Env) -> Tuple[int, int]:
    """
    离散型： 最常见：Box(1D) observation + Discrete action（CartPole）
    """
    obs_space = env.observation_space
    act_space = env.action_space

    assert hasattr(obs_space, "shape") and len(obs_space.shape) == 1,f"不支持的obs空间：{obs_space}"
    obs_dim = obs_space.shape[0]

    assert hasattr(act_space,"n") , f"不支持的动作空间:{act_space}"
    act_dim = act_space.n

    return obs_dim, act_dim
