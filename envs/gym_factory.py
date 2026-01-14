from __future__ import annotations 
# 允许在定义一个类的时候，在该类内部的方法中直接使用这个类本身作为返回值的类型注解。

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import gymnasium as gym
import numpy as np


@dataclass
class EnvSpec:
    env_id: str
    seed: int = 42
    render_mode: Optional[str] = None  # "human" / "rgb_array" / None


def make_env(spec: EnvSpec) -> gym.Env:
    """
    创建环境，并做一次 reset(seed) 来固定初始随机性。
    """
    env = gym.make(spec.env_id, spec.render_mode)
    env.reset(seed=spec.seed)
    return env


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
    返回：next_obs, reward, done, info
    """
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    return next_obs, float(reward), done, info


def get_obs_act_dims(env: gym.Env) -> Tuple[int, int]:
    """
    离散型： 最常见：Box(1D) observation + Discrete action（CartPole）
    """
    obs_space = env.observation_space
    act_space = env.action_space

    assert hasattr(obs_space, "shape") and len(obs_space.shape) == 1,f"不支持的obs空间：{obs_space}"
    obs_dim = obs_space.shape[0]

    assert hasattr(act_space,"n") and f"不支持的动作空间:{act_space}"
    act_dim = act_space.n

    return obs_dim, act_dim
