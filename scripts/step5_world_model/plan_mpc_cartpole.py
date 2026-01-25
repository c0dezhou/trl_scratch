from __future__ import annotations
# Step5 的“在线评估器”：拿训练好的 DynamicsTransformer（世界模型），
# 在真实 Gym 环境里用 MPC（随机射击/采样规划） 来控制 CartPole，然后打印每局回报。
# 模型负责“想象未来”，MPC 负责“挑一条未来最爽的动作序列”，最后只执行第一步动作
"""
流程是一条直线：

--ckpt 加载世界模型 checkpoint（包含 cfg、均值方差、权重）
构造 DynamicsTransformer 并 load_state_dict
构造 Gym env（history_len=1，让环境只吐当前帧）
每个 episode 循环：
    用 deque 维护最近 K 步的 (obs, action, timestep) 历史
    调用 mpc_action(...)：对 很多条候选动作序列做模型 rollout，打分，选最优第一步动作
    把动作扔给真实 env step_env
输出每局回报和平均回报

它不训练，只评估（model.eval() + @torch.no_grad()）。
"""
import argparse
from collections import deque

import numpy as np
import torch

from core.nn_utils import get_device
from envs.gym_factory import make_env, reset_env, step_env
from model_based.dynamics_transformer import DynamicsTransformer
from model_based.mpc import MPCConfig, mpc_action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device("cuda")

    pack = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = pack["cfg"]

    model = DynamicsTransformer(
        state_dim=int(pack["state_dim"]),
        act_dim=int(pack["act_dim"]),
        context_len=int(cfg["context_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        d_ff=int(cfg["d_ff"]),
        dropout=float(cfg["dropout"]),
        max_timestep=int(cfg["max_timestep"]),
    ).to(device)
    model.load_state_dict(pack["model"])
    model.eval()

    state_mean = np.asarray(pack["state_mean"], dtype=np.float32)
    state_std = np.asarray(pack["state_std"], dtype=np.float32)

    env = make_env(
        env_id=cfg["env_id"],
        seed=int(cfg.get("seed", 42)),
        pomdp_keep_idx=tuple(cfg["pomdp_keep_idx"]),
        history_len=1,  # MPC online eval 用 1（模型吃序列）
        # 也就是说：记忆放到模型侧，不放到 env wrapper 里，避免重复堆叠导致信息冗余
        use_delta_obs=bool(cfg["use_delta_obs"]),
    )()

    mpc_cfg = MPCConfig(
        horizon=int(args.horizon),
        num_samples=int(args.num_samples),
        done_threshold=float(cfg.get("mpc_done_threshold", 0.5)),
        gamma=float(cfg.get("mpc_gamma", 1.0)),
        seed=int(args.seed),
    )

    K = int(cfg["context_len"])

    returns = []
    for ep in range(int(args.episodes)):
        obs, _ = reset_env(env, seed=int(args.seed) + ep * 1000)
        ep_ret = 0.0
        t = 0
    
        # 用 deque 维护历史
        # history buffers: align states/actions/timesteps to the right
        # actions 与 states 同长度，最后一个 action 是“当前待决策”的占位符（会在 mpc_action 内被 overwrite）
        obs_hist = deque(maxlen=K)
        act_hist = deque(maxlen=K)
        t_hist = deque(maxlen=K)

        while True:
            # 先把“当前时刻”塞进历史
            obs_hist.append(np.asarray(obs, dtype=np.float32))
            act_hist.append(0)
            t_hist.append(t)

            # 转成 numpy 数组喂给 mpc_action
            obs_arr = np.stack(list(obs_hist), axis=0)                  # (L,D)
            act_arr = np.asarray(list(act_hist), dtype=np.int64)        # (L,)
            t_arr = np.asarray(list(t_hist), dtype=np.int64)            # (L,)

            a = mpc_action(
                model=model,
                obs_hist_raw=obs_arr,
                act_hist=act_arr,
                t_hist=t_arr,
                state_mean=state_mean,
                state_std=state_std,
                cfg=mpc_cfg,
                device=device,
            )

            # 把当前占位符 action 替换为真实执行的 action（用于下一步历史）
            act_hist[-1] = a

            obs, r, terminated, truncated, _ = step_env(env, a)
            ep_ret += float(r)
            t += 1

            if terminated or truncated:
                break

        returns.append(ep_ret)
        print(f"[mpc] ep={ep:03d} return={ep_ret:.1f}")

    print(f"[mpc] episodes={len(returns)} avg_return={np.mean(returns):.1f}")


if __name__ == "__main__":
    main()
