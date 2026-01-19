# 测试 Decision Transformer 的“目标控制”能力，彻底的“环境隔离”测试
# 可以通过命令行参数 --target_return 随意改变目标 cartpole是500
# 从硬盘加载 .pt 文件，重新初始化环境。这是最真实的部署模拟。如果独立脚本跑出来的分和训练时日志显示的分一致，那才说明模型真的练成了

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch

from core.nn_utils import get_device, seed_everything
from envs.gym_factory import make_env, reset_env, step_env
from offline.decision_transformer import DT

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--target_return", type=float, default=None,
                    help="override checkpoint cfg.target_return")
    ap.add_argument("--sample_actions", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)

    # 从ckpt里面取出配置
    pack = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = pack["cfg"]
    state_mean = np.asarray(pack["state_mean"], dtype=np.float32)
    state_std = np.asarray(pack["state_std"], dtype=np.float32)

    seed_everything(int(cfg.get("seed", 42)))

    # 搭架子，随机初始化
    model = DT(
        state_dim=int(pack["state_dim"]),
        act_dim=int(pack["act_dim"]),
        context_len=int(cfg["context_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
        max_timestep=int(cfg["max_timestep"]),
    ).to(device)
    # 填充灵魂（训练好的参数值
    model.load_state_dict(pack["model"])
    model.eval()

    target_return = float(cfg["target_return"]) if args.target_return is None else float(args.target_return)
    rtg_scale = float(cfg["rtg_scale"])

    env = make_env(
        env_id=cfg["env_id"],
        seed=int(cfg.get("seed", 42)),
        pomdp_keep_idx=tuple(cfg["pomdp_keep_idx"]),
        history_len=int(cfg["history_len"]),
        use_delta_obs=bool(cfg["use_delta_obs"]),
    )()

    K = int(cfg["context_len"])
    D = int(pack["state_dim"])

    returns: List[float] = []

    for epi in range(int(args.episodes)):
        obs, _ = reset_env(env, seed=int(cfg.get("seed", 42)) + 10000 + epi)
        done = False
        ep_ret = 0.0

        states: List[np.ndarray] = []
        acts: List[int] = []
        rtgs: List[float] = []
        tss: List[int] = []

        rtg_now = target_return / rtg_scale
        t = 0

        while not done:
            # 归一化，状态对齐，模型只认识归一化后的数值
            s = np.asarray(obs, dtype=np.float32)
            s = (s - state_mean) / state_std

            states.append(s)
            # 存储过去的动作。注意其中的 acts.append(0) 只是一个占位符，等模型预测出当前动作后会立即被替换。
            acts.append(0)
            # 这是 DT 的“灵魂”。它告诉模型：“为了拿到剩下的分数，你应该怎么做？”
            rtgs.append(rtg_now)
            # 给 Transformer 提供时间位置信息
            tss.append(t)

            # 滑动窗口 (Sliding Window) 逻辑
            # 在强化学习的实时推理中，我们总是把最新的数据 append 到列表的末尾（右侧）
            # -k表示保留最后k个元素
            if len(states) > K:
                states = states[-K:]
                acts = acts[-K:]
                rtgs = rtgs[-K:]
                tss = tss[-K:]

            valid_len = len(states)

            states_pad = np.zeros((K, D), dtype=np.float32)
            actions_pad = np.zeros((K,), dtype=np.int64)
            rtg_pad = np.zeros((K,), dtype=np.float32)
            ts_pad = np.zeros((K,), dtype=np.int64)
            valid_pad = np.zeros((K,), dtype=np.float32)

            # padding
            # 虽然我们做了 states[-K:] 截断，但在游戏刚开始的前 K 步（比如第 5 步），
            # 我们的记忆列表只有 5 个元素，但模型结构要求输入必须是固定长度 K（比如 30）
            states_pad[:valid_len] = np.stack(states, axis=0)
            actions_pad[:valid_len] = np.asarray(acts, dtype=np.int64)
            rtg_pad[:valid_len] = np.asarray(rtgs, dtype=np.float32)
            ts_pad[:valid_len] = np.asarray(tss, dtype=np.int64)
            valid_pad[:valid_len] = 1.0

            # 转tensor准备喂给模型
            states_t = torch.tensor(states_pad, device=device).unsqueeze(0)
            actions_t = torch.tensor(actions_pad, device=device).unsqueeze(0)
            rtg_t = torch.tensor(rtg_pad, device=device).unsqueeze(0)
            ts_t = torch.tensor(ts_pad, device=device).unsqueeze(0)
            valid_t = torch.tensor(valid_pad, device=device).unsqueeze(0)

            # 从概率到决策
            # 调用模型进行推理
            a = model.act(
                states=states_t,
                actions=actions_t,
                rtg=rtg_t,
                timesteps=ts_t,
                valid=valid_t,
                valid_len=valid_len,
                sample=bool(args.sample_actions),
            )
            # 补回placeholder
            acts[-1]=a

            # step_env 付诸实践
            obs, r, terminated, truncated, _info = step_env(env, a)
            done = bool(terminated or truncated)
            ep_ret += float(r)

            # 动态目标调整,这给了模型一种“进度感”。它不断观察自己离目标还有多远，从而调整动作的激进程度
            # rtg_t+1 = rtg_t - r_t/scale 
            rtg_now = rtg_now - float(r) / rtg_scale
            t += 1

            returns.append(ep_ret)

    print(f"[eval] episodes={args.episodes} avg_return={float(np.mean(returns)):.1f}")


if __name__ == "__main__":
    main()