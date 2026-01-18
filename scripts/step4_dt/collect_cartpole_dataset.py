# 用 step3 策略采离线数据(数据收集器)
# 请出一个练得不错的“老师傅”（Step 3 训好的 PPO 模型），让它去玩 200 局游戏，把所有的操作过程（状态、动作、奖励）录制成一个 .npz 文件
"""
这个脚本干三件事：

用 step3 的 config 重建环境与 ActorCritic（GPT/MLP 都支持）
加载 step3 的 best ckpt（state_dict）
跑 N 个 episode，把“每步当前观测 + 动作 + reward”存成一个 npz

注意：默认存的是“当前时刻的观测”，不是 history stack 的 flatten，这样 DT 才真的需要用 context_len 来“记忆”
"""

from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from typing import Any, Tuple, List

import numpy as np
import torch

from envs.gym_factory import make_env, reset_env, step_env
from core.nn_utils import seed_everything, get_device
from models.actor_critic_gpt import ActorCriticGPT
from models.actor_critic_mlp import ActorCriticMLP
from torch.distributions import Categorical

def load_cfg(cfg_path: str):
    # example: configs.step3_ppo_cartpole_pompd_tr
    mod = importlib.import_module(cfg_path)
    return mod.CFG


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    # 传给ppo (n帧)
    """flatten_obs：把 [History_Len, State_Dim] 拍扁成一维。
    这是给 PPO 模型 吃的，因为它需要这种格式来做前向传播"""
    # obs 可能是 [T, D]，也可能是 [D], 不管是啥通通拍扁
    return obs.reshape(-1).astype(np.float32)

def current_frame(obs: np.ndarray) -> np.ndarray:
    # 传给dt (1帧)
    """只取 obs[-1]。这是存入 Dataset 的，专门给未来的 DT 徒弟看。
    DT 会自己通过 Transformer 重新把这些单帧拼成时序"""
    if obs.ndim == 2:
        # 使用了“历史堆叠”（History Stack）(比如history_len=5 obs=[5,4]的矩阵)
        return obs[-1].astype(np.float32)
    # 没用堆叠（Single Frame） 如果 history_len = 1，环境直接给你一个形状为 [4] 的向量
    return obs.astype(np.float32)

def build_policy_model(cfg, obs0: np.ndarray, act_dim: int) -> torch.nn.Module:
    """重建step3策略网络，让我们可以加载检查点"""
    model_type = getattr(cfg, "model_type", "gpt")

    # obs0 is after wrappers,可能是[T,D] or [D]
    if obs0.ndim == 2:
        history_len = int(obs0.shape[0])
        obs_dim = int(obs0.shape[1])
    else:
        history_len = int(getattr(cfg, "history_len", 1))
        obs_dim = int(obs0.shape[0])

    if model_type == "gpt":
        # ActorCriticGPT 接受的是单步维度和历史长度,gpt不需要知道flat_dim
        return ActorCriticGPT(
            obs_dim=obs_dim,
            act_dim=act_dim,
            history_len=int(getattr(cfg, "history_len", history_len)),
            d_model=int(getattr(cfg, "d_model", 128)),
            n_heads=int(getattr(cfg, "n_heads", 4)),
            n_layers=int(getattr(cfg, "n_layers", 2)),
            dropout=float(getattr(cfg, "dropout", 0.1)),
        )
    
    # MLP 接受的是拍扁后的总维度
    flat_dim = int(flatten_obs(obs0).shape[0])
    return ActorCriticMLP(
        obs_dim=flat_dim,
        act_dim=act_dim,
        hidden=int(getattr(cfg, "hidden_dim", 128)),
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_config", type=str, required=True,
                    help="step3 config module path, e.g. configs.step3_ppo_cartpole_pomdp_tr")
    ap.add_argument("--policy_ckpt", type=str, required=True,
                    help="step3 best checkpoint path (.pt state_dict)")
    ap.add_argument("--out", type=str, default="data/cartpole_pomdp_from_step3.npz")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--sample_actions", action="store_true",
                    help="sample from policy distribution (more diverse); default argmax")
    args = ap.parse_args()

    cfg = load_cfg(args.policy_config)
    seed_everything(args.seed)

    # 按照step3创建环境
    env = make_env(
        env_id=getattr(cfg, "env_id", "CartPole-v1"),
        seed=args.seed,
        pomdp_keep_idx=tuple(getattr(cfg, "pomdp_keep_idx", (0, 2))),
        history_len=int(getattr(cfg, "history_len", 1)),
        use_delta_obs=bool(getattr(cfg, "use_delta_obs", False)),
    )()

    obs0, _ = reset_env(env, seed=args.seed)
    act_dim = int(env.action_space.n)

    device = get_device(args.device)
    # （全是随机初始化的乱码参数）
    model = build_policy_model(cfg, obs0, act_dim).to(device)

    # 加载step3训练好的pt文件
    sd = torch.load(args.policy_ckpt, map_location=device)
    # 把pt里面读出来的参数值装到model中
    model.load_state_dict(sd)
    # 不是为了训练，而是为了录制教学视频，所以用eval（关闭drop
    model.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    rew_buf: List[float] = []
    done_buf: List[float] = []
    episode_ends: List[int] = []

    total_steps = 0

    # 录制DT要用的npz
    for ep in range(args.episodes):
        # 每局一个新开端，给不同的seed保证训练时初始状态的差异
        # + 10000 的小细节：这通常是为了确保这些种子和你在训练 PPO 时用的种子（通常是 0-100 这种小数字）完全错开，避免数据重叠，保证数据的新鲜度
        obs, _ = reset_env(env, seed=args.seed + 10000 + ep)
        # 每局计分清空
        ep_ret = 0.0

        while True:
            # policy input 是flattened的history stack
            obs_flat = flatten_obs(obs)
            # PyTorch 的模型永远默认输入是成批处理的,所以要加unsqueeze0，前面加一个批次维度
            obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)

            # 让 PPO 去环境里跑 200 局，把它的所见所闻（State, Action, Reward）录制下来，保存成 .npz 文件
            with torch.no_grad():
                # logits：这是模型的输出结果，通常是未归一化的动作分数
                # _v：这是 PPO 模型的“价值头（Value Head）”输出，预测当前局势能拿多少分
                logits, _v = model(obs_t)

            if args.sample_actions:
                # 给 DT 容错机会：DT 学习采样出的数据，能看到“如果稍微走偏了一点，后面怎么救回来”。
                # Categorical 返回的是一个概率分布对象，而不是单纯的数值。
                # 你可以把它想象成一个抽奖箱
                """
                dist = Categorical(logits=logits)  # 1. 制造一个“抽奖箱”，内部概率根据 logits 决定
                a = dist.sample()                # 2. 从抽奖箱里“摸”一个球出来（这就是抽样）
                a = a.item()                     # 3. 把 PyTorch 的张量变成 Python 的数字
                """
                dist = Categorical(logits=logits)
                a = int(dist.sample().item())
            else:
                a = int(torch.argmax(logits, dim=-1).item())
            
            # 记录每一帧的动作，放入缓冲区buffer
            # DT dataset 只存当前帧 current frame (no history stack)
            # 1.数据清洗（抠出最新一帧
            s = current_frame(obs)

            # 2.环境交互（因果发生
            next_obs, r, terminated, truncated, _info = step_env(env, a)
            done = bool(terminated or truncated)

            # 3.存buffer
            obs_buf.append(s)
            act_buf.append(a)
            rew_buf.append(float(r))
            done_buf.append(1.0 if done else 0.0)

            ep_ret += float(r)
            total_steps += 1

            obs = next_obs
            if done:
                # 记录这一局结束时的当前总步数
                episode_ends.append(total_steps)  # end index (exclusive)
                print(f"[collect] ep={ep:4d} return={ep_ret:7.1f} steps_total={total_steps}")
                break

    obs_arr = np.asarray(obs_buf, dtype=np.float32)
    act_arr = np.asarray(act_buf, dtype=np.int64)
    rew_arr = np.asarray(rew_buf, dtype=np.float32)
    done_arr = np.asarray(done_buf, dtype=np.float32)
    ends_arr = np.asarray(episode_ends, dtype=np.int64)

    np.savez(
        args.out,
        obs=obs_arr,
        actions=act_arr,
        rewards=rew_arr,
        dones=done_arr,
        episode_ends=ends_arr,
        act_dim=np.int64(act_dim),
    )

    print(f"[collect] saved: {args.out}")
    print(f"[collect] obs shape={obs_arr.shape} act_dim={act_dim} episodes={len(ends_arr)}")


if __name__ == "__main__":
    main()




