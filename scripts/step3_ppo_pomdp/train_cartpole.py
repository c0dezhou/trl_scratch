from __future__ import annotations

import argparse
import importlib
import time
import numpy as np
import torch

from core.nn_utils import seed_everything, get_device
from envs.gym_factory import EnvSpec, make_env, reset_env, step_env
from envs.pomdp_wrappers import MaskObsWrapper, HistoryStackWrapper
from models.actor_critic_mlp import ActorCriticMLP
from models.actor_critic_gpt import ActorCriticGPT
from rl.buffer import RolloutBuffer
from rl.ppo import ppo_update


def flatten_obs(obs) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return obs.reshape(-1)


def load_cfg(module_path: str):
    """
    支持 config 文件里导出 CFG 实例 + 可选 RUN_NAME
    """
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "CFG"):
        raise ValueError(f"{module_path} 没有导出 CFG，请在文件末尾加：CFG = XXXConfig()")
    cfg = mod.CFG
    run_name = getattr(mod, "RUN_NAME", module_path.split(".")[-1])
    return cfg, run_name


def train_once(cfg, run_name: str):
    seed_everything(cfg.seed)
    device = get_device(getattr(cfg, "device", "cpu"))

    print(
        f"\n[run={run_name}] device={device}, env_id={cfg.env_id}, seed={cfg.seed}, "
        f"model={cfg.model_type}, keep_idx={cfg.pomdp_keep_idx}, history_len={cfg.history_len}"
    )

    # env + wrappers
    # --- env ---
    env = make_env(
        cfg.env_id,
        cfg.seed,
        pomdp_keep_idx=cfg.pomdp_keep_idx,
        history_len=cfg.history_len,
    )() # ← 这一对括号很关键：真正创建环境

    obs_shape = env.observation_space.shape
    act_dim = env.action_space.n

    # obs_shape 可能是 (D,) 或 (T, D)
    if len(obs_shape) == 1:
        T, D = 1, obs_shape[0]
    else:
        T, D = obs_shape[0], obs_shape[1]

    flat_obs_dim = T * D
    print(f"[step3] obs_shape={obs_shape}, flat_obs_dim={flat_obs_dim}, act_dim={act_dim}")


    # model
    if cfg.model_type == "mlp":
        model = ActorCriticMLP(flat_obs_dim, act_dim, hidden=cfg.hidden).to(device)
    else:
        model = ActorCriticGPT(
            obs_dim=D,
            act_dim=act_dim,
            history_len=T,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # state
    obs, _ = reset_env(env, seed=cfg.seed)
    obs = flatten_obs(obs)

    ep_ret = 0.0
    ep_rets: list[float] = []
    ep_count = 0
    t0 = time.time()
    best_score = -1e9
    solved_cnt = 0
    best_path = f"best_{run_name}.pt"

    for update in range(1, cfg.total_updates + 1):
        buf = RolloutBuffer(cfg.rollout_steps, flat_obs_dim, device=device)

        term_cnt = 0
        trunc_cnt = 0
        max_ep_len = 0
        cur_ep_len = 0

        model.eval()
        for _ in range(cfg.rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_t, logp_t, value_t = model.act(obs_t)

            action = int(action_t.item())
            next_obs, reward_env, terminated, truncated, _ = step_env(env, action)
            next_obs = flatten_obs(next_obs)

            done = terminated or truncated
            timeout = bool(truncated and (not terminated))

            terminal_value = 0.0
            if timeout:
                with torch.no_grad():
                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                    _, terminal_value = model(next_obs_t)

            buf.add(
                obs=obs_t,
                action=action_t,
                logp=logp_t,
                reward=float(reward_env),
                done=done,
                value=value_t,
                timeout=timeout,
                terminal_value=float(terminal_value),
            )

            ep_ret += float(reward_env)
            obs = next_obs

            cur_ep_len += 1
            if terminated:
                term_cnt += 1
            if truncated:
                trunc_cnt += 1

            if done:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                ep_count += 1
                max_ep_len = max(max_ep_len, cur_ep_len)
                cur_ep_len = 0

                obs, _ = reset_env(env, seed=cfg.seed + ep_count)
                obs = flatten_obs(obs)

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            _, last_value = model(obs_t)

        buf.compute_gae(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda)

        model.train()
        ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buf,
            update_epochs=cfg.update_epochs,
            minibatch_size=cfg.minibatch_size,
            clip_coef=cfg.clip_coef,
            vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
            max_grad_norm=cfg.max_grad_norm,
        )

        avg100 = float(np.mean(ep_rets[-100:])) if len(ep_rets) > 0 else 0.0
        if avg100 > best_score:
            best_score = avg100
            torch.save(model.state_dict(), best_path)

        solved_cnt = (solved_cnt + 1) if (avg100 >= 475) else 0

        if update % cfg.print_every == 0:
            print(
                f"[run={run_name}] update {update:3d}/{cfg.total_updates} | "
                f"terminated={term_cnt}, truncated={trunc_cnt}, max_ep_len={max_ep_len} | "
                f"episodes {len(ep_rets):4d} | avg_return(last100) {avg100:7.1f} | "
                f"time {time.time() - t0:.1f}s"
            )

        if solved_cnt >= 5:
            print(f"[run={run_name}] Solved, early stop.")
            break

    env.close()
    print(f"[run={run_name}] done. best_ckpt={best_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs.step3_ppo_cartpole_pomdp_tr"],  # 你可以改默认
        help="One or more config module paths, e.g. configs.step3_xxx configs.step3_yyy",
    )
    args = parser.parse_args()

    for mod_path in args.configs:
        cfg, run_name = load_cfg(mod_path)
        train_once(cfg, run_name)


if __name__ == "__main__":
    main()
