from __future__ import annotations

import argparse
import importlib
import os
import time
import numpy as np
import torch
import gymnasium as gym
from torch.distributions import Categorical

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

def _peek_history(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        head = obs[:3]
        tail = obs[-3:] if obs.size >= 3 else obs
        return head, tail
    head = obs[:3]
    tail = obs[-3:]
    return head, tail

def debug_pomdp_env(cfg) -> None:
    if not getattr(cfg, "debug_pomdp", False):
        return

    print("[debug] POMDP sanity checks")

    if cfg.pomdp_keep_idx is not None:
        base_env = gym.make(cfg.env_id)
        full_obs, _ = base_env.reset(seed=cfg.seed)
        base_env.close()

        wrapped_env = make_env(
            cfg.env_id,
            cfg.seed,
            pomdp_keep_idx=cfg.pomdp_keep_idx,
            history_len=1,
            use_delta_obs=getattr(cfg, "use_delta_obs", False),
        )()
        keep_obs, _ = reset_env(wrapped_env, seed=cfg.seed)
        wrapped_env.close()

        keep_obs = np.asarray(keep_obs, dtype=np.float32)
        if keep_obs.ndim > 1:
            keep_obs = keep_obs[-1]
        expected = np.asarray(full_obs, dtype=np.float32)[list(cfg.pomdp_keep_idx)]
        print(f"[debug] full={full_obs} keep={keep_obs} expected={expected}")

    if cfg.history_len and cfg.history_len > 1:
        hist_env = make_env(
            cfg.env_id,
            cfg.seed,
            pomdp_keep_idx=cfg.pomdp_keep_idx,
            history_len=cfg.history_len,
            use_delta_obs=getattr(cfg, "use_delta_obs", False),
        )()
        obs, _ = reset_env(hist_env, seed=cfg.seed)
        head, tail = _peek_history(obs)
        print(f"[debug] reset head={head} tail={tail}")

        steps = int(getattr(cfg, "debug_history_steps", 5))
        for i in range(steps):
            action = hist_env.action_space.sample()
            obs, _, terminated, truncated, _ = step_env(hist_env, action)
            head, tail = _peek_history(obs)
            print(f"[debug] step{i} term={terminated} trunc={truncated} head={head} tail={tail}")
            if terminated or truncated:
                obs, _ = reset_env(hist_env, seed=cfg.seed + 100 + i)
                head, tail = _peek_history(obs)
                print(f"[debug] after reset head={head} tail={tail}")
        hist_env.close()

def eval_policy(model, cfg, n_episodes: int = 5, *, greedy: bool = True) -> tuple[float, float]:
    eval_env = make_env(
        cfg.env_id,
        cfg.seed + 9999,
        pomdp_keep_idx=cfg.pomdp_keep_idx,
        history_len=cfg.history_len,
        use_delta_obs=getattr(cfg, "use_delta_obs", False),
    )()
    model.eval()
    rets: list[float] = []
    device = next(model.parameters()).device

    for i in range(n_episodes):
        obs, _ = reset_env(eval_env, seed=cfg.seed + 10000 + i)
        obs = flatten_obs(obs)
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.inference_mode():
                logits, _ = model(obs_t)
                if greedy:
                    action = int(torch.argmax(logits).item())
                else:
                    dist = Categorical(logits=logits)
                    action = int(dist.sample().item())
            obs, r, terminated, truncated, _ = step_env(eval_env, action)
            obs = flatten_obs(obs)
            done = terminated or truncated
            ep_ret += float(r)
        rets.append(ep_ret)

    eval_env.close()
    return float(np.mean(rets)), float(np.std(rets))

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

    debug_pomdp_env(cfg)

    # env + wrappers
    # --- env ---
    env = make_env(
        cfg.env_id,
        cfg.seed,
        pomdp_keep_idx=cfg.pomdp_keep_idx,
        history_len=cfg.history_len,
        use_delta_obs=getattr(cfg, "use_delta_obs", False),
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
        t_rollout = time.time()
        for _ in range(cfg.rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.inference_mode():
                action_t, logp_t, value_t = model.act(obs_t)

            action = int(action_t.item())
            next_obs, reward_env, terminated, truncated, _ = step_env(env, action)
            next_obs = flatten_obs(next_obs)

            done_env = terminated or truncated
            done_bootstrap = terminated
            timeout = bool(truncated and (not terminated))

            terminal_value = 0.0
            if timeout:
                with torch.inference_mode():
                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                    _, terminal_value = model(next_obs_t)

            buf.add(
                obs=obs_t,
                action=action_t,
                logp=logp_t,
                reward=float(reward_env),
                done=done_bootstrap,
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

            if done_env:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                ep_count += 1
                max_ep_len = max(max_ep_len, cur_ep_len)
                cur_ep_len = 0

                obs, _ = reset_env(env, seed=cfg.seed + ep_count)
                obs = flatten_obs(obs)

        rollout_s = time.time() - t_rollout

        with torch.inference_mode():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            _, last_value = model(obs_t)

        buf.compute_gae(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda)

        model.train()
        t_update = time.time()
        minibatch_size = min(cfg.minibatch_size, buf.ptr)
        with torch.enable_grad():
            ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buf,
                update_epochs=cfg.update_epochs,
                minibatch_size=minibatch_size,
                clip_coef=cfg.clip_coef,
                vf_coef=cfg.vf_coef,
                ent_coef=cfg.ent_coef,
                max_grad_norm=cfg.max_grad_norm,
                target_kl=getattr(cfg, "target_kl", None),
                clip_vloss=getattr(cfg, "clip_vloss", False),
            )
        update_s = time.time() - t_update

        avg100 = float(np.mean(ep_rets[-100:])) if len(ep_rets) > 0 else 0.0
        eval_mean = None
        eval_std = None
        eval_s = 0.0
        if getattr(cfg, "eval_every", 0) and update % cfg.eval_every == 0:
            t_eval = time.time()
            eval_mean, eval_std = eval_policy(
                model, cfg, n_episodes=getattr(cfg, "eval_episodes", 5), greedy=True
            )
            eval_s = time.time() - t_eval

        if avg100 > best_score:
            best_score = avg100
            torch.save(model.state_dict(), best_path)

        solved_cnt = (solved_cnt + 1) if (avg100 >= 475) else 0

        if update % cfg.print_every == 0:
            total_cycle = rollout_s + update_s + eval_s
            pct_roll = 100.0 * rollout_s / total_cycle if total_cycle > 0 else 0.0
            pct_upd = 100.0 * update_s / total_cycle if total_cycle > 0 else 0.0
            pct_eval = 100.0 * eval_s / total_cycle if total_cycle > 0 else 0.0
            steps_collected = buf.ptr
            sps = steps_collected / max(rollout_s, 1e-8)

            if getattr(cfg, "log_verbose", False):
                print(
                    f"[run={run_name}] update {update:3d}/{cfg.total_updates} | "
                    f"terminated={term_cnt}, truncated={trunc_cnt}, max_ep_len={max_ep_len} | "
                    f"episodes {len(ep_rets):4d} | avg_return(last100) {avg100:7.1f} | "
                    f"time {time.time() - t0:.1f}s | "
                    f"rollout {rollout_s:.2f}s ({pct_roll:.1f}%) | "
                    f"update {update_s:.2f}s ({pct_upd:.1f}%) | "
                    f"eval {eval_s:.2f}s ({pct_eval:.1f}%) | "
                    f"steps {steps_collected} | sps {sps:.1f}"
                )
                if eval_mean is not None:
                    print(
                        f"[run={run_name}] eval greedy: mean={eval_mean:.1f} std={eval_std:.1f}"
                    )
            else:
                line = (
                    f"[run={run_name}] upd {update:3d}/{cfg.total_updates} | "
                    f"max_ep_len={max_ep_len} | avg100 {avg100:5.1f} | steps {steps_collected} | sps {sps:6.0f}"
                )
                if eval_mean is not None:
                    line += f" | eval {eval_mean:.1f}±{eval_std:.1f}"
                print(line)

        if solved_cnt >= 5:
            print(f"[run={run_name}] Solved, early stop.")
            break

    env.close()

    final_eval_episodes = int(getattr(cfg, "final_eval_episodes", 0))
    if final_eval_episodes > 0 and os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
        best_mean, best_std = eval_policy(
            model, cfg, n_episodes=final_eval_episodes, greedy=True
        )
        print(
            f"[run={run_name}] best_ckpt eval: mean={best_mean:.1f} std={best_std:.1f} "
            f"(n={final_eval_episodes})"
        )

        sample_episodes = int(getattr(cfg, "final_eval_sample_episodes", 0))
        if sample_episodes > 0:
            sample_mean, sample_std = eval_policy(
                model, cfg, n_episodes=sample_episodes, greedy=False
            )
            print(
                f"[run={run_name}] best_ckpt eval(sample): mean={sample_mean:.1f} "
                f"std={sample_std:.1f} (n={sample_episodes})"
            )

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
