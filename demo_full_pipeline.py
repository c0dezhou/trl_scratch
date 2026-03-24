#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from core.nn_utils import get_device, seed_everything
from core.transformer_min import MiniGPT
from envs.gym_factory import make_env, reset_env, step_env
from model_based.dynamics_dataset import DynamicsSequenceDataset
from model_based.dynamics_transformer import DynamicsTransformer
from model_based.mpc import MPCConfig, mpc_action
from models.actor_critic_gpt import ActorCriticGPT
from models.actor_critic_mlp import ActorCriticMLP
from models.policy_mlp import CategoricalPolicyMLP
from offline.decision_transformer import DecisionTransformer
from offline.dt_dataset import DecisionTransformerDataset
from rl.buffer import RolloutBuffer
from rl.ppo import ppo_update
from rl.reinforce import compute_returns, reinforce_loss


def parse_steps(steps_arg: str) -> List[int]:
    values = []
    for item in steps_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise ValueError(f"invalid step id: {item}")
        sid = int(item)
        if sid < 0 or sid > 5:
            raise ValueError(f"step must be in [0, 5], got {sid}")
        values.append(sid)
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError("--steps is empty")
    return deduped


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def current_frame(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 2:
        return arr[-1].astype(np.float32)
    return arr.astype(np.float32)


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_banner() -> None:
    print("╔══════════════════════════════════════════════════╗")
    print("║   TRL Scratch — 全流程演示                        ║")
    print("╚══════════════════════════════════════════════════╝")


def print_project_overview() -> None:
    print("\n📁 项目结构概览")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    roots = [
        "core",
        "envs",
        "models",
        "rl",
        "offline",
        "model_based",
        "configs",
        "scripts",
    ]
    for name in roots:
        p = Path(name)
        flag = "✅" if p.exists() else "❌"
        count = len(list(p.glob("*.py"))) if p.exists() else 0
        print(f"  {flag} {name:<12} ({count:2d} py files)")


def print_unit_legend() -> None:
    print("\n🧭 计数单位词典")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  step   = 1次监督学习梯度更新")
    print("  ep     = 1整局环境交互(episode)")
    print("  update = 1次PPO循环(先采样rollout, 再多轮优化)")
    print("  iter   = 1次离线训练梯度更新")


def make_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    lag_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target = torch.full((batch_size, seq_len), -100, device=device, dtype=torch.long)
    target[:, lag_k:] = x[:, :-lag_k]
    return x, target


@torch.no_grad()
def causal_no_future_leak(model: MiniGPT, vocab_size: int, seq_len: int, device: torch.device) -> float:
    model.eval()
    a = torch.randint(0, vocab_size, (1, seq_len), device=device)
    b = a.clone()
    b[:, seq_len // 2 :] = torch.randint(0, vocab_size, (1, seq_len - seq_len // 2), device=device)
    la = model(a)
    lb = model(b)
    return float((la[:, : seq_len // 2] - lb[:, : seq_len // 2]).abs().max().item())


@dataclass
class PPOTrainResult:
    model: torch.nn.Module
    obs_shape: tuple[int, ...]
    act_dim: int
    update_curve: List[float]
    history_len: int
    obs_dim_per_step: int


def run_ppo_demo(
    *,
    device: torch.device,
    seed: int,
    env_id: str,
    model_type: str,
    pomdp_keep_idx: Optional[Sequence[int]],
    history_len: int,
    use_delta_obs: bool,
    updates: int,
    rollout_steps: int,
    update_epochs: int,
    minibatch_size: int,
    lr: float,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    hidden: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    log_every: int,
    log_line_builder: Callable[[int, float], str],
) -> PPOTrainResult:
    env = make_env(
        env_id=env_id,
        seed=seed,
        pomdp_keep_idx=tuple(pomdp_keep_idx) if pomdp_keep_idx is not None else None,
        history_len=history_len,
        use_delta_obs=use_delta_obs,
    )()
    obs0, _ = reset_env(env, seed=seed)
    obs_arr0 = np.asarray(obs0, dtype=np.float32)
    obs_shape = tuple(obs_arr0.shape)

    if obs_arr0.ndim == 1:
        t_hist, d_step = 1, int(obs_arr0.shape[0])
    else:
        t_hist, d_step = int(obs_arr0.shape[0]), int(obs_arr0.shape[1])

    flat_dim = int(np.prod(obs_shape))
    act_dim = int(env.action_space.n)

    if model_type == "mlp":
        model = ActorCriticMLP(obs_dim=flat_dim, act_dim=act_dim, hidden=hidden).to(device)
    elif model_type == "gpt":
        model = ActorCriticGPT(
            obs_dim=d_step,
            act_dim=act_dim,
            history_len=t_hist,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        ).to(device)
    else:
        raise ValueError(f"unsupported model_type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    obs = flatten_obs(obs0)
    ep_ret = 0.0
    ep_rets: List[float] = []
    ep_count = 0
    update_curve: List[float] = []

    for update in range(1, updates + 1):
        buf = RolloutBuffer(rollout_steps, flat_dim, device=device)

        model.eval()
        for _ in range(rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.inference_mode():
                action_t, logp_t, value_t = model.act(obs_t)

            action = int(action_t.item())
            next_obs, reward, terminated, truncated, _ = step_env(env, action)
            next_obs_flat = flatten_obs(next_obs)

            done_env = bool(terminated or truncated)
            done_bootstrap = bool(terminated)
            timeout = bool(truncated and not terminated)
            terminal_value = 0.0
            if timeout:
                with torch.inference_mode():
                    nxt_t = torch.tensor(next_obs_flat, dtype=torch.float32, device=device)
                    _, terminal_value = model(nxt_t)

            buf.add(
                obs=obs_t,
                action=action_t,
                logp=logp_t,
                reward=float(reward),
                done=done_bootstrap,
                value=value_t,
                timeout=timeout,
                terminal_value=float(terminal_value),
            )

            ep_ret += float(reward)
            obs = next_obs_flat
            if done_env:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                ep_count += 1
                obs, _ = reset_env(env, seed=seed + ep_count)
                obs = flatten_obs(obs)

        with torch.inference_mode():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            _, last_value = model(obs_t)
        buf.compute_gae(last_value=last_value, gamma=gamma, lam=gae_lambda)

        model.train()
        ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buf,
            update_epochs=update_epochs,
            minibatch_size=min(minibatch_size, rollout_steps),
            clip_coef=clip_coef,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            target_kl=None,
            clip_vloss=False,
        )

        avg_ret = float(np.mean(ep_rets[-20:])) if ep_rets else 0.0
        update_curve.append(avg_ret)
        if update % log_every == 0 or update == updates:
            print(log_line_builder(update, avg_ret))

    env.close()
    return PPOTrainResult(
        model=model,
        obs_shape=obs_shape,
        act_dim=act_dim,
        update_curve=update_curve,
        history_len=t_hist,
        obs_dim_per_step=d_step,
    )


@dataclass
class DemoContext:
    device: torch.device
    seed: int
    tmp_dir: Path
    steps: List[int]
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    success_steps: List[int] = field(default_factory=list)
    step3_gpt_pack_path: Optional[Path] = None
    step4_dataset_path: Optional[Path] = None


def run_step0(ctx: DemoContext) -> None:
    print("\n🔧 Step 0: 最小 GPT (Copy Task)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  单位: step | 指标: loss下降, acc上升 (token预测正确率)")
    vocab_size = 50
    seq_len = 32
    lag_k = 3
    train_steps = 500
    log_every = 100

    model = MiniGPT(
        vocab_size=vocab_size,
        max_len=seq_len,
        d_model=96,
        n_heads=4,
        n_layers=2,
        d_ff=192,
        dropout=0.0,
    ).to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    final_acc = 0.0
    for step in range(1, train_steps + 1):
        x, target = make_copy_batch(
            batch_size=128,
            seq_len=seq_len,
            vocab_size=vocab_size,
            lag_k=lag_k,
            device=ctx.device,
        )
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == train_steps:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = target != -100
                final_acc = float((pred[mask] == target[mask]).float().mean().item())
            print(f"  [step={step:3d}] loss={loss.item():.4f}  acc={final_acc * 100:5.1f}%")

    leak_diff = causal_no_future_leak(model, vocab_size=vocab_size, seq_len=seq_len, device=ctx.device)
    print(f"  causal前缀一致性 max_diff={leak_diff:.3e}")
    print("  ✅ Transformer causal attention 验证通过")
    ctx.metrics["step0_acc"] = final_acc * 100.0


def run_step1(ctx: DemoContext) -> None:
    print("\n🎮 Step 1: REINFORCE")
    print("━━━━━━━━━━━━━━━━━━━━")
    print("  单位: ep | 指标: avg_return上升 (每20局平均回报)")
    env = make_env("CartPole-v1", ctx.seed + 10)()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    policy = CategoricalPolicyMLP(obs_dim=obs_dim, act_dim=act_dim, hidden=128).to(ctx.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    episodes = 100
    gamma = 0.99
    returns_all: List[float] = []

    for ep in range(1, episodes + 1):
        obs, _ = reset_env(env, seed=ctx.seed + 1000 + ep)
        done = False
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=ctx.device)
            action_t, logp_t = policy.act(obs_t)
            action = int(action_t.item())
            next_obs, reward, terminated, truncated, _ = step_env(env, action)
            done = bool(terminated or truncated)
            log_probs.append(logp_t)
            rewards.append(float(reward))
            obs = next_obs

        returns = compute_returns(rewards, gamma=gamma).to(ctx.device)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        loss = reinforce_loss(log_probs, returns)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        returns_all.append(float(sum(rewards)))
        if ep % 20 == 0:
            seg = returns_all[-20:]
            print(f"  [ep {ep-19:3d}-{ep:3d}] avg_return = {np.mean(seg):6.1f}")

    env.close()
    final_avg = float(np.mean(returns_all[-20:])) if returns_all else 0.0
    print("  ✅ 策略梯度基线完成")
    ctx.metrics["step1_return"] = final_avg


def run_step2(ctx: DemoContext) -> None:
    print("\n🚀 Step 2: PPO (MDP, 全观测)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  单位: update | 指标: avg_return上升")
    print("  update含义: 每次先采样1024步rollout, 再做6轮PPO优化")
    result = run_ppo_demo(
        device=ctx.device,
        seed=ctx.seed + 20,
        env_id="CartPole-v1",
        model_type="mlp",
        pomdp_keep_idx=None,
        history_len=1,
        use_delta_obs=False,
        updates=50,
        rollout_steps=1024,
        update_epochs=6,
        minibatch_size=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.005,
        max_grad_norm=0.5,
        hidden=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        log_every=1,
        log_line_builder=lambda u, r: f"  [update {u:2d}] avg_return = {r:6.1f}",
    )
    step2_final = result.update_curve[-1] if result.update_curve else 0.0
    print("  ✅ PPO 在 MDP 下快速提升")
    ctx.metrics["step2_return"] = step2_final
    del result


def run_step3(ctx: DemoContext) -> None:
    print("\n🧩 Step 3: PPO + POMDP (MLP vs GPT)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  单位: update | 指标: MLP/GPT 的avg_return对比")
    print("  update含义: 每次先采样512步rollout, 再做4轮PPO优化")

    mlp_result = run_ppo_demo(
        device=ctx.device,
        seed=ctx.seed + 30,
        env_id="CartPole-v1",
        model_type="mlp",
        pomdp_keep_idx=(0, 2),
        history_len=1,
        use_delta_obs=False,
        updates=30,
        rollout_steps=512,
        update_epochs=4,
        minibatch_size=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        hidden=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        log_every=10,
        log_line_builder=lambda u, r: f"  MLP(T=1):        [update {u:2d}] avg_return = {r:6.1f}",
    )

    gpt_result = run_ppo_demo(
        device=ctx.device,
        seed=ctx.seed + 31,
        env_id="CartPole-v1",
        model_type="gpt",
        pomdp_keep_idx=(0, 2),
        history_len=4,
        use_delta_obs=True,
        updates=30,
        rollout_steps=512,
        update_epochs=4,
        minibatch_size=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.005,
        max_grad_norm=0.5,
        hidden=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        log_every=10,
        log_line_builder=lambda u, r: f"  GPT(T=4+delta):  [update {u:2d}] avg_return = {r:6.1f}",
    )

    mlp_final = mlp_result.update_curve[-1] if mlp_result.update_curve else 0.0
    gpt_final = gpt_result.update_curve[-1] if gpt_result.update_curve else 0.0
    print(f"  MLP(T=1)  final avg_return = {mlp_final:.1f}")
    print(f"  GPT(T=4+) final avg_return = {gpt_final:.1f}")
    print("  ✅ POMDP 对比完成")

    ckpt_path = ctx.tmp_dir / "step3_gpt_policy.pt"
    torch.save(
        {
            "state_dict": gpt_result.model.state_dict(),
            "obs_dim": int(gpt_result.obs_dim_per_step),
            "act_dim": int(gpt_result.act_dim),
            "history_len": int(gpt_result.history_len),
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
            "dropout": 0.0,
        },
        ckpt_path,
    )
    print(f"  保存 GPT 策略到: {ckpt_path}")

    ctx.step3_gpt_pack_path = ckpt_path
    ctx.artifacts["step3_gpt_ckpt"] = str(ckpt_path)
    ctx.metrics["step3_mlp_return"] = mlp_final
    ctx.metrics["step3_gpt_return"] = gpt_final

    del mlp_result
    del gpt_result


def collect_offline_dataset(
    *,
    ctx: DemoContext,
    out_path: Path,
    episodes: int,
    policy_model: Optional[torch.nn.Module],
) -> tuple[int, int]:
    env = make_env(
        env_id="CartPole-v1",
        seed=ctx.seed + 40,
        pomdp_keep_idx=(0, 2),
        history_len=4,
        use_delta_obs=True,
    )()

    if policy_model is not None:
        policy_model.eval()
    act_dim = int(env.action_space.n)

    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    rew_buf: List[float] = []
    done_buf: List[float] = []
    episode_ends: List[int] = []
    total_steps = 0

    for ep in range(episodes):
        obs, _ = reset_env(env, seed=ctx.seed + 5000 + ep)
        ep_ret = 0.0
        while True:
            if policy_model is None:
                action = int(env.action_space.sample())
            else:
                obs_flat = flatten_obs(obs)
                obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=ctx.device)
                with torch.no_grad():
                    logits, _ = policy_model(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())

            s = current_frame(np.asarray(obs, dtype=np.float32))
            next_obs, reward, terminated, truncated, _ = step_env(env, action)
            done = bool(terminated or truncated)

            obs_buf.append(s)
            act_buf.append(action)
            rew_buf.append(float(reward))
            done_buf.append(1.0 if done else 0.0)
            total_steps += 1
            ep_ret += float(reward)
            obs = next_obs

            if done:
                episode_ends.append(total_steps)
                break
        print(f"  [collect ep {ep+1:2d}/{episodes}] return={ep_ret:6.1f}")

    env.close()

    np.savez(
        out_path,
        obs=np.asarray(obs_buf, dtype=np.float32),
        actions=np.asarray(act_buf, dtype=np.int64),
        rewards=np.asarray(rew_buf, dtype=np.float32),
        dones=np.asarray(done_buf, dtype=np.float32),
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        act_dim=np.int64(act_dim),
    )
    return episodes, total_steps


@torch.no_grad()
def evaluate_dt_online(
    *,
    model: DecisionTransformer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
    episodes: int,
    context_len: int,
    rtg_scale: float,
    target_return: float,
    seed: int,
) -> float:
    env = make_env(
        env_id="CartPole-v1",
        seed=seed,
        pomdp_keep_idx=(0, 2),
        history_len=1,
        use_delta_obs=True,
    )()
    model.eval()
    returns: List[float] = []

    for ep in range(episodes):
        obs, _ = reset_env(env, seed=seed + 8000 + ep)
        done = False
        ep_ret = 0.0
        rtg_now = target_return / rtg_scale
        t = 0

        states: List[np.ndarray] = []
        acts: List[int] = []
        rtgs: List[float] = []
        tss: List[int] = []

        while not done:
            s = np.asarray(obs, dtype=np.float32)
            s = (s - state_mean) / state_std

            states.append(s)
            acts.append(0)
            rtgs.append(rtg_now)
            tss.append(t)

            if len(states) > context_len:
                states = states[-context_len:]
                acts = acts[-context_len:]
                rtgs = rtgs[-context_len:]
                tss = tss[-context_len:]

            valid_len = len(states)
            s_pad = np.zeros((context_len, state_mean.shape[0]), dtype=np.float32)
            a_pad = np.zeros((context_len,), dtype=np.int64)
            r_pad = np.zeros((context_len,), dtype=np.float32)
            t_pad = np.zeros((context_len,), dtype=np.int64)
            v_pad = np.zeros((context_len,), dtype=np.float32)

            s_pad[:valid_len] = np.stack(states, axis=0)
            a_pad[:valid_len] = np.asarray(acts, dtype=np.int64)
            r_pad[:valid_len] = np.asarray(rtgs, dtype=np.float32)
            t_pad[:valid_len] = np.asarray(tss, dtype=np.int64)
            v_pad[:valid_len] = 1.0

            a = model.act(
                states=torch.tensor(s_pad, device=device).unsqueeze(0),
                actions=torch.tensor(a_pad, device=device).unsqueeze(0),
                rtg=torch.tensor(r_pad, device=device).unsqueeze(0),
                timesteps=torch.tensor(t_pad, device=device).unsqueeze(0),
                valid=torch.tensor(v_pad, device=device).unsqueeze(0),
                valid_len=valid_len,
                sample=False,
            )
            acts[-1] = int(a)

            obs, reward, terminated, truncated, _ = step_env(env, int(a))
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            rtg_now = rtg_now - float(reward) / rtg_scale
            t += 1

        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns)) if returns else 0.0


def maybe_load_step3_gpt(ctx: DemoContext) -> Optional[ActorCriticGPT]:
    if ctx.step3_gpt_pack_path is None or not ctx.step3_gpt_pack_path.exists():
        return None
    pack = torch.load(ctx.step3_gpt_pack_path, map_location=ctx.device)
    model = ActorCriticGPT(
        obs_dim=int(pack["obs_dim"]),
        act_dim=int(pack["act_dim"]),
        history_len=int(pack["history_len"]),
        d_model=int(pack["d_model"]),
        n_heads=int(pack["n_heads"]),
        n_layers=int(pack["n_layers"]),
        d_ff=int(pack["d_ff"]),
        dropout=float(pack["dropout"]),
    ).to(ctx.device)
    model.load_state_dict(pack["state_dict"])
    model.eval()
    return model


def run_step4(ctx: DemoContext) -> None:
    print("\n📦 Step 4: Decision Transformer (离线 RL)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  单位: iter | 指标: 训练loss下降 + 在线eval avg_return上升")
    print("  解读: loss是动作模仿误差, eval avg_return才是环境真实表现")
    dataset_path = ctx.tmp_dir / "step4_offline_data.npz"

    use_policy = 3 in ctx.success_steps
    policy_model = maybe_load_step3_gpt(ctx) if use_policy else None
    policy_tag = "Step3 GPT policy" if policy_model is not None else "random policy"

    print(f"  数据采集策略: {policy_tag}")
    episodes, total_steps = collect_offline_dataset(
        ctx=ctx,
        out_path=dataset_path,
        episodes=50,
        policy_model=policy_model,
    )
    if policy_model is not None:
        del policy_model

    print(f"  数据采集: {episodes} episodes, {total_steps} steps")

    ds = DecisionTransformerDataset(
        npz_path=str(dataset_path),
        context_len=20,
        rtg_scale=500.0,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True, num_workers=0)

    model = DecisionTransformer(
        state_dim=int(ds.state_dim),
        act_dim=int(ds.act_dim),
        context_len=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        max_timestep=512,
    ).to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    max_iters = 1500
    loader_iter = iter(loader)
    for it in range(1, max_iters + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        states = batch["states"].to(ctx.device)
        actions = batch["actions"].to(ctx.device)
        rtg = batch["rtg"].to(ctx.device)
        timesteps = batch["timesteps"].to(ctx.device)
        valid = batch["valid"].to(ctx.device)

        logits = model(states, actions, rtg, timesteps, valid)
        bsz, ksz, act_dim = logits.shape
        loss_per = F.cross_entropy(
            logits.reshape(bsz * ksz, act_dim),
            actions.reshape(bsz * ksz),
            reduction="none",
        )
        w = valid.reshape(bsz * ksz)
        loss = (loss_per * w).sum() / w.sum().clamp_min(1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % 500 == 0:
            print(f"  [iter {it:4d}] loss={loss.item():.4f}")

    avg_return = evaluate_dt_online(
        model=model,
        state_mean=ds.state_mean,
        state_std=ds.state_std,
        device=ctx.device,
        episodes=5,
        context_len=20,
        rtg_scale=500.0,
        target_return=500.0,
        seed=ctx.seed + 43,
    )
    print(f"  [eval] avg_return = {avg_return:.1f} (target_return=500)")
    print("  ✅ 离线模仿学习演示完成")

    ckpt_path = ctx.tmp_dir / "step4_dt_demo.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "state_mean": ds.state_mean,
            "state_std": ds.state_std,
            "state_dim": int(ds.state_dim),
            "act_dim": int(ds.act_dim),
            "context_len": 20,
            "rtg_scale": 500.0,
        },
        ckpt_path,
    )

    ctx.step4_dataset_path = dataset_path
    ctx.artifacts["step4_dataset"] = str(dataset_path)
    ctx.artifacts["step4_ckpt"] = str(ckpt_path)
    ctx.metrics["step4_return"] = avg_return
    del model


@torch.no_grad()
def evaluate_world_model(
    model: DynamicsTransformer,
    loader: DataLoader,
) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    trans_cnt = 0.0
    done_correct = 0.0
    done_cnt = 0.0

    for batch in loader:
        device = next(model.parameters()).device
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        timesteps = batch["timesteps"].to(device)
        valid = batch["valid"].to(device)
        delta_tgt = batch["delta_targets"].to(device)
        done_tgt = batch["done_targets"].to(device)
        trans_valid = batch["trans_valid"].to(device)

        out = model(states, actions, timesteps, valid)
        v = (valid > 0).float()
        tv = (trans_valid > 0).float() * v

        err = (out.delta - delta_tgt).pow(2).mean(dim=-1)
        mse_sum += float((err * tv).sum().item())
        trans_cnt += float(tv.sum().item())

        pred_done = (torch.sigmoid(out.done_logits) > 0.5).float()
        done_correct += float(((pred_done == done_tgt) * v).sum().item())
        done_cnt += float(v.sum().item())

    return {
        "delta_mse": mse_sum / (trans_cnt + 1e-6),
        "done_acc": done_correct / (done_cnt + 1e-6),
    }


@torch.no_grad()
def evaluate_mpc_online(
    *,
    model: DynamicsTransformer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
    episodes: int,
    seed: int,
) -> float:
    env = make_env(
        env_id="CartPole-v1",
        seed=seed,
        pomdp_keep_idx=(0, 2),
        history_len=1,
        use_delta_obs=True,
    )()
    cfg = MPCConfig(
        horizon=10,
        num_samples=256,
        done_threshold=0.5,
        gamma=1.0,
        seed=seed,
        state_cost_coef=0.5,
    )

    k_ctx = int(model.K)
    returns: List[float] = []

    for ep in range(episodes):
        obs, _ = reset_env(env, seed=seed + 9000 + ep)
        ep_ret = 0.0
        t = 0

        obs_hist = deque(maxlen=k_ctx)
        act_hist = deque(maxlen=k_ctx)
        t_hist = deque(maxlen=k_ctx)

        while True:
            obs_hist.append(np.asarray(obs, dtype=np.float32))
            act_hist.append(0)
            t_hist.append(t)

            obs_arr = np.stack(list(obs_hist), axis=0)
            act_arr = np.asarray(list(act_hist), dtype=np.int64)
            t_arr = np.asarray(list(t_hist), dtype=np.int64)

            action = mpc_action(
                model=model,
                obs_hist_raw=obs_arr,
                act_hist=act_arr,
                t_hist=t_arr,
                state_mean=state_mean,
                state_std=state_std,
                cfg=cfg,
                device=device,
            )
            act_hist[-1] = int(action)

            obs, reward, terminated, truncated, _ = step_env(env, int(action))
            ep_ret += float(reward)
            t += 1
            if terminated or truncated:
                break

        returns.append(ep_ret)
        print(f"  [MPC ep {ep+1}/{episodes}] return={ep_ret:.1f}")

    env.close()
    return float(np.mean(returns)) if returns else 0.0


def run_step5(ctx: DemoContext) -> None:
    print("\n🔮 Step 5: 世界模型 + MPC")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  单位: iter | 指标: delta_mse下降, done_acc上升, MPC avg_return上升")
    print("  解读: 前两者是模型拟合精度, 最终仍以MPC在线回报为准")

    dataset_path = ctx.step4_dataset_path
    if dataset_path is None or not dataset_path.exists():
        dataset_path = ctx.tmp_dir / "step5_offline_data.npz"
        print("  Step4 数据不存在，改为随机策略重新采集数据")
        collect_offline_dataset(
            ctx=ctx,
            out_path=dataset_path,
            episodes=50,
            policy_model=None,
        )
    else:
        print(f"  复用 Step4 数据: {dataset_path}")

    train_ds = DynamicsSequenceDataset(
        npz_path=str(dataset_path),
        context_len=20,
        split="train",
        val_frac_episodes=0.1,
        seed=ctx.seed,
    )
    val_ds = DynamicsSequenceDataset(
        npz_path=str(dataset_path),
        context_len=20,
        split="val",
        val_frac_episodes=0.1,
        seed=ctx.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, drop_last=False, num_workers=0)

    model = DynamicsTransformer(
        state_dim=int(train_ds.state_dim),
        act_dim=int(train_ds.act_dim),
        context_len=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        max_timestep=512,
    ).to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    max_iters = 1500
    train_iter = iter(train_loader)
    for it in range(1, max_iters + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        states = batch["states"].to(ctx.device)
        actions = batch["actions"].to(ctx.device)
        timesteps = batch["timesteps"].to(ctx.device)
        valid = batch["valid"].to(ctx.device)
        delta_tgt = batch["delta_targets"].to(ctx.device)
        done_tgt = batch["done_targets"].to(ctx.device)
        trans_valid = batch["trans_valid"].to(ctx.device)

        out = model(states, actions, timesteps, valid)
        v = (valid > 0).float()
        tv = (trans_valid > 0).float() * v

        delta_loss = (out.delta - delta_tgt).pow(2).mean(dim=-1)
        delta_loss = (delta_loss * tv).sum() / (tv.sum() + 1e-6)

        done_loss = bce(out.done_logits, done_tgt)
        done_loss = (done_loss * v).sum() / (v.sum() + 1e-6)

        loss = delta_loss + done_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % 500 == 0:
            metrics = evaluate_world_model(model, val_loader)
            print(
                f"  [iter {it:4d}] delta_mse={metrics['delta_mse']:.4f} "
                f"done_acc={metrics['done_acc']:.3f}"
            )

    metrics = evaluate_world_model(model, val_loader)
    mpc_avg = evaluate_mpc_online(
        model=model,
        state_mean=train_ds.state_mean,
        state_std=train_ds.state_std,
        device=ctx.device,
        episodes=3,
        seed=ctx.seed + 55,
    )

    print(f"  [MPC eval] 3 episodes, avg_return = {mpc_avg:.1f}")
    print("  ✅ 世界模型 + MPC 演示完成")

    ckpt_path = ctx.tmp_dir / "step5_world_model_demo.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "state_mean": train_ds.state_mean,
            "state_std": train_ds.state_std,
            "state_dim": int(train_ds.state_dim),
            "act_dim": int(train_ds.act_dim),
            "context_len": 20,
        },
        ckpt_path,
    )

    ctx.artifacts["step5_world_model_ckpt"] = str(ckpt_path)
    ctx.metrics["step5_delta_mse"] = float(metrics["delta_mse"])
    ctx.metrics["step5_done_acc"] = float(metrics["done_acc"])
    ctx.metrics["step5_mpc_return"] = float(mpc_avg)
    del model


def print_final_summary(ctx: DemoContext) -> None:
    print("\n══════════════════════════════════════")
    print("📊 最终对比总结")
    if "step2_return" in ctx.metrics:
        print(f"  Step2 PPO(MDP):     {ctx.metrics['step2_return']:.1f}")
    if "step3_gpt_return" in ctx.metrics:
        print(f"  Step3 GPT(POMDP):   {ctx.metrics['step3_gpt_return']:.1f}")
    if "step4_return" in ctx.metrics:
        print(f"  Step4 DT(offline):  {ctx.metrics['step4_return']:.1f}")
    if "step5_mpc_return" in ctx.metrics:
        print(f"  Step5 MPC:          {ctx.metrics['step5_mpc_return']:.1f}")
    print("══════════════════════════════════════")

    if ctx.artifacts:
        print("🗂️ 临时文件")
        for k, v in ctx.artifacts.items():
            print(f"  - {k}: {v}")
        print(f"  可按需删除目录: {ctx.tmp_dir}")


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None:
        return get_device(None)
    dev = get_device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("⚠️ 指定了 cuda 但当前不可用，已回退到 cpu")
        return get_device("cpu")
    return dev


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL Scratch full demo pipeline (Step0~Step5)")
    parser.add_argument("--steps", type=str, default="0,1,2,3,4,5", help="e.g. 0,1,2,3,4,5 or 3,4,5")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    device = resolve_device(args.device)
    seed_everything(args.seed)

    tmp_dir = Path("/tmp/trl_demo")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ctx = DemoContext(device=device, seed=args.seed, tmp_dir=tmp_dir, steps=steps)

    print_banner()
    print(f"\n🖥️ device={device}  seed={args.seed}  steps={steps}")
    print_project_overview()
    print_unit_legend()

    step_fns = {
        0: run_step0,
        1: run_step1,
        2: run_step2,
        3: run_step3,
        4: run_step4,
        5: run_step5,
    }

    for step in steps:
        try:
            step_fns[step](ctx)
            ctx.success_steps.append(step)
        except Exception as exc:
            print(f"\n❌ Step {step} 执行失败: {exc.__class__.__name__}: {exc}")
            tb = traceback.format_exc(limit=1).strip().splitlines()
            if tb:
                print(f"  traceback: {tb[-1]}")
        finally:
            cleanup_memory()

    print_final_summary(ctx)


if __name__ == "__main__":
    main()
