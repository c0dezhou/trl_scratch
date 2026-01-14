from __future__ import annotations

# 环境准备 -> rollout -> 价值收尾(booststrap) -> ppo update
import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym

from core.nn_utils import seed_everything, get_device
from configs.step2_ppo_cartpole import PPOCartPoleConfig
from envs.gym_factory import EnvSpec, make_env, reset_env, step_env, get_obs_act_dims

from models.actor_critic_mlp import ActorCriticMLP
from rl.buffer import RolloutBuffer
from rl.ppo import ppo_update

def main():
    cfg = PPOCartPoleConfig()
    seed_everything(cfg.seed)
    device = get_device("cpu")
    print(f"[step2] device={device}, env_id={cfg.env_id}")

    # 1.init
    # ---env----
    env = make_env(EnvSpec(env_id=cfg.env_id, seed=cfg.seed))
    obs_dim, act_dim = get_obs_act_dims(env)

    # ---model/optim----
    model = ActorCriticMLP(obs_dim, act_dim, hidden=cfg.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)

    #---traning state----
    obs, _ = reset_env(env, seed=cfg.seed)
    ep_ret = 0.0
    ep_rets: list[float] = []
    ep_count = 0
    t0 = time.time()

    # 2.train
    for update in range(1,cfg.total_updates + 1):
        buf = RolloutBuffer(cfg.rollout_steps, obs_dim, device=device)

        term_cnt = 0
        trunc_cnt = 0
        max_ep_len = 0
        cur_ep_len = 0
        #===rollout collect data(onpolicy先采样，再复盘更新)===
        # 模型在环境中连续跑 T 步，把这期间的所见所闻（经验）存入 buffer
        model.eval() # （用于 Rollout采样）
        # 它会关闭 Dropout，让模型的输出在当前参数下更稳定。
        # 它会固定 BatchNorm 的均值和方差，防止模型在采样过程中因为观察到的数据波动而意外改变了自己的内部统计特性。
        for t in range(cfg.rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                action_t, logp_t, value_t = model.act(obs_t) #tensors

            action = int(action_t.item())
            next_obs, reward_env, terminated, truncated, _ = step_env(env, action)
            done = terminated or truncated

            # 训练用 reward：如果是时间截断，就把下一状态的价值补进去
            # reward_env：环境给你的真实即时奖励 rt（CartPole 每步基本是 1）
            # reward_train：我们为了正确处理 truncated，可能会把它改成
            # reward_env + gamma * V(next_obs)（只在 time limit 截断时）
            reward_train = reward_env # 默认训练奖励 = 环境奖励
            # ***截断处理***
            if truncated and (not terminated):
                with torch.no_grad():
                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                _, v_next = model(next_obs_t)
                reward_train = reward_env + cfg.gamma * float(v_next.item())

            buf.add(
                obs=obs_t,
                action=action_t,
                logp=logp_t,
                reward=reward_train,
                done=done,
                value=value_t,
            )

            ep_ret += reward_env
            obs = next_obs

            cur_ep_len += 1
            if terminated:
                term_cnt += 1
            if truncated:
                trunc_cnt += 1
            
            if done: # 一个episode结束
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                ep_count += 1
                max_ep_len = max(max_ep_len, cur_ep_len)
                cur_ep_len = 0
                # 给每个 episode 一个不同 seed（可选，但复现更稳）
                obs, _ = reset_env(env, seed=cfg.seed + ep_count)
            
        # print(f"terminated={term_cnt}, truncated={trunc_cnt}, max_ep_len={max_ep_len}")


        # rollout 结束：每次 rollout 结束后算一次 GAE，用最后一个状态的 V(s_T) 做 bootstrap
        # GAE 的定义是对整段 rollout 的 TD error 做“从后往前”的递推累积，所以需要拿到整段轨迹后再算一次
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            _, last_value = model(obs_t)

        buf.compute_gae(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda)
        
        #=== update(ppo) ===
        model.train() #（用于 Update）
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

        #=== logging ===
        if update % cfg.print_every == 0:
            avg100 = float(np.mean(ep_rets[-100:])) if len(ep_rets) > 0 else 0.0
            print(
                f"update {update:3d}/{cfg.total_updates} | "
                f"terminated={term_cnt}, truncated={trunc_cnt}, max_ep_len={max_ep_len} | "
                f"episodes {len(ep_rets):4d} | "
                f"avg_return(last100) {avg100:7.1f} | "
                f"time {time.time() - t0:.1f}s"
            )

    env.close()
    print("done.")

if __name__ == "__main__":
    main()
