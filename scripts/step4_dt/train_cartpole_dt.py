# 读取之前收集的专家录像（.npz 数据集），
# 训练一个 Transformer 模型来模仿这些行为，
# 并最终在真实环境中测试这个“学徒”的水平
from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import asdict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.nn_utils import get_device, seed_everything
from envs.gym_factory import make_env, reset_env, step_env
from offline.dt_dataset import DecisionTransformerDataset
from offline.decision_transformer import DecisionTransformer

def load_cfg(cfg_module: str):
    mod = importlib.import_module(cfg_module)
    if not hasattr(mod, "Step4DecisionTransformerConfig"):
        raise AttributeError(f"{cfg_module} has no Step4DecisionTransformerConfig")
    return mod.Step4DecisionTransformerConfig()

@torch.no_grad()
def evaluate_dt(cfg, model: DecisionTransformer, ds: DecisionTransformerDataset):
    """这个函数是把模型拉到真实的实时游戏环境里，看看它在没有标准答案参考的情况下，能不能根据你给的目标（RTG）自己玩通关。
    在强化学习中，光看训练时的 Loss（损失函数）下降是不够的。Loss 只能说明模型“模仿专家动作”模仿得像不像，但不能代表它玩得好不好。
    所以在训练时我们要暂停几局，让模型在真正的环境中跑几局算出平均分
    注意：必须用训练集里的那套 ds.state_mean/std 对实时观测进行同样的缩放。如果不缩放直接喂给模型，
    模型看到的数值会极其巨大，导致预测出的动作完全乱套"""

    env = make_env(
        env_id=cfg.env_id,
        seed=cfg.seed,
        pomdp_keep_idx=tuple(cfg.pomdp_keep_idx),
        history_len=int(cfg.history_len),
        use_delta_obs=bool(cfg.use_delta_obs),
    )()

    device = next(model.parameters()).device
    K = int(cfg.context_len)
    D = int(ds.state_dim)

    returns: List[float] = []

    for epi in range(int(cfg.eval_episodes)):
        obs, _ = reset_env(env, seed=cfg.seed + 10000 + epi)
        done = False
        ep_ret = 0.0

        states: List[np.ndarray] = []
        acts: List[int] = []
        rtgs: List[float] = []
        tss: List[int] = []

        # 缩放
        rtg_now = float(cfg.target_return) / float(cfg.rtg_scale)
        t = 0

        # DecisionTransformer推理（测试）阶段最核心的数据准备与窗口滑动逻辑
        while not done:
            s = np.asarray(obs, dtype=np.float32)
            # 归一化:因为模型在训练时看的是“归一化”后的数据，
            # 所以测试时也必须把环境的反馈转换成同样的“比例尺”，否则模型会“看不懂”当前的状态
            s = (s - ds.state_mean) / ds.state_std

            states.append(s)
            acts.append(0)  # placeholder, action 采样后会填充
            # Transformer 预测动作 a_t 需要输入序列。但在这一瞬间，我们还没预测出 a_t 呢
            rtgs.append(rtg_now)
            tss.append(t)

            # Context Window，维持有限的记忆
            if len(states) > K:
                states = states[-K:]
                acts = acts[-K:]
                rtgs = rtgs[-K:]
                tss = tss[-K:]

            valid_len = len(states)

            # 输入序列长度每秒钟都在变，实现“动态数据，静态输入”，让DecisionTransformer稳定地流式处理数据
            # 1.创建全0矩阵 
            states_pad = np.zeros((K, D), dtype=np.float32)
            actions_pad = np.zeros((K,), dtype=np.int64)
            rtg_pad = np.zeros((K,), dtype=np.float32)
            ts_pad = np.zeros((K,), dtype=np.int64)
            valid_pad = np.zeros((K,), dtype=np.float32)

            # 2.填充数据 padding
            states_pad[:valid_len] = np.stack(states, axis=0)
            actions_pad[:valid_len] = np.asarray(acts, dtype=np.int64)
            rtg_pad[:valid_len] = np.asarray(rtgs, dtype=np.float32)
            ts_pad[:valid_len] = np.asarray(tss, dtype=np.int64)
            valid_pad[:valid_len] = 1.0

            # 3.张量化，增加batch维
            # torch.tensor：把数据从 CPU 的 NumPy 搬到 GPU 的 Tensor 上
            # PyTorch 模型永远预期第一个维度是 Batch Size，实时评估时，我们一次只测一个环境，所以 Batch Size 是 1
            states_t = torch.tensor(states_pad, device=device).unsqueeze(0)
            actions_t = torch.tensor(actions_pad, device=device).unsqueeze(0)
            rtg_t = torch.tensor(rtg_pad, device=device).unsqueeze(0)
            ts_t = torch.tensor(ts_pad, device=device).unsqueeze(0)
            valid_t = torch.tensor(valid_pad, device=device).unsqueeze(0)

            # 决策的瞬间：调用 DecisionTransformer 模型进行推理的核心接口
            """“自回归（Autoregressive）” 的推理过程： 
            模型产生的动作（Output）会影响环境，环境的反馈（Reward/Next Obs）又会变成模型下一秒的输入（Input）。"""
            # 接受context win的数据（r,s,a,t,valid）
            # 根据这K步（K=len_ctx_win），结合目标：rtg,计算出当前最合理的动作a
            a = model.act(
                states=states_t,
                actions=actions_t,
                rtg=rtg_t,
                timesteps=ts_t,
                valid=valid_t,
                valid_len=valid_len,
                sample=bool(cfg.sample_actions),
            )

            # 填补刚刚站位的placeholder
            # 保证了下一秒（下一轮循环）模型在看“历史动作”时，能看到这一步它真实做出的选择，而不是那个虚假的占位符
            acts[-1] = a 

            obs, r, terminated, truncated, _info = step_env(env, a)
            done = bool(terminated or truncated)
            ep_ret += float(r)

            # 目标的动态调整
            rtg_now = rtg_now - float(r) / float(cfg.rtg_scale)
            t += 1

        returns.append(ep_ret)
    
    # 返回几局评估的均值
    return float(np.mean(returns))

# 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="configs.step4_dt_cartpole_pomdp")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed_everything(int(cfg.seed))

    device = get_device(getattr(cfg, "device", None))

    # dataset
    ds = DecisionTransformerDataset(
        npz_path=str(cfg.dataset_path),
        context_len=int(cfg.context_len),
        rtg_scale=float(cfg.rtg_scale),
    )

    # 虽然 ds 包含了所有数据，
    # 但模型不能一次性吞下几万步的数据。
    # DataLoader 的作用就是把庞大的数据集切割成一小块一小块的（Batch），并有节奏地喂给模型。
    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True, # 每一轮训练开始前，把所有数据的顺序随机洗牌 抽取是iid的
        num_workers=0,
        drop_last=True, # 如果总数据量不能被 batch_size 整除，丢弃最后一个batch
    )

    # model
    model = DecisionTransformer(
        state_dim=int(ds.state_dim),
        act_dim=int(ds.act_dim),
        context_len=int(cfg.context_len),
        d_model=int(cfg.d_model),
        n_heads=int(cfg.n_heads),
        n_layers=int(cfg.n_layers),
        dropout=float(cfg.dropout),
        max_timestep=int(cfg.max_timestep),
    ).to(device)

    # optim:adamw:自适应学习率
    # weight_decay:这是一种“惩罚机制”。如果某个神经元的权重值变得特别大，优化器就会给它一记重锤
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    os.makedirs("checkpoints", exist_ok=True)
    # 一个“极小的负无穷大”,确保第一次评估的结果（哪怕分数值很低）一定能超过这个初始值
    best_eval = -1e9

    it = 0
    # 采用了 “基于迭代次数 (Iteration-based)” 而非 “基于轮次 (Epoch-based)” 的训练方式
    # 你可以精确控制模型跑多少步（比如 100,000 步），而不是跑多少遍数据集。当数据集很大时，这种方式更灵活，可以随时进行评估和保存
    loader_iter = iter(loader)

    while it < int(cfg.max_iters):
        it += 1
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
    
        states = batch["states"].to(device)        # [B,K,D] already normalized
        actions = batch["actions"].to(device)      # [B,K]
        rtg = batch["rtg"].to(device)              # [B,K]
        timesteps = batch["timesteps"].to(device)  # [B,K]
        valid = batch["valid"].to(device)          # [B,K]

        logits = model(states, actions, rtg, timesteps, valid) # [B,K,A]

        B,K,A = logits.shape
        # PyTorch 的 F.cross_entropy 函数通常期望输入是二维的 [N, Classes]
        # 对B*K个独立任务分别预测对应动作
        loss_per = F.cross_entropy(
            logits.reshape(B*K, A),
            actions.reshape(B*K),
            reduction="none", # 不做平均，返回列表，因为有些是padding
        )
        w = valid.reshape(B * K)
        # 乘法过滤：loss_per * w， 如果该位置是补零数据（0），Loss 变成 0
        # 归一化：w.sum().clamp_min(1.0) 除以 w.sum()（即这个 Batch 里真实 Token 的总数），得到每个有效动作的平均损失
        # clamp_min(1.0)万一某个 Batch 里全是 Padding（虽然理论上不会），分母变成 0 会导致程序崩溃。强制分母最小为 1，保证了数学上的稳定性
        loss = (loss_per * w).sum() / w.sum().clamp_min(1.0)

        opt.zero_grad(set_to_none=True)
        # 计算误差 修改.grad()
        loss.backward()
        # 梯度裁剪（Gradient Clipping）是为了解决梯度爆炸,检查.grad(),修改.grad()
        # 计算所有参数梯度的总模长（L2 Norm）。如果总模长超过了 max_norm（即 cfg.grad_clip），就将所有梯度等比例缩小
        # 它只改变梯度的大小，不改变梯度的方向
        torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.grad_clip))
        # 执行更新,查看.grad(),修改.data()
        opt.step()

        if it % int(cfg.log_every) == 0:
            print(f"[train] it={it:6d}/{int(cfg.max_iters)} loss={loss.item():.4f}")

        if it % int(cfg.eval_every) == 0:
            # 模式切换
            # .eval() 只是把模型的状态调成考试模式（实战）(开关，它并不负责管理内存或梯度
            model.eval() # 冻结dropout 冻结batchnorm（停止更新均值和方差
            eval_ret = evaluate_dt(cfg, model, ds)
            # 调整到学习（训练）模式
            model.train()

            print(f"[eval] it={it:6d} avg_return={eval_ret:.1f}")

            if eval_ret > best_eval:
                best_eval = eval_ret
                ckpt_path = f"checkpoints/{cfg.run_name}_best.pt"
                torch.save( # 不止存权重文件
                    {
                        "model": model.state_dict(),
                        "cfg":asdict(cfg),
                        # 模型是在经过归一化的数据上练成的。所以要存缩放标准
                        # 如果你只存了权重，当你把模型拿去给别人用时，别人不知道该如何缩放输入的 obs
                        "state_mean": ds.state_mean,
                        "state_std": ds.state_std,
                        "act_dim": int(ds.act_dim),
                        "state_dim": int(ds.state_dim),
                        # 记录了这是第几次迭代跑出来的最高分，方便你后续回溯实验
                        "best_eval": float(best_eval),
                        "it": int(it),
                    },
                    ckpt_path,
                )
                print(f"[save] best -> {ckpt_path} (best_eval={best_eval:.1f})")

    print(f"[done] best_eval={best_eval:.1f}")

if __name__ == "__main__":
    main()