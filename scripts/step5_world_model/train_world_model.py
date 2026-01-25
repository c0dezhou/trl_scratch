from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from core.nn_utils import get_device, seed_everything
from model_based.dynamics_dataset import DynamicsSequenceDataset
from model_based.dynamics_transformer import DynamicsTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="e.g. configs.step5_world_model_cartpole_pomdp")
    return p.parse_args()


# 在验证集上跑一遍，不更新参数，只看模型目前的预测精度如何
@torch.no_grad()
def evaluate(model: DynamicsTransformer, loader: DataLoader, done_coef: float) -> dict:
    model.eval()
    total = 0.0
    total_delta = 0.0
    total_done = 0.0
    n_batches = 0

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    mse = torch.nn.MSELoss(reduction="none")

    for batch in loader:
        states = batch["states"].to(next(model.parameters()).device)         # (B,K,D)
        actions = batch["actions"].to(states.device)                         # (B,K)
        timesteps = batch["timesteps"].to(states.device)                     # (B,K)
        valid = batch["valid"].to(states.device)                             # (B,K)
        delta_tgt = batch["delta_targets"].to(states.device)                 # (B,K,D)
        done_tgt = batch["done_targets"].to(states.device)                   # (B,K)
        trans_valid = batch["trans_valid"].to(states.device)                 # (B,K)

        out = model(states, actions, timesteps, valid)

        # masks
        v = (valid > 0).float()
        # tv 确保了 delta_loss 只在真正存在“下一时刻”的地方计算
        tv = (trans_valid > 0).float() * v

        # 物理预测误差 (MSE)
        delta_loss = mse(out.delta, delta_tgt).mean(dim=-1)                  # (B,K)
        delta_loss = (delta_loss * tv).sum() / (tv.sum() + 1e-6)

        # 存活预测误差 (BCE)
        done_loss = bce(out.done_logits, done_tgt)
        done_loss = (done_loss * v).sum() / (v.sum() + 1e-6)

        # 这是一个多目标学习。物理位移（delta）和是否死亡（done）的量级不同，
        # 我们需要一个系数来平衡它们，确保模型不会为了死磕物理规律而忘了学习判断死亡边界
        loss = delta_loss + done_coef * done_loss

        total += float(loss.item())
        total_delta += float(delta_loss.item())
        total_done += float(done_loss.item())
        n_batches += 1

    return {
        "loss": total / max(1, n_batches),
        "delta_loss": total_delta / max(1, n_batches),
        "done_loss": total_done / max(1, n_batches),
    }


# 负责基础设施建设。它加载配置、初始化 GPU/CPU 环境、实例化数据集（train_ds, val_ds）以及模型（DynamicsTransformer）
def main():
    args = parse_args()
    m = importlib.import_module(args.config)
    cfg = m.Step5WorldModelConfig()

    seed_everything(int(cfg.seed))
    device = get_device(cfg.device)

    os.makedirs("checkpoints", exist_ok=True)

    # dataset
    train_ds = DynamicsSequenceDataset(
        npz_path=cfg.dataset_path,
        context_len=int(cfg.context_len),
        split="train",
        val_frac_episodes=float(cfg.val_frac_episodes),
        seed=int(cfg.seed),
    )
    val_ds = DynamicsSequenceDataset(
        npz_path=cfg.dataset_path,
        context_len=int(cfg.context_len),
        split="val",
        val_frac_episodes=float(cfg.val_frac_episodes),
        seed=int(cfg.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False)

    # model
    model = DynamicsTransformer(
        state_dim=int(train_ds.state_dim),
        act_dim=int(train_ds.act_dim),
        context_len=int(cfg.context_len),
        d_model=int(cfg.d_model),
        n_heads=int(cfg.n_heads),
        n_layers=int(cfg.n_layers),
        d_ff=int(cfg.d_ff),
        dropout=float(cfg.dropout),
        max_timestep=int(cfg.max_timestep),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    mse = torch.nn.MSELoss(reduction="none")

    ckpt_last = f"checkpoints/{cfg.run_name}_last.pt"
    ckpt_best = f"checkpoints/{cfg.run_name}_best.pt"

    best = float("inf")

    model.train()
    it = 0
    while it < int(cfg.max_iters):
        for batch in train_loader:
            it += 1
            model.train()

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

            delta_loss = mse(out.delta, delta_tgt).mean(dim=-1)  # (B,K)
            delta_loss = (delta_loss * tv).sum() / (tv.sum() + 1e-6)

            done_loss = bce(out.done_logits, done_tgt)
            done_loss = (done_loss * v).sum() / (v.sum() + 1e-6)

            loss = delta_loss + float(cfg.done_coef) * done_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()

            if it % int(cfg.log_every) == 0:
                print(
                    f"[train] it={it:6d} "
                    f"loss={loss.item():.6f} "
                    f"delta={delta_loss.item():.6f} "
                    f"done={done_loss.item():.6f}"
                )

            if it % int(cfg.eval_every) == 0:
                metrics = evaluate(model, val_loader, done_coef=float(cfg.done_coef))
                print(
                    f"[val]   it={it:6d} "
                    f"loss={metrics['loss']:.6f} "
                    f"delta={metrics['delta_loss']:.6f} "
                    f"done={metrics['done_loss']:.6f}"
                )

                # save last
                pack = {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    # 输入的原始数据必须使用和训练时完全一样的均值和方差进行归一化
                    "state_mean": train_ds.state_mean,
                    "state_std": train_ds.state_std,
                    "state_dim": train_ds.state_dim,
                    "act_dim": train_ds.act_dim,
                    "it": it,
                    "val_loss": metrics["loss"],
                }
                torch.save(pack, ckpt_last)

                # save best
                if metrics["loss"] < best:
                    best = metrics["loss"]
                    torch.save(pack, ckpt_best)
                    print(f"[ckpt] best updated: {ckpt_best} (val_loss={best:.6f})")

            if it >= int(cfg.max_iters):
                break

    print(f"[done] last={ckpt_last} best={ckpt_best}")


if __name__ == "__main__":
    main()
