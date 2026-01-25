from __future__ import annotations
"""
拿验证集（val）的序列数据，评估 DynamicsTransformer 预测的：

    状态增量 delta 的均方误差（MSE）
    done（终止） 的分类准确率（accuracy）

它不和环境交互，纯离线
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.nn_utils import get_device
from model_based.dynamics_dataset import DynamicsSequenceDataset
from model_based.dynamics_transformer import DynamicsTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    # python -m scripts.xxx --ckpt checkpoints/xxx.pt --batch_size 256
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device("cuda")

    # PyTorch 2.6+ 默认 weights_only=True(限制反序列化的对象类型)，ckpt 里有 numpy array 时会报错
    # 训练的 ckpt 可以安全用 weights_only=False
    pack = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = pack["cfg"]

    # 从 ckpt 拿 cfg，然后构造验证集 Dataset
    ds = DynamicsSequenceDataset(
        npz_path=cfg["dataset_path"],
        context_len=int(cfg["context_len"]),
        split="val",
        val_frac_episodes=float(cfg["val_frac_episodes"]),
        seed=int(cfg["seed"]),
    )
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)

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
    # ckpt 里 pack["model"] 存的是 state_dict，不是整个 model 对象，这也更安全、更可迁移
    model.load_state_dict(pack["model"])
    model.eval()

    # 四个计数器：你在累计“加权平均”
    # mse_sum / trans_cnt = delta 的平均 MSE（只在有效 transition 上算）
    # done_acc_sum / done_cnt = done 的平均 accuracy（只在有效 timestep 上算）
    mse_sum = 0.0
    done_acc_sum = 0.0
    done_cnt = 0.0
    trans_cnt = 0.0

    for batch in loader:

        states = batch["states"]            # (B, K, state_dim)  归一化后的 state（大概率）
        actions = batch["actions"]          # (B, K)             离散动作 id
        timesteps = batch["timesteps"]      # (B, K)             时间步
        valid = batch["valid"]              # (B, K)             真实数据=1，padding=0

        delta_tgt = batch["delta_targets"]  # (B, K, state_dim)  目标 delta：s_{t+1}-s_t（归一化空间）
        done_tgt = batch["done_targets"]    # (B, K)             目标 done(0/1)

        trans_valid = batch["trans_valid"]  # (B, K)             transition 是否有效
        """
        valid：这个 timestep 是否真实存在（不是 padding）
        trans_valid：这个 timestep 是否有“下一步”（能定义 delta）
        通常最后一步没有 s_{t+1}，所以 trans_valid 最后一个位置会是 0
        """

        out = model(states, actions, timesteps, valid)

        v = (valid > 0).float() # 所有有效 timestep
        tv = (trans_valid > 0).float() * v # 所有有效 transition（有下一步可算 delta 的 timestep）

        # delta 的 MSE（归一化空间）
        err = (out.delta - delta_tgt) ** 2 # 每个维度的预测误差的平方
        err = err.mean(dim=-1)  # (B,K) 把 state_dim 上的平方误差平均成一个标量 （每个 timestep 一个标量误差）
        mse_sum += float((err * tv).sum().item()) # * tv：只保留有效 transition 的误差， sum：累加到全局
        trans_cnt += float(tv.sum().item()) # trans_cnt：统计一共算了多少个 transition

        # done accuracy
        pred_done = (torch.sigmoid(out.done_logits) > 0.5).float()
        done_acc_sum += float(((pred_done == done_tgt) * v).sum().item())
        done_cnt += float(v.sum().item())

    mse = mse_sum / (trans_cnt + 1e-6)
    done_acc = done_acc_sum / (done_cnt + 1e-6)

    print(f"[offline eval] delta_mse(norm)={mse:.6f}  done_acc={done_acc:.4f}")

    # 额外：把 delta_mse 从 normalized 变回 raw 尺度（近似）
    state_std = torch.tensor(pack["state_std"], device=device).float()
    raw_mse = mse * float((state_std ** 2).mean().item())
    print(f"[offline eval] delta_mse(raw approx)={raw_mse:.6f}")


if __name__ == "__main__":
    main()
