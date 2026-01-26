from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Step5WorldModelConfig:
    """Step5: Dynamics Transformer (World Model) + MPC(Random Shooting).

    训练：离线监督学习
      输入: 最近 K 步 (s_t, a_t)
      输出: 预测 delta_s = s_{t+1}-s_t, 以及 done_t

    评估：
      1) offline one-step 预测误差
      2) online MPC：用模型在想象里 rollout N 条动作序列，选第一步动作执行
    """

    # --- 通用 ---
    run_name: str = "step5_wm_cartpole_pomdp"
    seed: int = 42
    device: str = "cuda"  # "cpu" 也行

    # --- env相关（用于 MPC 在线评估，不用于训练）---
    env_id: str = "CartPole-v1"
    pomdp_keep_idx: Tuple[int, ...] = (0, 2)  # 只保留 [x, theta]
    use_delta_obs: bool = True                # 和 step3/step4 保持一致
    history_len: int = 1                      # online eval 用 1（模型自己吃序列）

    # --- 离线数据集 ---
    dataset_path: str = "data/cartpole_pomdp_from_step3_mixed.npz"
    context_len: int = 20
    # 按 episode 切分 train/val（更稳妥，不泄漏）
    val_frac_episodes: float = 0.1

    # --- 模型 ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    max_timestep: int = 512

    # --- 训练 ---
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    max_iters: int = 30_000
    log_every: int = 100
    eval_every: int = 2000

    # done loss 权重（delta_loss + done_coef * done_loss）
    done_coef: float = 1.0
    # rollout loss（多步预测稳定性）
    rollout_steps: int = 5
    rollout_coef: float = 1.0

    # --- MPC ---
    mpc_horizon: int = 25
    mpc_num_samples: int = 1024
    mpc_done_threshold: float = 0.5
    mpc_gamma: float = 1.0
    # reward shaping inside MPC (does NOT affect env return)
    mpc_state_cost_coef: float = 1.0
