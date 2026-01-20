from dataclasses import dataclass
from typing import Tuple


@dataclass
class Step4DecisionTransformerConfig:
    # --- 通用 ---
    run_name: str = "step4_dt_cartpole_pomdp"
    seed: int = 42
    device: str = "cuda"  # "cpu" 也行

    # --- env相关（用于评估 DecisionTransformer，不用于训练）---
    env_id: str = "CartPole-v1"
    pomdp_keep_idx: Tuple[int, ...] = (0, 2)  # 只给 [x, theta]
    use_delta_obs: bool = True                # 如果 step3 用了 delta_obs，建议这里也一致
    history_len: int = 1                      # DecisionTransformer 自己用 context_len“记忆”，这里 env 不需要堆历史

    # --- offline数据集 ---
    dataset_path: str = "data/cartpole_pomdp_from_step3_full.npz"
    # RTG 缩放：CartPole 最多 500 分，rtg / 500 会更稳定
    rtg_scale: float = 500.0

    # --- Decision Transformer 结构 ---
    context_len: int = 20         # DecisionTransformer 一次看多少步历史（K）
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    # timestep embedding 的上限（CartPole max_episode_steps 常见是 500）
    max_timestep: int = 512

    # --- 训练超参 ---
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    max_iters: int = 30_000
    log_every: int = 100
    eval_every: int = 2000
    eval_episodes: int = 10

    # --- 推理控制 ---
    #希望 DecisionTransformer 在评估时最大期望得分
    target_return: float = 500.0
    # True: 按分布采样（更随机，轨迹更像 step3 那种有熵的策略）
    # False: 直接 argmax（更稳，但可能缺探索）
    sample_actions: bool = True
