from dataclasses import dataclass
from typing import Optional

@dataclass
class PPOCartPoleConfig:
    env_id: str = "CartPole-v1"
    seed: int = 42

    # rollout
    num_envs: int = 1
    rollout_steps: int = 2048   # 每轮采样多少步再更新

    # discount / GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # optimization
    # lr: float = 3e-4
    lr: float = 1e-4
    update_epochs: int = 10
    minibatch_size: int = 256
    clip_coef: float = 0.2
    target_kl: Optional[float] = None

    vf_coef: float = 0.5
    clip_vloss: bool = False
    # ent_coef: float = 0.01
    ent_coef: float = 0.005
    max_grad_norm: float = 0.5

    # model
    hidden: int = 128

    # training length
    total_updates: int = 600   # 500轮 rollout+update
    print_every: int = 1

# # 1. 缩短历史
# self.history_len = 4  # 32 -> 4

# # 2. 降低学习率，防止震荡
# self.lr = 1e-4        # 3e-4 -> 1e-4

# # 3. (可选) 稍微降低 entropy 系数，让它后期稳一点
# self.ent_coef = 0.005 # 0.01 -> 0.005
