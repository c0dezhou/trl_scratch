from dataclasses import dataclass

@dataclass
class ReinforceCartPoleConfig:
    # 环境
    env_id: str = "CartPole-v1"
    seed: int = 42

    # 算法
    gamma: float = 0.99
    normalize_returns: bool = True  # 降低方差，训练更稳

    # 训练
    episodes: int = 800
    print_every: int = 20 # 打印interval
    lr: float = 1e-3
    grad_clip: float = 1.0

    # 模型
    hidden: int = 128