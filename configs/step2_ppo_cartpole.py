from dataclasses import dataclass

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
    lr: float = 3e-4
    update_epochs: int = 10
    minibatch_size: int = 256
    clip_coef: float = 0.2

    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # model
    hidden: int = 128

    # training length
    total_updates: int = 100   # 100轮 rollout+update
    print_every: int = 1
