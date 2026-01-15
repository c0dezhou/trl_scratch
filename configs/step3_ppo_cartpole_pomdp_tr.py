# Step3 基类 + 默认 GPT-T32
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from configs.step2_ppo_cartpole import PPOCartPoleConfig

ModelType = Literal["mlp", "gpt"]


@dataclass
class PPOCartPolePOMDPTRConfig(PPOCartPoleConfig):
    # runtime
    device: str = "cpu"  # "cpu" or "cuda"

    # POMDP: CartPole obs=[x, x_dot, theta, theta_dot]
    # keep_idx=(0,2) -> 只给 [x, theta]
    pomdp_keep_idx: Optional[Tuple[int, ...]] = (0, 2)

    # sanity check
    # pomdp_keep_idx=None

    # history length T: env 输出 (T, D)
    # history_len: int = 32
    history_len: int = 1

    # model backbone
    model_type: ModelType = "gpt"  # "mlp" baseline / "gpt" transformer

    # transformer (GPT policy) hparams (only for model_type="gpt")
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.0


# ---- exports for trainer ----
RUN_NAME = "step3_pomdp_gpt_t32"
CFG = PPOCartPolePOMDPTRConfig()
