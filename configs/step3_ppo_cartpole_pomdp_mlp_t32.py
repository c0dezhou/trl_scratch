# 控制组：MLP + T=32
from __future__ import annotations

from configs.step3_ppo_cartpole_pomdp_tr import PPOCartPolePOMDPTRConfig

RUN_NAME = "step3_pomdp_mlp_t32"
CFG = PPOCartPolePOMDPTRConfig()

# 控制组：给 MLP 同样的历史长度，看它能不能追上 GPT
CFG.model_type = "mlp"
CFG.history_len = 32

CFG.pomdp_keep_idx = (0, 2)

# CFG.device = "cuda"
