# baseline：MLP + T=1
from __future__ import annotations

from configs.step3_ppo_cartpole_pomdp_tr import PPOCartPolePOMDPTRConfig

RUN_NAME = "step3_pomdp_mlp_t1"
CFG = PPOCartPolePOMDPTRConfig()

# baseline: 只看当前（T=1）
CFG.model_type = "mlp"
CFG.history_len = 1

# 仍然是 POMDP（只给 x, theta）
CFG.pomdp_keep_idx = (0, 2)

# sanity check：
# CFG.pomdp_keep_idx=None

# 可选：想用 cuda 就改这里
# CFG.device = "cuda"
