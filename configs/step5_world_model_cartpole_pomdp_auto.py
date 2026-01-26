from __future__ import annotations

from configs.step5_world_model_cartpole_pomdp import Step5WorldModelConfig as BaseConfig


class Step5WorldModelConfig(BaseConfig):
    # auto-generated: override dataset path for this run
    dataset_path: str = 'data/cartpole_pomdp_from_step3_mixed.npz'
