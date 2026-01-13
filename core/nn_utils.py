# 小工具：init、mask、device、seed等

import os
import random
import numpy as np
import torch
from typing import Optional

def seed_everything(seed: int = 42) -> None:
    """固定随机数
    -Transformer:每次dropout丢掉相同的参数
    -RL:在对比超参时，降低随机数导致的波动对调参的影响"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed) # gpu

    # 让cuDNN可复现（会慢一点）
    # 强制使用“确定性算法”
    torch.backends.cudnn.deterministic = True
    # 关闭“自动寻找最快算法”的机制
    torch.backends.cudnn.benchmark = False

def get_device(device: Optional[str] = None) -> torch.device:
    """
    自动选择设备：
    - CUDA: gpu
    - MPS: mac
    - CPU
    """
    if device:
        return torch.device(device)
    else:    
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
