from dataclasses import dataclass

@dataclass
class Step0Config:
    # 模型超参（小模型就够）
    vocab_size: int = 50
    max_len: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    # 训练超参
    batch_size: int = 128
    lr: float = 3e-4
    steps: int = 2000

    # 玩具任务：预测 K 步之前的 token
    # target[t] = x[t-K]（前 K 个位置没有 label，用 ignore）
    lag_k: int = 3

    # 随机种子
    seed: int = 42
