import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.transformer_min import MiniGPT
from configs.step0_transformer import Step0Config


@torch.no_grad() # 测试时不记梯度，省显存，跑得快
def test_shape():
    cfg = Step0Config(dropout=0.0)
    model = MiniGPT(
        vocab_size=cfg.vocab_size,
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=0.0
    )
    # batch = 2, length = 10
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    # 跑一次向前传播
    y = model(x)
    # 确保：输入进来 [B, T]，出去变成 [B, T, V]
    assert y.shape == (2, 10, cfg.vocab_size), y.shape
    print("* shape test passed:", y.shape)


@torch.no_grad()
def test_causal_mask_no_future_leak():
    """
    核心测试：
    - 构造两条序列 a,b
    - 它们前半段相同，后半段不同
    - 由于 causal mask，前半段位置的 logits 必须完全一样（不能受未来 token 影响）
    """
    # 1. 必须用 eval() 模式！关掉 Dropout！
    # 如果不开 eval，Dropout 会随机扔掉一些神经元，导致两次输出本该一样却不一样了。
    torch.manual_seed(0)
    cfg = Step0Config(dropout=0.0)
    model = MiniGPT(
        vocab_size=cfg.vocab_size,
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=0.0
    ).eval()

    T = 16
    # 2. 构造两个序列 A 和 B
    # A for example: [10, 20, 30, 40, 50, 60]
    a = torch.randint(0, cfg.vocab_size, (1, T))
    b = a.clone()

    # 3. 修改 B 的后半段（未来）
    # B: [10, 20, 30, 99, 88, 77]
    # 注意：前 3 个 token (10,20,30) 是一模一样的！
    b[:, T//2:] = torch.randint(0, cfg.vocab_size, (1, T - T//2))

    la = model(a)  # [1,T,V]
    lb = model(b)

    # 4. 比较前半段 logits 的最大差异
    # 我们只看前半段的输出 (la[:, :T//2] vs lb[:, :T//2])
    # 理论上：预测第 3 个词时，模型只能看到前 2 个词。
    # 既然 A 和 B 的前 2 个词一样，那么模型对第 3 个词的预测 logits 必须完全一样！
    # 哪怕 A 和 B 的后半段天差地别，也不应该影响前半段的预测。
    """
    Sequence A: [A, B, C | D, E]
    Sequence B: [A, B, C | X, Y]
                      ^
                      |
       在此刻，模型眼里 A 和 B 应该是一模一样的。
       未来的 D/E 或 X/Y 不应该改变它对 C 的理解。
    """
    # 标准写法。它的目的是把两个复杂的、多维的张量对比，简化为一个单一的数值，用来衡量它们之间的“最大不一致程度
    diff = (la[:, :T//2] - lb[:, :T//2]).abs().max().item() # .item() （转为 Python 数值）
    assert diff < 1e-6, diff
    print("* causal mask test passed, max_diff =", diff)


def test_backward_has_grads():
    torch.manual_seed(0)
    # 1.初始化模型
    cfg = Step0Config(dropout=0.0)
    model = MiniGPT(
        vocab_size=cfg.vocab_size,
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=0.0 #关闭了网络中所有的 Dropout 层
    )
    x = torch.randint(0, cfg.vocab_size, (4, 12))
    logits = model(x)  # [B,T,V]

    # 2.随便构造一个目标，让 loss 能反传
    target = torch.randint(0, cfg.vocab_size, (4, 12))
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1))
    
    # 3.反向传播
    loss.backward()

    # 4.检查梯度
    grad_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sum += p.grad.abs().sum().item()

    # 如果 grad_sum == 0，说明梯度断了，模型永远学不会任何东西
    assert grad_sum > 0.0
    print("* backward test passed, grad_sum =", grad_sum)


if __name__ == "__main__":
    test_shape()
    test_causal_mask_no_future_leak()
    test_backward_has_grads()
    print("* All Step0 sanity tests passed.")
