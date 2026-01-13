# 滞后预测任务 (Lag Prediction Task):预测 K 步之前的 token（比如 K=3）。
# 这件事必须“看历史”，所以 attention 的意义会更明显

import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.nn_utils import seed_everything, get_device
from core.transformer_min import MiniGPT
from configs.step0_transformer import Step0Config
"""
1. 它是怎么做到“往回看”的？（坐标系的魔力）
Attention 本身是不分先后的（词袋模型），是“位置编码 (Positional Encoding)”给了它坐标感。
即: x = self.tok_emb(idx) + self.pos_emb(pos)
没有位置编码时：模型眼中只有一堆词，就像把 [A, B, C] 扔进磁场，它分不清谁在前谁在后。
加上位置编码后：每个 Token 的向量里都揉进了一份“坐标信息”。

第 0 位的 A 会带上“我是 0”的特征。
第 3 位的 D 会带上“我是 3”的特征。

关键点：Attention 公式里的 Q 和 K 并不是盲目扫描，而是计算向量之间的相关性。模型会学习出一套权重，使得当 Q 携带“我是位置 3”的信息时，它与携带“我是位置 0”信息的 K 产生极高的点积（Dot Product）得分。

2. 注意力公式储存了这个信息吗？

公式本身不储存信息，它是**“检索逻辑”**。真正的“知识”储存在 W_q, W_k, W_v 这三个矩阵参数里。
让我们复盘一下 Q · K^T 的精确退格过程：

Q (Query)：位置 3 的向量经过 W_q 变换，变成了一个“找 3 步之前”的搜索请求。
K (Key)：位置 0, 1, 2 的向量经过 W_k 变换，变成了各自身份的标签。
计算分数：Q_3 · K_0^T 的结果最大。为什么？因为 W_q 和 W_k 被训练成这样了。它们配合默契，能够精准识别出“距离为 3”的坐标差。

Softmax：把分数变成概率。此时位置 3 对位置 0 的权重接近 1.0，对其他位置接近 0。
搬运：用这个权重去乘 V_0（也就是 x0 的特征），信息就这样被“瞬移”到了当前位置。
结论：信息储存在参数矩阵里，这些参数形成了一种“模式识别”，专门识别特定的位置偏移。

3. 注意力是怎么做到“精确”的？

注意力并不是在“数数”。它在做的是高维空间里的对齐。
可以把 W_q 想象成一把特制的钥匙，把 W_k 想象成一排锁。
当模型在位置 t 时，钥匙的形状是由“位置 t 的编码”决定的。
由于任务要求找 t-3，在训练过程中，梯度下降会不断打磨这把钥匙，直到它能精准打开位置 t-3 那把锁。

总结整个闭环

训练前：模型是个色盲且散光的家伙，乱看一气。
训练中：模型随机看，结果错了。
AdamW 跑过来，计算出：“如果你把位置 3 的 W_q 稍微转个角度，它就能对准位置 0 的 W_k 了。”
经过几百次调整，参数矩阵里终于形成了一个**“偏移 3 位”的映射关系**。
训练后：无论你给什么序列，只要位置编码输入进来，参数矩阵就会自动把 t 的能量导向 t-3。
这就是为什么 Acc 能到 100%。 因为这个规律太死板了，只要位置编码准确，Transformer 这种全连接的注意力机制学这种“平移规律”就像满分学霸做 1+1 一样简单。
"""

def make_batch(batch_size: int, T: int, vocab: int, K: int, device: torch.device):
    """
    生成随机序列 x: [B,T]
    训练目标：预测 K 步之前的 token

    target[t] = x[t-K]  (t>=K)
    target[t] = -100    (t<K)  # 用 ignore_index 忽略前 K 个位置
    """
    # 1. 生成随机原始序列
    x = torch.randint(0, vocab, (batch_size, T), device=device)  # [B,T]

    # 2. 初始化 Target 为 -100
    # 在 PyTorch 的 CrossEntropyLoss 中，ignore_index=-100 是一个默认协议
    target = torch.full((batch_size, T), -100, device=device, dtype=torch.long)  # [B,T]
    
    # 3. 制造偏移（滞后 K 步）
    # x[:, :-K] 取出从开头到倒数第 K 个词
    # target[:, K:] 把这些词放在从第 K 个位置开始的地方
    target[:, K:] = x[:, :-K]  # [B,T-K] 复制到后面
    # 即x: [A, B, C, D, E]
    # target: [-100, -100, A, B, C]

    return x, target


# make_batch：制造了一堆“有时差”的填词题。
# model(x)：Transformer 睁开“注意力之眼”去序列里到处看。
# CrossEntropy：对比模型搬运过来的词和 K 步前的真词对不对。
# loss.backward()：如果看错了位置，就微调 Attention 的参数，让它下次看准点。
def main():
    cfg = Step0Config()
    seed_everything(cfg.seed)
    device = get_device()

    model = MiniGPT(
        vocab_size=cfg.vocab_size,
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"[Step0] device={device}, vocab={cfg.vocab_size}, T={cfg.max_len}, K={cfg.lag_k}")

    model.train()
    for step in range(1, cfg.steps + 1):
        x, target = make_batch(cfg.batch_size, cfg.max_len, cfg.vocab_size, cfg.lag_k, device)

        logits = model(x)  # [B,T,V]

        # CrossEntropy 需要：
        # logits: [B*T, V]
        # target: [B*T]
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            target.reshape(-1),
            ignore_index=-100 # 核心：忽略掉没法预测的前 K 步
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 如果梯度的范数超过 1.0，就强行把它缩小。这能防止模型在训练过程中“跑飞”（Loss 突然变成 NaN）。
        # 在 RL 训练中，这一行几乎是必加的，因为 RL 的信号极其不稳定。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0: # 200step打印一下
            # 计算有效位置的准确率（t>=K）
            with torch.no_grad():
                pred = logits.argmax(dim=-1)  # [B,T]
                mask = (target != -100) # 找出哪些位置是有意义的，待会儿比较的时候会压平成一维
                acc = (pred[mask] == target[mask]).float().mean().item() # item（） 从tensor变成python 浮点数
            # 成功标准：如果 acc 达到了 95% 以上，说明 Causal Self-Attention 能够非常精准地在 T 维（时间轴）上定位到 t-K 的位置。
            print(f"step {step:4d} | loss {loss.item():.4f} | acc {acc*100:.1f}%")

    print("done.")


if __name__ == "__main__":
    main()
