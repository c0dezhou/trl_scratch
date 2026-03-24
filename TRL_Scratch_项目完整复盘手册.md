# TRL Scratch 项目完整复盘手册

> **定位**：这是一份帮你在 20 分钟内重建全部记忆、自信应对老师任何追问的项目复盘文档。

---

## 第一阶段：项目全景还原

### 1️⃣ 项目一句话本质

**一句话**：从零手搓一整条「序列建模 × 强化学习」流水线，在 CartPole POMDP 上跑通 REINFORCE → PPO → Decision Transformer → 世界模型+MPC 五个阶段，证明 Transformer 能统一处理"记忆 / 决策 / 离线模仿 / 物理预测"四类 RL 子问题。

**实际应用场景**：现实中大量控制问题都是部分可观测的（POMDP）——你看不到全部状态。比如自动驾驶只有摄像头看不到绝对速度、机器人关节传感器有噪声或缺失。本项目模拟这种场景：把 CartPole 的速度信息遮掉，只给位置和角度，然后让不同方法在这种"盲区"下做决策。

**研究方向**：离线强化学习（Offline RL）+ 序列决策建模（Sequence Decision Making）+ Model-Based RL（世界模型）。

**和常规方法的不同**：

- 常规做法是在 MDP（全观测）环境上直接跑 PPO 就结束了。
- 本项目的特色是 **(1)** 人为构造 POMDP 来研究"信息缺失下的决策"；**(2)** 用同一个 Transformer 骨架贯穿在线 PPO、离线 DT、世界模型三条路线，形成对比实验；**(3)** 所有模块从零手写（包括 LayerNorm、MultiHeadAttention），不依赖 HuggingFace 等高层库。

---

### 2️⃣ 项目整体架构图

```
TRL_Scratch
├── 🧱 核心组件层 (core/)
│   ├── transformer_min.py    ← 手搓最小 GPT（LayerNorm / MHA / FFN / Block / MiniGPT）
│   ├── gpt_decoder_core.py   ← 可复用 Decoder backbone（给 PPO-GPT / DT / 世界模型共用）
│   └── nn_utils.py           ← seed 固定 / device 自动选择
│
├── 🌍 环境层 (envs/)
│   ├── pomdp_wrappers.py     ← MaskObs(遮速度) + DeltaObs(差分近似速度) + HistoryStack(滑动窗口)
│   └── gym_factory.py        ← 环境工厂：统一创建 MDP / POMDP 环境
│
├── 🧠 模型层 (models/)
│   ├── policy_mlp.py         ← REINFORCE 用的最简策略网络
│   ├── actor_critic_mlp.py   ← PPO 用的 Actor-Critic MLP
│   └── actor_critic_gpt.py   ← PPO 用的 Actor-Critic GPT（Transformer policy）
│
├── ⚡ 在线 RL 层 (rl/)
│   ├── reinforce.py          ← REINFORCE 算法（compute_returns + loss）
│   ├── ppo.py                ← PPO Clipped Objective + GAE
│   └── buffer.py             ← Rollout Buffer（预分配显存 + GAE + minibatch 迭代器）
│
├── 📦 离线 RL 层 (offline/)
│   ├── dt_dataset.py         ← 离线轨迹切片（Episode 还原 / RTG 计算 / 滑动窗口 / padding）
│   └── decision_transformer.py ← Decision Transformer 模型（3-token 交织 / causal+padding mask）
│
├── 🔮 世界模型层 (model_based/)
│   ├── dynamics_dataset.py   ← 世界模型的训练数据集（delta_target / trans_valid）
│   ├── dynamics_transformer.py ← Dynamics Transformer（预测 Δs + done）
│   └── mpc.py                ← Random Shooting MPC（N 条候选 × H 步想象 rollout）
│
├── 📊 配置层 (configs/)
│   └── step0~step5 各 dataclass 配置文件
│
└── 🚀 脚本层 (scripts/)
    ├── step0_gpt/            ← 最小 GPT 验证（copy task）
    ├── step1_reinforce/      ← REINFORCE 训练
    ├── step2_ppo/            ← PPO（MDP 全观测）
    ├── step3_ppo_pomdp/      ← PPO（POMDP + MLP/GPT 对比）
    ├── step4_dt/             ← 数据采集 → DT 训练 → DT 评估
    └── step5_world_model/    ← 世界模型训练 → 离线评估 → MPC 在线评估
```

**每一层的设计逻辑与取舍**：

**核心组件层**：为什么手搓 Transformer？因为项目的核心目的就是"理解"。直接调 HuggingFace 的 GPT-2 虽然方便，但你无法我的解释 LayerNorm 的公式、causal mask 怎么实现。手搓保证每一行代码你都能讲清楚。如果不手搓，你就只是一个"调包侠"。

**环境层**：为什么要做 POMDP？因为 CartPole 的 MDP 版本太简单了，一个 MLP 就能秒杀（Step2 直接拿满分 500）。只有在 POMDP 下（遮掉速度），才能体现 Transformer 利用历史信息的优势。DeltaObs 是一个 trick：用相邻两帧的差分来近似速度——类似物理学里的"用位移变化率估计速度"。

**模型层**：为什么同时有 MLP 和 GPT？这是对照实验的关键设计。MLP 只看当前帧（或把历史 flatten 后看），GPT 则用 attention 机制理解历史序列。对比两者的性能才能回答"Transformer 在 POMDP 下到底有没有用"。

**离线 RL 层**：为什么要做 Decision Transformer？因为它代表了一种全新的 RL 范式——把 RL 问题转化成序列预测（条件生成）问题。不需要 Bellman 方程，不需要 value function 的 bootstrap，只用监督学习就能做 RL。这是和 PPO 完全正交的方法。

**世界模型层**：为什么要做世界模型 + MPC？因为它代表了 Model-Based RL 的思路——先学物理规律（dynamics），再在想象中规划（planning）。这和 PPO（Model-Free, 试错学习）和 DT（离线模仿）形成三足鼎立的方法论对比。

---

## 第二阶段：技术细节深挖

### 3️⃣ 核心技术清单

#### 🔹 REINFORCE（Policy Gradient 的祖师爷）

**原理（讲给小白听）**：想象你在学打篮球。你每次投篮后记下"我的姿势"和"进没进"。投进了，你就记住这个姿势多用；没投进，就少用。REINFORCE 就是这个过程的数学版本。

**公式直觉**：`loss = -log_prob × Return`
- `log_prob` 是"我选这个动作的概率有多大"
- `Return` 是"这个动作最终带来了多少奖励"
- 乘在一起就是"好动作的概率增大，坏动作的概率减小"
- 加负号是因为优化器默认最小化 loss

**在项目中的作用**：Step1 的 baseline，最简单的策略梯度。

**缺点**：方差极大——一局好运不代表策略好，但 REINFORCE 会大幅增加好运局里所有动作的概率。收敛慢且不稳定。

**为什么不用它做后面的实验**：方差太大，不适合复杂环境。PPO 是它的大幅改进版。

---

#### 🔹 PPO（Proximal Policy Optimization）

**原理（类比）**：REINFORCE 是"考完试立刻烧掉试卷"（on-policy, 数据用完即弃）。PPO 的改进是"同一张试卷多看几遍"——通过 importance sampling ratio（新旧策略的概率比值）重复利用数据。但为了防止"抄作业抄太狠"（策略变化太大导致崩溃），加了一个 clip 机制。

**核心公式**：
```
ratio = π_new(a|s) / π_old(a|s)
L_clip = min(ratio × Adv, clip(ratio, 1-ε, 1+ε) × Adv)
```

- `ratio` > 1 说明新策略更倾向于选这个动作
- `clip` 把 ratio 限制在 [0.8, 1.2]（ε=0.2），防止一步走太远
- `Adv`（Advantage）= 实际回报 - 基线估计，正值代表"比预期好"

**在项目中的作用**：Step2（MDP）和 Step3（POMDP）的核心算法。Step3 是整个项目的枢纽——它训练的策略既用于在线评估，又用于收集离线数据给 Step4/5。

**关键 trick**：
- **GAE (λ)**：把 TD error 用指数衰减加权，平衡"方差 vs 偏差"。λ=0.95 接近 Monte Carlo（低偏差高方差），λ=0 就是纯 TD（高偏差低方差）。
- **Advantage 标准化**：`adv = (adv - mean) / std`，让梯度量级稳定。
- **梯度裁剪**：`max_grad_norm=0.5`，防止梯度爆炸。
- **Entropy Bonus**：`ent_coef × entropy`，鼓励探索，防止策略过早收敛到确定性策略。

---

#### 🔹 Transformer（Decoder-Only, GPT-style）

**原理（类比）**：想象你在读一本书。Transformer 的 self-attention 让你在读第 100 页时，能同时"回忆"前面所有页的信息，并自动判断哪些页最相关。Causal mask 则保证你不能"偷看"后面的页。

**在项目中的结构**：
- `d_model=128`：每个 token 被表示为 128 维向量
- `n_heads=4`：4 个注意力头，每个看 32 维
- `n_layers=2~3`：2-3 层堆叠
- `d_ff=256`：FFN 中间层 256 维
- Pre-LN 结构：`x = x + Attn(LN(x))`，比原始 Post-LN 更稳定

**为什么用 Pre-LN 而不是 Post-LN**：原始 Transformer 论文用 Post-LN（`LN(x + Attn(x))`），但实践中发现 Pre-LN 训练更稳定，梯度流更顺畅。GPT 系列都用 Pre-LN。

**context_len 的选择**：
- PPO-GPT (Step3): history_len=4，太长反而不好（短历史 + delta 特征更稳定）
- DT (Step4): context_len=20，需要看更长的轨迹来理解目标
- 世界模型 (Step5): context_len=20，需要足够的物理历史来预测未来

---

#### 🔹 POMDP Wrappers（核心工程设计）

**MaskObsWrapper**：把 CartPole 的 `[x, v, θ, ω]` → `[x, θ]`，遮掉速度 v 和角速度 ω。

**DeltaObsWrapper**：`obs_new = [obs_t, obs_t - obs_{t-1}]`。直觉：如果连续两帧 x 从 0.1 变到 0.15，那 Δx=0.05 就近似了速度。这是用"有限差分"弥补被遮掉的速度信息。

**HistoryStackWrapper**：把最近 T 帧堆成 `(T, obs_dim)` 的矩阵。这给了模型"记忆"——不是只看当前一帧，而是看一段视频。

**三者叠加的数据流**：
```
原始: [x, v, θ, ω] 
→ Mask: [x, θ]  
→ Delta: [x, θ, Δx, Δθ]  (obs_dim=4)
→ History(T=4): shape (4, 4)，即 4 帧 × 4 维
→ Flatten: 16 维向量喂给 Actor-Critic
```

---

#### 🔹 Decision Transformer

**原理（类比）**：传统 RL 是"边玩边学"。Decision Transformer 的思路完全不同——它把 RL 变成了"阅读理解"。给它一堆别人玩过的录像，它通过阅读这些录像学会玩。

**关键创新**：token 排列是 `(R_t, S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, ...)`。
- R_t 是 Return-to-Go（"从现在开始还能拿多少分"）
- 推理时，你给它一个很高的 R_t（比如 500），它就会输出能拿到 500 分的动作

**直觉理解**：就像你在和一个 GPS 对话：
- 你说："我要到终点（RTG=500）"
- GPS 看你的位置（State）
- GPS 告诉你下一步怎么走（Action）

**损失函数**：纯监督学习的 CrossEntropy。模型预测的动作 vs 数据集里的真实动作。

**关键实现细节**：
- RTG 缩放：除以 500（CartPole 最高分），防止数值过大
- State 归一化：Z-score，均值方差从训练集计算
- 右 padding + valid mask：防止模型关注到填充的垃圾数据
- 用 state token 的 hidden 预测 action（不是用 action token 或 RTG token）

---

#### 🔹 世界模型（Dynamics Transformer）+ MPC

**世界模型原理（类比）**：想象你在脑子里模拟下棋——"如果我走这一步，对手大概率会这样反应，然后我可以……"。世界模型就是让 AI 学会"在脑子里模拟物理世界"。

**Token 排列**：`[s_0, a_0, s_1, a_1, ...]`（2K 长度）。在 action token 位置预测：
- `delta_s = s_{t+1} - s_t`（状态变化量）
- `done_prob`（游戏是否结束的概率）

**为什么预测 delta 而不是直接预测 s_{t+1}**：如果直接预测下一帧，模型可以"偷懒"——直接输出当前帧（因为相邻帧差别很小，Loss 已经很低了）。预测 delta 强迫模型学习"动作对环境的因果影响"。

**MPC（Random Shooting）**：
1. 复制当前历史 N=1024 份
2. 为每份随机生成 H=25 步的动作序列
3. 用世界模型在想象中 rollout 25 步
4. 评分：`reward ≈ (1 - done_prob) - state_cost`
5. 选分数最高的那条序列的第一步执行
6. 到下一步重新规划（Receding Horizon）

---

### 4️⃣ 训练过程还原

#### 数据从哪里来？

| Step | 数据来源 |
|------|---------|
| Step0 | 人工构造的 copy task（预测 K 步前的 token） |
| Step1-3 | 在线采样：agent 边玩边收集（on-policy） |
| Step4-5 | 离线数据集：由 Step3 训练好的策略收集 200~500 条 episode，存为 .npz |

离线数据格式：`obs [N, state_dim]`, `actions [N]`, `rewards [N]`, `dones [N]`, `episode_ends [E]`。用 `episode_ends` 切回一局一局。

#### 损失函数

| Step | Loss |
|------|------|
| Step0 | CrossEntropy（预测 token） |
| Step1 | `-Σ(G_t × log π(a_t\|s_t)) `（REINFORCE） |
| Step2-3 | PPO: `L_clip + 0.5 × MSE(V, Return) - 0.005 × Entropy` |
| Step4 | CrossEntropy（DT 预测动作 vs 真实动作，只算 valid 位置） |
| Step5 | `MSE(Δs_pred, Δs_true) + 1.0 × BCE(done_logits, done_true)` |

#### 超参数一览（关键的）

**决定性参数**（调了效果变很多的）：
- `history_len`：从 32 调到 4 是重大改进。32 太长导致信噪比低，Transformer 注意力分散。
- `use_delta_obs`：开启 delta 特征是 POMDP 下的关键 trick，等于"免费恢复了速度信息"。
- `lr`：从 3e-4 降到 1e-4 让 PPO 更稳。
- `ent_coef`：从 0.01 降到 0.005 让后期策略更确定。
- `context_len`（DT/WM）：20 是合理值，太短记忆不够，太长训练慢且容易过拟合。
- `rtg_scale`：DT 的 RTG 必须缩放（除以 500），否则数值太大梯度爆炸。
- `mpc_num_samples`：MPC 的采样数量越多越好，但速度越慢。1024 是性能/速度的平衡点。

**调了没什么用的参数**：
- `clip_vloss`：开不开 value loss clipping 对 CartPole 影响不大。
- `target_kl`：KL 早停在这个简单环境中通常不会触发。
- `dropout`：PPO 用 0.0（RL 中通常不需要 dropout，因为数据本身就有探索噪声），DT/WM 用 0.1。

---

### 5️⃣ 典型问题推断与分析

#### 问题 1：PPO 在 POMDP 下不收敛 / 回报极低

**现象**：MLP + history_len=1 在 POMDP 下性能极差，因为只看 `[x, θ]` 两个数，完全没有速度信息。

**本质原因**：CartPole 的最优控制需要速度信息。没有速度，策略无法判断杆子是"正在倒"还是"正在恢复"。

**修复**：(1) 加 delta_obs 近似速度；(2) 加 history stack 让模型从历史中推断速度；(3) 用 Transformer 替代 MLP 来更好地利用历史。

**我的解释**："CartPole 本质上是一个需要知道一阶导数（速度）的控制问题。当我们遮掉速度时，就变成了 POMDP。DeltaObs 用有限差分近似了速度，HistoryStack 则给了模型通过多帧推断速度的能力。"

---

#### 问题 2：history_len=32 反而不如 history_len=4

**现象**：GPT + T=32 收敛比 T=4 慢，且最终性能可能更差。

**本质原因**：CartPole 的 POMDP 只需要 2-3 帧就能推断速度（差分）。32 帧带来了大量冗余信息，Transformer 的注意力被稀释，且计算量大幅增加。

**修复**：缩短 history_len 到 4。

**我的解释**："这其实是一个 context length 的 sweet spot 问题。太短缺信息，太长信噪比低。对于 CartPole 这种低维 POMDP，短窗口 + delta 特征就够了。如果是更复杂的 POMDP（如 Atari 的部分可见），可能需要更长的 context。"

---

#### 问题 3：Decision Transformer 推理时分数很低

**可能现象**：训练 loss 在降，但评估分数不高。

**本质原因**：(1) 推理时忘记归一化 state；(2) RTG 没有正确缩放；(3) padding/valid mask 不一致；(4) 推理时的窗口管理（滑动/截断）和训练时不匹配。

**修复**：确保 state_mean/state_std 在 ckpt 中保存，推理时加载同一组值；RTG 推理时也要除以 rtg_scale；padding 方式保持右 padding。

**我的解释**："这是离线 RL 的经典问题——训练和推理的分布不一致（distribution shift）。DT 训练时看到的是归一化后的数据，推理时如果喂原始数据，模型就'看不懂'了。"

---

#### 问题 4：世界模型 MPC 回报只有 80 多分

**现象**：离线评估 delta_mse 很低（0.004），但 MPC 在线回报只有 82.8。

**本质原因**：
- **误差累积（compounding error）**：one-step 预测很准，但 MPC rollout 25 步时误差会滚雪球般放大。这是 Model-Based RL 的经典难题。
- **Random Shooting 的效率低**：在高维动作序列空间中随机采样，命中好方案的概率很低。
- **训练数据分布 vs 规划分布不匹配**：训练数据来自 Step3 的策略，但 MPC 产生的状态-动作序列可能偏离这个分布（OOD 问题）。

**修复方向**：增加 num_samples、缩短 horizon、加入 reward shaping（state_cost 惩罚偏离中心的状态）、用更好的规划算法（CEM 替代 Random Shooting）。

**我的解释**："这体现了 Model-Based RL 的核心矛盾：模型预测的精度只在训练数据分布附近有保证，一旦规划过程探索到 OOD 区域，预测就不准了，而不准的预测又会导致规划进一步偏离，形成恶性循环。"

---

#### 问题 5：Transformer 输入对齐 / Padding / Mask 错误

**可能现象**：训练 loss 不降、模型输出随机、或者偶尔的 NaN。

**本质原因**：
- causal mask 和 padding mask 的组合方式不对（比如用 `|` 而不是 `&`）
- `valid` 的 dtype 不对（float vs bool），导致 mask 失效
- DT 的 3-token interleave 索引错误（`1::3` 取 state 位置 vs `0::3` 取 RTG 位置搞混）

**修复**：项目中 `_build_attn_mask` 函数仔细处理了这个问题：
```python
attn_mask = causal_mask & key_padding_mask  # 两者取 AND
```
True=允许看，False=禁止看。causal 禁止看未来，key_padding 禁止看 padding。

---

## 第三阶段：讲解准备

### 6️⃣ 20 分钟讲解大纲

#### 第一部分：动机（3 分钟）

**讲什么**：为什么做这个项目？解决什么问题？

**怎么讲**：
- 开场："现实中很多控制问题是部分可观测的——你看不到全部状态。"
- 举例：自动驾驶的传感器盲区、机器人的噪声传感器。
- 引出 CartPole POMDP："我用 CartPole 作为 testbed，人为遮掉速度信息，模拟这种场景。"
- 项目目标："我想用一套统一的 Transformer 架构，同时解决'记忆、决策、模仿、预测'四个问题。"

#### 第二部分：方法（8 分钟）

**讲什么**：五步流水线。

**怎么讲**：
- Step0 一笔带过："先手搓了一个最小 GPT 验证 Transformer 的基本能力。"
- Step1-2 快速过："用 REINFORCE 和 PPO 在全观测 MDP 上做了 baseline。"
- **Step3 重点讲**（3 分钟）：
  - POMDP 怎么构造的（MaskObs + DeltaObs + HistoryStack）
  - MLP vs Transformer 的对比实验设计
  - history_len=4 + delta 是最优配置
- **Step4 重点讲**（3 分钟）：
  - DT 的核心思想：RL as Sequence Modeling
  - 3-token 交织序列 `(R, S, A)`
  - 推理时用 target_return 控制行为
- **Step5 重点讲**（2 分钟）：
  - 世界模型学 Δs 和 done
  - MPC Random Shooting 的 imagine-then-act 范式

#### 第三部分：实验结果（4 分钟）

**讲什么**：各 Step 的结果。

**怎么讲**：

| Step | 结果 | 要点 |
|------|------|------|
| Step2 (PPO, MDP) | 500.0 ± 0.0 | 满分，说明 PPO 实现正确 |
| Step3 (PPO, POMDP, GPT) | 500.0 ± 0.0 | Transformer + delta 在 POMDP 下也能满分 |
| Step4 (DT, 离线) | 500.0 | 离线模仿也能达到满分 |
| Step5 (世界模型 + MPC) | 82.8 | 远低于其他方法，体现了 Model-Based 的挑战 |

- 重点分析 Step5 为什么低：误差累积、Random Shooting 效率低、OOD 问题。
- 这不是"失败"，而是一个有价值的 negative result，揭示了 Model-Based 方法的实际瓶颈。

#### 第四部分：问题与改进（3 分钟）

**讲什么**：遇到了什么问题，怎么解决的，还有哪些改进空间。

**怎么讲**：
- history_len 从 32 调到 4 的故事
- delta_obs 的发现过程
- 世界模型的 compounding error 问题
- 改进方向：CEM 替代 Random Shooting、多步 rollout loss、更好的数据覆盖

#### 第五部分：总结（2 分钟）

**讲什么**：总结贡献和思考。

**怎么讲**：
- "这个项目验证了 Transformer 作为'通用序列模型'在 RL 中的三种角色：记忆器（PPO-GPT）、决策器（DT）、模拟器（世界模型）。"
- "Model-Free（PPO）和 Offline（DT）在简单环境中效果很好，Model-Based（世界模型+MPC）则暴露了泛化和误差累积的根本挑战。"
- "所有代码从零实现，确保了对每个组件的深入理解。"

---

### 7️⃣  15 个问题

#### Q1: 为什么选 CartPole 而不是更复杂的环境？

**标准回答**：CartPole 是 RL 最经典的 benchmark，维度低（4D state, 2 actions）便于调试和理解。项目的目标是"理解原理"而非"刷 SOTA"。

**深一层**：CartPole 虽然简单，但它的 POMDP 版本（遮掉速度）已经足以产生有意义的对比实验。如果用 MuJoCo 等连续动作空间环境，还需要额外处理连续动作的 PPO（Gaussian policy），增加了无关的工程复杂度。

我选择了一个足够简单的环境来聚焦于方法论的理解和对比，如果要扩展到更复杂环境，需要处理连续动作空间和更高维的观测。

---

#### Q2: DeltaObs 和直接用更长的 history 有什么区别？用哪个更好？

**标准回答**：DeltaObs 是显式地给模型"差分信息"（近似速度），而长 history 需要模型自己通过 attention 去发现这个规律。DeltaObs 是一种先验知识注入。

**深一层**：实验表明 history_len=4 + delta 优于 history_len=32 无 delta。因为 delta 降低了模型的学习难度（不需要自己发现"相邻帧差分≈速度"这个物理规律）。但 delta 只能近似一阶导数，如果需要二阶信息（加速度），可能还需要更长的 history 或更复杂的特征。

---

#### Q3: Decision Transformer 的 RTG 条件控制真的有效吗？如果给它 RTG=1000 会怎样？

**标准回答**：RTG 是一种"条件生成"机制。给高 RTG 会让模型输出它在训练中看到的"高分行为"。

**深一层**：如果 RTG 设置超过训练数据中见过的最高值（比如 1000，而数据最高是 500），模型会进入 OOD 区域，行为不可预测。这是 DT 的根本局限——它只能做到训练数据中见过的最好水平，不能"超越老师"。这与 Online RL（PPO）不同，PPO 可以通过探索发现比初始策略更好的行为。

---

#### Q4: PPO 的 clip 机制是怎么防止策略崩溃的？

**标准回答**：clip 把新旧策略的概率比值限制在 [1-ε, 1+ε] 之间，防止一次更新走太远。

**深一层**：不 clip 的话，如果某个 batch 里某个动作的 advantage 特别大，ratio 可能变成 10 或 100，策略直接"跳"到一个完全不同的地方。而新的数据是在旧策略下收集的，在新策略看来完全不能用了（importance weight 偏差太大），导致后续更新全部歪掉——这就是"策略崩溃"。clip 相当于"每次只允许走一小步"。

---

#### Q5: 为什么世界模型预测 delta 而不是直接预测下一个 state？

**标准回答**：相邻状态非常接近，直接预测 s_{t+1} 模型会偷懒（直接输出 s_t，MSE 已经很低）。预测 delta 强迫模型学习因果关系。

**深一层**：这类似于 ResNet 的残差学习思想——学习"变化量"比学习"绝对值"更容易。此外，delta 的数值范围更小更集中，有利于训练稳定性。

---

#### Q6: 为什么用 Pre-LN 而不是 Post-LN？

**标准回答**：Pre-LN 在前向和反向传播中梯度流更稳定，不容易梯度消失。GPT 系列都用 Pre-LN。

**深一层**：Post-LN 的残差连接是 `LN(x + sublayer(x))`，LayerNorm 在求和之后才做，如果 sublayer 输出很大，求和后的分布可能偏移很多。Pre-LN 是 `x + sublayer(LN(x))`，先归一化再处理，梯度可以通过残差直接流回去（类似 highway），训练更稳。

---

#### Q7: MPC 的 Random Shooting 有什么缺点？有更好的方法吗？

**标准回答**：Random Shooting 在高维动作空间中效率很低——命中好方案纯靠运气。

**深一层**：更好的方法有 CEM（Cross-Entropy Method）：先随机采样，挑出 top-k 好方案，用它们的分布重新采样，迭代几轮。还有 MPPI（Model Predictive Path Integral），用带权的分布更新。如果动作空间可微，还可以用基于梯度的规划（如 PETS 中的方法）。

---

#### Q8: 离线数据质量对 DT 和世界模型的影响？

**标准回答**：非常关键。DT 只能模仿数据中见过的行为，数据质量上限就是 DT 的性能上限。世界模型也只在训练数据覆盖的状态-动作对上预测准确。

**深一层**：项目中用了两种数据：`full`（全部来自好策略）和 `mixed`（混合好策略和随机策略）。mixed 数据对世界模型更有利，因为覆盖的状态空间更广，减少了 MPC 规划时遇到 OOD 的风险。这体现了"探索 vs 利用"的 tradeoff。

---

#### Q9: 正交初始化（orthogonal init）在 RL 中为什么重要？

**标准回答**：正交初始化让权重矩阵的行向量互相垂直且模长为 1，保证信号强度在前向传播中不放大也不缩小。

**深一层**：在 RL 中，模型初始阶段的行为接近随机，如果初始化不好（比如全零或太大的随机值），梯度要么消失要么爆炸，导致最初的探索阶段就学不到任何有用信息。正交初始化让模型从一个"中性"的起点出发，第一批经验就能产生有效梯度。

---

#### Q10: 为什么 PPO 需要 minibatch 而 REINFORCE 不需要？

**标准回答**：REINFORCE 是纯 on-policy，每条轨迹只用一次就丢弃。PPO 通过 importance sampling ratio 可以重复利用同一批数据多次更新。

**深一层**：minibatch + 打乱顺序消除了时间相关性。如果按时间顺序更新，连续几步都是"杆子往左倒"的经验，模型会过度拟合这个局部模式。打乱后，每个 batch 混合了不同时间、不同 episode 的经验，学习更均匀。

---

#### Q11: 如果要扩展到连续动作空间（如 MuJoCo），需要改什么？

**标准回答**：主要改动：(1) Policy 输出从 Categorical 改为 Gaussian（输出均值和方差）；(2) DT 的 action embedding 从 nn.Embedding 改为 nn.Linear；(3) DT 的损失从 CrossEntropy 改为 MSE。

**深一层**：还需要处理动作范围裁剪（tanh squashing）、PPO 中的 log_prob 计算（从离散的 log_softmax 变成高斯的 log density），以及可能需要归一化 action space。

---

#### Q12: Transformer 的注意力机制在 RL 中具体在"关注"什么？

**标准回答**：在 POMDP 的 PPO 中，attention 学会了关注"最近几帧的变化趋势"来推断速度。在 DT 中，attention 关注与当前 RTG 匹配的历史行为模式。

**深一层**：如果做 attention visualization（可视化注意力权重），大概率会看到最后一帧对前 2-3 帧的注意力权重最高，因为这足以通过差分推断速度。对于更远的历史帧，注意力会衰减。

---

#### Q13: 为什么 Step3 的训练结果是所有后续步骤的基础？

**标准回答**：Step3 训练的 PPO-GPT 策略被用来收集离线数据集，这个数据集同时供 Step4（DT）和 Step5（世界模型）使用。数据质量直接决定了后续步骤的上限。

**深一层**：这其实模拟了现实中的 offline RL pipeline——你用一个已有的策略（可能不完美）收集数据，然后尝试从这些数据中学到更好的策略或更好的世界模型。如果 Step3 的策略很差，收集的数据覆盖面窄，后续步骤都会受限。

---

#### Q14: GAE 中的 λ 参数直觉上在控制什么？

**标准回答**：λ 控制"偏差 vs 方差"的 tradeoff。λ=0 是纯 TD（一步 bootstrap，低方差高偏差），λ=1 是纯 Monte Carlo（用完整回报，低偏差高方差）。

**深一层**：λ=0.95（项目默认值）意味着"主要信任实际经历的回报，但用 value function 做少量的平滑"。在 CartPole 这种 episode 不太长（最多 500 步）的环境中，高 λ 效果更好，因为完整回报的信号更准。在 episode 极长或 reward 很稀疏的环境中，可能需要更低的 λ。

---

#### Q15: 项目的真实应用可行性如何？这套方法能用在工业场景吗？

**标准回答**：项目的核心方法论（POMDP + Transformer policy、DT、世界模型 + MPC）在工业界都有实际应用。但 CartPole 是简化的 testbed，工业场景需要处理高维观测、连续动作、更复杂的 dynamics。

**深一层**：
- PPO + Transformer 已经在机器人控制（如 OpenAI 的 Dactyl 手指操控）中使用。
- Decision Transformer 的思想启发了后续的 GATO（DeepMind）、RT-2（Google）等多任务机器人模型。
- 世界模型 + 规划在自动驾驶（如 Waymo 的预测模型）和游戏 AI（如 Dreamer 系列）中广泛应用。
- 项目的工程实现（手搓 Transformer、模块化设计）体现了对底层原理的深入理解，这在工业场景的调试和定制化中非常重要。

---

### 8️⃣ 30 秒 + 3 分钟版本

#### 🎤 30 秒电梯陈述

"我做了一个从零手搓的强化学习实验平台。核心想法是：在一个信息不完整的控制环境中（POMDP），用同一个 Transformer 架构分别实现三种 RL 范式——在线策略优化（PPO+GPT），离线序列模仿（Decision Transformer），以及学习物理模型做想象规划（世界模型+MPC）。实验表明 Transformer 作为记忆器和决策器在 POMDP 下表现优异，但作为物理模拟器做长程规划时仍面临误差累积的挑战。所有代码从头实现，包括 Transformer 本身。"

#### 🎤 3 分钟小答辩陈述

"我的项目叫 TRL Scratch，是一个循序渐进的强化学习与序列建模实验集。

**动机**：现实中的控制问题往往是部分可观测的——比如你看不到所有传感器数据。我想研究的是：在这种信息缺失的条件下，Transformer 能不能作为一个通用的'大脑'来做不同类型的决策？

**方法**：我用 CartPole 环境做了 POMDP 改造——遮掉速度信息，只保留位置和角度。然后分五步搭建了实验流水线：先手搓了一个最小 GPT 验证 Transformer 的基本能力；然后从 REINFORCE 到 PPO，在全观测和部分可观测下训练策略；接着实现了 Decision Transformer——把 RL 变成序列预测的监督学习问题；最后做了世界模型加 MPC——让 AI 学会物理规律后在想象中做规划。

**核心发现**有三个：第一，在 POMDP 下，Transformer + 短窗口 + 差分特征（delta obs）的组合表现最好，history_len=4 优于 32；第二，Decision Transformer 用纯监督学习就能达到和 PPO 同等的满分表现，验证了'RL as Sequence Modeling'的可行性；第三，世界模型的 one-step 预测精度很高（MSE≈0.004），但 MPC 的在线表现只有 82.8 分——这暴露了 Model-Based RL 中误差累积和 OOD 问题的根本挑战。

**工程特色**：整个项目所有模块从零实现，包括 LayerNorm、MultiHeadAttention、PPO 的 GAE、DT 的 3-token 交织序列、世界模型的 Random Shooting MPC。没有依赖 HuggingFace 或 Stable Baselines 等高层库，确保了对每个组件的深入理解。

**改进方向**：世界模型部分可以用 CEM 替代 Random Shooting 提升规划效率；可以加入 multi-step rollout loss 提升长程预测稳定性；也可以扩展到连续动作空间的环境如 MuJoCo。"

---

## 第四阶段：帮你真正理解

### Transformer 在这个项目中到底是怎么"想"的？

让我用一个统一的比喻把三种 Transformer 用法串起来——把 Transformer 想象成一个**会议室里的团队**：

**场景 1：PPO-GPT（在线决策）**

会议室里坐着 4 个人（history_len=4），每人代表一个时间帧的观测 `[x, θ, Δx, Δθ]`。老板（最后一帧）要做决策。他通过 attention 问前面三个人："你们各自记得什么？"第 3 个人说"我当时角度偏了 0.05"，第 2 个人说"我偏了 0.03"。老板综合后判断："角度在加速偏离，必须向右推！"然后 actor head 输出动作，critic head 输出"我觉得当前局面大概能撑 300 步"。

**场景 2：Decision Transformer（离线模仿）**

会议室里坐着 60 个人（3 × context_len=20）。每 3 个人一组：第一个人拿着写了"目标分数"的牌子（RTG），第二个人拿着"当时的状态"照片，第三个人拿着"当时做了什么"的记录。Transformer 浏览整个房间，看到"目标 500 分 + 角度偏右 → 往右推"的模式反复出现，就学会了。推理时，你把牌子写成"500"，它就输出高分策略。

**场景 3：世界模型（物理模拟）**

会议室里坐着 40 个人（2 × context_len=20）。状态和动作交替排列。Transformer 的任务是"预测物理后果"——看到 `[s_t, a_t=向右推]`，它要推理出 `s_{t+1} - s_t`（角度减小了多少，位置移动了多少）以及"这一步会不会游戏结束"。它就像一个物理引擎，学会了牛顿定律的 CartPole 版本。

### 关键公式的直觉

**GAE 的直觉**：
<!-- ``` -->
> $ δ_t = r_t + γ·V(s_{t+1}) - V(s_t) $
<!-- ``` -->
δ_t 就是"这一步比我预期的好多少"。如果 δ>0，说明这步做得好；δ<0 说明做得差。GAE 把连续多步的 δ 用衰减加权（$ λ^k $），得到一个综合评价。

**PPO Clip 的直觉**：
<!-- ``` -->
> $ clip(ratio, 0.8, 1.2) × Advantage $
<!-- ``` -->
想象你在调音量旋钮。ratio 是"新策略和旧策略的差距"。clip 就是给旋钮加了限位器——不管你多兴奋，一次最多只能转 20%。这防止了"一步登天"式的策略突变。

**DT 的 RTG 条件生成**：
<!-- ``` -->
> $ P(a_t | s_0, a_0, R_0, ..., s_t, R_t) $
<!-- ``` -->
RTG 就像 GPS 的目的地。你告诉模型"我要到 500 分"，模型就按"500 分级别的驾驶技术"来开车。如果你说"100 分"，它就松松垮垮地开（因为训练数据里 100 分对应的是差策略的行为）。

---

### 理解漏洞

1. **register_buffer vs Parameter**：buffer 不会被梯度更新（比如 causal mask），但会随 model.to(device) 移动设备。Parameter 会参与梯度更新。

2. **为什么 DT 用 state token 的 hidden 预测 action**：因为在 causal mask 下，state token `S_t` 能看到它左边的 `R_t`（目标）但看不到右边的 `A_t`（当前动作还没决定）。所以 `S_t` 的 hidden state 包含了"知道目标 + 知道当前状态"但"不知道要做什么"的信息，正好适合预测动作。

3. **timeout 处理**：CartPole-v1 在 500 步后会 truncate（超时结束），这不是真正的 terminal（杆子没倒）。buffer 中的 `timeout` 标志位和 `terminal_value` 就是为了正确处理这种情况——truncate 时不应该把 V(s_{t+1}) 设为 0，而应该用模型估计的 V 来 bootstrap。

4. **为什么 MPC 用 `1 - done_prob` 作为 reward**：CartPole 的原始 reward 是每活一步 +1。在想象 rollout 中，模型预测的 done_prob 越低，说明这条路越安全（活得越久），所以 `1 - done_prob` 就是"预期的一步奖励"。

5. **世界模型的 state_cost 是什么**：MPC 的评分函数不仅看"活着"，还惩罚状态偏离中心（`x²+θ²`）。这是 reward shaping——鼓励小车停在中间、杆子保持竖直，提升 MPC 的规划质量。

---

## 第五阶段：代码审阅后跳转



### 1️⃣0️⃣ Step 主入口快速跳转

| Step | 主入口 | 训练核心 | 评估/推理核心 |
|------|--------|----------|---------------|
| Step0 GPT 基础验证 | [`scripts/step0_gpt/train_copy_task.py#L85`](scripts/step0_gpt/train_copy_task.py#L85) | [`core/transformer_min.py#L197`](core/transformer_min.py#L197) | [`scripts/step0_gpt/sanity_tests.py#L34`](scripts/step0_gpt/sanity_tests.py#L34) |
| Step1 REINFORCE | [`scripts/step1_reinforce/train_cartpole.py#L20`](scripts/step1_reinforce/train_cartpole.py#L20) | [`rl/reinforce.py#L5`](rl/reinforce.py#L5) | [`models/policy_mlp.py#L42`](models/policy_mlp.py#L42) |
| Step2 PPO (MDP) | [`scripts/step2_ppo/train_cartpole.py#L19`](scripts/step2_ppo/train_cartpole.py#L19) | [`rl/ppo.py#L40`](rl/ppo.py#L40), [`rl/buffer.py#L39`](rl/buffer.py#L39) | [`models/actor_critic_mlp.py#L35`](models/actor_critic_mlp.py#L35) |
| Step3 PPO (POMDP) | [`scripts/step3_ppo_pomdp/train_cartpole.py#L133`](scripts/step3_ppo_pomdp/train_cartpole.py#L133) | [`envs/pomdp_wrappers.py#L14`](envs/pomdp_wrappers.py#L14), [`models/actor_critic_gpt.py#L80`](models/actor_critic_gpt.py#L80) | [`scripts/step3_ppo_pomdp/train_cartpole.py#L86`](scripts/step3_ppo_pomdp/train_cartpole.py#L86) |
| Step4 Decision Transformer | [`scripts/step4_dt/train_cartpole_dt.py#L143`](scripts/step4_dt/train_cartpole_dt.py#L143) | [`offline/dt_dataset.py#L37`](offline/dt_dataset.py#L37), [`offline/decision_transformer.py#L25`](offline/decision_transformer.py#L25) | [`scripts/step4_dt/eval_cartpole_dt.py#L17`](scripts/step4_dt/eval_cartpole_dt.py#L17) |
| Step5 World Model + MPC | [`scripts/step5_world_model/train_world_model.py#L75`](scripts/step5_world_model/train_world_model.py#L75) | [`model_based/dynamics_transformer.py#L23`](model_based/dynamics_transformer.py#L23), [`model_based/mpc.py#L40`](model_based/mpc.py#L40) | [`scripts/step5_world_model/eval_world_model.py#L30`](scripts/step5_world_model/eval_world_model.py#L30), [`scripts/step5_world_model/plan_mpc_cartpole.py#L43`](scripts/step5_world_model/plan_mpc_cartpole.py#L43) |

### 1️⃣1️⃣ 关键概念到实现跳转

| 概念 | 代码位置 |
|------|----------|
| Pre-LN Transformer Block | [`core/transformer_min.py#L147`](core/transformer_min.py#L147) |
| causal mask 真正生效位置 | [`core/transformer_min.py#L108`](core/transformer_min.py#L108) |
| POMDP 三件套（Mask / Delta / History） | [`envs/pomdp_wrappers.py#L14`](envs/pomdp_wrappers.py#L14), [`envs/pomdp_wrappers.py#L31`](envs/pomdp_wrappers.py#L31), [`envs/pomdp_wrappers.py#L70`](envs/pomdp_wrappers.py#L70) |
| PPO ratio + clip | [`rl/ppo.py#L69`](rl/ppo.py#L69), [`rl/ppo.py#L74`](rl/ppo.py#L74) |
| GAE 与 timeout bootstrap | [`rl/buffer.py#L39`](rl/buffer.py#L39), [`rl/buffer.py#L55`](rl/buffer.py#L55) |
| DT 的 3-token 交织 | [`offline/decision_transformer.py#L170`](offline/decision_transformer.py#L170) |
| DT 的 `causal & padding` mask | [`offline/decision_transformer.py#L93`](offline/decision_transformer.py#L93), [`offline/decision_transformer.py#L129`](offline/decision_transformer.py#L129) |
| DT 用 state token 预测动作（`1::3`） | [`offline/decision_transformer.py#L187`](offline/decision_transformer.py#L187), [`offline/decision_transformer.py#L191`](offline/decision_transformer.py#L191) |
| World Model action token 预测 `delta/done` | [`model_based/dynamics_transformer.py#L139`](model_based/dynamics_transformer.py#L139), [`model_based/dynamics_transformer.py#L142`](model_based/dynamics_transformer.py#L142) |
| MPC 评分函数 `(1-done_prob) - state_cost` | [`model_based/mpc.py#L117`](model_based/mpc.py#L117), [`model_based/mpc.py#L118`](model_based/mpc.py#L118) |

### 1️⃣2️⃣ 全量函数/类跳转索引（自动生成）



| 位置 | 符号 |
|------|------|
| [model_based/mpc.py:14](model_based/mpc.py#L14) | class MPCConfig |
| [model_based/mpc.py:25](model_based/mpc.py#L25) | def _pad_right |
| [model_based/mpc.py:40](model_based/mpc.py#L40) | def mpc_action |
| [configs/step0_transformer.py:4](configs/step0_transformer.py#L4) | class Step0Config |
| [rl/buffer.py:5](rl/buffer.py#L5) | class RolloutBuffer |
| [configs/step5_world_model_cartpole_pomdp.py:8](configs/step5_world_model_cartpole_pomdp.py#L8) | class Step5WorldModelConfig |
| [rl/reinforce.py:5](rl/reinforce.py#L5) | def compute_returns |
| [rl/reinforce.py:22](rl/reinforce.py#L22) | def reinforce_loss |
| [offline/decision_transformer.py:17](offline/decision_transformer.py#L17) | class DecisionTransformerBatch |
| [offline/decision_transformer.py:25](offline/decision_transformer.py#L25) | class DecisionTransformer |
| [offline/dt_dataset.py:14](offline/dt_dataset.py#L14) | class Episode |
| [offline/dt_dataset.py:21](offline/dt_dataset.py#L21) | def compute_rtg |
| [offline/dt_dataset.py:37](offline/dt_dataset.py#L37) | class DecisionTransformerDataset |
| [configs/step2_ppo_cartpole.py:5](configs/step2_ppo_cartpole.py#L5) | class PPOCartPoleConfig |
| [configs/step4_dt_cartpole_pomdp.py:6](configs/step4_dt_cartpole_pomdp.py#L6) | class Step4DecisionTransformerConfig |
| [models/policy_mlp.py:17](models/policy_mlp.py#L17) | class CategoricalPolicyMLP |
| [rl/ppo.py:40](rl/ppo.py#L40) | def ppo_update |
| [models/actor_critic_mlp.py:5](models/actor_critic_mlp.py#L5) | class ActorCriticMLP |
| [scripts/step5_world_model/eval_world_model.py:21](scripts/step5_world_model/eval_world_model.py#L21) | def parse_args |
| [scripts/step5_world_model/eval_world_model.py:31](scripts/step5_world_model/eval_world_model.py#L31) | def main |
| [model_based/dynamics_transformer.py:11](model_based/dynamics_transformer.py#L11) | class DynOut |
| [model_based/dynamics_transformer.py:23](model_based/dynamics_transformer.py#L23) | class DynamicsTransformer |
| [scripts/step5_world_model/plan_mpc_cartpole.py:31](scripts/step5_world_model/plan_mpc_cartpole.py#L31) | def parse_args |
| [scripts/step5_world_model/plan_mpc_cartpole.py:43](scripts/step5_world_model/plan_mpc_cartpole.py#L43) | def main |
| [scripts/step5_world_model/run_pipeline.py:12](scripts/step5_world_model/run_pipeline.py#L12) | def _run |
| [scripts/step5_world_model/run_pipeline.py:19](scripts/step5_world_model/run_pipeline.py#L19) | def _module_to_path |
| [scripts/step5_world_model/run_pipeline.py:24](scripts/step5_world_model/run_pipeline.py#L24) | def _write_config |
| [scripts/step5_world_model/run_pipeline.py:32](scripts/step5_world_model/run_pipeline.py#L32) | class Step5WorldModelConfig |
| [scripts/step5_world_model/run_pipeline.py:40](scripts/step5_world_model/run_pipeline.py#L40) | def _get_run_name |
| [scripts/step5_world_model/run_pipeline.py:46](scripts/step5_world_model/run_pipeline.py#L46) | def parse_args |
| [scripts/step5_world_model/run_pipeline.py:74](scripts/step5_world_model/run_pipeline.py#L74) | def main |
| [configs/step5_world_model_cartpole_pomdp_auto.py:6](configs/step5_world_model_cartpole_pomdp_auto.py#L6) | class Step5WorldModelConfig |
| [models/actor_critic_gpt.py:9](models/actor_critic_gpt.py#L9) | class ActorCriticGPT |
| [envs/gym_factory.py:13](envs/gym_factory.py#L13) | class EnvSpec |
| [envs/gym_factory.py:33](envs/gym_factory.py#L33) | def make_env |
| [envs/gym_factory.py:60](envs/gym_factory.py#L60) | def reset_env |
| [envs/gym_factory.py:69](envs/gym_factory.py#L69) | def step_env |
| [envs/gym_factory.py:78](envs/gym_factory.py#L78) | def get_obs_act_dims |
| [model_based/dynamics_dataset.py:12](model_based/dynamics_dataset.py#L12) | class Episode |
| [model_based/dynamics_dataset.py:18](model_based/dynamics_dataset.py#L18) | class DynamicsSequenceDataset |
| [envs/pomdp_wrappers.py:14](envs/pomdp_wrappers.py#L14) | class MaskObsWrapper |
| [envs/pomdp_wrappers.py:31](envs/pomdp_wrappers.py#L31) | class DeltaObsWrapper |
| [envs/pomdp_wrappers.py:70](envs/pomdp_wrappers.py#L70) | class HistoryStackWrapper |
| [scripts/step2_ppo/train_cartpole.py:19](scripts/step2_ppo/train_cartpole.py#L19) | def main |
| [scripts/step5_world_model/train_world_model.py:16](scripts/step5_world_model/train_world_model.py#L16) | def parse_args |
| [scripts/step5_world_model/train_world_model.py:24](scripts/step5_world_model/train_world_model.py#L24) | def evaluate |
| [scripts/step5_world_model/train_world_model.py:75](scripts/step5_world_model/train_world_model.py#L75) | def main |
| [scripts/step0_gpt/train_copy_task.py:56](scripts/step0_gpt/train_copy_task.py#L56) | def make_batch |
| [scripts/step0_gpt/train_copy_task.py:85](scripts/step0_gpt/train_copy_task.py#L85) | def main |
| [configs/step1_reinforce_cartpole.py:4](configs/step1_reinforce_cartpole.py#L4) | class ReinforceCartPoleConfig |
| [core/nn_utils.py:9](core/nn_utils.py#L9) | def seed_everything |
| [core/nn_utils.py:26](core/nn_utils.py#L26) | def get_device |
| [scripts/step0_gpt/sanity_tests.py:13](scripts/step0_gpt/sanity_tests.py#L13) | def test_shape |
| [scripts/step0_gpt/sanity_tests.py:34](scripts/step0_gpt/sanity_tests.py#L34) | def test_causal_mask_no_future_leak |
| [scripts/step0_gpt/sanity_tests.py:88](scripts/step0_gpt/sanity_tests.py#L88) | def test_backward_has_grads |
| [core/transformer_min.py:13](core/transformer_min.py#L13) | class LayerNorm |
| [core/transformer_min.py:50](core/transformer_min.py#L50) | class MultiHeadSelfAttention |
| [core/transformer_min.py:128](core/transformer_min.py#L128) | class FFN |
| [core/transformer_min.py:147](core/transformer_min.py#L147) | class Block |
| [core/transformer_min.py:197](core/transformer_min.py#L197) | class MiniGPT |
| [scripts/step3_ppo_pomdp/train_cartpole.py:21](scripts/step3_ppo_pomdp/train_cartpole.py#L21) | def flatten_obs |
| [scripts/step3_ppo_pomdp/train_cartpole.py:25](scripts/step3_ppo_pomdp/train_cartpole.py#L25) | def _peek_history |
| [scripts/step3_ppo_pomdp/train_cartpole.py:35](scripts/step3_ppo_pomdp/train_cartpole.py#L35) | def debug_pomdp_env |
| [scripts/step3_ppo_pomdp/train_cartpole.py:86](scripts/step3_ppo_pomdp/train_cartpole.py#L86) | def eval_policy |
| [scripts/step3_ppo_pomdp/train_cartpole.py:121](scripts/step3_ppo_pomdp/train_cartpole.py#L121) | def load_cfg |
| [scripts/step3_ppo_pomdp/train_cartpole.py:133](scripts/step3_ppo_pomdp/train_cartpole.py#L133) | def train_once |
| [scripts/step3_ppo_pomdp/train_cartpole.py:362](scripts/step3_ppo_pomdp/train_cartpole.py#L362) | def main |
| [demo_full_pipeline.py:34](demo_full_pipeline.py#L34) | def parse_steps |
| [demo_full_pipeline.py:52](demo_full_pipeline.py#L52) | def flatten_obs |
| [demo_full_pipeline.py:56](demo_full_pipeline.py#L56) | def current_frame |
| [demo_full_pipeline.py:63](demo_full_pipeline.py#L63) | def cleanup_memory |
| [demo_full_pipeline.py:69](demo_full_pipeline.py#L69) | def print_banner |
| [demo_full_pipeline.py:75](demo_full_pipeline.py#L75) | def print_project_overview |
| [demo_full_pipeline.py:95](demo_full_pipeline.py#L95) | def print_unit_legend |
| [demo_full_pipeline.py:104](demo_full_pipeline.py#L104) | def make_copy_batch |
| [demo_full_pipeline.py:118](demo_full_pipeline.py#L118) | def causal_no_future_leak |
| [demo_full_pipeline.py:129](demo_full_pipeline.py#L129) | class PPOTrainResult |
| [demo_full_pipeline.py:138](demo_full_pipeline.py#L138) | def run_ppo_demo |
| [demo_full_pipeline.py:288](demo_full_pipeline.py#L288) | class DemoContext |
| [demo_full_pipeline.py:300](demo_full_pipeline.py#L300) | def run_step0 |
| [demo_full_pipeline.py:355](demo_full_pipeline.py#L355) | def run_step1 |
| [demo_full_pipeline.py:406](demo_full_pipeline.py#L406) | def run_step2 |
| [demo_full_pipeline.py:445](demo_full_pipeline.py#L445) | def run_step3 |
| [demo_full_pipeline.py:541](demo_full_pipeline.py#L541) | def collect_offline_dataset |
| [demo_full_pipeline.py:612](demo_full_pipeline.py#L612) | def evaluate_dt_online |
| [demo_full_pipeline.py:697](demo_full_pipeline.py#L697) | def maybe_load_step3_gpt |
| [demo_full_pipeline.py:716](demo_full_pipeline.py#L716) | def run_step4 |
| [demo_full_pipeline.py:828](demo_full_pipeline.py#L828) | def evaluate_world_model |
| [demo_full_pipeline.py:867](demo_full_pipeline.py#L867) | def evaluate_mpc_online |
| [demo_full_pipeline.py:938](demo_full_pipeline.py#L938) | def run_step5 |
| [demo_full_pipeline.py:1061](demo_full_pipeline.py#L1061) | def print_final_summary |
| [demo_full_pipeline.py:1081](demo_full_pipeline.py#L1081) | def resolve_device |
| [demo_full_pipeline.py:1091](demo_full_pipeline.py#L1091) | def main |
| [core/gpt_decoder_core.py:12](core/gpt_decoder_core.py#L12) | class GPTDecoderCore |
| [scripts/step4_dt/collect_cartpole_dataset.py:30](scripts/step4_dt/collect_cartpole_dataset.py#L30) | def load_cfg |
| [scripts/step4_dt/collect_cartpole_dataset.py:36](scripts/step4_dt/collect_cartpole_dataset.py#L36) | def flatten_obs |
| [scripts/step4_dt/collect_cartpole_dataset.py:43](scripts/step4_dt/collect_cartpole_dataset.py#L43) | def current_frame |
| [scripts/step4_dt/collect_cartpole_dataset.py:53](scripts/step4_dt/collect_cartpole_dataset.py#L53) | def build_policy_model |
| [scripts/step4_dt/collect_cartpole_dataset.py:86](scripts/step4_dt/collect_cartpole_dataset.py#L86) | def main |
| [scripts/step1_reinforce/train_cartpole.py:20](scripts/step1_reinforce/train_cartpole.py#L20) | def main |
| [scripts/step4_dt/eval_cartpole_dt.py:17](scripts/step4_dt/eval_cartpole_dt.py#L17) | def main |
| [configs/step3_ppo_cartpole_pomdp_tr.py:13](configs/step3_ppo_cartpole_pomdp_tr.py#L13) | class PPOCartPolePOMDPTRConfig |
| [scripts/step4_dt/train_cartpole_dt.py:22](scripts/step4_dt/train_cartpole_dt.py#L22) | def load_cfg |
| [scripts/step4_dt/train_cartpole_dt.py:29](scripts/step4_dt/train_cartpole_dt.py#L29) | def evaluate_dt |
| [scripts/step4_dt/train_cartpole_dt.py:143](scripts/step4_dt/train_cartpole_dt.py#L143) | def main |
