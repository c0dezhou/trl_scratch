# TRL Scratch 项目总览

本项目是一个循序渐进的强化学习与序列建模实验集，围绕 **CartPole-v1** 展开：
从最小化 Transformer（Step0）到 REINFORCE（Step1）、PPO（Step2）、POMDP+Transformer 的 PPO（Step3）、
离线 Decision Transformer（Step4），再到世界模型 + MPC（Step5）。

核心目标：
- 用尽可能少的代码复现 RL 与序列建模的关键思想。
- 形成一条“数据收集 → 离线训练 → 在线评估/规划”的可复现流水线。

---

## 目录结构一览

- `configs/`：各 Step 的配置文件（超参、数据路径、模型结构）。
- `core/`：最小化 Transformer 组件、通用工具（seed / device）。
- `envs/`：POMDP wrapper（观测遮挡、历史堆叠、delta 特征）和环境工厂。
- `models/`：Actor-Critic 的 MLP / GPT 版本。
- `offline/`：Decision Transformer 模型与离线数据集处理。
- `model_based/`：世界模型（Dynamics Transformer）与 MPC。
- `rl/`：REINFORCE / PPO 核心算法与 Rollout Buffer。
- `scripts/`：各 Step 训练、评估、数据收集脚本。
- `data/`：离线数据集（`.npz`）。
- `checkpoints/`：训练产出的模型权重。

---

## Step-by-Step 逻辑主线

### Step0: 最小 GPT（序列任务）
- 位置：`scripts/step0_gpt/train_copy_task.py`
- 任务：预测 “K 步之前的 token”。验证 causal attention 的基本能力。
- 配置：`configs/step0_transformer.py`

### Step1: REINFORCE
- 位置：`scripts/step1_reinforce/train_cartpole.py`
- 模型：`models/policy_mlp.py`（简单策略网络）
- 算法：`rl/reinforce.py`

### Step2: PPO（全观测 MDP）
- 位置：`scripts/step2_ppo/train_cartpole.py`
- 模型：`models/actor_critic_mlp.py`
- 算法：`rl/ppo.py` + `rl/buffer.py`
- 配置：`configs/step2_ppo_cartpole.py`

### Step3: POMDP + PPO + Transformer/MLP
- 位置：`scripts/step3_ppo_pomdp/train_cartpole.py`
- 环境：`envs/pomdp_wrappers.py`（观测遮挡 + history stack + delta）
- 模型：
  - MLP：`models/actor_critic_mlp.py`
  - GPT：`models/actor_critic_gpt.py` + `core/gpt_decoder_core.py`
- 配置：
  - `configs/step3_ppo_cartpole_pomdp_tr.py`（GPT 默认）
  - `configs/step3_ppo_cartpole_pomdp_mlp_t1.py`
  - `configs/step3_ppo_cartpole_pomdp_mlp_t32.py`

### Step4: Decision Transformer（离线）
- 数据采集：`scripts/step4_dt/collect_cartpole_dataset.py`
- 数据集处理：`offline/dt_dataset.py`
- 模型：`offline/decision_transformer.py`
- 训练：`scripts/step4_dt/train_cartpole_dt.py`
- 评估：`scripts/step4_dt/eval_cartpole_dt.py`

### Step5: 世界模型 + MPC（Model-Based）
- 数据集：`model_based/dynamics_dataset.py`
- 模型：`model_based/dynamics_transformer.py`
- MPC：`model_based/mpc.py` + `scripts/step5_world_model/plan_mpc_cartpole.py`
- 训练/评估：`scripts/step5_world_model/train_world_model.py` / `eval_world_model.py`
- 一键管线：`scripts/step5_world_model/run_pipeline.py`

---

## 环境与数据

### POMDP 观测设计
- 默认只保留 CartPole 的 `[x, theta]`（位置与角度），速度被屏蔽。
- `DeltaObsWrapper` 会扩展为 `[obs, obs_t - obs_{t-1}]`。
- `HistoryStackWrapper` 会把最近 T 步拼成序列。

### 离线数据格式（`.npz`）
数据采集脚本会保存如下字段（用于 Step4 / Step5）：
- `obs`: `[N, state_dim]`
- `actions`: `[N]`
- `rewards`: `[N]`
- `dones`: `[N]`
- `episode_ends`: `[E]`（每个 episode 的结束索引，不含 end）
- `act_dim`: 标量动作空间大小

---

## 复现指南（按 Step）

下面命令默认在项目根目录执行。

### 1) Step0：最小 GPT 任务
```
python -m scripts.step0_gpt.train_copy_task
```

### 2) Step1：REINFORCE
```
python -m scripts.step1_reinforce.train_cartpole
```

### 3) Step2：PPO（MDP）
```
python -m scripts.step2_ppo.train_cartpole
```

### 4) Step3：POMDP + PPO（GPT/MLP）
```
# GPT 版本（默认配置）
python -m scripts.step3_ppo_pomdp.train_cartpole --configs configs.step3_ppo_cartpole_pomdp_tr

# MLP 版本（示例）
python -m scripts.step3_ppo_pomdp.train_cartpole --configs configs.step3_ppo_cartpole_pomdp_mlp_t32
```

### 5) Step4：离线 Decision Transformer

#### 5.1 采集离线数据
```
python -m scripts.step4_dt.collect_cartpole_dataset \
  --policy_config configs.step3_ppo_cartpole_pomdp_tr \
  --policy_ckpt best_step3_pomdp_gpt_t4_delta.pt \
  --out data/cartpole_pomdp_from_step3_full.npz \
  --episodes 200
```

#### 5.2 训练 Decision Transformer
```
python -m scripts.step4_dt.train_cartpole_dt --config configs.step4_dt_cartpole_pomdp
```

#### 5.3 独立评估（加载 ckpt）
```
python -m scripts.step4_dt.eval_cartpole_dt \
  --ckpt checkpoints/step4_dt_cartpole_pomdp_best.pt
```

### 6) Step5：世界模型 + MPC

#### 6.1 一键管线（收集 + 训练 + eval + MPC）
```
python -m scripts.step5_world_model.run_pipeline \
  --collect --train --eval --mpc \
  --mode mixed \
  --policy_config configs.step3_ppo_cartpole_pomdp_tr \
  --policy_ckpt best_step3_pomdp_gpt_t4_delta.pt \
  --episodes 500 \
  --out data/cartpole_pomdp_from_step3_mixed.npz \
  --write_config
```

#### 6.2 单独训练 / 评估 / MPC
```
# 训练世界模型
python -m scripts.step5_world_model.train_world_model \
  --config configs.step5_world_model_cartpole_pomdp

# 离线预测误差评估
python -m scripts.step5_world_model.eval_world_model \
  --ckpt checkpoints/step5_wm_cartpole_pomdp_best.pt

# MPC 在线评估
python -m scripts.step5_world_model.plan_mpc_cartpole \
  --ckpt checkpoints/step5_wm_cartpole_pomdp_best.pt \
  --episodes 10
```

---

## 预期训练曲线 / 指标 与当前结果（示例）

说明：以下“预期曲线/指标”描述正常训练应看到的趋势；“当前结果”来自本机对仓库内已有 ckpt 的评估（随机种子/设备不同会有差异）。

### Step0（最小 GPT 任务）
- 预期曲线/指标：`loss` 下降、`acc` 上升；在 2000 steps 内常见 `acc > 95%`（只统计 t>=K 的有效位）。
- 当前结果：本仓库未保存 Step0 权重/日志，需运行 `scripts/step0_gpt/train_copy_task.py` 以生成结果。

### Step1（REINFORCE）
- 预期曲线/指标：`avg_return`（每 20 eps 打印）噪声大但总体上升，可能从 20~50 提升到 200+（不稳定）。
- 当前结果：Step1 不保存 ckpt，训练日志即最终结果。

### Step2（PPO, MDP）
- 预期曲线/指标：`avg_return(last100)` 逐步上升，达到 ≥475 且连续多次即 “Solved”。训练脚本每 update 打印该指标。
- 当前结果：`best.pt` 贪婪评估 20 episodes，`avg_return=500.0`，`std=0.0`。

### Step3（POMDP + PPO + Transformer/MLP）
- 预期曲线/指标：训练前期上升更慢，但 greedy eval 通常可达 475+；带 `delta_obs` 与较短 history 更稳定。
- 当前结果：`best_step3_pomdp_gpt_t4_delta.pt` 贪婪评估 20 episodes，`avg_return=500.0`，`std=0.0`。

### Step4（Decision Transformer）
- 预期曲线/指标：`loss` 下降；在线评估 `avg_return` 上升，`target_return=500` 时可逼近满分。
- 当前结果：`checkpoints/step4_dt_cartpole_pomdp_best.pt` 评估 20 episodes，`avg_return=500.0`。

### Step5（世界模型 + MPC）
- 预期曲线/指标：离线 `delta_mse` 下降、`done_acc` 上升趋近 1.0；MPC 回报随模型质量与采样数增加而提高。
- 当前结果：
  - 离线评估：`delta_mse(norm)=0.004263`，`done_acc=0.9965`，`delta_mse(raw approx)=0.000027`。
  - MPC 评估（`--num_samples 512 --horizon 20`，5 episodes）：`avg_return=82.8`。
  - 说明：默认 `num_samples=1024, horizon=25` 更慢；回报对采样数、规划长度和模型误差敏感。

---

## 已存在的数据与模型（本仓库内）

- 数据集（`data/`）：
  - `cartpole_pomdp_from_step3_full.npz`
  - `cartpole_pomdp_from_step3_mixed.npz`
  - `cartpole_pomdp_from_step3_debug.npz`

- Checkpoints（`checkpoints/`）：
  - `step4_dt_cartpole_pomdp_best.pt`
  - `step5_wm_cartpole_pomdp_best.pt`
  - `step5_wm_cartpole_pomdp_last.pt`

- Step3 PPO 预训练权重（根目录）：
  - `best_step3_pomdp_gpt_t4_delta.pt`
  - `best_step3_pomdp_gpt_t8.pt`
  - `best_step3_pomdp_gpt_t32.pt`
  - `best_step3_pomdp_mlp_t1.pt`
  - `best_step3_pomdp_mlp_t32.pt`

---

## 关键实现要点（复现实验时容易踩坑的地方）

- **POMDP 一致性**：Step3/4/5 若使用 `use_delta_obs=True`，请确保数据采集与模型训练/评估一致。
- **归一化**：Decision Transformer 与世界模型都对 state 做了 z-score 归一化，推理时必须使用同一组均值方差（已保存在 ckpt 中）。
- **时间步嵌入**：DT 和世界模型都依赖 `timesteps`，因此序列拼接/填充方式必须与训练时一致。
- **MPC 依赖世界模型质量**：MPC 的回报极其依赖 dynamics 预测的稳定性，训练数据分布与规划分布尽量匹配。

---

## 最小依赖（运行前）

项目未提供 requirements 文件，实际运行需至少包含：
- Python 3.x
- PyTorch
- gymnasium
- numpy

可按常规方式自行安装。

---

## 推荐复现实验顺序（最省力）

1) Step3：训练一个 POMDP PPO（或直接用已给的 ckpt）
2) Step4：采集数据 → 训练 DT → eval
3) Step5：用同一批数据训练世界模型 → MPC 在线评估

---

如需我补充：
- 更完整的依赖列表（按你本机环境冻结）
- 训练曲线的可视化脚本/日志解析
- 将本文件改成 README 主文档
告诉我即可。
