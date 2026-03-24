# `demo_full_pipeline.py` 结果解读



```bash
python demo_full_pipeline.py --steps 0,1,2,3,4,5 --device cpu
```
明白：
- 每个 Step 在做什么
- 指标是什么意思
- 为什么会出现看到的数值
- 各 Step 之间是如何互相影响的

---

## 1. 先给结论

这次结果的主线是：

1. **Step0/1/2 正常**：模型从“不会”到“会一些”趋势明显。  
2. **Step3 有提升但不够强**：GPT 比 MLP 好，说明“记忆”确实有用，但训练预算短、模型还不稳。  
3. **Step4 明显掉下去**：离线数据质量差（大部分 episode 很短，回报很低），DT 学到了“差策略”。  
4. **Step5 指标看起来不错但回报不高**：世界模型在“单步预测”上可以，但用 MPC 连续规划时误差累积，最终分数一般。

一句话总结：  
**前半段（在线学习）是“能学到东西”的；后半段（离线/模型式）受数据质量强烈制约，这正是该 pipeline 的教育价值。**

---

## 2. 单位

日志里的 4 个单位不一样，不能混着比：

- `step`：1 次监督学习梯度更新（Step0）
- `ep`：1 局完整游戏（Step1）
- `update`：1 次 PPO 大循环 = 采样一批数据 + 多轮优化（Step2/3）
- `iter`：1 次离线训练梯度更新（Step4/5）

这就是为什么：
- Step0 看 `loss/acc`
- Step1/2/3 看 `avg_return`
- Step4/5 同时看 `loss` 或预测误差 + `avg_return`

---

## 3. 全流程设计图

1. Step0：先证明 Transformer 的“看历史不看未来”机制是对的。  
2. Step1/2：在标准 CartPole 上学策略（从 REINFORCE 到 PPO）。  
3. Step3：故意把环境变成 POMDP，比较“没记忆的 MLP” vs “有记忆的 GPT”。  
4. Step4：拿 Step3 策略收集离线数据，训练 Decision Transformer。  
5. Step5：用同一批离线数据训练世界模型，再用 MPC 做规划控制。

所以后面步骤会继承前面的优缺点，尤其是**数据质量会层层传递**。

---

## 3.1 数据来源与数据组成（全局账本）

这一节专门回答：每个阶段“数据从哪来、长什么样”。

### Step0（监督学习玩具任务）
- 数据来源：脚本在线随机生成，不来自环境，不依赖文件。  
- 数据组成：  
  - `x`: 随机 token 序列，形状约 `[B, T]`  
  - `target`: 同形状 `[B, T]`，前 `K` 位为 `-100`（忽略），后面是“滞后 K 步”的标签  
- 作用：验证 Transformer 的 causal 机制是否工作。

### Step1（REINFORCE 在线学习）
- 数据来源：与 CartPole 环境实时交互（on-policy）。  
- 数据组成（按每个 episode）：
  - `obs_t`: 当前状态（全观测，4维）  
  - `action_t`: 离散动作（0/1）  
  - `reward_t`: 每步奖励（CartPole 通常每步 +1）  
  - `log_prob_t`: 策略当时选该动作的对数概率  
  - 局末计算 `returns(G_t)` 作为学习信号  
- 作用：证明最基础策略梯度能“从回报学习”。

### Step2（PPO 在线学习，MDP）
- 数据来源：同样是实时环境交互（on-policy）。  
- 数据组成（按每个 update）：
  - 先采样 `rollout_steps=1024` 条 transition  
  - 放进 `RolloutBuffer`：`obs/actions/logp/reward/done/value`  
  - 再计算 `GAE advantages` 和 `returns`  
- 作用：用更稳定的 PPO 训练流程替代 REINFORCE。

### Step3（POMDP 对比：MLP vs GPT）
- 数据来源：实时交互，但环境被人为“降维+加历史”。  
- POMDP 配置：
  - `keep_idx=(0,2)` 只保留 `x, theta`
  - GPT 分支额外 `use_delta_obs=True`
  - GPT 分支 `history_len=4`（有短记忆）
- 数据组成：
  - MLP 分支：当前单帧信息（T=1，低维）  
  - GPT 分支：历史窗口信息（T=4）+ 差分特征（delta）  
  - 直观形状（GPT 分支）：单步是 `[x, theta, dx, dtheta]`（约4维），历史堆叠后近似 `[4,4]`，再 flatten 给 PPO  
  - 每个 update 采样 `512` 步写入 PPO buffer  
- 作用：验证“序列记忆”在 POMDP 下的价值。

### Step4（Decision Transformer 离线学习）
- 数据来源：先用 Step3 策略（或随机策略）采集离线轨迹，再存成 `.npz`。  
- 原始 `.npz` 字段：
  - `obs`、`actions`、`rewards`、`dones`、`episode_ends`、`act_dim`
  - 这次运行是 `50 episodes / 679 steps`，所以可理解为 `obs` 约 `[679, D]`、`actions` 约 `[679]`
- 训练样本组成（窗口化后）：
  - `states[B,K,D]`（z-score 归一化）  
  - `actions[B,K]`  
  - `rtg[B,K]`（return-to-go，且除以 `rtg_scale`）  
  - `timesteps[B,K]`  
  - `valid[B,K]`（padding mask）  
- 作用：把 RL 变成“离线序列建模+动作模仿”。

### Step5（世界模型 + MPC）
- 数据来源：复用 Step4 同一离线 `.npz`。  
- 世界模型训练样本组成：
  - `states/actions/timesteps/valid`
  - `delta_targets`（归一化空间的 `s_{t+1}-s_t`）
  - `done_targets`
  - `trans_valid`（哪些位置真的有下一帧）  
  - 数据切分方式：按 episode 切 train/val，避免同一局轨迹泄漏到验证集  
- MPC 在线输入组成：
  - `obs_hist_raw`（原始观测历史）
  - `act_hist`（动作历史）
  - `t_hist`（时间步历史）
  - 以及训练集统计量 `state_mean/std` 做同分布归一化  
- 作用：先学“世界怎么变”，再用规划算法选动作。

---

## 3.2 数据流-ASCII 时序图

下面这张图可以直接在讲解时展示，按箭头从左到右讲：

```text
Step0 (监督玩具任务)
随机token x[B,T] + lag标签target[B,T]
          |
          v
MiniGPT -> loss/acc -> 验证causal mask


Step1 (在线RL: REINFORCE)
CartPole env --(obs,action,reward,log_prob 按episode记录)-->
trajectory list --(计算returns G_t)--> reinforce update
          |
          v
avg_return(每20局统计)


Step2 (在线RL: PPO, MDP)
CartPole env --(rollout_steps=1024)-->
RolloutBuffer{obs,act,logp,reward,done,value}
          |
          +--> compute GAE/returns
          |
          +--> ppo_update(6轮优化)
          v
avg_return(按update统计)


Step3 (在线RL: PPO, POMDP对比)
CartPole env
   |-- MLP支路: keep_idx=(0,2), T=1
   |-- GPT支路: keep_idx=(0,2) + delta + history(T=4)
          |
          v
两套PPO训练 -> 对比avg_return(MLP vs GPT)
          |
          v
保存Step3 GPT策略: /tmp/trl_demo/step3_gpt_policy.pt


Step4 (离线RL: Decision Transformer)
Step3 GPT策略(或随机策略)采样50局
          |
          v
step4_offline_data.npz
{obs, actions, rewards, dones, episode_ends, act_dim}
          |
          v
DecisionTransformerDataset
  -> 按episode切分
  -> 计算RTG并缩放
  -> state归一化
  -> 切窗口K并构造valid mask
          |
          v
DT训练(iter) -> loss下降
          |
          v
在线eval avg_return


Step5 (世界模型 + MPC)
复用step4_offline_data.npz
          |
          v
DynamicsSequenceDataset(train/val episode级切分)
{states,actions,timesteps,valid,delta_targets,done_targets,trans_valid}
          |
          v
DynamicsTransformer训练(iter)
  -> delta_mse
  -> done_acc
          |
          v
MPC在线规划
输入: obs_hist_raw + act_hist + t_hist + state_mean/std
输出: mpc_action -> 与真实env交互 -> MPC avg_return
```


- Step0~3 是“在线拿数据边学边改策略”。
- Step4~5 是“先固定离线数据，再做离线训练与规划”。
- Step4 和 Step5 的上限都被 Step3 采出来的数据质量约束。

---

## 3.3 失败链路图-为什么这次 Step4/5 低分

这张图专门解释这次结果里“后半段分数不高”的因果关系：

```text
Step3 GPT(最终约39.7) 还不够稳
            |
            v
用于采样Step4离线数据时，很多episode很短（大量9~13分）
数据规模也偏小（50 episodes / 679 steps）
            |
            v
Step4 DT 训练loss下降（学会模仿）
但模仿对象主要是“短失败轨迹”
            |
            v
Step4 在线eval avg_return仅10.6
            |
            v
Step5 复用同一批数据训练世界模型
单步指标看起来不错（delta_mse下降, done_acc较高）
            |
            v
MPC做多步规划时误差累积 + 数据分布窄
            |
            v
Step5 MPC avg_return约20.3（明显低于Step2/Step3）
```

解释：
- 所以不是 Step4/Step5 “代码坏了”，而是“数据老师不够强”，后半段忠实继承了前面采样数据的上限。


- 这次后半段分数低，根因不是模型没学，而是离线数据主要来自短失败轨迹。  
- DT 学会了模仿这些轨迹，世界模型也在同一分布上训练，所以 MPC 最终回报被卡住。  
- 这正好说明了这条 pipeline 的核心工程事实：数据质量决定后续上限。

---

## 4. 逐步解读真实结果

---

### Step0：最小 GPT（Copy Task）

数据来源与组成：
- 来源：脚本随机造数据，不依赖外部文件/环境。  
- 输入：随机 token 序列 `x`。  
- 标签：`target[t]=x[t-K]`（前 `K` 个位置忽略）。  
- 为什么这样设计：只测“时序依赖和因果遮罩”，不掺杂环境噪声。

日志摘录：
- step100: loss=3.0882, acc=42.1%
- step200: loss=0.1266, acc=100%
- step500: loss=0.0129, acc=100%
- causal 前缀一致性 `max_diff=0`

怎么理解：

1. `loss` 快速下降、`acc` 到 100%，说明模型学会了这个玩具任务。  
2. `max_diff=0` 非常关键：改未来 token 不会影响过去位置输出，说明 causal mask 正常。  
3. 这是“结构正确性验证”，不是 RL 强弱对比。

为什么这么快：
- 任务规则非常固定（预测 K 步前 token），难度比真实决策问题低很多。

---

### Step1：REINFORCE

数据来源与组成：
- 来源：在线与 CartPole 环境交互。  
- 每局记录：`(obs, action, reward, log_prob)` 序列。  
- 局末加工：把 `reward` 序列反向累加成 `returns(G_t)`。  
- 为什么这样设计：REINFORCE 天然按“整局回报”更新。

日志摘录（每 20 局平均）：
- 29.8 → 63.5 → 152.6 → 95.5 → 236.8

怎么理解：

1. 总体是上升的，说明策略梯度在工作。  
2. 中间出现回落（152.6 到 95.5）是正常现象，不是“坏了”。  

为什么会抖：
- REINFORCE 是高方差算法，更新信号来自整局回报，噪声大。  
- 它能学，但稳定性通常不如 PPO。

---

### Step2：PPO（MDP，全观测）

数据来源与组成：
- 来源：在线交互（全观测 MDP）。  
- 每个 `update`：
  - 先采样 `1024` 步到 `RolloutBuffer`  
  - 再算 `advantage/return`  
  - 再做 `6` 轮 mini-batch 更新  
- 为什么这样设计：PPO 用“批量采样+多轮复用”提升样本效率与稳定性。

最终：
- 从 17.1 附近逐步爬到 110.0（中间有波动）

怎么理解：

1. 比 Step1 更“工程化”：有 value baseline、GAE、clip，训练更稳。  
2. 这里是**有限更新预算 + CPU 快速演示配置**，能到 110 已经证明“跑通并显著提升”。  
3. 没到 500 不奇怪，因为这不是收敛训练脚本，是课堂演示脚本。

---

### Step3：PPO + POMDP（MLP vs GPT）

数据来源与组成：
- 来源：在线交互，但观测经过 POMDP wrapper。  
- MLP 分支数据：
  - 只看当前帧（`T=1`），信息不完整。  
- GPT 分支数据：
  - `keep_idx=(0,2)` 后再加 delta，再堆历史 `T=4`。  
  - 实际上输入是“短时序窗口”，不是单帧（可理解为每次喂最近 4 帧）。  
- 为什么这样设计：故意制造“当前观测不够用”的场景，看记忆模块是否有优势。

关键结果：
- MLP(T=1) 最终 `19.0`
- GPT(T=4+delta) 最终 `39.7`

怎么理解：

1. 这是本项目最核心对比：  
   - MLP 只看当前帧（T=1），在 POMDP 下信息不完整，容易“盲走”。  
   - GPT 看历史（T=4）+ delta 特征，能部分恢复速度信息。  
2. GPT 明显高于 MLP，说明“序列记忆”有效。  
3. 但 GPT 也不高（39.7），说明当前训练预算还偏短、模型还未充分学好。

---

### Step4：Decision Transformer（离线 RL）

数据来源与组成：
- 来源：Step3 策略采集 50 局，保存为 `step4_offline_data.npz`。  
- 文件字段：
  - `obs/actions/rewards/dones/episode_ends/act_dim`  
  - 这次的真实规模：`N=679`（总步数），`E=50`（episode 数）  
- 训练前加工：
  - 按 `episode_ends` 切回每局轨迹  
  - 计算 `RTG` 并缩放（`rtg_scale=500`）  
  - 对状态做 z-score 归一化  
  - 切成长度 `K` 的窗口样本并加 `valid` mask  
- 为什么这样设计：DT 是“给定目标回报 + 历史状态动作”的条件序列模型。

最需要解释清楚的一步：

日志核心：
- 数据来源：Step3 GPT policy
- 采集 50 episodes，共 679 steps（平均每局约 13.6 步，非常短）
- 训练 loss：0.0932 → 0.0579 → 0.0304（看起来很好）
- 在线评估：`avg_return = 10.6`（很差）

为什么“loss 很好但 return 很差”？

这是离线 RL 的典型现象，原因是：

1. **数据质量差**  
   - 大多数采集 episode 回报是 9~13，说明演示数据里主要是“很快失败”的轨迹。  
   - 模型学得再好，也是在模仿“差老师”。  

2. **loss 只代表“拟合数据”**  
   - 它回答的是“我像不像数据里的动作”；  
   - 不是“我在真实环境里能不能拿高分”。  

3. **分布偏窄**  
   - 数据几乎都在失败附近，模型没见过长期稳定平衡的状态。  

所以 Step4 结果非常合理：  
**它不是训练坏了，而是“忠实地学会了坏数据”。**

---

### Step5：世界模型 + MPC

数据来源与组成：
- 来源：复用 Step4 的同一离线数据。  
- 世界模型训练目标：
  - 预测 `delta_targets = next_state_norm - state_norm`  
  - 预测 `done_targets`  
- MPC 在线规划输入：
  - 最近一段真实观测历史 + 动作历史 + 时间步历史  
  - 先按训练集 `mean/std` 归一化后再喂给世界模型  
- 为什么这样设计：让规划阶段看到的数值分布与训练阶段一致，减少分布偏移。

日志核心：
- 用 Step4 同一数据
- `delta_mse`: 0.0189 → 0.0140 → 0.0100（下降）
- `done_acc`: 0.910 → 0.935 → 0.928（较高）
- MPC avg_return: 20.3（不高）

为什么预测指标不错，但 MPC 分数一般？

1. **单步预测 vs 多步规划不是一回事**  
   - `delta_mse/done_acc` 是“单步拟合”指标。  
   - MPC 需要在模型里连续滚动很多步（horizon=10），误差会累积放大。  

2. **训练数据仍然偏差**  
   - 数据大多是短失败轨迹，模型对“高回报长轨迹区域”理解不够。  

3. **MPC 本身是近似解**  
   - `num_samples=256, horizon=10` 是为了加速演示，不是极致性能配置。  

因此 Step5 的结果也是符合预期的：  
**模型学到一些动力学规律，但规划性能被数据分布和误差累积限制。**

---

## 5. 为什么 Step3 明明有 39.7，Step4 收集却大多 10~12？

这是答辩时很可能被问到的点。

可能原因（都合理）：

1. Step3 打印的是训练过程里的滑动平均，不等于“稳定最佳策略”。  
2. demo 脚本保存的是 Step3 最后模型，不一定是最佳 checkpoint。  
3. 策略一旦偏向某个坏动作，离线采样会迅速退化成“短失败数据”。  
4. 在 POMDP 下，策略对初始化和噪声较敏感。

这不矛盾，反而说明了工程真实情况：  
**离线链路非常依赖“收集策略是否真的强且稳定”。**

---

## 6. 这次结果背后的“因果链”



1. Step3 策略还不够强。  
2. 用它去采数据，得到很多低质量轨迹。  
3. Step4 忠实模仿了这些低质量轨迹，所以在线表现低。  
4. Step5 用同一批低质量数据训练世界模型，单步拟合可以，但 MPC 的长期决策仍被限制。  

所以：  
**后半段不是算法失效，而是被前面的数据质量卡住了。**

---

## 7. 这组结果“说明了什么设计思想”

它非常适合教学，因为完整展示了：

1. 从“结构正确性”(Step0) 到 “在线策略学习”(Step1/2/3)；  
2. 再到“离线模仿”(Step4) 与 “模型预测控制”(Step5)；  
3. 并且真实体现了一个工业事实：**数据比模型更重要**。

---

## 8. 简明解读


> 这次实验前半段是成功的：Step0 验证了 causal attention，Step1/2 展示了在线 RL 回报上升，Step3 证明在 POMDP 下 GPT 的时序记忆优于 MLP。  
> 后半段回报低不是代码错，而是数据链路问题：Step3 用于采集离线数据的策略不够强，导致 Step4 学到的是低质量行为，Step5 也继承了同一数据分布。  
> Step5 的 delta_mse 和 done_acc 说明世界模型在单步上学到了规律，但 MPC 多步规划会放大误差，所以在线回报仍有限。  
> 整体上，这次结果完整展示了“从在线学习到离线学习再到模型式规划”的可运行闭环，以及数据质量对后续环节的决定性影响。

---

## 9. 后续改进

1. **先把 Step3 策略练强再采集**  
   - 增加 Step3 updates 或保存/加载 best checkpoint 采集。  

2. **提高离线数据覆盖面**  
   - 混合策略采样（强策略 + 少量随机），避免数据过窄。  

3. **Step4 增加评估频率并保留最佳模型**  
   - 不只看 loss，重点看在线 return。  

4. **Step5 增强 MPC 搜索预算**  
   - 适当增 `num_samples`、`horizon`，并可做 warm-start。  

---

## 10. 最后一句话

这次不是“做失败了”，而是**把真实机器学习系统里最重要的问题演示出来了**：  
**模型会忠实学习它看到的数据；数据质量决定上限。**
