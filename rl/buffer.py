# 先收集一批数据（Rollout），进行复杂的处理（计算 GAE），
# 然后反复多次利用这些数据进行训练。
import torch

class RolloutBuffer:
    """存一段rollout(T步)用于ppo更新"""
    def __init__(self, T: int, obs_dim: int, device: torch.device):
        self.T = T
        self.device = device
        # 预分配内存：
        # 在训练循环中动态地往 list 里 append 再转 tensor 会非常慢。
        # 预先在显存中开辟好一块空间，每次只需要通过 self.ptr 指针填入数据，速度极快。
        self.obs = torch.zeros((T, obs_dim),device=device)
        self.actions = torch.zeros((T, ),dtype=torch.long, device=device)
        self.logp = torch.zeros((T,),device=device)
        self.rewards = torch.zeros((T,),device=device)
        self.dones = torch.zeros((T,),device=device) # 1 if done else 0
        self.values = torch.zeros((T,),device=device)
        self.timeouts = torch.zeros((T,), device=device)
        self.terminal_values = torch.zeros((T,), device=device)

        self.advantages = torch.zeros((T,), device=device)
        self.returns = torch.zeros((T,), device=device)

        self.ptr = 0

    def add(self, obs, action, logp, reward, done, value, timeout=False, terminal_value=0.0):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logp[i] = logp
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.timeouts[i] = float(timeout)
        self.terminal_values[i] = terminal_value
        self.ptr += 1
    
    def compute_gae(self, last_value:torch.Tensor, gamma: float, lam: float):
        """
        GAE:（广义优势估计）
        通过参数lam把Gt（reinforce）和V(s)(critic)揉在一起
        TD error: delta_t = r_t + gamma * V_{st+1} * (1-done) - V_st
        Advantage: A_t = delta_t + gamma*lam*(1-done)*A_{t+1}  通过reverse实现累加
        return_t = A_t + V_t
        """
        T = self.T
        adv = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T-1 else self.values[t+1]
            # nonterminal = 1.0 - self.dones[t] 如果这一步游戏结束了（done=True），那么下一时刻的 Value 就不应该被计入。
            # 这个开关确保了奖励计算不会跨越两个不同的游戏局
            nonterminal = 1.0 - self.dones[t]
            reward = self.rewards[t]
            if self.timeouts[t].item() > 0:
                reward = reward + gamma * self.terminal_values[t]
                nonterminal = 0.0
            delta = reward + gamma * next_value * nonterminal - self.values[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.advantages[t] = adv
        
        self.returns = self.advantages + self.values

        # 标准化 adv(稳定训练，常用)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # 这是 PPO 区别于 A2C/REINFORCE 的物理标志
    def get_minibatches(self, minibatch_size: int):
        idx = torch.randperm(self.T, device=self.device) # 打乱顺序 (randperm): 我们把收到的时间序列数据全部打乱。因为神经网络在训练时，
        # 如果数据太连续（比如全是杆子往左倒的画面），容易导致过拟合。随机打乱让训练更均匀
        for start in range(0, self.T, minibatch_size):
            mb = idx[start:start+minibatch_size]
            yield(
                self.obs[mb],
                self.actions[mb],
                self.logp[mb],
                self.advantages[mb],
                self.returns[mb],
                self.values[mb],
            )
        """
        暂停-苏醒机制：
        因为在 PPO 训练的外层循环里，我们接住这批 obs, actions 后，要进行复杂的 loss.backward() 和 optimizer.step()。这些操作很费时间。
        如果用 return，程序会先一口气把所有的几百个 Minibatch 切片全算好塞进列表（浪费内存和时间）。 
        用 yield，程序是：
            切一小块数据（Yield 出去）。
            函数停住。
            外面拿这块数据去更新网络（Backward, Step）。
            更新完了，外层循环回来找函数要下一块。
            函数苏醒，再切下一小块。
        这就是“按需取货”！ 它是为了让 GPU 训练和数据准备能够节奏一致地配合，而不是先把仓库塞爆再开工。
        """
