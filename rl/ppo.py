"""
1.加一个 Critic(V)：学V(s)当 baseline，显著降方差
2.用 GAE(λ) 计算 advantage：更稳、更常用
3.实现 PPO clipped objective：工业最常见策略优化
4.训练循环变成：rollout batch + update epochs + minibatch
"""