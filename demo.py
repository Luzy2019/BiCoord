import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 超参数
dim = 2         # 数据维度（2D点）
num_samples = 1000
num_steps = 100  # ODE求解步数
lr = 1e-3
epochs = 5000

# 目标分布：正弦曲线上的点（x1坐标）
x1_samples = torch.rand(num_samples, 1) * 4 * torch.pi  # 0到4π
y1_samples = torch.sin(x1_samples)                      # y=sin(x)
target_data = torch.cat([x1_samples, y1_samples], dim=1)

# 噪声分布：高斯噪声（x0坐标）
noise_data = torch.randn(num_samples, dim) * 2

class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # 输入维度: x (2) + t (1) = 3
            nn.ReLU(),
            nn.Linear(64, dim)
        )
  
    def forward(self, x, t):
        # 直接拼接x和t（t的形状需为(batch_size, 1)）
        return self.net(torch.cat([x, t], dim=1))
        
model = VectorField()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # 随机采样噪声点和目标点
    idx = torch.randperm(num_samples)
    x0 = noise_data[idx]  # 起点：噪声
    x1 = target_data[idx] # 终点：正弦曲线

    # 时间t的形状为 (batch_size, 1)
    t = torch.rand(x0.size(0), 1)  # 例如：shape (1000, 1)
  
    # 线性插值生成中间点
    xt = (1 - t) * x0 + t * x1
  
    # 模型预测向量场（直接传入t，无需squeeze）
    vt_pred = model(xt, t)  # t的维度保持不变
  
    # 目标向量场：x1 - x0
    vt_target = x1 - x0
  
    # 损失函数
    loss = torch.mean((vt_pred - vt_target)**2)
  
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = noise_data.clone()  # 用全部噪声点做动画
trajectory = [x.detach().numpy()]

# 数值求解ODE（欧拉法）
t = 0
delta_t = 1 / num_steps
with torch.no_grad():
    for i in range(num_steps):
        t_batch = torch.full((x.size(0), 1), t, dtype=torch.float32)
        vt = model(x, t_batch)
        t += delta_t
        x = x + vt * delta_t  # x(t+Δt) = x(t) + v(t)Δt
        trajectory.append(x.detach().numpy())

trajectory = torch.tensor(trajectory)

# 转成numpy方便动画更新（shape: [num_steps+1, num_samples, 2]）
trajectory_np = trajectory.numpy()

# 动画：noise_data沿着trajectory逐步移动到target分布
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(target_data[:, 0], target_data[:, 1], c='blue', alpha=0.3, s=12, label='Target (sin(x))')
moving_points = ax.scatter(trajectory_np[0, :, 0], trajectory_np[0, :, 1], c='red', s=12, label='Moving Noise')

ax.set_xlim(
    min(noise_data[:, 0].min().item(), target_data[:, 0].min().item()) - 1,
    max(noise_data[:, 0].max().item(), target_data[:, 0].max().item()) + 1
)
ax.set_ylim(
    min(noise_data[:, 1].min().item(), target_data[:, 1].min().item()) - 1,
    max(noise_data[:, 1].max().item(), target_data[:, 1].max().item()) + 1
)
ax.set_title("Flow Matching Animation: Noise -> Target")
ax.legend(loc='upper right')

def update(frame):
    moving_points.set_offsets(trajectory_np[frame])
    ax.set_xlabel(f"Step: {frame}/{num_steps}")
    return moving_points,

ani = FuncAnimation(
    fig,
    update,
    frames=trajectory_np.shape[0],
    interval=80,
    blit=True,
    repeat=True
)

plt.show()