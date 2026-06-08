import os
from pathlib import Path

import deepinv
import torch
from torchvision.utils import save_image


# DDPM CPU 采样脚本。
# 作用：加载 DDPM_cpu.py 训练出的模型，从随机噪声开始逐步反向去噪，生成图片。

# 固定使用 CPU，不依赖 CUDA。
device = torch.device("cpu")

# 使用 CPU 线程加速。机器卡顿时可以手动调小。
torch.set_num_threads(max(1, os.cpu_count() or 1))

# 图片大小必须与训练脚本一致。
image_size = 32

# 一次生成的图片数量。CPU 过慢时可以改成 8 或 16。
n_samples = 32

# beta 调度参数必须与训练脚本一致。
beta_start = 1e-4
beta_end = 0.02

# 必须与 DDPM_cpu.py 中的 timesteps 一致。
timesteps = 100

# CPU 模型权重路径。
checkpoint_path = Path("checkpoints/trained_diffusion_model_cpu.pth")

# 生成图片保存路径。
output_path = Path("outputs/sample_cpu.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

if not checkpoint_path.exists():
    raise FileNotFoundError(f"找不到 {checkpoint_path}，请先运行 DDPM_cpu.py 完成训练。")

# 创建与训练时相同结构的模型，然后加载 CPU 权重。
model = deepinv.models.DiffUNet(
    in_channels=1,
    out_channels=1,
    pretrained=None,
).to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

# 推理模式：禁用训练阶段特有行为。
model.eval()

# 构造反向去噪所需的系数表。
betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 采样不需要计算梯度，关闭梯度可以节省内存和计算量。
with torch.no_grad():
    # 从纯高斯噪声开始，形状是 [样本数, 通道数, 高, 宽]。
    x = torch.randn(n_samples, 1, image_size, image_size, device=device)

    # 从最后一个时间步反向迭代到第 0 步。
    for t in reversed(range(timesteps)):
        # 当前 batch 的所有样本使用同一个时间步。
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # 模型预测当前图像中的噪声。
        predicted_noise = model(x, t_tensor, type_t="timestep")

        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]

        # 除最后一步外，每一步都加入少量随机性，符合 DDPM 采样过程。
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        # DDPM 反向采样公式：由 x_t 得到更干净的 x_{t-1}。
        x = (
            (1 / torch.sqrt(alpha))
            * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            + torch.sqrt(beta) * noise
        )

    # 限制到图片像素范围并保存网格图。
    x = torch.clamp(x, 0.0, 1.0)
    save_image(x, output_path, nrow=8)

print(f"Generated CPU samples saved to {output_path}")
