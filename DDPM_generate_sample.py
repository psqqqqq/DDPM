import deepinv
import torch
from torchvision.utils import save_image
from pathlib import Path


# DDPM GPU 采样脚本。
# 作用：加载已经训练好的噪声预测模型，从纯高斯噪声开始逐步反向去噪，生成 MNIST 风格图片。

# 固定使用 CUDA。没有 GPU 时请运行 DDPM_traing_cpu.py。
device = "cuda"
if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，请运行 DDPM_traing_cpu.py。")

# 图片大小必须与训练脚本保持一致。
image_size = 32

# 一次生成的图片数量。
n_samples = 32

# beta 调度参数必须与训练脚本保持一致。
beta_start = 1e-4
beta_end = 0.02

# todo:扩散步数必须与训练脚本保持一致，否则模型看到的时间步分布会不匹配。还有alphas_cumprod，sqrt之类的参数都必须和训练阶段完美匹配才行
# otherwise 在推理阶段，特定时间步上的噪声量，就不会与训练时所见的噪声量相对应
timesteps = 1000

# 训练好的模型参数路径。
checkpoint_path = Path("trained_diffusion_model.pth")

# 生成图片保存路径。
output_path = Path("outputs/sample_gpu.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

if not checkpoint_path.exists():
    raise FileNotFoundError(f"找不到 {checkpoint_path}，请先运行 DDPM.py 完成训练。")

# 创建与训练时完全相同结构的 DiffUNet，然后加载权重。
model = deepinv.models.DiffUNet(
    in_channels=1,
    out_channels=1,
    pretrained=None,
).to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

# 切换到推理模式，关闭训练阶段特有行为。
model.eval()

# 构造反向采样公式需要的系数表。
betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 采样阶段不需要梯度，关闭梯度可以节省显存并加快推理。
with torch.no_grad():
    # 从纯高斯噪声 x_T 开始。
    x = torch.randn(n_samples, 1, image_size, image_size, device=device)

    # 从 T-1 到 0 逐步反向去噪。
    for t in reversed(range(timesteps)):
        # 当前 batch 中所有样本使用同一个时间步 t。
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # 预测当前 x_t 中包含的噪声。
        predicted_noise = model(x, t_tensor, type_t="timestep")

        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]

        # 最后一步不再额外加随机噪声。
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        # DDPM 反向采样公式：根据预测噪声从 x_t 推到 x_{t-1}。
        x = (
            (1 / torch.sqrt(alpha))
            * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            + torch.sqrt(beta) * noise
        )

    # 将生成结果限制到 0~1，方便保存为图片。
    x = torch.clamp(x, 0.0, 1.0)
    save_image(x, output_path, nrow=8)

print(f"Generated samples saved to {output_path}")
