import os
import deepinv
import torch
from torchvision import datasets, transforms

# DDPM CPU 训练脚本。
# 目标和 GPU 版相同：训练一个噪声预测网络 epsilon_theta(x_t, t)。
# CPU 计算速度远低于 GPU，所以这里默认使用更小的 batch、epoch 和 timesteps，保证能在普通电脑上跑起来。

# 固定使用 CPU，不依赖 CUDA。
device = torch.device("cpu")

# 使用当前机器可用 CPU 线程数。线程数过高导致卡顿时，可以手动改成较小数字。
torch.set_num_threads(max(1, os.cpu_count() or 1))

# CPU 训练建议小 batch，避免内存压力和过长单步耗时。
batch_size = 8

# MNIST 原图是 28x28，这里放大到 32x32，方便 U-Net 结构处理。
image_size = 32

# 学习率：Adam 优化器每次参数更新的基础步长。
lr = 1e-4

# CPU 默认只跑 1 个 epoch。想要更好效果可以增大，但训练会明显变慢。
epochs = 1

# beta 控制每个扩散步加入的噪声强度。
beta_start = 1e-4
beta_end = 0.02

# CPU 默认使用 100 个扩散步，速度比 1000 步快很多。
# 采样脚本必须使用相同 timesteps。
timesteps = 100

# 模型权重保存路径。
checkpoint_path = "checkpoints/trained_diffusion_model_cpu.pth"
os.makedirs("checkpoints", exist_ok=True)

# 构建 MNIST 数据预处理流程。
transform = transforms.Compose(
    [
        # 将每张图缩放到 32x32。
        transforms.Resize(image_size),
        # 转换为 PyTorch 张量，像素值范围为 0~1。
        transforms.ToTensor(),
        # 保持数值不变，保留这个步骤便于以后改成其他归一化方式。
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

# DataLoader 负责按 batch 读取训练数据。
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    # Windows/CPU 环境下 num_workers=0 通常最稳定。
    num_workers=0,
)

# 创建噪声预测 U-Net。输入和输出都是单通道 MNIST 图像。
model = deepinv.models.DiffUNet(
    in_channels=1,
    out_channels=1,
    pretrained=None,
).to(device)

# 优化器和损失函数。
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = deepinv.loss.MSE()

# 构造 DDPM 前向扩散所需的系数表。
betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# 开始训练。
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for step, (data, _) in enumerate(train_loader, start=1):
        # 标签在无条件 DDPM 中不用，只训练图像分布。
        imgs = data.to(device)

        # 生成与图片同形状的真实噪声。
        noise = torch.randn_like(imgs)

        # 每张图随机选择一个扩散时间步。
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

        # 一步得到 x_t，而不是真的循环加噪 t 次。
        noised_imgs = (
            sqrt_alphas_cumprod[t, None, None, None] * imgs
            + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

        # 预测噪声并最小化预测噪声与真实噪声的 MSE。
        optimizer.zero_grad()
        estimated_noise = model(noised_imgs, t, type_t="timestep")
        loss = mse(estimated_noise, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 每 100 个 batch 打印一次，便于观察 CPU 训练是否正常推进。
        if step % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Step [{step}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.6f}")

# 保存 CPU 训练得到的模型。
torch.save(model.state_dict(), checkpoint_path)
print(f"CPU model saved to {checkpoint_path}")
