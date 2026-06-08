import deepinv
import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm # 进度条模块
from torchvision.utils import save_image  # 生成监控具体图片
# DDPM GPU 训练脚本。
# 目标：训练一个噪声预测网络 epsilon_theta(x_t, t)。
# 给定原图 x_0 和随机时间步 t，先按前向扩散公式得到带噪图 x_t，
# 再让模型根据 x_t 和 t 预测当初加入的真实噪声 epsilon。

# 固定使用 CUDA。没有 NVIDIA GPU 或 CUDA 环境时，这个脚本会报错。
device = "cuda"
if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，请运行 DDPM_cpu.py。")

# TODO:hyper parameters
# 每次送入模型的图片数量。GPU 显存不足时可以调小。
batch_size = 32

# MNIST 原图是 28x28，这里放大到 32x32，方便 U-Net 做多层下采样和上采样。
image_size = 32

# 学习率：Adam 优化器每次参数更新的基础步长。
lr = 1e-4

# 训练轮数：完整遍历训练集的次数。
epochs = 100

# beta 控制每个扩散步加入的噪声强度。
# 线性 beta 调度会让图像从清晰逐步变成近似纯高斯噪声。
beta_start = 1e-4
beta_end = 0.02

# 扩散总步数 T，时间步 t 的取值范围是 [0, timesteps - 1]。
timesteps = 1000

sample_dir = "outputs/monitor_samples"
sample_interval = 10  # 和保存权重频率对应
n_sample_images = 16  # 在训练过程中进行效果监控时，每次执行采样（生成图片）要一次性生成 16（4*4）张数字图片。

os.makedirs(sample_dir, exist_ok=True)

# 保存训练后模型参数的位置。在经历 100 轮（epochs = 100）训练、看遍了MNIST 数据集后学到的权重 Weights和偏置 Biases）。
# pth（或 .pt）是 PyTorch 框架官方推荐的模型文件后缀名,它本质上是一个经过序列化（Pickle）压缩的文件，只有 PyTorch 能够直接读取它。
checkpoint_dir="checkpoints"  # 存到checkpoints文件夹下
checkpoint_interval = 10  # 每隔10个epoch保存一次权重文件pth
final_checkpoint_path = f"{checkpoint_dir}/final_trained_diffusion_model.pth"


# TODO:setup the dataset
# 构建 MNIST 数据预处理流程。
transform = transforms.Compose(
    [
        # 将图片从 28x28 缩放到 32x32。
        transforms.Resize(image_size),
        # 将 PIL 图片转换为形状 [C, H, W] 的张量，像素范围变为 0~1。
        transforms.ToTensor(),
        # 这里 mean=0、std=1，数值实际不变，保留为预处理占位。
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

# DataLoader 负责分批读取训练数据。这里是从网上下载数据集
train_loader = torch.utils.data.DataLoader(
    # train=True 使用训练集；download=True 在本地没有数据时自动下载到 ./data。
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    # 每个 epoch 打乱数据顺序，减少训练顺序带来的偏差。
    shuffle=True,
)
# %% TODO:setup the model
# DiffUNet 是 deepinv 提供的扩散 U-Net。
# in_channels=1、out_channels=1 对应 MNIST 灰度图：输入带噪图，输出预测噪声图。
# 有了deepinv就简单了，只需要实例化DiffusionUNetL类，并指定输入和输出通道的数量
model = deepinv.models.DiffUNet(
    in_channels=1,
    out_channels=1,
    pretrained=None, # 这个视频的例子中，是从头开始训练的，所以没有加载任何预训练权重
).to(device)

# Adam 根据 loss.backward() 计算出的梯度更新模型参数。
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# MSE 损失：让模型预测噪声 estimated_noise 尽量接近真实噪声 noise。这里就是使用简单的均方误差
mse = deepinv.loss.MSE()

# %% TODO:Diffusion constants
# 构造扩散过程需要的系数表，全部放在 GPU 上避免设备不一致。beta scale,描述了扩散过程中每一步添加的噪声水平，线性添加，后面还可以选择余弦添加
betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

# alpha_t = 1 - beta_t，表示每一步保留原信号的比例。
alphas = 1.0 - betas

# alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t。
# 它表示从 x_0 一次性扩散到 x_t 时，原图信号累计保留了多少。
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 前向扩散闭式公式：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon。
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# TODO:监控图函数
def sample_current_model(model, epoch):
    """
    用当前训练到第 epoch 轮的模型，从纯噪声开始反向采样，
    保存一张 4x4 的数字生成监控图。
    """
    model.eval()

    with torch.no_grad():
        x = torch.randn(
            n_sample_images,
            1,
            image_size,
            image_size,
            device=device,
        )

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(
                n_sample_images,
                device=device,
                dtype=torch.long,
            ) * t

            predicted_noise = model(x, t_tensor, type_t="timestep")

            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (
                (1 / torch.sqrt(alpha))
                * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
                + torch.sqrt(beta) * noise
            )

        x = torch.clamp(x, 0.0, 1.0)

        save_path = f"{sample_dir}/epoch_{epoch:03d}.png"
        save_image(x, save_path, nrow=4)

    model.train()
    print(f"第 {epoch} 个 epoch 的生成监控图已保存到：{save_path}")

# %% 训练主循环。TODO:Training loop,每一步从0-1000之间随机抽取一个时间步t,从标准正态中抽取一个噪声epsilon
# TODO然后用之前推导的variance-preserving formulation，来计算出带噪声的图像
for epoch in range(epochs):
    # 启用训练模式。对 Dropout、BatchNorm 等层，训练和推理行为不同。
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # 加个进度条
    for data, _ in progress_bar:
        # data 的形状是 [B, 1, 32, 32]。
        # 第二个返回值是 MNIST 标签，DDPM 是无条件生成模型，这里不使用标签。
        imgs = data.to(device) # TODO:这就是X0

        # 从标准正态分布采样真实噪声 epsilon，形状与图片完全相同。
        noise = torch.randn_like(imgs)

        # 每张图片随机采样一个时间步 t，形状是 [B]。
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

        # 根据 DDPM 前向扩散公式构造 x_t。
        # [B] 的系数通过 None 扩展成 [B, 1, 1, 1]，从而能和图片张量广播相乘。
        # TODO:我们在理论中使用的公式和实际实现中的完全一致，
        noised_imgs = (
            sqrt_alphas_cumprod[t, None, None, None] * imgs
            + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

        # 清空上一轮反向传播累积的梯度。
        optimizer.zero_grad()

        # 模型输入带噪图 x_t 和离散时间步 t，输出对噪声 epsilon 的预测。
        estimated_noise = model(noised_imgs, t, type_t="timestep")

        # 损失越小，说明预测噪声越接近真实噪声。
        loss = mse(estimated_noise, noise).mean() # 这里要加mean()改成标量

        # 计算梯度并更新参数。
        loss.backward() # 直接调用 loss.backward() 时，loss 必须是一个标量，也就是只有一个数
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    # 监控训练过程
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}") # :表示格式化操作的开始，.6表示保留6位小数，f表示作为浮点数来格式化
    if (epoch + 1) % checkpoint_interval == 0:  # 每隔interval保存一次结果
        save_path = f"{checkpoint_dir}/model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"epoch为{epoch}时权重中途已保存")
    # 每隔 sample_interval 个 epoch 保存一次生成图片
    if (epoch + 1) == 1 or (epoch + 1) % sample_interval == 0:
        sample_current_model(model, epoch + 1)
    # %% TODO:保存模型参数，之后采样或继续训练时可以重新加载。
torch.save(model.state_dict(), final_checkpoint_path) #
print(f" model 最后跑完所有epoch已保存到  {final_checkpoint_path}")
