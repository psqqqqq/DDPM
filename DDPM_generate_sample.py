import deepinv
import torch
from torchvision.utils import save_image
from pathlib import Path
from datetime import datetime  # 动态修改文件名

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 作用：加载已经训练好的噪声预测模型，从纯高斯噪声开始逐步反向去噪，生成 MNIST 风格图片。固定使用 CUDA。没有 GPU 时请运行 DDPM_traing_cpu.py。
device = "cuda"
if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，请运行 DDPM_traing_cpu.py。")

image_size = 32  # 图片大小必须与训练脚本保持一致。

# 一次生成的图片数量。
# n_samples = 32
n_samples = 4  # 要生成storyboard32太多了

# beta 调度参数必须与训练脚本保持一致。
beta_start = 1e-4
beta_end = 0.02

# todo:扩散步数必须与训练脚本保持一致，否则模型看到的时间步分布会不匹配。还有alphas_cumprod，sqrt之类的参数都必须和训练阶段完美匹配才行
# otherwise 在推理阶段，特定时间步上的噪声量，就不会与训练时所见的噪声量相对应
timesteps = 1000

# 生成图片保存路径。
output_path = Path(f"outputs/samples/generate_samples_{timestamp}.png")

sequence_path = Path(f"outputs/samples/generation_sequence_{timestamp}.png")  # storyboard图路径

# TODO:训练好的模型参数路径。
epoch_weight = 60  # 选用第几轮epoch输出的权重来生成
checkpoint_path = Path(f"./checkpoints/model_epoch_{epoch_weight}.pth")  # ./就是同一层级下

# 创建与训练时完全相同结构的 DiffUNet，然后加载权重。
model = deepinv.models.DiffUNet(
    in_channels=1,
    out_channels=1,
    pretrained=None,  # TODO:导入权重
).to(device)

state_dict = torch.load(checkpoint_path, map_location=device,weights_only=True)
model.load_state_dict(state_dict,strict=False)  # 使用非严格模式载入权重这样PyTorch会自动完美对齐U-Net的卷积层注意力层等核心权重把多余的"sqrt_alphas_cumprod"忽略
# TODO:上面7行载入权重的地方可以灵活修改，得益于deepinverse,我们可以轻松地导入不同数据集的预训练权重
# pretrained="mnist",  # 👈 直接指定数据集名称，库会自动在后台执行 download 并加载
# model = deepinv.models.DiffUNet(
#     in_channels=3,       # 👈 彩色人脸是 3 通道
#     out_channels=3,
#     pretrained="celeba", # 👈 自动下载并加载官方明星人脸预训练权重
# ).to(device)
# model=deepinv.models.DiffUNet(
#     in_channels=1,
#     out_channels=1,
#     pretrained="download"  # 替换
# ).to(device)
# model = deepinv.models.DiffUNet(
#     in_channels=3,     # 👈 FFHQ 是 RGB 彩色图，必须是 3 通道
#     out_channels=3,    # 👈 输出也必须是 3 通道
#     pretrained="ffhq", # 👈 告诉 deepinv 自动从云端下载并导入 FFHQ 的官方预训练权重
# ).to(device)


# 切换到推理模式，关闭训练阶段特有行为。
model.eval()

# 构造反向采样公式需要的系数表。
betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

#  初始化一个列表，用来缓存各个时间节点的图片快照
frames = []
save_interval = 200

# TODO:sample processing 采样阶段不需要梯度，关闭梯度可以节省显存并加快推理。
with torch.no_grad():
    # 从纯高斯噪声 x_T 开始。
    x = torch.randn(n_samples, 1, image_size, image_size, device=device)

    # 从 T-1 到 0 逐步反向去噪。
    for t in reversed(range(timesteps)):
        # 当前 batch 中所有样本使用同一个时间步 t。
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # 预测当前 x_t 中包含的噪声。同时传入时间步t
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
        # 定时拦截。每隔指定步长，或者数到最后一步 t=0 时，拷贝当前图像快照
        # if t % save_interval == 0 or t == 0:
        #     frames.append(torch.clamp(x.clone(), 0.0, 1.0))
        #  修改拦截条件：前 800 步不记录，从 200 步开始每隔 40 步记录一次
        if t <= 200 and (t % 40 == 0 or t == 0) or t == 900:
            frames.append(torch.clamp(x.clone(), 0.0, 1.0))

    # 将生成结果限制到 0~1，方便保存为图片。保存最终单张的大图结果
    x = torch.clamp(x, 0.0, 1.0)  # clamp 夹住 像素值 < 0.0（比如 -0.15），强行变成 0.0 如果像素值 > 1.0（比如 1.2），强行变成 1.0
    save_image(x, output_path, nrow=8)
    print(f"生成的图片已保存到{output_path}")
    # TODO:storyboard图
    # 高维 张量 变换逻辑（核心视觉排版算法）
    # 目前 frames 列表中有 7 个元素（1张纯噪声 + 6张去噪中途图），每个形状为 [n_samples, 1, 32, 32]
    num_timestamps = len(frames)
    # Step 1: 堆叠成五维张量 -> [num_timestamps, n_samples, 1, 32, 32]
    stacked_frames = torch.stack(frames, dim=0)
    # Step 2: 交换维度，把样本轴提到最前 -> [n_samples, num_timestamps, 1, 32, 32]
    # 这样能保证“同一个样本的不同时间步”在内存里紧密连续排列
    permuted_frames = stacked_frames.permute(1, 0, 2, 3, 4)
    # Step 3: 展平前两维，融合成一个大 Batch -> [n_samples * num_timestamps, 1, 32, 32]
    sequence_tensor = permuted_frames.reshape(-1, 1, image_size, image_size)
    # Step 4: 设定 nrow 等于时间步数。save_image 会横向填满 num_timestamps 张图后换行
    # 最终视觉效果：每一行完整展现一个样本从噪声变成数字的全生命周期
    save_image(sequence_tensor, sequence_path, nrow=num_timestamps)
print(f"从噪声到生成的完整演变顺序图已成功保存至: {sequence_path}")