#
# import torch
# from torchvision.utils import save_image
# from pathlib import Path
# from datetime import datetime  # 动态修改文件名
# import os
# os.environ["TORCH_HOME"] = r"D:\DDPM\torch_cache"  # 让权重文件在附近
# import deepinv
#
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # 作用：自动加载官方预训练的人像模型，从 3 通道纯高斯噪声开始逐步反向去噪，生成高清彩色人像。
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # 🚨 核心修改 1：deepinv 官方人像（FFHQ/CelebA）网络对应的图片大小是 64x64
# image_size = 64
#
# # 一次生成的图片数量。
# n_samples = 4  # 保持 4 个样本，方便演变 storyboard 图横向排版
#
# # beta 调度参数，匹配官方人像模型的训练设置
# beta_start = 1e-4
# beta_end = 0.02
# timesteps = 1000
#
# # 生成图片保存路径（修改文件名提示为 faces）
# output_path = Path(f"outputs/samples/generate_faces_{timestamp}.png")
# sequence_path = Path(f"outputs/samples/face_sequence_{timestamp}.png")  # storyboard图路径
#
# # 核心修改 2：创建 3 通道彩色 DiffUNet 结构，并直接指定 pretrained="ffhq"
# model = deepinv.models.DiffUNet(
#     in_channels=3,       # 👈 1. 彩色人脸必须是 3 通道
#     out_channels=3,      # 👈 2. 输出也必须是 3 通道
#     pretrained="download",   # 👈 3. 自动导入官方人脸权重（也可以换成 "celeba"）
# ).to(device)
# # pretrained="download" 背后用的是 torch.hub，它固定会在：TORCH_HOME\hub\checkpoints\
#
# # 切换到推理模式，关闭训练阶段特有行为。
# model.eval()
#
# # 构造反向采样公式需要的系数表。
# betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
# alphas = 1.0 - betas
# alphas_cumprod = torch.cumprod(alphas, dim=0)
#
# # 初始化一个列表，用来缓存各个时间节点的图片快照
# frames = []
#
# # 采样阶段不需要梯度，关闭梯度可以节省显存并加快推理。
# print("开始反向去噪采样...")
# with torch.no_grad():
#     # 🚨 核心修改 3：从 3 通道的纯高斯彩色噪声 x_T 开始 [4, 3, 64, 64]
#     x = torch.randn(n_samples, 3, image_size, image_size, device=device)
#
#     # 从 T-1 到 0 逐步反向去噪。
#     for t in reversed(range(timesteps)):
#         # 当前 batch 中所有样本使用同一个时间步 t。
#         t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
#
#         # 预测当前 x_t 中包含的噪声。同时传入时间步 t
#         model_output = model(x, t_tensor, type_t="timestep")
#
#         if model_output.shape[1] == 2 * x.shape[1]:
#             predicted_noise = model_output[:, :x.shape[1], :, :]
#             model_var_values = model_output[:, x.shape[1]:, :, :]
#         else:
#             predicted_noise = model_output
#             model_var_values = None
#
#         alpha = alphas[t]
#         alpha_cumprod = alphas_cumprod[t]
#         alphas_cumprod_prev = torch.cat(
#             [torch.ones(1, device=device), alphas_cumprod[:-1]],
#             dim=0
#         )
#
#         posterior_variance = (
#                 betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
#         )
#
#         posterior_log_variance_clipped = torch.log(
#             torch.clamp(posterior_variance, min=1e-20)
#         )
#         beta = betas[t]
#
#         # 最后一步不再额外加随机噪声。
#         model_mean = (
#                 (1 / torch.sqrt(alpha))
#                 * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
#         )
#
#         if t > 0:
#             noise = torch.randn_like(x)
#
#             if model_var_values is not None:
#                 min_log = posterior_log_variance_clipped[t]
#                 max_log = torch.log(beta)
#
#                 frac = (model_var_values + 1.0) / 2.0
#                 model_log_variance = frac * max_log + (1.0 - frac) * min_log
#
#                 x = model_mean + torch.exp(0.5 * model_log_variance) * noise
#             else:
#                 x = model_mean + torch.sqrt(beta) * noise
#         else:
#             x = model_mean
#
#         # 保留你的高级定时拦截条件：前 800 步不记录，从 200 步开始每隔 40 步记录一次，并带上第 900 步
#         if t <= 200 and (t % 40 == 0 or t == 0) or t == 900:
#             x_vis = (x.clone().clamp(-1.0, 1.0) + 1.0) / 2.0
#             frames.append(x_vis)
#
#     # 将生成结果限制到 0~1，保存最终彩色人脸的大图结果
#     x_vis = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
#     save_image(x_vis, output_path, nrow=2)
#     print(f"✨ 最终生成的彩色人像已保存到：{output_path}")
#
#     # ==================== 你的核心 Storyboard 视觉排版算法 ====================
#     num_timestamps = len(frames)
#     # Step 1: 堆叠成五维张量 -> [num_timestamps, n_samples, 3, 64, 64]
#     stacked_frames = torch.stack(frames, dim=0)
#     # Step 2: 交换维度 -> [n_samples, num_timestamps, 3, 64, 64]
#     permuted_frames = stacked_frames.permute(1, 0, 2, 3, 4)
#     # 🚨 核心修改 4：展平时，通道数从 1 改为 3 对应彩色图 -> [n_samples * num_timestamps, 3, 64, 64]
#     sequence_tensor = permuted_frames.reshape(-1, 3, image_size, image_size)
#     # Step 4: 设定 nrow 等于时间步数，横向展现全生命周期
#     save_image(sequence_tensor, sequence_path, nrow=num_timestamps)
#
# print(f"🎨 从噪声到彩色人像的完整演变顺序图已成功保存至: {sequence_path}")

import os
from pathlib import Path

# 必须放在 import deepinv 前面
ROOT_DIR = Path(__file__).resolve().parent
os.environ["TORCH_HOME"] = str(ROOT_DIR / "torch_cache")

import deepinv
import torch
from torchvision.utils import save_image
from datetime import datetime


# ==================== 基础设置 ====================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备：", device)

# deepinv 官方 FFHQ 人像模型对应 64x64 彩色图像
image_size = 64

# 一次生成 4 张人脸
n_samples = 4

# beta 调度参数
beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

# 输出路径
output_dir = ROOT_DIR / "outputs" / "samples"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"generate_faces_{timestamp}.png"
sequence_path = output_dir / f"face_sequence_{timestamp}.png"


# ==================== 加载官方预训练模型 ====================

print("正在初始化并加载官方 FFHQ 人像预训练权重...")

model = deepinv.models.DiffUNet(
    in_channels=3,
    out_channels=3,
    pretrained="download",
).to(device)

model.eval()

print("模型加载完成。")


# ==================== 构造扩散参数 ====================

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 用于 learned variance 的 posterior variance
alphas_cumprod_prev = torch.cat(
    [torch.ones(1, device=device), alphas_cumprod[:-1]],
    dim=0
)

posterior_variance = (
    betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
)

posterior_log_variance_clipped = torch.log(
    torch.clamp(posterior_variance, min=1e-20)
)


# ==================== 工具函数：把模型内部图像转成可保存图片 ====================

def to_image_range(x):
    """
    很多 diffusion 模型内部图像范围是 [-1, 1]。
    保存图片前需要映射到 [0, 1]。
    """
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


# ==================== 反向采样 ====================

frames = []

print("开始反向去噪采样...")

with torch.no_grad():
    # 从 3 通道纯高斯噪声开始
    x = torch.randn(
        n_samples,
        3,
        image_size,
        image_size,
        device=device,
    )

    for t in reversed(range(timesteps)):
        t_tensor = torch.full(
            (n_samples,),
            t,
            device=device,
            dtype=torch.long,
        )

        # 模型输出
        model_output = model(x, t_tensor, type_t="timestep")

        # deepinv 官方 FFHQ 模型可能输出 6 通道：
        # 前 3 通道：预测噪声 epsilon
        # 后 3 通道：预测方差相关值
        if model_output.shape[1] == 2 * x.shape[1]:
            predicted_noise = model_output[:, :x.shape[1], :, :]
            model_var_values = model_output[:, x.shape[1]:, :, :]
        else:
            predicted_noise = model_output
            model_var_values = None

        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]

        # 根据预测噪声计算均值部分
        model_mean = (
            (1.0 / torch.sqrt(alpha))
            * (
                x
                - (beta / torch.sqrt(1.0 - alpha_cumprod))
                * predicted_noise
            )
        )

        # 加入方差项，得到 x_{t-1}
        if t > 0:
            noise = torch.randn_like(x)

            if model_var_values is not None:
                min_log = posterior_log_variance_clipped[t]
                max_log = torch.log(beta)

                # learned_range variance，限制在 [0, 1]，避免方差爆炸
                frac = torch.clamp((model_var_values + 1.0) / 2.0, 0.0, 1.0)

                model_log_variance = (
                    frac * max_log
                    + (1.0 - frac) * min_log
                )

                x = model_mean + torch.exp(0.5 * model_log_variance) * noise
            else:
                x = model_mean + torch.sqrt(beta) * noise
        else:
            x = model_mean

        # 打印几个关键时间步的数值，方便判断有没有爆炸
        if t in [999, 900, 700, 500, 200, 100, 50, 0]:
            print(
                f"t={t:03d} | "
                f"x min={x.min().item():.3f}, "
                f"x max={x.max().item():.3f}, "
                f"x mean={x.mean().item():.3f}, "
                f"x std={x.std().item():.3f}, "
                f"model_output shape={tuple(model_output.shape)}"
            )

        # 保存 storyboard 中间过程
        # 注意：采样顺序是 999 -> 0，所以这里记录的是逐渐去噪过程
        if (t == 900) or (t <= 200 and (t % 40 == 0 or t == 0)):
            frames.append(to_image_range(x.clone()))

    # 保存最终生成图
    final_img = to_image_range(x)
    save_image(final_img, output_path, nrow=2)

print(f"最终生成彩色人像已保存到：{output_path}")


# ==================== 保存 storyboard 演变图 ====================

if len(frames) > 0:
    num_timestamps = len(frames)

    # [num_timestamps, n_samples, 3, 64, 64]
    stacked_frames = torch.stack(frames, dim=0)

    # [n_samples, num_timestamps, 3, 64, 64]
    permuted_frames = stacked_frames.permute(1, 0, 2, 3, 4)

    # [n_samples * num_timestamps, 3, 64, 64]
    sequence_tensor = permuted_frames.reshape(
        -1,
        3,
        image_size,
        image_size,
    )

    save_image(sequence_tensor, sequence_path, nrow=num_timestamps)

    print(f"从噪声到人像的演变图已保存到：{sequence_path}")
else:
    print("没有保存任何中间帧。")

