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

# deepinv 官方 FFHQ 预训练模型对应 256x256 彩色图像
image_size = 256

# 先只生成 1 张，避免显存爆炸
n_samples = 1

# 是否保存从噪声到人脸的中间演变图
# 第一次建议 False，先确认最终人脸能正常生成
save_storyboard = True

# storyboard 想保存的时间步，前期更密一些，过渡会更丝滑
storyboard_timesteps = [
    999, 950, 900, 800, 700, 650, 550, 500,
    450, 400,  300, 250, 200, 160, 80,  0
]

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

# 用于处理 learned variance
alphas_cumprod_prev = torch.cat(
    [torch.ones(1, device=device), alphas_cumprod[:-1]],
    dim=0,
)

posterior_variance = (
    betas
    * (1.0 - alphas_cumprod_prev)
    / (1.0 - alphas_cumprod)
)

posterior_log_variance_clipped = torch.log(
    torch.clamp(posterior_variance, min=1e-20)
)


# ==================== 工具函数 ====================

def to_image_range(x):
    """
    diffusion 模型内部图像范围通常是 [-1, 1]。
    保存成图片前需要映射到 [0, 1]。
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

        # FFHQ 模型可能输出 6 通道：
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

        # DDPM 反向均值
        model_mean = (
            (1.0 / torch.sqrt(alpha))
            * (
                x
                - (beta / torch.sqrt(1.0 - alpha_cumprod))
                * predicted_noise
            )
        )

        # 方差项
        if t > 0:
            noise = torch.randn_like(x)

            if model_var_values is not None:
                min_log = posterior_log_variance_clipped[t]
                max_log = torch.log(beta)

                frac = torch.clamp(
                    (model_var_values + 1.0) / 2.0,
                    0.0,
                    1.0,
                )

                model_log_variance = (
                    frac * max_log
                    + (1.0 - frac) * min_log
                )

                x = model_mean + torch.exp(0.5 * model_log_variance) * noise
            else:
                x = model_mean + torch.sqrt(beta) * noise
        else:
            x = model_mean

        # 打印关键时间步的数值，观察有没有爆炸
        if t in [999, 900, 700, 500, 200, 100, 50, 0]:
            print(
                f"t={t:03d} | "
                f"x min={x.min().item():.3f}, "
                f"x max={x.max().item():.3f}, "
                f"x mean={x.mean().item():.3f}, "
                f"x std={x.std().item():.3f}, "
                f"model_output shape={tuple(model_output.shape)}"
            )

        # 可选：保存中间过程
        # if save_storyboard:
        #     if (t == 900) or (t <= 200 and (t % 40 == 0 or t == 0)):
        #         frames.append(to_image_range(x.clone()))
        if save_storyboard and t in storyboard_timesteps:
            frames.append(to_image_range(x.clone()))
        #生成人像不要和数字一样

    # 保存最终生成图
    final_img = to_image_range(x)
    save_image(final_img, output_path, nrow=8)  # nrow参数是指定一行放几张图片

print(f"最终生成彩色人像已保存到：{output_path}")


# ==================== 可选：保存 storyboard 演变图 ====================

if save_storyboard and len(frames) > 0:
    num_timestamps = len(frames)

    stacked_frames = torch.stack(frames, dim=0)
    permuted_frames = stacked_frames.permute(1, 0, 2, 3, 4)

    sequence_tensor = permuted_frames.reshape(
        -1,
        3,
        image_size,
        image_size,
    )

    save_image(sequence_tensor, sequence_path, nrow=num_timestamps)

    print(f"从噪声到人像的演变图已保存到：{sequence_path}")
else:
    print("未保存 storyboard，只保存最终生成图。")
