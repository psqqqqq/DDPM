import torch # 神经网络需要的所有工具
import deepinv # deep_inverse,基于pytorch构建，专门用于解决图像逆问题，提供了现成的DDPM中使用的神经网络架构，以及多个预训练权重pre_trained weights
from torchvision import datasets, transforms

device = "cuda"
batch_size = 32
image_size = 32

# %% setup the dataset Mnist,小型灰度图像
transform = transforms.Compose(
    [
        transforms.Resize(image_size), # 每张图片调整为32*32像素
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)


lr = 1e-4
epochs = 100

model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(
    device
)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = deepinv.loss.MSE()

beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

for epoch in range(epochs):
    model.train()
    for data, _ in train_loader:
        imgs = data.to(device)
        noise = torch.randn_like(imgs)
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

        noised_imgs = (
            sqrt_alphas_cumprod[t, None, None, None] * imgs
            + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

        optimizer.zero_grad()
        estimated_noise = model(noised_imgs, t, type_t="timestep")
        loss = mse(estimated_noise, noise)
        loss.backward()
        optimizer.step()

torch.save(
    model.state_dict(),
    "trained_diffusion_model.pth",
)