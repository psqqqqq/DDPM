# DDPM

diffusion model

# 当前虚拟环境使用 D:\\envs\\myfirstpytorchenv\\python.exe"来创建的

关系，包独立
(.venv) PS D:\\DDPM> Get-Content ..venv\\pyvenv.cfg
home = D:\\envs\\myfirstpytorchenv
include-system-site-packages = false
version = 3.11.14
executable = D:\\envs\\myfirstpytorchenv\\python.exe
command = D:\\envs\\myfirstpytorchenv\\python.exe -m venv D:\\DDPM.venv

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install deepinv

相比直接pip
python -m pip install --upgrade pip
python -m pip install -r requirements.txt要更稳健装到当前环境

D:\DDPM
├── train_cpu.py
├── train_gpu.py
├── sample_cpu.py
├── sample_gpu.py
├── requirements.txt
├── README.md
├── .gitignore
├── checkpoints
│   └── .gitkeep
└── outputs
    └── .gitkeep

额外加了一个 `outputs/`，用于保存生成图片。
