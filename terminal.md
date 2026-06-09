# idea

给出稀疏的医学图像，如CT图像，用diffusion model去补充缺失时间步的图像数据，然后做下游分析

针对每一个患者，不同的t有可能做了CT,也有可能没有做，所以需要根据每个患者的具体的数据补充其特定时间步

的缺失

| t      | 1 | 2 | 3 | 4 | ·· | T |
| ------ | - | - | - | - | ---- | - |
| 患者1  | 1 | 0 | 1 | 0 |      | 1 |
| 患者2  | 0 | 0 | 1 | 1 |      | 0 |
| 患者3  | 1 | 1 | 0 | 1 |      | 1 |
| ··· |   |   |   |   |      |   |
| 患者n  | 0 | 1 | 1 | 1 |      | 0 |

(表格中1代表有CT数据，0代表缺失CT数据)

## issue

但是目前DDPM是无条件生成的，也就是生成出的图片其实是完全独立的，而这里显然是要根据具体的患者的其他有数据的时间步的数据来生成缺失时间步的数据？

普通 DDPM：$x_t→x_0$

**我的目标：$x_T + 患者已有 CT + 时间 t + mask → 缺失时间点 CT$**

显然是不能用普通无条件 DDPM，普通 DDPM 只能学：所有 CT 图像长什么样，但是我想让模型知道同一个患者不同时间点之间如何变化。否则模型可能只是生成一张“看起来像 CT”的图，而不是这个患者特定时间点的 CT。

## how

做成“条件扩散模型用于纵向医学影像缺失时间点的概率性插补”，并不能用 DDPM 直接补图。**难点**在于条件设计、时间建模、缺失机制处理、医学验证，以及不能把生成图直接当成真实 CT。


# related paper

**Conditional Medical Image Generation WithDiffusion Models------------------------------**

条件扩散医学图像生成

**Med-cDiff** ，就是把 diffusion model 用于条件医学图像生成，任务包括 MRI 超分辨率、X-ray 去噪、MRI 图像到图像转换等。这个工作说明 conditional diffusion 在医学图像任务中是可行的

**ReMiND: Recovery of Missing Neuroimaging using Diffusion Models** -------------------------------

纵向医学影像生成

用 3D diffusion model 对缺失结构 MRI 进行插补，并且其在 whole-brain image imputation 上优于替代方法。和“缺失影像补全”逻辑非常接近，只是它主要是神经影像 MRI 场景。

SADM: Sequence-Aware Diffusion Model for Longitudinal Medical Image Generation------------------------

针对纵向医学影像生成，考虑序列长度不一、缺失数据、高维医学图像等问题，并用 sequence-aware transformer 作为扩散模型的条件模块来学习纵向依赖。

# Innovation？

生成图像是否能改善下游统计分析或预测模型？

影像组学特征提取
病灶体积变化趋势
治疗反应评估
预后预测
风险分层模型

能否证明补全后比“删掉缺失病例”或“简单插补”更稳定，这个就有临床分析价值。收窄到特定纵向 CT 场景，加入不规则时间、缺失机制、多重插补、不确定性传播和下游临床验证，就可以形成一个有论文价值的课题。

[innovation hit](https://chatgpt.com/share/6a28181a-8984-83ec-9331-0c45bb2a2585)
