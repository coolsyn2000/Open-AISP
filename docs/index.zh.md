# Open-AISP

Open-AISP 是一个开源 AI-ISP 实验框架，用于搭建从 RAW 退化模拟到神经网络重建、HDR、智能色调映射和扩散式后处理增强的端到端图像信号处理流程。

项目按五个模块组织：

| 模块 | 状态 | 作用 |
| --- | --- | --- |
| `raw-sim` | 已实现 | 将 sRGB 图像转换为 noisy packed RAW，并生成 clean linear RGB 监督信号。 |
| `JDD` | 已实现 | 从 noisy RAW burst 恢复 clean linear RGB，完成联合去噪和去马赛克。 |
| `HDR` | 占位 | 后续用于多帧多曝光高动态范围合成。 |
| `AITM` | 占位 | 后续用于从 linear/HDR 图像到显示图像的 AI tone mapping。 |
| `DiffIPE` | 占位 | 后续用于基于扩散模型的图像后处理增强。 |

当前两个已实现模块通过统一退化接口连接：`JDD` 调用 `raw-sim` 生成 RAW 训练样本。后续只要在 `raw-sim` 中更新相机模型、CFA、PSF blur 或噪声模型，JDD 的训练数据退化也会同步更新。

## 总体流程

当前训练流程为：

```text
sRGB image
  -> raw-sim camera degradation
  -> noisy packed RAW burst
  -> JDD restoration network
  -> clean linear RGB
```

`raw-sim` 负责物理启发式 RAW 退化，`JDD` 负责从 noisy RAW 恢复 linear RGB。

## 语言

英文是默认文档语言。中文版本可通过语言切换入口访问。
