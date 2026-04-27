# JDD

JDD 是联合去噪与去马赛克模块，目标是从 noisy packed RAW burst 恢复 clean linear RGB：

$$
f_\theta: \{R_t^{noisy}\}_{t=1}^{T}, M_\sigma \rightarrow I_{lin}
$$

当前训练配置中 $T=3$，$R_t^{noisy}$ 是独立噪声采样得到的 RAW 帧，$M_\sigma$ 是 `raw-sim` 生成的噪声标准差图。

## 任务定义

输入由多帧 packed RAW 和一个 noise map 通道拼接得到：

$$
X = \operatorname{concat}(R_1, R_2, R_3, M_\sigma)
$$

对于 Bayer 或 2x2 binning RAW，每帧有 4 个 packed channel，因此：

$$
C_{in}=3 \times 4 + 1 = 13
$$

对于 Quad Bayer RAW，每帧有 16 个 packed channel：

$$
C_{in}=3 \times 16 + 1 = 49
$$

训练目标是完整空间分辨率的 clean linear RGB：

$$
Y = I_{lin} \in [0,1]^{3 \times H \times W}
$$

## 随机退化

JDD 默认不保存固定 noisy 数据集，而是在训练过程中在线采样 RGB patch，并通过 `raw-sim` 的公开 batch 接口生成 RAW 退化数据。

每个 sample 会执行：

1. 随机选择一张 RGB 图像。
2. 随机裁剪 `data.patch_size` 大小的 patch。
3. 从 camera module 中随机采样 analog gain。
4. 将 sRGB 转成 clean linear RGB target。
5. 调用 `raw-sim` pipeline 生成 3 帧 noisy RAW。
6. 基于同一套噪声标定参数生成 noise map。

这样每个 epoch 都会看到不同 crop、不同噪声随机数和不同 analog gain。

## Analog Gain 采样

camera JSON 中的 `camera.analog_gain` 可以是标量、二元范围，或者包含 `min` 和 `max` 的对象。JDD 根据 `camera.analog_gain_sampling` 选择采样分布。

均匀采样：

$$
AG = AG_{min} + u(AG_{max} - AG_{min}), \quad u \sim \mathcal{U}(0,1)
$$

对数均匀采样：

$$
AG = \exp\left(\log AG_{min} + u(\log AG_{max}-\log AG_{min})\right)
$$

当前 ISO 约定继承自 `raw-sim`：

$$
ISO = 100 \cdot AG
$$

由于 Poisson-Gaussian 噪声模型会随 analog gain 缩放，随机 gain 采样能让模型覆盖更宽的噪声强度，而不是只学习某一个固定 ISO。

## 多帧 Burst Restoration

对于同一个 RGB patch，JDD 会从同一个 clean signal 合成 3 帧 noisy RAW：

$$
R_t^{noisy} = \mathcal{D}(I_{srgb}; AG, \epsilon_t), \quad t \in \{1,2,3\}
$$

其中 $\mathcal{D}$ 是 `raw-sim` 退化 pipeline，$\epsilon_t$ 是每一帧独立的噪声随机种子。

这 3 帧共享同一个 crop、camera 参数、analog gain、CFA 排列和 lens filter，但噪声采样相互独立。这样网络可以利用 burst 冗余做去噪，同时学习去马赛克和 RAW 到 linear RGB 的恢复。

## 训练数据路径

当 `data.simulate_on_device` 开启时，DataLoader 只返回：

- RGB patch
- random seed
- sampled analog gain
- metadata

训练循环中再调用：

```python
from raw_sim.batch import simulate_burst_batch_on_device

x, y = simulate_burst_batch_on_device(
    rgb=batch["rgb"],
    analog_gain=batch["analog_gain"],
    seed=batch["seed"],
    camera=camera,
    frames=3,
    noise_map_reduce="mean",
)
```

这样 JDD 的退化逻辑只依赖 `raw-sim` 接口，而不是在 JDD 内部复制一份模拟代码。后续 `raw-sim` 中的 lens PSF、CFA packing 或噪声模型更新，会自动进入 JDD 训练。

## 模型结构

当前网络是一个轻量 NAFNet-style restoration model：

```text
packed RAW burst + noise map
  -> input convolution
  -> NAF blocks
  -> output convolution
  -> PixelShuffle CFA upsampling
  -> clean linear RGB
```

PixelShuffle 的上采样倍率由 CFA 类型决定：

| CFA 类型 | Packed scale | 每帧 RAW channel | 3 帧输入 channel |
| --- | ---: | ---: | ---: |
| Bayer / binning | 2 | 4 | 13 |
| Quad Bayer | 4 | 16 | 49 |

## 训练命令

```bash
cd JDD
bash ./scripts/train.sh --config ./configs/train_JDD_patch128_3f_iter100000.json
```

常用训练字段：

| 字段 | 作用 |
| --- | --- |
| `data.simulate_on_device` | 使用 GPU 侧 `raw-sim` batch 退化。 |
| `data.frames` | burst 帧数。当前 JDD 期望为 3。 |
| `data.noise_map_reduce` | 将多通道噪声图合成为输入 noise map 的方式。 |
| `train.batch_size` | 每 step 的 RGB patch 数量。 |
| `train.amp` | CUDA 混合精度训练。 |
| `train.channels_last` | CUDA channels-last memory format。 |
| `train.val_every` | validation 间隔。 |

