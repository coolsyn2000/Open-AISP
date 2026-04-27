# raw-sim

`raw-sim` 反向模拟相机早期成像链路：从 clean sRGB 图像生成 noisy packed RAW，同时保留 clean linear RGB 作为监督信号。该模块也是 JDD 等下游学习模块的数据退化来源。

## 从 sRGB 到 Noisy RAW 的 Pipeline

对每张输入 sRGB 图像或 crop，`raw-sim` 依次执行：

```text
sRGB
  -> 逆 gamma
  -> 逆 CCM
  -> 逆 AWB
  -> lens PSF / filter
  -> sensor level mapping
  -> CFA sampling and packing
  -> Poisson-Gaussian noise
  -> noisy packed RAW
```

clean target 是逆 gamma 后的 linear RGB。镜头模糊、CFA 采样和传感器噪声只作用在 RAW 分支。

## 1. 逆 Gamma

输入图像先归一化到 $[0, 1]$，clean linear RGB target 为：

$$
I_{lin} = \operatorname{clip}(I_{srgb}, 0, 1)^\gamma
$$

示例配置默认使用 $\gamma = 2.2$。

## 2. CCM

相机 JSON 中的 CCM 表示从 camera RGB 到 linear RGB 的颜色校正：

$$
I_{lin} = I_{cam} C^\top
$$

反向 unprocess 时使用逆变换：

$$
I_{cam}' = I_{lin}(C^{-1})^\top
$$

其中 $C$ 对应 `ccm.matrix`。

## 3. AWB

AWB 通过除以每个颜色通道的 gain 来反向模拟：

$$
I_{cam,c} = \frac{I_{cam,c}'}{g_c}, \quad c \in \{R,G,B\}
$$

配置字段为 `awb.red_gain`、`awb.green_gain` 和 `awb.blue_gain`。

## 4. PSF Blur 和 Guided Filter

`lens_psf` 作用在 camera RGB 上，位于 sensor level 映射和 CFA 采样之前。

Gaussian blur 使用归一化高斯核：

$$
K(x,y) = \frac{1}{Z}\exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
$$

示例：

```json
{
  "lens_psf": {
    "enabled": true,
    "kernel": "gaussian",
    "kernel_size": 5,
    "sigma": 1.0
  }
}
```

也支持 guided filter：

```json
{
  "lens_psf": {
    "enabled": true,
    "kernel": "guided",
    "kernel_size": 5,
    "eps": 0.001
  }
}
```

在局部窗口 $\omega_k$ 中，guided filter 假设：

$$
q_i = a_k G_i + b_k, \quad i \in \omega_k
$$

其中：

$$
a_k = \frac{\operatorname{cov}_{\omega_k}(G, p)}
{\operatorname{var}_{\omega_k}(G) + \epsilon},
\qquad
b_k = \bar{p}_k - a_k \bar{G}_k
$$

`raw-sim` 使用亮度作为 guide：

$$
G = 0.299R + 0.587G + 0.114B
$$

相比纯 Gaussian blur，guided filter 可以提供边缘感知的平滑。

## 5. Sensor Level Mapping

camera RGB 被映射到传感器数字范围：

$$
S = B + I_{cam}(W - B)
$$

其中 $B$ 是 black level，$W$ 是 white level，二者都按最大 DN 归一化：

$$
DN_{max} = 2^{bitdepth} - 1
$$

如果 `black_level` 为 `"auto"`，默认 10-bit black level reference 是 32 DN：

| Bit depth | Auto black level |
| --- | --- |
| 10 | 32 |
| 12 | 128 |
| 14 | 512 |

如果开启 `quantize`，数值会按 bit depth 量化。

## 6. CFA Sampling

`raw-sim` 支持 Bayer/binning 2x2 packing 和 Quad Bayer 4x4 packing。

### Bayer 与 2x2 Binning

以 `RGGB` 为例，2x2 tile 为：

```text
R   Gr
Gb  B
```

packed RAW 形状为：

$$
H \times W \times 3 \rightarrow \frac{H}{2} \times \frac{W}{2} \times 4
$$

通道顺序为：

```text
[R, Gr, Gb, B]
```

当 `cfa.type = "binning"` 时，每个 packed channel 可以继续做 2x2 局部滤波。`binning_operation: "average"` 保持信号幅度接近 full readout，`"sum"` 表示累加局部信号。

### Quad Bayer 4x4

Quad Bayer 会把每个 Bayer 颜色扩展为 2x2 同色块。以 `RGGB` 为例，4x4 tile 为：

```text
R   R   Gr  Gr
R   R   Gr  Gr
Gb  Gb  B   B
Gb  Gb  B   B
```

packed RAW 形状为：

$$
H \times W \times 3 \rightarrow \frac{H}{4} \times \frac{W}{4} \times 16
$$

通道顺序是 4x4 tile 的 raster order：`ch0` 到 `ch15`。

## 7. Poisson-Gaussian 噪声

active signal 会先减 black level，再除以有效传感器范围：

$$
x = \operatorname{clip}\left(\frac{S - B}{W - B}, 0, 1\right)
$$

每个通道的噪声方差为：

$$
\sigma^2(x, c) = K_c x + \sigma_{read,c}^2
$$

noisy RAW 使用 Poisson-Gaussian 的高斯近似采样：

$$
\tilde{S}_c = S_c + \mathcal{N}\left(0, \sigma^2(x,c)\right)
$$

最后裁剪到 $[0,1]$。

### Analog Gain 与 ISO

模拟器定义：

$$
ISO = 100 \cdot AG
$$

其中 $AG$ 是 `camera.analog_gain`。

噪声标定参数位于 `base_analog_gain`。运行时缩放比例为：

$$
r = \frac{AG}{AG_{base}}
$$

$$
K_c(AG) = K_{c,base} \cdot r^{\alpha_{shot}}
$$

$$
\sigma_{read,c}(AG) = \sigma_{read,c,base} \cdot r^{\alpha_{read}}
$$

指数由 `shot_noise_gain_exponent` 和 `read_noise_gain_exponent` 控制。

对于 binning readout，`raw-sim` 还会根据 analog/digital binning 以及 average/sum 操作调整有效 shot noise 和 read noise。

## Batch 接口

公共 batch 接口为：

```python
from raw_sim.batch import srgb_to_raw_burst_batch

x, y = srgb_to_raw_burst_batch(
    rgb=batch_rgb,
    camera=camera,
    analog_gain=analog_gain,
    seed=seed,
    frames=3,
    noise_map_reduce="mean",
)
```

返回：

- `x`：noisy RAW burst 和 noise map，形状为 `(B, frames * raw_channels + 1, H/scale, W/scale)`。
- `y`：clean linear RGB target，形状为 `(B, 3, H, W)`。

JDD 直接调用该接口，因此 `raw-sim` 中的退化更新会同步影响训练数据。
