# RTMPose-StarNetCA 模型架构详解

> 基于 RTMPose 框架，以 StarNet-S3 为骨干网络，融合 Coordinate Attention (CA) 机制的人体姿态估计模型。
> 输入尺寸: 192×256×3 (W×H×C)，输出: 17 个 COCO 关键点坐标

---

## 整体流水线 (TopdownPoseEstimator)

```
Input Image (256×192×3)
    │
    ▼
┌─────────────────────────┐
│   Data Preprocessor     │  RGB归一化: mean=[123.675, 116.28, 103.53]
│                         │             std=[58.395, 57.12, 57.375]
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│   Backbone: StarNetCA   │  创新点1: StarNet 轻量级骨干网络 (星操作)
│   (arch=S3)             │  创新点2: Coordinate Attention (Stage 2,3)
└─────────┬───────────────┘
          │ 输出: (B, 256, 8, 6)
          ▼
┌─────────────────────────┐
│   Head: RTMCCHead       │  SimCC 坐标分类
│   (GAU + SimCC)         │
└─────────┬───────────────┘
          ▼
    pred_x: (B, 17, 384)
    pred_y: (B, 17, 512)
```

---

## 一、Backbone: StarNetCA-S3 详细结构

### 1.1 Stem 层

```
Input: (B, 3, 256, 192)
    │
    ├── Conv2d(3, 32, k=3, s=2, p=1)      → (B, 32, 128, 96)
    ├── BatchNorm2d(32)
    └── ReLU6()
    │
Output: (B, 32, 128, 96)
```

### 1.2 Stage 1 — StarBlock ×2 (无 CA)

```
Input: (B, 32, 128, 96)
    │
    ├── DownSampler: ConvBN(32, 32, k=3, s=2, p=1)   → (B, 32, 64, 48)
    │       ├── Conv2d(32, 32, k=3, s=2, p=1)
    │       └── BatchNorm2d(32)
    │
    ├── StarBlock_0 (dim=32, mlp_ratio=4)
    │       详见 [1.5 StarBlock 内部结构]
    │
    └── StarBlock_1 (dim=32, mlp_ratio=4)
    │
Output: (B, 32, 64, 48)
```

### 1.3 Stage 2 — StarBlock ×2 (无 CA)

```
Input: (B, 32, 64, 48)
    │
    ├── DownSampler: ConvBN(32, 64, k=3, s=2, p=1)   → (B, 64, 32, 24)
    │       ├── Conv2d(32, 64, k=3, s=2, p=1)
    │       └── BatchNorm2d(64)
    │
    ├── StarBlock_0 (dim=64, mlp_ratio=4)
    │
    └── StarBlock_1 (dim=64, mlp_ratio=4)
    │
Output: (B, 64, 32, 24)
```

### 1.4 Stage 3 — CAStarBlock ×8 (带 Coordinate Attention) ★创新点

```
Input: (B, 64, 32, 24)
    │
    ├── DownSampler: ConvBN(64, 128, k=3, s=2, p=1)  → (B, 128, 16, 12)
    │       ├── Conv2d(64, 128, k=3, s=2, p=1)
    │       └── BatchNorm2d(128)
    │
    ├── CAStarBlock_0 (dim=128, mlp_ratio=4, ca_reduction=32)
    ├── CAStarBlock_1
    ├── CAStarBlock_2
    ├── CAStarBlock_3
    ├── CAStarBlock_4
    ├── CAStarBlock_5
    ├── CAStarBlock_6
    └── CAStarBlock_7
    │       每个 CAStarBlock 详见 [1.6 CAStarBlock 内部结构]
    │
Output: (B, 128, 16, 12)
```

### 1.5 Stage 4 — CAStarBlock ×4 (带 Coordinate Attention) ★创新点

```
Input: (B, 128, 16, 12)
    │
    ├── DownSampler: ConvBN(128, 256, k=3, s=2, p=1) → (B, 256, 8, 6)
    │       ├── Conv2d(128, 256, k=3, s=2, p=1)
    │       └── BatchNorm2d(256)
    │
    ├── CAStarBlock_0 (dim=256, mlp_ratio=4, ca_reduction=32)
    ├── CAStarBlock_1
    ├── CAStarBlock_2
    └── CAStarBlock_3
    │
Output: (B, 256, 8, 6)  ← out_indices=(3,) 取此输出
```

---

## 1.6 StarBlock 内部结构 (以 dim=D 为例)

```
Input: x (B, D, H, W)
    │
    ├── DWConv: Conv2d(D, D, k=7, s=1, p=3, groups=D) + BN
    │       → (B, D, H, W)    [深度可分离卷积，局部空间信息聚合]
    │
    ├── 分支1: f1 = Conv2d(D, 4D, k=1)     → (B, 4D, H, W)  [无BN]
    ├── 分支2: f2 = Conv2d(D, 4D, k=1)     → (B, 4D, H, W)  [无BN]
    │
    ├── ★ Star Operation: ReLU6(f1) * f2   → (B, 4D, H, W)
    │       [逐元素乘法 — StarNet 核心操作，隐式高阶特征交互]
    │
    ├── g: Conv2d(4D, D, k=1) + BN         → (B, D, H, W)   [通道压缩]
    │
    ├── DWConv2: Conv2d(D, D, k=7, s=1, p=3, groups=D)
    │       → (B, D, H, W)    [无BN，二次空间信息融合]
    │
    └── 残差连接: output = x + DropPath(dwconv2_out)
    │
Output: (B, D, H, W)
```

## 1.7 CAStarBlock 内部结构 (以 dim=D 为例) ★创新点

```
Input: x (B, D, H, W)
    │
    ├── DWConv: Conv2d(D, D, k=7, s=1, p=3, groups=D) + BN
    │       → (B, D, H, W)
    │
    ├── 分支1: f1 = Conv2d(D, 4D, k=1)     → (B, 4D, H, W)
    ├── 分支2: f2 = Conv2d(D, 4D, k=1)     → (B, 4D, H, W)
    │
    ├── ★ Star Operation: ReLU6(f1) * f2   → (B, 4D, H, W)
    │
    ├── g: Conv2d(4D, D, k=1) + BN         → (B, D, H, W)
    │
    ├── ★★ Coordinate Attention (CA)        → (B, D, H, W)
    │       详见 [1.8 CA 模块内部结构]
    │
    ├── DWConv2: Conv2d(D, D, k=7, s=1, p=3, groups=D)
    │       → (B, D, H, W)
    │
    └── 残差连接: output = x + DropPath(dwconv2_out)
    │
Output: (B, D, H, W)
```

## 1.8 Coordinate Attention (CA) 模块内部结构 ★创新点

```
Input: x (B, C, H, W)
    │
    ├─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
  pool_h: AdaptiveAvgPool2d((H, 1))        pool_w: AdaptiveAvgPool2d((1, W))
    │ → x_h: (B, C, H, 1)                    │ → x_w: (B, C, 1, W)
    │                                         │
    │                                    permute(0,1,3,2)
    │                                         │ → (B, C, W, 1)
    │                                         │
    └──────── cat(dim=2) ─────────────────────┘
                    │
                    ▼
              y: (B, C, H+W, 1)
                    │
    ┌───────────────┤  Shared Transform
    │  Conv2d(C, C//32, k=1)    [mid_channels = max(8, C//32)]
    │  BatchNorm2d(mid_channels)
    │  ReLU()
    │               │
    │          (B, mid_ch, H+W, 1)
    │               │
    │         split(dim=2, [H, W])
    │               │
    │     ┌─────────┴─────────┐
    │     ▼                   ▼
    │  x_h_out              x_w_out
    │  (B, mid_ch, H, 1)   (B, mid_ch, W, 1)
    │     │                   │
    │     │              permute(0,1,3,2)
    │     │                   │ → (B, mid_ch, 1, W)
    │     ▼                   ▼
    │  Conv2d(mid_ch, C, 1)  Conv2d(mid_ch, C, 1)
    │  Sigmoid()             Sigmoid()
    │     │                   │
    │  attn_h: (B,C,H,1)   attn_w: (B,C,1,W)
    │     │                   │
    └─────┴───────────────────┘
                    │
                    ▼
        out = x * attn_h * attn_w     [逐元素乘法，双向注意力加权]
                    │
Output: (B, C, H, W)

注: conv_h 和 conv_w 采用零初始化策略，
    使 sigmoid 初始输出≈0.5，CA 模块初始时近似恒等映射，
    保证预训练权重的兼容性。
```

---

## 二、Head: RTMCCHead 详细结构

```
Input: feats[-1] → (B, 256, 8, 6)   [取 backbone 最后一个 stage 输出]
    │
    │ ① Final Layer (大核卷积)
    ├── Conv2d(256, 17, k=7, s=1, p=3)
    │       → (B, 17, 8, 6)          [每个通道对应一个关键点]
    │
    │ ② Flatten
    ├── flatten(dim=2)
    │       → (B, 17, 48)            [H×W = 8×6 = 48]
    │
    │ ③ MLP (维度变换)
    ├── ScaleNorm(48)                 [L2归一化 × 可学习缩放因子]
    ├── Linear(48, 256, bias=False)
    │       → (B, 17, 256)
    │
    │ ④ Gated Attention Unit (GAU) — RTMCCBlock
    ├── RTMCCBlock(
    │       num_token=17,
    │       in_token_dims=256,
    │       out_token_dims=256,
    │       s=128,                    [注意力特征维度]
    │       expansion_factor=2,       [e = 256×2 = 512]
    │       act_fn='SiLU',
    │       use_rel_bias=False,
    │       pos_enc=False
    │   )
    │   详见 [2.1 GAU 内部结构]
    │       → (B, 17, 256)
    │
    │ ⑤ SimCC 坐标分类器
    ├── cls_x: Linear(256, 384, bias=False)   [W × split_ratio = 192×2]
    ├── cls_y: Linear(256, 512, bias=False)   [H × split_ratio = 256×2]
    │
Output:
    pred_x: (B, 17, 384)   → X方向 1D 概率分布
    pred_y: (B, 17, 512)   → Y方向 1D 概率分布
```

### 2.1 RTMCCBlock (Gated Attention Unit) 内部结构

```
Input: x (B, 17, 256)
    │
    ├── res_shortcut = x              [残差分支]
    │
    │ ① ScaleNorm
    ├── ScaleNorm(256, eps=1e-5)
    │       scale = 256^(-0.5)
    │       x_norm = x / ||x|| * scale
    │       → (B, 17, 256)
    │
    │ ② 线性投影 + 激活
    ├── Linear(256, 2×512+128 = 1152, bias=False)
    ├── SiLU()
    │       → (B, 17, 1152)
    │
    │ ③ 拆分 u, v, base
    ├── split → u: (B, 17, 512)      [门控向量]
    │          v: (B, 17, 512)      [值向量]
    │          base: (B, 17, 128)   [注意力基向量]
    │
    │ ④ 生成 Q, K
    ├── base.unsqueeze(2) * gamma + beta
    │       gamma, beta: 可学习参数 (2, 128)
    │       → (B, 17, 2, 128)
    ├── unbind(dim=2) → q: (B, 17, 128), k: (B, 17, 128)
    │
    │ ⑤ 注意力计算
    ├── qk = bmm(q, k^T)             → (B, 17, 17)
    ├── kernel = ReLU(qk / √128)²    [平方ReLU注意力]
    │       → (B, 17, 17)
    │
    │ ⑥ 门控聚合
    ├── x = u * bmm(kernel, v)        → (B, 17, 512)
    │       [门控向量 × 注意力加权值]
    │
    │ ⑦ 输出投影
    ├── Linear(512, 256, bias=False)  → (B, 17, 256)
    │
    │ ⑧ 残差连接
    └── output = Scale(res_shortcut) + DropPath(x)
    │
Output: (B, 17, 256)
```

---

## 三、损失函数

```
KLDiscretLoss:
    ├── 对 pred_x, pred_y 做 log_softmax
    ├── 对 gt_x, gt_y 做 softmax (label_softmax=True)
    ├── KL散度: KL(gt || pred) × beta(=10.0)
    └── 加权: × keypoint_weights (关键点可见性权重)
```

---

## 四、Backbone 各 Stage 特征图尺寸汇总

| 层级 | 模块 | 输出尺寸 (H×W×C) | 下采样率 | Block 类型 | Block 数量 |
|------|------|-------------------|----------|------------|------------|
| Input | — | 256×192×3 | 1× | — | — |
| Stem | Conv+BN+ReLU6 | 128×96×32 | 2× | — | — |
| Stage 1 | DownSample + StarBlock | 64×48×32 | 4× | StarBlock | 2 |
| Stage 2 | DownSample + StarBlock | 32×24×64 | 8× | StarBlock | 2 |
| Stage 3 | DownSample + CAStarBlock | 16×12×128 | 16× | CAStarBlock (★CA) | 8 |
| Stage 4 | DownSample + CAStarBlock | 8×6×256 | 32× | CAStarBlock (★CA) | 4 |

---

## 五、两个创新点总结

### 创新点 1: StarNet 骨干网络替换 CSPNeXt

- 原始 RTMPose 使用 CSPNeXt 作为骨干网络
- 本文替换为 StarNet-S3，核心是 **Star Operation (星操作)**：`ReLU6(f1) × f2`
- 通过逐元素乘法实现隐式高阶特征交互，无需显式注意力机制
- 优势：计算量更低，推理速度更快，适合移动端部署

### 创新点 2: Coordinate Attention 融合

- 在 StarNet 的 Stage 3 和 Stage 4 中引入 Coordinate Attention
- CA 将通道注意力分解为水平 (H) 和垂直 (W) 两个 1D 编码
- 保留精确的空间位置信息，对人体关键点定位尤为关键
- 零初始化策略确保与 StarNet 预训练权重的兼容性
- 仅在深层 stage 使用 CA，浅层保持原始 StarBlock，平衡精度与效率
