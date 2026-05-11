<div align="center">
  <h1>🌟 StarNet-Pose</h1>
  <h3>Heatmap-Free Lightweight Pose Estimation via Multiplicative Feature Interaction and Occlusion-Aware Training</h3>

  <p>
    <a href="https://arxiv.org/abs/2403.19967"><img src="https://img.shields.io/badge/Backbone-StarNet-blue" alt="StarNet"></a>
    <a href="https://arxiv.org/abs/2103.02907"><img src="https://img.shields.io/badge/Module-Coordinate%20Attention-green" alt="Coordinate Attention"></a>
    <a href="https://github.com/open-mmlab/mmpose"><img src="https://img.shields.io/badge/Built_on-MMPose-orange" alt="MMPose"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-darkgreen" alt="License"></a>
    <a href="https://www.sciencedirect.com/journal/neurocomputing"><img src="https://img.shields.io/badge/Journal-Neurocomputing-red" alt="Neurocomputing"></a>
    <br>
    <a href="https://github.com/lechan775/starnet-pose"><img src="https://img.shields.io/github/stars/lechan775/starnet-pose?style=social" alt="GitHub stars"></a>
  </p>

  <p>
    <a href="#-installation">🛠️ Installation</a> |
    <a href="#-quick-start">🚀 Quick Start</a> |
    <a href="#-model-zoo">📊 Model Zoo</a> |
    <a href="#-ablation-studies">📈 Ablations</a> |
    <a href="#-citation">📜 Citation</a>
  </p>
</div>

---

## 📄 Introduction

**StarNet-Pose** is an efficient 2D human pose estimation framework that leverages the lightweight **StarNet** backbone based on element-wise multiplicative feature interaction. It achieves state-of-the-art performance among lightweight top-down pose estimators while running at **up to 313.8 FPS**.

The core contributions:

- **StarNet Backbone**: A lightweight CNN architecture based on the **"Star Operation"** (element-wise multiplication of two feature branches), achieving strong feature representation with significantly fewer parameters than conventional backbones.
- **Multiplicative Feature Interaction**: Replaces traditional additive feature fusion (e.g., skip connections in HRNet) with multiplicative interaction, enabling richer feature representation at lower computational cost.
- **Occlusion-Aware Training**: Systematic evaluation of CoarseDropout augmentation strategies for improving robustness under heavy occlusion.
- **Two Efficient Variants**: **StarNet-Pose-T** (3.5M params, 72.00 AP, 313.8 FPS) and **StarNet-Pose-S** (6.4M params, 72.99 AP, 173.5 FPS).

This repository provides the official implementation of **"StarNet-Pose: Heatmap-Free Lightweight Pose Estimation via Multiplicative Feature Interaction and Occlusion-Aware Training"**, accepted at **Neurocomputing**.

> 🔗 **Built upon [MMPose](https://github.com/open-mmlab/mmpose)** — the OpenMMLab Pose Estimation Toolbox.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🪶 **Lightweight** | Two variants: **3.5M** (Tiny) and **6.4M** (Small) parameters |
| ⚡ **Ultra-Fast** | Up to **313.8 FPS** (Tiny) on RTX 5090 |
| 🎯 **High Precision** | **72.99 AP** (Small) on COCO val2017, surpassing RTMPose-s by **+1.4 AP** |
| 🛡️ **Occlusion Robust** | Occlusion-aware training via CoarseDropout augmentation |
| 🔌 **Plug-and-Play** | Compatible with RTMPose head and full MMPose ecosystem |
| 📦 **Two Variants** | StarNet-Pose-T (Tiny) and StarNet-Pose-S (Small) |

## 🛠️ Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.8 ([Install Guide](https://pytorch.org/get-started/locally/))
- CUDA (optional, for GPU training)

### Step 1: Install MMPose

StarNet-Pose is built upon MMPose. First install the base framework:

```bash
# Install MMEngine & MMCV
pip install mmengine mmcv

# Install MMPose (the full framework)
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
cd ..
```

### Step 2: Install StarNet-Pose

```bash
git clone https://github.com/lechan775/starnet-pose.git
cd starnet-pose

# Install dependencies
pip install -r requirements.txt

# Copy backbone modules into your MMPose installation
cp -r mmpose/models/backbones/* <mmpose_install_path>/mmpose/models/backbones/
cp -r configs/body_2d_keypoint/rtmpose/coco/* <mmpose_install_path>/configs/body_2d_keypoint/rtmpose/coco/
```

### Step 3: Download Pre-trained Weights

Pre-trained StarNet backbone weights (ImageNet-1K) and StarNet-Pose model checkpoints are distributed via **GitHub Releases**.

```bash
# StarNet-S1 backbone (for StarNet-Pose-T)
wget https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar

# StarNet-S3 backbone (for StarNet-Pose-S)
wget https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar
```

> Full StarNet-Pose checkpoints are available at [GitHub Releases](https://github.com/lechan775/starnet-pose/releases).

## 🚀 Quick Start

### Standalone StarNet (no MMPose dependency)

```python
import torch
from starnet import starnet_s3

# Load pre-trained StarNet-S3 (ImageNet-1K)
model = starnet_s3(pretrained=True)
model.eval()

x = torch.randn(1, 3, 224, 224)
feat = model(x)  # 1000-way ImageNet logits
```

### Training on COCO

```bash
cd <mmpose_install_path>

# Train StarNet-Pose-S (recommended: 8 GPUs)
GPUS=8 bash tools/dist_train.sh \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py 8

# Fast evaluation on COCO-mini
python tools/train.py \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnetca-s3_8xb256-420e_coco-256x192_mini.py
```

### Evaluation

```bash
# Evaluate trained model on COCO val2017
python tools/test.py \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py \
    <path_to_checkpoint>.pth
```

### FLOPs & Speed Benchmark

```bash
# Compute FLOPs and parameter count
python tools/analysis_tools/get_flops.py \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py

# Measure inference latency and FPS
python tools/benchmark_latency.py \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py \
    <path_to_checkpoint>.pth
```

## 📊 Model Zoo

### Main Results on COCO val2017 (256×192)

| Model | Backbone | Params (M) | GFLOPs | AP | AP⁵⁰ | AP⁷⁵ | AR | Latency (ms) | FPS |
|-------|----------|------------|--------|-----|------|------|-----|-------------|-----|
| **StarNet-Pose-T** | StarNet-S1 | **3.504** | **0.435** | **72.00** | **91.51** | **79.62** | **75.03** | **3.19** | **313.8** |
| **StarNet-Pose-S** | StarNet-S3 + CA | **6.428** | **0.765** | **72.99** | **91.64** | **80.69** | **76.01** | **5.76** | **173.5** |

> ⚡ FPS measured on **NVIDIA RTX 5090** under unified PyTorch/MMPose forward benchmark (batch=1). StarNet-Pose-T surpasses RTMPose-t by **+3.8 AP** while running at a higher frame rate.

### Comparison with Lightweight SOTA (COCO val2017, 256×192)

| Method | Backbone | Params (M) | GFLOPs | AP | AP⁵⁰ | AP⁷⁵ | AR |
|--------|----------|------------|--------|-----|------|------|-----|
| RTMPose-t | CSPNeXt-t | 3.34 | 0.360 | 68.20 | 88.30 | 75.90 | 73.60 |
| RTMPose-s | CSPNeXt-s | 5.47 | 0.680 | 71.60 | 89.20 | 78.90 | 76.80 |
| Lite-HRNet-30 | Lite-HRNet-30 | 1.80 | 0.319 | 67.20 | 88.00 | 75.00 | 73.30 |
| X-HRNet-30 | X-HRNet-30 | 2.10 | 0.300 | 67.40 | 87.50 | 75.40 | 73.50 |
| LMFormer-L | LMFormer-L | 4.10 | 1.400 | 68.90 | 88.30 | 76.40 | 74.70 |
| LGM-Pose | LGM-Pose | 1.10 | 0.600 | 69.30 | 89.50 | 76.20 | 73.70 |
| **StarNet-Pose-T** | StarNet-S1 | 3.504 | 0.435 | **72.00** | **91.51** | **79.62** | **75.03** |
| **StarNet-Pose-S** | StarNet-S3 + CA | 6.428 | 0.765 | **72.99** | **91.64** | **80.69** | **76.01** |

### Efficiency-Oriented Comparison

| Model | Params (M) | FLOPs (G) | AP | AP⁷⁵ | AR | Latency (ms) | FPS |
|-------|------------|-----------|-----|------|-----|-------------|-----|
| Lite-HRNet-18 | **1.10** | 0.205 | 64.80 | 73.00 | 71.20 | 18.14 | 55.1 |
| RTMPose-t | 3.34 | 0.360 | 68.20 | 75.90 | 73.60 | 3.51 | 285.1 |
| **StarNet-Pose-T** | 3.504 | 0.435 | **72.00** | **79.62** | **75.03** | **3.19** | **313.8** |
| Lite-HRNet-30 | **1.80** | 0.319 | 67.20 | 75.00 | 73.30 | 30.38 | 32.9 |
| RTMPose-s | 5.47 | 0.680 | 71.60 | 78.90 | **76.80** | **3.71** | **269.6** |
| **StarNet-Pose-S** | 6.428 | 0.765 | **72.99** | **80.69** | 76.01 | 5.76 | 173.5 |

### Model Specification

| Component | StarNet-Pose-T (Tiny) | StarNet-Pose-S (Small) |
|-----------|----------------------|------------------------|
| Stem | 3×3 Conv-BN-ReLU6, stride 2 | 3×3 Conv-BN-ReLU6, stride 2 |
| Stage 1 | 24 ch, depth 2, no CA | 32 ch, depth 2, no CA |
| Stage 2 | 48 ch, depth 2, no CA | 64 ch, depth 2, no CA |
| Stage 3 | 96 ch, depth 8, no CA | 128 ch, depth 8, CA enabled |
| Stage 4 | 192 ch, depth 3, no CA | 256 ch, depth 4, CA enabled |
| Head | RTMCCHead (SimCC), in-ch 192 | RTMCCHead (SimCC), in-ch 256 |

## 📈 Ablation Studies

### Attention Mechanism Ablation (COCO-mini, 210 epochs)

| Configuration | AP | AP⁵⁰ | AP⁷⁵ | AR | Best AP (epoch) |
|---------------|-----|------|------|-----|-----------------|
| StarNet baseline (w/o attention) | **50.62** | 77.94 | **54.08** | **54.42** | 50.67 (180) |
| StarNet + CBAM | 49.54 | 77.21 | 54.47 | 53.35 | 50.21 (190) |
| StarNet + CA | 49.65 | **78.15** | 51.26 | 53.46 | 50.60 (170) |

### Cross-Scale Component Ablation (COCO-mini, S1/Tiny level)

| Configuration | Backbone | CoarseDropout | AP |
|---------------|----------|---------------|-----|
| RTMPose-T baseline | CSPNeXt-Tiny | 1.0→0.5 schedule | 42.14 |
| StarNet-S1 (w/o CA) | StarNet-S1 | 1.0→0.5 schedule | 47.30 |
| StarNetCA-S1 | StarNet-S1 | 1.0→0.5 schedule | 46.86 |
| StarNetCA-S1 + fixed dropout | StarNet-S1 | Fixed p=0.6 | 47.13 |
| StarNet-S1 + fixed dropout | StarNet-S1 | Fixed p=0.6 | **47.48** |

> To reproduce these experiments: `bash experiments/occlusion_prob_study/run_fast_track_experiments.sh`

## 🏗️ Architecture

```
Input (256×192×3)
    │
    ▼
┌──────────────┐
│  Stem (Conv) │  3×3 Conv-BN-ReLU6, stride=2
└──────┬───────┘
       ▼
┌──────────────┐
│  Stage 1     │  StarBlock ×N₁  (no CA)
│  (64×48)     │  
└──────┬───────┘
       ▼  Downsample (3×3 Conv, stride=2)
┌──────────────┐
│  Stage 2     │  StarBlock ×N₂  (no CA)
│  (32×24)     │
└──────┬───────┘
       ▼  Downsample
┌──────────────┐
│  Stage 3     │  StarBlock ×N₃  (CA enabled in S variant)
│  (16×12)     │
└──────┬───────┘
       ▼  Downsample
┌──────────────┐
│  Stage 4     │  StarBlock ×N₄  (CA enabled in S variant)
│  (8×6)       │
└──────┬───────┘
       ▼
┌──────────────┐
│  RTMCC Head  │  SimCC-based coordinate classification
└──────┬───────┘
       ▼
   Keypoints (17 × 2·(W+H) logits)
```

The **StarBlock** core operation — multiplicative feature interaction:

```
         ┌─────────┐          ┌─────────┐
   x ───▶│ DWConv  │──▶ f₁ ─▶│ ReLU6   │──┐
         │  7×7    │          └─────────┘  │  element-wise
         └─────────┘                       ├──▶ multiply ──▶ g ──▶ DWConv2 ──▶ + ──▶ out
         ┌─────────┐                       │                        ▲
         │ DWConv  │──▶ f₂ ───────────────┘                        │
         │  7×7    │                                   input ──────┘ (residual)
         └─────────┘
```

**Coordinate Attention** (applied after the `g` projection in StarNet-Pose-S):

```
x (B,C,H,W) ──┬──▶ Pool H (B,C,H,1) ──┐
               │                        ├──▶ Concat ──▶ Conv1×1+BN+ReLU ──▶ Split
               └──▶ Pool W (B,C,1,W) ──┘       │                    │
                                                ▼                    ▼
                                          ConvH → Sigmoid     ConvW → Sigmoid
                                                │                    │
                                                └────▶ x × attnH × attnW ──▶ out
```

See `figures/` for detailed architecture diagrams (`.drawio` format).

## 📁 Repository Structure

```
starnet-pose/
├── starnet.py                    # Standalone StarNet implementation (no MMPose dep.)
├── mmpose/models/backbones/
│   ├── starnet.py                # StarNet backbone for MMPose (StarNet-S1/S3)
│   ├── starnet_ca.py             # StarNetCA backbone (StarNet + Coordinate Attention)
│   └── utils/
│       └── coordinate_attention.py  # Coordinate Attention module (CVPR 2021)
├── configs/body_2d_keypoint/rtmpose/coco/
│   ├── rtmpose_starnet-s3_*.py            # StarNet-Pose-S (vanilla StarNet-S3)
│   ├── rtmpose_starnetca-s3_*.py          # StarNet-Pose-S (StarNet-S3 + CA stages 3-4)
│   ├── rtmpose_starnetca-s3_*_a800.py     # A800-optimized config
│   └── rtmpose_*_mini.py                  # COCO-mini fast evaluation configs
├── experiments/occlusion_prob_study/
│   ├── datasets.py               # Dataset path configuration
│   ├── experiment_matrix.py      # Experiment suite definitions
│   ├── generate_configs.py       # Config generator
│   ├── run_fast_track_experiments.sh  # Fast evaluation pipeline (~2 hours)
│   ├── run_all_paper_experiments.sh   # Full experiment suite
│   └── generated_configs/        # Auto-generated experiment .py files
├── tools/
│   ├── train.py                  # Training launch script
│   ├── benchmark_latency.py      # FPS & latency benchmark
│   ├── test_starnet_*.py         # Unit tests
│   └── visualize_*.py            # Visualization utilities
├── demo/                         # Demo examples
├── figures/                      # Architecture diagrams (.drawio)
├── requirements.txt              # Python dependencies
├── LICENSE                       # Apache 2.0 License
├── CITATION.cff                  # Citation metadata
└── README.md                     # This file
```

## 🤝 Acknowledgements

This work builds upon several excellent open-source projects:

- **[StarNet / Rewrite the Stars](https://github.com/ma-xu/Rewrite-the-Stars)** (CVPR 2024) — The StarNet backbone architecture by Xu Ma et al.
- **[Coordinate Attention](https://arxiv.org/abs/2103.02907)** (CVPR 2021) — The CA module by Qibin Hou et al.
- **[MMPose](https://github.com/open-mmlab/mmpose)** — The OpenMMLab Pose Estimation Toolbox.
- **[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)** — Real-time multi-person pose estimation framework.

## 📜 Citation

If you use StarNet-Pose in your research, please cite:

```bibtex
@article{pan2025starnetpose,
  title   = {StarNet-Pose: Heatmap-Free Lightweight Pose Estimation via
             Multiplicative Feature Interaction and Occlusion-Aware Training},
  author  = {Pan, Li},
  journal = {Neurocomputing},
  year    = {2025},
  url     = {https://github.com/lechan775/starnet-pose}
}
```

Also consider citing the foundational works:

```bibtex
@inproceedings{ma2024starnet,
  title   = {Rewrite the Stars},
  author  = {Ma, Xu and ...},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition (CVPR)},
  year    = {2024}
}

@inproceedings{hou2021ca,
  title   = {Coordinate Attention for Efficient Mobile Network Design},
  author  = {Hou, Qibin and Zhou, Daquan and Feng, Jiashi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition (CVPR)},
  year    = {2021}
}

@misc{mmpose2020,
  title  = {OpenMMLab Pose Estimation Toolbox and Benchmark},
  author = {MMPose Contributors},
  year   = {2020},
  url    = {https://github.com/open-mmlab/mmpose}
}
```

## 📝 License

This project is released under the [Apache 2.0 License](LICENSE).

> **Note**: The StarNet backbone code is adapted from [Rewrite-the-Stars](https://github.com/ma-xu/Rewrite-the-Stars). The Coordinate Attention module is adapted from [CoordAttention](https://arxiv.org/abs/2103.02907). This project inherits the Apache 2.0 license from MMPose.

## 🔗 Links

- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [Papers with Code — Pose Estimation](https://paperswithcode.com/task/pose-estimation)
- [COCO Keypoint Dataset](https://cocodataset.org/#keypoints)
- [OCHuman Dataset](https://github.com/liruilong940607/OCHumanApi)

---

<div align="center">
  <sub>Built with ❤️ upon the MMPose ecosystem</sub>
</div>
