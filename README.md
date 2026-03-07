
# GR-DSW: Generative-Resilient Dual-Stream Watermarking

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

Official PyTorch implementation of the framework proposed in: **"An Adaptive Generative-Resilient Dual-Stream Watermarking Framework for Image Authentication and Semantic Recovery"** (Under Review).

## Overview
Traditional spatial watermarking schemes suffer from catastrophic failure under extreme localized tampering (e.g., >25% cropping) and global signal degradation. **GR-DSW** introduces a novel dual-stream architecture that bridges the gap between cryptographic fragility and generative artificial intelligence.

By decoupling semantic memory from spatial coordinates via a Vision Transformer (ViT) bottleneck, and distributing it redundantly across the high-frequency Discrete Wavelet Transform (DWT) domain, the framework achieves unprecedented state-of-the-art recovery.

### Key Features
* **Hyper-Redundant Latent Memory:** Compresses host images into a 256-bit latent representation using a ViT autoencoder, embedded redundantly (16x) in the $cH2$ DWT subband.
* **Hyper-Lorenz Chaotic Encryption:** Secures the latent payload against unauthorized extraction.
* **Extreme Spatial Recovery:** Achieves **>34 dB PSNR** under massive localized data annihilation (e.g., 50% continuous cropping, semantic splicing).
* **Tamper-Aware Circuit Breaker:** Utilizes a dynamic spatial density check ($k > 50$ clusters) to identify global degradation (JPEG, Gaussian Noise), actively bypassing the AI to prevent mode collapse and ensuring graceful degradation (~28 dB via median filtering).

## Repository Structure
```text
GR-DSW/
├── gr_dsw/
│   ├── models/            # ViT Autoencoder architecture
│   ├── crypto/            # Hyper-Lorenz chaotic sequences
│   ├── watermark/         # DWT Embedding, Fragile LSB, and Extractor
│   └── utils/             # PSNR, SSIM, and visualization metrics
├── scripts/
│   ├── train_vit.py                       # Script to train/generate ViT weights
│   └── evaluate_comprehensive_attacks.py  # Main pipeline execution
├── RawImages/             # Standard datasets (Lena, Peppers, etc.)
├── Results/               # Output directory for generated PDF reports
├── README.md
└── requirements.txt
```

## Installation & Model Setup

1. **Clone the repository:**

```bash
git clone https://github.com/aryankanojia/GR-DSW.git
cd GR-DSW
```

2. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

3. **Generate Pre-trained ViT Weights:**

To keep the repository lightweight, the large `.pth` model weights are not included. Before running the evaluation, you must generate the pre-trained weights locally by running the initialization/training script:

```bash
python scripts/train_vit.py
```

*(Note: This will output `pretrained_vit.pth` into the `gr_dsw/models/` directory).*

## Usage & Evaluation

To reproduce the experimental results and generate the comprehensive SOTA attack report (PDF), place your test images in the `RawImages/` directory and execute the automated evaluation pipeline:

```bash
python scripts/evaluate_comprehensive_attacks.py
```

This script will automatically:

1. Embed the dual-stream watermark ($\alpha = 8.0$, achieving ~40.0 dB PSNR).
2. Subject the images to a rigorous suite of attacks (Cropping, Grid Tampering, Semantic Splicing, JPEG Compression, Gaussian Noise).
3. Route the attacked images through the Dual-Condition Circuit Breaker.
4. Output a multi-page PDF report (`Results/GR_DSW_Comprehensive_Attacks_Report.pdf`) containing the original, attacked, tamper maps, and final recovered images with quantitative metrics.

<!-- ## Citation

If you find this code or framework useful in your research, please cite our paper:

```bibtex
@article{kanojia2026grdsw,
  title={An Adaptive Generative-Resilient Dual-Stream Watermarking Framework for Image Authentication and Semantic Recovery},
  author={Kanojia, Aryan},
  journal={IEEE Transactions on Information Forensics and Security (Under Review)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. Copyright (c) 2026 Aryan Kanojia. -->
