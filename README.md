# ERP Warping & Quality Metrics for 360° YUV (10-bit/16-bit) with PyTorch

This repo provides:
- YUV420 10-bit texture & 16-bit depth reader (as `uint16`),
- ERP-based view warping with depth + relative pose (R, T),
- Simple hole inpainting (neighbor-average),
- WS-PSNR / PSNR / SSIM metrics on ERP,
- A minimal demo pipeline compatible with the paper.

**Author:** Yuan Yue  
**License:** MIT  
**Python:** 3.9+  
**PyTorch:** 2.x  
**OS:** Linux/Windows/Mac (CUDA optional)

## Install

```bash
git clone https://github.com/yuanaiya/CASSMVS/erp-warp-360.git
cd erp-warp-360
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
