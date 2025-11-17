# ERP Warping & Quality Metrics for 360° YUV

This repository provides a **lightweight PyTorch pipeline** for ERP-based (Equirectangular Projection) view warping and quality evaluation on 360° video data.  
It includes YUV readers, depth-guided warping, simple inpainting, and WS-PSNR/SSIM metrics — designed for research in 360° image/video quality, rendering, and depth-based view synthesis.

---

## Features

- **YUV420 10-bit texture & 16-bit depth reader** (as `uint16`)
- **Quality metrics:** WS-PSNR, PSNR, SSIM for ERP domain


**Author:** [Yuan Yue](https://github.com/yuanaiya)  
**License:** MIT  
**Python:** 3.9+  
**PyTorch:** 2.x  

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yuanaiya/CASSMVS/erp-warp-360.git
cd erp-warp-360

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
---

## Dataset

All sample data used by this repo comes from the **MPEG-I Immersive Video (MIV) Content Database** (ISO/IEC MPEG official).

- Official page (registration required): https://mpeg-miv.org/index.php/content-database-2/
- Intended use: academic research and evaluation; follow the provider’s license terms.
- Sequences used in our demo: mainly **Classroom** and **Carpark**.

**Typical properties**
- Views: 9–15 per sequence; 17 frames per view
- Resolution: Classroom 4096×2048, Carpark 1920×1088
- Texture: YUV420 10-bit (yuv420p10le, read as uint16)

**Directory layout (example)**
```
datasets/
├─ Classroom/
│  ├─ Texture/  v0_texture_4096x2048_yuv420p10le.yuv, v1_*.yuv, ...
│  └─ Depth/    v0_depth_4096x2048_yuv420p16le.yuv,  v1_*.yuv, ...
└─ Carpark/
   ├─ Texture/  v0_texture_1920x1088_yuv420p10le.yuv, ...
   └─ Depth/    v0_depth_1920x1088_yuv420p16le.yuv,  ...
```
