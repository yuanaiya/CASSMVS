# -*- coding: utf-8 -*-

"""
ERP Warping & Quality Metrics for 360° YUV (10-bit/16-bit) with PyTorch
=======================================================================

This module provides:
1) YUV420 10-bit texture & 16-bit depth reader (as uint16 on device),
2) ERP-based view warping with depth + relative pose (R, T),
3) Simple hole inpainting (neighbor-average),
4) WS-PSNR / PSNR / SSIM metrics on ERP,
5) A minimal demo pipeline compatible with your paper.

Author: Yuan Yue
License: MIT
Python: 3.9+
PyTorch: 2.x
"""

from __future__ import annotations

import os
import numpy as np
from typing import Tuple, Optional, List

import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# ----------------------------
# Configuration dataclass
# ----------------------------

class Config:
    """
    Global configuration for image size / frame count / inverse depth range.

    Args:
        W: ERP width in pixels (e.g., 4096)
        H: ERP height in pixels (e.g., 2048)
        n_frames: Number of frames to read from each YUV file
        Rnear_inv: Inverse of near radius (1 / near_depth)
        Rfar_inv: Inverse of far radius (1 / far_depth)
    """
    def __init__(self,
                 W: int = 4096,
                 H: int = 2048,
                 n_frames: int = 17,
                 Rnear_inv: float = 1 / 0.8,
                 Rfar_inv: float = 1 / 1000):
        self.W = W
        self.H = H
        self.n_frames = n_frames
        self.Rnear_inv = Rnear_inv
        self.Rfar_inv = Rfar_inv


# ----------------------------
# IO: Read YUV420P10/16 as uint16 tensors
# ----------------------------

def yuv_read(file_path: str, config: Config, device: torch.device = DEVICE
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read YUV420 (10-bit packed into uint16) frames and upsample U/V to full res.

    Assumptions
    ----------
    - Layout: Y plane [H*W], then U [H/2*W/2], then V [H/2*W/2]
    - Depth files are also read with this function but only the Y plane is used.

    Returns
    -------
    (Y, U_up, V_up): float32 tensors on device with shapes
        Y:     [n_frames, H, W]
        U_up:  [n_frames, H, W]
        V_up:  [n_frames, H, W]
    """
    W, H, n_frames = config.W, config.H, config.n_frames
    frame_len = int(1.5 * W * H)  # Y + U + V for 4:2:0

    # Pre-allocate (host)
    y = np.zeros((n_frames, H, W), dtype=np.uint16)
    u = np.zeros((n_frames, H // 2, W // 2), dtype=np.uint16)
    v = np.zeros((n_frames, H // 2, W // 2), dtype=np.uint16)
    u_up = np.zeros((n_frames, H, W), dtype=np.uint16)
    v_up = np.zeros((n_frames, H, W), dtype=np.uint16)

    with open(file_path, 'rb') as f:
        stream = np.fromfile(f, dtype=np.uint16)

    # Safety check
    expected = n_frames * frame_len
    if stream.size < expected:
        raise ValueError(
            f"File {file_path} too small: {stream.size} < {expected} elements "
            f"for n_frames={n_frames}, H={H}, W={W}"
        )

    indices = np.arange(0, n_frames * frame_len, frame_len)

    for i in range(n_frames):
        frame = stream[indices[i]:indices[i] + frame_len]
        y[i] = frame[:W * H].reshape(H, W)
        u[i] = frame[W * H:W * H + (W * H) // 4].reshape(H // 2, W // 2)
        v[i] = frame[W * H + (W * H) // 4:].reshape(H // 2, W // 2)

        # Bicubic upsample to full-res for convenience in warping and metrics
        u_up[i] = np.array(Image.fromarray(u[i]).resize([W, H], resample=Image.BICUBIC))
        v_up[i] = np.array(Image.fromarray(v[i]).resize([W, H], resample=Image.BICUBIC))

    # Move to GPU as float32
    y_t = torch.tensor(y.astype(np.float32), device=device)
    u_t = torch.tensor(u_up.astype(np.float32), device=device)
    v_t = torch.tensor(v_up.astype(np.float32), device=device)

    return y_t, u_t, v_t

def pre_process(i_frame: int,
                Y_s: torch.Tensor, U_s: torch.Tensor, V_s: torch.Tensor
                ) -> torch.Tensor:
    """
    Stack single-frame Y/U/V into (N=1, C=3, H, W) tensor for grid_sample.

    Args:
        i_frame: frame index
        Y_s/U_s/V_s: [n_frames, H, W] float32 tensors

    Returns:
        yuv: [1, 3, H, W]
    """
    Y = Y_s[i_frame].unsqueeze(0)
    U = U_s[i_frame].unsqueeze(0)
    V = V_s[i_frame].unsqueeze(0)
    return torch.stack([Y, U, V], dim=1)

# ----------------------------
# Geometry: rotation utility
# ----------------------------

def euler_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert roll/pitch/yaw (XYZ) Euler angles (in radians) to rotation matrix.

    Args:
        euler_angles: [3] tensor (roll, pitch, yaw) on device

    Returns:
        3x3 rotation matrix (torch.float32) on the same device
    """
    roll, pitch, yaw = euler_angles
    c, s = torch.cos, torch.sin

    Rx = torch.tensor([[1, 0, 0],
                       [0, c(roll), -s(roll)],
                       [0, s(roll),  c(roll)]],
                      dtype=torch.float32, device=euler_angles.device)
    
    Ry = torch.tensor([[ c(pitch), 0, s(pitch)],
                       [0,         1, 0],
                       [-s(pitch), 0, c(pitch)]],
                      dtype=torch.float32, device=euler_angles.device)

    Rz = torch.tensor([[ c(yaw), -s(yaw), 0],
                       [ s(yaw),  c(yaw), 0],
                       [0,        0,      1]],
                      dtype=torch.float32, device=euler_angles.device)

    return Rz @ Ry @ Rx

# ----------------------------
# Core: ERP warping with depth
# ----------------------------

def warp_erp_yuv(pos_t: torch.Tensor,
                 pos_s: torch.Tensor,
                 rot_t: torch.Tensor,
                 rot_s: torch.Tensor,
                 yuv_tensor: torch.Tensor,
                 erp_s_disp_Y: torch.Tensor,
                 config: Config,
                 device: torch.device = DEVICE) -> torch.Tensor:
    """
    Warp a source ERP YUV frame into the target camera using source depth/disp.

    Args:
        pos_t: target position [3]
        pos_s: source position  [3]
        rot_t: target Euler (rad) [3]
        rot_s: source Euler (rad) [3]
        yuv_tensor: [1, 3, H, W] source texture (float32, 0..65535)
        erp_s_disp_Y: [H, W] source disparity (float32), same units as used
                      in R_s computation below
        config: global W/H and inverse-depth bounds
        device: torch device

    Returns:
        output: [3, H, W] warped YUV, bilinear-sampled (float32)
    """
    W, H = config.W, config.H
    Rnear_inv, Rfar_inv = config.Rnear_inv, config.Rfar_inv

    # Assume 10-bit content stored in uint16 range [0..65535].
    v_max = 65535.0

    # Relative pose T (s -> t) and R = Rt^T * Rs
    T = pos_s - pos_t
    R_s = euler_to_matrix(rot_s)
    R_t = euler_to_matrix(rot_t)
    R_mat = torch.matmul(R_t.T, R_s)

    # Build ERP lon/lat grid (target longitudes; source lat reused as param)
    n2, m2 = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.int32),
        torch.arange(W, device=device, dtype=torch.int32),
        indexing='ij'
    )

    lat_s = 0.5 * torch.pi - ((n2 + 0.5) / H) * torch.pi    # [-pi/2, pi/2]
    lng_t = 2 * torch.pi * ((m2 + 0.5) / W) - torch.pi      # [-pi, pi]

    # Convert disparity to radius (depth on sphere); linear mapping example:
    R_src = 1.0 / (((Rnear_inv - Rfar_inv) * erp_s_disp_Y / v_max) + Rfar_inv)

    # Spherical -> Cartesian for source pixel directions, but parameterized
    # by target longitude. This follows your original formulation.
    X_s = R_src * torch.cos(lat_s) * torch.cos(lng_t)
    Y_s = R_src * torch.sin(lat_s)
    Z_s = -R_src * torch.cos(lat_s) * torch.sin(lng_t)

    # Transform points into target camera: p_t = R * p_s + T
    pts_s = torch.stack((X_s.ravel(), Y_s.ravel(), Z_s.ravel()))
    pts_t = torch.matmul(R_mat, pts_s) + T[:, None]
    X_t, Y_t, Z_t = pts_t.reshape(3, H, W)

    # Back to lon/lat in the target
    R_t_est = torch.sqrt(X_t ** 2 + Y_t ** 2 + Z_t ** 2)  # (unused directly)
    lat_t = torch.arcsin(Y_t / (R_t_est + 1e-8))
    lng_t_est = torch.atan2(-Z_t, X_t)

    # Project to pixel coords in target ERP
    m1 = ((lng_t_est / (2 * torch.pi)) + 0.5) * W
    n1 = (0.5 - lat_t / torch.pi) * H

    # Clamp and round to int32—for nearest-like sampling grid (but we later
    # feed as float to grid_sample). Keeping clamp to valid range.
    m1 = torch.round(torch.clamp(m1, 0, W - 1)).to(torch.int32)
    n1 = torch.round(torch.clamp(n1, 0, H - 1)).to(torch.int32)

    # Assemble normalized sampling grid for grid_sample (x,y in [-1,1])
    grid = torch.stack((m1, n1), dim=-1).to(torch.float32)
    grid[..., 0] = (grid[..., 0] / ((W - 1.0) / 2.0)) - 1.0  # x
    grid[..., 1] = (grid[..., 1] / ((H - 1.0) / 2.0)) - 1.0  # y
    grid = grid[..., [0, 1]]  # already (x,y)

    out = F.grid_sample(
        input=yuv_tensor,              # [1, 3, H, W]
        grid=grid.unsqueeze(0),        # [1, H, W, 2]
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return out[0]  # [3, H, W]

# ----------------------------
# Utility: merge two warped images (simple valid/mean)
# ----------------------------

def merge_images(output1: torch.Tensor,
                 output2: torch.Tensor) -> torch.Tensor:
    """
    Merge two warped results by: nonzero preference + average on overlaps.

    Args:
        output1/output2: [3, H, W] float32

    Returns:
        merged: [3, H, W]
    """
    valid1 = output1 != 0
    valid2 = output2 != 0

    merged = torch.zeros_like(output1)
    merged[valid1 & ~valid2] = output1[valid1 & ~valid2]
    merged[valid2 & ~valid1] = output2[valid2 & ~valid1]
    merged[valid1 & valid2] = 0.5 * (output1[valid1 & valid2] + output2[valid1 & valid2])
    return merged

# ----------------------------
# Inpainting: neighbor-average, fixed iterations
# ----------------------------

def inpaint_holes(input_tensor: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  max_iterations: int = 10) -> torch.Tensor:
    """
    Fill zeros/NaNs by iterative neighbor averaging (3x3 mean).

    Args:
        input_tensor: [H, W] or [B, C, H, W] float32
        mask: same shape (1=known, 0=hole). If None, auto (val>=eps and not NaN)
        max_iterations: number of passes

    Returns:
        inpainted: same shape as input
    """
    x = input_tensor
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    eps = 1e-6
    if mask is None:
        mask = ~torch.isnan(x) & (x >= eps)
    else:
        mask = mask.bool()

    out = x.clone()
    out[~mask] = 0.0

    kernel = torch.ones(3, 3, device=x.device, dtype=torch.float32).view(1, 1, 3, 3)

    for _ in range(max_iterations):
        neigh_sum = F.conv2d(out, kernel, padding=1)
        neigh_cnt = F.conv2d(mask.float(), kernel, padding=1)
        neigh_cnt = torch.clamp(neigh_cnt, min=1.0)
        filled = neigh_sum / neigh_cnt
        out = torch.where(mask, out, filled)
        mask = mask | (neigh_cnt > 0)

    return out.squeeze(0).squeeze(0) if input_tensor.ndim == 2 else out

# ----------------------------
# Metrics: PSNR / WS-PSNR / SSIM
# ----------------------------

def psnr_my(img: torch.Tensor, imgn: torch.Tensor, bit_depth: int = 10) -> Tuple[float, float]:
    """
    PSNR between two tensors.

    Args:
        img/imgn: same shape, float32
        bit_depth: nominal bit depth (10 -> MAX=1023)

    Returns:
        (psnr, mse)
    """
    MAX = (2 ** bit_depth) - 1
    mse = torch.mean((img - imgn) ** 2)
    psnr = 10 * torch.log10((MAX ** 2) / (mse + 1e-12))
    return float(psnr.item()), float(mse.item())

def ws_psnr(target_Y: torch.Tensor,
           target_U: torch.Tensor,
           target_V: torch.Tensor,
           ref_Y: torch.Tensor,
           ref_U: torch.Tensor,
           ref_V: torch.Tensor,
           bit_depth: int = 10) -> Tuple[float, float, float, float]:
    """
    Weighted Spherical PSNR (WS-PSNR) for ERP images.

    All tensors are float32 with ranges consistent to their source (here uint16),
    but PSNR MAX uses 2^bit_depth-1 (e.g., 1023 for 10-bit).

    Shapes:
        Y: [H, W], U/V: [H/2, W/2]

    Returns:
        (PSNR_Y, PSNR_U, PSNR_V, final_PSNR)
    """

    MAX = (2 ** bit_depth) - 1
    H, W = target_Y.shape
    w = torch.zeros((H, W), device=target_Y.device)
    # cos weight along latitude
    h_t = torch.tensor(H, dtype=torch.float32, device=target_Y.device)
    for j in range(H):
        w[j, :] = torch.cos(((j + 0.5 - h_t / 2) * torch.pi) / h_t)

    w_uv = w[:H // 2, :W // 2]

    WMSE_Y = torch.sum((target_Y - ref_Y) ** 2 * w) / torch.sum(w)
    WMSE_U = torch.sum((target_U - ref_U) ** 2 * w_uv) / torch.sum(w_uv)
    WMSE_V = torch.sum((target_V - ref_V) ** 2 * w_uv) / torch.sum(w_uv)

    def _psnr(mse):
        return 10 * torch.log10((MAX ** 2) / (mse + 1e-12))

    PSNR_Y = _psnr(WMSE_Y)
    PSNR_U = _psnr(WMSE_U)
    PSNR_V = _psnr(WMSE_V)
    final = (6 * PSNR_Y + PSNR_U + PSNR_V) / 8.0

    return float(PSNR_Y.item()), float(PSNR_U.item()), float(PSNR_V.item()), float(final.item())

# ----------------------------
# IO: write warped result & evaluate metrics
# ----------------------------

def write_psnr(output: torch.Tensor,
               path: str,
               H: int, W: int,
               Y_ref: torch.Tensor, U_ref: torch.Tensor, V_ref: torch.Tensor,
               bit_depth: int = 10) -> bool:
    """
    Save warped result to YUV420 (uint16) and print PSNR/WS-PSNR/SSIM.

    Args:
        output: [3, H, W] float32
        path: file path to write
        Y_ref/U_ref/V_ref: reference full-res planes (U/V will be downsampled here)
    """
    Y_out = output[0]
    U_out = output[1]
    V_out = output[2]

    # 4:2:0 downsample for U/V (nearest stride as in original code)
    U_out_d = U_out[::2, ::2]
    V_out_d = V_out[::2, ::2]
    U_ref_d = U_ref[::2, ::2]
    V_ref_d = V_ref[::2, ::2]

    # Simple inpainting for zeros (optional but improves metrics stability)
    Y_fill = inpaint_holes(Y_out)
    U_fill = inpaint_holes(U_out_d)
    V_fill = inpaint_holes(V_out_d)


    # Write YUV420 as uint16
    with open(path, 'wb') as f:
        f.write(Y_fill.clamp(0, 65535).cpu().numpy().astype(np.uint16).tobytes())
        f.write(U_fill.clamp(0, 65535).cpu().numpy().astype(np.uint16).tobytes())
        f.write(V_fill.clamp(0, 65535).cpu().numpy().astype(np.uint16).tobytes())

    # Flatten to a single vector (Y + U + V) for PSNR like your original
    yuv_out = torch.cat([Y_fill.reshape(-1), U_fill.reshape(-1), V_fill.reshape(-1)])
    yuv_ref = torch.cat([Y_ref.reshape(-1), U_ref_d.reshape(-1), V_ref_d.reshape(-1)])

    psnr_all, mse = psnr_my(yuv_out, yuv_ref, bit_depth=bit_depth)
    print(f'PSNR (YUV420 combined, {bit_depth}-bit MAX): {psnr_all:.2f} dB')

    PSNR_Y, PSNR_U, PSNR_V, WS = ws_psnr(
        Y_fill, U_fill, V_fill, Y_ref, U_ref_d, V_ref_d, bit_depth=bit_depth
    )
    print(f'WS-PSNR: Y={PSNR_Y:.2f} U={PSNR_U:.2f} V={PSNR_V:.2f}  final={WS:.2f} dB')

    # SSIM per plane (uses 16-bit dynamic range)
    ssim_Y = ssim(Y_fill.cpu().numpy(), Y_ref.cpu().numpy(), data_range=65535)
    ssim_U = ssim(U_fill.cpu().numpy(), U_ref_d.cpu().numpy(), data_range=65535)
    ssim_V = ssim(V_fill.cpu().numpy(), V_ref_d.cpu().numpy(), data_range=65535)
    print(f'SSIM: Y={ssim_Y:.3f} U={ssim_U:.3f} V={ssim_V:.3f}')

    return True

# ----------------------------
# Demo pipeline (kept close to your original main)
# ----------------------------

def main() -> bool:
    """
    Minimal demo:
      - Read a target view (texture)
      - For each frame, warp two source views with their depths
      - Merge and (optionally) save/evaluate

    Adjust file paths to your dataset layout.
    """
    cfg = Config()

    # ---- target texture (reference for metrics)
    ref_tex_path = "../Content/ClassroomImage/v0_texture_4096x2048_yuv420p10le.yuv"
    Y_t, U_t, V_t = yuv_read(ref_tex_path, cfg, DEVICE)

    # Two source index lists (pairs)
    frame1 = [1, 2, 3, 7, 9, 10, 12]
    frame2 = [6, 5, 4, 8, 14, 13, 11]

    # Target camera (origin)
    pos_t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
    rot_t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)

    # Source rigs (positions) paired with frame1 & frame2 above
    positions1 = torch.tensor([
        [-0.0519615242,  0.03,  0.0],
        [-0.0519615242, -0.03,  0.0],
        [ 0.0,           0.06,  0.0],
        [ 0.0,           0.0,  -0.06],
        [-0.1039230485,  0.0,   0.0],
        [-0.0519615242,  0.09,  0.0],
        [ 0.0519615242,  0.09,  0.0]
    ], dtype=torch.float32, device=DEVICE)

    positions2 = torch.tensor([
        [ 0.0519615242, -0.03,  0.0],
        [ 0.0519615242,  0.03,  0.0],
        [ 0.0,          -0.06,  0.0],
        [ 0.0,           0.0,   0.06],
        [ 0.1039230485,  0.0,   0.0],
        [ 0.0519615242, -0.09,  0.0],
        [-0.0519615242, -0.09,  0.0]
    ], dtype=torch.float32, device=DEVICE)

    rot_s = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)

    # Optional output folder
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    # ---- iterate frames
    for i_frame in range(cfg.n_frames):
        for idx, (i, j) in enumerate(zip(frame1, frame2)):
            # Texture & depth paths for each source camera
            tex1 = f"../Content/ClassroomImage/v{i}_texture_4096x2048_yuv420p10le.yuv"
            dep1 = f"../Content/ClassroomImage/v{i}_depth_4096x2048_yuv420p16le.yuv"
            tex2 = f"../Content/ClassroomImage/v{j}_texture_4096x2048_yuv420p10le.yuv"
            dep2 = f"../Content/ClassroomImage/v{j}_depth_4096x2048_yuv420p16le.yuv"

            # Read depth (use Y only) and texture for both sources
            Y_r1, _, _ = yuv_read(dep1, cfg, DEVICE)
            Y_r2, _, _ = yuv_read(dep2, cfg, DEVICE)
            Y_s1, U_s1, V_s1 = yuv_read(tex1, cfg, DEVICE)
            Y_s2, U_s2, V_s2 = yuv_read(tex2, cfg, DEVICE)

            # Pack 1-frame YUV for grid_sample
            yuv1 = pre_process(i_frame, Y_s1, U_s1, V_s1)
            yuv2 = pre_process(i_frame, Y_s2, U_s2, V_s2)

            # Warp into target using source disparity (depth proxy) maps
            out1 = warp_erp_yuv(pos_t, positions1[idx], rot_t, rot_s, yuv1, Y_r1[i_frame], cfg, DEVICE)
            out2 = warp_erp_yuv(pos_t, positions2[idx], rot_t, rot_s, yuv2, Y_r2[i_frame], cfg, DEVICE)

            merged = merge_images(out1, out2)

            # (Optional) evaluate & save one example per pair
            out_path = os.path.join(out_dir, f"warp_f{i_frame:02d}_pair{idx:02d}.yuv")
            write_psnr(
                merged, out_path, cfg.H, cfg.W,
                Y_t[i_frame], U_t[i_frame], V_t[i_frame], bit_depth=10
            )

    return True

if __name__ == "__main__":
    # Keep the entrypoint minimal; users can import functions when integrating.
    main()

 
