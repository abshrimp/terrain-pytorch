#!/usr/bin/env python3
"""
expand.py  無限地形拡張

使い方:
  python expand.py --ckpt checkpoints/best_G.pt \
                   --seed-pkl your_model.pkl \
                   --size 8   # 8×8グリッドを生成
                   --out  out_infinite/

仕組み:
  1. (0,0) を既存StyleGANで生成（シード）
  2. (1,0),(2,0)... → 左パッチが既知 → 上・左上は zero padding
  3. (0,1),(1,1)... → 上パッチが既知 → ラスタースキャンで順次予測
  4. 全パッチが揃ったら結合してhillshade保存
"""

import os, math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import dnnlib, legacy
import model as M

DEVICE = (torch.device("cuda")   if torch.cuda.is_available() else
          torch.device("mps")    if torch.backends.mps.is_available() else
          torch.device("cpu"))

PATCH_PX    = 512   # 学習時と合わせる
MAX_ELEV    = 1600.0


def load_expander(ckpt_path, base_ch=32):
    G = M.TerrainExpander(base_ch).to(DEVICE)
    G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    G.eval()
    return G


def seed_patch_from_stylegan(G_style, seed, device, patch_px=512):
    """既存StyleGANで最初のパッチを生成"""
    import legacy
    rng = np.random.RandomState(seed)
    z   = torch.from_numpy(rng.randn(1, G_style.z_dim)).float().to(device)
    lbl = torch.zeros([1, G_style.c_dim], device=device)
    with torch.no_grad():
        img = G_style(z, lbl, truncation_psi=0.7)
    p = img[0].mean(0).cpu().numpy()
    p = (p + 1) / 2                         # [-1,1] → [0,1]
    if p.shape[0] != patch_px:
        p = cv2.resize(p, (patch_px, patch_px), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def expand_grid(expander, G_style, grid_size, seed, style_pkl):
    """
    grid_size × grid_size のパッチグリッドを生成
    grid[iy][ix] = [1,1,512,512] tensor
    """
    N = grid_size
    grid = [[None]*N for _ in range(N)]
    zero = torch.zeros(1, 1, PATCH_PX, PATCH_PX, device=DEVICE)

    print(f"Generating {N}×{N} grid (seed={seed})...")

    # (0,0) はStyleGANで生成
    grid[0][0] = seed_patch_from_stylegan(G_style, seed, DEVICE, PATCH_PX).to(DEVICE)
    print(f"  [0,0] seeded from StyleGAN")

    # ラスタースキャン順で予測
    for iy in range(N):
        for ix in range(N):
            if iy == 0 and ix == 0:
                continue  # 既に生成済み

            topleft = grid[iy-1][ix-1] if iy>0 and ix>0 else zero
            top     = grid[iy-1][ix]   if iy>0          else zero
            left    = grid[iy][ix-1]   if ix>0          else zero

            with torch.no_grad():
                pred = expander(topleft, top, left)

            grid[iy][ix] = pred
            print(f"  [{iy},{ix}] predicted  "
                  f"min={pred.min():.3f} max={pred.max():.3f}")

    return grid


def stitch_grid(grid, overlap_px=64):
    """
    グリッドをコサイン窓ブレンドで合成
    overlap_px: パッチ境界のブレンド幅
    """
    N = len(grid)
    P = PATCH_PX
    stride = P - overlap_px
    canvas_px = stride * (N-1) + P

    canvas  = np.zeros((canvas_px, canvas_px), dtype=np.float64)
    weights = np.zeros((canvas_px, canvas_px), dtype=np.float64)

    # コサイン窓
    w1d = np.ones(P, dtype=np.float64)
    ramp = 0.5*(1 - np.cos(np.pi*np.arange(overlap_px)/overlap_px))
    w1d[:overlap_px] = ramp;  w1d[-overlap_px:] = ramp[::-1]
    pw = np.outer(w1d, w1d)

    for iy in range(N):
        for ix in range(N):
            p = grid[iy][ix][0,0].cpu().numpy()
            y0, x0 = iy*stride, ix*stride
            canvas [y0:y0+P, x0:x0+P] += p * pw
            weights[y0:y0+P, x0:x0+P] += pw

    return (canvas / (weights + 1e-8)).astype(np.float32)


def save_result(heightmap, outdir, seed, grid_size):
    os.makedirs(outdir, exist_ok=True)
    H, W = heightmap.shape
    elev = heightmap * MAX_ELEV

    dx,dy = np.gradient(elev)
    slope  = np.pi/2 - np.arctan(np.sqrt(dx**2+dy**2))
    aspect = np.arctan2(-dy, dx)
    shade  = np.clip(
        np.sin(45*np.pi/180)*np.sin(slope) +
        np.cos(45*np.pi/180)*np.cos(slope)*np.cos(315*np.pi/180-aspect),
        0, 1)

    fig, axes = plt.subplots(1,2, figsize=(16,8))
    fig.suptitle(f"Infinite Terrain  seed={seed}  {grid_size}×{grid_size} patches")
    axes[0].imshow(elev, cmap="terrain", vmin=0, vmax=MAX_ELEV)
    axes[0].set_title("Elevation"); axes[0].axis("off")
    axes[1].imshow(elev, cmap="terrain", vmin=0, vmax=MAX_ELEV)
    axes[1].imshow(shade, cmap="gray", alpha=0.4)
    axes[1].set_title("Hillshade"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{outdir}/infinite_seed{seed:04d}_{grid_size}x{grid_size}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    h16 = (heightmap*65535).clip(0,65535).astype(np.uint16)
    cv2.imwrite(f"{outdir}/infinite_seed{seed:04d}_height16.png", h16)
    print(f"Saved → {outdir}/")


def main():
    import click

    @click.command()
    @click.option("--ckpt",     required=True, help="best_G.pt path")
    @click.option("--seed-pkl", required=True, help="StyleGAN pkl for seed patch")
    @click.option("--seed",     default=0)
    @click.option("--size",     default=8,     help="Grid size (NxN)")
    @click.option("--out",      default="out_infinite")
    def run(ckpt, seed_pkl, seed, size, out):
        # モデルロード
        expander = load_expander(ckpt)
        print(f"TerrainExpander loaded: {ckpt}")

        with dnnlib.util.open_url(seed_pkl) as f:
            G_style = legacy.load_network_pkl(f)["G_ema"].to(DEVICE)
        G_style.eval()
        print(f"StyleGAN loaded: {seed_pkl}")

        # グリッド生成
        grid = expand_grid(expander, G_style, size, seed, seed_pkl)

        # 合成
        heightmap = stitch_grid(grid, overlap_px=64)

        # 保存
        save_result(heightmap, out, seed, size)

    run()

if __name__ == "__main__":
    main()
