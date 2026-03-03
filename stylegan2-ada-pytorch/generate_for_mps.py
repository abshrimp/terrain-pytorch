#!/usr/bin/env python3
import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import cv2
import matplotlib.pyplot as plt

import legacy

# ------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def num_range(s: str) -> List[int]:
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in s.split(',')]

# ------------------------------------------------------------
def save_image(img_tensor, path, save_heightmap_color=True, save_hillshade=True):
    """
    img_tensor: (1,C,H,W) torch.Tensor [-1,1]
    path: 保存パス (.png)
    save_heightmap_color: カラーマップ付き高さマップを保存
    save_hillshade: hillshade を保存（従来方式）
    imshow_hillshade: matplotlib imshow で高さマップ＋半透明陰影を保存
    """
    img_tensor = img_tensor.detach().to(torch.float32).cpu()
    N, C, H, W = img_tensor.shape

    # --- 元画像 16bit RGB PNG 保存 ---
    img_np = img_tensor[0].permute(1,2,0).numpy()
    img_16 = ((img_np * 0.5 + 0.5) * 65535).clip(0,65535).astype(np.uint16)
    if C >= 3:
        img_bgr = img_16[..., [2,1,0]]  # RGB→BGR
        cv2.imwrite(path, img_bgr)
    else:
        cv2.imwrite(path, img_16[...,0])

    # --- 高さマップ ---
    if save_heightmap_color or save_hillshade:
        if C >= 3:
            heightmap = img_tensor[0].mean(dim=0, keepdim=True)
        else:
            heightmap = img_tensor[0:1]

        # 0-1600 スケーリング
        real_elevation = ((heightmap[0].squeeze().numpy() + 1.0) / 2.0) * 1600.0

        # カラーマップ 8bit RGB
        color_img = (plt.get_cmap('terrain')(real_elevation / 1600.0)[:, :, :3] * 255).astype(np.uint8)

    # --- 高さマップカラー保存 ---
    if save_heightmap_color:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.axis('off')
        # 高さマップ terrain カラーマップ
        ax.imshow(real_elevation, cmap='terrain', vmin=0, vmax=1600)
        plt.tight_layout(pad=0)
        plt.savefig(path.replace(".png","_height_color.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # --- imshow 合成保存 ---
    if save_hillshade:
        dx, dy = np.gradient(real_elevation)
        slope = np.pi/2. - np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dy, dx)
        azimuth = 315.0 * np.pi / 180.0
        altitude = 45.0 * np.pi / 180.0
        shade = np.sin(altitude)*np.sin(slope) + np.cos(altitude)*np.cos(slope)*np.cos(azimuth - aspect)
        shade = np.clip(shade, 0, 1)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.axis('off')
        # 高さマップ terrain カラーマップ
        ax.imshow(real_elevation, cmap='terrain', vmin=0, vmax=1600)
        # 半透明陰影
        ax.imshow(shade, cmap='gray', alpha=0.4)
        plt.tight_layout(pad=0)
        plt.savefig(path.replace(".png","_hillshade.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# ------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', required=True)
@click.option('--seeds', type=num_range)
@click.option('--trunc', 'truncation_psi', type=float, default=1)
@click.option('--class', 'class_idx', type=int)
@click.option('--noise-mode', type=click.Choice(['const','random','none']), default='const')
@click.option('--projected-w', type=str, metavar='FILE')
@click.option('--outdir', required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    device = get_device()
    print(f'Using device: {device}')
    print(f'Loading network from "{network_pkl}"')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        G.eval()

    os.makedirs(outdir, exist_ok=True)

    # --- projected W から生成 ---
    if projected_w is not None:
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            save_image(img, f'{outdir}/proj{idx:02d}.png', imshow_hillshade=True)
        return

    if seeds is None:
        ctx.fail('--seeds is required when not using --projected-w')

    # --- ラベル ---
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Conditional network requires --class')
        label[:, class_idx] = 1

    # --- 生成 ---
    for seed in seeds:
        print(f'Generating seed {seed}')
        rnd = np.random.RandomState(seed)
        z = torch.from_numpy(rnd.randn(1, G.z_dim)).float().to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        save_image(img, f'{outdir}/seed{seed:04d}.png')

# ------------------------------------------------------------
if __name__ == "__main__":
    generate_images()