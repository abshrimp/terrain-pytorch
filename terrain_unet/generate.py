#!/usr/bin/env python3
"""
学習済みモデルで平滑化地形画像 → 詳細地形を生成します。

使い方:
    python terrain_unet/generate.py \
        --checkpoint=terrain_unet/checkpoints/latest.pt \
        --input=images/smooth/smooth_terrain.png \
        --outdir=images/generated

出力:
    <outdir>/<stem>_detail.png    16bit グレースケール (詳細地形)
    <outdir>/<stem>_hillshade.png ヒルシェード可視化
    <outdir>/<stem>_compare.png   smooth vs detail 比較
"""

import os
import sys

import click
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_unet.model import UNetGenerator, get_device


# ----------------------------------------------------------------
def load_smooth_image(path: str, size: int):
    """
    平滑化画像を読み込み [1,1,H,W] float32 tensor [-1,1] に変換。
    16bit グレースケール / 8bit グレースケール / RGB いずれも対応。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"読み込み失敗: {path}")

    # BGR→グレースケール
    if img.ndim == 3:
        img = img.mean(axis=2)

    # uint16 / uint8 → float32 [-1, 1]
    if img.dtype == np.uint16:
        img_f = img.astype(np.float32) / 32767.5 - 1.0
    else:
        img_f = img.astype(np.float32) / 127.5 - 1.0

    # リサイズ
    if img_f.shape[0] != size or img_f.shape[1] != size:
        img_f = cv2.resize(img_f, (size, size), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_f[None, None]).float()  # [1,1,H,W]
    return img_f, tensor


# ----------------------------------------------------------------
def save_outputs(out_f: np.ndarray, smooth_f: np.ndarray, path_base: str):
    """
    生成結果を 3 種類のファイルで保存。
      *_detail.png   : 16bit グレースケール
      *_hillshade.png: ヒルシェード + カラーマップ
      *_compare.png  : smooth / detail 並列比較
    """
    # 16bit グレースケール保存
    out_16 = ((out_f + 1.0) / 2.0 * 65535.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(path_base + "_detail.png", out_16)

    # ヒルシェード計算
    real_elev = (out_f + 1.0) / 2.0 * 1600.0
    dx, dy = np.gradient(real_elev)
    slope  = np.pi / 2.0 - np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    az, alt = 315.0 * np.pi / 180.0, 45.0 * np.pi / 180.0
    shade  = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade  = np.clip(shade, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    ax.imshow(real_elev, cmap="terrain", vmin=0, vmax=1600)
    ax.imshow(shade, cmap="gray", alpha=0.4)
    plt.tight_layout(pad=0)
    plt.savefig(path_base + "_hillshade.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # smooth vs detail 比較
    smooth_elev = (smooth_f + 1.0) / 2.0 * 1600.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(smooth_elev, cmap="terrain", vmin=0, vmax=1600)
    axes[0].set_title("入力 (smooth)")
    axes[0].axis("off")
    axes[1].imshow(real_elev, cmap="terrain", vmin=0, vmax=1600)
    axes[1].set_title("生成 (detail)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path_base + "_compare.png", bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------
@click.command()
@click.option("--checkpoint", required=True, metavar="FILE", help="学習済みチェックポイント .pt")
@click.option("--input",  "input_fname", required=True, metavar="FILE", help="平滑化された地形画像")
@click.option("--outdir", required=True, metavar="DIR",  help="出力ディレクトリ")
@click.option("--ngf",    default=None,  type=int,       help="Generator チャンネル数 (省略時はチェックポイントから自動取得)")
@click.option("--size",   default=512,   show_default=True, type=int, help="処理解像度")
def generate(checkpoint, input_fname, outdir, ngf, size):
    """
    学習済み pix2pix モデルで平滑化地形 → 詳細地形を生成します。

    例:
        python terrain_unet/generate.py \\
            --checkpoint=terrain_unet/checkpoints/latest.pt \\
            --input=images/smooth/smooth_terrain.png \\
            --outdir=images/generated
    """
    device = get_device()
    print(f"デバイス: {device}")

    # チェックポイント読み込み
    print(f"チェックポイント: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ngf = ngf or ckpt.get("ngf", 64)
    epoch = ckpt.get("epoch", "?")
    print(f"  ngf={ngf}, 学習済みエポック={epoch if isinstance(epoch, str) else epoch+1}")

    G = UNetGenerator(in_ch=1, out_ch=1, ngf=ngf).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # 入力画像読み込み
    print(f"入力: {input_fname}")
    smooth_f, smooth_t = load_smooth_image(input_fname, size)
    smooth_t = smooth_t.to(device)

    # 推論
    with torch.no_grad():
        out_t = G(smooth_t)  # [1, 1, H, W], [-1, 1]
    out_f = out_t[0, 0].cpu().numpy()

    # 保存
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_fname))[0]
    path_base = os.path.join(outdir, stem)
    save_outputs(out_f, smooth_f, path_base)

    print("出力:")
    print(f"  {path_base}_detail.png")
    print(f"  {path_base}_hillshade.png")
    print(f"  {path_base}_compare.png")


if __name__ == "__main__":
    generate()
