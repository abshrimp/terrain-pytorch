#!/usr/bin/env python3
"""
ランダムな平滑地形画像 (smooth_terrain.png) を生成するスクリプト。

複数スケールのガウシアンノイズを重ね合わせることで、
自然な大起伏の地形を表す 16bit グレースケール PNG を生成します。

使い方:
    # 1 枚生成
    python terrain_unet/gen_smooth.py --outdir=images/smooth

    # シードを固定して 5 枚生成
    python terrain_unet/gen_smooth.py --count=5 --seed=42 --outdir=images/smooth

    # 滑らかさを強める (sigma を大きくする)
    python terrain_unet/gen_smooth.py --sigma=80 --outdir=images/smooth

    # 島のような形状 (中央が高い)
    python terrain_unet/gen_smooth.py --island --outdir=images/smooth
"""

import os
import sys

import click
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_smooth_terrain(
    size: int = 512,
    sigma: float = 40.0,
    seed: int | None = None,
    island: bool = False,
) -> np.ndarray:
    """
    マルチスケールのガウシアンノイズを重ねてスムーズな地形を生成する。

    戻り値: float32 [0, 1] の [size, size] 配列
    """
    rng = np.random.default_rng(seed)

    terrain = np.zeros((size, size), dtype=np.float64)

    # 大・中・小スケールのノイズを重ね合わせ
    # sigma が大きいほど滑らか・広域な起伏になる
    octaves = [
        (sigma * 2.0, 1.0),   # 最も広域な起伏 (最大影響)
        (sigma * 0.8, 0.4),   # 中程度の起伏
        (sigma * 0.3, 0.1),   # 小さめの起伏
    ]
    for oct_sigma, amplitude in octaves:
        noise = rng.standard_normal((size, size))
        terrain += gaussian_filter(noise, sigma=oct_sigma) * amplitude

    # 島マスク: 中央が高く、周辺が低い放射状グラデーション
    if island:
        y, x = np.ogrid[:size, :size]
        cx, cy = size / 2, size / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # 端で 0、中央付近で 1 になるマスク
        radius = size * 0.45
        mask = np.clip(1.0 - dist / radius, 0, 1) ** 1.5
        terrain = terrain * mask + mask * 0.3  # 周辺を海面下にシフト

    # [0, 1] に正規化
    terrain -= terrain.min()
    terrain /= terrain.max() + 1e-8

    return terrain.astype(np.float32)


def save_smooth_terrain(terrain: np.ndarray, path_base: str):
    """
    terrain: float32 [0, 1], shape [H, W]

    保存ファイル:
      <path_base>.png         16bit グレースケール
      <path_base>_preview.png カラーマップ可視化
    """
    # 16bit PNG
    terrain_16 = (terrain * 65535).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(path_base + ".png", terrain_16)

    # プレビュー画像
    real_elev = terrain * 1600.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(real_elev, cmap="terrain", vmin=0, vmax=1600)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path_base + "_preview.png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()


@click.command()
@click.option("--outdir", default="images/smooth", show_default=True, metavar="DIR",  help="出力ディレクトリ")
@click.option("--count",  default=1,               show_default=True, type=int,       help="生成枚数")
@click.option("--size",   default=512,              show_default=True, type=int,       help="画像サイズ (px)")
@click.option("--sigma",  default=40.0,             show_default=True, type=float,     help="平滑化強度 (大きいほど滑らか)")
@click.option("--seed",   default=None,             type=int,                          help="乱数シード (省略時はランダム)")
@click.option("--island", is_flag=True, default=False,                                help="島状の地形 (中央が高い)")
def gen_smooth(outdir, count, size, sigma, seed, island):
    """
    ランダムな平滑地形画像を生成します。
    生成した .png を terrain_unet/generate.py の --input に渡せます。

    例:
        python terrain_unet/gen_smooth.py --count=3 --sigma=40 --outdir=images/smooth
        python terrain_unet/generate.py \\
            --checkpoint=terrain_unet/checkpoints/latest.pt \\
            --input=images/smooth/smooth_000.png \\
            --outdir=images/generated
    """
    os.makedirs(outdir, exist_ok=True)

    for i in range(count):
        s = (seed + i) if seed is not None else None
        terrain = make_smooth_terrain(size=size, sigma=sigma, seed=s, island=island)

        stem = f"smooth_{i:03d}"
        path_base = os.path.join(outdir, stem)
        save_smooth_terrain(terrain, path_base)
        print(f"  {path_base}.png  (sigma={sigma}{'  island' if island else ''})")

    print(f"\n{count} 枚生成完了 → {outdir}/")
    print(f"次のコマンドで地形を生成できます:")
    print(f"  python terrain_unet/generate.py \\")
    print(f"      --checkpoint=terrain_unet/checkpoints/latest.pt \\")
    print(f"      --input={os.path.join(outdir, 'smooth_000.png')} \\")
    print(f"      --outdir=images/generated")


if __name__ == "__main__":
    gen_smooth()
