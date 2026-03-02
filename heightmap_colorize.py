#!/usr/bin/env python3
"""
グレースケール標高マップ（16bit PNG）をカラー標高マップに変換するスクリプト

使い方:
    python heightmap_colorize.py input.png output.png [--colormap terrain] [--legend]
"""

import argparse
import numpy as np
import cv2
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def create_terrain_colormap():
    """地形風のカスタムカラーマップ（海底→平地→山→雪）"""
    colors = [
        (0.00, (0.00, 0.20, 0.50)),  # 深海（濃い青）
        (0.15, (0.10, 0.40, 0.70)),  # 浅い海（青）
        (0.25, (0.20, 0.60, 0.40)),  # 海岸（青緑）
        (0.30, (0.30, 0.70, 0.20)),  # 低地（緑）
        (0.45, (0.50, 0.80, 0.20)),  # 平地（明るい緑）
        (0.55, (0.80, 0.80, 0.20)),  # 丘陵（黄緑）
        (0.65, (0.85, 0.65, 0.15)),  # 高地（黄土色）
        (0.75, (0.70, 0.45, 0.15)),  # 山（茶色）
        (0.85, (0.55, 0.35, 0.20)),  # 高山（濃い茶）
        (0.92, (0.75, 0.75, 0.75)),  # 岩肌（灰色）
        (1.00, (1.00, 1.00, 1.00)),  # 雪（白）
    ]
    positions = [c[0] for c in colors]
    rgb_values = [c[1] for c in colors]
    return LinearSegmentedColormap.from_list("terrain_custom", list(zip(positions, rgb_values)), N=65536)


# 利用可能なカラーマップ
COLORMAPS = {
    "terrain":  create_terrain_colormap,
    "viridis":  lambda: cm.get_cmap("viridis", 65536),
    "inferno":  lambda: cm.get_cmap("inferno", 65536),
    "plasma":   lambda: cm.get_cmap("plasma", 65536),
    "magma":    lambda: cm.get_cmap("magma", 65536),
    "cividis":  lambda: cm.get_cmap("cividis", 65536),
    "turbo":    lambda: cm.get_cmap("turbo", 65536),
    "gist_earth": lambda: cm.get_cmap("gist_earth", 65536),
}


def colorize_heightmap(input_path: str, output_path: str, colormap_name: str = "terrain",
                       add_legend: bool = False, add_hillshade: bool = False,
                       hillshade_strength: float = 0.5):
    """
    グレースケール標高マップをカラー画像に変換して保存する

    Args:
        input_path: 入力グレースケールPNGのパス
        output_path: 出力カラーPNGのパス
        colormap_name: 使用するカラーマップ名
        add_legend: 凡例（カラーバー）を追加するか
        add_hillshade: 陰影起伏を重ねるか
        hillshade_strength: 陰影の強さ (0.0〜1.0)
    """
    # 画像読み込み（16bit or 8bit グレースケール）
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {input_path}")

    # カラー画像の場合はグレースケールに変換
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 正規化 (0.0 〜 1.0)
    if img.dtype == np.uint16:
        normalized = img.astype(np.float64) / 65535.0
    elif img.dtype == np.uint8:
        normalized = img.astype(np.float64) / 255.0
    else:
        normalized = img.astype(np.float64)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-10)

    print(f"入力画像: {img.shape}, dtype={img.dtype}")
    print(f"値の範囲: {img.min()} 〜 {img.max()}")
    print(f"カラーマップ: {colormap_name}")

    # カラーマップ適用
    if colormap_name not in COLORMAPS:
        print(f"警告: '{colormap_name}' が見つかりません。'terrain' を使用します。")
        colormap_name = "terrain"

    cmap = COLORMAPS[colormap_name]()
    colored = cmap(normalized)[:, :, :3]  # RGBA → RGB (float 0-1)

    # 陰影起伏（Hillshade）の追加
    if add_hillshade:
        hillshade = compute_hillshade(normalized)
        # 陰影をカラー画像にブレンド
        for c in range(3):
            colored[:, :, c] = colored[:, :, c] * (1 - hillshade_strength) + \
                               colored[:, :, c] * hillshade * hillshade_strength

    # 16bit PNGとして保存
    colored_16 = (colored * 65535).clip(0, 65535).astype(np.uint16)
    colored_bgr = cv2.cvtColor(colored_16, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, colored_bgr)
    print(f"カラー標高マップを保存しました: {output_path}")

    # 凡例付き画像の保存
    if add_legend:
        legend_path = output_path.replace(".png", "_legend.png")
        save_with_legend(normalized, cmap, legend_path)
        print(f"凡例付き画像を保存しました: {legend_path}")


def compute_hillshade(elevation: np.ndarray, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    """
    標高データから陰影起伏（Hillshade）を計算する

    Args:
        elevation: 正規化された標高データ (0-1)
        azimuth: 光源の方位角（度）
        altitude: 光源の高度角（度）
    Returns:
        陰影マップ (0-1)
    """
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    # 勾配計算
    dy, dx = np.gradient(elevation)

    # 傾斜と方位
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    # 陰影計算
    hillshade = (np.sin(alt_rad) * np.cos(slope) +
                 np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))

    return np.clip(hillshade, 0, 1)


def save_with_legend(normalized: np.ndarray, cmap, output_path: str):
    """matplotlibでカラーバー付き画像を保存"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Elevation Map", fontsize=16)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation (normalized)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="グレースケール標高マップをカラー化")
    parser.add_argument("input", help="入力グレースケールPNG")
    parser.add_argument("output", help="出力カラーPNG")
    parser.add_argument("--colormap", "-c", default="terrain",
                        choices=list(COLORMAPS.keys()),
                        help=f"カラーマップ（デフォルト: terrain）")
    parser.add_argument("--legend", "-l", action="store_true",
                        help="凡例（カラーバー）付き画像も生成")
    parser.add_argument("--hillshade", action="store_true",
                        help="陰影起伏を重ねる")
    parser.add_argument("--hillshade-strength", type=float, default=0.5,
                        help="陰影の強さ 0.0〜1.0（デフォルト: 0.5）")

    args = parser.parse_args()
    colorize_heightmap(
        args.input, args.output,
        colormap_name=args.colormap,
        add_legend=args.legend,
        add_hillshade=args.hillshade,
        hillshade_strength=args.hillshade_strength,
    )


if __name__ == "__main__":
    main()