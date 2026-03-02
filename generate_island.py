#!/usr/bin/env python3
"""
generate_island.py

StyleGAN2-ADA で学習した地形モデルから複数タイルを生成・縫い合わせて、
約 6000 km² の島 heightmap を作成する。

タイル 1 枚 = 20km × 20km (512×512px)
デフォルト 5×5 グリッド = 100km × 100km に円形マスクで島を切り出す。

使い方:
    python generate_island.py --network snapshots/network-snapshot-000400.pkl
    python generate_island.py --network ... --seed-start 100 --colorize
    python generate_island.py --network ... --no-generate --colorize  # タイル再利用
"""

import argparse
import os
import subprocess

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# ── 定数 ────────────────────────────────────────────────────────────────────
TILE_KM = 20        # タイル 1 辺の実距離 (km)
TILE_PX = 512       # タイル 1 辺のピクセル数
KM_PER_PX = TILE_KM / TILE_PX   # ≈ 0.039 km/px ≈ 39 m/px


# ── タイル生成 ───────────────────────────────────────────────────────────────
def generate_tiles(network: str, seed_start: int, n_tiles: int,
                   trunc: float, outdir: str) -> None:
    """StyleGAN2-ADA で n_tiles 枚の地形タイルを生成する。"""
    seeds = ",".join(str(seed_start + i) for i in range(n_tiles))
    cmd = [
        "python", "stylegan2-ada-pytorch/generate.py",
        f"--outdir={outdir}",
        f"--trunc={trunc}",
        f"--seeds={seeds}",
        f"--network={network}",
    ]
    print("タイル生成:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_tile(path: str) -> np.ndarray:
    """PNG タイルを float32 [0, 1] の 512×512 配列として読み込む。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"タイルが読み込めません: {path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32) / 255.0


def load_all_tiles(outdir: str, seed_start: int, n_tiles: int) -> list:
    tiles = []
    for i in range(n_tiles):
        seed = seed_start + i
        path = os.path.join(outdir, f"seed{seed:04d}.png")
        tiles.append(load_tile(path))
        print(f"  読み込み: seed{seed:04d}.png", end="\r")
    print()
    return tiles


# ── タイル縫い合わせ ─────────────────────────────────────────────────────────
def stitch_tiles(tiles: list, grid_w: int, grid_h: int,
                 blend_width: int = 64) -> np.ndarray:
    """
    タイルを grid_w × grid_h に縫い合わせる。

    1. 隣接タイルの境界で平均標高を合わせるオフセット補正
       (水平方向 → 垂直方向の順に適用)
    2. シーム幅 blend_width px の Gaussian ブレンドで不連続を滑らか化
    """
    adjusted = [t.copy() for t in tiles]

    # 水平方向オフセット補正（左 → 右）
    for row in range(grid_h):
        for col in range(1, grid_w):
            L = adjusted[row * grid_w + col - 1]
            R = adjusted[row * grid_w + col]
            diff = float(L[:, -16:].mean() - R[:, :16].mean())
            adjusted[row * grid_w + col] = np.clip(R + diff, 0.0, 1.0)

    # 垂直方向オフセット補正（上 → 下）
    for col in range(grid_w):
        for row in range(1, grid_h):
            T = adjusted[(row - 1) * grid_w + col]
            B = adjusted[row * grid_w + col]
            diff = float(T[-16:, :].mean() - B[:16, :].mean())
            adjusted[row * grid_w + col] = np.clip(B + diff, 0.0, 1.0)

    # キャンバスに配置
    out_h = grid_h * TILE_PX
    out_w = grid_w * TILE_PX
    canvas = np.zeros((out_h, out_w), dtype=np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            idx = row * grid_w + col
            r0, c0 = row * TILE_PX, col * TILE_PX
            canvas[r0:r0 + TILE_PX, c0:c0 + TILE_PX] = adjusted[idx]

    # シーム部分の Gaussian ブレンド
    half = blend_width // 2
    sigma = half / 3.0

    # 垂直シーム (x = col * TILE_PX) → 水平方向にブレンド
    for col in range(1, grid_w):
        c = col * TILE_PX
        c0, c1 = max(0, c - half), min(out_w, c + half)
        canvas[:, c0:c1] = gaussian_filter1d(canvas[:, c0:c1],
                                             sigma=sigma, axis=1)

    # 水平シーム (y = row * TILE_PX) → 垂直方向にブレンド
    for row in range(1, grid_h):
        r = row * TILE_PX
        r0, r1 = max(0, r - half), min(out_h, r + half)
        canvas[r0:r1, :] = gaussian_filter1d(canvas[r0:r1, :],
                                             sigma=sigma, axis=0)

    return np.clip(canvas, 0.0, 1.0)


# ── 島マスク ─────────────────────────────────────────────────────────────────
def make_island_mask(h: int, w: int, land_frac: float = 0.87,
                     roughness: int = 6, seed: int = 42) -> np.ndarray:
    """
    有機的な輪郭の島マスク [0, 1] を返す。1 = 陸地、0 = 海。

    land_frac : 楕円の半径 / 画像の半幅。
                5×5 グリッドで land_frac=0.87 ≈ 6000 km²。
    roughness : 海岸線の凸凹レベル（大きいほどフィヨルドや半島が増える）。
    """
    rng = np.random.default_rng(seed)
    cy, cx = h / 2.0, w / 2.0
    ry = (h / 2.0) * land_frac
    rx = (w / 2.0) * land_frac

    yy, xx = np.mgrid[0:h, 0:w]
    dy = (yy - cy) / ry
    dx = (xx - cx) / rx
    theta = np.arctan2(dy, dx)           # [-π, π]
    r_norm = np.sqrt(dx ** 2 + dy ** 2)  # 1.0 = 楕円境界

    # 低周波正弦波を重ねて海岸線に凸凹をつける
    perturbation = np.zeros_like(theta)
    for k in range(1, roughness + 1):
        amp   = rng.uniform(0.04, 0.12) / k
        phase = rng.uniform(0, 2 * np.pi)
        perturbation += amp * np.sin(k * theta + phase)

    boundary = 1.0 + perturbation

    # sigmoid で海岸線をソフトマスク化（値が大きいほど急峻な崖）
    coast_sharpness = 0.06
    signed_dist = (boundary - r_norm) / coast_sharpness
    mask = 1.0 / (1.0 + np.exp(-signed_dist))
    return mask.astype(np.float32)


# ── メイン ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="StyleGAN2 地形モデルから ~6000 km² の島 heightmap を生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--network",     required=True,
                    help="学習済みモデル (.pkl)")
    ap.add_argument("--outdir",      default="images/island",
                    help="出力ディレクトリ")
    ap.add_argument("--seed-start",  type=int, default=0,
                    help="タイル生成の先頭シード値")
    ap.add_argument("--trunc",       type=float, default=0.7,
                    help="Truncation psi (小さいほど典型的な地形)")
    ap.add_argument("--grid-w",      type=int, default=5,
                    help="横タイル数 (5 = 100 km)")
    ap.add_argument("--grid-h",      type=int, default=5,
                    help="縦タイル数 (5 = 100 km)")
    ap.add_argument("--blend-width", type=int, default=64,
                    help="シームブレンド幅 (px)")
    ap.add_argument("--island-frac", type=float, default=0.87,
                    help="島サイズ係数 (楕円半径 / 画像半幅)。"
                         "5×5 グリッドで 0.87 ≈ 6000 km²")
    ap.add_argument("--roughness",   type=int, default=6,
                    help="海岸線の複雑さ (1〜10)")
    ap.add_argument("--mask-seed",   type=int, default=42,
                    help="島形状の乱数シード")
    ap.add_argument("--no-generate", action="store_true",
                    help="タイル生成をスキップして既存ファイルを使用")
    ap.add_argument("--colorize",    action="store_true",
                    help="カラー版 heightmap も生成")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tile_dir = os.path.join(args.outdir, "tiles")
    os.makedirs(tile_dir, exist_ok=True)

    n_tiles   = args.grid_w * args.grid_h
    width_km  = args.grid_w * TILE_KM
    height_km = args.grid_h * TILE_KM
    approx_area = (np.pi
                   * (width_km / 2 * args.island_frac)
                   * (height_km / 2 * args.island_frac))

    print("=" * 50)
    print(f"グリッド    : {args.grid_w}×{args.grid_h} タイル"
          f" = {width_km}km × {height_km}km")
    print(f"タイル枚数  : {n_tiles}")
    print(f"島の概算面積: {approx_area:.0f} km²")
    print(f"解像度      : {KM_PER_PX * 1000:.0f} m/px")
    print("=" * 50)

    # 1) タイル生成
    if not args.no_generate:
        generate_tiles(args.network, args.seed_start, n_tiles,
                       args.trunc, tile_dir)
    else:
        print("タイル生成をスキップします。")

    # 2) タイル読み込み
    print(f"\n{n_tiles} 枚のタイルを読み込み中...")
    tiles = load_all_tiles(tile_dir, args.seed_start, n_tiles)

    # 3) 縫い合わせ
    print(f"タイルを縫い合わせ中 (blend_width={args.blend_width}px)...")
    heightmap = stitch_tiles(tiles, args.grid_w, args.grid_h, args.blend_width)
    h, w = heightmap.shape
    print(f"縫い合わせ完了: {w}×{h} px")

    # 4) 島マスク生成 & 適用
    print(f"島マスクを生成中 (roughness={args.roughness}, seed={args.mask_seed})...")
    mask = make_island_mask(h, w,
                            land_frac=args.island_frac,
                            roughness=args.roughness,
                            seed=args.mask_seed)

    # 陸地には最低高さを確保（~16m）して島らしく、海は 0（海面）
    MIN_LAND_HEIGHT = 0.01
    island = np.clip(heightmap * mask + MIN_LAND_HEIGHT * mask, 0.0, 1.0)

    # 5) 16bit PNG として保存
    out_hm = os.path.join(args.outdir, "island_heightmap.png")
    Image.fromarray((island * 65535).astype(np.uint16)).save(out_hm)

    # 陸地面積を計算
    px_area_km2   = KM_PER_PX ** 2
    land_px       = int((mask > 0.5).sum())
    island_area   = land_px * px_area_km2

    print()
    print("=" * 50)
    print("完了!")
    print(f"  出力       : {out_hm}")
    print(f"  解像度     : {w}×{h} px"
          f" ({w * KM_PER_PX:.1f}km × {h * KM_PER_PX:.1f}km)")
    print(f"  陸地面積   : {island_area:.0f} km²")
    print("=" * 50)

    # 6) カラー版
    if args.colorize:
        color_out = out_hm.replace(".png", "_color.png")
        subprocess.run(
            [
                "python", "heightmap_colorize.py",
                out_hm, color_out,
                "--hillshade", "--hillshade-strength", "0.4",
            ],
            check=True,
        )
        print(f"カラー版   : {color_out}")


if __name__ == "__main__":
    main()
