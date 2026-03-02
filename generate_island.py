#!/usr/bin/env python3
"""
generate_island.py

根本的アプローチ: 手続き的基盤 + AI テクスチャ

【問題】
  AI タイルを縫い合わせるアプローチの限界:
    - 各タイルが独立に生成されるため大規模構造(山脈・谷)がタイルごとに無関係
    - シームブレンドを工夫してもタイル境界で地形の「意味」が断絶する

【解決策: 役割分担】
  ① 大規模構造 → FFT フラクタルノイズ (全島で空間的一貫性を保証)
  ② 局所テクスチャ → AI タイルの高周波成分 (Hanning 窓 + 半ストライド重畳)
  ③ ① + ② を合成 → 島マスクを適用

  AI タイルは「貼り合わせる構造素材」ではなく「繰り返し適用できるテクスチャ」として使う。
  Hanning 窓により各テクスチャタイルのエッジが自然にゼロになるため、
  シームレスに重ね合わせることができる。

使い方:
    python generate_island.py --network snapshots/network-snapshot-000400.pkl
    python generate_island.py --network ... --base-seed 7 --mask-seed 3 --colorize
    python generate_island.py --no-generate --base-seed 7 --colorize  # タイル再利用
"""

import argparse
import os
import subprocess

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ── 定数 ────────────────────────────────────────────────────────────────────
TILE_KM   = 20
TILE_PX   = 512
ISLAND_KM = 100                         # 島を包む正方形 (km)
KM_PER_PX = TILE_KM / TILE_PX          # ≈ 0.039 km/px ≈ 39 m/px
MAP_PX    = int(ISLAND_KM / KM_PER_PX) # = 2560 px


# ── AI タイル ────────────────────────────────────────────────────────────────
def generate_tiles(network: str, seed_start: int, n: int,
                   trunc: float, outdir: str) -> None:
    seeds = ",".join(str(seed_start + i) for i in range(n))
    cmd = [
        "python", "stylegan2-ada-pytorch/generate.py",
        f"--outdir={outdir}", f"--trunc={trunc}",
        f"--seeds={seeds}", f"--network={network}",
    ]
    print("AI タイル生成:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_tile(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"読み込み失敗: {path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return img.astype(np.float32) / scale


# ── ① 手続き的基盤地形 (FFT フラクタルノイズ) ───────────────────────────────
def fractal_terrain(h: int, w: int, seed: int = 0, beta: float = 2.2) -> np.ndarray:
    """
    ガウス乱数場をフーリエ域でスペクトル成形して地形的なフラクタルを生成する。

    パワースペクトル ∝ k^(-beta):
      beta = 2.0  → 茶色雑音（やや滑らか）
      beta = 2.2  → 地形らしいスペクトル（デフォルト）
      beta = 2.5  → 急峻な地形

    全解像度で一度に生成するため、タイル結合の問題が根本的に存在しない。
    """
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))

    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    freq = np.sqrt(fy ** 2 + fx ** 2)
    freq[0, 0] = 1.0            # DC 特異点を回避
    power = freq ** (-beta / 2.0)
    power[0, 0] = 0.0           # DC 成分 = 0（オフセットなし）

    field = np.real(np.fft.ifft2(F * power)).astype(np.float32)
    lo, hi = field.min(), field.max()
    return (field - lo) / (hi - lo + 1e-8)


def shape_island_base(raw: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    フラクタルノイズを島らしい地形に成形する。

    成形内容:
      - 中央高・海岸低のラジアルプロファイルをブレンド（島の標高分布）
      - リッジノイズで平坦部を尾根状に変換
      - 累乗で山の尖りを強調
    """
    h, w = raw.shape
    cy, cx = h / 2.0, w / 2.0

    # 中央 = 1.0、端 = 0.0 の放射状プロファイル
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.clip(
        np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2), 0.0, 1.0
    )
    profile = (1.0 - dist) ** 1.5   # 中央が高く、海岸付近はなだらか

    # リッジノイズ: 滑らかな丘 → 尖った尾根に変換
    ridge = 1.0 - np.abs(2.0 * raw - 1.0)

    # ブレンド比率: フラクタル 45%、島プロファイル 40%、リッジ 15%
    terrain = raw * 0.45 + profile * 0.40 + ridge * 0.15

    # 累乗で高地をより尖らせ、低地をより低く
    terrain = np.power(np.clip(terrain, 0.0, 1.0), 1.4)

    lo, hi = terrain.min(), terrain.max()
    return ((terrain - lo) / (hi - lo + 1e-8)).astype(np.float32)


# ── ② AI テクスチャ詳細 ──────────────────────────────────────────────────────
def extract_windowed_detail(tile: np.ndarray, sigma: float = 48.0) -> np.ndarray:
    """
    AI タイルから高周波詳細を抽出する。

    処理:
      1. ガウシアン高域通過フィルタで大域トレンドを除去
         → ゼロ平均のローカルテクスチャのみ残る
      2. Hanning 窓でエッジをゼロに揃える
         → 重ね合わせ時にシームが自然に消える
    """
    # 高域通過: タイル - ガウシアン平滑化
    detail = tile - gaussian_filter(tile, sigma=sigma)

    # Hanning 窓（エッジ = 0、中央 = 1）
    h, w = detail.shape
    window = (
        np.hanning(h).astype(np.float32)[:, None]
        * np.hanning(w).astype(np.float32)[None, :]
    )
    return (detail * window).astype(np.float32)


def create_texture_map(details: list, H: int, W: int, seed: int = 0,
                       stride_frac: float = 0.5) -> np.ndarray:
    """
    Hanning 窓付き詳細タイルを半ストライドで重ね合わせてテクスチャマップを生成する。

    Hanning 窓の重ね合わせ正規化により:
      - 各位置で weight_sum ≈ 一定になる
      - 隣接タイル間でシームレスに混合される
      - AI タイルをランダムに選択することで繰り返しパターンを回避
    """
    rng    = np.random.default_rng(seed)
    stride = max(1, int(TILE_PX * stride_frac))

    canvas = np.zeros((H, W), dtype=np.float32)
    wsum   = np.zeros((H, W), dtype=np.float32)

    # Hanning 窓の 2 乗（正規化係数として使用）
    win_sq = (
        np.hanning(TILE_PX).astype(np.float32)[:, None]
        * np.hanning(TILE_PX).astype(np.float32)[None, :]
    ) ** 2

    for r in range(0, H + 1, stride):
        for c in range(0, W + 1, stride):
            # 端でクリップして常に TILE_PX × TILE_PX の完全タイルを使う
            r0 = min(r, H - TILE_PX)
            c0 = min(c, W - TILE_PX)

            detail = details[rng.integers(len(details))]
            canvas[r0:r0 + TILE_PX, c0:c0 + TILE_PX] += detail
            wsum[r0:r0 + TILE_PX, c0:c0 + TILE_PX]   += win_sq

    valid = wsum > 1e-6
    canvas[valid] /= wsum[valid]
    return canvas


# ── 島マスク ─────────────────────────────────────────────────────────────────
def make_island_mask(h: int, w: int, land_frac: float = 0.87,
                     roughness: int = 6, seed: int = 42) -> np.ndarray:
    """
    有機的な輪郭の島マスク [0, 1] を生成する。

    手法: 楕円境界に低周波正弦波の重ね合わせで凸凹を加え、
          sigmoid でソフトな海岸線にする。
    land_frac=0.87 かつ ISLAND_KM=100km のとき:
      楕円半径 ≈ 43.5km → 面積 ≈ π × 43.5² ≈ 5946 km²
    """
    rng = np.random.default_rng(seed)
    cy, cx = h / 2.0, w / 2.0
    ry = (h / 2.0) * land_frac
    rx = (w / 2.0) * land_frac

    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = (yy - cy) / ry, (xx - cx) / rx
    theta  = np.arctan2(dy, dx)
    r_norm = np.sqrt(dx ** 2 + dy ** 2)

    # 低周波正弦波で海岸線に湾・半島を生成
    perturbation = np.zeros_like(theta)
    for k in range(1, roughness + 1):
        amp   = rng.uniform(0.04, 0.12) / k
        phase = rng.uniform(0.0, 2.0 * np.pi)
        perturbation += amp * np.sin(k * theta + phase)

    boundary = 1.0 + perturbation
    # coast_width が小さいほど急崖、大きいほど緩やかな浜辺
    coast_width = 0.06
    signed_dist = (boundary - r_norm) / coast_width
    return (1.0 / (1.0 + np.exp(-signed_dist))).astype(np.float32)


# ── メイン ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="手続き的基盤 + AI テクスチャで ~6000 km² の島 heightmap を生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--network",
                    help="学習済みモデル (.pkl)。--no-generate 時は省略可")
    ap.add_argument("--outdir",       default="images/island")
    ap.add_argument("--seed-start",   type=int,   default=0,
                    help="AI タイル生成の先頭シード値")
    ap.add_argument("--trunc",        type=float, default=0.7,
                    help="truncation psi")
    ap.add_argument("--n-tex",        type=int,   default=9,
                    help="生成する AI テクスチャタイル数 (多いほど繰り返しが減る)")
    ap.add_argument("--base-seed",    type=int,   default=0,
                    help="手続き的基盤地形の乱数シード")
    ap.add_argument("--tex-strength", type=float, default=0.30,
                    help="AI テクスチャの加算強度 (0=手続き的のみ、0.5 で強め)")
    ap.add_argument("--beta",         type=float, default=2.2,
                    help="フラクタルスペクトル指数 (2.0=滑らか、2.5=険しい)")
    ap.add_argument("--island-frac",  type=float, default=0.87,
                    help="島サイズ係数 (楕円半径 / 画像半幅)")
    ap.add_argument("--roughness",    type=int,   default=6,
                    help="海岸線の複雑さ (1=円形、10=フィヨルド)")
    ap.add_argument("--mask-seed",    type=int,   default=42,
                    help="島形状の乱数シード")
    ap.add_argument("--no-generate",  action="store_true",
                    help="AI タイル生成をスキップ（既存ファイルを使用）")
    ap.add_argument("--colorize",     action="store_true",
                    help="カラー版 heightmap も生成")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tile_dir = os.path.join(args.outdir, "tiles")
    os.makedirs(tile_dir, exist_ok=True)

    H = W = MAP_PX
    approx_area = np.pi * (ISLAND_KM / 2 * args.island_frac) ** 2

    print("=" * 60)
    print(f"出力サイズ  : {W}×{H} px = {ISLAND_KM}km × {ISLAND_KM}km")
    print(f"解像度      : {KM_PER_PX * 1000:.0f} m/px")
    print(f"島概算面積  : {approx_area:.0f} km²")
    print(f"手法        : FFT フラクタル基盤 + AI テクスチャ (強度={args.tex_strength})")
    print("=" * 60)

    # Step 1: AI タイル生成
    if not args.no_generate:
        if not args.network:
            ap.error("--network が必要です (--no-generate 未指定時)")
        generate_tiles(args.network, args.seed_start, args.n_tex,
                       args.trunc, tile_dir)
    else:
        print("AI タイル生成をスキップします。")

    # Step 2: AI タイルから高周波テクスチャ詳細を抽出
    print(f"\nAI テクスチャ詳細を抽出中 ({args.n_tex} 枚)...")
    details = []
    for i in range(args.n_tex):
        path = os.path.join(tile_dir, f"seed{args.seed_start + i:04d}.png")
        details.append(extract_windowed_detail(load_tile(path)))
    print(f"  → {len(details)} 種類のテクスチャパターンを取得")

    # Step 3: 手続き的基盤地形 (空間的に一貫したフラクタルノイズ)
    print(f"\n手続き的基盤地形を生成中 (seed={args.base_seed}, β={args.beta})...")
    raw_base = fractal_terrain(H, W, seed=args.base_seed, beta=args.beta)
    base = shape_island_base(raw_base, seed=args.base_seed)
    print(f"  → {W}×{H} px の地形生成完了")

    # Step 4: AI テクスチャマップを全域に敷き詰める (シームレス)
    print("AI テクスチャマップを生成中...")
    texture = create_texture_map(details, H, W, seed=args.base_seed)
    print(f"  → テクスチャ標準偏差: {texture.std():.4f}")

    # Step 5: 基盤 + テクスチャ合成
    print("合成中...")
    combined = np.clip(base + args.tex_strength * texture, 0.0, 1.0)

    # Step 6: 島マスク
    print(f"島マスクを適用中 (roughness={args.roughness}, seed={args.mask_seed})...")
    mask = make_island_mask(H, W,
                            land_frac=args.island_frac,
                            roughness=args.roughness,
                            seed=args.mask_seed)
    MIN_LAND_HEIGHT = 0.01   # 最低標高 ~16m（海岸がゼロにならないように）
    island = np.clip(combined * mask + MIN_LAND_HEIGHT * mask, 0.0, 1.0)

    # Step 7: 16bit PNG 保存
    out_hm = os.path.join(args.outdir, "island_heightmap.png")
    Image.fromarray((island * 65535).astype(np.uint16)).save(out_hm)

    land_area_km2 = int((mask > 0.5).sum()) * KM_PER_PX ** 2

    print()
    print("=" * 60)
    print("完了!")
    print(f"  出力     : {out_hm}")
    print(f"  解像度   : {W}×{H} px")
    print(f"  陸地面積 : {land_area_km2:.0f} km²")
    print("=" * 60)

    # Step 8 (オプション): カラー版
    if args.colorize:
        color_out = out_hm.replace(".png", "_color.png")
        subprocess.run(
            ["python", "heightmap_colorize.py", out_hm, color_out,
             "--hillshade", "--hillshade-strength", "0.4"],
            check=True,
        )
        print(f"カラー版 : {color_out}")


if __name__ == "__main__":
    main()
