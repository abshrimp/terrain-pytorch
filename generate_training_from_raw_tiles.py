#!/usr/bin/env python3
"""
generate_training_from_raw_tiles.py

raw_tiles/{z}_{x}_{y}.npy (256x256, float32, 欠損値=-9999.0) を読み込み、
隣接する 2x2 タイルブロックを 512x512 パッチとして組み立て、
フィルタリング・正規化した後にグレースケール PNG として
training_samples/ へ書き出す。

切り出しの考え方（stride=256、タイルサイズ=256 なので自然にタイル境界に一致する）:

  ┌───┬───┐
  │x,y│x+1│  → 512x512 パッチ 1枚
  ├───┼───┤
  │x,y│x+1│
  │+1 │+1 │
  └───┴───┘

"""

import os
import re
import numpy as np
import skimage.measure
from PIL import Image
from tqdm import tqdm

# ------------------------------------------------------------------ 設定 --

RAW_TILES_DIR = './raw_tiles'
OUTPUT_DIR    = './training_samples'

MISSING_VALUE = -9999.0
TILE_SIZE     = 256
SAMPLE_SIZE   = TILE_SIZE * 2   # = 512

# フィルタリング閾値（必要に応じて調整）
MIN_ELEV_RANGE    = 40.0   # [m] 最大-最小がこれ未満 → 平坦すぎて除外
MAX_NEAR_MIN_FRAC = 0.20   # 最小値付近のピクセル割合がこれ超 → 水域として除外
MIN_ENTROPY       = 10.0   # Shannon entropy がこれ未満 → データ異常として除外

# ブラックリスト: ズームレベルごとに除外したいタイル座標範囲を指定する。
# パッチを構成する 2×2 タイルのいずれかが範囲内に入れば除外される。
BLACKLIST = {
    12: [
        {"name": "富士山",         "x_range": (3624, 3628), "y_range": (1615, 1620)},
        {"name": "大山（鳥取）",   "x_range": (3566, 3569), "y_range": (1615, 1618)},
        {"name": "浅間山",         "x_range": (3622, 3625), "y_range": (1601, 1604)},
        {"name": "箱根カルデラ",   "x_range": (3627, 3631), "y_range": (1617, 1620)},
        {"name": "阿蘇山カルデラ", "x_range": (3537, 3541), "y_range": (1649, 1653)},
        {"name": "追加エリア",     "x_range": (3620, 3623), "y_range": (1606, 1609)},
        {"name": "追加エリア",     "x_range": (3627, 3629), "y_range": (1600, 1602)},
        {"name": "追加エリア",     "x_range": (3630, 3632), "y_range": (1599, 1601)},
        {"name": "追加エリア",     "x_range": (3650, 3651), "y_range": (1543, 1544)},
        {"name": "追加エリア",     "x_range": (3648, 3648), "y_range": (1554, 1555)},
        {"name": "追加エリア",     "x_range": (3651, 3652), "y_range": (1552, 1553)},
        {"name": "追加エリア",     "x_range": (3640, 3642), "y_range": (1584, 1586)},
        {"name": "追加エリア",     "x_range": (3643, 3645), "y_range": (1539, 1541)},
        {"name": "追加エリア",     "x_range": (3640, 3641), "y_range": (1563, 1564)},
        {"name": "追加エリア",     "x_range": (3535, 3538), "y_range": (1663, 1667)},
        {"name": "追加エリア",     "x_range": (3649, 3650), "y_range": (1507, 1508)},
        {"name": "追加エリア",     "x_range": (3649, 3651), "y_range": (1510, 1511)},
        {"name": "追加エリア",     "x_range": (3648, 3649), "y_range": (1518, 1519)},
        {"name": "追加エリア",     "x_range": (3655, 3656), "y_range": (1508, 1509)},
        {"name": "追加エリア",     "x_range": (3689, 3692), "y_range": (1494, 1496)},
        {"name": "追加エリア",     "x_range": (3686, 3688), "y_range": (1497, 1499)},
        {"name": "追加エリア",     "x_range": (3654, 3656), "y_range": (1469, 1471)},
        {"name": "追加エリア",     "x_range": (3631, 3646), "y_range": (1624, 1671)},
    ],
}

# ------------------------------------------------------------------ 関数 --

def build_blacklist_tileset(areas):
    """
    ブラックリストエリアのリストから、除外すべきタイル座標の集合を作る。
    パッチは 2×2 タイルで構成されるため、エリア境界から 1 タイル外側まで拡張する。
    （x0, y0) のパッチは (x0, x0+1) × (y0, y0+1) のタイルを使うので、
    ブラックリストタイル (bx, by) を含むパッチの左上は
    x0 ∈ {bx-1, bx}、y0 ∈ {by-1, by} になる。
    """
    blacklisted = set()
    for area in areas:
        x0_bl, x1_bl = area['x_range']
        y0_bl, y1_bl = area['y_range']
        for bx in range(x0_bl, x1_bl + 1):
            for by in range(y0_bl, y1_bl + 1):
                blacklisted.add((bx, by))
    return blacklisted


def is_blacklisted(x0, y0, blacklisted_tiles):
    """
    パッチ左上 (x0, y0) の 2×2 ブロックがブラックリストに重なるか判定する。
    """
    for tx in (x0, x0 + 1):
        for ty in (y0, y0 + 1):
            if (tx, ty) in blacklisted_tiles:
                return True
    return False


def load_tile_index(tiles_dir):
    """
    tiles_dir 内の *.npy を走査し、{z: {(x, y): path}} を返す。
    """
    pattern = re.compile(r'^(\d+)_(\d+)_(\d+)\.npy$')
    by_zoom = {}
    for fname in sorted(os.listdir(tiles_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        z, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        by_zoom.setdefault(z, {})[(x, y)] = os.path.join(tiles_dir, fname)
    return by_zoom


def load_patch(tile_map, x0, y0):
    """
    (x0, y0) を左上とする 2×2 タイルブロックを読み込み、
    512×512 float32 配列を返す。

    - 4 枚のうち 1 枚でも存在しなければ None を返す。
    - 欠損値 (-9999.0) および負の標高は 0.0m（海面）にクリップする。

    タイル配置（XYZ 座標: x→東、y→南）:
      (x0,   y0  ) → patch[  0:256,   0:256]
      (x0+1, y0  ) → patch[  0:256, 256:512]
      (x0,   y0+1) → patch[256:512,   0:256]
      (x0+1, y0+1) → patch[256:512, 256:512]
    """
    patch = np.empty((SAMPLE_SIZE, SAMPLE_SIZE), dtype=np.float32)
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        key = (x0 + dx, y0 + dy)
        if key not in tile_map:
            return None
        try:
            tile = np.load(tile_map[key]).astype(np.float32)
        except Exception:
            return None
        if tile.shape != (TILE_SIZE, TILE_SIZE):
            return None
        np.clip(tile, 0.0, None, out=tile)  # 欠損値(-9999)・負の標高 → 0m にクリップ
        r, c = dy * TILE_SIZE, dx * TILE_SIZE
        patch[r:r + TILE_SIZE, c:c + TILE_SIZE] = tile
    return patch


def clean_sample(patch):
    """
    512×512 float32 パッチをフィルタリングし、[0, 1] 正規化した配列を返す。
    不適切なパッチは None を返す。
    """
    # 高低差が小さすぎる（平坦地）
    elev_range = float(patch.max() - patch.min())
    if elev_range < MIN_ELEV_RANGE:
        return None

    # 最小値付近のピクセルが多い（水域・平野）
    near_min_frac = float((patch < patch.min() + 8.0).sum()) / patch.size
    if near_min_frac > MAX_NEAR_MIN_FRAC:
        return None

    # Shannon エントロピーが低い（データ破損・均質データ）
    if skimage.measure.shannon_entropy(patch) < MIN_ENTROPY:
        return None

    # [0, 1] に正規化（パッチ内相対値）
    return (patch - patch.min()) / elev_range


def get_variants(a):
    """
    4 回転 × (元画像 + 転置) = 8 通りのバリエーションを yield する。
    地形に方向性の偏りがないことを前提とした ×8 データ拡張。
    """
    for b in (a, a.T):
        for k in range(4):
            yield np.rot90(b, k)


def save_png(a, path):
    """[0, 1] float32 配列を 16bit グレースケール PNG として保存する。"""
    Image.fromarray(np.round(a * 65535).astype(np.uint16)).save(path)


# --------------------------------------------------------------------- main --

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    by_zoom = load_tile_index(RAW_TILES_DIR)
    if not by_zoom:
        print(f'エラー: {RAW_TILES_DIR} にタイルが見つかりません')
        return

    training_id = 0

    for z in sorted(by_zoom):
        tile_map = by_zoom[z]
        xs = sorted({x for x, _ in tile_map})
        ys = sorted({y for _, y in tile_map})
        print(f'\nズームレベル {z}: {len(tile_map)} タイル  '
              f'x=[{xs[0]}..{xs[-1]}]  y=[{ys[0]}..{ys[-1]}]')

        # ブラックリストのタイル集合を構築
        blacklisted_tiles = build_blacklist_tileset(BLACKLIST.get(z, []))
        print(f'  ブラックリスト: {len(BLACKLIST.get(z, []))} エリア '
              f'({len(blacklisted_tiles)} タイル)')

        accepted        = 0
        skip_missing    = 0   # タイルファイル欠損で除外
        skip_blacklist  = 0   # ブラックリストで除外
        skip_filter     = 0   # フィルタリングで除外

        total_candidates = len(xs) * len(ys)
        with tqdm(total=total_candidates, unit='patch',
                  desc=f'z={z}') as pbar:
            for x in xs:
                for y in ys:
                    # ブラックリスト判定
                    if is_blacklisted(x, y, blacklisted_tiles):
                        skip_blacklist += 1
                        pbar.update(1)
                        continue

                    # 2×2 ブロックを読み込む
                    patch = load_patch(tile_map, x, y)
                    if patch is None:
                        skip_missing += 1
                        pbar.update(1)
                        continue

                    # フィルタリング & 正規化
                    cleaned = clean_sample(patch)
                    if cleaned is None:
                        skip_filter += 1
                        pbar.update(1)
                        continue

                    # 8 バリエーションを保存
                    for variant in get_variants(cleaned):
                        save_png(variant,
                                 os.path.join(OUTPUT_DIR, f'{training_id}.png'))
                        training_id += 1
                    accepted += 1
                    pbar.update(1)
                    pbar.set_postfix(accepted=accepted, images=training_id)

        print(f'  候補パッチ数        : {total_candidates}')
        print(f'  採用                : {accepted}  ({accepted * 8} 枚、×8バリエーション込み)')
        print(f'  除外(タイル欠損)    : {skip_missing}')
        print(f'  除外(ブラックリスト): {skip_blacklist}')
        print(f'  除外(フィルタ)      : {skip_filter}')

    print(f'\n完了。{training_id} 枚のトレーニング画像を {OUTPUT_DIR}/ に書き出しました。')


if __name__ == '__main__':
    main()
