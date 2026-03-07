#!/usr/bin/env python3
"""
generate_training_from_raw_tiles.py (究極高速化版・修正済み)

- ProcessPoolExecutorによる完全マルチプロセス化 (GIL回避)
- PNG低圧縮保存によるCPU負荷軽減
- GPU上で正確なシャノンエントロピー計算（バグ修正済）
"""

import os
import re
import numpy as np
import cupy as cp
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp

# ------------------------------------------------------------------ 設定 --

RAW_TILES_DIR = './raw_tiles'
OUTPUT_DIR    = './training_samples'

MISSING_VALUE = -9999.0
TILE_SIZE     = 256
SAMPLE_SIZE   = TILE_SIZE * 2   # = 512

MIN_ELEV_RANGE    = 40.0
MAX_NEAR_MIN_FRAC = 0.20
MIN_ENTROPY       = 10.0

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
    blacklisted = set()
    for area in areas:
        x0_bl, x1_bl = area['x_range']
        y0_bl, y1_bl = area['y_range']
        for bx in range(x0_bl, x1_bl + 1):
            for by in range(y0_bl, y1_bl + 1):
                blacklisted.add((bx, by))
    return blacklisted

def is_blacklisted(x0, y0, blacklisted_tiles):
    for tx in (x0, x0 + 1):
        for ty in (y0, y0 + 1):
            if (tx, ty) in blacklisted_tiles:
                return True
    return False

def load_tile_index(tiles_dir):
    pattern = re.compile(r'^(\d+)_(\d+)_(\d+)\.npy$')
    by_zoom = {}
    for fname in sorted(os.listdir(tiles_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        z, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        by_zoom.setdefault(z, {})[(x, y)] = os.path.join(tiles_dir, fname)
    return by_zoom

def load_patch_cpu(tile_map, x0, y0):
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
        
        np.clip(tile, 0.0, None, out=tile)
        r, c = dy * TILE_SIZE, dx * TILE_SIZE
        patch[r:r + TILE_SIZE, c:c + TILE_SIZE] = tile
    return patch

def gpu_shannon_entropy(patch_cp):
    """元の skimage 互換の計算（正確なエントロピーを算出）"""
    _, counts = cp.unique(patch_cp, return_counts=True)
    p = counts / counts.sum()
    return -cp.sum(p * cp.log2(p))

def clean_sample_gpu(patch_np):
    patch = cp.asarray(patch_np)

    elev_range = float(patch.max() - patch.min())
    if elev_range < MIN_ELEV_RANGE:
        return None

    near_min_frac = float((patch < patch.min() + 8.0).sum()) / patch.size
    if near_min_frac > MAX_NEAR_MIN_FRAC:
        return None

    if gpu_shannon_entropy(patch) < MIN_ENTROPY:
        return None

    MAX_ELEV = 1600.0
    p_max = float(patch.max())
    if p_max > MAX_ELEV:
        patch = patch - (p_max - MAX_ELEV)
        p_min = float(patch.min())
        if p_min < 0.0:
            patch = (patch - p_min) / (MAX_ELEV - p_min) * MAX_ELEV
            
    return patch / MAX_ELEV

def get_variants_gpu(a_cp):
    for b in (a_cp, a_cp.T):
        for k in range(4):
            yield cp.rot90(b, k)

def save_png_from_gpu(a_cp, path):
    a_np = cp.asnumpy(a_cp)
    # compress_level=1 を指定してPNG圧縮のCPU計算を大幅にカット
    Image.fromarray(np.round(a_np * 65535).astype(np.uint16)).save(path, compress_level=1)

# -------------------------------------------------------- ワーカー処理 --

def process_single_patch(args):
    x, y, tile_map, blacklisted_tiles, current_id = args

    if is_blacklisted(x, y, blacklisted_tiles):
        return 'blacklist', 0

    patch_np = load_patch_cpu(tile_map, x, y)
    if patch_np is None:
        return 'missing', 0

    cleaned_cp = clean_sample_gpu(patch_np)
    if cleaned_cp is None:
        return 'filter', 0

    saved_count = 0
    for i, variant_cp in enumerate(get_variants_gpu(cleaned_cp)):
        save_path = os.path.join(OUTPUT_DIR, f'{current_id + i}.png')
        save_png_from_gpu(variant_cp, save_path)
        saved_count += 1
        
    return 'accepted', saved_count


# --------------------------------------------------------------------- main --

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    by_zoom = load_tile_index(RAW_TILES_DIR)
    if not by_zoom:
        print(f'エラー: {RAW_TILES_DIR} にタイルが見つかりません')
        return

    # CPUのコア数に応じて調整（推奨：論理コア数と同じか少し少なめ）
    MAX_WORKERS = max(1, os.cpu_count() - 2)

    for z in sorted(by_zoom):
        tile_map = by_zoom[z]
        xs = sorted({x for x, _ in tile_map})
        ys = sorted({y for _, y in tile_map})
        print(f'\nズームレベル {z}: {len(tile_map)} タイル  '
              f'x=[{xs[0]}..{xs[-1]}]  y=[{ys[0]}..{ys[-1]}]')

        blacklisted_tiles = build_blacklist_tileset(BLACKLIST.get(z, []))
        
        accepted = 0
        skip_missing = 0
        skip_blacklist = 0
        skip_filter = 0

        tasks = []
        training_id = 0
        for x in xs:
            for y in ys:
                tasks.append((x, y, tile_map, blacklisted_tiles, training_id))
                training_id += 8

        total_candidates = len(tasks)
        
        print(f'  マルチプロセス({MAX_WORKERS}プロセス)で処理を開始します...')
        
        # マルチプロセスでCUDAを安全に使うために 'spawn' コンテキストを使用
        ctx = mp.get_context('spawn')
        
        with tqdm(total=total_candidates, unit='patch', desc=f'z={z}') as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
                futures = [executor.submit(process_single_patch, task) for task in tasks]
                
                for future in concurrent.futures.as_completed(futures):
                    result_type, _ = future.result()
                    
                    if result_type == 'blacklist':
                        skip_blacklist += 1
                    elif result_type == 'missing':
                        skip_missing += 1
                    elif result_type == 'filter':
                        skip_filter += 1
                    elif result_type == 'accepted':
                        accepted += 1
                        
                    pbar.update(1)
                    pbar.set_postfix(accepted=accepted)

        print(f'  候補パッチ数        : {total_candidates}')
        print(f'  採用                : {accepted}  ({accepted * 8} 枚、×8バリエーション込み)')
        print(f'  除外(タイル欠損)    : {skip_missing}')
        print(f'  除外(ブラックリスト): {skip_blacklist}')
        print(f'  除外(フィルタ)      : {skip_filter}')

    print(f'\n完了。トレーニング画像を {OUTPUT_DIR}/ に書き出しました。')

if __name__ == '__main__':
    # Windows等の環境でマルチプロセスを正常に動かすためのおまじない
    mp.freeze_support()
    main()