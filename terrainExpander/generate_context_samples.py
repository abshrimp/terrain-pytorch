#!/usr/bin/env python3
"""
学習データ生成:
  各パッチに対して「左・上・左上」の3枚をコンテキストとして保存する

出力:
  context_samples/
    {id}_ctx_topleft.png   左上パッチ (512×512)
    {id}_ctx_top.png       上パッチ   (512×512)
    {id}_ctx_left.png      左パッチ   (512×512)
    {id}_target.png        予測対象   (512×512)
    meta.csv
"""

import os, re, csv
import numpy as np
import skimage.measure
from PIL import Image
from tqdm import tqdm

RAW_TILES_DIR = "../raw_tiles"
OUTPUT_DIR    = "./context_samples"
TILE_SIZE     = 256
PATCH_PX      = TILE_SIZE * 2
MAX_ELEV      = 1600.0
MISSING_ELV   = -9999.0

# フィルタ閾値（既存スクリプトと同じ）
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

def build_blacklist_tileset(areas):
    s = set()
    for a in areas:
        for bx in range(a["x_range"][0], a["x_range"][1]+1):
            for by in range(a["y_range"][0], a["y_range"][1]+1):
                s.add((bx, by))
    return s

def load_tile_index(d):
    pat = re.compile(r"^(\d+)_(\d+)_(\d+)\.npy$")
    bz = {}
    for f in sorted(os.listdir(d)):
        m = pat.match(f)
        if not m: continue
        z,x,y = int(m.group(1)),int(m.group(2)),int(m.group(3))
        bz.setdefault(z,{})[(x,y)] = os.path.join(d,f)
    return bz

def load_patch(tile_map, x0, y0):
    p = np.empty((PATCH_PX, PATCH_PX), dtype=np.float32)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        key = (x0+dx, y0+dy)
        if key not in tile_map: return None
        try: tile = np.load(tile_map[key]).astype(np.float32)
        except: return None
        if tile.shape != (TILE_SIZE,TILE_SIZE): return None
        np.clip(tile, 0.0, None, out=tile)
        p[dy*TILE_SIZE:(dy+1)*TILE_SIZE, dx*TILE_SIZE:(dx+1)*TILE_SIZE] = tile
    return p

def normalize(patch):
    if float(patch.max()-patch.min()) < MIN_ELEV_RANGE: return None
    if float((patch < patch.min()+8.0).sum())/patch.size > MAX_NEAR_MIN_FRAC: return None
    if skimage.measure.shannon_entropy(patch) < MIN_ENTROPY: return None
    p_max = float(patch.max())
    if p_max > MAX_ELEV:
        patch = patch - (p_max - MAX_ELEV)
        p_min = float(patch.min())
        if p_min < 0.0:
            patch = (patch-p_min)/(MAX_ELEV-p_min)*MAX_ELEV
    return patch / MAX_ELEV

def save16(arr, path):
    Image.fromarray(np.round(arr*65535).astype(np.uint16)).save(path)

def get_variants(tl, t, l, target):
    """4回転 × 反転 = 8通り（4枚を同時に変換）"""
    for flip in [False, True]:
        if flip:
            tl2,t2,l2,tgt2 = np.fliplr(tl),np.fliplr(t),np.fliplr(l),np.fliplr(target)
        else:
            tl2,t2,l2,tgt2 = tl,t,l,target
        for k in range(4):
            yield (np.rot90(tl2,k), np.rot90(t2,k),
                   np.rot90(l2,k),  np.rot90(tgt2,k))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    by_zoom = load_tile_index(RAW_TILES_DIR)
    meta_rows = []
    sid = 0

    for z in sorted(by_zoom):
        tile_map = by_zoom[z]
        bl = build_blacklist_tileset(BLACKLIST.get(z,[]))
        xs = sorted({x for x,_ in tile_map})
        ys = sorted({y for _,y in tile_map})
        print(f"zoom={z}: {len(tile_map)} tiles")

        stats = dict(ok=0, miss=0, filt=0)

        # ステップ=2 (2×2パッチなので隣は+2)
        with tqdm(total=len(xs)*len(ys)) as pbar:
            for x0 in xs:
                for y0 in ys:
                    # 4パッチの座標（左上・上・左・ターゲット）
                    coords = {
                        "topleft": (x0-2, y0-2),
                        "top":     (x0,   y0-2),
                        "left":    (x0-2, y0  ),
                        "target":  (x0,   y0  ),
                    }

                    # ブラックリスト
                    skip = False
                    for name,(cx,cy) in coords.items():
                        for tx in range(cx,cx+2):
                            for ty in range(cy,cy+2):
                                if (tx,ty) in bl: skip=True
                    if skip:
                        pbar.update(1); continue

                    # 読み込み
                    patches = {}
                    for name,(cx,cy) in coords.items():
                        p = load_patch(tile_map, cx, cy)
                        if p is None: break
                        patches[name] = p
                    if len(patches) < 4:
                        stats["miss"] += 1; pbar.update(1); continue

                    # 正規化
                    normed = {}
                    bad = False
                    for name,p in patches.items():
                        n = normalize(p.copy())
                        if n is None: bad=True; break
                        normed[name] = n
                    if bad:
                        stats["filt"] += 1; pbar.update(1); continue

                    # ×8 拡張して保存
                    for vtl,vt,vl,vtgt in get_variants(
                            normed["topleft"], normed["top"],
                            normed["left"],    normed["target"]):
                        save16(vtl,  f"{OUTPUT_DIR}/{sid}_ctx_topleft.png")
                        save16(vt,   f"{OUTPUT_DIR}/{sid}_ctx_top.png")
                        save16(vl,   f"{OUTPUT_DIR}/{sid}_ctx_left.png")
                        save16(vtgt, f"{OUTPUT_DIR}/{sid}_target.png")
                        meta_rows.append(dict(id=sid, z=z,
                            x0=x0, y0=y0))
                        sid += 1

                    stats["ok"] += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=stats["ok"], samples=sid)

        print(f"  ok={stats['ok']} ({stats['ok']*8}samples) "
              f"miss={stats['miss']} filt={stats['filt']}")

    import csv
    with open(f"{OUTPUT_DIR}/meta.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","z","x0","y0"])
        w.writeheader(); w.writerows(meta_rows)
    print(f"完了: {sid} サンプル")

if __name__ == "__main__":
    main()
