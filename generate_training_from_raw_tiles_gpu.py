#!/usr/bin/env python3

import os
import re
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ------------------------------ 設定

RAW_TILES_DIR = "./raw_tiles"
OUTPUT_DIR = "./training_samples"

MISSING_VALUE = -9999.0
TILE_SIZE = 256
SAMPLE_SIZE = TILE_SIZE * 2

MIN_ELEV_RANGE = 40.0
MAX_NEAR_MIN_FRAC = 0.20
MIN_ENTROPY = 10.0

MAX_ELEV = 1600.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = os.cpu_count()

# ------------------------------ blacklist

BLACKLIST = {
    12: [
        {"name":"富士山","x_range":(3624,3628),"y_range":(1615,1620)},
        {"name":"大山","x_range":(3566,3569),"y_range":(1615,1618)},
    ]
}

# ------------------------------ utilities


def build_blacklist_tileset(areas):

    blacklisted = set()

    for area in areas:
        x0,x1 = area["x_range"]
        y0,y1 = area["y_range"]

        for x in range(x0,x1+1):
            for y in range(y0,y1+1):
                blacklisted.add((x,y))

    return blacklisted


def is_blacklisted(x0,y0,blacklisted):

    for tx in (x0,x0+1):
        for ty in (y0,y0+1):

            if (tx,ty) in blacklisted:
                return True

    return False


# ------------------------------ tile loading


def load_patch(tile_map,x0,y0):

    patch = np.empty((SAMPLE_SIZE,SAMPLE_SIZE),dtype=np.float32)

    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:

        key=(x0+dx,y0+dy)

        if key not in tile_map:
            return None

        try:
            tile = np.load(tile_map[key], mmap_mode="r")
        except:
            return None

        if tile.shape!=(TILE_SIZE,TILE_SIZE):
            return None

        tile=np.clip(tile,0,None)

        r=dy*TILE_SIZE
        c=dx*TILE_SIZE

        patch[r:r+TILE_SIZE,c:c+TILE_SIZE]=tile

    return torch.from_numpy(patch).to(DEVICE)


# ------------------------------ filtering


def clean_sample(patch):

    elev_range=(patch.max()-patch.min()).item()

    if elev_range < MIN_ELEV_RANGE:
        return None

    near_min_frac=((patch < patch.min()+8).float().mean()).item()

    if near_min_frac > MAX_NEAR_MIN_FRAC:
        return None

    hist=torch.histc(patch,bins=256,min=0,max=3000)

    p=hist/hist.sum()

    entropy=-(p*torch.log2(p+1e-12)).sum().item()

    if entropy < MIN_ENTROPY:
        return None

    p_max=patch.max()

    if p_max > MAX_ELEV:

        patch = patch-(p_max-MAX_ELEV)

        p_min=patch.min()

        if p_min < 0:

            patch=(patch-p_min)/(MAX_ELEV-p_min)*MAX_ELEV

    return patch/MAX_ELEV


# ------------------------------ augmentation


def get_variants(a):

    variants=[]

    for b in (a,a.T):

        for k in range(4):

            variants.append(torch.rot90(b,k,[0,1]))

    return variants


# ------------------------------ save


def save_png(a,path):

    a=a.cpu().numpy()

    img=np.round(a*65535).astype(np.uint16)

    Image.fromarray(img).save(path)


# ------------------------------ tile index


def load_tile_index(dir):

    pattern=re.compile(r'^(\d+)_(\d+)_(\d+)\.npy$')

    by_zoom={}

    for fname in os.listdir(dir):

        m=pattern.match(fname)

        if not m:
            continue

        z,x,y=map(int,m.groups())

        by_zoom.setdefault(z,{})[(x,y)] = os.path.join(dir,fname)

    return by_zoom


# ------------------------------ worker


def process_candidate(args):

    tile_map,x,y,blacklisted=args

    if is_blacklisted(x,y,blacklisted):
        return []

    patch=load_patch(tile_map,x,y)

    if patch is None:
        return []

    cleaned=clean_sample(patch)

    if cleaned is None:
        return []

    variants=get_variants(cleaned)

    return variants


# ------------------------------ main


def main():

    os.makedirs(OUTPUT_DIR,exist_ok=True)

    by_zoom=load_tile_index(RAW_TILES_DIR)

    training_id=0

    for z in sorted(by_zoom):

        tile_map=by_zoom[z]

        xs=sorted({x for x,_ in tile_map})
        ys=sorted({y for _,y in tile_map})

        print(f"\nzoom {z} tiles {len(tile_map)}")

        blacklisted=build_blacklist_tileset(BLACKLIST.get(z,[]))

        candidates=[]

        for x in xs:
            for y in ys:
                candidates.append((tile_map,x,y,blacklisted))

        with ProcessPoolExecutor(NUM_WORKERS) as executor:

            for variants in tqdm(
                executor.map(process_candidate,candidates),
                total=len(candidates)
            ):

                for v in variants:

                    path=os.path.join(
                        OUTPUT_DIR,
                        f"{training_id}.png"
                    )

                    save_png(v,path)

                    training_id+=1

    print("\ncompleted")
    print("images:",training_id)


if __name__=="__main__":
    main()