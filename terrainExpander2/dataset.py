# dataset.py
import os
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from config import BLACKLIST, IMAGE_SIZE, ELEVATION_MIN, ELEVATION_MAX

def is_blacklisted(z, x, y):
    if z not in BLACKLIST: return False
    for area in BLACKLIST[z]:
        x_min, x_max = area["x_range"]
        y_min, y_max = area["y_range"]
        if (x_min <= x <= x_max) and (y_min <= y <= y_max): return True
    return False

def process_and_convert_tile(npy_file_path):
    basename = os.path.basename(npy_file_path)
    match = re.match(r"(\d+)_(\d+)_(\d+)\.npy", basename)
    if not match: return None
    z, x, y = map(int, match.groups())
    
    if is_blacklisted(z, x, y): return None
        
    data = np.load(npy_file_path)
    data = np.clip(data, ELEVATION_MIN, ELEVATION_MAX)
    normalized = data / ELEVATION_MAX
    data_16bit = (normalized * 65535.0).astype(np.uint16)
    return data_16bit

class TerrainOutpaintingDataset(Dataset):
    def __init__(self, file_paths):
        self.valid_data = []
        print("データセットを構築中...")
        for path in tqdm(file_paths):
            tile_16bit = process_and_convert_tile(path)
            if tile_16bit is not None:
                self.valid_data.append(tile_16bit)
        print(f"有効なタイル数: {len(self.valid_data)}")

    def __len__(self):
        return len(self.valid_data)
        
    def generate_mask(self, size=IMAGE_SIZE):
        mask = torch.zeros((1, size, size), dtype=torch.float32)
        direction = random.choice([
            'top', 'bottom', 'left', 'right', 
            'top_left', 'top_right', 'bottom_left', 'bottom_right'
        ])
        ctx = 64 
        if 'top' in direction: mask[:, :ctx, :] = 1.0
        if 'bottom' in direction: mask[:, -ctx:, :] = 1.0
        if 'left' in direction: mask[:, :, :ctx] = 1.0
        if 'right' in direction: mask[:, :, -ctx:] = 1.0
        return mask

    def __getitem__(self, idx):
        data_uint16 = self.valid_data[idx]
        data_float = data_uint16.astype(np.float32) / 65535.0
        target_terrain = torch.from_numpy(data_float).unsqueeze(0)
        
        mask = self.generate_mask(size=IMAGE_SIZE)
        masked_terrain = target_terrain * mask
        input_tensor = torch.cat([masked_terrain, mask], dim=0)
        
        return input_tensor, target_terrain