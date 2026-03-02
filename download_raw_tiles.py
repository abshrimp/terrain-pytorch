import os
import time
import requests
import numpy as np
from tqdm import tqdm
import traceback

def download_all_tiles(z=12, x_start=3440, x_end=3802, y_start=1457, y_end=1815, output_dir="./raw_tiles"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"タイルの事前一括ダウンロードを開始します (z={z})")
    print(f"保存先: {output_dir}")

    total_tiles = (x_end - x_start) * (y_end - y_start)
    
    with tqdm(total=total_tiles, desc="Downloading") as pbar:
        for y in range(y_start, y_end):
            print(f"Processing row y={y}...")
            for x in range(x_start, x_end):
                filename = os.path.join(output_dir, f"{z}_{x}_{y}.npy")
                
                # 既にダウンロード済みならスキップ
                if os.path.exists(filename):
                    pbar.update(1)
                    continue

                url = f"https://cyberjapandata.gsi.go.jp/xyz/dem/{z}/{x}/{y}.txt"
                try:
                    time.sleep(1) # 待機
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()

                    data = []
                    for line in response.text.strip().split("\n"):
                        # 'e' (No Data) を -9999.0m に置換
                        row = [-9999.0 if v == "e" else float(v) for v in line.split(",")]
                        data.append(row)

                    arr = np.array(data, dtype=np.float32)
                    if arr.shape == (256, 256):
                        np.save(filename, arr) # ローカルに保存
                except requests.exceptions.RequestException:
                    # ネットワークエラー等が起きてもスルーして次へ
                    pass
                except Exception as e:
                    traceback.print_exc()

                pbar.update(1)
                
    print("\nダウンロード完了")

if __name__ == "__main__":
    download_all_tiles()