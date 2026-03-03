#!/usr/bin/env python3
"""
既存の地形画像にガウシアンブラーをかけて平滑化画像を生成するユーティリティ。
生成した smooth_terrain.png を project_terrain.py の入力として使います。

使い方:
    python make_smooth_terrain.py --input=../images/gen_mps/seed10000.png --sigma=30
    python make_smooth_terrain.py --input=../images/gen_mps/seed10000.png --sigma=30 --outdir=../images/smooth
"""

import os
import click
import cv2
import numpy as np
import matplotlib.pyplot as plt


@click.command()
@click.option("--input",  "input_fname", required=True, metavar="FILE", help="元の地形画像 (16bit PNG など)")
@click.option("--sigma",  default=30.0,  show_default=True, type=float, help="ガウシアンブラーの強さ (大きいほど平滑化)")
@click.option("--outdir", default=".",   show_default=True, metavar="DIR", help="出力ディレクトリ")
@click.option("--name",   default="smooth_terrain", show_default=True, help="出力ファイル名 (拡張子なし)")
def make_smooth(input_fname: str, sigma: float, outdir: str, name: str):
    """
    地形画像をガウシアンブラーで平滑化します。

    例:
        # seed10000.png をぼかして smooth_terrain.png を作成
        python make_smooth_terrain.py --input=../images/gen_mps/seed10000.png --sigma=30

        # 強くぼかす
        python make_smooth_terrain.py --input=../images/gen_mps/seed10000.png --sigma=80

    出力ファイル:
        <outdir>/<name>.png         ← 平滑化された 16bit グレースケール PNG
        <outdir>/<name>_compare.png ← 元画像 vs 平滑化画像の比較
    """
    img = cv2.imread(input_fname, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {input_fname}")

    print(f"入力: {input_fname}")
    print(f"  dtype={img.dtype}, shape={img.shape}")

    # グレースケール化
    if img.ndim == 3:
        # BGR → グレースケール (16bit 対応)
        gray = img.mean(axis=2).astype(img.dtype)
    else:
        gray = img

    # ガウシアンブラー
    # kernel size は sigma の 6 倍以上の奇数にする
    ksize = int(sigma * 6) | 1  # 奇数保証
    blurred = cv2.GaussianBlur(gray.astype(np.float64), (ksize, ksize), sigma)
    blurred = blurred.clip(0, np.iinfo(img.dtype).max).astype(img.dtype)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{name}.png")
    cv2.imwrite(out_path, blurred)
    print(f"保存: {out_path}  (sigma={sigma})")

    # 比較画像を保存
    def to_float(arr):
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 65535.0
        return arr.astype(np.float32) / 255.0

    orig_f = to_float(gray)
    blur_f = to_float(blurred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig_f, cmap="terrain", vmin=0, vmax=1)
    axes[0].set_title("元画像")
    axes[0].axis("off")
    axes[1].imshow(blur_f, cmap="terrain", vmin=0, vmax=1)
    axes[1].set_title(f"平滑化 (sigma={sigma})")
    axes[1].axis("off")
    plt.tight_layout()
    compare_path = os.path.join(outdir, f"{name}_compare.png")
    plt.savefig(compare_path, bbox_inches="tight")
    plt.close()
    print(f"比較画像: {compare_path}")
    print()
    print("次のコマンドで GAN Inversion を実行できます:")
    print(f"  python project_terrain.py --network=<snap.pkl> --target={out_path} --outdir=out/projected")


if __name__ == "__main__":
    make_smooth()
