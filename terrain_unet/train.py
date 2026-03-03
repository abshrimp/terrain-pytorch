#!/usr/bin/env python3
"""
pix2pix 地形学習スクリプト

training_samples/ の 16bit グレースケール PNG から、
ランダムなガウシアンブラーで (smooth, detail) ペアを自動生成して学習します。

使い方:
    # 基本 (MPS, workers=0 推奨)
    python terrain_unet/train.py --datadir=training_samples --workers=0

    # RTX 4090 (AMP + 大バッチ)
    python terrain_unet/train.py --datadir=training_samples --batch=8 --amp

    # L1 損失のみ (GAN なし, より安定)
    python terrain_unet/train.py --datadir=training_samples --no-gan --workers=0

    # チェックポイントから再開
    python terrain_unet/train.py --datadir=training_samples --resume=terrain_unet/checkpoints/latest.pt
"""

import os
import sys
import glob
import random
from contextlib import nullcontext

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_unet.model import UNetGenerator, PatchDiscriminator, get_device


# ----------------------------------------------------------------
class TerrainDataset(Dataset):
    """
    16bit グレースケール地形画像のデータセット。
    __getitem__ でランダムなガウシアンブラーをかけ (smooth, detail) ペアを返す。
    """
    def __init__(
        self,
        image_dir: str,
        size: int = 512,
        sigma_min: float = 5.0,
        sigma_max: float = 60.0,
    ):
        self.files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        assert len(self.files) > 0, f"PNG が見つかりません: {image_dir}"
        self.size = size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        print(f"Dataset: {len(self.files)} 枚  sigma=[{sigma_min}, {sigma_max}]")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx], cv2.IMREAD_UNCHANGED)  # uint16 [H, W]

        # uint16 → float32 [-1, 1]
        img_f = img.astype(np.float32) / 32767.5 - 1.0

        # リサイズ (必要な場合)
        if img_f.shape[0] != self.size or img_f.shape[1] != self.size:
            img_f = cv2.resize(img_f, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # ランダムなガウシアンブラー (強さを毎回変える)
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        ksize = int(sigma * 6) | 1  # 奇数保証
        smooth_f = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)

        # データ拡張: 90°回転 + 水平フリップ
        k = random.randint(0, 3)
        img_f    = np.rot90(img_f,    k).copy()
        smooth_f = np.rot90(smooth_f, k).copy()
        if random.random() > 0.5:
            img_f    = img_f[:, ::-1].copy()
            smooth_f = smooth_f[:, ::-1].copy()

        # [H, W] → [1, H, W] tensor
        detail = torch.from_numpy(img_f[None]).float()
        smooth = torch.from_numpy(smooth_f[None]).float()
        return smooth, detail


# ----------------------------------------------------------------
def save_sample(G, device, sample_smooth, sample_detail, path):
    """学習進捗を可視化する比較画像を保存"""
    G.eval()
    with torch.no_grad():
        fake = G(sample_smooth.to(device)).cpu()
    G.train()

    n = min(4, sample_smooth.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    if n == 1:
        axes = axes[:, None]

    for row, (data, title) in enumerate(zip(
        [sample_smooth, fake, sample_detail],
        ["入力 (smooth)", "生成 (fake)", "正解 (real)"],
    )):
        for col in range(n):
            axes[row, col].imshow(data[col, 0].numpy(), cmap="terrain", vmin=-1, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------
@click.command()
@click.option("--datadir",    default="training_samples",         show_default=True, help="学習画像ディレクトリ")
@click.option("--outdir",     default="terrain_unet/checkpoints", show_default=True, help="チェックポイント保存先")
@click.option("--epochs",     default=200,  show_default=True, type=int,   help="総エポック数")
@click.option("--batch",      default=4,    show_default=True, type=int,   help="バッチサイズ (RTX 4090 なら 8〜16)")
@click.option("--lr",         default=2e-4, show_default=True, type=float, help="初期学習率")
@click.option("--lambda-l1",  default=100.0,show_default=True, type=float, help="L1 損失の重み")
@click.option("--no-gan",     is_flag=True, default=False, help="GAN なし (L1 損失のみ, より安定)")
@click.option("--amp",        is_flag=True, default=False, help="AMP (混合精度) 使用 [CUDA のみ]")
@click.option("--workers",    default=4,    show_default=True, type=int,   help="DataLoader ワーカー数 (MPS では 0 推奨)")
@click.option("--ngf",        default=64,   show_default=True, type=int,   help="Generator ベースチャンネル数")
@click.option("--resume",     default=None, metavar="FILE",               help="チェックポイントから再開")
@click.option("--save-every", default=10,   show_default=True, type=int,   help="N エポックごとにチェックポイント保存")
@click.option("--sigma-min",  default=5.0,  show_default=True, type=float, help="ブラー sigma 最小値")
@click.option("--sigma-max",  default=60.0, show_default=True, type=float, help="ブラー sigma 最大値")
def train(
    datadir, outdir, epochs, batch, lr, lambda_l1,
    no_gan, amp, workers, ngf, resume, save_every,
    sigma_min, sigma_max,
):
    """平滑化地形 → 詳細地形の pix2pix モデルを学習します。"""

    device = get_device()
    print(f"デバイス: {device}")

    use_amp = amp and str(device) == "cuda"
    if amp and not use_amp:
        print("AMP は CUDA のみ有効。スキップします。")

    # --- データ ---
    dataset = TerrainDataset(datadir, sigma_min=sigma_min, sigma_max=sigma_max)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=(str(device) == "cuda"),
        persistent_workers=False,
        drop_last=True,
    )
    print(f"バッチ数/エポック: {len(loader)}")

    # 進捗可視化用のサンプルを固定
    sample_smooth, sample_detail = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))

    # --- モデル ---
    G = UNetGenerator(in_ch=1, out_ch=1, ngf=ngf).to(device)
    D = PatchDiscriminator(in_ch=2).to(device) if not no_gan else None
    print(f"Generator パラメータ数: {sum(p.numel() for p in G.parameters()):,}")

    # --- オプティマイザ ---
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999)) if D else None

    # 学習率: 前半 constant → 後半 linear decay (pix2pix 標準)
    decay_start = epochs // 2
    def lr_lambda(ep):
        if ep < decay_start:
            return 1.0
        return max(0.0, 1.0 - (ep - decay_start) / max(1, epochs - decay_start))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda) if opt_D else None

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 損失: LSGAN (MSE) の方が BCE より安定
    criterion_adv = nn.MSELoss()
    criterion_l1  = nn.L1Loss()

    # --- チェックポイント再開 ---
    start_epoch = 0
    os.makedirs(outdir, exist_ok=True)
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        opt_G.load_state_dict(ckpt["opt_G"])
        sched_G.load_state_dict(ckpt["sched_G"])
        if D and "D" in ckpt:
            D.load_state_dict(ckpt["D"])
            opt_D.load_state_dict(ckpt["opt_D"])
            sched_D.load_state_dict(ckpt["sched_D"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"再開: epoch {start_epoch}")

    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    # --- 学習ループ ---
    epoch_bar = tqdm(range(start_epoch, epochs), desc="Epoch", unit="ep")
    for epoch in epoch_bar:
        G.train()
        if D:
            D.train()

        sum_G = 0.0
        sum_D = 0.0

        batch_bar = tqdm(loader, desc=f"  Ep {epoch+1:>4d}", leave=False, unit="batch")
        for i, (smooth, detail) in enumerate(batch_bar):
            smooth = smooth.to(device)
            detail = detail.to(device)

            # ---- Discriminator ----
            if D:
                with amp_ctx():
                    fake_detach = G(smooth).detach()
                    real_pred = D(smooth, detail)
                    fake_pred = D(smooth, fake_detach)
                    ones  = torch.ones_like(real_pred)
                    zeros = torch.zeros_like(fake_pred)
                    loss_D = 0.5 * (criterion_adv(real_pred, ones) + criterion_adv(fake_pred, zeros))

                opt_D.zero_grad()
                scaler.scale(loss_D).backward()
                scaler.step(opt_D)
                sum_D += loss_D.item()

            # ---- Generator ----
            with amp_ctx():
                fake = G(smooth)
                loss_l1 = criterion_l1(fake, detail) * lambda_l1
                if D:
                    fake_pred = D(smooth, fake)
                    loss_G = criterion_adv(fake_pred, torch.ones_like(fake_pred)) + loss_l1
                else:
                    loss_G = loss_l1

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            sum_G += loss_G.item()

            # バッチバーのリアルタイム損失表示
            postfix = {"G": f"{sum_G/(i+1):.4f}"}
            if D:
                postfix["D"] = f"{sum_D/(i+1):.4f}"
            batch_bar.set_postfix(postfix)

        sched_G.step()
        if sched_D:
            sched_D.step()

        avg_G = sum_G / len(loader)
        postfix = {"G": f"{avg_G:.4f}", "lr": f"{opt_G.param_groups[0]['lr']:.2e}"}
        if D:
            postfix["D"] = f"{sum_D/len(loader):.4f}"
        epoch_bar.set_postfix(postfix)

        # サンプル画像保存
        save_sample(
            G, device, sample_smooth, sample_detail,
            os.path.join(outdir, f"sample_ep{epoch+1:04d}.png"),
        )

        # チェックポイント保存
        ckpt = {
            "epoch": epoch,
            "ngf": ngf,
            "G": G.state_dict(),
            "opt_G": opt_G.state_dict(),
            "sched_G": sched_G.state_dict(),
        }
        if D:
            ckpt["D"] = D.state_dict()
            ckpt["opt_D"] = opt_D.state_dict()
            ckpt["sched_D"] = sched_D.state_dict()

        torch.save(ckpt, os.path.join(outdir, "latest.pt"))
        if (epoch + 1) % save_every == 0:
            path = os.path.join(outdir, f"epoch_{epoch+1:04d}.pt")
            torch.save(ckpt, path)
            print(f"  → 保存: {path}")

    print("学習完了！")


if __name__ == "__main__":
    train()
