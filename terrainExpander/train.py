#!/usr/bin/env python3
"""
train.py  TerrainExpander 学習スクリプト

損失関数:
  L_GAN     (PatchGAN)         生成品質
  L_L1      (L1 loss × 100)   低周波構造の保持
  L_edge    (Sobel edge loss)  稜線・谷の鮮明さ
"""

import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import model as M   # model.py と同ディレクトリに置く

# ---------------------------------------------------------------- config --

SAMPLES_DIR = "./context_samples"
CKPT_DIR    = "./checkpoints"
LOG_EVERY   = 100
SAVE_EVERY  = 5     # epoch

BATCH_SIZE  = 4
LR          = 2e-4
BETAS       = (0.5, 0.999)
EPOCHS      = 100
BASE_CH_G   = 32
BASE_CH_D   = 64
LAMBDA_L1   = 100
LAMBDA_EDGE = 10

DEVICE = (torch.device("cuda")   if torch.cuda.is_available() else
          torch.device("mps")    if torch.backends.mps.is_available() else
          torch.device("cpu"))


# ---------------------------------------------------------------- dataset --

class ContextDataset(Dataset):
    def __init__(self, samples_dir):
        self.ids = sorted({
            os.path.basename(f).split("_")[0]
            for f in glob.glob(f"{samples_dir}/*_target.png")
        })
        self.d = samples_dir

    def __len__(self): return len(self.ids)

    def _load(self, path):
        img = np.array(Image.open(path)).astype(np.float32) / 65535.0
        return torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

    def __getitem__(self, i):
        sid = self.ids[i]
        return {
            "topleft": self._load(f"{self.d}/{sid}_ctx_topleft.png"),
            "top":     self._load(f"{self.d}/{sid}_ctx_top.png"),
            "left":    self._load(f"{self.d}/{sid}_ctx_left.png"),
            "target":  self._load(f"{self.d}/{sid}_target.png"),
        }


# ----------------------------------------------------------------- losses --

def sobel_edge(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                       dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = kx.transpose(2,3)
    ex = F.conv2d(x, kx, padding=1)
    ey = F.conv2d(x, ky, padding=1)
    return torch.sqrt(ex**2 + ey**2 + 1e-8)

def edge_loss(pred, target):
    return F.l1_loss(sobel_edge(pred), sobel_edge(target))

def gan_loss(pred, is_real):
    label = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, label)


# ------------------------------------------------------------------ train --

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    ds     = ContextDataset(SAMPLES_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)
    print(f"Dataset: {len(ds)} samples  Device: {DEVICE}")

    G = M.TerrainExpander(BASE_CH_G).to(DEVICE)
    D = M.PatchDiscriminator(BASE_CH_D).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR,    betas=BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR*0.5, betas=BETAS)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=EPOCHS)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS+1):
        G.train(); D.train()
        g_losses, d_losses = [], []

        for step, batch in enumerate(tqdm(loader, desc=f"Ep{epoch}")):
            tl  = batch["topleft"].to(DEVICE)
            t   = batch["top"].to(DEVICE)
            l   = batch["left"].to(DEVICE)
            tgt = batch["target"].to(DEVICE)
            ctx = [tl, t, l]

            # ---- Discriminator ----
            with torch.no_grad():
                fake = G(tl, t, l)
            d_real = D(ctx, tgt)
            d_fake = D(ctx, fake.detach())
            loss_D = (gan_loss(d_real, True) + gan_loss(d_fake, False)) * 0.5
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ---- Generator ----
            fake   = G(tl, t, l)
            d_fake = D(ctx, fake)
            loss_G = (gan_loss(d_fake, True)
                      + LAMBDA_L1   * F.l1_loss(fake, tgt)
                      + LAMBDA_EDGE * edge_loss(fake, tgt))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

            if step % LOG_EVERY == 0:
                print(f"  step={step} G={loss_G.item():.4f} "
                      f"D={loss_D.item():.4f}")

        scheduler_G.step()
        mean_G = np.mean(g_losses)
        print(f"Epoch {epoch}: G={mean_G:.4f}  D={np.mean(d_losses):.4f}")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
            }, f"{CKPT_DIR}/ckpt_ep{epoch:04d}.pt")

        if mean_G < best_loss:
            best_loss = mean_G
            torch.save(G.state_dict(), f"{CKPT_DIR}/best_G.pt")
            print(f"  → best model saved (G={best_loss:.4f})")

if __name__ == "__main__":
    train()
