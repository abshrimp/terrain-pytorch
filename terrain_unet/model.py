"""
pix2pix モデル定義
  - UNetGenerator  : smooth terrain → detailed terrain
  - PatchDiscriminator : 70×70 PatchGAN
"""

import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DownBlock(nn.Module):
    """Conv stride=2 → InstanceNorm → LeakyReLU"""
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsample × 2 → Conv3×3 → InstanceNorm → (Dropout) → ReLU"""
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    pix2pix U-Net Generator (512×512 グレースケール用)

    入力 : [B, 1, 512, 512]  smooth terrain, [-1, 1]
    出力 : [B, 1, 512, 512]  detailed terrain, [-1, 1]

    エンコーダ 8 段 (512 → 2) + デコーダ 8 段 (2 → 512) + スキップ接続
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, ngf: int = 64):
        super().__init__()
        nf = ngf

        # ---- Encoder ----
        # 最初のレイヤーは InstanceNorm なし (pix2pix の標準)
        self.e0 = nn.Sequential(
            nn.Conv2d(in_ch, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )                                          # 512→256, ch: 1→nf
        self.e1 = DownBlock(nf,   nf*2)           # 256→128, ch: nf→nf*2
        self.e2 = DownBlock(nf*2, nf*4)           # 128→64,  ch: nf*2→nf*4
        self.e3 = DownBlock(nf*4, nf*8)           # 64→32,   ch: nf*4→nf*8
        self.e4 = DownBlock(nf*8, nf*8)           # 32→16,   ch: nf*8→nf*8
        self.e5 = DownBlock(nf*8, nf*8)           # 16→8,    ch: nf*8→nf*8
        self.e6 = DownBlock(nf*8, nf*8)           # 8→4,     ch: nf*8→nf*8
        self.e7 = DownBlock(nf*8, nf*8, norm=False)  # 4→2 (bottleneck)

        # ---- Decoder (スキップ接続で入力チャンネルが 2 倍) ----
        self.d1 = UpBlock(nf*8,    nf*8, dropout=True)  # 2→4,   in=nf*8
        self.d2 = UpBlock(nf*8*2,  nf*8, dropout=True)  # 4→8,   in=nf*8*2 (d1+e6)
        self.d3 = UpBlock(nf*8*2,  nf*8, dropout=True)  # 8→16,  in=nf*8*2 (d2+e5)
        self.d4 = UpBlock(nf*8*2,  nf*8)                # 16→32, in=nf*8*2 (d3+e4)
        self.d5 = UpBlock(nf*8*2,  nf*4)                # 32→64, in=nf*8*2 (d4+e3)
        self.d6 = UpBlock(nf*4*2,  nf*2)                # 64→128, in=nf*4*2 (d5+e2)
        self.d7 = UpBlock(nf*2*2,  nf)                  # 128→256, in=nf*2*2 (d6+e1)
        self.d8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 256→512
            nn.Conv2d(nf*2, out_ch, 3, 1, 1),  # in=nf*2 (d7+e0)
            nn.Tanh(),
        )

    def forward(self, x):
        e0 = self.e0(x)   # [B, nf,   256, 256]
        e1 = self.e1(e0)  # [B, nf*2, 128, 128]
        e2 = self.e2(e1)  # [B, nf*4,  64,  64]
        e3 = self.e3(e2)  # [B, nf*8,  32,  32]
        e4 = self.e4(e3)  # [B, nf*8,  16,  16]
        e5 = self.e5(e4)  # [B, nf*8,   8,   8]
        e6 = self.e6(e5)  # [B, nf*8,   4,   4]
        e7 = self.e7(e6)  # [B, nf*8,   2,   2]

        d1 = self.d1(e7)                        # [B, nf*8,   4,   4]
        d2 = self.d2(torch.cat([d1, e6], 1))    # [B, nf*8,   8,   8]
        d3 = self.d3(torch.cat([d2, e5], 1))    # [B, nf*8,  16,  16]
        d4 = self.d4(torch.cat([d3, e4], 1))    # [B, nf*8,  32,  32]
        d5 = self.d5(torch.cat([d4, e3], 1))    # [B, nf*4,  64,  64]
        d6 = self.d6(torch.cat([d5, e2], 1))    # [B, nf*2, 128, 128]
        d7 = self.d7(torch.cat([d6, e1], 1))    # [B, nf,   256, 256]
        d8 = self.d8(torch.cat([d7, e0], 1))    # [B, 1,    512, 512]
        return d8


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN Discriminator

    入力 : smooth と detail を cat した [B, 2, H, W]
    出力 : [B, 1, 30, 30] パッチごとの真偽スコア
    """
    def __init__(self, in_ch: int = 2, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # 最初のレイヤーは norm なし
            nn.Conv2d(in_ch, ndf,   4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf,   ndf*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # stride=1 でパッチサイズを調整
            nn.Conv2d(ndf*8, ndf*8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 1),  # パッチマップ出力
        )

    def forward(self, smooth, detail):
        return self.net(torch.cat([smooth, detail], dim=1))
