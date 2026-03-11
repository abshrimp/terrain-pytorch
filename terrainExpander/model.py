#!/usr/bin/env python3
"""
TerrainExpander モデル

構造:
  3枚のコンテキストパッチ（左上・上・左）を個別にエンコード
    ↓
  Cross-Attention で文脈を統合
    ↓
  デコーダで次のパッチを生成

  ┌──────┬──────┐
  │左上  │上    │
  │Enc↓  │Enc↓  │
  └──────┴──────┘
  ┌──────┐
  │左    │     → [Cross-Attention] → Decoder → 生成パッチ
  │Enc↓  │
  └──────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- parts --

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # x を 1/2ch にアップサンプル → skip と concat → conv
        self.up   = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch//2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --------------------------------------------------------- patch encoder --

class PatchEncoder(nn.Module):
    """
    512×512 パッチ → 特徴マップ列 (各スケール)
    UNetのEncoder部分を共有して使う
    """
    def __init__(self, base_ch=32):
        super().__init__()
        self.inc  = DoubleConv(1, base_ch)        # 512  (16bit グレースケール入力)
        self.d1   = Down(base_ch,   base_ch*2)    # 256
        self.d2   = Down(base_ch*2, base_ch*4)    # 128
        self.d3   = Down(base_ch*4, base_ch*8)    # 64
        self.d4   = Down(base_ch*8, base_ch*16)   # 32
        self.bot  = Down(base_ch*16, base_ch*16)  # 16 (bottleneck)

    def forward(self, x):
        s0 = self.inc(x)   # [B, 32,  512, 512]
        s1 = self.d1(s0)   # [B, 64,  256, 256]
        s2 = self.d2(s1)   # [B, 128, 128, 128]
        s3 = self.d3(s2)   # [B, 256,  64,  64]
        s4 = self.d4(s3)   # [B, 512,  32,  32]
        bot = self.bot(s4) # [B, 512,  16,  16]
        return s0, s1, s2, s3, s4, bot


# ------------------------------------------------------- context fusion --

class ContextFusion(nn.Module):
    """
    3枚のbottleneckを concat + conv で統合。
    MultiheadAttention の代わりにシンプルな畳み込みを使う。

    入力: 3枚の [B, C, H, W]  → concat で [B, 3C, H, W]
    出力: [B, C, H, W]
    """
    def __init__(self, ch=512, heads=8):  # heads は互換性のため残す（未使用）
        super().__init__()
        # 3C → C に圧縮する 1×1 conv + 3×3 conv
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),   # チャネル圧縮
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        # 左パッチ（最も近い文脈）との残差接続用
        self.residual = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, bot_list):
        """
        bot_list: [topleft_bot, top_bot, left_bot]  各 [B, C, H, W]
        """
        # 3枚を channel 方向に concat → [B, 3C, H, W]
        x = torch.cat(bot_list, dim=1)
        out = self.fuse(x)
        # 左パッチ（bot_list[-1]）との残差接続
        out = out + self.residual(bot_list[-1])
        return out


# ---------------------------------------------------- terrain expander --

class TerrainExpander(nn.Module):
    """
    入力:  left_top (左上), top (上), left (左)  各 1ch 512×512
    出力:  next_patch  1ch 512×512
    """
    def __init__(self, base_ch=32):
        super().__init__()
        ch = base_ch

        # 共有エンコーダ（重みを共有してパラメータ削減）
        self.encoder = PatchEncoder(base_ch=ch)

        # コンテキスト統合
        self.fusion = ContextFusion(ch=ch*16, heads=8)

        # デコーダ（UNet スタイル）
        # Up(in_ch, skip_ch, out_ch)
        # in_ch  : このUpブロックへの入力チャンネル数
        # skip_ch: エンコーダからのskip connectionのチャンネル数
        # out_ch : 出力チャンネル数
        #
        # fused: [B, 512, 16, 16]
        # s4_l:  [B, 512, 32, 32]   s3_l: [B, 256, 64, 64]
        # s2_l:  [B, 128,128,128]   s1_l: [B,  64,256,256]
        # s0_l:  [B,  32,512,512]
        self.up1 = Up(ch*16, ch*16, ch*8)  # 512→256, +s4(512)→768→256  [32px]
        self.up2 = Up(ch*8,  ch*8,  ch*4)  # 256→128, +s3(256)→384→128  [64px]
        self.up3 = Up(ch*4,  ch*4,  ch*2)  # 128→ 64, +s2(128)→192→ 64  [128px]
        self.up4 = Up(ch*2,  ch*2,  ch)    #  64→ 32, +s1( 64)→ 96→ 32  [256px]
        self.up5 = Up(ch,    ch,    ch)    #  32→ 16, +s0( 32)→ 48→ 32  [512px]
        self.out = nn.Conv2d(ch, 1, 1)     # 16bit グレースケール出力
        self.act = nn.Sigmoid()

    def forward(self, topleft, top, left):
        """
        各入力: [B, 1, 512, 512]  値域 [0,1]
        出力:   [B, 1, 512, 512]  値域 [0,1]
        """
        # 3枚を個別にエンコード（重み共有）
        s0_tl, s1_tl, s2_tl, s3_tl, s4_tl, bot_tl = self.encoder(topleft)
        s0_t,  s1_t,  s2_t,  s3_t,  s4_t,  bot_t  = self.encoder(top)
        s0_l,  s1_l,  s2_l,  s3_l,  s4_l,  bot_l  = self.encoder(left)

        # bottleneck を Attention で統合
        fused = self.fusion([bot_tl, bot_t, bot_l])  # [B, C, 16, 16]

        # skip connection は「左パッチ」のものを使う
        # （左パッチが一番近い文脈）
        x = self.up1(fused,  s4_l)
        x = self.up2(x,      s3_l)
        x = self.up3(x,      s2_l)
        x = self.up4(x,      s1_l)
        x = self.up5(x,      s0_l)
        return self.act(self.out(x))   # [B, 1, 512, 512]


# ------------------------------------------------------- discriminator --

class PatchDiscriminator(nn.Module):
    """
    PatchGAN識別器: 本物/偽物を 32×32 の局所パッチで判定
    入力: [コンテキスト3枚 concat + 対象パッチ] = 4ch
    """
    def __init__(self, base_ch=64):
        super().__init__()
        def block(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride=stride, padding=1, bias=not norm)]
            if norm: layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 入力: コンテキスト3枚(1ch) + ターゲット(1ch) = 4ch
        self.net = nn.Sequential(
            *block(4, base_ch,    norm=False),
            *block(base_ch,   base_ch*2),
            *block(base_ch*2, base_ch*4),
            *block(base_ch*4, base_ch*8, stride=1),
            nn.Conv2d(base_ch*8, 1, 4, padding=1),
        )

    def forward(self, ctx_list, target):
        """
        ctx_list: [topleft, top, left] 各 [B,1,512,512]
        target:   [B,1,512,512]
        """
        # 3枚の平均をコンテキストとして結合
        ctx_mean = torch.stack(ctx_list, dim=1).mean(dim=1)  # [B,1,512,512]
        # ダウンサンプルして512→256に揃える
        ctx_ds  = F.avg_pool2d(ctx_mean, 2)
        tgt_ds  = F.avg_pool2d(target, 2)
        # 4ch入力
        x = torch.cat([
            F.avg_pool2d(ctx_list[0], 2),
            F.avg_pool2d(ctx_list[1], 2),
            F.avg_pool2d(ctx_list[2], 2),
            tgt_ds,
        ], dim=1)
        return self.net(x)


if __name__ == "__main__":
    # 動作確認
    model = TerrainExpander(base_ch=32)
    disc  = PatchDiscriminator(base_ch=64)
    B = 2
    x = [torch.rand(B,1,512,512) for _ in range(3)]
    out = model(*x)
    print(f"Generator output: {out.shape}")   # [2,1,512,512]
    d = disc(x, out)
    print(f"Discriminator output: {d.shape}") # [2,1,H,W]
    params_G = sum(p.numel() for p in model.parameters())/1e6
    params_D = sum(p.numel() for p in disc.parameters())/1e6
    print(f"Generator: {params_G:.1f}M params")
    print(f"Discriminator: {params_D:.1f}M params")