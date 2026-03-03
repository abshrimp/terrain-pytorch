#!/usr/bin/env python3
"""
平滑化された地形画像から詳細な地形を生成するスクリプト。
GAN Inversion により潜在空間 W を最適化して地形の詳細を復元します。
MPS (Apple Silicon) / CUDA / CPU に対応。

使い方:
    python project_terrain.py --network=snap.pkl --target=smooth.png --outdir=out

入力画像:
    - 16bit グレースケール PNG (高さマップ)
    - 8bit グレースケール PNG
    - 8bit RGB PNG (3ch 平均を高さマップとして扱う)
"""

import copy
import os
from time import perf_counter
from typing import Optional

import click
import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import dnnlib
import legacy

# ----------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------
def save_image(img_tensor, path, save_heightmap_color=True, save_hillshade=True):
    """
    img_tensor: (1,C,H,W) torch.Tensor [-1,1]
    path: 保存パス (.png)
    """
    img_tensor = img_tensor.detach().to(torch.float32).cpu()
    N, C, H, W = img_tensor.shape

    # --- 元画像 16bit RGB PNG 保存 ---
    img_np = img_tensor[0].permute(1, 2, 0).numpy()
    img_16 = ((img_np * 0.5 + 0.5) * 65535).clip(0, 65535).astype(np.uint16)
    if C >= 3:
        img_bgr = img_16[..., [2, 1, 0]]
        cv2.imwrite(path, img_bgr)
    else:
        cv2.imwrite(path, img_16[..., 0])

    # --- 高さマップ ---
    if save_heightmap_color or save_hillshade:
        if C >= 3:
            heightmap = img_tensor[0].mean(dim=0, keepdim=True)
        else:
            heightmap = img_tensor[0:1]
        real_elevation = ((heightmap[0].squeeze().numpy() + 1.0) / 2.0) * 1600.0

    if save_heightmap_color:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        ax.imshow(real_elevation, cmap="terrain", vmin=0, vmax=1600)
        plt.tight_layout(pad=0)
        plt.savefig(path.replace(".png", "_height_color.png"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    if save_hillshade:
        dx, dy = np.gradient(real_elevation)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dy, dx)
        azimuth = 315.0 * np.pi / 180.0
        altitude = 45.0 * np.pi / 180.0
        shade = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
        shade = np.clip(shade, 0, 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        ax.imshow(real_elevation, cmap="terrain", vmin=0, vmax=1600)
        ax.imshow(shade, cmap="gray", alpha=0.4)
        plt.tight_layout(pad=0)
        plt.savefig(path.replace(".png", "_hillshade.png"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)


# ----------------------------------------------------------------
def load_target_image(target_fname: str, resolution: int, img_channels: int) -> torch.Tensor:
    """
    平滑化画像を読み込み、GAN の入力形式に変換する。
    戻り値: [C, H, W] float32 Tensor, 範囲 [0, 255]
    """
    img = cv2.imread(target_fname, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {target_fname}")

    # チャンネル数・ビット深度を判定
    if img.ndim == 2:
        # グレースケール (8bit or 16bit)
        if img.dtype == np.uint16:
            img_8 = (img.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
        else:
            img_8 = img.astype(np.uint8)
        # 3ch に複製 (VGG / 3ch モデル対応)
        img_rgb = np.stack([img_8, img_8, img_8], axis=2)
    else:
        # BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.dtype == np.uint16:
            img_rgb = (img_rgb.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

    # リサイズ
    pil = PIL.Image.fromarray(img_rgb)
    w, h = pil.size
    s = min(w, h)
    pil = pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    pil = pil.resize((resolution, resolution), PIL.Image.LANCZOS)

    arr = np.array(pil, dtype=np.uint8)  # [H, W, 3]

    # チャンネルを GAN に合わせる
    if img_channels == 1:
        arr = arr.mean(axis=2, keepdims=True).astype(np.uint8)  # [H, W, 1]

    return torch.tensor(arr.transpose([2, 0, 1]), dtype=torch.float32)  # [C, H, W]


# ----------------------------------------------------------------
def multiscale_l2_loss(synth: torch.Tensor, target: torch.Tensor, scales: int = 4) -> torch.Tensor:
    """
    マルチスケール L2 損失。
    synth, target: [N, C, H, W], 範囲 [0, 255]
    """
    loss = torch.tensor(0.0, device=synth.device)
    for i in range(scales):
        if i > 0:
            synth = F.avg_pool2d(synth, kernel_size=2)
            target = F.avg_pool2d(target, kernel_size=2)
        loss = loss + (synth - target).pow(2).mean()
    return loss


# ----------------------------------------------------------------
def project(
    G,
    target: torch.Tensor,          # [C, H, W], [0,255]
    *,
    num_steps: int = 1000,
    w_avg_samples: int = 10000,
    initial_learning_rate: float = 0.05,
    initial_noise_factor: float = 0.05,
    lr_rampdown_length: float = 0.25,
    lr_rampup_length: float = 0.05,
    noise_ramp_length: float = 0.75,
    regularize_noise_weight: float = 1e5,
    use_lpips: bool = False,
    verbose: bool = True,
    device: torch.device,
) -> torch.Tensor:
    """
    target 画像に最も近い W 潜在ベクトルを最適化して返す。
    戻り値: [num_steps, num_ws, C] の W 列
    """
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution), \
        f"target shape mismatch: {target.shape} vs expected ({G.img_channels}, {G.img_resolution}, {G.img_resolution})"

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # W 空間の平均・標準偏差を計算
    logprint(f"W 空間の統計を計算中 ({w_avg_samples} サンプル)...")
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).float().to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)   # [1, 1, C]
    w_std = float((np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5)

    # ノイズバッファ
    noise_bufs = {name: buf for name, buf in G.synthesis.named_buffers() if "noise_const" in name}

    # VGG16 (LPIPS) の準備 - オプション
    vgg16 = None
    if use_lpips:
        logprint("VGG16 をロード中...")
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        try:
            # VGG は CPU で実行 (MPS 互換性のため)
            with dnnlib.util.open_url(url) as f:
                vgg16 = torch.jit.load(f).eval().to("cpu")
            logprint("VGG16 ロード完了")
        except Exception as e:
            logprint(f"VGG16 のロードに失敗しました: {e}")
            logprint("→ マルチスケール L2 損失にフォールバックします")
            vgg16 = None

    # ターゲット特徴量
    target_images = target.unsqueeze(0).to(device).float()  # [1, C, H, W], [0,255]

    if vgg16 is not None:
        # VGG 用に 3ch・CPU へ移動
        t_vgg = target_images.to("cpu")
        if t_vgg.shape[1] == 1:
            t_vgg = t_vgg.repeat(1, 3, 1, 1)
        if t_vgg.shape[2] > 256:
            t_vgg = F.interpolate(t_vgg, size=(256, 256), mode="area")
        target_features = vgg16(t_vgg, resize_images=False, return_lpips=True)
    else:
        target_features = None

    # W の初期値
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # ノイズ初期化
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    logprint(f"最適化開始 ({num_steps} ステップ)...")
    for step in range(num_steps):
        # 学習率スケジュール
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 生成
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth = G.synthesis(ws, noise_mode="const")  # [-1, 1]
        synth_255 = (synth + 1) * (255.0 / 2)        # [0, 255]

        # 損失: LPIPS + マルチスケール L2
        if vgg16 is not None and target_features is not None:
            s_vgg = synth_255.to("cpu")
            if s_vgg.shape[1] == 1:
                s_vgg = s_vgg.repeat(1, 3, 1, 1)
            if s_vgg.shape[2] > 256:
                s_vgg = F.interpolate(s_vgg, size=(256, 256), mode="area")
            synth_features = vgg16(s_vgg, resize_images=False, return_lpips=True)
            lpips_loss = (target_features - synth_features).square().sum()
            pixel_loss = multiscale_l2_loss(synth_255, target_images)
            dist = lpips_loss + pixel_loss * 0.1
        else:
            # マルチスケール L2 のみ
            dist = multiscale_l2_loss(synth_255, target_images)

        # ノイズ正則化
        reg_loss = torch.tensor(0.0, device=device)
        for v in noise_bufs.values():
            noise = v[None, None, :, :]
            while True:
                reg_loss = reg_loss + (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss = reg_loss + (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        loss = dist + reg_loss * regularize_noise_weight

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if verbose and (step % 100 == 0 or step == num_steps - 1):
            logprint(f"  step {step+1:>5d}/{num_steps}  dist={float(dist):.4f}  loss={float(loss):.4f}")

        w_out[step] = w_opt.detach()[0]

        # ノイズ正規化
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])


# ----------------------------------------------------------------
@click.command()
@click.option("--network",   "network_pkl", required=True, help="学習済みネットワーク pickle ファイル")
@click.option("--target",    "target_fname", required=True, metavar="FILE", help="平滑化された地形画像 (16bit グレースケール対応)")
@click.option("--outdir",    required=True, metavar="DIR", help="出力ディレクトリ")
@click.option("--num-steps", default=1000, show_default=True, type=int, help="最適化ステップ数")
@click.option("--seed",      default=303,  show_default=True, type=int, help="乱数シード")
@click.option("--use-lpips", is_flag=True, default=False, help="VGG16 LPIPS 損失を使用 (要インターネット接続)")
@click.option("--save-progress", is_flag=True, default=False, help="途中経過を保存")
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    num_steps: int,
    seed: int,
    use_lpips: bool,
    save_progress: bool,
):
    """
    平滑化された地形画像から GAN Inversion で詳細な地形を生成します。

    例:
        python project_terrain.py --network=snap.pkl --target=smooth.png --outdir=out

        # 16bit グレースケール高さマップを入力
        python project_terrain.py --network=snap.pkl --target=heightmap_smooth.png --outdir=out

        # VGG16 損失を使用 (精度向上、要インターネット)
        python project_terrain.py --network=snap.pkl --target=smooth.png --outdir=out --use-lpips
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    print(f"使用デバイス: {device}")

    # ネットワーク読み込み
    print(f'ネットワークを読み込み中: "{network_pkl}"')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].requires_grad_(False).to(device)
    G.eval()
    print(f"  解像度: {G.img_resolution}, チャンネル数: {G.img_channels}")

    # 平滑化画像の読み込み
    print(f'ターゲット画像を読み込み中: "{target_fname}"')
    target = load_target_image(target_fname, G.img_resolution, G.img_channels).to(device)
    print(f"  画像サイズ: {target.shape}")

    os.makedirs(outdir, exist_ok=True)

    # ターゲット画像を参照用に保存
    t_vis = target.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    PIL.Image.fromarray(t_vis if t_vis.shape[2] == 3 else t_vis[:, :, 0]).save(
        f"{outdir}/target_smooth.png"
    )

    # 最適化
    start_time = perf_counter()
    w_steps = project(
        G,
        target=target,
        num_steps=num_steps,
        device=device,
        use_lpips=use_lpips,
        verbose=True,
    )
    print(f"最適化完了: {perf_counter() - start_time:.1f} 秒")

    # 最終結果を保存
    final_w = w_steps[-1].unsqueeze(0)
    final_img = G.synthesis(final_w, noise_mode="const")
    save_image(final_img, f"{outdir}/projected_terrain.png")
    print(f"最終画像を保存: {outdir}/projected_terrain.png")

    # W ベクトルを保存
    np.savez(f"{outdir}/projected_w.npz", w=final_w.cpu().numpy())
    print(f"W ベクトルを保存: {outdir}/projected_w.npz")

    # 途中経過を保存
    if save_progress:
        progress_dir = f"{outdir}/progress"
        os.makedirs(progress_dir, exist_ok=True)
        save_every = max(1, num_steps // 20)
        print(f"途中経過を保存中 (20 枚)...")
        for i, w in enumerate(w_steps):
            if i % save_every == 0 or i == len(w_steps) - 1:
                img = G.synthesis(w.unsqueeze(0), noise_mode="const")
                save_image(
                    img,
                    f"{progress_dir}/step{i:05d}.png",
                    save_heightmap_color=False,
                    save_hillshade=True,
                )

    print("完了！")


# ----------------------------------------------------------------
if __name__ == "__main__":
    run_projection()
