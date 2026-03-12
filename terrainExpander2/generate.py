# generate.py
import torch
from PIL import Image
from tqdm import tqdm
import glob
import os

from model import TerrainUNet
from scheduler import DDPMScheduler
from dataset import TerrainOutpaintingDataset

@torch.no_grad()
def generate_seamless_terrain(model, scheduler, context_tensor, mask_tensor, device="cuda"):
    model.eval()
    x_t = torch.randn(1, 1, 256, 256).to(device)
    context_tensor = context_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    context_and_mask = torch.cat([context_tensor, mask_tensor], dim=1)

    print("地形を推論中...")
    for i in tqdm(reversed(range(scheduler.num_train_timesteps)), total=scheduler.num_train_timesteps):
        t = torch.tensor([i], device=device)
        model_input = torch.cat([x_t, context_and_mask], dim=1)
        predicted_noise = model(model_input)

        alpha_t = scheduler.alphas[t]
        alpha_prod_t = scheduler.alphas_cumprod[t]
        beta_t = scheduler.betas[t]

        noise = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)
        x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_prod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise

        if i > 0:
            noisy_context = scheduler.add_noise(context_tensor, torch.randn_like(context_tensor), t)
            x_t = mask_tensor * noisy_context + (1.0 - mask_tensor) * x_t
        else:
            x_t = mask_tensor * context_tensor + (1.0 - mask_tensor) * x_t

    return torch.clamp(x_t, 0.0, 1.0)

def tensor_to_16bit_png(tensor, output_filename="generated_tile.png"):
    terrain_array = tensor.squeeze().cpu().numpy()
    terrain_16bit = (terrain_array * 65535.0).astype(np.uint16)
    img = Image.fromarray(terrain_16bit, mode='I;16')
    img.save(output_filename)
    print(f"✅ 生成完了！保存先: {output_filename}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # モデルのロード
    model = TerrainUNet(n_channels=3, n_classes=1).to(device)
    if os.path.exists("terrain_unet.pth"):
        model.load_state_dict(torch.load("terrain_unet.pth", map_location=device))
        print("学習済みモデルを読み込みました。")
    else:
        print("警告: 'terrain_unet.pth' が見つからないため、初期状態のモデルを使用します。")
        
    scheduler = DDPMScheduler(device=device)

    # 推論用のヒント（コンテキスト）を取得するために一時的にDatasetを使用
    npy_files = glob.glob(os.path.join("../raw_tiles", "*.npy"))
    if not npy_files:
        print("推論テスト用の .npy データが ../raw_tiles にありません。")
        return
        
    dataset = TerrainOutpaintingDataset(npy_files)
    test_input, _ = dataset[0]
    context_tensor = test_input[0:1].unsqueeze(0) # [1, 1, 256, 256]
    mask_tensor = test_input[1:2].unsqueeze(0)    # [1, 1, 256, 256]

    # 生成と保存
    generated_terrain = generate_seamless_terrain(model, scheduler, context_tensor, mask_tensor, device)
    tensor_to_16bit_png(generated_terrain, "output_terrain.png")

if __name__ == "__main__":
    main()