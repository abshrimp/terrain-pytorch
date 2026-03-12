# train.py
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from dataset import TerrainOutpaintingDataset
from model import TerrainUNet
from scheduler import DDPMScheduler

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # データ読み込み
    data_dir = "./data"
    npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not npy_files:
        print("エラー: './data' に .npy ファイルがありません。")
        return

    dataset = TerrainOutpaintingDataset(npy_files)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # モデル・スケジューラーの初期化
    model = TerrainUNet(n_channels=3, n_classes=1).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = DDPMScheduler(device=device)
    mse_loss = nn.MSELoss()

    num_epochs = 100  # 全体の目標エポック数
    start_epoch = 0
    checkpoint_path = "terrain_checkpoint.pth"

    # ==========================================
    # 途中再開（レジューム）処理
    # ==========================================
    if os.path.exists(checkpoint_path):
        print(f"\nチェックポイント '{checkpoint_path}' を発見しました。読み込み中...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # モデルの重み、オプティマイザの状態、エポック数を復元
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"✅ エポック {start_epoch + 1} から学習を再開します！")
    else:
        print("\n新規に学習を開始します。")

    # ==========================================
    # 学習ループ
    # ==========================================
    print("\n--- 学習開始 ---")
    model.train()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for input_data, target_terrain in pbar:
            target_terrain = target_terrain.to(device)
            context_and_mask = input_data.to(device) 
            b_size = target_terrain.shape[0]
            
            noise = torch.randn_like(target_terrain).to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (b_size,), device=device).long()
            noisy_terrain = scheduler.add_noise(target_terrain, noise, timesteps)
            
            model_input = torch.cat([noisy_terrain, context_and_mask], dim=1)
            noise_pred = model(model_input)
            
            loss = mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
        # ==========================================
        # エポック終了ごとの途中経過保存（チェックポイント）
        # ==========================================
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"💾 エポック {epoch+1} の途中経過を保存しました。")
        
        # ※ 推論用スクリプトのために、モデル単体の重みも別途保存しておくと便利です
        torch.save(model.state_dict(), "terrain_unet.pth")

if __name__ == "__main__":
    main()