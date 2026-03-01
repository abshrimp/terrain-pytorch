# Terrain Erosion 3 Ways

このプロジェクトは [dandrino/terrain-erosion-3-ways](https://github.com/dandrino/terrain-erosion-3-ways.git) のクローンで、StyleGAN2を使用して地形生成を行うものです。

## データについて

`raw_tiles/` ディレクトリには国土地理院のDEMから取得したデータが含まれています。

## セットアップ

### 必要なパッケージのインストール

```bash
pip install -r requirements-pip3.txt
```

### StyleGAN2のクローン

このリポジトリには [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch.git) のクローンが含まれています。

## 使用方法

### 1. 訓練画像の生成

`generate_training_from_raw_tiles.py` を使用して訓練画像を生成できます：

```bash
python generate_training_from_raw_tiles.py
```

生成された訓練画像は `training_samples/` ディレクトリに保存されます。

### 2. データセットZIPの作成

StyleGAN2用のデータセットZIPを作成します：

```bash
python stylegan2-ada-pytorch/dataset_tool.py \
    --source=training_samples \
    --dest=datasets/terrain.zip
```

### 3. トレーニングの実行

StyleGAN2モデルをトレーニングします：

```bash
python stylegan2-ada-pytorch/train.py \
    --outdir=training-runs \
    --data=datasets/terrain.zip \
    --gpus=1 \
    --cfg=auto \
    --mirror=0 \
    --snap=10
```

#### トレーニングパラメータの説明

- `--outdir`: トレーニング結果の出力ディレクトリ
- `--data`: 訓練データセットのパス
- `--gpus`: 使用するGPUの数
- `--cfg`: 自動的に設定を選択（`auto`）
- `--mirror`: 水平反転の有効化（0=無効、1=有効）
- `--snap`: スナップショットの保存間隔（Kimg単位）

## ライセンス

詳細は [LICENSE.txt](LICENSE.txt) を参照してください。
