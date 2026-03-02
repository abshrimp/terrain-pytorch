# Terrain Generation Project

StyleGAN2を使用して日本の地形を学習させ、地形生成を行うプロジェクトです。

[dandrino/terrain-erosion-3-ways](https://github.com/dandrino/terrain-erosion-3-ways.git) をめっちゃ参考にしています。

## セットアップ

### 必要なパッケージのインストール

```bash
pip install -r requirements.txt
```

### データについて

`raw_tiles/` ディレクトリには国土地理院のDEMから取得したデータが含まれています。

`download_raw_tiles.py` を使用してデータを取得できます：

```bash
python download_raw_tiles.py
```

### StyleGAN2 のクローン

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
    --snap=10 \
    --metrics=none
```

## ライセンス

詳細は [LICENSE.txt](LICENSE.txt) を参照してください。
