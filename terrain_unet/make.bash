python make_smooth_terrain.py --input=../images/gen_mps/seed10000.png --sigma=30 --outdir=../images/smooth
python generate.py --checkpoint=checkpoints/latest0.pt --input=../images/smooth/smooth_terrain.png --outdir=../images/generated
python generate.py --checkpoint=checkpoints/latest.pt --input=../images/generated/smooth_terrain_detail.png --outdir=../images/generated


python stylegan2-ada-pytorch/generate_for_mps.py --network=./snapshots/network-snapshot-003280.pkl --outdir=terrain_unet/images/gen_mps --trunc=0.7 --seeds=0
python terrain_unet/make_smooth_terrain.py --input=terrain_unet/images/gen_mps/seed0000.png --sigma=30 --outdir=terrain_unet/images/smooth
python terrain_unet/generate.py --checkpoint=terrain_unet/checkpoints/latest0.pt --input=terrain_unet/images/smooth/smooth_terrain.png --outdir=terrain_unet/images/generated
python terrain_unet/generate.py --checkpoint=terrain_unet/checkpoints/latest.pt --input=terrain_unet/images/generated/smooth_terrain_detail.png --outdir=terrain_unet/images/generated