# Training:

The code for training SSL can be found at [Solving Inefficiency of Self-supervised Representation Learning](https://github.com/wanggrun/triplet).

```shell
CUDA_VISIBLE_DEVICES=3,5,6,7   bash tools/dist_train.sh configs/selfsup/triplet/r50_bs4096_accumulate4_ep10_fp16_triplet_gpu3090_sysu30k.py    4  --pretrained   /scratch/local/ssd/guangrun/tmp/release_ep940.pth
```

# Testing:

```shell
python tools/extract_backbone_weights.py   work_dirs/selfsup/triplet/r50_bs4096_accumulate4_ep1000_fp16_triplet_gpu3090_backup/epoch_10.pth    work_dirs/selfsup/triplet/extract/sysu_ep10.pth
python test_sysu_combine.py  --gpu_ids 0  --name  debug   --test_dir   /scratch/local/ssd/guangrun/sysu_test_resize    --which_epoch 10  --batchsize 100
```
