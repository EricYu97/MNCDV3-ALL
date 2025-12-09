## Pretrain T1
```bash
accelerate launch train_seg.py --batch-size 32 --epochs 50 --num-classes-seg 6 --img-size 224 --num-channels 9 --dataset-root /home/weikang/MineNetCDV3/MineNetCDV3_Bitemporal_Cropped_Size224_Step112 --save-suffix Segformer_T1 --model Segformer
```