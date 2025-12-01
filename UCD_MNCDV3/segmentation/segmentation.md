```bash
accelerate launch train_seg.py --batch-size 16 --learning-rate 1e-4 --epochs 50 --num-classes-seg 2 --img-size 160 --num-channels 10 --dataset-root /home/weikang/datasets/MineNetCDV2_Cropped160_Step160 --save-suffix UperNet_ConvNext --model UperNet_ConvNext
```

```bash
accelerate launch train_seg.py --batch-size 32 --learning-rate 1e-4 --epochs 20 --num-classes-seg 2 --img-size 160 --num-channels 10 --dataset-root /home/yu34/dataset/MNCDV2/MineNetCDV2_Cropped160_Step160 --save-suffix UperNet_ConvNext --model UperNet_ConvNext
```

```bash
accelerate launch train_seg.py --batch-size 32 --learning-rate 1e-4 --epochs 20 --num-classes-seg 2 --img-size 160 --num-channels 10 --dataset-root /bigdata/3dabc/dataset/MineNetCDV2_Cropped160_Step160 --save-suffix UperNet_ConvNext --model UperNet_ConvNext
```