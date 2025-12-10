# from modeling_mask2former import Mask2FormerForUniversalSegmentation
from transformers import AutoConfig
import torch
import numpy as np
# from modeling_mask2former import Mask2FormerForUniversalSegmentation
from torch.utils import data
from torch import nn
import os

import torchvision.transforms as tfs

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs
import argparse

from dataset_MNCDV3 import MNCDV3_Dataset, MNCDV3_WSSCD_Dataset
# import evaluate
from functools import partial

from transformers import ResNetBackbone, ResNetConfig

from torchmetrics import F1Score
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from codecarbon import track_emissions

from ws_maskcd import WS_MaskCD
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

def collate_fn(batch):
    # print(batch[0]["images"].shape)

    processor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    images = [b["images"]["pre"] for b in batch]
    segmentation_maps = [b["labels"]["pre"] for b in batch]

    images_t2= [b["images"]["post"] for b in batch]
    weak_change_labels= [b["labels"]["image_label"] for b in batch]
    change_masks= [b["labels"]["change_mask"] for b in batch]

    batch = processor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors='pt',
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        input_data_format="channels_first"
    )
    # batch['orig_image'] = inputs[2]
    # batch['orig_mask'] = inputs[3]

    batch["pixel_values_t2"] = torch.stack(images_t2, dim=0)

    # print(weak_change_labels)
    batch["weak_change_labels"] = torch.tensor(weak_change_labels)
    batch["change_masks"] = torch.stack(change_masks, dim=0)
    return batch

# @track_emissions(allow_multiple_runs=True)
def main(args):
    if args.save_suffix is None:
        args.save_suffix=args.model
    batch_size=args.batch_size
    logger = get_logger(__name__)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device=accelerator.device
    # batch_size=16

    train_dataset=MNCDV3_WSSCD_Dataset(dataset_root=args.dataset_root, mode="train", filter_empty=False, normalization=True)
    val_dataset=MNCDV3_WSSCD_Dataset(dataset_root=args.dataset_root, mode="val", normalization=True)
    test_dataset=MNCDV3_WSSCD_Dataset(dataset_root=args.dataset_root, mode="test", normalization=True)
    collate_func = None

    # config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-base-ade-semantic", num_labels=args.num_classes_seg, ignore_mismatched_sizes=True)
    # config.backbone_config.num_channels = args.num_channels
    # config.backbone_config.image_size = args.img_size
    model=WS_MaskCD.from_pretrained("ericyu/Mask2Former_MNCDV3")
    model_type="huggingface"
    collate_func=partial(collate_fn)


    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_func)
    val_dataloader = data.DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False, num_workers=8, collate_fn=None)
    # val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size//2, shuffle=False, num_workers=8, collate_fn=None)

    print("Training Samples", len(train_dataset), "Validation Samples", len(val_dataset), "batch size:", batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-7, last_epoch=-1)

    model.to(device)
    
    model, optimizer, train_dataloader, scheduler=accelerator.prepare(model, optimizer, train_dataloader, scheduler)

    Best_Seg_F1, Best_Seg_Epoch =  0, 0
    for epoch in range(args.epochs):
        logger.info(f'Epoch:{epoch}',main_process_only=True)
        model.train()
        running_loss_seg = 0.0
        running_loss_cd = 0.0
        num_samples = 0
        
        for idx, batch in enumerate(tqdm(train_dataloader,disable=not accelerator.is_local_main_process, miniters=50)):
            # break
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            

            pixel_values = batch["pixel_values"].to(accelerator.device)
            pixel_values_t2 = batch["pixel_values_t2"].to(accelerator.device)

            pixel_values = torch.cat([pixel_values, pixel_values_t2], dim=0)

            mask_labels = [b.to(accelerator.device) for b in batch["mask_labels"]]
            class_labels = [b.to(accelerator.device) for b in batch["class_labels"]]
            weak_change_labels = [b.to(accelerator.device) for b in batch["weak_change_labels"]]
            outputs = model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            weak_change_labels=weak_change_labels,
            )
            # print(batch.keys())
            # outputs = model(**batch)

            loss_seg=outputs["outputs_seg"].loss

            weak_loss_cd=outputs["outputs_cd"].loss

            loss= 0.3* loss_seg + weak_loss_cd

                # logits=outputs.logits
                # loss=loss_fct(logits, labels.long())

            accelerator.backward(loss)

            running_loss_seg += loss_seg.item() * 0.3
            running_loss_cd += weak_loss_cd.item()

            num_samples += 1

            # Optimization
            optimizer.step()
            scheduler.step()
            if (num_samples+1)%50==0:
                accelerator.print("Epoch:", epoch, "Progress", f'{num_samples}/{len(train_dataloader)}', "Seg_Loss:", running_loss_seg/num_samples, "CD_Loss:", running_loss_cd/num_samples)

        accelerator.print("Epoch:", epoch, "Seg_Loss:", running_loss_seg/num_samples, "CD_Loss:", running_loss_cd/num_samples)

        if (epoch+1) % 1 ==0:

            model.eval()
            val_dataloader=accelerator.prepare(val_dataloader)

            b_f1=F1Score(task='multiclass', num_classes=2, average=None).to(accelerator.device)

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(val_dataloader,disable=not accelerator.is_local_main_process, miniters=50)):
                # Reset the parameter gradients

                # Forward pass

                    images, labels=batch["images"]["pre"], batch["labels"]["change_mask"]

                    outputs_cd = model(images)["outputs_cd"]

                    # print(outputs)

                    processor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
                    outputs_cd = processor.post_process_semantic_segmentation(outputs_cd, target_sizes=[(args.img_size,args.img_size) for i in range(batch_size//2)])
                    outputs_cd = torch.stack(outputs_cd, dim=0)

                    probs_calc, label_calc=outputs.permute(1,2,0).flatten(0), labels.permute(1,2,0).flatten(0)
                    b_f1.update(probs_calc, label_calc)

                accelerator.print("All evaluation batches have been collected, calculating f1 scores.")
                seg_f1=b_f1.compute()
                avg_seg_f1=sum(seg_f1)/len(seg_f1)
                # seg_f1_positive=seg_f1[1]
            # print(seg_f1, Best_Seg_F1)
            if avg_seg_f1>Best_Seg_F1:
                Best_Seg_Epoch=epoch
                Best_Seg_F1=avg_seg_f1

                save_pretrained_path=f"./exp/{args.save_suffix}/BestF1"
                os.makedirs(save_pretrained_path,exist_ok=True)
                if model_type=="huggingface":
                    accelerator.unwrap_model(model).save_pretrained(save_pretrained_path)
                elif model_type=="standalone":
                    if accelerator.is_local_main_process:
                        torch.save(accelerator.unwrap_model(model).state_dict(), save_pretrained_path+"/pytorch_model.bin")

            # print(f"Evaluation for Epoch {epoch} Completed, Seg_F1: {Seg_F1}, Seg_Acc: {Seg_Acc}, Seg_Pre: {Seg_Pre}, Seg_Rec: {Seg_Rec}, CD_F1: {CD_F1}, CD_Acc: {CD_Acc}, CD_Pre: {CD_Pre}, CD_Rec: {CD_Rec}")
            accelerator.print(f"Evaluation for Epoch {epoch} Completed, Seg_F1: {seg_f1}, Averaged_Current_Seg_F1:", {sum(seg_f1)/len(seg_f1)})
 
            accelerator.print(f'Current Best Seg F1 Score: {Best_Seg_F1}')

            save_pretrained_path=f"./exp/{args.save_suffix}/{epoch}"
            os.makedirs(save_pretrained_path,exist_ok=True)

            accelerator.unwrap_model(model).save_pretrained(save_pretrained_path)
    accelerator.print(f"Training Completed, the best epoch is achieved in {Best_Seg_Epoch}.")

    # if args.push_to_hub:
    #     push_to_hub_path=f"MNCDV2_Prompted_Pretrained"
    #     accelerator.unwrap_model(model).push_to_hub(push_to_hub_path)

def args():
    parser = argparse.ArgumentParser(description='MineNetCD Training Arguments')

    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--push-to-hub', type=bool, default=False, help='whether pushing trained models to your huggingface repo, you need to login before using this feature.')
    parser.add_argument('--dataset-root', type=str, default="/home/eric/MNCDV3-MODEL/MNCDV3_Bitemporal_Cropped_Size224_Step112")
    parser.add_argument('--save-suffix', type=str, default="Mask2Former_WSSCD", help='suffix for saving the model')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=args()
    main(args)

