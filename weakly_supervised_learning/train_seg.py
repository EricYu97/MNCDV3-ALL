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
# from models.upernet import UperNetForSemanticSegmentation
# from models.models_debugger import Multitemporal_Semantic_Change_Detection_Model
from transformers import ViTConfig
import argparse
# from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy, multiclass_recall, multiclass_precision, binary_accuracy, binary_f1_score, binary_recall, binary_precision
# from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
# from torcheval.metrics.toolkit import sync_and_compute

from dataset_MNCDV3 import MNCDV3_Dataset, MNCDV3_WSSCD_Dataset
# import evaluate
from functools import partial

from transformers import ResNetBackbone, ResNetConfig

from torchmetrics import F1Score
from transformers import UperNetForSemanticSegmentation, UperNetConfig, MobileNetV2ForSemanticSegmentation, MobileNetV2Config, Mask2FormerForUniversalSegmentation, MobileViTV2ForSemanticSegmentation, MobileViTV2Config, ViTConfig, Mask2FormerConfig, Mask2FormerImageProcessor, MobileViTForSemanticSegmentation, MobileViTConfig, SegformerConfig, SegformerForSemanticSegmentation, OneFormerConfig, OneFormerForUniversalSegmentation
from standalone_models import unet, pspnet, linknet, icnet, sqnet, deeplabv3_plus
from codecarbon import track_emissions

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

def collate_fn(batch):
    # print(batch[0]["images"].shape)

    processor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    images = [b["images"] for b in batch]
    segmentation_maps = [b["labels"] for b in batch]


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

    model_type="standalone"
    if args.model=="UperNet_ConvNext":
        config = UperNetConfig.from_pretrained("openmmlab/upernet-convnext-base", num_labels=args.num_classes_seg, ignore_mismatched_sizes=True)
        config.backbone_config.num_channels = args.num_channels
        config.backbone_config.image_size = args.img_size
        model=UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-base", config=config, ignore_mismatched_sizes=True)
        model_type="huggingface"
    elif args.model=="UperNet_ViT":
        model=UperNet_Vit()
        model_type="standalone"
    elif args.model=="UperNet_SwinT":
        config = UperNetConfig.from_pretrained("openmmlab/upernet-swin-base", num_labels=args.num_classes_seg, ignore_mismatched_sizes=True)
        config.backbone_config.num_channels = args.num_channels
        config.backbone_config.image_size = args.img_size
        model=UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-base", config=config, ignore_mismatched_sizes=True)
        model_type="huggingface"
    elif args.model=="MobileNetV2_DeeplabV3P":
        config = MobileNetV2Config.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513", num_labels=args.num_classes_seg, ignore_mismatched_sizes=True)
        config.num_channels = args.num_channels
        config.image_size = args.img_size
        model=MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513", config=config, ignore_mismatched_sizes=True)
        model_type="huggingface"
    elif args.model=="MobileViTV2_DeeplabV3":
        config = MobileViTV2Config.from_pretrained("apple/mobilevitv2-1.0-voc-deeplabv3", num_labels=args.num_classes_seg, ignore_mismatched_sizes=True)
        config.num_channels = args.num_channels
        config.image_size = args.img_size
        model=MobileViTV2ForSemanticSegmentation.from_pretrained("apple/mobilevitv2-1.0-voc-deeplabv3", config=config, ignore_mismatched_sizes=True)
        model_type="huggingface"
    elif args.model=="Mask2Former":
        config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-base-ade-semantic", num_labels=args.num_classes_seg, ignore_mismatched_sizes=True)
        config.backbone_config.num_channels = args.num_channels
        config.backbone_config.image_size = args.img_size
        model=Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-ade-semantic", config=config, ignore_mismatched_sizes=True)
        model_type="huggingface"
        collate_func=partial(collate_fn)

    elif args.model=="Segformer":
        # Using Segformer for semantic segmentation, Scale is set to b5 as default
        config = SegformerConfig.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", num_labels=args.num_classes_seg, ignore_mismatched_sizes=True)
        config.num_channels = args.num_channels
        config.image_size = 160
        model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", config=config, ignore_mismatched_sizes=True)
        model_type="huggingface"
    elif args.model=="UNet":
        model=unet(n_channels=args.num_channels, n_classes=args.num_classes_seg)
        model_type="standalone"
    elif args.model=="pspnet":
        model=pspnet(n_channels=args.num_channels, n_classes=args.num_classes_seg, input_size=(args.img_size, args.img_size))
        model_type="standalone"
    elif args.model=="linknet":
        model=linknet(n_channels=args.num_channels, n_classes=args.num_classes_seg)
        model_type="standalone"
    elif args.model=="icnet":
        model=icnet(n_channels=args.num_channels, n_classes=args.num_classes_seg, input_size=(args.img_size, args.img_size))
        model_type="standalone"
    elif args.model=="sqnet":
        model=sqnet(n_channels=args.num_channels, n_classes=args.num_classes_seg)
        model_type="standalone"
    elif args.model=="deeplabv3_plus":
        model=deeplabv3_plus(nInputChannels=args.num_channels, n_classes=args.num_classes_seg)
        model_type="standalone"

    loss_fct=torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

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
            
            if not args.model=="Mask2Former":
                # For Mask2Former, we need to pass the images and labels separately
                images, labels=batch["images"]["pre"], batch["labels"]["pre"]

            # print(images, labels)
            # Backward propagation
            if model_type=="huggingface" or args.model=="UperNet_ViT":
                if not args.model=="Mask2Former":
                    outputs = model(images, labels=labels.long())
                else:
                    
                    pixel_values = batch["pixel_values"].to(accelerator.device)

                    mask_labels = [b.to(accelerator.device) for b in batch["mask_labels"]]
                    class_labels = [b.to(accelerator.device) for b in batch["class_labels"]]
                    outputs = model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                    )
                    # print(batch.keys())
                    # outputs = model(**batch)
                loss=outputs.loss

                # logits=outputs.logits
                # loss=loss_fct(logits, labels.long())

            elif model_type=="standalone":
                _, logits=model(images)
                loss=loss_fct(logits, labels.long())
            accelerator.backward(loss)

            running_loss_seg += loss.item()

            num_samples += 1

            # Optimization
            optimizer.step()
            scheduler.step()
            if (num_samples+1)%50==0:
                accelerator.print("Epoch:", epoch, "Progress", f'{num_samples}/{len(train_dataloader)}', "Seg_Loss:", running_loss_seg/num_samples)

        accelerator.print("Epoch:", epoch, "Seg_Loss:", running_loss_seg/num_samples)

        if (epoch+1) % 1 ==0:

            model.eval()
            val_dataloader=accelerator.prepare(val_dataloader)

            b_f1=F1Score(task='multiclass', num_classes=args.num_classes_seg, average=None).to(accelerator.device)

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(val_dataloader,disable=not accelerator.is_local_main_process, miniters=50)):
                # Reset the parameter gradients

                # Forward pass
                
                    images, labels=batch["images"], batch["labels"]

                    outputs = model(images)

                    # print(outputs)

                    if args.model=="Mask2Former":
                        processor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
                        outputs = processor.post_process_semantic_segmentation(outputs, target_sizes=[(args.img_size,args.img_size) for i in range(batch_size//2)])
                        outputs = torch.stack(outputs, dim=0)
                    else:
                        if model_type=="huggingface" or args.model=="UperNet_ViT":
                            outputs = outputs.logits
                        elif model_type=="standalone":
                            _, outputs = outputs

                        if outputs.shape[-2]!=args.img_size or outputs.shape[-1]!=args.img_size:
                            outputs = nn.functional.interpolate(outputs, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)

                    if args.model=="Mask2Former":
                        probs_calc, label_calc=outputs.permute(1,2,0).flatten(0), labels.permute(1,2,0).flatten(0)
                        b_f1.update(probs_calc, label_calc)
                    else:
                        probs_calc, label_calc=outputs.permute(1,2,3,0).flatten(1), labels.permute(1,2,0).flatten(0)
                        b_f1.update(torch.argmax(probs_calc, dim=0), label_calc)

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

            if model_type=="huggingface":
                accelerator.unwrap_model(model).save_pretrained(save_pretrained_path)
            elif model_type=="standalone":
                if accelerator.is_local_main_process:
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_pretrained_path+"/pytorch_model.bin")
    accelerator.print(f"Training Completed, the best epoch is achieved in {Best_Seg_Epoch}.")

    # if args.push_to_hub:
    #     push_to_hub_path=f"MNCDV2_Prompted_Pretrained"
    #     accelerator.unwrap_model(model).push_to_hub(push_to_hub_path)

def args():
    parser = argparse.ArgumentParser(description='MineNetCD Training Arguments')

    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--push-to-hub', type=bool, default=False, help='whether pushing trained models to your huggingface repo, you need to login before using this feature.')
    parser.add_argument('--num-classes-seg', type=int, default=2)
    parser.add_argument('--img-size', type=int, default=160, help='image size')
    parser.add_argument('--num-channels', type=int, default=10, help='number of channels')
    parser.add_argument('--dataset-root', type=str, default="/home/eric/dataset/MineNetCDV2_Cropped128_Step64")
    parser.add_argument('--save-suffix', type=str, default="Temporal_Mixed_Prompted_2_2_Focal_Diff", help='suffix for saving the model')

    parser.add_argument('--model', type=str, default="UperNet_ConvNext", choices=["UperNet_ViT","UperNet_ConvNext", "UperNet_SwinT", "MobileNetV2_DeeplabV3P", "MobileViTV2_DeeplabV3", "Mask2Former", "Segformer", "UNet", "pspnet", "linknet", "icnet", "sqnet", "deeplabv3_plus"], help='model type')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=args()
    main(args)

