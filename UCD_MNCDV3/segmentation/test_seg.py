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
from dataset import Multitemporal_Semantic_Segmentation_Dataset 
# import evaluate
from functools import partial

from transformers import ResNetBackbone, ResNetConfig

from torchmetrics import F1Score
from transformers import UperNetForSemanticSegmentation, UperNetConfig, MobileNetV2ForSemanticSegmentation, MobileNetV2Config, Mask2FormerForUniversalSegmentation, MobileViTV2ForSemanticSegmentation, MobileViTV2Config, ViTConfig, Mask2FormerConfig, Mask2FormerImageProcessor, MobileViTForSemanticSegmentation, MobileViTConfig, SegformerConfig, SegformerForSemanticSegmentation, OneFormerConfig, OneFormerForUniversalSegmentation
from standalone_models import unet, pspnet, linknet, icnet, sqnet, deeplabv3_plus
from torchvision.utils import save_image
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

import torch

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def main(args):
    batch_size=args.batch_size
    logger = get_logger(__name__)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device=accelerator.device
    # batch_size=16

    train_dataset=Multitemporal_Semantic_Segmentation_Dataset(dataset_root=args.dataset_root, mode="train", year_range=range(2015,2025))
    val_dataset=Multitemporal_Semantic_Segmentation_Dataset(dataset_root=args.dataset_root, mode="val")
    test_dataset=Multitemporal_Semantic_Segmentation_Dataset(dataset_root=args.dataset_root, mode="test")
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False, num_workers=8, collate_fn=None)
    collate_func = None

    model_type="standalone"
    if args.model=="UperNet_ConvNext":
        model=UperNetForSemanticSegmentation.from_pretrained(args.model_path)
        model_type="huggingface"
    elif args.model=="UperNet_SwinT":
        model=UperNetForSemanticSegmentation.from_pretrained(args.model_path)
        model_type="huggingface"
    elif args.model=="MobileNetV2_DeeplabV3P":
        model=MobileNetV2ForSemanticSegmentation.from_pretrained(args.model_path)
        model_type="huggingface"
    elif args.model=="MobileViTV2_DeeplabV3":
        model=MobileViTV2ForSemanticSegmentation.from_pretrained(args.model_path)
        model_type="huggingface"
    elif args.model=="Mask2Former":
        model=Mask2FormerForUniversalSegmentation.from_pretrained(args.model_path)
        model_type="huggingface"

    elif args.model=="Segformer":
        # Using Segformer for semantic segmentation, Scale is set to b5 as default
        model=SegformerForSemanticSegmentation.from_pretrained(args.model_path)
        model_type="huggingface"
    elif args.model=="UNet":
        model=unet(n_channels=args.num_channels, n_classes=2)
        model_type="standalone"
    elif args.model=="pspnet":
        model=pspnet(n_channels=args.num_channels, n_classes=2, input_size=(args.img_size, args.img_size))
        model_type="standalone"
    elif args.model=="linknet":
        model=linknet(n_channels=args.num_channels, n_classes=2)
        model_type="standalone"
    elif args.model=="icnet":
        model=icnet(n_channels=args.num_channels, n_classes=2, input_size=(args.img_size, args.img_size))
        model_type="standalone"
    elif args.model=="sqnet":
        model=sqnet(n_channels=args.num_channels, n_classes=2)
        model_type="standalone"
    elif args.model=="deeplabv3_plus":
        model=deeplabv3_plus(nInputChannels=args.num_channels, n_classes=2)
        model_type="standalone"

    if model_type=="standalone":
        print(model)
        pretrained_weights = torch.load(args.model_path+"/pytorch_model.bin", map_location='cpu')
        model.load_state_dict(pretrained_weights)

    print("Testing", len(test_dataset), "batch size:", batch_size)
    
    model, test_dataloader=accelerator.prepare(model, test_dataloader)

    model.eval()
    # test_dataloader=accelerator.prepare(test_dataloader)

    b_f1=F1Score(task='multiclass', num_classes=args.num_classes_seg, average=None).to(accelerator.device)

    TP, TN, FP, FN = 0, 0, 0, 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader,disable=not accelerator.is_local_main_process, miniters=50)):
        # Reset the parameter gradients

        # Forward pass
        
            images, labels, domain, year, patch_num =batch["images"], batch["labels"], batch["domain"], batch["year"], batch["patch_num"]

            outputs = model(images)

            # print(outputs)

            if args.model=="Mask2Former":
                processor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
                outputs = processor.post_process_semantic_segmentation(outputs, target_sizes=[(args.img_size,args.img_size) for i in range(batch_size//2)])
                outputs = torch.stack(outputs, dim=0)
            else:
                if model_type=="huggingface":
                    outputs = outputs.logits
                elif model_type=="standalone":
                    _, outputs = outputs

                if outputs.shape[-2]!=args.img_size or outputs.shape[-1]!=args.img_size:
                    outputs = nn.functional.interpolate(outputs, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)

            if args.model=="Mask2Former":
                probs_calc, label_calc=outputs.squeeze(), labels
            else:
                probs_calc, label_calc=torch.argmax(outputs, dim=1), labels
            tp,fp,tn,fn=confusion(probs_calc,label_calc)
            assert tp+fp+tn+fn==probs_calc.shape.numel()
            TP+=tp
            TN+=tn
            FP+=fp
            FN+=fn

            for i in range(probs_calc.shape[0]):
                save_path = os.path.join(args.save_path, args.model, domain[i], str(year[i].item()))
                os.makedirs(save_path, exist_ok=True)
                save_image(probs_calc[i,:,:].float().cpu(), os.path.join(save_path, patch_num[i]+".png"))

    OA=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1=2*TP/(2*TP+FP+FN)
    ciou=TP/(TP+FP+FN)
    f1_each_device=f1
    ts_metrics_list=torch.FloatTensor([OA,f1,precision,recall,ciou]).cuda().unsqueeze(0)
    ts_eval_metric_gathered=accelerator.gather(ts_metrics_list)
    final_metric=torch.mean(ts_eval_metric_gathered, dim=0)
    accelerator.print(f'Accuracy={final_metric[0]:.04}, Precision={final_metric[2]:.04}, Recall={final_metric[3]:.04}, mF1={final_metric[1]:.04}, ciou={final_metric[4]:.04}')

    # if args.push_to_hub:
    #     push_to_hub_path=f"MNCDV2_Prompted_Pretrained"
    #     accelerator.unwrap_model(model).push_to_hub(push_to_hub_path)

def args():
    parser = argparse.ArgumentParser(description='MineNetCD Training Arguments')

    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--push-to-hub', type=bool, default=False, help='whether pushing trained models to your huggingface repo, you need to login before using this feature.')
    parser.add_argument('--img-size', type=int, default=160, help='image size')
    parser.add_argument('--num-channels', type=int, default=10, help='number of channels')
    parser.add_argument('--dataset-root', type=str, default="/home/eric/dataset/MineNetCDV2_Cropped128_Step64")
    parser.add_argument('--save-path', type=str, default="./results", help='path to save the results')
    parser.add_argument('--num-classes-seg', type=int, default=2)

    parser.add_argument('--model', type=str, default="UperNet_ConvNext", choices=["UperNet_ConvNext", "UperNet_SwinT", "MobileNetV2_DeeplabV3P", "MobileViTV2_DeeplabV3", "Mask2Former", "Segformer", "UNet", "pspnet", "linknet", "icnet", "sqnet", "deeplabv3_plus"], help='model type')
    parser.add_argument('--model-path', type=str, default="./pretrained_models/UperNet_ConvNext", help='path to the pretrained model')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=args()
    main(args)

