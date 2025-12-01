from transformers import UperNetForSemanticSegmentation, UperNetConfig, Mask2FormerConfig, Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, MobileViTForSemanticSegmentation, MobileViTConfig, MobileNetV2ForSemanticSegmentation, MobileNetV2Config, MobileViTV2ForSemanticSegmentation, MobileViTV2Config, SegformerConfig, SegformerForSemanticSegmentation, OneFormerConfig, OneFormerForUniversalSegmentation, OneFormerImageProcessor

from dataset import Multitemporal_Semantic_Segmentation_Dataset
from torch.utils.data import DataLoader
from functools import partial

# config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-base-ade-semantic", num_labels=2, ignore_mismatched_sizes=True)
# config.backbone_config.num_channels = 10
# config.backbone_config.image_size = 160

# print(config)

# model=Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-ade-semantic", config=config, ignore_mismatched_sizes=True)
# preprocessor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

# def collate_fn(batch, image_processor):
#     images = [b["images"] for b in batch]
#     segmentation_maps = [b["labels"] for b in batch]
#     batch = image_processor(
#         images,
#         segmentation_maps=segmentation_maps,
#         return_tensors='pt',
#         do_resize=False,
#         do_rescale=False,
#         do_normalize=False,
#         input_data_format="channels_first"
#     )
#     # batch['orig_image'] = inputs[2]
#     # batch['orig_mask'] = inputs[3]
#     return batch

# import torch

# image=torch.randn(10, 160, 160)  # Example input image
# label=torch.ones(160, 160)  # Example label

# batch=preprocessor(image,segmentation_maps=label,return_tensors='pt',input_data_format="channels_first")
# dataset=Multitemporal_Semantic_Segmentation_Dataset("/home/weikang/datasets/MineNetCDV2_Cropped160_Step160", mode="train", year_range=range(2015,2025,1))

# collate_func = partial(collate_fn, image_processor=preprocessor)
# dataloader_train=DataLoader(dataset, batch_size=8,num_workers=16,shuffle=True, collate_fn=collate_func)

# batch_0= next(iter(dataloader_train))
# # print(batch)
# # outputs = model(pixel_values=image, labels=label.long())

# outputs = model(**batch_0)
# print(outputs.class_queries_logits.shape)
# predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(160,160) for i in range(8)])

# print(predicted_segmentation_maps[0].shape)  # Should print the shape of the logits, e.g., (8, 2, 160, 160)

#ONEFORMER example
config = OneFormerConfig.from_pretrained("shi-labs/oneformer_ade20k_swin_large", num_labels=2, ignore_mismatched_sizes=True)
config.backbone_config.num_channels = 10
config.backbone_config.image_size = 160

print(config)

model=OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large", config=config, ignore_mismatched_sizes=True)
preprocessor=OneFormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False, class_info_file="/home/weikang/codes/UCD_MNCD/segmentation/cd.json")  # Path to the class info file)

def collate_fn(batch, image_processor):
    images = [b["images"] for b in batch]
    segmentation_maps = [b["labels"] for b in batch]
    batch = image_processor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors='pt',
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        input_data_format="channels_first",
        task_inputs=["semantic"],
    )
    # batch['orig_image'] = inputs[2]
    # batch['orig_mask'] = inputs[3]
    return batch

import torch

image=torch.randn(10, 160, 160)  # Example input image
label=torch.ones(160, 160)  # Example label

# batch=preprocessor(image,segmentation_maps=label,return_tensors='pt',input_data_format="channels_first")
dataset=Multitemporal_Semantic_Segmentation_Dataset("/home/weikang/datasets/MineNetCDV2_Cropped160_Step160", mode="train", year_range=range(2015,2025,1))

collate_func = partial(collate_fn, image_processor=preprocessor)
dataloader_train=DataLoader(dataset, batch_size=8,num_workers=16,shuffle=True, collate_fn=collate_func)

batch_0= next(iter(dataloader_train))
# print(batch)
# outputs = model(pixel_values=image, labels=label.long())

outputs = model(**batch_0)
print(outputs.class_queries_logits.shape)
predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(160,160) for i in range(8)])

print(predicted_segmentation_maps[0].shape)  # Should print the shape of the logits, e.g., (8, 2, 160, 160)

# 
# config = OneFormerConfig.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", num_labels=2, ignore_mismatched_sizes=True)
# config.num_channels = 10
# config.image_size = 160

# print(config)

# model=OneFormerForUniversalSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", config=config, ignore_mismatched_sizes=True)

# import torch

# image=torch.randn(8, 10, 160, 160)  # Example input image
# label=torch.ones(8, 160, 160)  # Example label

# outputs = model(pixel_values=image, labels=label.long())

# print(outputs.logits.shape)  # Should print the shape of the logits, e.g., (8, 2, 160, 160)