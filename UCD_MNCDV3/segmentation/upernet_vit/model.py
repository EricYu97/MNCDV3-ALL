import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss # type: ignore
from typing import Optional, Tuple, Union, List
from huggingface_hub import ModelHubMixin # type: ignore
from dataclasses import dataclass
from transformers.utils import ModelOutput # type: ignore

from transformers import SwinModel # type: ignore

from transformers.modeling_outputs import SemanticSegmenterOutput # type: ignore
from transformers.modeling_utils import PreTrainedModel # type: ignore
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings # type: ignore
#from transformers.utils.backbone_utils import load_backbone
from transformers import UperNetConfig # type: ignore

import torch.nn.functional as F # type: ignore
from .vit import TIMMVisionTransformer
from transformers import ViTConfig
from .hf_vit import ViTModel
# from ViT_MAE import ViT_MAE_for_FT, ViT_MAE_for_Pretraining

# General docstring
_CONFIG_FOR_DOC = "UperNetConfig"


class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output


class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            UperNetConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class UperNetPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (`Tuple[int]`):
            Pooling scales used in Pooling Pyramid Module.
        in_channels (`int`):
            Input channels.
        channels (`int`):
            Channels after modules, before conv_seg.
        align_corners (`bool`):
            align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = UperNetPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.add_module(str(i), block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UperNetHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).
    """

    def __init__(self, config, in_channels):
        super().__init__()

        self.config = config
        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = in_channels
        self.channels = config.hidden_size
        self.align_corners = False
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = UperNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = UperNetConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = UperNetConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = UperNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # build laterals
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class UperNetFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config:
            Configuration.
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self, config, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()

        self.config = config
        self.in_channels = config.auxiliary_in_channels
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            UperNetConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


class UperNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UperNetConfig
    main_input_name = "pixel_values"
    _no_split_modules = []

    def _init_weights(self, module):
        if isinstance(module, UperNetPreTrainedModel):
            module.backbone.init_weights()
            module.decode_head.init_weights()
            if module.auxiliary_head is not None:
                module.auxiliary_head.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        # self.backbone.init_weights()
        self.decode_head.init_weights()
        if self.auxiliary_head is not None:
            self.auxiliary_head.init_weights()


UPERNET_START_DOCSTRING = r"""
    Parameters:
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
        config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

UPERNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SegformerImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers in case the backbone has them. See
            `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers of the backbone. See `hidden_states` under
            returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """UperNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    UPERNET_START_DOCSTRING,
)
class UperNet(UperNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.img_size=config.img_size
        # Semantic segmentation head(s)
        self.decode_head = UperNetHead(config, in_channels=self.config.in_channels)
        self.auxiliary_head = UperNetFCNHead(config) if config.use_auxiliary_head else None
        self.vit_neck = MultiLevelNeck(in_channels=config.in_channels, out_channels=config.auxiliary_in_channels)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(UPERNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        backbone_feature_maps: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
        >>> model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits  # shape (batch_size, num_labels, height, width)
        >>> list(logits.shape)
        [1, 150, 512, 512]
        ```"""
        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        features=backbone_feature_maps

        # feature maps of larger size should be input first
        # features = backbone_feature_maps.feature_maps

        # print("input", features[0].shape, features[1].shape, features[2].shape, features[3].shape)
        
        # features = self.vit_neck(features)
        # print("necked", features[0].shape, features[1].shape, features[2].shape, features[3].shape)

        logits = self.decode_head(features)
        # print("decoded", logits.shape)
        logits = nn.functional.interpolate(logits, size=self.img_size, mode="bilinear", align_corners=False)
        # print(f'logits shape: {logits.shape}')
        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)
            auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=self.img_size, mode="bilinear", align_corners=False
            )
        
        loss = None
        if labels is not None:
            # compute weighted loss
            loss_fct = CrossEntropyLoss(ignore_index=self.config.loss_ignore_index)
            loss = loss_fct(logits, labels.long())
            if auxiliary_logits is not None:
                auxiliary_loss = loss_fct(auxiliary_logits, labels.long())
                loss += self.config.auxiliary_loss_weight * auxiliary_loss
        # print("upsampled", logits.shape)
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_feature_maps,
        )

class MultiLevelNeck(nn.Module):
    """MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=[4, 2, 1, 0.5]):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(
                    in_channel,
                    out_channels,
                    kernel_size=1))
        for _ in range(self.num_outs):
            self.convs.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # for len(inputs) not equal to self.num_outs
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]
        outs = []
        for i in range(self.num_outs):
            x_resize = F.interpolate(inputs[i], scale_factor=self.scales[i], mode='bilinear')
            outs.append(self.convs[i](x_resize))
        return tuple(outs)


@add_start_docstrings(
    """UperNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    UPERNET_START_DOCSTRING,
)
class UperNetCD(UperNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.img_size=config.img_size
        # Semantic segmentation head(s)
        self.fusion_strategy = config.fusion_strategy
        self.decode_head = UperNetHead(config, in_channels=self.config.in_channels)
        self.auxiliary_head = UperNetFCNHead(config) if config.use_auxiliary_head else None
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(UPERNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        backbone_outputs: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
        >>> model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits  # shape (batch_size, num_labels, height, width)
        >>> list(logits.shape)
        [1, 150, 512, 512]
        ```"""
        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        # if len(backbone_outputs) != 2:
        #     raise ValueError("The input for CD head should be a list contains two torch tensors.")

        # features_0, features_1 = backbone_outputs[0], backbone_outputs[1] # bug fix

        # Maybe can be replaced by Deform DETR module.
        """ if self.fusion_strategy == "diff":
            features = features_0 - features_1
        elif self.fusion_strategy == "concat":
            features = torch.cat([features_0, features_1], dim=1) """
        
        # bug fix
        # if self.fusion_strategy == "diff":
        #     features = tuple(t1 - t2 for t1, t2 in zip(features_0, features_1))
        # elif self.fusion_strategy == "concat":
        #     features = tuple(torch.cat([t1, t2], dim=1) for t1, t2 in zip(features_0, features_1))
        
        features=tuple([f for f in backbone_outputs])
        logits = self.decode_head(features)
        logits = nn.functional.interpolate(logits, size=self.img_size, mode="bilinear", align_corners=False)
        # print(f'logits shape: {logits.shape}')
        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)
            auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=self.img_size, mode="bilinear", align_corners=False
            )

        loss = None
        if labels is not None:
            # compute weighted loss
            loss_fct = CrossEntropyLoss(ignore_index=self.config.loss_ignore_index)
            loss = loss_fct(logits, labels.long())
            if auxiliary_logits is not None:
                auxiliary_loss = loss_fct(auxiliary_logits, labels.long())
                loss += self.config.auxiliary_loss_weight * auxiliary_loss

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            #hidden_states=backbone_outputs.hidden_states,
            hidden_states=features, # bug fix
        )

def SemanticSegmentationHead(num_labels=10, embed_dims=768, img_size=224):
    config=UperNetConfig(hidden_size=512, pool_scales=[1,2,3,6], auxiliary_channels=256, auxiliary_in_channels=embed_dims, auxiliary_num_convs=1, auxiliary_concat_input=False, num_labels = num_labels, auxiliary_loss_weight=0.4)
    config.update({"in_channels":[embed_dims,embed_dims,embed_dims,embed_dims], "img_size":img_size})
    head=UperNet(config=config)

    return head

def ChangeDetectionHead(num_labels=2, embed_dims=768, img_size=224, fusion_strategy="diff"):
    
    if fusion_strategy == "concat": # concat operation doubles the input channels for the head.
        embed_dims = embed_dims * 2
    config = UperNetConfig(hidden_size=512, pool_scales=[1,2,3,6], auxiliary_channels=256, auxiliary_in_channels=embed_dims, auxiliary_num_convs=1, auxiliary_concat_input=False, num_labels = num_labels, auxiliary_loss_weight=0.4)
    config.update({"in_channels":[embed_dims,embed_dims,embed_dims,embed_dims], "img_size":img_size, "fusion_strategy": fusion_strategy})
    head = UperNetCD(config=config)

    return head

class UperNet_Vit(nn.Module):
    def __init__(self):
        super().__init__()

        vit_config=ViTConfig(hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=9,
        qkv_bias=True,
        encoder_stride=16,
        pooler_output_size=None,
        pooler_act="tanh")

        self.backbone=ViTModel(vit_config, add_pooling_layer=False).to("cuda")

        # self.backbone=TIMMVisionTransformer(img_size=224, patch_size=16, in_chans=9, num_classes=1000,
        #          embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
        #          drop_rate=0., attn_drop_rate=0., drop_path_rate=0., layer_scale=True,
        #          act_layer=nn.GELU, window_attn=False, window_size=14, pretrained=None,
        #          with_cp=False)
        self.vit_neck= MultiLevelNeck(in_channels=[768, 768, 768, 768],out_channels=768,scales=[4, 2, 1, 0.5])
        self.decode_head = SemanticSegmentationHead(num_labels=6, embed_dims=768, img_size=224)

    def forward(self, images, labels=None) -> torch.Tensor:
        # print(images.shape)
        features= self.backbone(images).last_hidden_state
        # print(features[0].shape, features[1].shape, features[2].shape, features[3].shape)
        features = self.vit_neck(features)
        logits = self.decode_head(features, labels)
        # logits = nn.functional.interpolate(logits, size=224, mode="bilinear", align_corners=False)

        return logits

if __name__=="__main__":
    """ from fm_encoder import vit_base_224_patch16
    
    backbone=vit_base_224_patch16()
    imgs=torch.randn(8,3,224,224)

    config=UperNetConfig(hidden_size=512, pool_scales=[1,2,3,6], auxiliary_channels=256, auxiliary_in_channels=768, auxiliary_num_convs=1, auxiliary_concat_input=False, num_labels = 10, auxiliary_loss_weight=0.4)
    config.update({"in_channels":[768,768,768,768], "img_size":224})
    
    head=UperNet(config=config)

    wavelengths=[1,1,1]

    outputs=backbone(imgs, wavelengths)
    labels=torch.ones(8,224,224)
    outputs_seg=head(outputs, labels.long())

    print(outputs_seg.loss) """
    # from ViT_Configs import ViT_MAE_Base_Config
    # from transformers import ViTMAEConfig

    # config=ViTMAEConfig(**ViT_MAE_Base_Config, output_hidden_states=True)
    # # model=ViT_MAE_for_FT(config=config).cuda()
    # model=ViT_MAE_for_Pretraining(config=config).cuda()

    # input=torch.randn(8,10,128,128).cuda()
    # # print(model)
    # output=model(input)
    # print(output)
    # return_index_vit = [3,6,9,12] # from mmseg [2,5,8,11], huggingface model also returns the first embedding, so index+=1
    # for i in return_index_vit:
    #     print(output.hidden_states[i].shape)


    

    # neck=MultiLevelNeck(in_channels=[768,768,768,768],out_channels=768).to("cuda")
    # features=(torch.rand(8,768,14,14).to("cuda"),torch.rand(8,768,14,14).to("cuda"),torch.rand(8,768,14,14).to("cuda"),torch.rand(8,768,14,14).to("cuda"))
    # out=neck(features)
    # print(out[2].shape)


    


    model=UperNet_Vit().to("cuda")
    input=torch.randn(2,9,224,224).to("cuda")
    # output=vit_model(input, output_hidden_states=True)
    output=model(input)
    # for i in output.logits:
        # print(i.shape)
    # print(output.last_hidden_state.shape)
    print(output.logits.shape)