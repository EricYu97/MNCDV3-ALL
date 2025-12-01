from typing import List
import torchvision

from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor, reshape, stack

from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    ModuleList,
    Sigmoid,
    PReLU,
    Sequential,
    Upsample,
)


class PixelwiseLinear(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)




class MixingBlock(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)


class MixingMaskAttentionBlock(Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: List[int],
        fout: List[int],
        generate_masked: bool = False,
    ):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


class UpMask(Module):
    def __init__(
        self,
        scale_factor: float,
        nin: int,
        nout: int,
    ):
        super().__init__()
        self._upsample = Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            PReLU(),
            InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


class ChangeClassifier(Module):
    def __init__(
        self,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        freeze_backbone=False,
        in_channels=3,
    ):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone, in_channels
        )

        # Initialize mixing blocks:
        self._first_mix = MixingMaskAttentionBlock(in_channels*2, in_channels, [in_channels, 10, 5], [10, 5, 1])
        self._mixing_mask = ModuleList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
        )

        # Initialize Upsampling blocks:
        self._up = ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        features = self._encode(ref, test)
        latents = self._decode(features)
        return {"main_predictions":self._classify(latents)}

    def _encode(self, ref, test) -> List[Tensor]:
        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))
        return features

    def _decode(self, features) -> Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping


def _get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone, in_channels,
) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        pretrained=pretrained
    ).features
    print(entire_model[0][0])
    if in_channels!=3:
        # print(in_channels)
        # Original conv1 layer: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        old_conv = entire_model[0][0]
        # Create a new conv1 with the desired number of input channels
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, 
                            kernel_size=old_conv.kernel_size, 
                            stride=old_conv.stride,
                            padding=old_conv.padding,
                            bias=old_conv.bias is not None)

        # Initialize weights for new conv1
        with torch.no_grad():
            if in_channels == 3:
                new_conv.weight.copy_(old_conv.weight)
            elif in_channels > 3:
                # Repeat or average the pretrained weights to fit new number of channels
                new_conv.weight[:, :3] = old_conv.weight
                for i in range(3, in_channels):
                    new_conv.weight[:, i] = old_conv.weight[:, i % 3]  # repeat or average
            else:
                # If fewer than 3 channels, average the weights
                new_conv.weight.copy_(old_conv.weight[:, :in_channels].mean(dim=1, keepdim=True))

        # Replace the model's conv1
        entire_model[0][0] = new_conv

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model

if __name__=="__main__":
    model=ChangeClassifier()
    device=torch.device("mps")
    model=model.to(device)
    a=torch.rand(8,3,256,256).to(device)
    b=torch.rand(8,3,256,256).to(device)
    c=model(a,b)