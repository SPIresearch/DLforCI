import torch.nn as nn

from typing import Optional
from model.deeplabv3.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder
from ..base import SegmentationModel, SegmentationHead, ClassificationHead
from ..encoders import get_encoder



class Res_classifier(SegmentationModel):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_channels: int = 256,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.pool=nn.AdaptiveMaxPool2d(2)
        self.classifier=nn.Linear(self.encoder.out_channels[-1]*4,classes)
        # self.encoder.make_dilated(
        #     stage_list=[4, 5],
        #     dilation_list=[2, 4]
        # )

        # self.decoder = DeepLabV3Decoder(
        #     in_channels=self.encoder.out_channels[-1],
        #     out_channels=decoder_channels,
        # )

        # self.segmentation_head = SegmentationHead(
        #     in_channels=self.decoder.out_channels,
        #     out_channels=classes,
        #     activation=activation,
        #     kernel_size=1,
        #     upsampling=upsampling,
        # )

        # if aux_params is not None:
        #     self.classification_head = ClassificationHead(
        #         in_channels=self.encoder.out_channels[-1], **aux_params
        #     )
        # else:
        #     self.classification_head = None
    def forward(self, x):
        f=self.encoder(x)[-1]
        #import pdb;pdb.set_trace()
        x=self.pool(f)
        x=x.view(x.shape[0],-1)
        x=self.classifier(x)
        return x,f#super().forward(x)




