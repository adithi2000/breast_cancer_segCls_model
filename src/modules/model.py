import logging

import torch.nn as nn
from monai.networks.nets import SegResNet,DenseNet121
import torch

logger = logging.getLogger(__name__)

class MaskClassifyModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.seg_model=SegResNet(spatial_dims=2,in_channels=in_channels,out_channels=1)
        self.classifier=DenseNet121(spatial_dims=2,in_channels=in_channels,out_channels=num_classes)

    def forward(self,x):
        seg_out=self.seg_model(x)
        class_out=self.classifier(x)
        return seg_out,class_out

def get_model(in_channels,num_classes):
    logger.info("Initializing MaskClassifyModel: in_channels=%d, num_classes=%d", in_channels, num_classes)
    model=MaskClassifyModel(in_channels,num_classes)

    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5   # helps prevent overfitting
    )
    logger.info("Created Adam optimizer: lr=1e-4, weight_decay=1e-5")

    return {'model':model,'optimizer':optimizer}
