import torch.nn as nn
from monai.networks.nets import SegResNet,DenseNet121
import torch

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
    model=MaskClassifyModel(in_channels,num_classes)

    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5   # helps prevent overfitting
    )

    return {'model':model,'optimizer':optimizer}
