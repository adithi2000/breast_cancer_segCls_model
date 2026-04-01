import torch.nn as nn
from monai.networks.nets import SegResNet
import torch

class MaskClassifyModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.seg_model=SegResNet(spatial_dims=2,in_channels=in_channels,out_channels=1)
        self.classifier=nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            ,
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,num_classes)
        )



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
