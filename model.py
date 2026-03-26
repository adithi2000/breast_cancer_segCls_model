import torch.nn as nn
from monai.networks.nets import SegResNet
import torch

class MaskClassifyModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.backbone=SegResNet(spatial_dims=2,in_channels=in_channels,out_channels=1)
        self.classifier=self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),

        nn.Linear(1,16),
        nn.ReLU(),
         nn.Dropout(0.3),

        # nn.Linear(32,64),
        # nn.ReLU(),
        # nn.Dropout(0.3),

        # # nn.Linear(64,128),
        # # nn.ReLU(),
        # #too much expansion causes overfitting,instability
        # # nn.Dropout(0.3),

        # # nn.Linear(128,64),
        # # nn.ReLU(),

        # nn.Linear(64,32),
        # nn.ReLU(),
        #  nn.Dropout(0.3),
        # #logits
        
        nn.Linear(16,num_classes)

        # nn.Linear(1,num_classes)
        )

    def forward(self,x):
        seg_out=self.backbone(x)
        class_out=self.classifier(seg_out)
        return seg_out,class_out

def get_model(in_channels,num_classes):
    model=MaskClassifyModel(in_channels,num_classes)

    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5   # helps prevent overfitting
    )

    return {'model':model,'optimizer':optimizer}
