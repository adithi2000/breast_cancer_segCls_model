import logging
import os
from collections import Counter

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Lambdad, ResizeD, RandFlipd, RandRotate90d,RepeatChanneld
)
from monai.data import Dataset
from monai.data import DataLoader
from torch.utils.data import WeightedRandomSampler


logger = logging.getLogger(__name__)

def create_data_list(root):
    data_list=[]
    class_map = {
    "normal": 0,
    "benign": 1,
    "malignant": 2
        }
    for class_name in class_map.keys():
        class_path=os.path.join(root,class_name)
        if not os.path.isdir(class_path):
            logger.warning("Class directory not found: %s", class_path)
            continue
        for file in os.listdir(class_path):
            if '_mask' not in file:
                image_path=os.path.join(class_path,file)
                mask_name = file.replace(".png", "_mask.png")
                mask_path = os.path.join(class_path, mask_name)

                if os.path.exists(mask_path):
                    data_list.append({
                    "image": image_path,
                    "mask": mask_path,
                    "label": class_map[class_name]  # optional
                    })
                else:
                    logger.warning("Missing mask for image: %s", image_path)
    logger.info("Created data list from %s with %d samples", root, len(data_list))
    return data_list

def create_train_transforms():
    logger.debug("Creating training transforms")
    train_transforms=Compose(
    [
        LoadImaged(keys=['image','mask']),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Lambdad(keys=["image"], func=lambda x: x if x.shape[0] ==3 else x.repeat(3, 1, 1)),  # Ensure image is float32
        ResizeD(keys=["image", "mask"], spatial_size=(256, 256)),

        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "mask"], prob=0.3),

        Lambdad(keys="mask", func=lambda x: x[0:1, ...]),
        Lambdad(keys="mask", func=lambda x: x.astype("float32")),
        ScaleIntensityd(keys="image"),
        ToTensord(keys=['image','mask','label'])
        
        
    ]
    )
    return train_transforms

def create_val_transforms():
    logger.debug("Creating validation transforms")
    val_transforms=Compose(
    [
        LoadImaged(keys=['image','mask']),
        EnsureChannelFirstd(keys=["image", "mask"]),
        # RepeatChanneld(keys=["image"], repeats=3),  # Convert grayscale to 3-channel  
        Lambdad(keys=["image"], func=lambda x: x if x.shape[0] ==3 else x.repeat(3, 1, 1)), 
        ResizeD(keys=["image", "mask"], spatial_size=(256, 256)),
        Lambdad(keys="mask", func=lambda x: x[0:1, ...]),
        Lambdad(keys="mask", func=lambda x: x.astype("float32")),
        ScaleIntensityd(keys="image"),
        ToTensord(keys=['image','mask','label'])
        
        
    ]
    )
    return val_transforms

def get_loader(data_list, transforms_, batch_size=4, shuffle=False):
    ds=Dataset(data=data_list,transform=transforms_)
    labels = [item["label"] for item in data_list]
    
    class_counts = Counter(labels)
    logger.info("Creating data loader: samples=%d, batch_size=%d, class_counts=%s", len(data_list), batch_size, dict(class_counts))
    class_weights = {
    cls: 1.0 / count for cls, count in class_counts.items()
    }
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
    )

    loader=DataLoader(ds, batch_size=batch_size,sampler=sampler)
    logger.debug("Data loader created with weighted random sampler")
    return loader

