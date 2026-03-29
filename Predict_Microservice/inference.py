import base64

from Predict_Microservice.select_best_model import select_best_model
import torch
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import cv2

from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    ResizeD,
    ScaleIntensityd,
    ToTensord )

Inference_transforms=Compose(
    [
       
        ResizeD(keys=["image"], spatial_size=(256, 256)),
        ScaleIntensityd(keys="image"),
        ToTensord(keys=['image'])
    ]
)


def predict_pipeline(model, contents):
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    image_np = np.transpose(image_np, (2, 0, 1))

    data_dict = {"image": image_np}
    transformed_data = Inference_transforms(data_dict)
    input_tensor = transformed_data["image"].unsqueeze(0)

    with torch.no_grad():
        model.eval()
        seg_out, cls_out = model(input_tensor)
        predicted_class = torch.argmax(cls_out, dim=1).item()

        seg_out = torch.sigmoid(seg_out)
        seg_mask = seg_out.squeeze(0).squeeze(0).cpu().numpy()

        mask = cv2.resize(seg_mask, (image_np.shape[1], image_np.shape[0]))
        mask = (mask > 0.5).astype(np.uint8)

        colored_mask = np.zeros_like(image_np)
        colored_mask[0][mask == 1] = 255

        colored_mask = np.transpose(colored_mask, (1, 2, 0))
        image_np = np.transpose(image_np, (1, 2, 0))

        overlayed_image = cv2.addWeighted(image_np, 0.7, colored_mask, 0.3, 0)

        buffer = io.BytesIO()
        Image.fromarray(overlayed_image).save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "predicted_class": predicted_class,
        "overlayed_image": img_str
    }