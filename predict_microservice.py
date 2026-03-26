import base64

from select_best_model import select_best_model
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
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=["image"]),
        # RepeatChanneld(keys=["image"], repeats=3),  # Convert grayscale to 3-channel  
        Lambdad(keys=["image"], func=lambda x: x if x.shape[0] ==3 else x.repeat(3, 1, 1)), 
        ResizeD(keys=["image"], spatial_size=(256, 256)),
        # Lambdad(keys="mask", func=lambda x: x[0:1, ...]),
        # Lambdad(keys="mask", func=lambda x: x.astype("float32")),
        ScaleIntensityd(keys="image"),
        ToTensord(keys=['image'])
    ]
)

app=FastAPI()
model=select_best_model()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')  # Convert to RGB
    image_np = np.array(image)
    
    # Create a dictionary for the transforms
    data_dict = {"image": image_np}
    
    # Apply the transforms
    transformed_data = Inference_transforms(data_dict)
    
    # Get the transformed image tensor
    input_tensor = transformed_data["image"].unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        model.eval()
        seg_out,cls_out = model(input_tensor)
        predicted_class = torch.argmax(cls_out, dim=1).item()
        seg_mask=(seg_out > 0.5).float().squeeze().cpu().numpy()  # Convert to binary mask and remove batch dimension

        seg_mask=(seg_mask > 0 ).astype(np.uint8)  # Convert to binary mask and scale to [0, 255]
        mask=cv2.resize(seg_mask,(image_np.shape[1],image_np.shape[0]),interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(image_np)
        colored_mask[:,:,0] = mask * 255  # Red channel for the mask
        overlayed_image = cv2.addWeighted(image_np, 0.7, colored_mask, 0.3)
        overlayed_image_pil = Image.fromarray(overlayed_image)
        # overlayed_image_pil.save("output.png")
        buffer=io.BytesIO()
        overlayed_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        img_str=base64.b64encode(buffer.getvalue()).decode("utf-8")
        

    
    return {"predicted_class": predicted_class, "overlayed_image": img_str}


        
