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
       
        ResizeD(keys=["image"], spatial_size=(256, 256)),
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
    image_np=np.transpose(image_np, (2, 0, 1))  # Add channel dimension for grayscale images H,W to C,H,W
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
       
        #=================================================
        # Post-process the segmentation output to create a colored mask
        seg_out=torch.sigmoid(seg_out)  # Apply sigmoid to get probabilities binary mask
        seg_mask = seg_out.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions
        mask=cv2.resize(seg_mask, (image_np.shape[1], image_np.shape[0]))  # Resize to original image size
        mask=mask.squeeze()  # Remove any extra dimensions if present
        mask = (mask > 0.5).astype(np.uint8)  # Threshold the mask to create a binary mask
        colored_mask = np.zeros_like(image_np)  # Create an empty mask with the same shape as the original image
        colored_mask[:,:,0][mask==1]= 255
        colored_mask[:,:,1][mask==1] = 0
        colored_mask[:,:,2][mask==1] = 0
          # Color the mask red where the probability is greater than 0.5
        colored_mask = np.transpose(colored_mask, (1, 2, 0))  # Convert to HWC format for visualization
        image_np = np.transpose(image_np, (1, 2, 0))  # Convert back to HWC format for visualization
        print(f"image shape: {image_np.shape}, mask shape: {mask.shape}, colored_mask shape: {colored_mask.shape}")
        overlayed_image = cv2.addWeighted(image_np, 0.7, colored_mask, 0.3, 0)  # Overlay the original image with the colored mask

        #=================================================
       
       




        overlayed_image = cv2.addWeighted(image_np, 0.7, colored_mask, 0.3,0)
        overlayed_image_pil = Image.fromarray(overlayed_image)
        # overlayed_image_pil.save("output.png")
        buffer=io.BytesIO()
        overlayed_image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        img_str=base64.b64encode(buffer.getvalue()).decode("utf-8")
        

    
    return {"predicted_class": predicted_class, "overlayed_image": img_str}


        
