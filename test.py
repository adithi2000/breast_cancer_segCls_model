# test.py

import torch
import torch.nn as nn

from download_from_s3 import download_from_s3
from engine import validation
from model import get_model
from dataset import create_data_list, create_val_transforms, get_loader


def test():

    # -------------------------
    # 1. Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 2. Data
    # -------------------------
    test_root = "./data/original/test"
    download_from_s3("original/test/", test_root)

    test_data = create_data_list(test_root)

    # ✅ same as validation
    test_transforms = create_val_transforms()

    test_loader = get_loader(test_data, test_transforms, batch_size=4, shuffle=False)

    # -------------------------
    # 3. Model
    # -------------------------
    essential = get_model(in_channels=3, num_classes=3)
    model = essential['model'].to(device)

    # -------------------------
    # 4. Load BEST Model
    # -------------------------
    model_path = "./models/best_model.pth"
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    print("✅ Loaded best model")

    # -------------------------
    # 5. Metrics
    # -------------------------
    cls_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = nn.BCEWithLogitsLoss()
    
        # 🔥 add more metrics here (e.g., Dice, IoU, Accuracy)
    
    total_test_loss=validation(model, test_loader, device, cls_loss_fn, seg_loss_fn)
    print("Test Loss:", total_test_loss)
# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    test()