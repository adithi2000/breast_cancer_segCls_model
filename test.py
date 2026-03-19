# test.py

import torch
import torch.nn as nn

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
    test_root = "path_to_test_data"

    test_data = create_data_list(test_root)

    # ✅ same as validation
    test_transforms = create_val_transforms()

    test_loader = get_loader(test_data, test_transforms, batch_size=4, shuffle=False)

    # -------------------------
    # 3. Model
    # -------------------------
    essential = get_model(in_channels=1, num_classes=3).to(device)
    model = essential['model']

    # -------------------------
    # 4. Load BEST Model
    # -------------------------
    checkpoint = torch.load("/models/best_model.pth", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("✅ Loaded best model")

    # -------------------------
    # 5. Metrics
    # -------------------------
    total_cls_correct = 0
    total_samples = 0

    total_seg_dice = 0
    count_seg = 0

    # -------------------------
    # 6. Inference Loop
    # -------------------------
    with torch.no_grad():
        for batch in test_loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            seg_out, cls_out = model(images)

            # -------------------------
            # Classification Accuracy
            # -------------------------
            preds = torch.argmax(cls_out, dim=1)

            total_cls_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # -------------------------
            # Segmentation Dice Score
            # -------------------------
            seg_pred = (torch.sigmoid(seg_out) > 0.5).float()

            for i in range(masks.shape[0]):

                if masks[i].sum() > 0:

                    intersection = (seg_pred[i] * masks[i]).sum()

                    dice = (2 * intersection) / (
                        seg_pred[i].sum() + masks[i].sum() + 1e-5
                    )

                    total_seg_dice += dice.item()
                    count_seg += 1

    # -------------------------
    # 7. Final Metrics
    # -------------------------
    accuracy = total_cls_correct / total_samples

    if count_seg > 0:
        avg_dice = total_seg_dice / count_seg
    else:
        avg_dice = 0

    print("\n==============================")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Dice Score: {avg_dice:.4f}")
    print("==============================")


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    test()