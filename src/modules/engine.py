import logging

import torch
from monai.metrics import DiceMetric
from sklearn.metrics import f1_score


logger = logging.getLogger(__name__)

dice_metric = DiceMetric(include_background=True, reduction="mean")

def train_one_epoch(model,optimizer, loader, device,cls_loss_fn,seg_loss_fn):
    model.train()

    train_loss = 0
    seg_loss_total = 0
    cls_loss_total = 0
    count_batches = 0
    total_accuracy = 0

    logger.info("Starting training epoch with %d batches", len(loader))
    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device).long()
        type(labels)

        optimizer.zero_grad()

        seg_out, cls_out = model(images)
        preds=(seg_out > 0.5).float()  # Thresholding for binary segmentation   
        dice_metric(preds, masks)
        
        # classification loss (always)
        loss_cls = cls_loss_fn(cls_out, labels)
        

        
        # 🔥 per-sample segmentation handling
        loss_seg_batch = 0
        count = 0

        for i in range(masks.shape[0]):
            if torch.any(masks[i] > 0):
                loss_seg_batch += seg_loss_fn(
                    seg_out[i:i+1], masks[i:i+1]
                )
                count += 1

        if count > 0:
            loss_seg = loss_seg_batch / count
            loss = 0.7*loss_seg + 0.3 * loss_cls
            seg_loss_total += loss_seg.item()
        else:
            loss = 0.3 * loss_cls
        
        accuracy = (cls_out.argmax(dim=1) == labels).float().mean().item()
        total_accuracy += accuracy

        cls_loss_total += loss_cls.item()

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count_batches += 1
    
    logger.info(
        "Training epoch loss breakdown: seg_loss=%.6f, cls_loss=%.6f",
        seg_loss_total / count_batches,
        cls_loss_total / count_batches,
    )
    logger.info("Training epoch dice score: %.6f", dice_metric.aggregate().item())
    dice_met=dice_metric.aggregate().item()
    dice_metric.reset()

    return train_loss / count_batches,dice_met,total_accuracy / count_batches

def validation(model, loader, device,cls_loss_fn,seg_loss_fn):
    model.eval()

    val_loss = 0
    seg_loss_total = 0
    cls_loss_total = 0
    count_batches = 0
    total_accuracy = 0
    all_preds=[]
    all_labels=[]


    logger.info("Starting validation with %d batches", len(loader))
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device).long()

            # 🔥 forward pass (missing before)
            seg_out, cls_out = model(images)
            preds=(seg_out > 0.5).float()  # Thresholding for binary segmentation   
            dice_metric(preds, masks)

            # classification loss
            loss_cls = cls_loss_fn(cls_out, labels)

            # segmentation (per sample)
            loss_seg_batch = 0
            count = 0

            for i in range(masks.shape[0]):
                if torch.any(masks[i] > 0):
                    loss_seg_batch += seg_loss_fn(
                        seg_out[i:i+1], masks[i:i+1]
                    )
                    count += 1

            if count > 0:
                loss_seg = loss_seg_batch / count
                loss = loss_seg + 0.5 * loss_cls
                seg_loss_total += loss_seg.item()
            else:
                loss = 0.5 * loss_cls

            cls_loss_total += loss_cls.item()
            pred_class=torch.argmax(cls_out,dim=1)
            all_preds.extend(pred_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            accuracy = (cls_out.argmax(dim=1) == labels).float().mean().item()
            total_accuracy += accuracy


            val_loss += loss.item()
            count_batches += 1

    dice_met=dice_metric.aggregate().item()
    dice_metric.reset()
    f1=f1_score(all_labels,all_preds,average='macro')
    logger.info(
        "Validation completed: loss=%.6f, dice=%.6f, accuracy=%.6f, f1=%.6f",
        val_loss / count_batches,
        dice_met,
        total_accuracy / count_batches,
        f1,
    )

    return val_loss / count_batches,dice_met,total_accuracy / count_batches,f1
