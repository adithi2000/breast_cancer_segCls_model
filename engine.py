import torch

def train_one_epoch(model,optimizer, loader, device,cls_loss_fn,seg_loss_fn):
    model.train()

    train_loss = 0
    seg_loss_total = 0
    cls_loss_total = 0
    count_batches = 0

    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device).long().squeeze()
        type(labels)

        optimizer.zero_grad()

        seg_out, cls_out = model(images)
        # print("Labels",labels)
        # print("Unique",torch.unique(labels))
        # print("Min:",labels.min().item()," Max:",labels.max().item())
        # assert labels.min() >=0,"negative found"
        # assert labels.max() < 3, "label exceeds num_classes"

        
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
            loss = loss_seg + 0.5 * loss_cls
            seg_loss_total += loss_seg.item()
        else:
            loss = 0.5 * loss_cls

        cls_loss_total += loss_cls.item()

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count_batches += 1

    print("Train Seg Loss:", seg_loss_total / count_batches,
          "Train Cls Loss:", cls_loss_total / count_batches)

    return train_loss / count_batches

def validation(model, loader, device,cls_loss_fn,seg_loss_fn):
    model.eval()

    val_loss = 0
    seg_loss_total = 0
    cls_loss_total = 0
    count_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device).long().squeeze()

            # 🔥 forward pass (missing before)
            seg_out, cls_out = model(images)

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

            val_loss += loss.item()
            count_batches += 1

    print("Val Seg Loss:", seg_loss_total / count_batches,
          "Val Cls Loss:", cls_loss_total / count_batches)

    return val_loss / count_batches
