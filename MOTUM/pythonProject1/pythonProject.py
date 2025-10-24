# kodot e za rabota so cpu, predolgo vreme na izvrshuvanje

import os, json, torch, time, math
import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, RandSpatialCropd, RandFlipd, RandAffined,
    RandGaussianNoised, RandAdjustContrastd, EnsureTyped, SpatialPadd
)
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data.utils import decollate_batch

BASE = r"G:\My Drive\faks\7SEMESTAR\MANU\MOTUM-v.2.2"
DATASET_JSON = os.path.join(BASE, "data", "dataset.json")
SPLITS_JSON  = os.path.join(BASE, "data", "splits", "splits_5fold.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(DATASET_JSON) as f:
    DS = json.load(f)
with open(SPLITS_JSON) as f:
    SPLITS = json.load(f)


def build_lists(fold: int, setup: str):
    all_items = DS["training"]
    fold_info = [s for s in SPLITS if s["fold"] == fold][0]
    tr_ids, va_ids = set(fold_info["train"]), set(fold_info["val"])

    def sel_channels(imgs):
        if setup == "t1ce":
            return {"im": [imgs["t1ce"]]}
        elif setup == "t1ce_flair":
            return {"im": [imgs["t1ce"], imgs["flair"]]}
        elif setup == "all4":
            return {"im": [imgs[m] for m in ["flair", "t1", "t1ce", "t2"]]}
        else:
            raise ValueError("Unknown setup")

    train_list, val_list = [], []
    for it in all_items:
        sample = {
            "pid": it["patient_id"],
            "label_ce": it["labels"]["ce_core"],
            "label_fl": it["labels"]["flair_abn"],
        }
        sample.update(sel_channels(it["images"]))
        (train_list if it["patient_id"] in tr_ids else val_list).append(sample)
    return train_list, val_list


def get_transforms(patch_size=(128, 128, 128)):
    common = [
        LoadImaged(keys=["im", "label_ce", "label_fl"]),
        EnsureChannelFirstd(keys=["im", "label_ce", "label_fl"]),
        Orientationd(keys=["im", "label_ce", "label_fl"], axcodes="RAS", labels=None),
        Spacingd(keys=["im", "label_ce", "label_fl"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest", "nearest")),
        NormalizeIntensityd(keys=["im"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["im", "label_ce", "label_fl"]),
    ]
    train_aug = [
        SpatialPadd(keys=["im", "label_ce", "label_fl"], spatial_size=patch_size, method="symmetric"),
        RandSpatialCropd(keys=["im", "label_ce", "label_fl"], roi_size=patch_size, random_size=False),
        RandFlipd(keys=["im", "label_ce", "label_fl"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["im", "label_ce", "label_fl"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["im", "label_ce", "label_fl"], prob=0.5, spatial_axis=2),
        RandAffined(keys=["im", "label_ce", "label_fl"], prob=0.2,
                    rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest", "nearest")),
        RandAdjustContrastd(keys=["im"], prob=0.15, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["im"], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=["im", "label_ce", "label_fl"]),
    ]
    val_aug = [
        SpatialPadd(keys=["im", "label_ce", "label_fl"], spatial_size=patch_size, method="symmetric"),
        RandSpatialCropd(keys=["im", "label_ce", "label_fl"], roi_size=patch_size, random_size=False),
        EnsureTyped(keys=["im", "label_ce", "label_fl"]),
    ]
    return Compose(common + train_aug), Compose(common + val_aug)


def merge_labels_collate(batch):
    from torch.utils.data.dataloader import default_collate
    b = default_collate(batch)
    label = torch.zeros_like(b["label_ce"], dtype=torch.long).squeeze(1)
    fl = b["label_fl"].squeeze(1).long()
    ce = b["label_ce"].squeeze(1).long()
    label = label + (fl > 0).long() * 1
    label = torch.where(ce > 0, torch.tensor(2, device=label.device), label)
    b["label"] = label
    b.pop("label_ce", None)
    b.pop("label_fl", None)
    return b


def make_dataloaders(fold=0, setup="t1ce",
                     patch_size=(128,128,128), batch_size=1, num_workers=0, cache_rate=0.0):
    train_list, val_list = build_lists(fold, setup)
    train_tf, val_tf = get_transforms(patch_size)

    tr_ds = CacheDataset(train_list, transform=train_tf, cache_rate=cache_rate, num_workers=num_workers)
    va_ds = CacheDataset(val_list, transform=val_tf, cache_rate=cache_rate, num_workers=num_workers)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=False, collate_fn=merge_labels_collate)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False,
                           num_workers=num_workers, pin_memory=False, collate_fn=merge_labels_collate)
    return tr_loader, va_loader


def make_model(n_channels, n_classes=3, base_filters=32):
    return SegResNet(spatial_dims=3, init_filters=base_filters,
                     in_channels=n_channels, out_channels=n_classes,
                     norm='instance', dropout_prob=0.0).to(DEVICE)

def make_loss():
    return DiceCELoss(sigmoid=False, softmax=True, include_background=False,
                      to_onehot_y=False, lambda_dice=1.0, lambda_ce=1.0)

def make_optim(model, lr=2e-4, wd=1e-5):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


post_pred = AsDiscrete(argmax=True)
post_label = AsDiscrete()

@torch.no_grad()
def validate(model, loader, roi_size=(128,128,128)):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    total_time = 0.0
    for batch in loader:
        imgs = batch["im"].to(DEVICE)
        labels = batch["label"].to(DEVICE).long()
        labels = torch.nn.functional.one_hot(labels, num_classes=3)  # [B, D, H, W, C]
        labels = labels.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        t0 = time.time()
        preds = sliding_window_inference(imgs, roi_size=roi_size, sw_batch_size=1, predictor=model)
        total_time += time.time() - t0
        if isinstance(preds, torch.Tensor):
            preds_list = [post_pred(preds)]
        else:
            preds_list = [post_pred(p) for p in decollate_batch(preds)]

        if isinstance(labels, torch.Tensor):
            labs_list = [post_label(labels)]
        else:
            labs_list = [post_label(y) for y in decollate_batch(labels)]

        dice_metric(y_pred=preds_list, y=labs_list)
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice, total_time / max(1, len(loader))

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running = 0.0
    for batch in loader:
        imgs = batch["im"].to(DEVICE)
        labels = batch["label"].to(DEVICE).long()
        labels = torch.nn.functional.one_hot(labels, num_classes=3)  # [B, D, H, W, C]
        labels = labels.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


if __name__ == "__main__":
    SETUPS = [("t1ce", 1), ("t1ce_flair", 2), ("all4", 4)]
    GRIDS = [
        {"base_filters": 32, "lr": 2e-4},
        {"base_filters": 48, "lr": 2e-4},
        {"base_filters": 32, "lr": 1e-4},
    ]

    PATCH = (128, 128, 128)
    BATCH = 1
    EPOCHS = 2
    RESULTS = []

    for setup, in_ch in SETUPS:
        tr_loader, va_loader = make_dataloaders(fold=0, setup=setup, patch_size=PATCH, batch_size=BATCH)
        for cfg in GRIDS:
            model = make_model(n_channels=in_ch, base_filters=cfg["base_filters"])
            loss_fn = make_loss()
            opt = make_optim(model, lr=cfg["lr"])
            best_val = -1.0

            for ep in range(1, EPOCHS+1):
                tr_loss = train_one_epoch(model, tr_loader, opt, loss_fn)
                val_dice, val_time = validate(model, va_loader, roi_size=PATCH)
                print(f"[{setup} | f={cfg['base_filters']} | lr={cfg['lr']}] "
                      f"Ep {ep}/{EPOCHS} | train_loss={tr_loss:.4f} | "
                      f"val_dice={val_dice:.4f} | time/sample={val_time:.2f}s")
                best_val = max(best_val, val_dice)

            RESULTS.append({
                "setup": setup,
                "in_channels": in_ch,
                "base_filters": cfg["base_filters"],
                "lr": cfg["lr"],
                "best_val_dice": float(best_val),
            })

    print("\nResults: -----------------------------------------------")
    RESULTS = sorted(RESULTS, key=lambda x: x["best_val_dice"], reverse=True)
    for r in RESULTS:
        print(r)
