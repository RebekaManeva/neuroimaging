import os, json, time, warnings, copy, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    ConcatItemsd, DeleteItemsd, EnsureTyped,
    RandFlipd, RandAffined, RandGaussianNoised,
    RandAdjustContrastd, RandScaleIntensityd, RandGaussianSmoothd,
    RandCropByPosNegLabeld, RandShiftIntensityd, MapTransform,
)
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import VNet

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))

DATASET_JSON = os.path.join(PROJECT_ROOT, "datasets", "motum", "dataset_preprocessed.json")
SPLITS_JSON = os.path.join(PROJECT_ROOT, "datasets", "motum", "splits", "split_single.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODS = ["t1", "t1ce", "t2", "flair"]

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

with open(DATASET_JSON, "r", encoding="utf-8") as f:
    DS = json.load(f)

with open(SPLITS_JSON, "r", encoding="utf-8") as f:
    SPLIT = json.load(f)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_lists():
    all_items = DS["training"]
    tr_ids, va_ids = set(SPLIT["train"]), set(SPLIT["val"])

    train_list, val_list = [], []

    for it in all_items:
        pid = it["patient_id"]

        sample = {
            "pid": pid,
            "t1": it["images"]["t1"],
            "t1ce": it["images"]["t1ce"],
            "t2": it["images"]["t2"],
            "flair": it["images"]["flair"],
            "label": it["labels"]["merged_3class"],
        }

        (train_list if pid in tr_ids else val_list).append(sample)

    print(f"Train samples: {len(train_list)} | Val samples: {len(val_list)}")
    return train_list, val_list


class BinarizeLabeld(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            x = d[k]
            d[k] = (x > 0).to(x.dtype)
        return d


def get_transforms(patch_size=(112, 112, 112), num_samples=2, pos=2, neg=1):
    binarize = BinarizeLabeld(keys=["label"])

    common = [
        LoadImaged(keys=MODS + ["label"], image_only=True),
        EnsureChannelFirstd(keys=MODS + ["label"]),
        NormalizeIntensityd(keys=MODS, nonzero=True, channel_wise=True),

        ConcatItemsd(keys=MODS, name="im", dim=0),
        DeleteItemsd(keys=MODS),

        EnsureTyped(keys=["im", "label"], track_meta=False),
        binarize,
    ]

    train_tf = Compose(
        common + [
            RandCropByPosNegLabeld(
                keys=["im", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key="im",
                image_threshold=0.0,
            ),
            RandFlipd(keys=["im", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["im", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["im", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["im", "label"],
                prob=0.20,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
            ),
            RandAdjustContrastd(keys=["im"], prob=0.10, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["im"], prob=0.10, mean=0.0, std=0.01),
            RandScaleIntensityd(keys=["im"], factors=0.10, prob=0.10),
            RandShiftIntensityd(keys=["im"], offsets=0.10, prob=0.10),
            RandGaussianSmoothd(keys=["im"], sigma_x=(0.5, 1.5), prob=0.10),
        ]
    )

    val_tf = Compose(common)
    return train_tf, val_tf


def make_dataloaders(
    patch_size=(112, 112, 112),
    batch_size=1,
    num_workers=2,
    cache_rate=0.3,
    num_samples=2,
    pos=2,
    neg=1,
    seed=42,
):
    tr_list, va_list = build_lists()

    tr_tf, va_tf = get_transforms(
        patch_size=patch_size,
        num_samples=num_samples,
        pos=pos,
        neg=neg,
    )

    tr_ds = CacheDataset(tr_list, tr_tf, cache_rate=cache_rate, num_workers=num_workers)
    va_ds = CacheDataset(va_list, va_tf, cache_rate=cache_rate, num_workers=num_workers)

    g = torch.Generator()
    g.manual_seed(seed)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=list_data_collate,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    va_loader = DataLoader(
        va_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
    )

    return tr_loader, va_loader, tr_list


def make_model():
    model = VNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        act=("elu", {"inplace": True}),
        dropout_prob=0.1,
        dropout_dim=3,
        bias=False,
    )

    return model.to(DEVICE)


def estimate_global_pos_weight(train_list, max_cases=None, clamp_max=15.0):
    tf = Compose([
        LoadImaged(keys=["label"], image_only=True),
        EnsureChannelFirstd(keys=["label"]),
        EnsureTyped(keys=["label"], track_meta=False),
    ])

    items = train_list if max_cases is None else train_list[:max_cases]

    pos_sum = 0.0
    vox_sum = 0.0

    for it in items:
        d = tf({"label": it["label"]})
        lab = d["label"]
        lab = (lab > 0).float()

        pos_sum += float(lab.sum().item())
        vox_sum += float(lab.numel())

    neg_sum = max(0.0, vox_sum - pos_sum)

    pw = neg_sum / (pos_sum + 1e-8)
    pw = float(np.clip(pw, 1.0, clamp_max))

    return pw


def make_loss(global_pos_weight: float):
    dice = DiceLoss(sigmoid=True)

    def combined_loss(logits, target):
        loss_dice = dice(logits, target)

        pw = torch.tensor(
            [global_pos_weight],
            device=logits.device,
            dtype=logits.dtype,
        )

        loss_bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pw,
        )

        return 0.7 * loss_dice + 0.3 * loss_bce

    return combined_loss


def make_optim(model, lr=1e-4, wd=1e-5):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _safe_div(a, b, eps=1e-8):
    return (a + eps) / (b + eps)


def dice_from(tp, fp, fn):
    return _safe_div(2 * tp, 2 * tp + fp + fn)


def iou_from(tp, fp, fn):
    return _safe_div(tp, tp + fp + fn)


def precision_from(tp, fp):
    return _safe_div(tp, tp + fp)


def recall_from(tp, fn):
    return _safe_div(tp, tp + fn)


def f1_from(p, r):
    return _safe_div(2 * p * r, p + r)


def hd95_binary_np(pred, gt):
    try:
        from medpy.metric.binary import hd95
    except Exception:
        return np.nan

    ps, gs = int(pred.sum()), int(gt.sum())

    if ps == 0 and gs == 0:
        return 0.0

    if ps == 0 or gs == 0:
        return 999.0

    try:
        return float(hd95(pred, gt))
    except Exception:
        return np.nan


def save_history(history: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    epochs = list(range(1, len(history["TrainLoss"]) + 1))

    for k, values in history.items():
        if len(values) != len(epochs):
            continue

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, values, marker="o", markersize=3)
        plt.title(f"{k} vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(k)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, f"{k}_vs_epoch.png"),
            dpi=150,
        )

        plt.close()


@torch.no_grad()
def ema_update(ema_model, model, decay=0.999):
    msd = model.state_dict()
    esd = ema_model.state_dict()

    for k in esd.keys():
        if esd[k].dtype.is_floating_point:
            esd[k].mul_(decay).add_(msd[k], alpha=(1.0 - decay))
        else:
            esd[k].copy_(msd[k])


def train_one_epoch(
    model,
    ema_model,
    loader,
    opt,
    loss_fn,
    scaler,
    use_amp,
    grad_clip=1.0,
    ema_decay=0.999,
):
    model.train()

    run = 0.0
    steps = 0

    for step, b in enumerate(loader):
        imgs = b["im"].to(DEVICE)
        lbl = b["label"].to(DEVICE).float()

        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(imgs)
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

        else:
            logits = model(imgs)
            loss = loss_fn(logits, lbl)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

        ema_update(ema_model, model, decay=ema_decay)

        run += float(loss.item())
        steps += 1

        if step == 0:
            with torch.no_grad():
                pred_prob = torch.sigmoid(logits)
                pred_bin = (pred_prob > 0.5).float()

                fg_gt = float(lbl.mean().item())
                fg_pred = float(pred_bin.mean().item())

                print(f"[DBG] fg_gt={fg_gt:.6f} fg_pred={fg_pred:.6f}")

    return run / max(1, steps)


@torch.no_grad()
def validate_full_volume(
    model,
    loader,
    roi_size=(112, 112, 112),
    thr_list=(0.60, 0.65, 0.70, 0.75, 0.80),
    sw_batch_size=1,
    overlap=0.25,
):
    model.eval()

    acc = {
        thr: {"d": [], "i": [], "p": [], "r": [], "f1": [], "hd": []}
        for thr in thr_list
    }

    total_time = 0.0
    used = 0
    skipped_empty_gt = 0

    for b in loader:
        imgs = b["im"].to(DEVICE)
        gt = b["label"].to(DEVICE).float()

        t0 = time.time()

        logits = sliding_window_inference(
            imgs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )

        total_time += (time.time() - t0)

        prob = torch.sigmoid(logits)
        gt_b = (gt > 0.5)

        if gt_b.sum().item() == 0:
            skipped_empty_gt += 1
            continue

        gt_np = gt_b[0, 0].detach().cpu().numpy().astype(np.uint8)

        for thr in thr_list:
            pred = (prob > float(thr))

            tp = int((pred & gt_b).sum().item())
            fp = int((pred & (~gt_b)).sum().item())
            fn = int(((~pred) & gt_b).sum().item())

            p = float(precision_from(tp, fp))
            r = float(recall_from(tp, fn))

            acc[thr]["d"].append(float(dice_from(tp, fp, fn)))
            acc[thr]["i"].append(float(iou_from(tp, fp, fn)))
            acc[thr]["p"].append(p)
            acc[thr]["r"].append(r)
            acc[thr]["f1"].append(float(f1_from(p, r)))

            pred_np = pred[0, 0].detach().cpu().numpy().astype(np.uint8)
            acc[thr]["hd"].append(hd95_binary_np(pred_np, gt_np))

        used += 1

    mean_time = float(total_time / max(1, len(loader)))

    best_thr = thr_list[0]
    best_dice = -1.0

    out_by_thr = {}

    for thr in thr_list:
        d = float(np.mean(acc[thr]["d"])) if acc[thr]["d"] else 0.0

        if d > best_dice:
            best_dice = d
            best_thr = thr

        out_by_thr[thr] = {
            "Dice": d,
            "IoU": float(np.mean(acc[thr]["i"])) if acc[thr]["i"] else 0.0,
            "Precision": float(np.mean(acc[thr]["p"])) if acc[thr]["p"] else 0.0,
            "Recall": float(np.mean(acc[thr]["r"])) if acc[thr]["r"] else 0.0,
            "F1": float(np.mean(acc[thr]["f1"])) if acc[thr]["f1"] else 0.0,
            "HD95": float(np.nanmean(acc[thr]["hd"])) if acc[thr]["hd"] else float("nan"),
        }

    best = out_by_thr[best_thr]

    best.update({
        "BestThr": float(best_thr),
        "Time": mean_time,
        "UsedCases": used,
        "SkippedEmptyGT": skipped_empty_gt,
    })

    return best, out_by_thr


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    SEED = 42
    seed_everything(SEED)

    EPOCHS = 100

    PATCH = (112, 112, 112)
    ROI = (112, 112, 112)

    BATCH = 1
    LR = 1e-4

    NUM_WORKERS = 2
    NUM_SAMPLES = 2

    POS, NEG = 2, 1

    SW_BATCH_SIZE = 1
    OVERLAP = 0.25

    THR_LIST = (0.60, 0.65, 0.70, 0.75, 0.80)

    EMA_DECAY = 0.999
    GRAD_CLIP = 1.0
    CACHE_RATE = 0.3

    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available(), "| device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    run_dir = os.path.join(
        BASE,
        "runs_vnet_binary_ema",
        time.strftime("%Y%m%d_%H%M%S"),
    )

    os.makedirs(run_dir, exist_ok=True)

    tr_loader, va_loader, tr_list = make_dataloaders(
        patch_size=PATCH,
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        cache_rate=CACHE_RATE,
        num_samples=NUM_SAMPLES,
        pos=POS,
        neg=NEG,
        seed=SEED,
    )

    global_pw = estimate_global_pos_weight(
        tr_list,
        max_cases=None,
        clamp_max=15.0,
    )

    print(f"[INFO] Global pos_weight (neg/pos) clamped <=15: {global_pw:.3f}")

    model = make_model()

    ema_model = copy.deepcopy(model).to(DEVICE)

    for p in ema_model.parameters():
        p.requires_grad_(False)

    loss_fn = make_loss(global_pos_weight=global_pw)

    opt = make_optim(model, lr=LR)

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    WARMUP_EPOCHS = 5

    history = {
        k: []
        for k in [
            "TrainLoss",
            "Dice",
            "IoU",
            "Precision",
            "Recall",
            "F1",
            "HD95",
            "ValTime",
            "LR",
            "BestThr",
        ]
    }

    best = -1.0

    best_path = os.path.join(run_dir, "best_model_ema.pt")
    last_path = os.path.join(run_dir, "last_model_ema.pt")

    print(f"VNet | lr={LR} | epochs={EPOCHS}")
    print("----------------------------------------------------------")

    for ep in range(1, EPOCHS + 1):
        if ep <= WARMUP_EPOCHS:
            warm_lr = LR * (0.1 + 0.9 * (ep / WARMUP_EPOCHS))

            for pg in opt.param_groups:
                pg["lr"] = warm_lr

        tr_loss = train_one_epoch(
            model,
            ema_model,
            tr_loader,
            opt,
            loss_fn,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=GRAD_CLIP,
            ema_decay=EMA_DECAY,
        )

        best_val, all_thr = validate_full_volume(
            ema_model,
            va_loader,
            roi_size=ROI,
            thr_list=THR_LIST,
            sw_batch_size=SW_BATCH_SIZE,
            overlap=OVERLAP,
        )

        cur = float(best_val["Dice"])

        if cur > best:
            best = cur
            torch.save(ema_model.state_dict(), best_path)

        history["TrainLoss"].append(float(tr_loss))
        history["Dice"].append(float(best_val["Dice"]))
        history["IoU"].append(float(best_val["IoU"]))
        history["Precision"].append(float(best_val["Precision"]))
        history["Recall"].append(float(best_val["Recall"]))
        history["F1"].append(float(best_val["F1"]))

        history["HD95"].append(
            float(best_val["HD95"])
            if not np.isnan(best_val["HD95"])
            else float("nan")
        )

        history["ValTime"].append(float(best_val["Time"]))
        history["LR"].append(float(opt.param_groups[0]["lr"]))
        history["BestThr"].append(float(best_val["BestThr"]))

        thr_str = " | ".join([
            f"{t:.2f}:{all_thr[t]['Dice']:.3f}"
            for t in THR_LIST
        ])

        print(
            f"Epoch {ep:03d}/{EPOCHS} | "
            f"Train={tr_loss:.4f} | "
            f"Dice={best_val['Dice']:.4f} IoU={best_val['IoU']:.4f} "
            f"Prec={best_val['Precision']:.4f} "
            f"Rec={best_val['Recall']:.4f} "
            f"F1={best_val['F1']:.4f} "
            f"HD95={best_val['HD95']:.3f} | "
            f"Time={best_val['Time']:.3f}s | "
            f"Thr*={best_val['BestThr']:.2f} ({thr_str}) | "
            f"Used={best_val['UsedCases']} "
            f"SkipEmptyGT={best_val['SkippedEmptyGT']} | "
            f"LR={opt.param_groups[0]['lr']:.2e}"
        )

        if ep > WARMUP_EPOCHS:
            scheduler.step()

        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            save_history(history, run_dir)

    torch.save(ema_model.state_dict(), last_path)

    save_history(history, run_dir)

    print("----------------------------------------------------------")
    print(f"Best Dice (EMA): {best:.4f}")
    print(f"Saved best: {best_path}")
    print(f"Saved last: {last_path}")
    print(f"Run dir: {run_dir}")