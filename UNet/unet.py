import torch
import os, glob, random, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torchio as tio
from medpy.metric.binary import hd95
import shutil
import re
import torch.optim as optim
from scipy import ndimage

DATA_ROOT = "#"
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
VAL_DIR = os.path.join(DATA_ROOT, "Validation")
TEST_DIR = os.path.join(DATA_ROOT, "Test")
PLOTS_DIR = "#"
model_dir = "#"

TARGET_SHAPE = (128, 128, 128)

BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100


def load_nifty(path):
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    return arr


def zscore_normalize(x, mask=None):
    if mask is not None and mask.sum() > 0:
        vals = x[mask > 0]
        return (x - vals.mean()) / (vals.std() + 1e-8)
    else:
        return (x - x.mean()) / (x.std() + 1e-8)


class BratsT1Dataset(Dataset):
    def __init__(self, root_dir, target_shape=(128, 128, 128), augment=False):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise RuntimeError(f"Root {root_dir} does not exist.")
        t1_files = sorted([p for p in self.root.glob("**/*") if
                           p.is_file() and ('t1' in p.name.lower()) and (p.suffix in ['.nii', '.gz'])])
        pairs = []
        for t in t1_files:
            folder = t.parent
            segs = [p for p in folder.glob("*") if
                    p.is_file() and ('seg' in p.name.lower()) and (p.suffix in ['.nii', '.gz'])]
            if segs:
                pairs.append((str(t), str(segs[0])))
                continue

        self.pairs = pairs
        if len(self.pairs) == 0:
            raise RuntimeError(f"No T1 - seg pairs were found under {root_dir}.")

        self.augment = augment
        transforms = []
        transforms.append(tio.Resize(target_shape))

        if augment:
            transforms.append(tio.RandomFlip(axes=(0, 1, 2), p=0.5))
            transforms.append(tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5))
            transforms.append(tio.RandomNoise(mean=0, std=0.01, p=0.2))
        self.transform = tio.Compose(transforms)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        img_arr = load_nifty(img_path)
        seg_arr = load_nifty(seg_path)
        seg_bin = (seg_arr > 0).astype(np.uint8)
        img_arr = zscore_normalize(img_arr, mask=seg_bin)

        subj = tio.Subject(
            img=tio.ScalarImage(tensor=img_arr[np.newaxis, ...]),
            seg=tio.LabelMap(tensor=seg_bin[np.newaxis, ...].astype(np.uint8)))

        subj = self.transform(subj)
        img_t = subj['img'].data.clone().float()
        seg_t = subj['seg'].data.clone().float()
        seg_t = (seg_t > 0.5).float()
        return img_t, seg_t


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, fmaps=(32, 64, 128, 256, 512), dropout=0.2):
        super().__init__()
        self.inc = ConvBlock(in_channels, fmaps[0], dropout=dropout)
        self.down1 = Down(fmaps[0], fmaps[1], dropout=dropout)
        self.down2 = Down(fmaps[1], fmaps[2], dropout=dropout)
        self.down3 = Down(fmaps[2], fmaps[3], dropout=dropout)
        self.down4 = Down(fmaps[3], fmaps[4], dropout=dropout)

        self.up1 = Up(fmaps[4], fmaps[3], dropout=dropout)
        self.up2 = Up(fmaps[3], fmaps[2], dropout=dropout)
        self.up3 = Up(fmaps[2], fmaps[1], dropout=dropout)
        self.up4 = Up(fmaps[1], fmaps[0], dropout=dropout)
        self.outc = nn.Conv3d(fmaps[0], out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)


def precision_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)


def f1_score(pred, target, eps=1e-6):
    p = precision_score(pred, target, eps)
    r = recall_score(pred, target, eps)
    return (2 * p * r + eps) / (p + r + eps)


def hd95_score(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(np.uint8)
    target = target.cpu().numpy().astype(np.uint8)

    if pred.ndim == 5 and pred.shape[0] == 1 and pred.shape[1] == 1:
        pred = pred[0, 0]
        target = target[0, 0]
    elif pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]
        target = target[0]

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    elif pred.sum() == 0 or target.sum() == 0:
        return 999.0

    try:
        return hd95(pred, target)
    except Exception as e:
        return 999.0


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (self.alpha * (1 - pt) ** self.gamma)
        focal_loss = (focal_weight * bce).mean()

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.eps) / (pred.sum() + target.sum() + self.eps)
        dice_loss = 1 - dice

        return focal_loss + dice_loss


def plot_metrics(train_metrics, val_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(val_metrics['dice']) + 1)
    metric_names = ['dice', 'iou', 'precision', 'recall', 'f1', 'hd95']

    for metric_name in metric_names:
        if not val_metrics[metric_name] or not train_metrics[metric_name]:
            print(f"Missing data for metric: {metric_name.upper()}.")
            continue
        plt.figure(figsize=(10, 6))

        plt.plot(epochs, train_metrics[metric_name],
                 label=f'Train {metric_name.upper()}',
                 marker='o', linestyle='-', markersize=4, linewidth=2)

        plt.plot(epochs, val_metrics[metric_name],
                 label=f'Validation {metric_name.upper()}',
                 marker='x', linestyle='--', markersize=4, linewidth=2)

        plt.title(f'Training and Validation {metric_name.upper()} Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)

        if metric_name == 'hd95':
            plt.ylabel('HD95 Distance (lower = better)', fontsize=12)
        else:
            plt.ylabel(f'{metric_name.upper()} Score', fontsize=12)

        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f'{metric_name}_vs_epoch.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()


def main_train():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch.cuda.version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(model_dir, exist_ok=True)

    train_ds = BratsT1Dataset(TRAIN_DIR, target_shape=TARGET_SHAPE, augment=True)
    val_ds = BratsT1Dataset(VAL_DIR, target_shape=TARGET_SHAPE, augment=False)
    test_ds = BratsT1Dataset(TEST_DIR, target_shape=TARGET_SHAPE, augment=False)
    print("Dataset sizes: Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

    train_weights = []
    for i in range(len(train_ds)):
        _, seg = train_ds[i]
        tumor_ratio = seg.sum().item() / seg.numel()
        weight = 1.0 + tumor_ratio * 10
        train_weights.append(weight)

    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=NUM_WORKERS)

    img_t, seg_t = train_ds[0]
    print("Sample shapes (img, seg):", img_t.shape, seg_t.shape, "dtype:", img_t.dtype, seg_t.dtype)
    print(f"Image stats - min: {img_t.min():.3f}, max: {img_t.max():.3f}, mean: {img_t.mean():.3f}")
    print(f"Seg unique values: {torch.unique(seg_t)}")
    print(f"Seg positive pixels: {seg_t.sum().item()} / {seg_t.numel()} ({100 * seg_t.sum().item() / seg_t.numel():.2f}%)")

    model = UNet3D(in_channels=1, out_channels=1, fmaps=(32, 64, 128, 256, 512), dropout=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    criterion = FocalDiceLoss(alpha=0.5, gamma=2.0)

    sample_img, sample_seg = next(iter(train_loader))
    sample_img = sample_img.to(device)
    print("Input shape:", sample_img.shape)
    with torch.no_grad():
        out = model(sample_img)
    print("Output shape:", out.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'hd95': []}
    val_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'hd95': []}
    best_val_dice = -1.0
    best_model_path = os.path.join(model_dir, "best_model.pth")
    last_model_path = os.path.join(model_dir, "last_model.pth")
    patience_counter = 0
    early_stop_patience = 20

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_dice, train_iou, train_prec, train_rec, train_f1, train_hd95 = [], [], [], [], [], []

        for imgs, segs in train_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, segs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                pred = torch.sigmoid(logits)

                for i in range(imgs.size(0)):
                    train_dice.append(dice_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_iou.append(iou_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_prec.append(precision_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_rec.append(recall_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_f1.append(f1_score(pred[i:i + 1], segs[i:i + 1]).item())
                    train_hd95.append(hd95_score(pred[i].unsqueeze(0), segs[i].unsqueeze(0)))

        train_metrics['dice'].append(np.nanmean(train_dice))
        train_metrics['iou'].append(np.nanmean(train_iou))
        train_metrics['precision'].append(np.nanmean(train_prec))
        train_metrics['recall'].append(np.nanmean(train_rec))
        train_metrics['f1'].append(np.nanmean(train_f1))
        train_metrics['hd95'].append(np.nanmean(train_hd95))

        model.eval()
        val_dice, val_iou, val_prec, val_rec, val_f1, val_hd95 = [], [], [], [], [], []
        with torch.no_grad():
            for imgs, segs in val_loader:
                imgs, segs = imgs.to(device), segs.to(device)
                logits = model(imgs)
                pred = torch.sigmoid(logits)

                for i in range(imgs.size(0)):
                    val_dice.append(dice_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_iou.append(iou_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_prec.append(precision_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_rec.append(recall_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_f1.append(f1_score(pred[i:i + 1], segs[i:i + 1]).item())
                    val_hd95.append(hd95_score(pred[i].unsqueeze(0), segs[i].unsqueeze(0)))

        val_metrics['dice'].append(np.nanmean(val_dice))
        val_metrics['iou'].append(np.nanmean(val_iou))
        val_metrics['precision'].append(np.nanmean(val_prec))
        val_metrics['recall'].append(np.nanmean(val_rec))
        val_metrics['f1'].append(np.nanmean(val_f1))
        val_metrics['hd95'].append(np.nanmean(val_hd95))

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        print(f"Train metrics:")
        print(f"  Dice:      {train_metrics['dice'][-1]:.4f}")
        print(f"  IoU:       {train_metrics['iou'][-1]:.4f}")
        print(f"  Precision: {train_metrics['precision'][-1]:.4f}")
        print(f"  Recall:    {train_metrics['recall'][-1]:.4f}")
        print(f"  F1:        {train_metrics['f1'][-1]:.4f}")
        print(f"  HD95:      {train_metrics['hd95'][-1]:.4f}")

        print(f"\nValidation metrics:")
        print(f"  Dice:      {val_metrics['dice'][-1]:.4f}")
        print(f"  IoU:       {val_metrics['iou'][-1]:.4f}")
        print(f"  Precision: {val_metrics['precision'][-1]:.4f}")
        print(f"  Recall:    {val_metrics['recall'][-1]:.4f}")
        print(f"  F1:        {val_metrics['f1'][-1]:.4f}")
        print(f"  HD95:      {val_metrics['hd95'][-1]:.4f}")

        if val_metrics['dice'][-1] > best_val_dice:
            best_val_dice = val_metrics['dice'][-1]
            torch.save(model.state_dict(), best_model_path)
            print(f"\nBest model updated with Val Dice={best_val_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break

        scheduler.step(val_metrics['dice'][-1])

    torch.save(model.state_dict(), last_model_path)
    print(f"Training finished. Last model saved to {last_model_path}")
    print(f"Best model saved to {best_model_path}")
    plot_metrics(train_metrics, val_metrics, PLOTS_DIR)


def test_model(model_path, test_loader, device, save_dir):
    print("Testing...")
    model = UNet3D(in_channels=1, out_channels=1, fmaps=(32, 64, 128, 256, 512), dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    test_dice, test_iou, test_prec, test_rec, test_f1, test_hd95 = [], [], [], [], [], []

    with torch.no_grad():
        for imgs, segs in test_loader:
            imgs, segs = imgs.to(device), segs.to(device)
            logits = model(imgs)
            pred = torch.sigmoid(logits)

            for i in range(imgs.size(0)):
                test_dice.append(dice_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_iou.append(iou_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_prec.append(precision_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_rec.append(recall_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_f1.append(f1_score(pred[i:i + 1], segs[i:i + 1]).item())
                test_hd95.append(hd95_score(pred[i].unsqueeze(0), segs[i].unsqueeze(0)))

    metrics = {
        'Dice': np.nanmean(test_dice),
        'IoU': np.nanmean(test_iou),
        'Precision': np.nanmean(test_prec),
        'Recall': np.nanmean(test_rec),
        'F1': np.nanmean(test_f1),
        'HD95': np.nanmean(test_hd95)
    }

    std_metrics = {
        'Dice': np.nanstd(test_dice),
        'IoU': np.nanstd(test_iou),
        'Precision': np.nanstd(test_prec),
        'Recall': np.nanstd(test_rec),
        'F1': np.nanstd(test_f1),
        'HD95': np.nanstd(test_hd95)
    }

    print("Results: ")
    for metric_name, mean_val in metrics.items():
        std_val = std_metrics[metric_name]
        print(f"{metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, "test_results.txt")
    with open(results_file, 'w') as f:
        f.write("Test results: \n")
        for metric_name, mean_val in metrics.items():
            std_val = std_metrics[metric_name]
            f.write(f"{metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}\n")

    print(f"\n Results saved: {results_file}")

    fig, ax = plt.subplots(figsize=(12, 6))

    metric_names = list(metrics.keys())
    metric_values = [metrics[m] for m in metric_names]
    metric_stds = [std_metrics[m] for m in metric_names]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#95a5a6']

    bars = ax.bar(metric_names, metric_values, yerr=metric_stds,
                  capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Performance - All Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(metric_values) * 1.2])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val, std in zip(bars, metric_values, metric_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(save_dir, 'test_metrics_summary.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    all_scores = [test_dice, test_iou, test_prec, test_rec, test_f1, test_hd95]

    for idx, (metric_name, scores) in enumerate(zip(metric_names, all_scores)):
        ax = axes[idx]
        bp = ax.boxplot([scores], labels=[metric_name], patch_artist=True,
                        boxprops=dict(facecolor=colors[idx], alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    boxplot_path = os.path.join(save_dir, 'test_metrics_distribution.png')
    plt.savefig(boxplot_path, dpi=150)
    plt.close()
    return metrics

if __name__ == '__main__':
    main_train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = BratsT1Dataset(TEST_DIR, target_shape=TARGET_SHAPE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=NUM_WORKERS)

    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        test_results = test_model(best_model_path, test_loader, device, PLOTS_DIR)
    else:
        print(f" Best model not found: {best_model_path}")