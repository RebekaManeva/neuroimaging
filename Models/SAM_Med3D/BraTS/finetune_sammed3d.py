import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from segment_anything.build_sam3D import sam_model_registry3D
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tuning SAM-Med3D for BraTS whole-tumor segmentation"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Pretrained weights")
    p.add_argument("--train-data-path", required=True,
                   help="Directory with imagesTr and labelsTr for training")
    p.add_argument("--val-data-path", required=True,
                   help="Directory with imagesTr and labelsTr for validation")
    p.add_argument("--work-dir", default="work_dir/brats_wt",
                   help="Where to save checkpoints and logs")
    p.add_argument("--model-type", default="vit_b_ori",
                   help="Model type")
    p.add_argument("--img-size", type=int, default=128,
                   help="3D crop size")
    p.add_argument("--train-vision-encoder", action="store_true",
                   help="Fine-tune last 2 ViT blocks + neck")
    p.add_argument("--unfreeze-last-n-blocks", type=int, default=2,
                   help="Number of last ViT blocks to unfreeze")

    p.add_argument("--num-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--num-clicks", type=int, default=5,
                   help="Number of interactive click iterations during training")
    p.add_argument("--lr", type=float, default=8e-4,
                   help="LR for mask decoder")
    p.add_argument("--encoder-lr", type=float, default=8e-5,
                   help="LR for vision encoder")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--lr-patience", type=int, default=3,
                   help="Epochs without improvement before ReduceLROnPlateau")
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def get_transform(img_size: int, augment: bool) -> tio.Compose:
    ops = [
        tio.ToCanonical(),
        tio.CropOrPad(
            mask_name="label",
            target_shape=(img_size, img_size, img_size),
        ),
    ]
    if augment:
        ops.append(tio.RandomFlip(axes=(0, 1, 2)))
    return tio.Compose(ops)


def build_dataloader(data_path: str, img_size: int, batch_size: int,
                     num_workers: int, augment: bool,
                     threshold: int = 1000) -> DataLoader:
    dataset = Dataset_Union_ALL(
        paths=[data_path],
        transform=get_transform(img_size, augment),
        threshold=threshold,
    )
    return Union_Dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=True,
    )


def configure_trainable_params(model, args):
    for p in model.parameters():
        p.requires_grad_(False)

    for p in model.mask_decoder.parameters():
        p.requires_grad_(True)

    if args.train_vision_encoder:
        n = args.unfreeze_last_n_blocks
        log.info(f"Unfreeze: mask_decoder + neck + last {n} ViT blocks")

        if hasattr(model.image_encoder, "neck"):
            for p in model.image_encoder.neck.parameters():
                p.requires_grad_(True)

        if hasattr(model.image_encoder, "layers"):
            total = len(model.image_encoder.layers)
            for i in range(total - n, total):
                for p in model.image_encoder.layers[i].parameters():
                    p.requires_grad_(True)
            log.info(f"   ViT blocks {total - n}..{total - 1} trainable")
        else:
            log.warning("Cannot find model.image_encoder.layers")
    else:
        log.info("Unfreeze: only mask_decoder")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(f"   Trainable: {n_train:,} / {n_total:,} parameters")


def build_optimizer(model, args):
    encoder_params = [p for p in model.image_encoder.parameters()
                      if p.requires_grad]
    decoder_params = list(model.mask_decoder.parameters())

    param_groups = [{"params": decoder_params, "lr": args.lr, "name": "decoder"}]
    if encoder_params:
        param_groups.append(
            {"params": encoder_params, "lr": args.encoder_lr, "name": "encoder"}
        )

    return torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )


def batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                 points=None, device="cuda"):
    sparse_emb, dense_emb = sam_model.prompt_encoder(
        points=points, boxes=None, masks=low_res_masks,
    )
    low_res_masks, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding.to(device),
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    prev_masks = F.interpolate(
        low_res_masks, size=gt3D.shape[-3:],
        mode="trilinear", align_corners=False,
    )
    return low_res_masks, prev_masks


def interaction_loop(sam_model, image_embedding, gt3D, num_clicks,
                     img_size, device):
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    prev_masks = torch.zeros_like(gt3D).to(device)
    low_res = F.interpolate(
        prev_masks.float(),
        size=(img_size // 4, img_size // 4, img_size // 4),
    )

    click_points  = []
    click_labels  = []
    total_loss    = 0.0
    random_insert = np.random.randint(2, 9)

    for click_idx in range(num_clicks):
        batch_pts, batch_lbls = get_next_click3D_torch_2(prev_masks, gt3D)
        pts_co = torch.cat(batch_pts,  dim=0).to(device)
        pts_la = torch.cat(batch_lbls, dim=0).to(device)
        click_points.append(pts_co)
        click_labels.append(pts_la)

        if click_idx == random_insert or click_idx == num_clicks - 1:
            low_res, prev_masks = batch_forward(
                sam_model, image_embedding, gt3D, low_res,
                points=None, device=device,
            )
        else:
            pts_in  = torch.cat(click_points,  dim=1).to(device)
            lbls_in = torch.cat(click_labels,  dim=1).to(device)
            low_res, prev_masks = batch_forward(
                sam_model, image_embedding, gt3D, low_res,
                points=[pts_in, lbls_in], device=device,
            )

        total_loss += seg_loss(prev_masks, gt3D)

    return prev_masks, total_loss


def dice_score(pred_masks, gt3D):
    pred = (pred_masks > 0.5)
    true = (gt3D > 0)
    scores = []
    for i in range(true.shape[0]):
        vol_sum = true[i].sum() + pred[i].sum()
        if vol_sum == 0:
            scores.append(float("nan"))
            continue
        inter = (true[i] & pred[i]).sum()
        scores.append((2 * inter / vol_sum).item())
    valid = [s for s in scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else 0.0


def find_latest_checkpoint(work_dir: str):
    work_dir = Path(work_dir)
    latest = work_dir / "sam_model_latest.pth"
    return str(latest) if latest.exists() else None


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_dice, losses, dices):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dice":            best_dice,
        "losses":               losses,
        "dices":                dices,
    }, path)


def run_epoch(model, loader, optimizer, scaler, accum_steps, num_clicks,
              img_size, device, train=True):
    model.train(train)
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    total_loss = 0.0
    total_dice = 0.0
    n_steps    = 0

    ctx = torch.enable_grad if train else torch.no_grad
    if train:
        optimizer.zero_grad()

    with ctx():
        for step, data3D in enumerate(tqdm(loader, desc="train" if train else "val",
                                           leave=False)):
            try:
                image3D = data3D["image"]
                gt3D    = data3D["label"]
            except Exception as e:
                log.warning(f"Skipping batch {step}: {e}")
                continue

            image3D = norm_transform(image3D.squeeze(1)).unsqueeze(1)
            image3D = image3D.to(device)
            gt3D    = gt3D.to(device).float()

            device_type = "cuda" if "cuda" in str(device) else "cpu"
            with amp.autocast(device_type=device_type):
                img_emb = model.image_encoder(image3D)
                pred_masks, loss = interaction_loop(
                    model, img_emb, gt3D, num_clicks, img_size, device
                )

            total_loss += loss.item()
            total_dice += dice_score(pred_masks, gt3D)
            n_steps    += 1

            if train:
                scaler.scale(loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

    return total_loss / max(1, n_steps), total_dice / max(1, n_steps)


def train(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading SAM-Med3D ({args.model_type}): {args.checkpoint}")
    model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning(f"Missing keys: {missing[:5]}...")
    else:
        log.warning(f"Checkpoint not found: {args.checkpoint}. Training from scratch.")

    configure_trainable_params(model, args)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr,
    )
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = amp.GradScaler(device=device_type)

    log.info("Loading datasets.")
    train_loader = build_dataloader(
        args.train_data_path, args.img_size, args.batch_size,
        args.num_workers, augment=True,
    )
    val_loader = build_dataloader(
        args.val_data_path, args.img_size, args.batch_size,
        args.num_workers, augment=False,
    )

    start_epoch      = 0
    best_dice        = 0.0
    patience_counter = 0
    losses, dices    = [], []

    if args.resume:
        ckpt_path = find_latest_checkpoint(args.work_dir)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = state["epoch"] + 1
            best_dice   = state["best_dice"]
        else:
            log.warning("No checkpoint found for resume.")

    log.info("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_dice = run_epoch(
            model, train_loader, optimizer, scaler,
            args.accumulation_steps, args.num_clicks,
            args.img_size, device, train=True,
        )
        val_loss, val_dice = run_epoch(
            model, val_loader, optimizer, scaler,
            args.accumulation_steps, args.num_clicks,
            args.img_size, device, train=False,
        )

        scheduler.step(val_dice)

        log.info(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        save_checkpoint(
            work_dir / "sam_model_latest.pth",
            model, optimizer, scheduler, epoch,
            best_dice, losses, dices,
        )

        if val_dice > best_dice + 1e-4:
            best_dice = val_dice
            patience_counter = 0
            save_checkpoint(
                work_dir / "sam_model_best.pth",
                model, optimizer, scheduler, epoch,
                best_dice, losses, dices,
            )
            log.info(f"New best model: {best_dice:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            log.info("Early stopping triggered.")
            break

    log.info(f"Training finished. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)