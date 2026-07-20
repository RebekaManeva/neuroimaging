import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SamModel, SamProcessor

random.seed(42)
torch.manual_seed(42)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="flaviagiammarino/medsam-vit-base",
                    help="HF repo id of the MedSAM checkpoint (SAM/ViT-Base architecture, "
                         "transformers-compatible weights).")
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output_medsam")
    p.add_argument("--label_col", type=str, default="whole_tumor", choices=["whole_tumor"])
    p.add_argument("--modality", type=str, default="flair", choices=["flair", "t1", "t1ce", "t2"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--encoder_lr", type=float, default=1e-6,
                    help="Learning rate for the vision encoder when fine-tuning encoder layers. "
                         "Kept much lower than --lr since the encoder is already strongly "
                         "pretrained and large steps risk catastrophic forgetting.")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr_patience", type=int, default=2,
                    help="Epochs without val_loss improvement before ReduceLROnPlateau halves "
                         "the LR. Should be smaller than --early_stopping_patience so the LR "
                         "gets a chance to drop before training gives up entirely.")
    p.add_argument("--lr_factor", type=float, default=0.5,
                    help="Multiplicative factor applied to LR on each scheduler reduction.")
    p.add_argument("--min_lr", type=float, default=1e-7,
                    help="Floor for the LR scheduler -- it will not reduce LR below this.")
    p.add_argument("--min_bbox_area_frac", type=float, default=0.005,
                    help="Skip tumor slices whose GT bbox covers less than this fraction of the "
                         "image area (filters out near-empty/noise lesions).")
    p.add_argument("--box_jitter_px", type=float, default=10.0,
                    help="Random outward jitter (pixels) added to the GT box prompt during "
                         "training, as used in the MedSAM paper, so the model doesn't overfit "
                         "to perfectly tight boxes.")
    p.add_argument("--train_vision_encoder", action="store_true",
                    help="Fine-tune the ENTIRE image encoder (93M params). If not set, "
                         "the script defaults to the Custom Mode (~20M params).")
    p.add_argument("--early_stopping_patience", type=int, default=6,
                    help="Should be larger than --lr_patience so at least one or two LR drops "
                         "get a chance to help before training stops for good.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def load_records(path, label_col, modality, min_bbox_area_frac):
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("label") != label_col or r.get("modality") != modality:
                continue
            if not r.get("has_tumor"):
                continue
            bbox = r.get("bbox_xyxy_norm")
            if not bbox:
                continue
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < min_bbox_area_frac:
                continue
            if not os.path.exists(r.get("image_path", "")):
                continue
            if not r.get("mask_image_path") or not os.path.exists(r["mask_image_path"]):
                continue
            records.append(r)
    return records


class MedSamDataset(Dataset):
    def __init__(self, records, processor, augment=False, box_jitter_px=10.0):
        self.records = records
        self.processor = processor
        self.augment = augment
        self.box_jitter_px = box_jitter_px

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = Image.open(r["image_path"]).convert("RGB")
        mask = Image.open(r["mask_image_path"]).convert("L")
        mask_arr = (np.array(mask) > 0).astype(np.float32)

        w, h = image.size
        x1, y1, x2, y2 = r["bbox_xyxy_norm"]
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

        if self.augment and self.box_jitter_px > 0:
            x1 = max(0.0, x1 - random.uniform(0, self.box_jitter_px))
            y1 = max(0.0, y1 - random.uniform(0, self.box_jitter_px))
            x2 = min(float(w), x2 + random.uniform(0, self.box_jitter_px))
            y2 = min(float(h), y2 + random.uniform(0, self.box_jitter_px))

        inputs = self.processor(image, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.from_numpy(mask_arr)
        return inputs


def get_pred_logits(outputs):
    pm = outputs.pred_masks
    while pm.dim() > 3:
        pm = pm[:, 0]
    return pm


def dice_bce_loss(pred_logits, target, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred_prob = torch.sigmoid(pred_logits)
    pred_flat = pred_prob.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2 * intersection + eps) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)
    dice_loss = 1 - dice.mean()
    return bce + dice_loss


def run_forward_loss(model, batch, device):
    pixel_values = batch["pixel_values"].to(device)
    input_boxes = batch["input_boxes"].to(device)
    gt_masks = batch["ground_truth_mask"].to(device)

    outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
    pred_logits = get_pred_logits(outputs)
    pred_logits_up = F.interpolate(
        pred_logits.unsqueeze(1), size=gt_masks.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(1)
    return dice_bce_loss(pred_logits_up, gt_masks)


def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-epoch")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("checkpoint-epoch")[-1]))
    path = os.path.join(output_dir, ckpts[-1], "state.pt")
    if os.path.exists(path):
        print(f"Found checkpoint for resume: {path}")
        return path
    return None


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading MedSAM: {args.model_id}")
    processor = SamProcessor.from_pretrained(args.model_id)
    model = SamModel.from_pretrained(args.model_id).to(device)

    for param in model.prompt_encoder.parameters():
        param.requires_grad_(False)

    if args.train_vision_encoder:
        print(" Full Fine-Tuning Mode: Unfreezing the ENTIRE model.")
        for param in model.vision_encoder.parameters():
            param.requires_grad_(True)
    else:
        print(" Custom Mode: Unfreezing Mask Decoder + Neck + Last 2 ViT Blocks.")
        
        for param in model.vision_encoder.parameters():
            param.requires_grad_(False)

        if hasattr(model.vision_encoder, 'neck'):
            for param in model.vision_encoder.neck.parameters():
                param.requires_grad_(True)
                
        if hasattr(model.vision_encoder, 'layers'):
            num_blocks = len(model.vision_encoder.layers)
            for i in range(num_blocks - 2, num_blocks):
                for param in model.vision_encoder.layers[i].parameters():
                    param.requires_grad_(True)

    for param in model.mask_decoder.parameters():
        param.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    has_encoder_grad = any(p.requires_grad for p in model.vision_encoder.parameters())

    if has_encoder_grad:
        optimizer = torch.optim.AdamW([
            {"params": model.mask_decoder.parameters(), "lr": args.lr, "name": "decoder"},
            {"params": [p for p in model.vision_encoder.parameters() if p.requires_grad], "lr": args.encoder_lr, "name": "encoder"},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )

    print(f"\nLoading datasets (label={args.label_col}, modality={args.modality})...")
    train_records = load_records(args.train_path, args.label_col, args.modality, args.min_bbox_area_frac)
    val_records = load_records(args.val_path, args.label_col, args.modality, args.min_bbox_area_frac)
    print(f"  train: {len(train_records)} tumor-positive slices")
    print(f"  val:   {len(val_records)} tumor-positive slices")
    if not train_records:
        raise RuntimeError("No training records found after filtering.")

    train_ds = MedSamDataset(train_records, processor, augment=True, box_jitter_px=args.box_jitter_px)
    val_ds = MedSamDataset(val_records, processor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    ckpt_path = find_last_checkpoint(args.output_dir)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        patience_counter = state["patience_counter"]
        print(f"Resuming training from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
    else:
        print("No previous checkpoint found - starting training from scratch.")

    print("Starting training loop...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            loss = run_forward_loss(model, batch, device) / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum
            n_steps += 1
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item() * args.grad_accum:.4f}")

        avg_train_loss = running_loss / max(1, n_steps)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += run_forward_loss(model, batch, device).item()
                n_val += 1
        avg_val_loss = val_loss / max(1, n_val)

        scheduler.step(avg_val_loss)
        current_lrs = [round(g["lr"], 10) for g in optimizer.param_groups]
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  lr={current_lrs}")

        improved = avg_val_loss < best_val_loss - 1e-5
        if improved:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        }, os.path.join(ckpt_dir, "state.pt"))

        if improved:
            best_dir = os.path.join(args.output_dir, "best")
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  -> new best model (val_loss={best_val_loss:.4f}), saved to {best_dir}")

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Training complete. Final weights saved to {final_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)