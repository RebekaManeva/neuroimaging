"""
LoRA fine-tuning of MedGemma for tumor bounding box prediction.
Reads train.jsonl / val.jsonl produced by prepare_dataset.py.

Usage:
  python3 finetune_lora.py \
    --dataset-dir /path/to/dataset_out \
    --output-dir /path/to/finetuned_model \
    --model-id google/medgemma-1.5-4b-it \
    --epochs 5 \
    --batch-size 2 \
    --lr 2e-4 \
    --label ce_core
"""

import os, json, argparse, gc
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# prompt
def make_prompt(modality: str, label: str) -> str:
    seq_map = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1C+", "t2": "T2"}
    seq = seq_map.get(modality, "MRI")
    region = "contrast-enhancing tumor core" if label == "ce_core" else "FLAIR signal abnormality"
    return f"""You are an expert neuroradiologist analyzing a brain MRI slice ({seq} sequence).
Locate the {region} in this image and return a tight bounding box around it.

Return ONLY a valid JSON object inside <JSON> and </JSON> tags.

<JSON>
{{
  "has_tumor": true,
  "bbox_xyxy_norm": [x1, y1, x2, y2],
  "confidence": 0.0-1.0
}}
</JSON>

Rules:
- bbox_xyxy_norm: floats in [0,1], format [x1, y1, x2, y2] top-left to bottom-right
- has_tumor is always true for this dataset
- confidence: your certainty that the box covers the lesion (0.0-1.0)
- If no lesion is visible in this slice, return bbox_xyxy_norm=[0,0,0,0] and has_tumor=false
"""

def make_target(record: dict) -> str:
    bbox = record["bbox_xyxy_norm"]
    has_tumor = record["has_tumor"]
    conf = 0.9 if has_tumor else 0.0
    obj = {
        "has_tumor": has_tumor,
        "bbox_xyxy_norm": [round(x, 4) for x in bbox],
        "confidence": conf,
    }
    return f"<JSON>\n{json.dumps(obj, indent=2)}\n</JSON>"


# dataset
class BBoxDataset(Dataset):
    def __init__(self, jsonl_path, label_filter=None, modality_filter=None, max_image_side=512):
        self.records = []
        self.label_filter = label_filter
        self.modality_filter = modality_filter
        self.max_image_side = max_image_side
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line.strip())
                if label_filter and r.get("label") != label_filter:
                    continue
                if modality_filter and r.get("modality") != modality_filter:
                    continue
                if not os.path.exists(r["image_path"]):
                    continue
                self.records.append(r)
        print(f"  Loaded {len(self.records)} records from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > self.max_image_side:
            s = self.max_image_side / m
            img = img.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)
        prompt = make_prompt(r["modality"], r["label"])
        target = make_target(r)
        return {"image": img, "prompt": prompt, "target": target, "record": r}


# collate
def make_collate_fn(processor):
    def collate_fn(batch):
        inputs_list = []
        targets = []
        for item in batch:
            content = [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["prompt"]},
            ]
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]},
            ]
            inputs_list.append(messages)
            targets.append(item["target"])

        # tokenizing full conversations
        encoded = processor.apply_chat_template(
            inputs_list,
            add_generation_prompt=False,
            continue_final_message=False,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
            padding=True,
        )
        return encoded, targets
    return collate_fn


# training loop
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # loading processor and model
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    )

    # applying LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    # datasets
    print(f"\nLoading datasets (label={args.label})...")
    LABEL_MODALITY = {"ce_core": "t1ce", "flair_abn": "flair"}

    train_ds = BBoxDataset(
        str(Path(args.dataset_dir) / "train.jsonl"),
        label_filter=args.label,
        modality_filter=LABEL_MODALITY.get(args.label),
        max_image_side=args.max_image_side,
    )
    val_ds = BBoxDataset(
        str(Path(args.dataset_dir) / "val.jsonl"),
        label_filter=args.label,
        modality_filter=LABEL_MODALITY.get(args.label),
        max_image_side=args.max_image_side,
    )

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for step, (batch, _) in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

            if step % 20 == 0:
                print(f"  Epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train = train_loss / len(train_loader)

        # val
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch, _ in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch["input_ids"])
                val_loss += outputs.loss.item()
        avg_val = val_loss / max(1, len(val_loader))

        print(f"\nEpoch {epoch}/{args.epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        # saving best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = out_dir / f"best_{args.label}"
            model.save_pretrained(str(save_path))
            processor.save_pretrained(str(save_path))
            print(f"  Saved best model -> {save_path}")

        # saving latest
        latest_path = out_dir / f"latest_{args.label}"
        model.save_pretrained(str(latest_path))
        processor.save_pretrained(str(latest_path))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    p.add_argument("--label", default="ce_core", choices=["ce_core", "flair_abn"],
                   help="Which label to train on")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-image-side", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
