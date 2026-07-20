import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import directed_hausdorff
from transformers import SamModel, SamProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model-dir", required=True,
                    help="Path or HF repo id of the MedSAM model to evaluate -- e.g. "
                         "output_medsam/best, output_medsam/final, or "
                         "flaviagiammarino/medsam-vit-base for the un-fine-tuned baseline.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="whole_tumor", choices=["whole_tumor"])
    p.add_argument("--modality", default="flair", choices=["flair", "t1", "t1ce", "t2"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--min-bbox-area-frac", type=float, default=0.0,
                    help="Skip tumor slices whose GT bbox covers less than this fraction of the image.")
    return p.parse_args()


def mask_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = inter / union if union > 0 else (1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0)
    dice = (2 * inter) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else (
        1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0)
    precision = inter / pred.sum() if pred.sum() > 0 else (1.0 if gt.sum() == 0 else 0.0)
    recall = inter / gt.sum() if gt.sum() > 0 else (1.0 if pred.sum() == 0 else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if pred.sum() == 0 and gt.sum() == 0:
        hd95 = 0.0
    elif pred.sum() == 0 or gt.sum() == 0:
        # diagonal of the volume/image as a worst-case distance
        hd95 = float(np.sqrt(sum(s ** 2 for s in pred.shape)))
    else:
        coords_pred = np.argwhere(pred)
        coords_gt = np.argwhere(gt)
        d1 = directed_hausdorff(coords_pred, coords_gt)[0]
        d2 = directed_hausdorff(coords_gt, coords_pred)[0]
        hd95 = max(d1, d2)

    return float(iou), float(dice), float(precision), float(recall), float(f1), float(hd95)


class MedSamInferenceDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = Image.open(r["image_path"]).convert("RGB")
        gt_mask = (np.array(Image.open(r["mask_image_path"]).convert("L")) > 0).astype(np.uint8)
        w, h = image.size
        x1, y1, x2, y2 = r["bbox_xyxy_norm"]
        box = [x1 * w, y1 * h, x2 * w, y2 * h]
        return {"image": image, "box": box, "gt_mask": gt_mask, "record_idx": idx}


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "boxes": [b["box"] for b in batch],
        "gt_masks": [b["gt_mask"] for b in batch],
        "record_indices": [b["record_idx"] for b in batch],
    }


def run_slice_inference(args, records_to_process, results_path, pred_mask_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {args.model_dir}")
    model = SamModel.from_pretrained(args.model_dir).to(device)
    processor = SamProcessor.from_pretrained(args.model_dir)
    model.eval()

    dataset = MedSamInferenceDataset(records_to_process)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    processed_count = 0
    total = len(records_to_process)

    for batch in dataloader:
        inputs = processor(
            images=batch["images"],
            input_boxes=[[box] for box in batch["boxes"]],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"],
                input_boxes=inputs["input_boxes"],
                multimask_output=False,
            )

        pred_masks_batch = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        batch_results = []
        for i, record_idx in enumerate(batch["record_indices"]):
            r = records_to_process[record_idx]
            gt_mask = batch["gt_masks"][i]

            pred_mask = pred_masks_batch[i][0, 0].numpy().astype(np.uint8)
            if pred_mask.shape != gt_mask.shape:
                pred_img = Image.fromarray(pred_mask * 255).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                pred_mask = (np.array(pred_img) > 0).astype(np.uint8)

            fname = f"{r['pid']}__{r['modality']}__slice_{r['slice_index']:04d}_pred.png"
            pred_mask_path = str(pred_mask_dir / fname)
            Image.fromarray(pred_mask * 255).save(pred_mask_path)

            batch_results.append({
                "pid": r["pid"],
                "modality": r["modality"],
                "slice_index": r["slice_index"],
                "image_path": r["image_path"],
                "mask_image_path": r["mask_image_path"],
                "pred_mask_path": pred_mask_path,
            })

        with open(results_path, "a", encoding="utf-8") as f:
            for res_obj in batch_results:
                f.write(json.dumps(res_obj, ensure_ascii=False) + "\n")

        processed_count += len(batch["record_indices"])
        print(f"  [{processed_count}/{total}] slices inferred")


def aggregate_patient_level(results_path, patient_results_path):
    by_pid = defaultdict(list)
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_pid[r["pid"]].append(r)

    already_done = set()
    all_metrics = []
    if patient_results_path.exists():
        with open(patient_results_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                already_done.add(m["pid"])
                all_metrics.append(m)
        print(f"Found {len(already_done)} already-aggregated patients.")

    pids = sorted(by_pid.keys())
    print(f"Aggregating {len(pids)} patients into 3D volumes...")

    for pid in pids:
        if pid in already_done:
            continue

        slices = sorted(by_pid[pid], key=lambda r: r["slice_index"])

        pred_stack = []
        gt_stack = []
        for r in slices:
            gt = (np.array(Image.open(r["mask_image_path"]).convert("L")) > 0).astype(np.uint8)
            pred = (np.array(Image.open(r["pred_mask_path"]).convert("L")) > 0).astype(np.uint8)
            gt_stack.append(gt)
            pred_stack.append(pred)

        pred_volume = np.stack(pred_stack, axis=0)  # (D, H, W)
        gt_volume = np.stack(gt_stack, axis=0)

        iou, dice, precision, recall, f1, hd95 = mask_metrics(pred_volume, gt_volume)

        metrics = {
            "pid": pid,
            "modality": slices[0]["modality"],
            "n_slices": len(slices),
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hd95": hd95,
        }
        all_metrics.append(metrics)

        with open(patient_results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        print(f"  {pid:20s} | n_slices={len(slices):3d} | Dice={dice:.4f} | IoU={iou:.4f} | HD95={hd95:.1f}")

    return all_metrics


def main():
    args = parse_args()
    jsonl_path = Path(args.dataset_dir) / f"{args.split}.jsonl"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_results_path = out_dir / f"results_{args.label}_slices.jsonl"
    patient_results_path = out_dir / f"results_{args.label}_patient.jsonl"
    pred_mask_dir = out_dir / "pred_masks"
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    processed_image_paths = set()
    if slice_results_path.exists():
        print(f"Found existing slice results file: {slice_results_path}")
        with open(slice_results_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    res = json.loads(line)
                    processed_image_paths.add(res["image_path"])
                except Exception:
                    continue
        print(f"Identified {len(processed_image_paths)} already processed slices.")

    all_records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("label") != args.label or r.get("modality") != args.modality:
                continue
            if not r.get("has_tumor"):
                continue
            bbox = r.get("bbox_xyxy_norm")
            if not bbox:
                continue
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < args.min_bbox_area_frac:
                continue
            if not os.path.exists(r.get("image_path", "")):
                continue
            if not r.get("mask_image_path") or not os.path.exists(r["mask_image_path"]):
                continue
            all_records.append(r)

    records_to_process = [r for r in all_records if r["image_path"] not in processed_image_paths]
    print(f"Total tumor-positive slices in set: {len(all_records)}")
    print(f"Remaining slices to infer: {len(records_to_process)}")

    if records_to_process:
        run_slice_inference(args, records_to_process, slice_results_path, pred_mask_dir)
    else:
        print("All slices already inferred, skipping to aggregation.")

    all_metrics = aggregate_patient_level(slice_results_path, patient_results_path)

    if not all_metrics:
        print("No patients processed. Exiting.")
        return

    keys = ["iou", "dice", "precision", "recall", "f1", "hd95"]
    stats = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

    print(f"\n Final Patient-Level Results ({args.label}, {args.modality}). ")
    print(f"Total patients evaluated:  {len(all_metrics)}")
    print(f"Mean IoU:                  {stats['iou']:.4f}")
    print(f"Mean Dice:                 {stats['dice']:.4f}")
    print(f"Mean Precision:            {stats['precision']:.4f}")
    print(f"Mean Recall (Sensitivity): {stats['recall']:.4f}")
    print(f"Mean F1-Score:             {stats['f1']:.4f}")
    print(f"Mean HD95 (px distance):   {stats['hd95']:.4f}")
    print(f"Per-patient results saved at: {patient_results_path}")

    summary_path = out_dir / f"summary_{args.label}_patient.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_patients": len(all_metrics),
            "label": args.label,
            "modality": args.modality,
            "split": args.split,
            **{f"mean_{k}": round(v, 6) for k, v in stats.items()},
        }, f, indent=2)
    print(f"Summary saved at: {summary_path}")


if __name__ == "__main__":
    main()