"""
Patient-level evaluation of fine-tuned MedGemma bounding box predictions.
For each patient, takes the slice with the highest IoU as the best prediction.
Produces summary statistics, per-tumor-type breakdown, and visualization.

Usage:
  python3 analyze_results.py \
    --results-jsonl /path/to/your.jsonl \
    --out-dir /path/to/ \
    --label ce_core
"""

import json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="ce_core")
    return p.parse_args()

def get_tumor_type(pid: str) -> str:
    parts = pid.split("-")
    return parts[1] if len(parts) > 1 else "unknown"

def bbox_to_mask(box, size=1000):
    mask = np.zeros((size, size), dtype=bool)

    if box is None or len(box) != 4:
        return mask

    x1, y1, x2, y2 = box

    # if normalized [0,1], scale to mask size
    if max(x1, y1, x2, y2) <= 1.0:
        x1, y1, x2, y2 = [int(v * size) for v in box]
    else:
        x1, y1, x2, y2 = [int(v) for v in box]

    x1, x2 = sorted([max(0, x1), min(size, x2)])
    y1, y2 = sorted([max(0, y1), min(size, y2)])

    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True

    return mask


def bbox_metrics(pred_box, gt_box):
    pred = bbox_to_mask(pred_box)
    gt = bbox_to_mask(gt_box)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    f1 = dice

    return dice, precision, recall, f1

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load all results
    records = []
    with open(args.results_jsonl) as f:
        for line in f:
            r = json.loads(line.strip())
            records.append(r)
    print(f"Loaded {len(records)} slice-level records")

    # patient-level: best IoU per patient
    # group by pid, take slice with max IoU
    by_patient = defaultdict(list)
    for r in records:
        by_patient[r["pid"]].append(r)

    patient_results = []
    for pid, slices in by_patient.items():
        # only consider slices where GT has tumor
        tumor_slices = [s for s in slices if s.get("has_tumor", False)]
        if not tumor_slices:
            # no tumor in any GT slice for this patient — skip
            continue

        best = max(tumor_slices, key=lambda x: float(x.get("iou", 0.0)))
        dice, precision, recall, f1 = bbox_metrics(
            best.get("pred_bbox", [0,0,0,0]),
            best.get("bbox_xyxy_norm", [0,0,0,0])
        )

        patient_results.append({
            "pid": pid,
            "tumor_type": get_tumor_type(pid),
            "best_iou": float(best.get("iou", 0.0)),
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_slice": int(best.get("slice_index", -1)),
            "best_modality": best.get("modality", ""),
            "pred_bbox": best.get("pred_bbox", [0,0,0,0]),
            "gt_bbox": best.get("bbox_xyxy_norm", [0,0,0,0]),
            "n_tumor_slices": len(tumor_slices),
            "n_total_slices": len(slices),
        })

    patient_results.sort(key=lambda x: x["best_iou"], reverse=True)
    n_patients = len(patient_results)
    ious = np.array([r["best_iou"] for r in patient_results])
    dices = np.array([r["dice"] for r in patient_results])
    precisions = np.array([r["precision"] for r in patient_results])
    recalls = np.array([r["recall"] for r in patient_results])
    f1s = np.array([r["f1"] for r in patient_results])

    print(f"\nPatients with tumor ground truth: {n_patients}")
    print(f"Mean Dice:           {dices.mean():.4f}")
    print(f"Mean Precision:      {precisions.mean():.4f}")
    print(f"Mean Recall:         {recalls.mean():.4f}")
    print(f"Mean F1:             {f1s.mean():.4f}")

    print(f"\n{'='*50}")
    print(f"PATIENT-LEVEL RESULTS ({args.label})")
    print(f"{'='*50}")
    print(f"Patients evaluated:  {n_patients}")
    print(f"Mean IoU:            {ious.mean():.4f}")
    print(f"Median IoU:          {np.median(ious):.4f}")
    print(f"Std IoU:             {ious.std():.4f}")
    print(f"Min IoU:             {ious.min():.4f}")
    print(f"Max IoU:             {ious.max():.4f}")
    print(f"IoU >= 0.5:          {(ious >= 0.5).sum()} / {n_patients} ({100*(ious>=0.5).mean():.1f}%)")
    print(f"IoU >= 0.3:          {(ious >= 0.3).sum()} / {n_patients} ({100*(ious>=0.3).mean():.1f}%)")
    print(f"IoU >= 0.1:          {(ious >= 0.1).sum()} / {n_patients} ({100*(ious>=0.1).mean():.1f}%)")
    print(f"IoU == 0.0:          {(ious == 0.0).sum()} / {n_patients} ({100*(ious==0.0).mean():.1f}%)")

    # per tumor type breakdown
    print(f"\n{'─'*50}")
    print("PER TUMOR TYPE:")
    print(f"{'─'*50}")
    by_type = defaultdict(list)
    for r in patient_results:
        by_type[r["tumor_type"]].append(r["best_iou"])

    type_stats = []
    for ttype, type_ious in sorted(by_type.items()):
        arr = np.array(type_ious)
        type_stats.append({
            "tumor_type": ttype,
            "n": len(arr),
            "mean_iou": float(arr.mean()),
            "median_iou": float(np.median(arr)),
            "iou_gte_05": int((arr >= 0.5).sum()),
        })
        print(f"{ttype:20s} n={len(arr):2d}  mean={arr.mean():.3f}  "
              f"median={np.median(arr):.3f}  IoU>=0.5: {(arr>=0.5).sum()}/{len(arr)}")



    summary = {
        "label": args.label,
        "n_patients": n_patients,
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "std_iou": float(ious.std()),
        "iou_gte_05": int((ious >= 0.5).sum()),
        "iou_gte_03": int((ious >= 0.3).sum()),
        "iou_gte_01": int((ious >= 0.1).sum()),
        "iou_eq_00":  int((ious == 0.0).sum()),
        "per_tumor_type": type_stats,
        "per_patient": patient_results,
    }
    summary_path = out_dir / f"summary_{args.label}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")

    # plots

    # IoU distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ious, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(ious.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean={ious.mean():.3f}")
    ax.axvline(np.median(ious), color="orange", linestyle="--", linewidth=2, label=f"Median={np.median(ious):.3f}")
    ax.axvline(0.5, color="green", linestyle=":", linewidth=2, label="IoU=0.5 threshold")
    ax.set_xlabel("Best IoU per Patient", fontsize=13)
    ax.set_ylabel("Number of Patients", fontsize=13)
    ax.set_title(f"Patient-Level IoU Distribution\n{args.label} | n={n_patients}", fontsize=14)
    ax.legend()
    plt.tight_layout()
    hist_path = out_dir / f"iou_histogram_{args.label}.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Histogram saved to: {hist_path}")

    # Per tumor type bar chart
    type_names = [s["tumor_type"] for s in type_stats]
    type_means  = [s["mean_iou"]   for s in type_stats]
    type_ns     = [s["n"]          for s in type_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(type_names, type_means, color="steelblue", edgecolor="white")
    ax.axhline(0.5, color="green", linestyle=":", linewidth=2, label="IoU=0.5")
    ax.axhline(ious.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Overall mean={ious.mean():.3f}")
    for bar, n in zip(bars, type_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(0.7, max(type_means) + 0.15))
    ax.set_ylabel("Mean IoU (best slice per patient)", fontsize=12)
    ax.set_title(f"Mean IoU by Tumor Type\n{args.label}", fontsize=14)
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    bar_path = out_dir / f"iou_by_type_{args.label}.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Bar chart saved to: {bar_path}")

    # Sorted IoU per patient (waterfall)
    fig, ax = plt.subplots(figsize=(12, 5))
    sorted_ious = np.sort(ious)[::-1]
    colors = ["green" if v >= 0.5 else "orange" if v >= 0.3 else "red" for v in sorted_ious]
    ax.bar(range(len(sorted_ious)), sorted_ious, color=colors, edgecolor="none")
    ax.axhline(0.5, color="green", linestyle="--", linewidth=1.5, label="IoU=0.5")
    ax.axhline(0.3, color="orange", linestyle="--", linewidth=1.5, label="IoU=0.3")
    ax.set_xlabel("Patients (sorted by IoU)", fontsize=12)
    ax.set_ylabel("Best IoU", fontsize=12)
    ax.set_title(f"Per-Patient Best IoU (sorted)\n{args.label} | mean={ious.mean():.3f}", fontsize=14)
    ax.legend()
    plt.tight_layout()
    waterfall_path = out_dir / f"iou_waterfall_{args.label}.png"
    plt.savefig(waterfall_path, dpi=150)
    plt.close()
    print(f"Waterfall chart saved to: {waterfall_path}")

    print(f"\nAll outputs saved to: {out_dir}")



if __name__ == "__main__":
    main()