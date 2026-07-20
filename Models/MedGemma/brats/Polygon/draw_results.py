import json
import os
from pathlib import Path
from PIL import Image, ImageDraw

JSONL_PATH = "/home/hpc/users/ml_models/elena.nikolovska/Medgemma_Lora/output_polygon/results_whole_tumor.jsonl"
OUT_DIR = "/home/hpc/users/ml_models/elena.nikolovska/Medgemma_Lora/output_polygon/annotated_images"


def draw_polygons(img_path, pred_polygon, gt_polygon, out_path, iou, dice):
    if not os.path.exists(img_path):
        return

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)

    def norm_to_px_poly(poly):
        if not poly or len(poly) < 3:
            return None
        return [(int(p[0] * w), int(p[1] * h)) for p in poly]

    gt_px = norm_to_px_poly(gt_polygon)
    if gt_px:
        d.polygon(gt_px, outline=(0, 255, 0), width=3)

    pred_px = norm_to_px_poly(pred_polygon)
    if pred_px:
        d.polygon(pred_px, outline=(255, 0, 0), width=3)

    d.text((5, 5), f"IoU={iou:.2f} Dice={dice:.2f}", fill=(255, 255, 0))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    if not os.path.exists(JSONL_PATH):
        print(f"Error: File not found {JSONL_PATH}.")
        return

    print(f"Loading results from {JSONL_PATH}.")
    records = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    print(f"{len(records)} records found. Starting to draw polygons.")

    for i, r in enumerate(records, 1):
        pred_polygon = r.get("pred_polygon", [])
        gt_polygon = r.get("polygon_norm", [])
        iou = r.get("iou", 0.0)
        dice = r.get("dice", 0.0)

        out_img_path = os.path.join(OUT_DIR, f"{r['pid']}__{r['modality']}__slice_{r['slice_index']:04d}.png")

        draw_polygons(r["image_path"], pred_polygon, gt_polygon, out_img_path, iou, dice)

        if i % 200 == 0 or i == len(records):
            print(f" [{i}/{len(records)}] images drawn.")

    print(f"Done! All visualizations are saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()