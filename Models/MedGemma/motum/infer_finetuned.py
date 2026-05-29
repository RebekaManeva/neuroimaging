"""
Runs inference with fine-tuned MedGemma on val set and evaluates bbox predictions.
Outputs annotated images and a results JSONL with IoU scores.

Usage:
  python3 infer_finetuned.py \
    --dataset-dir /path/to/dataset_out \
    --model-dir /path/to/finetuned_model/best_ce_core \
    --base-model-id google/medgemma-1.5-4b-it \
    --out-dir /path/to/infer_out \
    --label ce_core
"""

import os, json, re, argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model-dir", required=True,
                   help="Path to saved LoRA adapter (best_ce_core or best_flair_abn)")
    p.add_argument("--base-model-id", default="google/medgemma-1.5-4b-it")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="ce_core", choices=["ce_core", "flair_abn"])
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-image-side", type=int, default=512)
    return p.parse_args()

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

def extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"<JSON>\s*(\{[\s\S]*?\})\s*</JSON>", text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return {"has_tumor": False, "bbox_xyxy_norm": [0,0,0,0], "confidence": 0.0}

# computing IoU between two [x1,y1,x2,y2] boxes.
def compute_iou(box_a, box_b):
    ax1,ay1,ax2,ay2 = box_a
    bx1,by1,bx2,by2 = box_b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def draw_boxes(img_path, pred_bbox, gt_bbox, out_path, iou):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)

    def norm_to_px(b):
        return [int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)]

    # ground truth in green
    if gt_bbox and gt_bbox != [0,0,0,0]:
        gx1,gy1,gx2,gy2 = norm_to_px(gt_bbox)
        d.rectangle([gx1,gy1,gx2,gy2], outline=(0,255,0), width=3)

    # prediction in red
    if pred_bbox and pred_bbox != [0,0,0,0]:
        px1,py1,px2,py2 = norm_to_px(pred_bbox)
        d.rectangle([px1,py1,px2,py2], outline=(255,0,0), width=3)

    # IoU label
    d.text((5, 5), f"IoU={iou:.3f}", fill=(255,255,0))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def main():
    args = parse_args()

    # load records
    jsonl_path = Path(args.dataset_dir) / f"{args.split}.jsonl"
    records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("label") != args.label:
                continue
            if not os.path.exists(r["image_path"]):
                continue
            records.append(r)
    print(f"Running inference on {len(records)} records ({args.split}, label={args.label})")

    # loading model
    print(f"Loading base model: {args.base_model_id}")
    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id, torch_dtype=dtype, device_map="auto"
    )
    print(f"Loading LoRA adapter: {args.model_dir}")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()
    print(f"Model device: {model.device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    ious = []

    for i, r in enumerate(records, 1):
        img = Image.open(r["image_path"]).convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > args.max_image_side:
            s = args.max_image_side / m
            img = img.resize((max(1,int(w*s)), max(1,int(h*s))), Image.BILINEAR)

        prompt = make_prompt(r["modality"], args.label)
        content = [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            tokenize=True, return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            gen = model.generate(
                **inputs, do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        resp = processor.post_process_image_text_to_text(gen, skip_special_tokens=True)[0]
        decoded_inp = processor.post_process_image_text_to_text(
            inputs["input_ids"], skip_special_tokens=True)[0]
        pos = resp.find(decoded_inp)
        if 0 <= pos <= 2:
            resp = resp[pos + len(decoded_inp):].strip()

        pred = extract_json(resp)
        pred_bbox = pred.get("bbox_xyxy_norm", [0,0,0,0])
        gt_bbox = r["bbox_xyxy_norm"]

        iou = 0.0
        if r["has_tumor"] and pred.get("has_tumor", False):
            iou = compute_iou(pred_bbox, gt_bbox)
        ious.append(iou)

        # saving annotated image
        out_img = str(out_dir / f"{r['pid']}__{r['modality']}__slice_{r['slice_index']:04d}.png")
        draw_boxes(r["image_path"], pred_bbox, gt_bbox, out_img, iou)

        result = {**r, "pred_bbox": pred_bbox, "pred_has_tumor": pred.get("has_tumor", False),
                  "pred_confidence": pred.get("confidence", 0.0), "iou": iou, "raw_output": resp}
        results.append(result)

        if i % 20 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}] mean_IoU={np.mean(ious):.4f}")

    # saving results
    results_path = out_dir / f"results_{args.label}.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # summary stats
    ious_arr = np.array(ious)
    tumor_mask = np.array([r["has_tumor"] for r in records])
    print(f"\n=== Results ({args.label}) ===")
    print(f"Total slices: {len(results)}")
    print(f"Slices with tumor (GT): {tumor_mask.sum()}")
    print(f"Mean IoU (all):         {ious_arr.mean():.4f}")
    print(f"Mean IoU (tumor only):  {ious_arr[tumor_mask].mean():.4f}" if tumor_mask.sum() > 0 else "")
    print(f"IoU >= 0.5:             {(ious_arr >= 0.5).sum()} / {len(ious_arr)}")
    print(f"IoU >= 0.3:             {(ious_arr >= 0.3).sum()} / {len(ious_arr)}")
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
