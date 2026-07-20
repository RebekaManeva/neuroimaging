import os
import json
import re
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.spatial.distance import directed_hausdorff

for _attr in ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"]:
    if not hasattr(torch, _attr):
        setattr(torch, _attr, torch.float32)

try:
    import transformers.masking_utils as _mu
    for _fn_name in ("create_sliding_window_causal_mask", "create_causal_mask"):
        if hasattr(_mu, _fn_name):
            _orig = getattr(_mu, _fn_name)
            def _make_patched(orig):
                def _patched(*args, **kwargs):
                    kwargs.pop("or_mask_function", None)
                    kwargs.pop("and_mask_function", None)
                    return orig(*args, **kwargs)
                return _patched
            setattr(_mu, _fn_name, _make_patched(_orig))
except ModuleNotFoundError:
    pass

from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--base-model-id", default="google/medgemma-1.5-4b-it")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="whole_tumor", choices=["whole_tumor"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--max-image-side", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


LABEL_REGION_DESC = {
    "whole_tumor": "whole tumor (WT)"}

LABEL_TO_MODALITY = {
    "whole_tumor": "flair"}

def make_prompt(modality: str, label: str) -> str:
    seq_map = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1C+", "t2": "T2"}
    seq = seq_map.get(modality, "MRI")
    region = LABEL_REGION_DESC.get(label, label)
    return f"""You are an expert neuroradiologist analyzing a brain MRI slice ({seq} sequence).
Locate the {region} in this image and return a tight bounding box around it.

Return ONLY a valid JSON object inside <JSON> and </JSON> tags.

<JSON>
{{
  "has_tumor": true or false,
  "bbox_xyxy_norm": [x1, y1, x2, y2],
  "confidence": 0.0-1.0
}}
</JSON>

Rules:
- bbox_xyxy_norm: floats in [0,1], format [x1, y1, x2, y2] top-left to bottom-right
- has_tumor: set to true if a tumor is visible, false if no tumor is visible
- confidence: your certainty that the box covers the lesion (0.0-1.0)
- If no lesion is visible in this slice, return bbox_xyxy_norm=[] and has_tumor=false
"""

def extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"<JSON>\s*(\{[\s\S]*?\})\s*</JSON>", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"has_tumor": False, "bbox_xyxy_norm": [], "confidence": 0.0}


def bbox_to_mask(bbox_norm, shape=(512, 512)):
    mask = np.zeros(shape, dtype=np.uint8)
    if not bbox_norm or len(bbox_norm) == 0:
        return mask
    h, w = shape
    x1 = int(bbox_norm[0] * w)
    y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w)
    y2 = int(bbox_norm[3] * h)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    mask[y1:y2, x1:x2] = 1
    return mask


def compute_advanced_metrics(pred_bbox, gt_bbox, shape=(512, 512)):
    mask_pred = bbox_to_mask(pred_bbox, shape)
    mask_gt   = bbox_to_mask(gt_bbox,   shape)

    inter = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()

    iou = inter / union if union > 0 else (1.0 if mask_pred.sum() == 0 and mask_gt.sum() == 0 else 0.0)
    dice = (2 * inter) / (mask_pred.sum() + mask_gt.sum()) if (mask_pred.sum() + mask_gt.sum()) > 0 else (
        1.0 if mask_pred.sum() == 0 and mask_gt.sum() == 0 else 0.0)

    precision = inter / mask_pred.sum() if mask_pred.sum() > 0 else (1.0 if mask_gt.sum() == 0 else 0.0)
    recall    = inter / mask_gt.sum()   if mask_gt.sum()   > 0 else (1.0 if mask_pred.sum() == 0 else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if mask_pred.sum() == 0 and mask_gt.sum() == 0:
        hd95 = 0.0
    elif mask_pred.sum() == 0 or mask_gt.sum() == 0:
        hd95 = np.sqrt(shape[0] ** 2 + shape[1] ** 2)
    else:
        coords_pred = np.argwhere(mask_pred)
        coords_gt   = np.argwhere(mask_gt)
        d1 = directed_hausdorff(coords_pred, coords_gt)[0]
        d2 = directed_hausdorff(coords_gt, coords_pred)[0]
        hd95 = max(d1, d2)

    return iou, dice, precision, recall, f1, hd95


class MedicalInferenceDataset(Dataset):
    def __init__(self, records, label, max_image_side):
        self.records = records
        self.label = label
        self.max_image_side = max_image_side

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > self.max_image_side:
            s = self.max_image_side / m
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        prompt = make_prompt(r["modality"], self.label)
        return {"image": img, "prompt": prompt, "record_idx": idx}


def collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "record_indices": [item["record_idx"] for item in batch]
    }


def main():
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    args = parse_args()

    target_modality = LABEL_TO_MODALITY[args.label]
    jsonl_path = Path(args.dataset_dir) / f"{args.split}.jsonl"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_{args.label}.jsonl"

    processed_image_paths = set()
    historical_metrics = {
        "ious": [], "dices": [], "precisions": [], "recalls": [], "f1s": [], "hd95s": []
    }

    if results_path.exists():
        print(f"Found existing results file: {results_path}")
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line.strip())
                        processed_image_paths.add(res["image_path"])
                        historical_metrics["ious"].append(res["iou"])
                        historical_metrics["dices"].append(res["dice"])
                        historical_metrics["precisions"].append(res["precision"])
                        historical_metrics["recalls"].append(res["recall"])
                        historical_metrics["f1s"].append(res["f1"])
                        historical_metrics["hd95s"].append(res["hd95"])
                    except Exception:
                        continue
        print(f"Identified {len(processed_image_paths)} already processed records.")

    all_records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("label") != args.label or r.get("modality") != target_modality:
                continue
            if not os.path.exists(r["image_path"]):
                continue
            if r.get("has_tumor") and r.get("bbox_xyxy_norm") is None:
                continue
            all_records.append(r)

    records_to_process = [r for r in all_records if r["image_path"] not in processed_image_paths]

    print(f"Total records in set: {len(all_records)}")
    print(f"Remaining records to process: {len(records_to_process)}")

    if len(records_to_process) == 0:
        print("All records have been processed already! Exiting script.")
        return

    print(f"Loading base model in 4-bit mode: {args.base_model_id}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    config = AutoConfig.from_pretrained(args.base_model_id, token=hf_token)
    config.use_sliding_window = False
    config._attn_implementation = "eager"
    if hasattr(config, "sliding_window"):
        config.sliding_window = None

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token,
    )

    print(f"Loading LoRA adapter from: {args.model_dir}")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.base_model_id, token=hf_token)

    dataset = MedicalInferenceDataset(records_to_process, args.label, args.max_image_side)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    all_ious       = historical_metrics["ious"]
    all_dices      = historical_metrics["dices"]
    all_precisions = historical_metrics["precisions"]
    all_recalls    = historical_metrics["recalls"]
    all_f1s        = historical_metrics["f1s"]
    all_hd95s      = historical_metrics["hd95s"]

    processed_count = len(processed_image_paths)

    ious_with_tumor = []
    dices_with_tumor = []
    false_positives_count = 0
    true_negatives_count = 0

    for batch in dataloader:
        batch_prompts  = batch["prompts"]
        batch_images   = batch["images"]
        batch_indices  = batch["record_indices"]

        individual_inputs = []
        for p, img in zip(batch_prompts, batch_images):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            single_input = processor(text=prompt_text, images=img, return_tensors="pt", padding=True, truncation=True)
            individual_inputs.append(single_input)

        inputs = {}
        for key in individual_inputs[0].keys():
            inputs[key] = torch.cat([item[key] for item in individual_inputs], dim=0).to("cuda")

        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
                stop_strings=["</JSON>"],
                tokenizer=processor.tokenizer
            )

        input_len = inputs["input_ids"].shape[1]
        batch_results_to_write = []

        for idx_in_batch, record_idx in enumerate(batch_indices):
            r = records_to_process[record_idx]
            generated_tokens = gen[idx_in_batch][input_len:]
            resp = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"DEBUG: Model Output for {r['pid']}: {resp}")

            pred = extract_json(resp)
            pred_bbox      = pred.get("bbox_xyxy_norm", [])
            gt_bbox        = r.get("bbox_xyxy_norm", [])
            has_tumor_gt   = r.get("has_tumor", False)
            pred_has_tumor = pred.get("has_tumor", False)

            iou, dice, precision, recall, f1, hd95 = compute_advanced_metrics(pred_bbox, gt_bbox)

            if has_tumor_gt:
                ious_with_tumor.append(iou)
                dices_with_tumor.append(dice)
            else:
                if pred_has_tumor or  len(pred_bbox) > 0:
                    false_positives_count += 1
                else:
                    true_negatives_count += 1

            all_ious.append(iou)
            all_dices.append(dice)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_hd95s.append(hd95)

            res_obj = {
                **r,
                "pred_bbox": pred_bbox,
                "pred_has_tumor": pred_has_tumor,
                "pred_confidence": pred.get("confidence", 0.0),
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hd95": hd95,
                "raw_output": resp,
            }
            batch_results_to_write.append(res_obj)

        with open(results_path, "a", encoding="utf-8") as f:
            for res_obj in batch_results_to_write:
                f.write(json.dumps(res_obj, ensure_ascii=False) + "\n")

        processed_count += len(batch_indices)
        print(f"  [{processed_count}/{len(all_records)}] Mean IoU={np.mean(all_ious):.4f} | Mean Dice={np.mean(all_dices):.4f}")

    print(f"\n=== Final Statistical Results ({args.label}) ===")
    print(f"Total slices evaluated:                          {len(all_ious)}")
    print(f"Mean IoU (all slices):                           {np.mean(all_ious):.4f}")
    print(f"Mean Dice (all slices):                          {np.mean(all_dices):.4f}")
    print(f"Mean IoU (tumor slices only):                    {np.mean(ious_with_tumor):.4f if ious_with_tumor else 0.0:.4f}")
    print(f"Mean Dice (tumor slices only):                   {np.mean(dices_with_tumor):.4f if dices_with_tumor else 0.0:.4f}")
    print(f"Mean Precision:                                  {np.mean(all_precisions):.4f}")
    print(f"Mean Recall (Sensitivity):                       {np.mean(all_recalls):.4f}")
    print(f"Mean F1-Score:                                   {np.mean(all_f1s):.4f}")
    print(f"Mean HD95 (px distance):                         {np.mean(all_hd95s):.4f}")
    print(f"--- Healthy Slices Detection ---")
    print(f"True Negatives (correctly healthy):              {true_negatives_count}")
    print(f"False Positives (hallucinated tumors):           {false_positives_count}")
    print(f"Results saved at: {results_path}")


if __name__ == "__main__":
    main()