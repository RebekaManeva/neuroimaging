"""
Converts MOTUM NIfTI volumes + segmentation masks into a fine-tuning dataset.
Outputs:
  - PNG slices for each patient/modality
  - train.jsonl and val.jsonl with (image_path, bbox, label) per slice

Usage:
  python3 prepare_dataset.py \
    --motum-root /path/to/MOTUM-v.2.2 \
    --out-dir /path/to/dataset_out \
    --png-size 512 \
    --slice-axis 2 \
    --trim-frac 0.0
"""

import os, json, re, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib
from PIL import Image

MODALITIES = ["flair", "t1", "t1ce", "t2"]

LABEL_TO_MODALITY = {
    "ce_core":   "t1ce",   # contrast-enhancing core - best seen on T1CE
    "flair_abn": "flair",  # FLAIR abnormality - best seen on FLAIR
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--motum-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--png-size", type=int, default=512)
    p.add_argument("--slice-axis", type=int, default=2, choices=[0,1,2])
    p.add_argument("--trim-frac", type=float, default=0.0,
                   help="Set to 0.0 to keep all slices")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def map_path(p: str, motum_root: str) -> str:
    if os.path.exists(p):
        return p
    m = re.search(r"(MOTUM-v\.2\.2)(/.*)$", p)
    if m:
        cand = os.path.join(motum_root, m.group(2).lstrip("/"))
        if os.path.exists(cand):
            return cand
    return p

def norm_to_uint8(vol, p_low=1, p_high=99):
    vmin, vmax = np.percentile(vol, (p_low, p_high))
    vol = np.clip(vol, vmin, vmax).astype(np.float32)
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return (vol * 255).astype(np.uint8)

def get_slice(vol, axis, k):
    if axis == 2: return vol[:, :, k]
    if axis == 1: return vol[:, k, :]
    return vol[k, :, :]

# Converting binary mask slice to normalized [x1,y1,x2,y2]. Returns None if empty.
def mask_to_bbox_norm(mask_slice):
    if mask_slice.max() == 0:
        return None
    ys, xs = np.where(mask_slice > 0)
    h, w = mask_slice.shape
    return [
        float(xs.min()) / w,
        float(ys.min()) / h,
        float(xs.max()) / w,
        float(ys.max()) / h,
    ]

def get_tumor_type(pid: str) -> str:
    parts = pid.split("-")
    return parts[1] if len(parts) > 1 else "unknown"

def stratified_split(items_by_type: dict, seed: int):
    """
    Split rules:
      1 patient  -> train
      2 patients -> 1 train, 1 val
      3 patients -> 2 train, 1 val
      4 patients -> 2 train, 2 val
      5 patients -> 3 train, 2 val (treat as 6-: n-2 train, 2 val)
      6+         -> n-2 train, 2 val
    """
    rng = np.random.default_rng(seed)
    train_pids, val_pids = [], []

    for tumor_type, pids in sorted(items_by_type.items()):
        pids = list(pids)
        rng.shuffle(pids)
        n = len(pids)
        if n == 1:
            n_val = 0
        elif n == 2:
            n_val = 1
        elif n == 3:
            n_val = 1
        elif n == 4:
            n_val = 2
        else:  # 5+
            n_val = 2
        train_pids.extend(pids[n_val:])
        val_pids.extend(pids[:n_val])
        print(f"  {tumor_type}: {n} patients -> {n - n_val} train, {n_val} val")

    return train_pids, val_pids

def process_patient(pid, images_dict, labels_dict, motum_root,
                    out_dir, png_size, slice_axis, trim_frac):
    records = []  # listing of dicts, one per slice per modality per label

    # loading all 4 modality volumes
    vols = {}
    for mod in MODALITIES:
        raw_path = images_dict.get(mod)
        if not raw_path:
            print(f"  WARNING: missing modality {mod} for {pid}")
            continue
        path = map_path(raw_path, motum_root)
        if not os.path.exists(path):
            print(f"  WARNING: file not found: {path}")
            continue
        vol = nib.load(path).get_fdata().astype(np.float32)
        assert vol.ndim == 3, f"Expected 3D volume for {pid}/{mod}"
        vols[mod] = vol

    if not vols:
        return records

    # use flair (or first available) to determine slice indices
    ref_vol = vols.get("flair", next(iter(vols.values())))
    n_total = ref_vol.shape[slice_axis]
    lo = int(round(n_total * trim_frac))
    hi = int(round(n_total * (1.0 - trim_frac)))
    hi = max(lo + 1, hi)
    slice_indices = list(range(lo, hi))

    # load segmentation masks
    masks = {}
    for label_name, raw_path in labels_dict.items():
        path = map_path(raw_path, motum_root)
        if not os.path.exists(path):
            print(f"  WARNING: mask not found: {path}")
            continue
        masks[label_name] = nib.load(path).get_fdata().astype(np.float32)

    # export PNGs and build records
    pid_dir = Path(out_dir) / "images" / pid
    pid_dir.mkdir(parents=True, exist_ok=True)

    for mod, vol in vols.items():
        vol_u8 = norm_to_uint8(vol)

        for k in slice_indices:
            sl = get_slice(vol_u8, slice_axis, k)
            img = Image.fromarray(sl).convert("RGB")
            if png_size:
                img = img.resize((png_size, png_size), Image.BILINEAR)

            fname = f"{pid}__{mod}__slice_{k:04d}.png"
            img_path = str(pid_dir / fname)
            img.save(img_path)

            # get bbox from each mask
            for label_name, mask_vol in masks.items():
                mask_sl = get_slice(mask_vol, slice_axis, k)

                # resize mask to match png_size for consistent coordinates
                if png_size:
                    mask_img = Image.fromarray((mask_sl > 0).astype(np.uint8) * 255)
                    mask_img = mask_img.resize((png_size, png_size), Image.NEAREST)
                    mask_sl_resized = np.array(mask_img) > 0
                else:
                    mask_sl_resized = mask_sl > 0

                bbox = mask_to_bbox_norm(mask_sl_resized)
                has_tumor = bbox is not None

                records.append({
                    "pid": pid,
                    "modality": mod,
                    "label": label_name,
                    "slice_index": k,
                    "image_path": img_path,
                    "has_tumor": has_tumor,
                    "bbox_xyxy_norm": bbox if bbox else [0.0, 0.0, 0.0, 0.0],
                    "tumor_type": get_tumor_type(pid),
                })

    return records

def main():
    args = parse_args()
    np.random.seed(args.seed)

    motum_root = args.motum_root.rstrip("/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_path = Path(motum_root) / "data" / "dataset.json"
    assert ds_path.exists(), f"dataset.json not found: {ds_path}"
    ds = json.loads(ds_path.read_text())
    items = ds["training"]
    print(f"Total patients: {len(items)}")

    # group by tumor type for stratified split
    by_type = defaultdict(list)
    for it in items:
        pid = it["patient_id"]
        by_type[get_tumor_type(pid)].append(pid)

    print("\nStratified split:")
    train_pids, val_pids = stratified_split(by_type, args.seed)
    print(f"\nTotal -> {len(train_pids)} train, {len(val_pids)} val")

    # save split info
    split_info = {"train": sorted(train_pids), "val": sorted(val_pids)}
    (out_dir / "split.json").write_text(json.dumps(split_info, indent=2))

    # build pid -> item lookup
    pid_to_item = {it["patient_id"]: it for it in items}

    # process all patients
    train_records, val_records = [], []

    all_pids = [(pid, "train") for pid in train_pids] + [(pid, "val") for pid in val_pids]
    for i, (pid, split) in enumerate(all_pids, 1):
        it = pid_to_item[pid]
        print(f"\n[{i}/{len(all_pids)}] {pid} ({split})")
        recs = process_patient(
            pid=pid,
            images_dict=it.get("images", {}),
            labels_dict=it.get("labels", {}),
            motum_root=motum_root,
            out_dir=str(out_dir),
            png_size=args.png_size,
            slice_axis=args.slice_axis,
            trim_frac=args.trim_frac,
        )
        print(f"  -> {len(recs)} records ({sum(1 for r in recs if r['has_tumor'])} with tumor)")
        if split == "train":
            train_records.extend(recs)
        else:
            val_records.extend(recs)

    # writing JSONL files
    for split_name, records in [("train", train_records), ("val", val_records)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        n_tumor = sum(1 for r in records if r["has_tumor"])
        print(f"\n{split_name}.jsonl: {len(records)} records, {n_tumor} with tumor ({100*n_tumor//max(1,len(records))}%)")

    print(f"\nDone. Dataset written to: {out_dir}")

if __name__ == "__main__":
    main()
