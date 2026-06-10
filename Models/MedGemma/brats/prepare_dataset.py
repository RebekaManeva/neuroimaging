import os
import json
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from PIL import Image

MODALITIES = ["flair", "t1", "t1ce", "t2"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--brats-root", required=True,
                   help="Root of BraTS2023 dataset (contains imagesTr/ and data/)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--png-size", type=int, default=512)
    p.add_argument("--slice-axis", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--trim-frac", type=float, default=0.0,
                   help="Fraction of slices to trim from each end (0.0 = keep all)")
    p.add_argument("--val-frac", type=float, default=0.15, help="Fraction of validation data")
    p.add_argument("--test-frac", type=float, default=0.15, help="Fraction of test data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def norm_to_uint8(vol, p_low=1, p_high=99):
    vmin, vmax = np.percentile(vol, (p_low, p_high))
    vol = np.clip(vol, vmin, vmax).astype(np.float32)
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return (vol * 255).astype(np.uint8)


def get_slice(vol, axis, k):
    if axis == 2:
        return vol[:, :, k]
    if axis == 1:
        return vol[:, k, :]
    return vol[k, :, :]


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


def mask_to_polygon_norm(mask_slice):
    mask_u8 = (mask_slice > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    h, w = mask_slice.shape
    points = [[round(float(p[0][0]) / w, 4), round(float(p[0][1]) / h, 4)] for p in approx]
    return points if len(points) >= 3 else None


def train_val_test_split(pids, seed, val_frac=0.15, test_frac=0.15):
    rng = np.random.default_rng(seed)
    pids = sorted(list(pids)) 
    rng.shuffle(pids)

    n = len(pids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_pids = pids[:n_test]
    val_pids = pids[n_test: n_test + n_val]
    train_pids = pids[n_test + n_val:]

    print(f"Total patients: {n}")
    print(f"Train: {len(train_pids)}")
    print(f"Val:   {len(val_pids)}")
    print(f"Test:  {len(test_pids)}")
    return train_pids, val_pids, test_pids


def process_patient(pid, images_dict, labels_dict, brats_root,
                    out_dir, png_size, slice_axis, trim_frac):
    records = []
    brats_root = str(brats_root)

    vols = {}
    for mod in MODALITIES:
        rel_path = images_dict.get(mod)
        if not rel_path:
            print(f"  WARNING: missing modality {mod} for {pid}")
            continue
        path = os.path.join(brats_root, rel_path)
        if not os.path.exists(path):
            print(f"  WARNING: file not found: {path}")
            continue
        vol = nib.load(path).get_fdata().astype(np.float32)
        assert vol.ndim == 3, f"Expected 3D volume for {pid}/{mod}"
        vols[mod] = vol

    if not vols:
        return records

    ref_vol = vols.get("flair", next(iter(vols.values())))
    n_total = ref_vol.shape[slice_axis]
    lo = int(round(n_total * trim_frac))
    hi = int(round(n_total * (1.0 - trim_frac)))
    hi = max(lo + 1, hi)
    slice_indices = list(range(lo, hi))

    masks = {}
    for label_name, rel_path in labels_dict.items():
        path = os.path.join(brats_root, rel_path)
        if not os.path.exists(path):
            print(f"  WARNING: mask not found: {path}")
            continue
        masks[label_name] = nib.load(path).get_fdata().astype(np.float32)

    pid_dir = Path(out_dir) / "images" / pid
    pid_dir.mkdir(parents=True, exist_ok=True)

    for mod, vol in vols.items():
        vol_u8 = norm_to_uint8(vol)

        for k in slice_indices:
            sl = get_slice(vol_u8, slice_axis, k)
            img = Image.fromarray(sl).convert("RGB")
            if png_size:
                img = img.resize((png_size, png_size), Image.Resampling.BILINEAR)

            fname = f"{pid}__{mod}__slice_{k:04d}.png"
            img_path = str(pid_dir / fname)
            img.save(img_path)

            for label_name, mask_vol in masks.items():
                mask_sl = get_slice(mask_vol, slice_axis, k)

                if png_size:
                    mask_img = Image.fromarray((mask_sl > 0).astype(np.uint8) * 255)
                    mask_img = mask_img.resize((png_size, png_size), Image.NEAREST)
                    mask_sl_resized = np.array(mask_img) > 0
                else:
                    mask_sl_resized = mask_sl > 0

                bbox = mask_to_bbox_norm(mask_sl_resized)
                has_tumor = bbox is not None

                polygon = mask_to_polygon_norm(mask_sl_resized) if has_tumor else None

                records.append({
                    "pid": pid,
                    "modality": mod,
                    "label": label_name,
                    "slice_axis": slice_axis,
                    "slice_index": k,
                    "image_path": img_path,
                    "has_tumor": has_tumor,
                    "bbox_xyxy_norm": bbox,
                    "polygon_norm": polygon,
                    "mask_path": os.path.join(brats_root, rel_path),
                })

    return records


def main():
    args = parse_args()
    brats_root = Path(args.brats_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_path = brats_root / "data" / "dataset.json"
    assert ds_path.exists(), (
        f"dataset.json not found: {ds_path}\n"
        f"Run generate_dataset_json.py first."
    )

    ds = json.loads(ds_path.read_text())
    items = ds["training"]
    print(f"Total patients in dataset.json: {len(items)}")

    all_pids = [it["patient_id"] for it in items]
    print("\nTrain/Val/Test split:")
    train_pids, val_pids, test_pids = train_val_test_split(
        all_pids, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac
    )

    split_info = {
        "train": sorted(train_pids),
        "val": sorted(val_pids),
        "test": sorted(test_pids)
    }
    (out_dir / "split.json").write_text(json.dumps(split_info, indent=2))

    pid_to_item = {it["patient_id"]: it for it in items}

    split_records = {"train": [], "val": [], "test": []}

    all_pids_split = (
            [(pid, "train") for pid in train_pids] +
            [(pid, "val") for pid in val_pids] +
            [(pid, "test") for pid in test_pids]
    )

    for i, (pid, split) in enumerate(all_pids_split, 1):
        it = pid_to_item[pid]
        print(f"\n[{i}/{len(all_pids_split)}] {pid} ({split})")

        recs = process_patient(
            pid=pid,
            images_dict=it.get("images", {}),
            labels_dict=it.get("labels", {}),
            brats_root=brats_root,
            out_dir=str(out_dir),
            png_size=args.png_size,
            slice_axis=args.slice_axis,
            trim_frac=args.trim_frac,
        )

        n_tumor = sum(1 for r in recs if r["has_tumor"])
        print(f"  -> {len(recs)} records ({n_tumor} with tumor)")
        split_records[split].extend(recs)

    for split_name in ["train", "val", "test"]:
        records = split_records[split_name]
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        n_tumor = sum(1 for r in records if r["has_tumor"])
        print(
            f"\n{split_name}.jsonl: {len(records)} records, "
            f"{n_tumor} with tumor "
            f"({100 * n_tumor // max(1, len(records))}%)"
        )

    print(f"\nDone. Dataset written to: {out_dir}")


if __name__ == "__main__":
    main()