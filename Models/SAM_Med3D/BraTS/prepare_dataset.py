import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


MODALITY_SUFFIX = {
    "flair": "-t2f.nii.gz",
    "t1":    "-t1n.nii.gz",
    "t1ce":  "-t1c.nii.gz",
    "t2":    "-t2w.nii.gz",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Preparing BraTS2023 for fine-tuning with SAM-Med3D"
    )
    p.add_argument("--brats-root", required=True,
                   help="BraTS2023 root directory (contains imagesTr/)")
    p.add_argument("--out-dir", required=True,
                   help="Where to save the organized dataset")
    p.add_argument("--modality", default="flair",
                   choices=list(MODALITY_SUFFIX.keys()),
                   help="Which MRI modality to use (default: flair)")
    p.add_argument("--val-frac",  type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symlink", action="store_true",
                   help="Use symlinks instead of copying (saves disk space)")
    return p.parse_args()


def split_patients(pids, val_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    pids = sorted(pids)
    rng.shuffle(pids)
    n = len(pids)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    return (
        pids[n_test + n_val:],
        pids[n_test:n_test + n_val],
        pids[:n_test],
    )


def link_or_copy(src: Path, dst: Path, use_symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    brats_root = Path(args.brats_root)
    out_dir    = Path(args.out_dir)
    images_dir = brats_root / "imagesTr"

    assert images_dir.exists(), f"Does not exist: {images_dir}"

    img_suffix   = MODALITY_SUFFIX[args.modality]
    mask_suffix  = "-wt_mask.nii.gz"

    cases  = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    valid  = []
    skipped = 0
    for case in cases:
        pid = case.name
        img_path  = case / f"{pid}{img_suffix}"
        mask_path = case / f"{pid}{mask_suffix}"
        if not img_path.exists():
            print(f"  SKIP {pid}: missing {img_path.name}")
            skipped += 1
            continue
        if not mask_path.exists():
            print(f"  SKIP {pid}: missing {mask_path.name}.")
            skipped += 1
            continue
        valid.append(pid)

    print(f"\nValid patients: {len(valid)}  |  Skipped: {skipped}")
    if not valid:
        raise SystemExit("No valid patients found.")

    train_pids, val_pids, test_pids = split_patients(
        valid, args.val_frac, args.test_frac, args.seed
    )
    print(f"Train: {len(train_pids)}  |  Val: {len(val_pids)}  |  "
          f"Test: {len(test_pids)}")

    split_info = {
        "train": sorted(train_pids),
        "val":   sorted(val_pids),
        "test":  sorted(test_pids),
    }
    (out_dir / "split.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(json.dumps(split_info, indent=2))

    dataset_name = f"brats_wt_{args.modality}"
    split_base   = out_dir / dataset_name

    split_to_pids = {
        "train": train_pids,
        "val":   val_pids,
        "test":  test_pids,
    }
    for split_name, pids in split_to_pids.items():
        img_out_dir  = split_base / split_name / "imagesTr"
        lbl_out_dir  = split_base / split_name / "labelsTr"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for pid in pids:
            src_img  = images_dir / pid / f"{pid}{img_suffix}"
            src_mask = images_dir / pid / f"{pid}{mask_suffix}"
            dst_img  = img_out_dir / f"{pid}.nii.gz"
            dst_mask = lbl_out_dir / f"{pid}.nii.gz"

            link_or_copy(src_img,  dst_img,  args.symlink)
            link_or_copy(src_mask, dst_mask, args.symlink)

        print(f"  [{split_name}] {len(pids)} patients ? {split_base / split_name}")

    train_path = str((split_base / "train").resolve())
    val_path   = str((split_base / "val").resolve())
    test_path  = str((split_base / "test").resolve())

    data_paths_content = f'''
# Paths to directories containing imagesTr/ and labelsTr/ subfolders.
# Dataset_Union_ALL scans them recursively.

img_datas = [
    "{train_path}",
]

# For validation (optional - used in some forks):
val_datas = [
    "{val_path}",
]

# For testing (used in validation.py):
test_datas = [
    "{test_path}",
]
'''

    utils_dir = out_dir / "utils"
    utils_dir.mkdir(exist_ok=True)
    data_paths_file = utils_dir / "data_paths.py"
    data_paths_file.write_text(data_paths_content)

if __name__ == "__main__":
    main()