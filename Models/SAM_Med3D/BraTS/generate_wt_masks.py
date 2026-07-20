import argparse, os
import numpy as np
import nibabel as nib
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--brats-root", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    brats_root = Path(args.brats_root)
    images_dir = brats_root / "imagesTr"

    labels_dir = brats_root / "labelsTr"
    labels_dir.mkdir(exist_ok=True)

    cases = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    print(f"Found {len(cases)} patients", flush=True)

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.name}...", flush=True)
        seg_candidates = list(case.glob("*-seg.nii.gz"))
        if not seg_candidates:
            print(f"     SKIPPED: no seg file", flush=True)
            continue
        try:
            img = nib.load(seg_candidates[0])
            data = img.get_fdata().astype(np.uint8)
            whole_mask = (data > 0).astype(np.uint8)

            out_path = labels_dir / f"{case.name}.nii.gz"
            nib.save(nib.Nifti1Image(whole_mask, img.affine, img.header), out_path)

            print(f" OK: {int(whole_mask.sum())} vox → {out_path}", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()