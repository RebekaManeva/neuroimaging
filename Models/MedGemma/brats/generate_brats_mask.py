import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--brats-root", required=True,
                   help="Root folder of BraTS2023 (contains imagesTr/)")
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.brats_root) / "imagesTr"

    if not images_dir.exists():
        print(f"ERROR: Folder {images_dir} does not exist!", flush=True)
        return

    cases = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    print(f"Found {len(cases)} patients in {images_dir}", flush=True)

    if not cases:
        print("WARNING: No subdirectories found in imagesTr!", flush=True)

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Processing patient: {case.name}...", flush=True)
        seg_candidates = list(case.glob("*-seg.nii.gz"))

        if not seg_candidates:
            print(f"  --> SKIPPED: No '-seg.nii.gz' found in {case.name}", flush=True)
            continue

        try:
            seg_path = seg_candidates[0]
            img = nib.load(seg_path)
            data = img.get_fdata().astype(np.uint8)

            whole_mask = (data > 0).astype(np.uint8)
            whole_path = case / f"{case.name}-wt_mask.nii.gz"
            nib.save(nib.Nifti1Image(whole_mask, img.affine, img.header), whole_path)

            wt_vox = int(whole_mask.sum())
            print(f"  --> OK: whole_tumor={wt_vox} vox", flush=True)

        except Exception as e:
            print(f"  --> ERROR processing {case.name}: {e}", flush=True)

    print("\nDone. Check patient folders for new mask files.", flush=True)


if __name__ == "__main__":
    main()