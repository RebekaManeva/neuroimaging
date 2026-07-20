import argparse, json
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--brats-root", required=True,
                   help="Root folder of BraTS2023 (contains imagesTr/)")
    return p.parse_args()


def main():
    args = parse_args()
    brats_root = Path(args.brats_root)
    images_dir = brats_root / "imagesTr"
    assert images_dir.exists(), f"imagesTr not found: {images_dir}"

    data_dir = brats_root / "data"
    data_dir.mkdir(exist_ok=True)

    cases = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    print(f"Found {len(cases)} cases")

    training = []
    skipped = 0

    for case in cases:
        pid = case.name
        expected = {
            "flair": f"{pid}-t2f.nii.gz",
            "t1":    f"{pid}-t1n.nii.gz",
            "t1ce":  f"{pid}-t1c.nii.gz",
            "t2":    f"{pid}-t2w.nii.gz",
        }

        label_files = {
            "whole_tumor":   f"{pid}-wt_mask.nii.gz",
        }

        ok = True
        for key, fname in {**expected, **label_files}.items():
            fpath = case / fname
            if not fpath.exists():
                print(f"  WARNING: missing {fname} in {pid} - skipping patient")
                ok = False
                break
        if not ok:
            skipped += 1
            continue

        images = {k: f"imagesTr/{pid}/{v}" for k, v in expected.items()}
        labels = {k: f"imagesTr/{pid}/{v}" for k, v in label_files.items()}

        training.append({
            "patient_id": pid,
            "images": images,
            "labels": labels,
        })

    dataset = {"training": training}
    out_path = data_dir / "dataset.json"
    out_path.write_text(json.dumps(dataset, indent=2))

    print(f"\nWrote {len(training)} patients to {out_path}")
    if skipped:
        print(f"Skipped {skipped} patients (missing files)")


if __name__ == "__main__":
    main()