import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
from segment_anything.build_sam3D import sam_model_registry3D
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference + Evaluation on fine-tuned SAM-Med3D"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to fine-tuned checkpoint (.pth)")
    p.add_argument("--test-data-path", required=True,
                   help="Directory with imagesTr/ and labelsTr/ for testing")
    p.add_argument("--out-dir", required=True,
                   help="Directory where results will be saved")
    p.add_argument("--model-type", default="vit_b_ori")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Recommended: 1 for inference (GPU memory)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-clicks", type=int, default=3,
                   help="Number of interactive click iterations during inference")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid threshold for binarizing predictions")
    p.add_argument("--save-pred-masks", action="store_true",
                   help="Save predicted 3D masks as .nii.gz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2023)
    return p.parse_args()


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    inter  = np.logical_and(pred, gt).sum()
    union  = np.logical_or(pred, gt).sum()
    p_sum  = pred.sum()
    g_sum  = gt.sum()

    if p_sum == 0 and g_sum == 0:
        return dict(dice=1.0, iou=1.0, precision=1.0,
                    recall=1.0, f1=1.0, hd95=0.0)
    if p_sum == 0 or g_sum == 0:
        diag = float(np.sqrt(sum(s**2 for s in pred.shape)))
        return dict(dice=0.0, iou=0.0, precision=0.0,
                    recall=0.0, f1=0.0, hd95=diag)

    dice      = float(2 * inter / (p_sum + g_sum))
    iou       = float(inter / union) if union > 0 else 0.0
    precision = float(inter / p_sum)
    recall    = float(inter / g_sum)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    coords_pred = np.argwhere(pred)
    coords_gt   = np.argwhere(gt)
    d1 = directed_hausdorff(coords_pred, coords_gt)[0]
    d2 = directed_hausdorff(coords_gt,   coords_pred)[0]
    hd95 = float(max(d1, d2))

    return dict(dice=dice, iou=iou, precision=precision,
                recall=recall, f1=f1, hd95=hd95)


def batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                  points=None, device="cuda"):
    sparse_emb, dense_emb = sam_model.prompt_encoder(
        points=points, boxes=None, masks=low_res_masks,
    )
    low_res_masks, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding.to(device),
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    prev_masks = F.interpolate(
        low_res_masks, size=gt3D.shape[-3:],
        mode="trilinear", align_corners=False,
    )
    return low_res_masks, prev_masks


@torch.no_grad()
def run_inference(sam_model, image3D, gt3D, num_clicks, img_size, device):
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    image3D = norm_transform(image3D.squeeze(1)).unsqueeze(1)
    image3D = image3D.to(device)
    gt3D    = gt3D.to(device).long()

    img_emb = sam_model.image_encoder(image3D)

    prev_masks = torch.zeros_like(gt3D, dtype=torch.float32).to(device)
    low_res    = F.interpolate(
        prev_masks,
        size=(img_size // 4, img_size // 4, img_size // 4),
    )

    click_points = []
    click_labels = []

    for click_idx in range(num_clicks):
        batch_pts, batch_lbls = get_next_click3D_torch_2(prev_masks, gt3D)
        pts_co = torch.cat(batch_pts,  dim=0).to(device)
        pts_la = torch.cat(batch_lbls, dim=0).to(device)
        click_points.append(pts_co)
        click_labels.append(pts_la)

        pts_in  = torch.cat(click_points,  dim=1).to(device)
        lbls_in = torch.cat(click_labels,  dim=1).to(device)
        low_res, prev_masks = batch_forward(
            sam_model, img_emb, gt3D, low_res,
            points=[pts_in, lbls_in], device=device,
        )

    return torch.sigmoid(prev_masks)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = args.device if torch.cuda.is_available() else "cpu"
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_pred_masks:
        (out_dir / "pred_masks").mkdir(exist_ok=True)

    results_path = out_dir / "results.jsonl"

    log.info(f"Loading model: {args.checkpoint}")
    model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    log.info("Model loaded successfully.")

    transform = tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(
            mask_name="label",
            target_shape=(args.img_size, args.img_size, args.img_size),
        ),
    ])

    dataset = Dataset_Union_ALL(
        paths=[args.test_data_path],
        transform=transform,
        threshold=0,
    )
    loader = Union_Dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    log.info(f"Test samples: {len(dataset)}")

    processed_ids = set()
    all_metrics: list[dict] = []

    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    processed_ids.add(r.get("subject_id", ""))
                    all_metrics.append(r)
                except Exception:
                    continue
        log.info(f"Already processed: {len(processed_ids)}")

    for batch_idx, data3D in enumerate(tqdm(loader, desc="Inference")):
        try:
            image3D  = data3D["image"]
            gt3D     = data3D["label"]
            subject_ids = data3D.get("subject_id", [f"subject_{batch_idx}"])
        except Exception as e:
            log.warning(f"Skipped batch {batch_idx}: {e}")
            continue

        if isinstance(subject_ids, (list, tuple)):
            sid = subject_ids[0]
        else:
            sid = str(subject_ids)

        if sid in processed_ids:
            continue

        pred_sigmoid = run_inference(
            model, image3D.clone(), gt3D.clone(),
            args.num_clicks, args.img_size, device,
        )

        pred_np = (pred_sigmoid.squeeze().cpu().numpy() > args.threshold)
        gt_np   = (gt3D.squeeze().cpu().numpy() > 0)

        metrics = compute_metrics(pred_np, gt_np)
        metrics["subject_id"] = sid
        all_metrics.append(metrics)

        if args.save_pred_masks:
            mask_path = out_dir / "pred_masks" / f"{sid}_pred.nii.gz"
            nib.save(
                nib.Nifti1Image(pred_np.astype(np.uint8), np.eye(4)),
                str(mask_path),
            )
            metrics["pred_mask_path"] = str(mask_path)

        with open(results_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        log.info(
            f"[{batch_idx+1}/{len(loader)}] {sid:30s} | "
            f"Dice={metrics['dice']:.4f}  "
            f"IoU={metrics['iou']:.4f}  "
            f"HD95={metrics['hd95']:.1f}"
        )

    if not all_metrics:
        log.warning("No samples processed. Please check --test-data-path.")
        return

    keys = ["dice", "iou", "precision", "recall", "f1", "hd95"]
    stats = {k: np.mean([m[k] for m in all_metrics if k in m]) for k in keys}

    print(f"  FINAL RESULTS  ({len(all_metrics)} samples)\n")
    print(f"  Mean Dice      (DSC):   {stats['dice']:.4f}")
    print(f"  Mean IoU       (Jac):   {stats['iou']:.4f}")
    print(f"  Mean Precision:         {stats['precision']:.4f}")
    print(f"  Mean Recall:            {stats['recall']:.4f}")
    print(f"  Mean F1-Score:          {stats['f1']:.4f}")
    print(f"  Mean HD95 (mm/vox):     {stats['hd95']:.4f}")
    print(f"  Detailed results: {results_path}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_subjects":  len(all_metrics),
            "checkpoint":  args.checkpoint,
            "num_clicks":  args.num_clicks,
            **{f"mean_{k}": round(v, 6) for k, v in stats.items()},
        }, f, indent=2)

    log.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()