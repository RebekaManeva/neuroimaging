# Brain Tumor Segmentation — MOTUM & BraTS 2023

Comparative evaluation of 3D segmentation architectures (UNet, VNet, SegResNet, DynUNet) and LoRA fine-tuned MedGemma for tumor localization on multi-modal brain MRI, across two datasets: **MOTUM** and **BraTS 2023**.

---

## Datasets

### MOTUM
- 69 patients, 4 MRI modalities per subject: T1, T1ce, T2, FLAIR
- Expert-annotated ground-truth masks (NIfTI volumes)
- Labels: `flair_abn` (FLAIR abnormality) and `ce_core` (contrast-enhancing core)
- Split: 53 training / 16 validation (no held-out test set due to small size)

### BraTS 2023
- 1,251 patients total; 200-patient subset used (hardware constraints)
- 4 MRI modalities: T1, T1ce, T2, FLAIR
- Tumor sub-regions: Enhancing Tumor (ET), Tumor Core (TC), Whole Tumor (WT)
- Split: 140 train / 30 validation / 30 test (70/15/15)

---

## Preprocessing

### MOTUM (`Preprocessing/preprocessing_motum.py`)
1. Reorient all volumes to RAS orientation
2. Resample to isotropic 1×1×1 mm spacing (bilinear for images, nearest-neighbor for masks)
3. Register T1ce, T2, FLAIR onto T1 reference grid
4. Merge tumor masks into a single 3-class volume (0 = background, 1 = flair_abn, 2 = ce_core)

### BraTS 2023 (`Preprocessing/preprocessing_BraTS2023.py`)
Starting from the BraTS-provided preprocessed data (DICOM→NIfTI, skull stripping, SRI24 registration, 1 mm³ resampling):
1. Bounding-box crop on T1 reference with 5-voxel margin
2. Apply same crop to all modalities and mask
3. Z-score normalization per modality (non-zero voxels only)
4. Stack 4 modalities into a single `(H, W, D, 4)` NumPy array; save mask separately

---

## Models

All architectures take a 4-channel input (T1, T1ce, T2, FLAIR stacked).
- UNet
- VNet
- SegResNet
- DynUnet


---

## MedGemma

Fine-tuning of `google/medgemma-1.5-4b-it` for 2D tumor bounding-box localization using LoRA.


### LoRA Setup
- Applied to attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- MOTUM: rank 32, α=32 — 23.8M trainable params (0.55% of model)
- BraTS: rank 32, α=32, extended to MLP layers — 65.6M trainable params (1.50% of model); 4-bit NF4 quantization
- Two separate models trained per dataset: one for `flair_abn`, one for `ce_core`
- Evaluation: patient-level IoU using best-scoring slice per patient

---

## Results Summary

### Segmentation (Dice, with preprocessing)

| Dataset | Model | Dice | Precision | Recall | IoU | HD95 (mm) |
|---|---|---|---|---|---|---|
| MOTUM | SegResNet | **0.82** | 0.78 | 0.89 | 0.72 | 21 |
| MOTUM | VNet | 0.77 | 0.79 | 0.79 | 0.67 | 13 |
| MOTUM | UNet | 0.80 | 0.72 | 0.93 | 0.68 | 33 |
| MOTUM | DynUNet | 0.72 | 0.61 | 0.64 | 0.79 | 23 |
| BraTS 2023 | UNet | **0.84**\* | 0.83 | 0.90 | 0.77 | 3 |
| BraTS 2023 | SegResNet | 0.81 | 0.82 | 0.87 | 0.74 | 4 |
| BraTS 2023 | VNet | 0.78 | 0.85 | 0.81 | 0.71 | 5 |
| BraTS 2023 | DynUNet | 0.53 | 0.45 | 0.95 | 0.44 | 20 |

\* Best BraTS result achieved *without* preprocessing (BraTS data is already standardized)

### MedGemma LoRA — MOTUM (2D bounding box IoU, patient-level best slice)

| Label | Mean IoU | Median IoU | ≥0.5 threshold |
|---|---|---|---|
| flair_abn | 0.369 | 0.323 | 40% of patients |
| ce_core | 0.190 | 0.120 | 10% of patients |

---

## Evaluation Metrics

- **Dice** — voxel overlap: `2TP / (2TP + FP + FN)`
- **IoU** — Jaccard index: `TP / (TP + FP + FN)`
- **Precision** — `TP / (TP + FP)`
- **Recall** — `TP / (TP + FN)`
- **F1** — harmonic mean of Precision and Recall
- **HD95** — 95th percentile Hausdorff distance (boundary error in mm)

---

## Key Findings

- **SegResNet** is the most consistent model across both datasets and preprocessing conditions.
- **Preprocessing is critical but model-dependent**: large gains for UNet on MOTUM; neutral or harmful for DynUNet on BraTS.
- **MedGemma zero-shot** fails entirely for tumor localization; LoRA fine-tuning enables meaningful but coarse localization.
- **Localization difficulty scales with lesion characteristics**: diffuse FLAIR lesions (IoU 0.37) are easier to localize than focal contrast-enhancing cores (IoU 0.19).
- A promising hybrid workflow: MedGemma as a fast ROI proposal → SegResNet for voxel-level refinement.

---

## Authors

**Rebeka Maneva** — MOTUM segmentation, MedGemma fine-tuning (MOTUM)
`rebeka.maneva@students.finki.ukim.mk`

**Elena Nikolovska** — BraTS 2023 segmentation, MedGemma fine-tuning (BraTS)
`elena.nikolovska@students.finki.ukim.mk`

Supervisors: **Ilinka Ivanoska PhD.** and **Katarina Trojachanec Dineva PhD.**
Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University, Skopje

