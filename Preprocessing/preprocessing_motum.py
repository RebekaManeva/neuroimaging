import os
import json
import random
import numpy as np
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    DeleteItemsd,
    EnsureTyped,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandGaussianSmoothd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    MapTransform,
)


BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))

DATASET_JSON = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "motum",
    "dataset_preprocessed.json"
)

SPLITS_JSON = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "motum",
    "splits",
    "split_single.json"
)

MODS = ["t1", "t1ce", "t2", "flair"]


with open(DATASET_JSON, "r", encoding="utf-8") as f:
    DS = json.load(f)

with open(SPLITS_JSON, "r", encoding="utf-8") as f:
    SPLIT = json.load(f)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lists():
    all_items = DS["training"]

    train_ids = set(SPLIT["train"])
    val_ids = set(SPLIT["val"])

    train_list = []
    val_list = []

    for item in all_items:
        pid = item["patient_id"]

        sample = {
            "pid": pid,

            "t1": item["images"]["t1"],
            "t1ce": item["images"]["t1ce"],
            "t2": item["images"]["t2"],
            "flair": item["images"]["flair"],

            # merged_3class -> binary
            "label": item["labels"]["merged_3class"],
        }

        if pid in train_ids:
            train_list.append(sample)
        else:
            val_list.append(sample)

    print(f"Train samples: {len(train_list)}")
    print(f"Validation samples: {len(val_list)}")

    return train_list, val_list

# label binarization
class BinarizeLabeld(MapTransform):

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):

        d = dict(data)

        for key in self.keys:
            x = d[key]

            # tumor = 1
            # background = 0
            d[key] = (x > 0).to(x.dtype)

        return d

# preprocessing transforms
def get_transforms(
    patch_size=(112, 112, 112),
    num_samples=2,
    pos=2,
    neg=1,
):

    binarize = BinarizeLabeld(keys=["label"])

    common = [

        # loading nii.gz
        LoadImaged(
            keys=MODS + ["label"],
            image_only=True
        ),

        # channel first
        EnsureChannelFirstd(
            keys=MODS + ["label"]
        ),

        # z-score normalization
        NormalizeIntensityd(
            keys=MODS,
            nonzero=True,
            channel_wise=True
        ),

        # concatenating modalities
        ConcatItemsd(
            keys=MODS,
            name="im",
            dim=0
        ),

        # removing original keys
        DeleteItemsd(keys=MODS),

        # tensor conversion
        EnsureTyped(
            keys=["im", "label"],
            track_meta=False
        ),

        # binary labels
        binarize,
    ]

    # traing transforms
    train_tf = Compose(
        common + [

            # balanced crop
            RandCropByPosNegLabeld(
                keys=["im", "label"],
                label_key="label",

                spatial_size=patch_size,

                pos=pos,
                neg=neg,

                num_samples=num_samples,

                image_key="im",
                image_threshold=0.0,
            ),

            # random flips
            RandFlipd(
                keys=["im", "label"],
                prob=0.5,
                spatial_axis=0
            ),

            RandFlipd(
                keys=["im", "label"],
                prob=0.5,
                spatial_axis=1
            ),

            RandFlipd(
                keys=["im", "label"],
                prob=0.5,
                spatial_axis=2
            ),

            # affine augmentation
            RandAffined(
                keys=["im", "label"],
                prob=0.20,

                rotate_range=(0.1, 0.1, 0.1),

                scale_range=(0.1, 0.1, 0.1),

                mode=("bilinear", "nearest"),
            ),

            # contrast augmentation
            RandAdjustContrastd(
                keys=["im"],
                prob=0.10,
                gamma=(0.7, 1.5),
            ),

            # gaussian noise
            RandGaussianNoised(
                keys=["im"],
                prob=0.10,
                mean=0.0,
                std=0.01,
            ),

            # intensity scaling
            RandScaleIntensityd(
                keys=["im"],
                factors=0.10,
                prob=0.10,
            ),

            # intensity shifting
            RandShiftIntensityd(
                keys=["im"],
                offsets=0.10,
                prob=0.10,
            ),

            # gaussian smoothing
            RandGaussianSmoothd(
                keys=["im"],
                sigma_x=(0.5, 1.5),
                prob=0.10,
            ),
        ]
    )

    # validation transforms
    val_tf = Compose(common)

    return train_tf, val_tf


if __name__ == "__main__":

    seed_everything(42)

    train_list, val_list = build_lists()

    train_tf, val_tf = get_transforms()

    print("\nPreprocessing pipeline ready.")