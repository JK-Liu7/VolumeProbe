# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0.

from monai import transforms
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
)


def get_transforms(args, dataset):
    if dataset == "BTCV":
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest"),
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=9,
                    neg=1,
                    num_samples=args.sw_batch_size,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2
                ),
                transforms.RandRotate90d(
                    keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3
                ),
                transforms.RandScaleIntensityd(
                    keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob
                ),
                transforms.RandShiftIntensityd(
                    keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob
                ),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest"),
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        return train_transform, val_transform
    if dataset == "Word":
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest"),
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=9,
                    neg=1,
                    num_samples=args.sw_batch_size,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2
                ),
                transforms.RandRotate90d(
                    keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3
                ),
                transforms.RandScaleIntensityd(
                    keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob
                ),
                transforms.RandShiftIntensityd(
                    keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest"),
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transform, val_transform
    if dataset == "Amos":
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest"),
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=9,
                    neg=1,
                    num_samples=args.sw_batch_size,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2
                ),
                transforms.RandRotate90d(
                    keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest"),
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transform, val_transform
    if dataset == "ATLAS-MRI":
        train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    k_divisible=[args.roi_x, args.roi_y, args.roi_z],
                ),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode="constant",
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.sw_batch_size,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
            ]
        )
        val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    k_divisible=[args.roi_x, args.roi_y, args.roi_z],
                ),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode="constant",
                ),
            ]
        )
        return train_transform, val_transform
    raise ValueError(f"Unknown dataset: {dataset}")
