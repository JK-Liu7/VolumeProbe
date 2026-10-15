from __future__ import annotations


import numpy as np
import random
import torch
from monai.transforms import (
    Affined,
    Compose,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    SpatialCropd,
    ToTensord,
)


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    b, c, d = -axis * np.sin(theta / 2.0)
    theta = np.pi * theta / 180
    mat_rot1 = np.array(
        [
            [np.cos(theta), np.sin(theta), 0, 0],
            [-np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    mat_rot2 = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return mat_rot1, mat_rot2


def translation_matrix(translation):
    T = np.eye(4)
    T[:3, 3] = translation[:3]
    return T


def scaling_matrix(scale):
    S = np.eye(4)
    S[0, 0] = scale[0]
    S[1, 1] = scale[1]
    S[2, 2] = scale[2]
    return S


def get_position_label(roi=96, base_roi=96, max_roi=384, num_crops=4):
    half = roi // 2
    center_x, center_y = (
        np.random.randint(low=half, high=max_roi - half),
        np.random.randint(low=half, high=max_roi - half),
    )

    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half

    total_area = roi * roi
    labels = []
    for i in range(num_crops):
        for j in range(num_crops):
            crop_x_min, crop_x_max = i * base_roi, (i + 1) * base_roi
            crop_y_min, crop_y_max = j * base_roi, (j + 1) * base_roi

            dx = min(crop_x_max, x_max) - max(crop_x_min, x_min)
            dy = min(crop_y_max, y_max) - max(crop_y_min, y_min)
            if dx <= 0 or dy <= 0:
                area = 0
            else:
                area = (dx * dy) / total_area
            labels.append(area)

    labels = np.asarray(labels).reshape(1, num_crops * num_crops)

    return center_x, center_y, labels


def get_vanilla_transform(num=2, num_crops=4, roi_small=64, roi=96, max_roi=384, aug=False):
    vanilla_trans = []
    labels = []
    for i in range(num):
        center_x, center_y, label = get_position_label(
            roi=roi, max_roi=max_roi, num_crops=num_crops
        )
        if aug:
            trans = Compose(
                [
                    SpatialCropd(
                        keys=["image"],
                        roi_center=[center_x, center_y, roi // 2],
                        roi_size=[roi, roi, roi],
                    ),
                    Resized(
                        keys=["image"],
                        mode="trilinear",
                        align_corners=True,
                        spatial_size=(roi_small, roi_small, roi_small),
                    ),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                    RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                    ToTensord(keys=["image"]),
                ]
            )
        else:
            trans = Compose(
                [
                    SpatialCropd(
                        keys=["image"],
                        roi_center=[center_x, center_y, roi // 2],
                        roi_size=[roi, roi, roi],
                    ),
                    Resized(
                        keys=["image"],
                        mode="trilinear",
                        align_corners=True,
                        spatial_size=(roi_small, roi_small, roi_small),
                    ),
                    ToTensord(keys=["image"]),
                ]
            )

        vanilla_trans.append(trans)
        labels.append(label)
    labels = np.concatenate(labels, 0).reshape(num, num_crops * num_crops)

    return vanilla_trans, labels


def get_centers(num_patch=3, num_token=6, roi_large=384):
    centers = []
    roi = roi_large / (num_patch * num_token)
    roi_z = roi_large / num_patch
    for i in range(num_patch * num_token):
        for j in range(num_patch * num_token):
            center_x = (i + 1 / 2) * roi
            center_y = (j + 1 / 2) * roi
            center_z = roi_z // 2
            center = np.array([center_x, center_y, center_z])
            centers.append(center)
    return np.stack(centers)


def get_crop_transform(num=4, roi_small=64, roi=96, aug=False):
    voco_trans = []

    for i in range(num):
        for j in range(num):
            center_x = (i + 1 / 2) * roi
            center_y = (j + 1 / 2) * roi
            center_z = roi // 2
            if aug:
                trans = Compose(
                    [
                        SpatialCropd(
                            keys=["image"],
                            roi_center=[center_x, center_y, center_z],
                            roi_size=[roi, roi, roi],
                        ),
                        Resized(
                            keys=["image"],
                            mode="trilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small),
                        ),
                        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                        RandAffined(
                            keys=["image"],
                            mode=("bilinear"),
                            prob=1.0,
                            spatial_size=(roi_small, roi_small, roi_small),
                            translate_range=(
                                random.randint(-roi // 4, roi // 4),
                                random.randint(-roi // 4, roi // 4),
                                0,
                            ),
                            rotate_range=(
                                np.pi * random.randint(-20, 20) / 180,
                                np.pi * random.randint(-20, 20) / 180,
                                0,
                            ),
                            scale_range=(random.uniform(0.95, 1.05), random.uniform(0.95, 1.05), 1),
                            padding_mode="zeros",
                        ),
                        ToTensord(keys=["image"]),
                    ],
                )
            else:
                trans = Compose(
                    [
                        SpatialCropd(
                            keys=["image"],
                            roi_center=[center_x, center_y, center_z],
                            roi_size=[roi, roi, roi],
                        ),
                        Resized(
                            keys=["image"],
                            mode="trilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small),
                        ),
                        ToTensord(keys=["image"]),
                    ],
                )

            voco_trans.append(trans)

    return voco_trans


def create_affine_matrix_3d(axis, translation, angle, scale):
    R1, R2 = rotation_matrix(axis, angle)
    S1 = scaling_matrix(scale)
    S2 = scaling_matrix(1 / np.array(scale))
    T1 = translation_matrix(translation)
    translation = np.array(translation)
    T2 = translation_matrix(translation)
    transform_matrix1 = T1 @ R1 @ S1
    transform_matrix2 = T2 @ R2 @ S2

    return transform_matrix1, transform_matrix2


def get_affine_transform(roi=96, roi_large=384):
    translation = (random.randint(-roi // 4, roi // 4), random.randint(-roi // 4, roi // 4), 0)
    angle = random.randint(-30, 30)
    s = random.uniform(0.9, 1.2)
    scale = (s, s, s)
    axis = [0, 0, 1]
    affine_matrix1, affine_matrix2 = create_affine_matrix_3d(axis, translation, angle, scale)
    final_transform_matrix = affine_matrix1
    final_transform_matrix = torch.tensor(final_transform_matrix)
    affine_matrix2 = torch.tensor(affine_matrix2)
    trans = Compose(
        [
            Affined(keys=["image"], affine=final_transform_matrix, padding_mode="zeros"),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            ToTensord(keys=["image"]),
        ],
    )
    inv_affine_matrix = np.linalg.inv(affine_matrix2.numpy())
    inv_affine_matrix = torch.tensor(inv_affine_matrix)
    return trans, affine_matrix2, inv_affine_matrix


class GeoAugmentation:
    def __init__(self, args, roi_large, crop_size=96):
        self.args = args
        self.roi_large = roi_large
        self.crop_size = crop_size
        self.num_patch_side = roi_large // crop_size
        num_layer_token_side = [args.roi_x // 2 ** (i + 1) for i in range(5)]
        self.num_token_side = num_layer_token_side[args.num_geo_layer]
        self.token_crop_size = roi_large // (self.num_patch_side * self.num_token_side)
        self.num = 0

    def __call__(self, x_in):
        x_in["image"] = x_in["image"][0, ...][None, ...]
        centers = get_centers(
            num_patch=self.num_patch_side, num_token=self.num_token_side, roi_large=self.roi_large
        )
        crops_vanilla = get_crop_transform(
            num=self.num_patch_side, roi=self.crop_size, roi_small=self.args.roi_x, aug=False
        )
        crops_trans_vanilla = get_crop_transform(
            num=self.num_patch_side, roi=self.crop_size, roi_small=self.args.roi_x, aug=True
        )

        affine_trans, mat, inv_mat = get_affine_transform(
            roi=self.crop_size, roi_large=self.roi_large
        )
        centers[:, 0, ...] = self.roi_large / 2 - centers[:, 0, ...]
        centers[:, 1, ...] = self.roi_large / 2 - centers[:, 1, ...]
        centers_ = torch.ones(centers.shape[0], 4).to(torch.float64)
        centers_[:, :3] = torch.tensor(centers)
        centers = centers_
        points01 = (mat @ centers.T).T
        points10 = (inv_mat @ centers.T).T
        points01[:, 0] = self.roi_large / 2 - points01[:, 0]
        points10[:, 0] = self.roi_large / 2 - points10[:, 0]
        points01[:, 1] = self.roi_large / 2 - points01[:, 1]
        points10[:, 1] = self.roi_large / 2 - points10[:, 1]
        points01 = points01 / self.token_crop_size
        points10 = points10 / self.token_crop_size

        points01[(points01 < 0) | (points01 >= self.num_patch_side * self.num_token_side)] = -1000
        points10[(points10 < 0) | (points10 >= self.num_patch_side * self.num_token_side)] = -1000
        points01 = np.floor(points01).long()
        points10 = np.floor(points10).long()
        nearest_index01 = (
            points01[:, 0] * self.num_patch_side * self.num_token_side + points01[:, 1]
        )
        nearest_index10 = (
            points10[:, 0] * self.num_patch_side * self.num_token_side + points10[:, 1]
        )
        nearest_index01_bool = np.ones(nearest_index01.shape[0])
        index_max = (self.num_token_side * self.num_patch_side) ** 2
        nearest_index01_bool[(nearest_index01 > index_max - 1) | (nearest_index01 < 0)] = 0
        nearest_index01_bool = np.bool_(nearest_index01_bool)

        loop_back = []
        for i in range(index_max):
            if nearest_index01[i] >= index_max or nearest_index01[i] < 0:
                loop_back.append(torch.tensor(-1))
            else:
                loop_back.append(nearest_index10[nearest_index01[i]])
        loop_back = torch.stack(loop_back, dim=0)
        mnn_0to1 = (loop_back == torch.arange(index_max)) & nearest_index01_bool
        conf_matrix_gt = torch.zeros(index_max, index_max)
        i_ids = nearest_index01[mnn_0to1.bool()]
        j_ids = nearest_index10[i_ids]
        conf_matrix_gt[j_ids, i_ids] = 1

        crops, crops_full_img_aug, crops_aug = [], [], []
        for trans in crops_vanilla:
            crop = trans(x_in)
            crops.append(crop)
        for trans in crops_trans_vanilla:
            crop = trans(x_in)
            crops_aug.append(crop)

        x_in = affine_trans(x_in)
        for trans in crops_vanilla:
            crop = trans(x_in)
            crops_full_img_aug.append(crop)

        centers[:, 0, ...] = self.roi_large / 2 - centers[:, 0, ...]
        centers[:, 1, ...] = self.roi_large / 2 - centers[:, 1, ...]
        return crops, crops_full_img_aug, crops_aug, conf_matrix_gt
