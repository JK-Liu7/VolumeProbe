from collections.abc import Sequence
import os
import pickle
from math import ceil
from pathlib import Path

import numpy as np
from monai.data import *
from monai.transforms import *

from collections.abc import Mapping
import collections
import os
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform
import functools
import torch.utils.data as torch_data


from volumeprobe_lite import (
    VolumeProbeLiteBuilder,
    VolumeProbeLiteConfig,
    load_probe_cache,
)


# -----------------------------------------------------------------------------
# VolumeProbe cache helpers
# -----------------------------------------------------------------------------
def _as_3tuple(x, default=(1.5, 1.5, 1.5)):
    if x is None:
        return tuple(default)
    if isinstance(x, (list, tuple)):
        if len(x) != 3:
            raise ValueError(f"Expected length-3 tuple/list, got {x}")
        return tuple(float(v) for v in x)
    return (float(x), float(x), float(x))


def _get_probe_crop_size(args):
    return (
        int(getattr(args, "probe_crop_x", 384)),
        int(getattr(args, "probe_crop_y", 384)),
        int(getattr(args, "probe_crop_z", 96)),
    )


def _get_probe_cfg(args):
    coarse_grid = getattr(args, "probe_coarse_grid", (4, 4, 2))
    if not isinstance(coarse_grid, (list, tuple)):
        coarse_grid = (4, 4, 2)
    return VolumeProbeLiteConfig(
        axcodes=str(getattr(args, "probe_axcodes", "RAS")),
        pixdim=_as_3tuple(getattr(args, "probe_pixdim", (1.5, 1.5, 1.5))),
        spacing_mode=str(getattr(args, "probe_spacing_mode", "bilinear")),
        a_min=float(args.a_min),
        a_max=float(args.a_max),
        b_min=float(args.b_min),
        b_max=float(args.b_max),
        intensity_clip=True,
        fg_threshold=float(getattr(args, "probe_fg_threshold", 0.3)),
        coarse_grid=tuple(int(v) for v in coarse_grid),
        top_coarse_bins=int(getattr(args, "probe_top_coarse_bins", 24)),
        fine_offset_frac=float(getattr(args, "probe_fine_offset_frac", 0.25)),
        top_k=int(getattr(args, "probe_top_k", 40)),
        roi_size=(int(args.roi_x), int(args.roi_y), int(args.roi_z)),
        alpha=float(getattr(args, "probe_alpha", 1.0)),
        beta=float(getattr(args, "probe_beta", 1.0)),
        rho=float(getattr(args, "probe_rho", 0.6)),
        grad_entropy_ratio=float(getattr(args, "probe_grad_entropy_ratio", 0.5)),
        min_center_distance_norm=getattr(args, "probe_min_center_distance_norm", None),
        min_center_distance_factor=float(getattr(args, "probe_min_center_distance_factor", 0.35)),
        entropy_bins=int(getattr(args, "probe_entropy_bins", 32)),
    )


def _strip_to_image_only(items):
    out = []
    for item in items:
        out.append({"image": item["image"]})
    return out


def _get_probe_cache_dir(args):
    if getattr(args, "probe_cache_dir", None) is not None:
        return Path(getattr(args, "probe_cache_dir"))
    default_name = f"volumeprobe_cache_{getattr(args, 'dataset_variant', 'full')}"
    return Path(args.data_dir) / "volumeprobe_cache" / default_name


def _attach_existing_probe_cache_paths(datalist, args):
    use_probe = bool(getattr(args, "use_volumeprobe_lite", True))
    if not use_probe:
        return datalist

    cache_dir = _get_probe_cache_dir(args)
    cfg = _get_probe_cfg(args)
    builder = VolumeProbeLiteBuilder(cfg)

    out = []
    num_missing = 0
    for item in datalist:
        item_new = dict(item)
        cache_path = builder._cache_path(cache_dir, item["image"])
        item_new["probe_cache_path"] = str(cache_path)
        if not cache_path.exists():
            num_missing += 1
        out.append(item_new)

    print(f"VolumeProbe cache dir: {cache_dir}")
    if num_missing > 0:
        print(f"Warning: {num_missing}/{len(out)} samples have no existing probe cache. They will fall back to random crop.")
    else:
        print(f"All {len(out)} samples found existing probe cache.")
    return out



class AdaptiveRandomMixCropd(RandomizableTransform, MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        roi_size,
        p_candidate=0.7,
        cache_path_key="probe_cache_path",
        check_shape_match=True,
        shape_mismatch_mode="warn_and_random",
        allow_missing_keys=False,
        jitter_frac=0.05,
        seed=0,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=1.0)

        self.roi_size = np.asarray(roi_size, dtype=np.int32)
        self.p_candidate = float(np.clip(p_candidate, 0.0, 1.0))
        self.cache_path_key = cache_path_key
        self.check_shape_match = bool(check_shape_match)
        self.shape_mismatch_mode = str(shape_mismatch_mode)

        self._cache_records = {}

        self._visit_counts = collections.defaultdict(
            functools.partial(collections.defaultdict, int)
        )

        self._shape_mismatch_warned = set()

        self.set_random_state(seed=seed)

        self._rand_center = None
        self._rand_use_candidate = False
        self._rand_chosen_info = None
        self._rand_shape_match = None
        self._rand_cache_shape = None
        self._rand_current_shape = None

        self.jitter_frac = float(max(jitter_frac, 0.0))

    def _load_cache(self, cache_path):
        if cache_path not in self._cache_records:
            self._cache_records[cache_path] = load_probe_cache(cache_path)
        return self._cache_records[cache_path]

    def _check_cache_shape_match(self, cache_record, spatial_shape, cache_path=None):
        cache_shape = cache_record.get("shape_after_preprocess", None)
        if cache_shape is None:
            return True, None

        cache_shape = tuple(int(v) for v in cache_shape)
        current_shape = tuple(int(v) for v in spatial_shape)
        if cache_shape == current_shape:
            return True, None

        message = (
            "[AdaptiveRandomMixCropd] shape_after_preprocess mismatch"
            f" | image={cache_record.get('image_path', 'unknown')}"
            f" | cache_shape={cache_shape}"
            f" | current_shape={current_shape}"
        )
        if cache_path is not None:
            message += f" | cache_path={cache_path}"
        return False, message

    def _random_center(self, spatial_shape):
        center = []
        for s in spatial_shape:
            s = int(s)
            if s <= 1:
                center.append(0)
            else:
                center.append(int(self.R.randint(0, s)))
        return tuple(center)

    def _jitter_center(self, center, spatial_shape):
        spatial_shape = np.asarray(spatial_shape, dtype=np.int32)
        jitter_vox = np.maximum(
            np.round(self.roi_size.astype(np.float32) * self.jitter_frac).astype(np.int32), 0
        )
        out = []
        for c, s, j in zip(center, spatial_shape.tolist(), jitter_vox.tolist()):
            if s <= 1 or j <= 0:
                out.append(int(np.clip(c, 0, max(s - 1, 0))))
                continue
            delta = int(self.R.randint(-j, j + 1))
            out.append(int(np.clip(int(c) + delta, 0, s - 1)))
        return tuple(out)

    def _crop_with_pad(self, img, center):
        spatial = np.asarray(img.shape[1:], dtype=np.int32)
        roi = self.roi_size
        center = np.asarray(center, dtype=np.int32)

        starts = []
        ends = []
        for dim, c, r in zip(spatial.tolist(), center.tolist(), roi.tolist()):
            if dim <= r:
                starts.append(0)
                ends.append(dim)
                continue
            st = int(c - r // 2)
            ed = st + int(r)
            if st < 0:
                ed -= st
                st = 0
            if ed > dim:
                st -= (ed - dim)
                ed = dim
            st = max(st, 0)
            ed = min(ed, dim)
            starts.append(st)
            ends.append(ed)

        patch = img[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
        patch_spatial = np.asarray(patch.shape[1:], dtype=np.int32)
        if np.all(patch_spatial == roi):
            return patch

        pad_width = [(0, 0)]
        for got, want in zip(patch_spatial.tolist(), roi.tolist()):
            total = max(int(want - got), 0)
            before = total // 2
            after = total - before
            pad_width.append((before, after))
        patch = np.pad(patch, pad_width=pad_width, mode="constant", constant_values=0)
        return patch

    def randomize(self, img, cache_path=None):
        super().randomize(None)

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        if img.ndim != 4:
            raise ValueError(f"Expected [C,H,W,D] image before adaptive crop, got {img.shape}")

        spatial_shape = img.shape[1:]

        self._rand_center = None
        self._rand_use_candidate = False
        self._rand_chosen_info = None
        self._rand_shape_match = None
        self._rand_cache_shape = None
        self._rand_current_shape = np.asarray(spatial_shape, dtype=np.int32)

        should_try_candidate = (
            cache_path is not None
            and os.path.exists(cache_path)
            and (self.R.rand() < self.p_candidate)
        )

        if should_try_candidate:
            try:
                cache_record = self._load_cache(cache_path)

                if self.check_shape_match:
                    is_match, mismatch_msg = self._check_cache_shape_match(
                        cache_record=cache_record,
                        spatial_shape=spatial_shape,
                        cache_path=cache_path,
                    )
                    self._rand_shape_match = is_match
                    self._rand_cache_shape = np.asarray(
                        cache_record.get("shape_after_preprocess", ()), dtype=np.int32
                    )

                    if not is_match:
                        if self.shape_mismatch_mode == "raise":
                            raise RuntimeError(mismatch_msg)
                        if cache_path not in self._shape_mismatch_warned:
                            print(mismatch_msg)
                            self._shape_mismatch_warned.add(cache_path)
                        center = self._random_center(spatial_shape)
                    else:
                        visit_counts = self._visit_counts[cache_path]
                        center, chosen_info = VolumeProbeLiteBuilder.sample_center_from_cache(
                            cache_record=cache_record,
                            rng=self.R,
                            coarse_visit_counts=visit_counts,
                            lambda_cov=None,
                            temperature=None,
                        )
                        if chosen_info is not None:
                            visit_counts[int(chosen_info["bin_id"])] += 1
                        self._rand_use_candidate = True
                        self._rand_chosen_info = chosen_info
                else:
                    visit_counts = self._visit_counts[cache_path]
                    center, chosen_info = VolumeProbeLiteBuilder.sample_center_from_cache(
                        cache_record=cache_record,
                        rng=self.R,
                        coarse_visit_counts=visit_counts,
                        lambda_cov=None,
                        temperature=None,
                    )
                    if chosen_info is not None:
                        visit_counts[int(chosen_info["bin_id"])] += 1
                    self._rand_use_candidate = True
                    self._rand_chosen_info = chosen_info
            except Exception as e:
                print(f"[AdaptiveRandomMixCropd] candidate sampling failed for {cache_path}: {e}. Fall back to random crop.")
                center = self._random_center(spatial_shape)
        else:
            center = self._random_center(spatial_shape)

        center = self._jitter_center(center, spatial_shape)
        self._rand_center = center

    def __call__(self, data: Mapping):
        d = dict(data)
        cache_path = d.get(self.cache_path_key, None)

        first_key = next(iter(self.key_iterator(d)))
        img0 = d[first_key]
        if not isinstance(img0, np.ndarray):
            img0 = np.asarray(img0)

        self.randomize(img0, cache_path=cache_path)

        for key in self.key_iterator(d):
            img = d[key]
            if not isinstance(img, np.ndarray):
                img = np.asarray(img)

            d[key] = self._crop_with_pad(img, self._rand_center)

        d["adaptive_crop_center"] = np.asarray(self._rand_center, dtype=np.int32)
        d["adaptive_crop_source"] = "candidate" if self._rand_use_candidate else "random"

        d["adaptive_crop_shape_match"] = (
            False if self._rand_shape_match is None else bool(self._rand_shape_match)
        )

        d["adaptive_crop_cache_shape"] = (
            np.asarray([-1, -1, -1], dtype=np.int32)
            if self._rand_cache_shape is None else self._rand_cache_shape
        )

        d["adaptive_crop_current_shape"] = (
            np.asarray([-1, -1, -1], dtype=np.int32)
            if self._rand_current_shape is None else self._rand_current_shape
        )

        d["adaptive_crop_bin_id"] = (
            -1 if self._rand_chosen_info is None else int(self._rand_chosen_info["bin_id"])
        )

        d["adaptive_crop_score"] = (
            -1.0 if self._rand_chosen_info is None else float(self._rand_chosen_info["lite_score"])
        )

        return d


# -----------------------------------------------------------------------------
# Original VoCo dataset helpers
# -----------------------------------------------------------------------------
def random_split(ls):
    length = len(ls)
    train_ls = ls[:ceil(length * 0.9)]
    val_ls = ls[ceil(length * 0.9):]
    return train_ls, val_ls


def threshold(x):
    return x > 0.3


class VoCoAugmentation:
    def __init__(self, args, aug):
        self.args = args
        self.aug = aug

        self.crops_trans = get_crop_transform(roi_small=self.args.roi_x, aug=self.aug)

    def __call__(self, x_in):

        crops_trans = self.crops_trans

        vanilla_trans, labels = get_vanilla_transform(
            num=self.args.sw_batch_size,
            roi_small=self.args.roi_x,
            aug=self.aug,
        )

        imgs = []
        for trans in vanilla_trans:
            imgs.append(trans(x_in))

        crops = []
        for trans in crops_trans:
            crops.append(trans(x_in))

        return imgs, labels, crops


def get_vanilla_transform(num=2, num_crops=4, roi_small=64, roi=96, max_roi=384, aug=False):
    vanilla_trans = []
    labels = []
    for _ in range(num):
        center_x, center_y, label = get_position_label(
            roi=roi,
            max_roi=max_roi,
            num_crops=num_crops,
        )
        if aug:
            trans = Compose([
                SpatialCropd(keys=['image'], roi_center=[center_x, center_y, roi // 2], roi_size=[roi, roi, roi]),
                Resized(keys=["image"], mode="bilinear", align_corners=True, spatial_size=(roi_small, roi_small, roi_small)),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                ToTensord(keys=["image"]),
            ])
        else:
            trans = Compose([
                SpatialCropd(keys=['image'], roi_center=[center_x, center_y, roi // 2], roi_size=[roi, roi, roi]),
                Resized(keys=["image"], mode="bilinear", align_corners=True, spatial_size=(roi_small, roi_small, roi_small)),
                ToTensord(keys=["image"]),
            ])
        vanilla_trans.append(trans)
        labels.append(label)

    labels = np.concatenate(labels, 0).reshape(num, num_crops * num_crops)
    return vanilla_trans, labels


def get_crop_transform(num=4, roi_small=64, roi=96, aug=False):
    voco_trans = []
    for i in range(num):
        for j in range(num):
            center_x = (i + 1 / 2) * roi
            center_y = (j + 1 / 2) * roi
            center_z = roi // 2

            if aug:
                trans = Compose([
                    SpatialCropd(keys=['image'], roi_center=[center_x, center_y, center_z], roi_size=[roi, roi, roi]),
                    Resized(keys=["image"], mode="bilinear", align_corners=True, spatial_size=(roi_small, roi_small, roi_small)),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                    RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                    ToTensord(keys=["image"]),
                ])
            else:
                trans = Compose([
                    SpatialCropd(keys=['image'], roi_center=[center_x, center_y, center_z], roi_size=[roi, roi, roi]),
                    Resized(keys=["image"], mode="bilinear", align_corners=True, spatial_size=(roi_small, roi_small, roi_small)),
                    ToTensord(keys=["image"]),
                ])
            voco_trans.append(trans)
    return voco_trans


def get_position_label(roi=96, base_roi=96, max_roi=384, num_crops=4):
    half = roi // 2
    center_x = np.random.randint(low=half, high=max_roi - half)
    center_y = np.random.randint(low=half, high=max_roi - half)

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


# -----------------------------------------------------------------------------
# Dataloaders with existing VolumeProbe cache
# -----------------------------------------------------------------------------
def _build_deterministic_transform(args):
    probe_cfg = _get_probe_cfg(args)
    return Compose([
        LoadImaged(keys=["image"], image_only=True, dtype=np.float32),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=probe_cfg.axcodes),
        Spacingd(keys=["image"], pixdim=probe_cfg.pixdim, mode=probe_cfg.spacing_mode),
        ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", select_fn=threshold),
    ])


def _build_random_transform(args):
    probe_cfg = _get_probe_cfg(args)
    probe_crop_size = _get_probe_crop_size(args)
    return Compose([
        AdaptiveRandomMixCropd(
            keys=["image"],
            roi_size=probe_crop_size,
            p_candidate=float(getattr(args, "probe_candidate_prob", 0.7)),
            jitter_frac=float(getattr(args, "probe_jitter_frac", 0.05)),
            cache_path_key="probe_cache_path",
            check_shape_match=bool(getattr(args, "probe_check_shape_match", True)),
            shape_mismatch_mode=str(getattr(args, "probe_shape_mismatch_mode", "warn_and_random")),
            seed=int(getattr(args, "probe_seed", args.seed + args.rank)),
        ),
        VoCoAugmentation(args, aug=True),
    ])


class TwoStageDataset(torch_data.Dataset):
    def __init__(self, det_ds, rand_transform):
        self.det_ds = det_ds
        self.rand_transform = rand_transform

    def __len__(self):
        return len(self.det_ds)

    def __getitem__(self, idx):
        item = self.det_ds[idx]
        return self.rand_transform(item)


def _build_val_transform(args):
    probe_cfg = _get_probe_cfg(args)
    probe_crop_size = _get_probe_crop_size(args)
    return Compose([
        LoadImaged(keys=["image"], image_only=True, dtype=np.float32),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=probe_cfg.axcodes),
        Spacingd(keys=["image"], pixdim=probe_cfg.pixdim, mode=probe_cfg.spacing_mode),
        ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", select_fn=threshold),
        CenterSpatialCropd(keys=["image"], roi_size=probe_crop_size),
        SpatialPadd(keys=["image"], spatial_size=probe_crop_size),
        VoCoAugmentation(args, aug=False),
    ])


def _make_train_dataset(datalist, args, num_workers):
    det_transform = _build_deterministic_transform(args)
    rand_transform = _build_random_transform(args)

    print("Using PersistentDataset (det only) + TwoStageDataset")
    det_ds = PersistentDataset(
        data=datalist,
        transform=det_transform,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
        cache_dir=args.cache_dir + 'VoCo-10k/',
    )
    return TwoStageDataset(det_ds, rand_transform)


def get_loader(args):
    args.dataset_variant = "full"
    list_dir = os.path.join(args.data_dir, "jsons")
    jsonlist1 = os.path.join(list_dir, "btcv.json")
    jsonlist2 = os.path.join(list_dir, "dataset_TCIAcovid19_0.json")
    jsonlist3 = os.path.join(list_dir, "dataset_LUNA16_0.json")
    jsonlist4 = os.path.join(list_dir, "stoic21.json")
    jsonlist5 = os.path.join(list_dir, "Totalsegmentator_dataset.json")
    jsonlist6 = os.path.join(list_dir, "flare23.json")
    jsonlist7 = os.path.join(list_dir, "HNSCC.json")

    datadir1 = os.path.join(args.data_dir, "BTCV")
    datadir2 = os.path.join(args.data_dir, "TCIAcovid19")
    datadir3 = os.path.join(args.data_dir, "Luna16")
    if not os.path.exists(datadir3):
        alt = os.path.join(args.data_dir, "Luna16-jx")
        if os.path.exists(alt):
            datadir3 = alt
    datadir4 = os.path.join(args.data_dir, "stoic21")
    datadir5 = os.path.join(args.data_dir, "Totalsegmentator_dataset")
    datadir6 = os.path.join(args.data_dir, "Flare23")
    datadir7 = os.path.join(args.data_dir, "HNSCC_convert_v1")

    num_workers = args.num_workers
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))
    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print("Dataset 3 Luna: number of data: {}".format(len(datalist3)))
    datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    print("Dataset 4 Stoic21: number of data: {}".format(len(datalist4)))
    datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    print("Dataset 5 Totalsegmentator: number of data: {}".format(len(datalist5)))
    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    print("Dataset 6 Flare23: number of data: {}".format(len(datalist6)))
    datalist7 = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    print("Dataset 7 HNSCC: number of data: {}".format(len(datalist7)))

    datalist = (
        _strip_to_image_only(datalist1)
        + datalist2
        + _strip_to_image_only(datalist3)
        + datalist4
        + datalist5
        + datalist6
        + datalist7
    )
    print("Dataset all training: number of data: {}".format(len(datalist)))

    datalist = _attach_existing_probe_cache_paths(datalist, args)

    train_ds = _make_train_dataset(datalist, args, num_workers)

    if args.distributed:
        train_sampler = DistributedSampler(
            dataset=train_ds,
            even_divisible=True,
            shuffle=True,
            rank=args.rank,
            num_replicas=args.world_size,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return train_loader


if __name__ == '__main__':
    center_x, center_y, labels = get_position_label()
    print(center_x, center_y, labels)
