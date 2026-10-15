from pretraining.data import PersistentDataset, read_datalist
import os
import pickle
from pathlib import Path

import numpy as np
from monai.data import DataLoader, DistributedSampler, load_decathlon_datalist
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    SpatialCropd,
    ToTensord,
)

from monai.transforms import RandomizableTransform
import torch.utils.data as torch_data


from volumeprobe_lite import (
    VolumeProbeLiteBuilder,
    VolumeProbeLiteConfig,
    load_probe_cache,
    build_probe_preprocess,
)


def _as_3tuple(x, default=(1.5, 1.5, 1.5)):
    if x is None:
        return tuple(default)
    if isinstance(x, (list, tuple)):
        if len(x) != 3:
            raise ValueError(f"Expected length-3 tuple/list, got {x}")
        return tuple(float(v) for v in x)
    return (float(x), float(x), float(x))


def _get_probe_cfg(args):
    coarse_grid = getattr(args, "probe_coarse_grid", (4, 4, 2))
    if not isinstance(coarse_grid, (list, tuple)):
        coarse_grid = (4, 4, 2)
    return VolumeProbeLiteConfig(
        preprocessing="voco",
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
    return Path(args.probe_cache_dir)


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
    if num_missing > 0 and getattr(args, "probe_shape_mismatch_mode", "raise") == "raise":
        raise FileNotFoundError(
            f"{num_missing} images have no probe cache in {cache_dir}. Build the cache first."
        )
    if num_missing > 0:
        print(
            f"Warning: {num_missing}/{len(out)} samples have no existing probe cache. They will fall back to random crop."
        )
    else:
        print(f"All {len(out)} samples found existing probe cache.")
    return out


class VoCoAugmentation(RandomizableTransform):
    def __init__(self, args, aug):
        super().__init__(prob=1.0)
        self.args = args
        self.aug = aug
        self.set_random_state(seed=getattr(args, "seed", 0) + getattr(args, "rank", 0))
        self.cache_records = {}
        self.crops_trans = get_crop_transform(roi_small=self.args.roi_x, aug=self.aug)

    def _query_centers(self, x_in):
        if not self.aug or not getattr(self.args, "use_volumeprobe_lite", True):
            return None
        path = x_in.get("probe_cache_path")
        try:
            if path not in self.cache_records:
                self.cache_records[path] = load_probe_cache(path)
            record = self.cache_records[path]
            if record.get("config", {}).get("preprocessing") != "voco":
                raise ValueError("VoCo requires a cache built with --preprocessing voco.")
            if tuple(record["shape_after_preprocess"]) != tuple(x_in["image"].shape[1:]):
                raise ValueError("VoCo probe cache shape does not match the resized foreground.")
            centers = []
            jitter = int(round(96 * getattr(self.args, "probe_jitter_frac", 0.05)))
            for _ in range(self.args.sw_batch_size):
                if self.R.rand() >= getattr(self.args, "probe_candidate_prob", 0.7):
                    centers.append(None)
                    continue
                center, _ = VolumeProbeLiteBuilder.sample_center_from_cache(record, rng=self.R)
                offset = self.R.randint(-jitter, jitter + 1, size=2)
                centers.append(tuple(np.clip(np.asarray(center[:2]) + offset, 48, 335).astype(int)))
            return centers
        except (OSError, TypeError, ValueError, KeyError) as exc:
            if getattr(self.args, "probe_shape_mismatch_mode", "raise") == "raise":
                raise RuntimeError(f"Invalid VoCo probe cache: {path}; rebuild the cache.") from exc
            print(f"Invalid VoCo probe cache: {path}; using random query centers ({exc}).")
            return None

    def __call__(self, x_in):

        crops_trans = self.crops_trans

        vanilla_trans, labels = get_vanilla_transform(
            num=self.args.sw_batch_size,
            roi_small=self.args.roi_x,
            aug=self.aug,
            centers=self._query_centers(x_in),
        )

        imgs = []
        for trans in vanilla_trans:
            imgs.append(trans(x_in))

        crops = []
        for trans in crops_trans:
            crops.append(trans(x_in))

        return imgs, labels, crops


def get_vanilla_transform(
    num=2, num_crops=4, roi_small=64, roi=96, max_roi=384, aug=False, centers=None
):
    vanilla_trans = []
    labels = []
    for i in range(num):
        center_x, center_y, label = get_position_label(
            roi=roi,
            max_roi=max_roi,
            num_crops=num_crops,
            center=None if centers is None else centers[i],
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
                        mode="bilinear",
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
                        mode="bilinear",
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
                            mode="bilinear",
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
                            roi_center=[center_x, center_y, center_z],
                            roi_size=[roi, roi, roi],
                        ),
                        Resized(
                            keys=["image"],
                            mode="bilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small),
                        ),
                        ToTensord(keys=["image"]),
                    ]
                )
            voco_trans.append(trans)
    return voco_trans


def get_position_label(roi=96, base_roi=96, max_roi=384, num_crops=4, center=None):
    half = roi // 2
    if center is None:
        center_x = np.random.randint(low=half, high=max_roi - half)
        center_y = np.random.randint(low=half, high=max_roi - half)
    else:
        center_x, center_y = center

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


def _build_deterministic_transform(args):
    return build_probe_preprocess(_get_probe_cfg(args))


def _build_random_transform(args):
    return Compose([VoCoAugmentation(args, aug=True)])


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
    return Compose([_build_deterministic_transform(args), VoCoAugmentation(args, aug=False)])


def _make_train_dataset(datalist, args, num_workers):
    det_transform = _build_deterministic_transform(args)
    rand_transform = _build_random_transform(args)

    print("Using PersistentDataset (det only) + TwoStageDataset")
    det_ds = PersistentDataset(
        data=datalist,
        transform=det_transform,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
        cache_dir=args.cache_dir,
    )
    return TwoStageDataset(det_ds, rand_transform)


def get_loader(args):
    args.dataset_variant = "full"
    list_dir = args.list_dir
    num_workers = args.num_workers
    if args.datalist:
        datalist = read_datalist(args.datalist, args.data_dir)
    else:
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
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": args.prefetch_factor} if num_workers > 0 else {}),
    )
    return train_loader
