from pretraining.data import PersistentDataset, read_datalist
from collections.abc import Mapping
import os
import pickle
from pathlib import Path

import monai
import numpy as np
import torch.utils.data as torch_data
from monai.config import KeysCollection
from monai.data import DataLoader, DistributedSampler, load_decathlon_datalist
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandomizableTransform,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
)

from volumeprobe_lite import (
    VolumeProbeLiteBuilder,
    VolumeProbeLiteConfig,
    load_probe_cache,
)


def threshold_infer(x, td=0.3):
    return x > td


def threshold(x, td=0.0):
    return x > td


class MaskLabel:
    def __call__(self, x_in):
        image = x_in["image"]
        image[image > 0] = 1
        x_in["mask_label"] = image
        return x_in


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

    roi_x = int(getattr(args, "roi_x", getattr(args, "roi_large", 96)))
    roi_y = int(getattr(args, "roi_y", getattr(args, "roi_large", 96)))
    roi_z = int(getattr(args, "roi_z", 96))

    return VolumeProbeLiteConfig(
        axcodes=str(getattr(args, "probe_axcodes", "RAS")),
        pixdim=_as_3tuple(getattr(args, "probe_pixdim", (1.5, 1.5, 1.5))),
        spacing_mode=str(getattr(args, "probe_spacing_mode", "bilinear")),
        a_min=float(getattr(args, "a_min", -175.0)),
        a_max=float(getattr(args, "a_max", 250.0)),
        b_min=float(getattr(args, "b_min", 0.0)),
        b_max=float(getattr(args, "b_max", 1.0)),
        intensity_clip=True,
        fg_threshold=float(getattr(args, "probe_fg_threshold", 0.3)),
        coarse_grid=tuple(int(v) for v in coarse_grid),
        top_coarse_bins=int(getattr(args, "probe_top_coarse_bins", 12)),
        fine_offset_frac=float(getattr(args, "probe_fine_offset_frac", 0.25)),
        top_k=int(getattr(args, "probe_top_k", 16)),
        roi_size=(roi_x, roi_y, roi_z),
        alpha=float(getattr(args, "probe_alpha", 1.0)),
        beta=float(getattr(args, "probe_beta", 1.0)),
        min_center_distance_norm=getattr(args, "probe_min_center_distance_norm", None),
        min_center_distance_factor=float(getattr(args, "probe_min_center_distance_factor", 0.35)),
        entropy_bins=int(getattr(args, "probe_entropy_bins", 32)),
    )


def _get_train_roi_size(args):
    return (
        int(getattr(args, "roi_large", getattr(args, "roi_x", 96))),
        int(getattr(args, "roi_large", getattr(args, "roi_y", 96))),
        int(getattr(args, "roi_z", 96)),
    )


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
            f"Warning: {num_missing}/{len(out)} samples have no existing probe cache. "
            "They will fall back to random crop."
        )
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
                st -= ed - dim
                ed = dim
            st = max(st, 0)
            ed = min(ed, dim)
            starts.append(st)
            ends.append(ed)

        patch = img[:, starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]]
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
                        center, chosen_info = VolumeProbeLiteBuilder.sample_center_from_cache(
                            cache_record=cache_record,
                            rng=self.R,
                        )

                        self._rand_use_candidate = True
                        self._rand_chosen_info = chosen_info
                else:
                    center, chosen_info = VolumeProbeLiteBuilder.sample_center_from_cache(
                        cache_record=cache_record,
                        rng=self.R,
                    )

                    self._rand_use_candidate = True
                    self._rand_chosen_info = chosen_info
            except Exception as e:
                if self.shape_mismatch_mode == "raise":
                    raise
                print(
                    f"[AdaptiveRandomMixCropd] candidate sampling failed for {cache_path}: {e}. "
                    "Fall back to random crop."
                )
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
            if self._rand_cache_shape is None
            else self._rand_cache_shape
        )
        d["adaptive_crop_current_shape"] = (
            np.asarray([-1, -1, -1], dtype=np.int32)
            if self._rand_current_shape is None
            else self._rand_current_shape
        )
        d["adaptive_crop_bin_id"] = (
            -1 if self._rand_chosen_info is None else int(self._rand_chosen_info["bin_id"])
        )
        d["adaptive_crop_score"] = (
            -1.0 if self._rand_chosen_info is None else float(self._rand_chosen_info["lite_score"])
        )
        return d


class TwoStageDataset(torch_data.Dataset):
    def __init__(self, det_ds, rand_transform):
        self.det_ds = det_ds
        self.rand_transform = rand_transform

    def __len__(self):
        return len(self.det_ds)

    def __getitem__(self, idx):
        item = self.det_ds[idx]
        return self.rand_transform(item)


def _strip_to_image_only(items):
    out = []
    for item in items:
        out.append({"image": item["image"]})
    return out


def _loader_kwargs(args, sampler=None, shuffle=False, drop_last=True):
    num_workers = int(getattr(args, "num_workers", 0))
    kwargs = dict(
        batch_size=int(getattr(args, "batch_size", 1)),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))
    return kwargs


def _make_sampler(dataset, args, shuffle=True):
    if bool(getattr(args, "distributed", False)):
        return DistributedSampler(
            dataset=dataset,
            even_divisible=True,
            shuffle=shuffle,
            rank=int(getattr(args, "rank", 0)),
            num_replicas=int(getattr(args, "world_size", 1)),
        )
    return None


def _make_train_dataset(datalist, det_transform, rand_transform, args, persistent_suffix):

    print("Using PersistentDataset (det only) + TwoStageDataset")
    det_ds = PersistentDataset(
        data=datalist,
        transform=det_transform,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
        cache_dir=args.cache_dir,
    )
    return TwoStageDataset(det_ds, rand_transform)


def _build_ct_deterministic_transform(args, use_infer_threshold=False):
    probe_cfg = _get_probe_cfg(args)
    select_fn = threshold_infer if use_infer_threshold else threshold
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=True, dtype=np.float32),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes=probe_cfg.axcodes),
            Spacingd(keys=["image"], pixdim=probe_cfg.pixdim, mode=probe_cfg.spacing_mode),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(getattr(args, "a_min", -175.0)),
                a_max=float(getattr(args, "a_max", 250.0)),
                b_min=float(getattr(args, "b_min", 0.0)),
                b_max=float(getattr(args, "b_max", 1.0)),
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image", select_fn=select_fn),
        ]
    )


def _build_mri_deterministic_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image"),
            NormalizeIntensityd(keys=["image"]),
        ]
    )


def _build_pet_deterministic_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"]),
        ]
    )


def _build_random_transform(args, post_aug):
    train_roi_size = _get_train_roi_size(args)
    return Compose(
        [
            AdaptiveRandomMixCropd(
                keys=["image"],
                roi_size=train_roi_size,
                p_candidate=float(getattr(args, "probe_candidate_prob", 0.7)),
                jitter_frac=float(getattr(args, "probe_jitter_frac", 0.05)),
                cache_path_key="probe_cache_path",
                check_shape_match=bool(getattr(args, "probe_check_shape_match", True)),
                shape_mismatch_mode=str(
                    getattr(args, "probe_shape_mismatch_mode", "warn_and_random")
                ),
                seed=args.seed + args.rank,
            ),
            Resized(
                keys=["image"],
                mode="trilinear",
                align_corners=True,
                spatial_size=train_roi_size,
            ),
            post_aug,
        ]
    )


def _build_center_val_transform(det_transform, post_aug, args):
    train_roi_size = _get_train_roi_size(args)
    return Compose(
        [
            det_transform,
            CenterSpatialCropd(keys=["image"], roi_size=train_roi_size),
            Resized(
                keys=["image"],
                mode="trilinear",
                align_corners=True,
                spatial_size=train_roi_size,
            ),
            post_aug,
        ]
    )


def _get_json_root(args):
    return args.list_dir


def _get_data_root(args):
    return args.data_dir


def _load_ct_1k_datalist(args):
    if args.datalist:
        return read_datalist(args.datalist, args.data_dir)
    list_dir = _get_json_root(args)
    data_root = _get_data_root(args)

    jsonlist1 = os.path.join(list_dir, "btcv.json")
    jsonlist2 = os.path.join(list_dir, "dataset_TCIAcovid19_0.json")

    datadir1 = os.path.join(data_root, "BTCV")
    datadir2 = os.path.join(data_root, "TCIAcovid19")

    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print(f"Dataset 1 BTCV: number of data: {len(datalist1)}")
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print(f"Dataset 2 Covid 19: number of data: {len(datalist2)}")

    datalist = _strip_to_image_only(datalist1) + datalist2
    print(f"Dataset all training: number of data: {len(datalist)}")
    return datalist


def _load_ct_10k_datalist(args):
    if args.datalist:
        return read_datalist(args.datalist, args.data_dir)

    list_dir = args.list_dir
    data_root = args.data_dir

    names = [
        ("btcv.json", "BTCV", True, "Dataset 1 BTCV"),
        ("dataset_TCIAcovid19_0.json", "TCIAcovid19", False, "Dataset 2 Covid 19"),
        ("dataset_LUNA16_0.json", "Luna16", True, "Dataset 3 Luna"),
        ("stoic21.json", "stoic21", False, "Dataset 4 Stoic21"),
        (
            "Totalsegmentator_dataset.json",
            "Totalsegmentator_dataset",
            False,
            "Dataset 5 Totalsegmentator",
        ),
        ("flare23.json", "Flare23", False, "Dataset 6 Flare23"),
        ("HNSCC.json", "HNSCC_convert_v1", False, "Dataset 7 HNSCC"),
    ]

    datalist = []
    for json_name, data_name, strip_image_only, label in names:
        json_path = os.path.join(list_dir, json_name)
        data_path = os.path.join(data_root, data_name)
        items = load_decathlon_datalist(json_path, False, "training", base_dir=data_path)
        print(f"{label}: number of data: {len(items)}")
        datalist.extend(_strip_to_image_only(items) if strip_image_only else items)

    print(f"Dataset all training: number of data: {len(datalist)}")
    return datalist


def get_infer_loader(args):
    datalist = _load_ct_1k_datalist(args)
    from .geometry import GeoAugmentation

    val_transform = _build_center_val_transform(
        _build_ct_deterministic_transform(args, use_infer_threshold=True),
        GeoAugmentation(args, roi_large=args.roi_large, crop_size=args.roi_z),
        args,
    )

    ds = monai.data.Dataset(data=datalist, transform=val_transform)
    sampler = _make_sampler(ds, args, shuffle=False)
    loader = DataLoader(ds, **_loader_kwargs(args, sampler=sampler, shuffle=False, drop_last=True))
    return loader


def get_1k_loader(args):
    args.dataset_variant = "1k"
    datalist = _load_ct_1k_datalist(args)
    datalist = _attach_existing_probe_cache_paths(datalist, args)

    from .geometry import GeoAugmentation

    det_transform = _build_ct_deterministic_transform(args, use_infer_threshold=True)
    rand_transform = _build_random_transform(
        args,
        GeoAugmentation(args, roi_large=args.roi_large, crop_size=args.roi_z),
    )
    train_ds = _make_train_dataset(datalist, det_transform, rand_transform, args, "S2DC-1k")

    sampler = _make_sampler(train_ds, args, shuffle=True)
    train_loader = DataLoader(
        train_ds, **_loader_kwargs(args, sampler=sampler, shuffle=True, drop_last=True)
    )
    return train_loader


def get_10k_loader(args):
    args.dataset_variant = "full"
    datalist = _load_ct_10k_datalist(args)
    datalist = _attach_existing_probe_cache_paths(datalist, args)

    from .geometry import GeoAugmentation

    det_transform = _build_ct_deterministic_transform(args, use_infer_threshold=True)
    rand_transform = _build_random_transform(
        args,
        GeoAugmentation(args, roi_large=args.roi_large, crop_size=args.roi_z),
    )
    train_ds = _make_train_dataset(datalist, det_transform, rand_transform, args, "S2DC-10k")

    sampler = _make_sampler(train_ds, args, shuffle=True)
    train_loader = DataLoader(
        train_ds, **_loader_kwargs(args, sampler=sampler, shuffle=True, drop_last=True)
    )
    return train_loader


def get_mri_loader(args, training=True):
    data_dir = args.data_dir
    datalist_json = args.datalist
    if not datalist_json:
        raise ValueError("MRI/PET training requires --datalist.")
    from .geometry import GeoAugmentation

    datalist = load_decathlon_datalist(datalist_json, False, "training", base_dir=data_dir)

    if training:
        args.dataset_variant = getattr(args, "dataset_variant", "mri")
        datalist = _attach_existing_probe_cache_paths(datalist, args)
        det_transform = _build_mri_deterministic_transform(args)
        rand_transform = _build_random_transform(
            args,
            GeoAugmentation(args, roi_large=args.roi_large, crop_size=args.roi_z),
        )
        train_ds = _make_train_dataset(datalist, det_transform, rand_transform, args, "S2DC-MRI")
        sampler = _make_sampler(train_ds, args, shuffle=True)
        return DataLoader(
            train_ds, **_loader_kwargs(args, sampler=sampler, shuffle=True, drop_last=True)
        )

    val_transform = _build_center_val_transform(
        _build_mri_deterministic_transform(args),
        GeoAugmentation(args, roi_large=args.roi_large, crop_size=args.roi_z),
        args,
    )
    val_ds = monai.data.Dataset(data=_strip_to_image_only(datalist), transform=val_transform)
    return DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(int(getattr(args, "num_workers", 0)), 2),
        pin_memory=True,
    )


def get_pet_loader(args, training=True):
    data_dir = args.data_dir
    datalist_json = args.datalist
    if not datalist_json:
        raise ValueError("MRI/PET training requires --datalist.")
    from .geometry import GeoAugmentation

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)

    if training:
        args.dataset_variant = getattr(args, "dataset_variant", "pet")
        datalist = _attach_existing_probe_cache_paths(datalist, args)
        det_transform = _build_pet_deterministic_transform(args)
        rand_transform = _build_random_transform(
            args,
            GeoAugmentation(args, roi_large=args.roi_large, crop_size=args.roi_z),
        )
        train_ds = _make_train_dataset(datalist, det_transform, rand_transform, args, "S2DC-PET")
        sampler = _make_sampler(train_ds, args, shuffle=True)
        return DataLoader(
            train_ds, **_loader_kwargs(args, sampler=sampler, shuffle=True, drop_last=True)
        )

    val_transform = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="SAR"),
            NormalizeIntensityd(keys=["image"]),
            CenterSpatialCropd(keys=["image"], roi_size=_get_train_roi_size(args)),
            Resized(
                keys=["image"],
                mode="trilinear",
                align_corners=True,
                spatial_size=_get_train_roi_size(args),
            ),
            GeoAugmentation(args, roi_large=args.roi_large, crop_size=args.roi_z),
        ]
    )
    val_ds = monai.data.Dataset(data=datalist, transform=val_transform)
    sampler = _make_sampler(val_ds, args, shuffle=False)
    return DataLoader(
        val_ds, **_loader_kwargs(args, sampler=sampler, shuffle=False, drop_last=True)
    )
