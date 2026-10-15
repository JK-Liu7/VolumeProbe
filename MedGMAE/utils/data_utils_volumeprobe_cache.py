from pretraining.data import PersistentDataset, read_datalist
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.utils.data as torch_data
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandomizableTransform,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.config import KeysCollection
from volumeprobe_lite import VolumeProbeLiteBuilder, VolumeProbeLiteConfig, load_probe_cache


def _as_3tuple(x, default=(1.5, 1.5, 1.5)):
    if x is None:
        return tuple(default)
    if isinstance(x, (list, tuple)):
        if len(x) != 3:
            raise ValueError(f"Expected length-3 tuple/list, got {x}")
        return tuple(float(v) for v in x)
    return (float(x), float(x), float(x))


def _threshold(x):
    return x > 0.3


def _get_probe_crop_size(args):
    img = int(getattr(args, "img_size", 96))
    return (
        int(getattr(args, "probe_crop_x", img)),
        int(getattr(args, "probe_crop_y", img)),
        int(getattr(args, "probe_crop_z", img)),
    )


def _get_probe_cfg(args):
    coarse_grid = getattr(args, "probe_coarse_grid", (4, 4, 2))
    if not isinstance(coarse_grid, (list, tuple)):
        coarse_grid = (4, 4, 2)
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
        top_coarse_bins=int(getattr(args, "probe_top_coarse_bins", 24)),
        fine_offset_frac=float(getattr(args, "probe_fine_offset_frac", 0.25)),
        top_k=int(getattr(args, "probe_top_k", 40)),
        roi_size=_get_probe_crop_size(args),
        alpha=float(getattr(args, "probe_alpha", 1.0)),
        beta=float(getattr(args, "probe_beta", 1.0)),
        min_center_distance_norm=getattr(args, "probe_min_center_distance_norm", None),
        min_center_distance_factor=float(getattr(args, "probe_min_center_distance_factor", 0.35)),
        entropy_bins=int(getattr(args, "probe_entropy_bins", 32)),
    )


def _strip_to_image_only(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"image": item["image"]} for item in items]


def _get_probe_cache_dir(args):
    return Path(args.probe_cache_dir)


def _attach_existing_probe_cache_paths(
    datalist: List[Dict[str, Any]], args
) -> List[Dict[str, Any]]:
    use_probe = bool(getattr(args, "use_volumeprobe_lite", True))
    if not use_probe:
        return datalist

    cache_dir = _get_probe_cache_dir(args)
    builder = VolumeProbeLiteBuilder(_get_probe_cfg(args))

    out: List[Dict[str, Any]] = []
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
        num_samples=3,
        seed=0,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=1.0)

        self.roi_size = np.asarray(roi_size, dtype=np.int32)
        self.p_candidate = float(np.clip(p_candidate, 0.0, 1.0))
        self.cache_path_key = cache_path_key
        self.check_shape_match = bool(check_shape_match)
        self.shape_mismatch_mode = str(shape_mismatch_mode)

        self._cache_records: Dict[str, Dict[str, Any]] = {}
        self._shape_mismatch_warned = set()
        self.set_random_state(seed=seed)

        self._rand_center = None
        self._rand_use_candidate = False
        self._rand_chosen_info = None
        self._rand_shape_match = None
        self._rand_cache_shape = None
        self._rand_current_shape = None
        self.jitter_frac = float(max(jitter_frac, 0.0))
        self.num_samples = max(int(num_samples), 1)

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
                    f"[AdaptiveRandomMixCropd] candidate sampling failed for {cache_path}: {e}. Fall back to random crop."
                )
                center = self._random_center(spatial_shape)
        else:
            center = self._random_center(spatial_shape)

        center = self._jitter_center(center, spatial_shape)
        self._rand_center = center

    def __call__(self, data):
        d = dict(data)
        cache_path = d.get(self.cache_path_key, None)
        first_key = next(iter(self.key_iterator(d)))
        img0 = d[first_key]
        if not isinstance(img0, np.ndarray):
            img0 = np.asarray(img0)

        outputs = []
        for _ in range(self.num_samples):
            self.randomize(img0, cache_path=cache_path)
            sample_d = dict(d)
            for key in self.key_iterator(d):
                img = d[key]
                if not isinstance(img, np.ndarray):
                    img = np.asarray(img)
                sample_d[key] = self._crop_with_pad(img, self._rand_center)

            sample_d["adaptive_crop_center"] = np.asarray(self._rand_center, dtype=np.int32)
            sample_d["adaptive_crop_source"] = "candidate" if self._rand_use_candidate else "random"
            sample_d["adaptive_crop_bin_id"] = (
                -1 if self._rand_chosen_info is None else int(self._rand_chosen_info["bin_id"])
            )
            sample_d["adaptive_crop_score"] = (
                -1.0
                if self._rand_chosen_info is None
                else float(self._rand_chosen_info["lite_score"])
            )
            outputs.append(sample_d)

        if self.num_samples == 1:
            return outputs[0]
        return outputs


class TwoStageDataset(torch_data.Dataset):
    def __init__(self, det_ds, rand_transform):
        self.det_ds = det_ds
        self.rand_transform = rand_transform

    def __len__(self):
        return len(self.det_ds)

    def __getitem__(self, idx):
        item = self.det_ds[idx]
        return self.rand_transform(item)


class ToImageDict:
    def __call__(self, x_in):
        return {"image": x_in["image"]}


class MedGMAEProbeDatasetFactory:
    def __init__(self, args):
        self.args = args

    def build_datalist(self):
        if self.args.datalist:
            datalist = read_datalist(self.args.datalist, self.args.data_dir)
        else:
            with open(self.args.dataset_train_list, encoding="utf-8") as handle:
                names = [line.strip().split()[0] for line in handle if line.strip()]
            datalist = []
            for name in dict.fromkeys(names):
                path = Path(self.args.data_dir) / "imagesTr" / f"{name}_0000.nii.gz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                datalist.append({"image": str(path.resolve())})
        if not datalist:
            raise ValueError("The training list is empty.")
        return _attach_existing_probe_cache_paths(datalist, self.args)

    def build_train_transform_det(self):
        probe_cfg = _get_probe_cfg(self.args)
        return Compose(
            [
                LoadImaged(keys=["image"], image_only=True, dtype=np.float32),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes=probe_cfg.axcodes),
                Spacingd(keys=["image"], pixdim=probe_cfg.pixdim, mode=probe_cfg.spacing_mode),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=float(getattr(self.args, "a_min", -175.0)),
                    a_max=float(getattr(self.args, "a_max", 250.0)),
                    b_min=float(getattr(self.args, "b_min", 0.0)),
                    b_max=float(getattr(self.args, "b_max", 1.0)),
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=_threshold),
            ]
        )

    def build_train_transform_rand(self):
        return Compose(
            [
                AdaptiveRandomMixCropd(
                    keys=["image"],
                    roi_size=_get_probe_crop_size(self.args),
                    p_candidate=float(getattr(self.args, "probe_candidate_prob", 0.7)),
                    jitter_frac=float(getattr(self.args, "probe_jitter_frac", 0.05)),
                    cache_path_key="probe_cache_path",
                    check_shape_match=bool(getattr(self.args, "probe_check_shape_match", True)),
                    shape_mismatch_mode=str(
                        getattr(self.args, "probe_shape_mismatch_mode", "warn_and_random")
                    ),
                    num_samples=int(getattr(self.args, "num_samples", 3)),
                    seed=self.args.seed + self.args.rank,
                ),
                ToTensord(keys=["image"]),
                ToImageDict(),
            ]
        )

    def build_val_transform(self):
        probe_cfg = _get_probe_cfg(self.args)
        return Compose(
            [
                LoadImaged(keys=["image"], image_only=True, dtype=np.float32),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes=probe_cfg.axcodes),
                Spacingd(keys=["image"], pixdim=probe_cfg.pixdim, mode=probe_cfg.spacing_mode),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=float(getattr(self.args, "a_min", -175.0)),
                    a_max=float(getattr(self.args, "a_max", 250.0)),
                    b_min=float(getattr(self.args, "b_min", 0.0)),
                    b_max=float(getattr(self.args, "b_max", 1.0)),
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=_threshold),
                CenterSpatialCropd(keys=["image"], roi_size=_get_probe_crop_size(self.args)),
                SpatialPadd(keys=["image"], spatial_size=_get_probe_crop_size(self.args)),
                ToTensord(keys=["image"]),
                ToImageDict(),
            ]
        )

    def make_train_dataset(self):
        det_ds = PersistentDataset(
            data=self.build_datalist(),
            transform=self.build_train_transform_det(),
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            cache_dir=self.args.cache_dir,
        )
        return TwoStageDataset(det_ds, self.build_train_transform_rand())

    def make_val_dataset(self):
        datalist = self.build_datalist()
        val_items = datalist[: min(len(datalist), int(getattr(self.args, "vis_val_num", 8)))]
        return PersistentDataset(
            data=val_items,
            transform=self.build_val_transform(),
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            cache_dir=self.args.cache_dir,
        )


def _identity_collate(batch):
    return batch
