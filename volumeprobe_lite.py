from __future__ import annotations

import hashlib
import math
import os
import pickle
import traceback
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
)


@dataclass
class VolumeProbeLiteConfig:
    preprocessing: str = "standard"

    axcodes: str = "RAS"
    pixdim: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    spacing_mode: str = "bilinear"
    a_min: float = -175.0
    a_max: float = 250.0
    b_min: float = 0.0
    b_max: float = 1.0
    intensity_clip: bool = False
    fg_threshold: float = 0.30

    coarse_grid: Tuple[int, int, int] = (4, 4, 2)
    top_coarse_bins: int = 24
    fine_offset_frac: float = 0.25
    top_k: int = 40

    roi_size: Optional[Tuple[int, int, int]] = (64, 64, 64)

    alpha: float = 1.0
    beta: float = 1.0
    min_center_distance_norm: Optional[float] = None
    min_center_distance_factor: float = 0.35

    entropy_bins: int = 32


def _foreground_mask(image, threshold):
    return image > threshold


def build_probe_preprocess(cfg):
    if cfg.preprocessing not in {"standard", "voco"}:
        raise ValueError(f"Unknown preprocessing profile: {cfg.preprocessing}")
    voco = cfg.preprocessing == "voco"
    transforms = [
        LoadImaged(keys=["image"], image_only=True, dtype=np.int16 if voco else np.float32),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=cfg.axcodes),
    ]
    if not voco:
        transforms.append(Spacingd(keys=["image"], pixdim=cfg.pixdim, mode=cfg.spacing_mode))
    transforms.extend(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg.a_min,
                a_max=cfg.a_max,
                b_min=cfg.b_min,
                b_max=cfg.b_max,
                clip=cfg.intensity_clip,
            ),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                select_fn=partial(_foreground_mask, threshold=cfg.fg_threshold),
                start_coord_key="foreground_start_coord",
                end_coord_key="foreground_end_coord",
            ),
        ]
    )
    if voco:
        transforms.append(
            Resized(keys="image", mode="bilinear", align_corners=True, spatial_size=(384, 384, 96))
        )
    return Compose(transforms)


class VolumeProbeLiteBuilder:
    def __init__(self, cfg: VolumeProbeLiteConfig):
        self.cfg = cfg
        self.preprocess = build_probe_preprocess(cfg)

    def build_and_cache_all(
        self,
        datalist: Sequence[Dict],
        cache_dir: str | os.PathLike,
        overwrite: bool = False,
        verbose: bool = True,
        failed_log_path: Optional[str | os.PathLike] = None,
        failure_mode: str = "continue",
    ) -> List[Path]:
        """Build static caches for all samples."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        log_path = Path(failed_log_path) if failed_log_path is not None else None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        saved_files: List[Path] = []
        total = len(datalist)
        num_failed = 0
        for idx, item in enumerate(datalist):
            image_path = str(item["image"])
            cache_path = self._cache_path(cache_dir, image_path)
            if cache_path.exists() and not overwrite:
                saved_files.append(cache_path)
                if verbose:
                    print(f"[{idx + 1:04d}/{total:04d}] skip existing: {cache_path.name}")
                continue

            try:
                record = self.build_single(item)
                with open(cache_path, "wb") as f:
                    pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
                saved_files.append(cache_path)
                if verbose:
                    print(f"[{idx + 1:04d}/{total:04d}] saved: {cache_path.name}")
            except Exception as e:
                num_failed += 1
                err = {
                    "image_path": image_path,
                    "sample_name": Path(image_path).name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                }
                if log_path is not None:
                    import json

                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                if verbose:
                    print(
                        f"[{idx + 1:04d}/{total:04d}] failed and skipped: {err['sample_name']} "
                        f"({err['error_type']}: {err['error_message']})"
                    )
                if str(failure_mode).lower() != "continue":
                    raise

        if verbose and num_failed > 0:
            if log_path is not None:
                print(f"Finished with {num_failed} skipped sample(s). Failure log: {log_path}")
            else:
                print(f"Finished with {num_failed} skipped sample(s).")
        return saved_files

    def build_single(self, item: Dict) -> Dict:
        data = {"image": item["image"]}
        out = self.preprocess(data)
        image = out["image"]
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Expected image shape [1, H, W, D], got {image.shape}")

        volume = image[0].astype(np.float32)
        shape = np.asarray(volume.shape, dtype=np.int32)
        grad_mag = self._compute_gradient_magnitude(volume)

        coarse_candidates = self._score_coarse_bins(volume, grad_mag)
        coarse_candidates = sorted(coarse_candidates, key=lambda x: x["static_score"], reverse=True)
        coarse_candidates = coarse_candidates[: self.cfg.top_coarse_bins]

        fine_candidates = self._score_fine_candidates(volume, grad_mag, coarse_candidates)
        topk_candidates = self._select_topk_candidates(fine_candidates, shape)

        record = {
            "image_path": item["image"],
            "shape_after_preprocess": tuple(int(v) for v in shape.tolist()),
            "foreground_start_coord": tuple(
                int(v) for v in np.asarray(out["foreground_start_coord"]).tolist()
            ),
            "foreground_end_coord": tuple(
                int(v) for v in np.asarray(out["foreground_end_coord"]).tolist()
            ),
            "config": asdict(self.cfg),
            "coarse_candidates": coarse_candidates,
            "topk_candidates": topk_candidates,
        }
        return record

    def _score_coarse_bins(self, volume: np.ndarray, grad_mag: np.ndarray) -> List[Dict]:
        shape = np.asarray(volume.shape, dtype=np.int32)
        gx, gy, gz = self.cfg.coarse_grid

        raw = []
        for ix in range(gx):
            for iy in range(gy):
                for iz in range(gz):
                    bounds = self._coarse_bin_bounds(shape, (ix, iy, iz))
                    patch = self._crop_by_bounds(volume, bounds)
                    grad_patch = self._crop_by_bounds(grad_mag, bounds)
                    norm_center = self._coarse_bin_center_norm((ix, iy, iz))

                    grad_score = float(grad_patch.mean()) if grad_patch.size > 0 else 0.0
                    ent_score = self._entropy(patch)
                    loc_score = self._location_prior(norm_center)
                    margin_score = self._margin_score(norm_center)
                    raw.append(
                        {
                            "bin_id": self._flatten_bin_id((ix, iy, iz)),
                            "bin_index": (ix, iy, iz),
                            "bounds": bounds,
                            "norm_center": tuple(float(v) for v in norm_center),
                            "grad_raw": grad_score,
                            "entropy_raw": ent_score,
                            "loc_score": loc_score,
                            "margin_score": margin_score,
                        }
                    )

        grad_norm = self._normalize([x["grad_raw"] for x in raw])
        ent_norm = self._normalize([x["entropy_raw"] for x in raw])

        out = []
        for i, x in enumerate(raw):
            anatomy = x["loc_score"] + x["margin_score"]
            informative = grad_norm[i] + ent_norm[i]
            static_score = self._compose_static_score(
                anatomy_aware=anatomy, informativeness=informative
            )
            out.append(
                {
                    "bin_id": x["bin_id"],
                    "bin_index": x["bin_index"],
                    "bounds": x["bounds"],
                    "norm_center": x["norm_center"],
                    "anatomy_aware": float(anatomy),
                    "informativeness": float(informative),
                    "static_score": float(static_score),
                    "lite_score": float(static_score),
                    "grad_score": float(grad_norm[i]),
                    "entropy_score": float(ent_norm[i]),
                }
            )
        return out

    def _score_fine_candidates(
        self,
        volume: np.ndarray,
        grad_mag: np.ndarray,
        coarse_candidates: Sequence[Dict],
    ) -> List[Dict]:
        shape = np.asarray(volume.shape, dtype=np.int32)
        raw = []

        for coarse in coarse_candidates:
            ix, iy, iz = coarse["bin_index"]
            bounds = coarse["bounds"]
            centers = self._fine_centers_in_bin(shape, (ix, iy, iz), bounds)
            for candidate_id, norm_center in enumerate(centers):
                voxel_center = self._norm_to_voxel(norm_center, shape)
                patch_size = self._get_eval_patch_size(shape, bounds)
                patch = self._crop_by_center(volume, voxel_center, patch_size)
                grad_patch = self._crop_by_center(grad_mag, voxel_center, patch_size)

                grad_score = float(grad_patch.mean()) if grad_patch.size > 0 else 0.0
                ent_score = self._entropy(patch)
                loc_score = self._location_prior(norm_center)
                margin_score = self._margin_score(norm_center)

                raw.append(
                    {
                        "candidate_id": candidate_id,
                        "bin_id": coarse["bin_id"],
                        "bin_index": coarse["bin_index"],
                        "norm_center": tuple(float(v) for v in norm_center),
                        "voxel_center": tuple(int(v) for v in voxel_center),
                        "grad_raw": grad_score,
                        "entropy_raw": ent_score,
                        "loc_score": loc_score,
                        "margin_score": margin_score,
                    }
                )

        grad_norm = self._normalize([x["grad_raw"] for x in raw])
        ent_norm = self._normalize([x["entropy_raw"] for x in raw])

        out = []
        for i, x in enumerate(raw):
            anatomy = x["loc_score"] + x["margin_score"]
            informative = grad_norm[i] + ent_norm[i]
            static_score = self._compose_static_score(
                anatomy_aware=anatomy, informativeness=informative
            )
            out.append(
                {
                    "candidate_id": x["candidate_id"],
                    "bin_id": x["bin_id"],
                    "bin_index": x["bin_index"],
                    "norm_center": x["norm_center"],
                    "voxel_center": x["voxel_center"],
                    "anatomy_aware": float(anatomy),
                    "informativeness": float(informative),
                    "static_score": float(static_score),
                    "lite_score": float(static_score),
                    "grad_score": float(grad_norm[i]),
                    "entropy_score": float(ent_norm[i]),
                }
            )
        return sorted(out, key=lambda x: x["static_score"], reverse=True)

    def _select_topk_candidates(self, candidates: Sequence[Dict], shape: np.ndarray) -> List[Dict]:
        if len(candidates) == 0:
            return []
        return self._greedy_diverse_topk(candidates, shape)

    def _greedy_diverse_topk(self, candidates: Sequence[Dict], shape: np.ndarray) -> List[Dict]:
        remaining = [dict(c) for c in candidates]
        selected: List[Dict] = []

        while remaining and len(selected) < self.cfg.top_k:
            best_idx = 0
            best_adjusted = -float("inf")
            best_penalty = 0.0
            for i, cand in enumerate(remaining):
                penalty = self._redundancy_penalty(cand, selected, shape)
                adjusted = float(cand["static_score"]) - float(self.cfg.beta) * penalty
                if adjusted > best_adjusted:
                    best_idx = i
                    best_adjusted = adjusted
                    best_penalty = penalty

            chosen = remaining.pop(best_idx)
            selected.append(self._decorate_selected_candidate(chosen, best_penalty, best_adjusted))

        return selected

    def _decorate_selected_candidate(
        self,
        cand: Dict,
        diversity_penalty: float,
        selection_score: float,
    ) -> Dict:
        out = dict(cand)
        out["diversity_penalty"] = float(diversity_penalty)
        out["redundancy_penalty"] = float(diversity_penalty)
        out["selection_score"] = float(selection_score)
        return out

    def _redundancy_penalty(self, cand: Dict, selected: Sequence[Dict], shape: np.ndarray) -> float:
        if len(selected) == 0:
            return 0.0
        c = np.asarray(cand["norm_center"], dtype=np.float32)
        delta = self._get_min_center_distance_delta(shape)
        penalties = []
        for prev in selected:
            p = np.asarray(prev["norm_center"], dtype=np.float32)
            dist = np.abs(c - p)
            axis_close = np.clip(1.0 - dist / (delta + 1e-8), 0.0, 1.0)
            penalties.append(float(np.prod(axis_close)))
        return float(max(penalties)) if penalties else 0.0

    @staticmethod
    def sample_center_from_cache(
        cache_record: Dict,
        rng: Optional[np.random.RandomState] = None,
        sampling_mode: str = "uniform",
    ) -> Tuple[Tuple[int, int, int], Dict]:
        if rng is None:
            rng = np.random.RandomState()

        candidates = cache_record.get("topk_candidates", [])
        if len(candidates) == 0:
            raise ValueError("No cached candidates found.")

        mode = str(sampling_mode).strip().lower()
        if mode in {"uniform", "bank", "static"}:
            idx = int(rng.randint(len(candidates)))
        elif mode in {"score", "weighted", "score_proportional"}:
            scores = np.asarray(
                [float(c.get("static_score", c.get("lite_score", 0.0))) for c in candidates],
                dtype=np.float32,
            )
            scores = scores - scores.min()
            if float(scores.sum()) <= 1e-8:
                probs = np.full(len(candidates), 1.0 / len(candidates), dtype=np.float32)
            else:
                probs = scores / scores.sum()
            idx = int(rng.choice(len(candidates), p=probs))
        else:
            raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

        chosen = candidates[idx]
        return tuple(int(v) for v in chosen["voxel_center"]), chosen

    def _compose_static_score(self, anatomy_aware: float, informativeness: float) -> float:
        return float(anatomy_aware + self.cfg.alpha * informativeness)

    def _cache_path(self, cache_dir: Path, image_path: str) -> Path:
        stem = Path(image_path).name
        digest = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:10]
        return cache_dir / f"{stem}.{digest}.{self.cfg.preprocessing}.probe.pkl"

    def _coarse_bin_center_norm(self, idx: Tuple[int, int, int]) -> Tuple[float, float, float]:
        gx, gy, gz = self.cfg.coarse_grid
        ix, iy, iz = idx
        return (
            (ix + 0.5) / gx,
            (iy + 0.5) / gy,
            (iz + 0.5) / gz,
        )

    def _coarse_bin_bounds(
        self, shape: np.ndarray, idx: Tuple[int, int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        gx, gy, gz = self.cfg.coarse_grid
        ix, iy, iz = idx
        x_edges = np.linspace(0, int(shape[0]), gx + 1, dtype=np.int32)
        y_edges = np.linspace(0, int(shape[1]), gy + 1, dtype=np.int32)
        z_edges = np.linspace(0, int(shape[2]), gz + 1, dtype=np.int32)
        return (
            (int(x_edges[ix]), int(x_edges[ix + 1])),
            (int(y_edges[iy]), int(y_edges[iy + 1])),
            (int(z_edges[iz]), int(z_edges[iz + 1])),
        )

    def _fine_centers_in_bin(
        self,
        shape: np.ndarray,
        idx: Tuple[int, int, int],
        bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ) -> List[Tuple[float, float, float]]:
        cx, cy, cz = self._coarse_bin_center_norm(idx)
        gx, gy, gz = self.cfg.coarse_grid

        dx = self.cfg.fine_offset_frac / gx
        dy = self.cfg.fine_offset_frac / gy
        dz = self.cfg.fine_offset_frac / gz

        (x0, x1), (y0, y1), (z0, z1) = bounds
        eps = 1e-4
        xmin = (x0 + 0.5) / float(shape[0])
        xmax = (x1 - 0.5) / float(shape[0])
        ymin = (y0 + 0.5) / float(shape[1])
        ymax = (y1 - 0.5) / float(shape[1])
        zmin = (z0 + 0.5) / float(shape[2])
        zmax = (z1 - 0.5) / float(shape[2])

        candidates = [(cx, cy, cz)]
        if (xmax - xmin) > 2.0 * (dx + eps):
            candidates += [(cx - dx, cy, cz), (cx + dx, cy, cz)]
        if (ymax - ymin) > 2.0 * (dy + eps):
            candidates += [(cx, cy - dy, cz), (cx, cy + dy, cz)]
        if (zmax - zmin) > 2.0 * (dz + eps):
            candidates += [(cx, cy, cz - dz), (cx, cy, cz + dz)]

        clamped = []
        seen = set()
        for x, y, z in candidates:
            p = (
                float(np.clip(x, xmin + eps, xmax - eps)),
                float(np.clip(y, ymin + eps, ymax - eps)),
                float(np.clip(z, zmin + eps, zmax - eps)),
            )
            key = tuple(np.round(np.asarray(p, dtype=np.float32), 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            clamped.append(p)
        return clamped

    def _flatten_bin_id(self, idx: Tuple[int, int, int]) -> int:
        gx, gy, gz = self.cfg.coarse_grid
        ix, iy, iz = idx
        return ix * (gy * gz) + iy * gz + iz

    def _norm_to_voxel(
        self, norm_center: Sequence[float], shape: np.ndarray
    ) -> Tuple[int, int, int]:
        voxel = []
        for c, n in zip(shape, norm_center):
            v = int(round(float(n) * (int(c) - 1)))
            v = int(np.clip(v, 0, int(c) - 1))
            voxel.append(v)
        return tuple(voxel)

    def _get_effective_roi_size(self, shape: np.ndarray) -> np.ndarray:
        if self.cfg.roi_size is not None:
            return np.minimum(
                np.asarray(self.cfg.roi_size, dtype=np.float32), shape.astype(np.float32)
            )
        return np.maximum(
            shape.astype(np.float32) / np.asarray(self.cfg.coarse_grid, dtype=np.float32) * 0.5, 8.0
        )

    def _get_min_center_distance_delta(self, shape: np.ndarray) -> np.ndarray:
        if self.cfg.min_center_distance_norm is not None:
            return np.full(3, float(self.cfg.min_center_distance_norm), dtype=np.float32)
        roi_size = self._get_effective_roi_size(shape)
        delta = (roi_size / np.maximum(shape.astype(np.float32), 1.0)) * float(
            self.cfg.min_center_distance_factor
        )
        return delta.astype(np.float32)

    def _get_eval_patch_size(
        self,
        shape: np.ndarray,
        bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ) -> Tuple[int, int, int]:
        if self.cfg.roi_size is not None:
            roi = np.minimum(np.asarray(self.cfg.roi_size, dtype=np.int32), shape)
            return tuple(int(v) for v in roi.tolist())

        bin_size = np.asarray(
            [
                bounds[0][1] - bounds[0][0],
                bounds[1][1] - bounds[1][0],
                bounds[2][1] - bounds[2][0],
            ],
            dtype=np.int32,
        )
        patch = np.maximum(bin_size // 2, 8)
        return tuple(int(v) for v in patch.tolist())

    @staticmethod
    def _crop_by_bounds(
        arr: np.ndarray,
        bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ) -> np.ndarray:
        (x0, x1), (y0, y1), (z0, z1) = bounds
        return arr[x0:x1, y0:y1, z0:z1]

    @staticmethod
    def _crop_by_center(arr: np.ndarray, center: Sequence[int], size: Sequence[int]) -> np.ndarray:
        shape = np.asarray(arr.shape, dtype=np.int32)
        center = np.asarray(center, dtype=np.int32)
        size = np.asarray(size, dtype=np.int32)
        half = size // 2

        start = center - half
        end = start + size

        start = np.maximum(start, 0)
        end = np.minimum(end, shape)

        for axis in range(3):
            cur = end[axis] - start[axis]
            if cur < size[axis]:
                deficit = int(size[axis] - cur)
                shift_left = min(int(start[axis]), deficit // 2 + deficit % 2)
                shift_right = min(int(shape[axis] - end[axis]), deficit - shift_left)
                start[axis] -= shift_left
                end[axis] += shift_right
                cur = end[axis] - start[axis]
                if cur < size[axis]:
                    if start[axis] == 0:
                        end[axis] = min(int(shape[axis]), int(size[axis]))
                    else:
                        start[axis] = max(0, int(shape[axis] - size[axis]))
                        end[axis] = int(shape[axis])

        x0, y0, z0 = [int(v) for v in start]
        x1, y1, z1 = [int(v) for v in end]
        return arr[x0:x1, y0:y1, z0:z1]

    @staticmethod
    def _compute_gradient_magnitude(arr: np.ndarray) -> np.ndarray:
        gx, gy, gz = np.gradient(arr.astype(np.float32), edge_order=1)
        grad = np.sqrt(gx * gx + gy * gy + gz * gz)
        return grad.astype(np.float32)

    def _entropy(self, patch: np.ndarray) -> float:
        if patch.size == 0:
            return 0.0
        hist, _ = np.histogram(
            patch.astype(np.float32),
            bins=self.cfg.entropy_bins,
            range=(self.cfg.b_min, self.cfg.b_max),
        )
        prob = hist.astype(np.float64)
        prob = prob / np.clip(prob.sum(), 1e-12, None)
        prob = prob[prob > 0]
        if prob.size == 0:
            return 0.0
        ent = -np.sum(prob * np.log(prob + 1e-12))
        ent = ent / math.log(self.cfg.entropy_bins)
        return float(ent)

    @staticmethod
    def _normalize(values: Sequence[float]) -> List[float]:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return []
        lo = float(arr.min())
        hi = float(arr.max())
        if hi - lo < 1e-8:
            return [0.0 for _ in values]
        return [float(v) for v in ((arr - lo) / (hi - lo + 1e-8)).tolist()]

    @staticmethod
    def _margin_score(norm_center: Sequence[float]) -> float:
        c = np.asarray(norm_center, dtype=np.float32)
        margin = np.minimum(c, 1.0 - c)
        score = 2.0 * float(margin.min())
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _location_prior(norm_center: Sequence[float]) -> float:
        c = np.asarray(norm_center, dtype=np.float32)
        centered = (c - 0.5) / 0.5
        weights = np.asarray([1.0, 1.0, 0.5], dtype=np.float32)
        dist = math.sqrt(float(np.sum(weights * centered * centered) / np.sum(weights)))
        score = 1.0 - dist
        return float(np.clip(score, 0.0, 1.0))


def load_probe_cache(cache_path: str | os.PathLike) -> Dict:
    with open(cache_path, "rb") as f:
        return pickle.load(f)
