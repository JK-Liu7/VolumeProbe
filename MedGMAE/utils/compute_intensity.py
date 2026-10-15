from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint


def _as_matrix(inv_covariances: torch.Tensor) -> torch.Tensor:
    """Convert [N, 9] or [N, 3, 3] inverse covariances to [N, 3, 3]."""
    if inv_covariances.ndim == 3:
        if inv_covariances.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected inv_covariances [...,3,3], got {tuple(inv_covariances.shape)}"
            )
        return inv_covariances
    if inv_covariances.ndim == 2 and inv_covariances.shape[-1] == 9:
        return inv_covariances.view(-1, 3, 3)
    raise ValueError(
        f"inv_covariances must have shape [N, 9] or [N, 3, 3], got {tuple(inv_covariances.shape)}"
    )


def _normalize_inputs(
    gaussian_centers: torch.Tensor,
    points: torch.Tensor,
    intensities: torch.Tensor,
    inv_covariances: torch.Tensor,
    scalings: Optional[torch.Tensor],
):
    device = gaussian_centers.device
    dtype = gaussian_centers.dtype

    points = points.detach().to(device=device, dtype=dtype)
    intensities = intensities.to(device=device, dtype=dtype)
    inv_covariances = inv_covariances.to(device=device, dtype=dtype)
    if scalings is not None:
        scalings = scalings.to(device=device, dtype=dtype)

    if gaussian_centers.ndim != 2 or gaussian_centers.shape[-1] != 3:
        raise ValueError(f"gaussian_centers must be [N, 3], got {tuple(gaussian_centers.shape)}")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must be [M, 3], got {tuple(points.shape)}")
    if intensities.ndim != 2 or intensities.shape[-1] != 1:
        raise ValueError(f"intensities must be [N, 1], got {tuple(intensities.shape)}")

    inv_covariances = _as_matrix(inv_covariances)
    if inv_covariances.shape[0] != gaussian_centers.shape[0]:
        raise ValueError(
            "gaussian_centers and inv_covariances size mismatch: "
            f"{gaussian_centers.shape[0]} vs {inv_covariances.shape[0]}"
        )
    if intensities.shape[0] != gaussian_centers.shape[0]:
        raise ValueError(
            "gaussian_centers and intensities size mismatch: "
            f"{gaussian_centers.shape[0]} vs {intensities.shape[0]}"
        )
    if scalings is not None and scalings.shape != gaussian_centers.shape:
        raise ValueError(
            f"scalings must be [N, 3], got {tuple(scalings.shape)} for centers {tuple(gaussian_centers.shape)}"
        )

    return points, intensities, inv_covariances, scalings


def _point_chunk_size(device: torch.device, num_gaussians: int) -> int:
    if num_gaussians <= 0:
        return 1
    if device.type == "cpu":
        target = 256 * 1024
    else:
        target = 1024 * 1024
    return max(256, int(target // max(num_gaussians, 1)))


def _evaluate_chunk(
    centers,
    points,
    amplitudes,
    inv_covariances,
    scalings,
    grid_size=(96, 96, 96),
    voxel_indices=None,
):
    diff = points[:, None, :] - centers[None, :, :]
    mahal = torch.einsum("mni,nij,mnj->mn", diff, inv_covariances, diff)
    weights = torch.exp(-0.5 * mahal)
    if scalings is not None:
        size = centers.new_tensor(grid_size)
        mean = centers * size
        expand = (2.0 * scalings) * size
        lower = torch.trunc(mean - expand).to(torch.int64).clamp_min(0)
        upper = torch.minimum(torch.trunc(mean + expand).to(torch.int64), size.to(torch.int64) - 1)
        if voxel_indices is None:
            coordinates = points * (size - 1)
            voxel_indices = (coordinates.sign() * torch.floor(coordinates.abs() + 0.5)).to(
                torch.int64
            )
        support_mask = (
            (voxel_indices[:, None, :] >= lower) & (voxel_indices[:, None, :] <= upper)
        ).all(dim=-1)
        weights = weights * support_mask.to(weights.dtype)
    return (weights * amplitudes.view(1, -1)).sum(dim=1, keepdim=True)


def _compute_gaussian_sum_sparse(
    gaussian_centers: torch.Tensor,
    sparse_points: torch.Tensor,
    intensities: torch.Tensor,
    inv_covariances: torch.Tensor,
    scalings: Optional[torch.Tensor],
    grid_size=(96, 96, 96),
    dense=False,
) -> torch.Tensor:
    points, intensities, inv_covariances, scalings = _normalize_inputs(
        gaussian_centers, sparse_points, intensities, inv_covariances, scalings
    )

    num_points = points.shape[0]
    if num_points == 0:
        return torch.zeros((0, 1), device=gaussian_centers.device, dtype=gaussian_centers.dtype)

    num_gaussians = gaussian_centers.shape[0]
    chunk_size = _point_chunk_size(gaussian_centers.device, num_gaussians)
    outputs = []

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        pts = points[start:end]
        indices = None
        if dense:
            linear = torch.arange(start, end, device=points.device)
            indices = torch.stack(
                (
                    linear // (grid_size[1] * grid_size[2]),
                    (linear // grid_size[2]) % grid_size[1],
                    linear % grid_size[2],
                ),
                dim=-1,
            )
        inputs = (gaussian_centers, pts, intensities, inv_covariances, scalings, grid_size, indices)
        if torch.is_grad_enabled() and any(torch.is_tensor(t) and t.requires_grad for t in inputs):
            out = checkpoint(_evaluate_chunk, *inputs, use_reentrant=False)
        else:
            out = _evaluate_chunk(*inputs)
        outputs.append(out)

    return torch.cat(outputs, dim=0)


def compute_intensity(
    gaussian_centers: torch.Tensor,
    grid_points: torch.Tensor,
    intensities: torch.Tensor,
    inv_covariances: torch.Tensor,
    scalings: Optional[torch.Tensor],
    intensity_grid: torch.Tensor,
) -> torch.Tensor:
    if intensity_grid.ndim != 5 or intensity_grid.shape[0] != 1 or intensity_grid.shape[-1] != 1:
        raise ValueError("intensity_grid must have shape [1, D, H, W, 1].")
    grid_size = tuple(int(v) for v in intensity_grid.shape[1:4])
    if (
        min(grid_size) < 1
        or grid_points.shape[-1] != 3
        or grid_points.numel() != intensity_grid.numel() * 3
    ):
        raise ValueError("grid_points must contain one 3D coordinate per intensity-grid voxel.")

    flat_points = grid_points.reshape(-1, 3)
    flat_out = _compute_gaussian_sum_sparse(
        gaussian_centers=gaussian_centers,
        sparse_points=flat_points,
        intensities=intensities,
        inv_covariances=inv_covariances,
        scalings=scalings,
        grid_size=grid_size,
        dense=True,
    )

    contrib = flat_out.view_as(intensity_grid)
    return intensity_grid.to(device=contrib.device, dtype=contrib.dtype) + contrib


def compute_intensity_sparse(
    gaussian_centers: torch.Tensor,
    sparse_grid_points: torch.Tensor,
    intensities: torch.Tensor,
    inv_covariances: torch.Tensor,
    scalings: Optional[torch.Tensor],
    sparse_intensity_values_out: torch.Tensor,
    image_size=(96, 96, 96),
) -> torch.Tensor:
    computed = _compute_gaussian_sum_sparse(
        gaussian_centers=gaussian_centers,
        sparse_points=sparse_grid_points,
        intensities=intensities,
        inv_covariances=inv_covariances,
        scalings=scalings,
        grid_size=image_size,
    )

    return computed


__all__ = ["compute_intensity", "compute_intensity_sparse"]
