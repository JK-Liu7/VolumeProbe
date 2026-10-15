import torch
import numpy as np
from torch import nn

from utils.general_utils import (
    inverse_sigmoid,
    strip_symmetric,
    build_scaling_rotation,
    build_scaling_rotation_inverse,
)


from utils.compute_intensity import compute_intensity, compute_intensity_sparse


def _get_tensor_device(*tensors):
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * x2.new_tensor(0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


class GaussianModel:
    def __init__(self, fea_dim=0, with_motion_mask=True, **kwargs):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance

        self._xyz = torch.empty(0)
        self._cmesh = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.with_motion_mask = with_motion_mask
        if self.with_motion_mask:
            fea_dim += 1
        self.fea_dim = fea_dim
        self.feature = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.density_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def param_names(self):
        return ["_xyz", "_scaling", "_rotation", "_density", "xyz_gradient_accum"]

    @classmethod
    def build_from(cls, gs, **kwargs):
        new_gs = GaussianModel(**kwargs)
        new_gs._xyz = nn.Parameter(gs._xyz)
        new_gs._scaling = nn.Parameter(gs._scaling)
        new_gs._rotation = nn.Parameter(gs._rotation)
        new_gs._density = nn.Parameter(gs._density)
        new_gs.feature = nn.Parameter(gs.feature)
        return new_gs

    @property
    def motion_mask(self):
        if self.with_motion_mask:
            return torch.sigmoid(self.feature[..., -1:])
        return torch.ones_like(self._xyz[..., :1])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_rotation_bias(self, rotation_bias=None):
        rotation_bias = rotation_bias if rotation_bias is not None else 0.0
        return self.rotation_activation(self._rotation + rotation_bias)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1, d_rotation=None, gs_rot_bias=None):
        if d_rotation is not None:
            rotation = quaternion_multiply(self._rotation, d_rotation)
        else:
            rotation = self._rotation
        if gs_rot_bias is not None:
            rotation = rotation / rotation.norm(dim=-1, keepdim=True)
            rotation = quaternion_multiply(gs_rot_bias, rotation)
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)

    def get_covariance_phy(self, scaling_modifier=1, d_rotation=None, gs_rot_bias=None):
        if d_rotation is not None:
            rotation = quaternion_multiply(self._rotation, d_rotation)
        else:
            rotation = self._rotation
        if gs_rot_bias is not None:
            rotation = rotation / rotation.norm(dim=-1, keepdim=True)
            rotation = quaternion_multiply(gs_rot_bias, rotation)
        return strip_symmetric(
            self.covariance_activation(self.get_scaling, scaling_modifier, rotation)
        )

    def get_covariance_inv(self):
        L = build_scaling_rotation_inverse(self.get_scaling, self._rotation)
        actual_covariance_inv = L @ L.transpose(1, 2)
        return actual_covariance_inv

    def state_dict(self):
        return {
            "_xyz": self._xyz,
            "_density": self._density,
            "_scaling": self._scaling,
            "_rotation": self._rotation,
        }

    def load_state_dict(self, state_dict):
        self._xyz = nn.Parameter(state_dict["_xyz"].clone().detach().requires_grad_(True))
        self._density = nn.Parameter(state_dict["_density"].clone().detach().requires_grad_(True))
        self._scaling = nn.Parameter(state_dict["_scaling"].clone().detach().requires_grad_(True))
        self._rotation = nn.Parameter(state_dict["_rotation"].clone().detach().requires_grad_(True))


class StandardGaussianModel(GaussianModel):
    def __init__(self, fea_dim=0, with_motion_mask=True, all_the_same=False):
        super().__init__(fea_dim, with_motion_mask)
        self.all_the_same = all_the_same

    @property
    def get_scaling(self):
        scaling = (
            self._scaling.mean()[None, None].expand_as(self._scaling)
            if self.all_the_same
            else self._scaling.mean(dim=1, keepdim=True).expand_as(self._scaling)
        )
        return self.scaling_activation(scaling)


def create_grid_3d(c, h, w, device=None, dtype=torch.float32):
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, steps=c, device=device, dtype=dtype),
        torch.linspace(0, 1, steps=h, device=device, dtype=dtype),
        torch.linspace(0, 1, steps=w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid


def get_unmasked_grid_points(image_size, pixel_mask):
    c, h, w = image_size
    device = pixel_mask.device

    if pixel_mask.dim() == 5 and pixel_mask.shape[1] == 1:
        mask_3d = pixel_mask[0, 0]
    else:
        raise ValueError(f"Expected pixel_mask with shape [B, 1, C, H, W], got {pixel_mask.shape}")

    grid_coords = create_grid_3d(c, h, w, device=device, dtype=torch.float32)
    unmasked_indices = torch.nonzero(mask_3d, as_tuple=False)
    unmasked_grid_points = grid_coords[
        unmasked_indices[:, 0], unmasked_indices[:, 1], unmasked_indices[:, 2]
    ]

    return unmasked_grid_points, unmasked_indices


def rendering(pc: GaussianModel, image_size):
    img_dim = image_size
    device = pc.get_xyz.device
    grid = create_grid_3d(*img_dim, device=device, dtype=torch.float32)
    grid = grid.unsqueeze(0).repeat(1, 1, 1, 1, 1)

    grid_point = grid.unsqueeze(-2)
    z, x, y = grid_point.shape[1:4]
    density_grid = torch.zeros(
        1, z, x, y, 1, device=device, dtype=torch.float32, requires_grad=True
    )

    inv_covariance = pc.get_covariance_inv()

    density_grid = compute_intensity(
        pc.get_xyz.contiguous(),
        grid_point.contiguous(),
        pc.get_density.contiguous(),
        inv_covariance.contiguous(),
        pc.get_scaling.contiguous(),
        density_grid.contiguous(),
    )

    return density_grid


def rendering_sparse(pc: GaussianModel, image_size, pixel_mask):
    c, h, w = image_size
    device = pc.get_xyz.device

    unmasked_grid_points, unmasked_indices = get_unmasked_grid_points(image_size, pixel_mask)

    if unmasked_grid_points.shape[0] == 0:
        return torch.zeros((1, c, h, w, 1), device=device, dtype=torch.float32)

    xyz = pc.get_xyz.contiguous()
    density = pc.get_density.contiguous()
    scaling = pc.get_scaling.contiguous()
    inv_covariance = pc.get_covariance_inv().contiguous()

    num_unmasked_points = unmasked_grid_points.shape[0]
    sparse_intensity_values = torch.zeros(
        num_unmasked_points, 1, device=device, dtype=torch.float32, requires_grad=True
    )

    sparse_intensity_values = compute_intensity_sparse(
        xyz,
        unmasked_grid_points.contiguous(),
        density,
        inv_covariance,
        scaling,
        sparse_intensity_values,
        image_size=image_size,
    )

    dense_output = torch.zeros((1, c, h, w, 1), device=device, dtype=torch.float32)
    dense_output[0, unmasked_indices[:, 0], unmasked_indices[:, 1], unmasked_indices[:, 2], 0] = (
        sparse_intensity_values.view(-1)
    )

    return dense_output


def rendering2(xyz, scaling, rotation, density, image_size):
    xyz = xyz.float()
    scaling = scaling.float()
    rotation = rotation.float()
    density = density.float()
    device = xyz.device

    img_dim = image_size
    grid = create_grid_3d(*img_dim, device=device, dtype=torch.float32)
    grid = grid.unsqueeze(0).repeat(1, 1, 1, 1, 1)

    grid_point = grid.unsqueeze(-2)
    z, x, y = grid_point.shape[1:4]
    density_grid = torch.zeros(
        1, z, x, y, 1, device=device, dtype=torch.float32, requires_grad=True
    )

    activated_scaling = scaling
    activated_rotation = rotation
    activated_density = density

    L = build_scaling_rotation_inverse(activated_scaling, activated_rotation)
    inv_covariance = L @ L.transpose(1, 2)

    try:
        density_grid = compute_intensity(
            xyz.contiguous(),
            grid_point.contiguous(),
            0.1 * activated_density.contiguous(),
            inv_covariance.contiguous(),
            activated_scaling.contiguous(),
            density_grid.contiguous(),
        )
    except Exception as e:
        print(f"Error rendering: {e}")
        print(f"Positions shape: {xyz.shape}, dtype: {xyz.dtype}, device: {xyz.device}")
        print(
            f"Scales shape: {activated_scaling.shape}, dtype: {activated_scaling.dtype}, device: {activated_scaling.device}"
        )
        print(
            f"Rotations shape: {activated_rotation.shape}, dtype: {activated_rotation.dtype}, device: {activated_rotation.device}"
        )
        print(
            f"Densities shape: {activated_density.shape}, dtype: {activated_density.dtype}, device: {activated_density.device}"
        )
        raise e

    return density_grid


def rendering2_sparse(xyz, scaling, rotation, density, image_size, pixel_mask):
    c, h, w = image_size
    device = xyz.device

    unmasked_grid_points, unmasked_indices = get_unmasked_grid_points(image_size, pixel_mask)

    if unmasked_grid_points.shape[0] == 0:
        return torch.zeros((1, c, h, w, 1), device=device, dtype=torch.float32)

    xyz = xyz.float().contiguous()
    scaling = scaling.float().contiguous()
    rotation = rotation.float().contiguous()
    density = density.float().contiguous()
    unmasked_grid_points = unmasked_grid_points.float().contiguous()

    num_unmasked_points = unmasked_grid_points.shape[0]
    sparse_intensity_values = torch.zeros(
        num_unmasked_points, 1, device=device, dtype=torch.float32, requires_grad=True
    )

    L = build_scaling_rotation_inverse(scaling, rotation)
    inv_covariance = (L @ L.transpose(1, 2)).contiguous()

    sparse_intensity_values = compute_intensity_sparse(
        xyz,
        unmasked_grid_points,
        0.1 * density,
        inv_covariance,
        scaling,
        sparse_intensity_values,
        image_size=image_size,
    )

    dense_output = torch.zeros((1, c, h, w, 1), device=device, dtype=torch.float32)
    dense_output[0, unmasked_indices[:, 0], unmasked_indices[:, 1], unmasked_indices[:, 2], 0] = (
        sparse_intensity_values.view(-1)
    )

    return dense_output


def batch_rendering2(positions, scales, rotations, densities, image_size):
    positions = positions.float()
    scales = scales.float()
    rotations = rotations.float()
    densities = densities.float()

    batch_size = positions.shape[0]
    rendered_images = []

    for b in range(batch_size):
        rendered = rendering2(
            positions[b],
            scales[b],
            rotations[b],
            densities[b],
            image_size,
        )

        c, h, w = image_size
        if rendered.dim() == 5:
            rendered = rendered.reshape(1, 1, c, h, w)
        elif rendered.dim() == 4:
            rendered = rendered.unsqueeze(1)
        else:
            print(f"Warning: Unexpected rendered shape: {rendered.shape}")
            rendered = rendered.reshape(1, 1, c, h, w)

        rendered_images.append(rendered)

    rendered_batch = torch.cat(rendered_images, dim=0)
    return rendered_batch


def batch_rendering2_sparse(positions, scales, rotations, densities, image_size, pixel_masks):
    positions = positions.float()
    scales = scales.float()
    rotations = rotations.float()
    densities = densities.float()

    batch_size = positions.shape[0]
    rendered_images = []

    for b in range(batch_size):
        rendered = rendering2_sparse(
            positions[b],
            scales[b],
            rotations[b],
            densities[b],
            image_size,
            pixel_masks[b : b + 1],
        )

        c, h, w = image_size

        if rendered.dim() == 5:
            rendered = rendered.reshape(1, 1, c, h, w)
        elif rendered.dim() == 4:
            rendered = rendered.unsqueeze(1)
        else:
            print(f"Warning: Unexpected rendered shape: {rendered.shape}")
            rendered = rendered.reshape(1, 1, c, h, w)

        rendered_images.append(rendered)

    rendered_batch = torch.cat(rendered_images, dim=0)
    return rendered_batch


def sparse_render_to_patches(
    positions,
    scales,
    rotations,
    densities,
    input_size,
    pixel_mask,
    msk_length,
    patch_size,
    in_chans=1,
):
    batch_size = positions.shape[0]
    device = positions.device

    positions_f32 = positions.float().contiguous()
    scales_f32 = scales.float().contiguous()
    rotations_f32 = rotations.float().contiguous()
    densities_f32 = densities.float().contiguous()

    all_unmasked_points = []
    all_batch_indices = []

    for b in range(batch_size):
        unmasked_points, _ = get_unmasked_grid_points(input_size, pixel_mask[b : b + 1])
        if unmasked_points.shape[0] > 0:
            all_unmasked_points.append(unmasked_points)
            all_batch_indices.append(torch.full((unmasked_points.shape[0],), b, device=device))

    pixels_per_patch = int(np.prod(patch_size) * in_chans)
    if not all_unmasked_points:
        return torch.zeros(
            batch_size, msk_length, pixels_per_patch, device=device, dtype=torch.float32
        )

    all_points = torch.cat(all_unmasked_points, dim=0)
    batch_indices = torch.cat(all_batch_indices, dim=0)

    all_batch_results = []
    for b in range(batch_size):
        batch_mask = batch_indices == b
        if not batch_mask.any():
            continue

        current_points = all_points[batch_mask]
        L = build_scaling_rotation_inverse(scales_f32[b], rotations_f32[b])
        inv_covariance = (L @ L.transpose(1, 2)).contiguous()
        current_values = torch.zeros(current_points.shape[0], 1, device=device, dtype=torch.float32)
        current_values = compute_intensity_sparse(
            positions_f32[b],
            current_points,
            0.1 * densities_f32[b],
            inv_covariance,
            scales_f32[b],
            current_values,
        )
        all_batch_results.append((b, current_values))

    pixels_per_batch = msk_length * pixels_per_patch
    batch_values = []
    current_idx = 0

    for b in range(batch_size):
        batch_mask = batch_indices == b
        num_points = int(batch_mask.sum().item())
        if num_points > 0:
            batch_values.append(all_batch_results[current_idx][1])
            current_idx += 1
        else:
            batch_values.append(torch.zeros(0, 1, device=device, dtype=torch.float32))

    recon_x = torch.zeros(
        batch_size, msk_length, pixels_per_patch, device=device, dtype=torch.float32
    )

    for b in range(batch_size):
        batch_points = batch_values[b]
        if batch_points.shape[0] == pixels_per_batch:
            recon_x[b] = batch_points.reshape(msk_length, pixels_per_patch)
        elif batch_points.shape[0] > 0:
            print(
                f"Warning: Pixel count mismatch for batch {b}, expected {pixels_per_batch}, got {batch_points.shape[0]}"
            )
            if batch_points.shape[0] < pixels_per_batch:
                padding = torch.zeros(
                    pixels_per_batch - batch_points.shape[0], 1, device=device, dtype=torch.float32
                )
                padded = torch.cat([batch_points, padding], dim=0)
                recon_x[b] = padded.reshape(msk_length, pixels_per_patch)
            else:
                recon_x[b] = batch_points[:pixels_per_batch].reshape(msk_length, pixels_per_patch)

    return recon_x
