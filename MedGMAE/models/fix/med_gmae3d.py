import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.layers import to_3tuple


from models.fix.networks.gaussian_renderer import batch_rendering2


__all__ = ["MedGMAE3D"]


def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.0):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d, indexing="ij")
    assert embed_dim % 6 == 0, (
        "Embed dimension must be divisible by 6 for 3D sin-cos position embedding"
    )
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_d = torch.einsum("m,d->md", [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [
            torch.sin(out_h),
            torch.cos(out_h),
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_d),
            torch.cos(out_d),
        ],
        dim=1,
    )[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def build_perceptron_position_embedding(grid_size, embed_dim, num_tokens=1):
    pos_emb = torch.rand([1, np.prod(grid_size), embed_dim])
    nn.init.normal_(pos_emb, std=0.02)

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    return pos_embed


def patchify_image(x, patch_size):

    B, C, H, W, D = x.shape
    patch_size = to_3tuple(patch_size)
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x = x.reshape(
        B, C, grid_size[0], patch_size[0], grid_size[1], patch_size[1], grid_size[2], patch_size[2]
    )
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size), np.prod(patch_size) * C)

    return x


def batched_shuffle_indices(batch_size, length, device):
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        in_chan_last=True,
    ):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)

        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.in_chan_last = in_chan_last

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        B, L, S = x.shape
        assert S == np.prod(self.img_size) * self.in_chans, (
            f"Input image total size {S} doesn't match model configuration"
        )
        if self.in_chan_last:
            x = x.reshape(B * L, *self.img_size, self.in_chans).permute(0, 4, 1, 2, 3)
        else:
            x = x.reshape(B * L, self.in_chans, *self.img_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class MedGMAE3D(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.img_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size

        out_chans = args.in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        embed_layer = PatchEmbed3D
        self.encoder = encoder(
            patch_size=patch_size, in_chans=args.in_chans, embed_layer=embed_layer
        )

        self.num_gaussians = getattr(args, "num_gaussians", 512)

        decoder_kwargs = {"num_gaussians": self.num_gaussians, "patch_size": patch_size}

        if hasattr(args, "decoder_type") and args.decoder_type == "cgs":
            decoder_kwargs["cgs_layers"] = getattr(args, "cgs_layers", [2, 5, 8, 11])
            decoder_kwargs["upsample_ratios"] = getattr(args, "cgs_upsample_ratios", [1, 2, 4, 8])

        self.decoder = decoder(**decoder_kwargs)

        self.encoder_to_decoder = nn.Linear(
            self.encoder.embed_dim, self.decoder.embed_dim, bias=True
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder.embed_dim))

        self.use_image_norm = getattr(args, "use_image_norm", False)
        if self.use_image_norm:
            self.image_norm = nn.LayerNorm(
                normalized_shape=list(self.input_size), eps=1e-6, elementwise_affine=False
            )
        else:
            self.image_norm = None

        self.criterion = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_ssim = SSIM3DLoss(
            window_size=getattr(args, "ssim_window_size", 7),
            sigma=getattr(args, "ssim_sigma", 1.5),
            data_range=1.0,
            channel=args.in_chans,
        )

        if args.pos_embed_type == "sincos":
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(
                    grid_size, self.encoder.embed_dim, num_tokens=0
                )
                self.decoder_pos_embed = build_3d_sincos_position_embedding(
                    grid_size, self.decoder.embed_dim, num_tokens=0
                )
        elif args.pos_embed_type == "perceptron":
            self.encoder_pos_embed = build_perceptron_position_embedding(
                grid_size, self.encoder.embed_dim, num_tokens=0
            )
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(
                    grid_size, self.decoder.embed_dim, num_tokens=0
                )

        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        nn.init.normal_(self.mask_token, std=0.02)

        self.gaussian_query_tokens = nn.Parameter(
            torch.zeros(1, self.num_gaussians, self.decoder.embed_dim)
        )
        nn.init.normal_(self.gaussian_query_tokens, std=0.02)

        self.global_step = 0

        self.initial_scale_penalty_weight = getattr(args, "scale_penalty_weight", 0.1)

        self.use_sparse_rendering = getattr(args, "use_sparse_rendering", True)

    def get_decayed_scale_penalty_weight(self):
        if self.global_step == 0:
            return self.initial_scale_penalty_weight

        decay_factor = 0.1 ** (self.global_step / 1000.0)
        decayed_weight = self.initial_scale_penalty_weight * decay_factor

        return decayed_weight

    def render_gaussians(self, positions, scales, rotations, densities, pixel_mask=None):
        with torch.autocast(device_type=positions.device.type, enabled=False):
            return batch_rendering2(
                positions.float(),
                scales.float(),
                rotations.float(),
                densities.float(),
                self.input_size,
            )

    def apply_image_norm(self, img):
        if not self.use_image_norm or self.image_norm is None:
            return img

        B, C, H, W, D = img.shape

        if C == 1:
            img_squeezed = img.squeeze(1)
            img_normed = self.image_norm(img_squeezed)
            return img_normed.unsqueeze(1)
        else:
            img_list = []
            for c in range(C):
                channel_img = img[:, c, :, :, :]
                channel_normed = self.image_norm(channel_img)
                img_list.append(channel_normed.unsqueeze(1))
            return torch.cat(img_list, dim=1)

    def forward(self, x, return_image=False):
        args = self.args
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == args.in_chans
        out_chans = self.out_chans

        x_patches = patchify_image(x, self.patch_size)

        length = np.prod(self.grid_size)
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        shuffle_indices = batched_shuffle_indices(batch_size, length, device=x.device)

        shuffled_x = x_patches.gather(
            dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans)
        )
        sel_x = shuffled_x[:, :sel_length, :]

        sel_indices = shuffle_indices[:, :sel_length]

        sel_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1).gather(
            dim=1, index=sel_indices[:, :, None].expand(-1, -1, self.encoder.embed_dim)
        )

        sel_x = self.encoder(sel_x, sel_encoder_pos_embed)

        sel_x = self.encoder_to_decoder(sel_x)

        query_tokens = self.gaussian_query_tokens.expand(batch_size, -1, -1)

        all_tokens = torch.cat([sel_x[:, :1], query_tokens, sel_x[:, 1:]], dim=1)

        positions, scales, rotations, densities = self.decoder(all_tokens)

        scale_penalty = 0.0
        if hasattr(args, "max_gaussian_scale"):
            max_scale = getattr(args, "max_gaussian_scale", 0.5)
            scale_penalty = torch.nn.functional.relu(scales - max_scale).pow(2).mean()
            current_scale_penalty_weight = self.get_decayed_scale_penalty_weight()
        else:
            current_scale_penalty_weight = 0.0

        pixel_mask = torch.zeros_like(x)

        patch_mask = torch.zeros(batch_size, length, device=x.device)
        patch_mask.scatter_(1, shuffle_indices[:, -msk_length:], 1.0)
        patch_mask = patch_mask.reshape(batch_size, *self.grid_size)
        patch_mask = patch_mask.unsqueeze(1)

        for dim_idx, patch_dim in enumerate(self.patch_size):
            patch_mask = patch_mask.repeat_interleave(patch_dim, dim=dim_idx + 2)

        pixel_mask = patch_mask

        rendered_image = self.render_gaussians(positions, scales, rotations, densities, pixel_mask)
        masked_target_image = x * pixel_mask

        rendered_normed = self.apply_image_norm(rendered_image)
        target_normed = self.apply_image_norm(masked_target_image)
        reconstruction_loss = self.criterion(rendered_normed, target_normed)

        loss = reconstruction_loss

        if not return_image and self.training:
            loss_stats = {
                "reconstruction_loss": reconstruction_loss.item(),
                "uniformity_loss": self.compute_uniformity_loss(positions, batch_size).item(),
                "total_loss": loss.item(),
                "global_step": self.global_step,
            }

            if current_scale_penalty_weight > 0:
                loss_stats["scale_penalty"] = scale_penalty.item()
                loss_stats["scale_penalty_weight"] = current_scale_penalty_weight

            positions_mean = positions.mean().item()
            positions_std = positions.std().item()
            scales_mean = scales.mean().item()
            scales_std = scales.std().item()
            densities_mean = densities.mean().item()
            densities_std = densities.std().item()

            param_stats = {
                "positions_mean": positions_mean,
                "positions_std": positions_std,
                "scales_mean": scales_mean,
                "scales_std": scales_std,
                "densities_mean": densities_mean,
                "densities_std": densities_std,
            }

            for i, axis in enumerate(["x", "y", "z"]):
                param_stats[f"positions_{axis}_mean"] = positions[:, :, i].mean().item()
                param_stats[f"positions_{axis}_std"] = positions[:, :, i].std().item()
                param_stats[f"scales_{axis}_mean"] = scales[:, :, i].mean().item()
                param_stats[f"scales_{axis}_std"] = scales[:, :, i].std().item()

            geom_stats = {}
            if batch_size > 0 and positions.shape[1] > 1:
                try:
                    distances = torch.cdist(positions[0], positions[0])
                    mask = ~torch.eye(positions.shape[1], dtype=torch.bool, device=positions.device)
                    masked_distances = distances[mask]

                    geom_stats = {
                        "min_distance": masked_distances.min().item(),
                        "max_distance": masked_distances.max().item(),
                        "mean_distance": masked_distances.mean().item(),
                        "std_distance": masked_distances.std().item(),
                    }
                except Exception as e:
                    print(f"Failed to compute distance statistics: {e}")

            self.current_loss_stats = loss_stats
            self.current_param_stats = param_stats
            self.current_geom_stats = geom_stats

            if (
                hasattr(self, "use_wandb")
                and self.use_wandb
                and hasattr(self, "_trainer_wandb_log")
            ):
                try:
                    stats_to_log = {
                        "loss/reconstruction": reconstruction_loss.item(),
                        "loss/uniformity": self.compute_uniformity_loss(
                            positions, batch_size
                        ).item(),
                        "loss/total": loss.item(),
                        "global_step": self.global_step,
                    }

                    if current_scale_penalty_weight > 0:
                        stats_to_log["loss/scale_penalty"] = scale_penalty.item()
                        stats_to_log["loss/scale_penalty_weight"] = current_scale_penalty_weight

                    for key, val in param_stats.items():
                        stats_to_log[f"params/{key}"] = val

                    for key, val in geom_stats.items():
                        stats_to_log[f"geometry/{key}"] = val

                    self._trainer_wandb_log(stats_to_log, step=self.global_step)
                except Exception as e:
                    print(f"Wandb logging failed: {e}")

        if return_image:
            masked_x = x * (1.0 - pixel_mask)

            return loss, x.detach(), rendered_image.detach(), masked_x.detach(), pixel_mask.detach()
        else:
            self.global_step += 1
            return loss

    def compute_uniformity_loss(self, positions, batch_size):
        uniformity_loss = 0.0

        for b in range(batch_size):
            pos = positions[b]

            x1 = pos.unsqueeze(1)
            x2 = pos.unsqueeze(0)
            dist_sq = torch.sum((x1 - x2) ** 2, dim=-1)

            mask = ~torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
            dist_sq = dist_sq[mask]

            t = 0.5
            uniformity_term = torch.exp(-dist_sq / t).mean()

            uniformity_loss += uniformity_term

        return uniformity_loss / batch_size


def build_vit_base_med_gmae_3d(args):
    from models.fix.networks.vit import vit_base_patch16_96

    if hasattr(args, "decoder_type") and args.decoder_type == "cgs":
        from models.fix.networks.cgs_mae_decoder import cgs_med_gaussian_decoder_base

        decoder = cgs_med_gaussian_decoder_base
    else:
        from models.fix.networks.med_gaussian_decoder import med_gaussian_decoder_base

        decoder = med_gaussian_decoder_base

    return MedGMAE3D(args=args, encoder=vit_base_patch16_96, decoder=decoder)


def build_vit_large_med_gmae_3d(args):
    from models.fix.networks.vit import vit_large_patch16_96

    if hasattr(args, "decoder_type") and args.decoder_type == "cgs":
        from models.fix.networks.cgs_mae_decoder import cgs_med_gaussian_decoder_large

        decoder = cgs_med_gaussian_decoder_large
    else:
        from models.fix.networks.med_gaussian_decoder import med_gaussian_decoder_large

        decoder = med_gaussian_decoder_large

    return MedGMAE3D(args=args, encoder=vit_large_patch16_96, decoder=decoder)


def build_vit_large_med_gmae_p12_3d(args):
    from models.fix.networks.vit import vit_large_patch12_96

    if hasattr(args, "decoder_type") and args.decoder_type == "cgs":
        from models.fix.networks.cgs_mae_decoder import cgs_med_gaussian_decoder_large

        decoder = cgs_med_gaussian_decoder_large
    else:
        from models.fix.networks.med_gaussian_decoder import med_gaussian_decoder_large

        decoder = med_gaussian_decoder_large

    model = MedGMAE3D(args=args, encoder=vit_large_patch12_96, decoder=decoder)
    print(model)
    return model


def build_vit_base_med_gmae_p8_3d(args):
    from models.fix.networks.vit import vit_base_patch8_96

    if hasattr(args, "decoder_type") and args.decoder_type == "cgs":
        from models.fix.networks.cgs_mae_decoder import cgs_med_gaussian_decoder_base

        decoder = cgs_med_gaussian_decoder_base
    else:
        from models.fix.networks.med_gaussian_decoder import med_gaussian_decoder_base

        decoder = med_gaussian_decoder_base

    model = MedGMAE3D(args=args, encoder=vit_base_patch8_96, decoder=decoder)
    print(model)
    return model


def gaussian_kernel_3d(window_size: int, sigma: float) -> torch.Tensor:
    """Generate 3D Gaussian kernel"""
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    g_3d = g.view(1, -1, 1, 1) * g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    g_3d = g_3d.expand(1, 1, window_size, window_size, window_size).contiguous()

    return g_3d


def ssim_3d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    size_average: bool = True,
    channel: int = 1,
) -> torch.Tensor:
    window = gaussian_kernel_3d(window_size, sigma)
    window = window.expand(channel, 1, window_size, window_size, window_size).contiguous()

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).mean(1)


class SSIM3D(nn.Module):
    """3D SSIM loss function module"""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        size_average: bool = True,
        channel: int = 1,
    ):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim_3d(
            img1,
            img2,
            self.window_size,
            self.sigma,
            self.data_range,
            self.size_average,
            self.channel,
        )


class SSIM3DLoss(nn.Module):
    """3D SSIM loss function (1 - SSIM)"""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        size_average: bool = True,
        channel: int = 1,
    ):
        super(SSIM3DLoss, self).__init__()
        self.ssim = SSIM3D(window_size, sigma, data_range, size_average, channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1 - self.ssim(img1, img2)
