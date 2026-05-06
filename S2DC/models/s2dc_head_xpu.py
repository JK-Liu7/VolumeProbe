import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep

from losses.loss_xpu import GeCoLoss


class XpuSafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = (x_fp32 - mean).pow(2).mean(dim=-1, keepdim=True)
        y = (x_fp32 - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            y = y * self.weight.float() + self.bias.float()
        return y.to(dtype=x.dtype)


class projection_head(nn.Module):
    def __init__(self, in_dim:int=768, hidden_dim:int=2048, out_dim:int=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Swin(nn.Module):
    def __init__(self, args, geco: bool = True, cl: bool = True):
        super().__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=XpuSafeLayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )

        norm_name = "instance"
        self.geco = bool(geco)
        self.cl = bool(cl)
        self.geo_layer = int(args.num_geo_layer)
        self.token_num = [48, 96, 192, 384, 768]
        in_dim = self.token_num[self.geo_layer]
        in_dim_cl = 768

        if self.geco:
            self.proj_head_geo = projection_head(in_dim=in_dim, hidden_dim=2048, out_dim=512)
        if self.cl:
            self.encoder_cof = UnetrBasicBlock(
                spatial_dims=args.spatial_dims,
                in_channels=16 * args.feature_size,
                out_channels=16 * args.feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )
            self.proj_head_cl = projection_head(in_dim=in_dim_cl, hidden_dim=2048, out_dim=512)
            self.avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.num_crops_side = max(1, int(args.roi_large // args.roi_z))
        self.args = args

    def _base_batch_size(self, x_in: torch.Tensor) -> int:
        crops_per_sample = max(1, self.num_crops_side * self.num_crops_side)
        return max(1, x_in.shape[0] // crops_per_sample)

    def _reassemble_crop_grid(self, feat: torch.Tensor, base_batch: int) -> torch.Tensor:
        feat = rearrange(
            feat,
            "(b c) d x y z -> b c x y z d",
            b=base_batch,
            c=self.num_crops_side * self.num_crops_side,
        )
        feat = rearrange(
            feat,
            "b (i j) x y z d -> b (i x) (j y) z d",
            i=self.num_crops_side,
            j=self.num_crops_side,
        )
        return feat

    def forward(self, x_in: torch.Tensor, visual_flag: bool = False, step: int = 0):
        del step
        hidden_states_out = self.swinViT(x_in)
        base_batch = self._base_batch_size(x_in)

        out_geo = hidden_states_out[self.geo_layer]
        b_total, f_d, s_t = out_geo.shape[:3]
        out_geo = out_geo.contiguous().reshape(b_total, f_d, s_t, s_t, s_t)

        out_cl = hidden_states_out[-1]
        b_total_cl, f_d_cl, s_t_cl = out_cl.shape[:3]
        out_cl = out_cl.contiguous().reshape(b_total_cl, f_d_cl, s_t_cl, s_t_cl, s_t_cl)
        if self.cl:
            out_cl = self.proj_head_cl(self.avg_pooling(out_cl).reshape(b_total_cl, -1))
            out_cl = F.normalize(out_cl, dim=1)
        else:
            out_cl = None

        if visual_flag:
            features_re = self._reassemble_crop_grid(out_geo, base_batch)
            _ = features_re  # placeholder for optional visualization hook

        out_geo_re = self._reassemble_crop_grid(out_geo, base_batch)
        out_geo_re = out_geo_re.reshape(base_batch, -1, s_t, f_d)

        if self.geco:
            out_geo_re = torch.stack(
                [F.normalize(self.proj_head_geo(out_geo_re[i, ...]), p=2, dim=2) for i in range(base_batch)],
                dim=0,
            )

        else:
            out_geo_re = None

        return out_geo_re, out_cl


class S2DCTokenHead(nn.Module):
    def __init__(self, args, exp: int = 200, dim: int = 512, num_patch_side: int = 4):
        super().__init__()
        del num_patch_side
        self.geco = bool(args.use_geo)
        self.cl = bool(args.use_cl)
        self.student = Swin(args, geco=self.geco, cl=self.cl)
        self.teacher = Swin(args, geco=self.geco, cl=self.cl)
        self.dim = int(dim)
        self.args = args
        self.criterion = GeCoLoss(args.weight_p2p, args.weight_p2s, args.weight_global, sinkhorn=args.sinkhorn)
        self.K = max(1, self.args.batch_size * exp)
        self.num_crops = int((self.args.roi_large // self.args.roi_x) ** 2)
        num_layer_token_side = [48, 24, 12, 6, 3]
        self.num_token_side = num_layer_token_side[args.num_geo_layer]

        if self.cl:
            self.register_buffer("queue_cl", torch.randn(self.K, dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.queue_cl = nn.functional.normalize(self.queue_cl, dim=1)

        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        momentum = 0.999
        for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1.0 - momentum) * param.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        end = ptr + batch_size
        if end <= self.K:
            self.queue_cl[ptr:end, :] = keys
        else:
            first = self.K - ptr
            self.queue_cl[ptr:, :] = keys[:first]
            self.queue_cl[: end - self.K, :] = keys[first:]
        self.queue_ptr[0] = end % self.K

    def reshape_sharp(self, sharps: List[torch.Tensor]) -> torch.Tensor:
        if len(sharps) == 0:
            return torch.empty(0)
        side = int(round(math.sqrt(sharps[0].numel())))
        depth = len(sharps)
        out_geo_re = torch.zeros(side, side, depth, device=sharps[0].device, dtype=sharps[0].dtype)
        for sharp, l in zip(sharps, range(depth)):
            flat = sharp.reshape(-1)
            for i in range(side):
                for j in range(side):
                    out_geo_re[i, j, l] = flat[i + j * side]
        return out_geo_re

    def forward(self, img: dict, visual_flag: bool = False, step: int = 0):
        src_crops = img["src"]
        crops_full_img_aug = img["aug_full"]
        crops_aug = img["aug_crop"]
        conf_matrix_gt = img["gt"]
        n = src_crops.shape[0] if self.num_crops <= 0 else max(1, src_crops.shape[0] // self.num_crops)

        q_geo, q_cl = self.student(src_crops, visual_flag, step)
        if self.geco and q_geo is not None:
            q_geo = torch.transpose(q_geo, 1, 2).contiguous().reshape(n * self.num_token_side, -1, q_geo.shape[-1])
        if self.cl and q_cl is not None:
            q_cl = torch.stack([q_cl[i * self.num_crops : (i + 1) * self.num_crops] for i in range(n)], dim=0)

        geo_pos = None
        cl_pos = None
        cl_pos_save = None
        if self.training:
            with torch.no_grad():
                self._EMA_update_encoder_teacher()
                geo_pos, _ = self.teacher(crops_full_img_aug)
                if self.cl:
                    _, cl_pos = self.teacher(crops_aug)
                    _, cl_pos_save = self.teacher(src_crops)
                    cl_pos = torch.stack([cl_pos[i * self.num_crops : (i + 1) * self.num_crops] for i in range(n)], dim=0).detach()
                    cl_pos_save = torch.stack(
                        [cl_pos_save[i * self.num_crops : (i + 1) * self.num_crops] for i in range(n)],
                        dim=0,
                    ).detach()
                if self.geco and geo_pos is not None:
                    geo_pos = torch.transpose(geo_pos, 1, 2).contiguous().reshape(
                        n * self.num_token_side,
                        -1,
                        geo_pos.shape[-1],
                    )
                    conf_matrix_gt = torch.cat(
                        [torch.stack([conf_matrix_gt[i, ...]] * self.num_token_side) for i in range(n)],
                        dim=0,
                    )
        else:
            _, _ = self.student(src_crops)
            geo_pos, _ = self.student(crops_full_img_aug)
            if self.geco and geo_pos is not None:
                geo_pos = torch.transpose(geo_pos, 1, 2).contiguous().reshape(
                    n * self.num_token_side,
                    -1,
                    geo_pos.shape[-1],
                )
                conf_matrix_gt = conf_matrix_gt.contiguous().expand(
                    n * self.num_token_side,
                    conf_matrix_gt.shape[1],
                    conf_matrix_gt.shape[2],
                )

        que_k_cl = self.queue_cl.clone().detach() if self.cl else None

        if q_geo is not None:
            q_geo = q_geo.float()
        if q_cl is not None:
            q_cl = q_cl.float()
        if geo_pos is not None:
            geo_pos = geo_pos.float()
        if cl_pos is not None:
            cl_pos = cl_pos.float()
        if cl_pos_save is not None and torch.is_tensor(cl_pos_save):
            cl_pos_save = cl_pos_save.float()
        if conf_matrix_gt is not None:
            conf_matrix_gt = conf_matrix_gt.float()

        with torch.autocast(device_type=q_geo.device.type if q_geo is not None else q_cl.device.type, enabled=False):
            sharpness, cl_pos_save, loss, geo_pos_loss, geo_neg_loss, loss_sharp, cl_loss = self.criterion(
                q_geo, q_cl,
                geo_pos=geo_pos,
                cl_pos=cl_pos,
                label=conf_matrix_gt,
                que_k_cl=que_k_cl.float() if que_k_cl is not None else None,
                cl_pos_save=cl_pos_save,
                step=step,
            )

        if visual_flag:
            return self.reshape_sharp(sharpness)
        if self.cl and not visual_flag and torch.is_tensor(cl_pos_save) and cl_pos_save.numel() > 0:
            self._dequeue_and_enqueue(cl_pos_save)
        return loss, geo_pos_loss, geo_neg_loss, loss_sharp, cl_loss


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)
