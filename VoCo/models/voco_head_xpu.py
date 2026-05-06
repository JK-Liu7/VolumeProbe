# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from monai.utils import ensure_tuple_rep
import argparse
import torch.nn.functional as F


import torch
import torch.nn as nn


class XpuSafeBatchNorm1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long),
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def _check_input_dim(self, x: torch.Tensor):
        if x.dim() not in (2, 3):
            raise ValueError(
                f"XpuSafeBatchNorm1d only supports 2D or 3D input, got shape {tuple(x.shape)}"
            )
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected channel dim = {self.num_features}, got {x.size(1)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        orig_dtype = x.dtype
        x_fp32 = x.float()

        if x.dim() == 2:
            # x: [N, C]
            reduce_dims = (0,)
            stat_shape = (1, self.num_features)
            elems_per_channel = x_fp32.shape[0]
        else:
            # x: [N, C, L]
            reduce_dims = (0, 2)
            stat_shape = (1, self.num_features, 1)
            elems_per_channel = x_fp32.shape[0] * x_fp32.shape[2]

        use_batch_stats = self.training or (not self.track_running_stats)

        if use_batch_stats:
            mean = x_fp32.mean(dim=reduce_dims)  # [C]
            var_biased = (x_fp32 - mean.view(*stat_shape)).pow(2).mean(dim=reduce_dims)  # [C]

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)

                    self.running_mean.mul_(1.0 - self.momentum).add_(
                        mean.detach(), alpha=self.momentum
                    )
                    if elems_per_channel > 1:
                        var_unbiased = var_biased.detach() * elems_per_channel / (elems_per_channel - 1)
                    else:
                        var_unbiased = var_biased.detach()

                    self.running_var.mul_(1.0 - self.momentum).add_(
                        var_unbiased, alpha=self.momentum
                    )
        else:
            mean = self.running_mean
            var_biased = self.running_var

        y = (x_fp32 - mean.view(*stat_shape)) * torch.rsqrt(var_biased.view(*stat_shape) + self.eps)

        if self.affine:
            y = y * self.weight.float().view(*stat_shape) + self.bias.float().view(*stat_shape)

        return y.to(orig_dtype)



class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Swin(nn.Module):
    def __init__(self, args):
        super(Swin, self).__init__()
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
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )

        norm_name = 'instance'

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.feature_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=16 * args.feature_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=True,
        )
        self.proj_head = projection_head(in_dim=1152, hidden_dim=2048, out_dim=2048)

    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x_in):
        b = x_in.size()[0]
        hidden_states_out = self.swinViT(x_in)

        enc0  = self.encoder1(x_in)
        enc1  = self.encoder2(hidden_states_out[0])
        enc2  = self.encoder3(hidden_states_out[1])
        enc3  = self.encoder4(hidden_states_out[2])
        dec4  = self.encoder10(hidden_states_out[4])

        encs = [enc0, enc1, enc2, enc3, dec4]
        out  = self.forward_encs(encs)
        out  = self.proj_head(out.view(b, -1))
        return out


class VoCoHead(nn.Module):
    def __init__(self, args):
        super(VoCoHead, self).__init__()
        self.student = Swin(args)
        self.teacher = Swin(args)

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        momentum = 0.9
        for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1.0 - momentum) * param.data

    def forward(self, img, crops, labels):
        batch_size = labels.size(0)
        total_size = img.size(0)
        sw_size = total_size // batch_size

        img = img.contiguous()
        crops = crops.contiguous()
        labels = labels.contiguous()

        inputs = torch.cat([img, crops], dim=0)

        students_all = self.student(inputs)
        self._EMA_update_encoder_teacher()
        with torch.no_grad():
            teachers_all = self.teacher(inputs).detach()

        dev = students_all.device
        dt = students_all.dtype
        pos = torch.zeros((), device=dev, dtype=dt)
        neg = torch.zeros((), device=dev, dtype=dt)
        total_b_loss = torch.zeros((), device=dev, dtype=dt)

        x_stu_all, bases_stu_all = students_all[:total_size], students_all[total_size:]
        x_tea_all, bases_tea_all = teachers_all[:total_size], teachers_all[total_size:]

        for i in range(batch_size):
            label = labels[i]

            x_stu = x_stu_all[i * sw_size:(i + 1) * sw_size]
            bases_stu = bases_stu_all[i * 16:(i + 1) * 16]
            x_tea = x_tea_all[i * sw_size:(i + 1) * sw_size]
            bases_tea = bases_tea_all[i * 16:(i + 1) * 16]

            logits1 = online_assign(x_stu, bases_tea)
            logits2 = online_assign(x_tea, bases_stu)
            logits = 0.5 * (logits1 + logits2)

            pos_loss, neg_loss = ce_loss(label, logits)
            pos = pos + pos_loss
            neg = neg + neg_loss

            b_loss = regularization_loss(bases_stu)
            total_b_loss = total_b_loss + b_loss

        pos = pos / batch_size
        neg = neg / batch_size
        total_b_loss = total_b_loss / batch_size
        return pos, neg, total_b_loss


def online_assign(feats, bases):
    b, c = feats.size()
    k, _ = bases.size()
    assert bases.size()[1] == c, print(feats.size(), bases.size())

    logits = []
    for i in range(b):
        feat  = feats[i].unsqueeze(0)
        simi  = F.cosine_similarity(feat, bases, dim=1).unsqueeze(0)
        logits.append(simi)
    logits = torch.cat(logits, dim=0)
    logits = F.relu(logits)
    return logits


def regularization_loss(bases):
    k, c = bases.size()
    loss_all = 0
    num = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            num += 1
            simi = F.cosine_similarity(bases[i].unsqueeze(0),
                                       bases[j].unsqueeze(0).detach(), dim=1)
            simi = F.relu(simi)
            loss_all += simi ** 2
    loss_all = loss_all / num
    return loss_all


def ce_loss(labels, logits):
    pos_dis  = torch.abs(labels - logits)
    pos_loss = -labels * torch.log(1 - pos_dis + 1e-6)
    pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

    neg_lab  = (labels == 0).long()
    neg_loss = neg_lab * (logits ** 2)
    neg_loss = neg_loss.sum() / (neg_lab.sum() + 1e-6)
    return pos_loss, neg_loss
