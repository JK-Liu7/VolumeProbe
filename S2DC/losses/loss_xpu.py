import random
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .optimal_transport import log_optimal_transport
from .utils import MatchLoss


def _infer_device(*tensors) -> torch.device:
    for t in tensors:
        if torch.is_tensor(t):
            return t.device
    return torch.device("cpu")


class GeCoContrast(nn.Module):
    def __init__(self, temperature: float = 0.9):
        super().__init__()
        self.temp = float(temperature)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, q: torch.Tensor, all_k: torch.Tensor) -> torch.Tensor:
        n = q.shape[0]
        sim = torch.einsum("nc,kc->nk", q, all_k)
        l_pos = torch.diag(sim).unsqueeze(-1)
        l_neg = sim[:, n:]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temp
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return self.criterion(logits, labels)


class SharpeLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        labels: torch.Tensor,
        step: int = 0,
    ):
        del step
        b_q = q.shape[0]
        labels = labels.to(device=q.device)
        sharpe_list = []
        losses = []

        for i in range(b_q):
            label = labels[i]
            sim_matrix = (1.0 + torch.einsum("lc,sc->ls", q[i], k[i])) / 2.0

            std0 = torch.std(sim_matrix, dim=0, unbiased=False).clamp_min(self.eps)
            std1 = torch.std(sim_matrix, dim=1, unbiased=False).clamp_min(self.eps)
            sharpe1 = F.softmax((torch.max(sim_matrix, dim=0)[0] - torch.mean(sim_matrix, dim=0)) / std0, dim=0)
            sharpe2 = F.softmax((torch.max(sim_matrix, dim=1)[0] - torch.mean(sim_matrix, dim=1)) / std1, dim=0)
            sharpe_list.append(sharpe1)

            pos_mask = label == 1
            neg_mask = label == 0
            sim_matrix = torch.clamp(sim_matrix, self.eps, 1.0 - self.eps)
            loss_mat = -torch.log(sim_matrix)
            loss1 = torch.sum((loss_mat * sharpe1[None, ...])[pos_mask]) if pos_mask.any() else sim_matrix.new_tensor(0.0)
            loss2 = torch.sum((loss_mat * sharpe2[..., None])[pos_mask]) if pos_mask.any() else sim_matrix.new_tensor(0.0)

            if neg_mask.any():
                loss3 = torch.mean(-torch.log(1.0 - sim_matrix[neg_mask]))
            else:
                loss3 = sim_matrix.new_tensor(0.0)

            losses.append((loss1 + loss2) / 2.0 + loss3)

        return sharpe_list, torch.mean(torch.stack(losses))


class GeometricLoss(nn.Module):
    def __init__(self, sinkhorn: bool = False):
        super().__init__()
        self.loss = MatchLoss()
        self.sinkhorn = bool(sinkhorn)
        self.bin_score = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.skh_iters = 100

    def forward(self, q: torch.Tensor, k: torch.Tensor, labels: torch.Tensor):
        losses = []
        pos_losses = []
        neg_losses = []
        labels = labels.to(device=q.device)

        for i in range(q.shape[0]):
            label = labels[i]
            all_k = k[i][None, ...]
            sim_matrix = torch.einsum("nlc,nsc->nls", q[i][None, ...], all_k)

            if self.sinkhorn:
                log_assign_matrix = log_optimal_transport(sim_matrix, self.bin_score, self.skh_iters)
                conf_matrix = log_assign_matrix.exp()[:, :-1, :-1]
            else:
                conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

            gt = torch.zeros_like(conf_matrix)
            gt[0, :, :] = label
            loss, pos_loss, neg_loss = self.loss(conf_matrix, gt)
            losses.append(loss)
            pos_losses.append(pos_loss)
            neg_losses.append(neg_loss)

        return (
            torch.mean(torch.stack(losses)),
            torch.mean(torch.stack(pos_losses)),
            torch.mean(torch.stack(neg_losses)),
        )


class GeCoLoss(nn.Module):
    def __init__(
        self,
        weight_p2p: float = 0.5,
        weight_p2s: float = 0.5,
        weight_global: float = 0.5,
        sinkhorn: bool = False,
    ):
        super().__init__()
        self.contrast_loss = GeCoContrast()
        self.geco_loss = GeometricLoss(sinkhorn=sinkhorn)
        self.sharp_loss = SharpeLoss()
        self.weight_p2p = float(weight_p2p)
        self.weight_p2s = float(weight_p2s)
        self.weight_global = float(weight_global)

    def forward(
        self,
        q_geo: Optional[torch.Tensor],
        q_cl: Optional[torch.Tensor],
        geo_pos: Optional[torch.Tensor] = None,
        cl_pos: Optional[torch.Tensor] = None,
        que_k_geo: Optional[torch.Tensor] = None,
        que_k_cl: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cl_pos_save: Optional[torch.Tensor] = None,
        step: int = 0,
    ):
        del que_k_geo
        device = _infer_device(q_geo, q_cl, geo_pos, cl_pos, label, que_k_cl, cl_pos_save)
        zero = torch.tensor(0.0, device=device)
        sharpe = []

        if geo_pos is not None and q_geo is not None and label is not None:
            loss_geo, geo_pos_loss, geo_neg_loss = self.geco_loss(q_geo, geo_pos, label)
            sharpe, loss_sharp = self.sharp_loss(q_geo, geo_pos, label, step)
            loss_geo = self.weight_p2p * loss_geo + self.weight_p2s * loss_sharp
        else:
            loss_geo = zero
            geo_pos_loss = zero
            geo_neg_loss = zero
            loss_sharp = zero

        if cl_pos is not None and q_cl is not None:
            n, p, _ = q_cl.shape
            if p <= 0:
                loss_cl = zero
                cl_pos_save_out = zero
            else:
                idx = [random.randrange(p) for _ in range(n)]
                q_cl_sel = torch.cat([q_cl[i, j, ...][None, ...] for i, j in zip(range(n), idx)], dim=0)
                cl_pos_work = cl_pos.clone()

                if que_k_cl is not None:
                    cl_pos_save_out = torch.cat(
                        [cl_pos_save[i, j, ...][None, ...] for i, j in zip(range(n), idx)],
                        dim=0,
                    )
                    for i, j in zip(range(n), idx):
                        tmp = cl_pos_work[i, 0, :].clone()
                        cl_pos_work[i, 0, :] = cl_pos_work[i, j, :]
                        cl_pos_work[i, j, :] = tmp
                    cl_pos_cat = torch.cat(
                        [cl_pos_work[i, 0, ...][None, ...] for i in range(n)]
                        + [cl_pos_work[i, 1:, ...] for i in range(n)],
                        dim=0,
                    )
                    cl_pos_cat = torch.cat([cl_pos_cat, que_k_cl], dim=0)
                else:
                    for i, j in zip(range(n), idx):
                        tmp = cl_pos_work[i, 0, :].clone()
                        cl_pos_work[i, 0, :] = cl_pos_work[i, j, :]
                        cl_pos_work[i, j, :] = tmp
                    cl_pos_save_out = torch.cat([cl_pos_work[i, 0, ...][None, ...] for i in range(n)], dim=0)
                    cl_pos_cat = torch.cat(
                        [cl_pos_work[i, 0, ...][None, ...] for i in range(n)]
                        + [cl_pos_work[i, 1:, ...] for i in range(n)],
                        dim=0,
                    )

                loss_cl = self.contrast_loss(q_cl_sel, cl_pos_cat)
        else:
            loss_cl = zero
            cl_pos_save_out = zero

        loss_cl = loss_cl * self.weight_global
        return sharpe, cl_pos_save_out, loss_geo + loss_cl, geo_pos_loss, geo_neg_loss, loss_sharp, loss_cl
