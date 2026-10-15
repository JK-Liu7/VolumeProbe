import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.s2dc_head import S2DCTokenHead
from utils.data_utils_volumeprobe_cache import (
    get_1k_loader,
    get_10k_loader,
    get_mri_loader,
    get_pet_loader,
)
from utils.ops import concat_image
from pretraining.runtime import add_runtime_args, run_training


def build_parser():
    roi = 96
    parser = argparse.ArgumentParser(description="S2DC DDP Training with VolumeProbe")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")

    parser.add_argument("--queue_num", default=200, type=int)
    parser.add_argument("--in_channels", default=1, type=int, choices=[1])
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--dropout_path_rate", default=0.0, type=float)
    parser.add_argument(
        "--use_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--data_type", default="data_10k", choices=["data_1k", "data_10k", "mri", "pet"]
    )
    parser.add_argument("--use_geo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sinkhorn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spatial_dims", default=3, type=int, choices=[3])

    parser.add_argument("--a_min", default=-175.0, type=float)
    parser.add_argument("--a_max", default=250.0, type=float)
    parser.add_argument("--b_min", default=0.0, type=float)
    parser.add_argument("--b_max", default=1.0, type=float)

    parser.add_argument("--weight_p2s", default=0.5, type=float)
    parser.add_argument("--weight_p2p", default=0.5, type=float)
    parser.add_argument("--weight_global", default=0.5, type=float)

    parser.add_argument("--roi_large", default=384, type=int, help="S2DC training crop size")
    parser.add_argument("--roi_x", default=roi, type=int)
    parser.add_argument("--roi_y", default=roi, type=int)
    parser.add_argument("--roi_z", default=roi, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_geo_layer", default=-1, type=int)

    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--decay", default=1e-1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lrdecay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--lr_schedule", default="warmup_cosine", choices=["warmup_cosine", "poly"])

    parser.add_argument("--list_dir", default=None, type=str)

    parser.add_argument("--projection_dim", type=int, default=2048)
    parser.set_defaults(sync_batchnorm=True, ddp_find_unused_parameters=True)
    return add_runtime_args(parser)


def build_optimizer(args, model):
    if args.opt == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.opt == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.opt == "sgd":
        return optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay
        )
    raise ValueError(f"Unsupported optimizer: {args.opt}")


def get_train_loader(args):
    if args.data_type == "data_1k":
        return get_1k_loader(args)
    if args.data_type == "data_10k":
        return get_10k_loader(args)
    if args.data_type == "mri":
        return get_mri_loader(args)
    if args.data_type == "pet":
        return get_pet_loader(args)
    raise TypeError(f"Unsupported data_type: {args.data_type}")


def move_any_to_device(x, device, dtype=None):
    if torch.is_tensor(x):
        if dtype is not None and x.is_floating_point():
            return x.to(device=device, dtype=dtype, non_blocking=True)
        return x.to(device=device, non_blocking=True)
    if isinstance(x, np.ndarray):
        t = torch.as_tensor(x)
        if dtype is not None and t.is_floating_point():
            return t.to(device=device, dtype=dtype, non_blocking=True)
        return t.to(device=device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        out = [move_any_to_device(v, device, dtype=dtype) for v in x]
        return type(x)(out) if isinstance(x, tuple) else out
    if isinstance(x, dict):
        return {k: move_any_to_device(v, device, dtype=dtype) for k, v in x.items()}
    return x


def compute_loss(model, batch, device):
    src, aug_full, aug_crop, gt = batch
    inputs = {
        key: concat_image(value).as_tensor().to(device, dtype=torch.float32, non_blocking=True)
        for key, value in zip(("src", "aug_full", "aug_crop"), (src, aug_full, aug_crop))
    }
    inputs["gt"] = move_any_to_device(gt, device, torch.float32)
    loss, pos, neg, sharp, contrast = model(inputs)
    return loss, {"pos": pos, "neg": neg, "sharp": sharp, "contrast": contrast}


def main():
    args = build_parser().parse_args()
    if args.num_steps is None:
        args.num_steps = 100000
    if len({args.roi_x, args.roi_y, args.roi_z}) != 1 or args.roi_x % 32:
        raise ValueError("Swin crop dimensions must be equal multiples of 32.")
    if args.feature_size % 3:
        raise ValueError("feature_size must be divisible by 3.")
    if args.roi_large % args.roi_x or args.roi_large < args.roi_x * 2:
        raise ValueError(
            "roi_large must be a multiple of roi_x and contain at least a 2x2 crop grid."
        )
    if not (args.use_geo or args.use_cl):
        raise ValueError("Enable at least one of use_geo and use_cl.")
    run_training(args, S2DCTokenHead, get_train_loader, compute_loss, build_optimizer)


if __name__ == "__main__":
    main()
