import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.voco_head import VoCoHead
from utils.data_utils_volumeprobe_cache import get_loader
from utils.ops import concat_image
from pretraining.runtime import add_runtime_args, run_training


def build_parser():
    roi = 64
    parser = argparse.ArgumentParser(description="VoCo DDP Training")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument(
        "--in_channels", default=1, type=int, choices=[1], help="number of input channels"
    )
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument(
        "--use_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--spatial_dims", default=3, type=int, choices=[3], help="spatial dimension of input data"
    )
    parser.add_argument("--a_min", default=-175.0, type=float)
    parser.add_argument("--a_max", default=250.0, type=float)
    parser.add_argument("--b_min", default=0.0, type=float)
    parser.add_argument("--b_max", default=1.0, type=float)
    parser.add_argument("--roi_x", default=roi, type=int)
    parser.add_argument("--roi_y", default=roi, type=int)
    parser.add_argument("--roi_z", default=roi, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--sw_batch_size", default=2, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--decay", default=1e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lrdecay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", choices=["warmup_cosine", "poly"])

    parser.add_argument("--list_dir", help="Directory containing the seven VoCo JSON lists.")
    parser.add_argument("--projection_dim", type=int, default=2048)
    parser.set_defaults(sync_batchnorm=True, ddp_find_unused_parameters=False)
    return add_runtime_args(parser)


def build_optimizer(args, model):
    if args.opt == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.opt == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.decay)
    if args.opt == "sgd":
        return optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay
        )
    raise ValueError(f"Unsupported optimizer: {args.opt}")


def compute_loss(model, batch, device):
    img, labels, crops = batch
    img = concat_image(img).as_tensor().to(device, dtype=torch.float32, non_blocking=True)
    crops = concat_image(crops).as_tensor().to(device, dtype=torch.float32, non_blocking=True)
    labels = labels.to(device, dtype=torch.float32, non_blocking=True)
    pos, neg, base = model(img, crops, labels)
    return (pos + neg + base).mean(), {"pos": pos, "neg": neg, "base": base}


def main():
    args = build_parser().parse_args()
    if args.num_steps is None:
        args.num_steps = 100000
    if len({args.roi_x, args.roi_y, args.roi_z}) != 1 or args.roi_x % 32:
        raise ValueError("Swin crop dimensions must be equal multiples of 32.")
    if args.feature_size % 3:
        raise ValueError("feature_size must be divisible by 3.")
    if args.roi_x < 64:
        raise ValueError(
            "VoCo requires crops of at least 64 voxels per axis for instance normalization."
        )
    run_training(args, VoCoHead, get_loader, compute_loss, build_optimizer)


if __name__ == "__main__":
    main()
