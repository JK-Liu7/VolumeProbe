
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0

import argparse
import functools
import logging
import os
import random
import resource
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import DataLoader

try:
    import intel_extension_for_pytorch as ipex
except Exception:
    ipex = None

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer_baseline_xpu import run_training
from utils.data_utils_baseline import get_loader
from utils.ddp_utils import (
    build_sampler,
    cleanup,
    init_distributed,
    is_main_process,
    silence_non_main,
    wrap_module,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*qfac.*")
warnings.filterwarnings("ignore", message=".*pixdim.*")
logging.getLogger("nibabel").setLevel(logging.ERROR)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "28890")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

_original_load = torch.load
torch.load = functools.partial(_original_load, weights_only=False)


class NullLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


_LOGGERS = {}


def init_log(name, log_dir, rank=0, level=logging.INFO):
    if rank != 0:
        return NullLogger()

    key = (name, os.path.abspath(log_dir), level)
    if key in _LOGGERS:
        return _LOGGERS[key]

    logger = logging.getLogger(f"{name}_{os.path.abspath(log_dir)}")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s")

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    _LOGGERS[key] = logger
    return logger


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.manual_seed(seed)
            torch.xpu.manual_seed_all(seed)
        except Exception:
            pass


def maybe_rebuild_loader_with_distributed_sampler(loader, ctx, shuffle=True):
    if loader is None or (not ctx.distributed):
        return loader
    if getattr(loader, "sampler", None) is not None and hasattr(loader.sampler, "set_epoch"):
        return loader

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader

    sampler = build_sampler(dataset, ctx, shuffle=shuffle)
    kwargs = {
        "dataset": dataset,
        "batch_size": loader.batch_size,
        "sampler": sampler,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": getattr(loader, "pin_memory", False),
        "drop_last": getattr(loader, "drop_last", False),
        "timeout": getattr(loader, "timeout", 0),
        "worker_init_fn": getattr(loader, "worker_init_fn", None),
        "persistent_workers": getattr(loader, "persistent_workers", False),
    }
    prefetch_factor = getattr(loader, "prefetch_factor", None)
    if prefetch_factor is not None and loader.num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    pin_memory_device = getattr(loader, "pin_memory_device", "")
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    return DataLoader(**kwargs)



def load_model_weights(model, ckpt_path, logger=None):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        clean_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if logger is not None:
        logger.info("Loaded model weights from %s", ckpt_path)
        logger.info("Missing keys: %d", len(missing))
        logger.info("Unexpected keys: %d", len(unexpected))
    print(f"Loaded model weights from {ckpt_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return checkpoint


parser = argparse.ArgumentParser(description="Baseline segmentation pipeline for XPU/CUDA with DDP")
parser.add_argument("--device_type", default="auto", choices=["auto", "cuda", "xpu", "cpu"], help="training device type")
parser.add_argument("--dist-backend", default="xccl", choices=["xccl", "ccl", "gloo"], help="distributed backend")
parser.add_argument("--dataset", default="BTCV", help="dataset")
parser.add_argument("--model", default="VoCo", help="model")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--seed", default=2026, type=int, help="random seed")
parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=2000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--amp", default=True, help="use amp for training/inference")
parser.add_argument("--noamp", action="store_true", help="disable AMP training")
parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])

parser.add_argument("--val_every", default=20, type=int, help="validation frequency")
parser.add_argument("--curr_ckpt_interval", default=20, type=int, help="current checkpoint saving frequency")
parser.add_argument("--ckpt_interval", default=200, type=int, help="checkpoint saving frequency")
parser.add_argument("--distributed", default=False, action="store_true", help="distributed training (overwritten by env WORLD_SIZE)")
parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="legacy arg, unused after ddp_utils migration")
parser.add_argument("--dist-url", default="env://", help="legacy arg, unused after ddp_utils migration")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--norm_name", default="group", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--num_classes", default=14, type=int)
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")

parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.5, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.50, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=100, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", default=False, action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=True, help="use self-supervised pretrained weights")
parser.add_argument("--squared_dice", default=True, help="use squared Dice")



def build_optimizer(args, model):
    if args.optim_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.weight_decay)
    if args.optim_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.weight_decay)
    if args.optim_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.optim_lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
    raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))


def build_scheduler(args, optimizer, last_epoch=-1):
    if args.lrschedule == "warmup_cosine":
        return LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
            last_epoch=last_epoch,
        )
    if args.lrschedule == "cosine_anneal":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    return None


def main():
    args = parser.parse_args()

    sw_dict = {
        'VoCo': 4,
        'S2DC': 4,
        'MedGMAE': 4,
    }
    args.sw_batch_size = sw_dict[args.model]

    args.amp = False if args.noamp else bool(args.amp)
    args.amp_dtype_torch = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    ctx = init_distributed(args.device_type, args.dist_backend)
    silence_non_main(ctx)

    args.local_rank = ctx.local_rank
    args.rank = ctx.rank
    args.world_size = ctx.world_size
    args.distributed = ctx.distributed
    args.device = ctx.device
    args.gpu = ctx.local_rank
    args.device_type = ctx.device.type

    if args.device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    setup_seed(args.seed + args.rank)
    args.test_mode = False
    args.inf_size = [args.roi_x // 4, args.roi_y // 4, args.roi_z // 4]

    logger = init_log("global", args.log_dir, rank=args.rank, level=logging.INFO)
    if is_main_process(ctx):
        logger.info("=" * 60)
        logger.info("Experiment start: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Dist backend : %s", args.dist_backend)
        logger.info("Rank / World : %d / %d", args.rank, args.world_size)
        logger.info("Local rank   : %d", args.local_rank)
        logger.info("Device       : %s", args.device)
        logger.info("Log dir      : %s", args.log_dir)
        logger.info("Data dir     : %s", args.data_dir)
        logger.info("AMP          : %s (%s)", args.amp, args.amp_dtype)
        logger.info("Batch / rank : %d", args.batch_size)
        logger.info("Global batch : %d", args.batch_size * max(1, args.world_size))
        logger.info("=" * 60)

    try:
        loaders, train_dataset = get_loader(args)
        train_loader, val_loader = loaders[0], loaders[1]
        train_loader = maybe_rebuild_loader_with_distributed_sampler(train_loader, ctx, shuffle=True)

        if model is None:
            raise ValueError(f"Unsupported model: {args.model}")

        model = model.to(args.device)
        logger.info("Total parameters count: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

        global_step = 0
        start_epoch = 0
        best_dice = 0.0
        checkpoint = None

        resume_path = args.checkpoint
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)

            if isinstance(checkpoint, dict):
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
                global_step = int(checkpoint.get("global_step", 0))
                best_dice = float(checkpoint.get("best_dice", 0.0))

            logger.info(
                "Resume from %s | start_epoch=%d | global_step=%d | best_dice=%.6f",
                resume_path, start_epoch, global_step, best_dice
            )

        if args.squared_dice:
            dice_loss = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                squared_pred=True,
                smooth_nr=args.smooth_nr,
                smooth_dr=args.smooth_dr,
            ).to(args.device)
        else:
            dice_loss = DiceCELoss(include_background=False, to_onehot_y=False, softmax=True).to(args.device)

        post_label = AsDiscrete(to_onehot=args.num_classes)
        post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
        dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)

        optimizer = build_optimizer(args, model)

        if resume_path is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Optimizer state restored from checkpoint.")

        if args.device.type == "xpu":
            if ipex is None:
                raise RuntimeError("intel_extension_for_pytorch is required for XPU training but is not available.")
            model, optimizer = ipex.optimize(
                model,
                optimizer=optimizer,
                dtype=args.amp_dtype_torch if args.amp else torch.float32,
                inplace=True,
            )

        model = wrap_module(model, ctx)

        scheduler = build_scheduler(args, optimizer, last_epoch=start_epoch - 1)

        if resume_path is not None and isinstance(checkpoint,
                                                  dict) and "scheduler" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("Scheduler state restored from checkpoint.")

        inf_size = [args.roi_x, args.roi_y, args.roi_z]
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size * 4 if args.model == 'MedGMAE' else args.sw_batch_size * 6,
            predictor=model,
            overlap=args.infer_overlap,
        )

        return run_training(
            args=args,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_dice=dice_loss,
            acc_func=dice_acc,
            logger=logger,
            model_inferer=model_inferer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            start_global_step=global_step,
            start_best_dice=best_dice,
            post_label=post_label,
            post_pred=post_pred,
        )
    finally:
        cleanup(ctx)


if __name__ == "__main__":
    main()
