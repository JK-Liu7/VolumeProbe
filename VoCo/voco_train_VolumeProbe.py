import argparse
import datetime
import logging
import os
import random
import resource
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import intel_extension_for_pytorch as ipex

from models.voco_head_xpu import VoCoHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from utils.data_utils_volumeprobe_cache import get_loader
from utils.ddp_utils import cleanup, init_distributed, is_main_process, silence_non_main, wrap_module, unwrap_state_dict, build_sampler
from utils.ops import concat_image
from utils.utils import AverageMeter

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "28890")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

import functools

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

    key = (name, log_dir, level)
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
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.manual_seed(seed)
            torch.xpu.manual_seed_all(seed)
        except Exception:
            pass


def maybe_rebuild_loader_with_distributed_sampler(loader, ctx):
    if loader is None or (not ctx.distributed):
        return loader
    if getattr(loader, "sampler", None) is not None and hasattr(loader.sampler, "set_epoch"):
        return loader
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader

    sampler = build_sampler(dataset, ctx, shuffle=True)
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


def raw_model(model):
    return model.module if hasattr(model, "module") else model


def save_ckp(state, checkpoint_path):
    torch.save(state, checkpoint_path)


def build_parser():
    roi = 64
    parser = argparse.ArgumentParser(description="VoCo DDP Training")
    parser.add_argument("--logdir", default="logs", type=str, help="directory name to save logs")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of optimizer steps")
    parser.add_argument("--eval_num", default=500, type=int, help="checkpoint frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-175.0, type=float)
    parser.add_argument("--a_max", default=250.0, type=float)
    parser.add_argument("--b_min", default=0.0, type=float)
    parser.add_argument("--b_max", default=1.0, type=float)
    parser.add_argument("--space_x", default=1.5, type=float)
    parser.add_argument("--space_y", default=1.5, type=float)
    parser.add_argument("--space_z", default=1.5, type=float)
    parser.add_argument("--roi_x", default=roi, type=int)
    parser.add_argument("--roi_y", default=roi, type=int)
    parser.add_argument("--roi_z", default=roi, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--sw_batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--decay", default=1e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lrdecay", default=True)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="disable AMP training")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", default="xpu", choices=["auto", "xpu", "cuda", "cpu"])
    parser.add_argument("--dist-backend", default="xccl", choices=["xccl", "ccl", "gloo"])
    parser.add_argument("--smartcache_dataset", default=False)
    parser.add_argument("--cache_dataset", default=False)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=2026, type=int)

    # VolumeProbe Config
    parser.add_argument("--probe_candidate_prob", default=0.7, type=float)
    parser.add_argument("--probe_jitter_frac", default=0.05, type=float)
    parser.add_argument("--probe_axcodes", type=str, default="RAS")
    parser.add_argument("--probe_pixdim", default=(1.5, 1.5, 1.5), type=float, nargs=3)
    parser.add_argument("--probe_spacing_mode", type=str, default="bilinear")
    parser.add_argument("--probe_temperature", default=0.40, type=float)
    return parser


def build_optimizer(args, model):
    if args.opt == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.opt == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.decay)
    if args.opt == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    raise ValueError(f"Unsupported optimizer: {args.opt}")


def build_scheduler(args, optimizer, last_epoch=-1):
    if not args.lrdecay:
        return None
    if args.lr_schedule == "warmup_cosine":
        return WarmupCosineSchedule(
            optimizer,
            warmup_steps=args.warmup_steps,
            t_total=args.num_steps,
            last_epoch=last_epoch,
        )
    if args.lr_schedule == "poly":
        def lambdas(step):
            return (1 - float(step) / float(args.num_steps)) ** 0.9
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambdas, last_epoch=last_epoch
        )
    raise ValueError(f"Unsupported lr_schedule: {args.lr_schedule}")


def build_scaler(args, amp_dtype):
    if (not args.amp) or amp_dtype != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("xpu", enabled=True)
    except Exception:
        try:
            return torch.xpu.amp.GradScaler(enabled=True)
        except Exception as exc:
            raise RuntimeError(
                "FP16 AMP was requested, but no XPU GradScaler is available. Switch to --amp_dtype bf16 or upgrade the env."
            ) from exc


def train_one_epoch(args, model, optimizer, scheduler, scaler, train_loader, device, logger, global_step):
    model.train()
    run_loss = AverageMeter()
    pos_avg = AverageMeter()
    neg_avg = AverageMeter()
    base_avg = AverageMeter()

    iterator = enumerate(train_loader)
    if (not args.distributed) or args.rank == 0:
        iterator = tqdm(iterator, total=len(train_loader), ncols=100)


    for step, batch in iterator:
        t1 = time()

        img, labels, crops = batch

        img = concat_image(img).as_tensor().to(device=device, dtype=torch.float32).contiguous()
        crops = concat_image(crops).as_tensor().to(device=device, dtype=torch.float32).contiguous()
        labels = labels.to(device=device, dtype=torch.float32).contiguous()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=args.amp_dtype_torch, enabled=args.amp):
            pos, neg, b_loss = model(img, crops, labels)
            loss = pos + neg + b_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_item = float(loss.detach().item())
        pos_item = float(pos.detach().item())
        neg_item = float(neg.detach().item())
        base_item = float(b_loss.detach().item())

        run_loss.update(loss_item, n=args.batch_size)
        pos_avg.update(pos_item, n=args.batch_size)
        neg_avg.update(neg_item, n=args.batch_size)
        base_avg.update(base_item, n=args.batch_size)

        lr = optimizer.param_groups[0]["lr"]
        global_step += 1

        if global_step % 20 == 0 and args.rank == 0:
            logger.info(
                "step=%d/%d loss=%.4f pos=%.4f neg=%.4f base=%.4f lr=%.2e time=%.3fs",
                global_step,
                args.num_steps,
                run_loss.avg,
                pos_avg.avg,
                neg_avg.avg,
                base_avg.avg,
                lr,
                time() - t1,
            )

        if args.rank == 0 and global_step % args.eval_num == 0:
            ckpt_path = os.path.join(args.log_dir, "model_save", "model_current_epoch.pt")
            checkpoint = {
                "global_step": global_step,
                "state_dict": unwrap_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
            save_ckp(checkpoint, ckpt_path)
            logger.info(">>> checkpoint saved: %s", ckpt_path)

        if args.rank == 0 and global_step % 1000 == 0:
            ckpt_path = os.path.join(args.log_dir, "model_save", f"model_step{global_step}.pt")
            checkpoint = {
                "global_step": global_step,
                "state_dict": unwrap_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
            save_ckp(checkpoint, ckpt_path)
            logger.info(">>> milestone checkpoint saved: %s", ckpt_path)

        if global_step >= args.num_steps:
            break

    return global_step


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.amp = not args.noamp
    args.amp_dtype_torch = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    ctx = init_distributed(args.device, args.dist_backend)
    silence_non_main(ctx)

    args.local_rank = ctx.local_rank
    args.rank = ctx.rank
    args.world_size = ctx.world_size
    args.distributed = ctx.distributed
    device = ctx.device

    setup_seed(args.seed + ctx.rank)

    logger = init_log("global", args.log_dir, rank=args.rank, level=logging.INFO)
    if is_main_process(ctx):
        logger.info("=" * 60)
        logger.info("Experiment start: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Dist backend : %s", args.dist_backend)
        logger.info("Rank / World : %d / %d", args.rank, args.world_size)
        logger.info("Local rank   : %d", args.local_rank)
        logger.info("Device       : %s", device)
        logger.info("Log dir      : %s", args.log_dir)
        logger.info("Data dir     : %s", args.data_dir)
        logger.info("AMP          : %s (%s)", args.amp, args.amp_dtype)
        logger.info("Batch / rank : %d", args.batch_size)
        logger.info("Global batch : %d", args.batch_size * max(1, args.world_size))
        logger.info("=" * 60)

    try:
        model = VoCoHead(args).to(device)
        optimizer = build_optimizer(args, model)

        global_step = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)  # load BEFORE ipex wraps the model
            if isinstance(checkpoint, dict) and "global_step" in checkpoint:
                global_step = int(checkpoint["global_step"])

        model, optimizer = ipex.optimize(
            model,
            optimizer=optimizer,
            dtype=args.amp_dtype_torch if args.amp else torch.float32,
            inplace=True,
        )

        if checkpoint is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if args.rank == 0:
                logger.info("after optimizer load:", optimizer.param_groups[0]["lr"])
                logger.info("Resumed from %s at global_step=%d", args.resume, global_step)

        scheduler = build_scheduler(args, optimizer, last_epoch=global_step - 1)
        if (
                scheduler is not None
                and checkpoint is not None
                and isinstance(checkpoint, dict)
                and "scheduler" in checkpoint
                and checkpoint["scheduler"] is not None
        ):
            scheduler.load_state_dict(checkpoint["scheduler"])

        if args.rank == 0:
            logger.info("after scheduler build:", optimizer.param_groups[0]["lr"])

        scaler = build_scaler(args, args.amp_dtype_torch)

        model = wrap_module(model, ctx)

        train_loader = get_loader(args)
        train_loader = maybe_rebuild_loader_with_distributed_sampler(train_loader, ctx)

        if args.resume and global_step > 0:
            epoch = global_step // len(train_loader)
        else:
            epoch = 0

        while global_step < args.num_steps:
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            global_step = train_one_epoch(
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_loader=train_loader,
                device=device,
                logger=logger,
                global_step=global_step,
            )
            epoch += 1

        final_checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": unwrap_state_dict(model),
            "optimizer": optimizer.state_dict(),
        }
        if args.rank == 0:
            final_model_path = os.path.join(args.log_dir, "model_save", "final_model.pth")
            final_ckpt_path = os.path.join(args.log_dir, "model_save", "model_final_epoch.pt")
            torch.save(unwrap_state_dict(model), final_model_path)
            save_ckp(final_checkpoint, final_ckpt_path)
            logger.info("Saved final model to %s", final_model_path)
            logger.info("Saved final checkpoint to %s", final_ckpt_path)
    finally:
        cleanup(ctx)


if __name__ == "__main__":
    main()
