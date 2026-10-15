import argparse
import json
import logging
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from monai.utils import set_determinism

from pretraining.distributed import cleanup, init_distributed, unwrap_state_dict, wrap_module


def add_runtime_args(parser):
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--datalist",
        help="JSON image list, or a JSON object with a training list; paths are relative to data_dir.",
    )
    parser.add_argument("--log_dir", "--logdir", dest="log_dir", required=True)
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Full output checkpoint path, including the filename; updated at each save.",
    )
    parser.add_argument("--weights_path", help="Optional output path for final model weights.")
    parser.add_argument("--resume", help="Explicit path to a training checkpoint.")
    parser.add_argument(
        "--cache_dir", help="Optional directory for deterministic preprocessing caches."
    )
    parser.add_argument(
        "--probe_cache_dir",
        help="Existing VolumeProbe cache directory; required when VolumeProbe is enabled.",
    )
    parser.add_argument(
        "--use_volumeprobe_lite", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dist-backend", choices=["auto", "nccl", "gloo"], default="auto")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--noamp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--num_steps", type=int, help="Total optimizer steps, including steps restored from resume."
    )
    parser.add_argument(
        "--eval_num", type=int, default=500, help="Checkpoint save interval in steps."
    )
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--grad_clip", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--probe_candidate_prob", type=float, default=0.7)
    parser.add_argument("--probe_jitter_frac", type=float, default=0.05)
    parser.add_argument("--probe_axcodes", default="RAS")
    parser.add_argument("--probe_pixdim", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--probe_spacing_mode", default="bilinear")
    parser.add_argument(
        "--probe_shape_mismatch_mode", choices=["raise", "warn_and_random"], default="raise"
    )
    return parser


def validate_args(args):
    for name in ("data_dir", "probe_cache_dir", "list_dir"):
        value = getattr(args, name, None)
        if value and not Path(value).is_dir():
            raise NotADirectoryError(f"--{name}: {value}")
    for name in ("datalist", "resume", "dataset_train_list", "dataset_val_list", "pretrained_ckpt"):
        value = getattr(args, name, None)
        if value and not Path(value).is_file():
            raise FileNotFoundError(f"--{name}: {value}")
    if args.use_volumeprobe_lite and not args.probe_cache_dir:
        raise ValueError("Pass --probe_cache_dir or --no-use_volumeprobe_lite.")
    if (
        not args.datalist
        and not getattr(args, "list_dir", None)
        and not getattr(args, "dataset_train_list", None)
    ):
        raise ValueError(
            "Pass --datalist, --list_dir (VoCo/S2DC), or --dataset_train_list (MedGMAE)."
        )
    for name in ("batch_size", "eval_num", "log_every", "prefetch_factor"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive.")
    if args.num_workers < 0 or (args.num_steps is not None and args.num_steps < 1):
        raise ValueError("num_workers must be nonnegative and num_steps must be positive.")
    if hasattr(args, "epochs") and args.epochs < 1:
        raise ValueError("epochs must be positive.")
    if not 0 <= args.probe_candidate_prob <= 1:
        raise ValueError("probe_candidate_prob must be between zero and one.")
    args.data_dir = str(Path(args.data_dir).resolve())
    args.log_dir = str(Path(args.log_dir).resolve())
    if (
        args.weights_path
        and Path(args.weights_path).resolve() == Path(args.checkpoint_path).resolve()
    ):
        raise ValueError("weights_path and checkpoint_path must be different files.")


def init_log(args, ctx):
    logger = logging.getLogger(f"pretraining.rank{ctx.rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    if ctx.rank == 0:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("[%(asctime)s] %(message)s")
        for handler in (
            logging.StreamHandler(),
            logging.FileHandler(Path(args.log_dir) / "train.log", encoding="utf-8"),
        ):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def make_scheduler(args, optimizer, steps_per_epoch):
    if not getattr(args, "lrdecay", True):
        return None
    warmup = getattr(args, "warmup_steps", None)
    if warmup is None:
        warmup = args.warmup_epochs * steps_per_epoch
    total = (
        getattr(args, "epochs", 0) * steps_per_epoch
        if hasattr(args, "warmup_epochs")
        else args.num_steps
    )
    total = max(total, args.num_steps)

    def factor(step):
        if getattr(args, "lr_schedule", "warmup_cosine") == "poly":
            return max(0.0, 1.0 - step / max(1, total)) ** 0.9
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def ensure_finite(value, ctx, label, raise_error=True):
    valid = torch.isfinite(value.detach()).all().to(device=ctx.device, dtype=torch.int32)
    if ctx.distributed:
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    finite = bool(valid.item())
    if not finite and raise_error:
        raise FloatingPointError(
            f"Non-finite {label}; training stopped before saving invalid weights."
        )
    return finite


def run_training(args, model_factory, loader_factory, loss_fn, optimizer_factory=None):
    validate_args(args)
    ctx = init_distributed(args.device, args.dist_backend)
    try:
        args.rank, args.world_size, args.local_rank = ctx.rank, ctx.world_size, ctx.local_rank
        args.distributed = ctx.distributed
        args.device_obj = ctx.device
        args.amp = not args.noamp and ctx.device.type == "cuda"
        args.amp_dtype_torch = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
        if args.amp and args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported on this GPU; select --amp_dtype fp16.")
        set_determinism(args.seed + ctx.rank)
        logger = init_log(args, ctx)
        loader = loader_factory(args)
        if len(loader) == 0:
            raise ValueError(
                "The training loader has zero batches. Reduce batch_size or supply more images."
            )
        if args.num_steps is None:
            args.num_steps = args.epochs * len(loader)
        model = model_factory(args).to(ctx.device)
        if getattr(args, "pretrained_ckpt", None):
            weights = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=True)
            weights = weights.get("state_dict", weights)
            model.load_state_dict(
                {k.removeprefix("module."): v for k, v in weights.items()}, strict=True
            )
        if optimizer_factory:
            optimizer = optimizer_factory(args, model)
        else:
            decay, no_decay = [], []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    (
                        no_decay
                        if param.ndim <= 1 or "token" in name or "pos_embed" in name
                        else decay
                    ).append(param)
            optimizer = torch.optim.AdamW(
                [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
                lr=args.base_lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )
        scheduler = make_scheduler(args, optimizer, len(loader))
        scaler = torch.amp.GradScaler(
            "cuda", enabled=args.amp and args.amp_dtype == "fp16", init_scale=1024.0
        )
        global_step, epoch, batch_in_epoch = 0, 0, 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            state = {k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(state, strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            global_step = int(checkpoint["global_step"])
            if scheduler is not None:
                current_lrs = [
                    base_lr * schedule(global_step)
                    for base_lr, schedule in zip(scheduler.base_lrs, scheduler.lr_lambdas)
                ]
                for group, lr in zip(optimizer.param_groups, current_lrs):
                    group["lr"] = lr
                scheduler._last_lr = current_lrs
            epoch = int(checkpoint.get("epoch", global_step // len(loader)))
            batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
            logger.info("Resumed step=%d epoch=%d from %s", global_step, epoch, args.resume)
        if hasattr(model, "global_step"):
            model.global_step = global_step
        model = wrap_module(
            model,
            ctx,
            sync_batchnorm=getattr(args, "sync_batchnorm", False),
            find_unused_parameters=getattr(args, "ddp_find_unused_parameters", True),
        )
        random.seed(args.seed + ctx.rank)
        np.random.seed(args.seed + ctx.rank)
        torch.manual_seed(args.seed + ctx.rank)
        logger.info(
            "device=%s backend=%s world_size=%d amp=%s dtype=%s samples=%d batches_per_rank=%d parameters=%d",
            ctx.device,
            ctx.backend,
            ctx.world_size,
            args.amp,
            args.amp_dtype,
            len(loader.dataset),
            len(loader),
            sum(p.numel() for p in model.parameters()),
        )
        logger.info("checkpoint_path=%s", args.checkpoint_path)

        def save():
            if ctx.rank == 0:
                atomic_save(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "state_dict": unwrap_state_dict(model),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "scaler": scaler.state_dict(),
                        "world_size": ctx.world_size,
                        "num_steps": args.num_steps,
                    },
                    args.checkpoint_path,
                )

        model.train()
        last_metrics = {}
        consecutive_overflows = 0
        while global_step < args.num_steps:
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                if batch_index < batch_in_epoch:
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(ctx.device.type, dtype=args.amp_dtype_torch, enabled=args.amp):
                    loss, metrics = loss_fn(model, batch, ctx.device)
                ensure_finite(loss, ctx, "loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not ensure_finite(
                    grad_norm, ctx, "gradient norm", raise_error=not scaler.is_enabled()
                ):
                    consecutive_overflows += 1
                    scaler.step(optimizer)
                    scaler.update()
                    batch_in_epoch = batch_index + 1
                    logger.warning(
                        "FP16 gradient overflow; skipped update and reduced scale to %.1f",
                        scaler.get_scale(),
                    )
                    if consecutive_overflows >= 16:
                        raise FloatingPointError(
                            "Repeated FP16 overflows; try --amp_dtype bf16 or --noamp."
                        )
                    continue
                consecutive_overflows = 0
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
                global_step += 1
                batch_in_epoch = batch_index + 1
                raw = model.module if hasattr(model, "module") else model
                if hasattr(raw, "global_step"):
                    raw.global_step = global_step
                values = torch.stack(
                    [loss.detach().float().reshape(())]
                    + [v.detach().float().reshape(()) for v in metrics.values()]
                )
                if ctx.distributed:
                    dist.all_reduce(values)
                    values /= ctx.world_size
                last_metrics = dict(zip(["loss"] + list(metrics), values.tolist()))
                if global_step % args.log_every == 0 or global_step == args.num_steps:
                    logger.info(
                        "step=%d/%d %s grad_norm=%.6f lr=%.8g",
                        global_step,
                        args.num_steps,
                        json.dumps(last_metrics),
                        grad_norm.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                if global_step % args.eval_num == 0:
                    save()
                if global_step >= args.num_steps:
                    break
            if batch_in_epoch >= len(loader):
                epoch += 1
                batch_in_epoch = 0
        save()
        if ctx.rank == 0:
            if args.weights_path:
                atomic_save(unwrap_state_dict(model), args.weights_path)
            result = {
                "global_step": global_step,
                "epoch": epoch,
                "world_size": ctx.world_size,
                "device": str(ctx.device),
                "metrics": last_metrics,
            }
            (Path(args.log_dir) / "training_summary.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            logger.info("Training complete at step=%d; saved %s", global_step, args.checkpoint_path)
        if ctx.distributed:
            dist.barrier()
    finally:
        cleanup(ctx)
