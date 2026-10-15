import argparse
import json
from pathlib import Path

import torch
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from downstream.data import get_loader
from downstream.factory import (
    BASELINES,
    build_model,
    checkpoint_weights,
    load_pretrained,
)
from downstream.lr_scheduler import LinearWarmupCosineAnnealingLR
from downstream.trainer import run_training
from pretraining.distributed import cleanup, init_distributed, wrap_module
from pretraining.runtime import init_log


DATASETS = {
    "BTCV": {"classes": 14, "epochs": 1000, "warmup": 100, "samples": 4},
    "Word": {"classes": 17, "epochs": 1000, "warmup": 5, "samples": 4},
    "Amos": {"classes": 16, "epochs": 1000, "warmup": 50, "samples": 4},
    "ATLAS-MRI": {"classes": 3, "epochs": 1000, "warmup": 5, "samples": 4},
}


def get_parser(dataset):
    defaults = DATASETS[dataset]
    parser = argparse.ArgumentParser(description=f"{dataset} segmentation on CUDA or torchrun DDP.")
    parser.set_defaults(dataset=dataset)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--json_list",
        required=True,
        help="Split JSON; absolute, relative to the working directory, or relative to data_dir.",
    )
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--cache_dir", help="Optional deterministic preprocessing cache directory.")
    parser.add_argument(
        "--checkpoint_path", help="Full output checkpoint path, including filename."
    )
    parser.add_argument(
        "--best_checkpoint_path",
        help="Optional full output path for the best checkpoint.",
    )
    parser.add_argument("--metrics_path", help="Optional per-case validation CSV path.")
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument(
        "--pretrained_ckpt", help="Full path to baseline pretraining weights."
    )
    initialization.add_argument(
        "--resume", "--checkpoint", dest="resume", help="Full training checkpoint path."
    )
    initialization.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--model", choices=BASELINES, default="VoCo")
    parser.add_argument("--device", "--device_type", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dist-backend", choices=["auto", "nccl", "gloo"], default="auto")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--sync_batchnorm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noamp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sw_batch_size", type=int, default=defaults["samples"])
    parser.add_argument("--infer_sw_batch_size", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_split", choices=["validation", "test"], default="validation")
    parser.add_argument("--optim_lr", type=float, default=3e-4)
    parser.add_argument("--optim_name", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument(
        "--lrschedule",
        choices=["warmup_cosine", "cosine_anneal", "none"],
        default="warmup_cosine",
    )
    parser.add_argument("--warmup_epochs", type=int, default=defaults["warmup"])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument(
        "--swin_v2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Infer Swin convolution blocks from supplied weights unless explicitly overridden.",
    )
    parser.add_argument("--unetr_feature_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=1536)
    parser.add_argument("--mlp_dim", type=int, default=6144)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=defaults["classes"])
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--dropout_path_rate", type=float, default=0.0)
    parser.add_argument("--use_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    for key, value in (
        ("a_min", -175.0),
        ("a_max", 250.0),
        ("b_min", 0.0),
        ("b_max", 1.0),
    ):
        parser.add_argument(f"--{key}", type=float, default=value)
    for axis in "xyz":
        parser.add_argument(f"--space_{axis}", type=float, default=1.5)
        parser.add_argument(f"--roi_{axis}", type=int, default=96)
    for key, value in (
        ("RandFlipd_prob", 0.2),
        ("RandRotate90d_prob", 0.2),
        ("RandScaleIntensityd_prob", 0.1),
        ("RandShiftIntensityd_prob", 0.5),
    ):
        parser.add_argument(f"--{key}", type=float, default=value)
    parser.add_argument("--infer_overlap", type=float, default=0.75)
    parser.add_argument("--smooth_dr", type=float, default=1e-6)
    parser.add_argument("--smooth_nr", type=float, default=0.0)
    parser.add_argument("--squared_dice", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(args, parser):
    if not Path(args.data_dir).is_dir():
        parser.error("data_dir must be an existing directory.")
    for name in ("pretrained_ckpt", "resume"):
        path = getattr(args, name)
        if path and not Path(path).is_file():
            parser.error(f"{name} must be an existing file.")
    if not args.eval_only and not args.checkpoint_path:
        parser.error("Training requires --checkpoint_path, including the output filename.")
    if args.eval_only and not args.resume:
        parser.error("Evaluation requires --resume with a trained segmentation checkpoint.")
    if args.checkpoint_path and args.best_checkpoint_path:
        if Path(args.checkpoint_path).resolve() == Path(args.best_checkpoint_path).resolve():
            parser.error("checkpoint_path and best_checkpoint_path must be different files.")
    if args.pretrained_ckpt:
        for output in (args.checkpoint_path, args.best_checkpoint_path):
            if output and Path(output).resolve() == Path(args.pretrained_ckpt).resolve():
                parser.error("Output checkpoints must not overwrite pretraining weights.")
    for name in (
        "batch_size",
        "sw_batch_size",
        "infer_sw_batch_size",
        "max_epochs",
        "val_every",
        "save_every",
        "cpu_threads",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive.")
    if args.workers < 0 or not 0 <= args.infer_overlap < 1:
        parser.error("workers must be nonnegative and infer_overlap must be in [0, 1).")
    if args.lrschedule == "warmup_cosine" and not 1 < args.warmup_epochs < args.max_epochs:
        parser.error(
            "warmup_cosine requires 1 < warmup_epochs < max_epochs; use --lrschedule none for short smoke runs."
        )


def main(dataset, argv=None):
    parser = get_parser(dataset)
    args = parser.parse_args(argv)
    validate_args(args, parser)
    torch.set_num_threads(args.cpu_threads)
    ctx = init_distributed(args.device, args.dist_backend)
    args.device = ctx.device
    args.rank, args.world_size = ctx.rank, ctx.world_size
    args.distributed = ctx.distributed
    args.amp = not args.noamp and ctx.device.type == "cuda"
    set_determinism(seed=args.seed + ctx.rank)
    logger = init_log(args, ctx)
    try:
        checkpoint = None
        initial_path = args.resume or args.pretrained_ckpt
        if initial_path:
            checkpoint = torch.load(initial_path, map_location="cpu", weights_only=False)
        if args.swin_v2 is None:
            args.swin_v2 = checkpoint is not None and any(
                ".layers1c." in key for key in checkpoint_weights(checkpoint)
            )
        model = build_model(args)
        if args.pretrained_ckpt:
            load_pretrained(model, args.pretrained_ckpt, args.model, logger, checkpoint=checkpoint)
            checkpoint = None
        if args.resume:
            model.load_state_dict(checkpoint_weights(checkpoint), strict=True)
            saved = checkpoint.get("config", {})
            for key in ("dataset", "model", "out_channels", "roi_x", "roi_y", "roi_z"):
                if key in saved and saved[key] != getattr(args, key):
                    raise ValueError(f"Resume {key} differs from the saved training configuration.")
        model = wrap_module(
            model.to(ctx.device),
            ctx,
            sync_batchnorm=args.sync_batchnorm,
            find_unused_parameters=False,
        )
        loaders, _ = get_loader(args)
        if args.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.optim_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True,
            )
        else:
            cls = torch.optim.AdamW if args.optim_name == "adamw" else torch.optim.Adam
            optimizer = cls(model.parameters(), lr=args.optim_lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.lrschedule == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
            )
        elif args.lrschedule == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if checkpoint is not None and not args.eval_only:
            if "optimizer" not in checkpoint or "epoch" not in checkpoint:
                raise ValueError(
                    "Resume requires a full training checkpoint, including optimizer and epoch."
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler and checkpoint.get("scheduler"):
                scheduler.load_state_dict(checkpoint["scheduler"])
        loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=args.squared_dice,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr,
        )
        if ctx.rank == 0:
            config = {
                key: str(value) if isinstance(value, torch.device) else value
                for key, value in vars(args).items()
            }
            (Path(args.log_dir) / "run_config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )
        logger.info(
            "Dataset=%s model=%s device=%s ranks=%d",
            dataset,
            args.model,
            ctx.device,
            ctx.world_size,
        )
        resume_state = None
        if checkpoint is not None:
            resume_state = {
                key: checkpoint[key]
                for key in ("epoch", "global_step", "best_dice", "scaler")
                if key in checkpoint
            }
        del checkpoint
        return run_training(
            args,
            model,
            loaders[0],
            loaders[1],
            optimizer,
            loss,
            logger,
            scheduler=scheduler,
            checkpoint=resume_state,
        )
    finally:
        cleanup(ctx)
