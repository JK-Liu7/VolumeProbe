import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from monai.inferers import sliding_window_inference

from pretraining.distributed import unwrap_state_dict
from pretraining.runtime import atomic_save


def autocast(args):
    if not args.amp:
        return nullcontext()
    dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def tensor_to_device(value, device):
    if hasattr(value, "as_tensor"):
        value = value.as_tensor()
    return value.to(device, non_blocking=device.type == "cuda")


def train_epoch(args, model, loader, optimizer, scaler, loss_function, global_step):
    model.train()
    totals = torch.zeros(2, dtype=torch.float64, device=args.device)
    for batch in loader:
        image = tensor_to_device(batch["image"], args.device).float()
        label = tensor_to_device(batch["label"], args.device).long()
        optimizer.zero_grad(set_to_none=True)
        with autocast(args):
            loss = loss_function(model(image), label)
        finite = torch.isfinite(loss).to(torch.int32)
        if args.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Training loss is not finite.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        totals[0] += loss.detach().double() * image.shape[0]
        totals[1] += image.shape[0]
        global_step += 1
    if args.distributed:
        dist.all_reduce(totals)
    return (totals[0] / totals[1].clamp_min(1)).item(), global_step


@torch.no_grad()
def val_epoch(args, model, loader, collect_records=False):
    predictor = model.module if hasattr(model, "module") else model
    predictor.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=args.device)
    records = []
    for index, batch in enumerate(loader):
        image = tensor_to_device(batch["image"], args.device).float()
        label = tensor_to_device(batch["label"], args.device).long()
        with autocast(args):
            logits = sliding_window_inference(
                image,
                (args.roi_x, args.roi_y, args.roi_z),
                args.infer_sw_batch_size,
                predictor,
                overlap=args.infer_overlap,
            )
        prediction = logits.argmax(1)
        truth = label[:, 0]
        for item in range(image.shape[0]):
            scores = []
            row = {}
            for cls in range(1, args.out_channels):
                predicted = prediction[item] == cls
                expected = truth[item] == cls
                denominator = predicted.sum() + expected.sum()
                score = (
                    (2.0 * (predicted & expected).sum() / denominator).item()
                    if denominator
                    else float("nan")
                )
                row[f"dice_class_{cls}"] = score
                if np.isfinite(score):
                    scores.append(score)
            mean = float(np.mean(scores)) if scores else float("nan")
            if scores:
                totals[0] += mean
                totals[1] += 1
            if collect_records:
                metadata = getattr(batch["image"], "meta", {})
                names = metadata.get("filename_or_obj", [f"case_{args.rank}_{index}_{item}"])
                row["image"] = str(names[item] if isinstance(names, (list, tuple)) else names)
                row["mean_dice"] = mean
                records.append(row)
    if args.distributed:
        dist.all_reduce(totals)
        if collect_records:
            gathered = [None] * args.world_size
            dist.all_gather_object(gathered, records)
            records = [row for shard in gathered for row in shard]
    mean = (totals[0] / totals[1]).item() if totals[1] else float("nan")
    if args.rank == 0 and collect_records:
        path = Path(args.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["image", "mean_dice"] + [f"dice_class_{i}" for i in range(1, args.out_channels)]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(records, key=lambda row: row["image"]))
    return mean, int(totals[1].item())


def run_training(
    args,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_function,
    logger,
    scheduler=None,
    checkpoint=None,
):
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.amp_dtype == "fp16")
    if checkpoint and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    global_step = int(checkpoint.get("global_step", 0)) if checkpoint else 0
    best_dice = float(checkpoint.get("best_dice", -1.0)) if checkpoint else -1.0
    start_epoch = int(checkpoint.get("epoch", -1)) + 1 if checkpoint else 0
    if args.eval_only:
        mean, count = val_epoch(args, model, val_loader, bool(args.metrics_path))
        logger.info("Evaluation mean Dice=%s cases=%d", mean, count)
        return mean
    if start_epoch >= args.max_epochs:
        raise ValueError("max_epochs must exceed the epoch restored from the checkpoint.")
    history = []
    for epoch in range(start_epoch, args.max_epochs):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        loss, global_step = train_epoch(
            args, model, train_loader, optimizer, scaler, loss_function, global_step
        )
        dice = None
        improved = False
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.max_epochs:
            dice, case_count = val_epoch(args, model, val_loader, bool(args.metrics_path))
            improved = np.isfinite(dice) and dice > best_dice
            if improved:
                best_dice = dice
            logger.info("Validation epoch=%d mean Dice=%s cases=%d", epoch + 1, dice, case_count)
        if scheduler:
            scheduler.step()
        logger.info(
            "Epoch=%d global_step=%d loss=%.6f lr=%.8f",
            epoch + 1,
            global_step,
            loss,
            optimizer.param_groups[0]["lr"],
        )
        if args.rank == 0:
            state = {
                "state_dict": unwrap_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_dice": best_dice,
                "config": {
                    key: str(value) if isinstance(value, torch.device) else value
                    for key, value in vars(args).items()
                },
            }
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.max_epochs:
                atomic_save(state, args.checkpoint_path)
            if improved and args.best_checkpoint_path:
                atomic_save(state, args.best_checkpoint_path)
            history.append(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "loss": loss,
                    "dice": dice,
                    "best_dice": best_dice,
                }
            )
            with (Path(args.log_dir) / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(history[-1]) + "\n")
        if args.distributed:
            dist.barrier()
    return best_dice
