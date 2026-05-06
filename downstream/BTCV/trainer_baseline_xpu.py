import csv
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from torch import nn
from tqdm import tqdm

import numpy as np

from utils.utils import distributed_all_gather
from monai.data import decollate_batch


def _amp_enabled(args):
    device_type = getattr(args.device, "type", "cpu")
    if not bool(getattr(args, "amp", False)):
        return False
    if device_type not in ("cuda", "xpu", "cpu"):
        return False

    amp_mod = getattr(torch, "amp", None)
    autocast_mode = getattr(amp_mod, "autocast_mode", None)
    is_available_fn = getattr(autocast_mode, "is_autocast_available", None)
    if callable(is_available_fn):
        try:
            return bool(is_available_fn(device_type))
        except Exception:
            pass

    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cpu":
        return True
    return False


def _amp_dtype(device_type: str):
    if device_type == "cuda":
        return torch.float16
    if device_type in ("xpu", "cpu"):
        return torch.bfloat16
    return None


def _autocast_context(args):
    device_type = getattr(args.device, "type", "cpu")
    dtype = _amp_dtype(device_type)
    enabled = _amp_enabled(args)

    if device_type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "amp"):
        xpu_autocast = getattr(torch.xpu.amp, "autocast", None)
        if callable(xpu_autocast):
            return xpu_autocast(enabled=enabled, dtype=dtype)

    try:
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    except TypeError:
        return torch.autocast(device_type=device_type, enabled=enabled)


def _sync_device(device_type: str):
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _empty_device_cache(device_type: str):
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _get_grad_scaler(args):
    device_type = getattr(args.device, "type", "cpu")
    amp_enabled = _amp_enabled(args)
    use_grad_scaler = amp_enabled and _amp_dtype(device_type) == torch.float16
    if not use_grad_scaler:
        return None

    grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
    if grad_scaler_cls is None:
        if device_type == "cuda" and hasattr(torch.cuda, "amp"):
            return torch.cuda.amp.GradScaler(enabled=True)
        return None

    try:
        return grad_scaler_cls(device=device_type, enabled=True)
    except TypeError:
        try:
            return grad_scaler_cls(device_type, enabled=True)
        except TypeError:
            return grad_scaler_cls(enabled=True)


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(args, model, loader, optimizer, scaler, epoch, loss_dice, logger, global_step=0):
    model.train()
    run_loss = AverageMeter()

    progress_bar = tqdm(enumerate(loader), total=len(loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")

    for idx, batch_data in progress_bar:
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        data = data.to(args.device)
        target = target.to(args.device)

        with _autocast_context(args):
            logits = model(data)
            loss = loss_dice(logits, target)

        optimizer.zero_grad(set_to_none=True)
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        global_step += 1

        if args.distributed:
            loss_list = distributed_all_gather(
                [loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size,
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        steps = len(loader)
        interval = 1 if steps == 1 else max(1, steps // 2)
        log_now = (idx % interval == 0) and (steps == 1 or idx != 0)
        if log_now:
            if args.rank == 0:
                print(
                    "Epoch:{}, Global_step:{}, Seg_Loss:{:.4f}, Lr:{:.6f}".format(
                        epoch, global_step, loss.item(), optimizer.param_groups[0]["lr"]
                    )
                )
            logger.info(
                "Epoch:{}, Global_step:{}, Seg_Loss:{:.4f}, Lr:{:.6f}".format(
                    epoch, global_step, loss.item(), optimizer.param_groups[0]["lr"]
                )
            )

    return run_loss.avg, global_step


def _batch_mean_dice_from_onehot(y_pred, y, include_background=False, eps=1e-6):
    if not include_background:
        y_pred = y_pred[:, 1:]
        y = y[:, 1:]

    dims = tuple(range(2, y_pred.ndim))
    intersect = (y_pred * y).sum(dim=dims)
    denom = y_pred.sum(dim=dims) + y.sum(dim=dims)

    valid = (denom > 0).float()
    dice = torch.where(
        denom > 0,
        2.0 * intersect / denom.clamp_min(eps),
        torch.zeros_like(denom),
    )
    case_valid = valid.sum(dim=1)
    case_dice = (dice * valid).sum(dim=1) / case_valid.clamp_min(1.0)
    case_not_nans = (case_valid > 0).float()
    return case_dice, case_not_nans


def val_epoch(args, model, loader, epoch, acc_func, model_inferer, post_label, post_pred, logger):
    model.eval()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data = data.to(args.device)
            target = target.to(args.device)

            with _autocast_context(args):
                logits = model_inferer(data) if model_inferer is not None else model(data)

            val_labels = torch.stack(
                [post_label(x) for x in decollate_batch(target)], dim=0
            ).float().to(args.device)

            val_preds = torch.stack(
                [post_pred(x) for x in decollate_batch(logits)], dim=0
            ).float().to(args.device)

            acc, not_nans = _batch_mean_dice_from_onehot(
                val_preds, val_labels, include_background=False
            )

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans],
                    out_numpy=True,
                    is_valid=idx < loader.sampler.valid_length,
                )
                for al, nl in zip(acc_list, not_nans_list):
                    al = np.asarray(al).reshape(-1)
                    nl = np.asarray(nl).reshape(-1)
                    for a, n in zip(al, nl):
                        if n > 0:
                            run_acc.update(float(a), n=float(n))
            else:
                acc_np = acc.detach().cpu().numpy().reshape(-1)
                nn_np = not_nans.detach().cpu().numpy().reshape(-1)
                for a, n in zip(acc_np, nn_np):
                    if n > 0:
                        run_acc.update(float(a), n=float(n))

    _empty_device_cache(args.device.type)
    return run_acc.avg


def save_checkpoint(
    model,
    epoch,
    args,
    filename="model.pt",
    best_dice=0,
    optimizer=None,
    scheduler=None,
    global_step=0,
):
    epoch_name = str(epoch)
    filename = "checkpoint_%s.pth" % epoch_name
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
        "model": state_dict,
        "args": args,
        "epoch": epoch,
        "best_dice": best_dice,
        "global_step": int(global_step),
    }
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.model_dir, filename)
    torch.save(save_dict, filename)


def save_checkpoint_curr(
    model,
    epoch,
    args,
    filename="model_current.pt",
    best_dice=0,
    optimizer=None,
    scheduler=None,
    global_step=0,
):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
        "model": state_dict,
        "args": args,
        "epoch": epoch,
        "best_dice": best_dice,
        "global_step": int(global_step),
    }
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.model_dir, filename)
    torch.save(save_dict, filename)


def run_training(
    args,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_dice,
    acc_func,
    logger,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    start_global_step=0,
    start_best_dice=0.0,
    post_label=None,
    post_pred=None,
):
    scaler = _get_grad_scaler(args)

    global_step = int(start_global_step)
    val_acc_max = float(start_best_dice)

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_loss, global_step = train_epoch(
            args, model, train_loader, optimizer, scaler, epoch, loss_dice, logger, global_step=global_step
        )
        _sync_device(args.device.type)

        if args.rank == 0:
            if args.model_dir and (epoch % args.ckpt_interval == 0 or epoch + 1 == args.max_epochs):
                save_checkpoint(
                    model,
                    epoch,
                    args,
                    best_dice=val_acc_max,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                )

            if args.model_dir and epoch % args.curr_ckpt_interval == 0:
                save_checkpoint_curr(
                    model,
                    epoch,
                    args,
                    filename="model_current.pt",
                    best_dice=val_acc_max,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                )

        if (epoch + 1) % args.val_every == 0:
            val_avg_acc = val_epoch(args, model, val_loader, epoch, acc_func, model_inferer, post_label, post_pred, logger)
            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print("Validation Epoch:{}, Global_step:{}, Dice Score:{:.4f}".format(epoch, global_step, val_avg_acc))
                logger.info("Validation Epoch:{}, Global_step:{}, Dice Score:{:.4f}".format(epoch, global_step, val_avg_acc))

                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    logger.info("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc

        if scheduler is not None:
            scheduler.step()

    if args.rank == 0:
        print("Training Finished !, Best Dice: {}, Final Global_step: {}".format(val_acc_max, global_step))
    logger.info("Training Finished !, Best Dice: %s, Final Global_step: %d", val_acc_max, global_step)
    return val_acc_max
