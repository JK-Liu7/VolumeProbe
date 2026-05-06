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
import os

import numpy as np
import scipy.ndimage as ndimage
import torch

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None

from datetime import datetime
from time import time
import logging
import torch.distributed as dist



def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


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


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def _ensure_ipex_for_xpu(args):
    if getattr(args, "device_type", None) == "xpu" and ipex is None:
        raise ImportError(
            "Detected XPU/Dawn execution but intel_extension_for_pytorch is not available. "
            "Please load Dawn's pytorch-gpu conda environment."
        )


def get_device_type(args):
    user_device = getattr(args, "device_type", "auto")
    if user_device != "auto":
        return user_device

    if getattr(args, "server", None) == "Dawn":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def init_distributed_mode(args):
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.world_size = 1
    args.rank = 0
    args.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    args.device_type = get_device_type(args)

    if args.distributed:
        if args.device_type == "cuda":
            backend = "nccl"
            args.device = torch.device(f"cuda:{args.local_rank}")
            torch.cuda.set_device(args.local_rank)
            num_devices = torch.cuda.device_count()

        elif args.device_type == "xpu":
            os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
            os.environ.setdefault("CCL_ZE_IPC_EXCHANGE", "sockets")

            if os.environ.get("CCL_PROCESS_LAUNCHER") == "torchrun":
                os.environ["CCL_PROCESS_LAUNCHER"] = "torch"
            else:
                os.environ.setdefault("CCL_PROCESS_LAUNCHER", "torch")
            os.environ.setdefault("CCL_ZE_IPC_EXCHANGE", "sockets")
            os.environ.setdefault("CCL_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))
            os.environ.setdefault(
                "CCL_LOCAL_SIZE",
                os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
            )

            args.device = torch.device(f"xpu:{args.local_rank}")
            torch.xpu.set_device(args.local_rank)
            num_devices = torch.xpu.device_count()

            if hasattr(torch.distributed, "is_xccl_available") and torch.distributed.is_xccl_available():
                backend = "xccl"
            else:
                import oneccl_bindings_for_pytorch  # noqa: F401
                backend = "ccl"

        else:
            backend = "gloo"
            args.device = torch.device("cpu")
            num_devices = 1

        dist.init_process_group(backend=backend, init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()

        print(f"Setting up distributed training with {num_devices} {args.device_type.upper()} devices available")
        print(
            "Training in distributed mode. Process %d, total %d, device=%s."
            % (args.rank, args.world_size, str(args.device))
        )
    else:
        if args.device_type == "cuda":
            args.device = torch.device("cuda:0")
        elif args.device_type == "xpu":
            args.device = torch.device("xpu:1")
        else:
            args.device = torch.device("cpu")

        print(f"Training with a single process on device {args.device}.")

    assert args.rank >= 0


def create_logger(log_dir, distributed):
    today_date = datetime.today().strftime('%Y.%m.%d')
    if distributed:
        if dist.get_rank() == 0:
            logging.basicConfig(filename=log_dir + f"{today_date}.log",
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p',
                                force=True)
            logger = logging.getLogger(__name__)
        else:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
    else:
        logging.basicConfig(filename=log_dir + f"{today_date}.log",
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p',force=True)
        logger = logging.getLogger(__name__)

    return logger
