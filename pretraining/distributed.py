import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DDPContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str


def init_distributed(device_name="cuda", backend="auto"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install a CUDA-enabled PyTorch build or select --device cpu for debugging."
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "LOCAL_RANK exceeds the visible GPU count; use one torchrun process per GPU."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    if world_size > 1:
        if device.type == "cuda" and (backend != "nccl" or not dist.is_nccl_available()):
            raise RuntimeError(
                "Multi-GPU training requires Linux and an NCCL-enabled PyTorch build. CPU DDP debugging uses --device cpu --dist-backend gloo."
            )
        if device.type == "cpu" and backend != "gloo":
            raise ValueError("CPU distributed training requires the gloo backend.")
        kwargs = dict(backend=backend, init_method="env://", timeout=timedelta(minutes=10))
        if device.type == "cuda":
            kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    return DDPContext(world_size > 1, rank, world_size, local_rank, device, backend)


def is_main_process(ctx):
    return ctx.rank == 0


def wrap_module(module, ctx, sync_batchnorm=False, find_unused_parameters=True):
    if not ctx.distributed:
        return module
    use_sync_bn = sync_batchnorm and ctx.device.type == "cuda"
    if use_sync_bn:
        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
    kwargs = dict(broadcast_buffers=use_sync_bn, find_unused_parameters=find_unused_parameters)
    if ctx.device.type == "cuda":
        kwargs.update(device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return DistributedDataParallel(module, **kwargs)


def unwrap_state_dict(module):
    return (module.module if hasattr(module, "module") else module).state_dict()


def build_sampler(dataset, ctx, shuffle=True, seed=0):
    if not ctx.distributed:
        return None
    return DistributedSampler(
        dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=shuffle, seed=seed
    )


def cleanup(ctx):
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()
