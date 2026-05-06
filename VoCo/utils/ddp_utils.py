import inspect
import math
import os
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DDPContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

def _source_shell_env(script_path: str):
    if not script_path or not os.path.isfile(script_path):
        return
    if os.environ.get("ONECCL_VARS_SOURCED") == script_path:
        return
    proc = subprocess.run(
        ["bash", "-lc", f"source {script_path} >/dev/null 2>&1 && env -0"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return
    keep_prefixes = (
        "CCL_",
        "FI_",
        "I_MPI",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PATH",
        "CPATH",
        "PKG_CONFIG_PATH",
    )
    for chunk in proc.stdout.decode("utf-8", errors="ignore").split("\0"):
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        if key.startswith(keep_prefixes):
            os.environ[key] = value
    os.environ["ONECCL_VARS_SOURCED"] = script_path


def _prepare_xpu_distributed_backend(backend: str):
    if backend not in ("xccl", "ccl"):
        return
    try:
        import oneccl_bindings_for_pytorch as torch_ccl  # noqa: F401
    except Exception:
        torch_ccl = None
    if torch_ccl is not None:
        env_script = os.path.join(getattr(torch_ccl, "cwd", ""), "env", "setvars.sh")
        _source_shell_env(env_script)
    os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE")
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
    os.environ.setdefault("CCL_WORKER_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("I_MPI_OFFLOAD", "1")


def resolve_device(device_name: str = "auto", local_rank: int = 0) -> torch.device:
    if device_name not in ("auto", "cuda", "xpu", "cpu"):
        return torch.device(device_name)
    if device_name in ("auto", "xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        if hasattr(torch.xpu, "set_device"):
            torch.xpu.set_device(local_rank)
        return torch.device(f"xpu:{local_rank}")
    if device_name in ("auto", "cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def init_distributed(device_name: str = "auto", backend: str = "gloo") -> DDPContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    _prepare_xpu_distributed_backend(backend)
    device = resolve_device(device_name, local_rank=local_rank)
    if distributed and not dist.is_initialized():
        init_kwargs = {"backend": backend, "init_method": "env://"}
        try:
            if "device_id" in inspect.signature(dist.init_process_group).parameters:
                init_kwargs["device_id"] = device
        except Exception:
            pass
        dist.init_process_group(**init_kwargs)
    return DDPContext(distributed=distributed, rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def is_main_process(ctx: DDPContext) -> bool:
    return ctx.rank == 0


def silence_non_main(ctx: DDPContext):
    if not is_main_process(ctx):
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull


def wrap_module(module: torch.nn.Module, ctx: DDPContext) -> torch.nn.Module:
    if not ctx.distributed:
        return module
    ddp_kwargs = {
        "broadcast_buffers": False,
    }
    if ctx.device.type in ("xpu", "cuda"):
        ddp_kwargs["device_ids"] = [ctx.local_rank]
        ddp_kwargs["output_device"] = ctx.local_rank

    sig = None
    try:
        sig = inspect.signature(DDP.__init__)
    except Exception:
        sig = None

    ddp = None
    if sig is not None and "static_graph" in sig.parameters:
        try:
            ddp = DDP(module, static_graph=True, find_unused_parameters=False, **ddp_kwargs)
        except TypeError:
            ddp = None

    if ddp is None:
        ddp = DDP(module, find_unused_parameters=False, **ddp_kwargs)
        if hasattr(ddp, "_set_static_graph"):
            try:
                ddp._set_static_graph()
            except Exception:
                pass

    return ddp


def unwrap_state_dict(module: torch.nn.Module):
    if hasattr(module, "module"):
        return module.module.state_dict()
    return module.state_dict()


def build_sampler(dataset, ctx: DDPContext, shuffle: bool = True):
    if not ctx.distributed:
        return None
    return DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=shuffle, drop_last=False)


def aligned_steps_for_world(default_steps: int, world_size: int) -> int:
    return max(1, int(math.ceil(float(default_steps) / max(1, world_size))))


def cleanup(ctx: DDPContext):
    if ctx.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
