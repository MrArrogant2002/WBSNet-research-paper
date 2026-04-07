from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def init_distributed(backend: str = "nccl") -> DistributedState:
    world_size = _get_world_size()
    if world_size <= 1:
        return DistributedState(enabled=False, rank=0, world_size=1, local_rank=0)

    import torch
    import torch.distributed as dist

    local_rank = _get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return DistributedState(
        enabled=True,
        rank=_get_rank(),
        world_size=world_size,
        local_rank=local_rank,
    )


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(state: DistributedState) -> bool:
    return state.rank == 0


def barrier(state: DistributedState) -> None:
    if not state.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


def reduce_scalar(value: float, state: DistributedState, average: bool = True) -> float:
    if not state.enabled:
        return value

    import torch
    import torch.distributed as dist

    tensor = torch.tensor(float(value), device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= state.world_size
    return float(tensor.item())


def reduce_counts(values: dict[str, float], state: DistributedState) -> dict[str, float]:
    if not state.enabled:
        return values

    import torch
    import torch.distributed as dist

    keys = sorted(values)
    tensor = torch.tensor([float(values[key]) for key in keys], device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def gather_objects(payload: list[float], state: DistributedState) -> list[float]:
    if not state.enabled:
        return payload

    import torch.distributed as dist

    gathered: list[list[float]] = [list() for _ in range(state.world_size)]
    dist.all_gather_object(gathered, payload)
    merged: list[float] = []
    for chunk in gathered:
        merged.extend(chunk)
    return merged
