"""Multiprocess helpers that avoid BLAS/OpenMP thread oversubscription."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def configure_cpu_threads(num_threads: int = 1) -> None:
    """Limit BLAS/OpenMP threads in the current process.

    Call this before importing numpy or scipy. When using
    ``ProcessPoolExecutor``, also pass ``init_worker`` as ``initializer``.
    """
    n = str(max(1, int(num_threads)))
    for name in _THREAD_ENV_VARS:
        os.environ[name] = n


def init_worker(num_threads: int = 1) -> None:
    """ProcessPoolExecutor initializer: one BLAS thread per worker."""
    configure_cpu_threads(num_threads)


def make_process_pool(
    *,
    max_workers: int,
    threads_per_worker: int = 1,
    mp_start_method: str = "spawn",
) -> ProcessPoolExecutor:
    """Return a ProcessPoolExecutor configured for CPU-bound batch scoring."""
    ctx = mp.get_context(mp_start_method)
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(threads_per_worker,),
    )


def pool_kwargs(
    *,
    max_workers: int,
    threads_per_worker: int = 1,
    mp_start_method: str = "spawn",
) -> dict[str, Any]:
    """Keyword arguments for ``ProcessPoolExecutor(...)``."""
    ctx = mp.get_context(mp_start_method)
    return {
        "max_workers": max_workers,
        "mp_context": ctx,
        "initializer": init_worker,
        "initargs": (threads_per_worker,),
    }
