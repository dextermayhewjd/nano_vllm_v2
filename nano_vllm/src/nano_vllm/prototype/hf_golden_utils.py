"""Utility helpers for deterministic HF golden generation."""

from __future__ import annotations

import random
import time

import torch


def now_str() -> str:
    """Generate a compact local timestamp used for artifact names and metadata."""

    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def slug(text: str) -> str:
    """Turn free-form strings into filesystem-friendly fragments."""

    sanitized = text.replace("/", "_").replace(":", "_").replace(" ", "_")
    return "".join(ch for ch in sanitized if ch.isalnum() or ch in ("_", "-", "."))


def parse_dtype(dtype_name: str) -> torch.dtype:
    """Map CLI dtype string to a torch dtype."""

    lowered = dtype_name.lower()
    if lowered in ("fp16", "float16"):
        return torch.float16
    if lowered in ("bf16", "bfloat16"):
        return torch.bfloat16
    if lowered in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def set_seed(seed: int) -> None:
    """Set Python and torch seeds so the run is replayable.

    We set both CPU and CUDA seeds because model loading and generation can involve
    either side depending on selected device.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
