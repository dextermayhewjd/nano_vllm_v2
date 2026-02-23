"""Data structures used by the HF golden generator prototype."""

from dataclasses import dataclass


@dataclass
class RunMeta:
    """Serializable metadata that fully describes one golden-generation run.

    Keeping all runtime knobs in one place helps future replacement implementations
    replay exactly the same setup and detect mismatches faster.
    """

    model: str
    prompt: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    topk: int
    seed: int
    device: str
    dtype: str
    torch_version: str
    transformers_version: str
    created_at: str
    dump_breakpoints: bool
