"""T0-1 package bootstrap entry.
Why this file now:
- This unlocks incremental follow-up files (llm.py, sampling_params.py) without changing more than one file per step.
"""
from .llm import LLM
from .sampling_params import SamplingParams

__all__: list[str] = ["LLM", "SamplingParams"]