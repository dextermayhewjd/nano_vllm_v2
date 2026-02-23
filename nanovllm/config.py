import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class Config:
    """Configuration scaffold for bootstrap stage."""
    """Core runtime config aligned with the first nano-vllm commit shape."""

    model: str
    # max_num_batched_tokens: int = 16384
    # max_num_seqs: int = 512
    # max_model_len: int = 4096
    # gpu_memory_utilization: float = 0.9
    # tensor_parallel_size: int = 1
    # enforce_eager: bool = False
    # hf_config: AutoConfig | None = None
    # eos: int = -1
    # kvcache_block_size: int = 256
    # num_kvcache_blocks: int = -1


        # model: 本地目录或 HF repo_id（必须提供）
    model: str = ""

    # # batching / scheduling
    # max_num_batched_tokens: int = 16384
    # max_num_seqs: int = 512
    # max_model_len: int = 4096

    # # GPU memory
    # gpu_memory_utilization: float = 0.95

    # # execution
    # enforce_eager: bool = False

    # # filled later (after loading hf config)
    # hf_config: AutoConfig | None = None

    # # tokenizer/model eos id (filled later)
    # eos: int = -1

    # # KV cache
    # kvcache_block_size: int = 256
    # num_kvcache_blocks: int = -1
    '''
    post init 之后做的验证
    '''
    model: str = ""
    def __post_init__(self) -> None:
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 256 == 0
        # assert 1 <= self.tensor_parallel_size <= 8
        # self.hf_config = AutoConfig.from_pretrained(self.model)
        # self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # assert self.max_num_batched_tokens >= self.max_model_len