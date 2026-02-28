import torch
from torch import Tensor


class KVCache:
    """
    单层的 KV Cache，不带 batch 维度（batch=1）。

    预分配两块连续内存：
        k_cache: [num_kv_heads, max_seq_len, head_dim]
        v_cache: [num_kv_heads, max_seq_len, head_dim]

    seq_len 是一个指针，记录当前已经填入了多少个 token。
    prefill 时一次写入 S 个，decode 时每次写入 1 个。
    """

    def __init__(
        self,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.seq_len = 0  # 当前已填充的 token 数

        # 预分配，不初始化内容（zeros 只是方便 debug，实际无所谓）
        # 因为原本的 q k v 在换成 (b ,head_num ,seq_len ,head_dim)之后
        # 社区batch 维度之后是H Seq D_h 
        self.k_cache = torch.zeros(num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.v_cache = torch.zeros(num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)

    def write(self, k: Tensor, v: Tensor) -> None:
        """
        写入新的 K/V。
        k, v 形状: [num_kv_heads, new_tokens, head_dim]
        prefill 时 new_tokens = S
        decode 时 new_tokens = 1
        """
        new_tokens = k.shape[1] # 单个token只会是[head , 1,head_dim]
        
        # 这里是判断是否超过了最大上下文的限制 如果超过的话 kvcache 预留的内存会不够 
        assert self.seq_len + new_tokens <= self.max_seq_len, \
            f"KVCache overflow: seq_len={self.seq_len}, new={new_tokens}, max={self.max_seq_len}"

        # 这里取head_num 所有值， 以及head_dim 
        # seq_len 到seq_len +new_token 的值 和k以及v相同 
        # 维度要求是 k 为 [head , new_token_num, head_dim]
        self.k_cache[:, self.seq_len : self.seq_len + new_tokens, :] = k
        self.v_cache[:, self.seq_len : self.seq_len + new_tokens, :] = v
        self.seq_len += new_tokens

    def read(self) -> tuple[Tensor, Tensor]:
        """
        返回目前全部历史的 K/V。
        返回形状: ([num_kv_heads, seq_len, head_dim], [num_kv_heads, seq_len, head_dim])
        """
        return (
            self.k_cache[:, : self.seq_len, :],
            self.v_cache[:, : self.seq_len, :],
        )

    def reset(self) -> None:
        """清空 cache，下一轮新序列使用。"""
        self.seq_len = 0
