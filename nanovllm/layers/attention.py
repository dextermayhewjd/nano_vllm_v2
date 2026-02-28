import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

def scale_dot_product_attention(    
        Q: Float[Tensor, "batch ... queries d_k"],
        K: Float[Tensor, "batch ... keys d_k"],
        V: Float[Tensor, "batch ... keys d_v"],
        mask: Bool[Tensor, "... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
    
    d_k = Q.shape[-1]
    
    # 1. 计算缩放的点积得分
    # 使用 einsum 自动处理 Batch 和 Head 维度
    # multihead 的时候这里的d_k 和 d_k 都是head_dim 
    # 最后变成的是相似度矩阵 一般来说queries 和 keys都是 seq_len 的长度
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    
    # 2. 应用掩码
    if mask is not None:
        # 确保 mask 在同一个设备上，且取反 (~mask) 将 True(1) 变为 False(0)
        # 这样 mask 中为 False 的地方会被填充为 -inf
        scores = scores.masked_fill(~mask, float("-inf"))
        
    # 3. Softmax
    # 建议此处先用官方实现排查问题，如果过拟合成功，再换回自定义 softmax
    attn_probs = torch.nn.functional.softmax(scores, dim=-1)
    
    # 4. 加权求和
    # (..., q, k) @ (..., k, d_v) -> (..., q, d_v)
    result = einsum(attn_probs, V, "... q k, ... k d_v -> ... q d_v")
    
    return result




class GQAAttn(nn.Module):
    def __init__(self, num_q_heads: int, num_kv_heads: int, head_dim: int, causal: bool = True):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.groups = num_q_heads // num_kv_heads
        self.causal = causal

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                attn_mask=None) -> torch.Tensor:
        """
        q: [B, Hq,  Sq, D]   Sq = S (prefill) 或 1 (decode) 
        k: [B, Hkv, Sk, D]   Sk = 历史长度（含当前 token)
        v: [B, Hkv, Sk, D]    
        return: [B, Hq, Sq, D]
        """
        # 这是 q 和 k的shape  都是 (batch , head_num , seq_len , head_dim)         
        B, Hq, Sq, D = q.shape
        _, Hkv, Sk, Dk = k.shape
        
        assert Hq == self.num_q_heads 
        assert Hkv == self.num_kv_heads
        # prefill 要求 他们的seq 长是一样的 assert S == Sk 现在支持两种所以所以舍去
        # 注意：去掉 assert Sq == Sk，decode 时 Sq=1，Sk=历史 长度，不相等是正常的 
         
        #要求这里的head dim 是一样的  
        assert D == Dk

        # 1) reshape Q/K/V into grouped form                  
        # Q: [B, Hkv, G, Sq, D]  
        # K/V: [B*Hkv, G, Sk, D]（G 维度 expand）                                                   
        q = q.reshape(B, Hkv, self.groups, Sq, D).reshape(B * Hkv, self.groups, Sq, D)                                      
        k = k.reshape(B * Hkv, 1, Sk, D).expand(B * Hkv, self.groups, Sk, D)                                                
        v = v.reshape(B * Hkv, 1, Sk, D).expand(B * Hkv, self.groups, Sk, D) 
        # k/v 需要为每个 group “逻辑上共享”，但不 repeat

        is_causal=(self.causal and (Sq == Sk) and attn_mask is None)
        #如果没传 attn_mask，就用 is_causal=True（自动下三角）
            #如果你传了 attn_mask，就把 is_causal 关掉，完全依赖 attn_mask
        # 3) scaled_dot_product_attention
        # 这里用 PyTorch 的 SDPA，内部会自动走 FlashAttention / math / mem-efficient
        out = F.scaled_dot_product_attention( # 注意这里只接受3d或者4d 
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal
            
        )  # [B*Hkv*G, S, D]

        # 4) reshape 回来: [B, Hkv, G, S, D] -> [B, Hq, S, D]
        out = out.reshape(B, Hkv, self.groups, Sq, D).reshape(B, Hq, Sq, D)
        return out