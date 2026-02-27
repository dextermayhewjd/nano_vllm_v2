import torch
from torch import nn
import math

# 这是论文的实现方式 
# class RotaryEmbedding(nn.Module):

#     def __init__(
#         self,
#         head_dim:int,
#         max_seq_len:int,
#         rope_theta:int
#     ) -> None:
#         super().__init__()
#         self.head_dimension = head_dim
#         self.rope_theta = rope_theta
#         self.max_seq_len = max_seq_len

#         # 1. 使用 log 空间计算 inv_freq，增加数值稳定性
#         # inv_freq 形状: (head_dim/2,) [0,2,4, ... head_dim-2]
#         # 公式: theta^(-2i/head_dim)
#         power_part = torch.arange(0,head_dim,2).float() / head_dim
#         inv_freq = 1.0 / (self.rope_theta ** power_part)
        
#         # 2. 预计算所有位置的 angles
#         # pos 形状: (max_seq_len,)
#         pos = torch.arange(self.max_seq_len,dtype=torch.float)
        
#         # 外积 angles: (seq_len, head_dim/2)  = pos[:,None] * inv_freq[None,:]
#         angles = pos[:, None] * inv_freq[None, :]
                
#         # 3. 预计算 cos 和 sin 并注册为 buffer
#         # 形状都是 (max_seq_len, head_dim/2)
#         cos = torch.cos(angles)
#         sin = torch.sin(angles)

#         self.register_buffer("cos_cached", cos, persistent=False)
#         self.register_buffer("sin_cached", sin, persistent=False)
        
        
#     def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         实现交替旋转: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
#         """
#         # 确保内存连续，防止 flatten 导致数据乱序
#         x = x.contiguous()
        
#         x_even = x[..., 0::2] # x0, x2...
#         x_odd = x[..., 1::2]  # x1, x3...
        
#         # 拼接成 [..., head_dim/2, 2] 然后展平
#         rot = torch.stack((-x_odd, x_even), dim=-1)
#         return rot.flatten(-2)
    
#     def forward(self,
#                 x: torch.Tensor,
#                 token_positions: torch.Tensor) -> torch.Tensor:
        
#         token_positions = token_positions.to(device=x.device)
        
#         # 1. 根据 token_positions 提取对应的 cos/sin
#         cos = self.cos_cached[token_positions]
#         sin = self.sin_cached[token_positions]
        
#         # 2. 适配 4D Tensor (B, H, T, D)
#         # 如果输入有 Head 维度，我们需要在 dim=1 插入一个维度以便广播
#         if x.ndim == 4:
#             cos = cos.unsqueeze(1) # (B, 1, T, d_k/2)
#             sin = sin.unsqueeze(1) # (B, 1, T, d_k/2)
            
#         #  将 head_dim/2 扩展回 d_k
#         # 使用 repeat_interleave 配合交替旋转逻辑: [c0, c1] -> [c0, c0, c1, c1]
#         cos = cos.repeat_interleave(2, dim=-1).to(dtype=x.dtype)
#         sin = sin.repeat_interleave(2, dim=-1).to(dtype=x.dtype)
        
#          # 4. 应用旋转矩阵公式
#          # 注意旋转矩阵公式是 Rx 旋转矩阵在左边 x是列向量
#         # 
#         return (x * cos) + (self._rotate_half(x) * sin)


# 这是hf的实现方式  
class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim:int,
        max_seq_len:int,
        rope_theta:int
    ) -> None:
        super().__init__()
        self.head_dimension = head_dim
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        # 1. 使用 log 空间计算 inv_freq，增加数值稳定性
        # inv_freq 形状: (head_dim/2,) [0,2,4, ... head_dim-2]
        # 公式: theta^(-2i/head_dim)
        power_part = torch.arange(0,head_dim,2).float() / head_dim
        inv_freq = 1.0 / (self.rope_theta ** power_part)
        
        # 2. 预计算所有位置的 angles
        # pos 形状: (max_seq_len,)
        pos = torch.arange(self.max_seq_len,dtype=torch.float)
        
        # 外积 angles: (seq_len, head_dim/2)  = pos[:,None] * inv_freq[None,:]
        angles = pos[:, None] * inv_freq[None, :]
                
        # 3. 预计算 cos 和 sin 并注册为 buffer
        # 形状都是 (max_seq_len, head_dim/2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # 这里存的精度是bf16 
        self.register_buffer("cos_cached", cos.to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", sin.to(torch.bfloat16), persistent=False)
        
        
    def _apply_rotary_emb(self, 
                          x: torch.Tensor,
                          cos: torch.Tensor,
                            sin: torch.Tensor,
                          ) -> torch.Tensor:
        """
        实现交替旋转: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        """
        # 确保内存连续，防止 flatten 导致数据乱序
        x = x.contiguous()
        
        a, b = torch.chunk(x, 2, dim=-1)
        
        a1 = a * cos - b * sin
        b1 = b * cos + a * sin
        return torch.cat((a1, b1), dim=-1)
    
    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:
        
        token_positions = token_positions.to(device=x.device)
        
        # 1. 根据 token_positions 提取对应的 cos/sin
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
                # 2. 适配 4D Tensor (B, H, T, D)
        # 如果输入有 Head 维度，我们需要在 dim=1 插入一个维度以便广播
        if x.ndim == 4:
            cos = cos.unsqueeze(1) # (B, 1, T, d_k/2)
            sin = sin.unsqueeze(1) # (B, 1, T, d_k/2)

        x = self._apply_rotary_emb(x,cos=cos,sin=sin)
        return x
