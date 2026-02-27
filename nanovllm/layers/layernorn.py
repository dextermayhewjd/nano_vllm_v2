import torch
import torch.nn as nn 
import math 
from einops import reduce

class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1,keepdim=True)
        x.mul_(torch.rsqrt(mean_square + self.eps))
        
        # 必须要转换回 bf16后再和本地的layernorm层 相乘
        x.to(in_dtype).mul_(self.weight)
        return x