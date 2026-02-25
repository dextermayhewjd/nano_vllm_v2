import torch
import torch.nn as nn
from einops import einsum
class Linear(nn.Module):
    def __init__(self,
                in_features:int,
                out_features:int,
                device:torch.device | None = None,
                dtype: torch.dtype |None = None
                ):
        super().__init__
        
        self.weight = nn.Parameter(torch.empty(out_features,in_features))
        
    def forward(self, x:torch.Tensor):
        return einsum(x,self.weight,'b seq input_f, out_f input_f -> batch seq out_f')
                