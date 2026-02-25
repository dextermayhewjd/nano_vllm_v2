import torch.nn as nn
import torch
import math

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()
        '''
        num_embeddings: int 
                      Size of the vocabulary
        embedding_dim: int 
                      Dimension of the embedding vectors, 
                      i.e., dmodel
        device: torch.device | None = None 
                      Device to store the parameters on
        dtype: torch.dtype | None = None 
                      Data type of the parameters
        '''

        self.weights = nn.Parameter(data=torch.empty(num_embeddings,
                                                      embedding_dim,
                                                      device= device,
                                                      dtype=dtype)
                                    )
    def weight_loader(param:nn.Parameter, loaded_weight:torch.Tensor):
        
        param.data.copy_(loaded_weight)
        
    def forward(self,token_ids:torch.Tensor)-> torch.Tensor:
        return self.weights[token_ids]
