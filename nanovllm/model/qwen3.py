import torch
from torch import nn
from transformers import Qwen3Config
from nanovllm.layers.linear import Linear
from nanovllm.layers.layernorn import RMSNorm
from nanovllm.layers.embed_head import Embedding
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config:Qwen3Config ):
        super().__init__()
        
        head_dim = config.head_dim
        
        # q_norm/k_norm：RMSNorm(head_dim)
        self.q_norm = RMSNorm(hidden_size=head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(hidden_size=head_dim, eps=config.rms_norm_eps)
        
        # 这四个必须存在，名字要完全对上：q_proj/k_proj/v_proj/o_proj
        # 维度：常见是
        # q: hidden -> num_heads*head_dim
        # k,v: hidden -> num_kv_heads*head_dim
        # o: (num_heads*head_dim) -> hidden
        
        q_out = config.num_attention_heads * head_dim
        kv_out = config.num_key_value_heads * head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, config.hidden_size, bias=False)
        
        
class Qwen3MLP(nn.Module):
    def __init__(
        self,
        config:Qwen3Config ):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    

class Qwen3DecoderLayer(nn.Module):
    def __init__(self,
                 config: Qwen3Config
                 ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps) # 放一个layer 带weight的
        self.mlp = Qwen3MLP(config=config) 
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps) # 放一个layer 带weight的
        self.self_attn = Qwen3Attention(config=config)
        

class Qwen3Model(nn.Module):
    def __init__(self,
                 config:Qwen3Config
                 ):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, 
                                         embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config=config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(
                            hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps                
                            )
    
class Qwen3ForCausalLM(nn.Module):
    def __init__(self,
                config:Qwen3Config
                ):
        super().__init__()
        
        self.model = Qwen3Model(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        