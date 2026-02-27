import torch
from torch import nn
from transformers import Qwen3Config
from nanovllm.layers.linear import Linear
from nanovllm.layers.layernorn import RMSNorm
from nanovllm.layers.embed_head import Embedding
from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.layers.attention import GQAAttn
import torch.nn.functional as F
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config:Qwen3Config ):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        
        self.head_dim = config.head_dim
        self.num_attention_heads =config.num_attention_heads # Hq
        self.num_key_value_heads = config.num_key_value_heads # Hkv
        
        # rope 相关的
        
        # q_norm/k_norm：RMSNorm(head_dim)
        self.q_norm = RMSNorm(hidden_size=self.head_dim, eps=self.rms_norm_eps)
        self.k_norm = RMSNorm(hidden_size=self.head_dim, eps=self.rms_norm_eps)
        
        # 这四个必须存在，名字要完全对上：q_proj/k_proj/v_proj/o_proj
        # 维度：常见是
        # q: hidden -> num_heads*head_dim
        # k,v: hidden -> num_kv_heads*head_dim
        # o: (num_heads*head_dim) -> hidden
        
        q_out = self.num_attention_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, self.hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(
                                    head_dim = self.head_dim,
                                    max_seq_len = config.max_position_embeddings,
                                    rope_theta = config.rope_parameters["rope_theta"]
                                    )
        self.attention = GQAAttn(num_q_heads= self.num_attention_heads,
                                 num_kv_heads= self.num_key_value_heads,
                                 head_dim= self.head_dim,
                                 causal=True
                                 )
    def forward(
                self,
                token_position:torch.Tensor,
                hidden_states:torch.Tensor
                ):
        
        B, S, _ = hidden_states.shape
        Hq, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        
        # ---- 1) QKV 投影 ----
        q = self.q_proj(hidden_states)  # [B, S, Hq*D]
        k = self.k_proj(hidden_states)  # [B, S, Hkv*D]
        v = self.v_proj(hidden_states)  # [B, S, Hkv*D]
        
                # ---- 2) reshape 成多头 ----
        # 变成 [B, H, S, D]（注意 transpose）
        q = q.view(B, S, Hq,  D).transpose(1, 2)   # [B, Hq,  S, D]
        k = k.view(B, S, Hkv, D).transpose(1, 2)   # [B, Hkv, S, D]
        v = v.view(B, S, Hkv, D).transpose(1, 2)   # [B, Hkv, S, D]
        
        # ---- 3) Qwen3 的 q_norm / k_norm（按 head_dim 做 RMSNorm）----
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ---- 4) RoPE（在这里对 q/k 做旋转）----
        q = self.rope(
                hidden_states = q,
                token_position=token_position        
                        )
        k = self.rope(
                    hidden_states = k,
                    token_position = token_position
                    )
        # ---- 5) GQA Attention ----
        # q: [B, Hq, S, D], k/v: [B, Hkv, S, D] -> out: [B, Hq, S,D]
        attn_out = self.attention(q,k,v)
        
        # ---- 6) reshape 回 [B, S, Hq*D] 再过 o_proj ----
        attn_out = attn_out.transpose(1,2).contiguous().reshape(B, S, Hq * D)
        output = self.o_proj(attn_out)
        
        return output
        
class Qwen3MLP(nn.Module):
    def __init__(
        self,
        config:Qwen3Config ):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # silu_x = torch.sigmoid(gated_up) * gated_up
        # glu_part = silu_x * up_proj
        # 下面的等价于数学上面的这个
        x = F.silu(gate) * up
        x = self.down_proj(x)
        
        return x
        
class Qwen3DecoderLayer(nn.Module):
    def __init__(self,
                 config: Qwen3Config
                 ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps) # 放一个layer 带weight的
        self.mlp = Qwen3MLP(config=config) 
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps) # 放一个layer 带weight的
        self.self_attn = Qwen3Attention(config=config)
        
    def forward(self,
                hidden_states:torch.Tensor,
                token_position:torch.Tensor
                    ):
        # input layernorm 
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(token_position, hidden_states)
        hidden_states = residual + hidden_states
        
        # 
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

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
    def forward(
        self,
        input_ids: torch.Tensor,
        token_position: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)          
             # [B, S,hidden_size]
        for layer in self.layers:
            hidden_states = layer( hidden_states,token_position)
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
class Qwen3ForCausalLM(nn.Module):
    def __init__(self,
                config:Qwen3Config
                ):
        super().__init__()
        
        self.model = Qwen3Model(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self,input_ids,token_position):
        hidden_states = self.model(input_ids, token_position)
        logits = self.lm_head(hidden_states)
        return logits