import torch
from pathlib import Path
from transformers import AutoTokenizer, Qwen3Config

from nanovllm.model.qwen3 import Qwen3ForCausalLM, Qwen3Attention
from nanovllm.layers.kv_cache import KVCache
from nanovllm.utils.loader import load_weights
from nanovllm.engine.request import Request


class ModelRunner:
    def __init__(self, model_dir: str | Path, device: str, dtype: torch.dtype):
        model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype

        config = Qwen3Config.from_pretrained(model_dir)
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = Qwen3ForCausalLM(config)
        load_weights(self.model, model_dir)
        self.model = self.model.to(device=device, dtype=dtype)
        self.model.eval()

    def allocate_request_cache(self, request: Request, max_seq_len: int) -> None:
        """为 request 的每层创建 KVCache，存入 request.kv_caches。"""
        # 这里先每一层按照id 分配一下kv cache 确保数目相同
        for layer_id in range(self.num_layers):
            request.kv_caches[layer_id] = KVCache(
                num_kv_heads=self.num_kv_heads,
                max_seq_len=max_seq_len,
                head_dim=self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )

    def bind_request(self, request: Request) -> None:
        """把 request 的每层 KVCache 挂到对应 attention layer 上。"""
        # zip(A, B) 会返回一个迭代器（lazy），每次 next() 给你一个二元组：
        for layer, kv_cache in zip(self.model.model.layers, request.kv_caches):
            layer.self_attn.kv_cache = kv_cache

    def unbind_request(self) -> None:
        """把所有 attention layer 的 kv_cache 解绑。"""
        # 这里是对的 之前的model是 qwen3causalLM 
        # 访问了model才是真的qwen3 model再访问才是layers
        for layer in self.model.model.layers:
            layer.self_attn.kv_cache = None

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """处理整个 prompt,写满 KV cache, 返回第一个生成的 token id。"""
        prompt_len = len(request.prompt_token_ids)
        # 这里的request的 prompt_token_ids是list
        # 这里把list变成tensor 的时候加一个[]的shape是[1,seq_len]
        input_ids = torch.tensor([request.prompt_token_ids], device=self.device)
        # 这里是从arange的[0,1,2,3...len-1]的shape[len] 变成[1,len] 
        positions = torch.arange(prompt_len, device=self.device).unsqueeze(0)

        logits = self.model(input_ids, positions)  # [1, prompt_len, vocab_size]
        return int(logits[:, -1, :].argmax(dim=-1).item())

    # 其实就是decode逻辑放在这里 
    # 输入是单个token shape:[1,1]
    #          positions：[1,1]
    # 返回是最大概率的token的id
    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """用当前 cache 做一步 decode，返回下一个 token id。"""
        # len(request)的定义是 token_ids的长度（包括prompt和generate的）
        cur_pos = len(request) - 1
        '''
        torch.tensor(request.last_token_id)
        tensor(91)
        shape = [] 这是 0维 tensor 和上面的已经是list的不一样
        '''
        input_ids = torch.tensor([[request.last_token_id]], device=self.device)
        positions = torch.tensor([[cur_pos]], device=self.device)

        logits = self.model(input_ids, positions)  # [1, 1, vocab_size]
        return int(logits[:, -1, :].argmax(dim=-1).item())

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        request = Request(
            request_id=0,
            prompt_token_ids=prompt_ids,
            num_layers=self.num_layers,
        )

        max_total = len(request) + max_new_tokens
        self.allocate_request_cache(request, max_total)
        self.bind_request(request)
        # 把model的kv和request的kv绑在一起
        
        # prefill
        first_token = self.prefill(request)
        request.append_token(first_token)

        # decode loop
        for _ in range(max_new_tokens - 1):
            if request.finished:
                break
            token = self.decode_step(request)
            request.append_token(token)
            if token == self.tokenizer.eos_token_id:
                request.finished = True

        self.unbind_request()

        output_ids = request.output_token_ids
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)