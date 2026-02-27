"""
用 HuggingFace 官方 Qwen3 模型跑一次 prefill，
把每一层关键节点的中间结果存下来，用于验证 nano 实现。

用法:
    python -m nanovllm.scripts.dump_hf_reference

输出:
    qwen3_reference.pt  — 包含所有中间结果的 dict
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("/home/fredkeira/Data/models/Qwen/Qwen3-8B")
OUTPUT_PATH = Path("/home/fredkeira/Data/reference/qwen3_reference.pt")
PROMPT = "Hello, how are you?"


def main():
    # ---- 1) 加载模型和 tokenizer ----
    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cpu",          # 先在 CPU 上跑，确保可复现
        trust_remote_code=True, # 允许加载模型仓库里自定义的 Python tokenizer/model 代码
    )
    model.eval()

    # ---- 2) 准备输入 ----
    inputs = tokenizer(PROMPT, return_tensors="pt")    
    input_ids = inputs["input_ids"]                         # [1, S]
    S = input_ids.shape[1]
    token_position = torch.arange(S).unsqueeze(0)           # [1, S]

    print(f"Input: {PROMPT!r}")
    print(f"Token IDs: {input_ids.tolist()}")
    print(f"Seq len: {S}")

    # ---- 3) 注册 hooks，截获中间结果 ----
    ref = {}                        # 存所有中间结果
    hooks = []                      # 用于结束后清理

    def make_hook(name):
        """生成一个 hook 函数，把 output 存到 ref[name]"""
        def hook_fn(module, input, output):
            # 有些模块返回 tuple（如 HF Attention 返回 (attn_out, attn_weights)）
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            ref[name] = out.detach().clone().float()   # 统一存 float32 方便对比
        return hook_fn

    hf_model = model.model   # Qwen3ForCausalLM.model -> Qwen3Model

    # embed_tokens
    hooks.append(hf_model.embed_tokens.register_forward_hook(make_hook("embed")))

    # 每一层 decoder layer
    for i, layer in enumerate(hf_model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"layer.{i}.input_ln")))
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(f"layer.{i}.attn")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"layer.{i}.post_ln")))
        hooks.append(layer.mlp.register_forward_hook(
            make_hook(f"layer.{i}.mlp")))
        # decoder layer 整体输出（残差加完之后）
        hooks.append(layer.register_forward_hook(
            make_hook(f"layer.{i}.output")))

    # final norm
    hooks.append(hf_model.norm.register_forward_hook(make_hook("final_norm")))

    # lm_head
    hooks.append(model.lm_head.register_forward_hook(make_hook("logits")))

    # ---- 4) 前向传播 ----
    print("Running forward pass ...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # ---- 5) 清理 hooks ----
    for h in hooks:
        h.remove()

    # ---- 6) 额外存入输入信息 ----
    ref["input_ids"] = input_ids
    ref["token_position"] = token_position

    # ---- 7) 保存 ----
    torch.save(ref, OUTPUT_PATH)
    print(f"\nSaved {len(ref)} checkpoints to {OUTPUT_PATH}")
    print("Keys:")
    for k, v in ref.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
