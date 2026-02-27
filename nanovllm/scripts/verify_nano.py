"""
验证 nano 实现与 HF 官方实现的中间结果是否一致。

前置条件：
    先跑 dump_hf_reference.py 生成 qwen3_reference.pt

用法:
    python -m nanovllm.scripts.verify_nano
"""

import torch
from pathlib import Path
from transformers import Qwen3Config

from nanovllm.model.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_weights

MODEL_DIR = Path("/home/fredkeira/Data/models/Qwen/Qwen3-8B")
REF_PATH = Path("/home/fredkeira/Data/reference/qwen3_reference.pt")


def main():
    # ---- 1) 加载 reference ----
    print(f"Loading reference from {REF_PATH} ...")
    ref = torch.load(REF_PATH, map_location="cpu", weights_only=True)
    input_ids = ref["input_ids"]                # [1, S]
    token_position = ref["token_position"]      # [1, S]

    # ---- 2) 构建 nano 模型并加载权重 ----
    print(f"Building nano model and loading weights ...")
    config = Qwen3Config.from_pretrained(MODEL_DIR)
    model = Qwen3ForCausalLM(config)
    load_weights(model, MODEL_DIR)
    model = model.to(torch.bfloat16)
    model.eval()

    # ---- 3) 注册 hooks，截获中间结果 ----
    nano_outputs = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            nano_outputs[name] = out.detach().clone().float()
        return hook_fn

    nano_model = model.model  # Qwen3ForCausalLM.model -> Qwen3Model

    # embed_tokens
    hooks.append(nano_model.embed_tokens.register_forward_hook(make_hook("embed")))

    # 每一层 decoder layer
    for i, layer in enumerate(nano_model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"layer.{i}.input_ln")))
        hooks.append(layer.self_attn.register_forward_hook(
            make_hook(f"layer.{i}.attn")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"layer.{i}.post_ln")))
        hooks.append(layer.mlp.register_forward_hook(
            make_hook(f"layer.{i}.mlp")))
        hooks.append(layer.register_forward_hook(
            make_hook(f"layer.{i}.output")))

    # final norm
    hooks.append(nano_model.norm.register_forward_hook(make_hook("final_norm")))

    # lm_head
    hooks.append(model.lm_head.register_forward_hook(make_hook("logits")))

    # ---- 4) 前向传播 ----
    print("Running nano forward pass ...")
    with torch.no_grad():
        logits = model(input_ids, token_position)

    # ---- 5) 清理 hooks ----
    for h in hooks:
        h.remove()

    # ---- 6) 逐个对比 ----
    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)

    pass_count = 0
    fail_count = 0

    for name in sorted(ref.keys()):
        # 跳过输入信息（不是计算结果）
        if name in ("input_ids", "token_position"):
            continue

        if name not in nano_outputs:
            print(f"  [SKIP] {name}: not captured in nano model")
            continue

        ref_tensor = ref[name]
        nano_tensor = nano_outputs[name]

        if ref_tensor.shape != nano_tensor.shape:
            print(f"  [FAIL] {name}: shape mismatch ref={ref_tensor.shape} nano={nano_tensor.shape}")
            fail_count += 1
            continue

        max_diff = (ref_tensor - nano_tensor).abs().max().item()
        if torch.allclose(ref_tensor, nano_tensor, atol=1e-4, rtol=1e-4):
            print(f"  [PASS] {name}  (max_diff={max_diff:.2e})")
            pass_count += 1
        else:
            print(f"  [FAIL] {name}  (max_diff={max_diff:.2e})")
            fail_count += 1

    print("=" * 60)
    print(f"PASS: {pass_count}  FAIL: {fail_count}")


if __name__ == "__main__":
    main()
