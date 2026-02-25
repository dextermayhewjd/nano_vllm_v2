import os, json, sys
from transformers import Qwen3Config
from pathlib import Path
from nanovllm.model.qwen3 import Qwen3ForCausalLM

def main():
    
    model_dir = Path("/home/fredkeira/Data/models/Qwen/Qwen3-8B")  # 改成你的路径
    
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    wm = json.load(open(index_path, "r", encoding="utf-8"))["weight_map"]
    ckpt_keys = set(wm.keys())

    # 1) 构建你的空壳模型（只要能init就行）

    config = Qwen3Config.from_pretrained(model_dir)
    model = Qwen3ForCausalLM(config)

    model_keys = set(dict(model.named_parameters()).keys())

    missing = sorted(list(ckpt_keys - model_keys))
    extra = sorted(list(model_keys - ckpt_keys))

    print("ckpt keys:", len(ckpt_keys))
    print("model keys:", len(model_keys))
    print("\nMISSING (ckpt有但model没有):", len(missing))
    print("\n".join(missing[:50]))

    print("\nEXTRA (model有但ckpt没有):", len(extra))
    print("\n".join(extra[:50]))

    # 2) 重点检查几个关心的
    must = [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    print("\n=== must-have keys exist? ===")
    for k in must:
        print(k, k in model_keys, "-> shard:", wm.get(k))

if __name__ == "__main__":
    main()