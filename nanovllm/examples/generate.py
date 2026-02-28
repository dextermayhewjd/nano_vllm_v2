
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config

from nanovllm.model.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline generation with nano Qwen3 model (no KV cache)."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/fredkeira/Data/models/Qwen/Qwen3-8B"),
        help="Path to local HF model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请简单介绍一下你自己。",
        help="Input prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda:0","cuda:1"],
        help="Inference device.",
    )
    return parser.parse_args()


def greedy_generate_no_cache(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    generated = input_ids # [1,S]
    for _ in range(max_new_tokens):
        seq_len = generated.shape[1]
        # 注意此处的token_positions就是[0,1,2,3....,seq_len-1] shape 是[s,]->[1,S]
        token_positions = torch.arange(
            seq_len, device=generated.device, dtype=torch.long
        ).unsqueeze(0)
        with torch.no_grad():
            logits = model(generated, token_positions)
        #这里的的B S V 中 取的是 最后一个token 的logits 
        # [1,seq_len,vocab_size] -> [1,vocab_size] -> [1,1]
        # 选择最大的vocab的index 作为新的token的id
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        #cat要求的是两个tensor只有在dim上维度不同 
        # 此处的generated 和 next_token都是只有dim1上不同
        generated = torch.cat([generated, next_token], dim=1)
        
        #（但仅适用于 B=1）。
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
    return generated


def greedy_generate_with_cache(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    """
    带 KV Cache 的 greedy decoding。
    要求调用前已执行 model.setup_cache(...)。

    两个阶段：
      prefill  — 把整个 prompt 一次性 forward，cache 写入所有 prompt 的 K/V
      decode   — 每步只传 1 个新 token，position = 当前 cache 已填充长度
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]            # prompt 有多少个 token

    # ---- prefill: 一次性处理整个 prompt ----
    # token_positions = [0, 1, ..., prompt_len-1]，和 no_cache 的第一步完全相同
    token_positions = torch.arange(prompt_len, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids, token_positions)  # [1, prompt_len, vocab_size]

    # 取 prompt 最后一个 token 的 logits，得到第一个生成 token
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
    generated = torch.cat([input_ids, next_token], dim=1)

    if eos_token_id is not None and int(next_token.item()) == eos_token_id:
        return generated

    # ---- decode: 逐 token 生成 ----
    for _ in range(max_new_tokens - 1):
        # 关键区别：只传新 token，不传整个序列
        # position = 当前序列长度 - 1（因为 generated 已经 cat 了 next_token）
        cur_pos = generated.shape[1] - 1
        token_positions = torch.tensor([[cur_pos]], device=device, dtype=torch.long)

        with torch.no_grad():
            logits = model(next_token, token_positions)  # [1, 1, vocab_size]

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    return generated


def main() -> None:
    args = parse_args()

    # 获取tokenizer和qwen的config用于构建qwen3模型
    print(f"Loading tokenizer/config from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    config = Qwen3Config.from_pretrained(args.model_dir)

    # 使用config来构建qwen3 模型
    # 并且使用自建的load weight来加载weight
    print("Building nano model and loading weights...")
    model = Qwen3ForCausalLM(config)
    load_weights(model, args.model_dir)

    dtype = torch.bfloat16 if args.device == "cuda:1" else torch.float32
    model = model.to(device=args.device, dtype=dtype)
    model.eval()

    #注意此处tokenizer即使拿到的是str也会返回的encoded字典里的[input_ids]也是[B,S]
    encoded = tokenizer(args.prompt, return_tensors="pt")
    #也就是这里暂时是[1,S]
    input_ids = encoded["input_ids"].to(args.device)

    # 注意此处是单条prompt
    print(f"Prompt token length: {input_ids.shape[1]}")

    # ======== 原始 no_cache 版本（已验证正确）========
    # output_ids = greedy_generate_no_cache(
    #     model=model,
    #     input_ids=input_ids,
    #     max_new_tokens=args.max_new_tokens,
    #     eos_token_id=tokenizer.eos_token_id,
    # )
    #
    # #注意这里只有一个逗号 是取原本input_ids.shape[1] 也就是s之后的所有token
    # completion_ids = output_ids[:, input_ids.shape[1] :]
    # # 这里decode 两个东西 一个是全文对应full text
    # # 另一个对应completion_text 用的completion_text
    # full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
    #
    # print("\n=== Generation Result ===")
    # print(f"Prompt: {args.prompt}")
    # print(f"Completion: {completion_text}")
    # print(f"Full text: {full_text}")

    # ======== KV Cache 版本 ========
    prompt_len = input_ids.shape[1]
    max_total_len = prompt_len + args.max_new_tokens
    # setup_cache 在 forward 之前调用，给每一层的 Qwen3Attention 分配 cache 内存
    model.setup_cache(max_seq_len=max_total_len, dtype=dtype, device=args.device)

    output_ids = greedy_generate_with_cache(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    #注意这里只有一个逗号 是取原本input_ids.shape[1] 也就是s之后的所有token
    completion_ids = output_ids[:, prompt_len:]
    # 这里decode 两个东西 一个是全文对应full text
    # 另一个对应completion_text 用的completion_text
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)

    print("\n=== Generation Result (with KV Cache) ===")
    print(f"Prompt: {args.prompt}")
    print(f"Completion: {completion_text}")
    print(f"Full text: {full_text}")

    # 用完之后 reset，下一个请求前调用
    model.reset_cache()
'''
原始 no_cache 版本的输出（已验证正确）：
Prompt token length: 6

=== Generation Result ===
Prompt: 你好，请简单介绍一下你自己。
Completion:  你好！我是一个大型语言模型，名叫通义千问，由通义实验室研发。我能够进行多轮对话，回答各种问题，


=== Generation Result (with KV Cache) ===
Prompt: 你好，请简单介绍一下你自己。
Completion:  你好！我是一个大型语言模型，名叫通义千问，由通义实验室研发。我能够进行多轮对话，回答各种问题，
Full text: 你好，请简单介绍一下你自己。 你好！我是一个大型语言模型，名叫通义千问，由通义实验室研发。我能够进行多轮对话，回答各种问题，
'''


if __name__ == "__main__":
    main()

