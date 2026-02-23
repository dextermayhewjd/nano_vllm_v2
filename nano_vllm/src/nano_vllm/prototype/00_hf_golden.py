"""Phase-A entrypoint: generate HF golden artifacts for regression baselining.

This file is intentionally thin after refactor:
- parse CLI arguments
- call the reusable runner
- print deterministic, machine-readable paths
"""

from __future__ import annotations

import argparse

from nano_vllm.prototype.hf_golden_runner import run_hf_golden


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for A1/A2 golden generation workflow."""

    parser = argparse.ArgumentParser(description="Generate HF golden artifacts (.json + .pt).")
    parser.add_argument("--model", required=True, help="HF repo id or local model directory")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Greedy decode length")
    parser.add_argument("--topk", type=int, default=50, help="Store top-k of prefill last-token logits")
    parser.add_argument("--seed", type=int, default=1234, help="Global seed for deterministic replay")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Model dtype")
    parser.add_argument("--trust_remote_code", action="store_true", help="Forwarded to HF loaders")

    # A2 enhancement switch: dump a few intermediate activations.
    parser.add_argument(
        "--dump_breakpoints",
        action="store_true",
        help="Dump layer0/final_norm outputs when model architecture exposes them.",
    )

    # A1 acceptance switch: run same config twice and assert identical tensors.
    parser.add_argument(
        "--verify_determinism",
        action="store_true",
        help="Run the same config twice and fail fast if outputs differ.",
    )
    return parser


def main() -> None:
    """CLI entrypoint used by local experiments and CI smoke checks."""

    args = build_parser().parse_args()
    json_path, pt_path = run_hf_golden(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        topk=args.topk,
        seed=args.seed,
        device_name=args.device,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        dump_breakpoints=args.dump_breakpoints,
        verify_determinism=args.verify_determinism,
    )
    print(f"[saved-json] {json_path}")
    print(f"[saved-pt] {pt_path}")


if __name__ == "__main__":
    main()
