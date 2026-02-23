"""Core implementation for Phase-A HF golden artifact generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hf_golden_types import RunMeta
from .hf_golden_utils import now_str, parse_dtype, set_seed, slug


class BreakpointCollector:
    """Collect a few high-value intermediate activations via forward hooks.

    The goal is to keep payload small but still useful for quickly locating where
    a future replacement implementation diverges from HF reference behavior.
    """

    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.breakpoint_tensors: dict[str, torch.Tensor] = {}

    def _register_hook(self, module: torch.nn.Module, name: str) -> None:
        def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], outputs: Any) -> None:
            # A lot of transformer blocks return tuples. We always take the first
            # tensor-like output as the representative checkpoint for debugging.
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            if isinstance(out, torch.Tensor):
                self.breakpoint_tensors[name] = out.detach().float().cpu()

        self._handles.append(module.register_forward_hook(_hook))

    def attach(self) -> None:
        """Attach hooks on model-specific modules when available.

        We support common HF structures:
        - LLaMA/Mistral style: model.layers + model.norm
        - GPT-2 style: transformer.h + transformer.ln_f
        """

        llama_like = getattr(self._model, "model", None)
        if llama_like is not None:
            layers = getattr(llama_like, "layers", None)
            if layers is not None and len(layers) > 0:
                self._register_hook(layers[0], "layer0_output")
            final_norm = getattr(llama_like, "norm", None)
            if final_norm is not None:
                self._register_hook(final_norm, "final_norm_output")

        gpt2_like = getattr(self._model, "transformer", None)
        if gpt2_like is not None:
            blocks = getattr(gpt2_like, "h", None)
            if blocks is not None and len(blocks) > 0:
                self._register_hook(blocks[0], "layer0_output")
            final_norm = getattr(gpt2_like, "ln_f", None)
            if final_norm is not None:
                self._register_hook(final_norm, "final_norm_output")

    def detach(self) -> None:
        """Always remove hooks to prevent side effects in repeated runs."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _project_root() -> Path:
    # file: nano_vllm/src/nano_vllm/prototype/hf_golden_runner.py
    # parent[3] -> nano_vllm project root
    return Path(__file__).resolve().parents[3]


def _build_artifact_stem(model_name: str, seed: int, dtype_name: str, device_name: str) -> str:
    return f"hf_run_{now_str()}__{slug(model_name)}__seed{seed}__{dtype_name}_{device_name}"


def _assert_repeatable(payload_1: dict[str, torch.Tensor], payload_2: dict[str, torch.Tensor]) -> None:
    """Validate A1 acceptance criteria: same prompt/seed/params => identical outputs."""

    for key in ("input_ids", "gen_ids", "logits_last_topk_indices", "logits_last_topk_values"):
        if not torch.equal(payload_1[key], payload_2[key]):
            raise RuntimeError(f"Determinism check failed for field: {key}")


@torch.inference_mode()
def run_hf_golden(
    *,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    topk: int,
    seed: int,
    device_name: str,
    dtype_name: str,
    trust_remote_code: bool,
    dump_breakpoints: bool,
    verify_determinism: bool,
) -> tuple[Path, Path]:
    """Execute one (or two) golden runs and write JSON/PT artifacts.

    Returns:
        (json_path, pt_path)
    """

    def _single_pass() -> tuple[dict[str, Any], RunMeta]:
        set_seed(seed)

        device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")
        dtype = parse_dtype(dtype_name)

        # Tokenizer/model loading is intentionally explicit and centralized so that
        # any future replacement path can mirror exactly the same setup.
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype if device.type == "cuda" else torch.float32,
            device_map=None,
        )
        model.eval()
        model.to(device)

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        breakpoint_collector = BreakpointCollector(model)
        if dump_breakpoints:
            breakpoint_collector.attach()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        logits_last = logits[0, -1, :]

        clipped_topk = min(topk, logits_last.numel())
        topk_values, topk_indices = torch.topk(logits_last, k=clipped_topk, dim=-1)

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        if dump_breakpoints:
            breakpoint_collector.detach()

        payload: dict[str, Any] = {
            "input_ids": input_ids.detach().to(torch.int64).cpu(),
            "gen_ids": gen_ids.detach().to(torch.int64).cpu(),
            "logits_last_topk_indices": topk_indices.detach().to(torch.int64).cpu(),
            "logits_last_topk_values": topk_values.detach().float().cpu(),
        }
        if dump_breakpoints:
            payload["breakpoints"] = breakpoint_collector.breakpoint_tensors

        metadata = RunMeta(
            model=model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            topk=clipped_topk,
            seed=seed,
            device=str(device),
            dtype=dtype_name,
            torch_version=torch.__version__,
            transformers_version=__import__("transformers").__version__,
            created_at=now_str(),
            dump_breakpoints=dump_breakpoints,
        )
        return payload, metadata

    payload_1, metadata = _single_pass()

    if verify_determinism:
        payload_2, _ = _single_pass()
        _assert_repeatable(payload_1, payload_2)

    artifact_dir = _project_root() / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stem = _build_artifact_stem(model_name, seed, dtype_name, device_name)
    json_path = artifact_dir / f"{stem}.json"
    pt_path = artifact_dir / f"{stem}.pt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)
    torch.save(payload_1, pt_path)

    return json_path, pt_path
