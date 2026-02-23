from __future__ import annotations

from typing import Any

class LLM:
    """最小对外入口：只保证可以被 import，不包含任何模型实现细节。"""

    def __init__(self, model: str, **kwargs: Any) -> None:
        # 这里只存参数，不加载模型、不初始化 tokenizer/scheduler/runner
        self.model = model
        self.kwargs = kwargs

    def generate(self, prompts: Any, sampling_params: Any) -> Any:
        # 后续 ticket 再实现
        raise NotImplementedError