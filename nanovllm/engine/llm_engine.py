import torch
from pathlib import Path

from nanovllm.engine.request import Request
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    把 ModelRunner（执行） 和 Scheduler（调度） 组合起来。

    使用方式：
        engine = LLMEngine(model_dir, device, dtype)
        rid1 = engine.add_request("你好", max_new_tokens=32)
        rid2 = engine.add_request("介绍一下KV cache", max_new_tokens=64)
        engine.run_until_done()
        print(engine.get_output(rid1))
        print(engine.get_output(rid2))
    """

    def __init__(self, model_dir: str | Path, device: str, dtype: torch.dtype):
        self.runner = ModelRunner(model_dir, device, dtype)
        self.scheduler = Scheduler()
        self._request_counter = 0

    def add_request(self, prompt: str, max_new_tokens: int) -> int:
        """
        提交一个新请求。
        分配好 KV cache 后放入 scheduler 的 waiting 队列。
        返回 request_id，后续用来取结果。
        
        ========
        可以从modelrunner的generate得到启发 
        这里每次加入一个request 都会把前置的tokenizer encode prompt解决
        创造request 
            并且使用runner把request的kvcache创建好
            (通过计算propmt和maxtoken的和来)
        
        同时把这个request加入scheduler
        被加入的request 现在waiting区里面
        """
        prompt_ids = self.runner.tokenizer(prompt)["input_ids"]
        request = Request(
            request_id=self._request_counter,
            prompt_token_ids=prompt_ids,
            num_layers=self.runner.num_layers,
            max_new_tokens=max_new_tokens,
        )
        # cache 在 add 时就分配好，避免运行时动态分配
        max_total = len(prompt_ids) + max_new_tokens
        self.runner.allocate_request_cache(request, max_total)

        self.scheduler.add_request(request)
        self._request_counter += 1
        return request.request_id

    def step(self) -> None:
        """
        执行一轮调度：
          1. prefill waiting 中的下一个请求（如果有）
          2. 对所有 running 中的请求各做一步 decode
        """
        to_prefill, to_decode = self.scheduler.schedule()

        # prefill 阶段：整个 prompt 一次 forward
        for request in to_prefill:
            self.runner.bind_request(request)
            token = self.runner.prefill(request)
            self.runner.unbind_request()
            self.scheduler.on_token(request, token, self.runner.tokenizer.eos_token_id)

        # decode 阶段：每个 running request 各走一步
        for request in to_decode:
            self.runner.bind_request(request)
            token = self.runner.decode_step(request)
            self.runner.unbind_request()
            self.scheduler.on_token(request, token, self.runner.tokenizer.eos_token_id)

    def run_until_done(self) -> None:
        """循环调用 step，直到所有请求完成。"""
        while not self.scheduler.is_idle():
            self.step()

    def get_output(self, request_id: int) -> str | None:
        """从 finished 队列里找到对应请求，解码并返回生成文本。"""
        for req in self.scheduler.finished:
            if req.request_id == request_id:
                return self.runner.tokenizer.decode(
                    req.output_token_ids, skip_special_tokens=True
                )
        return None
