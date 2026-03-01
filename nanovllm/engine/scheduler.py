from nanovllm.engine.request import Request


class Scheduler:
    def __init__(self):
        # 三个队列，分别对应请求的三个生命周期阶段
        self.waiting: list[Request] = []   # 还没 prefill 的请求
        self.running: list[Request] = []   # 已 prefill，在 decode 中
        self.finished: list[Request] = []  # 已完成

    def add_request(self, request: Request) -> None:
        """外部提交新请求，进 waiting 队列。"""
        self.waiting.append(request)

    def schedule(self) -> tuple[list[Request], list[Request]]:
        """
        决定本轮执行谁。
        返回 (to_prefill, to_decode)：
          - to_prefill：从 waiting 取出、本轮要 prefill 的请求（现在每轮只取 1 个）
          - to_decode：running 里除了刚 prefill 的，所有请求各做一步 decode

        为什么刚 prefill 的不立刻 decode？
        因为 prefill 已经产出了第一个 token，decode 从第二个 token 开始。
        """
        to_prefill = []
        if self.waiting:
            req = self.waiting.pop(0)       # FCFS：先来先服务
            self.running.append(req)
            to_prefill.append(req)

        # running 里除了刚加入的，都做 decode
        to_decode = [r for r in self.running if r not in to_prefill]
        return to_prefill, to_decode

    def on_token(self, request: Request, token_id: int, eos_token_id: int) -> None:
        """
        Runner 每生成一个 token 就调用一次。
        Scheduler 负责判断是否结束，并做状态流转：running -> finished。
        
        =====
        为什么放这里的原因是因为每次产生的token 只是一个id 在runner层面
        但是决定是否要继续这个request是scheduler的事情
        
        如何判断是否要停止的条件
            如果这个tokenid是 eos 或者output token ids输出大于最大输出数值
        作为scheduler保留地running里面就应该清除这个request
        就应该
            1. 在scheduling的running里面里面移除这个request 同时在finished里加入
            2. 同时把这个request的kv cache全部释放
        """
        request.append_token(token_id)

        # 两种结束条件：遇到 eos，或者达到 max_new_tokens
        if token_id == eos_token_id or len(request.output_token_ids) >= request.max_new_tokens:
            request.finished = True
            self.running.remove(request)
            self.finished.append(request)
            # 释放这个请求占用的 KV cache 内存
            request.kv_caches = [None] * len(request.kv_caches)

    def is_idle(self) -> bool:
        """waiting 和 running 都空了，说明所有请求处理完毕。"""
        return not self.waiting and not self.running
