class Request:
      def __init__(self, request_id: int, prompt_token_ids: list[int], num_layers:
  int):
          self.request_id = request_id
          self.prompt_token_ids = list(prompt_token_ids)
          self.output_token_ids: list[int] = []
          self.finished = False
          self.kv_caches = [None] * num_layers   # 每层一个 KVCache，由 runner 分配
        # 把kv cache从模型放在request这里 
        # 通过这样子模型本身只管运算逻辑
        # 而每个request 应够保留自己的kv caches
    
    # property是用来管作为getter或者setter的 这里明显是getter
    # 这个是组合获取全部id的方法 包括prompt的生成的
      @property
      def token_ids(self) -> list[int]:
          return self.prompt_token_ids + self.output_token_ids
    
    # 这个是获取最后一个id的方法同时包括有output的时候和没有的时候
      @property
      def last_token_id(self) -> int:
          return self.token_ids[-1]
    # list的增长方式 append
      def append_token(self, token_id: int) -> None:
          self.output_token_ids.append(token_id)
    
      def __len__(self) -> int:
          return len(self.token_ids)