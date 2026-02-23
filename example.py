
from nanovllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(model="dummy-model-path")
    params = SamplingParams()
    print("LLM:", llm)
    print("SamplingParams:", params)


if __name__ == "__main__":
    main()
