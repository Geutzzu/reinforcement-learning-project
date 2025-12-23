from vllm import LLM, SamplingParams
from typing import List, Optional


class VLLM:
    def __init__(self, model: str, dtype: str = "auto", max_model_len: Optional[int] = None, gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, **kwargs):
        self.model_name = model
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()
    

    def predict_batch(self, prompts: List[str], max_output_tokens: int = 512, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 20, stop: Optional[List[str]] = None, **kwargs) -> List[str]:
        sampling_params = SamplingParams(max_tokens=max_output_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop, **kwargs)
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
        

if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/maze.parquet")
    model = VLLM("Qwen/Qwen2.5-0.5B-Instruct")
    df = df.sample(5, random_state=42)
    df['pred'] = model.predict_batch(df['prompt'].tolist())
    print(df['pred'].iloc[0])
    print(df['answer'].iloc[0])
