from vllm import LLM, SamplingParams

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Union


class VLLM:
    def __init__(self, model: str, dtype: str = "auto", max_model_len: Optional[int] = None, gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, **kwargs):
        self.model_name = model
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()
    

    def predict_batch(self, prompts: List[str], max_output_tokens: int = 512, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 20, stop: Optional[List[str]] = None, **kwargs) -> List[str]:
        sampling_params = SamplingParams(max_tokens=max_output_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop, **kwargs)
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


class LLM:
    def __init__(self, model: str, dtype: str = "auto", max_model_len: Optional[int] = None, gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, **kwargs):
        self.model_name = model
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        torch_dtype = "auto"
        if dtype != "auto":
            if dtype == "float16": torch_dtype = torch.float16
            elif dtype == "bfloat16": torch_dtype = torch.bfloat16
            elif dtype == "float32": torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype, device_map=self.device, trust_remote_code=True, **kwargs)

    def predict_batch(self, prompts: List[str], max_output_tokens: int = 512, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 20, stop: Optional[List[str]] = None, batch_size: int = 8, **kwargs) -> List[str]:
        gen_kwargs = {"max_new_tokens": max_output_tokens, "top_p": top_p, "top_k": top_k, "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id}
        
        if temperature <= 1e-5:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            
        gen_kwargs.update(kwargs)

        final_outputs = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
                
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]
            decoded_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for text in decoded_outputs:
                if stop:
                    for s in stop:
                        if s in text:
                            text = text.split(s)[0]
                final_outputs.append(text)
            
        return final_outputs

        

if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/maze.parquet")
    model = VLLM("Qwen/Qwen2.5-0.5B-Instruct")
    df = df.sample(5, random_state=42)
    df['pred'] = model.predict_batch(df['prompt'].tolist())
    print(df['pred'].iloc[0])
    print(df['answer'].iloc[0])
