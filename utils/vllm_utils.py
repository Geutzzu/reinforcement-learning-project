import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Union
# from mlx_lm import load as mlx_lm_load
# from mlx_lm import generate as mlx_lm_generate

if torch.cuda.is_available():
    from vllm import LLM as _VLLM, SamplingParams

class VLLM:
    """VLLM backend for CUDA GPUs - fastest for NVIDIA hardware"""
    
    def __init__(self, model: str, dtype: str = "auto", max_model_len: Optional[int] = None, gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, **kwargs):
        self.model_name = model
        self.llm = _VLLM(model=model, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()
    

    def predict_batch(self, prompts: List[str], max_output_tokens: int = 5000, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 20, stop: Optional[List[str]] = None, **kwargs) -> List[str]:
        sampling_params = SamplingParams(max_tokens=max_output_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop, **kwargs)
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


class VLLMVLM:
    """VLLM backend for Vision-Language Models on CUDA GPUs (Qwen3-VL, etc.)
    
    Supports both text-only and image+text inference using vLLM.
    Uses OpenAI-style chat messages with image content.
    
    Example models:
        - Qwen/Qwen3-VL-8B-Instruct
        - Qwen/Qwen2.5-VL-7B-Instruct
    """
    
    def __init__(
        self, 
        model: str, 
        dtype: str = "auto", 
        max_model_len: Optional[int] = None, 
        gpu_memory_utilization: float = 0.9, 
        tensor_parallel_size: int = 1, 
        limit_mm_per_prompt: Optional[dict] = None,
        **kwargs
    ):
        from transformers import AutoProcessor
        
        self.model_name = model
        
        # Default limit for multimodal inputs per prompt
        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 4}  # Allow up to 4 images per prompt
        
        self.llm = _VLLM(
            model=model, 
            dtype=dtype, 
            max_model_len=max_model_len, 
            gpu_memory_utilization=gpu_memory_utilization, 
            tensor_parallel_size=tensor_parallel_size, 
            trust_remote_code=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            **kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    
    def _load_image(self, image_path: str):
        """Load image from path or URL and return PIL Image"""
        from PIL import Image
        import requests
        from io import BytesIO
        
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(image_path).convert("RGB")
    
    def _build_messages(self, prompt: str, image_path: Optional[str] = None) -> List[dict]:
        """Build OpenAI-style chat messages for VLM inference"""
        content = []
        
        # Add image if provided
        if image_path is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_path if image_path.startswith(("http://", "https://")) else f"file://{image_path}"}
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        return [{"role": "user", "content": content}]
    
    def predict_batch(
        self, 
        prompts: List[str], 
        max_output_tokens: int = 512, 
        temperature: float = 0.0, 
        top_p: float = 0.95, 
        top_k: int = 20, 
        stop: Optional[List[str]] = None, 
        batch_size: int = 8,
        images: Optional[List[Optional[str]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of text prompts
            images: Optional list of image paths/URLs. If provided, must match length of prompts.
                   Use None for text-only prompts: [None, "path/to/img.jpg", None]
            ... other standard args
        """
        # If no images provided, create list of None
        if images is None:
            images = [None] * len(prompts)
        
        assert len(images) == len(prompts), "images list must match prompts length"
        
        sampling_params = SamplingParams(
            max_tokens=max_output_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            stop=stop
        )
        
        # Build chat inputs for vLLM
        all_inputs = []
        for prompt, image in zip(prompts, images):
            messages = self._build_messages(prompt, image)
            
            # Apply chat template
            formatted_prompt = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Build vLLM input with multi_modal_data if image present
            if image is not None:
                pil_image = self._load_image(image)
                all_inputs.append({
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": pil_image}
                })
            else:
                all_inputs.append({"prompt": formatted_prompt})
        
        # Run batch generation
        outputs = self.llm.generate(all_inputs, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
    
    def predict_single(
        self, 
        prompt: str, 
        image: Optional[str] = None,
        max_output_tokens: int = 512, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Convenience method for single prompt inference"""
        return self.predict_batch(
            prompts=[prompt],
            images=[image],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            **kwargs
        )[0]


class MLXLLM:
    """MLX backend for Apple Silicon - fastest for M1/M2/M3 Macs (text-only LLMs)"""
    
    def __init__(self, model: str, dtype: str = "auto", max_model_len: Optional[int] = None, gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, **kwargs):
        self.model_name = model
        self.model, self.tokenizer = mlx_lm_load(model)
        self.max_model_len = max_model_len
    
    def predict_batch(self, prompts: List[str], max_output_tokens: int = 512, temperature: float = 0.0, top_p: float = 0.95, top_k: int = 20, stop: Optional[List[str]] = None, batch_size: int = 8, **kwargs) -> List[str]:
        from mlx_lm.sample_utils import make_sampler
        
        sampler = make_sampler(
            temp=temperature if temperature > 0 else 0.0,
            top_p=top_p if top_p < 1.0 else 0.0,
            top_k=top_k if top_k > 0 else 0
        )
        
        final_outputs = []

        for prompt in prompts:
            response = mlx_lm_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_output_tokens,
                sampler=sampler,
                verbose=False
            )
            
            text = response
            if stop:
                for s in stop:
                    if s in text:
                        text = text.split(s)[0]
            
            final_outputs.append(text)
        
        return final_outputs


class MLXVLM:
    """MLX backend for Vision-Language Models on Apple Silicon (Qwen3-VL, etc.)
    
    Supports both text-only and image+text inference.
    Uses mlx-vlm package.
    
    Example models:
        - mlx-community/Qwen3-VL-8B-Instruct-4bit
        - mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    """
    
    def __init__(self, model: str, dtype: str = "auto", max_model_len: Optional[int] = None, gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, **kwargs):
        from mlx_vlm import load as mlx_vlm_load
        
        self.model_name = model
        self.model, self.processor = mlx_vlm_load(model)
        self.max_model_len = max_model_len
    
    def predict_batch(
        self, 
        prompts: List[str], 
        max_output_tokens: int = 512, 
        temperature: float = 0.0, 
        top_p: float = 0.95, 
        top_k: int = 20, 
        stop: Optional[List[str]] = None, 
        batch_size: int = 8,
        images: Optional[List[Optional[str]]] = None,  # List of image paths (None for text-only)
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of text prompts
            images: Optional list of image paths. If provided, must match length of prompts.
                   Use None for text-only prompts: [None, "path/to/img.jpg", None]
            ... other standard args
        """
        from mlx_vlm import generate as mlx_vlm_generate
        
        # If no images provided, create list of None
        if images is None:
            images = [None] * len(prompts)
        
        assert len(images) == len(prompts), "images list must match prompts length"
        
        final_outputs = []

        for prompt, image in zip(prompts, images):
            result = mlx_vlm_generate(
                self.model,
                self.processor,
                prompt=prompt,
                image=image,  # None for text-only, path/URL for image
                max_tokens=max_output_tokens,
                temp=temperature if temperature > 0 else 0.0,
                top_p=top_p,
                verbose=False
            )
            
            # Extract text from GenerationResult
            text = result.text if hasattr(result, 'text') else str(result)
            
            if stop:
                for s in stop:
                    if s in text:
                        text = text.split(s)[0]
            
            final_outputs.append(text)
        
        return final_outputs
    
    def predict_single(
        self, 
        prompt: str, 
        image: Optional[str] = None,
        max_output_tokens: int = 512, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Convenience method for single prompt inference"""
        return self.predict_batch(
            prompts=[prompt],
            images=[image],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            **kwargs
        )[0]



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


def get_best_backend():
    """Auto-detect the best backend for current hardware"""
    if torch.cuda.is_available():
        return "vllm"
    elif torch.backends.mps.is_available():
        try:
            import mlx_lm
            return "mlx"
        except ImportError:
            return "transformers"
    else:
        return "transformers"


def AutoLLM(model: str, **kwargs):
    """
    Auto-select the best LLM backend based on available hardware.
    
    - CUDA available → VLLM (fastest for NVIDIA)
    - Apple Silicon + mlx-lm installed → MLXLLM (fastest for Mac)
    - Otherwise → LLM (transformers fallback)
    """
    backend = get_best_backend()
    
    if backend == "vllm":
        print(f"Using VLLM backend (CUDA)")
        return VLLM(model, **kwargs)
    elif backend == "mlx":
        print(f"Using MLX backend (Apple Silicon)")
        return MLXLLM(model, **kwargs)
    else:
        print(f"Using Transformers backend (fallback)")
        return LLM(model, **kwargs)


def AutoVLM(model: str, **kwargs):
    """
    Auto-select the best VLM (Vision-Language Model) backend based on available hardware.
    
    - CUDA available → VLLMVLM (fastest for NVIDIA)
    - Apple Silicon + mlx-vlm installed → MLXVLM (fastest for Mac)
    - Otherwise → raises NotImplementedError (no fallback for VLM)
    
    Example models:
        - CUDA: "Qwen/Qwen3-VL-8B-Instruct"
        - MLX: "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    """
    backend = get_best_backend()
    
    if backend == "vllm":
        print(f"Using VLLMVLM backend (CUDA)")
        return VLLMVLM(model, **kwargs)
    elif backend == "mlx":
        try:
            import mlx_vlm
            print(f"Using MLXVLM backend (Apple Silicon)")
            return MLXVLM(model, **kwargs)
        except ImportError:
            raise ImportError("mlx-vlm is required for VLM inference on Apple Silicon. Install with: pip install mlx-vlm")
    else:
        raise NotImplementedError("VLM inference requires either CUDA (vLLM) or Apple Silicon (mlx-vlm). No fallback available.")

        

if __name__ == "__main__":
    import pandas as pd
    
    # Auto-select best backend
    print(f"Best backend: {get_best_backend()}")
    
    # Example usage
    # model = AutoLLM("mlx-community/Qwen2.5-7B-Instruct-4bit")  # For MLX
    # model = AutoLLM("Qwen/Qwen2.5-7B-Instruct")  # For VLLM/Transformers
    
    # df = pd.read_parquet("/Users/geo/facultate/rl/rl/data/maze.parquet")
    # df = df.sample(5, random_state=42)
    # df['pred'] = model.predict_batch(df['prompt'].tolist())
    # print(df['pred'].iloc[0])
    # print(df['answer'].iloc[0])

