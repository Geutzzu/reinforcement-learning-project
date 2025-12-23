from dataclasses import dataclass

@dataclass
class BaseConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    num_epochs: int = 3
    max_steps: int = -1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    max_length: int = 512
    use_liger_kernel: bool = False
    
    output_dir: str = "/Users/geo/facultate/rl/rl/results"
    logging_steps: int = 10
    save_steps: int = 100
    
    train_dataset_path: str = None
    eval_dataset_path: str = None
    eval_split_ratio: float = 0.0
    
    experiment_name: str = None  # If None, uses "experiment"
    reward_fn: str = "maze"


@dataclass
class SFTConfig(BaseConfig):
    learning_rate: float = 2e-5


@dataclass
class GRPOConfig(BaseConfig):
    learning_rate: float = 1e-5
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    beta: float = 0.1
    
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1
    vllm_enable_sleep_mode: bool = False
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 240.0
